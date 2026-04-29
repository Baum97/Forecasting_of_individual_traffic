"""
Endpunkte:
  GET  /api/health                - Liveness-Check
  GET  /api/models                - Liste gespeicherter .joblib-Modelle
  POST /api/forecast              - CSV hochladen, trainieren, Forecast liefern
  POST /api/forecast/{model_id}   - Vorhersage aus vorhandenem Modell

Starten:
  python -m uvicorn backend:app --reload --port 8000
"""
from __future__ import annotations

import io
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from csv_forecaster import (
    Config, FEATURE_COLS, Forecaster,
    load_csv, make_features,
)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Traffic Forecaster API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Shape-Konvertierung: Forecaster → Frontend-JSON ──────────────────────────

def _predict_full(model: Forecaster, horizons: int) -> List[dict]:
    """Wie Forecaster.predict, ergaenzt aber P10/P90 aus den Estimator-Baeumen."""
    last_idx = len(model.daily) - 1
    last_date = model.daily["date"].iloc[last_idx]
    weekdays = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    out: List[dict] = []

    for offset in range(1, horizons + 1):
        target = last_date + pd.Timedelta(days=offset)
        feat = make_features(model.daily, last_idx, model.cfg.history_days, offset)
        X = np.array([[feat[c] if not pd.isna(feat[c]) else 0.0 for c in FEATURE_COLS]])

        p_used = float(model.clf.predict_proba(X)[0][1])

        def quantiles(reg):
            if reg is None:
                return None, None, None
            preds = np.array([t.predict(X)[0] for t in reg.estimators_])
            return (float(preds.mean()),
                    float(np.percentile(preds, 10)),
                    float(np.percentile(preds, 90)))

        dep, dep10, dep90 = quantiles(model.reg_dep)
        ret, ret10, ret90 = quantiles(model.reg_ret)
        wd = target.weekday()
        hour_prof = model.profile[wd].tolist() if model.profile is not None else [p_used] * 24

        out.append({
            "forecastDay": offset,
            "date": target.strftime("%Y-%m-%d"),
            "weekday": weekdays[wd],
            "pUsed": round(p_used, 3),
            "depEst": round(dep, 1) if dep is not None else None,
            "depP10": round(dep10, 1) if dep10 is not None else None,
            "depP90": round(dep90, 1) if dep90 is not None else None,
            "retEst": round(ret, 1) if ret is not None else None,
            "retP10": round(ret10, 1) if ret10 is not None else None,
            "retP90": round(ret90, 1) if ret90 is not None else None,
            "hourProfile": [round(v, 4) for v in hour_prof],
        })
    return out


def _rolling_usage(daily: pd.DataFrame, window: int = 90) -> List[dict]:
    """Letzte `window` Tage als rolling 7/14-Kurve."""
    tail = daily.tail(window).copy()
    tail["roll7"] = tail["is_used"].rolling(7,  min_periods=1).mean()
    tail["roll14"] = tail["is_used"].rolling(14, min_periods=1).mean()
    return [
        {"date": r["date"].strftime("%Y-%m-%d"),
         "roll7":  round(float(r["roll7"]),  4),
         "roll14": round(float(r["roll14"]), 4)}
        for _, r in tail.iterrows()
    ]


def _to_result(model: Forecaster, model_id: str, horizons: int) -> dict:
    return {
        "modelId": model_id,
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "days": _predict_full(model, horizons),
        "rollingUsage": _rolling_usage(model.daily),
        "hourlyProfile": [[round(v, 4) for v in row] for row in model.profile],
    }


# ── Endpunkte ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "models_dir": str(MODELS_DIR)}


@app.get("/api/models")
def list_models() -> List[dict]:
    return [
        {"id": p.stem, "path": str(p),
         "size_kb": round(p.stat().st_size / 1024, 1),
         "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat()}
        for p in sorted(MODELS_DIR.glob("*.joblib"))
    ]


@app.post("/api/forecast")
async def forecast_from_csv(
    file: UploadFile = File(...),
    horizons: int = Form(7),
    history_days: int = Form(100),
    date_col: str = Form("datetime"),
    signal_col: str = Form("in_use"),
    save_model: bool = Form(False),
) -> dict:
    """CSV hochladen → trainieren → Forecast. Optional speichern."""
    try:
        raw = await file.read()
        tmp_path = MODELS_DIR / f"_upload_{uuid.uuid4().hex}.csv"
        tmp_path.write_bytes(raw)
        hourly = load_csv(tmp_path, date_col, signal_col)
        tmp_path.unlink(missing_ok=True)

        model = Forecaster(Config(history_days=history_days)).fit(hourly)
        model_id = f"upload_{uuid.uuid4().hex[:8]}"
        if save_model:
            model.save(MODELS_DIR / f"{model_id}.joblib")

        return _to_result(model, model_id, max(1, min(7, horizons)))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interner Fehler: {e}")


@app.post("/api/forecast/{model_id}")
def forecast_from_model(model_id: str, horizons: int = 7) -> dict:
    """Forecast aus bereits gespeichertem Modell."""
    path = MODELS_DIR / f"{model_id}.joblib"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Modell '{model_id}' nicht gefunden")
    model = Forecaster.load(path)
    return _to_result(model, model_id, max(1, min(7, horizons)))
