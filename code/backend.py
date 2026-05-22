"""
Endpunkte:
  GET  /api/health                       - Liveness-Check
  GET  /api/models                       - Liste gespeicherter .joblib-Modelle
  POST /api/forecast                     - CSV hochladen, trainieren, Forecast liefern
  POST /api/forecast/{model_id}          - Vorhersage aus vorhandenem Modell
  GET  /api/runs                         - Liste aller Run-Verzeichnisse in predictions/
  GET  /api/runs/{model}/{run}           - Inhalt eines Runs (Forecast + Artefakte)
  GET  /runs/{model}/{run}/{datei}       - Statische Auslieferung einzelner Artefakte

Starten:
  python -m uvicorn backend:app --reload --port 8000
"""
from __future__ import annotations

import io
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from model_scripts.base import run_output_dir
from model_scripts.csv_forecaster import (
    Config, FEATURE_COLS, Forecaster,
    load_csv, make_features,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)
PRED_DIR = Path(__file__).resolve().parents[1] / "predictions"
PRED_DIR.mkdir(exist_ok=True)


_FORECASTER_META = {
    "realworldev_forecaster": (
        "Real-World EV Forecaster",
        "1-7-Tage-Forecaster auf der realen EV-Timeline (cars-real-world-electric).",
        "real_world",
    ),
    "emobpy_forecaster": (
        "emobpy Forecaster",
        "1-7-Tage-Forecaster ueber alle emobpy-Fahrzeuge (gemeinsam trainiert).",
        "simulation",
    ),
    "ved_forecaster": (
        "VED Forecaster",
        "1-7-Tage-Forecaster ueber mehrere VED-Fahrzeuge (gemeinsam trainiert).",
        "real_world",
    ),
    "routine_forecaster": (
        "Routine Forecaster",
        "1-7-Tage-Forecaster auf der synthetischen Routine-Zeitreihe.",
        "simulation",
    ),
    "car_full_forecaster": (
        "Car Full Forecaster (Legacy)",
        "Aelterer Forecaster auf cars-real-world-electric — heute durch realworldev_forecaster ersetzt.",
        "real_world",
    ),
}


def _describe_model(stem: str) -> dict:
    """Leitet Label, Beschreibung und Typ aus dem Dateinamen ab."""
    if stem in _FORECASTER_META:
        label, description, type_ = _FORECASTER_META[stem]
        return {"label": label, "description": description, "type": type_}
    if stem.startswith("driving_"):
        source = stem.removeprefix("driving_")
        pretty = source.replace("_", " ").title()
        return {
            "label": f"Driving Classifier - {pretty}",
            "description": f'RandomForest "faehrt vs. geparkt", trainiert auf Datenquelle "{source}".',
            "type": "driving_classifier",
        }
    return {"label": stem, "description": "", "type": "unknown"}

app = FastAPI(title="Traffic Forecaster API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Run-Artefakte (CSV, PNG, TXT, ...) als statische Dateien ausliefern.
app.mount("/runs", StaticFiles(directory=str(PRED_DIR)), name="runs")


# ── Run-Verzeichnisse als "Simulations-Objekte" ──────────────────────────────

_RUN_DIR_RE = re.compile(
    r"^(?P<dataset>.+?)(?:_(?P<days>\d+)d)?_T(?P<date>\d{2}-\d{2}-\d{4})$"
)


def _parse_run_dir_name(name: str) -> dict:
    """dataset_<N>d_T<DD-MM-YYYY> -> {dataset, days?, date}."""
    m = _RUN_DIR_RE.match(name)
    if not m:
        return {"dataset": name, "days": None, "date": None}
    days = int(m["days"]) if m["days"] else None
    return {"dataset": m["dataset"], "days": days, "date": m["date"]}


def _artifact_kind(suffix: str) -> str:
    s = suffix.lower()
    if s == ".csv":
        return "csv"
    if s in {".png", ".jpg", ".jpeg", ".svg"}:
        return "image"
    if s in {".txt", ".md", ".log"}:
        return "text"
    if s == ".json":
        return "json"
    return "other"


def _list_artifacts(run_dir: Path, rel_url_prefix: str) -> List[dict]:
    items: List[dict] = []
    for f in sorted(run_dir.iterdir()):
        if not f.is_file():
            continue
        items.append({
            "name": f.name,
            "kind": _artifact_kind(f.suffix),
            "size_kb": round(f.stat().st_size / 1024, 1),
            "url": f"{rel_url_prefix}/{f.name}",
        })
    return items


def _scan_runs() -> List[dict]:
    """Scant predictions/<model>/<run>/ und gibt eine flache Liste zurueck."""
    runs: List[dict] = []
    if not PRED_DIR.exists():
        return runs
    for model_dir in sorted(p for p in PRED_DIR.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            meta = _parse_run_dir_name(run_dir.name)
            runs.append({
                "model": model_dir.name,
                "run": run_dir.name,
                **meta,
                "artifactCount": sum(1 for f in run_dir.iterdir() if f.is_file()),
                "modified": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
            })
    return runs


def _load_run(model: str, run: str) -> dict:
    run_dir = PRED_DIR / model / run
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{model}/{run}' nicht gefunden")

    meta = _parse_run_dir_name(run)
    artifacts = _list_artifacts(run_dir, rel_url_prefix=f"/runs/{model}/{run}")

    rows: List[dict] = []
    csv_file = run_dir / "forecast.csv"
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            rows = df.to_dict(orient="records")
        except Exception as e:
            rows = []
            artifacts.append({"name": "forecast.csv", "kind": "error",
                              "size_kb": 0, "url": "",
                              "error": f"CSV nicht lesbar: {e}"})

    notes = ""
    notes_file = run_dir / "notes.txt"
    if notes_file.exists():
        notes = notes_file.read_text(encoding="utf-8", errors="ignore")

    return {
        "model": model,
        "run": run,
        **meta,
        "artifacts": artifacts,
        "rows": rows,
        "notes": notes,
    }


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


def _persist_run_csv(model_name: str, dataset: str, horizons: int,
                     days: List[dict]) -> Path:
    """Schreibt die Forecast-Tage als CSV in predictions/<model>/<dataset>_<N>d_T<DD-MM-YYYY>/."""
    run_dir = run_output_dir(
        model_name=model_name,
        dataset=dataset,
        forecast_days=horizons,
    )
    out = run_dir / "forecast.csv"
    pd.DataFrame(days).drop(columns=["hourProfile"], errors="ignore").to_csv(
        out, index=False
    )
    return out


def _to_result(model: Forecaster, model_id: str, horizons: int,
               dataset: str = "default") -> dict:
    days = _predict_full(model, horizons)
    csv_path = _persist_run_csv(model_id, dataset, horizons, days)
    return {
        "modelId": model_id,
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "days": days,
        "rollingUsage": _rolling_usage(model.daily),
        "hourlyProfile": [[round(v, 4) for v in row] for row in model.profile],
        "runCsvPath": str(csv_path),
    }


# ── Endpunkte ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "models_dir": str(MODELS_DIR)}


@app.get("/api/runs")
def list_runs() -> List[dict]:
    """Listet alle Simulations-Verzeichnisse unter predictions/."""
    return _scan_runs()


@app.get("/api/runs/{model}/{run}")
def get_run(model: str, run: str) -> dict:
    """Liefert Forecast-Tage + Artefakt-Liste eines konkreten Runs."""
    return _load_run(model, run)


@app.get("/api/models")
def list_models() -> List[dict]:
    return [
        {
            "id": p.stem,
            **_describe_model(p.stem),
            "path": str(p),
            "size_kb": round(p.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
        }
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
    dataset: str = Form("upload"),
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

        return _to_result(model, model_id, max(1, min(7, horizons)), dataset=dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interner Fehler: {e}")


@app.post("/api/forecast/{model_id}")
def forecast_from_model(model_id: str, horizons: int = 7,
                        dataset: str = "default") -> dict:
    """Forecast aus bereits gespeichertem Modell."""
    path = MODELS_DIR / f"{model_id}.joblib"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Modell '{model_id}' nicht gefunden")
    model = Forecaster.load(path)
    return _to_result(model, model_id, max(1, min(7, horizons)), dataset=dataset)
