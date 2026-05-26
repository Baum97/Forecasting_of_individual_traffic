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
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model_scripts.base import run_output_dir
from model_scripts.forecast.randomforest_forecaster import (
    Config, FEATURE_COLS, RandomForestForecaster,
    load_csv, make_features,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)
PRED_DIR = Path(__file__).resolve().parents[1] / "predictions"
PRED_DIR.mkdir(exist_ok=True)


# Quellen-Metadaten (Label + Typ) fuer das Frontend.
_SOURCE_META = {
    "realworldev": ("Real-World EV", "Echte EV-Timeline (cars-real-world-electric).", "real_world"),
    "emobpy":      ("emobpy",        "Simulierte emobpy-Fahrzeuge.",                "simulation"),
    "ved":         ("VED",           "Vehicle Energy Dataset (Univ. of Michigan).", "real_world"),
    "routine":     ("Routine",       "Synthetische Routine-Zeitreihe.",             "simulation"),
}

# Lesbare Algo-Namen.
_ALGO_LABELS = {"rf": "RandomForest", "lgbm": "LightGBM"}


def _describe_model(stem: str) -> dict:
    """Leitet Label, Beschreibung und Typ aus dem Dateinamen ab.

    Neues Namensschema: <source>_forecaster_<algo>
    Legacy: <source>_forecaster (kein Algo-Suffix, defaultet auf rf)
    Legacy: car_full_forecaster
    Driving-Classifier: driving_<source>
    """
    # Legacy Sonderfall.
    if stem == "car_full_forecaster":
        return {
            "label": "Car Full Forecaster (Legacy)",
            "description": "Aelterer RF-Forecaster auf cars-real-world-electric.",
            "type": "real_world",
        }

    if stem.endswith("_forecaster") or "_forecaster_" in stem:
        # Splitten in source / algo.
        if "_forecaster_" in stem:
            source, algo = stem.split("_forecaster_", 1)
        else:
            source = stem.removesuffix("_forecaster")
            algo = "rf"
        src_label, src_desc, src_type = _SOURCE_META.get(
            source, (source.title(), f"Datenquelle '{source}'.", "unknown")
        )
        algo_label = _ALGO_LABELS.get(algo, algo.upper())
        return {
            "label": f"{src_label} — {algo_label}",
            "description": f"1-7-Tage-Forecaster ({algo_label}) auf {src_desc}",
            "type": src_type,
        }

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

WEEKDAYS_DE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


def _hourly_grid_to_json(grid: pd.DataFrame) -> List[dict]:
    """Wandelt das MC-Grid in eine schmale JSON-freundliche Liste."""
    out: List[dict] = []
    for _, r in grid.iterrows():
        ts: pd.Timestamp = r["timestamp"]
        out.append({
            "timestamp": ts.isoformat(),
            "date": ts.strftime("%Y-%m-%d"),
            "hour": int(ts.hour),
            "weekday": int(ts.weekday()),
            "pMean": round(float(r["p_mean"]), 3),
            "p10": round(float(r["p10"]), 3),
            "p50": round(float(r["p50"]), 3),
            "p90": round(float(r["p90"]), 3),
        })
    return out


def _extract_trips(p_means: np.ndarray, p10s: np.ndarray, p90s: np.ndarray,
                   threshold: float) -> List[dict]:
    """Findet zusammenhaengende Aktiv-Bloecke in einem 24-h-Tag.

    Eine Folge benachbarter Stunden mit p_mean >= threshold gilt als ein Trip.
    Der Returns-Block beschreibt jeweils Start- und End-Stunde, Dauer,
    Peak- und Mittel-Wahrscheinlichkeit sowie das pessimistischste/optimistischste
    Quantil im Block.
    """
    active = p_means >= threshold
    blocks: List[dict] = []
    i, n = 0, len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i + 1
        while j < n and active[j]:
            j += 1
        blocks.append({
            "startH": int(i),
            "endH": int(j),            # exklusiv (Trip endet beim Beginn dieser Stunde)
            "durationH": int(j - i),
            "peakP": round(float(np.max(p_means[i:j])), 3),
            "meanP": round(float(np.mean(p_means[i:j])), 3),
            "p10":   round(float(np.min(p10s[i:j])), 3),
            "p90":   round(float(np.max(p90s[i:j])), 3),
        })
        i = j
    return blocks


def _adaptive_threshold(base_rate: float) -> float:
    """Pro-Fahrzeug-Aktivitaetsschwelle. Eine Stunde gilt als aktiv, wenn
    ihre vorhergesagte Wahrscheinlichkeit deutlich ueber dem typischen
    Aktivitaetsniveau dieses Fahrzeugs liegt.

    Geclampt auf [0.30, 0.60]:
    - Untergrenze 0.30 verhindert, dass bei sehr niedriger Basisrate jede
      Zufallsanstiegung als 'aktiv' gilt.
    - Obergrenze 0.60 verhindert, dass bei hoher Basisrate (Pendler) der
      Forecast nichts mehr als aktiv markiert.
    """
    return max(0.30, min(0.60, base_rate * 1.5))


def _aggregate_days_from_hourly(grid: pd.DataFrame, base_rate: float = 0.5) -> List[dict]:
    """Tageskarten aus dem Stunden-Grid aggregieren (Hybrid-Pipeline).

    - threshold = adaptive aus base_rate (siehe _adaptive_threshold).
    - pUsed: Anteil der Stunden mit p_mean >= threshold (Aktivitaetsbreite).
    - depEst / retEst: erste / letzte Stunde mit p_mean >= threshold.
    - P10/P90 zu dep/ret aus den entsprechenden Quantilen (im 'soft'-Modus
      identisch zu Est).
    - hourProfile: p_mean pro Stunde fuer diesen Tag (24-Werte).
    """
    threshold = _adaptive_threshold(base_rate)

    grid = grid.copy()
    grid["date_str"] = grid["timestamp"].dt.strftime("%Y-%m-%d")
    grid["hour"] = grid["timestamp"].dt.hour

    out: List[dict] = []
    for day_idx, (date_str, day_df) in enumerate(grid.groupby("date_str", sort=True), start=1):
        day_df = day_df.sort_values("hour").reset_index(drop=True)
        wd = day_df["timestamp"].iloc[0].weekday()

        p_means = day_df["p_mean"].to_numpy()
        p10s = day_df["p10"].to_numpy()
        p90s = day_df["p90"].to_numpy()

        active_mean = p_means >= threshold
        active_low  = p10s   >= threshold  # konservativ: auch P10 hoch
        active_high = p90s   >= threshold  # optimistisch: P90 reicht

        def _first_or_none(mask):
            idx = np.where(mask)[0]
            return float(idx[0]) if len(idx) else None

        def _last_or_none(mask):
            idx = np.where(mask)[0]
            return float(idx[-1]) if len(idx) else None

        dep_est = _first_or_none(active_mean)
        ret_est = _last_or_none(active_mean)
        # P10-Band: enger (nur Stunden, die selbst im pessimistischen Quantil aktiv sind)
        dep_p10 = _first_or_none(active_high)  # frueheste plausible Abfahrt
        dep_p90 = _first_or_none(active_low)   # spaeteste konservative Abfahrt
        ret_p10 = _last_or_none(active_low)    # frueheste konservative Rueckkehr
        ret_p90 = _last_or_none(active_high)   # spaeteste plausible Rueckkehr

        p_used = float(active_mean.mean()) if len(active_mean) else 0.0
        trips = _extract_trips(p_means, p10s, p90s, threshold)

        out.append({
            "forecastDay": day_idx,
            "date": date_str,
            "weekday": WEEKDAYS_DE[wd],
            "pUsed": round(p_used, 3),
            "depEst": round(dep_est, 1) if dep_est is not None else None,
            "depP10": round(dep_p10, 1) if dep_p10 is not None else None,
            "depP90": round(dep_p90, 1) if dep_p90 is not None else None,
            "retEst": round(ret_est, 1) if ret_est is not None else None,
            "retP10": round(ret_p10, 1) if ret_p10 is not None else None,
            "retP90": round(ret_p90, 1) if ret_p90 is not None else None,
            "hourProfile": [round(float(v), 4) for v in p_means],
            # Multi-Trip-Liste: jeder Eintrag {startH, endH, durationH, peakP, meanP, p10, p90}.
            "trips": trips,
            "tripCount": len(trips),
            "thresholdUsed": round(threshold, 3),
        })
    return out


def _predict_full(model, horizons: int) -> List[dict]:
    """Wie Forecaster.predict, ergaenzt aber P10/P90 aus den Estimator-Baeumen."""
    last_idx = len(model.daily) - 1
    last_date = model.daily["date"].iloc[last_idx]
    weekdays = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    out: List[dict] = []

    for offset in range(1, horizons + 1):
        target = last_date + pd.Timedelta(days=offset)
        feat = make_features(model.daily, last_idx, model.cfg.history_days, offset)
        X = np.array([[feat[c] if not pd.isna(feat[c]) else 0.0 for c in FEATURE_COLS]])

        # Algo-agnostisch: jede Forecaster-Klasse stellt predict_proba_day +
        # predict_quantiles bereit (Schnittstelle aus forecaster_registry).
        p_used = model.predict_proba_day(X)
        dep, dep10, dep90 = model.predict_quantiles(X, "dep")
        ret, ret10, ret90 = model.predict_quantiles(X, "ret")
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


def _to_result(model, model_id: str, horizons: int,
               dataset: str = "default", mc_samples: int = 100,
               mode: str = "mc") -> dict:
    """Hybrid: wenn das Modell ein Hourly-Schwesternmodell hat, ist die
    Stunden-Vorhersage die Quelle der Wahrheit; Tageskarten werden daraus
    aggregiert. Sonst Fallback auf den alten Day-Pfad.

    mode: 'mc' (Monte-Carlo, mit Streuung) oder 'soft' (deterministisch,
          kontinuierliche Lag-Propagation, stabiler bei niedrigen Basisraten).
    """
    hourly_grid_json: List[dict] = []
    if getattr(model, "hourly_clf", None) is not None:
        grid = model.predict_hourly_mc(
            n_days=horizons, n_samples=mc_samples, mode=mode,
        )
        # Basisrate des Default-Fahrzeugs aus self.daily holen.
        try:
            base_rate = float(model.daily["is_used"].mean()) if model.daily is not None else 0.3
        except Exception:
            base_rate = 0.3
        days = _aggregate_days_from_hourly(grid, base_rate=base_rate)
        hourly_grid_json = _hourly_grid_to_json(grid)
    else:
        days = _predict_full(model, horizons)

    csv_path = _persist_run_csv(model_id, dataset, horizons, days)
    return {
        "modelId": model_id,
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "days": days,
        "rollingUsage": _rolling_usage(model.daily),
        "hourlyProfile": [[round(v, 4) for v in row] for row in model.profile],
        "hourlyGrid": hourly_grid_json,
        "runCsvPath": str(csv_path),
        "propagationMode": mode,
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

        # CSV-Upload trainiert immer einen RandomForest-Forecaster — der
        # `/api/forecast`-Endpunkt erlaubt aktuell keine Algo-Auswahl.
        model = RandomForestForecaster(Config(history_days=history_days)).fit(hourly)
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
                        dataset: str = "default", mode: str = "mc") -> dict:
    """Forecast aus bereits gespeichertem Modell.

    mode: 'mc' (Monte-Carlo, Default) oder 'soft' (deterministische
          kontinuierliche Lag-Propagation, stabiler bei niedrigen Basisraten).
    """
    path = MODELS_DIR / f"{model_id}.joblib"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Modell '{model_id}' nicht gefunden")
    if mode not in {"mc", "soft"}:
        raise HTTPException(status_code=400, detail=f"unbekannter Mode '{mode}'")
    model = joblib.load(path)
    return _to_result(model, model_id, max(1, min(7, horizons)),
                      dataset=dataset, mode=mode)


# ── Training-Jobs (Subprocess) ───────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_SCRIPTS = {
    "emobpy":      _REPO_ROOT / "code" / "model_scripts" / "forecast" / "train_emobpy_forecaster.py",
    "realworldev": _REPO_ROOT / "code" / "model_scripts" / "forecast" / "train_realworldev_forecaster.py",
    "routine":     _REPO_ROOT / "code" / "model_scripts" / "forecast" / "train_routine_forecaster.py",
    "ved":         _REPO_ROOT / "code" / "model_scripts" / "forecast" / "train_ved_forecaster.py",
}
_ALLOWED_ALGOS = {"rf", "lgbm"}

# In-Memory-Jobs. Pro Job: status, log-Zeilen, exit_code, command. Nur fuer
# Single-User-Setup gedacht — neustart-fluechtig.
_JOBS: Dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


class TrainRequest(BaseModel):
    source: str
    algo: str
    limit_vehicles: Optional[int] = None
    history_days: Optional[int] = None


def _run_training_job(job_id: str, cmd: List[str]) -> None:
    """Hintergrund-Thread: subprocess starten, stdout/stderr in den Job pumpen."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(_REPO_ROOT),
        )
        with _JOBS_LOCK:
            _JOBS[job_id]["pid"] = proc.pid
        assert proc.stdout is not None
        for line in proc.stdout:
            with _JOBS_LOCK:
                _JOBS[job_id]["log"].append(line.rstrip("\n"))
        proc.wait()
        with _JOBS_LOCK:
            _JOBS[job_id]["exit_code"] = proc.returncode
            _JOBS[job_id]["status"] = "done" if proc.returncode == 0 else "error"
    except Exception as e:
        with _JOBS_LOCK:
            _JOBS[job_id]["log"].append(f"[backend] subprocess fehlgeschlagen: {e}")
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["exit_code"] = -1


@app.post("/api/train")
def start_training(req: TrainRequest) -> dict:
    """Startet ein train_<source>_forecaster.py in einem Hintergrund-Thread."""
    if req.source not in _TRAIN_SCRIPTS:
        raise HTTPException(status_code=400, detail=f"unbekannte Quelle '{req.source}'")
    if req.algo not in _ALLOWED_ALGOS:
        raise HTTPException(status_code=400, detail=f"unbekannter Algo '{req.algo}'")

    script = _TRAIN_SCRIPTS[req.source]
    if not script.exists():
        raise HTTPException(status_code=500, detail=f"Skript fehlt: {script}")

    cmd: List[str] = [sys.executable, str(script), "--model", req.algo]
    if req.limit_vehicles is not None and req.source in {"emobpy", "ved"}:
        cmd += ["--limit-vehicles", str(req.limit_vehicles)]
    if req.history_days is not None:
        cmd += ["--history-days", str(req.history_days)]

    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "running",
            "source": req.source,
            "algo": req.algo,
            "command": " ".join(cmd),
            "log": [],
            "exit_code": None,
            "started_at": datetime.utcnow().isoformat() + "Z",
        }

    threading.Thread(target=_run_training_job, args=(job_id, cmd), daemon=True).start()
    return {"job_id": job_id, "command": " ".join(cmd)}


@app.get("/api/train/{job_id}")
def get_training_status(job_id: str, offset: int = 0) -> dict:
    """Status + neue Log-Zeilen ab `offset` zurueckgeben."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' nicht gefunden")
        log = job["log"][offset:]
        return {
            "status": job["status"],
            "exit_code": job["exit_code"],
            "command": job["command"],
            "source": job["source"],
            "algo": job["algo"],
            "log": log,
            "log_offset": offset + len(log),
        }
