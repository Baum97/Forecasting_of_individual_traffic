"""
Shared training and evaluation logic for the binary "driving vs parked" task.

Per default every per-source model uses the same algorithm
(RandomForestClassifier) and the same feature set, so the cross-validation
matrix isolates the effect of *data quality* rather than algorithm choice.
For the ablation study a second algorithm (LightGBM) can be selected via
`train_model(..., algo="lgbm")`; the feature set stays identical so that the
algorithm effect can be read off against the RandomForest baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

FEATURE_COLS = [
    "hour",
    "weekday",
    "is_weekend",
    "lag_1",
    "lag_2",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
]


@dataclass
class TrainConfig:
    n_estimators: int = 150
    min_samples_leaf: int = 5
    max_depth: int = 16
    random_state: int = 42
    n_jobs: int = -1


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar + lag + rolling-mean features.

    Input must have columns vehicle_id, timestamp, driving on an hourly grid
    that is contiguous within each vehicle.
    """
    df = df.sort_values(["vehicle_id", "timestamp"]).copy()
    df["hour"] = df["timestamp"].dt.hour.astype(np.int16)
    df["weekday"] = df["timestamp"].dt.weekday.astype(np.int16)
    df["is_weekend"] = (df["weekday"] >= 5).astype(np.int8)

    g = df.groupby("vehicle_id", sort=False)["driving"]
    df["lag_1"] = g.shift(1)
    df["lag_2"] = g.shift(2)
    df["lag_24"] = g.shift(24)
    df["lag_168"] = g.shift(168)

    shifted = g.shift(1)
    df["rolling_mean_24"] = (
        shifted.groupby(df["vehicle_id"]).rolling(24, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    df["rolling_mean_168"] = (
        shifted.groupby(df["vehicle_id"]).rolling(168, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    return df


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features if not already present, then drop rows with NaN lags."""
    if not set(FEATURE_COLS).issubset(df.columns):
        df = make_features(df)
    return df.dropna(subset=FEATURE_COLS)


def split_by_time(df: pd.DataFrame, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time so that lag features computed on the full
    timeline remain valid (i.e. test rows can reference earlier train rows)."""
    cut = df["timestamp"].quantile(train_frac)
    return df[df["timestamp"] <= cut].copy(), df[df["timestamp"] > cut].copy()


def _build_classifier(algo: str, cfg: TrainConfig):
    """Konstruiert den (ungefitteten) Klassifikator. Beide Algorithmen nutzen
    denselben Feature-Satz und balancierte Klassengewichte, damit die Matrix
    den Algorithmus-Effekt isoliert."""
    if algo == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            min_samples_leaf=cfg.min_samples_leaf,
            max_depth=cfg.max_depth,
            class_weight="balanced",
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
    if algo == "lgbm":
        # LightGBM ist optional — nur importieren, wenn tatsaechlich genutzt.
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=cfg.n_estimators,
            min_child_samples=max(cfg.min_samples_leaf, 5),
            max_depth=cfg.max_depth,
            class_weight="balanced",
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            verbosity=-1,
        )
    if algo in ("lstm_tab", "lstm_seq"):
        # torch ist optional — nur importieren, wenn tatsaechlich genutzt.
        from model_scripts.classifier.lstm_classifier import (
            LSTMConfig,
            LSTMSequenceClassifier,
            LSTMTabularClassifier,
        )
        lcfg = LSTMConfig(random_state=cfg.random_state)
        if algo == "lstm_tab":
            return LSTMTabularClassifier(lcfg)
        return LSTMSequenceClassifier(lcfg)
    raise ValueError(
        f"unknown algo: {algo!r} (erwartet 'rf', 'lgbm', 'lstm_tab' oder 'lstm_seq')"
    )


def _with_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features sicherstellen, aber NaN-Zeilen behalten (Sequenzmodelle
    brauchen die Vorgeschichte, die dropna verwerfen wuerde)."""
    return df if set(FEATURE_COLS).issubset(df.columns) else make_features(df)


def train_model(df: pd.DataFrame, cfg: TrainConfig | None = None,
                algo: str = "rf"):
    """Fit the binary driving classifier (algo='rf'/'lgbm'/'lstm_tab'/
    'lstm_seq', siehe _build_classifier). Vorhersagen nutzen die
    Standardschwelle 0.5."""
    cfg = cfg or TrainConfig()
    model = _build_classifier(algo, cfg)

    # Sequenzmodelle bauen ihre Fenster aus der zusammenhaengenden Reihe und
    # brauchen daher den Frame, nicht die Feature-Matrix.
    if getattr(model, "needs_frame", False):
        return model.fit_frame(_with_features(df))

    feats = _ensure_features(df)
    X = feats[FEATURE_COLS].to_numpy()
    y = feats["driving"].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        raise ValueError("training data has only one class — cannot fit classifier")

    model.fit(X, y)
    return model


def evaluate(model, df: pd.DataFrame) -> dict:
    if getattr(model, "needs_frame", False):
        frame = _with_features(df)
        feats = frame.sort_values(["vehicle_id", "timestamp"]).dropna(subset=FEATURE_COLS)
        if len(feats) == 0:
            return {"n": 0}
        proba_all = model.predict_proba_frame(frame)
        preds = (proba_all[:, 1] >= 0.5).astype(int)
    else:
        feats = _ensure_features(df)
        if len(feats) == 0:
            return {"n": 0}
        X = feats[FEATURE_COLS].to_numpy()
        proba_all = model.predict_proba(X)
        preds = model.predict(X)

    y = feats["driving"].astype(int).to_numpy()
    out = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
    }

    proba = proba_all[:, 1]
    # Brier-Score = MSE der Wahrscheinlichkeiten (Kalibrierung); auch bei nur
    # einer Klasse definiert.
    out["brier"] = float(brier_score_loss(y, proba, pos_label=1))

    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, proba))
        # PR-AUC (Average Precision): imbalance-ehrliche Detektion der
        # Minderheitsklasse; Zufalls-Baseline = positive_rate.
        out["pr_auc"] = float(average_precision_score(y, proba))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


PREDICTIONS_DIR = Path(__file__).resolve().parents[2] / "predictions"


def run_output_dir(
    model_name: str,
    dataset: str,
    forecast_days: Optional[int] = None,
    base: Optional[Path] = None,
    when: Optional[datetime] = None,
) -> Path:
    """Return predictions/<model>/<dataset>[_<N>d]_T<DD-MM-YYYY>/ and create it.

    Note: ':' is not allowed in Windows paths, so the requested 'T<DD:MM:YYYY>'
    tag uses '-' as separator instead.
    """
    base = base or PREDICTIONS_DIR
    when = when or datetime.now()
    date_tag = "T" + when.strftime("%d-%m-%Y")
    parts = [dataset]
    if forecast_days is not None:
        parts.append(f"{forecast_days}d")
    parts.append(date_tag)
    out = base / model_name / "_".join(parts)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    return joblib.load(path)
