"""
Shared training and evaluation logic for the binary "driving vs parked" task.

All four per-source models use the same algorithm (RandomForestClassifier)
and the same feature set, so the cross-validation matrix isolates the effect
of *data quality* rather than algorithm choice.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
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
    """Split per-vehicle by time so that lag features computed on the full
    timeline remain valid (i.e. test rows can reference earlier train rows)."""
    cut = df["timestamp"].quantile(train_frac)
    return df[df["timestamp"] <= cut].copy(), df[df["timestamp"] > cut].copy()


def train_model(df: pd.DataFrame, cfg: TrainConfig | None = None) -> RandomForestClassifier:
    cfg = cfg or TrainConfig()
    feats = _ensure_features(df)
    X = feats[FEATURE_COLS].to_numpy()
    y = feats["driving"].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        raise ValueError("training data has only one class — cannot fit classifier")

    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        min_samples_leaf=cfg.min_samples_leaf,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    model.fit(X, y)
    return model


def evaluate(model: RandomForestClassifier, df: pd.DataFrame) -> dict:
    feats = _ensure_features(df)
    X = feats[FEATURE_COLS].to_numpy()
    y = feats["driving"].astype(int).to_numpy()

    if len(y) == 0:
        return {"n": 0}

    preds = model.predict(X)
    out = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
    }

    if len(np.unique(y)) > 1:
        proba = model.predict_proba(X)[:, 1]
        out["roc_auc"] = float(roc_auc_score(y, proba))
    else:
        out["roc_auc"] = float("nan")
    return out


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    return joblib.load(path)
