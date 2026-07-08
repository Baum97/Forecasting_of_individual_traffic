"""
Naive Baselines fuer den stuendlichen Driving-Classifier, gescort auf demselben
zeitlichen Test-Split (80/20) wie die Cross-Validation, ueber alle Quellen.

Baselines (kein Training ausser einer aus dem Train abgeleiteten Klimatologie):
  - majority    : sagt immer 'geparkt' (0); Prior-Wahrscheinlichkeit = Basisrate.
  - persist168  : sagt driving_t = driving_{t-168} (gleiche Stunde Vorwoche).
  - climatology : P(driving | weekday, hour), geschaetzt auf dem Train-Split.

Kein Modell-Training noetig -> reine Referenzwerte, um die ML-Guete als
Skill (Modell minus Baseline) interpretierbar zu machen.

Aus dem Repo-Root:
    python code/model_scripts/classifier/run_baselines.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_scripts.base import FEATURE_COLS, make_features, run_output_dir, split_by_time
from model_scripts.classifier.run_cross_validation import build_datasets

ORDER = ["emobpy", "real_world_ev", "ved", "routine", "yjmob"]


def _metrics(y: np.ndarray, proba: np.ndarray, preds: np.ndarray) -> dict:
    out = {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "brier": float(brier_score_loss(y, proba, pos_label=1)),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, proba))
        out["pr_auc"] = float(average_precision_score(y, proba))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def baselines_for_source(df: pd.DataFrame) -> list[dict]:
    prepared = make_features(df)
    train, test = split_by_time(prepared)
    test = test.dropna(subset=FEATURE_COLS).copy()
    if len(test) == 0:
        return []

    y = test["driving"].astype(int).to_numpy()
    train_rate = float(train["driving"].mean())

    rows = []

    # majority: immer 0; Prior-Wahrscheinlichkeit = Train-Basisrate.
    proba = np.full(len(y), train_rate)
    preds = np.zeros(len(y), dtype=int)
    rows.append({"baseline": "majority", **_metrics(y, proba, preds)})

    # persistence lag-168: gleiche Stunde der Vorwoche (0/1 als Wahrscheinlichkeit).
    proba = test["lag_168"].astype(float).to_numpy()
    preds = (proba >= 0.5).astype(int)
    rows.append({"baseline": "persist168", **_metrics(y, proba, preds)})

    # climatology: P(driving | weekday, hour) aus dem Train.
    tab = train.groupby(["weekday", "hour"])["driving"].mean().to_dict()
    proba = np.array([tab.get((wd, hr), train_rate)
                      for wd, hr in zip(test["weekday"], test["hour"])], dtype=float)
    preds = (proba >= 0.5).astype(int)
    rows.append({"baseline": "climatology", **_metrics(y, proba, preds)})

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emobpy-vehicles", type=int, default=30)
    parser.add_argument("--ved-vehicles", type=int, default=15)
    parser.add_argument("--ved-files", type=int, default=4)
    parser.add_argument("--yjmob-vehicles", type=int, default=50)
    parser.add_argument("--dataset-tag", default="all_sources_baselines")
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    run_dir = run_output_dir("cross_validation", args.dataset_tag, None)
    out_csv = args.out_csv or run_dir / "baselines_matrix.csv"

    datasets = build_datasets(args.emobpy_vehicles, args.ved_vehicles,
                              args.ved_files, args.yjmob_vehicles)

    rows = []
    for name, df in datasets.items():
        for r in baselines_for_source(df):
            r["source"] = name
            rows.append(r)

    matrix = pd.DataFrame(rows)
    cols = ["baseline", "source", "n", "positive_rate", "accuracy", "f1",
            "precision", "recall", "roc_auc", "pr_auc", "brier"]
    matrix = matrix[[c for c in cols if c in matrix.columns]]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    for bl in ["majority", "persist168", "climatology"]:
        sub = matrix[matrix["baseline"] == bl].set_index("source")
        print(f"\n=== baseline: {bl} ===")
        print(sub[["positive_rate", "accuracy", "f1", "roc_auc", "pr_auc", "brier"]]
              .reindex(ORDER).round(3).to_string())


if __name__ == "__main__":
    main()
