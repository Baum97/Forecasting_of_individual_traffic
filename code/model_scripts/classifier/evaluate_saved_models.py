"""
Eval-only re-scoring of the already-trained driving classifiers.

Loads the per-source .joblib models saved by run_cross_validation.py and
re-scores them on the identical time-based test splits to add metrics
(PR-AUC, Brier) WITHOUT retraining. Because models and splits are
deterministic, the existing accuracy/F1/ROC-AUC are reproduced exactly; only
the new columns are added.

Run from the project root:
    python code/model_scripts/classifier/evaluate_saved_models.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_scripts.base import (
    evaluate,
    load_model,
    make_features,
    run_output_dir,
    split_by_time,
)
from model_scripts.classifier.run_cross_validation import MODELS_DIR, build_datasets

ORDER = ["emobpy", "real_world_ev", "ved", "routine", "yjmob"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--emobpy-vehicles", type=int, default=30)
    p.add_argument("--ved-vehicles", type=int, default=15)
    p.add_argument("--ved-files", type=int, default=4)
    p.add_argument("--yjmob-vehicles", type=int, default=50)
    p.add_argument("--algos", default="rf,lgbm",
                   help="Komma-Liste der Algorithmen (rf, lgbm).")
    p.add_argument("--dataset-tag", default="all_sources_eval")
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    run_dir = run_output_dir("cross_validation", args.dataset_tag, None)
    out_csv = args.out_csv or run_dir / "cross_validation_matrix_eval.csv"
    print(f"run output dir: {run_dir}")
    print(f"algorithms: {algos}")

    datasets = build_datasets(args.emobpy_vehicles, args.ved_vehicles,
                              args.ved_files, args.yjmob_vehicles)
    # Identische Pipeline wie im Trainingslauf, damit die Test-Splits exakt
    # denselben Zeilen entsprechen.
    prepared = {name: make_features(df) for name, df in datasets.items()}
    splits = {name: split_by_time(df) for name, df in prepared.items()}

    rows = []
    for algo in algos:
        for model_name in splits:
            fname = (f"driving_{model_name}.joblib" if algo == "rf"
                     else f"driving_{model_name}_{algo}.joblib")
            path = MODELS_DIR / fname
            if not path.exists():
                print(f"skip: model not found {path}")
                continue
            model = load_model(path)
            for data_name, (_, test_df) in splits.items():
                print(f"evaluating [{algo}] {model_name:>14s} -> {data_name}")
                metrics = evaluate(model, test_df)
                metrics["algo"] = algo
                metrics["model_trained_on"] = model_name
                metrics["evaluated_on"] = data_name
                rows.append(metrics)

    matrix = pd.DataFrame(rows)
    column_order = [
        "algo", "model_trained_on", "evaluated_on", "n", "positive_rate",
        "accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc", "brier",
    ]
    matrix = matrix[[c for c in column_order if c in matrix.columns]]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    for algo in algos:
        sub = matrix[matrix["algo"] == algo]
        diag = (sub[sub["model_trained_on"] == sub["evaluated_on"]]
                .set_index("evaluated_on"))
        print(f"\n=== diagonal (in-distribution) [{algo}] ===")
        print(diag[["positive_rate", "accuracy", "f1", "roc_auc", "pr_auc", "brier"]]
              .reindex(ORDER).round(3).to_string())


if __name__ == "__main__":
    main()
