"""
Train one model per data source and evaluate every model on every source.

Each dataset is split into train/test by time (80/20). Each model is fit on
the train portion of its own source, then scored on the test portion of all
four sources. The result is a 4x4 matrix of metrics plus an F1 heatmap.

Run from the project root:
    python code/model_scripts/run_cross_validation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_scripts.base import (
    TrainConfig,
    evaluate,
    make_features,
    save_model,
    split_by_time,
    train_model,
)
from model_scripts.data_adapters import (
    load_emobpy,
    load_real_world_ev,
    load_routine,
    load_ved,
)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
PRED_DIR = Path(__file__).resolve().parents[2] / "predictions"


def build_datasets(emobpy_n: int, ved_n: int, ved_files: int) -> dict:
    print("loading datasets...")
    out = {}
    print(f"  emobpy (limit_vehicles={emobpy_n})...")
    out["emobpy"] = load_emobpy(limit_vehicles=emobpy_n)
    print(f"  real_world_ev...")
    out["real_world_ev"] = load_real_world_ev()
    print(f"  ved (limit_vehicles={ved_n}, limit_files={ved_files})...")
    out["ved"] = load_ved(limit_vehicles=ved_n, limit_files=ved_files)
    print(f"  routine...")
    out["routine"] = load_routine()

    for name, df in out.items():
        print(
            f"    {name}: rows={len(df):,} "
            f"vehicles={df['vehicle_id'].nunique()} "
            f"positive_rate={df['driving'].mean():.3f}"
        )
    return out


def heatmap(matrix: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    pivot = matrix.pivot(index="model_trained_on", columns="evaluated_on", values=metric)
    order = ["emobpy", "real_world_ev", "ved", "routine"]
    pivot = pivot.reindex(index=order, columns=order)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    values = pivot.values.astype(float)
    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("evaluated on")
    ax.set_ylabel("trained on")
    ax.set_title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                txt = "n/a"
                color = "white"
            else:
                txt = f"{v:.2f}"
                color = "white" if v < 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color)

    plt.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emobpy-vehicles", type=int, default=30)
    parser.add_argument("--ved-vehicles", type=int, default=15)
    parser.add_argument("--ved-files", type=int, default=4)
    parser.add_argument("--out-csv", type=Path,
                        default=PRED_DIR / "cross_validation_matrix.csv")
    parser.add_argument("--heatmap-out", type=Path,
                        default=PRED_DIR / "cross_validation_f1_heatmap.png")
    parser.add_argument("--accuracy-heatmap-out", type=Path,
                        default=PRED_DIR / "cross_validation_accuracy_heatmap.png")
    args = parser.parse_args()

    datasets = build_datasets(args.emobpy_vehicles, args.ved_vehicles, args.ved_files)
    # Compute features on the full timeline first, then split.
    # Otherwise lag_168 (1 week) would invalidate short test sets like VED.
    prepared = {name: make_features(df) for name, df in datasets.items()}
    splits = {name: split_by_time(df) for name, df in prepared.items()}

    cfg = TrainConfig()
    models = {}
    for name, (train_df, _) in splits.items():
        print(f"training model on {name}...")
        models[name] = train_model(train_df, cfg)
        save_model(models[name], MODELS_DIR / f"driving_{name}.joblib")

    rows = []
    for model_name, model in models.items():
        for data_name, (_, test_df) in splits.items():
            print(f"evaluating {model_name:>14s} -> {data_name}")
            metrics = evaluate(model, test_df)
            metrics["model_trained_on"] = model_name
            metrics["evaluated_on"] = data_name
            rows.append(metrics)

    matrix = pd.DataFrame(rows)
    column_order = [
        "model_trained_on", "evaluated_on", "n", "positive_rate",
        "accuracy", "f1", "precision", "recall", "roc_auc",
    ]
    matrix = matrix[[c for c in column_order if c in matrix.columns]]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}")

    heatmap(matrix, "f1", "F1 — driving classifier cross-validation", args.heatmap_out)
    heatmap(matrix, "accuracy", "Accuracy — driving classifier cross-validation",
            args.accuracy_heatmap_out)

    print("\n=== summary (F1) ===")
    pivot = matrix.pivot(index="model_trained_on", columns="evaluated_on", values="f1")
    order = ["emobpy", "real_world_ev", "ved", "routine"]
    print(pivot.reindex(index=order, columns=order).round(3).to_string())


if __name__ == "__main__":
    main()
