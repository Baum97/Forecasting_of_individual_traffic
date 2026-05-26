"""Train the driving/parked classifier on the VED data source."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_scripts.base import (
    TrainConfig,
    evaluate,
    save_model,
    split_by_time,
    train_model,
)
from model_scripts.data_adapters import load_ved

MODEL_OUT = Path(__file__).resolve().parents[3] / "models" / "driving_ved.joblib"
SOURCE_NAME = "ved"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-vehicles", type=int, default=15)
    parser.add_argument("--limit-files", type=int, default=4,
                        help="Number of weekly VED files to read (each has ~500k rows).")
    parser.add_argument("--model-out", type=Path, default=MODEL_OUT)
    args = parser.parse_args()

    print(f"[{SOURCE_NAME}] loading data (limit_vehicles={args.limit_vehicles}, "
          f"limit_files={args.limit_files})...")
    df = load_ved(limit_vehicles=args.limit_vehicles, limit_files=args.limit_files)
    print(f"[{SOURCE_NAME}] rows={len(df):,} vehicles={df['vehicle_id'].nunique()} "
          f"positive_rate={df['driving'].mean():.3f}")

    train_df, test_df = split_by_time(df)
    print(f"[{SOURCE_NAME}] train={len(train_df):,} test={len(test_df):,}")

    print(f"[{SOURCE_NAME}] training RandomForestClassifier...")
    model = train_model(train_df, TrainConfig())

    metrics = evaluate(model, test_df)
    print(f"[{SOURCE_NAME}] in-source test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    save_model(model, args.model_out)
    print(f"[{SOURCE_NAME}] saved -> {args.model_out}")


if __name__ == "__main__":
    main()
