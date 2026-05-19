"""Train the driving/parked classifier on the real-world EV data source."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_scripts.base import (
    TrainConfig,
    evaluate,
    save_model,
    split_by_time,
    train_model,
)
from model_scripts.data_adapters import load_real_world_ev

MODEL_OUT = Path(__file__).resolve().parents[2] / "models" / "driving_real_world_ev.joblib"
SOURCE_NAME = "real_world_ev"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-out", type=Path, default=MODEL_OUT)
    args = parser.parse_args()

    print(f"[{SOURCE_NAME}] loading data...")
    df = load_real_world_ev()
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
