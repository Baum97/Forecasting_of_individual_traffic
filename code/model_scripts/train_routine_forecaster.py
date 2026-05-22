"""Train a Forecaster on the synthetic routine data source."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_scripts.csv_forecaster import Config, Forecaster
from model_scripts.data_adapters import load_routine

MODEL_OUT = Path(__file__).resolve().parents[2] / "models" / "routine_forecaster.joblib"
SOURCE_NAME = "routine"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--model-out", type=Path, default=MODEL_OUT)
    args = parser.parse_args()

    print(f"[{SOURCE_NAME}] loading data...")
    df = load_routine().rename(columns={"timestamp": "datetime", "driving": "in_use"})
    print(f"[{SOURCE_NAME}] rows={len(df):,} vehicles={df['vehicle_id'].nunique()}")

    print(f"[{SOURCE_NAME}] fitting Forecaster (history_days={args.history_days})...")
    model = Forecaster(Config(history_days=args.history_days)).fit(df)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)


if __name__ == "__main__":
    main()
