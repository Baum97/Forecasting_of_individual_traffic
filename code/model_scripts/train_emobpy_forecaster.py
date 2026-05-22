"""Train a Forecaster on the emobpy data source (multi-vehicle)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_scripts.csv_forecaster import Config, Forecaster
from model_scripts.data_adapters import load_emobpy

MODEL_OUT = Path(__file__).resolve().parents[2] / "models" / "emobpy_forecaster.joblib"
SOURCE_NAME = "emobpy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-vehicles", type=int, default=30,
                        help="Anzahl emobpy-Fahrzeuge (max 200).")
    parser.add_argument("--history-days", type=int, default=100)
    parser.add_argument("--model-out", type=Path, default=MODEL_OUT)
    args = parser.parse_args()

    print(f"[{SOURCE_NAME}] loading data (limit_vehicles={args.limit_vehicles})...")
    df = (
        load_emobpy(limit_vehicles=args.limit_vehicles)
        .rename(columns={"timestamp": "datetime", "driving": "in_use"})
    )
    print(f"[{SOURCE_NAME}] rows={len(df):,} vehicles={df['vehicle_id'].nunique()}")

    print(f"[{SOURCE_NAME}] fitting Forecaster (history_days={args.history_days})...")
    model = Forecaster(Config(history_days=args.history_days)).fit(df)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)


if __name__ == "__main__":
    main()
