"""Train a Forecaster on the VED data source (multi-vehicle)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_scripts.csv_forecaster import Config, Forecaster
from model_scripts.data_adapters import load_ved

MODEL_OUT = Path(__file__).resolve().parents[2] / "models" / "ved_forecaster.joblib"
SOURCE_NAME = "ved"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-vehicles", type=int, default=15)
    parser.add_argument("--limit-files", type=int, default=4,
                        help="Anzahl wochenweiser VED-CSV-Dateien.")
    parser.add_argument("--history-days", type=int, default=100)
    parser.add_argument("--model-out", type=Path, default=MODEL_OUT)
    args = parser.parse_args()

    print(f"[{SOURCE_NAME}] loading data "
          f"(limit_vehicles={args.limit_vehicles}, limit_files={args.limit_files})...")
    df = (
        load_ved(limit_vehicles=args.limit_vehicles, limit_files=args.limit_files)
        .rename(columns={"timestamp": "datetime", "driving": "in_use"})
    )
    print(f"[{SOURCE_NAME}] rows={len(df):,} vehicles={df['vehicle_id'].nunique()}")

    print(f"[{SOURCE_NAME}] fitting Forecaster (history_days={args.history_days})...")
    model = Forecaster(Config(history_days=args.history_days)).fit(df)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)


if __name__ == "__main__":
    main()
