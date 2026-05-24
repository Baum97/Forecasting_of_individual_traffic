"""Train a Forecaster on the VED data source (multi-vehicle)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_scripts.randomforest_forecaster import Config
from model_scripts.forecaster_registry import available_algos, get_forecaster_class
from model_scripts.data_adapters import load_ved

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
SOURCE_NAME = "ved"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=available_algos(), default="rf",
                        help="Algorithmus (siehe forecaster_registry.REGISTRY).")
    parser.add_argument("--limit-vehicles", type=int, default=15)
    parser.add_argument("--limit-files", type=int, default=4,
                        help="Anzahl wochenweiser VED-CSV-Dateien.")
    parser.add_argument("--history-days", type=int, default=100)
    parser.add_argument("--model-out", type=Path, default=None,
                        help="Default: models/<source>_forecaster_<algo>.joblib")
    args = parser.parse_args()

    print(f"[{SOURCE_NAME}] loading data "
          f"(limit_vehicles={args.limit_vehicles}, limit_files={args.limit_files})...")
    df = (
        load_ved(limit_vehicles=args.limit_vehicles, limit_files=args.limit_files)
        .rename(columns={"timestamp": "datetime", "driving": "in_use"})
    )
    print(f"[{SOURCE_NAME}] rows={len(df):,} vehicles={df['vehicle_id'].nunique()}")

    Cls = get_forecaster_class(args.model)
    print(f"[{SOURCE_NAME}] fitting {Cls.__name__} (history_days={args.history_days})...")
    model = Cls(Config(history_days=args.history_days)).fit(df)

    out = args.model_out or MODELS_DIR / f"{SOURCE_NAME}_forecaster_{args.model}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)


if __name__ == "__main__":
    main()
