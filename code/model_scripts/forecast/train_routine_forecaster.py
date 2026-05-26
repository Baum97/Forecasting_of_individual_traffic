"""Train a Forecaster on the synthetic routine data source."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_scripts.forecast.randomforest_forecaster import Config
from model_scripts.forecast.forecaster_registry import available_algos, get_forecaster_class
from model_scripts.data_adapters import load_routine

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
SOURCE_NAME = "routine"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=available_algos(), default="rf",
                        help="Algorithmus (siehe forecaster_registry.REGISTRY).")
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--model-out", type=Path, default=None,
                        help="Default: models/<source>_forecaster_<algo>.joblib")
    args = parser.parse_args()

    print(f"[{SOURCE_NAME}] loading data...")
    df = load_routine().rename(columns={"timestamp": "datetime", "driving": "in_use"})
    print(f"[{SOURCE_NAME}] rows={len(df):,} vehicles={df['vehicle_id'].nunique()}")

    Cls = get_forecaster_class(args.model)
    print(f"[{SOURCE_NAME}] fitting {Cls.__name__} (history_days={args.history_days})...")
    model = Cls(Config(history_days=args.history_days)).fit(df)

    out = args.model_out or MODELS_DIR / f"{SOURCE_NAME}_forecaster_{args.model}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)


if __name__ == "__main__":
    main()
