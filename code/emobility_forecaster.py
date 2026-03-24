from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ForecasterConfig:
    history_days: int = 100
    horizons: Tuple[int, ...] = (2, 3, 4, 5, 6, 7)
    random_state: int = 42
    n_estimators: int = 350
    min_samples_leaf: int = 2


class MultiHorizonDailyForecaster:
    def __init__(self, config: ForecasterConfig):
        self.config = config
        self.models: Dict[int, RandomForestRegressor] = {}
        self.feature_columns: List[str] = []
        self.trained_target_column: str = ""

    @staticmethod
    def read_emobpy_csv(path: Path) -> pd.DataFrame:
        # emobpy format has two header rows and a metadata row before time series values.
        hdr = pd.read_csv(path, nrows=2, header=None, dtype=str)
        top = hdr.iloc[0].fillna("").astype(str).str.strip()
        sub = hdr.iloc[1].fillna("").astype(str).str.strip()

        columns: List[str] = []
        for i in range(len(top)):
            if i == 0:
                columns.append("date")
                continue
            left = top.iloc[i]
            right = sub.iloc[i]
            if left and right:
                columns.append(f"{left}__{right}")
            elif left:
                columns.append(left)
            elif right:
                columns.append(right)
            else:
                columns.append(f"col_{i}")

        df = pd.read_csv(
            path,
            skiprows=3,
            header=None,
            names=columns,
            parse_dates=[0],
            low_memory=False,
        )
        return df

    @staticmethod
    def read_long_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, parse_dates=["date"], low_memory=False)

    @staticmethod
    def to_daily_panel(
        df: pd.DataFrame,
        vehicle_col: str,
        date_col: str,
        target_col: str,
        agg: str = "sum",
    ) -> pd.DataFrame:
        work = df[[vehicle_col, date_col, target_col]].copy()
        work[date_col] = pd.to_datetime(work[date_col])
        work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
        work = work.dropna(subset=[vehicle_col, date_col, target_col])
        work["date_day"] = work[date_col].dt.floor("D")

        if agg == "mean":
            grouped = (
                work.groupby([vehicle_col, "date_day"], as_index=False)[target_col].mean()
            )
        else:
            grouped = (
                work.groupby([vehicle_col, "date_day"], as_index=False)[target_col].sum()
            )

        grouped = grouped.rename(columns={vehicle_col: "vehicle_id", "date_day": "date", target_col: "target"})
        return grouped.sort_values(["vehicle_id", "date"]).reset_index(drop=True)

    def _build_examples_for_group(self, g: pd.DataFrame) -> List[dict]:
        g = g.sort_values("date").copy()
        all_days = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(all_days).rename_axis("date").reset_index()
        g["vehicle_id"] = g["vehicle_id"].ffill().bfill()
        g["target"] = g["target"].fillna(0.0)

        y = g["target"].to_numpy(dtype=float)
        d = g["date"].to_numpy()
        out: List[dict] = []

        max_h = max(self.config.horizons)
        hist = self.config.history_days
        if len(g) < hist + max_h + 1:
            return out

        for t in range(hist, len(g) - max_h):
            window = y[t - hist : t]
            stats = {
                "lag_mean_7": float(window[-7:].mean()),
                "lag_mean_14": float(window[-14:].mean()),
                "lag_mean_30": float(window[-30:].mean()),
                "lag_std_7": float(window[-7:].std()),
                "lag_std_30": float(window[-30:].std()),
                "lag_min_14": float(window[-14:].min()),
                "lag_max_14": float(window[-14:].max()),
            }

            base = {
                "vehicle_id": g.loc[t, "vehicle_id"],
                "anchor_date": pd.Timestamp(d[t]),
            }
            for i in range(1, hist + 1):
                base[f"lag_{i}"] = float(window[-i])
            base.update(stats)

            for h in self.config.horizons:
                target_idx = t + h
                target_date = pd.Timestamp(d[target_idx])
                row = base.copy()
                row["horizon"] = h
                row["target_date"] = target_date
                row["pred_dow_sin"] = float(np.sin(2.0 * np.pi * target_date.dayofweek / 7.0))
                row["pred_dow_cos"] = float(np.cos(2.0 * np.pi * target_date.dayofweek / 7.0))
                row["y"] = float(y[target_idx])
                out.append(row)

        return out

    def _build_supervised(self, daily_panel: pd.DataFrame) -> pd.DataFrame:
        rows: List[dict] = []
        for _, g in daily_panel.groupby("vehicle_id", sort=False):
            rows.extend(self._build_examples_for_group(g))
        if not rows:
            raise ValueError(
                "Not enough data to build training examples. "
                "Check history_days and ensure each vehicle has enough daily rows."
            )
        return pd.DataFrame(rows)

    def fit(self, daily_panel: pd.DataFrame, validation_days: int = 30) -> pd.DataFrame:
        data = self._build_supervised(daily_panel)

        all_feature_cols = [
            c
            for c in data.columns
            if c not in {"y", "target_date", "anchor_date", "vehicle_id", "horizon"}
        ]
        # Keep horizon-specific calendar features and lag features as final input schema.
        all_feature_cols = sorted(all_feature_cols, key=lambda x: (not x.startswith("lag_"), x))
        self.feature_columns = all_feature_cols + ["pred_dow_sin", "pred_dow_cos"]
        self.feature_columns = list(dict.fromkeys(self.feature_columns))

        max_date = data["target_date"].max()
        split_date = max_date - pd.Timedelta(days=validation_days)

        metrics: List[dict] = []
        self.models = {}

        for h in self.config.horizons:
            subset = data[data["horizon"] == h].copy()
            train = subset[subset["target_date"] <= split_date]
            valid = subset[subset["target_date"] > split_date]

            if train.empty:
                raise ValueError(f"No training rows available for horizon={h}.")

            model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=-1,
                min_samples_leaf=self.config.min_samples_leaf,
            )

            x_train = train[self.feature_columns]
            y_train = train["y"]
            model.fit(x_train, y_train)
            self.models[h] = model

            if not valid.empty:
                x_valid = valid[self.feature_columns]
                y_valid = valid["y"]
                pred = model.predict(x_valid)
                mae = mean_absolute_error(y_valid, pred)
                rmse = mean_squared_error(y_valid, pred, squared=False)
                metrics.append({"horizon": h, "mae": float(mae), "rmse": float(rmse), "n_valid": int(len(valid))})
            else:
                metrics.append({"horizon": h, "mae": np.nan, "rmse": np.nan, "n_valid": 0})

        return pd.DataFrame(metrics)

    def predict_from_history(
        self,
        history: pd.DataFrame,
        days: Iterable[int],
        date_col: str = "date",
        target_col: str = "target",
    ) -> pd.DataFrame:
        days = sorted(set(int(d) for d in days))
        for d in days:
            if d not in self.models:
                raise ValueError(f"Requested horizon={d} is not available in trained models.")

        h = self.config.history_days
        hist = history[[date_col, target_col]].copy()
        hist[date_col] = pd.to_datetime(hist[date_col])
        hist[target_col] = pd.to_numeric(hist[target_col], errors="coerce")
        hist = hist.dropna().sort_values(date_col)

        if len(hist) < h:
            raise ValueError(f"History must contain at least {h} daily rows.")

        recent = hist.tail(h).copy()
        recent_values = recent[target_col].to_numpy(dtype=float)
        last_date = recent[date_col].max()

        base = {}
        for i in range(1, h + 1):
            base[f"lag_{i}"] = float(recent_values[-i])

        base["lag_mean_7"] = float(recent_values[-7:].mean())
        base["lag_mean_14"] = float(recent_values[-14:].mean())
        base["lag_mean_30"] = float(recent_values[-30:].mean())
        base["lag_std_7"] = float(recent_values[-7:].std())
        base["lag_std_30"] = float(recent_values[-30:].std())
        base["lag_min_14"] = float(recent_values[-14:].min())
        base["lag_max_14"] = float(recent_values[-14:].max())

        rows = []
        for d in days:
            pred_date = pd.Timestamp(last_date) + pd.Timedelta(days=d)
            row = base.copy()
            row["pred_dow_sin"] = float(np.sin(2.0 * np.pi * pred_date.dayofweek / 7.0))
            row["pred_dow_cos"] = float(np.cos(2.0 * np.pi * pred_date.dayofweek / 7.0))
            x = pd.DataFrame([row])[self.feature_columns]
            y_hat = float(self.models[d].predict(x)[0])
            rows.append({"forecast_day": d, "date": pred_date, "prediction": y_hat})

        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        payload = {
            "config": self.config,
            "models": self.models,
            "feature_columns": self.feature_columns,
            "trained_target_column": self.trained_target_column,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "MultiHorizonDailyForecaster":
        payload = joblib.load(path)
        obj = cls(config=payload["config"])
        obj.models = payload["models"]
        obj.feature_columns = payload["feature_columns"]
        obj.trained_target_column = payload.get("trained_target_column", "")
        return obj


def parse_horizons(raw: str) -> Tuple[int, ...]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one horizon must be provided.")
    if any(v < 1 for v in vals):
        raise ValueError("Horizons must be positive integers.")
    return tuple(sorted(set(vals)))


def train_command(args: argparse.Namespace) -> None:
    config = ForecasterConfig(
        history_days=args.history_days,
        horizons=parse_horizons(args.horizons),
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
    )
    model = MultiHorizonDailyForecaster(config)

    data_path = Path(args.data_path)
    if args.data_format == "emobpy":
        raw = model.read_emobpy_csv(data_path)
    else:
        raw = model.read_long_csv(data_path)

    daily = model.to_daily_panel(
        raw,
        vehicle_col=args.vehicle_col,
        date_col=args.date_col,
        target_col=args.target_col,
        agg=args.agg,
    )

    model.trained_target_column = args.target_col
    metrics = model.fit(daily, validation_days=args.validation_days)

    out_model = Path(args.model_out)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_model)

    if args.metrics_out:
        out_metrics = Path(args.metrics_out)
        out_metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out_metrics, index=False)

    print("Training completed.")
    print(f"Rows in daily panel: {len(daily)}")
    print(f"Vehicles: {daily['vehicle_id'].nunique()}")
    print("Validation metrics by horizon:")
    print(metrics.to_string(index=False))
    print(f"Saved model: {out_model}")


def predict_command(args: argparse.Namespace) -> None:
    model = MultiHorizonDailyForecaster.load(Path(args.model_in))
    horizons = parse_horizons(args.horizons)

    hist = pd.read_csv(Path(args.history_path), parse_dates=[args.date_col], low_memory=False)
    pred = model.predict_from_history(
        hist,
        days=horizons,
        date_col=args.date_col,
        target_col=args.target_col,
    )

    out_path = Path(args.pred_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False)

    print("Forecast completed.")
    print(pred.to_string(index=False))
    print(f"Saved forecast: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train and use a multi-horizon daily forecaster for e-mobility time series."
    )
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="Train model on panel time series data.")
    tr.add_argument("--data-path", required=True, help="Path to CSV data.")
    tr.add_argument("--data-format", choices=["long", "emobpy"], default="long")
    tr.add_argument("--vehicle-col", default="vehicle_id")
    tr.add_argument("--date-col", default="date")
    tr.add_argument("--target-col", required=True)
    tr.add_argument("--agg", choices=["sum", "mean"], default="sum")
    tr.add_argument("--history-days", type=int, default=100)
    tr.add_argument("--horizons", default="2,3,4,5,6,7")
    tr.add_argument("--validation-days", type=int, default=30)
    tr.add_argument("--n-estimators", type=int, default=350)
    tr.add_argument("--min-samples-leaf", type=int, default=2)
    tr.add_argument("--random-state", type=int, default=42)
    tr.add_argument("--model-out", required=True)
    tr.add_argument("--metrics-out", default="")
    tr.set_defaults(func=train_command)

    pr = sub.add_parser("predict", help="Run forecast from one-person daily history.")
    pr.add_argument("--model-in", required=True)
    pr.add_argument("--history-path", required=True, help="CSV with at least date,target columns.")
    pr.add_argument("--date-col", default="date")
    pr.add_argument("--target-col", default="target")
    pr.add_argument("--horizons", default="2,3,4,5,6,7")
    pr.add_argument("--pred-out", required=True)
    pr.set_defaults(func=predict_command)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
