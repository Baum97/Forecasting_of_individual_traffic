from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ForecasterConfig:
    history_days: int = 100
    horizons: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
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

            tree_values = []
            for est in getattr(self.models[d], "estimators_", []):
                tree_values.append(float(est.predict(x)[0]))

            if tree_values:
                tree_preds = np.asarray(tree_values, dtype=float)
                p10 = float(np.percentile(tree_preds, 10))
                p50 = float(np.percentile(tree_preds, 50))
                p90 = float(np.percentile(tree_preds, 90))
                std = float(np.std(tree_preds))
            else:
                p10 = y_hat
                p50 = y_hat
                p90 = y_hat
                std = 0.0

            rows.append(
                {
                    "forecast_day": d,
                    "date": pred_date,
                    "prediction": y_hat,
                    "p10": p10,
                    "p50": p50,
                    "p90": p90,
                    "std": std,
                }
            )

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

    summary = pred.sort_values("forecast_day").head(args.summary_days).copy()
    if summary.empty:
        return

    total_pred_km = float(summary["prediction"].sum())
    total_p10_km = float(summary["p10"].sum())
    total_p90_km = float(summary["p90"].sum())
    print(
        f"Naechste {len(summary)} Tage (gleichbleibendes Fahrverhalten): "
        f"{total_pred_km:.1f} km (Band {total_p10_km:.1f}-{total_p90_km:.1f} km)"
    )

    capacity_kwh = float(args.battery_capacity_kwh)
    if capacity_kwh <= 0 and args.capacity_ah > 0 and args.nominal_voltage > 0:
        capacity_kwh = args.capacity_ah * args.nominal_voltage / 1000.0

    consumption = estimate_consumption_from_history(
        hist,
        target_col=args.target_col,
        energy_col=args.energy_col,
        soc_used_col=args.soc_used_col,
        battery_capacity_kwh=capacity_kwh,
    )

    if consumption["source"] == "none":
        print(
            "Hinweis: Kein Akku-Bedarf berechnet, da keine Verbrauchsspalte gefunden wurde. "
            "Nutze --energy-col oder --soc-used-col."
        )
    else:
        if not np.isnan(consumption["soc_per_km"]):
            soc_need = total_pred_km * consumption["soc_per_km"]
            soc_need_p10 = total_p10_km * consumption["soc_per_km"]
            soc_need_p90 = total_p90_km * consumption["soc_per_km"]
            print(
                f"Geschaetzter Akku-Bedarf naechste {len(summary)} Tage: "
                f"ca. {soc_need:.1f}% (Band {soc_need_p10:.1f}-{soc_need_p90:.1f}%)"
            )

            if 0 <= args.current_soc_percent <= 100:
                soc_left = args.current_soc_percent - soc_need
                print(
                    f"Aktueller SoC: {args.current_soc_percent:.1f}% -> "
                    f"voraussichtlich verbleibend: {soc_left:.1f}%"
                )

        if not np.isnan(consumption["kwh_per_km"]):
            energy_need = total_pred_km * consumption["kwh_per_km"]
            energy_need_p10 = total_p10_km * consumption["kwh_per_km"]
            energy_need_p90 = total_p90_km * consumption["kwh_per_km"]
            print(
                f"Geschaetzter Energiebedarf naechste {len(summary)} Tage: "
                f"{energy_need:.2f} kWh (Band {energy_need_p10:.2f}-{energy_need_p90:.2f} kWh)"
            )

        print(
            f"Gelerntes Verbrauchsprofil ({consumption['source']}): "
            f"{consumption['km_per_unit']:.2f} km pro Einheit"
        )

    if args.plot_out:
        out_plot = Path(args.plot_out)
        out_plot.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8.5, 3.8), constrained_layout=True)
        x = pred["date"]
        ax.plot(x, pred["prediction"], color="#ff8a00", linewidth=2.0, label="Mittel")
        ax.fill_between(x, pred["p10"], pred["p90"], color="#ff8a00", alpha=0.25, label="P10-P90")
        ax.scatter(x, pred["prediction"], color="#ff8a00", s=25)
        ax.set_title("Forecast mit Unsicherheitsband")
        ax.set_xlabel("Datum")
        ax.set_ylabel("Prognose")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(out_plot, dpi=170)
        plt.close(fig)
        print(f"Saved forecast plot: {out_plot}")


def estimate_consumption_from_history(
    hist: pd.DataFrame,
    target_col: str,
    energy_col: str,
    soc_used_col: str,
    battery_capacity_kwh: float,
) -> Dict[str, float | str]:
    work = hist.copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col])
    work = work[work[target_col] > 0]

    if work.empty:
        return {"source": "none", "kwh_per_km": np.nan, "soc_per_km": np.nan, "km_per_unit": np.nan}

    if energy_col and energy_col in work.columns:
        work[energy_col] = pd.to_numeric(work[energy_col], errors="coerce")
        part = work.dropna(subset=[energy_col])
        part = part[part[energy_col] > 0]
        if not part.empty:
            ratio = part[energy_col] / part[target_col]
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            ratio = ratio[ratio > 0]
            if not ratio.empty:
                kwh_per_km = float(np.median(ratio))
                soc_per_km = np.nan
                if battery_capacity_kwh > 0:
                    soc_per_km = float(100.0 * kwh_per_km / battery_capacity_kwh)
                return {
                    "source": f"{energy_col}",
                    "kwh_per_km": kwh_per_km,
                    "soc_per_km": soc_per_km,
                    "km_per_unit": float(1.0 / kwh_per_km),
                }

    if soc_used_col and soc_used_col in work.columns:
        work[soc_used_col] = pd.to_numeric(work[soc_used_col], errors="coerce")
        part = work.dropna(subset=[soc_used_col])
        part = part[part[soc_used_col] > 0]
        if not part.empty:
            ratio = part[soc_used_col] / part[target_col]
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            ratio = ratio[ratio > 0]
            if not ratio.empty:
                soc_per_km = float(np.median(ratio))
                kwh_per_km = np.nan
                if battery_capacity_kwh > 0:
                    kwh_per_km = float((soc_per_km / 100.0) * battery_capacity_kwh)
                return {
                    "source": f"{soc_used_col}",
                    "kwh_per_km": kwh_per_km,
                    "soc_per_km": soc_per_km,
                    "km_per_unit": float(1.0 / soc_per_km),
                }

    return {"source": "none", "kwh_per_km": np.nan, "soc_per_km": np.nan, "km_per_unit": np.nan}


def range_command(args: argparse.Namespace) -> None:
    if args.capacity_ah <= 0:
        raise ValueError("--capacity-ah must be > 0")
    if args.ref_capacity_ah <= 0:
        raise ValueError("--ref-capacity-ah must be > 0")
    if not 0 <= args.soc_percent <= 100:
        raise ValueError("--soc-percent must be between 0 and 100")
    if not 0 <= args.ref_soc_percent <= 100:
        raise ValueError("--ref-soc-percent must be between 0 and 100")
    if args.nominal_voltage <= 0:
        raise ValueError("--nominal-voltage must be > 0")
    if not 0 <= args.reserve_percent < 100:
        raise ValueError("--reserve-percent must be in [0, 100)")
    if args.bad_km <= 0 or args.avg_min_km <= 0 or args.avg_max_km <= 0 or args.good_km <= 0:
        raise ValueError("All range scenario values must be > 0")
    if args.bad_km > args.avg_min_km:
        raise ValueError("--bad-km must be <= --avg-min-km")
    if args.avg_min_km > args.avg_max_km:
        raise ValueError("--avg-min-km must be <= --avg-max-km")
    if args.avg_max_km > args.good_km:
        raise ValueError("--avg-max-km must be <= --good-km")
    if args.samples < 1000:
        raise ValueError("--samples must be at least 1000 for stable statistics")

    usable_fraction = max(0.0, 1.0 - args.reserve_percent / 100.0)
    energy_kwh = (args.soc_percent / 100.0) * args.capacity_ah * args.nominal_voltage / 1000.0
    energy_kwh *= usable_fraction

    ref_energy_kwh = (args.ref_soc_percent / 100.0) * args.ref_capacity_ah * args.nominal_voltage / 1000.0
    ref_energy_kwh *= usable_fraction
    if ref_energy_kwh <= 0:
        raise ValueError("Reference energy must be > 0. Check ref-soc and ref-capacity.")

    scale = energy_kwh / ref_energy_kwh
    bad_km = args.bad_km * scale
    avg_min_km = args.avg_min_km * scale
    avg_max_km = args.avg_max_km * scale
    good_km = args.good_km * scale

    mode = 0.5 * (avg_min_km + avg_max_km)
    rng = np.random.default_rng(args.random_state)
    samples = rng.triangular(left=bad_km, mode=mode, right=good_km, size=args.samples)

    p10, p25, p50, p75, p90 = np.percentile(samples, [10, 25, 50, 75, 90])
    mean_km = float(np.mean(samples))

    print("Range estimation completed.")
    print(f"Akku: {args.soc_percent:.1f}% / {args.capacity_ah:.1f}Ah")
    print(f"Nutzbare Energie: {energy_kwh:.2f} kWh")
    print(f"Distanz (schlechtes Fahrverhalten): {bad_km:.1f} km")
    print(f"Distanz (Durchschnitt): {avg_min_km:.1f}-{avg_max_km:.1f} km")
    print(f"Distanz (gutes Fahrverhalten): {good_km:.1f} km")
    print(f"Erwartungswert: {mean_km:.1f} km")
    print(f"Unsicherheitsbereich P10/P50/P90: {p10:.1f} / {p50:.1f} / {p90:.1f} km")

    out_plot = Path(args.plot_out)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 2.2), constrained_layout=True)
    bp = ax.boxplot(
        samples,
        vert=False,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
    )
    for box in bp["boxes"]:
        box.set(facecolor="#ff8a00", edgecolor="#ff8a00", linewidth=1.5)
    for whisker in bp["whiskers"]:
        whisker.set(color="#ff8a00", linewidth=1.4)
    for cap in bp["caps"]:
        cap.set(color="#ff8a00", linewidth=1.4)
    for median in bp["medians"]:
        median.set(color="#9c4f00", linewidth=1.8)

    ax.axvspan(avg_min_km, avg_max_km, color="#ff8a00", alpha=0.18, label="Durchschnitt")
    ax.scatter([bad_km, good_km], [1, 1], color="#ff8a00", s=30, zorder=3)
    ax.text(bad_km, 1.16, "schlecht", ha="center", va="bottom", fontsize=9)
    ax.text(good_km, 1.16, "gut", ha="center", va="bottom", fontsize=9)

    ax.set_yticks([])
    ax.set_xlabel("Reichweite in km")
    ax.set_title("Reichweitenverteilung")
    ax.grid(axis="x", alpha=0.25)
    ax.set_facecolor("#0f1216")
    fig.patch.set_facecolor("#0f1216")
    ax.tick_params(colors="#e6e6e6")
    ax.xaxis.label.set_color("#e6e6e6")
    ax.title.set_color("#e6e6e6")

    fig.savefig(out_plot, dpi=180)
    plt.close(fig)
    print(f"Saved uncertainty plot: {out_plot}")

    if args.stats_out:
        out_stats = Path(args.stats_out)
        out_stats.parent.mkdir(parents=True, exist_ok=True)
        stats_df = pd.DataFrame(
            [
                {
                    "soc_percent": args.soc_percent,
                    "capacity_ah": args.capacity_ah,
                    "nominal_voltage": args.nominal_voltage,
                    "reserve_percent": args.reserve_percent,
                    "usable_energy_kwh": energy_kwh,
                    "bad_km": bad_km,
                    "avg_min_km": avg_min_km,
                    "avg_max_km": avg_max_km,
                    "good_km": good_km,
                    "p10_km": float(p10),
                    "p25_km": float(p25),
                    "p50_km": float(p50),
                    "mean_km": mean_km,
                    "p75_km": float(p75),
                    "p90_km": float(p90),
                }
            ]
        )
        stats_df.to_csv(out_stats, index=False)
        print(f"Saved stats: {out_stats}")


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
    tr.add_argument("--horizons", default="1,2,3,4,5,6,7")
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
    pr.add_argument("--horizons", default="1,2,3,4,5,6,7")
    pr.add_argument("--summary-days", type=int, default=3, help="How many next days are summarized.")
    pr.add_argument("--energy-col", default="", help="Optional column with used energy per day in kWh.")
    pr.add_argument("--soc-used-col", default="", help="Optional column with used SoC per day in percent points.")
    pr.add_argument("--battery-capacity-kwh", type=float, default=0.0, help="Battery capacity in kWh.")
    pr.add_argument("--capacity-ah", type=float, default=0.0, help="Battery capacity in Ah (alternative to kWh).")
    pr.add_argument("--nominal-voltage", type=float, default=400.0, help="Nominal voltage in V for Ah->kWh conversion.")
    pr.add_argument("--current-soc-percent", type=float, default=-1.0, help="Optional current SoC to estimate remaining SoC.")
    pr.add_argument("--pred-out", required=True)
    pr.add_argument("--plot-out", default="", help="Optional PNG path for uncertainty forecast plot.")
    pr.set_defaults(func=predict_command)

    rr = sub.add_parser("range", help="Estimate EV range from battery state with uncertainty.")
    rr.add_argument("--soc-percent", type=float, required=True, help="Current battery state in percent.")
    rr.add_argument("--capacity-ah", type=float, required=True, help="Current battery capacity in Ah.")
    rr.add_argument("--nominal-voltage", type=float, default=400.0, help="Nominal pack voltage in V.")
    rr.add_argument("--reserve-percent", type=float, default=5.0, help="Reserved SoC that should not be used.")
    rr.add_argument("--bad-km", type=float, default=20.0, help="Bad-driving range at reference battery state.")
    rr.add_argument("--avg-min-km", type=float, default=35.0, help="Average-driving min range at reference battery state.")
    rr.add_argument("--avg-max-km", type=float, default=50.0, help="Average-driving max range at reference battery state.")
    rr.add_argument("--good-km", type=float, default=70.0, help="Good-driving range at reference battery state.")
    rr.add_argument("--ref-soc-percent", type=float, default=30.0, help="Reference SoC used for scenario calibration.")
    rr.add_argument("--ref-capacity-ah", type=float, default=33.0, help="Reference Ah used for scenario calibration.")
    rr.add_argument("--samples", type=int, default=20000, help="Number of Monte Carlo samples.")
    rr.add_argument("--random-state", type=int, default=42)
    rr.add_argument("--plot-out", required=True, help="Path to PNG output for uncertainty plot.")
    rr.add_argument("--stats-out", default="", help="Optional path to CSV statistics output.")
    rr.set_defaults(func=range_command)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
