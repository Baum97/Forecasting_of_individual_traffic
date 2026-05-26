r"""
lgbm_forecaster.py
──────────────────
Forecaster auf Basis von LightGBM. Gleiche Feature-Pipeline und gleiche
Trip-Block-Logik wie der RandomForest-Forecaster, aber:

- Tageswahrscheinlichkeit via LGBMClassifier (Boosted Trees, sequenziell).
- Abfahrts-/Rueckkehrstunde via drei Quantil-LGBMRegressor pro Target
  (alpha=0.1/0.5/0.9) → echte Quantile statt RF-Tree-Variance.

LightGBM wird nur importiert, wenn diese Datei tatsaechlich geladen wird
(Registry-Lazy-Import). Wer den Algo nicht nutzt, braucht es nicht.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError as e:
    raise ImportError(
        "lightgbm ist nicht installiert. Bitte 'pip install lightgbm' ausfuehren."
    ) from e

from model_scripts.forecast.randomforest_forecaster import (
    Config,
    FEATURE_COLS,
    WEEKDAY_NAMES,
    _segment_blocks,
    hourly_profile,
    hourly_profile_multi,
    make_features,
    to_daily,
    to_daily_multi,
    to_matrix,
    to_matrix_multi,
)
from model_scripts.base import (
    FEATURE_COLS as HOURLY_FEATURE_COLS,
    make_features as make_hourly_features,
)


class LGBMForecaster:
    """LightGBM-Variante mit echter Quantile Regression."""

    ALGO_NAME = "lgbm"

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.clf: Optional[LGBMClassifier] = None
        # Fuer jeden Target halten wir drei Modelle: (p10, p50, p90).
        self.reg_dep: Optional[Tuple[LGBMRegressor, LGBMRegressor, LGBMRegressor]] = None
        self.reg_ret: Optional[Tuple[LGBMRegressor, LGBMRegressor, LGBMRegressor]] = None
        self.profile: Optional[np.ndarray] = None
        self.daily: Optional[pd.DataFrame] = None
        self.vehicle_id: Optional[str] = None
        self.per_vehicle_daily: Optional[pd.DataFrame] = None
        # Stuendliches Modell fuer Monte-Carlo-Forecast (Hybrid-Pipeline).
        self.hourly_clf: Optional[LGBMClassifier] = None
        self.hourly_recent: Optional[pd.DataFrame] = None

    @staticmethod
    def _pick_default_vehicle(hourly: pd.DataFrame) -> str:
        stats = (
            hourly.groupby("vehicle_id")
            .agg(active_hours=("in_use", "sum"),
                 last_ts=("datetime", "max"))
            .sort_values(["last_ts", "active_hours"], ascending=[False, False])
        )
        return str(stats.index[0])

    def _lgbm_common(self) -> dict:
        return dict(
            n_estimators=self.cfg.n_estimators,
            min_child_samples=max(self.cfg.min_samples_leaf, 5),
            random_state=self.cfg.random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    def _fit_quantile_triplet(self, X: np.ndarray, y: np.ndarray
                              ) -> Tuple[LGBMRegressor, LGBMRegressor, LGBMRegressor]:
        params = self._lgbm_common()
        models = []
        for alpha in (0.1, 0.5, 0.9):
            m = LGBMRegressor(objective="quantile", alpha=alpha, **params)
            m.fit(X, y)
            models.append(m)
        return tuple(models)  # type: ignore[return-value]

    def fit(self, hourly: pd.DataFrame) -> "LGBMForecaster":
        multi = "vehicle_id" in hourly.columns and hourly["vehicle_id"].nunique() > 1

        if multi:
            self.per_vehicle_daily = to_daily_multi(hourly)
            self.profile = hourly_profile_multi(hourly)
            mat = to_matrix_multi(self.per_vehicle_daily, self.cfg.history_days)
            if len(mat) < 10:
                raise ValueError(
                    f"Zu wenig Trainingsdaten ueber alle Fahrzeuge ({len(mat)})."
                )
            self.vehicle_id = self._pick_default_vehicle(hourly)
            self.daily = (
                self.per_vehicle_daily[self.per_vehicle_daily["vehicle_id"] == self.vehicle_id]
                .drop(columns=["vehicle_id"])
                .sort_values("date")
                .reset_index(drop=True)
            )
            n_vehicles = hourly["vehicle_id"].nunique()
        else:
            single = hourly.drop(columns=["vehicle_id"]) if "vehicle_id" in hourly.columns else hourly
            self.daily = to_daily(single)
            self.profile = hourly_profile(single)
            mat = to_matrix(self.daily, self.cfg.history_days)
            if len(mat) < 10:
                raise ValueError(f"Zu wenig Trainingsdaten ({len(mat)}).")
            n_vehicles = 1

        X = mat[FEATURE_COLS].fillna(0).values
        y_used = mat["y_used"].astype(int).values

        self.clf = LGBMClassifier(**self._lgbm_common()).fit(X, y_used)

        mask = y_used == 1
        if mask.sum() >= 5:
            self.reg_dep = self._fit_quantile_triplet(
                X[mask], np.nan_to_num(mat["y_dep"].values[mask], nan=8.0))
            self.reg_ret = self._fit_quantile_triplet(
                X[mask], np.nan_to_num(mat["y_ret"].values[mask], nan=17.0))

        scope = (f"{n_vehicles} Fahrzeuge, default={self.vehicle_id}"
                 if multi else f"{len(self.daily)} Tage")
        print(f"[fit-lgbm] {scope}, {len(mat)} Samples, "
              f"Nutzungsrate {self.daily['is_used'].mean():.0%}")

        self._fit_hourly(hourly)
        return self

    def _fit_hourly(self, hourly: pd.DataFrame) -> None:
        """LGBM-Variante des stuendlichen Schwesternmodells."""
        df = hourly.rename(columns={"datetime": "timestamp", "in_use": "driving"}).copy()
        if "vehicle_id" not in df.columns:
            df["vehicle_id"] = "single"

        feats = make_hourly_features(df).dropna(subset=HOURLY_FEATURE_COLS)
        if len(feats) < 200:
            print(f"[fit-lgbm-hourly] zu wenig Samples ({len(feats)}) — uebersprungen.")
            return
        X = feats[HOURLY_FEATURE_COLS].to_numpy()
        y = feats["driving"].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            print("[fit-lgbm-hourly] nur eine Klasse — uebersprungen.")
            return

        # class_weight='balanced' korrigiert die Klassen-Imbalance (Aktiv-Rate
        # oft 3-10%) — sonst saturieren die Wahrscheinlichkeiten unter 0.5.
        self.hourly_clf = LGBMClassifier(
            class_weight="balanced",
            **self._lgbm_common(),
        ).fit(X, y)

        target = df[df["vehicle_id"] == (self.vehicle_id or "single")] if self.vehicle_id else df
        target = target.sort_values("timestamp").tail(168)[["timestamp", "driving"]]
        target = target.rename(columns={"timestamp": "datetime", "driving": "in_use"})
        self.hourly_recent = target.reset_index(drop=True)
        print(f"[fit-lgbm-hourly] {len(feats):,} Samples, "
              f"Nutzungsrate {y.mean():.0%}, "
              f"Recent-Window {len(self.hourly_recent)} h.")

    def predict_hourly_mc(self, n_days: int = 7, n_samples: int = 100,
                          mode: str = "mc") -> pd.DataFrame:
        """Identisch zum RF — siehe RandomForestForecaster.predict_hourly_mc."""
        if self.hourly_clf is None or self.hourly_recent is None:
            raise RuntimeError("Hourly-Modell fehlt. Bitte Forecaster neu trainieren.")

        horizon_h = max(1, n_days) * 24
        hist_initial = self.hourly_recent["in_use"].astype(np.float32).to_numpy()
        last_ts = self.hourly_recent["datetime"].max()

        if len(hist_initial) < 168:
            pad = np.zeros(168 - len(hist_initial), dtype=np.float32)
            hist_initial = np.concatenate([pad, hist_initial])
        else:
            hist_initial = hist_initial[-168:]

        effective_samples = 1 if mode == "soft" else n_samples
        hist = np.zeros((effective_samples, 168 + horizon_h), dtype=np.float32)
        hist[:, :168] = hist_initial

        rng = np.random.default_rng(self.cfg.random_state)
        p_matrix = np.zeros((effective_samples, horizon_h), dtype=np.float32)

        time_features = np.zeros((horizon_h, 3), dtype=np.float32)
        for h_offset in range(horizon_h):
            ts = last_ts + pd.Timedelta(hours=h_offset + 1)
            time_features[h_offset] = (ts.hour, ts.weekday(), 1.0 if ts.weekday() >= 5 else 0.0)

        for h_offset in range(horizon_h):
            idx = 168 + h_offset
            lag_1   = hist[:, idx - 1]
            lag_2   = hist[:, idx - 2]
            lag_24  = hist[:, idx - 24]
            lag_168 = hist[:, idx - 168]
            roll_24  = hist[:, idx - 24:idx].mean(axis=1)
            roll_168 = hist[:, idx - 168:idx].mean(axis=1)

            t = time_features[h_offset]
            X = np.column_stack([
                np.full(effective_samples, t[0], dtype=np.float32),
                np.full(effective_samples, t[1], dtype=np.float32),
                np.full(effective_samples, t[2], dtype=np.float32),
                lag_1, lag_2, lag_24, lag_168,
                roll_24, roll_168,
            ])
            probs = self.hourly_clf.predict_proba(X)[:, 1]
            p_matrix[:, h_offset] = probs

            if mode == "soft":
                hist[:, idx] = probs
            else:
                hist[:, idx] = (rng.random(effective_samples) < probs).astype(np.float32)

        timestamps = [last_ts + pd.Timedelta(hours=h + 1) for h in range(horizon_h)]
        p_mean = p_matrix.mean(axis=0)
        if mode == "soft":
            return pd.DataFrame({
                "timestamp": timestamps,
                "p_mean": p_mean,
                "p10": p_mean.copy(),
                "p50": p_mean.copy(),
                "p90": p_mean.copy(),
            })
        return pd.DataFrame({
            "timestamp": timestamps,
            "p_mean": p_mean,
            "p10":    np.quantile(p_matrix, 0.1, axis=0),
            "p50":    np.quantile(p_matrix, 0.5, axis=0),
            "p90":    np.quantile(p_matrix, 0.9, axis=0),
        })

    def predict_proba_day(self, X: np.ndarray) -> float:
        return float(self.clf.predict_proba(X)[0][1])

    def predict_quantiles(self, X: np.ndarray, target: str
                          ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        triple = self.reg_dep if target == "dep" else self.reg_ret
        if triple is None:
            return None, None, None
        low, mid, high = triple
        # Mean wird hier durch den Median (alpha=0.5) ersetzt — bei
        # Quantil-Regression der konsistente Punktschaetzer.
        return (
            float(mid.predict(X)[0]),
            float(low.predict(X)[0]),
            float(high.predict(X)[0]),
        )

    def predict(self, n_days: int = 7, threshold: float = 0.25,
                min_active_p: float = 0.5) -> pd.DataFrame:
        if self.clf is None:
            raise RuntimeError("Erst fit() aufrufen.")
        n_days = max(1, min(7, n_days))
        last_idx = len(self.daily) - 1
        last_date = self.daily["date"].iloc[last_idx]

        wd_base = (
            self.daily.groupby("weekday")["is_used"].mean()
            .reindex(range(7)).fillna(self.daily["is_used"].mean()).values
        )

        rows = []
        for offset in range(1, n_days + 1):
            target_date = last_date + pd.Timedelta(days=offset)
            wd = target_date.weekday()
            feat = make_features(self.daily, last_idx,
                                 self.cfg.history_days, offset)
            X = np.array([[feat[c] if not pd.isna(feat[c]) else 0.0
                           for c in FEATURE_COLS]])

            p_day = self.predict_proba_day(X)
            base = float(wd_base[wd]) if wd_base[wd] > 0 else 1.0
            is_active = p_day >= min_active_p * base

            hourly_p = self.profile[wd] if is_active else np.zeros(24)
            blocks = _segment_blocks(hourly_p >= threshold) if is_active else []

            common = {
                "day": offset,
                "date": target_date.date(),
                "weekday": WEEKDAY_NAMES[wd],
                "p_used": round(p_day, 3),
                "active": int(is_active),
            }
            if not blocks:
                rows.append({**common, "trip": 0,
                             "start_h": None, "end_h": None,
                             "duration_h": None, "peak_p": None})
                continue
            for n, (s, e) in enumerate(blocks, start=1):
                rows.append({
                    **common,
                    "trip": n,
                    "start_h": s,
                    "end_h": e,
                    "duration_h": e - s,
                    "peak_p": round(float(hourly_p[s:e].max()), 3),
                })
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)
        print(f"[save] Modell gespeichert: {path}")

    @staticmethod
    def load(path: Path) -> "LGBMForecaster":
        return joblib.load(path)
