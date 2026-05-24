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

from model_scripts.randomforest_forecaster import (
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
        return self

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
