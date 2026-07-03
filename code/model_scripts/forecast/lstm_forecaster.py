r"""
lstm_forecaster.py
──────────────────
Forecaster auf Basis eines PyTorch-LSTM. Gleiche Feature-Pipeline und gleiche
Trip-Block-Logik wie der RandomForest- und der LGBM-Forecaster, aber:

- Tageswahrscheinlichkeit via LSTM-Classifier. Die 7 Tages-Lags
  (used/dep/ret) werden als Sequenz (7 Zeitschritte x 3 Kanaele) gefuettert,
  die uebrigen Kalender-/Rolling-Features als statischer Kontext an den Kopf.
- Abfahrts-/Rueckkehrstunde via LSTM-Quantilsregression (Pinball-Loss,
  alpha=0.1/0.5/0.9) → echte Quantile, konsistent zur LGBM-Variante.

torch wird nur importiert, wenn diese Datei tatsaechlich geladen wird
(Registry-Lazy-Import). Wer den Algo nicht nutzt, braucht es nicht. Die
Netze sind normale nn.Module und damit joblib-/pickle-fest, sodass save()/
load() identisch zu den anderen Forecastern bleiben.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as e:
    raise ImportError(
        "torch ist nicht installiert. Bitte 'pip install torch' ausfuehren."
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

# ── LSTM-Hyperparameter ──────────────────────────────────────────────────────
# Bewusst kleine Kapazitaet: die Datensaetze sind pro Fahrzeug kurz und die
# Aktiv-Rate niedrig — groessere Netze ueberpassen sofort.
_HIDDEN = 32
_EPOCHS = 40
_LR = 1e-2
_BATCH = 256
_DROPOUT = 0.1
_QUANTILES = (0.1, 0.5, 0.9)


# ── Netz + Trainings-Bausteine ───────────────────────────────────────────────

class _SeqNet(nn.Module):
    """LSTM ueber eine kurze Sequenz + MLP-Kopf mit statischem Kontext."""

    def __init__(self, n_channels: int, n_static: int,
                 hidden: int = _HIDDEN, n_out: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, hidden),
            nn.ReLU(),
            nn.Dropout(_DROPOUT),
            nn.Linear(hidden, n_out),
        )

    def forward(self, seq: "torch.Tensor", static: "torch.Tensor") -> "torch.Tensor":
        _, (h_n, _) = self.lstm(seq)
        z = torch.cat([h_n[-1], static], dim=1)
        return self.head(z)


def _pinball_loss(pred: "torch.Tensor", target: "torch.Tensor",
                  quantiles: "torch.Tensor") -> "torch.Tensor":
    """Mittlerer Pinball-Loss ueber alle Quantile (pred: (B, Q), target: (B,))."""
    err = target.unsqueeze(1) - pred
    return torch.maximum(quantiles * err, (quantiles - 1) * err).mean()


def _fit_scaler(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mittelwert/Std ueber die letzte Achse; Nullstd → 1 (keine Division/0)."""
    flat = a.reshape(-1, a.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


class _ScaledSeqModel:
    """Picklebares Bundle: LSTM-Netz + Feature-/Target-Scaler + fit/predict.

    task='clf'  → 1 Logit, BCE mit pos_weight (Klassen-Imbalance).
    task='reg'  → len(quantiles) Ausgaben, Pinball-Loss, Ziel standardisiert.
    """

    def __init__(self, n_channels: int, n_static: int, task: str,
                 quantiles: Tuple[float, ...] = _QUANTILES, seed: int = 42):
        self.task = task
        self.quantiles = quantiles
        self.seed = seed
        n_out = 1 if task == "clf" else len(quantiles)
        torch.manual_seed(seed)
        self.net = _SeqNet(n_channels, n_static, n_out=n_out)
        self.seq_mean = self.seq_std = None
        self.stat_mean = self.stat_std = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def _prep(self, seq: np.ndarray, static: np.ndarray
              ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        seq = (seq - self.seq_mean) / self.seq_std
        static = (static - self.stat_mean) / self.stat_std
        return (torch.from_numpy(seq.astype(np.float32)),
                torch.from_numpy(static.astype(np.float32)))

    def fit(self, seq: np.ndarray, static: np.ndarray, y: np.ndarray,
            *, pos_weight: Optional[float] = None) -> "_ScaledSeqModel":
        self.seq_mean, self.seq_std = _fit_scaler(seq)
        self.stat_mean, self.stat_std = _fit_scaler(static)
        seq_t, stat_t = self._prep(seq, static)

        if self.task == "reg":
            self.y_mean = float(y.mean())
            self.y_std = float(y.std()) or 1.0
            y_use = (y - self.y_mean) / self.y_std
        else:
            y_use = y
        y_t = torch.from_numpy(y_use.astype(np.float32))

        torch.manual_seed(self.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=_LR)
        if self.task == "clf":
            pw = torch.tensor(pos_weight, dtype=torch.float32) if pos_weight else None
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            q_t = torch.tensor(self.quantiles, dtype=torch.float32)

        n = len(y_t)
        self.net.train()
        for _ in range(_EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, _BATCH):
                idx = perm[i:i + _BATCH]
                opt.zero_grad()
                out = self.net(seq_t[idx], stat_t[idx])
                if self.task == "clf":
                    loss = loss_fn(out.squeeze(1), y_t[idx])
                else:
                    loss = _pinball_loss(out, y_t[idx], q_t)
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    def predict_proba(self, seq: np.ndarray, static: np.ndarray) -> np.ndarray:
        """P(Klasse 1) je Zeile (nur task='clf')."""
        seq_t, stat_t = self._prep(seq, static)
        with torch.no_grad():
            logit = self.net(seq_t, stat_t).squeeze(1)
            return torch.sigmoid(logit).numpy()

    def predict_quantiles(self, seq: np.ndarray, static: np.ndarray) -> np.ndarray:
        """(n, Q) Quantile in Original-Einheit, je Zeile aufsteigend sortiert."""
        seq_t, stat_t = self._prep(seq, static)
        with torch.no_grad():
            out = self.net(seq_t, stat_t).numpy()
        out = out * self.y_std + self.y_mean
        return np.sort(out, axis=1)


# ── Feature-Umformung (flach → Sequenz) ──────────────────────────────────────

def _split_daily(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Zerlegt die flache FEATURE_COLS-Matrix in (seq, static).

    Spaltenlayout (siehe randomforest_forecaster.FEATURE_COLS):
        [0:8]   statische Features (wd_sin/cos, roll*, same_wd_*)
        [8:15]  used_lag1..7   [15:22] dep_lag1..7   [22:29] ret_lag1..7
    Die drei Lag-Bloecke werden zu einer Sequenz (7 Schritte x 3 Kanaele)
    gestapelt, chronologisch (lag7 = aeltester Schritt zuerst).
    """
    X = np.asarray(X, dtype=np.float32)
    static = X[:, :8]
    used = X[:, 8:15][:, ::-1]
    dep = X[:, 15:22][:, ::-1]
    ret = X[:, 22:29][:, ::-1]
    seq = np.stack([used, dep, ret], axis=2)  # (n, 7, 3)
    return np.ascontiguousarray(seq), np.ascontiguousarray(static)


# ── Stuendliches LSTM-Schwesternmodell (Monte-Carlo-Rollout) ─────────────────

class _HourlyLSTM:
    """LSTM-Pendant zum stuendlichen RF-/LGBM-Classifier.

    Erwartet Feature-Matrizen in HOURLY_FEATURE_COLS-Reihenfolge:
        [hour, weekday, is_weekend, lag_1, lag_2, lag_24, lag_168,
         rolling_mean_24, rolling_mean_168]
    Die vier Lags bilden die Sequenz (chronologisch lag_168→lag_1), der Rest
    ist statischer Kontext. predict_proba() liefert (n, 2) wie sklearn, damit
    predict_hourly_mc unveraendert bleibt.
    """

    def __init__(self, seed: int = 42):
        self.model = _ScaledSeqModel(n_channels=1, n_static=5, task="clf", seed=seed)

    @staticmethod
    def _split(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        # Sequenz aus [lag_168, lag_24, lag_2, lag_1] (aeltester zuerst).
        seq = X[:, [6, 5, 4, 3]][:, :, None]  # (n, 4, 1)
        static = X[:, [0, 1, 2, 7, 8]]        # hour, wd, weekend, roll24, roll168
        return np.ascontiguousarray(seq), np.ascontiguousarray(static)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_HourlyLSTM":
        seq, static = self._split(X)
        pos = float(y.sum())
        pos_weight = (len(y) - pos) / pos if pos > 0 else 1.0
        self.model.fit(seq, static, y.astype(np.float32), pos_weight=pos_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        seq, static = self._split(X)
        p1 = self.model.predict_proba(seq, static)
        return np.column_stack([1.0 - p1, p1])


# ── Forecaster (LSTM-Familie) ────────────────────────────────────────────────

class LSTMForecaster:
    """LSTM-Variante mit Quantilsregression fuer Abfahrt/Rueckkehr."""

    ALGO_NAME = "lstm"

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.clf: Optional[_ScaledSeqModel] = None
        self.reg_dep: Optional[_ScaledSeqModel] = None
        self.reg_ret: Optional[_ScaledSeqModel] = None
        self.profile: Optional[np.ndarray] = None
        self.daily: Optional[pd.DataFrame] = None
        self.vehicle_id: Optional[str] = None
        self.per_vehicle_daily: Optional[pd.DataFrame] = None
        # Stuendliches Modell fuer Monte-Carlo-Forecast (Hybrid-Pipeline).
        self.hourly_clf: Optional[_HourlyLSTM] = None
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

    def fit(self, hourly: pd.DataFrame) -> "LSTMForecaster":
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
        seq, static = _split_daily(X)
        y_used = mat["y_used"].astype(int).values

        pos = float(y_used.sum())
        pos_weight = (len(y_used) - pos) / pos if pos > 0 else 1.0
        self.clf = _ScaledSeqModel(
            n_channels=3, n_static=8, task="clf", seed=self.cfg.random_state
        ).fit(seq, static, y_used.astype(np.float32), pos_weight=pos_weight)

        mask = y_used == 1
        if mask.sum() >= 5:
            self.reg_dep = _ScaledSeqModel(
                n_channels=3, n_static=8, task="reg", seed=self.cfg.random_state
            ).fit(seq[mask], static[mask],
                  np.nan_to_num(mat["y_dep"].values[mask], nan=8.0).astype(np.float32))
            self.reg_ret = _ScaledSeqModel(
                n_channels=3, n_static=8, task="reg", seed=self.cfg.random_state
            ).fit(seq[mask], static[mask],
                  np.nan_to_num(mat["y_ret"].values[mask], nan=17.0).astype(np.float32))

        scope = (f"{n_vehicles} Fahrzeuge, default={self.vehicle_id}"
                 if multi else f"{len(self.daily)} Tage")
        print(f"[fit-lstm] {scope}, {len(mat)} Samples, "
              f"Nutzungsrate {self.daily['is_used'].mean():.0%}")

        self._fit_hourly(hourly)
        return self

    def _fit_hourly(self, hourly: pd.DataFrame) -> None:
        """LSTM-Variante des stuendlichen Schwesternmodells."""
        df = hourly.rename(columns={"datetime": "timestamp", "in_use": "driving"}).copy()
        if "vehicle_id" not in df.columns:
            df["vehicle_id"] = "single"

        feats = make_hourly_features(df).dropna(subset=HOURLY_FEATURE_COLS)
        if len(feats) < 200:
            print(f"[fit-lstm-hourly] zu wenig Samples ({len(feats)}) — uebersprungen.")
            return
        X = feats[HOURLY_FEATURE_COLS].to_numpy()
        y = feats["driving"].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            print("[fit-lstm-hourly] nur eine Klasse — uebersprungen.")
            return

        # pos_weight korrigiert die Klassen-Imbalance (Aktiv-Rate oft 3-10%) —
        # sonst saturieren die Wahrscheinlichkeiten unter 0.5.
        self.hourly_clf = _HourlyLSTM(seed=self.cfg.random_state).fit(X, y)

        target = df[df["vehicle_id"] == (self.vehicle_id or "single")] if self.vehicle_id else df
        target = target.sort_values("timestamp").tail(168)[["timestamp", "driving"]]
        target = target.rename(columns={"timestamp": "datetime", "driving": "in_use"})
        self.hourly_recent = target.reset_index(drop=True)
        print(f"[fit-lstm-hourly] {len(feats):,} Samples, "
              f"Nutzungsrate {y.mean():.0%}, "
              f"Recent-Window {len(self.hourly_recent)} h.")

    def predict_hourly_mc(self, n_days: int = 7, n_samples: int = 100,
                          mode: str = "mc") -> pd.DataFrame:
        """Identisch zum RF/LGBM — siehe RandomForestForecaster.predict_hourly_mc."""
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
        seq, static = _split_daily(X)
        return float(self.clf.predict_proba(seq, static)[0])

    def predict_quantiles(self, X: np.ndarray, target: str
                          ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        reg = self.reg_dep if target == "dep" else self.reg_ret
        if reg is None:
            return None, None, None
        seq, static = _split_daily(X)
        low, mid, high = reg.predict_quantiles(seq, static)[0]
        # Median (alpha=0.5) als Punktschaetzer, konsistent zur LGBM-Variante.
        return float(mid), float(low), float(high)

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
    def load(path: Path) -> "LSTMForecaster":
        return joblib.load(path)
