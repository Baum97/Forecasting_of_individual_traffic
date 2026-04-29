r"""
csv_forecaster.py
─────────────────
trainiert aus einer stündlichen CSV ein Modell,
das fuer die naechsten 1-7 Tage Nutzung, Abfahrts- und Rueckkehrzeit
vorhersagt.

CSV-Format (Mindestanforderung):
    datetime,in_use
    2024-01-01 00:00:00,0
    2024-01-01 01:00:00,1
    ...

Subcommands:
    fit      Modell trainieren und als .joblib speichern
        py code/csv_forecaster.py fit --csv-path data/cars-real-world-electric/car_full_timeline.csv --model-out models/car_full_forecaster.joblib


    predict  Vorhersage fuer 1-7 Tage, optional CSV-Export
        py code/csv_forecaster.py predict --model-in models/car_full_forecaster.joblib --horizons 4 --pred-out predictions/car_full_next7.csv

"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

WEEKDAY_NAMES = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


@dataclass
class Config:
    history_days: int = 100
    n_estimators: int = 200
    min_samples_leaf: int = 3
    random_state: int = 42


# ── Datenaufbereitung ────────────────────────────────────────────────────────

def load_csv(path: Path, date_col: str = "datetime",
             signal_col: str = "in_use") -> pd.DataFrame:
    """Laedt stuendliche CSV und fuellt Luecken mit 0 (geparkt)."""
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.rename(columns={date_col: "datetime", signal_col: "in_use"})
    df = df[["datetime", "in_use"]]
    df["in_use"] = pd.to_numeric(df["in_use"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    df["datetime"] = df["datetime"].dt.floor("h")
    df = df.drop_duplicates("datetime").set_index("datetime")

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx, fill_value=0).reset_index()
    df.columns = ["datetime", "in_use"]
    return df


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregiert stuendliche Daten auf Tagesebene."""
    h = hourly.copy()
    h["date"] = h["datetime"].dt.normalize()
    h["hour"] = h["datetime"].dt.hour

    rows = []
    for date, grp in h.groupby("date"):
        active = grp.loc[grp["in_use"] == 1, "hour"].values
        used = len(active) > 0
        rows.append({
            "date": pd.Timestamp(date),
            "weekday": pd.Timestamp(date).weekday(),
            "is_used": int(used),
            "first_dep": float(active.min()) if used else np.nan,
            "last_ret": float(active.max()) if used else np.nan,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _segment_blocks(active: np.ndarray) -> List[tuple]:
    """Liefert (start, end)-Paare zusammenhaengender True-Stunden, end exklusiv."""
    blocks: List[tuple] = []
    i, n = 0, len(active)
    while i < n:
        if active[i]:
            j = i + 1
            while j < n and active[j]:
                j += 1
            blocks.append((i, j))
            i = j
        else:
            i += 1
    return blocks


def hourly_profile(hourly: pd.DataFrame) -> np.ndarray:
    """7x24 Matrix: P(in_use | Wochentag, Stunde)."""
    h = hourly.copy()
    h["wd"] = h["datetime"].dt.weekday
    h["hr"] = h["datetime"].dt.hour
    return (
        h.groupby(["wd", "hr"])["in_use"].mean()
        .unstack(fill_value=0).reindex(range(7)).values
    )


# ── Features ────────────────────────────────────────────────────────────────

def make_features(daily: pd.DataFrame, ref_idx: int,
                  history_days: int, offset: int) -> Dict[str, float]:
    """Feature-Vektor fuer den Tag an Position `ref_idx + offset`."""
    target_wd = (daily.iloc[ref_idx]["weekday"] + offset) % 7
    hist = daily.iloc[max(0, ref_idx - history_days + 1): ref_idx + 1]

    feat: Dict[str, float] = {
        "wd_sin": np.sin(2 * np.pi * target_wd / 7),
        "wd_cos": np.cos(2 * np.pi * target_wd / 7),
        "roll7":  hist.tail(7)["is_used"].mean(),
        "roll14": hist.tail(14)["is_used"].mean(),
        "roll30": hist.tail(30)["is_used"].mean(),
    }

    same_wd = hist[hist["weekday"] == target_wd].tail(4)
    feat["same_wd_usage"] = same_wd["is_used"].mean() if len(same_wd) else feat["roll7"]
    feat["same_wd_dep"]   = same_wd["first_dep"].dropna().mean()
    feat["same_wd_ret"]   = same_wd["last_ret"].dropna().mean()

    for lag in range(1, 8):
        i = ref_idx - lag + 1
        if 0 <= i < len(daily):
            feat[f"used_lag{lag}"] = daily.iloc[i]["is_used"]
            feat[f"dep_lag{lag}"]  = daily.iloc[i]["first_dep"]
            feat[f"ret_lag{lag}"]  = daily.iloc[i]["last_ret"]
        else:
            feat[f"used_lag{lag}"] = daily["is_used"].mean()
            feat[f"dep_lag{lag}"]  = daily["first_dep"].mean()
            feat[f"ret_lag{lag}"]  = daily["last_ret"].mean()

    return feat


FEATURE_COLS = (
    ["wd_sin", "wd_cos", "roll7", "roll14", "roll30",
     "same_wd_usage", "same_wd_dep", "same_wd_ret"]
    + [f"used_lag{i}" for i in range(1, 8)]
    + [f"dep_lag{i}"  for i in range(1, 8)]
    + [f"ret_lag{i}"  for i in range(1, 8)]
)


def to_matrix(daily: pd.DataFrame, history_days: int) -> pd.DataFrame:
    """Trainings-Matrix: eine Zeile pro Zieltag (Horizont +1)."""
    rows = []
    for i in range(history_days, len(daily)):
        feat = make_features(daily, ref_idx=i - 1,
                             history_days=history_days, offset=1)
        target = daily.iloc[i]
        feat["y_used"] = target["is_used"]
        feat["y_dep"]  = target["first_dep"]
        feat["y_ret"]  = target["last_ret"]
        rows.append(feat)
    return pd.DataFrame(rows)


# ── Forecaster ───────────────────────────────────────────────────────────────

class Forecaster:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.clf: Optional[RandomForestClassifier] = None
        self.reg_dep: Optional[RandomForestRegressor] = None
        self.reg_ret: Optional[RandomForestRegressor] = None
        self.profile: Optional[np.ndarray] = None
        self.daily: Optional[pd.DataFrame] = None

    def fit(self, hourly: pd.DataFrame) -> "Forecaster":
        self.daily = to_daily(hourly)
        self.profile = hourly_profile(hourly)

        mat = to_matrix(self.daily, self.cfg.history_days)
        if len(mat) < 10:
            raise ValueError(
                f"Zu wenig Trainingsdaten ({len(mat)}). "
                f"Mindestens {self.cfg.history_days + 10} Tage noetig."
            )

        X = mat[FEATURE_COLS].fillna(0).values
        y_used = mat["y_used"].astype(int).values
        rf = dict(n_estimators=self.cfg.n_estimators,
                  min_samples_leaf=self.cfg.min_samples_leaf,
                  random_state=self.cfg.random_state, n_jobs=-1)

        self.clf = RandomForestClassifier(**rf).fit(X, y_used)

        mask = y_used == 1
        if mask.sum() >= 5:
            self.reg_dep = RandomForestRegressor(**rf).fit(
                X[mask], np.nan_to_num(mat["y_dep"].values[mask], nan=8.0))
            self.reg_ret = RandomForestRegressor(**rf).fit(
                X[mask], np.nan_to_num(mat["y_ret"].values[mask], nan=17.0))

        print(f"[fit] {len(self.daily)} Tage, {len(mat)} Samples, "
              f"Nutzungsrate {self.daily['is_used'].mean():.0%}")
        return self

    def predict(self, n_days: int = 7, threshold: float = 0.25,
                min_active_p: float = 0.5) -> pd.DataFrame:
        """
        Pro Forecast-Tag werden mehrere Trip-Bloecke ausgegeben.

        Zwei-stufiges Vorgehen:
        1. RF-Classifier liefert eine relative Tageswahrscheinlichkeit
           p_day. Sie wird durch die Wochentags-Basisrate normiert -> ein
           Tag gilt als "aktiv", wenn p_day >= min_active_p * Basisrate.
        2. Auf aktiven Tagen wird das empirische 7x24-Stundenprofil gegen
           threshold geprueft; zusammenhaengende Stunden bilden je eine
           Fahrt.
        """
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

            p_day = float(self.clf.predict_proba(X)[0][1])
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
    def load(path: Path) -> "Forecaster":
        return joblib.load(path)


# ── CLI ──────────────────────────────────────────────────────────────────────

def cmd_fit(args: argparse.Namespace) -> None:
    hourly = load_csv(Path(args.csv_path), args.date_col, args.signal_col)
    model = Forecaster(Config(history_days=args.history_days)).fit(hourly)
    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)


def cmd_predict(args: argparse.Namespace) -> None:
    model = Forecaster.load(Path(args.model_in))
    pred = model.predict(args.horizons, threshold=args.threshold,
                         min_active_p=args.min_active_p)
    print(pred.to_string(index=False))
    if args.pred_out:
        out = Path(args.pred_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(out, index=False)
        print(f"[predict] CSV gespeichert: {out}")


def main() -> None:
    p = argparse.ArgumentParser(prog="csv_forecaster.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit", help="Modell aus CSV trainieren")
    pf.add_argument("--csv-path", required=True)
    pf.add_argument("--model-out", required=True)
    pf.add_argument("--date-col", default="datetime")
    pf.add_argument("--signal-col", default="in_use")
    pf.add_argument("--history-days", type=int, default=100)
    pf.set_defaults(func=cmd_fit)

    pp = sub.add_parser("predict", help="1-7 Tage vorhersagen")
    pp.add_argument("--model-in", required=True)
    pp.add_argument("--horizons", type=int, default=7)
    pp.add_argument("--threshold", type=float, default=0.25,
                    help="Stunden mit P(in_use) >= threshold zaehlen als Fahrt")
    pp.add_argument("--min-active-p", type=float, default=0.5,
                    help="Tag aktiv, wenn p_day >= min_active_p * Basisrate")
    pp.add_argument("--pred-out")
    pp.set_defaults(func=cmd_predict)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
