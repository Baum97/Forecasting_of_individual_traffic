"""
time_pattern_forecaster.py
──────────────────────────
Erkennt Zeitdaten-Trends aus Fahrzeugnutzungsdaten und sagt voraus,
wann ein Nutzer das Fahrzeug in den nächsten 1-7 Tagen nutzt/parkt.

Unterstützte Eingaben:
  - MAT-Dateien (Real-world EV Dataset: Drive/Charge Folders)
      Signale: Epoch [s] (Unix-Timestamp), Curr [A] (positiv = Fahren)
  - Stündliche CSV mit Binärsignal (in_use 0/1)

Subcommands:
  fit      - Modell trainieren (MAT-Ordner oder CSV)
  predict  - Vorhersage für nächste 1-7 Tage
  report   - Trend-Heatmap + Rolling-Trend-Plot aus gespeichertem Modell
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore")

WEEKDAY_NAMES = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


# ── Konfiguration ────────────────────────────────────────────────────────────────

@dataclass
class TimePatternConfig:
    history_days: int = 100
    random_state: int = 42
    n_estimators: int = 200
    min_samples_leaf: int = 3


# ── Datenladen ───────────────────────────────────────────────────────────────────

def load_mat_sessions(
    drive_folders: List[Path],
    charge_folders: List[Path],
) -> pd.DataFrame:
    """
    Lädt Raw.mat-Dateien aus Drive- und Charge-Ordnern.

    Aus den Drive-Ordnern werden Fahrereignisse (Curr != 0) extrahiert,
    aus den Charge-Ordnern Ladeereignisse.
    Beide werden als in_use=1 behandelt.

    Rückgabe: DataFrame [datetime (stündlich), in_use (0/1)]
    """
    try:
        import scipy.io
    except ImportError:
        raise ImportError("scipy fehlt - installieren mit: pip install scipy")

    records: List[Dict] = []

    def _process(mat_path: Path, event_type: str) -> None:
        if not mat_path.exists():
            print(f"  [WARN] Nicht gefunden: {mat_path}")
            return
        try:
            mat = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        except Exception as exc:
            print(f"  [WARN] Fehler beim Laden von {mat_path}: {exc}")
            return

        # Unix-Timestamps aus Epoch-Feld
        epoch = None
        for key in ("Epoch", "epoch", "TimeEpoch", "time_epoch"):
            if key in mat:
                epoch = np.array(mat[key]).flatten().astype(float)
                break
        if epoch is None or len(epoch) == 0:
            print(f"  [WARN] Kein Epoch-Feld in {mat_path}")
            return

        # Ungültige Timestamps filtern (vor 2000-01-01 oder nach 2035-01-01)
        valid_mask = (epoch > 9.46e8) & (epoch < 2.05e9)
        epoch = epoch[valid_mask]
        if len(epoch) == 0:
            return

        timestamps = pd.to_datetime(epoch, unit="s", utc=True).tz_localize(None)
        for ts in timestamps:
            records.append({"datetime": ts, "event": event_type})

    for folder in drive_folders:
        _process(Path(folder) / "Raw.mat", "driving")
    for folder in charge_folders:
        _process(Path(folder) / "Raw.mat", "charging")

    if not records:
        raise ValueError(
            "Keine Daten geladen - prüfe die Ordner-Pfade und das Epoch-Feld."
        )

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # Auf Stunden-Raster aggregieren: Stunde gilt als in_use=1 wenn mindestens
    # ein Ereignis darin liegt
    df["hour"] = df["datetime"].dt.floor("h")
    hourly = (
        df.groupby("hour")["event"]
        .agg(lambda s: "driving" if "driving" in s.values else "charging")
        .reset_index()
    )
    hourly.columns = ["datetime", "event"]
    hourly["in_use"] = 1
    return hourly[["datetime", "in_use"]]


def load_hourly_csv(
    path: Path,
    date_col: str = "datetime",
    signal_col: str = "in_use",
) -> pd.DataFrame:
    """
    Lädt eine stündliche CSV mit Binärsignal (0/1).
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.rename(columns={date_col: "datetime", signal_col: "in_use"})
    df["in_use"] = pd.to_numeric(df["in_use"], errors="coerce").fillna(0).clip(0, 1)
    df = df.sort_values("datetime")[["datetime", "in_use"]]
    return df


def fill_hourly_grid(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Füllt Lücken im Stundenraster mit 0 (geparkt).
    Gibt vollständiges Raster von erster bis letzter Stunde zurück.
    """
    df = hourly_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("h")
    df = df.drop_duplicates("datetime").set_index("datetime")
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx, fill_value=0)
    df.index.name = "datetime"
    df = df.reset_index()
    if "in_use" not in df.columns:
        df["in_use"] = 0
    return df[["datetime", "in_use"]]


# ── Feature Engineering ──────────────────────────────────────────────────────────

def extract_daily_features(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrahiert pro Tag:
      date, weekday, is_used, first_dep_hour, last_ret_hour, total_hours
    """
    df = hourly_df.copy()
    df["date"] = df["datetime"].dt.normalize()
    df["hour"] = df["datetime"].dt.hour

    rows = []
    for date, grp in df.groupby("date"):
        active = grp.loc[grp["in_use"] == 1, "hour"].values
        is_used = int(len(active) > 0)
        rows.append({
            "date": pd.Timestamp(date),
            "weekday": pd.Timestamp(date).weekday(),
            "is_used": is_used,
            "first_dep_hour": float(active.min()) if is_used else np.nan,
            "last_ret_hour": float(active.max()) if is_used else np.nan,
            "total_hours": float(len(active)),
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_hourly_profile(hourly_df: pd.DataFrame) -> np.ndarray:
    """
    Berechnet 7×24-Matrix: P(in_use | Wochentag, Stunde).
    Zeilen = Wochentage (0=Mo…6=So), Spalten = Stunden (0-23).
    """
    df = hourly_df.copy()
    df["weekday"] = df["datetime"].dt.weekday
    df["hour"] = df["datetime"].dt.hour

    profile = np.zeros((7, 24))
    for wd in range(7):
        for h in range(24):
            subset = df.loc[(df["weekday"] == wd) & (df["hour"] == h), "in_use"]
            if len(subset) > 0:
                profile[wd, h] = float(subset.mean())
    return profile


def _build_feature_row(
    daily_df: pd.DataFrame,
    ref_index: int,
    history_days: int,
    offset: int = 0,
) -> Dict:
    """
    Erstellt einen Feature-Vektor für den Tag an Position `ref_index + offset`.

    ref_index: letzter verfügbarer Tag in daily_df
    offset:    wie viele Tage nach ref_index vorhergesagt wird (1-7)
    """
    target_wd = (daily_df.iloc[ref_index]["weekday"] + offset) % 7
    wd_sin = np.sin(2 * np.pi * target_wd / 7)
    wd_cos = np.cos(2 * np.pi * target_wd / 7)

    # Lags: is_used, dep, ret der letzten 7 Tage (relativ zu ref_index)
    lags: Dict = {}
    for lag in range(1, 8):
        idx = ref_index - lag + 1
        if 0 <= idx < len(daily_df):
            lags[f"used_lag{lag}"] = daily_df.iloc[idx]["is_used"]
            lags[f"dep_lag{lag}"] = daily_df.iloc[idx]["first_dep_hour"]
            lags[f"ret_lag{lag}"] = daily_df.iloc[idx]["last_ret_hour"]
        else:
            lags[f"used_lag{lag}"] = daily_df["is_used"].mean()
            lags[f"dep_lag{lag}"] = daily_df["first_dep_hour"].dropna().mean()
            lags[f"ret_lag{lag}"] = daily_df["last_ret_hour"].dropna().mean()

    hist = daily_df.iloc[max(0, ref_index - history_days + 1): ref_index + 1]
    roll7 = hist.tail(7)["is_used"].mean()
    roll14 = hist.tail(14)["is_used"].mean()
    roll30 = hist.tail(30)["is_used"].mean()

    same_wd = hist[hist["weekday"] == target_wd].tail(4)
    same_wd_usage = same_wd["is_used"].mean() if len(same_wd) > 0 else roll7
    same_wd_dep = same_wd["first_dep_hour"].dropna().mean() if len(same_wd) > 0 else np.nan
    same_wd_ret = same_wd["last_ret_hour"].dropna().mean() if len(same_wd) > 0 else np.nan

    return {
        "wd_sin": wd_sin,
        "wd_cos": wd_cos,
        "roll7_usage": roll7,
        "roll14_usage": roll14,
        "roll30_usage": roll30,
        "same_wd_usage": same_wd_usage,
        "same_wd_dep": same_wd_dep,
        "same_wd_ret": same_wd_ret,
        **lags,
    }


FEATURE_COLS = [
    "wd_sin", "wd_cos",
    "roll7_usage", "roll14_usage", "roll30_usage",
    "same_wd_usage", "same_wd_dep", "same_wd_ret",
    *[f"used_lag{i}" for i in range(1, 8)],
    *[f"dep_lag{i}" for i in range(1, 8)],
    *[f"ret_lag{i}" for i in range(1, 8)],
]


def build_training_matrix(
    daily_df: pd.DataFrame, history_days: int
) -> pd.DataFrame:
    """
    Erstellt Feature-Matrix + Targets für das ML-Training.
    Jede Zeile = ein Vorhersage-Tag (nur Horizont +1).
    """
    rows = []
    n = len(daily_df)
    for i in range(history_days, n):
        feat = _build_feature_row(daily_df, ref_index=i - 1, history_days=history_days, offset=1)
        target = daily_df.iloc[i]
        feat["_is_used"] = target["is_used"]
        feat["_dep"] = target["first_dep_hour"]
        feat["_ret"] = target["last_ret_hour"]
        rows.append(feat)
    return pd.DataFrame(rows)


# ── Forecaster ────────────────────────────────────────────────────────────────────

class TimePatternForecaster:
    """
    Erkennt Zeitdaten-Trends aus historischen Fahrzeugnutzungsdaten (100 Tage)
    und sagt voraus, wann das Fahrzeug in den nächsten 1-7 Tagen genutzt wird.

    Modelle:
      clf_used  - RandomForestClassifier: P(is_used | features)
      reg_dep   - RandomForestRegressor:  first_dep_hour (nur wenn is_used=1)
      reg_ret   - RandomForestRegressor:  last_ret_hour  (nur wenn is_used=1)

    Stündliches Profil:
      hourly_profile - 7×24-Matrix P(in_use | weekday, hour) aus Trainingsdaten
    """

    def __init__(self, config: Optional[TimePatternConfig] = None):
        self.config = config or TimePatternConfig()
        self.clf_used: Optional[RandomForestClassifier] = None
        self.reg_dep: Optional[RandomForestRegressor] = None
        self.reg_ret: Optional[RandomForestRegressor] = None
        self.hourly_profile: Optional[np.ndarray] = None
        self.daily_df: Optional[pd.DataFrame] = None
        self.hourly_df: Optional[pd.DataFrame] = None

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, hourly_df: pd.DataFrame) -> "TimePatternForecaster":
        """
        Trainiert das Modell auf stündlichen Nutzungsdaten.

        hourly_df: DataFrame mit Spalten [datetime, in_use (0/1)]
        """
        hourly_df = fill_hourly_grid(hourly_df)
        self.hourly_df = hourly_df

        daily_df = extract_daily_features(hourly_df)
        self.daily_df = daily_df
        self.hourly_profile = build_hourly_profile(hourly_df)

        feat_df = build_training_matrix(daily_df, self.config.history_days)
        if len(feat_df) < 10:
            raise ValueError(
                f"Zu wenig Trainingsdaten: {len(feat_df)} Samples. "
                f"Mindestens {self.config.history_days + 10} Tage nötig."
            )

        X = feat_df[FEATURE_COLS].fillna(0).values
        y_used = feat_df["_is_used"].astype(int).values
        y_dep = feat_df["_dep"].values
        y_ret = feat_df["_ret"].values

        rf_kwargs = dict(
            n_estimators=self.config.n_estimators,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

        self.clf_used = RandomForestClassifier(**rf_kwargs)
        self.clf_used.fit(X, y_used)

        used_mask = y_used == 1
        if used_mask.sum() >= 5:
            X_used = X[used_mask]
            self.reg_dep = RandomForestRegressor(**rf_kwargs)
            self.reg_dep.fit(X_used, np.where(np.isnan(y_dep[used_mask]), 8.0, y_dep[used_mask]))
            self.reg_ret = RandomForestRegressor(**rf_kwargs)
            self.reg_ret.fit(X_used, np.where(np.isnan(y_ret[used_mask]), 17.0, y_ret[used_mask]))

        usage_rate = daily_df["is_used"].mean()
        print(f"[fit] {len(daily_df)} Tage geladen, {len(feat_df)} Trainings-Samples")
        print(f"[fit] Gesamte Nutzungsrate: {usage_rate:.1%}")
        for wd in range(7):
            wd_rate = daily_df.loc[daily_df["weekday"] == wd, "is_used"].mean()
            print(f"      {WEEKDAY_NAMES[wd]}: {wd_rate:.0%}")
        return self

    # ── Vorhersage ────────────────────────────────────────────────────────────

    def predict(self, n_days: int = 7) -> pd.DataFrame:
        """
        Gibt Forecast-DataFrame für die nächsten n_days zurück (1 ≤ n_days ≤ 7).

        Spalten:
          forecast_day, date, weekday,
          p_used,
          dep_est, dep_p10, dep_p90,
          ret_est, ret_p10, ret_p90,
          hour_profile (Liste mit 24 Werten: P(in_use) je Stunde)
        """
        if self.clf_used is None:
            raise RuntimeError("Modell nicht trainiert - erst fit() aufrufen.")

        n_days = max(1, min(7, n_days))
        last_idx = len(self.daily_df) - 1
        last_date = self.daily_df["date"].iloc[last_idx]
        rows = []

        for offset in range(1, n_days + 1):
            target_date = last_date + pd.Timedelta(days=offset)
            wd = target_date.weekday()

            feat = _build_feature_row(
                self.daily_df, ref_index=last_idx,
                history_days=self.config.history_days, offset=offset
            )
            X = np.array([[feat.get(c, 0.0) if not np.isnan(feat.get(c, 0.0)) else 0.0
                           for c in FEATURE_COLS]])

            p_used = float(self.clf_used.predict_proba(X)[0][1])

            dep_est = dep_p10 = dep_p90 = None
            ret_est = ret_p10 = ret_p90 = None

            if self.reg_dep is not None:
                tree_dep = np.array([t.predict(X)[0] for t in self.reg_dep.estimators_])
                dep_est = float(np.mean(tree_dep))
                dep_p10 = float(np.percentile(tree_dep, 10))
                dep_p90 = float(np.percentile(tree_dep, 90))

            if self.reg_ret is not None:
                tree_ret = np.array([t.predict(X)[0] for t in self.reg_ret.estimators_])
                ret_est = float(np.mean(tree_ret))
                ret_p10 = float(np.percentile(tree_ret, 10))
                ret_p90 = float(np.percentile(tree_ret, 90))

            hour_profile = (
                self.hourly_profile[wd].tolist()
                if self.hourly_profile is not None
                else [p_used] * 24
            )

            rows.append({
                "forecast_day": offset,
                "date": target_date,
                "weekday": WEEKDAY_NAMES[wd],
                "p_used": round(p_used, 3),
                "dep_est": round(dep_est, 1) if dep_est is not None else None,
                "dep_p10": round(dep_p10, 1) if dep_p10 is not None else None,
                "dep_p90": round(dep_p90, 1) if dep_p90 is not None else None,
                "ret_est": round(ret_est, 1) if ret_est is not None else None,
                "ret_p10": round(ret_p10, 1) if ret_p10 is not None else None,
                "ret_p90": round(ret_p90, 1) if ret_p90 is not None else None,
                "hour_profile": hour_profile,
            })

        return pd.DataFrame(rows)

    def print_forecast(self, pred_df: pd.DataFrame) -> None:
        """Gibt die Vorhersage übersichtlich auf der Konsole aus."""
        print("\n" + "=" * 60)
        print("  FAHRZEUGNUTZUNGS-VORHERSAGE")
        print("=" * 60)
        for _, row in pred_df.iterrows():
            p = row["p_used"]
            indicator = "██" if p > 0.7 else ("▒▒" if p > 0.4 else "░░")
            print(f"\n  Tag +{row['forecast_day']}  {row['date'].date()}  ({row['weekday']})")
            print(f"  Nutzungswahrsch.:  {indicator} {p:.0%}")
            if row["dep_est"] is not None:
                d, d10, d90 = row["dep_est"], row["dep_p10"], row["dep_p90"]
                print(f"  Abfahrt (erw.):    {int(d):02d}:00  "
                      f"[P10 {int(d10):02d}:00 - P90 {int(d90):02d}:00]")
            if row["ret_est"] is not None:
                r, r10, r90 = row["ret_est"], row["ret_p10"], row["ret_p90"]
                print(f"  Rückkehr (erw.):   {int(r):02d}:00  "
                      f"[P10 {int(r10):02d}:00 - P90 {int(r90):02d}:00]")
        print("\n" + "=" * 60 + "\n")

    # ── Persistenz ────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        joblib.dump(self, path)
        print(f"[save] Modell gespeichert: {path}")

    @staticmethod
    def load(path: Path) -> "TimePatternForecaster":
        obj = joblib.load(path)
        print(f"[load] Modell geladen: {path}")
        return obj


# ── Plots ─────────────────────────────────────────────────────────────────────────

def plot_forecast(pred_df: pd.DataFrame, output_path: Path) -> None:
    """
    Kombinierter Forecast-Plot (3 Subplots):
      1. P(is_used) Balkendiagramm
      2. Abfahrt / Rückkehr Zeitfenster (P10-P90)
      3. Stündliches Nutzungsprofil als Heatmap
    """
    n = len(pred_df)
    fig, axes = plt.subplots(3, 1, figsize=(max(8, n * 1.6), 11), constrained_layout=True)
    fig.suptitle("Fahrzeugnutzungs-Vorhersage", fontsize=14, fontweight="bold")

    x = np.arange(n)
    xlabels = [f"{r['weekday']}\n{r['date'].date()}" for _, r in pred_df.iterrows()]

    # ── Subplot 1: P(is_used) ─────────────────────────────────────────────────
    ax1 = axes[0]
    colors = [
        "#e74c3c" if p > 0.7 else ("#f39c12" if p > 0.4 else "#2ecc71")
        for p in pred_df["p_used"]
    ]
    bars = ax1.bar(x, pred_df["p_used"], color=colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("P(Nutzung)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=9)
    ax1.set_title("Nutzungswahrscheinlichkeit")
    for bar, val in zip(bars, pred_df["p_used"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.0%}", ha="center", va="bottom", fontsize=9
        )

    # ── Subplot 2: Abfahrt / Rückkehr ─────────────────────────────────────────
    ax2 = axes[1]
    has_times = pred_df["dep_est"].notna().any()
    if has_times:
        for i, (_, row) in enumerate(pred_df.iterrows()):
            if row["dep_est"] is None:
                continue
            ax2.plot([i, i], [row["dep_p10"], row["dep_p90"]],
                     color="#3498db", linewidth=6, alpha=0.35, solid_capstyle="round")
            ax2.plot(i, row["dep_est"], "o", color="#3498db", markersize=9,
                     label="Abfahrt" if i == 0 else "")
            ax2.plot([i, i], [row["ret_p10"], row["ret_p90"]],
                     color="#e67e22", linewidth=6, alpha=0.35, solid_capstyle="round")
            ax2.plot(i, row["ret_est"], "s", color="#e67e22", markersize=9,
                     label="Rückkehr" if i == 0 else "")
        ax2.set_ylim(0, 24)
        ax2.set_yticks(range(0, 25, 2))
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{int(v):02d}:00")
        )
        ax2.set_ylabel("Uhrzeit")
        ax2.set_xticks(x)
        ax2.set_xticklabels(xlabels, fontsize=9)
        ax2.set_title("Erwartete Abfahrt & Rückkehr  (Band = P10-P90)")
        ax2.legend(loc="upper right", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "Keine Zeitdaten verfügbar",
                 ha="center", va="center", transform=ax2.transAxes, color="gray")
        ax2.set_title("Abfahrt / Rückkehr")

    # ── Subplot 3: Stündliches Profil (Heatmap) ───────────────────────────────
    ax3 = axes[2]
    profiles = np.array([r["hour_profile"] for _, r in pred_df.iterrows()])  # n × 24
    im = ax3.imshow(
        profiles.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
        origin="lower", interpolation="nearest"
    )
    ax3.set_yticks(range(0, 24, 2))
    ax3.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(xlabels, fontsize=9)
    ax3.set_ylabel("Stunde")
    ax3.set_title("Stündliches Nutzungsprofil (historisch, wochentagsbedingt)")
    plt.colorbar(im, ax=ax3, label="P(in_use)", fraction=0.046, pad=0.04)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Forecast-Plot gespeichert: {output_path}")


def plot_trend_report(forecaster: TimePatternForecaster, output_path: Path) -> None:
    """
    Trend-Report (3 Subplots):
      1. 7×24 Heatmap: P(in_use | Wochentag, Stunde)
      2. Rolling 7/14-Tage Nutzungsrate über den Beobachtungszeitraum
      3. Rolling Abfahrts-/Rückkehrzeit über den Beobachtungszeitraum
    """
    daily_df = forecaster.daily_df.copy()
    profile = forecaster.hourly_profile

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    fig.suptitle("Trend-Analyse Fahrzeugnutzung", fontsize=14, fontweight="bold")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    ax1 = axes[0]
    im = ax1.imshow(
        profile.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
        origin="lower", interpolation="nearest"
    )
    ax1.set_xticks(range(7))
    ax1.set_xticklabels(WEEKDAY_NAMES, fontsize=10)
    ax1.set_yticks(range(0, 24, 2))
    ax1.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=8)
    ax1.set_xlabel("Wochentag")
    ax1.set_ylabel("Stunde")
    ax1.set_title("Historisches Nutzungsprofil: P(in_use | Wochentag, Stunde)")
    plt.colorbar(im, ax=ax1, label="P(in_use)", fraction=0.046, pad=0.04)

    # ── Rolling Nutzungsrate ──────────────────────────────────────────────────
    ax2 = axes[1]
    daily_df["roll7"] = daily_df["is_used"].rolling(7, min_periods=1).mean()
    daily_df["roll14"] = daily_df["is_used"].rolling(14, min_periods=1).mean()
    ax2.fill_between(daily_df["date"], daily_df["roll7"], alpha=0.25, color="#3498db")
    ax2.plot(daily_df["date"], daily_df["roll7"], label="7-Tage", color="#3498db", linewidth=1.8)
    ax2.plot(daily_df["date"], daily_df["roll14"], label="14-Tage",
             color="#e74c3c", linewidth=1.8, linestyle="--")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Nutzungsrate")
    ax2.set_title("Rolling Nutzungsrate über Beobachtungszeitraum")
    ax2.legend(fontsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.tick_params(axis="x", rotation=30)

    # ── Rolling Abfahrt / Rückkehr ────────────────────────────────────────────
    ax3 = axes[2]
    used = daily_df[daily_df["is_used"] == 1].copy()
    if len(used) >= 3:
        used["roll_dep"] = used["first_dep_hour"].rolling(7, min_periods=1).mean()
        used["roll_ret"] = used["last_ret_hour"].rolling(7, min_periods=1).mean()
        ax3.plot(used["date"], used["roll_dep"], label="Abfahrt (7T-Roll.)",
                 color="#3498db", linewidth=1.8)
        ax3.plot(used["date"], used["roll_ret"], label="Rückkehr (7T-Roll.)",
                 color="#e67e22", linewidth=1.8)
        ax3.fill_between(used["date"], used["roll_dep"], used["roll_ret"],
                         alpha=0.1, color="gray")
        ax3.set_ylim(0, 24)
        ax3.set_yticks(range(0, 25, 2))
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):02d}:00"))
        ax3.set_ylabel("Uhrzeit")
        ax3.set_title("Rolling Abfahrts- & Rückkehrzeit (nur Nutzungstage)")
        ax3.legend(fontsize=9)
    else:
        ax3.text(0.5, 0.5, "Zu wenig Nutzungstage für Trend",
                 ha="center", va="center", transform=ax3.transAxes, color="gray")
        ax3.set_title("Abfahrt / Rückkehr Trend")
    ax3.tick_params(axis="x", rotation=30)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[report] Trend-Report gespeichert: {output_path}")


# ── Multi-Fahrzeug-Unterstützung ─────────────────────────────────────────────────

def load_emobpy_multi(path: Path, target_col: str = "VehicleMobility__Distance_km") -> Dict[str, pd.DataFrame]:
    """
    Lädt die emobpy-Zeitreihendatei (stündlich oder original) und gibt ein
    Dict {vehicle_id -> hourly_df} zurück.

    hourly_df hat Spalten [datetime, in_use] wobei in_use=1 wenn target > 0.
    """
    hdr = pd.read_csv(path, nrows=2, header=None, dtype=str)
    top = hdr.iloc[0].fillna("").str.strip().tolist()
    sub = hdr.iloc[1].fillna("").str.strip().tolist()

    columns: List[str] = []
    for i in range(len(top)):
        if i == 0:
            columns.append("date")
            continue
        t, s = top[i], sub[i]
        if t and s:
            columns.append(f"{t}__{s}")
        elif t:
            columns.append(t)
        elif s:
            columns.append(s)
        else:
            columns.append(f"col_{i}")

    df = pd.read_csv(path, skiprows=3, header=None, names=columns,
                     parse_dates=[0], low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    # Fahrzeug-Spalten = alle ausser 'date' und 'ID__ID'
    vehicle_cols = [c for c in df.columns if c not in ("date", "ID__ID")]

    # Wenn target_col direkt vorhanden: eine Spalte = ein Fahrzeug
    # Wenn nicht: alle numerischen Spalten als separate Fahrzeuge
    if target_col in vehicle_cols:
        candidates = [target_col]
    else:
        # Versuche Distance_km-artige Spalten
        candidates = [c for c in vehicle_cols if "Distance_km" in c or "distance" in c.lower()]
        if not candidates:
            candidates = vehicle_cols[:1]

    results: Dict[str, pd.DataFrame] = {}
    for col in candidates:
        series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        hourly = pd.DataFrame({
            "datetime": df["date"],
            "in_use": (series > 0).astype(int),
        })
        results[col] = hourly

    return results


def load_vehicle_folder(folder: Path, date_col: str = "datetime",
                        signal_col: str = "in_use") -> Dict[str, pd.DataFrame]:
    """
    Lädt alle CSV-Dateien aus einem Ordner als separate Fahrzeuge.
    Dateiname ohne Extension wird als vehicle_id verwendet.
    """
    results: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(folder.glob("*.csv")):
        vehicle_id = csv_path.stem
        try:
            df = load_hourly_csv(csv_path, date_col, signal_col)
            results[vehicle_id] = df
        except Exception as exc:
            print(f"  [WARN] {csv_path.name} übersprungen: {exc}")
    return results


def _evaluate_forecaster(
    forecaster: TimePatternForecaster,
    vehicles: Dict[str, pd.DataFrame],
    split_label: str,
) -> pd.DataFrame:
    """
    Evaluiert ein trainiertes Modell auf einer Menge von Fahrzeugen.
    Gibt je Fahrzeug Accuracy (is_used) und MAE (dep/ret) zurück.
    """
    rows = []
    for vid, hourly_df in vehicles.items():
        try:
            hourly_filled = fill_hourly_grid(hourly_df)
            daily_df = extract_daily_features(hourly_filled)
            if len(daily_df) < forecaster.config.history_days + 5:
                continue

            # Rollende Evaluation: jeder Tag ab history_days als Ziel
            y_true_used, y_pred_used = [], []
            y_true_dep, y_pred_dep = [], []
            y_true_ret, y_pred_ret = [], []

            for i in range(forecaster.config.history_days, len(daily_df)):
                feat = _build_feature_row(daily_df, ref_index=i - 1,
                                          history_days=forecaster.config.history_days,
                                          offset=1)
                X = np.array([[feat.get(c, 0.0) if not (isinstance(feat.get(c), float)
                               and np.isnan(feat.get(c, 0.0))) else 0.0
                               for c in FEATURE_COLS]])

                p_used = float(forecaster.clf_used.predict_proba(X)[0][1])
                pred_used = int(p_used >= 0.5)
                actual = daily_df.iloc[i]

                y_true_used.append(int(actual["is_used"]))
                y_pred_used.append(pred_used)

                if forecaster.reg_dep is not None and actual["is_used"] == 1:
                    p_dep = float(forecaster.reg_dep.predict(X)[0])
                    p_ret = float(forecaster.reg_ret.predict(X)[0])
                    if not np.isnan(actual["first_dep_hour"]):
                        y_true_dep.append(actual["first_dep_hour"])
                        y_pred_dep.append(p_dep)
                    if not np.isnan(actual["last_ret_hour"]):
                        y_true_ret.append(actual["last_ret_hour"])
                        y_pred_ret.append(p_ret)

            n = len(y_true_used)
            if n == 0:
                continue
            accuracy = float(np.mean(np.array(y_true_used) == np.array(y_pred_used)))
            mae_dep = float(np.mean(np.abs(np.array(y_true_dep) - np.array(y_pred_dep)))) if y_true_dep else np.nan
            mae_ret = float(np.mean(np.abs(np.array(y_true_ret) - np.array(y_pred_ret)))) if y_true_ret else np.nan

            rows.append({
                "split": split_label,
                "vehicle_id": vid,
                "n_eval_days": n,
                "accuracy_used": round(accuracy, 4),
                "mae_dep_h": round(mae_dep, 2) if not np.isnan(mae_dep) else None,
                "mae_ret_h": round(mae_ret, 2) if not np.isnan(mae_ret) else None,
            })
        except Exception as exc:
            print(f"  [WARN] Eval fehlgeschlagen für {vid}: {exc}")

    return pd.DataFrame(rows)


def fit_multi(
    all_vehicles: Dict[str, pd.DataFrame],
    n_train: int,
    n_val: int,
    config: Optional[TimePatternConfig] = None,
    seed: int = 42,
) -> Tuple["TimePatternForecaster", pd.DataFrame]:
    """
    Trainiert ein globales Modell über mehrere Fahrzeuge.

    Aufteilung:
      - Fahrzeuge 0..n_train-1          → Training
      - Fahrzeuge n_train..n_train+n_val-1 → Validierung
      - Rest                             → Test

    Rückgabe:
      forecaster  – trainiertes globales Modell
      metrics_df  – Metriken je Fahrzeug und Split
    """
    config = config or TimePatternConfig()
    rng = np.random.RandomState(seed)

    vehicle_ids = list(all_vehicles.keys())
    rng.shuffle(vehicle_ids)

    train_ids = vehicle_ids[:n_train]
    val_ids   = vehicle_ids[n_train: n_train + n_val]
    test_ids  = vehicle_ids[n_train + n_val:]

    print(f"\n[fit_multi] Fahrzeuge gesamt: {len(vehicle_ids)}")
    print(f"  Training:    {len(train_ids)}")
    print(f"  Validierung: {len(val_ids)}")
    print(f"  Test:        {len(test_ids)}")

    # ── Trainings-Matrix über alle Training-Fahrzeuge ─────────────────────────
    all_feat_rows: List[Dict] = []
    pooled_hourly: List[pd.DataFrame] = []

    for vid in train_ids:
        try:
            hourly = fill_hourly_grid(all_vehicles[vid])
            daily = extract_daily_features(hourly)
            if len(daily) < config.history_days + 5:
                print(f"  [SKIP] {vid}: nur {len(daily)} Tage")
                continue
            feat_df = build_training_matrix(daily, config.history_days)
            all_feat_rows.append(feat_df)
            pooled_hourly.append(hourly)
        except Exception as exc:
            print(f"  [WARN] Training-Fahrzeug {vid} fehlgeschlagen: {exc}")

    if not all_feat_rows:
        raise ValueError("Keine Trainingsdaten verfügbar.")

    combined = pd.concat(all_feat_rows, ignore_index=True)
    print(f"\n[fit_multi] Trainings-Samples: {len(combined):,}")

    # ── Stündliches Profil aus allen Training-Fahrzeugen ─────────────────────
    pooled_df = pd.concat(pooled_hourly, ignore_index=True)

    X = combined[FEATURE_COLS].fillna(0).values
    y_used = combined["_is_used"].astype(int).values
    y_dep  = combined["_dep"].values
    y_ret  = combined["_ret"].values

    rf_kwargs = dict(
        n_estimators=config.n_estimators,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        n_jobs=-1,
    )

    forecaster = TimePatternForecaster(config)
    forecaster.clf_used = RandomForestClassifier(**rf_kwargs)
    forecaster.clf_used.fit(X, y_used)
    forecaster.hourly_profile = build_hourly_profile(pooled_df)

    used_mask = y_used == 1
    if used_mask.sum() >= 5:
        X_used = X[used_mask]
        forecaster.reg_dep = RandomForestRegressor(**rf_kwargs)
        forecaster.reg_dep.fit(
            X_used,
            np.where(np.isnan(y_dep[used_mask]), 8.0, y_dep[used_mask])
        )
        forecaster.reg_ret = RandomForestRegressor(**rf_kwargs)
        forecaster.reg_ret.fit(
            X_used,
            np.where(np.isnan(y_ret[used_mask]), 17.0, y_ret[used_mask])
        )

    # Dummy daily_df damit predict() nicht crasht
    first_hourly = fill_hourly_grid(all_vehicles[train_ids[0]])
    forecaster.daily_df = extract_daily_features(first_hourly)
    forecaster.hourly_df = first_hourly

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n[fit_multi] Evaluiere auf Validierungsset ...")
    train_veh = {vid: all_vehicles[vid] for vid in train_ids}
    val_veh   = {vid: all_vehicles[vid] for vid in val_ids}
    test_veh  = {vid: all_vehicles[vid] for vid in test_ids}

    metrics_parts = []
    if train_veh:
        metrics_parts.append(_evaluate_forecaster(forecaster, train_veh, "train"))
    if val_veh:
        metrics_parts.append(_evaluate_forecaster(forecaster, val_veh, "val"))
    if test_veh:
        print("[fit_multi] Evaluiere auf Testset ...")
        metrics_parts.append(_evaluate_forecaster(forecaster, test_veh, "test"))

    metrics_df = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    if not metrics_df.empty:
        print("\n[fit_multi] Ergebnisse je Split:")
        print(f"  {'Split':<12} {'Fahrzeuge':>9} {'Accuracy':>10} {'MAE_dep':>10} {'MAE_ret':>10}")
        print("  " + "-" * 53)
        for split in ["train", "val", "test"]:
            grp = metrics_df[metrics_df["split"] == split]
            if grp.empty:
                continue
            acc = grp["accuracy_used"].mean()
            md  = grp["mae_dep_h"].dropna().mean()
            mr  = grp["mae_ret_h"].dropna().mean()
            print(f"  {split:<12} {len(grp):>9}    {acc:>8.1%}   "
                  f"{'%.2f h' % md if not np.isnan(md) else '  -':>9}   "
                  f"{'%.2f h' % mr if not np.isnan(mr) else '  -':>9}")

    return forecaster, metrics_df


# ── CLI ──────────────────────────────────────────────────────────────────────────

def cmd_fit(args: argparse.Namespace) -> None:
    config = TimePatternConfig(history_days=args.history_days)
    forecaster = TimePatternForecaster(config)

    if args.csv_path:
        print(f"[fit] Lade CSV: {args.csv_path}")
        hourly_df = load_hourly_csv(Path(args.csv_path), args.date_col, args.signal_col)
    else:
        drive_dirs = [Path(p) for p in args.drive_dirs] if args.drive_dirs else []
        charge_dirs = [Path(p) for p in args.charge_dirs] if args.charge_dirs else []
        print(f"[fit] Lade {len(drive_dirs)} Drive- + {len(charge_dirs)} Charge-Ordner")
        hourly_df = load_mat_sessions(drive_dirs, charge_dirs)

    forecaster.fit(hourly_df)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    forecaster.save(model_out)

    if args.report_out:
        plot_trend_report(forecaster, Path(args.report_out))


def cmd_predict(args: argparse.Namespace) -> None:
    forecaster = TimePatternForecaster.load(Path(args.model_in))
    n_days = max(1, min(7, args.horizons))
    pred_df = forecaster.predict(n_days)
    forecaster.print_forecast(pred_df)

    if args.pred_out:
        out = Path(args.pred_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pred_df.drop(columns=["hour_profile"]).to_csv(out, index=False)
        print(f"[predict] CSV gespeichert: {out}")

    if args.plot_out:
        out = Path(args.plot_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_forecast(pred_df, out)


def cmd_report(args: argparse.Namespace) -> None:
    forecaster = TimePatternForecaster.load(Path(args.model_in))
    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_trend_report(forecaster, out)


def cmd_fit_multi(args: argparse.Namespace) -> None:
    config = TimePatternConfig(history_days=args.history_days)

    # Daten laden
    if args.emobpy_path:
        print(f"[fit-multi] Lade emobpy-Datei: {args.emobpy_path}")
        all_vehicles = load_emobpy_multi(
            Path(args.emobpy_path), target_col=args.target_col
        )
    else:
        print(f"[fit-multi] Lade Fahrzeug-CSVs aus Ordner: {args.vehicle_dir}")
        all_vehicles = load_vehicle_folder(
            Path(args.vehicle_dir),
            date_col=args.date_col,
            signal_col=args.signal_col,
        )

    if not all_vehicles:
        print("[ERROR] Keine Fahrzeugdaten geladen.")
        return

    n_total = len(all_vehicles)
    n_train = args.n_train
    n_val   = args.n_val
    n_test  = n_total - n_train - n_val

    if n_test < 0:
        print(f"[ERROR] n_train ({n_train}) + n_val ({n_val}) > verfügbare Fahrzeuge ({n_total})")
        return

    forecaster, metrics_df = fit_multi(
        all_vehicles,
        n_train=n_train,
        n_val=n_val,
        config=config,
        seed=args.seed,
    )

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    forecaster.save(model_out)

    if args.metrics_out:
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out, index=False)
        print(f"[fit-multi] Metriken gespeichert: {out}")

    if args.report_out:
        plot_trend_report(forecaster, Path(args.report_out))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="time_pattern_forecaster.py",
        description="Zeitdaten-Trend-Erkennung & Fahrzeugnutzungs-Vorhersage",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── fit ──────────────────────────────────────────────────────────────────
    p_fit = sub.add_parser("fit", help="Modell aus Daten trainieren")
    src = p_fit.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--csv-path",
        help="Pfad zur stündlichen CSV (Spalten: date + in_use)",
    )
    src.add_argument(
        "--drive-dirs", nargs="+",
        help="Pfade zu Drive/FolderX/-Ordnern (Raw.mat)",
    )
    p_fit.add_argument(
        "--charge-dirs", nargs="+", default=[],
        help="Pfade zu Charge/FolderX/-Ordnern (Raw.mat)",
    )
    p_fit.add_argument("--date-col", default="datetime",
                       help="Spaltenname für Datum/Zeit (CSV-Modus)")
    p_fit.add_argument("--signal-col", default="in_use",
                       help="Spaltenname für Binärsignal (CSV-Modus)")
    p_fit.add_argument("--history-days", type=int, default=100,
                       help="Anzahl historischer Tage für Features (Standard: 100)")
    p_fit.add_argument("--model-out", required=True,
                       help="Ausgabepfad für Modell (.joblib)")
    p_fit.add_argument("--report-out",
                       help="Optional: Trend-Report PNG direkt nach dem Training")

    # ── predict ──────────────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Vorhersage erstellen")
    p_pred.add_argument("--model-in", required=True,
                        help="Pfad zum trainierten Modell (.joblib)")
    p_pred.add_argument("--horizons", type=int, default=7,
                        help="Anzahl Vorhersage-Tage: 1-7 (Standard: 7)")
    p_pred.add_argument("--pred-out",
                        help="Ausgabe-CSV (ohne hour_profile-Spalte)")
    p_pred.add_argument("--plot-out",
                        help="Ausgabe-PNG (kombinierter Forecast-Plot)")

    # ── report ───────────────────────────────────────────────────────────────
    p_rep = sub.add_parser("report", help="Trend-Report aus gespeichertem Modell")
    p_rep.add_argument("--model-in", required=True,
                       help="Pfad zum trainierten Modell (.joblib)")
    p_rep.add_argument("--report-out", required=True,
                       help="Ausgabe-PNG des Trend-Reports")

    # ── fit-multi ─────────────────────────────────────────────────────────────
    p_fm = sub.add_parser(
        "fit-multi",
        help="Globales Modell über viele Fahrzeuge trainieren (Train/Val/Test-Split)",
    )
    src2 = p_fm.add_mutually_exclusive_group(required=True)
    src2.add_argument(
        "--emobpy-path",
        help="Pfad zur emobpy-Zeitreihendatei (stündlich oder original)",
    )
    src2.add_argument(
        "--vehicle-dir",
        help="Ordner mit einer CSV-Datei pro Fahrzeug (Spalten: datetime + in_use)",
    )
    p_fm.add_argument("--target-col", default="VehicleMobility__Distance_km",
                      help="Zielspalte in emobpy-Datei (Standard: VehicleMobility__Distance_km)")
    p_fm.add_argument("--date-col", default="datetime",
                      help="Datumsspalte (CSV-Ordner-Modus)")
    p_fm.add_argument("--signal-col", default="in_use",
                      help="Binärsignal-Spalte (CSV-Ordner-Modus)")
    p_fm.add_argument("--n-train", type=int, required=True,
                      help="Anzahl Fahrzeuge fuer Training (z.B. 50)")
    p_fm.add_argument("--n-val", type=int, required=True,
                      help="Anzahl Fahrzeuge fuer Validierung (z.B. 30)")
    p_fm.add_argument("--history-days", type=int, default=100,
                      help="Historische Tage fuer Features (Standard: 100)")
    p_fm.add_argument("--seed", type=int, default=42,
                      help="Zufallsseed fuer Fahrzeug-Shuffle (Standard: 42)")
    p_fm.add_argument("--model-out", required=True,
                      help="Ausgabepfad fuer Modell (.joblib)")
    p_fm.add_argument("--metrics-out",
                      help="Ausgabe-CSV mit Metriken je Fahrzeug und Split")
    p_fm.add_argument("--report-out",
                      help="Optional: Trend-Report PNG nach dem Training")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    {
        "fit": cmd_fit,
        "predict": cmd_predict,
        "report": cmd_report,
        "fit-multi": cmd_fit_multi,
    }[args.command](args)


if __name__ == "__main__":
    main()
