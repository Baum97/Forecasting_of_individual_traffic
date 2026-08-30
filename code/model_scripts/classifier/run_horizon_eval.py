"""
Prognosegüte über den Vorhersagehorizont (1 h bis 168 h).

Die Kreuzvalidierung in `run_cross_validation.py` bewertet jede Teststunde mit
der *beobachteten* Vergangenheit in den Lag-Merkmalen. Das ist eine Prognose
eine Stunde voraus. Dieses Skript misst stattdessen den mehrtägigen Horizont:
Ab einem Startzeitpunkt (Origin) im Testteil werden 168 Stunden autoregressiv
fortgeschrieben, wobei die Lag- und Rolling-Merkmale ausschliesslich aus den
selbst erzeugten Werten stammen. Die Unsicherheit entsteht ueber einen
Monte-Carlo-Rollout mit mehreren Pfaden; als Prognosewahrscheinlichkeit dient
der Pfadmittelwert.

Verglichen wird gegen zwei Referenzen auf denselben Zeitpunkten:
  * Klimatologie  P(fahren | Wochentag, Stunde) aus dem Trainingsteil
  * Persistenz-168 (gleiche Stunde der Vorwoche; fuer h <= 168 stets beobachtet)

Aus dem Projekt-Root:
    python code/model_scripts/classifier/run_horizon_eval.py
    python code/model_scripts/classifier/run_horizon_eval.py --algos rf lgbm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_scripts.base import (  # noqa: E402
    FEATURE_COLS,
    TrainConfig,
    make_features,
    split_by_time,
    train_model,
)
from model_scripts.data_adapters import (  # noqa: E402
    load_emobpy,
    load_real_world_ev,
    load_routine,
    load_ved,
    load_yjmob,
)
from sklearn.metrics import average_precision_score  # noqa: E402

PRED_DIR = Path(__file__).resolve().parents[3] / "predictions"
IMG_DIR = Path(__file__).resolve().parents[3] / "thesis" / "images"

HORIZON = 168          # maximaler Horizont in Stunden (sieben Tage)
LOOKBACK = 168         # benoetigte beobachtete Vorgeschichte je Origin
SOURCE_ORDER = ["emobpy", "real_world_ev", "ved", "routine", "yjmob"]
MIN_ORIGINS = 3          # weniger vollstaendige Fenster gelten als nicht auswertbar
BUCKETS = [(1, 24, "1--24 h"), (25, 72, "25--72 h"), (73, 168, "73--168 h")]


# ---------------------------------------------------------------- Datenzugriff
def build_datasets(emobpy_n: int, ved_n: int, ved_files: int, yjmob_n: int) -> dict:
    """Gleiche Quellen und Grenzen wie die Kreuzvalidierung."""
    print("loading datasets...")
    out = {
        "emobpy": load_emobpy(limit_vehicles=emobpy_n),
        "real_world_ev": load_real_world_ev(),
        "ved": load_ved(limit_vehicles=ved_n, limit_files=ved_files),
        "routine": load_routine(),
        "yjmob": load_yjmob(limit_vehicles=yjmob_n),
    }
    for name, df in out.items():
        print(f"  {name}: rows={len(df):,} vehicles={df['vehicle_id'].nunique()} "
              f"positive_rate={df['driving'].mean():.3f}")
    return out


def vehicle_series(df: pd.DataFrame) -> dict:
    """vehicle_id -> (timestamps, driving) auf dem lueckenlosen Stundenraster."""
    series = {}
    for vid, g in df.sort_values("timestamp").groupby("vehicle_id", sort=False):
        series[vid] = (g["timestamp"].to_numpy(),
                       g["driving"].astype(np.int8).to_numpy())
    return series


def pick_origins(series: dict, t_split: pd.Timestamp, stride_h: int,
                 max_origins: int, min_vehicles: int) -> list:
    """Origins im Testteil, an denen genuegend Fahrzeuge Vor- und Nachlauf haben."""
    starts, ends = [], []
    for ts, _ in series.values():
        starts.append(pd.Timestamp(ts[0]))
        ends.append(pd.Timestamp(ts[-1]))
    first = max(min(starts) + pd.Timedelta(hours=LOOKBACK), t_split)
    last = max(ends) - pd.Timedelta(hours=HORIZON)
    if first > last:
        return []

    cand, t = [], first
    while t <= last:
        usable = sum(
            1 for ts, _ in series.values()
            if pd.Timestamp(ts[0]) <= t - pd.Timedelta(hours=LOOKBACK)
            and pd.Timestamp(ts[-1]) >= t + pd.Timedelta(hours=HORIZON)
        )
        if usable >= min_vehicles:
            cand.append(t)
        t += pd.Timedelta(hours=stride_h)
    if len(cand) <= max_origins:
        return cand
    # gleichmaessig ueber den Testzeitraum streuen statt nur den Anfang zu nehmen
    idx = np.linspace(0, len(cand) - 1, max_origins).round().astype(int)
    return [cand[i] for i in sorted(set(idx))]


def window_at(series: dict, t0: pd.Timestamp):
    """Fahrzeuge mit vollstaendigem Fenster: (hist 168, future 168) je Fahrzeug."""
    hist, future, vids = [], [], []
    for vid, (ts, y) in series.items():
        idx = int(np.searchsorted(ts, np.datetime64(t0)))
        # Historie endet auf t0 (inklusive), Prognose startet bei t0+1h
        lo, hi = idx - (LOOKBACK - 1), idx + 1 + HORIZON
        if lo < 0 or hi > len(ts) or ts[idx] != np.datetime64(t0):
            continue
        # Rasterluecken ausschliessen: das Fenster muss stundenweise anschliessen
        span = (ts[hi - 1] - ts[lo]).astype("timedelta64[h]")
        if int(span / np.timedelta64(1, "h")) != (hi - 1 - lo):
            continue
        hist.append(y[lo:idx + 1])
        future.append(y[idx + 1:hi])
        vids.append(vid)
    if not vids:
        return None
    return np.asarray(hist, dtype=np.float64), np.asarray(future, dtype=np.int8), vids


# ------------------------------------------------------------------- Rollout
def rollout(model, hist: np.ndarray, t0: pd.Timestamp, n_paths: int,
            rng: np.random.Generator) -> np.ndarray:
    """Autoregressiver Monte-Carlo-Rollout.

    hist: (V, 168) beobachtete Vorgeschichte. Rueckgabe: (V, HORIZON)
    mittlere Fahrwahrscheinlichkeit je Stunde ueber die Pfade.
    """
    n_veh = hist.shape[0]
    # (V*P, 168): jeder Pfad startet auf derselben beobachteten Historie
    buf = np.repeat(hist, n_paths, axis=0)
    out = np.zeros((n_veh, HORIZON), dtype=np.float64)

    for h in range(1, HORIZON + 1):
        t = t0 + pd.Timedelta(hours=h)
        n = buf.shape[0]
        X = np.empty((n, len(FEATURE_COLS)), dtype=np.float64)
        X[:, 0] = t.hour
        X[:, 1] = t.weekday()
        X[:, 2] = 1.0 if t.weekday() >= 5 else 0.0
        X[:, 3] = buf[:, -1]        # lag_1
        X[:, 4] = buf[:, -2]        # lag_2
        X[:, 5] = buf[:, -24]       # lag_24
        X[:, 6] = buf[:, -168]      # lag_168
        X[:, 7] = buf[:, -24:].mean(axis=1)    # rolling_mean_24
        X[:, 8] = buf[:, -168:].mean(axis=1)   # rolling_mean_168

        p = model.predict_proba(X)[:, 1]
        out[:, h - 1] = p.reshape(n_veh, n_paths).mean(axis=1)
        drawn = (rng.random(n) < p).astype(np.float64)
        buf = np.concatenate([buf[:, 1:], drawn[:, None]], axis=1)
    return out


# ------------------------------------------------------------------ Baselines
def climatology_table(train_df: pd.DataFrame) -> np.ndarray:
    """(7, 24)-Tabelle P(fahren | Wochentag, Stunde) aus dem Trainingsteil."""
    t = train_df["timestamp"]
    tab = (train_df.assign(weekday=t.dt.weekday, hour=t.dt.hour)
           .groupby(["weekday", "hour"])["driving"].mean()
           .reindex(pd.MultiIndex.from_product([range(7), range(24)]))
           .fillna(train_df["driving"].mean()).to_numpy().reshape(7, 24))
    return tab


def ap(y: np.ndarray, p: np.ndarray) -> float:
    """Average Precision; undefiniert ohne beide Klassen."""
    y = np.asarray(y).ravel()
    if y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, np.asarray(p).ravel()))


# ----------------------------------------------------------------------- Lauf
def evaluate_source(name: str, df: pd.DataFrame, algos: list, n_paths: int,
                    stride_h: int, max_origins: int, seed: int) -> list:
    feats = make_features(df)
    train_df, test_df = split_by_time(feats, 0.8)
    t_split = pd.Timestamp(train_df["timestamp"].max())
    series = vehicle_series(feats)

    origins = pick_origins(series, t_split, stride_h, max_origins, min_vehicles=1)
    if len(origins) < MIN_ORIGINS:
        # Zu wenige vollstaendige 168-h-Fenster: eine PR-AUC waere hier ein
        # Artefakt weniger Stunden und wird bewusst nicht berichtet.
        print(f"  {name}: nur {len(origins)} vollstaendige 168-h-Fenster im "
              f"Testteil -- nicht auswertbar")
        return []
    print(f"  {name}: {len(origins)} Origins ab {origins[0]}")

    clim = climatology_table(train_df)
    rows = []

    for algo in algos:
        model = train_model(train_df, TrainConfig(random_state=seed), algo=algo)
        rng = np.random.default_rng(seed)

        y_all, p_model, p_clim, p_pers = [], [], [], []
        for t0 in origins:
            win = window_at(series, t0)
            if win is None:
                continue
            hist, future, _ = win
            p_model.append(rollout(model, hist, t0, n_paths, rng))
            y_all.append(future)

            hours = [t0 + pd.Timedelta(hours=h) for h in range(1, HORIZON + 1)]
            p_clim.append(np.tile(
                np.array([clim[t.weekday(), t.hour] for t in hours]),
                (future.shape[0], 1)))
            # Persistenz-168: Prognose fuer t0+h ist y(t0+h-168); da die
            # Historie auf t0 endet, ist das genau hist[:, h-1].
            p_pers.append(hist.astype(np.float64))

        if not y_all:
            continue
        y = np.concatenate(y_all, axis=0)
        preds = {
            f"modell_{algo}": np.concatenate(p_model, axis=0),
            "klimatologie": np.concatenate(p_clim, axis=0),
            "persistenz168": np.concatenate(p_pers, axis=0),
        }

        for label, p in preds.items():
            if label == "klimatologie" and algo != algos[0]:
                continue  # Baselines nur einmal je Quelle berichten
            if label == "persistenz168" and algo != algos[0]:
                continue
            for h in range(1, HORIZON + 1):
                rows.append({
                    "quelle": name, "verfahren": label, "horizont_h": h,
                    "pr_auc": ap(y[:, h - 1], p[:, h - 1]),
                    "n": int(y.shape[0]),
                    "aktiv_rate": float(y[:, h - 1].mean()),
                })
            for lo, hi, blabel in BUCKETS:
                sl = slice(lo - 1, hi)
                rows.append({
                    "quelle": name, "verfahren": label, "horizont_h": -1,
                    "bucket": blabel,
                    "pr_auc": ap(y[:, sl], p[:, sl]),
                    "n": int(y.shape[0] * (hi - lo + 1)),
                    "aktiv_rate": float(y[:, sl].mean()),
                })
    return rows


def plot_curves(df: pd.DataFrame, out_path: Path, bin_h: int = 12) -> None:
    steps = df[df["horizont_h"] > 0].copy()
    steps["bin"] = ((steps["horizont_h"] - 1) // bin_h) * bin_h + bin_h / 2
    # Quellen ohne genuegend Fahrstunden je Bin ergeben nur Rauschen und
    # werden ausschliesslich in der Bucket-Tabelle berichtet.
    pos_per_bin = (steps.groupby("quelle")
                   .apply(lambda g: (g["n"] * g["aktiv_rate"]).median() * bin_h,
                          include_groups=False))
    sources = [s for s in SOURCE_ORDER
               if s in steps["quelle"].unique() and pos_per_bin.get(s, 0) >= 20]

    ncol = 3
    nrow = int(np.ceil(len(sources) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()

    styles = {"klimatologie": dict(ls="--", color="#888888"),
              "persistenz168": dict(ls=":", color="#bbbbbb")}
    for ax, src in zip(axes, sources):
        sub = steps[steps["quelle"] == src]
        for verf, g in sub.groupby("verfahren"):
            m = g.groupby("bin")["pr_auc"].mean()
            ax.plot(m.index, m.values, label=verf, lw=1.6,
                    **styles.get(verf, {}))
        ax.set_title(src, fontsize=10)
        ax.set_xlabel("Horizont [h]")
        ax.set_ylabel("PR-AUC")
        ax.set_xticks([24, 72, 120, 168])
        ax.grid(alpha=0.3)
    for ax in axes[len(sources):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  -> {out_path}")


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--algos", nargs="+", default=["rf"])
    ap_.add_argument("--paths", type=int, default=50)
    ap_.add_argument("--stride", type=int, default=168)
    ap_.add_argument("--max-origins", type=int, default=8)
    ap_.add_argument("--seed", type=int, default=42)
    ap_.add_argument("--limit-emobpy", type=int, default=30)
    ap_.add_argument("--limit-ved", type=int, default=15)
    ap_.add_argument("--limit-ved-files", type=int, default=4)
    ap_.add_argument("--limit-yjmob", type=int, default=50)
    args = ap_.parse_args()

    datasets = build_datasets(args.limit_emobpy, args.limit_ved,
                              args.limit_ved_files, args.limit_yjmob)

    rows = []
    for name in SOURCE_ORDER:
        rows += evaluate_source(name, datasets[name], args.algos, args.paths,
                                args.stride, args.max_origins, args.seed)

    out = pd.DataFrame(rows)
    out_dir = PRED_DIR / "horizon"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "horizon_metrics.csv", index=False)
    print(f"  -> {out_dir / 'horizon_metrics.csv'}")

    buckets = out[out["horizont_h"] == -1].pivot_table(
        index=["quelle", "verfahren"], columns="bucket", values="pr_auc")
    buckets.to_csv(out_dir / "horizon_buckets.csv")
    print(buckets.round(3).to_string())

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    plot_curves(out, IMG_DIR / "horizon_prauc.png")


if __name__ == "__main__":
    main()
