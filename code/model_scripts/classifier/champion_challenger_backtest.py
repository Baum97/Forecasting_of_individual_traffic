"""
Aufbau
------
Champion  : RF+LGBM-Ensemble, EINMAL trainiert auf allen Fahrzeugen der Quelle
            im fruehen Zeitfenster [0, champion_train_days]. Repraesentiert das
            vorab verfuegbare generische Modell (Cold-Start-Fallback).
Challenger: RF+LGBM-Ensemble, pro Fahrzeug auf dessen EIGENER Historie [0, d]
            trainiert -- waechst mit d (Tage eigener Historie).
Bewertung : prequential. Beide werden auf demselben *ungesehenen* Zukunftsfenster
            (d, d+eval_window] des Fahrzeugs gescort -> kein Leakage.

Ausgabe: CSV (pro Fahrzeug/Checkpoint/Modell), Crossover-Plot (PR-AUC vs. eigene
Historie) und der Median-Crossover-Tag ueber alle Fahrzeuge.

Aus dem Repo-Root:
    python code/model_scripts/classifier/champion_challenger_backtest.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_scripts.base import (
    FEATURE_COLS,
    TrainConfig,
    _build_classifier,
    make_features,
    run_output_dir,
)
from model_scripts.data_adapters import load_emobpy


def _fit_ensemble(X: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> list:
    """RF+LGBM-Ensemble (die zwei besten Modelle) auf denselben Daten."""
    return [_build_classifier(algo, cfg).fit(X, y) for algo in ("rf", "lgbm")]


def _ensemble_proba(models: list, X: np.ndarray) -> np.ndarray:
    """Mittelwert der Positiv-Wahrscheinlichkeiten beider Modelle."""
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def _metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    preds = (proba >= 0.5).astype(int)
    out = {
        "n": int(len(y)),
        "pos": int(y.sum()),
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "brier": float(brier_score_loss(y, proba, pos_label=1)),
        "pr_auc": (float(average_precision_score(y, proba))
                   if len(np.unique(y)) > 1 else float("nan")),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit-vehicles", type=int, default=10)
    p.add_argument("--champion-train-days", type=int, default=120,
                   help="Fruehes globales Fenster fuer das generische Modell.")
    p.add_argument("--warmup-days", type=int, default=28,
                   help="Erste eigene Historie, ab der der Challenger geprueft wird.")
    p.add_argument("--eval-every-days", type=int, default=21,
                   help="Schrittweite der Checkpoints entlang der Eigenhistorie.")
    p.add_argument("--eval-window-days", type=int, default=14,
                   help="Laenge des ungesehenen Zukunftsfensters pro Checkpoint.")
    p.add_argument("--dataset-tag", default="champion_challenger_emobpy")
    args = p.parse_args()

    cfg = TrainConfig()
    run_dir = run_output_dir("cross_validation", args.dataset_tag, None)
    out_csv = run_dir / "champion_challenger.csv"
    plot_png = run_dir / "champion_challenger_crossover.png"

    print(f"loading emobpy (limit_vehicles={args.limit_vehicles})...")
    df = load_emobpy(limit_vehicles=args.limit_vehicles)
    feats = make_features(df).dropna(subset=FEATURE_COLS).copy()
    feats["day"] = (feats["timestamp"] - feats["timestamp"].min()).dt.days

    # --- Champion: generisch, alle Fahrzeuge, fruehes Fenster ---------------
    champ_train = feats[feats["day"] < args.champion_train_days]
    champion = _fit_ensemble(
        champ_train[FEATURE_COLS].to_numpy(),
        champ_train["driving"].astype(int).to_numpy(),
        cfg,
    )
    print(f"[champion] {len(champ_train):,} Zeilen, "
          f"{feats['vehicle_id'].nunique()} Fahrzeuge, "
          f"erste {args.champion_train_days} Tage.")

    # --- Challenger: pro Fahrzeug, wachsende Eigenhistorie ------------------
    rows = []
    for vid, g in feats.groupby("vehicle_id", sort=False):
        g = g.sort_values("timestamp").copy()
        g["vday"] = (g["timestamp"] - g["timestamp"].min()).dt.days
        max_day = int(g["vday"].max())

        for d in range(args.warmup_days,
                       max_day - args.eval_window_days + 1,
                       args.eval_every_days):
            train = g[g["vday"] < d]
            evalw = g[(g["vday"] >= d) & (g["vday"] < d + args.eval_window_days)]
            y_eval = evalw["driving"].astype(int).to_numpy()
            y_train = train["driving"].astype(int).to_numpy()
            # Guards: genug Trainingsdaten, beide Klassen, Positive im Fenster.
            if len(train) < 50 or len(np.unique(y_train)) < 2 or y_eval.sum() == 0:
                continue

            X_eval = evalw[FEATURE_COLS].to_numpy()
            challenger = _fit_ensemble(train[FEATURE_COLS].to_numpy(), y_train, cfg)

            m_ch = _metrics(y_eval, _ensemble_proba(challenger, X_eval))
            m_cp = _metrics(y_eval, _ensemble_proba(champion, X_eval))
            rows.append({"vehicle_id": vid, "own_history_days": d,
                         "model": "challenger", **m_ch})
            rows.append({"vehicle_id": vid, "own_history_days": d,
                         "model": "champion", **m_cp})
        print(f"  {vid}: fertig")

    res = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    # --- Aggregation + Crossover-Plot (PR-AUC) ------------------------------
    piv = (res.groupby(["model", "own_history_days"])["pr_auc"].mean()
           .reset_index()
           .pivot(index="own_history_days", columns="model", values="pr_auc"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(piv.index, piv["champion"], marker="o", label="Champion (generisch, RF+LGBM)")
    ax.plot(piv.index, piv["challenger"], marker="s", label="Challenger (personalisiert, RF+LGBM)")
    ax.set_xlabel("Eigene Historie [Tage]")
    ax.set_ylabel("PR-AUC (Mittel über Fahrzeuge)")
    ax.set_title("Personalisierungs-Crossover auf emobpy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_png, dpi=150)
    plt.close(fig)
    print(f"wrote {plot_png}")

    # --- Crossover-Tag pro Fahrzeug (erstes d mit Challenger >= Champion) ---
    wide = res.pivot_table(index=["vehicle_id", "own_history_days"],
                           columns="model", values="pr_auc").reset_index()
    crossings = []
    for vid, gg in wide.groupby("vehicle_id"):
        gg = gg.sort_values("own_history_days")
        won = gg[gg["challenger"] >= gg["champion"]]
        crossings.append(float(won["own_history_days"].iloc[0]) if len(won) else np.nan)
    cross = pd.Series(crossings, dtype=float)

    print("\n=== Crossover (PR-AUC) ===")
    print(f"Fahrzeuge gesamt        : {len(cross)}")
    print(f"davon je ueberholt      : {int(cross.notna().sum())}")
    if cross.notna().any():
        print(f"Median-Crossover-Tag    : {cross.median():.0f}")
        print(f"Mittel-Crossover-Tag    : {cross.mean():.0f}")

    print("\n=== Mittelwerte je Historie (PR-AUC) ===")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
