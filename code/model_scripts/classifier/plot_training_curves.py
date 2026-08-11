"""
Diagnose der Overfitting-Kontrolle: Trainings- vs. Validierungs-Loss ueber den
Trainingsverlauf fuer die iterativ trainierten Verfahren (LightGBM und LSTM).

Zweck: zeigen, dass der Validierungs-Loss nicht wieder ansteigt, waehrend der
Trainings-Loss faellt (kein Overfitting), und den von Early Stopping gewaehlten
Haltepunkt markieren. Random Forest hat keinen iterativen Trainingsverlauf und
wird ueber die Trainings-vs-Test-Luecke der Kreuzvalidierung beurteilt, nicht
ueber eine Lernkurve.

Dies ist ein *diagnostischer* Lauf: er erzeugt eigenstaendige Lernkurven mit
denselben Hyperparametern wie die berichteten Modelle, veraendert aber weder die
Kreuzvalidierungs-Matrix noch die berichteten Metriken.

Aus dem Projekt-Root:
    python code/model_scripts/classifier/plot_training_curves.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_scripts.base import (
    FEATURE_COLS,
    TrainConfig,
    make_features,
    run_output_dir,
    split_by_time,
)
from model_scripts.classifier.lstm_classifier import (
    LSTMConfig,
    LSTMSequenceClassifier,
    LSTMTabularClassifier,
)
from model_scripts.data_adapters import (
    load_emobpy,
    load_real_world_ev,
    load_routine,
    load_ved,
    load_yjmob,
)

IMAGES_DIR = Path(__file__).resolve().parents[3] / "thesis" / "images"

# Quellen, die die 100-Fahrzeugwochen-Schranke der Sequenzmodelle erfuellen.
LSTM_SOURCES = ["emobpy", "yjmob100k"]


def build_datasets(emobpy_n: int, ved_n: int, ved_files: int, yjmob_n: int) -> dict:
    print("loading datasets...")
    return {
        "emobpy": load_emobpy(limit_vehicles=emobpy_n),
        "real_world_ev": load_real_world_ev(),
        "ved": load_ved(limit_vehicles=ved_n, limit_files=ved_files),
        "routine": load_routine(),
        "yjmob100k": load_yjmob(limit_vehicles=yjmob_n),
    }


def lgbm_curve(train_df: pd.DataFrame, cfg: TrainConfig) -> dict | None:
    """Fit LightGBM mit zeitlichem Val-Tail und Early Stopping; liefert die
    train/val-Logloss-Kurve ueber die Boosting-Runden."""
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    feats = train_df.dropna(subset=FEATURE_COLS)
    # Zeitlicher Validierungs-Tail (kein zufaelliger Split -> kein Leakage).
    tr, va = split_by_time(feats, train_frac=0.9)
    if len(va) < 50 or tr["driving"].nunique() < 2 or va["driving"].nunique() < 2:
        return None

    Xtr, ytr = tr[FEATURE_COLS].to_numpy(), tr["driving"].astype(int).to_numpy()
    Xva, yva = va[FEATURE_COLS].to_numpy(), va["driving"].astype(int).to_numpy()

    model = LGBMClassifier(
        n_estimators=cfg.n_estimators,
        min_child_samples=max(cfg.min_samples_leaf, 5),
        max_depth=cfg.max_depth,
        class_weight="balanced",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        verbosity=-1,
    )
    model.fit(
        Xtr, ytr,
        eval_set=[(Xtr, ytr), (Xva, yva)],
        eval_names=["train", "val"],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(20, verbose=False), log_evaluation(0)],
    )
    res = model.evals_result_
    return {
        "train_loss": res["train"]["binary_logloss"],
        "val_loss": res["val"]["binary_logloss"],
        "best_epoch": (model.best_iteration_ - 1) if model.best_iteration_ else None,
    }


def lstm_curve(train_df: pd.DataFrame, variant: str, seed: int) -> dict | None:
    """Fit eine LSTM-Variante und liefere ihre Lernkurve (Early Stopping ist
    im Trainer bereits aktiv)."""
    lcfg = LSTMConfig(random_state=seed)
    if variant == "tab":
        feats = train_df.dropna(subset=FEATURE_COLS)
        if feats["driving"].nunique() < 2:
            return None
        clf = LSTMTabularClassifier(lcfg)
        clf.fit(feats[FEATURE_COLS].to_numpy(),
                feats["driving"].astype(int).to_numpy())
    else:
        clf = LSTMSequenceClassifier(lcfg)
        clf.fit_frame(train_df)
    return clf.history


def _plot_panel(ax, hist: dict, title: str, xlabel: str) -> None:
    tr = hist["train_loss"]
    va = hist["val_loss"]
    ep = range(1, len(tr) + 1)
    ax.plot(ep, tr, label="Training", color="#5B7FB0", linewidth=1.6)
    if any(np.isfinite(va)):
        ax.plot(ep, va, label="Validierung", color="#D08A3E", linewidth=1.6)
    be = hist.get("best_epoch")
    if be is not None and be >= 0:
        ax.axvline(be + 1, color="#7A7A7A", linestyle="--", linewidth=1.0)
        ax.text(be + 1, ax.get_ylim()[1], "  Early Stop", color="#7A7A7A",
                va="top", ha="left", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emobpy-vehicles", type=int, default=30)
    parser.add_argument("--ved-vehicles", type=int, default=15)
    parser.add_argument("--ved-files", type=int, default=4)
    parser.add_argument("--yjmob-vehicles", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(random_state=args.seed)
    run_dir = run_output_dir(model_name="training_curves", dataset="all_sources")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(args.emobpy_vehicles, args.ved_vehicles,
                              args.ved_files, args.yjmob_vehicles)
    prepared = {n: make_features(df) for n, df in datasets.items()}
    splits = {n: split_by_time(df) for n, df in prepared.items()}

    order = ["emobpy", "real_world_ev", "ved", "routine", "yjmob100k"]
    hist_rows = []

    # ----- LightGBM: eine Kurve je Quelle ------------------------------------
    print("\n== LightGBM ==")
    lgbm_hist = {}
    for name in order:
        train_df, _ = splits[name]
        print(f"  {name}...")
        h = lgbm_curve(train_df, cfg)
        if h is not None:
            lgbm_hist[name] = h
            for i, (t, v) in enumerate(zip(h["train_loss"], h["val_loss"])):
                hist_rows.append({"algo": "lgbm", "source": name, "iter": i + 1,
                                  "train_loss": t, "val_loss": v})

    lgbm_panels = [(f"LightGBM — {name}", h) for name, h in lgbm_hist.items()]
    ncol = 2
    nrow = (len(lgbm_panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow),
                             squeeze=False)
    flat = axes.flatten()
    for ax, (title, h) in zip(flat, lgbm_panels):
        _plot_panel(ax, h, title, "Boosting-Runde")
    for ax in flat[len(lgbm_panels):]:
        ax.axis("off")
    fig.tight_layout()
    p = IMAGES_DIR / "training_curves_lgbm.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"wrote {p}")

    # ----- LSTM: tabellarisch + sequentiell je zugelassener Quelle -----------
    print("\n== LSTM ==")
    panels = []
    for name in LSTM_SOURCES:
        train_df, _ = splits[name]
        for variant, label in (("tab", "tabellarisch"), ("seq", "sequentiell")):
            print(f"  {name} / {variant}...")
            h = lstm_curve(train_df, variant, args.seed)
            if h is not None and h.get("train_loss"):
                panels.append((f"LSTM {label} — {name}", h))
                for i, (t, v) in enumerate(zip(h["train_loss"], h["val_loss"])):
                    hist_rows.append({"algo": f"lstm_{variant}", "source": name,
                                      "iter": i + 1, "train_loss": t, "val_loss": v})

    ncol = 2
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow),
                             squeeze=False)
    flat = axes.flatten()
    for ax, (title, h) in zip(flat, panels):
        _plot_panel(ax, h, title, "Epoche")
    for ax in flat[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    p = IMAGES_DIR / "training_curves_lstm.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"wrote {p}")

    csv_path = run_dir / "training_curves.csv"
    pd.DataFrame(hist_rows).to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
