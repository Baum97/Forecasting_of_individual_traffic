"""
Train one model per data source and evaluate every model on every source.

Each dataset is split into train/test by time (80/20). Each model is fit on
the train portion of its own source, then scored on the test portion of all
sources. The result is, per algorithm, a 5x5 matrix of metrics plus F1- and
accuracy-heatmaps.

By default the cross-validation runs for two algorithms (RandomForest and
LightGBM, controlled via --algos), so the ablation study can compare the
algorithm effect against an identical feature set. The output CSV carries an
`algo` column to separate the matrices.

Run from the project root:
    python code/model_scripts/classifier/run_cross_validation.py
    python code/model_scripts/classifier/run_cross_validation.py --algos rf
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
    TrainConfig,
    evaluate,
    make_features,
    run_output_dir,
    save_model,
    split_by_time,
    train_model,
)
from model_scripts.data_adapters import (
    load_emobpy,
    load_real_world_ev,
    load_routine,
    load_ved,
    load_yjmob,
)

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
PRED_DIR = Path(__file__).resolve().parents[3] / "predictions"


def build_datasets(emobpy_n: int, ved_n: int, ved_files: int, yjmob_n: int) -> dict:
    print("loading datasets...")
    out = {}
    print(f"  emobpy (limit_vehicles={emobpy_n})...")
    out["emobpy"] = load_emobpy(limit_vehicles=emobpy_n)
    print(f"  real_world_ev...")
    out["real_world_ev"] = load_real_world_ev()
    print(f"  ved (limit_vehicles={ved_n}, limit_files={ved_files})...")
    out["ved"] = load_ved(limit_vehicles=ved_n, limit_files=ved_files)
    print(f"  routine...")
    out["routine"] = load_routine()
    print(f"  yjmob (limit_vehicles={yjmob_n})...")
    out["yjmob"] = load_yjmob(limit_vehicles=yjmob_n)

    for name, df in out.items():
        print(
            f"    {name}: rows={len(df):,} "
            f"vehicles={df['vehicle_id'].nunique()} "
            f"positive_rate={df['driving'].mean():.3f}"
        )
    return out


def heatmap(matrix: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    pivot = matrix.pivot(index="model_trained_on", columns="evaluated_on", values=metric)
    order = ["emobpy", "real_world_ev", "ved", "routine", "yjmob"]
    pivot = pivot.reindex(index=order, columns=order)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    values = pivot.values.astype(float)
    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("evaluated on")
    ax.set_ylabel("trained on")
    ax.set_title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                txt = "n/a"
                color = "white"
            else:
                txt = f"{v:.2f}"
                color = "white" if v < 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color)

    plt.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emobpy-vehicles", type=int, default=30)
    parser.add_argument("--ved-vehicles", type=int, default=15)
    parser.add_argument("--ved-files", type=int, default=4)
    parser.add_argument("--yjmob-vehicles", type=int, default=50)
    parser.add_argument("--algos", default="rf,lgbm",
                        help="Komma-Liste der Algorithmen "
                             "(rf, lgbm, lstm_tab, lstm_seq).")
    parser.add_argument("--lstm-min-vehicle-weeks", type=int, default=100,
                        help="Mindestumfang, ab dem eine Quelle als LSTM-"
                             "Trainingsquelle zugelassen wird. Vorab "
                             "festgelegtes Kriterium (siehe Methodik): "
                             "Sequenzmodelle brauchen unabhaengige "
                             "Fahrzeugwochen, nicht nur Zeilen. Quellen "
                             "darunter bleiben als Trainingszeile leer, "
                             "werden aber weiterhin evaluiert.")
    parser.add_argument("--seeds", default="42",
                        help="Komma-Liste der Zufallsinitialisierungen. Jeder "
                             "Seed wiederholt den kompletten Lauf; berichtet "
                             "werden Mittelwert und Standardabweichung ueber "
                             "die Wiederholungen. Noetig, weil die Streuung "
                             "der neuronalen Varianten sonst nicht von einem "
                             "echten Effekt zu unterscheiden ist.")
    parser.add_argument("--no-save-models", action="store_true",
                        help="Modelle nicht nach models/ schreiben "
                             "(Probelaeufe, die den Bestand nicht anfassen).")
    parser.add_argument("--dataset-tag", default="all_sources",
                        help="Rahmenbedingung im Run-Ordnernamen.")
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    run_dir = run_output_dir(
        model_name="cross_validation",
        dataset=args.dataset_tag,
        forecast_days=None,
    )
    if args.out_csv is None:
        args.out_csv = run_dir / "cross_validation_matrix.csv"
    print(f"run output dir: {run_dir}")
    print(f"algorithms: {algos}")

    datasets = build_datasets(args.emobpy_vehicles, args.ved_vehicles, args.ved_files,
                              args.yjmob_vehicles)
    # Compute features on the full timeline first, then split.
    # Otherwise lag_168 (1 week) would invalidate short test sets like VED.
    prepared = {name: make_features(df) for name, df in datasets.items()}
    splits = {name: split_by_time(df) for name, df in prepared.items()}

    # Umfang je Quelle in Fahrzeugwochen — massgeblich fuer Sequenzmodelle,
    # weil sich ueberlappende Fenster fast alle Zeitschritte teilen.
    vehicle_weeks = {
        name: float(len(df) / 168.0) for name, df in prepared.items()
    }
    print("\nUmfang (Fahrzeugwochen):")
    for name, w in vehicle_weeks.items():
        print(f"  {name:15s} {w:8.1f}")

    rows = []
    for algo in algos:
        print(f"\n========== algorithm: {algo} ==========")
        for seed in seeds:
            cfg = TrainConfig(random_state=seed)
            models = {}
            for name, (train_df, _) in splits.items():
                if algo.startswith("lstm") and vehicle_weeks[name] < args.lstm_min_vehicle_weeks:
                    if seed == seeds[0]:
                        print(f"skipping {algo} on {name}: {vehicle_weeks[name]:.1f} "
                              f"< {args.lstm_min_vehicle_weeks} Fahrzeugwochen")
                    continue
                print(f"training {algo} model on {name} (seed {seed})...")
                models[name] = train_model(train_df, cfg, algo=algo)
                # Gespeichert wird nur die erste Wiederholung -- die Modelle
                # dienen dem Backend, die Statistik kommt aus der CSV.
                if seed == seeds[0] and not args.no_save_models:
                    # rf behaelt den rueckwaertskompatiblen Namen.
                    fname = (f"driving_{name}.joblib" if algo == "rf"
                             else f"driving_{name}_{algo}.joblib")
                    save_model(models[name], MODELS_DIR / fname)

            for model_name, model in models.items():
                for data_name, (_, test_df) in splits.items():
                    print(f"evaluating [{algo}/{seed}] {model_name:>14s} -> {data_name}")
                    metrics = evaluate(model, test_df)
                    metrics["algo"] = algo
                    metrics["seed"] = seed
                    metrics["model_trained_on"] = model_name
                    metrics["evaluated_on"] = data_name
                    rows.append(metrics)

    raw = pd.DataFrame(rows)
    column_order = [
        "algo", "seed", "model_trained_on", "evaluated_on", "n", "positive_rate",
        "accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc", "brier",
    ]
    raw = raw[[c for c in column_order if c in raw.columns]]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}")

    # Aggregat ueber die Wiederholungen: Mittelwert und Standardabweichung.
    # Die Heatmaps und die Thesis-Tabellen beziehen sich auf den Mittelwert,
    # die Streuung entscheidet, ob ein Unterschied berichtbar ist.
    metric_cols = [c for c in ("accuracy", "f1", "precision", "recall",
                               "roc_auc", "pr_auc", "brier") if c in raw.columns]
    keys = ["algo", "model_trained_on", "evaluated_on"]
    agg = raw.groupby(keys, as_index=False).agg(
        n_seeds=("seed", "nunique"),
        n=("n", "first"),
        positive_rate=("positive_rate", "first"),
        **{f"{m}_{stat}": (m, stat)
           for m in metric_cols for stat in ("mean", "std")},
    )
    agg_path = args.out_csv.with_name(
        args.out_csv.stem + "_aggregated" + args.out_csv.suffix)
    agg.to_csv(agg_path, index=False)
    print(f"wrote {agg_path}")

    # Fuer Heatmaps/Zusammenfassung: Mittelwert unter den Originalnamen.
    matrix = agg.rename(columns={f"{m}_mean": m for m in metric_cols})

    order = ["emobpy", "real_world_ev", "ved", "routine", "yjmob"]
    if len(seeds) > 1:
        print(f"\n=== Streuung ueber {len(seeds)} Seeds "
              f"(max. Standardabweichung je Algorithmus) ===")
        for algo in algos:
            sub = agg[agg["algo"] == algo]
            if sub.empty:
                continue
            worst = {m: float(sub[f"{m}_std"].max()) for m in ("f1", "pr_auc", "roc_auc")}
            print(f"  {algo:9s} " + "  ".join(f"{k}={v:.4f}" for k, v in worst.items()))

    for algo in algos:
        sub = matrix[matrix["algo"] == algo]
        if sub.empty:
            continue
        heatmap(sub, "f1", f"F1 — {algo} driving classifier cross-validation",
                run_dir / f"cross_validation_f1_heatmap_{algo}.png")
        heatmap(sub, "accuracy", f"Accuracy — {algo} driving classifier cross-validation",
                run_dir / f"cross_validation_accuracy_heatmap_{algo}.png")

        print(f"\n=== summary F1 [{algo}] ===")
        pivot = sub.pivot(index="model_trained_on", columns="evaluated_on", values="f1")
        print(pivot.reindex(index=order, columns=order).round(3).to_string())

        print(f"\n=== diagonal (in-distribution) [{algo}] ===")
        diag = sub[sub["model_trained_on"] == sub["evaluated_on"]]
        print(diag.set_index("evaluated_on")[["positive_rate", "accuracy", "f1", "roc_auc"]]
              .reindex(order).round(3).to_string())


if __name__ == "__main__":
    main()
