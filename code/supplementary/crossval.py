"""Rendert die F1-Isolationsmatrix je Algorithmus als Thesis-Abbildung.

Liest die cross_validation_matrix.csv eines Runs und schreibt pro Wert der
Spalte `algo` eine Heatmap nach thesis/images/. Ohne Argumente wird der
neueste Run unter predictions/cross_validation/ verwendet.

    python code/supplementary/crossval.py
    python code/supplementary/crossval.py --run all_sources_T21-07-2026
    python code/supplementary/crossval.py --metric accuracy --algos rf,lgbm
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "predictions" / "cross_validation"
IMAGES = ROOT / "thesis" / "images"
ORDER = ["emobpy", "real_world_ev", "ved", "routine", "yjmob"]

# rf behaelt den Dateinamen ohne Suffix, weil 05evaluation.tex ihn so einbindet.
STEM = {"rf": "cv_f1_5x5"}


def newest_run() -> Path:
    runs = [p for p in RUNS.iterdir()
            if p.is_dir() and (p / "cross_validation_matrix.csv").exists()]
    if not runs:
        raise SystemExit(f"kein Run mit cross_validation_matrix.csv unter {RUNS}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def heatmap(df: pd.DataFrame, metric: str, out: Path) -> pd.DataFrame:
    mat = (df.pivot(index="model_trained_on", columns="evaluated_on", values=metric)
             .reindex(index=ORDER, columns=ORDER))
    values = mat.values.astype(float)

    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    im = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(ORDER)), ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(ORDER)), ORDER)
    ax.set_xlabel("getestet auf")
    ax.set_ylabel("trainiert auf")

    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            v = values[i, j]
            # Leere Zeilen entstehen, wenn eine Quelle das Fahrzeugwochen-
            # Kriterium nicht erfuellt und nicht als Trainingsquelle zugelassen
            # ist (siehe Methodik) -- als n/a kenntlich, nicht als 0.
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}",
                    ha="center", va="center",
                    color="white" if np.isnan(v) or v < 0.5 else "black")

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(metric.upper() if metric == "f1" else metric)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return mat


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None,
                   help="Ordnername unter predictions/cross_validation/ "
                        "(Default: neuester Run).")
    p.add_argument("--metric", default="f1")
    p.add_argument("--algos", default=None,
                   help="Komma-Liste; Default: alle in der CSV vorhandenen.")
    args = p.parse_args()

    run_dir = RUNS / args.run if args.run else newest_run()
    csv = run_dir / "cross_validation_matrix.csv"
    print(f"run: {run_dir.name}")

    df = pd.read_csv(csv)
    algos = ([a.strip() for a in args.algos.split(",") if a.strip()]
             if args.algos else list(df["algo"].unique()))

    for algo in algos:
        sub = df[df["algo"] == algo]
        if sub.empty:
            print(f"  [skip] {algo}: nicht in der CSV")
            continue
        stem = (STEM.get(algo, f"cv_f1_5x5_{algo}") if args.metric == "f1"
                else f"cv_{args.metric}_5x5_{algo}")
        out = IMAGES / f"{stem}.png"
        mat = heatmap(sub, args.metric, out)
        print(f"written: {out}")
        print(mat.round(2).to_string(), end="\n\n")


if __name__ == "__main__":
    main()
