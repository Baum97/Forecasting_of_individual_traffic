"""
Ablationsstudie ueber Merkmalsgruppen (nicht ueber Modelle).

Je Datenquelle und Algorithmus wird der Driving-Classifier mehrfach trainiert:
einmal mit dem vollstaendigen Feature-Satz (Referenz) und je einmal mit einer
entfernten beziehungsweise isolierten Merkmalsgruppe. Gemessen wird der Gueteverlust
gegenueber der Referenz

Merkmalsgruppen:
  kalender : hour, weekday, is_weekend
  lag      : lag_1, lag_2, lag_24, lag_168
  rolling  : rolling_mean_24, rolling_mean_168

Aufruf aus dem Projekt-Root:
    python code/model_scripts/classifier/run_feature_ablation.py
    python code/model_scripts/classifier/run_feature_ablation.py --algos rf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_scripts.base import (  # noqa: E402
    FEATURE_COLS,
    TrainConfig,
    _build_classifier,
    make_features,
    split_by_time,
)
from model_scripts.data_adapters import (  # noqa: E402
    load_emobpy,
    load_real_world_ev,
    load_routine,
    load_ved,
    load_yjmob,
)

GROUPS: dict[str, list[str]] = {
    "kalender": ["hour", "weekday", "is_weekend"],
    "lag": ["lag_1", "lag_2", "lag_24", "lag_168"],
    "rolling": ["rolling_mean_24", "rolling_mean_168"],
}


def variants() -> dict[str, list[str]]:
    """Name -> verwendete Feature-Spalten."""
    out: dict[str, list[str]] = {"voll": list(FEATURE_COLS)}
    for g, cols in GROUPS.items():
        out[f"ohne_{g}"] = [c for c in FEATURE_COLS if c not in cols]
        out[f"nur_{g}"] = list(cols)
    return out


def build_datasets(emobpy_n: int, ved_n: int, ved_files: int, yjmob_n: int) -> dict:
    print("lade Datenquellen ...", flush=True)
    out = {}
    out["emobpy"] = load_emobpy(limit_vehicles=emobpy_n)
    out["real_world_ev"] = load_real_world_ev()
    out["ved"] = load_ved(limit_vehicles=ved_n, limit_files=ved_files)
    out["routine"] = load_routine()
    out["yjmob100k"] = load_yjmob(limit_vehicles=yjmob_n)
    for k, v in out.items():
        print(f"  {k:15s} {len(v):>10,} Zeilen", flush=True)
    return out


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features berechnen, NaN anhand des VOLLEN Satzes verwerfen, dann splitten."""
    feats = make_features(df).dropna(subset=FEATURE_COLS)
    return split_by_time(feats, train_frac=0.8)


def score(train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
          algo: str, cfg: TrainConfig) -> dict | None:
    y_tr = train["driving"].astype(int).to_numpy()
    y_te = test["driving"].astype(int).to_numpy()
    if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
        return None

    model = _build_classifier(algo, cfg)
    if getattr(model, "needs_frame", False):
        # lstm_seq arbeitet auf der rohen Reihe und kennt keine
        # Spalten-Teilmengen -- fuer eine Merkmals-Ablation nicht definiert.
        raise ValueError(f"algo {algo!r} ist nicht ablationsfaehig (needs_frame)")
    if hasattr(model, "set_feature_names"):
        model.set_feature_names(cols)
    model.fit(train[cols].to_numpy(), y_tr)

    X_te = test[cols].to_numpy()
    preds = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]

    res = {
        "n_test": int(len(y_te)),
        "aktiv_rate": float(y_te.mean()),
        "accuracy": float(accuracy_score(y_te, preds)),
        "f1": float(f1_score(y_te, preds, zero_division=0)),
        "brier": float(brier_score_loss(y_te, proba, pos_label=1)),
    }
    if len(np.unique(y_te)) > 1:
        res["roc_auc"] = float(roc_auc_score(y_te, proba))
        res["pr_auc"] = float(average_precision_score(y_te, proba))
    else:
        res["roc_auc"] = float("nan")
        res["pr_auc"] = float("nan")
    return res


def main() -> None:
    p = argparse.ArgumentParser(prog="run_feature_ablation.py")
    p.add_argument("--emobpy-vehicles", type=int, default=30)
    p.add_argument("--ved-vehicles", type=int, default=15)
    p.add_argument("--ved-files", type=int, default=4)
    p.add_argument("--yjmob-vehicles", type=int, default=50)
    p.add_argument("--algos", default="rf,lgbm")
    p.add_argument("--out-csv", type=Path,
                   default=Path(__file__).resolve().parents[3]
                   / "predictions" / "feature_ablation.csv")
    args = p.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    cfg = TrainConfig()
    vars_ = variants()

    data = build_datasets(args.emobpy_vehicles, args.ved_vehicles,
                          args.ved_files, args.yjmob_vehicles)

    rows: list[dict] = []
    for src, df in data.items():
        train, test = prepare(df)
        print(f"\n{src}: train={len(train):,}  test={len(test):,}", flush=True)
        for algo in algos:
            for vname, cols in vars_.items():
                res = score(train, test, cols, algo, cfg)
                if res is None:
                    print(f"  [skip] {algo:5s} {vname:15s} (nur eine Klasse)", flush=True)
                    continue
                rows.append({"quelle": src, "algo": algo, "variante": vname,
                             "n_features": len(cols), **res})
                print(f"  {algo:5s} {vname:15s} "
                      f"F1={res['f1']:.3f}  PR-AUC={res['pr_auc']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\ngeschrieben: {args.out_csv}", flush=True)

    # Delta gegenueber der Referenz "voll"
    print("\n=== Delta gegenueber 'voll' (negativ = Verschlechterung) ===", flush=True)
    for algo in algos:
        sub = out[out["algo"] == algo]
        if sub.empty:
            continue
        ref = sub[sub["variante"] == "voll"].set_index("quelle")
        print(f"\n-- {algo} --", flush=True)
        piv = []
        for vname in vars_:
            if vname == "voll":
                continue
            cur = sub[sub["variante"] == vname].set_index("quelle")
            common = ref.index.intersection(cur.index)
            piv.append({
                "variante": vname,
                **{q: round(cur.loc[q, "pr_auc"] - ref.loc[q, "pr_auc"], 3)
                   for q in common},
            })
        print(pd.DataFrame(piv).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
