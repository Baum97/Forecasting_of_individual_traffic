"""Registry der verfuegbaren Forecaster-Implementierungen.

Jede Klasse muss die folgende Schnittstelle erfuellen (Duck-Typing, keine
Vererbung noetig):

    __init__(cfg)              -> instance
    fit(hourly_df)             -> self
    predict(n_days, threshold, min_active_p) -> pd.DataFrame
    predict_proba_day(X)       -> float
    predict_quantiles(X, target) -> (mean, p10, p90) | (None, None, None)
    save(path)                 -> None
    load(path)                 -> instance   (staticmethod)
    Attribute: daily, profile, clf, reg_dep, reg_ret, cfg
                vehicle_id (optional), per_vehicle_daily (optional)

Die Klassen werden lazy importiert — fehlende Libraries (z.B. lightgbm,
torch) brauchen nur installiert zu sein, wenn der jeweilige Algo verwendet
wird.
"""
from __future__ import annotations

from importlib import import_module
from typing import Type

REGISTRY = {
    "rf":   ("model_scripts.randomforest_forecaster", "RandomForestForecaster"),
    "lgbm": ("model_scripts.lgbm_forecaster",         "LGBMForecaster"),
}


def get_forecaster_class(name: str) -> Type:
    """Lazy-Import der Forecaster-Klasse fuer einen Algo-Namen."""
    if name not in REGISTRY:
        raise ValueError(
            f"unknown forecaster '{name}', expected one of {list(REGISTRY)}"
        )
    mod_name, cls_name = REGISTRY[name]
    try:
        module = import_module(mod_name)
    except ImportError as e:
        raise ImportError(
            f"forecaster '{name}' benoetigt Modul '{mod_name}': {e}. "
            f"Ist die zugehoerige Library installiert?"
        ) from e
    return getattr(module, cls_name)


def available_algos() -> list[str]:
    return list(REGISTRY)
