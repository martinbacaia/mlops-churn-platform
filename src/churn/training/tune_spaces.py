"""Per-model Optuna search spaces.

Each ``suggest_*_params`` function describes the hyperparameter space for one
model, returning a kwargs dict ready to splat into the model's constructor.
The :data:`SUGGEST_PARAMS` dispatcher routes by the same name used in
:data:`churn.models.registry.MODEL_REGISTRY` — adding a fourth model is a
matter of writing one suggest function and adding one entry here.

Spaces are deliberately narrow: 3-6 hyperparams each. A wider space wastes
trials; the goal is to demonstrate disciplined tuning, not to brute-force
the loss surface.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import optuna


def suggest_logreg_params(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    """Search space for :class:`LogRegModel`."""
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "max_iter": 1000,
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def suggest_xgboost_params(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    """Search space for :class:`XGBoostModel`.

    Tree shape (depth + n_estimators) and regularization are the typical
    high-leverage knobs; subsampling values are kept above 0.7 because lower
    rates hurt this dataset's small effective signal.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }


# Hidden-layer architectures kept as named choices so Optuna treats them as
# categorical (avoids spurious "10 dim integer space" exploration).
_TABULAR_MLP_HIDDEN_CHOICES = {
    "32x16": (32, 16),
    "64x32": (64, 32),
    "128x64": (128, 64),
}


def suggest_tabular_mlp_params(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    """Search space for :class:`TabularMLPModel`."""
    hidden_key = trial.suggest_categorical("hidden", list(_TABULAR_MLP_HIDDEN_CHOICES.keys()))
    return {
        "hidden": _TABULAR_MLP_HIDDEN_CHOICES[hidden_key],
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 10, 50, step=5),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    }


SUGGEST_PARAMS: dict[str, Callable[[optuna.trial.BaseTrial], dict[str, Any]]] = {
    "logreg": suggest_logreg_params,
    "xgboost": suggest_xgboost_params,
    "tabular_mlp": suggest_tabular_mlp_params,
}


def suggest_params(model_name: str, trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    """Dispatch to the per-model suggest function. Raises on unknown name."""
    if model_name not in SUGGEST_PARAMS:
        raise KeyError(f"Unknown model {model_name!r}. " f"Known: {sorted(SUGGEST_PARAMS.keys())}.")
    return SUGGEST_PARAMS[model_name](trial)
