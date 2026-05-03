from __future__ import annotations

import pytest
from optuna.trial import FixedTrial

from churn.models.logreg import LogRegModel
from churn.models.tabular_mlp import TabularMLPModel
from churn.models.xgboost_model import XGBoostModel
from churn.training.tune_spaces import (
    SUGGEST_PARAMS,
    suggest_logreg_params,
    suggest_params,
    suggest_tabular_mlp_params,
    suggest_xgboost_params,
)

# ---- LogReg --------------------------------------------------------------


def test_logreg_returns_required_keys():
    trial = FixedTrial({"C": 1.0, "class_weight": "balanced"})
    out = suggest_logreg_params(trial)
    assert {"C", "max_iter", "class_weight"}.issubset(out.keys())


def test_logreg_params_can_instantiate_the_model():
    trial = FixedTrial({"C": 0.5, "class_weight": None})
    params = suggest_logreg_params(trial)
    model = LogRegModel(**params)  # must not raise
    assert model.C == 0.5
    assert model.class_weight is None


# ---- XGBoost -------------------------------------------------------------


def test_xgboost_returns_required_keys():
    trial = FixedTrial(
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
        }
    )
    out = suggest_xgboost_params(trial)
    assert {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    } == set(out.keys())


def test_xgboost_params_can_instantiate_the_model():
    trial = FixedTrial(
        {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
        }
    )
    model = XGBoostModel(**suggest_xgboost_params(trial))
    assert model.max_depth == 4


# ---- TabularMLP ----------------------------------------------------------


def test_tabular_mlp_returns_required_keys():
    trial = FixedTrial(
        {
            "hidden": "64x32",
            "dropout": 0.2,
            "lr": 1e-3,
            "epochs": 20,
            "batch_size": 256,
            "weight_decay": 1e-4,
        }
    )
    out = suggest_tabular_mlp_params(trial)
    assert {"hidden", "dropout", "lr", "epochs", "batch_size", "weight_decay"} == set(out.keys())


def test_tabular_mlp_hidden_is_tuple_of_two_ints():
    trial = FixedTrial(
        {
            "hidden": "128x64",
            "dropout": 0.1,
            "lr": 1e-3,
            "epochs": 10,
            "batch_size": 128,
            "weight_decay": 1e-5,
        }
    )
    out = suggest_tabular_mlp_params(trial)
    assert out["hidden"] == (128, 64)


def test_tabular_mlp_params_can_instantiate_the_model():
    trial = FixedTrial(
        {
            "hidden": "32x16",
            "dropout": 0.3,
            "lr": 1e-3,
            "epochs": 5,
            "batch_size": 128,
            "weight_decay": 1e-4,
        }
    )
    model = TabularMLPModel(**suggest_tabular_mlp_params(trial))
    assert model.hidden == (32, 16)
    assert model.dropout == 0.3


# ---- Dispatcher ---------------------------------------------------------


def test_dispatcher_routes_to_correct_function():
    trial = FixedTrial({"C": 1.0, "class_weight": "balanced"})
    direct = suggest_logreg_params(trial)
    via_dispatcher = suggest_params("logreg", FixedTrial({"C": 1.0, "class_weight": "balanced"}))
    assert direct == via_dispatcher


def test_dispatcher_unknown_model_raises():
    with pytest.raises(KeyError, match="Unknown model"):
        suggest_params("transformer", FixedTrial({}))


def test_dispatcher_keys_match_model_registry():
    """The tuning layer must cover every model in the model registry."""
    from churn.models.registry import MODEL_REGISTRY

    assert set(SUGGEST_PARAMS.keys()) == set(MODEL_REGISTRY.keys())
