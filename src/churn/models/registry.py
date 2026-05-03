"""Mapping of model names to concrete :class:`Model` classes.

The training loop iterates ``MODEL_REGISTRY.values()`` and trains every entry
against the same data and feature pipeline. Adding a fourth model is one line
in this file and the rest of the platform picks it up automatically.

The dict key is the canonical name (also used as MLflow tag and as the
identifier passed to ``make promote MODEL=<name> VERSION=<n>``); it must equal
the class-level :attr:`Model.name` attribute.
"""

from __future__ import annotations

from churn.models.base import Model
from churn.models.logreg import LogRegModel
from churn.models.tabular_mlp import TabularMLPModel
from churn.models.xgboost_model import XGBoostModel

MODEL_REGISTRY: dict[str, type[Model]] = {
    LogRegModel.name: LogRegModel,
    XGBoostModel.name: XGBoostModel,
    TabularMLPModel.name: TabularMLPModel,
}
