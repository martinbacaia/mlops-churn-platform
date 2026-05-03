"""Load the registered model + its feature pipeline from MLflow at startup.

Resolution order:

1. Look up the latest version of the registered model in the configured stage
   (``Settings.model_stage``, default ``Production``).
2. Load the model via ``mlflow.sklearn.load_model``.
3. Download the ``feature_pipeline.joblib`` artifact from the *same run* and
   load it. Same run = the model and pipeline have always been a matched pair,
   guaranteeing the input transformation matches what the model was trained on.
4. Return both plus enough metadata to populate the ``/model_info`` endpoint.

A loader failure at startup is intentionally fatal — the server should refuse
to start if it can't serve a real model. ``/health`` reflects this.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer

from churn.config import get_settings
from churn.logging_setup import get_logger
from churn.models.base import Model

_log = get_logger(__name__)


@dataclass(frozen=True)
class ProductionArtifacts:
    """Everything the API needs to serve a single prediction."""

    model: Model
    feature_pipeline: ColumnTransformer
    registered_model_name: str
    model_version: str
    stage: str
    run_id: str
    model_type: str
    feature_pipeline_version: str | None


class NoActiveModelError(RuntimeError):
    """No registered version exists in the requested stage."""


def load_production_artifacts(
    model_name: str | None = None,
    stage: str | None = None,
) -> ProductionArtifacts:
    """Load the active version + its feature pipeline.

    Args:
        model_name: Registered model name. Defaults to ``Settings.model_name``.
        stage: Target stage. Defaults to ``Settings.model_stage`` (``Production``).

    Raises:
        NoActiveModelError: If no version of ``model_name`` is in ``stage``.
    """
    settings = get_settings()
    name = model_name or settings.model_name
    target_stage = stage or settings.model_stage

    client = MlflowClient()
    # ``get_latest_versions`` is deprecated (stages are being phased out for
    # aliases). The spec is explicit about stages, so we suppress that single
    # FutureWarning locally — same pattern as in promote.py.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            versions = client.get_latest_versions(name, stages=[target_stage])
    except MlflowException as exc:
        # Raised when the registered model itself does not exist.
        raise NoActiveModelError(
            f"Registered model {name!r} not found. Train and register one first."
        ) from exc
    if not versions:
        raise NoActiveModelError(
            f"No version of {name!r} is in stage {target_stage!r}. "
            f"Promote one with `make promote VERSION=<n>` before starting the API."
        )
    mv = versions[0]
    run = client.get_run(mv.run_id)

    model_uri = f"models:/{name}/{target_stage}"
    raw_model: Any = mlflow.sklearn.load_model(model_uri)
    if not isinstance(raw_model, Model):
        raise TypeError(
            f"Loaded artifact at {model_uri} is {type(raw_model).__name__}; "
            f"expected a churn.models.base.Model subclass."
        )

    pipeline_local_path = Path(
        mlflow.artifacts.download_artifacts(
            run_id=mv.run_id,
            artifact_path="feature_pipeline/feature_pipeline.joblib",
        )
    )
    feature_pipeline = joblib.load(pipeline_local_path)
    if not isinstance(feature_pipeline, ColumnTransformer):
        raise TypeError(
            f"feature_pipeline.joblib in run {mv.run_id} is "
            f"{type(feature_pipeline).__name__}; expected ColumnTransformer."
        )

    artifacts = ProductionArtifacts(
        model=raw_model,
        feature_pipeline=feature_pipeline,
        registered_model_name=name,
        model_version=str(mv.version),
        stage=target_stage,
        run_id=mv.run_id,
        model_type=run.data.tags.get("model_type", "unknown"),
        feature_pipeline_version=run.data.tags.get("feature_pipeline_version"),
    )
    _log.info(
        "production_artifacts_loaded",
        model_name=name,
        version=artifacts.model_version,
        stage=target_stage,
        model_type=artifacts.model_type,
        run_id=mv.run_id,
    )
    return artifacts
