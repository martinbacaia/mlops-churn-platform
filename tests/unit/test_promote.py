from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from churn.training.promote import promote_version

REGISTRY_NAME = "test_churn_classifier"


@pytest.fixture
def isolated_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    db = tmp_path / "mlruns.db"
    tracking_uri = f"sqlite:///{db}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_promote")
    monkeypatch.setenv("MODEL_NAME", REGISTRY_NAME)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test_promote")
    yield tracking_uri


def _create_registered_version(model_type: str = "logreg") -> str:
    """Train a tiny model, register it, return the new version id."""
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    with mlflow.start_run():
        mlflow.set_tag("model_type", model_type)
        m = LogisticRegression()
        m.fit(X, y)
        mlflow.sklearn.log_model(
            sk_model=m,
            artifact_path="model",
            registered_model_name=REGISTRY_NAME,
        )

    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{REGISTRY_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    return str(latest.version)


def test_promote_moves_version_to_production(isolated_mlflow: str):
    version = _create_registered_version()
    promote_version(version=version, model_name=REGISTRY_NAME)

    client = MlflowClient()
    mv = client.get_model_version(REGISTRY_NAME, version)
    assert mv.current_stage == "Production"


def test_archive_existing_demotes_previous_production(isolated_mlflow: str):
    """Promoting a new version archives whatever was previously Production."""
    v1 = _create_registered_version()
    promote_version(version=v1, model_name=REGISTRY_NAME)

    v2 = _create_registered_version()
    promote_version(version=v2, model_name=REGISTRY_NAME)

    client = MlflowClient()
    mv1 = client.get_model_version(REGISTRY_NAME, v1)
    mv2 = client.get_model_version(REGISTRY_NAME, v2)
    assert mv1.current_stage == "Archived"
    assert mv2.current_stage == "Production"


def test_archive_existing_false_keeps_old_version_in_stage(isolated_mlflow: str):
    """Without archive, both versions can coexist in Production (rare but supported)."""
    v1 = _create_registered_version()
    promote_version(version=v1, model_name=REGISTRY_NAME)

    v2 = _create_registered_version()
    promote_version(version=v2, model_name=REGISTRY_NAME, archive_existing=False)

    client = MlflowClient()
    assert client.get_model_version(REGISTRY_NAME, v1).current_stage == "Production"
    assert client.get_model_version(REGISTRY_NAME, v2).current_stage == "Production"


def test_promote_to_staging(isolated_mlflow: str):
    version = _create_registered_version()
    promote_version(version=version, model_name=REGISTRY_NAME, stage="Staging")
    client = MlflowClient()
    assert client.get_model_version(REGISTRY_NAME, version).current_stage == "Staging"


def test_invalid_stage_raises(isolated_mlflow: str):
    with pytest.raises(ValueError, match="stage must be"):
        promote_version(version="1", model_name=REGISTRY_NAME, stage="Live")


def test_expected_model_type_mismatch_refuses(isolated_mlflow: str):
    """Safety check: if the version's model_type tag doesn't match, refuse."""
    version = _create_registered_version(model_type="logreg")
    with pytest.raises(ValueError, match="model_type"):
        promote_version(
            version=version,
            model_name=REGISTRY_NAME,
            expected_model_type="xgboost",
        )

    # Verify the version was *not* transitioned.
    client = MlflowClient()
    mv = client.get_model_version(REGISTRY_NAME, version)
    assert mv.current_stage == "None"


def test_expected_model_type_match_promotes(isolated_mlflow: str):
    version = _create_registered_version(model_type="xgboost")
    promote_version(
        version=version,
        model_name=REGISTRY_NAME,
        expected_model_type="xgboost",
    )
    client = MlflowClient()
    assert client.get_model_version(REGISTRY_NAME, version).current_stage == "Production"


def test_unknown_version_raises(isolated_mlflow: str):
    from mlflow.exceptions import MlflowException

    _create_registered_version()
    with pytest.raises(MlflowException):
        promote_version(version="999", model_name=REGISTRY_NAME)


def test_promote_does_not_emit_unsuppressed_warning(isolated_mlflow: str):
    """The deprecation warning from MLflow's stages API is suppressed locally."""
    version = _create_registered_version()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        promote_version(version=version, model_name=REGISTRY_NAME)

    relevant = [
        w
        for w in caught
        if issubclass(w.category, FutureWarning)
        and "transition_model_version_stage" in str(w.message)
    ]
    assert relevant == []
