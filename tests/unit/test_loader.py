from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import mlflow
import pytest
from sklearn.compose import ColumnTransformer

from churn.data.ingest import load_raw, preprocess
from churn.data.splits import make_splits
from churn.features.pipeline import build_feature_pipeline
from churn.models.base import Model
from churn.models.logreg import LogRegModel
from churn.serving.loader import (
    NoActiveModelError,
    ProductionArtifacts,
    load_production_artifacts,
)
from churn.training.promote import promote_version
from churn.training.train import train_all_models

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "telco_sample.csv"
REGISTRY_NAME = "test_serving_classifier"


@pytest.fixture
def isolated_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    db = tmp_path / "mlruns.db"
    tracking_uri = f"sqlite:///{db}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_serving")
    monkeypatch.setenv("MODEL_NAME", REGISTRY_NAME)
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri


@pytest.fixture
def production_setup(isolated_mlflow: str):
    """Train one model, register it, promote it to Production. Yields version."""
    df = preprocess(load_raw(FIXTURE_PATH))
    splits = make_splits(df, random_state=42)
    pipeline = build_feature_pipeline().fit(splits.X_train)

    with patch("churn.training.train.MODEL_REGISTRY", {"logreg": LogRegModel}):
        results = train_all_models(
            splits=splits,
            feature_pipeline=pipeline,
            dataset_md5="test-md5",
            register=True,
        )

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name = '{REGISTRY_NAME}'")
    # MLflow's ModelVersion.version oscillates between int and str across
    # versions; normalize to str (which is what the loader returns).
    version = str(max(versions, key=lambda v: int(v.version)).version)
    promote_version(version=version, model_name=REGISTRY_NAME)
    return {"version": version, "results": results}


def test_loader_returns_production_artifacts(production_setup):
    artifacts = load_production_artifacts(model_name=REGISTRY_NAME)
    assert isinstance(artifacts, ProductionArtifacts)
    assert isinstance(artifacts.model, Model)
    assert isinstance(artifacts.feature_pipeline, ColumnTransformer)


def test_loader_metadata_matches_registered_version(production_setup):
    artifacts = load_production_artifacts(model_name=REGISTRY_NAME)
    assert artifacts.registered_model_name == REGISTRY_NAME
    assert artifacts.model_version == production_setup["version"]
    assert artifacts.stage == "Production"
    assert artifacts.model_type == "logreg"
    assert artifacts.feature_pipeline_version == "v1"


def test_loader_predicts_end_to_end(production_setup):
    """Loaded model + pipeline must transform raw rows and predict (n, 2) probas."""
    df = preprocess(load_raw(FIXTURE_PATH))
    splits = make_splits(df, random_state=42)
    raw_one = splits.X_test.iloc[:3]

    artifacts = load_production_artifacts(model_name=REGISTRY_NAME)
    X_t = artifacts.feature_pipeline.transform(raw_one)
    proba = artifacts.model.predict_proba(X_t)
    assert proba.shape == (3, 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_loader_no_production_version_raises(isolated_mlflow: str):
    """Without any registered version, the loader refuses to start."""
    with pytest.raises(NoActiveModelError):
        load_production_artifacts(model_name=REGISTRY_NAME)


def test_loader_respects_stage_argument(production_setup):
    """Querying Staging when only Production is populated returns NoActiveModelError."""
    with pytest.raises(NoActiveModelError):
        load_production_artifacts(model_name=REGISTRY_NAME, stage="Staging")
