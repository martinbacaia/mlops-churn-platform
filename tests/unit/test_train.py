from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import mlflow
import pytest

from churn.data.ingest import load_raw, preprocess
from churn.data.splits import make_splits
from churn.features.pipeline import build_feature_pipeline
from churn.models.logreg import LogRegModel
from churn.models.tabular_mlp import TabularMLPModel
from churn.models.xgboost_model import XGBoostModel
from churn.training.train import train_all_models

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "telco_sample.csv"


@pytest.fixture
def isolated_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    db = tmp_path / "mlruns.db"
    tracking_uri = f"sqlite:///{db}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_train")
    monkeypatch.setenv("MODEL_NAME", "test_churn_classifier")
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri


@pytest.fixture(scope="module")
def small_splits_and_pipeline():
    """Splits + fitted pipeline backed by the 200-row fixture."""
    df = preprocess(load_raw(FIXTURE_PATH))
    splits = make_splits(df, test_size=0.2, val_size=0.2, random_state=42)
    pipeline = build_feature_pipeline().fit(splits.X_train)
    return splits, pipeline


# Cheap defaults so the orchestrator test stays under ~30 s while still
# exercising the *real* model code paths (no mocks).
_FAST_REGISTRY = {
    "logreg": LogRegModel,
    "xgboost": lambda: XGBoostModel(n_estimators=10),
    "tabular_mlp": lambda: TabularMLPModel(epochs=1, batch_size=64),
}


def test_full_training_run_produces_a_complete_record(
    isolated_mlflow: str, small_splits_and_pipeline
):
    """One end-to-end training pass, then assert every invariant at once.

    Bundling the assertions saves ~40s vs. one test per invariant — each
    test would otherwise fit three models, and the model-fit cost dominates
    every other line. A single rich assertion block is also closer to how a
    reviewer would inspect the run themselves.
    """
    splits, pipeline = small_splits_and_pipeline
    with patch("churn.training.train.MODEL_REGISTRY", _FAST_REGISTRY):
        results = train_all_models(
            splits=splits,
            feature_pipeline=pipeline,
            dataset_md5="abc123",
            register=True,
        )

    # Result shape: one entry per registered model, each carrying its run id
    # and both metric blocks.
    assert set(results.keys()) == {"logreg", "xgboost", "tabular_mlp"}

    expected_metric_keys = {
        "val_roc_auc",
        "val_pr_auc",
        "val_f1",
        "val_log_loss",
        "val_brier",
        "test_roc_auc",
        "test_pr_auc",
        "test_f1",
        "test_log_loss",
        "test_brier",
    }

    for model_name, payload in results.items():
        assert {"run_id", "val", "test"}.issubset(payload.keys())

        run = mlflow.get_run(payload["run_id"])
        # Provenance tags
        assert run.data.tags["model_type"] == model_name
        assert run.data.tags["dataset_md5"] == "abc123"
        assert run.data.tags["feature_pipeline_version"] == "v1"
        # Canonical metric set on both splits
        assert expected_metric_keys.issubset(run.data.metrics.keys())
        # Hyperparams logged
        assert run.data.params, "Hyperparameters were not logged"

    # Registry contract: all three models live under one registered name
    # (the model-agnostic claim of the platform).
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name = 'test_churn_classifier'")
    assert len(versions) == 3
    assert {v.name for v in versions} == {"test_churn_classifier"}

    # Round-trip a logged model: it loads back and predicts the same shape.
    loaded = mlflow.sklearn.load_model(f"runs:/{results['logreg']['run_id']}/model")
    proba = loaded.predict_proba(pipeline.transform(splits.X_test))
    assert proba.shape == (len(splits.X_test), 2)


def test_register_false_skips_registry_versions(isolated_mlflow: str, small_splits_and_pipeline):
    """``register=False`` keeps runs but does not create registered model versions."""
    splits, pipeline = small_splits_and_pipeline
    # Single fastest model is enough — the branch we're testing is independent of model.
    with patch("churn.training.train.MODEL_REGISTRY", {"logreg": LogRegModel}):
        results = train_all_models(splits=splits, feature_pipeline=pipeline, register=False)

    assert "logreg" in results
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name = 'test_churn_classifier'")
    assert versions == []
