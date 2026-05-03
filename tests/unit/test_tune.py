from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pytest

from churn.data.ingest import load_raw, preprocess
from churn.data.splits import make_splits
from churn.features.pipeline import build_feature_pipeline
from churn.training.tune import run_study

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "telco_sample.csv"


@pytest.fixture
def isolated_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    db = tmp_path / "mlruns.db"
    tracking_uri = f"sqlite:///{db}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_tune")
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri


@pytest.fixture(scope="module")
def feature_matrix():
    df = preprocess(load_raw(FIXTURE_PATH))
    splits = make_splits(df, random_state=42)
    pipeline = build_feature_pipeline().fit(splits.X_train)
    X = pipeline.transform(splits.X_train).astype(np.float64)
    y = splits.y_train.to_numpy().astype(np.int64)
    return X, y


def test_run_study_returns_optuna_study_with_best_value(isolated_mlflow: str, feature_matrix):
    X, y = feature_matrix
    study = run_study("logreg", X, y, n_trials=2, cv_splits=2, dataset_md5="md5")
    assert isinstance(study, optuna.Study)
    assert study.best_value is not None
    assert 0.0 <= study.best_value <= 1.0
    # Two trials → two completed FrozenTrial entries.
    assert len(study.trials) == 2


def test_unknown_model_name_raises():
    with pytest.raises(KeyError, match="Unknown model"):
        run_study("transformer", np.zeros((4, 4)), np.array([0, 1, 0, 1]))


def test_parent_run_logs_best_value_and_nested_trials(isolated_mlflow: str, feature_matrix):
    X, y = feature_matrix
    run_study("logreg", X, y, n_trials=2, cv_splits=2, dataset_md5="md5sum")

    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name("test_tune")
    assert experiment is not None
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'tune_logreg'",
    )
    assert len(runs) == 1
    parent = runs[0]

    # Parent run carries the canonical tuning summary.
    assert parent.data.tags["phase"] == "tuning"
    assert parent.data.tags["model_type"] == "logreg"
    assert parent.data.tags["dataset_md5"] == "md5sum"
    assert "best_roc_auc_cv" in parent.data.metrics
    assert any(k.startswith("best_") for k in parent.data.params)

    # Two nested runs (one per trial), each with the per-trial metric.
    nested = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
    )
    assert len(nested) == 2
    for trial_run in nested:
        assert "roc_auc_cv_mean" in trial_run.data.metrics


def test_run_study_is_deterministic_given_seed(isolated_mlflow: str, feature_matrix):
    X, y = feature_matrix
    a = run_study("logreg", X, y, n_trials=3, cv_splits=2, random_state=7)
    b = run_study("logreg", X, y, n_trials=3, cv_splits=2, random_state=7)
    # TPESampler with the same seed + same data → same best params.
    assert a.best_params == b.best_params
    assert a.best_value == pytest.approx(b.best_value, abs=1e-9)
