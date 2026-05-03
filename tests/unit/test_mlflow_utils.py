from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest

from churn.training.mlflow_utils import (
    configure_tracking,
    get_git_sha,
    get_or_create_experiment,
    log_provenance_tags,
)


@pytest.fixture
def isolated_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Point MLflow at a per-test SQLite store so runs don't pollute each other."""
    db = tmp_path / "mlruns.db"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    tracking_uri = f"sqlite:///{db}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_experiment")
    # MLflow caches the active client; reset state between tests.
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri


def test_get_git_sha_returns_string_or_none():
    """Outcome depends on whether tests run inside a git checkout."""
    sha = get_git_sha()
    assert sha is None or (isinstance(sha, str) and len(sha) >= 4)


def test_get_or_create_experiment_creates_when_absent(isolated_mlflow: str):
    exp_id = get_or_create_experiment("brand_new_exp")
    assert exp_id is not None
    assert mlflow.get_experiment(exp_id).name == "brand_new_exp"


def test_get_or_create_experiment_returns_existing_id(isolated_mlflow: str):
    first = get_or_create_experiment("dup")
    second = get_or_create_experiment("dup")
    assert first == second


def test_configure_tracking_sets_uri_and_experiment(isolated_mlflow: str):
    with configure_tracking():
        assert mlflow.get_tracking_uri() == isolated_mlflow
        active = mlflow.get_experiment_by_name("test_experiment")
        assert active is not None


def test_configure_tracking_is_idempotent(isolated_mlflow: str):
    with configure_tracking():
        pass
    with configure_tracking():  # second entry must not raise
        pass


def test_log_provenance_tags_writes_only_set_values(isolated_mlflow: str):
    with configure_tracking(), mlflow.start_run() as run:
        log_provenance_tags(model_type="logreg", dataset_md5="abc123")
        run_id = run.info.run_id

    fetched = mlflow.get_run(run_id)
    assert fetched.data.tags["model_type"] == "logreg"
    assert fetched.data.tags["dataset_md5"] == "abc123"
    # Optional tag was not provided, so it should not appear.
    assert "feature_pipeline_version" not in fetched.data.tags


def test_log_provenance_tags_includes_all_provided(isolated_mlflow: str):
    with configure_tracking(), mlflow.start_run() as run:
        log_provenance_tags(
            model_type="xgboost",
            dataset_md5="md5sum",
            feature_pipeline_version="v1",
        )
        run_id = run.info.run_id

    tags = mlflow.get_run(run_id).data.tags
    assert tags["model_type"] == "xgboost"
    assert tags["dataset_md5"] == "md5sum"
    assert tags["feature_pipeline_version"] == "v1"


def test_log_provenance_tags_skips_none_values(isolated_mlflow: str):
    """Don't pollute the tag store with the literal string 'None'."""
    with configure_tracking(), mlflow.start_run() as run:
        log_provenance_tags(model_type="tabular_mlp", dataset_md5=None)
        run_id = run.info.run_id

    tags = mlflow.get_run(run_id).data.tags
    assert tags["model_type"] == "tabular_mlp"
    assert "dataset_md5" not in tags
