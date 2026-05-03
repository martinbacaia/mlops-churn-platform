"""MLflow tracking helpers for explicit (non-autolog) logging.

Autolog hides what gets recorded; on a multi-model platform that's a debugging
disaster — when two runs disagree, you can't tell whether the divergence comes
from the model, the data, or a logged-tag-you-didn't-set. Every tag and metric
this platform records is set by code we control.

The standard run shape produced by the orchestrator is::

    Tags:    model_type, dataset_md5, feature_pipeline_version, git_sha
    Params:  the model's get_params() output
    Metrics: val_<name>, test_<name>  for the canonical metric set
    Artifacts: the model (sklearn flavor), the feature pipeline (joblib),
               and requirements.txt (full pinned dep list)

The tracking URI defaults to ``sqlite:///mlruns/mlflow.db`` from settings;
override via ``MLFLOW_TRACKING_URI`` to point at a remote server in production.
"""

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Iterator

import mlflow

from churn.config import get_settings
from churn.logging_setup import get_logger

_log = get_logger(__name__)


def get_git_sha() -> str | None:
    """Return the short HEAD SHA of the current repo, or ``None`` if unavailable.

    Used as a tag so every MLflow run links back to the exact code revision
    that produced it. Failure is non-fatal — running outside a git checkout
    (e.g. in a container build) just elides the tag.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def get_or_create_experiment(name: str) -> str:
    """Return the experiment ID for ``name``, creating it if absent."""
    existing = mlflow.get_experiment_by_name(name)
    if existing is not None:
        return str(existing.experiment_id)
    return str(mlflow.create_experiment(name))


@contextlib.contextmanager
def configure_tracking() -> Iterator[None]:
    """Wire MLflow to the configured tracking URI and experiment for the block.

    Idempotent: safe to nest or call repeatedly within one process.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    get_or_create_experiment(settings.mlflow_experiment_name)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    _log.info(
        "mlflow_tracking_configured",
        tracking_uri=settings.mlflow_tracking_uri,
        experiment=settings.mlflow_experiment_name,
    )
    yield


def log_provenance_tags(
    model_type: str,
    dataset_md5: str | None = None,
    feature_pipeline_version: str | None = None,
) -> None:
    """Stamp the active run with the standard provenance tags.

    Must be called inside an ``mlflow.start_run()`` block. Missing optional
    values are skipped rather than logged as ``"None"`` strings — the absence
    of a tag is more honest than a literal None.
    """
    tags: dict[str, str] = {"model_type": model_type}
    if dataset_md5:
        tags["dataset_md5"] = dataset_md5
    if feature_pipeline_version:
        tags["feature_pipeline_version"] = feature_pipeline_version
    git_sha = get_git_sha()
    if git_sha:
        tags["git_sha"] = git_sha
    mlflow.set_tags(tags)
