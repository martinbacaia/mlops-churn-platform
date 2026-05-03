"""Promote a registered model version to a target stage (default ``Production``).

The promotion CLI is intentionally model-agnostic: every concrete classifier
ends up registered under the same name (:attr:`Settings.model_name`, default
``churn_classifier``), so promoting "the next version" never depends on which
runtime the version contains. The ``--model`` flag is a safety check, not a
selector — when provided, it verifies the version's ``model_type`` tag matches
before transitioning, so a typo can't put logreg version 3 into Production
when you meant xgboost version 3.

Stage transitions use ``archive_existing_versions=True`` so at most one
version per stage is active. This is the discipline that makes the serving
layer's ``models:/churn_classifier/Production`` URI unambiguous.
"""

from __future__ import annotations

import argparse
import warnings

from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from churn.config import get_settings
from churn.logging_setup import configure_logging, get_logger
from churn.training.mlflow_utils import configure_tracking

VALID_STAGES = ("Production", "Staging", "Archived", "None")
_log = get_logger(__name__)


def promote_version(
    version: int | str,
    model_name: str | None = None,
    stage: str = "Production",
    archive_existing: bool = True,
    expected_model_type: str | None = None,
) -> ModelVersion:
    """Move a registered model version to ``stage``.

    Args:
        version: Numeric version id of the registered model.
        model_name: Registered model name. Defaults to ``Settings.model_name``.
        stage: Target stage. One of :data:`VALID_STAGES`.
        archive_existing: If True, every other version currently in ``stage``
            is moved to ``Archived``, guaranteeing a single live version.
        expected_model_type: Optional safety check — fail if the version's
            ``model_type`` tag (set during training) does not match. Useful for
            ``make promote MODEL=xgboost VERSION=3``: a typo in VERSION won't
            quietly promote a different model family.

    Returns:
        The transitioned :class:`ModelVersion`.
    """
    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of {VALID_STAGES}; got {stage!r}.")

    settings = get_settings()
    name = model_name or settings.model_name
    client = MlflowClient()

    mv = client.get_model_version(name=name, version=str(version))

    if expected_model_type is not None:
        run = client.get_run(mv.run_id)
        actual = run.data.tags.get("model_type")
        if actual != expected_model_type:
            raise ValueError(
                f"Version {version} of {name!r} has model_type={actual!r}; "
                f"expected {expected_model_type!r}. Refusing to promote."
            )

    # ``transition_model_version_stage`` emits a FutureWarning starting with
    # MLflow 2.9 (aliases are the new path). The behavior still works and the
    # spec is explicit about stages, so we suppress the warning locally.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        promoted: ModelVersion = client.transition_model_version_stage(
            name=name,
            version=str(version),
            stage=stage,
            archive_existing_versions=archive_existing,
        )

    _log.info(
        "model_promoted",
        model_name=name,
        version=str(version),
        stage=stage,
        archive_existing=archive_existing,
        run_id=mv.run_id,
    )
    return promoted


def main(
    version: int,
    model_name: str | None = None,
    stage: str = "Production",
    expected_model_type: str | None = None,
) -> ModelVersion:
    configure_logging()
    with configure_tracking():
        return promote_version(
            version=version,
            model_name=model_name,
            stage=stage,
            expected_model_type=expected_model_type,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Promote a registered model version to a stage. "
            "Use --model to assert the version's model_type before transitioning."
        )
    )
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Registered model name (defaults to Settings.model_name).",
    )
    parser.add_argument("--stage", default="Production", choices=VALID_STAGES)
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model_type safety check (logreg / xgboost / tabular_mlp).",
    )
    args = parser.parse_args()
    main(
        version=args.version,
        model_name=args.model_name,
        stage=args.stage,
        expected_model_type=args.model,
    )
