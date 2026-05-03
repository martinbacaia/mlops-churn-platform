"""End-to-end training orchestrator: fit every model in the registry, log to MLflow.

The function reads as the platform's claim in code: iterate over
``MODEL_REGISTRY``, train each entry on the same splits with the same fitted
feature pipeline, log the same metrics, register every result under the same
``churn_classifier`` registry name.

Anything model-specific lives inside the model class itself; ``train_all_models``
is identical for LogReg, XGBoost, TabularMLP, and any future fourth implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer

from churn.config import get_settings
from churn.data.download import compute_md5, download_telco
from churn.data.ingest import ingest_to_splits
from churn.data.splits import Splits
from churn.features.pipeline import (
    FEATURE_PIPELINE_VERSION,
    build_feature_pipeline,
    save_pipeline,
)
from churn.logging_setup import configure_logging, get_logger
from churn.models.registry import MODEL_REGISTRY
from churn.training.metrics import compute_classification_metrics
from churn.training.mlflow_utils import (
    configure_tracking,
    log_provenance_tags,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_log = get_logger(__name__)


def train_all_models(
    splits: Splits,
    feature_pipeline: ColumnTransformer,
    dataset_md5: str | None = None,
    register: bool = True,
    requirements_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Train every registered model, evaluate on val + test, log artifacts.

    Args:
        splits: Output of :func:`churn.data.ingest.ingest_to_splits`.
        feature_pipeline: A *fitted* ColumnTransformer; the same instance is
            applied to all three models so comparisons are apples-to-apples.
        dataset_md5: Optional dataset hash; logged as a run tag.
        register: If True, log each model under the registered name from
            ``Settings.model_name``, creating a new version per call. The
            promotion to ``Production`` is handled by ``promote.py``, not here.
        requirements_path: Optional ``requirements.txt`` to attach as artifact;
            preserves the exact dep set the model was trained with.

    Returns:
        ``{model_name: {"run_id": ..., "val": {...}, "test": {...}}}``
    """
    settings = get_settings()
    X_train_t = feature_pipeline.transform(splits.X_train)
    X_val_t = feature_pipeline.transform(splits.X_val)
    X_test_t = feature_pipeline.transform(splits.X_test)
    y_train = splits.y_train.to_numpy()
    y_val = splits.y_val.to_numpy()
    y_test = splits.y_test.to_numpy()

    # Persist the feature pipeline once; every run re-uses this artifact.
    pipe_artifact = Path("feature_pipeline.joblib")
    save_pipeline(feature_pipeline, pipe_artifact)

    results: dict[str, dict[str, Any]] = {}

    try:
        with configure_tracking():
            for model_name, model_cls in MODEL_REGISTRY.items():
                model = model_cls()

                with mlflow.start_run(run_name=model_name) as run:
                    log_provenance_tags(
                        model_type=model_name,
                        dataset_md5=dataset_md5,
                        feature_pipeline_version=FEATURE_PIPELINE_VERSION,
                    )
                    mlflow.log_params(model.get_params())

                    model.fit(X_train_t, y_train)

                    val_metrics = compute_classification_metrics(
                        y_val, model.predict_proba(X_val_t)
                    )
                    test_metrics = compute_classification_metrics(
                        y_test, model.predict_proba(X_test_t)
                    )
                    mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
                    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

                    mlflow.log_artifact(str(pipe_artifact), artifact_path="feature_pipeline")
                    if requirements_path is not None and requirements_path.exists():
                        mlflow.log_artifact(str(requirements_path))

                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name=settings.model_name if register else None,
                    )

                    results[model_name] = {
                        "run_id": run.info.run_id,
                        "val": val_metrics,
                        "test": test_metrics,
                    }
                    _log.info(
                        "model_trained",
                        model=model_name,
                        run_id=run.info.run_id,
                        val_roc_auc=val_metrics["roc_auc"],
                        test_roc_auc=test_metrics["roc_auc"],
                    )
    finally:
        pipe_artifact.unlink(missing_ok=True)

    return results


def main(register: bool = True) -> dict[str, dict[str, Any]]:
    """CLI entry point: download → ingest → fit pipeline → train all models."""
    configure_logging()
    settings = get_settings()

    raw_path = download_telco()
    dataset_md5 = compute_md5(raw_path)

    splits = ingest_to_splits(raw_path=raw_path, random_state=settings.random_state)
    feature_pipeline = build_feature_pipeline().fit(splits.X_train)

    return train_all_models(
        splits=splits,
        feature_pipeline=feature_pipeline,
        dataset_md5=dataset_md5,
        register=register,
        requirements_path=PROJECT_ROOT / "requirements.txt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train every registered model.")
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip MLflow Model Registry registration (keeps runs but no versions).",
    )
    args = parser.parse_args()
    main(register=not args.no_register)
