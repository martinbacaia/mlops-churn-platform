"""Optuna study runner: optimize CV ROC-AUC, log every trial to MLflow.

One study per model. The objective is mean ROC-AUC across stratified k-fold
splits — chosen over single-holdout to make the optimization signal less
sensitive to one lucky fold and to match how production champion-selection
should reason about generalization.

MLflow shape per study::

    parent run  (run_name = "tune_<model>")
      ├── tag: phase=tuning, model_type=<model>
      ├── metric: best_roc_auc_cv
      ├── params: best_<param>=<value>
      └── nested run trial_0
          ├── params: <hyperparams sampled by Optuna>
          └── metric: roc_auc_cv_mean
          ...
"""

from __future__ import annotations

import argparse
from typing import Any

import mlflow
import numpy as np
import numpy.typing as npt
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from churn.config import get_settings
from churn.data.download import compute_md5, download_telco
from churn.data.ingest import ingest_to_splits
from churn.features.pipeline import (
    FEATURE_PIPELINE_VERSION,
    build_feature_pipeline,
)
from churn.logging_setup import configure_logging, get_logger
from churn.models.registry import MODEL_REGISTRY
from churn.training.mlflow_utils import configure_tracking, log_provenance_tags
from churn.training.tune_spaces import suggest_params

_log = get_logger(__name__)


def _make_objective(
    model_name: str,
    X: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.int64],
    cv_splits: int,
    random_state: int,
) -> Any:
    model_cls = MODEL_REGISTRY[model_name]
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(model_name, trial)
        scores: list[float] = []
        for train_idx, val_idx in skf.split(X, y):
            model = model_cls(**params)
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[val_idx])[:, 1]
            scores.append(float(roc_auc_score(y[val_idx], proba)))
        return float(np.mean(scores))

    return objective


def run_study(
    model_name: str,
    X: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.int64],
    n_trials: int = 30,
    cv_splits: int = 5,
    random_state: int = 42,
    dataset_md5: str | None = None,
) -> optuna.Study:
    """Run one Optuna study for ``model_name``; log all trials to MLflow.

    Each Optuna trial becomes a nested MLflow run inside one parent run named
    ``tune_<model>``. The parent run carries the best value and best params,
    so MLflow's UI groups the whole search under a single collapsible row.
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {model_name!r}. Known: {sorted(MODEL_REGISTRY.keys())}.")

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"tune_{model_name}",
    )

    objective = _make_objective(model_name, X, y, cv_splits, random_state)

    with configure_tracking(), mlflow.start_run(run_name=f"tune_{model_name}") as parent:
        log_provenance_tags(
            model_type=model_name,
            dataset_md5=dataset_md5,
            feature_pipeline_version=FEATURE_PIPELINE_VERSION,
        )
        mlflow.set_tag("phase", "tuning")
        mlflow.log_params({"n_trials": n_trials, "cv_splits": cv_splits})

        def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(trial.params)
                if trial.value is not None:
                    mlflow.log_metric("roc_auc_cv_mean", trial.value)

        study.optimize(objective, n_trials=n_trials, callbacks=[log_trial])

        mlflow.log_metric("best_roc_auc_cv", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

        _log.info(
            "tuning_completed",
            model=model_name,
            n_trials=n_trials,
            best_roc_auc_cv=study.best_value,
            parent_run_id=parent.info.run_id,
        )

    return study


def main(
    model_name: str = "all",
    n_trials: int = 30,
    cv_splits: int = 5,
) -> dict[str, optuna.Study]:
    """CLI entry point: tune one model (or all) with the canonical ingest path."""
    configure_logging()
    settings = get_settings()

    raw_path = download_telco()
    dataset_md5 = compute_md5(raw_path)
    splits = ingest_to_splits(raw_path=raw_path, random_state=settings.random_state)

    pipeline = build_feature_pipeline().fit(splits.X_train)
    X_full = pipeline.transform(splits.X_train.copy().reset_index(drop=True)).astype(np.float64)
    y_full = splits.y_train.to_numpy().astype(np.int64)

    targets = list(MODEL_REGISTRY.keys()) if model_name == "all" else [model_name]
    return {
        name: run_study(
            name,
            X_full,
            y_full,
            n_trials=n_trials,
            cv_splits=cv_splits,
            random_state=settings.random_state,
            dataset_md5=dataset_md5,
        )
        for name in targets
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune one or all registered models.")
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", *MODEL_REGISTRY.keys()],
        help="Which model to tune (default: all three).",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--cv-splits", type=int, default=5)
    args = parser.parse_args()
    main(model_name=args.model, n_trials=args.n_trials, cv_splits=args.cv_splits)
