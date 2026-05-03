"""Sklearn ColumnTransformer wiring all features for the churn task.

The pipeline is **single, shared, and serializable** — every model (LogReg,
XGBoost, TabularMLP) consumes the same fitted transform output, so model
comparisons are apples-to-apples and a model swap in production never silently
shifts the input distribution.

Topology (a multi-branch ``ColumnTransformer``):

* **numerical** → ``StandardScaler`` over ``[tenure, MonthlyCharges, TotalCharges,
  SeniorCitizen]``. ``SeniorCitizen`` is already 0/1 but goes through the same
  scaler so the MLP sees a consistent value range.
* **tenure_bucket** → ``Pipeline(TenureBucketizer → OneHotEncoder)`` over
  ``tenure``. The same column also feeds the numerical branch — buckets capture
  customer-lifecycle nonlinearities that linear models would otherwise miss.
* **categorical** → ``OneHotEncoder`` over the remaining string columns. We
  use ``handle_unknown="ignore"`` so an unseen category at inference time
  zero-encodes instead of crashing the request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.features.transformers import TenureBucketizer

FEATURE_PIPELINE_VERSION: Final[str] = "v1"

NUMERICAL_COLUMNS: Final[list[str]] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

CATEGORICAL_COLUMNS: Final[list[str]] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def build_feature_pipeline() -> ColumnTransformer:
    """Construct the feature pipeline; call ``.fit(X)`` on training data."""
    tenure_bucket_branch = Pipeline(
        steps=[
            ("bucketize", TenureBucketizer()),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=float,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), NUMERICAL_COLUMNS),
            ("tenure_bucket", tenure_bucket_branch, ["tenure"]),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=float,
                ),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
        # Dense output keeps downstream models simple (notably PyTorch, which
        # has no native sparse-matrix path on tabular data).
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def save_pipeline(pipeline: ColumnTransformer, path: Path) -> Path:
    """Serialize a fitted pipeline to ``path`` (joblib format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(path: Path) -> ColumnTransformer:
    """Load a fitted pipeline previously written by :func:`save_pipeline`."""
    obj = joblib.load(path)
    if not isinstance(obj, ColumnTransformer):
        raise TypeError(f"Expected ColumnTransformer at {path}, got {type(obj).__name__}.")
    return obj
