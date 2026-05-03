"""Cross-model comparison report — quality, latency, and size side-by-side.

The output is the table that goes into the README's "Evaluation results"
section. It collapses the three models into one DataFrame so a reader can
form an opinion in a single glance — no clicking through three runs in
MLflow, no eyeballing scattered numbers.

Latency is measured single-row to mirror the online serving path; batched
throughput is a different question and would deserve its own column. The
serialized size is computed via joblib (the same flavor used for both the
``feature_pipeline.joblib`` artifact and ``mlflow.sklearn.log_model``), so
the reported number reflects what an MLflow Model Registry version actually
occupies.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

from churn.models.base import Model
from churn.training.metrics import compute_classification_metrics


def _measure_inference_latency_ms(
    model: Model,
    X: npt.NDArray[np.floating[Any]],
    n_samples: int = 100,
) -> dict[str, float]:
    """Single-row inference latency percentiles, in milliseconds.

    Warm-up call is excluded — first calls under PyTorch / XGBoost can be
    artificially slow because of lazy kernel loading. We care about steady-state
    serving cost, which is what matters when an HTTP endpoint hands the model
    one row at a time.
    """
    one_row = X[:1]
    model.predict_proba(one_row)  # warm-up; discarded

    timings_ms: list[float] = []
    for _ in range(n_samples):
        start = time.perf_counter_ns()
        model.predict_proba(one_row)
        timings_ms.append((time.perf_counter_ns() - start) / 1e6)
    timings_ms.sort()

    return {
        "p50_ms": timings_ms[len(timings_ms) // 2],
        "p95_ms": timings_ms[int(len(timings_ms) * 0.95)],
        "p99_ms": timings_ms[int(len(timings_ms) * 0.99)],
    }


def _serialized_size_kb(model: Model) -> float:
    """KB on disk after ``joblib.dump`` — the size MLflow's sklearn flavor stores."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
        path = Path(tmp.name)
    try:
        joblib.dump(model, path)
        return path.stat().st_size / 1024.0
    finally:
        path.unlink(missing_ok=True)


COMPARE_COLUMNS = [
    "roc_auc",
    "pr_auc",
    "f1",
    "log_loss",
    "brier",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "size_kb",
]


def compare_models(
    fitted_models: dict[str, Model],
    X: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.int64],
    n_latency_samples: int = 100,
) -> pd.DataFrame:
    """Side-by-side comparison: quality metrics + latency + size, one row per model.

    Args:
        fitted_models: Dict ``{name: fitted Model}``. Names become the index.
        X: Holdout feature matrix (post feature-pipeline transform).
        y: Holdout labels.
        n_latency_samples: How many single-row predictions to time per model.

    Returns:
        DataFrame indexed by model name, with columns
        ``roc_auc, pr_auc, f1, log_loss, brier, p50_ms, p95_ms, p99_ms, size_kb``.
    """
    if not fitted_models:
        raise ValueError("compare_models requires at least one fitted model.")

    rows: list[dict[str, Any]] = []
    for name, model in fitted_models.items():
        proba = model.predict_proba(X)
        metrics = compute_classification_metrics(y, proba)
        latency = _measure_inference_latency_ms(model, X, n_samples=n_latency_samples)
        rows.append(
            {
                "model": name,
                **metrics,
                **latency,
                "size_kb": _serialized_size_kb(model),
            }
        )

    return pd.DataFrame(rows).set_index("model")[COMPARE_COLUMNS]
