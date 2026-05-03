"""Classification metrics computed uniformly across every model implementation.

The training loop and the evaluation layer share this function: same metric
definitions, same float64 outputs, so a metric drift between two runs always
points at the model — never at the metric pipeline.

The metric set covers the three things that matter for churn:

* **roc_auc** — ranking quality (threshold-independent).
* **pr_auc** — ranking quality on the minority class (more informative than
  ROC-AUC under class imbalance).
* **f1** — operating-point quality at a chosen threshold.
* **log_loss** — calibration / probabilistic quality.
* **brier** — calibration score; the closer to zero, the better the
  probabilities reflect reality.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: npt.ArrayLike,
    y_proba: npt.NDArray[np.floating[Any]],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the platform's canonical classification metric set.

    Args:
        y_true: Ground-truth labels, shape ``(n,)`` with values in ``{0, 1}``.
        y_proba: Predicted probabilities, shape ``(n, 2)`` exactly as returned
            by ``Model.predict_proba`` — column 1 is the positive class.
        threshold: Decision threshold used for ``f1`` only. Defaults to 0.5;
            the evaluation layer sweeps this for the comparison table.

    Returns:
        Mapping of metric name → float. Always returns the same five keys
        regardless of input — callers can rely on the schema for dashboards
        and MLflow run comparisons.
    """
    y_arr = np.asarray(y_true)
    if y_proba.ndim != 2 or y_proba.shape[1] != 2:
        raise ValueError(f"y_proba must have shape (n, 2); got {y_proba.shape}.")
    if y_proba.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"y_proba and y_true row counts disagree: " f"{y_proba.shape[0]} vs {y_arr.shape[0]}."
        )

    y_pos = y_proba[:, 1]
    y_pred = (y_pos >= threshold).astype(np.int64)

    return {
        "roc_auc": float(roc_auc_score(y_arr, y_pos)),
        "pr_auc": float(average_precision_score(y_arr, y_pos)),
        "f1": float(f1_score(y_arr, y_pred, zero_division=0.0)),
        "log_loss": float(log_loss(y_arr, y_proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_arr, y_pos)),
    }
