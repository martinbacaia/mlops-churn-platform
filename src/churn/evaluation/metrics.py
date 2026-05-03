"""Threshold-aware diagnostics: precision/recall sweep, calibration, confusion matrix.

These functions extend the point metrics in :mod:`churn.training.metrics` with
the operating-point and calibration views you need to *choose* a threshold or
*trust* a probability. The output is always a tidy ``pd.DataFrame`` (or a small
dict) — easy to log to MLflow, render in the README, or feed into a plotting
layer later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix


def _validate_proba(y_proba: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    if y_proba.ndim != 2 or y_proba.shape[1] != 2:
        raise ValueError(f"y_proba must have shape (n, 2); got {y_proba.shape}.")
    return y_proba


def threshold_sweep(
    y_true: npt.ArrayLike,
    y_proba: npt.NDArray[np.floating[Any]],
    thresholds: npt.NDArray[np.floating[Any]] | None = None,
) -> pd.DataFrame:
    """Compute precision / recall / F1 + confusion-matrix counts at every threshold.

    Args:
        y_true: Ground-truth labels (0/1), shape ``(n,)``.
        y_proba: Predicted probabilities, shape ``(n, 2)``.
        thresholds: Decision points to evaluate. Defaults to 19 evenly-spaced
            values in ``[0.05, 0.95]``, which is enough granularity for a
            decision-curve plot without overcrowding the README table.

    Returns:
        DataFrame with one row per threshold and columns
        ``threshold, precision, recall, f1, tp, fp, fn, tn``.
    """
    _validate_proba(y_proba)
    y_arr = np.asarray(y_true)
    y_pos = y_proba[:, 1]
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    rows: list[dict[str, float]] = []
    for t in thresholds:
        y_pred = (y_pos >= t).astype(np.int64)
        tn, fp, fn, tp = _sk_confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }
        )
    return pd.DataFrame(rows)


def calibration_curve_data(
    y_true: npt.ArrayLike,
    y_proba: npt.NDArray[np.floating[Any]],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability-diagram data: per-bin predicted vs observed positive rate.

    Returns:
        DataFrame with columns ``bin_lower, bin_upper, predicted_mean,
        observed_rate, count``. Empty bins keep their row but with NaN means
        and ``count = 0`` — preserving them makes the diagram honest about
        coverage gaps.
    """
    _validate_proba(y_proba)
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}.")

    y_arr = np.asarray(y_true)
    y_pos = y_proba[:, 1]
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows: list[dict[str, float]] = []
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_pos >= low) & (y_pos <= high)
        else:
            mask = (y_pos >= low) & (y_pos < high)
        count = int(mask.sum())
        if count > 0:
            predicted_mean = float(y_pos[mask].mean())
            observed_rate = float(y_arr[mask].mean())
        else:
            predicted_mean = float("nan")
            observed_rate = float("nan")
        rows.append(
            {
                "bin_lower": float(low),
                "bin_upper": float(high),
                "predicted_mean": predicted_mean,
                "observed_rate": observed_rate,
                "count": count,
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix_at(
    y_true: npt.ArrayLike,
    y_proba: npt.NDArray[np.floating[Any]],
    threshold: float = 0.5,
) -> dict[str, int]:
    """Confusion-matrix counts at a single threshold.

    Returns a dict with keys ``tp, fp, fn, tn``. Always uses ``labels=[0, 1]``
    so the order of returned counts is stable even when one class is absent
    from the predictions.
    """
    _validate_proba(y_proba)
    y_arr = np.asarray(y_true)
    y_pos = y_proba[:, 1]
    y_pred = (y_pos >= threshold).astype(np.int64)
    tn, fp, fn, tp = _sk_confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
