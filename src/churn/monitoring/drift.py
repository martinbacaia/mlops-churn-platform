"""Distribution-shift detectors: PSI, KS test, prediction drift.

We compute drift statistics rather than dragging in a heavy library. The math
is short, the implementation is auditable, and a single docstring per function
documents exactly what's being measured. The trade-off is that we don't get
e.g. Evidently's pretty dashboards out of the box — we render our own JSON +
HTML in :mod:`churn.monitoring.report`.

Conventions:

* **PSI** (Population Stability Index): the de-facto industry threshold is
  ``0.2`` — anything above signals "investigate". We expose the threshold as a
  parameter so the README's drift example can demonstrate both states.
* **KS test**: returns the standard ``(statistic, p_value)``. ``p < 0.05`` is
  the conventional alert threshold.
* The DataFrame produced by :func:`detect_drift` is the canonical "feature
  drift table" — sorted by descending PSI so the worst offenders appear first.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import ks_2samp

DEFAULT_PSI_THRESHOLD: float = 0.2
DEFAULT_KS_ALPHA: float = 0.05
_EPSILON: float = 1e-6


def population_stability_index(
    baseline: npt.ArrayLike,
    current: npt.ArrayLike,
    n_bins: int = 10,
) -> float:
    """PSI on a numerical column.

    Bin edges are derived from the baseline so the interpretation is "how much
    has the *current* distribution moved relative to the *training-time*
    binning". Empty bins are smoothed with a small epsilon to avoid log(0).
    """
    base = np.asarray(baseline, dtype=float)
    curr = np.asarray(current, dtype=float)
    if base.size == 0 or curr.size == 0:
        raise ValueError("PSI requires non-empty inputs.")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}.")

    # Quantile bins from the baseline so each bucket sees ~equal mass at training time.
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(base, quantiles)
    # Collapse degenerate (constant) features to a trivial single-bin baseline
    # rather than raising — drift on a constant column is a no-op.
    edges = np.unique(edges)
    if edges.size < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    base_counts, _ = np.histogram(base, bins=edges)
    curr_counts, _ = np.histogram(curr, bins=edges)
    base_prop = np.where(base_counts == 0, _EPSILON, base_counts / base.size)
    curr_prop = np.where(curr_counts == 0, _EPSILON, curr_counts / curr.size)

    return float(np.sum((curr_prop - base_prop) * np.log(curr_prop / base_prop)))


def population_stability_index_categorical(
    baseline: npt.ArrayLike,
    current: npt.ArrayLike,
) -> float:
    """PSI on a categorical column. Categories from both sides are unioned."""
    base_series = pd.Series(np.asarray(baseline))
    curr_series = pd.Series(np.asarray(current))
    if base_series.empty or curr_series.empty:
        raise ValueError("PSI requires non-empty inputs.")

    categories = sorted(set(base_series.unique()) | set(curr_series.unique()))
    base_props = (
        base_series.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    )
    curr_props = (
        curr_series.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    )
    base_props = np.where(base_props == 0, _EPSILON, base_props)
    curr_props = np.where(curr_props == 0, _EPSILON, curr_props)

    return float(np.sum((curr_props - base_props) * np.log(curr_props / base_props)))


def ks_test(
    baseline: npt.ArrayLike,
    current: npt.ArrayLike,
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns ``(statistic, p_value)``."""
    base = np.asarray(baseline)
    curr = np.asarray(current)
    if base.size == 0 or curr.size == 0:
        raise ValueError("KS test requires non-empty inputs.")
    result = ks_2samp(base, curr)
    return float(result.statistic), float(result.pvalue)


def detect_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ks_alpha: float = DEFAULT_KS_ALPHA,
) -> pd.DataFrame:
    """Compute PSI + KS per feature; return one row per feature.

    Columns:
        * ``feature``: name
        * ``type``: ``"numerical"`` or ``"categorical"``
        * ``psi``: PSI between baseline and current
        * ``ks_statistic``: KS statistic (numerical only; NaN for categorical)
        * ``ks_pvalue``: KS p-value (numerical only; NaN for categorical)
        * ``psi_alert``: True if ``psi >= psi_threshold``
        * ``ks_alert``: True if numerical and ``ks_pvalue < ks_alpha``

    Sorted by descending PSI so the worst offenders are at the top.
    """
    rows: list[dict[str, Any]] = []
    for col in numerical_columns:
        psi = population_stability_index(baseline[col], current[col])
        stat, pval = ks_test(baseline[col], current[col])
        rows.append(
            {
                "feature": col,
                "type": "numerical",
                "psi": psi,
                "ks_statistic": stat,
                "ks_pvalue": pval,
                "psi_alert": bool(psi >= psi_threshold),
                "ks_alert": bool(pval < ks_alpha),
            }
        )
    for col in categorical_columns:
        psi = population_stability_index_categorical(baseline[col], current[col])
        rows.append(
            {
                "feature": col,
                "type": "categorical",
                "psi": psi,
                "ks_statistic": float("nan"),
                "ks_pvalue": float("nan"),
                "psi_alert": bool(psi >= psi_threshold),
                "ks_alert": False,
            }
        )

    return (
        pd.DataFrame(rows).sort_values("psi", ascending=False, kind="stable").reset_index(drop=True)
    )


def prediction_drift(
    baseline_scores: npt.NDArray[np.floating[Any]],
    current_scores: npt.NDArray[np.floating[Any]],
    n_bins: int = 10,
) -> dict[str, float]:
    """Drift on the model's predicted positive-class probabilities.

    Even when input features look stable, the predicted-score distribution
    shifting is a clean signal that something downstream of the features
    changed (label rebalancing, model retraining, etc.).

    Returns:
        Dict with ``psi``, ``baseline_mean``, ``current_mean``, ``baseline_std``,
        ``current_std``.
    """
    base = np.asarray(baseline_scores, dtype=float)
    curr = np.asarray(current_scores, dtype=float)
    return {
        "psi": population_stability_index(base, curr, n_bins=n_bins),
        "baseline_mean": float(base.mean()),
        "current_mean": float(curr.mean()),
        "baseline_std": float(base.std()),
        "current_std": float(curr.std()),
    }
