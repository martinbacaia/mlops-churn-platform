from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.evaluation.metrics import (
    calibration_curve_data,
    confusion_matrix_at,
    threshold_sweep,
)


def _proba_from_pos(y_pos):
    return np.column_stack([1.0 - y_pos, y_pos])


# --- threshold_sweep ------------------------------------------------------


def test_threshold_sweep_default_returns_19_rows():
    y_true = np.array([0, 0, 1, 1])
    y_pos = np.array([0.1, 0.4, 0.6, 0.9])
    out = threshold_sweep(y_true, _proba_from_pos(y_pos))
    assert len(out) == 19
    assert list(out.columns) == [
        "threshold",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    ]


def test_threshold_sweep_counts_sum_to_n():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pos = np.array([0.1, 0.4, 0.6, 0.7, 0.9])
    out = threshold_sweep(y_true, _proba_from_pos(y_pos))
    for _, row in out.iterrows():
        assert row["tp"] + row["fp"] + row["fn"] + row["tn"] == len(y_true)


def test_threshold_sweep_precision_recall_in_unit_range():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pos = rng.uniform(size=200)
    out = threshold_sweep(y_true, _proba_from_pos(y_pos))
    assert (out["precision"].between(0.0, 1.0)).all()
    assert (out["recall"].between(0.0, 1.0)).all()
    assert (out["f1"].between(0.0, 1.0)).all()


def test_threshold_sweep_low_threshold_predicts_more_positives():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pos = np.array([0.1, 0.55, 0.45, 0.6, 0.4])
    out = threshold_sweep(y_true, _proba_from_pos(y_pos))
    low = out.iloc[0]
    high = out.iloc[-1]
    assert low["tp"] + low["fp"] >= high["tp"] + high["fp"]


def test_threshold_sweep_custom_thresholds():
    y_true = np.array([0, 1, 0, 1])
    y_pos = np.array([0.2, 0.7, 0.4, 0.6])
    custom = np.array([0.5])
    out = threshold_sweep(y_true, _proba_from_pos(y_pos), thresholds=custom)
    assert len(out) == 1
    assert out.iloc[0]["threshold"] == pytest.approx(0.5)


def test_threshold_sweep_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        threshold_sweep(np.array([0, 1]), np.array([0.5, 0.5]))


# --- calibration_curve_data -----------------------------------------------


def test_calibration_returns_n_bins_rows_with_expected_columns():
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.integers(0, 2, size=n)
    y_pos = rng.uniform(size=n)
    out = calibration_curve_data(y_true, _proba_from_pos(y_pos), n_bins=10)
    assert len(out) == 10
    assert list(out.columns) == [
        "bin_lower",
        "bin_upper",
        "predicted_mean",
        "observed_rate",
        "count",
    ]


def test_calibration_perfect_model_observed_matches_predicted():
    """A well-calibrated model has predicted_mean ≈ observed_rate per bin."""
    rng = np.random.default_rng(0)
    n = 5000
    y_pos = rng.uniform(size=n)
    y_true = (rng.uniform(size=n) < y_pos).astype(np.int64)  # perfectly calibrated
    out = calibration_curve_data(y_true, _proba_from_pos(y_pos), n_bins=10)
    populated = out[out["count"] > 0]
    diffs = np.abs(populated["predicted_mean"] - populated["observed_rate"])
    assert diffs.max() < 0.1


def test_calibration_keeps_empty_bins_with_nan():
    """Empty bins are preserved as NaN rows so plots show coverage gaps honestly."""
    y_true = np.array([0, 1, 0, 1])
    y_pos = np.array([0.05, 0.95, 0.05, 0.95])  # only outermost bins populated
    out = calibration_curve_data(y_true, _proba_from_pos(y_pos), n_bins=10)
    middle_bins = out.iloc[3:7]
    assert (middle_bins["count"] == 0).all()
    assert middle_bins["predicted_mean"].isna().all()


def test_calibration_counts_sum_to_total_samples():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_pos = rng.uniform(size=n)
    out = calibration_curve_data(y_true, _proba_from_pos(y_pos), n_bins=8)
    assert int(out["count"].sum()) == n


def test_calibration_rejects_too_few_bins():
    y_true = np.array([0, 1])
    y_proba = _proba_from_pos(np.array([0.4, 0.6]))
    with pytest.raises(ValueError, match="n_bins"):
        calibration_curve_data(y_true, y_proba, n_bins=1)


# --- confusion_matrix_at --------------------------------------------------


def test_confusion_matrix_at_returns_canonical_keys():
    y_true = np.array([0, 1, 0, 1])
    y_pos = np.array([0.2, 0.8, 0.6, 0.7])
    out = confusion_matrix_at(y_true, _proba_from_pos(y_pos), threshold=0.5)
    assert set(out.keys()) == {"tp", "fp", "fn", "tn"}


def test_confusion_matrix_known_values():
    y_true = np.array([0, 0, 1, 1])
    y_pos = np.array([0.1, 0.6, 0.7, 0.4])  # 1 FP (idx 1), 1 FN (idx 3)
    out = confusion_matrix_at(y_true, _proba_from_pos(y_pos), threshold=0.5)
    assert out == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_confusion_matrix_handles_all_one_class_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_pos = np.array([0.01, 0.02, 0.03, 0.04])  # all predicted negative
    out = confusion_matrix_at(y_true, _proba_from_pos(y_pos), threshold=0.5)
    assert out["tp"] == 0
    assert out["fp"] == 0
    assert out["fn"] == 2
    assert out["tn"] == 2


def test_confusion_matrix_counts_sum_to_n():
    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    y_pos = rng.uniform(size=n)
    out = confusion_matrix_at(y_true, _proba_from_pos(y_pos))
    assert sum(out.values()) == n


def test_returns_pandas_dataframe_types():
    """Threshold-sweep output is a DataFrame, calibration too — easy to log to MLflow."""
    y_true = np.array([0, 1, 0, 1])
    y_proba = _proba_from_pos(np.array([0.2, 0.8, 0.4, 0.6]))
    assert isinstance(threshold_sweep(y_true, y_proba), pd.DataFrame)
    assert isinstance(calibration_curve_data(y_true, y_proba, n_bins=4), pd.DataFrame)
