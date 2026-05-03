from __future__ import annotations

import numpy as np
import pytest

from churn.training.metrics import compute_classification_metrics


def _proba_from_pos(y_pos):
    return np.column_stack([1.0 - y_pos, y_pos])


def test_returns_canonical_metric_keys():
    y_true = np.array([0, 1, 0, 1])
    y_proba = _proba_from_pos(np.array([0.1, 0.9, 0.2, 0.8]))
    out = compute_classification_metrics(y_true, y_proba)
    assert set(out.keys()) == {"roc_auc", "pr_auc", "f1", "log_loss", "brier"}


def test_perfect_predictions_score_perfectly():
    y_true = np.array([0, 0, 1, 1])
    y_proba = _proba_from_pos(np.array([0.01, 0.02, 0.98, 0.99]))
    out = compute_classification_metrics(y_true, y_proba)
    assert out["roc_auc"] == pytest.approx(1.0)
    assert out["pr_auc"] == pytest.approx(1.0)
    assert out["f1"] == pytest.approx(1.0)
    assert out["log_loss"] < 0.1
    assert out["brier"] < 0.01


def test_chance_predictions_score_near_baseline():
    rng = np.random.default_rng(0)
    n = 1000
    y_true = rng.integers(0, 2, size=n)
    y_pos = rng.uniform(size=n)
    out = compute_classification_metrics(y_true, _proba_from_pos(y_pos))
    assert 0.4 < out["roc_auc"] < 0.6
    assert 0.4 < out["pr_auc"] < 0.6


def test_all_outputs_are_python_floats():
    """MLflow's ``log_metrics`` is happiest with native floats, not numpy scalars."""
    y_true = np.array([0, 1, 0, 1])
    y_proba = _proba_from_pos(np.array([0.1, 0.9, 0.2, 0.8]))
    out = compute_classification_metrics(y_true, y_proba)
    for k, v in out.items():
        assert isinstance(v, float), f"{k} should be float, got {type(v).__name__}"


def test_threshold_changes_f1_only():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pos = np.array([0.45, 0.55, 0.45, 0.55, 0.45])
    y_proba = _proba_from_pos(y_pos)

    at_half = compute_classification_metrics(y_true, y_proba, threshold=0.5)
    aggressive = compute_classification_metrics(y_true, y_proba, threshold=0.4)

    # Threshold-free metrics are identical regardless of threshold.
    assert at_half["roc_auc"] == aggressive["roc_auc"]
    assert at_half["pr_auc"] == aggressive["pr_auc"]
    assert at_half["log_loss"] == aggressive["log_loss"]
    assert at_half["brier"] == aggressive["brier"]
    # F1 changes (more positives predicted at lower threshold).
    assert at_half["f1"] != aggressive["f1"]


def test_raises_on_wrong_proba_shape():
    y_true = np.array([0, 1, 0, 1])
    bad = np.array([0.1, 0.9, 0.2, 0.8])  # 1-D, not (n, 2)
    with pytest.raises(ValueError, match="shape"):
        compute_classification_metrics(y_true, bad)


def test_raises_on_row_count_mismatch():
    y_true = np.array([0, 1, 0, 1])
    y_proba = _proba_from_pos(np.array([0.1, 0.9, 0.2]))  # only 3 rows
    with pytest.raises(ValueError, match="row counts"):
        compute_classification_metrics(y_true, y_proba)


def test_handles_imbalanced_inputs_without_zero_division():
    """f1 should be 0.0 (not NaN) when no positives are predicted."""
    y_true = np.array([0, 0, 1, 0])
    y_pos = np.array([0.1, 0.1, 0.4, 0.1])  # all below 0.5 threshold
    out = compute_classification_metrics(y_true, _proba_from_pos(y_pos))
    assert out["f1"] == 0.0
