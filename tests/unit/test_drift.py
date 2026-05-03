from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.monitoring.drift import (
    DEFAULT_PSI_THRESHOLD,
    detect_drift,
    ks_test,
    population_stability_index,
    population_stability_index_categorical,
    prediction_drift,
)

# --- PSI numerical ---------------------------------------------------------


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    psi = population_stability_index(x, x)
    assert psi < 0.01


def test_psi_grows_with_distribution_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(loc=0, scale=1, size=2000)
    mild_shift = rng.normal(loc=0.3, scale=1, size=2000)
    severe_shift = rng.normal(loc=2.0, scale=1, size=2000)
    psi_mild = population_stability_index(base, mild_shift)
    psi_severe = population_stability_index(base, severe_shift)
    assert psi_severe > psi_mild
    assert psi_severe > DEFAULT_PSI_THRESHOLD


def test_psi_handles_constant_baseline_without_raising():
    """A constant feature has degenerate quantiles; PSI returns 0 instead of crashing."""
    base = np.ones(100)
    curr = np.ones(100)
    assert population_stability_index(base, curr) == 0.0


def test_psi_rejects_empty_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        population_stability_index(np.array([]), np.array([1.0, 2.0]))


def test_psi_rejects_too_few_bins():
    with pytest.raises(ValueError, match="n_bins"):
        population_stability_index(np.array([1.0, 2.0]), np.array([1.0, 2.0]), n_bins=1)


# --- PSI categorical ------------------------------------------------------


def test_psi_categorical_zero_for_identical():
    base = np.array(["a", "b", "a", "c", "b"] * 100)
    psi = population_stability_index_categorical(base, base)
    assert psi < 0.01


def test_psi_categorical_grows_when_proportions_shift():
    rng = np.random.default_rng(0)
    base = rng.choice(["a", "b"], size=2000, p=[0.5, 0.5])
    shifted = rng.choice(["a", "b"], size=2000, p=[0.9, 0.1])
    psi = population_stability_index_categorical(base, shifted)
    assert psi > DEFAULT_PSI_THRESHOLD


def test_psi_categorical_handles_new_unseen_category():
    """A novel category in current must not crash and should register some drift."""
    base = np.array(["a"] * 100 + ["b"] * 100)
    curr = np.array(["a"] * 100 + ["b"] * 80 + ["c"] * 20)
    psi = population_stability_index_categorical(base, curr)
    assert psi > 0.0


# --- KS test --------------------------------------------------------------


def test_ks_test_high_pvalue_for_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    _, pval = ks_test(x, y)
    assert pval > 0.05  # likely fails to reject


def test_ks_test_low_pvalue_for_shifted_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0, size=500)
    y = rng.normal(loc=2, size=500)
    stat, pval = ks_test(x, y)
    assert pval < 0.001
    assert stat > 0.4


def test_ks_test_rejects_empty_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        ks_test(np.array([]), np.array([1.0]))


# --- detect_drift --------------------------------------------------------


def _toy_baseline_current(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = 500
    base = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, size=n),
            "MonthlyCharges": rng.uniform(20, 120, size=n),
            "gender": rng.choice(["Female", "Male"], size=n, p=[0.5, 0.5]),
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"], size=n, p=[0.5, 0.3, 0.2]
            ),
        }
    )
    curr = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, size=n),
            "MonthlyCharges": rng.uniform(50, 150, size=n),  # shifted up
            "gender": rng.choice(["Female", "Male"], size=n, p=[0.7, 0.3]),  # shifted
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"], size=n, p=[0.5, 0.3, 0.2]
            ),
        }
    )
    return base, curr


def test_detect_drift_returns_one_row_per_feature():
    base, curr = _toy_baseline_current()
    out = detect_drift(
        baseline=base,
        current=curr,
        numerical_columns=["tenure", "MonthlyCharges"],
        categorical_columns=["gender", "Contract"],
    )
    assert set(out["feature"]) == {"tenure", "MonthlyCharges", "gender", "Contract"}
    assert list(out.columns) == [
        "feature",
        "type",
        "psi",
        "ks_statistic",
        "ks_pvalue",
        "psi_alert",
        "ks_alert",
    ]


def test_detect_drift_sorts_by_descending_psi():
    base, curr = _toy_baseline_current()
    out = detect_drift(base, curr, ["tenure", "MonthlyCharges"], ["gender", "Contract"])
    psis = out["psi"].to_numpy()
    assert (psis[:-1] >= psis[1:]).all()


def test_detect_drift_categorical_rows_have_nan_ks():
    base, curr = _toy_baseline_current()
    out = detect_drift(base, curr, ["tenure"], ["gender"])
    cat = out[out["type"] == "categorical"].iloc[0]
    assert pd.isna(cat["ks_statistic"])
    assert pd.isna(cat["ks_pvalue"])
    assert cat["ks_alert"] is False or cat["ks_alert"] == False  # noqa: E712


def test_detect_drift_flags_shifted_features():
    base, curr = _toy_baseline_current()
    out = detect_drift(base, curr, ["tenure", "MonthlyCharges"], ["gender", "Contract"])
    monthly = out[out["feature"] == "MonthlyCharges"].iloc[0]
    assert monthly["psi_alert"] or monthly["ks_alert"]


def test_detect_drift_ignores_columns_not_listed():
    base, curr = _toy_baseline_current()
    base["unused"] = 1
    curr["unused"] = 2
    out = detect_drift(base, curr, ["tenure"], ["gender"])
    assert "unused" not in out["feature"].tolist()


# --- prediction_drift -----------------------------------------------------


def test_prediction_drift_zero_psi_for_same_scores():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, size=500)
    out = prediction_drift(s, s)
    assert out["psi"] < 0.01
    assert out["baseline_mean"] == pytest.approx(out["current_mean"])


def test_prediction_drift_detects_shift_in_scores():
    rng = np.random.default_rng(0)
    base = rng.beta(2, 8, size=2000)  # mean ~0.2
    curr = rng.beta(8, 2, size=2000)  # mean ~0.8
    out = prediction_drift(base, curr)
    assert out["psi"] > DEFAULT_PSI_THRESHOLD
    assert out["current_mean"] > out["baseline_mean"] + 0.4


def test_prediction_drift_returns_canonical_keys():
    rng = np.random.default_rng(0)
    s = rng.uniform(size=200)
    out = prediction_drift(s, s)
    assert set(out.keys()) == {
        "psi",
        "baseline_mean",
        "current_mean",
        "baseline_std",
        "current_std",
    }
