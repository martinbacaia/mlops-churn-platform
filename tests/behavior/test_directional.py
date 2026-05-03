"""Directional tests: features whose effect on the prediction has a known sign.

For Telco churn, a long-tenure customer who has spent more in total is, by
domain logic, *less* likely to churn — high lifetime value correlates with
loyalty. We assert this on **average** across a cohort, not per-record:
single-record monotonicity is a stronger claim than the data supports
(individual customers can have idiosyncratic patterns), but the population
mean should respect the direction.

Failure here is a model-quality alert worth blocking promotion on: it means
the model has learned a perverse correlation that will surprise stakeholders.
"""

from __future__ import annotations

import pytest

from churn.serving.loader import load_production_artifacts

HIGH_TENURE_THRESHOLD = 48  # months — "long-loyal" customers
TOTAL_CHARGES_BUMP = 1.5  # multiply existing TotalCharges by this factor

pytestmark = pytest.mark.behavior


@pytest.fixture
def artifacts(production_env):
    return load_production_artifacts()


def test_increasing_total_charges_in_long_tenure_customers_reduces_churn_prob(
    artifacts, production_env
):
    splits = production_env.splits
    long_tenure = splits.X_test[splits.X_test["tenure"] >= HIGH_TENURE_THRESHOLD].copy()
    if len(long_tenure) == 0:
        pytest.skip("No long-tenure customers in the holdout split.")

    boosted = long_tenure.copy()
    boosted["TotalCharges"] = boosted["TotalCharges"] * TOTAL_CHARGES_BUMP

    base_proba = artifacts.model.predict_proba(artifacts.feature_pipeline.transform(long_tenure))[
        :, 1
    ]
    boosted_proba = artifacts.model.predict_proba(artifacts.feature_pipeline.transform(boosted))[
        :, 1
    ]

    # Directional claim: the cohort mean should decrease (or at least not
    # increase) when total spend goes up among long-loyal customers.
    base_mean = float(base_proba.mean())
    boosted_mean = float(boosted_proba.mean())

    assert boosted_mean <= base_mean + 1e-3, (
        f"Cohort mean churn prob went UP from {base_mean:.4f} to "
        f"{boosted_mean:.4f} after boosting TotalCharges in long-tenure "
        f"customers. The model has the direction wrong on a domain sanity check."
    )


def test_directional_check_finds_meaningful_cohort_size():
    """Sanity: if the cohort is empty, the test above is vacuous. Fail loud here."""
    # Re-resolve the cohort from the same fixture state.
    from churn.data.ingest import load_raw, preprocess
    from churn.data.splits import make_splits
    from tests.conftest import TELCO_SAMPLE_CSV

    df = preprocess(load_raw(TELCO_SAMPLE_CSV))
    splits = make_splits(df, random_state=42)
    n_long = int((splits.X_test["tenure"] >= HIGH_TENURE_THRESHOLD).sum())
    assert n_long >= 5, (
        f"Holdout has only {n_long} long-tenure customers; the directional "
        f"test needs more rows to be meaningful."
    )
