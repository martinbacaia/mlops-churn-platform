"""Invariance tests: features that should not move the prediction much.

A textbook fairness sanity check: flipping ``gender`` on the same customer
record should not significantly change the churn probability. If it does,
either the data leaks gender into the target or the model picked up a
spurious correlation — both surfaces worth knowing about before promoting.

These tests load the **Production-stage** model (whatever is current),
precisely so a future model regression on this property breaks CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from churn.serving.loader import load_production_artifacts

INVARIANCE_TOLERANCE = 0.05  # 5 percentage points

pytestmark = pytest.mark.behavior


@pytest.fixture
def artifacts(production_env):
    return load_production_artifacts()


def test_flipping_gender_does_not_move_predictions_significantly(artifacts, production_env):
    splits = production_env.splits
    sample = splits.X_test.copy()
    if "gender" not in sample.columns or len(sample) == 0:
        pytest.skip("Holdout has no gender column or is empty.")

    flipped = sample.copy()
    flipped["gender"] = flipped["gender"].map({"Female": "Male", "Male": "Female"})

    base_proba = artifacts.model.predict_proba(artifacts.feature_pipeline.transform(sample))[:, 1]
    flipped_proba = artifacts.model.predict_proba(artifacts.feature_pipeline.transform(flipped))[
        :, 1
    ]

    # Mean absolute change across the holdout.
    mean_delta = float(np.abs(base_proba - flipped_proba).mean())
    max_delta = float(np.abs(base_proba - flipped_proba).max())

    assert mean_delta <= INVARIANCE_TOLERANCE, (
        f"Mean churn-prob change after flipping gender = {mean_delta:.4f}, "
        f"exceeds invariance tolerance {INVARIANCE_TOLERANCE}."
    )
    # Per-record changes can be slightly larger due to one-hot interactions
    # with other features. Cap at 2x the mean tolerance to flag outliers.
    assert max_delta <= INVARIANCE_TOLERANCE * 2, (
        f"Max per-record gender-flip change = {max_delta:.4f}, "
        f"exceeds {INVARIANCE_TOLERANCE * 2}."
    )
