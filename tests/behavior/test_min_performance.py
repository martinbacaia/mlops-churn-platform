"""Minimum-performance gate: the Production model must clear a quality bar.

A regression test for the deployed model. If a future training run produces
a champion that drops ROC-AUC below the floor on the held-out split, this
test fails and blocks promotion (when wired into CI).

The threshold is conservative — set high enough that obviously-broken models
fail, low enough that normal training-noise variation passes.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from churn.serving.loader import load_production_artifacts

MIN_HOLDOUT_ROC_AUC = 0.80

pytestmark = pytest.mark.behavior


@pytest.fixture
def artifacts(production_env):
    return load_production_artifacts()


def test_production_model_meets_minimum_roc_auc_on_holdout(artifacts, production_env):
    splits = production_env.splits
    X_test_t = artifacts.feature_pipeline.transform(splits.X_test)
    proba = artifacts.model.predict_proba(X_test_t)[:, 1]
    roc_auc = float(roc_auc_score(splits.y_test.to_numpy(), proba))

    assert roc_auc >= MIN_HOLDOUT_ROC_AUC, (
        f"Production model ROC-AUC on holdout = {roc_auc:.4f}, "
        f"below the minimum {MIN_HOLDOUT_ROC_AUC}. Refusing to certify."
    )


def test_production_model_beats_random_baseline(artifacts, production_env):
    """A defensive sanity: the model must beat 50/50 by at least 10 points."""
    splits = production_env.splits
    X_test_t = artifacts.feature_pipeline.transform(splits.X_test)
    proba = artifacts.model.predict_proba(X_test_t)[:, 1]
    roc_auc = float(roc_auc_score(splits.y_test.to_numpy(), proba))
    assert roc_auc > 0.60, (
        f"ROC-AUC {roc_auc:.4f} is barely above chance. The model is broken "
        f"or the holdout is the wrong distribution."
    )


def test_production_model_predicts_both_classes(artifacts, production_env):
    """A model that always predicts one class is technically valid but useless."""
    splits = production_env.splits
    X_test_t = artifacts.feature_pipeline.transform(splits.X_test)
    proba = artifacts.model.predict_proba(X_test_t)[:, 1]
    # The model should give *some* customers a high probability and *some* a low one.
    assert float(np.std(proba)) > 0.05, (
        f"Predicted-probability std = {np.std(proba):.4f}. "
        f"The model is collapsing toward a single value."
    )
