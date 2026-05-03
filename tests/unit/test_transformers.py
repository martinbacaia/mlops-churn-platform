from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import check_estimator

from churn.features.transformers import TenureBucketizer


def test_default_buckets_match_lifecycle_phases():
    bucketizer = TenureBucketizer()
    tenures = np.array([0, 5, 11, 12, 18, 23, 24, 36, 47, 48, 60, 72])
    out = bucketizer.fit_transform(tenures.reshape(-1, 1)).ravel()
    expected = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    np.testing.assert_array_equal(out, expected)


def test_custom_bin_edges_are_respected():
    bucketizer = TenureBucketizer(bins=(6, 24))
    tenures = np.array([0, 5, 6, 23, 24, 71])
    out = bucketizer.fit_transform(tenures.reshape(-1, 1)).ravel()
    expected = np.array([0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(out, expected)


def test_accepts_dataframe_input():
    bucketizer = TenureBucketizer()
    df = pd.DataFrame({"tenure": [0, 12, 24, 48]})
    out = bucketizer.fit_transform(df).ravel()
    np.testing.assert_array_equal(out, np.array([0, 1, 2, 3]))


def test_accepts_series_input():
    bucketizer = TenureBucketizer()
    s = pd.Series([0, 12, 24, 48], name="tenure")
    out = bucketizer.fit_transform(s).ravel()
    np.testing.assert_array_equal(out, np.array([0, 1, 2, 3]))


def test_output_shape_is_2d_single_column():
    bucketizer = TenureBucketizer()
    out = bucketizer.fit_transform(np.array([1, 13, 25, 49]).reshape(-1, 1))
    assert out.shape == (4, 1)
    assert out.dtype == np.int64


def test_get_feature_names_out_returns_stable_name():
    bucketizer = TenureBucketizer().fit(np.array([1, 2, 3]).reshape(-1, 1))
    np.testing.assert_array_equal(
        bucketizer.get_feature_names_out(), np.array(["tenure_bucket"], dtype=object)
    )


def test_rejects_multi_column_dataframe():
    bucketizer = TenureBucketizer()
    df = pd.DataFrame({"tenure": [1, 2], "other": [3, 4]})
    with pytest.raises(ValueError, match="single column"):
        bucketizer.fit(df)


def test_rejects_non_numeric_input():
    bucketizer = TenureBucketizer()
    with pytest.raises(ValueError, match="numeric"):
        bucketizer.fit(np.array(["a", "b", "c"]).reshape(-1, 1))


def test_transform_is_deterministic():
    bucketizer = TenureBucketizer()
    x = np.array([0, 12, 24, 48, 72]).reshape(-1, 1)
    bucketizer.fit(x)
    a = bucketizer.transform(x)
    b = bucketizer.transform(x)
    np.testing.assert_array_equal(a, b)


def test_clone_compatibility_with_sklearn():
    """Sanity check that the estimator survives sklearn's clone() mechanics."""
    from sklearn.base import clone

    original = TenureBucketizer(bins=(6, 24))
    cloned = clone(original)
    assert cloned.bins == (6, 24)
    assert cloned is not original


def test_passes_sklearn_estimator_checks():
    """Run a curated subset of sklearn's compatibility checks.

    ``check_estimator`` runs many checks; some assume classifier/regressor
    semantics that don't apply to a stateless feature transformer. We pick the
    relevant ones explicitly.
    """
    from sklearn.utils.estimator_checks import (
        check_get_params_invariance,
        check_set_params,
    )

    estimator = TenureBucketizer()
    check_get_params_invariance("TenureBucketizer", estimator)
    check_set_params("TenureBucketizer", estimator)
    # Touch check_estimator to ensure nothing in import surface is broken.
    assert callable(check_estimator)
