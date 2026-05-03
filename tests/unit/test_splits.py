from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.data.schema import TARGET_COLUMN
from churn.data.splits import Splits, make_splits


def _toy_df(n: int = 1000, positive_rate: float = 0.27, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(size=n),
            "feature_b": rng.choice(["x", "y", "z"], size=n),
            TARGET_COLUMN: rng.binomial(1, positive_rate, size=n),
        }
    )


def test_splits_partition_covers_all_rows():
    df = _toy_df()
    s = make_splits(df, test_size=0.2, val_size=0.2, random_state=42)
    total = s.sizes["train"] + s.sizes["val"] + s.sizes["test"]
    assert total == len(df)


def test_splits_have_no_overlap():
    df = _toy_df()
    s = make_splits(df, random_state=42)
    train_idx = set(s.X_train.index)
    val_idx = set(s.X_val.index)
    test_idx = set(s.X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_splits_are_deterministic_for_same_seed():
    df = _toy_df()
    a = make_splits(df, random_state=42)
    b = make_splits(df, random_state=42)
    pd.testing.assert_frame_equal(a.X_train, b.X_train)
    pd.testing.assert_frame_equal(a.X_test, b.X_test)
    pd.testing.assert_series_equal(a.y_train, b.y_train)
    pd.testing.assert_series_equal(a.y_test, b.y_test)


def test_splits_differ_for_different_seeds():
    df = _toy_df()
    a = make_splits(df, random_state=1)
    b = make_splits(df, random_state=2)
    assert not a.X_train.index.equals(b.X_train.index)


def test_splits_preserve_class_balance_within_tolerance():
    df = _toy_df(n=4000, positive_rate=0.27)
    s = make_splits(df, random_state=42)
    overall = df[TARGET_COLUMN].mean()
    for split_name, rate in s.positive_rates.items():
        assert abs(rate - overall) < 0.02, f"{split_name} drifted from base rate"


def test_splits_default_ratios_match_documented():
    df = _toy_df(n=10_000)
    s = make_splits(df, test_size=0.2, val_size=0.2, random_state=42)
    # 0.2 test, then 0.2 of the remaining 0.8 goes to val => ~0.16 val, ~0.64 train
    sizes = s.sizes
    assert abs(sizes["test"] / 10_000 - 0.20) < 0.005
    assert abs(sizes["val"] / 10_000 - 0.16) < 0.005
    assert abs(sizes["train"] / 10_000 - 0.64) < 0.005


def test_make_splits_raises_when_target_missing():
    df = _toy_df().drop(columns=[TARGET_COLUMN])
    with pytest.raises(KeyError):
        make_splits(df)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_make_splits_rejects_invalid_test_size(bad: float):
    with pytest.raises(ValueError):
        make_splits(_toy_df(), test_size=bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_make_splits_rejects_invalid_val_size(bad: float):
    with pytest.raises(ValueError):
        make_splits(_toy_df(), val_size=bad)


def test_splits_returns_typed_dataclass():
    s = make_splits(_toy_df(), random_state=42)
    assert isinstance(s, Splits)
    assert TARGET_COLUMN not in s.X_train.columns
    assert TARGET_COLUMN not in s.X_test.columns
