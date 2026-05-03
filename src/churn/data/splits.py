"""Deterministic stratified train / validation / test splits.

The same ``random_state`` produces the same partition byte-for-byte across
machines — this guarantee underpins the reproducibility claim of the platform.

Returns a typed :class:`Splits` dataclass rather than a tuple of six dataframes
so callers can address ``.X_train`` / ``.y_test`` by name and add new fields
later (e.g. group keys for time-aware splits) without reshuffling positional
arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.data.schema import TARGET_COLUMN


@dataclass(frozen=True)
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series[int]
    X_val: pd.DataFrame
    y_val: pd.Series[int]
    X_test: pd.DataFrame
    y_test: pd.Series[int]

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.X_train),
            "val": len(self.X_val),
            "test": len(self.X_test),
        }

    @property
    def positive_rates(self) -> dict[str, float]:
        return {
            "train": float(self.y_train.mean()),
            "val": float(self.y_val.mean()),
            "test": float(self.y_test.mean()),
        }


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
    target: str = TARGET_COLUMN,
) -> Splits:
    """Split ``df`` into train / val / test, stratified on ``target``.

    The ratios apply hierarchically: first ``test_size`` is held out, then
    ``val_size`` is taken from what remains. Defaults of (0.20, 0.20) yield
    roughly 64 / 16 / 20 — train large enough to learn, val sized to tune,
    test held back as the comparison surface for all three models.
    """
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not found in dataframe.")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size}.")
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size must be in (0, 1); got {val_size}.")

    X = df.drop(columns=[target])
    y = df[target]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        stratify=y_trainval,
        random_state=random_state,
    )

    return Splits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
