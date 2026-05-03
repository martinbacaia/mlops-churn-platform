from __future__ import annotations

import os
import random

import numpy as np
import pytest

from churn.seeds import set_global_seed, set_torch_deterministic


def test_set_global_seed_makes_python_random_deterministic():
    set_global_seed(42)
    a = [random.random() for _ in range(5)]
    set_global_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_global_seed_makes_numpy_deterministic():
    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_global_seed_sets_pythonhashseed():
    set_global_seed(123)
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_different_seeds_produce_different_sequences():
    set_global_seed(1)
    a = np.random.rand(5)
    set_global_seed(2)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)


def test_set_torch_deterministic_makes_torch_reproducible():
    torch = pytest.importorskip("torch")
    set_torch_deterministic(42)
    a = torch.rand(5)
    set_torch_deterministic(42)
    b = torch.rand(5)
    assert torch.equal(a, b)
