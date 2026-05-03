"""Deterministic seed control across stdlib random, NumPy, and PyTorch.

Reproducibility guarantee: a fresh process that calls ``set_global_seed(N)``
(or ``set_torch_deterministic(N)`` for any code path that touches PyTorch)
produces identical outputs run-to-run on the same Python / library versions.
The training pipeline propagates ``settings.random_state`` to every stochastic
component: splits, cross-validation folds, Optuna samplers, model RNGs.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python's stdlib random, NumPy, and PYTHONHASHSEED."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_torch_deterministic(seed: int) -> None:
    """Seed PyTorch and force deterministic kernels in addition to global seeds.

    Notes:
        * ``CUBLAS_WORKSPACE_CONFIG`` is required by ``torch.use_deterministic_algorithms``
          on some CUDA versions; harmless on CPU. Set unconditionally so the same code
          path reproduces across machines.
        * ``use_deterministic_algorithms(True)`` *raises* if a non-deterministic op is
          called — that is intentional. We want training to fail loudly rather than
          silently produce different outputs run-to-run.
        * Imported lazily so non-DL code paths (LogReg, XGBoost) don't pay torch's
          import cost.
    """
    set_global_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
