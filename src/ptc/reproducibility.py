"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_algorithms: bool = False) -> None:
    """Set seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Global seed.
    deterministic_algorithms:
        If true, requests deterministic PyTorch algorithms where available.
        This can slow down training and may warn on unsupported CUDA kernels.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
