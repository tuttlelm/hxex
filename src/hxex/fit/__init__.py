# SPDX-License-Identifier: GPL-3.0-or-later
"""Fitting: initialisation, optimiser, K selection and non-finite protection.

Batched across measurements; nested K start; Adam with a cosine schedule, then L-BFGS.
:func:`fit_models` is the entry point; its input is :class:`Measurements` and its output a
:class:`FitResult`.
"""

from .config import CRITERIA, FitConfig
from .init import Start, kmeans_start, split_start
from .measurements import Measurements
from .models import fit_models
from .optimise import FitError, Prepared, cosine_lr, fit_batch, prepare, stable_keys
from .result import FitResult, KFit

__all__ = [
    "CRITERIA",
    "FitConfig",
    "FitError",
    "FitResult",
    "KFit",
    "Measurements",
    "Prepared",
    "Start",
    "cosine_lr",
    "fit_batch",
    "fit_models",
    "kmeans_start",
    "prepare",
    "split_start",
    "stable_keys",
]
