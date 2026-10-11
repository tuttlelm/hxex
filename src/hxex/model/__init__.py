# SPDX-License-Identifier: GPL-3.0-or-later
"""The binomial mixture model: parameterisation, shapes and loss functions.

Gaussian (constant baseline noise) and Poisson losses; see :mod:`hxex.model.mixture`.
Device and precision are an explicit :class:`Backend`, never global.
"""

from .backend import Backend
from .mixture import (
    NOISE_KINDS,
    MixtureBatch,
    PaddedBatch,
    aic,
    bic,
    convolve,
    log_binom_pmf,
    logit,
    n_params,
    noise_loss,
    normalise_shape,
    shape_weights,
)

__all__ = [
    "NOISE_KINDS",
    "Backend",
    "MixtureBatch",
    "PaddedBatch",
    "aic",
    "bic",
    "convolve",
    "log_binom_pmf",
    "logit",
    "noise_loss",
    "n_params",
    "normalise_shape",
    "shape_weights",
]
