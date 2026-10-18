# SPDX-License-Identifier: GPL-3.0-or-later
"""The multimodality screen: which measurements cannot select a larger K, whatever its fit.

Every K-population mixture predicts the measurement's shape convolved with a non-negative
deuterium distribution on ``0 .. ceil(n_max)`` (the sum of its components' binomials, each
supported on ``0 .. ceil(n)``, ``n < n_max``). The best non-negative distribution of all
(non-negative least squares) therefore fits at least as well as any mixture, of any K: its loss
is a floor under the loss of every K. With the Gaussian loss, K cannot be selected where even
that floor plus K's penalty does not beat the best criterion of the smaller K already fitted.
The screen is exact: it skips only fits that cannot be selected, so the selected K is that of
fitting every K.

The floor needs whole-number bins and the Gaussian loss (the Poisson loss of
:mod:`hxex.model` drops a constant and can be negative); other measurements are never skipped.
"""
# evidence: docs/HISTORY.md item 85

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from scipy.optimize import nnls

from ..model import n_params, normalise_shape

log = logging.getLogger(__name__)

# a fit is skipped only if its best possible criterion is worse than the best one by more than
# this (absolute plus relative to the best): float32 losses are not exact
_TOLERANCE_ABS = 1e-3
_TOLERANCE_REL = 1e-4


def loss_floor(
    bins: Sequence[np.ndarray],
    scaled: Sequence[np.ndarray],
    shapes: Sequence[np.ndarray],
    n_max: np.ndarray,
    noise_kind: Sequence[str],
) -> np.ndarray:
    """The Gaussian loss of the best non-negative deuterium distribution on ``0 ..
    ceil(n_max)`` convolved with each measurement's shape, against its noise-scaled spectrum
    ``scaled``: a lower bound on the loss of a mixture of any number of components. ``-inf``
    where there is no such bound (Poisson loss, or bins that are not whole numbers)."""
    out = np.full(len(bins), -np.inf)
    not_converged = 0
    for i, (x, y, shape) in enumerate(zip(bins, scaled, shapes, strict=True)):
        x = np.asarray(x, dtype=np.float64)
        if noise_kind[i] != "gaussian" or not np.array_equal(x, np.round(x)) or np.any(x < 0):
            continue
        y = np.nan_to_num(np.asarray(y, dtype=np.float64))
        d = normalise_shape(shape)
        diff = x.astype(np.int64)[:, None] - np.arange(int(np.ceil(n_max[i])) + 1)[None, :]
        inside = (diff >= 0) & (diff < len(d))
        design = np.where(inside, d[np.clip(diff, 0, len(d) - 1)], 0.0)
        try:
            _, residual = nnls(design, y, maxiter=50 * design.shape[1])
        except RuntimeError:  # did not converge: the loss is still at least 0
            out[i] = 0.0
            not_converged += 1
            continue
        out[i] = 0.5 * residual * residual
    if not_converged:
        log.warning("the loss floor of %d measurement(s) is 0 (the non-negative least squares "
                    "did not converge): the screen cannot skip any of their fits", not_converged)
    return out


def penalty(criterion: str, k: int, sample_size: np.ndarray) -> np.ndarray:
    """The penalty term of ``criterion`` (BIC or AIC) for K components (the criterion is the
    penalty plus twice the loss)."""
    if criterion == "bic":
        return n_params(k) * np.log(np.maximum(sample_size, 2.0))
    return np.full(len(sample_size), 2.0 * n_params(k))


def can_win(
    criterion: str, k: int, sample_size: np.ndarray, floor: np.ndarray, best: np.ndarray
) -> np.ndarray:
    """Whether K could still be selected: the smallest criterion it can reach (its penalty
    plus twice the loss floor) is not worse than ``best``, the best criterion of the Ks
    fitted so far, by more than the tolerance."""
    reach = penalty(criterion, k, sample_size) + 2.0 * floor
    margin = _TOLERANCE_ABS + _TOLERANCE_REL * np.abs(best)
    return ~(reach > best + margin)
