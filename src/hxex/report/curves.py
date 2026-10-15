# SPDX-License-Identifier: GPL-3.0-or-later
"""The fitted curves in plain NumPy, so that plotting does not need the optimiser (PyTorch).

Same model as :mod:`hxex.model.mixture`: each component is a real-valued binomial (the gamma
function continuation, nonzero for ``0 <= b < n + 1``) convolved with the natural-abundance
shape, scaled by its amplitude. ``tests/test_report.py`` checks it against the PyTorch model.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp


def log_binom(b: np.ndarray, n: float, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Log of the real-valued binomial pmf at the bins ``b``, normalised to sum to 1 over its
    valid bins ``0 .. ceil(n)``, and the mask of bins that have probability (the same rule as
    ``hxex.model.mixture``, including its normalisation)."""
    b = np.asarray(b, dtype=float)
    log_pmf, valid = _log_binom_raw(b, n, p)
    grid, grid_valid = _log_binom_raw(np.arange(int(np.ceil(max(n, 0.0))) + 2, dtype=float), n, p)
    log_norm = logsumexp(grid[grid_valid]) if grid_valid.any() else 0.0
    return log_pmf - log_norm, valid


def _log_binom_raw(b: np.ndarray, n: float, p: float) -> tuple[np.ndarray, np.ndarray]:
    b_safe = np.clip(b, 0.0, None)
    rest = np.clip(n - b + 1.0, 1e-12, None)
    valid = (b >= 0) & (n - b + 1.0 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pmf = (
            gammaln(n + 1) - gammaln(b_safe + 1) - gammaln(rest)
            + b_safe * np.log(max(p, 1e-10)) + (n - b_safe) * np.log(max(1.0 - p, 1e-10))
        )
    return log_pmf, valid


def component_curves(
    bins: np.ndarray, amplitude: np.ndarray, n: np.ndarray, p: np.ndarray, shape: np.ndarray
) -> np.ndarray:
    """The intensity of each component at ``bins``, ``(K, len(bins))``, in the units of
    ``amplitude``. ``shape`` is the natural-abundance envelope (normalised here)."""
    bins = np.asarray(bins, dtype=float)
    shape = np.asarray(shape, dtype=float)
    total = shape.sum()
    shape = shape / total if total > 0 else shape
    out = np.zeros((len(amplitude), len(bins)))
    for k, (a, n_k, p_k) in enumerate(zip(amplitude, n, p, strict=True)):
        for j, w in enumerate(shape):
            log_pmf, valid = log_binom(bins - j, float(n_k), float(p_k))
            out[k] += w * np.where(valid, np.exp(np.where(valid, log_pmf, 0.0)), 0.0)
        out[k] *= a
    return out
