# SPDX-License-Identifier: GPL-3.0-or-later
"""Starting points for the optimiser: shape-corrected k-means and warm starts by splitting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..model import logit


@dataclass(frozen=True)
class Start:
    """Starting parameters for M measurements and K components, each of shape ``(M, K)`` in the
    scaled units the fit works in (``log_A``, ``n_raw``, ``logit_p``)."""

    log_A: np.ndarray
    n_raw: np.ndarray
    logit_p: np.ndarray


def kmeans_start(
    bins: np.ndarray, observed: np.ndarray, k: int, shape: np.ndarray, n_max: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Starting ``(log_A, n_raw, logit_p)`` (each of length K) for one measurement; see
    :func:`kmeans_starts`."""
    la, nr, lp = kmeans_starts([bins], [observed], k, [shape], np.array([n_max]), [seed])
    return la[0], nr[0], lp[0]


def kmeans_starts(
    bins: Sequence[np.ndarray],
    observed: Sequence[np.ndarray],
    k: int,
    shapes: Sequence[np.ndarray],
    n_max: np.ndarray,
    seeds: Sequence[int],
    iters: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Starting ``(log_A, n_raw, logit_p)``, each ``(M, K)``, for M measurements at once.

    The bins are clustered with a one-dimensional k-means weighted by ``round(intensity)`` (at
    most 2000 per bin; the same clusters as k-means on that many copies of each bin, without
    making the copies). The centroids start at the weighted quantiles
    (deterministic); a cluster left empty is moved to the worst-explained bin. Each cluster
    gives a method-of-moments estimate corrected for the shape's own mean and variance::

        n*p        ~= mean(cluster) - mean(shape)
        n*p*(1-p)  ~= var(cluster)  - var(shape)

    and where ``n`` is clipped to its range (``0.1`` to ``0.98 n_max``), ``p`` is set to keep
    the mean ``n * p``.

    ``n_raw`` is the inverse of the sigmoid parameterisation of ``n``. A measurement with fewer
    distinct bins than components is padded with jittered copies (from ``seeds``).
    """
    m = len(bins)
    width = max((len(b) for b in bins), default=1)
    x = np.zeros((m, width))
    w = np.zeros((m, width))
    total = np.zeros(m)
    mean_d = np.zeros(m)
    var_d = np.zeros(m)
    for i, (bc, ob, shape) in enumerate(zip(bins, observed, shapes, strict=True)):
        bc = np.asarray(bc, dtype=np.float64)
        ob = np.nan_to_num(np.asarray(ob, dtype=np.float64), nan=0.0).clip(min=0)
        counts = np.round(ob).clip(min=0, max=2000)
        if counts.sum() < k:  # too few pseudo-samples: every bin once
            counts = np.ones(len(bc))
        x[i, : len(bc)] = bc
        w[i, : len(bc)] = counts
        total[i] = ob.sum()
        j = np.arange(len(shape), dtype=np.float64)
        mean_d[i] = np.sum(j * shape)
        var_d[i] = np.sum((j - mean_d[i]) ** 2 * shape)
    k_eff = np.minimum(k, np.array([len(np.unique(x[i][w[i] > 0])) for i in range(m)]))
    active = np.arange(k)[None, :] < k_eff[:, None]  # (M, K)
    rows = np.arange(m)

    # initial centroids at the weighted quantiles (i + 0.5) / k_eff
    cum = np.cumsum(w, axis=1)
    cum = cum / np.maximum(cum[:, -1:], 1e-300)
    q = (np.arange(k)[None, :] + 0.5) / np.maximum(k_eff, 1)[:, None]
    pos = np.array([np.searchsorted(cum[i], q[i]) for i in range(m)]).clip(max=width - 1)
    cents = np.where(active, x[rows[:, None], pos], np.inf)

    def assign(c):
        return np.argmin(np.abs(x[:, :, None] - c[:, None, :]), axis=2)  # (M, B); inf never wins

    labels = assign(cents)
    for _ in range(iters):
        onehot = (labels[:, :, None] == np.arange(k)) * w[:, :, None]  # (M, B, K)
        wsum = onehot.sum(axis=1)
        new = np.where(wsum > 0, (onehot * x[:, :, None]).sum(axis=1) / np.maximum(wsum, 1e-300),
                       cents)
        empty = active & (wsum <= 0)
        if empty.any():  # move each empty cluster to the bin its own centroid explains worst
            err = w * (x - np.take_along_axis(new, labels, axis=1)) ** 2
            for i, c in zip(*np.nonzero(empty), strict=True):
                b = int(np.argmax(err[i]))
                new[i, c] = x[i, b]
                err[i, b] = 0.0
        new = np.where(active, new, np.inf)
        new_labels = assign(new)
        if np.array_equal(new_labels, labels) and np.allclose(new[active], cents[active]):
            cents = new
            break
        cents, labels = new, new_labels

    onehot = (labels[:, :, None] == np.arange(k)) * w[:, :, None]
    wsum = onehot.sum(axis=1)
    mu = np.where(wsum > 0, (onehot * x[:, :, None]).sum(axis=1) / np.maximum(wsum, 1e-300),
                  np.where(np.isfinite(cents), cents, 0.0))
    var = (onehot * (x[:, :, None] - mu[:, None, :]) ** 2).sum(axis=1) / np.maximum(wsum, 1e-300)
    var = np.where(wsum > 1, var, np.maximum(var_d + 0.5, 0.5)[:, None])
    np_est = np.maximum(mu - mean_d[:, None], 0.1)
    npq_est = np.maximum(var - var_d[:, None], 0.01)
    p_est = np.clip(1.0 - npq_est / np.maximum(np_est, 1e-6), 0.02, 0.98)
    n_cap = np.asarray(n_max, dtype=np.float64)[:, None]
    n_est = np.clip(np_est / np.maximum(p_est, 1e-6), 0.1, n_cap * 0.98)
    # keep the cluster's mean where n was clipped: a weak spectrum's noise spread over many bins
    # makes the variance exceed the mean, p_est is clipped low and n hits its cap, and the
    # start's mean (n p) collapsed to about 0.5 D wherever the data were; a start that does
    # not overlap the data never moves (HISTORY 176)
    p_est = np.clip(np_est / n_est, 0.02, 0.98)
    r = np.clip(n_est / n_cap, 1e-4, 1.0 - 1e-4)
    log_a = np.broadcast_to(
        np.log(np.maximum(total / np.maximum(k_eff, 1), 1.0))[:, None], (m, k)
    ).copy()
    n_raw = np.log(r / (1.0 - r))
    logit_p = np.log(p_est / (1.0 - p_est))
    for i in np.nonzero(k_eff < k)[0]:  # fewer distinct bins than components: jittered copies
        rng = np.random.default_rng(seeds[i])
        last = max(int(k_eff[i]), 1) - 1
        for c in range(last + 1, k):
            log_a[i, c] = log_a[i, c - 1] + rng.normal(0, 0.3)
            n_raw[i, c] = n_raw[i, c - 1] + rng.normal(0, 0.3)
            logit_p[i, c] = logit_p[i, c - 1] + rng.normal(0, 0.3)
    return log_a, n_raw, logit_p


def split_start(
    log_A: np.ndarray, n_raw: np.ndarray, p: np.ndarray, delta: float = 0.5
) -> Start:
    """A K-component start from a fitted (K-1)-component solution (scaled units, ``(M, K-1)``):
    the components are kept and the heaviest one of each measurement is split in two, with half
    the amplitude each and ``p`` shifted by -/+ ``delta`` on the logit scale. Measurements whose
    previous fit was not finite start from zeros."""
    la, nr = log_A.copy(), n_raw.copy()
    lp = logit(p)
    rows = np.arange(la.shape[0])
    heavy = np.argmax(la, axis=1)
    new_la = la[rows, heavy] - np.log(2.0)
    new_nr = nr[rows, heavy]
    new_lp = lp[rows, heavy] + delta
    la[rows, heavy] -= np.log(2.0)
    lp[rows, heavy] -= delta
    la = np.column_stack([la, new_la])
    nr = np.column_stack([nr, new_nr])
    lp = np.column_stack([lp, new_lp])
    bad = ~(np.isfinite(la).all(1) & np.isfinite(nr).all(1) & np.isfinite(lp).all(1))
    la[bad] = 0.0
    nr[bad] = 0.0
    lp[bad] = 0.0
    return Start(la, nr, lp)
