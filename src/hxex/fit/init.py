# SPDX-License-Identifier: GPL-3.0-or-later
"""Starting points for the optimiser: shape-corrected k-means and warm starts by splitting."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2

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
    """Starting ``(log_A, n_raw, logit_p)`` (each of length K) for one measurement.

    The intensity-weighted bins are clustered with k-means; each cluster gives a method-of-moments
    estimate corrected for the shape's own mean and variance::

        n*p        ~= mean(cluster) - mean(shape)
        n*p*(1-p)  ~= var(cluster)  - var(shape)

    ``n_raw`` is the inverse of the sigmoid parameterisation of ``n``.
    """
    rng = np.random.default_rng(seed)
    bc = np.asarray(bins, dtype=np.float64)
    ob = np.nan_to_num(np.asarray(observed, dtype=np.float64), nan=0.0).clip(min=0)

    j_arr = np.arange(len(shape), dtype=np.float64)
    mean_d = float(np.sum(j_arr * shape))
    var_d = float(np.sum((j_arr - mean_d) ** 2 * shape))

    counts_int = np.round(ob).astype(int).clip(min=0, max=2000)
    pseudo = np.repeat(bc, counts_int)
    if len(pseudo) < k:
        pseudo = bc.copy()

    k_eff = min(k, len(np.unique(pseudo)))
    centroids = None
    labels = None
    for attempt in range(12):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                cents, labs = kmeans2(
                    pseudo, k_eff, minit="points" if attempt == 0 else "random",
                    iter=30, seed=int(seed) + attempt * 17,
                )
            except Exception:  # noqa: BLE001 - kmeans2 raises several types on degenerate data
                continue
        for _ in range(k_eff):  # heal empty clusters by splitting the largest
            counts_k = np.array([np.sum(labs == i) for i in range(k_eff)])
            empty = np.where(counts_k == 0)[0]
            if len(empty) == 0:
                break
            largest = int(np.argmax(counts_k))
            for e in empty:
                cents[e] = cents[largest] + rng.uniform(-0.5, 0.5)
            labs = np.argmin(np.abs(pseudo[:, None] - cents[None, :]), axis=1)
        counts_k = np.array([np.sum(labs == i) for i in range(k_eff)])
        if np.all(counts_k > 0):
            centroids, labels = cents, labs
            break
    if centroids is None:
        centroids = np.linspace(bc.min(), bc.max(), k_eff)
        labels = np.argmin(np.abs(pseudo[:, None] - centroids[None, :]), axis=1)

    n_total = float(ob.sum())
    log_a, n_raw, logit_p = [], [], []
    for i in range(k_eff):
        pts = pseudo[labels == i]
        mu = float(pts.mean()) if len(pts) > 0 else float(centroids[i])
        var = float(pts.var()) if len(pts) > 1 else max(var_d + 0.5, 0.5)
        np_est = max(mu - mean_d, 0.1)
        npq_est = max(var - var_d, 0.01)
        p_est = float(np.clip(1.0 - npq_est / max(np_est, 1e-6), 0.02, 0.98))
        n_est = float(np.clip(np_est / max(p_est, 1e-6), 0.1, n_max * 0.98))
        log_a.append(np.log(max(n_total / k_eff, 1.0)))
        r = float(np.clip(n_est / n_max, 1e-4, 1.0 - 1e-4))
        n_raw.append(float(np.log(r / (1.0 - r))))
        logit_p.append(float(np.log(p_est / (1.0 - p_est))))
    while len(log_a) < k:  # fewer distinct bins than components: pad with jittered copies
        log_a.append(log_a[-1] + rng.normal(0, 0.3))
        n_raw.append(n_raw[-1] + rng.normal(0, 0.3))
        logit_p.append(logit_p[-1] + rng.normal(0, 0.3))
    return np.array(log_a[:k]), np.array(n_raw[:k]), np.array(logit_p[:k])


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
