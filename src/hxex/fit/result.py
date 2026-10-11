# SPDX-License-Identifier: GPL-3.0-or-later
"""Fit results as NumPy arrays (no pandas): per-K fits, K selection and a tidy components table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..noise import NoiseEstimate
from .config import FitConfig


@dataclass(frozen=True)
class KFit:
    """The K-population fits of every measurement, arrays with one row per measurement.

    ``A``, ``n`` and ``p`` are ``(M, K)``; ``A`` is in the intensity units of the data.
    ``log_A`` (scaled units), ``n_raw`` and ``logit_p`` are the optimiser's own parameters, kept
    so that a fit can be resumed (nested starts, bootstrap). ``nll``, ``bic`` and ``aic`` are
    per measurement, in the units of the scaled likelihood. ``passes_min_pop`` is False where
    any population fraction is below ``FitConfig.min_population``.
    """

    k: int
    A: np.ndarray
    n: np.ndarray
    p: np.ndarray
    log_A: np.ndarray
    n_raw: np.ndarray
    logit_p: np.ndarray
    nll: np.ndarray
    bic: np.ndarray
    aic: np.ndarray
    passes_min_pop: np.ndarray

    @property
    def fractions(self) -> np.ndarray:
        """Population fraction ``A_k / sum(A)`` of every component, ``(M, K)``."""
        return self.A / np.maximum(self.A.sum(axis=1, keepdims=True), 1e-10)

    @property
    def means(self) -> np.ndarray:
        """Mean ``n * p`` (in deuterons) of every component, ``(M, K)``."""
        return self.n * self.p


@dataclass(frozen=True)
class FitResult:
    """Everything a fit produced; the order of measurements is that of the input.

    ``fits[K]`` for each K tried (``k_values``); ``selected_K`` is the K each measurement chose
    (the only K if one was tried). ``score`` is the criterion used for selection, ``(M, nK)``,
    with ``inf`` where the minimum-population filter rejected that K. ``obs_scale`` is the noise
    scale each spectrum was divided by, ``sample_size`` the ``N`` of the BIC penalty, ``n_max``
    the bound on ``n`` and ``noise_kind`` the noise model of each measurement.
    """

    config: FitConfig
    spectrum_id: np.ndarray
    data_id: np.ndarray
    time: np.ndarray
    k_values: tuple[int, ...]
    fits: dict[int, KFit]
    selected_K: np.ndarray
    score: np.ndarray
    obs_scale: np.ndarray
    sample_size: np.ndarray
    n_max: np.ndarray
    noise_kind: np.ndarray  # the noise model of each measurement
    noise_estimate: NoiseEstimate | None = None  # set when the noise was estimated

    def __len__(self) -> int:
        return len(self.selected_K)

    def grid(self, name: str) -> np.ndarray:
        """``nll``, ``bic``, ``aic`` or ``passes_min_pop`` for every K tried, ``(M, nK)``."""
        return np.column_stack([getattr(self.fits[k], name) for k in self.k_values])

    def selected(self, name: str) -> np.ndarray:
        """``A``, ``n``, ``p``, ``means`` or ``fractions`` at each measurement's selected K,
        ``(M, k_max)``, padded with NaN beyond that K."""
        width = max(self.k_values)
        out = np.full((len(self), width), np.nan)
        for k in self.k_values:
            rows = self.selected_K == k
            out[rows, :k] = getattr(self.fits[k], name)[rows]
        return out

    def table(self, selected_only: bool = False) -> dict[str, np.ndarray]:
        """One row per measurement, K and component, as a dict of equal-length columns.

        ``k_ordered`` ranks the components of a fit by ascending mean. Rows are ordered by
        measurement, then K, then component.
        """
        cols: dict[str, list] = {name: [] for name in _TABLE_COLUMNS}
        for k in self.k_values:
            fit = self.fits[k]
            order = np.argsort(np.argsort(fit.means, axis=1), axis=1) + 1
            frac = fit.fractions
            for m in range(len(self)):
                is_sel = bool(self.selected_K[m] == k)
                if selected_only and not is_sel:
                    continue
                for c in range(k):
                    row = {
                        "measurement": m, "spectrum_id": self.spectrum_id[m],
                        "data_id": self.data_id[m], "time": self.time[m],
                        "selected_K": self.selected_K[m], "fit_K": k, "is_selected": is_sel,
                        "component": c + 1, "k_ordered": order[m, c],
                        "amplitude": fit.A[m, c], "n": fit.n[m, c], "p": fit.p[m, c],
                        "population_fraction": frac[m, c], "mean": fit.n[m, c] * fit.p[m, c],
                        "variance": fit.n[m, c] * fit.p[m, c] * (1.0 - fit.p[m, c]),
                        "nll": fit.nll[m], "bic": fit.bic[m], "aic": fit.aic[m],
                        "passes_min_population": bool(fit.passes_min_pop[m]),
                    }
                    for name in _TABLE_COLUMNS:
                        cols[name].append(row[name])
        out = {name: np.array(values) for name, values in cols.items()}
        # row order is (K, measurement, component); sort to (measurement, K, component)
        order = np.lexsort((out["component"], out["fit_K"], out["measurement"]))
        return {name: arr[order] for name, arr in out.items()}


_TABLE_COLUMNS = (
    "measurement", "spectrum_id", "data_id", "time", "selected_K", "fit_K", "is_selected",
    "component", "k_ordered", "amplitude", "n", "p", "population_fraction", "mean", "variance",
    "nll", "bic", "aic", "passes_min_population",
)
