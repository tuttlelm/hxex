# SPDX-License-Identifier: GPL-3.0-or-later
"""Fit results as NumPy arrays (no pandas): per-K fits, K selection and a tidy components table."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
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
    any population (a component's share) is below ``FitConfig.min_population``.
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
    def populations(self) -> np.ndarray:
        """Population (share of the molecules) ``A_k / sum(A)`` of every component, ``(M, K)``."""
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
    the bound on ``n`` and ``noise_kind`` the noise model of each measurement. ``role`` is each
    measurement's role (``'undeut'``, ``'fulldeut'`` or ``''``, the spectra's; all ``''`` when
    not given): the controls of a series are the measurements with a role.
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
    role: np.ndarray | None = None

    def __post_init__(self):
        if self.role is None:  # built without roles: time points only
            object.__setattr__(self, "role", np.full(len(self.spectrum_id), "", dtype="<U8"))
        else:
            object.__setattr__(self, "role", np.asarray(self.role, dtype=str))

    def __len__(self) -> int:
        return len(self.selected_K)

    @classmethod
    def merge(cls, parts: Sequence[FitResult], rows: Sequence[Sequence[int]]) -> FitResult:
        """One result from fits of separate groups of measurements: ``rows[j]`` are the positions
        (in the whole input) of the measurements of ``parts[j]``. The groups must cover every
        position once and share the settings; the merged order is that of the whole input."""
        if not parts or len(parts) != len(rows):
            raise ValueError("merge needs one list of rows per part")
        positions = np.concatenate([np.asarray(r, dtype=np.int64) for r in rows])
        if any(len(part) != len(r) for part, r in zip(parts, rows, strict=True)):
            raise ValueError("each part needs one row position per measurement")
        if not np.array_equal(np.sort(positions), np.arange(len(positions))):
            raise ValueError("the rows of the parts must cover every measurement exactly once")
        first = parts[0]
        for part in parts:
            if part.config != first.config or part.k_values != first.k_values:
                raise ValueError("the parts were fitted with different settings")
        order = np.argsort(positions)

        def cat(name: str) -> np.ndarray:
            return np.concatenate([getattr(p, name) for p in parts])[order]

        fits = {
            k: KFit(k=k, **{
                name: np.concatenate([getattr(p.fits[k], name) for p in parts])[order]
                for name in _KFIT_ARRAYS
            })
            for k in first.k_values
        }
        estimates = [p.noise_estimate for p in parts]
        if all(e is None for e in estimates):
            estimate = None
        elif any(e is None for e in estimates):
            raise ValueError("only some of the parts have a noise estimate")
        else:
            estimate = NoiseEstimate(**{
                f.name: np.concatenate([getattr(e, f.name) for e in estimates])[order]
                for f in dataclasses.fields(NoiseEstimate)
            })
        return cls(
            config=first.config, spectrum_id=cat("spectrum_id"), data_id=cat("data_id"),
            time=cat("time"), role=cat("role"), k_values=first.k_values, fits=fits,
            selected_K=cat("selected_K"),
            score=cat("score"), obs_scale=cat("obs_scale"), sample_size=cat("sample_size"),
            n_max=cat("n_max"), noise_kind=cat("noise_kind"), noise_estimate=estimate,
        )

    def select(self, rows) -> FitResult:
        """The result of the measurements at the positions ``rows`` (in that order), as if
        they had been fitted alone (the inverse of :meth:`merge`)."""
        rows = np.asarray(rows, dtype=np.int64)
        fits = {k: dataclasses.replace(f, **{name: getattr(f, name)[rows]
                                             for name in _KFIT_ARRAYS})
                for k, f in self.fits.items()}
        estimate = None
        if self.noise_estimate is not None:
            estimate = NoiseEstimate(**{
                f.name: getattr(self.noise_estimate, f.name)[rows]
                for f in dataclasses.fields(NoiseEstimate)
            })
        return dataclasses.replace(
            self, spectrum_id=self.spectrum_id[rows], data_id=self.data_id[rows],
            time=self.time[rows], role=self.role[rows], fits=fits,
            selected_K=self.selected_K[rows],
            score=self.score[rows], obs_scale=self.obs_scale[rows],
            sample_size=self.sample_size[rows], n_max=self.n_max[rows],
            noise_kind=self.noise_kind[rows], noise_estimate=estimate,
        )

    def fitted(self, k: int) -> np.ndarray:
        """Whether K was fitted for each measurement (False where the screen skipped it,
        :attr:`FitConfig.screen`; a fit that failed counts as fitted, with an ``inf`` loss)."""
        return ~np.isnan(self.fits[k].nll)

    def require_fitted(self, i: int, k: int) -> None:
        """Raise ValueError if K was not fitted for measurement ``i``."""
        if k not in self.fits:
            raise ValueError(f"K = {k} was not fitted (tried {self.k_values})")
        if not self.fitted(k)[i]:
            raise ValueError(
                f"{self.spectrum_id[i]}: K = {k} was not fitted (the screen showed it cannot be "
                "selected; fit without FitConfig.screen to have it)"
            )

    def grid(self, name: str) -> np.ndarray:
        """``nll``, ``bic``, ``aic`` or ``passes_min_pop`` for every K tried, ``(M, nK)``."""
        return np.column_stack([getattr(self.fits[k], name) for k in self.k_values])

    def selected(self, name: str) -> np.ndarray:
        """``A``, ``n``, ``p``, ``means`` or ``populations`` at each measurement's selected K,
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
        measurement, then K, then component. Fits the screen skipped have no rows.
        """
        cols: dict[str, list] = {name: [] for name in _TABLE_COLUMNS}
        for k in self.k_values:
            fit = self.fits[k]
            order = np.argsort(np.argsort(fit.means, axis=1), axis=1) + 1
            frac = fit.populations
            fitted = self.fitted(k)
            for m in range(len(self)):
                is_sel = bool(self.selected_K[m] == k)
                if (selected_only and not is_sel) or not fitted[m]:
                    continue
                for c in range(k):
                    row = {
                        "measurement": m, "spectrum_id": self.spectrum_id[m],
                        "data_id": self.data_id[m], "time": self.time[m],
                        "role": self.role[m],
                        "selected_K": self.selected_K[m], "fit_K": k, "is_selected": is_sel,
                        "component": c + 1, "k_ordered": order[m, c],
                        "amplitude": fit.A[m, c], "n": fit.n[m, c], "p": fit.p[m, c],
                        "population": frac[m, c], "mean": fit.n[m, c] * fit.p[m, c],
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


_KFIT_ARRAYS = (
    "A", "n", "p", "log_A", "n_raw", "logit_p", "nll", "bic", "aic", "passes_min_pop",
)

_TABLE_COLUMNS = (
    "measurement", "spectrum_id", "data_id", "time", "role", "selected_K", "fit_K", "is_selected",
    "component", "k_ordered", "amplitude", "n", "p", "population", "mean", "variance",
    "nll", "bic", "aic", "passes_min_population",
)
