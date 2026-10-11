# SPDX-License-Identifier: GPL-3.0-or-later
"""Parametric bootstrap: uncertainties on populations, fractions and the fitted curve.

For every measurement the fitted curve is resampled with the noise model of the fit (Gaussian
with unit sd on the noise-scaled spectrum, or Poisson) and refitted at the measurement's selected
K, starting from the original fit. Components are matched to the original fit by their mean
``n * p``, which identifies a population (the variance must not enter the matching: for components
of similar variance it swaps the labels). Standard errors and percentile intervals are reported for
the amplitude, ``n``, ``p``, the mean ``n * p`` and the population fraction, plus a per-bin band of
the total fitted curve (which accounts for the correlation between components). For weakly
populated components ``n`` and ``p`` are strongly anticorrelated, so their intervals can be wide
while the mean and the fraction stay well determined.

Measurements are independent: a resample that fails for one measurement is dropped for that
measurement only (``n_success`` counts the ones that count; a measurement whose own fit failed is
not resampled at all), and every random number of a measurement comes from a generator seeded by
the seed, the iteration and the spectrum id.
"""

from __future__ import annotations

import itertools
import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .fit import FitError, FitResult, Measurements, Start, fit_batch, prepare, stable_keys
from .hooks import CancelHook, ProgressHook, check_cancel, report
from .model import MixtureBatch, PaddedBatch, convolve, shape_weights

log = logging.getLogger(__name__)

_PERMUTATION_LIMIT = 4  # up to this K all permutations are compared at once


@dataclass(frozen=True)
class BootstrapConfig:
    """Settings of :func:`bootstrap`.

    ``n_boot`` resamples, intervals at ``ci`` percent. Each refit uses ``n_restarts`` starts (the
    first is the original fit with ``warm_noise`` jitter), ``adam_steps`` Adam steps with a cosine
    learning rate from ``lr_adam`` to ``lr_adam_end`` and ``polish_steps`` Levenberg-Marquardt
    iterations. ``keep_draws`` keeps the individual fits of the selected K (``A``, ``n`` and
    ``p`` of every successful resample); ``curve_band`` computes the band of the total curve.
    """

    n_boot: int = 200
    ci: float = 95.0
    n_restarts: int = 3
    adam_steps: int = 400
    lr_adam: float = 0.02
    lr_adam_end: float = 0.002
    warm_noise: float = 0.10
    polish_steps: int = 40
    grad_clip: str = "per_measurement"
    seed: int = 42
    keep_draws: bool = False
    curve_band: bool = True

    def __post_init__(self):
        if self.n_boot < 2:
            raise ValueError("n_boot must be at least 2")
        if not 0.0 < self.ci < 100.0:
            raise ValueError("ci must be between 0 and 100 (percent)")
        if self.n_restarts < 1 or self.adam_steps < 0 or self.polish_steps < 0:
            raise ValueError("n_restarts must be >= 1; adam_steps and polish_steps >= 0")
        if not self.lr_adam > 0 or not self.lr_adam_end > 0:
            raise ValueError("learning rates must be positive")
        if self.warm_noise < 0 or self.seed < 0:
            raise ValueError("warm_noise and seed must be non-negative")
        if self.grad_clip not in ("global", "per_measurement"):
            raise ValueError("grad_clip must be 'global' or 'per_measurement'")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BootstrapConfig:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown bootstrap setting(s): {', '.join(sorted(unknown))}")
        return cls(**data)


@dataclass(frozen=True)
class BootstrapDraws:
    """The individual fits at the selected K: ``(M, n_boot, k_max)`` arrays padded with NaN beyond
    each measurement's K, in the component order of the original fit; ``ok`` is ``(M, n_boot)``."""

    A: np.ndarray
    n: np.ndarray
    p: np.ndarray
    ok: np.ndarray


@dataclass(frozen=True)
class BootstrapResult:
    """Standard errors (``*_se``, ``(M, k_max)``) and percentile intervals (``*_ci``,
    ``(M, k_max, 2)``) at each measurement's selected K, padded with NaN beyond that K, in the
    component order of the original fit. ``n_success`` counts the successful resamples of each
    measurement (statistics need at least 2). ``curve_lo`` and ``curve_hi`` are the band of the
    total fitted curve in intensity units: flat arrays, with spectrum ``i`` in
    ``curve_offsets[i]:curve_offsets[i + 1]``."""

    config: BootstrapConfig
    selected_K: np.ndarray
    n_success: np.ndarray
    A_se: np.ndarray
    n_se: np.ndarray
    p_se: np.ndarray
    mean_se: np.ndarray
    frac_se: np.ndarray
    A_ci: np.ndarray
    n_ci: np.ndarray
    p_ci: np.ndarray
    mean_ci: np.ndarray
    frac_ci: np.ndarray
    curve_lo: np.ndarray
    curve_hi: np.ndarray
    curve_offsets: np.ndarray
    draws: BootstrapDraws | None = None

    def __len__(self) -> int:
        return len(self.selected_K)

    @classmethod
    def merge(
        cls, parts: Sequence[BootstrapResult], rows: Sequence[Sequence[int]]
    ) -> BootstrapResult:
        """One result from bootstraps of separate groups of measurements (``rows[j]`` are the
        positions of ``parts[j]`` in the whole input, covering every position once)."""
        if not parts or len(parts) != len(rows):
            raise ValueError("merge needs one list of rows per part")
        positions = np.concatenate([np.asarray(r, dtype=np.int64) for r in rows])
        if any(len(part) != len(r) for part, r in zip(parts, rows, strict=True)):
            raise ValueError("each part needs one row position per measurement")
        if not np.array_equal(np.sort(positions), np.arange(len(positions))):
            raise ValueError("the rows of the parts must cover every measurement exactly once")
        if any(p.config != parts[0].config for p in parts):
            raise ValueError("the parts were made with different settings")
        if len({p.draws is None for p in parts}) != 1:
            raise ValueError("only some of the parts have draws")
        order = np.argsort(positions)

        def cat(name: str) -> np.ndarray:
            return np.concatenate([getattr(p, name) for p in parts])[order]

        where = np.empty((len(positions), 2), dtype=np.int64)  # part and local row of each
        for j, r in enumerate(rows):
            where[np.asarray(r, dtype=np.int64)] = np.column_stack(
                [np.full(len(r), j), np.arange(len(r))]
            )
        lo, hi, offsets = [], [], [0]
        for j, local in where:
            a, b = parts[j].curve_offsets[local], parts[j].curve_offsets[local + 1]
            lo.append(parts[j].curve_lo[a:b])
            hi.append(parts[j].curve_hi[a:b])
            offsets.append(offsets[-1] + (b - a))
        draws = None
        if parts[0].draws is not None:
            draws = BootstrapDraws(*(
                np.concatenate([getattr(p.draws, name) for p in parts])[order]
                for name in ("A", "n", "p", "ok")
            ))
        names = ("selected_K", "n_success", "A_se", "n_se", "p_se", "mean_se", "frac_se", "A_ci",
                 "n_ci", "p_ci", "mean_ci", "frac_ci")
        return cls(
            config=parts[0].config, **{name: cat(name) for name in names},
            curve_lo=np.concatenate(lo), curve_hi=np.concatenate(hi),
            curve_offsets=np.asarray(offsets, dtype=np.int64), draws=draws,
        )

    def curve_band(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """``(lo, hi)`` of the total fitted curve of spectrum ``i``."""
        a, b = self.curve_offsets[i], self.curve_offsets[i + 1]
        return self.curve_lo[a:b], self.curve_hi[a:b]

    def table(self, fit: FitResult, selected_only: bool = True) -> dict[str, np.ndarray]:
        """``fit.table()`` with the interval columns added for the selected K (NaN for the other
        rows): ``<name>_se``, ``<name>_ci_lo`` and ``<name>_ci_hi`` for ``amplitude``, ``n``,
        ``p``, ``mean`` and ``population_fraction``, and ``n_bootstrap``."""
        base = fit.table(selected_only=selected_only)
        rows = len(base["measurement"])
        sources = {
            "amplitude": (self.A_se, self.A_ci), "n": (self.n_se, self.n_ci),
            "p": (self.p_se, self.p_ci), "mean": (self.mean_se, self.mean_ci),
            "population_fraction": (self.frac_se, self.frac_ci),
        }
        cols = {f"{name}_{suffix}": np.full(rows, np.nan)
                for name in sources for suffix in ("se", "ci_lo", "ci_hi")}
        cols["n_bootstrap"] = np.full(rows, np.nan)
        for r in range(rows):
            if not base["is_selected"][r]:
                continue
            m, c = int(base["measurement"][r]), int(base["component"][r]) - 1
            cols["n_bootstrap"][r] = self.n_success[m]
            for name, (se, ci) in sources.items():
                cols[f"{name}_se"][r] = se[m, c]
                cols[f"{name}_ci_lo"][r], cols[f"{name}_ci_hi"][r] = ci[m, c]
        return {**base, **cols}


def match_components(ref_means: np.ndarray, draw_means: np.ndarray) -> np.ndarray:
    """Component order of every draw that best matches the reference by the mean ``n * p``.

    ``ref_means`` is ``(M, K)``, ``draw_means`` ``(M, N, K)``. Returns ``perm`` ``(M, N, K)`` so
    that ``take_along_axis(draw, perm, axis=2)`` lists the draw's components in the reference
    order (minimum total squared difference of the means).
    """
    m, n, k = draw_means.shape
    if k <= _PERMUTATION_LIMIT:
        perms = np.array(list(itertools.permutations(range(k))))  # (P, K)
        picked = draw_means[:, :, perms]  # (M, N, P, K)
        cost = ((picked - ref_means[:, None, None, :]) ** 2).sum(axis=-1)  # (M, N, P)
        cost = np.where(np.isnan(cost), np.inf, cost)
        return perms[np.argmin(cost, axis=-1)]
    perm = np.zeros((m, n, k), dtype=np.int64)
    for i in range(m):
        for j in range(n):
            cost = (ref_means[i][:, None] - draw_means[i, j][None, :]) ** 2
            if not np.isfinite(cost).all():
                perm[i, j] = np.arange(k)
                continue
            rows, cols = linear_sum_assignment(cost)
            perm[i, j, rows] = cols
    return perm


def _nan_stats(x: np.ndarray, ok: np.ndarray, ci: float):
    """SE (ddof 1) and percentile interval of ``x`` (``(M, N, K)``) over the draws where ``ok``."""
    masked = np.where(ok[:, :, None], x, np.nan)
    enough = ok.sum(axis=1) >= 2
    lo, hi = (100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        se = np.nanstd(masked, axis=1, ddof=1)
        quant = np.nanpercentile(masked, [lo, hi], axis=1)  # (2, M, K)
    se[~enough] = np.nan
    interval = np.moveaxis(quant, 0, -1)  # (M, K, 2)
    interval[~enough] = np.nan
    return se, interval


def bootstrap(
    data: Measurements,
    fit: FitResult,
    config: BootstrapConfig = BootstrapConfig(),  # noqa: B008
    *,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> BootstrapResult:
    """Bootstrap every measurement of ``fit`` at its selected K (see the module docstring).

    ``data`` must be the measurements ``fit`` was made from. The noise model, the noise scale and
    the device come from the fit. ``progress(done, total)`` counts resamples; ``cancel`` raises
    :class:`~hxex.hooks.Cancelled`.
    """
    if len(data) != len(fit) or not np.array_equal(data.spectrum_id, fit.spectrum_id):
        raise ValueError("data must be the measurements the fit was made from")
    backend, kinds = fit.config.backend, fit.noise_kind
    m_total, n_boot = len(fit), config.n_boot
    k_max = max(fit.k_values)
    scale, keys = fit.obs_scale, stable_keys(data.spectrum_id)
    # a measurement whose own fit failed (non-finite loss, placeholder parameters) is not resampled
    fit_ok = np.array([np.isfinite(fit.fits[int(k)].nll[i]) for i, k in enumerate(fit.selected_K)])
    groups = {
        int(k): np.flatnonzero((fit.selected_K == k) & fit_ok) for k in np.unique(fit.selected_K)
    }
    groups = {k: idx for k, idx in groups.items() if len(idx)}

    draw_a = np.full((m_total, n_boot, k_max), np.nan)
    draw_n, draw_p = draw_a.copy(), draw_a.copy()
    ok = np.zeros((m_total, n_boot), dtype=bool)

    # the fitted curve (noise-scaled) and the starting point of every group
    ref: dict[int, dict[str, Any]] = {}
    for k, idx in groups.items():
        kf = fit.fits[k]
        bins = [data.bins[i] for i in idx]
        batch = PaddedBatch.from_arrays(
            bins, [np.zeros(len(b)) for b in bins], data.n_max[idx], backend
        )
        shape_w = shape_weights([data.shapes[i] for i in idx], k, backend)
        model = MixtureBatch(
            batch, shape_w, backend.tensor(kf.log_A[idx]), backend.tensor(kf.n_raw[idx]),
            backend.tensor(kf.logit_p[idx]), [str(kinds[i]) for i in idx],
        )
        pred = model.predict().detach().cpu().numpy()
        ref[k] = {
            "start": Start(kf.log_A[idx], kf.n_raw[idx], kf.logit_p[idx]),
            "pred": [pred[: len(b), j].astype(np.float64) for j, b in enumerate(bins)],
            "means": kf.means[idx], "bins": bins, "batch": batch, "shape_w": shape_w,
        }

    for it in range(n_boot):
        check_cancel(cancel)
        for k, idx in groups.items():
            info = ref[k]
            observed = []
            for j, m in enumerate(idx):
                rng = np.random.default_rng([config.seed, 3, it, int(keys[m])])
                pred = info["pred"][j]
                if kinds[m] == "gaussian":
                    observed.append(pred + rng.standard_normal(len(pred)))
                else:
                    observed.append(rng.poisson(np.maximum(pred, 0.0)).astype(np.float64))
            prep = prepare(
                info["bins"], observed, [data.shapes[i] for i in idx], data.n_max[idx],
                np.ones(len(idx)), [str(kinds[i]) for i in idx], backend, keys=keys[idx],
            )
            try:
                model, nll = fit_batch(
                    prep, k, backend, n_restarts=config.n_restarts, adam_steps=config.adam_steps,
                    lbfgs_steps=0, lr_adam=config.lr_adam, lr_adam_end=config.lr_adam_end,
                    seed=config.seed + 1000 * (it + 1), warm=info["start"],
                    warm_noise=config.warm_noise, grad_clip=config.grad_clip,
                    polish_steps=config.polish_steps, cancel=cancel,
                )
            except FitError:
                log.warning("resample %d, K=%d: every measurement failed", it, k)
                continue
            par = model.parameters_numpy()
            amp = par["A"] * scale[idx][:, None]
            good = (
                np.isfinite(nll) & np.isfinite(amp).all(1)
                & np.isfinite(par["n"]).all(1) & np.isfinite(par["p"]).all(1)
            )
            draw_a[idx[good], it, :k] = amp[good]
            draw_n[idx[good], it, :k] = par["n"][good]
            draw_p[idx[good], it, :k] = par["p"][good]
            ok[idx[good], it] = True
        report(progress, it + 1, n_boot)

    # put the components of every draw in the order of the original fit
    for k, idx in groups.items():
        sub = (slice(None), slice(None), slice(0, k))
        a, n, p = draw_a[idx][sub], draw_n[idx][sub], draw_p[idx][sub]
        perm = match_components(ref[k]["means"], n * p)
        for arr, src in ((draw_a, a), (draw_n, n), (draw_p, p)):
            arr[idx, :, :k] = np.take_along_axis(src, perm, axis=2)

    mean_d = draw_n * draw_p
    frac_d = draw_a / np.maximum(np.nansum(draw_a, axis=2, keepdims=True), 1e-10)
    stats = {name: _nan_stats(x, ok, config.ci) for name, x in
             (("A", draw_a), ("n", draw_n), ("p", draw_p), ("mean", mean_d), ("frac", frac_d))}

    curve_lo, curve_hi, offsets = _curve_band(data, fit, ref, groups, draw_a, draw_n, draw_p, ok,
                                              config, backend, scale)
    draws = BootstrapDraws(draw_a, draw_n, draw_p, ok) if config.keep_draws else None
    return BootstrapResult(
        config=config, selected_K=fit.selected_K.copy(), n_success=ok.sum(axis=1),
        A_se=stats["A"][0], n_se=stats["n"][0], p_se=stats["p"][0], mean_se=stats["mean"][0],
        frac_se=stats["frac"][0], A_ci=stats["A"][1], n_ci=stats["n"][1], p_ci=stats["p"][1],
        mean_ci=stats["mean"][1], frac_ci=stats["frac"][1], curve_lo=curve_lo,
        curve_hi=curve_hi, curve_offsets=offsets, draws=draws,
    )


def _curve_band(data, fit, ref, groups, draw_a, draw_n, draw_p, ok, config, backend, scale):
    """Per-bin percentiles of the total curve over the resamples (intensity units)."""
    sizes = np.array([len(b) for b in data.bins])
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    lo_all = np.full(offsets[-1], np.nan)
    hi_all = np.full(offsets[-1], np.nan)
    if not config.curve_band:
        return lo_all, hi_all, offsets
    lo_q, hi_q = (100.0 - config.ci) / 2.0, 100.0 - (100.0 - config.ci) / 2.0
    chunk = 400
    for k, idx in groups.items():
        for start in range(0, len(idx), chunk):
            rows = np.arange(start, min(start + chunk, len(idx)))
            members = idx[rows]
            bins = [ref[k]["bins"][r] for r in rows]
            batch = PaddedBatch.from_arrays(
                bins, [np.zeros(len(b)) for b in bins], data.n_max[members], backend
            )
            shape_w = shape_weights([data.shapes[i] for i in members], k, backend)
            curves = np.full((config.n_boot, batch.b.shape[0], len(members)), np.nan)
            for it in range(config.n_boot):
                good = ok[members, it]
                if not good.any():
                    continue
                a = draw_a[members, it, :k] / scale[members][:, None]  # noise-scaled amplitudes
                comp = convolve(
                    batch.b, backend.tensor(np.nan_to_num(draw_n[members, it, :k], nan=1.0)),
                    backend.tensor(np.nan_to_num(draw_p[members, it, :k], nan=0.5)), shape_w,
                ).detach().cpu().numpy()  # (B, M, K)
                total = (np.nan_to_num(a, nan=0.0)[None] * comp).sum(axis=2)
                curves[it][:, good] = total[:, good]
            enough = ok[members].sum(axis=1) >= 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lo = np.nanpercentile(curves, lo_q, axis=0) * scale[members][None, :]
                hi = np.nanpercentile(curves, hi_q, axis=0) * scale[members][None, :]
            for j, m in enumerate(members):
                if enough[j]:
                    lo_all[offsets[m]:offsets[m + 1]] = lo[: sizes[m], j]
                    hi_all[offsets[m]:offsets[m + 1]] = hi[: sizes[m], j]
    return lo_all, hi_all, offsets
