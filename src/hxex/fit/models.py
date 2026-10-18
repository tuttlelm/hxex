# SPDX-License-Identifier: GPL-3.0-or-later
"""Fit binomial mixtures over a range of K and select K per measurement."""

from __future__ import annotations

import logging

import numpy as np

from ..hooks import CancelHook, ProgressHook, report
from ..model import aic, bic
from ..noise import estimate_noise, noise_scale, sample_size
from .config import FitConfig
from .init import split_start
from .measurements import Measurements
from .optimise import fit_batch, prepare, stable_keys
from .result import FitResult, KFit
from .screen import can_win, loss_floor

log = logging.getLogger(__name__)


def fit_models(
    data: Measurements,
    config: FitConfig = FitConfig(),  # noqa: B008
    *,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> FitResult:
    """Fit K = ``config.k_min`` to ``config.k_max`` populations to every measurement in parallel
    and select K per measurement by BIC or AIC.

    Each spectrum is divided by its noise scale (:func:`hxex.noise.noise_scale`) before fitting,
    and the fitted amplitudes are returned in the original intensity units. A K is rejected for a
    measurement if one of its components has a population below ``config.min_population``; if
    every K is rejected the smallest K is selected. With one K there is no selection. If
    ``config.noise.estimate`` is set, the noise of each series with enough replicates is estimated
    from them (see :func:`hxex.noise.estimate_noise`); the result keeps the estimate and the flags.
    ``progress(done, total)`` counts optimiser restarts; ``cancel`` raises
    :class:`~hxex.hooks.Cancelled`.

    With ``config.screen`` each K above 1 is fitted only for the measurements that could still
    select it (:mod:`hxex.fit.screen`); the others get NaN for that K (parameters, loss and
    criteria) and an ``inf`` score. The selected K is the same as without the screen.
    """
    backend = config.backend
    m = len(data)
    scale = noise_scale(data.observed, config.noise, data.references)
    kinds = [config.noise.model] * m
    estimate = None
    if config.noise.estimate:
        # series with enough replicates keep the Gaussian loss, with the sd their estimated gain
        # gives at the spectrum's mean bin intensity (variance = gain x intensity, so the
        # expected sum of squared residuals is the estimated one); the others keep the noise
        # settings above. The Poisson loss is never chosen here (docs/HISTORY.md item 57).
        estimate = estimate_noise(
            data.bins, data.observed, data.data_id, data.time, config.noise, role=data.role
        )
        mean_intensity = np.array([float(np.mean(o)) for o in data.observed])
        sd = np.sqrt(estimate.gain * mean_intensity)
        usable = estimate.estimated & np.isfinite(sd) & (sd > 0)
        scale = np.where(usable, sd, scale)
        n_series = len(np.unique(data.data_id))
        n_est = len(np.unique(data.data_id[estimate.estimated]))
        log.info("noise estimated from replicates for %d of %d series", n_est, n_series)
        if estimate.outlier.any():
            log.warning(
                "%d spectra are inconsistent with their replicates (score above %g) and were "
                "left out of the noise estimate", int(estimate.outlier.sum()),
                config.noise.outlier_score,
            )
    prep = prepare(
        data.bins, data.observed, data.shapes, data.n_max, scale, kinds, backend,
        keys=stable_keys(data.spectrum_id),
    )
    sizes = sample_size(prep.observed, prep.noise_kind)
    k_values = config.k_values
    extra = len(k_values) - 1 if config.nested_init and config.kmeans_after_warm else 0
    total = config.n_restarts * len(k_values) + extra  # the k-means start of a nested K
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        report(progress, done, total)

    floor = None
    if config.screen and any(k > 1 for k in k_values):  # nothing to skip with K = 1 alone
        floor = loss_floor(prep.bins, prep.observed, prep.shapes, prep.n_max, prep.noise_kind)
    best = np.full(m, np.inf)  # best criterion so far (for the screen)
    fits: dict[int, KFit] = {}
    for k in k_values:
        rows = np.arange(m)
        if floor is not None and k > 1:
            rows = np.flatnonzero(can_win(config.criterion, k, sizes, floor, best))
            log.info("screen: K=%d can be selected for %d of %d measurements", k, len(rows), m)
        warm = None
        if config.nested_init and (k - 1) in fits:
            prev = fits[k - 1]
            warm = split_start(prev.log_A[rows], prev.n_raw[rows], prev.p[rows],
                               config.split_delta)
        if len(rows) == 0:
            fits[k] = _unfitted(k, m)
            for _ in range(config.n_restarts + (1 if warm is not None and extra else 0)):
                tick()
            continue
        sub = prep if len(rows) == m else prep.select(rows, backend)
        log.info("fitting K=%d for %d measurements", k, len(rows))
        model, nll = fit_batch(
            sub, k, backend, n_restarts=config.n_restarts, adam_steps=config.adam_steps_for(k),
            lbfgs_steps=config.lbfgs_steps, lr_adam=config.lr_adam,
            lr_adam_end=config.lr_adam_end, seed=config.seed, warm=warm,
            warm_noise=config.warm_noise, kmeans_after_warm=config.kmeans_after_warm,
            grad_clip=config.grad_clip, polish_steps=config.polish_steps,
            cancel=cancel, on_restart=tick,
        )
        par = model.parameters_numpy()
        amplitudes = par["A"] * scale[rows, None]
        fractions = amplitudes / np.maximum(amplitudes.sum(axis=1, keepdims=True), 1e-10)
        fit = _unfitted(k, m)
        fit.A[rows], fit.n[rows], fit.p[rows] = amplitudes, par["n"], par["p"]
        fit.log_A[rows] = model.log_A.detach().cpu().numpy()
        fit.n_raw[rows] = model.n_raw.detach().cpu().numpy()
        fit.logit_p[rows] = model.logit_p.detach().cpu().numpy()
        fit.nll[rows] = nll
        fit.bic[rows] = bic(k, sizes[rows], nll)
        fit.aic[rows] = aic(k, nll)
        fit.passes_min_pop[rows] = np.all(fractions >= config.min_population, axis=1)
        fits[k] = fit
        score_k = getattr(fit, config.criterion)
        best = np.where(fit.passes_min_pop & (score_k < best), score_k, best)

    raw = np.column_stack(
        [getattr(fits[k], config.criterion) for k in k_values]
    )
    if len(k_values) == 1:
        score = raw
    else:
        passes = np.column_stack([fits[k].passes_min_pop for k in k_values])
        score = np.where(passes, raw, np.inf)
    selected = np.array(k_values)[np.argmin(score, axis=1)]
    return FitResult(
        config=config, spectrum_id=data.spectrum_id, data_id=data.data_id, time=data.time,
        role=data.role,
        k_values=k_values, fits=fits, selected_K=selected, score=score, obs_scale=scale,
        sample_size=sizes, n_max=data.n_max, noise_kind=np.array(prep.noise_kind, dtype=str),
        noise_estimate=estimate,
    )


def _unfitted(k: int, m: int) -> KFit:
    """A K-population result for M measurements with nothing fitted: NaN everywhere (the
    screen skipped them) and no K passing the minimum population."""
    def nan(*shape):
        return np.full(shape, np.nan)

    return KFit(
        k=k, A=nan(m, k), n=nan(m, k), p=nan(m, k), log_A=nan(m, k), n_raw=nan(m, k),
        logit_p=nan(m, k), nll=nan(m), bic=nan(m), aic=nan(m),
        passes_min_pop=np.zeros(m, dtype=bool),
    )
