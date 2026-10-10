# SPDX-License-Identifier: GPL-3.0-or-later
"""Fit binomial mixtures over a range of K and select K per measurement."""

from __future__ import annotations

import logging

import numpy as np

from ..hooks import CancelHook, ProgressHook, report
from ..model import aic, bic
from ..noise import noise_scale, sample_size
from .config import FitConfig
from .init import split_start
from .measurements import Measurements
from .optimise import fit_batch, prepare, stable_keys
from .result import FitResult, KFit

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
    measurement if one of its populations has a fraction below ``config.min_population``; if
    every K is rejected the smallest K is selected. With one K there is no selection.
    ``progress(done, total)`` counts optimiser restarts; ``cancel`` raises
    :class:`~hxex.hooks.Cancelled`.
    """
    backend = config.backend
    m = len(data)
    scale = noise_scale(data.observed, config.noise, data.references)
    prep = prepare(
        data.bins, data.observed, data.shapes, data.n_max, scale, config.noise.model, backend,
        keys=stable_keys(data.spectrum_id),
    )
    sizes = sample_size(prep.observed, config.noise)
    k_values = config.k_values
    total = config.n_restarts * len(k_values)
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        report(progress, done, total)

    fits: dict[int, KFit] = {}
    for k in k_values:
        warm = None
        if config.nested_init and (k - 1) in fits:
            prev = fits[k - 1]
            warm = split_start(prev.log_A, prev.n_raw, prev.p, config.split_delta)
        log.info("fitting K=%d for %d measurements", k, m)
        model, nll = fit_batch(
            prep, k, backend, n_restarts=config.n_restarts, adam_steps=config.adam_steps,
            lbfgs_steps=config.lbfgs_steps, lr_adam=config.lr_adam,
            lr_adam_end=config.lr_adam_end, seed=config.seed, warm=warm,
            warm_noise=config.warm_noise, kmeans_after_warm=config.kmeans_after_warm,
            grad_clip=config.grad_clip, polish_steps=config.polish_steps,
            cancel=cancel, on_restart=tick,
        )
        par = model.parameters_numpy()
        amplitudes = par["A"] * scale[:, None]
        fractions = amplitudes / np.maximum(amplitudes.sum(axis=1, keepdims=True), 1e-10)
        fits[k] = KFit(
            k=k, A=amplitudes, n=par["n"], p=par["p"],
            log_A=model.log_A.detach().cpu().numpy().astype(np.float64),
            n_raw=model.n_raw.detach().cpu().numpy().astype(np.float64),
            logit_p=model.logit_p.detach().cpu().numpy().astype(np.float64),
            nll=nll, bic=bic(k, sizes, nll), aic=aic(k, nll),
            passes_min_pop=np.all(fractions >= config.min_population, axis=1),
        )

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
        k_values=k_values, fits=fits, selected_K=selected, score=score, obs_scale=scale,
        sample_size=sizes, n_max=data.n_max,
    )
