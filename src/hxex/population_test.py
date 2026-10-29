# SPDX-License-Identifier: GPL-3.0-or-later
"""The population test: whether a spectrum, or a set of spectra, holds K + 1 populations rather
than K (docs/ANALYSIS.md "The population test").

The statistic of a spectrum at step K (K against K + 1) is ``T = 2 (nll_K - nll_K+1)`` from its
own fits, 0 where negative or below the precision of the losses (:func:`statistic_of`). Its
null is simulated: data drawn from the spectrum's K-population fit with the fit's noise (the
noise-scaled curve plus unit Gaussian noise, as the bootstrap draws it) and refitted at 1 to
K + 1 as the run fits, ``simulations`` times; each spectrum keeps the mean and variance of its
null draws. A set of spectra adds up their statistics and the
moments of their nulls (independent spectra), and a gamma with those moments gives its p-value
(:func:`gamma_tail`; a chi-square null is a gamma too). A spectrum or set has at least K + 1
populations when ``p < significance`` and the added population (the smallest share of the
K + 1 fits, averaged over the set) is at least ``min_population``; the steps run from K = 1 and
stop at the first that does not reject.

The width check, a flag only (owner, 2026-10-26: not yet justified as a rule; HISTORY 175): a
population of mean mu deuterons whose amides each hold at most the D2O fraction r of the
labelling buffer has a variance of at least mu (1 - r). A K + 1 fit with a component of at
least ``WIDTH_MIN_MEAN`` D and ``WIDTH_MIN_SHARE`` narrower than ``WIDTH_FACTOR``
sqrt(mu (1 - r)) may hold something other than deuterium (a co-eluting species on the isotope
grid, a single-bin spike); the spectrum is flagged (``narrow``), and its test is unchanged.

With :attr:`hxex.fit.FitConfig.screen` the run fits K + 1 only where the loss floor leaves the
statistic room to reach 1 (:func:`hxex.fit.screen.can_exceed`); a skipped spectrum enters
with T = 0 and the chi-square null of 3 degrees of freedom (the three parameters a population
adds) instead of a simulation. ``null = 'chi_square'`` gives every spectrum that null (no
simulation): a quick look, not the results (HISTORY 162, 169).
"""
# evidence: docs/HISTORY.md items 139, 161-162, 168-173

from __future__ import annotations

import contextlib
import dataclasses
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy.stats import gamma

from .fit import FitConfig, FitResult, Measurements, fit_models
from .hooks import CancelHook, ProgressHook, check_cancel, report
from .model import MixtureBatch, PaddedBatch, shape_weights
from .noise import NoiseConfig, bin_variances

log = logging.getLogger(__name__)

NULLS = ("simulated", "chi_square")
CHI_SQUARE_DF = 3  # the parameters one more population adds: amplitude, n, p
WIDTH_FACTOR = 0.3
WIDTH_MIN_MEAN = 1.0  # D
WIDTH_MIN_SHARE = 0.10
DEFAULT_D2O_FRACTION = 0.9  # [run.controls] d2o_fraction's default, for runs without controls
MIN_DRAWS = 0.5  # share of the simulations that must give a statistic for a spectrum's null
# a statistic below the precision of the losses is 0 (float32 losses are not exact; the same
# tolerance as the screen's, hxex.fit.screen): an exact fit at K and K + 1 otherwise gives
# rounding noise against rounding noise, and an arbitrary p-value (HISTORY 176)
LOSS_TOLERANCE_ABS = 1e-3
LOSS_TOLERANCE_REL = 1e-4


def statistic_of(nll_k, nll_k1) -> np.ndarray:
    """``2 (nll_K - nll_K+1)``, 0 where it is negative or below the precision of the losses."""
    nll_k, nll_k1 = np.asarray(nll_k, dtype=float), np.asarray(nll_k1, dtype=float)
    t = 2.0 * (nll_k - nll_k1)
    tol = 2.0 * (LOSS_TOLERANCE_ABS + LOSS_TOLERANCE_REL * np.abs(nll_k))
    with np.errstate(invalid="ignore"):
        return np.where(t > tol, t, np.where(np.isnan(t), np.nan, 0.0))

# kinds of a spectrum's null at a step
SIMULATED, CHI_SQUARE, SCREENED, UNTESTED = "simulated", "chi_square", "screened", ""


@dataclass(frozen=True)
class PopulationTestConfig:
    """Settings of the population test (the ``[run.population_test]`` table; the table turns
    the stage on).

    - ``simulations``: null simulations per spectrum and step (20: at 20 the sets' verdicts
      agree with 100 simulations on 97-99 % of sets).
    - ``significance``: a step adds a population when its p-value is below this (0.01).
    - ``min_population``: ... and the added population's share is at least this (0.10).
    - ``null``: ``'simulated'`` (each spectrum's own null) or ``'chi_square'`` (no
      simulation: a quick look, about 96 % of the sets' verdicts of the simulated null).
    - ``seed``: of the null simulations.
    """
    # the test and its numbers: docs/ANALYSIS.md "The population test"

    simulations: int = 20
    significance: float = 0.01
    min_population: float = 0.10
    null: str = "simulated"
    seed: int = 0

    def __post_init__(self):
        if self.simulations < 2:
            raise ValueError("simulations must be at least 2")
        if not 0.0 < self.significance < 1.0:
            raise ValueError("significance must be between 0 and 1")
        if not 0.0 <= self.min_population < 0.5:
            raise ValueError("min_population must be in [0, 0.5)")
        if self.null not in NULLS:
            raise ValueError(f"null must be one of {', '.join(NULLS)}")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PopulationTestConfig:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown population test setting(s): {', '.join(sorted(unknown))}")
        return cls(**data)


def check_fit_config(fit: FitConfig) -> None:
    """Raise ValueError if the fit settings cannot carry the test."""
    if fit.k_min != 1:
        raise ValueError("the population test needs [run.fit] k_min = 1 (it tests one "
                         "population against two first)")
    if fit.k_max < 2:
        raise ValueError("the population test needs [run.fit] k_max of 2 or more")
    if fit.noise.model != "gaussian":
        raise ValueError("the population test needs the Gaussian noise model (its null draws "
                         "the fit's Gaussian noise)")


@dataclass(frozen=True)
class PopulationTest:
    """The test of every spectrum of a fit (rows in the fit's order) at each step
    (``steps``: K = 1, 2, ... against K + 1; columns of the ``(M, S)`` arrays).

    ``statistic`` is T (NaN where untested), ``null_mean`` and ``null_var`` the moments of its
    null, ``null_kind`` how the null was made (``'simulated'``, ``'chi_square'``,
    ``'screened'``: the screen skipped the K + 1 fit, T = 0 with the chi-square null; ``''``:
    untested, the fits missing or failed), ``null_draws`` the simulated statistics ``(M, S,
    simulations)`` (NaN for the others), ``added_population`` the smallest share of the K + 1
    fit, ``narrow`` the width check's flag (the K + 1 fit has a component narrower than
    deuterium allows; a flag only), ``p`` the spectrum's own p-value. ``selected_K`` is each
    spectrum's K by the test (tested as a set of one), ``criterion_K`` the K the fit's
    criterion chose. ``d2o_fraction`` is the r of the width check."""

    config: PopulationTestConfig
    steps: tuple[int, ...]
    statistic: np.ndarray
    null_mean: np.ndarray
    null_var: np.ndarray
    null_kind: np.ndarray
    null_draws: np.ndarray
    added_population: np.ndarray
    narrow: np.ndarray
    p: np.ndarray
    selected_K: np.ndarray
    criterion_K: np.ndarray
    d2o_fraction: float

    def __len__(self) -> int:
        return len(self.selected_K)

    def column(self, k: int) -> int:
        """The column of step K (K against K + 1)."""
        return self.steps.index(k)

    def tested(self) -> np.ndarray:
        """``(M, S)``: whether each spectrum has a statistic and a null at each step."""
        return self.null_kind != UNTESTED

    @classmethod
    def merge(cls, parts: Sequence[PopulationTest], rows: Sequence[Sequence[int]]
              ) -> PopulationTest:
        """One result from the tests of separate groups of measurements (as
        :meth:`hxex.fit.FitResult.merge`)."""
        first = parts[0]
        if any(p.config != first.config or p.steps != first.steps
               or p.d2o_fraction != first.d2o_fraction for p in parts):
            raise ValueError("the parts were tested with different settings")
        positions = np.concatenate([np.asarray(r, dtype=np.int64) for r in rows])
        order = np.argsort(positions)
        arrays = {name: np.concatenate([getattr(p, name) for p in parts])[order]
                  for name in _ARRAYS}
        return dataclasses.replace(first, **arrays)

    def select(self, rows) -> PopulationTest:
        rows = np.asarray(rows, dtype=np.int64)
        return dataclasses.replace(self, **{name: getattr(self, name)[rows]
                                            for name in _ARRAYS})


_ARRAYS = ("statistic", "null_mean", "null_var", "null_kind", "null_draws", "added_population",
           "narrow", "p", "selected_K", "criterion_K")


def gamma_tail(t, mean, var) -> np.ndarray:
    """The p-value of statistics ``t`` under nulls with these moments: the upper tail of the
    gamma with that mean and variance (a chi-square null is one). NaN where the moments are not
    finite; where the null has no spread, 1 if ``t`` is at most its mean, else 0."""
    t, mean, var = (np.asarray(x, dtype=float) for x in (t, mean, var))
    t, mean, var = np.broadcast_arrays(t, mean, var)
    out = np.full(t.shape, np.nan)
    ok = np.isfinite(t) & np.isfinite(mean) & np.isfinite(var)
    spread = ok & (mean > 0) & (var > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[spread] = gamma.sf(np.maximum(t[spread], 0.0), a=mean[spread] ** 2 / var[spread],
                               scale=var[spread] / mean[spread])
    flat = ok & ~spread
    out[flat] = np.where(t[flat] <= mean[flat], 1.0, 0.0)
    return out


def chi_square_moments() -> tuple[float, float]:
    """Mean and variance of the chi-square null of one spectrum and step."""
    return float(CHI_SQUARE_DF), float(2 * CHI_SQUARE_DF)


def narrow_components(fit: FitResult, k: int, d2o_fraction: float) -> np.ndarray:
    """``(M,)``: whether the K-population fit of each spectrum has a component of at least
    ``WIDTH_MIN_MEAN`` D and ``WIDTH_MIN_SHARE`` narrower than ``WIDTH_FACTOR`` times the
    narrowest a deuterated population of its mean can be (sd ``sqrt(mu (1 - r))``)."""
    f = fit.fits[k]
    n, p, share = (np.asarray(x, dtype=float) for x in (f.n, f.p, f.populations))
    mu = n * p
    with np.errstate(invalid="ignore"):
        sd = np.sqrt(np.clip(n * p * (1.0 - p), 0.0, None))
        bound = WIDTH_FACTOR * np.sqrt(np.clip(mu, 0.0, None) * (1.0 - d2o_fraction))
        hit = (sd < bound) & (mu >= WIDTH_MIN_MEAN) & (share >= WIDTH_MIN_SHARE)
    return np.asarray(hit.any(axis=1), dtype=bool)


def steps_K(p: np.ndarray, added: np.ndarray, usable: np.ndarray,
            config: PopulationTestConfig) -> np.ndarray:
    """The number of populations of each row from its steps (``(N, S)`` p-values, added
    populations, and whether the step can be used): one, plus one for every step in order that
    rejects (``p < significance``, added population at least ``min_population``), up to the
    first that does not."""
    n_rows, n_steps = p.shape
    k = np.ones(n_rows, dtype=np.int64)
    going = np.ones(n_rows, dtype=bool)
    for s in range(n_steps):
        with np.errstate(invalid="ignore"):
            rejects = (usable[:, s] & (p[:, s] < config.significance)
                       & (added[:, s] >= config.min_population))
        going &= rejects
        k += going
    return k


def population_test(
    data: Measurements,
    fit: FitResult,
    config: PopulationTestConfig = PopulationTestConfig(),  # noqa: B008
    *,
    d2o_fraction: float = DEFAULT_D2O_FRACTION,
    test_screen: bool = False,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> PopulationTest:
    """Test every spectrum of ``fit`` (made from ``data``) at each step from K = 1 to
    ``k_max - 1`` (see the module docstring). A fit made with the screen must have been made
    with the test's screen (``fit_models(..., test_screen=True)``; say so with
    ``test_screen``): the criterion's screen skips fits whose small gains the sets add up.
    ``progress(done, total)`` counts simulations; ``cancel`` raises
    :class:`~hxex.hooks.Cancelled`."""
    if len(data) != len(fit) or not np.array_equal(data.spectrum_id, fit.spectrum_id):
        raise ValueError("data must be the measurements the fit was made from")
    check_fit_config(fit.config)
    if fit.config.screen and not test_screen:
        raise ValueError("the fit was made with the criterion's screen: fit with "
                         "fit_models(..., test_screen=True) for the population test")
    steps = tuple(range(1, fit.config.k_max))
    m, n_sim, n_steps = len(fit), config.simulations, len(steps)
    shape = (m, n_steps)
    statistic, null_mean, null_var = np.full(shape, np.nan), np.full(shape, np.nan), np.full(
        shape, np.nan)
    kind = np.full(shape, UNTESTED, dtype=object)
    draws = np.full((m, n_steps, n_sim), np.nan, dtype=np.float32)
    added = np.full(shape, np.nan)
    narrow = np.zeros(shape, dtype=bool)
    chi_mean, chi_var = chi_square_moments()
    simulate = config.null == "simulated"
    plans = []
    for s, k in enumerate(steps):
        lo, hi = fit.fits[k].nll, fit.fits[k + 1].nll
        both = np.isfinite(lo) & np.isfinite(hi)
        # the screen skipped K + 1 (or K): the loss floor bounds T below 1
        screened = fit.config.screen & (np.isnan(hi) | np.isnan(lo)) & ~(np.isinf(lo)
                                                                         | np.isinf(hi))
        statistic[both, s] = statistic_of(lo[both], hi[both])
        statistic[screened, s] = 0.0
        kind[screened, s] = SCREENED
        null_mean[screened, s], null_var[screened, s] = chi_mean, chi_var
        added[both, s] = np.nanmin(fit.fits[k + 1].populations[both], axis=1)
        narrow[:, s] = both & narrow_components(fit, k + 1, d2o_fraction)
        if simulate:
            plans.append((s, k, np.flatnonzero(both)))
        else:
            kind[both, s] = CHI_SQUARE
            null_mean[both, s], null_var[both, s] = chi_mean, chi_var
    total = sum(len(rows) for _, _, rows in plans) * n_sim
    done = 0
    for s, k, rows in plans:
        if not len(rows):
            continue
        log.info("population test, %d against %d: %d null simulations of %d spectra", k, k + 1,
                 n_sim, len(rows))

        def tick(n: int) -> None:
            nonlocal done
            done += n
            report(progress, done, total)

        sim = simulate_null(data, fit, rows, k, n_sim, seed=config.seed, cancel=cancel,
                            on_simulation=lambda: tick(len(rows)))  # noqa: B023
        draws[rows, s] = sim
        enough = np.isfinite(sim).sum(axis=1) >= MIN_DRAWS * n_sim
        clipped = np.clip(sim, 0.0, None)
        with np.errstate(invalid="ignore"), _warnings_off():
            mean, var = np.nanmean(clipped, axis=1), np.nanvar(clipped, axis=1)
        good = rows[enough]
        null_mean[good, s], null_var[good, s] = mean[enough], var[enough]
        kind[good, s] = SIMULATED
        if (~enough).any():
            log.warning("population test, %d against %d: the null simulations failed for %d "
                        "spectra, which are not tested at this step", k, k + 1,
                        int((~enough).sum()))
    tested = kind != UNTESTED
    p = np.where(tested, gamma_tail(statistic, null_mean, null_var), np.nan)
    added = np.where(kind == SCREENED, np.nan, added)  # no K + 1 fit to say
    usable = tested
    selected = steps_K(p, added, usable, config)
    criterion = np.asarray(fit.selected_K, dtype=np.int64)
    if narrow.any():
        log.info("population test: %d spectra have a component narrower than deuterium "
                 "allows (narrow_component in spectra.csv; a flag only)",
                 int(narrow.any(axis=1).sum()))
    log.info("population test: %d spectra read one population, %d two%s (the criterion: %d, "
             "%d)", int((selected == 1).sum()), int((selected == 2).sum()),
             "" if n_steps < 2 else f", {int((selected >= 3).sum())} three or more",
             int((criterion == 1).sum()), int((criterion == 2).sum()))
    return PopulationTest(
        config=config, steps=steps, statistic=statistic, null_mean=null_mean, null_var=null_var,
        null_kind=kind.astype(str), null_draws=draws, added_population=added, narrow=narrow,
        p=p, selected_K=selected, criterion_K=criterion, d2o_fraction=float(d2o_fraction),
    )


@contextlib.contextmanager
def _warnings_off():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


@contextlib.contextmanager
def _quiet_fits():
    """The fit's per-K log lines, once per simulation, kept out of the log."""
    logger = logging.getLogger("hxex.fit.models")
    level = logger.level
    logger.setLevel(max(level, logging.WARNING))
    try:
        yield
    finally:
        logger.setLevel(level)


def scaled_prediction(data: Measurements, fit: FitResult, rows: np.ndarray, k: int
                      ) -> list[np.ndarray]:
    """The noise-scaled fitted curve at K = k of the measurements at ``rows``."""
    backend = fit.config.backend
    kf = fit.fits[k]
    bins = [data.bins[i] for i in rows]
    batch = PaddedBatch.from_arrays(bins, [np.zeros(len(b)) for b in bins], data.n_max[rows],
                                    backend)
    shape_w = shape_weights([data.shapes[i] for i in rows], k, backend)
    model = MixtureBatch(batch, shape_w, backend.tensor(kf.log_A[rows]),
                         backend.tensor(kf.n_raw[rows]), backend.tensor(kf.logit_p[rows]),
                         [str(fit.noise_kind[i]) for i in rows])
    pred = model.predict().detach().cpu().numpy()
    return [pred[: len(b), j].astype(np.float64) for j, b in enumerate(bins)]


def simulate_null(
    data: Measurements, fit: FitResult, rows: np.ndarray, k: int, n_sim: int, *, seed: int = 0,
    cancel: CancelHook | None = None, on_simulation=None,
) -> np.ndarray:
    """``(len(rows), n_sim)`` null statistics ``2 (nll_K - nll_K+1)`` of the measurements at
    ``rows``: data drawn from their K-population fit (noise-scaled curve plus unit Gaussian
    noise, or with per-bin noise (:attr:`FitResult.noise_curve`) each bin's own sd, the
    simulations refitted with the same weights), refitted at 1 to K + 1 with the run's fit
    settings (no screen). Each draw is seeded
    by ``seed``, the simulation and what the spectrum is (``random_id``), so the result does not
    depend on the other spectra or the project's name. NaN where a refit failed."""
    rows = np.asarray(rows, dtype=np.int64)
    cfg = dataclasses.replace(fit.config, k_min=1, k_max=k + 1, screen=False,
                              noise=NoiseConfig(model="gaussian", sd=1.0, percent=None))
    pred = scaled_prediction(data, fit, rows, k)
    keys = data.random_keys()[rows]
    bins = [data.bins[i] for i in rows]
    shapes = [data.shapes[i] for i in rows]
    n_max = data.n_max[rows]
    sd = [np.ones(len(b)) for b in bins]  # noise sd of every bin, in the scaled units
    if fit.noise_curve is not None:
        variances = bin_variances([data.observed[i] for i in rows], data.charges()[rows],
                                  fit.noise_curve)
        sd = [np.sqrt(v) / fit.obs_scale[i] for v, i in zip(variances, rows, strict=True)]
    out = np.full((len(rows), n_sim), np.nan)
    for s in range(n_sim):
        check_cancel(cancel)
        observed = [curve + w * np.random.default_rng([seed, 11, k, s, int(key)]).standard_normal(
            len(curve)) for curve, key, w in zip(pred, keys, sd, strict=True)]
        ids = [f"null{k}.{s}:{sid}" for sid in data.spectrum_id[rows]]
        rids = [f"null{k}.{s}:{rid}" for rid in data.random_id[rows]]
        sim = Measurements.from_arrays(bins, observed, shapes, n_max, spectrum_id=ids,
                                       random_id=rids)
        with _quiet_fits():
            res = fit_models(sim, cfg, cancel=cancel,
                             bin_variance=None if fit.noise_curve is None else
                             [w * w for w in sd])
        lo, hi = res.fits[k].nll, res.fits[k + 1].nll
        out[:, s] = np.where(np.isfinite(lo) & np.isfinite(hi), statistic_of(lo, hi), np.nan)
        if on_simulation is not None:
            on_simulation()
    return out
