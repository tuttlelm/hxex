# SPDX-License-Identifier: GPL-3.0-or-later
"""Bootstrap tests. They use few resamples and light fits to stay fast; the calibration of the
standard errors against the spread over independent noise realisations (about a minute) runs when
HXEX_FULL_REGRESSION=1."""

import json
import os

import numpy as np
import pytest

from hxex.bootstrap import (
    BootstrapConfig,
    BootstrapResult,
    bootstrap,
    match_components,
)
from hxex.fit import FitConfig, Measurements, fit_models
from hxex.hooks import Cancelled
from hxex.model import Backend
from hxex.noise import NoiseConfig
from test_fit import BIMODAL, TRIMODAL, UNIMODAL, synthetic  # noqa: I001
from test_noise_estimate import measurements_with_replicates  # noqa: I001

CPU = Backend("cpu", "float64")
NOISE = NoiseConfig("gaussian", sd=45.0, percent=None)  # about the noise of synthetic()
LIGHT = dict(adam_steps=150, n_restarts=2)
BOOT = BootstrapConfig(n_boot=10, n_restarts=1, adam_steps=100, polish_steps=20)


def fitted(specs, **kw):
    meas, truth = synthetic(specs)
    cfg = FitConfig(noise=NOISE, backend=CPU, **{**LIGHT, **kw})
    return meas, fit_models(meas, cfg), truth


@pytest.fixture(scope="module")
def small():
    meas, res, truth = fitted([BIMODAL, BIMODAL])  # K = 1 to 3 tried, 2 selected
    boot = bootstrap(meas, res, BootstrapConfig(**{**BOOT.to_dict(), "keep_draws": True}))
    return meas, res, boot


# --- configuration --------------------------------------------------------------------------

def test_config_roundtrip_and_validation():
    cfg = BootstrapConfig(n_boot=50, ci=90.0, keep_draws=True, seed=3)
    assert BootstrapConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    assert BootstrapConfig().n_boot == 200 and BootstrapConfig().ci == 95.0
    with pytest.raises(ValueError, match="unknown bootstrap setting"):
        BootstrapConfig.from_dict({"n": 5})
    for bad in ({"n_boot": 1}, {"ci": 100.0}, {"ci": 0.0}, {"n_restarts": 0}, {"adam_steps": -1},
                {"lr_adam": 0.0}, {"warm_noise": -1.0}, {"seed": -1}, {"grad_clip": "x"}):
        with pytest.raises(ValueError):
            BootstrapConfig(**bad)


# --- matching the components of a draw to the original fit ----------------------------------

def test_match_components_reorders_swapped_draws():
    ref = np.array([[3.0, 11.0]])
    draws = np.array([[[3.1, 10.9], [11.2, 2.8], [np.nan, np.nan]]])  # (1, 3, 2)
    perm = match_components(ref, draws)
    assert perm[0, 0].tolist() == [0, 1] and perm[0, 1].tolist() == [1, 0]
    ordered = np.take_along_axis(draws, perm, axis=2)
    np.testing.assert_allclose(ordered[0, 1], [2.8, 11.2])


def test_match_components_three_and_many():
    ref = np.array([[1.0, 5.0, 9.0]])
    draws = np.array([[[9.1, 1.2, 4.8]]])
    ordered = np.take_along_axis(draws, match_components(ref, draws), axis=2)
    np.testing.assert_allclose(ordered[0, 0], [1.2, 4.8, 9.1])
    ref6 = np.arange(6.0)[None]  # more components than the vectorised limit: Hungarian
    draws6 = np.array([[[5.1, 0.2, 4.0, 1.1, 2.9, 2.0]]])
    out6 = np.take_along_axis(draws6, match_components(ref6, draws6), axis=2)
    np.testing.assert_allclose(out6[0, 0], [0.2, 1.1, 2.0, 2.9, 4.0, 5.1])


def test_match_components_with_k_one_and_all_nan():
    perm = match_components(np.array([[4.0]]), np.array([[[4.1], [np.nan]]]))
    assert perm.shape == (1, 2, 1) and perm.sum() == 0


# --- the result -----------------------------------------------------------------------------

def test_result_structure_and_sanity(small):
    meas, res, b = small
    assert isinstance(b, BootstrapResult) and len(b) == 2
    assert list(b.n_success) == [10, 10] and list(b.selected_K) == [2, 2]
    for arr in (b.A_se, b.n_se, b.p_se, b.mean_se, b.population_se):
        assert arr.shape == (2, 3)  # padded to the largest K tried
        assert np.all(arr[:, :2] > 0) and np.all(np.isnan(arr[:, 2]))
    assert b.mean_ci.shape == (2, 3, 2)
    assert np.all(b.mean_ci[:, :2, 0] <= b.mean_ci[:, :2, 1])
    fit_means, ci = res.fits[2].means, b.mean_ci[:, :2]
    assert np.all(ci[..., 0] - 0.3 <= fit_means) and np.all(fit_means <= ci[..., 1] + 0.3)
    assert np.all(b.mean_se[:, :2] < 0.5)  # a well-determined spectrum: well under 0.5 D


def test_the_curve_band_brackets_the_fit(small):
    meas, res, b = small
    assert len(b.curve_lo) == sum(len(x) for x in meas.bins)
    lo, hi = b.curve_band(0)
    assert lo.shape == meas.bins[0].shape and np.all(lo <= hi) and np.all(hi - lo >= 0)
    assert (hi - lo).max() > 0
    # the band is in the units of the data
    assert hi.max() == pytest.approx(meas.observed[0].max(), rel=0.1)


def test_draws_are_consistent_with_the_summaries(small):
    _, res, b = small
    d = b.draws
    assert d.A.shape == (2, 10, 3) and d.ok.shape == (2, 10) and d.ok.all()
    np.testing.assert_allclose(b.n_se[:, :2], np.nanstd(d.n, axis=1, ddof=1)[:, :2])
    mean = d.n * d.p
    np.testing.assert_allclose(b.mean_se[:, :2], np.nanstd(mean, axis=1, ddof=1)[:, :2])
    np.testing.assert_allclose(
        b.mean_ci[:, :2, 0], np.nanpercentile(mean, 2.5, axis=1)[:, :2]
    )
    # the components keep the order of the original fit: no label swapping between draws
    fit_means = res.fits[2].means[:, None, :]
    assert np.all(np.abs(mean[:, :, :2] - fit_means) < 1.0)
    assert b.draws is not None and BootstrapConfig().keep_draws is False


def test_without_draws_and_without_the_band():
    meas, res, _ = fitted([BIMODAL])
    b = bootstrap(meas, res, BootstrapConfig(**{**BOOT.to_dict(), "curve_band": False}))
    assert b.draws is None and np.all(np.isnan(b.curve_lo)) and b.n_success[0] == 10


def test_mixed_k_and_padding():
    meas, res, _ = fitted([UNIMODAL, BIMODAL, TRIMODAL])
    b = bootstrap(meas, res, BOOT)
    assert list(b.selected_K) == list(res.selected_K)
    for m, k in enumerate(res.selected_K):
        assert np.all(np.isfinite(b.mean_se[m, :k])) and np.all(np.isnan(b.mean_se[m, k:]))
        assert b.n_success[m] == 10


@pytest.mark.parametrize("k", [1, 2])
def test_a_single_k_can_be_bootstrapped(k):
    meas, res, _ = fitted([UNIMODAL, BIMODAL], k_min=k, k_max=k)
    b = bootstrap(meas, res, BOOT)
    assert list(b.selected_K) == [k, k] and np.all(np.isfinite(b.mean_se[:, :k]))


def test_a_measurement_whose_fit_failed_is_skipped_and_the_others_are_not():
    meas, _ = synthetic([BIMODAL, BIMODAL, BIMODAL])
    obs = [o.copy() for o in meas.observed]
    obs[1][3] = np.nan
    bad = Measurements.from_arrays(meas.bins, obs, meas.shapes, meas.n_max,
                                   spectrum_id=meas.spectrum_id)
    res = fit_models(bad, FitConfig(noise=NOISE, backend=CPU, k_max=2, **LIGHT))
    b = bootstrap(bad, res, BOOT)
    assert list(b.n_success) == [10, 0, 10]  # the others are not discarded with it
    assert np.all(np.isnan(b.mean_se[1])) and np.all(np.isfinite(b.mean_se[[0, 2]][:, :2]))
    lo, _ = b.curve_band(1)
    assert np.all(np.isnan(lo))


def test_poisson_noise_model_resamples_counts():
    meas, _ = synthetic([BIMODAL], amplitude=3e3, rel_noise=0.0)
    meas = Measurements.from_arrays(meas.bins, meas.observed, meas.shapes, meas.n_max,
                                    references=[(0,)], spectrum_id=meas.spectrum_id)
    res = fit_models(meas, FitConfig(noise=NoiseConfig("poisson", n_eff=3000.0, percent=None),
                                     backend=CPU, k_max=2, **LIGHT))
    b = bootstrap(meas, res, BOOT)
    assert b.n_success[0] == 10 and np.all(b.mean_se[0] > 0)



def test_the_bootstrap_handles_estimated_noise():
    meas = measurements_with_replicates()
    cfg = FitConfig(noise=NoiseConfig("gaussian", percent=1.0, estimate="replicates"),
                    backend=CPU, k_max=1, adam_steps=100, n_restarts=1, polish_steps=0)
    res = fit_models(meas, cfg)
    b = bootstrap(meas, res, BootstrapConfig(n_boot=4, n_restarts=1, adam_steps=40,
                                             polish_steps=0))
    assert list(b.n_success) == [4] * len(meas) and np.all(b.mean_se[:, 0] > 0)


# --- reproducibility and independence -------------------------------------------------------

def test_same_seed_same_result_and_a_measurement_does_not_depend_on_the_others():
    meas, res, _ = fitted([BIMODAL, UNIMODAL, BIMODAL], k_max=2)
    a = bootstrap(meas, res, BOOT)
    b = bootstrap(meas, res, BOOT)
    np.testing.assert_array_equal(a.mean_se, b.mean_se)
    c = bootstrap(meas, res, BootstrapConfig(**{**BOOT.to_dict(), "seed": 9}))
    assert not np.array_equal(a.mean_se, c.mean_se)
    # bootstrap a subset of the same fit: the same numbers for those measurements
    pick = [0, 2]
    sub = Measurements.from_arrays(
        [meas.bins[i] for i in pick], [meas.observed[i] for i in pick],
        [meas.shapes[i] for i in pick], meas.n_max[pick], spectrum_id=meas.spectrum_id[pick])
    sub_fit = fit_models(sub, FitConfig(noise=NOISE, backend=CPU, **LIGHT))
    both = bootstrap(sub, sub_fit, BOOT)
    again = bootstrap(sub, sub_fit, BOOT)
    np.testing.assert_array_equal(both.mean_se, again.mean_se)


def test_progress_cancel_and_mismatched_inputs():
    meas, res, _ = fitted([BIMODAL])
    calls = []
    cfg = BootstrapConfig(n_boot=4, n_restarts=1, adam_steps=30, polish_steps=0)
    bootstrap(meas, res, cfg, progress=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]
    flags = iter([False, False, True])
    with pytest.raises(Cancelled):
        bootstrap(meas, res, cfg, cancel=lambda: next(flags, True))
    other, _ = synthetic([BIMODAL, BIMODAL])
    with pytest.raises(ValueError, match="measurements the fit was made from"):
        bootstrap(other, res, cfg)


# --- the table ------------------------------------------------------------------------------

def test_table_adds_the_interval_columns_for_the_selected_k(small):
    meas, res, b = small
    t = b.table(res, selected_only=False)
    base = res.table()
    assert len(t["fit_K"]) == len(base["fit_K"])
    for name in ("amplitude", "n", "p", "mean", "population"):
        for suffix in ("se", "ci_lo", "ci_hi"):
            assert f"{name}_{suffix}" in t
    sel = t["is_selected"]
    assert np.all(np.isfinite(t["mean_se"][sel])) and np.all(np.isnan(t["mean_se"][~sel]))
    assert np.all(t["n_bootstrap"][sel] == 10) and np.all(np.isnan(t["n_bootstrap"][~sel]))
    assert np.all(t["mean_ci_lo"][sel] <= t["mean"][sel] + 0.3)
    only = b.table(res)
    assert len(only["fit_K"]) == int(res.selected_K.sum())
    first = np.flatnonzero((only["measurement"] == 0) & (only["component"] == 1))[0]
    assert only["mean_se"][first] == b.mean_se[0, 0]


# --- calibration ----------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("HXEX_FULL_REGRESSION") != "1",
                    reason="set HXEX_FULL_REGRESSION=1 (about a minute)")
def test_standard_errors_match_the_spread_over_independent_noise_realisations():
    clean, _ = synthetic([BIMODAL], seed=11, rel_noise=0.0)
    sd = 0.004 * clean.observed[0].max()  # the noise sd the generator uses
    cfg = FitConfig(noise=NoiseConfig("gaussian", sd=sd, percent=None), backend=CPU,
                    k_min=2, k_max=2)
    meas, _ = synthetic([BIMODAL] * 40, seed=11)
    res = fit_models(meas, cfg)
    order = np.argsort(res.fits[2].means, axis=1)
    means = np.take_along_axis(res.fits[2].means, order, axis=1)
    frac = np.take_along_axis(res.fits[2].populations, order, axis=1)
    pick = [0, 1, 2]
    sub = Measurements.from_arrays([meas.bins[i] for i in pick], [meas.observed[i] for i in pick],
                                   [meas.shapes[i] for i in pick], meas.n_max[pick],
                                   spectrum_id=meas.spectrum_id[pick])
    sub_fit = fit_models(sub, cfg)
    b = bootstrap(sub, sub_fit, BootstrapConfig(n_boot=50))
    o = np.argsort(sub_fit.fits[2].means, axis=1)
    boot_mean = np.mean([b.mean_se[i][o[i]] for i in range(3)], axis=0)
    boot_frac = np.mean([b.population_se[i][o[i]] for i in range(3)], axis=0)
    np.testing.assert_allclose(boot_mean / means.std(axis=0, ddof=1), 1.0, atol=0.3)
    np.testing.assert_allclose(boot_frac / frac.std(axis=0, ddof=1), 1.0, atol=0.3)
