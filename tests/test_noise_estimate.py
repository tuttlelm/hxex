# SPDX-License-Identifier: GPL-3.0-or-later
"""Noise estimated from replicates: the estimator, its safeguards, and its use in the fit."""

import json
import logging

import numpy as np
import pytest
import torch

from hxex.fit import FitConfig, Measurements, fit_models
from hxex.model import Backend, MixtureBatch, PaddedBatch, shape_weights
from hxex.noise import NoiseConfig, estimate_noise, sample_size
from test_fit import BINS, SHAPE, synthetic  # noqa: I001

CPU = Backend("cpu", "float64")
GAIN = 60.0  # variance = GAIN * intensity


def curve(means_fracs, amplitude=2e5):
    """The noise-free spectrum of a mixture with the helpers' n = 14 (see test_fit)."""
    meas, _ = synthetic([means_fracs], rel_noise=0.0, amplitude=amplitude)
    return meas.observed[0]


def series_data(n_series=2, times=(0.0, 60.0, 300.0, 1e6), reps=3, gain=GAIN, seed=0,
                scale_spread=0.2, replicated=None):
    """Replicate spectra with variance = gain * intensity and a random scale per replicate.

    Returns the arguments of ``estimate_noise`` plus the clean curves. ``replicated`` lists, per
    series, the number of replicates at each time (default ``reps`` everywhere)."""
    rng = np.random.default_rng(seed)
    clean = [curve([(2.0, 1.0)]), curve([(5.0, 0.5), (11.0, 0.5)]),
             curve([(8.0, 1.0)]), curve([(12.0, 1.0)])]
    bins, obs, ids, times_out, truth = [], [], [], [], []
    for s in range(n_series):
        for ti, t in enumerate(times):
            n = reps if replicated is None else replicated[s][ti]
            for _ in range(n):
                scale = 1.0 + scale_spread * rng.standard_normal()
                mu = clean[ti % len(clean)] * scale
                obs.append(np.maximum(mu + rng.normal(0, 1, len(mu)) * np.sqrt(gain * mu), 0.0))
                bins.append(BINS.copy())
                ids.append(s)
                times_out.append(t)
                truth.append(ti)
    return bins, obs, np.array(ids), np.array(times_out), np.array(truth)


# --- configuration --------------------------------------------------------------------------

def test_estimate_settings_validation_and_roundtrip():
    cfg = NoiseConfig(estimate="replicates", min_replicated=4, min_share=0.01, outlier_score=6.0)
    assert NoiseConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    assert NoiseConfig().estimate is None  # off by default: nothing changes
    for bad in ({"estimate": "residuals"}, {"min_replicated": 0}, {"min_share": 0.0},
                {"min_share": 1.0}, {"outlier_score": 0.0}):
        with pytest.raises(ValueError):
            NoiseConfig(**bad)


# --- the estimator --------------------------------------------------------------------------

def test_the_gain_is_recovered():
    bins, obs, ids, times, _ = series_data(n_series=2, seed=1)
    est = estimate_noise(bins, obs, ids, times, NoiseConfig(estimate="replicates"))
    assert est.estimated.all() and np.all(est.reason == "")
    assert np.all(est.n_replicated == 4)
    for s in (0, 1):
        g = est.gain[ids == s]
        assert np.all(g == g[0]) and g[0] == pytest.approx(GAIN, rel=0.2)  # one gain per series


def test_a_second_gain_in_another_series_is_told_apart():
    b1, o1, i1, t1, _ = series_data(n_series=1, gain=30.0, seed=2)
    b2, o2, i2, t2, _ = series_data(n_series=1, gain=120.0, seed=3)
    est = estimate_noise(b1 + b2, o1 + o2, np.concatenate([i1, i2 + 1]),
                         np.concatenate([t1, t2]), NoiseConfig(estimate="replicates"))
    g = [est.gain[np.concatenate([i1, i2 + 1]) == s][0] for s in (0, 1)]
    assert g[0] == pytest.approx(30.0, rel=0.25) and g[1] == pytest.approx(120.0, rel=0.25)


def test_the_scale_of_a_replicate_does_not_count_as_noise():
    flat = series_data(n_series=1, scale_spread=0.0, seed=4)
    wild = series_data(n_series=1, scale_spread=0.6, seed=4)
    cfg = NoiseConfig(estimate="replicates")
    g0 = estimate_noise(*flat[:4], cfg).gain[0]
    g1 = estimate_noise(*wild[:4], cfg).gain[0]
    assert g1 == pytest.approx(g0, rel=0.25)  # the total of each spectrum is rescaled away


def test_replicates_with_different_numbers_of_bins_are_aligned():
    bins, obs, ids, times, _ = series_data(n_series=1, seed=5)
    # drop the zero-filled tail of some replicates, as the picker does
    cut = [len(b) - (k % 4) for k, b in enumerate(bins)]
    bins = [b[:c] for b, c in zip(bins, cut, strict=True)]
    obs = [o[:c] for o, c in zip(obs, cut, strict=True)]
    est = estimate_noise(bins, obs, ids, times, NoiseConfig(estimate="replicates"))
    assert est.estimated.all() and est.gain[0] == pytest.approx(GAIN, rel=0.3)


# --- the safeguards -------------------------------------------------------------------------

def test_a_bad_spectrum_is_flagged_and_cannot_move_the_estimate():
    bins, obs, ids, times, _ = series_data(n_series=1, reps=4, seed=6)
    cfg = NoiseConfig(estimate="replicates")
    clean = estimate_noise(bins, obs, ids, times, cfg)
    bad = [o.copy() for o in obs]
    victim = 5  # a replicate of the second time point: the shape of another deuteration level
    bad[victim] = curve([(9.0, 1.0)]) * (bad[victim].sum() / curve([(9.0, 1.0)]).sum())
    est = estimate_noise(bins, bad, ids, times, cfg)
    assert est.outlier[victim] and est.score[victim] > 4
    assert est.outlier.sum() <= 2  # the others are not condemned with it
    assert est.gain[0] == pytest.approx(clean.gain[0], rel=0.25)  # robust: not inflated
    assert np.all(np.isfinite(est.score[[i for i in range(len(obs)) if i != victim]][:3]))


def test_the_median_is_robust_where_a_plain_pooled_variance_is_not():
    bins, obs, ids, times, _ = series_data(n_series=1, reps=4, seed=7)
    bad = [o.copy() for o in obs]
    for victim in (2, 9):  # two spectra with far too much noise
        bad[victim] = bad[victim] + np.random.default_rng(victim).normal(0, 1, len(BINS)) * (
            0.3 * bad[victim])
    est = estimate_noise(bins, bad, ids, times, NoiseConfig(estimate="replicates"))
    assert est.gain[0] < 2.5 * GAIN  # a plain pooled estimate would be several times too high
    assert est.outlier[[2, 9]].all()


def test_series_without_enough_replicates_are_not_estimated():
    # series 0 has no replicates at all, series 1 has only two replicated time points
    bins, obs, ids, times, _ = series_data(
        n_series=3, replicated=[[1, 1, 1, 1], [2, 2, 1, 1], [3, 3, 3, 3]], seed=8
    )
    est = estimate_noise(bins, obs, ids, times, NoiseConfig(estimate="replicates"))
    for s, ok, why in ((0, False, "no replicates"), (1, False, "2 replicated"), (2, True, "")):
        sel = ids == s
        assert est.estimated[sel].all() == ok and np.all(np.isnan(est.gain[sel]) != ok)
        assert all(why in r for r in est.reason[sel])
    relaxed = estimate_noise(bins, obs, ids, times, NoiseConfig(estimate="replicates",
                                                                  min_replicated=2))
    assert relaxed.estimated[ids == 1].all()


def test_pairs_give_a_gain_but_no_score():
    bins, obs, ids, times, _ = series_data(n_series=1, reps=2, seed=9)
    est = estimate_noise(bins, obs, ids, times, NoiseConfig(estimate="replicates"))
    assert est.estimated.all() and np.all(np.isnan(est.score)) and not est.outlier.any()
    assert est.gain[0] == pytest.approx(GAIN, rel=0.4)


# --- use in the fit -------------------------------------------------------------------------

def measurements_with_replicates():
    """Series 0 has replicates; series 1 has none (single spectra)."""
    bins0, obs0, ids0, t0, _ = series_data(n_series=1, seed=10)
    shape = SHAPE
    bins1 = [BINS.copy() for _ in range(4)]
    obs1 = [curve([(m, 1.0)]) for m in (2.0, 6.0, 9.0, 12.0)]
    rng = np.random.default_rng(11)
    obs1 = [o + rng.normal(0, 1, len(o)) * np.sqrt(GAIN * np.maximum(o, 0)) for o in obs1]
    n = len(obs0) + 4
    return Measurements.from_arrays(
        bins0 + bins1, obs0 + obs1, [shape] * n, [16.8] * n, [None] * n,
        spectrum_id=[f"s{i}" for i in range(n)],
        data_id=np.concatenate([ids0, [1] * 4]), time=np.concatenate([t0, [0.0, 60.0, 300.0, 1e6]]),
    )


def test_fit_uses_the_estimate_where_there_are_replicates_and_the_fallback_elsewhere():
    meas = measurements_with_replicates()
    cfg = FitConfig(noise=NoiseConfig("gaussian", percent=1.0, estimate="replicates"),
                    backend=CPU, k_max=2, adam_steps=120, n_restarts=2)
    res = fit_models(meas, cfg)
    est = res.noise_estimate
    assert est is not None and est.estimated[:12].all() and not est.estimated[12:].any()
    # the loss stays Gaussian (docs/HISTORY.md item 57): an estimated series gets the sd its gain
    # gives at the spectrum's mean intensity, the others the configured level
    assert set(res.noise_kind) == {"gaussian"}
    np.testing.assert_allclose(res.obs_scale[:12], [np.sqrt(est.gain[i] * np.mean(o))
                                                    for i, o in enumerate(meas.observed[:12])])
    np.testing.assert_allclose(res.obs_scale[12:], [0.01 * o.max() for o in meas.observed[12:]])
    assert np.all(np.isfinite(res.grid("nll")))
    assert np.all(res.sample_size == len(BINS))  # the BIC's N is the number of bins
    with pytest.raises(ValueError, match="Gaussian noise level"):
        NoiseConfig("poisson", percent=None, estimate="replicates")


def test_without_the_option_nothing_changes():
    meas = measurements_with_replicates()
    cfg = FitConfig(noise=NoiseConfig("gaussian", percent=1.0), backend=CPU, k_max=2,
                    adam_steps=80, n_restarts=1)
    res = fit_models(meas, cfg)
    assert res.noise_estimate is None and set(res.noise_kind) == {"gaussian"}


def test_flagged_spectra_are_reported_by_the_fit(caplog):
    meas = measurements_with_replicates()
    obs = list(meas.observed)
    obs[2] = curve([(9.0, 1.0)]) * obs[2].sum() / curve([(9.0, 1.0)]).sum()
    bad = Measurements.from_arrays(meas.bins, obs, meas.shapes, meas.n_max, meas.references,
                                   meas.spectrum_id, meas.data_id, meas.time)
    cfg = FitConfig(noise=NoiseConfig("gaussian", percent=1.0, estimate="replicates"),
                    backend=CPU, k_max=1, adam_steps=60, n_restarts=1, polish_steps=0)
    with caplog.at_level(logging.WARNING, logger="hxex.fit.models"):
        res = fit_models(bad, cfg)
    assert res.noise_estimate.outlier[2] and "inconsistent with their replicates" in caplog.text


def test_mixed_models_in_one_batch_match_each_alone():
    meas, _ = synthetic([[(6.0, 1.0)], [(6.0, 1.0)]], seed=3)
    batch = PaddedBatch.from_arrays(meas.bins, [o / 100.0 for o in meas.observed], meas.n_max, CPU)
    w = shape_weights(meas.shapes, 1, CPU)
    zero = torch.zeros(2, 1, dtype=torch.float64)
    args = (batch, w, zero + 9.0, zero, zero)
    mixed = MixtureBatch(*args, ["gaussian", "poisson"])
    g = MixtureBatch(*args, "gaussian").nll(batch.obs)
    p = MixtureBatch(*args, "poisson").nll(batch.obs)
    nll = mixed.nll(batch.obs)
    assert nll[0].item() == pytest.approx(g[0].item())
    assert nll[1].item() == pytest.approx(p[1].item())
    with pytest.raises(ValueError, match="one per measurement"):
        MixtureBatch(*args, ["gaussian"])
    np.testing.assert_allclose(
        sample_size([np.ones(3), np.ones(3) * 2], ["gaussian", "poisson"]), [3.0, 6.0]
    )
