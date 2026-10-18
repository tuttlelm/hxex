# SPDX-License-Identifier: GPL-3.0-or-later
"""Fitting tests: synthetic mixtures with known answers, the K-range rules, independence from the
batch, hooks and failure handling, and statistical agreement with the prototype's results.

The prototype baseline (tests/golden/engine_baseline_cpu.npz) is one run of the prototype on the
542 measurements of the Bimodal test set. The prototype is not reproducible run to run (it never
seeds torch's random generator), so agreement is statistical, and the model differs from it in the
one bin just above n (see docs/PLAN.md), so losses are comparable but not equal. The full
comparison takes about a minute and runs when HXEX_FULL_REGRESSION=1; a quick one on a few series
always runs.
"""

import dataclasses
import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from hxex import chem
from hxex.fit import (
    FitConfig,
    FitError,
    Measurements,
    cosine_lr,
    fit_models,
    kmeans_start,
    kmeans_starts,
    split_start,
    stable_keys,
)
from hxex.fit.optimise import clip_per_measurement
from hxex.hooks import Cancelled
from hxex.model import Backend, MixtureBatch, PaddedBatch, shape_weights
from hxex.noise import NoiseConfig

CPU = Backend("cpu", "float64")
PEPTIDE = "RDKVQKEYALFYKLD"  # 15 residues
SHAPE = chem.natural_abundance(PEPTIDE)
N_SITES = chem.count_amides(PEPTIDE)
BINS = np.arange(N_SITES + len(SHAPE) + 3, dtype=float)
FAST = dict(adam_steps=250, n_restarts=3)


def synthetic(specs, seed=0, rel_noise=0.004, amplitude=1e5):
    """Measurements from specs, each a list of (mean, fraction); n = N_SITES, p = mean / n."""
    rng = np.random.default_rng(seed)
    observed, truth = [], []
    for spec in specs:
        means = np.array([m for m, _ in spec], float)
        fracs = np.array([f for _, f in spec], float)
        k = len(spec)
        batch = PaddedBatch.from_arrays([BINS], [np.zeros(len(BINS))], [1.2 * N_SITES], CPU)
        logit = lambda x: np.log(x / (1 - x))  # noqa: E731
        n_raw = np.full((1, k), np.log(N_SITES / (1.2 * N_SITES - N_SITES)))  # n = N_SITES
        model = MixtureBatch(
            batch, shape_weights([SHAPE], k, CPU),
            CPU.tensor(np.log(amplitude * fracs)[None]), CPU.tensor(n_raw),
            CPU.tensor(logit(means / N_SITES)[None]),
        )
        with torch.no_grad():
            pred = model.predict()[:, 0].numpy()
        observed.append(pred + rng.normal(0, rel_noise * pred.max(), len(pred)))
        truth.append((means, fracs))
    m = len(specs)
    meas = Measurements.from_arrays(
        [BINS] * m, observed, [SHAPE] * m, [1.2 * N_SITES] * m,
        references=[None] * m, spectrum_id=[f"syn{i}" for i in range(m)],
    )
    return meas, truth


UNIMODAL = [(6.0, 1.0)]
BIMODAL = [(3.0, 0.5), (11.0, 0.5)]
TRIMODAL = [(2.0, 0.3), (7.0, 0.4), (12.5, 0.3)]
CFG = FitConfig(noise=NoiseConfig("gaussian", percent=1.0), backend=CPU, **FAST)


@pytest.fixture(scope="module")
def fitted():
    meas, truth = synthetic([UNIMODAL, BIMODAL, TRIMODAL, BIMODAL, UNIMODAL])
    return meas, truth, fit_models(meas, CFG)


# --- configuration --------------------------------------------------------------------------

def test_config_defaults_roundtrip_and_k_values():
    cfg = FitConfig()
    assert cfg.k_values == (1, 2, 3) and cfg.grad_clip == "per_measurement"
    assert cfg.lbfgs_steps == 0 and cfg.criterion == "bic"
    assert FitConfig(k_min=2, k_max=3).k_values == (2, 3)
    assert FitConfig(k_min=2, k_max=2).k_values == (2,)
    assert [FitConfig(k1_adam_steps=200).adam_steps_for(k) for k in (1, 2)] == [200, 400]
    assert FitConfig().adam_steps_for(1) == 400
    full = FitConfig(k_min=2, seed=5, noise=NoiseConfig("poisson", n_eff=500.0, percent=None),
                     backend=Backend("cpu", "float32"))
    assert FitConfig.from_dict(json.loads(json.dumps(full.to_dict()))) == full
    with pytest.raises(ValueError, match="unknown fit setting"):
        FitConfig.from_dict({"k_maxx": 3})


@pytest.mark.parametrize(
    "kwargs",
    [{"k_min": 0}, {"k_min": 3, "k_max": 2}, {"criterion": "hqic"}, {"min_population": 1.0},
     {"min_population": -0.1}, {"n_restarts": 0}, {"adam_steps": -1}, {"lr_adam": 0.0},
     {"warm_noise": -1.0}, {"grad_clip": "none"}, {"seed": -1}, {"k_min": 2, "screen": True},
     {"k1_adam_steps": -1}],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        FitConfig(**kwargs)


# --- the building blocks --------------------------------------------------------------------

def test_cosine_lr_endpoints():
    assert cosine_lr(0, 100, 0.05, 0.002) == pytest.approx(0.05)
    assert cosine_lr(100, 100, 0.05, 0.002) == pytest.approx(0.002)
    assert cosine_lr(50, 100, 0.05, 0.002) == pytest.approx(0.026)
    assert cosine_lr(500, 100, 0.05, 0.002) == pytest.approx(0.002)  # clamped after the end


def test_stable_keys_are_deterministic_and_distinct():
    keys = stable_keys(["a", "b", "a"])
    assert keys[0] == keys[2] != keys[1] and keys.dtype == np.int64
    assert np.array_equal(stable_keys(["a", "b"]), stable_keys(["a", "b"]))


def test_kmeans_start_finds_the_populations():
    meas, _ = synthetic([BIMODAL])
    log_a, n_raw, logit_p = kmeans_start(
        meas.bins[0], meas.observed[0] / 100.0, 2, SHAPE, 1.2 * N_SITES, seed=0
    )
    n = 1.2 * N_SITES / (1 + np.exp(-n_raw))
    means = np.sort(n / (1 + np.exp(-logit_p)))
    assert means[0] < 6.0 < means[1] and np.all(np.isfinite(log_a))


def test_kmeans_start_pads_when_there_are_fewer_distinct_bins_than_components():
    out = kmeans_start(np.array([0.0, 1.0]), np.array([100.0, 50.0]), 4, np.array([1.0]), 10.0)
    assert all(len(x) == 4 and np.all(np.isfinite(x)) for x in out)


def test_kmeans_starts_of_a_batch_equal_those_of_each_measurement():
    meas, _ = synthetic([BIMODAL, UNIMODAL, TRIMODAL])
    obs = [o / 100.0 for o in meas.observed]
    batch = kmeans_starts(meas.bins, obs, 3, meas.shapes, meas.n_max, [0, 1, 2])
    for i in range(3):
        one = kmeans_start(meas.bins[i], obs[i], 3, meas.shapes[i], meas.n_max[i], seed=i)
        for a, b in zip(batch, one, strict=True):
            np.testing.assert_allclose(a[i], b, rtol=1e-12)


def test_split_start_keeps_the_amplitude_and_splits_the_heaviest():
    log_a = np.log(np.array([[10.0, 30.0], [5.0, 1.0]]))
    n_raw = np.zeros((2, 2))
    p = np.array([[0.2, 0.6], [0.5, 0.5]])
    start = split_start(log_a, n_raw, p, delta=0.5)
    assert start.log_A.shape == (2, 3)
    np.testing.assert_allclose(np.exp(start.log_A).sum(axis=1), [40.0, 6.0])
    np.testing.assert_allclose(np.exp(start.log_A[0]), [10.0, 15.0, 15.0])  # heaviest halved
    np.testing.assert_allclose(np.exp(start.log_A[1]), [2.5, 1.0, 2.5])
    lp = start.logit_p
    assert lp[0, 2] - lp[0, 1] == pytest.approx(1.0)  # the two halves are 2 * delta apart
    bad = split_start(np.array([[np.nan, 1.0]]), np.zeros((1, 2)), np.full((1, 2), 0.5))
    assert np.all(bad.log_A == 0) and np.all(bad.n_raw == 0) and np.all(bad.logit_p == 0)


def test_clip_per_measurement():
    p1 = torch.nn.Parameter(torch.zeros(3, 2))
    p2 = torch.nn.Parameter(torch.zeros(3, 2))
    p1.grad = torch.tensor([[30.0, 40.0], [0.3, 0.4], [0.0, 0.0]])
    p2.grad = torch.zeros(3, 2)
    clip_per_measurement([p1, p2], 5.0)
    norms = p1.grad.norm(dim=1)
    assert norms[0].item() == pytest.approx(5.0, rel=1e-4)
    np.testing.assert_allclose(p1.grad[1].numpy(), [0.3, 0.4])  # below the limit: unchanged
    assert torch.all(p1.grad[2] == 0)


# --- Measurements ---------------------------------------------------------------------------

def test_measurements_validation_and_defaults():
    m = Measurements.from_arrays([np.arange(4)], [np.ones(4)], [None], [5.0])
    assert len(m) == 1 and m.spectrum_id[0] == "m0" and m.references == (None,)
    np.testing.assert_array_equal(m.shapes[0], [1.0])
    with pytest.raises(ValueError, match="must match"):
        Measurements.from_arrays([np.arange(4)], [np.ones(3)], [None], [5.0])
    with pytest.raises(ValueError, match="n_max"):
        Measurements.from_arrays([np.arange(4)], [np.ones(4)], [None], [0.0])
    with pytest.raises(ValueError, match="one entry per measurement"):
        Measurements.from_arrays(
            [np.arange(4)], [np.ones(4)], [None], [5.0], spectrum_id=["a", "b"]
        )


def test_measurements_from_binned():
    from hxex.picking import PickerConfig, pick_spectra, plan_bins
    from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet

    cfg = PickerConfig(zero_filling=3, noise_threshold=100.0)
    entry = PeptideEntry("f", "S", PEPTIDE, "0001-0015", 1, 15, 2, modification="Hex:2")
    plan = plan_bins(PEPTIDE, 2, modification="Hex:2", config=cfg)
    spectra = []
    for i, t in enumerate((0.0, 60.0, 0.0)):
        mz = plan.pred_mz[:4]
        spectra.append(Spectrum(mz, np.array([1000.0, 800.0, 300.0, 100.0]), entry, t, i,
                                data_id=0 if i < 2 else 1, spectrum_id=f"p:s{i}"))
    meas = Measurements.from_binned(pick_spectra(SpectrumSet.from_spectra(spectra), cfg),
                                    n_max_factor=1.2)
    spec = chem.parse_modification("Hex:2")
    sites = chem.exchangeable_sites(PEPTIDE, mod=spec)
    assert meas.n_max[0] == pytest.approx(1.2 * sites) and sites == N_SITES + 2
    np.testing.assert_allclose(meas.shapes[0], chem.natural_abundance(PEPTIDE, mod=spec))
    assert meas.references == ((0,), (0,), (2,))  # the undeuterated control of each series
    kin = Measurements.from_binned(pick_spectra(SpectrumSet.from_spectra(spectra), cfg),
                                   kinetics=[False, True, False])
    assert kin.references == ((0,), (1,), (2,))  # a kinetics scan: from its own run
    with pytest.raises(ValueError, match="kinetics"):
        Measurements.from_binned(pick_spectra(SpectrumSet.from_spectra(spectra), cfg),
                                 kinetics=[True])
    assert list(meas.spectrum_id) == ["p:s0", "p:s1", "p:s2"]
    np.testing.assert_array_equal(meas.bins[0], np.arange(len(meas.bins[0])))
    with pytest.raises(ValueError):
        Measurements.from_binned(pick_spectra(SpectrumSet.from_spectra(spectra), cfg),
                                 n_max_factor=0.0)


# --- fitting synthetic mixtures -------------------------------------------------------------

def test_recovers_k_means_and_fractions(fitted):
    meas, truth, res = fitted
    assert list(res.selected_K) == [len(t[0]) for t in truth]
    for m, (means, fracs) in enumerate(truth):
        fit = res.fits[len(means)]
        order = np.argsort(fit.means[m])
        np.testing.assert_allclose(fit.means[m][order], means, atol=0.5)
        np.testing.assert_allclose(fit.populations[m][order], fracs, atol=0.06)


def test_amplitudes_are_in_the_data_units(fitted):
    _, _, res = fitted
    assert res.fits[1].A[0, 0] == pytest.approx(1e5, rel=0.05)
    assert res.selected("A")[1, :2].sum() == pytest.approx(1e5, rel=0.05)


def test_result_arrays_and_grids(fitted):
    meas, _, res = fitted
    assert len(res) == 5 and res.k_values == (1, 2, 3) and set(res.fits) == {1, 2, 3}
    assert res.grid("bic").shape == (5, 3) and res.grid("passes_min_pop").dtype == bool
    assert np.all(np.isfinite(res.grid("nll")))
    sel = res.selected("means")
    assert sel.shape == (5, 3) and np.isnan(sel[0, 1:]).all() and np.isfinite(sel[2]).all()
    assert list(res.spectrum_id) == list(meas.spectrum_id)
    np.testing.assert_allclose(res.sample_size, [len(BINS)] * 5)  # gaussian: the number of bins
    # more components never fit worse
    assert np.all(res.grid("nll")[:, 1] <= res.grid("nll")[:, 0] + 1e-6)


def test_table(fitted):
    _, _, res = fitted
    t = res.table()
    assert len(t["fit_K"]) == 5 * (1 + 2 + 3)
    assert all(len(v) == len(t["fit_K"]) for v in t.values())
    assert np.all(np.diff(t["measurement"]) >= 0)  # ordered by measurement, then K
    only = res.table(selected_only=True)
    assert len(only["fit_K"]) == int(res.selected_K.sum()) and only["is_selected"].all()
    rows = (only["measurement"] == 1)
    np.testing.assert_allclose(only["population"][rows].sum(), 1.0)
    assert sorted(only["k_ordered"][rows]) == [1, 2]
    means, rank = only["mean"][rows], only["k_ordered"][rows]
    assert means[rank == 1] < means[rank == 2]


def test_scaling_the_data_scales_only_the_amplitudes():
    meas, _ = synthetic([BIMODAL, UNIMODAL])
    big = Measurements.from_arrays(meas.bins, [o * 1000 for o in meas.observed], meas.shapes,
                                   meas.n_max, spectrum_id=meas.spectrum_id)
    cfg = FitConfig(k_max=2, noise=NoiseConfig("gaussian", percent=1.0),
                    backend=CPU, **FAST)
    refs = ((0,), (1,))  # each spectrum is its own noise reference: the scale follows the data
    a = fit_models(Measurements.from_arrays(meas.bins, meas.observed, meas.shapes, meas.n_max,
                                            references=refs, spectrum_id=meas.spectrum_id), cfg)
    b = fit_models(Measurements.from_arrays(big.bins, big.observed, big.shapes, big.n_max,
                                            references=refs, spectrum_id=big.spectrum_id), cfg)
    np.testing.assert_allclose(b.obs_scale, 1000 * a.obs_scale, rtol=1e-9)
    # the scaled problems are identical up to rounding, which the optimiser amplifies a little
    fa, fb = a.fits[2], b.fits[2]
    np.testing.assert_allclose(fb.A[0], 1000 * fa.A[0], rtol=1e-3)
    np.testing.assert_allclose(fb.means[0], fa.means[0], rtol=1e-3, atol=1e-3)
    # K = 2 on the unimodal spectrum: how its one population is split is not determined (the
    # rounding moved the split by 2 % on Windows), only the total and its mean are
    np.testing.assert_allclose(fb.A[1].sum(), 1000 * fa.A[1].sum(), rtol=1e-3)
    np.testing.assert_allclose(np.average(fb.means[1], weights=fb.A[1]),
                               np.average(fa.means[1], weights=fa.A[1]), rtol=1e-3)


# --- K range --------------------------------------------------------------------------------

def test_k_range_two_to_three():
    meas, _ = synthetic([BIMODAL, TRIMODAL])
    res = fit_models(meas, FitConfig(k_min=2, k_max=3, noise=CFG.noise, backend=CPU, **FAST))
    assert res.k_values == (2, 3) and set(res.fits) == {2, 3}
    assert list(res.selected_K) == [2, 3]
    assert res.grid("nll").shape == (2, 2) and res.selected("means").shape == (2, 3)


@pytest.mark.parametrize("k", [1, 2])
def test_a_single_k_has_no_selection(k):
    meas, _ = synthetic([UNIMODAL, BIMODAL])
    res = fit_models(meas, FitConfig(k_min=k, k_max=k, noise=CFG.noise, backend=CPU, **FAST))
    assert res.k_values == (k,) and list(res.selected_K) == [k, k]
    assert res.fits[k].A.shape == (2, k)
    assert res.score.shape == (2, 1) and np.all(np.isfinite(res.score))


def test_single_k_ignores_the_minimum_population_filter_but_reports_it():
    meas, _ = synthetic([UNIMODAL])  # a unimodal spectrum fitted with two populations
    res = fit_models(meas, FitConfig(k_min=2, k_max=2, min_population=0.45, noise=CFG.noise,
                                     backend=CPU, **FAST))
    assert res.selected_K[0] == 2 and np.isfinite(res.score[0, 0])
    assert not res.fits[2].passes_min_pop[0]  # the weaker population is below 45 %


def test_minimum_population_filter_rejects_a_k():
    meas, _ = synthetic([BIMODAL])
    res = fit_models(meas, FitConfig(min_population=0.7, noise=CFG.noise, backend=CPU, **FAST))
    assert not res.fits[2].passes_min_pop[0] and np.isinf(res.score[0, 1])
    assert res.selected_K[0] == 1  # K = 2 fits better but is rejected, K = 1 passes trivially
    off = fit_models(meas, FitConfig(min_population=0.0, noise=CFG.noise, backend=CPU, **FAST))
    assert off.selected_K[0] == 2
    everything_rejected = fit_models(
        meas, FitConfig(k_min=2, k_max=3, min_population=0.7, noise=CFG.noise, backend=CPU, **FAST)
    )
    assert everything_rejected.selected_K[0] == 2  # every K rejected: the smallest K is used


def test_aic_criterion_runs():
    meas, _ = synthetic([BIMODAL])
    res = fit_models(meas, FitConfig(criterion="aic", noise=CFG.noise, backend=CPU, **FAST))
    assert res.selected_K[0] == 2
    np.testing.assert_allclose(res.score[0, 1], res.fits[2].aic[0])


# --- reproducibility and independence -------------------------------------------------------

def test_same_seed_same_result_other_seed_different_start():
    meas, _ = synthetic([BIMODAL, TRIMODAL])
    a = fit_models(meas, CFG)
    b = fit_models(meas, CFG)
    for k in (1, 2, 3):
        np.testing.assert_array_equal(a.fits[k].nll, b.fits[k].nll)
        np.testing.assert_array_equal(a.fits[k].n, b.fits[k].n)
    c = fit_models(meas, FitConfig(seed=7, noise=CFG.noise, backend=CPU, **FAST))
    assert np.all(np.isfinite(c.grid("nll")))
    assert not np.array_equal(a.fits[3].nll, c.fits[3].nll)  # another seed: another path


def test_a_measurement_does_not_depend_on_the_others():
    specs = [BIMODAL, UNIMODAL, TRIMODAL, BIMODAL, UNIMODAL, TRIMODAL]
    meas, _ = synthetic(specs)
    cfg = FitConfig(noise=NoiseConfig("gaussian", sd=1500.0, percent=None), backend=CPU, **FAST)
    full = fit_models(meas, cfg)
    pick = [1, 3, 4]
    sub = fit_models(
        Measurements.from_arrays(
            [meas.bins[i] for i in pick], [meas.observed[i] for i in pick],
            [meas.shapes[i] for i in pick], meas.n_max[pick], spectrum_id=meas.spectrum_id[pick],
        ),
        cfg,
    )
    assert list(sub.selected_K) == list(full.selected_K[pick])
    for k in (1, 2, 3):
        np.testing.assert_allclose(sub.fits[k].nll, full.fits[k].nll[pick], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            sub.fits[k].means, full.fits[k].means[pick], rtol=1e-5, atol=1e-5
        )
    # changing the data of one measurement leaves the others untouched
    o2 = [o.copy() for o in meas.observed]
    o2[0] = o2[0] * 1.001
    other = fit_models(Measurements.from_arrays(
        meas.bins, o2, meas.shapes, meas.n_max, spectrum_id=meas.spectrum_id), cfg)
    for k in (1, 2, 3):
        np.testing.assert_allclose(other.fits[k].nll[1:], full.fits[k].nll[1:], rtol=1e-9)


def test_a_nested_k_is_not_worse_than_the_same_k_fitted_alone():
    meas, _ = synthetic([BIMODAL, UNIMODAL, TRIMODAL, BIMODAL, TRIMODAL])
    cfg = FitConfig(noise=NoiseConfig("gaussian", sd=1500.0, percent=None), backend=CPU,
                    k_min=1, k_max=3, **FAST)
    nested = fit_models(meas, cfg)
    for k in (2, 3):
        alone = fit_models(meas, dataclasses.replace(cfg, k_min=k, k_max=k))
        # the best restart is chosen before the final polish, hence a small tolerance
        assert np.all(nested.fits[k].nll <= alone.fits[k].nll + 0.1)


def test_float32_agrees_with_float64():
    meas, truth = synthetic([BIMODAL, UNIMODAL])
    r32 = fit_models(meas, FitConfig(noise=CFG.noise, backend=Backend("cpu", "float32"), **FAST))
    assert list(r32.selected_K) == [2, 1]
    for m, (means, _) in enumerate(truth):
        fit = r32.fits[len(means)]
        np.testing.assert_allclose(np.sort(fit.means[m]), means, atol=0.4)


def test_poisson_noise_model_runs():
    meas, _ = synthetic([BIMODAL], amplitude=2e3, rel_noise=0.0)
    cfg = FitConfig(noise=NoiseConfig("poisson", n_eff=2000.0, percent=None),
                    backend=CPU, k_max=2, **FAST)
    meas = Measurements.from_arrays(meas.bins, meas.observed, meas.shapes, meas.n_max,
                                    references=[(0,)], spectrum_id=meas.spectrum_id)
    res = fit_models(meas, cfg)
    assert res.selected_K[0] == 2
    assert res.sample_size[0] == pytest.approx(meas.observed[0].sum() / res.obs_scale[0])


# --- hooks and failures ---------------------------------------------------------------------

def test_progress_and_cancel():
    meas, _ = synthetic([UNIMODAL, BIMODAL])
    calls = []
    cfg = FitConfig(noise=CFG.noise, backend=CPU, k_max=2, adam_steps=30, n_restarts=2)
    fit_models(meas, cfg, progress=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]  # K=2 has the k-means start too
    calls.clear()
    fit_models(meas, dataclasses.replace(cfg, kmeans_after_warm=False),
               progress=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]
    flags = iter([False, False, True])
    with pytest.raises(Cancelled):
        fit_models(meas, cfg, cancel=lambda: next(flags, True))


def test_a_non_finite_measurement_does_not_stop_the_others(caplog):
    meas, _ = synthetic([BIMODAL, UNIMODAL, BIMODAL])
    obs = [o.copy() for o in meas.observed]
    obs[1][3] = np.nan
    bad = Measurements.from_arrays(meas.bins, obs, meas.shapes, meas.n_max,
                                   spectrum_id=meas.spectrum_id)
    cfg = FitConfig(noise=NoiseConfig("gaussian", sd=1500.0, percent=None), backend=CPU, **FAST)
    with caplog.at_level(logging.WARNING, logger="hxex.fit.optimise"):
        res = fit_models(bad, cfg)
    assert "non-finite" in caplog.text
    assert np.isinf(res.fits[1].nll[1]) and np.all(np.isfinite(res.fits[1].nll[[0, 2]]))
    assert res.selected_K[0] == 2 and res.selected_K[2] == 2
    assert np.all(np.isfinite(res.fits[2].n[1]))  # placeholder parameters are finite


def test_all_non_finite_raises():
    meas, _ = synthetic([UNIMODAL])
    obs = [meas.observed[0].copy()]
    obs[0][:] = np.nan
    bad = Measurements.from_arrays(meas.bins, obs, meas.shapes, meas.n_max)
    cfg = FitConfig(noise=NoiseConfig("gaussian", sd=1.0, percent=None), backend=CPU, k_max=1,
                    adam_steps=5, n_restarts=1)
    with pytest.raises(FitError, match="non-finite"):
        fit_models(bad, cfg)


def test_a_missing_reference_means_the_spectrum_itself():
    meas, _ = synthetic([UNIMODAL, BIMODAL])
    assert meas.references == (None, None)
    cfg = FitConfig(k_max=1, noise=NoiseConfig("gaussian", percent=2.0), backend=CPU, adam_steps=20,
                    n_restarts=1)
    res = fit_models(meas, cfg)
    np.testing.assert_allclose(res.obs_scale, [0.02 * o.max() for o in meas.observed])


# --- the prototype as a statistical oracle --------------------------------------------------

BASE = np.load(Path(__file__).parent / "golden" / "engine_baseline_cpu.npz")
INPUTS = np.load(Path(__file__).parent / "golden" / "engine_inputs.npz")


def baseline_measurements(rows):
    def unragged(flat, off, i):
        return flat[off[i] : off[i + 1]]

    bins = [unragged(INPUTS["bins"], INPUTS["bins_off"], i) for i in rows]
    obs = [unragged(INPUTS["obs"], INPUTS["bins_off"], i) for i in rows]
    shapes = [unragged(INPUTS["shape"], INPUTS["shape_off"], i) for i in rows]
    index = {int(r): j for j, r in enumerate(rows)}
    refs = []
    for i in rows:
        u = unragged(INPUTS["und"], INPUTS["und_off"], i)
        refs.append(tuple(index[int(x)] for x in u) if INPUTS["has_und"][i] else None)
    return Measurements.from_arrays(
        bins, obs, shapes, [1.5 * float(INPUTS["namides"][i]) for i in rows], refs,
        spectrum_id=[str(INPUTS["name"][i]) for i in rows], time=INPUTS["time"][rows],
    )


def true_n_populations():
    """Number of populations of every baseline measurement, from the known solutions."""
    text = (Path(__file__).parent / "golden" / "bimodal_pops.txt").read_text().split("\n")[1:]
    sol = {float(a): int(b) for a, b in (ln.split() for ln in text if ln)}
    return np.array([sol[t] for t in INPUTS["time"]])


def prototype_config():
    return FitConfig(noise=NoiseConfig("gaussian", percent=3.0), backend=CPU)


def test_baseline_inputs_are_the_542_measurements():
    assert len(INPUTS["name"]) == 542 and len(set(INPUTS["group"])) == 23
    assert BASE["best_K"].shape == (542,) and BASE["nll_grid"].shape == (542, 3)
    np.testing.assert_allclose(BASE["obs_scale"] > 0, True)


def test_noise_scale_matches_the_prototype_on_the_baseline_inputs():
    from hxex.noise import noise_scale

    meas = baseline_measurements(np.arange(542))
    scale = noise_scale(meas.observed, prototype_config().noise, meas.references)
    np.testing.assert_allclose(scale, BASE["obs_scale"], rtol=1e-12)
    from hxex.noise import sample_size

    np.testing.assert_allclose(sample_size(meas.observed, prototype_config().noise), BASE["n_eff"])


def test_a_few_series_agree_with_the_prototype():
    groups = list(dict.fromkeys(INPUTS["group"]))[:2]
    rows = np.flatnonzero(np.isin(INPUTS["group"], groups))
    res = fit_models(baseline_measurements(rows), prototype_config())
    agree = np.mean(res.selected_K == BASE["best_K"][rows])
    assert agree >= 0.9
    for k in (1, 2, 3):
        diff = (res.fits[k].nll - BASE["nll_grid"][rows, k - 1]) / np.maximum(
            np.abs(BASE["nll_grid"][rows, k - 1]), 1.0)
        assert np.median(diff) < 1e-3  # not worse (the polish makes most fits better)


@pytest.mark.skipif(os.environ.get("HXEX_FULL_REGRESSION") != "1",
                    reason="set HXEX_FULL_REGRESSION=1 (about a minute)")
def test_full_regression_against_the_prototype():
    meas = baseline_measurements(np.arange(542))
    res = fit_models(meas, prototype_config())
    agree = np.mean(res.selected_K == BASE["best_K"])
    # the prototype agrees with itself for 541 of 542; the polished fits differ mostly where the
    # prototype's fits stopped short
    assert agree >= 0.93, agree
    kt = true_n_populations()
    correct, base_correct = np.mean(res.selected_K == kt), np.mean(BASE["best_K"] == kt)
    assert correct >= base_correct - 0.02, (correct, base_correct)
    for k in (1, 2, 3):
        rel = (res.fits[k].nll - BASE["nll_grid"][:, k - 1]) / np.maximum(
            np.abs(BASE["nll_grid"][:, k - 1]), 1.0)
        assert np.median(rel) < 1e-3  # not worse than the prototype
        assert np.mean(rel < 1e-6) > 0.7  # as good or better for most fits


# --- the multimodality screen ---------------------------------------------------------------

@pytest.fixture(scope="module")
def screened():
    specs = [UNIMODAL, BIMODAL, UNIMODAL, TRIMODAL, [(9.0, 1.0)], UNIMODAL]
    meas, _ = synthetic(specs, seed=4)
    full = fit_models(meas, CFG)
    return meas, full, fit_models(meas, dataclasses.replace(CFG, screen=True))


def test_the_loss_floor_is_below_the_loss_of_every_k(screened):
    from hxex.fit.screen import loss_floor

    meas, full, _ = screened
    scaled = [o / s for o, s in zip(meas.observed, full.obs_scale, strict=True)]
    floor = loss_floor(meas.bins, scaled, meas.shapes, meas.n_max, ["gaussian"] * len(meas))
    assert np.all(np.isfinite(floor)) and np.all(floor >= 0)
    for k in full.k_values:
        assert np.all(floor <= full.fits[k].nll + 1e-9)
    # no floor for the Poisson loss or bins that are not whole numbers
    half = [b + 0.5 for b in meas.bins]
    assert np.all(loss_floor(half, scaled, meas.shapes, meas.n_max, ["gaussian"] * len(meas))
                  == -np.inf)
    assert np.all(loss_floor(meas.bins, scaled, meas.shapes, meas.n_max, ["poisson"] * len(meas))
                  == -np.inf)


def test_the_screen_is_idle_when_only_k1_is_tried(screened):
    """With K = 1 alone there is nothing to skip: no loss floor is computed."""
    from unittest import mock

    meas, _, _ = screened
    with mock.patch("hxex.fit.models.loss_floor", side_effect=AssertionError("not needed")):
        fit = fit_models(meas, dataclasses.replace(CFG, screen=True, k_min=1, k_max=1))
    assert fit.k_values == (1,) and len(fit) == len(meas)


def test_the_screen_keeps_the_selected_k_and_marks_what_it_skipped(screened):
    _, full, fit = screened
    assert np.array_equal(fit.selected_K, full.selected_K)
    assert list(full.selected_K) == [1, 2, 1, 3, 1, 1]
    for k in (1, 2, 3):  # what was fitted is the full fit
        done = fit.fitted(k)
        np.testing.assert_allclose(fit.fits[k].nll[done], full.fits[k].nll[done], rtol=1e-6)
    assert fit.fitted(1).all() and full.fitted(2).all()
    skipped = ~fit.fitted(2)
    assert skipped[[0, 2, 4, 5]].all() and not skipped[[1, 3]].any()
    assert np.all(np.isnan(fit.fits[2].A[skipped])) and np.all(np.isinf(fit.score[skipped, 1]))
    assert not fit.fitted(3)[1]  # the bimodal one cannot select K=3 once K=2 is fitted
    rows = fit.table()
    assert not np.any((rows["fit_K"] == 2) & np.isin(rows["measurement"], np.flatnonzero(skipped)))
    with pytest.raises(ValueError, match="screen"):
        fit.require_fitted(0, 2)
    fit.require_fitted(1, 2)
    merged = type(fit).merge([fit.select([0, 1]), fit.select([2, 3, 4, 5])], [[0, 1], [2, 3, 4, 5]])
    assert np.array_equal(merged.fitted(2), fit.fitted(2))
