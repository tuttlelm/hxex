# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU and GPU agree. These tests need CUDA and are skipped without it (CI has no GPU); the
numbers they check were measured on an RTX A4500 and are recorded in docs/HISTORY.md.

The float32 GPU fit and the float64 CPU fit take different numerical paths through a non-convex
optimisation, so the agreement is checked statistically for the full set (selected K, loss, the
share of spectra whose means move) and strictly for clear synthetic data. A fit depends only on
its own data and seed (see docs/HISTORY.md), so the GPU does not change which spectra are
affected by which.
"""

import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from hxex.bootstrap import BootstrapConfig, bootstrap
from hxex.fit import FitConfig, fit_models
from hxex.io import open_bundle, write_bundle
from hxex.model import Backend
from hxex.noise import NoiseConfig
from test_fit import BIMODAL, TRIMODAL, UNIMODAL, synthetic  # noqa: I001
from test_noise_estimate import measurements_with_replicates  # noqa: I001

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
CPU64 = Backend("cpu", "float64")
GPU32, GPU64 = Backend("cuda", "float32"), Backend("cuda", "float64")
NOISE = NoiseConfig("gaussian", sd=45.0, percent=None)
LIGHT = dict(adam_steps=200, n_restarts=2)


def clear_data():
    return synthetic([UNIMODAL, BIMODAL, TRIMODAL, BIMODAL, UNIMODAL, TRIMODAL], seed=2)[0]


def selected_loss(res):
    """The loss of every spectrum at its selected K."""
    grid = res.grid("nll")
    return grid[np.arange(len(res)), [res.k_values.index(int(k)) for k in res.selected_K]]


def ordered(res):
    """Means and fractions at the selected K, components ordered by mean (NaN-padded)."""
    means, fracs = res.selected("means"), res.selected("fractions")
    order = np.argsort(np.where(np.isnan(means), np.inf, means), axis=1)
    return np.take_along_axis(means, order, axis=1), np.take_along_axis(fracs, order, axis=1)


@needs_cuda
def test_float32_on_the_gpu_matches_float64_on_the_cpu_for_clear_data():
    meas = clear_data()
    cpu = fit_models(meas, FitConfig(noise=NOISE, backend=CPU64, **LIGHT))
    gpu = fit_models(meas, FitConfig(noise=NOISE, backend=GPU32, **LIGHT))
    assert list(cpu.selected_K) == list(gpu.selected_K)
    (mc, fc), (mg, fg) = ordered(cpu), ordered(gpu)
    np.testing.assert_allclose(mg, mc, atol=0.05)
    np.testing.assert_allclose(fg, fc, atol=0.02)
    np.testing.assert_allclose(selected_loss(gpu), selected_loss(cpu), rtol=5e-3, atol=0.2)
    assert gpu.config.backend == GPU32 and gpu.fits[1].A.dtype == np.float64  # results are float64


@needs_cuda
def test_float64_on_the_gpu_is_the_cpu_result():
    meas = clear_data()
    cpu = fit_models(meas, FitConfig(noise=NOISE, backend=CPU64, **LIGHT))
    gpu = fit_models(meas, FitConfig(noise=NOISE, backend=GPU64, **LIGHT))
    assert list(cpu.selected_K) == list(gpu.selected_K)
    np.testing.assert_allclose(ordered(gpu)[0], ordered(cpu)[0], atol=1e-3)
    np.testing.assert_allclose(selected_loss(gpu), selected_loss(cpu), rtol=1e-4, atol=1e-3)


@needs_cuda
def test_a_gpu_run_is_reproducible():
    meas = clear_data()
    cfg = FitConfig(noise=NOISE, backend=GPU32, **LIGHT)
    a, b = fit_models(meas, cfg), fit_models(meas, cfg)
    for k in a.k_values:
        np.testing.assert_array_equal(a.fits[k].nll, b.fits[k].nll)
        np.testing.assert_array_equal(a.fits[k].n_raw, b.fits[k].n_raw)


@needs_cuda
def test_a_measurement_does_not_depend_on_the_others_on_the_gpu():
    meas = clear_data()
    cfg = FitConfig(noise=NOISE, backend=GPU32, **LIGHT)
    full = fit_models(meas, cfg)
    pick = [1, 3, 5]
    sub_meas = type(meas).from_arrays(
        [meas.bins[i] for i in pick], [meas.observed[i] for i in pick],
        [meas.shapes[i] for i in pick], meas.n_max[pick], spectrum_id=meas.spectrum_id[pick])
    sub = fit_models(sub_meas, cfg)
    assert list(sub.selected_K) == list(full.selected_K[pick])
    np.testing.assert_allclose(ordered(sub)[0], ordered(full)[0][pick], atol=0.02)


@needs_cuda
def test_mixed_noise_models_the_estimate_and_the_polish_run_on_the_gpu():
    meas = measurements_with_replicates()
    kw = dict(noise=NoiseConfig("gaussian", percent=1.0, estimate="replicates"), k_max=2,
              adam_steps=120, n_restarts=2)
    cpu = fit_models(meas, FitConfig(backend=CPU64, **kw))
    gpu = fit_models(meas, FitConfig(backend=GPU32, **kw))
    assert list(gpu.noise_kind) == list(cpu.noise_kind) and set(gpu.noise_kind) == {
        "poisson", "gaussian"}
    np.testing.assert_allclose(gpu.obs_scale, cpu.obs_scale)  # the estimate does not use the GPU
    assert np.all(np.isfinite(gpu.grid("nll"))) and list(gpu.selected_K) == list(cpu.selected_K)


@needs_cuda
def test_the_bootstrap_agrees_between_cpu_and_gpu():
    meas = clear_data()
    cpu = fit_models(meas, FitConfig(noise=NOISE, backend=CPU64, **LIGHT))
    gpu = fit_models(meas, FitConfig(noise=NOISE, backend=GPU32, **LIGHT))
    cfg = BootstrapConfig(n_boot=24, n_restarts=1, adam_steps=80, polish_steps=20)
    bc, bg = bootstrap(meas, cpu, cfg), bootstrap(meas, gpu, cfg)
    assert list(bc.n_success) == list(bg.n_success) == [24] * len(meas)
    ratio = np.nanmedian(bg.mean_se / bc.mean_se)  # the same noise, resampled differently
    assert 0.7 < ratio < 1.4


@needs_cuda
def test_an_unavailable_device_is_reported_not_guessed():
    assert Backend("cuda:0").torch_device().type == "cuda"
    with pytest.raises((RuntimeError, AssertionError)):
        torch.zeros(1, device="cuda:99")


def test_a_bundle_of_a_gpu_run_opens_without_cuda(tmp_path):
    """The manifest only records the device; loading never needs it."""
    meas = clear_data()
    cpu = fit_models(meas, FitConfig(noise=NOISE, backend=CPU64, k_max=2, **LIGHT))
    as_gpu = replace(cpu, config=replace(cpu.config, backend=GPU32))
    path = write_bundle(tmp_path / "b", measurements=meas, fit=as_gpu)
    with open_bundle(path) as b:
        assert b.fit().config.backend == GPU32
        np.testing.assert_array_equal(b.fit().fits[2].nll, cpu.fits[2].nll)


@needs_cuda
@pytest.mark.skipif(os.environ.get("HXEX_FULL_REGRESSION") != "1",
                    reason="set HXEX_FULL_REGRESSION=1 (about two minutes)")
def test_the_full_set_agrees_between_cpu_and_gpu():
    from test_fit import baseline_measurements, prototype_config

    meas = baseline_measurements(np.arange(542))
    noise = prototype_config().noise
    cpu = fit_models(meas, FitConfig(noise=noise, backend=CPU64))
    gpu = fit_models(meas, FitConfig(noise=noise, backend=GPU32))
    assert np.mean(cpu.selected_K == gpu.selected_K) >= 0.99  # 542 of 542 when measured
    for k, share in ((1, 0.95), (2, 0.9), (3, 0.85)):
        rel = np.abs(gpu.fits[k].nll - cpu.fits[k].nll) / np.maximum(np.abs(cpu.fits[k].nll), 1.0)
        assert np.mean(rel < 1e-3) >= share  # 100, 98 and 91 % when measured
        assert np.median(rel) < 1e-5
    same = cpu.selected_K == gpu.selected_K
    (mc, fc), (mg, fg) = ordered(cpu), ordered(gpu)
    moved = np.nanmax(np.abs(mc - mg), axis=1)[same]
    assert np.mean(moved > 0.5) <= 0.01 and np.median(moved) < 0.01  # 0.2 % and 0.0 D measured
    assert np.nanmax(np.abs(fc - fg)[same]) < 0.2
