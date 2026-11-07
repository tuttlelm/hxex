# SPDX-License-Identifier: GPL-3.0-or-later
"""Fitting through the instrument's per-scan cut (PLAN Next steps 24): reduced profile scans
delete each scan's weak peaks, so a dim envelope summed over its scans reads low at its weak
positions; passing the prediction through the same cut recovers the deuteration where the plain
fit is biased. The model, the K = 1 rule, the bootstrap, the drawn curves and the bundle's
columns."""

import dataclasses

import numpy as np
import pyarrow as pa
import pytest
import torch
from scipy.stats import binom

from hxex.bootstrap import BootstrapConfig, bootstrap
from hxex.fit import FitConfig, Measurements, ScanCut, fit_models
from hxex.io._arrow import _cut_columns, _cuts
from hxex.model import Backend
from hxex.model.mixture import CUT_HALF, CUT_STEEPNESS, cut_reading
from hxex.report.curves import component_curves, cut_curves

SHAPE = np.array([1.0, 0.45, 0.12, 0.02])
SITES, SCANS = 8, 60


def _kept(mu):
    return 1.0 / (1.0 + (CUT_HALF / np.maximum(mu, 1e-30)) ** CUT_STEEPNESS)


def _elution():
    w = np.exp(-0.5 * ((np.arange(SCANS) - SCANS / 2) / (SCANS / 6)) ** 2)
    return w / w.sum()


def _cut_set(rng, p, apex, reps):
    """``reps`` envelopes of deuteration ``p`` whose apex scan's tallest peak is ``apex`` times the
    noise (1), each scan's peaks deleted with the measured response."""
    w = _elution()
    env = np.convolve(binom.pmf(np.arange(SITES + 1), SITES, p), SHAPE)
    env = env / env.max()
    mu = apex / w.max() * env[:, None] * w[None, :]
    out = []
    for _ in range(reps):
        kept = rng.random(mu.shape) < _kept(mu)
        y = np.where(kept, mu * (1 + 0.05 * rng.standard_normal(mu.shape)), 0.0).sum(axis=1)
        out.append((np.clip(y, 0, None), ScanCut(w.copy(), np.ones((len(env), SCANS)))))
    return out


def test_the_cut_reading_keeps_bright_peaks_and_loses_weak_ones():
    pred = torch.tensor([[1e4], [10.0], [1.0]], dtype=torch.float64)  # (B, M)
    w = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    sigma = torch.ones(3, 1, 2, dtype=torch.float64)
    read = cut_reading(pred, w, torch.log(w).unsqueeze(0) - torch.log(sigma)).squeeze(1)
    expect = [x * _kept(x / 2) for x in (1e4, 10.0, 1.0)]
    np.testing.assert_allclose(read.numpy(), expect, rtol=1e-12)
    assert read[0] / 1e4 > 0.999 and read[2] < 0.15  # half lost near 1.9 x the noise per scan
    # the drawn curves read the same through the cut
    cut = ScanCut(np.array([0.5, 0.5]), np.ones((3, 2)))
    np.testing.assert_allclose(cut_curves(np.array([[1e4, 10.0, 1.0]]), cut)[0], expect,
                               rtol=1e-12)


def test_fitting_through_the_cut_recovers_dim_envelopes_at_k1_only():
    rng = np.random.default_rng(3)
    rows = [(p, apex, y, cut) for p in (0.1, 0.8) for apex in (5, 8)
            for y, cut in _cut_set(rng, p, apex, 4)]
    data = Measurements.from_arrays(
        [np.arange(len(r[2]), dtype=float) for r in rows], [r[2] for r in rows],
        [SHAPE] * len(rows), [1.2 * SITES] * len(rows))
    cut = dataclasses.replace(data, scan_cut=tuple(r[3] for r in rows))
    assert cut.cut.all() and not data.cut.any() and cut.select([0, 2]).cut.all()
    config = FitConfig(k_min=1, k_max=2, n_restarts=2)
    plain_fit, cut_fit = fit_models(data, config), fit_models(cut, config)
    p = np.array([r[0] for r in rows])
    low = p == 0.1

    def bias(fit):
        return fit.fits[1].n[:, 0] * fit.fits[1].p[:, 0] - SITES * p

    # the plain fit reads the lightly deuterated envelopes low (their heavy side is cut away)
    assert np.median(bias(plain_fit)[low]) < -0.1
    assert abs(np.median(bias(cut_fit)[low])) < 0.06
    assert abs(np.median(bias(cut_fit)[~low])) < 0.06
    # through the cut: K = 1 only
    assert np.all(cut_fit.selected_K == 1) and np.all(np.isnan(cut_fit.fits[2].nll))
    # the bootstrap resamples and refits through the cut; its band is the cut reading
    boot = bootstrap(cut, cut_fit, BootstrapConfig(n_boot=4, n_restarts=1))
    assert np.all(boot.n_success == 4)
    lo, hi = boot.curve_band(0)
    kf = cut_fit.fits[1]
    drawn = cut_curves(component_curves(cut.bins[0], kf.A[0], kf.n[0], kf.p[0], SHAPE),
                       cut.scan_cut[0])[0]
    tallest = int(np.argmax(drawn))
    assert lo[tallest] <= drawn[tallest] * 1.05 and hi[tallest] >= drawn[tallest] * 0.95


def test_the_cut_columns_round_trip():
    cuts = (ScanCut(np.array([0.25, 0.75]), np.arange(6.0).reshape(3, 2)), None)
    bins = [np.arange(3.0), np.arange(4.0)]
    back = _cuts(pa.table(_cut_columns(cuts)), bins)
    assert back[1] is None
    np.testing.assert_array_equal(back[0].weights, cuts[0].weights)
    np.testing.assert_array_equal(back[0].noise, cuts[0].noise)
    assert _cuts(pa.table({"x": [1, 2]}), bins) is None


def test_without_a_cut_the_batch_is_unchanged():
    from hxex.model import PaddedBatch

    backend = Backend("cpu")
    bins, obs = [np.arange(3.0), np.arange(2.0)], [np.ones(3), np.ones(2)]
    plain = PaddedBatch.from_arrays(bins, obs, [5.0, 5.0], backend)
    mixed = PaddedBatch.from_arrays(bins, obs, [5.0, 5.0], backend,
                                    cut=[None, ScanCut(np.array([1.0]), np.ones((2, 1))).scaled(1)])
    assert plain.scan_w is None and mixed.scan_w.shape == (2, 1)
    pred = torch.full((3, 2), 7.0, dtype=mixed.scan_c.dtype)
    read = cut_reading(pred, mixed.scan_w, mixed.scan_c)
    assert read[0, 0] == pytest.approx(7.0)  # the measurement without a cut reads as predicted


def test_spectra_below_the_scans_noise_are_flagged():
    """below_noise: a spectrum fitted through the cut whose tallest peak, in its largest scan,
    is below 4 times the scan's recorded noise (the default); signal_to_noise is that size
    (about the apex given to the simulation; 0 when nothing was read); not assessed without the
    scans' record; kept by the bundle."""
    from hxex.io.bundle import _quality_table, _read_quality
    from hxex.quality import QualityConfig, assess_quality, signal_to_noise

    rng = np.random.default_rng(5)
    rows = [(apex, y, cut) for apex in (0.8, 3.0, 30.0) for y, cut in _cut_set(rng, 0.3, apex, 6)
            if y.any()]  # a spectrum with nothing read is never fitted
    data = Measurements.from_arrays(
        [np.arange(len(r[1]), dtype=float) for r in rows], [r[1] for r in rows],
        [SHAPE] * len(rows), [1.2 * SITES] * len(rows))
    cut = dataclasses.replace(data, scan_cut=tuple(r[2] for r in rows))
    fit = fit_models(cut, FitConfig(k_min=1, k_max=1, n_restarts=1))
    q = assess_quality(cut, fit)
    apex = np.array([r[0] for r in rows])
    bright = apex == 30.0
    np.testing.assert_allclose(q.signal_to_noise[bright], 30.0, rtol=0.2)
    assert (apex == 0.8).sum() >= 2
    np.testing.assert_array_equal(q.below_noise, apex < 4)  # the default: 4 x the noise
    at_one = assess_quality(cut, fit, QualityConfig(min_signal_to_noise=1.0))
    np.testing.assert_array_equal(at_one.below_noise, apex < 1)
    empty = dataclasses.replace(cut, observed=(np.zeros_like(rows[0][1]),) + cut.observed[1:])
    assert signal_to_noise(empty)[0] == 0.0
    assert not assess_quality(cut, fit, QualityConfig(min_signal_to_noise=None)).below_noise.any()
    plain = assess_quality(data, fit_models(data, FitConfig(k_min=1, k_max=1, n_restarts=1)))
    assert np.isnan(plain.signal_to_noise).all() and not plain.below_noise.any()
    back = _read_quality(_quality_table(q), q.config.__dict__, q.spectrum_id)
    np.testing.assert_array_equal(back.below_noise, q.below_noise)
    np.testing.assert_allclose(back.signal_to_noise, q.signal_to_noise)
