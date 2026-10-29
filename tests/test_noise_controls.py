# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-bin noise from the undeuterated controls (``[run.fit.noise] estimate = "controls"``):
the curve z b1 I + (m I)^2, its use in the fit, the screen, the bootstrap and the population
test's null, and its record in the bundle."""

import dataclasses

import numpy as np
import pytest

from hxex import chem
from hxex.bootstrap import BootstrapConfig, bootstrap
from hxex.cli import _noise_curve_text
from hxex.fit import FitConfig, FitResult, Measurements, fit_models
from hxex.io import open_bundle
from hxex.model import Backend
from hxex.noise import (
    ControlsNoise,
    NoiseConfig,
    SampleNoise,
    bin_variances,
    controls_noise,
    distortion_onset,
)
from hxex.picking import PickerConfig, plan_bins
from hxex.population_test import simulate_null
from hxex.run import RunConfig, run
from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet
from test_picking import sticks

CPU = Backend("cpu", "float64")
SHAPE = np.array([0.45, 0.33, 0.15, 0.05, 0.02])  # a natural-abundance envelope


def controls(n_series=40, b1=8.0, a2=0.0, charges=(1, 2, 3), seed=0, sample=None,
             b1_of=None):
    """Undeuterated controls (three per series) whose noise variance is a2 + z b1 I (``b1_of``:
    a series' own b1; no floor by default, as the model has none)."""
    rng = np.random.default_rng(seed)
    bins, observed, shapes, role, charge, data_id, samples = [], [], [], [], [], [], []
    for s in range(n_series):
        z = charges[s % len(charges)]
        for _ in range(3):
            h = 10 ** rng.uniform(3, 5)
            mean = h * SHAPE / SHAPE.max()
            bins.append(np.arange(len(SHAPE), dtype=float))
            scale = b1 if b1_of is None else b1_of(s)
            observed.append(mean + rng.standard_normal(len(mean)) * np.sqrt(a2 + z * scale * mean))
            shapes.append(SHAPE)
            role.append("undeut")
            charge.append(z)
            data_id.append(s)
            samples.append("A" if sample is None else sample(s))
    return bins, observed, shapes, role, charge, data_id, samples


def test_the_counting_scale_is_recovered_with_charge():
    bins, observed, shapes, role, charge, data_id, _ = controls(n_series=120)
    curve = controls_noise(bins, observed, shapes, role, charge, data_id, model_error=0.09)
    assert curve is not None
    assert curve.b1 == pytest.approx(8.0, rel=0.15)  # about 5 % high on average
    assert curve.n_series == 120 and curve.n_controls == 360 and np.isinf(curve.onset)
    assert np.isfinite(curve.b1_se) and curve.b1_se < 0.25 * curve.b1
    # without the charges the scale would be off: the variance grows with z
    flat = controls_noise(bins, observed, shapes, role, np.ones(len(bins)), data_id,
                          model_error=0.09)
    assert flat.b1 > 1.5 * curve.b1


def test_too_few_series_gives_none():
    bins, observed, shapes, role, charge, data_id, _ = controls(n_series=4)
    assert controls_noise(bins, observed, shapes, role, charge, data_id, model_error=0.09) is None


def test_a_sample_with_another_counting_scale_is_named():
    def sample(s):
        return "B" if s >= 30 else "A"

    # sample B: eight times the counting noise of sample A
    data = controls(n_series=60, sample=sample, b1_of=lambda s: 64.0 if s >= 30 else 8.0)
    bins, observed, shapes, role, charge, data_id, samples = data
    curve = controls_noise(bins, observed, shapes, role, charge, data_id, model_error=0.09,
                           sample=samples)
    by = {s.sample: s for s in curve.by_sample}
    assert set(by) == {"A", "B"} and by["B"].b1 > 2 * by["A"].b1
    assert [s.sample for s in curve.disagreeing_samples()] == ["B"] or \
        {s.sample for s in curve.disagreeing_samples()} == {"A", "B"}


def test_bright_distorted_controls_are_left_out_above_the_onset():
    bins, observed, shapes, role, charge, data_id, _ = controls(n_series=30)
    rng = np.random.default_rng(2)
    for _ in range(15):  # bright controls whose tallest peak reads low (compression)
        mean = 3e6 * SHAPE / SHAPE.max()
        mean[0] *= 0.7
        bins.append(np.arange(len(SHAPE), dtype=float))
        observed.append(mean + rng.standard_normal(len(mean)) * np.sqrt(8 * mean))
        shapes.append(SHAPE)
        role.append("undeut")
        charge.append(2)
        data_id.append(100)
    assert distortion_onset(bins, observed, shapes, role) == 2e6
    curve = controls_noise(bins, observed, shapes, role, charge, data_id, model_error=0.09)
    assert curve.onset == 2e6 and curve.n_controls == 90



def test_dim_controls_that_read_flattened_are_no_onset():
    # picked noise lifts a dim control's small heavy-side peaks: the dimmest band departs, but
    # distortion cannot start at no brightness, so only a departure above a matching band counts
    def band(top, heavy=1.0, n=12):
        mean = top * SHAPE / SHAPE.max()
        mean[2:] *= heavy
        return [[np.arange(len(SHAPE), dtype=float)] * n, [mean.copy() for _ in range(n)],
                [SHAPE] * n, ["undeut"] * n]

    def join(*parts):
        return [sum((list(p[j]) for p in parts), []) for j in range(4)]

    assert np.isinf(distortion_onset(*join(band(1e4, heavy=1.5), band(2e5))))
    bright = band(3e6)
    bright[1] = [m * np.array([0.7, 1, 1, 1, 1]) for m in bright[1]]  # compression above
    assert distortion_onset(*join(band(1e4, heavy=1.5), band(2e5), bright)) == 2e6
    assert np.isinf(distortion_onset(*join(band(1e4, heavy=1.5), bright)))  # nothing matched

def test_bin_variances_follow_the_curve_with_a_floor():
    curve = ControlsNoise(b1=5.0, model_error=0.1, onset=np.inf, n_controls=10, n_series=5,
                          b1_se=0.5)
    y = np.array([0.0, 0.0, 1000.0, 1000.0, 1000.0, 0.0, 0.0])
    (var,) = bin_variances([y], [2.0], curve)
    at = np.convolve(y, np.ones(3) / 3, mode="same")
    expect = np.maximum(2 * 5 * at + (0.1 * at) ** 2, (0.005 * 1000) ** 2)
    np.testing.assert_allclose(var, expect)
    assert var[0] == pytest.approx(25.0)  # an empty bin keeps the floor, 0.5 % of the top
    assert ControlsNoise.from_dict(curve.to_dict()) == curve


def test_noise_config_checks_the_new_settings():
    assert NoiseConfig(estimate="controls").model_error == 9.0
    with pytest.raises(ValueError, match="gaussian"):
        NoiseConfig("poisson", percent=3.0, estimate="controls")
    with pytest.raises(ValueError, match="model_error"):
        NoiseConfig(estimate="controls", model_error=-1.0)
    with pytest.raises(ValueError, match="estimate"):
        NoiseConfig(estimate="pairs")
    cfg = NoiseConfig(estimate="controls", model_error=7.0)
    assert NoiseConfig.from_dict(cfg.to_dict()) == cfg


def spectra(n=24, seed=0):
    """Labelled spectra (one or two populations) plus undeuterated controls, as Measurements."""
    rng = np.random.default_rng(seed)
    shape = chem.isotope_clusters("RDKVQKEYAL")[0]
    bins, observed, shapes, role, data_id = [], [], [], [], []
    for s in range(n):
        x = np.arange(16, dtype=float)
        for r in ("undeut", ""):
            if r == "undeut":
                mean = 2e4 * np.convolve(np.r_[1.0, np.zeros(15)], shape)[:16]
            else:
                one = np.exp(-0.5 * ((x - 5) / 1.3) ** 2)
                two = np.exp(-0.5 * ((x - 11) / 1.3) ** 2) if s % 2 else 0 * x
                mean = 2e4 * np.convolve(one + 0.8 * two, shape)[:16]
            observed.append(np.clip(mean + rng.standard_normal(16) * np.sqrt(9 * mean + 100),
                                    0, None))
            bins.append(x)
            shapes.append(shape)
            role.append(r)
            data_id.append(s)
    return Measurements.from_arrays(bins, observed, shapes, np.full(len(bins), 12.0),
                                    data_id=data_id, role=role, charge=np.full(len(bins), 2.0))


def fit_config(**noise):
    return FitConfig(k_max=2, adam_steps=80, n_restarts=2, backend=CPU,
                     noise=NoiseConfig(**noise))


def test_a_curve_with_the_constant_sd_reproduces_that_fit():
    data = spectra()
    sd = 500.0
    variances = [np.full(len(o), sd ** 2) for o in data.observed]
    plain = fit_models(data, fit_config(sd=sd, percent=None))
    per_bin = fit_models(data, fit_config(estimate="controls"), bin_variance=variances)
    np.testing.assert_array_equal(plain.selected_K, per_bin.selected_K)
    for k in (1, 2):
        np.testing.assert_allclose(per_bin.fits[k].nll, plain.fits[k].nll, rtol=1e-6)
    assert plain.noise_curve is None


def test_the_screen_skips_nothing_it_should_not_with_per_bin_noise():
    data = spectra()
    full = fit_models(data, fit_config(estimate="controls"))
    screened = fit_models(data, dataclasses.replace(fit_config(estimate="controls"),
                                                    screen=True))
    assert full.noise_curve is not None
    np.testing.assert_array_equal(full.selected_K, screened.selected_K)


def test_the_curve_follows_the_fit_through_merge_select_bootstrap_and_null():
    data = spectra()
    fit = fit_models(data, fit_config(estimate="controls"))
    rows = [np.arange(0, len(data), 2), np.arange(1, len(data), 2)]
    merged = FitResult.merge([fit.select(r) for r in rows], rows)
    assert merged.noise_curve == fit.noise_curve
    boot = bootstrap(data, fit, BootstrapConfig(n_boot=3, n_restarts=1, adam_steps=30,
                                                polish_steps=0))
    labelled = data.role == ""
    assert np.isfinite(boot.mean_se[labelled, 0]).all()
    stats = simulate_null(data, fit, np.flatnonzero(labelled)[:4], 1, 2)
    assert np.isfinite(stats).all()


PICK = PickerConfig(zero_filling=3, noise_threshold=100.0)
PEPTIDE = "RDKVQKEYAL"


def raw_with_counting_noise(n_series=6):
    plan = plan_bins(PEPTIDE, 2, config=PICK)
    shape = chem.isotope_clusters(PEPTIDE)[0]
    shape = shape / shape.max()
    spectra_, rng = [], np.random.default_rng(0)
    for s in range(n_series):
        entry = PeptideEntry(f"f{s}.xlsx", "S", PEPTIDE, "0001-0010", 1, 10, 2)
        for k, (t, rep) in enumerate([(0.0, 1), (0.0, 2), (0.0, 3), (60.0, 1), (60.0, 2)]):
            height = 10 ** rng.uniform(3.5, 4.5)
            if t == 0.0:
                peaks = {i: height * shape[i] for i in range(min(len(shape), len(plan.pred_mz)))
                         if height * shape[i] > 20}
            else:
                peaks = {i: height * np.exp(-0.5 * ((i - 3.0) / 1.2) ** 2)
                         for i in range(len(plan.pred_mz))}
            mz, y = sticks(plan, peaks)
            y = np.clip(y + rng.standard_normal(len(y)) * np.sqrt(4 * y + 25), 0, None)
            spectra_.append(Spectrum(mz, y, entry, np.nan if t == 0.0 else t, rep, data_id=s,
                                     label=f"s{s}c{k}", spectrum_id=f"p:f{s}.xlsx#{k + 1}",
                                     role="undeut" if t == 0.0 else ""))
    return SpectrumSet.from_spectra(spectra_)


def test_a_run_measures_the_curve_once_and_the_bundle_keeps_it(tmp_path):
    raw = raw_with_counting_noise()
    fit = FitConfig(k_max=2, adam_steps=60, n_restarts=2, backend=CPU,
                    noise=NoiseConfig("gaussian", percent=3.0, estimate="controls"))
    cfg = RunConfig(picker=PICK, fit=fit, bootstrap=BootstrapConfig(
        n_boot=2, n_restarts=1, adam_steps=30, polish_steps=0), controls=None)
    res = run(raw, cfg, out=tmp_path / "b")
    curve = res.fit.noise_curve
    assert curve is not None and curve.b1 > 0 and curve.n_series == 6
    assert [s.sample for s in curve.by_sample] == ["S"]
    # fitting in parts uses the one curve of the whole bundle
    parts = run(raw, dataclasses.replace(cfg, max_chunk=1, bootstrap=None))
    assert parts.fit.noise_curve == curve
    np.testing.assert_array_equal(parts.fit.selected_K, res.fit.selected_K)
    with open_bundle(tmp_path / "b") as b:
        assert b.fit().noise_curve == curve
        text = _noise_curve_text(b.fit())
        assert text.startswith("per bin, z x b1 x I + (9 % of I)^2") and "S " in text


def test_info_describes_the_constant_noise_and_unmeasured_samples():
    data = spectra(n=4)
    fit = fit_models(data, fit_config(percent=3.0))
    assert _noise_curve_text(fit) == "one sd per spectrum, 3 % of its reference's tallest peak"
    few = fit_models(data, fit_config(estimate="controls"))  # four series: too few
    assert few.noise_curve is None
    assert "too few undeuterated controls" in _noise_curve_text(few)
    curve = ControlsNoise(b1=8.0, model_error=0.09, onset=np.inf, n_controls=30, n_series=10,
                          b1_se=0.5, by_sample=(SampleNoise("A", 8.2, 0.7, 6),
                                                SampleNoise("B", np.nan, np.nan, 4)))
    text = _noise_curve_text(dataclasses.replace(fit, noise_curve=curve))
    assert text.endswith("b1 by sample: A 8.2 +- 0.7, B not measured (4 series)")
    assert curve.disagreeing_samples() == []
