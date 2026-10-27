# SPDX-License-Identifier: GPL-3.0-or-later
"""Quality flags: planted problems are flagged, normal spectra are not, and the edge cases."""

import dataclasses

import numpy as np
import pytest

from hxex.noise import NoiseEstimate
from hxex.quality import QualityConfig, assess_quality, reduced_deviance
from test_bundle import build_run


@pytest.fixture(scope="module")
def run():
    raw, binned, meas, fit, boot = build_run(n_boot=2)
    return meas, fit


def replace_observed(meas, i, observed):
    obs = list(meas.observed)
    obs[i] = observed
    return dataclasses.replace(meas, observed=tuple(obs))


def test_a_clean_run_is_not_flagged(run):
    meas, fit = run
    q = assess_quality(meas, fit)
    assert len(q) == len(fit) and not q.flagged.any()
    assert np.isfinite(q.deviance).all() and np.isfinite(q.deviance_rel).all()
    assert np.isfinite(q.misfit).all() and np.isfinite(q.misfit_rel).all()
    assert np.isfinite(q.intensity_rel).all()  # every time point of the fixture is replicated
    assert np.median(q.misfit_rel) == pytest.approx(1.0, abs=0.5)


def test_a_spectrum_that_the_fit_does_not_describe_is_flagged_poor_fit(run):
    meas, fit = run
    i = 3
    spoiled = np.array(meas.observed[i], dtype=float)
    spoiled[-1] += 0.5 * spoiled.sum()  # a large spurious peak the fit knows nothing about
    q = assess_quality(replace_observed(meas, i, spoiled), fit)
    assert q.poor_fit[i] and not np.delete(q.poor_fit, i).any()
    assert q.deviance[i] > 10 * np.median(q.deviance)


def test_a_fit_better_than_the_noise_model_is_never_poor(run):
    meas, fit = run
    i = 3
    spoiled = np.array(meas.observed[i], dtype=float)
    spoiled[-1] += 0.5 * spoiled.sum()
    bad = replace_observed(meas, i, spoiled)
    assert assess_quality(bad, fit).poor_fit[i]
    q = assess_quality(bad, fit, QualityConfig(min_deviance=1e9))
    assert q.misfit_rel[i] > 10 and not q.poor_fit.any()


def test_a_weak_spectrum_is_flagged_low_intensity_and_only_it(run):
    meas, fit = run
    i = 4  # one of three replicates at 60 s
    q = assess_quality(replace_observed(meas, i, np.asarray(meas.observed[i]) * 0.1), fit)
    assert q.low_intensity[i] and q.intensity_rel[i] == pytest.approx(0.1, abs=0.05)
    assert not np.delete(q.low_intensity, i).any()  # the others still see 2 normal replicates


def test_unreplicated_time_points_and_small_series_are_not_assessed(run):
    meas, fit = run
    # keep one of the two fully deuterated spectra
    keep = np.setdiff1d(np.arange(len(fit)), np.flatnonzero(fit.role == "fulldeut")[1:])
    sub_meas = dataclasses.replace(
        meas, **{f: tuple(getattr(meas, f)[i] for i in keep) if isinstance(getattr(meas, f), tuple)
                 else getattr(meas, f)[keep]
                 for f in ("bins", "observed", "shapes", "n_max", "references", "spectrum_id",
                           "data_id", "time", "role", "random_id")})
    sub_fit = dataclasses.replace(
        fit, spectrum_id=fit.spectrum_id[keep], data_id=fit.data_id[keep], time=fit.time[keep],
        role=fit.role[keep],
        selected_K=fit.selected_K[keep], score=fit.score[keep], obs_scale=fit.obs_scale[keep],
        sample_size=fit.sample_size[keep], n_max=fit.n_max[keep], noise_kind=fit.noise_kind[keep],
        fits={k: dataclasses.replace(
            kf, **{n: getattr(kf, n)[keep] for n in (
                "A", "n", "p", "log_A", "n_raw", "logit_p", "nll", "bic", "aic", "passes_min_pop")})
            for k, kf in fit.fits.items()},
        noise_estimate=None)
    q = assess_quality(sub_meas, sub_fit)
    assert np.isnan(q.intensity_rel[sub_fit.role == "fulldeut"]).all()
    assert not q.low_intensity[sub_fit.role == "fulldeut"].any()
    q = assess_quality(sub_meas, sub_fit, QualityConfig(min_series_fits=50))
    assert np.isnan(q.misfit_rel).all() and not q.poor_fit.any()


def test_noise_outliers_are_passed_through(run):
    meas, fit = run
    bare = dataclasses.replace(fit, noise_estimate=None)
    assert not assess_quality(meas, bare).noise_outlier.any()
    e = fit.noise_estimate
    flagged = np.zeros(len(fit), dtype=bool)
    flagged[2] = True
    est = NoiseEstimate(e.gain, e.estimated, e.reason, e.score, flagged, e.n_replicated)
    q = assess_quality(meas, dataclasses.replace(fit, noise_estimate=est))
    assert q.noise_outlier[2] and q.flagged[2] and q.flagged.sum() == 1


def test_gaussian_noise_uses_the_squared_residual(run):
    meas, fit = run
    gauss = dataclasses.replace(fit, noise_kind=np.array(["gaussian"] * len(fit)))
    d = reduced_deviance(meas, gauss)
    assert np.isfinite(d).all() and (d > 0).all()


def test_inputs_are_checked_and_the_config_round_trips(run):
    meas, fit = run
    other = dataclasses.replace(fit, spectrum_id=fit.spectrum_id[::-1].copy())
    with pytest.raises(ValueError, match="same spectra"):
        assess_quality(meas, other)
    cfg = QualityConfig(poor_fit_factor=5.0, min_series_fits=3, min_intensity=0.2)
    assert QualityConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown"):
        QualityConfig.from_dict({"bogus": 1})
    for bad in ({"poor_fit_factor": 1.0}, {"min_series_fits": 1}, {"min_intensity": 1.5}):
        with pytest.raises(ValueError):
            QualityConfig(**bad)


def test_a_series_whose_peaks_are_shifted_is_flagged_mz_shift(run):
    meas, fit = run
    # four series (the fixture has one)
    fit = dataclasses.replace(fit, data_id=np.arange(len(fit)) % 4)
    meas = dataclasses.replace(meas, data_id=fit.data_id)
    series = np.unique(fit.data_id)
    rng = np.random.default_rng(0)
    offsets = 6.0 + rng.normal(0.0, 2.0, len(fit))  # a calibration offset of +6 ppm
    moved = fit.data_id == series[0]
    offsets[moved] += 25.0  # e.g. a wrong C-terminus
    offsets[(fit.data_id == series[1]) & (np.arange(len(fit)) % 2 == 0)] = np.nan
    q = assess_quality(meas, fit, mz_offset_ppm=offsets)
    assert (q.mz_shift == moved).all() and (q.flagged >= q.mz_shift).all()
    assert q.mz_offset[moved] == pytest.approx(np.full(moved.sum(), 25.0), abs=3.0)
    assert np.isfinite(q.mz_offset).all() and np.abs(q.mz_offset[~moved]).max() < 3.0
    off = assess_quality(meas, fit, QualityConfig(max_mz_offset_ppm=None), mz_offset_ppm=offsets)
    assert not off.mz_shift.any() and np.isfinite(off.mz_offset).all()
    plain = assess_quality(meas, fit)  # without the picker's offsets: not assessed
    assert np.isnan(plain.mz_offset).all() and not plain.mz_shift.any()
    with pytest.raises(ValueError, match="one value per measurement"):
        assess_quality(meas, fit, mz_offset_ppm=offsets[:-1])
    with pytest.raises(ValueError, match="max_mz_offset_ppm"):
        QualityConfig(max_mz_offset_ppm=0)


def _scale_control_peaks(meas, rows, band, factor):
    """Scale the picked sticks of the controls ``rows`` whose natural-abundance shape puts them
    in ``band`` (relative to the tallest predicted peak) by ``factor``."""
    for i in rows:
        shape = np.asarray(meas.shapes[i])
        k = np.asarray(meas.bins[i]).astype(int)
        rel = np.where(k < len(shape), shape[np.minimum(k, len(shape) - 1)] / shape.max(), 0.0)
        y = np.array(meas.observed[i], dtype=float)
        sel = (rel >= band[0]) & (rel < band[1])
        y[sel] *= factor
        meas = replace_observed(meas, i, y)
    return meas


def test_compressed_tails_in_the_controls_flag_them_and_the_brighter_spectra(run):
    meas, fit = run
    controls = np.flatnonzero(np.asarray(meas.role) == "undeut")
    assert len(controls) == 3
    planted = _scale_control_peaks(meas, controls, (0.01, 0.1), 0.5)
    q = assess_quality(planted, fit)
    tops = np.array([np.max(o) for o in planted.observed])
    # the brighter half of the controls (at or above their median top) is judged: two of three
    judged = controls[tops[controls] >= np.median(tops[controls])]
    assert len(judged) == 2 and q.tails_low[judged].all() and q.tails_low.sum() == 2
    assert not q.saturated.any()
    # bright: the series' other spectra at least as bright as a flagged control, and only those
    others = np.flatnonzero(np.asarray(meas.role) != "undeut")  # labelled and fully deuterated
    assert np.array_equal(q.bright[others], tops[others] >= tops[judged].min())
    assert q.bright.any() and not q.bright[controls].any()
    j = int(others[q.bright[others]][0])
    scale = 0.5 * tops[judged].min() / tops[j]
    dimmer = replace_observed(planted, j, np.asarray(planted.observed[j]) * scale)
    assert not assess_quality(dimmer, fit).bright[j]
    # one replicate of three is not "most": nothing flagged
    one = _scale_control_peaks(meas, controls[:1], (0.01, 0.1), 0.5)
    assert not assess_quality(one, fit).tails_low.any()
    # switched off
    assert not assess_quality(planted, fit, QualityConfig(tails_low_ratio=None)).tails_low.any()


def test_a_low_tallest_peak_flags_the_controls_saturated(run):
    meas, fit = run
    controls = np.flatnonzero(np.asarray(meas.role) == "undeut")
    planted = _scale_control_peaks(meas, controls, (0.3, 1.0), 1.3)  # beside the tallest: high
    q = assess_quality(planted, fit)
    assert q.saturated[controls].sum() == 2 and not q.tails_low.any()  # the brighter two
    assert np.array_equal(q.flagged[controls], q.saturated[controls])
    assert not assess_quality(meas, fit).saturated.any()
    for bad in ({"saturated_ratio": 1.0}, {"tails_low_ratio": 1.0}, {"tails_low_ratio": 0}):
        with pytest.raises(ValueError):
            QualityConfig(**bad)


def test_flags_written_before_the_brightness_flags_read_as_false(run):
    from hxex.io.bundle import _quality_table, _read_quality

    meas, fit = run
    q = assess_quality(meas, fit)
    table = _quality_table(q)
    old = table.drop_columns(["saturated", "tails_low", "bright"])
    back = _read_quality(old, q.config.to_dict(), q.spectrum_id)
    assert not back.saturated.any() and not back.tails_low.any() and not back.bright.any()
    assert np.array_equal(back.flagged, q.flagged)
