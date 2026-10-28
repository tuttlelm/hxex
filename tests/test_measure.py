# SPDX-License-Identifier: GPL-3.0-or-later
"""The bins' value: heights or areas, chosen per run from the width effect (picking.measure)."""

import math

import numpy as np
import pytest

from hxex.io import open_bundle
from hxex.picking import PeakMeasure, PickerConfig, choose_measure, pick_spectra, plan_bins
from hxex.picking.measure import HEIGHTS, MIN_CONTROLS, area_values, peak_fwhm
from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet

PEPTIDE = "RDKVQKEYAL"
CFG = PickerConfig()
FWHM_PPM = 47.0  # a Q-TOF's peak width
SPACING_PPM = 18.0  # its profile point spacing
SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))  # sigma of a Gaussian of unit FWHM


def gaussians(centres, heights, fwhm_ppm, lo, hi, spacing_ppm=SPACING_PPM):
    """A profile from ``lo`` to ``hi`` (points every ``spacing_ppm``) holding Gaussian peaks; each
    peak's FWHM in ppm from ``fwhm_ppm`` (a number or one per peak)."""
    mz = lo * (1 + spacing_ppm * 1e-6) ** np.arange(int(math.log(hi / lo) / (spacing_ppm * 1e-6)))
    y = np.zeros_like(mz)
    widths = np.broadcast_to(np.asarray(fwhm_ppm, dtype=float), np.shape(centres))
    for c, h, w in zip(centres, heights, widths, strict=True):
        y += h * np.exp(-0.5 * ((mz - c) / (SIGMA * w * 1e-6 * c)) ** 2)
    return mz, y


def control(plan, top, widening):
    """An undeuterated control: natural abundance in its peaks' areas, the tallest peak
    ``FWHM_PPM`` wide and a peak of share s of the tallest ``1 + widening (1 - s)`` times as
    wide (so its height is that much lower)."""
    shape = plan.clusters[0][: len(plan.pred_mz)]
    share = shape / shape.max()
    keep = share > 0.002
    centres, share = plan.pred_mz[: len(shape)][keep], share[keep]
    widen = 1 + widening * (1 - share)
    return gaussians(centres, top * share / widen, FWHM_PPM * widen,
                     centres.min() - 2.0, centres.max() + 2.0)


def control_set(n, widening, seed=0, labelled=0):
    """``n`` undeuterated controls of one series (and ``labelled`` copies of them at 60 s)."""
    rng = np.random.default_rng(seed)
    plan = plan_bins(PEPTIDE, 2, config=CFG)
    entry = PeptideEntry("f", "S", PEPTIDE, "0001-0010", 1, 10, 2)
    out = []
    for k in range(n + labelled):
        mz, y = control(plan, float(10 ** rng.uniform(4, 6)), widening)
        un = k < n
        out.append(Spectrum(mz, y, entry, float("nan") if un else 60.0, k + 1, data_id=0,
                            spectrum_id=f"p:{'un' if un else 't'}{k}",
                            role="undeut" if un else ""))
    return SpectrumSet.from_spectra(out)


def test_a_peak_of_the_runs_width_keeps_its_height_and_a_wider_one_counts_for_more():
    m = PeakMeasure("area", "area", fwhm_ppm=FWHM_PPM, area_fwhm=1.5)
    c = np.array([1000.0, 1001.0])
    mz, y = gaussians(c, [500.0, 500.0], [FWHM_PPM, 2 * FWHM_PPM], 998.0, 1003.0)
    v = area_values(mz, y, c, np.array([500.0, 500.0]), 1, m)
    assert v[0] == pytest.approx(500.0, rel=0.01)
    # twice as wide: twice the area over the same range, less what falls outside +-1.5 FWHM
    assert v[1] == pytest.approx(500.0 * 2 * math.erf(2 * math.sqrt(math.log(2)) * 0.75)
                                 / math.erf(2 * math.sqrt(math.log(2)) * 1.5), rel=0.01)
    # linear interpolation at 2.6 points per width reads it a few per cent wide
    assert peak_fwhm(mz, y, c[0], np.median(np.diff(mz))) == pytest.approx(FWHM_PPM, rel=0.05)


def test_areas_leave_empty_bins_centroided_spectra_and_gaps_alone():
    m = PeakMeasure("area", "area", fwhm_ppm=FWHM_PPM)
    c = np.array([1000.0, 1001.0])
    sticks = (np.array([1000.0, 1001.0]), np.array([500.0, 300.0]))
    np.testing.assert_array_equal(area_values(*sticks, c, np.array([500.0, 300.0]), 1, m),
                                  [500.0, 300.0])  # centroided: heights
    mz, y = gaussians(c[:1], [500.0], FWHM_PPM, 998.0, 1003.0)
    v = area_values(mz, y, np.array([1000.0, np.nan]), np.array([500.0, 0.0]), 1, m)
    assert v[1] == 0.0 and v[0] == pytest.approx(500.0, rel=0.01)
    # a gap in the profile under the peak counts as nothing, not as a straight line across it
    cut = (mz < 1000.0 - 40e-6 * 1000) | (mz > 1000.0 + 40e-6 * 1000) | (mz < 1000.0)
    gap = area_values(mz[cut], y[cut], c[:1], np.array([500.0]), 1, m)
    assert gap[0] < 0.75 * 500.0


def test_the_area_range_stays_inside_half_the_isotope_spacing():
    # 1.5 widths of 400 ppm would reach the next isotope peak of a 2+ ion (0.5 m/z away)
    m = PeakMeasure("area", "area", fwhm_ppm=400.0, area_fwhm=1.5)
    c = np.array([1000.0])
    mz, y = gaussians(c, [100.0], 400.0, 995.0, 1005.0, spacing_ppm=40.0)
    alone = area_values(mz, y, c, np.array([100.0]), 2, m)
    assert alone[0] == pytest.approx(100.0, rel=0.02)  # the capped range is normalised as such
    near = gaussians([1000.5], [1e4], 47.0, 995.0, 1005.0, spacing_ppm=40.0)[1]
    assert area_values(mz, y + near, c, np.array([100.0]), 2, m)[0] == pytest.approx(
        alone[0], rel=1e-3)  # the neighbouring isotope peak is outside it


def test_auto_takes_areas_when_weak_peaks_are_wider_and_heights_when_they_are_not():
    wide = choose_measure(control_set(MIN_CONTROLS + 5, widening=0.25), CFG)
    assert wide.measure == "area" and wide.setting == "auto"
    assert wide.width_ratio > 1.1 and wide.fwhm_ppm == pytest.approx(FWHM_PPM, rel=0.05)
    assert "wide as the tallest" in wide.reason
    same = choose_measure(control_set(MIN_CONTROLS + 5, widening=0.0), CFG)
    assert same.measure == "height" and same.width_ratio == pytest.approx(1.0, abs=0.03)
    assert "below" in same.reason
    few = choose_measure(control_set(MIN_CONTROLS - 5, widening=0.25), CFG)
    assert few.measure == "height" and "fewer than" in few.reason
    assert choose_measure(control_set(3, 0.25), PickerConfig(measure="height")) is HEIGHTS
    forced = choose_measure(control_set(3, 0.0), PickerConfig(measure="area"))
    assert forced.measure == "area" and forced.reason == "set"


def test_centroided_controls_keep_heights():
    plan = plan_bins(PEPTIDE, 2, config=CFG)
    entry = PeptideEntry("f", "S", PEPTIDE, "0001-0010", 1, 10, 2)
    shape = plan.clusters[0][: len(plan.pred_mz)]
    sets = [Spectrum(plan.pred_mz[: len(shape)], 1e5 * shape, entry, float("nan"), k + 1,
                     spectrum_id=f"p:{k}", role="undeut") for k in range(MIN_CONTROLS + 1)]
    m = choose_measure(SpectrumSet.from_spectra(sets), CFG)
    assert m.measure == "height" and m.reason == "the spectra are centroided"


def test_areas_recover_the_envelope_that_heights_under_read():
    s = control_set(MIN_CONTROLS + 5, widening=0.25)
    heights = pick_spectra(s, PickerConfig(measure="height"))
    areas = pick_spectra(s, CFG)
    plan = plan_bins(PEPTIDE, 2, config=CFG)
    shape = plan.clusters[0] / plan.clusters[0].max()

    def misfit(b):
        _, y, nd = b.spectrum(0)
        y = y / y.max()
        n = min(len(y), 4)
        return float(np.max(np.abs(y[:n] - shape[nd[:n]])))

    assert misfit(heights) > 0.03 and misfit(areas) < 0.01
    assert areas.spectrum(0)[1].max() == pytest.approx(heights.spectrum(0)[1].max(), rel=0.01)


def test_the_setting_is_checked_and_round_trips():
    for bad in ({"measure": "integral"}, {"area_fwhm": 0}, {"area_fwhm": True}):
        with pytest.raises(ValueError):
            PickerConfig(**bad)
    cfg = PickerConfig(measure="area", area_fwhm=2.0)
    assert PickerConfig.from_dict(cfg.to_dict()) == cfg and PickerConfig().measure == "auto"
    m = PeakMeasure("height", "auto", reason="few")  # NaN widths are written as null
    assert m.to_dict()["fwhm_ppm"] is None
    assert PeakMeasure.from_dict(m.to_dict()).to_dict() == m.to_dict()
    full = PeakMeasure("area", "auto", 47.0, 0.1, 1.12, 30, 40, 1.5, "wide", 1.06)
    assert PeakMeasure.from_dict(full.to_dict()) == full


def test_a_run_records_its_choice_and_the_bundle_keeps_it(tmp_path):
    from hxex.run import RunConfig, run

    result = run(control_set(MIN_CONTROLS + 5, widening=0.25, labelled=3),
                 RunConfig(bootstrap=None), out=tmp_path / "b")
    assert result.peak_measure.measure == "area"
    with open_bundle(tmp_path / "b") as b:
        assert b.peak_measure() == result.peak_measure
        assert b.manifest["peak_measure"]["measure"] == "area"
    assert "areas (weak peaks" in result.peak_measure.describe()
