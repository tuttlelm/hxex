# SPDX-License-Identifier: GPL-3.0-or-later
"""Bin values from a fit of the raw profile (``[run.picker] pick = "profile_fit"``,
hxex.picking.profile)."""

import math

import numpy as np
import pytest
from scipy.stats import binom

from hxex import chem
from hxex.io import open_bundle
from hxex.picking import PeakMeasure, PickerConfig, choose_measure, pick_spectra, plan_bins, profile
from hxex.picking.profile import (
    PeakShape,
    bin_positions,
    fine_moments,
    ion_moments,
    search_offsets,
)
from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet

PEPTIDE = "RDKVQKEYAL"
CHARGE = 2
FWHM_PPM = 47.0  # a Q-TOF's peak width
SPACING_PPM = 12.0
CFG = PickerConfig(pick="profile_fit")
ENTRY = PeptideEntry("f", "S", PEPTIDE, "0001-0010", 1, 10, CHARGE)


def envelope(deuteration: float, top: float, shift_ppm: float = 0.0, extra=()):
    """A profile spectrum of the peptide at ``deuteration`` (binomial over its sites; 0: the
    natural envelope): every bin a Gaussian whose AREA is its abundance (its width the run's
    widened by the bin's isotopologue spread), the whole moved by ``shift_ppm``; ``extra``
    adds Gaussians of the run's width ``(m/z, height)`` (another species). Returns the m/z,
    the intensities and the bins' true abundances (tallest 1)."""
    plan = plan_bins(PEPTIDE, CHARGE, config=CFG)
    mom = ion_moments(PEPTIDE)
    sites = plan.n_sites
    d = (np.r_[1.0, np.zeros(sites)] if deuteration <= 0
         else binom.pmf(np.arange(sites + 1), sites, deuteration / sites))
    n = len(plan.pred_mz)
    mean, spread = bin_positions(d, mom, n)
    a, _, _ = mom
    abundance = np.zeros(n)
    for k, dk in enumerate(d):
        idx = k + np.arange(len(a))
        keep = idx < n
        abundance[idx[keep]] += dk * a[keep]
    pos = (plan.undeut_mz + mean / CHARGE) * (1 + shift_ppm * 1e-6)
    fw = pos * FWHM_PPM * 1e-6
    width = np.sqrt(fw ** 2 + (profile.FWHM_SD * spread / CHARGE) ** 2)
    lo, hi = pos[0] - 2.0, pos[-1] + 2.0
    mz = lo * (1 + SPACING_PPM * 1e-6) ** np.arange(int(math.log(hi / lo) / (SPACING_PPM * 1e-6)))
    y = np.zeros_like(mz)
    keep = abundance > 1e-4 * abundance.max()
    for c, area, w in zip(pos[keep], abundance[keep], width[keep], strict=True):
        h = top * area / abundance.max() * fw.mean() / w
        y += h * np.exp(-0.5 * ((mz - c) / (w / profile.FWHM_SD)) ** 2)
    for c, h in extra:
        y += h * np.exp(-0.5 * ((mz - c) / (c * FWHM_PPM * 1e-6 / profile.FWHM_SD)) ** 2)
    return mz, y, abundance / abundance.max()


def run_set(n_controls=24, labelled=((3.0, 0.0),), shift_ppm=0.0, extra=(), more=()):
    """``n_controls`` undeuterated controls and one labelled spectrum per ``(deuteration,
    shift)`` (its own shift added to ``shift_ppm``), each in its own injection; ``more``
    spectra added as they are."""
    rng = np.random.default_rng(0)
    out = []
    for k in range(n_controls):
        mz, y, _ = envelope(0.0, float(10 ** rng.uniform(4.5, 5.5)), shift_ppm)
        out.append(Spectrum(mz, y, ENTRY, float("nan"), k + 1, data_id=0,
                            spectrum_id=f"p:un{k}", role="undeut"))
    for k, (level, own) in enumerate(labelled):
        mz, y, _ = envelope(level, 1e5, shift_ppm + own, extra)
        out.append(Spectrum(mz, y, ENTRY, 60.0, k + 1, data_id=0, spectrum_id=f"p:t{k}"))
    return SpectrumSet.from_spectra([*out, *more])


def test_the_fine_structure_places_the_bins_where_the_composition_does():
    """The moments' abundances and mean masses agree with hxex's natural isotope clusters."""
    plan = plan_bins(PEPTIDE, CHARGE, config=PickerConfig())
    a, mu, var = ion_moments(PEPTIDE)
    shape, offsets = plan.clusters
    n = min(len(shape), 5)
    np.testing.assert_allclose(a[:n], shape[:n], atol=2e-3)
    np.testing.assert_allclose(mu[:n], offsets[:n], atol=2e-4)
    assert mu[0] == 0.0 and var[0] == 0.0 and var[1] > 0  # M+1 mixes 13C, 15N, 2H...
    # one carbon: the M+1 shell sits at the 13C mass difference
    np.testing.assert_allclose(fine_moments({"C": 1})[1][1], 1.0033548378, atol=1e-9)


def test_the_run_measures_its_peak_shape_and_records_it():
    m = choose_measure(run_set(), CFG)
    assert m.measure == "profile_fit" and m.fitted and m.shape.measured
    assert m.fwhm_ppm == pytest.approx(FWHM_PPM, rel=0.05)
    gauss = PeakShape.gaussian()
    for row, ref in zip(m.shape.rows, gauss.rows, strict=True):  # Gaussian peaks: a Gaussian
        assert np.abs(row - ref).max() < 0.08
    again = PeakMeasure.from_dict(m.to_dict())
    np.testing.assert_allclose(again.shape.rows[0], m.shape.rows[0], atol=1e-5)
    assert again.shape.n_peaks == m.shape.n_peaks and "fitted peak areas" in again.describe()
    few = choose_measure(run_set(n_controls=5), CFG)
    assert not few.shape.measured and "fewer than" in few.shape.describe()


def test_without_undeuterated_controls_the_fit_is_refused():
    s = run_set(n_controls=0, labelled=((3.0, 0.0),))
    with pytest.raises(ValueError, match="profile_fit"):
        choose_measure(s, CFG)


def test_the_fit_reads_the_envelope_and_its_offset():
    s = run_set(labelled=((3.0, 0.0), (6.0, 0.0)), shift_ppm=4.0)
    b = pick_spectra(s, CFG, workers=1)
    for i in (len(b) - 2, len(b) - 1):
        level = 3.0 if i == len(b) - 2 else 6.0
        _, _, truth = envelope(level, 1.0)
        _, v, nd = b.spectrum(i)
        np.testing.assert_allclose(v / v.max(), truth[nd], atol=0.02)
        assert b.fit_offset_ppm[i] == pytest.approx(4.0, abs=0.6)
        assert b.purity[i] > 0.98 and b.stray_share[i] < 0.01
        assert np.isfinite(b.peak_mz[b.offsets[i]:b.offsets[i + 1]][v > 0]).all()
    assert np.isfinite(b.fit_offset_ppm).all()


def test_another_species_goes_to_a_stray_peak_not_into_the_bins():
    """A peak of another species 1.5 widths heavy of a weak bin: the pick's area takes in its
    flank, the fit gives it a peak of its own."""
    _, _, truth = envelope(3.0, 1.0)
    plan = plan_bins(PEPTIDE, CHARGE, config=CFG)
    b0 = int(np.flatnonzero(truth > 0.15)[-1])  # the last bin above 15 % of the tallest
    at = plan.pred_mz[b0] * (1 + 1.5 * FWHM_PPM * 1e-6)
    s = run_set(extra=((at, 0.5e5),))
    fitted = pick_spectra(s, CFG, workers=1)
    picked = pick_spectra(s, PickerConfig(measure="area"))
    i = len(s) - 1

    def misfit(b):
        _, v, nd = b.spectrum(i)
        return float(np.abs(v / v.max() - truth[nd]).max())

    assert misfit(fitted) < 0.03 < misfit(picked)
    assert fitted.purity[i] > 0.95 and fitted.stray_share[i] > 0.05  # a weak bin: little weight


def test_purity_counts_only_what_touches_the_peptides_peaks():
    """A species resolved between two bins costs no purity (it shows in the stray share); one
    beside a bin, under its peak, does."""
    plan = plan_bins(PEPTIDE, CHARGE, config=CFG)
    _, _, truth = envelope(3.0, 1.0)
    top = int(np.argmax(truth))

    def quality(extra):
        b = pick_spectra(run_set(extra=extra), CFG, workers=1)
        return b.purity[-1], b.stray_share[-1]

    clean, _ = quality(())
    between, stray = quality(((plan.pred_mz[top] + 1.00335 / CHARGE / 2, 1e5),))
    beside, _ = quality(((plan.pred_mz[top] * (1 + 0.7 * FWHM_PPM * 1e-6), 0.8e5),))
    assert clean > 0.98 and between == pytest.approx(clean, abs=0.01) and stray > 0.1
    assert beside < 0.9


def test_fragments_other_ion_types_and_centroided_spectra_keep_the_pick():
    plan = plan_bins(PEPTIDE, CHARGE, config=CFG)
    shape = plan.clusters[0][: len(plan.pred_mz)]
    sticks = Spectrum(plan.pred_mz[: len(shape)], 1e5 * shape, ENTRY, 30.0, 1, data_id=1,
                      spectrum_id="p:centroided")
    c_ion = PeptideEntry("f", "S", "RDKVQ", "0001-0005", 1, 5, 1, modification="ion_type:c")
    c_plan = plan_bins("RDKVQ", 1, modification="ion_type:c", config=CFG)
    mz = np.linspace(c_plan.pred_mz[0] - 2, c_plan.pred_mz[0] + 6, 8000)
    frag = Spectrum(mz, 1e5 * np.exp(-0.5 * ((mz - c_plan.pred_mz[0]) / 0.01) ** 2), c_ion,
                    30.0, 1, data_id=2, spectrum_id="p:c")
    s = run_set(more=(sticks, frag))
    b = pick_spectra(s, CFG, workers=1)
    assert np.isfinite(b.fit_offset_ppm[: len(s) - 2]).all()
    assert np.isnan(b.fit_offset_ppm[-2:]).all() and np.isnan(b.purity[-2:]).all()
    assert np.isnan(b.stray_share[-2:]).all()
    plain = pick_spectra(s, PickerConfig(measure="area"))
    for i in (len(s) - 2, len(s) - 1):  # as the pick seeds the fit: with areas
        np.testing.assert_array_equal(b.spectrum(i)[1], plain.spectrum(i)[1])


def test_the_search_centres_on_each_injection_and_spans_its_scatter():
    own = np.array([1.0, 1.4, 0.6, 5.0, 5.4, 4.6, np.nan])
    inj = ["a", "a", "a", "b", "b", "b", "c"]
    found = search_offsets(own, inj)
    np.testing.assert_allclose(found.centre, [1, 1, 1, 5, 5, 5, 3.0])  # "c": the run's median
    assert found.limit == profile.SEARCH_FLOOR_PPM and found.step == profile.MIN_STEP_PPM
    wide = search_offsets(np.array([0.0, 4.0, -4.0, 8.0, -8.0, 2.0]), ["a"] * 6)
    assert wide.limit == pytest.approx(profile.SEARCH_SD * wide.scatter)


def test_processes_give_what_one_process_gives(monkeypatch):
    s = run_set(n_controls=20, labelled=((3.0, 0.0), (5.0, 2.0), (8.0, -2.0)))
    one = pick_spectra(s, CFG, workers=1)
    monkeypatch.setattr(profile, "PARALLEL_FROM", 1)
    monkeypatch.setattr(profile, "CHUNK", 5)
    many = pick_spectra(s, CFG, workers=2)
    # equal up to rounding, not bit for bit: the workers run one thread of linear algebra,
    # this process as many as its libraries take, and the two sum the Gram matrices in
    # different orders (the lowest-versions CI job differed by 9e-7 on values near 1e5)
    np.testing.assert_allclose(one.intensity, many.intensity, rtol=1e-8,
                               atol=1e-9 * np.nanmax(one.intensity))
    np.testing.assert_array_equal(one.fit_offset_ppm, many.fit_offset_ppm)


def test_the_setting_is_a_choice_of_pick():
    assert PickerConfig.from_dict(CFG.to_dict()) == CFG
    with pytest.raises(ValueError, match="resolving_power"):
        PickerConfig(pick="profile_fit", resolving_power=20000)
    from hxex.config import project_from

    project = project_from({"project": "p", "input": {"format": "specexport", "spectra": "x"},
                            "run": {"picker": {"pick": "profile_fit"}}})
    assert project.run.picker.pick == "profile_fit"


def test_a_run_records_the_fit_and_the_bundle_keeps_it(tmp_path):
    from hxex.pipeline import RunConfig, run
    from hxex.report.notebook import load_results
    from hxex.report.tables import spectra_table

    result = run(run_set(labelled=((3.0, 0.0), (5.0, 0.0), (8.0, 0.0))),
                 RunConfig(picker=CFG, bootstrap=None), out=tmp_path / "b")
    assert result.peak_measure.fitted
    with open_bundle(tmp_path / "b") as b:
        assert b.manifest["peak_measure"]["measure"] == "profile_fit"
        assert b.peak_measure().shape is not None
        binned = b.binned()
        assert np.isfinite(binned.fit_offset_ppm).all() and np.isfinite(binned.purity).all()
        assert np.isfinite(binned.stray_share).all()
    table = spectra_table(load_results(tmp_path / "b"))
    assert np.isfinite(table["fit_offset_ppm"]).all()
    assert np.isfinite(table["purity"]).all() and "stray_share" in table
    # the run judges each spectrum's purity (clean synthetic spectra: none below the level)
    assert np.array_equal(result.quality.low_purity,
                          result.binned.purity[result.kept] < result.config.quality.min_purity)
    assert not result.quality.low_purity.any()


def test_moments_skip_elements_without_isotopes_here():
    assert fine_moments({"C": 2, "Se": 1})[0][0] == pytest.approx(0.9893 ** 2)
    assert chem.parse_modification("").ion_type == "M"


def test_the_picking_check_draws_the_fit_of_one_spectrum():
    from hxex.pipeline import RunConfig, run
    from hxex.report import Results
    from hxex.report.picking_check import picking_check_data, plot_picking_check

    s = run_set(labelled=((3.0, 0.0),),
                extra=((plan_bins(PEPTIDE, CHARGE, config=CFG).pred_mz[3] + 1.00335 / CHARGE / 2,
                        5e4),))
    results = Results.from_run(run(s, RunConfig(picker=CFG, bootstrap=None)), spectra=s)
    sid = str(s.spectrum_id[-1])
    d = picking_check_data(sid, results)
    b = results.binned
    i = b.find(sid)
    assert d["purity"] == pytest.approx(b.purity[i]) and d["offset_ppm"] == b.fit_offset_ppm[i]
    assert d["strays"].max() > 0 and np.all(d["bins"] >= 0)
    fig = plot_picking_check(sid, results)
    gids = {a.get_gid() for a in fig.axes[0].get_children()}
    assert {"raw", "bins", "strays", "values"} <= gids
    assert "purity" in fig.axes[0].get_title(loc="left")


def test_the_picking_check_refuses_a_spectrum_that_kept_the_pick():
    from hxex.pipeline import RunConfig, run
    from hxex.report import Results
    from hxex.report.picking_check import plot_picking_check

    s = run_set(n_controls=20)
    results = Results.from_run(run(s, RunConfig(picker=PickerConfig(), bootstrap=None)),
                               spectra=s)
    with pytest.raises(ValueError, match="profile fit"):
        plot_picking_check(str(s.spectrum_id[-1]), results)
