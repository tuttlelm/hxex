# SPDX-License-Identifier: GPL-3.0-or-later
"""The picker's m/z check (hxex.picking.recentre): a series whose undeuterated envelope sits off
its m/z, compared with its run, is flagged off_mz and its windows are moved onto its envelope.
Synthetic profile spectra: one run of twelve series, one of them 20 ppm off, picked with a
+-10 ppm window (which an offset of 20 ppm puts on the flank of the peaks)."""

import dataclasses

import numpy as np
import pytest

from hxex import load_results
from hxex.fit import FitConfig
from hxex.model import Backend
from hxex.noise import NoiseConfig
from hxex.picking import BinnedSet, PickerConfig, pick_spectra
from hxex.picking.binned import MZ_CHECK_COLUMNS
from hxex.picking.recentre import MIN_LIMIT_PPM, envelope_offset, series_offsets
from hxex.run import RunConfig, run
from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet
from test_screen import PLAN, envelope, profile

PEPTIDE, Z = "RDKVQKEYAL", 2
NARROW = PickerConfig(resolution_ppm=10.0)
OFF = 20.0  # ppm: the off series' envelope, beyond the run's calibration
CALIBRATION = 1.0  # ppm: the run's own offset


def spectra_set(n=12, off=OFF, labelled=True, samples=False) -> SpectrumSet:
    """An undeuterated control and a labelled spectrum for each of ``n`` series of one run
    (``samples``: each series in a sample of its own, so each run has one control); series 0
    sits ``off`` ppm from the others, which scatter by about 0.5 ppm."""
    rng = np.random.default_rng(1)
    out = []
    for s in range(n):
        entry = PeptideEntry("f", f"S{s}" if samples else "S", PEPTIDE,
                             f"{s + 1:04d}-{s + 10:04d}", s + 1, s + 10, Z)
        shift = CALIBRATION + rng.normal(0, 0.5) + (off if s == 0 else 0.0)
        for level, time, role in ((0.0, np.nan, "undeut"), (2.0, 60.0, "")):
            if role == "" and not labelled:
                continue
            pos, h = envelope(level)
            out.append(Spectrum(*profile((pos * (1 + shift * 1e-6), h)), entry, time, 1,
                                data_id=s, spectrum_id=f"{s}:{level}", role=role))
    return SpectrumSet.from_spectra(out)


def series_rows(b: BinnedSet, s: int) -> np.ndarray:
    return np.flatnonzero(np.asarray(b.data_id) == s)


def test_the_envelope_offset_is_found_beyond_the_window():
    pos, h = envelope(0.0)
    assert envelope_offset(*profile((pos, h)), PLAN.pred_mz, Z, 10.0) == pytest.approx(0, abs=1.0)
    shifted = profile((pos * (1 + 25e-6), h))
    assert envelope_offset(*shifted, PLAN.pred_mz, Z, 10.0) == pytest.approx(25, abs=0.5)
    # beyond the search range (40 ppm, or twice the half-width): nothing found
    far = profile((pos * (1 + 150e-6), h))
    assert np.isnan(envelope_offset(*far, PLAN.pred_mz, Z, 10.0))


def test_an_envelope_whose_peaks_disagree_has_no_offset():
    pos, h = envelope(0.0)
    pos = pos.copy()
    pos[0] *= 1 + 25e-6  # only the monoisotopic peak moved: not one envelope
    assert np.isnan(envelope_offset(*profile((pos, h)), PLAN.pred_mz, Z, 10.0))


def test_a_series_off_its_mz_is_flagged_and_its_windows_moved():
    raw = spectra_set()
    moved = pick_spectra(raw, NARROW)
    off = series_rows(moved, 0)
    rest = np.flatnonzero(np.asarray(moved.data_id) != 0)
    assert moved.envelope_offset_ppm[off] == pytest.approx(OFF, abs=1.0)  # calibration out
    assert np.all(np.abs(moved.envelope_offset_ppm[rest]) < moved.envelope_offset_limit_ppm[rest])
    assert np.all(moved.envelope_offset_limit_ppm >= MIN_LIMIT_PPM)
    assert moved.window_shift_ppm[off] == pytest.approx(OFF + CALIBRATION, abs=1.0)
    assert np.all(moved.window_shift_ppm[rest] == 0)
    # the peaks' offset is still measured from the predicted m/z
    assert moved.mz_offset_ppm[off[0]] == pytest.approx(OFF + CALIBRATION, abs=1.5)
    # with the windows on the peaks, the off series reads its peaks' tops, as the others do
    flank = pick_spectra(raw, dataclasses.replace(NARROW, recentre=False))
    m_moved, m_flank = moved.spectrum(off[0])[1][0], flank.spectrum(off[0])[1][0]
    m_other = moved.spectrum(rest[0])[1][0]
    assert m_moved == pytest.approx(m_other, rel=0.02)
    assert m_flank < 0.9 * m_moved  # the window edge on the flank of a 50 ppm wide peak
    # only the off series changed
    assert np.array_equal(moved.intensity[moved.offsets[rest[0]]:moved.offsets[rest[0] + 1]],
                          flank.intensity[flank.offsets[rest[0]]:flank.offsets[rest[0] + 1]])


def test_recentre_false_flags_only_and_off_mz_sd_false_checks_nothing():
    raw = spectra_set()
    moved = pick_spectra(raw, NARROW)
    flagged = pick_spectra(raw, dataclasses.replace(NARROW, recentre=False))
    assert np.array_equal(flagged.envelope_offset_ppm, moved.envelope_offset_ppm)
    assert np.all(flagged.window_shift_ppm == 0)
    unchecked = pick_spectra(raw, dataclasses.replace(NARROW, off_mz_sd=None))
    assert np.isnan(unchecked.envelope_offset_ppm).all()
    assert np.isnan(unchecked.envelope_offset_limit_ppm).all()
    assert np.all(unchecked.window_shift_ppm == 0)
    assert np.array_equal(unchecked.intensity, flagged.intensity)
    # a larger multiple of the scatter lets the off series pass
    loose = pick_spectra(raw, dataclasses.replace(NARROW, off_mz_sd=1e3))
    assert np.all(loose.window_shift_ppm == 0)


def test_a_run_with_few_controls_takes_the_scatter_of_every_control():
    b = pick_spectra(spectra_set(samples=True), NARROW)  # every run has one control
    shift = b.window_shift_ppm
    assert shift[series_rows(b, 0)] == pytest.approx(OFF + CALIBRATION, abs=1.0)
    assert np.all(shift[np.asarray(b.data_id) != 0] == 0)


def test_series_without_controls_and_skipped_series_are_not_assessed():
    raw = spectra_set()
    keep = ~((np.asarray(raw.data_id) == 1) & (np.asarray(raw.role) == "undeut"))
    b = pick_spectra(raw.select(keep), NARROW)  # series 1 has no undeuterated control
    assert np.isnan(b.envelope_offset_ppm[series_rows(b, 1)]).all()
    checked = series_offsets(raw, NARROW, skip={0})  # fragment ions are skipped this way
    assert 0 not in set(checked.data_id) and 1 in set(checked.data_id)
    with pytest.raises(ValueError, match="off_mz_sd"):
        series_offsets(raw, dataclasses.replace(NARROW, off_mz_sd=None))


def test_the_check_follows_select_and_older_sets_are_not_assessed():
    b = pick_spectra(spectra_set(), NARROW)
    one = b.select(np.asarray(b.data_id) == 0)
    assert np.all(one.window_shift_ppm > 0)
    older = dataclasses.replace(b, **{name: None for name in MZ_CHECK_COLUMNS})
    assert np.isnan(older.envelope_offset_ppm).all() and np.all(older.window_shift_ppm == 0)


def test_settings_are_checked():
    with pytest.raises(ValueError, match="off_mz_sd"):
        PickerConfig(off_mz_sd=0)
    with pytest.raises(ValueError, match="recentre"):
        PickerConfig(recentre="yes")
    cfg = PickerConfig(off_mz_sd=4.0, recentre=False)
    assert PickerConfig.from_dict(cfg.to_dict()) == cfg


def test_the_run_flags_off_mz_and_the_bundle_keeps_the_check(tmp_path):
    cpu = Backend("cpu", "float64")
    fit = FitConfig(k_max=1, adam_steps=60, n_restarts=1, backend=cpu,
                    noise=NoiseConfig("gaussian", percent=3.0))
    result = run(spectra_set(), RunConfig(picker=NARROW, fit=fit), out=tmp_path / "b")
    q = result.quality
    by_series = np.asarray(result.fit.data_id)
    assert np.all(q.off_mz[by_series == 0]) and not np.any(q.off_mz[by_series != 0])
    assert q.envelope_offset[by_series == 0] == pytest.approx(OFF, abs=1.0)
    assert not np.any(q.mz_shift)  # a checked series is not also judged by mz_shift
    loaded = load_results(tmp_path / "b")
    assert np.array_equal(loaded.quality.off_mz, q.off_mz)
    assert np.allclose(loaded.binned.window_shift_ppm, result.binned.window_shift_ppm)
    assert np.array_equal(np.isnan(loaded.binned.envelope_offset_ppm),
                          np.isnan(result.binned.envelope_offset_ppm))
