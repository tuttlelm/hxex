# SPDX-License-Identifier: GPL-3.0-or-later
"""The drift gate's m/z position test (hxex.io.drift_gate): a chosen drift range off the ion's
m/z with another drift peak on it is named and flagged; profiles measured at a file's offset."""

from types import SimpleNamespace

import numpy as np
import pytest

from hxex.io.drift_gate import (
    MIN_POSITION_IONS,
    DriftProfiles,
    _peak_span,
    drift_peaks,
    drift_record,
    scan_profile,
)

DRIFTS = np.round(np.arange(1.0, 2.0, 0.07), 6)


def _peak(centre: int, height: float, width: float = 0.8) -> np.ndarray:
    j = np.arange(len(DRIFTS))
    return height * np.exp(-0.5 * ((j - centre) / width) ** 2)


def _add(profiles, key, parts, rng):
    """One file's profile of ``key``: ``parts`` are (centre bin, height, m/z position ppm)."""
    on = np.zeros(len(DRIFTS))
    w = np.zeros(len(DRIFTS))
    wp = np.zeros(len(DRIFTS))
    for centre, height, ppm in parts:
        p = _peak(centre, height)
        on += p
        w += p
        wp += p * (ppm + rng.normal(0, 0.5))
    profiles.add(key, DRIFTS, on, np.zeros(len(DRIFTS)), w, wp, file="un.mzML")


def _run(chosen_ppm: float, other_ppm: float, other_height: float = 0.3):
    rng = np.random.default_rng(1)
    profiles = DriftProfiles()
    for i in range(MIN_POSITION_IONS + 5):  # ions with one drift peak on their m/z
        _add(profiles, (f"P{i}", "", 1, 9, 2), [(6, 1e5, 3.0 + rng.normal(0, 2))], rng)
    # the ion: its tallest drift peak at chosen_ppm, a smaller one 4 bins earlier at other_ppm
    _add(profiles, ("ION", "", 1, 11, 3), [(7, 1e5, chosen_ppm), (3, other_height * 1e5,
                                                                    other_ppm)], rng)
    return profiles, profiles.ranges()


def test_a_drift_peak_on_the_ions_mz_is_named_when_the_chosen_one_sits_off_it():
    profiles, ranges = _run(chosen_ppm=34.0, other_ppm=3.0)
    r = ranges[("ION", "", 1, 11, 3)]
    assert r.apex == pytest.approx(DRIFTS[7])
    assert r.second < 0.5  # below the isotope test: seen only through the m/z
    assert r.offset_ppm == pytest.approx(34.0, abs=2)
    assert r.better is not None and r.better[2] == pytest.approx(DRIFTS[3])
    assert r.better[0] <= DRIFTS[3] <= r.better[1] < DRIFTS[7]  # its own span, not the apex's
    assert r.unclear
    centre, sd, n = profiles.position
    assert centre == pytest.approx(3.0, abs=1.5) and 1.0 <= sd < 4 and n == MIN_POSITION_IONS + 5
    # the single-peak ions are left alone
    assert not any(v.unclear for k, v in ranges.items() if k[0] != "ION")
    rec = drift_record(ranges, [7], profiles.position, {"un.mzML": 31.4})
    entry = next(e for e in rec["ranges"] if e["peptide"] == "ION")
    assert entry["better"][2] == pytest.approx(DRIFTS[3]) and len(entry["peaks"]) == 2
    assert rec["position"]["ions"] == MIN_POSITION_IONS + 5
    assert rec["profile_offsets"] == {"un.mzML": 31.4}


def test_no_better_range_when_the_chosen_peak_is_on_its_mz_or_no_peak_is():
    _, ranges = _run(chosen_ppm=3.0, other_ppm=34.0)  # the other drift peak is the other ion
    assert ranges[("ION", "", 1, 11, 3)].better is None
    assert not ranges[("ION", "", 1, 11, 3)].unclear
    _, ranges = _run(chosen_ppm=40.0, other_ppm=30.0)  # nothing on its m/z: off_mz's case
    assert ranges[("ION", "", 1, 11, 3)].better is None


def test_too_few_single_peak_ions_measure_no_position_scatter():
    rng = np.random.default_rng(2)
    profiles = DriftProfiles()
    for i in range(5):
        _add(profiles, (f"P{i}", "", 1, 9, 2), [(6, 1e5, 3.0)], rng)
    _add(profiles, ("ION", "", 1, 11, 3), [(7, 1e5, 34.0), (3, 3e4, 3.0)], rng)
    ranges = profiles.ranges()
    assert profiles.position is None and ranges[("ION", "", 1, 11, 3)].better is None
    assert drift_record(ranges, [])["position"] is None


def test_a_smaller_peaks_span_stops_at_the_valley():
    on = _peak(7, 1.0) + _peak(3, 0.3)
    lo, hi = _peak_span(on, 3)
    valley = 3 + int(np.argmin(on[3:8]))
    assert lo <= 3 <= hi < valley  # the valley bin holds both peaks' tails: left out
    found = drift_peaks(DRIFTS, on, on, on * 5.0)
    assert [round(h, 2) for _, _, h, _ in found] == [0.3, 1.0]
    assert all(o == pytest.approx(5.0) for *_, o in found)


def test_the_profile_is_measured_at_the_files_offset():
    """Peaks 30 ppm above the prediction (an uncorrected file): with the offset they sit on
    the isotope peaks and their position is ~0 ppm; without it, the narrow window misses them."""
    pred = np.array([500.0, 500.5, 501.0])
    mz = pred * (1 + 30e-6)
    scan = SimpleNamespace(binned=(mz, np.array([100.0, 60.0, 20.0]), np.array([1.5, 1.5, 1.5])))
    half = pred * 10e-6
    d, on, between, w, wp = scan_profile([scan], pred, half, 30.0, positions=True)
    assert on[0] == pytest.approx(180.0) and wp[0] / w[0] == pytest.approx(0.0, abs=1e-6)
    _, on0, *_ = scan_profile([scan], pred, half, positions=True)
    assert on0[0] == 0.0
    assert len(scan_profile([scan], pred, half)) == 3  # the plain form stays three arrays
