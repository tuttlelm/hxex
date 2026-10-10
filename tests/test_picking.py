# SPDX-License-Identifier: GPL-3.0-or-later
"""Peak-picker tests: synthetic spectra for the logic, and pyHXExpress output as the oracle.

The golden fixture (tests/golden/picker_reference.npz) holds raw spectra and the reference
``peak_picker`` output from an October 2026 pyHXExpress run (Zero_Filling 3, setNoise 50,
Nterm_subtract 2). Where pyHXExpress and hxex differ on purpose (see docs/PLAN.md) the tests
check the relation between the two, not equality. HXEX_PICKER_ORACLE can point to a whole run
directory (``allrawdata_*.zip`` and ``alldeutdata_*.csv``) for the same checks on every spectrum.
"""

import csv
import io
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from hxex import chem
from hxex.hooks import Cancelled
from hxex.picking import (
    BinnedSet,
    PickerConfig,
    envelope,
    pick_spectra,
    pick_spectrum,
    plan_bins,
    predicted_mz,
)
from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet

PEPTIDE = "RDKVQKEYAL"  # 10 residues
CFG = PickerConfig(zero_filling=3, noise=100.0)
GOLDEN_CFG = PickerConfig(zero_filling=3, noise=50.0, nterm_subtract=2)


def sticks(plan, heights, spread=0.002):
    """A spectrum with a few points around each predicted bin; ``heights`` maps bin -> height."""
    mz, y = [], []
    for i, h in sorted(heights.items()):
        for dx, f in ((-spread, 0.3), (0.0, 1.0), (spread, 0.3)):
            mz.append(plan.pred_mz[i] + dx)
            y.append(h * f)
    return np.array(mz), np.array(y)


def pick(heights, cfg=CFG, peptide=PEPTIDE, charge=2, **kw):
    plan = plan_bins(peptide, charge, config=cfg)
    mz, y = sticks(plan, heights)
    return plan, pick_spectrum(mz, y, peptide, charge, config=cfg, **kw)


# --- bins -----------------------------------------------------------------------------------

def test_predicted_mz_uses_exms_steps():
    z, n = 2, 8
    steps = [1.003355, 1.003355, 1.004816, 1.004816, 1.006277, 1.006277, 1.006277, 1.006277]
    expected = [500.0]
    for s in steps:
        expected.append(expected[-1] + s / z)
    # the reference accumulates the steps first and adds the m/z last
    acc = [0.0]
    for s in steps:
        acc.append(acc[-1] + s / z)
    np.testing.assert_array_equal(predicted_mz(500.0, z, n), np.array(acc) + 500.0)
    np.testing.assert_allclose(predicted_mz(500.0, z, n), expected, rtol=1e-14)


def test_plan_bins():
    plan = plan_bins(PEPTIDE, 2, config=CFG)
    assert plan.pred_mz[0] == chem.mz(PEPTIDE, 2)
    assert plan.max_namides == chem.count_amides(PEPTIDE)
    n_na = len(chem.natural_abundance(PEPTIDE))
    assert len(plan.pred_mz) == plan.max_namides + (n_na - 1) + CFG.zero_filling + 1


def test_plan_bins_modification_and_ion_type():
    base = plan_bins(PEPTIDE, 2, config=CFG)
    hexed = plan_bins(PEPTIDE, 2, modification="Hex:2", config=CFG)
    assert len(hexed.pred_mz) == len(base.pred_mz) + 2 and hexed.max_namides == base.max_namides
    assert plan_bins(PEPTIDE, 2, modification="undeut_mz:400.5", config=CFG).pred_mz[0] == 400.5
    c = plan_bins(PEPTIDE, 2, modification="ion_type:c", config=CFG)
    assert c.pred_mz[0] == chem.mz(PEPTIDE, 2, "c")
    shifted = plan_bins(PEPTIDE, 2, modification="H:1", config=CFG)
    assert shifted.pred_mz[0] == pytest.approx(base.pred_mz[0] + 1.007825 / 2, abs=1e-6)


def test_plan_bins_with_ptm_and_nterm_subtract():
    plan = plan_bins("acKPEPTIDE", 2, config=PickerConfig(nterm_subtract=2))
    assert plan.max_namides == chem.count_amides("acKPEPTIDE", nterm_subtract=2) == 5
    assert plan.pred_mz[0] == chem.mz("acKPEPTIDE", 2)


# --- picking --------------------------------------------------------------------------------

def test_picks_window_maximum_at_predicted_mz():
    plan, p = pick({0: 1000.0, 1: 800.0, 2: 500.0})
    np.testing.assert_array_equal(p.mz, plan.pred_mz[: len(p.mz)])  # predicted, not observed
    np.testing.assert_array_equal(p.intensity[:3], [1000.0, 800.0, 500.0])
    np.testing.assert_array_equal(p.n_deut, np.arange(len(p.mz)))


def test_zero_filling_follows_the_last_signal_bin():
    _, p = pick({0: 1000.0, 1: 800.0, 2: 500.0})
    assert len(p.mz) == 2 + 1 + CFG.zero_filling
    assert not p.capped and not p.empty and not p.no_signal
    assert np.all(p.intensity[3:] == 0)


def test_valley_does_not_stop_picking():
    """pyHXExpress counted below-threshold bins cumulatively and stopped inside this valley."""
    heights = {0: 1000.0, 1: 800.0, 2: 500.0, 9: 700.0, 10: 900.0, 11: 400.0}
    _, p = pick(heights)
    assert len(p.mz) == 11 + 1 + CFG.zero_filling
    np.testing.assert_array_equal(p.intensity[9:12], [700.0, 900.0, 400.0])
    assert np.all(p.intensity[3:9] == 0)


def test_below_threshold_intensities_are_kept_as_is():
    _, p = pick({0: 1000.0, 1: 30.0, 2: 500.0, 3: 20.0})
    assert p.intensity[1] == 30.0 and p.intensity[3] == 20.0  # not zeroed
    assert len(p.mz) == 2 + 1 + CFG.zero_filling  # 20 is below the threshold: not a signal bin


def test_cap_and_capped_flag():
    plan = plan_bins(PEPTIDE, 2, config=CFG)
    n_last = len(plan.pred_mz) - 1
    _, p = pick({0: 1000.0, n_last: 500.0})
    assert len(p.mz) == n_last + 1 and p.capped
    _, p = pick({0: 1000.0, n_last - 1: 500.0})
    assert len(p.mz) == n_last + 1 and p.capped  # fewer than zero_filling bins left
    _, p = pick({0: 1000.0, n_last - CFG.zero_filling: 500.0})
    assert len(p.mz) == n_last + 1 and not p.capped  # fits exactly


def test_window_edges_are_inclusive():
    plan = plan_bins(PEPTIDE, 2, config=CFG)
    lo, hi = plan.pred_mz[0] * (1.0 + np.array([-1.0, 1.0]) * 50.0 / 1e6)
    inside = pick_spectrum(np.array([lo, hi]), np.array([300.0, 700.0]), PEPTIDE, 2, config=CFG)
    assert inside.intensity[0] == 700.0
    outside = pick_spectrum(
        np.array([np.nextafter(lo, 0), np.nextafter(hi, 1e9)]), np.array([300.0, 700.0]),
        PEPTIDE, 2, config=CFG,
    )
    assert outside.intensity[0] == 0.0 and outside.no_signal


def test_resolution_ppm():
    plan = plan_bins(PEPTIDE, 2, config=CFG)
    mz = np.array([plan.pred_mz[0] * (1 + 30e-6)])
    y = np.array([500.0])
    assert pick_spectrum(mz, y, PEPTIDE, 2, config=CFG).intensity[0] == 500.0
    narrow = PickerConfig(zero_filling=3, noise=100.0, resolution_ppm=20.0)
    assert pick_spectrum(mz, y, PEPTIDE, 2, config=narrow).intensity[0] == 0.0


def profile(plan, shape, n=12):
    """``n`` points spread across bin 0's window with the given intensities."""
    half = plan.pred_mz[0] * 50e-6
    return plan.pred_mz[0] + np.linspace(-0.9 * half, 0.9 * half, n), np.asarray(shape, float)


def test_shoulder_rejection_applies_to_windows_with_many_points():
    plan = plan_bins(PEPTIDE, 2, config=CFG)
    rising = np.linspace(100, 1000, 12)  # maximum at the window edge
    mz, y = profile(plan, rising)
    assert pick_spectrum(mz, y, PEPTIDE, 2, config=CFG).intensity[0] == 0.0
    peak = np.array([100, 200, 400, 700, 1000, 900, 600, 300, 200, 150, 120, 100], float)
    mz, y = profile(plan, peak)
    assert pick_spectrum(mz, y, PEPTIDE, 2, config=CFG).intensity[0] == 1000.0
    # a window with fewer than shoulder_min_points points is treated as stick data
    mz, y = profile(plan, np.linspace(100, 1000, 9), n=9)
    assert pick_spectrum(mz, y, PEPTIDE, 2, config=CFG).intensity[0] == 1000.0
    lenient = PickerConfig(zero_filling=3, noise=100.0, shoulder_min_points=20)
    mz, y = profile(plan, rising)
    assert pick_spectrum(mz, y, PEPTIDE, 2, config=lenient).intensity[0] == 1000.0


def test_ties_resolve_to_the_first_maximum():
    plan = plan_bins(PEPTIDE, 2, config=CFG)
    tie_edge = np.array([1000, 200, 300, 400, 300, 200, 300, 400, 300, 200, 100, 1000], float)
    mz, y = profile(plan, tie_edge)
    assert pick_spectrum(mz, y, PEPTIDE, 2, config=CFG).intensity[0] == 0.0
    tie_inner = np.array([100, 200, 900, 400, 300, 200, 300, 900, 300, 200, 100, 50], float)
    mz, y = profile(plan, tie_inner)
    assert pick_spectrum(mz, y, PEPTIDE, 2, config=CFG).intensity[0] == 900.0


def test_unsorted_input_is_sorted():
    plan = plan_bins(PEPTIDE, 2, config=CFG)
    mz, y = sticks(plan, {0: 1000.0, 1: 800.0, 2: 500.0})
    a = pick_spectrum(mz, y, PEPTIDE, 2, config=CFG)
    b = pick_spectrum(mz[::-1], y[::-1], PEPTIDE, 2, config=CFG)
    np.testing.assert_array_equal(a.intensity, b.intensity)


# --- noise threshold and flags --------------------------------------------------------------

def test_noise_threshold_absolute_and_percent():
    plan = plan_bins(PEPTIDE, 2, config=CFG)
    mz, y = sticks(plan, {0: 100000.0, 1: 800.0})
    percent = PickerConfig(zero_filling=3, noise_percent=1.0)
    p = pick_spectrum(mz, y, PEPTIDE, 2, config=percent)
    assert p.threshold == pytest.approx(0.01 * 100000.0) and len(p.mz) == 0 + 1 + 3
    zero = PickerConfig(zero_filling=3, noise=0, noise_percent=1.0)  # 0 means "use the percent"
    assert pick_spectrum(mz, y, PEPTIDE, 2, config=zero).threshold == p.threshold
    absolute = PickerConfig(zero_filling=3, noise=50.0)
    q = pick_spectrum(mz, y, PEPTIDE, 2, config=absolute)
    assert q.threshold == 50.0 and len(q.mz) == 1 + 1 + 3


def test_flags_for_empty_and_noise_only_spectra():
    p = pick_spectrum(np.array([]), np.array([]), PEPTIDE, 2, config=CFG)
    assert p.empty and p.no_signal and not p.capped
    assert np.isnan(p.env_width) and np.isnan(p.env_symm)
    assert len(p.mz) == CFG.zero_filling + 1
    noise_only = pick_spectrum(*sticks(plan_bins(PEPTIDE, 2, config=CFG), {0: 40.0, 1: 30.0}),
                               PEPTIDE, 2, config=CFG)
    assert noise_only.no_signal and not noise_only.empty  # below threshold, but not all zero
    no_data = pick_spectrum(np.array([]), np.array([]), PEPTIDE, 2,
                            config=PickerConfig(zero_filling=3, noise_percent=1.0))
    assert np.isnan(no_data.threshold) and no_data.no_signal


# --- envelope -------------------------------------------------------------------------------

def test_envelope_interpolates_edges():
    x = np.arange(6.0)
    y = np.array([0, 2, 10, 6, 1, 0.0])
    left, right, left_int = envelope(x, y, 3.0)
    assert (left, right, left_int) == (pytest.approx(1.125), pytest.approx(3.6), 3.0)


def test_envelope_first_point_above_height():
    left, right, left_int = envelope(np.arange(4.0), np.array([5, 10, 6, 0.0]), 3.0)
    assert (left, right, left_int) == (0.0, pytest.approx(2.5), 5.0)


def test_envelope_to_the_last_point():
    left, right, _ = envelope(np.arange(4.0), np.array([0, 4, 9, 8.0]), 3.0)
    assert right == 3.0 and left == pytest.approx(0.75)


@pytest.mark.parametrize("height", [0.0, -1.0, 10.0])
def test_envelope_without_a_crossing(height):
    assert all(np.isnan(envelope(np.arange(3.0), np.array([1.0, 5.0, 2.0]), height)))


def test_envelope_columns_follow_the_reference_formulas():
    _, p = pick({0: 1000.0, 1: 800.0, 2: 500.0, 3: 90.0})
    ymax = p.intensity.max()
    left, right, left_int = envelope(p.mz, p.intensity, ymax * 0.1)
    assert p.env_width == pytest.approx(2 * (right - left))
    assert p.env_symm == pytest.approx(2.0 - (ymax - left_int) / ymax)
    assert p.env_symm == pytest.approx(2.0)  # the first bin is the maximum: left edge = its height
    _, q = pick({0: 120.0, 1: 800.0, 2: 1000.0, 3: 400.0})  # first bin above the 10% height (100)
    assert q.env_symm == pytest.approx(2.0 - (1000.0 - 120.0) / 1000.0)
    _, r = pick({0: 50.0, 1: 800.0, 2: 1000.0})  # first bin below it: edge interpolated
    assert r.env_symm == pytest.approx(1.1)


# --- configuration --------------------------------------------------------------------------

def test_config_round_trip_and_json():
    cfg = PickerConfig(zero_filling=3, noise=50.0, nterm_subtract=2,
                       amide_rules=chem.AmideRules({"c": 1}))
    data = json.loads(json.dumps(cfg.to_dict()))
    assert PickerConfig.from_dict(data) == cfg
    assert PickerConfig.from_dict({}) == PickerConfig()


@pytest.mark.parametrize(
    "kwargs",
    [{"resolution_ppm": 0}, {"zero_filling": -1}, {"zero_filling": 1.5}, {"noise": -1},
     {"noise_percent": -1}, {"env_threshold": 0}, {"env_threshold": 1.5},
     {"nterm_subtract": -1}, {"shoulder_min_points": 0}],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        PickerConfig(**kwargs)


def test_config_unknown_key():
    with pytest.raises(chem.ChemistryError, match="unknown picker setting"):
        PickerConfig.from_dict({"zero_fill": 3})


# --- pick_spectra and BinnedSet -------------------------------------------------------------

def make_set(times=(0.0, 60.0, 60.0, 600.0), modification=""):
    plan = plan_bins(PEPTIDE, 2, modification=modification, config=CFG)
    entry = PeptideEntry("f", "S", PEPTIDE, "0001-0010", 1, 10, 2, modification=modification)
    spectra = []
    for k, t in enumerate(times):
        mz, y = sticks(plan, {0: 1000.0, 1: 500.0} if t < 600 else {})
        spectra.append(Spectrum(mz, y, entry, t, 1 + (k == 2), data_id=0, spectrum_id=f"p:s{k}"))
    return SpectrumSet.from_spectra(spectra)


def test_pick_spectra_builds_a_binned_set():
    s = make_set()
    b = pick_spectra(s, CFG)
    assert isinstance(b, BinnedSet) and len(b) == 4
    assert b.offsets[-1] == len(b.mz) == len(b.intensity) == len(b.n_deut)
    mz, y, nd = b.spectrum(1)
    assert y[0] == 1000.0 and nd[0] == 0 and len(mz) == 1 + 1 + 3
    assert list(b.spectrum_id) == ["p:s0", "p:s1", "p:s2", "p:s3"]
    assert list(b.empty) == [False, False, False, True]
    assert list(b.no_signal) == [False, False, False, True]
    assert b.max_namides[0] == chem.count_amides(PEPTIDE)
    assert b.find("p:s2") == 2
    with pytest.raises(KeyError):
        b.find("p:nope")


def test_pick_spectra_matches_pick_spectrum():
    s = make_set()
    b = pick_spectra(s, CFG)
    for i in range(len(s)):
        mz, y = s.spectrum(i)
        p = pick_spectrum(mz, y, PEPTIDE, 2, config=CFG)
        bm, by, _ = b.spectrum(i)
        np.testing.assert_array_equal(bm, p.mz)
        np.testing.assert_array_equal(by, p.intensity)
        assert b.env_width[i] == p.env_width or np.isnan(p.env_width)


def test_pick_spectra_uses_each_spectrums_modification():
    plain = pick_spectra(make_set(), CFG)
    ion_c = pick_spectra(make_set(modification="ion_type:c"), CFG)
    assert ion_c.mz[0] == chem.mz(PEPTIDE, 2, "c") != plain.mz[0]
    assert list(ion_c.modification) == ["ion_type:c"] * 4


def test_select_drops_flagged_spectra_and_time_idx_is_rederived():
    b = pick_spectra(make_set(), CFG)
    assert list(b.time_idx()) == [0, 1, 1, 2]
    kept = b.select(~b.empty)
    assert len(kept) == 3 and list(kept.spectrum_id) == ["p:s0", "p:s1", "p:s2"]
    assert kept.offsets[-1] == len(kept.mz) and list(kept.time_idx()) == [0, 1, 1]
    np.testing.assert_array_equal(kept.spectrum(1)[1], b.spectrum(1)[1])
    only_late = b.select(np.array([False, True, True, True]))
    assert list(only_late.time_idx()) == [0, 0, 1]
    with pytest.raises(ValueError, match="one element per spectrum"):
        b.select(np.array([True, False]))


def test_pick_spectra_progress_cancel_and_empty():
    s = make_set()
    calls = []
    pick_spectra(s, CFG, progress=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]
    flags = iter([False, False, True])
    with pytest.raises(Cancelled):
        pick_spectra(s, CFG, cancel=lambda: next(flags))
    none = pick_spectra(SpectrumSet.from_spectra([]), CFG)
    assert len(none) == 0 and len(none.mz) == 0 and len(none.time_idx()) == 0


def test_spectrum_set_select():
    s = make_set()
    sub = s.select(np.array([True, False, True, False]))
    assert len(sub) == 2 and list(sub.spectrum_id) == ["p:s0", "p:s2"]
    np.testing.assert_array_equal(sub.spectrum(1)[0], s.spectrum(2)[0])
    assert sub.offsets[-1] == len(sub.mz) and set(sub.modification) == {""}
    with pytest.raises(ValueError):
        s.select(np.array([True]))


# --- pyHXExpress output as the oracle -------------------------------------------------------

def check_against_reference(p, ref_mz, ref_y, ref_env_width, ref_env_symm, ref_namides):
    """The relations between hxex and pyHXExpress output for one spectrum."""
    n = min(len(p.mz), len(ref_mz))
    np.testing.assert_array_equal(p.mz[:n], ref_mz[:n])  # the bins both picked are identical
    np.testing.assert_array_equal(p.intensity[:n], ref_y[:n])
    assert p.max_namides == ref_namides
    if len(ref_mz) > n:  # pyHXExpress kept more bins: all of them below the threshold
        assert np.all(ref_y[n:] < p.threshold)
    if len(p.mz) == len(ref_mz):
        assert p.env_width == pytest.approx(ref_env_width, rel=1e-9)
        assert p.env_symm == pytest.approx(ref_env_symm, rel=1e-9)


def test_golden_spectra_match_pyhxexpress():
    z = np.load(Path(__file__).parent / "golden" / "picker_reference.npz")
    seen = defaultdict(int)
    for i, cls in enumerate(z["classes"]):
        a, b = z["raw_offsets"][i], z["raw_offsets"][i + 1]
        p = pick_spectrum(
            z["raw_mz"][a:b], z["raw_intensity"][a:b], str(z["peptide"][i]), int(z["charge"][i]),
            config=GOLDEN_CFG,
        )
        seen[str(cls)] += 1
        if not z["in_reference"][i]:
            assert cls == "dropped_by_reference" and p.empty  # pyHXExpress drops it; hxex flags it
            continue
        r0, r1 = z["ref_offsets"][i], z["ref_offsets"][i + 1]
        check_against_reference(
            p, z["ref_mz"][r0:r1], z["ref_Intensity"][r0:r1], z["ref_env_width"][i],
            z["ref_env_symm"][i], int(z["ref_max_namides"][i]),
        )
        extra = p.intensity[r1 - r0:]
        if cls == "same_length":
            assert len(p.mz) == r1 - r0
        elif cls == "reference_stopped_early":
            assert len(extra) > 0 and np.all(extra < p.threshold)
        elif cls == "reference_missed_signal":
            assert (extra >= p.threshold).any()
        elif cls == "reference_missed_major_signal":
            assert extra.max() > 0.5 * p.intensity.max()  # e.g. the main population after a valley
        elif cls == "reference_longer":
            assert len(p.mz) < r1 - r0
    assert len(seen) >= 5  # the fixture covers the cases that matter


ORACLE = os.environ.get("HXEX_PICKER_ORACLE")


@pytest.mark.skipif(not ORACLE, reason="HXEX_PICKER_ORACLE not set")
def test_whole_run_matches_pyhxexpress():
    run = Path(ORACLE)
    raw_zip = next(run.glob("allrawdata_*.zip"))
    ref_csv = next(run.glob("alldeutdata_*.csv"))
    with zipfile.ZipFile(raw_zip) as zf:
        text = zf.read(zf.namelist()[0]).decode()
    raw, meta = defaultdict(list), {}
    for r in csv.DictReader(io.StringIO(text)):
        k = (r["file"], r["time"], r["rep"], r["charge"])
        raw[k].append((float(r["mz"]), float(r["Intensity"])))
        meta[k] = r
    ref = defaultdict(list)
    with ref_csv.open() as fh:
        for r in csv.DictReader(fh):
            ref[(r["file"], r["time"], r["rep"], r["charge"])].append(r)
    dropped = 0
    for k, pts in raw.items():
        a = np.array(sorted(pts))
        m = meta[k]
        charge = int(float(m["charge"]))
        p = pick_spectrum(a[:, 0], a[:, 1], m["peptide"], charge, config=GOLDEN_CFG)
        if k not in ref:
            assert p.empty
            dropped += 1
            continue
        rows = ref[k]
        check_against_reference(
            p, np.array([float(r["mz"]) for r in rows]),
            np.array([float(r["Intensity"]) for r in rows]),
            float(rows[0]["env_width"]), float(rows[0]["env_symm"]), int(rows[0]["max_namides"]),
        )
    assert dropped <= 5
