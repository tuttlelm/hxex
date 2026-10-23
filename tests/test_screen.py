# SPDX-License-Identifier: GPL-3.0-or-later
"""Spectrum quality screens (hxex.picking.screen): synthetic profile and centroided spectra
with one kind of interference each. The scores were checked against the prototype on a
26,000-spectrum benchmark (identical; HISTORY 151); the private case below runs when
HXEX_SCREEN_BUNDLE points to that benchmark's bundle and HXEX_SCREEN_CASE names the peptide
range of its confirmed case (charge 2)."""

import dataclasses
import os

import numpy as np
import pytest
from scipy.stats import binom

from hxex.picking import BinnedSet, PickerConfig, pick_spectra, pick_spectrum, plan_bins
from hxex.picking.pick import composition_mz
from hxex.picking.screen import SCREEN_SCORES, STEP
from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet

PEPTIDE = "RDKVQKEYAL"
Z = 2
CFG = PickerConfig()  # the composition ladder, a 50 ppm window
PLAN = plan_bins(PEPTIDE, Z, config=CFG)
S = STEP / Z  # one isotope step in m/z
TOP = 1e4


def envelope(level: float, height: float = TOP) -> tuple[np.ndarray, np.ndarray]:
    """Peak positions and heights of the peptide at ``level`` deuterons (one population)."""
    shape, sites = PLAN.clusters[0], PLAN.n_sites
    h = np.convolve(binom.pmf(np.arange(sites + 1), sites, level / sites), shape)
    return composition_mz(PLAN, level)[:len(h)], height * h / h.max()


def profile(*peaks, resolving_power=20000.0, baseline=1.0):
    """A profile spectrum: Gaussian peaks (FWHM m/z / R) on a flat baseline, from three steps
    below the monoisotopic m/z to a step beyond the last bin, a point every 4 ppm."""
    centres = np.concatenate([p[0] for p in peaks])
    heights = np.concatenate([p[1] for p in peaks])
    lo, hi = PLAN.undeut_mz - 3 * S, PLAN.pred_mz[-1] + S
    mz = np.arange(lo, hi, lo * 4e-6)
    sigma = centres / resolving_power / 2.3548
    y = baseline + (heights * np.exp(-0.5 * ((mz[:, None] - centres) / sigma) ** 2)).sum(axis=1)
    return mz, y


def scores(mz, y) -> dict[str, float]:
    p = pick_spectrum(mz, y, PEPTIDE, Z, config=CFG)
    names = ("screen_below_mono", "screen_between", "screen_between_separate",
             "screen_beyond_reach")
    return dict(zip(names, p.screen, strict=True))


def test_a_clean_spectrum_scores_nothing():
    s = scores(*profile(envelope(3.0)))
    assert s["screen_below_mono"] == 0.0
    assert s["screen_between"] == s["screen_between_separate"] == 0.0
    assert s["screen_beyond_reach"] < 0.01  # the baseline picked in the zero filling


def test_a_species_below_the_monoisotopic_mz():
    pos, h = envelope(0.0)
    other = (pos[:4] - 2 * S, 0.6 * h[:4])  # an undeuterated species two steps lower
    s = scores(*profile(envelope(0.0), other))
    assert s["screen_below_mono"] > 0.10
    # a lone peak one step below continues with the peptide's own shape
    lone = scores(*profile(envelope(0.0), (pos[:1] - S, np.array([0.5 * TOP]))))
    assert 0.05 < lone["screen_below_mono"] < s["screen_below_mono"]


def test_below_mono_is_not_assessed_without_the_raw_spectrum_below():
    mz, y = profile(envelope(3.0))
    keep = mz > PLAN.undeut_mz - S  # starts one step below: not far enough
    assert np.isnan(scores(mz[keep], y[keep])["screen_below_mono"])


def test_peaks_between_positions_touching_and_separate():
    pos, h = envelope(1.0)
    touching = (pos[1:3] + 0.5 * S, np.full(2, 0.5 * TOP))
    s = scores(*profile(envelope(1.0), touching))
    assert s["screen_between"] > 0.2 and s["screen_between_separate"] == 0.0
    # a separate cluster above the envelope, still inside the reachable range
    far = (composition_mz(PLAN, 1.0)[9:11] + 0.5 * S, np.full(2, 0.5 * TOP))
    s = scores(*profile(envelope(1.0), far))
    assert s["screen_between_separate"] > 0.2 and s["screen_between"] == 0.0
    # one peak between positions is not enough
    one = (pos[1:2] + 0.5 * S, np.array([0.5 * TOP]))
    assert scores(*profile(envelope(1.0), one))["screen_between"] == 0.0


def test_intensity_beyond_the_reachable_range():
    last = len(PLAN.pred_mz) - 1
    beyond = (composition_mz(PLAN, 3.0)[last:last + 1], np.array([0.3 * TOP]))
    s = scores(*profile(envelope(3.0), beyond))
    assert s["screen_beyond_reach"] > 0.05


def test_centroided_spectra():
    pos, h = envelope(3.0)
    s = scores(pos, h)
    assert s["screen_between"] == 0.0 and s["screen_beyond_reach"] == 0.0
    assert np.isnan(s["screen_below_mono"])  # nothing below the monoisotopic stick
    mz = np.concatenate([pos, pos[2:4] + 0.5 * S])
    y = np.concatenate([h, np.full(2, 0.5 * h.max())])
    order = np.argsort(mz)
    assert scores(mz[order], y[order])["screen_between"] > 0.1


def test_an_empty_spectrum_is_not_assessed():
    mz, y = profile(envelope(3.0))
    assert all(np.isnan(v) for v in scores(mz, np.zeros_like(y)).values())


def spectrum_set(shifted_ppm: float, calibration_ppm: float = 0.0) -> SpectrumSet:
    """An undeuterated control and a labelled spectrum whose three lowest peaks are shifted;
    ``calibration_ppm`` shifts both spectra throughout."""
    entry = PeptideEntry("f", "S", PEPTIDE, "0001-0010", 1, 10, Z)
    out = []
    for level, time, role, shift in ((0.0, np.nan, "undeut", 0.0), (2.0, 60.0, "", shifted_ppm)):
        pos, h = envelope(level)
        pos = pos * (1 + calibration_ppm / 1e6)
        pos[:3] *= 1 + shift / 1e6
        out.append(Spectrum(*profile((pos, h)), entry, time, 1, spectrum_id=f"p:{level}",
                            role=role))
    return SpectrumSet.from_spectra(out)


def test_peaks_off_their_predicted_position():
    clean = pick_spectra(spectrum_set(0.0), CFG)
    assert list(clean.screen_off_position) == [0.0, 0.0]
    # 35 ppm: inside the 50 ppm window, beyond half of it
    b = pick_spectra(spectrum_set(35.0), CFG)
    _, y, nd = b.spectrum(1)
    assert b.screen_off_position[0] == 0.0
    assert b.screen_off_position[1] == pytest.approx(y[:3].sum() / y.sum())
    # a shift of the whole series, control included, is its calibration: taken out first
    calibrated = pick_spectra(spectrum_set(0.0, calibration_ppm=35.0), CFG)
    assert list(calibrated.screen_off_position) == [0.0, 0.0]


def test_the_scores_follow_the_set_and_a_set_picked_elsewhere_has_none():
    b = pick_spectra(spectrum_set(35.0), CFG)
    one = b.select(np.array([False, True]))
    assert one.screen_off_position[0] == b.screen_off_position[1]
    elsewhere = dataclasses.replace(b, **{name: None for name in SCREEN_SCORES})
    assert isinstance(elsewhere, BinnedSet)
    assert all(np.isnan(getattr(elsewhere, name)).all() for name in SCREEN_SCORES)


@pytest.mark.skipif(not (os.environ.get("HXEX_SCREEN_BUNDLE")
                         and os.environ.get("HXEX_SCREEN_CASE")),
                    reason="HXEX_SCREEN_BUNDLE and HXEX_SCREEN_CASE not set")
def test_a_contaminated_peptide_on_real_data():
    """The benchmark's confirmed case: a second species of the same charge below the
    monoisotopic m/z of one peptide, removed at the loose level in most of its spectra."""
    import hxex

    res = hxex.load_results(os.environ["HXEX_SCREEN_BUNDLE"])
    case = os.environ["HXEX_SCREEN_CASE"]
    b = res.binned
    rows = np.flatnonzero((np.asarray(b.peptide_range) == case) & (np.asarray(b.charge) == 2))
    assert len(rows)
    sp = res.loader([str(b.spectrum_id[j]) for j in rows])
    picked = pick_spectra(sp, res.settings.picker)
    below = picked.screen_below_mono
    assert np.mean(below[np.isfinite(below)] >= 0.10) > 0.5
