# SPDX-License-Identifier: GPL-3.0-or-later
"""Spectrum quality screens: fit-free scores for another species in a spectrum (HISTORY 149).

Each score is a share of intensity, 0 for a clean spectrum, computed from the raw spectrum and
its picked bins right after picking. The tolerances come from the picker's window
(:meth:`~hxex.picking.PickerConfig.half_width`), so the scores mean the same on instruments of
different resolving power.

- ``screen_below_mono`` (R1): a species below the monoisotopic m/z. The peaks one and two
  isotope steps below it (within a quarter step) are extrapolated into the three lowest bins,
  with the peptide's natural-abundance shape when the second is small, else with their own
  ratio; the score is the extrapolated intensity (at most what was picked there) over the
  picked intensity. NaN when the raw spectrum does not reach 1.25 steps below the
  monoisotopic m/z (0.75 for centroided data).
- ``screen_off_position`` (R2): picked intensity in the five lowest bins whose peak lies more
  than half the picking window (at least ``mass_accuracy_ppm``) from its predicted m/z, after
  the series' own offset (the median over its undeuterated controls of each one's
  intensity-weighted mean offset; a series without one uses the spectrum's own), over the
  picked intensity.
- ``screen_between`` (R3t) and ``screen_between_separate`` (R3s): peaks between the isotope
  positions, outside the picking window and at least a quarter step from every predicted
  position, over all peaks in the inspected range: from half a step below the monoisotopic
  m/z to half a step above the last reachable bin. ``screen_between`` counts those up to the
  top edge of the envelope (the last bin connected to the tallest reachable bin at 10 % of its
  height or more, gaps of one bin allowed) plus half a step, ``screen_between_separate`` those
  above it. Each needs two such peaks, else it is 0.
- ``screen_beyond_reach`` (R4): picked intensity beyond the reachable bins (the exchangeable
  sites plus the width of the natural-abundance envelope, less one) over the picked intensity.

Peaks count above 5 % of the tallest point in the inspected range and 5 times its baseline
(the median intensity of profile data, the lower quartile of centroided data), and in profile
data with a prominence of 3 times the baseline. Data with fewer than 4 points per isotope step
in the range are centroided: every point is a peak. A spectrum without picked intensity, or
with fewer than three raw points, is not assessed (NaN); so are spectra picked elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.signal import find_peaks

from ..spectra import UNDEUT
from .config import PickerConfig

SCREEN_SCORES = ("screen_below_mono", "screen_off_position", "screen_between",
                 "screen_between_separate", "screen_beyond_reach")
RAW_SCORES = ("screen_below_mono", "screen_between", "screen_between_separate",
              "screen_beyond_reach")  # what raw_scores returns, in its order

STEP = 1.003355  # Da, the 13C isotope step
NEAR = 0.25  # steps: a peak this close to a position belongs to it
PEAK_SHARE = 0.05  # a peak reaches this share of the tallest point ...
PEAK_BASELINE = 5.0  # ... and this many baselines
PROMINENCE_BASELINE = 3.0  # profile peaks: prominence in baselines
STICK_POINTS = 4  # fewer points per isotope step: centroided data
ENVELOPE_SHARE = 0.10  # the envelope's bins reach this share of its tallest bin
SHAPE_SHARE = 0.01  # natural-abundance peaks counted in the reachable width
LOW_BINS = 4  # screen_off_position looks at bins 0 to 4


def raw_scores(mz: np.ndarray, y: np.ndarray, pred: np.ndarray, picked: np.ndarray,
               n_deut: np.ndarray, charge: int, shape: np.ndarray, n_sites: int,
               half_width: np.ndarray) -> tuple[float, float, float, float]:
    """``screen_below_mono``, ``screen_between``, ``screen_between_separate`` and
    ``screen_beyond_reach`` of one spectrum: the raw ``mz`` (sorted) and ``y``, the picked bins'
    predicted m/z ``pred``, intensities ``picked`` and ``n_deut``, the natural-abundance
    ``shape``, the number of exchangeable sites and the picking window's half-width at each
    bin."""
    nan = float("nan")
    total = float(picked.sum())
    if not total > 0 or len(mz) < 3:
        return nan, nan, nan, nan
    shape = np.asarray(shape, dtype=np.float64)
    shape = shape / shape.max()
    reach = n_deut <= n_sites + int((shape >= SHAPE_SHARE).sum()) - 1
    if not reach.any():
        return nan, nan, nan, nan
    beyond = float(picked[~reach].sum() / total)
    step = STEP / charge
    zero = n_deut == 0
    mono = float(pred[zero][0]) if zero.any() else float(pred.min())
    lo, hi = mono - 0.5 * step, float(pred[reach].max()) + 0.5 * step
    a, b = np.searchsorted(mz, lo, side="left"), np.searchsorted(mz, hi, side="right")
    if b - a < 2 or y[a:b].max() <= 0:
        return nan, nan, nan, beyond
    seg = y[a:b]
    sticks = (b - a) / max((hi - lo) / step, 1e-9) < STICK_POINTS
    base = float(np.quantile(seg, 0.25)) if sticks else float(np.median(seg))
    below = nan
    if mz[0] <= mono - 1.25 * step or (sticks and mz[0] <= mono - 0.75 * step):
        below = _below_mono(mz, y, pred, shape, mono, step, base)
    floor = max(PEAK_SHARE * float(seg.max()), PEAK_BASELINE * base)
    if sticks:
        peaks = a + np.flatnonzero(seg >= floor)
    else:  # the range alone, as if the spectrum were zero outside it
        left, right = int(a > 0), int(b < len(y))
        padded = np.concatenate([np.zeros(left), seg, np.zeros(right)])
        found, _ = find_peaks(padded, height=floor,
                              prominence=PROMINENCE_BASELINE * base if base > 0 else None)
        peaks = a + found - left
    between = between_separate = 0.0
    if len(peaks):
        dist = np.abs(mz[peaks][:, None] - pred[None, :])
        off_grid = peaks[np.all(dist > np.maximum(half_width, NEAR * step)[None, :], axis=1)]
        edge = _envelope_top(pred, np.where(reach, picked, 0.0), reach) + 0.5 * step
        touching, separate = off_grid[mz[off_grid] <= edge], off_grid[mz[off_grid] > edge]
        all_peaks = float(y[peaks].sum())
        if len(touching) >= 2:
            between = float(y[touching].sum() / all_peaks)
        if len(separate) >= 2:
            between_separate = float(y[separate].sum() / all_peaks)
    return below, between, between_separate, beyond


def _window_max(mz: np.ndarray, y: np.ndarray, centres, half: float) -> np.ndarray:
    """The tallest raw point within ``half`` of each centre (0 where there is none)."""
    centres = np.asarray(centres, dtype=np.float64)
    lo = np.searchsorted(mz, centres - half, side="left")
    hi = np.searchsorted(mz, centres + half, side="right")
    out = np.zeros(len(centres))
    filled = hi > lo
    if filled.any():
        bounds = np.column_stack([lo[filled], hi[filled]]).ravel()
        out[filled] = np.maximum.reduceat(np.append(y, 0.0), bounds)[::2]
    return out


def _below_mono(mz, y, pred, shape, mono, step, base) -> float:
    m1, m2 = _window_max(mz, y, [mono - step, mono - 2 * step], NEAR * step)
    m1 = 0.0 if m1 < PROMINENCE_BASELINE * base else float(m1)
    m2 = 0.0 if m2 < PROMINENCE_BASELINE * base else float(m2)
    at_bins = _window_max(mz, y, pred, NEAR * step)
    lowest = at_bins[np.argsort(pred)[:3]]
    if m1 <= 0:
        extrapolated = np.zeros(3)
    elif m2 < 0.2 * m1:  # one peak below: the peptide's own shape continues it
        extrapolated = m1 * np.array([shape[q] if q < len(shape) else 0.0 for q in (1, 2, 3)])
    else:
        extrapolated = m1 * min(1.0, m1 / m2) ** np.arange(1, 4)
    return float(np.minimum(lowest, extrapolated[:len(lowest)]).sum()
                 / max(float(at_bins.sum()), 1e-12))


def _envelope_top(pred: np.ndarray, reached: np.ndarray, reach: np.ndarray) -> float:
    """The m/z of the envelope's last bin: walking up from the tallest reachable bin over bins
    at ``ENVELOPE_SHARE`` of it or more, across gaps of one bin."""
    if not reached.max() > 0:
        return float(pred[reach].max())
    top = int(np.argmax(reached))
    up = reached[top:] >= ENVELOPE_SHARE * reached.max()
    gaps = np.flatnonzero(~up[:-1] & ~up[1:])
    end = int(gaps[0]) if len(gaps) else len(up)
    return float(pred[top + np.flatnonzero(up[:end])[-1]])


def off_position_scores(binned, config: PickerConfig, resolving_power: np.ndarray,
                        resolving_power_mz: np.ndarray) -> np.ndarray:
    """``screen_off_position`` of every spectrum of a picked set (NaN without picked intensity
    or picked peak positions), with each spectrum's recorded resolving power (NaN for none)."""
    n = len(binned)
    out = np.full(n, np.nan)
    offsets = np.asarray(binned.offsets)
    pred = np.asarray(binned.mz, dtype=np.float64)
    w = np.asarray(binned.intensity, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        ppm = 1e6 * (np.asarray(binned.peak_mz, dtype=np.float64) - pred) / pred
    usable = np.isfinite(ppm) & (w > 0)
    row = np.repeat(np.arange(n), np.diff(offsets))
    weight = np.bincount(row[usable], w[usable], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        own = np.bincount(row[usable], (w * ppm)[usable], minlength=n) / weight
    data_id = np.asarray(binned.data_id)
    controls = (np.asarray(binned.role) == UNDEUT) & np.isfinite(own)
    series: Mapping[int, float] = {
        int(d): float(np.median(own[controls & (data_id == d)]))
        for d in np.unique(data_id[controls])}
    charge = np.asarray(binned.charge)
    n_deut = np.asarray(binned.n_deut)
    for i in range(n):
        sl = slice(offsets[i], offsets[i + 1])
        total = float(w[sl].sum())
        if not total > 0 or not np.isfinite(ppm[sl]).any():
            continue
        ref = series.get(int(data_id[i]), own[i] if np.isfinite(own[i]) else 0.0)
        half = config.half_width(pred[sl], int(charge[i]),
                                 resolving_power=float(resolving_power[i]),
                                 resolving_power_mz=float(resolving_power_mz[i]))
        tolerance = np.maximum(0.5e6 * half / pred[sl], float(config.mass_accuracy_ppm))
        o = ppm[sl]
        off = (n_deut[sl] <= LOW_BINS) & np.isfinite(o) & (np.abs(o - ref) > tolerance)
        out[i] = float(w[sl][off].sum() / total)
    return out
