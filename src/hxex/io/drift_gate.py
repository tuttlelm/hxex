# SPDX-License-Identifier: GPL-3.0-or-later
"""Drift ranges of peptide ions in scans written with their ion mobility bins (PLAN Next steps
22).

Such scans (msconvert's ``--combineIonMobilitySpectra``, Waters HDMS data) hold every drift
bin, so a spectrum summed over all of them also holds the ions at a nearby m/z that drift at
another time. ``[input.scans] drift_gate = "auto"`` keeps each peptide ion's own drift range:

- the range is measured once per peptide ion (peptide, modification, residues and charge), on
  its undeuterated controls (every sample's): the intensity within the picker's window of its
  natural isotope peaks (clusters of at least :data:`MIN_CLUSTER` of the tallest), per drift
  time, summed over each control's scans (:class:`DriftProfiles`);
- the range is the profile's apex and the drift times around it down to :data:`SHARE` of the
  apex (:func:`find_range`), and it is applied in every file: deuteration does not move an ion
  in drift time;
- when a second peak outside that range reaches :data:`SECOND_PEAK` of the apex, the ion's own
  isotope spacing decides between the two: the intensity on its isotope peaks against that at
  the midpoints between them (another charge state at the same m/z, which drifts at another
  time, puts its peaks between the ion's too); the peak whose intensity sits on the ion's
  peaks by :data:`ON_SHARE_MARGIN` more is taken, whichever is taller (:class:`DriftProfiles`);
- the profile is measured on m/z corrected by each undeuterated file's own offset (measured on
  its peptide spectra summed over every drift time, as ``mz_calibration`` measures it later on
  the gated spectra), so that its peaks are where the ion's are; and every drift peak of at least
  :data:`MIN_PEAK` of the apex gets the m/z position of its monoisotopic and M+1 intensity
  (ppm from the prediction): when the chosen range sits off the ion's m/z (beyond
  :data:`OFF_SD` of the run's position scatter, the robust sd over the ions with one drift peak)
  and another drift peak sits on it (within :data:`ON_SD`, closer by :data:`GAP_SD`), that
  peak's range is named (``better``) and the series is flagged ``drift_unclear``: another ion
  that drifts at another time and lies a few ppm away (closer than the peaks' width) fills the
  chosen range (:class:`DriftProfiles`);
- a peptide table row's own ``drift_start``, ``drift_end`` (ms) take the place of the measured
  range (:func:`hxex.spectra.entry_drift_window`).

Drift times are those the file records (ms); no instrument setting is needed. A peptide ion
without an undeuterated control with signal keeps every drift time, and so does a fragment
(its drift is not its own: fragments formed after the drift cell take their precursor's).
The series of such an ion, of one whose profile has a second peak of at least
:data:`SECOND_PEAK` of the apex outside its range that the isotope spacing does not tell apart,
and of one with a better drift peak on its m/z, is flagged ``drift_unclear``.

Measured on an intact-peptide dataset of Q-TOF HDMS files (HISTORY 304): HDExaminer's
gates are 5 drift bins (of 200), the same peptide ion within one bin across files and no shift
with deuteration; the ranges found here (median 4 bins) gave the same K as HDExaminer's gates in
97 % of spectra, centroids within 0.009 D (median), and a replicate scatter as small. The m/z
position test (HISTORY 376): on that dataset (341 peptide ions) and a pooled dataset of the same
kind (319), 4 ions had their chosen range off their m/z with another drift peak on it, in two of
them because the profile had been measured on m/z not yet corrected (files 16 to 38 ppm off); the
position scatter of single-peak ions was 2.3-2.4 ppm (robust sd). The share was 0.1 until then:
on the corrected profiles some edge bins sat just above it (heights 0.100-0.11 of the apex), and
the labelled spectra of the ranges they widened scattered twice as much; at 0.2 (no range wider
than at 0.1) both datasets gained on every yardstick but the pooled one's fully deuterated
charge pairs (undeuterated |K = 1 mean| q90 0.096 -> 0.082 and 0.176 -> 0.160 D, labelled
replicate sd q90 0.190 -> 0.178 and 0.910 -> 0.806, labelled charge pairs > 0.5 D 5.7 -> 5.5 and
36.1 -> 33.8 %), keeping a median 91 % (benchmark) and 86 % (pooled) of the ion's drift
intensity.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

import numpy as np

SHARE = 0.2  # the range: the drift times around the apex down to this share of it
SECOND_PEAK = 0.5  # a peak outside the range this high (share of the apex): drift_unclear
MIN_CLUSTER = 0.05  # natural isotope clusters at least this share of the tallest make a profile
ON_SHARE_MARGIN = 0.2  # a second drift peak is told apart when its on-peak shares differ this much
_DECIMALS = 6  # drift times (ms) compared at this precision (the same bins in every file)
MIN_PEAK = 0.1  # drift peaks at least this share of the apex get an m/z position
ON_SD = 2.5  # a drift peak within this many position sd of the ion's m/z sits on it
OFF_SD = 4.0  # a chosen range beyond this many sd sits off it
GAP_SD = 3.0  # and the other peak is closer by at least this many sd
MIN_POSITION_IONS = 20  # ions with one drift peak needed to measure the run's position scatter
MIN_POSITION_SD = 1.0  # ppm: the position scatter is taken as at least this


def drift_key(entry) -> tuple:
    """The peptide ion a drift range belongs to: one range for every sample and file."""
    return (entry.peptide, entry.modification, int(entry.start_seq), int(entry.end_seq),
            int(entry.charge))


def drift_times(values: np.ndarray) -> np.ndarray:
    """Drift times (ms) as they are compared (rounded, so that equal bins match)."""
    return np.round(np.asarray(values, dtype=np.float64), _DECIMALS)


def isotope_peaks(plan) -> np.ndarray:
    """The m/z of the undeuterated ion's natural isotope peaks that make its drift profile (a
    :class:`hxex.picking.BinPlan` with its clusters)."""
    shape, offsets = plan.clusters
    shape, offsets = np.asarray(shape, dtype=float), np.asarray(offsets, dtype=float)
    keep = shape >= MIN_CLUSTER * shape.max()
    return plan.undeut_mz + offsets[keep] / plan.charge


def scan_profile(scans, peaks: np.ndarray, half: np.ndarray, offset_ppm: float = 0.0,
                 positions: bool = False):
    """The drift profile of ``scans`` (with :attr:`hxex.io.mzml.Scan.binned`) at ``peaks``
    (m/z, each +- its ``half``; moved by the file's m/z offset ``offset_ppm`` when it is known):
    the distinct drift times, the intensity at each, and the intensity at the midpoints between
    neighbouring peaks (each +- the larger half-width of the two), summed over the scans; None
    when no scan has drift times. With ``positions``, also per drift time the intensity within
    the half-widths of the monoisotopic and M+1 peaks and that intensity times its distance from
    them (ppm): their sum over drift times divided by the intensity is the m/z position."""
    parts = [s.binned for s in scans if s.binned is not None]
    if not parts:
        return None
    mz = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    d = drift_times(np.concatenate([p[2] for p in parts]))
    peaks = np.atleast_1d(np.asarray(peaks, dtype=float))
    if np.isfinite(offset_ppm) and offset_ppm:
        peaks = peaks * (1.0 + offset_ppm * 1e-6)
    half = np.broadcast_to(np.asarray(half, dtype=float), peaks.shape)
    near = np.zeros(len(mz), dtype=bool)
    for p, h in zip(peaks, half, strict=True):
        near |= np.abs(mz - p) <= h
    between = np.zeros(len(mz), dtype=bool)
    for p, h in zip(0.5 * (peaks[:-1] + peaks[1:]), np.maximum(half[:-1], half[1:]), strict=True):
        between |= np.abs(mz - p) <= h
    between &= ~near
    values, where = np.unique(d, return_inverse=True)
    n = len(values)
    out = (values, np.bincount(where, weights=np.where(near, y, 0.0), minlength=n),
           np.bincount(where, weights=np.where(between, y, 0.0), minlength=n))
    if not positions:
        return out
    pos = np.zeros(len(mz), dtype=bool)
    ppm = np.zeros(len(mz))
    for p, h in zip(peaks[:2], half[:2], strict=True):
        sel = np.abs(mz - p) <= h
        pos |= sel
        ppm[sel] = 1e6 * (mz[sel] - p) / p
    return out + (np.bincount(where, weights=np.where(pos, y, 0.0), minlength=n),
                  np.bincount(where, weights=np.where(pos, y * ppm, 0.0), minlength=n))


@dataclass(frozen=True)
class DriftRange:
    """A peptide ion's drift range (ms): ``start`` and ``end``, the profile's ``apex``, the
    tallest peak outside the range as a share of the apex (``second``; 0 without one), the
    number of undeuterated files whose profiles made it, the share of the range's intensity on
    the ion's isotope peaks rather than between them (``on_share``; NaN when not measured),
    whether the isotope spacing chose it over a second peak (``decided``), the m/z position of
    the range's monoisotopic and M+1 intensity (``offset_ppm``, ppm from the prediction; NaN when
    not measured), the drift peaks of at least :data:`MIN_PEAK` of the apex (``peaks``: drift
    time, height as a share of the apex, m/z position), and a drift range on the ion's m/z when
    the chosen one sits off it (``better``: start, end, apex; None otherwise)."""

    start: float
    end: float
    apex: float
    second: float = 0.0
    n_files: int = 0
    on_share: float = float("nan")
    decided: bool = False
    offset_ppm: float = float("nan")
    peaks: tuple = ()
    better: tuple | None = None

    @property
    def unclear(self) -> bool:
        return (self.second >= SECOND_PEAK and not self.decided) or self.better is not None


def find_range(drifts: np.ndarray, profile: np.ndarray, share: float = SHARE,
               at: int | None = None) -> DriftRange | None:
    """The drift range of a profile (drift times in increasing order, intensity at each): the
    apex (or the peak at position ``at``) and the drift times either side of it down to
    ``share`` of it; None without signal."""
    profile = np.asarray(profile, dtype=float)
    if not len(profile) or not profile.max() > 0:
        return None
    apex = int(np.argmax(profile)) if at is None else int(at)
    top = profile[apex]
    lo = hi = apex
    while lo > 0 and profile[lo - 1] >= share * top:
        lo -= 1
    while hi < len(profile) - 1 and profile[hi + 1] >= share * top:
        hi += 1
    peaks = [j for j in range(len(profile)) if not lo <= j <= hi and profile[j] > 0
             and (j == 0 or profile[j] >= profile[j - 1])
             and (j == len(profile) - 1 or profile[j] >= profile[j + 1])]
    second = max((profile[j] / top for j in peaks), default=0.0)
    return DriftRange(float(drifts[lo]), float(drifts[hi]), float(drifts[apex]), float(second))


def _second_peak(profile: np.ndarray, found: DriftRange, drifts: np.ndarray) -> int:
    """The position of the tallest local maximum outside ``found``'s range."""
    outside = (drifts < found.start) | (drifts > found.end)
    n = len(profile)
    peaks = [j for j in np.flatnonzero(outside) if profile[j] > 0
             and (j == 0 or profile[j] >= profile[j - 1])
             and (j == n - 1 or profile[j] >= profile[j + 1])]
    return int(max(peaks, key=lambda j: profile[j]))


def _on_share(drifts, on, between, r: DriftRange) -> float:
    inside = (drifts >= r.start) & (drifts <= r.end)
    total = float(on[inside].sum() + between[inside].sum())
    return float(on[inside].sum()) / total if total > 0 else float("nan")


def choose_range(drifts: np.ndarray, on: np.ndarray, between: np.ndarray | None = None,
                 share: float = SHARE) -> DriftRange | None:
    """The drift range of a profile (:func:`find_range`) and, when a second peak reaches
    :data:`SECOND_PEAK` of the apex, the one of the two whose intensity sits on the ion's
    isotope peaks (``on``) rather than between them (``between``) by at least
    :data:`ON_SHARE_MARGIN`; else the taller, left unclear."""
    drifts = np.asarray(drifts, dtype=float)
    on = np.asarray(on, dtype=float)
    found = find_range(drifts, on, share)
    if found is None or between is None:
        return found
    between = np.asarray(between, dtype=float)
    first = replace(found, on_share=_on_share(drifts, on, between, found))
    if not found.second >= SECOND_PEAK:
        return first
    other = find_range(drifts, on, share, at=_second_peak(on, found, drifts))
    other = replace(other, on_share=_on_share(drifts, on, between, other))
    if not (np.isfinite(first.on_share) and np.isfinite(other.on_share)):
        return first
    if first.on_share >= other.on_share + ON_SHARE_MARGIN:
        return replace(first, decided=True)
    if other.on_share >= first.on_share + ON_SHARE_MARGIN:
        return replace(other, decided=True)
    return first


def _peak_span(profile: np.ndarray, j: int, share: float = SHARE) -> tuple[int, int]:
    """The drift positions either side of the peak at ``j`` down to ``share`` of it, stopping
    before a valley (where the profile rises again away from the peak): a smaller peak's own
    range, without its neighbour's tail."""
    top = profile[j]
    lo = hi = j
    while lo > 0 and share * top <= profile[lo - 1] <= profile[lo]:
        lo -= 1
    while hi < len(profile) - 1 and share * top <= profile[hi + 1] <= profile[hi]:
        hi += 1
    # the valley itself (the profile rises beyond it) holds both peaks' tails: left out
    if hi > j and hi < len(profile) - 1 and profile[hi + 1] > profile[hi]:
        hi -= 1
    if lo < j and lo > 0 and profile[lo - 1] > profile[lo]:
        lo += 1
    return lo, hi


def drift_peaks(drifts: np.ndarray, on: np.ndarray, weight: np.ndarray,
                weighted_ppm: np.ndarray, min_peak: float = MIN_PEAK) -> list[tuple]:
    """The local maxima of a profile of at least ``min_peak`` of its apex, each as (position,
    drift time, height as a share of the apex, m/z position (ppm) over the peak's drift time and
    its two neighbours)."""
    top = float(on.max())
    n = len(on)
    out = []
    for j in range(n):
        if on[j] >= min_peak * top and (j == 0 or on[j] >= on[j - 1]) and (
                j == n - 1 or on[j] >= on[j + 1]):
            sel = slice(max(0, j - 1), j + 2)
            w = float(weight[sel].sum())
            out.append((j, float(drifts[j]), float(on[j] / top),
                        float(weighted_ppm[sel].sum()) / w if w > 0 else float("nan")))
    return out


class DriftProfiles:
    """Drift profiles summed per peptide ion over the undeuterated controls' files."""

    def __init__(self):
        self._profiles: dict[tuple, dict[float, list[float]]] = {}
        self._files: dict[tuple, set] = {}
        self.position: tuple | None = None  # (centre, sd, ions) after ranges(), when measured

    def add(self, key: tuple, drifts: np.ndarray, intensity: np.ndarray,
            between: np.ndarray | None = None, weight: np.ndarray | None = None,
            weighted_ppm: np.ndarray | None = None, *, file: str) -> None:
        """Add one file's profile: intensity on the ion's isotope peaks per drift time, between
        them, and the monoisotopic and M+1 intensity with its distance from them
        (:func:`scan_profile` with ``positions``)."""
        summed = self._profiles.setdefault(key, {})
        n = len(intensity)
        between = np.zeros(n) if between is None else between
        weight = np.zeros(n) if weight is None else weight
        weighted_ppm = np.zeros(n) if weighted_ppm is None else weighted_ppm
        for d, v, b, w, wp in zip(drifts.tolist(), intensity.tolist(), between.tolist(),
                                  weight.tolist(), weighted_ppm.tolist(), strict=True):
            cell = summed.setdefault(d, [0.0, 0.0, 0.0, 0.0])
            cell[0] += v
            cell[1] += b
            cell[2] += w
            cell[3] += wp
        self._files.setdefault(key, set()).add(file)

    def ranges(self, share: float = SHARE) -> dict[tuple, DriftRange]:
        """The range of every peptide ion whose profile has signal (:func:`choose_range`), with
        the m/z positions of its drift peaks and, where the chosen range sits off the ion's m/z
        and another drift peak on it, that peak's range (``better``; see the module notes)."""
        out = {}
        arrays = {}
        for key, summed in self._profiles.items():
            drifts = np.array(sorted(summed))
            cells = np.array([summed[d] for d in drifts])
            found = choose_range(drifts, cells[:, 0], cells[:, 1], share)
            if found is None:
                continue
            peaks = drift_peaks(drifts, cells[:, 0], cells[:, 2], cells[:, 3])
            inside = (drifts >= found.start) & (drifts <= found.end)
            w = float(cells[inside, 2].sum())
            offset = float(cells[inside, 3].sum()) / w if w > 0 else float("nan")
            out[key] = replace(found, n_files=len(self._files[key]), offset_ppm=offset,
                               peaks=tuple((d, round(h, 4), round(o, 2)) for _, d, h, o in peaks))
            arrays[key] = (drifts, cells[:, 0], peaks)
        # the run's position scatter: ions with a single drift peak
        single = np.array([r.offset_ppm for r in out.values()
                           if len(r.peaks) == 1 and np.isfinite(r.offset_ppm)])
        if len(single) < MIN_POSITION_IONS:
            return out
        centre = float(np.median(single))
        sd = max(1.4826 * float(np.median(np.abs(single - centre))), MIN_POSITION_SD)
        for key, r in out.items():
            c = abs(r.offset_ppm - centre)
            if not c > OFF_SD * sd:
                continue
            drifts, on, peaks = arrays[key]
            others = [(j, abs(o - centre)) for j, d, _, o in peaks
                      if not r.start <= d <= r.end and np.isfinite(o)]
            if not others:
                continue
            j, b = min(others, key=lambda x: x[1])
            if b <= ON_SD * sd and c - b > GAP_SD * sd:
                lo, hi = _peak_span(on, j, share)
                out[key] = replace(r, better=(float(drifts[lo]), float(drifts[hi]),
                                              float(drifts[j])))
        self.position = (centre, sd, len(single))
        return out


def drift_record(ranges: Mapping[tuple, DriftRange], unclear: Iterable[int],
                 position: tuple | None = None, offsets: Mapping[str, float] | None = None
                 ) -> dict:
    """The read record's ``drift_gate`` entry: the share, every range found, the series
    flagged ``drift_unclear``, the run's position scatter (``position``: centre and sd in ppm,
    ions measured; None when too few ions have one drift peak) and the undeuterated files'
    m/z offsets the profiles were measured with (``offsets``, ppm by file)."""
    return {
        "share": SHARE,
        "second_peak": SECOND_PEAK,
        "position": None if position is None else {
            "centre_ppm": round(position[0], 2), "sd_ppm": round(position[1], 2),
            "ions": int(position[2])},
        "profile_offsets": {k: (None if not np.isfinite(v) else round(v, 2))
                            for k, v in (offsets or {}).items()},
        "ranges": [{"peptide": k[0], "modification": k[1], "start_seq": k[2], "end_seq": k[3],
                    "charge": k[4], "drift_start": r.start, "drift_end": r.end,
                    "apex": r.apex, "second": round(r.second, 4), "n_files": r.n_files,
                    "on_share": None if not np.isfinite(r.on_share) else round(r.on_share, 4),
                    "decided": r.decided,
                    "offset_ppm": None if not np.isfinite(r.offset_ppm) else round(r.offset_ppm, 2),
                    "peaks": [list(p) for p in r.peaks],
                    "better": None if r.better is None else list(r.better)}
                   for k, r in sorted(ranges.items())],
        "unclear": sorted(int(d) for d in unclear),
    }


def unclear_series(record: Mapping | None) -> list[int]:
    """The series (data_id) flagged ``drift_unclear`` in a read record's ``drift_gate`` entry."""
    if not record:
        return []
    return [int(d) for d in record.get("unclear", ())]
