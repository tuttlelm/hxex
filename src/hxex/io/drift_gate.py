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
- a peptide table row's own ``drift_start``, ``drift_end`` (ms) take the place of the measured
  range (:func:`hxex.spectra.entry_drift_window`).

Drift times are those the file records (ms); no instrument setting is needed. A peptide ion
without an undeuterated control with signal keeps every drift time, and so does a fragment
(its drift is not its own: fragments formed after the drift cell take their precursor's).
The series of such an ion, and of one whose profile has a second peak of at least
:data:`SECOND_PEAK` of the apex outside its range, is flagged ``drift_unclear``.

Measured on an intact-peptide dataset of Q-TOF HDMS files (HISTORY 304): HDExaminer's
gates are 5 drift bins (of 200), the same peptide ion within one bin across files and no shift
with deuteration; the ranges found here (median 4 bins) gave the same K as HDExaminer's gates in
97 % of spectra, centroids within 0.009 D (median), and a replicate scatter as small.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

SHARE = 0.1  # the range: the drift times around the apex down to this share of it
SECOND_PEAK = 0.5  # a peak outside the range this high (share of the apex): drift_unclear
MIN_CLUSTER = 0.05  # natural isotope clusters at least this share of the tallest make a profile
_DECIMALS = 6  # drift times (ms) compared at this precision (the same bins in every file)


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


def scan_profile(scans, peaks: np.ndarray, half: np.ndarray):
    """The drift profile of ``scans`` (with :attr:`hxex.io.mzml.Scan.binned`) at ``peaks``
    (m/z, each +- its ``half``): the distinct drift times and the intensity at each, summed
    over the scans; None when no scan has drift times."""
    parts = [s.binned for s in scans if s.binned is not None]
    if not parts:
        return None
    mz = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    d = drift_times(np.concatenate([p[2] for p in parts]))
    near = np.zeros(len(mz), dtype=bool)
    for p, h in zip(np.atleast_1d(peaks), np.broadcast_to(half, np.shape(peaks)), strict=True):
        near |= np.abs(mz - p) <= h
    values, where = np.unique(d, return_inverse=True)
    return values, np.bincount(where, weights=np.where(near, y, 0.0), minlength=len(values))


@dataclass(frozen=True)
class DriftRange:
    """A peptide ion's drift range (ms): ``start`` and ``end``, the profile's ``apex``, the
    tallest peak outside the range as a share of the apex (``second``; 0 without one), and the
    number of undeuterated files whose profiles made it."""

    start: float
    end: float
    apex: float
    second: float = 0.0
    n_files: int = 0

    @property
    def unclear(self) -> bool:
        return self.second >= SECOND_PEAK


def find_range(drifts: np.ndarray, profile: np.ndarray, share: float = SHARE
               ) -> DriftRange | None:
    """The drift range of a profile (drift times in increasing order, intensity at each): the
    apex and the drift times either side of it down to ``share`` of the apex; None without
    signal."""
    profile = np.asarray(profile, dtype=float)
    if not len(profile) or not profile.max() > 0:
        return None
    apex = int(np.argmax(profile))
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


class DriftProfiles:
    """Drift profiles summed per peptide ion over the undeuterated controls' files."""

    def __init__(self):
        self._profiles: dict[tuple, dict[float, float]] = {}
        self._files: dict[tuple, set] = {}

    def add(self, key: tuple, drifts: np.ndarray, intensity: np.ndarray, file: str) -> None:
        summed = self._profiles.setdefault(key, {})
        for d, v in zip(drifts.tolist(), intensity.tolist(), strict=True):
            summed[d] = summed.get(d, 0.0) + v
        self._files.setdefault(key, set()).add(file)

    def ranges(self, share: float = SHARE) -> dict[tuple, DriftRange]:
        """The range of every peptide ion whose profile has signal."""
        out = {}
        for key, summed in self._profiles.items():
            drifts = np.array(sorted(summed))
            found = find_range(drifts, np.array([summed[d] for d in drifts]), share)
            if found is not None:
                out[key] = DriftRange(found.start, found.end, found.apex, found.second,
                                      len(self._files[key]))
        return out


def drift_record(ranges: Mapping[tuple, DriftRange], unclear: Iterable[int]) -> dict:
    """The read record's ``drift_gate`` entry: the share, every range found and the series
    flagged ``drift_unclear``."""
    return {
        "share": SHARE,
        "second_peak": SECOND_PEAK,
        "ranges": [{"peptide": k[0], "modification": k[1], "start_seq": k[2], "end_seq": k[3],
                    "charge": k[4], "drift_start": r.start, "drift_end": r.end,
                    "apex": r.apex, "second": round(r.second, 4), "n_files": r.n_files}
                   for k, r in sorted(ranges.items())],
        "unclear": sorted(int(d) for d in unclear),
    }


def unclear_series(record: Mapping | None) -> list[int]:
    """The series (data_id) flagged ``drift_unclear`` in a read record's ``drift_gate`` entry."""
    if not record:
        return []
    return [int(d) for d in record.get("unclear", ())]
