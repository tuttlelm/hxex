# SPDX-License-Identifier: GPL-3.0-or-later
"""Finding a series' retention time from its exact m/z (``[input.scans] find_retention_time``;
docs/HISTORY.md items 200-201, docs/PLAN.md Next steps 16).

A peptide table's retention time can point at another species: on an Orbitrap digest the
series the picker's m/z check flags (:mod:`hxex.picking.mz_check`) were mostly another,
near-isobaric peptide at the table's time, the peptide itself eluting 0.3 to 2 minutes away at
exactly its predicted m/z (HISTORY 200). While the undeuterated controls are read, this module
looks for where the peptide's exact m/z elutes in each of them:

- Traces: in every MS1 scan of an undeuterated control, the peak tops of the profile
  (:func:`scan_tops`); per series, the top nearest M-1, M, M+1 and M+2 of its predicted
  envelope, counted only when it lies within a tolerance of the prediction moved by the
  injection's calibration (the m/z check's line through the injection's controls). The
  tolerance is ``SCAN_TOLERANCE`` times the m/z check's limit for the injection
  (``off_mz_sd`` times its scatter, at least 3 ppm), or ``rt_search_ppm``. A top, not the
  tallest point in a window: a species 10 to 50 ppm away never counts.
- Candidates, per injection: the maxima of the M trace (a running mean over
  ``SMOOTH_SCANS``) that are the tallest within one peak width (the run's typical width at half
  maximum, measured by ``find_peak``) and stand ``PROMINENCE`` of their height above the
  valleys either side; their top scans are those at half the apex or more.
- Agreement: a candidate counts when every undeuterated injection of the sample shows one
  within the retention tolerance of it (the larger of the run's typical peak width and the
  99th percentile of how far apart the injections put the series the m/z check passes, or
  ``rt_search_tolerance``), each with at least ``MIN_SCANS`` top scans, and when the
  injections' summed top scans look like the peptide: M+1/M and M+2/M within ``M1_RANGE`` and
  ``M2_RANGE`` of the natural-abundance prediction, M+1 present in ``PRESENT`` of the top scans,
  M-1/M at most ``MAX_M_MINUS1`` (else M is another species' isotope), and the median
  calibrated offset of M within the m/z check's limit (the peak would not be flagged
  ``off_mz`` itself).
- Verdict: ``here`` (one candidate, or the tallest at least ``DOMINANCE`` times the next, within
  the retention tolerance of where the series was read), ``elsewhere`` (such a candidate at
  another time), ``uncertain`` (elsewhere, more than ``UNCERTAIN_TOLERANCES`` tolerances away
  and weaker than the exact-m/z signal where the series was read), ``ambiguous`` (several, none
  dominant) or ``not found``.

Searched: the peptide series (find_peak applies) of every sample with undeuterated controls
that the m/z check flags ``off_mz``, that have no undeuterated offset (the check could not
assess them), or that have no undeuterated spectrum at all. ``suggest`` records the verdicts
(the bundle's ``retention_suggestions.csv``); ``automatic`` also reads every ``elsewhere``
series found in two injections or more at its found time, in every run, and the run flags it
``rt_found``. Measured on the Orbitrap digest (HISTORY 201): of 40 series with a known
retention time, 34 found and right, 0 wrong.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks as _maxima

from .. import chem
from ..picking import PickerConfig
from ..picking.mz_check import MIN_LIMIT_PPM, limit_ppm, series_offsets
from ..picking.pick import GAP_SPACING, plan_bins
from ..spectra import UNDEUT, Spectrum, SpectrumSet, entry_rt

log = logging.getLogger(__name__)

MODES = ("off", "suggest", "automatic")
VERDICTS = ("here", "elsewhere", "uncertain", "ambiguous", "not found")
SEARCH_PPM = 60.0  # a top further than this from a prediction is not looked at
SCAN_TOLERANCE = 1.25  # a scan's top counts within this many times the m/z check's limit
SMOOTH_SCANS = 3  # the M trace's running mean
PROMINENCE = 0.5  # a candidate stands this share of its height above the valleys either side
MIN_SCANS = 3  # top scans with M in each injection
PRESENT = 0.6  # share of the top scans with M+1
M1_RANGE = (0.6, 1.5)  # M+1/M over its prediction (true peaks: 0.72 to 1.31, q02 to q99)
M2_RANGE = (0.3, 2.5)  # M+2/M over its prediction (0.29 to 1.79)
MAX_M_MINUS1 = 0.5  # M-1/M: above it, "M" is another species' isotope
DOMINANCE = 3.0  # the tallest candidate this many times the next: one answer
UNCERTAIN_TOLERANCES = 2.0
RT_QUANTILE = 99.0  # of the injections' disagreement over the series the m/z check passes
MIN_RT_SERIES = 10  # fewer such series: the peak width alone
SCAN_CHUNK = 400  # scans read at once
STEP = 1.003354835  # the 13C isotope step (Da)


@dataclass(frozen=True)
class _Target:
    """A searched target: its position in the read's targets, its predictions (M-1, M, M+1,
    M+2 m/z) and the predicted M+1/M and M+2/M."""

    index: int
    mz: np.ndarray
    r1: float
    r2: float


@dataclass
class _Candidate:
    rt: float  # s
    height: float
    n_scans: int
    n_present: int
    sums: np.ndarray  # M-1, M, M+1, M+2 over the top scans
    offsets: np.ndarray  # the calibrated M offsets (ppm) of the top scans


@dataclass
class _File:
    """What one undeuterated injection gave: its run, its tolerance and limit (ppm), its peak
    width (s), and per searched target the candidates and the exact-m/z M trace."""

    run: object
    name: str
    tolerance_ppm: float
    limit_ppm: float
    calibration: tuple[float, float, float]
    width: float
    times: np.ndarray
    candidates: dict[int, list[_Candidate]]
    m_trace: dict[int, np.ndarray]


def scan_tops(mz: np.ndarray, y: np.ndarray, centroid: bool | None) -> tuple[np.ndarray,
                                                                             np.ndarray]:
    """The peak tops of one scan (``mz`` sorted): (m/z, height). Profile data: a point above 0
    higher than the point before it and at least as high as the one after (a gap, more than
    ``GAP_SPACING`` times the local spacing, counting as lower), at the vertex of the parabola
    through it and its neighbours when both are lower and evenly spaced; the local spacing is a
    running median, since a scan's spacing grows with m/z. Centroided data: every point above
    0."""
    mz = np.asarray(mz, dtype=float)
    y = np.asarray(y, dtype=float)
    if centroid or len(mz) < 3:
        keep = y > 0
        return mz[keep], y[keep]
    d = np.diff(mz)
    local = median_filter(d, size=15, mode="nearest")
    gap = d > GAP_SPACING * local
    rises = np.r_[True, gap | (y[:-1] < y[1:])]
    falls = np.r_[gap | (y[1:] <= y[:-1]), True]
    j = np.flatnonzero(rises & falls & (y > 0))
    x, h = mz[j].copy(), y[j].copy()
    inner = (j > 0) & (j < len(mz) - 1)
    inner[inner] &= ~gap[j[inner] - 1] & ~gap[j[inner]]
    k = j[inner]
    y0, y1, y2 = y[k - 1], y[k], y[k + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        den = y0 - 2 * y1 + y2
        delta = np.where(den < 0, 0.5 * (y0 - y2) / den, 0.0)
    delta = np.clip(np.nan_to_num(delta), -0.5, 0.5)
    x[inner] = mz[k] + delta * 0.5 * (mz[k + 1] - mz[k - 1])
    h[inner] = y1 - 0.25 * (y0 - y2) * delta
    order = np.argsort(x, kind="stable")
    return x[order], h[order]


def _targets(targets, picker: PickerConfig, registry: chem.ModificationRegistry
             ) -> dict[int, _Target]:
    """The targets the search applies to (those find_peak applies to: MS1 peptides without
    their own rt_start, rt_end)."""
    from .mzml import _finds_peak

    out = {}
    plans: dict[tuple, _Target] = {}
    for i, t in enumerate(targets):
        if not _finds_peak(t):
            continue
        e = t.entry
        key = (e.peptide, e.charge, e.modification)
        if key not in plans:
            plan = plan_bins(e.peptide, e.charge, modification=e.modification, config=picker,
                             registry=registry)
            shape, offsets = plan.clusters
            mono = float(plan.undeut_mz)
            z = int(e.charge)
            plans[key] = _Target(i, np.array([mono - STEP / z, mono, mono + offsets[1] / z,
                                              mono + offsets[2] / z]),
                                 float(shape[1] / shape[0]), float(shape[2] / shape[0]))
        out[i] = replace(plans[key], index=i)
    return out


class RetentionSearch:
    """The search over a read's undeuterated controls (module docstring): :meth:`add_file`
    while each control's file is open, then :meth:`decide`."""

    def __init__(self, mode: str, config, picker: PickerConfig,
                 registry: chem.ModificationRegistry, targets: Sequence, tie: bool):
        self.mode = mode
        self.config = config
        self.picker = picker
        self.registry = registry
        self.targets = list(targets)
        self.tie = tie
        self.searchable = _targets(targets, picker, registry)
        self.off_mz_sd = picker.off_mz_sd if picker.off_mz_sd is not None else 6.0
        self.files: list[_File] = []
        self.record: dict | None = None

    @classmethod
    def start(cls, config, picker, registry, targets, runs, tie) -> RetentionSearch | None:
        """The search for this read, or None: off, find_peak off, no undeuterated control
        with a fixed time, or no target find_peak applies to."""
        mode = config.find_retention_time
        if mode == "off" or not config.find_peak:
            return None
        if not any(r.role == UNDEUT and r.fixed for r in runs):
            log.info("find_retention_time: no undeuterated control with a fixed time to search")
            return None
        search = cls(mode, config, picker, registry, targets, tie)
        return search if search.searchable else None

    def reads(self, run) -> bool:
        """Whether ``run`` is searched (an undeuterated control with a fixed time)."""
        return run.role == UNDEUT and run.fixed

    def _applying(self, run) -> list[_Target]:
        from .mzml import target_applies

        return [t for i, t in self.searchable.items()
                if target_applies(self.targets[i], run, self.tie)]

    def add_file(self, f, run, run_spectra: Sequence[Spectrum], width: float,
                 cancel=None) -> None:
        """Search the open file ``f`` of the undeuterated control ``run``, whose spectra were
        just read (``run_spectra``) and whose typical peak width is ``width`` (s; NaN: not
        measured)."""
        from ..hooks import check_cancel

        wanted = self._applying(run)
        if not wanted:
            return
        mine = [s for s in run_spectra if _peptide_control(s)]
        intercept, slope, scatter = 0.0, 0.0, float("nan")
        if mine:
            checked = series_offsets(SpectrumSet.from_spectra(mine), replace(
                self.picker, off_mz_sd=self.off_mz_sd), registry=self.registry)
            line = checked.calibration.get((str(run.sample), str(run.rep)))
            if line is None and checked.calibration:
                line = next(iter(checked.calibration.values()))
            if line is not None and np.isfinite(line[0]):
                intercept, slope, scatter = line
        limit = limit_ppm(scatter, self.off_mz_sd)
        tolerance = (float(self.config.rt_search_ppm) if self.config.rt_search_ppm is not None
                     else SCAN_TOLERANCE * limit)
        table = f.scans()
        ms1 = np.flatnonzero(np.asarray(table.ms_level) == 1)
        times = np.asarray(table.time, dtype=float)[ms1]
        if not np.isfinite(width) or width <= 0:
            width = 3.0 * float(np.median(np.diff(times))) if len(times) > 1 else 1.0
        pred = np.stack([t.mz for t in wanted])  # (targets, 4)
        calibrated = pred * (1 + (intercept + slope * pred[:, 1:2]) * 1e-6)
        flat = calibrated.ravel()
        order = np.argsort(flat)
        sorted_flat = flat[order]
        heights = np.zeros((len(ms1), flat.size), dtype=np.float32)
        offsets = np.full((len(ms1), flat.size), np.nan, dtype=np.float32)
        windows = [(float(p.min() * (1 - SEARCH_PPM * 1e-6)) - 0.05,
                    float(p.max() * (1 + SEARCH_PPM * 1e-6)) + 0.05) for p in calibrated]
        for a in range(0, len(ms1), SCAN_CHUNK):
            check_cancel(cancel)
            chunk = ms1[a:a + SCAN_CHUNK]
            scans = f.read_scans([int(table.index[k]) for k in chunk], windows)
            for n, k in enumerate(chunk):
                s = scans[int(table.index[k])]
                x, h = scan_tops(s.mz, s.intensity, s.centroid)
                if not len(x):
                    continue
                pos = np.searchsorted(x, sorted_flat)
                lo = np.clip(pos - 1, 0, len(x) - 1)
                hi = np.clip(pos, 0, len(x) - 1)
                near = np.where(np.abs(x[lo] - sorted_flat) <= np.abs(x[hi] - sorted_flat),
                                lo, hi)
                ppm = (x[near] - sorted_flat) / sorted_flat * 1e6
                ok = np.abs(ppm) <= SEARCH_PPM
                offsets[a + n, order[ok]] = ppm[ok]
                heights[a + n, order[ok]] = h[near][ok]
        offsets = offsets.reshape(len(ms1), len(wanted), 4)
        heights = heights.reshape(len(ms1), len(wanted), 4)
        exact = np.where(np.abs(np.nan_to_num(offsets, nan=np.inf)) <= tolerance, heights, 0.0)
        candidates, m_trace = {}, {}
        for j, t in enumerate(wanted):
            candidates[t.index] = _candidates(times, exact[:, j, :], offsets[:, j, 1], width)
            m_trace[t.index] = exact[:, j, 1].astype(np.float32)
        self.files.append(_File(run, f.path.name, tolerance, limit, (intercept, slope, scatter),
                                width, times, candidates, m_trace))
        log.info("%s: retention search: %d series, tops within %.1f ppm of the calibrated m/z "
                 "(%+.2f ppm at m/z 0, %+.3f ppm per 100 m/z; scatter %.2f ppm)", f.path.name,
                 len(wanted), tolerance, intercept, 100 * slope, scatter)

    def decide(self, undeut: Sequence[Spectrum], series: Mapping[tuple[str, int], int]
               ) -> dict[tuple[str, int], float]:
        """The verdicts over every searched injection (``undeut``: the undeuterated spectra
        read; ``series``: (sample, target position) -> data_id), kept in :attr:`record`; with
        ``automatic``, the retention times (min) to read the series at, by (sample, target
        position)."""
        if not self.files:
            self.record = None
            return {}
        own = [s for s in undeut if _peptide_control(s)]
        checked = (series_offsets(SpectrumSet.from_spectra(own), replace(
            self.picker, off_mz_sd=self.off_mz_sd), registry=self.registry) if own else None)
        assessed = set() if checked is None else {int(d) for d in checked.data_id}
        off = set() if checked is None else {int(d) for d in checked.data_id[checked.off]}
        read_rt: dict[int, list[float]] = {}
        for s in own:
            if np.isfinite(s.rt):
                read_rt.setdefault(int(s.data_id), []).append(float(s.rt))
        spreads = [60.0 * float(np.ptp(v)) for d, v in read_rt.items()
                   if d in assessed and d not in off and len(v) >= 2]
        width = float(np.nanmedian([f.width for f in self.files]))
        if self.config.rt_search_tolerance is not None:
            tolerance = float(self.config.rt_search_tolerance)
        elif len(spreads) >= MIN_RT_SERIES:
            tolerance = max(float(np.percentile(spreads, RT_QUANTILE)), width)
        else:
            tolerance = width
        by_sample: dict[str, list[_File]] = {}
        for f in self.files:
            by_sample.setdefault(str(f.run.sample), []).append(f)
        rows, moved = [], {}
        for sample, files in by_sample.items():
            for i in sorted({i for f in files for i in f.candidates}):
                d = series.get((sample, i))
                if d is None or d not in read_rt:
                    why = "no undeuterated spectrum"
                elif d not in assessed:
                    why = "no undeuterated offset"
                elif d in off:
                    why = "off_mz"
                else:
                    continue
                mine = [f for f in files if i in f.candidates]
                table_rt = entry_rt(self.targets[i].entry)
                where = (float(np.median(read_rt[d])) if d is not None and d in read_rt
                         else table_rt)
                row = self._verdict(mine, i, where, tolerance)
                entry = self.targets[i].entry
                applied = (self.mode == "automatic" and row["verdict"] == "elsewhere"
                           and row["n_injections"] >= 2)
                if applied:
                    moved[(sample, i)] = row["proposed_rt"]
                rows.append({"sample": sample, "target": i, "data_id": d,
                             "peptide": entry.peptide, "peptide_range": entry.peptide_range,
                             "charge": int(entry.charge), "modification": entry.modification,
                             "table_rt": table_rt if np.isfinite(table_rt) else None,
                             "read_rt": (float(np.median(read_rt[d])) if d is not None
                                         and d in read_rt else None),
                             "searched_because": why, **row, "applied": applied})
        self.record = {
            "mode": self.mode, "rt_tolerance_s": tolerance, "peak_width_s": width,
            "files": [{"file": f.name, "sample": str(f.run.sample), "rep": str(f.run.rep),
                       "tolerance_ppm": f.tolerance_ppm, "limit_ppm": f.limit_ppm,
                       "calibration": list(f.calibration)} for f in self.files],
            "series": rows,
        }
        counts = {v: sum(r["verdict"] == v for r in rows) for v in VERDICTS}
        log.info("retention search: %d series searched (retention tolerance %.1f s): %s",
                 len(rows), tolerance, ", ".join(f"{n} {v}" for v, n in counts.items()))
        found = counts["elsewhere"]
        if found and self.mode == "suggest":
            log.warning("%d series: the peptide's exact m/z elutes at another time than the "
                        "peptide table's (retention_suggestions.csv); check those times and "
                        "correct the table, or set [input.scans] find_retention_time = "
                        "\"automatic\"", found)
        elif moved:
            log.warning("%d series read at the retention time found for their exact m/z "
                        "(flagged rt_found; retention_suggestions.csv)", len(moved))
        return moved

    def _verdict(self, files: list[_File], i: int, where: float, tolerance: float) -> dict:
        """The candidates of target ``i`` found in every injection ``files``, checked, and the
        verdict (``where``: the time, min, the series was read at)."""
        target = self.searchable[i]
        lists = [[c for c in f.candidates[i] if c.n_scans >= MIN_SCANS] for f in files]
        limit = float(np.mean([f.limit_ppm for f in files]))
        matched = []
        used = [set() for _ in lists]
        for x in sorted(lists[0], key=lambda c: -c.height):
            members = [x]
            for n, other in enumerate(lists[1:], start=1):
                near = [j for j, y in enumerate(other)
                        if j not in used[n] and abs(y.rt - x.rt) <= tolerance]
                if not near:
                    break
                j = max(near, key=lambda j: other[j].height)
                used[n].add(j)
                members.append(other[j])
            else:
                if _looks_like(members, target, limit):
                    matched.append((float(np.mean([c.rt for c in members])),
                                    float(np.mean([c.height for c in members]))))
        matched.sort(key=lambda c: -c[1])
        out = {"n_injections": len(files), "proposed_rt": None,
               "candidates": [[round(rt / 60.0, 4), height] for rt, height in matched]}
        if not matched:
            return {**out, "verdict": "not found"}
        if len(matched) > 1 and matched[0][1] < DOMINANCE * matched[1][1]:
            return {**out, "verdict": "ambiguous"}
        rt, height = matched[0]
        out["proposed_rt"] = rt / 60.0
        if np.isfinite(where) and abs(rt - 60.0 * where) <= tolerance:
            return {**out, "verdict": "here"}
        there = 0.0
        if np.isfinite(where):
            for f in files:
                near = np.abs(f.times - 60.0 * where) <= tolerance
                if near.any():
                    there += float(f.m_trace[i][near].max()) / len(files)
        far = not np.isfinite(where) or abs(rt - 60.0 * where) > UNCERTAIN_TOLERANCES * tolerance
        return {**out, "verdict": "uncertain" if far and height < there else "elsewhere"}


def _peptide_control(s: Spectrum) -> bool:
    """Whether ``s`` is an undeuterated control of a peptide ion (not a fragment)."""
    return s.role == UNDEUT and "fragment" not in s.entry.extra


def _candidates(times: np.ndarray, exact: np.ndarray, m_offset: np.ndarray,
                width: float) -> list[_Candidate]:
    """The candidate elution peaks of one series in one injection (``exact``: per scan the
    heights of M-1, M, M+1 and M+2 within the tolerance, else 0; ``m_offset``: the M top's
    calibrated offset)."""
    m = exact[:, 1].astype(float)
    if not m.any():
        return []
    smooth = np.convolve(m, np.ones(SMOOTH_SCANS) / SMOOTH_SCANS, mode="same")
    spacing = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    peaks, props = _maxima(smooth, distance=max(1, int(round(width / spacing))), prominence=0)
    out = []
    for p, prominence in zip(peaks, props["prominences"], strict=True):
        if smooth[p] <= 0 or prominence < PROMINENCE * smooth[p]:
            continue
        lo = hi = p
        while lo > 0 and smooth[lo - 1] >= 0.5 * smooth[p]:
            lo -= 1
        while hi < len(smooth) - 1 and smooth[hi + 1] >= 0.5 * smooth[p]:
            hi += 1
        top = np.arange(lo, hi + 1)
        with_m = m[top] > 0
        out.append(_Candidate(
            rt=float(times[p]), height=float(m[top].max()), n_scans=int(with_m.sum()),
            n_present=int((with_m & (exact[top, 2] > 0)).sum()),
            sums=exact[top].sum(axis=0).astype(float),
            offsets=np.asarray(m_offset[top][with_m], dtype=float)))
    return out


def _looks_like(members: list[_Candidate], target: _Target, limit: float) -> bool:
    """Whether the injections' summed top scans look like the peptide (module docstring)."""
    sums = np.sum([c.sums for c in members], axis=0)
    n_m = sum(c.n_scans for c in members)
    if sums[1] <= 0 or not n_m:
        return False
    offsets = np.concatenate([c.offsets for c in members])
    offsets = offsets[np.isfinite(offsets)]
    return bool(
        sum(c.n_present for c in members) / n_m >= PRESENT
        and M1_RANGE[0] <= sums[2] / sums[1] / target.r1 <= M1_RANGE[1]
        and M2_RANGE[0] <= sums[3] / sums[1] / target.r2 <= M2_RANGE[1]
        and sums[0] / sums[1] <= MAX_M_MINUS1
        and len(offsets) and abs(float(np.median(offsets))) <= max(limit, MIN_LIMIT_PPM))


def suggestions_table(record: Mapping | None) -> dict[str, list] | None:
    """``retention_suggestions.csv`` from a read record's ``retention_search`` (None without
    one): one row per searched series."""
    if not record or not record.get("series"):
        return None
    rows = record["series"]

    def text(c):
        return "; ".join(f"{rt:.2f} ({h:.3g})" for rt, h in c)

    return {
        "sample": [r["sample"] for r in rows],
        "peptide_range": [r["peptide_range"] for r in rows],
        "peptide": [r["peptide"] for r in rows],
        "charge": [r["charge"] for r in rows],
        "modification": [r["modification"] for r in rows],
        "searched_because": [r["searched_because"] for r in rows],
        "table_rt": [r["table_rt"] for r in rows],
        "read_rt": [r["read_rt"] for r in rows],
        "verdict": [r["verdict"] for r in rows],
        "proposed_rt": [None if r["proposed_rt"] is None else round(r["proposed_rt"], 3)
                        for r in rows],
        "applied": [bool(r["applied"]) for r in rows],
        "n_injections": [r["n_injections"] for r in rows],
        "candidates": [text(r["candidates"]) for r in rows],
    }


def moved_series(record: Mapping | None) -> list[int]:
    """The series (data_id) read at a found retention time (flag ``rt_found``)."""
    if not record:
        return []
    return [int(r["data_id"]) for r in record.get("series", ())
            if r.get("applied") and r.get("data_id") is not None]


def with_rt(target, rt: float):
    """``target`` with its peptide table ``rt`` (min) replaced."""
    entry = target.entry
    return replace(target, entry=replace(entry, extra={**dict(entry.extra), "rt": f"{rt:.4f}"}))
