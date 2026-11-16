# SPDX-License-Identifier: GPL-3.0-or-later
"""Retention traces of a spectrum read from scan files (PLAN Next steps 26, the review's trace
panel).

A spectrum read from an mzML or Thermo raw file sums the scans of one retention window (and,
for scans with ion mobility bins, one drift range). Whether that window sits on the peptide
ion's own elution peak, or on a neighbour's, shows in its retention trace: per scan, the
intensity at the ion's predicted bins, as ``find_peak`` measures it (the tallest point within
the picker's window of each bin, summed over the bins). :func:`retention_traces` opens the
spectrum's file again from the read record (the bundle's ``spectra_source``), with the same
sample's other runs of the same series, and gives each run's trace around its window, the
window summed, and, for files with ion mobility bins, the selected spectrum's drift profile and
the drift range it was summed over.

Nothing here is stored in a bundle: the traces are read when asked (about half a second per
file on the intact-peptide benchmark's 200 MB ion-mobility files). Each file's m/z offset
(``mz_calibration``) moves the predicted bins as it moved the spectra.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .. import chem
from ..hooks import CancelHook, check_cancel
from ..picking import PickerConfig, plan_bins
from ..spectra import Spectrum, split_spectrum_id
from .drift_gate import drift_times, isotope_peaks, scan_profile
from .mzml import (
    _MZML,
    SCAN_READERS,
    MzmlConfig,
    MzmlError,
    MzmlRun,
    MzmlTarget,
    _channel,
    _gated_kind,
    _ladder_trace,
    _run_file_name,
    _scan_record,
    _target_label,
    sum_centroid,
    sum_mobility_bins,
    sum_profile,
    target_applies,
    target_scans,
)

MARGIN = 0.5  # minutes shown either side of the windows at least (and one window's width)
_SCANS = re.compile(r"/scans?(\d+)(?:-(\d+))?$")


@dataclass(frozen=True)
class Trace:
    """One run's retention trace of a peptide ion: ``times`` (minutes, the acquisition clock),
    ``intensity`` (per scan, at the ion's predicted bins; within ``drift`` when the spectrum
    was summed over a drift range), the ``window`` its spectrum summed (minutes, first and last
    scan), and the run: ``file`` (name), ``rep``, ``role`` and ``time`` (s; None for a control
    or a kinetics run), ``spectrum_id``, and whether it is the ``selected`` spectrum's run."""

    spectrum_id: str
    file: str
    rep: int
    role: str | None
    time: float | None
    times: np.ndarray
    intensity: np.ndarray
    window: tuple[float, float]
    drift: tuple[float, float] | None = None
    selected: bool = False


@dataclass(frozen=True)
class DriftProfile:
    """The selected spectrum's drift profile: ``drift`` (ms, the times the file records) and
    the ``intensity`` on the ion's natural isotope peaks at each, summed over the scans of its
    window, with the ``range`` the spectrum was summed over (None: every drift time)."""

    drift: np.ndarray
    intensity: np.ndarray
    range: tuple[float, float] | None


@dataclass(frozen=True)
class IonTrace:
    """The trace of another ion of the selected spectrum's peptide in the selected run
    (:meth:`TraceReader.ion_traces`): a ``label`` (a charge state of the peptide in MS1, or a
    fragment channel: its ion types, activation, MS level and precursor charge), the
    ``times`` (minutes) and ``intensity`` (summed over the ``n_ions`` ions it holds)."""

    label: str
    times: np.ndarray
    intensity: np.ndarray
    n_ions: int = 1


@dataclass(frozen=True)
class RetentionTraces:
    """The traces of one spectrum's series in its sample (:func:`retention_traces`): the peptide
    ion's ``label``, one :class:`Trace` per run (the selected first), the selected spectrum's
    :class:`DriftProfile` (None without ion mobility bins), a ``note`` saying why there is
    nothing to draw (no traces) or what is missing (a file not found, or not the file the run
    read), and ``ions``: the peptide's other ions in the selected run (:class:`IonTrace`; empty
    unless asked for)."""

    label: str
    traces: list[Trace]
    drift: DriftProfile | None = None
    note: str | None = None
    ions: list[IonTrace] = field(default_factory=list)
    others_label: str = "the sample's other files"  # the traces behind, as the legend names them
    ions_label: str = "this peptide's other ions in this file"
    # the times (minutes, sorted) of every scan of the ion in the selected run's whole file: a
    # window is a set of scans, so the window editor's boxes snap to them (None: not read)
    scan_times: np.ndarray | None = None


@dataclass(frozen=True)
class Summed:
    """A spectrum summed again over another window (:meth:`TraceReader.summed`) and, when
    asked for, the :class:`DriftProfile` of the scans it summed (None: not asked for, or the
    scans have no ion mobility bins)."""

    spectrum: Spectrum
    profile: DriftProfile | None = None


def readable(source: Mapping | None) -> bool:
    """Whether spectra with this read record (a bundle's ``spectra_source``) were read from
    scan files, so that their retention traces can be read again."""
    return bool(source) and source.get("reader") in SCAN_READERS


_READS = threading.Lock()  # every TraceReader's reads (see TraceReader.__init__)


class TraceReader:
    """Reads retention traces from the scan files of one read record, keeping each file it
    opened (and its scan table) for the next spectrum. ``data_root`` is the project's (relative
    recorded paths start there); ``directory``, when given, holds the files by name (data that
    has moved); ``thermo_dir`` is Thermo's assemblies (Thermo raw files only)."""

    def __init__(self, source: Mapping, *, data_root: str | Path | None = None,
                 directory: str | Path | None = None, thermo_dir: str | Path | None = None):
        if not readable(source):
            raise ValueError("retention traces need spectra read from mzML or Thermo raw files")
        self.source = source
        self.runs = [MzmlRun.from_dict(r) for r in source["runs"]]
        self.targets = [MzmlTarget.from_dict(t) for t in source["targets"]]
        self.config = MzmlConfig.from_dict(source["config"])
        self.picker = PickerConfig.from_dict(source["picker"])
        self.registry = chem.ModificationRegistry.from_dict(source["registry"])
        self.of_series = {int(s["data_id"]): int(s["target"]) for s in source["series"]}
        self.windows = [tuple(w) for w in source.get("windows") or []]
        offsets = source.get("mz_calibration") or []
        self.offsets = [float((o or {}).get("offset_ppm") or 0.0) for o in offsets]
        self.offsets += [0.0] * (len(self.runs) - len(self.offsets))
        self.data_root = None if data_root is None else Path(data_root)
        self.directory = None if directory is None else Path(directory)
        self.thermo_dir = thermo_dir
        if source.get("reader") == "thermo":
            # Thermo's assemblies loaded here, in the thread that makes the reader (the
            # window's): .NET must have been started on a thread that stays attached to it.
            # Started from a worker thread, the main thread later touches .NET objects
            # unattached and the process aborts ("Cannot enter GC unsafe region if the thread
            # is not attached"). The window starts .NET before Qt (hxex.gui.start_dotnet):
            # started here, inside the running window, Mono crashed on a desktop display
            from .thermo import _thermo_api

            _thermo_api(thermo_dir)
        self._open: dict[int, object] = {}
        self._changed: dict[int, str | None] = {}  # per run opened: not the file the run read
        self._closed = False  # closed: nothing is read any more (audit of 2026-11-15, F-18)
        # one read at a time in the whole process, whichever reader asks: two threads reading
        # Thermo raw files through .NET at once locked the window for good (2026-11-09)
        self._lock = _READS

    # --- files ---------------------------------------------------------------------------

    def path(self, run_i: int) -> Path:
        """Where run ``run_i``'s file is now (as :func:`hxex.io.mzml.read_mzml_source` looks)."""
        run = self.runs[run_i]
        if self.directory is not None:
            known = (self.source.get("files") or [None] * len(self.runs))[run_i] or {}
            return self.directory / (known.get("name") or _run_file_name(run.path))
        p = Path(run.path)
        if not p.is_absolute() and self.data_root is not None:
            p = self.data_root / p
        return p

    def _file(self, run_i: int):
        if self._closed:  # a read queued on a reader whose files were closed would open them
            raise ValueError("the trace reader was closed (other results are shown)")
        if run_i not in self._open:
            path = self.path(run_i)
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found")
            self._changed[run_i] = self._identity_note(run_i, path)
            if self.source.get("reader") == "thermo":
                from .thermo import _format
                self._open[run_i] = _format(self.thermo_dir).open(path)
            else:
                self._open[run_i] = _MZML.open(path)
        return self._open[run_i]

    def _identity_note(self, run_i: int, path: Path) -> str | None:
        """Whether ``path`` is the file the run read (the record's ``files`` entry): its size,
        and the checksum an indexed mzML file carries at its end (cheap to read); a SHA-256 of
        the whole file is not computed here (seconds on a large raw file at every first open).
        A file that differs is still read: the note says so (audit of 2026-11-15, F-14)."""
        known = (self.source.get("files") or [None] * len(self.runs))[run_i] or {}
        if not known.get("size"):
            return None
        why = None
        try:
            if path.stat().st_size != int(known["size"]):
                why = "its size differs"
            elif known.get("algorithm") == "mzml-sha1":
                if _tail_checksum(path) != str(known.get("checksum") or "").lower():
                    why = "its checksum differs"
        except OSError:
            return None
        if why is None:
            return None
        return (f"{path.name} is not the file the run read ({why}): its trace and the preview "
                "may not match the results")

    def changed_file(self, run_i: int) -> str | None:
        """The note of :meth:`_identity_note` for a run whose file was opened (None: the file
        the run read, or not opened yet)."""
        return self._changed.get(run_i)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for f in self._open.values():
                f.close()
            self._open.clear()

    # --- spectra -------------------------------------------------------------------------

    def locate(self, spectrum_id: str, data_id: int) -> tuple[int, int, int] | None:
        """The run (position in the record), and the first and last scan (positions in the
        file's scan table) that ``spectrum_id`` of series ``data_id`` summed; None when the id
        does not name one of the record's runs."""
        target = self.targets[self.of_series[int(data_id)]]
        try:
            _, local = split_spectrum_id(spectrum_id)
        except ValueError:
            return None
        m = _SCANS.search(local)
        if m is None:
            return None
        first, last = int(m.group(1)), int(m.group(2) or m.group(1))
        head = local[:m.start()]
        label = "/" + _target_label(target)
        if not head.endswith(label):
            return None
        head = head[:-len(label)]
        for run_i, run in enumerate(self.runs):
            if head == f"{run.sample}/{_run_file_name(run.path)}":
                return run_i, first, last
        return None

    def traces(self, spectrum_ids: Sequence[str], data_id: int, *,
               drift_ranges: Mapping[str, tuple[float, float]] | None = None,
               cancel: CancelHook | None = None) -> RetentionTraces:
        """The retention traces of ``spectrum_ids`` (the selected spectrum first, then the
        others of its series ``data_id`` to draw behind it), each run's around the windows of
        all of them (:data:`MARGIN` minutes, or a window's width, either side). A run with
        several spectra of the series (a kinetics run) gives one trace, with the window of the
        first of them listed. ``drift_ranges`` (ms, by spectrum id) are the ranges the spectra
        were summed over (spectra.csv ``drift_start``, ``drift_end``). ``cancel`` is polled
        before each file (:class:`hxex.hooks.Cancelled` when it says stop: the window moved
        on, and a read holding the lock made the next spectrum wait)."""
        with self._lock:
            return self._traces(spectrum_ids, data_id, drift_ranges or {}, cancel)

    def _traces(self, spectrum_ids, data_id, drift_ranges, cancel=None) -> RetentionTraces:
        t_i = self.of_series[int(data_id)]
        target = self.targets[t_i]
        e = target.entry
        label = f"{e.peptide_range} {e.peptide} {e.charge}+"
        if e.modification:
            label += f" {e.modification}"
        found: dict[int, tuple[str, int, int]] = {}
        for sid in spectrum_ids:
            where = self.locate(sid, data_id)
            if where is not None and where[0] not in found:
                found[where[0]] = (sid, where[1], where[2])
        if not spectrum_ids or not found or self.locate(spectrum_ids[0], data_id) is None:
            return RetentionTraces(label, [], note="the spectrum's scans are not named in its "
                                   "id (read by an older version): read the data again")
        selected_run = self.locate(spectrum_ids[0], data_id)[0]
        bins = plan_bins(e.peptide, e.charge, modification=e.modification, config=self.picker,
                         registry=self.registry)
        pred0 = np.asarray(bins.pred_mz, dtype=float)
        peaks0 = isotope_peaks(bins)
        lo_mz, hi_mz = (self.windows[t_i] if t_i < len(self.windows)
                        else (pred0.min() - 1.0, pred0.max() + 1.0))
        missing = []
        scanned = {}
        for run_i, (sid, first, last) in found.items():
            check_cancel(cancel)
            try:
                f = self._file(run_i)
            except FileNotFoundError as exc:
                missing.append(str(exc))
                continue
            table = f.scans()
            rows = np.flatnonzero((table.ms_level == target.ms_level)
                                  & (table.index >= first) & (table.index <= last))
            if not len(rows):
                continue
            scanned[run_i] = (sid, f, table, (float(table.time[rows].min()) / 60.0,
                                              float(table.time[rows].max()) / 60.0))
        if selected_run not in scanned:
            note = "; ".join(missing) or "the spectrum's scans are not in its file"
            return RetentionTraces(label, [], note=note)
        missing += [n for r in scanned if (n := self._changed.get(r))]
        # every scan of the ion in the selected run's whole file (the editor's boxes snap to
        # them; audit of 2026-11-15, F-1)
        sel_table = scanned[selected_run][2]
        every = target_scans(sel_table, self.runs[selected_run],
                             replace(target, rt_window=None), self.config)
        scan_times = np.sort(np.asarray(sel_table.time[every], dtype=float)) / 60.0
        starts = [w[0] for *_, w in scanned.values()]
        ends = [w[1] for *_, w in scanned.values()]
        width = max(b - a for a, b in zip(starts, ends, strict=True))
        pad = max(MARGIN, width)
        span = (60.0 * (min(starts) - pad), 60.0 * (max(ends) + pad))
        traces: list[Trace] = []
        profile = None
        for run_i in [selected_run, *[r for r in scanned if r != selected_run]]:
            check_cancel(cancel)
            sid, f, table, window = scanned[run_i]
            run = self.runs[run_i]
            factor = 1.0 + self.offsets[run_i] * 1e-6
            pred, peaks = pred0 * factor, peaks0 * factor
            rows = target_scans(table, run, replace(target, rt_window=span), self.config)
            rows = rows[np.argsort(table.time[rows], kind="stable")]
            by_index = f.read_scans(table.index[rows], [(lo_mz * factor, hi_mz * factor)],
                                    drift=True)
            scans = [by_index[int(table.index[r])] for r in rows]
            recorded = np.array([s.resolution for s in scans], dtype=float)
            recorded = recorded[np.isfinite(recorded)]
            r_mz = float(getattr(f, "resolving_power_mz", np.nan))
            rp = float(np.median(recorded)) if len(recorded) else None
            half = self.picker.half_width(pred, e.charge, resolving_power=rp,
                                          resolving_power_mz=r_mz)
            drift = drift_ranges.get(sid)
            drift = None if drift is None or not np.all(np.isfinite(drift)) else (
                float(drift[0]), float(drift[1]))
            intensity = np.array([_at_bins(s, pred, half, drift) for s in scans])
            times = np.array([s.time for s in scans], dtype=float) / 60.0
            traces.append(Trace(
                spectrum_id=sid, file=_run_file_name(run.path), rep=int(run.rep),
                role=run.role, time=run.time, times=times, intensity=intensity,
                window=window, drift=drift, selected=run_i == selected_run))
            if run_i == selected_run:
                inside = [s for s in scans if window[0] - 1e-9 <= s.time / 60.0 <= window[1]
                          + 1e-9]
                half_peaks = self.picker.half_width(peaks, e.charge, resolving_power=rp,
                                                    resolving_power_mz=r_mz)
                found_profile = scan_profile(inside, peaks, half_peaks)
                if found_profile is not None:
                    profile = DriftProfile(found_profile[0], found_profile[1], drift)
        note = "; ".join(missing) if missing else None
        return RetentionTraces(label, traces, profile, note, scan_times=scan_times)


    def ion_traces(self, spectrum_id: str, data_id: int, others: Sequence[tuple[str, int]],
                   span: tuple[float, float], *,
                   drift_ranges: Mapping[str, tuple[float, float]] | None = None,
                   cancel: CancelHook | None = None) -> list[IonTrace]:
        """The traces of the peptide's other ions in the selected spectrum's run, over ``span``
        (minutes): ``others`` are their spectra (id, series) in that run. A charge state of the
        peptide in MS1 gives a trace of its own; the fragments of one channel (the same scans:
        activation, MS level, precursor charge) give one, summed over their ions, so that the
        selected ion is compared with its peptide's whole ladder. Each ion within its spectrum's
        drift range (``drift_ranges``, ms, by id) for files with ion mobility bins. ``cancel``
        as for :meth:`traces`, polled before each group of ions."""
        with self._lock:
            return self._ion_traces(spectrum_id, data_id, others, span, drift_ranges or {},
                                    cancel)

    def _ion_traces(self, spectrum_id, data_id, others, span, drift_ranges,
                    cancel=None) -> list[IonTrace]:
        from .mzml import _channel

        where = self.locate(spectrum_id, data_id)
        if where is None:
            return []
        run_i = where[0]
        groups: dict[tuple, list[tuple[str, MzmlTarget]]] = {}
        for sid, d in others:
            at = self.locate(sid, d)
            if at is None or at[0] != run_i:
                continue
            target = self.targets[self.of_series[int(d)]]
            extra = dict(target.entry.extra)
            key = (("fragment", _channel(target)) if "fragment" in extra and target.ms_level >= 2
                   else ("ion", _target_label(target)))
            groups.setdefault(key, []).append((sid, target))
        if not groups:
            return []
        f = self._file(run_i)
        table = f.scans()
        run = self.runs[run_i]
        factor = 1.0 + self.offsets[run_i] * 1e-6
        r_mz = float(getattr(f, "resolving_power_mz", np.nan))
        out = []
        for key, members in sorted(groups.items(), key=lambda kv: _group_order(kv[1][0][1])):
            check_cancel(cancel)
            first = members[0][1]
            rows = target_scans(table, run, replace(first, rt_window=(60.0 * span[0],
                                                                      60.0 * span[1])),
                                self.config)
            if not len(rows):
                continue
            rows = rows[np.argsort(table.time[rows], kind="stable")]
            preds = []
            for sid, t in members:
                e = t.entry
                pred = np.asarray(plan_bins(e.peptide, e.charge, modification=e.modification,
                                            config=self.picker, registry=self.registry).pred_mz,
                                  dtype=float) * factor
                drift = drift_ranges.get(sid)
                drift = None if drift is None or not np.all(np.isfinite(drift)) else (
                    float(drift[0]), float(drift[1]))
                preds.append((pred, e.charge, drift))
            windows = [(p.min() - 1.0, p.max() + 1.0) for p, _, _ in preds]
            by_index = f.read_scans(table.index[rows], windows, drift=True)
            scans = [by_index[int(table.index[r])] for r in rows]
            recorded = np.array([sc.resolution for sc in scans], dtype=float)
            recorded = recorded[np.isfinite(recorded)]
            rp = float(np.median(recorded)) if len(recorded) else None
            y = np.zeros(len(scans))
            for pred, charge, drift in preds:
                half = self.picker.half_width(pred, charge, resolving_power=rp,
                                              resolving_power_mz=r_mz)
                y += np.array([_at_bins(sc, pred, half, drift) for sc in scans])
            times = np.array([sc.time for sc in scans], dtype=float) / 60.0
            out.append(IonTrace(_group_label(key[0], [t for _, t in members]), times, y,
                                len(members)))
        return out

    def spectrum(self, spectrum_id: str, data_id: int, window: tuple[float, float], *,
                 drift: tuple[float, float] | None = None) -> Spectrum:
        """``spectrum_id`` (of series ``data_id``) summed again over another retention
        ``window`` (minutes, the acquisition clock; the scans inside it, ends included) and,
        when given, over the ``drift`` range (ms) of scans with ion mobility bins: the scans
        summed as the reader sums them (:func:`hxex.io.mzml.read_mzml`), within the record's m/z
        window, with the file's m/z offset taken out, and, for a fragment ion of a file the run
        fitted through the instrument's per-scan cut (the record's ``scan_cut``), with the
        scans' record the run gave its spectra (``Spectrum.scans``; audit of 2026-11-15, F-3).
        Its id names the new scans. Raises
        ValueError when the spectrum cannot be summed again (a kinetics run, no scan in the
        window, an id that names no run) and FileNotFoundError when its file is not found."""
        return self.summed(spectrum_id, data_id, window, drift=drift).spectrum

    def summed(self, spectrum_id: str, data_id: int, window: tuple[float, float], *,
               drift: tuple[float, float] | None = None, profile: bool = False) -> Summed:
        """:meth:`spectrum`, with the drift profile of the scans summed when ``profile`` is
        asked for (:class:`Summed`; the window editor's drift panel follows the window being
        edited)."""
        with self._lock:
            return self._spectrum(spectrum_id, data_id, window, drift, profile)

    def _cut_members(self, run_i: int, t_i: int) -> list[int] | None:
        """The targets of the channel of fragment target ``t_i`` in run ``run_i`` (itself
        among them) when the run fitted that file's fragments through the per-scan cut; None
        when it did not, or the target is no fragment ion."""
        target = self.targets[t_i]
        if not ("fragment" in dict(target.entry.extra) and target.ms_level >= 2):
            return None
        cut = (self.source.get("scan_cut") or {}).get("files") or []
        if _run_file_name(self.runs[run_i].path) not in cut:
            return None
        run, key = self.runs[run_i], _channel(target)
        return [j for j, t in enumerate(self.targets)
                if "fragment" in dict(t.entry.extra) and t.ms_level >= 2 and _channel(t) == key
                and target_applies(t, run)]

    def _spectrum(self, spectrum_id, data_id, window, drift, profile=False) -> Summed:
        where = self.locate(spectrum_id, data_id)
        if where is None:
            raise ValueError("the spectrum's scans are not named in its id (read by an older "
                             "version): read the data again")
        run_i = where[0]
        run = self.runs[run_i]
        if not run.fixed:
            raise ValueError("a kinetics run's spectra are not read by retention windows")
        start, end = (float(v) for v in window)
        if start > end:  # one scan (start = end) is a window
            raise ValueError(f"the window's end ({end:g} min) must not be before its start "
                             f"({start:g} min)")
        t_i = self.of_series[int(data_id)]
        target = self.targets[t_i]
        f = self._file(run_i)
        table = f.scans()
        rows = target_scans(table, run, replace(target, rt_window=(60.0 * start, 60.0 * end)),
                            self.config)
        rows = rows[np.argsort(table.time[rows], kind="stable")]
        if not len(rows):
            raise ValueError(f"no scan of the peptide ion between {start:.3f} and {end:.3f} min")
        lo, hi = self.windows[t_i]
        members = self._cut_members(run_i, t_i)
        windows = [(lo, hi)] + [tuple(self.windows[j]) for j in (members or []) if j != t_i]
        by_index = f.read_scans(table.index[rows], windows, drift=drift is not None or profile)
        scans = [by_index[int(table.index[r])] for r in rows]
        parts, gated = [], False
        for s in scans:  # as hxex.io.mzml._summed sums them
            if s.binned is not None and drift is not None:
                m, y, d = s.binned
                d = drift_times(d)
                inside = (m >= lo) & (m <= hi) & (d >= drift[0]) & (d <= drift[1])
                parts.append(sum_mobility_bins(m[inside], y[inside]))
                gated = True
                continue
            inside = (s.mz >= lo) & (s.mz <= hi)
            parts.append((s.mz[inside], s.intensity[inside]))
        if len(parts) == 1:
            mz, y = parts[0]
        elif all(s.centroid for s in scans):
            mz, y = sum_centroid(parts, self.config.sum_ppm)
        else:
            mz, y = sum_profile(parts, self.config.gap_factor)
        order = np.argsort(mz, kind="stable")
        mz, y = np.asarray(mz, float)[order], np.asarray(y, float)[order]
        offset = self.offsets[run_i] if _gated_kind(target) else 0.0
        if offset:
            mz = mz / (1.0 + offset * 1e-6)
        first, last = int(table.index[rows[0]]), int(table.index[rows[-1]])
        scan_text = f"scan{first}" if first == last else f"scans{first}-{last}"
        new_id = spectrum_id[:_SCANS.search(spectrum_id).start()] + "/" + scan_text
        times = np.array([s.time for s in scans], dtype=float)
        weight = np.array([float(np.sum(p[1])) for p in parts])
        rt = float(np.average(times, weights=weight) if weight.sum() > 0 else times.mean())
        resolutions = np.array([s.resolution for s in scans], dtype=float)
        resolutions = resolutions[np.isfinite(resolutions)]
        rp = float(np.median(resolutions)) if len(resolutions) else None
        r_mz = float(getattr(f, "resolving_power_mz", np.nan))
        record = None
        if members is not None:  # as hxex.io.mzml._scan_records weighs the channel's scans
            preds, halves = [], []
            for j in members:
                e = self.targets[j].entry
                pred = np.asarray(plan_bins(e.peptide, e.charge, modification=e.modification,
                                            config=self.picker, registry=self.registry).pred_mz,
                                  dtype=float)
                preds.append(pred)
                halves.append(np.broadcast_to(np.asarray(self.picker.half_width(
                    pred, e.charge, resolving_power=rp, resolving_power_mz=r_mz), dtype=float),
                    pred.shape))
            trace = _ladder_trace(scans, np.concatenate(preds), np.concatenate(halves))
            try:
                record = _scan_record(f, table, rows, np.arange(len(rows)), trace, (lo, hi),
                                      {}, by_index)
            except MzmlError as exc:
                raise ValueError("the new window's spectrum cannot be fitted through the "
                                 f"per-scan cut as the run's was: {exc}") from exc
        found_profile = None
        if profile:
            e = target.entry
            plan = plan_bins(e.peptide, e.charge, modification=e.modification,
                             config=self.picker, registry=self.registry)
            factor = 1.0 + self.offsets[run_i] * 1e-6
            peaks = isotope_peaks(plan) * factor
            half_peaks = self.picker.half_width(peaks, e.charge, resolving_power=rp,
                                                resolving_power_mz=r_mz)
            got = scan_profile(scans, peaks, half_peaks)
            if got is not None:
                found_profile = DriftProfile(got[0], got[1], drift if gated else None)
        spectrum = Spectrum(
            mz, y, replace(target.entry, sample=run.sample), run.spectrum_time, run.rep,
            data_id=int(data_id), label=f"{_run_file_name(run.path)} {scan_text}",
            spectrum_id=new_id, rt=rt / 60.0,
            resolving_power=rp if rp is not None else np.nan,
            resolving_power_mz=r_mz,
            role=run.role, rt_start=float(times.min()) / 60.0, rt_end=float(times.max()) / 60.0,
            drift_start=drift[0] if gated else np.nan, drift_end=drift[1] if gated else np.nan,
            mz_calibration_ppm=offset if offset else np.nan, scans=record)
        return Summed(spectrum, found_profile)


def _tail_checksum(path: Path) -> str | None:
    """The checksum an indexed mzML file carries at its end (``<fileChecksum>``, as
    :func:`hxex.io.identity.file_identity` reads it); None when it has none."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        fh.seek(max(0, size - 4096))
        m = re.search(rb"<fileChecksum>\s*([0-9a-fA-F]{40})\s*</fileChecksum>", fh.read())
    return m.group(1).decode().lower() if m else None


def _at_bins(scan, pred: np.ndarray, half: np.ndarray,
             drift: tuple[float, float] | None) -> float:
    """A scan's intensity at the predicted bins: the tallest point within ``half`` of each,
    summed (``find_peak``'s chromatogram), within the ``drift`` range (ms) when given and the
    scan has its ion mobility bins."""
    if drift is not None and scan.binned is not None:
        m, y, d = scan.binned
        d = drift_times(d)
        keep = (d >= drift[0]) & (d <= drift[1])
        mz, y = sum_mobility_bins(m[keep], y[keep])
    else:
        mz, y = scan.mz, scan.intensity
    if not len(mz):
        return 0.0
    a = np.searchsorted(mz, pred - half)
    z = np.searchsorted(mz, pred + half)
    return float(sum(float(y[i:j].max()) for i, j in zip(a, z, strict=True) if j > i))


def _group_order(target: MzmlTarget) -> tuple:
    """MS1 charge states first (by charge), then fragment channels (by MS level, activation,
    precursor charge)."""
    return (target.ms_level, str(target.activation or ""), str(target.preactivation or ""),
            target.precursor_charge or 0, target.entry.charge)


def _group_label(kind: str, targets: Sequence[MzmlTarget]) -> str:
    """A charge state of the peptide (``3+ (MS1)``) or a fragment channel (``c-1, z-dot, M
    ions: ETD MS2, 3+ precursor``)."""
    t = targets[0]
    if kind == "ion":
        if t.ms_level == 1:
            return f"{t.entry.charge}+ (MS1)"
        return _target_label(t)
    types = []
    for target in targets:
        mod = str(target.entry.modification or "")
        kind_of = next((m.split(":", 1)[1] for m in mod.split() if m.startswith("ion_type:")),
                       "?")
        if kind_of not in types:
            types.append(kind_of)
    how = " ".join(str(x) for x in (
        f"{t.preactivation} then" if t.preactivation else None, t.activation,
        f"MS{t.ms_level}") if x)
    prec = f", {t.precursor_charge}+ precursor" if t.precursor_charge else ""
    return f"{', '.join(types)} ions: {how}{prec}"
