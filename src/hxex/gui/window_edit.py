# SPDX-License-Identifier: GPL-3.0-or-later
"""A spectrum's retention window (and drift range) changed in the review (PLAN Next steps 26).

Without Qt: :func:`preview` sums the spectrum again over the new window
(:meth:`hxex.io.retention_trace.TraceReader.spectrum`), picks it and fits it with the run's
settings, so that the window can draw it beside the spectrum the results hold; and
:func:`save_window` writes the window as the peptide's row for that file (every ion of it:
its charge states and fragments), or as the ion's own row, in the project's
``[input] rt_elution_windows`` table (:mod:`hxex.io.elution_windows`; made, and named in the
project file, by the first save), where the next ``hxex run`` reads it. The peptide table is
never changed, and nothing in the results changes until the project is run again.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..fit import Measurements, fit_models
from ..picking import pick_spectra
from ..spectra import Spectrum, SpectrumSet
from .user_tables import read_table, table_problem, write_table

log = logging.getLogger(__name__)

# the preview's fit starts every K from the run's fit of the spectrum whose window changed and
# takes these few steps (measured on the intact-peptide benchmark, 2026-11-07: 0.6 s per
# spectrum on one thread against 9 s for the run's 6 starts of 400 steps; the same K as the run
# for all 32 spectra over their own windows, and as a full fit for 29 of 32 over windows two
# scans narrower, one of the three a better fit than the full one's). Each K also starts from
# the new spectrum's own k-means start and keeps the better of the two (owner, 2026-11-11: a
# window moved off a co-eluting species left K = 1 on that species' envelope, D 21.9 against
# 0.2 from the spectrum's own start, nll 249 against 63; the two starts take 0.8 s)
PREVIEW_STEPS = {"n_restarts": 1, "kmeans_after_warm": True, "adam_steps": 50,
                 "polish_steps": 10}

# one preview fit at a time: the fit's polish uses PyTorch's forward-mode derivatives, whose
# levels are the process's, so two fits in two threads (a stale preview still running when a
# new one starts) break each other ("no level exists"); the thread count is the process's too
_FIT_LOCK = threading.Lock()


@dataclass(frozen=True)
class Preview:
    """A spectrum summed over another window: the ``spectrum``, its picked ``binned`` row, the
    one-spectrum ``results`` it was fitted into (with the run's settings;
    :func:`hxex.report.plot_spectrum` draws them; None until fitted), the ``k`` the run's
    criterion chooses (None until fitted), ``notes`` on what differs from a run (the tooltip of
    the line above the figure), and whether the fit started from the run's fit of the spectrum
    as well as from the spectrum itself, in a few steps (``warm``)."""

    spectrum: Spectrum
    binned: object  # hxex.picking.BinnedSet, one row
    results: object = None  # hxex.report.Results
    k: int | None = None
    notes: tuple[str, ...] = ()
    warm: bool = False

    @property
    def spectra(self) -> SpectrumSet:
        return SpectrumSet.from_spectra([self.spectrum])


def preview(results, spectrum: Spectrum, *, fit: bool = True, threads: int | None = None,
            start_from: str | None = None) -> Preview:
    """Pick and fit ``spectrum`` (summed over a new window) as the run picked and fitted the
    spectrum it replaces in ``results``: the run's picker settings and bin measure, the series'
    envelope, the fit settings at every K (no screen; on the CPU, with ``threads`` PyTorch
    threads when given: one is faster for a single spectrum) and the run's noise curve. The
    population test, the bootstrap and the controls are not redone (a rerun does them).
    ``fit`` False: picked only (:func:`fit_preview` fits it). ``start_from`` (a spectrum id of
    ``results``) starts the fit from that spectrum's fit and from the new spectrum's own start,
    keeping the better, in a few steps (:data:`PREVIEW_STEPS`); None: the run's own starts and
    steps."""
    picked = _picked(results, spectrum)
    return (fit_preview(results, picked, threads=threads, start_from=start_from) if fit
            else picked)


def _picked(results, spectrum: Spectrum) -> Preview:
    settings = results.settings
    config = settings.run if settings is not None else None
    if config is None:
        raise ValueError("these results do not record the run's settings")
    notes = []
    picker = replace(config.picker, off_mz_sd=None)  # the m/z check needs the whole series
    one = SpectrumSet.from_spectra([spectrum])
    binned = pick_spectra(one, picker, registry=config.registry, measure=settings.measure)
    if config.envelope.mode != "theory":
        notes.append(f"picked on the ion's own isotope peaks (the run's envelope is "
                     f"{config.envelope.mode!r}; its fit shape is kept)")
    if binned.no_signal[0]:
        raise ValueError("no signal in the new window: no bin reaches the picker's threshold")
    return Preview(spectrum, binned, notes=tuple(notes))


def fit_preview(results, picked: Preview, *, threads: int | None = None, cancel=None,
                start_from: str | None = None) -> Preview:
    """``picked`` (:func:`preview` with ``fit=False``) fitted with the run's settings;
    ``cancel`` (:data:`hxex.hooks.CancelHook`) stops a fit no longer wanted
    (:class:`hxex.hooks.Cancelled`); ``start_from`` as in :func:`preview`."""
    import torch

    from ..report import Results

    settings = results.settings
    config = settings.run
    binned, spectrum = picked.binned, picked.spectrum
    notes = list(picked.notes)
    data_id = int(spectrum.data_id)
    shapes = {}
    rows = np.flatnonzero(np.asarray(results.measurements.data_id) == data_id)
    if len(rows):
        shapes[data_id] = results.measurements.shapes[int(rows[0])]
    measurements = Measurements.from_binned(
        binned, n_max_factor=config.n_max_factor, nterm_subtract=config.picker.nterm_subtract,
        amide_rules=config.picker.amide_rules, registry=config.registry, shapes=shapes)
    # every K, on the CPU (one spectrum; the run may have used a GPU this computer lacks)
    fit_config = replace(config.fit, screen=False, backend=replace(config.fit.backend,
                                                                   device="cpu"))
    warm = None
    if start_from is not None:
        warm = _warm(results, start_from, measurements)
        if warm:  # how it was fitted, not how it differs from a run: no note (owner)
            fit_config = replace(fit_config, **PREVIEW_STEPS)
    curve = results.fit.noise_curve
    if curve is None:
        notes.append("noise scaled to the spectrum itself (the run had no noise curve)")
    with _FIT_LOCK:
        before = torch.get_num_threads()
        if threads is not None:
            torch.set_num_threads(threads)
        try:
            fitted = fit_models(measurements, fit_config, noise_curve=curve, cancel=cancel,
                                warm=warm or None)
        finally:
            torch.set_num_threads(before)
    if results.population_test is not None:
        notes.append("K chosen by the criterion; the run chose K by the population test")
    one_results = Results(measurements=measurements, fit=fitted, binned=binned,
                          spectra=picked.spectra, settings=settings, controls=results.controls,
                          figures=results.figures)
    return replace(picked, results=one_results, k=int(fitted.selected_K[0]),
                   notes=tuple(notes), warm=bool(warm))


def _warm(results, spectrum_id: str, measurements) -> dict:
    """The run's fit of ``spectrum_id`` at each K it was fitted at, as starts for the new
    window's spectrum (amplitudes scaled by the new over the old intensity); {} when it was not
    fitted."""
    from ..fit.init import Start

    ids = list(np.asarray(results.fit.spectrum_id))
    if spectrum_id not in ids:
        return {}
    i = ids.index(spectrum_id)
    old = float(np.sum(results.measurements.observed[i]))
    ratio = float(np.sum(measurements.observed[0])) / old if old > 0 else 1.0
    starts = {}
    for k, kf in results.fit.fits.items():
        if not (np.isfinite(kf.A[i]).all() and np.isfinite(kf.n_raw[i]).all()
                and np.isfinite(kf.logit_p[i]).all()):
            continue  # not fitted at this K (the screen, the cut): the fit's own start
        starts[k] = Start(np.log(np.maximum(kf.A[i:i + 1], 1e-300) * max(ratio, 1e-12)),
                          kf.n_raw[i:i + 1], kf.logit_p[i:i + 1])
    return starts


# --- the elution-window table ----------------------------------------------------------------

DEFAULT_NAME = "rt_elution_windows.csv"  # beside the project file, when it names no table


@dataclass(frozen=True)
class SavedWindow:
    """A window written to the project's ``[input] rt_elution_windows`` table: each file
    written with its bytes ``before`` (None: it did not exist) and ``after`` (Undo writes
    ``before`` back), the row as written (``row``, by column; the peptide's row for a window
    saved for the whole peptide), whether it was ``added`` (else that row was changed), and,
    for a peptide's window, ``own``: the other ions of the peptide whose rows in that file keep
    a window of their own. The table is the first file; the project file follows when the first
    save named the table in it."""

    files: tuple[tuple[Path, bytes | None, bytes], ...]
    row: dict[str, str]
    added: bool
    own: tuple[str, ...] = ()
    samples: tuple[str, ...] = ()  # every sample a row was written for (the key's first)

    @property
    def path(self) -> Path:
        return self.files[0][0]


def _text(value: float, places: int, up: bool) -> str:
    """``value`` to ``places`` decimals, rounded outwards (``up``: towards larger), so that a
    window written to the table still holds the scans it summed."""
    scale = 10 ** places
    v = (math.ceil if up else math.floor)(value * scale - (1e-6 if up else -1e-6)) / scale
    return f"{v:.{places}f}"


def table_path(project_path: str | Path) -> tuple[Path, bool]:
    """The project's elution-window table and whether the project file names it already (else
    the default name beside the project file, named there by the first save)."""
    from ..config import load_project

    project_path = Path(project_path)
    named = load_project(project_path).input.rt_elution_windows
    return (Path(named), True) if named is not None else (project_path.parent / DEFAULT_NAME,
                                                          False)


# --- the words for what a window is saved for (owner, 2026-11-11: "run" and "ion" were the
# fragment projects' words; for a peptide project "ion" means a charge state, and "run" is a
# file, while hxex also says run for hxex run and for an analysis's separate projects)


def has_fragments(source) -> bool:
    """Whether a read record (or a :class:`~hxex.io.retention_trace.TraceReader`) reads fragment
    ions (targets beyond MS1): the choices then name ions, else charge states."""
    targets = getattr(source, "targets", None)
    if targets is None:
        targets = (source or {}).get("targets") or ()
    return any(int((t.get("ms_level") if isinstance(t, dict) else t.ms_level) or 1) > 1
               for t in targets)


def charge_of(ion: str) -> int | None:
    """The charge an MS1 ion label (``0129-0151-SEQUENCE-z3``) ends in; None for another."""
    m = re.search(r"-z(\d+)$", ion)
    return int(m.group(1)) if m else None


def scope_words(ion: str, fragments: bool) -> dict[str, str]:
    """The window editor's two choices (by their data: ``peptide`` and ``ion``) for the
    selected ``ion`` (as spectrum ids name it), worded for a project with or without fragment
    ions."""
    if fragments:
        return {"ion": "this ion only, in this file",
                "peptide": "this peptide, its charge states and fragment ions, in this file",
                "replicates": "this peptide and its ions, in this file and its replicates "
                              "(each shifted)",
                "everywhere": "this peptide and its ions, in every file of the peptide "
                              "(each shifted)"}
    z = charge_of(ion)
    return {"ion": (f"this charge state only (z = {z}), in this file" if z
                    else "this charge state only, in this file"),
            "peptide": "every charge state of this peptide, in this file",
            "replicates": "every charge state, in this file and its replicates (each shifted)",
            "everywhere": "every charge state, in every file of the peptide (each shifted)"}


def compare_words(fragments: bool) -> dict[str, str]:
    """The elution tab's Compare with choices, one for each save level beyond this charge
    state alone (``ions``: level 2, ``replicates``: 3, ``everywhere``: 4; owner, 2026-11-11),
    worded as :func:`scope_words`."""
    one, others = ("this ion", "other ions") if fragments else ("this charge state",
                                                               "other charge states")
    return {"ions": f"this peptide's {others} in this file",
            "replicates": f"{one} in its replicates",
            "everywhere": f"{one} in every file of the peptide"}


# the save level each Compare with view shows the evidence for (and the reverse)
COMPARE_FOR_LEVEL = {"peptide": "ions", "replicates": "replicates", "everywhere": "everywhere"}


def saved_words(saved, *, peptide: str, ion: str, file: str, whole: bool, fragments: bool,
                drift_written: bool) -> str:
    """What a save wrote, for the status bar: the table, what the window is for, the file and
    every sample it was written for, and the drift range's own row when it has one."""
    if whole:
        what = (f"peptide {peptide} (its charge states and fragment ions)" if fragments
                else f"peptide {peptide} (every charge state)")
    else:
        z = None if fragments else charge_of(ion)
        what = f"{peptide} z = {z} only" if z else f"{ion} only"
    samples = saved.samples or ()
    whose = (f"sample {samples[0]}" if len(samples) <= 1
             else f"samples {', '.join(samples)} (the file is read in each)")
    action = "added" if saved.added else "changed"
    text = (f"{saved.path.name}: window for {what} in {file} {action} "
            f"({saved.row.get('rt_start')}-{saved.row.get('rt_end')} min), for {whose}")
    if whole and drift_written:
        own = "this ion" if fragments else f"z = {charge_of(ion)}"
        text += (f"; the drift range is saved for {own} only (drift times differ between "
                 "charge states)")
    if saved.own:
        text += f"; {len(saved.own)} other ion(s) of it keep a window of their own"
    if len(saved.files) > 1:
        text += "; named in the project file as [input] rt_elution_windows"
    return f"{text}; {APPLY}"


def samples_reading(reader, spectrum_id: str, data_id: int, ions: Sequence[str],
                    peptide: str | None = None) -> tuple[str, ...]:
    """The samples that read the file of ``spectrum_id`` (series ``data_id``; the same recorded
    path, :class:`hxex.io.retention_trace.TraceReader` ``reader``) and one of ``ions`` (as the
    spectrum ids name them; of ``peptide`` when given) from it, the spectrum's own sample first:
    the samples a window saved from it is written for. A file read in several samples (controls
    shared by the samples of a state) is one injection, so one window. () when the id names no
    run of the record."""
    from ..io.elution_windows import peptide_label
    from ..io.mzml import _target_label, target_applies

    where = reader.locate(spectrum_id, data_id)
    if where is None:
        return ()
    run = reader.runs[where[0]]
    wanted = set(ions)
    out = [run.sample]
    for other in reader.runs:
        if other.sample in out or not _same_path(other.path, run.path):
            continue
        if any(_target_label(t) in wanted and target_applies(t, other)
               and (peptide is None or peptide_label(t.entry) == peptide)
               for t in reader.targets):
            out.append(other.sample)
    return tuple(out)


# how a saved window reaches the results, said wherever a window is saved or shown (owner,
# 2026-11-11: "requires a new run" did not say how, nor that it must replace the results)
APPLY = ("to use it, run the project again with overwrite enabled: Run on the Run tab with "
         "[output] overwrite = true, or Replace when it asks (hxex run --overwrite); only the "
         "series whose spectra changed are fitted again, and your review decisions are kept")


# --- a window spread to other files (owner, 2026-11-11: four levels for MS1; this spectrum,
# this file's charge states, the replicates at its time point or control, the whole peptide)

SPREAD_LEVELS = ("ion", "peptide", "replicates", "everywhere")
SHIFT_SEARCH = 0.5  # minutes either way the trace match looks (30 s)
MATCH_FLOOR = 0.5  # the matched traces' correlation below which a shift is not trusted
_GRID = 0.002  # minutes: the common time grid of a match


@dataclass(frozen=True)
class FilePlan:
    """Where a window spread to another file goes: the ``file`` (its name) of run ``path``,
    the ``samples`` that read it, the ``shift`` (minutes; the file's elution against the edited
    file's, from their traces of the charge state) and the shifted ``window``; ``why`` (and no
    window) when the match is not trusted. ``correlation`` is the match's; ``trace`` the
    file's trace of the charge state the match used (:class:`hxex.io.retention_trace.Trace`;
    None when it could not be read)."""

    file: str
    path: str
    samples: tuple[str, ...]
    shift: float | None
    window: tuple[float, float] | None
    correlation: float | None = None
    why: str | None = None
    trace: object = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Alignment:
    """The other files a window saved at ``level`` reaches, each matched against the edited
    file's ``reference`` trace (within ``drift``, ms, when given) by :func:`align_files`:
    ``plans`` hold each file's shift (their ``window`` is None until :meth:`for_window`).
    ``spectrum_id`` is the edited spectrum's."""

    spectrum_id: str
    level: str
    drift: tuple[float, float] | None
    reference: object
    plans: tuple[FilePlan, ...]

    def for_window(self, window: tuple[float, float]) -> list[FilePlan]:
        """The plans with ``window`` (minutes, the edited file's clock) shifted into each file."""
        return [replace(p, window=None if p.shift is None
                        else (window[0] + p.shift, window[1] + p.shift)) for p in self.plans]


def trace_shift(ref_times, ref_intensity, times, intensity,
                search: float = SHIFT_SEARCH) -> tuple[float | None, float | None, str | None]:
    """How far (minutes) one trace's elution sits from a reference's: the lag of the largest
    correlation of the two (each less its minimum) within ``search`` either way, as (shift,
    correlation, None), or (None, correlation, why) when the match is not trusted: no signal,
    a best lag at the search's edge, or a correlation below :data:`MATCH_FLOOR`."""
    ref_t, ref_i = np.asarray(ref_times, float), np.asarray(ref_intensity, float)
    t, i = np.asarray(times, float), np.asarray(intensity, float)
    if len(ref_t) < 3 or len(t) < 3 or np.ptp(i) <= 0 or np.ptp(ref_i) <= 0:
        return None, None, "no signal in its trace"
    grid = np.arange(min(ref_t.min(), t.min()) - search, max(ref_t.max(), t.max()) + search,
                     _GRID)
    a = np.interp(grid, ref_t, ref_i - ref_i.min(), left=np.nan, right=np.nan)
    b = np.interp(grid, t, i - i.min(), left=np.nan, right=np.nan)
    steps = int(round(search / _GRID))
    best = (-np.inf, 0)
    for lag in range(-steps, steps + 1):  # b(t + lag) against a(t)
        x = a[max(0, -lag):len(a) - max(0, lag)]
        y = b[max(0, lag):len(b) - max(0, -lag)]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10 or np.ptp(x[ok]) <= 0 or np.ptp(y[ok]) <= 0:
            continue
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        if r > best[0]:
            best = (r, lag)
    r, lag = best
    if not np.isfinite(r):
        return None, None, "its trace does not overlap"
    if abs(lag) >= steps - 1:
        return None, r, f"the best match is at the search's edge ({search * 60:.0f} s)"
    if r < MATCH_FLOOR:
        return None, r, f"the traces do not match (correlation {r:.2f})"
    return lag * _GRID, r, None


def spread_files(reader, results, spectrum_id: str, data_id: int, *, peptide: str, ion: str,
                 level: str) -> list[tuple[str, int]]:
    """The other files a window saved at ``level`` (``replicates``: the spectrum's sample's
    files at its time point or control; ``everywhere``: every file of every sample that reads
    the charge state ``ion`` of ``peptide``) reaches, as one spectrum (id, series) of ``ion``
    per file: from the spectra ``results`` fitted (their ids and series), the edited file left
    out. A file without a spectrum of this charge state is not listed."""
    from ..io.elution_windows import peptide_label
    from ..spectra import split_spectrum_id

    if level not in ("replicates", "everywhere"):
        return []
    where = reader.locate(spectrum_id, data_id)
    if where is None:
        return []
    here = reader.runs[where[0]]
    out, seen = [], {Path(str(here.path)).as_posix()}
    ids = np.asarray(results.fit.spectrum_id).astype(str)
    series = np.asarray(results.measurements.data_id)
    for sid, d in zip(ids, series, strict=True):
        try:
            local = split_spectrum_id(sid)[1]
        except ValueError:
            continue
        parts = local.split("/")
        if len(parts) < 3 or parts[2] != ion:
            continue
        d = int(d)
        if d not in reader.of_series:
            continue
        if peptide_label(reader.targets[reader.of_series[d]].entry) != peptide:
            continue
        found = reader.locate(sid, d)
        if found is None:
            continue
        run = reader.runs[found[0]]
        key = Path(str(run.path)).as_posix()
        if key in seen:
            continue
        if level == "replicates" and not (run.sample == here.sample and run.role == here.role
                                          and run.time == here.time):
            continue
        seen.add(key)
        out.append((sid, d))
    return out


def plan_spread(reader, results, spectrum_id: str, data_id: int,
                window: tuple[float, float], *, peptide: str, ion: str, ions: Sequence[str],
                level: str, drift: tuple[float, float] | None = None,
                cancel=None) -> list[FilePlan]:
    """Each other file a window saved at ``level`` reaches (:func:`spread_files`), with the
    window shifted by that file's elution against the edited file's (:func:`align_files`), or
    why not."""
    aligned = align_files(reader, results, spectrum_id, data_id, peptide=peptide, ion=ion,
                          ions=ions, level=level, drift=drift, cancel=cancel)
    return [] if aligned is None else aligned.for_window(window)


def align_files(reader, results, spectrum_id: str, data_id: int, *, peptide: str, ion: str,
                ions: Sequence[str], level: str, drift: tuple[float, float] | None = None,
                cancel=None) -> Alignment | None:
    """Each other file a window saved at ``level`` reaches (:func:`spread_files`), matched
    against the edited file's elution (:func:`trace_shift`, the charge state's traces within
    ``drift`` (ms) when given: the species chosen in drift time, not a brighter one at another
    drift time on the same m/z, which elutes with shifts of its own; owner's case, 2026-11-11:
    fully deuterated files 4-8 s further than over all drift times); None when no other file
    has a spectrum of it. ``ions`` (the peptide's, as spectrum ids name them) decide which
    samples read each file. The elution tab draws it (the files aligned, or one row each) and
    a save at that level uses it."""
    from ..hooks import Cancelled
    from ..io.mzml import run_name

    others = spread_files(reader, results, spectrum_id, data_id, peptide=peptide, ion=ion,
                          level=level)
    if not others:
        return None
    ranges = None if drift is None else {spectrum_id: drift}
    ref = reader.traces([spectrum_id], data_id, drift_ranges=ranges).traces
    if not ref:
        raise ValueError("the edited spectrum's trace could not be read")
    ref = ref[0]
    plans = []
    for sid, d in others:
        if cancel is not None and cancel():
            raise Cancelled("the window's spread was not wanted any more")
        found = reader.locate(sid, d)
        run = reader.runs[found[0]]
        file, samples = run_name(run), samples_reading(reader, sid, d, ions, peptide)
        got = reader.traces([sid], d,
                            drift_ranges=None if drift is None else {sid: drift}).traces
        if not got:
            plans.append(FilePlan(file, str(run.path), samples, None, None,
                                  why="its trace could not be read"))
            continue
        trace = replace(got[0], selected=False)
        shift, r, why = trace_shift(ref.times, ref.intensity, trace.times, trace.intensity)
        plans.append(FilePlan(file, str(run.path), samples, shift, None, r, why, trace=trace))
    return Alignment(spectrum_id, level, drift, ref, tuple(plans))


def spread_words(plans: Sequence[FilePlan], kept: Sequence[str] = ()) -> str:
    """The files a spread reached, each with its shift, those it did not (and why), and those
    that kept a window of their own (``kept``, file names), for the status bar."""
    done = [p for p in plans if p.window is not None and p.file not in kept]
    out = []
    if len(done) > 4:  # the files one by one in the log; their range here
        shifts = [p.shift * 60 for p in done]
        log.info("window spread: %s", ", ".join(f"{p.file} {p.shift * 60:+.1f} s" for p in done))
        out.append(f"spread to {len(done)} files, shifted {min(shifts):+.0f} to "
                   f"{max(shifts):+.0f} s (each in the log)")
    elif done:
        out.append(f"spread to {len(done)} file(s): " + ", ".join(
            f"{p.file} {p.shift * 60:+.0f} s" for p in done))
    missed = [p for p in plans if p.window is None]
    if missed:
        out.append(f"not spread to {len(missed)} (set them by hand): " + ", ".join(
            f"{p.file} ({p.why})" for p in missed))
    if kept:
        out.append(f"kept their own window: {', '.join(kept)}")
    return "; ".join(out)


def own_windows(project_path, peptide: str, ions: Sequence[str],
                places: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Which ``places`` (sample, file) already hold a window of the peptide's own in the
    project's elution-window table: a row naming the peptide, or one of its ``ions``, with a
    window or a drift range (set by hand or by an earlier save); a spread asks before replacing
    them (owner, 2026-11-11)."""
    path, _named = table_path(project_path)
    if not path.is_file():
        return []
    table = read_table(path)
    names = [c.strip().lower() for c in table.columns]
    if not {"sample", "file"} <= set(names):
        return []

    def cell(r, column):
        return r[names.index(column)].strip() if column in names else ""

    wanted, out = set(places), []
    mine = set(ions)
    for r in table.rows:
        place = (cell(r, "sample"), cell(r, "file"))
        if place not in wanted or place in out:
            continue
        pep, ion = cell(r, "peptide"), cell(r, "ion")
        if not (pep == peptide or (ion in mine and pep in ("", peptide))):
            continue
        if any(cell(r, c) for c in ("rt_start", "rt_end", "drift_start", "drift_end")):
            out.append(place)
    return out


def _same_path(a, b) -> bool:
    """Whether two recorded run paths name the same file (as recorded: a project records its
    runs' paths one way)."""
    return Path(str(a)).as_posix() == Path(str(b)).as_posix()


def save_window(project_path: str | Path, key: tuple[str, str, str],
                window: tuple[float, float], drift: tuple[float, float] | None = None, *,
                peptide: str | None = None, whole: bool = False, siblings=(),
                shared: bool = False, table: str | Path | None = None,
                samples: Sequence[str] = (),
                more: Sequence[tuple[str, str, tuple[float, float]]] = ()) -> SavedWindow:
    """Write ``window`` (minutes: the first and last scan of the spectrum summed again over it)
    to the project's ``[input] rt_elution_windows`` table for ``key`` (sample, file name, ion
    as the spectrum ids name it) of ``peptide`` (as :func:`hxex.io.elution_windows
    .peptide_label` names the ion's peptide).

    ``whole``: the window is the peptide's row in that file: every ion of it takes it (its
    charge states and fragments; the same molecules elute together), and the ion's own row
    gives up its window (it is removed, or keeps only its drift range; a row naming the ion
    without a peptide only when no other peptide has an ion of that name, ``shared`` False).
    ``siblings`` (the other ions of the peptide, as spectrum ids name them) are looked up for
    :attr:`SavedWindow.own`. ``drift`` (ms) is then written to the ion's own row only (drift
    times differ between charge states): give it only when it was changed. Otherwise the row is
    the ion's own, naming its peptide too (two parents can give fragment ions of the same
    name), with ``drift``: an exception, such as interference at its own m/z.

    ``samples``: the other samples that read the same file (:func:`samples_reading`; one
    injection, so one window, owner 2026-11-11): the same rows are written for each of them, as
    the table keys its rows by sample. ``more``: (sample, file, window) of other files the window
    is spread to (:func:`plan_spread`, shifted by each file's elution), written as ``whole``
    rows are (the peptide's row; ``drift`` on the ion's own row), and there every ion of the
    peptide gives up a window of its own (the window asks first, :func:`own_windows`).

    The row is changed, or one is added. When the project file names no table, the first save
    writes ``table`` (the window asks where, :func:`hxex.gui.table_choice.ask_table`; else
    :data:`DEFAULT_NAME` beside the project file), making it or adding to it, and names it in
    the project file. The
    times are written rounded outwards (4 decimals of a minute, 3 of a millisecond), so the run
    reads the same scans. The table is checked as the project reads it before anything is
    written (ValueError with the reader's words; nothing written). The peptide table is never
    changed."""
    from ..config import _toml_value
    from ..io.elution_windows import COLUMNS
    from ..project_edit import set_settings
    from .user_tables import Table

    if whole and not peptide:
        raise ValueError("a window for the whole peptide needs the peptide")
    project_path = Path(project_path)
    path, named = table_path(project_path)
    if not named and table is not None:
        path = Path(table)
    before = path.read_bytes() if path.is_file() else None
    table = read_table(path) if before is not None else Table(list(COLUMNS))
    table = table.copy()
    names = [c.strip().lower() for c in table.columns]
    for column in COLUMNS:
        if column not in names:
            table.columns.append(column)
            names.append(column)
            for r in table.rows:
                r.append("")
    at = {c: names.index(c) for c in COLUMNS}
    first, file, ion = key
    pep = peptide or ""
    written = tuple(dict.fromkeys([first, *samples]))

    def cell(r, column):
        return r[at[column]].strip()

    def span(row, values, places, start, end):
        row[at[start]] = "" if values is None else _text(values[0], places, up=False)
        row[at[end]] = "" if values is None else _text(values[1], places, up=True)

    def write_for(sample, file=file, window=window, everyone=False):
        def find(pep_value, ion_value):
            return [r for r in table.rows if (cell(r, "sample"), cell(r, "file"),
                                              cell(r, "peptide"), cell(r, "ion"))
                    == (sample, file, pep_value, ion_value)]

        def put(pep_value, ion_value):
            own = find(pep_value, ion_value)
            if own:
                return own[0], False
            row = [""] * len(table.columns)
            row[at["sample"]], row[at["file"]] = sample, file
            row[at["peptide"]], row[at["ion"]] = pep_value, ion_value
            table.rows.append(row)
            return row, True

        own_rows: set[str] = set()
        if whole:
            row, added = put(pep, "")
            span(row, window, 4, "rt_start", "rt_end")
            mine = find(pep, ion) + ([] if shared else find("", ion))
            followers = mine + ([r for s_ in siblings if s_ != ion
                                 for r in find(pep, s_) + ([] if shared else find("", s_))]
                                if everyone else [])
            for r in followers:  # the ion (in a spread file, every ion) follows its peptide
                span(r, None, 4, "rt_start", "rt_end")
            if drift is not None:
                target, _ = (mine[0], False) if mine else put(pep, ion)
                span(target, drift, 3, "drift_start", "drift_end")
            table.rows[:] = [r for r in table.rows if not cell(r, "ion") or any(
                cell(r, c) for c in ("rt_start", "rt_end", "drift_start", "drift_end"))]
            own_rows = {cell(r, "ion") for s_ in siblings for r in find(pep, s_) + find("", s_)
                        if cell(r, "rt_start") and (s_ != ion or r not in mine)}
        else:
            row, added = put(pep, ion)
            span(row, window, 4, "rt_start", "rt_end")
            span(row, drift, 3, "drift_start", "drift_end")
        return row, added, own_rows

    row, added, own_set = write_for(first)
    for sample in written[1:]:
        own_set |= write_for(sample)[2]
    if more and not (whole and peptide):
        raise ValueError("a window spread to other files is the whole peptide's")
    for sample, other_file, other_window in more:
        write_for(sample, other_file, other_window, everyone=True)
    own_rows = tuple(sorted(own_set))
    problem = table_problem("input.rt_elution_windows", table, project_path)
    if problem:
        raise ValueError(problem)
    project_before = project_path.read_bytes()
    write_table(table, path)
    files = [(path, before, path.read_bytes())]
    if not named:
        try:
            try:  # relative to the project's folder when inside it (forward slashes)
                shown = path.resolve().relative_to(project_path.parent.resolve()).as_posix()
            except ValueError:
                shown = str(path)
            set_settings(project_path, [("input.rt_elution_windows", _toml_value(shown))],
                         unfinished=True)
        except Exception:
            _restore(path, before)
            raise
        files.append((project_path, project_before, project_path.read_bytes()))
    return SavedWindow(tuple(files), {c: v for c, v in zip(table.columns, row, strict=True)},
                       added, own_rows, written)


def _restore(path: Path, before: bytes | None) -> None:
    from ..project_edit import replace_file

    if before is None:
        path.unlink(missing_ok=True)
        return
    tmp = path.with_name(path.name + ".hxex-tmp")
    tmp.write_bytes(before)
    replace_file(tmp, path)


def undo_window(saved: SavedWindow) -> None:
    """Write the files ``saved`` changed back as they were (the table removed when the save
    made it; ValueError when one was changed since, by hand or by another program: nothing is
    written)."""
    for path, _before, after in saved.files:
        if not path.is_file() or path.read_bytes() != after:
            raise ValueError(f"{path} was changed since the window was saved: not undone")
    for path, before, _after in saved.files:
        _restore(path, before)


def saved_window(project_path: str | Path, key: tuple[str, str, str],
                 peptide: str | None = None
                 ) -> tuple[tuple[float, float], tuple[float, float] | None, str] | None:
    """The window (minutes), drift range (ms, or None) and where the window comes from
    (``"ion"``: the ion's own row; ``"peptide"``: its peptide's row) that the project's
    elution-window table gives ``key`` (sample, file, ion) of ``peptide`` (as
    :func:`hxex.io.elution_windows.peptide_label` names it); None without a table or a row with
    a window for it."""
    from ..io.elution_windows import by_key, by_peptide, read_elution_windows, window_for

    path, _named = table_path(project_path)
    if not path.is_file():
        return None
    rows = read_elution_windows(path)
    sample, file, ion = key
    window, drift, used = window_for(ion, peptide or "", sample, file, by_key(rows),
                                     by_peptide(rows))
    if window is None:
        return None
    return window, drift, ("peptide" if used[0].is_peptide else "ion")
