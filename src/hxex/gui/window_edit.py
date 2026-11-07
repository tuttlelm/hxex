# SPDX-License-Identifier: GPL-3.0-or-later
"""A spectrum's retention window (and drift range) changed in the review (PLAN Next steps 26).

Without Qt: :func:`preview` sums the spectrum again over the new window
(:meth:`hxex.io.retention_trace.TraceReader.spectrum`), picks it and fits it with the run's
settings, so that the window can draw it beside the spectrum the results hold; and
:func:`save_window` writes the window as the run's own row of the project's peptide table (the
``run`` column with ``rt_start``, ``rt_end`` and, for an ion-mobility file, ``drift_start``,
``drift_end``), where the next ``hxex run`` reads it. Nothing in the results changes until then.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..fit import Measurements, fit_models
from ..picking import pick_spectra
from ..spectra import Spectrum, SpectrumSet
from .user_tables import read_table, table_problem, write_table

# the preview's fit starts every K from the run's fit of the spectrum whose window changed and
# takes these few steps (measured on the intact-peptide benchmark, 2026-11-07: 0.6 s per
# spectrum on one thread against 9 s for the run's 6 starts of 400 steps; the same K as the run
# for all 32 spectra over their own windows, and as a full fit for 29 of 32 over windows two
# scans narrower, one of the three a better fit than the full one's)
PREVIEW_STEPS = {"n_restarts": 1, "kmeans_after_warm": False, "adam_steps": 50,
                 "polish_steps": 10}

# one preview fit at a time: the fit's polish uses PyTorch's forward-mode derivatives, whose
# levels are the process's, so two fits in two threads (a stale preview still running when a
# new one starts) break each other ("no level exists"); the thread count is the process's too
_FIT_LOCK = threading.Lock()

# the peptide table's columns a run's own row is matched by (the reader's names, any case)
_KEY = ("peptide", "start_seq", "end_seq", "charge", "modification")


@dataclass(frozen=True)
class Preview:
    """A spectrum summed over another window: the ``spectrum``, its picked ``binned`` row, the
    one-spectrum ``results`` it was fitted into (with the run's settings;
    :func:`hxex.report.plot_spectrum` draws them; None until fitted), the ``k`` the run's
    criterion chooses (None until fitted), and ``notes`` on what differs from a run (said under
    the figure)."""

    spectrum: Spectrum
    binned: object  # hxex.picking.BinnedSet, one row
    results: object = None  # hxex.report.Results
    k: int | None = None
    notes: tuple[str, ...] = ()

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
    ``results``) starts the fit from that spectrum's fit, in a few steps
    (:data:`PREVIEW_STEPS`); None: the run's own starts and steps."""
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
        if warm:
            fit_config = replace(fit_config, **PREVIEW_STEPS)
            notes.append("fitted from the run's fit of the spectrum, in a few steps")
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
                   notes=tuple(notes))


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


# --- the peptide table -------------------------------------------------------------------------


@dataclass(frozen=True)
class SavedWindow:
    """A window written to the peptide table at ``path``: the file's bytes ``before`` and
    ``after`` (Undo writes ``before`` back), the run's row as written (``row``, by column),
    and whether it was ``added`` (else an existing row of the run was changed)."""

    path: Path
    before: bytes
    after: bytes
    row: dict[str, str]
    added: bool


def _text(value: float, places: int, up: bool) -> str:
    """``value`` to ``places`` decimals, rounded outwards (``up``: towards larger), so that a
    window written to the table still holds the scans it summed."""
    scale = 10 ** places
    v = (math.ceil if up else math.floor)(value * scale - (1e-6 if up else -1e-6)) / scale
    return f"{v:.{places}f}"


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _same(a: str, b: str, numeric: bool) -> bool:
    if numeric:
        x, y = _number(a.strip()), _number(b.strip())
        return x is not None and x == y
    return a.strip() == b.strip()


def save_window(project_path: str | Path, peptides: str | Path, spectrum: Spectrum, run: str,
                window: tuple[float, float], drift: tuple[float, float] | None = None
                ) -> SavedWindow:
    """Write ``window`` (minutes: the first and last scan of ``spectrum``, summed again over
    it) as the row of run ``run`` (its file name) in the peptide table ``peptides`` of the
    project at ``project_path``: the run's existing row for the peptide ion is changed, else
    a copy of the peptide's general row (its sample's, or the one for every run) is added with
    the run's name. ``drift`` (ms) is written for an ion-mobility file. The times are written
    rounded outwards (4 decimals of a minute, 3 of a millisecond), so the run reads the same
    scans. The table is checked as the project reads it before it is written (ValueError with
    the reader's words, nothing written)."""
    path = Path(peptides)
    before = path.read_bytes()
    table = read_table(path)
    names = [c.strip().lower() for c in table.columns]
    if not all(k in names for k in _KEY[:4]):
        raise ValueError(f"{path} has no {', '.join(k for k in _KEY[:4] if k not in names)} "
                         "column")
    e = spectrum.entry
    want = {"peptide": e.peptide, "start_seq": str(e.start_seq), "end_seq": str(e.end_seq),
            "charge": str(e.charge), "modification": e.modification or ""}

    def cell(row: list[str], name: str) -> str:
        return row[names.index(name)] if name in names else ""

    def is_ion(row: list[str]) -> bool:
        return all(_same(cell(row, k), want[k], k in ("start_seq", "end_seq", "charge"))
                   for k in _KEY if k in names or want[k])

    table = table.copy()
    for column in ("run", "rt_start", "rt_end", *(("drift_start", "drift_end") if drift
                                                  else ())):
        if column not in names:
            table.columns.append(column)
            names.append(column)
            for r in table.rows:
                r.append("")
    own = [r for r in table.rows if is_ion(r) and cell(r, "run").strip() == run]
    added = not own
    if own:
        row = own[0]
    else:
        general = [r for r in table.rows if is_ion(r) and not cell(r, "run").strip()
                   and cell(r, "sample").strip() in ("", e.sample)]
        if not general:
            raise ValueError(f"{path}: no row of {e.peptide} (charge {e.charge}, residues "
                             f"{e.start_seq}-{e.end_seq}) to copy for run {run}")
        # the sample's own row before the one for every run
        general.sort(key=lambda r: cell(r, "sample").strip() != e.sample)
        row = list(general[0])
        table.rows.insert(table.rows.index(general[0]) + 1, row)
    row[names.index("run")] = run
    row[names.index("rt_start")] = _text(window[0], 4, up=False)
    row[names.index("rt_end")] = _text(window[1], 4, up=True)
    if drift:
        row[names.index("drift_start")] = _text(drift[0], 3, up=False)
        row[names.index("drift_end")] = _text(drift[1], 3, up=True)
    problem = table_problem("input.peptides", table, project_path)
    if problem:
        raise ValueError(problem)
    write_table(table, path)
    return SavedWindow(path, before, path.read_bytes(),
                       {c: v for c, v in zip(table.columns, row, strict=True)}, added)


def undo_window(saved: SavedWindow) -> None:
    """Write the peptide table back as it was before ``saved`` (ValueError when it was changed
    since, by hand or by another program: nothing is written)."""
    if saved.path.read_bytes() != saved.after:
        raise ValueError(f"{saved.path} was changed since the window was saved: not undone")
    tmp = saved.path.with_name(saved.path.name + ".hxex-tmp")
    tmp.write_bytes(saved.before)
    from ..project_edit import replace_file

    replace_file(tmp, saved.path)


def saved_window(peptides: str | Path, entry, run: str
                 ) -> tuple[tuple[float, float], tuple[float, float] | None] | None:
    """The window (minutes) and drift range (ms, or None) of run ``run``'s own row for the
    peptide ion ``entry`` in the peptide table ``peptides``; None without such a row (or a
    window in it)."""
    table = read_table(peptides)
    names = [c.strip().lower() for c in table.columns]
    if "run" not in names:
        return None
    want = {"peptide": entry.peptide, "start_seq": str(entry.start_seq),
            "end_seq": str(entry.end_seq), "charge": str(entry.charge),
            "modification": entry.modification or ""}
    for row in table.rows:
        def cell(name: str, row=row) -> str:
            return row[names.index(name)].strip() if name in names else ""

        if cell("run") != run or not all(
                _same(cell(k), want[k], k in ("start_seq", "end_seq", "charge"))
                for k in _KEY if k in names or want[k]):
            continue
        start, end = _number(cell("rt_start")), _number(cell("rt_end"))
        if start is None or end is None:
            return None
        lo, hi = _number(cell("drift_start")), _number(cell("drift_end"))
        return (start, end), (None if lo is None or hi is None else (lo, hi))
    return None
