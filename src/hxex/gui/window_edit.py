# SPDX-License-Identifier: GPL-3.0-or-later
"""A spectrum's retention window (and drift range) changed in the review (PLAN Next steps 26).

Without Qt: :func:`preview` sums the spectrum again over the new window
(:meth:`hxex.io.retention_trace.TraceReader.spectrum`), picks it and fits it with the run's
settings, so that the window can draw it beside the spectrum the results hold; and
:func:`save_window` writes the window as the ion's row for that file in the project's
``[input] rt_elution_windows`` table (:mod:`hxex.io.elution_windows`; made, and named in the
project file, by the first save), where the next ``hxex run`` reads it. The peptide table is
never changed, and nothing in the results changes until the project is run again.
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


# --- the elution-window table ----------------------------------------------------------------

DEFAULT_NAME = "rt_elution_windows.csv"  # beside the project file, when it names no table


@dataclass(frozen=True)
class SavedWindow:
    """A window written to the project's ``[input] rt_elution_windows`` table: each file
    written with its bytes ``before`` (None: it did not exist) and ``after`` (Undo writes
    ``before`` back), the row as written (``row``, by column), and whether it was ``added``
    (else the ion's row in that file was changed). The table is the first file; the project file
    follows when the first save named the table in it."""

    files: tuple[tuple[Path, bytes | None, bytes], ...]
    row: dict[str, str]
    added: bool

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


def save_window(project_path: str | Path, key: tuple[str, str, str],
                window: tuple[float, float], drift: tuple[float, float] | None = None
                ) -> SavedWindow:
    """Write ``window`` (minutes: the first and last scan of the spectrum summed again over it)
    and ``drift`` (ms, ion-mobility files) as the row of ``key`` (sample, file name, ion as the
    spectrum ids name it) in the project's ``[input] rt_elution_windows`` table: that row is
    changed, or one is added. The first save makes the table (:data:`DEFAULT_NAME`, beside the
    project file) and names it in the project file. The times are written rounded outwards (4
    decimals of a minute, 3 of a millisecond), so the run reads the same scans. The table is
    checked as the project reads it before anything is written (ValueError with the reader's
    words; nothing written). The peptide table is never changed."""
    from ..io.elution_windows import COLUMNS
    from ..project_edit import set_settings
    from .user_tables import Table

    project_path = Path(project_path)
    path, named = table_path(project_path)
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
    sample, file, ion = key
    own = [r for r in table.rows if (r[at["sample"]].strip(), r[at["file"]].strip(),
                                     r[at["ion"]].strip()) == key]
    added = not own
    if own:
        row = own[0]
    else:
        row = [""] * len(table.columns)
        row[at["sample"]], row[at["file"]], row[at["ion"]] = sample, file, ion
        table.rows.append(row)
    row[at["rt_start"]] = _text(window[0], 4, up=False)
    row[at["rt_end"]] = _text(window[1], 4, up=True)
    row[at["drift_start"]] = "" if drift is None else _text(drift[0], 3, up=False)
    row[at["drift_end"]] = "" if drift is None else _text(drift[1], 3, up=True)
    problem = table_problem("input.rt_elution_windows", table, project_path)
    if problem:
        raise ValueError(problem)
    project_before = project_path.read_bytes()
    write_table(table, path)
    files = [(path, before, path.read_bytes())]
    if not named:
        try:
            set_settings(project_path, [("input.rt_elution_windows",
                                         f'"{DEFAULT_NAME}"')], unfinished=True)
        except Exception:
            _restore(path, before)
            raise
        files.append((project_path, project_before, project_path.read_bytes()))
    return SavedWindow(tuple(files), {c: v for c, v in zip(table.columns, row, strict=True)},
                       added)


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


def saved_window(project_path: str | Path, key: tuple[str, str, str]
                 ) -> tuple[tuple[float, float], tuple[float, float] | None] | None:
    """The window (minutes) and drift range (ms, or None) the project's elution-window table
    gives ``key`` (sample, file, ion); None without a table or such a row."""
    from ..io.elution_windows import read_elution_windows

    path, _named = table_path(project_path)
    if not path.is_file():
        return None
    for w in read_elution_windows(path):
        if w.key == key:
            return (w.rt_start, w.rt_end), w.drift
    return None
