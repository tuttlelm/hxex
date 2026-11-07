# SPDX-License-Identifier: GPL-3.0-or-later
"""The elution tab's window editing (PLAN Next steps 26): an **Edit window** switch in the
tab's bar, the window's start and end (minutes) and, for an ion-mobility file, its drift range
(ms) as boxes, a drag across the trace or the drift profile to set them, the spectrum summed
over the new window drawn beside the one the results hold (:func:`hxex.gui.window_edit.preview`),
**Save to peptide table** (the run's own row, :func:`hxex.gui.window_edit.save_window`) and
**Undo** (the table as it was before the last save)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from .tasks import Latest

log = logging.getLogger(__name__)

_WAIT_MS = 400  # a changed box is previewed once it has been left alone this long


@dataclass
class TraceContext:
    """What the elution tab shows: the ``results`` (from ``results_path``), the selected
    ``spectrum_id`` of series ``data_id``, its ``traces`` (the latest read), the figure's
    ``title``, the series' ``mz_range`` and the trace ``reader``."""

    results: object
    results_path: Path | None
    spectrum_id: str
    data_id: int
    traces: object
    title: str
    mz_range: tuple[float, float] | None
    reader: object


class WindowEditor(QObject):
    """The editing controls of the elution ``panel`` (added to its bar). ``context()`` gives
    the tab's :class:`TraceContext` (None: nothing shown); ``plain()`` draws the tab without
    editing. :attr:`message` says what happened (the window's status bar)."""

    message = Signal(str)
    saved = Signal(str)  # the peptide table written (its path)

    def __init__(self, panel, context: Callable[[], TraceContext | None],
                 plain: Callable[[], None], parent=None):
        super().__init__(parent)
        self.panel = panel
        self.context = context
        self.plain = plain
        self.preview = None
        self.failure: str | None = None
        self.history: list = []  # SavedWindow, the latest last (Undo)
        self._target: tuple | None = None  # (results, spectrum id) the boxes were set for
        self._has_drift = False  # the spectrum was summed over a drift range
        self._previewed: tuple | None = None  # (spectrum id, window, drift) of a fitted preview
        self._summed: tuple | None = None  # the same, of the spectrum summed (fitted or not)
        self._task = Latest(self)
        self._generation = 0  # a preview's fit stops when a newer one is asked for
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_WAIT_MS)
        self._timer.timeout.connect(self.refresh)

        self.bar = QWidget()
        row = QHBoxLayout(self.bar)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit = QCheckBox("Edit window")
        self.edit.setToolTip(
            "change the scans this spectrum sums: drag across the trace (or the drift profile) "
            "or type the times; the spectrum summed over the new window is drawn beside the "
            "one the results hold")
        self.edit.toggled.connect(self._toggled)
        self.start, self.end = self._box("minutes"), self._box("minutes")
        self.start.setToolTip("the window's start (min, the acquisition clock)")
        self.end.setToolTip("the window's end (min)")
        self.drift_start, self.drift_end = self._box("ms"), self._box("ms")
        self.drift_start.setToolTip("the drift range's start (ms)")
        self.drift_end.setToolTip("the drift range's end (ms)")
        self.drift_label = QLabel("drift")
        self.each_k = QCheckBox("Fits at each K")
        self.each_k.setToolTip("the new window's spectrum fitted at every K the run tried")
        self.each_k.toggled.connect(lambda _on: self.draw())
        self.save = QPushButton("Save to peptide table")
        self.save.setToolTip("write the window as this run's own row of the project's peptide "
                             "table; the results change when the project is run again")
        self.save.clicked.connect(self._save)
        self.undo = QPushButton("Undo")
        self.undo.setToolTip("write the peptide table back as it was before the last save")
        self.undo.clicked.connect(self._undo)
        self.undo.setEnabled(False)
        self.controls = [QLabel("window"), self.start, QLabel("to"), self.end, QLabel("min"),
                         self.drift_label, self.drift_start, QLabel("to"), self.drift_end,
                         QLabel("ms"), self.each_k, self.save]
        row.addWidget(self.edit)
        for w in self.controls:
            row.addWidget(w)
        row.addWidget(self.undo)
        self._drift_widgets = self.controls[5:10]
        self._show_controls(False)
        panel.add_option(self.bar)
        panel.spanned.connect(self._spanned)

    def _box(self, unit: str) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setDecimals(3)
        box.setRange(0.0, 1e5)
        box.setSingleStep(0.01)
        box.setKeyboardTracking(False)
        box.valueChanged.connect(lambda _v: self._changed())
        return box

    @property
    def active(self) -> bool:
        return self.edit.isChecked()

    def _show_controls(self, on: bool, drift: bool = False) -> None:
        for w in self.controls:
            w.setVisible(on)
        for w in self._drift_widgets:
            w.setVisible(on and drift)

    # --- the spectrum shown ------------------------------------------------------------------

    def _toggled(self, on: bool) -> None:
        self.panel.set_span_mode(on)
        if not on:
            self._show_controls(False)
            self._generation += 1  # a fit still running stops
            self._task.submit(lambda: None, lambda _r: None, lambda _m: None)  # drop a preview
            self.plain()
            return
        self._target = None
        self.update()

    def update(self) -> None:
        """The tab shows new traces (another spectrum, or its other runs read): the boxes set
        for a new spectrum (its saved window, else the window it summed) and the figure drawn
        again."""
        if not self.active:
            return
        ctx = self.context()
        if ctx is None or not ctx.traces.traces:
            self._show_controls(False)
            self.plain()
            return
        target = (ctx.results, ctx.spectrum_id)
        selected = ctx.traces.traces[0]
        drift = selected.drift
        if target != self._target:
            self._target = target
            self.preview, self.failure, self._previewed, self._summed = None, None, None, None
            window, drift_range = selected.window, drift
            saved = self._saved_row(ctx)
            if saved is not None:
                window, drift_range = saved[0], saved[1] or drift
                self.message.emit("this run has a window of its own in the peptide table "
                                  f"({window[0]:.4g}-{window[1]:.4g} min): shown here; the "
                                  "results change when the project is run again")
            self._has_drift = drift is not None
            self._set_boxes(window, drift_range)
            self._show_controls(True, drift=self._has_drift)
            self._save_state(ctx)
            self.refresh()
            return
        self.draw()

    def _set_boxes(self, window, drift) -> None:
        boxes = [(self.start, window[0]), (self.end, window[1])]
        if drift is not None:
            boxes += [(self.drift_start, drift[0]), (self.drift_end, drift[1])]
        for box, value in boxes:
            box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(False)

    def window(self) -> tuple[float, float]:
        return self.start.value(), self.end.value()

    def drift(self) -> tuple[float, float] | None:
        if not self._has_drift:
            return None
        return self.drift_start.value(), self.drift_end.value()

    def _changed(self) -> None:
        if self.active:
            self._timer.start()

    def _spanned(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """A drag across the trace (the window) or the drift profile (the drift range)."""
        fig = self.panel.view.figure
        if not self.active or fig is None:
            return
        for ax in fig.axes:
            if not ax.get_position().contains(x0, y0):
                continue
            label = ax.get_xlabel()
            to_data = ax.transData.inverted()
            a = to_data.transform(fig.transFigure.transform((x0, y0)))[0]
            b = to_data.transform(fig.transFigure.transform((x1, y1)))[0]
            a, b = sorted((float(a), float(b)))
            if label.startswith("retention time"):
                self._set_boxes((a, b), None)
            elif label.startswith("drift time") and self.drift() is not None:
                self._set_boxes(self.window(), (a, b))
            else:
                return
            self._timer.stop()
            self.refresh()
            return

    # --- the preview ---------------------------------------------------------------------------

    def refresh(self) -> None:
        """Sum the spectrum again over the boxes' window (in the background) and draw it."""
        from .window_edit import fit_preview, preview

        ctx = self.context()
        if not self.active or ctx is None:
            return
        window, drift = self.window(), self.drift()
        key = (ctx.spectrum_id, window, drift)
        if key == self._previewed and self.preview is not None:
            self.draw()
            return
        self.preview, self.failure, self._summed = None, None, None
        self._generation += 1
        generation = self._generation
        self.draw()
        reader, results, sid, data_id = ctx.reader, ctx.results, ctx.spectrum_id, ctx.data_id

        def work():  # summed and picked: drawn at once (a fraction of a second)
            return preview(results, reader.spectrum(sid, data_id, window, drift=drift),
                           fit=False)

        def done(picked) -> None:
            self.preview, self._summed = picked, key
            self.draw()
            self._task.submit(lambda: fit_preview(
                results, picked, threads=1, start_from=sid,
                cancel=lambda: generation != self._generation), fitted, failed)

        def fitted(shown) -> None:  # then fitted (seconds: one spectrum, every K and start)
            self.preview, self._previewed = shown, key
            self.draw()

        def failed(message: str) -> None:
            self.failure = message
            self.draw()

        self._task.submit(work, done, failed)

    def draw(self) -> None:
        """The trace with the window being edited, and the spectra now and over it."""
        from ..report.retention import plot_window_edit

        ctx = self.context()
        if not self.active or ctx is None:
            return
        shown, failure = self.preview, self.failure
        window, drift, each_k = self.window(), self.drift(), self.each_k.isChecked()
        self.panel.draw(lambda: plot_window_edit(
            ctx.traces, ctx.results, ctx.spectrum_id, shown, edit=window, drift_edit=drift,
            each_k=each_k, title=ctx.title, mz_range=ctx.mz_range, waiting=failure))

    # --- the peptide table ---------------------------------------------------------------------

    def _project(self, ctx: TraceContext):
        """The project's input settings, when the results came from a project file of a scan
        format (else None: nothing can be saved)."""
        path = ctx.results_path
        if path is None or Path(path).suffix != ".toml":
            return None
        from ..config import load_project

        try:
            inp = load_project(path, analysis_only=True).input
        except Exception as exc:  # noqa: BLE001 (the project file does not load: say so)
            log.info("window editing: %s (%s)", path, exc)
            return None
        if inp is None or inp.format not in ("mzml", "thermo") or inp.peptides is None:
            return None
        return inp

    def _run_name(self, ctx: TraceContext) -> str | None:
        from ..io.mzml import run_name

        where = ctx.reader.locate(ctx.spectrum_id, ctx.data_id)
        return None if where is None else run_name(ctx.reader.runs[where[0]])

    def _saved_row(self, ctx: TraceContext):
        from .window_edit import saved_window

        inp, run = self._project(ctx), self._run_name(ctx)
        if inp is None or run is None:
            return None
        entry = ctx.reader.targets[ctx.reader.of_series[ctx.data_id]].entry
        try:
            return saved_window(inp.peptides, entry, run)
        except (OSError, ValueError):
            return None

    def _save_state(self, ctx: TraceContext) -> None:
        if self._project(ctx) is None:
            self.save.setEnabled(False)
            self.save.setToolTip("open the results from their project file (an mzML or Thermo "
                                 "project) to save windows to its peptide table")
        else:
            self.save.setEnabled(True)
            self.save.setToolTip("write the window as this run's own row of the project's "
                                 "peptide table; the results change when the project is run "
                                 "again")

    def _save(self) -> None:
        from .window_edit import save_window

        ctx = self.context()
        if ctx is None:
            return
        if self.preview is None or self._summed != (ctx.spectrum_id, self.window(),
                                                    self.drift()):
            self.message.emit("wait for the new window's spectrum, then save")
            return
        inp, run = self._project(ctx), self._run_name(ctx)
        if inp is None or run is None:
            return
        spectrum = self.preview.spectrum
        drift = self.drift()
        try:
            saved = save_window(ctx.results_path, inp.peptides, spectrum, run,
                                (spectrum.rt_start, spectrum.rt_end), drift)
        except (OSError, ValueError) as exc:
            self.message.emit(f"not saved: {exc}")
            return
        self.history.append(saved)
        self.undo.setEnabled(True)
        what = "added" if saved.added else "changed"
        self.message.emit(
            f"{Path(saved.path).name}: run {run}'s row {what} ({saved.row.get('rt_start')}-"
            f"{saved.row.get('rt_end')} min); run the project again (hxex run --overwrite) to "
            "use it")
        self.saved.emit(str(saved.path))

    def _undo(self) -> None:
        from .window_edit import undo_window

        if not self.history:
            return
        last = self.history[-1]
        try:
            undo_window(last)
        except (OSError, ValueError) as exc:
            self.message.emit(f"not undone: {exc}")
            return
        self.history.pop()
        self.undo.setEnabled(bool(self.history))
        self.message.emit(f"{Path(last.path).name}: the last saved window undone")
        self.saved.emit(str(last.path))
