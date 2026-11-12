# SPDX-License-Identifier: GPL-3.0-or-later
"""The elution tab's window editing (PLAN Next steps 26): an **Edit window** switch in the
tab's bar, the window's start and end (minutes) and, for an ion-mobility file, its drift range
(ms) as boxes, a drag across the trace or the drift profile to set them, the spectrum summed
over the new window drawn beside the one the results hold (:func:`hxex.gui.window_edit.preview`),
**Save window** for the whole peptide in this file (default: its charge states and fragments
elute together) or for this charge state (ion) only (interference at its m/z), as rows for that
file, one for each sample that reads it, in the project's
``[input] rt_elution_windows`` table (:func:`hxex.gui.window_edit.save_window`; the peptide
table is not changed) and
**Undo** (the table as it was before the last save)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
)

from .flow_layout import FlowLayout
from .tasks import Latest
from .tips import tip
from .window_edit import APPLY, scope_words

log = logging.getLogger(__name__)

_WAIT_MS = 400  # a changed box is previewed once it has been left alone this long


# what the save choices mean (worded for both kinds of project: the choices' own text follows
# the project, hxex.gui.window_edit.scope_words)
SCOPE_TIP = (
    "What the saved window is for. This charge state (ion) only: for interference at its own "
    "m/z, in this file. Every charge state, in this file: the whole peptide (and its fragment "
    "ions, when they are read); the same molecules elute together (Compare with: this peptide's "
    "other charge states shows whether they do). In this file and its replicates: also the "
    "sample's other files at this time point (or control). In every file of the peptide: every "
    "file of every sample that reads it. In other files the window is shifted by that file's "
    "elution, matched on this charge state's trace; a file whose match is unclear is left for "
    "you, and you are asked before windows set in a file before are replaced. A changed drift "
    "range is saved for this charge state only (drift times differ between charge states). A "
    "file that several samples read gets the window in each (one injection, one window)")


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


def _group(*widgets: QWidget) -> QWidget:
    """Widgets kept on one line when their row wraps (a box and its label and unit)."""
    group = QWidget()
    line = QHBoxLayout(group)
    line.setContentsMargins(0, 0, 0, 0)
    for w in widgets:
        line.addWidget(w)
    return group


class WindowEditor(QObject):
    """The editing controls of the elution ``panel`` (added to its bar). ``context()`` gives
    the tab's :class:`TraceContext` (None: nothing shown); ``plain()`` draws the tab without
    editing. :attr:`message` says what happened (the window's status bar)."""

    message = Signal(str)
    saved = Signal(str)  # the elution-window table written (its path)
    level_chosen = Signal(str)  # the user chose how to save (the tab shows its evidence)

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
        self._drift_shown: tuple | None = None  # the drift range the boxes were set to
        self._previewed: tuple | None = None  # (spectrum id, window, drift) of a fitted preview
        self._summed: tuple | None = None  # the same, of the spectrum summed (fitted or not)
        self._task = Latest(self)
        self._generation = 0  # a preview's fit stops when a newer one is asked for
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_WAIT_MS)
        self._timer.timeout.connect(self.refresh)

        self.bar = QWidget()
        row = FlowLayout(self.bar)
        self.edit = QCheckBox("Edit window")
        self.edit.setToolTip(
            tip("change the scans this spectrum sums: drag across the trace (or the drift profile) "
                "or type the times; the spectrum summed over the new window is drawn beside the "
                "one the results hold"))
        self.edit.toggled.connect(self._toggled)
        self.start, self.end = self._box("minutes"), self._box("minutes")
        self.start.setToolTip(tip("the window's start (min, the acquisition clock)"))
        self.end.setToolTip(tip("the window's end (min)"))
        self.drift_start, self.drift_end = self._box("ms"), self._box("ms")
        self.drift_start.setToolTip(tip("the drift range's start (ms)"))
        self.drift_end.setToolTip(tip("the drift range's end (ms)"))
        self.drift_label = QLabel("drift")
        self.each_k = QCheckBox("Fits at each K")
        self.each_k.setToolTip(tip("the new window's spectrum fitted at every K the run tried"))
        self.each_k.toggled.connect(lambda _on: self.draw())
        self.scope = QComboBox()
        for data, text in scope_words("", False).items():
            self.scope.addItem(text, data)
        self.scope.setToolTip(SCOPE_TIP)
        # the user's own choice only (not the editor's, setting it to a saved window's kind)
        self.scope.activated.connect(lambda _i: self.level_chosen.emit(self.scope.currentData()))
        self._spread_task = Latest(self)  # the trace match of a window spread to other files
        self.save = QPushButton("Save window")
        self.save.clicked.connect(self._save)
        self.undo = QPushButton("Undo")
        self.undo.setToolTip(tip("put the elution-window table (and the project file, when the "
                                 "first save named the table there) back as before the last save"))
        self.undo.clicked.connect(self._undo)
        self.undo.setEnabled(False)
        # two rows of their own under the figure's buttons (in the buttons' row they made the
        # window wider than the screen: owner, 2026-11-08): the switch and the window, then
        # saving; each wraps between its groups where the window is narrower (one row set the
        # window's narrowest width: 1,102 px with Windows' fonts, 2026-11-11)
        self.save_bar = QWidget()
        save_row = FlowLayout(self.save_bar)
        drift_group = _group(self.drift_label, self.drift_start, QLabel("to"), self.drift_end,
                             QLabel("ms"))
        self.controls = [_group(QLabel("window"), self.start, QLabel("to"), self.end,
                                QLabel("min")), drift_group, self.each_k]
        row.addWidget(self.edit)
        for w in self.controls:
            row.addWidget(w)
        for w in (_group(self.save, QLabel("for"), self.scope), self.undo):
            save_row.addWidget(w)
        self._drift_widgets = [drift_group]
        # what the preview is (and how it differs from a run), above the figure at the
        # window's own font size (owner: not small print inside the figure)
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.hide()
        layout = panel.layout()
        layout.insertWidget(layout.indexOf(panel.title) + 1, self.notes)
        self._show_controls(False)
        panel.add_row(self.bar)
        panel.add_row(self.save_bar)
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
        self.notes.setVisible(on)
        self.save_bar.setVisible(on)
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
            self.panel.set_task(None)
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
        # the results by identity: compared as values, two Results compare their arrays (and
        # fail) when the results were read again with the switch on (owner, 2026-11-08)
        if not (self._target is not None and self._target[0] is target[0]
                and self._target[1] == target[1]):
            self._target = target
            self.preview, self.failure, self._previewed, self._summed = None, None, None, None
            window, drift_range = selected.window, drift
            self._word_scope(ctx)
            saved = self._saved_row(ctx)
            if saved is not None:
                window, drift_range = saved[0], saved[1] or drift
                whose = ("this charge state (ion) has a window of its own"
                         if saved[2] == "ion" else "its peptide has a window")
                self.scope.setCurrentIndex(self.scope.findData(saved[2]))
                self.message.emit(f"{whose} in this file ({window[0]:.4g}-{window[1]:.4g} min, "
                                  "the elution-window table), shown here; if the results do "
                                  "not show it yet, " + APPLY.replace("to use it, ", ""))
            else:
                self.scope.setCurrentIndex(self.scope.findData("peptide"))
            self._has_drift = drift is not None
            self._drift_shown = drift_range
            self._set_boxes(window, drift_range)
            self._show_controls(True, drift=self._has_drift)
            self._save_state(ctx)
            self.refresh()
            return
        self.draw()

    def _word_scope(self, ctx: TraceContext) -> None:
        """The save choices worded for the project (fragment ions or charge states) and the
        selected ion (its charge)."""
        from .window_edit import has_fragments

        key = self._key(ctx)
        words = scope_words(key[2] if key else "", has_fragments(ctx.reader))
        for i in range(self.scope.count()):
            self.scope.setItemText(i, words[self.scope.itemData(i)])

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
            self.panel.set_task("Fitting the new window's spectrum…")
            self.draw()
            self._task.submit(lambda: fit_preview(
                results, picked, threads=1, start_from=sid,
                cancel=lambda: generation != self._generation), fitted, failed)

        def fitted(shown) -> None:  # then fitted (seconds: one spectrum, every K and start)
            self.preview, self._previewed = shown, key
            self.panel.set_task(None)
            self.draw()

        def failed(message: str) -> None:
            self.failure = message
            self.panel.set_task(None)
            self.draw()

        # the raw file is read (the bundle holds each spectrum only as summed over its window)
        self.panel.set_task("Reading the new window's scans…")
        self._task.submit(work, done, failed)

    def draw(self) -> None:
        """The trace with the window being edited, and the spectra now and over it."""
        from ..report.retention import plot_window_edit, preview_text

        ctx = self.context()
        if not self.active or ctx is None:
            return
        shown, failure = self.preview, self.failure
        self.notes.setText(preview_text(shown))
        window, drift, each_k = self.window(), self.drift(), self.each_k.isChecked()
        self.panel.draw(lambda: plot_window_edit(
            ctx.traces, ctx.results, ctx.spectrum_id, shown, edit=window, drift_edit=drift,
            each_k=each_k, title=ctx.title, mz_range=ctx.mz_range, waiting=failure))

    # --- the elution-window table -------------------------------------------------------------

    def _project(self, ctx: TraceContext) -> Path | None:
        """The project file, when the results came from one of a scan format (else None:
        nothing can be saved)."""
        path = ctx.results_path
        if path is None or Path(path).suffix != ".toml":
            return None
        from ..config import load_project

        try:
            inp = load_project(path, analysis_only=True).input
        except Exception as exc:  # noqa: BLE001 (the project file does not load: say so)
            log.info("window editing: %s (%s)", path, exc)
            return None
        if inp is None or inp.format not in ("mzml", "thermo"):
            return None
        return Path(path)

    def _key(self, ctx: TraceContext) -> tuple[str, str, str] | None:
        """The spectrum's row in the table: its sample, file name, and ion (as its id names
        it)."""
        from ..io.mzml import _target_label, run_name

        where = ctx.reader.locate(ctx.spectrum_id, ctx.data_id)
        if where is None:
            return None
        run = ctx.reader.runs[where[0]]
        target = ctx.reader.targets[ctx.reader.of_series[ctx.data_id]]
        return run.sample, run_name(run), _target_label(target)

    def _peptide(self, ctx: TraceContext) -> tuple[str, tuple[str, ...], bool]:
        """The spectrum's peptide (as the table's peptide rows name it), its ions (as spectrum
        ids name them: its charge states, and its fragments when it is a parent), and whether
        another peptide has an ion of this spectrum's name (two parents sharing a terminus)."""
        from ..io.elution_windows import peptide_label
        from ..io.mzml import _target_label

        target = ctx.reader.targets[ctx.reader.of_series[ctx.data_id]]
        name, label = peptide_label(target.entry), _target_label(target)
        ions = tuple(_target_label(t) for t in ctx.reader.targets
                     if peptide_label(t.entry) == name)
        shared = any(_target_label(t) == label and peptide_label(t.entry) != name
                     for t in ctx.reader.targets)
        return name, ions, shared

    def _saved_row(self, ctx: TraceContext):
        from .window_edit import saved_window

        project, key = self._project(ctx), self._key(ctx)
        if project is None or key is None:
            return None
        try:
            return saved_window(project, key, self._peptide(ctx)[0])
        except Exception as exc:  # noqa: BLE001 (a table that does not read: not shown)
            log.info("window editing: %s", exc)
            return None

    def _save_state(self, ctx: TraceContext) -> None:
        if self._project(ctx) is None:
            self.save.setEnabled(False)
            self.save.setToolTip(tip("open the results from their project file (an mzML or Thermo "
                                     "project) to save windows"))
        else:
            self.save.setEnabled(True)
            self.save.setToolTip(tip("write the window for this file in the project's "
                                     "elution-window table ([input] rt_elution_windows; made by "
                                     "the first save), for the whole peptide or this charge "
                                     "state (ion) only (the choice beside), for every sample "
                                     "that reads this file; the peptide table is not changed, "
                                     "and the results change when the project is run again "
                                     "with overwrite enabled (Run on the Run tab); only the "
                                     "series whose spectra changed are fitted again"))

    def _save(self) -> None:
        from .window_edit import (
            DEFAULT_NAME,
            has_fragments,
            own_windows,
            plan_spread,
            samples_reading,
            save_window,
            saved_words,
            spread_words,
            table_path,
        )

        ctx = self.context()
        if ctx is None:
            return
        if self.preview is None or self._summed != (ctx.spectrum_id, self.window(),
                                                    self.drift()):
            self.message.emit("wait for the new window's spectrum, then save")
            return
        project, key = self._project(ctx), self._key(ctx)
        if project is None or key is None:
            return
        chosen = None
        if not table_path(project)[1]:  # the first save: where the table goes (owner)
            from .table_choice import ask_table

            chosen = ask_table(self.panel, project, "input.rt_elution_windows", DEFAULT_NAME,
                               "elution-window table")
            if chosen is None:
                self.message.emit("not saved: no table chosen for the windows")
                return
        spectrum = self.preview.spectrum
        level = self.scope.currentData()
        whole = level != "ion"
        peptide, siblings, shared = self._peptide(ctx)
        drift = self.drift()
        if whole and drift is not None and self._drift_shown is not None and all(
                abs(a - b) < 5e-4 for a, b in zip(drift, self._drift_shown, strict=True)):
            drift = None  # not changed: the ion's drift range stays as it is
        samples = samples_reading(ctx.reader, ctx.spectrum_id, ctx.data_id,
                                  siblings if whole else (key[2],), peptide)
        window = (spectrum.rt_start, spectrum.rt_end)

        def write(plans=(), kept=()):
            more = [(sample, p.file, p.window) for p in plans
                    if p.window is not None and p.file not in kept for sample in p.samples]
            try:
                saved = save_window(project, key, window, drift, peptide=peptide, whole=whole,
                                    siblings=siblings, shared=shared, table=chosen,
                                    samples=samples, more=more)
            except Exception as exc:  # noqa: BLE001 (said in the window: nothing was written)
                self.message.emit(f"not saved: {exc}")
                return
            self.history.append(saved)
            self.undo.setEnabled(True)
            text = saved_words(saved, peptide=peptide, ion=key[2], file=key[1], whole=whole,
                               fragments=has_fragments(ctx.reader),
                               drift_written=drift is not None)
            if level in ("replicates", "everywhere"):
                spread = spread_words(plans, kept) or "no other file of it to spread to"
                text = text.replace(f"; {APPLY}", f"; {spread}; {APPLY}", 1)
            self.message.emit(text)
            self.saved.emit(str(saved.path))

        if level not in ("replicates", "everywhere"):
            write()
            return
        results = ctx.results
        self.panel.set_task("Matching the other files' traces…")
        self.save.setEnabled(False)

        def work():
            return plan_spread(ctx.reader, results, ctx.spectrum_id, ctx.data_id, window,
                               peptide=peptide, ion=key[2], ions=siblings, level=level,
                               drift=self.drift())  # the species chosen in drift time

        def done(plans) -> None:
            self.panel.set_task(None)
            self.save.setEnabled(True)
            places = [(sample, p.file) for p in plans if p.window is not None
                      for sample in p.samples]
            own = own_windows(project, peptide, siblings, places)
            kept: list[str] = []
            if own:
                files = sorted({f for _s, f in own})
                box = QMessageBox(QMessageBox.Icon.Question, "Windows set before",
                                  f"{len(files)} of the files already have a window of this "
                                  f"peptide set before: {', '.join(files)}. Replace them with "
                                  "the shifted window, or keep them?", parent=self.panel)
                replace_button = box.addButton("Replace them", QMessageBox.ButtonRole.AcceptRole)
                keep_button = box.addButton("Keep them", QMessageBox.ButtonRole.RejectRole)
                box.addButton(QMessageBox.StandardButton.Cancel)
                box.exec()
                clicked = box.clickedButton()
                if clicked is keep_button:
                    kept = files
                elif clicked is not replace_button:
                    self.message.emit("not saved")
                    return
            write(plans, kept)

        def failed(message: str) -> None:
            self.panel.set_task(None)
            self.save.setEnabled(True)
            self.message.emit(f"not saved: {message}")

        self._spread_task.submit(work, done, failed)

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
