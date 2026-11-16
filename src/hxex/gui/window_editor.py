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

import contextlib
import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
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
from .tips import rich_tip, tip
from .window_edit import (
    APPLY,
    neighbour_scan,
    scope_tips,
    scope_words,
    snap_window,
    widen_drift,
)

log = logging.getLogger(__name__)

_WAIT_MS = 400  # a changed box is previewed once it has been left alone this long


class _SnapBox(QDoubleSpinBox):
    """A box whose arrows (and wheel) step to the next scan or drift bin rather than by a
    fixed amount (``step_to(value, steps)``; None: the fixed step), so that a step never snaps
    back onto the same scan (audit of 2026-11-15, F-1)."""

    step_to: Callable[[float, int], float] | None = None

    def stepBy(self, steps: int) -> None:  # noqa: N802 - Qt's name
        if self.step_to is None:
            super().stepBy(steps)
            return
        self.setValue(self.step_to(self.value(), steps))


def set_choice_tips(combo: QComboBox, tips: dict[str, str]) -> None:
    """Give each choice of ``combo`` its tooltip (``tips`` by the choice's data, bold between
    ``**``), shown on the open list's items and, for the selected choice, on the box (owner,
    2026-11-12: one long tooltip for every choice was hard to read)."""
    for i in range(combo.count()):
        combo.setItemData(i, rich_tip(tips.get(combo.itemData(i), "")),
                          Qt.ItemDataRole.ToolTipRole)
    combo.setToolTip(combo.itemData(combo.currentIndex(), Qt.ItemDataRole.ToolTipRole) or "")
    if not combo.property("tips_follow"):  # connected once: the box follows the selection
        combo.setProperty("tips_follow", True)
        combo.currentIndexChanged.connect(lambda i: combo.setToolTip(
            combo.itemData(i, Qt.ItemDataRole.ToolTipRole) or ""))


@dataclass
class TraceContext:
    """What the elution tab shows: the ``results`` (from ``results_path``), the selected
    ``spectrum_id`` of series ``data_id``, its ``traces`` (the latest read), the figure's
    ``title``, the series' ``mz_range``, the trace ``reader``, and what 'Show elution traces
    for' shows (``compare``: ``replicates`` and ``everywhere`` are the files a spread reaches,
    within this sample)."""

    results: object
    results_path: Path | None
    spectrum_id: str
    data_id: int
    traces: object
    title: str
    mz_range: tuple[float, float] | None
    reader: object
    compare: str | None = None


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

    def __init__(self, panel, context: Callable[[], TraceContext | None],
                 plain: Callable[[], None], parent=None):
        super().__init__(parent)
        self.panel = panel
        self.context = context
        self.plain = plain
        self.preview = None
        self.failure: str | None = None
        self.history: list = []  # SavedWindow, the latest last (Undo); the results' own
        self._results_path: Path | None = None  # where the results shown come from
        # the window's context manager recording a write of the project file for the Project
        # editor's Undo (hxex.gui.window.MainWindow.recording; None: not recorded; Q-1)
        self.recording = None
        self._target: tuple | None = None  # (trace reader, spectrum id) the boxes were set for
        self._has_drift = False  # the spectrum was summed over a drift range
        self._drift_shown: tuple | None = None  # the drift range the table or the run gives
        # the window and drift range exactly as set (the boxes show them to their decimals; a
        # window is a set of scans, so the boxes snap to the ion's scan times and the drift
        # range widens to the file's drift bins: audit of 2026-11-15, F-1)
        self._window: tuple[float, float] | None = None
        self._drift: tuple[float, float] | None = None
        self._scan_times: np.ndarray | None = None  # the ion's scans in the file (minutes)
        self._drift_bins: np.ndarray | None = None  # the file's drift bins (ms)
        self._saving: str | None = None  # the spectrum whose save is matching other files
        self._project_cache: tuple | None = None  # (results path, signature, project or None)
        self._previewed: tuple | None = None  # (spectrum id, window, drift) of a fitted preview
        self._summed: tuple | None = None  # the same, of the spectrum summed (fitted or not)
        self._task = Latest(self)
        self._align_task = Latest(self)  # the other files matched against this one's trace
        self.alignment = None  # hxex.gui.window_edit.Alignment (or None: no other file)
        self._aligned_key: tuple | None = None  # (spectrum, level, drift) it was made for
        self._aligning: tuple | None = None  # the same, of the match being made
        self._generation = 0  # a preview's fit stops when a newer one is asked for
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_WAIT_MS)
        self._timer.timeout.connect(self.refresh)

        self.bar = QWidget()
        row = FlowLayout(self.bar)
        self.edit = QCheckBox("Edit window")
        self.edit.setToolTip(
            tip("Change the scans this spectrum sums: drag across the trace or type the times. "
                "The new spectrum is drawn beside the current one."))
        self.edit.toggled.connect(self._toggled)
        self.start, self.end = self._box("minutes"), self._box("minutes")
        self.start.setToolTip(tip("the window's start (min, the acquisition clock): the first "
                                  "scan summed; the arrows step scan by scan"))
        self.end.setToolTip(tip("the window's end (min): the last scan summed; the arrows step "
                                "scan by scan"))
        self.drift_start, self.drift_end = self._box("ms"), self._box("ms")
        self.drift_start.setToolTip(tip("the drift range's start (ms)"))
        self.drift_end.setToolTip(tip("the drift range's end (ms)"))
        self.drift_label = QLabel("drift")
        self.each_k = QCheckBox("Fits at each K")
        self.each_k.setToolTip(tip("the new window's spectrum fitted at every K the run tried"))
        self.each_k.toggled.connect(lambda _on: self.draw())
        self.each_file = QCheckBox("Each file's window")
        self.each_file.setToolTip(tip(
            "With Show elution traces for on 3 or 4: one row per file, showing the scans it "
            "sums now and where the new window would go."))
        self.each_file.toggled.connect(lambda _on: self.draw())
        self.scope = QComboBox()
        for data, text in scope_words("", False).items():
            self.scope.addItem(text, data)
        set_choice_tips(self.scope, scope_tips(False))
        self._spread_task = Latest(self)  # the trace match of a window spread to other files
        self.save = QPushButton("Save window")
        self.save.clicked.connect(self._save)
        self.undo = QPushButton("Undo")
        self.undo.setToolTip(tip("Undo the last save."))
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
                                QLabel("min")), drift_group, self.each_k,
                         self.each_file]
        row.addWidget(self.edit)
        for w in self.controls:
            row.addWidget(w)
        for w in (_group(self.save, QLabel("for"), self.scope), self.undo):
            save_row.addWidget(w)
        self._drift_widgets = [drift_group]
        # that the figure is a preview, above its title at the window's own font size (owner:
        # not small print inside the figure); one fixed line, how it differs from a run on hover
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.hide()
        self._show_controls(False)
        panel.add_row(self.bar)
        panel.add_row(self.save_bar)
        layout = panel.layout()  # after the editing rows: directly above the title
        layout.insertWidget(layout.indexOf(panel.title), self.notes)
        panel.spanned.connect(self._spanned)
        self.saved.connect(self._after_save)

    def _box(self, unit: str) -> QDoubleSpinBox:
        box = _SnapBox()
        box.setDecimals(4 if unit == "minutes" else 3)
        box.setRange(0.0, 1e5)
        box.setSingleStep(0.01)
        box.setKeyboardTracking(False)
        box.valueChanged.connect(lambda _v: self._changed())
        if unit == "minutes":
            box.step_to = self._next_scan
        return box

    def _next_scan(self, value: float, steps: int) -> float:
        """A minutes box stepped: the scan ``steps`` scans from the one at ``value``."""
        if self._scan_times is None or not len(self._scan_times):
            return value + 0.01 * steps
        return neighbour_scan(value, self._scan_times, steps)

    @property
    def active(self) -> bool:
        return self.edit.isChecked()

    def _show_controls(self, on: bool, drift: bool = False) -> None:
        self.notes.setVisible(on)
        self.save_bar.setVisible(on)
        ctx = self.context() if on else None
        spread = ctx is not None and ctx.compare in ("replicates", "everywhere")
        self.each_file.setEnabled(spread)
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

    def reset(self, results_path: Path | None) -> None:
        """Other results are shown (from ``results_path``): the saves of the results before
        cannot be undone from here (Undo rewrote the previous project's table and project file
        under another project; audit of 2026-11-15, D-5)."""
        self._results_path = results_path
        if self.history:
            log.info("the elution tab's Undo history (%d save(s)) left behind with %s",
                     len(self.history), self.history[-1].path)
        self.history.clear()
        self.undo.setEnabled(False)

    def forget(self) -> None:
        """Another spectrum is selected: what is still being read, fitted or matched for the
        one before is dropped (done, it drew the previous spectrum's view over the cleared tab
        before the new spectrum's trace was read; owner, 2026-11-12). A save still matching
        the other files is dropped too, and said: its question and message would have come
        under the new spectrum (audit of 2026-11-15, D-11, F-16)."""
        self._generation += 1  # a preview's fit stops
        self._aligning = None  # and a match (its cancel hook asks for its key)
        self._target = None
        self._task.submit(lambda: None, lambda _r: None, lambda _m: None)
        self._align_task.submit(lambda: None, lambda _r: None, lambda _m: None)
        if self._saving is not None:
            self._spread_task.submit(lambda: None, lambda _r: None, lambda _m: None)
            self.message.emit("not saved: another spectrum was selected while the other "
                              "files were being matched; select it again and save")
            self._saving = None
            self.save.setEnabled(True)
        self.panel.set_task(None)

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
        # the spectrum and the read record (its trace reader: one per record, kept across
        # review decisions): a decision gives new Results for the same spectrum, which must
        # not throw the window being edited away (audit of 2026-11-15, F-2); results read
        # again from disk give a new reader, and the boxes are set again
        target = (ctx.reader, ctx.spectrum_id)
        selected = ctx.traces.traces[0]
        drift = selected.drift
        if not (self._target is not None and self._target[0] is target[0]
                and self._target[1] == target[1]):
            self._target = target
            self.preview, self.failure, self._previewed, self._summed = None, None, None, None
            self.alignment, self._aligned_key, self._aligning = None, None, None
            self._scan_times = ctx.traces.scan_times
            self._drift_bins = (None if ctx.traces.drift is None
                                else np.asarray(ctx.traces.drift.drift, dtype=float))
            window, drift_range = selected.window, drift
            self._word_scope(ctx)
            saved, problem = self._saved_row(ctx)
            if problem:  # a table that does not read: said, not passed over (F-9)
                self.message.emit(f"the elution-window table was not read: {problem}; the "
                                  "window the results hold is shown")
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
            self._window, self._drift = None, None  # set afresh for this spectrum
            self._set_boxes(window, drift_range)
            self._show_controls(True, drift=self._has_drift)
            self._save_state(ctx)
            self.refresh()
            return
        if self._aligned_now(ctx) is None and self._matching(ctx):
            return  # the other files arrived: drawn once matched (left out until then)
        self.draw()

    def _word_scope(self, ctx: TraceContext) -> None:
        """The save choices worded for the project (fragment ions or charge states) and the
        selected ion (its charge)."""
        from .window_edit import has_fragments

        key = self._key(ctx)
        words = scope_words(key[2] if key else "", has_fragments(ctx.reader))
        for i in range(self.scope.count()):
            self.scope.setItemText(i, words[self.scope.itemData(i)])
        set_choice_tips(self.scope, scope_tips(has_fragments(ctx.reader)))

    def _set_boxes(self, window, drift) -> None:
        """Set the window (minutes; snapped to the ion's scans) and, when given, the drift
        range (ms; widened to the file's drift bins), and show them in the boxes."""
        self._window = snap_window(window, self._scan_times)
        boxes = [(self.start, self._window[0]), (self.end, self._window[1])]
        if drift is not None:
            self._drift = widen_drift(drift, self._drift_bins)
            boxes += [(self.drift_start, self._drift[0]), (self.drift_end, self._drift[1])]
        for box, value in boxes:
            box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(False)

    def window(self) -> tuple[float, float]:
        """The window being edited (minutes): the scans' own times, not the boxes' rounded
        values."""
        if self._window is None:
            return self.start.value(), self.end.value()
        return self._window

    def drift(self) -> tuple[float, float] | None:
        if not self._has_drift:
            return None
        if self._drift is None:
            return self.drift_start.value(), self.drift_end.value()
        return self._drift

    def _changed(self) -> None:
        """A box was edited (typed, stepped): snapped, then previewed once left alone."""
        if not self.active:
            return
        self._set_boxes((self.start.value(), self.end.value()),
                        (self.drift_start.value(), self.drift_end.value())
                        if self._has_drift else None)
        self._timer.start()

    def _spanned(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """A drag across the trace (the window) or the drift profile (the drift range): mapped
        with the figure whose picture is on screen (a new figure's axes, set before its
        picture is drawn, gave a drag the wrong places: audit of 2026-11-15, F-19)."""
        view = self.panel.view
        fig = getattr(view, "shown_figure", view.figure)
        if not self.active or fig is None:
            return
        for ax in fig.axes:
            if not ax.get_position().contains(x0, y0):
                continue
            label = "retention time" if ax.get_label() == "this file" else ax.get_xlabel()
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
        with_profile = ctx.traces.drift is not None  # the drift panel follows the window

        def work():  # summed and picked: drawn at once (a fraction of a second)
            summed = reader.summed(sid, data_id, window, drift=drift, profile=with_profile)
            return preview(results, summed.spectrum, fit=False, profile=summed.profile)

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

    def _align(self, ctx: TraceContext) -> None:
        """Match the files the plots show (its replicates, or every file of this sample)
        against this one's trace within the drift range in the boxes, in the background, once
        per spectrum, level and drift range (the window does not change the match); drawn
        when ready (:meth:`draw`) and used by a save at that level."""
        from .window_edit import align_files

        level = ctx.compare if ctx.compare in ("replicates", "everywhere") else None
        key = (ctx.spectrum_id, level, self.drift())
        if level is None or key in (self._aligned_key, self._aligning):
            return
        key_ = self._key(ctx)
        if key_ is None:
            return
        self._aligning = key
        peptide, siblings, _shared = self._peptide(ctx)
        reader, results, sid, data_id, drift = (ctx.reader, ctx.results, ctx.spectrum_id,
                                                ctx.data_id, self.drift())

        def done(aligned) -> None:
            if self._aligning != key:
                return
            self.alignment, self._aligned_key, self._aligning = aligned, key, None
            self.draw()

        def failed(message: str) -> None:
            if self._aligning == key:
                self._aligning = None
                self.message.emit(f"the other files' traces were not matched: {message}")

        self._align_task.submit(lambda: align_files(
            reader, results, sid, data_id, peptide=peptide, ion=key_[2], ions=siblings,
            level=level, drift=drift, cancel=lambda: self._aligning != key), done, failed)

    def _matching(self, ctx: TraceContext) -> bool:
        """Whether the files the plots show are still being matched to this one's trace
        (started if not yet): they are drawn once matched, not first where they are and then
        moved (owner, 2026-11-12: the figure jumped)."""
        self._align(ctx)
        level = ctx.compare if ctx.compare in ("replicates", "everywhere") else None
        return level is not None and self._aligning == (ctx.spectrum_id, level, self.drift())

    def _aligned_now(self, ctx: TraceContext):
        """The match the tab draws: the one for the drift range in the boxes, else, while that
        one is being made, the previous match of this spectrum and level (another drift
        range): the traces are those the run summed, so only the other files' shifts can
        change when it lands (owner, 2026-11-12: option A). None before any match of this
        spectrum (a save uses only the match for the boxes' drift range)."""
        level = ctx.compare if ctx.compare in ("replicates", "everywhere") else None
        a = self.alignment
        if level is None or a is None or self._aligned_key is None:
            return None
        if self._aligned_key[:2] != (ctx.spectrum_id, level):
            return None
        return a

    def draw(self) -> None:
        """The trace with the window being edited, and the spectra now and over it; with
        the plots on view 3 or 4 (replicates or every file), those files aligned on this
        file's clock by the trace match (owner's choice, 2026-11-11: draft A), or one row per
        file (Each file's window: draft B). The traces are always those the run summed (their
        own drift ranges); the match, within the drift range in the boxes, gives only the shifts."""
        from ..report.retention import (
            plot_each_file,
            plot_window_edit,
            preview_notes,
            preview_text,
        )

        ctx = self.context()
        if not self.active or ctx is None:
            return
        self._align(ctx)
        spread = ctx.compare in ("replicates", "everywhere")
        aligned = self._aligned_now(ctx)
        # the rows view needs other files: none found by the match, the switch would do
        # nothing (audit of 2026-11-15, F-17)
        none_found = (self.alignment is None and self._aligned_key is not None
                      and self._aligned_key[:2] == (ctx.spectrum_id, ctx.compare))
        self.each_file.setEnabled(spread and not none_found)
        shown, failure = self.preview, self.failure
        self.notes.setText(preview_text(shown))
        notes = preview_notes(shown)  # how this preview differs from a run, on hover
        self.notes.setToolTip(tip(notes) if notes else "")
        window, drift, each_k = self.window(), self.drift(), self.each_k.isChecked()
        if spread and self.each_file.isChecked() and aligned is not None:
            self.panel.draw(lambda: plot_each_file(ctx.traces, aligned, edit=window,
                                                   title=ctx.title))
            return
        traces = ctx.traces
        if spread and aligned is None and self._matching(ctx):  # this file's trace alone
            traces = dataclasses.replace(traces, traces=traces.traces[:1])
        self.panel.draw(lambda: plot_window_edit(
            traces, ctx.results, ctx.spectrum_id, shown, edit=window, drift_edit=drift,
            each_k=each_k, title=ctx.title, mz_range=ctx.mz_range, waiting=failure,
            aligned=aligned))

    # --- the elution-window table -------------------------------------------------------------

    def _project(self, ctx: TraceContext) -> Path | None:
        """The project file, when the results came from one of a scan format (else None:
        nothing can be saved). Loaded once per project file as long as it is unchanged on
        disk (audit of 2026-11-15, F-12: three loads per spectrum shown)."""
        from .window_edit import _signature

        path = ctx.results_path
        if path is None or Path(path).suffix != ".toml":
            return None
        sig = _signature(Path(path))
        kept = self._project_cache
        if kept is not None and kept[0] == path and kept[1] == sig and sig is not None:
            return kept[2]
        from ..config import load_project

        project: Path | None = Path(path)
        try:
            inp = load_project(path, analysis_only=True).input
        except Exception as exc:  # noqa: BLE001 (the project file does not load: say so)
            log.info("window editing: %s (%s)", path, exc)
            project = None
        else:
            if inp is None or inp.format not in ("mzml", "thermo"):
                project = None
        self._project_cache = (path, sig, project)
        return project

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

    def _saved_row(self, ctx: TraceContext) -> tuple:
        """The table's window for the spectrum shown (:func:`hxex.gui.window_edit
        .saved_window`, or None) and, when the table does not read, the reader's words."""
        from .window_edit import saved_window

        project, key = self._project(ctx), self._key(ctx)
        if project is None or key is None:
            return None, None
        try:
            return saved_window(project, key, self._peptide(ctx)[0]), None
        except Exception as exc:  # noqa: BLE001 (a table that does not read: said)
            log.info("window editing: %s", exc)
            return None, str(exc)

    def _after_save(self, _path: str) -> None:
        """A save or an Undo wrote the table: what the table now gives this spectrum is the
        baseline the next save compares the drift range with, and Save's state follows the
        project file (audit of 2026-11-15, D-13)."""
        ctx = self.context()
        if ctx is None or not ctx.traces.traces:
            return
        saved, _problem = self._saved_row(ctx)
        own = ctx.traces.traces[0].drift
        self._drift_shown = (saved[1] or own) if saved is not None else own
        self._save_state(ctx)

    def _save_state(self, ctx: TraceContext) -> None:
        if self._project(ctx) is None:
            self.save.setEnabled(False)
            self.save.setToolTip(tip("open the results from their project file (an mzML or Thermo "
                                     "project) to save windows"))
        else:
            self.save.setEnabled(True)
            self.save.setToolTip(tip("Save the new retention times (and drift range, for "
                                     "ion-mobility files) to the project's elution-window table "
                                     "([input] rt_elution_windows). The first save asks where "
                                     "to put the table."))

    def _save(self) -> None:
        from .window_edit import (
            DEFAULT_NAME,
            has_fragments,
            own_windows,
            plan_spread,
            samples_reading,
            save_window,
            saved_words,
            sibling_windows,
            spread_words,
            table_path,
        )

        ctx = self.context()
        if ctx is None:
            return
        if self.failure is not None:  # nothing is coming (audit of 2026-11-15, F-15)
            self.message.emit(f"not saved: {self.failure}")
            return
        boxes_drift = self.drift()  # read once, here (F-7: a worker read the boxes later)
        if self.preview is None or self._summed != (ctx.spectrum_id, self.window(),
                                                    boxes_drift):
            self.message.emit("wait for the new window's spectrum, then save")
            return
        project, key = self._project(ctx), self._key(ctx)
        if project is None:  # it loaded when the results were opened (Save was enabled)
            self.message.emit("not saved: the project file does not load any more (or no "
                              "longer reads mzML or Thermo raw files)")
            return
        if key is None:
            return
        chosen = None
        try:
            named = table_path(project)[1]
        except Exception as exc:  # noqa: BLE001 (a project file that stopped loading: D-10)
            self.message.emit(f"not saved: {exc}")
            return
        if not named:  # the first save: where the table goes (owner)
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
        drift = boxes_drift
        # not changed since the table or the run gave it: the row's drift range stays as it
        # is, at every level (D open question 3: a level-1 save no longer pins the reader's
        # own drift choice)
        if drift is not None and self._drift_shown is not None and all(
                abs(a - b) < 5e-4 for a, b in zip(drift, self._drift_shown, strict=True)):
            drift = None
        samples = samples_reading(ctx.reader, ctx.spectrum_id, ctx.data_id,
                                  siblings if whole else (key[2],), peptide)
        window = (spectrum.rt_start, spectrum.rt_end)
        fragments = has_fragments(ctx.reader)
        # the edited file's other ions with windows of their own: a save for the whole peptide
        # replaces them, as in the files it is spread to, after asking (audit of 2026-11-15,
        # D-8)
        own_here = sibling_windows(project, peptide, key[2], siblings,
                                   [(sample, key[1]) for sample in (key[0], *samples)],
                                   shared) if whole else []
        here = (f"{key[1]} (this file: {len(own_here)} other "
                + ("ion(s)" if fragments else "charge state(s)") + ")")

        def write(plans=(), kept=(), keep_own=False):
            more = [(sample, p.file, p.window) for p in plans
                    if p.window is not None and p.file not in kept for sample in p.samples]
            record = self.recording
            try:
                with (record("an elution window saved (the table named in the project file)")
                      if record is not None else contextlib.nullcontext()):
                    saved = save_window(project, key, window, drift, peptide=peptide,
                                        whole=whole, siblings=siblings, shared=shared,
                                        table=chosen, samples=samples, more=more,
                                        keep_own=keep_own)
            except Exception as exc:  # noqa: BLE001 (said in the window: nothing was written)
                self.message.emit(f"not saved: {exc}")
                return
            self.history.append(saved)
            self.undo.setEnabled(True)
            text = saved_words(saved, peptide=peptide, ion=key[2], file=key[1], whole=whole,
                               fragments=fragments, drift_written=drift is not None)
            if level in ("replicates", "everywhere"):
                spread = spread_words(plans, kept) or "no other file of it to spread to"
                text = text.replace(f"; {APPLY}", f"; {spread}; {APPLY}", 1)
            self.message.emit(text)
            self.saved.emit(str(saved.path))

        if level not in ("replicates", "everywhere"):
            answer = self._replace_or_keep([here]) if own_here else "replace"
            if answer is None:
                self.message.emit("not saved")
                return
            write(keep_own=answer == "keep")
            return
        results = ctx.results
        ready = self.alignment if self._aligned_key == (ctx.spectrum_id, level,
                                                         boxes_drift) else None
        reader, sid, data_id = ctx.reader, ctx.spectrum_id, ctx.data_id

        def work():
            return plan_spread(reader, results, sid, data_id, window, peptide=peptide,
                               ion=key[2], ions=siblings, level=level,
                               drift=boxes_drift)  # the species chosen in drift time

        def done(plans) -> None:
            self._saving = None
            self.panel.set_task(None)
            self.save.setEnabled(True)
            places = [(sample, p.file) for p in plans if p.window is not None
                      for sample in p.samples]
            try:
                own = own_windows(project, peptide, siblings, places)
            except Exception as exc:  # noqa: BLE001 (the table or project changed: D-10)
                self.message.emit(f"not saved: {exc}")
                return
            kept: list[str] = []
            files = sorted({f for _s, f in own})
            answer = "replace"
            if own or own_here:
                answer = self._replace_or_keep(([here] if own_here else []) + files)
                if answer is None:
                    self.message.emit("not saved")
                    return
                if answer == "keep":
                    kept = files
            write(plans, kept, keep_own=answer == "keep")

        def failed(message: str) -> None:
            self._saving = None
            self.panel.set_task(None)
            self.save.setEnabled(True)
            self.message.emit(f"not saved: {message}")

        if ready is not None:  # the tab matched these files already: the same match
            done(ready.for_window(window))
            return
        self.panel.set_task("Matching the other files' traces…")
        self.save.setEnabled(False)
        self._saving = sid
        self._spread_task.submit(work, done, failed)

    def _replace_or_keep(self, files: list[str]) -> str | None:
        """Asks whether the windows of this peptide set before in ``files`` (their names, the
        edited file's with what it holds) are replaced or kept: ``"replace"``, ``"keep"``, or
        None (cancelled)."""
        box = QMessageBox(QMessageBox.Icon.Question, "Windows set before",
                          f"{len(files)} of the files already have a window of this peptide "
                          f"set before: {', '.join(files)}. Replace them with the new window "
                          "(shifted in each other file), or keep them?", parent=self.panel)
        replace_button = box.addButton("Replace them", QMessageBox.ButtonRole.AcceptRole)
        keep_button = box.addButton("Keep them", QMessageBox.ButtonRole.RejectRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is keep_button:
            return "keep"
        return "replace" if clicked is replace_button else None

    def _undo(self) -> None:
        from .window_edit import undo_window

        if not self.history:
            return
        last = self.history[-1]
        project = getattr(last, "project", None)
        shown = None if self._results_path is None else Path(self._results_path).absolute()
        if project is not None and shown != project:
            self.message.emit(f"not undone: that window was saved in {project}, not in the "
                              "project shown")
            return
        record = self.recording
        try:
            with (record("an elution window's save undone") if record is not None
                  else contextlib.nullcontext()):
                undo_window(last)
        except (OSError, ValueError) as exc:
            self.message.emit(f"not undone: {exc}")
            return
        self.history.pop()
        self.undo.setEnabled(bool(self.history))
        self.message.emit(f"{Path(last.path).name}: the last saved window undone")
        self.saved.emit(str(last.path))
