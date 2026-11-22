# SPDX-License-Identifier: GPL-3.0-or-later
"""Where the project's results stand, stage by stage, in the Run tab (docs/PLAN.md Next steps
40, step 4): one mark per stage of the results (read, pick, noise, fit, population test,
correction and flags, bootstrap; :mod:`hxex.stages`) and, after a divider, one per kind of
output made from them (tables, figures, analysis folders), each ``✓`` done, ``!`` out of date
(bold, in the link colour), ``○`` not done, or ``–`` off, with its causes and what ``hxex run``
does for it in its tooltip; under them, the Run button and what it does now (owner,
2026-11-21): **Run** (no results yet; greyed out when the results are up to date) or **Run
(update)**. **Start again…** (``hxex run --fresh``) is kept apart, at the right end of the
commands' row, behind a warning.

Working the standing out takes seconds on a real project (5 to 18 s measured; nearly all of it
the check of the bootstrap at each set's K), so it runs off the window's thread
(:class:`hxex.gui.tasks.Latest`: a newer request replaces an older one) in two passes: the quick
one without that check, then the whole. The marks stay where they are meanwhile, greyed. The
log lines the check gives (an old bundle's notes, read again at every check) are kept out of the
window's messages and the terminal: the run and the commands say them."""

from __future__ import annotations

import contextlib
import html
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..stages import DONE, LABELS, NOT_DONE, OFF, OUT_OF_DATE, OUTPUT_LABELS, STAGES
from .flow_layout import FlowLayout
from .tasks import Latest
from .tips import tip

log = logging.getLogger(__name__)

SYMBOLS = {DONE: "✓", OUT_OF_DATE: "!", NOT_DONE: "○", OFF: "–"}
SETTLE_MS = 500  # a request waits this long for more changes (a save reports several)
RUN, UPDATE = "Run", "Run (update)"

# --- the check's log lines kept quiet --------------------------------------------------------

_QUIET: set[int] = set()  # the threads working a standing out now


class _QuietFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread not in _QUIET


QUIET = _QuietFilter()


def keep_quiet(handler: logging.Handler) -> None:
    """Keep the stage check's log lines out of ``handler`` (once)."""
    if QUIET not in handler.filters:
        handler.addFilter(QUIET)


@contextlib.contextmanager
def _quiet():
    me = threading.get_ident()
    _QUIET.add(me)
    try:
        yield
    finally:
        _QUIET.discard(me)


# --- the standing, worked out off the window's thread -----------------------------------------

@dataclass
class RowState:
    """What the row shows: ``problem`` (the project file does not load: the row says only
    that), the stages and outputs (:class:`hxex.stages.Stage`), ``run`` (what Run does),
    ``notes``, whether there are results, whether the project can run (an ``[input]``), whether
    the results are ``up_to_date``, and whether the whole check was done (``complete``)."""

    path: Path | None = None
    problem: str = ""
    stages: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    plan: list = field(default_factory=list)
    run: str = ""
    notes: list = field(default_factory=list)
    has_results: bool = False
    can_run: bool = True
    up_to_date: bool = False
    complete: bool = True

    @property
    def late(self) -> list:
        """The stages of the results that ``hxex run`` would do (out of date or not done)."""
        return [s for s in self.stages if s.state in (OUT_OF_DATE, NOT_DONE)]


def work_out(path: Path, slow: bool) -> RowState:
    """Where the results of the project file at ``path`` stand (``slow``: with the check of the
    bootstrap at each set's K), as the row shows it."""
    from ..config import ProjectError, load_project
    from ..stages import output_marks, output_states, standing

    with _quiet():
        try:
            project = load_project(path, analysis_only=True)
        except (ProjectError, OSError, ValueError) as exc:
            return RowState(path, problem=str(exc) or type(exc).__name__)
        bundle = project.output.bundle
        has = bundle is not None and Path(bundle).exists()
        st = standing(project, bundle, slow=slow)
        outputs = output_marks(output_states(project)) if bundle is not None else []
        return RowState(path, stages=st.stages, outputs=outputs, plan=st.plan, run=st.run,
                        notes=st.notes, has_results=has, can_run=project.input is not None,
                        up_to_date=has and st.up_to_date, complete=st.complete)


# --- the row ------------------------------------------------------------------------------------

class StageRow(QWidget):
    """The marks, the line saying what Run does, and the Run button (``run_button``: the Run
    workspace connects it; Start again is with the other commands, :mod:`hxex.gui.
    run_workspace`)."""

    changed = Signal()  # a new standing is shown (the window's status-bar line follows it)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Path | None = None
        self.state: RowState | None = None  # the standing shown (None: none yet)
        self.paused = False  # a command of the project runs: no check until it ends
        self.checking = False
        self._latest = Latest(self)
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(SETTLE_MS)
        self._settle.timeout.connect(self._start)

        self.title = QLabel("Results")
        self.title.setToolTip(tip("where the results stand against the project file, stage by "
                                  "stage: ✓ done; ! out of date (Run does it again); ○ not done; "
                                  "– off (not asked for); the reason is in each mark's tooltip"))
        self.marks: dict[str, QLabel] = {}
        flow = FlowLayout(spacing=4)
        for i, name in enumerate(STAGES):
            if i:
                flow.addWidget(QLabel("─"))
            self.marks[name] = QLabel()
            flow.addWidget(self.marks[name])
        self.divider = QLabel("  │  ")  # the stages of the results, then what is made from them
        flow.addWidget(self.divider)
        for name in OUTPUT_LABELS:
            self.marks[name] = QLabel()
            flow.addWidget(self.marks[name])
        self.problem = QLabel()
        self.problem.setWordWrap(True)
        self.problem.hide()
        marks = QWidget()
        marks.setLayout(flow)
        self.marks_widget = marks

        self.plan = QLabel()
        self.plan.setWordWrap(True)
        self.run_button = QPushButton(UPDATE)
        # as wide as its longer words: the line beside it stays put when the words change
        self.run_button.setMinimumWidth(self.run_button.sizeHint().width())
        self.run_button.setText(RUN)
        top = QHBoxLayout()
        top.addWidget(self.title)
        top.addWidget(marks, 1)
        bottom = QHBoxLayout()  # the buttons first, on the left (owner, 2026-11-21)
        bottom.addWidget(self.run_button)
        bottom.addWidget(self.plan, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self.problem)
        layout.addLayout(bottom)
        self._show()

    # --- asking -------------------------------------------------------------------------------

    def show_project(self, path: Path | None) -> None:
        """The project file shown (None: none): its standing worked out again."""
        if path != self.project:
            self.project = path
            self.state = None
            self._latest.drop()
        self.refresh(now=True)

    def refresh(self, now: bool = False) -> None:
        """Work the standing out again (after :data:`SETTLE_MS`, unless ``now``); nothing
        while a command of the project runs (:meth:`pause`)."""
        if self.project is None:
            self._latest.drop()
            self.checking = False
            self._show()
            return
        if self.paused:
            return
        if now:
            self._settle.stop()
            self._start()
        else:
            self._settle.start()

    def pause(self, on: bool) -> None:
        """A command of the project runs (``on``): the row says so and checks nothing until it
        ends; then it is worked out again."""
        self.paused = on
        if on:
            self._settle.stop()
            self._latest.drop()
            self.checking = False
            self._show()
        else:
            self.refresh(now=True)

    def _start(self, slow: bool = False) -> None:
        if self.project is None or self.paused:
            return
        path = self.project
        self.checking = True
        self._show()
        self._latest.submit(lambda: work_out(path, slow),
                            lambda state: self._arrived(state, slow),
                            self._failed)

    def _arrived(self, state: RowState, slow: bool) -> None:
        if state.path != self.project or self.paused:
            return
        self.state = state
        if not slow and not state.complete and not state.problem:
            self._start(slow=True)  # the whole check, the quick answer shown meanwhile
        else:
            self.checking = False
        self._show()
        self.changed.emit()

    def _failed(self, message: str) -> None:
        log.info("the results could not be checked: %s", message)
        self.checking = False
        self.state = RowState(self.project, problem=f"the results could not be checked: "
                                                    f"{message}")
        self._show()
        self.changed.emit()

    # --- showing ------------------------------------------------------------------------------

    @property
    def run_label(self) -> str:
        """Run with no results yet (or none known), Run (update) when they are out of date."""
        state = self.state
        return UPDATE if state is not None and state.has_results and state.late else RUN

    def run_enabled(self) -> bool:
        """Whether Run does something: not when the results are known to be up to date, the
        project file does not load, or it has no ``[input]``."""
        state = self.state
        if state is None:
            return self.project is not None
        return not state.problem and state.can_run and not state.up_to_date

    def fresh_enabled(self) -> bool:
        state = self.state
        return (state is not None and not state.problem and state.can_run
                and state.has_results)

    def summary(self) -> str:
        """``results out of date (3 stages)`` for the status bar while the results are out of
        date; "" otherwise (none known, up to date, no results, a command running)."""
        state = self.state
        if state is None or self.paused or state.problem or not state.has_results:
            return ""
        n = len(state.late)
        if not n:
            return ""
        return f"results out of date ({n} stage{'s' if n != 1 else ''})"

    def _show(self) -> None:
        state = self.state
        grey = self.palette().color(QPalette.ColorRole.PlaceholderText).name()
        link = self.palette().color(QPalette.ColorRole.Link).name()
        problem = state.problem if state is not None else ""
        self.problem.setVisible(bool(problem))
        self.marks_widget.setVisible(not problem)
        if problem:
            self.problem.setText(f"The project file does not load: {problem}"
                                 if not problem.startswith("where the results stand")
                                 else problem[:1].upper() + problem[1:])
        by_name = {}
        if state is not None and not problem:
            by_name = {s.name: s for s in [*state.stages, *state.outputs]}
        dim = self.checking or self.paused or state is None
        for name, label in self.marks.items():
            text = LABELS.get(name) or OUTPUT_LABELS[name]
            stage = by_name.get(name)
            if name in OUTPUT_LABELS:
                label.setVisible(stage is not None)
            if stage is None:
                label.setText(f'<span style="color:{grey}">{html.escape(text)}</span>')
                label.setToolTip(tip("not checked yet" if not self.paused
                                     else "a command is running"))
                continue
            symbol = SYMBOLS[stage.state]
            body = f"{symbol} {html.escape(text)}"
            if stage.state == OUT_OF_DATE:
                body = f"<b>{body}</b>"
            colour = (grey if dim or stage.state == OFF
                      else link if stage.state == OUT_OF_DATE else None)
            label.setText(f'<span style="color:{colour}">{body}</span>' if colour else body)
            label.setToolTip(self._mark_tip(stage, state))
        self.divider.setVisible(bool(state and state.outputs) and not problem)
        self.plan.setText(self._plan_text())
        self.plan.setToolTip(tip("; ".join([state.run, *state.notes]) if state is not None
                                 and state.run else None))
        self.run_button.setText(self.run_label)
        self.run_button.setToolTip(tip(self._run_tip()))

    def _mark_tip(self, stage, state: RowState) -> str:
        lines = [f"{stage.label}: {stage.state}"]
        lines += [f"  {cause}" for cause in stage.causes]
        doing = [text for name, text in state.plan if name == stage.name]
        if doing and stage.state != OFF:
            lines.append("Run: " + ", ".join(doing))
        if stage.name == "bootstrap" and not state.complete:
            lines.append("(still checking the bootstrap at each set's K)")
        return "\n".join(lines)

    def _plan_text(self) -> str:
        state = self.state
        if self.project is None:
            return ""
        if self.paused:
            return "A command is running: the results are checked again when it ends."
        if state is None:
            return "Checking the results…"
        if state.problem:
            return ""
        if not state.can_run:
            return "This project file has no [input], so it cannot run (it analyses results)."
        if not state.has_results:
            return "No results yet: Run does every stage."
        if state.up_to_date:
            return "Nothing to do: the results are up to date."
        text = "Run: " + state.run
        return text + (" (checking…)" if self.checking else "")

    def _run_tip(self) -> str:
        state = self.state
        if state is not None and state.up_to_date:
            return "hxex run: the results are up to date, nothing to do (Start again does " \
                   "everything anew)"
        if state is not None and state.has_results:
            return "hxex run: brings the results up to date, doing again only what your " \
                   "changes touch (your review decisions are kept)"
        return "hxex run: read, pick, fit, and bootstrap: the results bundle"
