# SPDX-License-Identifier: GPL-3.0-or-later
"""The review panel of the Results workspace (docs/PLAN.md, "GUI", "Results and review"): the
decision on what is selected, a spectrum, or every spectrum of a time point or a series. A
status (not reviewed, accepted, rejected), a K (the fit's choice or one the run fitted) and a note
are written at once to the project's user review table (:mod:`hxex.gui.review`), the same file a
spreadsheet user edits, and laid over the results in memory, so the tree and the figures show
them at once. A table another program holds keeps the decisions waiting (Try again). A K
without a bootstrap at it offers Run, as ``hxex run`` adds one when it brings the results up to
date.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QStyle,
    QStyleOptionGroupBox,
    QToolButton,
    QToolTip,
    QWidget,
)

from .flow_layout import FlowLayout
from .review import Decision, make_review_table, table_path, table_stamp, write_decisions
from .tips import rich_tip, tip
from .user_tables import TableLocked

log = logging.getLogger(__name__)

# what ◀ ▶ step through: (words, the tree's kind of row)
# (owner: up to the series; a peptide's series have no view together yet)
STEP_LEVELS = (("spectrum", "spectrum"), ("time point", "point"), ("series", "series"))
# (status, words, key in the tree)
STATUS_WORDS = (("", "not reviewed", "U"), ("accepted", "accepted", "A"),
                ("rejected", "rejected", "R"))


class ReviewPanel(QGroupBox):
    """Decisions on the spectra of the selection. :meth:`show_spectra` sets them; the panel
    emits :attr:`decided` with the positions of the spectra whose decision changed."""

    decided = Signal(list)
    resume_asked = Signal()
    step = Signal(int)  # to the previous (-1) or next (+1) spectrum of the tree

    def __init__(self, workspace):
        super().__init__("Review")
        self.workspace = workspace
        self._tree_level = 0  # the step level chosen in the tree (restored for the next tree row)
        self.rows: list[int] = []  # the spectra's positions in the results
        self.pending: dict[str, Decision] = {}  # decisions not written yet (the table held)
        # the notes of the bundle's own review layer, read when first needed: (key, notes)
        self._bundle_cache: tuple[tuple | None, dict[str, str]] = (None, {})
        self.heading = QLabel("Select a spectrum, a time point, or a series.")
        self.heading.setWordWrap(True)
        self.status = QButtonGroup(self)
        # the choices wrap onto a second line when the panel is narrow (in one line they set the
        # results panel's narrowest width: 522 px with Windows' font; owner, 2026-11-08)
        self.status_row = QWidget()
        status_row = FlowLayout(self.status_row)
        for i, (value, words, key) in enumerate(STATUS_WORDS):
            button = QRadioButton(words)
            button.setProperty("status", value)
            button.setToolTip(tip(f"{words} (key {key} in the tree)"))
            self.status.addButton(button, i)
            status_row.addWidget(button)
        self.status.buttonClicked.connect(lambda b: self.decide(status=b.property("status")))
        self.k = QComboBox()
        self.k.setToolTip(tip("the number of populations to use for these spectra (review_K)"))
        self.k.activated.connect(self._k_chosen)
        self.note = QLineEdit()
        self.note.setPlaceholderText("a note (kept with the decision)")
        self.note.editingFinished.connect(self._noted)
        # the note the box shows for the selection: only a different text is written (Qt
        # reports a box emptied by the program as edited when it loses focus, and an empty
        # note written to a series wiped its notes; owner, 2026-11-04)
        self._shown_note = ""
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.retry = QPushButton("Try again")
        self.retry.clicked.connect(lambda: self.decide())
        self.retry.hide()
        self.resume = QPushButton("Add the bootstrap (Run)…")
        self.resume.setToolTip(tip("hxex run: brings the results up to date, which bootstraps "
                                   "the spectra shown at a K without one"))
        self.resume.clicked.connect(self.resume_asked)
        self.resume.hide()
        self.previous = QPushButton("◀")
        self.previous.clicked.connect(lambda: self.step.emit(-1))
        self.next = QPushButton("▶")
        self.next.clicked.connect(lambda: self.step.emit(1))
        for button in (self.previous, self.next):
            button.setFixedWidth(36)
        self.level = QComboBox()  # what ◀ ▶ step through (owner)
        for words, kind in STEP_LEVELS:
            self.level.addItem(words, kind)
        self.level.setToolTip(tip("what ◀ and ▶ step through"))
        self.level.currentIndexChanged.connect(self._level_tips)
        self._level_tips()
        top = QHBoxLayout()
        top.addWidget(self.heading, 1)
        top.addWidget(self.level)
        top.addWidget(self.previous)
        top.addWidget(self.next)
        # the selected spectra's quality flags (hxex.flags): a button at the end of the Status
        # row, their meanings on hover and, on a click, in a box that closes on a click elsewhere
        # (owner, 2026-11-13: a line of its own took the panel's room)
        self.flags = FlagsButton()
        status_line = QHBoxLayout()
        status_line.setContentsMargins(0, 0, 0, 0)
        status_line.addWidget(self.status_row, 1)
        status_line.addWidget(self.flags, 0, Qt.AlignmentFlag.AlignTop)
        # the label as tall as one line of the choices, its text in the middle: it stays beside
        # the first line when they wrap (owner, 2026-11-13: it sat above the choices' middle)
        status_label = QLabel("Status")
        status_label.setFixedHeight(max(self.status.buttons()[0].sizeHint().height(),
                                        self.flags.sizeHint().height()))
        status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        grid = QGridLayout(self)
        grid.addLayout(top, 0, 0, 1, 2)
        grid.addWidget(status_label, 1, 0, Qt.AlignmentFlag.AlignTop)
        grid.addLayout(status_line, 1, 1)
        grid.addWidget(QLabel("K"), 2, 0)
        grid.addWidget(self.k, 2, 1)
        grid.addWidget(QLabel("Note"), 3, 0)
        grid.addWidget(self.note, 3, 1)
        grid.addWidget(self.message, 4, 0, 1, 2)
        buttons = QHBoxLayout()
        buttons.addWidget(self.retry)
        buttons.addWidget(self.resume)
        buttons.addStretch()
        grid.addLayout(buttons, 5, 0, 1, 2)
        # where the whole table is: a "?" on the title line, at its right (owner, 2026-11-17)
        self.table_help = TableHelpButton(self)
        self.show_spectra([], "")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_table_help()

    def _place_table_help(self) -> None:
        """The "?" at the right end of the title line, its bottom on the frame's top line (where
        a style draws that line through the title, level with the title): the box must not
        cover the frame below it (owner, 2026-11-17)."""
        option = QStyleOptionGroupBox()
        self.initStyleOption(option)
        style = self.style()
        label = style.subControlRect(QStyle.ComplexControl.CC_GroupBox, option,
                                     QStyle.SubControl.SC_GroupBoxLabel, self)
        frame = style.subControlRect(QStyle.ComplexControl.CC_GroupBox, option,
                                     QStyle.SubControl.SC_GroupBoxFrame, self)
        bottom = frame.top() if frame.top() >= label.bottom() else label.bottom() + 1
        button = self.table_help
        metrics = button.fontMetrics()
        # a small box: the text's height and a little room on each side of the "?"
        height = max(1, min(metrics.height() + 2, bottom))
        width = metrics.horizontalAdvance("?") + 10
        button.setGeometry(self.width() - width - 4, bottom - height, width, height)
        button.raise_()

    def table_help_text(self) -> str:
        """Where the whole review table is, for the "?" on the title line."""
        project = self.project
        path = None
        if project is not None:
            try:
                path = table_path(project)
            except (OSError, ValueError):
                path = None
        if path is None or not path.exists():
            return "No review table saved yet"
        return (f"Find the whole editable review table {path.name} on the Project → Tables "
                "tab.")

    def _level_tips(self, *_) -> None:
        if not self.level.isEnabled():  # the List tab: row by row
            self.previous.setToolTip(tip("the previous row of the list, in its order"))
            self.next.setToolTip(tip("the next row of the list, in its order"))
            return
        words = self.level.currentText()
        self.previous.setToolTip(tip(f"the previous {words}"))
        self.next.setToolTip(tip(f"the next {words} (through everything the tree shows)"))

    def set_list_mode(self, on: bool) -> None:
        """On the List tab ◀ ▶ step through the list's rows one at a time, so the choice of what
        they step through is greyed out at "spectrum" (owner, 2026-11-08: offered there, it did
        nothing); the Tree tab gets its choice back."""
        if on == (not self.level.isEnabled()):
            return
        if on:
            self._tree_level = self.level.currentIndex()
            self.level.blockSignals(True)
            self.level.setCurrentIndex(0)
            self.level.blockSignals(False)
            self.level.setEnabled(False)
            self.level.setToolTip(tip("the list steps one row at a time, in its order (the tree "
                                      "steps by spectrum, time point, or series)"))
        else:
            self.level.setEnabled(True)
            self.level.blockSignals(True)
            self.level.setCurrentIndex(self._tree_level)
            self.level.blockSignals(False)
            self.level.setToolTip(tip("what ◀ and ▶ step through"))
        self._level_tips()

    # --- what is shown ---------------------------------------------------------------------

    @property
    def project(self) -> Path | None:
        """The project whose review table takes the decisions: only results opened through it."""
        window = self.workspace.window
        if window is None or not window.project_results:
            return None
        return window.project

    def forget_pending(self) -> None:
        """Other results are shown: decisions still waiting for a held table were the results
        before's, and Try again would have written them into the new project's table (audit
        of 2026-11-15, D-7)."""
        if self.pending:
            log.warning("%d review decision(s) waiting for a held table dropped: other "
                        "results are shown (%s)", len(self.pending), ", ".join(self.pending))
        self.pending = {}
        self.retry.hide()
        if self.message.text().startswith("Not saved"):
            self.message.setText("")

    def show_spectra(self, rows: list[int], what: str) -> None:
        """The decision of the spectra at ``rows`` (``what`` names them). A note being typed
        for the spectra shown before is written to them first (on macOS a click on another
        row does not move the keyboard focus out of the box: audit of 2026-11-15, D-3)."""
        self.commit_note()
        self.rows = [int(r) for r in rows]
        results = self.workspace.results
        usable = bool(self.rows) and results is not None and self.project is not None
        for widget in (*self.status.buttons(), self.k, self.note):
            widget.setEnabled(usable)
        n = len(self.rows)
        self.heading.setText(what + (f" ({n} spectra: a decision applies to each)" if n > 1
                                     else ""))
        # results opened without their project file: nowhere to record a decision, said under
        # the Note row, apart from what is selected (owner, 2026-11-13: once in the heading)
        if results is not None and self.rows and self.project is None:
            self.message.setText(RESULTS_ONLY)
            self.message.setStyleSheet("color: gray")
            self.message.setToolTip(tip("decisions go in the project's review table: open the "
                                        "results through their project file (File > Open "
                                        "project file)"))
        elif self.message.text() == RESULTS_ONLY:
            self.message.setText("")
            self.message.setStyleSheet("")
            self.message.setToolTip("")
        self._show_values()

    def _show_values(self) -> None:
        results = self.workspace.results
        self.status.setExclusive(False)
        for b in self.status.buttons():
            b.setChecked(False)
        self.status.setExclusive(True)
        self.k.blockSignals(True)
        self.k.clear()
        self.note.blockSignals(True)
        self.note.setText("")
        self._shown_note = ""
        self.flags.show_flags(flag_items(results, self.rows), len(self.rows))
        if results is None or not self.rows:
            self.k.blockSignals(False)
            self.note.blockSignals(False)
            return
        status = {str(results.review_status[i]) for i in self.rows}
        status = {"" if s in ("unreviewed", "None") else s for s in status}
        if len(status) == 1:
            value = status.pop()
            for b in self.status.buttons():
                b.setChecked(b.property("status") == value)
        chosen = {self._chosen_k(i) for i in self.rows}
        fit_k = {int(results.fit.selected_K[i]) for i in self.rows}
        self.k.addItem("the fit's choice" + (f" (K={next(iter(fit_k))})" if len(fit_k) == 1
                                             else ""), None)
        for k in results.fit.config.k_values:
            self.k.addItem(f"K={k}", int(k))
        if len(chosen) == 1:
            index = self.k.findData(chosen.pop())
            self.k.setCurrentIndex(max(index, 0))
        else:
            self.k.insertItem(0, "(differs between the spectra)", "mixed")
            self.k.setCurrentIndex(0)
        notes = {self._note_of(i) for i in self.rows}
        self.note.setToolTip("")
        if len(notes) == 1:  # one spectrum, or every one with the same note: shown
            self._shown_note = notes.pop()
            self.note.setText(self._shown_note)
            if self._shown_note and self._notes_from_bundle(self._shown_note):
                self.note.setToolTip(tip(BUNDLE_NOTE))
            self.note.setPlaceholderText("a note (kept with the decision)" if len(self.rows) == 1
                                         else "a note for each of these spectra")
        else:  # left empty: a note typed here replaces each one's, leaving it empty keeps them
            self.note.setPlaceholderText("their notes differ: a note typed here replaces them "
                                         "all")
        self.k.blockSignals(False)
        self.note.blockSignals(False)

    def _bundle_notes(self) -> dict[str, str]:
        """The notes the bundle's own review layer holds (set from Python, or carried over by
        a rerun), by spectrum, for the note box's tooltip (audit of 2026-11-15, D-2). Read
        from the bundle once per results."""
        project, results = self.project, self.workspace.results
        if project is None or results is None:
            return {}
        created = results.settings.created if results.settings is not None else ""
        key = (str(project), created)
        if self._bundle_cache[0] != key:
            notes: dict[str, str] = {}
            try:
                from ..io import open_bundle

                bundle = self.workspace.window.fresh_project_state().bundle
                if bundle is not None and Path(bundle).exists():
                    with open_bundle(bundle) as b:
                        review = b.review()
                    notes = {str(s): str(n) for s, n in zip(review.spectrum_id, review.note,
                                                            strict=True) if n}
            except Exception as exc:  # noqa: BLE001 (only the note's tooltip depends on it)
                log.info("the bundle's own review notes were not read: %s", exc)
            self._bundle_cache = (key, notes)
        return self._bundle_cache[1]

    def _notes_from_bundle(self, note: str) -> bool:
        """Whether ``note``, shown for every spectrum of the selection, is the bundle's own."""
        ids = [str(self.workspace.results.fit.spectrum_id[i]) for i in self.rows]
        held = self._bundle_notes()
        return bool(ids) and all(held.get(sid) == note for sid in ids)

    def _note_of(self, i: int) -> str:
        review = self.workspace.results.review
        entry = review.get(str(self.workspace.results.fit.spectrum_id[i])) \
            if review is not None else None
        return str(entry["note"] or "") if entry else ""

    def _chosen_k(self, i: int):
        results = self.workspace.results
        review = results.review
        if review is None:
            return None
        entry = review.get(str(results.fit.spectrum_id[i]))
        return entry["k_override"] if entry and entry["k_override"] else None

    # --- deciding --------------------------------------------------------------------------

    def _k_chosen(self, index: int) -> None:
        data = self.k.itemData(index)
        if data != "mixed":
            self.decide(k=data, k_given=True)

    def _noted(self) -> None:
        self.commit_note()

    def commit_note(self) -> bool:
        """Write the note typed for the spectra shown, when it differs from the one shown (the
        window calls it before reading the results again, so a note being typed is kept).
        False when there was nothing to write or it is waiting."""
        text = self.note.text().strip()
        if not self.rows or self._deciding or text == self._shown_note.strip():
            return False
        return self.decide(note=text)

    _deciding = False  # a decision being written: what it sets off writes no second one

    def decide(self, status: str | None = None, k: int | None = None, k_given: bool = False,
               note: str | None = None) -> bool:
        """Write the decision for the spectra shown (with any still waiting), and lay it over the
        results. False when it is waiting (the table is held) or nothing is shown."""
        results = self.workspace.results
        project = self.project
        if results is None or project is None or self._deciding:
            return False
        self._deciding = True
        try:
            return self._decide(status, k, k_given, note)
        finally:
            self._deciding = False

    def _decide(self, status, k, k_given, note) -> bool:
        results = self.workspace.results
        project = self.project
        if note is None and self.rows:
            # a note being typed joins the decision (a status clicked or a K chosen before the
            # box lost the focus, which on macOS a button click does not take: D-3)
            typed = self.note.text().strip()
            if typed != self._shown_note.strip():
                note = typed
        ids = [str(results.fit.spectrum_id[i]) for i in self.rows]
        if status is not None or k_given or note is not None:
            for sid in ids:
                old = self.pending.get(sid, Decision())
                st = status
                if status == "" and self._layer_status(sid):
                    # not reviewed: written as such, as an empty cell keeps a status the
                    # bundle's own review layer holds (audit of 2026-11-15, D-2)
                    st = "unreviewed"
                if note == "" and st is None and old.status is None \
                        and not (k_given or old.k_given):
                    # the note emptied with no other decision: a row that says something
                    # replaces the stored note with its own (empty) note cell, an empty row
                    # keeps it (the notes rule, owner 2026-11-15): the status shown is
                    # written as it is, or unreviewed
                    st = self._layer_status(sid) or "unreviewed"
                self.pending[sid] = Decision(
                    status=old.status if st is None else st,
                    k=k if k_given else old.k, k_given=old.k_given or k_given,
                    note=old.note if note is None else note)
        if not self.pending:
            return False
        bundle_k = {}
        if results.review is not None:
            for sid in self.pending:
                entry = results.review.get(sid)
                if entry and entry["k_override"]:
                    bundle_k[sid] = entry["k_override"]
        chosen = None
        if table_path(project) is None:  # the first decision: where the table goes (owner)
            from ..review_table import SUGGESTED_NAME
            from .table_choice import ask_table

            chosen = ask_table(self, project, "review.table", SUGGESTED_NAME, "review table")
            if chosen is None:
                self.message.setText("Not saved: no review table chosen")
                return False
        window = self.workspace.window
        try:
            record = getattr(window, "recording", None)
            if window is not None:  # naming the table is no reason to read the results again
                window.naming_review_table = True
            try:
                with (record("the review table named in the project file") if record is not None
                      else contextlib.nullcontext()):
                    path = make_review_table(project, results, chosen)
            finally:
                if window is not None:
                    window.naming_review_table = False
            write_decisions(path, self.pending, bundle_k, results=results)
        except TableLocked as exc:
            self.message.setText(f"Not saved yet: {exc}")
            self.retry.show()
            log.warning("%s", exc)
            return False
        except (OSError, ValueError) as exc:
            self.message.setText(f"Not saved: {exc}")
            log.error("the review decision was not saved: %s", exc)
            return False
        if window is not None:  # the results shown have this table's decisions
            window.results_review_stamp = table_stamp(project)
            window.watch_review_table()  # a table just made: a change elsewhere is seen
        changed = self._lay_over(self.pending)
        log.info("review: %d spectra in %s", len(self.pending), path.name)
        if note is not None:  # what the rows show now (a selection made while the workspace
            self._shown_note = note  # redraws must not write it again)
        self.pending = {}
        self.retry.hide()
        self.message.setText(f"Saved in {path.name}.")
        self._offer_resume()
        self.decided.emit(changed)
        self._show_values()
        return True

    def _layer_status(self, sid: str) -> str:
        """The status the results' review layer holds for ``sid`` ('' for none)."""
        review = self.workspace.results.review
        entry = review.get(sid) if review is not None else None
        return str(entry["status"] or "") if entry else ""

    def _lay_over(self, decisions: dict[str, Decision]) -> list[int]:
        """The decisions laid over the results' review layer in memory (as reading the table
        again does: a row that says something carries its note cell, empty included); the
        positions of the spectra they touch."""
        from ..io.bundle import ReviewTable

        results = self.workspace.results
        review = results.review if results.review is not None else ReviewTable.empty()
        index = {str(s): i for i, s in enumerate(results.fit.spectrum_id)}
        for sid, d in decisions.items():
            status = "" if d.status == "unreviewed" else d.status
            review = review.update(
                sid, status=status, k_override=(d.k or 0) if d.k_given else None, note=d.note)
        self.workspace.results = results.with_review(review)
        return [index[s] for s in decisions if s in index]

    def _offer_resume(self) -> None:
        from ..review_table import missing_bootstraps

        missing = missing_bootstraps(self.workspace.results)
        shown = [str(self.workspace.results.fit.spectrum_id[i]) for i in self.rows]
        waiting = [s for s in shown if s in missing]
        self.resume.setVisible(bool(waiting))
        if waiting:
            unfitted = self.workspace.results.unfitted_choices
            ks = ", ".join(str(k) for k in sorted({missing[s] for s in waiting}))
            if any(s in unfitted for s in waiting):  # the screen skipped it: drawn at the fit's K
                self.message.setText(f"{self.message.text()} K={ks} was not fitted for these "
                                     "spectra (the screen skipped it): shown at the fit's K "
                                     "until Run fits it.")
            else:
                self.message.setText(f"{self.message.text()} No bootstrap at K={ks} for these "
                                     "spectra yet: its errors come with Run.")

    def key(self, letter: str) -> None:
        """A, R or U in the tree: accepted, rejected, not reviewed."""
        status = {key.lower(): value for value, _, key in STATUS_WORDS}.get(letter.lower())
        if status is not None and self.status.buttons()[0].isEnabled():
            self.decide(status=status)


# the note box's tooltip when its note is the bundle's own review layer's (D-2; emptying the
# box clears it, as the row then says something: the notes rule)
BUNDLE_NOTE = "from the run's review layer (one set from Python, or carried over by a rerun)"
# the message line when results are open without their project file
RESULTS_ONLY = ("these results were opened on their own: open their project file to record "
                "decisions")


def flag_items(results, rows: list[int]) -> list[tuple[str, int, str]]:
    """The quality flags of the spectra at ``rows``: each flag's name, how many of them carry it,
    and its meaning (:mod:`hxex.flags`), in the order of the flags."""
    from .. import flags
    from ..quality import FLAG_NAMES

    q = getattr(results, "quality", None) if results is not None else None
    if q is None or not rows:
        return []
    items = []
    for name in FLAG_NAMES:
        n = int(np.sum(np.asarray(getattr(q, name))[rows]))
        if n:
            e = flags.entry(name)
            items.append((name, n, flags.plain(flags.render(e.short)) if e is not None else ""))
    return items


class TableHelpButton(QToolButton):
    """"?" on the review panel's title line: where the whole review table can be seen and
    edited, said on hover or a click. The text is worked out when it is shown, so a table made
    by the first decision, or named in the project file since, is named at once."""

    def __init__(self, panel: ReviewPanel):
        super().__init__(panel)
        self.panel = panel
        self.setText("?")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.clicked.connect(lambda: QToolTip.showText(
            self.mapToGlobal(self.rect().bottomLeft()), self._refresh(), self))
        self._refresh()

    def _refresh(self) -> str:
        text = tip(self.panel.table_help_text())
        self.setToolTip(text)
        return text

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            self._refresh()
        return super().event(event)


class FlagsButton(QToolButton):
    """'Flags (N)': the N quality flags of the spectra shown, each with its meaning on hover (a
    click shows it at once); greyed 'Flags' without any. Its width fits the widest text, so a
    count changing moves nothing. (A box that stayed open on a click added nothing, as it
    closed at the next click anywhere: owner, 2026-11-13.)"""

    def __init__(self):
        super().__init__()
        self.setText("Flags (99)")
        self.setFixedWidth(self.sizeHint().width())
        self.clicked.connect(lambda: QToolTip.showText(
            self.mapToGlobal(self.rect().bottomLeft()), self.toolTip(), self))
        self.show_flags([])

    def show_flags(self, items: list[tuple[str, int, str]], n_spectra: int = 1) -> None:
        """``items`` (:func:`flag_items`) of ``n_spectra`` spectra."""
        self.items = items
        if not items:
            self.setText("Flags")
            self.setEnabled(False)
            self.setToolTip(tip("no quality flags on these spectra"))
            return
        of = (lambda n: f" ({n} of {n_spectra})") if n_spectra > 1 else (lambda n: "")
        self.setText(f"Flags ({len(items)})")
        self.setEnabled(True)
        self.setToolTip(rich_tip("\n".join(f"**{name}**{of(n)}: {meaning}"
                                           for name, n, meaning in items)))
