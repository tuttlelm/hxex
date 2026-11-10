# SPDX-License-Identifier: GPL-3.0-or-later
"""The review panel of the Results workspace (docs/PLAN.md, "GUI", "Results and review"): the
decision on what is selected, a spectrum, or every spectrum of a time point or a series. A
status (not reviewed, accepted, rejected), a K (the fit's choice or one the run fitted) and a note
are written at once to the project's user review table (:mod:`hxex.gui.review`), the same file a
spreadsheet user edits, and laid over the results in memory, so the tree and the figures show
them at once. A table another program holds keeps the decisions waiting (Try again). A K
without a bootstrap at it offers Resume, as ``hxex resume`` adds one.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
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
    QWidget,
)

from .flow_layout import FlowLayout
from .review import Decision, make_review_table, table_stamp, write_decisions
from .tips import tip
from .user_tables import TableLocked

log = logging.getLogger(__name__)

# what ◀ ▶ step through: (words, the tree's kind of row)
# (owner: up to the charge, which is a series; a peptide's charges have no view together yet)
STEP_LEVELS = (("spectrum", "spectrum"), ("time point", "point"), ("charge", "series"))
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
        self.rows: list[int] = []  # the spectra's positions in the results
        self.pending: dict[str, Decision] = {}  # decisions not written yet (the table held)
        self.heading = QLabel("Select a spectrum, a time point or a series.")
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
        self.resume = QPushButton("Add the bootstrap (Resume)…")
        self.resume.setToolTip(tip("hxex resume: bootstraps the spectra shown at a K without one"))
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
        grid = QGridLayout(self)
        grid.addLayout(top, 0, 0, 1, 2)
        grid.addWidget(QLabel("Status"), 1, 0, Qt.AlignmentFlag.AlignTop)  # by its first line
        grid.addWidget(self.status_row, 1, 1)
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
        self.show_spectra([], "")

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
                                      "steps by spectrum, time point or charge)"))
        else:
            self.level.setEnabled(True)
            self.level.blockSignals(True)
            self.level.setCurrentIndex(getattr(self, "_tree_level", 0))
            self.level.blockSignals(False)
            self.level.setToolTip(tip("what ◀ and ▶ step through"))
        self._level_tips()

    # --- what is shown ---------------------------------------------------------------------

    @property
    def project(self) -> Path | None:
        """The project whose review table takes the decisions: only results opened through it."""
        window = self.workspace.window
        if window is None or window.project is None or window.results_path != window.project:
            return None
        return window.project

    def show_spectra(self, rows: list[int], what: str) -> None:
        """The decision of the spectra at ``rows`` (``what`` names them)."""
        self.rows = [int(r) for r in rows]
        results = self.workspace.results
        usable = bool(self.rows) and results is not None and self.project is not None
        for widget in (*self.status.buttons(), self.k, self.note):
            widget.setEnabled(usable)
        if results is not None and self.rows and self.project is None:
            self.heading.setText(f"{what}. Decisions go in the project's review table: open the "
                                 "results through the project file (File > Open project file) "
                                 "to record them.")
        else:
            n = len(self.rows)
            self.heading.setText(what + (f" ({n} spectra: a decision applies to each)" if n > 1
                                         else ""))
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
        if len(notes) == 1:  # one spectrum, or every one with the same note: shown
            self._shown_note = notes.pop()
            self.note.setText(self._shown_note)
            self.note.setPlaceholderText("a note (kept with the decision)" if len(self.rows) == 1
                                         else "a note for each of these spectra")
        else:  # left empty: a note typed here replaces each one's, leaving it empty keeps them
            self.note.setPlaceholderText("their notes differ: a note typed here replaces them "
                                         "all")
        self.k.blockSignals(False)
        self.note.blockSignals(False)

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
        if not self.rows or text == self._shown_note.strip():
            return False
        return self.decide(note=text)

    def decide(self, status: str | None = None, k: int | None = None, k_given: bool = False,
               note: str | None = None) -> bool:
        """Write the decision for the spectra shown (with any still waiting), and lay it over the
        results. False when it is waiting (the table is held) or nothing is shown."""
        results = self.workspace.results
        project = self.project
        if results is None or project is None:
            return False
        ids = [str(results.fit.spectrum_id[i]) for i in self.rows]
        if status is not None or k_given or note is not None:
            for sid in ids:
                old = self.pending.get(sid, Decision())
                self.pending[sid] = Decision(
                    status=old.status if status is None else status,
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
        try:
            path = make_review_table(project, results)
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
        if self.workspace.window is not None:  # the results shown have this table's decisions
            self.workspace.window.results_review_stamp = table_stamp(project)
        changed = self._lay_over(self.pending)
        log.info("review: %d spectra in %s", len(self.pending), path.name)
        self.pending = {}
        self.retry.hide()
        self.message.setText(f"Saved in {path.name}.")
        self._offer_resume()
        self.decided.emit(changed)
        self._show_values()
        return True

    def _lay_over(self, decisions: dict[str, Decision]) -> list[int]:
        """The decisions laid over the results' review layer in memory (as reading the table
        again would); the positions of the spectra they touch."""
        from ..io.bundle import ReviewTable

        results = self.workspace.results
        review = results.review if results.review is not None else ReviewTable.empty()
        index = {str(s): i for i, s in enumerate(results.fit.spectrum_id)}
        for sid, d in decisions.items():
            review = review.update(
                sid, status=d.status, k_override=(d.k or 0) if d.k_given else None,
                note=d.note)
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
                                     "until Resume fits it.")
            else:
                self.message.setText(f"{self.message.text()} No bootstrap at K={ks} for these "
                                     "spectra yet: its errors come with Resume.")

    def key(self, letter: str) -> None:
        """A, R or U in the tree: accepted, rejected, not reviewed."""
        status = {key.lower(): value for value, _, key in STATUS_WORDS}.get(letter.lower())
        if status is not None and self.status.buttons()[0].isEnabled():
            self.decide(status=status)


def counts(results, data_id: int) -> tuple[int, int]:
    """A series' rejected spectra and those with more than one population (not rejected)."""
    rows = np.flatnonzero(results.fit.data_id == data_id)
    status = np.asarray(results.review_status).astype(str)[rows]
    rejected = int((status == "rejected").sum())
    multi = int(((results.effective_K[rows] > 1) & (status != "rejected")).sum())
    return rejected, multi
