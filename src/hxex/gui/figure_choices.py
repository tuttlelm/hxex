# SPDX-License-Identifier: GPL-3.0-or-later
"""The choices inside a kind of figure (the report redesign, step 1; owner, 2026-11-22): the
Analysis tree has a row per kind at each place (a sample, a group), and what varies among its
figures there (the sample, time, form, or page) is chosen in a row of lists above the figure,
each with ◀ and ▶, instead of a row per figure.

The lists follow each other: a list offers only the values the figures chosen in the lists
before it have (the times of the sample chosen, the pages of that sample and time), keeping
its value when it can. A list with nothing to choose is hidden (a kind with one figure shows
no lists at all): the tree's row names the sample, the figure's title its time.
"""

from __future__ import annotations

import re

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget

from ..report.catalogue import PAGE, TIME, FigureChoices
from .tips import tip

_WORDS = {
    "sample": "which sample's figure",
    TIME: "which time point's figure",
    "form": "which form of the pool (the first entry: the figure of the samples)",
    "class": "which class of the pool (the first entry: the figure of the samples)",
    "parent": "which parent peptide (sample, residues, and precursor charge)",
    "condition and ion": "which condition and ion series (and fragment charge)",
    "ion": "which ion series (and fragment charge)",
    "ions": "which pair of ion series (N- and C-terminal)",
    PAGE: "which page of the figure (a figure that grows goes on over pages)",
}


def shown_value(label: str, value: str, n_pages: int = 0) -> str:
    """A value as the list shows it: a time as written (``1p5min`` -> ``1.5 min``), a page with
    the number of pages, the figure of the samples beside a pool's forms named so."""
    if label == PAGE:
        return f"{value} of {n_pages}" if n_pages else value
    if value == "":
        return "(the samples)"
    if label == TIME:
        value = re.sub(r"(?<=\d)p(?=\d)", ".", value)
        return re.sub(r"(?<=\d)(?=[a-zµ])", " ", value)
    return value


_CHARGE = re.compile(r"_z\d+\Z")


def _offered(label: str, keep: str | None, values: list[str]) -> str | None:
    """``keep``, or the value of ``values`` that names the same thing: a fragment figure's
    parent is named with its precursor charge (``S_2-12_z3``) in the traces, ladders, and
    closure, without it in the residue rates of a parent read at several charges
    (``S_2-12``), so moving between them kept no parent and showed the first (owner,
    2026-11-23): the same parent is kept, at the same charge where there is one."""
    if keep is None or keep in values or label != "parent":
        return keep
    base = _CHARGE.sub("", keep)
    return next((v for v in values if _CHARGE.sub("", v) == base), keep)


class ChoicesRow(QWidget):
    """A list (with ◀ ▶) for each part that varies among one kind's figures."""

    chosen = Signal(str)  # the stem of the figure the lists now name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._choices: FigureChoices | None = None
        self._lists: dict[str, tuple[QLabel, QPushButton, QComboBox, QPushButton]] = {}
        # what was chosen in each list, kept while kinds without it are shown: a parent with no
        # closure figure showed the first parent there, and that one followed to the next kind
        # (owner, 2026-11-23)
        self._remembered: dict[str, str] = {}
        self._missing: list[tuple[str, str, str]] = []  # (list, chosen, shown instead)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.note = QLabel()  # says when this kind has no figure for what was chosen
        self.note.setStyleSheet("font-style: italic")
        self.note.hide()
        self._layout.addWidget(self.note)
        self._layout.addStretch(1)
        self.hide()

    @property
    def stem(self) -> str | None:
        """The figure the lists name (None: no figures)."""
        if self._choices is None:
            return None
        return self._choices.stem(self.values())

    def values(self) -> dict[str, str]:
        """The value chosen in every list."""
        return {label: box.currentData() for label, (_, _, box, _) in self._lists.items()}

    def set_figures(self, choices: FigureChoices | None, keep: dict[str, str] | None = None,
                    stem: str | None = None) -> str | None:
        """Show the lists of ``choices`` (None: none), choosing the figure ``stem`` when it is
        one of them, else the values of ``keep`` where they are offered (the sample and time of
        the kind shown before), else the first figure. Says nothing (no :attr:`chosen`): the
        figure chosen is returned."""
        self._choices = choices
        for widgets in self._lists.values():
            for w in widgets:  # hidden now: deleted only when the event loop runs
                self._layout.removeWidget(w)
                w.hide()
                w.deleteLater()
        self._lists = {}
        if choices is None or not choices.entries:
            self.hide()
            return None
        wanted = {**(keep or {}), **self._remembered}
        if stem is not None and choices.of(stem):
            wanted = choices.of(stem)
        self._missing = []
        for label in choices.labels:
            name = QLabel(label)
            box = QComboBox()
            box.setToolTip(tip(_WORDS.get(label, f"which {label}")))
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            prev, nxt = QPushButton("◀"), QPushButton("▶")
            for button, step, words in ((prev, -1, "the previous one (from the first: the "
                                                   "last)"),
                                        (nxt, 1, "the next one (from the last: the first)")):
                button.setFixedWidth(30)
                button.setToolTip(tip(words))
                button.clicked.connect(lambda _c=False, b=box, d=step: self._step(b, d))
            at = self._layout.indexOf(self.note)  # before the note and the stretch
            for k, w in enumerate((name, prev, box, nxt)):
                self._layout.insertWidget(at + k, w)
            self._lists[label] = (name, prev, box, nxt)
            box.currentIndexChanged.connect(lambda _i, lab=label: self._changed(lab))
        self._fill(0, wanted)
        self._show_note()
        self.show()
        return self.stem

    def _show_note(self) -> None:
        missing = self._missing
        if not missing:
            self.note.hide()
            return
        label, wanted, _shown = missing[0]
        self.note.setText(f"no figure for {shown_value(label, wanted)}")
        self.note.show()

    def _fill(self, start: int, wanted: dict[str, str]) -> None:
        """Fill the lists from the ``start``-th on with the values the lists before them allow,
        each keeping its value in ``wanted`` (else its current one) when it is offered."""
        labels = self._choices.labels
        for i in range(start, len(labels)):
            label = labels[i]
            name, prev, box, nxt = self._lists[label]
            before = {lab: self._lists[lab][2].currentData() for lab in labels[:i]}
            values = self._choices.values(label, before)
            asked = wanted.get(label, box.currentData())
            keep = _offered(label, asked, values)
            box.blockSignals(True)
            box.clear()
            for v in values:
                box.addItem(shown_value(label, v, len(values)), v)
            box.setCurrentIndex(values.index(keep) if keep in values else 0)
            box.blockSignals(False)
            if label != PAGE:  # pages differ from kind to kind
                chosen = box.currentData()
                if label not in self._remembered or (keep == asked and keep in values):
                    self._remembered[label] = chosen
                elif keep not in values and i == 0 and asked is not None \
                        and len(values) > 1:
                    # the first list (the sample, form, class, or a fragment's parent): this
                    # kind has no figure for it; the one remembered comes back after (a list of
                    # one value is hidden: the tree's row names it)
                    self._missing.append((label, asked, chosen))
            # nothing to choose (the tree's row names the sample, the figure its time)
            hidden = len(values) <= 1
            for w in (name, box):
                w.setVisible(not hidden)
            for button in (prev, nxt):
                button.setVisible(not hidden)
                button.setEnabled(len(values) > 1)

    def _changed(self, label: str) -> None:
        i = self._choices.labels.index(label)
        self._remembered[label] = self._lists[label][2].currentData()  # chosen here
        self._missing = []
        self._fill(i + 1, {})
        self._show_note()
        stem = self.stem
        if stem is not None:
            self.chosen.emit(stem)

    @staticmethod
    def _step(box: QComboBox, step: int) -> None:
        if box.count():
            box.setCurrentIndex((box.currentIndex() + step) % box.count())

    def box(self, label: str) -> QComboBox:
        """The list of ``label`` (for tests)."""
        return self._lists[label][2]
