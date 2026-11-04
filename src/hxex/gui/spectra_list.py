# SPDX-License-Identifier: GPL-3.0-or-later
"""The Results workspace's flat list of spectra (docs/PLAN.md, "GUI", "Results and review"):
every spectrum of the results, one row each, with the columns of ``spectra.csv`` (as
``hxex report`` writes it, the review applied) and why the analysis leaves it out; sorted by a
click on a column's header, narrowed by a filter, a sample and a search, and stepped through in
the order shown, to work through a dataset's problems one spectrum after another (owner,
2026-11-03). Choosing a row shows that spectrum as the tree does, with the review panel.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QItemSelectionModel, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .data_table import RowsModel

# the list's filters: (words, what they keep)
FILTERS = {"all": ("all spectra", "every spectrum"),
           "flagged": ("flagged", "spectra with a quality flag"),
           "multi": ("K above 1", "spectra with more than one population"),
           "unreviewed": ("not reviewed", "spectra not accepted or rejected yet"),
           "rejected": ("rejected", "spectra rejected in the review"),
           "left_out": ("left out", "spectra the analysis settings leave out")}


def list_rows(results, left: list[str] | None = None, rows=None) -> tuple[list[str], list[dict]]:
    """The list's columns and rows: ``spectra.csv``'s (:func:`hxex.report.tables.spectra_table`;
    ``rows``: only those spectra, by position), and ``left_out`` (``left``, the tree's reasons)
    after the review columns. Each row also carries its position (``_index``, not shown)."""
    from ..report.tables import spectra_table

    table = spectra_table(results, rows=rows)
    picked = range(len(results.fit)) if rows is None else rows
    columns = list(table)
    at = columns.index("note") + 1 if "note" in columns else len(columns)
    columns.insert(at, "left_out")
    out = []
    for n, index in enumerate(picked):
        row = {name: _plain(values[n]) for name, values in table.items()}
        row["left_out"] = left[index] if left is not None else ""
        row["_index"] = int(index)
        out.append(row)
    return columns, out


def _plain(value):
    """A table value as Python's own (numbers, text, booleans), NaN kept for numbers."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value if isinstance(value, (int, float, str)) else str(value)


class _Grid(QTableView):
    """The list's table, whose keys A, R and U are review decisions (as the tree's)."""

    review_key = Signal(str)

    def keyPressEvent(self, event):  # noqa: N802 - Qt's name
        if event.modifiers() in (Qt.KeyboardModifier.NoModifier,
                                 Qt.KeyboardModifier.ShiftModifier) \
                and event.text().lower() in ("a", "r", "u"):
            self.review_key.emit(event.text())
            event.accept()
            return
        super().keyPressEvent(event)


class SpectraList(QWidget):
    """Every spectrum as a sortable, filtered list; :attr:`chosen` gives the row picked."""

    chosen = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.columns: list[str] = []
        self.rows: list[dict] = []  # every spectrum's row (the model shows the filtered ones)
        self.filter = QComboBox()
        for i, (key, (text, meaning)) in enumerate(FILTERS.items()):
            self.filter.addItem(text, key)
            self.filter.setItemData(i, meaning, Qt.ItemDataRole.ToolTipRole)
        self.filter.setToolTip("which spectra the list shows")
        self.sample = QComboBox()
        self.sample.setToolTip("one sample's spectra, or every sample's")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a peptide, residues or spectrum")
        self.search.setClearButtonEnabled(True)
        self.count = QLabel("Building the list…")
        for box in (self.filter, self.sample):
            box.currentIndexChanged.connect(lambda _i: self._apply())
        self.search.textChanged.connect(lambda _t: self._apply())
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(self.filter)
        bar.addWidget(self.sample)
        bar.addWidget(self.search, 1)
        bar.addWidget(self.count)
        self.model = RowsModel(self)
        self.grid = _Grid()
        self.grid.setModel(self.model)
        self.grid.setSortingEnabled(True)
        self.grid.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.grid.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.grid.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.grid.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.grid.setAlternatingRowColors(True)
        self.grid.verticalHeader().setVisible(False)
        self.grid.selectionModel().currentRowChanged.connect(self._current)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.grid, 1)

    def set_rows(self, columns: list[str], rows: list[dict]) -> None:
        """Show the results' spectra (``list_rows``); the filter, sample and search stay."""
        self.columns, self.rows = columns, rows
        keep = self.sample.currentData()
        samples = sorted({str(r.get("sample", "")) for r in rows})
        self.sample.blockSignals(True)
        self.sample.clear()
        self.sample.addItem("every sample", None)
        for name in samples:
            self.sample.addItem(name, name)
        self.sample.setCurrentIndex(max(self.sample.findData(keep), 0))
        self.sample.blockSignals(False)
        self._apply()
        if rows:
            self.grid.resizeColumnsToContents()

    def clear(self, message: str = "") -> None:
        self.columns, self.rows = [], []
        self.model.set_rows([], [])
        self.count.setText(message)

    def shown(self, row: dict) -> bool:
        """Whether ``row`` passes the filter, the sample and the search."""
        how = self.filter.currentData() or "all"
        status = str(row.get("review_status", ""))
        if how == "flagged" and not row.get("flags"):
            return False
        if how == "multi" and not _number(row.get("K")) > 1:
            return False
        if how == "unreviewed" and status in ("accepted", "rejected"):
            return False
        if how == "rejected" and status != "rejected":
            return False
        if how == "left_out":
            why = str(row.get("left_out") or "")
            if not why or why.startswith("kept"):  # kept at K = 1 is not left out
                return False
        sample = self.sample.currentData()
        if sample is not None and str(row.get("sample")) != sample:
            return False
        text = self.search.text().strip().lower()
        if text:
            words = " ".join(str(row.get(k, "")) for k in
                             ("sample", "peptide_range", "peptide", "spectrum_id")).lower()
            if text not in words:
                return False
        return True

    def _apply(self) -> None:
        """Show the rows the filter, sample and search keep, in the order sorted."""
        current = self.current_row()
        shown = [r for r in self.rows if self.shown(r)]
        self.model.set_rows(self.columns, shown)
        header = self.grid.horizontalHeader()
        if header.sortIndicatorSection() >= 0:
            self.model.sort(header.sortIndicatorSection(), header.sortIndicatorOrder())
        self.count.setText(f"{len(shown):,} of {len(self.rows):,} spectra" if self.rows else
                           self.count.text())
        if current is not None:
            self.select(current["spectrum_id"], emit=False)

    def update_rows(self, rows: list[dict]) -> None:
        """Rows made again (after review decisions): their values replaced in place, the
        filter left as it is (a row decided stays in view until the filter changes)."""
        by_index = {r["_index"]: r for r in rows}
        for row in self.rows:
            new = by_index.get(row["_index"])
            if new is not None:
                row.update(new)
        top, bottom = self.model.index(0, 0), self.model.index(
            max(self.model.rowCount() - 1, 0), max(self.model.columnCount() - 1, 0))
        self.model.dataChanged.emit(top, bottom)

    def current_row(self) -> dict | None:
        index = self.grid.selectionModel().currentIndex()
        if not index.isValid() or index.row() >= len(self.model.rows):
            return None
        return self.model.rows[index.row()]

    def select(self, spectrum_id: str, emit: bool = True) -> bool:
        """Make the row of ``spectrum_id`` current (False when it is not shown)."""
        for i, row in enumerate(self.model.rows):
            if str(row.get("spectrum_id")) == spectrum_id:
                if not emit:
                    self.grid.selectionModel().blockSignals(True)
                index = self.model.index(i, 0)
                self.grid.selectionModel().setCurrentIndex(
                    index, QItemSelectionModel.SelectionFlag.ClearAndSelect
                    | QItemSelectionModel.SelectionFlag.Rows)
                self.grid.scrollTo(index)
                if not emit:
                    self.grid.selectionModel().blockSignals(False)
                return True
        return False

    def step(self, step: int) -> bool:
        """The next (``step`` 1) or previous (-1) row, in the order shown; False at either
        end."""
        n = self.model.rowCount()
        if not n:
            return False
        index = self.grid.selectionModel().currentIndex()
        row = (0 if step > 0 else n - 1) if not index.isValid() else index.row() + step
        if not 0 <= row < n:
            return False
        self.select(str(self.model.rows[row]["spectrum_id"]))
        return True

    def _current(self, current, _previous) -> None:
        if current.isValid() and current.row() < len(self.model.rows):
            self.chosen.emit(self.model.rows[current.row()])


def _number(value) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x


__all__ = ["FILTERS", "SpectraList", "list_rows"]
