# SPDX-License-Identifier: GPL-3.0-or-later
"""The Results workspace's flat list of spectra (docs/PLAN.md, "GUI", "Results and review"):
every spectrum of the results, one row each, with the columns of ``spectra.csv`` (as
``hxex report`` writes it, the review applied) and why the analysis leaves it out; sorted by a
click on a column's header, narrowed by a filter, a sample and a search, and stepped through in
the order shown, to work through a dataset's problems one spectrum after another (owner,
2026-11-03). Choosing a row shows that spectrum as the tree does, with the review panel.

A right click on a column's header filters by that column (owner, 2026-11-04): ticks among its
values when it has few (:data:`FEW_VALUES`; a time is ticked by its label), a range when it
holds many numbers (a time with its units, as in a project file), or text it contains; the
filters of several columns combine with the list's own filter, sample and search.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QItemSelectionModel, Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .data_table import RowsModel, cell_text, size_columns
from .flow_layout import FlowLayout
from .order import natural_key
from .results_tree import search_matches
from .table_copy import (
    ask_csv_path,
    chosen_rows,
    install_copy,
    rows_text,
    save_rows,
    saved_or_said,
    set_clipboard,
)
from .tips import ERROR_COLOUR, set_choice_tips, tip

FEW_VALUES = 30  # a column with at most this many values is filtered by ticking them
# the report's tables the list shows (owner, 2026-11-04: the Results tables, read only)
TABLES = {"spectra": "**spectra.csv**: one row per spectrum",
          "series": "**series.csv**: one row per series (a peptide at one charge in one sample)",
          "components": "**components.csv**: one row per spectrum and population",
          "uptake": "**uptake.csv**: replicate averages per series and time"}

# the list's filters: (words, what they keep)
FILTERS = {"all": ("all spectra", "every spectrum"),
           "flagged": ("flagged", "spectra with a quality flag"),
           "multi": ("K above 1", "spectra with more than one population"),
           "unreviewed": ("not reviewed", "spectra not accepted or rejected yet"),
           "rejected": ("rejected", "spectra rejected in the review"),
           "left_out": ("left out", "spectra the analysis settings leave out")}


def list_rows(results, left: list[str] | None = None, rows=None, *,
              sets=None) -> tuple[list[str], list[dict]]:
    """The list's columns and rows: ``spectra.csv``'s (:func:`hxex.report.tables.spectra_table`;
    ``rows``: only those spectra, by position; ``sets``: the test of each spectrum's set,
    :func:`hxex.analysis.set_test.spectrum_sets`, for ``set_K``, None for none), and
    ``left_out`` (``left``, the tree's reasons) after the review columns. Each row also carries
    its position (``_index``, not shown)."""
    from ..report.series import time_text
    from ..report.tables import spectra_table

    table = spectra_table(results, rows=rows, sets=sets)
    picked = range(len(results.fit)) if rows is None else rows
    columns = list(table)
    at = columns.index("note") + 1 if "note" in columns else len(columns)
    columns.insert(at, "left_out")
    if "time" in columns:  # the time as the figures write it, beside the seconds
        columns.insert(columns.index("time") + 1, "time_label")
    scale = getattr(results, "time_scale", "seconds")
    out = []
    for n, index in enumerate(picked):
        row = {name: plain_value(values[n]) for name, values in table.items()}
        row["left_out"] = left[index] if left is not None else ""
        row["time_label"] = time_text(float(row.get("time", math.nan)),
                                       str(row.get("role", "")), scale)
        row["_index"] = int(index)
        out.append(row)
    return columns, out


def plain_value(value):
    """A table value as Python's own (numbers, text, booleans), NaN kept for numbers."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value if isinstance(value, (int, float, str)) else str(value)


_plain = plain_value  # the name other modules import


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
    table_asked = Signal(str)  # another table chosen: its rows are wanted (set_rows)
    update_asked = Signal()  # the table shown, made again with the latest decisions

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table_name = "spectra"
        self.tables = QComboBox()
        for name in TABLES:
            self.tables.addItem(f"{name}.csv", name)
        set_choice_tips(self.tables, TABLES)  # each table's own tooltip, the chosen one's shown
        self.tables.currentIndexChanged.connect(
            lambda _i: self.table_asked.emit(self.tables.currentData()))
        self.save_button = QPushButton("Save table…")
        self.save_button.setToolTip(tip("save the rows shown, in the order shown, as a CSV file"))
        self.save_button.clicked.connect(self.save_dialog)
        self.stale = QLabel()
        self.stale.setWordWrap(True)
        self.update_button = QPushButton("Update")
        self.update_button.setToolTip(tip("make this table again with your latest decisions"))
        self.update_button.clicked.connect(self.update_asked.emit)
        self.columns: list[str] = []
        self.rows: list[dict] = []  # every spectrum's row (the model shows the filtered ones)
        # the row chosen when the list was cleared for a reread of the results: chosen again,
        # quietly, when the rows come back (audit of 2026-11-15, C open question 5)
        self._kept_key: tuple | None = None
        self.filter = QComboBox()
        for key, (text, _meaning) in FILTERS.items():
            self.filter.addItem(text, key)
        set_choice_tips(self.filter, {key: f"**{text}**: {meaning}"
                                      for key, (text, meaning) in FILTERS.items()})
        self.sample = QComboBox()
        self.sample.setToolTip(tip("one sample's spectra, or every sample's"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find: residue 20, peptide 12-23, or text")
        self.search.setToolTip(tip("every word must match; 20: a residue the peptide covers; "
                                   "12-23: its residues; other words: sample, peptide, or "
                                   "spectrum id"))
        self.search.setClearButtonEnabled(True)
        self.count = QLabel("Building the list…")
        self.column_filters: dict[str, tuple] = {}  # column -> ('values', set), ('range', lo,
        # hi) or ('contains', text)
        self.clear_filters = QPushButton("Clear column filters")
        self.clear_filters.setToolTip(tip("show every row again (the filters set by a right click "
                                          "on a column's header)"))
        self.clear_filters.clicked.connect(self.clear_column_filters)
        self.clear_filters.hide()
        for box in (self.filter, self.sample):
            box.currentIndexChanged.connect(lambda _i: self._apply())
        self.search.textChanged.connect(lambda _t: self._apply())
        # wraps onto further lines when the list is narrow (owner, 2026-11-08: in one row it
        # set the window's narrowest width)
        self.bar = QWidget()
        bar = FlowLayout(self.bar)
        self.search.setMinimumWidth(200)
        self.search.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        for w in (self.tables, self.filter, self.sample, self.clear_filters, self.save_button):
            bar.addWidget(w)
        bar.add_filling(self.search)  # the rest of its line (owner, 2026-11-08)
        # the count (x of y spectra) is not in the bar: the Results workspace shows it on its
        # status line under the list, so that the list can be narrow
        stale = QHBoxLayout()
        stale.setContentsMargins(0, 0, 0, 0)
        stale.addWidget(self.stale, 1)
        stale.addWidget(self.update_button)
        self.stale_bar = QWidget()
        self.stale_bar.setLayout(stale)
        self.stale_bar.hide()
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
        install_copy(self.grid, lambda: self.copy(every=False))
        self.grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self._menu)
        header = self.grid.horizontalHeader()
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(
            lambda pos: self.ask_filter(header.logicalIndexAt(pos)))
        header.setToolTip(tip("click: sort by the column; right-click: filter by it"))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.bar)
        layout.addWidget(self.stale_bar)
        layout.addWidget(self.grid, 1)

    def set_rows(self, columns: list[str], rows: list[dict], table: str = "spectra") -> None:
        """Show a table's rows (``list_rows`` for the spectra); the sample and search stay,
        and so do the column filters of the same table. The filter list is the spectra's."""
        if table != self.table_name:
            self.column_filters = {}
            self.model.marked = {}
            self.clear_filters.hide()
        self.table_name = table
        self.filter.setVisible(table == "spectra")
        self.stale_bar.hide()
        if self.tables.currentData() != table:
            self.tables.blockSignals(True)
            self.tables.setCurrentIndex(max(self.tables.findData(table), 0))
            self.tables.blockSignals(False)
            # the box's tooltip follows the choice through its signal, blocked here
            self.tables.setToolTip(self.tables.itemData(self.tables.currentIndex(),
                                                        Qt.ItemDataRole.ToolTipRole) or "")
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
            size_columns(self.grid)
        kept, self._kept_key = self._kept_key, None
        if kept is not None and self.current_row() is None:
            for i, row in enumerate(self.model.rows):
                if _row_key(row) == kept:
                    self.select_index(i, emit=False)
                    break

    def clear(self, message: str = "") -> None:
        current = self.current_row()
        if current is not None:
            self._kept_key = _row_key(current)
        self.columns, self.rows = [], []
        self.model.set_rows([], [])
        self.count.setText(message)

    def shown(self, row: dict) -> bool:
        """Whether ``row`` passes the filter, the sample and the search."""
        how = (self.filter.currentData() or "all") if self.table_name == "spectra" else "all"
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
        text = self.search.text().strip()
        if text and not search_matches(text, str(row.get("sample", "")),
                                       str(row.get("peptide_range", "")),
                                       str(row.get("peptide", "")),
                                       f"{row.get('spectrum_id')} {row.get('identity', '')} "
                                       f"{row.get('candidates', '')}"):
            return False
        return all(_passes(row.get(column), rule) for column, rule in self.column_filters.items())

    # --- the column filters -------------------------------------------------------------------

    def filter_kind(self, column: str) -> str:
        """How ``column`` is filtered: ``'values'`` (few values, ticked), ``'range'`` (many
        numbers) or ``'contains'`` (much text)."""
        shown = {cell_text(r.get(column)) for r in self.rows}
        if len(shown) <= FEW_VALUES:
            return "values"
        numbers = [r.get(column) for r in self.rows if cell_text(r.get(column))]
        if numbers and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           for v in numbers):
            return "range"
        return "contains"

    def values_of(self, column: str) -> list[str]:
        """The values of a few-valued column as shown, in a sensible order (numbers by size, a
        time by time, the rest with their numbers by value; empty last)."""
        seen: dict[str, object] = {}
        for r in self.rows:
            seen.setdefault(cell_text(r.get(column)), _order_key(r, column))
        return sorted(seen, key=lambda text: (text == "", seen[text]))

    def set_column_filter(self, column: str, rule: tuple | None) -> None:
        """Filter by ``column`` with ``rule`` (None: no filter on it), and show the result."""
        if rule is None:
            self.column_filters.pop(column, None)
        else:
            self.column_filters[column] = rule
        self.model.marked = {c: _describe(c, r) for c, r in self.column_filters.items()}
        self.clear_filters.setVisible(bool(self.column_filters))
        self._apply()
        self.model.headerDataChanged.emit(Qt.Orientation.Horizontal, 0,
                                          max(self.model.columnCount() - 1, 0))

    def clear_column_filters(self) -> None:
        self.column_filters = {}
        self.set_column_filter("", None)

    def ask_filter(self, section: int) -> None:
        """The filter of the column at ``section`` (a right click on its header), asked in a
        small window."""
        if not 0 <= section < len(self.model.columns):
            return
        column = self.model.columns[section]
        dialog = ColumnFilterDialog(self, column)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.set_column_filter(column, dialog.rule())

    def _apply(self) -> None:
        """Show the rows the filter, sample and search keep, in the order sorted; the row chosen
        stays chosen (by the row itself, whichever table: audit of 2026-11-15, C-5)."""
        current = self.current_row()
        shown = [r for r in self.rows if self.shown(r)]
        self.model.set_rows(self.columns, shown)
        header = self.grid.horizontalHeader()
        if header.sortIndicatorSection() >= 0:
            self.model.sort(header.sortIndicatorSection(), header.sortIndicatorOrder())
        if self.rows:  # spectra.csv has one row per spectrum; the others count rows (Q-41)
            what = "spectra" if self.table_name == "spectra" else "rows"
            self.count.setText(f"{len(shown):,} of {len(self.rows):,} {what}")
        if current is not None:
            for i, row in enumerate(self.model.rows):
                if row is current:
                    self.select_index(i, emit=False)
                    break

    def copied_text(self, every: bool) -> str:
        """The row chosen (``every``: every row shown, in the order shown), with the header,
        tab-separated: a spreadsheet takes it as rows and columns (:mod:`hxex.gui.table_copy`)."""
        rows = self.model.rows if every else [self.model.rows[i] for i in chosen_rows(self.grid)
                                              if i < len(self.model.rows)]
        return rows_text(self.model.columns, rows)

    def copy(self, every: bool = False) -> None:
        set_clipboard(self.copied_text(every))

    def row_menu(self):
        """The right-click menu: Copy row, Copy rows shown."""
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        one = menu.addAction("Copy row")
        one.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        one.setEnabled(self.current_row() is not None)
        one.triggered.connect(lambda: self.copy(every=False))
        every = menu.addAction(f"Copy rows shown ({len(self.model.rows):,})")
        every.setEnabled(bool(self.model.rows))
        every.triggered.connect(lambda: self.copy(every=True))
        return menu

    def _menu(self, pos) -> None:
        index = self.grid.indexAt(pos)
        if index.isValid():  # the row right-clicked is chosen, as by a left click
            self.grid.selectionModel().setCurrentIndex(
                index, QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QItemSelectionModel.SelectionFlag.Rows)
        menu = self.row_menu()
        menu.exec(self.grid.viewport().mapToGlobal(pos))
        menu.deleteLater()

    def mark_stale(self) -> None:
        """The table shown is from before the latest decisions (not made again until asked)."""
        self.stale.setText(f"{self.table_name}.csv is from before your latest review "
                           "decisions.")
        self.stale_bar.show()

    def save_dialog(self) -> None:
        """Save the rows shown as a CSV file chosen in the save dialog; a file that cannot be
        written is said on the count line (audit of 2026-11-15, Q-13: it raised inside the
        slot)."""
        if not self.model.rows:
            return
        path = ask_csv_path(self, f"{self.table_name}.csv")
        if path is not None:
            saved_or_said(lambda: self.save(path), self.count.setText)

    def save(self, path):
        """Write the rows shown, in the order shown (``hxex report``'s CSV writing)."""
        return save_rows(self.model.columns, self.model.rows, path)

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

    def set_red(self, spectrum_ids) -> None:
        """Draw the rows of ``spectrum_ids`` in red (windows saved since the run), the others
        as before."""
        red = set(spectrum_ids)
        if red == self.model.red:
            return
        self.model.red = red
        top, bottom = self.model.index(0, 0), self.model.index(
            max(self.model.rowCount() - 1, 0), max(self.model.columnCount() - 1, 0))
        self.model.dataChanged.emit(top, bottom, [Qt.ItemDataRole.ForegroundRole])

    def current_row(self) -> dict | None:
        index = self.grid.selectionModel().currentIndex()
        if not index.isValid() or index.row() >= len(self.model.rows):
            return None
        return self.model.rows[index.row()]

    def select(self, spectrum_id: str, emit: bool = True) -> bool:
        """Make the first row of ``spectrum_id`` current (False when it is not shown)."""
        for i, row in enumerate(self.model.rows):
            if str(row.get("spectrum_id")) == spectrum_id:
                self.select_index(i, emit=emit)
                return True
        return False

    def select_index(self, i: int, emit: bool = True) -> None:
        """Make the row at position ``i`` (as shown) current; ``emit``: as a click would
        (:attr:`chosen`), else quietly."""
        if not emit:
            self.grid.selectionModel().blockSignals(True)
        index = self.model.index(i, 0)
        self.grid.selectionModel().setCurrentIndex(
            index, QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QItemSelectionModel.SelectionFlag.Rows)
        self.grid.scrollTo(index)
        if not emit:
            self.grid.selectionModel().blockSignals(False)

    def step(self, step: int) -> bool:
        """The next (``step`` 1) or previous (-1) row, in the order shown, by its place (a
        spectrum with two populations has two rows of components.csv: audit of 2026-11-15, C-6);
        False at either end."""
        n = self.model.rowCount()
        if not n:
            return False
        index = self.grid.selectionModel().currentIndex()
        row = (0 if step > 0 else n - 1) if not index.isValid() else index.row() + step
        if not 0 <= row < n:
            return False
        self.select_index(row)
        return True

    def _current(self, current, _previous) -> None:
        if current.isValid() and current.row() < len(self.model.rows):
            self.chosen.emit(self.model.rows[current.row()])


class ColumnFilterDialog(QDialog):
    """The filter of one column: ticks for few values, from and to for numbers (a time with its
    units), or text it contains; Clear removes it."""

    def __init__(self, owner: SpectraList, column: str):
        super().__init__(owner)
        self.column, self.kind = column, owner.filter_kind(column)
        now = owner.column_filters.get(column)
        self.setWindowTitle(f"Filter: {column}")
        layout = QVBoxLayout(self)
        self.cleared = False
        if self.kind == "values":
            self.ticks = QListWidget()
            for text in owner.values_of(column):
                item = QListWidgetItem(text or "(empty)")
                item.setData(Qt.ItemDataRole.UserRole, text)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                ticked = now is None or text in now[1]
                item.setCheckState(Qt.CheckState.Checked if ticked else Qt.CheckState.Unchecked)
                self.ticks.addItem(item)
            every, none = QPushButton("All"), QPushButton("None")
            every.clicked.connect(lambda: self._tick_all(True))
            none.clicked.connect(lambda: self._tick_all(False))
            row = QHBoxLayout()
            row.addWidget(every)
            row.addWidget(none)
            row.addStretch(1)
            layout.addWidget(QLabel("Show the rows with the values ticked:"))
            layout.addWidget(self.ticks)
            layout.addLayout(row)
        elif self.kind == "range":
            time = column == "time"
            self.low, self.high = QLineEdit(), QLineEdit()
            for box, value in ((self.low, now[1] if now else None),
                               (self.high, now[2] if now else None)):
                box.setPlaceholderText("any")
                if value is not None:
                    box.setText(f"{value:g}")
            layout.addWidget(QLabel("Show the rows from, to (either empty: no limit)"
                                    + (", in seconds or with units (30 s, 2 min)" if time
                                       else "") + ":"))
            row = QHBoxLayout()
            row.addWidget(QLabel("from"))
            row.addWidget(self.low)
            row.addWidget(QLabel("to"))
            row.addWidget(self.high)
            layout.addLayout(row)
        else:
            self.text = QLineEdit(now[1] if now else "")
            self.text.setPlaceholderText("text the value contains")
            layout.addWidget(QLabel("Show the rows whose value contains:"))
            layout.addWidget(self.text)
        self.problem = QLabel()
        self.problem.setStyleSheet(f"color: {ERROR_COLOUR};")
        self.problem.hide()
        layout.addWidget(self.problem)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        clear = buttons.addButton("Clear", QDialogButtonBox.ButtonRole.ResetRole)
        clear.clicked.connect(self._clear)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _tick_all(self, on: bool) -> None:
        for i in range(self.ticks.count()):
            self.ticks.item(i).setCheckState(Qt.CheckState.Checked if on
                                             else Qt.CheckState.Unchecked)

    def _clear(self) -> None:
        self.cleared = True
        self.accept()

    def _accept(self) -> None:
        try:
            self.rule()
        except ValueError as exc:
            self.problem.setText(str(exc))
            self.problem.show()
            return
        self.accept()

    def rule(self) -> tuple | None:
        """The filter chosen (None: none, every value ticked or Clear pressed)."""
        if self.cleared:
            return None
        if self.kind == "values":
            ticked = {self.ticks.item(i).data(Qt.ItemDataRole.UserRole)
                      for i in range(self.ticks.count())
                      if self.ticks.item(i).checkState() == Qt.CheckState.Checked}
            return None if len(ticked) == self.ticks.count() else ("values", ticked)
        if self.kind == "range":
            low, high = (_bound(box.text(), self.column) for box in (self.low, self.high))
            return None if low is None and high is None else ("range", low, high)
        text = self.text.text().strip()
        return ("contains", text) if text else None


def _bound(text: str, column: str) -> float | None:
    """A range's limit as typed: empty for none, a number, or a time with units."""
    text = text.strip()
    if not text:
        return None
    if column == "time":
        from ..io.times import parse_time

        try:
            return parse_time(text, "the time", allow_negative=True)
        except Exception as exc:  # noqa: BLE001 - said in the window
            raise ValueError(f"not a time: {text!r} (a number of seconds, or 30 s, 2 min, "
                             "1 h)") from exc
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"not a number: {text!r}") from None


def _passes(value, rule: tuple) -> bool:
    kind = rule[0]
    if kind == "values":
        return cell_text(value) in rule[1]
    if kind == "range":
        x = _number(value)
        if math.isnan(x):
            return False
        return (rule[1] is None or x >= rule[1]) and (rule[2] is None or x <= rule[2])
    return rule[1].lower() in str(value if value is not None else "").lower()


def _describe(column: str, rule: tuple) -> str:
    """A column's filter in words, for its header's tooltip."""
    if rule[0] == "values":
        shown = sorted((t or "(empty)" for t in rule[1]), key=natural_key)
        return f"{column}: {', '.join(shown[:8])}" + (" ..." if len(shown) > 8 else "")
    if rule[0] == "range":
        low = "" if rule[1] is None else f"from {rule[1]:g} "
        high = "" if rule[2] is None else f"to {rule[2]:g}"
        return f"{column}: {low}{high}".strip()
    return f"{column}: contains {rule[1]!r}"


def _order_key(row: dict, column: str):
    """How a few-valued column's values are ordered in its ticks: a time label by its time
    (controls at the ends), numbers by size, text with its numbers by value (residue ranges
    2-12, 13-30, 108-127: :func:`natural_key`)."""
    if column == "time_label":
        from ..spectra import point_order_key

        return point_order_key(_number(row.get("time")), str(row.get("role", "")))
    value = row.get(column)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value) if not math.isnan(float(value)) else math.inf, ())
    return (1, 0.0, natural_key(value))


# the columns that tell a row of any of the tables apart, for finding it again
_KEY_COLUMNS = ("spectrum_id", "data_id", "time", "role", "rep", "description", "K", "rank")


def _row_key(row: dict) -> tuple:
    """What tells ``row`` apart from the other rows of its table."""
    return tuple((c, row.get(c)) for c in _KEY_COLUMNS if c in row)


def _number(value) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x


__all__ = ["FILTERS", "SpectraList", "list_rows"]
