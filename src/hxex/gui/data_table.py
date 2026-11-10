# SPDX-License-Identifier: GPL-3.0-or-later
"""The data tab beside a figure (docs/PLAN.md, "GUI"): the values of what the figure draws, one
row per bar or line (:func:`hxex.report.hover.set_row`, gathered by
:func:`hxex.gui.figures.figure_rows`), so the table holds exactly what the figure shows. Read
only: sorted by a click on a column's header, copied (Ctrl+C, with the header), saved as CSV
(as *hxex*'s other tables are written), and a row double-clicked, or chosen from its right-click
menu (:func:`spectra_menu`, also the figure's), goes to its spectra.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .order import natural_key
from .tips import tip

log = logging.getLogger(__name__)


def cell_text(value) -> str:
    """A value as the table shows it: numbers to four significant digits (whole numbers as
    they are), booleans as yes or nothing, NaN as nothing."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "" if math.isnan(float(value)) else repr(float(value))
        return f"{float(value):.4g}"
    return str(value)


MENU_LIMIT = 12  # peptides listed in a right-click menu; the rest are in the data tab


def spectra_choices(rows: list[dict]) -> list[tuple[str, dict]]:
    """What a right-click menu offers for ``rows`` (data-table rows): for each peptide, its
    spectra in the row's sample and, on a comparison (a Woods bar), in the reference too, as
    (the menu's words, the row to go to); rows without a peptide's sample and residues give
    nothing."""
    out: list[tuple[str, dict]] = []
    for row in rows:
        if not {"sample", "start", "end"} <= set(row):
            continue
        name = f"{row.get('peptide', '')} ({int(row['start'])}–{int(row['end'])})".strip()
        out.append((f"Spectra of {name}, {row['sample']}", row))
        if row.get("reference"):
            out.append((f"Spectra of {name}, {row['reference']}",
                        {**row, "sample": row["reference"]}))
    return out


def spectra_menu(rows: list[dict], go, parent=None) -> QMenu | None:
    """A menu of :func:`spectra_choices` (up to :data:`MENU_LIMIT` peptides), calling
    ``go(row)`` for the one chosen; None when there is nothing to offer."""
    choices = spectra_choices(rows[:MENU_LIMIT])
    if not choices:
        return None
    menu = QMenu(parent)
    for text, row in choices:
        menu.addAction(text).triggered.connect(lambda _=False, r=row: go(r))
    if len(rows) > MENU_LIMIT:
        more = menu.addAction(f"and {len(rows) - MENU_LIMIT} more: see the data tab")
        more.setEnabled(False)
    return menu


def _sort_key(value):
    """Numbers before text, missing values (None, NaN, '') last, whichever the direction; text
    in :func:`natural_key` order."""
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return (2, 0.0, ())
    if isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating)):
        return (0, float(value), ())
    return (1, 0.0, natural_key(value))


class RowsModel(QAbstractTableModel):
    """Rows of values (dicts) under named columns, read only, sortable."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.columns: list[str] = []
        self.rows: list[dict] = []
        self.marked: dict[str, str] = {}  # filtered columns: their filter in words

    def set_rows(self, columns: list[str], rows: list[dict]) -> None:
        self.beginResetModel()
        self.columns, self.rows = list(columns), list(rows)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):  # noqa: N802, B008 - Qt's name and signature
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):  # noqa: N802, B008 - Qt's name and signature
        return 0 if parent.isValid() else len(self.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.rows[index.row()].get(self.columns[index.column()])
        if role == Qt.ItemDataRole.DisplayRole:
            return cell_text(value)
        if role == Qt.ItemDataRole.ToolTipRole:
            return tip(cell_text(value))
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(
                value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if orientation != Qt.Orientation.Horizontal or not 0 <= section < len(self.columns):
            return None
        name = self.columns[section]
        if role == Qt.ItemDataRole.DisplayRole:
            return f"{name} ▼" if name in self.marked else name  # a filter on it
        if role == Qt.ItemDataRole.ToolTipRole and name in self.marked:
            return tip(f"filtered: {self.marked[name]} (right-click to change)")
        return None

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        if not 0 <= column < len(self.columns):
            return
        name = self.columns[column]
        self.layoutAboutToBeChanged.emit()
        present = [r for r in self.rows if _sort_key(r.get(name))[0] < 2]
        missing = [r for r in self.rows if _sort_key(r.get(name))[0] == 2]
        present.sort(key=lambda r: _sort_key(r.get(name)),
                     reverse=order == Qt.SortOrder.DescendingOrder)
        self.rows = present + missing  # missing values last either way
        self.layoutChanged.emit()


class DataPanel(QWidget):
    """The data tab: the figure's rows in a table, with Save table… and a count."""

    activated = Signal(dict)  # a row double-clicked: go to its spectra

    def __init__(self, parent=None):
        super().__init__(parent)
        self.save_name = ""  # the file name the save dialog suggests (the figure's)
        self.model = RowsModel(self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSortIndicatorShown(True)
        self.table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.doubleClicked.connect(self._double_clicked)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._menu)
        self.count = QLabel()
        self.save_button = QPushButton("Save table…")
        self.save_button.setToolTip(tip("save these rows, in the order shown, as a CSV file (as "
                                        "hxex's other tables)"))
        self.save_button.clicked.connect(self.save_dialog)
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(self.save_button)
        bar.addWidget(self.count, 1)
        self._bar = bar
        hint = QLabel("Click a column's header to sort. Ctrl+C copies the rows selected. "
                      "Double-click a peptide's row, or right-click it, to go to its spectra.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholder-text);")
        self.note = QLabel()  # what the rows are (from which table), when said
        self.note.setWordWrap(True)
        self.note.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.note)
        layout.addWidget(self.table, 1)
        layout.addWidget(hint)
        self.show_rows([], [])

    def show_rows(self, columns: list[str], rows: list[dict], name: str = "") -> None:
        """Show ``rows`` under ``columns`` (in the figure's order); ``name``: the figure's."""
        self.model.set_rows(columns, rows)
        self.table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.save_name = name
        self.save_button.setEnabled(bool(rows))
        self.count.setText(f"{len(rows)} row(s)" if rows else
                           "No data table for this figure.")
        if rows:
            self.table.resizeColumnsToContents()

    def clear(self) -> None:
        self.show_rows([], [])
        self.set_note("")

    def add_option(self, widget: QWidget) -> None:
        """Put ``widget`` (a choice of what the rows are) in the bar, after the count."""
        self._bar.addWidget(widget)

    def set_note(self, text: str) -> None:
        """A line above the table saying what its rows are ('' for none)."""
        self.note.setText(text)
        self.note.setVisible(bool(text))

    def select_row(self, row: dict) -> bool:
        """Select the row equal to ``row`` and scroll to it (False when it is not shown)."""
        for i, shown in enumerate(self.model.rows):
            if shown == row:
                self.table.selectRow(i)
                self.table.scrollTo(self.model.index(i, 0))
                return True
        return False

    def _double_clicked(self, index) -> None:
        if index.isValid():
            self.activated.emit(self.model.rows[index.row()])

    def row_menu(self, index) -> QMenu | None:
        """The right-click menu of the row at ``index``: its spectra (a peptide's row), then
        Copy (the rows selected, as Ctrl+C); None outside the rows."""
        if not index.isValid():
            return None
        menu = (spectra_menu([self.model.rows[index.row()]], self.activated.emit, self)
                or QMenu(self))
        if menu.actions():
            menu.addSeparator()
        copy = menu.addAction("Copy")
        copy.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        copy.triggered.connect(self.copy)
        return menu

    def _menu(self, pos) -> None:
        menu = self.row_menu(self.table.indexAt(pos))
        if menu is not None:
            menu.exec(self.table.viewport().mapToGlobal(pos))
            menu.deleteLater()

    # --- copying and saving ------------------------------------------------------------------

    def keyPressEvent(self, event):  # noqa: N802 - Qt's name
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy()
            event.accept()
            return
        super().keyPressEvent(event)

    def copy(self) -> None:
        """The selected rows (all when none is selected), with the header, as tab-separated
        text a spreadsheet pastes into columns."""
        picked = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        rows = [self.model.rows[i] for i in picked] or self.model.rows
        lines = ["\t".join(self.model.columns)]
        lines += ["\t".join(cell_text(r.get(c)) for c in self.model.columns) for r in rows]
        QApplication.clipboard().setText("\n".join(lines) + "\n")

    def save_dialog(self) -> None:
        if not self.model.rows:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save table",
                                              f"{self.save_name or 'figure'}.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            self.save(path)
        except OSError as exc:
            self.count.setText(f"Not saved: {exc}")
            log.error("the table was not saved: %s", exc)

    def save(self, path: str | Path) -> Path:
        """Write the rows, in the order shown, as CSV (:func:`hxex.report.tables.write_csv`:
        NaN empty, numbers to six significant digits)."""
        from ..report.tables import write_csv

        columns = {name: np.array([r.get(name, "") for r in self.model.rows], dtype=object)
                   for name in self.model.columns}
        path = write_csv(columns, path)
        log.info("saved the table %s", path)
        self.count.setText(f"{len(self.model.rows)} row(s); saved as {Path(path).name}")
        return path


__all__ = ["DataPanel", "RowsModel", "cell_text", "spectra_choices", "spectra_menu"]
