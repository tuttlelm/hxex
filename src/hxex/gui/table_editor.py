# SPDX-License-Identifier: GPL-3.0-or-later
"""The Tables view of the Project workspace: the tables the project file names (the peptide
table or table of files, the retention-time table, the user review table), edited as a grid
(docs/PLAN.md, "GUI", "Editing what users edit").

Unlike the settings, a table's changes are kept until Save (a row is often filled in several
steps). Save checks the table with its reader first (:func:`hxex.gui.user_tables.table_problem`)
and offers to save anyway when it has a problem (a table being filled in); a file changed outside
the window since it was opened is not written over without asking; a file a spreadsheet holds is
reported and the changes kept. Leaving a table with unsaved changes asks first.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QFileSystemWatcher, QModelIndex, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .order import natural_key
from .user_tables import (
    ReviewMatch,
    Table,
    TableKind,
    TableLocked,
    kinds,
    read_table,
    results_ids,
    review_match,
    table_problem,
    write_table,
)

log = logging.getLogger(__name__)

# a review-table row naming a spectrum the results do not have: a translucent red, readable on
# light and dark backgrounds
UNMATCHED = QColor(220, 50, 47, 70)
UNMATCHED_TEXT = QColor("#b3261e")


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


class TableModel(QAbstractTableModel):
    """The editor's table for the grid: its rows in the order shown (``order``, indices into
    ``table.rows``), edited in place. A model, not a cell object per value: a review table of
    26,000 spectra and 43 columns took 4-5 s to show and again for every sort (owner: a click
    during it landed on another header)."""

    def __init__(self, editor: TableEditor):
        super().__init__(editor)
        self.editor = editor

    @property
    def table(self) -> Table | None:
        return self.editor.table

    def rowCount(self, parent=QModelIndex()):  # noqa: N802, B008 - Qt's name and signature
        return 0 if parent.isValid() or self.table is None else len(self.editor._order)

    def columnCount(self, parent=QModelIndex()):  # noqa: N802, B008 - Qt's name and signature
        return 0 if parent.isValid() or self.table is None else len(self.table.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.editor._order[index.row()]
        if role in (Qt.ItemDataRole.BackgroundRole, Qt.ItemDataRole.ToolTipRole):
            return self.editor.row_mark(row, role)
        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return None
        return self.table.rows[row][index.column()]

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):  # noqa: N802 - Qt's name
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        row = self.table.rows[self.editor._order[index.row()]]
        if row[index.column()] == str(value):
            return False
        row[index.column()] = str(value)
        self.dataChanged.emit(index, index, [role])
        self.editor._changed()
        return True

    def flags(self, index):
        return (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEditable)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if orientation == Qt.Orientation.Vertical and self.table is not None \
                and role in (Qt.ItemDataRole.ForegroundRole, Qt.ItemDataRole.ToolTipRole) \
                and 0 <= section < len(self.editor._order):
            return self.editor.row_mark(self.editor._order[section], role, header=True)
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.table.columns[section] if self.table is not None else None
        # the row's place in the file, also when the view is sorted (one less than a
        # spreadsheet's number, which counts the header)
        return str(self.editor._order[section] + 1) if section < len(self.editor._order) \
            else str(section + 1)

    def refresh(self) -> None:
        self.beginResetModel()
        self.endResetModel()


class _Grid(QTableView):
    """The editor's grid, whose Ctrl+C copies every selected cell (Qt's own copied only the
    current one; owner, 2026-11-04)."""

    def __init__(self, editor: TableEditor):
        super().__init__()
        self.editor = editor

    def keyPressEvent(self, event):  # noqa: N802 - Qt's name
        if event.matches(QKeySequence.StandardKey.Copy):
            self.editor.copy_selection()
            event.accept()
            return
        super().keyPressEvent(event)


class TableEditor(QWidget):
    """One table as a grid, with Save and Revert."""

    saved = Signal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.kind: TableKind | None = None
        self.path: Path | None = None
        self.project: Path | None = None
        self.table: Table | None = None
        self.modified = False
        self._stamp = None
        self.watcher = QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._file_changed)

        self.heading = QLabel()
        font = self.heading.font()
        font.setBold(True)
        self.heading.setFont(font)
        self.problem = QLabel()
        self.problem.setWordWrap(True)
        self.problem.setStyleSheet("color: #b00020;")
        self.problem.hide()
        # review-table rows naming spectra the results do not have (owner, 2026-11-08: hxex
        # check's warning named the spectrum, not where it is in the table)
        self._results = None  # the results' spectrum ids (user_tables.ResultsIds)
        self.match: ReviewMatch | None = None
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet(f"color: {UNMATCHED_TEXT.name()};")
        self.next_unmatched = QPushButton("Show the next one")
        self.next_unmatched.setToolTip("select the next shaded row")
        self.next_unmatched.clicked.connect(lambda checked=False: self.show_next_unmatched())
        self.notice_row = QWidget()
        notice = QHBoxLayout(self.notice_row)
        notice.setContentsMargins(0, 0, 0, 0)
        notice.addWidget(self.notice, 1)
        notice.addWidget(self.next_unmatched, 0, Qt.AlignmentFlag.AlignTop)
        self.notice_row.hide()
        # a click on a column's header sorts the view (owner, 2026-11-04); the file keeps its
        # own order. The header no longer selects the column (a cell in it does, for removing)
        self._order: list[int] = []  # the table's rows in the order shown
        self._sort: tuple[int, bool] | None = None  # (column, ascending)
        self.model = TableModel(self)
        self.grid = _Grid(self)
        self.grid.setModel(self.model)
        self.grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self._menu)
        self.grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._sized = False  # the columns fitted to their contents once per table
        header = self.grid.horizontalHeader()
        header.sectionPressed.disconnect()  # Qt's own: select the column
        header.sectionClicked.connect(self._sort_by)
        header.setSortIndicatorShown(True)
        self.buttons = {}
        bar = QHBoxLayout()
        for name, text, tip, slot in (
                ("save", "Save", "write the table (its reader checks it first)", self.save),
                ("revert", "Revert", "back to the file as it is", self.revert),
                ("add_row", "Add a row", "a new row at the end", self.add_row),
                ("remove", "Remove the selected rows", "", self.remove_rows),
                ("add_column", "Add a column…", "a column of your own (hxex keeps it)",
                 self.add_column),
                ("remove_column", "Remove the selected columns",
                 "the columns of the selected cells (select a cell in each)",
                 self.remove_columns)):
            button = QPushButton(text)
            button.setToolTip(tip)
            # called without arguments: clicked passes "checked" (False), which add_column
            # would take for a column name
            button.clicked.connect(lambda checked=False, slot=slot: slot())
            bar.addWidget(button)
            self.buttons[name] = button
        bar.addStretch()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.heading)
        layout.addLayout(bar)
        layout.addWidget(self.problem)
        layout.addWidget(self.notice_row)
        layout.addWidget(self.grid, 1)
        self._enable()

    # --- the questions (replaced in tests) -------------------------------------------------

    def ask_save_anyway(self, problem: str) -> bool:
        return QMessageBox.question(
            self, "The table has a problem",
            f"{problem}\n\nSave it anyway (a table you are still filling in), or keep "
            "editing?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel
        ) == QMessageBox.StandardButton.Save

    def ask_overwrite(self) -> bool:
        return QMessageBox.question(
            self, "Changed outside the window",
            f"{self.path.name} was changed by another program since it was opened here. "
            "Write your version over it?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel
        ) == QMessageBox.StandardButton.Save

    def ask_remove_columns(self, names: list[str]) -> bool:
        return QMessageBox.question(
            self, "Remove columns?",
            f"Remove the column(s) {', '.join(repr(n) for n in names)} and everything in them? "
            "(Nothing is written until Save; Revert brings them back.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        ) == QMessageBox.StandardButton.Yes

    def ask_unsaved(self) -> QMessageBox.StandardButton:
        return QMessageBox.question(
            self, "Save the table?", f"{self.path.name} has changes that are not saved.",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel)

    def tell(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    # --- opening ---------------------------------------------------------------------------

    def open(self, kind: TableKind | None, path: Path | None, project: Path | None) -> None:
        """Show the table ``kind`` at ``path`` (None: no table)."""
        if self.watcher.files():
            self.watcher.removePaths(self.watcher.files())
        self.kind, self.path, self.project = kind, path, project
        self.modified = False
        self._sized = False  # another table: its columns fitted to it
        self.problem.hide()
        if kind is None or path is None:
            self.table = None
            self.heading.setText("" if kind is None else f"No {kind.title}.")
            self._fill()
            return
        self.heading.setText(f"{kind.title}: {path}")
        self._results = None
        if kind.setting == "review.table" and project is not None:
            self._results = results_ids(project)
        self.reload()
        self.watcher.addPath(str(path))

    def reload(self) -> None:
        try:
            self.table = read_table(self.path)
        except (OSError, UnicodeDecodeError) as exc:
            self.table = None
            self._show_problem(str(exc))
        self._stamp = _stamp(self.path)
        self.modified = False
        self._fill()

    def revert(self) -> None:
        if self.path is not None:
            self.reload()
            self.problem.hide()

    def _file_changed(self, path: str) -> None:
        if Path(path).exists() and path not in self.watcher.files():
            self.watcher.addPath(path)  # a replaced file is no longer watched
        if not self.modified and _stamp(Path(path)) != self._stamp:
            self.reload()
            log.info("%s changed outside the window: shown as it is now", Path(path).name)

    # --- the grid --------------------------------------------------------------------------

    def _fill(self) -> None:
        self._apply_sort()
        self._match()
        self.model.refresh()
        if self.table is not None and not self._sized:
            self.grid.resizeColumnsToContents()  # the rows in view: quick at any size
            self._sized = True
        self._enable()

    def cell(self, row: int, column: int) -> str:
        """The value shown at ``row`` (in the order shown) and ``column``."""
        return self.model.data(self.model.index(row, column))

    def set_cell(self, row: int, column: int, value: str) -> None:
        """Type ``value`` into the cell at ``row`` (in the order shown) and ``column``."""
        self.model.setData(self.model.index(row, column), value)

    def copied_text(self) -> str:
        """The selected cells as a spreadsheet takes them: the block from the first selected row
        and column to the last, tab-separated, a cell not selected inside it left empty."""
        picked = self.grid.selectionModel().selectedIndexes()
        if not picked:
            current = self.grid.currentIndex()
            picked = [current] if current.isValid() else []
        if not picked:
            return ""
        chosen = {(i.row(), i.column()) for i in picked}
        rows = range(min(r for r, _ in chosen), max(r for r, _ in chosen) + 1)
        columns = range(min(c for _, c in chosen), max(c for _, c in chosen) + 1)
        return "\n".join("\t".join(self.cell(r, c) if (r, c) in chosen else "" for c in columns)
                         for r in rows) + "\n"

    def cell_menu(self):
        """The grid's right-click menu: Copy (the selected cells, as Ctrl+C)."""
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        copy = menu.addAction("Copy")
        copy.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        copy.setEnabled(bool(self.copied_text()))
        copy.triggered.connect(self.copy_selection)
        return menu

    def _menu(self, pos) -> None:
        menu = self.cell_menu()
        menu.exec(self.grid.viewport().mapToGlobal(pos))
        menu.deleteLater()

    def copy_selection(self) -> None:
        from PySide6.QtWidgets import QApplication

        text = self.copied_text()
        if text:
            QApplication.clipboard().setText(text)

    def _selected_rows(self) -> list[int]:
        return sorted({i.row() for i in self.grid.selectionModel().selectedIndexes()})

    def _enable(self) -> None:
        has = self.table is not None
        for name in ("add_row", "remove", "add_column", "remove_column"):
            self.buttons[name].setEnabled(has)
        self.buttons["save"].setEnabled(has and self.modified)
        self.buttons["revert"].setEnabled(has and self.modified)

    def _sort_by(self, column: int) -> None:
        """Sort the view by ``column``; a second click on it reverses the order."""
        ascending = not (self._sort is not None and self._sort == (column, True))
        self._sort = (column, ascending)
        self._fill()

    def _apply_sort(self) -> None:
        """The order the rows are shown in: the file's, or by the sorted column (numbers as
        numbers when every value of the column is one, else text with its numbers by value:
        residue ranges 2-12, 13-30, 108-127; empty cells last either way)."""
        n = len(self.table.rows) if self.table is not None else 0
        self._order = list(range(n))
        header = self.grid.horizontalHeader()
        if self._sort is None or self.table is None or self._sort[0] >= len(self.table.columns):
            header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
            return
        column, ascending = self._sort
        values = [self.table.rows[i][column].strip() for i in range(n)]
        filled = [i for i in range(n) if values[i]]
        try:
            keys = {i: float(values[i]) for i in filled}
        except ValueError:
            keys = {i: natural_key(values[i]) for i in filled}
        filled.sort(key=lambda i: keys[i], reverse=not ascending)
        self._order = filled + [i for i in range(n) if not values[i]]
        header.setSortIndicator(column, Qt.SortOrder.AscendingOrder if ascending
                                else Qt.SortOrder.DescendingOrder)

    def _changed(self) -> None:
        self.modified = True
        before = self.match.unmatched if self.match is not None else {}
        self._match()
        if (self.match.unmatched if self.match is not None else {}) != before:
            self.grid.viewport().update()
            self.grid.verticalHeader().viewport().update()
        self._enable()

    # --- review rows the results do not have -----------------------------------------------

    def _match(self) -> None:
        """Mark the review table's rows naming spectra the results do not have, and say how
        many (nothing for other tables, or without results)."""
        self.match = None
        if self.table is not None and self._results is not None:
            self.match = review_match(self.table, self._results)
        rows = sorted(self.match.unmatched) if self.match is not None else []
        if not rows:
            self.notice_row.hide()
            return
        # numbered as the grid's rows are (a spreadsheet counts the header too: one more)
        numbers = ", ".join(str(r + 1) for r in rows[:5])
        more = f" and {len(rows) - 5} more" if len(rows) > 5 else ""
        self.notice.setText(
            f"{len(rows)} row(s) name spectra the results in {self.match.bundle} do not have, "
            f"so hxex skips them: row {numbers}{more} (shaded; one more in a spreadsheet, "
            "which counts the header). Copy each spectrum_id from spectra.csv, or remove the "
            "rows.")
        self.notice_row.show()

    def row_mark(self, row: int, role, header: bool = False):
        """The shading (or, on the row's number, the colour) and tooltip of the table's row
        ``row`` when the results do not have its spectrum; for a row whose decision follows its
        spectrum to a new id, the tooltip only."""
        if self.match is None:
            return None
        why = self.match.unmatched.get(row)
        if why is None:
            moved = self.match.moved.get(row)
            return moved if role == Qt.ItemDataRole.ToolTipRole else None
        if role == Qt.ItemDataRole.ToolTipRole:
            return why
        if header:
            return QBrush(UNMATCHED_TEXT) if role == Qt.ItemDataRole.ForegroundRole else None
        return QBrush(UNMATCHED) if role == Qt.ItemDataRole.BackgroundRole else None

    def show_next_unmatched(self) -> int | None:
        """Select the next shaded row after the current one (in the order shown, round to the
        first); returns its position in the grid."""
        if self.match is None or not self.match.unmatched:
            return None
        shown = [i for i, r in enumerate(self._order) if r in self.match.unmatched]
        current = self.grid.currentIndex().row() if self.grid.currentIndex().isValid() else -1
        position = next((i for i in shown if i > current), shown[0])
        index = self.model.index(position, 0)
        self.grid.setCurrentIndex(index)
        self.grid.selectRow(position)
        self.grid.scrollTo(index, QAbstractItemView.ScrollHint.PositionAtCenter)
        return position

    def add_row(self) -> None:
        self.table.rows.append([""] * len(self.table.columns))
        self._fill()
        self._changed()
        self.grid.setCurrentIndex(
            self.model.index(self._order.index(len(self.table.rows) - 1), 0))

    def remove_rows(self) -> None:
        rows = sorted({self._order[r] for r in self._selected_rows()}, reverse=True)
        for r in rows:
            del self.table.rows[r]
        if rows:
            self._fill()
            self._changed()

    def remove_columns(self) -> None:
        """Remove the columns of the selected cells, after asking (their contents go too)."""
        columns = sorted({i.column() for i in self.grid.selectionModel().selectedIndexes()})
        if not columns:
            return
        names = [self.table.columns[c] or f"(column {c + 1}, no name)" for c in columns]
        if not self.ask_remove_columns(names):
            return
        if self._sort is not None and self._sort[0] in columns:
            self._sort = None  # its column is gone
        elif self._sort is not None:
            self._sort = (self._sort[0] - sum(c < self._sort[0] for c in columns), self._sort[1])
        for c in reversed(columns):
            del self.table.columns[c]
            for row in self.table.rows:
                del row[c]
        self._fill()
        self._changed()

    def add_column(self, name: str | None = None) -> None:
        if name is None:
            name, ok = QInputDialog.getText(self, "Add a column", "Name of the new column:")
            if not ok:
                return
        name = name.strip()
        if not name or name in self.table.columns:
            self.tell("Column not added", f"a column needs a name not already used ({name!r})")
            return
        self.table.columns.append(name)
        for row in self.table.rows:
            row.append("")
        self._fill()
        self._changed()

    # --- saving ----------------------------------------------------------------------------

    def save(self) -> bool:
        """Write the table (after its reader's checks, and asking when it has a problem or
        was changed outside the window). False when it was not written."""
        if self.table is None or self.path is None:
            return False
        problem = table_problem(self.kind.setting, self.table, self.project) \
            if self.project is not None else None
        if problem is not None:
            self._show_problem(problem)
            if not self.ask_save_anyway(problem):
                return False
        if self._stamp is not None and _stamp(self.path) != self._stamp \
                and not self.ask_overwrite():
            return False
        try:
            watched = self.watcher.files()  # not watched while written (Windows refuses)
            if watched:
                self.watcher.removePaths(watched)
            try:
                write_table(self.table, self.path)
            finally:
                self.watcher.addPath(str(self.path))
        except TableLocked as exc:
            log.warning("%s", exc)
            self.tell("Table not saved", str(exc))
            return False
        self._stamp = _stamp(self.path)
        self.modified = False
        if problem is None:
            self.problem.hide()
        self._enable()
        log.info("saved %s (%d rows)", self.path, len(self.table.rows))
        self.saved.emit(self.path)
        return True

    def may_leave(self) -> bool:
        """Before another table, project or the end: unsaved changes are saved, dropped, or
        the leaving cancelled (False)."""
        if not self.modified:
            return True
        answer = self.ask_unsaved()
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        self.modified = False
        return True

    def _show_problem(self, text: str) -> None:
        self.problem.setText(text)
        self.problem.show()


class TablesView(QWidget):
    """The project's tables (left) and the one chosen (right). :meth:`show_project` lists them
    from the project file's settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Path | None = None
        self.values: dict = {}
        self.fmt = "specexport"
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._chosen)
        self.new_button = QPushButton("New table…")
        self.new_button.setToolTip("a new table with the columns hxex reads, named in the "
                                   "project file")
        self.new_button.clicked.connect(lambda checked=False: self.new_table())
        self.editor = TableEditor()
        left = QWidget()
        side = QVBoxLayout(left)
        side.setContentsMargins(0, 0, 0, 0)
        side.addWidget(self.list, 1)
        side.addWidget(self.new_button)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(self.editor)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 900])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split)
        self._row = -1

    def show_project(self, project: Path | None, values: dict, fmt: str) -> None:
        """List the tables of the project file (its settings ``values``, input format
        ``fmt``); the table shown stays when it is still named."""
        self.project, self.values, self.fmt = project, values, fmt
        keep = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for kind in kinds(fmt):
            path = self.path_of(kind)
            state = "not set" if path is None else (path.name if path.exists()
                                                    else f"{path.name} (no such file)")
            item = QListWidgetItem(f"{kind.title}\n    {state}")
            item.setData(Qt.ItemDataRole.UserRole, kind.setting)
            self.list.addItem(item)
        self.list.blockSignals(False)
        if self.editor.modified and self.editor.kind is not None:
            return  # unsaved edits stay on screen
        self.list.setCurrentRow(keep if 0 <= keep < self.list.count() else 0)
        self._chosen(self.list.currentRow(), force=True)

    def path_of(self, kind: TableKind) -> Path | None:
        value = self.values.get(kind.setting)
        if not value or self.project is None:
            return None
        path = Path(str(value))
        return path if path.is_absolute() else self.project.parent / path

    def kind_at(self, row: int) -> TableKind | None:
        return kinds(self.fmt)[row] if 0 <= row < len(kinds(self.fmt)) else None

    def show_table(self, setting: str) -> None:
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == setting:
                self.list.setCurrentRow(i)
                return
        raise KeyError(setting)

    def _chosen(self, row: int, force: bool = False) -> None:
        if row == self._row and not force:
            return
        if row != self._row and not self.editor.may_leave():
            self.list.blockSignals(True)
            self.list.setCurrentRow(self._row)
            self.list.blockSignals(False)
            return
        self._row = row
        kind = self.kind_at(row)
        path = self.path_of(kind) if kind is not None else None
        exists = path is not None and path.exists()
        self.editor.open(kind, path if exists else None, self.project)
        self.new_button.setEnabled(kind is not None and self.project is not None)

    def new_table(self, path: Path | None = None) -> Path | None:
        """A new table of the kind chosen, with the columns its reader takes, saved where the
        user says and named in the project file."""
        from ..config import ProjectError
        from ..project_edit import set_settings
        from .project_editor import _quoted, relative

        kind = self.kind_at(self.list.currentRow())
        if kind is None or self.project is None or not self.editor.may_leave():
            return None
        if path is None:
            name, _ = QFileDialog.getSaveFileName(
                self, f"New {kind.title}", str(self.project.parent / _suggested(kind)),
                "tables (*.csv)")
            if not name:
                return None
            path = Path(name)
        if path.suffix.lower() != ".csv":
            path = path.with_name(path.name + ".csv")
        try:
            write_table(Table(list(kind.columns)), path)
            set_settings(self.project, [(kind.setting,
                                         _quoted(relative(path, self.project.parent)))],
                         unfinished=True)
        except (OSError, ProjectError) as exc:
            self.editor.tell("Table not made", str(exc))
            return None
        log.info("made %s with the columns %s, named in %s (%s)", path.name,
                 ", ".join(kind.columns), self.project.name, kind.setting)
        return path


def _suggested(kind: TableKind) -> str:
    from ..review_table import SUGGESTED_NAME

    return {"input.peptides": "peptides.csv", "input.retention_times": "retention_times.csv",
            "input.rt_elution_windows": "rt_elution_windows.csv",
            "review.table": SUGGESTED_NAME}.get(kind.setting, "table.csv")
