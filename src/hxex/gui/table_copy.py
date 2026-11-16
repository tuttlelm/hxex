# SPDX-License-Identifier: GPL-3.0-or-later
"""Copying a grid's cells and saving its rows, the same way in every table of the window (audit
of 2026-11-15, Q-12 and Q-13).

Copied text is tab-separated, one line per row, so that a spreadsheet pastes it into rows and
columns (:func:`block_text`). A grid of rows (the data tab, the list) copies whole rows with the
column names first (:func:`rows_text`); a grid of cells (the Tables view, a list of tables in the
Project editor) copies the block of selected cells, with the column names when every cell is
selected (:func:`selection_text`). With nothing selected, every grid copies the same thing: the
current row, with the column names. Ctrl+C is caught by :func:`install_copy`. Rows are saved as
*hxex*'s own CSV tables are written (:func:`save_rows`), and a failure is said beside the table
(:func:`saved_or_said`), never raised inside a Qt slot.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QAbstractItemView, QFileDialog, QWidget

log = logging.getLogger(__name__)


def copy_text(value) -> str:
    """A value as it is copied or saved: as the CSV files write it (six significant digits, an
    m/z with its 0.0001, booleans as True or False, NaN empty); the cells shown round more
    (audit of 2026-11-15, C-3: an m/z 652.3412 pasted as 652.3)."""
    from ..report.tables import csv_cell

    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    return csv_cell(value)


def block_text(rows: Iterable[Sequence[str]]) -> str:
    """``rows`` of cell texts as tab-separated lines, each ended by a line break ('' for no
    rows)."""
    lines = ["\t".join(str(cell) for cell in row) for row in rows]
    return "\n".join(lines) + "\n" if lines else ""


def rows_text(columns: Sequence[str], rows: Sequence[dict]) -> str:
    """``rows`` (one dict each) under ``columns``, the column names first, each value as the
    CSV files write it (:func:`copy_text`); '' for no rows."""
    if not rows:
        return ""
    return block_text([list(columns)]
                      + [[copy_text(row.get(c)) for c in columns] for row in rows])


def chosen_rows(view: QAbstractItemView) -> list[int]:
    """The rows of a grid of rows to copy: those selected, top to bottom; with none selected,
    the current row; none without one."""
    picked = sorted({i.row() for i in view.selectionModel().selectedIndexes()})
    if picked:
        return picked
    current = view.currentIndex()
    return [current.row()] if current.isValid() else []


def selection_text(view: QAbstractItemView, cell: Callable[[int, int], str],
                   names: Sequence[str]) -> str:
    """A grid of cells' selection as a spreadsheet takes it: the block from the first selected
    row and column to the last, a cell not selected inside it left empty; with every cell
    selected (Ctrl+A), the column ``names`` first (owner, 2026-11-09). With nothing selected,
    the current row with the names; '' without one. ``cell(row, column)``: a cell's text."""
    model = view.model()
    chosen = {(i.row(), i.column()) for i in view.selectionModel().selectedIndexes()}
    if not chosen:
        current = view.currentIndex()
        if not current.isValid():
            return ""
        return block_text([list(names), [cell(current.row(), c) for c in range(len(names))]])
    rows = range(min(r for r, _ in chosen), max(r for r, _ in chosen) + 1)
    columns = range(min(c for _, c in chosen), max(c for _, c in chosen) + 1)
    lines = [[cell(r, c) if (r, c) in chosen else "" for c in columns] for r in rows]
    if len(chosen) == model.rowCount() * model.columnCount():
        lines.insert(0, [names[c] for c in columns])
    return block_text(lines)


def set_clipboard(text: str) -> None:
    """Put ``text`` on the clipboard (nothing when it is empty)."""
    from PySide6.QtWidgets import QApplication

    if text:
        QApplication.clipboard().setText(text)


class _CopyKey(QObject):
    """Ctrl+C (the platform's Copy key) on a grid runs ``action`` instead of Qt's own copy of
    the current cell."""

    def __init__(self, view: QWidget, action: Callable[[], None]):
        super().__init__(view)
        self.action = action

    def eventFilter(self, watched, event):  # noqa: N802 - Qt's name
        if event.type() == QEvent.Type.KeyPress and event.matches(QKeySequence.StandardKey.Copy):
            self.action()
            event.accept()
            return True
        return False


def install_copy(view: QWidget, action: Callable[[], None]) -> None:
    """Make Ctrl+C on ``view`` run ``action`` (which copies what the grid's rule says)."""
    view.installEventFilter(_CopyKey(view, action))


def save_rows(columns: Sequence[str], rows: Sequence[dict], path: str | Path) -> Path:
    """Write ``rows`` (in the order given) under ``columns`` as a CSV file, as *hxex*'s tables
    are written (:func:`hxex.report.tables.write_csv`: NaN empty, numbers to six significant
    digits). An OSError reaches the caller (:func:`saved_or_said` says it)."""
    from ..report.tables import write_csv

    table = {name: np.array([r.get(name, "") for r in rows], dtype=object) for name in columns}
    return Path(write_csv(table, path))


def ask_csv_path(parent: QWidget | None, name: str, title: str = "Save table") -> Path | None:
    """A CSV file's path asked with the save dialog (``name`` suggested; ``.csv`` added when
    the name has no such suffix); None when cancelled."""
    path, _ = QFileDialog.getSaveFileName(parent, title, name, "CSV (*.csv)")
    if not path:
        return None
    return Path(path if path.lower().endswith(".csv") else path + ".csv")


def saved_or_said(write: Callable[[], Path], say: Callable[[str], None],
                  what: str = "the table") -> Path | None:
    """``write()`` the file; a failure (a folder that cannot be written, a file open in
    another program) is said with ``say`` ("Not saved: ...") and logged, and gives None."""
    try:
        return write()
    except OSError as exc:
        say(f"Not saved: {exc}")
        log.error("%s was not saved: %s", what, exc)
        return None


__all__ = ["ask_csv_path", "block_text", "chosen_rows", "copy_text", "install_copy",
           "rows_text", "save_rows", "saved_or_said", "selection_text", "set_clipboard"]
