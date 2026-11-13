# SPDX-License-Identifier: GPL-3.0-or-later
"""The Analysis workspace's "HDX Summary Table" tab: the HDX data summary of Masson et al. 2019
(:mod:`hxex.report.summary`) of the analysis shown, as ``hxex analyse`` writes it to
``summary.csv`` and ``summary.md``: one row per item, one column per sample, and the notes saying
how the values were made below it (also the tooltip of the item they belong to). A row whose
value is the same for every sample shows it once, across the samples' columns (owner,
2026-11-13). Copied like the data tab (Ctrl+C, tab-separated for a spreadsheet) and saved as CSV
or as the Markdown table."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QHeaderView, QLabel

from .data_table import DataPanel, RowsModel
from .tips import tip

log = logging.getLogger(__name__)

EMPTY = "Not computed yet: Compute makes the summary with the figures."


class SummaryModel(RowsModel):
    """The summary's rows; an item's tooltip is its note."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.notes: dict[int, str] = {}  # row -> its note
        self.merged: set[int] = set()  # rows whose value is the same for every sample

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if (role == Qt.ItemDataRole.ToolTipRole and index.isValid() and index.column() == 0
                and index.row() in self.notes):
            return tip(self.notes[index.row()])
        if (role == Qt.ItemDataRole.TextAlignmentRole and index.isValid()
                and index.column() > 0 and index.row() in self.merged):
            return int(Qt.AlignmentFlag.AlignCenter)
        return super().data(index, role)


class SummaryPanel(DataPanel):
    """The summary table of the analysis shown (:meth:`show_summary`), in the order hxex
    writes it: not sortable, its texts wrapped to the panel's width."""

    HINT = ("The HDX data summary (Masson et al. 2019), as hxex analyse writes summary.csv and "
            "summary.md. Ctrl+C copies the rows selected. Your own items: [analysis.summary] in "
            "the project file.")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = SummaryModel(self)
        self.table.setModel(self.model)
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setSortIndicatorShown(False)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.save_button.setToolTip(tip("save the summary as CSV (summary.csv) or as a "
                                        "Markdown table (summary.md), as hxex analyse writes "
                                        "them"))
        self.notes = QLabel()  # how the values were made, numbered as in the item names
        self.notes.setWordWrap(True)
        self.notes.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.layout().insertWidget(self.layout().indexOf(self.table) + 1, self.notes)
        self._columns: dict | None = None
        self._table: dict | None = None  # the rows above the notes
        self._notes: list[str] = []
        self.show_summary(None)

    def show_summary(self, columns: dict | None, note: str = "") -> None:
        """Show the summary's ``columns`` (``item``, then one per sample; None: not computed,
        :data:`EMPTY`), with ``note`` above it (e.g. that it is out of date)."""
        from ..report.summary import note_number, split_notes

        self._columns = columns
        self.table.clearSpans()
        if not columns:
            self._table, self._notes = None, []
            self.model.notes, self.model.merged = {}, set()
            self.show_rows([], [], "summary")
            self.count.setText(EMPTY)
            self.notes.hide()
            self.set_note("")
            return
        table, notes = split_notes(columns)
        self._table, self._notes = table, notes
        names = list(table)
        rows = [{n: table[n][i] for n in names} for i in range(len(table["item"]))]
        numbers = {i: note_number(item) for i, item in enumerate(table["item"])}
        self.model.notes = {i: notes[k - 1] for i, k in numbers.items()
                            if k is not None and 1 <= k <= len(notes)}
        self.show_rows(names, rows, "summary")
        self.count.setText(f"{len(rows)} items, {len(names) - 1} sample(s)")
        self.set_note(note)
        self.notes.setText("\n".join(f"{k}. {text}" for k, text in enumerate(notes, 1)))
        self.notes.setVisible(bool(notes))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for i in range(1, len(names)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        # a value the same for every sample: once, across their columns
        self.model.merged = ({i for i, row in enumerate(rows)
                              if len({row[n] for n in names[1:]}) == 1}
                             if len(names) > 2 else set())
        for i in self.model.merged:
            self.table.setSpan(i, 1, 1, len(names) - 1)

    def set_out_of_date(self, why: str) -> None:
        """Say above the table that it is out of date ('': it is not)."""
        if self._columns:
            self.set_note(f"Out of date: {why}. Recompute to see the new summary." if why
                          else "")

    def save_dialog(self) -> None:
        if not self.model.rows:
            return
        path, chosen = QFileDialog.getSaveFileName(
            self, "Save the HDX summary table", "summary.csv",
            "CSV (*.csv);;Markdown table (*.md)")
        if not path:
            return
        markdown = path.lower().endswith(".md") or (
            "Markdown" in chosen and not path.lower().endswith(".csv"))
        if not Path(path).suffix:
            path += ".md" if markdown else ".csv"
        try:
            self.save(path)
        except OSError as exc:
            self.count.setText(f"Not saved: {exc}")
            log.error("the summary was not saved: %s", exc)

    def save(self, path: str | Path) -> Path:
        """Write the summary as ``summary.csv`` holds it (the notes after a blank row), or as
        the Markdown table for a path ending in ``.md``."""
        from ..report.summary import summary_markdown
        from ..report.tables import write_csv

        path = Path(path)
        columns = {n: np.asarray(v, dtype=object) for n, v in (self._columns or {}).items()}
        if path.suffix.lower() == ".md":
            path.write_text(summary_markdown(columns), encoding="utf-8")
        else:
            path = Path(write_csv(columns, path))
        log.info("saved the summary %s", path)
        self.count.setText(f"{len(self.model.rows)} items; saved as {path.name}")
        return path


__all__ = ["EMPTY", "SummaryPanel"]
