# SPDX-License-Identifier: GPL-3.0-or-later
"""Exporting from the Results view (**Export data**; docs/PLAN.md Next steps 32): a series as
an HX-Express spreadsheet, or the data behind its figures as CSV (a series, a time point, or
one spectrum).

The dialog asks for the folder, an optional prefix and suffix for the names, and the export's
options, shows the names it will write, and asks before replacing anything."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..report.export import (
    PLOT_TABLES,
    ExportError,
    export_hxexpress,
    hxexpress_name,
    plot_data_files,
    plot_data_name,
    write_plot_data,
)
from .tips import tip

log = logging.getLogger(__name__)

HXEXPRESS, PLOT_DATA = "hxexpress", "plot data"
SHOWN_NAMES = 4  # names listed in the dialog before "and N more"


@dataclass
class Target:
    """One thing to export: a series (``spectrum_ids`` None), or some of its spectra, with
    ``part`` naming them in the plot-data file names (``30 s``, ``30 s rep 2``)."""

    data_id: int
    spectrum_ids: list[str] | None = None
    part: str = ""


class ExportDialog(QDialog):
    """Export ``targets`` of ``results`` as ``kind`` (:data:`HXEXPRESS` or :data:`PLOT_DATA`)
    into a folder (``folder``: the one suggested)."""

    def __init__(self, results, kind: str, targets: list[Target], folder: Path, parent=None):
        super().__init__(parent)
        if kind not in (HXEXPRESS, PLOT_DATA):
            raise ValueError(f"kind must be {HXEXPRESS!r} or {PLOT_DATA!r}")
        self.results, self.kind, self.targets = results, kind, list(targets)
        self.written: list[Path] = []
        self.setWindowTitle("Export for HX-Express" if kind == HXEXPRESS else "Export plot data")
        form = QFormLayout()
        self.folder = QLineEdit(str(folder))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("Folder", holder)
        self.prefix, self.suffix = QLineEdit(), QLineEdit()
        where = ("before the sample in the file name (a reader takes it as part of the "
                 "sample's name)" if kind == HXEXPRESS else "at the start of the file names")
        rule = "; no '-' (it separates the name's fields)" if kind == HXEXPRESS else ""
        self.prefix.setToolTip(tip(f"optional; {where}{rule}"))
        self.suffix.setToolTip(tip("optional; " + (
            "at the end of the file name" if kind == HXEXPRESS
            else "after the series' name, before _bins.csv and the others") + rule))
        form.addRow("Prefix", self.prefix)
        form.addRow("Suffix", self.suffix)
        if kind == HXEXPRESS:
            self.spectra = QComboBox()
            self.spectra.addItem("raw spectra", "raw")
            self.spectra.addItem("picked sticks", "picked")
            self.spectra.setToolTip(tip("raw: the spectra as read, for HX-Express to pick; "
                                        "picked: each deuterium bin's m/z and value, as hxex "
                                        "fitted them"))
            self.controls = QComboBox()
            self.controls.addItem("every replicate", "all")
            self.controls.addItem("the first of each", "first")
            self.controls.setToolTip(tip("the undeuterated and fully deuterated controls"))
            form.addRow("Spectra", self.spectra)
            form.addRow("Controls", self.controls)
        else:
            self.ks = QComboBox()
            self.ks.addItem("the K in effect", "selected")
            self.ks.addItem("every fitted K", "all")
            self.ks.setToolTip(tip("the fits written to bins.csv: each spectrum's K (the "
                                   "review's choice, else the selected K), or every K fitted"))
            form.addRow("Fits", self.ks)
        self.names = QLabel()
        self.names.setWordWrap(True)
        self.names.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.export_button = buttons.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self._export)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.names)
        layout.addWidget(buttons)
        for edit in (self.prefix, self.suffix, self.folder):
            edit.textChanged.connect(self._show_names)
        self._show_names()

    # --- names -----------------------------------------------------------------------------

    def name_of(self, target: Target) -> str:
        """The file name (HX-Express) or the start of the file names (plot data) of
        ``target``; raises
        :class:`~hxex.report.export.ExportError` for a prefix or suffix a name cannot hold."""
        prefix, suffix = self.prefix.text(), self.suffix.text()
        if self.kind == HXEXPRESS:
            return hxexpress_name(self.results, target.data_id, prefix=prefix, suffix=suffix)
        return plot_data_name(self.results, target.data_id, part=target.part, prefix=prefix,
                              suffix=suffix)

    def folder_path(self) -> Path:
        return Path(self.folder.text().strip()).expanduser()

    def files(self) -> list[Path]:
        """Every file the export may write."""
        if self.kind == HXEXPRESS:
            return [self.folder_path() / self.name_of(t) for t in self.targets]
        return [f for t in self.targets for f in plot_data_files(self.folder_path(),
                                                                  self.name_of(t))]

    def existing(self) -> list[Path]:
        """The files that would be replaced."""
        return [p for p in self.files() if p.exists()]

    def _show_names(self) -> None:
        try:
            names = [self.name_of(t) for t in self.targets]
        except ExportError as exc:
            self.names.setText(str(exc))
            self.export_button.setEnabled(False)
            return
        self.export_button.setEnabled(bool(self.folder.text().strip()))
        shown = names[:SHOWN_NAMES]
        more = len(names) - len(shown)
        listed = ", ".join(shown) + (f", and {more} more" if more else "")
        if self.kind == HXEXPRESS:
            text = f"Writes {len(names)} file{'s' * (len(names) != 1)}: {listed}"
        else:
            ends = ", ".join(f"_{t}.csv" for t in PLOT_TABLES)
            text = (f"Writes {len(PLOT_TABLES)} CSV files named {listed} followed by {ends}"
                    if len(names) == 1 else
                    f"Writes {len(PLOT_TABLES)} CSV files for each of {len(names)} names: "
                    f"{listed}, each followed by {ends}")
        self.names.setText(text)

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Export into", self.folder.text())
        if chosen:
            self.folder.setText(chosen)

    # --- exporting -------------------------------------------------------------------------

    def ask_replace(self, there: list[Path]) -> bool:
        """Whether to replace what is ``there`` (a question to the user)."""
        listed = "\n".join(p.name for p in there[:SHOWN_NAMES])
        if len(there) > SHOWN_NAMES:
            listed += f"\n… and {len(there) - SHOWN_NAMES} more"
        answer = QMessageBox.question(
            self, "Replace?", f"{len(there)} file{'s' * (len(there) != 1)} already "
            f"exist{'s' * (len(there) == 1)} in {there[0].parent}:\n{listed}\n\n"
            "Replace them?")
        return answer == QMessageBox.StandardButton.Yes

    def export(self) -> list[str]:
        """Write every target (after asking before replacing anything); return the problems,
        one line per target that could not be written (empty when all were)."""
        try:
            there = self.existing()
        except ExportError as exc:
            return [str(exc)]
        if there and not self.ask_replace(there):
            return ["not exported: nothing was replaced"]
        folder = self.folder_path()
        problems = []
        for target in self.targets:
            name = self.name_of(target)
            try:
                if self.kind == HXEXPRESS:
                    written = export_hxexpress(
                        self.results, target.data_id, folder, spectra=self.spectra.currentData(),
                        controls=self.controls.currentData(), prefix=self.prefix.text(),
                        suffix=self.suffix.text(), overwrite=True)
                    self.written.append(written)
                else:
                    which = ({"data_id": target.data_id} if target.spectrum_ids is None
                             else {"spectrum_ids": target.spectrum_ids})
                    self.written += write_plot_data(self.results, folder, name=name,
                                                    ks=self.ks.currentData(), overwrite=True,
                                                    **which)
            except (ExportError, OSError, ValueError) as exc:
                problems.append(f"{name}: {exc}")
                log.warning("not exported: %s: %s", name, exc)
        if self.written:
            log.info("exported %d %s into %s", len(self.written),
                     "HX-Express file(s)" if self.kind == HXEXPRESS else "plot-data file(s)",
                     folder)
        return problems

    def _export(self) -> None:
        problems = self.export()
        if problems and problems != ["not exported: nothing was replaced"]:
            QMessageBox.warning(self, "Not everything was exported", "\n".join(problems))
        if self.written:
            self.accept()
