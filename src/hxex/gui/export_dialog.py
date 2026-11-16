# SPDX-License-Identifier: GPL-3.0-or-later
"""Exporting from the Results view (**Export data**; docs/PLAN.md Next steps 32): a series as
an HX-Express spreadsheet, or the data behind its figures as CSV (a series, a time point, or
one spectrum).

The dialog asks for the folder (a relative one counted from the results' folder), an optional
prefix and suffix for the names, and the export's options, shows the names it will write
(refusing two targets of one name), and asks before replacing anything; it never deletes a
file."""

from __future__ import annotations

import logging
import os
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
    ExportError,
    export_hxexpress,
    hxexpress_name,
    hxexpress_warning,
    plot_data_files,
    plot_data_leftovers,
    plot_data_name,
    plot_data_tables,
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
    into a folder (``folder``: the one suggested). A relative folder typed in the dialog is
    counted from ``base``, the results' folder (as the suggested one is; the bundle's folder
    when not given, else the folder ``hxex gui`` was started in)."""

    def __init__(self, results, kind: str, targets: list[Target], folder: Path, parent=None, *,
                 base: Path | None = None):
        super().__init__(parent)
        if kind not in (HXEXPRESS, PLOT_DATA):
            raise ValueError(f"kind must be {HXEXPRESS!r} or {PLOT_DATA!r}")
        self.results, self.kind, self.targets = results, kind, list(targets)
        self.base = Path(base) if base is not None else _results_folder(results)
        self.written: list[Path] = []
        self._tables: dict[int, tuple[str, ...]] = {}  # target position -> its plot-data tables
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
        self.where = QLabel()  # the full location of a relative folder
        self.where.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("", self.where)
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
        """The folder written into: as typed when it is a full location, else counted from the
        results' folder (``~`` the home folder)."""
        path = Path(self.folder.text().strip()).expanduser()
        if not path.is_absolute():
            path = self.base / path
        return Path(os.path.normpath(path))

    def tables_of(self, n: int) -> tuple[str, ...]:
        """The plot-data tables the export writes for target ``n`` (worked out once)."""
        if n not in self._tables:
            t = self.targets[n]
            self._tables[n] = plot_data_tables(
                self.results, **({"data_id": t.data_id} if t.spectrum_ids is None
                                 else {"spectrum_ids": t.spectrum_ids}))
        return self._tables[n]

    def files(self) -> list[Path]:
        """Every file the export writes."""
        if self.kind == HXEXPRESS:
            return [self.folder_path() / self.name_of(t) for t in self.targets]
        return [f for n, t in enumerate(self.targets)
                for f in plot_data_files(self.folder_path(), self.name_of(t), self.tables_of(n))]

    def existing(self) -> list[Path]:
        """The files that would be replaced; raises
        :class:`~hxex.report.export.ExportError` for a name the file system cannot take."""
        try:
            return [p for p in self.files() if p.exists()]
        except OSError as exc:
            raise ExportError(f"cannot use the folder or a name: {exc.strerror or exc}"
                              ) from exc

    def left_over(self) -> list[Path]:
        """Files of these names an earlier plot-data export wrote and this one does not: left
        as they are (nothing of the user's is deleted)."""
        if self.kind == HXEXPRESS:
            return []
        try:
            return [p for n, t in enumerate(self.targets)
                    for p in plot_data_leftovers(self.folder_path(), self.name_of(t),
                                                 self.tables_of(n))]
        except (ExportError, ValueError, OSError):
            return []

    def problems(self) -> list[str]:
        """Why the export cannot be made as it stands (empty when it can): a prefix, suffix,
        or name the files cannot take, or two targets of one name."""
        try:
            names = [self.name_of(t) for t in self.targets]
            if self.kind == PLOT_DATA:
                for n in range(len(self.targets)):
                    self.tables_of(n)
        except (ExportError, ValueError, OSError) as exc:
            return [str(exc)]
        seen: dict[str, int] = {}
        out = []
        for name in names:
            key = name.lower()  # Windows and macOS do not tell WT from wt
            seen[key] = seen.get(key, 0) + 1
            if seen[key] == 2:
                out.append(f"two of the things to export would be written as {name}: export "
                           "them one at a time")
        return out

    def warnings(self) -> list[str]:
        """What the user should know about the files (written all the same): a series hxex's
        HX-Express reader cannot name from its file name alone."""
        if self.kind != HXEXPRESS:
            return []
        out = []
        for t in self.targets:
            try:
                warning = hxexpress_warning(self.results, t.data_id)
            except (ExportError, ValueError):
                continue
            if warning and warning not in out:
                out.append(warning)
        return out

    def _show_names(self) -> None:
        typed = self.folder.text().strip()
        relative = bool(typed) and not Path(typed).expanduser().is_absolute()
        self.where.setText(f"→ {self.folder_path()}" if relative else "")
        self.where.setVisible(relative)
        problems = self.problems()
        if problems:
            self.names.setText("\n".join(problems))
            self.export_button.setEnabled(False)
            return
        names = [self.name_of(t) for t in self.targets]
        self.export_button.setEnabled(bool(typed))
        shown = names[:SHOWN_NAMES]
        more = len(names) - len(shown)
        listed = ", ".join(shown) + (f", and {more} more" if more else "")
        if self.kind == HXEXPRESS:
            text = f"Writes {len(names)} file{'s' * (len(names) != 1)}: {listed}"
        else:
            sets = {self.tables_of(n) for n in range(len(self.targets))}
            if len(sets) == 1:
                tables = next(iter(sets))
                ends = ", ".join(f"_{t}.csv" for t in tables)
                text = (f"Writes {len(tables)} CSV files named {listed} followed by {ends}"
                        if len(names) == 1 else
                        f"Writes {len(tables)} CSV files for each of {len(names)} names: "
                        f"{listed}, each followed by {ends}")
            else:
                count = sum(len(self.tables_of(n)) for n in range(len(self.targets)))
                text = (f"Writes {count} CSV files for {len(names)} names: {listed}, each "
                        "followed by _bins.csv and the others")
        warnings = self.warnings()
        if warnings:
            text += "\n\nWarning: " + "\nWarning: ".join(warnings)
        self.names.setText(text)

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Export into", self.folder.text())
        if chosen:
            self.folder.setText(chosen)

    # --- exporting -------------------------------------------------------------------------

    def replace_text(self, there: list[Path], left: list[Path] = ()) -> str:
        """The Replace question: the files of this export that are there, and the files of an
        earlier export of the same names this one does not write (left as they are)."""
        def listing(paths):
            text = "\n".join(p.name for p in paths[:SHOWN_NAMES])
            if len(paths) > SHOWN_NAMES:
                text += f"\n… and {len(paths) - SHOWN_NAMES} more"
            return text

        text = (f"{len(there)} of the files this export writes already "
                f"exist{'s' * (len(there) == 1)} in {there[0].parent}:\n{listing(there)}\n\n"
                f"Replace {'it' if len(there) == 1 else 'them'}?")
        if left:
            text += (f"\n\nLeft as {'it is' if len(left) == 1 else 'they are'} from an earlier "
                     f"export (this one does not write {'it' if len(left) == 1 else 'them'}, "
                     f"and nothing is deleted):\n{listing(list(left))}")
        return text

    def ask_replace(self, there: list[Path], left: list[Path] = ()) -> bool:
        """Whether to replace what is ``there`` (a question to the user; ``left``: the files
        an earlier export left that this one does not write, named in it)."""
        answer = QMessageBox.question(self, "Replace?", self.replace_text(there, left))
        return answer == QMessageBox.StandardButton.Yes

    def export(self) -> list[str]:
        """Write every target (after asking before replacing anything); return the problems,
        one line per target that could not be written (empty when all were)."""
        problems = self.problems()
        if problems:
            return problems
        try:
            there = self.existing()
        except (ExportError, ValueError, OSError) as exc:
            return [str(exc)]
        left = self.left_over()
        if there and not self.ask_replace(there, left):
            return ["not exported: nothing was replaced"]
        folder = self.folder_path()
        problems = []
        for target in self.targets:
            try:
                name = self.name_of(target)
            except (ExportError, ValueError, OSError) as exc:
                problems.append(str(exc))
                log.warning("not exported: %s", exc)
                continue
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


def _results_folder(results) -> Path:
    """The folder the results' bundle is in (the suggested folder's base), else the folder
    ``hxex gui`` was started in."""
    bundle = getattr(getattr(results, "loader", None), "__self__", None)
    path = getattr(bundle, "path", None)
    return Path(path).parent if path is not None else Path.cwd()
