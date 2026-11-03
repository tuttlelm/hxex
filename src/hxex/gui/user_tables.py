# SPDX-License-Identifier: GPL-3.0-or-later
"""The tables users write, for hxex's window (docs/PLAN.md, "GUI", "Editing what users edit"):
the peptide table (the scan formats') or the table of files (the folder formats'), the
retention-time table, and the user review table, each named in the project file.

A table is read whole, edited in memory and written back in one go (:func:`write_table`): the
columns keep their order and the columns hxex does not use stay; the file is written with commas
(the only separator the readers accept) as UTF-8, with a byte-order mark when it had one (a
spreadsheet's "CSV UTF-8"), through a temporary file, so a failed write leaves the old one. A
file another program holds (Excel locks the file it has open on Windows) raises
:class:`TableLocked`, and the edits are kept by the caller. :func:`table_problem` runs the
reader's own checks on the edited table before it is written. This module does not use Qt.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass, field
from pathlib import Path


class TableLocked(OSError):
    """The table could not be written: another program holds it (a spreadsheet, say)."""


@dataclass(frozen=True)
class TableKind:
    """A table a project names: the setting that names it, its name in words, and the columns
    a new one starts with."""

    setting: str
    title: str
    columns: tuple[str, ...]


def kinds(fmt: str) -> list[TableKind]:
    """The tables a project of the input format ``fmt`` can name."""
    from ..config import FORMATS

    if fmt not in FORMATS:
        raise ValueError(f"unknown input format {fmt!r}")
    if fmt in ("mzml", "thermo"):
        peptides = TableKind("input.peptides", "peptide table",
                             ("sample", "peptide", "start_seq", "end_seq", "charge",
                              "modification", "rt_start", "rt_end"))
    else:
        peptides = TableKind("input.peptides", "table of files",
                             ("file", "sample", "peptide", "start_seq", "end_seq", "charge"))
    return [peptides,
            TableKind("input.retention_times", "retention-time table",
                      ("sample", "start_seq", "end_seq", "charge", "rt")),
            TableKind("review.table", "user review table",
                      ("spectrum_id", "review_status", "review_K", "note"))]


@dataclass
class Table:
    """A table in memory: its column names and rows (text, one list per row, as long as the
    header), and how its file was written (kept when it is written back)."""

    columns: list[str]
    rows: list[list[str]] = field(default_factory=list)
    bom: bool = False
    newline: str = "\r\n"  # the csv module's default, and Excel's
    separator: str = ","  # the file's; written back with commas

    def copy(self) -> Table:
        return Table(list(self.columns), [list(r) for r in self.rows], self.bom, self.newline,
                     self.separator)


def read_table(path: str | Path) -> Table:
    """Read the CSV at ``path`` (UTF-8 with or without a byte-order mark, or UTF-16 as Excel's
    "Unicode text"; commas, or semicolons or tabs, which are written back as commas)."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except PermissionError as exc:
        raise TableLocked(f"{path} cannot be read: another program holds it (close it in "
                          "the spreadsheet, then try again)") from exc
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text, bom = raw.decode("utf-16"), True
    else:
        bom = raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8-sig")
    newline = "\r\n" if "\r\n" in text else "\n"
    first = text.splitlines()[0] if text else ""
    separator = ","
    if "," not in first:
        separator = next((s for s in (";", "\t") if s in first), ",")
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=separator))
    rows = [r for r in rows if any(cell.strip() for cell in r)] if rows else []
    if not rows:
        return Table([], [], bom, newline, separator)
    columns = [c.strip() for c in rows[0]]
    # a row longer than the header keeps its extra cells, under columns without a name
    width = max(len(r) for r in rows)
    columns += [""] * (width - len(columns))
    body = [r + [""] * (width - len(r)) for r in rows[1:]]
    return Table(columns, body, bom, newline, separator)


def table_text(table: Table) -> str:
    """The table as CSV with commas, in the file's line endings."""
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator=table.newline)
    writer.writerow(table.columns)
    writer.writerows(table.rows)
    return out.getvalue()


def write_table(table: Table, path: str | Path) -> Path:
    """Write ``table`` to ``path`` through a temporary file beside it; a file another program
    holds raises :class:`TableLocked` (the old file is left as it was)."""
    path = Path(path)
    data = ("﻿" if table.bom else "") + table_text(table)
    tmp = path.with_name(path.name + ".hxex-tmp")
    try:
        tmp.write_text(data, encoding="utf-8", newline="")
        os.replace(tmp, path)
    except PermissionError as exc:
        tmp.unlink(missing_ok=True)
        raise TableLocked(f"{path} could not be written: another program holds it (a "
                          "spreadsheet with the file open, say): close it there and save "
                          "again; your changes are kept") from exc
    return path


def table_problem(setting: str, table: Table, project_path: str | Path) -> str | None:
    """What the reader of the table named by ``setting`` would say about ``table`` (None:
    nothing), read from a copy beside the project file so that its paths resolve as the
    project's do. The peptide table of a scan format is also checked against the project's
    runs (a row's sample must be a run's)."""
    import dataclasses
    import tempfile

    from ..config import ProjectError, load_project
    from ..io import ReadError

    project_path = Path(project_path)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=project_path.parent,
                                     delete=False, encoding="utf-8", newline="") as fh:
        fh.write(table_text(table))
        copy = Path(fh.name)
    try:
        if setting == "review.table":
            from ..review_table import read_review_table

            read_review_table(copy)
        elif setting == "input.retention_times":
            from ..io.entries import read_retention_times

            read_retention_times(copy)
        elif setting == "input.peptides":
            from ..io.entries import SCAN_REQUIRED, read_peptide_table

            try:
                project = load_project(project_path)
            except (ProjectError, OSError, ValueError):
                project = None
            fmt = project.input.format if project is not None else None
            if fmt in ("mzml", "thermo"):
                from ..config import scan_peptides

                scan_peptides(dataclasses.replace(
                    project, input=dataclasses.replace(project.input, peptides=copy)))
            elif fmt is not None:  # a folder format's table of files
                read_peptide_table(copy)
            else:  # the project does not load: the columns every peptide table needs
                read_peptide_table(copy, required=SCAN_REQUIRED)
        else:
            raise ValueError(f"{setting} names no table the window edits")
    except (ReadError, ProjectError, ValueError) as exc:
        return str(exc).replace(str(copy), "the table")
    finally:
        copy.unlink(missing_ok=True)
    return None
