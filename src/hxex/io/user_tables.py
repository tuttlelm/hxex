# SPDX-License-Identifier: GPL-3.0-or-later
"""The tables users write, for hxex's window and Python (docs/PLAN.md, "GUI", "Editing what
users edit"):
the peptide table (the scan formats') or the table of files (the folder formats'), the
retention-time table, and the user review table, each named in the project file.

A table is read whole, edited in memory and written back in one go (:func:`write_table`): the
columns keep their order and the columns hxex does not use stay; the file is written with commas
(the only separator the readers accept) as UTF-8, with a byte-order mark when it had one (a
spreadsheet's "CSV UTF-8"), through a temporary file, so a failed write leaves the old one. A
file another program holds (Excel locks the file it has open on Windows) raises
:class:`TableLocked`, and the edits are kept by the caller. :func:`table_problem` runs the
reader's own checks on the edited table before it is written. Moved here from ``hxex.gui``
(audit of 2026-11-15, Q-4: the window's file-format logic belongs to the library).
"""

from __future__ import annotations

import csv
import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..project_edit import replace_file

log = logging.getLogger(__name__)


class TableLocked(OSError):
    """The table could not be written: another program holds it (a spreadsheet, say)."""


class TableReadOnly(OSError):
    """The table or its folder is read-only: it is not written over (audit of 2026-11-15,
    B-11, B-20: a read-only file was silently replaced on Linux and blamed on another program
    on Windows)."""


@dataclass(frozen=True)
class TableKind:
    """A table a project names: the setting that names it, its name in words, and the columns
    a new one starts with."""

    setting: str
    title: str
    columns: tuple[str, ...]


def kinds(fmt: str) -> list[TableKind]:
    """The tables a project of the input format ``fmt`` can name (none for a format hxex does
    not know: the Settings page says so; a misspelt format stopped the window opening the
    project, audit of 2026-11-15, B-3)."""
    from ..config import FORMATS

    if fmt not in FORMATS:
        return []
    if fmt in ("mzml", "thermo"):
        peptides = TableKind("input.peptides", "peptide table",
                             ("sample", "peptide", "start_seq", "end_seq", "charge",
                              "modification", "rt_start", "rt_end"))
    else:
        peptides = TableKind("input.peptides", "table of files",
                             ("file", "sample", "peptide", "start_seq", "end_seq", "charge"))
    extra = []
    if fmt in ("mzml", "thermo"):
        from .elution_windows import COLUMNS

        extra = [TableKind("input.rt_elution_windows", "elution-window table", COLUMNS)]
    return [peptides, *extra,
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
    encoding: str = "utf-8"  # the file's; written back as UTF-8

    def copy(self) -> Table:
        return Table(list(self.columns), [list(r) for r in self.rows], self.bom, self.newline,
                     self.separator, self.encoding)

    @property
    def converted(self) -> str:
        """What writing the table back changes about its file, in words (""): its encoding
        and its separator (audit of 2026-11-15, D-12)."""
        out = []
        if self.encoding != "utf-8":
            out.append(f"read as {self.encoding}, written back as UTF-8"
                       + (" with a byte-order mark" if self.bom else ""))
        if self.separator != ",":
            out.append(f"read with {'semicolons' if self.separator == ';' else 'tabs'}, "
                       "written back with commas")
        return "; ".join(out)


def read_table(path: str | Path) -> Table:
    """Read the CSV at ``path`` (UTF-8 with or without a byte-order mark, or UTF-16 as Excel's
    "Unicode text"; commas, or semicolons or tabs, which are written back as commas)."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except PermissionError as exc:
        raise TableLocked(f"{path} cannot be read: another program holds it (close it in "
                          "the spreadsheet, then try again)") from exc
    from ..review_table import delimiter

    encoding = "utf-8"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text, bom, encoding = raw.decode("utf-16"), True, "UTF-16"
    else:
        bom = raw.startswith(b"\xef\xbb\xbf")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            # a spreadsheet's plain "CSV" (Windows-1252): read, said, written back as UTF-8
            # (audit of 2026-11-15, B-18)
            text, encoding = raw.decode("cp1252"), "Windows-1252"
    newline = "\r\n" if "\r\n" in text else "\n"
    first = text.splitlines()[0] if text else ""
    separator = delimiter(first)  # as the readers decide it (audit of 2026-11-15, D-14)
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=separator))
    # a blank line stays a row (empty), so the grid's row numbers are the readers' (B-21)
    while rows and not any(cell.strip() for cell in rows[-1]):
        rows.pop()
    if not rows or not any(cell.strip() for cell in rows[0]):
        return Table([], [], bom, newline, separator, encoding)
    columns = [c.strip() for c in rows[0]]
    # a row longer than the header keeps its extra cells, under columns without a name
    width = max(len(r) for r in rows)
    columns += [""] * (width - len(columns))
    body = [r + [""] * (width - len(r)) for r in rows[1:]]
    return Table(columns, body, bom, newline, separator, encoding)


def table_text(table: Table) -> str:
    """The table as CSV with commas, in the file's line endings."""
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator=table.newline)
    writer.writerow(table.columns)
    for row in table.rows:  # a blank line read stays a blank line
        writer.writerow(row if any(cell.strip() for cell in row) else [])
    return out.getvalue()


def write_table(table: Table, path: str | Path) -> Path:
    """Write ``table`` to ``path`` through a temporary file beside it; a file another program
    holds raises :class:`TableLocked` (the old file is left as it was)."""
    path = Path(path)
    if path.exists() and not os.access(path, os.W_OK):
        raise TableReadOnly(f"{path} is read-only: make it writable (its properties), then save "
                            "again; your changes are kept")
    if not os.access(path.parent if path.parent.exists() else Path.cwd(), os.W_OK):
        raise TableReadOnly(f"the folder {path.parent} is read-only: nothing can be written "
                            "there; your changes are kept")
    data = ("﻿" if table.bom else "") + table_text(table)
    tmp = path.with_name(path.name + ".hxex-tmp")
    try:
        tmp.write_text(data, encoding="utf-8", newline="")
        replace_file(tmp, path)  # tried again while Windows briefly refuses it
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
    from .errors import ReadError

    project_path = Path(project_path)
    # beside the project file so that paths resolve as the project's do; in the system's
    # temporary folder when that folder is read-only (audit of 2026-11-15, B-11)
    folder = project_path.parent if os.access(project_path.parent, os.W_OK) else None
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=folder, delete=False,
                                     encoding="utf-8", newline="") as fh:
        fh.write(table_text(table))
        copy = Path(fh.name)
    try:
        if setting == "review.table":
            from ..review_table import read_review_table

            read_review_table(copy)
        elif setting == "input.rt_elution_windows":
            from ..config import elution_window_problems, load_project

            project = load_project(project_path)
            problems = elution_window_problems(dataclasses.replace(
                project, input=dataclasses.replace(project.input, rt_elution_windows=copy)))
            if problems:
                return "; ".join(problems).replace(str(copy), "the table")
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
    except (ReadError, ProjectError, ValueError, OSError) as exc:
        return str(exc).replace(str(copy), "the table")
    finally:
        copy.unlink(missing_ok=True)
    return None
