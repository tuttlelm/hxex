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
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..project_edit import replace_file

log = logging.getLogger(__name__)


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
    extra = []
    if fmt in ("mzml", "thermo"):
        from ..io.elution_windows import COLUMNS

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
    except (ReadError, ProjectError, ValueError) as exc:
        return str(exc).replace(str(copy), "the table")
    finally:
        copy.unlink(missing_ok=True)
    return None


@dataclass(frozen=True)
class ResultsIds:
    """The spectra of a project's results, for checking its review table: the bundle's name,
    its spectrum ids, and whether each came from a kinetics run (whose entries never move)."""

    bundle: str
    ids: tuple[str, ...]
    kinetics: tuple[bool, ...]


def results_ids(project_path: str | Path) -> ResultsIds | None:
    """The spectra of the results the project at ``project_path`` writes to (its ``[output]
    bundle``); None when the project does not load or has no results yet."""
    from ..config import ProjectError, load_project
    from ..io import ReadError, open_bundle
    from ..io.bundle import bundle_manifest
    from ..io.mzml import kinetics_spectra

    try:
        project = load_project(project_path)
    except (ProjectError, OSError, ValueError):
        return None
    bundle = project.output.bundle
    if bundle is None or not bundle.exists() or bundle_manifest(bundle) is None:
        return None
    try:
        with open_bundle(bundle) as b:
            ids = tuple(map(str, b.spectrum_ids()))
            kinetics = tuple(bool(k) for k in kinetics_spectra(b.spectra_source(), ids))
    except (ReadError, OSError, ValueError, KeyError) as exc:
        log.warning("the review table is not checked against %s: %s", bundle, exc)
        return None
    return ResultsIds(bundle.name, ids, kinetics)


@dataclass(frozen=True)
class ReviewMatch:
    """How the rows of a review table match the results (:func:`review_match`): ``unmatched``
    (row index -> why the results have no such spectrum: the row is skipped, as ``hxex check``
    and ``hxex report`` say) and ``moved`` (row index -> where its decision went: the
    spectrum's new id after a rerun summed other scans)."""

    bundle: str
    unmatched: dict[int, str]
    moved: dict[int, str]


def review_match(table: Table, results: ResultsIds) -> ReviewMatch:
    """Which rows of the review ``table`` name a spectrum ``results`` do not have. Only rows the
    reader keeps count (a spectrum_id and a status, a K or a note; ``unreviewed`` is no status);
    a row whose spectrum was summed from other scans in a rerun follows it
    (:func:`hxex.review_table.move_entries`) and is not unmatched."""
    from ..review_table import move_entries

    def column(name):
        return table.columns.index(name) if name in table.columns else None

    sid_at = column("spectrum_id")
    if sid_at is None:
        return ReviewMatch(results.bundle, {}, {})
    status_at, k_at, note_at = column("review_status"), column("review_K"), column("note")

    def text(row, at):
        return row[at].strip() if at is not None else ""

    rows = {}
    for i, row in enumerate(table.rows):
        sid = text(row, sid_at)
        status = text(row, status_at)
        if status.lower() == "unreviewed":  # spectra.csv's word for no decision
            status = ""
        if sid and (status or text(row, k_at) or text(row, note_at)):
            rows[i] = sid
    known = set(results.ids)
    moves = move_entries(list(dict.fromkeys(rows.values())), results.ids, results.kinetics)
    unmatched, moved = {}, {}
    for i, sid in rows.items():
        if sid in known:
            continue
        if sid in moves.moved:
            moved[i] = (f"{results.bundle} summed this ion from other scans: the decision "
                        f"follows it to {moves.moved[sid]}")
        elif sid in moves.superseded:
            moved[i] = (f"{results.bundle} summed this ion from other scans, and its new "
                        f"spectrum {moves.superseded[sid]} has a row of its own, which is used "
                        "instead of this one")
        elif sid in moves.several:
            unmatched[i] = (f"{results.bundle} has no spectrum of this id; its ion has several "
                            f"spectra in that file now ({', '.join(moves.several[sid][:3])}), so "
                            "the decision does not follow any: the row is skipped")
        else:
            unmatched[i] = (f"{results.bundle} has no spectrum of this id: the row is skipped "
                            "(copy the spectrum_id from spectra.csv, or remove the row)")
    return ReviewMatch(results.bundle, unmatched, moved)
