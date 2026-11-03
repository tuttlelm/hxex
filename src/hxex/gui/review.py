# SPDX-License-Identifier: GPL-3.0-or-later
"""Review decisions written to the user review table, for hxex's window (docs/PLAN.md, "GUI",
"Results and review"). This module does not use Qt.

The table is the project's ``[review] table``, the same file a spreadsheet user edits
(:mod:`hxex.review_table` reads it for every command). The window writes a decision as a user
would type it: ``review_status`` (``accepted``, ``rejected``, or empty), ``review_K`` (a K, or
empty for the fit's choice; ``0`` clears a K the bundle itself chose), and ``note``, in the row of
the spectrum (a row is added for a spectrum the table does not list). Other columns, and columns
of the user's own, stay as they are. When the project names no table, :func:`make_review_table`
writes one from the results' spectra (the guide's suggested name, beside the project file) and
names it in the project file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..review_table import SUGGESTED_NAME
from .user_tables import Table, read_table, write_table

COLUMNS = ("review_status", "review_K", "note")
STATUSES = ("", "accepted", "rejected")


@dataclass(frozen=True)
class Decision:
    """What the window writes for a spectrum: a status (``''``: none), a K (None: the fit's
    choice), and a note (None: left as it is)."""

    status: str | None = None
    k: int | None = None
    k_given: bool = False  # whether ``k`` is part of the decision (None then means the fit's)
    note: str | None = None


def table_path(project: Path) -> Path | None:
    """The project's ``[review] table`` (None when it names none)."""
    from ..project_edit import live_settings

    value = live_settings(project.read_text(encoding="utf-8-sig"))[0].get("review.table")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else project.parent / path


def make_review_table(project: Path, results) -> Path:
    """A user review table made from the results' spectra (a copy of ``spectra.csv``) beside the
    project file, named in its ``[review] table``. A table already named is returned as it is."""
    from ..project_edit import set_settings
    from ..report.tables import spectra_table
    from .project_editor import _quoted, relative

    named = table_path(project)
    if named is not None and named.exists():
        return named
    path = named or project.parent / SUGGESTED_NAME
    if not path.exists():
        cols = spectra_table(results)
        names = list(cols)
        rows = [[_text(cols[n][i]) for n in names] for i in range(len(cols["spectrum_id"]))]
        write_table(Table(names, rows), path)
    if named is None:
        set_settings(project, [("review.table", _quoted(relative(path, project.parent)))],
                     unfinished=True)
    return path


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if value != value else repr(value)
    return str(value)


def write_decisions(path: Path, decisions: dict[str, Decision], bundle_k: dict[str, int]
                    | None = None) -> None:
    """Write each spectrum's decision into its row of the table at ``path`` (a row added where
    the table has none); ``bundle_k``: the K chosen for a spectrum in the results shown; a
    decision for the fit's choice empties the table's own ``review_K``, or, when the table chose
    none (the bundle's review layer did), clears it with ``review_K = 0``. Raises
    :class:`hxex.gui.user_tables.TableLocked` when another program holds the file."""
    table = read_table(path)
    for name in ("spectrum_id", *COLUMNS):
        if name not in table.columns:
            table.columns.append(name)
            for row in table.rows:
                row.append("")
    col = {name: table.columns.index(name) for name in ("spectrum_id", *COLUMNS)}
    rows = {row[col["spectrum_id"]]: row for row in table.rows}
    for sid, d in decisions.items():
        row = rows.get(sid)
        if row is None:
            row = [""] * len(table.columns)
            row[col["spectrum_id"]] = sid
            table.rows.append(row)
            rows[sid] = row
        if d.status is not None:
            if d.status not in STATUSES:
                raise ValueError(f"review status must be accepted, rejected or empty, not "
                                 f"{d.status!r}")
            row[col["review_status"]] = d.status
        if d.k_given:
            if d.k is None:  # the fit's choice: the table's K emptied; a K the bundle chose
                # (none in this table) cleared with 0
                had = row[col["review_K"]].strip() not in ("", "0")
                row[col["review_K"]] = "0" if (bundle_k or {}).get(sid) and not had else ""
            else:
                row[col["review_K"]] = str(int(d.k))
        if d.note is not None:
            row[col["note"]] = d.note
    write_table(table, path)
