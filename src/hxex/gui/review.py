# SPDX-License-Identifier: GPL-3.0-or-later
"""Review decisions written to the user review table, for hxex's window (docs/PLAN.md, "GUI",
"Results and review"). This module does not use Qt.

The table is the project's ``[review] table``, the same file a spreadsheet user edits
(:mod:`hxex.review_table` reads it for every command). The window writes a decision as a user
would type it: ``review_status`` (``accepted``, ``rejected``, empty, or ``unreviewed``, which
clears a status the bundle itself holds), ``review_K`` (a K, or empty for the fit's choice; ``0``
clears a K the bundle itself chose), and ``note``, in the row of the spectrum (in every row of it,
when the table lists it twice; a row is added for a spectrum the table does not list). Other
columns, and columns of the user's own, stay as they are. When the project names no table,
:func:`make_review_table` writes one from the results' spectra (the guide's suggested name, beside
the project file) and names it in the project file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..review_table import SUGGESTED_NAME
from .user_tables import Table, read_table, write_table

COLUMNS = ("review_status", "review_K", "note")
# "unreviewed": no decision, clearing one the bundle's review layer holds (an empty cell keeps it)
STATUSES = ("", "accepted", "rejected", "unreviewed")


@dataclass(frozen=True)
class Decision:
    """What the window writes for a spectrum: a status (``''``: none), a K (None: the fit's
    choice), and a note (None: left as it is). ``unreviewed`` as the status clears a status
    the bundle's review layer holds, which an empty cell leaves in place."""

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


def table_stamp(project: Path | None) -> tuple | None:
    """When the project's review table last changed, to tell a change of it: (path, modified
    time in nanoseconds, size), with None for a table not made yet; None without a project or
    a table named."""
    if project is None:
        return None
    try:
        path = table_path(project)
    except (OSError, ValueError):
        return None
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return (str(path), None, None)
    return (str(path), stat.st_mtime_ns, stat.st_size)


def make_review_table(project: Path, results, table: Path | None = None) -> Path:
    """A user review table made from the results' spectra (a copy of ``spectra.csv``), named in
    the project file's ``[review] table``: the table named there when it exists (returned as it
    is), else ``table`` (the window asks where, :func:`hxex.gui.table_choice.ask_table`; a
    table that exists is used as it is) or the suggested name beside the project file.

    The spectra an ``[[review.exclude]]`` rule rejects are written without a status or a note
    of the rule's: the rules apply when the table is read, so a rule removed later leaves
    them as the bundle has them (audit of 2026-11-15, D-15)."""
    from ..project_edit import set_settings
    from ..report.tables import _RULE_NOTE, spectra_table
    from .project_editor import _quoted, relative

    named = table_path(project)
    if named is not None and named.exists():
        return named
    path = named or (Path(table) if table is not None else project.parent / SUGGESTED_NAME)
    if not path.exists():
        cols = spectra_table(results)
        by_rule = np.char.startswith(np.asarray(cols["note"]).astype(str), _RULE_NOTE)
        if by_rule.any():  # the bundle's own status and note come back at reading
            cols["review_status"] = np.where(by_rule, "", cols["review_status"])
            cols["note"] = np.where(by_rule, "", cols["note"])
        names = list(cols)
        rows = [[_text(cols[n][i]) for n in names] for i in range(len(cols["spectrum_id"]))]
        write_table(Table(names, rows), path)
    if named is None:
        set_settings(project, [("review.table", _quoted(relative(path, project.parent)))],
                     unfinished=True)
    return path


def _text(value) -> str:
    """A cell as spectra.csv writes it (numpy's repr would write ``np.float64(4.0)``)."""
    from ..report.tables import _cell

    return "" if value is None else _cell(value)


def write_decisions(path: Path, decisions: dict[str, Decision], bundle_k: dict[str, int]
                    | None = None, results=None) -> None:
    """Write each spectrum's decision into its row of the table at ``path`` (a row added where
    the table has none); ``bundle_k``: the K chosen for a spectrum in the results shown; a
    decision for the fit's choice empties the table's own ``review_K``, or, when the table chose
    none (the bundle's review layer did), clears it with ``review_K = 0``. With ``results``
    (those shown), a row naming scans they no longer sum that follows its spectrum
    (:func:`hxex.review_table.move_entries`) is the spectrum's row: a decision for it rewrites
    that row's ``spectrum_id`` to the new one, keeping its other cells. Rows are matched by
    their ``spectrum_id`` without surrounding spaces, as the commands read it, and a spectrum
    the table lists in several rows gets the decision in each (the commands refuse two rows of
    one spectrum that say different things; audit of 2026-11-15, D-1). Raises
    :class:`hxex.gui.user_tables.TableLocked` when another program holds the file."""
    for d in decisions.values():
        if d.status is not None and d.status not in STATUSES:
            raise ValueError(f"review status must be accepted, rejected, unreviewed or empty, "
                             f"not {d.status!r}")
    table = read_table(path)
    for name in ("spectrum_id", *COLUMNS):
        if name not in table.columns:
            table.columns.append(name)
            for row in table.rows:
                row.append("")
    col = {name: table.columns.index(name) for name in ("spectrum_id", *COLUMNS)}
    rows: dict[str, list[list[str]]] = {}
    for row in table.rows:
        rows.setdefault(row[col["spectrum_id"]].strip(), []).append(row)
    rows.pop("", None)
    moved = {}
    if results is not None:
        from ..review_table import results_moves

        moved = {new: old for old, new in results_moves(results, list(rows)).moved.items()}
    for sid, d in decisions.items():
        sid = str(sid).strip()
        mine = rows.get(sid)
        if mine is None and sid in moved:
            mine = rows.pop(moved[sid])
            for row in mine:
                row[col["spectrum_id"]] = sid
            rows[sid] = mine
        if mine is None:
            row = [""] * len(table.columns)
            row[col["spectrum_id"]] = sid
            table.rows.append(row)
            mine = rows[sid] = [row]
        if d.k_given and d.k is None:  # the fit's choice: the table's K emptied; a K the
            # bundle chose (none in this table) cleared with 0
            had = any(r[col["review_K"]].strip() not in ("", "0") for r in mine)
            fit_k = "0" if (bundle_k or {}).get(sid) and not had else ""
        for row in mine:
            if d.status is not None:
                row[col["review_status"]] = d.status
            if d.k_given:
                row[col["review_K"]] = fit_k if d.k is None else str(int(d.k))
            if d.note is not None:
                row[col["note"]] = d.note
    write_table(table, path)
