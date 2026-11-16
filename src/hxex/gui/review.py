# SPDX-License-Identifier: GPL-3.0-or-later
"""Review decisions written to the user review table, for hxex's window (docs/PLAN.md, "GUI",
"Results and review"). This module does not use Qt.

The rules of the table are the library's (:mod:`hxex.review_table`: :func:`write_decisions`,
:func:`make_review_table`, :class:`Decision`; moved there from here, audit of 2026-11-15,
Q-4 and Q-6), so the window, the commands and Python write and read the same table. This module
keeps the names the window uses and :func:`table_stamp`, which tells the window when the table
changed on disk.
"""

from __future__ import annotations

from pathlib import Path

from ..review_table import DECISION_COLUMNS as COLUMNS
from ..review_table import DECISION_STATUSES as STATUSES
from ..review_table import Decision, make_review_table, write_decisions
from ..review_table import review_table_path as table_path

__all__ = ["COLUMNS", "STATUSES", "Decision", "make_review_table", "table_path",
           "table_stamp", "write_decisions"]


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
