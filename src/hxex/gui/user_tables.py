# SPDX-License-Identifier: GPL-3.0-or-later
"""The tables users write, for hxex's window: the names the window uses, from the library
(:mod:`hxex.io.user_tables`: reading and writing a table whole, keeping the user's columns;
:mod:`hxex.review_table`: the review table checked against the results). Moved to the library
(audit of 2026-11-15, Q-4), so the window and Python read and write the same tables. This
module does not use Qt.
"""

from __future__ import annotations

from ..io.user_tables import (
    Table,
    TableKind,
    TableLocked,
    TableReadOnly,
    kinds,
    read_table,
    table_problem,
    table_text,
    write_table,
)
from ..review_table import ResultsIds, ReviewMatch, results_ids, review_match

__all__ = ["ResultsIds", "ReviewMatch", "Table", "TableKind", "TableLocked", "TableReadOnly",
           "kinds", "read_table", "results_ids", "review_match", "table_problem", "table_text",
           "write_table"]
