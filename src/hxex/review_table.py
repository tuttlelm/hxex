# SPDX-License-Identifier: GPL-3.0-or-later
"""The review settings of a project (its ``[review]`` table) and the user review table: review
decisions made in a spreadsheet (docs/PLAN.md, "Excluding spectra"; the guide's "Reviewing
results").

:class:`ReviewSettings` holds what a user decided after looking at the results: the user review
table (``[review] table``) and the exclusion rules (``[[review.exclude]]``). ``hxex run`` (its
tables), ``hxex report`` and ``hxex analyse`` all apply them, in memory (:func:`apply_view`).

The user saves a copy of ``spectra.csv`` (one row per spectrum) under a name and in a folder of
their own (the guide suggests ``user_review_table.csv`` beside the project file), edits its
``review_status`` (``rejected`` or ``accepted``), ``review_K`` (the number of populations to
use) and ``note`` columns, and names it in ``[review] table``.
:func:`read_review_table` reads only ``spectrum_id`` and those three columns (every other
column is ignored), and :func:`apply_review_table` lays the decisions over the results' review
layer in memory, as the ``[[review.exclude]]`` rules are: the bundle is not changed, and
editing the table needs only the command run again. A row with both
``review_status`` and ``review_K`` empty changes nothing.

A chosen K is drawn and tabulated with the errors of a bootstrap at that K: :func:`chosen_k`
lists the spectra shown at a K other than the fit's choice and :func:`missing_bootstraps` those
of them without a bootstrap at it, which ``hxex resume`` (and ``hxex run``) add.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SUGGESTED_NAME = "user_review_table.csv"  # the guide's suggestion; any name works
STATUSES = ("rejected", "accepted")
_COLUMNS = ("spectrum_id", "review_status", "review_K", "note")


@dataclass(frozen=True)
class ReviewSettings:
    """The ``[review]`` table of a project file: ``table``, the user review table (a path, or
    None), and ``exclude``, the rules (:class:`hxex.exclusions.Exclusion`) whose spectra are
    treated as rejected."""

    table: str | None = None
    exclude: tuple = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return bool(self.table or self.exclude)


@dataclass(frozen=True)
class ReviewSheet:
    """The decisions of a user review table: one entry per spectrum with ``status``
    (``rejected``, ``accepted`` or empty), ``k`` (0: none) and ``note``; ``path`` is the file."""

    spectrum_id: tuple[str, ...]
    status: tuple[str, ...]
    k: tuple[int, ...]
    note: tuple[str, ...]
    path: str = ""

    def __len__(self) -> int:
        return len(self.spectrum_id)

    def describe(self) -> str:
        """``N decisions: R rejected, A accepted, C with a chosen K`` (for ``hxex check``)."""
        rejected = sum(s == "rejected" for s in self.status)
        accepted = sum(s == "accepted" for s in self.status)
        chosen = sum(k > 0 for k in self.k)
        return (f"{len(self)} spectra: {rejected} rejected, {accepted} accepted, {chosen} with "
                "a chosen K")


def _delimiter(header: str) -> str:
    """The separator of a CSV header line: a comma, or the semicolon or tab some spreadsheet
    programs write."""
    counts = {d: header.count(d) for d in (",", ";", "\t")}
    return max(counts, key=counts.get) if any(counts.values()) else ","


def read_review_table(path: str | Path) -> ReviewSheet:
    """The decisions of the user review table at ``path`` (a CSV with a ``spectrum_id`` column
    and any of ``review_status``, ``review_K`` and ``note``; other columns are ignored). Rows
    with neither a status nor a K are left out. A value that cannot be read raises ValueError
    naming its row (the header is row 1)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")  # a spreadsheet's "CSV UTF-8" has a BOM
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    except OSError as exc:
        raise ValueError(f"the review table {path} cannot be read ({exc})") from None
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"the review table {path} is empty")
    reader = csv.reader(lines, delimiter=_delimiter(lines[0]))
    header = [h.strip() for h in next(reader)]
    where = {name: header.index(name) for name in _COLUMNS if name in header}
    if "spectrum_id" not in where:
        raise ValueError(f"the review table {path} has no spectrum_id column: start from a "
                         "copy of spectra.csv in the tables folder")
    if "review_status" not in where and "review_K" not in where:
        raise ValueError(f"the review table {path} has neither a review_status nor a review_K "
                         "column")
    entries: dict[str, tuple[str, int, str, int]] = {}
    problems: list[str] = []
    for row_number, row in enumerate(reader, start=2):
        def cell(name: str, row=row) -> str:
            j = where.get(name)
            return row[j].strip() if j is not None and j < len(row) else ""

        sid = cell("spectrum_id")
        status, k_text, note = cell("review_status").lower(), cell("review_K"), cell("note")
        if status == "unreviewed":  # spectra.csv's word for no decision
            status = ""
        if not sid or (not status and not k_text):
            continue
        if status not in ("", *STATUSES):
            problems.append(f"row {row_number}: review_status {status!r} (rejected, accepted, "
                            "unreviewed, or empty)")
            continue
        k = 0
        if k_text:
            try:
                value = float(k_text)
            except ValueError:
                value = 0.5
            if value != int(value) or value < 1:
                problems.append(f"row {row_number}: review_K {k_text!r} (a whole number of at "
                                "least 1, or empty)")
                continue
            k = int(value)
        if sid in entries and entries[sid][:2] != (status, k):
            problems.append(f"row {row_number}: {sid} is already in row {entries[sid][3]} with "
                            "other values")
            continue
        entries.setdefault(sid, (status, k, note, row_number))
    if problems:
        raise ValueError(f"the review table {path}: " + "; ".join(problems[:10])
                         + (f"; and {len(problems) - 10} more" if len(problems) > 10 else ""))
    ids = list(entries)
    return ReviewSheet(tuple(ids), tuple(entries[s][0] for s in ids),
                       tuple(entries[s][1] for s in ids), tuple(entries[s][2] for s in ids),
                       str(path))


def apply_review_table(results, sheet: ReviewSheet | None):
    """``results`` (:class:`hxex.report.Results`) with the decisions of ``sheet`` laid over its
    review layer, in memory only (the bundle is not changed): a status replaces the stored
    one, a chosen K replaces the stored one, and a note replaces the stored one when given.
    Spectra the results do not have are skipped with a warning; a chosen K that was not fitted
    for its spectrum raises ValueError."""
    from .io.bundle import ReviewTable

    if sheet is None or not len(sheet):
        return results
    fit = results.fit
    index = {str(s): i for i, s in enumerate(fit.spectrum_id)}
    unknown = [s for s in sheet.spectrum_id if s not in index]
    if unknown:
        log.warning("the review table %s names %d spectra these results do not have (for "
                    "example %s): they are skipped; a table from an earlier run may name "
                    "spectra since left out", sheet.path, len(unknown), ", ".join(unknown[:3]))
    bad = []
    for sid, k in zip(sheet.spectrum_id, sheet.k, strict=True):
        if k and sid in index and (k not in fit.k_values or not fit.fitted(k)[index[sid]]):
            bad.append(f"{sid} (K = {k})")
    if bad:
        raise ValueError(f"the review table {sheet.path} chooses a K that was not fitted for "
                         f"{len(bad)} spectra ({', '.join(bad[:3])}); the run tried K = "
                         f"{', '.join(map(str, fit.k_values))}")
    review = results.review if results.review is not None else ReviewTable.empty()
    stored = {str(s): j for j, s in enumerate(review.spectrum_id)}
    status = list(review.status.astype(object))
    k_override = list(review.k_override.astype(np.int64))
    note = list(review.note.astype(object))
    updated = list(review.updated.astype(object))
    stale = list(review.stale)
    source = (list(review.source) if review.source is not None
              else ["review"] * len(review))
    ids = list(map(str, review.spectrum_id))
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    for sid, st, k, nt in zip(sheet.spectrum_id, sheet.status, sheet.k, sheet.note,
                              strict=True):
        if sid not in index:
            continue
        j = stored.get(sid)
        if j is None:
            ids.append(sid)
            status.append(st)
            k_override.append(k)
            note.append(nt)
            updated.append(stamp)
            stale.append(False)
            source.append("review table")
            continue
        if st:
            status[j] = st
        if k:
            k_override[j] = k
        if nt:
            note[j] = nt
        stale[j] = False
        source[j] = "review table"
    merged = ReviewTable(
        spectrum_id=np.array(ids, dtype=object), status=np.array(status, dtype=object),
        k_override=np.array(k_override, dtype=np.int64), note=np.array(note, dtype=object),
        updated=np.array(updated, dtype=object), stale=np.array(stale, dtype=bool),
        source=np.array(source, dtype=object))
    log.info("review table %s: %s", sheet.path, sheet.describe())
    return results.with_review(merged)


def chosen_k(results) -> dict[str, int]:
    """The spectra shown at a K other than the fit's choice (the review's chosen K, from the
    bundle's review layer or a review table), with that K."""
    fit = results.fit
    effective = results.effective_K
    return {str(fit.spectrum_id[i]): int(effective[i])
            for i in np.flatnonzero(effective != fit.selected_K)}


def missing_bootstraps(results) -> dict[str, int]:
    """The spectra of :func:`chosen_k` without a bootstrap at their chosen K, when the results
    have a bootstrap (without any, no spectrum has bootstrap errors and nothing is missing)."""
    if results.bootstrap is None and not results.bootstrap_by_k:
        return {}
    index = {str(s): i for i, s in enumerate(results.fit.spectrum_id)}
    out = {}
    for sid, k in chosen_k(results).items():
        boot = results.bootstrap_for(index[sid], k)
        if boot is None or not boot.covers(index[sid]):
            out[sid] = k
    return out


def apply_view(results, review: ReviewSettings | None):
    """``results`` as the project's ``[review]`` settings present them: the review table's
    decisions, then the ``[[review.exclude]]`` rules (which reject whatever the table says), in
    memory."""
    from .exclusions import exclude_from_results

    if not review:
        return results
    if review.table:
        results = apply_review_table(results, read_review_table(review.table))
    return exclude_from_results(results, review.exclude)


def table_choices(review: ReviewSettings | None) -> dict[str, int]:
    """The K the project's review table chooses, by spectrum (none without a table)."""
    if review is None or not review.table:
        return {}
    sheet = read_review_table(review.table)
    return {s: k for s, k in zip(sheet.spectrum_id, sheet.k, strict=True) if k}
