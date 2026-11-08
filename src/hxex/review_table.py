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
import io
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SUGGESTED_NAME = "user_review_table.csv"  # the guide's suggestion; any name works
CLEARED = -1  # review_K = 0: the K chosen before (in the bundle's review layer) is cleared
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
    (``rejected``, ``accepted`` or empty), ``k`` (0: none; :data:`CLEARED`: ``review_K = 0``,
    which clears a K chosen before) and ``note``; ``path`` is the file."""

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
        cleared = sum(k == CLEARED for k in self.k)
        noted = sum(not s and not k for s, k in zip(self.status, self.k, strict=True))
        return (f"{len(self)} spectra: {rejected} rejected, {accepted} accepted, {chosen} with "
                "a chosen K" + (f", {cleared} with a chosen K cleared" if cleared else "")
                + (f", {noted} with a note only" if noted else ""))


def _delimiter(header: str) -> str:
    """The separator of a CSV header line: a comma, or the semicolon or tab some spreadsheet
    programs write."""
    counts = {d: header.count(d) for d in (",", ";", "\t")}
    return max(counts, key=counts.get) if any(counts.values()) else ","


def read_review_table(path: str | Path) -> ReviewSheet:
    """The decisions of the user review table at ``path`` (a CSV with a ``spectrum_id`` column
    and any of ``review_status``, ``review_K`` and ``note``; other columns are ignored). Rows
    with no status, no K and no note are left out (a note on its own is kept: it changes
    nothing in the fits or the analysis, but it is the reviewer's; owner, 2026-11-05). A value
    that cannot be read raises ValueError naming its row (the header is row 1)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")  # a spreadsheet's "CSV UTF-8" has a BOM
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    except OSError as exc:
        raise ValueError(f"the review table {path} cannot be read ({exc})") from None
    if not text.strip():
        raise ValueError(f"the review table {path} is empty")
    # the separator from the header line; the rows read as a whole, so a quoted note may hold
    # line breaks (row numbers count rows, as a spreadsheet does)
    reader = csv.reader(io.StringIO(text, newline=""),
                        delimiter=_delimiter(text.splitlines()[0]))
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
        if not sid or (not status and not k_text and not note):
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
            if not math.isfinite(value) or value != int(value) or value < 0:
                problems.append(f"row {row_number}: review_K {k_text!r} (a whole number of at "
                                "least 1, 0 to clear a K chosen before, or empty)")
                continue
            k = int(value) if value else CLEARED
        if sid in entries and entries[sid][:2] != (status, k):
            problems.append(f"row {row_number}: {sid} is already in row {entries[sid][3]} with "
                            "other values")
            continue
        entries.setdefault(sid, (status, k, note, row_number))
    if problems:
        raise ValueError(f"the review table {path} (rows numbered as a spreadsheet numbers "
                         "them, the header being row 1): " + "; ".join(problems[:10])
                         + (f"; and {len(problems) - 10} more" if len(problems) > 10 else ""))
    ids = list(entries)
    return ReviewSheet(tuple(ids), tuple(entries[s][0] for s in ids),
                       tuple(entries[s][1] for s in ids), tuple(entries[s][2] for s in ids),
                       str(path))


_SCANS = re.compile(r"/scans?\d+(?:-\d+)?$")


def scan_stem(spectrum_id: str) -> str | None:
    """``spectrum_id`` without the scans it summed (``.../scans1322-1623`` -> ``...``): the
    sample, file, and ion, which a new retention window does not change; None for an id that
    names no scans (SpecExport, pooled, HX-Express: their ids do not depend on a window)."""
    stem = _SCANS.sub("", str(spectrum_id))
    return None if stem == str(spectrum_id) else stem


@dataclass(frozen=True)
class Moves:
    """Review entries matched to the spectra of new results by :func:`move_entries`: ``moved``
    (old id -> the new id taking its decision), ``several`` (old id -> the new ids it would fit,
    left stale), and ``superseded`` (old id -> the new id that has an entry of its own, which
    wins)."""

    moved: dict[str, str] = field(default_factory=dict)
    several: dict[str, tuple[str, ...]] = field(default_factory=dict)
    superseded: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.moved or self.several or self.superseded)

    def describe(self) -> str:
        """``N moved to their spectrum's new scans[, S matching several spectra (left as they
        were)][, D replaced by the new spectrum's own entry]``, for the log and hxex check."""
        out = [f"{len(self.moved)} moved to their spectrum's new scans"]
        if self.several:
            out.append(f"{len(self.several)} matching several spectra (left as they were)")
        if self.superseded:
            out.append(f"{len(self.superseded)} replaced by the new spectrum's own entry")
        return ", ".join(out)

    def examples(self, n: int = 3) -> str:
        """A few of the moves and of the entries left as they were, ``old -> new``."""
        parts = [f"{o} -> {s}" for o, s in list(self.moved.items())[:n]]
        parts += [f"{o} (fits {', '.join(s[:3])}{', ...' if len(s) > 3 else ''})"
                  for o, s in list(self.several.items())[:n]]
        return "; ".join(parts)


def move_entries(entry_ids, current_ids, kinetics=None) -> Moves:
    """Where the review entries ``entry_ids`` whose spectrum the results ``current_ids`` no
    longer have would go (docs/PLAN.md Next steps 27): a rerun that sums other scans (a moved
    retention window, ``find_peak``, ``find_fragment_peak``) gives a spectrum a new id, as the
    id names its scans. Such an entry follows its spectrum by sample, file, and ion
    (:func:`scan_stem`) when exactly one spectrum of the results has that stem and it came from
    a run with a fixed exchange time (one spectrum per ion and file). Entries of kinetics runs
    (``kinetics``: a bool per current id; every scan group is its own time point, named by its
    scans) never move. An entry matching several spectra, or one whose spectrum another missing
    entry matches too, is not moved; one whose new spectrum has an entry of its own is
    superseded by it."""
    current = [str(s) for s in current_ids]
    have = set(current)
    kin = (np.zeros(len(current), dtype=bool) if kinetics is None
           else np.asarray(kinetics, dtype=bool))
    by_stem: dict[str, list[int]] = {}
    for i, sid in enumerate(current):
        stem = scan_stem(sid)
        if stem is not None:
            by_stem.setdefault(stem, []).append(i)
    entries = [str(s) for s in entry_ids]
    listed = set(entries)
    target: dict[str, str] = {}
    out = Moves()
    for sid in entries:
        if sid in have or (stem := scan_stem(sid)) is None or stem not in by_stem:
            continue
        hits = by_stem[stem]
        if any(kin[i] for i in hits):
            continue  # a kinetics run: its scans are its time points
        if len(hits) > 1:
            out.several[sid] = tuple(current[i] for i in hits)
            continue
        new = current[hits[0]]
        if new in listed:
            out.superseded[sid] = new
            continue
        target[sid] = new
    taken: dict[str, list[str]] = {}
    for old, new in target.items():
        taken.setdefault(new, []).append(old)
    for old, new in target.items():
        if len(taken[new]) == 1:
            out.moved[old] = new
        else:  # two old entries for one spectrum: neither is chosen over the other
            out.several[old] = (new,)
    return out


def results_moves(results, entry_ids) -> Moves:
    """:func:`move_entries` of ``entry_ids`` against the spectra of ``results``
    (:class:`hxex.report.Results`), kinetics runs told by the results' read record."""
    from .io.mzml import kinetics_spectra

    ids = results.fit.spectrum_id
    source = results.settings.spectra_source if results.settings is not None else None
    return move_entries(entry_ids, ids, kinetics_spectra(source, ids))


def moved_sheet(sheet: ReviewSheet, moves: Moves) -> ReviewSheet:
    """``sheet`` with the entries of ``moves.moved`` under their spectrum's new id."""
    if not moves.moved:
        return sheet
    import dataclasses

    return dataclasses.replace(sheet, spectrum_id=tuple(moves.moved.get(s, s)
                                                        for s in sheet.spectrum_id))


def _source_of(spectrum_id: str, origin: dict[str, str]) -> str:
    old = origin.get(spectrum_id)
    return "review table" if old is None else f"review table (moved from {old})"


def apply_review_table(results, sheet: ReviewSheet | None):
    """``results`` (:class:`hxex.report.Results`) with the decisions of ``sheet`` laid over its
    review layer, in memory only (the bundle is not changed): a status replaces the stored
    one, a chosen K replaces the stored one, and a note replaces the stored one when given.
    An entry naming scans the results no longer sum follows its spectrum (:func:`move_entries`;
    logged). Spectra the results do not have are skipped with a warning; a chosen K that was
    not fitted for its spectrum (one the run did not try, or one its screen skipped, which
    ``hxex resume`` fits) raises ValueError saying which."""
    from .io.bundle import ReviewTable

    if sheet is None or not len(sheet):
        return results
    fit = results.fit
    index = {str(s): i for i, s in enumerate(fit.spectrum_id)}
    moves = results_moves(results, sheet.spectrum_id)
    if moves:
        log.info("the review table %s names %d spectra by scans these results no longer "
                 "sum (a new retention window): %s (%s)", sheet.path,
                 len(moves.moved) + len(moves.several) + len(moves.superseded),
                 moves.describe(), moves.examples())
        sheet = moved_sheet(sheet, moves)
    unknown = [s for s in sheet.spectrum_id if s not in index and s not in moves.superseded]
    if unknown:
        log.warning("the review table %s names %d spectra these results do not have (for "
                    "example %s): they are skipped; check their spectrum_id (a typo, or a row "
                    "copied from another project), or the table may be from an earlier run "
                    "that had spectra since left out", sheet.path, len(unknown),
                    ", ".join(unknown[:3]))
    origin = {new: old for old, new in moves.moved.items()}
    tried = fit.config.k_values
    beyond = [f"{sid} (K = {k})" for sid, k in zip(sheet.spectrum_id, sheet.k, strict=True)
              if k > 0 and sid in index and k not in tried]
    if beyond:
        raise ValueError(f"the review table {sheet.path} chooses a K the run did not try for "
                         f"{len(beyond)} spectra ({', '.join(beyond[:3])}); the run tried K = "
                         f"{', '.join(map(str, tried))}: change [run.fit] k_min or k_max and "
                         "run again, or choose another K")
    skipped = [f"{sid} (K = {k})" for sid, k in zip(sheet.spectrum_id, sheet.k, strict=True)
               if k > 0 and sid in index and not fit.fitted(k)[index[sid]]]
    if skipped:
        raise ValueError(f"the review table {sheet.path} chooses a K the run's screen skipped "
                         f"for {len(skipped)} spectra ({', '.join(skipped[:3])}): 'hxex resume "
                         "hxex.toml' fits it for them (and bootstraps them), then report again")
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
        if j is None and not st and not nt and k == CLEARED:
            continue  # clearing a K never chosen changes nothing (owner, 2026-11-02)
        if j is None:
            ids.append(sid)
            status.append(st)
            k_override.append(max(k, 0))
            note.append(nt)
            updated.append(stamp)
            stale.append(False)
            source.append(_source_of(sid, origin))
            continue
        if st:
            status[j] = st
        if k:
            k_override[j] = max(k, 0)  # review_K = 0 clears the K chosen before
        if nt:
            note[j] = nt
        stale[j] = False
        source[j] = _source_of(sid, origin)
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
    """The K the project's review table chooses, by spectrum (none without a table), with
    :data:`CLEARED` where it clears a K chosen before (laid over the bundle's review layer's
    choices, it removes them: :func:`merge_choices`)."""
    if review is None or not review.table:
        return {}
    sheet = read_review_table(review.table)
    return {s: k for s, k in zip(sheet.spectrum_id, sheet.k, strict=True) if k}


def moved_choices(choices, spectrum_id, source=None) -> dict[str, int]:
    """``choices`` (spectrum id -> K, :func:`table_choices`) with the ids naming scans the
    spectra ``spectrum_id`` no longer sum replaced by their spectrum's new id
    (:func:`move_entries`; kinetics runs told by the read record ``source``)."""
    if not choices:
        return dict(choices or {})
    from .io.mzml import kinetics_spectra

    moves = move_entries(list(choices), spectrum_id, kinetics_spectra(source, spectrum_id))
    return {moves.moved.get(s, s): k for s, k in choices.items()}


def merge_choices(*layers) -> dict[str, int]:
    """The K chosen per spectrum from ``layers`` (spectrum id -> K mappings, later ones laid
    over earlier ones), without the cleared ones."""
    merged: dict[str, int] = {}
    for layer in layers:
        merged.update(layer or {})
    return {s: k for s, k in merged.items() if k > 0}


def choices_outside(choices, k_values) -> list[str]:
    """The table's choices (:func:`table_choices`) of a K the run does not try (``k_values``,
    :attr:`hxex.fit.FitConfig.k_values`), as ``id (K = k)``."""
    return [f"{s} (K = {k})" for s, k in choices.items() if k > 0 and k not in k_values]
