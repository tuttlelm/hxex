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
editing the table needs only the command run again. A row with ``review_status``,
``review_K`` and ``note`` empty changes nothing; ``review_status = unreviewed``, written,
clears a status the bundle's review layer holds (as ``review_K = 0`` clears a K: audit of
2026-11-15, D-2).

A chosen K is drawn and tabulated with the errors of a bootstrap at that K: :func:`chosen_k`
lists the spectra shown at a K other than the fit's choice and :func:`missing_bootstraps` those
of them without a bootstrap at it, which ``hxex run`` adds. A chosen K the
screen skipped for a spectrum is shown at the fit's K until ``hxex run`` fits it (and
bootstraps it); :func:`missing_bootstraps` lists it too, and :func:`refuse_unfitted` stops an
analysis on it.
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
    which clears a K chosen before) and ``note``; ``path`` is the file. ``unreviewed``: the
    spectra whose ``review_status`` says ``unreviewed`` (written, not left empty), which clears
    a status the bundle's review layer holds; a row saying only that is not an entry."""

    spectrum_id: tuple[str, ...]
    status: tuple[str, ...]
    k: tuple[int, ...]
    note: tuple[str, ...]
    path: str = ""
    unreviewed: tuple[str, ...] = ()
    # whether the table has a note column: a row that says something (a status, unreviewed
    # included, or a K) then replaces the stored note with its own note cell, empty included
    # (the notes rule; audit of 2026-11-15, scope D); without the column notes stay
    has_notes: bool = True

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


def delimiter(header: str) -> str:
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
                        delimiter=delimiter(text.splitlines()[0]))
    header = [h.strip() for h in next(reader)]
    where = {name: header.index(name) for name in _COLUMNS if name in header}
    if "spectrum_id" not in where:
        raise ValueError(f"the review table {path} has no spectrum_id column: start from a "
                         "copy of spectra.csv in the tables folder")
    if "review_status" not in where and "review_K" not in where:
        raise ValueError(f"the review table {path} has neither a review_status nor a review_K "
                         "column")
    entries: dict[str, tuple[str, int, str, int]] = {}
    unreviewed: dict[str, None] = {}  # written as such: clears the bundle's status
    seen: dict[str, tuple[str, int, int]] = {}  # each id's first row: status, K, row number
    problems: list[str] = []
    for row_number, row in enumerate(reader, start=2):
        def cell(name: str, row=row) -> str:
            j = where.get(name)
            return row[j].strip() if j is not None and j < len(row) else ""

        sid = cell("spectrum_id")
        status, k_text, note = cell("review_status").lower(), cell("review_K"), cell("note")
        said_unreviewed = status == "unreviewed"  # spectra.csv's word for no decision
        if said_unreviewed:
            status = ""
        if not sid or (not status and not k_text and not note and not said_unreviewed):
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
        if sid in seen and seen[sid][:2] != (status, k):
            problems.append(f"row {row_number}: {sid} is already in row {seen[sid][2]} with "
                            "other values")
            continue
        seen.setdefault(sid, (status, k, row_number))
        if said_unreviewed:
            unreviewed.setdefault(sid)
        if not k and not note and not status:  # unreviewed alone: a clear, not an entry
            continue
        entries.setdefault(sid, (status, k, note, row_number))
    if problems:
        raise ValueError(f"the review table {path} (rows numbered as a spreadsheet numbers "
                         "them, the header being row 1): " + "; ".join(problems[:10])
                         + (f"; and {len(problems) - 10} more" if len(problems) > 10 else ""))
    ids = list(entries)
    return ReviewSheet(tuple(ids), tuple(entries[s][0] for s in ids),
                       tuple(entries[s][1] for s in ids), tuple(entries[s][2] for s in ids),
                       str(path), tuple(unreviewed), has_notes="note" in where)


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

    return dataclasses.replace(
        sheet, spectrum_id=tuple(moves.moved.get(s, s) for s in sheet.spectrum_id),
        unreviewed=tuple(moves.moved.get(s, s) for s in sheet.unreviewed))


def _source_of(spectrum_id: str, origin: dict[str, str]) -> str:
    old = origin.get(spectrum_id)
    return "review table" if old is None else f"review table (moved from {old})"


def apply_review_table(results, sheet: ReviewSheet | None, *, unfitted: str = "raise"):
    """``results`` (:class:`hxex.report.Results`) with the decisions of ``sheet`` laid over its
    review layer, in memory only (the bundle is not changed): a status replaces the stored
    one, a chosen K replaces the stored one, and a row that says something (a status,
    ``unreviewed`` included, or a K) replaces the stored note with its own note cell, empty
    included (a table without a note column leaves the stored notes); a row with a note alone
    replaces the note; ``review_status = unreviewed`` (written) clears the stored status, and
    ``review_K = 0`` the stored K.
    An entry naming scans the results no longer sum follows its spectrum (:func:`move_entries`;
    logged). Spectra the results do not have are skipped with a warning; a chosen K that was
    not fitted for its spectrum (one the run did not try, or one its screen skipped, which
    ``hxex run`` fits) raises ValueError saying which; with ``unfitted="keep"`` a K the
    screen skipped is kept in the review layer and the spectrum shown at the fit's K
    (:attr:`hxex.report.Results.unfitted_choices`; the window loads results so, the commands
    refuse: audit of 2026-11-15, D-9)."""
    from .io.bundle import ReviewTable

    if unfitted not in ("raise", "keep"):
        raise ValueError(f"unfitted must be 'raise' or 'keep', not {unfitted!r}")
    if sheet is None or not (len(sheet) or sheet.unreviewed):
        return results
    fit = results.fit
    index = {str(s): i for i, s in enumerate(fit.spectrum_id)}
    moves = results_moves(results, list(dict.fromkeys((*sheet.spectrum_id,
                                                       *sheet.unreviewed))))
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
    if skipped and unfitted == "raise":
        raise ValueError(f"the review table {sheet.path} chooses a K the run's screen skipped "
                         f"for {len(skipped)} spectra ({', '.join(skipped[:3])}): 'hxex run "
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
    unreviewed = set(sheet.unreviewed)
    for sid in unreviewed:  # a status the bundle holds, cleared (a K or note set below); the
        # row says something, so its note cell (empty, when the row is only that) is the note
        j = stored.get(sid)
        if j is not None and sid in index and (status[j] or (sheet.has_notes and note[j])):
            status[j] = ""
            if sheet.has_notes:
                note[j] = ""
            stale[j] = False
            source[j] = _source_of(sid, origin)
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
        if nt or sheet.has_notes:  # the row says something: its note cell is the note
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
    out = dict(results.unfitted_choices)  # not even fitted yet: hxex run fits and bootstraps
    if results.bootstrap is None and not results.bootstrap_by_k:
        return out
    index = {str(s): i for i, s in enumerate(results.fit.spectrum_id)}
    for sid, k in chosen_k(results).items():
        boot = results.bootstrap_for(index[sid], k)
        if boot is None or not boot.covers(index[sid]):
            out[sid] = k
    return out


def refuse_unfitted(results) -> None:
    """Raise ValueError when the review chooses a K the screen skipped for some spectra: they
    would be analysed at the fit's K. ``hxex run`` fits them."""
    if unfitted := results.unfitted_choices:
        shown = ", ".join(f"{s} at K = {k}" for s, k in list(unfitted.items())[:2])
        raise ValueError(f"the review chooses a K that was not fitted for {len(unfitted)} "
                         f"spectra (the screen skipped it; for example {shown}): hxex run "
                         "fits it (and bootstraps it), then analyse again")


def apply_view(results, review: ReviewSettings | None, *, unfitted: str = "raise"):
    """``results`` as the project's ``[review]`` settings present them: the review table's
    decisions, then the ``[[review.exclude]]`` rules (which reject whatever the table says), in
    memory. ``unfitted`` as :func:`apply_review_table`."""
    from .exclusions import exclude_from_results

    if not review:
        return results
    if review.table:
        results = apply_review_table(results, read_review_table(review.table),
                                     unfitted=unfitted)
    return exclude_from_results(results, review.exclude)


def table_choices(review: ReviewSettings | None) -> dict[str, int]:
    """The K the project's review table chooses, by spectrum (none without a table), with
    :data:`CLEARED` where it clears a K chosen before (laid over the bundle's review layer's
    choices, it removes them: :func:`merge_choices`)."""
    if review is None or not review.table:
        return {}
    sheet = read_review_table(review.table)
    return {s: k for s, k in zip(sheet.spectrum_id, sheet.k, strict=True) if k}


def rejected_spectra(review: ReviewSettings | None, index, *, stored=None, source=None
                     ) -> np.ndarray:
    """Which spectra of ``index`` (a binned or spectrum set: their ids, and what
    ``[[review.exclude]]`` rules match on) the review rejects, as the analysis sees them: the
    stored review layer's current statuses (``stored``, a :class:`hxex.io.bundle.ReviewTable`
    of the bundle a run replaces), the project's review table laid over them (a status
    replaces the stored one, ``unreviewed`` clears it; entries naming scans the spectra no
    longer sum follow their spectrum, :func:`move_entries`), then ``[[review.exclude]]``,
    which rejects whatever the table says. ``source``: the read record (the reader, for rules
    naming a file; kinetics runs). The run leaves rejected undeuterated controls out of the
    per-bin noise (owner, 2026-11-20)."""
    from .exclusions import matches
    from .io.mzml import kinetics_spectra

    ids = np.asarray(index.spectrum_id, dtype=str)
    status: dict[str, str] = {}
    if stored is not None and len(stored):
        stale = (np.zeros(len(stored), dtype=bool) if stored.stale is None
                 else np.asarray(stored.stale, dtype=bool))
        for sid, st, old in zip(stored.spectrum_id, stored.status, stale, strict=True):
            if st and not old:
                status[str(sid)] = str(st)
    if review is not None and review.table:
        sheet = read_review_table(review.table)
        named = list(dict.fromkeys((*sheet.spectrum_id, *sheet.unreviewed)))
        moves = move_entries(named, ids, kinetics_spectra(source, ids)) if named else None
        if moves:
            sheet = moved_sheet(sheet, moves)
        for sid in sheet.unreviewed:
            status.pop(sid, None)
        for sid, st in zip(sheet.spectrum_id, sheet.status, strict=True):
            if st:
                status[sid] = st
    out = np.array([status.get(s) == "rejected" for s in ids], dtype=bool)
    if review is not None and review.exclude:
        reader = (source or {}).get("reader")
        for rule in review.exclude:
            out |= matches(rule, index, reader)
    return out


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


# --- the window's side of the table (moved here from hxex.gui, audit of 2026-11-15, Q-4: the
# rules of the table belong to the library, so the window, the commands and Python agree)
#
# The window writes a decision as a user would type it: ``review_status`` (``accepted``,
# ``rejected``, empty, or ``unreviewed``, which clears a status the bundle itself holds),
# ``review_K`` (a K, or empty for the fit's choice; ``0`` clears a K the bundle itself chose),
# and ``note``, in the row of the spectrum (in every row of it, when the table lists it twice;
# a row is added for a spectrum the table does not list). Other columns, and columns of the
# user's own, stay as they are. A row that says something carries its note cell, empty
# included (the commands read it so: the notes rule, owner 2026-11-15), so a decision without
# a note of its own fills an empty cell with the note shown, and a note emptied on a spectrum
# with no other decision is written with ``unreviewed`` (or the status shown) beside it. When
# the project names no table, :func:`make_review_table` writes one from the results' spectra
# (the suggested name, beside the project file) and names it in the project file.

DECISION_COLUMNS = ("review_status", "review_K", "note")
# "unreviewed": no decision, clearing one the bundle's review layer holds (an empty cell keeps it)
DECISION_STATUSES = ("", "accepted", "rejected", "unreviewed")


@dataclass(frozen=True)
class Decision:
    """What the window writes for a spectrum: a status (``''``: none), a K (None: the fit's
    choice), and a note (None: left as it is). ``unreviewed`` as the status clears a status
    the bundle's review layer holds, which an empty cell leaves in place."""

    status: str | None = None
    k: int | None = None
    k_given: bool = False  # whether ``k`` is part of the decision (None then means the fit's)
    note: str | None = None


def review_table_path(project: Path) -> Path | None:
    """The project's ``[review] table`` (None when it names none)."""
    from .project_edit import live_settings

    value = live_settings(project.read_text(encoding="utf-8-sig"))[0].get("review.table")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else project.parent / path


def make_review_table(project: Path, results, table: Path | None = None) -> Path:
    """A user review table made from the results' spectra (a copy of ``spectra.csv``), named in
    the project file's ``[review] table``: the table named there when it exists (returned as it
    is), else ``table`` (the window asks where, :func:`hxex.gui.table_choice.ask_table`; a
    table that exists is used as it is) or the suggested name beside the project file.

    The spectra an ``[[review.exclude]]`` rule rejects are written without a status or a note
    of the rule's: the rules apply when the table is read, so a rule removed later leaves
    them as the bundle has them (audit of 2026-11-15, D-15)."""
    from .io.user_tables import Table, write_table
    from .project_edit import quoted, relative_path, set_settings
    from .report.tables import _RULE_NOTE, spectra_table

    named = review_table_path(project)
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
        rows = [[_cell_text(cols[n][i]) for n in names] for i in range(len(cols["spectrum_id"]))]
        write_table(Table(names, rows), path)
    if named is None:
        set_settings(project, [("review.table", quoted(relative_path(path, project.parent)))],
                     unfinished=True)
    return path


def _cell_text(value) -> str:
    """A cell as spectra.csv writes it (numpy's repr would write ``np.float64(4.0)``)."""
    from .report.tables import csv_cell

    return "" if value is None else csv_cell(value)


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
    :class:`hxex.io.user_tables.TableLocked` when another program holds the file."""
    for d in decisions.values():
        if d.status is not None and d.status not in DECISION_STATUSES:
            raise ValueError(f"review status must be accepted, rejected, unreviewed, or empty, "
                             f"not {d.status!r}")
    from .io.user_tables import read_table, write_table

    table = read_table(path)
    for name in ("spectrum_id", *DECISION_COLUMNS):
        if name not in table.columns:
            table.columns.append(name)
            for row in table.rows:
                row.append("")
    col = {name: table.columns.index(name) for name in ("spectrum_id", *DECISION_COLUMNS)}
    rows: dict[str, list[list[str]]] = {}
    for row in table.rows:
        rows.setdefault(row[col["spectrum_id"]].strip(), []).append(row)
    rows.pop("", None)
    moved = {}
    if results is not None:
        moved = {new: old for old, new in results_moves(results, list(rows)).moved.items()}
    # a row that says something replaces the stored note with its own note cell, empty
    # included (the notes rule): a decision without a note of its own writes the note shown
    # (the results' review layer's) into a row whose cell is empty, so that a status or K
    # chosen does not lose the run's own note
    shown: dict[str, str] = {}
    if results is not None and results.review is not None:
        for sid, d in decisions.items():
            if d.note is None and (d.status is not None or d.k_given):
                entry = results.review.get(str(sid).strip())
                if entry and entry["note"]:
                    shown[str(sid).strip()] = str(entry["note"])
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
            elif sid in shown and not row[col["note"]].strip():
                row[col["note"]] = shown[sid]
    write_table(table, path)


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
    from .config import ProjectError, load_project
    from .io import ReadError, open_bundle
    from .io.bundle import bundle_manifest
    from .io.mzml import kinetics_spectra

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


def review_match(table, results: ResultsIds) -> ReviewMatch:
    """Which rows of the review ``table`` (:class:`hxex.io.user_tables.Table`) name a spectrum
    ``results`` do not have. Only rows the reader keeps count (a spectrum_id and a status, a K
    or a note; ``unreviewed`` is no status); a row whose spectrum was summed from other scans in
    a rerun follows it (:func:`move_entries`) and is not unmatched."""
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
