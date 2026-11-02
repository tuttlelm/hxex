# SPDX-License-Identifier: GPL-3.0-or-later
"""Peptide tables: the metadata CSV (pyHXExpress's metadf), FASTA files, entry filters and
retention-time tables."""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from ..chem.modifications import SequenceError, split_termini
from ..spectra import PeptideEntry, SpectrumSet, entry_rt, entry_rt_window
from ._common import not_text_error, open_text
from .errors import ReadError

log = logging.getLogger(__name__)

_REQUIRED = ("file", "sample", "peptide", "start_seq", "end_seq", "charge")
# the peptide table of the scan formats: file is for reference only, and an empty or missing
# sample means every run
SCAN_REQUIRED = ("peptide", "start_seq", "end_seq", "charge")
_RANGE = re.compile(r"\s*0*(\d+)\s*-\s*0*(\d+)\s*")
# the column names a table is read by: a header cell is matched to them without regard to case
# or surrounding spaces (Excel capitalises; a user adds a space); other columns keep their
# names, stripped
KNOWN_COLUMNS = frozenset((*_REQUIRED, "peptide_range", "modification", "index", "rt",
                           "rt_start", "rt_end", "run", "not_runs"))
_SEPARATORS = {";": "';' (Excel's regional setting)", "\t": "tabs"}


def _table_reader(fh, path: Path) -> csv.DictReader:
    """A CSV reader of a table with a header row, its column names normalised
    (:data:`KNOWN_COLUMNS`); a table whose columns are separated by semicolons or tabs is
    refused, saying so."""
    reader = csv.DictReader(fh)
    names = reader.fieldnames or []
    if len(names) == 1:
        for sep, text in _SEPARATORS.items():
            if sep in names[0]:
                raise ReadError(f"{path}: the columns are separated by {text}, not commas: "
                                "save the table as CSV with commas")
    reader.fieldnames = [_column_name(n) for n in names]
    return reader


def _column_name(name: str | None) -> str:
    text = (name or "").strip()
    return text.lower() if text.lower() in KNOWN_COLUMNS else text


def _columns_found(reader: csv.DictReader) -> str:
    names = [n for n in (reader.fieldnames or []) if n]
    return f"the columns found are {', '.join(names)}" if names else "the header row is empty"


def _rows(reader: csv.DictReader, path: Path):
    """``(line, row)`` of each row of ``reader`` that has a cell, every cell stripped; rows of
    empty cells (``,,,,`` left by Excel, a blank line) are skipped and counted in the log; a row
    with fewer or more cells than the header has columns is refused, naming the line."""
    skipped = 0
    for line, row in enumerate(reader, start=2):
        cells = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
        if all(not v for k, v in cells.items() if k is not None) and not cells.get(None):
            skipped += 1
            continue
        if None in cells.values():
            raise ReadError(f"{path} line {line}: fewer cells than the header has columns "
                            f"({len(reader.fieldnames or [])})")
        if cells.get(None):
            raise ReadError(f"{path} line {line}: more cells than the header has columns "
                            f"({len(reader.fieldnames or [])})")
        yield line, cells
    if skipped:
        log.info("%s: %d empty row(s) skipped", path, skipped)


def peptide_range_text(start: int, end: int) -> str:
    """The range label of residues ``start`` to ``end``, zero-padded as in the file names of
    the folder formats (``0001-0010``)."""
    return f"{start:04d}-{end:04d}"


def read_peptide_table(path: str | Path, *,
                       required: Sequence[str] = _REQUIRED) -> list[PeptideEntry]:
    """Read a metadata CSV (columns as in pyHXExpress's ``metadf``).

    Required (``required``, by default): file, sample, peptide, start_seq, end_seq, charge;
    the scan formats need :data:`SCAN_REQUIRED` only (a missing file or sample column reads as
    empty). ``peptide_range`` (the range label, text) is optional: an empty or missing one is
    made from ``start_seq`` and ``end_seq`` (:func:`peptide_range_text`), and one that names
    other residues is logged as a warning (the residues are ``start_seq`` and ``end_seq``).
    Optional too:
    modification, ``rt`` (the retention time in minutes, kept with the spectra), ``rt_start``
    and ``rt_end`` (the peptide's retention window in minutes: for mzML and Thermo runs with a
    fixed time, the scans summed for it and its fragments). The ``Index``
    column is ignored and any other column is kept in ``extra``. Values are text
    (``peptide_range`` keeps its leading zeros). In a folder table (``required`` the default),
    a row equal to an earlier one in every column of the entry is refused, naming both lines;
    rows that differ in their modification only are two series.
    """
    path = Path(path)
    try:
        with open_text(path, newline="") as fh:
            reader = _table_reader(fh, path)
            missing = [c for c in required if c not in (reader.fieldnames or [])]
            if missing:
                raise ReadError(f"{path}: missing column(s) {', '.join(missing)}; "
                                f"{_columns_found(reader)}")
            entries = []
            # a folder table's row listed twice (a copy in Excel) would read the same spectra
            # as two series; rows that differ in their modification are two series by design
            folder_table = tuple(required) == _REQUIRED
            seen: dict[tuple, int] = {}
            for line, row in _rows(reader, path):
                try:
                    entry = _entry_from_row(row)
                except (ValueError, TypeError) as exc:
                    raise ReadError(f"{path} line {line}: {exc}") from exc
                if folder_table:
                    key = (entry.file, entry.sample, entry.peptide, entry.start_seq,
                           entry.end_seq, entry.charge, entry.modification)
                    if key in seen:
                        raise ReadError(
                            f"{path} line {line}: listed again (the same file, sample, "
                            f"peptide, residues, charge and modification as line {seen[key]}); "
                            "delete one of the two lines")
                    seen[key] = line
                m = _RANGE.fullmatch(entry.peptide_range)
                if m is None or (int(m[1]), int(m[2])) != (entry.start_seq, entry.end_seq):
                    log.warning("%s line %d: peptide_range %r does not match start_seq %d and "
                                "end_seq %d; the residues are %d to %d (peptide_range is only "
                                "the label)", path, line, entry.peptide_range, entry.start_seq,
                                entry.end_seq, entry.start_seq, entry.end_seq)
                entries.append(entry)
    except (OSError, UnicodeDecodeError) as exc:
        raise not_text_error(path, exc) from exc
    return entries


def _whole(value: str | None, name: str) -> int:
    """An integer column value (``2`` or ``2.0``; empty, a fraction or a non-number is an error
    naming the column)."""
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{name} is empty")
    try:
        number = float(text)
    except ValueError:
        hint = ": use a decimal point" if "," in text else ""
        raise ValueError(f"{name} must be a whole number, not {text!r}{hint}") from None
    if not number.is_integer():
        raise ValueError(f"{name} must be a whole number, not {text!r}")
    return int(number)


def _entry_from_row(row: dict[str, str]) -> PeptideEntry:
    """The entry of one table row (its cells already stripped, :func:`_rows`)."""
    known = set(_REQUIRED) | {"peptide_range", "modification", "index"}
    start, end = _whole(row["start_seq"], "start_seq"), _whole(row["end_seq"], "end_seq")
    peptide = (row.get("peptide") or "").strip()
    if not peptide:
        raise ValueError("peptide is empty")
    entry = PeptideEntry(
        file=(row.get("file") or "").strip(),
        sample=(row.get("sample") or "").strip(),
        peptide=peptide,
        peptide_range=(row.get("peptide_range") or "").strip() or peptide_range_text(start, end),
        start_seq=start,
        end_seq=end,
        charge=_whole(row["charge"], "charge"),
        modification=(row.get("modification") or "").strip(),
        extra={k: (v or "").strip() for k, v in row.items() if k not in known and k is not None},
    )
    entry_rt(entry)  # a retention time must be a number
    entry_rt_window(entry)  # and a window both ends, in order
    return entry


# --- retention times ---------------------------------------------------------------------------

_RT_KEYS = ("sample", "start_seq", "end_seq", "peptide", "charge")


def read_retention_times(path: str | Path) -> list[dict]:
    """A retention-time table (CSV): ``rt`` (minutes) and the columns that say which spectra
    it belongs to, any of ``sample``, ``start_seq`` and ``end_seq`` (or ``peptide_range``,
    ``'12-20'``), ``peptide`` and ``charge`` (absent or empty: any). Returns one dict per row
    with the keys given."""
    path = Path(path)
    rows = []
    try:
        with open_text(path, newline="") as fh:
            reader = _table_reader(fh, path)
            names = reader.fieldnames or []
            if "rt" not in names:
                raise ReadError(f"{path}: no rt column (retention time, minutes); "
                                f"{_columns_found(reader)}")
            if not set(names) & {"start_seq", "peptide_range", "peptide"}:
                raise ReadError(f"{path}: name the peptides (start_seq and end_seq, "
                                f"peptide_range or peptide); {_columns_found(reader)}")
            for line, row in _rows(reader, path):
                try:
                    rows.append(_rt_row(row))
                except ValueError as exc:
                    raise ReadError(f"{path} line {line}: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise not_text_error(path, exc) from exc
    return rows


def _rt_row(row: dict) -> dict:
    """The retention time and the columns naming its spectra of one row (cells stripped)."""
    rt = (row.get("rt") or "").strip()
    if not rt:
        raise ValueError("rt is empty")
    try:
        out: dict = {"rt": float(rt)}
    except ValueError:
        raise ValueError(f"rt {rt!r} is not a number of minutes") from None
    rng = (row.get("peptide_range") or "").strip()
    if rng and not (row.get("start_seq") or "").strip():
        first, _, last = rng.partition("-")
        row = {**row, "start_seq": first, "end_seq": last}
    for key in _RT_KEYS:
        value = (row.get(key) or "").strip()
        if value:
            out[key] = _whole(value, key) if key in ("start_seq", "end_seq", "charge") else value
    if not set(out) - {"rt"}:
        raise ValueError("no sample, peptide, range or charge: the row would apply to every "
                         "spectrum")
    return out


def apply_retention_times(spectra: SpectrumSet, rows: Sequence[dict]) -> SpectrumSet:
    """``spectra`` with the retention times of ``rows`` (:func:`read_retention_times`): each
    spectrum takes the value of the row that matches it on the most columns (rows that name a
    column it does not match are skipped); other spectra keep theirs. Two rows that match one
    spectrum on the same number of columns with different values are an error naming both."""
    import dataclasses

    values = {"sample": spectra.sample, "start_seq": spectra.start_seq,
              "end_seq": spectra.end_seq, "peptide": spectra.peptide, "charge": spectra.charge}
    rt = np.array(spectra.rt, dtype=float, copy=True)
    best = np.full(len(spectra), -1)
    by = np.full(len(spectra), -1)  # the row each spectrum took its value from
    matched = np.zeros(len(rows), dtype=bool)

    def described(row: dict) -> str:
        return ", ".join(f"{k} {row[k]}" for k in (*_RT_KEYS, "rt") if k in row)

    for n, row in enumerate(rows):
        keep = np.ones(len(spectra), dtype=bool)
        for key in _RT_KEYS:
            if key in row:
                keep &= np.asarray(values[key]).astype(str) == str(row[key])
        score = sum(key in row for key in _RT_KEYS)
        clash = keep & (best == score) & ~np.isclose(rt, row["rt"])
        if clash.any():
            i = int(np.flatnonzero(clash)[0])
            raise ReadError(
                f"two retention times for {spectra.sample[i]} {spectra.peptide[i]}: the rows "
                f"({described(rows[by[i]])}) and ({described(row)}) both match it on {score} "
                "column(s); give one of them more columns, or remove one")
        take = keep & (score > best)
        rt[take] = row["rt"]
        best[take] = score
        by[take] = n
        matched[n] = keep.any()
    if not matched.all():
        log.warning("retention times: %d row(s) match no spectrum", int((~matched).sum()))
    log.info("retention times: %d of %d spectra from the table", int((best >= 0).sum()),
             len(spectra))
    return dataclasses.replace(spectra, rt=rt)


def read_fasta_records(path: str | Path) -> list[tuple[str, str]]:
    """``(id, sequence)`` for each FASTA record, in file order.

    The id is the header up to the first whitespace. Sequence lines are joined and whitespace
    removed. Records with no sequence are dropped.
    """
    records: list[tuple[str, list[str]]] = []
    try:
        with open_text(path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    header = line[1:].split(maxsplit=1)
                    records.append((header[0] if header else "", []))
                elif line and records:
                    records[-1][1].append("".join(line.split()))
    except (OSError, UnicodeDecodeError) as exc:
        raise not_text_error(path, exc) from exc
    return [(name, "".join(parts)) for name, parts in records if parts]


def read_fasta_sequence(path: str | Path) -> str:
    """The sequence of a FASTA file. With several records the last is used, as pyHXExpress does;
    a sequence written wholly in lower case is upper-cased (plain FASTA is case-insensitive)."""
    return _last_record(path)[1]


# a Pfam-style record id, ``NAME/START-END``: the residue numbers of the sequence's first and
# last residues (START may be zero or negative when a tag precedes residue 1)
_NUMBERED_ID = re.compile(r"/(-?\d+)-(-?\d+)$")


def read_fasta_numbered(path: str | Path) -> tuple[str, int]:
    """The sequence of a FASTA file (as :func:`read_fasta_sequence`) and the residue number of
    its first residue.

    The number comes from a Pfam-style header, ``>NAME/START-END`` (the record id ends in
    ``/START-END``): the first residue is residue START, so that a FASTA holding a tag or a
    signal peptide can be numbered as the data were (START may be zero or negative). Without
    such a header the first residue is residue 1. A header whose range does not have as many
    residues as the sequence (counted as modX residues, terminal groups left out) is refused.
    """
    name, sequence = _last_record(path)
    m = _NUMBERED_ID.search(name)
    if m is None:
        return sequence, 1
    first, last = int(m[1]), int(m[2])
    count = fasta_residue_count(sequence)
    if last - first + 1 != count:
        raise ReadError(
            f"{path}: the header numbers the residues {first} to {last} ({last - first + 1} "
            f"residues), but the sequence has {count}; correct the numbers after the '/' of "
            f"the header line (>{fasta_name(name) or 'NAME'}/FIRST-LAST, the numbers of the "
            "first and last residues)")
    return sequence, first


def fasta_name(record_id: str) -> str:
    """A FASTA record id without its Pfam-style residue numbers (``/START-END``)."""
    return _NUMBERED_ID.sub("", record_id)


def fasta_residue_count(sequence: str) -> int:
    """The residues of a FASTA sequence: its capital letters, terminal groups left out (a modX
    label such as ``acK`` is one residue), so that no registry is needed to count them."""
    try:
        core = split_termini(sequence)[1]
    except SequenceError:
        core = sequence
    return sum(c.isupper() for c in core)


def _last_record(path: str | Path) -> tuple[str, str]:
    """``(id, sequence)`` of a FASTA file's last record (a warning when it has several), its
    sequence upper-cased when it is written wholly in lower case."""
    records = read_fasta_records(path)
    if not records:
        raise ReadError(f"{path}: no sequence found")
    if len(records) > 1:
        log.warning("%s has %d records; using the last", path, len(records))
    name, sequence = records[-1]
    if sequence.islower():
        log.info("%s: the sequence is in lower case; read in capitals", path)
        sequence = sequence.upper()
    return name, sequence


def parse_range(text: str) -> tuple[int, int]:
    """``(start, end)`` of a range written ``START-END`` (leading zeros and spaces ignored:
    ``"0001-0010"`` and ``"1-10"`` are the same range); a :class:`ValueError` otherwise."""
    m = _RANGE.fullmatch(str(text))
    if m is None:
        raise ValueError(f"{text!r} is not a range START-END (for example 1-10 or 0001-0010)")
    return int(m[1]), int(m[2])


def filter_entries(
    entries: Iterable[PeptideEntry],
    *,
    samples: Sequence[str] | None = None,
    peptide_ranges: Sequence[str] | None = None,
    peptide_region: Sequence[int] | None = None,
) -> list[PeptideEntry]:
    """Keep the entries of the listed samples (None: all) whose residues are one of the
    ``peptide_ranges`` (``START-END``, compared as numbers with ``start_seq`` and ``end_seq``)
    or lie wholly within ``peptide_region`` (``[first, last]`` residue); with neither, every
    peptide; with both, a peptide that matches either."""
    exact = None if peptide_ranges is None else {parse_range(r) for r in peptide_ranges}
    region = None if peptide_region is None else tuple(int(x) for x in peptide_region)

    def wanted(e: PeptideEntry) -> bool:
        if exact is None and region is None:
            return True
        return ((exact is not None and (e.start_seq, e.end_seq) in exact)
                or (region is not None and region[0] <= e.start_seq and e.end_seq <= region[1]))

    kept = [
        e
        for e in entries
        if (samples is None or e.sample in samples) and wanted(e)
    ]
    return kept


def sort_entries(entries: Iterable[PeptideEntry]) -> list[PeptideEntry]:
    """Order by peptide range, sample, charge, as pyHXExpress does."""
    return sorted(entries, key=lambda e: (e.peptide_range, e.sample, e.charge))
