# SPDX-License-Identifier: GPL-3.0-or-later
"""Peptide tables: the metadata CSV (pyHXExpress's metadf), FASTA files, entry filters and
retention-time tables."""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from ..spectra import PeptideEntry, SpectrumSet, entry_rt, entry_rt_window
from .errors import ReadError

log = logging.getLogger(__name__)

_REQUIRED = ("file", "sample", "peptide", "peptide_range", "start_seq", "end_seq", "charge")


def read_peptide_table(path: str | Path) -> list[PeptideEntry]:
    """Read a metadata CSV (columns as in pyHXExpress's ``metadf``).

    Required: file, sample, peptide, peptide_range, start_seq, end_seq, charge. Optional:
    modification, ``rt`` (the retention time in minutes, kept with the spectra), ``rt_start``
    and ``rt_end`` (the peptide's retention window in minutes: for mzML and Thermo runs with a
    fixed time, the scans summed for it and its fragments). The ``Index``
    column is ignored and any other column is kept in ``extra``. Values are text
    (``peptide_range`` keeps its leading zeros).
    """
    path = Path(path)
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in _REQUIRED if c not in (reader.fieldnames or [])]
            if missing:
                raise ReadError(f"{path}: missing column(s) {', '.join(missing)}")
            entries = []
            for line, row in enumerate(reader, start=2):
                try:
                    entries.append(_entry_from_row(row))
                except (ValueError, TypeError) as exc:
                    raise ReadError(f"{path} line {line}: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ReadError(f"cannot read {path}: {exc}") from exc
    return entries


def _whole(value: str, name: str) -> int:
    """An integer column value (``2`` or ``2.0``; a fraction or a non-number is an error)."""
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"{name} must be a whole number, not {value!r}")
    return int(number)


def _entry_from_row(row: dict[str, str]) -> PeptideEntry:
    known = set(_REQUIRED) | {"modification", "Index"}
    entry = PeptideEntry(
        file=row["file"],
        sample=row["sample"],
        peptide=row["peptide"],
        peptide_range=row["peptide_range"],
        start_seq=_whole(row["start_seq"], "start_seq"),
        end_seq=_whole(row["end_seq"], "end_seq"),
        charge=_whole(row["charge"], "charge"),
        modification=(row.get("modification") or "").strip(),
        extra={k: v for k, v in row.items() if k not in known and k is not None},
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
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            names = reader.fieldnames or []
            if "rt" not in names:
                raise ReadError(f"{path}: no rt column (retention time, minutes)")
            if not set(names) & {"start_seq", "peptide_range", "peptide"}:
                raise ReadError(f"{path}: name the peptides (start_seq and end_seq, "
                                "peptide_range or peptide)")
            for line, row in enumerate(reader, start=2):
                try:
                    rows.append(_rt_row(row))
                except ValueError as exc:
                    raise ReadError(f"{path} line {line}: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ReadError(f"cannot read {path}: {exc}") from exc
    return rows


def _rt_row(row: dict) -> dict:
    if None in row.values():
        raise ValueError("fewer cells than the header has columns")
    if row.get(None):
        raise ValueError("more cells than the header has columns")
    out: dict = {"rt": float(row["rt"])}
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
    column it does not match are skipped); other spectra keep theirs. Two rows matching on the
    same columns with different values are an error."""
    import dataclasses

    values = {"sample": spectra.sample, "start_seq": spectra.start_seq,
              "end_seq": spectra.end_seq, "peptide": spectra.peptide, "charge": spectra.charge}
    rt = np.array(spectra.rt, dtype=float, copy=True)
    best = np.full(len(spectra), -1)
    matched = np.zeros(len(rows), dtype=bool)
    for n, row in enumerate(rows):
        keep = np.ones(len(spectra), dtype=bool)
        for key in _RT_KEYS:
            if key in row:
                keep &= np.asarray(values[key]).astype(str) == str(row[key])
        score = sum(key in row for key in _RT_KEYS)
        clash = keep & (best == score) & ~np.isclose(rt, row["rt"])
        if clash.any():
            raise ReadError(f"two retention times ({rt[clash][0]:g} and {row['rt']:g} min) for "
                            f"{spectra.sample[clash][0]} {spectra.peptide[clash][0]}")
        take = keep & (score > best)
        rt[take] = row["rt"]
        best[take] = score
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
        with Path(path).open(encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    header = line[1:].split(maxsplit=1)
                    records.append((header[0] if header else "", []))
                elif line and records:
                    records[-1][1].append("".join(line.split()))
    except (OSError, UnicodeDecodeError) as exc:
        raise ReadError(f"cannot read {path}: {exc}") from exc
    return [(name, "".join(parts)) for name, parts in records if parts]


def read_fasta_sequence(path: str | Path) -> str:
    """The sequence of a FASTA file. With several records the last is used, as pyHXExpress does."""
    records = read_fasta_records(path)
    if not records:
        raise ReadError(f"{path}: no sequence found")
    if len(records) > 1:
        log.warning("%s has %d records; using the last", path, len(records))
    return records[-1][1]


def filter_entries(
    entries: Iterable[PeptideEntry],
    *,
    samples: Sequence[str] | None = None,
    peptide_ranges: Sequence[str] | None = None,
) -> list[PeptideEntry]:
    """Keep entries whose sample and peptide range are listed (exact match; None keeps all)."""
    kept = [
        e
        for e in entries
        if (samples is None or e.sample in samples)
        and (peptide_ranges is None or e.peptide_range in peptide_ranges)
    ]
    return kept


def sort_entries(entries: Iterable[PeptideEntry]) -> list[PeptideEntry]:
    """Order by peptide range, sample, charge, as pyHXExpress does."""
    return sorted(entries, key=lambda e: (e.peptide_range, e.sample, e.charge))
