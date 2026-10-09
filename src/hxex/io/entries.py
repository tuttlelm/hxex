# SPDX-License-Identifier: GPL-3.0-or-later
"""Peptide tables: the metadata CSV (pyHXExpress's metadf), FASTA files and entry filters."""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from ..spectra import PeptideEntry
from .errors import ReadError

log = logging.getLogger(__name__)

_REQUIRED = ("file", "sample", "peptide", "peptide_range", "start_seq", "end_seq", "charge")


def read_peptide_table(path: str | Path) -> list[PeptideEntry]:
    """Read a metadata CSV (columns as in pyHXExpress's ``metadf``).

    Required: file, sample, peptide, peptide_range, start_seq, end_seq, charge. Optional:
    modification. The ``Index`` column is ignored and any other column is kept in ``extra``.
    Values are text (``peptide_range`` keeps its leading zeros).
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
    except OSError as exc:
        raise ReadError(f"cannot read {path}: {exc}") from exc
    return entries


def _entry_from_row(row: dict[str, str]) -> PeptideEntry:
    known = set(_REQUIRED) | {"modification", "Index"}
    return PeptideEntry(
        file=row["file"],
        sample=row["sample"],
        peptide=row["peptide"],
        peptide_range=row["peptide_range"],
        start_seq=int(float(row["start_seq"])),
        end_seq=int(float(row["end_seq"])),
        charge=int(float(row["charge"])),
        modification=(row.get("modification") or "").strip(),
        extra={k: v for k, v in row.items() if k not in known and k is not None},
    )


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
    except OSError as exc:
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
