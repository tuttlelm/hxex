# SPDX-License-Identifier: GPL-3.0-or-later
"""Pooled SpecExport folders: peptides from more than one protein in a single data directory.

Layout::

    DATA_DIR/START-END-SEQUENCE/TIME-REP-zCHARGE.csv

There is no sample level and no FASTA file to cut sequences from (a residue range is ambiguous
when several proteins share the folder), so the sequence is taken from the directory name and
may carry modX labels such as ``acK``. :func:`match_candidates` then finds which proteins of a
pooled FASTA file contain each peptide.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..chem import DEFAULT_MODIFICATIONS, ModificationRegistry, SequenceError, parse_sequence
from ..spectra import PeptideEntry
from .entries import filter_entries, read_fasta_records, sort_entries
from .errors import ReadError
from .specexport import parse_specexport_filename

log = logging.getLogger(__name__)

_POOLED_DIR = re.compile(r"(?P<start>\d+)-(?P<end>\d+)-(?P<seq>.+)")


def scan_specexport_pooled(
    data_dir: str | Path,
    *,
    sample: str = "",
    peptide_ranges: Sequence[str] | None = None,
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
) -> list[PeptideEntry]:
    """List the entries (peptide range, sequence, charge) in a pooled SpecExport folder.

    Every entry gets the same ``sample`` label (default empty). Directories
    not named ``START-END-SEQUENCE`` are skipped, as are those whose sequence does not parse.
    A sequence whose residue count differs from the range length is kept with a warning.
    """
    entries = []
    for spec_dir in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        m = _POOLED_DIR.fullmatch(spec_dir.name)
        if m is None:
            log.debug("skipping %s: not a START-END-SEQUENCE directory", spec_dir)
            continue
        try:
            residues = parse_sequence(m["seq"], registry)
        except SequenceError as exc:
            log.warning("skipping %s: %s", spec_dir.name, exc)
            continue
        start, end = int(m["start"]), int(m["end"])
        if len(residues) != end - start + 1:
            log.warning(
                "%s: %d residues but the range %d-%d has %d",
                spec_dir.name, len(residues), start, end, end - start + 1,
            )
        charges = {
            parsed[2]
            for f in spec_dir.glob("*.csv")
            if (parsed := parse_specexport_filename(f.name))
        }
        for charge in sorted(charges):
            entries.append(
                PeptideEntry(
                    file=spec_dir.name,
                    sample=sample,
                    peptide=m["seq"],
                    peptide_range=f"{m['start']}-{m['end']}",
                    start_seq=start,
                    end_seq=end,
                    charge=charge,
                )
            )
    return sort_entries(filter_entries(entries, peptide_ranges=peptide_ranges))


@dataclass(frozen=True)
class CandidateMatch:
    """The pooled-FASTA records that contain an entry's peptide, and the classes they fall in."""

    entry: PeptideEntry
    matches: tuple[str, ...]
    classes: tuple[str, ...]


def match_candidates(
    entries: Iterable[PeptideEntry],
    pooled_fasta: str | Path,
    *,
    classes: Sequence[str] = (),
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
) -> list[CandidateMatch]:
    """For each entry, the FASTA record ids whose sequence contains its peptide.

    Modifications are ignored when matching (``acK`` matches ``K``); the residues are tokenised,
    not read as characters. ``matches`` follows FASTA order. ``classes`` are labels such as
    ``'TBA'``: each one that appears in the text of any matching record id is reported, in the
    order given.
    """
    records = read_fasta_records(pooled_fasta)
    if not records:
        raise ReadError(f"{pooled_fasta}: no sequences found")
    out = []
    for entry in entries:
        residues = parse_sequence(entry.peptide, registry)
        plain = "".join(registry.base_residue(r) for r in residues)
        matches = tuple(name for name, seq in records if plain in seq)
        found = tuple(c for c in classes if any(c in name for name in matches))
        out.append(CandidateMatch(entry, matches, found))
    return out
