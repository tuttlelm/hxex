# SPDX-License-Identifier: GPL-3.0-or-later
"""Pooled SpecExport folders: peptides from more than one protein, whose identities are not
known in advance (a pooled FASTA lists what may be there: :func:`match_candidates`).

Two layouts, told apart by what ``DATA_DIR`` holds (:func:`pooled_layout`)::

    DATA_DIR/START-END-SEQUENCE/TIME-REP-zCHARGE.csv          one sample, named by ``sample``
    DATA_DIR/STATE/START-END-SEQUENCE/TIME-REP-zCHARGE.csv    each state folder a sample

The second is HDExaminer's export of states, and also several pooled folders (one per
condition) side by side under one parent. There is no FASTA to cut sequences from (a residue
range is ambiguous when several proteins share the folder), so the sequence is taken from the
peptide folder's name and may carry modX labels such as ``acK``. With state folders an entry's
``file`` is ``STATE/START-END-SEQUENCE``, so its spectrum ids name the state as SpecExport's do.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ..chem import DEFAULT_MODIFICATIONS, ModificationRegistry, SequenceError, parse_sequence
from ..spectra import PeptideEntry, SpectrumSet
from .entries import filter_entries, read_fasta_records, sort_entries
from .errors import ReadError
from .specexport import _holds_peptide_folders, parse_specexport_filename

log = logging.getLogger(__name__)

_POOLED_DIR = re.compile(r"(?P<start>\d+)-(?P<end>\d+)-(?P<seq>.+)")

LAYOUTS = ("peptides", "states")


def pooled_layout(data_dir: str | Path) -> str:
    """``'peptides'`` when ``data_dir`` holds peptide folders (``START-END-SEQUENCE``),
    ``'states'`` when it holds none but holds folders that do (each a sample), ``''`` when it
    holds neither. Peptide folders decide: a folder holding both is read as peptide folders,
    and :func:`scan_specexport_pooled` names the folders of peptide folders it leaves."""
    root = Path(data_dir)
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if any(_POOLED_DIR.fullmatch(p.name) for p in dirs):
        return "peptides"
    return "states" if any(_holds_peptide_folders(p) for p in dirs) else ""


def _state_like(root: Path) -> list[str]:
    """The folders of ``root`` that hold peptide folders (not peptide folders themselves)."""
    return sorted(p.name for p in root.iterdir() if p.is_dir()
                  and not _POOLED_DIR.fullmatch(p.name) and _holds_peptide_folders(p))


def state_folders(data_dir: str | Path) -> list[str]:
    """The state folders of a pooled ``data_dir`` (sorted names; empty for the other layouts)."""
    root = Path(data_dir)
    return _state_like(root) if pooled_layout(root) == "states" else []


def scan_specexport_pooled(
    data_dir: str | Path,
    *,
    sample: str = "",
    samples: Sequence[str] | None = None,
    peptide_ranges: Sequence[str] | None = None,
    peptide_region: Sequence[int] | None = None,
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
) -> list[PeptideEntry]:
    """List the entries (peptide range, sequence, charge) in a pooled SpecExport folder.

    Peptide folders in ``data_dir``: every entry gets the sample ``sample`` (default empty).
    State folders (:func:`pooled_layout`): each state folder is a sample of that name, its
    entries' ``file`` is ``STATE/START-END-SEQUENCE``, and a ``sample`` given is not used
    (a warning says so); ``samples`` selects states. Directories not named
    ``START-END-SEQUENCE`` are skipped, as are those whose sequence does not parse. A sequence
    whose residue count differs from the range length is kept with a warning, and so are what
    will not be read: peptide folders without SpecExport files, and a folder with neither
    peptide folders nor state folders.
    """
    root = Path(data_dir)
    layout = pooled_layout(root)
    entries, empty = [], []
    if layout == "states":
        states = state_folders(root)
        if sample:
            log.warning("%s holds state folders (%s): each is read as a sample of its own, and "
                        "input.sample = %r is not used", root,
                        ", ".join(states[:3]) + (", ..." if len(states) > 3 else ""), sample)
        for state in states:
            if samples is not None and state not in samples:
                continue
            found, none = _scan_peptide_folders(root / state, state, f"{state}/", registry)
            entries += found
            empty += none
    elif layout == "peptides":
        entries, empty = _scan_peptide_folders(root, sample, "", registry)
        left = _state_like(root)
        if left:
            log.warning("%s holds peptide folders, read as one sample, and also folders of "
                        "peptide folders that are not read (%s): to read those as states, move "
                        "the peptide folders into a folder of their own", root,
                        ", ".join(left[:3]) + (", ..." if len(left) > 3 else ""))
    _pooled_warnings(root, layout, empty)
    return sort_entries(filter_entries(entries, samples=samples, peptide_ranges=peptide_ranges,
                                       peptide_region=peptide_region))


def _scan_peptide_folders(folder: Path, sample: str, prefix: str,
                          registry: ModificationRegistry) -> tuple[list[PeptideEntry], list[str]]:
    """The entries of one folder of peptide folders, and the peptide folders without
    SpecExport files (named with ``prefix``)."""
    entries, empty = [], []
    for spec_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        m = _POOLED_DIR.fullmatch(spec_dir.name)
        if m is None:
            log.debug("skipping %s: not a START-END-SEQUENCE directory", spec_dir)
            continue
        name = prefix + spec_dir.name
        try:
            residues = parse_sequence(m["seq"], registry)
        except SequenceError as exc:
            log.warning("skipping %s: %s", name, exc)
            continue
        start, end = int(m["start"]), int(m["end"])
        if len(residues) != end - start + 1:
            log.warning(
                "%s: %d residues but the range %d-%d has %d",
                name, len(residues), start, end, end - start + 1,
            )
        charges = {
            parsed[2]
            for f in spec_dir.glob("*.csv")
            if (parsed := parse_specexport_filename(f.name))
        }
        if not charges:
            empty.append(name)
        for charge in sorted(charges):
            entries.append(
                PeptideEntry(
                    file=name,
                    sample=sample,
                    peptide=m["seq"],
                    peptide_range=f"{m['start']}-{m['end']}",
                    start_seq=start,
                    end_seq=end,
                    charge=charge,
                )
            )
    return entries, empty


def _pooled_warnings(root: Path, layout: str, empty: list[str]) -> None:
    """Warn about what a pooled scan will not read: peptide folders without SpecExport files
    (``TIME-...-REP-zN.csv``), and a folder with neither peptide folders nor state folders."""
    if empty:
        log.warning("%s: %d peptide folder(s) hold no SpecExport files (TIME-...-REP-zN.csv) "
                    "and are not read: %s", root, len(empty),
                    ", ".join(empty[:3]) + (", ..." if len(empty) > 3 else ""))
    if not layout:
        log.warning("%s holds no peptide folders (START-END-SEQUENCE), and no state folders "
                    "holding them: nothing is read", root)


def nest_table_entries(entries: Sequence[PeptideEntry], data_dir: str | Path
                       ) -> list[PeptideEntry]:
    """A pooled peptide table's entries with their folders found: with state folders
    (:func:`pooled_layout`), a ``file`` naming only the peptide folder is read from the state
    folder its ``sample`` names (``STATE/START-END-SEQUENCE``), as a SpecExport table is; a
    ``file`` that already names its state folder, and every entry of a folder of peptide
    folders, stay as they are."""
    root = Path(data_dir)
    if not root.is_dir() or pooled_layout(root) != "states":
        return list(entries)
    return [e if "/" in e.file else replace(e, file=f"{e.sample}/{e.file}") for e in entries]


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
    ``'ALPHA'``: each one that appears in the text of any matching record id is reported, in the
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


# --- possible identities (docs/ANALYSIS.md "Pooled samples: possible identities") ----------

CANDIDATES_RECORD = "pooled_candidates"  # the key of the candidates in a read record
LEVELS = ("isoform", "class", "ambiguous", "none")


def candidates_record(pooled_fasta: str | Path,
                      classes: Sequence[str] | Mapping[str, Sequence[str]] = ()) -> dict:
    """What may be in a pooled sample, as a read record keeps it: the FASTA file's name, its
    records (``[id, sequence]``, in file order), the class names, and each class's record ids
    (``class_members``). ``classes`` is a list of class names, each the text found in its
    records' ids (pyHXExpress's rule), or a table naming each class's record ids (owner,
    2026-11-09: a class name need not be in the record names). Refused (:class:`ReadError`)
    when the file holds no sequences, a record id repeats (an identity would not name one
    record), a class name is empty or given twice, or a table names a record the FASTA does
    not hold or a class with no record. A class name found in no record id is a warning."""
    records = read_fasta_records(pooled_fasta)
    if not records:
        raise ReadError(f"input.pooled_fasta {pooled_fasta}: no sequences found")
    ids = [name for name, _ in records]
    twice = sorted({x for x in ids if ids.count(x) > 1})
    if twice:
        raise ReadError(f"input.pooled_fasta {pooled_fasta}: record id(s) given twice: "
                        f"{', '.join(twice)} (each candidate must have its own id, the first "
                        "word of its header)")
    if isinstance(classes, Mapping):
        names = [str(c) for c in classes]
        members = {str(c): [str(m) for m in ([v] if isinstance(v, str) else v)]
                   for c, v in classes.items()}
    else:
        names = [str(c) for c in classes]
        members = {c: [i for i in ids if c in i] for c in names}
    if any(not c.strip() for c in names) or len(set(names)) != len(names):
        raise ReadError("input.pooled_classes: each class name must be given once, and not "
                        "empty")
    if isinstance(classes, Mapping):
        unknown = sorted({m for ms in members.values() for m in ms} - set(ids))
        if unknown:
            raise ReadError(f"input.pooled_classes names record(s) {', '.join(unknown)} that "
                            f"{Path(pooled_fasta).name} does not hold (a record is named by "
                            "the first word of its > line)")
        empty = [c for c, ms in members.items() if not ms]
        if empty:
            raise ReadError(f"input.pooled_classes: class(es) {', '.join(empty)} name no "
                            "record")
    else:
        for c, ms in members.items():
            if not ms:
                log.warning("input.pooled_classes: %r is in no record id of %s, so no "
                            "candidate is of that class (to name a class's records, use "
                            "[input.pooled_classes] %s = [\"ID\", ...])", c,
                            Path(pooled_fasta).name, c)
    return {"file": Path(pooled_fasta).name,
            "records": [[name, seq.upper()] for name, seq in records],
            "classes": names, "class_members": members}


def record_classes(record: dict, name: str) -> set[str]:
    """The classes of the candidate ``name`` in a candidates ``record``
    (:func:`candidates_record`); a record kept before class members were stored names its
    classes by the text in the ids."""
    members = record.get("class_members")
    if members is not None:
        return {c for c, ms in members.items() if name in ms}
    return {c for c in record.get("classes") or [] if c in name}


def with_candidates(spectra: SpectrumSet, pooled_fasta: str | Path,
                    classes: Sequence[str] | Mapping[str, Sequence[str]] = ()) -> SpectrumSet:
    """``spectra`` whose read record also holds the candidates of :func:`candidates_record`,
    so the results carry what their peptides' identities were worked out from."""
    record = candidates_record(pooled_fasta, classes)
    return replace(spectra, source={**(spectra.source or {}), CANDIDATES_RECORD: record})


@dataclass(frozen=True)
class PeptideIdentity:
    """What a pooled peptide may be. ``candidates``: the record ids whose sequence holds it
    (modifications aside), in FASTA order; ``classes``: the class names among them;
    ``identity``: the one record, or the one class every candidate falls in, or the classes
    joined by ``;`` when the candidates fall in several, or empty; ``level``: ``isoform``,
    ``class``, ``ambiguous`` (several candidates not all of one class) or ``none``;
    ``offsets``: for each candidate, where the peptide starts in it minus the peptide's first
    residue number (0: the folder's numbering holds there; the occurrence nearest that
    number when it occurs more than once)."""

    candidates: tuple[str, ...]
    classes: tuple[str, ...]
    identity: str
    level: str
    offsets: tuple[int, ...]

    @property
    def offset_text(self) -> str:
        """The candidates whose numbering differs, as ``id:+n`` joined by ``;`` (empty when
        every candidate agrees)."""
        return ";".join(f"{c}:{o:+d}" for c, o in zip(self.candidates, self.offsets,
                                                      strict=True) if o != 0)


def identify(peptide: str, start: int, record: dict, *,
             registry: ModificationRegistry = DEFAULT_MODIFICATIONS) -> PeptideIdentity:
    """The identity of one peptide (its sequence, modX labels and terminal groups allowed;
    residues tokenised, never sliced by character) among the candidates of ``record``
    (:func:`candidates_record`)."""
    plain = "".join(registry.base_residue(r) for r in parse_sequence(peptide, registry))
    classes = list(record.get("classes") or [])
    found, offsets = [], []
    for name, seq in record["records"]:
        at = _occurrences(seq, plain)
        if at:
            found.append(name)
            offsets.append(min((p + 1 - int(start) for p in at), key=lambda o: (abs(o), o)))
    per = [record_classes(record, name) for name in found]
    among = tuple(c for c in classes if any(c in got for got in per))
    if not found:
        identity, level = "", "none"
    elif len(found) == 1:
        identity, level = found[0], "isoform"
    elif len(among) == 1 and all(got == {among[0]} for got in per):
        identity, level = among[0], "class"
    else:
        identity, level = ";".join(among), "ambiguous"
    return PeptideIdentity(tuple(found), among, identity, level, tuple(offsets))


def _occurrences(sequence: str, part: str) -> list[int]:
    """Every 0-based position where ``part`` starts in ``sequence`` (overlaps included)."""
    out, at = [], sequence.find(part)
    while part and at >= 0:
        out.append(at)
        at = sequence.find(part, at + 1)
    return out


def identities(source: dict | None, peptides: Iterable[tuple[str, int]], *,
               registry: ModificationRegistry = DEFAULT_MODIFICATIONS
               ) -> dict[tuple[str, int], PeptideIdentity] | None:
    """The identity of each ``(peptide, first residue)`` from the candidates a read record
    holds (:data:`CANDIDATES_RECORD`); None when it holds none (not pooled, or no
    ``pooled_fasta``). A sequence that does not parse is left out."""
    record = (source or {}).get(CANDIDATES_RECORD)
    if not record:
        return None
    out = {}
    for peptide, start in set(peptides):
        try:
            out[(peptide, start)] = identify(peptide, start, record, registry=registry)
        except SequenceError:
            continue
    return out
