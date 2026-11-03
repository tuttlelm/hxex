# SPDX-License-Identifier: GPL-3.0-or-later
"""HDExaminer SpecExport data.

Layout under the data directory::

    SAMPLE.fasta
    SAMPLE/START-END-SEQUENCE/TIME-REP-zCHARGE.csv      e.g. 60s-4-z3.csv, Non-D-1-z3.csv

``TIME`` is ``Non`` (undeuterated), ``Full`` (fully deuterated) or a number plus ``s``, ``m`` or
``h``. A file has no header and two columns, m/z and intensity. The peptide sequence is cut
from the sample's FASTA file using the residue range in the directory name, and checked against
the sequence the name carries. The FASTA's first residue is residue 1 unless its header numbers
it, Pfam-style: ``>NAME/START-END`` (a tag before residue 1: ``>WT/-3-150``).
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..chem.modifications import (
    DEFAULT_MODIFICATIONS,
    ModificationRegistry,
    SequenceError,
    join_termini,
    parse_sequence,
    split_termini,
)
from ..spectra import PeptideEntry, Spectrum, point_order_key, time_point
from ._common import clean_spectrum, not_text_error, open_text, to_number
from .entries import (
    fasta_name,
    filter_entries,
    read_fasta_numbered,
    read_fasta_records,
    sort_entries,
)
from .errors import ReadError
from .times import format_time, parse_specexport_token

log = logging.getLogger(__name__)

_FILE = re.compile(r"(?P<time>[^-]+)(?:-[^-]+)*?-(?P<rep>\d+)-z(?P<charge>\d+)\.csv")
_RANGE_DIR = re.compile(r"(?P<start>\d+)-(?P<end>\d+)-(?P<seq>.*)")


def parse_specexport_filename(name: str) -> tuple[str, int, int] | None:
    """``(time token, replicate, charge)`` from a SpecExport file name, or None if it is not one."""
    m = _FILE.fullmatch(name)
    return (m["time"], int(m["rep"]), int(m["charge"])) if m else None


def scan_specexport(
    data_dir: str | Path,
    *,
    samples: Sequence[str] | None = None,
    peptide_ranges: Sequence[str] | None = None,
    peptide_region: Sequence[int] | None = None,
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
    warn: bool = True,
) -> list[PeptideEntry]:
    """List the entries (sample, peptide range, charge) found under ``data_dir``.

    A sample needs both ``SAMPLE.fasta`` and a ``SAMPLE`` directory; folders and FASTA files
    that do not pair up are named in warnings (:func:`_pairing_warnings`). Directories that are
    not named ``START-END-SEQUENCE`` are skipped. The peptide is cut from the FASTA sequence by
    residues (modX: ``acK`` is one residue, of the ``registry``), keeping a terminal group of
    the sequence on a peptide at that end, and numbered as the FASTA's header says
    (:func:`~hxex.io.entries.read_fasta_numbered`; residue 1 first without one).

    Each folder's own sequence (the text after ``START-END-``) is compared with the cut, by
    base residues (modX labels and terminal groups aside): folders that disagree, or whose range
    lies outside the FASTA, are refused together in one :class:`ReadError`, which suggests the
    header line that numbers the FASTA as the folders are when every folder's sequence is found
    in it at one numbering. A folder whose sequence cannot be read is not checked (a warning).
    The entries keep the folders' residue numbers. ``warn`` False logs none of the warnings
    (for a second scan of a folder already read).
    """
    warning = log.warning if warn else _silent
    root = Path(data_dir)
    fasta = {p.stem: p for p in root.glob("*.fasta")}
    if not fasta:
        warning("no .fasta files found in %s", root)
    dirs = {p.name: p for p in root.iterdir() if p.is_dir()}
    if warn:
        _pairing_warnings(root, fasta, dirs, samples)
    found = sorted(set(fasta) & set(dirs))
    if samples is not None:
        for missing in sorted(set(samples) - set(found)):
            warning("samples: %s is not a sample in %s (%s)", missing, root,
                        _samples_found(found, missing))
        found = [s for s in found if s in samples]
    entries = []
    problems: list[str] = []
    fixes: list[str] = []
    # sample -> (FASTA, the folders that differ, {residue: (name's, FASTA's)}) where the folder
    # names differ from the FASTA at a few residues (:func:`_substitutions`)
    substituted: dict[str, tuple[Path, int, dict[int, tuple[str, str]]]] = {}
    for sample in found:
        path = fasta[sample]
        sequence, first = read_fasta_numbered(path)
        try:
            nterm, _, cterm = split_termini(sequence)
            residues = parse_sequence(sequence, registry)
        except SequenceError as exc:  # a letter outside the registry (X, say): cut by letters
            warning("%s: %s; the sequence is cut by its letters, one residue per letter "
                        "(a modX label such as acK in it would count as three)", path, exc)
            nterm, cterm, residues = "H-", "-OH", tuple(sequence)
        last = first + len(residues) - 1
        plain = "".join(registry.base_residue(r) for r in residues)
        named: list[tuple[int, str]] = []  # (first residue, base residues) of each folder name
        differ: list[tuple[str, int, str, str]] = []  # (folder, first residue, name, FASTA)
        covered: set[int] = set()  # the residues whose folder names were compared
        before = len(problems)
        for spec_dir in sorted(p for p in (root / sample).iterdir() if p.is_dir()):
            m = _RANGE_DIR.fullmatch(spec_dir.name)
            if m is None:
                log.debug("skipping %s: not a START-END-SEQUENCE directory", spec_dir)
                continue
            start, end = int(m["start"]), int(m["end"])
            own = _base_residues(m["seq"], registry)
            if isinstance(own, SequenceError):
                warning("%s: the sequence in the folder's name cannot be read (%s), so it "
                            "is not checked against %s", spec_dir, own, path.name)
            else:
                named.append((start, own))
            where = f"{sample}/{spec_dir.name}"
            if not first <= start <= end <= last:
                problems.append(f"{where}: residues {start}-{end} are outside {path.name} "
                                f"(residues {first}-{last})")
                continue
            cut = plain[start - first:end - first + 1]
            if not isinstance(own, SequenceError):
                covered.update(range(start, end + 1))
                if own != cut:
                    differ.append((where, start, own, cut))
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
                        peptide=join_termini(nterm if start == first else "H-",
                                             residues[start - first:end - first + 1],
                                             cterm if end == last else "-OH"),
                        peptide_range=f"{m['start']}-{m['end']}",
                        start_seq=start,
                        end_seq=end,
                        charge=charge,
                    )
                )
        if differ:
            changes = _substitutions(differ, len(covered))
            if changes is None:
                problems += [f"{folder}: the folder names {name}, {path.name} gives {given} "
                             f"at residues {at}-{at + len(given) - 1}"
                             for folder, at, name, given in differ]
            else:
                substituted[sample] = (path, differ, changes)
        if len(problems) > before:
            fixes.append(_numbering_fix(path, sample, plain, named))
    if not problems:
        if warn:
            _substitution_warnings(substituted, len(found))
    if problems:
        shown = problems[:_SHOWN]
        if len(problems) > _SHOWN:
            shown.append(f"and {len(problems) - _SHOWN} more")
        raise ReadError(
            f"{len(problems)} peptide folder(s) do not match their sample's FASTA file, so "
            "their sequences (and m/z) would be wrong:\n  " + "\n  ".join(shown + fixes))
    return sort_entries(filter_entries(entries, samples=samples, peptide_ranges=peptide_ranges,
                                       peptide_region=peptide_region))


_SHOWN = 10  # the folders a refusal names before "and N more"


def _base_residues(text: str, registry: ModificationRegistry) -> str | SequenceError:
    """The base residues of a sequence (modX labels as their residue, terminal groups left
    out), or the :class:`SequenceError` that says why it cannot be read."""
    try:
        return "".join(registry.base_residue(r) for r in parse_sequence(text, registry))
    except SequenceError as exc:
        return exc


# a folder name differs from the FASTA by substitutions (a mutant's folders named from the
# reference sequence: seen in every mutant of the real SpecExport data, one to three residues of
# 100 to 200 covered) rather than by a shift or another sequence (which leaves few residues in
# common) when no more than this share of the residues covered differ
_SUBSTITUTED_SHARE = 1 / 3


def _silent(*args, **kwargs) -> None:
    """A warning not logged."""


def _substitutions(differ: Sequence[tuple[str, int, str, str]],
                   covered: int) -> dict[int, tuple[str, str]] | None:
    """The residues at which the folder names (``differ``: folder, first residue, its name's
    base residues, the FASTA's) differ from the FASTA, ``{residue: (name's, FASTA's)}``, when
    they are substitutions: a few residues of the ``covered`` ones (:data:`_SUBSTITUTED_SHARE`),
    the names as long as the ranges. None otherwise (another numbering or another sequence)."""
    changes: dict[int, tuple[str, str]] = {}
    for _, start, own, cut in differ:
        if len(own) != len(cut):
            return None
        changes.update({start + i: (a, b) for i, (a, b) in enumerate(zip(own, cut, strict=True))
                        if a != b})
    return changes if len(changes) <= _SUBSTITUTED_SHARE * covered else None


def _substitution_warnings(substituted: Mapping[str, tuple], n_samples: int) -> None:
    """Warn where folder names differ from their FASTA at a few residues (``substituted``, by
    sample, :func:`scan_specexport`). A difference shared by every sample (of ``n_samples``,
    more than one) is not a mutation: the FASTA files and the sequence HDExaminer searched
    against disagree there. One in some samples only is a mutant's substitution: HDExaminer
    names a mutant's folders from the reference sequence. With one sample the two cannot be
    told apart, and the warning says both."""
    shared: dict[int, tuple[str, str]] = {}
    if n_samples > 1 and len(substituted) == n_samples:
        changes = [c for _, _, c in substituted.values()]
        shared = {n: ab for n, ab in changes[0].items() if all(c.get(n) == ab for c in changes)}
    if shared:
        log.warning("every sample's FASTA (%s) differs from the peptide folder names at %s: the "
                    "FASTA files and the sequence HDExaminer searched against disagree there "
                    "(another variant, or an error in one of them), and a mutation would be "
                    "in its own sample only; the FASTA's residues are used: check which is "
                    "right", ", ".join(p.name for p, _, _ in substituted.values()),
                    _changes_text(shared))
    for path, differ, changes in substituted.values():
        own = {n: ab for n, ab in changes.items() if n not in shared}
        if not own:
            continue
        n_folders = sum(any(at + i in own for i, (a, b)
                            in enumerate(zip(name, given, strict=True)) if a != b)
                        for _, at, name, given in differ)
        cause = ("as a mutant's FASTA does when HDExaminer named the folders from the reference "
                 "sequence" if n_samples > 1 else "as a mutant's FASTA does when HDExaminer "
                 "named the folders from the reference sequence, or a FASTA that differs from "
                 "the sequence HDExaminer searched against")
        log.warning("%s differs from the names of %d of its peptide folders at %s, %s: the "
                    "FASTA's residues are used (check that these are the sample's "
                    "substitutions)", path, n_folders, _changes_text(own), cause)


def _changes_text(changes: dict[int, tuple[str, str]], shown: int = 10) -> str:
    """``residue 109 (D in the folder names, H in the FASTA)``, for each residue (``shown`` of
    them at most)."""
    items = sorted(changes.items())
    text = ", ".join(f"{n} ({a} in the folder names, {b} in the FASTA)" for n, (a, b)
                     in items[:shown])
    more = f" and {len(items) - shown} more" if len(items) > shown else ""
    return f"residue{'s' if len(items) > 1 else ''} {text}{more}"


def _numbering_fix(path: Path, sample: str, plain: str, named: Sequence[tuple[int, str]]) -> str:
    """What to do about a FASTA whose folders disagree with it: the header line that numbers it
    as the folders are, when every folder's sequence (``named``: first residue, base residues)
    is found in it at one numbering; otherwise, that it is another construct or a mutant."""
    numberings: set[int] | None = None
    for start, own in named:
        here = set()
        at = plain.find(own)
        while at >= 0:
            here.add(start - at)  # the number of the FASTA's first residue
            at = plain.find(own, at + 1)
        numberings = here if numberings is None else numberings & here
    if numberings is not None and len(numberings) == 1:
        (first,) = numberings
        records = read_fasta_records(path)
        name = fasta_name(records[-1][0]) if records else ""
        line = f">{name or sample}/{first}-{first + len(plain) - 1}"
        which = " of its last record" if len(records) > 1 else ""
        return (f"{path}: the FASTA's numbering differs from HDExaminer's (every folder's "
                f"sequence is found in it with its first residue numbered {first}, a tag or "
                f"a signal peptide, say): number it as the folders are by making {line} the "
                f"header line{which}")
    return (f"{path}: the FASTA's numbering differs from HDExaminer's, or it is another "
            "construct or a mutant (the folders' sequences are not all found in it at one "
            f"numbering): give sample {sample} the FASTA its data were searched against")


def _samples_found(found: Sequence[str], name: str) -> str:
    """The samples found, for a message about ``name``; with the case hint when one of them
    differs from it only in case."""
    text = f"the samples found are {', '.join(found)}" if found else "no samples were found"
    if any(s.lower() == name.lower() for s in found):
        text += "; names are case-sensitive"
    return text


def _holds_peptide_folders(folder: Path) -> bool:
    try:
        return any(p.is_dir() and _RANGE_DIR.fullmatch(p.name) for p in folder.iterdir())
    except OSError:
        return False


def _pairing_warnings(root: Path, fasta: dict[str, Path], dirs: dict[str, Path],
                      samples: Sequence[str] | None) -> None:
    """Warn about what will not be read: peptide folders straight in ``root`` (a sample folder
    or a pooled folder given as ``data_dir``), a sample folder (one that holds ``START-END-*``
    folders) without its FASTA beside it, and a FASTA without its folder; names that match only
    without regard to case are pointed out. With ``samples``, only their names (in any case)."""
    wanted = None if samples is None else {s.lower() for s in samples}

    def asked(name: str) -> bool:
        return wanted is None or name.lower() in wanted

    if any(_RANGE_DIR.fullmatch(name) for name in dirs):
        log.warning('%s holds peptide folders (START-END-SEQUENCE) itself: this looks like a '
                    'sample folder or a pooled folder: give the folder above it as data_dir, '
                    'or format = "pooled"', root)
    by_case = {name.lower(): name for name in fasta}
    told: set[str] = set()  # FASTA files already named beside a folder
    for name, folder in sorted(dirs.items()):
        if name in fasta or not asked(name) or not _holds_peptide_folders(folder):
            continue
        other = by_case.get(name.lower())
        inside = sorted(folder.glob("*.fasta"))
        if other is not None:
            told.add(other)
            log.warning("%s: sample folder %s has no %s.fasta beside it, so it is not read "
                        "(names are case-sensitive: %s.fasta and folder %s); rename one",
                        root, name, name, other, name)
        elif inside:
            log.warning("%s: a sample's FASTA file must sit beside its folder, as %s, not "
                        "inside it; sample folder %s is not read", inside[0],
                        root / f"{name}.fasta", name)
        else:
            log.warning("%s: sample folder %s has no %s.fasta beside it, so it is not read "
                        "(add the FASTA file of its sequence)", root, name, name)
    folders_by_case = {name.lower(): name for name in dirs}
    for name, path in sorted(fasta.items()):
        if name in dirs or name in told or not asked(name):
            continue
        other = folders_by_case.get(name.lower())
        if other is not None:
            log.warning("%s has no folder %s beside it, so it is not used (names are "
                        "case-sensitive: %s.fasta and folder %s); rename one", path, name,
                        name, other)
        else:
            log.warning("%s has no folder %s beside it, so it is not used", path, name)


def _read_two_columns(path: Path) -> tuple[list[float | None], list[float | None]]:
    mz: list[float | None] = []
    intensity: list[float | None] = []
    try:
        with open_text(path, newline="") as fh:
            for line, row in enumerate(csv.reader(fh), start=1):
                if not row:
                    continue
                if len(row) < 2:
                    raise ReadError(f"{path}, line {line}: expected two columns (m/z and "
                                    f"intensity, separated by a comma){_separator_hint(row[0])}")
                where = f"{path.name} line {line}"
                mz.append(to_number(row[0], where))
                intensity.append(to_number(row[1], where))
    except (OSError, UnicodeDecodeError) as exc:
        raise not_text_error(path, exc) from exc
    return mz, intensity


def _separator_hint(cell: str) -> str:
    """Why a line has one cell: its values are separated by semicolons or tabs."""
    if ";" in cell:
        return "; the values are separated by ';' (Excel's regional setting), not commas"
    if "\t" in cell:
        return "; the values are separated by tabs, not commas"
    return ""


def specexport_points(spec_dir: str | Path, entry: PeptideEntry) -> list[tuple]:
    """The spectrum files of ``entry``'s charge in one peptide directory, as
    ``(time, role, rep, path)`` in the order :func:`read_specexport_dir` reads them, from the
    file names alone (no file is opened). Raises :class:`ReadError` as the read does: no file
    for the charge, or two files for one point and replicate."""
    spec_dir = Path(spec_dir)
    found = []
    for path in spec_dir.glob("*.csv"):
        parsed = parse_specexport_filename(path.name)
        if parsed is not None and parsed[2] == entry.charge:
            try:
                time, role = parse_specexport_token(parsed[0])
            except ReadError as exc:
                raise ReadError(f"{path}: {exc}") from None
            found.append((time, role, parsed[1], path))
    if not found:
        present = sorted({parsed[2] for path in spec_dir.glob("*.csv")
                          if (parsed := parse_specexport_filename(path.name))})
        if present:
            note = (f"files found for charge{'s' if len(present) > 1 else ''} "
                    f"{', '.join(map(str, present))}")
        else:
            note = "no spectrum files of any charge"
        raise ReadError(f"{spec_dir}: no spectrum files for charge {entry.charge} ({note})")
    # two files for one point and replicate (``60s-1`` and ``1m-1``) cannot both be that
    # spectrum
    seen: dict[tuple[str, int], Path] = {}
    for time, role, rep, path in sorted(found, key=lambda t: t[3].name):
        key = (str(time_point([time], [role])[0]), rep)
        other = seen.setdefault(key, path)
        if other is not path:
            raise ReadError(
                f"{spec_dir}: {other.name} and {path.name} are both replicate {rep} at "
                f"{role or format_time(time)} for charge {entry.charge}; rename or remove one")
    return sorted(found, key=lambda t: _point_order(t))


def read_specexport_dir(
    spec_dir: str | Path,
    entry: PeptideEntry,
    *,
    data_id: int = 0,
    id_prefix: str | None = None,
) -> list[Spectrum]:
    """Read every spectrum of ``entry``'s charge in one peptide directory.

    ``Non`` files are the undeuterated controls and ``Full`` files the fully deuterated ones
    (the role set, no time; :func:`parse_specexport_token`). Spectra are ordered by point
    (the undeuterated control, the times, the fully deuterated control), then replicate, then
    file name. ``spectrum_id`` is ``id_prefix/FILE``, with ``id_prefix`` defaulting to
    ``SAMPLE/PEPTIDE_DIR``.
    """
    if id_prefix is None:
        id_prefix = f"{entry.sample}/{entry.file}"
    spectra = []
    for time, role, rep, path in specexport_points(spec_dir, entry):
        mz, intensity = clean_spectrum(*_read_two_columns(path))
        spectra.append(
            Spectrum(
                mz, intensity, entry, time, rep,
                data_id=data_id, label=path.name, spectrum_id=f"{id_prefix}/{path.name}",
                role=role,
            )
        )
    return spectra


def _point_order(item) -> tuple:
    """Sort key: the undeuterated control, the times in order, the fully deuterated control;
    then replicate and file name."""
    time, role, rep, path = item
    return (*point_order_key(float(time), role), rep, path.name)
