# SPDX-License-Identifier: GPL-3.0-or-later
"""HDExaminer SpecExport data.

Layout under the data directory::

    SAMPLE.fasta
    SAMPLE/START-END-SEQUENCE/TIME-REP-zCHARGE.csv      e.g. 60s-4-z3.csv, Non-D-1-z3.csv

``TIME`` is ``Non`` (undeuterated), ``Full`` (fully deuterated) or a number plus ``s``, ``m`` or
``h``. A file has no header and two columns, m/z and intensity. The peptide sequence is cut
from the sample's FASTA file using the residue range in the directory name.
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Sequence
from pathlib import Path

from ..spectra import PeptideEntry, Spectrum, point_order_key, time_point
from ._common import clean_spectrum, to_number
from .entries import filter_entries, read_fasta_sequence, sort_entries
from .errors import ReadError
from .times import format_time, parse_specexport_token

log = logging.getLogger(__name__)

_FILE = re.compile(r"(?P<time>[^-]+)(?:-[^-]+)*?-(?P<rep>\d+)-z(?P<charge>\d+)\.csv")
_RANGE_DIR = re.compile(r"(?P<start>\d+)-(?P<end>\d+)-.*")


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
) -> list[PeptideEntry]:
    """List the entries (sample, peptide range, charge) found under ``data_dir``.

    A sample needs both ``SAMPLE.fasta`` and a ``SAMPLE`` directory. Directories that are not
    named ``START-END-SEQUENCE`` are skipped.
    """
    root = Path(data_dir)
    fasta = {p.stem: p for p in root.glob("*.fasta")}
    if not fasta:
        log.warning("no .fasta files found in %s", root)
    dirs = {p.name for p in root.iterdir() if p.is_dir()}
    found = sorted(set(fasta) & dirs)
    if samples is not None:
        for missing in sorted(set(samples) - set(found)):
            log.warning("no FASTA file and directory for sample %s", missing)
        found = [s for s in found if s in samples]
    entries = []
    for sample in found:
        sequence = read_fasta_sequence(fasta[sample])
        for spec_dir in sorted(p for p in (root / sample).iterdir() if p.is_dir()):
            m = _RANGE_DIR.fullmatch(spec_dir.name)
            if m is None:
                log.debug("skipping %s: not a START-END-SEQUENCE directory", spec_dir)
                continue
            start, end = int(m["start"]), int(m["end"])
            if not 1 <= start <= end <= len(sequence):
                raise ReadError(f"{spec_dir}: range {start}-{end} is outside {sample}.fasta")
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
                        peptide=sequence[start - 1 : end],
                        peptide_range=f"{m['start']}-{m['end']}",
                        start_seq=start,
                        end_seq=end,
                        charge=charge,
                    )
                )
    return sort_entries(filter_entries(entries, samples=samples, peptide_ranges=peptide_ranges,
                                       peptide_region=peptide_region))


def _read_two_columns(path: Path) -> tuple[list[float | None], list[float | None]]:
    mz: list[float | None] = []
    intensity: list[float | None] = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for line, row in enumerate(csv.reader(fh), start=1):
                if not row:
                    continue
                if len(row) < 2:
                    raise ReadError(f"{path}, line {line}: expected two columns")
                where = f"{path.name} line {line}"
                mz.append(to_number(row[0], where))
                intensity.append(to_number(row[1], where))
    except (OSError, UnicodeDecodeError) as exc:
        raise ReadError(f"cannot read {path}: {exc}") from exc
    return mz, intensity


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
    spec_dir = Path(spec_dir)
    found = []
    for path in spec_dir.glob("*.csv"):
        parsed = parse_specexport_filename(path.name)
        if parsed is not None and parsed[2] == entry.charge:
            time, role = parse_specexport_token(parsed[0])
            found.append((time, role, parsed[1], path))
    if not found:
        raise ReadError(f"{spec_dir}: no spectrum files for charge {entry.charge}")
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
    spectra = []
    for time, role, rep, path in sorted(found, key=lambda t: _point_order(t)):
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
