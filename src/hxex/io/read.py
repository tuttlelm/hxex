# SPDX-License-Identifier: GPL-3.0-or-later
"""One entry point over the folder formats (HX-Express, SpecExport, pooled SpecExport), and the
repetition of a recorded read of any format (:func:`read_source`)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal

from ..chem import DEFAULT_MODIFICATIONS, ModificationRegistry
from ..hooks import CancelHook, ProgressHook, check_cancel, report
from ..spectra import PeptideEntry, Spectrum, SpectrumSet, check_role, qualify_spectrum_id
from .entries import filter_entries
from .errors import ReadError
from .hxexpress import read_hxexpress_file, scan_hxexpress
from .identity import CHANGE_ACTIONS, compare_identity, identity
from .pooled import scan_specexport_pooled
from .specexport import read_specexport_dir, scan_specexport

log = logging.getLogger(__name__)

Format = Literal["hxexpress", "specexport", "pooled"]


def scan(
    data_dir: str | Path,
    fmt: Format,
    *,
    samples: Sequence[str] | None = None,
    peptide_ranges: Sequence[str] | None = None,
    peptide_region: Sequence[int] | None = None,
    sample: str = "",
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
) -> list[PeptideEntry]:
    """List the entries in a data directory of the given format.

    ``fmt='pooled'`` is a SpecExport folder with peptides from several proteins; ``sample`` is
    the label given to its entries (the other formats take sample names from the data), and
    ``registry`` the modified residues its folder names may hold.
    """
    if fmt == "hxexpress":
        return scan_hxexpress(data_dir, samples=samples, peptide_ranges=peptide_ranges,
                              peptide_region=peptide_region)
    if fmt == "specexport":
        return scan_specexport(data_dir, samples=samples, peptide_ranges=peptide_ranges,
                               peptide_region=peptide_region, registry=registry)
    if fmt == "pooled":
        return scan_specexport_pooled(data_dir, sample=sample, peptide_ranges=peptide_ranges,
                                      peptide_region=peptide_region,
                                      registry=registry)
    raise ValueError(f"unknown format {fmt!r}; use 'hxexpress', 'specexport' or 'pooled'")


PATH_MODES = ("path", "name")


def read_spectra(
    entries: Sequence[PeptideEntry],
    data_dir: str | Path,
    fmt: Format,
    *,
    project: str,
    samples: Sequence[str] | None = None,
    peptide_ranges: Sequence[str] | None = None,
    peptide_region: Sequence[int] | None = None,
    record_paths: str = "path",
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> SpectrumSet:
    """Read the raw spectra of ``entries`` into one :class:`SpectrumSet`.

    ``project`` names the data set (non-empty, no ``:``); it prefixes every ``spectrum_id``
    (``PROJECT:LOCAL_ID``) so ids stay unique when sets from different directories are combined.

    ``data_id`` is each entry's position in ``entries`` (before filtering by ``samples`` and
    ``peptide_ranges``, so ids stay stable when a filter is applied). The control labels
    (``undeut``/``Non`` and ``TD``/``Full``) give their spectra the role ``undeut`` or
    ``fulldeut`` and no time (:mod:`hxex.io.times`). Raises
    :class:`~hxex.hooks.Cancelled` if ``cancel`` returns True between entries.

    The spectra's ``source`` records the read (the format, ``project``, the data directory, the
    entries and the filters) and the identity of every entry's input (its
    HX-Express file, or its SpecExport folder: :mod:`hxex.io.identity`), so that
    :func:`read_spectra_source` repeats it and notices a changed input. ``record_paths`` is
    ``'path'`` (the data directory as given) or ``'name'`` (its name only, for results to be
    shared; the read is then repeated with ``directory``).
    """
    if fmt not in ("hxexpress", "specexport", "pooled"):
        raise ValueError(f"unknown format {fmt!r}; use 'hxexpress', 'specexport' or 'pooled'")
    if record_paths not in PATH_MODES:
        raise ValueError(f"record_paths must be one of {', '.join(PATH_MODES)}")
    qualify_spectrum_id(project, "")  # validate the name before any reading
    root = Path(data_dir)
    return _read(entries, root, fmt, project=project, samples=samples,
                 peptide_ranges=peptide_ranges, peptide_region=peptide_region,
                 recorded_dir=str(root) if record_paths == "path" else root.name,
                 record_paths=record_paths, known=None, on_change="warn", progress=progress,
                 cancel=cancel)


def spectrum_index(
    entries: Sequence[PeptideEntry],
    data_dir: str | Path,
    fmt: Format,
    *,
    project: str,
    samples: Sequence[str] | None = None,
    peptide_ranges: Sequence[str] | None = None,
    peptide_region: Sequence[int] | None = None,
):
    """The spectra :func:`read_spectra` would make, without reading them
    (:class:`hxex.exclusions.SpectrumIndex`): SpecExport and pooled folders from their file
    names, HX-Express files from their first row. The same entries, filters, ids and order as
    the read; raises :class:`ReadError` where it would."""
    from ..exclusions import SpectrumIndex
    from .hxexpress import hxexpress_columns, read_hxexpress_header
    from .specexport import specexport_points

    root = Path(data_dir)
    kept = filter_entries(entries, samples=samples, peptide_ranges=peptide_ranges,
                          peptide_region=peptide_region)
    wanted = {id(e) for e in kept}
    shared = _shared_inputs(entries, fmt)
    rows = []
    for data_id, entry in enumerate(entries):
        if id(entry) not in wanted:
            continue
        rel = _input_path(fmt, entry)
        suffix = _id_suffix(fmt, entry, data_id, shared)
        if not (root / rel).exists():
            raise ReadError(f"{root / rel} not found (give the directory that holds the data)")
        common = {"sample": entry.sample, "start_seq": entry.start_seq,
                  "end_seq": entry.end_seq, "charge": entry.charge, "file": entry.file}
        if fmt == "hxexpress":
            for col, label, time, role, rep in hxexpress_columns(
                    read_hxexpress_header(root / rel), root / rel):
                rows.append({**common, "time": _time(time), "role": check_role(role),
                             "rep": rep, "label": label,
                             "spectrum_id": qualify_spectrum_id(
                                 project, f"{entry.file}#{col + 1}{suffix}")})
        else:
            for time, role, rep, path in specexport_points(root / rel, entry):
                rows.append({**common, "time": _time(time), "role": check_role(role),
                             "rep": rep, "label": path.name,
                             "spectrum_id": qualify_spectrum_id(
                                 project, f"{rel}/{path.name}{suffix}")})
    return SpectrumIndex.from_rows(rows)


def _time(time) -> float:
    return float("nan") if time is None else float(time)


def _input_key(fmt: str, entry: PeptideEntry) -> tuple[str, int]:
    """What an entry's spectra are read from: its input and charge (a SpecExport folder holds
    every charge, one file each; an HX-Express file is read whole, whatever the charge)."""
    return _input_path(fmt, entry), (0 if fmt == "hxexpress" else int(entry.charge))


def _shared_inputs(entries: Sequence[PeptideEntry], fmt: str) -> set[tuple[str, int]]:
    """The inputs (:func:`_input_key`) that several entries of the whole peptide table read (a
    peptide listed with two modifications, or twice): their spectra would have the same ids."""
    counts = Counter(_input_key(fmt, e) for e in entries)
    return {key for key, n in counts.items() if n > 1}


def _id_suffix(fmt: str, entry: PeptideEntry, data_id: int, shared) -> str:
    """What the local spectrum id of ``entry`` (number ``data_id`` of the whole table) gains:
    ``#series`` and its number where its input is shared (:func:`_shared_inputs`), else
    nothing, so that the ids of every other entry stay as they were and a filter never changes
    an id."""
    return f"#series{data_id}" if _input_key(fmt, entry) in shared else ""


def _input_path(fmt: str, entry: PeptideEntry) -> str:
    """The entry's input relative to the data directory: a file or a folder."""
    if fmt == "hxexpress" or fmt == "pooled":
        return entry.file
    return f"{entry.sample}/{entry.file}"


def _read(entries, root: Path, fmt: str, *, project, samples,
          peptide_ranges, recorded_dir: str, record_paths: str, known: Mapping | None,
          peptide_region=None,
          on_change: str, progress, cancel) -> SpectrumSet:
    kept = filter_entries(entries, samples=samples, peptide_ranges=peptide_ranges,
                          peptide_region=peptide_region)
    wanted = {id(e) for e in kept}
    jobs = [(i, e) for i, e in enumerate(entries) if id(e) in wanted]
    shared = _shared_inputs(entries, fmt)
    spectra: list[Spectrum] = []
    files: dict[str, dict] = {}
    for done, (data_id, entry) in enumerate(jobs):
        check_cancel(cancel)
        log.debug("reading %s %s z=%d", entry.sample, entry.peptide_range, entry.charge)
        rel = _input_path(fmt, entry)
        if rel not in files:  # several entries can share one HX-Express file
            if not (root / rel).exists():
                raise ReadError(f"{root / rel} not found (give the directory that holds the "
                                "data)")
            files[rel] = (identity(root / rel) if known is None
                          else compare_identity(root / rel, known.get(rel), on_change,
                                                ReadError))
        if fmt == "hxexpress":
            read = read_hxexpress_file(root / rel, entry, data_id=data_id)
        else:
            read = read_specexport_dir(root / rel, entry, data_id=data_id, id_prefix=rel)
        if suffix := _id_suffix(fmt, entry, data_id, shared):
            read = [replace(sp, spectrum_id=sp.spectrum_id + suffix) for sp in read]
        spectra += read
        report(progress, done + 1, len(jobs))
    source = {
        "reader": fmt,
        "project": project,
        "data_dir": recorded_dir,
        "record_paths": record_paths,
        "entries": [e.to_dict() for e in entries],
        "samples": None if samples is None else list(samples),
        "peptide_ranges": None if peptide_ranges is None else list(peptide_ranges),
        **({} if peptide_region is None else {"peptide_region": [int(x) for x in peptide_region]}),
        "files": files,
    }
    return SpectrumSet.from_spectra(
        (replace(s, spectrum_id=qualify_spectrum_id(project, s.spectrum_id)) for s in spectra),
        source=json.loads(json.dumps(source)),
    )


def read_spectra_source(
    source: Mapping,
    *,
    directory: str | Path | None = None,
    on_change: str = "warn",
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> SpectrumSet:
    """Repeat the read recorded in the ``source`` of spectra made by :func:`read_spectra` (from
    the spectra, a spectra file or a results bundle), from the recorded data directory or from
    ``directory`` (data that has moved, or a record of the name only). Every input is compared
    with its record: a changed one (data updated between runs) is read with a warning
    (``on_change='warn'``) or refused (``'error'``, :class:`~hxex.io.ReadError`), and the new
    record keeps its earlier identity under ``previous``. Unchanged, the record stays as it was.
    """
    fmt = source.get("reader")
    if fmt not in ("hxexpress", "specexport", "pooled"):
        raise ValueError(f"not a read_spectra record: reader {fmt!r}")
    if on_change not in CHANGE_ACTIONS:
        raise ValueError(f"on_change must be one of {', '.join(CHANGE_ACTIONS)}")
    root = Path(directory) if directory is not None else Path(source["data_dir"])
    if not root.is_dir():
        raise ReadError(f"{root} not found (give the directory that holds the data)")
    if "undeut_time" in source or "fulldeut_time" in source:  # records before the roles
        log.info("the record's control times (undeut_time, fulldeut_time) are no longer used: "
                 "the control labels give their spectra a role")
    return _read(
        [PeptideEntry.from_dict(e) for e in source["entries"]], root, fmt,
        project=source["project"], samples=source.get("samples"),
        peptide_ranges=source.get("peptide_ranges"), recorded_dir=source["data_dir"],
        peptide_region=source.get("peptide_region"),
        record_paths=source.get("record_paths", "path"), known=source.get("files", {}),
        on_change=on_change, progress=progress, cancel=cancel,
    )


def read_source(source: Mapping, **kwargs) -> SpectrumSet:
    """Repeat any recorded read: :func:`~hxex.io.read_mzml_source` for an mzML read,
    :func:`~hxex.io.read_thermo_source` for a Thermo raw read, :func:`read_spectra_source` for
    the folder formats (``kwargs`` as those take)."""
    if source.get("reader") == "mzml":
        from .mzml import read_mzml_source

        return read_mzml_source(source, **kwargs)
    if source.get("reader") == "thermo":
        from .thermo import read_thermo_source

        return read_thermo_source(source, **kwargs)
    return read_spectra_source(source, **kwargs)
