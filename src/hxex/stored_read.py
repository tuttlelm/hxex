# SPDX-License-Identifier: GPL-3.0-or-later
"""Taking the stored spectra of the bundle a run replaces in place of reading the input again
(owner, 2026-11-16).

A run that updates a bundle (``hxex run`` on a project with results, or ``--keep-previous``; the
window's Run (update)) after a change that does not touch the read, such as a new preset,
would read every file again to get the spectra the bundle already holds. A fresh read records
its *read plan* (:func:`read_plan`) with the spectra: the fingerprint of the readers' code, the
``[input]`` settings that decide the read, what they give before any file is opened (the
entries or targets of the peptide table, the elution-window rows, the picker and registry the
scan readers use), and the identities of the small files the read takes beside the data. A
later run computes the plan again; when it is the same and every data file still matches the
identity the read recorded (:mod:`hxex.io.identity`: size and checksum), the stored spectra are
taken (:func:`stored_spectra_for`). Anything else, a bundle written before plans were recorded
included, reads the input as before, and the log names what differs. ``hxex run --fresh``
always reads.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

log = logging.getLogger(__name__)

PLAN_KEY = "read_plan"  # the read record's key holding the plan
# [input] settings that are not the read's: applied to the spectra after it (time_scale, and
# the stored spectra themselves) or when fitting ([[input.exclude]])
_NOT_THE_READ = ("spectra", "exclude", "time_scale")
# what each part of the plan is, for the log
_PARTS = {
    "reader": "the reading code (another hxex)",
    "project": "the project name",
    "entries": "the peptides read (the peptide table or the data folders)",
    "targets": "the targets (the peptide table, parents or fragments)",
    "elution_windows": "the elution-window table",
    "picker": "[run.picker]",
    "registry": "the modification registry",
    "fasta": "the FASTA files",
    "data_dir": "the data folder",
    "data_root": "the data root",
    "retention_times": "the retention-time table",
    "pooled_fasta": "the pooled FASTA file",
}


def _normal(value):
    """``value`` as JSON gives it back (tuples as lists, keys as text), for comparisons."""
    return json.loads(json.dumps(value, default=str))


def _resolved(path) -> str | None:
    return None if path is None else str(Path(path).resolve())


def read_plan(project) -> dict | None:
    """What the read of ``project`` (a :class:`hxex.config.Project`) depends on, worked out
    without opening a data file (see the module); None when it cannot be (the read itself then
    says why)."""
    from ._engine import reader_fingerprint
    from .config import FOLDER_FORMATS, _input_record, folder_entries, scan_targets
    from .io.identity import file_identity

    inp = project.input
    if inp is None:
        return None
    try:
        record = _input_record(replace(inp, spectra=None, exclude=()), "path")
        for key in _NOT_THE_READ:
            record.pop(key, None)
        plan = {"reader": reader_fingerprint(), "project": project.project, "input": record}
        if inp.format in FOLDER_FORMATS:
            plan["entries"] = [e.to_dict() for e in folder_entries(project)]
            plan["data_dir"] = _resolved(inp.data_dir)
            if inp.format == "specexport" and inp.data_dir is not None:
                from .io.specexport import _fasta_files

                plan["fasta"] = [file_identity(p) for p in _fasta_files(Path(inp.data_dir))]
        else:
            plan["targets"] = [t.to_dict() for t in scan_targets(project)]
            plan["picker"] = project.run.picker.to_dict()
            plan["registry"] = project.run.registry.to_dict()
            plan["data_root"] = _resolved(inp.data_root)
            if inp.rt_elution_windows is not None:
                from .io.elution_windows import read_elution_windows

                plan["elution_windows"] = [w.to_dict() for w in
                                           read_elution_windows(inp.rt_elution_windows)]
        if inp.retention_times is not None:
            plan["retention_times"] = file_identity(inp.retention_times)
        if inp.pooled_fasta is not None:
            plan["pooled_fasta"] = file_identity(inp.pooled_fasta)
            plan["pooled_classes"] = inp.pooled_classes
        return _normal(plan)
    except Exception as exc:  # noqa: BLE001 - the read reports the problem itself
        log.debug("no read plan: %s", exc)
        return None


def plan_differences(stored: dict, now: dict) -> list[str]:
    """The parts of two read plans that differ, in words (``[input] KEY`` for a setting)."""
    out = []
    for key in sorted(set(stored) | set(now)):
        if stored.get(key) == now.get(key):
            continue
        if key == "input":
            a, b = stored.get(key) or {}, now.get(key) or {}
            out += [f"[input] {k}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        else:
            out.append(_PARTS.get(key, key))
    return out


_SAME = ("size", "algorithm", "checksum", "n_files")


def changed_files(project, source: dict) -> list[str]:
    """The data files (or SpecExport folders) of ``project`` that differ from the identities
    recorded in the read record ``source`` (size and checksum: :mod:`hxex.io.identity`), or are
    missing; empty when every one matches."""
    from .config import FOLDER_FORMATS
    from .io.identity import file_identity, identity

    inp = project.input
    files = source.get("files")
    if inp.format in FOLDER_FORMATS:
        if not isinstance(files, dict) or inp.data_dir is None:
            return ["the record of the files read"]
        wanted = [(Path(inp.data_dir) / rel, known, identity) for rel, known in files.items()]
    else:
        root = None if inp.data_root is None else Path(inp.data_root)
        if not isinstance(files, list) or len(files) != len(inp.runs):
            return ["the record of the files read"]
        wanted = [(Path(r.path) if root is None else root / r.path, known, file_identity)
                  for r, known in zip(inp.runs, files, strict=True)]
    seen: dict[Path, dict] = {}
    out = []
    for path, known, how in wanted:
        if not path.exists():
            out.append(f"{path} (not found)")
            continue
        key = path.resolve()
        if key not in seen:
            seen[key] = how(path)
        now = seen[key]
        if not all(now.get(k) == (known or {}).get(k) for k in _SAME):
            out.append(str(path))
    return out


def stored_spectra_for(project, bundle):
    """The raw spectra stored in ``bundle`` (the bundle a run of ``project`` replaces), when
    reading the input again would give them: the read plan recorded with them is the project's
    now, and every data file is unchanged (see the module). They come with the steps taken after
    a read (:func:`hxex.config.after_read`). None otherwise, with one log line saying why the
    input is read."""
    from .config import ProjectError, after_read, load_stored_spectra, stored_spectra_source

    try:
        source = stored_spectra_source(bundle)
    except ProjectError:  # a bundle without raw spectra (keep_spectra = false)
        log.info("reading the input: %s holds no raw spectra to take", bundle)
        return None
    if not source:
        return None
    stored = source.get(PLAN_KEY)
    if stored is None:
        log.info("reading the input: the spectra stored in %s were read before hxex recorded "
                 "what a read depends on", bundle)
        return None
    now = read_plan(project)
    if now is None:  # the read says what is wrong with the settings
        log.info("reading the input: what it depends on could not be worked out from the "
                 "settings alone")
        return None
    if differ := plan_differences(stored, now):
        log.info("reading the input: %s changed since the spectra stored in %s were read",
                 _listed(differ), bundle)
        return None
    if changed := changed_files(project, source):
        log.info("reading the input: %s changed since the spectra stored in %s were read",
                 _listed(changed), bundle)
        return None
    log.info("taking the raw spectra stored in %s: the input settings and files are those "
             "they were read from (hxex run --fresh reads the input again)", bundle)
    return after_read(load_stored_spectra(bundle), project.input, stored=True)


def _listed(items: list[str], most: int = 4) -> str:
    return ", ".join(items[:most]) + (f" and {len(items) - most} more" if len(items) > most
                                      else "")
