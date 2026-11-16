# SPDX-License-Identifier: GPL-3.0-or-later
"""The Project workspace's sequence view (owner, 2026-11-04: "a nice looking sequence viewer"):
the protein sequence of each sample as the analysis numbers it, in blocks of ten with a ruler,
one row per distinct sequence (samples with the same one share a row), a residue that differs
from the first row's highlighted, prolines (no amide to exchange) in grey, the regions of
``[[analysis.regions]]`` as bars under the residues and the sites of ``[analysis.sites]``
marked. A residue clicked says what it is. With results open for the project, each sample's
coverage runs under its row: a block per residue as high as the number of MS1 peptides covering
it (those with a spectrum not rejected; fragment ions are not counted), a dot where none does
(owner, 2026-11-05 and 2026-11-15).

The sequences are those the analysis reads: ``[analysis] fasta`` (one file, or one per sample),
else, for SpecExport, the ``SAMPLE.fasta`` files beside the sample folders (a ``SAMPLE.fasta``
with no sample folder of its name beside it is not read, so it is left out, with a note); a
FASTA numbered by its header (``>NAME/START-END``) starts at START. The files come first; a
SpecExport run also keeps each sample's sequence in its bundle (the read record's
``sequences``): a sample whose file is not found (or does not read), whether a ``SAMPLE.fasta``
or a file ``[analysis] fasta`` names, is shown with that stored copy and a note naming the
file, and a file that no longer matches its copy is said (owner, 2026-11-04 and 2026-11-15).

Pooled data (owner, 2026-11-10) show the records of ``[input] pooled_fasta`` instead, grouped
by class (``[input] pooled_classes``), each record's differences from the first of its class
highlighted; with results open, each record's coverage is that of the peptides it may hold, in
its own numbering, the residues only that record's peptides cover in black. How many residues
a line holds is chosen in the view (:data:`PER_LINE_CHOICES`).
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .tips import tip

log = logging.getLogger(__name__)

PER_LINE = 50  # residues a line: five blocks of ten
PER_LINE_CHOICES = (30, 40, 50, 60, 70, 80, 90, 100)  # the view's choice: blocks of ten
ABOUT = "Click a residue to see what it is."  # the line under the sequences, at the start
REGION_COLOURS = ("#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#76b7b2", "#edc948", "#9c755f")
DIFFERS, PROLINE, SITE = "#ffd54f", "#9e9e9e", "#c62828"
COVER, UNCOVERED = "#546e7a", "#e57373"  # coverage blocks, and the dot of a residue not covered
OWN = "#000000"  # pooled: a residue covered by a peptide only that record may hold
BLOCKS = "▁▂▃▄▅▆▇█"  # the coverage levels where the font draws them at a letter's width
BAR = "━"  # a region's bar
# a font that lacks some of them (Courier New) borrows them from another font at that font's
# width, so the lines drift: these are in every fixed font (the WGL4 set, hxex.gui.fonts)
BLOCKS_SAFE, BAR_SAFE = "░▒▓█", "─"


@dataclass
class SequenceRow:
    """One distinct sequence: the ``samples`` that have it, its ``residues`` (modX labels) and
    the number of its ``first`` residue."""

    samples: list[str]
    residues: tuple[str, ...]
    first: int = 1
    differs: set[int] = field(default_factory=set)  # residue numbers unlike the first row's
    # pooled: the record's class ('' for a record of no class, and for other data); a row's
    # differences are counted against the first row of its group
    group: str = ""

    @property
    def last(self) -> int:
        return self.first + len(self.residues) - 1

    def at(self, number: int) -> str:
        """The residue numbered ``number`` ('' outside the sequence)."""
        i = number - self.first
        return self.residues[i] if 0 <= i < len(self.residues) else ""


def residues_of(sequence: str) -> tuple[str, ...]:
    """A sequence's residues as modX labels (never its characters one by one): the registry's
    reading, or for letters it does not know (an X in a FASTA) each label as written."""
    from ..chem import parse_sequence

    try:
        return tuple(parse_sequence(str(sequence)))
    except Exception:  # noqa: BLE001 - a FASTA with letters hxex does not weigh
        return tuple(re.findall(r"[a-z]*[A-Z]", str(sequence)))


def stored_sequences(bundle: Path | None) -> dict[str, tuple[str, int, str]]:
    """The sequences a SpecExport run stored in ``bundle`` (sample -> (sequence, first residue
    number, FASTA file name)); none without a bundle or for another format."""
    from ..io.bundle import bundle_manifest

    if bundle is None:
        return {}
    try:
        manifest = bundle_manifest(bundle) or {}
    except Exception:  # noqa: BLE001 - no bundle to read: nothing stored
        return {}
    source = (manifest.get("configs") or {}).get("spectra_source") or {}
    return {str(name): (str(v["sequence"]), int(v.get("first", 1)), str(v.get("file", "")))
            for name, v in (source.get("sequences") or {}).items()}


def project_sequences(project) -> tuple[list[SequenceRow], object, str]:
    """The rows of the project's sequences, its annotations, and a message: why there is no
    sequence when there are no rows, else what to say of where they came from ('' when the
    files were read as they are). ``project``: the project file's path, or the window's
    :class:`hxex.gui.project_state.ProjectState` of it (read once for every panel, Q-2)."""
    from ..analysis.compare import Annotations
    from ..io import read_fasta_numbered
    from .project_state import ProjectState

    state = project if isinstance(project, ProjectState) else ProjectState.read(project)
    project = state.path
    notes: list[str] = []
    if state.loaded is None:
        return [], Annotations(), f"The project file does not read: {state.problem}"
    loaded = state.loaded
    config = loaded.analysis
    stored = stored_sequences(loaded.output.bundle)
    annotations = config.annotations if config is not None else Annotations()
    inputs = loaded.input
    if (inputs is not None and (inputs.format == "pooled" or inputs.pooled_fasta is not None)
            and (config is None or config.fasta is None)):
        if inputs.pooled_fasta is None:
            return [], annotations, ("No sequence: name the FASTA of what the pool may hold in "
                                     "[input] pooled_fasta (and its classes in [input] "
                                     "pooled_classes).")
        rows, message = _pooled_rows(inputs)
        return rows, annotations, message
    found: dict[str, tuple[tuple[str, ...], int]] = {}
    # only the samples the project reads (owner, 2026-11-13: projects split by sample from one
    # data folder); a FASTA is named by its sample, in any case, as the reader pairs them
    wanted = (None if inputs is None or not inputs.samples
              else {str(s).lower() for s in inputs.samples})

    def read(name: str) -> bool:
        return wanted is None or name.lower() in wanted
    # samples whose FASTA is not there (or does not read): (sample, file name, why), each then
    # shown with the copy the run stored in its bundle where there is one (E-14)
    lost: list[tuple[str, str, str]] = []
    if config is not None and config.fasta is not None:
        try:
            made, gone = _analysis_fasta(config.fasta, project.parent)
        except Exception as exc:  # noqa: BLE001 - said in the view
            return [], annotations, f"[analysis] fasta does not read: {exc}"
        for name, value in made.items():
            if ":" in name or name == EVERY or read(name):
                found[name] = value
        for name, file in gone:
            if name == EVERY:  # one file for every sample: each sample's stored copy
                if stored:
                    lost += [(s, file, "is not found") for s in stored if read(s)]
                else:
                    notes.append(f"{file} ([analysis] fasta) is not found, and the run stored "
                                 "no copy of the sequence in its bundle.")
            elif ":" in name or name not in stored:  # a run's sample, or nothing stored
                if ":" in name or read(name):
                    notes.append(f"{file} ([analysis] fasta, {name}) is not found, and the run "
                                 "stored no copy of its sequence in its bundle.")
            elif read(name):
                lost.append((name, file, "is not found"))
    else:
        values = state.values
        if str(values.get("input.format", "")).strip('"') == "specexport":
            folder = Path(str(values.get("input.data_dir", "")).strip('"'))
            folder = folder if folder.is_absolute() else project.parent / folder
            alone: list[str] = []  # SAMPLE.fasta without its sample folder: not read (Q7)
            if folder.is_dir():
                dirs = {p.name for p in folder.iterdir() if p.is_dir()}
                for path in sorted(p for p in folder.iterdir()
                                   if p.suffix.lower() == ".fasta" and p.is_file()
                                   and read(p.stem)):
                    if path.stem not in dirs:  # as the reader pairs them: same name, same case
                        alone.append(path.name)
                        continue
                    try:
                        seq, first = read_fasta_numbered(path)
                    except Exception as exc:  # noqa: BLE001 - one file: said, the rest shown
                        log.warning("%s does not read: %s", path.name, exc)
                        lost.append((path.stem, path.name, f"does not read ({exc})"))
                        continue
                    found[path.stem] = (residues_of(seq), first)
                    kept = stored.get(path.stem)
                    if kept is not None and (residues_of(kept[0]), kept[1]) != (
                            residues_of(seq), first):
                        notes.append(f"{path.name} differs from the copy the run stored in its "
                                     "bundle (the file is shown; rerun to read the peptides "
                                     "against it)")
            if alone:
                one = len(alone) == 1
                notes.append(f"{', '.join(alone)} {'has' if one else 'have'} no sample folder "
                             f"of {'its' if one else 'their'} name beside "
                             f"{'it' if one else 'them'}, so hxex does not read "
                             f"{'it' if one else 'them'}: left out.")
            left_out = {Path(a).stem for a in alone}
            told = {name for name, _, _ in lost}
            lost += [(name, file or f"{name}.fasta", "is not found")
                     for name, (_, _, file) in stored.items()
                     if name not in found and name not in told and name not in left_out
                     and read(name)]
    if not found and not lost and stored:  # nothing named, nothing found: the run's copies
        lost = [(name, file or f"{name}.fasta", "is not found")
                for name, (_, _, file) in stored.items() if read(name)]
    shown = [(name, file, why) for name, file, why in lost
             if name in stored and name not in found]
    for name, _, _ in shown:
        seq, first, _ = stored[name]
        found[name] = (residues_of(seq), first)
    for name, file, why in lost:
        if name not in stored:
            notes.append(f"{file} {why}, and the run stored no copy of its sequence in its "
                         "bundle.")
    if shown:
        files = list(dict.fromkeys(f for _, f, _ in shown))
        if len(shown) > 1 and len(shown) == len(found):  # every sequence the stored copy
            notes.append((f"{files[0]} is not found" if len(files) == 1 else
                          f"The FASTA files are not found ({', '.join(files)})")
                         + ": the sequences the run stored in its bundle are shown.")
        else:
            notes += [f"{file} {why}: the sequence the run stored in its bundle is shown for "
                      f"{name}." for name, file, why in shown]
    if not found:
        return [], annotations, "".join(f"{n} " for n in notes) + (
            "No sequence: name the FASTA file(s) in [analysis] fasta (or, for SpecExport data, "
            "put SAMPLE.fasta beside each sample's folder).")
    # the reference first (each other row then shows its own differences), then [analysis]
    # samples' order, the rest alphabetically
    ref = config.reference if config is not None and isinstance(config.reference, str) else ""
    order = [ref] + list(config.samples or ()) if config is not None else []
    found = dict(sorted(found.items(), key=lambda item: (
        order.index(item[0]) if item[0] in order else len(order), item[0])))
    rows: list[SequenceRow] = []
    for name, (residues, first) in found.items():
        same = next((r for r in rows if r.residues == residues and r.first == first), None)
        if same is not None:
            same.samples.append(name)
        else:
            rows.append(SequenceRow([name], residues, first))
    _mark_differences(rows)
    return rows, annotations, " ".join(notes)


EVERY = "every sample"  # the row name of one [analysis] fasta file for every sample


def _analysis_fasta(fasta, base: Path) -> tuple[dict, list[tuple[str, str]]]:
    """The sequences of ``[analysis] fasta`` (one file, or one per sample, a run's own under
    ``run:sample``), relative paths from ``base``: name -> (residues, first residue number),
    and the (name, file name) of each file that is not found (shown with the run's stored copy
    instead, open question 6). A file that is there but does not read raises."""
    from collections.abc import Mapping

    from ..io import read_fasta_numbered

    made: dict[str, tuple[tuple[str, ...], int]] = {}
    gone: list[tuple[str, str]] = []

    def one(name: str, text) -> None:
        path = Path(str(text)).expanduser()
        path = path if path.is_absolute() else base / path
        if not path.is_file():
            gone.append((name, path.name))
            return
        sequence, first = read_fasta_numbered(path)
        made[name] = (residues_of(sequence), first)

    if isinstance(fasta, Mapping):
        for key, value in fasta.items():
            if isinstance(value, Mapping):  # a run's samples
                for sample, text in value.items():
                    one(f"{key}:{sample}", text)
            else:
                one(str(key), value)
    else:
        one(EVERY, fasta)
    return made, gone


def _pooled_rows(inputs) -> tuple[list[SequenceRow], str]:
    """The rows of a pooled project: each record of ``[input] pooled_fasta``, grouped by class
    in the order ``[input] pooled_classes`` names them (records of no class last; a record of
    several classes in its first), each in FASTA order and numbered from 1; records with one
    sequence in one class share a row. A message when the file does not read."""
    from ..io.pooled import candidates_record

    try:
        record = candidates_record(inputs.pooled_fasta, inputs.pooled_classes or ())
    except Exception as exc:  # noqa: BLE001 - said in the view
        return [], f"[input] pooled_fasta does not read: {exc}"
    members = record.get("class_members") or {}
    group_of: dict[str, str] = {}
    for name in record.get("classes") or []:
        for rid in members.get(name, ()):
            group_of.setdefault(str(rid), str(name))
    order = [str(c) for c in record.get("classes") or []]
    records = sorted(((str(rid), str(seq)) for rid, seq in record["records"]),
                     key=lambda r: order.index(group_of[r[0]]) if r[0] in group_of
                     else len(order))  # sorted() is stable: FASTA order within a class
    rows: list[SequenceRow] = []
    for rid, seq in records:
        residues, group = residues_of(seq), group_of.get(rid, "")
        same = next((r for r in rows if r.residues == residues and r.group == group), None)
        if same is not None:
            same.samples.append(rid)
        else:
            rows.append(SequenceRow([rid], residues, 1, group=group))
    _mark_differences(rows)
    classes = len({r.group for r in rows if r.group})
    return rows, (f"The records of {Path(inputs.pooled_fasta).name}"
                  + (f", in {classes} class(es); each differs from the first of its class"
                     if classes else "") + ".")


def _mark_differences(rows: list[SequenceRow]) -> None:
    """Each row's residues unlike those of the first row of its group (all rows one group for
    data other than pooled)."""
    first: dict[str, SequenceRow] = {}
    for row in rows:
        base = first.setdefault(row.group, row)
        if base is not row:
            row.differs = {n for n in range(row.first, row.last + 1)
                           if base.at(n) and row.at(n) and row.at(n) != base.at(n)}


def fragment_rows(results) -> np.ndarray:
    """For each spectrum of ``results.binned``: whether its series is a fragment ion
    (:func:`hxex.controls.fragment_ids`: an ion type other than M, or a target of a fragment
    list, the precursor read in the fragment scans included), so not an MS1 peptide."""
    b = results.binned
    n = len(b.spectrum_id)
    modification = getattr(b, "modification", None)
    data_id = getattr(b, "data_id", None)
    if modification is None or data_id is None:
        return np.zeros(n, dtype=bool)
    from ..analysis.screening import spectra_source
    from ..chem import parse_modification
    from ..io.mzml import fragment_series

    # as fragment_ids, each distinct modification read once (the view redraws on every
    # review decision)
    mods, inverse = np.unique(np.asarray(modification).astype(str), return_inverse=True)
    other = np.array([parse_modification(m).ion_type != "M" for m in mods], dtype=bool)
    listed = list(fragment_series(spectra_source(results)))
    return other[inverse.ravel()] | np.isin(np.asarray(data_id), listed)


def coverage_of(results) -> dict[str, list[tuple]]:
    """Each sample's MS1 peptides in ``results`` (:class:`hxex.report.Results`) with at least
    one spectrum the review did not reject: ``(first residue, last residue, peptide_range)``,
    each peptide once whatever its charges, in residue order; fragment ions are not counted
    (:func:`fragment_rows`; owner, 2026-11-15). For pooled results with a pooled FASTA: each
    record's peptides instead (:func:`pooled_coverage_of`)."""
    fit, b = results.fit, results.binned
    if b is None:
        return {}
    pooled = pooled_coverage_of(results)
    if pooled is not None:
        return pooled
    index = {str(s): j for j, s in enumerate(b.spectrum_id)}
    status = np.asarray(results.review_status).astype(str)
    sample = np.asarray(b.sample).astype(str)
    start, end = np.asarray(b.start_seq), np.asarray(b.end_seq)
    ranges = np.asarray(b.peptide_range).astype(str)
    fragment = fragment_rows(results)
    out: dict[str, set[tuple[int, int, str]]] = {}
    for i, sid in enumerate(map(str, fit.spectrum_id)):
        j = index.get(sid)
        if j is None or status[i] == "rejected" or fragment[j]:
            continue
        out.setdefault(sample[j], set()).add((int(start[j]), int(end[j]), ranges[j]))
    return {s: sorted(v) for s, v in sorted(out.items())}


def pooled_coverage_of(results) -> dict[str, list[tuple]] | None:
    """Each record of the pooled FASTA with the peptides of ``results`` it may hold (every
    sample together; an MS1 peptide with a spectrum the review did not reject): ``(first residue,
    last residue, peptide_range, own)`` in the record's own numbering (the peptide's residues
    moved by the record's offset), ``own`` when no other record holds it. None for results
    without a pooled FASTA."""
    from ..report.tables import series_identities

    known = series_identities(results) if getattr(results, "settings", None) else None
    if known is None:
        return None
    fit, b = results.fit, results.binned
    index = {str(s): j for j, s in enumerate(b.spectrum_id)}
    status = np.asarray(results.review_status).astype(str)
    start, end = np.asarray(b.start_seq), np.asarray(b.end_seq)
    ranges = np.asarray(b.peptide_range).astype(str)
    fragment = fragment_rows(results)
    out: dict[str, set[tuple]] = {}
    for i, (sid, d) in enumerate(zip(map(str, fit.spectrum_id), fit.data_id, strict=True)):
        j, found = index.get(sid), known.get(int(d))
        if j is None or found is None or status[i] == "rejected" or fragment[j]:
            continue
        own = len(found.candidates) == 1
        for record, offset in zip(found.candidates, found.offsets, strict=True):
            out.setdefault(record, set()).add((int(start[j]) + offset, int(end[j]) + offset,
                                               ranges[j], own))
    return {r: sorted(v) for r, v in sorted(out.items())}


def _coverage_samples(row: SequenceRow, coverage) -> list[str]:
    """The samples of ``coverage`` that a row's names stand for: a sample's own name, the
    sample part of ``run:sample``, or every sample for one sequence shared by all."""
    if not coverage:
        return []
    if row.samples == [EVERY]:
        return list(coverage)
    out = []
    for name in row.samples:
        sample = name.split(":", 1)[1] if ":" in name and name not in coverage else name
        if sample in coverage and sample not in out:
            out.append(sample)
    return out


def depth(peptides, number: int) -> int:
    """How many of ``peptides`` (:func:`coverage_of`) cover residue ``number``."""
    return sum(1 for p in peptides if p[0] <= number <= p[1])


def _own(peptides, number: int) -> bool:
    """Whether a peptide only this record may hold covers residue ``number`` (pooled)."""
    return any(len(p) > 3 and p[3] and p[0] <= number <= p[1] for p in peptides)


def sequence_html(rows: list[SequenceRow], annotations, per_line: int = PER_LINE,
                  coverage=None, family: str = "monospace", blocks: str = BLOCKS,
                  bar: str = BAR) -> str:
    """The sequences as HTML for Qt's text view: lines of ``per_line`` residues in blocks of
    ten in the fixed-width font ``family``, a ruler above each line, each row named on the
    left, every residue a link (``res:<row>:<number>``) for the view to say what it is.
    ``coverage`` (:func:`coverage_of`) adds each sample's coverage under its row, drawn with
    ``blocks`` (the levels) and ``bar`` (a region's bar): the view passes the safe set when
    ``family`` draws the others at another width (:func:`hxex.gui.fonts.even_width`)."""
    if not rows:
        return ""
    first = min(r.first for r in rows)
    last = max(r.last for r in rows)
    shown = {k: _coverage_samples(r, coverage) for k, r in enumerate(rows)}
    names = [", ".join(r.samples) for r in rows] + [s for v in shown.values() for s in v]
    label = min(max(max(len(n) for n in names), 6), 28)
    top = max((depth(coverage[s], n) for v in shown.values() for s in v
               for n in range(first, last + 1)), default=0)
    regions = list(getattr(annotations, "regions", ()))
    sites = {s.residue: s.label for s in getattr(annotations, "sites", ())}
    colour = {id(reg): REGION_COLOURS[i % len(REGION_COLOURS)] for i, reg in enumerate(regions)}
    out = [f'<div style="font-family: {html.escape(family)}, monospace; font-size: 11pt; '
           'white-space: pre;">']
    if regions or sites or top:
        keys = [f'<span style="color:{colour[id(r)]};">{bar * 2}</span> {html.escape(r.name)} '
                f'({r.start}–{r.end})' for r in regions]
        if sites:
            keys.append(f'<span style="color:{SITE};">▲</span> site')
        if top:
            keys.append(f'<span style="color:{COVER};">{blocks[0]}…{blocks[-1]}</span> '
                        f'MS1 peptides covering a residue (1 to {top}; fragment ions '
                        'not counted), '
                        f'<span style="color:{UNCOVERED};">·</span> none')
            if any(len(p) > 3 and p[3] for v in coverage.values() for p in v):
                keys.append(f'<span style="color:{OWN};">{blocks[-1]}</span> covered by a '
                            "peptide only this record may hold")
        out.append('<span style="font-family: sans-serif; font-size: 9pt;">'
                   + "    ".join(keys) + "</span>\n\n")
    start = first - ((first - 1) % per_line)  # lines begin at 1, 51, 101, ...
    for line_start in range(start, last + 1, per_line):
        numbers = range(line_start, min(line_start + per_line, last + 1))  # not past the end
        out.append(_ruler(numbers, label) + "\n")
        for k, row in enumerate(rows):
            if k and row.group != rows[k - 1].group:
                out.append("\n")  # pooled: a line between the classes
            name = ", ".join(row.samples)
            name = name if len(name) <= label else name[:label - 1] + "…"
            cells = []
            for n in numbers:
                residue = row.at(n)
                text = html.escape(residue[-1]) if residue else " "  # the base letter
                if residue:
                    style = []
                    if n in row.differs:
                        style.append(f"background:{DIFFERS}; font-weight:bold")
                    elif residue[-1] == "P":
                        style.append(f"color:{PROLINE}")
                    if n in sites:
                        style.append(f"text-decoration: underline; color:{SITE}")
                    text = (f'<a href="res:{k}:{n}" style="text-decoration:none; color:inherit;">'
                            f'<span style="{"; ".join(style)}">{text}</span></a>')
                cells.append(text)
                if (n - start + 1) % 10 == 0:
                    cells.append(" ")
            out.append(f"{html.escape(name):<{label}}  {''.join(cells)}\n")
            for sample in shown[k]:
                cover = _coverage_line(numbers, row, coverage[sample], top, start, blocks)
                out.append(f'<span style="color:{COVER};">{html.escape(sample[:label]):<{label}}'
                           f'</span>  {cover}\n')
        bars = _region_line(numbers, regions, colour, start, bar)
        if bars.strip():
            out.append(f"{'':<{label}}  {bars}\n")
        out.append("\n")
    out.append("</div>")
    return "".join(out)


def _ruler(numbers: range, label: int) -> str:
    """The numbers of a line: each tenth residue's number ending above it."""
    width = len(numbers) + len(numbers) // 10
    line = [" "] * (width + 6)
    for i, n in enumerate(numbers):
        if n % 10 == 0:
            text = str(n)
            end = i + i // 10  # the column of residue n (a space after every ten)
            for j, ch in enumerate(reversed(text)):
                if end - j >= 0:
                    line[end - j] = ch
    ruler = "".join(line).rstrip()
    return f'<span style="color:#757575;">{"":<{label}}  {html.escape(ruler)}</span>'


def _coverage_line(numbers: range, row: SequenceRow, peptides, top: int, start: int,
                   blocks: str = BLOCKS) -> str:
    """A sample's coverage under its row: a block per residue as high as the peptides covering
    it (scaled to ``top``, the most over every sample), a dot where none does."""
    cells = []
    for n in numbers:
        if not row.at(n):
            cells.append(" ")
        else:
            d = depth(peptides, n)
            if d:
                level = blocks[min(len(blocks) - 1, max(0, -(-d * len(blocks) // top) - 1))]
                cells.append(f'<span style="color:{OWN if _own(peptides, n) else COVER};">'
                             f'{level}</span>')
            else:
                cells.append(f'<span style="color:{UNCOVERED};">·</span>')
        if (n - start + 1) % 10 == 0:
            cells.append(" ")
    return "".join(cells)


def _region_line(numbers: range, regions, colour: dict, start: int, bar: str = BAR) -> str:
    """Bars under the residues of each region (the first region's colour where they overlap)."""
    cells = []
    for n in numbers:
        hit = next((r for r in regions if r.start <= n <= r.end), None)
        cells.append(f'<span style="color:{colour[id(hit)]};">{bar}</span>' if hit else " ")
        if (n - start + 1) % 10 == 0:
            cells.append(" ")
    return "".join(cells)


def describe(rows: list[SequenceRow], annotations, row: int, number: int,
             coverage=None) -> str:
    """What a residue clicked is: its sample(s), number and residue, how the other rows differ
    there, the regions and site it belongs to, and (``coverage``) the peptides covering it in
    each sample of its row."""
    here = rows[row]
    parts = [f"{', '.join(here.samples)}" + (f" (class {here.group})" if here.group else "")
             + f": residue {number}, {here.at(number)}"]
    if here.at(number).endswith("P"):
        parts.append("a proline (no amide hydrogen to exchange)")
    others = [f"{', '.join(r.samples)} {r.at(number) or '(none)'}" for i, r in enumerate(rows)
              if i != row and r.group == here.group and r.at(number) != here.at(number)]
    if others:
        parts.append("differs: " + "; ".join(others))
    regions = [f"{r.name} ({r.start}–{r.end})" for r in getattr(annotations, "regions", ())
               if r.start <= number <= r.end]
    if regions:
        parts.append("region " + ", ".join(regions))
    site = next((s for s in getattr(annotations, "sites", ()) if s.residue == number), None)
    if site is not None:
        parts.append(f"site {site.label}".rstrip())
    for sample in _coverage_samples(here, coverage):
        over = [f"{p[0]}-{p[1]}" + (" (only this record)" if len(p) > 3 and p[3] else "")
                for p in coverage[sample] if p[0] <= number <= p[1]]
        parts.append(f"{sample}: " + (f"{len(over)} peptide(s), " + ", ".join(over[:8])
                                       + (", ..." if len(over) > 8 else "")
                                       if over else "not covered"))
    return "; ".join(parts)


try:  # the widget (the module's functions above import without Qt)
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import (
        QComboBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover - the gui extra is not installed
    QWidget = object  # type: ignore[assignment,misc]


class SequenceView(QWidget):  # type: ignore[misc,valid-type]
    """The sequences of the project shown (:meth:`show_project`), a residue clicked described."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list[SequenceRow] = []
        self.annotations = None
        self.project: Path | None = None
        self.coverage: dict | None = None  # the open results' peptides (coverage_of)
        self._message = ""
        self.per_line_box = QComboBox()  # residues a line (owner, 2026-11-10)
        for n in PER_LINE_CHOICES:
            self.per_line_box.addItem(f"{n} residues a line", n)
        self.per_line_box.setCurrentIndex(PER_LINE_CHOICES.index(PER_LINE))
        self.per_line_box.setToolTip(tip("how many residues each line holds (in blocks of "
                                         "ten); kept in the window's session file"))
        self.per_line_box.currentIndexChanged.connect(lambda _: self._draw()
                                                      if self.rows else None)
        self.text = QTextBrowser()
        self.text.setOpenLinks(False)
        self.text.anchorClicked.connect(self._clicked)
        self.about = QLabel(ABOUT)
        self.about.setWordWrap(True)
        refresh = QPushButton("Read again")
        refresh.setToolTip(tip("read the FASTA files again (after editing one)"))
        refresh.clicked.connect(lambda: self.show_project(self.project))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.text, 1)
        layout.addWidget(self.about)
        buttons = QHBoxLayout()
        buttons.addWidget(self.per_line_box)
        buttons.addStretch(1)
        buttons.addWidget(refresh)
        layout.addLayout(buttons)

    @property
    def per_line(self) -> int:
        """The residues a line the view shows."""
        return int(self.per_line_box.currentData() or PER_LINE)

    def set_per_line(self, n) -> None:
        """Show ``n`` residues a line (one of :data:`PER_LINE_CHOICES`; others, and None, the
        default), as a session file restores it."""
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = PER_LINE
        self.per_line_box.setCurrentIndex(PER_LINE_CHOICES.index(
            n if n in PER_LINE_CHOICES else PER_LINE))

    def view_settings(self) -> dict:
        """How the view shows the sequences, for the session (only what differs from the
        default)."""
        return {} if self.per_line == PER_LINE else {"residues_per_line": self.per_line}

    def show_project(self, project) -> None:
        """Show the sequences of ``project``: its path, or the window's parsed state of it
        (None: no project); Read again reads the files again."""
        from .project_state import ProjectState

        if isinstance(project, ProjectState):
            project = project if project.path is not None else None
        self.project = project.path if isinstance(project, ProjectState) else project
        if project is None:
            self.rows, self.annotations = [], None
            self._clear("No project file open.")
            return
        self.rows, self.annotations, message = project_sequences(project)
        if not self.rows:
            self._clear(message)
            return
        self._message = message
        self._draw()

    def _clear(self, text: str) -> None:
        """No sequences to show: ``text`` instead, and nothing left of the project before (its
        counts under the text, its message)."""
        self.text.setPlainText(text)
        self._message = ""
        self.about.setText(ABOUT)

    def show_coverage(self, coverage: dict | None) -> None:
        """The peptides of the project's results (:func:`coverage_of`; None: no results), drawn
        under the sequences."""
        self.coverage = coverage or None
        if self.rows:
            self._draw()

    def _draw(self) -> None:
        from .fonts import even_width, fixed_family, fixed_font

        bar = self.text.verticalScrollBar().value()
        font = fixed_font()
        even = even_width(font, BLOCKS + BAR + "·▲")
        self.text.setHtml(sequence_html(self.rows, self.annotations, per_line=self.per_line,
                                        coverage=self.coverage, family=fixed_family(),
                                        blocks=BLOCKS if even else BLOCKS_SAFE,
                                        bar=BAR if even else BAR_SAFE))
        self.text.verticalScrollBar().setValue(bar)
        n = sum(len(r.samples) for r in self.rows)
        what = "record(s)" if any(r.group for r in self.rows) else "sample(s)"
        self.about.setText(f"{n} {what}, {len(self.rows)} distinct sequence(s). "
                           + (f"{self._message} " if self._message else "")
                           + ("Coverage from the open results: MS1 peptides only (fragment "
                              "ions and rejected spectra left out). "
                              if self.coverage else "")
                           + "Click a residue to see what it is.")

    def _clicked(self, url: QUrl) -> None:
        text = url.toString()
        if not text.startswith("res:"):
            return
        _, row, number = text.split(":")
        self.about.setText(describe(self.rows, self.annotations, int(row), int(number),
                                    self.coverage))


__all__ = ["PER_LINE_CHOICES", "SequenceRow", "SequenceView", "coverage_of", "depth",
           "describe", "fragment_rows", "pooled_coverage_of", "project_sequences", "residues_of",
           "sequence_html", "stored_sequences"]
