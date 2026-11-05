# SPDX-License-Identifier: GPL-3.0-or-later
"""The Project workspace's sequence view (owner, 2026-11-04: "a nice looking sequence viewer"):
the protein sequence of each sample as the analysis numbers it, in blocks of ten with a ruler,
one row per distinct sequence (samples with the same one share a row), a residue that differs
from the first row's highlighted, prolines (no amide to exchange) in grey, the regions of
``[[analysis.regions]]`` as bars under the residues and the sites of ``[analysis.sites]``
marked. A residue clicked says what it is. Basic to start (owner): coverage comes later.

The sequences are those the analysis reads: ``[analysis] fasta`` (one file, or one per sample),
else, for SpecExport, the ``SAMPLE.fasta`` files beside the sample folders; a FASTA numbered by
its header (``>NAME/START-END``) starts at START. The files come first; a SpecExport run also
keeps each sample's sequence in its bundle (the read record's ``sequences``), shown when the
files are not found, and a file that no longer matches it is said (owner, 2026-11-04).
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

PER_LINE = 50  # residues a line: five blocks of ten
REGION_COLOURS = ("#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#76b7b2", "#edc948", "#9c755f")
DIFFERS, PROLINE, SITE = "#ffd54f", "#9e9e9e", "#c62828"


@dataclass
class SequenceRow:
    """One distinct sequence: the ``samples`` that have it, its ``residues`` (modX labels) and
    the number of its ``first`` residue."""

    samples: list[str]
    residues: tuple[str, ...]
    first: int = 1
    differs: set[int] = field(default_factory=set)  # residue numbers unlike the first row's

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


def project_sequences(project: Path) -> tuple[list[SequenceRow], object, str]:
    """The rows of the project's sequences, its annotations, and a message: why there is no
    sequence when there are no rows, else what to say of where they came from ('' when the
    files were read as they are)."""
    from ..analysis.compare import Annotations
    from ..config import load_project
    from ..io import read_fasta_numbered
    from ..project_edit import live_settings

    notes: list[str] = []
    try:
        loaded = load_project(project, analysis_only=True)
        config = loaded.analysis
    except Exception as exc:  # noqa: BLE001 - said in the view
        return [], Annotations(), f"The project file does not read: {exc}"
    stored = stored_sequences(loaded.output.bundle)
    annotations = config.annotations if config is not None else Annotations()
    found: dict[str, tuple[tuple[str, ...], int]] = {}
    if config is not None and config.fasta is not None:
        try:
            made = config.sequences(project.parent)
        except Exception as exc:  # noqa: BLE001 - said in the view
            return [], annotations, f"[analysis] fasta does not read: {exc}"
        if isinstance(made, dict):
            for name, value in made.items():
                if isinstance(value, dict):  # a run's samples
                    for sample, seq in value.items():
                        found[f"{name}:{sample}"] = (residues_of(seq), getattr(seq, "first", 1))
                else:
                    found[name] = (residues_of(value), getattr(value, "first", 1))
        else:
            found["every sample"] = (residues_of(made), getattr(made, "first", 1))
    else:
        values, _ = live_settings(project.read_text(encoding="utf-8-sig"))
        if str(values.get("input.format", "")).strip('"') == "specexport":
            folder = Path(str(values.get("input.data_dir", "")).strip('"'))
            folder = folder if folder.is_absolute() else project.parent / folder
            if folder.is_dir():
                for path in sorted(p for p in folder.iterdir()
                                   if p.suffix.lower() == ".fasta" and p.is_file()):
                    try:
                        seq, first = read_fasta_numbered(path)
                    except Exception as exc:  # noqa: BLE001 - one file: said, the rest shown
                        log.warning("%s does not read: %s", path.name, exc)
                        continue
                    found[path.stem] = (residues_of(seq), first)
                    kept = stored.get(path.stem)
                    if kept is not None and (residues_of(kept[0]), kept[1]) != (
                            residues_of(seq), first):
                        notes.append(f"{path.name} differs from the copy the run stored in its "
                                     "bundle (the file is shown; rerun to read the peptides "
                                     "against it)")
    if not found and stored:  # the files gone: the run's copies
        found = {name: (residues_of(seq), first) for name, (seq, first, _) in stored.items()}
        notes.append("The FASTA files are not found: the sequences the run stored in its bundle "
                     "are shown.")
    if not found:
        return [], annotations, ("No sequence: name the FASTA file(s) in [analysis] fasta (or, "
                                 "for SpecExport data, put SAMPLE.fasta beside each sample's "
                                 "folder).")
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
    base = rows[0]
    for row in rows[1:]:
        row.differs = {n for n in range(row.first, row.last + 1)
                       if base.at(n) and row.at(n) and row.at(n) != base.at(n)}
    return rows, annotations, " ".join(notes)


def sequence_html(rows: list[SequenceRow], annotations, per_line: int = PER_LINE) -> str:
    """The sequences as HTML for Qt's text view: monospace lines of ``per_line`` residues in
    blocks of ten, a ruler above each line, each row named on the left, every residue a link
    (``res:<row>:<number>``) for the view to say what it is."""
    if not rows:
        return ""
    first = min(r.first for r in rows)
    last = max(r.last for r in rows)
    label = max(len(", ".join(r.samples)) for r in rows)
    label = min(max(label, 6), 28)
    regions = list(getattr(annotations, "regions", ()))
    sites = {s.residue: s.label for s in getattr(annotations, "sites", ())}
    colour = {id(reg): REGION_COLOURS[i % len(REGION_COLOURS)] for i, reg in enumerate(regions)}
    out = ['<div style="font-family: monospace; font-size: 11pt; white-space: pre;">']
    if regions or sites:
        keys = [f'<span style="color:{colour[id(r)]};">━━</span> {html.escape(r.name)} '
                f'({r.start}–{r.end})' for r in regions]
        if sites:
            keys.append(f'<span style="color:{SITE};">▲</span> site')
        out.append('<span style="font-family: sans-serif; font-size: 9pt;">'
                   + "    ".join(keys) + "</span>\n\n")
    start = first - ((first - 1) % per_line)  # lines begin at 1, 51, 101, ...
    for line_start in range(start, last + 1, per_line):
        numbers = range(line_start, min(line_start + per_line, last + 1))  # not past the end
        out.append(_ruler(numbers, label) + "\n")
        for k, row in enumerate(rows):
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
        bars = _region_line(numbers, regions, colour, start)
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


def _region_line(numbers: range, regions, colour: dict, start: int) -> str:
    """Bars under the residues of each region (the first region's colour where they overlap)."""
    cells = []
    for n in numbers:
        hit = next((r for r in regions if r.start <= n <= r.end), None)
        cells.append(f'<span style="color:{colour[id(hit)]};">━</span>' if hit else " ")
        if (n - start + 1) % 10 == 0:
            cells.append(" ")
    return "".join(cells)


def describe(rows: list[SequenceRow], annotations, row: int, number: int) -> str:
    """What a residue clicked is: its sample(s), number and residue, how the other rows differ
    there, and the regions and site it belongs to."""
    here = rows[row]
    parts = [f"{', '.join(here.samples)}: residue {number}, {here.at(number)}"]
    if here.at(number).endswith("P"):
        parts.append("a proline (no amide hydrogen to exchange)")
    others = [f"{', '.join(r.samples)} {r.at(number) or '(none)'}" for i, r in enumerate(rows)
              if i != row and r.at(number) != here.at(number)]
    if others:
        parts.append("differs: " + "; ".join(others))
    regions = [f"{r.name} ({r.start}–{r.end})" for r in getattr(annotations, "regions", ())
               if r.start <= number <= r.end]
    if regions:
        parts.append("region " + ", ".join(regions))
    site = next((s for s in getattr(annotations, "sites", ()) if s.residue == number), None)
    if site is not None:
        parts.append(f"site {site.label}".rstrip())
    return "; ".join(parts)


try:  # the widget (the module's functions above import without Qt)
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget
except ImportError:  # pragma: no cover - the gui extra is not installed
    QWidget = object  # type: ignore[assignment,misc]


class SequenceView(QWidget):  # type: ignore[misc,valid-type]
    """The sequences of the project shown (:meth:`show_project`), a residue clicked described."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list[SequenceRow] = []
        self.annotations = None
        self.project: Path | None = None
        self.text = QTextBrowser()
        self.text.setOpenLinks(False)
        self.text.anchorClicked.connect(self._clicked)
        self.about = QLabel("Click a residue to see what it is.")
        self.about.setWordWrap(True)
        refresh = QPushButton("Read again")
        refresh.setToolTip("read the FASTA files again (after editing one)")
        refresh.clicked.connect(lambda: self.show_project(self.project))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.text, 1)
        layout.addWidget(self.about)
        layout.addWidget(refresh)

    def show_project(self, project: Path | None) -> None:
        self.project = project
        if project is None:
            self.rows, self.annotations = [], None
            self.text.setPlainText("No project file open.")
            return
        self.rows, self.annotations, message = project_sequences(project)
        if not self.rows:
            self.text.setPlainText(message)
            return
        self.text.setHtml(sequence_html(self.rows, self.annotations))
        n = sum(len(r.samples) for r in self.rows)
        self.about.setText(f"{n} sample(s), {len(self.rows)} distinct sequence(s). "
                           + (f"{message} " if message else "")
                           + "Click a residue to see what it is.")

    def _clicked(self, url: QUrl) -> None:
        text = url.toString()
        if not text.startswith("res:"):
            return
        _, row, number = text.split(":")
        self.about.setText(describe(self.rows, self.annotations, int(row), int(number)))


__all__ = ["SequenceRow", "SequenceView", "describe", "project_sequences", "residues_of",
           "sequence_html", "stored_sequences"]
