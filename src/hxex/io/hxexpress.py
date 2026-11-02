# SPDX-License-Identifier: GPL-3.0-or-later
"""HX-Express style data: one spreadsheet (.xlsx or .csv) per sample, peptide and charge.

Row 1 holds one time label per spectrum (``undeut``, ``TD``, ``60 s``, ``5 min``); each label
sits above an (m/z, intensity) column pair and the data start in row 2.
"""

from __future__ import annotations

import csv
import logging
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from ..spectra import PeptideEntry, Spectrum, time_point
from ._common import clean_spectrum, not_text_error, open_text, to_number
from .entries import filter_entries, sort_entries
from .errors import ReadError
from .times import parse_hxexpress_label

log = logging.getLogger(__name__)


def parse_hxexpress_filename(name: str) -> PeptideEntry:
    """Entry from ``SAMPLE_START-END-PEPTIDE-zCHARGE-usertext.xlsx``.

    ``SAMPLE`` may contain underscores and hyphen-free text; the text after the last
    underscore of the first field is the start residue.
    """
    labels = name.split("-")

    def fail(why: str) -> ReadError:
        return ReadError(f"cannot read sample/peptide/charge from file name {name!r} "
                         f"(expected SAMPLE_START-END-PEPTIDE-zCHARGE-text.xlsx): {why}")

    if len(labels) < 5:
        raise fail(f"expected at least five '-' separated fields, found {len(labels)}")
    if len(labels) > 5:
        raise fail("the last field (free text) cannot contain '-', nor can the sample")
    if "_" not in labels[0]:
        raise fail(f"the first field {labels[0]!r} needs SAMPLE_START (the sample, '_', "
                   "the first residue number)")
    sample, start = labels[0].rsplit("_", 1)
    end, peptide, charge = labels[1], labels[2], labels[3]
    if not start.isdigit():
        raise fail(f"START {start!r} is not a residue number (the first field is SAMPLE_START; "
                   "a terminal group such as Ac- cannot be written in the name: list the "
                   "files in a table)")
    if not end.isdigit():
        raise fail(f"END {end!r} is not a residue number")
    if not peptide:
        raise fail("the PEPTIDE field is empty")
    if not charge.startswith("z") or not charge[1:].isdigit():
        raise fail(f"the charge field {charge!r} must be 'z' and the charge (z2)")
    return PeptideEntry(
        file=name,
        sample=sample,
        peptide=peptide,
        peptide_range=f"{start}-{end}",
        start_seq=int(start),
        end_seq=int(end),
        charge=int(charge[1:]),
    )


def scan_hxexpress(
    data_dir: str | Path,
    *,
    samples: Sequence[str] | None = None,
    peptide_ranges: Sequence[str] | None = None,
    peptide_region: Sequence[int] | None = None,
) -> list[PeptideEntry]:
    """List the entries for the ``.xlsx`` and ``.csv`` files in ``data_dir`` named in the
    HX-Express way (the suffix in any case).

    Files whose names do not parse are skipped with a warning, as is a sample of ``samples``
    that no file name has.
    """
    entries = []
    for path in sorted(Path(data_dir).iterdir()):
        if path.suffix.lower() not in (".xlsx", ".csv") or not path.is_file():
            continue
        if path.name.startswith("~$"):  # Excel lock file
            continue
        try:
            entries.append(parse_hxexpress_filename(path.name))
        except ReadError as exc:
            log.warning("skipping %s: %s", path.name, exc)
    if samples is not None:
        names = sorted({e.sample for e in entries})
        for missing in sorted(set(samples) - set(names)):
            case = ("; names are case-sensitive" if any(n.lower() == missing.lower()
                                                        for n in names) else "")
            log.warning("samples: no file of sample %s in %s (the samples found, the start of "
                        "the file names, are %s%s)", missing, data_dir,
                        ", ".join(names) or "none", case)
    return sort_entries(filter_entries(entries, samples=samples, peptide_ranges=peptide_ranges,
                                       peptide_region=peptide_region))


def _has_cells(rows) -> bool:
    return any(c is not None and str(c).strip() for r in rows for c in r)


def _first_sheet(wb, path: Path, max_row: int | None = None) -> list[tuple]:
    """The rows of the workbook's first worksheet (``max_row`` of them at most). When that
    sheet is empty and another holds data, say so: only the first sheet is read."""
    rows = [tuple(r) for r in wb.worksheets[0].iter_rows(max_row=max_row, values_only=True)]
    if not _has_cells(rows) and len(wb.worksheets) > 1:
        for sheet in wb.worksheets[1:]:
            if _has_cells(sheet.iter_rows(values_only=True)):
                raise ReadError(f"{path}: the first worksheet is empty and {sheet.title!r} "
                                "holds data; hxex reads the first sheet only (move the data "
                                "there)")
    return rows


def _read_rows(path: Path) -> list[tuple]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".xlsx":
            from openpyxl import load_workbook

            wb = load_workbook(path, read_only=True, data_only=True)
            try:
                return _first_sheet(wb, path)
            finally:
                wb.close()
        if suffix == ".csv":
            with open_text(path, newline="") as fh:
                return [tuple(r) for r in csv.reader(fh)]
    except (OSError, UnicodeDecodeError) as exc:
        raise not_text_error(path, exc) from exc
    except ReadError:
        raise
    except Exception as exc:  # openpyxl raises many types for damaged files
        raise ReadError(f"cannot read {path}: {exc}") from exc
    raise ReadError(f"{path}: unsupported file type {suffix!r} (use .xlsx or .csv)")


def read_hxexpress_header(path: str | Path) -> tuple:
    """The first row of an HX-Express file (its labels), without reading the spectra."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".xlsx":
            from openpyxl import load_workbook

            wb = load_workbook(path, read_only=True, data_only=True)
            try:
                rows = _first_sheet(wb, path, max_row=1)
                return rows[0] if rows else ()
            finally:
                wb.close()
        if suffix == ".csv":
            with open_text(path, newline="") as fh:
                return tuple(next(csv.reader(fh), ()))
    except (OSError, UnicodeDecodeError) as exc:
        raise not_text_error(path, exc) from exc
    except ReadError:
        raise
    except Exception as exc:  # openpyxl raises many types for damaged files
        raise ReadError(f"cannot read {path}: {exc}") from exc
    raise ReadError(f"{path}: unsupported file type {suffix!r} (use .xlsx or .csv)")


def hxexpress_columns(header: tuple, path: Path) -> list[tuple]:
    """The spectra of an HX-Express file from its first row: ``(column, label, time, role,
    rep)`` in column order, as :func:`read_hxexpress_file` reads them (``rep`` counts the
    spectra at the same point so far)."""
    labelled = [
        (col, str(cell).strip())
        for col, cell in enumerate(header)
        if cell is not None and str(cell).strip() not in ("", "nan")
    ]
    if not labelled:
        raise ReadError(f"{path}: first row has no time labels")
    seen: Counter[str] = Counter()
    out = []
    for col, label in labelled:
        try:
            time, role = parse_hxexpress_label(label)
        except ReadError as exc:
            raise ReadError(f"{path}, column {col + 1}: {exc}") from exc
        point = str(time_point([time], [role])[0])
        seen[point] += 1
        out.append((col, label, time, role, seen[point]))
    return out


def read_hxexpress_file(
    path: str | Path,
    entry: PeptideEntry,
    *,
    data_id: int = 0,
) -> list[Spectrum]:
    """Read every spectrum in one HX-Express file.

    Spectra come in column order. ``rep`` counts the spectra at the same point so far (a time,
    or a control label: ``undeut`` columns are the undeuterated controls, ``TD`` columns the
    fully deuterated ones, with the role set and no time; :func:`parse_hxexpress_label`).
    """
    path = Path(path)
    rows = _read_rows(path)
    if not rows:
        raise ReadError(f"{path}: file is empty")
    spectra = []
    for col, label, time, role, rep in hxexpress_columns(rows[0], path):
        mz_col = [r[col] if col < len(r) else None for r in rows[1:]]
        in_col = [r[col + 1] if col + 1 < len(r) else None for r in rows[1:]]
        mz_col, in_col = _without_notes(mz_col, in_col, path, label)
        where = f"{path.name}, label {label!r}"
        mz, intensity = clean_spectrum(
            [to_number(c, where) for c in mz_col], [to_number(c, where) for c in in_col]
        )
        spectra.append(
            Spectrum(
                mz, intensity, entry, time, rep,
                data_id=data_id, label=label, spectrum_id=f"{entry.file}#{col + 1}", role=role,
            )
        )
    return spectra


def _cell_kind(cell) -> str:
    """``'empty'`` (None, blank, NaN), ``'number'`` or ``'text'`` (anything else)."""
    if cell is None:
        return "empty"
    if isinstance(cell, bool):
        return "text"
    if isinstance(cell, str):
        text = cell.strip()
        if not text or text.lower() == "nan":
            return "empty"
        try:
            float(text)
        except ValueError:
            return "text"
        return "number"
    try:
        return "empty" if math.isnan(float(cell)) else "number"
    except (TypeError, ValueError):
        return "text"


def _without_notes(mz_col: list, in_col: list, path: Path, label: str) -> tuple[list, list]:
    """The two columns of a spectrum cut before a note written under its data: a text cell
    with numbers above it and none below it in either column (a remark at the bottom of the
    sheet). Text with numbers below it is left to be refused as not a number."""
    kinds = [(_cell_kind(m), _cell_kind(y)) for m, y in zip(mz_col, in_col, strict=True)]
    first = next((i for i, k in enumerate(kinds) if "text" in k), None)
    if first is None:
        return mz_col, in_col
    above = any("number" in k for k in kinds[:first])
    below = any("number" in k for k in kinds[first + 1:])
    if not above or below:
        return mz_col, in_col
    log.debug("%s, label %r: the text from row %d down is ignored (a note below the data)",
              path.name, label, first + 2)
    return mz_col[:first], in_col[:first]
