# SPDX-License-Identifier: GPL-3.0-or-later
"""Retention windows edited by hand: the ``[input] rt_elution_windows`` table (PLAN Next
steps 26).

The peptide table says what a project reads, in every file alike; this table holds the
exceptions, per sample and file: the retention window (minutes, the acquisition clock) the
spectra sum in that file, and, for files with ion mobility bins, a drift range (ms). The window's
elution tab writes it (Edit window, Save); it can be edited by hand.

A row names either a **peptide** or an **ion**:

- ``peptide`` (its range and sequence, with its modification when it has one, as
  :func:`peptide_label` gives it: ``2-13-ASKELVDPLRKE``): every ion of that peptide in that file
  takes the window: its charge states, and, with ``[[input.fragments]]``, the peptide as a parent
  and every fragment ion read from its scans. They are the same molecules, which elute together
  (charge states within 0.4 s, fragments within 0.4 s of their parent on the data measured, docs
  HISTORY), so one window is the physical choice. A peptide row gives no drift range: drift
  times differ between charge states.
- ``ion`` (as the spectrum ids name it: ``2-13-ASKELVDPLRKE-z1-ion_type:c-1@ETD-pz3-ETD``, the
  part between the file and the scans): that ion alone, an exception such as interference at
  its own m/z. Its window takes the place of its peptide's; a row with only a drift range keeps
  the peptide's window (or the one the reader would choose) and sets the ion's drift range.
  With ``peptide`` as well, the row is for that ion of that peptide only: two parents that
  share their N-terminus (or C-terminus) give fragment ions of the same name at the same
  precursor charge (on the private targeted dataset 174 ion names belonged to two parents),
  which elute with their own parents. An ion row without a peptide is for the ion of every
  peptide that has one of that name.

A row takes the place of the window the peptide table or ``find_peak`` /
``find_fragment_peak`` would give in that file only; the spectra stay in their series. A table
without a ``peptide`` column (written before peptide rows existed) holds ion rows only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ._common import not_text_error, open_text
from .entries import _rows, _table_reader
from .errors import ReadError

COLUMNS = ("sample", "file", "peptide", "ion", "rt_start", "rt_end", "drift_start", "drift_end")
REQUIRED = ("sample", "file", "rt_start", "rt_end")


def peptide_label(entry) -> str:
    """The peptide an entry belongs to, as a ``peptide`` row names it: a fragment's parent
    (its range and sequence), else the entry's own range and sequence, followed by its
    modification when it has one other than ``ion_type:M`` (spaces as underscores, as in the
    spectrum ids)."""
    extra = dict(entry.extra)
    if extra.get("parent_peptide"):
        return f"{extra.get('parent_range')}-{extra['parent_peptide']}"
    mod = [m for m in str(entry.modification or "").split() if m != "ion_type:M"]
    return f"{entry.peptide_range}-{entry.peptide}" + (f"-{'_'.join(mod)}" if mod else "")


@dataclass(frozen=True)
class ElutionWindow:
    """One row: the ``sample``, the ``file`` (its name, as the run's ``path`` ends), what it is
    for (``peptide``, as :func:`peptide_label` names it, and/or ``ion``, as spectrum ids name
    it: a peptide alone is every ion of it; an ion is that ion, of that peptide when one is
    given), the window (``rt_start``, ``rt_end``, minutes; None on an ion row that
    gives only a drift range) and the drift range (``drift_start``, ``drift_end``, ms; None: as
    the reader would choose it; ion rows only)."""

    sample: str
    file: str
    ion: str | None
    rt_start: float | None
    rt_end: float | None
    drift_start: float | None = None
    drift_end: float | None = None
    peptide: str | None = None

    @property
    def key(self) -> tuple[str, str, str, str]:
        """(sample, file, peptide, ion), empty strings for what the row does not name."""
        return self.sample, self.file, self.peptide or "", self.ion or ""

    @property
    def is_peptide(self) -> bool:
        """A row for every ion of a peptide (a peptide and no ion)."""
        return bool(self.peptide) and not self.ion

    @property
    def name(self) -> str:
        """What the row is for, in words: ``peptide X``, the ion, or ``ION of peptide X``."""
        if self.is_peptide:
            return f"peptide {self.peptide}"
        return str(self.ion) + (f" of peptide {self.peptide}" if self.peptide else "")

    @property
    def window(self) -> tuple[float, float] | None:
        return None if self.rt_start is None else (self.rt_start, self.rt_end)

    @property
    def drift(self) -> tuple[float, float] | None:
        return None if self.drift_start is None else (self.drift_start, self.drift_end)

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in COLUMNS}

    @classmethod
    def from_dict(cls, data: Mapping) -> ElutionWindow:
        return cls(**{name: data.get(name) for name in COLUMNS})


def _number(text: str, name: str, unit: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{name} {text!r} is not a number of {unit}") from None


def read_elution_windows(path: str | Path) -> list[ElutionWindow]:
    """The rows of the table at ``path``. A row needs its sample and file, and a peptide, an
    ion, or both (that ion of that peptide). A peptide row needs rt_start and rt_end (minutes,
    the end after the start) and gives no drift range; an ion row needs rt_start and rt_end,
    or drift_start and drift_end (ms), or both pairs (each pair both or neither). One row
    twice for the same sample, file, peptide, and ion is refused (ReadError, with the
    line)."""
    path = Path(path)
    rows: list[ElutionWindow] = []
    seen: dict[tuple, int] = {}
    try:
        with open_text(path, newline="") as fh:
            reader = _table_reader(fh, path)
            # its own columns by name, without regard to case or surrounding spaces
            reader.fieldnames = [str(n or "").strip().lower() for n in reader.fieldnames or []]
            names = reader.fieldnames
            missing = [c for c in REQUIRED if c not in names]
            if "peptide" not in names and "ion" not in names:
                missing.append("peptide or ion")
            if missing:
                raise ReadError(f"{path}: no {', '.join(missing)} column; a row names the "
                                "sample, the file, and the peptide (every ion of it in that "
                                "file) or one ion (as the spectrum ids name it), with the "
                                "window's rt_start and rt_end (minutes)")
            for line, row in _rows(reader, path):
                cells = {k: str(row.get(k) or "").strip() for k in COLUMNS}
                if not any(cells.values()):
                    continue
                try:
                    for name in ("sample", "file"):
                        if not cells[name]:
                            raise ValueError(f"no {name}")
                    if not cells["peptide"] and not cells["ion"]:
                        raise ValueError("give a peptide (every ion of it in that file) or an "
                                         "ion (that ion only)")
                    span = (cells["rt_start"], cells["rt_end"])
                    drift = (cells["drift_start"], cells["drift_end"])
                    if any(span) and not all(span):
                        raise ValueError("give rt_start and rt_end both, or neither")
                    if any(drift) and not all(drift):
                        raise ValueError("give drift_start and drift_end both, or neither")
                    if cells["peptide"] and not cells["ion"]:
                        if not all(span):
                            raise ValueError("a peptide row needs rt_start and rt_end")
                        if any(drift):
                            raise ValueError("a peptide row gives no drift range (drift times "
                                             "differ between charge states): give it on a row "
                                             "for the ion")
                    elif not all(span) and not all(drift):
                        raise ValueError("an ion row needs rt_start and rt_end, or drift_start "
                                         "and drift_end")
                    start = end = lo = hi = None
                    if all(span):
                        start = _number(span[0], "rt_start", "minutes")
                        end = _number(span[1], "rt_end", "minutes")
                        if not start < end:
                            raise ValueError(f"rt_end {end:g} must be after rt_start {start:g}")
                    if all(drift):
                        lo = _number(drift[0], "drift_start", "milliseconds")
                        hi = _number(drift[1], "drift_end", "milliseconds")
                        if not lo <= hi:
                            raise ValueError(f"drift_end {hi:g} must not be before "
                                             f"drift_start {lo:g}")
                except ValueError as exc:
                    raise ReadError(f"{path} line {line}: {exc}") from None
                window = ElutionWindow(cells["sample"], cells["file"], cells["ion"] or None,
                                       start, end, lo, hi, cells["peptide"] or None)
                if window.key in seen:
                    raise ReadError(f"{path} line {line}: {window.name} in {window.file} "
                                    f"({window.sample}) has a window already (line "
                                    f"{seen[window.key]}); keep one row")
                seen[window.key] = line
                rows.append(window)
    except (OSError, UnicodeDecodeError) as exc:
        raise not_text_error(path, exc) from exc
    return rows


def by_key(windows: Sequence[ElutionWindow] | None) -> dict[tuple[str, str, str, str],
                                                             ElutionWindow]:
    """The ion rows by (sample, file, peptide or "", ion)."""
    return {w.key: w for w in windows or () if not w.is_peptide}


def by_peptide(windows: Sequence[ElutionWindow] | None) -> dict[tuple[str, str, str],
                                                                 ElutionWindow]:
    """The peptide rows by (sample, file, peptide)."""
    return {(w.sample, w.file, w.peptide): w for w in windows or () if w.is_peptide}


def window_for(entry_ion: str, entry_peptide: str, sample: str, file: str,
               ions: Mapping, peptides: Mapping
               ) -> tuple[tuple[float, float] | None, tuple[float, float] | None, list]:
    """The window (minutes) and drift range (ms) the table gives an ion (``entry_ion``, of
    peptide ``entry_peptide``) in one file, from its ion row (``ions``, :func:`by_key`: the row
    naming its peptide, else the one naming the ion alone) and its peptide's row
    (``peptides``, :func:`by_peptide`), and the rows used: the ion row's window when it gives
    one, else the peptide's; the drift range from the ion row only. (None, None, []) when the
    table says nothing about it."""
    ion = ions.get((sample, file, entry_peptide, entry_ion))
    if ion is None:
        ion = ions.get((sample, file, "", entry_ion))
    pep = peptides.get((sample, file, entry_peptide))
    window = drift = None
    used = []
    if ion is not None and ion.window is not None:
        window = ion.window
        used.append(ion)
    elif pep is not None:
        window = pep.window
        used.append(pep)
    if ion is not None and ion.drift is not None:
        drift = ion.drift
        if ion not in used:
            used.append(ion)
    return window, drift, used
