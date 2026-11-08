# SPDX-License-Identifier: GPL-3.0-or-later
"""Retention windows edited for one ion in one file: the ``[input] rt_elution_windows`` table
(PLAN Next steps 26).

The peptide table says what a project reads, in every file alike; this table holds the
exceptions, one row per sample, file, and ion: the retention window (minutes, the acquisition
clock) that ion's spectrum sums in that file, and, for files with ion mobility bins, its drift
range (ms). The window's elution tab writes it (Edit window, Save); it can be edited by hand. The
ion is named as the spectrum ids name it (``2-13-ASKELVDPLRKE-z1-ion_type:c-1@ETD-pz3-ETD``: the
part between the file and the scans), so a row can be for an MS1 peptide, a fragment, or a
fragment channel's intact precursor. A row takes the place of the window the peptide table or
``find_peak`` / ``find_fragment_peak`` would give that ion in that file only; the spectrum stays
in its series.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ._common import not_text_error, open_text
from .entries import _rows, _table_reader
from .errors import ReadError

COLUMNS = ("sample", "file", "ion", "rt_start", "rt_end", "drift_start", "drift_end")
REQUIRED = ("sample", "file", "ion", "rt_start", "rt_end")


@dataclass(frozen=True)
class ElutionWindow:
    """One row: the ``sample``, the ``file`` (its name, as the run's ``path`` ends), the ``ion``
    (as spectrum ids name it), the window (``rt_start``, ``rt_end``, minutes) and the drift
    range (``drift_start``, ``drift_end``, ms; None: as the reader would choose it)."""

    sample: str
    file: str
    ion: str
    rt_start: float
    rt_end: float
    drift_start: float | None = None
    drift_end: float | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.sample, self.file, self.ion

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
    """The rows of the table at ``path``. A row needs its sample, file, ion, rt_start and
    rt_end (minutes, the end after the start); drift_start and drift_end (ms) come both or
    neither. One sample, file, and ion twice is refused (ReadError, with the line)."""
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
            if missing:
                raise ReadError(f"{path}: no {', '.join(missing)} column; a row names the "
                                "sample, the file and the ion (as the spectrum ids name it) "
                                "with the window's rt_start and rt_end (minutes)")
            for line, row in _rows(reader, path):
                cells = {k: str(row.get(k) or "").strip() for k in COLUMNS}
                if not any(cells.values()):
                    continue
                try:
                    for name in REQUIRED:
                        if not cells[name]:
                            raise ValueError(f"no {name}")
                    start = _number(cells["rt_start"], "rt_start", "minutes")
                    end = _number(cells["rt_end"], "rt_end", "minutes")
                    if not start < end:
                        raise ValueError(f"rt_end {end:g} must be after rt_start {start:g}")
                    drift = (cells["drift_start"], cells["drift_end"])
                    if any(drift) and not all(drift):
                        raise ValueError("give drift_start and drift_end both, or neither")
                    lo = hi = None
                    if all(drift):
                        lo = _number(drift[0], "drift_start", "milliseconds")
                        hi = _number(drift[1], "drift_end", "milliseconds")
                        if not lo <= hi:
                            raise ValueError(f"drift_end {hi:g} must not be before "
                                             f"drift_start {lo:g}")
                except ValueError as exc:
                    raise ReadError(f"{path} line {line}: {exc}") from None
                window = ElutionWindow(cells["sample"], cells["file"], cells["ion"], start,
                                       end, lo, hi)
                if window.key in seen:
                    raise ReadError(f"{path} line {line}: {window.ion} in {window.file} "
                                    f"({window.sample}) has a window already (line "
                                    f"{seen[window.key]}); keep one row")
                seen[window.key] = line
                rows.append(window)
    except (OSError, UnicodeDecodeError) as exc:
        raise not_text_error(path, exc) from exc
    return rows


def by_key(windows: Sequence[ElutionWindow] | None) -> dict[tuple[str, str, str],
                                                             ElutionWindow]:
    """The rows by (sample, file, ion)."""
    return {w.key: w for w in windows or ()}
