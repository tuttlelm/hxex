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
``find_fragment_peak`` would give in that file only; the spectra stay in their series. A row is
for one sample: a file read by several samples (controls shared by the samples of a state) needs
a row for each, which the window's Save writes (one injection, one window; owner, 2026-11-11).
A table without a ``peptide`` column (written before peptide rows existed) holds ion rows only.
"""

from __future__ import annotations

import math
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


# --- the table written by the window's review (moved here from hxex.gui.window_edit, audit
# of 2026-11-15, Q-4: the rules of the table belong to the library)

DEFAULT_NAME = "rt_elution_windows.csv"  # beside the project file, when it names no table

# what was read from the project file and the table, kept until the file changes on disk or
# this module writes it (audit of 2026-11-15, F-12: every spectrum shown with Edit window on
# loaded the project file three times and read the whole table, 0.2 s on a 12,000-row table)
_CACHE: dict[tuple[str, Path], tuple[tuple, object]] = {}


def file_signature(path: Path) -> tuple | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _cached(kind: str, path: Path, make):
    """``make()`` for ``path``, kept under ``kind`` while the file's size and modification
    time stay (None: the file is not there; made every time)."""
    sig = file_signature(path)
    key = (kind, path)
    if sig is None:
        _CACHE.pop(key, None)
        return make()
    kept = _CACHE.get(key)
    if kept is not None and kept[0] == sig:
        return kept[1]
    value = make()
    _CACHE[key] = (sig, value)
    return value


def forget_cached(path: str | Path | None = None) -> None:
    """Drop what was read from ``path`` (None: everything): this module writes a file, or a
    test replaces one in place."""
    if path is None:
        _CACHE.clear()
        return
    path = Path(path)
    for key in [k for k in _CACHE if k[1] == path]:
        del _CACHE[key]


@dataclass(frozen=True)
class SavedWindow:
    """A window written to the project's ``[input] rt_elution_windows`` table: each file
    written with its bytes ``before`` (None: it did not exist) and ``after`` (Undo writes
    ``before`` back), the row as written (``row``, by column; the peptide's row for a window
    saved for the whole peptide), whether it was ``added`` (else that row was changed), and,
    for a peptide's window, ``own``: the other ions of the peptide whose rows in that file keep
    a window of their own. The table is the first file; the project file follows when the first
    save named the table in it."""

    files: tuple[tuple[Path, bytes | None, bytes], ...]
    row: dict[str, str]
    added: bool
    own: tuple[str, ...] = ()
    samples: tuple[str, ...] = ()  # every sample a row was written for (the key's first)
    project: Path | None = None  # the project file the window was saved for

    @property
    def path(self) -> Path:
        return self.files[0][0]


def _text(value: float, places: int, up: bool) -> str:
    """``value`` to ``places`` decimals, rounded outwards (``up``: towards larger), so that a
    window written to the table still holds the scans it summed."""
    scale = 10 ** places
    v = (math.ceil if up else math.floor)(value * scale - (1e-6 if up else -1e-6)) / scale
    return f"{v:.{places}f}"


def table_path(project_path: str | Path) -> tuple[Path, bool]:
    """The project's elution-window table and whether the project file names it already (else
    the default name beside the project file, named there by the first save)."""
    from ..config import load_project

    project_path = Path(project_path)
    named = _cached("table_path", project_path,
                    lambda: load_project(project_path).input.rt_elution_windows)
    return (Path(named), True) if named is not None else (project_path.parent / DEFAULT_NAME,
                                                          False)


def own_windows(project_path, peptide: str, ions: Sequence[str],
                places: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Which ``places`` (sample, file) already hold a window of the peptide's own in the
    project's elution-window table: a row naming the peptide, or one of its ``ions``, with a
    window or a drift range (set by hand or by an earlier save); a spread asks before replacing
    them (owner, 2026-11-11)."""
    from .user_tables import read_table

    path, _named = table_path(project_path)
    if not path.is_file():
        return []
    table = read_table(path)
    names = [c.strip().lower() for c in table.columns]
    if not {"sample", "file"} <= set(names):
        return []

    def cell(r, column):
        return r[names.index(column)].strip() if column in names else ""

    wanted, out = set(places), []
    mine = set(ions)
    for r in table.rows:
        place = (cell(r, "sample"), cell(r, "file"))
        if place not in wanted or place in out:
            continue
        pep, ion = cell(r, "peptide"), cell(r, "ion")
        if not (pep == peptide or (ion in mine and pep in ("", peptide))):
            continue
        if any(cell(r, c) for c in ("rt_start", "rt_end", "drift_start", "drift_end")):
            out.append(place)
    return out


def sibling_windows(project_path, peptide: str, ion: str, siblings: Sequence[str],
                    places: Sequence[tuple[str, str]], shared: bool = False) -> list[str]:
    """The ions of ``siblings`` other than ``ion`` (the one edited) with a retention window of
    their own in ``places`` (sample, file: the edited file, for each sample reading it) of the
    project's elution-window table: a save for the whole peptide replaces them, so the window
    asks first (audit of 2026-11-15, D-8). A row naming the ion without a peptide counts unless
    another peptide has an ion of that name (``shared``)."""
    from .user_tables import read_table

    path, _named = table_path(project_path)
    if not path.is_file():
        return []
    table = read_table(path)
    names = [c.strip().lower() for c in table.columns]
    if not {"sample", "file", "ion", "rt_start"} <= set(names):
        return []

    def cell(r, column):
        return r[names.index(column)].strip() if column in names else ""

    wanted, others = set(places), {s for s in siblings if s != ion}
    out: dict[str, None] = {}
    for r in table.rows:
        if (cell(r, "sample"), cell(r, "file")) not in wanted or cell(r, "ion") not in others:
            continue
        if cell(r, "peptide") == peptide or (not shared and not cell(r, "peptide")):
            if cell(r, "rt_start") or cell(r, "rt_end"):
                out.setdefault(cell(r, "ion"))
    return list(out)


def save_window(project_path: str | Path, key: tuple[str, str, str],
                window: tuple[float, float], drift: tuple[float, float] | None = None, *,
                peptide: str | None = None, whole: bool = False, siblings=(),
                shared: bool = False, table: str | Path | None = None,
                samples: Sequence[str] = (),
                more: Sequence[tuple[str, str, tuple[float, float]]] = (),
                keep_own: bool = False) -> SavedWindow:
    """Write ``window`` (minutes: the first and last scan of the spectrum summed again over it)
    to the project's ``[input] rt_elution_windows`` table for ``key`` (sample, file name, ion
    as the spectrum ids name it) of ``peptide`` (as :func:`peptide_label` names the ion's peptide).

    ``whole``: the window is the peptide's row in that file: every ion of it takes it (its
    charge states and fragments; the same molecules elute together), and the ion's own row
    gives up its window (it is removed, or keeps only its drift range; a row naming the ion
    without a peptide only when no other peptide has an ion of that name, ``shared`` False).
    ``siblings`` (the ions of the peptide, as spectrum ids name them) give up their windows in
    that file too, as in the files the window is spread to (the window asks first,
    :func:`sibling_windows`), unless ``keep_own``: then those keeping one are named in
    :attr:`SavedWindow.own`. ``drift`` (ms) is then written to the ion's own row only (drift
    times differ between charge states): give it only when it was changed. Otherwise the row is
    the ion's own, naming its peptide too (two parents can give fragment ions of the same
    name), with ``drift``: an exception, such as interference at its own m/z. A row naming the
    ion without a peptide is that row (unless ``shared``), and names the peptide from then on.

    ``samples``: the other samples that read the same file (:func:`samples_reading`; one
    injection, so one window, owner 2026-11-11): the same rows are written for each of them, as
    the table keys its rows by sample. ``more``: (sample, file, window) of other files the window
    is spread to (:func:`plan_spread`, shifted by each file's elution), written as ``whole``
    rows are (the peptide's row; ``drift`` on the ion's own row), and there every ion of the
    peptide gives up a window of its own (the window asks first, :func:`own_windows`).
    ``drift`` None leaves the drift range an ion's own row holds as it is (a save that moved
    only the retention window does not pin the reader's drift choice; audit of 2026-11-15, D
    open question 3).

    The row is changed, or one is added. When the project file names no table, the first save
    writes ``table`` (the window asks where: :func:`hxex.gui.table_choice.ask_table`; else
    :data:`DEFAULT_NAME` beside the project file), making it or adding to it, and names it in
    the project file. The
    times are written rounded outwards (4 decimals of a minute, 3 of a millisecond), so the run
    reads the same scans. The table is checked as the project reads it before anything is
    written (ValueError with the reader's words; nothing written). The peptide table is never
    changed."""
    from ..config import toml_value
    from ..project_edit import relative_path, set_settings
    from .user_tables import Table, read_table, table_problem, write_table

    if whole and not peptide:
        raise ValueError("a window for the whole peptide needs the peptide")
    project_path = Path(project_path)
    path, named = table_path(project_path)
    if not named and table is not None:
        path = Path(table)
    before = path.read_bytes() if path.is_file() else None
    table = read_table(path) if before is not None else Table(list(COLUMNS))
    table = table.copy()
    names = [c.strip().lower() for c in table.columns]
    for column in COLUMNS:
        if column not in names:
            table.columns.append(column)
            names.append(column)
            for r in table.rows:
                r.append("")
    at = {c: names.index(c) for c in COLUMNS}
    first, file, ion = key
    pep = peptide or ""
    written = tuple(dict.fromkeys([first, *samples]))

    def cell(r, column):
        return r[at[column]].strip()

    def span(row, values, places, start, end):
        row[at[start]] = "" if values is None else _text(values[0], places, up=False)
        row[at[end]] = "" if values is None else _text(values[1], places, up=True)

    def write_for(sample, file=file, window=window, everyone=False):
        def find(pep_value, ion_value):
            return [r for r in table.rows if (cell(r, "sample"), cell(r, "file"),
                                              cell(r, "peptide"), cell(r, "ion"))
                    == (sample, file, pep_value, ion_value)]

        def put(pep_value, ion_value):
            own = find(pep_value, ion_value)
            if not own and pep_value and ion_value and not shared:
                # the ion's row naming no peptide (written before peptide rows existed, or
                # from an HDExaminer export) is its row, as the reader takes it: it names its
                # peptide from now on (audit of 2026-11-15, D-4)
                own = find("", ion_value)
                if own:
                    own[0][at["peptide"]] = pep_value
            if own:
                return own[0], False
            row = [""] * len(table.columns)
            row[at["sample"]], row[at["file"]] = sample, file
            row[at["peptide"]], row[at["ion"]] = pep_value, ion_value
            table.rows.append(row)
            return row, True

        own_rows: set[str] = set()
        if whole:
            row, added = put(pep, "")
            span(row, window, 4, "rt_start", "rt_end")
            mine = find(pep, ion) + ([] if shared else find("", ion))
            followers = mine + ([r for s_ in siblings if s_ != ion
                                 for r in find(pep, s_) + ([] if shared else find("", s_))]
                                if everyone else [])
            for r in followers:  # the ion (unless its siblings keep theirs, every ion)
                # follows its peptide
                span(r, None, 4, "rt_start", "rt_end")
            if drift is not None:
                target, _ = (mine[0], False) if mine else put(pep, ion)
                span(target, drift, 3, "drift_start", "drift_end")
            table.rows[:] = [r for r in table.rows if not cell(r, "ion") or any(
                cell(r, c) for c in ("rt_start", "rt_end", "drift_start", "drift_end"))]
            own_rows = {cell(r, "ion") for s_ in siblings for r in find(pep, s_) + find("", s_)
                        if cell(r, "rt_start") and (s_ != ion or r not in mine)}
        else:
            row, added = put(pep, ion)
            span(row, window, 4, "rt_start", "rt_end")
            if drift is not None:  # else the row's own drift range stays (D open question 3)
                span(row, drift, 3, "drift_start", "drift_end")
        return row, added, own_rows

    # the edited file's other ions follow the peptide's row too, as in the files it is spread
    # to (audit of 2026-11-15, D-8), unless the user chose to keep theirs (``keep_own``)
    row, added, own_set = write_for(first, everyone=whole and not keep_own)
    for sample in written[1:]:
        own_set |= write_for(sample, everyone=whole and not keep_own)[2]
    if more and not (whole and peptide):
        raise ValueError("a window spread to other files is the whole peptide's")
    for sample, other_file, other_window in more:
        write_for(sample, other_file, other_window, everyone=True)
    own_rows = tuple(sorted(own_set))
    problem = table_problem("input.rt_elution_windows", table, project_path)
    if problem:
        raise ValueError(problem)
    project_before = project_path.read_bytes()
    write_table(table, path)
    forget_cached(path)
    files = [(path, before, path.read_bytes())]
    if not named:
        try:
            shown = relative_path(path, project_path.parent)  # forward slashes when inside
            set_settings(project_path, [("input.rt_elution_windows", toml_value(shown))],
                         unfinished=True)
        except Exception:
            _restore(path, before)
            raise
        forget_cached(project_path)
        files.append((project_path, project_before, project_path.read_bytes()))
    return SavedWindow(tuple(files), {c: v for c, v in zip(table.columns, row, strict=True)},
                       added, own_rows, written, project=project_path.absolute())


def _restore(path: Path, before: bytes | None) -> None:
    from ..project_edit import replace_file

    forget_cached(path)
    if before is None:
        path.unlink(missing_ok=True)
        return
    tmp = path.with_name(path.name + ".hxex-tmp")
    tmp.write_bytes(before)
    replace_file(tmp, path)


def undo_window(saved: SavedWindow) -> None:
    """Write the files ``saved`` changed back as they were (the table removed when the save
    made it; ValueError when one was changed since, by hand or by another program: nothing is
    written)."""
    for path, _before, after in saved.files:
        if not path.is_file() or path.read_bytes() != after:
            raise ValueError(f"{path} was changed since the window was saved: not undone")
    for path, before, _after in saved.files:
        _restore(path, before)


def saved_window(project_path: str | Path, key: tuple[str, str, str],
                 peptide: str | None = None
                 ) -> tuple[tuple[float, float], tuple[float, float] | None, str] | None:
    """The window (minutes), drift range (ms, or None) and where the window comes from
    (``"ion"``: the ion's own row; ``"peptide"``: its peptide's row) that the project's
    elution-window table gives ``key`` (sample, file, ion) of ``peptide`` (as
    :func:`peptide_label` names it); None without a table or a row with
    a window for it."""
    path, _named = table_path(project_path)
    if not path.is_file():
        return None
    rows = _cached("rows", path, lambda: read_elution_windows(path))
    sample, file, ion = key
    window, drift, used = window_for(ion, peptide or "", sample, file, by_key(rows),
                                     by_peptide(rows))
    if window is None:
        return None
    return window, drift, ("peptide" if used[0].is_peptide else "ion")
