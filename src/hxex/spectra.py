# SPDX-License-Identifier: GPL-3.0-or-later
"""Spectrum data contracts: one raw spectrum, and a set stored as flat NumPy arrays."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np


def qualify_spectrum_id(project: str, local_id: str) -> str:
    """``PROJECT:LOCAL_ID``. The project name must be non-empty and contain no ``:``."""
    if not project or ":" in project:
        raise ValueError(f"project name {project!r} must be non-empty and contain no ':'")
    return f"{project}:{local_id}"


def split_spectrum_id(spectrum_id: str) -> tuple[str, str]:
    """``(project, local_id)`` from an id made by :func:`qualify_spectrum_id`."""
    project, sep, local_id = spectrum_id.partition(":")
    if not sep or not project:
        raise ValueError(f"{spectrum_id!r} is not a PROJECT:ID spectrum id")
    return project, local_id


UNDEUT = "undeut"
FULLDEUT = "fulldeut"
ROLES = (UNDEUT, FULLDEUT)
ROLE_TEXT = "undeut or fulldeut (a control), or empty (a time point)"


def check_role(value: object, where: str = "role") -> str:
    """``value`` as a role: ``'undeut'``, ``'fulldeut'`` or ``''`` (none; None is taken as
    ``''``). Anything else raises ValueError naming ``where``."""
    if value is None:
        return ""
    if not isinstance(value, str) or value not in (*ROLES, ""):
        raise ValueError(f"{where} must be {ROLE_TEXT}, got {value!r}")
    return value


def _no_role(n: int) -> np.ndarray:
    return np.full(n, "", dtype="<U8")


def time_point(time, role=None) -> np.ndarray:
    """One hashable key per spectrum for grouping spectra measured at the same point: the
    role (``'undeut'``, ``'fulldeut'``) for a control that carries no time, else the time
    (rounded to the nanosecond, as text). Replicates share a key; a control that also carries
    a time (a time point serving as a reference) is keyed by its time, with the others
    measured then."""
    time = np.asarray(time, dtype=float)
    role = _no_role(len(time)) if role is None else np.asarray(role, dtype=str)
    out = np.empty(len(time), dtype=object)
    for i, (t, r) in enumerate(zip(time, role, strict=True)):
        out[i] = f"t={float(np.round(t, 9))!r}" if np.isfinite(t) else (str(r) or "t=nan")
    return out


def point_order_key(t: float, r: str) -> tuple[int, float]:
    """Sort key of a spectrum's point: the undeuterated control first, the times in order,
    the fully deuterated control last (a control that carries a time sorts by that time)."""
    if np.isfinite(t):
        return (1, float(t))
    return (0, 0.0) if r == UNDEUT else (2, 0.0) if r == FULLDEUT else (1, np.inf)


def time_index(data_id: np.ndarray, time: np.ndarray, role=None) -> np.ndarray:
    """Rank of each spectrum's point among the distinct points of its data set (``data_id``):
    the undeuterated control (no time) is 0, the times follow in order and the fully
    deuterated control (no time) comes last; replicates share a rank."""
    time = np.asarray(time, dtype=float)
    role = _no_role(len(time)) if role is None else np.asarray(role, dtype=str)
    out = np.zeros(len(time), dtype=np.int64)
    for d in np.unique(data_id):
        sel = np.flatnonzero(data_id == d)
        keys = [point_order_key(float(time[i]), str(role[i])) for i in sel]
        distinct = sorted(set(keys))
        rank = {k: n for n, k in enumerate(distinct)}
        out[sel] = [rank[k] for k in keys]
    return out


def ragged_mask(offsets: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Boolean mask over the flat array for the segments (spectra) where ``keep`` is True."""
    return np.repeat(keep, np.diff(offsets))


def _check_mask(keep: np.ndarray, n: int) -> np.ndarray:
    keep = np.asarray(keep)
    if keep.dtype != bool:
        raise TypeError(f"mask must be boolean (one flag per spectrum), got dtype {keep.dtype}; "
                        "use np.isin or a boolean array, not indices")
    if keep.shape != (n,):
        raise ValueError(f"mask must have one element per spectrum ({n}), got shape {keep.shape}")
    return keep


@dataclass(frozen=True)
class PeptideEntry:
    """One peptide/charge/sample to analyse: a row of pyHXExpress's ``metadf``.

    ``file`` is the HX-Express file name, or the SpecExport peptide directory name.
    ``modification`` is the pyHXExpress modification string (see :mod:`hxex.chem`). Other
    columns of a metadata table are kept as strings in ``extra``.
    """

    file: str
    sample: str
    peptide: str
    peptide_range: str
    start_seq: int
    end_seq: int
    charge: int
    modification: str = ""
    extra: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}), compare=False)

    def __post_init__(self):
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    def to_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        out["extra"] = dict(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping) -> PeptideEntry:
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown entry field(s): {', '.join(sorted(unknown))}")
        return cls(**data)


@dataclass(frozen=True)
class Spectrum:
    """One raw spectrum (m/z sorted, no missing values) and where it came from.

    ``time`` is in seconds, NaN for a control that carries none; ``role`` says what the spectrum
    is to the series: ``'undeut'`` (an undeuterated control), ``'fulldeut'`` (a fully
    deuterated control) or ``''`` (a time point; :data:`ROLES`). A control may also carry a
    time, a time point that serves as the series' reference. ``rep`` counts spectra at the
    same point within a data set; ``data_id`` is the position of the entry in the list that
    was read; ``label`` names the source for display (a column label or file name).

    ``spectrum_id`` identifies the spectrum for later reference. The file readers set the local
    part, built from where the spectrum was read, so it is unique within a data directory and
    does not depend on read order, filtering or the fully deuterated time: ``FILE#COLUMN``
    (1-based column of the label) for HX-Express, ``SAMPLE/PEPTIDE_DIR/FILE`` for SpecExport and
    ``PEPTIDE_DIR/FILE`` for pooled folders. :func:`hxex.io.read_spectra` prefixes the
    user-chosen project name (``PROJECT:LOCAL``, see :func:`qualify_spectrum_id`) so ids stay
    unique when data from several directories are combined.

    ``rt`` is the chromatographic retention time in minutes (NaN: unknown); a reader that sees
    the scans sets it, otherwise it comes from the entry's ``rt`` column (:func:`entry_rt`).
    ``resolving_power`` is the instrument's resolving power recorded with the scans, quoted at
    ``resolving_power_mz`` (both NaN when the file does not record it; Thermo raw files do).
    """

    mz: np.ndarray
    intensity: np.ndarray
    entry: PeptideEntry
    time: float
    rep: int
    data_id: int = 0
    label: str = ""
    spectrum_id: str = ""
    rt: float = float("nan")
    resolving_power: float = float("nan")
    resolving_power_mz: float = float("nan")
    role: str = ""

    def __post_init__(self):
        object.__setattr__(self, "role", check_role(self.role))


def entry_rt(entry: PeptideEntry) -> float:
    """The retention time (minutes) of a peptide table row's ``rt`` column; NaN when absent
    or empty."""
    value = str(dict(entry.extra).get("rt", "") or "").strip()
    if not value:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"rt {value!r} of {entry.sample} {entry.peptide}: not a number of "
                         "minutes") from None


def entry_rt_window(entry: PeptideEntry) -> tuple[float, float] | None:
    """The retention window of a peptide table row's ``rt_start`` and ``rt_end`` columns
    (minutes in the table), in seconds as :class:`~hxex.io.MzmlTarget` takes it; None when both
    are absent or empty. One without the other, a non-number or an end not after the start is
    an error."""
    extra = dict(entry.extra)
    values = [str(extra.get(k, "") or "").strip() for k in ("rt_start", "rt_end")]
    if not any(values):
        return None
    where = f"{entry.sample} {entry.peptide} {entry.charge}+"
    if not all(values):
        raise ValueError(f"rt_start and rt_end of {where}: give both or neither")
    try:
        lo, hi = (float(v) for v in values)
    except ValueError:
        raise ValueError(f"rt_start {values[0]!r}, rt_end {values[1]!r} of {where}: not "
                         "numbers of minutes") from None
    if not lo < hi:
        raise ValueError(f"rt_start {lo:g} and rt_end {hi:g} of {where}: the end must be after "
                         "the start")
    return 60.0 * lo, 60.0 * hi


@dataclass(frozen=True)
class SpectrumSet:
    """Many spectra as flat arrays plus an index with one entry per spectrum.

    Spectrum ``i`` is ``mz[offsets[i]:offsets[i + 1]]`` and the same slice of ``intensity``.
    The index arrays are parallel, one element per spectrum.

    ``source`` records how the spectra were read, as plain JSON-compatible data (a reader that
    selects and sums scans, such as :func:`hxex.io.read_mzml`, sets it so the read can be
    repeated); it is kept by :meth:`select`, spectra files and results bundles, and is None when
    nothing was recorded. ``rt`` is each spectrum's retention time in minutes (NaN: unknown;
    all NaN when not given); ``resolving_power`` and ``resolving_power_mz`` the instrument's
    resolving power recorded with each spectrum's scans and the m/z it is quoted at (NaN when
    not recorded; see :class:`Spectrum`). ``role`` is each spectrum's role (``'undeut'``,
    ``'fulldeut'`` or ``''``; all ``''`` when not given): the controls of a series are the
    spectra with a role, never a time.
    """

    mz: np.ndarray
    intensity: np.ndarray
    offsets: np.ndarray
    sample: np.ndarray
    peptide: np.ndarray
    peptide_range: np.ndarray
    file: np.ndarray
    label: np.ndarray
    spectrum_id: np.ndarray
    start_seq: np.ndarray
    end_seq: np.ndarray
    charge: np.ndarray
    rep: np.ndarray
    data_id: np.ndarray
    time: np.ndarray
    modification: np.ndarray
    rt: np.ndarray | None = None
    source: Mapping | None = field(default=None, compare=False)
    resolving_power: np.ndarray | None = None
    resolving_power_mz: np.ndarray | None = None
    role: np.ndarray | None = None

    def __post_init__(self):  # built without retention times or resolving powers: unknown
        for name in ("rt", "resolving_power", "resolving_power_mz"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.full(len(self.offsets) - 1, np.nan))
        if self.role is None:  # built without roles: time points only
            object.__setattr__(self, "role", _no_role(len(self.offsets) - 1))
        else:
            object.__setattr__(self, "role", np.asarray(self.role, dtype=str))

    def is_control(self, role: str) -> np.ndarray:
        """Whether each spectrum is a control of this ``role`` (``'undeut'`` or ``'fulldeut'``)."""
        return self.role == check_role(role) if role else np.zeros(len(self), dtype=bool)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def spectrum(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(m/z, intensity) views for spectrum ``i``."""
        lo, hi = self.offsets[i], self.offsets[i + 1]
        return self.mz[lo:hi], self.intensity[lo:hi]

    def find(self, spectrum_id: str) -> int:
        """Position of the spectrum with this ``spectrum_id``; raises KeyError if absent."""
        hits = np.flatnonzero(self.spectrum_id == spectrum_id)
        if len(hits) == 0:
            raise KeyError(spectrum_id)
        return int(hits[0])

    def time_idx(self) -> np.ndarray:
        """Index of each spectrum's point among the distinct points of its data set
        (:func:`time_index`: the undeuterated control first, then the times, the fully
        deuterated control last).

        Derived from the spectra in this set, so it changes if spectra are dropped. Prefer
        ``spectrum_id`` to refer to spectra.
        """
        return time_index(self.data_id, self.time, self.role)

    def select(self, keep: np.ndarray) -> SpectrumSet:
        """A new set with the spectra where the boolean array ``keep`` is True."""
        keep = _check_mask(keep, len(self))
        pts = ragged_mask(self.offsets, keep)
        sizes = np.diff(self.offsets)[keep]
        fields = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("mz", "intensity", "offsets", "source")
        }
        return SpectrumSet(
            mz=self.mz[pts],
            intensity=self.intensity[pts],
            offsets=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
            **{k: v[keep] for k, v in fields.items()},
            source=self.source,
        )

    @classmethod
    def from_spectra(
        cls, spectra: Iterable[Spectrum], source: Mapping | None = None
    ) -> SpectrumSet:
        spectra = list(spectra)
        sizes = [len(s.mz) for s in spectra]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        empty = np.empty(0, dtype=np.float64)
        return cls(
            mz=np.concatenate([s.mz for s in spectra]) if spectra else empty,
            intensity=np.concatenate([s.intensity for s in spectra]) if spectra else empty,
            offsets=offsets,
            sample=np.array([s.entry.sample for s in spectra], dtype=str),
            peptide=np.array([s.entry.peptide for s in spectra], dtype=str),
            peptide_range=np.array([s.entry.peptide_range for s in spectra], dtype=str),
            file=np.array([s.entry.file for s in spectra], dtype=str),
            label=np.array([s.label for s in spectra], dtype=str),
            spectrum_id=np.array([s.spectrum_id for s in spectra], dtype=str),
            start_seq=np.array([s.entry.start_seq for s in spectra], dtype=np.int64),
            end_seq=np.array([s.entry.end_seq for s in spectra], dtype=np.int64),
            charge=np.array([s.entry.charge for s in spectra], dtype=np.int64),
            rep=np.array([s.rep for s in spectra], dtype=np.int64),
            data_id=np.array([s.data_id for s in spectra], dtype=np.int64),
            time=np.array([s.time for s in spectra], dtype=np.float64),
            modification=np.array([s.entry.modification for s in spectra], dtype=str),
            rt=np.array([s.rt if np.isfinite(s.rt) else entry_rt(s.entry) for s in spectra],
                        dtype=np.float64),
            source=source,
            resolving_power=np.array([s.resolving_power for s in spectra], dtype=np.float64),
            resolving_power_mz=np.array([s.resolving_power_mz for s in spectra],
                                        dtype=np.float64),
            role=np.array([s.role for s in spectra], dtype="<U8"),
        )
