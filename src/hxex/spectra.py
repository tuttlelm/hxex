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


def time_index(data_id: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Rank of each time among the sorted distinct times of its data set (``data_id``)."""
    out = np.zeros(len(time), dtype=np.int64)
    for d in np.unique(data_id):
        sel = data_id == d
        out[sel] = np.searchsorted(np.unique(time[sel]), time[sel])
    return out


def ragged_mask(offsets: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Boolean mask over the flat array for the segments (spectra) where ``keep`` is True."""
    return np.repeat(keep, np.diff(offsets))


def _check_mask(keep: np.ndarray, n: int) -> np.ndarray:
    keep = np.asarray(keep, dtype=bool)
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


@dataclass(frozen=True)
class Spectrum:
    """One raw spectrum (m/z sorted, no missing values) and where it came from.

    ``time`` is in seconds (0 for the undeuterated control, the user-set time for the fully
    deuterated one); ``rep`` counts spectra with the same time within a data set; ``data_id`` is
    the position of the entry in the list that was read; ``label`` names the source for display
    (a column label or file name).

    ``spectrum_id`` identifies the spectrum for later reference. The file readers set the local
    part, built from where the spectrum was read, so it is unique within a data directory and
    does not depend on read order, filtering or the fully deuterated time: ``FILE#COLUMN``
    (1-based column of the label) for HX-Express, ``SAMPLE/PEPTIDE_DIR/FILE`` for SpecExport and
    ``PEPTIDE_DIR/FILE`` for pooled folders. :func:`hxex.io.read_spectra` prefixes the
    user-chosen project name (``PROJECT:LOCAL``, see :func:`qualify_spectrum_id`) so ids stay
    unique when data from several directories are combined.
    """

    mz: np.ndarray
    intensity: np.ndarray
    entry: PeptideEntry
    time: float
    rep: int
    data_id: int = 0
    label: str = ""
    spectrum_id: str = ""


@dataclass(frozen=True)
class SpectrumSet:
    """Many spectra as flat arrays plus an index with one entry per spectrum.

    Spectrum ``i`` is ``mz[offsets[i]:offsets[i + 1]]`` and the same slice of ``intensity``.
    The index arrays are parallel, one element per spectrum.
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
        """Index of each spectrum's time among the sorted distinct times of its data set.

        Derived from the spectra in this set, so it changes if spectra are dropped. Prefer
        ``spectrum_id`` to refer to spectra.
        """
        return time_index(self.data_id, self.time)

    def select(self, keep: np.ndarray) -> SpectrumSet:
        """A new set with the spectra where the boolean array ``keep`` is True."""
        keep = _check_mask(keep, len(self))
        pts = ragged_mask(self.offsets, keep)
        sizes = np.diff(self.offsets)[keep]
        fields = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("mz", "intensity", "offsets")
        }
        return SpectrumSet(
            mz=self.mz[pts],
            intensity=self.intensity[pts],
            offsets=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
            **{k: v[keep] for k, v in fields.items()},
        )

    @classmethod
    def from_spectra(cls, spectra: Iterable[Spectrum]) -> SpectrumSet:
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
        )
