# SPDX-License-Identifier: GPL-3.0-or-later
"""Spectrum data contracts: one raw spectrum, and a set stored as flat NumPy arrays."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np


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
    the position of the entry in the list that was read; ``label`` names the source (a column
    label or file name).
    """

    mz: np.ndarray
    intensity: np.ndarray
    entry: PeptideEntry
    time: float
    rep: int
    data_id: int = 0
    label: str = ""


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
    start_seq: np.ndarray
    end_seq: np.ndarray
    charge: np.ndarray
    rep: np.ndarray
    data_id: np.ndarray
    time: np.ndarray

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def spectrum(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(m/z, intensity) views for spectrum ``i``."""
        lo, hi = self.offsets[i], self.offsets[i + 1]
        return self.mz[lo:hi], self.intensity[lo:hi]

    def time_idx(self) -> np.ndarray:
        """Index of each spectrum's time among the sorted distinct times of its data set."""
        out = np.zeros(len(self), dtype=np.int64)
        for d in np.unique(self.data_id):
            sel = self.data_id == d
            out[sel] = np.searchsorted(np.unique(self.time[sel]), self.time[sel])
        return out

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
            start_seq=np.array([s.entry.start_seq for s in spectra], dtype=np.int64),
            end_seq=np.array([s.entry.end_seq for s in spectra], dtype=np.int64),
            charge=np.array([s.entry.charge for s in spectra], dtype=np.int64),
            rep=np.array([s.rep for s in spectra], dtype=np.int64),
            data_id=np.array([s.data_id for s in spectra], dtype=np.int64),
            time=np.array([s.time for s in spectra], dtype=np.float64),
        )
