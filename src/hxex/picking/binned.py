# SPDX-License-Identifier: GPL-3.0-or-later
"""The binned table: picked deuterium sticks for many spectra, as flat NumPy arrays."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from ..spectra import _check_mask, ragged_mask, time_index

_FLAT = ("mz", "intensity", "n_deut", "offsets")


@dataclass(frozen=True)
class BinnedSet:
    """Picked bins for many spectra. Bin ``j`` of spectrum ``i`` is element
    ``offsets[i] + j`` of ``mz``, ``intensity`` and ``n_deut``.

    ``mz`` is the *predicted* m/z of the bin (as in pyHXExpress's ``alldeutdata``), not the
    observed position. ``env_width`` is in mass units, ``max_namides`` is the backbone amide
    count (without user-specified ``Hex`` sites). Per-spectrum flags: ``empty`` (all picked
    intensities zero, the criterion pyHXExpress uses to drop a SpecExport spectrum), ``no_signal``
    (no bin reaches the threshold) and ``capped`` (the bin limit cut the zero filling short).
    Nothing is dropped here; drop flagged spectra with :meth:`select`. The index arrays are
    copied from the :class:`~hxex.spectra.SpectrumSet` that was picked.
    """

    mz: np.ndarray
    intensity: np.ndarray
    n_deut: np.ndarray
    offsets: np.ndarray
    env_width: np.ndarray
    env_symm: np.ndarray
    max_namides: np.ndarray
    threshold: np.ndarray
    empty: np.ndarray
    no_signal: np.ndarray
    capped: np.ndarray
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

    def spectrum(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(predicted m/z, intensity, n_deut) views for spectrum ``i``."""
        lo, hi = self.offsets[i], self.offsets[i + 1]
        return self.mz[lo:hi], self.intensity[lo:hi], self.n_deut[lo:hi]

    def find(self, spectrum_id: str) -> int:
        """Position of the spectrum with this ``spectrum_id``; raises KeyError if absent."""
        hits = np.flatnonzero(self.spectrum_id == spectrum_id)
        if len(hits) == 0:
            raise KeyError(spectrum_id)
        return int(hits[0])

    def time_idx(self) -> np.ndarray:
        """Rank of each spectrum's time among the distinct times of its data set, derived from
        the spectra in this set (so dropping spectra can change it)."""
        return time_index(self.data_id, self.time)

    def select(self, keep: np.ndarray) -> BinnedSet:
        """A new set with the spectra where the boolean array ``keep`` is True."""
        keep = _check_mask(keep, len(self))
        bins = ragged_mask(self.offsets, keep)
        sizes = np.diff(self.offsets)[keep]
        out = {
            f.name: getattr(self, f.name)[keep]
            for f in dataclasses.fields(self)
            if f.name not in _FLAT
        }
        return BinnedSet(
            mz=self.mz[bins],
            intensity=self.intensity[bins],
            n_deut=self.n_deut[bins],
            offsets=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
            **out,
        )
