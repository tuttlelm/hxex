# SPDX-License-Identifier: GPL-3.0-or-later
"""The binned table: picked deuterium sticks for many spectra, as flat NumPy arrays."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from ..spectra import _check_mask, ragged_mask, time_index

_FLAT = ("mz", "intensity", "n_deut", "offsets", "peak_mz")


@dataclass(frozen=True)
class BinnedSet:
    """Picked bins for many spectra. Bin ``j`` of spectrum ``i`` is element
    ``offsets[i] + j`` of ``mz``, ``intensity`` and ``n_deut``.

    ``mz`` is the *predicted* m/z of the bin (as in pyHXExpress's ``alldeutdata``); ``peak_mz``
    is where the picked peak was found (NaN for an empty bin, and throughout for tables read
    from elsewhere), and ``mz_offset_ppm`` (per spectrum) the intensity-weighted median of the
    picked peaks' distance from their predicted m/z, in ppm, over the bins at or above the
    threshold: an offset common to every spectrum of a series points to a wrong composition
    (for example a terminal group or an ion type), one common to a file to the calibration.
    ``env_width`` is in mass units, ``max_namides`` is the backbone amide count (without
    user-specified ``Hex`` sites). Per-spectrum flags: ``empty`` (all picked intensities zero,
    the usual reason to drop a SpecExport spectrum), ``no_signal`` (no bin
    reaches the threshold) and ``capped`` (the bin limit cut the zero filling short).
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
    peak_mz: np.ndarray | None = None
    mz_offset_ppm: np.ndarray | None = None
    rt: np.ndarray | None = None
    role: np.ndarray | None = None

    def __post_init__(self):
        if self.rt is None:  # retention times (minutes) not given: unknown
            object.__setattr__(self, "rt", np.full(len(self.offsets) - 1, np.nan))
        if self.peak_mz is None:
            object.__setattr__(self, "peak_mz", np.full(len(self.mz), np.nan))
        if self.mz_offset_ppm is None:
            object.__setattr__(self, "mz_offset_ppm", np.full(len(self.offsets) - 1, np.nan))
        if self.role is None:  # built without roles: time points only
            object.__setattr__(self, "role", np.full(len(self.offsets) - 1, "", dtype="<U8"))
        else:
            object.__setattr__(self, "role", np.asarray(self.role, dtype=str))

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
        return time_index(self.data_id, self.time, self.role)

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
            peak_mz=self.peak_mz[bins],
            **out,
        )
