# SPDX-License-Identifier: GPL-3.0-or-later
"""The binned table: picked deuterium sticks for many spectra, as flat NumPy arrays."""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field

import numpy as np

from ..spectra import _check_mask, _objects, ragged_mask, time_index
from .screen import SCREEN_SCORES

_FLAT = ("mz", "intensity", "n_deut", "offsets", "peak_mz")
# the m/z check of each spectrum's series (hxex.picking.mz_check)
MZ_CHECK_COLUMNS = ("envelope_offset_ppm", "envelope_offset_limit_ppm", "window_shift_ppm")
# the profile fit of each spectrum (hxex.picking.profile; NaN: picked, not fitted)
PROFILE_COLUMNS = ("fit_offset_ppm", "purity", "stray_share")


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
    reaches the threshold) and ``capped`` (the bin limit cut the zero filling short). The
    ``screen_*`` scores (:mod:`hxex.picking.screen`) measure another species in the raw spectrum
    (NaN: not assessed). The m/z check of the spectrum's series (:mod:`hxex.picking.mz_check`):
    ``envelope_offset_ppm``, the series' undeuterated envelope offset from its runs'
    calibration, ``envelope_offset_limit_ppm``, the limit it was compared with (both NaN: not
    assessed), and ``window_shift_ppm``, how far the picker's windows were moved from the
    predicted m/z (0: not moved; only bundles picked with the former ``recentre``, removed in
    HISTORY 200, have others). With ``pick = 'profile_fit'`` (:mod:`hxex.picking.profile`):
    ``fit_offset_ppm``, the m/z offset the profile fit took, ``purity``, the share of the profile
    the bins' fitted peaks account for within a peak width of each bin, and ``stray_share``, the
    share of the fitted region the stray peaks account for
    (:func:`hxex.picking.profile.fit_quality`; all NaN for a spectrum that kept the pick);
    ``peak_mz`` is then each bin's fitted position. With ``pick = 'peak'`` and the run's peak
    shape measured, ``purity`` and ``stray_share`` come from one fit at the spectrum's own m/z
    offset and ``fit_offset_ppm`` stays NaN (the bins are the pick's).
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
    rt_start: np.ndarray | None = None
    rt_end: np.ndarray | None = None
    drift_start: np.ndarray | None = None  # ms: the drift range summed (NaN: not gated)
    drift_end: np.ndarray | None = None
    mz_calibration_ppm: np.ndarray | None = None  # the file's m/z offset taken out (NaN: none)
    screen_below_mono: np.ndarray | None = None
    screen_off_position: np.ndarray | None = None
    screen_between: np.ndarray | None = None
    screen_between_separate: np.ndarray | None = None
    screen_beyond_reach: np.ndarray | None = None
    envelope_offset_ppm: np.ndarray | None = None
    envelope_offset_limit_ppm: np.ndarray | None = None
    window_shift_ppm: np.ndarray | None = None
    fit_offset_ppm: np.ndarray | None = None
    purity: np.ndarray | None = None
    stray_share: np.ndarray | None = None
    # per spectrum, its scans' noise at its bins and shares of the signal (ScanCut; None: not
    # recorded): such spectra are fitted through the instrument's per-scan cut
    scan_cut: np.ndarray | None = field(default=None, compare=False)

    def __post_init__(self):
        for name in SCREEN_SCORES:  # not assessed: picked elsewhere, or by an older version
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.full(len(self.offsets) - 1, np.nan))
        for name in ("envelope_offset_ppm", "envelope_offset_limit_ppm",  # not assessed
                     *PROFILE_COLUMNS):  # picked, not fitted
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.full(len(self.offsets) - 1, np.nan))
        if self.window_shift_ppm is None:  # picked elsewhere or by an older version: not moved
            object.__setattr__(self, "window_shift_ppm", np.zeros(len(self.offsets) - 1))
        # retention times (minutes) and drift ranges (ms) not given: unknown
        for name in ("rt", "rt_start", "rt_end", "drift_start", "drift_end",
                     "mz_calibration_ppm"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.full(len(self.offsets) - 1, np.nan))
        if self.peak_mz is None:
            object.__setattr__(self, "peak_mz", np.full(len(self.mz), np.nan))
        if self.mz_offset_ppm is None:
            object.__setattr__(self, "mz_offset_ppm", np.full(len(self.offsets) - 1, np.nan))
        if self.scan_cut is None:  # no scans recorded
            object.__setattr__(self, "scan_cut", _objects([None] * (len(self.offsets) - 1)))
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


def _digest(ion, mz, intensity) -> bytes:
    """The ion's parts and the m/z and intensity arrays (as float64), hashed."""
    h = hashlib.blake2b(digest_size=16)
    for part in ion:
        h.update(str(part).encode())
        h.update(b"\x1f")
    h.update(np.ascontiguousarray(mz, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(intensity, dtype=np.float64).tobytes())
    return h.digest()


def measurement_keys(binned: BinnedSet) -> list[bytes | None]:
    """What each spectrum of ``binned`` is as a measurement, ``(M,)`` digests: the ion
    (peptide, modification and charge) and the picked bins' m/z and intensity, hashed to the
    last bit; None for a spectrum without signal (every picked intensity 0), which is never the
    same as another. Two spectra with one key are one measurement read twice
    (:func:`same_measurement`); keys of two bundles can be compared the same way."""
    out: list[bytes | None] = []
    for i in range(len(binned)):
        a, b = int(binned.offsets[i]), int(binned.offsets[i + 1])
        intensity = np.asarray(binned.intensity[a:b], dtype=np.float64)
        if not (intensity > 0).any():
            out.append(None)
            continue
        out.append(_digest((binned.peptide[i], binned.modification[i], binned.charge[i]),
                           binned.mz[a:b], intensity))
    return out


def raw_keys(spectra, rows) -> dict[int, bytes | None]:
    """What each raw spectrum at ``rows`` of a :class:`~hxex.spectra.SpectrumSet` is as a
    measurement: the ion (peptide, modification and charge) and its m/z and intensity arrays,
    hashed to the last bit; None for a spectrum without signal (never the same as another). The
    raw counterpart of :func:`measurement_keys`, for the stages before picking."""
    out: dict[int, bytes | None] = {}
    for i in rows:
        i = int(i)
        mz, y = spectra.spectrum(i)
        y = np.asarray(y, dtype=np.float64)
        if not (y > 0).any():
            out[i] = None
            continue
        out[i] = _digest((spectra.peptide[i], spectra.modification[i], spectra.charge[i]),
                         mz, y)
    return out


def same_measurement(binned: BinnedSet) -> np.ndarray:
    """Which spectra of ``binned`` are one measurement read more than once, ``(M,)`` integers:
    spectra of the same ion (peptide, modification and charge) whose picked bins are identical
    in m/z and intensity, to the last bit, share a number (the index of the first of them in
    spectrum-id order); every other spectrum has its own index. This is what two samples or
    conditions that read the same file (a copy, a link, or one scan file listed twice) give:
    the same raw data picked the same way. A spectrum without signal (every picked intensity 0)
    is never the same as another (:func:`measurement_keys`)."""
    keys = measurement_keys(binned)
    order = np.argsort(np.asarray(binned.spectrum_id, dtype=str), kind="stable")
    out = np.arange(len(binned), dtype=np.int64)
    first: dict[bytes, int] = {}
    for i in order.tolist():
        if keys[i] is not None:
            out[i] = first.setdefault(keys[i], i)
    return out


def copies_in_one_series(binned: BinnedSet, measurement: np.ndarray | None = None
                         ) -> list[tuple[int, int]]:
    """The pairs ``(first, copy)`` of spectra of ``binned`` (positions) that belong to the same
    series (``data_id``: one sample, ion and condition) and are one measurement
    (``measurement``, :func:`same_measurement`; computed when not given): two replicates, or a
    replicate and a control, read from one file (a copy under another name, or one file listed
    twice). Each copy is paired with the first spectrum of its measurement and series in
    spectrum-id order; empty when there are none. Copies in different series are not listed: a
    file read for several samples (a pooled project's controls), or one undeuterated file read
    for several conditions of a sample (the in-source energies of a fragment list), is one
    measurement used by each, which the controls, the per-bin noise, and the shared envelopes
    count once."""
    group = same_measurement(binned) if measurement is None else np.asarray(measurement)
    series = np.asarray(binned.data_id)
    shared = np.bincount(group, minlength=len(group))[group] > 1
    first: dict[tuple, int] = {}
    out = []
    for i in np.argsort(np.asarray(binned.spectrum_id, dtype=str), kind="stable").tolist():
        if not shared[i]:
            continue
        j = first.setdefault((int(group[i]), int(series[i])), i)
        if j != i:
            out.append((j, i))
    return sorted(out)
