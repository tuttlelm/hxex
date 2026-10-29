# SPDX-License-Identifier: GPL-3.0-or-later
"""Series whose undeuterated envelope sits away from its m/z, and windows moved onto it.

An undeuterated control's isotope peaks sit where the composition puts them, up to the
instrument's calibration, which a run shares. A series whose controls' peaks sit further from
the run's calibration than the run's other series, by the same amount in every control, most
likely shares its peaks with another, near-isobaric species that the resolving power cannot
separate from it: the picker's window, centred on the prediction, then reads the flank of the
merged peak, a share of the intensity that changes from bin to bin (docs/HISTORY.md item 166).

- The offset of one undeuterated spectrum: for M, M+1 and M+2, the top of the peak nearest the
  prediction (from the profile point nearest it, uphill through evenly spaced points, refined to
  the vertex of a parabola), searched within the larger of ``SEARCH_PPM`` and twice the picking
  half-width, never beyond ``SPACING_SHARE`` of the isotope spacing; the median of the three
  when they agree within the larger of ``AGREE_PPM`` and a quarter of the half-width (else the
  spectrum has none). Unlike the picker's ``mz_offset_ppm``, measured inside the window, this
  finds an envelope wherever its peaks are. Centroided spectra give the nearest stick.
- A run is one undeuterated injection: the controls of one sample and replicate. Its
  calibration is a straight line in m/z through its controls' offsets (Theil-Sen, robust to
  the off series; a TOF's calibration drifts by about 1 ppm per 100 m/z within a run, an
  Orbitrap's hardly at all) and its scatter the robust sd of the offsets about that line
  (1.4826 times the median absolute deviation); a run with fewer than ``MIN_RUN_CONTROLS``
  offsets takes the line and scatter of every control.
- A series' offset is the median over its controls of (offset minus the run's calibration at
  the control's monoisotopic m/z);
  the series is off its m/z when that is beyond ``off_mz_sd`` times its runs' scatter, and at
  least ``MIN_LIMIT_PPM``. Its windows are then moved by its measured offset (the median of
  its controls' offsets, calibration included) when ``recentre`` is on.

Not assessed: fragment ions (an ion-type mixture moves their peaks; ``poor_envelope`` and the
fit-stage ``mz_shift`` cover them) and series without an undeuterated control with an offset.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import theilslopes

from .. import chem
from .config import ISOTOPE_STEP, PickerConfig
from .screen import STICK_POINTS

SEARCH_PPM = 40.0  # the least search range either side of each prediction
SPACING_SHARE = 0.4  # the search never reaches past this share of the isotope spacing
AGREE_PPM = 5.0  # M, M+1 and M+2 agree within this (or a quarter of the half-width)
CLIMB_GAP = 1.6  # profile points closer than this times the local spacing are one peak
MIN_RUN_CONTROLS = 10  # a run with fewer offsets takes the scatter of every control
MIN_LIMIT_PPM = 3.0  # an offset is never called off within this
_MAD_SD = 1.4826


def _peak_top(mz: np.ndarray, y: np.ndarray, target: float, limit: float,
              spacing: float) -> float:
    """The m/z of the top of the peak nearest ``target`` (NaN when none lies within
    ``limit``, in m/z): the climb takes neighbours closer than ``CLIMB_GAP`` times the
    spectrum's point ``spacing`` (in m/z; infinite for centroided data: the nearest stick)."""
    if not len(mz):
        return float("nan")
    j = int(np.argmin(np.abs(mz - target)))
    if abs(mz[j] - target) > limit or not y[j] > 0:
        return float("nan")
    if not np.isfinite(spacing):  # centroided: one point per peak
        return float(mz[j])
    step = CLIMB_GAP * spacing
    k = j
    for d in (1, -1):  # towards the higher neighbour, if any
        while 0 <= k + d < len(y) and abs(mz[k + d] - mz[k]) <= step and y[k + d] > y[k]:
            k += d
        if k != j:
            break
    x = float(mz[k])
    if 0 < k < len(y) - 1 and y[k - 1] > 0 and y[k + 1] > 0 and y[k - 1] < y[k] >= y[k + 1]:
        x0, x1, x2 = mz[k - 1:k + 2]
        y0, y1, y2 = y[k - 1:k + 2]
        den = (x0 - x1) * (x0 - x2) * (x1 - x2)
        a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
        b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
        if a < 0:
            x = float(min(max(-b / (2 * a), x0), x2))
    return x if abs(x - target) <= limit else float("nan")


def envelope_offset(mz: np.ndarray, intensity: np.ndarray, pred: np.ndarray, charge: int,
                    half_width_ppm: float) -> float:
    """The offset (ppm) of an undeuterated spectrum's envelope from its predicted m/z ``pred``
    (M first): the median of the M, M+1 and M+2 peak tops' offsets when they agree, else NaN
    (see the module docstring)."""
    mz = np.asarray(mz, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    if len(mz) > 1 and np.any(np.diff(mz) < 0):
        order = np.argsort(mz, kind="stable")
        mz, y = mz[order], y[order]
    pred = np.asarray(pred, dtype=np.float64)[:3]
    if not len(pred):
        return float("nan")
    reach = min(max(SEARCH_PPM, 2.0 * half_width_ppm),
                SPACING_SHARE * 1e6 * ISOTOPE_STEP / max(int(charge), 1) / pred[0])
    d = np.diff(mz)
    spacing = float(np.median(d[d > 0])) if (d > 0).any() else np.inf
    if (ISOTOPE_STEP / max(int(charge), 1)) / spacing < STICK_POINTS:
        spacing = np.inf  # centroided (as the picker and the screens tell them)
    offsets = np.array([1e6 * (_peak_top(mz, y, p, reach * 1e-6 * p, spacing) - p) / p
                        for p in pred])
    found = offsets[np.isfinite(offsets)]
    if len(found) < 2 or np.ptp(found) > max(AGREE_PPM, half_width_ppm / 4.0):
        return float("nan")
    return float(np.median(found))


@dataclass(frozen=True)
class SeriesOffsets:
    """The m/z check of the series with an undeuterated offset (one entry per series):
    ``offset_ppm`` (the median of its controls' offsets, the run's calibration included: how
    far its windows move when it is off), ``relative_ppm`` (the median of its controls' offsets
    minus their runs' calibration), ``limit_ppm`` (what that was compared with) and
    ``n_controls``."""

    data_id: np.ndarray
    offset_ppm: np.ndarray
    relative_ppm: np.ndarray
    limit_ppm: np.ndarray
    n_controls: np.ndarray

    @property
    def off(self) -> np.ndarray:
        """Whether each series is off its m/z."""
        return np.abs(self.relative_ppm) > self.limit_ppm

    def per_spectrum(self, data_id: np.ndarray, recentre: bool
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(relative offset, limit, window shift)`` for spectra of the series ``data_id``:
        NaN, NaN, 0 for a series not assessed; the shift is the series' offset where it is off
        and ``recentre`` is on, else 0."""
        data_id = np.asarray(data_id)
        rel = np.full(len(data_id), np.nan)
        lim = np.full(len(data_id), np.nan)
        shift = np.zeros(len(data_id))
        off = self.off
        for j, d in enumerate(self.data_id):
            rows = data_id == d
            rel[rows] = self.relative_ppm[j]
            lim[rows] = self.limit_ppm[j]
            if recentre and off[j]:
                shift[rows] = self.offset_ppm[j]
        return rel, lim, shift


def series_offsets(spectra, config: PickerConfig, *,
                   registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
                   skip: Collection[int] = (),
                   clusters: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
                   ) -> SeriesOffsets:
    """The m/z check of every series of ``spectra`` (a :class:`~hxex.spectra.SpectrumSet`)
    with undeuterated controls, except those in ``skip`` (fragment ions); ``config.off_mz_sd``
    sets the limit (it must not be None). ``clusters`` as for :func:`pick_spectra`."""
    from .pick import plan_bins

    if config.off_mz_sd is None:
        raise ValueError("off_mz_sd is None: the m/z check is off")
    clusters = clusters or {}
    skip = {int(d) for d in skip}
    role = np.asarray(spectra.role).astype(str)
    data_id = np.asarray(spectra.data_id)
    rows = [i for i in np.flatnonzero(role == "undeut") if int(data_id[i]) not in skip]
    plans: dict[tuple, np.ndarray] = {}
    offsets, run_of, mono = [], [], []
    for i in rows:
        series = int(data_id[i])
        own = series if series in clusters else None
        key = (str(spectra.peptide[i]), int(spectra.charge[i]), str(spectra.modification[i]), own)
        if key not in plans:
            plans[key] = np.asarray(plan_bins(
                key[0], key[1], modification=key[2], config=config, registry=registry,
                clusters=None if own is None else clusters[own]).pred_mz[:3], dtype=np.float64)
        pred = plans[key]
        mz, y = spectra.spectrum(i)
        half = float(np.atleast_1d(config.half_width(
            pred[:1], key[1], resolving_power=float(spectra.resolving_power[i]),
            resolving_power_mz=float(spectra.resolving_power_mz[i])))[0])
        offsets.append(envelope_offset(mz, y, pred, key[1], 1e6 * half / pred[0]))
        run_of.append((str(spectra.sample[i]), str(spectra.rep[i])))
        mono.append(float(pred[0]))
    offsets_arr = np.asarray(offsets, dtype=np.float64)
    mono_arr = np.asarray(mono, dtype=np.float64)
    found = np.isfinite(offsets_arr)
    run_index: dict[tuple, list[int]] = defaultdict(list)
    for j, run in enumerate(run_of):
        if found[j]:
            run_index[run].append(j)

    def calibration(at: np.ndarray) -> tuple[float, float, float]:
        """(intercept, slope per m/z, scatter) of the controls ``at``."""
        x, v = mono_arr[at], offsets_arr[at]
        slope, intercept = (theilslopes(v, x)[:2] if np.ptp(x) > 0
                            else (0.0, float(np.median(v))))
        resid = v - (intercept + slope * x)
        return float(intercept), float(slope), _MAD_SD * float(np.median(np.abs(resid)))

    whole = calibration(np.flatnonzero(found)) if found.any() else (np.nan, 0.0, np.nan)
    runs = {run: calibration(np.asarray(at)) if len(at) >= MIN_RUN_CONTROLS else whole
            for run, at in run_index.items()}
    by_series: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for j, (i, o, run) in enumerate(zip(rows, offsets_arr, run_of, strict=True)):
        if np.isfinite(o):
            intercept, slope, scatter = runs[run]
            by_series[int(data_id[i])].append((o, o - intercept - slope * mono_arr[j], scatter))
    ids = np.array(sorted(by_series), dtype=np.int64)
    take = lambda k: np.array([float(np.median([v[k] for v in by_series[d]]))  # noqa: E731
                               for d in ids], dtype=np.float64)
    scatter = take(2)
    return SeriesOffsets(
        data_id=ids, offset_ppm=take(0), relative_ppm=take(1),
        limit_ppm=np.maximum(float(config.off_mz_sd) * np.nan_to_num(scatter), MIN_LIMIT_PPM),
        n_controls=np.array([len(by_series[d]) for d in ids], dtype=np.int64),
    )
