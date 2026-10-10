# SPDX-License-Identifier: GPL-3.0-or-later
"""Peak picking at predicted deuterium positions (ExMS spacing).

For each spectrum the bins sit at ``undeut_mz + n * ExMS spacing / z``. Each bin takes the
tallest point inside a +/- ppm window. Picking covers every bin up to a limit set by the amide
count and ends ``zero_filling`` bins after the last bin that reaches the noise threshold, so a
valley inside a multimodal envelope never ends picking early (pyHXExpress stopped after a
cumulative count of below-threshold bins).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .. import chem
from ..hooks import CancelHook, ProgressHook, check_cancel, report
from ..spectra import SpectrumSet
from .binned import BinnedSet
from .config import DEFAULT_PICKER_CONFIG, PickerConfig

log = logging.getLogger(__name__)

# ExMS isotope spacing (Da) for the first five steps; later steps repeat the last value.
EXMS_DELTA_M = (1.003355, 1.003355, 1.004816, 1.004816, 1.006277)


@dataclass(frozen=True)
class BinPlan:
    """Where the bins of one peptide are: predicted m/z for bins ``0..n``, and the amide count."""

    pred_mz: np.ndarray
    max_namides: int


@dataclass(frozen=True)
class PickedSpectrum:
    """Result for one spectrum (see :class:`BinnedSet` for the meaning of the fields)."""

    mz: np.ndarray
    intensity: np.ndarray
    n_deut: np.ndarray
    env_width: float
    env_symm: float
    max_namides: int
    threshold: float
    empty: bool
    no_signal: bool
    capped: bool


def predicted_mz(undeut_mz: float, charge: int, n_bins: int) -> np.ndarray:
    """Predicted m/z of bins ``0..n_bins`` (``n_bins + 1`` values), ExMS spacing."""
    steps = np.array(
        [EXMS_DELTA_M[k] if k < len(EXMS_DELTA_M) else EXMS_DELTA_M[-1] for k in range(n_bins)]
    )
    return np.concatenate([[0.0], np.cumsum(steps / charge)]) + undeut_mz


def plan_bins(
    peptide: str,
    charge: int,
    *,
    modification: str = "",
    config: PickerConfig = DEFAULT_PICKER_CONFIG,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
) -> BinPlan:
    """Predicted bin positions for a peptide.

    The last bin is the highest possible deuteration plus the full natural-abundance envelope,
    plus the zero filling: ``amides + Hex + (natural-abundance peaks - 1) + zero_filling``. It
    only bounds the search, because picking ends ``zero_filling`` bins after the last bin that
    reaches the threshold (pyHXExpress used ``peaks // 2`` and stopped there, cutting the
    natural-abundance tail of strongly deuterated spectra). ``modification`` is the pyHXExpress
    string (``'H:1 Hex:1 ion_type:c undeut_mz:...'``).
    """
    spec = chem.parse_modification(modification)
    amides = chem.count_amides(
        peptide,
        ion_type=spec.ion_type,
        nterm_subtract=config.nterm_subtract,
        rules=config.amide_rules,
        registry=registry,
    )
    shape = chem.natural_abundance(
        peptide,
        spec.ion_type,
        mod=spec,
        nterm_subtract=config.nterm_subtract,
        rules=config.amide_rules,
        registry=registry,
    )
    n_bins = amides + spec.hex + len(shape) - 1 + config.zero_filling
    undeut = chem.mz(peptide, charge, spec.ion_type, mod=spec, registry=registry)
    return BinPlan(predicted_mz(undeut, charge, n_bins), amides)


def envelope(
    x: np.ndarray, y: np.ndarray, height: float
) -> tuple[float, float, float]:
    """Left and right m/z where the profile ``y(x)`` crosses ``height`` (linear interpolation),
    and the intensity at the left edge.

    The envelope runs from the first to the last point above ``height``. If the first (last)
    point is already above it, that point's m/z is the edge. Returns NaNs if ``height <= 0`` or
    no point is above it.
    """
    nan = (float("nan"),) * 3
    above = np.flatnonzero(y > height)
    if not height > 0 or len(above) == 0:
        return nan
    lower, upper = above[0], above[-1]
    if y[0] > height:
        left_mz, left_int = x[0], y[0]
    else:
        left_mz = x[lower] + (x[lower - 1] - x[lower]) * (height - y[lower]) / (
            y[lower - 1] - y[lower]
        )
        left_int = height
    if y[-1] > height:
        right_mz = x[-1]
    else:
        right_mz = x[upper] + (x[upper + 1] - x[upper]) * (height - y[upper]) / (
            y[upper + 1] - y[upper]
        )
    return float(left_mz), float(right_mz), float(left_int)


def _bin_values(
    mz: np.ndarray, y: np.ndarray, pred: np.ndarray, config: PickerConfig
) -> np.ndarray:
    """Window maximum for every bin (0 where the window is empty or rejected as a shoulder)."""
    factors = 1.0 + np.array([-1.0, 1.0]) * config.resolution_ppm / 1e6
    lo = np.searchsorted(mz, pred * factors[0], side="left")
    hi = np.searchsorted(mz, pred * factors[1], side="right")
    values = np.zeros(len(pred))
    for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
        window = y[a:b]
        if len(window) == 0:
            continue
        top = int(np.argmax(window))  # first maximum, so ties are resolved deterministically
        if len(window) >= config.shoulder_min_points and top in (0, len(window) - 1):
            continue  # the maximum is at the window edge: a shoulder of a neighbouring peak
        values[i] = window[top]
    return values


def pick_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    peptide: str,
    charge: int,
    *,
    modification: str = "",
    config: PickerConfig = DEFAULT_PICKER_CONFIG,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
    plan: BinPlan | None = None,
) -> PickedSpectrum:
    """Pick one spectrum (arrays of m/z and intensity; sorted by m/z if they are not)."""
    plan = plan or plan_bins(
        peptide, charge, modification=modification, config=config, registry=registry
    )
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if len(mz) > 1 and np.any(np.diff(mz) < 0):
        order = np.argsort(mz, kind="stable")
        mz, intensity = mz[order], intensity[order]
    if config.noise:
        threshold = float(config.noise)
    elif len(intensity):
        threshold = config.noise_percent / 100.0 * float(intensity.max())
    else:
        threshold = float("nan")

    pred = plan.pred_mz
    values = _bin_values(mz, intensity, pred, config)
    n_last = len(pred) - 1
    signal = np.flatnonzero(values >= threshold)  # False throughout if threshold is NaN
    last = int(signal[-1]) if len(signal) else -1
    end = min(max(last, 0) + config.zero_filling, n_last)
    picked_mz, picked_y = pred[: end + 1], values[: end + 1]

    width = symm = float("nan")
    if len(picked_y):
        ymax = float(picked_y.max())
        left, right, left_int = envelope(picked_mz, picked_y, ymax * config.env_threshold)
        width = charge * (right - left)
        symm = 2.0 - (ymax - left_int) / ymax if ymax else float("nan")
    return PickedSpectrum(
        mz=picked_mz,
        intensity=picked_y,
        n_deut=np.arange(end + 1, dtype=np.int64),
        env_width=width,
        env_symm=symm,
        max_namides=plan.max_namides,
        threshold=threshold,
        empty=bool(picked_y.sum() == 0),
        no_signal=last < 0,
        capped=last >= 0 and last + config.zero_filling > n_last,
    )


def pick_spectra(
    spectra: SpectrumSet,
    config: PickerConfig = DEFAULT_PICKER_CONFIG,
    *,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> BinnedSet:
    """Pick every spectrum of ``spectra``. Nothing is dropped; see the flags on the result."""
    plans: dict[tuple[str, int, str], BinPlan] = {}
    picked: list[PickedSpectrum] = []
    for i in range(len(spectra)):
        check_cancel(cancel)
        key = (str(spectra.peptide[i]), int(spectra.charge[i]), str(spectra.modification[i]))
        if key not in plans:
            plans[key] = plan_bins(
                key[0], key[1], modification=key[2], config=config, registry=registry
            )
        mz, y = spectra.spectrum(i)
        picked.append(
            pick_spectrum(
                mz, y, key[0], key[1], modification=key[2], config=config, registry=registry,
                plan=plans[key],
            )
        )
        report(progress, i + 1, len(spectra))
    return _assemble(spectra, picked)


def _assemble(spectra: SpectrumSet, picked: list[PickedSpectrum]) -> BinnedSet:
    sizes = [len(p.mz) for p in picked]
    cat: Callable[[str, type], np.ndarray] = lambda name, dt: np.array(  # noqa: E731
        [getattr(p, name) for p in picked], dtype=dt
    )
    empty = np.empty(0)
    return BinnedSet(
        mz=np.concatenate([p.mz for p in picked]) if picked else empty,
        intensity=np.concatenate([p.intensity for p in picked]) if picked else empty,
        n_deut=np.concatenate([p.n_deut for p in picked]) if picked else empty.astype(np.int64),
        offsets=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
        env_width=cat("env_width", np.float64),
        env_symm=cat("env_symm", np.float64),
        max_namides=cat("max_namides", np.int64),
        threshold=cat("threshold", np.float64),
        empty=cat("empty", bool),
        no_signal=cat("no_signal", bool),
        capped=cat("capped", bool),
        sample=spectra.sample,
        peptide=spectra.peptide,
        peptide_range=spectra.peptide_range,
        file=spectra.file,
        label=spectra.label,
        spectrum_id=spectra.spectrum_id,
        start_seq=spectra.start_seq,
        end_seq=spectra.end_seq,
        charge=spectra.charge,
        rep=spectra.rep,
        data_id=spectra.data_id,
        time=spectra.time,
        modification=spectra.modification,
    )
