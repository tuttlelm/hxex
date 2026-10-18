# SPDX-License-Identifier: GPL-3.0-or-later
"""Peak picking at predicted deuterium positions.

Bin ``n`` is the isotope cluster of nominal mass +n. By default its position is the cluster's
abundance-weighted mass from the composition (the composition ladder); pyHXExpress's ExMS
spacing is an option and also gives the first pass that estimates the deuteration level. Each
bin takes the tallest point inside a +/- ppm window (docs/PEAK_PICKING.md). Picking covers
every bin up to a limit set by the amide count and ends ``zero_filling`` bins after the last bin
that reaches the noise threshold, so a valley inside a multimodal envelope never ends picking
early.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

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
    """Where the bins of one peptide are: predicted m/z for bins ``0..n`` (ExMS ladder), and the
    amide count; for the composition ladder also the undeuterated m/z, the charge, the number
    of exchangeable sites and the natural isotope clusters (abundance, mass offset in Da)."""

    pred_mz: np.ndarray
    max_namides: int
    undeut_mz: float = float("nan")
    charge: int = 0
    n_sites: int = 0
    clusters: tuple[np.ndarray, np.ndarray] | None = None


@dataclass(frozen=True)
class PickedSpectrum:
    """Result for one spectrum (see :class:`BinnedSet` for the meaning of the fields)."""

    mz: np.ndarray
    intensity: np.ndarray
    n_deut: np.ndarray
    peak_mz: np.ndarray
    mz_offset_ppm: float
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
    clusters: tuple[np.ndarray, np.ndarray] | None = None,
) -> BinPlan:
    """Predicted bin positions for a peptide.

    ``clusters`` replaces the ion's natural isotope clusters, for an ion-type mixture
    (:func:`hxex.chem.mixed_clusters`); bins then reach as far as the mixture does.

    The last bin is the highest possible deuteration plus the full natural-abundance envelope,
    plus the zero filling: ``amides + Hex + (natural-abundance peaks - 1) + zero_filling``. It
    only bounds the search, because picking ends ``zero_filling`` bins after the last bin that
    reaches the threshold. ``modification`` is the pyHXExpress modification string
    (``'H:1 Hex:1 ion_type:c undeut_mz:...'``).
    """
    spec = chem.parse_modification(modification)
    amides = chem.count_amides(
        peptide,
        ion_type=spec.ion_type,
        nterm_subtract=config.nterm_subtract,
        rules=config.amide_rules,
        registry=registry,
        parent_pos=spec.parent_pos,
    )
    if clusters is None:
        clusters = chem.isotope_clusters(
            peptide,
            spec.ion_type,
            mod=spec,
            nterm_subtract=config.nterm_subtract,
            rules=config.amide_rules,
            registry=registry,
        )
    shape, offsets = clusters
    sites = chem.exchangeable_sites(
        peptide, mod=spec, ion_type=spec.ion_type, nterm_subtract=config.nterm_subtract,
        rules=config.amide_rules, registry=registry,
    )
    n_bins = amides + spec.hex + len(shape) - 1 + config.zero_filling
    undeut = chem.mz(peptide, charge, spec.ion_type, mod=spec, registry=registry)
    return BinPlan(predicted_mz(undeut, charge, n_bins), amides, undeut, charge, sites,
                   (shape, offsets))


COMPOSITION_PASSES = 2  # estimates of the deuteration level for the composition ladder
D_MINUS_H = 2.01410177812 - 1.00782503207  # mass of deuterium minus hydrogen (Da)


def composition_mz(plan: BinPlan, deuteration: float) -> np.ndarray:
    """Predicted m/z of every bin of ``plan`` from the composition: bin ``n`` is the cluster of
    nominal mass +n, the natural isotope clusters convolved with a binomial distribution of
    deuterium over the exchangeable sites with mean ``deuteration``, and sits at the cluster's
    abundance-weighted mass. Bins beyond the clusters' reach continue by one deuterium step."""
    shape, offsets = plan.clusters
    n = len(plan.pred_mz)
    s = plan.n_sites
    p = min(max(deuteration / s, 1e-9), 1 - 1e-9) if s > 0 else 0.0
    k = np.arange(s + 1)
    d = binom.pmf(k, s, p) if s > 0 else np.array([1.0])
    weight = np.zeros(n)
    mass = np.zeros(n)
    for j, (a, off) in enumerate(zip(shape, offsets, strict=True)):
        idx = j + k
        keep = idx < n
        w = d[keep] * a
        weight[idx[keep]] += w
        mass[idx[keep]] += w * (k[keep] * D_MINUS_H + off)
    known = weight > 1e-12 * weight.max()
    out = np.full(n, np.nan)
    out[known] = mass[known] / weight[known]
    first = int(np.flatnonzero(known)[0])
    out[:first] = out[first] - (first - np.arange(first)) * D_MINUS_H  # not reached in practice
    for i in range(first + 1, n):
        if not known[i]:
            out[i] = out[i - 1] + D_MINUS_H
    return plan.undeut_mz + out / plan.charge


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
    mz: np.ndarray, y: np.ndarray, pred: np.ndarray, config: PickerConfig, charge: int,
    recorded: Mapping[str, float | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Window maximum for every bin (0 where the window is empty or rejected as a shoulder),
    and where it is: the m/z of the maximum, refined by a parabola through it and its two
    neighbours in the spectrum when they are lower (NaN where the value is 0). The window is
    ``config.half_width`` either side of the predicted m/z (``recorded``: the spectrum's
    recorded resolving power and its reference m/z)."""
    lower, upper = config.window_edges(pred, charge, **recorded)
    lo = np.searchsorted(mz, lower, side="left")
    hi = np.searchsorted(mz, upper, side="right")
    values = np.zeros(len(pred))
    where = np.full(len(pred), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
        window = y[a:b]
        if len(window) == 0:
            continue
        top = int(np.argmax(window))  # first maximum, so ties are resolved deterministically
        if len(window) >= config.shoulder_min_points and top in (0, len(window) - 1):
            continue  # the maximum is at the window edge: a shoulder of a neighbouring peak
        values[i] = window[top]
        if values[i] > 0:
            where[i] = _apex(mz, y, a + top, a, b)
    return values, where


def _apex(mz: np.ndarray, y: np.ndarray, j: int, a: int, b: int) -> float:
    """The m/z of the local maximum at point ``j`` of the window ``a:b``: the vertex of the
    parabola through it and its neighbours when both are inside the window, lower and evenly
    spaced (profile data), else the point itself (centroided data, or an edge)."""
    if a < j < b - 1 and y[j - 1] < y[j] > y[j + 1] and y[j - 1] > 0 and y[j + 1] > 0:
        x0, x1, x2 = mz[j - 1], mz[j], mz[j + 1]
        if max(x1 - x0, x2 - x1) < 2.5 * min(x1 - x0, x2 - x1):  # no gap in the profile
            y0, y1, y2 = y[j - 1], y[j], y[j + 1]
            den = (x0 - x1) * (x0 - x2) * (x1 - x2)
            a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
            b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
            if a < 0:
                return float(min(max(-b / (2 * a), x0), x2))
    return float(mz[j])


def mz_offset_ppm(peak_mz: np.ndarray, pred_mz: np.ndarray, intensity: np.ndarray,
                  threshold: float) -> float:
    """How far the picked peaks are from their predicted m/z: the intensity-weighted median of
    ``(peak - predicted) / predicted`` in ppm over the bins at or above ``threshold`` (NaN if
    there are none)."""
    keep = np.isfinite(peak_mz) & (intensity >= threshold) & (intensity > 0)
    if not keep.any():
        return float("nan")
    ppm = 1e6 * (peak_mz[keep] - pred_mz[keep]) / pred_mz[keep]
    w = intensity[keep]
    order = np.argsort(ppm)
    cum = np.cumsum(w[order])
    return float(ppm[order][np.searchsorted(cum, 0.5 * cum[-1])])


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
    resolving_power: float | None = None,
    resolving_power_mz: float | None = None,
) -> PickedSpectrum:
    """Pick one spectrum (arrays of m/z and intensity; sorted by m/z if they are not).
    ``resolving_power`` and ``resolving_power_mz`` are the spectrum's recorded values, used
    when ``config.resolving_power`` is ``'recorded'`` (:meth:`PickerConfig.half_width`)."""
    recorded = {"resolving_power": resolving_power, "resolving_power_mz": resolving_power_mz}
    plan = plan or plan_bins(
        peptide, charge, modification=modification, config=config, registry=registry
    )
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if len(mz) > 1 and np.any(np.diff(mz) < 0):
        order = np.argsort(mz, kind="stable")
        mz, intensity = mz[order], intensity[order]
    if config.noise_threshold:
        threshold = float(config.noise_threshold)
    elif len(intensity):
        threshold = config.noise_percent / 100.0 * float(intensity.max())
    else:
        threshold = float("nan")

    pred = plan.pred_mz
    values, where = _bin_values(mz, intensity, pred, config, charge, recorded)
    if config.ladder == "composition" and plan.clusters is not None:
        # the deuteration level from a first pass with the ExMS ladder, then the composition
        # ladder; the level is estimated once more from those picks (a first pass that missed
        # peaks biases it), and the bins are picked again
        natural = float((np.arange(len(plan.clusters[0])) * plan.clusters[0]).sum())
        for _ in range(COMPOSITION_PASSES):
            strong = np.where(values >= threshold, values, 0.0)
            if strong.sum() <= 0:
                break
            centre = float((np.arange(len(strong)) * strong).sum() / strong.sum())
            level = min(max(centre - natural, 0.0), float(plan.n_sites))
            pred = composition_mz(plan, level)
            values, where = _bin_values(mz, intensity, pred, config, charge, recorded)
    n_last = len(pred) - 1
    signal = np.flatnonzero(values >= threshold)  # False throughout if threshold is NaN
    last = int(signal[-1]) if len(signal) else -1
    end = min(max(last, 0) + config.zero_filling, n_last)
    picked_mz, picked_y = pred[: end + 1], values[: end + 1]
    peak_mz = where[: end + 1]

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
        peak_mz=peak_mz,
        mz_offset_ppm=mz_offset_ppm(peak_mz, picked_mz, picked_y, threshold),
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
    clusters: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> BinnedSet:
    """Pick every spectrum of ``spectra``. Nothing is dropped; see the flags on the result.
    ``clusters`` gives the isotope clusters of some series (``data_id``), for example an
    ion-type mixture (:mod:`hxex.picking.envelope`); the others use their ion's natural ones."""
    clusters = clusters or {}
    plans: dict[tuple, BinPlan] = {}
    picked: list[PickedSpectrum] = []
    if config.uses_recorded:
        missing = int((~np.isfinite(spectra.resolving_power)).sum())
        if missing:
            log.warning("resolving_power = \"recorded\": %d of %d spectra have no recorded "
                        "resolving power (not read from Thermo raw files), so their window is "
                        "resolution_ppm = %g", missing, len(spectra), config.resolution_ppm)
    for i in range(len(spectra)):
        check_cancel(cancel)
        series = int(spectra.data_id[i])
        own = series if series in clusters else None
        key = (str(spectra.peptide[i]), int(spectra.charge[i]), str(spectra.modification[i]), own)
        if key not in plans:
            plans[key] = plan_bins(
                key[0], key[1], modification=key[2], config=config, registry=registry,
                clusters=None if own is None else clusters[own],
            )
        mz, y = spectra.spectrum(i)
        picked.append(
            pick_spectrum(
                mz, y, key[0], key[1], modification=key[2], config=config, registry=registry,
                plan=plans[key], resolving_power=float(spectra.resolving_power[i]),
                resolving_power_mz=float(spectra.resolving_power_mz[i]),
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
        peak_mz=np.concatenate([p.peak_mz for p in picked]) if picked else empty,
        mz_offset_ppm=cat("mz_offset_ppm", np.float64),
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
        rt=spectra.rt,
        role=spectra.role,
    )
