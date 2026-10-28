# SPDX-License-Identifier: GPL-3.0-or-later
"""What value a picked peak gives its bin: its height or its area (``[run.picker] measure``).

On some instruments (the Q-TOF exports measured so far) weak peaks are wider than tall ones, so
a weak peak's height under-reads it against a tall one and the heavy side of an envelope reads
low. Its area does not: the undeuterated controls then match their natural-abundance envelope
far better. On others (an Orbitrap's reduced profiles) the widths do not change with height and
the two agree, so heights are kept (docs/HISTORY.md, items 181 and 185).

``measure = 'auto'`` (the default) decides per run from its undeuterated controls in profile
mode (:func:`choose_measure`): each control is picked once with heights, the full width at half
maximum (FWHM) of its tallest peak and of its weak peaks (``WEAK_SHARE`` of the tallest) is
measured from the raw profile, and the run takes areas when the median ratio of a weak peak's
width to its spectrum's tallest, over the brighter half of the controls (dim spectra give
unreliable widths), is at least ``WIDTH_EFFECT``, from at least ``MIN_CONTROLS`` controls. Without
enough controls, or with centroided spectra, the run keeps heights.

An area (:func:`area_values`) is the integral of the profile over ``area_fwhm`` peak widths
either side of the peak's apex (never beyond 0.45 of the isotope spacing), divided by the area
that the run's tallest peaks have per unit of height and width (measured from the controls, so
it holds whatever their line shape): a peak of the run's width keeps its height on average, a
wider one counts for more. The width at an m/z comes from the controls' tallest peaks, as a
power law in m/z (flat on a TOF, rising as its square root on an Orbitrap), measured, never
assumed from the instrument. An area also does not depend on where the profile points fall on
the peak, as a height does: with two or three points across a peak, the tallest point can lie
up to half a point spacing from the apex and read up to a tenth low.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import erf
from scipy.stats import theilslopes

from ..spectra import UNDEUT
from .config import ISOTOPE_STEP, PickerConfig
from .screen import STICK_POINTS

log = logging.getLogger(__name__)

WIDTH_EFFECT = 1.06  # weak / tallest peak width at or above this: the run takes areas
MIN_CONTROLS = 20  # undeuterated controls with a measured width needed to decide
WEAK_SHARE = (0.05, 0.30)  # weak peaks: this share of their spectrum's tallest
GAP_SPACING = 2.5  # profile points further apart than this many median spacings: a gap (as pick)
WINDOW_CAP = 0.9  # the area's half-range is at most this share of half the isotope spacing
REFERENCE_MZ = 1000.0  # the m/z at which the peak width is quoted
_GAUSS = math.sqrt(math.pi / (4.0 * math.log(2.0)))  # area of a Gaussian, unit height and FWHM
_ERF = 2.0 * math.sqrt(math.log(2.0))  # erf(_ERF * x): its share within x FWHM of the apex


@dataclass(frozen=True)
class PeakMeasure:
    """The run's choice of bin value and what it was made from.

    ``measure`` is ``'height'`` or ``'area'``; ``setting`` the ``[run.picker] measure`` it came
    from. ``fwhm_ppm`` is the peak width at m/z ``REFERENCE_MZ`` and ``fwhm_exponent`` how it
    grows with m/z (``fwhm(m) = fwhm_ppm (m / 1000) ** e`` ppm), from the controls' tallest peaks;
    ``width_ratio`` the median weak / tallest width over the brighter half of the controls, from
    ``n_controls`` controls with a measured width (``n_weak`` weak peaks); ``area_fwhm`` the
    area's half-range in peak widths; ``reason`` says in words why. ``area_scale`` is the
    median area of the controls' tallest peaks over that range per unit of height (their
    tallest point) and of width (NaN: a Gaussian's, ``1.064`` at 1.5 widths)."""

    measure: str
    setting: str
    fwhm_ppm: float = float("nan")
    fwhm_exponent: float = 0.0
    width_ratio: float = float("nan")
    n_controls: int = 0
    n_weak: int = 0
    area_fwhm: float = 1.5
    reason: str = ""
    area_scale: float = float("nan")

    def fwhm_mz(self, mz) -> np.ndarray:
        """The peak width (FWHM, in m/z) at each ``mz``."""
        mz = np.asarray(mz, dtype=np.float64)
        return mz * 1e-6 * self.fwhm_ppm * (mz / REFERENCE_MZ) ** self.fwhm_exponent

    def to_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Mapping) -> PeakMeasure:
        values = {k: (float("nan") if v is None and k in _NAN_FIELDS else v)
                  for k, v in dict(data).items()}
        return cls(**values)

    def describe(self) -> str:
        """One line for people: the measure and why."""
        what = "areas" if self.measure == "area" else "heights"
        width = (f"; peak width {self.fwhm_ppm:.1f} ppm at m/z {REFERENCE_MZ:g}"
                 if math.isfinite(self.fwhm_ppm) else "")
        return f"{what} ({self.reason}){width}"


HEIGHTS = PeakMeasure("height", "height", reason="set")
_NAN_FIELDS = ("fwhm_ppm", "width_ratio", "area_scale")


def _profile(mz: np.ndarray, charge: int) -> tuple[bool, float]:
    """Whether a spectrum is in profile mode (enough points per isotope step), and its median
    point spacing."""
    d = np.diff(mz)
    spacing = float(np.median(d[d > 0])) if (d > 0).any() else math.inf
    return len(mz) >= 3 and (ISOTOPE_STEP / max(charge, 1)) / spacing >= STICK_POINTS, spacing


def peak_fwhm(mz: np.ndarray, y: np.ndarray, apex: float, spacing: float) -> float:
    """The full width at half maximum, in ppm of ``apex``, of the profile peak at ``apex``: from
    the point nearest it (moved to the taller neighbour while one is taller) out to where the
    profile falls below half its height, interpolated linearly; NaN at the spectrum's edge or
    where a gap in the profile (``GAP_SPACING``) lies inside the half-height range."""
    n = len(mz)
    if n < 3 or not math.isfinite(apex):
        return float("nan")
    j = int(np.clip(np.searchsorted(mz, apex), 1, n - 1))
    j = j if abs(mz[j] - apex) < abs(mz[j - 1] - apex) else j - 1
    while 0 < j < n - 1 and max(y[j - 1], y[j + 1]) > y[j]:
        j = j - 1 if y[j - 1] > y[j + 1] else j + 1
    half = 0.5 * float(y[j])
    if not half > 0:
        return float("nan")
    gap = GAP_SPACING * spacing
    left = j
    while left > 0 and y[left] > half:
        if mz[left] - mz[left - 1] > gap:
            return float("nan")
        left -= 1
    right = j
    while right < n - 1 and y[right] > half:
        if mz[right + 1] - mz[right] > gap:
            return float("nan")
        right += 1
    if y[left] > half or y[right] > half or left == j or right == j:
        return float("nan")
    xl = mz[left] + (half - y[left]) / (y[left + 1] - y[left]) * (mz[left + 1] - mz[left])
    xr = mz[right - 1] + (y[right - 1] - half) / (y[right - 1] - y[right]) * (
        mz[right] - mz[right - 1])
    return float((xr - xl) / apex * 1e6)


def _cumulative(mz: np.ndarray, y: np.ndarray, spacing: float) -> np.ndarray:
    """The profile's running integral at each point (trapezoids; a gap counts as zero)."""
    dx = np.diff(mz)
    seg = 0.5 * (y[1:] + y[:-1]) * dx * (dx <= GAP_SPACING * spacing)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _integral_to(mz, y, cum, spacing, x) -> np.ndarray:
    """The integral of the linearly interpolated profile from its first point to each ``x``."""
    x = np.clip(np.asarray(x, dtype=np.float64), mz[0], mz[-1])
    k = np.clip(np.searchsorted(mz, x, side="right") - 1, 0, len(mz) - 2)
    h = mz[k + 1] - mz[k]
    t = x - mz[k]
    with np.errstate(divide="ignore", invalid="ignore"):
        yx = y[k] + (y[k + 1] - y[k]) * np.where(h > 0, t / h, 0.0)
    part = np.where(h <= GAP_SPACING * spacing, 0.5 * t * (y[k] + yx), 0.0)
    return cum[k] + part


def _raw_areas(mz, y, centres, half, spacing) -> np.ndarray:
    """The integral of the profile over ``centres -/+ half`` (``mz`` sorted)."""
    cum = _cumulative(mz, y, spacing)
    return (_integral_to(mz, y, cum, spacing, centres + half)
            - _integral_to(mz, y, cum, spacing, centres - half))


def _half_range(measure: PeakMeasure, width: np.ndarray, charge: int) -> np.ndarray:
    return np.minimum(measure.area_fwhm * width,
                      WINDOW_CAP * ISOTOPE_STEP / (2.0 * max(int(charge), 1)))


def area_values(mz: np.ndarray, y: np.ndarray, where: np.ndarray, values: np.ndarray,
                charge: int, measure: PeakMeasure) -> np.ndarray:
    """The area of each picked peak (apex ``where``; ``values`` its height, 0 where nothing was
    picked) as a height-equivalent (see the module docstring); ``values`` unchanged for a
    centroided spectrum. ``mz`` sorted."""
    profile, spacing = _profile(mz, charge)
    out = np.array(values, dtype=np.float64)
    use = np.flatnonzero((out > 0) & np.isfinite(where))
    if not profile or not len(use):
        return out
    c = where[use]
    width = measure.fwhm_mz(c)
    half = _half_range(measure, width, charge)
    full = float(erf(_ERF * measure.area_fwhm))
    scale = measure.area_scale if math.isfinite(measure.area_scale) else _GAUSS * full
    norm = scale * width * erf(_ERF * half / width) / full  # a capped range holds less
    out[use] = np.where(norm > 0, _raw_areas(mz, y, c, half, spacing) / norm, out[use])
    return out


def choose_measure(spectra, config: PickerConfig, *, registry=None) -> PeakMeasure:
    """The run's :class:`PeakMeasure` under ``config.measure`` from the undeuterated controls
    (role ``'undeut'``) of ``spectra`` (a :class:`~hxex.spectra.SpectrumSet`); see the module
    docstring. ``'height'`` is returned without measuring."""
    from .. import chem
    from .pick import pick_spectrum, plan_bins

    if config.measure == "height":
        return HEIGHTS
    registry = registry or chem.DEFAULT_MODIFICATIONS
    role = np.asarray(spectra.role if spectra.role is not None else [""] * len(spectra), dtype=str)
    rows = np.flatnonzero(role == UNDEUT)
    heights = PickerConfig.from_dict({**config.to_dict(), "measure": "height"})
    plans: dict = {}
    tops, top_fw, top_mz, weak, tall = [], [], [], [], []
    n_centroided = 0
    for i in rows:
        mz, y = spectra.spectrum(int(i))
        mz, y = np.asarray(mz, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if len(mz) > 1 and np.any(np.diff(mz) < 0):
            order = np.argsort(mz, kind="stable")
            mz, y = mz[order], y[order]
        charge = int(spectra.charge[i])
        profile, spacing = _profile(mz, charge)
        if not profile:
            n_centroided += 1
            continue
        key = (str(spectra.peptide[i]), charge, str(spectra.modification[i]))
        if key not in plans:
            plans[key] = plan_bins(key[0], key[1], modification=key[2], config=heights,
                                   registry=registry)
        p = pick_spectrum(mz, y, key[0], key[1], modification=key[2], config=heights,
                          registry=registry, plan=plans[key])
        v, apex = np.asarray(p.intensity, dtype=np.float64), np.asarray(p.peak_mz, dtype=np.float64)
        if not len(v) or not v.max() > 0:
            continue
        t = int(np.argmax(v))
        w0 = peak_fwhm(mz, y, apex[t], spacing)
        if not math.isfinite(w0) or not w0 > 0:
            continue
        tops.append(float(v[t]))
        top_fw.append(w0)
        top_mz.append(float(apex[t]))
        near = (mz >= apex[t] - 0.5 * ISOTOPE_STEP / charge) & (
            mz <= apex[t] + 0.5 * ISOTOPE_STEP / charge)
        tall.append((mz[near], y[near], float(apex[t]), float(v[t]), charge, spacing))
        share = v / v[t]
        ratios = [peak_fwhm(mz, y, apex[k], spacing) / w0
                  for k in np.flatnonzero((share >= WEAK_SHARE[0]) & (share < WEAK_SHARE[1]))]
        weak.append([r for r in ratios if math.isfinite(r)])
    n = len(tops)
    if n == 0:
        reason = ("the spectra are centroided" if n_centroided
                  else "no undeuterated control with a measurable peak width")
        if config.measure == "area":
            log.warning("measure = \"area\": %s, so the bins take heights", reason)
        return PeakMeasure("height", config.measure, area_fwhm=config.area_fwhm, reason=reason)
    fw, fmz = np.array(top_fw), np.array(top_mz)
    lx, ly = np.log(fmz / REFERENCE_MZ), np.log(fw)
    if n >= 5 and float(fmz.max() / fmz.min()) > 1.2:
        slope, intercept = theilslopes(ly, lx)[:2]
    else:
        slope, intercept = 0.0, float(np.median(ly))
    bright = np.array(tops) >= np.median(tops)
    pooled = [r for b, rs in zip(bright, weak, strict=True) if b for r in rs]
    ratio = float(np.median(pooled)) if pooled else float("nan")
    base = dict(setting=config.measure, fwhm_ppm=float(np.exp(intercept)),
                fwhm_exponent=float(slope), width_ratio=ratio, n_controls=n,
                n_weak=len(pooled), area_fwhm=config.area_fwhm)
    law = PeakMeasure("area", config.measure, base["fwhm_ppm"], base["fwhm_exponent"],
                      area_fwhm=config.area_fwhm)
    per = []
    for x, v, c, h, z, sp in tall:
        width = law.fwhm_mz(np.array([c]))
        half = _half_range(law, width, z)
        if half[0] < config.area_fwhm * width[0]:
            continue  # capped: not the full range
        per.append(float(_raw_areas(x, v, np.array([c]), half, sp)[0] / (h * width[0])))
    base["area_scale"] = float(np.median(per)) if per else float("nan")
    if config.measure == "area":
        return PeakMeasure("area", reason="set", **base)
    if n < MIN_CONTROLS or not pooled:
        return PeakMeasure("height", reason=f"{n} undeuterated controls with a measured peak "
                           f"width, fewer than the {MIN_CONTROLS} needed to judge the width "
                           "effect", **base)
    if ratio >= WIDTH_EFFECT:
        return PeakMeasure("area", reason=f"weak peaks {ratio:.2f} times as wide as the tallest",
                           **base)
    return PeakMeasure("height", reason=f"weak peaks {ratio:.2f} times as wide as the tallest, "
                       f"below {WIDTH_EFFECT:g}", **base)
