# SPDX-License-Identifier: GPL-3.0-or-later
"""Bin values from a fit of the raw profile (``[run.picker] pick = "profile_fit"``).

Today's pick (``pick = "peak"``) gives every bin the tallest peak whose apex lies in its window.
The profile fit reads the same bins from a fit of the profile around the whole envelope
(docs/PLAN.md, Next steps 31; docs/HISTORY.md, item 383 and this build's item):

1. The pick runs first. Its values give a deuterium distribution: a non-negative fit of the
   natural-abundance shape shifted along the deuterium grid, any shape and any number of
   populations (:func:`deuterium_distribution`).
2. Each bin sits at the abundance-weighted mass of its isotopologues, deuterium and the fine
   structure of 13C, 15N, 18O, 34S by nominal shell (:func:`bin_positions`); its peak is the
   run's peak widened by those isotopologues' mass spread.
3. The spectrum's m/z offset is searched about its injection's median offset, ``SEARCH_SD``
   times the run's robust scatter either side and at least ``SEARCH_FLOOR_PPM``
   (:func:`search_offsets`). An injection is a scan file (mzML, Thermo raw), else a sample,
   role, time, and replicate.
4. Non-negative least squares on the profile points within ``REGION_WIDTHS`` peak widths of the
   bins: one peak per bin, with the run's measured shape (:class:`PeakShape`), and free stray
   peaks at the profile's local maxima (above ``STRAY_MIN`` of the region's tallest point) more
   than ``STRAY_WIDTHS`` peak widths from every bin position, so that a neighbouring species
   has somewhere else to go.
5. Two passes: the second takes its positions from the first pass's values.
6. A bin's value is its peak's area as a height-equivalent: the height of a peak of the run's
   width with the same area. A bin without a profile point within about 0.6 widths of its
   position reads 0 (sparse profiles leave zero points out).
7. Two quality numbers per spectrum (:func:`fit_quality`): ``purity``, the share of the profile
   the bins' peaks account for within ``FOOTPRINT_WIDTHS`` peak widths of each bin (bins whose
   fitted peak reaches ``PURITY_MIN`` of the tallest bin's, weighted by their fitted peak; what
   lies between the bins does not count, so a species resolved from the peptide's peaks costs
   nothing), and ``stray_share``, the share of the whole region the stray peaks account for.
   Measured (2026-11-14, docs/HISTORY.md item 384): low purity picks out spectra far from their
   replicates as well as or better than a whole-region share, and keeps one scale on Q-TOF and
   Orbitrap data (median 0.94-0.97).

The offset search is a grid of ``SEARCH_STEPS`` steps either side (steps no finer than
``MIN_STEP_PPM``). Where the run's peak width is flat in m/z (a time-of-flight instrument,
``|fwhm_exponent| < COARSE_EXPONENT``) each pass searches coarse then fine: every third offset,
then the steps around the best. On the Q-TOF benchmark that chose the full grid's offset in all
but 3 of 1,500 spectra at half the time (a second pass that tried only the first pass's offset
and its neighbours missed it in 18); on an Orbitrap the coarse search missed it in 8 %, so the
full grid stays there.

Fragment ions, ion types other than ``M``, series with their own envelopes
(``[run.envelope]``), and centroided spectra keep the pick.

With ``pick = "peak"`` the same spectra get a purity too (:func:`hxex.picking.pick_spectra`):
one fit at the spectrum's own offset (:func:`own_offset`, the injection's median where it has
none), with no search, whose ``purity`` and ``stray_share`` are kept and whose values are not
(the bins stay the pick's). Measured on four datasets (2026-11-22, docs/HISTORY.md, item
436): about 3-5 ms a spectrum against 21-45 ms for the searched fit; Spearman 0.93-
0.98 with the searched fit's purity, and the spectra it puts below 0.7 that no other check
marks are as far from their replicates as the searched fit's. It misses some: where a
neighbouring species' peaks are the brightest, the own offset follows them.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import nnls

from .. import chem

log = logging.getLogger(__name__)

D_MINUS_H = 2.01410177812 - 1.00782503207  # mass of deuterium minus hydrogen (Da)
FWHM_SD = 2.0 * math.sqrt(2.0 * math.log(2.0))  # a Gaussian's FWHM over its sd (2.3548)
GAUSS_AREA = math.sqrt(math.pi / (4.0 * math.log(2.0)))  # a Gaussian's area, unit height and FWHM
REGION_WIDTHS = 1.5  # the fit reads the profile this many peak widths beyond the end bins
STRAY_WIDTHS = 1.0  # a local maximum further than this from every bin is a stray peak
STRAY_MIN = 0.002  # stray candidates: local maxima above this share of the region's tallest
STRAY_CLASS = 0.4  # the shape class of a stray peak (a mid-height peak)
SEEN = 0.5  # a bin is fitted when its peak reaches this share of its height at a profile point
PASSES = 2
SEARCH_SD = 3.0  # the offset search reaches this many times the run's scatter either side
SEARCH_FLOOR_PPM = 3.0  # and at least this far
SEARCH_STEPS = 12  # steps either side of the injection's median
MIN_STEP_PPM = 0.25
COARSE_EXPONENT = 0.25  # |fwhm_exponent| below this: a TOF's flat width, the coarse search
FOOTPRINT_WIDTHS = 1.0  # purity reads the profile this many peak widths either side of a bin
PURITY_MIN = 0.05  # bins whose fitted peak is below this share of the tallest bin's do not count
BRIGHT = 0.5  # a spectrum's own offset: its bins at or above this share of the tallest
_MAD_SD = 1.4826
# the peak shape: classes by the bin's expected share of its spectrum's tallest
SHAPE_CLASSES = (0.6, 0.2, 0.05)  # lower edges: >= 60 %, 20-60 %, 5-20 %
SHAPE_GRID = (-3.0, 3.0, 241)  # u = (m/z - apex) / FWHM
SHAPE_REACH = 3.2  # profile points within this many widths of the apex make a curve
SHAPE_MIN_POINTS = 8
MIN_SHAPE_CONTROLS = 20  # undeuterated controls needed to measure the shape (as measure's)
MIN_SHAPE_PEAKS = 20  # peaks needed in a class (fewer: that class is a Gaussian)
PARALLEL_FROM = 400  # spectra: fewer are fitted in this process
MAX_WORKERS = 8
CHUNK = 64  # spectra per task in a process pool
# a worker process runs one thread of linear algebra: with the libraries' own threads on top,
# eight workers took every core of a 48-core machine many times over
_ONE_THREAD = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

# exact isotope masses (u) and abundances (IUPAC/NIST)
ISOTOPES = {
    "C": [(12.0, 0.9893), (13.0033548378, 0.0107)],
    "H": [(1.00782503207, 0.999885), (2.0141017778, 0.000115)],
    "N": [(14.0030740048, 0.99636), (15.0001088982, 0.00364)],
    "O": [(15.99491461956, 0.99757), (16.99913170, 0.00038), (17.9991610, 0.00205)],
    "S": [(31.97207100, 0.9499), (32.97145876, 0.0075), (33.96786690, 0.0425),
          (35.96708076, 0.0001)],
    "P": [(30.97376163, 1.0)],
}
SHELLS = 60


# --- positions -----------------------------------------------------------------------------

def _conv(a, b):
    """Convolve two moment arrays (abundance, first and second mass moments) over shells."""
    c = lambda x, y: np.convolve(x, y)[:SHELLS]  # noqa: E731
    return (c(a[0], b[0]), c(a[1], b[0]) + c(a[0], b[1]),
            c(a[2], b[0]) + 2 * c(a[1], b[1]) + c(a[0], b[2]))


def _element(el: str):
    light = ISOTOPES[el][0][0]
    p, m1, m2 = np.zeros(SHELLS), np.zeros(SHELLS), np.zeros(SHELLS)
    for mass, abundance in ISOTOPES[el]:
        s = int(round(mass - light))
        defect = mass - light - s
        p[s] += abundance
        m1[s] += abundance * defect
        m2[s] += abundance * defect * defect
    return p, m1, m2


def _power(x, n: int):
    out = (np.r_[1.0, np.zeros(SHELLS - 1)], np.zeros(SHELLS), np.zeros(SHELLS))
    while n:
        if n & 1:
            out = _conv(out, x)
        x = _conv(x, x)
        n >>= 1
    return out


_MOMENTS: dict = {}


def fine_moments(composition: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per nominal shell of ``composition``: abundance (sums to 1), mean mass offset from the
    monoisotopic mass (Da; the shell plus its mass defect), and the variance of the
    isotopologue masses in it (Da^2). Elements other than C, H, N, O, S, and P add no isotopes."""
    key = tuple(sorted((k, int(v)) for k, v in composition.items()))
    if key in _MOMENTS:
        return _MOMENTS[key]
    out = (np.r_[1.0, np.zeros(SHELLS - 1)], np.zeros(SHELLS), np.zeros(SHELLS))
    for el, n in key:
        if n > 0 and el in ISOTOPES:
            out = _conv(out, _power(_element(el), n))
    p, m1, m2 = out
    last = int(np.flatnonzero(p > 1e-9 * p.max())[-1]) + 1
    p, m1, m2 = p[:last], m1[:last], m2[:last]
    mean = m1 / p
    result = (p / p.sum(), np.arange(last) + mean, np.maximum(m2 / p - mean * mean, 0.0))
    _MOMENTS[key] = result
    return result


def ion_moments(peptide: str, *, modification: str = "", nterm_subtract: int = 1,
                rules: chem.AmideRules = chem.DEFAULT_AMIDE_RULES,
                registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS):
    """:func:`fine_moments` of the ion's non-exchangeable composition (the exchangeable sites'
    hydrogens are what deuterium replaces)."""
    spec = chem.parse_modification(modification)
    comp = dict(chem.composition(peptide, spec.ion_type, mod=spec, registry=registry))
    comp["H"] = comp.get("H", 0) - chem.exchangeable_sites(
        peptide, mod=spec, ion_type=spec.ion_type, nterm_subtract=nterm_subtract, rules=rules,
        registry=registry)
    return fine_moments(comp)


def deuterium_distribution(sticks: np.ndarray, shape: np.ndarray, n_sites: int) -> np.ndarray:
    """The non-negative deuterium distribution ``d`` over ``0..n_sites`` (as far as the bins
    reach) with ``sticks ~ sum_k d_k shape[i - k]``."""
    n = len(sticks)
    kmax = max(min(int(n_sites), n - 1), 0)
    basis = np.zeros((n, kmax + 1))
    for k in range(kmax + 1):
        m = min(len(shape), n - k)
        basis[k:k + m, k] = shape[:m]
    return nnls(basis, np.asarray(sticks, dtype=np.float64))[0]


def bin_positions(d: np.ndarray, moments, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean mass offset (Da, from the monoisotopic mass) and mass sd (Da) of bins
    ``0..n_bins-1`` for the deuterium distribution ``d``; a bin that holds nothing continues
    from the one before it by one deuterium step (the first by 0), with sd 0."""
    a, mu, var = moments
    w_all, m1_all, m2_all = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    for k, dk in enumerate(d):
        if dk <= 0:
            continue
        idx = k + np.arange(len(a))
        keep = idx < n_bins
        w = dk * a[keep]
        m = k * D_MINUS_H + mu[keep]
        w_all[idx[keep]] += w
        m1_all[idx[keep]] += w * m
        m2_all[idx[keep]] += w * (var[keep] + m * m)
    mean = np.full(n_bins, np.nan)
    sd = np.zeros(n_bins)
    has = w_all > 1e-12 * max(w_all.max(initial=0.0), 1e-300)
    mean[has] = m1_all[has] / w_all[has]
    sd[has] = np.sqrt(np.maximum(m2_all[has] / w_all[has] - mean[has] ** 2, 0.0))
    for b in range(n_bins):
        if not np.isfinite(mean[b]):
            mean[b] = mean[b - 1] + D_MINUS_H if b else 0.0
    return mean, sd


# --- the run's peak shape ------------------------------------------------------------------

def _class_of(share: np.ndarray) -> np.ndarray:
    """Shape class (0: >= 60 %, 1: 20-60 %, 2: below) of bins of relative height ``share``."""
    share = np.asarray(share, dtype=np.float64)
    return np.where(share >= SHAPE_CLASSES[0], 0, np.where(share >= SHAPE_CLASSES[1], 1, 2))


@dataclass(frozen=True)
class PeakShape:
    """The run's peak shape, a median profile in units of the run's peak width
    (``u = (m/z - apex) / FWHM``), per class of a bin's share of its spectrum's tallest
    (``SHAPE_CLASSES``), each at 1 at its apex; ``n_peaks`` the peaks each was measured from (0:
    a Gaussian, too few peaks), ``n_controls`` the undeuterated controls they came from."""

    rows: tuple[np.ndarray, np.ndarray, np.ndarray]
    n_peaks: tuple[int, int, int] = (0, 0, 0)
    n_controls: int = 0

    @staticmethod
    def grid() -> np.ndarray:
        return np.linspace(*SHAPE_GRID[:2], SHAPE_GRID[2])

    @classmethod
    def gaussian(cls, n_controls: int = 0) -> PeakShape:
        g = np.exp(-0.5 * (cls.grid() * FWHM_SD) ** 2)
        return cls((g, g.copy(), g.copy()), (0, 0, 0), n_controls)

    @property
    def measured(self) -> bool:
        return any(self.n_peaks)

    def __call__(self, u: np.ndarray, share: np.ndarray) -> np.ndarray:
        """The shape at ``u`` (points x peaks) for peaks of relative height ``share``."""
        grid = self.grid()
        out = np.zeros_like(u)
        cls = _class_of(share)
        for c in range(3):
            k = cls == c
            if k.any():
                out[:, k] = np.interp(u[:, k], grid, self.rows[c], left=0.0, right=0.0)
        return out

    def integral(self, share: np.ndarray) -> np.ndarray:
        """The area of the shape, in widths, for peaks of relative height ``share``."""
        grid = self.grid()
        areas = np.array([trapezoid(r, grid) for r in self.rows])
        return areas[_class_of(share)]

    def to_dict(self) -> dict:
        return {"grid": list(SHAPE_GRID), "rows": [[round(float(v), 5) for v in r]
                                                   for r in self.rows],
                "n_peaks": list(self.n_peaks), "n_controls": int(self.n_controls)}

    @classmethod
    def from_dict(cls, data: Mapping) -> PeakShape:
        grid = data.get("grid", list(SHAPE_GRID))
        rows = [np.asarray(r, dtype=np.float64) for r in data["rows"]]
        if list(grid) != list(SHAPE_GRID):  # another grid: onto this one
            g = np.linspace(grid[0], grid[1], int(grid[2]))
            rows = [np.interp(cls.grid(), g, r, left=0.0, right=0.0) for r in rows]
        return cls(tuple(rows), tuple(int(n) for n in data.get("n_peaks", (0, 0, 0))),
                   int(data.get("n_controls", 0)))

    def describe(self) -> str:
        if not self.measured:
            few = self.n_controls < MIN_SHAPE_CONTROLS
            return "a Gaussian of the run's peak width" + (
                f" ({self.n_controls} undeuterated controls, fewer than the "
                f"{MIN_SHAPE_CONTROLS} needed to measure the shape)" if few else "")
        tenth = []
        grid = self.grid()
        for r in self.rows:
            above = grid[r >= 0.1]
            tenth.append(float(above.max() - above.min()) if len(above) else float("nan"))
        return (f"measured from {self.n_controls} undeuterated controls "
                f"({', '.join(str(n) for n in self.n_peaks)} peaks by height class); width at a "
                f"tenth {tenth[0]:.2f} peak widths (a Gaussian's 1.82)")


def shape_curves(mz: np.ndarray, y: np.ndarray, apex: np.ndarray, expected: np.ndarray,
                 charge: int, fwhm_mz) -> list[tuple[int, np.ndarray]]:
    """Each picked peak of one undeuterated control as a curve on the shape grid: ``(class,
    profile / its height at the apex)``, for the bins whose ``expected`` share of the tallest
    (natural abundance) is at least the lowest class's and whose apex is known; a peak whose
    neighbours would enter ``SHAPE_REACH`` widths, or with too few points, gives none.
    ``fwhm_mz`` gives the run's peak width at an m/z."""
    grid = PeakShape.grid()
    out = []
    for b in range(min(len(apex), len(expected))):
        if not np.isfinite(apex[b]) or expected[b] < SHAPE_CLASSES[2]:
            continue
        fw = float(fwhm_mz(np.array([apex[b]]))[0])
        if not fw > 0 or 3 * fw > 0.45 / max(int(charge), 1):
            continue
        k = np.abs(mz - apex[b]) <= SHAPE_REACH * fw
        if k.sum() < SHAPE_MIN_POINTS:
            continue
        top = float(np.interp(apex[b], mz[k], y[k]))
        if not top > 0:
            continue
        out.append((int(_class_of(np.array([expected[b]]))[0]),
                    np.interp(grid, (mz[k] - apex[b]) / fw, y[k] / top, left=np.nan,
                              right=np.nan)))
    return out


def peak_shape(curves: Sequence[tuple[int, np.ndarray]], n_controls: int) -> PeakShape:
    """The :class:`PeakShape` from the controls' ``curves`` (:func:`shape_curves`): the median
    per class, a Gaussian for a class with fewer than ``MIN_SHAPE_PEAKS`` curves, and for all
    of them with fewer than ``MIN_SHAPE_CONTROLS`` controls."""
    gauss = PeakShape.gaussian(n_controls)
    if n_controls < MIN_SHAPE_CONTROLS:
        return gauss
    rows, counts = [], []
    for c in range(3):
        cs = [v for k, v in curves if k == c]
        if len(cs) < MIN_SHAPE_PEAKS:
            rows.append(gauss.rows[c])
            counts.append(0)
            continue
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a grid point no curve reaches
            m = np.nanmedian(np.array(cs), axis=0)
        rows.append(np.where(np.isfinite(m), np.clip(m, 0.0, None), 0.0))
        counts.append(len(cs))
    return PeakShape(tuple(rows), tuple(counts), n_controls)


# --- one spectrum --------------------------------------------------------------------------

@dataclass(frozen=True)
class ProfileFit:
    """The fit of one spectrum: ``values`` (height-equivalent areas, one per bin), ``where``
    (each bin's fitted m/z, NaN where its value is 0), ``offset_ppm`` (the spectrum's m/z
    offset), ``purity`` and ``stray_share`` (:func:`fit_quality`). With ``model``: the
    region's m/z (``x``) and profile there (``y``), the bins' and the strays' model curves, and
    the strays' apexes."""

    values: np.ndarray
    where: np.ndarray
    offset_ppm: float
    purity: float
    stray_share: float
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    bins_curve: np.ndarray | None = None
    stray_curve: np.ndarray | None = None
    strays: np.ndarray | None = None


def _nnls(m: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares of ``y`` on the columns of ``m``, solved on the normal
    equations (a Cholesky factor of the few columns' cross products: the same solution as on
    the long profile matrix, a fraction of the time)."""
    g = m.T @ m
    b = m.T @ y
    g = g + np.eye(len(g)) * (1e-12 * np.trace(g) / max(len(g), 1) + 1e-300)
    try:
        lower = np.linalg.cholesky(g)
    except np.linalg.LinAlgError:
        return nnls(m, y)[0]
    return nnls(lower.T, np.linalg.solve(lower, b))[0]


class _Pass:
    """One pass of :func:`fit_spectrum`: the region's profile ``x``, ``y``, the bins'
    positions ``pos`` and peak sds ``sig`` (m/z), their shares of the tallest, and the stray
    candidates; :meth:`solve` fits at one offset (cached)."""

    def __init__(self, x, y, pos, sig, share, fwhm_mz, shape: PeakShape):
        self.x, self.y, self.pos, self.sig, self.share, self.shape = x, y, pos, sig, share, shape
        inner = (y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > STRAY_MIN * y.max())
        self.apex = x[1:-1][inner]
        self.fw_apex = fwhm_mz(self.apex)
        self.stray_all = shape((x[:, None] - self.apex[None, :]) / self.fw_apex[None, :],
                               np.full(len(self.apex), STRAY_CLASS))
        self.cache: dict = {}

    def solve(self, off: float):
        """(residual norm, offset, amplitudes, bin peaks, strays' curve, strays' apexes)."""
        key = round(float(off), 9)
        if key in self.cache:
            return self.cache[key]
        x, n = self.x, len(self.pos)
        c = self.pos * (1.0 + off * 1e-6)
        peaks = self.shape((x[:, None] - c[None, :]) / (FWHM_SD * self.sig[None, :]),
                           self.share)
        seen = peaks.max(axis=0) >= SEEN
        amp = np.zeros(n)
        far = (np.abs(self.apex[:, None] - c[None, :]).min(axis=1) > STRAY_WIDTHS * self.fw_apex
               if len(self.apex) else np.zeros(0, dtype=bool))
        strays = self.stray_all[:, far]
        stray = np.zeros(len(x))
        if seen.any():
            sol = _nnls(np.hstack([peaks[:, seen], strays]), self.y)
            amp[seen] = sol[:seen.sum()]
            stray = strays @ sol[seen.sum():]
        resid = float(np.linalg.norm(self.y - peaks @ amp - stray))
        self.cache[key] = (resid, float(off), amp, peaks, stray, self.apex[far])
        return self.cache[key]


def fit_spectrum(mz: np.ndarray, y: np.ndarray, sticks: np.ndarray, plan, moments,
                 fwhm_mz, shape: PeakShape, *, centre: float, limit: float, step: float,
                 coarse: bool, model: bool = False) -> ProfileFit | None:
    """The profile fit of one spectrum (``mz`` sorted, ``y`` its intensities) whose pick gave
    ``sticks`` (one value per bin of ``plan``, a :class:`~hxex.picking.pick.BinPlan` with its
    natural clusters); ``moments`` from :func:`ion_moments`, ``fwhm_mz`` the run's peak width
    at an m/z. The offset is searched over ``centre -/+ limit`` ppm in steps of ``step``
    (``coarse``: see the module docstring). None when there is nothing to fit (no signal, or
    fewer than three profile points in the region)."""
    n = len(sticks)
    z = int(plan.charge)
    sticks = np.asarray(sticks, dtype=np.float64)
    if not n or not sticks.max() > 0 or plan.clusters is None:
        return None
    full = np.arange(centre - limit, centre + limit + 1e-9, step)
    best = None
    for _ in range(PASSES):
        d = deuterium_distribution(sticks, plan.clusters[0], plan.n_sites)
        mean, spread = bin_positions(d, moments, n)
        pos = plan.undeut_mz + mean / z
        sig = np.sqrt((fwhm_mz(pos) / FWHM_SD) ** 2 + (spread / z) ** 2)
        lo = pos[0] - REGION_WIDTHS * FWHM_SD * sig[0]
        hi = pos[-1] + REGION_WIDTHS * FWHM_SD * sig[-1]
        a, b = np.searchsorted(mz, lo, side="left"), np.searchsorted(mz, hi, side="right")
        if b - a < 3 or not y[a:b].max() > 0:
            return None
        one = _Pass(mz[a:b], y[a:b], pos, sig, sticks / sticks.max(), fwhm_mz, shape)
        if coarse:  # every third offset, then the steps around the best
            tried = [one.solve(o) for o in full[::3]]
            j = int(np.argmin(np.abs(full - min(tried, key=lambda t: t[0])[1])))
            tried += [one.solve(full[m]) for m in range(j - 2, j + 3) if 0 <= m < len(full)]
        else:
            tried = [one.solve(o) for o in full]
        best = min(tried, key=lambda t: t[0])
        sticks = best[2]
        if not sticks.max() > 0:
            break
    x, yy = one.x, one.y
    _, off, amp, peaks, stray, stray_at = best
    pos, sig = one.pos, one.sig
    share = amp / amp.max() if amp.max() > 0 else np.zeros(n)
    values = amp * (FWHM_SD * sig) * shape.integral(share) / (fwhm_mz(pos) * GAUSS_AREA)
    where = np.where(values > 0, pos * (1.0 + off * 1e-6), np.nan)
    purity, stray_share = fit_quality(x, yy, peaks * amp[None, :], stray, where, fwhm_mz)
    if not model:
        return ProfileFit(values, where, off, purity, stray_share)
    return ProfileFit(values, where, off, purity, stray_share, x, yy, peaks @ amp, stray, stray_at)


def fit_quality(x: np.ndarray, y: np.ndarray, bin_peaks: np.ndarray, stray: np.ndarray,
                where: np.ndarray, fwhm_mz) -> tuple[float, float]:
    """``(purity, stray_share)`` of a fitted region: ``x``, ``y`` the profile, ``bin_peaks`` each
    bin's fitted peak there (one column per bin), ``stray`` the strays' curve, ``where`` the bins'
    fitted m/z (NaN: not fitted), ``fwhm_mz`` the run's peak width at an m/z.

    ``purity``: within ``FOOTPRINT_WIDTHS`` peak widths of each fitted bin, the sum of the
    smaller of the bins' curve and the profile over the profile's sum, averaged over the bins
    whose peak reaches ``PURITY_MIN`` of the tallest bin's, weighted by the bins' curve there;
    NaN without such a bin. ``stray_share``: the same sum for the strays over the whole region
    (0 without strays). Both 0-1; the smaller of curve and profile, so a model that overshoots
    does not score itself up."""
    total = float(y.sum())
    if not total > 0:
        return float("nan"), float("nan")
    curve = bin_peaks.sum(axis=1)
    tops = bin_peaks.max(axis=0) if bin_peaks.size else np.zeros(0)
    tallest = float(tops.max()) if len(tops) else 0.0
    num = den = 0.0
    for k in np.flatnonzero(np.isfinite(where)):
        if not tallest > 0 or tops[k] < PURITY_MIN * tallest:
            continue
        inside = np.abs(x - where[k]) <= FOOTPRINT_WIDTHS * float(fwhm_mz(where[k]))
        held = float(y[inside].sum())
        if not inside.any() or not held > 0:
            continue
        weight = float(curve[inside].sum())
        num += weight * float(np.minimum(curve[inside], y[inside]).sum()) / held
        den += weight
    purity = num / den if den > 0 else float("nan")
    return purity, float(np.minimum(stray, y).sum() / total)


# --- the run -------------------------------------------------------------------------------

def own_offset(sticks: np.ndarray, where: np.ndarray, pos: np.ndarray) -> float:
    """A spectrum's own offset (ppm) from its pick: the intensity-weighted median of its bright
    bins' apexes (``BRIGHT`` of the tallest) about the positions ``pos`` its envelope supports;
    NaN without one."""
    sticks = np.asarray(sticks, dtype=np.float64)
    top = sticks.max() if len(sticks) else 0.0
    bright = (sticks >= BRIGHT * top) & np.isfinite(where) & (sticks > 0)
    if not bright.any():
        return float("nan")
    r = 1e6 * (where[bright] - pos[bright]) / pos[bright]
    w = sticks[bright]
    order = np.argsort(r)
    cum = np.cumsum(w[order])
    return float(r[order][np.searchsorted(cum, 0.5 * cum[-1])])


@dataclass(frozen=True)
class OffsetSearch:
    """Where each spectrum's offset is searched: ``centre`` (ppm, its injection's median),
    the run's ``scatter`` (ppm, robust sd of the spectra's own offsets about their injection's
    median; NaN: too few), the search's ``limit`` and ``step`` (ppm)."""

    centre: np.ndarray
    scatter: float
    limit: float
    step: float


def search_offsets(own: np.ndarray, injections: Sequence) -> OffsetSearch:
    """The :class:`OffsetSearch` from the spectra's ``own`` offsets (NaN: none) and their
    ``injections`` (any hashable per spectrum)."""
    own = np.asarray(own, dtype=np.float64)
    groups: dict = defaultdict(list)
    for j, key in enumerate(injections):
        if np.isfinite(own[j]):
            groups[key].append(own[j])
    medians = {k: float(np.median(v)) for k, v in groups.items()}
    finite = own[np.isfinite(own)]
    run = float(np.median(finite)) if len(finite) else 0.0
    centre = np.array([medians.get(k, run) for k in injections], dtype=np.float64)
    dist = own - centre
    dist = dist[np.isfinite(dist)]
    scatter = _MAD_SD * float(np.median(np.abs(dist))) if len(dist) >= 5 else float("nan")
    limit = max(SEARCH_SD * scatter, SEARCH_FLOOR_PPM) if np.isfinite(scatter) else SEARCH_FLOOR_PPM
    return OffsetSearch(centre, scatter, limit, max(limit / SEARCH_STEPS, MIN_STEP_PPM))


def _fit_chunk(args) -> list[tuple[int, ProfileFit | None]]:
    """Fit a chunk of spectra (a process pool's task)."""
    jobs, fwhm, shape, limit, step, coarse = args
    from .measure import PeakMeasure

    law = PeakMeasure("area", "auto", fwhm[0], fwhm[1])
    return [(i, fit_spectrum(mz, y, sticks, plan, moments, law.fwhm_mz, shape, centre=centre,
                             limit=limit, step=step, coarse=coarse))
            for i, mz, y, sticks, plan, moments, centre in jobs]


@contextmanager
def _single_threaded_children():
    """While the pool starts its processes, they inherit one thread of linear algebra each
    (the libraries read these variables when they load, so the processes are spawned, not
    forked from this one, whose libraries are loaded already)."""
    before = {k: os.environ.get(k) for k in _ONE_THREAD}
    os.environ.update({k: "1" for k in _ONE_THREAD})
    try:
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def fit_all(jobs: list, measure, shape: PeakShape, search: OffsetSearch, *,
            coarse: bool, workers: int | None = None, progress=None, cancel=None,
            ) -> dict[int, ProfileFit | None]:
    """:func:`fit_spectrum` for every job ``(i, mz, y, sticks, plan, moments, centre)``, in a
    process pool when there are ``PARALLEL_FROM`` or more (``workers``: None for up to
    ``MAX_WORKERS``, the CPUs this process may use; 1 for this process)."""
    from ..hooks import check_cancel, report

    fwhm = (measure.fwhm_ppm, measure.fwhm_exponent)
    if workers is None:
        try:
            cpus = len(os.sched_getaffinity(0))
        except AttributeError:  # macOS, Windows
            cpus = os.cpu_count() or 1
        workers = min(MAX_WORKERS, cpus)
    out: dict[int, ProfileFit | None] = {}
    chunks = [jobs[k:k + CHUNK] for k in range(0, len(jobs), CHUNK)]
    task = lambda c: (c, fwhm, shape, search.limit, search.step, coarse)  # noqa: E731
    done = 0
    if workers <= 1 or len(jobs) < PARALLEL_FROM:
        for c in chunks:
            check_cancel(cancel)
            out.update(_fit_chunk(task(c)))
            done += len(c)
            report(progress, done, len(jobs))
        return out
    from ..io._parallel import _main_left_out

    with _single_threaded_children(), ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        with _main_left_out():  # the processes start here, without the main script: a script
            # that picks at its top (no main guard) would pick again in each of them
            futures = [pool.submit(_fit_chunk, task(c)) for c in chunks]
        try:
            for f in futures:
                check_cancel(cancel)
                part = f.result()
                out.update(part)
                done += len(part)
                report(progress, done, len(jobs))
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    return out


def model_spectrum(mz: np.ndarray, y: np.ndarray, peptide: str, charge: int, *,
                   offset_ppm: float, measure, config, modification: str = "",
                   registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
                   resolving_power: float | None = None,
                   resolving_power_mz: float | None = None) -> ProfileFit | None:
    """One spectrum's fit again, with its model curves, at the m/z offset its run took
    (``offset_ppm``; the stored ``fit_offset_ppm``), for drawing (:mod:`hxex.report.picking_check`):
    the pick seeds it with areas, as :func:`~hxex.picking.pick_spectra` does. ``measure`` is the
    run's :class:`~hxex.picking.PeakMeasure` (with its ``shape``), ``config`` its picker
    settings. None when there is nothing to fit."""
    import dataclasses

    from .pick import clean_profile, pick_spectrum, plan_bins

    if measure is None or measure.shape is None:
        raise ValueError("the run was not picked with the profile fit (no peak shape)")
    plan = plan_bins(peptide, charge, modification=modification, config=config, registry=registry)
    seed = pick_spectrum(mz, y, peptide, charge, modification=modification, config=config,
                         registry=registry, plan=plan, resolving_power=resolving_power,
                         resolving_power_mz=resolving_power_mz,
                         measure=dataclasses.replace(measure, measure="area"))
    if seed.empty or not len(seed.intensity):
        return None
    moments = ion_moments(peptide, modification=modification, nterm_subtract=config.nterm_subtract,
                          rules=config.amide_rules, registry=registry)
    mz, y = clean_profile(mz, y)
    return fit_spectrum(mz, y, seed.intensity, plan, moments, measure.fwhm_mz, measure.shape,
                        centre=float(offset_ppm), limit=0.0, step=1.0, coarse=False, model=True)
