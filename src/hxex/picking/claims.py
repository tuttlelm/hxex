# SPDX-License-Identifier: GPL-3.0-or-later
"""Whose isotope cluster is it? Claims of each undeuterated control's cluster by the listed
ions (docs/PLAN.md Next steps 34; docs/HISTORY.md items 438 and the build after it).

The m/z check (:mod:`hxex.picking.mz_check`) asks how far a series' undeuterated envelope sits
from its own m/z; this asks onto what. Every ion read in a sample (every peptide, charge and
modification of its peptide table, fragment ions apart) is a claimant of every undeuterated
cluster of that sample whose grid it reaches; a series whose cluster another listed ion claims
and it does not is reading that ion (the tutorial's 13-24 2+ is 55-67 2+'s spectrum in every
sample, 20.6 ppm off its own grid and on the other's).

- *Peaks:* the local maxima of the stored profile (the vertex of the parabola through a top
  and its neighbours) at or above ``PEAK_SHARE`` of its tallest; every point of a centroided
  spectrum.
- *The cluster:* the isotope series at the ion's charge (steps of the isotope spacing over the
  charge, each within the larger of the picking half-width and ``SERIES_PPM``, at most
  ``SERIES_PPM``) from the tallest peak inside the ion's picking windows of its first
  ``OWN_BINS`` natural bins; at least two peaks, else ``no_cluster``.
- *Offset to an ion's grid:* the intensity-weighted median of the cluster's peaks at or above
  ``STRONG_SHARE`` of its tallest, each to the ion's nearest predicted bin (its natural
  isotope clusters, then the deuterium ladder), in ppm, after the run's calibration (the m/z
  check's line in its injection plus its term for the charge: one calibration for both).
- *Shape:* a scale-only fit of the cluster's peak heights by the ion's natural pattern at its
  bins (peaks before its monoisotopic bin and predicted peaks missing both count); a leftover
  above ``GROSS_LEFTOVER`` (a wrong monoisotopic peak) rejects the claimant. Shape decides
  nothing between compositions of the same nominal mass (measured), so it is a gross check
  only.
- *Scatter:* the robust sd of the own clusters' offsets in classes of the cluster's height
  (dim clusters scatter more), interpolated in log height; a claim lies within ``CLAIM_SD``
  times the scatter at the cluster's height (measured: the lost and isomer verdicts do not
  move between 3 and 5).
- *Retention:* another ion's claim needs its window (and drift range) in the same injection to
  overlap the series' own, where both were read with windows; exported spectra have none, so
  there the claim stands on position alone (``claim_by_position``).
- *Verdict of one control:* ``own`` (only the ion claims), ``isomer`` (the ion and listed ions
  of the same composition only: MS1 cannot tell them apart), ``shared`` (the ion and another
  composition), ``lost`` (another ion claims, the ion does not), ``wrong_shape`` (on the ion's
  grid, shape rejected, nobody else claims), ``unclaimed`` (nobody listed claims) or
  ``no_cluster``.
- *Verdict of a series:* its undeuterated controls' most common verdict (ties: the worse, in
  the order of :data:`RANK`), with the other ion most often named there and the median
  offsets; its labelled spectra share it (deuterated peaks move).

The quality flags follow (:mod:`hxex.quality`): ``lost`` is ``other_ion`` (left out of the
analysis by default, ``[analysis.filters] other_ion``), ``isomer`` ``isomer_overlap`` and
``shared`` ``shared_cluster`` (flags only). Not assessed: fragment ions, series without an
undeuterated control, and every series when the run's own clusters are too few for a scatter.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from .. import chem
from .config import ISOTOPE_STEP, PickerConfig
from .mz_check import SeriesOffsets, injection
from .screen import STICK_POINTS

PEAK_SHARE = 0.02  # a peak is at least this share of the profile's tallest
SERIES_PPM = 25.0  # the most a cluster's next isotope peak may sit from its expected place
OWN_BINS = 5  # the ion's own cluster starts in its windows of this many first natural bins
STRONG_SHARE = 0.2  # the peaks that place a cluster: at least this share of its tallest
GROSS_LEFTOVER = 0.5  # a shape leftover above this rejects a claimant
CLAIM_SD = 4.0  # a claim lies within this many times the run's scatter at the cluster's height
SCATTER_CLASSES = 10  # the scatter is measured in up to this many classes of height
SCATTER_CLASS_SIZE = 20  # of at least this many own clusters
MIN_CLUSTERS = 5  # fewer own clusters on their grid: no scatter, no claims
REACH_BELOW = 6  # a claimant's grid may start this many isotope steps below the cluster

VERDICTS = ("own", "isomer", "shared", "lost", "wrong_shape", "unclaimed", "no_cluster")
# a series' tie goes to the worse verdict
RANK = {"lost": 6, "unclaimed": 5, "shared": 4, "wrong_shape": 3, "isomer": 2, "own": 1,
        "no_cluster": 0}

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Claims:
    """The claims' verdict of every spectrum's series (one entry per spectrum of the set
    claimed; the series' values on every spectrum of it): ``claim`` (one of :data:`VERDICTS`,
    empty where not assessed), ``claim_other`` (the other ion named: ``START-END SEQUENCE z+``,
    with its modification), ``claim_offset_ppm`` (the series' own grid's offset from its
    cluster after the calibration, ppm), ``claim_other_offset_ppm`` (the other ion's) and
    ``claim_by_position`` (no retention window compared the two: the claim stands on position
    alone)."""

    claim: np.ndarray
    claim_other: np.ndarray
    claim_offset_ppm: np.ndarray
    claim_other_offset_ppm: np.ndarray
    claim_by_position: np.ndarray

    @classmethod
    def none(cls, n: int) -> Claims:
        return cls(np.full(n, "", dtype=object), np.full(n, "", dtype=object),
                   np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=bool))


def ion_label(peptide_range: str, peptide: str, charge: int, modification: str = "") -> str:
    """How a claim names an ion: ``0055-0067 LRAPSWFDTGLSE 2+`` (and its modification)."""
    text = f"{peptide_range} {peptide} {int(charge)}+"
    return f"{text} {modification}" if modification else text


def peaks(mz: np.ndarray, y: np.ndarray, charge: int) -> tuple[np.ndarray, np.ndarray]:
    """The peaks of a profile (or of a centroided spectrum, every stick) at or above
    :data:`PEAK_SHARE` of its tallest: their m/z and heights."""
    n = len(y)
    if n == 0 or not y.max() > 0:
        return np.zeros(0), np.zeros(0)
    d = np.diff(mz)
    spacing = float(np.median(d[d > 0])) if (d > 0).any() else np.inf
    floor = PEAK_SHARE * y.max()
    if (ISOTOPE_STEP / max(int(charge), 1)) / spacing < STICK_POINTS:  # centroided
        keep = y >= floor
        return mz[keep].astype(float), y[keep].astype(float)
    if n < 3:
        return np.zeros(0), np.zeros(0)
    loc = np.flatnonzero((y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > 0)) + 1
    loc = loc[y[loc] >= floor]
    at = mz[loc].astype(float)
    for j, k in enumerate(loc):
        if y[k - 1] > 0 and y[k + 1] > 0:
            x0, x1, x2 = mz[k - 1:k + 2]
            y0, y1, y2 = y[k - 1:k + 2]
            den = (x0 - x1) * (x0 - x2) * (x1 - x2)
            if den != 0:
                a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
                b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
                if a < 0:
                    at[j] = min(max(-b / (2 * a), x0), x2)
    return at, y[loc].astype(float)


def cluster(at: np.ndarray, height: np.ndarray, seed: int, charge: int,
            tol_ppm: float) -> np.ndarray:
    """The peaks (indices, by m/z) of the isotope series at ``charge`` through ``seed``: from
    it each way, the tallest peak within ``tol_ppm`` of the next isotope position."""
    members = [seed]
    for direction in (1, -1):
        cur = seed
        while True:
            target = at[cur] + direction * ISOTOPE_STEP / charge
            near = np.flatnonzero(np.abs(at - target) <= tol_ppm * 1e-6 * target)
            if not len(near):
                break
            nxt = int(near[np.argmax(height[near])])
            if nxt in members:
                break
            members.append(nxt)
            cur = nxt
    return np.array(sorted(members, key=lambda m: at[m]), dtype=np.int64)


@dataclass(frozen=True)
class _Grid:
    mz: np.ndarray  # predicted bins: the natural clusters, then the deuterium ladder
    natural: np.ndarray  # the natural pattern, tallest 1
    composition: tuple


def _grid(peptide: str, charge: int, modification: str, config: PickerConfig,
          registry) -> _Grid | None:
    from .pick import plan_bins

    try:
        plan = plan_bins(peptide, charge, modification=modification, config=config,
                         registry=registry)
        spec = chem.parse_modification(modification)
        comp = tuple(sorted(chem.composition(peptide, spec.ion_type, mod=spec,
                                             registry=registry).items()))
    except Exception as err:  # noqa: BLE001  (an ion the chemistry cannot place: no claim)
        log.debug("claims: no grid for %s %d+ %s: %s", peptide, charge, modification, err)
        return None
    shape, offs = (np.asarray(a, dtype=np.float64) for a in plan.clusters)
    pred = np.asarray(plan.pred_mz, dtype=np.float64)
    out = pred.copy()
    n = min(len(offs), len(pred))
    out[:n] = plan.undeut_mz + offs[:n] / charge  # the composition's own cluster masses
    if len(pred) > n:
        out[n:] = out[n - 1] + (pred[n:] - pred[n - 1])
    return _Grid(out, shape / shape.max(), comp)


def score(at: np.ndarray, height: np.ndarray, grid: _Grid, cal_ppm: float
          ) -> tuple[float, float]:
    """``(offset ppm, shape leftover)`` of a cluster (its peaks ``at``, ``height``) against an
    ion's ``grid`` moved by the run's calibration ``cal_ppm``."""
    pm = grid.mz * (1 + cal_ppm * 1e-6)
    k = np.clip(np.searchsorted(pm, at), 1, len(pm) - 1)
    k = np.where(np.abs(at - pm[k - 1]) < np.abs(at - pm[k]), k - 1, k)
    step = pm[1] - pm[0]
    below = at < pm[0] - 0.5 * step
    d = 1e6 * (at - pm[k]) / pm[k]
    strong = height >= STRONG_SHARE * height.max()
    order = np.argsort(d[strong])
    w = np.cumsum(height[strong][order])
    offset = float(d[strong][order][np.searchsorted(w, 0.5 * w[-1])])
    # the cluster's heights by bin (before the monoisotopic bin: negative) against the pattern
    kk = np.where(below, -np.round((pm[0] - at) / step).astype(np.int64), k)
    nat = grid.natural
    lo, hi = min(int(kk.min()), 0), max(int(kk.max()), len(nat) - 1)
    h = np.zeros(hi - lo + 1)
    np.add.at(h, kk - lo, height)
    q = np.zeros(hi - lo + 1)
    q[-lo:-lo + len(nat)] = nat
    a = float(h @ q / (q @ q)) if q @ q > 0 else 0.0
    return offset, float(np.abs(h - a * q).sum() / h.sum())


def _scatter(top: np.ndarray, offset: np.ndarray):
    """The robust sd of ``offset`` by height ``top`` (classes of log height): a function of
    the height; None when the offsets are too few."""
    ok = np.isfinite(offset) & (top > 0)
    top, offset = top[ok], offset[ok]
    if len(offset) < MIN_CLUSTERS:
        return None
    classes = max(1, min(SCATTER_CLASSES, len(offset) // SCATTER_CLASS_SIZE))
    order = np.argsort(top)
    centres, spreads = [], []
    for part in np.array_split(order, classes):
        v = offset[part]
        centres.append(float(np.median(np.log(top[part]))))
        spreads.append(1.4826 * float(np.median(np.abs(v - np.median(v)))))
    centres_a, spreads_a = np.array(centres), np.array(spreads)
    return lambda h: float(np.interp(np.log(max(float(h), 1e-300)), centres_a, spreads_a))


def _overlap(a: tuple, b: tuple) -> bool | None:
    """Whether two ions' retention windows (and drift ranges) overlap; None: not known."""
    if not (np.isfinite(a[0]) and np.isfinite(a[1]) and np.isfinite(b[0])
            and np.isfinite(b[1])):
        return None
    if min(a[1], b[1]) - max(a[0], b[0]) <= 0:
        return False
    if all(np.isfinite(v) for v in (a[2], a[3], b[2], b[3])):
        return min(a[3], b[3]) - max(a[2], b[2]) > 0
    return True


@dataclass
class _Control:
    row: int
    key: tuple
    run: tuple
    verdict: str = "no_cluster"
    offset: float = float("nan")
    other: tuple | None = None
    other_offset: float = float("nan")
    by_position: bool = False


def claims(spectra, config: PickerConfig, checked: SeriesOffsets, *,
           registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
           skip: Collection[int] = ()) -> Claims:
    """The claims of every series of ``spectra`` (a :class:`~hxex.spectra.SpectrumSet`) with
    undeuterated controls, except those in ``skip`` (fragment ions), with the m/z check's
    calibration ``checked`` (see the module docstring)."""
    from .pick import clean_profile

    n = len(spectra)
    out = Claims.none(n)
    if n == 0:
        return out
    skip = {int(d) for d in skip}
    role = np.asarray(spectra.role).astype(str)
    data_id = np.asarray(spectra.data_id)
    sample = np.asarray(spectra.sample).astype(str)
    mods = np.asarray(spectra.modification).astype(str)
    keys = [(str(spectra.peptide[i]), int(spectra.charge[i]), mods[i]) for i in range(n)]
    ranges = {(sample[i], keys[i]): str(spectra.peptide_range[i]) for i in range(n)
              if int(data_id[i]) not in skip}
    grids: dict[tuple, _Grid | None] = {}

    def grid_of(key):
        if key not in grids:
            grids[key] = _grid(*key, config, registry)
        return grids[key]

    # every listed ion of a sample by charge, sorted by the first bin's m/z (the claimants)
    listed: dict[tuple[str, int], list[tuple[float, tuple]]] = defaultdict(list)
    for s, key in ranges:
        g = grid_of(key)
        if g is not None:
            listed[(s, key[1])].append((float(g.mz[0]), key))
    for v in listed.values():
        v.sort()
    first_mz = {k: np.array([m for m, _ in v]) for k, v in listed.items()}
    rows = [i for i in np.flatnonzero(role == "undeut") if int(data_id[i]) not in skip]
    runs = [injection(spectra, i) for i in rows]
    # each ion's windows in each injection (its undeuterated spectrum there)
    windows = {}
    for i, run in zip(rows, runs, strict=True):
        windows[(run, keys[i])] = tuple(float(np.asarray(getattr(spectra, c))[i]) for c in
                                        ("rt_start", "rt_end", "drift_start", "drift_end"))
    controls: list[_Control] = []
    found = []  # (cluster peaks, heights, calibration) per control with a cluster
    for i, run in zip(rows, runs, strict=True):
        c = _Control(row=int(i), key=keys[i], run=run)
        controls.append(c)
        g = grid_of(keys[i])
        if g is None:
            found.append(None)
            continue
        z = keys[i][1]
        mz, y = clean_profile(*spectra.spectrum(int(i)))
        at, height = peaks(mz, y, z)
        pred = g.mz[:OWN_BINS]
        half = np.atleast_1d(config.half_width(
            pred, z, resolving_power=float(spectra.resolving_power[i]),
            resolving_power_mz=float(spectra.resolving_power_mz[i])))
        inside = (np.any(np.abs(at[:, None] - pred[None, :]) <= half[None, :], axis=1)
                  if len(at) else np.zeros(0, dtype=bool))
        cal = checked.calibration_ppm(run, float(g.mz[0]), z)
        if not inside.any() or not np.isfinite(cal):
            found.append(None)
            continue
        seed = int(np.flatnonzero(inside)[np.argmax(height[inside])])
        tol = min(SERIES_PPM, 1e6 * float(half[0]) / float(pred[0]))
        members = cluster(at, height, seed, z, tol)
        if len(members) < 2:
            found.append(None)
            continue
        c.offset, left = score(at[members], height[members], g, cal)
        found.append((at[members], height[members], cal, left))
    # the run's scatter of the own clusters' offsets by height (each measurement once)
    from .binned import raw_keys

    raw = raw_keys(spectra, rows)
    seen: set[bytes] = set()
    tops, offs = [], []
    for c, f in zip(controls, found, strict=True):
        if f is None:
            continue
        k = raw[c.row]
        if k is not None and k in seen:
            continue
        if k is not None:
            seen.add(k)
        tops.append(float(f[1].max()))
        offs.append(c.offset)
    sd_at = _scatter(np.array(tops), np.array(offs))
    if sd_at is None:
        log.info("claims: %d undeuterated clusters, fewer than the %d a scatter needs: no "
                 "series assessed", len(offs), MIN_CLUSTERS)
        return out
    for c, f in zip(controls, found, strict=True):
        if f is None:
            continue
        at, height, cal, self_left = f
        z = c.key[1]
        limit = CLAIM_SD * sd_at(float(height.max()))
        self_ok = abs(c.offset) <= limit and self_left <= GROSS_LEFTOVER
        own_window = windows.get((c.run, c.key), (np.nan,) * 4)
        others = []
        s = str(sample[c.row])
        pool = listed.get((s, z), [])
        if pool:
            lo = float(at.min()) - REACH_BELOW * ISOTOPE_STEP / z
            first = first_mz[(s, z)]
            for j in range(int(np.searchsorted(first, lo)),
                           int(np.searchsorted(first, float(at.max()) + 0.1, side="right"))):
                key = pool[j][1]
                if key == c.key:
                    continue
                o, left = score(at, height, grids[key], cal)
                if abs(o) > limit or left > GROSS_LEFTOVER:
                    continue
                seen_together = _overlap(own_window, windows.get((c.run, key), (np.nan,) * 4))
                if seen_together is False:
                    continue
                others.append((abs(o), o, key, seen_together is None))
        others.sort(key=lambda t: t[0])
        if self_ok:
            if others and all(grids[k].composition == grids[c.key].composition
                              for _, _, k, _ in others):
                c.verdict = "isomer"
            else:
                c.verdict = "shared" if others else "own"
        elif others:
            c.verdict = "lost"
        elif abs(c.offset) <= limit:
            c.verdict = "wrong_shape"
        else:
            c.verdict = "unclaimed"
        if others:
            _, c.other_offset, c.other, c.by_position = others[0]
    # each series from its controls
    by_series: dict[int, list[_Control]] = defaultdict(list)
    for c in controls:
        by_series[int(data_id[c.row])].append(c)
    claim = np.full(n, "", dtype=object)
    other = np.full(n, "", dtype=object)
    offset = np.full(n, np.nan)
    other_offset = np.full(n, np.nan)
    by_position = np.zeros(n, dtype=bool)
    for d, cs in by_series.items():
        count = Counter(c.verdict for c in cs)
        top = max(count.values())
        verdict = max((v for v in count if count[v] == top), key=RANK.get)
        these = [c for c in cs if c.verdict == verdict]
        at = data_id == d
        claim[at] = verdict
        mine = [c.offset for c in these if np.isfinite(c.offset)]
        if mine:
            offset[at] = float(np.median(mine))
        named = Counter(c.other for c in these if c.other is not None)
        if named:
            key = named.most_common(1)[0][0]
            s = str(sample[these[0].row])
            other[at] = ion_label(ranges[(s, key)], *key)
            other_offset[at] = float(np.median([c.other_offset for c in these
                                                if c.other == key]))
            by_position[at] = all(c.by_position for c in these if c.other == key)
    n_lost = len({int(d) for d in data_id[claim == "lost"]})
    if n_lost:
        log.warning("%d series read another listed ion's peaks in their undeuterated controls "
                    "(flagged other_ion): left out of the analysis ([analysis.filters] "
                    "other_ion); check their retention times", n_lost)
    return Claims(claim, other, offset, other_offset, by_position)
