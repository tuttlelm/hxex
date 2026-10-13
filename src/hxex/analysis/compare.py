# SPDX-License-Identifier: GPL-3.0-or-later
"""Comparisons within a run (docs/ANALYSIS.md, phase 3): every sample against a reference sample,
per peptide and per residue, and averages over user-defined regions.

Differences are in fractional deuteration (``fD``) at a fixed K and rank (K=1, the average, by
default), or in the population of a K >= 2 rank (``quantity='population'``): whether a sample
shifts the balance between the populations rather than their deuteration. Per residue,
:func:`compare_populations` compares the lever-rule population of the choice between one and two
populations. Per peptide and time a difference is significant by the hybrid rule (Hageman and Weis,
Anal. Chem. 2019): Welch's t-test on the replicate spectra of the two samples gives ``p < alpha``,
and the difference exceeds a global threshold ``delta_crit = t(1 - alpha/2, 2n - 2) s_pooled
sqrt(2/n)``, from the pooled standard deviation of replicate spectra over every peptide, time and
sample (``n`` the typical number of replicate spectra). Per residue, where the replicates are
already combined, a difference is significant when ``|delta| / se`` exceeds the normal quantile of
the same ``alpha`` and ``|delta|`` exceeds the peptides' ``delta_crit``.

Regions and sites are protein-specific and user-defined (:class:`Annotations`), never derived.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields

import numpy as np
from scipy import stats

from .residues import ResidueMap
from .species import SpeciesFilters, _combine, _records


@dataclass(frozen=True)
class Region:
    """A named stretch of the protein, residues ``start`` to ``end`` inclusive."""

    name: str
    start: int
    end: int

    def __post_init__(self):
        if self.end < self.start or self.start < 1:
            raise ValueError(f"region {self.name!r}: need 1 <= start <= end")


@dataclass(frozen=True)
class Site:
    """A single residue to mark (a modification site, a mutation)."""

    residue: int
    label: str = ""


@dataclass(frozen=True)
class Annotations:
    """The regions and sites of a protein (regions may overlap)."""

    regions: tuple[Region, ...] = ()
    sites: tuple[Site, ...] = field(default_factory=tuple)

    def marks(self) -> list[float]:
        """Residues for dashed lines: region boundaries (between residues) and sites."""
        edges = {r.start - 0.5 for r in self.regions} | {r.end + 0.5 for r in self.regions}
        return sorted(edges | {float(s.residue) for s in self.sites})

    def to_dict(self) -> dict:
        return {"regions": [{"name": r.name, "start": r.start, "end": r.end} for r in self.regions],
                "sites": [{"residue": s.residue, "label": s.label} for s in self.sites]}

    @classmethod
    def from_dict(cls, data: Mapping) -> Annotations:
        unknown = set(data) - {"regions", "sites"}
        if unknown:
            raise ValueError(f"unknown annotation field(s): {', '.join(sorted(unknown))}")
        return cls(regions=tuple(Region(**r) for r in data.get("regions", ())),
                   sites=tuple(Site(**s) for s in data.get("sites", ())))


def _columns(cls, rows: list[dict]) -> dict:
    names = [f.name for f in fields(cls) if f.name not in _SCALAR_FIELDS]
    return {name: np.array([r[name] for r in rows]) for name in names}


_SCALAR_FIELDS = ("alpha", "delta_crit", "reference", "K", "rank", "quantity")
QUANTITIES = ("fD", "population")


@dataclass(frozen=True)
class PeptideComparison:
    """One row per sample (not the reference), peptide and time: ``peptide``, ``modification``,
    ``start``, ``end``, ``time``, ``value_reference`` and ``value`` (the ``quantity``, ``fD`` or
    ``population``, combined over replicates and charge states) with their ``*_se``, ``delta``
    (``value - value_reference``) and ``delta_se``, ``t`` and ``p`` (Welch's test on the
    replicate spectra; NaN with fewer than two on a side), ``n_reference`` and ``n`` (spectra) and
    ``significant``; with the ``reference`` sample, ``K``, ``rank``, ``alpha`` and the global
    ``delta_crit``."""

    sample: np.ndarray
    peptide: np.ndarray
    modification: np.ndarray
    start: np.ndarray
    end: np.ndarray
    time: np.ndarray
    value_reference: np.ndarray
    value_reference_se: np.ndarray
    value: np.ndarray
    value_se: np.ndarray
    delta: np.ndarray
    delta_se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    n_reference: np.ndarray
    n: np.ndarray
    significant: np.ndarray
    reference: str = ""
    K: int = 1
    rank: int = 1
    alpha: float = 0.05
    delta_crit: float = 0.0
    quantity: str = "fD"

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name not in _SCALAR_FIELDS}

    def where(self, *, sample=None, time=None) -> np.ndarray:
        keep = np.ones(len(self), dtype=bool)
        for name, value in (("sample", sample), ("time", time)):
            if value is not None:
                values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
                keep &= np.isin(getattr(self, name), values)
        return keep


def compare_peptides(
    results,
    reference: str,
    *,
    samples: Sequence[str] | None = None,
    K: int = 1,
    rank: int = 1,
    quantity: str = "fD",
    alpha: float = 0.05,
    exclude_flagged: bool = False,
    filters: SpeciesFilters | None = SpeciesFilters(),  # noqa: B008
) -> PeptideComparison:
    """Every sample (or those in ``samples``) against ``reference``, per peptide and time, at
    ``K`` and ``rank``, in ``quantity`` (``'fD'``, or ``'population'`` for K >= 2; see the module
    docstring for the hybrid rule). The values are those of
    :func:`hxex.analysis.species_uptake` with the same filters. ``results`` may be a
    :class:`hxex.analysis.RunSet` (samples ``run:sample``; ``delta_crit`` is then pooled over
    every run)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {', '.join(QUANTITIES)}")
    if quantity == "population" and K < 2:
        raise ValueError("populations are compared at K >= 2")
    at = 0 if quantity == "fD" else 2  # the record's value (its standard error follows it)
    ks = (1, K) if K != 1 else (1,)
    # quiet: species_uptake reports what the filters left out
    groups, *_ = _records(results, ks, exclude_flagged, filters, quiet=True)
    by_sample: dict[str, dict[tuple, np.ndarray]] = {}
    ses: dict[tuple, np.ndarray] = {}
    for key, recs in groups.items():
        sample, peptide, mod, start, end, time, k, r = key
        if (k, r) != (K, rank):
            continue
        by_sample.setdefault(sample, {})[(peptide, mod, start, end, time)] = np.array(
            [x[at] for x in recs])
        ses[(sample, peptide, mod, start, end, time)] = np.array([x[at + 1] for x in recs])
    if reference not in by_sample:
        raise ValueError(f"no reference sample {reference!r} at K={K} rank {rank} (samples: "
                         f"{', '.join(sorted(by_sample))})")
    # the pooled standard deviation of replicate spectra and the typical replicate count
    num = den = 0.0
    counts = []
    for values in (v for per in by_sample.values() for v in per.values()):
        if len(values) >= 2:
            num += (len(values) - 1) * float(np.var(values, ddof=1))
            den += len(values) - 1
            counts.append(len(values))
    if den > 0:
        n_typ = float(np.median(counts))
        s_pooled = np.sqrt(num / den)
        df = max(2 * n_typ - 2, 1.0)
        delta_crit = float(stats.t.ppf(1 - alpha / 2, df) * s_pooled * np.sqrt(2 / n_typ))
    else:
        delta_crit = np.nan
    others = [s for s in (samples if samples is not None else sorted(by_sample))
              if s != reference]
    ref = by_sample[reference]
    rows = []
    for sample in others:
        for key, values in sorted(by_sample.get(sample, {}).items()):
            if key not in ref:
                continue
            a, b = ref[key], values
            ma, sa = _combine(a, ses[(reference, *key)])
            mb, sb = _combine(b, ses[(sample, *key)])
            if len(a) >= 2 and len(b) >= 2 and (np.var(a) > 0 or np.var(b) > 0):
                t, p = stats.ttest_ind(b, a, equal_var=False)
            else:
                t, p = np.nan, np.nan
            delta = mb - ma
            sig = bool(np.isfinite(p) and p < alpha and np.isfinite(delta_crit)
                       and abs(delta) > delta_crit)
            rows.append({"sample": sample, "peptide": key[0], "modification": key[1],
                         "start": key[2], "end": key[3], "time": key[4],
                         "value_reference": ma, "value_reference_se": sa, "value": mb,
                         "value_se": sb, "delta": delta, "delta_se": float(np.hypot(sa, sb)),
                         "t": float(t), "p": float(p), "n_reference": len(a), "n": len(b),
                         "significant": sig})
    cols = _columns(PeptideComparison, rows) if rows else {
        f.name: np.array([]) for f in fields(PeptideComparison) if f.name not in _SCALAR_FIELDS}
    return PeptideComparison(**cols, reference=reference, K=K, rank=rank, alpha=alpha,
                             delta_crit=delta_crit, quantity=quantity)


@dataclass(frozen=True)
class ResidueComparison:
    """One row per sample (not the reference), time and residue: ``value_reference`` and
    ``value`` (the ``quantity``) with their ``*_se``, ``delta`` and ``delta_se``, ``z``
    (``delta / delta_se``) and ``significant``; with the ``reference``, ``K``, ``rank``, ``alpha``
    and the ``delta_crit`` used."""

    sample: np.ndarray
    time: np.ndarray
    residue: np.ndarray
    value_reference: np.ndarray
    value_reference_se: np.ndarray
    value: np.ndarray
    value_se: np.ndarray
    delta: np.ndarray
    delta_se: np.ndarray
    z: np.ndarray
    significant: np.ndarray
    reference: str = ""
    K: int = 1
    rank: int = 1
    alpha: float = 0.05
    delta_crit: float = 0.0
    quantity: str = "fD"

    def __len__(self) -> int:
        return len(self.sample)

    def where(self, *, sample=None, time=None) -> np.ndarray:
        keep = np.ones(len(self), dtype=bool)
        for name, value in (("sample", sample), ("time", time)):
            if value is not None:
                values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
                keep &= np.isin(getattr(self, name), values)
        return keep


def _residue_rows(value: dict, reference: str, alpha: float, delta_crit: float) -> list[dict]:
    """Rows of a residue comparison from ``(sample, time, residue) -> (value, se)``."""
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    rows = []
    for (sample, time, res), (v, e) in sorted(value.items()):
        if sample == reference or (reference, time, res) not in value:
            continue
        rv, re_ = value[(reference, time, res)]
        if not (np.isfinite(v) and np.isfinite(rv)):
            continue
        delta = v - rv
        se = float(np.hypot(e, re_))
        z = delta / se if se > 0 else np.nan
        sig = bool(np.isfinite(z) and abs(z) > z_crit and abs(delta) > delta_crit)
        rows.append({"sample": sample, "time": time, "residue": res, "value_reference": rv,
                     "value_reference_se": re_, "value": v, "value_se": e, "delta": delta,
                     "delta_se": se, "z": z, "significant": sig})
    return rows


def _residue_comparison(rows, **scalars) -> ResidueComparison:
    names = [f.name for f in fields(ResidueComparison) if f.name not in _SCALAR_FIELDS]
    cols = {name: np.array([r[name] for r in rows]) for name in names}
    return ResidueComparison(**cols, **scalars)


def compare_residues(
    residues: ResidueMap,
    reference: str,
    *,
    K: int = 1,
    rank: int = 1,
    alpha: float = 0.05,
    delta_crit: float = 0.0,
) -> ResidueComparison:
    """Every sample against ``reference`` residue by residue at ``K`` and ``rank``;
    ``delta_crit`` is usually the peptides' (:func:`compare_peptides`)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    sel = (residues.K == K) & (residues.rank == rank)
    value = {(str(s), float(t), int(r)): (float(v), float(e)) for s, t, r, v, e in zip(
        residues.sample[sel], residues.time[sel], residues.residue[sel], residues.fD[sel],
        residues.fD_se[sel], strict=True)}
    if not any(k[0] == reference for k in value):
        raise ValueError(f"no reference sample {reference!r} at K={K} rank {rank}")
    return _residue_comparison(_residue_rows(value, reference, alpha, delta_crit),
                               reference=reference, K=K, rank=rank, alpha=alpha,
                               delta_crit=delta_crit, quantity="fD")


def compare_populations(choice, reference: str, *, alpha: float = 0.05,
                        delta_crit: float = 0.0) -> ResidueComparison:
    """Every sample against ``reference`` residue by residue in the lever-rule population of the
    low-D component (:func:`hxex.analysis.choose_populations`), where both have one (the K=1
    value between the K=2 ranks); ``delta_crit`` is usually the peptides'
    (:func:`compare_peptides` with ``quantity='population'``)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    value = {(str(s), float(t), int(r)): (float(v), float(e)) for s, t, r, v, e in zip(
        choice.sample, choice.time, choice.residue, choice.population_low,
        choice.population_low_se, strict=True)}
    if not any(k[0] == reference for k in value):
        raise ValueError(f"no reference sample {reference!r} in the choice")
    return _residue_comparison(_residue_rows(value, reference, alpha, delta_crit),
                               reference=reference, K=2, rank=1, alpha=alpha,
                               delta_crit=delta_crit, quantity="population")


@dataclass(frozen=True)
class RegionTable:
    """One row per sample, time and region: ``region``, ``start``, ``end``, ``fD`` (the mean
    over the region's residues that have a value) and ``fD_se`` (the mean of their standard
    errors: the residues share peptides, so their errors are not independent and this keeps the
    error from shrinking with the region's length), ``n_residues`` (with a value) and
    ``coverage`` (their share of the region); at ``K`` and ``rank``."""

    sample: np.ndarray
    time: np.ndarray
    region: np.ndarray
    start: np.ndarray
    end: np.ndarray
    fD: np.ndarray
    fD_se: np.ndarray
    n_residues: np.ndarray
    coverage: np.ndarray
    K: int = 1
    rank: int = 1

    def __len__(self) -> int:
        return len(self.sample)


def region_summary(
    residues: ResidueMap,
    regions: Sequence[Region] | Annotations,
    *,
    K: int = 1,
    rank: int = 1,
) -> RegionTable:
    """The mean ``fD`` of every region for every sample and time of ``residues``."""
    regions = regions.regions if isinstance(regions, Annotations) else tuple(regions)
    if not regions:
        raise ValueError("no regions given")
    sel = (residues.K == K) & (residues.rank == rank)
    rows = []
    for sample, time in sorted({(str(s), float(t)) for s, t in zip(
            residues.sample[sel], residues.time[sel], strict=True)}):
        here = sel & (residues.sample == sample) & (residues.time == time)
        res, val, err = residues.residue[here], residues.fD[here], residues.fD_se[here]
        for region in regions:
            inside = (res >= region.start) & (res <= region.end) & np.isfinite(val)
            n = int(inside.sum())
            rows.append({"sample": sample, "time": time, "region": region.name,
                         "start": region.start, "end": region.end,
                         "fD": float(np.mean(val[inside])) if n else np.nan,
                         "fD_se": float(np.nanmean(err[inside])) if n else np.nan,
                         "n_residues": n,
                         "coverage": n / (region.end - region.start + 1)})
    names = [f.name for f in fields(RegionTable) if f.name not in ("K", "rank")]
    cols = {name: np.array([r[name] for r in rows]) for name in names}
    return RegionTable(**cols, K=K, rank=rank)
