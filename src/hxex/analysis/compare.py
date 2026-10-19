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
from .runs import resolve_sample
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


_SCALAR_FIELDS = ("alpha", "delta_crit", "reference", "reference_run", "K", "rank", "quantity")
QUANTITIES = ("fD", "population")
MATCHES = ("range", "sequence")


def _pairs(ref: dict, other: dict, match: str) -> list[tuple[tuple, tuple]]:
    """Pairs ``(reference key, sample key)`` of species keys ``(peptide, modification, start,
    end, time)``. ``'sequence'``: the same key. ``'range'``: the same start, end and time; where
    either side has several species there (e.g. a modified and an unmodified form), only those
    with the same sequence are paired, so forms are never crossed."""
    if match == "sequence":
        return [(k, k) for k in sorted(other) if k in ref]
    by_range: dict[tuple, tuple[list, list]] = {}
    for side, keys in ((0, ref), (1, other)):
        for k in keys:
            by_range.setdefault(k[2:], ([], []))[side].append(k)
    out = []
    for _, (a, b) in sorted(by_range.items()):
        if len(a) == 1 and len(b) == 1:
            out.append((a[0], b[0]))
        else:
            out.extend((k, k) for k in sorted(set(a) & set(b)))
    return out


@dataclass(frozen=True)
class PeptideComparison:
    """One row per run and sample (not the reference), peptide and time: ``run`` and ``sample``
    (as in the species table; the reference is ``(reference_run, reference)``), ``peptide``,
    ``modification``,
    ``start``, ``end``, ``time``, ``value_reference`` and ``value`` (the ``quantity``, ``fD`` or
    ``population``, combined over replicates and charge states) with their ``*_se``, ``delta``
    (``value - value_reference``) and ``delta_se``, ``t`` and ``p`` (Welch's test on the
    replicate spectra; NaN with fewer than two on a side), ``n_reference`` and ``n`` (spectra) and
    ``significant``; with the ``reference`` sample, ``K``, ``rank``, ``alpha`` and the global
    ``delta_crit``. ``peptide`` and ``modification`` are the sample's; ``peptide_reference`` and
    ``modification_reference`` the reference's, and ``same_sequence`` says whether they are the
    same (peptides matched by range may differ, e.g. by a mutation; see
    :func:`compare_peptides`). ``control_fallback_reference`` and ``control_fallback``: whether
    each side's back-exchange correction used a fallback control; where they differ the two
    values were corrected differently."""

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
    peptide_reference: np.ndarray | None = None
    modification_reference: np.ndarray | None = None
    same_sequence: np.ndarray | None = None
    control_fallback_reference: np.ndarray | None = None
    control_fallback: np.ndarray | None = None
    run: np.ndarray | None = None
    reference_run: str = ""

    def __post_init__(self):  # built without the reference's sequences: the same as the sample's
        if self.run is None:
            object.__setattr__(self, "run", np.full(len(self.sample), "", dtype=object))
        else:
            object.__setattr__(self, "run", np.asarray(self.run, dtype=object))
        if self.peptide_reference is None:
            object.__setattr__(self, "peptide_reference", np.asarray(self.peptide))
        if self.modification_reference is None:
            object.__setattr__(self, "modification_reference", np.asarray(self.modification))
        if self.same_sequence is None:
            object.__setattr__(self, "same_sequence", (
                (np.asarray(self.peptide_reference) == np.asarray(self.peptide))
                & (np.asarray(self.modification_reference) == np.asarray(self.modification))))
        for name in ("control_fallback_reference", "control_fallback"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.zeros(len(self.sample), dtype=bool))

    @property
    def corrected_differently(self) -> np.ndarray:
        """Rows whose two values were corrected differently (one with a fallback control)."""
        return np.asarray(self.control_fallback_reference, dtype=bool) != np.asarray(
            self.control_fallback, dtype=bool)

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return {"run": self.run, **{f.name: getattr(self, f.name) for f in fields(self)
                                    if f.name not in _SCALAR_FIELDS and f.name != "run"}}

    def where(self, *, sample=None, time=None, run=None) -> np.ndarray:
        keep = np.ones(len(self), dtype=bool)
        for name, value in (("run", run), ("sample", sample), ("time", time)):
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
    match: str = "range",
    controls: Mapping | None = None,
) -> PeptideComparison:
    """Every sample (or those in ``samples``) against ``reference``, per peptide and time, at
    ``K`` and ``rank``, in ``quantity`` (``'fD'``, or ``'population'`` for K >= 2; see the module
    docstring for the hybrid rule). The values are those of
    :func:`hxex.analysis.species_uptake` with the same filters. ``results`` may be a
    :class:`hxex.analysis.RunSet` (``reference`` and ``samples`` are then sample names, or
    ``run:sample`` for one run's, :func:`hxex.analysis.resolve_sample`; ``delta_crit`` is pooled
    over every run). ``match`` says which peptides are compared: ``'range'`` (the default) pairs the
    peptides over the same residues, so that a peptide containing a mutation
    is compared with the reference's (``same_sequence`` False), except where a sample has several
    species over the same residues (a modified and an unmodified form): those are paired only
    with the same sequence; where one sample has only the modified form and the other only the
    unmodified one over a range, the two are paired by range, and
    ``modification`` and ``modification_reference`` show it. ``'sequence'`` pairs only
    identical sequences and modifications.
    ``controls``: control settings to change (:func:`hxex.analysis.species.analysis_controls`)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if match not in MATCHES:
        raise ValueError(f"match must be one of {', '.join(MATCHES)}")
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {', '.join(QUANTITIES)}")
    if quantity == "population" and K < 2:
        raise ValueError("populations are compared at K >= 2")
    at = 0 if quantity == "fD" else 2  # the record's value (its standard error follows it)
    ks = (1, K) if K != 1 else (1,)
    # quiet: species_uptake reports what the filters left out
    groups, *_ = _records(results, ks, exclude_flagged, filters, quiet=True,
                          controls_changes=controls)
    by_sample: dict[tuple[str, str], dict[tuple, np.ndarray]] = {}
    ses: dict[tuple, np.ndarray] = {}
    fell_back: dict[tuple, bool] = {}
    for key, recs in groups.items():
        run, sample, peptide, mod, start, end, point, k, r = key
        if (k, r) != (K, rank) or point.role:  # the controls are not time points
            continue
        time = point.time
        by_sample.setdefault((run, sample), {})[(peptide, mod, start, end, time)] = np.array(
            [x[at] for x in recs])
        ses[(run, sample, peptide, mod, start, end, time)] = np.array(
            [x[at + 1] for x in recs])
        fell_back[(run, sample, peptide, mod, start, end, time)] = any(x[7] for x in recs)
    present = sorted(by_sample)
    runs_present = [p[0] for p in present]
    samples_present = [p[1] for p in present]
    try:
        reference = resolve_sample(runs_present, samples_present, reference, "reference sample")
    except ValueError as exc:
        raise ValueError(f"{exc} at K={K} rank {rank}") from exc
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
    wanted = ([resolve_sample(runs_present, samples_present, s) for s in samples]
              if samples is not None else present)
    others = [s for s in wanted if s != reference]
    ref = by_sample[reference]
    rows = []
    for state in others:
        per = by_sample.get(state, {})
        for rkey, key in _pairs(ref, per, match):
            a, b = ref[rkey], per[key]
            ma, sa = _combine(a, ses[(*reference, *rkey)])
            mb, sb = _combine(b, ses[(*state, *key)])
            if len(a) >= 2 and len(b) >= 2 and (np.var(a) > 0 or np.var(b) > 0):
                t, p = stats.ttest_ind(b, a, equal_var=False)
            else:
                t, p = np.nan, np.nan
            delta = mb - ma
            sig = bool(np.isfinite(p) and p < alpha and np.isfinite(delta_crit)
                       and abs(delta) > delta_crit)
            rows.append({"run": state[0], "sample": state[1], "peptide": key[0],
                         "modification": key[1], "start": key[2], "end": key[3], "time": key[4],
                         "value_reference": ma, "value_reference_se": sa, "value": mb,
                         "value_se": sb, "delta": delta, "delta_se": float(np.hypot(sa, sb)),
                         "t": float(t), "p": float(p), "n_reference": len(a), "n": len(b),
                         "significant": sig, "peptide_reference": rkey[0],
                         "modification_reference": rkey[1],
                         "same_sequence": rkey[:2] == key[:2],
                         "control_fallback_reference": fell_back[(*reference, *rkey)],
                         "control_fallback": fell_back[(*state, *key)]})
    cols = _columns(PeptideComparison, rows) if rows else {
        f.name: np.array([]) for f in fields(PeptideComparison) if f.name not in _SCALAR_FIELDS}
    if rows:
        cols["run"] = cols["run"].astype(object)
    return PeptideComparison(**cols, reference=reference[1], reference_run=reference[0], K=K,
                             rank=rank, alpha=alpha, delta_crit=delta_crit, quantity=quantity)


@dataclass(frozen=True)
class ResidueComparison:
    """One row per run and sample (not the reference), time and residue: ``value_reference``
    and ``value`` (the ``quantity``) with their ``*_se``, ``delta`` and ``delta_se``, ``z``
    (``delta / delta_se``) and ``significant``; with the ``reference`` (``(reference_run,
    reference)``), ``K``, ``rank``, ``alpha``
    and the ``delta_crit`` used. ``control_fallback_reference`` and ``control_fallback``: whether
    each side's value rests on a fallback control (see :class:`PeptideComparison`).
    ``two_reference`` and ``two``: whether each side's one-or-two choice
    (:func:`hxex.analysis.choose_populations`) keeps two populations at the residue, so that a
    difference is read beside whether each side supports two populations; ``None`` when the
    comparison was made without a choice. Rows are kept whatever the choices."""

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
    control_fallback_reference: np.ndarray | None = None
    control_fallback: np.ndarray | None = None
    two_reference: np.ndarray | None = None
    two: np.ndarray | None = None
    run: np.ndarray | None = None
    reference_run: str = ""

    def __post_init__(self):  # built without them: no fallback, one run
        for name in ("control_fallback_reference", "control_fallback"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.zeros(len(self.sample), dtype=bool))
        if self.run is None:
            object.__setattr__(self, "run", np.full(len(self.sample), "", dtype=object))
        else:
            object.__setattr__(self, "run", np.asarray(self.run, dtype=object))

    def columns(self) -> dict[str, np.ndarray]:
        return {"run": self.run, **{f.name: getattr(self, f.name) for f in fields(self)
                                    if f.name not in _SCALAR_FIELDS and f.name != "run"
                                    and getattr(self, f.name) is not None}}

    @property
    def corrected_differently(self) -> np.ndarray:
        """Rows whose two values were corrected differently (one with a fallback control)."""
        return np.asarray(self.control_fallback_reference, dtype=bool) != np.asarray(
            self.control_fallback, dtype=bool)

    @property
    def choices_differ(self) -> np.ndarray:
        """Rows where one side keeps two populations and the other one (all False without a
        choice)."""
        if self.two is None:
            return np.zeros(len(self), dtype=bool)
        return np.asarray(self.two_reference, dtype=bool) != np.asarray(self.two, dtype=bool)

    def __len__(self) -> int:
        return len(self.sample)

    def where(self, *, sample=None, time=None, run=None) -> np.ndarray:
        keep = np.ones(len(self), dtype=bool)
        for name, value in (("run", run), ("sample", sample), ("time", time)):
            if value is not None:
                values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
                keep &= np.isin(getattr(self, name), values)
        return keep


def _choices(choice) -> dict | None:
    """``(run, sample, time, residue) -> two_populations`` of a choice, where it has a K=1
    value (as :func:`hxex.analysis.compare_choices`)."""
    if choice is None:
        return None
    return {(str(u), str(s), float(t), int(r)): bool(two) for u, s, t, r, avg, two in zip(
        choice.run, choice.sample, choice.time, choice.residue, choice.fD_avg,
        choice.two_populations, strict=True) if np.isfinite(avg)}


def _residue_rows(value: dict, reference: tuple[str, str], alpha: float, delta_crit: float,
                  two: dict | None = None) -> list[dict]:
    """Rows of a residue comparison from ``(run, sample, time, residue) -> (value, se,
    control_fallback)``, with each side's choice from ``two`` (:func:`_choices`) if given."""
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    rows = []
    for (run, sample, time, res), (v, e, fb) in sorted(value.items()):
        if (run, sample) == reference or (*reference, time, res) not in value:
            continue
        rv, re_, rfb = value[(*reference, time, res)]
        if not (np.isfinite(v) and np.isfinite(rv)):
            continue
        delta = v - rv
        se = float(np.hypot(e, re_))
        z = delta / se if se > 0 else np.nan
        sig = bool(np.isfinite(z) and abs(z) > z_crit and abs(delta) > delta_crit)
        rows.append({"run": run, "sample": sample, "time": time, "residue": res,
                     "value_reference": rv, "value_reference_se": re_, "value": v,
                     "value_se": e, "delta": delta, "delta_se": se, "z": z, "significant": sig,
                     "control_fallback_reference": rfb, "control_fallback": fb})
        if two is not None:
            rows[-1]["two_reference"] = two.get((*reference, time, res), False)
            rows[-1]["two"] = two.get((run, sample, time, res), False)
    return rows


def _residue_comparison(rows, with_choice: bool, reference: tuple[str, str], **scalars
                        ) -> ResidueComparison:
    names = [f.name for f in fields(ResidueComparison) if f.name not in _SCALAR_FIELDS
             and (with_choice or f.name not in ("two_reference", "two"))]
    scalars.update(reference=reference[1], reference_run=reference[0])
    if not rows:
        return ResidueComparison(**{name: np.array([]) for name in names}, **scalars)
    cols = {name: np.array([r[name] for r in rows]) for name in names}
    cols["run"] = cols["run"].astype(object)
    return ResidueComparison(**cols, **scalars)


def compare_residues(
    residues: ResidueMap,
    reference: str,
    *,
    K: int = 1,
    rank: int = 1,
    alpha: float = 0.05,
    delta_crit: float = 0.0,
    choice=None,
) -> ResidueComparison:
    """Every sample against ``reference`` residue by residue at ``K`` and ``rank``;
    ``delta_crit`` is usually the peptides' (:func:`compare_peptides`). With a ``choice``
    (:func:`hxex.analysis.choose_populations`) each row records whether each side keeps two
    populations (``two_reference``, ``two``); no row is left out because of it."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    sel = (residues.K == K) & (residues.rank == rank)
    value = {(str(u), str(s), float(t), int(r)): (float(v), float(e), bool(fb))
             for u, s, t, r, v, e, fb in zip(
                 residues.run[sel], residues.sample[sel], residues.time[sel],
                 residues.residue[sel], residues.fD[sel], residues.fD_se[sel],
                 residues.control_fallback[sel], strict=True)}
    try:
        reference = resolve_sample([k[0] for k in value], [k[1] for k in value], reference,
                                   "reference sample")
    except ValueError as exc:
        raise ValueError(f"{exc} at K={K} rank {rank}") from exc
    return _residue_comparison(
        _residue_rows(value, reference, alpha, delta_crit, _choices(choice)),
        choice is not None, reference=reference, K=K, rank=rank, alpha=alpha,
        delta_crit=delta_crit, quantity="fD")


def compare_populations(choice, reference: str, *, alpha: float = 0.05,
                        delta_crit: float = 0.0) -> ResidueComparison:
    """Every sample against ``reference`` residue by residue in the lever-rule population of the
    low-D component (:func:`hxex.analysis.choose_populations`), where both have one (the K=1
    value between the K=2 ranks); ``delta_crit`` is usually the peptides'
    (:func:`compare_peptides` with ``quantity='population'``). Each row records whether each
    side's choice keeps two populations (``two_reference``, ``two``): the lever rule can apply
    where the choice keeps one."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    value = {(str(u), str(s), float(t), int(r)): (float(v), float(e), bool(fb))
             for u, s, t, r, v, e, fb in zip(
                 choice.run, choice.sample, choice.time, choice.residue, choice.population_low,
                 choice.population_low_se, choice.control_fallback, strict=True)}
    reference = resolve_sample(choice.run, choice.sample, reference, "reference sample")
    return _residue_comparison(
        _residue_rows(value, reference, alpha, delta_crit, _choices(choice)), True,
        reference=reference, K=2, rank=1, alpha=alpha, delta_crit=delta_crit,
        quantity="population")


@dataclass(frozen=True)
class RegionTable:
    """One row per run, sample, time and region: ``region``, ``start``, ``end``, ``fD`` (the mean
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
    run: np.ndarray | None = None

    def __post_init__(self):
        if self.run is None:
            object.__setattr__(self, "run", np.full(len(self.sample), "", dtype=object))
        else:
            object.__setattr__(self, "run", np.asarray(self.run, dtype=object))

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return {"run": self.run, **{f.name: getattr(self, f.name) for f in fields(self)
                                    if f.name not in ("K", "rank", "run")}}


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
    for run, sample, time in sorted({(str(u), str(s), float(t)) for u, s, t in zip(
            residues.run[sel], residues.sample[sel], residues.time[sel], strict=True)}):
        here = sel & (residues.run == run) & (residues.sample == sample) & (residues.time == time)
        res, val, err = residues.residue[here], residues.fD[here], residues.fD_se[here]
        for region in regions:
            inside = (res >= region.start) & (res <= region.end) & np.isfinite(val)
            n = int(inside.sum())
            rows.append({"run": run, "sample": sample, "time": time, "region": region.name,
                         "start": region.start, "end": region.end,
                         "fD": float(np.mean(val[inside])) if n else np.nan,
                         "fD_se": float(np.nanmean(err[inside])) if n else np.nan,
                         "n_residues": n,
                         "coverage": n / (region.end - region.start + 1)})
    names = [f.name for f in fields(RegionTable) if f.name not in ("K", "rank")]
    cols = {name: np.array([r[name] for r in rows]) for name in names}
    cols["run"] = cols["run"].astype(object)
    return RegionTable(**cols, K=K, rank=rank)
