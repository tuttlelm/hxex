# SPDX-License-Identifier: GPL-3.0-or-later
"""Comparisons within a run (docs/ANALYSIS.md, phase 3): every sample against a reference sample,
per peptide and per residue, and averages over user-defined regions.

Differences are in fractional deuteration (``fD``) at a fixed K and rank (K=1, the average, by
default), or in the lever-rule population of the K=2 low-D component (``quantity='population'``):
whether a sample shifts the balance between the populations rather than their deuteration. The
population is ``(fD_high - fD_avg) / (fD_high - fD_low)``: each replicate spectrum's K=1 value (a
precise proxy of the average deuteration while the K=1 fit spans the envelope) between the sample's
K=2 positions averaged over its replicates. Each sample's population is shown on its own
positions, but a difference between two samples is judged on SHARED positions wherever the K=2
``fD`` comparison finds no difference in either position (the states are the same, differently
populated: owner, HISTORY 128; HISTORY 281): the positions averaged over both samples, so that
their errors cancel and the difference is the K=1 difference divided by the separation
(``positions`` says which). Where a position differs, each sample keeps its own positions (the
states themselves differ). Per residue,
:func:`compare_populations` compares the lever-rule population of the choice between one and two
populations. Per peptide and time a difference is significant by the hybrid rule (Hageman and Weis,
Anal. Chem. 2019): Welch's t-test on the replicate spectra of the two samples gives ``p < alpha``,
and the difference exceeds a threshold ``delta_crit = t(1 - alpha/2, 2n - 2) s sqrt(2/n)``, with
``s`` the typical standard deviation of replicate spectra over every peptide, time and sample,
estimated robustly (the median of the replicate variances, not their plain pool as in the original
rule: :func:`_delta_crit`), and ``n`` the typical number of replicate spectra. A population's
threshold is the K=1 ``delta_crit`` divided by the separation of the K=2 positions (the mean of the
two samples'): with fixed positions a population difference is the fD difference divided by the
separation, so the K=1 standard carries over, where a pooled threshold would be dominated by poorly
separated peptides. Per residue, where the replicates are already combined, a difference is
significant when ``|delta| / se`` exceeds the normal quantile of the same ``alpha`` and ``|delta|``
exceeds the threshold (the peptides' K=1 ``delta_crit``, divided by the residue's separation for a
population). ``hxex analyse`` reports each sample's own residue values and judges whether a
residue difference is significant from the peptides both samples measured
(:func:`paired_residue_maps`, :func:`judged_on_shared`; HISTORY 202).

Regions and sites are protein-specific and user-defined (:class:`Annotations`), never derived.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace

import numpy as np
from scipy import stats

from .. import chem
from .populations import ChoiceConfig, lever_population
from .residues import ResidueMap, residue_map, separate_modified
from .runs import resolve_sample
from .species import SpeciesFilters, _by_replicate, _combine, _k2_average_records, _records

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Region:
    """A named stretch of the protein, residues ``start`` to ``end`` inclusive."""

    name: str
    start: int
    end: int

    def __post_init__(self):
        if self.end < self.start or self.start < 1:
            raise ValueError(f"region {self.name!r}: start ({self.start}) and end ({self.end}) "
                             "are residue numbers, start at least 1 and end not before start")


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
    """The columns of ``cls`` that ``rows`` fill (the others keep their defaults)."""
    names = [f.name for f in fields(cls) if f.name not in _SCALAR_FIELDS and f.name in rows[0]]
    return {name: np.array([r[name] for r in rows]) for name in names}


_SCALAR_FIELDS = ("alpha", "delta_crit", "reference", "reference_run", "K", "rank", "quantity",
                  "scaled", "share")
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
    replicates, NaN with fewer than two on a side; for a population ``delta / delta_se``
    and its normal p, the K=2 positions' error included), ``n_reference`` and ``n`` (replicates:
    each replicate's charge states averaged first, since HISTORY 128) and
    ``significant``; with the ``reference`` sample, ``K``, ``rank``, ``alpha`` and the global
    ``delta_crit`` (for a population, the K=1 one). ``threshold``: the threshold each row was
    judged by (``delta_crit``, or for a population ``delta_crit`` divided by the separation);
    ``reason``: why a population was not compared (empty otherwise; see
    :func:`compare_peptides`). ``peptide`` and ``modification`` are the sample's;
    ``peptide_reference`` and ``modification_reference`` the reference's, and
    ``same_sequence`` says whether they are the same (peptides matched by range may differ,
    e.g. by a mutation; see
    :func:`compare_peptides`). ``control_fallback_reference`` and ``control_fallback``: whether
    each side's back-exchange correction used a fallback control; where they differ the two
    values were corrected differently. ``separation``: for a population, the fD between the
    peptide's two K=2 positions (the mean of the two samples'; NaN for fD rows and populations
    not compared); ``scaled``: True for :meth:`scaled_by_separation`'s copy. ``positions``: for
    a population compared, ``'shared'`` where neither K=2 position differs between the two
    samples (``delta`` is then the difference of the shares on the positions averaged over both,
    ``(fD_avg_reference - fD_avg) / separation`` for the low-D share, and the row takes the K=1
    comparison's ``t``, ``p`` and ``significant``: the same measurement; HISTORY 281) and
    ``'own'`` where one does (each side's own positions, the positions' errors included); empty
    for fD rows and populations not compared."""

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
    threshold: np.ndarray | None = None
    reason: np.ndarray | None = None
    separation: np.ndarray | None = None
    scaled: bool = False
    share: str = "low"  # a population comparison: of the low-D share, or the high-D one
    positions: np.ndarray | None = None

    def __post_init__(self):  # built without the reference's sequences: the same as the sample's
        if self.positions is None:
            object.__setattr__(self, "positions", np.full(len(self.sample), "", dtype=object))
        if self.threshold is None:
            object.__setattr__(self, "threshold", np.full(len(self.sample), self.delta_crit))
        if self.separation is None:
            object.__setattr__(self, "separation", np.full(len(self.sample), np.nan))
        if self.reason is None:
            object.__setattr__(self, "reason", np.full(len(self.sample), "", dtype=object))
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

    def scaled_by_separation(self) -> PeptideComparison:
        """A population comparison in fD units, as the Woods plot draws it (owner,
        2026-10-21): ``delta`` and ``delta_se`` times each row's ``separation`` (the deuterium
        that moved between the two states, as a fraction of the amides), judged by one
        threshold, ``delta_crit``, as the fD comparisons are. ``significant`` is unchanged: the
        test ``|delta| > delta_crit / separation`` is the same. The values themselves are not
        scaled. An fD comparison is returned as it is."""
        if self.quantity != "population" or self.scaled:
            return self
        sep = np.asarray(self.separation, dtype=float)
        return replace(self, delta=np.asarray(self.delta, dtype=float) * sep,
                       delta_se=np.asarray(self.delta_se, dtype=float) * sep,
                       threshold=np.full(len(self), self.delta_crit), scaled=True)

    def as_high_share(self) -> PeptideComparison:
        """A population comparison of the high-D population's share instead of the low-D one's
        (one minus it), as the Woods plot draws it so that a difference above zero is more
        deuterium in the sample, as on the fD panels (owner, 2026-11-02): ``value`` and
        ``value_reference`` are one minus the low-D shares and ``delta`` changes sign; the
        errors, the threshold and ``significant`` are the same. An fD comparison, or one
        already of the high-D share, is returned as it is."""
        if self.quantity != "population" or self.share == "high":
            return self
        return replace(self, value=1.0 - np.asarray(self.value, dtype=float),
                       value_reference=1.0 - np.asarray(self.value_reference, dtype=float),
                       delta=-np.asarray(self.delta, dtype=float), share="high")

    @property
    def corrected_differently(self) -> np.ndarray:
        """Rows whose two values were corrected differently (one with a fallback control)."""
        return np.asarray(self.control_fallback_reference, dtype=bool) != np.asarray(
            self.control_fallback, dtype=bool)

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        """The table's columns; ``threshold`` is written as ``delta_crit``."""
        out = {"run": self.run, **{f.name: getattr(self, f.name) for f in fields(self)
                                   if f.name not in _SCALAR_FIELDS
                                   and f.name not in ("run", "threshold")}}
        out["delta_crit"] = self.threshold
        return out

    def where(self, *, sample=None, time=None, run=None) -> np.ndarray:
        keep = np.ones(len(self), dtype=bool)
        for name, value in (("run", run), ("sample", sample), ("time", time)):
            if value is not None:
                values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
                keep &= np.isin(getattr(self, name), values)
        return keep


def _delta_crit(values, alpha: float) -> float:
    """``t(1 - alpha/2, 2n - 2) s sqrt(2/n)`` from replicate value arrays (the hybrid rule's
    threshold; NaN without any replicated value). ``s`` is the typical replicate standard
    deviation, robustly: each array's variance scaled by the median of the chi-square
    distribution of its degrees of freedom, then the median over the arrays (a deliberate
    difference from the plain pooled variance of Hageman and Weis: a few replicates that read
    a different envelope dominate the plain pool; owner, 2026-10-22, HISTORY 140); ``n`` is the
    typical replicate count."""
    scaled, counts = [], []
    for v in values:
        if len(v) >= 2:
            df = len(v) - 1
            scaled.append(float(np.var(v, ddof=1)) * df / stats.chi2.median(df))
            counts.append(len(v))
    if not scaled:
        return np.nan
    n_typ = float(np.median(counts))
    df = max(2 * n_typ - 2, 1.0)
    return float(stats.t.ppf(1 - alpha / 2, df) * np.sqrt(np.median(scaled))
                 * np.sqrt(2 / n_typ))


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
    choice: ChoiceConfig = ChoiceConfig(),  # noqa: B008
    best_sets: Mapping[tuple, tuple[int, float]] | None = None,
    two_population_sets: bool | None = None,
    populations: str | None = None,
    k2_averages: Collection[tuple] | None = None,
    comparisons: Sequence[PeptideComparison] | None = None,
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
    ``controls``: control settings to change (:func:`hxex.analysis.species.analysis_controls`).

    With ``quantity='population'`` (K = 2) the value is the lever-rule population of rank 1 (the
    low-D component; of rank 2, one minus it; see the module docstring), compared only where,
    on both sides, the sample's K=2 positions pass the tests of the choice between one and two
    populations (``choice``: separated at its ``confidence``, both errors below ``max_fD_se``),
    every replicate's K=1 value lies between them, and the population is within
    ``[min_population, 1 - min_population]``; otherwise the row's values are NaN and ``reason``
    names the first failing test and the side (``'not separated (reference)'``; the reasons of
    :func:`hxex.analysis.choose_populations`, plus ``'no K=2'``).

    At K >= 2 (positions and populations) a peptide is compared only where both sides' sets are
    best described by two populations (``best_set_K = 2``; owner, 2026-10-21, HISTORY 137):
    elsewhere one side's K=2 fit is a small tail, not a second state. Such a row's difference
    is NaN and ``reason`` says ``'one population (sample)'``, ``'(reference)'`` or ``'(both)'``.
    A set with more than two populations counts as two or more and is compared at its K = 2
    description, with a reason that says so (``'three populations (sample): K = 2 compared'``;
    owner, 2026-11-02). A pair whose side has no description at ``(K, rank)`` (all of it left
    out by the filters, reviewed to one population, or not fitted) keeps a row with the other
    side's value, no difference, and ``reason`` ``'no K=2 (sample)'`` (``'(reference)'``,
    ``'(both)'``): the samples and pairs are those with a K = 1 value at a time point.
    ``best_sets`` (:func:`hxex.analysis.best_sets`) gives each species' choice; without it, it
    is computed from the results with the same settings and the ``populations`` rule (None:
    the results', as ``hxex analyse`` takes it, :func:`hxex.analysis.settings.population_rule`).
    ``two_population_sets=False`` turns this off (the default with ``populations =
    "bootstrap"``): every K=2 description is compared.

    ``k2_averages`` (:func:`hxex.analysis.species.k2_average_sets` of the species table) names
    the sets whose average is the K = 2 description's population-weighted mean, each replicate
    spectrum's own (two populations far apart, where the K = 1 fit settles on one; HISTORY
    210): their K = 1 values, and the average of the lever rule, are taken from it.

    ``comparisons`` (a population only): the K=1 comparison and the K=2 rank 1 and rank 2
    comparisons made with the same settings, which decide where the positions are shared (see
    :class:`PeptideComparison` ``positions``); computed when None."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if match not in MATCHES:
        raise ValueError(f"match must be one of {', '.join(MATCHES)}")
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {', '.join(QUANTITIES)}")
    if quantity == "population" and K != 2:
        raise ValueError("populations are compared at K = 2 (the lever rule)")
    from .settings import population_rule

    populations = population_rule(populations, results)
    if two_population_sets is None:
        two_population_sets = populations != "bootstrap"
    if not two_population_sets:
        best_sets = None
    elif K >= 2 and best_sets is None:
        from .species import best_sets as _best_sets
        from .species import species_uptake

        best_sets = _best_sets(species_uptake(
            results, (1, K), exclude_flagged=exclude_flagged, filters=filters,
            controls=controls, min_population=choice.min_population,
            populations=populations))
    if quantity == "population":
        return _compare_populations_peptides(results, reference, samples, rank, alpha,
                                             exclude_flagged, filters, match, controls, choice,
                                             best_sets, k2_averages, comparisons)
    at = 0 if quantity == "fD" else 2  # the record's value (its standard error follows it)
    ks = (1, K) if K != 1 else ((1, 2) if k2_averages else (1,))
    # quiet: species_uptake reports what the filters left out
    groups, *_ = _records(results, ks, exclude_flagged, filters, quiet=True,
                          controls_changes=controls)
    if K == 1:
        groups = _k2_average_records(groups, k2_averages)
    by_sample: dict[tuple[str, str], dict[tuple, np.ndarray]] = {}
    # every species and time with a K=1 value: the pairs compared (at K >= 2 a pair without a
    # description at (K, rank) on a side gets a row saying so, never an error)
    measured: dict[tuple[str, str], dict[tuple, None]] = {}
    ses: dict[tuple, np.ndarray] = {}
    fell_back: dict[tuple, bool] = {}
    for key, recs in groups.items():
        run, sample, peptide, mod, start, end, point, k, r = key
        if point.role or (k, r) not in ((K, rank), (1, 1)):  # the controls are not time points
            continue
        skey = (peptide, mod, start, end, point.time)
        if (k, r) == (1, 1):
            measured.setdefault((run, sample), {})[skey] = None
            fell_back.setdefault((run, sample, *skey), any(x[7] for x in recs))
        if (k, r) != (K, rank):
            continue
        values, errors = _by_replicate(recs, at)  # the charge states of a replicate averaged
        by_sample.setdefault((run, sample), {})[skey] = values
        ses[(run, sample, *skey)] = errors
        fell_back[(run, sample, *skey)] = any(x[7] for x in recs)
    present = sorted(measured)
    runs_present = [p[0] for p in present]
    samples_present = [p[1] for p in present]
    reference = resolve_sample(runs_present, samples_present, reference, "reference sample")
    # the typical standard deviation of replicate spectra and the typical replicate count; at
    # K >= 2 with the gate, from the sets best described by two populations only (the others'
    # K=2 fits are splits of one population, whose scatter is not that of a second state)
    def counted(state, key) -> bool:
        return (best_sets is None or K < 2
                or (best_sets.get((*state, *key)) or (0, np.nan))[0] >= 2)

    delta_crit = _delta_crit((v for state, per in by_sample.items() for key, v in per.items()
                              if counted(state, key)), alpha)
    wanted = ([resolve_sample(runs_present, samples_present, s) for s in samples]
              if samples is not None else present)
    others = [s for s in wanted if s != reference]
    ref = by_sample.get(reference, {})
    rows = []
    for state in others:
        per = by_sample.get(state, {})
        for rkey, key in _pairs(measured[reference], measured.get(state, {}), match):
            missing = [side for side, have in (("reference", rkey in ref), ("sample", key in per))
                       if not have]
            if missing:  # no description at (K, rank) on a side: a row that says so
                rows.append(_not_described(K, missing, state, key, reference, rkey,
                                           ref.get(rkey), per.get(key), ses, fell_back))
                continue
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
            why = (_one_population(best_sets, state, key, reference, rkey)
                   if K >= 2 and best_sets is not None else "")
            if why and not why.endswith(MORE_COMPARED):  # one side is not two populations
                delta = t = p = np.nan
                sig = False
            rows.append({"run": state[0], "sample": state[1], "peptide": key[0],
                         "modification": key[1], "start": key[2], "end": key[3], "time": key[4],
                         "value_reference": ma, "value_reference_se": sa, "value": mb,
                         "value_se": sb, "delta": delta, "delta_se": float(np.hypot(sa, sb)),
                         "t": float(t), "p": float(p), "n_reference": len(a), "n": len(b),
                         "significant": sig, "peptide_reference": rkey[0],
                         "modification_reference": rkey[1],
                         "same_sequence": rkey[:2] == key[:2],
                         "control_fallback_reference": fell_back[(*reference, *rkey)],
                         "control_fallback": fell_back[(*state, *key)], "reason": why})
    cols = _columns(PeptideComparison, rows) if rows else {
        f.name: np.array([]) for f in fields(PeptideComparison)
        if f.name not in _SCALAR_FIELDS
        and f.name not in ("threshold", "reason", "separation", "positions")}
    if rows:
        cols["run"] = cols["run"].astype(object)
        cols["reason"] = cols["reason"].astype(object)
    return PeptideComparison(**cols, reference=reference[1], reference_run=reference[0], K=K,
                             rank=rank, alpha=alpha, delta_crit=delta_crit, quantity=quantity)


def _replicate_mean(spectra: list[dict]) -> dict | None:
    """One replicate's charge states averaged: ``(k, rank) -> (fD, fD_se, ..., fallback)`` with
    the mean fD and the error ``sqrt(sum se_i^2) / c`` (as :func:`hxex.analysis.species
    ._by_replicate`); None without any charge state."""
    if not spectra:
        return None
    out = {}
    for kr in spectra[0]:
        values = np.array([s[kr][0] for s in spectra])
        own = np.array([s[kr][1] for s in spectra], dtype=float)
        own = own[np.isfinite(own)]
        se = float(np.sqrt(np.sum(own**2)) / len(own)) if len(own) else np.nan
        out[kr] = (float(values.mean()), se, *spectra[0][kr][2:7],
                   any(s[kr][7] for s in spectra), spectra[0][kr][8])
    return out


def _by_pair(comparison: PeptideComparison) -> dict:
    """``(run, sample, peptide, modification, start, end, time) -> {t, p, significant}`` of a
    peptide comparison's compared rows."""
    out = {}
    for i in range(len(comparison)):
        if np.isfinite(comparison.delta[i]):
            out[(str(comparison.run[i]), str(comparison.sample[i]), str(comparison.peptide[i]),
                 str(comparison.modification[i]), int(comparison.start[i]),
                 int(comparison.end[i]), float(comparison.time[i]))] = {
                "t": float(comparison.t[i]), "p": float(comparison.p[i]),
                "significant": bool(comparison.significant[i])}
    return out


def shared_lever(reference, sample) -> tuple[float, float, float]:
    """The difference in the low-D share (sample minus reference) on positions shared by the two
    sides, each ``(avg, avg_se, low, low_se, high, high_se)`` (the K=1 value and the K=2
    positions, fD): the positions averaged over both, ``S = high - low``, and the difference
    ``(avg_reference - avg) / S``, whose error takes the two K=1 errors over ``S`` and the
    separation's error through ``|difference| se_S / S`` only (the positions' errors cancel;
    HISTORY 281). Returns the difference, its error and ``S`` (NaN where ``S <= 0``)."""
    a_avg, a_avg_se, a_low, a_low_se, a_high, a_high_se = (float(x) for x in reference)
    b_avg, b_avg_se, b_low, b_low_se, b_high, b_high_se = (float(x) for x in sample)
    sep = 0.5 * (a_high + b_high) - 0.5 * (a_low + b_low)
    if not sep > 0:
        return np.nan, np.nan, np.nan
    delta = (a_avg - b_avg) / sep
    sep_se = 0.5 * float(np.sqrt(a_low_se**2 + b_low_se**2 + a_high_se**2 + b_high_se**2))
    se = float(np.hypot(np.hypot(a_avg_se, b_avg_se) / sep, abs(delta) * sep_se / sep))
    return float(delta), se, float(sep)


def _lever_sides(groups, choice: ChoiceConfig) -> dict:
    """Per (run, sample) and species key ``(peptide, modification, start, end, time)``: the
    lever-rule populations of the low-D component of the replicates (the charge states of each
    replicate averaged: its K=1 value over every charge state that has one, the K=2 positions
    over those with both; each replicate's K=1 value between the K=2 positions averaged over
    the replicates), the population's mean and standard error,
    the separation, whether a fallback control was used, and the reason it is not compared
    (empty when it is)."""
    per: dict[tuple, dict] = {}
    for key, recs in groups.items():
        run, sample, peptide, mod, start, end, point, k, r = key
        if point.role or (k, r) not in ((1, 1), (2, 1), (2, 2)):
            continue
        spectra = per.setdefault((run, sample), {}).setdefault(
            (peptide, mod, start, end, point.time), {})
        for x in recs:  # one spectrum: its replicate (its run, merged samples, and rep), charge
            spectra.setdefault((x[8], x[6]), {}).setdefault(x[4], {})[(k, r)] = x
    lo = choice.min_population
    out: dict[tuple, dict] = {}
    for side, species in per.items():
        for skey, replicates in species.items():
            if not any((1, 1) in s for charges in replicates.values() for s in charges.values()):
                continue  # no K=1 value: not a species of the comparison
            # one entry per replicate, its charge states averaged: K=1 over every charge state
            # that has it (as compare_peptides.csv), the K=2 positions over those with both
            ones = [_replicate_mean([{(1, 1): s[(1, 1)]} for s in charges.values()
                                     if (1, 1) in s]) for charges in replicates.values()]
            twos = [_replicate_mean([{kr: s[kr] for kr in ((2, 1), (2, 2))}
                                     for s in charges.values() if (2, 1) in s and (2, 2) in s])
                    for charges in replicates.values()]
            ones = [s for s in ones if s is not None]
            full = [s for s in twos if s is not None]
            fell_back = any(x[7] for charges in replicates.values() for s in charges.values()
                            for x in s.values())
            info = {"values": np.array([]), "value": np.nan, "se": np.nan, "sep": np.nan,
                    "fell_back": fell_back,
                    "reason": "no K=2"}
            out.setdefault(side, {})[skey] = info
            if not full:
                continue
            avg = np.array([s[(1, 1)][0] for s in ones])
            low, low_se = _combine(np.array([s[(2, 1)][0] for s in full]),
                                   np.array([s[(2, 1)][1] for s in full]))
            high, high_se = _combine(np.array([s[(2, 2)][0] for s in full]),
                                     np.array([s[(2, 2)][1] for s in full]))
            avg_mean, avg_se = _combine(avg, np.array([s[(1, 1)][1] for s in ones]))
            sep = high - low
            known = np.isfinite(low_se) and np.isfinite(high_se)
            pop, pop_se = lever_population(avg_mean, low, high, avg_se, low_se, high_se)
            tests = (("not separated", known and sep - choice.z * (low_se + high_se) >= 0),
                     ("error", known and low_se < choice.max_fD_se
                      and high_se < choice.max_fD_se),
                     ("K=1 outside", bool(np.all((low < avg) & (avg < high)))),
                     ("population", bool(lo <= pop <= 1 - lo)))
            info["reason"] = next((why for why, ok in tests if not ok), "")
            info["sep"] = float(sep)
            info["lever"] = (float(avg_mean), float(avg_se), float(low), float(low_se),
                             float(high), float(high_se))
            if not info["reason"]:
                info.update(values=(high - avg) / sep, value=float(pop), se=float(pop_se))
    return out


def _not_described(K, missing, state, key, reference, rkey, a, b, ses, fell_back) -> dict:
    """The row of a pair without a description at the comparison's K and rank on the
    ``missing`` side(s) (``'no K=2 (sample)'``): the other side's value kept, no difference."""
    side = missing[0] if len(missing) == 1 else "both"
    ma = sa = mb = sb = np.nan
    if a is not None:
        ma, sa = _combine(a, ses[(*reference, *rkey)])
    if b is not None:
        mb, sb = _combine(b, ses[(*state, *key)])
    return {"run": state[0], "sample": state[1], "peptide": key[0], "modification": key[1],
            "start": key[2], "end": key[3], "time": key[4], "value_reference": ma,
            "value_reference_se": sa, "value": mb, "value_se": sb, "delta": np.nan,
            "delta_se": np.nan, "t": np.nan, "p": np.nan,
            "n_reference": 0 if a is None else len(a), "n": 0 if b is None else len(b),
            "significant": False, "peptide_reference": rkey[0],
            "modification_reference": rkey[1], "same_sequence": rkey[:2] == key[:2],
            "control_fallback_reference": fell_back[(*reference, *rkey)],
            "control_fallback": fell_back[(*state, *key)], "reason": f"no K={K} ({side})"}


# the end of the reason of a pair compared at K = 2 although a side's set has more than two
# populations (owner, 2026-11-02: a set of three counts as two or more)
MORE_COMPARED = "K = 2 compared"
_NUMBER_WORDS = {3: "three", 4: "four", 5: "five"}


def _one_population(best_sets: Mapping, state, key, reference, rkey) -> str:
    """Why a K=2 comparison of ``key`` (sample ``state``) with ``rkey`` (``reference``) is not
    made: the side whose set is not best described by two populations or more; '' when both
    are two. A side with more than two populations is compared at its K = 2 description, and
    the reason says so (``'three populations (sample): K = 2 compared'``, ending in
    :data:`MORE_COMPARED`)."""
    n_sample = (best_sets.get((*state, *key)) or (0, np.nan))[0]
    n_ref = (best_sets.get((*reference, *rkey)) or (0, np.nan))[0]
    sample, ref = n_sample >= 2, n_ref >= 2
    if not (sample and ref):
        return ("one population (reference)" if sample
                else "one population (sample)" if ref else "one population (both)")
    more = [(side, n) for side, n in (("reference", n_ref), ("sample", n_sample)) if n > 2]
    if not more:
        return ""
    if len(more) == 2 and more[0][1] == more[1][1]:
        more = [("both", more[0][1])]
    return "; ".join(f"{_NUMBER_WORDS.get(n, str(n))} populations ({side})"
                     for side, n in more) + f": {MORE_COMPARED}"


def _compare_populations_peptides(results, reference, samples, rank, alpha, exclude_flagged,
                                  filters, match, controls, choice, best_sets, k2_averages=None,
                                  comparisons=None) -> PeptideComparison:
    """:func:`compare_peptides` with ``quantity='population'``."""
    # the K=1 comparison and the K=2 positions' decide where the positions are shared
    same = dict(samples=samples, alpha=alpha, exclude_flagged=exclude_flagged, filters=filters,
                match=match, controls=controls, choice=choice, best_sets=best_sets,
                two_population_sets=best_sets is not None, k2_averages=k2_averages)
    if comparisons is None:
        comparisons = [compare_peptides(results, reference, **same),
                       *(compare_peptides(results, reference, K=2, rank=r, **same)
                         for r in (1, 2))]
    at_k1, *at_k2 = (_by_pair(c) for c in comparisons)
    groups, *_ = _records(results, (1, 2), exclude_flagged, filters, quiet=True,
                          controls_changes=controls)
    groups = _k2_average_records(groups, k2_averages)
    k1 = (_by_replicate(recs)[0] for key, recs in groups.items()
          if key[7:] == (1, 1) and not key[6].role)
    delta_crit = _delta_crit(k1, alpha)  # the K=1 threshold, carried over by the separation
    sides = _lever_sides(groups, choice)
    present = sorted(sides)
    runs_present, samples_present = [p[0] for p in present], [p[1] for p in present]
    reference = resolve_sample(runs_present, samples_present, reference, "reference sample")
    wanted = ([resolve_sample(runs_present, samples_present, s) for s in samples]
              if samples is not None else present)
    ref = sides[reference]

    def flip(x):  # rank 2: the high-D component's share
        return x if rank == 1 else 1.0 - x

    rows = []
    for state in (s for s in wanted if s != reference):
        for rkey, key in _pairs(ref, sides.get(state, {}), match):
            a, b = ref[rkey], sides[state][key]
            gate = (_one_population(best_sets, state, key, reference, rkey)
                    if best_sets is not None else "")
            more = gate if gate.endswith(MORE_COMPARED) else ""  # compared, with a note
            missing = [side for side, x in (("reference", a), ("sample", b))
                       if x["reason"] == "no K=2"]  # first, as the positions' rows say it
            reason = ((f"no K=2 ({missing[0] if len(missing) == 1 else 'both'})"
                       if missing else "")
                      or (gate if not more else "")
                      or (f"{a['reason']} (reference)" if a["reason"]
                          else f"{b['reason']} (sample)" if b["reason"] else ""))
            t = p = delta = dse = np.nan
            ma = mb = sa = sb = threshold = sep = np.nan
            sig, positions = False, ""
            if not reason:
                ma, sa, mb, sb = flip(a["value"]), a["se"], flip(b["value"]), b["se"]
                pair = (*state, *key)
                one = at_k1.get(pair)
                shared = (one is not None and all(c.get(pair) is not None for c in at_k2)
                          and not any(c[pair]["significant"] for c in at_k2))
                d = np.nan
                if shared:  # the same states: the K=1 difference over the shared separation
                    d, dse, sep = shared_lever(a["lever"], b["lever"])
                if np.isfinite(d):
                    delta = d if rank == 1 else -d  # rank 2: the high-D share
                    t = float(np.sign(delta) * abs(one["t"]))
                    p, sig, positions = one["p"], bool(one["significant"]), "shared"
                    threshold = delta_crit / sep
                else:
                    delta = mb - ma
                    # the test takes the whole error of each population, the K=2 positions'
                    # as well as the replicates' (a Welch test on the replicates alone, with
                    # the positions held fixed, overstated it: HISTORY 128, 144)
                    dse = float(np.hypot(sa, sb))
                    if np.isfinite(dse) and dse > 0:
                        t = delta / dse
                        p = float(2 * stats.norm.sf(abs(t)))
                    sep = 0.5 * (a["sep"] + b["sep"])
                    threshold = delta_crit / sep
                    sig = bool(np.isfinite(p) and p < alpha and abs(delta) > threshold)
                    positions = "own"
            rows.append({"run": state[0], "sample": state[1], "peptide": key[0],
                         "modification": key[1], "start": key[2], "end": key[3], "time": key[4],
                         "value_reference": ma, "value_reference_se": sa, "value": mb,
                         "value_se": sb, "delta": delta,
                         "delta_se": float(dse) if positions else float(np.hypot(sa, sb)),
                         "t": t, "p": p, "n_reference": len(a["values"]), "n": len(b["values"]),
                         "significant": sig, "peptide_reference": rkey[0],
                         "modification_reference": rkey[1], "same_sequence": rkey[:2] == key[:2],
                         "control_fallback_reference": a["fell_back"],
                         "control_fallback": b["fell_back"], "threshold": threshold,
                         "reason": reason or more, "separation": sep, "positions": positions})
    for row in rows:  # the rows of pairs not described at K = 2
        row.setdefault("positions", "")
    names = [f.name for f in fields(PeptideComparison) if f.name not in _SCALAR_FIELDS]
    cols = ({name: np.array([r[name] for r in rows]) for name in names} if rows
            else {name: np.array([]) for name in names})
    cols["run"] = cols["run"].astype(object)
    cols["reason"] = cols["reason"].astype(object)
    cols["positions"] = cols["positions"].astype(object)
    return PeptideComparison(**cols, reference=reference[1], reference_run=reference[0], K=2,
                             rank=rank, alpha=alpha, delta_crit=delta_crit, quantity="population")


@dataclass(frozen=True)
class ResidueComparison:
    """One row per run and sample (not the reference), time and residue: ``value_reference``
    and ``value`` (the ``quantity``) with their ``*_se``, ``delta`` and ``delta_se``, ``z``
    (``delta / delta_se``) and ``significant``; with the ``reference`` (``(reference_run,
    reference)``), ``K``, ``rank``, ``alpha``
    and the ``delta_crit`` used (for a population, the K=1 one; ``threshold`` is each row's,
    ``delta_crit`` divided by the residue's separation). ``control_fallback_reference`` and
    ``control_fallback``: whether each side's value rests on a fallback control (see
    :class:`PeptideComparison`).
    ``two_reference`` and ``two``: whether each side's one-or-two choice
    (:func:`hxex.analysis.choose_populations`) keeps two populations at the residue, so that a
    difference is read beside whether each side supports two populations; ``None`` when the
    comparison was made without a choice. Rows are kept whatever the choices.
    ``delta_shared`` and ``delta_shared_se``: with :func:`judged_on_shared`, the difference and
    error the test used, from the peptides both samples measured (``z``, ``significant`` and
    ``threshold`` are then that test's); ``None`` otherwise. ``peptide_support`` and
    ``residue_test_only``: with :func:`gated_by_peptides`, whether a covering peptide carries the
    difference (True, False, or NaN where no peptide comparison can judge it) and whether the
    residue test passed without that support (then not ``significant``); ``None`` otherwise.
    ``block_start`` and ``block_end``: the block of the sample's residue map the residue lies in
    (:class:`ResidueMap`; with :func:`judged_on_shared`, the block of the shared peptides the
    test used); NaN where unknown, ``None`` for a population comparison. ``positions``: for a
    population comparison, ``'shared'`` where the difference was judged on the K=2 positions
    averaged over both samples (no position differs there) and ``'own'`` where each side kept
    its own (:func:`compare_populations`); ``None`` for an fD comparison."""

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
    threshold: np.ndarray | None = None
    delta_shared: np.ndarray | None = None
    delta_shared_se: np.ndarray | None = None
    peptide_support: np.ndarray | None = None
    residue_test_only: np.ndarray | None = None
    block_start: np.ndarray | None = None
    block_end: np.ndarray | None = None
    positions: np.ndarray | None = None

    def __post_init__(self):  # built without them: no fallback, one run
        if self.threshold is None:
            object.__setattr__(self, "threshold", np.full(len(self.sample), self.delta_crit))
        for name in ("control_fallback_reference", "control_fallback"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.zeros(len(self.sample), dtype=bool))
        if self.run is None:
            object.__setattr__(self, "run", np.full(len(self.sample), "", dtype=object))
        else:
            object.__setattr__(self, "run", np.asarray(self.run, dtype=object))

    def columns(self) -> dict[str, np.ndarray]:
        """The table's columns; ``threshold`` is written as ``delta_crit``."""
        out = {"run": self.run, **{f.name: getattr(self, f.name) for f in fields(self)
                                   if f.name not in _SCALAR_FIELDS
                                   and f.name not in ("run", "threshold")
                                   and getattr(self, f.name) is not None}}
        out["delta_crit"] = self.threshold
        return out

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
                  two: dict | None = None, separation: dict | None = None,
                  both_two: bool = False) -> list[dict]:
    """Rows of a residue comparison from ``(run, sample, time, residue) -> (value, se,
    control_fallback)``, with each side's choice from ``two`` (:func:`_choices`) if given (with
    ``both_two``, a difference is significant only where both sides keep two populations);
    with ``separation`` (the same keys -> the K=2 positions' separation) each row's threshold
    is ``delta_crit`` divided by the mean of the two sides' separations."""
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    rows = []
    for (run, sample, time, res), (v, e, fb, *block) in sorted(value.items()):
        if (run, sample) == reference or (*reference, time, res) not in value:
            continue
        rv, re_, rfb, *_ = value[(*reference, time, res)]
        if not (np.isfinite(v) and np.isfinite(rv)):
            continue
        delta = v - rv
        se = float(np.hypot(e, re_))
        z = delta / se if se > 0 else np.nan
        threshold = delta_crit
        if separation is not None:
            sep = 0.5 * (separation[(run, sample, time, res)]
                         + separation[(*reference, time, res)])
            threshold = delta_crit / sep if sep > 0 else np.nan
        sig = bool(np.isfinite(z) and abs(z) > z_crit and abs(delta) > threshold)
        if two is not None and both_two and not (two.get((*reference, time, res), False)
                                                 and two.get((run, sample, time, res), False)):
            sig = False  # a K=2 value is compared only where both sides keep two populations
        rows.append({"run": run, "sample": sample, "time": time, "residue": res,
                     "value_reference": rv, "value_reference_se": re_, "value": v,
                     "value_se": e, "delta": delta, "delta_se": se, "z": z, "significant": sig,
                     "control_fallback_reference": rfb, "control_fallback": fb,
                     "threshold": threshold})
        if block:
            rows[-1]["block_start"], rows[-1]["block_end"] = block
        if two is not None:
            rows[-1]["two_reference"] = two.get((*reference, time, res), False)
            rows[-1]["two"] = two.get((run, sample, time, res), False)
    return rows


def _residue_comparison(rows, with_choice: bool, reference: tuple[str, str], **scalars
                        ) -> ResidueComparison:
    names = [f.name for f in fields(ResidueComparison) if f.name not in _SCALAR_FIELDS
             and f.name not in ("delta_shared", "delta_shared_se",  # judged_on_shared's
                                "peptide_support", "residue_test_only")  # and the gate's
             and (with_choice or f.name not in ("two_reference", "two"))
             and (not rows or f.name not in ("block_start", "block_end")
                  or "block_start" in rows[0])
             and (f.name != "positions" or (rows and "positions" in rows[0]))]
    scalars.update(reference=reference[1], reference_run=reference[0])
    if not rows:
        return ResidueComparison(**{name: np.array([]) for name in names}, **scalars)
    cols = {name: np.array([r[name] for r in rows]) for name in names}
    cols["run"] = cols["run"].astype(object)
    if "positions" in cols:
        cols["positions"] = cols["positions"].astype(object)
    return ResidueComparison(**cols, **scalars)


def paired_residue_maps(
    uptake,
    reference: str,
    *,
    match: str = "range",
    modified: str = "separate",
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
    **residue_kw,
) -> dict[tuple[str, str], ResidueMap]:
    """The residue maps of every state of ``uptake`` (a :class:`hxex.analysis.SpeciesUptake`)
    paired with ``reference``, each built from the peptides measured in both, so that a
    residue's difference between them is not judged on a difference in which peptides were
    averaged (owner, 2026-10-31; HISTORY 202): ``(run, state)`` -> a map holding that state's
    rows and the reference's, at each labelling time from the peptide ranges both have a K=1
    value for then (paired as :func:`compare_peptides` pairs them, by ``match``), at every K
    and rank. A modified form kept as a state of its own (``modified='separate'``) is paired
    as the map names it; the controls keep every peptide. Each state's own map
    (:func:`hxex.analysis.residue_map`) keeps all its peptides and gives the values; these
    maps only judge whether a difference is significant (:func:`judged_on_shared`).
    ``residue_kw`` goes to :func:`hxex.analysis.residue_map` (``sequences``,
    ``weight_exponent``, ...)."""
    if match not in MATCHES:
        raise ValueError(f"match must be one of {', '.join(MATCHES)}")
    pep = uptake.select(uptake.kind == "peptide")
    named, base = separate_modified(pep, registry) if modified == "separate" else (pep, {})
    present = list(dict.fromkeys(zip((str(u) for u in named.run),
                                     (str(s) for s in named.sample), strict=True)))
    reference = resolve_sample([s[0] for s in present], [s[1] for s in present], reference,
                               "reference sample")
    valued = ((named.K == 1) & (named.rank == 1) & (named.role == "")
              & np.isfinite(np.asarray(named.fD, dtype=float)))
    keys: dict[tuple[str, str], set] = {}
    for i in np.flatnonzero(valued):
        keys.setdefault((str(named.run[i]), str(named.sample[i])), set()).add(
            (str(named.peptide[i]), str(named.modification[i]), int(named.start[i]),
             int(named.end[i]), float(named.time[i])))
    states = list(zip(uptake.run.astype(str), uptake.sample.astype(str), strict=True))
    out = {}
    for state in present:
        if state == reference:
            continue
        only: dict[tuple[str, str, float], set] = {}
        for side in (reference, state):  # a time either side has: no shared peptide, no rows
            for key in keys.get(side, ()):
                only.setdefault((*side, key[4]), set())
        for rkey, key in _pairs(keys.get(reference, set()), keys.get(state, set()), match):
            only[(*reference, rkey[4])].add(rkey[2:4])
            only[(*state, key[4])].add(key[2:4])
        sides = {(s[0], base.get(s, s[1])) for s in (reference, state)}
        rows = np.array([s in sides for s in states], dtype=bool)
        both = residue_map(uptake.select(rows), modified=modified, registry=registry,
                           only=only, **residue_kw)
        keep = np.array([(str(u), str(s)) in (reference, state)
                         for u, s in zip(both.run, both.sample, strict=True)], dtype=bool)
        paired = replace(both, **{f.name: getattr(both, f.name)[keep] for f in fields(both)
                                  if getattr(both, f.name) is not None})
        if not len(paired.time_points()):  # no peptide in common at any labelled time (a
            # modified form against the unmodified reference when matched by sequence)
            log.info("no peptide of %s is paired with the reference %s at a labelled time "
                     "(peptide_match = %r): its residue differences are not judged",
                     ":".join(x for x in state if x), ":".join(x for x in reference if x),
                     match)
            continue
        out[state] = paired
    return out


def concat_tables(tables: Sequence):
    """One table from several of the same kind (a comparison per pair of samples, for
    example): every column joined, the other fields (reference, thresholds, ...) from the
    first; None for none."""
    tables = list(tables)
    if not tables:
        return None
    first = tables[0]
    return replace(first, **{f.name: np.concatenate([getattr(t, f.name) for t in tables])
                             for f in fields(first)
                             if isinstance(getattr(first, f.name), np.ndarray)})


def judged_on_shared(own: ResidueComparison, shared: ResidueComparison | None
                     ) -> ResidueComparison:
    """``own`` (a residue comparison of the samples' own residue values) with whether each
    difference is significant judged from ``shared``, the same comparison made from the
    peptides both samples measured (:func:`paired_residue_maps`; owner, 2026-10-31, HISTORY
    202): the values and ``delta`` stay each sample's own, ``z``, ``significant`` and
    ``threshold`` are the shared test's, and ``delta_shared`` and ``delta_shared_se`` are the
    difference and error it used (for a population, ``positions`` too). A residue no shared
    peptide covers keeps its values and is not significant (``delta_shared`` NaN); how many is
    logged."""
    n = len(own)
    z, sig = np.full(n, np.nan), np.zeros(n, dtype=bool)
    d_shared, se_shared = np.full(n, np.nan), np.full(n, np.nan)
    blocks = np.full((n, 2), np.nan)
    positions = (None if shared is None or shared.positions is None
                 else np.full(n, "", dtype=object))
    threshold = np.asarray(own.threshold, dtype=float).copy()
    index = ({} if shared is None else
             {key: j for j, key in enumerate(zip(shared.run, shared.sample, shared.time,
                                                  shared.residue, strict=True))})
    for i, key in enumerate(zip(own.run, own.sample, own.time, own.residue, strict=True)):
        j = index.get(key)
        if j is not None:
            z[i], sig[i] = shared.z[j], bool(shared.significant[j])
            d_shared[i], se_shared[i] = shared.delta[j], shared.delta_se[j]
            threshold[i] = shared.threshold[j]
            if positions is not None:
                positions[i] = shared.positions[j]
            if shared.block_start is not None:
                blocks[i] = shared.block_start[j], shared.block_end[j]
    lost = int(np.sum(np.isnan(d_shared)))
    if lost:
        log.info("%d of %d residue differences (%s) not tested: no peptide measured in both "
                 "samples at that time covers the residue", lost, n, own.quantity)
    with_blocks = shared is not None and shared.block_start is not None
    return replace(own, z=z, significant=sig, threshold=threshold, delta_shared=d_shared,
                   delta_shared_se=se_shared,
                   positions=positions if positions is not None else own.positions,
                   block_start=blocks[:, 0] if with_blocks else own.block_start,
                   block_end=blocks[:, 1] if with_blocks else own.block_end)


def _covers(peptides: PeptideComparison, nterm_subtract: int, registry) -> list[np.ndarray]:
    """Each compared peptide's exchanging residues (:func:`hxex.analysis.exchanging_residues`:
    an ``Ac-`` peptide keeps its first residue, prolines have none)."""
    from .residues import exchanging_residues

    return [exchanging_residues(str(p), int(a), nterm_subtract, registry)
            for p, a in zip(peptides.peptide, peptides.start, strict=True)]


def gated_by_peptides(
    comparison: ResidueComparison,
    peptides: PeptideComparison,
    *,
    nterm_subtract: int = 0,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
    modified: str = "separate",
) -> ResidueComparison:
    """``comparison`` with a residue difference significant only where a peptide covering the
    residue carries it (owner, 2026-10-31: a safeguard against error bars that are wrong, such
    as from a noise model that does not fit the data; HISTORY 206): a peptide of ``peptides``
    (the peptide comparison of the same quantity) at the same sample and time whose exchanging
    residues (:func:`hxex.analysis.exchanging_residues`: after the first ``nterm_subtract``
    unless acetylated, prolines left out) include the residue, that is significant, and
    whose difference has the sign of the residue's (``delta_shared`` where present, else
    ``delta``). ``peptide_support`` records it (NaN where no peptide comparison exists for the
    state and time, e.g. a modified form of the reference: such rows are left as they are);
    ``residue_test_only`` marks the rows the residue test called but no peptide supports, which
    are then not significant. Several nearly significant peptides can make a residue
    significant on their own: those are the ``residue_test_only`` rows, kept visible.
    ``modified``: how the residue map took modified peptides (``[analysis]
    modified_peptides``): ``'separate'`` keys a modified peptide to its own state (``'WT
    pS12'``), ``'combined'`` to its sample's."""
    from .residues import modified_state

    by_state: dict[tuple[str, str, float], list[int]] = {}
    for j in range(len(peptides)):
        sites = modified_state(str(peptides.peptide[j]), int(peptides.start[j]),
                               str(peptides.modification[j]), registry)
        # a modified form is a state of its own unless the residue map combined it
        state = (f"{peptides.sample[j]} {sites}" if sites and modified != "combined"
                 else str(peptides.sample[j]))
        by_state.setdefault((str(peptides.run[j]), state, float(peptides.time[j])), []).append(j)
    covers = _covers(peptides, nterm_subtract, registry)
    passed = np.asarray(peptides.significant, dtype=bool)
    sign = np.sign(np.asarray(peptides.delta, dtype=float))
    diff = np.asarray(comparison.delta_shared if comparison.delta_shared is not None
                      else comparison.delta, dtype=float)
    n = len(comparison)
    support = np.full(n, np.nan, dtype=object)
    for i in range(n):
        rows = by_state.get((str(comparison.run[i]), str(comparison.sample[i]),
                             float(comparison.time[i])))
        if rows is None:
            continue
        rows = np.asarray(rows)
        r = int(comparison.residue[i])
        support[i] = any(r in covers[j] and passed[j] and sign[j] == np.sign(diff[i])
                         for j in rows)
    called = np.asarray(comparison.significant, dtype=bool)
    unsupported = np.array([s is False for s in support], dtype=bool)
    return replace(comparison, significant=called & ~unsupported, peptide_support=support,
                   residue_test_only=called & unsupported)


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
    populations (``two_reference``, ``two``); at K >= 2 a difference is significant only where
    both do (HISTORY 137). No row is left out because of it. The controls' rows take no part
    (:meth:`ResidueMap.time_points`)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    residues = residues.time_points()
    sel = (residues.K == K) & (residues.rank == rank)
    value = {(str(u), str(s), float(t), int(r)): (float(v), float(e), bool(fb), float(b0),
                                                  float(b1))
             for u, s, t, r, v, e, fb, b0, b1 in zip(
                 residues.run[sel], residues.sample[sel], residues.time[sel],
                 residues.residue[sel], residues.fD[sel], residues.fD_se[sel],
                 residues.control_fallback[sel], residues.block_start[sel],
                 residues.block_end[sel], strict=True)}
    try:
        reference = resolve_sample([k[0] for k in value], [k[1] for k in value], reference,
                                   "reference sample")
    except ValueError as exc:
        raise ValueError(f"{exc} at K={K} rank {rank}") from exc
    return _residue_comparison(
        _residue_rows(value, reference, alpha, delta_crit, _choices(choice), both_two=K >= 2),
        choice is not None, reference=reference, K=K, rank=rank, alpha=alpha,
        delta_crit=delta_crit, quantity="fD")


def compare_populations(choice, reference: str, *, alpha: float = 0.05,
                        delta_crit: float = 0.0,
                        positions: Sequence[ResidueComparison] | None = None
                        ) -> ResidueComparison:
    """Every sample against ``reference`` residue by residue in the lever-rule population of the
    low-D component (:func:`hxex.analysis.choose_populations`), where both have one (the K=1
    value between the K=2 ranks). ``delta_crit`` is the peptides' K=1 threshold
    (:func:`compare_peptides`); each row's threshold is it divided by the mean separation of
    the two sides' K=2 values at the residue (see the module docstring). Each row records
    whether each side's choice keeps two populations (``two_reference``, ``two``): the lever
    rule can apply where the choice keeps one, but a difference is significant only where both
    keep two (HISTORY 137).

    ``positions``: the K=2 position comparisons (:func:`compare_residues` at K = 2, rank 1 and
    rank 2, as the report judges them: on shared peptides and gated by the peptides). Where
    neither is significant at a residue the two samples' states are taken as the same, and the
    difference is judged on the positions averaged over both (:func:`shared_lever`: ``delta``
    is ``(fD_avg_reference - fD_avg) / S``, its error the K=1 errors over ``S``, its threshold
    ``delta_crit / S``; ``positions`` ``'shared'``; HISTORY 281); elsewhere, and everywhere
    without ``positions``, each side keeps its own positions (``'own'``) and the error includes
    theirs. The values stay each sample's own."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    value = {(str(u), str(s), float(t), int(r)): (float(v), float(e), bool(fb))
             for u, s, t, r, v, e, fb in zip(
                 choice.run, choice.sample, choice.time, choice.residue, choice.population_low,
                 choice.population_low_se, choice.control_fallback, strict=True)}
    separation = {(str(u), str(s), float(t), int(r)): float(hi - lo)
                  for u, s, t, r, lo, hi in zip(choice.run, choice.sample, choice.time,
                                                choice.residue, choice.fD_low, choice.fD_high,
                                                strict=True)}
    reference = resolve_sample(choice.run, choice.sample, reference, "reference sample")
    two = _choices(choice)
    rows = _residue_rows(value, reference, alpha, delta_crit, two, separation, both_two=True)
    differ = None
    if positions is not None:  # residues where a position differs: the states differ
        differ = {(str(u), str(s), float(t), int(r))
                  for c in positions if c is not None
                  for u, s, t, r, sig in zip(c.run, c.sample, c.time, c.residue,
                                             c.significant, strict=True) if sig}
    lever = {(str(u), str(s), float(t), int(r)): x for u, s, t, r, *x in zip(
        choice.run, choice.sample, choice.time, choice.residue, choice.fD_avg,
        choice.fD_avg_se, choice.fD_low, choice.fD_low_se, choice.fD_high, choice.fD_high_se,
        strict=True)}
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    for row in rows:
        row["positions"] = "own"
        key = (row["time"], row["residue"])
        if differ is None or (row["run"], row["sample"], *key) in differ:
            continue
        d, se, sep = shared_lever(lever[(*reference, *key)],
                                  lever[(row["run"], row["sample"], *key)])
        if not (np.isfinite(d) and np.isfinite(se) and se > 0):
            continue
        threshold = delta_crit / sep
        both = two.get((*reference, *key), False) and two.get(
            (row["run"], row["sample"], *key), False)
        row.update(delta=d, delta_se=se, z=d / se, threshold=threshold, positions="shared",
                   significant=bool(both and abs(d / se) > z_crit and abs(d) > threshold))
    return _residue_comparison(
        rows, True,
        reference=reference, K=2, rank=1, alpha=alpha, delta_crit=delta_crit,
        quantity="population")


BORDERLINE_EVIDENCE = 10.0  # summed criterion: the two-population side at least this below 0
# with the population test (owner, 2026-10-26, HISTORY 175): a difference is clear when the side
# with more populations has p below significance / this and the other side above significance
# times this
BORDERLINE_P_FACTOR = 10.0


@dataclass(frozen=True)
class PopulationCountComparison:
    """One row per run and sample (not the reference), peptide and time: whether each side's
    set of spectra is best described by one or two populations (``K_reference``, ``K``: its
    ``best_set_K``), with the evidence (``delta_criterion_reference``, ``delta_criterion``: the
    summed criterion at K = 2 minus K = 1, negative favouring two), ``differs`` and
    ``borderline``: a difference whose evidence is weak (the two-population side's criterion is
    not at least ``evidence`` below 0, or the one-population side's not above 0). With the
    population test, ``p_reference`` and ``p`` are each side's p-value at the step where they
    differ (``set_p_1`` for one against two), and ``borderline`` comes from them instead: clear
    only when the side with more populations has p below ``significance /``
    :data:`BORDERLINE_P_FACTOR` and the other side above ``significance`` times it (NaN p:
    borderline). The peptides are paired as in :func:`compare_peptides` (``peptide_reference``,
    ``modification_reference``). ``reference`` and ``reference_run`` name the reference."""

    sample: np.ndarray
    peptide: np.ndarray
    modification: np.ndarray
    start: np.ndarray
    end: np.ndarray
    time: np.ndarray
    peptide_reference: np.ndarray
    modification_reference: np.ndarray
    K_reference: np.ndarray
    K: np.ndarray
    delta_criterion_reference: np.ndarray
    delta_criterion: np.ndarray
    differs: np.ndarray
    borderline: np.ndarray
    run: np.ndarray | None = None
    p_reference: np.ndarray | None = None
    p: np.ndarray | None = None
    reference: str = ""
    reference_run: str = ""
    evidence: float = BORDERLINE_EVIDENCE

    def __post_init__(self):
        if self.run is None:
            object.__setattr__(self, "run", np.full(len(self.sample), "", dtype=object))
        for name in ("p_reference", "p"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.full(len(self.sample), np.nan))

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return {"run": self.run, **{f.name: getattr(self, f.name) for f in fields(self)
                                    if f.name not in ("run", "reference", "reference_run",
                                                      "evidence")}}

    def where(self, *, sample=None, time=None, run=None) -> np.ndarray:
        keep = np.ones(len(self), dtype=bool)
        for name, value in (("run", run), ("sample", sample), ("time", time)):
            if value is not None:
                values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
                keep &= np.isin(getattr(self, name), values)
        return keep


def _step_p(p: tuple[float, ...] | None, step: int) -> float:
    """The p-value of step K (K against K + 1) of a set's ``(set_p_1, set_p_2, ...)``."""
    if p is None or step - 1 >= len(p):
        return np.nan
    return float(p[step - 1])


def compare_population_counts(
    best_sets: Mapping[tuple, tuple[int, float]],
    reference: str,
    *,
    samples: Sequence[str] | None = None,
    match: str = "range",
    evidence: float = BORDERLINE_EVIDENCE,
    set_p: Mapping[tuple, tuple[float, ...]] | None = None,
    significance: float = 0.01,
) -> PopulationCountComparison:
    """Every sample against ``reference``: per peptide and time, whether each is best described
    by one or two populations (``best_sets``, :func:`hxex.analysis.best_sets`), paired as in
    :func:`compare_peptides`. A sample that has two populations where the reference has one (or
    the other way round) is a finding of its own, beside the K=2 comparisons, which are made
    only where both have two (owner, 2026-10-21; HISTORY 137). ``set_p`` (the population
    test's p-values of each set, by the keys of ``best_sets``: :func:`hxex.analysis.set_pvalues`)
    judges ``borderline`` by the test, at its ``significance``, instead of ``evidence``."""
    if match not in MATCHES:
        raise ValueError(f"match must be one of {', '.join(MATCHES)}")
    by_state: dict[tuple[str, str], dict[tuple, tuple[int, float]]] = {}
    for (run, sample, peptide, mod, start, end, time), value in best_sets.items():
        by_state.setdefault((run, sample), {})[(peptide, mod, start, end, time)] = value
    present = sorted(by_state)
    runs_present, samples_present = [p[0] for p in present], [p[1] for p in present]
    reference = resolve_sample(runs_present, samples_present, reference, "reference sample")
    wanted = ([resolve_sample(runs_present, samples_present, s) for s in samples]
              if samples is not None else present)
    ref = by_state[reference]
    rows = []
    for state in (s for s in wanted if s != reference):
        for rkey, key in _pairs(ref, by_state.get(state, {}), match):
            (kr, dr), (k, d) = ref[rkey], by_state[state][key]
            differs = kr != k and kr > 0 and k > 0
            borderline = False
            p_ref = p_here = np.nan
            if set_p is not None and kr > 0 and k > 0:
                step = min(kr, k)  # the step the two sides differ at (K against K + 1)
                p_ref = _step_p(set_p.get((*reference, *rkey)), step)
                p_here = _step_p(set_p.get((*state, *key)), step)
            if differs and set_p is not None:
                more, fewer = (p_here, p_ref) if k > kr else (p_ref, p_here)
                borderline = not (more < significance / BORDERLINE_P_FACTOR
                                  and fewer > significance * BORDERLINE_P_FACTOR)
            elif differs:
                two, one = (d, dr) if k > kr else (dr, d)
                borderline = not (two <= -evidence and one > 0)
            rows.append({"run": state[0], "sample": state[1], "peptide": key[0],
                         "modification": key[1], "start": key[2], "end": key[3],
                         "time": key[4], "peptide_reference": rkey[0],
                         "modification_reference": rkey[1], "K_reference": kr, "K": k,
                         "delta_criterion_reference": dr, "delta_criterion": d,
                         "differs": differs, "borderline": borderline,
                         "p_reference": p_ref, "p": p_here})
    names = [f.name for f in fields(PopulationCountComparison)
             if f.name not in ("reference", "reference_run", "evidence")]
    cols = {name: np.array([r[name] for r in rows]) for name in names} if rows else {
        name: np.array([]) for name in names}
    cols["run"] = np.asarray(cols["run"], dtype=object)
    for name in ("differs", "borderline"):
        cols[name] = np.asarray(cols[name], dtype=bool)
    for name in ("K_reference", "K"):
        cols[name] = np.asarray(cols[name], dtype=np.int64)
    for name in ("p_reference", "p"):
        cols[name] = np.asarray(cols[name], dtype=np.float64)
    return PopulationCountComparison(**cols, reference=reference[1], reference_run=reference[0],
                                     evidence=evidence)


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


@dataclass(frozen=True)
class RegionComparison:
    """One row per run, sample (not the reference), time and region: ``region``, ``start``,
    ``end``, ``n_residues`` (the region's residues compared on the shared peptides) and
    ``coverage`` (their share of the region), ``delta`` (the mean of its residues' own
    differences), ``delta_shared`` and ``delta_shared_se`` (the mean of their shared
    differences and of their errors: residues share peptides, so the error is not shrunk),
    ``z``, ``significant``, ``peptide_support`` (a significant peptide of the same sign with at
    least half of its exchanging residues inside the region; NaN where no peptide comparison
    exists for the state) and ``region_test_only`` (the test passed without that support: not
    significant), ``threshold`` (the ``delta_crit`` applied). The reference is
    ``(reference_run, reference)``."""

    sample: np.ndarray
    time: np.ndarray
    region: np.ndarray
    start: np.ndarray
    end: np.ndarray
    n_residues: np.ndarray
    coverage: np.ndarray
    delta: np.ndarray
    delta_shared: np.ndarray
    delta_shared_se: np.ndarray
    z: np.ndarray
    significant: np.ndarray
    peptide_support: np.ndarray
    region_test_only: np.ndarray
    threshold: np.ndarray
    run: np.ndarray | None = None
    reference: str = ""
    reference_run: str = ""
    alpha: float = 0.05

    def __post_init__(self):
        if self.run is None:
            object.__setattr__(self, "run", np.full(len(self.sample), "", dtype=object))
        else:
            object.__setattr__(self, "run", np.asarray(self.run, dtype=object))

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        """The table's columns; ``threshold`` is written as ``delta_crit``."""
        out = {"run": self.run, **{f.name: getattr(self, f.name) for f in fields(self)
                                   if f.name not in ("run", "reference", "reference_run",
                                                     "alpha", "threshold")}}
        out["delta_crit"] = self.threshold
        return out

    def where(self, *, sample=None, time=None, run=None) -> np.ndarray:
        keep = np.ones(len(self), dtype=bool)
        for name, value in (("run", run), ("sample", sample), ("time", time)):
            if value is not None:
                values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
                keep &= np.isin(getattr(self, name), values)
        return keep


def compare_regions(
    comparison: ResidueComparison,
    regions: Sequence[Region] | Annotations,
    peptides: PeptideComparison,
    *,
    nterm_subtract: int = 0,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
    modified: str = "separate",
) -> RegionComparison:
    """Each sample against the reference per region and time (owner, 2026-10-31; HISTORY 209),
    from ``comparison`` (the K = 1 residue comparison judged on shared peptides,
    :func:`judged_on_shared`): the region's difference is the mean of its residues' shared
    differences and its error the mean of their errors (not shrunk: residues share peptides);
    significant when ``|delta| / se`` exceeds the normal quantile of ``alpha`` and ``|delta|``
    the residues' threshold, and (the gate of :func:`gated_by_peptides`) a significant peptide of
    ``peptides`` at that sample and time, of the same sign, has at least half of its exchanging
    residues (:func:`hxex.analysis.exchanging_residues`) inside the region; the test's other
    calls are ``region_test_only``. A region without a compared residue is not tested (NaN).
    Evaluated on the noise simulation's truth before building (private residue_gate
    README). ``modified`` as in :func:`gated_by_peptides`."""
    from .residues import modified_state

    regions = regions.regions if isinstance(regions, Annotations) else tuple(regions)
    if not regions:
        raise ValueError("no regions given")
    z_crit = float(stats.norm.ppf(1 - comparison.alpha / 2))
    by_state: dict[tuple[str, str, float], list[int]] = {}
    for j in range(len(peptides)):
        sites = modified_state(str(peptides.peptide[j]), int(peptides.start[j]),
                               str(peptides.modification[j]), registry)
        # a modified form is a state of its own unless the residue map combined it
        state = (f"{peptides.sample[j]} {sites}" if sites and modified != "combined"
                 else str(peptides.sample[j]))
        by_state.setdefault((str(peptides.run[j]), state, float(peptides.time[j])), []).append(j)
    covers = _covers(peptides, nterm_subtract, registry)
    passed = np.asarray(peptides.significant, dtype=bool)
    p_sign = np.sign(np.asarray(peptides.delta, dtype=float))
    shared = np.asarray(comparison.delta_shared if comparison.delta_shared is not None
                        else comparison.delta, dtype=float)
    shared_se = np.asarray(comparison.delta_shared_se if comparison.delta_shared_se is not None
                           else comparison.delta_se, dtype=float)
    rows = []
    for run, sample, time in sorted({(str(u), str(s), float(t)) for u, s, t in zip(
            comparison.run, comparison.sample, comparison.time, strict=True)}):
        here = ((comparison.run == run) & (comparison.sample == sample)
                & (comparison.time == time))
        res = np.asarray(comparison.residue)[here]
        own = np.asarray(comparison.delta, dtype=float)[here]
        d_sh, se_sh = shared[here], shared_se[here]
        crit = np.asarray(comparison.threshold, dtype=float)[here]
        peps = np.asarray(by_state.get((run, sample, time), []), dtype=np.int64)
        for region in regions:
            inside = (res >= region.start) & (res <= region.end)
            tested = inside & np.isfinite(d_sh) & np.isfinite(se_sh)
            n = int(tested.sum())
            delta = float(np.nanmean(own[inside])) if inside.any() else np.nan
            d = float(d_sh[tested].mean()) if n else np.nan
            se = float(se_sh[tested].mean()) if n else np.nan
            threshold = float(crit[tested][0]) if n else np.nan
            z = d / se if n and se > 0 else np.nan
            called = bool(np.isfinite(z) and abs(z) > z_crit and abs(d) > threshold)
            support: object = np.nan
            if len(peps):
                support = any(
                    passed[j] and p_sign[j] == np.sign(d) and len(covers[j])
                    and np.sum((covers[j] >= region.start) & (covers[j] <= region.end))
                    >= 0.5 * len(covers[j]) for j in peps)
            unsupported = support is False
            rows.append({"run": run, "sample": sample, "time": time, "region": region.name,
                         "start": region.start, "end": region.end, "n_residues": n,
                         "coverage": n / (region.end - region.start + 1), "delta": delta,
                         "delta_shared": d, "delta_shared_se": se, "z": z,
                         "significant": called and not unsupported,
                         "peptide_support": support, "region_test_only": called and unsupported,
                         "threshold": threshold})
    names = [f.name for f in fields(RegionComparison)
             if f.name not in ("reference", "reference_run", "alpha")]
    cols = {name: np.array([r[name] for r in rows], dtype=object if name in (
        "run", "peptide_support", "region") else None) for name in names} if rows else {
        name: np.array([]) for name in names}
    for name in ("significant", "region_test_only"):
        cols[name] = np.asarray(cols[name], dtype=bool)
    return RegionComparison(**cols, reference=comparison.reference,
                            reference_run=comparison.reference_run, alpha=comparison.alpha)


def region_summary(
    residues: ResidueMap,
    regions: Sequence[Region] | Annotations,
    *,
    K: int = 1,
    rank: int = 1,
) -> RegionTable:
    """The mean ``fD`` of every region for every sample and time point of ``residues`` (the
    controls' rows take no part: :meth:`ResidueMap.time_points`)."""
    residues = residues.time_points()
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
