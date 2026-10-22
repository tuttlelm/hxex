# SPDX-License-Identifier: GPL-3.0-or-later
"""Comparisons within a run (docs/ANALYSIS.md, phase 3): every sample against a reference sample,
per peptide and per residue, and averages over user-defined regions.

Differences are in fractional deuteration (``fD``) at a fixed K and rank (K=1, the average, by
default), or in the lever-rule population of the K=2 low-D component (``quantity='population'``):
whether a sample shifts the balance between the populations rather than their deuteration. The
population is ``(fD_high - fD_avg) / (fD_high - fD_low)``: each replicate spectrum's K=1 value (a
precise proxy of the average deuteration while the K=1 fit spans the envelope) between the sample's
K=2 positions averaged over its replicates (each sample keeps its own positions: whether the states
themselves differ is the K=2 ``fD`` comparison's question). Per residue,
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
population).

Regions and sites are protein-specific and user-defined (:class:`Annotations`), never derived.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace

import numpy as np
from scipy import stats

from .populations import ChoiceConfig, lever_population
from .residues import ResidueMap
from .runs import resolve_sample
from .species import SpeciesFilters, _by_replicate, _combine, _records


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
    """The columns of ``cls`` that ``rows`` fill (the others keep their defaults)."""
    names = [f.name for f in fields(cls) if f.name not in _SCALAR_FIELDS and f.name in rows[0]]
    return {name: np.array([r[name] for r in rows]) for name in names}


_SCALAR_FIELDS = ("alpha", "delta_crit", "reference", "reference_run", "K", "rank", "quantity",
                  "scaled")
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
    replicate spectra, NaN with fewer than two on a side; for a population ``delta / delta_se``
    and its normal p, the K=2 positions' error included), ``n_reference`` and ``n`` (spectra) and
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
    not compared); ``scaled``: True for :meth:`scaled_by_separation`'s copy."""

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

    def __post_init__(self):  # built without the reference's sequences: the same as the sample's
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
    two_population_sets: bool = True,
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
    ``best_sets`` (:func:`hxex.analysis.best_sets`) gives each species' choice; without it, it
    is computed from the results with the same settings. ``two_population_sets=False`` turns
    this off (``[analysis] populations = "bootstrap"``): every K=2 description is compared."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if match not in MATCHES:
        raise ValueError(f"match must be one of {', '.join(MATCHES)}")
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {', '.join(QUANTITIES)}")
    if quantity == "population" and K != 2:
        raise ValueError("populations are compared at K = 2 (the lever rule)")
    if not two_population_sets:
        best_sets = None
    elif K >= 2 and best_sets is None:
        from .species import best_sets as _best_sets
        from .species import species_uptake

        best_sets = _best_sets(species_uptake(
            results, (1, K), exclude_flagged=exclude_flagged, filters=filters,
            controls=controls, min_population=choice.min_population))
    if quantity == "population":
        return _compare_populations_peptides(results, reference, samples, rank, alpha,
                                             exclude_flagged, filters, match, controls, choice,
                                             best_sets)
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
        values, errors = _by_replicate(recs, at)  # the charge states of a replicate averaged
        by_sample.setdefault((run, sample), {})[(peptide, mod, start, end, time)] = values
        ses[(run, sample, peptide, mod, start, end, time)] = errors
        fell_back[(run, sample, peptide, mod, start, end, time)] = any(x[7] for x in recs)
    present = sorted(by_sample)
    runs_present = [p[0] for p in present]
    samples_present = [p[1] for p in present]
    try:
        reference = resolve_sample(runs_present, samples_present, reference, "reference sample")
    except ValueError as exc:
        raise ValueError(f"{exc} at K={K} rank {rank}") from exc
    # the typical standard deviation of replicate spectra and the typical replicate count; at
    # K >= 2 with the gate, from the sets best described by two populations only (the others'
    # K=2 fits are splits of one population, whose scatter is not that of a second state)
    def counted(state, key) -> bool:
        return (best_sets is None or K < 2
                or (best_sets.get((*state, *key)) or (0, np.nan))[0] == 2)

    delta_crit = _delta_crit((v for state, per in by_sample.items() for key, v in per.items()
                              if counted(state, key)), alpha)
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
            why = (_one_population(best_sets, state, key, reference, rkey)
                   if K >= 2 and best_sets is not None else "")
            if why:  # not compared: one side is not two populations
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
        if f.name not in _SCALAR_FIELDS and f.name not in ("threshold", "reason", "separation")}
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


def _lever_sides(groups, choice: ChoiceConfig) -> dict:
    """Per (run, sample) and species key ``(peptide, modification, start, end, time)``: the
    lever-rule populations of the low-D component of the replicates (the charge states of each
    replicate averaged; each replicate's K=1 value between the K=2 positions averaged over
    them), the population's mean and standard error,
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
            # one entry per replicate: the charge states with all three values, averaged
            full = [_replicate_mean([s for s in charges.values() if len(s) == 3])
                    for charges in replicates.values()]
            full = [s for s in full if s is not None]
            fell_back = any(x[7] for charges in replicates.values() for s in charges.values()
                            for x in s.values())
            info = {"values": np.array([]), "value": np.nan, "se": np.nan, "sep": np.nan,
                    "fell_back": fell_back,
                    "reason": "no K=2"}
            out.setdefault(side, {})[skey] = info
            if not full:
                continue
            avg = np.array([s[(1, 1)][0] for s in full])
            low, low_se = _combine(np.array([s[(2, 1)][0] for s in full]),
                                   np.array([s[(2, 1)][1] for s in full]))
            high, high_se = _combine(np.array([s[(2, 2)][0] for s in full]),
                                     np.array([s[(2, 2)][1] for s in full]))
            avg_mean, avg_se = _combine(avg, np.array([s[(1, 1)][1] for s in full]))
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
            if not info["reason"]:
                info.update(values=(high - avg) / sep, value=float(pop), se=float(pop_se))
    return out


def _one_population(best_sets: Mapping, state, key, reference, rkey) -> str:
    """Why a K=2 comparison of ``key`` (sample ``state``) with ``rkey`` (``reference``) is not
    made: the side whose set is not best described by two populations; '' when both are."""
    sample = (best_sets.get((*state, *key)) or (0, np.nan))[0] == 2
    ref = (best_sets.get((*reference, *rkey)) or (0, np.nan))[0] == 2
    return ("" if sample and ref else "one population (reference)" if sample
            else "one population (sample)" if ref else "one population (both)")


def _compare_populations_peptides(results, reference, samples, rank, alpha, exclude_flagged,
                                  filters, match, controls, choice, best_sets
                                  ) -> PeptideComparison:
    """:func:`compare_peptides` with ``quantity='population'``."""
    groups, *_ = _records(results, (1, 2), exclude_flagged, filters, quiet=True,
                          controls_changes=controls)
    k1 = (_by_replicate(recs)[0] for key, recs in groups.items()
          if key[7:] == (1, 1) and not key[6].role)
    delta_crit = _delta_crit(k1, alpha)  # the K=1 threshold, carried over by the separation
    sides = _lever_sides(groups, choice)
    present = sorted(sides)
    runs_present, samples_present = [p[0] for p in present], [p[1] for p in present]
    try:
        reference = resolve_sample(runs_present, samples_present, reference, "reference sample")
    except ValueError as exc:
        raise ValueError(f"{exc} at K=2 (population)") from exc
    wanted = ([resolve_sample(runs_present, samples_present, s) for s in samples]
              if samples is not None else present)
    ref = sides[reference]

    def flip(x):  # rank 2: the high-D component's share
        return x if rank == 1 else 1.0 - x

    rows = []
    for state in (s for s in wanted if s != reference):
        for rkey, key in _pairs(ref, sides.get(state, {}), match):
            a, b = ref[rkey], sides[state][key]
            reason = ((_one_population(best_sets, state, key, reference, rkey)
                       if best_sets is not None else "")
                      or (f"{a['reason']} (reference)" if a["reason"]
                          else f"{b['reason']} (sample)" if b["reason"] else ""))
            t = p = delta = np.nan
            ma = mb = sa = sb = threshold = sep = np.nan
            if not reason:
                ma, sa, mb, sb = flip(a["value"]), a["se"], flip(b["value"]), b["se"]
                delta = mb - ma
                # the test takes the whole error of each population, the K=2 positions' as
                # well as the replicates' (a Welch test on the replicates alone, with the
                # positions held fixed, overstated it: HISTORY 128, 144)
                dse = float(np.hypot(sa, sb))
                if np.isfinite(dse) and dse > 0:
                    t = delta / dse
                    p = float(2 * stats.norm.sf(abs(t)))
                sep = 0.5 * (a["sep"] + b["sep"])
                threshold = delta_crit / sep
            sig = bool(np.isfinite(p) and p < alpha and abs(delta) > threshold)
            rows.append({"run": state[0], "sample": state[1], "peptide": key[0],
                         "modification": key[1], "start": key[2], "end": key[3], "time": key[4],
                         "value_reference": ma, "value_reference_se": sa, "value": mb,
                         "value_se": sb, "delta": delta, "delta_se": float(np.hypot(sa, sb)),
                         "t": t, "p": p, "n_reference": len(a["values"]), "n": len(b["values"]),
                         "significant": sig, "peptide_reference": rkey[0],
                         "modification_reference": rkey[1], "same_sequence": rkey[:2] == key[:2],
                         "control_fallback_reference": a["fell_back"],
                         "control_fallback": b["fell_back"], "threshold": threshold,
                         "reason": reason, "separation": sep})
    names = [f.name for f in fields(PeptideComparison) if f.name not in _SCALAR_FIELDS]
    cols = ({name: np.array([r[name] for r in rows]) for name in names} if rows
            else {name: np.array([]) for name in names})
    cols["run"] = cols["run"].astype(object)
    cols["reason"] = cols["reason"].astype(object)
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
    threshold: np.ndarray | None = None

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
    for (run, sample, time, res), (v, e, fb) in sorted(value.items()):
        if (run, sample) == reference or (*reference, time, res) not in value:
            continue
        rv, re_, rfb = value[(*reference, time, res)]
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
    populations (``two_reference``, ``two``); at K >= 2 a difference is significant only where
    both do (HISTORY 137). No row is left out because of it."""
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
        _residue_rows(value, reference, alpha, delta_crit, _choices(choice), both_two=K >= 2),
        choice is not None, reference=reference, K=K, rank=rank, alpha=alpha,
        delta_crit=delta_crit, quantity="fD")


def compare_populations(choice, reference: str, *, alpha: float = 0.05,
                        delta_crit: float = 0.0) -> ResidueComparison:
    """Every sample against ``reference`` residue by residue in the lever-rule population of the
    low-D component (:func:`hxex.analysis.choose_populations`), where both have one (the K=1
    value between the K=2 ranks). ``delta_crit`` is the peptides' K=1 threshold
    (:func:`compare_peptides`); each row's threshold is it divided by the mean separation of
    the two sides' K=2 values at the residue (see the module docstring). Each row records
    whether each side's choice keeps two populations (``two_reference``, ``two``): the lever
    rule can apply where the choice keeps one, but a difference is significant only where both
    keep two (HISTORY 137)."""
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
    return _residue_comparison(
        _residue_rows(value, reference, alpha, delta_crit, _choices(choice), separation,
                      both_two=True), True,
        reference=reference, K=2, rank=1, alpha=alpha, delta_crit=delta_crit,
        quantity="population")


BORDERLINE_EVIDENCE = 10.0  # summed criterion: the two-population side at least this below 0


@dataclass(frozen=True)
class PopulationCountComparison:
    """One row per run and sample (not the reference), peptide and time: whether each side's
    set of spectra is best described by one or two populations (``K_reference``, ``K``: its
    ``best_set_K``), with the evidence (``delta_criterion_reference``, ``delta_criterion``: the
    summed criterion at K = 2 minus K = 1, negative favouring two), ``differs`` and
    ``borderline``: a difference whose evidence is weak (the two-population side's criterion is
    not at least ``evidence`` below 0, or the one-population side's not above 0). The peptides
    are paired as in :func:`compare_peptides` (``peptide_reference``,
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
    reference: str = ""
    reference_run: str = ""
    evidence: float = BORDERLINE_EVIDENCE

    def __post_init__(self):
        if self.run is None:
            object.__setattr__(self, "run", np.full(len(self.sample), "", dtype=object))

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


def compare_population_counts(
    best_sets: Mapping[tuple, tuple[int, float]],
    reference: str,
    *,
    samples: Sequence[str] | None = None,
    match: str = "range",
    evidence: float = BORDERLINE_EVIDENCE,
) -> PopulationCountComparison:
    """Every sample against ``reference``: per peptide and time, whether each is best described
    by one or two populations (``best_sets``, :func:`hxex.analysis.best_sets`), paired as in
    :func:`compare_peptides`. A sample that has two populations where the reference has one (or
    the other way round) is a finding of its own, beside the K=2 comparisons, which are made
    only where both have two (owner, 2026-10-21; HISTORY 137)."""
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
            if differs:
                two, one = (d, dr) if k > kr else (dr, d)
                borderline = not (two <= -evidence and one > 0)
            rows.append({"run": state[0], "sample": state[1], "peptide": key[0],
                         "modification": key[1], "start": key[2], "end": key[3],
                         "time": key[4], "peptide_reference": rkey[0],
                         "modification_reference": rkey[1], "K_reference": kr, "K": k,
                         "delta_criterion_reference": dr, "delta_criterion": d,
                         "differs": differs, "borderline": borderline})
    names = [f.name for f in fields(PopulationCountComparison)
             if f.name not in ("reference", "reference_run", "evidence")]
    cols = {name: np.array([r[name] for r in rows]) for name in names} if rows else {
        name: np.array([]) for name in names}
    cols["run"] = np.asarray(cols["run"], dtype=object)
    for name in ("differs", "borderline"):
        cols[name] = np.asarray(cols[name], dtype=bool)
    for name in ("K_reference", "K"):
        cols[name] = np.asarray(cols[name], dtype=np.int64)
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
