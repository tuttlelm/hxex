# SPDX-License-Identifier: GPL-3.0-or-later
"""Residue-level fractional deuteration (fD) from overlapping peptides, with redundancy and
resolution.

Each peptide's exchanging residues are its residues after the first ``nterm_subtract`` (which
exchange back too fast to count) that are not prolines, as in the amide count. The value of a
residue is the weighted mean of the peptides that cover it, each peptide weighing
``(1 / its exchanging residues) ** weight_exponent`` normalised over the residue's peptides (the
pyHDX weighting with exponent 1: a short peptide says more about each of its residues), and its
standard error ``sqrt(sum(w^2 se^2))`` (independent peptides). A peptide without a value (NaN)
does not contribute, so a residue covered only by such peptides is NaN; a residue's standard
error is unknown (NaN) when any peptide contributing to it has an unknown one (no bootstrap and
a single spectrum). Redundancy is the number of peptides covering a residue; resolution is the
length of the block it lies in, the blocks being the stretches between consecutive peptide
boundaries (where a peptide's exchanging residues start or end), so a residue resolved by
peptides differing by one residue has resolution 1.

Only peptides are used (``kind == 'peptide'``); fragments are analysed differently
(docs/ANALYSIS.md, the fragment branch).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np

from .. import chem
from ..spectra import UNDEUT, point_order_key, time_point
from .species import SpeciesUptake


class SequenceMismatch(ValueError):
    """A peptide does not match the protein sequence at its position."""


@dataclass(frozen=True)
class ResidueMap:
    """One row per run, sample, time, ``K``, ``rank`` and residue (columns of equal length):
    ``run`` and ``sample`` as in the species table (``run`` '' for one run or a merged sample),
    ``residue`` (number on the protein), ``aa`` (its letter, '' if unknown), ``fD`` and
    ``fD_se`` (NaN where no peptide covers the residue's amide, or no covering peptide has a
    value; ``fD_se`` also where a contributing peptide's error is unknown), ``redundancy``,
    ``resolution`` (NaN where uncovered), ``proline``, ``covered``, ``control_fallback`` (a
    peptide covering the residue was corrected with a fallback control) and ``no_amide`` (the
    residue has no exchanging amide: a proline, or the first residue(s) of the protein, which
    no peptide can measure; its row is empty whatever the coverage). ``role`` is the point's
    (``'undeut'``, ``'fulldeut'`` or ``''``; a control's ``time`` is NaN unless it was also a
    time point). ``linked_next``: one peptide measures both this residue and the next (a
    proline or another residue without an amide between them included), so a line may join
    them. ``two_population_weight`` (K=1 rows, with ``two_population_peptides``): the share of
    the residue's weight carried by peptides whose set is best described by two populations
    (``best_set_K = 2``); NaN otherwise. With the population test (``population_test``, K=1
    rows): ``test_strength_2`` and ``test_agreement_2`` (two populations or more: the
    combined p-value of the covering sets and the share of their weight that has two or more),
    ``test_strength_3`` and ``test_agreement_3`` (three or more), and ``population_tier``
    (:func:`population_tiers`); NaN and '' otherwise."""

    sample: np.ndarray
    time: np.ndarray
    K: np.ndarray
    rank: np.ndarray
    residue: np.ndarray
    aa: np.ndarray
    fD: np.ndarray
    fD_se: np.ndarray
    redundancy: np.ndarray
    resolution: np.ndarray
    proline: np.ndarray
    covered: np.ndarray
    control_fallback: np.ndarray | None = None
    no_amide: np.ndarray | None = None
    role: np.ndarray | None = None
    run: np.ndarray | None = None
    linked_next: np.ndarray | None = None
    two_population_weight: np.ndarray | None = None
    test_strength_2: np.ndarray | None = None
    test_agreement_2: np.ndarray | None = None
    test_strength_3: np.ndarray | None = None
    test_agreement_3: np.ndarray | None = None
    population_tier: np.ndarray | None = None

    def __post_init__(self):  # built without them: no fallback; prolines only; time points
        for name in TEST_COLUMNS:
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.full(len(self.sample), np.nan))
        if self.population_tier is None:
            object.__setattr__(self, "population_tier",
                               np.full(len(self.sample), "", dtype=object))
        if self.linked_next is None:
            object.__setattr__(self, "linked_next", np.zeros(len(self.sample), dtype=bool))
        if self.two_population_weight is None:
            object.__setattr__(self, "two_population_weight",
                               np.full(len(self.sample), np.nan))
        if self.run is None:
            object.__setattr__(self, "run", np.full(len(self.sample), "", dtype=object))
        else:
            object.__setattr__(self, "run", np.asarray(self.run, dtype=object))
        if self.control_fallback is None:
            object.__setattr__(self, "control_fallback", np.zeros(len(self.sample), dtype=bool))
        if self.no_amide is None:
            object.__setattr__(self, "no_amide", np.asarray(self.proline, dtype=bool).copy())
        if self.role is None:
            object.__setattr__(self, "role", np.full(len(self.sample), "", dtype="<U8"))
        else:
            object.__setattr__(self, "role", np.asarray(self.role, dtype=str))

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return {"run": self.run, **{f.name: getattr(self, f.name) for f in fields(self)
                                    if f.name != "run"}}

    def time_points(self) -> ResidueMap:
        """The rows of the labelling time points: the controls' rows left out. Controls take no
        part in any per-residue choice or comparison (:func:`hxex.analysis.choose_populations`,
        the residue comparisons and :func:`hxex.analysis.region_summary`); their values stay in
        the residue table."""
        keep = self.role == ""
        if keep.all():
            return self
        return dataclasses.replace(self, **{f.name: getattr(self, f.name)[keep]
                                            for f in fields(self)})

    def where(self, *, sample=None, time=None, K=None, rank=None, role=None,
              run=None) -> np.ndarray:
        keep = np.ones(len(self), dtype=bool)
        for name, value in (("run", run), ("sample", sample), ("time", time), ("K", K),
                            ("rank", rank), ("role", role)):
            if value is not None:
                values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
                keep &= np.isin(getattr(self, name), values)
        return keep


def exchanging_residues(peptide: str, start: int, nterm_subtract: int,
                        registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS
                        ) -> np.ndarray:
    """Residue numbers of the exchanging amides of ``peptide`` (modX) starting at ``start``
    (``nterm_subtract`` does not apply after an ``Ac-`` group)."""
    residues = chem.parse_sequence(peptide, registry)
    nterm_subtract = chem.effective_nterm_subtract(peptide, nterm_subtract)
    return np.array([start + i for i, label in enumerate(residues)
                     if i >= nterm_subtract and registry.base_residue(label) != "P"],
                    dtype=np.int64)


def _protein_nterm(peptides, nterm_subtract: int) -> int:
    """How many of the protein's first residues have no amide: ``nterm_subtract``, or 0 when a
    peptide ``(sequence, start, end)`` starting at residue 1 is acetylated (``Ac-``)."""
    acetylated = any(start == 1 and chem.split_termini(seq)[0] == "Ac-"
                     for seq, start, _ in peptides)
    return 0 if acetylated else nterm_subtract


def _sequence_letters(sequence: str, registry) -> list[str]:
    return [registry.base_residue(label) for label in chem.parse_sequence(sequence, registry)]


def check_sequence(peptide: str, start: int, letters: list[str], registry) -> None:
    """Raise :class:`SequenceMismatch` if ``peptide`` is not ``letters`` at ``start`` (base
    residues compared, so ``pS`` matches ``S``)."""
    own = _sequence_letters(peptide, registry)
    there = letters[start - 1 : start - 1 + len(own)]
    if own != there:
        raise SequenceMismatch(f"{peptide} at {start} does not match the sequence there "
                               f"({''.join(there)!r})")


def _letters(state: tuple[str, str], peptides, seq_of: dict, registry) -> list[str]:
    """The residue letters of ``state`` (run, sample): its sequence (each peptide checked
    against it), else what the peptides show ('' where none covers a residue)."""
    letters = seq_of.get(state)
    if letters is not None:
        for p, s, _ in peptides:
            check_sequence(p, s, letters, registry)
        return list(letters)
    known = [""] * max(e for _, _, e in peptides)
    for p, s, _ in peptides:
        for off, aa in enumerate(_sequence_letters(p, registry)):
            if 0 <= s - 1 + off < len(known):
                known[s - 1 + off] = aa
    return known


def _sequences(pep: SpeciesUptake, sequences, registry, base: Mapping | None = None
               ) -> dict[tuple[str, str], list[str]]:
    """(run, sample) -> residue letters, from ``sequences`` (one for every sample, or by
    sample with per-run entries, :func:`hxex.analysis.sample_setting`); ``base`` maps a
    modified state (run, state) to its sample (:func:`separate_modified`)."""
    from .runs import sample_setting

    seq_of: dict[tuple[str, str], list[str]] = {}
    for state in states(pep):
        sample = base.get(state, state[1]) if base else state[1]
        seq = sample_setting(sequences, state[0], sample)
        if seq is not None:
            seq_of[state] = _sequence_letters(seq, registry)
    return seq_of


def states(table) -> list[tuple[str, str]]:
    """The distinct ``(run, sample)`` pairs of a table, in order of first appearance."""
    return list(dict.fromkeys(zip((str(r) for r in table.run), (str(s) for s in table.sample),
                                  strict=True)))


MODIFIED = ("separate", "combined")


def modified_state(peptide: str, start: int, modification: str = "",
                   registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS) -> str:
    """The sites of a peptide's modifications, e.g. ``'pS59'`` or ``'pS59,acK64'`` (modified
    residues of the modX sequence, numbered on the protein), with its ``modification`` text in
    brackets if it changes the composition; '' for an unmodified peptide. Terminal groups are
    not counted."""
    sites = [f"{label}{start + i}" for i, label in
             enumerate(chem.parse_sequence(peptide, registry))
             if registry.base_residue(label) != label]
    spec = chem.parse_modification(modification or None)
    if spec.composition or spec.hex:
        kept = [t for t in modification.split() if not t.startswith(("ion_type:", "undeut_mz:"))]
        sites.append(f"[{' '.join(kept)}]")
    return ",".join(sites)


def separate_modified(uptake: SpeciesUptake,
                      registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS
                      ) -> tuple[SpeciesUptake, dict[tuple[str, str], str]]:
    """``uptake`` with every modified peptide moved to a state of its own, labelled
    ``'<sample> <sites>'`` (:func:`modified_state`) in ``sample``, and the base sample of
    every state ``(run, state label)``."""
    labels = []
    base: dict[tuple[str, str], str] = {}
    for u, s, p, a, m in zip(uptake.run, uptake.sample, uptake.peptide, uptake.start,
                             uptake.modification, strict=True):
        state = modified_state(str(p), int(a), str(m), registry)
        label = f"{s} {state}" if state else str(s)
        labels.append(label)
        base[(str(u), label)] = str(s)
    return dataclasses.replace(uptake, sample=np.array(labels, dtype=str)), base


def _peptides_only(uptake: SpeciesUptake) -> SpeciesUptake:
    pep = uptake.select(uptake.kind == "peptide")
    if not len(pep):
        raise ValueError("no peptides in the uptake table")
    return pep


TEST_COLUMNS = ("test_strength_2", "test_agreement_2", "test_strength_3", "test_agreement_3")
# the residue tiers of the population test (owner, HISTORY 148; up to three, HISTORY 170)
TIER_STRENGTH = 0.01  # strong: the combined p-value below this
TIER_AGREEMENT = 1 / 3  # agreeing: this share of the covering weight has the populations
TIER_LOCALISED = 0.5  # localised: at least this share
Z_CLIP = 3.09  # a set's z = Phi^-1(1 - p), clipped (p = 0.001)
TIERS = ("no sign", "undecided", "two", "two, localised", "three", "three, localised")


def _step_tier(strength: np.ndarray, agreement: np.ndarray) -> np.ndarray:
    """0 no sign, 1 undecided (one criterion), 2 strong and agreeing, 3 also localised; -1
    where unknown."""
    with np.errstate(invalid="ignore"):
        strong = strength < TIER_STRENGTH
        agree = agreement >= TIER_AGREEMENT - 1e-12
        code = np.where(strong & (agreement >= TIER_LOCALISED - 1e-12), 3,
                        np.where(strong & agree, 2, np.where(strong ^ agree, 1, 0)))
    return np.where(np.isfinite(strength) & np.isfinite(agreement), code, -1)


def population_tiers(strength_2, agreement_2, strength_3, agreement_3) -> np.ndarray:
    """The residue tier (:data:`TIERS`; '' where unknown) from the strength and agreement of
    each step: the first step's four tiers (two, localised: strong and agreement >= 0.5; two:
    strong and agreement >= 1/3; undecided: one of the two; no sign), and "three" or "three,
    localised" where the first step has two and the second step reaches its two tiers by the
    same rule. Two neighbouring two-state regions can read as three at their junction
    (HISTORY 169)."""
    one = _step_tier(np.asarray(strength_2, float), np.asarray(agreement_2, float))
    two = _step_tier(np.asarray(strength_3, float), np.asarray(agreement_3, float))
    code = np.where((one >= 2) & (two >= 2), two + 2, one)
    names = np.array(TIERS + ("",), dtype=object)
    return names[np.where(code < 0, len(TIERS), code)]


def _test_levels(z: np.ndarray, p: np.ndarray, reads: np.ndarray, exponent: float,
                 tested: np.ndarray):
    """Per residue: the strength (the weighted Stouffer combination of the covering sets' z
    values; a set tested at step one without a p-value at this step, one that read one, enters
    with z = 0) and the agreement (the share of the covering weight whose sets read at least
    the step's populations); both NaN where no set has a p-value at this step. The sets never
    tested (``tested``: a p-value at step one) take no part (audit 2026-10-28)."""
    from scipy.stats import norm

    if not np.isfinite(p).any():  # no set tested at this step (k_max below it)
        nan = np.full(z.shape[1], np.nan)
        return nan, nan.copy()
    w = np.where((z > 0) & tested[:, None], z**exponent, 0.0)
    total = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        zs = np.where(np.isfinite(p), np.clip(norm.isf(np.clip(p, 0.0, 1.0)), -Z_CLIP, Z_CLIP),
                      0.0)
        combined = (w * zs[:, None]).sum(axis=0) / np.sqrt((w**2).sum(axis=0))
        strength = np.where(total > 0, norm.sf(combined), np.nan)
        agreement = np.where(total > 0, (w * reads.astype(float)[:, None]).sum(axis=0)
                             / np.where(total > 0, total, 1.0), np.nan)
    return strength, agreement


def residue_map(
    uptake: SpeciesUptake,
    *,
    sequences: str | Mapping[str, str] | None = None,
    weight_exponent: float = 1.0,
    modified: str = "separate",
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
    two_population_peptides: bool | None = None,
    population_test: bool | None = None,
    only: Mapping[tuple[str, str, float], Collection[tuple[int, int]]] | None = None,
) -> ResidueMap:
    """The residue-level values of every sample, time, K and rank of ``uptake`` (peptides only).

    With ``two_population_peptides`` (owner, 2026-10-21 and 2026-10-22; HISTORY 137 and 141)
    the K=1 rows carry ``two_population_weight``, the share of the residue's weight that the
    peptides whose set at that time is best described by two populations (``best_set_K = 2``,
    :func:`hxex.analysis.species_uptake`) carry (every covering peptide votes, by the residue's
    weights), which :func:`hxex.analysis.choose_populations` requires to be at least half. The
    K=2 values use every covering peptide, as K=1 does, so the three describe the same peptides
    (averaging the K=2 values over the two-population peptides alone gave curves that did not
    bracket K=1); the vote decides which residues have two populations.

    With ``population_test`` (the species table's sets tested, ``populations =
    'population_test'``) the K=1 rows also carry the residue levels of the test
    (:data:`TEST_COLUMNS`, ``population_tier``): each covering peptide's set enters with its
    p-values (``set_p_1``, ``set_p_2``) and the test's number of populations (``set_test_K``),
    weighted as the values are.

    Both follow the rule the species table was made with when not given
    (:attr:`SpeciesUptake.populations`): ``two_population_peptides`` with ``best_set`` and the
    population test (where the table has K = 2), ``population_test`` with the test.

    ``modified`` says what happens to modified peptides (a modified residue in the modX
    sequence, or a composition change): ``'separate'`` (the default) gives
    each modified form a state of its own, ``'<sample> <sites>'`` such as ``'WT pS59'``
    (:func:`separate_modified`), so the sample's values come from its unmodified peptides;
    ``'combined'`` averages every peptide of a sample together.

    ``sequences`` is the protein sequence (one for every sample, or one per sample, with
    per-run entries under a run's name; :func:`hxex.analysis.sample_setting`), residue 1
    first; with it, every residue of the protein appears (uncovered ones with NaN), prolines are
    marked everywhere, and each peptide is checked against it. Without it, the residues run from
    1 to the last covered one and are known only where a peptide covers them. Redundancy and
    resolution are those of the peptides measured at that time (:func:`coverage_table` gives
    them over all times).

    ``only`` restricts the peptides of a state at a labelling time: ``(run, state, time)`` (the
    state as the map names it, a modified form's included) -> the ``(start, end)`` ranges
    to use, at every K and rank; a state and time it does not name keep every peptide, one
    left without a peptide has no rows, and the controls are never restricted.
    :func:`hxex.analysis.paired_residue_maps` uses it to build both sides of a comparison from
    the same peptides.
    """
    if weight_exponent < 0:
        raise ValueError("weight_exponent must be >= 0")
    if modified not in MODIFIED:
        raise ValueError(f"modified must be one of {', '.join(MODIFIED)}")
    rule = uptake.populations
    if population_test is None:
        population_test = rule == "population_test"
    if two_population_peptides is None:
        two_population_peptides = (rule in ("best_set", "population_test")
                                   and bool(np.any(np.asarray(uptake.K) == 2)))
    pep = _peptides_only(uptake)
    base = None
    if modified == "separate":
        pep, base = separate_modified(pep, registry)  # a state has the sequence of its sample
    seq_of = _sequences(pep, sequences, registry, base)
    cols: dict[str, list] = {f.name: [] for f in fields(ResidueMap)}
    points = time_point(pep.time, pep.role)  # a time, or a control role
    first = {}
    for i, key in enumerate(points):
        first.setdefault(key, i)
    conditions = sorted({(str(u), str(s), key, int(k), int(r)) for u, s, key, k, r in
                         zip(pep.run, pep.sample, points, pep.K, pep.rank, strict=True)},
                        key=lambda c: (c[0], c[1],
                                       point_order_key(float(pep.time[first[c[2]]]),
                                                       str(pep.role[first[c[2]]])),
                                       c[3], c[4]))
    best_k = np.asarray(pep.best_set_K) if two_population_peptides else None
    for run, sample, key, k, rank in conditions:
        rows = np.flatnonzero(pep.where(run=run, sample=sample, K=k, rank=rank)
                              & (points == key))
        time, role = float(pep.time[rows[0]]), str(pep.role[rows[0]])
        if only is not None and not role and (run, sample, time) in only:
            ranges = only[(run, sample, time)]
            rows = rows[[(int(pep.start[i]), int(pep.end[i])) in ranges for i in rows]]
            if not len(rows):
                continue
        peptides = [(str(pep.peptide[i]), int(pep.start[i]), int(pep.end[i])) for i in rows]
        known = _letters((run, sample), peptides, seq_of, registry)
        z, redundancy, resolution = _coverage(peptides, len(known), uptake.nterm_subtract,
                                              registry)
        values, ses = _weighted(z, pep.fD[rows], pep.fD_se[rows], weight_exponent)
        linked = _linked(peptides, len(known), uptake.nterm_subtract, registry)
        n_free = _protein_nterm(peptides, uptake.nterm_subtract)  # residues without an amide
        support = np.full(len(known), np.nan)
        if best_k is not None and k == 1:  # every covering peptide votes, by its weight
            w = np.where(z > 0, z**weight_exponent, 0.0)
            total = w.sum(axis=0)
            two = (best_k[rows] == 2).astype(float)
            support = np.where(total > 0, (w * two[:, None]).sum(axis=0)
                               / np.where(total > 0, total, 1.0), np.nan)
        levels = {name: np.full(len(known), np.nan) for name in TEST_COLUMNS}
        tier = np.full(len(known), "", dtype=object)
        if population_test and k == 1:
            tested = np.isfinite(np.asarray(pep.set_p_1, float)[rows])
            for step, name in ((1, "2"), (2, "3")):
                s, a = _test_levels(z, np.asarray(getattr(pep, f"set_p_{step}"), float)[rows],
                                    np.asarray(pep.set_test_K)[rows] >= step + 1,
                                    weight_exponent, tested)
                levels[f"test_strength_{name}"], levels[f"test_agreement_{name}"] = s, a
            tier = population_tiers(*(levels[name] for name in TEST_COLUMNS))
        fell_back = ((z > 0) & np.asarray(pep.control_fallback[rows], dtype=bool)[:, None]).any(
            axis=0)
        for res in range(len(known)):
            for name, value in (
                ("run", run), ("sample", sample), ("time", time), ("role", role), ("K", k),
                ("rank", rank), ("residue", res + 1), ("aa", known[res]), ("fD", values[res]),
                ("fD_se", ses[res]), ("redundancy", redundancy[res]),
                ("resolution", resolution[res]), ("proline", known[res] == "P"),
                ("covered", redundancy[res] > 0), ("control_fallback", bool(fell_back[res])),
                ("no_amide", known[res] == "P" or res + 1 <= n_free),
                ("linked_next", bool(linked[res])),
                ("two_population_weight", float(support[res])),
                *((name, float(levels[name][res])) for name in TEST_COLUMNS),
                ("population_tier", str(tier[res])),
            ):
                cols[name].append(value)
    out = {name: np.array(values) for name, values in cols.items()}
    for name in ("K", "rank", "residue", "redundancy"):
        out[name] = out[name].astype(np.int64)
    for name in ("time", "fD", "fD_se", "resolution", "two_population_weight", *TEST_COLUMNS):
        out[name] = out[name].astype(np.float64)
    for name in ("proline", "covered", "control_fallback", "no_amide", "linked_next"):
        out[name] = out[name].astype(bool)
    out["run"] = out["run"].astype(object)
    out["population_tier"] = out["population_tier"].astype(object)
    return ResidueMap(**out)


class NoPeptideAtTime(ValueError):
    """:func:`coverage_table` was asked for times at which no peptide was measured."""


@dataclass(frozen=True)
class CoverageTable:
    """One row per run, sample and residue: ``residue``, ``aa``, ``redundancy``, ``resolution`` (NaN
    where uncovered), ``proline``, ``covered`` and ``no_amide`` (a proline or the first
    residue(s) of the protein: nothing can cover it), from the peptides of the sample at the
    ``times`` used (empty for every time). ``count`` says what redundancy counts:
    ``'peptides'``, or ``'spectra'`` (every replicate and charge state)."""

    sample: np.ndarray
    residue: np.ndarray
    aa: np.ndarray
    redundancy: np.ndarray
    resolution: np.ndarray
    proline: np.ndarray
    covered: np.ndarray
    count: str = "peptides"
    times: tuple[float, ...] = ()
    no_amide: np.ndarray | None = None
    run: np.ndarray | None = None

    def __post_init__(self):  # built without it: prolines only
        if self.no_amide is None:
            object.__setattr__(self, "no_amide", np.asarray(self.proline, dtype=bool).copy())
        if self.run is None:
            object.__setattr__(self, "run", np.full(len(self.sample), "", dtype=object))
        else:
            object.__setattr__(self, "run", np.asarray(self.run, dtype=object))

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return {"run": self.run, **{f.name: getattr(self, f.name) for f in fields(self)
                                    if f.name not in ("count", "times", "run")}}


COUNTS = ("peptides", "spectra")


def coverage_table(
    uptake: SpeciesUptake,
    *,
    sequences: str | Mapping[str, str] | None = None,
    count: str = "peptides",
    time: float | Sequence[float] | str | None = None,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
) -> CoverageTable:
    """Redundancy and resolution of every residue of every sample, from its peptides at ``time``:
    the undeuterated controls (the rows with the role ``'undeut'``) by default, as coverage is
    judged on the undeuterated spectra; a time; a list of times (the peptides measured at any
    of them); or ``'all'`` for every point. ``sequences`` as for :func:`residue_map`."""
    if count not in COUNTS:
        raise ValueError(f"count must be one of {', '.join(COUNTS)}")
    pep = _peptides_only(uptake)
    one = pep.where(K=int(pep.K.min()), rank=1)  # each spectrum once
    times: tuple[float, ...] = ()
    if time is None:
        one &= pep.role == UNDEUT
        if not one.any():
            raise NoPeptideAtTime("no peptide has an undeuterated control")
    elif not (isinstance(time, str) and time == "all"):
        if isinstance(time, str):
            raise ValueError("time must be a time, a list of times, 'all' or None")
        times = tuple(float(t) for t in np.atleast_1d(time))
        one &= np.isclose(pep.time[:, None], np.array(times)[None, :]).any(axis=1)
        if not one.any():
            shown = ", ".join(f"{t:g}" for t in times)
            raise NoPeptideAtTime(f"no peptide was measured at time(s) {shown} s")
    pep = pep.select(one)
    seq_of = _sequences(pep, sequences, registry)
    cols: dict[str, list] = {f.name: [] for f in fields(CoverageTable)
                             if f.name not in ("count", "times")}
    for run, sample in sorted(states(pep)):
        rows = np.flatnonzero(pep.where(run=run, sample=sample))
        spectra: dict[tuple, int] = {}
        for i in rows:
            key = (str(pep.peptide[i]), int(pep.start[i]), int(pep.end[i]),
                   str(pep.modification[i]))
            spectra[key] = spectra.get(key, 0) + int(pep.n_spectra[i])
        peptides = [k[:3] for k in spectra]
        weights = [spectra[k] if count == "spectra" else 1 for k in spectra]
        known = _letters((run, sample), peptides, seq_of, registry)
        _, redundancy, resolution = _coverage(peptides, len(known), uptake.nterm_subtract,
                                              registry, weights)
        n_free = _protein_nterm(peptides, uptake.nterm_subtract)
        for res in range(len(known)):
            for name, value in (("run", run), ("sample", sample), ("residue", res + 1),
                                ("aa", known[res]),
                                ("redundancy", redundancy[res]), ("resolution", resolution[res]),
                                ("proline", known[res] == "P"), ("covered", redundancy[res] > 0),
                                ("no_amide", known[res] == "P"
                                 or res + 1 <= n_free)):
                cols[name].append(value)
    out = {name: np.array(values) for name, values in cols.items()}
    out["residue"] = out["residue"].astype(np.int64)
    out["redundancy"] = out["redundancy"].astype(np.int64)
    out["resolution"] = out["resolution"].astype(np.float64)
    for name in ("proline", "covered", "no_amide"):
        out[name] = out[name].astype(bool)
    out["run"] = out["run"].astype(object)
    return CoverageTable(**out, count=count, times=times)


def _coverage(peptides, n_res: int, nterm: int, registry, weights=None):
    """For peptides ``(sequence, start, end)``: the coefficient matrix ``z`` (``1 / exchanging
    residues`` where a peptide's amide exchanges), redundancy (``weights`` per peptide, 1 by
    default) and resolution of residues 1..n_res."""
    z = np.zeros((len(peptides), n_res))
    bounds: set[int] = set()
    for a, (seq, start, _) in enumerate(peptides):
        ex = exchanging_residues(seq, start, nterm, registry)
        ex = ex[(ex >= 1) & (ex <= n_res)]
        if len(ex):
            z[a, ex - 1] = 1.0 / len(ex)
            # block boundaries: where this peptide's exchanging stretch starts and ends
            bounds.update((int(ex.min()), int(ex.max()) + 1))
    w = np.ones(len(peptides)) if weights is None else np.asarray(weights, dtype=float)
    redundancy = ((z > 0) * w[:, None]).sum(axis=0).astype(np.int64)
    resolution = np.full(n_res, np.nan)
    edges = sorted(bounds)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        block = np.arange(lo, hi)  # residue numbers in [lo, hi)
        inside = block[(block >= 1) & (block <= n_res)]
        inside = inside[redundancy[inside - 1] > 0]
        resolution[inside - 1] = len(block)
    return z, redundancy, resolution


def _linked(peptides, n_res: int, nterm: int, registry) -> np.ndarray:
    """Per residue 1..n_res: whether one of ``peptides`` measures both it and the next residue
    (both within the stretch of its exchanging amides, residues without an amide included)."""
    linked = np.zeros(n_res, dtype=bool)
    for seq, start, _ in peptides:
        ex = exchanging_residues(seq, start, nterm, registry)
        ex = ex[(ex >= 1) & (ex <= n_res)]
        if len(ex) > 1:
            linked[int(ex.min()) - 1:int(ex.max()) - 1] = True  # residue r links to r + 1
    return linked


def _weighted(z: np.ndarray, values: np.ndarray, ses: np.ndarray, exponent: float):
    """Per-residue weighted values and standard errors (see the module docstring): peptides
    without a value are left out of the design, and an unknown standard error (NaN) of a
    contributing peptide makes the residue's unknown."""
    values, ses = np.asarray(values, dtype=float), np.asarray(ses, dtype=float)
    covered = (z > 0) & np.isfinite(values)[:, None]
    w = np.where(covered, z**exponent, 0.0)
    total = w.sum(axis=0)
    w = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
    any_ = covered.any(axis=0)
    out = np.where(any_, (w * np.nan_to_num(values)[:, None]).sum(axis=0), np.nan)
    var = np.where(w > 0, w**2 * ses[:, None] ** 2, 0.0).sum(axis=0)  # NaN only where it counts
    se = np.where(any_, np.sqrt(var), np.nan)
    return out, se
