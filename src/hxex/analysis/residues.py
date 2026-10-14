# SPDX-License-Identifier: GPL-3.0-or-later
"""Residue-level fractional deuteration (fD) from overlapping peptides, with redundancy and
resolution.

Each peptide's exchanging residues are its residues after the first ``nterm_subtract`` (which
exchange back too fast to count) that are not prolines, as in the amide count. The value of a
residue is the weighted mean of the peptides that cover it, each peptide weighing
``(1 / its exchanging residues) ** weight_exponent`` normalised over the residue's peptides (the
pyHDX weighting with exponent 1: a short peptide says more about each of its residues), and its
standard error ``sqrt(sum(w^2 se^2))`` (independent peptides). Redundancy is the number of
peptides covering a residue; resolution is the length of the block it lies in, the blocks being
the stretches between consecutive peptide boundaries (where a peptide's exchanging residues start
or end), so a residue resolved by peptides differing by one residue has resolution 1.

Only peptides are used (``kind == 'peptide'``); fragments are analysed differently
(docs/ANALYSIS.md, the fragment branch).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np

from .. import chem
from .species import SpeciesUptake


class SequenceMismatch(ValueError):
    """A peptide does not match the protein sequence at its position."""


@dataclass(frozen=True)
class ResidueMap:
    """One row per sample, time, ``K``, ``rank`` and residue (columns of equal length):
    ``residue`` (number on the protein), ``aa`` (its letter, '' if unknown), ``fD`` and
    ``fD_se`` (NaN where no peptide covers the residue's amide), ``redundancy``,
    ``resolution`` (NaN where uncovered), ``proline``, ``covered`` and ``control_fallback`` (a
    peptide covering the residue was corrected with a fallback control)."""

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

    def __post_init__(self):  # built without it: no fallback
        if self.control_fallback is None:
            object.__setattr__(self, "control_fallback", np.zeros(len(self.sample), dtype=bool))

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def where(self, *, sample=None, time=None, K=None, rank=None) -> np.ndarray:
        keep = np.ones(len(self), dtype=bool)
        for name, value in (("sample", sample), ("time", time), ("K", K), ("rank", rank)):
            if value is not None:
                values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
                keep &= np.isin(getattr(self, name), values)
        return keep


def exchanging_residues(peptide: str, start: int, nterm_subtract: int,
                        registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS
                        ) -> np.ndarray:
    """Residue numbers of the exchanging amides of ``peptide`` (modX) starting at ``start``."""
    residues = chem.parse_sequence(peptide, registry)
    return np.array([start + i for i, label in enumerate(residues)
                     if i >= nterm_subtract and registry.base_residue(label) != "P"],
                    dtype=np.int64)


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


def _letters(sample: str, peptides, seq_of: dict, registry) -> list[str]:
    """The residue letters of ``sample``: its sequence (each peptide checked against it), else
    what the peptides show ('' where none covers a residue)."""
    letters = seq_of.get(sample)
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


def _sequences(pep: SpeciesUptake, sequences, registry) -> dict[str, list[str]]:
    seq_of: dict[str, list[str]] = {}
    for sample in np.unique(pep.sample):
        seq = sequences if isinstance(sequences, str) else (
            sequences.get(str(sample)) if sequences is not None else None)
        if seq is not None:
            seq_of[str(sample)] = _sequence_letters(seq, registry)
    return seq_of


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
                      ) -> tuple[SpeciesUptake, dict[str, str]]:
    """``uptake`` with every modified peptide moved to a state of its own, labelled
    ``'<sample> <sites>'`` (:func:`modified_state`), and the base sample of every state."""
    labels = []
    base: dict[str, str] = {}
    for s, p, a, m in zip(uptake.sample, uptake.peptide, uptake.start, uptake.modification,
                          strict=True):
        state = modified_state(str(p), int(a), str(m), registry)
        label = f"{s} {state}" if state else str(s)
        labels.append(label)
        base[label] = str(s)
    return dataclasses.replace(uptake, sample=np.array(labels, dtype=str)), base


def _peptides_only(uptake: SpeciesUptake) -> SpeciesUptake:
    pep = uptake.select(uptake.kind == "peptide")
    if not len(pep):
        raise ValueError("no peptides in the uptake table")
    return pep


def residue_map(
    uptake: SpeciesUptake,
    *,
    sequences: str | Mapping[str, str] | None = None,
    weight_exponent: float = 1.0,
    modified: str = "separate",
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
) -> ResidueMap:
    """The residue-level values of every sample, time, K and rank of ``uptake`` (peptides only).

    ``modified`` says what happens to modified peptides (a modified residue in the modX
    sequence, or a composition change): ``'separate'`` (the default; owner, 2026-10-13) gives
    each modified form a state of its own, ``'<sample> <sites>'`` such as ``'WT pS59'``
    (:func:`separate_modified`), so the sample's values come from its unmodified peptides;
    ``'combined'`` averages every peptide of a sample together.

    ``sequences`` is the protein sequence (one for every sample, or one per sample), residue 1
    first; with it, every residue of the protein appears (uncovered ones with NaN), prolines are
    marked everywhere, and each peptide is checked against it. Without it, the residues run from
    1 to the last covered one and are known only where a peptide covers them. Redundancy and
    resolution are those of the peptides measured at that time (:func:`coverage_table` gives
    them over all times).
    """
    if weight_exponent < 0:
        raise ValueError("weight_exponent must be >= 0")
    if modified not in MODIFIED:
        raise ValueError(f"modified must be one of {', '.join(MODIFIED)}")
    pep = _peptides_only(uptake)
    if modified == "separate":
        pep, base = separate_modified(pep, registry)
        if isinstance(sequences, Mapping):  # a state has the sequence of its sample
            sequences = {state: sequences[s] for state, s in base.items() if s in sequences}
    seq_of = _sequences(pep, sequences, registry)
    cols: dict[str, list] = {f.name: [] for f in fields(ResidueMap)}
    conditions = sorted({(str(s), float(t), int(k), int(r)) for s, t, k, r in
                         zip(pep.sample, pep.time, pep.K, pep.rank, strict=True)})
    for sample, time, k, rank in conditions:
        rows = np.flatnonzero(pep.where(sample=sample, time=time, K=k, rank=rank))
        peptides = [(str(pep.peptide[i]), int(pep.start[i]), int(pep.end[i])) for i in rows]
        known = _letters(sample, peptides, seq_of, registry)
        z, redundancy, resolution = _coverage(peptides, len(known), uptake.nterm_subtract,
                                              registry)
        values, ses = _weighted(z, pep.fD[rows], pep.fD_se[rows], weight_exponent)
        fell_back = ((z > 0) & np.asarray(pep.control_fallback[rows], dtype=bool)[:, None]).any(
            axis=0)
        for res in range(len(known)):
            for name, value in (
                ("sample", sample), ("time", time), ("K", k), ("rank", rank),
                ("residue", res + 1), ("aa", known[res]), ("fD", values[res]),
                ("fD_se", ses[res]), ("redundancy", redundancy[res]),
                ("resolution", resolution[res]), ("proline", known[res] == "P"),
                ("covered", redundancy[res] > 0), ("control_fallback", bool(fell_back[res])),
            ):
                cols[name].append(value)
    out = {name: np.array(values) for name, values in cols.items()}
    for name in ("K", "rank", "residue", "redundancy"):
        out[name] = out[name].astype(np.int64)
    for name in ("time", "fD", "fD_se", "resolution"):
        out[name] = out[name].astype(np.float64)
    for name in ("proline", "covered", "control_fallback"):
        out[name] = out[name].astype(bool)
    return ResidueMap(**out)


@dataclass(frozen=True)
class CoverageTable:
    """One row per sample and residue: ``residue``, ``aa``, ``redundancy``, ``resolution`` (NaN
    where uncovered), ``proline`` and ``covered``, from the peptides of the sample at the
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

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name not in ("count", "times")}


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
    the undeuterated control (``uptake.undeut_time``) by default, as coverage is judged on the
    undeuterated spectra; another time; a list of times (the peptides measured at any of them);
    or ``'all'`` for every time. ``sequences`` as for :func:`residue_map`."""
    if count not in COUNTS:
        raise ValueError(f"count must be one of {', '.join(COUNTS)}")
    pep = _peptides_only(uptake)
    one = pep.where(K=int(pep.K.min()), rank=1)  # each spectrum once
    if time is None:
        time = uptake.undeut_time
    times: tuple[float, ...] = ()
    if not (isinstance(time, str) and time == "all"):
        if isinstance(time, str):
            raise ValueError("time must be a time, a list of times, 'all' or None")
        times = tuple(float(t) for t in np.atleast_1d(time))
        one &= np.isclose(pep.time[:, None], np.array(times)[None, :]).any(axis=1)
        if not one.any():
            shown = ", ".join(f"{t:g}" for t in times)
            raise ValueError(f"no peptide was measured at time(s) {shown}")
    pep = pep.select(one)
    seq_of = _sequences(pep, sequences, registry)
    cols: dict[str, list] = {f.name: [] for f in fields(CoverageTable)
                             if f.name not in ("count", "times")}
    for sample in sorted(set(pep.sample.tolist())):
        rows = np.flatnonzero(pep.sample == sample)
        spectra: dict[tuple, int] = {}
        for i in rows:
            key = (str(pep.peptide[i]), int(pep.start[i]), int(pep.end[i]),
                   str(pep.modification[i]))
            spectra[key] = spectra.get(key, 0) + int(pep.n_spectra[i])
        peptides = [k[:3] for k in spectra]
        weights = [spectra[k] if count == "spectra" else 1 for k in spectra]
        known = _letters(sample, peptides, seq_of, registry)
        _, redundancy, resolution = _coverage(peptides, len(known), uptake.nterm_subtract,
                                              registry, weights)
        for res in range(len(known)):
            for name, value in (("sample", sample), ("residue", res + 1), ("aa", known[res]),
                                ("redundancy", redundancy[res]), ("resolution", resolution[res]),
                                ("proline", known[res] == "P"), ("covered", redundancy[res] > 0)):
                cols[name].append(value)
    out = {name: np.array(values) for name, values in cols.items()}
    out["residue"] = out["residue"].astype(np.int64)
    out["redundancy"] = out["redundancy"].astype(np.int64)
    out["resolution"] = out["resolution"].astype(np.float64)
    for name in ("proline", "covered"):
        out[name] = out[name].astype(bool)
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


def _weighted(z: np.ndarray, values: np.ndarray, ses: np.ndarray, exponent: float):
    """Per-residue weighted values and standard errors (see the module docstring)."""
    covered = z > 0
    w = np.where(covered, z**exponent, 0.0)
    total = w.sum(axis=0)
    w = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
    any_ = covered.any(axis=0)
    out = np.where(any_, (w * values[:, None]).sum(axis=0), np.nan)
    se = np.where(any_, np.sqrt((w**2 * np.nan_to_num(ses)[:, None] ** 2).sum(axis=0)), np.nan)
    return out, se
