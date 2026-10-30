# SPDX-License-Identifier: GPL-3.0-or-later
"""Deamidated fully deuterated controls (``[run.controls] deamidation``, opt-in).

A fully deuterated control made by heating the protein near neutral pH can deamidate its
Asn-Gly sites during the heating (docs/ANALYSIS.md "Deamidated fully deuterated controls",
HISTORY 197). Asn -> Asp adds 0.984 Da, which falls inside one deuterium bin (1.006 Da), so a
deamidated share of the molecules lands one bin up: the control's mean reads that share of a
deuteron too high, and the retention of every peptide holding the site is overstated. The
undeuterated controls and the labelled spectra are not heated.

The envelope cannot tell the share apart from a wider envelope (a binomial of ``n`` sites
mixed with a share ``p`` of itself one bin up is exactly a binomial of ``n + 1`` sites), but the
mass can: the deamidated copy is 0.022 Da lighter than a deuteron, so it moves the picked peaks
down in m/z by ``share x 0.022 Da / M`` (``M`` the ion's mass). The assessment, per sample:

- Candidates: the peptide series holding an Asn followed by Gly (base residues of the modX
  labels), or ending in an Asn whose next residue in the protein, read from an overlapping
  peptide of the same sample, is Gly. Parent-type ions only (fragments are not corrected).
- Each control spectrum's m/z offset (the picker's ``mz_offset_ppm``) relative to the median
  of the other peptides of its injection (sample, replicate, time, and role; at least
  :data:`MIN_REFERENCES`): this removes the injection's calibration.
- Each series' share: its fully deuterated controls' median relative offset minus its
  undeuterated controls', divided by the shift a fully deamidated copy gives
  (``-0.022 Da / M``).
- The sample's share: the median over its candidate series minus the median over its other
  series (at least :data:`MIN_SERIES` of each, else not assessed; tables picked elsewhere carry
  no offsets and are not assessed either).
- Deamidated when the share is at least ``deamidation_min_share``: every candidate series of
  the sample keeps its own fully deuterated control with its mean lowered by the share (in
  deuterons: a deamidated molecule sits one whole bin up), flagged ``fd_deamidated``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .spectra import FULLDEUT, UNDEUT

log = logging.getLogger(__name__)

# Asn -> Asp (O for NH) and a deuteron (2H - 1H), in Da
DEAMIDATION_DA = 0.984016
DEUTERON_DA = 1.006277
LIGHTER_DA = DEUTERON_DA - DEAMIDATION_DA  # the deamidated copy against a deuteron, 0.022 Da
MIN_SERIES = 2  # the fewest candidate and other series a sample's share is measured from
MIN_REFERENCES = 5  # the fewest other spectra an injection's m/z reference is taken from
DECISIONS = ("deamidated", "not_found", "not_assessed")


@dataclass(frozen=True)
class SampleDeamidation:
    """One sample's assessment: ``share`` (the deamidated share of its fully deuterated
    controls' Asn-Gly peptides; NaN: not assessed), ``share_se`` (its bootstrap standard error),
    ``n_candidates`` and ``n_others`` (the series with both controls measured it came from),
    and ``decision`` (:data:`DECISIONS`)."""

    sample: str
    share: float
    share_se: float
    n_candidates: int
    n_others: int
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return {"sample": self.sample, "share": self.share, "share_se": self.share_se,
                "n_candidates": self.n_candidates, "n_others": self.n_others,
                "decision": self.decision}


@dataclass(frozen=True)
class Deamidation:
    """The run's assessment: ``samples`` (one per sample with a candidate series) and
    ``candidates`` (``data_id`` -> sample of every candidate series)."""

    samples: tuple[SampleDeamidation, ...]
    candidates: Mapping[int, str]

    def of(self, data_id: int) -> SampleDeamidation | None:
        """The assessment of a candidate series' sample; None for any other series."""
        sample = self.candidates.get(int(data_id))
        if sample is None:
            return None
        return next((s for s in self.samples if s.sample == sample), None)

    def shift(self, data_id: int) -> float:
        """How much the series' fully deuterated mean is lowered (deuterons; 0 when its sample
        was not found deamidated)."""
        s = self.of(data_id)
        return s.share if s is not None and s.decision == "deamidated" else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"samples": [s.to_dict() for s in self.samples],
                "candidates": {str(d): s for d, s in self.candidates.items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Deamidation:
        def sample(s: Mapping[str, Any]) -> SampleDeamidation:  # null: not assessed (NaN)
            return SampleDeamidation(**{k: (float("nan") if v is None else v)
                                        for k, v in s.items()})
        return cls(samples=tuple(sample(s) for s in data.get("samples", ())),
                   candidates={int(d): str(s) for d, s in data.get("candidates", {}).items()})


def _base(label: str) -> str:
    """The residue of a modX label (``acK`` -> ``K``)."""
    return label[-1]


def asn_gly_series(charges: Mapping[int, tuple], residues: Mapping[int, tuple] | None,
                   registry=None) -> dict[int, str]:
    """``data_id`` -> sample of the series whose peptide holds an Asn followed by Gly, or ends
    in an Asn followed by Gly in the protein (the next residue read from an overlapping peptide
    of the same sample; :func:`hxex.controls.series_residues`). ``charges`` (``data_id`` ->
    ((sample, peptide, modification), charge), :func:`hxex.controls.charge_groups`) lists the
    peptide series (fragments are not in it)."""
    from . import chem

    registry = chem.DEFAULT_MODIFICATIONS if registry is None else registry
    residues = residues or {}
    # each sample's protein positions, from its positioned peptides
    known: dict[str, dict[int, str]] = {}
    for d, (labels, first, _amides) in residues.items():
        key = charges.get(int(d))
        if key is None:
            continue
        at = known.setdefault(key[0][0], {})
        for i, label in enumerate(labels):
            at.setdefault(first + i, _base(label))
    out: dict[int, str] = {}
    parsed: dict[str, list[str]] = {}
    for d, ((sample, peptide, _mod), _z) in charges.items():
        if peptide not in parsed:
            parsed[peptide] = [_base(x) for x in chem.parse_sequence(peptide, registry)]
        bases = parsed[peptide]
        hit = any(a == "N" and b == "G" for a, b in zip(bases, bases[1:], strict=False))
        if not hit and bases and bases[-1] == "N" and int(d) in residues:
            first = residues[int(d)][1]
            hit = known.get(sample, {}).get(first + len(bases)) == "G"
        if hit:
            out[int(d)] = sample
    return out


def _median(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def assess_deamidation(
    data_id,
    role,
    time,
    usable,
    charges: Mapping[int, tuple],
    residues: Mapping[int, tuple] | None,
    *,
    sample,
    rep,
    mass,
    mz_offset_ppm,
    min_share: float,
    registry=None,
) -> Deamidation:
    """The deamidation of the run's fully deuterated controls, per sample (module docstring).
    Per-spectrum arrays: ``data_id``, ``role``, ``time``, ``usable`` (the control spectra that
    may serve: not screened, not rejected), ``sample``, ``rep``, ``mass`` (the ion's mass: the
    m/z of its first bin times its charge) and ``mz_offset_ppm`` (the picker's; NaN when not
    measured). ``charges`` and ``residues`` as :func:`asn_gly_series`."""
    candidates = asn_gly_series(charges, residues, registry)
    if not candidates:
        return Deamidation(samples=(), candidates={})
    data_id = np.asarray(data_id, dtype=np.int64)
    role = np.asarray(role, dtype=str)
    sample = np.asarray(sample, dtype=str)
    offset = np.asarray(mz_offset_ppm, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    peptide = np.isin(data_id, list(charges))
    control = (np.asarray(usable, dtype=bool) & peptide & np.isin(role, [UNDEUT, FULLDEUT])
               & np.isfinite(offset))
    is_cand = np.isin(data_id, list(candidates))

    # each control spectrum's offset relative to the other peptides of its injection
    injection = np.array([f"{s}\x1f{r}\x1f{t!r}\x1f{x}" for s, r, t, x in
                          zip(sample, np.asarray(rep, dtype=str), np.asarray(time), role,
                              strict=True)])
    relative = np.full(len(role), np.nan)
    for u in np.unique(injection[control]):
        at = control & (injection == u)
        ref = at & ~is_cand
        if ref.sum() >= MIN_REFERENCES:
            relative[at] = offset[at] - np.median(offset[ref])

    rng = np.random.default_rng(0)
    results = []
    for s in sorted(set(candidates.values())):
        shares: dict[int, float] = {}
        for d in np.unique(data_id[control & (sample == s)]):
            rows = control & (data_id == d)
            fd, un = rows & (role == FULLDEUT), rows & (role == UNDEUT)
            m = _median(mass[fd])
            value = (_median(relative[fd]) - _median(relative[un])) / (-LIGHTER_DA / m * 1e6)
            if np.isfinite(value):
                shares[int(d)] = value
        cand = np.array([v for d, v in shares.items() if d in candidates])
        other = np.array([v for d, v in shares.items() if d not in candidates])
        if len(cand) < MIN_SERIES or len(other) < MIN_SERIES:
            results.append(SampleDeamidation(s, float("nan"), float("nan"), len(cand),
                                             len(other), "not_assessed"))
            log.info("deamidation: sample %s not assessed (%d Asn-Gly and %d other series with "
                     "both controls and their m/z offsets; %d of each needed)", s, len(cand),
                     len(other), MIN_SERIES)
            continue
        share = float(np.median(cand) - np.median(other))
        boot = [np.median(rng.choice(cand, len(cand))) for _ in range(500)]
        decision = "deamidated" if share >= min_share else "not_found"
        results.append(SampleDeamidation(s, share, float(np.std(boot)), len(cand), len(other),
                                         decision))
        log.info("deamidation: sample %s, %d Asn-Gly series: share %.2f (se %.2f) -> %s", s,
                 len(cand), share, float(np.std(boot)),
                 "deamidated: their own fully deuterated controls lowered by the share "
                 "(fd_deamidated)" if decision == "deamidated" else "not deamidated")
    return Deamidation(samples=tuple(results), candidates=candidates)


def deamidation_from_picked(picked, spectrum_id, role, usable, charges, residues, *,
                            min_share: float, registry=None) -> Deamidation:
    """:func:`assess_deamidation` for the spectra ``spectrum_id`` (with their ``role`` and
    ``usable``, a fit's order) of a picked set (:class:`hxex.picking.BinnedSet`, which may hold
    more spectra)."""
    index = {str(s): i for i, s in enumerate(np.asarray(picked.spectrum_id).astype(str))}
    rows = np.array([index[str(s)] for s in np.asarray(spectrum_id)], dtype=np.int64)
    offsets = np.asarray(picked.offsets)
    has_bins = offsets[rows + 1] > offsets[rows]
    first = np.where(has_bins, np.asarray(picked.mz, dtype=np.float64)[
        np.minimum(offsets[rows], len(picked.mz) - 1)], np.nan)
    return assess_deamidation(
        np.asarray(picked.data_id)[rows], role, np.asarray(picked.time)[rows], usable, charges,
        residues, sample=np.asarray(picked.sample)[rows], rep=np.asarray(picked.rep)[rows],
        mass=first * np.asarray(picked.charge, dtype=np.float64)[rows],
        mz_offset_ppm=np.asarray(picked.mz_offset_ppm, dtype=np.float64)[rows],
        min_share=min_share, registry=registry)
