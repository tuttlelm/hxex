# SPDX-License-Identifier: GPL-3.0-or-later
"""The choice between one and two populations, residue by residue (docs/ANALYSIS.md, phase 2).

Each residue has three values from the fixed-K fits (:func:`hxex.analysis.residue_map`): the K=1
value ``fD_avg`` and the K=2 ranks ``fD_low`` and ``fD_high``, each with a standard error. Many
peptides report on a residue, so whether its data support two populations is judged there rather
than spectrum by spectrum (the owner's pyHXExpress ``choose_fits``). The two-population
description is kept when every test passes:

- the populations are separated: ``(fD_high - fD_low) - z (se_low + se_high) >= 0``, with ``z``
  for the ``confidence`` interval (50 % by default, as ``choose_fits``);
- both standard errors are below ``max_fD_se``;
- the K=1 value lies between them, ``fD_low < fD_avg < fD_high`` (the same molecules);
- the lever-rule population of the low-D component, ``(fD_high - fD_avg) / (fD_high - fD_low)``,
  is within ``[min_population, 1 - min_population]``.

Otherwise the K=1 value describes the residue. The standard error of the population is propagated
from all three values (``choose_fits`` left a derivative unsquared).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from scipy.stats import norm

from .residues import ResidueMap

REASONS = ("", "no K=2", "not separated", "error", "K=1 outside", "population")


@dataclass(frozen=True)
class ChoiceConfig:
    """Limits of :func:`choose_populations` (defaults as the owner's ``choose_fits``)."""

    confidence: float = 0.5
    max_fD_se: float = 0.25
    min_population: float = 0.05

    def __post_init__(self):
        if not 0 <= self.confidence < 1:
            raise ValueError("confidence must be in [0, 1)")
        if self.max_fD_se <= 0:
            raise ValueError("max_fD_se must be positive")
        if not 0 <= self.min_population < 0.5:
            raise ValueError("min_population must be in [0, 0.5)")

    @property
    def z(self) -> float:
        return float(norm.ppf(0.5 + self.confidence / 2)) if self.confidence > 0 else 0.0

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data) -> ChoiceConfig:
        return cls(**data)


@dataclass(frozen=True)
class PopulationChoice:
    """One row per sample, time and residue (columns of equal length): ``fD_avg`` (K=1),
    ``fD_low`` and ``fD_high`` (K=2 ranks 1 and 2) with their ``*_se``; ``population_low`` (the
    low-D component's share by the lever rule, NaN where it does not apply) with
    ``population_low_se``; ``two_populations`` (the choice) and ``reason`` (why not: one of
    ``REASONS``; empty when two populations are chosen or the residue has no K=1 value either).
    The chosen description is ``fD_low`` and ``fD_high`` where ``two_populations``, else
    ``fD_avg``. ``control_fallback``: a peptide behind the residue was corrected with a fallback
    control."""

    sample: np.ndarray
    time: np.ndarray
    residue: np.ndarray
    fD_avg: np.ndarray
    fD_avg_se: np.ndarray
    fD_low: np.ndarray
    fD_low_se: np.ndarray
    fD_high: np.ndarray
    fD_high_se: np.ndarray
    population_low: np.ndarray
    population_low_se: np.ndarray
    two_populations: np.ndarray
    reason: np.ndarray
    config: ChoiceConfig = ChoiceConfig()  # noqa: RUF009
    control_fallback: np.ndarray | None = None

    def __post_init__(self):  # built without it: no fallback
        if self.control_fallback is None:
            object.__setattr__(self, "control_fallback", np.zeros(len(self.sample), dtype=bool))

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "config"}

    def where(self, *, sample=None, time=None) -> np.ndarray:
        keep = np.ones(len(self), dtype=bool)
        for name, value in (("sample", sample), ("time", time)):
            if value is not None:
                values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
                keep &= np.isin(getattr(self, name), values)
        return keep


def lever_population(avg, low, high, se_avg, se_low, se_high):
    """The low-D component's share ``(high - avg) / (high - low)`` and its standard error from
    the three standard errors (first-order propagation; NaN where ``high <= low``)."""
    avg, low, high = (np.asarray(x, dtype=float) for x in (avg, low, high))
    span = np.where(high > low, high - low, np.nan)
    p = (high - avg) / span
    d_avg = -1.0 / span
    d_low = (high - avg) / span**2
    d_high = (avg - low) / span**2
    se = np.sqrt((d_avg * np.asarray(se_avg)) ** 2 + (d_low * np.asarray(se_low)) ** 2
                 + (d_high * np.asarray(se_high)) ** 2)
    return p, se


def choose_populations(residues: ResidueMap, config: ChoiceConfig = ChoiceConfig()  # noqa: B008
                       ) -> PopulationChoice:
    """The choice for every sample, time and residue of ``residues`` that has a K=1 value or
    K=2 values (see the module docstring)."""
    table: dict[tuple, dict] = {}
    fell_back: dict[tuple, bool] = {}
    for i in range(len(residues)):
        k, rank = int(residues.K[i]), int(residues.rank[i])
        if (k, rank) not in ((1, 1), (2, 1), (2, 2)):
            continue
        key = (str(residues.sample[i]), float(residues.time[i]), int(residues.residue[i]))
        name = {(1, 1): "avg", (2, 1): "low", (2, 2): "high"}[(k, rank)]
        table.setdefault(key, {})[name] = (float(residues.fD[i]), float(residues.fD_se[i]))
        fell_back[key] = fell_back.get(key, False) or bool(residues.control_fallback[i])
    keys = sorted(table)
    nan = (np.nan, np.nan)

    def column(name, part):
        return np.array([table[key].get(name, nan)[part] for key in keys], dtype=float)

    avg, avg_se = column("avg", 0), column("avg", 1)
    low, low_se = column("low", 0), column("low", 1)
    high, high_se = column("high", 0), column("high", 1)
    pop, pop_se = lever_population(avg, low, high, avg_se, low_se, high_se)
    lo = config.min_population
    with np.errstate(invalid="ignore"):
        has_k2 = np.isfinite(low) & np.isfinite(high) & np.isfinite(avg)
        separated = (high - low) - config.z * (np.nan_to_num(low_se) + np.nan_to_num(high_se)) >= 0
        precise = ~(low_se >= config.max_fD_se) & ~(high_se >= config.max_fD_se)
        between = (low < avg) & (avg < high)
        in_range = (pop >= lo) & (pop <= 1 - lo)
    reason = np.full(len(keys), "", dtype=object)
    for test, why in ((in_range, "population"), (between, "K=1 outside"), (precise, "error"),
                      (separated, "not separated"), (has_k2, "no K=2")):
        reason = np.where(~test, why, reason)  # the first failing test, in REASONS order
    two = has_k2 & separated & precise & between & in_range
    reason = np.where(two | ~np.isfinite(avg), "", reason)
    usable = has_k2 & between
    return PopulationChoice(
        sample=np.array([k[0] for k in keys]), time=np.array([k[1] for k in keys], dtype=float),
        residue=np.array([k[2] for k in keys], dtype=np.int64), fD_avg=avg, fD_avg_se=avg_se,
        fD_low=low, fD_low_se=low_se, fD_high=high, fD_high_se=high_se,
        population_low=np.where(usable, pop, np.nan),
        population_low_se=np.where(usable, pop_se, np.nan), two_populations=two,
        reason=reason.astype(str), config=config,
        control_fallback=np.array([fell_back[k] for k in keys], dtype=bool),
    )
