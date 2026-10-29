# SPDX-License-Identifier: GPL-3.0-or-later
"""The choice between one and two populations, residue by residue (docs/ANALYSIS.md, phase 2).

Each residue has three values from the fixed-K fits (:func:`hxex.analysis.residue_map`): the K=1
value ``fD_avg`` and the K=2 ranks ``fD_low`` and ``fD_high``, each with a standard error. Many
peptides report on a residue, so whether its data support two populations is judged there rather
than spectrum by spectrum. The two-population
description is kept when every test passes:

- the populations are separated: ``(fD_high - fD_low) - z (se_low + se_high) >= 0``, with ``z``
  for the ``confidence`` interval (50 % by default);
- both standard errors are below ``max_fD_se``;

An unknown standard error (NaN: no bootstrap and a single spectrum behind the residue) is
treated as unknown, not as 0: the residue counts as neither separated nor precise, and the K=1
value describes it (``reason`` ``'not separated'``).
- the K=1 value lies between them, ``fD_low < fD_avg < fD_high`` (the same molecules);
- the lever-rule population of the low-D component, ``(fD_high - fD_avg) / (fD_high - fD_low)``,
  is within ``[min_population, 1 - min_population]``.

Otherwise the K=1 value describes the residue. The standard error of the population is propagated
from all three values.

When the residue map was made from the two-population peptides
(``residue_map(two_population_peptides=True)``; owner, 2026-10-21, HISTORY 137), one more test
comes first: the peptides whose set is best described by two populations carry at least half of
the residue's weight (``two_population_weight >= 0.5``); otherwise ``reason`` is
``'one-population peptides'``.

With the population test (``tiers``; ``[analysis] populations = "population_test"``; owner,
2026-10-26, HISTORY 177) the residue's tier decides instead
(:func:`hxex.analysis.residues.population_tiers`): two populations where the tier is "two" or
above (``ChoiceConfig.tier = 'two'``, the default: strong evidence and a third of the covering
weight) or "two, localised" or above (``'localised'``: half of it), and the residue has K=2
values to draw. The tests above do not apply (the bootstrap gives errors only, never the
decision); ``reason`` then names the tier that did not reach it (``'tier: undecided'``,
``'tier: no sign'``, ``'tier: two'``) or ``'no K=2'``. A residue in a "three" tier is drawn
with its K=2 values (``reason`` ``'tier: three, drawn as two'``).
"""
# the method follows pyHXExpress's choose_fits, per residue instead of per spectrum; its
# population error left a derivative unsquared

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from scipy.stats import norm

from .residues import TIERS, ResidueMap

TIER_LEVELS = ("two", "localised")  # ChoiceConfig.tier: the least tier that is two populations
REASONS = ("", "no K=2", "one-population peptides", "not separated", "error", "K=1 outside",
           "population")
TWO_POPULATION_SHARE = 0.5  # the weight two-population peptides must carry (HISTORY 137)


@dataclass(frozen=True)
class ChoiceConfig:
    """Limits of :func:`choose_populations` (the ``[analysis.choice]`` table; see the module
    docstring).

    - ``confidence``: a fraction (0.5: 50 %), the level of the interval that must separate the
      two K=2 values for them to count as separated. (The bootstrap's ``ci`` is a percentage.)
    - ``max_fD_se``: both K=2 standard errors must be below this.
    - ``min_population``: the lever-rule population of the low-D component must be within
      ``[min_population, 1 - min_population]`` (the fit and the filters have limits of the same
      name on each spectrum's populations).

    A residue whose K=2 standard errors are unknown (NaN) fails the separation and precision
    tests: the one-population description is kept.
    - ``tier``: with the population test, the least residue tier that has two populations:
      ``'two'`` (a third of the covering weight agrees, the default) or ``'localised'`` (half
      of it); the three limits above do not apply then.
    """

    confidence: float = 0.5
    max_fD_se: float = 0.25
    min_population: float = 0.05
    tier: str = "two"

    def __post_init__(self):
        if not 0 <= self.confidence < 1:
            raise ValueError("confidence must be in [0, 1)")
        if self.max_fD_se <= 0:
            raise ValueError("max_fD_se must be positive")
        if not 0 <= self.min_population < 0.5:
            raise ValueError("min_population must be in [0, 0.5)")
        if self.tier not in TIER_LEVELS:
            raise ValueError(f"tier must be one of {', '.join(TIER_LEVELS)}")

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
    """One row per run, sample, time and residue (columns of equal length): ``fD_avg`` (K=1),
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
    run: np.ndarray | None = None

    def __post_init__(self):  # built without it: no fallback, one run
        if self.control_fallback is None:
            object.__setattr__(self, "control_fallback", np.zeros(len(self.sample), dtype=bool))
        if self.run is None:
            object.__setattr__(self, "run", np.full(len(self.sample), "", dtype=object))
        else:
            object.__setattr__(self, "run", np.asarray(self.run, dtype=object))

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return {"run": self.run, **{f.name: getattr(self, f.name) for f in fields(self)
                                    if f.name not in ("config", "run")}}

    def where(self, *, sample=None, time=None, run=None) -> np.ndarray:
        keep = np.ones(len(self), dtype=bool)
        for name, value in (("run", run), ("sample", sample), ("time", time)):
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


def choose_populations(residues: ResidueMap, config: ChoiceConfig = ChoiceConfig(),  # noqa: B008
                       *, tiers: bool | None = None) -> PopulationChoice:
    """The choice for every sample, time and residue of ``residues`` that has a K=1 value or
    K=2 values (see the module docstring); with ``tiers``, by the residue tiers of the
    population test (the residue map made with ``population_test=True``). ``tiers`` None: by
    the tiers whenever the residue map has them."""
    if tiers is None:
        tiers = residues.population_tier is not None and any(
            str(t) for t in residues.population_tier)
    table: dict[tuple, dict] = {}
    fell_back: dict[tuple, bool] = {}
    share: dict[tuple, float] = {}
    tier_of: dict[tuple, str] = {}
    weights = np.asarray(residues.two_population_weight, dtype=float)
    voted = bool(np.isfinite(weights).any())  # made from the two-population peptides
    for i in range(len(residues)):
        k, rank = int(residues.K[i]), int(residues.rank[i])
        if (k, rank) not in ((1, 1), (2, 1), (2, 2)):
            continue
        key = (str(residues.run[i]), str(residues.sample[i]), float(residues.time[i]),
               int(residues.residue[i]))
        name = {(1, 1): "avg", (2, 1): "low", (2, 2): "high"}[(k, rank)]
        table.setdefault(key, {})[name] = (float(residues.fD[i]), float(residues.fD_se[i]))
        if name == "avg":
            share[key] = float(weights[i])
            tier_of[key] = str(residues.population_tier[i])
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
        known = np.isfinite(low_se) & np.isfinite(high_se)  # an unknown error passes no test
        separated = known & ((high - low) - config.z * (low_se + high_se) >= 0)
        precise = known & (low_se < config.max_fD_se) & (high_se < config.max_fD_se)
        between = (low < avg) & (avg < high)
        in_range = (pop >= lo) & (pop <= 1 - lo)
        support = np.array([share.get(key, np.nan) for key in keys], dtype=float)
        enough = (support >= TWO_POPULATION_SHARE) if voted else np.ones(len(keys), dtype=bool)
    reason = np.full(len(keys), "", dtype=object)
    for test, why in ((in_range, "population"), (between, "K=1 outside"), (precise, "error"),
                      (separated, "not separated"), (enough, "one-population peptides"),
                      (has_k2, "no K=2")):
        reason = np.where(~test, why, reason)  # the first failing test, in REASONS order
    two = has_k2 & enough & separated & precise & between & in_range
    if tiers:  # the population test decides; the bootstrap gives the errors only
        tier = np.array([tier_of.get(key, "") for key in keys], dtype=object)
        least = 2 if config.tier == "two" else 3  # in TIERS order
        rank = np.array([TIERS.index(t) if t in TIERS else -1 for t in tier])
        rank = np.where(rank >= 4, rank - 2, rank)  # three, three localised: as two, localised
        reached = rank >= least
        two = has_k2 & reached
        named = np.array(["tier: " + str(t) for t in tier], dtype=str)  # numpy 1.26: no string +
        reason = np.where(~has_k2, "no K=2", np.where(reached, "", named))
        reason = np.where(np.isin(tier, ("three", "three, localised")) & two,
                          "tier: three, drawn as two", reason)
        reason = np.where(tier == "", np.where(has_k2, "tier: none", "no K=2"), reason)
    reason = np.where((two & (reason != "tier: three, drawn as two")) | ~np.isfinite(avg), "",
                      reason)
    usable = has_k2 & between
    return PopulationChoice(
        run=np.array([k[0] for k in keys], dtype=object), sample=np.array([k[1] for k in keys]),
        time=np.array([k[2] for k in keys], dtype=float),
        residue=np.array([k[3] for k in keys], dtype=np.int64), fD_avg=avg, fD_avg_se=avg_se,
        fD_low=low, fD_low_se=low_se, fD_high=high, fD_high_se=high_se,
        population_low=np.where(usable, pop, np.nan),
        population_low_se=np.where(usable, pop_se, np.nan), two_populations=two,
        reason=reason.astype(str), config=config,
        control_fallback=np.array([fell_back[k] for k in keys], dtype=bool),
    )
