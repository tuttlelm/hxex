# SPDX-License-Identifier: GPL-3.0-or-later
"""Fitting settings as one explicit, serialisable object."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from ..model import Backend
from ..noise import NoiseConfig

CRITERIA = ("bic", "aic")


@dataclass(frozen=True)
class FitConfig:
    """Settings of :func:`hxex.fit.fit_models` (the ``[run.fit]`` table): how many populations
    are tried and how each fit is found.

    - ``k_min``, ``k_max``: the numbers of populations tried (1 to 3 by default; one K: no model
      selection).
    - ``criterion``: ``'bic'`` or ``'aic'``, chooses K among those tried.
    - ``min_population``: a K in which any population is below this share is rejected (0: off;
      with one K it is only reported). The analysis has its own limits
      (:class:`hxex.analysis.SpeciesFilters`, :class:`hxex.analysis.ChoiceConfig`).
    - ``n_restarts``: starting points per K. Restart 0 is the (K-1)-population fit with its
      heaviest population split in two (``nested_init``; else a shape-corrected k-means start),
      the others are random. With ``kmeans_after_warm`` a K that starts from the nested fit is
      also fitted from the k-means start, so a K among several is never worse than the same K
      fitted alone.
    - ``adam_steps``: Adam steps per restart, with a cosine learning rate from ``lr_adam`` to
      ``lr_adam_end``; ``k1_adam_steps``: the steps of the K=1 fit only (None: ``adam_steps``;
      one population rarely has more than one optimum, so 200 is enough).
    - ``lbfgs_steps``: L-BFGS iterations after Adam (0, the default: it did not improve the fits,
      and it minimises the summed loss, so measurements then influence each other).
    - ``split_delta``: the shift of the two halves of a split population, on the logit scale;
      ``warm_noise``: the sd of the random jitter added to a warm start.
    - ``polish_steps``: Levenberg-Marquardt iterations per measurement after the restarts (0:
      none), taking only steps that lower its loss.
    - ``grad_clip``: how the gradient norm is limited, ``'per_measurement'`` (each fit on its
      own) or ``'global'`` over the whole batch (one spectrum then changes the step of all the
      others).
    - ``seed``: everything random is drawn from generators seeded by it, and each measurement
      by what it is (``Measurements.random_id``: sample, peptide, charge, replicate, time, role),
      so with the defaults a measurement's fit depends only on its own data, its noise reference
      and the seed, not on the other measurements in the batch or the project's name.
    - ``screen``: fit each K above 1 only for the measurements that could still select it, from
      a lower bound on the loss of any mixture (:mod:`hxex.fit.screen`); about a third of the
      time. The selected K does not change, but the skipped fits are NaN, so fixed-K analyses
      (a fixed-K bootstrap, the analysis' K=2 descriptions) lack them. Needs ``k_min = 1``.
    - ``noise``: the noise model and level (:class:`hxex.noise.NoiseConfig`); ``backend``: the
      device and precision (:class:`hxex.model.Backend`).
    """
    # k1_adam_steps and the screen: docs/HISTORY.md items 85 and 86

    k_min: int = 1
    k_max: int = 3
    criterion: str = "bic"
    min_population: float = 0.05
    n_restarts: int = 6
    adam_steps: int = 400
    k1_adam_steps: int | None = None
    lbfgs_steps: int = 0
    lr_adam: float = 0.05
    lr_adam_end: float = 0.002
    nested_init: bool = True
    kmeans_after_warm: bool = True
    split_delta: float = 0.5
    warm_noise: float = 0.05
    grad_clip: str = "per_measurement"
    polish_steps: int = 40
    seed: int = 0
    screen: bool = False
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    backend: Backend = field(default_factory=Backend)

    def __post_init__(self):
        if self.k_min < 1 or self.k_max < self.k_min:
            raise ValueError("need 1 <= k_min <= k_max")
        if self.criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}")
        if not 0.0 <= self.min_population < 1.0:
            raise ValueError("min_population must be in [0, 1)")
        if self.n_restarts < 1:
            raise ValueError("n_restarts must be at least 1")
        if self.k1_adam_steps is not None and self.k1_adam_steps < 0:
            raise ValueError("k1_adam_steps must be non-negative")
        if self.adam_steps < 0 or self.lbfgs_steps < 0:
            raise ValueError("adam_steps and lbfgs_steps must be non-negative")
        if not self.lr_adam > 0 or not self.lr_adam_end > 0:
            raise ValueError("learning rates must be positive")
        if self.grad_clip not in ("global", "per_measurement"):
            raise ValueError("grad_clip must be 'global' or 'per_measurement'")
        if self.polish_steps < 0:
            raise ValueError("polish_steps must be non-negative")
        if self.warm_noise < 0:
            raise ValueError("warm_noise must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.screen and self.k_min != 1:
            raise ValueError("the screen needs k_min = 1 (it compares every K with the K=1 fit)")

    def adam_steps_for(self, k: int) -> int:
        """The Adam steps of the K-population fit."""
        if k == 1 and self.k1_adam_steps is not None:
            return self.k1_adam_steps
        return self.adam_steps

    @property
    def k_values(self) -> tuple[int, ...]:
        return tuple(range(self.k_min, self.k_max + 1))

    def to_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["noise"] = self.noise.to_dict()
        out["backend"] = self.backend.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FitConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown fit setting(s): {', '.join(sorted(unknown))}")
        values = dict(data)
        if "noise" in values:
            values["noise"] = NoiseConfig.from_dict(values["noise"])
        if "backend" in values:
            values["backend"] = Backend.from_dict(values["backend"])
        return cls(**values)
