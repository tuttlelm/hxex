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
    """Settings of :func:`hxex.fit.fit_models`; defaults are the prototype's except where noted.

    ``k_min`` to ``k_max``: the numbers of populations to try (1 to 3, 2 to 3, only 1, only 2).
    With one K there is no model selection. ``criterion`` selects K among those tried;
    ``min_population`` rejects a K in which any component's population is below it (0 turns the
    filter off; with one K it is only reported).

    Each K is fitted from ``n_restarts`` starting points: restart 0 is the (K-1)-population
    solution with its heaviest component split in two when ``nested_init`` is set and K-1 was
    fitted (otherwise a shape-corrected k-means start), the others are random. With
    ``kmeans_after_warm`` (the default) a K that has a nested start is also fitted from the
    k-means start, as one restart more, so that a K in a run of several is never worse than the
    same K fitted alone. Adam runs
    ``adam_steps`` steps with a cosine learning rate from ``lr_adam`` to ``lr_adam_end``, then
    L-BFGS runs up to ``lbfgs_steps`` iterations (0, the default: it did not improve the fits, and
    it minimises the summed loss, so it makes measurements influence each other).
    ``split_delta`` is the shift of the two halves on the logit scale and ``warm_noise`` the sd
    of the random jitter added to a warm start.
    After the restarts, ``polish_steps`` Levenberg-Marquardt iterations (0 turns it off) polish
    every measurement, taking only steps that lower its loss; it works per measurement.
    ``grad_clip`` is how the gradient norm is limited: ``'per_measurement'`` (default) or
    ``'global'`` over the whole batch (the prototype; one spectrum then changes the step of all
    the others). With the defaults a measurement's fit depends only on its own data, its noise
    reference and the seed, not on the other measurements in the batch.
    Everything random is drawn from generators seeded by ``seed``.
    """

    k_min: int = 1
    k_max: int = 3
    criterion: str = "bic"
    min_population: float = 0.05
    n_restarts: int = 6
    adam_steps: int = 400
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
