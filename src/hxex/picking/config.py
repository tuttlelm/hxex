# SPDX-License-Identifier: GPL-3.0-or-later
"""Peak-picker settings as an explicit, serialisable object."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from ..chem import DEFAULT_AMIDE_RULES, AmideRules, ChemistryError


@dataclass(frozen=True)
class PickerConfig:
    """Settings of :func:`hxex.picking.pick_spectra`; defaults are pyHXExpress's.

    ``noise`` is an absolute intensity threshold; if it is None (or 0, as in pyHXExpress's
    ``setNoise``) the threshold is ``noise_percent`` percent of the spectrum's maximum.
    ``shoulder_min_points``: a window with fewer points is treated as stick data (no shoulder
    rejection). ``amide_rules`` and ``nterm_subtract`` set the amide count that bounds the
    number of bins.
    """

    resolution_ppm: float = 50.0
    zero_filling: int = 4
    noise: float | None = None
    noise_percent: float = 1.0
    env_threshold: float = 0.1
    nterm_subtract: int = 1
    shoulder_min_points: int = 10
    amide_rules: AmideRules = field(default=DEFAULT_AMIDE_RULES)

    def __post_init__(self):
        if not self.resolution_ppm > 0:
            raise ValueError("resolution_ppm must be positive")
        if self.zero_filling < 0 or int(self.zero_filling) != self.zero_filling:
            raise ValueError("zero_filling must be a non-negative integer")
        if self.noise is not None and not self.noise >= 0:
            raise ValueError("noise must be None or non-negative")
        if not self.noise_percent >= 0:
            raise ValueError("noise_percent must be non-negative")
        if not 0 < self.env_threshold <= 1:
            raise ValueError("env_threshold must be in (0, 1]")
        if self.nterm_subtract < 0 or int(self.nterm_subtract) != self.nterm_subtract:
            raise ValueError("nterm_subtract must be a non-negative integer")
        if self.shoulder_min_points < 1:
            raise ValueError("shoulder_min_points must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["amide_rules"] = self.amide_rules.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PickerConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ChemistryError(f"unknown picker setting(s): {', '.join(sorted(unknown))}")
        values = dict(data)
        if "amide_rules" in values:
            values["amide_rules"] = AmideRules.from_dict(values["amide_rules"])
        return cls(**values)


DEFAULT_PICKER_CONFIG = PickerConfig()
