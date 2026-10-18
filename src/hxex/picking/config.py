# SPDX-License-Identifier: GPL-3.0-or-later
"""Peak-picker settings as an explicit, serialisable object."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field, fields
from typing import Any

from ..chem import DEFAULT_AMIDE_RULES, AmideRules, ChemistryError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PickerConfig:
    """Settings of :func:`hxex.picking.pick_spectra` (the ``[run.picker]`` table): how a
    spectrum becomes sticks at the predicted isotope positions. See docs/PEAK_PICKING.md.

    - ``resolution_ppm``: the half-width, in ppm, of the window around each predicted m/z in
      which a peak is picked (not a resolving power).
    - ``zero_filling``: how many bins are picked after the last one above the noise threshold.
    - ``noise_threshold``: an absolute intensity below which a point is noise; None (or 0) means
      ``noise_percent`` percent of the spectrum's tallest point instead.
    - ``env_threshold``: the fraction of the tallest bin at which the envelope width is measured.
    - ``nterm_subtract``: N-terminal residues without an exchanging amide; the amide count
      (with ``amide_rules`` for fragment ions) bounds the number of bins.
    - ``shoulder_min_points``: in a window of this many points or more, a maximum at the
      window's edge counts as a neighbour's shoulder and the bin is 0.
    - ``ladder``: how the bin positions are predicted. ``'composition'`` (the default) puts each
      bin at the abundance-weighted mass of its isotope cluster, the natural isotope clusters of
      the composition convolved with a binomial deuterium distribution whose level comes from a
      first pass with the ExMS ladder; ``'exms'`` uses fixed steps (1.003355 Da for the first
      two, 1.004816 for the next two, then 1.006277).
    - ``amide_rules``: per fragment ion type, whether it keeps the amide at its cleavage site
      (:class:`hxex.chem.AmideRules`).

    ``noise`` is the former name of ``noise_threshold`` and is still accepted.
    """

    resolution_ppm: float = 50.0
    zero_filling: int = 4
    noise_threshold: float | None = None
    noise_percent: float = 1.0
    env_threshold: float = 0.1
    nterm_subtract: int = 1
    shoulder_min_points: int = 10
    ladder: str = "composition"
    amide_rules: AmideRules = field(default=DEFAULT_AMIDE_RULES)
    noise: InitVar[float | None] = None  # the former name of noise_threshold

    def __post_init__(self, noise):
        if noise is not None:
            if self.noise_threshold is not None and self.noise_threshold != noise:
                raise ValueError("noise is the former name of noise_threshold: give one of them")
            object.__setattr__(self, "noise_threshold", noise)
        if not self.resolution_ppm > 0:
            raise ValueError("resolution_ppm must be positive")
        if self.zero_filling < 0 or int(self.zero_filling) != self.zero_filling:
            raise ValueError("zero_filling must be a non-negative integer")
        if self.noise_threshold is not None and not self.noise_threshold >= 0:
            raise ValueError("noise_threshold must be None or non-negative")
        if not self.noise_percent >= 0:
            raise ValueError("noise_percent must be non-negative")
        if not 0 < self.env_threshold <= 1:
            raise ValueError("env_threshold must be in (0, 1]")
        if self.nterm_subtract < 0 or int(self.nterm_subtract) != self.nterm_subtract:
            raise ValueError("nterm_subtract must be a non-negative integer")
        if self.shoulder_min_points < 1:
            raise ValueError("shoulder_min_points must be at least 1")
        if self.ladder not in ("exms", "composition"):
            raise ValueError("ladder must be 'exms' or 'composition'")

    def to_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["amide_rules"] = self.amide_rules.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PickerConfig:
        values = dict(data)
        if "noise" in values:  # the former name (bundles written before it was renamed)
            old = values.pop("noise")
            if old is not None and old is not False:
                log.info("picker: noise is the former name of noise_threshold")
                values["noise"] = old
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known - {"noise"}
        if unknown:
            raise ChemistryError(f"unknown picker setting(s): {', '.join(sorted(unknown))}")
        if "amide_rules" in values:
            values["amide_rules"] = AmideRules.from_dict(values["amide_rules"])
        return cls(**values)


DEFAULT_PICKER_CONFIG = PickerConfig()
