# SPDX-License-Identifier: GPL-3.0-or-later
"""Peak-picker settings as an explicit, serialisable object."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field, fields
from typing import Any

import numpy as np

from ..chem import DEFAULT_AMIDE_RULES, AmideRules, ChemistryError

log = logging.getLogger(__name__)

RECORDED = "recorded"  # resolving_power: take each spectrum's value from the scan files
ISOTOPE_STEP = 1.00235  # Da, the average spacing of a peptide's isotope peaks

# Thermo quotes the resolving power at m/z 400 on the first Orbitrap generation and at m/z 200
# since the Q Exactive and the Tribrids; the picker's own reference m/z is 200
_REFERENCE_MZ_400 = ("elite", "velos", "orbitrap xl", "discovery", "ltq orbitrap")


def reference_mz(model: str | None) -> float:
    """The m/z at which a Thermo instrument ``model`` quotes its resolving power (200, or 400
    for the first Orbitrap generation); NaN for an unknown or empty model."""
    text = (model or "").strip().lower()
    if not text:
        return float("nan")
    if any(name in text for name in _REFERENCE_MZ_400) and "exploris" not in text:
        return 400.0
    return 200.0


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

    The window can instead follow the instrument's resolving power (:meth:`half_width`):

    - ``resolving_power``: R quoted at ``resolving_power_mz`` (200 by default, Thermo's
      convention since the Q Exactive), or ``'recorded'`` to take each spectrum's value from
      the scan files (Thermo raw files record it; a spectrum without one keeps the
      ``resolution_ppm`` window). None (the default): the ``resolution_ppm`` window.
    - ``resolving_power_exponent``: how R falls with m/z, ``R(m) = R0 (m0 / m) ** e``: 0.5 for
      an Orbitrap (the default), 0 for a TOF, 1 for an FT-ICR.
    - ``window_fwhm``: the half-width as a multiple of the peak's full width at half maximum,
      ``m / R(m)``.
    - ``mass_accuracy_ppm``: the half-width is never below this (the calibration error a very
      high R would otherwise fall under).
    - The half-width is never above half the isotope spacing, ``1.00235 / (2 z)``, so the
      windows of neighbouring bins do not overlap (the ``resolution_ppm`` window has no such
      cap).

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
    resolving_power: float | str | None = None
    resolving_power_mz: float = 200.0
    resolving_power_exponent: float = 0.5
    window_fwhm: float = 1.0
    mass_accuracy_ppm: float = 10.0
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
        r = self.resolving_power
        if isinstance(r, str):
            if r != RECORDED:
                raise ValueError(f"resolving_power must be a number or '{RECORDED}', not {r!r}")
        elif r is not None and (isinstance(r, bool) or not r > 0):
            raise ValueError("resolving_power must be positive")
        if not self.resolving_power_mz > 0:
            raise ValueError("resolving_power_mz must be positive")
        if self.resolving_power_exponent < 0:
            raise ValueError("resolving_power_exponent must be non-negative")
        if not self.window_fwhm > 0:
            raise ValueError("window_fwhm must be positive")
        if self.mass_accuracy_ppm < 0:
            raise ValueError("mass_accuracy_ppm must be non-negative")

    @property
    def uses_recorded(self) -> bool:
        """Whether the window takes each spectrum's recorded resolving power."""
        return self.resolving_power == RECORDED

    def half_width(self, mz, charge: int, *, resolving_power: float | None = None,
                   resolving_power_mz: float | None = None) -> np.ndarray:
        """The half-width, in m/z, of the picking window at each predicted ``mz`` of an ion of
        ``charge``: ``mz * resolution_ppm / 1e6`` by default; with ``resolving_power`` set, the
        larger of ``window_fwhm`` peak widths and ``mass_accuracy_ppm``, capped at half the
        isotope spacing. ``resolving_power`` and ``resolving_power_mz`` here are a spectrum's
        recorded values, used with ``'recorded'`` (NaN or None: the ``resolution_ppm`` window;
        a NaN reference m/z: the setting's)."""
        mz = np.asarray(mz, dtype=np.float64)
        r = self.resolving_power
        r_mz = self.resolving_power_mz
        if r == RECORDED:
            r = (float(resolving_power) if resolving_power is not None
                 and np.isfinite(resolving_power) and resolving_power > 0 else None)
            if resolving_power_mz is not None and np.isfinite(resolving_power_mz):
                r_mz = float(resolving_power_mz)
        if r is None:
            return mz * self.resolution_ppm / 1e6
        r_at = float(r) * (r_mz / mz) ** self.resolving_power_exponent
        width = np.maximum(self.window_fwhm * mz / r_at, mz * self.mass_accuracy_ppm / 1e6)
        return np.minimum(width, ISOTOPE_STEP / (2.0 * max(int(charge), 1)))

    def window_edges(self, mz, charge: int, **recorded) -> tuple[np.ndarray, np.ndarray]:
        """The lower and upper edges of the picking window at each predicted ``mz``
        (:meth:`half_width` either side; the ``resolution_ppm`` window is ``mz * (1 -/+ ppm /
        1e6)`` exactly, as pyHXExpress computes it)."""
        mz = np.asarray(mz, dtype=np.float64)
        half = self.half_width(mz, charge, **recorded)
        if self.resolving_power is None or (self.uses_recorded and not (
                recorded.get("resolving_power") is not None
                and np.isfinite(recorded["resolving_power"])
                and recorded["resolving_power"] > 0)):
            factors = 1.0 + np.array([-1.0, 1.0]) * self.resolution_ppm / 1e6
            return mz * factors[0], mz * factors[1]
        return mz - half, mz + half

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
