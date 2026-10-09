# SPDX-License-Identifier: GPL-3.0-or-later
"""Elemental composition, mass, m/z and natural-abundance shape from sequence and ion type."""

from __future__ import annotations

import numpy as np
from brainpy import isotopic_variants
from pyteomics import mass

from .amides import DEFAULT_AMIDE_RULES, AmideRules, exchangeable_sites
from .ions import check_ion_type
from .modifications import (
    DEFAULT_MODIFICATIONS,
    ChemistryError,
    ModificationRegistry,
    parse_sequence,
)
from .spec import ModificationSpec


def composition(
    sequence: str,
    ion_type: str = "M",
    *,
    mod: ModificationSpec | None = None,
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
) -> dict[str, int]:
    """Neutral elemental composition of the ion (pyteomics is the reference).

    ``mod.composition`` is added; ``mod.hex`` and ``mod.undeut_mz`` do not affect it.
    """
    check_ion_type(ion_type)
    residues = parse_sequence(sequence, registry)
    comp = mass.Composition(
        parsed_sequence=["H-", *residues, "-OH"], aa_comp=registry.aa_comp, ion_type=ion_type
    )
    if mod is not None:
        comp = comp + mass.Composition(dict(mod.composition))
    return {k: v for k, v in comp.items() if v != 0}


def neutral_mass(
    sequence: str,
    ion_type: str = "M",
    *,
    mod: ModificationSpec | None = None,
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
) -> float:
    """Monoisotopic neutral mass of :func:`composition`."""
    comp = composition(sequence, ion_type, mod=mod, registry=registry)
    return mass.calculate_mass(composition=comp)


def mz(
    sequence: str,
    charge: int,
    ion_type: str = "M",
    *,
    mod: ModificationSpec | None = None,
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
) -> float:
    """Undeuterated monoisotopic m/z, ``(mass + charge * proton) / charge``.

    If ``mod.undeut_mz`` is set it is returned unchanged.
    """
    if charge < 1:
        raise ChemistryError(f"charge must be a positive integer, got {charge}")
    if mod is not None and mod.undeut_mz is not None:
        return mod.undeut_mz
    comp = composition(sequence, ion_type, mod=mod, registry=registry)
    return mass.calculate_mass(composition=comp, charge=charge)


def natural_abundance(
    sequence: str,
    ion_type: str = "M",
    *,
    mod: ModificationSpec | None = None,
    nterm_subtract: int = 1,
    npeaks: int | None = None,
    rules: AmideRules = DEFAULT_AMIDE_RULES,
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
) -> np.ndarray:
    """Natural-abundance isotope shape (sums to 1) with the exchangeable hydrogens removed.

    The composition follows the ion type, so the shape is consistent with :func:`mz`. The
    hydrogens removed are the backbone amides (:func:`count_amides`) plus ``mod.hex``. With
    ``npeaks=None`` the number of peaks depends on the composition.
    """
    comp = composition(sequence, ion_type, mod=mod, registry=registry)
    n_ex = exchangeable_sites(
        sequence,
        mod=mod,
        ion_type=ion_type,
        nterm_subtract=nterm_subtract,
        rules=rules,
        registry=registry,
    )
    if comp.get("H", 0) < n_ex:
        raise ChemistryError(f"{n_ex} exchangeable sites exceed the {comp.get('H', 0)} hydrogens")
    comp["H"] -= n_ex
    peaks = isotopic_variants(comp, npeaks=npeaks, charge=1)
    shape = np.array([p.intensity for p in peaks])
    return shape / shape.sum()
