# SPDX-License-Identifier: GPL-3.0-or-later
"""Elemental composition, mass, m/z and natural-abundance shape from sequence and ion type."""

from __future__ import annotations

import numpy as np
from brainpy import isotopic_variants
from pyteomics import mass

from .amides import DEFAULT_AMIDE_RULES, AmideRules, exchangeable_sites
from .ions import check_ion_type, ion_terminus
from .modifications import (
    DEFAULT_MODIFICATIONS,
    TERMINAL_GROUPS,
    ChemistryError,
    ModificationRegistry,
    parse_sequence,
    split_termini,
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

    Terminal groups other than ``H-`` and ``-OH`` (such as the C-terminal amide ``-NH2``) replace
    the default ones. A fragment ion does not contain the far terminus, so a non-default group
    there (``PEPTIDE-NH2`` as a b ion, ``Ac-PEPTIDE`` as a y ion) is an error: write the fragment
    with the terminus it has, as :func:`hxex.fragments.fragment_entries` does. ``mod.composition``
    is added; ``mod.hex`` and ``mod.undeut_mz`` do not affect it.
    """
    check_ion_type(ion_type)
    residues = parse_sequence(sequence, registry)
    nterm, _, cterm = split_termini(sequence)
    end = ion_terminus(ion_type)
    if (end == "N" and cterm != "-OH") or (end == "C" and nterm != "H-"):
        far = cterm if end == "N" else nterm
        raise ChemistryError(
            f"{sequence!r} as a {ion_type} ion: the {'C' if end == 'N' else 'N'}-terminal group "
            f"{far!r} is not part of the fragment (write it with the terminus it has)"
        )
    comp = mass.Composition(
        parsed_sequence=["H-", *residues, "-OH"], aa_comp=registry.aa_comp, ion_type=ion_type
    )
    for group, default in ((nterm, "H-"), (cterm, "-OH")):
        if group != default:
            comp = (comp + mass.Composition(dict(TERMINAL_GROUPS[group]))
                    - mass.Composition(dict(TERMINAL_GROUPS[default])))
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
    return mass.Composition(comp).mass()


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
    # Composition.mass, not calculate_mass(composition=...), which fails in pyteomics 4.7
    return mass.Composition(comp).mass(charge=charge)


def isotope_clusters(
    sequence: str,
    ion_type: str = "M",
    *,
    mod: ModificationSpec | None = None,
    nterm_subtract: int = 1,
    npeaks: int | None = None,
    rules: AmideRules = DEFAULT_AMIDE_RULES,
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """The natural-abundance isotope clusters of :func:`natural_abundance` with their masses:
    ``(shape, offset)``, the abundance of each nominal-mass cluster (sums to 1) and its
    abundance-weighted mass relative to the first cluster (Da). Each cluster mixes isotopologues
    of one nominal mass (13C, 15N, 18O, 34S, ...), so the offsets are not multiples of one
    spacing (about 1.002 to 1.003 Da per step for peptides and proteins)."""
    comp = composition(sequence, ion_type, mod=mod, registry=registry)
    n_ex = exchangeable_sites(
        sequence, mod=mod, ion_type=ion_type, nterm_subtract=nterm_subtract, rules=rules,
        registry=registry,
    )
    if comp.get("H", 0) < n_ex:
        raise ChemistryError(f"{n_ex} exchangeable sites exceed the {comp.get('H', 0)} hydrogens")
    comp["H"] -= n_ex
    peaks = isotopic_variants(comp, npeaks=npeaks, charge=1)
    shape = np.array([p.intensity for p in peaks], dtype=float)
    masses = np.array([p.mz for p in peaks], dtype=float)  # charge 1: mass + one proton
    return shape / shape.sum(), masses - masses[0]


H_ATOM = 1.00782503207  # mass of a hydrogen atom (Da): the step between ion types such as c-1, c


def mixed_clusters(
    shape: np.ndarray, offset: np.ndarray, weights, step: float = H_ATOM
) -> tuple[np.ndarray, np.ndarray]:
    """The isotope clusters of a mixture of ion types that differ by one hydrogen each (for
    example ``c-1``, ``c`` and ``c+1``): component ``i`` (weight ``weights[i]``) is the lightest
    type plus ``i`` hydrogens, so its cluster ``j`` falls in the mixture's cluster ``i + j``, at
    ``i * step + offset[j]``. Returns ``(shape, offset)`` like :func:`isotope_clusters`: the
    abundance of each mixed cluster (sums to 1) and its abundance-weighted mass offset from the
    lightest type's monoisotopic mass."""
    shape = np.asarray(shape, dtype=float)
    offset = np.asarray(offset, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) == 0 or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be finite and non-negative with a positive sum")
    w = w / w.sum()
    n = len(shape) + len(w) - 1
    abundance = np.zeros(n)
    mass = np.zeros(n)
    for i, wi in enumerate(w):
        abundance[i:i + len(shape)] += wi * shape
        mass[i:i + len(shape)] += wi * shape * (i * step + offset)
    out = np.full(n, np.nan)
    has = abundance > 0
    out[has] = mass[has] / abundance[has]
    # a cluster no component reaches (a zero weight at the end): one step past the previous one
    for k in np.flatnonzero(~has):
        out[k] = out[k - 1] + step if k > 0 else 0.0
    return abundance / abundance.sum(), out


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
