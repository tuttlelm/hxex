# SPDX-License-Identifier: GPL-3.0-or-later
"""Exchangeable backbone amide count."""

from __future__ import annotations

from .ions import check_ion_type
from .modifications import DEFAULT_MODIFICATIONS, ModificationRegistry, parse_sequence
from .spec import ModificationSpec

# Side-chain exchangeable hydrogens counted when ``count_sc`` > 0, as in pyHXExpress.
_SC_ONE = frozenset("STYCDEHW")
_SC_TWO = frozenset("RKQN")


def count_amides(
    sequence: str,
    *,
    ion_type: str = "M",
    nterm_subtract: int = 1,
    count_sc: float = 0.0,
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
) -> int:
    """Number of exchanging backbone amides.

    Residues are tokenised (``acK`` is one residue). The count is the number of residues, minus
    ``nterm_subtract`` N-terminal residues, minus prolines after those, plus
    ``int(side_chain_hydrogens * count_sc)`` (off by default). A modified residue contributes
    its base residue's side chain.

    ``ion_type`` is validated but does not change the count yet: how the count depends on the
    ion type is an open question (docs/PLAN.md, "Ion types"), so every ion type currently uses
    the pyHXExpress rule.
    """
    check_ion_type(ion_type)
    residues = parse_sequence(sequence, registry)
    prolines = residues[nterm_subtract:].count("P")
    side_chain = 0
    for label in residues:
        base = registry.base_residue(label)
        side_chain += 1 if base in _SC_ONE else 2 if base in _SC_TWO else 0
    return max(0, len(residues) - prolines - nterm_subtract + int(side_chain * count_sc))


def exchangeable_sites(
    sequence: str,
    *,
    mod: ModificationSpec | None = None,
    ion_type: str = "M",
    nterm_subtract: int = 1,
    count_sc: float = 0.0,
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
) -> int:
    """Backbone amides (:func:`count_amides`) plus the user-specified ``mod.hex`` sites.

    Any ``namide_scale`` applied to the fit cap is the fitter's concern, not applied here.
    """
    n = count_amides(
        sequence,
        ion_type=ion_type,
        nterm_subtract=nterm_subtract,
        count_sc=count_sc,
        registry=registry,
    )
    return n + (mod.hex if mod is not None else 0)
