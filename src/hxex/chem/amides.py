# SPDX-License-Identifier: GPL-3.0-or-later
"""Exchangeable backbone amides, of peptides and of fragment ions.

An amide is counted by its position in the parent (the peptide or protein that was labelled):
residue ``i`` contributes its backbone N-H unless it is a proline or one of the parent's first
``nterm_subtract`` residues (whose N-H is the N-terminal amine, or back-exchanges too fast to be
kept). A fragment ion carries the N-H of the residues it contains, except at the cleavage site:

- N-terminal ions of ``L`` residues hold residues ``1..L``; a c ion also holds the N-H of
  residue ``L + 1`` (the N-Calpha bond is cut, so the next residue's amide nitrogen goes with
  it: c = b + NH3);
- C-terminal ions hold residues ``n - L + 1..n``; x and y ions also hold their first residue's
  N-H (they are cut at or before the peptide bond), a z ion does not (z = y - NH3);
- parent-type ions (``M``, ``M-H2O``, ``M-NH3``) are the whole parent.

These are the backbone nitrogens of each ion type's composition (pyteomics: L + 1 for the c
ions, L for a, b, x and y, L - 1 for the z ions), so the count follows from the formula. The
neutral-loss variants (``-H2O``, ``-NH3``) follow their base ion, since a lost NH3 mostly comes
from a side chain (K, R, N, Q) or the N-terminal amine, whose hydrogens are not counted; except
``c-NH3``, whose NH3 is taken to be the cleavage-site NH2 (it is then a b ion). These are
UNVERIFIED (docs/PLAN.md, "Risks and open decisions"); :class:`AmideRules` sets any ion type's
cleavage-site amide. In ``c-1`` (the c radical, which gave a hydrogen atom to its z partner) the
hydrogen moved is not an amide's (HISTORY item 66). b ions are counted as linear (L - 1 from
residue 2); an oxazolone b ion has replaced residue L's N-H with the mobile proton (L - 2,
``AmideRules({"b": False})``), and collisional b and y ions are scrambled anyway.

Consecutive fragments therefore differ by one residue's amide, at the cleavage site: c_k - c_(k-1)
is the amide of residue ``k + 1``, b_k - b_(k-1) and a_k - a_(k-1) that of residue ``k``, and for
C-terminal ions of the parent's last residues, y_j - y_(j-1) that of the y_j's first residue and
z_j - z_(j-1) that of the residue after the z_j's first (:func:`amide_positions`,
:func:`hxex.fragments.amide_difference`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .ions import check_ion_type, ion_terminus
from .modifications import (
    DEFAULT_MODIFICATIONS,
    ChemistryError,
    ModificationRegistry,
    parse_sequence,
)
from .spec import ModificationSpec

# Side-chain exchangeable hydrogens counted when ``count_sc`` > 0, as in pyHXExpress.
_SC_ONE = frozenset("STYCDEHW")
_SC_TWO = frozenset("RKQN")

# whether an ion type holds the N-H at its cleavage site, by the first letter of its name (the
# backbone nitrogens of its composition), and the ion types that differ from their letter
_BOUNDARY = {"a": False, "b": False, "c": True, "x": True, "y": True, "z": False}
_BOUNDARY_BY_TYPE = {"c-NH3": False}  # the cleavage-site NH2 lost: a b ion


@dataclass(frozen=True)
class AmideRules:
    """Per ion type, whether the fragment holds the amide N-H at its cleavage site (the
    "boundary" amide): for an N-terminal ion the N-H of the residue after it, for a C-terminal
    ion the N-H of its own first residue. The default follows the ion type's composition (c, x
    and y: yes; a, b and z: no; variants such as ``c-1`` or ``z-dot`` as their letter, except
    ``c-NH3``: no); entries here replace it, e.g. ``AmideRules.from_dict({"y-NH3": False})``.
    Parent-type ions have no boundary."""

    boundary: Mapping[str, bool] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        for ion_type, value in self.boundary.items():
            if ion_terminus(ion_type) == "M":
                raise ChemistryError(f"{ion_type!r} is not a fragment ion type")
            if not isinstance(value, bool):
                raise ChemistryError(f"the boundary amide of {ion_type!r} must be true or false, "
                                     f"got {value!r}")
        object.__setattr__(self, "boundary", MappingProxyType(dict(self.boundary)))

    def keeps_boundary(self, ion_type: str) -> bool:
        """Whether ``ion_type`` holds the N-H at its cleavage site (False for parent-type ions)."""
        if ion_terminus(ion_type) == "M":
            return False
        return self.boundary.get(ion_type,
                                 _BOUNDARY_BY_TYPE.get(ion_type, _BOUNDARY[ion_type[0]]))

    def with_boundary(self, ion_type: str, keeps: bool) -> AmideRules:
        """Return a copy with ``ion_type``'s boundary amide set to ``keeps``."""
        return AmideRules({**self.boundary, check_ion_type(ion_type): keeps})

    def to_dict(self) -> dict[str, bool]:
        return dict(self.boundary)

    @classmethod
    def from_dict(cls, data: Mapping[str, bool]) -> AmideRules:
        return cls(dict(data))


DEFAULT_AMIDE_RULES = AmideRules()


def _exchanges(label: str, position: int, nterm_subtract: int,
               registry: ModificationRegistry) -> bool:
    """Whether the residue ``label`` at ``position`` (1-based) in the parent has an exchanging
    backbone N-H."""
    return position > nterm_subtract and registry.base_residue(label) != "P"


def amide_positions(
    parent: str,
    ion_type: str = "M",
    length: int | None = None,
    *,
    nterm_subtract: int = 1,
    rules: AmideRules = DEFAULT_AMIDE_RULES,
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
) -> tuple[int, ...]:
    """The positions in ``parent`` (1-based residues) of the exchanging backbone amides that the
    ion ``ion_type`` of ``length`` residues holds (see the module docstring); ``length`` is the
    whole parent for parent-type ions. A fragment is 1 to ``n - 1`` residues long (``n`` would
    be the parent)."""
    residues = parse_sequence(parent, registry)
    n = len(residues)
    side = ion_terminus(ion_type)
    if side == "M":
        candidates = range(1, n + 1)
    else:
        if length is None or not 1 <= length < n:
            raise ChemistryError(f"a {ion_type} fragment of {parent} ({n} residues) is 1 to "
                                 f"{n - 1} residues long, got {length}")
        boundary = rules.keeps_boundary(ion_type)
        if side == "N":
            candidates = range(1, length + 1 + (1 if boundary else 0))
        else:
            first = n - length + 1
            candidates = range(first if boundary else first + 1, n + 1)
    return tuple(i for i in candidates
                 if _exchanges(residues[i - 1], i, nterm_subtract, registry))


def count_amides(
    sequence: str,
    *,
    ion_type: str = "M",
    nterm_subtract: int = 1,
    count_sc: float = 0.0,
    rules: AmideRules = DEFAULT_AMIDE_RULES,
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
    parent_pos: int | None = None,
) -> int:
    """Number of exchanging backbone amides of the ion ``ion_type`` whose residues are
    ``sequence``, as :func:`amide_positions` places them in the parent.

    Residues are tokenised (``acK`` is one residue). For a peptide (``M``) and an N-terminal
    ion, the sequence starts the parent: its first ``nterm_subtract`` residues and its prolines
    do not count. A c ion also counts the next residue's N-H (not part of ``sequence``; it is
    taken not to be a proline, at which electron-based dissociation does not cleave, so
    :func:`hxex.fragments.fragment_entries` leaves such c ions out) unless that residue is
    among the first ``nterm_subtract``. For a C-terminal ion, ``parent_pos`` is the position in
    the parent of its first residue: only residues of the parent's first ``nterm_subtract``
    are left out (None: the fragment is taken not to reach them). Then ``int(side_chain_hydrogens
    * count_sc)`` is added (off by default; a modified residue contributes its base residue's
    side chain).
    """
    residues = parse_sequence(sequence, registry)
    length = len(residues)
    side = ion_terminus(ion_type)
    if side == "C":
        start = parent_pos if parent_pos is not None else nterm_subtract + 1
        first = 0 if rules.keeps_boundary(ion_type) else 1
        n = sum(_exchanges(residues[i], start + i, nterm_subtract, registry)
                for i in range(first, length))
    else:
        if parent_pos not in (None, 1):
            raise ChemistryError(f"an {ion_type} ion starts its parent (parent_pos 1), got "
                                 f"{parent_pos}")
        n = sum(_exchanges(residues[i], i + 1, nterm_subtract, registry) for i in range(length))
        if rules.keeps_boundary(ion_type) and length + 1 > nterm_subtract:
            n += 1  # the next residue's N-H
    side_chain = 0
    for label in residues:
        base = registry.base_residue(label)
        side_chain += 1 if base in _SC_ONE else 2 if base in _SC_TWO else 0
    return max(0, n + int(side_chain * count_sc))


def exchangeable_sites(
    sequence: str,
    *,
    mod: ModificationSpec | None = None,
    ion_type: str = "M",
    nterm_subtract: int = 1,
    count_sc: float = 0.0,
    rules: AmideRules = DEFAULT_AMIDE_RULES,
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
) -> int:
    """Backbone amides (:func:`count_amides`, placed by ``mod.parent_pos``) plus the
    user-specified ``mod.hex`` sites.

    Any ``namide_scale`` applied to the fit cap is the fitter's concern, not applied here.
    """
    n = count_amides(
        sequence,
        ion_type=ion_type,
        nterm_subtract=nterm_subtract,
        count_sc=count_sc,
        rules=rules,
        registry=registry,
        parent_pos=mod.parent_pos if mod is not None else None,
    )
    return n + (mod.hex if mod is not None else 0)
