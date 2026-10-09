# SPDX-License-Identifier: GPL-3.0-or-later
"""Exchangeable backbone amide count."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .ions import check_ion_type
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


@dataclass(frozen=True)
class AmideRules:
    """Per-ion-type offset added to the amide count (default 0 for every ion type).

    How the exchangeable amide count depends on ion type is an open question (docs/PLAN.md,
    "Ion types"), so every ion type starts at the pyHXExpress rule (offset 0) and offsets are
    set explicitly, e.g. ``AmideRules.from_dict({"c": 1})``.

    UNVERIFIED for fragment ions other than M, b and y: no offsets have been validated yet.
    """

    offsets: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        for ion_type, offset in self.offsets.items():
            check_ion_type(ion_type)
            if isinstance(offset, bool) or not isinstance(offset, int):
                raise ChemistryError(f"offset for {ion_type!r} must be an integer, got {offset!r}")
        object.__setattr__(self, "offsets", MappingProxyType(dict(self.offsets)))

    def offset(self, ion_type: str) -> int:
        return self.offsets.get(check_ion_type(ion_type), 0)

    def with_offset(self, ion_type: str, offset: int) -> AmideRules:
        """Return a copy with ``ion_type`` set to ``offset``."""
        return AmideRules({**self.offsets, ion_type: offset})

    def to_dict(self) -> dict[str, int]:
        return dict(self.offsets)

    @classmethod
    def from_dict(cls, data: Mapping[str, int]) -> AmideRules:
        return cls(dict(data))


DEFAULT_AMIDE_RULES = AmideRules()


def count_amides(
    sequence: str,
    *,
    ion_type: str = "M",
    nterm_subtract: int = 1,
    count_sc: float = 0.0,
    rules: AmideRules = DEFAULT_AMIDE_RULES,
    registry: ModificationRegistry = DEFAULT_MODIFICATIONS,
) -> int:
    """Number of exchanging backbone amides.

    Residues are tokenised (``acK`` is one residue). The count is the number of residues, minus
    ``nterm_subtract`` N-terminal residues, minus prolines after those, plus
    ``int(side_chain_hydrogens * count_sc)`` (off by default). A modified residue contributes
    its base residue's side chain. ``rules.offset(ion_type)`` is added last (0 unless set, see
    :class:`AmideRules`); the result is never negative.
    """
    residues = parse_sequence(sequence, registry)
    prolines = residues[nterm_subtract:].count("P")
    side_chain = 0
    for label in residues:
        base = registry.base_residue(label)
        side_chain += 1 if base in _SC_ONE else 2 if base in _SC_TWO else 0
    n = len(residues) - prolines - nterm_subtract + int(side_chain * count_sc)
    return max(0, n + rules.offset(ion_type))


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
    """Backbone amides (:func:`count_amides`) plus the user-specified ``mod.hex`` sites.

    Any ``namide_scale`` applied to the fit cap is the fitter's concern, not applied here.
    """
    n = count_amides(
        sequence,
        ion_type=ion_type,
        nterm_subtract=nterm_subtract,
        count_sc=count_sc,
        rules=rules,
        registry=registry,
    )
    return n + (mod.hex if mod is not None else 0)
