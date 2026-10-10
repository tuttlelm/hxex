# SPDX-License-Identifier: GPL-3.0-or-later
"""PTM registry and residue tokenisation for modX sequences (acK, pS, pT, pY, ...).

The registry is a private, immutable object. It is handed to pyteomics through the ``aa_comp``
argument; pyteomics' global residue tables are never edited.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property

from pyteomics import mass

_LABEL = re.compile(r"[a-z]*[A-Z]")
_MOD_LABEL = re.compile(r"[a-z]+[A-Z]")


class ChemistryError(ValueError):
    """Base class for errors raised by :mod:`hxex.chem`."""


class SequenceError(ChemistryError):
    """The sequence contains something that is not a known residue."""


class IonTypeError(ChemistryError):
    """The ion type is not one pyteomics defines."""


@dataclass(frozen=True)
class ModifiedResidue:
    """A modified residue: its label, the standard residue it derives from, and the change."""

    label: str
    base: str
    delta: Mapping[str, int] = field(compare=False)


@dataclass(frozen=True)
class ModificationRegistry:
    """Label -> modified residue. Extend with :meth:`with_modification` (returns a copy)."""

    residues: tuple[ModifiedResidue, ...] = ()

    def with_modification(
        self, label: str, base: str, delta: str | Mapping[str, int]
    ) -> ModificationRegistry:
        """Return a new registry that also knows ``label``.

        ``delta`` is the composition added to residue ``base``: a formula such as ``'HPO3'`` or
        a mapping such as ``{'H': 1, 'P': 1, 'O': 3}``.
        """
        if not _MOD_LABEL.fullmatch(label):
            raise ChemistryError(
                f"modification label {label!r} must be a lowercase prefix plus a residue letter"
            )
        if label in self:
            raise ChemistryError(f"modification {label!r} is already defined")
        if label[-1] != base or base not in _standard_residues():
            raise ChemistryError(
                f"modification {label!r} must end with its standard residue letter {base!r}"
            )
        comp = mass.Composition(delta) if isinstance(delta, str) else mass.Composition(dict(delta))
        return ModificationRegistry((*self.residues, ModifiedResidue(label, base, dict(comp))))

    def __contains__(self, label: object) -> bool:
        return any(r.label == label for r in self.residues)

    @cached_property
    def aa_comp(self) -> dict:
        """Residue compositions for pyteomics' ``aa_comp`` argument (a private copy)."""
        table = _standard_aa_comp()
        for r in self.residues:
            table[r.label] = table[r.base] + mass.Composition(dict(r.delta))
        return table

    def base_residue(self, label: str) -> str:
        """The standard residue letter underlying ``label`` (itself for standard residues)."""
        for r in self.residues:
            if r.label == label:
                return r.base
        return label


def _standard_aa_comp() -> dict:
    """pyteomics' residue table without modX entries (a private copy).

    Other code, pyHXExpress for one, adds ``acK`` and similar labels to pyteomics' global table.
    They are left out so that the registry alone decides which modifications exist.
    """
    return {k: v for k, v in mass.std_aa_comp.items() if not _MOD_LABEL.fullmatch(k)}


def _standard_residues() -> set[str]:
    return {k for k in mass.std_aa_comp if len(k) == 1}


DEFAULT_MODIFICATIONS = (
    ModificationRegistry()
    .with_modification("acK", "K", "C2H2O")
    .with_modification("pS", "S", "HPO3")
    .with_modification("pT", "T", "HPO3")
    .with_modification("pY", "Y", "HPO3")
)


def parse_sequence(
    sequence: str, registry: ModificationRegistry = DEFAULT_MODIFICATIONS
) -> tuple[str, ...]:
    """Split a modX sequence into residue labels, e.g. ``'PEPacKTIDE'`` -> 8 residues.

    Raises :class:`SequenceError` for empty sequences and unknown residues.
    """
    if not sequence:
        raise SequenceError("empty sequence")
    residues: list[str] = []
    pos = 0
    while pos < len(sequence):
        m = _LABEL.match(sequence, pos)
        if m is None:
            raise SequenceError(f"cannot read a residue at position {pos} of {sequence!r}")
        residues.append(m.group())
        pos = m.end()
    for label in residues:
        if label not in registry.aa_comp:
            raise SequenceError(f"unknown residue {label!r} in {sequence!r}")
    return tuple(residues)
