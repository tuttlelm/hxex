# SPDX-License-Identifier: GPL-3.0-or-later
"""PTM registry and residue tokenisation for modX sequences (acK, pS, pT, pY, ...).

The registry is a private, immutable object. It is handed to pyteomics through the ``aa_comp``
argument; pyteomics' global residue tables are never edited.

A sequence may carry terminal groups in modX style: an N-terminal group ending in ``-``
(``Ac-PEPTIDE``) and a C-terminal group starting with ``-`` (``RPKPQQFFGLM-NH2``, the
C-terminal amide). The groups known are in :data:`TERMINAL_GROUPS`; the default termini are
``H-`` and ``-OH``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

from pyteomics import mass

_LABEL = re.compile(r"[a-z]*[A-Z]")
_MOD_LABEL = re.compile(r"[a-z]+[A-Z]")

# elemental composition of the terminal groups (the defaults are H- and -OH)
TERMINAL_GROUPS: Mapping[str, Mapping[str, int]] = {
    "H-": {"H": 1},
    "Ac-": {"C": 2, "H": 3, "O": 1},
    "-OH": {"O": 1, "H": 1},
    "-NH2": {"N": 1, "H": 2},
}


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
    delta: Mapping[str, int]

    def __hash__(self) -> int:  # the delta is a dict, so hash its sorted items
        return hash((self.label, self.base, tuple(sorted(dict(self.delta).items()))))


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

    def to_dict(self) -> dict[str, list[dict]]:
        """The registry as plain data (labels, base residues and composition changes), to be
        stored with results and read back by :meth:`from_dict`."""
        return {"residues": [
            {"label": r.label, "base": r.base,
             "delta": {k: int(v) for k, v in dict(r.delta).items()}}
            for r in self.residues
        ]}

    @classmethod
    def from_dict(cls, data: Mapping) -> ModificationRegistry:
        unknown = set(data) - {"residues"}
        if unknown:
            raise ValueError(f"unknown registry field(s): {', '.join(sorted(unknown))}")
        registry = cls()
        for r in data.get("residues", ()):
            registry = registry.with_modification(r["label"], r["base"], dict(r["delta"]))
        return registry

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


def split_termini(sequence: str) -> tuple[str, str, str]:
    """``(N-terminal group, residues, C-terminal group)`` of a modX sequence, the groups
    defaulting to ``'H-'`` and ``'-OH'``: ``'RPKPQQFFGLM-NH2'`` -> ``('H-', 'RPKPQQFFGLM',
    '-NH2')``. Raises :class:`SequenceError` for a group not in :data:`TERMINAL_GROUPS`."""
    parts = (sequence or "").split("-")
    if len(parts) == 1:
        return "H-", parts[0], "-OH"
    if len(parts) == 3:
        nterm, core, cterm = f"{parts[0]}-", parts[1], f"-{parts[2]}"
    elif len(parts) == 2 and f"{parts[0]}-" in TERMINAL_GROUPS:
        nterm, core, cterm = f"{parts[0]}-", parts[1], "-OH"
    elif len(parts) == 2:
        nterm, core, cterm = "H-", parts[0], f"-{parts[1]}"
    else:
        raise SequenceError(f"cannot read the terminal groups of {sequence!r}")
    if not core:
        raise SequenceError(f"no residues in {sequence!r}")
    for group in (nterm, cterm):
        if group not in TERMINAL_GROUPS:
            raise SequenceError(f"unknown terminal group {group!r} in {sequence!r}; known: "
                                f"{', '.join(TERMINAL_GROUPS)}")
    return nterm, core, cterm


def join_termini(nterm: str, residues, cterm: str) -> str:
    """The modX sequence of ``residues`` with these terminal groups (defaults left out)."""
    return "".join([nterm if nterm != "H-" else "", *residues, cterm if cterm != "-OH" else ""])


def parse_sequence(
    sequence: str, registry: ModificationRegistry = DEFAULT_MODIFICATIONS
) -> tuple[str, ...]:
    """Split a modX sequence into residue labels, e.g. ``'PEPacKTIDE'`` -> 8 residues.
    Terminal groups (:func:`split_termini`) are not residues and are left out.

    Raises :class:`SequenceError` for empty sequences and unknown residues.
    """
    if not sequence:
        raise SequenceError("empty sequence")
    sequence = split_termini(sequence)[1]
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
