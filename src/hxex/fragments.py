# SPDX-License-Identifier: GPL-3.0-or-later
"""Fragment-ion entries generated from a parent peptide (top-down and middle-down HDX).

A fragment list names the ion types, the fragment lengths (residues) and the charges, in the
syntax of pyHXExpress's ``hxex_fragments.py``: ``'c-1:ALL z-dot:2-6,9 M:ALL:1,2'``, that is
``ION[:LENGTHS[:CHARGES]]`` separated by spaces, where a list of lengths or charges holds
numbers and ranges (``2-6``, open ranges ``3-`` and ``-4``) or ``ALL`` (the default for both).

:func:`fragment_entries` turns a parent :class:`~hxex.spectra.PeptideEntry` into one entry per
fragment and charge, with the residues counted as tokens (``acK`` is one residue), so modified
residues stay whole. N-terminal ions (``a``, ``b``, ``c`` and their variants) keep the parent's
N-terminal group and C-terminal ions (``x``, ``y``, ``z``) its C-terminal group (the C-terminal
amide ``-NH2``, for example); parent-type ions (``M``, ``M-H2O``, ``M-NH3``) are the whole
sequence. :func:`fragment_targets` makes the mzML targets of those entries.

Deliberate differences from ``hxex_fragments.py`` (docs/PLAN.md): sequences are sliced by
residue, not by character; ``ALL`` and open ranges mean lengths 1 to the number of residues
(pyHXExpress used ``start_seq`` to ``end_seq``, the same only when the parent starts at 1); and a
parent whose ``modification`` holds more than ``ion_type:M`` is refused, since a composition
change, extra sites or an explicit m/z of the parent cannot be assigned to its fragments (put
modifications on residues instead, as modX labels).

UNVERIFIED: the exchangeable amide count of fragment ions (docs/PLAN.md, "Risks and open
decisions"); the entries use the same amide rules as peptides.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from . import chem
from .spectra import PeptideEntry

log = logging.getLogger(__name__)


def ion_terminus(ion_type: str) -> str:
    """``'N'`` for N-terminal ions (a, b, c), ``'C'`` for C-terminal ions (x, y, z), ``'M'`` for
    the parent-type ions."""
    chem.check_ion_type(ion_type)
    first = ion_type[0]
    return "N" if first in "abc" else "C" if first in "xyz" else "M"


@dataclass(frozen=True)
class FragmentSpec:
    """One item of a fragment list: the ion type and the lengths and charges as written
    (``'ALL'``, or numbers and ranges such as ``'1,3-5,8-'``), resolved by
    :func:`fragment_entries` once the parent's length and charge are known."""

    ion_type: str
    lengths: str = "ALL"
    charges: str = "ALL"

    def __post_init__(self):
        chem.check_ion_type(self.ion_type)
        _expand(self.lengths, 1, "lengths")  # check the syntax
        _expand(self.charges, 1, "charges")

    def __str__(self) -> str:
        if self.charges.upper() != "ALL":
            return f"{self.ion_type}:{self.lengths}:{self.charges}"
        return self.ion_type if self.lengths.upper() == "ALL" else (
            f"{self.ion_type}:{self.lengths}")


def _expand(text: str, top: int, what: str) -> list[int]:
    """``'ALL'`` -> 1 to ``top``; ``'1,3-5,8-'`` -> 1, 3, 4, 5 and 8 to ``top`` (``'-4'`` is 1
    to 4)."""
    if text.strip().upper() == "ALL":
        return list(range(1, top + 1))
    out: set[int] = set()
    for item in text.split(","):
        lo, sep, hi = item.strip().partition("-")
        try:
            a = int(lo) if lo else 1
            b = a if not sep else int(hi) if hi else top
            if a < 1 or (sep and hi and b < a):
                raise ValueError
        except ValueError:
            raise ValueError(f"cannot read {what} {item!r} in {text!r}") from None
        out.update(range(a, b + 1))
    return sorted(out)


def parse_fragments(text: str) -> tuple[FragmentSpec, ...]:
    """The items of a fragment list such as ``'c-1:ALL z-dot:2-6 y:1,2:1,2'`` (see the module
    docstring)."""
    specs = []
    for item in text.split():
        ion, _, rest = item.partition(":")
        lengths, _, charges = rest.partition(":")
        specs.append(FragmentSpec(ion, lengths or "ALL", charges or "ALL"))
    if not specs:
        raise ValueError("an empty fragment list")
    return tuple(specs)


def fragment_entries(
    parent: PeptideEntry,
    fragments: str | Sequence[FragmentSpec],
    *,
    limit_charge: bool = True,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
) -> list[PeptideEntry]:
    """One entry per fragment and charge of ``parent``, in the order of the list, charge by
    charge. ``ALL`` charges are 1 to the parent's charge for parent-type ions and to the parent's
    charge minus one for fragments (``limit_charge=False``: to the parent's charge); charges given
    explicitly are used as given. Lengths beyond the parent are skipped with a warning.

    Each entry keeps the parent's ``file`` and ``sample``; its ``modification`` is
    ``ion_type:ION`` and ``extra`` names the parent (``parent_peptide``, ``parent_range``,
    ``parent_charge``) and the fragment (``fragment``, for example ``c-1:4``).
    """
    mod = chem.parse_modification(parent.modification)
    if (mod.composition or mod.hex or mod.undeut_mz is not None or mod.ion_type != "M"):
        raise ValueError(f"the parent's modification {parent.modification!r} cannot be passed "
                         "to fragments; give modified residues as modX labels instead")
    specs = parse_fragments(fragments) if isinstance(fragments, str) else tuple(fragments)
    nterm, _, cterm = chem.split_termini(parent.peptide)
    residues = chem.parse_sequence(parent.peptide, registry)
    n = len(residues)
    out: list[PeptideEntry] = []
    for spec in specs:
        side = ion_terminus(spec.ion_type)
        top_charge = parent.charge - (1 if limit_charge and side != "M" else 0)
        lengths = [n] if side == "M" else _expand(spec.lengths, n, "lengths")
        too_long = [x for x in lengths if x > n]
        if too_long:
            log.warning("%s: fragment lengths %s are longer than the parent %s (%d residues)",
                        spec.ion_type, too_long, parent.peptide, n)
        for charge in _expand(spec.charges, top_charge, "charges"):
            for length in (x for x in lengths if x <= n):
                if side == "C":
                    first, part = n - length, residues[n - length:]
                    sequence = chem.join_termini("H-", part, cterm)
                else:
                    first, part = 0, residues[:length]
                    sequence = chem.join_termini(nterm, part, cterm if side == "M" else "-OH")
                start = parent.start_seq + first
                end = start + length - 1
                out.append(PeptideEntry(
                    file=parent.file, sample=parent.sample, peptide=sequence,
                    peptide_range=f"{start}-{end}", start_seq=start, end_seq=end,
                    charge=charge, modification=f"ion_type:{spec.ion_type}",
                    extra={"parent_peptide": parent.peptide,
                           "parent_range": parent.peptide_range,
                           "parent_charge": str(parent.charge),
                           "fragment": f"{spec.ion_type}:{length}"},
                ))
    return out


def fragment_targets(
    parent: PeptideEntry,
    fragments: str | Sequence[FragmentSpec],
    *,
    activation: str | None = None,
    precursor_mz: float | None = None,
    precursor_charge: int | str | None = "parent",
    filter_contains: str | None = None,
    filter_excludes: str | None = None,
    rt_window: tuple[float, float] | None = None,
    limit_charge: bool = True,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
):
    """The :class:`~hxex.io.MzmlTarget` (MS2 scans) of every entry of :func:`fragment_entries`.
    ``precursor_charge`` is the parent's charge by default (``'parent'``), or a charge, or None
    to select scans whatever charge they record (some files record a wrong one); the other scan
    conditions are those of the target (for example ``activation='ETD'``,
    ``filter_excludes='sid'``)."""
    from .io.mzml import MzmlTarget

    if isinstance(precursor_charge, str) and precursor_charge != "parent":
        raise ValueError("precursor_charge must be 'parent', a charge or None")
    pz = parent.charge if precursor_charge == "parent" else precursor_charge
    return [
        MzmlTarget(e, ms_level=2, activation=activation, precursor_mz=precursor_mz,
                   precursor_charge=pz, rt_window=rt_window, filter_contains=filter_contains,
                   filter_excludes=filter_excludes)
        for e in fragment_entries(parent, fragments, limit_charge=limit_charge,
                                  registry=registry)
    ]
