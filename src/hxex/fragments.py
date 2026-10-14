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

The amides of a fragment are counted by position in the parent (:mod:`hxex.chem.amides`): the
entries of C-terminal ions carry ``parent_pos`` for it, :func:`fragment_amides` lists the
residues whose amide a fragment holds and :func:`amide_difference` those that the difference of
two fragments measures (``c:6`` minus ``c:5`` is the amide of residue 7). The counts follow the
backbone nitrogens of each ion type's composition; the neutral-loss variants are unverified
(docs/PLAN.md, "Risks and open decisions").
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from . import chem
from .spectra import PeptideEntry

log = logging.getLogger(__name__)


ion_terminus = chem.ion_terminus


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
    explicitly are used as given. A fragment is 1 to ``n - 1`` residues long (``ALL`` is that
    range); longer lengths are skipped with a warning. c ions whose next residue is a proline
    and z ions that start with one are skipped (electron-based dissociation does not cut the
    N-Calpha bond of proline, which is in its ring).

    Each entry keeps the parent's ``file`` and ``sample``; its ``modification`` is
    ``ion_type:ION``, plus ``parent_pos:P`` for C-terminal ions (the parent position of its first
    residue, which places its amides: :func:`hxex.chem.count_amides`), and ``extra`` names the
    parent (``parent_peptide``, ``parent_range``, ``parent_charge``) and the fragment
    (``fragment``, for example ``c-1:4``).
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
        lengths = [n] if side == "M" else _expand(spec.lengths, n - 1, "lengths")
        too_long = [x for x in lengths if x >= n and side != "M"]
        if too_long:
            log.warning("%s: fragment lengths %s are not shorter than the parent %s "
                        "(%d residues)", spec.ion_type, too_long, parent.peptide, n)
        at_proline = [x for x in lengths if x < n and _at_proline(spec.ion_type, residues, x,
                                                                   registry)]
        if at_proline:
            log.info("%s: no %s ions of lengths %s of %s (a cut at a proline)", spec.ion_type,
                     spec.ion_type, at_proline, parent.peptide)
        lengths = [x for x in lengths if (x < n or side == "M") and x not in at_proline]
        for charge in _expand(spec.charges, top_charge, "charges"):
            for length in lengths:
                modification = f"ion_type:{spec.ion_type}"
                if side == "C":
                    first, part = n - length, residues[n - length:]
                    sequence = chem.join_termini("H-", part, cterm)
                    modification += f" parent_pos:{first + 1}"
                else:
                    first, part = 0, residues[:length]
                    sequence = chem.join_termini(nterm, part, cterm if side == "M" else "-OH")
                start = parent.start_seq + first
                end = start + length - 1
                out.append(PeptideEntry(
                    file=parent.file, sample=parent.sample, peptide=sequence,
                    peptide_range=f"{start}-{end}", start_seq=start, end_seq=end,
                    charge=charge, modification=modification,
                    extra={"parent_peptide": parent.peptide,
                           "parent_range": parent.peptide_range,
                           "parent_charge": str(parent.charge),
                           "fragment": f"{spec.ion_type}:{length}"},
                ))
    return out


def _at_proline(ion_type: str, residues: Sequence[str], length: int,
                registry: chem.ModificationRegistry) -> bool:
    """Whether the ``ion_type`` fragment of ``length`` cuts the N-Calpha bond of a proline (a c
    ion before one, a z ion starting with one)."""
    if ion_type[0] == "c":
        label = residues[length]
    elif ion_type[0] == "z":
        label = residues[len(residues) - length]
    else:
        return False
    return registry.base_residue(label) == "P"


def _fragment(parent: PeptideEntry, label: str) -> tuple[str, int]:
    """The ion type and length of a fragment label ``ION:LENGTH`` (``c-1:6``), or of a
    parent-type ion (``M``)."""
    ion, _, length = label.strip().rpartition(":") if ":" in label else (label.strip(), "", "")
    if chem.ion_terminus(ion) == "M":
        return ion, 0
    try:
        return ion, int(length)
    except ValueError:
        raise ValueError(f"cannot read fragment {label!r} (expected ION:LENGTH, e.g. "
                         "'c-1:6')") from None


def fragment_amides(
    parent: PeptideEntry,
    fragment: str,
    *,
    nterm_subtract: int = 1,
    rules: chem.AmideRules = chem.DEFAULT_AMIDE_RULES,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
) -> tuple[int, ...]:
    """The residues (numbered as the parent's ``start_seq``) whose backbone amide the fragment
    ``fragment`` of ``parent`` holds: a label ``ION:LENGTH`` as in the entries' ``fragment``
    (``'c-1:6'``, ``'z-dot:3'``), or ``'M'`` for the parent ion (:func:`hxex.chem.amide_positions`;
    their number is the fragment's amide count)."""
    ion, length = _fragment(parent, fragment)
    positions = chem.amide_positions(parent.peptide, ion, length or None,
                                     nterm_subtract=nterm_subtract, rules=rules,
                                     registry=registry)
    return tuple(parent.start_seq + p - 1 for p in positions)


def amide_difference(
    parent: PeptideEntry,
    larger: str,
    smaller: str,
    *,
    nterm_subtract: int = 1,
    rules: chem.AmideRules = chem.DEFAULT_AMIDE_RULES,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
) -> tuple[int, ...]:
    """The residues (numbered as the parent's ``start_seq``) whose amides make up the deuterium
    of ``larger`` minus that of ``smaller``, two fragments of ``parent`` (labels as in
    :func:`fragment_amides`): ``'c:6'`` minus ``'c:5'`` is residue 7 of the parent, ``'z-dot:6'``
    minus ``'z-dot:3'`` the 3 residues before the last 2, and ``'M'`` minus ``'c:5'`` those of
    the complementary z ion. The amides of ``smaller`` must all be in ``larger`` (else the
    difference is not a set of residues and :class:`ValueError` is raised)."""
    big = fragment_amides(parent, larger, nterm_subtract=nterm_subtract, rules=rules,
                          registry=registry)
    small = fragment_amides(parent, smaller, nterm_subtract=nterm_subtract, rules=rules,
                            registry=registry)
    missing = sorted(set(small) - set(big))
    if missing:
        raise ValueError(f"{smaller} holds amides {missing} that {larger} does not: "
                         f"{larger} - {smaller} is not a set of residues")
    return tuple(p for p in big if p not in set(small))


def parent_mz_range(
    parent: PeptideEntry,
    *,
    picker=None,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
) -> tuple[float, float]:
    """The m/z range of the parent ion's isotope envelope, undeuterated to fully deuterated (the
    lowest and highest m/z of its predicted bins with ``picker``, the default picker settings
    when None): the parent condition of its fragments' targets."""
    from .picking import DEFAULT_PICKER_CONFIG, plan_bins

    plan = plan_bins(parent.peptide, parent.charge, modification=parent.modification,
                     config=picker if picker is not None else DEFAULT_PICKER_CONFIG,
                     registry=registry)
    return float(plan.pred_mz.min()), float(plan.pred_mz.max())


def fragment_targets(
    parent: PeptideEntry,
    fragments: str | Sequence[FragmentSpec],
    *,
    ms_level: int = 2,
    activation: str | None = None,
    preactivation: str | None = None,
    parent_mz: tuple[float, float] | str | None = "parent",
    precursor_mz: float | None = None,
    precursor_charge: int | str | None = "parent",
    filter_contains: str | None = None,
    filter_excludes: str | None = None,
    rt_window: tuple[float, float] | None = None,
    limit_charge: bool = True,
    picker=None,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
):
    """The :class:`~hxex.io.MzmlTarget` (MS2 scans, or ``ms_level`` 3 and above) of every entry
    of :func:`fragment_entries`.

    The scans are tied to the parent: ``parent_mz`` is by default (``'parent'``) the m/z range
    of the parent ion (:func:`parent_mz_range` with ``picker``), which the isolation window of
    the scan's first stage must overlap (see :meth:`hxex.io.ScanTable.select`), so that a file
    with several precursors gives each fragment the scans of its own parent; a range
    ``(low, high)`` sets it, None leaves it out. ``precursor_charge`` is the parent's charge by
    default (``'parent'``), or a charge, or None to select scans whatever charge they record
    (some files record a wrong one). ``preactivation`` selects MS3 scans by the activation before
    the last stage (``'HCD'``; ``'none'`` for none). The other scan conditions are those of the
    target (for example ``activation='ETD'``, ``filter_excludes='sid'``)."""
    from .io.mzml import MzmlTarget

    if isinstance(precursor_charge, str) and precursor_charge != "parent":
        raise ValueError("precursor_charge must be 'parent', a charge or None")
    if isinstance(parent_mz, str) and parent_mz != "parent":
        raise ValueError("parent_mz must be 'parent', a range (low, high) or None")
    if ms_level < 2:
        raise ValueError("fragments are read from MS2 or higher scans (ms_level >= 2)")
    pz = parent.charge if precursor_charge == "parent" else precursor_charge
    pmz = (parent_mz_range(parent, picker=picker, registry=registry)
           if parent_mz == "parent" else parent_mz)
    return [
        MzmlTarget(e, ms_level=ms_level, activation=activation, precursor_mz=precursor_mz,
                   precursor_charge=pz, rt_window=rt_window, filter_contains=filter_contains,
                   filter_excludes=filter_excludes, parent_mz=pmz, preactivation=preactivation)
        for e in fragment_entries(parent, fragments, limit_charge=limit_charge,
                                  registry=registry)
    ]
