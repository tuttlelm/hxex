# SPDX-License-Identifier: GPL-3.0-or-later
"""The metadata ``modification`` string of pyHXExpress, e.g. ``'H:1 Hex:1 ion_type:c'``."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from .ions import check_ion_type
from .modifications import ChemistryError


@dataclass(frozen=True)
class ModificationSpec:
    """Composition change, extra exchangeable sites, ion type, explicit undeuterated m/z and,
    for a fragment ion, ``parent_pos``: the position in its parent of the fragment's first
    residue (1 for N-terminal ions; None when not known), which places the fragment's amides
    in the parent (see :func:`hxex.chem.count_amides`), and ``parent_nterm``: the parent's
    N-terminal group (``'H-'``, or ``'Ac-'``, after which ``nterm_subtract`` does not apply)."""

    composition: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    hex: int = 0
    ion_type: str = "M"
    undeut_mz: float | None = None
    parent_pos: int | None = None
    parent_nterm: str = "H-"


def parse_modification(text: str | None) -> ModificationSpec:
    """Parse ``'H:1 Hex:1 ion_type:c undeut_mz:523.1'`` (same syntax as pyHXExpress).

    Element keys give composition changes (integers), ``Hex`` adds exchangeable sites,
    ``ion_type`` selects the ion type, ``undeut_mz`` overrides the computed m/z, and
    ``parent_pos`` and ``parent_nterm`` (hxex only) place a fragment in its parent.
    """
    composition: dict[str, int] = {}
    hex_sites = 0
    ion_type = "M"
    undeut_mz = None
    parent_pos = None
    parent_nterm = "H-"
    seen: set[str] = set()
    for item in (text or "").split():
        key, sep, value = item.partition(":")
        if not sep or not key or not value:
            raise ChemistryError(f"cannot parse modification item {item!r} (expected key:value)")
        if key in seen:
            raise ChemistryError(f"modification key {key!r} is given twice in {text!r}")
        seen.add(key)
        try:
            if key == "ion_type":
                ion_type = check_ion_type(value)
            elif key == "undeut_mz":
                undeut_mz = float(value)
            elif key == "Hex":
                hex_sites = int(value)
                if hex_sites < 0:
                    raise ChemistryError(f"Hex must be 0 or more, got {item!r}")
            elif key == "parent_pos":
                parent_pos = int(value)
                if parent_pos < 1:
                    raise ChemistryError(f"parent_pos must be at least 1, got {item!r}")
            elif key == "parent_nterm":
                if value not in ("H-", "Ac-"):
                    raise ChemistryError(f"parent_nterm must be H- or Ac-, got {item!r}")
                parent_nterm = value
            else:
                composition[key] = int(value)
        except ValueError as exc:
            if isinstance(exc, ChemistryError):
                raise
            raise ChemistryError(f"bad value in modification item {item!r}") from exc
    return ModificationSpec(MappingProxyType(composition), hex_sites, ion_type, undeut_mz,
                            parent_pos, parent_nterm)
