# SPDX-License-Identifier: GPL-3.0-or-later
"""The metadata ``modification`` string of pyHXExpress, e.g. ``'H:1 Hex:1 ion_type:c'``."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from .ions import check_ion_type
from .modifications import ChemistryError


@dataclass(frozen=True)
class ModificationSpec:
    """Composition change, extra exchangeable sites, ion type and explicit undeuterated m/z."""

    composition: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    hex: int = 0
    ion_type: str = "M"
    undeut_mz: float | None = None


def parse_modification(text: str | None) -> ModificationSpec:
    """Parse ``'H:1 Hex:1 ion_type:c undeut_mz:523.1'`` (same syntax as pyHXExpress).

    Element keys give composition changes (integers), ``Hex`` adds exchangeable sites,
    ``ion_type`` selects the ion type, ``undeut_mz`` overrides the computed m/z.
    """
    composition: dict[str, int] = {}
    hex_sites = 0
    ion_type = "M"
    undeut_mz = None
    for item in (text or "").split():
        key, sep, value = item.partition(":")
        if not sep or not key or not value:
            raise ChemistryError(f"cannot parse modification item {item!r} (expected key:value)")
        try:
            if key == "ion_type":
                ion_type = check_ion_type(value)
            elif key == "undeut_mz":
                undeut_mz = float(value)
            elif key == "Hex":
                hex_sites = int(value)
            else:
                composition[key] = int(value)
        except ValueError as exc:
            if isinstance(exc, ChemistryError):
                raise
            raise ChemistryError(f"bad value in modification item {item!r}") from exc
    return ModificationSpec(MappingProxyType(composition), hex_sites, ion_type, undeut_mz)
