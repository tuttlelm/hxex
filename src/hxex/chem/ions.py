# SPDX-License-Identifier: GPL-3.0-or-later
"""Ion types: the set pyteomics defines (M, a, b, c, x, y, z and variants)."""

from __future__ import annotations

from pyteomics import mass

from .modifications import IonTypeError


def ion_types() -> tuple[str, ...]:
    """All ion types pyteomics defines, sorted."""
    return tuple(sorted(mass.std_ion_comp))


def check_ion_type(ion_type: str) -> str:
    """Return ``ion_type`` if pyteomics defines it, else raise :class:`IonTypeError`."""
    if ion_type not in mass.std_ion_comp:
        raise IonTypeError(f"unknown ion type {ion_type!r}; known: {', '.join(ion_types())}")
    return ion_type
