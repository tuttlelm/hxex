# SPDX-License-Identifier: GPL-3.0-or-later
"""What a drawn mark is, in words, for the window's tooltips (docs/PLAN.md, "GUI"): a plot
function names a bar or a region with :func:`set_hover`, and the window shows the text when the
pointer rests on it. The text travels with the figure (an attribute of the artist), so it
changes nothing in a saved file."""

from __future__ import annotations

import math

from matplotlib.artist import Artist

_ATTRIBUTE = "hxex_hover"


def set_hover(artist: Artist, *lines: str) -> None:
    """Give ``artist`` the tooltip ``lines`` (empty lines are left out)."""
    setattr(artist, _ATTRIBUTE, "\n".join(line for line in lines if line))


def hover_text(artist: Artist) -> str:
    """The tooltip of ``artist`` ('' when it has none)."""
    return getattr(artist, _ATTRIBUTE, "")


def peptide(sequence, modification="") -> str:
    """The peptide's sequence, with its modification when it has one."""
    return f"{sequence} ({modification})" if modification else str(sequence)


def counted(n_spectra, n_charges) -> str:
    """How many spectra and charge states a value comes from."""
    spectra, charges = int(n_spectra), int(n_charges)
    return (f"{spectra} spectrum" if spectra == 1 else f"{spectra} spectra") + (
        f", {charges} charge states" if charges > 1 else "")


def population(k: int, rank: int) -> str:
    """A fit's population as the tables name it: ``K=1``, ``K=2 low-D``, ``K=2 high-D``,
    ``K=3 mid-D``, and by rank beyond three, ``K=4 rank 2``."""
    k, rank = int(k), int(rank)
    if k == 1:
        return "K=1"
    if k > 3:
        return f"K={k} rank {rank}"
    return f"K={k} {'low-D' if rank == 1 else 'high-D' if rank == k else 'mid-D'}"


def residues(start, end) -> str:
    return f"residues {int(start)}–{int(end)}"


def value(name: str, x, se=None, digits: int = 2, signed: bool = False) -> str:
    """``name x ± se`` (``name not known`` when ``x`` is not a number; no ``± se`` when ``se``
    is not); ``signed``: a plus sign on a positive ``x`` (a difference)."""
    if x is None or not math.isfinite(float(x)):
        return f"{name} not known"
    x = float(x) if round(float(x), digits) != 0 else 0.0  # no "-0.00"
    text = f"{name} {x:{'+' if signed and x else ''}.{digits}f}"
    if se is not None and math.isfinite(float(se)):
        text += f" ± {float(se):.{digits}f}"
    return text
