# SPDX-License-Identifier: GPL-3.0-or-later
"""What a drawn mark is, for the window (docs/PLAN.md, "GUI"): a plot function names a bar or a
region in words with :func:`set_hover`, which the window shows when the pointer rests on it, and
gives it its values with :func:`set_row`, a row of the figure's data table (the window's data
tab, saved as CSV). Both travel with the figure (attributes of the artist), so they change
nothing in a saved file."""

from __future__ import annotations

import math

from matplotlib.artist import Artist

_ATTRIBUTE = "hxex_hover"
_ROW = "hxex_row"
_ROWS = "hxex_rows"


def set_hover(artist: Artist, *lines: str) -> None:
    """Give ``artist`` the tooltip ``lines`` (empty lines are left out)."""
    setattr(artist, _ATTRIBUTE, "\n".join(line for line in lines if line))


def hover_text(artist: Artist) -> str:
    """The tooltip of ``artist`` ('' when it has none)."""
    return getattr(artist, _ATTRIBUTE, "")


def set_row(artist: Artist, row: dict) -> None:
    """Give ``artist`` its row of the figure's data table: column name -> value (the names of
    the analysis tables: ``sample``, ``time``, ``start``, ``fD_se``, ...)."""
    setattr(artist, _ROW, dict(row))


def row_of(artist: Artist) -> dict | None:
    """The data-table row of ``artist`` (None when it has none)."""
    return getattr(artist, _ROW, None)


def set_rows(artist: Artist, rows: list[dict]) -> None:
    """Give ``artist`` several rows of the figure's data table: a line through many residues,
    one row each (its points cannot be clicked one by one, so they have no :func:`set_row`)."""
    setattr(artist, _ROWS, [dict(r) for r in rows])


def rows_of(artist: Artist) -> list[dict]:
    """Every data-table row of ``artist``: its one row, or its several (none: [])."""
    one = row_of(artist)
    return [one] if one else list(getattr(artist, _ROWS, []) or [])


def set_table(artist: Artist, columns: dict, keep=None, **extra) -> None:
    """Give ``artist`` the rows of a table's ``columns`` (name -> array, as an analysis table's
    ``columns()``), those where ``keep`` is true (default: all), each with the ``extra`` columns
    first (a map drawn from a whole table: its cells are not drawn one artist each). Added to
    the rows ``artist`` already has."""
    import numpy as np

    names = list(columns)
    arrays = [np.asarray(columns[n]) for n in names]
    n = len(arrays[0]) if arrays else 0
    chosen = np.ones(n, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    rows = [{**extra, **{name: _plain(a[i]) for name, a in zip(names, arrays, strict=True)}}
            for i in np.flatnonzero(chosen)]
    setattr(artist, _ROWS, list(getattr(artist, _ROWS, []) or []) + rows)


def _plain(value):
    """A table cell as a plain Python value (a NumPy scalar's ``item()``)."""
    return value.item() if hasattr(value, "item") else value


def time_label(time, scale: str = "seconds") -> str:
    """A time as the figures write it: ``30 min``, ``point 3`` when the times only order the
    points (``scale='order'``), ``undeuterated`` for NaN."""
    from ..io.times import format_time

    time = float(time)
    if not math.isfinite(time):
        return "undeuterated"
    return f"point {format_time(time, scale)}" if scale == "order" else format_time(time)


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
