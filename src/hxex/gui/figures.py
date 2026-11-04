# SPDX-License-Identifier: GPL-3.0-or-later
"""Figures for the GUI's SVG view, and saving them (docs/PLAN.md, "GUI"): a figure is saved to
SVG in memory and drawn by Qt's SVG renderer, and the user can save it as SVG, PDF or PNG from
the same :class:`~matplotlib.figure.Figure`. This module does not use Qt."""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
from matplotlib.figure import Figure

SAVE_FORMATS = {"svg": "SVG (vector)", "pdf": "PDF (vector)", "png": "PNG (picture)"}
PNG_DPI = 200  # a saved PNG: sharp enough for a slide or a document


# no date and fixed ids (hatching, clipping): the same figure gives the same SVG
_SVG_RC = {"svg.hashsalt": "hxex"}


# matplotlib writes a path without data for the glyph of a space and for a line with nothing to
# draw (no points, or every point NaN); Qt's renderer reports each ("Invalid path data", and
# "link ... is undefined" where such a glyph is used)
_PATH = re.compile(rb"<path\b([^>]*?)\s*/>")


def _with_data(match: re.Match) -> bytes:
    attrs = match.group(1)
    return match.group(0) if re.search(rb'\sd="', attrs) else b'<path d="M 0 0"' + attrs + b"/>"


def qt_ready(svg: bytes) -> bytes:
    """``svg`` for Qt's renderer: a path without data is given one that draws nothing (the files
    saved for the user are matplotlib's own)."""
    return _PATH.sub(_with_data, svg)


def figure_svg(fig: Figure) -> bytes:
    """``fig`` as SVG, ready for Qt's renderer (:func:`qt_ready`)."""
    buf = io.BytesIO()
    with matplotlib.rc_context(_SVG_RC):
        fig.savefig(buf, format="svg", metadata={"Date": None})
    return qt_ready(buf.getvalue())


@dataclass(frozen=True)
class Mark:
    """A drawn element the window knows: its ``gid`` (what a click reports; may be empty), its
    tooltip ``text`` (:func:`hxex.report.hover.set_hover`; may be empty) and its ``box``
    (left, top, width, height, as fractions of the figure from its top left corner)."""

    gid: str
    text: str
    box: tuple[float, float, float, float]

    def contains(self, x: float, y: float, gx: float = 0.0, gy: float = 0.0) -> bool:
        """Whether the point ``(x, y)`` (fractions, from the top left) is on the mark, allowing
        ``gx`` and ``gy`` around it."""
        left, top, width, height = self.box
        return left - gx <= x <= left + width + gx and top - gy <= y <= top + height + gy

    @property
    def area(self) -> float:
        return self.box[2] * self.box[3]


def figure_marks(fig: Figure) -> list[Mark]:
    """Every visible artist of ``fig`` with a ``gid`` or a tooltip, where it is drawn (each one,
    so a peptide drawn in several panels is found in each). Draws the figure once (matplotlib's
    Agg renderer) to know where things are."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from ..report.hover import hover_text

    found = [a for a in fig.findobj() if a.get_visible() and (a.get_gid() or hover_text(a))]
    if not found:
        return []
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    width, height = canvas.get_width_height()
    marks = []
    for artist in found:
        try:
            bb = artist.get_window_extent(renderer)
        except Exception:  # noqa: BLE001 - an artist without an extent
            continue
        box = (bb.x0 / width, 1 - bb.y1 / height, bb.width / width, bb.height / height)
        if all(math.isfinite(v) for v in box):
            marks.append(Mark(artist.get_gid() or "", hover_text(artist), box))
    return marks


def mark_at(marks: list[Mark], x: float, y: float, gx: float = 0.0, gy: float = 0.0, *,
            text: bool = False) -> Mark | None:
    """The smallest of ``marks`` at ``(x, y)`` (fractions of the figure, from its top left;
    ``gx``, ``gy`` allowed around each); ``text``: only marks with a tooltip."""
    found = [m for m in marks if (m.text if text else m.gid) and m.contains(x, y, gx, gy)]
    return min(found, key=lambda m: m.area) if found else None


def figure_title(fig: Figure) -> str:
    """The figure's title, shown above it in the window so it stays readable at any zoom: the
    page's title (``suptitle``), else the title of a figure of one plot; empty otherwise."""
    title = fig._suptitle.get_text() if fig._suptitle is not None else ""  # get_suptitle: 3.8+
    if not title.strip():
        axes = [ax for ax in fig.axes if ax.get_visible()]
        if len(axes) == 1:
            title = axes[0].get_title()
    return " ".join(title.split())  # one line


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Save ``fig`` in the format its file name says (``.svg``, ``.pdf`` or ``.png``)."""
    path = Path(path)
    fmt = path.suffix.lower().lstrip(".")
    if fmt not in SAVE_FORMATS:
        raise ValueError(f"cannot save a figure as '{path.suffix or path.name}': use "
                         + ", ".join(f".{f}" for f in SAVE_FORMATS))
    metadata = {"Date": None} if fmt == "svg" else None
    with matplotlib.rc_context(_SVG_RC):
        fig.savefig(path, format=fmt, dpi=PNG_DPI if fmt == "png" else None, metadata=metadata)
    return path
