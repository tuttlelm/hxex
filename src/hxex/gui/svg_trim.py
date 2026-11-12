# SPDX-License-Identifier: GPL-3.0-or-later
"""A figure's drawn data cut to its panels while the window's SVG is written. Qt's SVG renderer
ignores SVG clip paths (HISTORY 237), so a raw profile wider than the m/z range shown, say, was
drawn past the panel's frame; matplotlib's own drawings (saved files, the raster view) clip it
themselves and are not cut. :func:`trimmed_to_axes` cuts, in each panel's own coordinates (0 to
1 across the panel, whatever the layout), the lines, line collections, filled shapes, scatter
points, rectangles and polygons clipped to their panel, and puts every one back afterwards.
Images, meshes, texts and other patches are left as they are. This module does not use Qt."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import numpy as np
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox

_UNIT = Bbox.unit()  # a panel in its own coordinates
_EPS = 1e-9


def _inside(xy: np.ndarray) -> np.ndarray:
    """Whether each point (panel coordinates) is in the panel; NaN points count as inside (they
    draw nothing)."""
    x, y = xy[:, 0], xy[:, 1]
    return ~((x < -_EPS) | (x > 1 + _EPS) | (y < -_EPS) | (y > 1 + _EPS))


def clip_polyline(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The polyline ``xy`` (panel coordinates; NaN breaks it) cut to the panel (Liang-Barsky,
    every segment at once): its vertices, with a NaN row wherever it leaves the panel, and for
    each vertex the index of the point of ``xy`` it is (-1 where the cut made it)."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) < 2:
        keep = np.isfinite(xy).all(axis=1) & _inside(xy)
        return xy[keep], np.flatnonzero(keep)
    p0, p1 = xy[:-1], xy[1:]
    d = p1 - p0
    t0 = np.zeros(len(d))
    t1 = np.ones(len(d))
    ok = np.isfinite(p0).all(axis=1) & np.isfinite(p1).all(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in ((-d[:, 0], p0[:, 0]), (d[:, 0], 1 - p0[:, 0]),
                     (-d[:, 1], p0[:, 1]), (d[:, 1], 1 - p0[:, 1])):
            r = q / p
            t0 = np.where(p < 0, np.maximum(t0, r), t0)
            t1 = np.where(p > 0, np.minimum(t1, r), t1)
            ok &= ~((p == 0) & (q < -_EPS))
    ok &= t0 <= t1 + _EPS
    start = p0 + t0[:, None] * d
    end = p0 + t1[:, None] * d
    whole_start = t0 <= _EPS  # the segment starts at its own first point
    whole_end = t1 >= 1 - _EPS
    # a segment continues the one before when both are kept and meet at an inside point
    cont = np.zeros(len(d), dtype=bool)
    cont[1:] = ok[1:] & ok[:-1] & whole_end[:-1] & whole_start[1:]
    idx = np.flatnonzero(ok)
    if not len(idx):
        lone = np.isfinite(xy).all(axis=1) & _inside(xy)
        return _lone(xy, lone, np.empty((0, 2)), np.empty(0, dtype=int))
    out, src = [], []
    first = True
    for i in idx:  # vertices per kept segment: (break, start) unless it continues, then end
        if not cont[i]:
            if not first:
                out.append((np.nan, np.nan))
                src.append(-1)
            out.append(tuple(start[i]))
            src.append(i if whole_start[i] else -1)
        out.append(tuple(end[i]))
        src.append(i + 1 if whole_end[i] else -1)
        first = False
    out, src = np.array(out, dtype=float), np.array(src, dtype=int)
    # inside points no kept segment reaches (between two NaNs, say) are kept on their own
    covered = np.zeros(len(xy), dtype=bool)
    covered[src[src >= 0]] = True
    lone = np.isfinite(xy).all(axis=1) & _inside(xy) & ~covered
    return _lone(xy, lone, out, src)


def _lone(xy, lone, out, src):
    for j in np.flatnonzero(lone):
        if len(out):
            out = np.vstack([out, [[np.nan, np.nan]]])
            src = np.append(src, -1)
        out = np.vstack([out, xy[j:j + 1]])
        src = np.append(src, j)
    return out.reshape(-1, 2), src


def _to_panel(artist, ax, transform=None):
    """The transform from ``artist``'s data to its panel's coordinates (0 to 1)."""
    return (transform or artist.get_transform()) + ax.transAxes.inverted()


def _clipped_to(artist, ax) -> bool:
    """Whether ``artist`` is drawn clipped to its panel (matplotlib's default for data)."""
    if not (artist.get_visible() and artist.get_clip_on()):
        return False
    box = artist.get_clip_box()
    return box is not None and np.allclose(box.extents, ax.bbox.extents)


@contextlib.contextmanager
def trimmed_to_axes(fig: Figure) -> Iterator[Figure]:
    """``fig`` with the data of every artist clipped to its panel cut to the panel (see the
    module); everything is put back on leaving."""
    undo: list = []
    try:
        for ax in fig.axes:
            ax.get_xlim(), ax.get_ylim()  # the limits as drawn (autoscaling done)
            for a in ax.get_children():
                if a is ax.patch or not _clipped_to(a, ax):
                    continue
                if isinstance(a, Line2D):
                    _line(a, ax, undo)
                elif isinstance(a, LineCollection):
                    _segments(a, ax, undo)
                elif isinstance(a, PolyCollection):
                    _polygons(a, ax, undo)
                elif isinstance(a, PathCollection):
                    _offsets(a, ax, undo)
                elif isinstance(a, Rectangle):
                    _rectangle(a, ax, undo)
                elif isinstance(a, Polygon):
                    _polygon(a, ax, undo)
        yield fig
    finally:
        for restore in reversed(undo):
            restore()
        fig.stale = True


def _line(a: Line2D, ax, undo) -> None:
    x, y = a.get_data(orig=False)
    xy = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    if not len(xy) or a.get_drawstyle() not in ("default", None):
        return
    to = _to_panel(a, ax)
    pts = to.transform(xy)
    if _inside(pts).all():
        return
    has_line = a.get_linestyle() not in ("None", "", " ", "none")
    has_marker = a.get_marker() not in ("None", None, "", " ", "none")
    data, markevery = a.get_data(), a.get_markevery()
    if not has_line:  # markers only: those outside become NaN (drawn nowhere)
        pts[~_inside(pts)] = np.nan
        new = to.inverted().transform(pts)
        a.set_data(new[:, 0], new[:, 1])
    else:
        if has_marker and markevery is not None:
            return  # chosen markers: left as matplotlib draws them
        cut, src = clip_polyline(pts)
        new = to.inverted().transform(cut) if len(cut) else cut
        a.set_data(new[:, 0], new[:, 1])
        if has_marker:  # markers only on the line's own points, not where it was cut
            a.set_markevery([int(i) for i in np.flatnonzero(src >= 0)])
    undo.append(lambda: (a.set_data(*data), a.set_markevery(markevery)))


def _segments(a: LineCollection, ax, undo) -> None:
    to = _to_panel(a, ax)
    old = a.get_segments()
    pts = [to.transform(np.asarray(s, dtype=float).reshape(-1, 2)) for s in old]
    if all(_inside(p).all() for p in pts):
        return
    new = []
    for p in pts:
        cut = clip_polyline(p)[0] if len(p) else p
        # one segment per segment (colours and widths go by position); an empty one draws nothing
        new.append(to.inverted().transform(cut) if len(cut) else np.full((2, 2), np.nan))
    a.set_segments(new)
    undo.append(lambda: a.set_segments(old))


def _polygons(a: PolyCollection, ax, undo) -> None:
    to = _to_panel(a, ax)
    old = list(a.get_paths())
    shown = [p.transformed(to) for p in old]
    if all(_inside(p.vertices).all() for p in shown):
        return
    new = []
    for path, p in zip(old, shown, strict=True):
        if _inside(p.vertices).all():
            new.append(path)
            continue
        cut = p.clip_to_bbox(_UNIT, inside=True)
        new.append(Path(to.inverted().transform(cut.vertices), cut.codes)
                   if len(cut.vertices) else Path(np.full((2, 2), np.nan)))
    a._paths = new  # set_paths of a PolyCollection takes vertices, not paths
    a.stale = True

    def restore():
        a._paths = old
        a.stale = True

    undo.append(restore)


def _offsets(a: PathCollection, ax, undo) -> None:
    old = a.get_offsets()
    if not len(old):
        return
    to = _to_panel(a, ax, a.get_offset_transform())
    pts = to.transform(np.asarray(old, dtype=float))
    outside = ~_inside(pts)
    if not outside.any():
        return
    new = np.array(old, dtype=float)
    new[outside] = np.nan  # an offset that is not finite is not drawn (colours keep their place)
    a.set_offsets(new)
    undo.append(lambda: a.set_offsets(old))


def _rectangle(a: Rectangle, ax, undo) -> None:
    if a.get_angle():
        return
    to = _to_panel(a, ax)  # the patch transform included: the unit square in panel units
    corners = to.transform([[0.0, 0.0], [1.0, 1.0]])
    if _inside(corners).all() or not np.isfinite(corners).all():
        return
    lo, hi = np.minimum(*corners), np.maximum(*corners)
    lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    bounds = a.get_bbox().bounds
    visible = a.get_visible()
    if (hi - lo <= _EPS).any():  # wholly outside
        a.set_visible(False)
    else:
        data = _to_panel(a, ax, a.get_data_transform()).inverted().transform([lo, hi])
        x0, y0 = np.minimum(*data)
        x1, y1 = np.maximum(*data)
        a.set_bounds(x0, y0, x1 - x0, y1 - y0)
    undo.append(lambda: (a.set_bounds(*bounds), a.set_visible(visible)))


def _polygon(a: Polygon, ax, undo) -> None:
    to = _to_panel(a, ax)  # a Polygon's patch transform is the identity
    old = a.get_xy()
    pts = to.transform(np.asarray(old, dtype=float))
    if _inside(pts).all():
        return
    path = Path(pts, closed=True).clip_to_bbox(_UNIT, inside=True)
    cut = (path.vertices if path.codes is None
           else path.vertices[path.codes != Path.CLOSEPOLY])
    visible = a.get_visible()
    if len(cut) < 3:
        a.set_visible(False)
    else:
        a.set_xy(to.inverted().transform(cut))
    undo.append(lambda: (a.set_xy(old), a.set_visible(visible)))
