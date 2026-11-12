# SPDX-License-Identifier: GPL-3.0-or-later
"""The window's SVG with its data cut to the panels (Qt's renderer ignores clip paths)."""

import os

import numpy as np
import pytest
from matplotlib.figure import Figure

from hxex.gui.figures import figure_svg
from hxex.gui.svg_trim import clip_polyline, trimmed_to_axes


def test_a_polyline_is_cut_where_it_leaves_the_panel():
    xy = np.array([[-0.5, 0.5], [0.5, 0.5], [1.5, 0.5], [1.5, 0.2], [0.5, 0.2]])
    cut, src = clip_polyline(xy)
    finite = cut[np.isfinite(cut).all(axis=1)]
    assert finite.min() >= 0 and finite.max() <= 1
    # in at x = 0, its own point, out at x = 1; a break; back in at x = 1 to its last point
    assert np.allclose(cut[:3], [[0, 0.5], [0.5, 0.5], [1, 0.5]])
    assert np.isnan(cut[3]).all()
    assert np.allclose(cut[4:], [[1, 0.2], [0.5, 0.2]])
    assert list(src) == [-1, 1, -1, -1, -1, 4]


def test_points_inside_and_nan_breaks_are_kept():
    xy = np.array([[0.2, 0.2], [0.4, 0.4], [np.nan, np.nan], [0.6, 0.6], [2.0, 0.6]])
    cut, src = clip_polyline(xy)
    assert np.allclose(cut[:2], xy[:2]) and np.isnan(cut[2]).all()
    assert np.allclose(cut[3:], [[0.6, 0.6], [1.0, 0.6]])
    lone, src = clip_polyline(np.array([[0.5, 0.5], [np.nan, np.nan], [0.7, 0.3]]))
    assert np.allclose(lone[[0, 2]], [[0.5, 0.5], [0.7, 0.3]]) and list(src) == [0, -1, 2]


def _spilling_figure():
    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot()
    x = np.linspace(0, 10, 101)
    ax.plot(x, np.sin(x), gid="line")
    ax.plot(x[::10], np.cos(x[::10]), "o-", gid="marked")
    ax.fill_between(x, np.sin(x) - 0.2, np.sin(x) + 0.2, gid="band")
    ax.scatter([1, 5, 9], [0, 0, 0], c=["r", "g", "b"], gid="points")
    ax.vlines([1, 5, 9], 0, 1, gid="sticks")
    ax.bar([1, 6.8], [0.5, 0.5], width=1.0, gid="bars")  # outside; over the edge
    ax.set_xlim(3, 7)
    return fig, ax


def test_the_figure_is_cut_inside_and_put_back_after():
    fig, ax = _spilling_figure()
    lines = {a.get_gid(): a for a in ax.get_children() if a.get_gid()}
    before = {
        "line": np.array(lines["line"].get_xydata()),
        "marked": np.array(lines["marked"].get_xydata()),
        "band": [p.vertices.copy() for p in lines["band"].get_paths()],
        "points": np.array(lines["points"].get_offsets()),
        "sticks": [s.copy() for s in lines["sticks"].get_segments()],
    }
    bars = [p for p in ax.patches if p.get_gid() == "bars"] or list(ax.patches)
    bar_bounds = [p.get_bbox().bounds for p in bars]
    lo, hi = ax.get_xlim()
    with trimmed_to_axes(fig):
        xs = lines["line"].get_xdata()
        assert np.nanmin(xs) >= lo - 1e-9 and np.nanmax(xs) <= hi + 1e-9
        assert np.isclose(np.nanmin(xs), lo) and np.isclose(np.nanmax(xs), hi)
        # markers only on the line's own points (3, 4, 5, 6, 7), not where it was cut
        marked = lines["marked"]
        shown = np.asarray(marked.get_xdata())[marked.get_markevery()]
        assert np.allclose(shown, [3, 4, 5, 6, 7])
        for p in lines["band"].get_paths():
            v = p.vertices[np.isfinite(p.vertices).all(axis=1)]
            assert v[:, 0].min() >= lo - 1e-9 and v[:, 0].max() <= hi + 1e-9
        off = lines["points"].get_offsets()
        assert np.isnan(off[[0, 2]]).all() and np.allclose(off[1], [5, 0])  # colours keep place
        segs = lines["sticks"].get_segments()
        assert len(segs) == 3 and np.isnan(segs[0]).all() and np.allclose(segs[1], [[5, 0], [5, 1]])
        assert not bars[0].get_visible()  # wholly outside
        x0, _, w, _ = bars[1].get_bbox().bounds
        assert np.isclose(x0, 6.3) and np.isclose(x0 + w, hi)
    assert np.array_equal(lines["line"].get_xydata(), before["line"])
    assert np.array_equal(lines["marked"].get_xydata(), before["marked"])
    assert lines["marked"].get_markevery() is None
    assert all(np.array_equal(p.vertices, v) for p, v in
               zip(lines["band"].get_paths(), before["band"], strict=True))
    assert np.array_equal(lines["points"].get_offsets(), before["points"])
    assert all(np.array_equal(s, v) for s, v in
               zip(lines["sticks"].get_segments(), before["sticks"], strict=True))
    assert [p.get_bbox().bounds for p in bars] == bar_bounds and all(p.get_visible() for p in bars)


def test_nothing_is_drawn_past_the_frame_in_qt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QByteArray
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])  # a full one: the GUI tests after need it
    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot()
    ax.plot([-100, 100], [0.5, 0.5], lw=4, color="black")  # far wider than the panel
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()  # no frame or labels: only the line can put ink beside the panel
    renderer = QSvgRenderer(QByteArray(figure_svg(fig)))
    size = renderer.defaultSize()
    image = QImage(size.width(), size.height(), QImage.Format.Format_RGB32)
    image.fill(0xFFFFFFFF)
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    bits = np.frombuffer(image.constBits(), np.uint8)
    pixels = bits.reshape(size.height(), image.bytesPerLine() // 4, 4)[:, :size.width(), :3]
    ink = pixels.mean(axis=2) < 128
    left = int(ax.get_position().x0 * size.width())
    right = int(np.ceil(ax.get_position().x1 * size.width()))
    assert ink[:, left + 2:right - 2].any()  # the line is drawn in the panel
    # nowhere beside it (a line cut at the frame keeps its cap: half its width past it)
    assert not ink[:, :left - 3].any() and not ink[:, right + 3:].any()
