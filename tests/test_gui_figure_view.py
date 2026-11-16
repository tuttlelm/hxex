# SPDX-License-Identifier: GPL-3.0-or-later
"""The figure panel (hxex.gui.figure_view): a new figure's own picture and marks in the
matplotlib view, no white blink between figures, failures said on the panel, the wheel, Actual
size and saving (the GUI audit of 2026-11-15, scope C). Run offscreen (no screen needed)."""
import io
import threading

import pytest

pytest.importorskip("PySide6")

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QComboBox

from hxex.gui import figure_view
from hxex.gui.figure_view import FigurePanel
from hxex.gui.order import natural_key
from hxex.gui.tasks import drawing_pool


def _fig(gid: str = "", size=(3, 2)) -> Figure:
    fig = Figure(figsize=size, dpi=100)
    fig.add_subplot().bar([1, 2], [1, 2], gid=gid or None)
    return fig


def _panel(qtbot, render="matplotlib") -> FigurePanel:
    panel = FigurePanel(render=render)
    qtbot.addWidget(panel)
    panel.resize(500, 400)
    panel.show()
    return panel


def _block_drawing_thread():
    """Hold the one drawing thread until the event returned is set (so pictures queue)."""
    release, started = threading.Event(), threading.Event()

    def hold():
        started.set()
        release.wait(20)

    drawing_pool().start(hold)
    started.wait(20)
    return release


def test_a_new_figure_never_takes_the_figures_before_marks(qtbot, monkeypatch):
    """C-1: figure A's picture, still queued when figure B was set, came back after it, counted
    as current and gave B the marks (clicks, tooltips, rows) of A. (A long zoom-settling delay
    makes the order certain: A's picture comes back before anything of B's is asked for.)"""
    warm = _fig("warm")  # fonts and modules loaded: pictures come quickly
    figure_view.figure_marks(warm)
    warm.savefig(io.BytesIO(), format="png")
    settle = 1000
    monkeypatch.setattr(figure_view, "REDRAW_AFTER_MS", settle)
    panel = _panel(qtbot)
    view = panel.view
    release = _block_drawing_thread()
    try:
        panel.show_figure(_fig("A"))
        qtbot.wait(settle + 200)  # A's picture asked for, waiting behind the held thread
        panel.show_figure(_fig("B"))
    finally:
        release.set()
    qtbot.wait(settle + 200)
    qtbot.waitUntil(lambda: view.ready and not view._drawing.running, timeout=10000)
    assert {m.gid for m in view._marks} == {"B"}


def test_a_new_figure_keeps_the_picture_before_until_its_own_is_drawn(qtbot):
    """C-19: the panel went white between figures (the picture dropped, the new one asked for
    only after the zoom-settling delay); now the picture before stays under the veil, the new
    one is asked for at once, and the veil goes when it comes."""
    panel = _panel(qtbot)
    view = panel.view
    with qtbot.waitSignal(panel.drawn, timeout=10000):
        panel.draw(lambda: _fig("A", (8, 6)))
    qtbot.waitUntil(lambda: view.ready, timeout=10000)
    before = view._image
    release = _block_drawing_thread()
    try:
        second = _fig("B", (8, 6))
        panel.show_figure(second)
        assert view._image is before and not view.ready  # the picture before, under the veil
        assert not panel._veil.isHidden()
        assert view._drawing.running and not view._again.isActive()  # asked for at once
        assert panel.status.full() == "" and panel.busy.isHidden()  # the figure is set
    finally:
        release.set()
    with qtbot.waitSignal(view.pictured, timeout=10000):
        pass
    assert view.ready and view._image is not before and panel._veil.isHidden()


def test_a_picture_that_cannot_be_drawn_is_said_on_the_panel(qtbot):
    """C-25: a picture that failed went to the log only, leaving a blank panel."""

    class Broken(Line2D):
        def draw(self, renderer):
            raise ValueError("no picture")

    fig = _fig("A")
    fig.axes[0].add_line(Broken([0, 1], [0, 1]))
    panel = _panel(qtbot)
    with qtbot.waitSignal(panel.failed, timeout=10000) as failed:
        panel.show_figure(fig)
    assert "no picture" in failed.args[0]
    assert panel.status.full().startswith("Could not draw the figure")
    assert "color" in panel.status.styleSheet() and panel._veil.isHidden()
    assert panel.view._image is None


def test_marks_and_panels_are_taken_at_the_pictures_dpi(qtbot, monkeypatch):
    """C-28: the marks were found at the figure's own dpi while the picture was drawn at the
    view's, and a figure with a layout engine is laid out again at the dpi it is drawn at."""
    seen = []
    marks = figure_view.figure_marks
    monkeypatch.setattr(figure_view, "figure_marks", lambda fig: (seen.append(fig.dpi),
                                                                 marks(fig))[1])
    fig = Figure(figsize=(4, 3), dpi=100, layout="constrained")
    ax = fig.add_subplot()
    ax.bar([1], [2], gid="box")
    ax.set_ylabel("a long label on the left of the panel")
    panel = _panel(qtbot)
    panel.resize(900, 700)
    panel.show_figure(fig)
    view = panel.view
    qtbot.waitUntil(lambda: view.ready, timeout=10000)
    dpi = view.width() * view.devicePixelRatioF() / 4
    assert len(seen) == 1 and seen[0] == pytest.approx(dpi, rel=0.01) and seen[0] != 100
    assert fig.dpi == 100  # given back
    # the panel box drawn around the axes as the picture laid it out
    (box,) = view._panels
    pos = ax.get_position()
    left = figure_view.PANEL_ROOM[0]
    assert box.x() == pytest.approx((pos.x0 - left * pos.width) * view.figure_size().width())


def test_the_wheel_over_a_list_in_the_bar_does_not_zoom(qtbot):
    """C-13: the lists in the bar are watched for their ◀ ▶, and their wheel zoomed the figure
    at a point in the list's own coordinates."""
    panel = _panel(qtbot, render="svg")
    pages = QComboBox()
    pages.addItems(["page 1 of 3", "page 2 of 3", "page 3 of 3"])
    panel.add_option(pages)
    panel.show_figure(_fig())
    zoom = panel.view.zoom
    event = QWheelEvent(QPointF(5, 5), QPointF(pages.mapToGlobal(QPoint(5, 5))), QPoint(0, 0),
                        QPoint(0, -120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(pages, event)
    assert panel.view.zoom == zoom
    port = panel.scroll.viewport()  # over the figure it still zooms
    centre = QPointF(port.width() / 2, port.height() / 2)
    QApplication.sendEvent(port, QWheelEvent(
        centre, QPointF(port.mapToGlobal(centre.toPoint())), QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase,
        False))
    assert panel.view.zoom > zoom


def _centre(panel) -> tuple[float, float]:
    port = panel.scroll.viewport()
    p = panel.view.mapFrom(port, QPoint(port.width() // 2, port.height() // 2))
    return p.x() / panel.view.width(), p.y() / panel.view.height()


def test_actual_size_keeps_the_part_of_the_figure_in_view(qtbot):
    """C-18: Actual size set the zoom alone, and the scroll bars' pixels then showed another
    part of the figure. (A figure that at its own size is much larger than the view, so the
    part zoomed in on can be centred at that size.)"""
    panel = _panel(qtbot, render="svg")
    panel.show_figure(_fig(size=(16, 12)))
    panel.zoom_by(3.0, QPointF(400, 300))  # zoomed in on the lower right
    before = _centre(panel)
    panel.actual_size()
    after = _centre(panel)
    assert panel.view.size() == panel.view.figure_size() and not panel._fitted
    assert after == pytest.approx(before, abs=0.02)


def test_the_zoom_buttons_say_the_plain_wheel_zooms(qtbot):
    """C-22: their tooltips said "Ctrl and the mouse wheel"; the plain wheel zooms."""
    from PySide6.QtWidgets import QPushButton

    panel = FigurePanel()
    qtbot.addWidget(panel)
    tips = {b.text(): b.toolTip() for b in panel.bar.findChildren(QPushButton)}
    for text in ("−", "+"):
        assert "mouse wheel over the figure" in tips[text] and "Ctrl" not in tips[text]


def test_a_figure_is_saved_in_the_drawing_thread_with_the_chosen_format(qtbot, tmp_path,
                                                                       monkeypatch):
    """C-17: Save figure… drew in the window's thread while the drawing thread drew (the crash
    of HISTORY 282); C-23: "fig.v2" was refused although a format was chosen."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    threads = []
    save = figure_view.save_figure
    monkeypatch.setattr(figure_view, "save_figure", lambda fig, path: (
        threads.append(threading.current_thread() is threading.main_thread()),
        save(fig, path))[1])
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(
        lambda *a, **k: (str(tmp_path / "fig.v2"), "PNG (picture) (*.png)")))
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda *a, **k: warned.append(a) or QMessageBox.StandardButton.Ok))
    panel = _panel(qtbot, render="svg")
    panel.show_figure(_fig())
    release = _block_drawing_thread()
    try:
        panel.save_dialog()
        assert panel.saving and panel.status.full() == "Saving the figure…"
        assert not panel.busy.isHidden() and panel._veil.isHidden()  # the figure not faded
    finally:
        release.set()
    qtbot.waitUntil(lambda: not panel.saving, timeout=10000)
    assert (tmp_path / "fig.v2.png").exists() and threads == [False] and not warned
    assert panel.status.full() == "" and panel.busy.isHidden()


def test_changing_to_the_svg_view_draws_in_the_drawing_thread(qtbot, monkeypatch):
    """C-17: View > render as SVG made the figure's SVG and marks in the window's thread."""
    threads = []
    svg = figure_view.figure_svg
    monkeypatch.setattr(figure_view, "figure_svg", lambda fig: (
        threads.append(threading.current_thread() is threading.main_thread()), svg(fig))[1])
    panel = _panel(qtbot)
    fig = _fig("A")
    panel.show_figure(fig)
    qtbot.waitUntil(lambda: panel.view.ready, timeout=10000)
    with qtbot.waitSignal(panel.drawn, timeout=10000):
        panel.set_render("svg")
    assert panel.view.figure is fig and threads == [False]
    panel.set_render("matplotlib")  # and back: the picture drawn in the drawing thread
    qtbot.waitUntil(lambda: panel.view.ready, timeout=10000)
    assert panel._veil.isHidden()


def test_a_superscript_after_a_number_is_ordered(qtbot):
    """C-29: str.isdigit is true for '²', which int() refuses."""
    assert sorted(["10²", "9", "2-12"], key=natural_key) == ["2-12", "9", "10²"]
