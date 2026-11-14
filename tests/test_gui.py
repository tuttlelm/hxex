# SPDX-License-Identifier: GPL-3.0-or-later
"""hxex's window (hxex.gui): every figure renders through Qt's SVG renderer, the figure view
zooms, saves and tells what was clicked, and the window opens results and sessions. Run offscreen
(no screen needed)."""
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QPointF, QSettings, QSize, Qt  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402

from hxex.gui.figure_view import FigurePanel, FigureView  # noqa: E402
from hxex.gui.figures import qt_ready  # noqa: E402
from hxex.gui.session import load_session  # noqa: E402
from hxex.gui.tips import tip  # noqa: E402
from hxex.gui.window import MainWindow  # noqa: E402
from test_analysis import two_sample_bundle  # noqa: E402, F401 - the fixture
from test_run import config, raw_spectra, run  # noqa: E402


def _render_problems(svg: bytes, messages: list[str]) -> list[str]:
    renderer = QSvgRenderer(svg)
    return ([] if renderer.isValid() else ["not a valid SVG"]) + messages


def _window(qtbot, settings) -> MainWindow:
    from PySide6.QtWidgets import QMessageBox

    w = MainWindow(settings=QSettings(str(settings), QSettings.Format.IniFormat))
    w.ask_to_save = lambda: QMessageBox.StandardButton.Discard  # no dialog in a test
    qtbot.addWidget(w)
    return w


@pytest.fixture
def window(qtbot, tmp_path):
    return _window(qtbot, tmp_path / "gui.ini")


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="None has no reference count to lose")
def test_qt_calls_leave_the_reference_count_of_none_alone(qtbot):
    """PySide6 6.12.0 on Python 3.11 took a reference from None at each call of a function
    that returns nothing, and the window aborted ("deallocating None") once None had none left
    (HISTORY 249): pyproject.toml excludes it; this fails for any version that does the same."""
    from PySide6.QtWidgets import QApplication, QWidget

    widget = QWidget()
    qtbot.addWidget(widget)
    app = QApplication.instance()
    calls = (app.processEvents, widget.update, lambda: widget.resize(10, 10))
    before = sys.getrefcount(None)
    for _ in range(200):
        for call in calls:
            call()
    # a few references may come or go with what the calls do; one per call (600) may not
    assert abs(sys.getrefcount(None) - before) < 50


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("gui") / "bundle"
    run(raw_spectra(n_series=2), config(), out=out)
    return out


def test_every_figure_renders_through_qt_svg(two_sample_bundle, tmp_path, qtlog):  # noqa: F811
    """Each figure hxex writes as SVG (the series pages, the overviews, the analysis figures)
    draws in Qt's renderer without a warning: a figure feature Qt cannot draw fails here."""
    from hxex.analysis import AnalysisConfig, Annotations, Region, Site
    from hxex.io import open_bundle
    from hxex.report import Results, write_analysis
    from hxex.report.figures import PlotConfig, write_figures

    with open_bundle(two_sample_bundle) as b:
        results = Results.from_bundle(b)
        write_figures(results, tmp_path / "f", PlotConfig(series="all", format="svg"))
        write_figures(results, tmp_path / "k2", PlotConfig(series="all", format="svg", k=2,
                                                           overview=False))
        write_analysis(results, AnalysisConfig(
            reference="S", format="svg", residue_level=True, annotations=Annotations(
                regions=(Region("all", 1, 10),), sites=(Site(5, "x"),))), tmp_path / "a")
    files = sorted(tmp_path.glob("*/*.svg"))
    names = " ".join(p.name for p in files)
    for kind in ("spectra", "uptake", "populations", "overview_", "woods_", "coverage_",
                 "choice_", "overlay_residues_"):
        assert kind in names, kind
    problems = {}
    for path in files:
        before = len(qtlog.records)
        svg = qt_ready(path.read_bytes())  # as the window shows it
        QSvgRenderer(svg)
        found = _render_problems(svg, [r.message for r in qtlog.records[before:]])
        if found:
            problems[path.name] = found
    assert not problems


def test_the_figure_view_zooms_saves_and_tells_what_was_clicked(qtbot, tmp_path):
    fig = Figure(figsize=(4, 3), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    (bar,) = ax.bar([1], [2], width=1.6, gid="box", hatch="//")
    from hxex.report.hover import set_row

    set_row(bar, {"start": 1, "end": 9})
    panel = FigurePanel()
    qtbot.addWidget(panel)
    panel.resize(600, 500)
    with qtbot.waitSignal(panel.drawn, timeout=10000):
        panel.draw(lambda: fig)
    view: FigureView = panel.view
    qtbot.waitUntil(lambda: view.ready, timeout=10000)  # drawn: what can be clicked is known
    assert view.figure is fig and panel.save_button.isEnabled()
    own = view.figure_size()
    assert own.width() > 0
    panel.actual_size()
    assert view.size() == own
    panel.zoom_in()
    assert view.width() > own.width()
    panel.actual_size()
    # the bar spans x 0.2-1.8 of 0-4 and y 0-2 of 0-3 (y down in the view)
    w, h = view.width(), view.height()
    with qtbot.waitSignal(view.clicked, timeout=1000) as clicked, \
            qtbot.waitSignal(panel.clicked_row, timeout=1000) as row:
        qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPointF(w * 0.25, h * 0.8).toPoint())
    assert clicked.args == ["box"] and row.args == [{"start": 1, "end": 9}]
    # a double-click magnifies, and its clicks do not open what is under the pointer (owner:
    # it went straight to the peptide's spectra)
    at = QPointF(w * 0.25, h * 0.8).toPoint()
    with qtbot.assertNotEmitted(view.clicked, wait=1000):
        # as a mouse sends it (QTest's double-click alone sends only the double-click event)
        qtbot.mousePress(view, Qt.MouseButton.LeftButton, pos=at)
        qtbot.mouseRelease(view, Qt.MouseButton.LeftButton, pos=at)
        qtbot.mouseDClick(view, Qt.MouseButton.LeftButton, pos=at)
        qtbot.mouseRelease(view, Qt.MouseButton.LeftButton, pos=at)
    assert panel.magnified
    panel.actual_size()
    assert view.element_at(QPointF(w * 0.9, h * 0.1)) is None
    panel.fit()
    assert view.width() <= panel.scroll.viewport().width()
    panel.clear("nothing")
    assert view.figure is None and not panel.save_button.isEnabled()
    assert view.size() == QSize(0, 0)


@pytest.mark.parametrize("render", ["matplotlib", "svg"])
def test_resting_the_pointer_on_a_mark_shows_its_tooltip(qtbot, render, monkeypatch):
    """A peptide drawn in two panels (the same gid twice) has its tooltip in each, in both
    views; elsewhere there is none. What the view asks Qt to show is recorded, not Qt's tooltip
    window (Windows CI did not show it within seconds)."""
    from PySide6.QtCore import QEvent, QPoint
    from PySide6.QtGui import QHelpEvent
    from PySide6.QtWidgets import QApplication

    from hxex.gui import figure_view
    from hxex.report.hover import set_hover

    shown = []

    class Tip:
        @staticmethod
        def showText(pos, text, widget, rect):  # noqa: N802 - Qt's name
            shown.append((text, rect))

        @staticmethod
        def hideText():  # noqa: N802 - Qt's name
            shown.append(None)

    monkeypatch.setattr(figure_view, "QToolTip", Tip)

    fig = Figure(figsize=(4, 3), dpi=100)
    for n, x0 in enumerate((0.0, 0.5)):
        ax = fig.add_axes((x0, 0, 0.5, 1))
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 3)
        (bar,) = ax.bar([1], [2], width=1.6, gid="peptide:1-9")
        set_hover(bar, f"panel {n}", "residues 1–9")
    panel = FigurePanel(render=render)
    qtbot.addWidget(panel)
    panel.resize(600, 500)
    with qtbot.waitSignal(panel.drawn, timeout=10000):
        panel.draw(lambda: fig)
    view: FigureView = panel.view
    qtbot.waitUntil(lambda: view.ready and bool(view._marks), timeout=10000)
    w, h = view.width(), view.height()
    # each bar spans x 0.2-1.8 of 0-4 in its half and y 0-2 of 0-3 (y down in the view)
    assert view.tooltip_at(QPointF(w * 0.125, h * 0.8)) == "panel 0\nresidues 1–9"
    assert view.tooltip_at(QPointF(w * 0.625, h * 0.8)) == "panel 1\nresidues 1–9"
    assert view.element_at(QPointF(w * 0.625, h * 0.8)) == "peptide:1-9"
    assert view.tooltip_at(QPointF(w * 0.45, h * 0.1)) == ""
    for x, y in ((0.625, 0.8), (0.45, 0.1)):
        at = QPoint(round(w * x), round(h * y))
        QApplication.sendEvent(view, QHelpEvent(QEvent.Type.ToolTip, at, view.mapToGlobal(at)))
    (text, rect), nothing = shown
    assert text == "panel 1\nresidues 1–9" and nothing is None
    assert rect.contains(round(w * 0.625), round(h * 0.8)) and rect.left() >= w * 0.5 - 5


def test_garbage_is_collected_between_events_in_the_windows_thread(qtbot, monkeypatch):
    """Automatic collection ran wherever an allocation crossed its threshold: in a drawing
    thread it freed a window object off the window's thread (a crash, macOS CI), and in the
    window's thread, inside Qt's disconnect, it freed a widget waiting for the lock Qt held (a
    deadlock, HISTORY 365). While the window exists, collection runs from a timer in the
    window's thread, between events, also while work runs off it; closing gives it back."""
    import gc
    import threading

    from hxex.gui.tasks import collection

    assert gc.isenabled()
    panel = FigurePanel()
    qtbot.addWidget(panel)
    assert collection().active and not gc.isenabled()
    seen = {}

    def work():
        seen["enabled"] = gc.isenabled()
        fig = Figure()
        fig.add_subplot().plot([0, 1], [1, 0])
        return fig

    with qtbot.waitSignal(panel.drawn, timeout=10000):
        panel.draw(work)
    assert seen["enabled"] is False and not gc.isenabled()

    collected = []
    collect = gc.collect
    monkeypatch.setattr(gc, "get_count", lambda: gc.get_threshold())  # every generation due
    monkeypatch.setattr(gc, "collect", lambda generation=2: (
        collected.append((generation, threading.current_thread() is threading.main_thread())),
        collect(generation))[1])
    qtbot.waitUntil(lambda: bool(collected), timeout=5000)
    assert collected[0] == (2, True)  # the oldest due, in the window's thread
    monkeypatch.undo()

    collection().release()
    assert gc.isenabled() and not collection().active


def test_a_failed_drawing_is_reported_in_the_panel(qtbot):
    panel = FigurePanel()
    qtbot.addWidget(panel)

    def broken():
        raise ValueError("K = 4 was not fitted")

    with qtbot.waitSignal(panel.failed, timeout=10000) as failed:
        panel.draw(broken)
    assert "K = 4" in failed.args[0] and panel.view.figure is None
    # said on the status line under the figure, in red, without the busy bar
    assert panel.status.full().startswith("Could not draw the figure")
    assert "color" in panel.status.styleSheet() and panel.busy.isHidden()


def test_the_status_line_shows_what_the_panel_is_doing(qtbot):
    """Under the figure (owner: in the button bar "drawing…" was easy to miss): a moving busy
    bar and words while a figure is drawn or read, the figure shown faded meanwhile (it is the
    one before), why there is none, and nothing once it shows. The mouse help is on "?"."""
    import threading

    from hxex.gui.figure_view import HINT, SPAN_HINT

    panel = FigurePanel()
    qtbot.addWidget(panel)
    assert panel.help.text() == "?" and panel.help.toolTip() == tip(HINT)
    assert len(HINT.splitlines()) == 3  # one line per action
    panel.set_span_mode(True)
    assert panel.help.toolTip() == tip(HINT + SPAN_HINT)
    panel.set_span_mode(False)

    def figure():
        fig = Figure(figsize=(2, 2))
        fig.add_subplot().plot([0, 1], [0, 1])
        return fig

    with qtbot.waitSignal(panel.drawn, timeout=10000):
        panel.draw(figure)
    assert panel.status.full() == "" and panel.busy.isHidden()
    assert panel._veil.isHidden()
    assert panel.scroll.graphicsEffect() is None  # the figure is drawn as it is (not blurred)
    release = threading.Event()

    def slow():
        release.wait(10)
        return figure()

    height = panel.status_line.height()
    panel.draw(slow)
    assert panel.status.full() == "Drawing the figure…" and not panel.busy.isHidden()
    # the line keeps its height: the figure above it does not move while it is drawn
    assert panel.status_line.height() == height
    assert panel.status_line.minimumHeight() == panel.status_line.maximumHeight() == height
    assert not panel._veil.isHidden()  # the figure before, faded under a veil
    with qtbot.waitSignal(panel.drawn, timeout=10000):
        release.set()
    assert panel._veil.isHidden() and panel.busy.isHidden()
    panel.clear("reading the elution trace…")
    assert panel.status.full() == "Reading the elution trace…" and not panel.busy.isHidden()
    panel.clear("no elution trace: un2.mzML not found")
    assert panel.status.full().startswith("No elution trace") and panel.busy.isHidden()
    # a long message is shortened to the width there is: it does not widen the window
    from PySide6.QtWidgets import QApplication

    panel.resize(500, 400)
    panel.show()
    QApplication.processEvents()
    narrow = panel.minimumSizeHint().width()
    panel.clear("no elution trace: " + "/a/long/folder" * 40 + "/un2.mzML not found")
    QApplication.processEvents()
    assert panel.minimumSizeHint().width() == narrow
    assert panel.status.text().endswith("…") and "un2.mzML" in panel.status.toolTip()


def test_a_peptide_row_reviews_every_series_under_it(qtbot, window, bundle):
    """Selecting a peptide gives the review panel (its flags and any decision) the spectra of
    every series the tree shows under it, as the row's counts do (owner, 2026-11-13: it had
    taken its first series only); its first series then takes that series alone, although
    the figures do not change."""
    import numpy as np

    from hxex.gui.results_workspace import COLUMN

    window.open_path(bundle)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.results_page
    top = page.tree.topLevelItem(0)
    data_id = np.asarray(page.results.fit.data_id)
    page.tree.setCurrentItem(top)
    assert len(page.review.rows) == len(data_id) == int(top.text(COLUMN["spectra"]))
    # the synthetic run's two series share the charge: named as series
    assert ", 2 series (" in page.review.heading.text()
    assert page.review.flags.isEnabled() == bool(top.text(COLUMN["flagged"]))
    first = top.child(0)
    page.tree.setCurrentItem(first)
    rows = page.review.rows
    assert rows and len(rows) == int(first.text(COLUMN["spectra"])) < len(data_id)
    assert len(set(data_id[rows])) == 1
    page.tree.setCurrentItem(top)
    assert len(page.review.rows) == len(data_id)


def test_the_window_opens_results_draws_a_series_and_saves_the_session(qtbot, window, bundle,
                                                                        tmp_path):
    assert [window.tabs.tabText(i) for i in range(4)] == ["Project", "Run", "Results",
                                                          "Analysis"]
    # every tab says what it holds (owner, 2026-11-13), Project's own tabs too
    assert all(window.tabs.tabToolTip(i) for i in range(4))
    views = window.project_page.views
    assert all(views.tabToolTip(i) for i in range(views.count()))
    window.open_path(bundle)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.results_page
    # one peptide (the synthetic run's two series share sample, peptide and charge)
    assert page.tree.topLevelItemCount() == 1 and window.results_path == bundle.absolute()
    assert page.tree.topLevelItem(0).childCount() == 2
    uptake = page.panel("uptake")
    with qtbot.waitSignal(uptake.drawn, timeout=30000):
        page.tree.setCurrentItem(page.tree.topLevelItem(0).child(0))
    assert uptake.view.figure is not None
    chosen = page.selection()
    assert set(chosen) == {"sample", "peptide_range", "charge"}
    window.tabs.setCurrentIndex(2)
    assert window.modified()
    window.session_path = tmp_path / "view.hxexgui"
    assert window.save_session() and not window.modified()
    saved = load_session(tmp_path / "view.hxexgui")
    assert saved.results == bundle.absolute() and saved.selection == chosen
    assert saved.workspace == "results" and saved.project is None

    # a second window reopens the session: the results and the series
    other = _window(qtbot, tmp_path / "gui2.ini")
    other.open_path(tmp_path / "view.hxexgui")
    qtbot.waitUntil(lambda: not other.loading, timeout=30000)
    assert other.results_page.selection() == chosen
    assert other.tabs.currentIndex() == 2 and not other.modified()
    assert other.recent() == [str(tmp_path / "view.hxexgui")]


def test_closing_a_changed_session_asks_first(qtbot, window, bundle):
    from PySide6.QtWidgets import QMessageBox

    window.open_path(bundle)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    assert window.modified()
    window.ask_to_save = lambda: QMessageBox.StandardButton.Cancel
    window.show()
    assert not window.close() and window.isVisible()
    window.ask_to_save = lambda: QMessageBox.StandardButton.Discard
    assert window.close()


def test_a_session_naming_what_is_gone_opens_with_a_warning(qtbot, window, tmp_path, caplog):
    path = tmp_path / "old.hxexgui"
    path.write_text('version = 1\nworkspace = "results"\nresults = "gone"\n'
                    '[selection]\nsample = "S"\n')
    window.open_path(path)
    assert window.results_path is None and "no longer exists" in caplog.text
    assert window.tabs.currentIndex() == 2


def test_hxex_gui_without_the_extra_says_how_to_install_it(monkeypatch, capsys):
    from hxex import gui
    from hxex.cli import main

    monkeypatch.setattr(gui, "available", lambda: False)
    assert main(["gui"]) == 1
    assert 'pip install "hxex[gui]"' in capsys.readouterr().err


def test_the_icon_is_the_logo_for_the_desktops_scheme(qapp):
    from PySide6.QtGui import QColor

    from hxex.gui.icon import SIZES, logo_icon

    light, dark = logo_icon(dark=False), logo_icon(dark=True)
    assert sorted(s.width() for s in light.availableSizes()) == sorted(SIZES)

    def ink(icon):  # the darkest opaque pixel: the lettering of the light logo is dark
        image = icon.pixmap(64, 64).toImage()
        return min(QColor(image.pixel(x, y)).lightness() for x in range(64) for y in range(64)
                   if QColor.fromRgba(image.pixel(x, y)).alpha() > 200)

    assert ink(light) < ink(dark)
    assert not logo_icon().isNull()  # the current scheme's


def test_the_about_box_shows_the_logo(window):
    from hxex import version_text

    box = window.about_box()
    assert version_text() in box.text()
    logo = box.iconPixmap()
    assert not logo.isNull()
    assert logo.width() / logo.devicePixelRatio() == pytest.approx(220, abs=1)


def _wheel(widget, at, notches):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication

    event = QWheelEvent(at, widget.mapToGlobal(at), QPoint(0, 0), QPoint(0, 120 * notches),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)


def test_wheel_zooms_drag_moves_and_double_click_magnifies_a_panel(qtbot):
    fig = Figure(figsize=(8, 4), dpi=100)
    left, right = fig.subplots(1, 2)
    fig.subplots_adjust(top=0.7)
    fig.suptitle("a title above both panels")
    left.plot([0, 1], [0, 1])
    right.bar([0.5], [0.5], gid="bar")
    panel = FigurePanel()
    qtbot.addWidget(panel)
    panel.resize(700, 500)
    panel.show()
    panel.show_figure(fig)
    view, port = panel.view, panel.scroll.viewport()
    fitted = view.zoom
    # the wheel zooms at the pointer, in and out
    centre = QPointF(port.width() / 2, port.height() / 2)
    _wheel(port, centre, 2)
    assert view.zoom == pytest.approx(fitted * 1.15 ** 2) and not panel.magnified
    _wheel(port, centre, -2)
    assert view.zoom == pytest.approx(fitted)
    # double-click on the right panel: it fills the view; again: the whole figure
    w, h = view.width(), view.height()
    qtbot.mouseDClick(view, Qt.MouseButton.LeftButton, pos=QPointF(w * 0.75, h * 0.5).toPoint())
    assert panel.magnified and view.zoom > fitted * 1.2  # by how much depends on the fonts
    box = view.panel_at(QPointF(view.width() * 0.75, view.height() * 0.5))
    assert box.width() <= port.width() and box.width() >= 0.8 * port.width() or \
        box.height() >= 0.8 * port.height()
    assert panel.scroll.horizontalScrollBar().value() > 0  # scrolled to the right panel
    qtbot.mouseDClick(view, Qt.MouseButton.LeftButton, pos=QPointF(10, 10).toPoint())
    assert not panel.magnified and view.zoom == pytest.approx(fitted)
    # above the panels (the title): nothing to magnify
    assert view.panel_at(QPointF(view.width() / 2, 2)) is None
    # a drag moves a zoomed figure, and is not a click
    panel.zoom_by(3)
    bar = panel.scroll.horizontalScrollBar()
    bar.setValue(bar.maximum() // 2)
    before = bar.value()
    clicks = []
    view.clicked.connect(clicks.append)
    start = QPointF(view.width() * 0.75, view.height() * 0.5).toPoint()
    qtbot.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
    for step in range(1, 6):
        qtbot.mouseMove(view, start + QPointF(-10 * step, 0).toPoint())
    qtbot.mouseRelease(view, Qt.MouseButton.LeftButton, pos=start + QPointF(-50, 0).toPoint())
    assert bar.value() > before and clicks == []


def test_the_title_has_a_row_of_its_own_above_the_figure(qtbot):
    """The title sits on its own row between the buttons and the figure, centred (owner,
    2026-11-05: among the buttons it could cover them when the window was resized); a narrow
    panel shortens it, the whole title on hover."""
    fig = Figure(figsize=(4, 3))
    fig.add_subplot().set_title("WT 0042-0054 TSLSPFYLRPPSF z=2")
    panel = FigurePanel()
    qtbot.addWidget(panel)
    panel.resize(1900, 500)
    panel.show()
    panel.show_figure(fig)
    title, bar = panel.title, panel.bar
    qtbot.waitUntil(lambda: title.text() == "WT 0042-0054 TSLSPFYLRPPSF z=2", timeout=5000)
    assert title.geometry().top() >= bar.geometry().bottom()  # below the buttons, never on them
    assert abs(title.geometry().center().x() - panel.width() / 2) <= 2
    long = "WT 0042-0054 TSLSPFYLRPPSF z=2, " * 12  # wider than the panel can be
    panel.set_title(long)
    panel.resize(160, 500)  # as narrow as the buttons let it be
    qtbot.waitUntil(lambda: title.text().endswith("…"), timeout=5000)
    assert title.toolTip() == tip(long) and title.width() <= panel.width()
    assert title.geometry().top() >= bar.geometry().bottom()


def test_opening_another_project_or_session_asks_to_save_a_changed_one(qtbot, window, bundle,
                                                                       tmp_path):
    from PySide6.QtWidgets import QMessageBox

    from hxex.config import template

    window.open_path(bundle)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    assert window.modified()
    project = tmp_path / "hxex.toml"
    project.write_text(template(), encoding="utf-8")
    asked = []

    def answer(button):
        def ask():
            asked.append(button)
            return button
        return ask

    window.ask_to_save = answer(QMessageBox.StandardButton.Cancel)
    assert not window.open_asking(project)
    assert window.project is None and window.results_path == bundle.absolute()
    window.ask_to_save = answer(QMessageBox.StandardButton.Discard)
    assert window.open_asking(project)
    assert window.project == project.absolute() and window.results_path is None
    assert window.session_path is None and not window.modified()  # a new session, unchanged
    assert len(asked) == 2
    # results open within the session: no question
    window.open_asking(bundle)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    assert len(asked) == 2 and window.results_path == bundle.absolute()


def test_messages_start_afresh_for_another_project(qtbot, window, tmp_path):
    """Opened from one folder, then a project in another: the messages are about the project
    shown, named in full (both files may be called hxex.toml; owner, 2026-11-03)."""
    from hxex.config import template

    first, second = tmp_path / "a" / "hxex.toml", tmp_path / "b" / "hxex.toml"
    for path in (first, second):
        path.parent.mkdir()
        path.write_text(template(), encoding="utf-8")
    window.open_asking(first)
    assert str(first) in window.messages.toPlainText()
    window.open_asking(second)
    text = window.messages.toPlainText()
    assert str(second) in text and str(first) not in text
    assert str(second.parent) in window.run_page.where.text()


def _red_outside_the_frame(view, fig) -> bool:
    """Whether red is drawn left of the axes' frame (the line runs past the axis limits)."""
    from PySide6.QtGui import QColor

    image = view.grab().toImage()
    left = fig.axes[0].get_position().x0 * image.width()
    y = image.height() // 2
    return any(QColor(image.pixel(x, y)).red() > 200 and QColor(image.pixel(x, y)).green() < 90
               for x in range(2, int(left) - 4))


def test_neither_view_draws_past_the_frame(qtbot):
    """Qt's SVG renderer ignores SVG clip paths, so a line beyond the axis limits showed past
    the frame (the owner's raw profiles, 2026-11-03); the window's SVG now has its data cut to
    the panels (hxex.gui.svg_trim), and matplotlib's own drawing clips it."""
    fig = Figure(figsize=(4, 3), dpi=100)
    ax = fig.add_axes((0.3, 0.1, 0.6, 0.8))
    ax.plot([-10, 10], [0.5, 0.5], color="red", lw=4, gid="line")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    panel = FigurePanel(render="svg")
    qtbot.addWidget(panel)
    panel.resize(600, 500)
    panel.show()
    panel.show_figure(fig)
    assert not _red_outside_the_frame(panel.view, fig)  # Qt's SVG: cut at the frame
    panel.set_render("matplotlib")
    view = panel.view
    qtbot.waitUntil(lambda: view._image is not None, timeout=10000)
    assert not _red_outside_the_frame(view, fig)  # matplotlib: clipped at the frame
    # the same click-through and magnifying
    clicks = []
    panel.clicked.connect(clicks.append)
    w, h = view.width(), view.height()
    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPointF(w * 0.6, h * 0.5).toPoint())
    qtbot.waitUntil(lambda: clicks == ["line"], timeout=2000)  # once no second click came
    qtbot.mouseDClick(view, Qt.MouseButton.LeftButton, pos=QPointF(w * 0.6, h * 0.5).toPoint())
    assert panel.magnified
    # drawn again at the new size once the zoom settles
    qtbot.waitUntil(lambda: abs(view._image.width() - view.width()) <= 1, timeout=10000)
    panel.set_render("svg")
    assert panel.view.figure is fig and panel.render == "svg"


def test_a_figures_list_has_steps_that_go_round(qtbot):
    """◀ ▶ beside a figure's list (pages, the overview's pages) step through it and go round
    from the last entry to the first and back (owner, 2026-11-05); they hide with the list and
    do nothing with one entry."""
    from PySide6.QtWidgets import QComboBox

    panel = FigurePanel()
    qtbot.addWidget(panel)
    box = QComboBox()
    box.addItems(["page 1", "page 2", "page 3"])
    panel.add_option(box)
    panel.show()
    prev, nxt = panel._steppers[box]
    assert prev.isVisible() and nxt.isEnabled()
    seen = []
    box.currentIndexChanged.connect(seen.append)
    for _ in range(3):
        nxt.click()
    assert seen == [1, 2, 0]  # round from the last to the first
    prev.click()
    assert box.currentIndex() == 2  # and back from the first to the last
    box.hide()
    assert not prev.isVisible() and not nxt.isVisible()
    box.show()
    assert prev.isVisible()
    box.clear()
    box.addItem("one page")
    assert not nxt.isEnabled() and not prev.isEnabled()


def test_dotnet_starts_before_qt_when_pythonnet_is_installed(monkeypatch):
    """The window starts .NET before Qt (hxex.gui.start_dotnet): started inside the running
    window, Mono crashed the process on a desktop display. Nothing happens without pythonnet,
    and a .NET that does not start is not an error there."""
    import importlib.util
    import types

    from hxex import gui

    calls = []
    fake = types.SimpleNamespace(load=lambda: calls.append("load"))
    monkeypatch.setitem(sys.modules, "pythonnet", fake)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: object() if name == "pythonnet" else real(name, *a))
    assert gui.start_dotnet() is True
    assert calls == ["load"]

    def broken():
        raise RuntimeError("no runtime")

    fake.load = broken
    assert gui.start_dotnet() is False  # logged, not raised
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "pythonnet" else real(name, *a))
    fake.load = lambda: calls.append("again")
    gui.start_dotnet()
    assert calls == ["load"]  # without pythonnet: nothing


def test_closing_the_window_ends_the_process(tmp_path):
    """Closed, the window's process ends at once: after .NET was started (with pythonnet),
    Python's own exit waited on its shutdown, and the terminal did not come back for the owner
    (hxex.gui.end_process). Run as hxex gui runs, in a process of its own."""
    import subprocess
    import time
    from pathlib import Path

    script = tmp_path / "close.py"
    script.write_text(
        "from PySide6.QtCore import QTimer\n"
        "import hxex.gui.window as gw\n"
        "real = gw.MainWindow.show\n"
        "def show(self):\n"
        "    real(self)\n"
        "    QTimer.singleShot(500, self.close)\n"
        "gw.MainWindow.show = show\n"
        "import hxex.gui\n"
        "raise SystemExit(hxex.gui.launch(None))\n")
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen",
           "PYTHONPATH": os.pathsep.join([str(Path(__file__).parents[1] / "src"),
                                          os.environ.get("PYTHONPATH", "")])}
    start = time.monotonic()
    done = subprocess.run([sys.executable, str(script)], env=env, timeout=120,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    assert time.monotonic() - start < 60


def test_the_results_lists_bar_wraps_when_it_is_narrow(qtbot):
    """The Results list's row of choices wraps onto further lines when the list is narrow (in
    one row it set the window's narrowest width; owner, 2026-11-08), and back to one line when
    there is room."""
    from PySide6.QtWidgets import QApplication

    from hxex.gui.spectra_list import SpectraList

    lst = SpectraList()
    qtbot.addWidget(lst)
    lst.show()
    assert lst.minimumSizeHint().width() < 300
    heights = {}
    for width in (1000, 250, 1000):
        lst.resize(width, 400)
        QApplication.processEvents()
        heights.setdefault(width, []).append(lst.bar.height())
    assert heights[250][0] > heights[1000][0] == heights[1000][1]


def test_the_results_panel_counts_on_a_status_line_and_its_search_fills_the_row(qtbot):
    """The series and spectra counts are on a status line under the list (in the rows of
    choices they kept the panel wide; owner, 2026-11-08), the search boxes take the rest of
    their line, and on the List tab ◀ ▶ step row by row, so the choice of what they step
    through is greyed out there."""
    from PySide6.QtWidgets import QApplication

    from hxex.gui.results_workspace import ResultsWorkspace

    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.resize(1400, 800)
    w.show()
    QApplication.processEvents()
    assert w.count.parent() is w.left_status and w.list.count.parent() is w.left_status
    assert w.count.isVisible() and not w.list.count.isVisible()
    # the panel's narrowest width is no longer set by a row of choices in one line: the tree's
    # widest piece is the search box, and the review's status choices wrap (in one line they
    # made the panel 522 px wide with Windows' font)
    assert w.left_tabs.minimumSizeHint().width() <= w.search.minimumWidth() + 20
    radios = w.review.status.buttons()
    assert w.review.minimumSizeHint().width() < sum(b.sizeHint().width() for b in radios)
    w.review.resize(w.review.minimumSizeHint().width(), 400)
    QApplication.processEvents()
    inside = w.review.status_row.rect()
    assert all(inside.contains(b.geometry()) for b in radios)
    # the search box reaches the end of its line
    line_end = w.filters.width()
    assert w.search.geometry().right() >= line_end - 2
    level = w.review.level
    level.setCurrentIndex(2)  # by charge, on the tree
    w.left_tabs.setCurrentWidget(w.list)
    QApplication.processEvents()
    assert not level.isEnabled() and level.currentText() == "spectrum"
    assert w.list.count.isVisible() and not w.count.isVisible()
    assert "row" in w.review.next.toolTip()
    w.left_tabs.setCurrentIndex(0)
    assert level.isEnabled() and level.currentIndex() == 2  # the tree's choice back


def test_no_page_divider_drags_a_panel_out_of_sight(window):
    """A panel dragged to nothing left no sign that it was there (owner, 2026-11-08): the
    dividers stop at each panel's narrowest width."""
    from PySide6.QtWidgets import QSplitter

    splits = window.findChildren(QSplitter)
    assert len(splits) >= 4  # Results, Analysis, the project editor and Tables
    assert not any(s.childrenCollapsible() for s in splits)


def test_the_sequence_views_residues_a_line_go_into_the_session(window):
    """The sequence view's residues a line (owner, 2026-11-10) are kept in the session file
    when not the default, and a session restores them (without asking the results for a
    series of that name)."""
    from hxex.gui.sequence_view import PER_LINE
    from hxex.gui.session import GuiSession

    view = window.project_page.sequence
    assert "residues_per_line" not in window.session().selection
    view.set_per_line(70)
    assert window.session().selection["residues_per_line"] == 70
    view.set_per_line(PER_LINE)
    window._apply(GuiSession(selection={"residues_per_line": 90}))
    assert view.per_line == 90 and "residues_per_line" not in window._pending_selection


def test_tooltips_are_padded_on_a_cream_background(qtbot):
    """The tooltip box: more room and a neutral cream background on a light desktop (owner,
    2026-11-11), a warm dark grey on a dark one; applied to the application's style sheet."""
    from PySide6.QtWidgets import QApplication

    from hxex.gui.tips import TIP_COLOURS, style_tooltips, tooltip_style

    light, dark = tooltip_style(False), tooltip_style(True)
    assert "padding: 2px 4px" in light and TIP_COLOURS[False][0] in light
    assert TIP_COLOURS[True][0] in dark and "#ffffdc" not in light.lower()
    app = QApplication.instance()
    before = app.styleSheet()
    try:
        style_tooltips(app)
        assert "QToolTip" in app.styleSheet() and "padding" in app.styleSheet()
    finally:
        app.setStyleSheet(before)
