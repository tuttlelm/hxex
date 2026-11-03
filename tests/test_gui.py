# SPDX-License-Identifier: GPL-3.0-or-later
"""hxex's window (hxex.gui): every figure renders through Qt's SVG renderer, the figure view
zooms, saves and tells what was clicked, and the window opens results and sessions. Run offscreen
(no screen needed)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QPointF, QSettings, QSize, Qt  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402

from hxex.gui.figure_view import FigurePanel, FigureView  # noqa: E402
from hxex.gui.figures import qt_ready  # noqa: E402
from hxex.gui.session import load_session  # noqa: E402
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
            reference="S", format="svg", annotations=Annotations(
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
    ax.bar([1], [2], width=1.6, gid="box", hatch="//")
    panel = FigurePanel()
    qtbot.addWidget(panel)
    panel.resize(600, 500)
    with qtbot.waitSignal(panel.drawn, timeout=10000):
        panel.draw(lambda: fig)
    view: FigureView = panel.view
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
    with qtbot.waitSignal(view.clicked, timeout=1000) as clicked:
        qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPointF(w * 0.25, h * 0.8).toPoint())
    assert clicked.args == ["box"]
    assert view.element_at(QPointF(w * 0.9, h * 0.1)) is None
    panel.fit()
    assert view.width() <= panel.scroll.viewport().width()
    panel.clear("nothing")
    assert view.figure is None and not panel.save_button.isEnabled()
    assert view.size() == QSize(0, 0)


def test_a_failed_drawing_is_reported_in_the_panel(qtbot):
    panel = FigurePanel()
    qtbot.addWidget(panel)

    def broken():
        raise ValueError("K = 4 was not fitted")

    with qtbot.waitSignal(panel.failed, timeout=10000) as failed:
        panel.draw(broken)
    assert "K = 4" in failed.args[0] and panel.view.figure is None


def test_the_window_opens_results_draws_a_series_and_saves_the_session(qtbot, window, bundle,
                                                                        tmp_path):
    assert [window.tabs.tabText(i) for i in range(4)] == ["Project", "Run", "Results",
                                                          "Analysis"]
    window.open_path(bundle)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.results_page
    assert page.table.rowCount() == 2 and window.results_path == bundle.absolute()
    uptake = page.panels["uptake"]
    with qtbot.waitSignal(uptake.drawn, timeout=30000):
        page.table.selectRow(1)
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
    assert panel.magnified and view.zoom > fitted * 1.5
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


def test_the_title_is_centred_on_the_bar_clear_of_the_buttons(qtbot):
    fig = Figure(figsize=(4, 3))
    fig.add_subplot().set_title("WT 0042-0054 TSLSPFYLRPPSF z=2")
    panel = FigurePanel()
    qtbot.addWidget(panel)
    panel.resize(1400, 500)
    panel.show()
    panel.show_figure(fig)
    title, bar = panel.title, panel.bar
    assert title.text() == "WT 0042-0054 TSLSPFYLRPPSF z=2"
    assert abs(title.geometry().center().x() - bar.width() / 2) <= 2  # the whole bar's centre
    # a narrow bar: moved aside to clear the buttons, shortened, the whole title on hover
    panel.resize(520, 500)
    qtbot.waitUntil(lambda: bar.width() <= 520)
    assert title.geometry().left() > panel.save_button.geometry().right()
    assert title.geometry().right() <= bar.width()
    assert title.text().endswith("…") and title.toolTip() == "WT 0042-0054 TSLSPFYLRPPSF z=2"


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
