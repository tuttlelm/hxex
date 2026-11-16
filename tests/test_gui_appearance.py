# SPDX-License-Identifier: GPL-3.0-or-later
"""View > Appearance: the widget style, remembered in the settings and applied at the next
start, and the dark or light palette of ``hxex gui --dark`` / ``--light`` (hxex.gui.appearance).
Runs offscreen, where Qt offers the two styles every platform has (Fusion and Windows)."""
import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QMenu

from hxex.gui import appearance
from test_gui import _window


def _settings(tmp_path) -> QSettings:
    return QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)


def _menu(window) -> QMenu:
    """View > Appearance, found among the menu bar's descendants: not through
    ``QAction.menu()``, whose wrapper under PySide6 6.7 marks submenus deleted (HISTORY 393)."""
    return next(m for m in window.menuBar().findChildren(QMenu) if m.title() == "Appearance")


@pytest.fixture(autouse=True)
def _back_to_the_default(qapp):
    yield
    appearance.apply_style(appearance.SYSTEM)
    appearance.apply_scheme(appearance.SYSTEM)  # the palette Qt gave at start


def test_the_styles_offered_are_qts_and_fusion_is_one(qapp):
    assert "Fusion" in appearance.styles()
    assert appearance.default_style() in appearance.styles()


def test_choosing_a_style_applies_it_and_remembers_it(qtbot, tmp_path):
    window = _window(qtbot, tmp_path / "gui.ini")
    other = next(s for s in appearance.styles() if s != appearance.default_style())
    actions = {str(a.data()): a for a in _menu(window).actions() if a.data() is not None}
    assert actions[appearance.SYSTEM].isChecked()
    assert actions[appearance.SYSTEM].text().startswith("System default")
    actions[other].trigger()
    assert QApplication.instance().style().objectName().lower() == other.lower()
    assert actions[other].isChecked() and not actions[appearance.SYSTEM].isChecked()
    window.settings.sync()
    assert appearance.saved_style(_settings(tmp_path)) == other
    actions[appearance.SYSTEM].trigger()
    assert appearance.current_style() == appearance.SYSTEM
    assert appearance.saved_style(window.settings) == appearance.SYSTEM


def test_the_remembered_style_is_applied_at_the_next_start(tmp_path):
    other = next(s for s in appearance.styles() if s != appearance.default_style())
    settings = _settings(tmp_path)
    settings.setValue("style", other)
    appearance.apply_saved(settings)
    assert QApplication.instance().style().objectName().lower() == other.lower()


def test_a_style_this_qt_lacks_keeps_the_default_and_says_so(tmp_path, caplog):
    settings = _settings(tmp_path)
    settings.setValue("style", "windows11")  # remembered on another platform
    with caplog.at_level(logging.WARNING, logger="hxex.gui.appearance"):
        appearance.apply_saved(settings)
    assert appearance.current_style() == appearance.SYSTEM
    assert "windows11" in caplog.text and "Fusion" in caplog.text


def test_the_style_is_known_under_the_tooltip_style_sheet(qtbot, tmp_path):
    """Qt reports no style name once an application style sheet is set (hxex's tooltip
    sheet, in `launch`): the name applied last is kept instead."""
    from hxex.gui.tips import style_tooltips

    style_tooltips(QApplication.instance())
    try:
        assert QApplication.instance().style().objectName() == ""  # Qt's proxy
        other = next(s for s in appearance.styles() if s != appearance.default_style())
        window = _window(qtbot, tmp_path / "gui.ini")
        actions = {str(a.data()): a for a in _menu(window).actions() if a.data() is not None}
        assert actions[appearance.SYSTEM].isChecked()
        actions[other].trigger()
        assert appearance.current_style() == other and actions[other].isChecked()
    finally:
        QApplication.instance().setStyleSheet("")


def test_the_menu_offers_only_the_styles(qtbot, tmp_path):
    window = _window(qtbot, tmp_path / "gui.ini")
    texts = [a.text() for a in _menu(window).actions() if a.isCheckable()]
    assert texts[0].startswith("System default") and set(texts[1:]) == set(appearance.styles())


def test_a_dark_or_light_scheme_is_hxexs_palette_set_at_start(qapp):
    appearance.apply_scheme("dark")
    palette = qapp.palette()
    assert palette.color(QPalette.ColorRole.Window).name() == appearance.DARK[
        QPalette.ColorRole.Window]
    assert palette.color(QPalette.ColorRole.Text).lightness() > 200
    appearance.apply_scheme("light")
    assert qapp.palette().color(QPalette.ColorRole.Window).lightness() > 200
    appearance.apply_scheme(appearance.SYSTEM)
    assert qapp.palette().color(QPalette.ColorRole.Window).lightness() > 200
    with pytest.raises(ValueError, match="sepia"):
        appearance.apply_scheme("sepia")


def test_the_window_can_start_dark_or_light():
    """``hxex gui --dark`` / ``--light`` (and ``hxex-gui``): exclusive, default the desktop."""
    from hxex.cli import build_parser
    from hxex.gui import main

    parser = build_parser()
    assert parser.parse_args(["gui"]).scheme is None
    assert parser.parse_args(["gui", "--dark"]).scheme == "dark"
    assert parser.parse_args(["gui", "x.toml", "--light"]).scheme == "light"
    with pytest.raises(SystemExit):
        parser.parse_args(["gui", "--dark", "--light"])
    assert main.__doc__  # hxex-gui takes the same flags: its parser is built in main
