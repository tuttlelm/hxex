# SPDX-License-Identifier: GPL-3.0-or-later
"""Ctrl+mouse wheel over a table sets the font size of every table of the window, remembered in
its settings; Ctrl+0 goes back to the system's; a plain wheel still scrolls (owner, 2026-11-13).
Runs offscreen."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QTableView  # noqa: E402

from hxex.gui.data_table import DataPanel  # noqa: E402
from hxex.gui.table_zoom import SETTING  # noqa: E402
from test_gui import _window  # noqa: E402


def _wheel(view: QTableView, up: bool, ctrl: bool = True) -> bool:
    """A wheel step over ``view``; whether the event was taken."""
    event = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120 if up else -120),
                        Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.ControlModifier if ctrl
                        else Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(view.viewport(), event)
    return event.isAccepted()


def _panel(window) -> DataPanel:
    panel = DataPanel(window)
    panel.show_rows(["sample", "fD"], [{"sample": "S", "fD": 0.5}])
    panel.show()
    return panel


def test_ctrl_wheel_sets_every_tables_font_size_and_it_is_remembered(qtbot, tmp_path):
    window = _window(qtbot, tmp_path / "gui.ini")
    window.show()
    first, second = _panel(window), _panel(window)
    base = first.table.font().pointSize()
    _wheel(first.table, up=True)
    _wheel(first.table, up=True)
    assert first.table.font().pointSize() == base + 2
    assert second.table.font().pointSize() == base + 2  # one size for every table
    assert first.table.horizontalHeader().font().pointSize() == base + 2
    assert first.table.verticalHeader().defaultSectionSize() > 0
    assert int(window.settings.value(SETTING)) == base + 2
    _wheel(second.table, up=False)
    assert first.table.font().pointSize() == base + 1
    # a table shown later takes the size chosen
    third = _panel(window)
    assert third.table.font().pointSize() == base + 1
    # the next window remembers it
    window.settings.sync()
    again = _window(qtbot, tmp_path / "gui.ini")
    again.show()
    assert _panel(again).table.font().pointSize() == base + 1
    # Ctrl+0 over a table: the system's size again
    key = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_0, Qt.KeyboardModifier.ControlModifier)
    QApplication.sendEvent(first.table, key)
    assert first.table.font().pointSize() == base == third.table.font().pointSize()
    assert int(window.settings.value(SETTING)) == 0


def test_a_plain_wheel_still_scrolls_and_other_windows_are_left_alone(qtbot, tmp_path):
    window = _window(qtbot, tmp_path / "gui.ini")
    window.show()
    panel = _panel(window)
    base = panel.table.font().pointSize()
    _wheel(panel.table, up=True, ctrl=False)
    assert panel.table.font().pointSize() == base
    outside = QTableView()  # not one of the window's tables
    qtbot.addWidget(outside)
    outside.show()
    _wheel(outside, up=True)
    assert outside.font().pointSize() == base and window.settings.value(SETTING) in (None, 0, "0")
