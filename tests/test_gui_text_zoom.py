# SPDX-License-Identifier: GPL-3.0-or-later
"""Ctrl+mouse wheel over a table, list or text box sets the font size of all of them in the
window, remembered in its settings; Ctrl+0 goes back to the system's; a plain wheel still
scrolls (owner, 2026-11-13; lists and text boxes 2026-11-20). Runs offscreen."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QPlainTextEdit, QTableView, QTreeWidget, QTreeWidgetItem

from hxex.gui.data_table import DataPanel
from hxex.gui.text_zoom import SETTING
from test_gui import _window


def _wheel(view, up: bool, ctrl: bool = True) -> bool:
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


def test_only_the_tables_in_view_change_at_once_and_the_others_when_shown(qtbot, tmp_path):
    # changing every table at each wheel step, each measuring all its rows, made each step lag
    window = _window(qtbot, tmp_path / "gui.ini")
    window.show()
    shown = _panel(window)
    hidden = DataPanel(window)
    hidden.show_rows(["sample", "fD"], [{"sample": f"S{i}", "fD": i / 1000} for i in range(2000)])
    hidden.hide()
    base = shown.table.font().pointSize()
    _wheel(shown.table, up=True)
    assert shown.table.font().pointSize() == base + 1
    assert hidden.table.font().pointSize() == base  # left until it is shown
    hidden.show()
    assert hidden.table.font().pointSize() == base + 1
    assert hidden.table.horizontalHeader().font().pointSize() == base + 1


def test_text_boxes_and_trees_take_the_same_size(qtbot, tmp_path):
    window = _window(qtbot, tmp_path / "gui.ini")
    window.show()
    panel = _panel(window)
    text = QPlainTextEdit(window)  # editable, as the File tab is: Qt's own zoom leaves it alone
    text.setPlainText("[input]\nformat = \"specexport\"\n")
    text.show()
    tree = QTreeWidget(window)
    tree.setColumnCount(2)
    tree.addTopLevelItem(QTreeWidgetItem(["0001-0010", "2"]))
    tree.show()
    base = text.font().pointSize()
    assert _wheel(text, up=True)
    assert text.font().pointSize() == base + 1
    assert panel.table.font().pointSize() == tree.font().pointSize() == base + 1
    assert tree.header().font().pointSize() == base + 1
    _wheel(tree, up=True)
    assert text.font().pointSize() == panel.table.font().pointSize() == base + 2
    key = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_0, Qt.KeyboardModifier.ControlModifier)
    QApplication.sendEvent(text, key)
    assert text.font().pointSize() == tree.font().pointSize() == base


def test_the_sequence_view_is_drawn_again_at_the_new_size(qtbot, tmp_path):
    from hxex.gui.sequence_view import SequenceRow, SequenceView

    window = _window(qtbot, tmp_path / "gui.ini")
    window.show()
    view = SequenceView(window)
    view.rows = [SequenceRow(samples=["WT"], residues=tuple("MKPLAAGDKE"), first=1)]
    view._draw()
    view.show()
    assert "font-size: 11pt" in view.text.toHtml() or "font-size:11pt" in view.text.toHtml()
    for _ in range(3):
        _wheel(view.text, up=True)
    html = view.text.toHtml().replace(" ", "")
    assert "font-size:11pt" not in html and view._scale() > 1


def test_the_usual_size_again_reaches_a_box_hidden_at_the_time(qtbot, tmp_path):
    window = _window(qtbot, tmp_path / "gui.ini")
    window.show()
    text = QPlainTextEdit(window)
    text.show()
    other = QPlainTextEdit(window)
    other.show()
    base = text.font().pointSize()
    _wheel(text, up=True)
    other.hide()
    key = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_0, Qt.KeyboardModifier.ControlModifier)
    QApplication.sendEvent(text, key)
    assert text.font().pointSize() == base and other.font().pointSize() == base + 1
    other.show()
    assert other.font().pointSize() == base
