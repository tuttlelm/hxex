# SPDX-License-Identifier: GPL-3.0-or-later
"""Ctrl+mouse wheel sets the font size of the window's tables, as it does in the Run tab's log
(owner, 2026-11-13): the data tabs, the HDX Summary Table, the Results List, the Tables view and
the Project editor's lists. One size for every table, remembered in the window's settings
(``table_font_size``; 0 or none: the system's); Ctrl+0 over a table goes back to it.

One event filter on the application does it (:class:`TableZoom`), so no table needs code of its
own: a Ctrl+wheel over a table's viewport changes the size, and a table shown for the first time
takes the size chosen."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QSettings, Qt
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QApplication, QHeaderView, QTableView, QWidget

SETTING = "table_font_size"
SIZES = (6, 32)  # the smallest and largest point size


class TableZoom(QObject):
    """The table font size of the window ``root`` (its tables only), kept in ``settings``."""

    def __init__(self, root: QWidget, settings: QSettings):
        super().__init__(root)
        self.root = root
        self.settings = settings
        try:
            self.size = int(settings.value(SETTING, 0) or 0)
        except (TypeError, ValueError):
            self.size = 0
        if self.size and not SIZES[0] <= self.size <= SIZES[1]:
            self.size = 0

    def start(self) -> None:
        QApplication.instance().installEventFilter(self)

    def stop(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)

    # --- which table an event is for ---------------------------------------------------------

    def _table(self, obj) -> QTableView | None:
        """The window's table ``obj`` is (or whose viewport it is), else None."""
        if isinstance(obj, QTableView):
            view = obj
        elif isinstance(obj, QWidget) and isinstance(obj.parent(), QTableView) \
                and obj is obj.parent().viewport():
            view = obj.parent()
        else:
            return None
        return view if self.root.isAncestorOf(view) else None

    def eventFilter(self, obj, event):  # noqa: N802 - Qt's name
        kind = event.type()
        if kind == QEvent.Type.Wheel and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            view = self._table(obj)
            if view is not None:
                step = event.angleDelta().y()
                if step:
                    self.change(view, 1 if step > 0 else -1)
                return True  # a Ctrl+wheel over a table never scrolls it
        elif kind == QEvent.Type.KeyPress \
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier \
                and event.key() == Qt.Key.Key_0:
            view = self._table(obj)
            if view is not None:
                self.set_size(0)
                return True
        elif kind == QEvent.Type.Show and isinstance(obj, QTableView):
            if self.size and self.root.isAncestorOf(obj):
                self.apply(obj)
        return False

    # --- the size ----------------------------------------------------------------------------

    def default_size(self) -> int:
        size = QApplication.font().pointSize()
        return size if size > 0 else 10

    def change(self, view: QTableView, step: int) -> None:
        """One point larger (``step`` 1) or smaller (-1) than ``view`` shows now."""
        now = self.size or view.font().pointSize() or self.default_size()
        self.set_size(min(max(now + step, SIZES[0]), SIZES[1]))

    def set_size(self, size: int) -> None:
        """Every table of the window at ``size`` points (0: the system's), remembered."""
        self.size = size if size != self.default_size() else 0
        self.settings.setValue(SETTING, self.size)
        for view in self.root.findChildren(QTableView):
            self.apply(view)

    def apply(self, view: QTableView) -> None:
        """``view`` at the size chosen: its font, its rows' height, its columns' widths."""
        font = QFont(view.font())
        font.setPointSize(self.size or self.default_size())
        if font.pointSize() == view.font().pointSize():
            return
        view.setFont(font)
        for header in (view.horizontalHeader(), view.verticalHeader()):
            header.setFont(font)
        # rows of a fixed height take the font's; those sized to their contents follow it
        view.verticalHeader().setDefaultSectionSize(QFontMetrics(font).height() + 8)
        model, columns = view.model(), view.horizontalHeader()
        if model is not None and model.columnCount() and all(
                columns.sectionResizeMode(c) == QHeaderView.ResizeMode.Interactive
                for c in range(model.columnCount())):
            view.resizeColumnsToContents()  # stretched columns follow the panel instead


__all__ = ["SETTING", "SIZES", "TableZoom"]
