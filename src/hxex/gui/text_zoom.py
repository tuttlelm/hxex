# SPDX-License-Identifier: GPL-3.0-or-later
"""Ctrl+mouse wheel sets the font size of the window's tables, lists and text boxes (owner,
2026-11-13 for the tables; 2026-11-20 for the rest): the data tabs, the HDX Summary Table, the
Results tree and List, the Analysis tree, the Tables view, the Project editor's lists, the File
tab, the Run tab's log, the messages pane, and the sequence view. One size for all of them,
remembered in the window's settings (``table_font_size``, its name from when it was for tables
only; 0 or none: the system's); Ctrl+0 over one of them goes back to it.

One event filter on the application does it (:class:`TextZoom`), so no widget needs code of its
own: a Ctrl+wheel over one of them changes the size, and one shown for the first time (or again,
after the size changed while it was hidden) takes the size chosen. Only the widgets in view are
changed at once: changing every table at every wheel step, each measuring all its rows for its
columns' widths, made each step lag on a large project (owner, 2026-11-20). Columns are sized
from their first rows (:func:`hxex.gui.data_table.size_columns`), as the data tables are.

Labels, buttons and the figures are not changed: the wheel already zooms a figure, and its
fonts are those of the figure saved."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QSettings, Qt
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QPlainTextEdit,
    QTableView,
    QTextEdit,
    QTreeView,
    QWidget,
)

SETTING = "table_font_size"
SIZES = (6, 32)  # the smallest and largest point size
ZOOMED = (QAbstractItemView, QPlainTextEdit, QTextEdit)  # the widgets whose text is sized
ZOOMED_MARK = "text_zoomed"  # a widget's property: set to another size than the usual one


class TextZoom(QObject):
    """The font size of the tables, lists and text boxes of the window ``root``, kept in
    ``settings``."""

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

    # --- which widget an event is for --------------------------------------------------------

    def _target(self, obj) -> QWidget | None:
        """The window's table, list or text box ``obj`` is (or whose viewport or header it is),
        else None. Popups (a list's drop-down) are windows of their own and are left alone."""
        if isinstance(obj, QHeaderView):  # a table's or tree's header: the view it heads
            obj = obj.parent()
        if isinstance(obj, ZOOMED):
            view = obj
        elif isinstance(obj, QWidget) and isinstance(obj.parent(), ZOOMED) \
                and obj is obj.parent().viewport():
            view = obj.parent()
        else:
            return None
        if isinstance(view, QHeaderView):
            return None
        return view if self.root.isAncestorOf(view) else None

    def eventFilter(self, obj, event):  # noqa: N802 - Qt's name
        kind = event.type()
        if kind == QEvent.Type.Wheel and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            view = self._target(obj)
            if view is not None:
                step = event.angleDelta().y()
                if step:
                    self.change(view, 1 if step > 0 else -1)
                return True  # a Ctrl+wheel over them never scrolls (nor Qt's own text zoom)
        elif kind == QEvent.Type.KeyPress \
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier \
                and event.key() == Qt.Key.Key_0:
            view = self._target(obj)
            if view is not None:
                self.set_size(0)
                return True
        elif kind == QEvent.Type.Show and isinstance(obj, ZOOMED) \
                and not isinstance(obj, QHeaderView):
            if self.root.isAncestorOf(obj):
                self.apply(obj)  # nothing to do when it already has the size, or was never zoomed
        return False

    # --- the size ----------------------------------------------------------------------------

    def default_size(self) -> int:
        size = QApplication.font().pointSize()
        return size if size > 0 else 10

    def change(self, view: QWidget, step: int) -> None:
        """One point larger (``step`` 1) or smaller (-1) than ``view`` shows now."""
        now = self.size or view.font().pointSize() or self.default_size()
        self.set_size(min(max(now + step, SIZES[0]), SIZES[1]))

    def views(self) -> list[QWidget]:
        """The window's tables, lists and text boxes (not the headers)."""
        found = []
        for kind in ZOOMED:
            found += [w for w in self.root.findChildren(kind) if not isinstance(w, QHeaderView)]
        return list({id(w): w for w in found}.values())

    def set_size(self, size: int) -> None:
        """The size ``size`` points (0: the system's), remembered, for the widgets in view now;
        the others take it when they are next shown."""
        self.size = size if size != self.default_size() else 0
        self.settings.setValue(SETTING, self.size)
        for view in self.views():
            if view.isVisible():
                self.apply(view)

    def apply(self, view: QWidget) -> None:
        """``view`` at the size chosen: its font and, for a table or tree, its headers' fonts,
        its rows' height, and its columns' widths."""
        if not self.size and not view.property(ZOOMED_MARK):
            return  # the usual size, and never zoomed: its font is its own
        size = self.size or self.default_size()
        if view.font().pointSize() == size:
            return
        font = QFont(view.font())
        font.setPointSize(size)
        view.setFont(font)
        view.setProperty(ZOOMED_MARK, bool(self.size))
        if isinstance(view, QTableView):
            for header in (view.horizontalHeader(), view.verticalHeader()):
                header.setFont(font)
            # rows of a fixed height take the font's; those sized to their contents follow it
            view.verticalHeader().setDefaultSectionSize(QFontMetrics(font).height() + 8)
            model, columns = view.model(), view.horizontalHeader()
            if model is not None and model.columnCount() and all(
                    columns.sectionResizeMode(c) == QHeaderView.ResizeMode.Interactive
                    for c in range(model.columnCount())):
                # from the first rows: every row, as resizeColumnsToContents reads them, took
                # seconds on a large list at each wheel step; stretched columns follow the panel
                from .data_table import size_columns

                size_columns(view)
        elif isinstance(view, QTreeView):
            header = view.header()
            header.setFont(font)
            model = view.model()
            last = (model.columnCount() if model is not None else 0) - 1
            for c in range(last + 1):
                if header.sectionResizeMode(c) == QHeaderView.ResizeMode.Interactive \
                        and not header.isSectionHidden(c) \
                        and not (header.stretchLastSection() and c == last):
                    view.resizeColumnToContents(c)  # a tree reads only some rows for it


__all__ = ["SETTING", "SIZES", "TextZoom", "ZOOMED"]
