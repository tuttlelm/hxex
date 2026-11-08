# SPDX-License-Identifier: GPL-3.0-or-later
"""A row of controls that wraps onto further lines when the space is too narrow for it (the
Results list's bar: in one row it set the window's narrowest width; owner, 2026-11-08). After
Qt's flow layout example."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QWidget


class FlowLayout(QLayout):
    """Widgets left to right, wrapping onto the next line where the next one does not fit; its
    narrowest width is that of its widest widget."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 6):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._filling: list[QWidget] = []  # widgets that take the rest of their line
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt's name
        self._items.append(item)

    def add_filling(self, widget: QWidget) -> None:
        """Add ``widget`` to take the rest of its line (a search box after the choices; owner,
        2026-11-08: wider where there is room); it moves to the next line when less than its
        minimum width is left."""
        self.addWidget(widget)
        self._filling.append(widget)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._place(QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._place(rect, move=True)

    def sizeHint(self) -> QSize:  # noqa: N802
        widths = [item.sizeHint().width() for item in self._visible()]
        heights = [item.sizeHint().height() for item in self._visible()]
        return QSize(sum(widths) + self.spacing() * max(len(widths) - 1, 0),
                     max(heights, default=0))

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._visible():
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _visible(self) -> list[QLayoutItem]:
        return [i for i in self._items if not (i.widget() is not None and i.widget().isHidden())]

    def _place(self, rect: QRect, move: bool) -> int:
        """Lay the items out in ``rect`` (moving them when ``move``); returns the height used."""
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line = area.x(), area.y(), 0
        gap = self.spacing()
        for item in self._visible():
            hint = item.sizeHint()
            least = item.minimumSize().width()
            width = max(min(hint.width(), area.width()), least)
            filling = item.widget() is not None and item.widget() in self._filling
            if x > area.x() and x + (least if filling else width) > area.right() + 1:
                x, y, line = area.x(), y + line + gap, 0
            if filling:
                width = max(area.right() + 1 - x, least)
            if move:
                item.setGeometry(QRect(QPoint(x, y), QSize(width, hint.height())))
            x += width + gap
            line = max(line, hint.height())
        return y + line - rect.y() + m.bottom()
