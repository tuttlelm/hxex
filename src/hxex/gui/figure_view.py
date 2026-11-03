# SPDX-License-Identifier: GPL-3.0-or-later
"""The SVG figure view: a figure drawn by a report function, shown sharp at any size, zoomed,
saved as SVG, PDF or PNG, and clickable by its artists' ``gid`` (docs/PLAN.md, "GUI").

:class:`FigurePanel` draws a figure in a worker thread (the report functions use ``Figure()``,
never pyplot, so they can) and shows it in a :class:`FigureView` with a small toolbar.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from matplotlib.figure import Figure
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QPainter, QPalette
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .figures import SAVE_FORMATS, figure_ids, figure_svg, save_figure
from .tasks import Latest

log = logging.getLogger(__name__)

ZOOM_STEP = 1.25
ZOOM_RANGE = (0.1, 8.0)


class FigureView(QWidget):
    """A figure as SVG. ``zoom`` is relative to the figure's own size (1 = as matplotlib sizes
    it); ``fit`` scales it to the space the view has. Clicking an element with a ``gid`` emits
    :attr:`clicked` with it."""

    clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._renderer = QSvgRenderer(self)
        self._ids: list[str] = []
        self.figure: Figure | None = None
        self.zoom = 1.0
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.white)  # figures are on white
        self.setPalette(pal)

    def set_figure(self, fig: Figure, svg: bytes | None = None) -> None:
        """Show ``fig`` (``svg`` when already drawn, as by a worker thread)."""
        self.figure = fig
        self._ids = figure_ids(fig)
        self._renderer.load(svg if svg is not None else figure_svg(fig))
        self._resize()

    def clear(self) -> None:
        self.figure, self._ids = None, []
        self._renderer.load(b"")
        self.setFixedSize(QSize(0, 0))
        self.update()

    def figure_size(self) -> QSize:
        return self._renderer.defaultSize() if self._renderer.isValid() else QSize(0, 0)

    def set_zoom(self, zoom: float) -> None:
        self.zoom = min(max(zoom, ZOOM_RANGE[0]), ZOOM_RANGE[1])
        self._resize()

    def fit(self, size: QSize) -> None:
        """Zoom so the whole figure fits ``size``."""
        own = self.figure_size()
        if own.width() > 0 and own.height() > 0:
            self.set_zoom(min(size.width() / own.width(), size.height() / own.height()))

    def _resize(self) -> None:
        own = self.figure_size()
        self.setFixedSize(QSize(round(own.width() * self.zoom), round(own.height() * self.zoom)))
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt's name
        if not self._renderer.isValid():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._renderer.render(painter, QRectF(self.rect()))
        painter.end()

    def element_at(self, point: QPointF) -> str | None:
        """The ``gid`` of the drawn element under ``point`` (in the view's coordinates), the
        smallest when several overlap."""
        own = self.figure_size()
        if not self._ids or own.width() == 0:
            return None
        view = self._renderer.viewBoxF()
        sx, sy = view.width() / self.width(), view.height() / self.height()
        at = QPointF(view.x() + point.x() * sx, view.y() + point.y() * sy)
        best, area = None, None
        for gid in self._ids:
            if not self._renderer.elementExists(gid):
                continue
            box = self._renderer.transformForElement(gid).mapRect(
                self._renderer.boundsOnElement(gid))
            if box.contains(at) and (area is None or box.width() * box.height() < area):
                best, area = gid, box.width() * box.height()
        return best

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt's name
        if event.button() == Qt.MouseButton.LeftButton:
            gid = self.element_at(event.position())
            if gid is not None:
                self.clicked.emit(gid)
        super().mouseReleaseEvent(event)


class FigurePanel(QWidget):
    """A :class:`FigureView` in a scroll area, with buttons to zoom, fit and save. Give it a
    figure (:meth:`show_figure`) or a function that draws one (:meth:`draw`, run in a worker
    thread)."""

    drawn = Signal()  # a figure is shown
    failed = Signal(str)

    def __init__(self, parent=None, pool: QThreadPool | None = None):
        super().__init__(parent)
        self.view = FigureView()
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.view)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status = QLabel()
        self._fitted = True
        self._drawing = Latest(self, pool)

        buttons = QHBoxLayout()
        for text, tip, slot in (("Fit", "show the whole figure", self.fit),
                                ("−", "zoom out (Ctrl and the mouse wheel)", self.zoom_out),
                                ("+", "zoom in (Ctrl and the mouse wheel)", self.zoom_in),
                                ("100 %", "the figure's own size", self.actual_size),
                                ("Save figure…", "save as SVG, PDF or PNG", self.save_dialog)):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        self.save_button = button  # the last one
        buttons.addStretch()
        buttons.addWidget(self.status)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(buttons)
        layout.addWidget(self.scroll)
        self.save_button.setEnabled(False)

    # --- showing ---------------------------------------------------------------------------

    def show_figure(self, fig: Figure, svg: bytes | None = None) -> None:
        self.view.set_figure(fig, svg)
        self.save_button.setEnabled(True)
        self.status.setText("")
        if self._fitted:
            self.fit()
        self.drawn.emit()

    def draw(self, draw: Callable[[], Figure]) -> None:
        """Draw a figure with ``draw`` in a worker thread and show it when done; a newer call
        replaces an older one still drawing."""
        def work():
            fig = draw()
            return fig, figure_svg(fig)

        self.status.setText("drawing…")
        self._drawing.submit(work, lambda done: self.show_figure(*done), self._failed)

    @property
    def drawing(self) -> bool:
        return self._drawing.running

    def clear(self, message: str = "") -> None:
        """Show no figure (``message`` in its place)."""
        self.view.clear()
        self.save_button.setEnabled(False)
        self.status.setText(message)

    def _failed(self, message: str) -> None:
        self.status.setText("could not draw the figure")
        log.error("could not draw the figure: %s", message)
        self.failed.emit(message)

    # --- zooming ---------------------------------------------------------------------------

    def fit(self) -> None:
        self._fitted = True
        margin = 4
        self.view.fit(self.scroll.viewport().size() - QSize(margin, margin))

    def zoom_in(self) -> None:
        self._fitted = False
        self.view.set_zoom(self.view.zoom * ZOOM_STEP)

    def zoom_out(self) -> None:
        self._fitted = False
        self.view.set_zoom(self.view.zoom / ZOOM_STEP)

    def actual_size(self) -> None:
        self._fitted = False
        self.view.set_zoom(1.0)

    def resizeEvent(self, event):  # noqa: N802 - Qt's name
        super().resizeEvent(event)
        if self._fitted and self.view.figure is not None:
            self.fit()

    def showEvent(self, event):  # noqa: N802 - Qt's name
        super().showEvent(event)  # a figure drawn in a hidden tab was fitted to no space
        if self._fitted and self.view.figure is not None:
            self.fit()

    def wheelEvent(self, event):  # noqa: N802 - Qt's name
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            (self.zoom_in if event.angleDelta().y() > 0 else self.zoom_out)()
            event.accept()
        else:
            super().wheelEvent(event)

    # --- saving ----------------------------------------------------------------------------

    def save_dialog(self) -> None:
        if self.view.figure is None:
            return
        filters = ";;".join(f"{label} (*.{fmt})" for fmt, label in SAVE_FORMATS.items())
        path, chosen = QFileDialog.getSaveFileName(self, "Save figure", "", filters)
        if not path:
            return
        if "." not in path.rsplit("/", 1)[-1]:  # no suffix typed: the chosen format's
            path += "." + next(f for f, label in SAVE_FORMATS.items() if chosen.startswith(label))
        try:
            save_figure(self.view.figure, path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Figure not saved", str(exc))
            return
        self.status.setText(f"saved {path}")
        log.info("figure saved to %s", path)
