# SPDX-License-Identifier: GPL-3.0-or-later
"""The SVG figure view: a figure drawn by a report function, shown sharp at any size, zoomed,
saved as SVG, PDF or PNG, and clickable by its artists' ``gid`` (docs/PLAN.md, "GUI").

:class:`FigurePanel` draws a figure in a worker thread (the report functions use ``Figure()``,
never pyplot, so they can) and shows it in a :class:`FigureView` with a small toolbar.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable

from matplotlib.figure import Figure
from PySide6.QtCore import (
    QEvent,
    QPoint,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import QImage, QPainter, QPalette
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .figures import (
    SAVE_FORMATS,
    Mark,
    figure_marks,
    figure_svg,
    figure_title,
    mark_at,
    rows_at,
    save_figure,
)
from .tasks import Latest, drawing_pool

log = logging.getLogger(__name__)

ZOOM_STEP = 1.25
WHEEL_STEP = 1.15  # per notch of the mouse wheel
ZOOM_RANGE = (0.1, 16.0)
# around a panel when it is magnified: its tick labels and title (left, right, top, bottom, as
# shares of the panel's own width and height)
PANEL_ROOM = (0.24, 0.04, 0.12, 0.24)
REDRAW_AFTER_MS = 150  # the matplotlib view draws again once the zoom has settled this long
MAX_PIXELS = 40e6  # the most pixels the matplotlib view draws (a page zoomed far in)
DRAG_START = 4  # pixels the pointer moves before a press becomes a drag (not a click)
# pixels around a drawn mark that still count as clicking it (a thin line has no height of its
# own: a click had to land on its one pixel row)
CLICK_ALLOWANCE = 4
HINT = ("Mouse wheel: zoom in and out at the pointer. Drag: move a zoomed figure. "
        "Double-click a panel: magnify it (double-click again: the whole figure). "
        "Mouse over a peptide or bar for details.")
# added under the figures whose peptides lead to their spectra (the Analysis workspace's)
SPECTRA_HINT = " Right-click a peptide to go to its spectra."


class FigureView(QWidget):
    """A figure as SVG. ``zoom`` is relative to the figure's own size (1 = as matplotlib sizes
    it); ``fit`` scales it to the space the view has. Clicking an element with a ``gid`` emits
    :attr:`clicked` with it, and resting the pointer on one with a tooltip
    (:func:`hxex.report.hover.set_hover`) shows it; dragging emits :attr:`dragged` (how far, in
    pixels) and a double click :attr:`double_clicked` (where), which :class:`FigurePanel` turns
    into moving and magnifying."""

    clicked = Signal(str)
    clicked_row = Signal(dict)  # the clicked element's data-table row, when it has one
    # a right click on elements with data-table rows: those rows, and where (the screen's)
    rows_asked = Signal(list, QPoint)
    dragged = Signal(float, float)
    double_clicked = Signal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._renderer = QSvgRenderer(self)
        self._marks: list[Mark] = []  # what can be clicked or has a tooltip, and where
        self.figure: Figure | None = None
        self.zoom = 1.0
        self._panels: list[QRectF] = []  # each panel (matplotlib's axes) in the SVG's units
        self._press: QPointF | None = None
        self._dragging = False
        # a click acts once the double-click time has passed without a second click: the clicks
        # of a double-click (which magnifies) must not also open what is under the pointer
        self._clicked_mark: Mark | None = None
        self._after_double = False  # the release that ends a double-click is not a click
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(QApplication.styleHints().mouseDoubleClickInterval())
        self._click_timer.timeout.connect(self._click)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.white)  # figures are on white
        self.setPalette(pal)
        # tooltips through a filter: a Python event() override makes PySide6 6.7 hand the
        # mouse handlers a plain QEvent (no button())
        self.installEventFilter(self)

    def set_figure(self, fig: Figure, svg: bytes | None = None,
                   marks: list[Mark] | None = None) -> None:
        """Show ``fig`` (``svg`` and ``marks`` when already drawn, as by a worker thread)."""
        self.figure = fig
        self._marks = marks if marks is not None else figure_marks(fig)
        self._renderer.load(svg if svg is not None else figure_svg(fig))
        self._panels = self._panel_boxes(fig)
        self._resize()

    def _panel_boxes(self, fig: Figure) -> list[QRectF]:
        """Each panel (matplotlib's axes, where the figure placed it) with room for its tick
        labels, in the SVG's units. Taken from the figure: an SVG group's bounds include what
        the panel clips away (a band drawn across the whole axis), so they cannot be used."""
        view = self._view_box()
        boxes = []
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            pos = ax.get_position()  # figure fraction, from the bottom left
            left, right, top, bottom = PANEL_ROOM
            x0 = pos.x0 - left * pos.width
            x1 = pos.x1 + right * pos.width
            y1 = pos.y1 + top * pos.height
            y0 = pos.y0 - bottom * pos.height
            boxes.append(QRectF(view.x() + x0 * view.width(),
                                view.y() + (1 - y1) * view.height(),
                                (x1 - x0) * view.width(), (y1 - y0) * view.height()))
        return boxes

    def clear(self) -> None:
        self.figure, self._marks, self._panels = None, [], []
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

    def _view_box(self) -> QRectF:
        """The figure's own coordinates (the SVG's units: points from the top left)."""
        return self._renderer.viewBoxF()

    @property
    def ready(self) -> bool:
        """Whether the figure is drawn (and what can be clicked is known)."""
        return self.figure is not None

    def _figure_point(self, point: QPointF) -> QPointF:
        """``point`` (the view's pixels) in the SVG's units."""
        view = self._view_box()
        return QPointF(view.x() + point.x() * view.width() / self.width(),
                       view.y() + point.y() * view.height() / self.height())

    def panel_at(self, point: QPointF) -> QRectF | None:
        """The panel (axes) under ``point``, as a rectangle in the view's pixels; the smallest
        when several overlap (an inset); None between panels."""
        if not self._panels or self.width() == 0:
            return None
        at = self._figure_point(point)
        found = [b for b in self._panels if b.contains(at)]
        if not found:
            return None
        box = min(found, key=lambda b: b.width() * b.height())
        view = self._view_box()
        sx, sy = self.width() / view.width(), self.height() / view.height()
        return QRectF((box.x() - view.x()) * sx, (box.y() - view.y()) * sy,
                      box.width() * sx, box.height() * sy)

    def mark_at(self, point: QPointF, *, text: bool = False) -> Mark | None:
        """The drawn element under ``point`` (in the view's coordinates), the smallest when
        several overlap; ``text``: only those with a tooltip."""
        if not self._marks or self.width() == 0 or self.height() == 0:
            return None
        return mark_at(self._marks, point.x() / self.width(), point.y() / self.height(),
                       CLICK_ALLOWANCE / self.width(), CLICK_ALLOWANCE / self.height(),
                       text=text)

    def element_at(self, point: QPointF) -> str | None:
        """The ``gid`` of the drawn element under ``point`` (in the view's coordinates), the
        smallest when several overlap."""
        mark = self.mark_at(point)
        return mark.gid if mark is not None else None

    def rows_at(self, point: QPointF) -> list[dict]:
        """The data-table rows of every element under ``point`` (in the view's coordinates),
        top to bottom."""
        if not self._marks or self.width() == 0 or self.height() == 0:
            return []
        return rows_at(self._marks, point.x() / self.width(), point.y() / self.height(),
                       CLICK_ALLOWANCE / self.width(), CLICK_ALLOWANCE / self.height())

    def contextMenuEvent(self, event):  # noqa: N802 - Qt's name
        rows = self.rows_at(QPointF(event.pos()))
        if rows:
            self.rows_asked.emit(rows, event.globalPos())
            event.accept()
        else:
            event.ignore()

    def tooltip_at(self, point: QPointF) -> str:
        """The tooltip of the drawn element under ``point`` ('' when none has one)."""
        mark = self.mark_at(point, text=True)
        return mark.text if mark is not None else ""

    def eventFilter(self, watched, event):  # noqa: N802 - Qt's name
        if watched is self and event.type() == QEvent.Type.ToolTip:
            mark = self.mark_at(QPointF(event.pos()), text=True)
            if mark is None:
                QToolTip.hideText()
                event.ignore()
                return True
            left, top, width, height = mark.box  # gone once the pointer leaves the mark
            area = QRectF(left * self.width(), top * self.height(), width * self.width(),
                          height * self.height()).adjusted(
                -CLICK_ALLOWANCE, -CLICK_ALLOWANCE, CLICK_ALLOWANCE, CLICK_ALLOWANCE)
            QToolTip.showText(event.globalPos(), mark.text, self, area.toAlignedRect())
            return True
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event):  # noqa: N802 - Qt's name
        if event.button() == Qt.MouseButton.LeftButton:
            self._press, self._dragging = event.globalPosition(), False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt's name
        if self._press is not None and event.buttons() & Qt.MouseButton.LeftButton:
            move = event.globalPosition() - self._press
            if self._dragging or move.manhattanLength() >= DRAG_START:
                if not self._dragging:
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self._dragging = True
                self._press = event.globalPosition()
                self.dragged.emit(move.x(), move.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt's name
        if event.button() == Qt.MouseButton.LeftButton:
            if self._after_double:
                self._after_double = False
            elif not self._dragging:  # a click, not the end of a drag: acted on unless a
                mark = self.mark_at(event.position())  # second click makes it a double-click
                if mark is not None:
                    self._clicked_mark = mark
                    self._click_timer.start()
            self._press, self._dragging = None, False
            self.unsetCursor()
        super().mouseReleaseEvent(event)

    def _click(self) -> None:
        mark, self._clicked_mark = self._clicked_mark, None
        if mark is not None:
            self.clicked.emit(mark.gid)
            if mark.row:
                self.clicked_row.emit(mark.row)

    def mouseDoubleClickEvent(self, event):  # noqa: N802 - Qt's name
        if event.button() == Qt.MouseButton.LeftButton:
            self._click_timer.stop()  # its first click was not a click on what is there
            self._clicked_mark, self._after_double = None, True
            self.double_clicked.emit(event.position())
        super().mouseDoubleClickEvent(event)


class RasterView(FigureView):
    """A figure drawn by matplotlib itself (its Agg renderer) at the size shown, sharp at every
    zoom: drawn again, in a worker thread, when the size changes (the picture drawn before is
    scaled meanwhile). It shows exactly what a saved PNG shows, clipping included, which Qt's
    SVG renderer does not draw (on trial against :class:`FigureView`; HISTORY 237)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._marks_known = False  # found when first drawn (in the worker thread)
        self._size_pt = QSize(0, 0)
        self._drawing = Latest(self, drawing_pool())
        self._again = QTimer(self)
        self._again.setSingleShot(True)
        self._again.setInterval(REDRAW_AFTER_MS)
        self._again.timeout.connect(self._render)

    def set_figure(self, fig: Figure, svg: bytes | None = None,
                   marks: list[Mark] | None = None) -> None:
        self.figure = fig
        self._marks, self._marks_known = list(marks or []), marks is not None
        w, h = fig.get_size_inches()
        self._size_pt = QSize(round(w * 72), round(h * 72))  # as the SVG's points
        self._image = None
        self._panels = self._panel_boxes(fig)
        self._resize()

    def clear(self) -> None:
        self.figure, self._marks, self._panels = None, [], []
        self._image, self._marks_known, self._size_pt = None, False, QSize(0, 0)
        self.setFixedSize(QSize(0, 0))
        self.update()

    def _view_box(self) -> QRectF:
        return QRectF(0, 0, self._size_pt.width(), self._size_pt.height())

    @property
    def ready(self) -> bool:
        return self.figure is not None and self._image is not None

    def figure_size(self) -> QSize:
        return QSize(self._size_pt)

    def _resize(self) -> None:
        super()._resize()
        if self.figure is not None:
            self._again.start()  # once the zooming settles

    def _render(self) -> None:
        fig = self.figure
        if fig is None or self.width() == 0:
            return
        ratio = self.devicePixelRatioF()
        pixels = self.width() * ratio * self.height() * ratio
        scale = min(1.0, (MAX_PIXELS / pixels) ** 0.5) if pixels else 1.0
        dpi = 72 * self.zoom * ratio * scale
        find = not self._marks_known

        def work():
            marks = figure_marks(fig) if find else None
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            return buf.getvalue(), marks

        self._drawing.submit(work, self._rendered, lambda message: log.error(
            "could not draw the figure: %s", message))

    def _rendered(self, done) -> None:
        png, marks = done
        image = QImage.fromData(png, "PNG")
        image.setDevicePixelRatio(self.devicePixelRatioF())
        self._image = image
        if marks is not None:
            self._marks, self._marks_known = marks, True
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt's name
        if self._image is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRectF(self.rect()), self._image)
        painter.end()


class FigurePanel(QWidget):
    """A :class:`FigureView` in a scroll area, with buttons to zoom, fit and save. Give it a
    figure (:meth:`show_figure`) or a function that draws one (:meth:`draw`, run in a worker
    thread)."""

    drawn = Signal()  # a figure is shown
    failed = Signal(str)
    clicked = Signal(str)  # the gid of what was clicked (whichever view draws the figure)
    clicked_row = Signal(dict)  # the data-table row of what was clicked, when it has one
    rows_asked = Signal(list, QPoint)  # right-clicked: the rows there, and where

    def __init__(self, parent=None, pool: QThreadPool | None = None,
                 render: str = "matplotlib", hint: str = HINT):
        super().__init__(parent)
        self.render = render
        self.view = VIEWS[render]()
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.view)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.viewport().installEventFilter(self)  # the wheel zooms (not scrolls)
        self._connect(self.view)
        self.magnified = False  # a panel fills the view (double-click)
        self.save_name = ""  # the file name the save dialog suggests (none: it asks)
        self.status = QLabel()
        self._fitted = True
        self._drawing = Latest(self, pool or drawing_pool())

        self.bar = QWidget()
        buttons = QHBoxLayout(self.bar)
        buttons.setContentsMargins(0, 0, 0, 0)
        for text, tip, slot in (("Fit page", "show the whole figure in the space there is",
                                 self.fit),
                                ("−", "zoom out (Ctrl and the mouse wheel)", self.zoom_out),
                                ("+", "zoom in (Ctrl and the mouse wheel)", self.zoom_in),
                                ("100 %", "the figure's own size", self.actual_size),
                                ("Save figure…", "save as SVG, PDF or PNG", self.save_dialog)):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        self.save_button = button  # the last one
        self._bar_layout = buttons
        self._options: list[QWidget] = []  # a figure's own choices, after the buttons
        buttons.addStretch(1)
        buttons.addWidget(self.status)
        # the figure's title, readable at any zoom: centred on the whole bar (so on the figure
        # below), moved aside only to clear the buttons; placed by hand, outside the layout
        self.title = QLabel(self.bar)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        self._title_text = ""
        self.bar.installEventFilter(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.bar)
        layout.addWidget(self.scroll)
        # how to move about, under the figure (as a tooltip it covered the figure: owner)
        self.hint = QLabel(hint)
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: palette(placeholder-text);")
        layout.addWidget(self.hint)
        self.save_button.setEnabled(False)

    # --- showing ---------------------------------------------------------------------------

    def _connect(self, view: FigureView) -> None:
        view.dragged.connect(self._drag)
        view.double_clicked.connect(self._double_clicked)
        view.clicked.connect(self.clicked)
        view.clicked_row.connect(self.clicked_row)
        view.rows_asked.connect(self.rows_asked)

    def set_render(self, render: str) -> None:
        """Draw with Qt's SVG renderer (``'svg'``) or with matplotlib (``'matplotlib'``); the
        figure shown is shown again the new way."""
        if render == self.render:
            return
        figure, zoom, fitted = self.view.figure, self.view.zoom, self._fitted
        old = self.scroll.takeWidget()
        self.view = VIEWS[render]()
        self._connect(self.view)
        self.scroll.setWidget(self.view)
        self.render = render
        if old is not None:
            old.deleteLater()
        if figure is not None:
            self.view.set_figure(figure)
            if fitted:
                self.fit()
            else:
                self.view.set_zoom(zoom)

    def show_figure(self, fig: Figure, svg: bytes | None = None,
                    marks: list[Mark] | None = None) -> None:
        if self.render == "svg":
            self.view.set_figure(fig, svg, marks)
        else:
            self.view.set_figure(fig, marks=marks)
        self.set_title(figure_title(fig))
        self.save_button.setEnabled(True)
        self.status.setText("")
        if self._fitted:
            self.fit()
        self.drawn.emit()

    def draw(self, draw: Callable[[], Figure]) -> None:
        """Draw a figure with ``draw`` in a worker thread and show it when done; a newer call
        replaces an older one still drawing."""
        svg = self.render == "svg"  # the matplotlib view draws its own picture

        def work():
            fig = draw()
            if not svg:
                return fig, None, None
            return fig, figure_svg(fig), figure_marks(fig)

        self.status.setText("drawing…")
        self._drawing.submit(work, lambda done: self.show_figure(*done), self._failed)

    @property
    def drawing(self) -> bool:
        return self._drawing.running

    def clear(self, message: str = "") -> None:
        """Show no figure (``message`` in its place)."""
        self.view.clear()
        self.set_title("")
        self.save_button.setEnabled(False)
        self.status.setText(message)

    def _failed(self, message: str) -> None:
        self.status.setText("could not draw the figure")
        log.error("could not draw the figure: %s", message)
        self.failed.emit(message)

    # --- zooming ---------------------------------------------------------------------------

    def fit(self) -> None:
        self._fitted, self.magnified = True, False
        margin = 4
        self.view.fit(self.scroll.viewport().size() - QSize(margin, margin))

    def zoom_in(self) -> None:
        self.zoom_by(ZOOM_STEP)

    def zoom_out(self) -> None:
        self.zoom_by(1 / ZOOM_STEP)

    def actual_size(self) -> None:
        self._fitted, self.magnified = False, False
        self.view.set_zoom(1.0)

    def zoom_by(self, factor: float, at: QPointF | None = None) -> None:
        """Zoom by ``factor``, keeping the point of the figure under ``at`` (the viewport's
        pixels; default its centre) where it is."""
        self._fitted, self.magnified = False, False
        port = self.scroll.viewport()
        if at is None:
            at = QPointF(port.width() / 2, port.height() / 2)
        on_view = QPointF(self.view.mapFrom(port, at.toPoint()))
        fx = on_view.x() / max(self.view.width(), 1)
        fy = on_view.y() / max(self.view.height(), 1)
        self.view.set_zoom(self.view.zoom * factor)
        self._scroll_to(fx * self.view.width() - at.x(), fy * self.view.height() - at.y())

    def _scroll_to(self, x: float, y: float) -> None:
        """Scroll so the view's pixel (x, y) is at the viewport's top left (as far as the
        scroll bars go)."""
        self.scroll.widget().adjustSize()
        for bar, value in ((self.scroll.horizontalScrollBar(), x),
                           (self.scroll.verticalScrollBar(), y)):
            bar.setRange(0, max(0, round(self._extent(bar) - self._port(bar))))
            bar.setValue(round(value))

    def _extent(self, bar) -> int:
        return self.view.width() if bar is self.scroll.horizontalScrollBar() else \
            self.view.height()

    def _port(self, bar) -> int:
        port = self.scroll.viewport()
        return port.width() if bar is self.scroll.horizontalScrollBar() else port.height()

    def _drag(self, dx: float, dy: float) -> None:
        h, v = self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()
        h.setValue(round(h.value() - dx))
        v.setValue(round(v.value() - dy))

    def _double_clicked(self, point: QPointF) -> None:
        """Magnify the panel under ``point`` to fill the view; again, the whole figure."""
        if self.magnified:
            self.fit()
            return
        box = self.view.panel_at(point)
        if box is None:
            return
        port = self.scroll.viewport().size()
        margin = 0.96  # a little space around the panel
        factor = margin * min(port.width() / box.width(), port.height() / box.height())
        self._fitted = False
        before = self.view.zoom
        self.view.set_zoom(before * factor)
        scale = self.view.zoom / before
        centre = box.center() * scale
        self._scroll_to(centre.x() - port.width() / 2, centre.y() - port.height() / 2)
        self.magnified = True

    def resizeEvent(self, event):  # noqa: N802 - Qt's name
        super().resizeEvent(event)
        if self._fitted and self.view.figure is not None:
            self.fit()

    def showEvent(self, event):  # noqa: N802 - Qt's name
        super().showEvent(event)  # a figure drawn in a hidden tab was fitted to no space
        if self._fitted and self.view.figure is not None:
            self.fit()

    def add_option(self, widget: QWidget) -> None:
        """A choice of this figure's own (the uptake page's time axis), after the buttons."""
        self._bar_layout.insertWidget(self._bar_layout.indexOf(self.save_button) + 1
                                      + len(self._options), widget)
        self._options.append(widget)
        self._place_title()

    def set_title(self, text: str) -> None:
        self._title_text = text
        self.title.setToolTip(text)
        self._place_title()

    def _place_title(self) -> None:
        """Centre the title on the bar, clear of the buttons on the left and the status on
        the right; shortened with "…" when there is no room for all of it."""
        gap = 12
        last = self._options[-1] if self._options else self.save_button
        left = last.geometry().right() + gap
        right = (self.status.geometry().left() - gap if self.status.text()
                 else self.bar.width() - gap)
        room = max(0, right - left)
        metrics = self.title.fontMetrics()
        text = metrics.elidedText(self._title_text, Qt.TextElideMode.ElideRight, room)
        width = min(metrics.horizontalAdvance(text) + 4, room)
        x = round(self.bar.width() / 2 - width / 2)
        x = max(left, min(x, right - width))
        self.title.setText(text)
        self.title.setGeometry(x, 0, width, self.bar.height())

    def eventFilter(self, watched, event):  # noqa: N802 - Qt's name
        if watched is self.bar:
            if event.type() == QEvent.Type.Resize:
                self._place_title()
            elif event.type() == QEvent.Type.LayoutRequest:  # the status changed: once laid out
                QTimer.singleShot(0, self._place_title)
            return False
        if event.type() == QEvent.Type.Wheel and self.view.figure is not None:
            notches = event.angleDelta().y() / 120
            if notches:
                self.zoom_by(WHEEL_STEP ** notches, event.position())
            return True  # the wheel zooms; dragging and the scroll bars move
        return super().eventFilter(watched, event)

    # --- saving ----------------------------------------------------------------------------

    def save_dialog(self) -> None:
        if self.view.figure is None:
            return
        filters = ";;".join(f"{label} (*.{fmt})" for fmt, label in SAVE_FORMATS.items())
        path, chosen = QFileDialog.getSaveFileName(self, "Save figure", self.save_name, filters)
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


VIEWS = {"svg": FigureView, "matplotlib": RasterView}  # how the window draws its figures
