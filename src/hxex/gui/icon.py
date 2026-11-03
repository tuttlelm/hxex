# SPDX-License-Identifier: GPL-3.0-or-later
"""The window's icon: the hxex logo in the version for the desktop's colour scheme (dark
lettering on a light desktop, light lettering on a dark one), changed when the scheme changes.

The logos in ``icons/`` are copies of ``docs/images/hxex_logo*.svg`` (``make_logo.py`` writes
both). They are drawn here at the usual icon sizes, centred in a transparent square, with Qt's
SVG renderer, so no icon plugin is needed.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

ICONS = Path(__file__).resolve().parent / "icons"
SIZES = (16, 24, 32, 48, 64, 128, 256)


def dark_scheme() -> bool:
    """Whether the desktop is in dark mode (as Qt reports it; unknown counts as light)."""
    return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark


def _renderer(dark: bool | None) -> QSvgRenderer:
    if dark is None:
        dark = dark_scheme()
    return QSvgRenderer(str(ICONS / ("hxex_logo_dark.svg" if dark else "hxex_logo.svg")))


def logo_pixmap(width: int, dark: bool | None = None, ratio: float = 1.0) -> QPixmap:
    """The logo ``width`` pixels wide on a transparent background, for a dark or light desktop
    (``None``: the current scheme's); ``ratio`` is the screen's device pixel ratio, so the logo
    stays sharp on a high-resolution screen."""
    renderer = _renderer(dark)
    own = renderer.defaultSize()
    height = round(width * own.height() / own.width())
    pixmap = QPixmap(round(width * ratio), round(height * ratio))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(0, 0, pixmap.width(), pixmap.height()))
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def logo_icon(dark: bool | None = None) -> QIcon:
    """The logo as an icon, for a dark or light desktop (``None``: the current scheme's)."""
    renderer = _renderer(dark)
    own = renderer.defaultSize()
    icon = QIcon()
    for size in SIZES:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        scale = size / max(own.width(), own.height())
        w, h = own.width() * scale, own.height() * scale
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter, QRectF((size - w) / 2, (size - h) / 2, w, h))
        painter.end()
        icon.addPixmap(pixmap)
    return icon


def follow_scheme(apply) -> None:
    """Call ``apply(icon)`` now and whenever the desktop's colour scheme changes."""
    apply(logo_icon())
    QGuiApplication.styleHints().colorSchemeChanged.connect(
        lambda scheme: apply(logo_icon(scheme == Qt.ColorScheme.Dark)))
