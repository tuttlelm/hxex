# SPDX-License-Identifier: GPL-3.0-or-later
"""The fixed-width font of the window's console, command line, settings names and sequence
view. ``QFont("monospace")`` is a fontconfig alias on Linux (DejaVu Sans Mono or the like)
but no font on Windows, where Qt then gave Tahoma, a proportional font, so nothing aligned
(owner, 2026-11-16, HISTORY). :func:`fixed_font` takes the first installed family of
:data:`FAMILIES`, else Qt's system fixed font (Courier New on Windows), with the monospace
style hint. :func:`even_width` says whether a font draws some characters at the width of a
letter: glyphs a font lacks are borrowed from another font at that font's width, which
breaks a line of them (the sequence view's coverage blocks)."""

from __future__ import annotations

from PySide6.QtGui import QFont, QFontDatabase, QFontMetricsF

# in order of preference: Windows (Cascadia Mono since Windows 11 and Terminal, Consolas),
# macOS (Menlo), Linux (DejaVu Sans Mono, Liberation Mono)
FAMILIES = ("Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono", "Liberation Mono")
_chosen: str | None = None


def fixed_family() -> str:
    """The family of :func:`fixed_font` (the first of :data:`FAMILIES` installed, else the
    system's fixed font)."""
    global _chosen
    if _chosen is None:
        installed = set(QFontDatabase.families())
        _chosen = next((f for f in FAMILIES if f in installed),
                       QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family())
    return _chosen


def fixed_font(point_size: int | None = None) -> QFont:
    """A fixed-width font (the application's size unless ``point_size`` is given)."""
    font = QFont(fixed_family()) if point_size is None else QFont(fixed_family(), point_size)
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    return font


def even_width(font: QFont, text: str, like: str = "A") -> bool:
    """Whether every character of ``text`` is as wide as ``like`` in ``font``."""
    metrics = QFontMetricsF(font)
    width = metrics.horizontalAdvance(like)
    return all(abs(metrics.horizontalAdvance(ch) - width) < 0.01 for ch in text)
