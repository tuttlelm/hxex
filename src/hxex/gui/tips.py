# SPDX-License-Identifier: GPL-3.0-or-later
"""Tooltip text laid out for reading: wrapped at a fixed width, one line per item.

Qt shows a plain-text tooltip on one line until it is wider than the screen, so a long one
runs across the window. :func:`tip` wraps the text at :data:`TIP_WIDTH` characters (between
words) and starts each item on its own line: the parts of the text separated by a semicolon
outside brackets (a semicolon inside parentheses or square brackets belongs to the aside, so it
stays on its line) and by line breaks already in the text. The result is still plain text, so
the tooltip is not reflowed by Qt and no characters need escaping. :func:`style_tooltips` gives
the boxes more room and a neutral cream background (owner, 2026-11-11: Qt's pale yellow looked
dated), a warm dark grey on a dark desktop; it alone imports Qt, when called.
"""

from __future__ import annotations

import textwrap

TIP_WIDTH = 60  # characters: the widest a tooltip line runs before wrapping

_OPEN, _CLOSE = "([{", ")]}"


def items(text: str) -> list[str]:
    """The items of ``text``: split at line breaks and at semicolons outside brackets, each
    item keeping its semicolon."""
    parts: list[str] = []
    for line in text.split("\n"):
        depth, start = 0, 0
        for i, char in enumerate(line):
            if char in _OPEN:
                depth += 1
            elif char in _CLOSE:
                depth = max(depth - 1, 0)
            elif char == ";" and depth == 0:
                parts.append(line[start:i + 1])
                start = i + 1
        if start == 0 or line[start:].strip():  # an empty line stays (a paragraph break)
            parts.append(line[start:])
    return [part.strip() for part in parts]


def tip(text: str | None, width: int = TIP_WIDTH) -> str:
    """``text`` wrapped at ``width`` characters, each item (:func:`items`) starting on its own
    line. Empty text stays empty (Qt's 'no tooltip'); a word longer than ``width`` (a path) is
    kept whole."""
    if not text:
        return text or ""
    lines: list[str] = []
    for item in items(str(text)):
        lines.extend(textwrap.wrap(item, width, break_long_words=False, break_on_hyphens=False)
                     or [""])
    return "\n".join(lines)


def rich_tip(text: str | None, width: int = TIP_WIDTH) -> str:
    """``text`` laid out as :func:`tip`, as rich text, with the words between ``**`` in bold
    (Qt shows a tooltip that starts with a tag as rich text). Each line is kept whole
    (``white-space: pre``): Qt word-wraps a rich-text tooltip again at about half these lines'
    width, which broke each of them in two (owner, 2026-11-12: "the wrapping is a bit weird")."""
    import html
    import re

    if not text:
        return text or ""
    # wrapped without the markers (they take no room), the bold phrases found again after
    bold = re.findall(r"\*\*(.+?)\*\*", text)
    laid = html.escape(tip(text.replace("**", ""), width), quote=False)
    for phrase in bold:
        words = [re.escape(html.escape(w, quote=False)) for w in phrase.split()]
        laid = re.sub(r"\s+".join(words), lambda m: f"<b>{m.group(0)}</b>", laid, count=1)
    return "<qt><p style='white-space:pre'>" + laid.replace("\n", "<br>") + "</p></qt>"


# the tooltip box: (background, text, border) for a light and a dark desktop
TIP_COLOURS = {False: ("#f8f5ee", "#2a2723", "#cfc6b4"),
               True: ("#2e2b27", "#ece6da", "#5a5348")}
TIP_PADDING = "2px 4px"  # top and bottom, left and right (on Qt's own 2 px margin)


def tooltip_style(dark: bool) -> str:
    """The style sheet of the tooltip box on a light or ``dark`` desktop."""
    background, text, border = TIP_COLOURS[bool(dark)]
    return (f"QToolTip {{ background-color: {background}; color: {text}; "
            f"border: 1px solid {border}; padding: {TIP_PADDING}; }}")


def style_tooltips(app) -> None:
    """Give ``app``'s tooltips :func:`tooltip_style`, now and whenever the desktop's colour
    scheme changes (added to the application's own style sheet)."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication

    from .icon import dark_scheme

    base = app.styleSheet()

    def apply(dark: bool) -> None:
        app.setStyleSheet((base + "\n" if base else "") + tooltip_style(dark))

    apply(dark_scheme())
    QGuiApplication.styleHints().colorSchemeChanged.connect(
        lambda scheme: apply(scheme == Qt.ColorScheme.Dark))
