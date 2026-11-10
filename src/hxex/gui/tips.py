# SPDX-License-Identifier: GPL-3.0-or-later
"""Tooltip text laid out for reading: wrapped at a fixed width, one line per item.

Qt shows a plain-text tooltip on one line until it is wider than the screen, so a long one
runs across the window. :func:`tip` wraps the text at :data:`TIP_WIDTH` characters (between
words) and starts each item on its own line: the parts of the text separated by a semicolon
outside brackets (a semicolon inside parentheses or square brackets belongs to the aside, so it
stays on its line) and by line breaks already in the text. The result is still plain text, so
the tooltip is not reflowed by Qt and no characters need escaping. Nothing here imports Qt.
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
