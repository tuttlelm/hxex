# SPDX-License-Identifier: GPL-3.0-or-later
"""Colours for a project file shown as text (the File tab): headings, setting names, text,
numbers and true/false, and comments, in a light and a dark set chosen from the window's
palette. A switched-off setting (``# key = value``, ``#~ ...``) is a comment, so it is grey
like any other, which makes the settings in use stand out."""

from __future__ import annotations

import re

from PySide6.QtGui import QColor, QFont, QPalette, QSyntaxHighlighter, QTextCharFormat

# (light, dark): readable on a white and on a near-black background
COLOURS = {
    "heading": ("#1f4e9c", "#82b1ff"),
    "key": ("#7a3e9d", "#d7a6f2"),
    "string": ("#2e7d32", "#8fd18f"),
    "number": ("#a85400", "#f0a868"),
    "comment": ("#808080", "#8c8c8c"),
}

_HEADING = re.compile(r"^\s*\[\[?[^\]#]*\]\]?")
_KEY = re.compile(r'^\s*((?:[A-Za-z0-9_-]+|"[^"]*"|\'[^\']*\')'
                  r'(?:\s*\.\s*(?:[A-Za-z0-9_-]+|"[^"]*"|\'[^\']*\'))*)\s*=')
_WORD = re.compile(r"(?<![\w.-])(true|false|[+-]?(?:\d[\d_]*(?:\.[\d_]+)?(?:[eE][+-]?\d+)?|inf|nan)"
                   r"|\d{4}-\d{2}-\d{2}(?:[T ][\d:.]+)?(?:Z|[+-]\d{2}:\d{2})?)(?![\w.-])")
_IN_LONG = 1  # block state: inside a """ or ''' text going on over several lines
_IN_LONG_LITERAL = 2


def dark(palette: QPalette) -> bool:
    return palette.color(QPalette.ColorRole.Base).lightness() < 128


class TomlColours(QSyntaxHighlighter):
    """Colours the TOML text of a QTextDocument; :meth:`use_palette` picks the light or dark
    set (called again when the window's palette changes)."""

    def __init__(self, document, palette: QPalette):
        super().__init__(document)
        self.formats: dict[str, QTextCharFormat] = {}
        self.use_palette(palette, redraw=False)

    def use_palette(self, palette: QPalette, redraw: bool = True) -> None:
        which = 1 if dark(palette) else 0
        self.formats = {}
        for name, pair in COLOURS.items():
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(pair[which]))
            if name == "heading":
                fmt.setFontWeight(QFont.Weight.Bold)
            if name == "comment":
                fmt.setFontItalic(True)
            self.formats[name] = fmt
        if redraw:
            self.rehighlight()

    def highlightBlock(self, text: str) -> None:  # noqa: N802 - Qt's name
        self.setCurrentBlockState(0)
        start = 0
        state = self.previousBlockState()
        if state in (_IN_LONG, _IN_LONG_LITERAL):  # the rest of a text over several lines
            quote = '"""' if state == _IN_LONG else "'''"
            end = text.find(quote)
            if end < 0:
                self.setFormat(0, len(text), self.formats["string"])
                self.setCurrentBlockState(state)
                return
            start = end + 3
            self.setFormat(0, start, self.formats["string"])
        if start == 0:
            if text.lstrip().startswith("#"):
                self.setFormat(0, len(text), self.formats["comment"])
                return
            heading = _HEADING.match(text)
            if heading:
                self.setFormat(0, heading.end(), self.formats["heading"])
                start = heading.end()
            else:
                key = _KEY.match(text)
                if key:
                    self.setFormat(key.start(1), key.end(1) - key.start(1), self.formats["key"])
                    start = key.end()
        self._values(text, start)

    def _values(self, text: str, i: int) -> None:
        """Texts, numbers, true/false and the comment from ``i`` to the end of the line."""
        plain = i  # where the stretch outside texts began
        while i < len(text):
            char = text[i]
            if char == "#":
                self._words(text, plain, i)
                self.setFormat(i, len(text) - i, self.formats["comment"])
                return
            if char in "\"'":
                self._words(text, plain, i)
                long = text.startswith(char * 3, i)
                quote = char * 3 if long else char
                j = i + len(quote)
                while True:
                    end = text.find(quote, j)
                    if end < 0:
                        break
                    if char == '"' and _escaped(text, end):
                        j = end + 1
                        continue
                    break
                if end < 0:
                    self.setFormat(i, len(text) - i, self.formats["string"])
                    if long:
                        self.setCurrentBlockState(_IN_LONG if char == '"' else _IN_LONG_LITERAL)
                    return
                end += len(quote)
                self.setFormat(i, end - i, self.formats["string"])
                i = plain = end
                continue
            i += 1
        self._words(text, plain, len(text))

    def _words(self, text: str, start: int, end: int) -> None:
        for m in _WORD.finditer(text, start, end):
            self.setFormat(m.start(), m.end() - m.start(), self.formats["number"])


def _escaped(text: str, at: int) -> bool:
    """Whether the quote at ``at`` follows an odd number of backslashes."""
    n = 0
    while at - n - 1 >= 0 and text[at - n - 1] == "\\":
        n += 1
    return n % 2 == 1
