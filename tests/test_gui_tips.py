# SPDX-License-Identifier: GPL-3.0-or-later
"""Tooltip text: wrapped at a fixed width, one line per item (hxex.gui.tips; no Qt needed)."""

from hxex.gui.tips import TIP_WIDTH, items, tip


def test_long_text_wraps_between_words_at_the_width():
    text = "the spectrum summed over the new window is drawn beside the one the results hold " * 3
    lines = tip(text).split("\n")
    assert len(lines) > 1 and all(len(line) <= TIP_WIDTH for line in lines)
    assert " ".join(lines) == " ".join(text.split())  # only the spaces changed


def test_each_item_starts_a_line_and_keeps_its_semicolon():
    assert tip("click: sort by the column; right-click: filter by it") == (
        "click: sort by the column;\nright-click: filter by it")


def test_a_semicolon_inside_brackets_stays_on_its_line():
    text = "the fragment ion (c-1:4: ion type and length; M: the precursor) or empty"
    assert items(text) == [text]
    assert items("a table ([input] x; made by the first save); the rest") == [
        "a table ([input] x; made by the first save);", "the rest"]


def test_line_breaks_already_there_are_kept():
    assert tip("first\nsecond") == "first\nsecond"
    assert tip("a paragraph\n\nanother") == "a paragraph\n\nanother"


def test_short_empty_and_unbreakable_text():
    assert tip("empty the console") == "empty the console"
    assert tip("") == "" and tip(None) == ""
    path = "/a/very/long/path/without/any/spaces/" * 3
    assert tip(path) == path  # a word longer than the width is kept whole
    assert tip("trailing;") == "trailing;"
    assert tip(tip("x; " * 40)) == tip("x; " * 40)  # wrapping twice changes nothing
