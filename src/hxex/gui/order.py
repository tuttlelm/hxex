# SPDX-License-Identifier: GPL-3.0-or-later
"""The order the window lists text in (no Qt: the Results tree's model imports it too)."""

from __future__ import annotations

import re


def natural_key(text) -> tuple:
    """``text`` ordered as a person reads it: its runs of digits by their value, so that residue
    ranges go 2-12, 13-30, 108-127 (not 108-127, 13-30, 2-12), the rest case-insensitively."""
    return tuple((0, int(part), "") if part.isdigit() else (1, 0, part)
                 for part in re.split(r"(\d+)", str(text).lower()) if part)
