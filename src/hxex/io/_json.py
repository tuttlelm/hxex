# SPDX-License-Identifier: GPL-3.0-or-later
"""JSON as strict readers expect it: a value that is not finite (NaN, infinity) is written as
null, never as the ``NaN`` and ``Infinity`` tokens Python's :mod:`json` writes by default.
Files written before this keep their tokens; Python's :mod:`json` still reads them."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping


def _finite(value):
    if isinstance(value, float):  # numpy's float64 too
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


def dumps(value, **kwargs) -> str:
    """:func:`json.dumps` of ``value`` with every non-finite number as null."""
    return json.dumps(_finite(value), allow_nan=False, **kwargs)
