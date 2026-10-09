# SPDX-License-Identifier: GPL-3.0-or-later
"""Time labels of the two input formats, parsed in one place. Times are in seconds."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from .errors import ReadError

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600}

# Default time (s) recorded for the fully deuterated control; the user sets it per project.
DEFAULT_FULLDEUT_TIME = 1e6


def _to_seconds(number: str, unit: str, label: str) -> float:
    """``number`` (decimal text) times the unit's seconds, rounded once to the nearest float.

    Multiplying as decimals keeps ``4.1m`` at exactly 246.0 (``4.1 * 60`` in floating point is
    245.99999999999997), so equal times written differently compare equal.
    """
    try:
        seconds = _UNIT_SECONDS[unit[:1].lower()]
    except KeyError:
        raise ReadError(f"unknown time unit {unit!r} in label {label!r} (use s, m or h)") from None
    try:
        value = Decimal(number)
    except InvalidOperation:
        raise ReadError(f"cannot read a number from time label {label!r}") from None
    if not value.is_finite():
        raise ReadError(f"time in label {label!r} is not a finite number")
    return float(value * seconds)


def parse_hxexpress_time(label: object, fulldeut_time: float = DEFAULT_FULLDEUT_TIME) -> float:
    """Seconds for an HX-Express column label: ``undeut``, ``TD``, ``60 s``, ``5 min``, ``2 h``.

    Labels starting with ``undeut`` are 0 and labels starting with ``TD`` are ``fulldeut_time``.
    As in pyHXExpress, digits after the decimal point of a time of 1 or more are replicate
    markers, not fractions: ``1.2 min`` is 60 s (the second spectrum at that time), while a
    time below 1 such as ``0.5 min`` keeps its fraction.
    """
    if not isinstance(label, str):
        raise ReadError(f"time label {label!r} is not text")
    text = label.strip()
    if text.lower().startswith("undeut"):
        return 0.0
    if text.upper().startswith("TD"):
        return float(fulldeut_time)
    parts = text.split()
    if len(parts) < 2:
        raise ReadError(f"time label {label!r} needs a number and a unit, e.g. '5 min'")
    number, unit = parts[0], parts[-1]
    whole = number.split(".")[0]
    return _to_seconds(number if whole == "0" else whole, unit, label)


def parse_specexport_time(token: str, fulldeut_time: float = DEFAULT_FULLDEUT_TIME) -> float:
    """Seconds for a SpecExport file-name time: ``Non``, ``Full``, ``60s``, ``5m``, ``2h``."""
    if token == "Non":
        return 0.0
    if token == "Full":
        return float(fulldeut_time)
    return _to_seconds(token[:-1], token[-1:], token)
