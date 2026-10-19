# SPDX-License-Identifier: GPL-3.0-or-later
"""Times and control labels, parsed in one place. Times are in seconds.

The two folder formats label their controls in the time column (``undeut``/``TD`` in an
HX-Express sheet, ``Non``/``Full`` in a SpecExport file name): the readers turn those labels
into a *role* (``undeut`` or ``fulldeut``, :data:`hxex.spectra.ROLES`) and leave the time
blank (NaN), so that nothing downstream identifies a control by a number. Every other time
is a number of seconds, written with a unit or without one (:func:`parse_time`).
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

from ..spectra import FULLDEUT, UNDEUT
from .errors import ReadError

# seconds per unit, by the whole unit token (lower case)
_UNIT_SECONDS: dict[str, Decimal] = {
    **dict.fromkeys(("ms", "msec", "msecs", "millisecond", "milliseconds"), Decimal("0.001")),
    **dict.fromkeys(("s", "sec", "secs", "second", "seconds"), Decimal(1)),
    **dict.fromkeys(("m", "min", "mins", "minute", "minutes"), Decimal(60)),
    **dict.fromkeys(("h", "hr", "hrs", "hour", "hours"), Decimal(3600)),
}
UNITS_TEXT = "s, sec, m, min, h, hr or ms (or their long forms)"
_SPECEXPORT_TIME = re.compile(r"[0-9]*\.?[0-9]+[smhSMH]")
_PART = re.compile(r"\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([a-zA-Z]+)\s*")


def is_control_label(label: str) -> bool:
    """Whether a time label (either format) names a control rather than a time."""
    text = str(label).strip()
    return text.lower().startswith("undeut") or text.upper().startswith("TD") or text in (
        "Non", "Full")


def _to_seconds(number: str, unit: str, label: str) -> float:
    """``number`` (decimal text) times the unit's seconds, rounded once to the nearest float.

    Multiplying as decimals keeps ``4.1m`` at exactly 246.0 (``4.1 * 60`` in floating point is
    245.99999999999997), so equal times written differently compare equal. The unit is matched
    as a whole word (``s``, ``sec``, ``min``, ``hours``, ``ms``...), case-insensitively; any
    other unit raises :class:`ReadError`.
    """
    try:
        seconds = _UNIT_SECONDS[unit.strip().lower()]
    except KeyError:
        raise ReadError(f"unknown time unit {unit!r} in {label!r} (use {UNITS_TEXT})") from None
    try:
        value = Decimal(number)
    except InvalidOperation:
        raise ReadError(f"cannot read a number from time {label!r}") from None
    if not value.is_finite():
        raise ReadError(f"time {label!r} is not a finite number")
    return float(value * seconds)


def parse_time(value: object, where: str = "time") -> float:
    """Seconds of a time written in a project file or given from Python: a number (seconds), or
    text with units, ``"5 min"``, ``"20 h"``, ``"250 ms"``, ``"1.5 min"`` (90 s: a decimal is a
    fraction here), or several parts run together, ``"1h21m"``, ``"10m30s"``. Units are
    ``s``, ``sec``, ``m``, ``min``, ``h``, ``hr``, ``ms`` and their long forms, case-insensitive;
    a bare number of text form (``"60"``) is seconds. Raises :class:`ReadError` naming
    ``where`` for anything else (a boolean, an infinite number, an unknown unit)."""
    if isinstance(value, bool):
        raise ReadError(f"{where}: expected a time (seconds, or text such as \"5 min\"), got "
                        f"{value!r}")
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ReadError(f"{where}: a time must be a finite number of seconds, got {value!r}")
        return float(value)
    if not isinstance(value, str):
        raise ReadError(f"{where}: expected a time (seconds, or text such as \"5 min\"), got "
                        f"{value!r}")
    text = value.strip()
    if not text:
        raise ReadError(f"{where}: an empty time")
    try:
        bare = Decimal(text)
    except InvalidOperation:
        bare = None
    if bare is not None:
        if not bare.is_finite():
            raise ReadError(f"{where}: a time must be a finite number of seconds, got {value!r}")
        return float(bare)
    pos, total = 0, Decimal(0)
    while pos < len(text):
        m = _PART.match(text, pos)
        if m is None:
            raise ReadError(f"{where}: cannot read a time from {value!r} (a number of seconds, "
                            f"or a number with a unit such as \"5 min\" or \"20 h\"; units "
                            f"{UNITS_TEXT})")
        total += Decimal(_to_seconds(m.group(1), m.group(2), f"{where} {value!r}"))
        pos = m.end()
    return float(total)


TIME_SCALES = ("seconds", "order")  # [input] time_scale: real times, or an order of points


def format_time(seconds: float, scale: str = "seconds") -> str:
    """A time for a label, in the largest unit that keeps it a short number: ``4 s``,
    ``1.5 min``, ``20 h``, ``250 ms``, ``0 s``; NaN is ``""`` (a control without a time).
    With ``scale='order'`` (``[input] time_scale``: the times only order the points) the
    number alone, ``2``, since it is not a time."""
    if seconds is None or not math.isfinite(seconds):
        return ""
    if scale not in TIME_SCALES:
        raise ValueError(f"scale must be one of {TIME_SCALES}")
    t = float(seconds)
    if scale == "order":
        return f"{t:g}"
    sign = "-" if t < 0 else ""
    t = abs(t)
    for unit, size in (("h", 3600.0), ("min", 60.0)):
        value = t / size
        if t >= size and abs(value - round(value, 2)) < 1e-9:
            return f"{sign}{value:g} {unit}"
    if 0 < t < 1:
        value = t * 1000.0
        if abs(value - round(value, 3)) < 1e-9:
            return f"{sign}{value:g} ms"
    return f"{sign}{t:g} s"


def parse_hxexpress_label(label: object) -> tuple[float, str]:
    """``(seconds, role)`` of an HX-Express column label: ``undeut`` (NaN, ``'undeut'``),
    ``TD`` (NaN, ``'fulldeut'``), ``60 s``, ``5 min``, ``2 h`` (the time, ``''``).

    As in pyHXExpress, digits after the decimal point of a time of 1 or more are replicate
    markers, not fractions: ``1.2 min`` is 60 s (the second spectrum at that time), while a
    time below 1 such as ``0.5 min`` keeps its fraction. That is HX-Express's convention for its
    own files; a time in a project file has no such rule (:func:`parse_time`).
    """
    if not isinstance(label, str):
        raise ReadError(f"time label {label!r} is not text")
    text = label.strip()
    if text.lower().startswith("undeut"):
        return math.nan, UNDEUT
    if text.upper().startswith("TD"):
        return math.nan, FULLDEUT
    parts = text.split()
    if len(parts) < 2:
        raise ReadError(f"time label {label!r} needs a number and a unit, e.g. '5 min'")
    number, unit = parts[0], parts[-1]
    whole = number.split(".")[0]
    return _to_seconds(number if whole == "0" else whole, unit, f"label {label!r}"), ""


def parse_specexport_token(token: str) -> tuple[float, str]:
    """``(seconds, role)`` of a SpecExport file-name time: ``Non`` (NaN, ``'undeut'``),
    ``Full`` (NaN, ``'fulldeut'``), ``60s``, ``5m``, ``2h`` (the time, ``''``)."""
    if token == "Non":
        return math.nan, UNDEUT
    if token == "Full":
        return math.nan, FULLDEUT
    if _SPECEXPORT_TIME.fullmatch(token) is None:
        raise ReadError(f"SpecExport time {token!r}: expected Non, Full, or a number and one "
                        "letter, s, m or h (30s, 0.8s, 5m, 20h)")
    return _to_seconds(token[:-1], token[-1:], f"label {token!r}"), ""
