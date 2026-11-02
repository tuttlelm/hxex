# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers shared by the readers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .errors import ReadError


def to_number(cell: object, where: str) -> float | None:
    """A cell as a float; empty and NaN cells are None; other text is an error."""
    if cell is None:
        return None
    if isinstance(cell, str):
        text = cell.strip()
        if not text or text.lower() == "nan":
            return None
        try:
            return float(text)
        except ValueError:
            raise ReadError(f"{where}: {cell!r} is not a number") from None
    if isinstance(cell, bool):
        raise ReadError(f"{where}: {cell!r} is not a number")
    try:
        value = float(cell)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ReadError(f"{where}: {cell!r} is not a number") from None
    return None if math.isnan(value) else value


def clean_spectrum(
    mz: Sequence[float | None], intensity: Sequence[float | None]
) -> tuple[np.ndarray, np.ndarray]:
    """Drop points where m/z or intensity is missing, then sort by m/z (stable)."""
    pairs = [(m, i) for m, i in zip(mz, intensity, strict=True) if m is not None and i is not None]
    arr = np.array(pairs, dtype=np.float64).reshape(-1, 2)
    order = np.argsort(arr[:, 0], kind="stable")
    return arr[order, 0].copy(), arr[order, 1].copy()


def open_text(path, *, newline: str | None = None):
    """Open a text file for reading as UTF-8 (a byte-order mark skipped), or as UTF-16 when it
    starts with that encoding's byte-order mark (Excel's "Unicode text" files)."""
    from pathlib import Path

    p = Path(path)
    with p.open("rb") as fh:
        head = fh.read(2)
    encoding = "utf-16" if head in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    return p.open(encoding=encoding, newline=newline)


def not_text_error(path, exc: Exception) -> ReadError:
    """The :class:`ReadError` for a file that could not be read as text: a decoding error names
    the encoding the file should have; any other OS error is given as it is."""
    if isinstance(exc, UnicodeDecodeError):
        return ReadError(f"cannot read {path}: the file is not UTF-8 text (save it as CSV "
                         f"UTF-8, or as Unicode text) ({exc.reason} at byte {exc.start})")
    return ReadError(f"cannot read {path}: {exc}")
