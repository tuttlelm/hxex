# SPDX-License-Identifier: GPL-3.0-or-later
"""Replacing and removing a file, tried again for a while when Windows refuses it.

Windows cannot replace or delete a file another program has open, even briefly: a virus
scanner, an indexer, a file watcher, or the window's background check of where the results
stand reading a bundle's manifest (HISTORY 435). The refusal is a PermissionError and passes
within moments, so these try again for about two seconds before giving up. Elsewhere they are
``os.replace`` and ``Path.unlink`` (the first try succeeds)."""

from __future__ import annotations

import os
import time
from pathlib import Path

REPLACE_TRIES = 20  # attempts at a replacement Windows refuses, about two seconds in all


def replace_file(src: str | Path, dst: str | Path) -> None:
    """``os.replace(src, dst)``, tried again while Windows refuses it. Raises PermissionError
    when it is still refused."""
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES - 1:
                raise
            time.sleep(0.1)


def remove_file(path: str | Path) -> None:
    """``Path(path).unlink(missing_ok=True)``, tried again while Windows refuses it."""
    for attempt in range(REPLACE_TRIES):
        try:
            Path(path).unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES - 1:
                raise
            time.sleep(0.1)
