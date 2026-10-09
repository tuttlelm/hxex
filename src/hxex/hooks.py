# SPDX-License-Identifier: GPL-3.0-or-later
"""Progress and cancellation hooks shared by every long operation.

A progress hook is called as ``progress(done, total)``. A cancel hook is polled between units of
work and returns True to stop; the operation then raises :class:`Cancelled`.
"""

from __future__ import annotations

from collections.abc import Callable

ProgressHook = Callable[[int, int], None]
CancelHook = Callable[[], bool]


class Cancelled(Exception):
    """The operation was stopped by its cancel hook."""


def check_cancel(cancel: CancelHook | None) -> None:
    if cancel is not None and cancel():
        raise Cancelled()


def report(progress: ProgressHook | None, done: int, total: int) -> None:
    if progress is not None:
        progress(done, total)
