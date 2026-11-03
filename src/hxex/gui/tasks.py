# SPDX-License-Identifier: GPL-3.0-or-later
"""Work done off the window's thread (docs/PLAN.md, "GUI", rule 4): loading results and drawing
figures run in Qt's thread pool and report back to the window through signals.

:class:`Latest` runs one kind of work at a time from the window's point of view: a newer request
replaces an older one still running (the older one finishes, and its result is ignored)."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

log = logging.getLogger(__name__)


class _Signals(QObject):
    done = Signal(int, object)
    failed = Signal(int, str)


class _Job(QRunnable):
    def __init__(self, ticket: int, work: Callable[[], Any], signals: _Signals):
        super().__init__()
        self.ticket, self.work, self.signals = ticket, work, signals

    def run(self):
        try:
            result = self.work()
        except Exception as exc:  # noqa: BLE001 - reported to the window, details logged
            log.debug("background work failed:\n%s", traceback.format_exc())
            self._emit(self.signals.failed, f"{exc}" or type(exc).__name__)
            return
        self._emit(self.signals.done, result)

    def _emit(self, signal, value) -> None:
        try:
            signal.emit(self.ticket, value)
        except RuntimeError:  # the window that asked is gone: nobody wants the result
            pass


_DRAWING: QThreadPool | None = None


def drawing_pool() -> QThreadPool:
    """The one thread every matplotlib drawing runs in. matplotlib is not thread-safe (its font
    objects are shared between figures): two figures drawn at once in two threads crashed the
    window, so the drawings wait their turn here."""
    global _DRAWING
    if _DRAWING is None:
        _DRAWING = QThreadPool()
        _DRAWING.setMaxThreadCount(1)
    return _DRAWING


class Latest(QObject):
    """Runs ``work`` in the thread pool and calls ``done(result)`` or ``failed(message)`` in the
    window's thread, for the latest request only."""

    def __init__(self, parent=None, pool: QThreadPool | None = None):
        super().__init__(parent)
        self.pool = pool or QThreadPool.globalInstance()
        self._ticket = 0
        self._callbacks: tuple[Callable, Callable] | None = None
        self._signals = _Signals(self)
        self._signals.done.connect(self._done)
        self._signals.failed.connect(self._failed)
        self.running = False

    def submit(self, work: Callable[[], Any], done: Callable[[Any], None],
               failed: Callable[[str], None]) -> None:
        self._ticket += 1
        self._callbacks = (done, failed)
        self.running = True
        self.pool.start(_Job(self._ticket, work, self._signals))

    def _done(self, ticket: int, result) -> None:
        if ticket == self._ticket and self._callbacks is not None:
            self.running = False
            self._callbacks[0](result)

    def _failed(self, ticket: int, message: str) -> None:
        if ticket == self._ticket and self._callbacks is not None:
            self.running = False
            self._callbacks[1](message)
