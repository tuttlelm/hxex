# SPDX-License-Identifier: GPL-3.0-or-later
"""Work done off the window's thread (docs/PLAN.md, "GUI", rule 4): loading results and drawing
figures run in Qt's thread pool and report back to the window through signals.

:class:`Latest` runs one kind of work at a time from the window's point of view: a newer request
replaces an older one still running (the older one finishes, and its result is ignored).

Python's garbage collector frees reference cycles in whichever thread happens to trigger it; a
cycle holding a window object (a widget, a timer) freed in a worker thread crashes Qt (seen on
macOS: a collection during a drawing). While any work runs off the window's thread, automatic
collection is therefore paused, and it resumes in the window's thread when the last one ends
(:class:`_Collection`). Objects without cycles are freed as usual meanwhile."""

from __future__ import annotations

import gc
import logging
import threading
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

log = logging.getLogger(__name__)


class _Signals(QObject):
    done = Signal(int, object)
    failed = Signal(int, str)


class _Collection(QObject):
    """Automatic garbage collection paused while work runs off the window's thread: :meth:`hold`
    (in the window's thread, before the work starts) and :attr:`release` (emitted by the work
    when it ends, received in the window's thread, where collection resumes after the last)."""

    release = Signal()

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._running = 0
        self._was_enabled = True
        self.release.connect(self._released)

    def hold(self) -> None:
        with self._lock:
            if self._running == 0:
                self._was_enabled = gc.isenabled()
                gc.disable()
            self._running += 1

    def _released(self) -> None:
        with self._lock:
            if self._running == 0:  # after :meth:`settled`: already resumed
                return
            self._running -= 1
            if self._running == 0 and self._was_enabled:
                gc.enable()

    def settled(self) -> None:
        """Every pool has stopped (work waiting in a cleared pool never ends): collection
        resumes as it was."""
        with self._lock:
            if self._running and self._was_enabled:
                gc.enable()
            self._running = 0

    @property
    def running(self) -> int:
        return self._running


_COLLECTION: _Collection | None = None


def collection() -> _Collection:
    """The one :class:`_Collection` (made in the window's thread, by the first :class:`Latest`)."""
    global _COLLECTION
    if _COLLECTION is None:
        _COLLECTION = _Collection()
    return _COLLECTION


class _Job(QRunnable):
    def __init__(self, ticket: int, work: Callable[[], Any], signals: _Signals,
                 gate: _Collection):
        super().__init__()
        self.ticket, self.work, self.signals, self.gate = ticket, work, signals, gate

    def run(self):
        try:
            try:
                result = self.work()
            except Exception as exc:  # noqa: BLE001 - reported to the window, details logged
                log.debug("background work failed:\n%s", traceback.format_exc())
                self._emit(self.signals.failed, f"{exc}" or type(exc).__name__)
                return
            self._emit(self.signals.done, result)
        finally:
            try:
                self.gate.release.emit()
            except RuntimeError:  # the application is gone (closed while this ran)
                pass

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
        self._gate = collection()
        self.running = False

    def submit(self, work: Callable[[], Any], done: Callable[[Any], None],
               failed: Callable[[str], None]) -> None:
        self._ticket += 1
        self._callbacks = (done, failed)
        self.running = True
        self._gate.hold()
        self.pool.start(_Job(self._ticket, work, self._signals, self._gate))

    def _done(self, ticket: int, result) -> None:
        if ticket == self._ticket and self._callbacks is not None:
            self.running = False
            self._callbacks[0](result)

    def _failed(self, ticket: int, message: str) -> None:
        if ticket == self._ticket and self._callbacks is not None:
            self.running = False
            self._callbacks[1](message)
