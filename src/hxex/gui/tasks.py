# SPDX-License-Identifier: GPL-3.0-or-later
"""Work done off the window's thread (docs/PLAN.md, "GUI", rule 4): loading results and drawing
figures run in Qt's thread pool and report back to the window through signals.

:class:`Latest` runs one kind of work at a time from the window's point of view: a newer request
replaces an older one still running (the older one finishes, and its result is ignored).

Python's garbage collector frees reference cycles wherever an allocation happens to cross its
threshold. In a worker thread, a cycle holding a window object (a widget, a timer) freed there
crashed Qt (seen on macOS: a collection during a drawing). In the window's own thread, inside
Qt's code, it deadlocked: a weak reference's callback disconnected a signal, Qt held its
connection lock while asking Python about an override, that allocation started a collection,
and the collection freed a widget whose destructor waited for the same lock (CI and local
runs, HISTORY 365). So while the window exists, automatic collection is off, and the window
collects from a timer, between events, where Qt holds no lock (:class:`_Collection`). Objects
without cycles are freed as usual meanwhile."""

from __future__ import annotations

import gc
import logging
import threading
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

log = logging.getLogger(__name__)


class _Signals(QObject):
    done = Signal(int, object)
    failed = Signal(int, str)


class _Collection(QObject):
    """Garbage collection in the window's thread, between events: :meth:`take_over` (when the
    window's first background runner is made) turns automatic collection off and starts a timer
    that collects as automatic collection would (the oldest generation past its threshold);
    :meth:`release` (the window closed, every pool stopped) collects once and turns automatic
    collection back on as it was."""

    INTERVAL_MS = 250

    def __init__(self):
        super().__init__()
        self._was_enabled: bool | None = None  # None: not taken over
        self._timer = QTimer(self)
        self._timer.setInterval(self.INTERVAL_MS)
        self._timer.timeout.connect(self.collect_due)

    def take_over(self) -> None:
        if self._was_enabled is None:
            self._was_enabled = gc.isenabled()
            gc.disable()
        if not self._timer.isActive():
            self._timer.start()

    def collect_due(self) -> None:
        """The oldest generation whose count passed its threshold, as Python would."""
        counts, thresholds = gc.get_count(), gc.get_threshold()
        for generation in (2, 1, 0):
            if thresholds[generation] and counts[generation] >= thresholds[generation]:
                gc.collect(generation)
                return

    def release(self) -> None:
        self._timer.stop()
        if self._was_enabled is None:
            return
        gc.collect()
        if self._was_enabled:
            gc.enable()
        self._was_enabled = None

    @property
    def active(self) -> bool:
        return self._was_enabled is not None


_COLLECTION: _Collection | None = None


def collection() -> _Collection:
    """The one :class:`_Collection` (made in the window's thread, by the first :class:`Latest`)."""
    global _COLLECTION
    if _COLLECTION is None:
        _COLLECTION = _Collection()
    return _COLLECTION


# Held while a job is handed to its pool (``QThreadPool.start``): PySide makes the pool the
# job's parent, keeping a reference to the job, as part of that call, after the pool's own
# code, which has already handed the job to a waiting thread. A job quick enough to run and be
# deleted (auto-delete) before ``start`` returned, while the window's thread was slow to take
# the interpreter lock back on a loaded machine, was destroyed with no parent to be removed
# from, and then made the pool's child: the pool kept the dead job, and the job's work kept
# what it had captured, such as the results a list was built from (CI, 2026-11-23/24; the
# closed-window test of tests/test_gui_state.py). So a job runs only once its start is over.
_STARTING = threading.Lock()


class _Job(QRunnable):
    def __init__(self, ticket: int, work: Callable[[], Any], signals: _Signals,
                 current: Callable[[], int] | None = None):
        super().__init__()
        self.ticket, self.work, self.signals = ticket, work, signals
        self.current = current  # the ticket wanted now (None: always this one)

    def run(self):
        with _STARTING:  # the start that queued this job has returned (see _STARTING)
            pass
        if self.current is not None and self.current() != self.ticket:
            return  # replaced or dropped before it started: nobody wants it (audit C-12)
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
    window's thread, for the latest request only. A request replaced or dropped before its
    work started never runs (a held arrow key had queued a hundred drawings nobody would see,
    done 44 s after the last key; audit of 2026-11-15, C-12); one already running finishes."""

    def __init__(self, parent=None, pool: QThreadPool | None = None):
        super().__init__(parent)
        self.pool = pool or QThreadPool.globalInstance()
        self._ticket = 0
        self._callbacks: tuple[Callable, Callable] | None = None
        self._signals = _Signals(self)
        self._signals.done.connect(self._done)
        self._signals.failed.connect(self._failed)
        collection().take_over()
        self.running = False

    def submit(self, work: Callable[[], Any], done: Callable[[Any], None],
               failed: Callable[[str], None]) -> None:
        self._ticket += 1
        self._callbacks = (done, failed)
        self.running = True
        job = _Job(self._ticket, work, self._signals, current=lambda: self._ticket)
        with _STARTING:
            self.pool.start(job)

    def drop(self) -> None:
        """Forget the request still running: neither of its callbacks is called (its work runs
        to its end in the pool when it has started; otherwise it never starts)."""
        self._ticket += 1
        self._callbacks = None
        self.running = False

    def _done(self, ticket: int, result) -> None:
        self._deliver(ticket, 0, result)

    def _failed(self, ticket: int, message: str) -> None:
        self._deliver(ticket, 1, message)

    def _deliver(self, ticket: int, which: int, value) -> None:
        """Call the request's done (0) or failed (1) callback with ``value``, once, and let the
        callbacks go: their closures (the results a list was built from, say) would otherwise
        stay held until the next request or :meth:`drop`."""
        callbacks = self._callbacks
        if ticket != self._ticket or callbacks is None:
            return
        self.running = False
        try:
            callbacks[which](value)
        finally:
            if self._callbacks is callbacks:  # the callback made no new request
                self._callbacks = None
