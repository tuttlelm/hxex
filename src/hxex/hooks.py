# SPDX-License-Identifier: GPL-3.0-or-later
"""Progress and cancellation hooks shared by every long operation.

A progress hook is called as ``progress(done, total)``. A cancel hook is polled between units of
work and returns True to stop; the operation then raises :class:`Cancelled`.

An operation made of stages (such as :func:`hxex.pipeline.run`: pick, fit, bootstrap) also names
them: :func:`start_stage` tells a hook that has a ``start_stage(name, start, end)`` method that
the units from ``start`` to ``end`` are that stage (plain functions are not told).
:class:`ProgressLog` is such a hook: it logs the stage, how far it is and the time it has left.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
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


def start_stage(progress: ProgressHook | None, name: str, start: float, end: float) -> None:
    """Tell ``progress``, if it follows stages, that the units from ``start`` to ``end`` (of the
    ``total`` it is called with) are the stage ``name`` (``start == end`` for a stage that
    reports no units)."""
    hook = getattr(progress, "start_stage", None)
    if hook is not None:
        hook(name, start, end)


def format_duration(seconds: float) -> str:
    """``seconds`` as ``'45s'``, ``'4m 05s'`` or ``'2h 10m'``."""
    s = int(round(max(seconds, 0.0)))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


# no "time left" before this share of a stage is done and this many updates have come in: an
# early rate (the first resample, the fast K = 1 fits) misleads (audit 2026-11-01, Part 2 M1)
ESTIMATE_AFTER = 0.10
ESTIMATE_UPDATES = 2
LONGEST_ESTIMATE = 10 * 3600.0  # beyond it: "more than 10h left"


class ProgressLog:
    """A progress hook that logs how far an operation is: the stage (when the operation names
    its stages), the percent done of the stage, the time it has taken and the time it has left,
    estimated from its rate so far (the stages of a run go at very different rates, so the
    estimate is for the current stage only). It logs when a stage starts, then at most every
    ``interval`` seconds, and when everything is done.

    The time left is given only once the stage is at least a tenth done and two updates have
    come in (an early rate misleads), and a longer estimate than ten hours reads "more than 10h
    left". While a stage runs and no update arrives, a line still comes at the interval
    (``bootstrap: first results not in yet, 3m 00s so far``), so a slow step can be told from a
    hung one: a daemon thread (``heartbeat``; by default only with the real clock) that stops when
    the operation is done, when :meth:`close` is called (the hook is also a context manager),
    or when the hook is no longer used.

    ``name`` labels an operation that does not name its stages (``'progress'`` by default).
    ``logger`` defaults to the ``hxex.progress`` logger and ``level`` to INFO; ``clock`` (seconds)
    is for tests, which call :meth:`beat` themselves."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        name: str | None = None,
        interval: float = 30.0,
        level: int = logging.INFO,
        clock: Callable[[], float] = time.monotonic,
        heartbeat: bool | None = None,
    ):
        if interval < 0:
            raise ValueError("interval must be >= 0")
        self.logger = logger if logger is not None else logging.getLogger("hxex.progress")
        self.interval, self.level, self.clock = interval, level, clock
        self.name = name
        self.began = self.clock()
        self.stage: str | None = None
        self.span = (0.0, 0.0)  # the stage's units
        self.stage_began = self.began
        self.last = -float("inf")  # when the last line was logged
        self.finished = False
        self.latest: tuple[int, int] | None = None  # the stage's last update
        self.updates = 0  # updates of the stage that moved it on
        if heartbeat is None:
            heartbeat = clock is time.monotonic
        self.heartbeat = bool(heartbeat) and interval > 0
        self._lock = threading.RLock()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> ProgressLog:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """No more lines from the heartbeat (the operation ended, also by an error or Ctrl-C)."""
        self._stopped.set()

    def _start_heartbeat(self) -> None:
        if not self.heartbeat or self._thread is not None or self._stopped.is_set():
            return
        self._thread = threading.Thread(
            target=_heartbeat, args=(weakref.ref(self), self._stopped, self.interval),
            name="hxex-progress", daemon=True)
        self._thread.start()

    def start_stage(self, name: str, start: float, end: float) -> None:
        with self._lock:
            now = self.clock()
            if self.stage is not None:
                self.logger.log(self.level, "%s: done in %s", self.stage,
                                format_duration(now - self.stage_began))
            self.stage, self.span, self.stage_began = name, (float(start), float(end)), now
            self.latest, self.updates = None, 0
            self.logger.log(self.level, "%s: started", name)
            self.last = now
        self._start_heartbeat()

    def __call__(self, done: int, total: int) -> None:
        with self._lock:
            now = self.clock()
            if total >= 0 and done >= total:
                if not self.finished:
                    self.finished = True
                    self.close()
                    if self.stage is not None:
                        self.logger.log(self.level, "%s: done in %s", self.stage,
                                        format_duration(now - self.stage_began))
                    done_in = format_duration(now - self.began)
                    if self.name is not None and self.stage is None:
                        self.logger.log(self.level, "%s: done in %s", self.name, done_in)
                    else:
                        self.logger.log(self.level, "done in %s", done_in)
                return
            before = (self.latest[0] if self.latest is not None
                      else self.span[0] if self.stage is not None else 0)
            if done > before:
                self.updates += 1
            self.latest = (done, total)
            if now - self.last >= self.interval:
                self._line(now)
        self._start_heartbeat()

    def beat(self) -> None:
        """The heartbeat's line: when nothing has been logged for ``interval`` while a stage (or
        the operation) runs, the latest state again, or that no result is in yet."""
        with self._lock:
            if self.finished or self._stopped.is_set():
                return
            now = self.clock()
            if now - self.last < self.interval:
                return
            if self.stage is None and self.latest is None and self.name is None:
                return  # nothing begun yet
            self._line(now)

    def _fraction(self) -> float:
        if self.latest is None:
            return 0.0
        done, total = self.latest
        start, end = self.span if self.stage is not None else (0.0, float(total))
        return min(max((done - start) / (end - start), 0.0), 1.0) if end > start else 0.0

    def _line(self, now: float) -> None:
        self.last = now
        label = self.stage or self.name or "progress"
        spent = now - (self.stage_began if self.stage is not None else self.began)
        frac = self._fraction()
        if frac <= 0:  # a stage without units says it is working; else none is in yet
            what = ("still working" if self.stage is not None and self.span[1] <= self.span[0]
                    else "first results not in yet")
            self.logger.log(self.level, "%s: %s, %s so far", label, what, format_duration(spent))
            return
        left = ""
        if frac >= ESTIMATE_AFTER and self.updates >= ESTIMATE_UPDATES:
            rest = spent / frac * (1 - frac)
            left = (f", more than {LONGEST_ESTIMATE / 3600:.0f}h left" if rest > LONGEST_ESTIMATE
                    else f", about {format_duration(rest)} left")
        self.logger.log(self.level, "%s: %.0f%% after %s%s", label, 100 * frac,
                        format_duration(spent), left)


def _heartbeat(ref, stopped: threading.Event, interval: float) -> None:
    """The heartbeat thread of a :class:`ProgressLog`: wakes a few times per interval and asks
    the log for its line; ends when the log is closed or gone."""
    wait = max(interval / 4.0, 0.25)
    while not stopped.wait(wait):
        log = ref()
        if log is None:
            return
        try:
            log.beat()
        except Exception:  # noqa: BLE001 - a log line must never stop the operation
            return
        del log
