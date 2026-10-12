# SPDX-License-Identifier: GPL-3.0-or-later
"""Progress and cancellation hooks shared by every long operation.

A progress hook is called as ``progress(done, total)``. A cancel hook is polled between units of
work and returns True to stop; the operation then raises :class:`Cancelled`.

An operation made of stages (such as :func:`hxex.run.run`: pick, fit, bootstrap) also names
them: :func:`start_stage` tells a hook that has a ``start_stage(name, start, end)`` method that
the units from ``start`` to ``end`` are that stage (plain functions are not told).
:class:`ProgressLog` is such a hook: it logs the stage, how far it is and the time it has left.
"""

from __future__ import annotations

import logging
import time
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


class ProgressLog:
    """A progress hook that logs how far an operation is: the stage (when the operation names
    its stages), the percent done of the stage, the time it has taken and the time it has left,
    estimated from its rate so far (the stages of a run go at very different rates, so the
    estimate is for the current stage only). It logs when a stage starts, then at most every
    ``interval`` seconds, and when everything is done.

    ``logger`` defaults to the ``hxex.progress`` logger and ``level`` to INFO; ``clock`` (seconds)
    is for tests."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        interval: float = 30.0,
        level: int = logging.INFO,
        clock: Callable[[], float] = time.monotonic,
    ):
        if interval < 0:
            raise ValueError("interval must be >= 0")
        self.logger = logger if logger is not None else logging.getLogger("hxex.progress")
        self.interval, self.level, self.clock = interval, level, clock
        self.began = self.clock()
        self.stage: str | None = None
        self.span = (0.0, 0.0)  # the stage's units
        self.stage_began = self.began
        self.last = -float("inf")  # when the last line was logged
        self.finished = False

    def start_stage(self, name: str, start: float, end: float) -> None:
        now = self.clock()
        if self.stage is not None:
            self.logger.log(self.level, "%s: done in %s", self.stage,
                            format_duration(now - self.stage_began))
        self.stage, self.span, self.stage_began = name, (float(start), float(end)), now
        self.logger.log(self.level, "%s: started", name)
        self.last = now

    def __call__(self, done: int, total: int) -> None:
        now = self.clock()
        if total and done >= total:
            if not self.finished:
                self.finished = True
                if self.stage is not None:
                    self.logger.log(self.level, "%s: done in %s", self.stage,
                                    format_duration(now - self.stage_began))
                self.logger.log(self.level, "done in %s", format_duration(now - self.began))
            return
        if now - self.last < self.interval:
            return
        self.last = now
        start, end = self.span if self.stage is not None else (0.0, float(total))
        frac = min(max((done - start) / (end - start), 0.0), 1.0) if end > start else 0.0
        spent = now - (self.stage_began if self.stage is not None else self.began)
        left = f", about {format_duration(spent / frac * (1 - frac))} left" if frac > 0 else ""
        self.logger.log(self.level, "%s: %.0f%% after %s%s", self.stage or "progress",
                        100 * frac, format_duration(spent), left)
