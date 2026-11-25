# SPDX-License-Identifier: GPL-3.0-or-later
"""The window's background work (hxex.gui.tasks): a request replaced or dropped before it
started never runs (audit of 2026-11-15, C-12: a held arrow key queued a hundred drawings
nobody would see, done 44 s after the last key). Runs offscreen."""
import threading

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication

from hxex.gui.tasks import Latest


def test_a_request_replaced_or_dropped_before_it_started_never_runs(qtbot):
    pool = QThreadPool()
    pool.setMaxThreadCount(1)
    latest = Latest(pool=pool)
    gate, started = threading.Event(), threading.Event()
    ran, done = [], []

    def blocked():
        started.set()
        gate.wait(10)
        ran.append(1)
        return 1

    latest.submit(blocked, done.append, done.append)
    assert started.wait(10)  # running: it finishes (one not started would be skipped)
    latest.submit(lambda: (ran.append(2), 2)[1], done.append, done.append)  # replaced below
    latest.submit(lambda: (ran.append(3), 3)[1], done.append, done.append)
    gate.set()
    qtbot.waitUntil(lambda: done == [3], timeout=10000)
    assert pool.waitForDone(10000) and ran == [1, 3]  # the replaced one never started

    gate.clear()
    started.clear()
    latest.submit(blocked, done.append, done.append)
    assert started.wait(10)
    latest.submit(lambda: (ran.append(4), 4)[1], done.append, done.append)
    latest.drop()  # nothing of either is wanted
    gate.set()
    assert pool.waitForDone(10000)
    QApplication.processEvents()  # what the jobs sent back delivered (not a fixed wait: Q-29)
    assert ran == [1, 3, 1] and done == [3] and not latest.running


def test_a_job_runs_only_once_its_start_has_returned(qtbot, monkeypatch):
    """PySide makes the pool the job's parent after the pool's own ``start`` has handed the job
    to a thread; a job deleted before then is left as the pool's child for good, with its work
    and what it captured (a loaded CI runner, 2026-11-23/24). So a job waits for the start that
    queued it (hxex.gui.tasks._STARTING)."""
    from hxex.gui import tasks

    # re-entrant here, so that this thread can hold the lock across submit as the start would
    monkeypatch.setattr(tasks, "_STARTING", threading.RLock())
    pool = QThreadPool()
    pool.setMaxThreadCount(2)
    latest = Latest(pool=pool)
    ran, done = threading.Event(), []
    with tasks._STARTING:  # as during pool.start: the job is queued but must not run yet
        latest.submit(lambda: (ran.set(), 1)[1], done.append, done.append)
        assert not ran.wait(0.3)
    assert ran.wait(10)
    qtbot.waitUntil(lambda: done == [1], timeout=10000)
    assert pool.waitForDone(10000)
