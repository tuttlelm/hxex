# SPDX-License-Identifier: GPL-3.0-or-later
"""The window's background work (hxex.gui.tasks): a request replaced or dropped before it
started never runs (audit of 2026-11-15, C-12: a held arrow key queued a hundred drawings
nobody would see, done 44 s after the last key). Runs offscreen."""
import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QThreadPool  # noqa: E402

from hxex.gui.tasks import Latest  # noqa: E402


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
    qtbot.wait(50)
    assert ran == [1, 3, 1] and done == [3] and not latest.running
