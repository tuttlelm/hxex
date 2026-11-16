# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixtures every test shares."""

import sys

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    """Every drawing a test left in the window's drawing thread (:func:`hxex.gui.tasks
    .drawing_pool`) is finished before the next test starts. matplotlib is not thread-safe: a
    page still drawing there while the next test drew figures in the main thread broke the
    mathtext parser (``ParseException`` on ``$\\times10^{4}$``, an axis offset, on a random CI
    platform; HISTORY 282). The window itself draws only in that thread.

    A wrapper around the whole teardown, not a fixture: pytest-qt's own teardown (a wrapper
    too, placed last) processes the application's events once more after the fixtures are
    torn down, and a drawing's "done" delivered there may ask for the next one (a stage of a
    staged drawing, a figure tab opened by the test's last click), which then ran in the
    drawing thread while the next test drew in the main thread (the lowest-versions CI job,
    2026-11-15, a test of tests/test_spectra_pages.py). So the wait comes after everything,
    and events are processed and waited for again until nothing new starts."""
    yield
    tasks = sys.modules.get("hxex.gui.tasks")  # never imports Qt for tests that do not
    if tasks is None:
        return
    # the window's background work too (hxex.gui.tasks.Latest, Qt's global pool): a preview
    # fit a test left running (a newer one asked for, the old one stopping at its next
    # check) used PyTorch's forward-AD levels, which are process-wide, while the next test
    # fitted in the main thread ("Trying to access a forward AD level with an invalid
    # index", Windows CI, 2026-11-11). The window fits one at a time
    # (gui.window_edit._FIT_LOCK) and runs hxex in child processes.
    from PySide6.QtCore import QCoreApplication, QThreadPool

    pools = [pool for pool in (tasks._DRAWING, QThreadPool.globalInstance())
             if pool is not None]
    app = QCoreApplication.instance()
    for _ in range(10):
        for pool in pools:
            pool.waitForDone(120000)
        if app is None:
            break
        # what the finished work asked for (its callbacks): delivered, so that anything it
        # starts is waited for as well
        app.processEvents()
        if not any(pool.activeThreadCount() for pool in pools):
            break
    # the window's garbage collection (gui.tasks.collection: off while a window exists, run
    # from a timer between events): collected and given back between tests, so tests without
    # a window collect as usual
    if tasks._COLLECTION is not None:
        tasks._COLLECTION.release()


@pytest.fixture(autouse=True)
def _table_dialog_answered(monkeypatch):
    """The window asks where a project's first elution-window or review table goes
    (:func:`hxex.gui.table_choice.ask_table`, a dialog that would wait for a user): answered
    with the name it suggests, as a user pressing Save would. tests/test_table_choice.py puts
    the real dialog back to test it."""
    from pathlib import Path

    from hxex.gui import table_choice

    monkeypatch.setattr(table_choice, "ask_table", lambda parent, project, setting, base, what:
                        Path(project).parent / table_choice.suggested_name(project, base))
