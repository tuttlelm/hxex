# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixtures every test shares."""

import sys

import pytest


@pytest.fixture(autouse=True)
def _drawings_finished():
    """Every drawing a test left in the window's drawing thread (:func:`hxex.gui.tasks
    .drawing_pool`) is finished before the next test starts. matplotlib is not thread-safe: a
    page still drawing there while the next test drew figures in the main thread broke the
    mathtext parser (``ParseException`` on ``$\\times10^{4}$``, an axis offset, on a random CI
    platform; HISTORY 282). The window itself draws only in that thread."""
    yield
    tasks = sys.modules.get("hxex.gui.tasks")  # never imports Qt for tests that do not
    if tasks is not None and tasks._DRAWING is not None:
        tasks._DRAWING.waitForDone(120000)
    if tasks is not None:
        # the window's background work too (hxex.gui.tasks.Latest, Qt's global pool): a
        # preview fit a test left running (a newer one asked for, the old one stopping at its
        # next check) used PyTorch's forward-AD levels, which are process-wide, while the
        # next test fitted in the main thread ("Trying to access a forward AD level with an
        # invalid index", Windows CI, 2026-11-11). The window fits one at a time
        # (gui.window_edit._FIT_LOCK) and runs hxex in child processes.
        from PySide6.QtCore import QThreadPool

        QThreadPool.globalInstance().waitForDone(120000)
        # the window's garbage collection (gui.tasks.collection: off while a window exists,
        # run from a timer between events): collected and given back between tests, so tests
        # without a window collect as usual
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
