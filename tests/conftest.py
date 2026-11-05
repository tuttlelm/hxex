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
