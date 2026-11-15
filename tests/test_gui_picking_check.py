# SPDX-License-Identifier: GPL-3.0-or-later
"""The Results workspace's picking check tab (hxex.report.picking_check): on a spectrum of
results picked with the profile fit, and only there. Runs offscreen."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from hxex.gui.results_workspace import ResultsWorkspace  # noqa: E402
from hxex.pipeline import RunConfig, run  # noqa: E402
from hxex.report import Results  # noqa: E402
from test_profile_fit import CFG, run_set  # noqa: E402


@pytest.fixture(scope="module")
def results():
    s = run_set(labelled=((3.0, 0.0), (5.0, 0.0)))
    return Results.from_run(run(s, RunConfig(picker=CFG, bootstrap=None)), spectra=s)


def _spectrum_item(page):
    pep = page.tree.topLevelItem(0)
    series = pep.child(0)
    page.tree.expandItem(series)
    point = series.child(1)  # the first labelled time after the controls
    page.tree.expandItem(point)
    return point.child(0) if point.childCount() else point


def test_a_spectrum_of_profile_fit_results_has_a_picking_check(qtbot, results):
    page = ResultsWorkspace()
    qtbot.addWidget(page)
    page.show_results(results, None)
    with qtbot.waitSignal(page.panel("picking check").drawn, timeout=30000):
        page.tree.setCurrentItem(_spectrum_item(page))
    tabs = [page.figures.tabText(i) for i in range(page.figures.count())]
    assert tabs[:2] == ["spectrum", "picking check"]
    title = page.panel("picking check").view.figure.axes[0].get_title(loc="left")
    assert "purity" in title
