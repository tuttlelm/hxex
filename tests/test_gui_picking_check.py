# SPDX-License-Identifier: GPL-3.0-or-later
"""The Results workspace's picking check tab (hxex.report.picking_check): on a spectrum of
results picked with the profile fit, and only there. Runs offscreen."""

import pytest

pytest.importorskip("PySide6")

from hxex.gui.results_workspace import ResultsWorkspace
from hxex.pipeline import RunConfig, run
from hxex.report import Results
from test_profile_fit import CFG, run_set


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
    page.tree.setCurrentItem(_spectrum_item(page))
    tabs = [page.figures.tabText(i) for i in range(page.figures.count())]
    assert tabs[:2] == ["spectrum", "picking check"]
    with qtbot.waitSignal(page.panel("picking check").drawn, timeout=30000):
        page.figures.setCurrentWidget(page.panel("picking check"))  # drawn when opened
    title = page.panel("picking check").view.figure.axes[0].get_title(loc="left")
    assert "purity" in title


def test_a_spectrum_that_kept_the_pick_says_so_instead_of_failing(qtbot, results):
    """A spectrum the profile fit did not read (a fragment, another ion type, its own
    envelope, centroided data) has no picking check: the tab says so, rather than failing to
    draw (audit of 2026-11-15, the picking check tab read with the same eyes)."""
    import dataclasses

    import numpy as np

    b = results.binned
    offsets = np.asarray(b.fit_offset_ppm, dtype=float).copy()
    page = ResultsWorkspace()
    qtbot.addWidget(page)
    page.show_results(results, None)
    sid = _spectrum_item(page).data(0, 257)[3].spectrum_id
    offsets[b.find(sid)] = np.nan  # as a spectrum that kept the pick
    kept = dataclasses.replace(results, binned=dataclasses.replace(b, fit_offset_ppm=offsets))
    page.show_results(kept, None)
    page.tree.setCurrentItem(_spectrum_item(page))
    page.figures.setCurrentWidget(page.panel("picking check"))
    panel = page.panel("picking check")
    assert panel.view.figure is None and "kept the pick" in panel.status.full()
    assert "picking check" not in page._pending_draws
