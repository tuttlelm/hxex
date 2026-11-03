# SPDX-License-Identifier: GPL-3.0-or-later
"""The Results workspace's tree (hxex.gui.results_tree, hxex.gui.results_workspace): peptide,
charge, time point, replicate; the figures of each level; filters and search; a selection
restored by what outlasts a rerun. Runs offscreen."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from hxex.gui.results_tree import peptides, points, shown  # noqa: E402
from hxex.gui.results_workspace import ResultsWorkspace  # noqa: E402
from hxex.report import Results  # noqa: E402
from hxex.report.series import plot_series_spectra  # noqa: E402
from test_run import config, raw_spectra, run  # noqa: E402


@pytest.fixture(scope="module")
def results():
    return Results.from_run(run(raw_spectra(n_series=2), config(bootstrap=None)))


def test_the_tree_holds_peptides_series_points_and_spectra(results):
    peps = peptides(results)
    assert len(peps) == 1 and [s.data_id for s in peps[0].series] == [0, 1]
    pts = points(results, 0)
    assert [p.label for p in pts] == ["UN", "1 min", "FD"]  # controls at the ends
    assert [s.rep for s in pts[1].spectra] == [1, 2, 3] and pts[1].ks == "1 1 1"
    series = peps[0].series[0]
    assert shown(peps[0], series) and shown(peps[0], series, text="rdkv")
    assert not shown(peps[0], series, text="nothing") and not shown(peps[0], series, "flagged")
    assert shown(peps[0], series, "unreviewed")
    with pytest.raises(ValueError):
        shown(peps[0], series, "odd")


def test_one_time_points_replicates_on_the_series_range(results):
    pts = points(results, 0)
    full = plot_series_spectra(results, 0, show=("picked", "fit"))
    cut = plot_series_spectra(results, 0, rows=[s.index for s in pts[1].spectra],
                              show=("picked", "fit"))
    assert len(cut.axes) == 3 and len(full.axes) == 8
    assert np.allclose(cut.axes[0].get_xlim(), full.axes[0].get_xlim())  # the same m/z range
    with pytest.raises(ValueError, match="none of the rows"):
        plot_series_spectra(results, 1, rows=[s.index for s in pts[1].spectra])


@pytest.fixture
def page(qtbot, results):
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, None)
    return w


def _open_series(page, n=0):
    series = page.tree.topLevelItem(0).child(n)
    page.tree.expandItem(series)
    return series


def test_each_level_draws_its_figures(qtbot, page):
    series = _open_series(page)
    assert [series.child(i).text(0) for i in range(series.childCount())] == ["UN", "1 min",
                                                                              "FD"]
    point = series.child(1)
    assert point.childCount() == 3 and point.text(6) == "1 1 1"
    with qtbot.waitSignal(page.panel("spectra").drawn, timeout=30000):
        page.tree.setCurrentItem(point)
    tabs = [page.figures.tabText(i) for i in range(page.figures.count())]
    assert tabs == ["spectra", "uptake"]
    assert len(page.panel("spectra").view.figure.axes) == 3  # that time's replicates
    with qtbot.waitSignal(page.panel("fits at each K").drawn, timeout=30000):
        page.tree.setCurrentItem(point.child(0))
    assert [page.figures.tabText(i) for i in range(page.figures.count())] == [
        "spectrum", "fits at each K", "uptake"]
    qtbot.waitUntil(lambda: page.panel("spectrum").view.figure is not None, timeout=30000)
    sel = page.selection()
    assert sel["charge"] == 2 and sel["spectrum_id"] == point.child(0).toolTip(0)


def test_a_selection_is_found_again_and_filters_narrow_the_tree(page):
    series = _open_series(page)
    page.tree.setCurrentItem(series.child(2).child(1) if series.child(2).childCount() > 1
                             else series.child(2).child(0))
    chosen = page.selection()
    page.search.setText("nothing")
    assert page.tree.topLevelItemCount() == 0 and page.count.text() == "0 of 2 series"
    page.search.clear()
    page.filter.setCurrentIndex(list(page.filter.itemData(i) for i in
                                     range(page.filter.count())).index("flagged"))
    assert page.tree.topLevelItemCount() == 0
    assert page.select(chosen) and page.selection() == chosen  # the filter cleared to find it
    assert page.filter.currentData() == "all"
    assert page.select({"sample": "S", "peptide_range": "0001-0010", "charge": 2,
                        "point": "1 min"})
    assert page.selection()["point"] == "1 min"
    assert not page.select({"sample": "X", "peptide_range": "0001-0010"})
