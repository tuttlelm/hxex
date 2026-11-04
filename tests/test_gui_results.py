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
from hxex.gui.results_workspace import COLUMN, ResultsWorkspace  # noqa: E402
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
    assert point.childCount() == 3 and point.text(COLUMN["K"]) == "1 1 1"
    with qtbot.waitSignal(page.panel("spectra").drawn, timeout=30000):
        page.tree.setCurrentItem(point)
    tabs = [page.figures.tabText(i) for i in range(page.figures.count())]
    assert tabs == ["spectra", "fits at each K", "uptake"]
    qtbot.waitUntil(lambda: page.panel("fits at each K").view.figure is not None, timeout=30000)
    title = page.panel("fits at each K").view.figure.axes[0].get_title()
    assert title.splitlines()[-1].startswith("rep 1: K=")
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


def test_the_time_axis_is_chosen_in_the_window(qtbot, page):
    """The project's [figures] scale first (named), then log, linear, or by time point; the
    uptake page is drawn again with the choice (owner: uptake usually read on a log axis)."""
    assert page.time_axis.itemText(0) == "[figures] scale: linear"
    series = page.tree.topLevelItem(0).child(0)
    with qtbot.waitSignal(page.panel("uptake").drawn, timeout=30000):
        page.tree.setCurrentItem(series)
    ax = page.panel("uptake").view.figure.axes[0]
    assert ax.get_xscale() == "linear" and page.view_settings() == {}
    with qtbot.waitSignal(page.panel("uptake").drawn, timeout=30000):
        page.set_time_axis("log")
    assert page.panel("uptake").view.figure.axes[0].get_xscale() == "log"
    assert page.view_settings() == {"time_axis": "log"}
    with qtbot.waitSignal(page.panel("uptake").drawn, timeout=30000):
        page.set_time_axis("index")
    labels = [t.get_text() for t in page.panel("uptake").view.figure.axes[0].get_xticklabels()]
    assert labels[0] == "UN" and labels[-1] == "FD"


def test_a_change_to_the_projects_figures_reaches_the_results_shown(qtbot, tmp_path):
    """Results opened through the project carry its [figures] and [review]: changing those in
    the file (here as a text editor would) loads them again, keeping the place in the tree
    (owner: a scale changed to log was not respected)."""
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox

    from hxex.cli import main
    from hxex.gui.window import MainWindow
    from hxex.project_edit import set_setting
    from test_cli import mzml_project

    project = mzml_project(tmp_path)
    assert main(["-q", "run", str(project)]) == 0
    window = MainWindow(settings=QSettings(str(tmp_path / "g.ini"), QSettings.Format.IniFormat))
    window.ask_to_save = lambda: QMessageBox.StandardButton.Discard
    qtbot.addWidget(window)
    window.open_asking(project)
    qtbot.waitUntil(lambda: not window.loading and window.results_path == project.absolute(),
                    timeout=30000)
    page = window.results_page
    with qtbot.waitSignal(page.panel("uptake").drawn, timeout=30000):
        page.tree.setCurrentItem(page.tree.topLevelItem(0).child(0))
    chosen = page.selection()
    assert page.panel("uptake").view.figure.axes[0].get_xscale() == "linear"
    set_setting(project, "figures.scale", "log")
    window.project_page.editor._file_changed(str(project))  # as the watcher reports it
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    assert page.time_axis.itemText(0) == "[figures] scale: log" and page.selection() == chosen
    qtbot.waitUntil(lambda: page.panel("uptake").view.figure is not None
                    and page.panel("uptake").view.figure.axes[0].get_xscale() == "log",
                    timeout=30000)
    # a setting that does not change results already made: no reload
    loads = []
    window._loading.submit = lambda *a, **k: loads.append(1)
    set_setting(project, "run.bootstrap.n_boot", "7")
    window.project_page.editor._file_changed(str(project))
    assert loads == []
    # [analysis]: no reload, the tree's 'left out' column worked out again with the settings
    told = []
    page.analysis_changed = told.append
    set_setting(project, "analysis.exclude_flagged", "true")
    window.project_page.editor._file_changed(str(project))
    assert loads == [] and len(told) == 1 and told[0].exclude_flagged


def test_what_the_analysis_leaves_out_is_shown_at_every_level(qtbot, page, results):
    """Each spectrum says why the analysis settings leave it out (owner: which flags exclude a
    spectrum); its time point, series and peptide count them; a replicate's row says whether it
    is flagged or rejected; a change of settings is shown on the rows already made."""
    from hxex.analysis import AnalysisConfig
    from hxex.gui.results_tree import left_out

    left = left_out(results, AnalysisConfig(exclude_times=(60,)))
    one_min = np.isclose(results.fit.time, 60)
    assert all(w == "time left out (exclude_times)" for w in np.asarray(left)[one_min])
    assert not any(np.asarray(left)[~one_min])
    peps = peptides(results, left)
    assert peps[0].series[0].n_left_out == 3 and shown(peps[0], peps[0].series[0], "left_out")
    assert left_out(results, AnalysisConfig(filters=None)) == [""] * len(results.fit)

    series = _open_series(page)
    point = series.child(1)
    assert point.text(COLUMN["left out"]) == "" and series.text(COLUMN["left out"]) == ""
    page.analysis_changed(AnalysisConfig(exclude_times=(60,)))
    qtbot.waitUntil(lambda: point.text(COLUMN["left out"]) == "3", timeout=30000)
    assert point.child(0).text(COLUMN["left out"]) == "time left out (exclude_times)"
    assert series.text(COLUMN["left out"]) == "3"
    assert page.tree.topLevelItem(0).text(COLUMN["left out"]) == "6"  # both series' 1 min
    page.filter.setCurrentIndex(page.filter.findData("left_out"))
    assert page.count.text() == "2 of 2 series"
    page.analysis_changed(AnalysisConfig(exclude_times=(1e6,)))  # no such time
    qtbot.waitUntil(lambda: page.tree.topLevelItem(0) is None
                    or page.tree.topLevelItem(0).text(COLUMN["left out"]) == "", timeout=30000)


def test_a_time_point_of_one_replicate_shows_that_spectrum(qtbot, results, tmp_path):
    """A level with one item shows that item's view (owner: stepping into it drew the same
    spectrum another way)."""
    from hxex.gui.results_workspace import ResultsWorkspace

    page = ResultsWorkspace()
    qtbot.addWidget(page)
    page.show_results(results, None)
    series = page.tree.topLevelItem(0).child(0)
    page.tree.expandItem(series)
    fd = series.child(series.childCount() - 1)
    assert fd.childCount() == 2
    fd.data(0, 257)[2].spectra.pop()  # the time point now holds one replicate
    with qtbot.waitSignal(page.panel("spectrum").drawn, timeout=30000):
        page.tree.setCurrentItem(fd)
    assert [page.figures.tabText(i) for i in range(page.figures.count())] == [
        "spectrum", "fits at each K", "uptake"]
    assert "z=2, FD, rep 1" in page.review.heading.text()  # the spectrum, by name


def test_stepping_by_charge_goes_to_the_next_series(page):
    top = page.tree.topLevelItem(0)
    assert top.childCount() == 2
    first = top.child(0)
    page.tree.expandItem(first)
    page.tree.setCurrentItem(first.child(1).child(0))  # a spectrum of the first series
    assert page.step_spectrum(1, "series") and page.tree.currentItem() is top.child(1)
    assert page.step_spectrum(-1, "series") and page.tree.currentItem() is first
    assert page.step_spectrum(1, "spectrum")  # from a series: its first spectrum
    assert page.tree.currentItem() is first.child(0).child(0)
