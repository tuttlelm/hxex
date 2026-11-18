# SPDX-License-Identifier: GPL-3.0-or-later
"""The Results workspace's tree (hxex.gui.results_tree, hxex.gui.results_workspace): peptide,
charge, time point, replicate; the figures of each level; filters and search; a selection
restored by what outlasts a rerun. Runs offscreen."""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from hxex.gui.results_tree import peptides, points, shown
from hxex.gui.results_workspace import COLUMN, KIND, ResultsWorkspace
from hxex.report import Results
from hxex.report.series import plot_series_spectra
from test_run import config, raw_spectra, run


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
    page.tree.setCurrentItem(point)
    tabs = [page.figures.tabText(i) for i in range(page.figures.count())]
    assert tabs == ["spectra", "fits at each K", "uptake", "data", "overview"]
    # a tab's figure is drawn when the tab is in view (audit of 2026-11-15, M-3): the overview
    # stayed in view, so nothing was drawn yet
    assert page.figures.currentWidget() is page.panel("overview")
    assert set(page._pending_draws) == {"spectra", "fits at each K", "uptake"}
    with qtbot.waitSignal(page.panel("spectra").drawn, timeout=30000):
        page.figures.setCurrentWidget(page.panel("spectra"))
    assert len(page.panel("spectra").view.figure.axes) == 3  # that time's replicates
    with qtbot.waitSignal(page.panel("fits at each K").drawn, timeout=30000):
        page.figures.setCurrentWidget(page.panel("fits at each K"))
    title = page.panel("fits at each K").view.figure.axes[0].get_title()
    assert title.splitlines()[-1].startswith("rep 1: K=")
    assert "uptake" in page._pending_draws  # still waiting for its tab
    with qtbot.waitSignal(page.panel("fits at each K").drawn, timeout=30000):
        page.tree.setCurrentItem(point.child(0))  # the tab in view draws the new selection
    assert [page.figures.tabText(i) for i in range(page.figures.count())] == [
        "spectrum", "fits at each K", "uptake", "data", "overview"]
    with qtbot.waitSignal(page.panel("spectrum").drawn, timeout=30000):
        page.figures.setCurrentWidget(page.panel("spectrum"))
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


def test_the_overview_is_the_first_look_and_a_tab_at_every_level(qtbot, results):
    """Results open on the run's overview (owner: the summary pages, as hxex report writes them);
    every level keeps it as its last tab, drawn once per results and kept; a review decision
    (new results) draws it again."""
    import dataclasses

    from hxex.gui.results_workspace import ResultsWorkspace

    page = ResultsWorkspace()
    qtbot.addWidget(page)
    page.resize(1200, 800)
    page.show()
    overview = page.panel("overview")
    with qtbot.waitSignal(overview.drawn, timeout=30000):
        page.show_results(results, None)
    assert [page.figures.tabText(i) for i in range(page.figures.count())] == ["overview"]
    pages = [page.overview_page.itemData(i) for i in range(page.overview_page.count())]
    assert pages[0] == "k_selection" and set(pages) <= {"k_selection", "controls", "quality"}
    assert overview.save_name == "overview_k_selection"
    assert "Mouse over" not in overview.help_text  # no tooltips on the Results figures
    first = overview.view.figure
    if len(pages) > 1:  # another page: drawn, then kept
        with qtbot.waitSignal(overview.drawn, timeout=30000):
            page.overview_page.setCurrentIndex(1)
        assert overview.save_name == f"overview_{pages[1]}"
        page.overview_page.setCurrentIndex(0)
        assert overview.view.figure is first and not overview.drawing
    # a series selected: its pages, the overview last and still in view, not drawn again
    series = page.tree.topLevelItem(0).child(0)
    page.tree.setCurrentItem(series)
    assert [page.figures.tabText(i) for i in range(page.figures.count())] == [
        "spectra", "uptake", "populations", "data", "overview"]
    assert page.figures.currentWidget() is overview and overview.view.figure is first
    # new results (as a review decision makes): drawn again
    page.results = dataclasses.replace(results)
    with qtbot.waitSignal(overview.drawn, timeout=30000):
        page._redraw()
    assert overview.view.figure is not first


def test_a_peptide_and_its_first_charge_are_drawn_once(qtbot, page):
    """A peptide shows its first charge's pages: moving between the two draws nothing again
    (owner, 2026-11-07); another time point does."""
    peptide = page.tree.topLevelItem(0)
    series = _open_series(page)
    page.tree.setCurrentItem(series)
    with qtbot.waitSignal(page.panel("uptake").drawn, timeout=30000):
        page.figures.setCurrentWidget(page.panel("uptake"))
    figure = page.panel("uptake").view.figure
    with qtbot.assertNotEmitted(page.panel("uptake").drawn, wait=500):
        page.tree.setCurrentItem(peptide)
    with qtbot.assertNotEmitted(page.panel("uptake").drawn, wait=500):
        page.tree.setCurrentItem(series)
    assert page.panel("uptake").view.figure is figure
    assert page.selection()["charge"] == 2
    with qtbot.waitSignal(page.panel("uptake").drawn, timeout=30000):
        page.tree.setCurrentItem(series.child(1))


def test_the_time_axis_is_chosen_in_the_window(qtbot, page):
    """The project's [figures] scale first (named), then log, linear, or by time point; the
    uptake page is drawn again with the choice (owner: uptake usually read on a log axis)."""
    assert page.time_axis.itemText(0) == "[figures] scale: linear"
    series = page.tree.topLevelItem(0).child(0)
    page.tree.setCurrentItem(series)
    with qtbot.waitSignal(page.panel("uptake").drawn, timeout=30000):
        page.figures.setCurrentWidget(page.panel("uptake"))
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


def test_the_fits_at_each_k_show_their_residuals_when_ticked(qtbot, page):
    """The fits at each K tab's bar has a residuals box (owner, 2026-11-17): ticked, each fit
    gets its residual panel; the choice is the window's, kept in the session."""
    series = _open_series(page)
    page.tree.setCurrentItem(series.child(1).child(0))
    panel = page.panel("fits at each K")
    with qtbot.waitSignal(panel.drawn, timeout=30000):
        page.figures.setCurrentWidget(panel)
    n = len(panel.view.figure.axes)
    assert page.view_settings() == {}
    with qtbot.waitSignal(panel.drawn, timeout=30000):
        page.k_residuals.setChecked(True)
    assert len(panel.view.figure.axes) == 2 * n
    assert page.view_settings() == {"k_residuals": True}
    with qtbot.waitSignal(panel.drawn, timeout=30000):
        page.set_k_residuals(False)
    assert len(panel.view.figure.axes) == n


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
    page.tree.setCurrentItem(page.tree.topLevelItem(0).child(0))
    with qtbot.waitSignal(page.panel("uptake").drawn, timeout=30000):
        page.figures.setCurrentWidget(page.panel("uptake"))
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
    # nothing worked out until asked (owner): the column says it is out of date
    assert page.left_bar.isVisibleTo(page) and not page._left_task.running
    assert "out of date: the [analysis] settings changed" in page.left_note.text()
    assert point.text(COLUMN["left out"]) == ""
    page.left_update.click()
    qtbot.waitUntil(lambda: point.text(COLUMN["left out"]) == "3", timeout=30000)
    assert not page.left_bar.isVisibleTo(page)
    assert point.child(0).text(COLUMN["left out"]) == "time left out (exclude_times)"
    assert series.text(COLUMN["left out"]) == "3"
    assert page.tree.topLevelItem(0).text(COLUMN["left out"]) == "6"  # both series' 1 min
    page.filter.setCurrentIndex(page.filter.findData("left_out"))
    assert page.count.text() == "2 of 2 series"
    page.analysis_changed(AnalysisConfig(exclude_times=(1e6,)))  # no such time
    page.update_left_out()
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
    page.tree.setCurrentItem(fd)
    with qtbot.waitSignal(page.panel("spectrum").drawn, timeout=30000):
        page.figures.setCurrentWidget(page.panel("spectrum"))
    assert [page.figures.tabText(i) for i in range(page.figures.count())] == [
        "spectrum", "fits at each K", "uptake", "data", "overview"]
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


def test_the_left_out_column_does_not_repeat_the_controls_messages():
    """Worked out again after every review decision, the left-out column computed the controls
    each time and repeated their messages (owner); its own thread is quiet meanwhile, other
    threads are not."""
    import logging
    import threading

    from hxex.gui.results_tree import _quiet_here

    seen = []

    class Keep(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    handler = Keep()
    logger = logging.getLogger("hxex")
    logger.addHandler(handler)
    old = logger.level
    logger.setLevel(logging.INFO)
    try:
        with _quiet_here():
            logging.getLogger("hxex.controls").warning("from the left-out thread")
            other = threading.Thread(
                target=lambda: logging.getLogger("hxex.analysis").warning("from another"))
            other.start()
            other.join()
        logging.getLogger("hxex.controls").warning("after")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old)
    assert seen == ["from another", "after"]


def test_the_list_holds_every_spectrum_and_leads_to_it(qtbot, page, results):
    """Every spectrum as a row with spectra.csv's columns (owner: a flat list beside the
    tree); a row chosen shows that spectrum as the tree does; the review's arrows step in the
    list's order; the filter, sample and search narrow it."""
    from PySide6.QtCore import Qt

    qtbot.waitUntil(lambda: bool(page.list.rows), timeout=30000)
    lst = page.list
    assert len(lst.rows) == len(results.fit)
    assert {"spectrum_id", "sample", "K", "review_status", "left_out", "flags"} <= set(
        lst.columns)
    assert "_index" not in lst.columns and lst.count.text().endswith(f"{len(results.fit)} spectra")
    page.left_tabs.setCurrentWidget(lst)
    # sorted by a column: the arrows follow that order
    time = lst.columns.index("time")
    lst.grid.sortByColumn(time, Qt.SortOrder.DescendingOrder)
    order = [r["spectrum_id"] for r in lst.model.rows]
    assert lst.select(order[0])
    assert page.tree.currentItem().toolTip(0) == order[0]  # the tree shows that spectrum
    assert page.step_spectrum(1) and lst.current_row()["spectrum_id"] == order[1]
    assert page.tree.currentItem().toolTip(0) == order[1]
    assert page.step_spectrum(-1) and lst.current_row()["spectrum_id"] == order[0]
    # narrowed: a search, then a filter keeping nothing here
    lst.search.setText(order[0])
    assert [r["spectrum_id"] for r in lst.model.rows] == [order[0]]
    lst.search.clear()
    lst.filter.setCurrentIndex(lst.filter.findData("flagged"))
    assert lst.model.rowCount() == sum(bool(r["flags"]) for r in lst.rows)


def test_the_lists_columns_filter_by_a_right_click(qtbot, page, results):
    """A column's header filters by its values (owner): ticks for few values (a time by its
    label), a range for many numbers (a time with units), text for much text; the filters
    combine, the header says so, and Clear brings every row back."""
    from PySide6.QtCore import Qt

    from hxex.gui.spectra_list import ColumnFilterDialog, _bound

    qtbot.waitUntil(lambda: bool(page.list.rows), timeout=30000)
    lst = page.list
    n = len(lst.rows)
    assert "time_label" in lst.columns and lst.filter_kind("time_label") == "values"
    labels = lst.values_of("time_label")
    assert labels[0] == "UN" and labels[-1] == "FD" and "1 min" in labels  # in time order
    lst.set_column_filter("time_label", ("values", {"1 min"}))
    assert {r["time_label"] for r in lst.model.rows} == {"1 min"}
    header = lst.model.headerData(lst.columns.index("time_label"), Qt.Orientation.Horizontal)
    assert header == "time_label ▼" and not lst.clear_filters.isHidden()
    lst.set_column_filter("sample", ("values", {"nowhere"}))  # combined: nothing left
    assert lst.model.rowCount() == 0
    lst.clear_column_filters()
    assert lst.model.rowCount() == n and lst.clear_filters.isHidden()
    # a time range with units, and text contained
    assert _bound("1 min", "time") == 60.0 and _bound("", "time") is None
    with pytest.raises(ValueError, match="not a time"):
        _bound("soon", "time")
    lst.set_column_filter("time", ("range", 30.0, _bound("2 min", "time")))
    assert {r["time"] for r in lst.model.rows} == {60.0}
    lst.clear_column_filters()
    lst.set_column_filter("spectrum_id", ("contains", "#2"))
    assert all("#2" in r["spectrum_id"] for r in lst.model.rows) and lst.model.rowCount() >= 1
    lst.clear_column_filters()
    # the kinds: many numbers, much text
    many = [{**lst.rows[0], "centroid": float(i), "note": f"n{i}", "_index": i}
            for i in range(40)]
    lst.set_rows(lst.columns, many)
    assert lst.filter_kind("centroid") == "range" and lst.filter_kind("note") == "contains"
    # the window: ticks, a range, text
    ticks = ColumnFilterDialog(lst, "sample")
    assert ticks.kind == "values" and ticks.rule() is None  # every value ticked: no filter
    ticks.ticks.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert ticks.rule() == ("values", set())
    span = ColumnFilterDialog(lst, "centroid")
    span.low.setText("5")
    assert span.rule() == ("range", 5.0, None)
    text = ColumnFilterDialog(lst, "note")
    text.text.setText("n3")
    assert text.rule() == ("contains", "n3")


def test_the_search_reads_residue_numbers():
    """A number is a residue the peptide covers, two its residues however padded (owner:
    42-54 did not find 0042-0054); every word must match."""
    from hxex.gui.results_tree import search_matches

    row = ("WT", "0042-0054", "TSLSPFYLRPPSF")
    assert search_matches("42-54", *row) and search_matches("50", *row)
    assert not search_matches("41", *row) and not search_matches("12-23", *row)
    assert search_matches("wt 50", *row) and not search_matches("d109h 50", *row)
    assert search_matches("spfy", *row) and search_matches("z2", *row, "z2")


def test_every_figure_tab_says_what_it_shows(qtbot, page):
    """Each figure tab has its tooltip at every level (owner, 2026-11-13), the data tab's kept
    when its name gains its count."""
    from hxex.gui.results_workspace import TAB_TIPS
    from hxex.gui.tips import tip

    series = _open_series(page)  # its time points and spectra are filled when it opens
    point = series.child(1)
    for item in (series, point, point.child(0)):
        page.tree.setCurrentItem(item)
        for at in range(page.figures.count()):
            name = page.figures.tabText(at).split(" (")[0]
            assert page.figures.tabToolTip(at) == tip(TAB_TIPS[name]), name
    assert "the components" in TAB_TIPS["populations"]


def test_the_data_tab_holds_the_selections_report_rows(qtbot, page, results):
    """Beside the figures, the selection's rows of the report's tables (owner): a series'
    uptake.csv rows with its series.csv values said above them, a time point's spectra.csv
    rows, a spectrum's row as names and values; a row double-clicked opens its spectrum."""
    from hxex.report.tables import uptake_table

    series = page.tree.topLevelItem(0).child(0)
    page.tree.setCurrentItem(series)
    tabs = [page.figures.tabText(i) for i in range(page.figures.count())]
    assert tabs[-2:] == ["data", "overview"]
    page.figures.setCurrentWidget(page.table_view)
    view = page.table_view
    want = uptake_table(results, data_ids=[0])
    assert view.model.columns == list(want) and len(view.model.rows) == len(want["data_id"])
    # named with its rows, as the Analysis data tab (owner, 2026-11-05)
    at = page.figures.indexOf(view)
    assert page.figures.tabText(at) == f"data ({len(want['data_id'])})"
    assert view.note.text().startswith("uptake.csv: this series' rows, every K")
    assert "series.csv: n_spectra" in view.note.text()
    assert not page.series_table.isHidden()
    # its populations page's rows (components.csv), or its spectra's (spectra.csv)
    page.series_table.setCurrentText("components.csv")
    assert {"rank", "mean", "population"} <= set(view.model.columns)
    assert {r["data_id"] for r in view.model.rows} == {0}
    assert view.note.text().startswith("components.csv: this series' populations")
    page.series_table.setCurrentText("spectra.csv")
    assert "spectrum_id" in view.model.columns and {r["data_id"] for r in view.model.rows} == {0}
    page.series_table.setCurrentText("uptake.csv")
    page.tree.expandItem(series)
    point = series.child(1)
    page.tree.setCurrentItem(point)
    assert page.figures.currentWidget() is view  # the tab stays in view
    assert {r["spectrum_id"] for r in view.model.rows} == {s.spectrum_id for s in
                                                            point.data(0, 257)[2].spectra}
    first = view.model.rows[0]["spectrum_id"]
    view._double_clicked(view.model.index(0, 0))  # a row: its spectrum
    assert page.tree.currentItem().toolTip(0) == first
    assert page.series_table.isHidden()  # a time point's rows: its spectra alone
    leaf = point.child(0)
    page.tree.setCurrentItem(leaf)
    assert view.model.columns == ["column", "value"]
    assert page.figures.tabText(page.figures.indexOf(view)) == "data (1)"
    values = {r["column"]: r["value"] for r in view.model.rows}
    assert values["spectrum_id"] == leaf.toolTip(0) and "K" in values


def test_the_list_shows_the_reports_other_tables(qtbot, page, results, tmp_path):
    """series.csv, components.csv and uptake.csv in the list too (owner: the Results tables),
    each made once; a series row opens its series; the rows shown are saved as CSV; after a
    decision a table other than spectra.csv says it is out of date until Update."""
    import csv
    import dataclasses

    from hxex.report.tables import series_table, uptake_table

    qtbot.waitUntil(lambda: bool(page.list.rows), timeout=30000)
    lst = page.list
    page.left_tabs.setCurrentWidget(lst)
    lst.tables.setCurrentIndex(lst.tables.findData("series"))
    qtbot.waitUntil(lambda: lst.table_name == "series" and bool(lst.rows), timeout=30000)
    assert lst.columns == list(series_table(results)) and lst.filter.isHidden()
    row = lst.model.rows[1]
    lst._current(lst.model.index(1, 0), None)  # a series row chosen: its series
    item = page.tree.currentItem()
    assert item.data(0, 256) == "series" and item.data(0, 257)[1].data_id == row["data_id"]
    lst.tables.setCurrentIndex(lst.tables.findData("uptake"))
    qtbot.waitUntil(lambda: lst.table_name == "uptake" and bool(lst.rows), timeout=30000)
    assert len(lst.rows) == len(uptake_table(results)["data_id"])
    path = lst.save(tmp_path / "shown.csv")
    with open(path, encoding="utf-8") as fh:
        saved = list(csv.reader(fh))
    assert saved[0] == lst.model.columns and len(saved) == len(lst.model.rows) + 1
    # a decision (new results): this table is from before it, until Update
    page.results = dataclasses.replace(page.results)
    page._reviewed([0])
    assert not lst.stale_bar.isHidden() and "uptake.csv is from before" in lst.stale.text()
    lst.update_button.click()
    qtbot.waitUntil(lambda: lst.stale_bar.isHidden() and bool(lst.rows), timeout=30000)
    lst.tables.setCurrentIndex(lst.tables.findData("spectra"))  # kept: no rebuild
    assert lst.table_name == "spectra" and len(lst.rows) == len(results.fit)


def test_a_reread_keeps_the_chosen_table_coming(qtbot, page, results):
    """uptake.csv chosen, Update pressed, then the results read again (a first decision names
    the review table): the table chosen is made for the new results, not left at "Making ..."
    (owner: it seemed stuck; one background slot dropped the Update for the reread)."""
    import dataclasses

    qtbot.waitUntil(lambda: bool(page.list.rows), timeout=30000)
    lst = page.list
    page.left_tabs.setCurrentWidget(lst)
    lst.tables.setCurrentIndex(lst.tables.findData("uptake"))
    qtbot.waitUntil(lambda: lst.table_name == "uptake" and bool(lst.rows), timeout=30000)
    n = len(lst.rows)
    lst.update_button.click()  # made again ...
    again = dataclasses.replace(results)
    page.show_results(again, None)  # ... while the results are read again
    qtbot.waitUntil(lambda: lst.table_name == "uptake" and len(lst.rows) == n
                    and page._list_tables.get("uptake", (None,))[0] is again, timeout=30000)
    assert lst.count.text().endswith("rows") or "of" in lst.count.text()
    qtbot.waitUntil(lambda: "spectra" in page._list_tables, timeout=30000)  # made too


def test_the_list_copies_a_row_or_every_row_shown(qtbot, page):
    """Ctrl+C and the right-click menu copy the row chosen, or every row shown, with the
    header (owner: a copy menu on the tables)."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    qtbot.waitUntil(lambda: bool(page.list.rows), timeout=30000)
    lst = page.list
    sid = str(lst.model.rows[2]["spectrum_id"])
    assert lst.select(sid)
    qtbot.keyClick(lst.grid, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    lines = QApplication.clipboard().text().splitlines()
    assert lines[0].split("\t") == lst.model.columns and len(lines) == 2 and sid in lines[1]
    one, every = lst.row_menu().actions()
    assert one.text() == "Copy row" and every.text().startswith("Copy rows shown")
    every.trigger()
    assert len(QApplication.clipboard().text().splitlines()) == len(lst.model.rows) + 1


def test_the_fragment_columns_and_filters_show_only_for_fragment_runs(page):
    """Peptides read from a table (no scan targets): no ion, MS level or condition columns and no
    MS level or ion type lists; those come with mzML/Thermo results (tested in test_mzml)."""
    from hxex.gui.results_workspace import TARGET_COLUMNS

    assert all(page.tree.isColumnHidden(COLUMN[name]) for name in TARGET_COLUMNS)
    assert page.ms_filter.isHidden() and page.ion_filter.isHidden()
    assert not page.tree.isColumnHidden(COLUMN["spectra"])


def test_a_fragment_series_has_its_parents_fragments_tab(qtbot, page, monkeypatch):
    """Results with fragment ions get a fragments tab at every level (owner, 2026-11-17): every
    fragment of the selection's parent, the one selected thicker; a click on a fragment's name
    selects its series. Results without fragments have no such tab (above). The fragment table
    is a synthetic one whose series ids are this run's."""
    import numpy as np

    from hxex.gui.results_workspace import PAYLOAD
    from hxex.report.fragments import SERIES_GID
    from test_fragment_kinetics import synthetic

    u = synthetic(labels=("M:8", "c:1", "c:2", "c:4"))
    u = u.select(np.isin(u.data_id, [0, 1, 2, 3]))
    why = np.full(len(u), "", dtype=object)
    monkeypatch.setattr(type(page), "_with_fragments", lambda self: True)
    monkeypatch.setattr(type(page), "_fragment_table",
                        lambda self, results, flags=(): (u, why, np.zeros(len(u), dtype=bool)))
    page._show_tabs("start")  # the tabs again, now with fragments
    second = page.tree.topLevelItem(0).child(1)  # data_id 1: c:1 in the table
    page.tree.setCurrentItem(second)
    tabs = [page.figures.tabText(i) for i in range(page.figures.count())]
    assert "fragments" in tabs
    panel = page.panel("fragments")
    with qtbot.waitSignal(panel.drawn, timeout=30000):
        page.figures.setCurrentWidget(panel)  # drawn when opened
    (ax,) = panel.view.figure.axes
    texts = ax.get_legend().get_texts()
    assert [t.get_text() for t in texts] == ["c_1_z1", "c_2_z1", "c_4_z1"]
    assert [t.get_text() for t in texts if t.get_fontweight() == "bold"] == ["c_1_z1"]
    assert texts[0].get_gid() == f"{SERIES_GID}1"
    panel.clicked.emit(f"{SERIES_GID}0")  # a name of series 0: the tree moves there
    assert page.tree.currentItem().data(0, PAYLOAD)[1].data_id == 0
    panel.clicked.emit(f"{SERIES_GID}99")  # not in the tree: nothing moves
    assert page.tree.currentItem().data(0, PAYLOAD)[1].data_id == 0
    # its own time-axis list, one choice with the uptake page's (owner, 2026-11-17)
    box = page.fragment_time_axis
    assert box.itemData(0) is None and "kinetics" in box.itemText(0)
    with qtbot.waitSignal(panel.drawn, timeout=30000):
        box.setCurrentIndex(box.findData("index"))
    assert page.time_axis.currentData() == "index"
    assert panel.view.figure.axes[0].get_xlabel() == "time point"
    page.time_axis.setCurrentIndex(page.time_axis.findData("linear"))
    assert box.currentData() == "linear"


def _pooled(results, tmp_path, fasta: str, classes: list[str]):
    """``results`` as if read from pooled data with ``fasta`` as [input] pooled_fasta."""
    import dataclasses

    from hxex.io.pooled import candidates_record

    (tmp_path / "pool.fasta").write_text(fasta)
    record = candidates_record(tmp_path / "pool.fasta", classes)
    source = {"reader": "pooled", "pooled_candidates": record}
    configs = {**results.settings.configs, "spectra_source": source}
    return dataclasses.replace(results, settings=dataclasses.replace(results.settings,
                                                                     configs=configs))


def test_pooled_peptides_show_their_identity_and_filter_by_level_and_class(qtbot, results,
                                                                           tmp_path):
    """Pooled step 6 (owner, 2026-11-10): the tree names what each pooled peptide may be, and
    narrows by identity level and class; the list's search finds a peptide by its candidates.
    Results without a pooled FASTA show neither."""
    pooled = _pooled(results, tmp_path, ">A1\nRDKVQKEYALGG\n>A2\nRDKVQKEYALSS\n>B1\nMMMMM\n",
                     ["A", "B"])
    pep = peptides(pooled)[0]
    assert (pep.identity, pep.identity_level, pep.classes, pep.candidates) == (
        "A", "class", "A", "A1;A2")
    series = pep.series[0]
    assert shown(pep, series, identity_level="class") and shown(pep, series, pool_class="A")
    assert not shown(pep, series, identity_level="isoform")
    assert not shown(pep, series, pool_class="B")
    assert shown(pep, series, text="a2") and not shown(pep, series, text="b1")

    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(pooled, None)
    assert not w.tree.isColumnHidden(COLUMN["identity"])
    top = w.tree.topLevelItem(0)
    assert top.text(COLUMN["identity"]) == "A"
    assert "class level" in top.toolTip(COLUMN["identity"])
    assert "A1, A2" in top.toolTip(COLUMN["identity"])
    assert [w.level_filter.itemData(i) for i in range(w.level_filter.count())] == [None, "class"]
    assert [w.class_filter.itemText(i) for i in range(w.class_filter.count())] == [
        "every class", "A"]
    assert w.level_filter.isVisibleTo(w) and w.class_filter.isVisibleTo(w)
    w.level_filter.setCurrentIndex(1)
    assert w.tree.topLevelItemCount() == 1
    w.search.setText("b1")
    assert w.tree.topLevelItemCount() == 0
    assert w.select({"sample": pep.sample, "peptide_range": pep.peptide_range})
    assert w.level_filter.currentIndex() == 0 and not w.search.text()  # cleared to find it
    qtbot.waitUntil(lambda: bool(w.list.rows), timeout=30000)
    assert w.list.rows[0]["identity"] == "A"
    w.list.search.setText("A2")
    assert w.list.model.rowCount() == len(pooled.fit)
    w.list.search.setText("B1")
    assert w.list.model.rowCount() == 0

    plain = ResultsWorkspace()
    qtbot.addWidget(plain)
    plain.show_results(results, None)
    assert plain.tree.isColumnHidden(COLUMN["identity"])
    assert plain.level_filter.isHidden() and plain.class_filter.isHidden()


def test_ranges_sort_by_their_numbers():
    """Residue ranges (and other text with numbers) sort as a person reads them in the list's
    columns, its filter ticks and the fragment figures: 2-12 before 13-30 before 108-127."""
    from hxex.gui.data_table import _sort_key
    from hxex.gui.order import natural_key
    from hxex.gui.spectra_list import _order_key

    ranges = ["108-127", "13-30", "2-12", "2-13", "0013-0029"]
    expected = ["2-12", "2-13", "0013-0029", "13-30", "108-127"]
    assert sorted(ranges, key=natural_key) == expected
    assert sorted(ranges, key=_sort_key) == expected
    rows = [{"parent_range": r} for r in ranges]
    assert [r["parent_range"] for r in sorted(rows, key=lambda r: _order_key(r, "parent_range"))
            ] == expected
    stems = ["closure_S_108-127_z3_c-1+z-dot", "closure_S_13-30_z2_c-1+z-dot",
             "closure_S_2-12_z3_c-1+z-dot", "closure_S_2-12_z2_c-1+z-dot"]
    assert sorted(stems, key=natural_key) == [
        "closure_S_2-12_z2_c-1+z-dot", "closure_S_2-12_z3_c-1+z-dot",
        "closure_S_13-30_z2_c-1+z-dot", "closure_S_108-127_z3_c-1+z-dot"]
    assert _sort_key(3.0) < _sort_key("a") < _sort_key("")  # numbers, text, then empty


def test_the_elution_tab_reads_the_scan_files(qtbot, tmp_path, page):
    """Spectra read from scan files get an elution tab (PLAN Next steps 26): the selected
    spectrum's file drawn first, then with its replicates; results without scan files
    (the synthetic ones above) have no such tab."""
    from hxex.fit import FitConfig
    from hxex.io import MzmlRun, MzmlTarget, read_mzml
    from hxex.model import Backend
    from hxex.noise import NoiseConfig
    from hxex.picking import PickerConfig
    from hxex.pipeline import RunConfig, run
    from test_mzml import write_mzml
    from test_retention_search import PEPTIDES, _entry, lc_run

    series = _open_series(page)
    page.tree.setCurrentItem(series.child(1).child(0))
    assert "elution" not in [page.figures.tabText(i) for i in range(page.figures.count())]

    runs = [MzmlRun(str(write_mzml(tmp_path / "un1.mzML", lc_run(1))), "S", rep=1,
                    role="undeut"),
            MzmlRun(str(write_mzml(tmp_path / "un2.mzML", lc_run(2, shift=6.0))), "S", rep=2,
                    role="undeut")]
    rt = PEPTIDES["VDQIIMAK"]
    spectra = read_mzml(runs, [MzmlTarget(_entry("VDQIIMAK"), rt_window=(rt - 4, rt + 4))],
                        project="p")
    cfg = RunConfig(picker=PickerConfig(noise_threshold=100.0), quality=None,
                    fit=FitConfig(k_min=1, k_max=1, adam_steps=50, n_restarts=1,
                                  backend=Backend("cpu", "float64"),
                                  noise=NoiseConfig("gaussian", percent=3.0, estimate=None)),
                    bootstrap=None, controls=None)
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(Results.from_run(run(spectra, cfg)), None)
    series = _open_series(w)
    w.tree.setCurrentItem(series.child(0).child(1))  # UN, rep 2
    tabs = [w.figures.tabText(i) for i in range(w.figures.count())]
    assert tabs == ["spectrum", "fits at each K", "elution", "uptake", "data", "overview"]
    panel = w.panel("elution")
    assert panel.view.figure is None  # read only when in view
    w.figures.setCurrentIndex(tabs.index("elution"))

    def drawn(n):
        fig = panel.view.figure
        return fig is not None and sum(
            str(line.get_gid()).startswith("trace:") for line in fig.axes[0].get_lines()) == n

    qtbot.waitUntil(lambda: drawn(2), timeout=30000)  # its replicate behind it
    fig = panel.view.figure
    assert fig._suptitle.get_text() == "S 0001-0008 VDQIIMAK z=2, UN rep 2"  # which file
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert legend[0] == "this spectrum: undeuterated, rep 2 (un2.mzML)"
    assert legend[-1] == "its replicates (1)"


def test_the_export_folder_belongs_to_the_results_shown(page, results, tmp_path):
    """After an export from one project, every export from the next was suggested into the
    first project's folder (audit of 2026-11-15, E-10)."""
    page.export_folder = tmp_path / "a_exports"
    page.show_results(results, None)  # the same results again: kept
    assert page.export_folder == tmp_path / "a_exports"
    page.show_results(results, tmp_path / "b" / "out")
    assert page.export_folder is None
    assert page.default_export_folder() == tmp_path / "b" / "hxex_exports"


# --- the audit of 2026-11-15, scope C (the tree, the List, the figure view, the data tab) ----


def two_form_results():
    """Sample S with two peptides at the same residues 0001-0010 (an unmodified and a
    substituted sequence, as two forms of a pool), each at charge 2."""
    from hxex.picking import plan_bins
    from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet
    from test_bundle import envelope, natural
    from test_picking import sticks
    from test_run import LAYOUT, PICK, control

    spectra = []
    for s, peptide in enumerate(("RDKVQKEYAL", "RDKVQKEYAV")):
        plan = plan_bins(peptide, 2, config=PICK)
        entry = PeptideEntry(f"f{s}.xlsx", "S", peptide, "0001-0010", 1, 10, 2)
        for k, (t, rep, centre, width) in enumerate(LAYOUT):
            height = 4e4 * (0.9 + 0.1 * (k % 3))
            peaks = (natural(plan, height) if t == 0.0
                     else envelope(plan, centre + 0.5 * s, width, height))
            mz, y = sticks(plan, peaks)
            y = y + np.random.default_rng(100 * s + k).normal(0, 15, len(y))
            time, role = control(t)
            spectra.append(Spectrum(mz, y, entry, time, rep, data_id=s, label=f"s{s}c{k}",
                                    spectrum_id=f"proj:f{s}.xlsx#{k + 1}", role=role))
    return Results.from_run(run(SpectrumSet.from_spectra(spectra), config(bootstrap=None)))


@pytest.fixture(scope="module")
def two_forms():
    return two_form_results()


def _peptide_shown(page):
    item = page.tree.currentItem()
    kind, payload = item.data(0, 256), item.data(0, 257)
    return payload.peptide if kind == "peptide" else payload[0].peptide


def test_two_forms_at_one_range_are_told_apart(qtbot, two_forms):
    """Two peptides at the same residues in one sample (a modified and an unmodified form, two
    forms of a pool): the selection carries the sequence, so a go-to from Analysis and a
    selection restored after a reread land on the right one (audit of 2026-11-15, C-7)."""
    page = ResultsWorkspace()
    qtbot.addWidget(page)
    page.show_results(two_forms, None)
    assert page.tree.topLevelItemCount() == 2
    assert page.select_peptide("S", 1, 10, "RDKVQKEYAV")
    assert _peptide_shown(page) == "RDKVQKEYAV"
    assert page.selection()["peptide"] == "RDKVQKEYAV"
    second = page.tree.topLevelItem(1)
    for level in ("peptide", "series", "point"):
        item = second
        if level != "peptide":
            item = second.child(0)
            page.tree.expandItem(item)
            if level == "point":
                item = item.child(1)
        page.tree.setCurrentItem(item)
        chosen = page.selection()
        page.tree.setCurrentItem(page.tree.topLevelItem(0))
        assert page.select(chosen) and _peptide_shown(page) == "RDKVQKEYAV", level
    # a selection saved before the sequence was carried still finds the first form
    assert page.select({"sample": "S", "peptide_range": "0001-0010", "charge": 2})
    assert _peptide_shown(page) == "RDKVQKEYAL"


def test_a_find_keeps_the_row_and_the_place(qtbot, page):
    """Typing in Find rebuilds the tree with the row selected still selected; when the text
    leaves it out, the figures and the review panel still show it, so a decision redraws it
    and the selection is still known for a reread or a session (audit of 2026-11-15, C-8)."""
    series = _open_series(page)
    leaf = series.child(1).child(0)
    page.tree.setCurrentItem(leaf)
    chosen = page.selection()
    page.figures.setCurrentWidget(page.panel("spectrum"))
    qtbot.waitUntil(lambda: page.panel("spectrum").view.figure is not None, timeout=30000)
    page.search.setText("S")  # still finds the peptide: the row stays selected
    assert page.tree.currentItem() is not None and page.selection() == chosen
    page.search.setText("nothing")  # the tree is empty, the figures are not
    assert page.tree.topLevelItemCount() == 0 and page.tree.currentItem() is None
    assert page.selection() == chosen and "rep 1" in page.review.heading.text()
    with qtbot.waitSignal(page.panel("spectrum").drawn, timeout=30000):
        page._redraw()  # as a decision does
    page.search.clear()
    assert page.selection() == chosen and page.tree.currentItem() is not None


def test_the_counts_the_filters_use_follow_the_decisions(qtbot, page, results):
    """A rejected spectrum counts in its series' row, in the next rebuild of the tree, and in
    the 'rejected' filter (audit of 2026-11-15, C-9: the filters used the counts from when the
    results opened)."""
    from hxex.io.bundle import ReviewTable

    series = page.tree.topLevelItem(0).child(0)
    page.tree.setCurrentItem(series)
    data_id = series.data(0, 257)[1].data_id
    i = int(np.flatnonzero(results.fit.data_id == data_id)[0])
    review = (results.review if results.review is not None else ReviewTable.empty()).update(
        str(results.fit.spectrum_id[i]), status="rejected")
    page.results = results.with_review(review)
    page._reviewed([i])
    assert series.text(COLUMN["rejected"]) == "1"
    page.filter.setCurrentIndex(page.filter.findData("rejected"))
    assert page.tree.topLevelItemCount() == 1
    assert page.tree.topLevelItem(0).child(0).data(0, 257)[1].data_id == data_id
    assert page.tree.topLevelItem(0).child(0).text(COLUMN["rejected"]) == "1"
    page.filter.setCurrentIndex(page.filter.findData("unreviewed"))
    shown = [page.tree.topLevelItem(0).child(j).data(0, 257)[1].data_id
             for j in range(page.tree.topLevelItem(0).childCount())]
    assert data_id not in shown  # reviewed now
    page.filter.setCurrentIndex(0)


def test_a_series_data_row_opens_its_spectrum_before_the_series_is_opened(qtbot, page):
    """Double-clicking a spectrum's row in a series' data tab opens that spectrum, also when
    the series was never opened in the tree (audit of 2026-11-15, C-10)."""
    series = page.tree.topLevelItem(0).child(1)
    assert not series.isExpanded() and series.childCount() == 0
    page.tree.setCurrentItem(series)
    page.figures.setCurrentWidget(page.table_view)
    page.series_table.setCurrentText("spectra.csv")
    rows = page.table_view.model.rows
    assert rows and "spectrum_id" in rows[0]
    page.table_view.activated.emit(rows[3])  # as a double-click
    item = page.tree.currentItem()
    assert item.data(0, 256) == "spectrum"
    assert item.data(0, 257)[3].spectrum_id == rows[3]["spectrum_id"]
    page.series_table.setCurrentText("uptake.csv")


def test_the_data_tab_is_keyed_by_the_selection(qtbot, two_forms):
    """The data tab remembers what it shows by what names it, not by the address of the
    selection's tuple, which Python gives to a later selection (audit of 2026-11-15, C-11);
    the same selection drawn again keeps the tab's count (C-21); the note says the rows are
    the CSV tables' (open question 3)."""
    page = ResultsWorkspace()
    qtbot.addWidget(page)
    page.show_results(two_forms, None)
    first, second = page.tree.topLevelItem(0), page.tree.topLevelItem(1)
    page.tree.setCurrentItem(first)
    page.figures.setCurrentWidget(page.table_view)
    assert {r["peptide"] for r in page.table_view.model.rows} == {"RDKVQKEYAL"}
    at = page.figures.indexOf(page.table_view)
    named = page.figures.tabText(at)
    assert named.startswith("data (")
    assert "as hxex report writes them" in page.table_view.note.text()
    page.figures.setCurrentWidget(page.panel("uptake"))
    for item in (second, first, second):  # the tuples come and go while the tab is hidden
        page.tree.setCurrentItem(item)
    page.figures.setCurrentWidget(page.table_view)
    assert {r["peptide"] for r in page.table_view.model.rows} == {"RDKVQKEYAV"}
    page.time_axis.setCurrentIndex(2)  # drawn again for the same selection
    assert page.figures.tabText(at) == named
    page.time_axis.setCurrentIndex(0)


def test_a_peptide_rows_data_tab_lists_every_series_under_it(qtbot, page, results):
    """A peptide row shows its first charge's figures, but its data tab lists the rows of
    every series the tree shows under it (owner, 2026-11-15)."""
    from hxex.report.tables import uptake_table

    peptide = page.tree.topLevelItem(0)
    page.tree.setCurrentItem(peptide)
    page.figures.setCurrentWidget(page.table_view)
    view = page.table_view
    assert {int(r["data_id"]) for r in view.model.rows} == {0, 1}
    assert len(view.model.rows) == len(uptake_table(results, data_ids=[0, 1])["data_id"])
    assert "this peptide's 2 series'" in view.note.text()
    page.tree.setCurrentItem(peptide.child(0))  # its first charge: the same figures, one series
    assert {int(r["data_id"]) for r in view.model.rows} == {0}
    assert "this series'" in view.note.text() and "series.csv: n_spectra" in view.note.text()


def test_a_reread_keeps_the_tab_the_filters_find_and_the_lists_row(qtbot, results, tmp_path):
    """The same results read again (a decision, a changed setting, a finished run): the
    figure tab, the tree's filters and Find, and the List's row are kept, as the window keeps
    the place (audit of 2026-11-15, C-14 and open question 5)."""
    import dataclasses

    page = ResultsWorkspace()
    qtbot.addWidget(page)
    path = tmp_path / "hxex.toml"
    page.show_results(results, path)
    qtbot.waitUntil(lambda: bool(page.list.rows), timeout=30000)
    page.search.setText("S")
    series = page.tree.topLevelItem(0).child(0)
    page.tree.setCurrentItem(series)
    page.figures.setCurrentWidget(page.panel("uptake"))
    page.left_tabs.setCurrentWidget(page.list)
    page.list.select_index(3)
    row = page.list.current_row()["spectrum_id"]
    pending = {**page.selection(), **page.view_settings()}
    page.show_results(dataclasses.replace(results), path)  # as the window rereads them
    page.set_time_axis(pending.pop("time_axis", None))
    assert page.select(pending)
    assert page._tab_keys[page.figures.currentIndex()] == "uptake"
    assert page.search.text() == "S"
    qtbot.waitUntil(lambda: bool(page.list.rows), timeout=30000)
    assert page.list.current_row()["spectrum_id"] == row
    # other results: nothing selected, the overview first
    page.show_results(results, tmp_path / "other.toml")
    assert page.tree.currentItem() is None and page.selection() == {}
    assert page._tab_keys[page.figures.currentIndex()] == "overview"


def test_choosing_a_list_row_opens_only_its_own_series(qtbot, page):
    """A row chosen in the List opens the spectrum's own series, not every series of the
    peptide with that charge (audit of 2026-11-15, C-20: on fragments, dozens were opened)."""
    qtbot.waitUntil(lambda: bool(page.list.rows), timeout=30000)
    top = page.tree.topLevelItem(0)
    for j in range(top.childCount()):
        top.child(j).setExpanded(False)
    row = next(r for r in page.list.rows if int(r["data_id"]) == 1)
    page._list_chosen(row)
    item = page.tree.currentItem()
    assert item.data(0, 256) == "spectrum" and item.data(0, 257)[1].data_id == 1
    assert [top.child(j).isExpanded() for j in range(top.childCount())] == [False, True]
    # a selection by spectrum id alone (a session): the same
    for j in range(top.childCount()):
        top.child(j).setExpanded(False)
    page.tree.setCurrentItem(top)
    assert page.select({"sample": "S", "peptide_range": "0001-0010", "charge": 2,
                        "spectrum_id": row["spectrum_id"]})
    assert [top.child(j).isExpanded() for j in range(top.childCount())] == [False, True]


def test_the_arrows_pass_an_empty_populations_tab_by_series(qtbot, results):
    """With the populations tab in view but empty (one population everywhere), stepping by
    series goes to the next series, not through hidden spectra pages (audit of 2026-11-15,
    C-24)."""
    import dataclasses

    from hxex.report.figures import PlotConfig

    page = ResultsWorkspace()
    qtbot.addWidget(page)
    page.show_results(dataclasses.replace(results, figures=PlotConfig(page_rows=1)), None)
    top = page.tree.topLevelItem(0)
    page.tree.setCurrentItem(top.child(0))
    page.figures.setCurrentWidget(page.panel("spectra"))
    assert page.spectra_page.count() > 1
    page.figures.setCurrentWidget(page.panel("populations"))
    if page._populations_paged:
        pytest.skip("these results have a populations page")
    assert page.step_spectrum(1, "series") and page.tree.currentItem() is top.child(1)


def test_a_click_on_a_spectra_page_selects_that_spectrum(qtbot, page):
    """Each panel of a spectra page (and of the fits at each K) carries its spectrum's id as a
    data row, and a click on it selects the spectrum in the tree (PLAN "GUI"; audit of
    2026-11-15, C open question 1)."""
    series = page.tree.topLevelItem(0).child(0)
    page.tree.setCurrentItem(series)
    panel = page.panel("spectra")
    with qtbot.waitSignal(panel.drawn, timeout=30000):
        page.figures.setCurrentWidget(panel)
    qtbot.waitUntil(lambda: panel.view.ready, timeout=30000)
    rows = [m.row for m in panel.view._marks if m.row]
    ids = {r["spectrum_id"] for r in rows}
    assert len(ids) == 8 and "Click a spectrum's panel" in panel.help.toolTip()
    sid = sorted(ids)[3]
    panel.clicked_row.emit({"spectrum_id": sid})  # as a click on its panel
    item = page.tree.currentItem()
    assert item.data(0, 256) == "spectrum" and item.data(0, 257)[3].spectrum_id == sid


def test_a_bundle_opened_alone_says_the_left_out_column_uses_the_defaults(qtbot, results,
                                                                            tmp_path):
    """Results opened without their project file: the bar under the tree says the 'left out'
    column follows the default analysis settings (audit of 2026-11-15, C open question 4)."""
    page = ResultsWorkspace()
    qtbot.addWidget(page)
    page.show_results(results, tmp_path / "bundle")
    assert page.left_bar.isVisibleTo(page) and page.left_update.isHidden()
    assert "default analysis settings" in page.left_note.text()
    page.show_results(results, tmp_path / "hxex.toml")
    assert page.left_bar.isHidden()
    page._left_stale("your latest review decisions")
    assert not page.left_update.isHidden()


def _targets(parent_charges: dict[int, int]) -> dict:
    """Series targets that make series ``data_id`` the fragment c:2 of the fixture's peptide
    read from the precursor at ``parent_charges[data_id]`` (others: its MS1 series)."""
    from test_run import PEPTIDE

    out = {}
    for data_id, pz in parent_charges.items():
        out[data_id] = ({"fragment": "", "ion_type": "", "condition": "", "ms_level": 1,
                         "parent": "", "parent_range": "", "parent_peptide": "",
                         "parent_charge": 0} if pz is None else
                        {"fragment": "c:2", "ion_type": "c", "condition": "ETD", "ms_level": 2,
                         "parent": f"0001-0010 z{pz}", "parent_range": "0001-0010",
                         "parent_peptide": PEPTIDE, "parent_charge": pz})
    return out


def test_fragment_results_are_grouped_by_precursor(monkeypatch):
    """Owner, 2026-11-17: a top row per parent and precursor charge, its MS1 series first,
    then its precursor ions, then its fragments by ion series (N-terminal first), length,
    charge, MS level and condition; one fragment read from two precursors is under each; no
    "from ..."; a residue number finds a fragment by its own residues."""
    from types import SimpleNamespace

    import hxex.gui.results_tree as tree
    import hxex.report.tables as tables
    from hxex.gui.results_workspace import _series_names

    # (data_id, own range, own sequence, charge, target)
    parent = ("2-12", "ALKRINKELSD")
    made = [(0, *parent, 3, None), (1, *parent, 2, None),
            (2, "8-12", "KELSD", 1, ("z", "z:5", "ETD", 2, 3)),
            (3, "2-6", "ALKRI", 1, ("c", "c:5", "ETD_pre", 3, 3)),
            (4, "2-6", "ALKRI", 1, ("c", "c:5", "ETD", 2, 3)),
            (5, "2-3", "AL", 1, ("c", "c:2", "ETD", 2, 3)),
            (6, *parent, 1, ("M", "M:11", "ETD", 2, 3)),
            (7, "2-6", "ALKRI", 2, ("c", "c:5", "ETD", 2, 3)),
            (8, "2-6", "ALKRI", 1, ("c", "c:5", "EThcD", 2, 2)),
            (9, "20-25", "PEPTID", 2, None)]
    rows = {"data_id": [m[0] for m in made], "sample": ["S"] * len(made),
            "peptide_range": [m[1] for m in made], "peptide": [m[2] for m in made],
            "charge": [m[3] for m in made], **{n: [1] * len(made) for n in (
                "n_spectra", "n_flagged", "n_rejected", "n_multi")}}
    targets = {}
    for d, _r, _p, _z, t in made:
        targets[d] = ({"fragment": "", "ion_type": "", "condition": "", "ms_level": 1,
                       "parent": "", "parent_range": "", "parent_peptide": "",
                       "parent_charge": 0} if t is None else
                      {"fragment": t[1], "ion_type": t[0], "condition": t[2], "ms_level": t[3],
                       "parent": f"2-12 z{t[4]}", "parent_range": "2-12",
                       "parent_peptide": parent[1], "parent_charge": t[4]})
    monkeypatch.setattr(tables, "series_table", lambda results: rows)
    monkeypatch.setattr(tables, "series_identities", lambda results: {})
    monkeypatch.setattr(tree, "series_targets", lambda results: targets)
    results = SimpleNamespace(review_status=np.array([]),
                              fit=SimpleNamespace(data_id=np.array([], int)))
    peps = tree.peptides(results)
    assert [p.label for p in peps] == ["2-12 ALKRINKELSD 2+", "2-12 ALKRINKELSD 3+",
                                       "20-25 PEPTID 2+"]
    two, three, alone = peps
    assert [s.data_id for s in two.series] == [1, 8] and [s.data_id for s in alone.series] == [9]
    assert [s.data_id for s in three.series] == [0, 6, 5, 4, 3, 7, 2]
    assert _series_names(three.series) == [
        "MS1", "M  z=1  ETD, MS2", "c:2  z=1  ETD, MS2", "c:5  z=1  ETD, MS2",
        "c:5  z=1  ETD_pre, MS3", "c:5  z=2  ETD, MS2", "z:5  z=1  ETD, MS2"]
    # a residue finds a fragment by its own residues; the parent's series cover the parent's
    assert [s.data_id for s in three.series if tree.shown(three, s, text="5")] == [0, 6, 4, 3, 7]
    assert [s.data_id for s in three.series if tree.shown(three, s, text="2-6")] == [4, 3, 7]
    # results without fragments: a peptide row with its charges, as before
    monkeypatch.setattr(tree, "series_targets", lambda results: {})
    plain = tree.peptides(results)
    assert [p.label for p in plain] == ["2-3 AL", "2-6 ALKRI", "2-12 ALKRINKELSD", "8-12 KELSD",
                                        "20-25 PEPTID"]
    assert [s.name for s in plain[2].series] == ["z=1", "z=2", "z=3"]


def test_the_window_shows_and_finds_precursor_rows(qtbot, results, monkeypatch):
    """The window's tree of fragment results: the precursor rows, the parent column hidden
    (the row names it), a selection kept by the precursor's charge, a series found by its
    data_id wherever the tree holds it."""
    import hxex.gui.results_tree as tree

    monkeypatch.setattr(tree, "series_targets", lambda r: _targets({0: None, 1: 3}))
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, None)
    labels = [w.tree.topLevelItem(i).text(0) for i in range(w.tree.topLevelItemCount())]
    assert [lbl.split()[-1] for lbl in labels] == ["2+", "3+"]
    assert w.tree.isColumnHidden(COLUMN["parent"]) and not w.tree.isColumnHidden(COLUMN["ion"])
    fragment = w.tree.topLevelItem(1).child(0)
    assert fragment.text(0) == "c:2  z=2  ETD, MS2"
    w.tree.setCurrentItem(fragment)
    kept = w.selection()
    assert kept["parent_charge"] == 3 and kept["fragment"] == "c:2"
    w.tree.setCurrentItem(w.tree.topLevelItem(0))
    assert w.selection()["parent_charge"] == 2
    assert w.select(kept) and w.tree.currentItem() is fragment
    # the list names a fragment by its own residues: found by its data_id
    sid = str(results.fit.spectrum_id[np.flatnonzero(results.fit.data_id == 1)[0]])
    assert w.select({"sample": "S", "peptide_range": "0001-0003", "charge": 2, "data_id": 1,
                     "spectrum_id": sid})
    assert w.tree.currentItem().data(0, KIND) == "spectrum"
    assert w.tree.currentItem().parent().parent() is fragment
    # a session's selection from before the grouping (no parent_charge) finds a row of the range
    assert w.select({"sample": "S", "peptide_range": "0001-0010", "peptide": labels[0].split()[1]})
