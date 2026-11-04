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
    assert tabs == ["spectra", "fits at each K", "uptake", "table", "overview"]
    qtbot.waitUntil(lambda: page.panel("fits at each K").view.figure is not None, timeout=30000)
    title = page.panel("fits at each K").view.figure.axes[0].get_title()
    assert title.splitlines()[-1].startswith("rep 1: K=")
    assert len(page.panel("spectra").view.figure.axes) == 3  # that time's replicates
    with qtbot.waitSignal(page.panel("fits at each K").drawn, timeout=30000):
        page.tree.setCurrentItem(point.child(0))
    assert [page.figures.tabText(i) for i in range(page.figures.count())] == [
        "spectrum", "fits at each K", "uptake", "table", "overview"]
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
    assert "Mouse over" not in overview.hint.text()  # no tooltips on the Results figures
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
        "spectra", "uptake", "populations", "table", "overview"]
    assert page.figures.currentWidget() is overview and overview.view.figure is first
    # new results (as a review decision makes): drawn again
    page.results = dataclasses.replace(results)
    with qtbot.waitSignal(overview.drawn, timeout=30000):
        page._redraw()
    assert overview.view.figure is not first


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
    with qtbot.waitSignal(page.panel("spectrum").drawn, timeout=30000):
        page.tree.setCurrentItem(fd)
    assert [page.figures.tabText(i) for i in range(page.figures.count())] == [
        "spectrum", "fits at each K", "uptake", "table", "overview"]
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


def test_the_table_tab_holds_the_selections_report_rows(qtbot, page, results):
    """Beside the figures, the selection's rows of the report's tables (owner): a series'
    uptake.csv rows with its series.csv values said above them, a time point's spectra.csv
    rows, a spectrum's row as names and values; a row double-clicked opens its spectrum."""
    from hxex.report.tables import uptake_table

    series = page.tree.topLevelItem(0).child(0)
    page.tree.setCurrentItem(series)
    tabs = [page.figures.tabText(i) for i in range(page.figures.count())]
    assert tabs[-2:] == ["table", "overview"]
    page.figures.setCurrentWidget(page.table_view)
    view = page.table_view
    want = uptake_table(results, data_ids=[0])
    assert view.model.columns == list(want) and len(view.model.rows) == len(want["data_id"])
    assert view.note.text().startswith("uptake.csv: this series' rows. series.csv: n_spectra")
    page.tree.expandItem(series)
    point = series.child(1)
    page.tree.setCurrentItem(point)
    assert page.figures.currentWidget() is view  # the tab stays in view
    assert {r["spectrum_id"] for r in view.model.rows} == {s.spectrum_id for s in
                                                            point.data(0, 257)[2].spectra}
    first = view.model.rows[0]["spectrum_id"]
    view._double_clicked(view.model.index(0, 0))  # a row: its spectrum
    assert page.tree.currentItem().toolTip(0) == first
    leaf = point.child(0)
    page.tree.setCurrentItem(leaf)
    assert view.model.columns == ["column", "value"]
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
