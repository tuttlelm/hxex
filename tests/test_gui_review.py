# SPDX-License-Identifier: GPL-3.0-or-later
"""Review decisions from hxex's window written to the user review table (hxex.gui.review): the
table made from the results and named in the project, decisions in the spectrum's row, the
user's own columns kept, and every command reading them as typed. No Qt."""
import tomllib

import pytest

from hxex.cli import main
from hxex.gui.review import Decision, make_review_table, table_path, write_decisions
from hxex.gui.user_tables import read_table
from hxex.report import Results
from hxex.review_table import read_review_table
from test_cli import mzml_project


@pytest.fixture(scope="module")
def project_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("review")
    project = mzml_project(folder)
    assert main(["-q", "run", str(project)]) == 0
    from hxex.report.notebook import load_results

    return project, load_results(project)


def test_the_first_decision_makes_the_table_and_names_it(project_run):
    project, results = project_run
    assert table_path(project) is None
    path = make_review_table(project, results)
    assert path == project.parent / "user_review_table.csv"
    assert tomllib.loads(project.read_text())["review"]["table"] == "user_review_table.csv"
    table = read_table(path)
    assert {"spectrum_id", "review_status", "review_K", "note"} <= set(table.columns)
    assert len(table.rows) == len(results.fit)  # a copy of spectra.csv
    assert make_review_table(project, results) == path  # named already: kept


def test_decisions_go_in_the_spectrums_row_as_a_user_would_type_them(project_run):
    project, results = project_run
    path = make_review_table(project, results)
    table = read_table(path)
    table.columns.append("my notes")
    for row in table.rows:
        row.append("mine")
    from hxex.gui.user_tables import write_table

    write_table(table, path)
    ids = [str(s) for s in results.fit.spectrum_id]
    write_decisions(path, {ids[0]: Decision(status="rejected", note="a second species"),
                           ids[1]: Decision(status="accepted", k=1, k_given=True),
                           "not-in-the-table": Decision(status="accepted")})
    sheet = read_review_table(path)
    got = dict(zip(sheet.spectrum_id, zip(sheet.status, sheet.k, sheet.note, strict=True),
                   strict=True))
    assert got[ids[0]] == ("rejected", 0, "a second species")
    assert got[ids[1]] == ("accepted", 1, "")
    assert got["not-in-the-table"][0] == "accepted"  # a row added
    assert all(row[-1] in ("mine", "") for row in read_table(path).rows)  # the user's column
    # back to the fit's choice: the table's own K emptied (status cleared too) ...
    columns = read_table(path).columns
    write_decisions(path, {ids[1]: Decision(status="", k=None, k_given=True)},
                    bundle_k={ids[1]: 1})
    row = next(r for r in read_table(path).rows if r[0] == ids[1])
    assert row[columns.index("review_status")] == "" and row[columns.index("review_K")] == ""
    # ... and a K the bundle itself chose (none in the table) cleared with review_K = 0
    write_decisions(path, {ids[1]: Decision(k=None, k_given=True)}, bundle_k={ids[1]: 2})
    row = next(r for r in read_table(path).rows if r[0] == ids[1])
    assert row[columns.index("review_K")] == "0"
    with pytest.raises(ValueError, match="review status"):
        write_decisions(path, {ids[0]: Decision(status="maybe")})
    # the row added for a spectrum the results do not have goes again (hxex check names it)
    table = read_table(path)
    table.rows = [r for r in table.rows if r[0] != "not-in-the-table"]
    write_table(table, path)


def test_the_commands_read_the_windows_decisions(project_run, capsys):
    project, results = project_run
    path = make_review_table(project, results)
    sid = str(results.fit.spectrum_id[2])
    write_decisions(path, {sid: Decision(status="rejected")})
    assert main(["check", str(project)]) == 0
    assert "rejected" in capsys.readouterr().out
    from hxex.report.notebook import load_results

    reviewed = load_results(project)
    assert isinstance(reviewed, Results)
    assert reviewed.review_status[2] == "rejected"


# --- the review panel in the window (offscreen) --------------------------------------------------

@pytest.fixture
def window_on(qtbot, tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox

    from hxex.gui.window import MainWindow

    project = mzml_project(tmp_path)
    assert main(["-q", "run", str(project)]) == 0
    window = MainWindow(settings=QSettings(str(tmp_path / "g.ini"), QSettings.Format.IniFormat))
    window.ask_to_save = lambda: QMessageBox.StandardButton.Discard
    qtbot.addWidget(window)

    def open_(path):
        window.open_asking(path)
        qtbot.waitUntil(lambda: not window.loading and window.results_path is not None,
                        timeout=30000)
        return window.results_page

    return window, project, open_


def _col(name):
    """The tree's column ``name`` (Qt is imported only once the window's tests run)."""
    from hxex.gui.results_workspace import COLUMN

    return COLUMN[name]


def _first_spectrum(page):
    series = page.tree.topLevelItem(0).child(0)
    page.tree.expandItem(series)
    return series, series.child(0).child(0)


def test_a_key_in_the_tree_rejects_and_the_table_and_tree_show_it(qtbot, window_on):
    from PySide6.QtCore import Qt

    window, project, open_ = window_on
    page = open_(project)
    series, leaf = _first_spectrum(page)
    page.tree.setCurrentItem(leaf)
    panel = page.review
    assert panel.status.buttons()[0].isEnabled() and panel.status.checkedId() == 0
    qtbot.keyClick(page.tree, Qt.Key.Key_R)
    sid = leaf.toolTip(0)
    sheet = read_review_table(project.parent / "user_review_table.csv")
    assert dict(zip(sheet.spectrum_id, sheet.status, strict=True))[sid] == "rejected"
    assert leaf.text(_col("review")) == "rejected" and series.text(_col("rejected")) == "1"
    assert leaf.text(_col("rejected")) == "yes" and leaf.parent().text(_col("rejected")) == "1"
    assert panel.status.checkedId() == 2 and "Saved" in panel.message.text()
    # the note, then accepted with the A key
    panel.note.setText("checked by eye")
    panel.note.editingFinished.emit()
    qtbot.keyClick(page.tree, Qt.Key.Key_A)
    sheet = read_review_table(project.parent / "user_review_table.csv")
    got = dict(zip(sheet.spectrum_id, zip(sheet.status, sheet.note, strict=True), strict=True))
    assert got[sid] == ("accepted", "checked by eye") and series.text(_col("rejected")) == ""


def test_a_decision_on_a_series_reaches_every_spectrum(qtbot, window_on):
    window, project, open_ = window_on
    page = open_(project)
    series, _ = _first_spectrum(page)
    page.tree.setCurrentItem(series)
    assert "a decision applies to each" in page.review.heading.text()
    page.review.decide(status="rejected")
    n = int(series.text(2))
    assert series.text(_col("rejected")) == str(n)
    sheet = read_review_table(project.parent / "user_review_table.csv")
    assert sum(s == "rejected" for s in sheet.status) == n


def test_a_held_table_keeps_the_decision_until_it_can_be_written(qtbot, window_on,
                                                                 monkeypatch):
    import hxex.gui.review as review_module
    from hxex.gui.user_tables import TableLocked

    window, project, open_ = window_on
    page = open_(project)
    _, leaf = _first_spectrum(page)
    page.tree.setCurrentItem(leaf)
    real = review_module.write_table

    def held(table, path):
        raise TableLocked(f"{path} could not be written: another program holds it")

    monkeypatch.setattr(review_module, "write_table", held)
    assert not page.review.decide(status="rejected")
    assert page.review.retry.isVisibleTo(page.review) and page.review.pending
    monkeypatch.setattr(review_module, "write_table", real)
    page.review.retry.click()
    assert not page.review.pending and leaf.text(_col("review")) == "rejected"


def test_a_bundle_opened_alone_cannot_be_reviewed(qtbot, window_on):
    window, project, open_ = window_on
    page = open_(project.parent / "out")
    _, leaf = _first_spectrum(page)
    page.tree.setCurrentItem(leaf)
    assert not page.review.status.buttons()[0].isEnabled()
    assert "open the results through the project file" in page.review.heading.text()


def test_back_and_forward_step_through_every_spectrum(qtbot, window_on):
    """◀ and ▶ go through the spectra in the tree's order, into the next time point and series
    (owner, 2026-11-03)."""
    window, project, open_ = window_on
    page = open_(project)
    top = page.tree.topLevelItem(0)
    page.tree.setCurrentItem(top)  # a peptide: ▶ goes to its first spectrum
    seen = []
    while True:
        page.review.next.click()
        current = page.tree.currentItem()
        assert current.data(0, 256) == "spectrum"
        if seen and current.toolTip(0) == seen[-1]:
            break  # the end: nothing after the last
        seen.append(current.toolTip(0))
    n = sum(int(top.child(i).text(2)) for i in range(top.childCount()))
    assert len(seen) == n == len(set(seen))  # every spectrum once, across series and points
    page.review.previous.click()
    assert page.tree.currentItem().toolTip(0) == seen[-2]
    assert not page.step_spectrum(1) or page.tree.currentItem().toolTip(0) == seen[-1]


def test_each_status_names_its_own_key(window_on):
    """The tooltips and the keys agree (owner: the tooltips had A and R swapped)."""
    window, project, open_ = window_on
    page = open_(project)
    tips = {b.text(): b.toolTip() for b in page.review.status.buttons()}
    assert tips == {"not reviewed": "not reviewed (key U in the tree)",
                    "accepted": "accepted (key A in the tree)",
                    "rejected": "rejected (key R in the tree)"}


def test_back_and_forward_step_by_the_level_chosen(qtbot, window_on):
    """The level list beside ◀ ▶: spectrum, time point, charge, peptide (owner)."""
    window, project, open_ = window_on
    page = open_(project)
    top = page.tree.topLevelItem(0)
    series = top.child(0)
    page.tree.setCurrentItem(series)
    level = page.review.level
    level.setCurrentIndex(level.findData("point"))
    assert page.review.next.toolTip().startswith("the next time point")
    page.review.next.click()  # from the series: its first time point
    first = page.tree.currentItem()
    assert first.data(0, 256) == "point" and first.parent() is series
    page.review.next.click()
    assert page.tree.currentItem() is series.child(1)
    page.tree.setCurrentItem(series.child(1).child(0))  # a spectrum: the point after its own
    page.review.next.click()
    assert page.tree.currentItem() is series.child(2)
    level.setCurrentIndex(level.findData("series"))
    # from a time point: the series after its own; this project has one charge of one peptide
    assert top.childCount() == 1 and not page.step_spectrum(1)
    assert page.tree.currentItem() is series.child(2)  # nothing after it: no move
    assert page.step_spectrum(-1, "point") and page.tree.currentItem() is series.child(1)
    assert [level.itemData(i) for i in range(level.count())] == ["spectrum", "point", "series"]
