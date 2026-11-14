# SPDX-License-Identifier: GPL-3.0-or-later
"""Review decisions from hxex's window written to the user review table (hxex.gui.review): the
table made from the results and named in the project, decisions in the spectrum's row, the
user's own columns kept, and every command reading them as typed. The window's tests run
offscreen (set before pytest-qt makes the application, so a run of this file alone opens no
window on the screen)."""
import os
import tomllib

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
    text = path.read_text(encoding="utf-8")  # numbers as spectra.csv writes them
    assert "np." not in text and "float64" not in text
    assert read_table(project.parent / "out_tables" / "spectra.csv").rows == table.rows
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
    # the "left out" column is not worked out again in the middle of a review (owner): said
    assert not page._left_task.running and page.left_bar.isVisibleTo(page)
    assert "before your latest review decisions" in page.left_note.text()
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
    n = int(series.text(_col("spectra")))
    assert series.text(_col("rejected")) == str(n)
    sheet = read_review_table(project.parent / "user_review_table.csv")
    assert sum(s == "rejected" for s in sheet.status) == n


def _typing_ready(qtbot, window, open_, project):
    """The window active (so focus moves as a person moves it), the review table made and named
    (its first decision reads the results again), the first series' first spectrum accepted
    with a note of its own."""
    page = open_(project)
    window.show()
    window.windowHandle().requestActivate()
    qtbot.waitUntil(window.isActiveWindow, timeout=5000)
    _, leaf = _first_spectrum(page)
    page.tree.setCurrentItem(leaf)
    page.review.decide(status="accepted", note="its own note")
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.results_page
    series, leaf = _first_spectrum(page)
    return page, series, leaf


def _notes(project) -> dict:
    """Every note in the table as written (the reader leaves out rows without a decision)."""
    table = read_table(project.parent / "user_review_table.csv")
    sid, note = table.columns.index("spectrum_id"), table.columns.index("note")
    return {row[sid]: row[note] for row in table.rows if row[note]}


def _type_note(qtbot, panel, text):
    panel.note.setFocus()
    qtbot.waitUntil(panel.note.hasFocus, timeout=5000)
    qtbot.keyClicks(panel.note, text)


def test_a_series_note_is_kept_after_enter_and_leaving_the_box(qtbot, window_on):
    """Owner, 2026-11-04: a series note saved with Enter was wiped, with the spectra's own notes,
    when the box was left (the box emptied for the series read as an edit). Now the note stays
    and the series shows it."""
    from PySide6.QtCore import Qt

    window, project, open_ = window_on
    page, series, leaf = _typing_ready(qtbot, window, open_, project)
    assert set(_notes(project).values()) == {"its own note"}
    page.tree.setCurrentItem(series)
    n = int(series.text(_col("spectra")))
    panel = page.review
    _type_note(qtbot, panel, "series note")
    qtbot.keyClick(panel.note, Qt.Key.Key_Return)
    assert panel.note.text() == "series note"  # every spectrum's note, shown for the series
    rect = page.tree.visualItemRect(series)
    qtbot.mouseClick(page.tree.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    panel.status.buttons()[2].setFocus()  # and on to a status
    panel.decide(status="rejected")
    notes = _notes(project)
    assert len(notes) == n and set(notes.values()) == {"series note"}


def test_notes_that_differ_are_kept_when_the_box_is_left_empty(qtbot, window_on):
    """A series whose spectra have different notes shows none (and says so); leaving the empty
    box, or finishing it, writes nothing, so each keeps its own."""
    from PySide6.QtCore import Qt

    window, project, open_ = window_on
    page, series, leaf = _typing_ready(qtbot, window, open_, project)
    page.tree.setCurrentItem(series)
    panel = page.review
    assert panel.note.text() == "" and "differ" in panel.note.placeholderText()
    panel.note.setFocus()
    qtbot.waitUntil(panel.note.hasFocus, timeout=5000)
    rect = page.tree.visualItemRect(series)
    qtbot.mouseClick(page.tree.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    panel.note.editingFinished.emit()
    assert _notes(project) == {leaf.toolTip(0): "its own note"}


def test_a_note_being_typed_is_kept_when_the_results_are_read_again(qtbot, window_on):
    """The review table saved elsewhere while a note is typed: the note is written to the
    spectra it was typed for before the results are read again, not lost."""
    window, project, open_ = window_on
    page, series, _ = _typing_ready(qtbot, window, open_, project)
    page.tree.setCurrentItem(series)
    n = int(series.text(_col("spectra")))
    _type_note(qtbot, page.review, "typed while saved elsewhere")
    table = project.parent / "user_review_table.csv"
    table.write_text(table.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    qtbot.waitUntil(lambda: window.loading, timeout=10000)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    notes = _notes(project)
    assert len(notes) == n and set(notes.values()) == {"typed while saved elsewhere"}


def test_a_note_on_its_own_is_shown_again_after_a_reload(qtbot, window_on):
    """A note on a spectrum with no status and no K is read back with the table (owner,
    2026-11-05: it had been written but not shown once the results were read again)."""
    from hxex.report.notebook import load_results

    window, project, open_ = window_on
    page, series, leaf = _typing_ready(qtbot, window, open_, project)
    other = leaf.parent().child(1) if leaf.parent().childCount() > 1 else \
        series.child(1).child(0)
    sid = other.toolTip(0)
    page.tree.setCurrentItem(other)
    page.review.note.setText("only a note")
    page.review.note.editingFinished.emit()
    window.open_results(project)  # read again, as reopening the project does
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.results_page
    results = page.results
    i = results.find(sid)
    assert results.review.get(sid)["note"] == "only a note"
    assert results.review_status[i] == "unreviewed"
    assert load_results(project).review.get(sid)["note"] == "only a note"


def test_the_windows_decisions_give_the_same_report_tables_as_a_spreadsheet(
        qtbot, window_on, tmp_path_factory):
    """The G3 exit check (docs/PLAN.md, "Phases of the GUI"): review edits made in the window
    give the same `hxex report` tables as the same edits typed into a copy of spectra.csv (with
    a spreadsheet's BOM and semicolons). Checked by hand on the tutorial data too (HISTORY 285:
    window edits against a table saved by LibreOffice, every table identical)."""
    import csv
    import shutil

    window, project, open_ = window_on
    sheet = tmp_path_factory.mktemp("sheet") / "p"
    shutil.copytree(project.parent, sheet)  # before any decision: no review table yet
    page = open_(project)
    edits = []

    def act(selection, **decision):
        assert window.results_page.select(selection)
        panel = window.results_page.review
        ids = [str(window.results_page.results.fit.spectrum_id[i]) for i in panel.rows]
        if set(decision) == {"note"}:
            panel.note.setText(decision["note"])
            panel.note.editingFinished.emit()
        else:
            panel.decide(**decision)
        qtbot.waitUntil(lambda: not window.loading, timeout=30000)
        edits.append((ids, decision))

    series, leaf = _first_spectrum(page)
    page.tree.setCurrentItem(leaf)
    where = page.selection()
    page.tree.expandItem(series.child(1))
    one, other = leaf.toolTip(0), series.child(1).child(0).toolTip(0)  # two time points
    page.tree.setCurrentItem(series.child(2))
    point = page.selection()  # a time point, by what outlasts reading the results again
    whole = {k: where[k] for k in ("sample", "peptide_range", "charge")}
    act({**where, "spectrum_id": one}, status="rejected")  # makes and names the table
    act(point, status="accepted")
    act(point, note="a time point's note")
    act(whole, k=1, k_given=True)
    act(whole, note="a charge's note")
    act({**where, "spectrum_id": other}, note="a note on its own")
    act({**where, "spectrum_id": one}, status="")  # back to not reviewed
    act({**where, "spectrum_id": one}, k=None, k_given=True)  # back to the fit's choice
    # the same edits typed into a copy of spectra.csv
    with (sheet / "out_tables" / "spectra.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    col = {name: rows[0].index(name) for name in ("spectrum_id", "review_status", "review_K",
                                                   "note")}
    at = {row[col["spectrum_id"]]: row for row in rows[1:]}
    for ids, decision in edits:
        for sid in ids:
            if "status" in decision:
                at[sid][col["review_status"]] = decision["status"]
            if decision.get("k_given"):
                at[sid][col["review_K"]] = "" if decision["k"] is None else str(decision["k"])
            if "note" in decision:
                at[sid][col["note"]] = decision["note"]
    with (sheet / "my_review.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh, delimiter=";").writerows(rows)
    assert main(["-q", "set", str(sheet / "hxex.toml"), "review.table", "my_review.csv"]) == 0
    for folder in (project.parent, sheet):
        assert main(["-q", "report", str(folder / "hxex.toml"), "--overwrite"]) == 0

    def tables(folder):
        out = {}
        for path in sorted((folder / "out_tables").glob("*.csv")):
            with path.open(encoding="utf-8", newline="") as fh:
                got = list(csv.reader(fh))
            if path.name == "review.csv":  # the date of each decision differs
                j = got[0].index("updated")
                got = [row[:j] + row[j + 1:] for row in got]
            out[path.name] = got
        return out

    window_side, sheet_side = tables(project.parent), tables(sheet)
    assert {"spectra.csv", "review.csv", "uptake.csv"} <= set(window_side)
    assert window_side == sheet_side
    spectra = {row[0]: row for row in window_side["spectra.csv"][1:]}
    head = window_side["spectra.csv"][0]
    note, status = head.index("note"), head.index("review_status")
    assert spectra[other][note] == "a note on its own" and spectra[one][status] == "unreviewed"
    assert sum(row[status] == "accepted" for row in spectra.values()) == len(edits[1][0])
    n = len(edits[4][0])  # every spectrum of the charge but the one with its own note
    assert n > 2 and sum(row[note] == "a charge's note" for row in spectra.values()) == n - 1


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
    # said under the Note row, apart from what is selected (owner, 2026-11-13)
    from hxex.gui.review_panel import RESULTS_ONLY

    assert page.review.message.text() == RESULTS_ONLY
    assert "Decisions" not in page.review.heading.text()
    assert "File > Open project file" in page.review.message.toolTip()


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
    n = sum(int(top.child(i).text(_col("spectra"))) for i in range(top.childCount()))
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


def test_a_review_table_saved_elsewhere_is_read_again(qtbot, window_on):
    """The review table edited in the Tables view or a spreadsheet: the results are read again
    with it, at the same place (owner); the review panel's own decisions are not read again."""
    import csv

    window, project, open_ = window_on
    page = open_(project)
    series, leaf = _first_spectrum(page)
    page.tree.setCurrentItem(leaf)
    page.review.decide(status="rejected")  # makes the table and names it: read once
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    table = project.parent / "user_review_table.csv"
    assert table.exists()
    qtbot.waitUntil(lambda: bool(window.review_watcher.files()), timeout=10000)
    loads = []
    real = window.open_results
    window.open_results = lambda path: loads.append(path) or real(path)
    # a decision in the panel: written, but not read again
    series, leaf = _first_spectrum(window.results_page)
    window.results_page.tree.setCurrentItem(leaf.parent().child(1))
    window.results_page.review.decide(status="accepted")
    qtbot.wait(800)
    assert loads == []
    # the same table changed elsewhere: read again, at the same place
    chosen = window.results_page.selection()
    with table.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    head = rows[0]
    target = rows[-1]
    target[head.index("review_status")] = "rejected"
    with table.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)
    qtbot.waitUntil(lambda: bool(loads) and not window.loading, timeout=30000)
    results = window.results_page.results
    i = results.find(target[head.index("spectrum_id")])
    assert results.review_status[i] == "rejected"
    assert window.results_page.selection() == chosen


def test_a_decision_made_from_the_list_updates_its_row(qtbot, window_on):
    """Working through the list (owner): a decision on its row, by key, is written and the row
    shows it at once (made again: status, K and populations); the row stays in view."""
    from PySide6.QtCore import Qt

    window, project, open_ = window_on
    page = open_(project)
    qtbot.waitUntil(lambda: bool(page.list.rows), timeout=30000)
    page.left_tabs.setCurrentWidget(page.list)
    lst = page.list
    lst.filter.setCurrentIndex(lst.filter.findData("unreviewed"))
    sid = str(lst.model.rows[0]["spectrum_id"])
    assert lst.select(sid) and page.tree.currentItem().toolTip(0) == sid
    qtbot.keyClick(lst.grid, Qt.Key.Key_R)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.results_page  # the first decision names the table: read again
    qtbot.waitUntil(lambda: bool(page.list.rows), timeout=30000)
    row = next(r for r in page.list.rows if str(r["spectrum_id"]) == sid)
    assert row["review_status"] == "rejected"
    sheet = read_review_table(project.parent / "user_review_table.csv")
    assert dict(zip(sheet.spectrum_id, sheet.status, strict=True))[sid] == "rejected"
    # a second decision: the row made again in place, still listed under "not reviewed"
    page.left_tabs.setCurrentWidget(page.list)
    other = next(r for r in page.list.model.rows if str(r["spectrum_id"]) != sid)
    assert page.list.select(str(other["spectrum_id"]))
    page.review.decide(status="accepted")
    assert other["review_status"] == "accepted"
    assert any(r is other for r in page.list.model.rows)
