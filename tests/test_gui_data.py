# SPDX-License-Identifier: GPL-3.0-or-later
"""The data tab beside a figure (hxex.gui.data_table): the rows a figure's bars carry, sorted,
copied, saved as CSV, and a peptide's row (or bar) leading to its spectra. Runs offscreen."""
import csv
import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hxex.gui.data_table import DataPanel, cell_text  # noqa: E402
from hxex.gui.figures import figure_rows  # noqa: E402
from hxex.gui.tasks import drawing_pool  # noqa: E402
from hxex.report.hover import set_row  # noqa: E402
from test_analysis import two_sample_bundle  # noqa: E402, F401 - the fixture
from test_gui import _window  # noqa: E402

ROWS = [{"sample": "S", "start": 10, "end": 15, "fD": 0.5, "significant": True},
        {"sample": "S", "start": 2, "end": 9, "fD": math.nan, "significant": False},
        {"sample": "S", "start": 4, "end": 8, "fD": 0.25, "significant": False, "note": "x"}]


def test_a_figure_gives_the_rows_its_visible_bars_carry():
    fig = Figure()
    ax = fig.add_subplot()
    bars = ax.bar([1, 2, 3], [1, 2, 3])
    for bar, row in zip(bars, ROWS, strict=True):
        set_row(bar, row)
    bars[1].set_visible(False)
    columns, rows = figure_rows(fig)
    assert columns == ["sample", "start", "end", "fD", "significant", "note"]
    assert rows == [ROWS[0], ROWS[2]]
    assert figure_rows(None) == ([], [])
    assert [cell_text(v) for v in (0.123456, 3, True, False, math.nan, None, "a")] == [
        "0.1235", "3", "yes", "", "", "", "a"]


def test_the_table_sorts_copies_and_saves_what_it_shows(qtbot, tmp_path):
    panel = DataPanel()
    qtbot.addWidget(panel)
    assert not panel.save_button.isEnabled()
    panel.show_rows(["sample", "start", "end", "fD", "significant", "note"], ROWS, "coverage_S")
    assert panel.count.text() == "3 row(s)" and panel.save_button.isEnabled()
    panel.model.sort(3, Qt.SortOrder.DescendingOrder)
    assert [r["start"] for r in panel.model.rows] == [10, 4, 2]  # NaN last either way
    panel.model.sort(3, Qt.SortOrder.AscendingOrder)
    assert [r["start"] for r in panel.model.rows] == [4, 10, 2]
    assert panel.model.data(panel.model.index(0, 3)) == "0.25"
    # copied: the rows selected, with the header, tab-separated
    panel.table.selectRow(1)
    panel.copy()
    assert QApplication.clipboard().text().splitlines() == [
        "sample\tstart\tend\tfD\tsignificant\tnote", "S\t10\t15\t0.5\tyes\t"]
    # saved: every row, in the order shown, as hxex's tables are written
    path = panel.save(tmp_path / "t.csv")
    with path.open(encoding="utf-8") as fh:
        saved = list(csv.reader(fh))
    assert saved[0] == ["sample", "start", "end", "fD", "significant", "note"]
    assert [r[1] for r in saved[1:]] == ["4", "10", "2"]
    assert saved[3][3] == "" and saved[1][5] == "x" and saved[2][5] == ""
    # a row double-clicked asks for its spectra
    with qtbot.waitSignal(panel.activated) as asked:
        panel._double_clicked(panel.model.index(1, 0))
    assert asked.args[0]["start"] == 10
    assert panel.select_row(ROWS[1]) and not panel.select_row({"start": 99})


@pytest.fixture
def project(tmp_path, two_sample_bundle):  # noqa: F811
    path = tmp_path / "hxex.toml"
    path.write_text(f'project = "gui"\n\n[output]\nbundle = "{two_sample_bundle.as_posix()}"\n\n'
                    '[analysis]\nreference = "S"\nresidue_level = true\n', encoding="utf-8")
    return path


def test_a_coverage_figures_rows_lead_to_the_spectra(qtbot, tmp_path, project):
    """The data tab holds the coverage map's bars; a row double-clicked, or a peptide chosen
    from a right click's menu, shows that peptide in Results (owner: from the coverage maps to
    the spectra); a bar clicked selects its row."""
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    window.tabs.setCurrentWidget(page)
    window.show()
    page.compute()
    qtbot.waitUntil(lambda: page.outputs is not None and not page.computing, timeout=120000)
    kinds = {page.list.topLevelItem(i).data(0, 256): page.list.topLevelItem(i)
             for i in range(page.list.topLevelItemCount())}
    coverage = kinds["coverage"]
    assert coverage.childCount() >= 1
    # the computed analysis may already show this figure (the first one in the list), and
    # then choosing it draws nothing new: wait for the figure and its rows, not for a signal
    # (macOS CI, 2026-11-12: the first drawing was done before the wait began)
    stem = coverage.child(0).data(0, 257)
    page.list.setCurrentItem(coverage.child(0))
    qtbot.waitUntil(lambda: page.panel.save_name == stem and not page.panel.drawing
                    and page.panel.view.figure is not None and bool(page.data.model.rows),
                    timeout=60000)
    rows = page.data.model.rows
    assert rows and {"sample", "start", "end", "fD"} <= set(page.data.model.columns)
    assert page.views.tabText(1) == f"data ({len(rows)})"
    assert page.data.save_name == page.panel.save_name
    from hxex.gui.figure_view import SPECTRA_HINT

    assert SPECTRA_HINT in page.panel.help_text and "Right-click a peptide" in SPECTRA_HINT
    row = rows[0]
    assert page.go_to_spectra(row)
    assert window.tabs.currentWidget() is window.results_page
    chosen = window.results_page.selection()
    assert chosen["sample"] == row["sample"]
    assert window.results_page.tree.currentItem().data(0, 257).peptide == row["peptide"]
    # a bar clicked selects its row, staying here
    window.tabs.setCurrentWidget(page)
    page.panel.clicked_row.emit(rows[-1])
    assert window.tabs.currentWidget() is page
    picked = page.data.table.selectionModel().selectedRows()
    assert [page.data.model.rows[i.row()] for i in picked] == [rows[-1]]
    # a right click's menu: the peptide's spectra
    menu = page.rows_menu([rows[-1]])
    (action,) = menu.actions()
    assert action.text() == (f"Spectra of {rows[-1]['peptide']} ({rows[-1]['start']}–"
                             f"{rows[-1]['end']}), {rows[-1]['sample']}")
    action.trigger()
    assert window.tabs.currentWidget() is window.results_page
    assert window.results_page.tree.currentItem().data(0, 257).peptide == rows[-1]["peptide"]
    assert not page.go_to_spectra({"sample": "nowhere", "start": 1, "end": 2})
    assert "is not in the results tree" in page.status.text()


def test_a_right_click_lists_every_peptide_under_the_pointer(qtbot):
    """Overlapping bars (a crowded Woods plot) are all offered, top to bottom, each in its
    sample and, on a comparison, in the reference too; the data tab's rows have the same
    menu."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QContextMenuEvent

    from hxex.gui.data_table import MENU_LIMIT, spectra_choices, spectra_menu
    from hxex.gui.figure_view import FigurePanel

    fig = Figure(figsize=(4, 3), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 20)
    ax.set_ylim(-1, 1)
    low = {"sample": "M", "reference": "WT", "start": 2, "end": 12, "peptide": "AAA"}
    high = {"sample": "M", "reference": "WT", "start": 4, "end": 9, "peptide": "BBB"}
    for row, y in ((low, -0.005), (high, 0.005)):  # within the click allowance of y = 0
        (line,) = ax.plot([row["start"] - 0.5, row["end"] + 0.5], [y, y], lw=8,
                          gid=f"peptide:{row['start']}-{row['end']}")
        set_row(line, row)
    panel = FigurePanel()
    qtbot.addWidget(panel)
    panel.resize(600, 500)
    with qtbot.waitSignal(panel.drawn, timeout=10000):
        panel.draw(lambda: fig)
    view = panel.view
    qtbot.waitUntil(lambda: view.ready and bool(view._marks), timeout=10000)
    at = QPointF(view.width() * 0.3, view.height() * 0.5)
    assert view.rows_at(at) == [high, low]  # top to bottom, as drawn
    assert view.rows_at(QPointF(view.width() * 0.95, view.height() * 0.1)) == []
    with qtbot.waitSignal(panel.rows_asked) as asked:
        QApplication.sendEvent(view, QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse, at.toPoint(), view.mapToGlobal(at.toPoint())))
    assert asked.args[0] == [high, low]
    assert [text for text, _ in spectra_choices([high, low])] == [
        "Spectra of BBB (4–9), M", "Spectra of BBB (4–9), WT",
        "Spectra of AAA (2–12), M", "Spectra of AAA (2–12), WT"]
    gone = []
    menu = spectra_menu([high, low], gone.append)
    menu.actions()[1].trigger()
    assert gone == [{**high, "sample": "WT"}]
    many = [{**low, "start": n} for n in range(MENU_LIMIT + 3)]
    long_menu = spectra_menu(many, gone.append)  # kept: its actions go with it
    more = long_menu.actions()
    assert more[-1].text() == "and 3 more: see the data tab" and not more[-1].isEnabled()
    assert spectra_menu([{"residue": 3}], gone.append) is None  # no peptide: no menu
    data = DataPanel()
    qtbot.addWidget(data)
    data.show_rows(list(low), [low, high])
    with qtbot.waitSignal(data.activated) as chosen:
        data.row_menu(data.model.index(1, 0)).actions()[0].trigger()
    assert chosen.args == [high]


def test_a_kind_selected_gathers_every_figures_rows(qtbot, tmp_path, project):
    """A kind of figure selected (owner: the first time point's rows alone were confusing):
    the data tab has the rows of all its figures, each named by a figure column; one figure
    selected: its own rows."""
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    window.tabs.setCurrentWidget(page)
    window.show()
    page.compute()
    qtbot.waitUntil(lambda: page.outputs is not None and not page.computing, timeout=120000)
    kinds = {page.list.topLevelItem(i).data(0, 256): page.list.topLevelItem(i)
             for i in range(page.list.topLevelItemCount())}
    coverage = kinds["coverage"]
    stems = [coverage.child(i).data(0, 257) for i in range(coverage.childCount())]
    assert len(stems) == 2
    page.list.setCurrentItem(coverage)
    # gathered only when the data tab is shown (it draws every figure of the kind: minutes
    # for a run with hundreds; owner, 2026-11-04, ladders "stuck on drawing")
    assert page.data.note.text().endswith("gathered when this tab is shown.")
    page.views.setCurrentWidget(page.data)
    qtbot.waitUntil(lambda: page.data.model.columns[:1] == ["figure"], timeout=60000)
    gathered = page.data.model.rows
    assert {r["figure"] for r in gathered} == set(stems)
    assert page.data.note.text().startswith("Every figure of this kind (2 with rows)")
    with qtbot.waitSignal(page.panel.drawn, timeout=60000):
        page.list.setCurrentItem(coverage.child(1))
    one = page.data.model.rows
    assert "figure" not in page.data.model.columns and page.data.note.isHidden()
    assert len(one) == sum(r["figure"] == stems[1] for r in gathered)
    # a new selection stops a gathering under way: its result is never shown
    page.list.setCurrentItem(coverage)
    with qtbot.waitSignal(page.panel.drawn, timeout=60000):
        page.list.setCurrentItem(coverage.child(0))
    # both jobs done in the drawing thread, their results delivered, before the check
    assert drawing_pool().waitForDone(60000)
    qtbot.wait(50)  # their results delivered
    page._rows_gathered("coverage", None)  # a stopped gathering returns nothing
    assert "figure" not in page.data.model.columns


def test_the_data_tabs_menu_copies(qtbot):
    """The data tab's right-click menu has Copy beside a peptide's spectra."""
    from PySide6.QtWidgets import QApplication

    data = DataPanel()
    qtbot.addWidget(data)
    data.show_rows(["sample", "start", "end", "fD"], ROWS, "coverage_S")
    texts = [a.text() for a in data.row_menu(data.model.index(0, 0)).actions()]
    assert texts[0].startswith("Spectra of") and texts[-1] == "Copy"
    data.table.selectRow(0)
    data.row_menu(data.model.index(0, 0)).actions()[-1].trigger()
    assert QApplication.clipboard().text().splitlines()[1].startswith("S\t10")


def test_the_hdx_summary_table_tab_shows_the_analysis_summary(qtbot, tmp_path, project):
    """The tab beside figure and data: empty until Compute, then the summary hxex analyse
    writes (owner, 2026-11-13), marked out of date when [analysis.summary] changes, and saved
    as CSV or Markdown."""
    from hxex.gui.summary_panel import EMPTY
    from hxex.report.summary import MASSON_ROWS, SOFTWARE, base_item, split_notes

    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    assert page.views.tabText(2) == "HDX Summary Table"
    assert page.summary.count.text() == EMPTY and not page.summary.model.rows
    page.compute()
    qtbot.waitUntil(lambda: page.outputs is not None and not page.computing, timeout=120000)
    summary = page.summary
    assert summary.model.columns == ["item", "S", "V"]
    items = [base_item(r["item"]) for r in summary.model.rows]
    assert [i for i in items if i in MASSON_ROWS] == list(MASSON_ROWS)
    table, notes = split_notes(page.outputs.tables["summary"])
    assert summary.model.rows == [  # the table summary.csv holds, the notes below it
        {n: table[n][i] for n in table} for i in range(len(items))]
    assert summary.notes.text().splitlines()[0] == f"1. {notes[0]}"
    # a value the same for every sample: once, across the samples' columns
    software = items.index(SOFTWARE)
    assert summary.table.columnSpan(software, 1) == 2
    differ = next(i for i, r in enumerate(summary.model.rows) if r["S"] != r["V"])
    assert summary.table.columnSpan(differ, 1) == 1
    # an item's note is its tooltip
    first = next(i for i in summary.model.notes)
    tooltip = summary.model.data(summary.model.index(first, 0), Qt.ItemDataRole.ToolTipRole)
    assert notes[0].split()[0] in tooltip
    assert summary.note.isHidden()
    with open(project, "a", encoding="utf-8") as f:
        f.write('\n[analysis.summary]\nProtease = "pepsin"\n')
    window.analysis_page.refresh()
    qtbot.waitUntil(lambda: summary.note.text().startswith("Out of date"), timeout=10000)
    md = summary.save(tmp_path / "s.md")
    assert md.read_text(encoding="utf-8").startswith("# HDX data summary")
    with open(summary.save(tmp_path / "s.csv"), newline="", encoding="utf-8") as f:
        saved = [r["item"] for r in csv.DictReader(f)]
    assert saved == [str(x) for x in page.outputs.tables["summary"]["item"]]  # notes too


def test_cells_and_headers_naming_a_flag_say_what_it_means(qtbot):
    """A cell holding flags, control notes, or a left-out reason, and a column named after a
    flag or a screen, show their meanings on hover (hxex.flags); other cells their text."""
    from types import SimpleNamespace

    import numpy as np

    from hxex.gui.data_table import RowsModel
    from hxex.gui.review_panel import flag_items
    from hxex.quality import FLAG_NAMES

    model = RowsModel()
    model.set_rows(["flags", "reason", "screen_between", "sample"],
                   [{"flags": "off_mz;poor_fit", "reason": "stand-in control (screened)",
                     "screen_between": 0.5, "sample": "S"}])
    tip = Qt.ItemDataRole.ToolTipRole
    cell = model.data(model.index(0, 0), tip)
    assert "<b>off_mz</b>" in cell and "<b>poor_fit</b>" in cell and "hxex explain" not in cell
    assert "only a stand-in was left" in model.data(model.index(0, 1), tip)
    assert model.data(model.index(0, 3), tip) == "S"
    header = model.headerData(2, Qt.Orientation.Horizontal, tip)
    assert "<b>screen_between</b>" in header
    assert model.headerData(3, Qt.Orientation.Horizontal, tip) is None
    q = {name: np.zeros(3, dtype=bool) for name in FLAG_NAMES}
    q["poor_fit"][[0, 2]] = True
    results = SimpleNamespace(quality=SimpleNamespace(**q))
    # the review panel's Flags button: the flags with their meanings, counted over several
    # spectra, on hover; greyed without any
    from hxex.gui.review_panel import FlagsButton

    assert flag_items(results, [0]) == [
        ("poor_fit", 1, "The fit is much worse than the other fits of its series.")]
    assert flag_items(results, [0, 1, 2])[0][:2] == ("poor_fit", 2)
    assert flag_items(results, [1]) == [] and flag_items(results, []) == []
    button = FlagsButton()
    qtbot.addWidget(button)
    width = button.width()
    button.show_flags(flag_items(results, [0, 1, 2]), 3)
    assert button.text() == "Flags (1)" and button.isEnabled() and button.width() == width
    assert "<b>poor_fit</b> (2 of 3)" in button.toolTip()
    assert "hxex explain" not in button.toolTip()
    button.show_flags([], 1)
    assert button.text() == "Flags" and not button.isEnabled()
