# SPDX-License-Identifier: GPL-3.0-or-later
"""Small helpers shared by the window's tables and boxes (audit of 2026-11-15, Part 2, Q-12 to
Q-41): copying and saving rows, the watched file, the dropdowns' tooltips, the one error colour,
and the list's count and Find tooltip. Runs offscreen."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication, QComboBox, QFileDialog

from hxex.gui import tips
from hxex.gui.data_table import DataPanel
from hxex.gui.spectra_list import SpectraList
from hxex.gui.table_copy import block_text, rows_text, save_rows, saved_or_said
from hxex.gui.watched_file import WatchedFile
from hxex.project_edit import replace_file

ROWS = [{"sample": "S", "start": 2, "fD": 0.25, "ok": True},
        {"sample": "S", "start": 10, "fD": 652.3412, "ok": False}]
SPECTRA = [{"spectrum_id": f"s{i}", "sample": "S", "peptide_range": "0001-0010",
            "peptide": "RDKVQKEYAL", "K": 1, "review_status": "", "_index": i} for i in range(3)]


def test_block_and_rows_text_are_tab_separated_with_the_header():
    assert block_text([["a", "b"], ["1", ""]]) == "a\tb\n1\t\n"
    assert block_text([]) == ""
    assert rows_text(["sample", "fD", "ok"], ROWS[1:]) == "sample\tfD\tok\nS\t652.3412\tFalse\n"
    assert rows_text(["sample"], []) == ""


def test_nothing_selected_copies_the_current_row_with_the_header(qtbot):
    """The data tab and the list copy the same thing with nothing selected: the current row,
    with the column names (audit of 2026-11-15, Q-12: the data tab copied every row); Ctrl+C on
    the grid copies as the menu does."""
    panel = DataPanel()
    qtbot.addWidget(panel)
    panel.show_rows(["sample", "start"], ROWS)
    chosen = panel.table.selectionModel()
    chosen.setCurrentIndex(panel.model.index(1, 0), QItemSelectionModel.SelectionFlag.NoUpdate)
    assert not chosen.selectedIndexes()
    qtbot.keyClick(panel.table, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert QApplication.clipboard().text() == "sample\tstart\nS\t10\n"
    lst = SpectraList()
    qtbot.addWidget(lst)
    lst.set_rows(list(SPECTRA[0]), SPECTRA)
    lst.select_index(2, emit=False)
    qtbot.keyClick(lst.grid, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    lines = QApplication.clipboard().text().splitlines()
    assert lines[0].startswith("spectrum_id\t") and lines[1].startswith("s2\t") and len(lines) == 2


def test_rows_are_saved_as_hxex_writes_its_tables(tmp_path):
    path = save_rows(["sample", "fD"], ROWS, tmp_path / "t.csv")
    assert path.read_text(encoding="utf-8").splitlines() == ["sample,fD", "S,0.25",
                                                             "S,652.3412"]
    said = []
    assert saved_or_said(lambda: save_rows(["a"], ROWS, tmp_path), said.append) is None
    assert said[0].startswith("Not saved: ")


def test_the_list_says_a_table_it_could_not_save(qtbot, tmp_path, monkeypatch):
    """A path that cannot be written (here a folder) is said on the list's count line (audit of
    2026-11-15, Q-13: the error was raised inside the button's slot)."""
    lst = SpectraList()
    qtbot.addWidget(lst)
    lst.set_rows(list(SPECTRA[0]), SPECTRA)
    folder = tmp_path / "taken.csv"
    folder.mkdir()
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(folder), "CSV (*.csv)")))
    lst.save_dialog()  # no exception
    assert lst.count.text().startswith("Not saved: ")
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "shown"), "CSV (*.csv)")))
    lst.save_dialog()
    assert (tmp_path / "shown.csv").read_text(encoding="utf-8").startswith("spectrum_id,")


def test_the_list_counts_spectra_only_in_spectra_csv(qtbot):
    """series.csv, components.csv and uptake.csv have rows, not spectra (Q-41)."""
    lst = SpectraList()
    qtbot.addWidget(lst)
    lst.set_rows(list(SPECTRA[0]), SPECTRA)
    assert lst.count.text() == "3 of 3 spectra"
    series = [{"data_id": d, "sample": "S", "peptide_range": "0001-0010"} for d in range(2)]
    lst.set_rows(list(series[0]), series, table="series")
    assert lst.count.text() == "2 of 2 rows"


def test_the_lists_boxes_show_the_chosen_choices_tooltip(qtbot):
    """The list's tables and filter boxes: each choice its own tooltip, the box showing the
    chosen one's (audit of 2026-11-15, Q-18); the Find tooltip short, in lines (Q-40)."""
    lst = SpectraList()
    qtbot.addWidget(lst)
    assert "spectra.csv" in lst.tables.toolTip()
    lst.set_rows(["data_id"], [{"data_id": 0}], table="series")  # chosen without a signal
    assert "series.csv" in lst.tables.toolTip() and "spectra.csv" not in lst.tables.toolTip()
    lst.filter.setCurrentIndex(lst.filter.findData("rejected"))
    assert "rejected in the review" in lst.filter.toolTip()
    find = lst.search.toolTip()
    assert len(find.split()) <= 20 and len(find.splitlines()) >= 3  # was 30 words, one line
    assert all(len(line) <= tips.TIP_WIDTH for line in find.splitlines())
    assert find[0].islower() and not find.endswith(".")  # a phrase, as the others


def test_set_choice_tips_follows_the_selection(qtbot):
    box = QComboBox()
    qtbot.addWidget(box)
    box.addItem("one", "a")
    box.addItem("two", "b")
    tips.set_choice_tips(box, {"a": "**one**: the first", "b": "the second"})
    assert "<b>one</b>" in box.toolTip()
    box.setCurrentIndex(1)
    assert "the second" in box.toolTip()
    assert "the first" in box.itemData(0, Qt.ItemDataRole.ToolTipRole)


def test_one_error_colour():
    """The figure panel, the Project editor, the Tables view, and the list say problems in one
    red (Q-19)."""
    from hxex.gui import figure_view, project_editor, spectra_list, table_editor

    for module in (figure_view, project_editor, spectra_list, table_editor):
        assert module.ERROR_COLOUR is tips.ERROR_COLOUR
    assert table_editor.UNMATCHED_TEXT.name() == tips.ERROR_COLOUR


def test_dead_code_is_gone_and_span_mode_is_made_in_init(qtbot):
    """Q-20, Q-23: the Tables view's never-connected ``saved`` signal is gone; a figure panel
    has its span mode from the start."""
    from hxex.gui.figure_view import FigurePanel
    from hxex.gui.table_editor import TableEditor

    assert not hasattr(TableEditor, "saved")
    panel = FigurePanel()
    qtbot.addWidget(panel)
    assert panel.__dict__.get("span_mode", None) is False and panel.view.span_mode is False


def test_a_watched_file_is_watched_again_and_paused_while_written(qtbot, tmp_path):
    """Q-17: a file replaced is watched again when its change is reported; the window's own
    write is not watched, and the file is watched again after it, even when the write fails."""
    path = tmp_path / "hxex.toml"
    path.write_text("a = 1\n", encoding="utf-8")
    seen = []
    watched = WatchedFile(None, seen.append)
    assert watched.watch(path)  # True: watched
    assert watched.files() == [str(path)]
    with watched.paused():
        assert watched.files() == []
        path.write_text("a = 2\n", encoding="utf-8")
    assert watched.files() == [str(path)]
    with pytest.raises(OSError), watched.paused():
        raise OSError("refused")
    assert watched.files() == [str(path)]
    # replaced (as hxex writes its files, retrying while Windows briefly refuses the
    # replacement of a file open elsewhere): Qt drops it; the report watches it again
    new = tmp_path / "new.toml"
    new.write_text("a = 3\n", encoding="utf-8")
    replace_file(new, path)
    watched.watcher.removePaths(watched.files())  # as Qt does for a replaced file
    watched._reported(str(path))
    assert watched.files() == [str(path)] and seen == [str(path)]
    watched.watch(None)
    assert watched.files() == [] and watched.path is None
    missing = tmp_path / "later.csv"  # not there yet: watched once written
    watched.watch(missing)
    assert watched.files() == []
    with watched.paused():
        missing.write_text("x\n", encoding="utf-8")
    assert watched.files() == [str(missing)]
