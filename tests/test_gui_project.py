# SPDX-License-Identifier: GPL-3.0-or-later
"""The project editor of hxex's window (hxex.gui.project_editor): settings written as hxex set
writes them, refused changes, stages switched on and off, the format's settings, and changes
made outside the window. Runs offscreen."""
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

import tomllib

from PySide6.QtCore import QSettings

from hxex.config import template
from hxex.gui.project_editor import (
    ProjectEditor,
    full_location,
    plain,
    relative,
)
from hxex.gui.window import MainWindow, new_project


@pytest.fixture
def project(tmp_path):
    (tmp_path / "SpecExport").mkdir()
    path = tmp_path / "hxex.toml"
    path.write_text(template(), encoding="utf-8")
    return path


@pytest.fixture
def editor(qtbot, project):
    ed = ProjectEditor()
    qtbot.addWidget(ed)
    ed.open(project)
    return ed


def _page(editor, name):
    editor.show_table(name)


def _data(path):
    return tomllib.loads(path.read_text())


def test_a_typed_value_is_written_as_hxex_set_writes_it(editor, project):
    _page(editor, "run.controls")
    row = editor.rows["run.controls.d2o_fraction"]
    row.editor.setText("0.85")
    row.editor.editingFinished.emit()
    assert _data(project)["run"]["controls"]["d2o_fraction"] == 0.85
    assert row.label.font().bold() and row.reset.isEnabled()
    assert "run.controls" in editor.on  # setting it switched the stage on
    # text stays text, even when it looks like a number
    _page(editor, "output")
    bundle = editor.rows["output.bundle"]
    bundle.editor.setText("2024")
    bundle.editor.editingFinished.emit()
    assert _data(project)["output"]["bundle"] == "2024"
    # back to the default: the line gets a # in front, the comment stays
    _page(editor, "run.controls")
    editor.rows["run.controls.d2o_fraction"].reset.click()
    assert "d2o_fraction" not in _data(project)["run"]["controls"]
    assert not editor.rows["run.controls.d2o_fraction"].label.font().bold()


def test_a_refused_value_leaves_the_file_and_says_why(editor, project, caplog):
    before = project.read_text()
    _page(editor, "run.bootstrap")
    row = editor.rows["run.bootstrap.n_boot"]
    row.editor.setText("many")
    row.editor.editingFinished.emit()
    assert project.read_text() == before
    assert "n_boot" in caplog.text and not editor.problem.isHidden()
    assert row.editor.text() == ""  # as the file still has it (not set)


def test_true_false_and_choices(editor, project):
    _page(editor, "output")
    overwrite = editor.rows["output.overwrite"]
    assert not overwrite.editor.isChecked()  # the default
    overwrite.editor.setChecked(True)
    assert _data(project)["output"]["overwrite"] is True
    _page(editor, "input")
    fmt = editor.rows["input.format"]
    assert fmt.editor.currentText() == "specexport"
    # the format decides which input settings are shown
    assert "input.data_dir" in editor.rows and "input.data_root" not in editor.rows
    assert "input.runs" not in editor.rows  # a list of tables: its own page
    # the old format's settings are switched off with the change, when the user agrees
    asked = []
    editor.confirm_switch_off = lambda f, names: asked.append(names) or False
    fmt.editor.setCurrentText("mzml")
    fmt.editor.activated.emit(fmt.editor.currentIndex())
    assert asked == [["input.data_dir"]] and _data(project)["input"]["format"] == "specexport"
    assert editor.rows["input.format"].editor.currentText() == "specexport"
    # a file being filled in: written, and the banner says what mzml still needs
    editor.confirm_switch_off = lambda f, names: True
    fmt = editor.rows["input.format"]
    fmt.editor.setCurrentText("mzml")
    fmt.editor.activated.emit(fmt.editor.currentIndex())
    assert _data(project)["input"]["format"] == "mzml"
    assert "data_dir" not in _data(project)["input"]
    assert not editor.problem.isHidden() and "peptides" in editor.problem.text()
    assert "input.data_root" in editor.rows and "input.data_dir" not in editor.rows
    editor.every_format.setChecked(True)
    assert {"input.data_root", "input.data_dir"} <= set(editor.rows)


def test_a_stage_is_switched_on_and_off(editor, project):
    _page(editor, "run.bootstrap")
    assert editor.switch is not None and not editor.switch.isChecked()
    editor.switch.setChecked(True)
    assert "bootstrap" in _data(project)["run"]
    editor.switch.setChecked(False)
    assert "bootstrap" not in _data(project)["run"]
    _page(editor, "input.exclude")
    assert editor.switch is None and not editor.rows  # a list of tables: a text editor for now


def test_search_finds_settings_in_every_table(editor):
    editor.search.setText("d2o_fraction")
    assert {"run.controls.d2o_fraction", "analysis.labelling.d2o_fraction"} <= set(editor.rows)
    editor.search.setText("")
    assert "run.controls.d2o_fraction" not in editor.rows or editor.list.currentRow() >= 0


def test_a_change_made_outside_the_window_shows(editor, project, qtbot):
    _page(editor, "run.controls")
    project.write_text(project.read_text().replace("# d2o_fraction = 0.9", "d2o_fraction = 0.8", 1)
                       .replace("# [run.controls]", "[run.controls]"))
    editor._file_changed(str(project))
    assert editor.rows["run.controls.d2o_fraction"].editor.text() == "0.8"
    project.write_text(project.read_text().replace('format = "specexport"', 'format = "csv"'))
    editor._file_changed(str(project))
    assert not editor.problem.isHidden() and "csv" in editor.problem.text()


def test_paths_are_written_relative_to_the_project(tmp_path):
    assert relative(tmp_path / "data" / "x.csv", tmp_path) == "data/x.csv"
    elsewhere = tmp_path.parent / "other.csv"
    assert relative(elsewhere, tmp_path) == str(elsewhere)
    assert plain("See [Input formats](inputs.md) and `x`.") == "See Input formats and x."
    assert plain("belong to *hxex*: n * k") == "belong to hxex: n * k"  # italics, not products


def test_a_path_setting_shows_where_it_points(editor, project, tmp_path):
    """Under a file or folder setting, its full location: a short form counted from the
    project file's folder (owner, 2026-11-13: the [output] page did not say where the bundle
    goes), following what is typed; nothing for a setting left unset."""
    _page(editor, "output")
    bundle = editor.rows["output.bundle"]
    assert bundle.where.text() == f"→ {tmp_path / 'results'}"
    bundle.editor.setText("../elsewhere/run1")
    bundle.editor.textEdited.emit("../elsewhere/run1")
    assert bundle.where.text() == f"→ {tmp_path.parent / 'elsewhere' / 'run1'}"
    assert editor.rows["output.overwrite"].where is None  # not a path
    _page(editor, "input")
    assert editor.rows["input.data_dir"].where.text() == f"→ {tmp_path / 'SpecExport'}"
    assert editor.rows["input.retention_times"].where.isHidden()  # unset
    assert full_location("true", tmp_path) is None
    # a full location as written, on every system ("/abs/x.csv" has no drive on Windows,
    # where hxex reads it on the project's drive, as full_location does)
    elsewhere = tmp_path.parent / "abs" / "x.csv"
    assert full_location(f'"{elsewhere.as_posix()}"', tmp_path) == elsewhere
    assert full_location("~/x", None) == Path.home() / "x"
    assert full_location("x", None) is None  # no project file to count from


def test_a_new_project_opens_in_the_editor(qtbot, tmp_path):
    from PySide6.QtWidgets import QMessageBox

    path = new_project(tmp_path / "new.toml", "mzml", "trial")
    assert _data(path)["project"] == "trial" and _data(path)["input"]["format"] == "mzml"
    with pytest.raises(ValueError, match="project name"):
        new_project(tmp_path / "bad.toml", "mzml", "a:b")
    w = MainWindow(settings=QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat))
    w.ask_to_save = lambda: QMessageBox.StandardButton.Discard
    qtbot.addWidget(w)
    w.open_project(path)
    editor = w.project_page.editor
    assert editor.isEnabled() and editor.format == "mzml"
    assert "trial" in w.project_page.text.toPlainText()  # the File view


def test_a_project_never_saved_as_a_session_asks_on_closing(qtbot, tmp_path):
    """Open recent lists sessions, so a project closed without one would be lost from it
    (owner, 2026-11-13): closing asks even when the view did not change; once a session is
    saved, an unchanged one closes without asking, as before."""
    from PySide6.QtWidgets import QMessageBox

    from hxex.gui.session import SUFFIX

    path = new_project(tmp_path / "new.toml", "specexport", "trial")
    w = MainWindow(settings=QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    asked = []
    w.ask_to_save = lambda: asked.append(1) or QMessageBox.StandardButton.Cancel
    w._new_session_with(path)  # as File > New project does
    assert not w.modified() and w.never_saved()
    assert not w._may_discard() and asked == [1]  # asked; Cancel keeps it open
    w.session_path = tmp_path / f"new{SUFFIX}"
    assert w.save_session() and not w.never_saved()
    assert w._may_discard() and asked == [1]  # saved and unchanged: not asked


@pytest.fixture
def scan_editor(qtbot, tmp_path):
    path = tmp_path / "hxex.toml"
    path.write_text(template("mzml"), encoding="utf-8")
    ed = ProjectEditor()
    qtbot.addWidget(ed)
    ed.open(path)
    return ed, path


def test_a_list_of_tables_is_edited_entry_by_entry(scan_editor):
    editor, path = scan_editor
    _page(editor, "input.runs")
    grid = editor.list_editor.grid
    keys = editor.list_editor.keys
    assert grid.rowCount() == 2 and grid.item(0, keys.index("path")).text() == "UN_01.mzML"
    grid.item(1, keys.index("rep")).setText("2")  # written at once
    assert _data(path)["input"]["runs"][1]["rep"] == 2
    grid.item(1, keys.index("sample")).setText("2024")  # text stays text
    assert _data(path)["input"]["runs"][1]["sample"] == "2024"
    editor.list_editor.add()
    assert len(_data(path)["input"]["runs"]) == 3 and grid.rowCount() == 3
    grid.item(2, keys.index("path")).setText("D_5min.mzML")
    assert _data(path)["input"]["runs"][2] == {"path": "D_5min.mzML"}
    assert "sample" in editor.problem.text()  # what the new run still needs
    # moved down and back up: the file's order changes and the entry stays selected
    grid.setCurrentCell(0, keys.index("rep"))
    editor.list_editor.move(1)
    assert [r["path"] for r in _data(path)["input"]["runs"]][:2] == [
        "D_60s_01.mzML", "UN_01.mzML"]
    grid = editor.list_editor.grid
    assert grid.currentRow() == 1 and grid.currentColumn() == keys.index("rep")
    editor.list_editor.move(-1)
    assert _data(path)["input"]["runs"][0]["path"] == "UN_01.mzML"
    grid = editor.list_editor.grid
    grid.setCurrentCell(0, 0)
    editor.list_editor.remove_selected()
    assert [r["path"] for r in _data(path)["input"]["runs"]] == ["D_60s_01.mzML", "D_5min.mzML"]
    grid.item(0, keys.index("rep")).setText("second")  # a bad value: refused, cell restored
    assert _data(path)["input"]["runs"][0]["rep"] == 2
    assert grid.item(0, keys.index("rep")).text() == "2"
    assert "rep" in editor.problem.text()


def test_a_list_of_tables_is_copied_as_a_table(scan_editor, qtbot):
    """The selected cells (Ctrl+C) and the whole table with its column names, tab-separated, as
    a spreadsheet pastes them (owner, 2026-11-09: it could not be copied as a table)."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    editor, _ = scan_editor
    _page(editor, "input.runs")
    lists = editor.list_editor
    keys, grid = lists.keys, lists.grid
    path_c = keys.index("path")
    lines = lists.table_text().splitlines()
    assert lines[0].split("\t") == keys and len(lines) == 3
    assert lines[1].split("\t")[path_c] == "UN_01.mzML"
    lists.copy_table()
    assert QApplication.clipboard().text() == lists.table_text()
    grid.clearSelection()
    grid.setCurrentCell(0, path_c)
    grid.setCurrentCell(1, path_c, grid.selectionModel().SelectionFlag.Select)
    qtbot.keyClick(grid, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert QApplication.clipboard().text() == "UN_01.mzML\nD_60s_01.mzML\n"
    assert lists.copied_text().count("\t") == 0  # one column: no tabs
    qtbot.keyClick(grid, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)  # every cell
    qtbot.keyClick(grid, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert QApplication.clipboard().text() == lists.table_text()  # with the column names
    # nothing selected: the current row with the column names (audit of 2026-11-15, Q-12)
    grid.clearSelection()
    grid.setCurrentCell(1, path_c, grid.selectionModel().SelectionFlag.NoUpdate)
    lines = lists.copied_text().splitlines()
    assert lines[0].split("\t") == keys and lines[1].split("\t")[path_c] == "D_60s_01.mzML"


def test_a_table_of_names_is_edited(editor, project):
    _page(editor, "analysis.controls")
    names = editor.list_editor
    names.name.setText("d2o_fraction")
    names.value.setText("0.85")
    names.add()
    assert _data(project)["analysis"]["controls"]["d2o_fraction"] == 0.85
    assert names.grid.rowCount() == 1
    names.grid.item(0, 1).setText("0.8")
    assert _data(project)["analysis"]["controls"]["d2o_fraction"] == 0.8
    names.grid.setCurrentCell(0, 0)
    names.remove_selected()
    assert "controls" not in _data(project).get("analysis", {}) or \
        "d2o_fraction" not in _data(project)["analysis"]["controls"]


def test_one_fasta_for_every_sample_or_one_per_sample(editor, project):
    """[analysis] fasta is one file for every sample; [analysis.fasta] one per sample, on a page
    of its own, typed without quotes; writing one form takes the other out of the file (TOML
    holds one), and the single-file row says when the table is in use (owner, 2026-11-05: the
    row's description read as if the table could be typed there)."""
    _page(editor, "analysis")
    row = editor.rows["analysis.fasta"]
    assert "for one file per sample use [analysis.fasta]" in row.about.text()
    row.editor.setText("combined.fasta")
    row.editor.editingFinished.emit()
    assert _data(project)["analysis"]["fasta"] == "combined.fasta"
    _page(editor, "analysis.fasta")
    per = editor.list_editor
    per.name.setText("WT")
    per.value.setText("SpecExport/WT.fasta")
    per.add()
    per.name.setText("D109H")
    per.value.setText("SpecExport/D109H.fasta")
    per.add()
    assert _data(project)["analysis"]["fasta"] == {"WT": "SpecExport/WT.fasta",
                                                   "D109H": "SpecExport/D109H.fasta"}
    assert per.grid.item(0, 1).text().endswith(".fasta")  # shown without quotes
    _page(editor, "analysis")
    row = editor.rows["analysis.fasta"]
    assert "one file per sample (2 set)" in row.editor.placeholderText()
    row.editor.setText("one.fasta")
    row.editor.editingFinished.emit()
    assert _data(project)["analysis"]["fasta"] == "one.fasta"  # the table taken out


def test_pooled_classes_by_name_text_or_by_their_records(editor, project):
    """[input] pooled_classes is text in the record ids; [input.pooled_classes] names each
    class's records, typed as names separated by commas (owner, 2026-11-09)."""
    _page(editor, "input.pooled_classes")
    per = editor.list_editor
    assert per.value.placeholderText() == "its records, separated by commas"
    per.name.setText("first")
    per.value.setText("A1, A2")
    per.add()
    assert _data(project)["input"]["pooled_classes"] == {"first": ["A1", "A2"]}
    assert per.grid.item(0, 1).text() == "A1, A2"  # shown as typed


def test_the_samples_are_a_list_in_order(editor, project):
    editor.sample_names = lambda: ["S", "V", "WT"]  # the results' samples, as the window gives
    _page(editor, "analysis")
    order = editor.rows["analysis.samples"].editor
    order.show()
    assert [order.add_box.itemText(i) for i in range(order.add_box.count())] == ["S", "V", "WT"]
    for name in ("V", "WT"):
        order.add_box.setCurrentText(name)
        order.add_button.click()
    assert _data(project)["analysis"]["samples"] == ["V", "WT"]
    assert [order.add_box.itemText(i) for i in range(order.add_box.count())] == ["S"]
    order.list.setCurrentRow(1)
    order.up.click()  # WT first
    assert _data(project)["analysis"]["samples"] == ["WT", "V"]
    assert order.items() == ["WT", "V"]
    order.list.setCurrentRow(0)
    order.down.click()
    assert _data(project)["analysis"]["samples"] == ["V", "WT"]
    order.list.setCurrentRow(0)
    order.remove.click()
    order.list.setCurrentRow(0)
    order.remove.click()  # an empty list: back to every sample
    assert "samples" not in _data(project).get("analysis", {})


def test_sample_colours_are_typed_bare_and_shown_as_swatches(editor, project):
    from PySide6.QtCore import Qt

    _page(editor, "analysis.colours")
    names = editor.list_editor
    names.name.setText("WT")
    names.value.setText("orange")  # no quotes needed
    names.add()
    assert _data(project)["analysis"]["colours"]["WT"] == "orange"
    names.name.setText("V")
    names.value.setText("#1f77b4")  # a hex code is no comment
    names.add()
    assert _data(project)["analysis"]["colours"]["V"] == "#1f77b4"
    swatch = names.grid.item(0, 1).data(Qt.ItemDataRole.DecorationRole)
    assert swatch is not None and swatch.name() in ("#ffa500", "#1f77b4")


def test_the_top_level_tables_stand_out_in_the_list(editor):
    from PySide6.QtCore import Qt

    items = [editor.list.item(i) for i in range(editor.list.count())]
    named = [i for i in items if i.data(Qt.ItemDataRole.UserRole) is not None]
    bold = [i.text() for i in named if i.font().bold()]
    assert bold == ["Top level", "[input]", "[output]", "[run]", "[review]", "[analysis]",
                    "[figures]"]
    gaps = [i for i in items if i.data(Qt.ItemDataRole.UserRole) is None]
    assert len(gaps) == len(bold) - 1 and all(not g.flags() & Qt.ItemFlag.ItemIsSelectable
                                              for g in gaps)
    with pytest.raises(KeyError):
        editor.show_table("nothing")


def test_changes_are_undone_and_redone_and_the_last_one_is_shown(editor, project):
    """Settings are easily changed by accident (owner): every change the window makes can be
    undone and redone, the file put back byte for byte, and the last change is named."""
    original = project.read_bytes()
    assert not editor.undo_button.isEnabled() and "No change" in editor.last_change.text()
    _page(editor, "run.controls")
    row = editor.rows["run.controls.d2o_fraction"]
    row.editor.setText("0.85")
    row.editor.editingFinished.emit()
    changed = project.read_bytes()
    assert changed != original
    assert editor.last_change.text() == "Last change: [run.controls] d2o_fraction: default → 0.85"
    assert editor.undo_button.isEnabled() and "d2o_fraction" in editor.undo_button.toolTip()
    # the format, with the settings it switched off: one step
    _page(editor, "input")
    editor.confirm_switch_off = lambda f, names: True
    fmt = editor.rows["input.format"]
    fmt.editor.setCurrentText("mzml")
    fmt.editor.activated.emit(fmt.editor.currentIndex())
    assert _data(project)["input"]["format"] == "mzml" and len(editor.undo_steps) == 2
    assert editor.undo()
    assert project.read_bytes() == changed and editor.rows["input.format"].editor.currentText() \
        == "specexport"
    assert editor.undo() and project.read_bytes() == original
    assert not editor.undo()  # nothing left
    assert editor.redo() and project.read_bytes() == changed
    assert editor.redo() and _data(project)["input"]["format"] == "mzml"
    assert not editor.redo_button.isEnabled()
    # a new change after an undo: nothing to redo
    assert editor.undo()
    _page(editor, "output")
    editor.rows["output.overwrite"].editor.setChecked(True)
    assert not editor.redo_steps
    # changed elsewhere since: not undone, that change kept, and said
    project.write_text(project.read_text() + "\n# typed in an editor\n", encoding="utf-8")
    elsewhere = project.read_bytes()
    assert not editor.undo()
    assert project.read_bytes() == elsewhere and "not undone" in editor.problem.text()
    assert not editor.undo_steps and not editor.undo_button.isEnabled()


def test_the_mouse_wheel_does_not_change_a_list(editor, project):
    """One notch of the wheel over a list rewrote a setting while the form was scrolled
    (owner): the lists of the form ignore the wheel."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication

    from hxex.gui.project_editor import NoWheelComboBox

    _page(editor, "input")
    fmt = editor.rows["input.format"].editor
    assert isinstance(fmt, NoWheelComboBox)
    before = project.read_bytes()
    wheel = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(fmt, wheel)
    assert fmt.currentText() == "specexport" and project.read_bytes() == before
    assert not wheel.isAccepted()  # passed on: the form scrolls


def test_the_back_exchange_page_says_the_correction_is_automatic(editor, project):
    """[run.controls] off does not mean no correction (owner, 2026-11-09: the data decide):
    the page says which state the file is in, Always on switches the table on, and Off writes
    [run] controls = false, each taking the other form out of the file (TOML holds one)."""
    _page(editor, "run.controls")
    assert editor.controls_state.text().startswith("Now: automatic")
    assert "d2o_fraction" in editor.controls_state.text()
    assert not editor.switch.isChecked() and not editor.controls_off.isChecked()
    assert editor.switch.text().startswith("Always on")
    editor.controls_off.setChecked(True)
    assert _data(project)["run"]["controls"] is False
    assert editor.controls_state.text().startswith("Now: off")
    assert editor.controls_off.isChecked() and not editor.switch.isChecked()
    editor.switch.setChecked(True)  # always on: the false goes, the table comes
    assert isinstance(_data(project)["run"]["controls"], dict)
    assert editor.controls_state.text().startswith("Now: always on")
    assert not editor.controls_off.isChecked()
    editor.controls_off.setChecked(True)  # off again: the table is switched off first
    assert _data(project)["run"]["controls"] is False
    editor.controls_off.setChecked(False)  # automatic again
    assert "controls" not in _data(project)["run"]
    assert editor.controls_state.text().startswith("Now: automatic")
    _page(editor, "run.bootstrap")  # other stages keep their Use box
    assert editor.controls_state is None and editor.switch.text().startswith("Use [run.bootstrap]")


def test_a_preset_is_chosen_from_a_list_and_fills_in_its_settings(editor, project):
    _page(editor, "")
    preset = editor.rows["preset"]
    assert preset.editor.currentText().startswith("(none")
    preset.editor.setCurrentText("quick")
    preset.editor.activated.emit(preset.editor.currentIndex())
    assert _data(project)["preset"] == "quick"
    _page(editor, "run.fit")
    k_max = editor.rows["run.fit.k_max"]
    assert k_max.editor.text() == "" and k_max.editor.placeholderText() == "2 (preset quick)"
    assert k_max.reset.text() == "Preset" and not k_max.reset.isEnabled()
    k_max.editor.setText("3")
    k_max.editor.editingFinished.emit()
    assert _data(project)["run"]["fit"]["k_max"] == 3
    k_max.reset.click()  # back to the preset's value: the line goes
    assert "k_max" not in _data(project)["run"]["fit"]
    _page(editor, "")
    preset = editor.rows["preset"]
    preset.editor.setCurrentIndex(0)  # none: the line goes, the defaults return
    preset.editor.activated.emit(0)
    assert "preset" not in _data(project)
    _page(editor, "run.fit")
    assert editor.rows["run.fit.k_max"].editor.placeholderText() == "3"
    assert editor.rows["run.fit.k_max"].reset.text() == "Default"


# --- the audit of 2026-11-15 (B-5, B-9, B-10, B-12, B-15, B-16, B-19, B-22, B-23) ------------

def test_a_name_needing_quotes_is_added_changed_and_removed(editor, project):
    """Names with a colon or a space could not be added or changed, and a change to a quoted
    name added a second, bare line (B-5, B-1)."""
    from hxex.project_edit import live_settings

    _page(editor, "analysis.summary")
    names = editor.list_editor
    names.name.setText("HDX reaction details")
    names.value.setText("pH 7.0, 25 C")
    names.add()
    names.name.setText("A:WT")
    names.value.setText("apo")
    names.add()
    values = live_settings(project.read_text())[0]
    assert values["analysis.summary.HDX reaction details"] == "pH 7.0, 25 C"
    assert values["analysis.summary.A:WT"] == "apo"
    row = next(r for r in range(names.grid.rowCount()) if names.grid.item(r, 0).text() == "A:WT")
    names.grid.item(row, 1).setText("holo")
    names._changed(row, 1)
    text = project.read_text()
    assert sum(line.startswith('"A:WT" = ') for line in text.splitlines()) == 1
    assert live_settings(text)[0]["analysis.summary.A:WT"] == "holo"
    names.grid.setCurrentCell(row, 0)
    names.remove_selected()
    assert "analysis.summary.A:WT" not in live_settings(project.read_text())[0]


def test_changing_the_format_switches_the_old_formats_tables_off(qtbot, tmp_path):
    """mzml -> specexport left [input.scans], [input.target] and the [[input.runs]] on, and
    the file then refused to load with nothing in the window to switch them off (B-9)."""
    from hxex.project_edit import file_problem, live_settings

    path = tmp_path / "hxex.toml"
    path.write_text(template("mzml"), encoding="utf-8")
    (tmp_path / "SpecExport").mkdir()
    ed = ProjectEditor()
    qtbot.addWidget(ed)
    ed.open(path)
    asked = []
    ed.confirm_switch_off = lambda fmt, names: (asked.append(names), True)[1]
    _page(ed, "input")
    row = ed.rows["input.format"]
    row.editor.setCurrentIndex(row.editor.findText("specexport"))
    row._chosen(row.editor.currentIndex())
    assert asked and {"[input.scans]", "[input.target]", "[input.runs]"} <= set(asked[0])
    values, tables = live_settings(path.read_text())
    assert values["input.format"] == "specexport"
    assert not ({"input.scans", "input.target", "input.runs"} & tables)
    assert "are for mzml" not in (file_problem(path) or "")
    assert ed.undo()  # one step: the format and the tables back
    assert live_settings(path.read_text())[0]["input.format"] == "mzml"
    assert "input.scans" in live_settings(path.read_text())[1]


def test_per_sample_tables_are_listed_on_their_page(editor, project):
    """Per-sample and per-run tables were kept in the file but invisible in the window, whose
    page invited the user to change the setting for every sample (B-10)."""
    from PySide6.QtWidgets import QLabel

    project.write_text(project.read_text() + '\n[analysis.labelling.D109H]\nph_read = 7.4\n'
                       '\n[analysis.fasta.A]\nWT = "a.fasta"\n', encoding="utf-8")
    editor.reload()
    _page(editor, "analysis.labelling")
    notes = [w.text() for w in editor.page.findChildren(QLabel)]
    assert any("[analysis.labelling.D109H] ph_read = 7.4" in t for t in notes)
    _page(editor, "analysis.fasta")
    notes = [w.text() for w in editor.page.findChildren(QLabel)]
    assert any('[analysis.fasta.A] WT = "a.fasta"' in t for t in notes)


def _list_names(editor):
    from PySide6.QtCore import Qt

    return [editor.list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(editor.list.count())]


def test_groups_are_added_renamed_and_removed_on_their_page(editor, project):
    """[analysis.groups] (PLAN Next steps 38, step 4): the groups' page adds, renames, and
    removes a group, each group has a page of its own in the list after it, and adding the
    first group takes [analysis] reference and samples into it."""
    from PySide6.QtWidgets import QLabel

    from hxex.gui.group_editor import AddGroupDialog, GroupsEditor

    project.write_text(project.read_text() + '\n[analysis]\nreference = "B1"\n'
                       'samples = ["B1", "B2"]\n', encoding="utf-8")
    editor.reload()
    _page(editor, "analysis.groups")
    page = editor.list_editor
    assert isinstance(page, GroupsEditor) and page.grid.rowCount() == 0
    dialog = AddGroupDialog(editor)  # starts from [analysis] reference and samples
    assert dialog.samples.items() == ["B1", "B2"] and dialog.reference.currentText() == "B1"
    dialog.deleteLater()
    assert page.add("B", "B1", ["B1", "B2"])
    data = _data(project)["analysis"]
    assert data["groups"]["B"] == {"reference": "B1", "samples": ["B1", "B2"]}
    assert "reference" not in data and "samples" not in data
    names = _list_names(editor)
    assert names.index("analysis.groups.B") == names.index("analysis.groups") + 1
    _page(editor, "analysis.groups")
    page = editor.list_editor
    assert not page.add("B", "X", ["X"])  # refused, said above the form
    assert "a group named B already" in editor.problem.text()
    assert page.add("C", "C1", ["C1"]) and "analysis.groups.C" in _list_names(editor)
    _page(editor, "analysis.groups")
    page = editor.list_editor
    assert page.grid.rowCount() == 2 and page.grid.item(1, 3).text() == "shared"
    page.grid.selectRow(1)
    assert page.rename_selected("D")
    assert list(_data(project)["analysis"]["groups"]) == ["B", "D"]
    assert "analysis.groups.D" in _list_names(editor) and "analysis.groups.C" \
        not in _list_names(editor)
    _page(editor, "analysis.groups.D")
    assert editor.list_editor.name == "D"  # the group's page is shown
    _page(editor, "analysis.groups")
    editor.list_editor.grid.selectRow(1)
    assert editor.list_editor.remove_selected()
    assert list(_data(project)["analysis"]["groups"]) == ["B"]
    # the page of a removed group gives way to the groups' page
    assert _list_names(editor)[editor.list.currentRow()] == "analysis.groups"
    # the [analysis] page says the groups name their own reference and samples
    _page(editor, "analysis")
    notes = [w.text() for w in editor.page.findChildren(QLabel)]
    assert any("Groups of samples (B)" in t for t in notes)
    assert not any("for one sample or run each" in t for t in notes)


def test_a_groups_page_edits_its_reference_samples_labels_and_regions(editor, project):
    from hxex.gui.group_editor import GroupPage

    project.write_text(project.read_text() + '\n[analysis.groups.B]\nreference = "B1"\n'
                       'samples = ["B1", "B2"]\n\n[[analysis.regions]]\nname = "N"\n'
                       'start = 1\nend = 5\n', encoding="utf-8")
    editor.reload()
    _page(editor, "analysis.groups.B")
    page = editor.list_editor
    assert isinstance(page, GroupPage)
    assert page.samples.pairs() == [("B1", "B1"), ("B2", "B2")]
    assert "takes the shared [[analysis.regions]] (1)" in page.states["regions"].text()
    # a vial label: the samples become a table of label = sample
    page.samples.grid.item(1, 0).setText("A+B")
    group = _data(project)["analysis"]["groups"]["B"]
    assert group["samples"] == {"B1": "B1", "A+B": "B2"}
    page = editor.list_editor
    page.samples.grid.item(1, 0).setText("B2")  # back to the names: a list again
    assert _data(project)["analysis"]["groups"]["B"]["samples"] == ["B1", "B2"]
    page = editor.list_editor
    page.reference.setCurrentIndex(1)
    page.reference.activated.emit(1)
    assert _data(project)["analysis"]["groups"]["B"]["reference"] == "B2"
    page = editor.list_editor
    page.samples.add_box.setCurrentText("B3")
    page.samples._add()
    assert _data(project)["analysis"]["groups"]["B"]["samples"] == ["B1", "B2", "B3"]
    page = editor.list_editor
    page.samples.grid.setCurrentCell(2, 1)
    page.samples._move(-1)
    assert _data(project)["analysis"]["groups"]["B"]["samples"] == ["B1", "B3", "B2"]
    # the group's own regions, entry by entry, in place of the shared ones
    page = editor.list_editor
    page.lists["regions"].add()
    page = editor.list_editor
    grid = page.lists["regions"].grid
    for column, text in enumerate(["Own", "2", "9"]):
        grid.item(0, column).setText(text)
    regions = _data(project)["analysis"]["groups"]["B"]["regions"]
    assert regions == [{"name": "Own", "start": 2, "end": 9}]
    assert "1 of its own" in editor.list_editor.states["regions"].text()
    # a changed group's settings undo as any other change
    assert editor.undo() and "regions" in _data(project)["analysis"]["groups"]["B"]


def test_a_value_none_of_the_choices_is_shown_as_such(editor, project):
    """The format list showed "specexport" for a file saying mzML (B-12); the Tables view
    stopped the window opening the project for it (B-3)."""
    from hxex.gui.user_tables import kinds

    project.write_text(project.read_text().replace('format = "specexport"', 'format = "mzML"'),
                       encoding="utf-8")
    editor.reload()
    _page(editor, "input")
    row = editor.rows["input.format"]
    assert row.editor.currentText() == "mzML (not one of the choices)"
    row._chosen(row.editor.currentIndex())  # choosing it again writes nothing
    assert 'format = "mzML"' in project.read_text()
    assert kinds("mzML") == []
    row.editor.setCurrentIndex(row.editor.findText("specexport"))
    row._chosen(row.editor.currentIndex())
    assert 'format = "specexport"' in project.read_text()
    row = editor.rows["input.format"]  # the page was built again for the new format
    assert row.editor.findText("mzML (not one of the choices)") < 0


def test_an_unset_true_or_false_setting_is_half_ticked(editor, project):
    """A true/false setting whose default is unset ([figures] corrected) was shown as an
    unticked box, which read as off (B-15)."""
    from PySide6.QtCore import Qt

    _page(editor, "figures")
    row = editor.rows["figures.corrected"]
    assert row.editor.isTristate()
    assert row.editor.checkState() == Qt.CheckState.PartiallyChecked
    row.editor.setCheckState(Qt.CheckState.Checked)
    assert _data(project)["figures"]["corrected"] is True
    row.editor.setCheckState(Qt.CheckState.PartiallyChecked)
    assert "corrected" not in _data(project).get("figures", {})
    _page(editor, "run.quality")  # a default of true or false: two states, as before
    assert not editor.rows["run.quality.exclude_flagged"].editor.isTristate() \
        if "run.quality.exclude_flagged" in editor.rows else True


def test_a_refused_switch_shows_the_problem(editor, project, monkeypatch):
    """A refused Use switch only logged a warning; the problem line stayed hidden (B-16)."""
    import hxex.gui.project_editor as module
    from hxex.config import ProjectError

    _page(editor, "run.bootstrap")

    def refuse(path, table, on, any_table=False, unfinished=False):
        raise ProjectError("[run.bootstrap] was not switched on: a test refusal")

    monkeypatch.setattr("hxex.project_edit.switch_table", refuse)
    editor.switch.setChecked(not editor.switch.isChecked())
    assert editor.problem.isVisibleTo(editor) and "test refusal" in editor.problem.text()
    assert editor.switch.isChecked() == ("run.bootstrap" in editor.on)
    assert module is not None


def test_a_date_value_and_quoted_text_are_handled(editor, project):
    """A TOML date as a value raised in the form (B-22); text typed with quotes, as the File tab
    shows it, was written with the quotes inside (B-23)."""
    from hxex.gui.project_editor import _shown

    assert _shown(__import__("datetime").date(2026, 11, 14)) == "2026-11-14"
    _page(editor, "output")
    row = editor.rows["output.bundle"]
    row.editor.setText('"res"')
    row.editor.editingFinished.emit()
    assert _data(project)["output"]["bundle"] == "res"


def test_a_project_name_with_a_quote_gives_a_valid_file(tmp_path):
    """A project name with a quote or a backslash gave a project file that was not valid TOML
    (B-19)."""
    from hxex.config import load_project

    path = new_project(tmp_path / "q.toml", "specexport", 'my "quoted" \\ name')
    assert load_project(path, analysis_only=True).project == 'my "quoted" \\ name'
