# SPDX-License-Identifier: GPL-3.0-or-later
"""The project editor of hxex's window (hxex.gui.project_editor): settings written as hxex set
writes them, refused changes, stages switched on and off, the format's settings, and changes
made outside the window. Runs offscreen."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import tomllib  # noqa: E402

from PySide6.QtCore import QSettings  # noqa: E402

from hxex.config import template  # noqa: E402
from hxex.gui.project_editor import ProjectEditor, plain, relative  # noqa: E402
from hxex.gui.window import MainWindow, new_project  # noqa: E402


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
    grid.setCurrentCell(0, 0)
    editor.list_editor.remove_selected()
    assert [r["path"] for r in _data(path)["input"]["runs"]] == ["D_60s_01.mzML", "D_5min.mzML"]
    grid.item(0, keys.index("rep")).setText("second")  # a bad value: refused, cell restored
    assert _data(path)["input"]["runs"][0]["rep"] == 2
    assert grid.item(0, keys.index("rep")).text() == "2"
    assert "rep" in editor.problem.text()


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
