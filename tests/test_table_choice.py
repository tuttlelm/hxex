# SPDX-License-Identifier: GPL-3.0-or-later
"""Where a table the window writes for the first time goes (hxex.gui.table_choice; owner,
2026-11-11): the name suggested carries the project file's name, a file another project file
names is said to be its, and nothing is added to an existing file without a yes."""

import os

import pytest

from hxex.gui import table_choice
from hxex.gui.table_choice import ask_table as REAL_ASK  # before the shared fixture patches it
from hxex.gui.table_choice import named_by_others, suggested_name

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_the_suggested_name_carries_the_project_file(tmp_path):
    assert suggested_name(tmp_path / "hxex.toml", "rt_elution_windows.csv") == (
        "rt_elution_windows.csv")
    assert suggested_name(tmp_path / "hxex_findpeak.toml", "rt_elution_windows.csv") == (
        "hxex_findpeak_rt_elution_windows.csv")


def test_the_project_files_that_name_a_table_are_found(tmp_path):
    (tmp_path / "hxex.toml").write_text('project = "a"\n[input]\nformat = "mzml"\n'
                                        'rt_elution_windows = "rt_elution_windows.csv"\n')
    (tmp_path / "other.toml").write_text('project = "b"\n[input]\nformat = "mzml"\n')
    (tmp_path / "notes.toml").write_text("not = [a project")
    found = named_by_others(tmp_path / "rt_elution_windows.csv", tmp_path / "other.toml",
                            "input.rt_elution_windows")
    assert found == ["hxex.toml"]
    assert named_by_others(tmp_path / "rt_elution_windows.csv", tmp_path / "hxex.toml",
                           "input.rt_elution_windows") == []


def test_the_dialog_suggests_the_name_and_asks_before_adding_to_a_table(qtbot, tmp_path,
                                                                        monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    monkeypatch.setattr(table_choice, "ask_table", REAL_ASK)
    (tmp_path / "hxex.toml").write_text('project = "a"\n[input]\nformat = "mzml"\n'
                                        'rt_elution_windows = "rt_elution_windows.csv"\n')
    (tmp_path / "rt_elution_windows.csv").write_text("sample,file,ion,rt_start,rt_end\n")
    project = tmp_path / "hxex_findpeak.toml"
    project.write_text('project = "b"\n[input]\nformat = "mzml"\n')
    offered, asked = [], []

    def save_dialog(parent, title, start, filters, options=None):
        offered.append(start)
        return chosen[0], filters

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(save_dialog))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda parent, title, text: asked.append(text) or answer[0]))
    # a new name: no question
    chosen, answer = [str(tmp_path / "hxex_findpeak_rt_elution_windows.csv")], [None]
    path = table_choice.ask_table(None, project, "input.rt_elution_windows",
                                  "rt_elution_windows.csv", "elution-window table")
    assert path == tmp_path / "hxex_findpeak_rt_elution_windows.csv" and not asked
    assert offered[-1].endswith("hxex_findpeak_rt_elution_windows.csv")
    # the other project's table: said to be its, added to only on yes
    chosen, answer = [str(tmp_path / "rt_elution_windows.csv")], [QMessageBox.StandardButton.No]
    assert table_choice.ask_table(None, project, "input.rt_elution_windows",
                                  "rt_elution_windows.csv", "elution-window table") is None
    assert "elution-window table of hxex.toml" in asked[-1]
    answer = [QMessageBox.StandardButton.Yes]
    assert table_choice.ask_table(None, project, "input.rt_elution_windows",
                                  "rt_elution_windows.csv", "elution-window table") == (
        tmp_path / "rt_elution_windows.csv")
    # cancelled: nothing
    chosen = [""]
    assert table_choice.ask_table(None, project, "input.rt_elution_windows",
                                  "rt_elution_windows.csv", "elution-window table") is None


def test_a_window_saved_to_a_chosen_table_is_named_there(tmp_path):
    """save_window writes the table the window chose for a project that names none, and names
    it in the project file (relative to it)."""
    from hxex.config import load_project
    from hxex.gui.window_edit import save_window
    from test_elution_windows import ION, PEPTIDE, _project

    project = _project(tmp_path)
    chosen = tmp_path / "mine.csv"
    saved = save_window(project, ("S", "un1.mzML", ION), (1.0, 1.2), table=chosen)
    assert saved.path == chosen and chosen.is_file()
    assert not (tmp_path / "rt_elution_windows.csv").exists()
    assert load_project(project).input.rt_elution_windows == chosen
    assert PEPTIDE in chosen.read_text()


@pytest.mark.parametrize("name", ["hxex.toml", "arm.toml"])
def test_the_shared_fixture_answers_with_the_suggested_name(tmp_path, name):
    path = table_choice.ask_table(None, tmp_path / name, "review.table",
                                  "user_review_table.csv", "review table")
    assert path.name == suggested_name(tmp_path / name, "user_review_table.csv")
