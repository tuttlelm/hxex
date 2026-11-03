# SPDX-License-Identifier: GPL-3.0-or-later
"""The tables users write, for hxex's window (hxex.gui.user_tables): read and written back with
their columns, line endings and byte-order mark kept, checked by their readers, and a file
another program holds reported without losing the edits. No Qt."""
import pytest

from hxex.config import template
from hxex.gui.user_tables import (
    Table,
    TableLocked,
    kinds,
    read_table,
    table_problem,
    write_table,
)


def test_a_table_is_written_back_as_it_was_read(tmp_path):
    path = tmp_path / "peptides.csv"
    raw = "﻿peptide,start_seq,end_seq,charge,my notes\r\nAAAK,1,4,2,\"keep, this\"\r\n"
    path.write_bytes(raw.encode("utf-8"))
    table = read_table(path)
    assert table.columns == ["peptide", "start_seq", "end_seq", "charge", "my notes"]
    assert table.rows == [["AAAK", "1", "4", "2", "keep, this"]] and table.bom
    write_table(table, path)
    assert path.read_bytes() == raw.encode("utf-8")  # unchanged: columns, quotes, CRLF, BOM
    table.rows.append(["CCCK", "5", "8", "3", ""])
    write_table(table, path)
    assert read_table(path).rows[1] == ["CCCK", "5", "8", "3", ""]


def test_semicolons_and_utf16_are_read_and_written_as_the_readers_want(tmp_path):
    path = tmp_path / "rt.csv"
    path.write_bytes("sample;rt\nWT;3,5\n".encode("utf-16"))
    table = read_table(path)
    assert table.columns == ["sample", "rt"] and table.rows == [["WT", "3,5"]]
    assert table.separator == ";"
    write_table(table, path)
    assert path.read_bytes().decode("utf-8-sig") == 'sample,rt\nWT,"3,5"\n'


def test_short_and_long_rows_are_squared_to_the_header(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a,b,c\n1\n1,2,3,4,5\n\n,,\n")
    table = read_table(path)
    # nothing lost: the extra cells keep columns without a name; empty rows go
    assert table.columns == ["a", "b", "c", "", ""]
    assert table.rows == [["1", "", "", "", ""], ["1", "2", "3", "4", "5"]]


def test_a_file_held_by_another_program_keeps_the_edits(tmp_path, monkeypatch):
    import os

    path = tmp_path / "review.csv"
    table = Table(["spectrum_id", "review_status"], [["s1", "rejected"]])
    write_table(table, path)

    def locked(src, dst):
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(os, "replace", locked)
    table.rows[0][1] = "accepted"
    with pytest.raises(TableLocked, match="another program holds it"):
        write_table(table, path)
    assert "rejected" in path.read_text() and table.rows[0][1] == "accepted"
    assert not list(tmp_path.glob("*.hxex-tmp"))


def test_the_readers_check_a_table_before_it_is_saved(tmp_path):
    project = tmp_path / "hxex.toml"
    project.write_text(template("mzml"), encoding="utf-8")
    peptides, _, review = kinds("mzml")
    good = Table(list(peptides.columns), [["WT", "AAAK", "1", "4", "2", "", "", ""]])
    assert table_problem("input.peptides", good, project) is None
    other = Table(list(peptides.columns), [["mutant", "AAAK", "1", "4", "2", "", "", ""]])
    assert "mutant" in table_problem("input.peptides", other, project)  # not a run's sample
    missing = Table(["peptide", "charge"], [["AAAK", "2"]])
    problem = table_problem("input.peptides", missing, project)
    assert "start_seq" in problem and "tmp" not in problem  # named "the table", not the copy
    bad = Table(list(review.columns), [["s1", "maybe", "", ""]])
    assert "maybe" in table_problem("review.table", bad, project)
    assert table_problem("input.retention_times", Table(["sample"], [["WT"]]), project)
    assert not list(tmp_path.glob("*.csv"))  # the copies are gone
    folder = tmp_path / "folder.toml"
    folder.write_text(template("specexport"), encoding="utf-8")
    files = Table(list(kinds("specexport")[0].columns), [["a.csv", "WT", "AAAK", "1", "4", "2"]])
    assert table_problem("input.peptides", files, folder) is None
    with pytest.raises(ValueError):
        kinds("csv")


# --- the Tables view of the window (offscreen) ---------------------------------------------------

@pytest.fixture
def tables_view(qtbot, tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QMessageBox

    from hxex.gui.table_editor import TablesView
    from hxex.project_edit import live_settings

    project = tmp_path / "hxex.toml"
    project.write_text(template("mzml"), encoding="utf-8")
    view = TablesView()
    qtbot.addWidget(view)
    editor = view.editor
    editor.answers = {"anyway": False, "overwrite": False,
                      "unsaved": QMessageBox.StandardButton.Cancel}
    editor.told = []
    editor.ask_save_anyway = lambda problem: editor.answers["anyway"]
    editor.ask_overwrite = lambda: editor.answers["overwrite"]
    editor.ask_unsaved = lambda: editor.answers["unsaved"]
    editor.answers["remove columns"] = True
    editor.ask_remove_columns = lambda names: editor.answers["remove columns"]
    editor.tell = lambda title, text: editor.told.append(text)

    def refresh():
        view.show_project(project, live_settings(project.read_text())[0], "mzml")

    view.refresh = refresh
    refresh()
    return view, project


def test_a_new_table_is_made_named_in_the_project_and_edited(tables_view):
    import tomllib

    view, project = tables_view
    editor = view.editor
    view.show_table("input.peptides")
    assert editor.table is None and "No peptide table" in editor.heading.text()
    path = view.new_table(project.parent / "peptides")  # .csv added
    assert path.name == "peptides.csv"
    assert tomllib.loads(project.read_text())["input"]["peptides"] == "peptides.csv"
    view.refresh()
    assert editor.table.columns[:3] == ["sample", "peptide", "start_seq"]
    editor.add_row()
    for c, value in enumerate(["WT", "AAAK", "1", "4", "2", "", "", ""]):
        editor.grid.item(0, c).setText(value)
    assert editor.modified and editor.save() and not editor.modified
    assert read_table(path).rows == [["WT", "AAAK", "1", "4", "2", "", "", ""]]


def test_save_checks_the_table_and_asks_before_writing_a_problem(tables_view):
    view, project = tables_view
    editor = view.editor
    view.show_table("input.peptides")
    path = view.new_table(project.parent / "peptides.csv")
    view.refresh()
    editor.add_row()
    for c, value in enumerate(["mutant", "AAAK", "1", "4", "2"]):  # not a run's sample
        editor.grid.item(0, c).setText(value)
    assert not editor.save() and "mutant" in editor.problem.text()
    assert read_table(path).rows == [] and editor.modified  # not written, edits kept
    editor.answers["anyway"] = True
    assert editor.save() and read_table(path).rows[0][0] == "mutant"


def test_a_file_changed_elsewhere_is_not_written_over_without_asking(tables_view):
    import os

    view, project = tables_view
    editor = view.editor
    view.show_table("review.table")
    path = view.new_table(project.parent / "user_review_table.csv")
    view.refresh()
    editor.add_column("my column")
    path.write_text("spectrum_id,review_status,review_K,note\ns9,rejected,,\n")
    os.utime(path, ns=(1, 1))  # a different time, as another program would leave
    assert not editor.save()  # asked, and the answer was no
    assert "s9" in path.read_text()
    editor.answers["overwrite"] = True
    assert editor.save() and "my column" in path.read_text()


def test_unsaved_edits_are_asked_about_before_leaving(tables_view):
    from PySide6.QtWidgets import QMessageBox

    view, project = tables_view
    editor = view.editor
    view.show_table("review.table")
    view.new_table(project.parent / "user_review_table.csv")
    view.refresh()
    editor.add_row()
    view.show_table("input.peptides")  # cancelled: still on the review table
    assert editor.kind.setting == "review.table" and editor.modified
    editor.answers["unsaved"] = QMessageBox.StandardButton.Discard
    view.show_table("input.peptides")
    assert editor.kind.setting == "input.peptides" and not editor.modified


def test_a_held_file_is_reported_and_the_edits_kept(tables_view, monkeypatch):
    import os

    view, project = tables_view
    editor = view.editor
    view.show_table("review.table")
    view.new_table(project.parent / "user_review_table.csv")
    view.refresh()
    editor.add_row()
    editor.grid.item(0, 0).setText("s1")
    editor.grid.item(0, 1).setText("rejected")

    def locked(src, dst):
        raise PermissionError(13, "in use")

    monkeypatch.setattr(os, "replace", locked)
    assert not editor.save() and editor.modified
    assert "another program holds it" in editor.told[0]


def test_the_buttons_ask_for_a_name_and_a_file(tables_view, monkeypatch):
    """Clicked, not called: a button passes its checked state, which must not be taken for the
    column name or the file (the owner found Add a column doing nothing)."""
    from PySide6.QtWidgets import QFileDialog, QInputDialog

    view, project = tables_view
    editor = view.editor
    view.show_table("review.table")
    chosen = project.parent / "review.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(chosen), "")))
    view.new_button.click()
    assert chosen.exists()
    view.refresh()
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("my notes", True)))
    editor.buttons["add_column"].click()
    assert editor.table.columns[-1] == "my notes" and editor.modified
    editor.buttons["add_row"].click()
    assert len(editor.table.rows) == 1
    editor.buttons["save"].click()
    assert "my notes" in chosen.read_text() and not editor.modified


def test_selected_columns_are_removed_after_asking(tables_view):
    view, project = tables_view
    editor = view.editor
    view.show_table("review.table")
    path = view.new_table(project.parent / "review.csv")
    view.refresh()
    editor.add_column("my notes")
    editor.add_column("scratch")
    editor.add_row()
    editor.grid.item(0, 0).setText("s1")
    editor.grid.item(0, 4).setText("keep me")
    editor.answers["remove columns"] = False
    editor.grid.selectColumn(5)
    editor.buttons["remove_column"].click()
    assert editor.table.columns[-1] == "scratch"  # asked, and the answer was no
    editor.answers["remove columns"] = True
    editor.grid.selectColumn(5)
    editor.buttons["remove_column"].click()
    assert editor.table.columns == ["spectrum_id", "review_status", "review_K", "note",
                                    "my notes"]
    assert editor.table.rows == [["s1", "", "", "", "keep me"]]
    assert editor.save() and read_table(path).columns[-1] == "my notes"
