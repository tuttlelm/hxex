# SPDX-License-Identifier: GPL-3.0-or-later
"""The Project workspace's File tab (hxex.gui.file_view) and the library function behind its
Save (hxex.project_edit.write_text): finding text, editing and saving, refused text, Undo,
changes made outside while editing, and the question before leaving. Runs offscreen."""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox

from hxex.config import ProjectError, template
from hxex.gui.window import MainWindow
from hxex.project_edit import TextNotSaved, write_text


@pytest.fixture
def project(tmp_path):
    (tmp_path / "SpecExport").mkdir()
    path = tmp_path / "hxex.toml"
    path.write_text(template(), encoding="utf-8")
    return path


@pytest.fixture
def window(qtbot, tmp_path, project):
    w = MainWindow(settings=QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat))
    w.ask_to_save = lambda: QMessageBox.StandardButton.Discard  # no dialog in a test
    qtbot.addWidget(w)
    # the close at teardown asks about unsaved text: answered, never a real dialog
    w.project_page.text.ask_unsaved = lambda: QMessageBox.StandardButton.Discard
    w.open_project(project)
    return w


def _typed(view, old: str, new: str) -> None:
    text = view.toPlainText()
    assert old in text
    view.text.setPlainText(text.replace(old, new, 1))


def test_write_text_keeps_the_byte_order_mark_and_line_endings(project):
    project.write_bytes(b"\xef\xbb\xbf" + project.read_bytes().replace(b"\n", b"\r\n"))
    text = project.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    changed = text.replace("keep_previous = false ", "keep_previous = true  ")
    assert write_text(project, changed) is None
    raw = project.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and b"keep_previous = true" in raw
    assert raw.count(b"\r\n") == raw.count(b"\n")


def test_write_text_refuses_text_that_is_not_toml(project):
    before = project.read_bytes()
    with pytest.raises(TextNotSaved, match="not valid TOML") as caught:
        write_text(project, project.read_text() + "\n[broken\n")
    assert caught.value.line == project.read_text().count("\n") + 2
    assert isinstance(caught.value, ProjectError) and project.read_bytes() == before


def test_find_marks_every_match_and_steps_round_the_end(window):
    view = window.project_page.text
    view.find.setText("bundle")
    total = view.toPlainText().count("bundle")
    assert total > 1 and view.count.text().endswith(f"of {total}")
    assert len(view.text.extraSelections()) == total
    view.step(-1)
    first = view.count.text()
    view.step(1)
    assert view.count.text() != first
    for _ in range(total):
        view.step(1)
    assert view.text.textCursor().selectedText() == "bundle"
    view.find.setText("BUNDLE")
    assert view.count.text().endswith(f"of {total}")  # case ignored by default
    view.case.setChecked(True)
    assert view.count.text() == "no match"


def test_an_edit_is_written_only_on_save_and_undone_as_one_step(window, project):
    view = window.project_page.text
    views = window.project_page.views
    _typed(view, "keep_previous = false ", "keep_previous = true  ")
    assert view.modified and views.tabText(views.indexOf(view)) == "File (unsaved)"
    assert "keep_previous = true" not in project.read_text()
    assert view.save()
    assert "keep_previous = true" in project.read_text() and not view.modified
    assert views.tabText(views.indexOf(view)) == "File"
    editor = window.project_page.editor
    assert editor.values.get("output.keep_previous") is True  # the Settings form shows it
    assert "edited as text" in editor.last_change.text()
    assert editor.undo()
    assert "keep_previous = true" not in project.read_text()
    assert "keep_previous = true" not in view.toPlainText()


def test_text_that_is_not_toml_is_kept_and_the_line_shown(window, project):
    view = window.project_page.text
    before = project.read_bytes()
    view.text.setPlainText(view.toPlainText() + "\n[broken\n")
    assert not view.save()
    assert project.read_bytes() == before and view.modified
    assert "not valid TOML" in view.problem.text()
    assert view.text.textCursor().block().text() == "[broken"


def test_a_settings_problem_is_saved_and_shown(window, project):
    view = window.project_page.text
    _typed(view, 'format = "specexport"', 'format = "nonsense"')
    assert view.save()
    assert 'format = "nonsense"' in project.read_text()
    assert view.problem.isVisibleTo(view) and "input.format" in view.problem.text()


def test_edits_survive_a_change_made_outside_and_say_so(window, project):
    view = window.project_page.text
    _typed(view, "keep_previous = false ", "keep_previous = true  ")
    project.write_text(project.read_text() + "\n# from a text editor\n", encoding="utf-8")
    window.project_page.editor._file_changed(str(project))  # as the watcher reports it
    assert view.modified and "keep_previous = true" in view.toPlainText()
    assert view.outside_row.isVisibleTo(view)
    asked = []
    view.ask_overwrite = lambda: asked.append(1) or False
    assert not view.save() and asked == [1]  # Cancel: nothing written
    assert "keep_previous = true" not in project.read_text()
    view.discard()  # Load the file
    assert not view.modified and view.toPlainText().endswith("# from a text editor\n")
    assert not view.outside_row.isVisibleTo(view)


def test_leaving_the_project_asks_about_unsaved_text(window, project, tmp_path):
    view = window.project_page.text
    _typed(view, "keep_previous = false ", "keep_previous = true  ")
    view.ask_unsaved = lambda: QMessageBox.StandardButton.Cancel
    assert not window._may_replace()
    view.ask_unsaved = lambda: QMessageBox.StandardButton.Save
    assert window._may_replace() and "keep_previous = true" in project.read_text()
    _typed(view, "keep_previous = true  ", "keep_previous = false ")
    view.ask_unsaved = lambda: QMessageBox.StandardButton.Discard
    assert window._may_replace() and not view.modified
    assert "keep_previous = true" in project.read_text()


def test_no_project_means_nothing_to_edit(qtbot, tmp_path):
    w = MainWindow(settings=QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat))
    w.ask_to_save = lambda: QMessageBox.StandardButton.Discard
    qtbot.addWidget(w)
    view = w.project_page.text
    assert view.text.isReadOnly() and not view.modified
    assert view.may_leave()


def test_the_text_is_coloured_as_toml(qtbot):
    from PySide6.QtGui import QColor, QPalette, QTextDocument

    from hxex.gui.toml_colours import COLOURS, TomlColours

    doc = QTextDocument()
    doc.setPlainText('[run.controls]\nd2o_fraction = 0.85  # measured\n# overwrite = false\n'
                     'name = "a # not a comment"\nnotes = """two\nlines"""\nflag = true\n')
    light = QPalette()
    light.setColor(QPalette.ColorRole.Base, QColor("white"))
    colours = TomlColours(doc, light)
    colours.rehighlight()

    def colour_at(line: int, column: int) -> str | None:
        block = doc.findBlockByNumber(line)
        for fmt in block.layout().formats():
            if fmt.start <= column < fmt.start + fmt.length:
                return fmt.format.foreground().color().name()
        return None

    want = {name: QColor(pair[0]).name() for name, pair in COLOURS.items()}
    assert colour_at(0, 1) == want["heading"]
    assert colour_at(1, 0) == want["key"] and colour_at(1, 15) == want["number"]
    assert colour_at(1, 22) == want["comment"]
    assert colour_at(2, 3) == want["comment"]  # a switched-off setting
    assert colour_at(3, 12) == want["string"]  # a # inside text is text
    assert colour_at(5, 2) == want["string"]  # text over several lines
    assert colour_at(6, 8) == want["number"]
    dark = QPalette()
    dark.setColor(QPalette.ColorRole.Base, QColor("#1e1e1e"))
    colours.use_palette(dark)
    assert colour_at(0, 1) == QColor(COLOURS["heading"][1]).name()
