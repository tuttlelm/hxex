# SPDX-License-Identifier: GPL-3.0-or-later
"""Every action a user can reach in hxex's window, exercised through its widget (the menu
action triggered, the button clicked, the key pressed), with the file and message dialogs
answered as a user would: what each one does is checked, not only that nothing breaks (audit
of 2026-11-15, Q-27). Runs offscreen."""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QKeyCombination, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QFileDialog, QPushButton, QToolButton, QWidget

from hxex.config import template
from test_gui import _window, bundle  # noqa: F401 - the fixture
from test_gui_review import _first_spectrum, _second_spectrum, window_on  # noqa: F401
from test_window_edit import project as elution_project  # noqa: F401 - the fixture


@pytest.fixture
def window(qtbot, tmp_path):
    return _window(qtbot, tmp_path / "gui.ini")


@pytest.fixture
def project(tmp_path):
    (tmp_path / "SpecExport").mkdir()
    path = tmp_path / "hxex.toml"
    path.write_text(template(), encoding="utf-8")
    return path


class Asked:
    """A stand-in for one of QFileDialog's static functions: records each call's arguments and
    gives the answer set (a path, or "" for Cancel)."""

    def __init__(self, answer="", chosen=""):
        self.answer, self.chosen, self.calls = str(answer), chosen, []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.answer

    def pair(self, *args, **kwargs):  # getOpenFileName and getSaveFileName give two values
        self.calls.append((args, kwargs))
        return self.answer, self.chosen


def _ask(monkeypatch, name, answer="", chosen=""):
    asked = Asked(answer, chosen)
    function = asked if name == "getExistingDirectory" else asked.pair
    monkeypatch.setattr(QFileDialog, name, staticmethod(function))
    return asked


def _action(window, menu: str, text: str) -> QAction:
    """The action ``text`` of the menu bar's menu ``menu`` (shortcut ampersands ignored)."""
    for top in window.menuBar().actions():
        if top.text().replace("&", "") == menu:
            for action in top.menu().actions():
                if action.text().replace("&", "") == text:
                    return action
    raise LookupError(f"no {menu} > {text}")


def _button(widget, text: str, kind=QPushButton):
    (found,) = [b for b in widget.findChildren(kind) if b.text() == text]
    return found


def _press(qtbot, widget, key: QKeySequence.StandardKey) -> bool:
    """The platform's key for ``key`` pressed on ``widget``; False when this platform has
    none (Windows has no standard key for Save as or Quit)."""
    bindings = QKeySequence.keyBindings(key)
    if not bindings:
        return False
    combination: QKeyCombination = bindings[0][0]
    qtbot.keyClick(widget, combination.key(), combination.keyboardModifiers())
    return True


def _shown(qtbot, widget):
    with qtbot.waitExposed(widget):
        widget.show()
    widget.activateWindow()
    top = QWidget.window(widget)  # (a workspace's own .window is the main window it serves)
    qtbot.waitUntil(lambda: QApplication.activeWindow() is top, timeout=5000)


# --- the File and Help menus ------------------------------------------------------------------

def test_file_open_asks_for_a_session_project_or_bundle_and_opens_it(qtbot, window, project,
                                                                     monkeypatch):
    asked = _ask(monkeypatch, "getOpenFileName", project)
    _action(window, "File", "Open…").trigger()
    (args, _kw), = asked.calls
    assert args[1] == "Open" and "*.hxexgui" in args[3] and "*.toml" in args[3] \
        and "*.zip" in args[3]
    assert window.project == project.absolute()
    # cancelled: nothing changes
    _ask(monkeypatch, "getOpenFileName", "")
    _action(window, "File", "Open…").trigger()
    assert window.project == project.absolute()


def test_file_open_project_file_opens_the_one_chosen(qtbot, window, project, monkeypatch):
    asked = _ask(monkeypatch, "getOpenFileName", project)
    _action(window, "File", "Open project file…").trigger()
    (args, _kw), = asked.calls
    assert args[1] == "Open project file" and args[3].startswith("project files (*.toml)")
    assert window.project == project.absolute()
    assert window.project_page.editor.path == project.absolute()


def test_open_results_from_the_menu_and_the_results_button(qtbot, window, project, bundle,  # noqa: F811
                                                           monkeypatch):
    """File > Open results… and the Results workspace's own button: a folder asked for,
    starting in the project's folder, and its results shown."""
    window.open_asking(project)
    asked = _ask(monkeypatch, "getExistingDirectory", bundle)
    _action(window, "File", "Open results…").trigger()
    (args, _kw), = asked.calls
    assert args[1] == "Open results (a bundle folder)" and args[2] == str(project.parent)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    assert window.results_path == bundle.absolute()
    assert window.results_page.tree.topLevelItemCount() == 1
    # the button beside the results' path does the same
    asked = _ask(monkeypatch, "getExistingDirectory", "")
    _button(window.results_page, "Open results…").click()
    assert len(asked.calls) == 1 and window.results_path == bundle.absolute()


def test_file_new_project_file_writes_and_opens_it(qtbot, window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from hxex.gui import window as module

    monkeypatch.setattr(module.NewProjectDialog, "exec",
                        lambda self: (self.format.setCurrentText("mzml"),
                                      self.name.setText("trial"),
                                      QDialog.DialogCode.Accepted)[-1])
    asked = _ask(monkeypatch, "getSaveFileName", tmp_path / "made")
    _action(window, "File", "New project file…").trigger()
    (args, _kw), = asked.calls
    assert args[1:4] == ("Save the new project file", "hxex.toml", "project files (*.toml)")
    made = tmp_path / "made.toml"  # .toml added
    import tomllib

    data = tomllib.loads(made.read_text(encoding="utf-8"))
    assert data["project"] == "trial" and data["input"]["format"] == "mzml"
    assert window.project == made.absolute() and window.tabs.currentIndex() == 0


def test_save_session_as_asks_where_and_writes_the_session(qtbot, window, project, tmp_path,
                                                          monkeypatch):
    from hxex.gui.session import SUFFIX

    window.open_asking(project)
    asked = _ask(monkeypatch, "getSaveFileName", tmp_path / "view")
    _action(window, "File", "Save session as…").trigger()
    (args, _kw), = asked.calls
    assert args[1] == "Save session" and args[2] == str(project.with_suffix(SUFFIX).absolute())
    assert args[3] == f"hxex sessions (*{SUFFIX})"
    saved = tmp_path / f"view{SUFFIX}"
    assert saved.exists() and window.session_path == saved.absolute()
    assert window.recent()[0] == str(saved.absolute()) and not window.modified()
    # Save session writes it again without asking
    saved.unlink()
    _action(window, "File", "Save session").trigger()
    assert saved.exists() and len(asked.calls) == 1


def test_choosing_an_open_recent_entry_opens_it(qtbot, window, project):
    window._remember(project)
    (entry,) = window.recent_menu.actions()
    assert entry.text() == str(project) and window.recent_menu.isEnabled()
    entry.trigger()
    assert window.project == project.absolute()


def test_quit_closes_the_window(qtbot, window):
    window.show()
    _action(window, "File", "Quit").trigger()
    assert not window.isVisible()


def test_help_user_guide_opens_the_guide(qtbot, window, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    from hxex.gui.window import GUIDE_URL

    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl",
                        staticmethod(lambda url: opened.append(url.toString()) or True))
    _action(window, "Help", "User guide").trigger()
    assert opened == [GUIDE_URL]


def test_the_file_menus_keys(qtbot, window, project, tmp_path, monkeypatch):
    """Ctrl+O, Ctrl+Shift+S, Ctrl+S, Ctrl+N and Ctrl+Q (the platform's own keys)."""
    from hxex.gui.session import SUFFIX

    _shown(qtbot, window)
    asked = _ask(monkeypatch, "getOpenFileName", project)
    assert _press(qtbot, window, QKeySequence.StandardKey.Open)
    assert len(asked.calls) == 1 and window.project == project.absolute()
    saving = _ask(monkeypatch, "getSaveFileName", tmp_path / "keys")
    if _press(qtbot, window, QKeySequence.StandardKey.SaveAs):
        assert len(saving.calls) == 1 and (tmp_path / f"keys{SUFFIX}").exists()
    else:  # no Save as key here: Ctrl+S asks, the session never saved
        assert _press(qtbot, window, QKeySequence.StandardKey.Save)
        assert len(saving.calls) == 1
    (tmp_path / f"keys{SUFFIX}").unlink()
    assert _press(qtbot, window, QKeySequence.StandardKey.Save)  # saved again, not asked
    assert (tmp_path / f"keys{SUFFIX}").exists() and len(saving.calls) == 1
    assert _press(qtbot, window, QKeySequence.StandardKey.New)
    assert window.project is None and window.session_path is None
    if _press(qtbot, window, QKeySequence.StandardKey.Quit):
        assert not window.isVisible()


# --- the Project editor -----------------------------------------------------------------------

@pytest.fixture
def editor(qtbot, project):
    from hxex.gui.project_editor import ProjectEditor

    ed = ProjectEditor()
    qtbot.addWidget(ed)
    ed.open(project)
    return ed


def _data(path):
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_undo_and_redo_keys_in_the_project_editor(qtbot, editor, project):
    original = project.read_bytes()
    editor.show_table("run.controls")
    row = editor.rows["run.controls.d2o_fraction"]
    row.editor.setText("0.85")
    row.editor.editingFinished.emit()
    changed = project.read_bytes()
    assert changed != original
    _shown(qtbot, editor)
    editor.list.setFocus()
    assert _press(qtbot, editor.list, QKeySequence.StandardKey.Undo)
    assert project.read_bytes() == original
    assert _press(qtbot, editor.list, QKeySequence.StandardKey.Redo)
    assert project.read_bytes() == changed


def test_a_settings_browse_menu_chooses_a_file_or_a_folder(qtbot, editor, project, monkeypatch):
    editor.show_table("input")
    row = editor.rows["input.data_dir"]
    browse = _button(row, "Browse…", QToolButton)
    actions = {a.text(): a for a in browse.menu().actions()}
    asked = _ask(monkeypatch, "getExistingDirectory", project.parent / "SpecExport")
    actions["A folder…"].trigger()
    (args, _kw), = asked.calls
    assert args[1] == "input.data_dir" and args[2] == str(project.parent)
    assert _data(project)["input"]["data_dir"] == "SpecExport"  # relative to the project
    (project.parent / "elsewhere.txt").write_text("")
    asked = _ask(monkeypatch, "getOpenFileName", project.parent / "elsewhere.txt")
    actions["A file…"].trigger()
    (args, _kw), = asked.calls
    assert args[1] == "input.data_dir" and args[2] == str(project.parent)
    assert _data(project)["input"]["data_dir"] == "elsewhere.txt"


def test_browse_for_the_selected_cell_of_a_list_of_tables(qtbot, tmp_path, monkeypatch):
    from hxex.gui.project_editor import ProjectEditor

    path = tmp_path / "hxex.toml"
    path.write_text(template("mzml"), encoding="utf-8")
    ed = ProjectEditor()
    qtbot.addWidget(ed)
    ed.open(path)
    ed.show_table("input.runs")
    grid, keys = ed.list_editor.grid, ed.list_editor.keys
    asked = _ask(monkeypatch, "getOpenFileName", tmp_path / "data" / "new.mzML")
    grid.setCurrentCell(0, keys.index("sample"))  # not a file's column: not asked
    ed.list_editor.browse.click()
    assert asked.calls == []
    grid.setCurrentCell(0, keys.index("path"))
    ed.list_editor.browse.click()
    (args, _kw), = asked.calls
    assert args[1] == "path" and args[2] == str(tmp_path)
    assert _data(path)["input"]["runs"][0]["path"] == "data/new.mzML"


def test_the_names_editors_browse_chooses_a_fasta_file(qtbot, editor, project, monkeypatch):
    editor.show_table("analysis.fasta")
    names = editor.list_editor
    asked = _ask(monkeypatch, "getOpenFileName", project.parent / "SpecExport" / "WT.fasta")
    _button(names, "Browse…").click()
    (args, _kw), = asked.calls
    assert args[1] == "FASTA file" and args[2] == str(project.parent) and "*.fasta" in args[3]
    assert names.value.text() == "SpecExport/WT.fasta"
    names.name.setText("WT")
    names.add()
    assert _data(project)["analysis"]["fasta"] == {"WT": "SpecExport/WT.fasta"}


# --- the Run workspace ------------------------------------------------------------------------

@pytest.fixture
def run_page(qtbot, tmp_path, monkeypatch):
    from hxex.gui.run_workspace import RunWorkspace
    from test_cli import mzml_project

    page = RunWorkspace()
    qtbot.addWidget(page)
    page.show_project(mzml_project(tmp_path))
    started = []
    monkeypatch.setattr(page.runner, "start", lambda args, cwd: started.append((args, cwd)))
    return page, started


def test_report_draws_the_figures_too_adds_figures_to_report(run_page, tmp_path):
    page, started = run_page
    page.buttons["report"].click()
    page.figures.click()  # ticked
    assert page.figures.isChecked()
    page.buttons["report"].click()
    page.buttons["run"].click()  # only report takes it
    assert [args for args, _ in started] == [["report", "hxex.toml"],
                                             ["report", "hxex.toml", "--figures"],
                                             ["run", "hxex.toml"]]
    assert all(cwd == tmp_path for _, cwd in started)
    assert page.command.text() == "hxex run hxex.toml"


def test_clear_empties_the_console(run_page):
    page, _ = run_page
    page.console.appendPlainText("earlier output")
    assert page.console.toPlainText()
    _button(page, "Clear").click()
    assert page.console.toPlainText() == ""


# --- the Results workspace --------------------------------------------------------------------

@pytest.fixture(scope="module")
def results():
    from hxex.report import Results
    from test_run import config, raw_spectra, run

    return Results.from_run(run(raw_spectra(n_series=2), config(bootstrap=None)))


@pytest.fixture
def page(qtbot, results):
    from hxex.gui.results_workspace import ResultsWorkspace

    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show_results(results, None)
    return w


def test_double_clicking_a_data_tab_row_opens_its_spectrum(qtbot, page):
    series = page.tree.topLevelItem(0).child(0)
    page.tree.expandItem(series)
    point = series.child(1)
    page.tree.setCurrentItem(point)
    page.figures.setCurrentWidget(page.table_view)
    view = page.table_view
    _shown(qtbot, page)
    row = view.model.rows[1]
    index = view.model.index(1, 0)
    table = view.table
    table.scrollTo(index)
    at = table.visualRect(index).center()
    # a press first, as a user's double click starts (QTest's double click alone is not one
    # for an item view: it compares the row pressed)
    qtbot.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=at)
    assert page.tree.currentItem() is point  # one click opens nothing
    qtbot.mouseDClick(table.viewport(), Qt.MouseButton.LeftButton, pos=at)
    assert page.tree.currentItem().toolTip(0) == row["spectrum_id"]


def test_the_data_tabs_save_table_button_asks_and_writes_the_rows(qtbot, page, tmp_path,
                                                                 monkeypatch):
    series = page.tree.topLevelItem(0).child(0)
    page.tree.setCurrentItem(series)
    page.figures.setCurrentWidget(page.table_view)
    view = page.table_view
    asked = _ask(monkeypatch, "getSaveFileName", tmp_path / "rows", "CSV (*.csv)")
    view.save_button.click()
    (args, _kw), = asked.calls
    assert args[1] == "Save table" and args[2].endswith(".csv") and args[3] == "CSV (*.csv)"
    text = (tmp_path / "rows.csv").read_text(encoding="utf-8").splitlines()
    assert text[0].split(",") == view.model.columns and len(text) == len(view.model.rows) + 1


def test_the_column_filter_dialogs_all_none_and_clear(qtbot, page, monkeypatch):
    from hxex.gui.spectra_list import ColumnFilterDialog

    qtbot.waitUntil(lambda: bool(page.list.rows), timeout=30000)
    lst = page.list
    n = len(lst.rows)
    column = "time_label"
    labels = lst.values_of(column)
    header = lst.grid.horizontalHeader()
    section = lst.columns.index(column)

    def answered(*clicks, tick=None):
        """The dialog's buttons clicked in turn (then OK, unless one of them closes it)."""
        def exec_(dialog):
            for text in clicks:
                _button(dialog, text).click()
            if tick is not None:
                item = next(dialog.ticks.item(i) for i in range(dialog.ticks.count())
                            if dialog.ticks.item(i).data(Qt.ItemDataRole.UserRole) == tick)
                item.setCheckState(Qt.CheckState.Checked)
            if dialog.result() == 0:
                _button(dialog, "OK").click()
            return dialog.result()
        monkeypatch.setattr(ColumnFilterDialog, "exec", exec_)
        at = header.sectionViewportPosition(section) + 5
        from PySide6.QtCore import QPoint

        header.customContextMenuRequested.emit(QPoint(at, 5))

    answered("None", tick=labels[1])  # every value unticked, one ticked again
    assert {r[column] for r in lst.model.rows} == {labels[1]}
    answered("All")  # every value ticked: no filter
    assert lst.model.rowCount() == n and column not in lst.column_filters
    answered("None", tick=labels[0])
    assert lst.model.rowCount() < n
    answered("Clear")
    assert lst.model.rowCount() == n and column not in lst.column_filters


def test_the_figures_save_button_asks_and_saves(qtbot, tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    from hxex.gui.figure_view import FigurePanel

    panel = FigurePanel(render="matplotlib")
    qtbot.addWidget(panel)
    panel.resize(500, 400)
    panel.show()
    fig = Figure(figsize=(3, 2), dpi=100)
    fig.add_subplot().plot([1, 2], [1, 2])
    panel.save_name = "my_figure"  # as the workspaces name each figure
    panel.show_figure(fig)
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda *a, **k: warned.append(a) or QMessageBox.StandardButton.Ok))
    # no suffix typed: the format chosen in the dialog's list
    asked = _ask(monkeypatch, "getSaveFileName", tmp_path / "saved", "PDF (vector) (*.pdf)")
    assert panel.save_button.text() == "Save figure…"
    panel.save_button.click()
    (args, _kw), = asked.calls
    assert args[1] == "Save figure" and args[2] == "my_figure"
    assert "*.svg" in args[3] and "*.pdf" in args[3] and "*.png" in args[3]
    qtbot.waitUntil(lambda: not panel.saving, timeout=30000)
    assert not warned and (tmp_path / "saved.pdf").read_bytes().startswith(b"%PDF")


def test_the_export_dialogs_browse_chooses_the_folder(qtbot, results, tmp_path, monkeypatch):
    from hxex.gui.export_dialog import HXEXPRESS, ExportDialog, Target

    d = ExportDialog(results, HXEXPRESS, [Target(0)], tmp_path / "exports")
    qtbot.addWidget(d)
    asked = _ask(monkeypatch, "getExistingDirectory", tmp_path / "chosen")
    _button(d, "Browse…").click()
    (args, _kw), = asked.calls
    assert args[1] == "Export into" and args[2] == str(tmp_path / "exports")
    assert d.folder.text() == str(tmp_path / "chosen")
    _ask(monkeypatch, "getExistingDirectory", "")  # cancelled: kept
    _button(d, "Browse…").click()
    assert d.folder.text() == str(tmp_path / "chosen")


# --- the review panel -------------------------------------------------------------------------

def test_the_k_list_decides_and_says_when_the_spectra_differ(qtbot, window_on):  # noqa: F811
    from hxex.gui.user_tables import read_table

    window, project, open_ = window_on
    page = open_(project)
    series, leaf = _first_spectrum(page)
    page.tree.setCurrentItem(leaf)
    panel = page.review
    assert panel.k.currentData() is None  # the fit's choice
    at_one = panel.k.findData(1)
    qtbot.keyClick(panel.k, Qt.Key.Key_Down)  # as a user moves down the list
    assert panel.k.currentIndex() == at_one
    table = project.parent / "user_review_table.csv"

    def k_of(sid):
        cells = read_table(table)
        row = next(r for r in cells.rows if r[cells.columns.index("spectrum_id")] == sid)
        return row[cells.columns.index("review_K")]

    sid = leaf.toolTip(0)
    assert k_of(sid) == "1"
    # the series: one spectrum with a K given, the others not
    page.tree.setCurrentItem(series)
    panel = page.review
    assert panel.k.itemText(0) == "(differs between the spectra)"
    assert panel.k.currentIndex() == 0
    written = table.read_bytes()
    panel.k.activated.emit(0)  # choosing it decides nothing
    assert table.read_bytes() == written
    qtbot.keyClick(panel.k, Qt.Key.Key_Down)  # the fit's choice, for every spectrum
    assert panel.k.currentData() is None and k_of(sid) == ""


def test_add_the_bootstrap_runs_resume_in_the_run_workspace(qtbot, window_on, monkeypatch):  # noqa: F811
    """A K chosen without a bootstrap at it offers Resume; the button runs hxex resume."""
    import hxex.review_table as review_table
    from hxex.gui.session import WORKSPACES

    window, project, open_ = window_on
    page = open_(project)
    leaf = _second_spectrum(page)
    page.tree.setCurrentItem(leaf)
    sid = leaf.toolTip(0)
    monkeypatch.setattr(review_table, "missing_bootstraps", lambda results: {sid: 1})
    started = []
    monkeypatch.setattr(window.run_page.runner, "start",
                        lambda args, cwd: started.append((args, cwd)))
    panel = page.review
    assert panel.resume.isHidden()
    qtbot.keyClick(panel.k, Qt.Key.Key_Down)  # K = 1 chosen: no bootstrap at it
    assert panel.resume.isVisibleTo(panel) and "Resume" in panel.message.text()
    panel.resume.click()
    assert window.tabs.currentIndex() == WORKSPACES.index("run")
    assert started == [(["resume", project.name], project.parent)]


# --- the elution editor and the sequence view -------------------------------------------------

def test_fits_at_each_k_draws_the_new_windows_spectrum_at_every_k(qtbot, elution_project):  # noqa: F811
    from hxex.gui.results_workspace import ResultsWorkspace
    from test_window_edit import _read, _results

    project = elution_project
    spectra = _read(project)
    results = _results(spectra)
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, project)
    series = w.tree.topLevelItem(0).child(0)
    w.tree.expandItem(series)
    w.tree.setCurrentItem(series.child(0).child(1))
    tabs = [w.figures.tabText(i) for i in range(w.figures.count())]
    panel = w.panel("elution")
    w.figures.setCurrentIndex(tabs.index("elution"))
    qtbot.waitUntil(lambda: w._shown_traces is not None and not panel.drawing, timeout=30000)
    editor = w.window_editor
    editor.edit.setChecked(True)
    qtbot.waitUntil(lambda: editor._previewed is not None and editor.preview.k is not None
                    and not panel.drawing, timeout=60000)

    def new_windows():
        fig = panel.view.figure
        return [ax.get_title() for ax in fig.axes if ax.get_title().startswith("new window")]

    assert len(new_windows()) == 1
    fitted = sorted(editor.preview.results.fit.fits)
    with qtbot.waitSignal(panel.drawn, timeout=30000):
        editor.each_k.click()
    assert editor.each_k.isChecked()
    every = new_windows()
    assert len(every) == len(fitted) > 1
    assert all(any(f"K = {k}" in t for t in every) for k in fitted)
    with qtbot.waitSignal(panel.drawn, timeout=30000):
        editor.each_k.click()
    assert len(new_windows()) == 1
    editor.edit.setChecked(False)


def test_read_again_reads_the_fasta_files_again(qtbot, tmp_path):
    from hxex.gui.sequence_view import SequenceView
    from test_gui_sequence import _project

    project = _project(tmp_path, "")
    view = SequenceView()
    qtbot.addWidget(view)
    view.show_project(project)
    assert "3 sample(s), 2 distinct sequence(s)" in view.about.text()
    (tmp_path / "M1.fasta").write_text(">M1\nMKPLAAGDKE\n", encoding="utf-8")  # now as WT
    assert "2 distinct sequence(s)" in view.about.text()  # not read until asked
    _button(view, "Read again").click()
    assert "3 sample(s), 1 distinct sequence(s)" in view.about.text()
