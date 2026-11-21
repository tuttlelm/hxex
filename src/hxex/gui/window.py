# SPDX-License-Identifier: GPL-3.0-or-later
"""The main window: four workspaces (project, run, results, analysis), the session file, and a
pane of messages (docs/PLAN.md, "GUI").

The workspaces: the project editor and the user tables (G1), running the commands (G2), the
results with their review (G3), and the figures of ``hxex analyse`` (G4).
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from PySide6.QtCore import (
    QByteArray,
    QFileSystemWatcher,
    QObject,
    QSettings,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QKeySequence,
    QPalette,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .analysis_workspace import AnalysisWorkspace
from .file_view import ProjectFileView
from .project_state import ProjectState
from .results_workspace import ResultsWorkspace, left_out_or_none, sets_or_none
from .review import table_stamp
from .run_workspace import RunWorkspace
from .sequence_view import SequenceView
from .session import SUFFIX, WORKSPACES, GuiSession, SessionError, load_session, save_session
from .tasks import Latest
from .tips import tip

log = logging.getLogger(__name__)

GUIDE_URL = "https://github.com/tuttlelm/hxex/blob/main/docs/guide/README.md"
RECENT = 8  # sessions remembered in the File menu
# the tabs' tooltips (owner, 2026-11-13)
WORKSPACE_TIPS = {
    "project": "the project file: its settings, its tables (peptides, review, elution "
               "windows), and the sequence",
    "run": "run hxex's commands (check, run, report, analyse) and read their messages",
    "results": "every peptide, series, time point, and spectrum, with their figures and table "
               "rows",
    "analysis": "what hxex analyse makes from [analysis]: coverage, deuterium per peptide and "
                "residue, comparisons between samples, and their statistics",
}
PROJECT_TIPS = {
    "Settings": "the project file's settings, grouped as in the file",
    "Tables": "the tables the project file names (peptides or files, elution windows, "
              "review), to view and edit",
    "Sequence": "the protein sequence(s) as the analysis numbers them, with each residue's "
                "coverage",
    "File": "the project file as text: find (Ctrl+F) and edit it; Save writes it",
}
ABOUT_LOGO_WIDTH = 220  # pixels: the logo in the About box


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    return label


class _LogSignal(QObject):
    line = Signal(str, int)


class MessagesHandler(logging.Handler):
    """hxex's log messages into the window's messages pane (from any thread)."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.signal = _LogSignal()
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record):
        try:
            self.signal.line.emit(self.format(record), record.levelno)
        except RuntimeError:  # the window is gone
            pass


# --- the workspaces ---------------------------------------------------------------------------

class ProjectWorkspace(QWidget):
    """The project file: its settings as a form (:mod:`hxex.gui.project_editor`), its tables,
    its sequence, and the file itself as text (:mod:`hxex.gui.file_view`)."""

    file_changed = Signal()  # the project file changed (in the window or outside)

    def __init__(self, window: MainWindow):
        super().__init__()
        from .project_editor import ProjectEditor
        from .table_editor import TablesView

        self.window = window
        self.path_label = QLabel("No project file open.")
        new_button = QPushButton("New project file…")
        new_button.clicked.connect(window.new_project_dialog)
        open_button = QPushButton("Open project file…")
        open_button.clicked.connect(window.open_project_dialog)
        self.template_button = QPushButton("Save as template…")
        self.template_button.setToolTip(tip(
            "a copy of this project file to start another project from: every setting kept, "
            "the same data read, new results, no review"))
        self.template_button.clicked.connect(window.save_template_dialog)
        self.template_button.setEnabled(False)
        top = QHBoxLayout()
        top.addWidget(self.path_label, 1)
        top.addWidget(new_button)
        top.addWidget(open_button)
        top.addWidget(self.template_button)
        self.editor = ProjectEditor()
        self.editor.changed.connect(self._file_changed)
        self.text = ProjectFileView()
        self.text.write = window.write_project_text  # recorded for the editor's Undo
        self.text.modified_changed.connect(self._text_modified)
        self.tables = TablesView()
        self.tables.write_settings = window.write_settings  # recorded for the editor's Undo
        self.sequence = SequenceView()  # the protein sequence(s), as the analysis numbers them
        self.views = QTabWidget()
        for page, name in ((self.editor, "Settings"), (self.tables, "Tables"),
                           (self.sequence, "Sequence"), (self.text, "File")):
            self.views.setTabToolTip(self.views.addTab(page, name), tip(PROJECT_TIPS[name]))
        self._file_tab = self.views.indexOf(self.text)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.views, 1)
        self.path: Path | None = None

    def show_project(self, path: Path | None) -> None:
        self.path = path
        self.path_label.setText(str(path) if path is not None else "No project file open.")
        self.template_button.setEnabled(path is not None)
        self.editor.open(path)
        self._file_changed()

    def _file_changed(self) -> None:
        # the one parsed state of the project file every panel reads, renewed before any of
        # them looks (audit of 2026-11-15, Q-2)
        self.window.project_state = ProjectState.read(self.path)
        self.tables.show_project(self.path, self.editor.values, self.editor.format)
        self.sequence.show_project(self.window.project_state)
        self.file_changed.emit()
        self.text.show_file(self.path)

    def _text_modified(self, modified: bool) -> None:
        self.views.setTabText(self._file_tab, "File (unsaved)" if modified else "File")


def _only_the_table_named(before: dict, now: dict) -> bool:
    """Whether the only change between the [figures] and [review] settings ``before`` and
    ``now`` is ``[review] table`` naming a table."""
    if "review.table" in before["values"] or "review.table" not in now["values"]:
        return False
    rest = {k: v for k, v in now["values"].items() if k != "review.table"}
    tables = [t for t in now["tables"] if t != "review" or "review" in before["tables"]]
    return rest == before["values"] and tables == before["tables"]


def new_project(path: Path, fmt: str, name: str) -> Path:
    """Write a new project file with every default, as ``hxex init --format FMT --project NAME``
    does (the caller has asked about a file there already: it is replaced)."""
    from ..config import template

    if not name or ":" in name:
        raise ValueError(f"the project name must be text without ':', not {name!r}")
    path.write_text(template(fmt, name), encoding="utf-8")
    log.info("wrote %s (format %s): set the input paths in the Project workspace", path, fmt)
    return path


class NewProjectDialog(QDialog):
    """The input format and the project's name, for a new project file."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from ..config import FORMATS

        self.setWindowTitle("New project file")
        self.format = QComboBox()
        self.format.addItems(FORMATS)
        self.name = QLineEdit("my_project")
        form = QFormLayout()
        form.addRow("Input format", self.format)
        form.addRow("Project name", self.name)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(_note("The input formats are described in the guide's Input formats "
                               "page. Every setting starts at its default; the next step is the "
                               "input paths, under [input]."))
        layout.addLayout(form)
        layout.addWidget(buttons)


# --- the window -------------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """hxex's window."""

    def __init__(self, settings: QSettings | None = None):
        super().__init__()
        self.settings = settings or QSettings("hxex", "hxex-gui")
        self.session_path: Path | None = None
        self.project: Path | None = None
        # the project file as read last: one parsed state for every panel (Q-2), renewed by
        # the Project workspace whenever the file changes
        self.project_state = ProjectState()
        self._data_samples: tuple | None = None  # (state, names): the data's samples, read once
        self.results_path: Path | None = None
        self._saved_text: str | None = None
        self._restoring = False  # results of the session just opened are still loading
        self._loading = Latest(self)
        self._loading_path: Path | None = None  # the results being loaded (None: none)
        self._pending_selection: dict = {}
        self.setWindowTitle("hxex")
        self.resize(1200, 800)

        # how figures are drawn: matplotlib (its raster renderer, as for PNG; the default, owner)
        # or Qt's SVG renderer, which ignores clipping (HISTORY 237: the SVG's data is cut to
        # the panels instead, hxex.gui.svg_trim)
        # Ctrl+mouse wheel over any table, list or text box sets their font size (owner, 2026-11-13,
        # 2026-11-20)
        from .text_zoom import TextZoom

        self.text_zoom = TextZoom(self, self.settings)
        self.text_zoom.start()
        self.render = str(self.settings.value("render", "matplotlib"))
        if self.render not in ("svg", "matplotlib"):
            self.render = "matplotlib"
        self.project_page = ProjectWorkspace(self)
        self.run_page = RunWorkspace()
        self.run_page.project_values = lambda: self.project_state.values
        self.run_page.results_changed.connect(self._results_changed)
        self.run_page.show_setting.connect(self.show_setting)
        self.project_page.file_changed.connect(self.run_page.show_device)
        self.project_page.file_changed.connect(self._project_file_changed)
        self._view_settings: dict | None = None  # the [figures] and [review] the results have
        self._analysis_text: dict | None = None  # the [analysis] the tree's 'left out' follows
        # the review table as the results shown read it (gui.review.table_stamp)
        self.results_review_stamp: tuple | None = None
        self.naming_review_table = False  # the review panel is naming its table (D q7)
        # the review table, watched: saved from the Tables view or a spreadsheet, the results
        # are read again with it (owner, 2026-11-04); the review panel's own writes update the
        # stamp, so they are not read again
        self.review_watcher = QFileSystemWatcher(self)
        self._review_settle = QTimer(self)  # a save can report several changes: once settled
        self._review_settle.setSingleShot(True)
        self._review_settle.setInterval(300)
        self._review_settle.timeout.connect(self._review_table_changed)
        self.review_watcher.fileChanged.connect(lambda _path: self._review_settle.start())
        self.results_page = ResultsWorkspace(self)
        self.analysis_page = AnalysisWorkspace(self)
        # the sequence view's coverage follows the review (a rejected spectrum's peptide may
        # leave a residue uncovered)
        self.results_page.review.decided.connect(lambda _rows: self._show_coverage())
        # settings that list samples offer the samples of the results open
        self.project_page.editor.sample_names = self.sample_names
        self.tabs = QTabWidget()
        for name, page in zip(WORKSPACES, (self.project_page, self.run_page, self.results_page,
                                           self.analysis_page), strict=True):
            self.tabs.setTabToolTip(self.tabs.addTab(page, name.capitalize()),
                                    tip(WORKSPACE_TIPS[name]))
        self.setCentralWidget(self.tabs)

        self.messages = QPlainTextEdit()
        self.messages.setReadOnly(True)
        self.messages.setMaximumBlockCount(5000)
        dock = QDockWidget("Messages", self)
        dock.setObjectName("messages")
        dock.setWidget(self.messages)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [110], Qt.Orientation.Vertical)
        self.handler = MessagesHandler()
        self.handler.signal.line.connect(self._message)
        logging.getLogger("hxex").addHandler(self.handler)
        # the pane shows INFO lines: the hxex logger's level is lowered for the window's life
        # and given back when it closes (a Python session that opened a window kept the level
        # afterwards: audit of 2026-11-15, A-7)
        self._logger_level: int | None = None
        if logging.getLogger("hxex").getEffectiveLevel() > logging.INFO:
            self._logger_level = logging.getLogger("hxex").level
            logging.getLogger("hxex").setLevel(logging.INFO)

        self._menus()
        # a command running shows outside the Run tab too: the tab's title and colour, and a
        # line at the right of the status bar that opens the Run tab (owner, 2026-11-09)
        self.running = QToolButton()
        self.running.setAutoRaise(True)
        self.running.setToolTip(tip("a command is running: click to see it in the Run tab"))
        self.running.clicked.connect(
            lambda: self.tabs.setCurrentIndex(WORKSPACES.index("run")))
        self.running.hide()
        self.statusBar().addPermanentWidget(self.running)
        self.replacing = QLabel("the results shown are updated when the run finishes")
        self.replacing.setToolTip(tip("a run of the project whose results are shown "
                                      "is in progress; the window shows the new results when it "
                                      "ends"))
        self.replacing.hide()
        self.statusBar().addPermanentWidget(self.replacing)
        self.run_page.activity.connect(self._run_activity)
        self.run_page.ended.connect(self._run_ended)
        self.tabs.currentChanged.connect(self._tab_opened)
        self._saved_text = self._session_text()

    # --- a command running ------------------------------------------------------------------

    def _run_tab(self, note: str = "", colour: QColor | None = None) -> None:
        """The Run tab's title (``Running`` while a command runs, owner 2026-11-11; ``Run
        (stopped)`` after one ended unseen) and its text colour (None: the usual)."""
        i = WORKSPACES.index("run")
        title = "Running" if note == "running" else "Run" + (f" ({note})" if note else "")
        self.tabs.setTabText(i, title)
        self.tabs.tabBar().setTabTextColor(i, colour if colour is not None else QColor())

    def _run_activity(self, text: str) -> None:
        if text:
            self._run_tab("running", self.palette().color(QPalette.ColorRole.Highlight))
            self.running.setText(f"running: {text}")
        self.running.setVisible(bool(text))
        self._run_notice()

    def _run_notice(self) -> None:
        """While a run of the project shown runs, the Results panel and the status
        bar say that the results shown are updated when the run ends (audit of 2026-11-15, A open
        question 1: no blocking; the window reloads them then, with the bundle's stamp
        checked before any figure reads it)."""
        from .results_workspace import RUN_NOTICE
        from .run_workspace import CHANGES_RESULTS

        replacing = (self.run_page.running_command in CHANGES_RESULTS
                     and self.project is not None
                     and self.run_page.ran_on == self.project
                     and self.results_page.results is not None)
        self.results_page.set_notice(RUN_NOTICE if replacing else "")
        self.replacing.setVisible(replacing)

    def _run_ended(self, how: str) -> None:
        """Outside the Run tab, its title says how the command ended until it is opened."""
        self._run_notice()
        if self.tabs.currentIndex() == WORKSPACES.index("run"):
            self._run_tab()
            return
        word = how.split()[0] if how else ""
        self._run_tab(word if word in ("done", "stopped", "failed") else "ended")

    def _tab_opened(self, index: int) -> None:
        if index == WORKSPACES.index("run") and not self.run_page.runner.running:
            self._run_tab()

    # --- menus -----------------------------------------------------------------------------

    def _menus(self) -> None:
        file = self.menuBar().addMenu("&File")
        for text, key, slot in (("&New session", QKeySequence.StandardKey.New, self.new_session),
                                ("&Open…", QKeySequence.StandardKey.Open, self.open_dialog),
                                ("&Save session", QKeySequence.StandardKey.Save,
                                 self.save_session),
                                ("Save session &as…", QKeySequence.StandardKey.SaveAs,
                                 self.save_session_as)):
            action = QAction(text, self)
            action.setShortcut(key)
            action.triggered.connect(slot)
            file.addAction(action)
        self.recent_menu = file.addMenu("Open &recent")
        self._fill_recent()
        file.addSeparator()
        for text, slot in (("New project file…", self.new_project_dialog),
                           ("Save project file as template…", self.save_template_dialog),
                           ("Open project file…", self.open_project_dialog),
                           ("Open results…", self.open_results_dialog)):
            action = QAction(text, self)
            action.triggered.connect(slot)
            file.addAction(action)
        file.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file.addAction(quit_action)

        view = self.menuBar().addMenu("&View")
        draw = view.addMenu("Draw the figures with")
        group = QActionGroup(self)
        for mode, text in (("svg", "Qt's SVG renderer (vector)"),
                           ("matplotlib", "matplotlib (raster, as PNG)")):
            action = QAction(text, self, checkable=True)
            action.setChecked(mode == self.render)
            action.triggered.connect(lambda checked=False, m=mode: self.set_render(m))
            group.addAction(action)
            draw.addAction(action)

        self._appearance_menu(view)

        helpmenu = self.menuBar().addMenu("&Help")
        guide = QAction("User guide", self)
        guide.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(GUIDE_URL)))
        helpmenu.addAction(guide)
        about = QAction("About hxex", self)
        about.triggered.connect(self._about)
        helpmenu.addAction(about)

    def about_box(self) -> QMessageBox:
        """The About box: the logo (for the desktop's colour scheme), the version, and the
        licence."""
        from .. import version_text
        from .icon import logo_pixmap

        box = QMessageBox(self)
        box.setWindowTitle("About hxex")
        box.setIconPixmap(logo_pixmap(ABOUT_LOGO_WIDTH, ratio=self.devicePixelRatioF()))
        box.setText(f"{version_text()}\n\nMultimodal binomial-mixture analysis\nof HDX-MS "
                    "spectra.\n\nGPL-3.0-or-later.")
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.setStyleSheet("QLabel#qt_msgbox_label { min-width: 300px; }")  # the text, not wrapped
        return box

    def _appearance_menu(self, view) -> None:
        """View > Appearance: the widget style (hxex.gui.appearance; a light or dark palette
        is a start-up choice, ``hxex gui --dark`` / ``--light``)."""
        from . import appearance

        menu = view.addMenu("Appearance")
        menu.setToolTipsVisible(True)
        menu.addSection("Style")
        self._style_group = QActionGroup(self)
        current = appearance.current_style()
        choices = [(appearance.SYSTEM, f"System default ({appearance.default_style()})")]
        choices += [(name, name) for name in appearance.styles()]
        for name, text in choices:
            action = QAction(text, self, checkable=True)
            action.setData(name)
            action.setChecked(name.lower() == current.lower())
            action.setToolTip(tip("the widget style Qt chose at start" if name == appearance.SYSTEM
                                  else f"Qt's {name} widget style; remembered, "
                                       "applied at the next start too"))
            action.triggered.connect(lambda checked=False, n=name: self.set_style(n))
            self._style_group.addAction(action)
            menu.addAction(action)

    def set_style(self, name: str) -> None:
        """Use the widget style ``name`` (``system`` = Qt's choice at start; remembered)."""
        from . import appearance

        if appearance.apply_style(name):
            self.settings.setValue("style", name)
        current = appearance.current_style().lower()
        for action in self._style_group.actions():  # the mark follows what was applied
            action.setChecked(str(action.data()).lower() == current)

    def set_render(self, render: str) -> None:
        """Draw the figures with Qt's SVG renderer or with matplotlib (remembered)."""
        self.render = render
        self.settings.setValue("render", render)
        self.results_page.set_render(render)
        self.analysis_page.set_render(render)

    def _about(self) -> None:
        self.about_box().exec()

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        for path in self.recent():
            action = QAction(path, self)
            action.triggered.connect(lambda checked=False, p=path: self.open_asking(Path(p)))
            self.recent_menu.addAction(action)
        self.recent_menu.setEnabled(bool(self.recent()))

    def recent(self) -> list[str]:
        value = self.settings.value("recent", [])
        if isinstance(value, str):  # QSettings gives one item as a string
            value = [value]
        return [str(v) for v in value or []]

    def _remember(self, path: Path) -> None:
        paths = [str(path)] + [p for p in self.recent() if p != str(path)]
        self.settings.setValue("recent", paths[:RECENT])
        self._fill_recent()

    def _message(self, text: str, level: int) -> None:
        self.messages.appendPlainText(text)
        if level >= logging.WARNING:
            self.statusBar().showMessage(text, 10000)

    # --- opening ---------------------------------------------------------------------------

    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open", "", f"hxex sessions and project files (*{SUFFIX} *.toml);;"
                              "results bundles (*.zip);;all files (*)")
        if path:
            self.open_asking(Path(path))

    def open_asking(self, path: Path) -> bool:
        """Open ``path`` as the user asked: a project file or a session replaces what the
        session shows, so a session that changed is offered for saving first (results open
        within the session). A project file starts a new session. False when the user
        cancels."""
        if path.suffix == ".toml" and path.absolute() == self.project:
            # the project already open (Open recent, the Project tab's button): nothing to
            # replace, and no reason to stop its command (audit of 2026-11-15, A-12)
            self.statusBar().showMessage(f"{path} is the project open", 5000)
            return True
        replaces = path.suffix in (SUFFIX, ".toml")
        if replaces and not self._may_replace():
            return False
        if path.suffix == ".toml":
            self._new_session_with(path)
        else:
            self.open_path(path)
        return True

    def _may_replace(self) -> bool:
        """Before what the window shows is replaced (another project or session, a new one):
        unsaved table edits, then the session, then a command running (asked last: a command
        stopped before a cancelled question was gone for nothing; audit of 2026-11-15, A-2).
        False when the user cancels."""
        return (self.project_page.tables.editor.may_leave()
                and self.project_page.text.may_leave() and self._may_discard()
                and self.run_page.may_close())

    def _drop_load(self) -> None:
        """Results still loading are not wanted any more: another project's results arrived
        later and were shown under it (audit of 2026-11-15, A-1)."""
        self._loading.drop()
        self._loading_path = None
        self.statusBar().clearMessage()

    def _new_session_with(self, project: Path) -> None:
        """A new session (not yet saved) showing ``project``: what it shows now counts as
        unchanged, so only later changes ask to be saved."""
        self.session_path = None
        self._pending_selection = {}
        self._drop_load()
        self.results_path = None
        self.project_page.sequence.set_per_line(None)  # a new session: its default (E-24)
        self.results_page.show_results(None, None)
        self._show_coverage()
        self.analysis_page.results_changed()
        self.open_project(project)
        self._mark_saved()
        self._title()

    def new_project_dialog(self) -> None:
        """Ask for the input format, a project name and where to save, then write the project
        file (as ``hxex init`` does) and open it."""
        if not self._may_replace():
            return
        dialog = NewProjectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        # the dialog's own overwrite question would be about the name as typed, without the
        # .toml added below: asked here, about the file written (audit of 2026-11-15, B-4)
        path, _ = QFileDialog.getSaveFileName(self, "Save the new project file", "hxex.toml",
                                              "project files (*.toml)",
                                              options=QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return
        path = Path(path)
        if path.suffix != ".toml":
            path = path.with_name(path.name + ".toml")
        if path.exists() and not self.ask_replace_file(path):
            return
        try:
            new_project(path, dialog.format.currentText(), dialog.name.text().strip())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Project file not written", str(exc))
            return
        self._new_session_with(path)
        self.tabs.setCurrentIndex(0)

    def save_template_dialog(self) -> bool:
        """Copy the project file open as the start of another project (as the guide's "A
        project from another one" says): every setting and comment kept, its paths rewritten to
        read the same data from the copy's folder, a new results folder, and no review table,
        review exclusions, overwrite, or keep_previous. The copy is then opened. False when
        nothing was written."""
        from ..config import ProjectError
        from ..project_edit import save_template

        if self.project is None:
            QMessageBox.information(self, "No project file open",
                                    "Open a project file first: the template is made from it.")
            return False
        # the copy is made from the file on disk: text typed in the File tab first
        if not self.project_page.text.may_leave():
            return False
        target = self.ask_template_path()
        if target is None:
            return False
        if target.suffix != ".toml":
            target = target.with_name(target.name + ".toml")
        if target.absolute() == self.project:
            self.tell("Template not written",
                      "Choose another name: this is the project file itself.")
            return False
        if target.exists() and not self.ask_replace_file(target):
            return False
        bundle = self.ask_template_results(target)
        if bundle is None:
            return False
        try:
            said, problem = save_template(self.project, target, bundle)
        except (ProjectError, OSError) as exc:
            self.tell("Template not written", str(exc))
            return False
        log.info("wrote %s from %s:", target, self.project.name)
        for line in said:
            log.info("  %s", line)
        if problem:
            log.warning("%s: %s", target.name, problem)
        if self._may_replace():
            self._new_session_with(target)
            self.tabs.setCurrentIndex(0)
        return True

    def tell(self, title: str, text: str) -> None:
        """A warning the user must read (replaced in tests)."""
        QMessageBox.warning(self, title, text)

    def ask_template_path(self) -> Path | None:
        """Where the template goes (asked; replaced in tests)."""
        start = self.project.with_name(f"{self.project.stem}_copy.toml")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the project file as a template", str(start), "project files (*.toml)",
            options=QFileDialog.Option.DontConfirmOverwrite)
        return Path(path) if path else None

    def ask_template_results(self, target: Path) -> str | None:
        """The new project's results folder (asked; replaced in tests)."""
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self, "Results of the new project",
            f"The folder for {target.name}'s results, beside it (it must not exist yet, so "
            "neither project replaces the other's results):",
            QLineEdit.EchoMode.Normal, f"results_{target.stem}")
        return name.strip() if ok and name.strip() else None

    def ask_replace_file(self, path: Path) -> bool:
        """Whether to write over the file at ``path`` (asked before a new project file or a
        new table replaces one; replaced in tests)."""
        return QMessageBox.question(
            self, "Replace the file?",
            f"{path.name} exists in {path.parent}. Replace it with a new project file? Its "
            "settings and comments are lost (this cannot be undone).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    @contextlib.contextmanager
    def recording(self, what: str):
        """A block that writes the project file open through a library function
        (:func:`hxex.gui.review.make_review_table`, :func:`hxex.gui.window_edit.save_window`):
        recorded for the Project editor's Undo as ``what``, the file not watched meanwhile
        (audit of 2026-11-15, Q-1)."""
        editor = self.project_page.editor
        if self.project is None or editor.path != self.project:
            yield None
            return
        with editor.own_write() as step:
            step.what = what
            yield step
        editor._after_change()  # shown now (the file is not watched while written)

    def write_settings(self, changes, what: str) -> list:
        """Change settings of the project file open as the Project editor does (recorded for
        its Undo, the file not watched while written; audit of 2026-11-15, Q-1): the window's
        one route for every setting it writes. ``what`` names the change for Undo."""
        from ..project_edit import set_settings

        editor = self.project_page.editor
        if self.project is None or editor.path != self.project:
            return set_settings(self.project, changes, unfinished=True)
        with editor.own_write() as step:
            done = set_settings(self.project, changes, unfinished=True)
            step.what = what
        editor._after_change()  # shown now (the file is not watched while written)
        return done

    def write_project_text(self, text: str) -> str | None:
        """Write ``text`` as the whole project file open (the File tab's Save), recorded for
        the Project editor's Undo; the file's problem, if any (ProjectError when the text is
        not valid TOML: nothing written)."""
        from ..project_edit import write_text

        with self.recording("the project file edited as text"):
            return write_text(self.project, text)

    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project file", "",
                                              "project files (*.toml);;all files (*)")
        if path:
            self.open_asking(Path(path))

    def open_results_dialog(self) -> None:
        start = str(self.project.parent) if self.project else ""
        path = QFileDialog.getExistingDirectory(self, "Open results (a bundle folder)", start)
        if path:
            self.open_results(Path(path))

    def open_path(self, path: Path) -> None:
        """A session file, a project file, or a results bundle (folder or zip)."""
        if path.suffix == SUFFIX:
            self.open_session(path)
        elif path.suffix == ".toml":
            self.open_project(path)
        else:
            self.open_results(path)

    def open_session(self, path: Path) -> None:
        try:
            session = load_session(path)
        except (SessionError, OSError) as exc:
            QMessageBox.warning(self, "Session not opened", str(exc))
            return
        for line in session.missing():
            log.warning("%s: %s; left out", path.name, line)
        self.session_path = path.absolute()
        self._remember(self.session_path)
        self._apply(session)
        self._mark_saved()
        self._title()

    def _apply(self, session: GuiSession) -> None:
        layout = session.layout
        if "geometry" in layout:
            self.restoreGeometry(QByteArray.fromBase64(layout["geometry"].encode()))
        if "state" in layout:
            self.restoreState(QByteArray.fromBase64(layout["state"].encode()))
        project = session.project if session.project and session.project.exists() else None
        results = session.results if session.results and session.results.exists() else None
        self._pending_selection = dict(session.selection)
        self.project_page.sequence.set_per_line(
            self._pending_selection.pop("residues_per_line", None))
        self._drop_load()
        self.results_path = None
        self.results_page.show_results(None, None)
        self._show_coverage()
        self.analysis_page.results_changed()
        if project is not None:
            self.open_project(project, load_results=results is None)
        else:
            self._show_project(None)
        if results is not None:
            self.open_results(results)
        self.tabs.setCurrentIndex(WORKSPACES.index(session.workspace))

    def open_project(self, path: Path | None, load_results: bool = True) -> None:
        """Show a project file, and open its results when it names a bundle that exists."""
        self._show_project(path)
        if path is None or not load_results:
            return
        state = self.project_state
        if state.loaded is None:  # said in the log when the state was read
            return
        bundle = state.bundle
        if bundle is not None and bundle.exists():
            self.open_results(path)  # the project's results, with its review applied
        else:
            log.info("no results yet (hxex run %s, in %s, makes them)", path.name, path.parent)

    def _show_project(self, path: Path | None) -> None:
        new = path.absolute() if path is not None else None
        if new != self.project:
            # the messages were about the project shown before: start afresh, saying which
            # project the next ones are about (two projects may both be called hxex.toml)
            self.messages.clear()
            if new is not None:
                log.info("project file %s", new)
        self.project = new
        self.project_page.show_project(self.project)
        self.run_page.show_project(self.project)
        self.changed()

    def open_results(self, path: Path) -> None:
        """Load results in the background: a bundle (folder or zip), or a project file's own
        (with its review applied, as ``hxex report`` uses them)."""
        from ..report.notebook import load_results

        path = path.absolute()
        project = self.project if path == self.project else None
        self._loading_path = path
        self.statusBar().showMessage(f"loading {path}…")
        # the project file's settings as the load starts: a change made while the results
        # load is seen when they arrive (audit of 2026-11-15, A-14 and D-6)
        before = (self._figure_and_review_settings() if project is not None else None,
                  self._figure_and_review_settings(("analysis",)) if project is not None
                  else None)
        state = self.project_state if project is not None else ProjectState()

        def load():
            stamp = table_stamp(project)  # before reading: a change while it reads shows later
            # a review K the screen skipped is kept, the spectrum shown at the fit's K (the
            # review panel says so; the commands refuse it: audit of 2026-11-15, D-9)
            results = load_results(path, unfitted="keep")
            analysis = state.analysis  # the state as the load started (immutable)
            left = left_out_or_none(results, analysis)
            sets_or_none(results, analysis)  # the sets' K, held for the tree (cached)
            return results, analysis, left, stamp

        self._loading.submit(load, lambda loaded: self._results_loaded(*loaded, path, before),
                             lambda message: self._results_failed(path, message))

    @property
    def loading(self) -> bool:
        return self._loading.running

    def _release(self) -> None:
        """Let go of the results and figures the closing window holds. A closed window's
        Python side stays alive in its own cycles (its workspaces, bound methods and callback
        closures; Qt's parent-child ownership hides some references from the garbage
        collector), so each closed window kept its whole results and four figures, 60 to 300
        MB, in a process that makes several windows, such as a test session (audit of
        2026-11-15, M-5). The ``hxex gui`` process makes one window and ends with it."""
        self._loading.drop()  # its done/failed closures held the window
        self._pending_selection = {}
        self.results_page.release()
        self.analysis_page.release()
        self.project_page.editor.sample_names = None  # a bound method of this window
        self._data_samples = None
        self.project_page.sequence.show_coverage(None)
        self._view_settings = self._analysis_text = None
        self.project_state = ProjectState()

    def _results_loaded(self, results, analysis, left, stamp, path: Path,
                        before: tuple = (None, None)) -> None:
        if path != self._loading_path:  # not the results asked for last (A-1)
            return
        self._loading_path = None
        self.results_path = path
        self.results_review_stamp = stamp
        self._watch_review_table()
        again = False  # read once more: the review table or the project file changed meanwhile
        if path == self.project:
            self._view_settings = self._figure_and_review_settings()
            self._analysis_text = self._figure_and_review_settings(("analysis",))
            view_before, analysis_before = before
            if (view_before is not None and self._view_settings != view_before) \
                    or (stamp is not None and table_stamp(self.project) != stamp):
                again = True
            elif analysis_before is not None and self._analysis_text != analysis_before:
                analysis = self.project_state.analysis  # the tree's 'left out' column
                left = left_out_or_none(results, analysis)
        self.results_page.show_results(results, path, left, analysis)
        self.analysis_page.results_changed()
        self._show_coverage()
        self.statusBar().showMessage(f"opened {path}", 5000)
        pending = dict(self._pending_selection)
        self.results_page.set_time_axis(pending.pop("time_axis", None))
        self.results_page.set_k_residuals(pending.pop("k_residuals", False))
        if pending:
            if not self.results_page.select(pending):
                log.warning("the results do not have the series the session showed (%s); "
                            "left out", ", ".join(f"{k} {v}" for k, v in pending.items()))
        self._pending_selection = {}
        if self._restoring:
            self._mark_saved()
        self.changed()
        if again:
            log.info("the review table or the project file changed while the results were "
                     "being read: they are read again")
            self._pending_selection = {**self.results_page.selection(),
                                       **self.results_page.view_settings()}
            self.open_results(self.project)

    def _show_coverage(self) -> None:
        """The coverage of the project's results under its sequences (none for results opened
        on their own, or none open)."""
        from .sequence_view import coverage_of

        results = self.results_page.results
        mine = results is not None and self.project_results
        self.project_page.sequence.show_coverage(coverage_of(results) if mine else None)

    def sample_names(self) -> list[str]:
        """The samples to offer where a setting names samples (``[analysis] samples`` and
        ``reference``): those of the results open, named as the analysis settings name them;
        before there are results, ``[input] samples`` when the project file sets it, else the
        samples its data hold (:func:`hxex.config.data_samples`), so a new project's order and
        reference can be set from the start (owner, 2026-11-18)."""
        results = self.results_page.results
        if results is not None:
            from ..analysis import AnalysisConfig
            from ..report import analysis_references

            try:
                names = analysis_references(results, AnalysisConfig(reference="all"))
            except ValueError:  # results that do not list their samples
                names = []
            if names:
                return names
        state = self.project_state
        if self._data_samples is not None and self._data_samples[0] is state:
            return list(self._data_samples[1])  # read once per reading of the project file
        project = state.loaded
        names = []
        if project is not None and project.input is not None:
            if project.input.samples:
                names = list(project.input.samples)
            else:
                from ..config import data_samples

                names = data_samples(project)
        self._data_samples = (state, names)
        return list(names)

    def run_resume(self) -> None:
        """hxex run in the Run workspace, bringing the results up to date (a bootstrap for
        spectra shown at a K without one, a K the review chose that was not fitted)."""
        self.tabs.setCurrentIndex(WORKSPACES.index("run"))
        self.run_page.run_command("run")

    def show_setting(self, table: str) -> None:
        """The Project workspace's page of the table ``table`` (``"run.fit.backend"``)."""
        self.tabs.setCurrentIndex(WORKSPACES.index("project"))
        self.project_page.views.setCurrentIndex(0)
        self.project_page.editor.search.clear()
        self.project_page.editor.show_table(table)

    def show_peptide(self, sample: str, start: int, end: int, peptide: str = "") -> bool:
        """The Results workspace with the peptide ``start``-``end`` of ``sample`` selected (its
        series' pages: the spectra); False, staying where it is, when the tree lacks it."""
        if not self.results_page.select_peptide(sample, start, end, peptide):
            return False
        self.tabs.setCurrentIndex(WORKSPACES.index("results"))
        return True

    def fresh_project_state(self) -> ProjectState:
        """The project state, read again first when the file changed on disk since it was read
        (its size or modification time differ): for a panel answering an action before the
        file watcher's settle time has passed (the state is renewed when it fires too)."""
        state = self.project_state
        if state.path != self.project:
            state = ProjectState.read(self.project)
        else:
            state = state.fresh()
        self.project_state = state
        return state

    @property
    def project_results(self) -> bool:
        """Whether the results shown are the project file's own (opened through it, with its
        review applied): the project's review table takes the decisions, its [analysis]
        applies, and a change of the file reloads them (audit of 2026-11-15, Q-3)."""
        return self.project is not None and self.results_path == self.project

    def _figure_and_review_settings(self, keep=("figures", "review")) -> dict | None:
        """The project file's [figures] and [review] settings (the ones loaded results carry),
        or those of the tables ``keep``; None without a project, or when the file does not
        read (nothing to compare)."""
        if self.project is None:
            return None
        state = self.fresh_project_state()
        return state.of_tables(*keep) if state.parsed else None

    def watch_review_table(self) -> None:
        """Watch the review table of the project whose results are shown (none otherwise):
        called when the results are loaded, and by the review panel when its first decision
        makes the table (the results are not read again for that: D open question 7)."""
        self._watch_review_table()

    def _watch_review_table(self) -> None:
        """Watch the review table of the project whose results are shown (none otherwise)."""
        if self.review_watcher.files():
            self.review_watcher.removePaths(self.review_watcher.files())
        stamp = self.results_review_stamp
        if self.project_results and stamp and stamp[1] is not None:
            self._watch(stamp[0])

    def _watch(self, path: str) -> None:
        """Watch the review table at ``path``; a watch the system refuses (its limit of
        watches reached, say) is said, as a table saved elsewhere would then go unseen
        (audit of 2026-11-15, A-15)."""
        if not self.review_watcher.addPath(path):
            log.warning("the review table %s cannot be watched (the system refused the watch: "
                        "its limit of file watches may be reached): a change saved to it from "
                        "elsewhere is not seen until the results are opened again", path)

    def _review_table_changed(self) -> None:
        """The review table changed on disk: unless the review panel wrote it (its decisions are
        in the results already), the results are read again with it, at the same place."""
        stamp = table_stamp(self.project)
        if stamp and stamp[1] is not None and stamp[0] not in self.review_watcher.files():
            self._watch(stamp[0])  # a file replaced is no longer watched
        if not self.project_results or self.loading or stamp == self.results_review_stamp:
            return
        log.info("the review table changed: the results are read again with it")
        self.results_page.review.commit_note()  # a note being typed is kept
        self._pending_selection = {**self.results_page.selection(),
                                   **self.results_page.view_settings()}
        self.open_results(self.project)

    def _project_file_changed(self) -> None:
        """The project file changed: when the results shown are the project's and its
        [figures] or [review] changed, load them again (they carry those settings), keeping the
        place in the tree; when its [analysis] changed, the tree's 'left out' column is worked
        out again. Other settings do not change results already made."""
        if self.project_results and not self.loading:
            # the table it names, or the table's name, may have changed: the windows saved
            # since the run follow
            self.results_page.refresh_pending()
        analysis = self._figure_and_review_settings(("analysis",))
        analysis_before, self._analysis_text = self._analysis_text, analysis
        now = self._figure_and_review_settings()
        if (analysis_before is not None and analysis != analysis_before
                and self.project_results and now == self._view_settings and not self.loading):
            self.results_page.analysis_changed(self.project_state.analysis)
        before, self._view_settings = self._view_settings, now
        if before is None or now == before or not self.project_results or self.loading:
            return
        if self.naming_review_table and _only_the_table_named(before, now):
            # the review panel's first decision named its table: the decision is in the
            # results shown already (audit of 2026-11-15, D open question 7)
            return
        log.info("the project's [figures] or [review] changed: the results shown again with them")
        self.results_page.review.commit_note()  # a note being typed is kept
        self._pending_selection = {**self.results_page.selection(),
                                   **self.results_page.view_settings()}
        self.open_results(self.project)

    def _results_changed(self) -> None:
        """A run or report from the Run workspace finished: show the new results (the
        project's, with its review applied)."""
        if self.project is not None:
            self._pending_selection = self.results_page.selection()
            self.open_results(self.project)

    def _results_failed(self, path: Path, message: str) -> None:
        if path != self._loading_path:
            return
        self._loading_path = None
        self.statusBar().clearMessage()
        self.analysis_page.load_failed()
        log.error("cannot open the results %s: %s", path, message)
        if self._restoring:
            self._mark_saved()

    # --- the session -----------------------------------------------------------------------

    def session(self) -> GuiSession:
        # results opened through the project file are the project's: the session names the
        # project only, so reopening it opens whatever results the project then has
        results = None if self.project_results else self.results_path
        return GuiSession(
            project=self.project, results=results,
            workspace=WORKSPACES[self.tabs.currentIndex()],
            selection={**self.results_page.selection(), **self.results_page.view_settings(),
                       **self.project_page.sequence.view_settings()},
            layout={"geometry": bytes(self.saveGeometry().toBase64()).decode(),
                    "state": bytes(self.saveState().toBase64()).decode()})

    def _session_text(self) -> str:
        from .session import session_text

        session = self.session()
        session.layout = {}  # moving the window is not a change worth asking about
        base = self.session_path.parent if self.session_path else Path.cwd()
        return session_text(session, base)

    def changed(self) -> None:
        self._title()

    def _mark_saved(self) -> None:
        """What the window shows now is the session as opened or saved; results still loading
        for it count too, once they are in."""
        self._saved_text = self._session_text()
        self._restoring = self.loading

    def modified(self) -> bool:
        return self._session_text() != self._saved_text

    def _title(self) -> None:
        name = self.session_path.name if self.session_path else "new session"
        self.setWindowTitle(f"{name} – hxex")

    def new_session(self) -> None:
        if not self._may_replace():
            return
        self.session_path = None
        self._apply(GuiSession())
        self._mark_saved()
        self._title()

    def save_session(self) -> bool:
        if self.session_path is None:
            return self.save_session_as()
        save_session(self.session(), self.session_path)
        self._saved_text = self._session_text()
        self.statusBar().showMessage(f"session saved to {self.session_path}", 5000)
        return True

    def save_session_as(self) -> bool:
        start = (self.project.with_suffix(SUFFIX) if self.project else Path("hxex" + SUFFIX))
        path, _ = QFileDialog.getSaveFileName(self, "Save session", str(start),
                                              f"hxex sessions (*{SUFFIX})")
        if not path:
            return False
        path = Path(path)
        if path.suffix != SUFFIX:
            path = path.with_name(path.name + SUFFIX)
        self.session_path = path.absolute()
        self._remember(self.session_path)
        self._title()
        return self.save_session()

    def ask_to_save(self) -> QMessageBox.StandardButton:
        """Save, Discard or Cancel: the question asked before a changed session is closed, or
        a project's that was never saved (:meth:`never_saved`, the view unchanged)."""
        text = ("Save a session for this project, to reopen it from File > Open recent? "
                if self.never_saved() and not self.modified() else
                "Save what the window shows as a session file, to reopen it later? ")
        return QMessageBox.question(
            self, "Save the session?", text + "(Settings and review decisions are in their own "
            "files already.)",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel)

    def never_saved(self) -> bool:
        """A project file open in a session never saved: closing it would leave it off the
        Open recent list, which lists sessions (owner, 2026-11-13: ask even when nothing in
        the view changed)."""
        return self.project is not None and self.session_path is None

    def _may_discard(self) -> bool:
        """Ask whether to save a session that changed, or a project's never saved; False when
        the user cancels."""
        if self.project is None and self.results_path is None:
            return True
        if not self.modified() and not self.never_saved():
            return True
        answer = self.ask_to_save()
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        return answer == QMessageBox.StandardButton.Discard or self.save_session()

    def closeEvent(self, event: QCloseEvent):  # noqa: N802 - Qt's name
        if self._may_replace():
            logging.getLogger("hxex").removeHandler(self.handler)
            if self._logger_level is not None:  # the level as it was before the window (A-7)
                logging.getLogger("hxex").setLevel(self._logger_level)
                self._logger_level = None
            self.text_zoom.stop()
            # work still running is dropped (nothing arrives in the closed window; what has
            # not started never does) and what is running finishes first, with a limit;
            # automatic garbage collection comes back only once the window is gone
            # (hxex.gui.launch, the tests' conftest), not here with the window and its
            # queued results alive (audit of 2026-11-15, A-6)
            from .tasks import drawing_pool

            self._drop_load()
            self.results_page.drop_background()
            self.analysis_page.cancel()
            for pool in (drawing_pool(), QThreadPool.globalInstance()):
                pool.clear()
                pool.waitForDone(10000)
            try:
                self._release()
            except Exception:  # noqa: BLE001 - the window closes all the same
                log.exception("the closing window could not let go of its results")
            event.accept()
        else:
            event.ignore()
