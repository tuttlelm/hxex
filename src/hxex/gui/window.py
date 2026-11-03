# SPDX-License-Identifier: GPL-3.0-or-later
"""The main window: four workspaces (project, run, results, analysis), the session file, and a
pane of messages (docs/PLAN.md, "GUI").

Phase G0 builds the frame: the window, its menus, the session file, the messages pane, the SVG
figure view, and a first results view (a list of series and their figures). The project editor
(G1), running stages (G2), review (G3) and the analysis views (G4) come next; until then their
workspaces say how to do the same from the command line.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QByteArray, QObject, QSettings, Qt, QThreadPool, QUrl, Signal
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QDesktopServices, QFont, QKeySequence
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
    QVBoxLayout,
    QWidget,
)

from .results_workspace import ResultsWorkspace
from .run_workspace import RunWorkspace
from .session import SUFFIX, WORKSPACES, GuiSession, SessionError, load_session, save_session
from .tasks import Latest

log = logging.getLogger(__name__)

GUIDE_URL = "https://github.com/tuttlelm/hxex/blob/main/docs/guide/README.md"
RECENT = 8  # sessions remembered in the File menu
SERIES_KINDS = ("spectra", "uptake", "populations")
PEPTIDE_WIDTH = 170  # pixels: the widest the peptide column starts
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
    """The project file: its settings as a form (:mod:`hxex.gui.project_editor`), and the file
    itself, read-only."""

    file_changed = Signal()  # the project file changed (in the window or outside)

    def __init__(self, window: MainWindow):
        super().__init__()
        from .project_editor import ProjectEditor
        from .table_editor import TablesView

        self.path_label = QLabel("No project file open.")
        new_button = QPushButton("New project file…")
        new_button.clicked.connect(window.new_project_dialog)
        open_button = QPushButton("Open project file…")
        open_button.clicked.connect(window.open_project_dialog)
        top = QHBoxLayout()
        top.addWidget(self.path_label, 1)
        top.addWidget(new_button)
        top.addWidget(open_button)
        self.editor = ProjectEditor()
        self.editor.changed.connect(self._file_changed)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("monospace"))
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.tables = TablesView()
        self.views = QTabWidget()
        self.views.addTab(self.editor, "Settings")
        self.views.addTab(self.tables, "Tables")
        self.views.addTab(self.text, "File")
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.views, 1)
        self.path: Path | None = None

    def show_project(self, path: Path | None) -> None:
        self.path = path
        self.path_label.setText(str(path) if path is not None else "No project file open.")
        self.editor.open(path)
        self._file_changed()

    def _file_changed(self) -> None:
        self.tables.show_project(self.path, self.editor.values, self.editor.format)
        self.file_changed.emit()
        if self.path is None:
            self.text.setPlainText("")
            return
        try:
            self.text.setPlainText(self.path.read_text(encoding="utf-8-sig"))
        except OSError as exc:
            self.text.setPlainText(f"cannot read the project file: {exc}")


class AnalysisWorkspace(QWidget):
    """The analysis views (G4)."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(_note("The analysis views are not built yet: run hxex analyse "
                               "hxex.toml in a terminal and open the figures in its analysis "
                               "folder."))
        layout.addStretch()


def new_project(path: Path, fmt: str, name: str) -> Path:
    """Write a new project file with every default, as ``hxex init --format FMT --project NAME``
    does (a file chosen in a save dialog has been confirmed already, so one there is
    replaced)."""
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
        self.results_path: Path | None = None
        self._saved_text: str | None = None
        self._restoring = False  # results of the session just opened are still loading
        self._loading = Latest(self)
        self._pending_selection: dict = {}
        self.setWindowTitle("hxex")
        self.resize(1200, 800)

        # how figures are drawn: matplotlib (as saved files; the default, owner) or Qt's SVG
        # renderer, which ignores clipping (HISTORY 237)
        self.render = str(self.settings.value("render", "matplotlib"))
        if self.render not in ("svg", "matplotlib"):
            self.render = "matplotlib"
        self.project_page = ProjectWorkspace(self)
        self.run_page = RunWorkspace()
        self.run_page.results_changed.connect(self._results_changed)
        self.run_page.show_setting.connect(self.show_setting)
        self.project_page.file_changed.connect(self.run_page.show_device)
        self.project_page.file_changed.connect(self._project_file_changed)
        self._view_settings: dict | None = None  # the [figures] and [review] the results have
        self.results_page = ResultsWorkspace(self)
        self.analysis_page = AnalysisWorkspace()
        self.tabs = QTabWidget()
        for name, page in zip(WORKSPACES, (self.project_page, self.run_page, self.results_page,
                                           self.analysis_page), strict=True):
            self.tabs.addTab(page, name.capitalize())
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
        if logging.getLogger("hxex").getEffectiveLevel() > logging.INFO:
            logging.getLogger("hxex").setLevel(logging.INFO)

        self._menus()
        self.statusBar()
        self._saved_text = self._session_text()

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
                           ("matplotlib", "matplotlib (as saved files)")):
            action = QAction(text, self, checkable=True)
            action.setChecked(mode == self.render)
            action.triggered.connect(lambda checked=False, m=mode: self.set_render(m))
            group.addAction(action)
            draw.addAction(action)

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

    def set_render(self, render: str) -> None:
        """Draw the figures with Qt's SVG renderer or with matplotlib (remembered)."""
        self.render = render
        self.settings.setValue("render", render)
        self.results_page.set_render(render)

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
        replaces = path.suffix in (SUFFIX, ".toml")
        if replaces and not (self.run_page.may_close()
                             and self.project_page.tables.editor.may_leave()
                             and self._may_discard()):
            return False
        if path.suffix == ".toml":
            self._new_session_with(path)
        else:
            self.open_path(path)
        return True

    def _new_session_with(self, project: Path) -> None:
        """A new session (not yet saved) showing ``project``: what it shows now counts as
        unchanged, so only later changes ask to be saved."""
        self.session_path = None
        self._pending_selection = {}
        self.results_path = None
        self.results_page.show_results(None, None)
        self.open_project(project)
        self._mark_saved()
        self._title()

    def new_project_dialog(self) -> None:
        """Ask for the input format, a project name and where to save, then write the project
        file (as ``hxex init`` does) and open it."""
        if not (self.project_page.tables.editor.may_leave() and self._may_discard()):
            return
        dialog = NewProjectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save the new project file", "hxex.toml",
                                              "project files (*.toml)")
        if not path:
            return
        path = Path(path)
        if path.suffix != ".toml":
            path = path.with_name(path.name + ".toml")
        try:
            new_project(path, dialog.format.currentText(), dialog.name.text().strip())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Project file not written", str(exc))
            return
        self._new_session_with(path)
        self.tabs.setCurrentIndex(0)

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
        self.results_path = None
        self.results_page.show_results(None, None)
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
        from ..config import ProjectError, load_project

        try:
            project = load_project(path, analysis_only=True)
        except (ProjectError, OSError, ValueError) as exc:
            log.warning("%s: %s", path, exc)
            return
        bundle = project.output.bundle
        if bundle is not None and Path(bundle).exists():
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
        self.statusBar().showMessage(f"loading {path}…")
        self._loading.submit(lambda: load_results(path),
                             lambda results: self._results_loaded(results, path),
                             lambda message: self._results_failed(path, message))

    @property
    def loading(self) -> bool:
        return self._loading.running

    def _results_loaded(self, results, path: Path) -> None:
        self.results_path = path
        if path == self.project:
            self._view_settings = self._figure_and_review_settings()
        self.results_page.show_results(results, path)
        self.statusBar().showMessage(f"opened {path}", 5000)
        pending = dict(self._pending_selection)
        self.results_page.set_time_axis(pending.pop("time_axis", None))
        if pending:
            if not self.results_page.select(pending):
                log.warning("the results do not have the series the session showed (%s); "
                            "left out", ", ".join(f"{k} {v}" for k, v in pending.items()))
        self._pending_selection = {}
        if self._restoring:
            self._mark_saved()
        self.changed()

    def run_resume(self) -> None:
        """hxex resume in the Run workspace (a bootstrap for spectra shown at a K without one)."""
        self.tabs.setCurrentIndex(WORKSPACES.index("run"))
        self.run_page.run_command("resume")

    def show_setting(self, table: str) -> None:
        """The Project workspace's page of the table ``table`` (``"run.fit.backend"``)."""
        self.tabs.setCurrentIndex(WORKSPACES.index("project"))
        self.project_page.views.setCurrentIndex(0)
        self.project_page.editor.search.clear()
        self.project_page.editor.show_table(table)

    def _figure_and_review_settings(self) -> dict | None:
        """The project file's [figures] and [review] settings (the ones loaded results carry)."""
        if self.project is None:
            return None
        from ..project_edit import live_settings

        try:
            values, tables = live_settings(self.project.read_text(encoding="utf-8-sig"))
        except Exception:  # noqa: BLE001 - a file that does not read: nothing to compare
            return None
        keep = ("figures", "review")
        return {"values": {k: v for k, v in values.items() if k.split(".")[0] in keep},
                "tables": sorted(n for n in tables if n.split(".")[0] in keep)}

    def _project_file_changed(self) -> None:
        """The project file changed: when the results shown are the project's and its
        [figures] or [review] changed, load them again (they carry those settings), keeping the
        place in the tree. Other settings do not change results already made."""
        now = self._figure_and_review_settings()
        before, self._view_settings = self._view_settings, now
        if before is None or now == before or self.results_path != self.project \
                or self.project is None or self.loading:
            return
        log.info("the project's [figures] or [review] changed: the results shown again with them")
        self._pending_selection = {**self.results_page.selection(),
                                   **self.results_page.view_settings()}
        self.open_results(self.project)

    def _results_changed(self) -> None:
        """A run, resume or report from the Run workspace finished: show the new results (the
        project's, with its review applied)."""
        if self.project is not None:
            self._pending_selection = self.results_page.selection()
            self.open_results(self.project)

    def _results_failed(self, path: Path, message: str) -> None:
        self.statusBar().clearMessage()
        log.error("cannot open the results %s: %s", path, message)
        if self._restoring:
            self._mark_saved()

    # --- the session -----------------------------------------------------------------------

    def session(self) -> GuiSession:
        # results opened through the project file are the project's: the session names the
        # project only, so reopening it opens whatever results the project then has
        results = None if self.results_path == self.project else self.results_path
        return GuiSession(
            project=self.project, results=results,
            workspace=WORKSPACES[self.tabs.currentIndex()],
            selection={**self.results_page.selection(), **self.results_page.view_settings()},
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
        if not (self.project_page.tables.editor.may_leave() and self._may_discard()):
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
        """Save, Discard or Cancel: the question asked before a changed session is closed."""
        return QMessageBox.question(
            self, "Save the session?", "Save what the window shows as a session file, to "
            "reopen it later? (Settings and review decisions are in their own files already.)",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel)

    def _may_discard(self) -> bool:
        """Ask whether to save a session that changed; False when the user cancels."""
        if not self.modified() or (self.project is None and self.results_path is None):
            return True
        answer = self.ask_to_save()
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        return answer == QMessageBox.StandardButton.Discard or self.save_session()

    def closeEvent(self, event: QCloseEvent):  # noqa: N802 - Qt's name
        if self.run_page.may_close() and self.project_page.tables.editor.may_leave() \
                and self._may_discard():
            logging.getLogger("hxex").removeHandler(self.handler)
            # drawings and loads still running finish first (their results are dropped)
            from .tasks import drawing_pool

            for pool in (drawing_pool(), QThreadPool.globalInstance()):
                pool.clear()
                pool.waitForDone(10000)
            event.accept()
        else:
            event.ignore()
