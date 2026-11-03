# SPDX-License-Identifier: GPL-3.0-or-later
"""The main window: four workspaces (project, run, results, analysis), the session file, and a
pane of messages (docs/PLAN.md, "GUI").

Phase G0 builds the frame: the window, its menus, the session file, the messages pane, the SVG
figure view, and a first results view (a list of series and their figures). The project editor
(G1), running stages (G2), review (G3) and the analysis views (G4) come next; until then their
workspaces say how to do the same from the command line.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from PySide6.QtCore import QByteArray, QObject, QSettings, Qt, QThreadPool, QUrl, Signal
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QFont, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .figure_view import FigurePanel
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
    """The project file. G0 shows it; the settings editor comes in G1."""

    def __init__(self, window: MainWindow):
        super().__init__()
        self.path_label = QLabel("No project file open.")
        open_button = QPushButton("Open project file…")
        open_button.clicked.connect(window.open_project_dialog)
        top = QHBoxLayout()
        top.addWidget(self.path_label, 1)
        top.addWidget(open_button)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("monospace"))
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(_note("Changing settings in this window is not built yet: edit the "
                               "project file in a text editor, or with hxex set in a terminal "
                               "(hxex set hxex.toml run.controls.d2o_fraction 0.85)."))
        layout.addWidget(self.text, 1)

    def show_project(self, path: Path | None) -> None:
        if path is None:
            self.path_label.setText("No project file open.")
            self.text.setPlainText("")
            return
        self.path_label.setText(str(path))
        try:
            self.text.setPlainText(path.read_text(encoding="utf-8"))
        except OSError as exc:
            self.text.setPlainText(f"cannot read the project file: {exc}")


class RunWorkspace(QWidget):
    """Running stages. G2 runs them from here; until then the commands to type."""

    def __init__(self):
        super().__init__()
        self.commands = _note("")
        layout = QVBoxLayout(self)
        layout.addWidget(_note("Running stages from this window is not built yet. In a "
                               "terminal, in the project's folder:"))
        layout.addWidget(self.commands)
        layout.addStretch()
        self.show_project(None)

    def show_project(self, path: Path | None) -> None:
        name = path.name if path is not None else "hxex.toml"
        self.commands.setText("\n".join(f"hxex {c} {name}" for c in
                                        ("check", "run", "report", "analyse")))
        self.commands.setFont(QFont("monospace"))


class AnalysisWorkspace(QWidget):
    """The analysis views (G4)."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(_note("The analysis views are not built yet: run hxex analyse "
                               "hxex.toml in a terminal and open the figures in its analysis "
                               "folder."))
        layout.addStretch()


class ResultsWorkspace(QWidget):
    """A first results view: the series of the open results and their figures."""

    COLUMNS = (("sample", "sample"), ("peptide_range", "residues"), ("peptide", "peptide"),
               ("charge", "charge"), ("n_spectra", "spectra"), ("n_flagged", "flagged"),
               ("n_rejected", "rejected"), ("n_multi", "K > 1"))

    def __init__(self, window: MainWindow):
        super().__init__()
        self.window = window
        self.results = None
        self.data_ids: list[int] = []
        self.path_label = QLabel("No results open.")
        open_button = QPushButton("Open results…")
        open_button.setToolTip("a results bundle: a folder, or a .zip")
        open_button.clicked.connect(window.open_results_dialog)
        top = QHBoxLayout()
        top.addWidget(self.path_label, 1)
        top.addWidget(open_button)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label in self.COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._selected)

        self.figures = QTabWidget()
        self.panels: dict[str, FigurePanel] = {}
        for kind in SERIES_KINDS:
            panel = FigurePanel()
            panel.view.clicked.connect(lambda gid: window.statusBar().showMessage(gid, 5000))
            self.panels[kind] = panel
            self.figures.addTab(panel, kind)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.table)
        split.addWidget(self.figures)
        split.setStretchFactor(1, 1)
        split.setSizes([560, 900])  # the list wide enough for its columns; the figures the rest
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(split, 1)

    def show_results(self, results, path: Path | None) -> None:
        from ..report.tables import series_table

        self.results = results
        self.table.setSortingEnabled(False)
        self.table.clearContents()
        for panel in self.panels.values():
            panel.clear()
        if results is None:
            self.path_label.setText("No results open.")
            self.table.setRowCount(0)
            self.data_ids = []
            return
        self.path_label.setText(str(path))
        rows = series_table(results)
        self.data_ids = [int(d) for d in rows["data_id"]]
        self.table.setRowCount(len(self.data_ids))
        for r, data_id in enumerate(self.data_ids):
            for c, (key, _) in enumerate(self.COLUMNS):
                value = rows[key][r]
                item = QTableWidgetItem()
                if key in ("charge",) or key.startswith("n_"):
                    item.setData(Qt.ItemDataRole.DisplayRole, int(value))
                else:
                    item.setText(str(value))
                    item.setToolTip(str(value))
                item.setData(Qt.ItemDataRole.UserRole, data_id)
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        peptide = 2  # long peptides would push the counts out of view (the tooltip has it all)
        self.table.setColumnWidth(peptide, min(self.table.columnWidth(peptide), PEPTIDE_WIDTH))

    def selection(self) -> dict:
        """The selected series as sample, residues and charge (they outlast a rerun, which can
        renumber the series)."""
        row = self.table.currentRow()
        if row < 0 or not self.table.selectedItems():
            return {}
        return {"sample": self.table.item(row, 0).text(),
                "peptide_range": self.table.item(row, 1).text(),
                "charge": int(self.table.item(row, 3).data(Qt.ItemDataRole.DisplayRole))}

    def select(self, selection: dict) -> bool:
        """Select the series ``selection`` names; False when the results do not have it."""
        for row in range(self.table.rowCount()):
            if (self.table.item(row, 0).text() == selection.get("sample")
                    and self.table.item(row, 1).text() == selection.get("peptide_range")
                    and self.table.item(row, 3).data(Qt.ItemDataRole.DisplayRole)
                    == selection.get("charge")):
                self.table.selectRow(row)
                return True
        return False

    def _selected(self) -> None:
        from ..report.figures import PlotConfig, series_figures, series_kinds

        items = self.table.selectedItems()
        if not items or self.results is None:
            return
        data_id = int(items[0].data(Qt.ItemDataRole.UserRole))
        results = self.results
        config = results.figures or PlotConfig()
        drawn = series_kinds(results, data_id, dataclasses.replace(config, kinds=SERIES_KINDS))
        for kind, panel in self.panels.items():
            if kind not in drawn:
                panel.clear("one population in every spectrum: no population map")
                continue
            one = dataclasses.replace(config, kinds=(kind,))
            panel.draw(lambda one=one, kind=kind: series_figures(results, data_id, one)[kind])
        self.window.changed()


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

        self.project_page = ProjectWorkspace(self)
        self.run_page = RunWorkspace()
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
        for text, slot in (("Open project file…", self.open_project_dialog),
                           ("Open results…", self.open_results_dialog)):
            action = QAction(text, self)
            action.triggered.connect(slot)
            file.addAction(action)
        file.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file.addAction(quit_action)

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

    def _about(self) -> None:
        self.about_box().exec()

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        for path in self.recent():
            action = QAction(path, self)
            action.triggered.connect(lambda checked=False, p=path: self.open_path(Path(p)))
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
            self.open_path(Path(path))

    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project file", "",
                                              "project files (*.toml);;all files (*)")
        if path:
            self.open_project(Path(path))

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
            log.warning("%s: %s", path.name, exc)
            return
        bundle = project.output.bundle
        if bundle is not None and Path(bundle).exists():
            self.open_results(path)  # the project's results, with its review applied
        else:
            log.info("%s: no results yet (hxex run %s makes them)", path.name, path.name)

    def _show_project(self, path: Path | None) -> None:
        self.project = path.absolute() if path is not None else None
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
        self.results_page.show_results(results, path)
        self.statusBar().showMessage(f"opened {path}", 5000)
        if self._pending_selection:
            if not self.results_page.select(self._pending_selection):
                log.warning("the results do not have the series the session showed (%s); "
                            "left out", ", ".join(f"{k} {v}" for k, v in
                                                  self._pending_selection.items()))
            self._pending_selection = {}
        if self._restoring:
            self._mark_saved()
        self.changed()

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
            selection=self.results_page.selection(),
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
        if not self._may_discard():
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
        if self._may_discard():
            logging.getLogger("hxex").removeHandler(self.handler)
            # drawings still running finish first (their results are dropped)
            QThreadPool.globalInstance().clear()
            QThreadPool.globalInstance().waitForDone(10000)
            event.accept()
        else:
            event.ignore()
