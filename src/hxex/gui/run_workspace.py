# SPDX-License-Identifier: GPL-3.0-or-later
"""The Run workspace: hxex's commands run from the window (docs/PLAN.md, "GUI", "Running
stages"). Each button runs one command on the project file as a child process
(:class:`hxex.gui.runner.CommandRun`), shows the command as it would be typed in a terminal,
its output in a console, and a progress bar; Stop stops it after the current step, and a second
press aborts. One command runs at a time. A command that refuses to write over earlier output
offers to run again replacing it or keeping it as ``NAME.N`` (``--overwrite``,
``--keep-previous``); when a run, resume or report has finished, the window shows the new
results.

Every command's whole output also goes to a log file beside the project file
(``hxex_logs/COMMAND_DATE_TIME.log``; owner: the console keeps only its last lines, and a record
outlasts the window), with the command, the folder, the hxex version, and how it ended.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..hooks import format_duration, format_percent
from .runner import CommandRun, shown_command

log = logging.getLogger(__name__)

# (button, command, what it does): the commands a project runs, in the order of a project's life
COMMANDS = (("Check", "check", "check the project file and show what would be read"),
            ("Run", "run", "read, pick, fit and bootstrap: the results bundle"),
            ("Resume", "resume",
             "add to the results without fitting again: finish a stopped bootstrap, add or "
             "redo one (its settings changed), bootstrap the K your review chose; add or redo "
             "the population test; work the controls out again when [run.controls] changed; "
             "assess the quality flags again when an earlier hxex made them"),
            ("Report", "report", "write the tables again (with the review applied)"),
            ("Analyse", "analyse", "compare the samples: the analysis tables and figures"),
            ("Info", "info", "what the results hold"))
CHANGES_RESULTS = ("run", "resume", "report")  # the window shows the results again after these
DEVICE_SETTING = "[run.fit.backend] device"  # the project file's device setting, by its name
REFUSED = "--keep-previous"  # in the message of a command refusing to write over its output
LOG_FOLDER = "hxex_logs"  # beside the project file: one log per command run from the window


class RunWorkspace(QWidget):
    """The commands of the project shown, one at a time."""

    results_changed = Signal()  # a run, resume or report finished: the results are new
    show_setting = Signal(str)  # the user asked to change a setting (its table's name)

    def __init__(self, parent=None, program: list[str] | None = None):
        super().__init__(parent)
        self.project: Path | None = None
        self.args: list[str] = []
        self.runner = CommandRun(self, program)
        self.log_file = None  # the open log of the command running
        self.log_path: Path | None = None
        self._began = 0.0
        self.runner.line.connect(self._line)
        self.runner.progress.connect(self._progress)
        self.runner.finished.connect(self._finished)

        self.where = QLabel("No project file open.")
        self.where.setWordWrap(True)
        self.buttons: dict[str, QPushButton] = {}
        row = QHBoxLayout()
        for text, command, tip in COMMANDS:
            button = QPushButton(text)
            button.setToolTip(f"hxex {command}: {tip}")
            button.clicked.connect(lambda checked=False, c=command: self.run_command(c))
            row.addWidget(button)
            self.buttons[command] = button
        row.addStretch()
        self.device = QComboBox()
        self.device.addItems([DEVICE_SETTING, "cpu", "cuda"])
        self.device.setToolTip("for Run and Resume: where the fits run. The first entry is the "
                               "project file's setting ([run.fit.backend] device); cpu or cuda "
                               "use that device for this command only (--device)")
        change = QPushButton("Change…")
        change.setToolTip("the project file's device setting, in the Project workspace")
        change.clicked.connect(lambda checked=False: self.show_setting.emit("run.fit.backend"))
        self.figures = QCheckBox("Report draws the figures too")
        self.figures.setToolTip("hxex report hxex.toml --figures: with the [figures] settings")
        options = QHBoxLayout()
        options.addWidget(self.device)
        options.addWidget(change)
        options.addSpacing(16)
        options.addWidget(self.figures)
        options.addStretch()

        self.command = QLineEdit()
        self.command.setReadOnly(True)
        self.command.setPlaceholderText("the command, as you would type it in a terminal")
        self.command.setFont(QFont("monospace"))
        self.stop_button = QPushButton("Stop")
        self.stop_button.setToolTip("stop after the current step (press again to abort now)")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        line = QHBoxLayout()
        line.addWidget(self.command, 1)
        line.addWidget(self.stop_button)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.state = QLabel("")
        # the command's total run time, counted while it runs and kept when it ends (owner,
        # 2026-11-05)
        self.elapsed = QLabel("")
        self.elapsed.setToolTip("the total run time of the command")
        self._started_at: float | None = None
        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._tick)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("monospace"))
        self.console.setMaximumBlockCount(20000)
        self.wrap = QCheckBox("Wrap lines")
        self.wrap.setToolTip("long lines broken to the width of the console (off: one line "
                             "each, scrolled sideways)")
        self.wrap.toggled.connect(self._wrap)
        self.wrap.setChecked(True)
        clear = QPushButton("Clear")
        clear.setToolTip("empty the console")
        clear.clicked.connect(self.console.clear)
        progress = QHBoxLayout()  # the bar and its words together (owner)
        progress.addWidget(self.bar, 3)
        progress.addWidget(self.elapsed)
        progress.addWidget(self.state, 2)
        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(self.wrap)
        bottom.addWidget(clear)

        layout = QVBoxLayout(self)
        layout.addWidget(self.where)
        layout.addLayout(row)
        layout.addLayout(options)
        layout.addLayout(line)
        layout.addLayout(progress)
        layout.addWidget(self.console, 1)
        layout.addLayout(bottom)
        self.show_project(None)

    # --- the questions (replaced in tests) -------------------------------------------------

    def ask_output_exists(self) -> str | None:
        """'overwrite', 'keep-previous', or None (cancel): the command refused to write over
        earlier output."""
        box = QMessageBox(self)
        box.setWindowTitle("Earlier results")
        box.setText("The command found earlier output in its folder and did not touch it. "
                    "Run it again and replace that output, or keep it under the next free "
                    "name NAME.N?")
        replace = box.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
        keep = box.addButton("Keep it as NAME.N", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        return {replace: "overwrite", keep: "keep-previous"}.get(box.clickedButton())

    def ask_stop_to_close(self) -> bool:
        return QMessageBox.question(
            self, "A command is running",
            f"{shown_command(self.args)} is still running. Stop it and close? (Stopped after "
            "its current step, as Stop does; a bootstrap stopped part way can be finished "
            "with Resume.)", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        ) == QMessageBox.StandardButton.Yes

    # --- the project -----------------------------------------------------------------------

    def show_project(self, path: Path | None) -> None:
        self.project = path
        self.where.setText(f"Commands run on {path.name}, in {path.parent}" if path is not None
                           else "No project file open: open or make one in the Project "
                                "workspace.")
        self.show_device()
        self._enable()

    def show_device(self) -> None:
        """The first entry of the device list: the project file's setting, by name and value
        (its default, auto, when the file does not set it)."""
        value = "auto"
        if self.project is not None:
            try:
                from ..project_edit import live_settings

                text = self.project.read_text(encoding="utf-8-sig")
                value = str(live_settings(text)[0].get("run.fit.backend.device", "auto"))
            except Exception:  # noqa: BLE001 - a file that does not read: its default
                pass
        self.device.setItemText(0, f"{DEVICE_SETTING}: {value}")

    def _enable(self) -> None:
        idle = not self.runner.running
        for button in self.buttons.values():
            button.setEnabled(idle and self.project is not None)
        self.stop_button.setEnabled(not idle)

    # --- running ---------------------------------------------------------------------------

    def arguments(self, command: str, extra: tuple[str, ...] = ()) -> list[str]:
        """The command's arguments for the project shown, with the options chosen."""
        args = [command, self.project.name]
        if command in ("run", "resume") and self.device.currentIndex() > 0:
            args += ["--device", self.device.currentText()]
        if command == "report" and self.figures.isChecked():
            args.append("--figures")
        return args + list(extra)

    def run_command(self, command: str, extra: tuple[str, ...] = ()) -> None:
        if self.project is None or self.runner.running:
            return
        self.args = self.arguments(command, extra)
        typed = shown_command(self.args)
        self.command.setText(typed)
        self.console.appendPlainText(f"\n$ {typed}    (in {self.project.parent})")
        self.bar.setValue(0)
        self.bar.setRange(0, 0)  # busy until the first progress record
        self.state.setText("running…")
        self._started_at = time.monotonic()
        self._tick()
        self._clock.start()
        self._tail: list[str] = []
        self._open_log(typed)
        self.runner.start(self.args, self.project.parent)
        self._enable()
        self.stop_button.setText("Stop")

    def stop(self) -> None:
        sent = self.runner.stop()
        if sent == 1:
            self.state.setText("stopping after the current step… (Abort now: at once)")
            self.stop_button.setText("Abort now")
        elif sent >= 2:
            self.state.setText("aborting…")
            self.stop_button.setEnabled(False)

    def _wrap(self, on: bool) -> None:
        self.console.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth if on
                                     else QPlainTextEdit.LineWrapMode.NoWrap)

    def _open_log(self, typed: str) -> None:
        """A new log for the command (a folder that cannot be written: said once, no log)."""
        from .. import version_text

        self._began = time.monotonic()
        now = datetime.now()
        folder = self.project.parent / LOG_FOLDER
        path = folder / f"{self.args[0]}_{now:%Y-%m-%d_%H%M%S}.log"
        n = 1
        while path.exists():  # two commands in one second
            n += 1
            path = folder / f"{self.args[0]}_{now:%Y-%m-%d_%H%M%S}_{n}.log"
        try:
            folder.mkdir(exist_ok=True)
            self.log_file = path.open("w", encoding="utf-8")
        except OSError as exc:
            self.log_file, self.log_path = None, None
            self.console.appendPlainText(f"(no log file: {exc})")
            return
        self.log_path = path
        self.log_file.write(f"$ {typed}\nin {self.project.parent}\nstarted "
                            f"{now:%Y-%m-%d %H:%M:%S}, {version_text()}\n\n")
        self.log_file.flush()
        self.console.appendPlainText(f"(the whole output also goes to {path})")

    def _close_log(self, how: str) -> None:
        if self.log_file is None:
            return
        took = time.monotonic() - self._began
        self.log_file.write(f"\n[{shown_command(self.args)}: {how}, after "
                            f"{format_duration(took)}, {datetime.now():%Y-%m-%d %H:%M:%S}]\n")
        self.log_file.close()
        self.log_file = None

    def _line(self, text: str, which: str) -> None:
        self.console.appendPlainText(text)
        self._tail = (self._tail + [text])[-20:]
        if self.log_file is not None:
            self.log_file.write(text + "\n")
            self.log_file.flush()  # a record even if the window ends during the command

    def _progress(self, record: dict) -> None:
        overall = record.get("overall")
        if overall is not None:
            self.bar.setRange(0, 1000)
            self.bar.setValue(round(1000 * min(max(overall, 0.0), 1.0)))
        stage = record.get("stage") or ""
        text = f"{stage}: {format_percent(record.get('fraction') or 0)}"
        if record.get("left") is not None:
            text += f", about {format_duration(record['left'])} left"
        if record.get("event") == "done":
            text = f"{stage}: done"
        self.state.setText(text)

    def _tick(self) -> None:
        if self._started_at is not None:
            self.elapsed.setText(f"total {format_duration(time.monotonic() - self._started_at)}")

    def _finished(self, code: int, how: str) -> None:
        self._tick()  # the final total stays shown
        self._clock.stop()
        command = self.args[0] if self.args else ""
        self._close_log(how)
        self.bar.setRange(0, 1000)
        self.bar.setValue(1000 if code == 0 else self.bar.value())
        self.console.appendPlainText(f"[{shown_command(self.args)}: {how}]")
        self.state.setText(f"{command}: {how}")
        self.stop_button.setText("Stop")
        self._enable()
        if code != 0 and any(REFUSED in line for line in self._tail):
            choice = self.ask_output_exists()
            if choice is not None:
                self.run_command(command, (f"--{choice}",))
            return
        if code == 0 and command in CHANGES_RESULTS:
            self.results_changed.emit()

    def may_close(self) -> bool:
        """Before the window closes: a running command is stopped (after asking), and the
        window waits for it to end (aborted, then killed, if it does not stop in time)."""
        if not self.runner.running:
            return True
        if not self.ask_stop_to_close():
            return False
        self.runner.stop()
        process = self.runner.process
        if not process.waitForFinished(30000):
            self.runner.stop()
            if not process.waitForFinished(10000):
                self.runner.kill()
                process.waitForFinished(5000)
        return True

