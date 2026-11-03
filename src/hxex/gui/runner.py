# SPDX-License-Identifier: GPL-3.0-or-later
"""Running an hxex command from the window (docs/PLAN.md, "GUI", "Running stages"): the command
line as a child process (``python -m hxex``, with the window's own Python), its output streamed
line by line, its progress records (``--progress-format json``) turned into a progress bar,
and Stop sent on its input (``--stop-from-input``): the first stops after the current step,
the second aborts, and a command that does not end is killed after :data:`GRACE` seconds.

A crash or an out-of-memory error ends the command, never the window.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from ..cli import EXIT_ERROR, EXIT_STOPPED, PROGRESS_PREFIX

log = logging.getLogger(__name__)

GRACE = 10.0  # seconds after the abort before the command is killed
WINDOW_OPTIONS = ("--stop-from-input", "--progress-format", "json")


def shown_command(args: list[str]) -> str:
    """The command as a user would type it in a terminal."""
    def quoted(a: str) -> str:
        return f'"{a}"' if (" " in a or not a) else a

    return "hxex " + " ".join(quoted(a) for a in args)


def outcome(code: int, crashed: bool = False) -> str:
    """How a command ended, in words."""
    if crashed:
        return "ended without finishing (killed, or it crashed)"
    return {0: "done", EXIT_STOPPED: "stopped", EXIT_ERROR: "failed (the messages say why)",
            2: "not started: the command was not understood"}.get(code, f"ended (code {code})")


class CommandRun(QObject):
    """One hxex command in a child process. :meth:`start` it; it emits :attr:`line` for every
    line it writes (``'out'`` or ``'err'``), :attr:`progress` for every progress record, and
    :attr:`finished` once with its exit code and how it ended in words."""

    line = Signal(str, str)
    progress = Signal(dict)
    finished = Signal(int, str)

    def __init__(self, parent=None, program: list[str] | None = None):
        super().__init__(parent)
        # the command line with the window's own Python (tests give their own program); -P:
        # the project's folder is not searched first, so a folder named hxex there is not run
        self.program = program or [sys.executable, "-P", "-m", "hxex"]
        self.process: QProcess | None = None
        self.stops = 0
        self._buffers = {"out": "", "err": ""}
        self._kill = QTimer(self)
        self._kill.setSingleShot(True)
        self._kill.timeout.connect(self.kill)

    @property
    def running(self) -> bool:
        return self.process is not None and \
            self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, args: list[str], cwd: Path) -> None:
        if self.running:
            raise RuntimeError("a command is running")
        self.stops = 0
        self._buffers = {"out": "", "err": ""}
        process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")  # lines as they are written, not in bursts
        env.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(env)
        process.setWorkingDirectory(str(cwd))
        process.readyReadStandardOutput.connect(lambda: self._read("out"))
        process.readyReadStandardError.connect(lambda: self._read("err"))
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._error)
        self.process = process
        program, *first = self.program
        process.start(program, [*first, *WINDOW_OPTIONS, *args])

    def stop(self) -> int:
        """Stop the command: the first call after its current step, the second at once (and
        killed if it has not ended :data:`GRACE` seconds later). Returns how many stops were
        sent."""
        if not self.running:
            return self.stops
        self.stops += 1
        self.process.write(b"stop\n")
        if self.stops >= 2 and not self._kill.isActive():
            self._kill.start(round(GRACE * 1000))
        return self.stops

    def kill(self) -> None:
        if self.running:
            log.warning("the command did not end: killed")
            self.process.kill()

    def _read(self, which: str) -> None:
        data = (self.process.readAllStandardOutput() if which == "out"
                else self.process.readAllStandardError())
        text = self._buffers[which] + bytes(data).decode("utf-8", errors="replace")
        *lines, self._buffers[which] = text.replace("\r\n", "\n").split("\n")
        for line in lines:
            self._emit(which, line)

    def _emit(self, which: str, line: str) -> None:
        if which == "err" and line.startswith(PROGRESS_PREFIX):
            try:
                self.progress.emit(json.loads(line[len(PROGRESS_PREFIX):]))
            except json.JSONDecodeError:
                pass
            return
        self.line.emit(line, which)

    def _finished(self, code: int, status) -> None:
        self._kill.stop()
        for which in ("out", "err"):
            self._read(which)
            if self._buffers[which]:
                self._emit(which, self._buffers[which])
                self._buffers[which] = ""
        crashed = status == QProcess.ExitStatus.CrashExit
        if self.process is not None:
            self.process.closeWriteChannel()
        self.finished.emit(int(code), outcome(int(code), crashed))

    def _error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.line.emit(f"the command could not be started ({self.process.errorString()})",
                           "err")
            self.finished.emit(EXIT_ERROR, "not started")

