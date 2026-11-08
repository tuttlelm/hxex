# SPDX-License-Identifier: GPL-3.0-or-later
"""The logs the window writes, one per command, in ``hxex_logs`` beside the project file: which
belong to a project file, and how each command ended, so that reopening a project shows how its
last command went (owner, 2026-11-08). A log names its project file on its first line
(``$ hxex run hxex.toml``; logs made before 2026-11-08 are named by command and time only, so
the line, not the name, tells the project); its last line says how the command ended
(``[hxex run hxex.toml: done, after 17m 11s, 2026-11-05 14:00:54]``). This module does not use
Qt."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

LOG_FOLDER = "hxex_logs"  # beside the project file: one log per command run from the window
SHOWN_LINES = 3000  # the most of a log shown when a project is opened (its last lines)

_STARTED = re.compile(r"^started (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
_ENDED = re.compile(r"^\[hxex .*: (.+), after (\S+(?: \S+)?), (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]$")


@dataclass(frozen=True)
class CommandLog:
    """One command's log: its file, the command (``run``, ``resume``, ...), when it started,
    and how it ended (``ended``: ``done``, ``stopped``, ...; None when the log has no end, as
    when the window was closed during the command) and after how long."""

    path: Path
    command: str
    started: datetime
    ended: str | None = None
    took: str | None = None

    def describe(self) -> str:
        """``resume, 8 Nov 13:54: done after 11m 11s``."""
        when = f"{self.started.day} {self.started:%b %H:%M}"
        ended = None if self.ended is None else self.ended.split(" (")[0]  # "failed (...)"
        end = (f"{ended} after {self.took}" if ended is not None
               else "no end recorded (the window closed during it?)")
        return f"{self.command}, {when}: {end}"


def log_name(project: Path, command: str, when: datetime, n: int = 1) -> str:
    """The file name of a new log: the project file's name first, so that each project's logs
    sort together (``hxex_run_2026-11-08_132109.log``)."""
    return f"{project.stem}_{command}_{when:%Y-%m-%d_%H%M%S}{'' if n == 1 else f'_{n}'}.log"


def _read(path: Path, project: Path) -> CommandLog | None:
    """The log at ``path`` when it is one of ``project``'s (None: another project's, or not a
    log the window wrote)."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = [fh.readline() for _ in range(3)]
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(size - 4096, 0))
            tail = fh.read().splitlines()
    except OSError:
        return None
    first = head[0].strip()
    if not first.startswith("$ hxex "):
        return None
    try:
        words = shlex.split(first[2:])
    except ValueError:
        return None
    if len(words) < 3 or Path(words[2]).name != project.name:
        return None
    started = next((m.group(1) for line in head if (m := _STARTED.match(line.strip()))), None)
    try:
        when = (datetime.strptime(started, "%Y-%m-%d %H:%M:%S") if started
                else datetime.fromtimestamp(path.stat().st_mtime))
    except (ValueError, OSError):
        return None
    last = next((line.strip() for line in reversed(tail) if line.strip()), "")
    ended = _ENDED.match(last)
    return CommandLog(path, words[1], when, ended.group(1) if ended else None,
                      ended.group(2) if ended else None)


def project_logs(project: Path) -> list[CommandLog]:
    """The logs of the project file ``project``, newest first (none without the folder)."""
    folder = project.parent / LOG_FOLDER
    if not folder.is_dir():
        return []
    found = [log for path in folder.glob("*.log") if (log := _read(path, project)) is not None]
    return sorted(found, key=lambda log: (log.started, log.path.name), reverse=True)


def log_text(log: CommandLog, lines: int = SHOWN_LINES) -> str:
    """The last ``lines`` lines of ``log`` (with a line saying how many were left out)."""
    try:
        text = log.path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"(the log cannot be read: {exc})"
    if len(text) <= lines:
        return "\n".join(text)
    return "\n".join([f"(the first {len(text) - lines} lines are left out: the whole log is "
                      f"{log.path})", *text[-lines:]])
