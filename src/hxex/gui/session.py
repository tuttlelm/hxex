# SPDX-License-Identifier: GPL-3.0-or-later
"""The GUI session file: what the window showed, so a session can be reopened (docs/PLAN.md,
"GUI", "The GUI session file").

A session holds only the view: the project file, the results and analysis folders open, the
selections, the figures open with their changes, the workspace shown, and the window layout. It
never holds settings or review decisions, which stay in the project file and the user's tables,
so the command line sees the same project whether or not the GUI was used.

The file is TOML (suggested name ``NAME.hxexgui`` beside the project file). Paths are written
relative to the session file's folder when they are inside it, so a project folder can be moved
or copied with its session. This module does not use Qt.
"""

from __future__ import annotations

import json
import logging
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SUFFIX = ".hxexgui"
VERSION = 1  # of the file's layout; a newer file is refused, an older one read
WORKSPACES = ("project", "run", "results", "analysis")


class SessionError(ValueError):
    """A session file that cannot be read (not TOML, or written by a newer hxex)."""


@dataclass
class FigureState:
    """One figure open in the window: its kind (the report function that draws it), what it is
    drawn for (``selection``: sample, peptide, charge, ...), and the user's changes to it, in
    the order they were made (docs/PLAN.md, "Changing a figure")."""

    kind: str
    selection: dict[str, Any] = field(default_factory=dict)
    changes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GuiSession:
    """What the window showed. Paths are absolute in memory; :func:`save_session` writes them
    relative to the session file where it can."""

    project: Path | None = None
    results: Path | None = None
    analysis: Path | None = None
    workspace: str = "project"
    selection: dict[str, Any] = field(default_factory=dict)
    figures: list[FigureState] = field(default_factory=list)
    layout: dict[str, str] = field(default_factory=dict)  # Qt's window geometry and state

    def __post_init__(self):
        if self.workspace not in WORKSPACES:
            raise ValueError(f"workspace must be one of {', '.join(WORKSPACES)}, not "
                             f"'{self.workspace}'")

    def missing(self) -> list[str]:
        """What the session names that no longer exists, one line each (the window drops these
        and says so)."""
        lines = []
        for what, path in (("project file", self.project), ("results", self.results),
                           ("analysis folder", self.analysis)):
            if path is not None and not path.exists():
                lines.append(f"the {what} {path} no longer exists")
        return lines


def _string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)  # a JSON string is a TOML basic string


def _value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value == value:  # NaN: TOML spells it nan
            return "nan"
        return repr(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, Path):
        return _string(value.as_posix())
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    if isinstance(value, Mapping):
        return "{ " + ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"a session cannot hold {value!r}")


def _key(key) -> str:
    key = str(key)
    return key if key.replace("_", "").replace("-", "").isalnum() and key.isascii() else \
        _string(key)


def _relative(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    path = Path(path).absolute()
    try:
        return Path(os.path.relpath(path, base)).as_posix() if path.is_relative_to(base) else \
            path.as_posix()
    except ValueError:  # another drive on Windows
        return path.as_posix()


def session_text(session: GuiSession, base: Path) -> str:
    """The session as TOML, with paths relative to ``base`` where they are inside it."""
    lines = ["# hxex GUI session: what the window showed (no settings, no review decisions)",
             f"version = {VERSION}", f"workspace = {_string(session.workspace)}"]
    for name in ("project", "results", "analysis"):
        text = _relative(getattr(session, name), base)
        if text is not None:
            lines.append(f"{name} = {_string(text)}")
    if session.selection:
        lines += ["", "[selection]"] + [f"{_key(k)} = {_value(v)}"
                                        for k, v in session.selection.items()]
    if session.layout:
        lines += ["", "[layout]"] + [f"{_key(k)} = {_value(v)}" for k, v in session.layout.items()]
    for fig in session.figures:
        lines += ["", "[[figures]]", f"kind = {_string(fig.kind)}",
                  f"selection = {_value(fig.selection)}",
                  f"changes = {_value(fig.changes)}"]
    return "\n".join(lines) + "\n"


def save_session(session: GuiSession, path: str | Path) -> Path:
    """Write ``session`` to ``path`` (a temporary file, then replaced, so a failed write leaves
    the previous session as it was). Returns the path written."""
    path = Path(path).absolute()
    text = session_text(session, path.parent)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_session(path: str | Path) -> GuiSession:
    """Read a session file. Paths come back absolute (relative ones from the session file's
    folder); a file written by a newer hxex is refused, and keys this version does not know are
    left out with a warning."""
    path = Path(path).absolute()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise SessionError(f"{path} is not a session file: {exc}") from None
    version = data.get("version")
    if not isinstance(version, int):
        raise SessionError(f"{path} is not a session file: it has no version")
    if version > VERSION:
        raise SessionError(f"{path} was written by a newer hxex (session version {version}; "
                           f"this one reads up to {VERSION}): update hxex to open it")
    known = {"version", "workspace", "project", "results", "analysis", "selection", "layout",
             "figures"}
    extra = sorted(set(data) - known)
    if extra:
        log.warning("%s: left out what this hxex does not know: %s", path.name, ", ".join(extra))
    base = path.parent

    def where(name):
        value = data.get(name)
        return None if value is None else (base / value).resolve()

    workspace = data.get("workspace", "project")
    if workspace not in WORKSPACES:
        log.warning("%s: unknown workspace '%s'; the project workspace is shown", path.name,
                    workspace)
        workspace = "project"
    figures = []
    for item in data.get("figures", []):
        if not isinstance(item, Mapping) or not isinstance(item.get("kind"), str):
            log.warning("%s: a figure without a kind is left out", path.name)
            continue
        figures.append(FigureState(item["kind"], dict(item.get("selection", {})),
                                   [dict(c) for c in item.get("changes", [])]))
    return GuiSession(project=where("project"), results=where("results"),
                      analysis=where("analysis"), workspace=workspace,
                      selection=dict(data.get("selection", {})), figures=figures,
                      layout={str(k): str(v) for k, v in data.get("layout", {}).items()})
