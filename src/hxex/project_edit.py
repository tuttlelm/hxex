# SPDX-License-Identifier: GPL-3.0-or-later
"""Changing one setting of a project file from the command line (``hxex set``).

The file is edited line by line, so its comments and layout stay as they are: a setting
switched off (``# key = value  # comment``) is switched on with the new value and keeps its
comment; its heading is switched on too when it is off; a setting the file does not have is
added under its heading (the heading at the end of the file when there is none). With
``default=True`` the setting's line is switched off again (a ``# `` in front), so the setting
takes its default; its heading stays on. Lists of tables (``[[input.runs]]``) and values written
over several lines are left to the text editor.

The edited text is checked as ``hxex check`` checks a project file before anything is written;
an edit that makes the file unusable is refused and the file is left as it was.
"""

from __future__ import annotations

import difflib
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .config import ProjectError

_KEY = re.compile(r"[A-Za-z0-9_-]+")
_HEADING = re.compile(r"^\s*(#\s*)?(\[+)\s*([A-Za-z0-9_.\"-]+)\s*\]+")
_SETTING_LINE = re.compile(r"^\s*(#\s*)?[A-Za-z0-9_-]+\s*=")
_GO_ON = re.compile(r"^\s*#\s{2,}#")  # a comment going on from the line above


@dataclass(frozen=True)
class Change:
    """What :func:`set_setting` did: the setting, its value before and after (TOML text;
    None: not set, its default), the line it is on now (1-based) and the heading it switched
    on, if any."""

    setting: str
    before: str | None
    after: str | None
    line: int
    heading_on: str | None = None
    unchanged: bool = False
    bare_heading: str | None = None  # a heading left on with no setting under it

    def describe(self) -> str:
        """One line for the terminal."""
        if self.unchanged:
            state = self.after if self.after is not None else "at its default"
            return f"{self.setting} is already {state} (line {self.line}): nothing changed"
        before = self.before if self.before is not None else "not set (default)"
        after = self.after if self.after is not None else "not set (default)"
        text = f"{self.setting}: {before} -> {after}, line {self.line}"
        if self.heading_on:
            text += f"; switched on the heading [{self.heading_on}]"
        if self.bare_heading:
            text += (f"; the heading [{self.bare_heading}] stays on, with every setting at its "
                     "default (put a # in front of it to switch the table off)")
        return text


def value_text(raw: str) -> str:
    """The TOML text of a value typed on the command line: as typed when it is a TOML value
    (``0.85``, ``true``, ``[1, 2]``, ``"WT"``), otherwise the text in quotes (``controls``,
    ``20 min``), so text needs no quotes in the shell."""
    from .config import _toml_value

    raw = raw.strip()
    if not raw:
        raise ProjectError("no value given (for empty text, give \"\")")
    if "\n" in raw or "\r" in raw:
        raise ProjectError("a value on one line only: edit the file for longer values")
    try:
        tomllib.loads(f"v = {raw}")
    except tomllib.TOMLDecodeError:
        if len(raw) > 1 and raw[0] == raw[-1] and raw[0] in "\"'":
            raw = raw[1:-1]  # text quoted in the shell's way, a Windows path say
        return _toml_value(raw)
    return raw


def _tables() -> dict[str, tuple[bool, tuple[str, ...] | None]]:
    """Every table a project file accepts: name -> (a list of tables, its setting names; None:
    a table of free keys, such as sample names)."""
    from ._reference import tables

    return {t.name: (t.array, tuple(s.key for s in t.settings) or None) for t in tables()}


def _default_text(table: str, key: str) -> str | None:
    """The TOML text of a setting's default, as the settings reference shows it (None: it has
    none to write, such as an unset one)."""
    from ._reference import tables

    for t in tables():
        if t.name == table:
            for s in t.settings:
                if s.key == key and s.default.startswith("`") and s.default.endswith("`"):
                    return s.default[1:-1]
    return None


def _split(setting: str) -> tuple[str, str]:
    """``table.key`` -> (table, key), checked against the settings hxex knows."""
    parts = setting.split(".")
    if not all(_KEY.fullmatch(p) for p in parts):
        raise ProjectError(f"{setting!r} is not a setting name: write it as table.key, "
                           "e.g. run.controls.d2o_fraction (letters, digits, _ and -)")
    table, key = ".".join(parts[:-1]), parts[-1]
    known = _tables()
    if table in known:
        array, keys = known[table]
        if array:
            raise ProjectError(f"[[{table}]] is a list of tables (one per entry): hxex set "
                               "cannot tell which entry to change; edit the project file")
        if keys is None or key in keys:
            return table, key
        if key in known and not known[key][0] and table == "":
            raise ProjectError(f"{key} is a table, not a setting: name one of its settings "
                               f"({key}.NAME)")
        if f"{table}.{key}" in known:
            raise ProjectError(f"{setting} is a table, not a setting: name one of its "
                               f"settings ({setting}.NAME)")
    every = [f"{t}.{k}" if t else k for t, (_, keys) in known.items() for k in keys or ()]
    near = difflib.get_close_matches(setting, every, n=3, cutoff=0.6)
    if not near:
        near = [s for s in every if s.rsplit(".", 1)[-1] == key][:3]
    hint = f" (did you mean {' or '.join(near)}?)" if near else ""
    raise ProjectError(f"{setting} is not a setting of a project file{hint}; the settings are "
                       "listed in the guide's settings page (settings.md)")


def _value_end(rest: str) -> int:
    """Where the value at the start of ``rest`` ends: before its ``#`` comment, if any."""
    for i, char in enumerate(rest):
        if char != "#":
            continue
        try:
            tomllib.loads(f"v = {rest[:i]}")
        except tomllib.TOMLDecodeError:
            continue
        return i
    return len(rest)


@dataclass
class _Line:
    index: int  # 0-based
    on: bool
    value: str  # the value's TOML text
    comment_at: int | None  # the column of its comment


def _find_key(lines: list[str], start: int, end: int, key: str) -> _Line | None:
    """The line of ``key`` between ``start`` and ``end`` (0-based, end excluded): the one
    switched on if there is one, else the first switched off."""
    pattern = re.compile(rf"^(\s*)(#\s*)?{re.escape(key)}\s*=\s*(.*)$")
    found = None
    for i in range(start, end):
        m = pattern.match(lines[i].rstrip("\r\n"))
        if not m:
            continue
        rest = m[3]
        cut = _value_end(rest)
        value = rest[:cut].strip()
        if not value:
            continue
        try:
            tomllib.loads(f"v = {value}")
        except tomllib.TOMLDecodeError:
            if m[2] is None:  # a live value over several lines
                raise ProjectError(f"line {i + 1}: {key}'s value goes on over several lines; "
                                   "edit the project file") from None
            continue  # a comment that only looks like a setting
        comment_at = len(m[0]) - len(rest) + cut if cut < len(rest) else None
        line = _Line(i, m[2] is None, value, comment_at)
        if line.on:
            return line
        found = found or line
    return found


def _sections(lines: list[str]) -> list[tuple[int, str, bool, bool]]:
    """Every heading: (line index, table name, switched on, list of tables)."""
    out = []
    for i, text in enumerate(lines):
        m = _HEADING.match(text)
        if m:
            out.append((i, m[3].replace('"', ""), m[1] is None, len(m[2]) > 1))
    return out


def _edited(text: str, table: str, key: str, value: str | None) -> tuple[str, Change]:
    """``text`` with ``table.key`` set to the TOML text ``value`` (None: switched off)."""
    setting = f"{table}.{key}" if table else key
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += newline
    heads = _sections(lines)
    if table:
        mine = [h for h in heads if h[1] == table]
        if any(h[3] for h in mine):
            raise ProjectError(f"[[{table}]] is a list of tables in this file; edit the "
                               "project file")
        live = [h for h in mine if h[2]]
        head = (live or mine or [None])[0]
        if head is None:
            start = end = None
        else:
            start = head[0] + 1
            end = next((h[0] for h in heads if h[0] > head[0]), len(lines))
    else:
        head, start = None, 0
        end = heads[0][0] if heads else len(lines)
    found = None if start is None else _find_key(lines, start, end, key)

    if value is None:  # back to the default
        if found is None or not found.on:
            at = found.index + 1 if found else (head[0] + 1 if head else 1)
            return text, Change(setting, None, None, at, unchanged=True)
        old = lines[found.index].rstrip("\r\n")
        indent = old[:len(old) - len(old.lstrip())]
        shown = _default_text(table, key) or found.value  # the line as the template has it
        new = f"{indent}# {key} = {shown}"
        if found.comment_at is not None:
            new = new.ljust(found.comment_at) + old[found.comment_at:]
        lines[found.index] = new + newline
        bare = table if table and not any(
            _SETTING_LINE.match(lines[i]) and not lines[i].lstrip().startswith("#")
            for i in range(start, end)) else None
        return "".join(lines), Change(setting, found.value, None, found.index + 1,
                                      bare_heading=bare)

    if found is not None and found.on and found.value == value:
        return text, Change(setting, value, value, found.index + 1, unchanged=True)
    heading_on = None
    if head is not None and not head[2]:  # switch the heading on
        old = lines[head[0]]
        lines[head[0]] = re.sub(r"^(\s*)#\s*", r"\1", old, count=1)
        heading_on = table
    entry = f"{key} = {value}"
    if found is not None:
        old = lines[found.index].rstrip("\r\n")
        indent = old[:len(old) - len(old.lstrip())]
        new = indent + entry
        if found.comment_at is not None:
            comment = old[found.comment_at:]
            new = (new.ljust(found.comment_at) if len(new) < found.comment_at
                   else new + " ") + comment
        lines[found.index] = new + newline
        before = found.value if found.on else None
        return "".join(lines), Change(setting, before, value, found.index + 1, heading_on)
    if start is not None:  # under its heading, after its last setting (and that one's comment)
        at = start
        for i in range(start, end):
            if _SETTING_LINE.match(lines[i]):
                at = i + 1
                while at < end and _GO_ON.match(lines[at]):
                    at += 1
        lines.insert(at, entry + newline)
        return "".join(lines), Change(setting, None, value, at + 1, heading_on)
    while lines and not lines[-1].strip():
        lines.pop()
    lines += [newline, f"[{table}]{newline}", entry + newline]
    return "".join(lines), Change(setting, None, value, len(lines), table)


def _problem(text: str, path: Path, analysis_only: bool) -> tuple[str, str] | None:
    """What :func:`hxex.config.load_project` would say about ``text`` as the file at ``path``:
    (the message, with its hint), or None."""
    from .config import _toml_hint, _unknown_key_hint, project_from

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return str(exc), f"not valid TOML ({exc}){_toml_hint(text, exc)}"
    try:
        project_from(data, path.resolve().parent, path, analysis_only=analysis_only)
    except ProjectError as exc:
        return str(exc), f"{exc}{_unknown_key_hint(text, str(exc))}"
    return None


def set_setting(path: str | Path, setting: str, value: str | None) -> Change:
    """Set ``setting`` (``table.key``) of the project file at ``path`` to ``value``, as typed on
    the command line (:func:`value_text`), or with ``value=None`` switch it off so that it takes
    its default. The file is written only when the edited text passes the project file's
    checks (a problem it already had is no reason to refuse; the change is then written and
    the problem is left for ``hxex check``)."""
    path = Path(path)
    if not path.exists():
        raise ProjectError(f"no such project file: {path} (hxex init writes one)")
    raw = path.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ProjectError(f"{path} is not UTF-8 text: save it as UTF-8 (in Notepad, Save as, "
                           "Encoding UTF-8)") from exc
    table, key = _split(setting)
    try:
        analysis_only = "input" not in tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        from .config import _toml_hint

        raise ProjectError(f"{path} is not valid TOML ({exc}){_toml_hint(text, exc)}: correct "
                           "it in a text editor first (hxex check shows the problem)") from exc
    new_text, change = _edited(text, table, key,
                               None if value is None else value_text(value))
    if change.unchanged:
        return change
    problem = _problem(new_text, path, analysis_only)
    before = _problem(text, path, analysis_only)
    if problem is not None and (before is None or problem[0] != before[0]):
        shown = change.after if change.after is not None else "its default"
        raise ProjectError(f"{setting} was not set to {shown}; {path} is unchanged: "
                           f"{problem[1]}")
    tmp = path.with_name(path.name + ".hxex-set")
    tmp.write_bytes((b"\xef\xbb\xbf" if bom else b"") + new_text.encode("utf-8"))
    os.replace(tmp, path)
    return change
