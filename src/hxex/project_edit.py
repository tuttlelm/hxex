# SPDX-License-Identifier: GPL-3.0-or-later
"""Changing one setting of a project file from the command line (``hxex set``).

The file is edited line by line, so its comments and layout stay as they are: a setting
switched off (``# key = value  # comment``) is switched on with the new value and keeps its
comment; its heading is switched on too when it is off; a setting the file does not have is
added under its heading (the heading at the end of the file when there is none). With
``default=True`` the setting's line is switched off again (a ``# `` in front), so the setting
takes its default; its heading stays on. Lists of tables (``[[input.runs]]``) and values written
over several lines are left to the text editor.

:func:`switch_table` switches a whole table on or off (a stage such as ``[run.bootstrap]``,
whose heading alone turns it on), and :func:`live_settings` lists what a file sets: both are
for hxex's window, which edits the file the same way.

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

# tables whose heading alone changes what hxex does (a stage switched on with every setting at
# its default): the window offers to switch these on and off
SWITCHABLE = ("review", "run.controls", "run.bootstrap", "run.population_test", "run.quality",
              "analysis", "analysis.controls", "analysis.filters", "analysis.choice",
              "analysis.fasta", "analysis.runs", "analysis.merge", "analysis.fragments",
              "analysis.fragments.labels", "analysis.fragments.exchange", "analysis.labelling")

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


REPLACE_TRIES = 20  # attempts at a replacement Windows refuses, about two seconds in all


def replace_file(src: Path, dst: Path) -> None:
    """``os.replace(src, dst)``, tried again for a while when Windows refuses it: a file another
    program has open briefly (a virus scanner, an indexer, a file watcher) cannot be replaced
    there. Raises PermissionError when it is still refused."""
    import time

    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES - 1:
                raise
            time.sleep(0.1)


def _read(path: Path) -> tuple[str, bool]:
    """The project file's text and whether it starts with a byte-order mark."""
    if not path.exists():
        raise ProjectError(f"no such project file: {path} (hxex init writes one)")
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig"), raw.startswith(b"\xef\xbb\xbf")
    except UnicodeDecodeError as exc:
        raise ProjectError(f"{path} is not UTF-8 text: save it as UTF-8 (in Notepad, Save as, "
                           "Encoding UTF-8)") from exc


def _analysis_only(path: Path, text: str) -> bool:
    try:
        return "input" not in tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        from .config import _toml_hint

        raise ProjectError(f"{path} is not valid TOML ({exc}){_toml_hint(text, exc)}: correct "
                           "it in a text editor first (hxex check shows the problem)") from exc


def _write_checked(path: Path, text: str, new_text: str, bom: bool, what: str,
                   unfinished: tuple[str, ...] | None = None) -> str | None:
    """Write ``new_text`` over ``text`` unless it brings a problem ``text`` did not have.

    With ``unfinished`` (the keys being changed), a new problem that names none of them (a
    setting still needed elsewhere, as in a file being filled in) does not stop the change: the
    file is written and the problem returned. A problem with the changed values themselves
    still refuses it."""
    analysis_only = _analysis_only(path, text)
    problem = _problem(new_text, path, analysis_only)
    before = _problem(text, path, analysis_only)
    new = problem is not None and (before is None or problem[0] != before[0])
    if new and (unfinished is None
                or any(re.search(rf"\b{re.escape(k)}\b", problem[0]) for k in unfinished)):
        raise ProjectError(f"{what}; {path} is unchanged: {problem[1]}")
    data = (b"\xef\xbb\xbf" if bom else b"") + new_text.encode("utf-8")
    tmp = path.with_name(path.name + ".hxex-set")
    tmp.write_bytes(data)
    try:
        replace_file(tmp, path)
    except PermissionError:
        # still held (Windows): the text in place, not atomic but never lost
        tmp.unlink(missing_ok=True)
        path.write_bytes(data)
    return problem[1] if new else None


def file_problem(path: str | Path) -> str | None:
    """What ``hxex check`` would say first about the project file's settings (None: nothing),
    without reading any data."""
    path = Path(path)
    try:
        text, _ = _read(path)
        found = _problem(text, path, _analysis_only(path, text))
    except ProjectError as exc:
        return str(exc)
    return None if found is None else found[1]


def live_settings(text: str) -> tuple[dict[str, object], set[str]]:
    """What the project file ``text`` sets: ``table.key`` -> value for every setting that is
    switched on (lists of tables left out), and the names of the tables that are on."""
    values: dict[str, object] = {}
    tables: set[str] = set()

    def walk(data, prefix):
        for key, value in data.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                tables.add(name)
                walk(value, name)
            elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                tables.add(name)
            else:
                values[name] = value

    walk(tomllib.loads(text), "")
    return values, tables


def switch_table(path: str | Path, table: str, on: bool) -> str:
    """Switch ``table`` on (its heading, with its settings left as they are) or off (its
    heading, its settings and the tables under it get a ``# `` in front), checked as
    :func:`set_setting` checks a change. Returns one line saying what changed."""
    if table not in SWITCHABLE:
        raise ProjectError(f"[{table}] cannot be switched on or off: set one of its settings")
    path = Path(path)
    text, bom = _read(path)
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    heads = _sections(lines)
    mine = [h for h in heads if h[1] == table and not h[3]]
    if on:
        if any(h[2] for h in mine):
            return f"[{table}] is already on: nothing changed"
        if mine:
            i = mine[0][0]
            lines[i] = re.sub(r"^(\s*)#\s*", r"\1", lines[i], count=1)
            at = i + 1
        else:
            while lines and not lines[-1].strip():
                lines.pop()
            if lines and not lines[-1].endswith(("\n", "\r")):
                lines[-1] += newline
            lines += [newline, f"[{table}]{newline}"]
            at = len(lines)
        _write_checked(path, text, "".join(lines), bom, f"[{table}] was not switched on")
        return (f"[{table}] switched on, line {at} (every setting under it at its default "
                "unless set)")
    under = [h for h in heads if h[2] and (h[1] == table or h[1].startswith(table + "."))]
    if not under:
        return f"[{table}] is already off: nothing changed"
    starts = [h[0] for h in heads]
    for head in under:
        end = next((s for s in starts if s > head[0]), len(lines))
        for i in range(head[0], end):
            line = lines[i]
            if line.strip() and not line.lstrip().startswith("#"):
                indent = line[:len(line) - len(line.lstrip())]
                lines[i] = f"{indent}# {line.lstrip()}"
    _write_checked(path, text, "".join(lines), bom, f"[{table}] was not switched off")
    names = ", ".join(f"[{h[1]}]" if not h[3] else f"[[{h[1]}]]" for h in under)
    return f"switched off {names}, line {under[0][0] + 1} (a # in front of each line)"


# --- lists of tables ([[input.runs]], the exclusion rules, ...) -----------------------------------
# Each entry is changed in its own lines, so comments in the file (in other entries, and beside
# the settings of the one changed) stay where they are.

def list_entries(text: str, table: str) -> list[dict]:
    """The entries of the list of tables ``table`` in the project file ``text``."""
    data = tomllib.loads(text)
    for part in table.split("."):
        data = data.get(part, {}) if isinstance(data, dict) else {}
    return [dict(e) for e in data] if isinstance(data, list) else []


def _blocks(lines: list[str], table: str) -> list[tuple[int, int]]:
    """Each live ``[[table]]`` entry: (its heading's line, the line after its last setting)."""
    heads = _sections(lines)
    starts = [h[0] for h in heads if h[2]]  # in TOML only a live heading ends a table
    out = []
    for i, name, on, array in heads:
        if not (on and array and name == table):
            continue
        end = next((s for s in starts if s > i), len(lines))
        last = i + 1
        for j in range(i + 1, end):
            if _SETTING_LINE.match(lines[j]) and not lines[j].lstrip().startswith("#"):
                last = j + 1
                while last < end and _GO_ON.match(lines[last]):
                    last += 1
        out.append((i, last))
    return out


def _list_table(table: str) -> tuple[str, ...]:
    known = _tables()
    if table not in known or not known[table][0]:
        raise ProjectError(f"[[{table}]] is not a list of tables of a project file")
    return known[table][1] or ()


def _entry_text(lines: list[str], index: int, table: str, key: str,
                value: str | None) -> tuple[list[str], str]:
    """``lines`` with ``key`` of entry ``index`` set to the TOML text ``value`` (None: the line
    gets a ``# ``)."""
    blocks = _blocks(lines, table)
    if not 0 <= index < len(blocks):
        raise ProjectError(f"[[{table}]] has no entry {index + 1} (it has {len(blocks)})")
    start, end = blocks[index]
    newline = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
    found = _find_key(lines, start + 1, end, key)
    where = f"[[{table}]] entry {index + 1}: {key}"
    if value is None:
        if found is None or not found.on:
            return lines, f"{where} is not set: nothing changed"
        old = lines[found.index]
        indent = old[:len(old) - len(old.lstrip())]
        lines[found.index] = f"{indent}# {old.lstrip()}"
        return lines, f"{where} switched off (line {found.index + 1})"
    entry = f"{key} = {value}"
    if found is not None:
        old = lines[found.index].rstrip("\r\n")
        indent = old[:len(old) - len(old.lstrip())]
        new = indent + entry
        if found.comment_at is not None:
            comment = old[found.comment_at:]
            new = (new.ljust(found.comment_at) if len(new) < found.comment_at
                   else new + " ") + comment
        before = found.value if found.on else "not set"
        lines[found.index] = new + newline
        return lines, f"{where}: {before} -> {value}, line {found.index + 1}"
    lines.insert(end, entry + newline)
    return lines, f"{where}: not set -> {value}, line {end + 1}"


def set_entry(path: str | Path, table: str, index: int, key: str, value: str | None,
              unfinished: bool = False) -> str:
    """Set ``key`` of the entry ``index`` (from 0) of the list of tables ``table`` to ``value``
    (as :func:`set_setting` takes it; None switches its line off). Returns what changed."""
    keys = _list_table(table)
    if keys and key not in keys:
        near = difflib.get_close_matches(key, keys, n=2)
        raise ProjectError(f"[[{table}]] has no setting {key}"
                           + (f" (did you mean {' or '.join(near)}?)" if near else ""))
    path = Path(path)
    text, bom = _read(path)
    _analysis_only(path, text)
    lines = text.splitlines(keepends=True)
    lines, said = _entry_text(lines, index, table, key,
                              None if value is None else value_text(value))
    new_text = "".join(lines)
    if new_text != text:
        _write_checked(path, text, new_text, bom, f"{said} was refused",
                       (key,) if unfinished else None)
    return said


def add_entry(path: str | Path, table: str, values: dict[str, str] | None = None,
              unfinished: bool = False) -> str:
    """Add an entry to the list of tables ``table``, with ``values`` (key -> value as
    :func:`set_setting` takes it): after the table's last entry, else at the end of the
    settings of the table it belongs to, else at the end of the file."""
    keys = _list_table(table)
    values = values or {}
    for key in values:
        if keys and key not in keys:
            raise ProjectError(f"[[{table}]] has no setting {key}")
    path = Path(path)
    text, bom = _read(path)
    _analysis_only(path, text)
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += newline
    block = [f"[[{table}]]{newline}"] + [f"{k} = {value_text(v)}{newline}"
                                         for k, v in values.items()]
    blocks = _blocks(lines, table)
    if blocks:
        at = blocks[-1][1]
    else:
        at = None
        parent = table.rsplit(".", 1)[0] if "." in table else None
        heads = _sections(lines)
        for i, name, on, array in heads:
            if on and not array and name == parent:
                end = next((h[0] for h in heads if h[0] > i and h[2]), len(lines))
                at = i + 1
                for j in range(i + 1, end):
                    if _SETTING_LINE.match(lines[j]) and not lines[j].lstrip().startswith("#"):
                        at = j + 1
                        while at < end and _GO_ON.match(lines[at]):
                            at += 1
                break
        if at is None:
            while lines and not lines[-1].strip():
                lines.pop()
            at = len(lines)
    lines[at:at] = [newline] + block
    n = len(blocks) + 1
    said = f"[[{table}]] entry {n} added, line {at + 2}"
    _write_checked(path, text, "".join(lines), bom, f"{said} was refused",
                   tuple(values) if unfinished else None)
    return said


def remove_entry(path: str | Path, table: str, index: int, unfinished: bool = False) -> str:
    """Remove the entry ``index`` (from 0) of the list of tables ``table``: its heading and its
    settings (comments after its last setting are left, as they may belong to what follows)."""
    _list_table(table)
    path = Path(path)
    text, bom = _read(path)
    _analysis_only(path, text)
    lines = text.splitlines(keepends=True)
    blocks = _blocks(lines, table)
    if not 0 <= index < len(blocks):
        raise ProjectError(f"[[{table}]] has no entry {index + 1} (it has {len(blocks)})")
    start, end = blocks[index]
    del lines[start:end]
    if start < len(lines) and start > 0 and not lines[start].strip() \
            and not lines[start - 1].strip():
        del lines[start]  # no double blank line left behind
    said = f"[[{table}]] entry {index + 1} removed (it was at line {start + 1})"
    _write_checked(path, text, "".join(lines), bom, f"{said}: refused",
                   () if unfinished else None)
    return said


def set_setting(path: str | Path, setting: str, value: str | None) -> Change:
    """Set ``setting`` (``table.key``) of the project file at ``path`` to ``value``, as typed on
    the command line (:func:`value_text`), or with ``value=None`` switch it off so that it takes
    its default. The file is written only when the edited text passes the project file's
    checks (a problem it already had is no reason to refuse; the change is then written and
    the problem is left for ``hxex check``)."""
    return set_settings(path, [(setting, value)])[0]


def set_settings(path: str | Path, changes, unfinished: bool = False) -> list[Change]:
    """Several :func:`set_setting` changes made together and checked once, so that settings
    that only work together can change at once (the input format and the settings of the old
    format switched off, say).

    ``unfinished`` (hxex's window, where a file is filled in step by step): a change is written
    even when the file then still needs something else (a peptide table, a run); only a problem
    with the changed settings' own values refuses it. :func:`file_problem` then says what is
    still needed."""
    path = Path(path)
    text, bom = _read(path)
    _analysis_only(path, text)  # valid TOML first
    new_text, done = text, []
    for setting, value in changes:
        table, key = _split(setting)
        if value is not None and setting in SWITCHABLE and setting in live_settings(new_text)[1]:
            # a value and a table of one name ([run] controls and [run.controls]): TOML holds one
            raise ProjectError(f"{setting} was not set: the table [{setting}] is switched on; "
                               f"switch it off first (a # in front of [{setting}] and its "
                               "settings) to give it a value instead")
        new_text, change = _edited(new_text, table, key,
                                   None if value is None else value_text(value))
        done.append(change)
    if new_text == text:
        return done
    what = "; ".join(f"{c.setting} was not set to "
                     f"{c.after if c.after is not None else 'its default'}"
                     for c in done if not c.unchanged)
    keys = tuple(c.setting.rsplit(".", 1)[-1] for c in done) if unfinished else None
    _write_checked(path, text, new_text, bom, what, keys)
    return done
