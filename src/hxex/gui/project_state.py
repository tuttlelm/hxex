# SPDX-License-Identifier: GPL-3.0-or-later
"""The project file as the window read it last: one parsed state every panel reads (audit of
2026-11-15, Q-2: the file was parsed in fourteen places with two readers, each handling a
file that does not read in its own way, so a reader could not tell which view of the settings
a panel showed).

:class:`ProjectState` is immutable (a worker thread may hold one while the file changes) and
holds both readings hxex has of a project file: the settings as written
(:func:`hxex.project_edit.live_settings`: every key's text and which tables are switched on),
and the project as hxex loads it (:func:`hxex.config.load_project` with ``analysis_only``, so a
project file without an ``[input]`` loads too). A file that does not read or load gives a
state with the ``problem`` and the defaults, said once in the log. The window renews it
whenever the project file changes (:attr:`hxex.gui.window.MainWindow.project_state`). This
module does not use Qt.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectState:
    """The project file at ``path`` (None: no project open) as read: ``text`` (None when it
    cannot be read), the settings as written (``values``: key -> value; ``tables``: the tables
    switched on), the project as hxex loads it (``loaded``, :class:`hxex.config.Project` with
    ``analysis_only``; None when it does not load), ``problem`` (why the file does not read or
    load, in the words ``hxex check`` uses; None when it does), and the file's ``signature``
    (modification time and size) to tell a change."""

    path: Path | None = None
    text: str | None = None
    values: Mapping[str, object] = field(default_factory=dict)
    tables: frozenset[str] = frozenset()
    loaded: object | None = None
    problem: str | None = None
    signature: tuple | None = None
    parsed: bool = False  # the text read as settings (``values`` and ``tables`` are the file's)

    @classmethod
    def read(cls, path: Path | str | None) -> ProjectState:
        """The state of the project file at ``path`` now (the empty state for None)."""
        if path is None:
            return cls()
        from ..config import ProjectError, load_project
        from ..project_edit import live_settings

        path = Path(path)
        signature = None
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            log.warning("the project file %s cannot be read (%s): its settings are taken as "
                        "their defaults", path, exc)
            return cls(path, None, {}, frozenset(), None, f"cannot read the project file: "
                                                          f"{exc}", signature)
        values: Mapping[str, object] = {}
        tables: frozenset[str] = frozenset()
        problem, parsed = None, False
        try:
            got, on = live_settings(text)
            values, tables, parsed = dict(got), frozenset(on), True
        except Exception as exc:  # noqa: BLE001 - a file that does not parse, said once
            problem = f"the project file does not read: {exc}"
        loaded = None
        try:
            loaded = load_project(path, analysis_only=True)
        except (ProjectError, OSError, ValueError) as exc:
            problem = problem or str(exc)
        if problem is not None:
            log.warning("%s: %s (the window takes the settings it needs as their defaults)",
                        path, problem)
        return cls(path, text, values, tables, loaded, problem, signature, parsed)

    def fresh(self) -> ProjectState:
        """This state while the file is as it was read (its size and modification time), else
        the file read again: for a panel answering an action before the window's watcher
        fired (the window renews its state then too)."""
        if self.path is None:
            return self
        try:
            st = self.path.stat()
            signature = (st.st_mtime_ns, st.st_size)
        except OSError:
            signature = None
        return self if signature == self.signature else ProjectState.read(self.path)

    # --- what the panels ask for --------------------------------------------------------------

    @property
    def analysis(self):
        """The ``[analysis]`` settings (:class:`hxex.analysis.AnalysisConfig`; the defaults
        without a project, or when the file does not load)."""
        from ..analysis import AnalysisConfig

        loaded = self.loaded
        return (getattr(loaded, "analysis", None) or AnalysisConfig()) if loaded is not None \
            else AnalysisConfig()

    @property
    def input(self):
        """The ``[input]`` settings (:class:`hxex.config.InputSettings`), or None (no project,
        a file that does not load, or an analysis-only project file)."""
        return getattr(self.loaded, "input", None)

    @property
    def output(self):
        """The ``[output]`` settings, or None."""
        return getattr(self.loaded, "output", None)

    @property
    def bundle(self) -> Path | None:
        """The results bundle the project writes to (``[output] bundle``), or None."""
        bundle = getattr(self.output, "bundle", None)
        return Path(bundle) if bundle is not None else None

    @property
    def format(self) -> str | None:
        """The input format (from the loaded project, else from the text as written)."""
        inp = self.input
        if inp is not None:
            return str(inp.format)
        value = self.values.get("input.format")
        return str(value).strip('"') if value is not None else None

    def value(self, key: str, default=None):
        """The setting ``key`` (``table.key``) as written, or ``default``."""
        return self.values.get(key, default)

    def of_tables(self, *tables: str) -> dict:
        """The settings of ``tables`` as written, for telling a change of them: ``values``
        (key -> value), ``tables`` (those switched on, sorted), and ``lists``: each list of
        tables under them with its entries (``analysis.groups.B.regions`` -> its regions), which
        ``values`` leaves out; without them a region's residues changed, or a region or an
        exclusion rule added, went unseen (owner, 2026-11-18: Compute kept the old regions)."""
        return {"values": {k: v for k, v in self.values.items() if k.split(".")[0] in tables},
                "tables": sorted(n for n in self.tables if n.split(".")[0] in tables),
                "lists": self._lists(tables)}

    def _lists(self, tables) -> dict[str, list]:
        import tomllib

        if self.text is None:
            return {}
        try:
            data = tomllib.loads(self.text)
        except tomllib.TOMLDecodeError:
            return {}
        out: dict[str, list] = {}

        def walk(item: Mapping, prefix: str) -> None:
            for key, value in item.items():
                name = f"{prefix}.{key}"
                if isinstance(value, Mapping):
                    walk(value, name)
                elif isinstance(value, list) and value and all(isinstance(v, Mapping)
                                                               for v in value):
                    out[name] = [dict(v) for v in value]

        for table in tables:
            if isinstance(data.get(table), Mapping):
                walk(data[table], table)
        return out
