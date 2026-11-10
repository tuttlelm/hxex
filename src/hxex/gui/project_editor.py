# SPDX-License-Identifier: GPL-3.0-or-later
"""The project editor: the project file's settings as a form (docs/PLAN.md, "GUI", "Editing
what users edit").

The form is built from the settings reference (:func:`hxex._reference.tables`: each setting's
type, default and description, the text of the guide's settings page). A change is written at
once through :mod:`hxex.project_edit`, as ``hxex set`` writes it: the file keeps its comments and
layout, and a change that would make the file unusable is refused with the reason, the file left
as it was. Stages whose heading alone turns them on (``[run.bootstrap]``, ``[analysis]``, ...)
have a switch. The file is watched, so a change made in a text editor shows here too.

Lists of tables (``[[input.runs]]``, the exclusion rules, ...) are tables of entries, one row
each (:class:`ListEditor`; every entry changed in its own lines of the file), and tables of free
names (sample names, run names) are tables of names and values (:class:`NamesEditor`).
"""

from __future__ import annotations

import contextlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .tips import tip

log = logging.getLogger(__name__)

GROUP_GAP = 10  # pixels above each top-level table in the list of tables

# the input settings of each group of formats (the reference's ``formats`` tags)
_FORMAT_GROUPS = {"folder": ("specexport", "hxexpress", "pooled"), "scan": ("mzml", "thermo")}


def _choices() -> dict[str, tuple[str, ...]]:
    """Settings with a fixed set of text values, shown as a list to choose from."""
    from ..config import FORMATS
    from ..report.figures import FORMATS as FIGURE_FORMATS

    return {"input.format": FORMATS, "figures.format": FIGURE_FORMATS}


def _shown(value) -> str:
    """A value as the form shows it: text without quotes, anything else as TOML."""
    from ..config import _toml_value

    if isinstance(value, str):
        return value
    return _toml_value(value)


def _default(text: str) -> str:
    """The reference's default (```0.85```, ``unset``, ``required``) as the form shows it."""
    if text.startswith("`") and text.endswith("`"):
        inner = text[1:-1]
        return inner[1:-1] if len(inner) > 1 and inner[0] == inner[-1] == '"' else inner
    return f"({text})"


def applies(setting, fmt: str) -> bool:
    """Whether an input setting applies to the format ``fmt`` (others are hidden)."""
    tag = setting.formats
    if tag in ("", "all"):
        return True
    return fmt in _FORMAT_GROUPS.get(tag, (tag,))


class NoWheelComboBox(QComboBox):
    """A list the mouse wheel does not change: turning it over the list scrolls the form (one
    notch over a list rewrote a setting while the form was being scrolled; owner, 2026-11-04).
    A click, or the keyboard once it has focus, still chooses."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # not focused by the wheel either

    def wheelEvent(self, event):  # noqa: N802 - Qt's name
        event.ignore()  # to the form, which scrolls


@dataclass
class Step:
    """One change the window made to the project file, for Undo and Redo: the file's bytes
    ``before`` and ``after``, and ``what`` changed, in words."""

    what: str
    before: bytes
    after: bytes


class SettingRow(QWidget):
    """One setting: its editor, a button back to the default, and its description. Emits
    :attr:`edited` with the setting's name and the value to write (TOML text as ``hxex set``
    takes it; None: back to the default)."""

    edited = Signal(str, object)

    def __init__(self, table: str, setting, project_dir: Path | None, parent=None,
                 names=None):
        super().__init__(parent)
        self.name = f"{table}.{setting.key}" if table else setting.key
        self.setting = setting
        self.kind = setting.kind
        self.project_dir = project_dir
        self.default = _default(setting.default)
        choices = _choices().get(self.name)

        self.label = QLabel(setting.key)
        self.label.setFont(QFont("monospace"))
        self.label.setToolTip(tip(self.name))
        if self.name in ORDERED:  # a list whose order matters: its items, with up and down
            self.editor = OrderedList(names)
            self.editor.changed.connect(self._ordered)
        elif self.kind == "true/false":
            self.editor = QCheckBox()
            self.editor.toggled.connect(self._toggled)
        elif choices:
            self.editor = NoWheelComboBox()
            self.editor.addItems(choices)
            self.editor.activated.connect(self._chosen)
        else:
            self.editor = QLineEdit()
            self.editor.setPlaceholderText(self.default)
            self.editor.editingFinished.connect(self._typed)
        self.reset = QPushButton("Default")
        self.reset.setToolTip(
            tip(f"back to the default ({self.default}): the line gets a # in front"))
        self.reset.clicked.connect(lambda: self.edited.emit(self.name, None))
        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.addWidget(self.editor, 1)
        if "path" in self.kind:
            browse = QToolButton()
            browse.setText("Browse…")
            browse.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            menu = QMenu(browse)
            menu.addAction("A file…", lambda: self._browse(folder=False))
            menu.addAction("A folder…", lambda: self._browse(folder=True))
            browse.setMenu(menu)
            line.addWidget(browse)
        line.addWidget(self.reset)
        self.about = QLabel(plain(setting.text or ""))
        self.about.setWordWrap(True)
        self.about.setStyleSheet("color: palette(placeholder-text);")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 6)
        layout.addLayout(line)
        layout.addWidget(self.about)
        self._value = None

    def show_value(self, value) -> None:
        """Show the file's value (None: not set, so the default)."""
        self._value = value
        is_set = value is not None
        font = self.label.font()
        font.setBold(is_set)
        self.label.setFont(font)
        self.reset.setEnabled(is_set)
        self.editor.blockSignals(True)
        if isinstance(self.editor, OrderedList):
            self.editor.set_items([str(v) for v in value] if isinstance(value, list) else [])
        elif isinstance(self.editor, QCheckBox):
            self.editor.setChecked(bool(value) if is_set else self.default == "true")
        elif isinstance(self.editor, QComboBox):
            text = value if is_set else self.default
            index = self.editor.findText(str(text))
            self.editor.setCurrentIndex(max(index, 0))
        else:
            self.editor.setText("" if not is_set else _shown(value))
        self.editor.blockSignals(False)

    def _ordered(self, items: list) -> None:
        from ..config import _toml_value

        self.edited.emit(self.name, _toml_value(list(items)) if items else None)

    def _toggled(self, checked: bool) -> None:
        self.edited.emit(self.name, "true" if checked else "false")

    def _chosen(self, index: int) -> None:
        self.edited.emit(self.name, _quoted(self.editor.itemText(index)))

    def _typed(self) -> None:
        raw = self.editor.text().strip()
        now = "" if self._value is None else _shown(self._value)
        if raw == now:
            return
        if not raw:
            self.edited.emit(self.name, None)
        elif self.kind in ("text", "path"):
            self.edited.emit(self.name, _quoted(raw))  # text, even when it looks like a number
        else:
            self.edited.emit(self.name, raw)

    def _browse(self, folder: bool) -> None:
        start = str(self.project_dir or "")
        if folder:
            chosen = QFileDialog.getExistingDirectory(self, self.name, start)
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, self.name, start)
        if chosen:
            self.edited.emit(self.name, _quoted(relative(Path(chosen), self.project_dir)))


# list settings whose order matters, edited as a list with up and down
ORDERED = ("analysis.samples",)
# settings that are a value or a table of their own (one FASTA for every sample, or one per
# sample: owner, 2026-11-05): the value's row stays on its table's page, the table has a page,
# and writing one form takes the other out of the file (TOML holds one)
BOTH_FORMS = ("analysis.fasta",)
# tables whose values are files: shown and typed without quotes, with Browse
PATH_TABLES = ("analysis.fasta",)


class OrderedList(QWidget):
    """A list whose order matters (``[analysis] samples``: the order of every figure): its
    items, ↑ and ↓ to move the selected one, Remove, and a box to add one, offering the names
    ``names()`` gives (the samples of the results open) that the list does not hold yet.
    Emits :attr:`changed` with the items after every change."""

    changed = Signal(list)

    def __init__(self, names=None):
        super().__init__()
        self.names = names
        self.list = QListWidget()
        self.list.setMaximumHeight(110)
        self.up = QPushButton("↑")
        self.down = QPushButton("↓")
        self.remove = QPushButton("Remove")
        for button, words in ((self.up, "move the selected one up"),
                            (self.down, "move the selected one down"),
                            (self.remove, "remove the selected one")):
            button.setToolTip(tip(words))
        self.up.clicked.connect(lambda: self._move(-1))
        self.down.clicked.connect(lambda: self._move(1))
        self.remove.clicked.connect(self._remove)
        self.add_box = NoWheelComboBox()
        self.add_box.setEditable(True)
        self.add_box.setToolTip(tip("a sample to add: one of the results', or typed"))
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self._add)
        side = QVBoxLayout()
        for button in (self.up, self.down, self.remove):
            side.addWidget(button)
        side.addStretch()
        top = QHBoxLayout()
        top.addWidget(self.list, 1)
        top.addLayout(side)
        add = QHBoxLayout()
        add.addWidget(self.add_box, 1)
        add.addWidget(self.add_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addLayout(add)

    def items(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]

    def set_items(self, items: list[str]) -> None:
        row = self.list.currentRow()
        self.list.clear()
        self.list.addItems(items)
        if 0 <= row < len(items):
            self.list.setCurrentRow(row)
        self._offer()

    def _offer(self) -> None:
        """The names the add box offers: the results' samples not in the list."""
        held = set(self.items())
        names = [n for n in (self.names() if self.names else []) if n not in held]
        self.add_box.blockSignals(True)
        self.add_box.clear()
        self.add_box.addItems(names)
        self.add_box.setCurrentText("")
        self.add_box.blockSignals(False)

    def showEvent(self, event):  # noqa: N802 - Qt's name
        super().showEvent(event)
        self._offer()  # the results open may have changed

    def _move(self, step: int) -> None:
        row = self.list.currentRow()
        items = self.items()
        to = row + step
        if row < 0 or not 0 <= to < len(items):
            return
        items[row], items[to] = items[to], items[row]
        self.set_items(items)
        self.list.setCurrentRow(to)
        self.changed.emit(items)

    def _remove(self) -> None:
        row = self.list.currentRow()
        if row >= 0:
            items = self.items()
            del items[row]
            self.set_items(items)
            self.changed.emit(items)

    def _add(self) -> None:
        name = self.add_box.currentText().strip()
        items = self.items()
        if name and name not in items:
            items.append(name)
            self.set_items(items)
            self.changed.emit(items)


def _quoted(text: str) -> str:
    from ..config import _toml_value

    return _toml_value(text)


def relative(path: Path, folder: Path | None) -> str:
    """``path`` as a project file writes it: relative to the project's folder when inside it
    (forward slashes, so the file works on every system), else as it is."""
    if folder is not None:
        try:
            return path.resolve().relative_to(folder.resolve()).as_posix() or "."
        except ValueError:
            pass
    return str(path)


class ProjectEditor(QWidget):
    """The project file as a form. :meth:`open` shows a file; every change is written at once."""

    changed = Signal()  # the file changed (from the form or outside)

    def __init__(self, parent=None):
        super().__init__(parent)
        from .._reference import tables

        self.tables = tables()
        # the samples of the results open, offered where a setting lists samples (the window
        # sets it; none: typed)
        self.sample_names = None
        self.path: Path | None = None
        self.values: dict[str, object] = {}
        self.on: set[str] = set()
        self.rows: dict[str, SettingRow] = {}
        self.switch: QCheckBox | None = None  # the shown table's on/off switch
        self.list_editor: ListEditor | NamesEditor | None = None
        self._shown_format: str | None = None
        self.watcher = QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._file_changed)
        # every change the window makes, to undo and redo (owner: settings are easily changed
        # by accident, especially when starting out); for this file, this session
        self.undo_steps: list[Step] = []
        self.redo_steps: list[Step] = []
        self.last_change = QLabel()
        self.last_change.setWordWrap(True)
        self.last_change.setStyleSheet("color: palette(placeholder-text);")
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo)
        for keys, slot in ((QKeySequence.StandardKey.Undo, self.undo),
                           (QKeySequence.StandardKey.Redo, self.redo)):
            shortcut = QShortcut(keys, self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

        self.problem = QLabel()
        self.problem.setWordWrap(True)
        self.problem.setStyleSheet("color: #b00020;")
        self.problem.hide()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a setting (name or description)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._show_page)
        self.every_format = QCheckBox("Show the input settings of every format")
        self.every_format.toggled.connect(self._show_page)

        self.list = QListWidget()
        for table in self.tables:
            top = "." not in table.name  # [input], [run], [analysis], ...: bold, a gap above
            if top and self.list.count():
                gap = QListWidgetItem()
                gap.setFlags(Qt.ItemFlag.NoItemFlags)  # not selectable: the arrow keys skip it
                gap.setSizeHint(QSize(0, GROUP_GAP))
                self.list.addItem(gap)
            item = QListWidgetItem(table.heading)
            item.setData(Qt.ItemDataRole.UserRole, table.name)
            if top:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.list.addItem(item)
        self.list.currentRowChanged.connect(self._show_page)
        self.page = QWidget()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.page)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.list)
        split.addWidget(self.scroll)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)  # a panel dragged to nothing left no trace
        split.setSizes([240, 900])
        top = QHBoxLayout()
        top.addWidget(self.search, 1)
        top.addWidget(self.every_format)
        changes = QHBoxLayout()
        changes.addWidget(self.last_change, 1)
        changes.addWidget(self.undo_button)
        changes.addWidget(self.redo_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.problem)
        layout.addLayout(top)
        layout.addLayout(changes)
        layout.addWidget(split, 1)
        self._show_steps()
        self.setEnabled(False)

    # --- the file ----------------------------------------------------------------------------

    def open(self, path: Path | None) -> None:
        if self.watcher.files():
            self.watcher.removePaths(self.watcher.files())
        if path != self.path:  # another file: its own changes only
            self.undo_steps, self.redo_steps = [], []
        self.path = path
        self._show_steps()
        self.setEnabled(path is not None)
        if path is not None:
            self.watcher.addPath(str(path))
        self.reload()
        if self.list.currentRow() < 0:
            self.show_table("input")

    def show_table(self, name: str) -> None:
        """Show the page of the table ``name`` (``"run.controls"``)."""
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == name:
                self.list.setCurrentRow(i)
                return
        raise KeyError(name)

    def reload(self) -> None:
        """Read the file again and show its values and its problem, if any."""
        import tomllib

        from ..project_edit import file_problem, live_settings

        self.values, self.on = {}, set()
        problem = None
        if self.path is not None:
            try:
                self.values, self.on = live_settings(self.path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
                pass
            problem = file_problem(self.path)
        self.problem.setText(f"The project file has a problem: {problem}" if problem else "")
        self.problem.setVisible(bool(problem))
        if self.list_editor is not None:
            self.list_editor.fill()
        elif not self.rows or self.format != self._shown_format:
            self._show_page()  # the format decides which input settings are shown
            return
        for name, row in self.rows.items():  # in place: the scroll position and focus stay
            row.show_value(self.values.get(name))
        if self.switch is not None:
            self.switch.blockSignals(True)
            self.switch.setChecked(self.switch.property("table") in self.on)
            self.switch.blockSignals(False)

    @contextlib.contextmanager
    def own_write(self, record: bool = True):
        """The window writing the project file itself: not watched meanwhile (on Windows a
        watched file could not be replaced), watched again after. With ``record``, a change is
        kept for Undo (the file's bytes before and after); the block may name it, setting
        ``.what`` of the step it is given."""
        watched = [f for f in self.watcher.files()]
        if watched:
            self.watcher.removePaths(watched)
        step = Step("a change to the project file", self._bytes(), b"")
        try:
            yield step
        finally:
            if self.path is not None and self.path.exists():
                self.watcher.addPath(str(self.path))
        step.after = self._bytes()
        if record and step.after != step.before:
            self.undo_steps.append(step)
            self.redo_steps.clear()
            self._show_steps()

    def _bytes(self) -> bytes:
        try:
            return self.path.read_bytes() if self.path is not None else b""
        except OSError:
            return b""

    def undo(self) -> bool:
        """Put the project file back as it was before the last change the window made (False,
        said, when there is none, or the file changed elsewhere since)."""
        return self._step_back(self.undo_steps, self.redo_steps, "undone")

    def redo(self) -> bool:
        """Make again the change last undone."""
        return self._step_back(self.redo_steps, self.undo_steps, "redone")

    def _step_back(self, take: list, give: list, word: str) -> bool:
        if not take or self.path is None:
            return False
        step = take[-1]
        now, want = (step.after, step.before) if word == "undone" else (step.before, step.after)
        if self._bytes() != now:
            self.problem.setText(
                "The project file was changed since then by something other than this editor (a "
                "text editor, hxex set, or the review panel naming your review table): not "
                f"{word}, so that change is kept. Undo works again for changes made from now "
                "on.")
            self.problem.show()
            self.undo_steps, self.redo_steps = [], []
            self._show_steps()
            return False
        with self.own_write(record=False):
            self.path.write_bytes(want)
        take.pop()
        give.append(step)
        log.info("project file: %s: %s", word, step.what)
        self._show_steps()
        self._after_change()
        return True

    def _show_steps(self) -> None:
        last = self.undo_steps[-1].what if self.undo_steps else ""
        self.last_change.setText(f"Last change: {last}" if last else
                                 "No change made in this window yet.")
        self.undo_button.setEnabled(bool(self.undo_steps))
        self.undo_button.setToolTip(tip(f"Undo: {last} (Ctrl+Z)" if last else "nothing to undo"))
        nxt = self.redo_steps[-1].what if self.redo_steps else ""
        self.redo_button.setEnabled(bool(self.redo_steps))
        self.redo_button.setToolTip(
            tip(f"Redo: {nxt} (Ctrl+Shift+Z)" if nxt else "nothing to redo"))

    def _file_changed(self, path: str) -> None:
        # a file replaced (as hxex writes it, and many editors) is no longer watched: watch again
        if Path(path).exists() and path not in self.watcher.files():
            self.watcher.addPath(path)
        self.reload()
        self.changed.emit()

    @property
    def format(self) -> str:
        return str(self.values.get("input.format", "specexport"))

    # --- the page ----------------------------------------------------------------------------

    def _show_page(self, *_) -> None:
        old = self.scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self.rows, self.switch, self.list_editor = {}, None, None
        self._shown_format = self.format
        self.page = QWidget()
        form = QVBoxLayout(self.page)
        query = self.search.text().strip().lower()
        if query:
            shown = [(t, s) for t in self.tables if not t.array for s in t.settings
                     if query in f"{t.name}.{s.key}".lower() or query in (s.text or "").lower()]
            form.addWidget(_heading(f"{len(shown)} setting(s) found"))
            box = _form()
            for table, setting in shown:
                self._add_row(box, table, setting, prefix=True)
            form.addLayout(box)
        else:
            item = self.list.currentItem()
            if item is not None:
                self._table_page(form, item.data(Qt.ItemDataRole.UserRole))
        form.addStretch()
        self.scroll.setWidget(self.page)

    def _table_page(self, form: QVBoxLayout, name: str) -> None:
        from ..project_edit import SWITCHABLE

        table = next(t for t in self.tables if t.name == name)
        form.addWidget(_heading(table.heading))
        if table.intro:
            form.addWidget(_note(plain(table.intro)))
        if name in SWITCHABLE:
            switch = QCheckBox(f"Use {table.heading} (setting any of its settings switches it on)")
            switch.setChecked(name in self.on)
            switch.setProperty("table", name)
            switch.toggled.connect(lambda on, n=name: self._switch(n, on))
            form.addWidget(switch)
            self.switch = switch
        if table.array:
            self.list_editor = ListEditor(self, table)
            form.addWidget(self.list_editor)
            return
        if not table.settings:
            self.list_editor = NamesEditor(self, table)
            form.addWidget(self.list_editor)
            return
        names = {t.name for t in self.tables}  # a key that is a table has a page of its own
        settings = [s for s in table.settings
                    if f"{name}.{s.key}" not in names or f"{name}.{s.key}" in BOTH_FORMS]
        if name == "input" and not self.every_format.isChecked():
            hidden = [s for s in settings if not applies(s, self.format)]
            settings = [s for s in settings if applies(s, self.format)]
            if hidden:
                form.addWidget(_note(f"{len(hidden)} setting(s) of other formats hidden."))
        box = _form()
        for setting in settings:
            self._add_row(box, table, setting)
        form.addLayout(box)

    def _add_row(self, layout: QFormLayout, table, setting, prefix: bool = False) -> None:
        row = SettingRow(table.name, setting, self.path.parent if self.path else None,
                         names=self.sample_names)
        row.show_value(self.values.get(row.name))
        if row.name in BOTH_FORMS and isinstance(row.editor, QLineEdit):
            entries = [k for k in self.values if k.startswith(row.name + ".")]
            row.editor.setPlaceholderText(
                f"one file per sample ({len(entries)} set): the [{row.name}] page; a file "
                "typed here replaces them" if entries else row.default)
        row.edited.connect(self._edit)
        self.rows[row.name] = row
        if prefix:
            row.label.setText(row.name)
        layout.addRow(row.label, row)

    # --- changes -----------------------------------------------------------------------------

    def other_format_settings(self, fmt: str) -> list[str]:
        """The input settings the file sets that the format ``fmt`` does not use."""
        table = next(t for t in self.tables if t.name == "input")
        return [f"input.{s.key}" for s in table.settings
                if f"input.{s.key}" in self.values and not applies(s, fmt)]

    def confirm_switch_off(self, fmt: str, names: list[str]) -> bool:
        """Ask before the settings ``names`` of the old format are switched off."""
        from PySide6.QtWidgets import QMessageBox

        answer = QMessageBox.question(
            self, "Change the input format?",
            f"{', '.join(names)} {'is' if len(names) == 1 else 'are'} not used by {fmt}. "
            "Switch them off (a # in front of the line, so they can be switched on again) and "
            f"change the format to {fmt}?")
        return answer == QMessageBox.StandardButton.Yes

    def _edit(self, setting: str, value) -> None:
        from ..config import ProjectError
        from ..project_edit import set_settings

        if self.path is None:
            return
        changes = [(setting, value)]
        table_off = None  # a table form switched off before the value is written
        for both in BOTH_FORMS:  # one form replaces the other (said in the log)
            entries = [k for k in self.values if k.startswith(both + ".")]
            if setting == both and value is not None and entries:
                table_off = both
            elif (setting.startswith(both + ".") and value is not None
                  and self.values.get(both) is not None):
                changes = [(both, None)] + changes
        if setting == "input.format" and value is not None:
            fmt = value.strip('"')
            others = self.other_format_settings(fmt)
            if others:
                if not self.confirm_switch_off(fmt, others):
                    self.rows[setting].show_value(self.values.get(setting))
                    return
                changes = [(name, None) for name in others] + changes
        try:
            with self.own_write() as step:
                said = ""
                if table_off is not None:
                    from ..project_edit import switch_table

                    said = switch_table(self.path, table_off, False)
                    log.info("%s (%s replaces it)", said, setting)
                done = set_settings(self.path, changes, unfinished=True)
                step.what = "; ".join([said] * bool(said) + [
                    _change_words(c) for c in done if not c.unchanged])
        except ProjectError as exc:
            log.warning("%s", exc)
            self.problem.setText(str(exc))
            self.problem.show()
            row = self.rows.get(setting)
            if row is not None:
                row.show_value(self.values.get(setting))  # as the file still has it
            return
        for change in done:
            log.info("%s", change.describe())
        self._after_change()

    def write(self, setting: str, value) -> None:
        """Set one setting (None: back to its default), as a row of the form does."""
        self._edit(setting, value)

    def run(self, change, *args) -> None:
        """Make a change to a list of tables (:mod:`hxex.project_edit`'s ``set_entry``,
        ``add_entry`` or ``remove_entry``) and show the result."""
        from ..config import ProjectError

        if self.path is None:
            return
        try:
            with self.own_write() as step:
                said = change(self.path, *args, unfinished=True)
                step.what = str(said)
            log.info("%s", said)
        except ProjectError as exc:
            log.warning("%s", exc)
            self.problem.setText(str(exc))
            self.problem.show()
            if self.list_editor is not None:
                self.list_editor.fill()  # as the file still has it
            return
        self._after_change()

    def _switch(self, table: str, on: bool) -> None:
        from ..config import ProjectError
        from ..project_edit import set_settings, switch_table

        try:
            with self.own_write() as step:
                if on and table in BOTH_FORMS and self.values.get(table) is not None:
                    set_settings(self.path, [(table, None)], unfinished=True)  # one form
                said = switch_table(self.path, table, on)
                step.what = str(said)
            log.info("%s", said)
        except ProjectError as exc:
            log.warning("%s", exc)
        self._after_change()

    def _after_change(self) -> None:
        # the watcher reports the change too, but not at once: show it now
        self.reload()
        self.changed.emit()


def _change_words(change) -> str:
    """A setting's change in a few words: ``[analysis] reference: "WT" -> "D109H"``."""
    table, _, key = change.setting.rpartition(".")
    name = f"[{table}] {key}" if table else key
    before = change.before if change.before is not None else "default"
    after = change.after if change.after is not None else "default"
    return f"{name}: {before} → {after}"


def _written(kind: str, raw: str) -> str:
    """What the user typed, as the TOML text to write: text kinds always as text."""
    return _quoted(raw) if kind in ("text", "path") else raw


class _CopyGrid(QTableWidget):
    """A list editor's grid, whose Ctrl+C copies every selected cell (Qt's own copied only the
    current one; owner, 2026-11-09), as the Tables view's does."""

    def __init__(self, editor: ListEditor, columns: int):
        super().__init__(0, columns)
        self.editor = editor

    def keyPressEvent(self, event):  # noqa: N802 - Qt's name
        if event.matches(QKeySequence.StandardKey.Copy):
            self.editor.copy_selection()
            event.accept()
            return
        super().keyPressEvent(event)


class ListEditor(QWidget):
    """A list of tables (``[[input.runs]]``, ...): one row per entry, one column per setting.
    A changed cell is written at once to that entry's lines; an emptied cell switches the
    setting off."""

    def __init__(self, owner: ProjectEditor, table):
        super().__init__()
        self.owner, self.table = owner, table
        self.keys = [s.key for s in table.settings]
        self.kinds = {s.key: s.kind for s in table.settings}
        self.grid = _CopyGrid(self, len(self.keys))
        self.grid.setHorizontalHeaderLabels(self.keys)
        for c, s in enumerate(table.settings):
            self.grid.horizontalHeaderItem(c).setToolTip(tip(plain(s.text or "")))
        self.grid.verticalHeader().setVisible(True)
        self.grid.cellChanged.connect(self._changed)
        self.grid.setMinimumHeight(220)
        self.grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self._menu)
        add = QPushButton("Add an entry")
        add.clicked.connect(self.add)
        self.remove = QPushButton("Remove the selected entry")
        self.remove.clicked.connect(self.remove_selected)
        self.browse = QPushButton("Browse for the selected cell…")
        self.browse.setToolTip(tip("a file or folder for a cell of a path column"))
        self.browse.clicked.connect(self._browse)
        buttons = QHBoxLayout()
        copy = QPushButton("Copy the table")
        copy.setToolTip(tip("every entry, with the column names, as a spreadsheet pastes it"))
        copy.clicked.connect(self.copy_table)
        for b in (add, self.remove, self.browse, copy):
            buttons.addWidget(b)
        buttons.addStretch()
        about = QFormLayout()
        for s in table.settings:
            text = QLabel(plain(s.text or ""))
            text.setWordWrap(True)
            name = QLabel(s.key)
            name.setFont(QFont("monospace"))
            about.addRow(name, text)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(buttons)
        layout.addWidget(self.grid)
        layout.addWidget(_note("One row per entry; type in a cell and press Enter to write it, "
                               "empty a cell to switch that setting off. The columns:"))
        layout.addLayout(about)
        self.fill()

    def entries(self) -> list[dict]:
        from ..project_edit import list_entries

        if self.owner.path is None:
            return []
        try:
            return list_entries(self.owner.path.read_text(encoding="utf-8-sig"), self.table.name)
        except Exception:  # noqa: BLE001 - a file that does not read shows no entries
            return []

    def fill(self) -> None:
        entries = self.entries()
        self.grid.blockSignals(True)
        self.grid.setRowCount(len(entries))
        for r, entry in enumerate(entries):
            for c, key in enumerate(self.keys):
                value = entry.get(key)
                self.grid.setItem(r, c, QTableWidgetItem("" if value is None else _shown(value)))
        self.grid.blockSignals(False)
        self.grid.resizeColumnsToContents()
        self.remove.setEnabled(bool(entries))

    # --- copying ---------------------------------------------------------------------------

    def _cell(self, row: int, column: int) -> str:
        item = self.grid.item(row, column)
        return "" if item is None else item.text()

    def copied_text(self) -> str:
        """The selected cells as a spreadsheet takes them: the block from the first selected row
        and column to the last, tab-separated, a cell not selected inside it left empty; with
        every cell selected (Ctrl+A), the column names first (owner, 2026-11-09)."""
        chosen = {(i.row(), i.column()) for i in self.grid.selectedIndexes()}
        if not chosen and self.grid.currentRow() >= 0 and self.grid.currentColumn() >= 0:
            chosen = {(self.grid.currentRow(), self.grid.currentColumn())}
        if not chosen:
            return ""
        rows = range(min(r for r, _ in chosen), max(r for r, _ in chosen) + 1)
        columns = range(min(c for _, c in chosen), max(c for _, c in chosen) + 1)
        lines = ["\t".join(self._cell(r, c) if (r, c) in chosen else "" for c in columns)
                 for r in rows]
        if len(chosen) == self.grid.rowCount() * len(self.keys):  # Ctrl+A: with the names
            lines.insert(0, "\t".join(self.keys[c] for c in columns))
        return "\n".join(lines) + "\n"

    def table_text(self) -> str:
        """Every entry with the column names, tab-separated."""
        lines = ["\t".join(self.keys)]
        lines += ["\t".join(self._cell(r, c) for c in range(len(self.keys)))
                  for r in range(self.grid.rowCount())]
        return "\n".join(lines) + "\n"

    def copy_selection(self) -> None:
        from PySide6.QtWidgets import QApplication

        if text := self.copied_text():
            QApplication.clipboard().setText(text)

    def copy_table(self) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.table_text())

    def _menu(self, pos) -> None:
        menu = QMenu(self)
        copy = menu.addAction("Copy")
        copy.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        copy.setEnabled(bool(self.copied_text()))
        copy.triggered.connect(self.copy_selection)
        menu.addAction("Copy the table (with the column names)").triggered.connect(
            self.copy_table)
        menu.exec(self.grid.viewport().mapToGlobal(pos))
        menu.deleteLater()

    def _changed(self, row: int, column: int) -> None:
        from ..project_edit import set_entry

        key = self.keys[column]
        raw = self.grid.item(row, column).text().strip()
        self.owner.run(set_entry, self.table.name, row, key,
                       _written(self.kinds[key], raw) if raw else None)

    def add(self) -> None:
        from ..project_edit import add_entry

        self.owner.run(add_entry, self.table.name, {})

    def remove_selected(self) -> None:
        from ..project_edit import remove_entry

        row = self.grid.currentRow()
        if row >= 0:
            self.owner.run(remove_entry, self.table.name, row)

    def _browse(self) -> None:
        row, column = self.grid.currentRow(), self.grid.currentColumn()
        if row < 0 or column < 0:
            return
        folder = self.owner.path.parent if self.owner.path else None
        chosen, _ = QFileDialog.getOpenFileName(self, self.keys[column], str(folder or ""))
        if chosen:
            self.grid.item(row, column).setText(relative(Path(chosen), folder))


class NamesEditor(QWidget):
    """A table of names chosen by the user (sample or run names, ...): a row per name, with
    its value as TOML text (``"WT"``, ``0.85``, ``{ d2o_fraction = 0.85 }``)."""

    def __init__(self, owner: ProjectEditor, table):
        super().__init__()
        self.owner, self.table = owner, table
        self.grid = QTableWidget(0, 2)
        self.grid.setHorizontalHeaderLabels(["name", "value"])
        self.grid.horizontalHeader().setStretchLastSection(True)
        self.grid.cellChanged.connect(self._changed)
        self.grid.setMinimumHeight(160)
        self.paths = table.name in PATH_TABLES  # values are files: no quotes to type, Browse
        self.name = QLineEdit()
        self.name.setPlaceholderText("a sample" if self.paths else "a new name")
        self.value = QLineEdit()
        self.value.setPlaceholderText("its FASTA file" if self.paths else "its value")
        add = QPushButton("Add")
        add.clicked.connect(self.add)
        self.remove = QPushButton("Remove the selected name")
        self.remove.clicked.connect(self.remove_selected)
        line = QHBoxLayout()
        line.addWidget(self.name)
        line.addWidget(self.value, 1)
        if self.paths:
            browse = QPushButton("Browse…")
            browse.setToolTip(tip("choose the file"))
            browse.clicked.connect(self._browse)
            line.addWidget(browse)
        line.addWidget(add)
        line.addWidget(self.remove)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(line)
        layout.addWidget(self.grid)
        self.fill()

    def items(self) -> list[tuple[str, object]]:
        prefix = self.table.name + "."
        return [(k[len(prefix):], v) for k, v in self.owner.values.items()
                if k.startswith(prefix) and "." not in k[len(prefix):]]

    def fill(self) -> None:
        items = self.items()
        self.grid.blockSignals(True)
        self.grid.setRowCount(len(items))
        for r, (name, value) in enumerate(items):
            key = QTableWidgetItem(name)
            key.setFlags(key.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.grid.setItem(r, 0, key)
            cell = QTableWidgetItem(str(value) if self.paths else _shown(value))
            swatch = _swatch(value) if self.table.name == "analysis.colours" else None
            if swatch is not None:  # a colour shown as itself beside its name
                cell.setData(Qt.ItemDataRole.DecorationRole, swatch)
            self.grid.setItem(r, 1, cell)
        self.grid.blockSignals(False)
        self.remove.setEnabled(bool(items))

    def _toml(self, raw: str) -> str:
        """A value as written: a file in quotes (typed without them), else as typed."""
        return _quoted(raw.strip('"')) if self.paths else raw

    def _changed(self, row: int, column: int) -> None:
        name = self.grid.item(row, 0).text()
        raw = self.grid.item(row, 1).text().strip()
        self.owner.write(f"{self.table.name}.{name}", self._toml(raw) if raw else None)

    def add(self) -> None:
        name, raw = self.name.text().strip(), self.value.text().strip()
        if name and raw:
            self.owner.write(f"{self.table.name}.{name}", self._toml(raw))
            self.name.clear()
            self.value.clear()

    def _browse(self) -> None:
        folder = self.owner.path.parent if self.owner.path else None
        chosen, _ = QFileDialog.getOpenFileName(self, "FASTA file", str(folder or ""),
                                                "FASTA (*.fasta *.fa *.faa *.txt);;All files (*)")
        if chosen:
            self.value.setText(relative(Path(chosen), folder))

    def remove_selected(self) -> None:
        row = self.grid.currentRow()
        if row >= 0:
            self.owner.write(f"{self.table.name}.{self.grid.item(row, 0).text()}", None)


def _swatch(value):
    """A colour (``[analysis.colours]``) as Qt draws it; None for text that is no colour."""
    from matplotlib.colors import is_color_like, to_hex

    if not isinstance(value, str) or not is_color_like(value):
        return None
    return QColor(to_hex(value))


def _form() -> QFormLayout:
    box = QFormLayout()
    box.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return box


_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def plain(text: str) -> str:
    """The reference's text without its Markdown: ``[Input formats](inputs.md)`` becomes
    ``Input formats``, and backticks go."""
    return _LINK.sub(r"\1", text).replace("`", "")


def _heading(text: str) -> QLabel:
    label = QLabel(text)
    font = label.font()
    font.setBold(True)
    font.setPointSizeF(font.pointSizeF() * 1.15)
    label.setFont(font)
    return label


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label
