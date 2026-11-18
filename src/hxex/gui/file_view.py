# SPDX-License-Identifier: GPL-3.0-or-later
"""The Project workspace's File tab: the project file as text, to search and to edit.

Typing changes nothing on disk: **Save** writes the text (refused, with the line, when it is not
valid TOML; a settings problem is written and shown, as in the Settings form) and **Discard**
puts the file's text back. A save is one step of the Settings form's Undo. When the file
changes elsewhere while there are unsaved edits (the Settings form, a text editor, ``hxex
set``, the review panel), the edits are kept and a line says so, with a button to load the file
instead. Ctrl+F finds text: every match is marked, Enter and Shift+Enter step through them.
The text is coloured as TOML (:mod:`hxex.gui.toml_colours`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPalette, QShortcut, QTextCursor, QTextDocument
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .fonts import fixed_font
from .tips import ERROR_COLOUR, tip
from .toml_colours import TomlColours

log = logging.getLogger(__name__)

MAX_MARKED = 5000  # matches marked at most (a one-letter search of a long file)


class _FindLine(QLineEdit):
    """The find box: Enter finds the next match, Shift+Enter the one before, Escape clears."""

    step = Signal(int)

    def keyPressEvent(self, event):  # noqa: N802 - Qt's name
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            back = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            self.step.emit(-1 if back else 1)
            return
        if event.key() == Qt.Key.Key_Escape and self.text():
            self.clear()
            return
        super().keyPressEvent(event)


class ProjectFileView(QWidget):
    """The project file as text (:meth:`show_file`), searchable and editable; ``write`` (set by
    the window) writes a text and returns the file's problem, if any."""

    modified_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.path: Path | None = None
        self.base = ""  # the file's text the edits started from
        self.disk = ""  # the file's text as read last
        self.write: Callable[[str], str | None] | None = None
        self._modified = False
        self._matches: list[QTextCursor] = []
        self._at = -1

        self.find = _FindLine()
        self.find.setPlaceholderText("Find in the file (Ctrl+F)")
        self.find.setClearButtonEnabled(True)
        self.find.textChanged.connect(self._search)
        self.find.step.connect(self.step)
        self.case = QCheckBox("Match case")
        self.case.toggled.connect(self._search)
        self.previous = QToolButton()
        self.previous.setText("▲")
        self.previous.setToolTip(tip("the match before (Shift+Enter)"))
        self.previous.clicked.connect(lambda: self.step(-1))
        self.next = QToolButton()
        self.next.setText("▼")
        self.next.setToolTip(tip("the next match (Enter)"))
        self.next.clicked.connect(lambda: self.step(1))
        self.count = QLabel()
        self.count.setStyleSheet("color: palette(placeholder-text);")

        self.text = QPlainTextEdit()
        self.text.setFont(fixed_font())
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setReadOnly(True)  # until a project file is shown
        self.colours = TomlColours(self.text.document(), self.palette())
        self.text.textChanged.connect(self._edited)

        self.state = QLabel()
        self.state.setWordWrap(True)
        self.save_button = QPushButton("Save")
        self.save_button.setToolTip(tip("write this text as the project file (refused when it "
                                        "is not valid TOML)"))
        self.save_button.clicked.connect(self.save)
        self.discard_button = QPushButton("Discard changes")
        self.discard_button.setToolTip(tip("put the project file's text back"))
        self.discard_button.clicked.connect(self.discard)
        self.problem = QLabel()
        self.problem.setWordWrap(True)
        self.problem.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.problem.setStyleSheet(f"color: {ERROR_COLOUR};")
        self.problem.hide()
        self.outside = QLabel("The project file was changed outside this tab since you started "
                              "editing. Save writes your text over that change; Load the file "
                              "drops your edits.")
        self.outside.setWordWrap(True)
        self.load_button = QPushButton("Load the file")
        self.load_button.clicked.connect(self.discard)
        outside = QHBoxLayout()
        outside.addWidget(self.outside, 1)
        outside.addWidget(self.load_button)
        self.outside_row = QWidget()
        self.outside_row.setLayout(outside)
        outside.setContentsMargins(0, 0, 0, 0)
        self.outside_row.hide()

        finding = QHBoxLayout()
        finding.addWidget(self.find, 1)
        finding.addWidget(self.case)
        finding.addWidget(self.previous)
        finding.addWidget(self.next)
        finding.addWidget(self.count)
        saving = QHBoxLayout()
        saving.addWidget(self.state, 1)
        saving.addWidget(self.save_button)
        saving.addWidget(self.discard_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(finding)
        layout.addLayout(saving)
        layout.addWidget(self.problem)
        layout.addWidget(self.outside_row)
        layout.addWidget(self.text, 1)
        shortcut = QShortcut(QKeySequence.StandardKey.Find, self)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self.focus_find)
        self._show_state()

    def changeEvent(self, event):  # noqa: N802 - Qt's name
        if event.type() == QEvent.Type.PaletteChange:  # light or dark: the colours follow
            self.colours.use_palette(self.palette())
        super().changeEvent(event)

    # --- the file ----------------------------------------------------------------------------

    def show_file(self, path: Path | None) -> None:
        """Show the project file at ``path`` (read again). Unsaved edits of the same file are
        kept; a line says when the file changed under them."""
        if path != self.path:
            self.path = path
            self.base = self.disk = ""
            self._set_text("")
        if path is None:
            self._set_text("")
            self.disk = self.base = ""
            self.text.setReadOnly(True)
            self._show_state()
            return
        try:
            disk = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            disk = None
            shown = f"cannot read the project file: {exc}"
        self.text.setReadOnly(disk is None)
        if disk is None:
            self.disk = self.base = shown
            self._set_text(shown)
        elif not self._modified:
            self.disk = self.base = disk
            if self.text.toPlainText() != disk:
                self._set_text(disk)
        else:
            self.disk = disk
            if self.text.toPlainText() == disk:  # the change outside is the edit itself
                self.base = disk
                self._edited()
        self.outside_row.setVisible(self._modified and self.disk != self.base)
        self._show_state()

    def toPlainText(self) -> str:  # noqa: N802 - as QPlainTextEdit names it
        return self.text.toPlainText()

    @property
    def modified(self) -> bool:
        return self._modified

    def _set_text(self, text: str) -> None:
        bar = self.text.verticalScrollBar().value()
        cursor = self.text.textCursor().position()
        self.text.blockSignals(True)
        self.text.setPlainText(text)
        self.text.blockSignals(False)
        moved = self.text.textCursor()  # the place in a file read again stays where it was
        moved.setPosition(min(cursor, len(text)))
        self.text.setTextCursor(moved)
        self.text.verticalScrollBar().setValue(bar)
        self._edited()  # the matches marked again, the current one kept

    def _edited(self) -> None:
        modified = self.path is not None and not self.text.isReadOnly() \
            and self.text.toPlainText() != self.base
        if modified != self._modified:
            self._modified = modified
            if not modified:
                self.outside_row.hide()
            self.modified_changed.emit(modified)
        self._show_state()
        if self.find.text():
            self._search(keep=True)

    def _show_state(self) -> None:
        if self.path is None:
            text = "No project file open."
        elif self._modified:
            text = "Unsaved changes: Save writes them to the project file."
        else:
            text = "Type to edit; nothing is written until Save."
        self.state.setText(text)
        self.save_button.setEnabled(self._modified)
        self.discard_button.setEnabled(self._modified)

    def save(self) -> bool:
        """Write the text as the project file; False (said, the text kept) when refused."""
        from ..config import ProjectError
        from ..project_edit import TextNotSaved

        if not self._modified or self.path is None or self.write is None:
            return True
        if self.disk != self.base and not self.ask_overwrite():
            return False
        text = self.text.toPlainText()
        try:
            problem = self.write(text)
        except TextNotSaved as exc:
            self._show_problem(str(exc))
            if exc.line:
                self.go_to_line(exc.line)
            return False
        except (ProjectError, OSError) as exc:
            self._show_problem(f"not saved: {exc}")
            return False
        self.base = self.disk = text
        self.outside_row.hide()
        self._edited()
        self.problem.setVisible(bool(problem))
        self.problem.setText(f"Saved, but the project file has a problem: {problem}"
                             if problem else "")
        log.info("project file saved from the File tab: %s", self.path.name)
        return True

    def discard(self) -> None:
        """The file's text back in place of the edits."""
        self.problem.hide()
        self.base = self.disk
        self._set_text(self.disk)
        self.outside_row.hide()

    def may_leave(self) -> bool:
        """Before another project or the end: unsaved edits are saved, dropped, or the leaving
        cancelled (False)."""
        if not self._modified:
            return True
        answer = self.ask_unsaved()
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        self.discard()
        return True

    def go_to_line(self, line: int) -> None:
        block = self.text.document().findBlockByNumber(max(line - 1, 0))
        if block.isValid():
            cursor = QTextCursor(block)
            cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock,
                                QTextCursor.MoveMode.KeepAnchor)
            self.text.setTextCursor(cursor)
            self.text.centerCursor()

    def _show_problem(self, text: str) -> None:
        self.problem.setText(text)
        self.problem.show()

    # --- the questions (replaced in tests) -------------------------------------------------

    def ask_unsaved(self) -> QMessageBox.StandardButton:
        return QMessageBox.question(
            self, "Save the project file?",
            f"{self.path.name} has changes typed in the File tab that are not saved.",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel)

    def ask_overwrite(self) -> bool:
        return QMessageBox.question(
            self, "Changed outside the File tab",
            f"{self.path.name} was changed since you started editing it here. Write your "
            "text over that change?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel
        ) == QMessageBox.StandardButton.Save

    # --- finding -----------------------------------------------------------------------------

    def focus_find(self) -> None:
        self.find.setFocus()
        self.find.selectAll()

    def _search(self, *_, keep: bool = False) -> None:
        """Mark every match of the find box's text; the current one is selected."""
        query = self.find.text()
        doc = self.text.document()
        flags = QTextDocument.FindFlag.FindCaseSensitively if self.case.isChecked() \
            else QTextDocument.FindFlag(0)
        where = self._matches[self._at].selectionStart() if keep and self._at >= 0 else None
        self._matches = []
        if query:
            cursor = doc.find(query, 0, flags)
            while not cursor.isNull() and len(self._matches) < MAX_MARKED:
                self._matches.append(cursor)
                cursor = doc.find(query, cursor, flags)
        if not self._matches:
            self._at = -1
        else:
            start = where if where is not None else self.text.textCursor().selectionStart()
            self._at = next((i for i, c in enumerate(self._matches)
                             if c.selectionStart() >= start), 0)
        self._mark(select=not keep)

    def step(self, by: int) -> None:
        """The next (``by`` 1) or previous (-1) match, round the end of the file."""
        if not self._matches:
            self._search()
            if not self._matches:
                return
        self._at = (self._at + by) % len(self._matches)
        self._mark(select=True)

    def _mark(self, select: bool) -> None:
        colour = QColor(self.palette().color(QPalette.ColorRole.Highlight))
        colour.setAlpha(70)
        marks = []
        for cursor in self._matches:
            mark = QTextEdit.ExtraSelection()
            mark.cursor = cursor
            mark.format.setBackground(colour)
            marks.append(mark)
        self.text.setExtraSelections(marks)
        if not self.find.text():
            self.count.setText("")
        elif not self._matches:
            self.count.setText("no match")
        else:
            more = "+" if len(self._matches) >= MAX_MARKED else ""
            self.count.setText(f"{self._at + 1} of {len(self._matches)}{more}")
        if select and self._at >= 0:
            self.text.setTextCursor(self._matches[self._at])
            self.text.centerCursor()
