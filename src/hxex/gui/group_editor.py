# SPDX-License-Identifier: GPL-3.0-or-later
"""The project editor's pages for groups of samples (``[analysis.groups]``, docs/ANALYSIS.md
"Groups of samples"; PLAN Next steps 38, step 4; owner's layout, 2026-11-18): the
``[analysis.groups]`` page lists the groups, with buttons to add, rename, and remove one, and
each group has a page of its own in the list of tables, holding its reference, its samples (a
vial label beside each sample), and its own regions and sites, all on one page.

Every change is written at once to the project file, as the rest of the editor writes
(:mod:`hxex.project_edit`: :func:`~hxex.project_edit.add_group`,
:func:`~hxex.project_edit.rename_group`, :func:`~hxex.project_edit.remove_group`, and the group's
settings and lists of tables by their names, ``analysis.groups.B.reference``,
``[[analysis.groups.B.regions]]``). Adding the first group while ``[analysis]`` names a
reference or samples (refused together with groups) takes them into the new group in the same
change.
"""

from __future__ import annotations

import dataclasses
import logging
import tomllib

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .tips import tip

log = logging.getLogger(__name__)

GROUPS = "analysis.groups"


def file_groups(owner) -> dict[str, dict]:
    """The groups of the editor's project file as it is now: name -> its table (reference,
    samples, and its own regions and sites as written); empty when the file does not read."""
    if owner.path is None:
        return {}
    try:
        data = tomllib.loads(owner.path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return {}
    groups = (data.get("analysis") or {}).get("groups") or {}
    return {str(k): v for k, v in groups.items() if isinstance(v, dict)} \
        if isinstance(groups, dict) else {}


def pairs_of(samples) -> list[tuple[str, str]]:
    """A group's ``samples`` as (vial label, sample) pairs: a list's samples are their own
    labels."""
    if isinstance(samples, dict):
        return [(str(k), str(v)) for k, v in samples.items()]
    if isinstance(samples, list):
        return [(str(v), str(v)) for v in samples]
    return []


def samples_value(pairs: list[tuple[str, str]]):
    """The ``samples`` to write for (label, sample) pairs: a list when every label is its
    sample's name (or empty), else a table of label = sample."""
    pairs = [(label.strip(), sample.strip()) for label, sample in pairs if sample.strip()]
    if all(not label or label == sample for label, sample in pairs):
        return [sample for _, sample in pairs]
    return {label or sample: sample for label, sample in pairs}


def _names(owner) -> list[str]:
    """The samples of the results open (none without them)."""
    names = owner.sample_names
    try:
        return list(names() if callable(names) else names or [])
    except Exception:  # noqa: BLE001 - results that do not list their samples: typed instead
        return []


def _top_level(owner) -> tuple[str, list[str]]:
    """``[analysis]`` reference and samples as the file has them (refused with groups)."""
    ref = owner.values.get("analysis.reference")
    samples = owner.values.get("analysis.samples")
    ref = ref if isinstance(ref, str) else ""
    samples = [str(s) for s in samples] if isinstance(samples, list) else []
    if ref and ref not in samples:
        samples = [ref] + samples
    return ref, samples


class GroupsEditor(QWidget):
    """The ``[analysis.groups]`` page: one row per group (its reference, samples, and whether
    it has regions and sites of its own), with Add a group, Rename, and Remove. A row
    double-clicked opens the group's page."""

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.grid = QTableWidget(0, 5)
        self.grid.setHorizontalHeaderLabels(["group", "reference", "samples (vial label = sample)",
                                             "regions", "sites"])
        self.grid.horizontalHeader().setStretchLastSection(True)
        self.grid.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.grid.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.grid.cellDoubleClicked.connect(self._open)
        self.grid.setMinimumHeight(160)
        add = QPushButton("Add a group…")
        add.setToolTip(tip("a new group: its name, samples, and reference"))
        add.clicked.connect(self.ask_add)
        self.rename = QPushButton("Rename…")
        self.rename.setToolTip(tip("rename the selected group (its analysis folder is then "
                                   "results_analysis_NEW; the old folder stays)"))
        self.rename.clicked.connect(self.ask_rename)
        self.remove = QPushButton("Remove")
        self.remove.setToolTip(tip("remove the selected group, with its own regions and sites"))
        self.remove.clicked.connect(self.ask_remove)
        buttons = QHBoxLayout()
        for b in (add, self.rename, self.remove):
            buttons.addWidget(b)
        buttons.addStretch()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(buttons)
        layout.addWidget(self.grid)
        layout.addWidget(_note("Double-click a group, or choose it in the list on the left, to "
                               "change its reference, samples, regions, and sites."))
        self.fill()

    def fill(self) -> None:
        groups = file_groups(self.owner)
        row = self.grid.currentRow()
        self.grid.setRowCount(len(groups))
        for r, (name, item) in enumerate(groups.items()):
            pairs = pairs_of(item.get("samples"))
            cells = [name, str(item.get("reference", "")),
                     ", ".join(s if label == s else f"{label} = {s}" for label, s in pairs),
                     _own(item, "regions"), _own(item, "sites")]
            for c, text in enumerate(cells):
                self.grid.setItem(r, c, QTableWidgetItem(text))
        self.grid.resizeColumnsToContents()
        if 0 <= row < len(groups):
            self.grid.selectRow(row)
        self.rename.setEnabled(bool(groups))
        self.remove.setEnabled(bool(groups))

    def selected(self) -> str | None:
        row = self.grid.currentRow()
        item = self.grid.item(row, 0) if row >= 0 else None
        return item.text() if item is not None else None

    def _open(self, row: int, _column: int) -> None:
        item = self.grid.item(row, 0)
        if item is not None:
            self.owner.show_table(f"{GROUPS}.{item.text()}")

    # --- changes (the dialogs gather; these write) ------------------------------------------

    def add(self, name: str, reference: str, samples) -> bool:
        """Write the group ``name`` (False, said in the editor, when refused). ``[analysis]``
        reference and samples, refused together with groups, are switched off with it."""
        from ..project_edit import add_group

        take = any(self.owner.values.get(f"analysis.{k}") is not None
                   for k in ("reference", "samples"))
        return self.owner.run(add_group, name, reference, samples, take)

    def ask_add(self) -> None:
        dialog = AddGroupDialog(self.owner, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name, reference, samples = dialog.values()
            if self.add(name, reference, samples):
                self.owner.show_table(f"{GROUPS}.{name}")
        dialog.deleteLater()

    def rename_selected(self, new: str) -> bool:
        from ..project_edit import rename_group

        old = self.selected()
        return old is not None and self.owner.run(rename_group, old, new)

    def ask_rename(self) -> None:
        old = self.selected()
        if old is None:
            return
        new, ok = QInputDialog.getText(self, "Rename the group", f"New name for group {old}:",
                                       text=old)
        if ok and new.strip() and new.strip() != old:
            self.rename_selected(new.strip())

    def remove_selected(self) -> bool:
        from ..project_edit import remove_group

        name = self.selected()
        return name is not None and self.owner.run(remove_group, name)

    def ask_remove(self) -> None:
        name = self.selected()
        if name is None:
            return
        answer = QMessageBox.question(
            self, "Remove the group?",
            f"Remove group {name} from the project file, with its own regions and sites? Its "
            "analysis folder (results_analysis_" + name + ") is not touched.")
        if answer == QMessageBox.StandardButton.Yes:
            self.remove_selected()


def _own(item: dict, kind: str) -> str:
    value = item.get(kind)
    if value is None:
        return "shared"
    if value is False:
        return "none"
    return f"{len(value)} of its own" if isinstance(value, list) else "of its own"


class AddGroupDialog(QDialog):
    """A new group: its name, its samples (in order; one of the results' or typed), and its
    reference. With ``[analysis]`` reference or samples in the file, they start the new group
    (and are taken out of ``[analysis]``, where groups refuse them)."""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        from .project_editor import NoWheelComboBox, OrderedList

        self.setWindowTitle("Add a group")
        self.name = QLineEdit()
        self.name.setPlaceholderText("a name: letters, digits, _ and - (it names the group's "
                                     "analysis folder)")
        self.samples = OrderedList(lambda: _names(owner))
        self.reference = NoWheelComboBox()
        self.samples.changed.connect(self._offer)
        ref, top = _top_level(owner)
        form = QFormLayout()
        form.addRow("name", self.name)
        form.addRow("samples", self.samples)
        form.addRow("reference", self.reference)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        if ref or top:
            self.samples.set_items(top)
            layout.addWidget(_note(
                "[analysis] reference and samples start this group: with groups each group "
                "names its own, so they are taken out of [analysis] when the group is added."))
        layout.addWidget(_note("Vial labels (to line the groups up across the analysis) are "
                               "set on the group's page once it is added."))
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._offer(self.samples.items(), ref)

    def _offer(self, items: list[str], keep: str | None = None) -> None:
        keep = keep or self.reference.currentText()
        self.reference.clear()
        self.reference.addItems(items)
        if keep in items:
            self.reference.setCurrentText(keep)

    def values(self) -> tuple[str, str, list[str]]:
        return (self.name.text().strip(), self.reference.currentText(), self.samples.items())

    def accept(self) -> None:
        name, ref, samples = self.values()
        if name and samples and ref:
            super().accept()


class SamplesGrid(QWidget):
    """A group's samples in order, a vial label beside each: the label names the vial (the
    condition) the sample was measured in, the same label in every group that holds that vial
    ("A+B" in group A and in group B); left as the sample's own name, the samples are written
    as a list."""

    def __init__(self, page):
        super().__init__()
        from .project_editor import NoWheelComboBox

        self.page = page
        self.grid = QTableWidget(0, 2)
        self.grid.setHorizontalHeaderLabels(["vial label", "sample"])
        self.grid.horizontalHeader().setStretchLastSection(True)
        self.grid.setMinimumHeight(140)
        self.grid.cellChanged.connect(self._edited)
        self.add_box = NoWheelComboBox()
        self.add_box.setEditable(True)
        self.add_box.setToolTip(tip("a sample to add: one of the results', or typed"))
        add = QPushButton("Add")
        add.clicked.connect(self._add)
        self.remove = QPushButton("Remove the selected sample")
        self.remove.clicked.connect(self._remove)
        self.up = QPushButton("↑")
        self.up.setToolTip(tip("move the selected sample up: the figures follow this order"))
        self.up.clicked.connect(lambda: self._move(-1))
        self.down = QPushButton("↓")
        self.down.setToolTip(tip("move the selected sample down: the figures follow this order"))
        self.down.clicked.connect(lambda: self._move(1))
        line = QHBoxLayout()
        line.addWidget(self.add_box, 1)
        for b in (add, self.remove, self.up, self.down):
            line.addWidget(b)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(line)
        layout.addWidget(self.grid)

    def pairs(self) -> list[tuple[str, str]]:
        def text(r, c):
            item = self.grid.item(r, c)
            return item.text().strip() if item is not None else ""
        return [(text(r, 0), text(r, 1)) for r in range(self.grid.rowCount())]

    def set_pairs(self, pairs: list[tuple[str, str]]) -> None:
        row = self.grid.currentRow()
        self.grid.blockSignals(True)
        self.grid.setRowCount(len(pairs))
        for r, (label, sample) in enumerate(pairs):
            self.grid.setItem(r, 0, QTableWidgetItem(label))
            self.grid.setItem(r, 1, QTableWidgetItem(sample))
        self.grid.blockSignals(False)
        if 0 <= row < len(pairs):
            self.grid.setCurrentCell(row, max(self.grid.currentColumn(), 0))
        held = {s for _, s in pairs}
        self.add_box.blockSignals(True)
        self.add_box.clear()
        self.add_box.addItems([n for n in _names(self.page.owner) if n not in held])
        self.add_box.setCurrentText("")
        self.add_box.blockSignals(False)
        self.remove.setEnabled(bool(pairs))
        for b in (self.up, self.down):
            b.setEnabled(len(pairs) > 1)

    def _write(self, pairs: list[tuple[str, str]]) -> None:
        self.page.write_samples(pairs)

    def _edited(self, _row: int, _column: int) -> None:
        self._write(self.pairs())

    def _add(self) -> None:
        name = self.add_box.currentText().strip()
        pairs = self.pairs()
        if name and name not in {s for _, s in pairs}:
            self._write(pairs + [(name, name)])

    def _remove(self) -> None:
        row = self.grid.currentRow()
        pairs = self.pairs()
        if 0 <= row < len(pairs):
            del pairs[row]
            self._write(pairs)

    def _move(self, step: int) -> None:
        row = self.grid.currentRow()
        pairs = self.pairs()
        to = row + step
        if row < 0 or not 0 <= to < len(pairs):
            return
        pairs[row], pairs[to] = pairs[to], pairs[row]
        self._write(pairs)
        self.grid.setCurrentCell(to, max(self.grid.currentColumn(), 0))


class GroupPage(QWidget):
    """One group's page (``[analysis.groups.NAME]``): its reference, its samples with their
    vial labels, and its own regions and sites (each list replacing the shared one of its kind
    for this group; a group without its own takes the shared ones)."""

    def __init__(self, owner, name: str, tables: dict):
        super().__init__()
        from .project_editor import ListEditor, NoWheelComboBox, plain

        self.owner, self.name = owner, name
        self.reference = NoWheelComboBox()
        self.reference.setToolTip(tip(plain(tables["analysis.groups.NAME"].settings[0].text)))
        self.reference.activated.connect(self._chosen)
        self.samples = SamplesGrid(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("reference", self.reference)
        form.addRow("samples", self.samples)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(form)
        layout.addWidget(_note(
            "A vial label names the vial (the condition) a sample was measured in, the same "
            "label in every group that holds that vial (\"A+B\" in group A and in group B): the "
            "labels line the groups up across the analysis, and a label naming two groups (split "
            "at +) tells hxex which proteins share that vial. Left as the sample's name, the "
            "samples are written as a list."))
        self.lists = {}
        self.states = {}
        for kind in ("regions", "sites"):
            generic = tables[f"analysis.groups.NAME.{kind}"]
            table = dataclasses.replace(generic, name=f"{GROUPS}.{name}.{kind}")
            layout.addWidget(_bold(f"Own {kind}"))
            layout.addWidget(_note(plain(generic.intro)))
            self.states[kind] = _note("")
            layout.addWidget(self.states[kind])
            self.lists[kind] = ListEditor(owner, table)
            layout.addWidget(self.lists[kind])
        self.fill()

    def item(self) -> dict:
        return file_groups(self.owner).get(self.name, {})

    def fill(self) -> None:
        item = self.item()
        pairs = pairs_of(item.get("samples"))
        self.samples.set_pairs(pairs)
        names = [s for _, s in pairs]
        ref = str(item.get("reference", ""))
        self.reference.blockSignals(True)
        self.reference.clear()
        self.reference.addItems(names)
        if ref and ref not in names:  # as the file has it (hxex check says what is wrong)
            self.reference.addItem(ref)
        self.reference.setCurrentText(ref)
        self.reference.blockSignals(False)
        shared = {k: len(self.owner_list(k)) for k in ("regions", "sites")}
        for kind, editor in self.lists.items():
            value = item.get(kind)
            if value is False:
                words = (f"{kind} = false in the file: none for this group (a text editor "
                         "changes it).")
            elif value is None:
                words = (f"None of its own: the group takes the shared [[analysis.{kind}]] "
                         f"({shared[kind]}). Adding one here replaces them for this group.")
            else:
                words = f"{len(value)} of its own, in place of the shared [[analysis.{kind}]]."
            self.states[kind].setText(words)
            editor.setEnabled(value is not False)
            editor.fill()

    def owner_list(self, kind: str) -> list:
        if self.owner.path is None:
            return []
        try:
            data = tomllib.loads(self.owner.path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
            return []
        value = (data.get("analysis") or {}).get(kind)
        return value if isinstance(value, list) else []

    def _chosen(self, _index: int) -> None:
        from ..config import toml_value

        self.owner.write(f"{GROUPS}.{self.name}.reference",
                         toml_value(self.reference.currentText()))

    def write_samples(self, pairs: list[tuple[str, str]]) -> None:
        from ..project_edit import samples_text

        value = samples_value(pairs)
        if not value:
            self.owner.problem.setText("A group needs at least one sample (Remove the group on "
                                       "the [analysis.groups] page instead).")
            self.owner.problem.show()
            self.fill()
            return
        self.owner.write(f"{GROUPS}.{self.name}.samples", samples_text(value))
        self.fill()  # as the file has it now (a refused change: as it was)


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _bold(text: str) -> QLabel:
    label = QLabel(text)
    font = label.font()
    font.setBold(True)
    label.setFont(font)
    return label
