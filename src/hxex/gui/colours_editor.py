# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``[analysis.colours]`` page of the project editor: a row per sample, each with a list of
colours that starts with its default (owner, 2026-11-18).

The samples are the window's (those of the results open, else ``[input] samples``, else those
the data hold; :meth:`hxex.gui.window.MainWindow.sample_names`) in ``[analysis] samples``' order,
then any other name the file gives a colour (a ``run:sample``, a modified state, a name typed
here). Each list holds **default** (the colour the figures give the sample when the file names
none, worked out as :func:`hxex.report.coverage.sample_styles` does, so it follows the order and
the colours chosen for the others), the palette's colours, the file's own value when it is
another, **Other…** (a colour picker) and **Type a colour…**. Choosing default removes the
sample's line from the file; anything else writes it. Each entry shows its colour, so the box
shows the colour the figures will use.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .tips import tip

TABLE = "analysis.colours"
OTHER = "Other…"
TYPED = "Type a colour…"
_SWATCH = 14  # pixels


def palette_names() -> list[tuple[str, str]]:
    """(name as written in the file, hex) of hxex's sample colours, in their order."""
    from matplotlib.colors import TABLEAU_COLORS, to_hex

    from ..report.coverage import SAMPLE_STYLES

    named = {to_hex(v): k for k, v in TABLEAU_COLORS.items()}
    named["#000000"] = "black"
    return [(named.get(to_hex(c), c), to_hex(c)) for c, _, _ in SAMPLE_STYLES]


def default_colours(order: list[str], chosen: dict[str, str]) -> dict[str, str]:
    """Each sample's colour in the figures (hex), as the analysis gives them: ``chosen`` (the
    file's) where it names one, the defaults for the rest in ``order``."""
    from matplotlib.colors import is_color_like, to_hex

    from ..report.coverage import sample_styles

    usable = {k: v for k, v in chosen.items() if isinstance(v, str) and is_color_like(v)}
    return {name: to_hex(style.colour)
            for name, style in sample_styles(order, None, usable).items()}


def _icon(colour: str | None) -> QIcon:
    pixmap = QPixmap(_SWATCH, _SWATCH)
    if colour is None:
        pixmap.fill(Qt.GlobalColor.transparent)
    else:
        pixmap.fill(QColor(colour))
    return QIcon(pixmap)


def _hex(value) -> str | None:
    from matplotlib.colors import is_color_like, to_hex

    return to_hex(value) if isinstance(value, str) and is_color_like(value) else None


class ColoursEditor(QWidget):
    """The rows of ``[analysis.colours]`` (``owner`` is the project editor: its values, its
    ``write`` and its ``sample_names``)."""

    def __init__(self, owner, table=None):
        super().__init__()
        self.owner = owner
        self.added: list[str] = []  # names typed here, shown until they get a colour
        self._message = ""
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: palette(placeholder-text);")
        self.grid = QTableWidget(0, 2)
        self.grid.setHorizontalHeaderLabels(["sample", "colour"])
        self.grid.horizontalHeader().setStretchLastSection(True)
        self.grid.verticalHeader().hide()
        self.grid.setMinimumHeight(160)
        self.name = QLineEdit()
        self.name.setPlaceholderText("another name (run:sample, a modified state, ...)")
        add = QPushButton("Add")
        add.setToolTip(tip("a row for a name the list does not hold; it is written once it "
                           "has a colour"))
        add.clicked.connect(self.add)
        self.name.returnPressed.connect(self.add)
        line = QHBoxLayout()
        line.addWidget(self.name, 1)
        line.addWidget(add)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.note)
        layout.addWidget(self.grid)
        layout.addLayout(line)
        self.fill()

    # --- what is shown -----------------------------------------------------------------------

    def chosen(self) -> dict[str, object]:
        """The file's colours: name -> value."""
        prefix = TABLE + "."
        return {k[len(prefix):]: v for k, v in self.owner.values.items()
                if k.startswith(prefix) and "." not in k[len(prefix):]}

    def samples(self) -> list[str]:
        """The samples in ``[analysis] samples``' order, the others after them."""
        found = list(self.owner.sample_names() if self.owner.sample_names else [])
        order = self.owner.values.get("analysis.samples")
        order = [s for s in order if isinstance(s, str)] if isinstance(order, list) else []
        return list(dict.fromkeys([*order, *found]))

    def fill(self) -> None:
        chosen = self.chosen()
        samples = self.samples()
        rows = list(dict.fromkeys([*samples, *chosen, *self.added]))
        self.added = [n for n in self.added if n not in chosen]
        defaults = default_colours(samples, {k: v for k, v in chosen.items()
                                             if isinstance(v, str)}) if samples else {}
        # what each sample would take without its own colour: the others' choices kept
        alone = {name: default_colours(samples, {k: v for k, v in chosen.items()
                                                 if k != name and isinstance(v, str)})
                 .get(name) for name in samples}
        self.note.setText(
            "Each sample's colour in every figure. Default is the colour hxex gives it (the "
            "first sample of [analysis] samples black, the others in turn); choosing it removes "
            "the sample's line from the project file." if samples else
            "No samples found yet (no results open, no [input] samples, and no data read): "
            "add a name below, or set the input first.")
        self.grid.setRowCount(len(rows))
        for r, name in enumerate(rows):
            key = QTableWidgetItem(name)
            key.setFlags(key.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if name not in samples:
                key.setToolTip(tip("not one of the samples found: a run:sample, a modified "
                                   "state, or a name of results not open"))
            self.grid.setItem(r, 0, key)
            self.grid.setCellWidget(r, 1, self._box(name, chosen.get(name), alone.get(name),
                                                    defaults.get(name)))
        self.grid.resizeColumnToContents(0)

    def _box(self, name: str, value, default: str | None, now: str | None) -> QComboBox:
        from .project_editor import NoWheelComboBox

        box = NoWheelComboBox()
        box.setProperty("sample", name)
        names = dict((h, n) for n, h in palette_names())
        shown = names.get(default, default) if default else None
        box.addItem(_icon(default), f"default ({shown})" if shown else "default", None)
        for label, colour in palette_names():
            box.addItem(_icon(colour), label, label)
        if value is not None and not any(box.itemData(i) == value for i in range(box.count())):
            box.addItem(_icon(_hex(value)), str(value), value)  # the file's own value
        box.addItem(OTHER, OTHER)
        box.addItem(TYPED, TYPED)
        at = 0 if value is None else next(
            (i for i in range(box.count()) if box.itemData(i) == value), 0)
        box.setCurrentIndex(at)
        box.setToolTip(tip(f"{name}: the figures draw it in {names.get(now, now)}" if now
                           else f"{name}: its colour"))
        box.activated.connect(lambda index, b=box: self._chosen(b, index))
        return box

    # --- changes -----------------------------------------------------------------------------

    def _chosen(self, box: QComboBox, index: int) -> None:
        # after the box's own signal: writing rebuilds the rows, the box among them, and a
        # widget deleted inside its own slot can take the window down
        name, data = box.property("sample"), box.itemData(index)
        QTimer.singleShot(0, self, lambda: self.choose(name, data))

    def choose(self, name: str, data) -> None:
        """``name``'s list set to the entry holding ``data`` (None: default; OTHER and TYPED
        ask for the colour)."""
        self._message = ""
        if data == OTHER:
            value = self.ask_colour(name, self.chosen().get(name))
        elif data == TYPED:
            value = self.ask_text(name, self.chosen().get(name))
        else:
            value = data
        if data in (OTHER, TYPED) and value is None:
            self.fill()  # cancelled or refused: the list back as it was
            if self._message:
                self.note.setText(self._message)
            return
        self.write(name, value)

    def write(self, name: str, value: str | None) -> None:
        """Give ``name`` the colour ``value`` (None: back to its default, the line removed)."""
        from ..project_edit import key_text
        from .project_editor import _quoted

        if value is None and name not in self.chosen():
            return
        self.owner.write(f"{TABLE}.{key_text(name)}", None if value is None else _quoted(value))
        self.fill()  # also when the write was refused: the boxes show the file

    def add(self) -> None:
        name = self.name.text().strip()
        if name and name not in self.added:
            self.added.append(name)
            self.name.clear()
            self.fill()

    # --- the questions (replaced in tests) ---------------------------------------------------

    def ask_colour(self, name: str, value) -> str | None:
        start = QColor(_hex(value) or "#1f77b4")
        colour = QColorDialog.getColor(start, self, f"{name}'s colour")
        return colour.name() if colour.isValid() else None

    def ask_text(self, name: str, value) -> str | None:
        from matplotlib.colors import is_color_like

        text, ok = QInputDialog.getText(
            self, f"{name}'s colour", "Any colour matplotlib knows (tab:red, #1f77b4, navy):",
            QLineEdit.EchoMode.Normal, str(value or ""))
        text = text.strip()
        if not ok or not text:
            return None
        if not is_color_like(text):
            self._message = f"{text!r} is not a colour matplotlib knows: nothing written."
            return None
        return text
