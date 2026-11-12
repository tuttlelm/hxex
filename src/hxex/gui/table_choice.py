# SPDX-License-Identifier: GPL-3.0-or-later
"""Where a table the window writes for the first time goes (owner, 2026-11-11).

The window writes two tables of the user's when the project file names none: the elution
windows (``[input] rt_elution_windows``, from the elution tab's Save) and the review table
(``[review] table``, from the first review decision). Both once went to a fixed name beside the
project file without asking, so two project files in one folder (two arms of one dataset, say)
could end up writing into one table, each naming it. Now the first write asks where, as the
Tables view's New does: the name suggested carries the project file's name (unless it is
``hxex.toml``), and a file that exists already is said to exist, with the project files that
name it, before anything is added to it.

:func:`suggested_name` and :func:`named_by_others` import no Qt; :func:`ask_table` shows the
dialogs.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_PROJECT = "hxex"  # the project file hxex init writes: its tables keep their plain names


def suggested_name(project: Path, base: str) -> str:
    """The name suggested for a table ``base`` (``rt_elution_windows.csv``) of the project file
    ``project``: ``base`` for ``hxex.toml``, else the project file's name in front
    (``hxex_findpeak_rt_elution_windows.csv``), so two project files in one folder get tables of
    their own."""
    stem = Path(project).stem
    return base if stem == DEFAULT_PROJECT else f"{stem}_{base}"


def named_by_others(path: Path, project: Path, setting: str) -> list[str]:
    """The other project files beside ``project`` (``*.toml``) whose ``setting``
    (``input.rt_elution_windows``) names the file ``path``; a project file that does not read
    is passed over."""
    from ..project_edit import live_settings

    target = Path(path).resolve()
    out = []
    for other in sorted(Path(project).parent.glob("*.toml")):
        if other.resolve() == Path(project).resolve():
            continue
        try:
            value = live_settings(other.read_text(encoding="utf-8-sig"))[0].get(setting)
        except Exception:  # noqa: BLE001 - not a project file, or one that does not read
            continue
        if not value:
            continue
        named = Path(str(value))
        named = named if named.is_absolute() else other.parent / named
        if named.resolve() == target:
            out.append(other.name)
    return out


def ask_table(parent, project: Path, setting: str, base: str, what: str) -> Path | None:
    """Ask where the project's first ``what`` (``elution-window table``) goes: a save dialog
    beside the project file with :func:`suggested_name` filled in. A file that exists is said
    to, with the project files that name it, and its rows are added to only on Yes. None when
    the user cancels."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    parent = dialog_parent(parent)
    project = Path(project)
    name, _ = QFileDialog.getSaveFileName(
        parent, f"Where should {project.name}'s {what} go?",
        str(project.parent / suggested_name(project, base)), "tables (*.csv)",
        options=QFileDialog.Option.DontConfirmOverwrite)
    if not name:
        return None
    path = Path(name)
    if path.suffix.lower() != ".csv":
        path = path.with_name(path.name + ".csv")
    if path.exists():
        others = named_by_others(path, project, setting)
        text = (f"{path.name} exists already"
                + (f" and is the {what} of {', '.join(others)}" if others else "")
                + f". Add {project.name}'s rows to it and name it in {project.name}?")
        answer = QMessageBox.question(parent, f"{path.name} exists", text)
        if answer != QMessageBox.StandardButton.Yes:
            return None
    return path


def dialog_parent(owner):
    """The widget a dialog of ``owner`` belongs to: ``owner`` when it is a widget, else the first
    widget among its parents (the window editor is a QObject over its panel), else None; Qt's
    dialogs refuse any other parent (TypeError)."""
    from PySide6.QtWidgets import QWidget

    while owner is not None and not isinstance(owner, QWidget):
        owner = owner.parent() if hasattr(owner, "parent") else None
    return owner
