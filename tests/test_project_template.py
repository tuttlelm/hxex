# SPDX-License-Identifier: GPL-3.0-or-later
"""A project file copied as the start of another project (hxex.project_edit.save_template, the
window's Save as template): settings and comments kept, the same data read from the copy's
folder, a new results folder, and the ties to the original's results and review cleared."""
import tomllib

import pytest

from hxex.config import ProjectError, load_project, template
from hxex.project_edit import save_template


def _elsewhere(folder):
    """A full location outside the project (with its drive on Windows: "/abs" is not one
    there), as a project file writes it."""
    return (folder.parent / "elsewhere" / "d109h.fasta").as_posix()


def _specexport(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SpecExport").mkdir()
    text = template().replace("overwrite = false ", "overwrite = true  ", 1)
    text += f"""
[review]
table = "user_review_table.csv"

[[review.exclude]]
sample = "WT"

[[review.exclude]]
sample = "D109H"

[analysis]
reference = "WT"

[analysis.fasta]
WT = "sequences/wt.fasta"      # the reference's own sequence
D109H = "{_elsewhere(folder)}"
"""
    path = folder / "hxex.toml"
    path.write_text("# my own note at the top\n" + text, encoding="utf-8")
    return path


def _data(path):
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_a_copy_in_another_folder_reads_the_same_data(tmp_path):
    source = _specexport(tmp_path / "first")
    target = tmp_path / "second" / "hxex_v2.toml"
    said, problem = save_template(source, target, "results_v2")
    assert problem is None
    copy, original = load_project(target), load_project(source)
    # the same folder, from elsewhere
    assert copy.input.data_dir.resolve() == original.input.data_dir.resolve()
    assert _data(target)["input"]["data_dir"] == "../first/SpecExport"
    fasta = _data(target)["analysis"]["fasta"]
    assert fasta == {"WT": "../first/sequences/wt.fasta",
                     "D109H": _elsewhere(tmp_path / "first")}  # a full location: kept
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# my own note at the top")  # comments kept
    assert "# the reference's own sequence" in text
    assert copy.output.bundle == tmp_path / "second" / "results_v2"
    assert not copy.output.overwrite
    assert copy.review is None or copy.review.table is None
    assert "exclude" not in _data(target).get("review", {})
    assert any("2 rule(s) removed" in line for line in said)
    assert any("overwrite cleared" in line for line in said)
    assert _data(target)["analysis"]["reference"] == "WT"  # other settings kept


def test_a_copy_beside_the_original_keeps_its_paths(tmp_path):
    source = _specexport(tmp_path)
    target = tmp_path / "variant.toml"
    said, _ = save_template(source, target, "results_variant")
    data = _data(target)
    assert data["input"]["data_dir"] == "SpecExport"
    assert data["analysis"]["fasta"]["WT"] == "sequences/wt.fasta"
    assert not any("data_dir" in line for line in said)
    assert source.read_text(encoding="utf-8").count("overwrite = true") == 1  # untouched


def test_scan_runs_keep_their_files_through_data_root(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    text = template("mzml", "trial").replace('data_root = "."', "# data_root = \".\"", 1)
    source = first / "hxex.toml"
    source.write_text(text, encoding="utf-8")
    target = tmp_path / "other" / "copy.toml"
    save_template(source, target, "results")
    data = _data(target)
    assert data["input"]["data_root"] == "../first"
    assert data["input"]["peptides"] == "../first/peptides.csv"
    assert (load_project(target).input.data_root.resolve()
            == load_project(source).input.data_root.resolve())


@pytest.mark.parametrize("bundle, match", [("results", "own results"),
                                           ("taken", "exists already"),
                                           (" ", "a name")])
def test_the_results_folder_must_be_new(tmp_path, bundle, match):
    source = _specexport(tmp_path)
    (tmp_path / "taken").mkdir()
    with pytest.raises(ProjectError, match=match):
        save_template(source, tmp_path / "copy.toml", bundle)
    assert not (tmp_path / "copy.toml").exists()


def test_the_copy_is_another_file(tmp_path):
    source = _specexport(tmp_path)
    with pytest.raises(ProjectError, match="another file"):
        save_template(source, source, "results_x")


def test_the_window_writes_the_template_and_opens_it(qtbot, tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox

    from hxex.gui.window import MainWindow

    source = _specexport(tmp_path)
    w = MainWindow(settings=QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat))
    w.ask_to_save = lambda: QMessageBox.StandardButton.Discard
    qtbot.addWidget(w)
    assert not w.project_page.template_button.isEnabled()
    w.open_project(source)
    assert w.project_page.template_button.isEnabled()
    w.ask_template_path = lambda: tmp_path / "next"  # .toml added
    w.ask_template_results = lambda target: f"results_{target.stem}"
    assert w.save_template_dialog()
    target = tmp_path / "next.toml"
    assert w.project == target.absolute()
    assert _data(target)["output"]["bundle"] == "results_next"
    shown = []
    w.tell = lambda title, text: shown.append(text)
    w.ask_template_path = lambda: target  # the project file open now
    assert not w.save_template_dialog() and "project file itself" in shown[-1]
    other = tmp_path / "other.toml"
    other.write_text("# someone's file\n", encoding="utf-8")
    w.ask_template_path = lambda: other  # a file there already: asked first
    w.ask_replace_file = lambda path: False
    assert not w.save_template_dialog()
    assert other.read_text(encoding="utf-8") == "# someone's file\n"
    w.ask_replace_file = lambda path: True
    w.ask_template_results = lambda target: "results_next"  # the open project's results
    assert not w.save_template_dialog() and "own results" in shown[-1]
