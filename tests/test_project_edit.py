# SPDX-License-Identifier: GPL-3.0-or-later
"""hxex set: one setting of a project file changed in place, comments and layout kept."""

import tomllib

import pytest

from hxex.cli import main
from hxex.config import ProjectError, load_project, template
from hxex.project_edit import set_setting, value_text


@pytest.fixture
def project(tmp_path):
    (tmp_path / "SpecExport").mkdir()
    path = tmp_path / "hxex.toml"
    path.write_text(template(), encoding="utf-8")
    return path


def _line(path, start):
    return next(line for line in path.read_text().splitlines() if line.startswith(start))


def test_a_switched_off_setting_is_switched_on_with_its_heading(project):
    before = project.read_text().splitlines()
    change = set_setting(project, "run.controls.d2o_fraction", "0.85")
    after = project.read_text().splitlines()
    assert change.before is None and change.after == "0.85" and change.heading_on == "run.controls"
    assert "switched on the heading [run.controls]" in change.describe()
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(before) == len(after) and len(changed) == 2  # the heading and the setting
    assert after[change.line - 1].startswith("d2o_fraction = 0.85")
    old = before[change.line - 1]
    assert after[change.line - 1].index("#") == old.index("#", 2)  # the comment stays in place
    assert load_project(project).run.controls.d2o_fraction == 0.85
    # the other d2o_fraction lines, of other tables, are untouched
    assert sum(line.lstrip("# ").startswith("d2o_fraction") for line in after) == sum(
        line.lstrip("# ").startswith("d2o_fraction") for line in before)


def test_a_live_setting_is_replaced_and_its_comment_kept(project):
    old = _line(project, "overwrite = ")
    change = set_setting(project, "output.overwrite", "true")
    new = _line(project, "overwrite = ")
    assert (change.before, change.after) == ("false", "true")
    assert new.split("#", 1)[1] == old.split("#", 1)[1]
    assert load_project(project).output.overwrite is True


def test_setting_the_same_value_again_changes_nothing(project):
    set_setting(project, "run.bootstrap.n_boot", "50")
    text = project.read_bytes()
    change = set_setting(project, "run.bootstrap.n_boot", "50")
    assert change.unchanged and "nothing changed" in change.describe()
    assert project.read_bytes() == text


def test_default_switches_the_line_off_again_and_keeps_the_heading(project):
    original = _line(project, "# n_boot = ")
    set_setting(project, "run.bootstrap.n_boot", "50")
    change = set_setting(project, "run.bootstrap.n_boot", None)
    assert (change.before, change.after) == ("50", None)
    assert _line(project, "# n_boot = ") == original  # the template's line, its default again
    assert change.bare_heading == "run.bootstrap" and "stays on" in change.describe()
    data = tomllib.loads(project.read_text())
    assert data["run"]["bootstrap"] == {}
    assert set_setting(project, "run.bootstrap.n_boot", None).unchanged


def test_text_needs_no_quotes_and_typed_values_stay_typed():
    assert value_text("0.85") == "0.85"
    assert value_text("true") == "true"
    assert value_text("[1, 2]") == "[1, 2]"
    assert value_text('"WT"') == '"WT"'
    assert value_text("controls") == '"controls"'
    assert value_text("20 min") == '"20 min"'
    assert value_text(r"C:\data\SpecExport") == r'"C:\\data\\SpecExport"'
    assert value_text(r'"C:\data"') == r'"C:\\data"'  # quoted, but not valid TOML as it is
    with pytest.raises(ProjectError, match="no value"):
        value_text("  ")


def test_a_value_the_checks_refuse_leaves_the_file_as_it_was(project):
    text = project.read_bytes()
    with pytest.raises(ProjectError, match=r"unchanged.*pick must be one of"):
        set_setting(project, "run.picker.pick", "nonsense")
    with pytest.raises(ProjectError, match="expected a whole number"):
        set_setting(project, "run.bootstrap.n_boot", "fifty")
    assert project.read_bytes() == text
    assert not list(project.parent.glob("*.hxex-set"))


def test_names_that_are_not_settings_are_refused_with_a_suggestion(project):
    with pytest.raises(ProjectError, match="did you mean run.controls.d2o_fraction"):
        set_setting(project, "run.controls.d2o_frac", "0.85")
    with pytest.raises(ProjectError, match="table, not a setting"):
        set_setting(project, "run.fit", "1")
    with pytest.raises(ProjectError, match="list of tables"):
        set_setting(project, "input.runs.path", "a.mzML")
    with pytest.raises(ProjectError, match="not a setting name"):
        set_setting(project, "run controls", "1")


def test_a_short_file_gets_the_setting_or_its_table_added(tmp_path):
    (tmp_path / "SpecExport").mkdir()
    path = tmp_path / "hxex.toml"
    path.write_text('project = "p"\n\n[input]\nformat = "specexport"\ndata_dir = "SpecExport"\n'
                    "# a comment of the next part\n\n[output]\nbundle = \"results\"\n")
    set_setting(path, "input.time_scale", "order")
    set_setting(path, "run.bootstrap.n_boot", "30")
    lines = path.read_text().splitlines()
    assert lines[lines.index('data_dir = "SpecExport"') + 1] == 'time_scale = "order"'
    assert lines[-2:] == ["[run.bootstrap]", "n_boot = 30"]
    project = load_project(path)
    assert project.input.time_scale == "order" and project.run.bootstrap.n_boot == 30


def test_line_endings_and_a_byte_order_mark_are_kept(project):
    text = project.read_text().replace("\n", "\r\n")
    project.write_bytes(b"\xef\xbb\xbf" + text.encode())
    set_setting(project, "run.controls.d2o_fraction", "0.85")
    raw = project.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert raw.count(b"\n") == raw.count(b"\r\n")
    assert load_project(project).run.controls.d2o_fraction == 0.85


def test_a_problem_the_file_already_had_does_not_stop_a_change(project):
    text = project.read_text().replace('pick = "peak"', 'pick = "bogus"')
    assert 'pick = "bogus"' in text
    project.write_text(text)
    set_setting(project, "run.bootstrap.n_boot", "30")  # unrelated: written
    assert "n_boot = 30" in project.read_text()
    with pytest.raises(ProjectError, match="hxex check"):
        project.write_text("project = [\n")
        set_setting(project, "project", "p")


def test_the_command(project, capsys):
    folder = str(project)
    assert main(["set", folder, "run.controls.undeut_at", "20", "min"]) == 0
    assert '-> "20 min"' in capsys.readouterr().out
    assert load_project(project).run.controls.undeut_at == 1200
    assert main(["set", folder, "run.controls.undeut_at", "--default"]) == 0
    assert main(["set", folder, "run.bootstrap.n_boot"]) == 2
    assert main(["set", folder, "run.bootstrap.n_boot", "5", "--default"]) == 2
    assert main(["set", folder, "run.picker.pick", "nonsense"]) == 1


# --- for hxex's window: what a file sets, and stages switched on and off ------------------------

def test_live_settings_lists_what_the_file_sets(project):
    from hxex.project_edit import live_settings

    values, tables = live_settings(project.read_text())
    assert values["input.format"] == "specexport" and "run.bootstrap.n_boot" not in values
    assert {"input", "run", "run.picker", "run.registry.residues"} <= tables
    assert "run.bootstrap" not in tables  # its heading is off in the template


def test_a_stage_is_switched_on_and_off_with_its_settings_and_tables(project):
    from hxex.project_edit import file_problem, live_settings, switch_table

    assert "switched on" in switch_table(project, "run.bootstrap", True)
    assert "run.bootstrap" in live_settings(project.read_text())[1]
    assert "already on" in switch_table(project, "run.bootstrap", True)
    set_setting(project, "run.bootstrap.n_boot", "50")
    switch_table(project, "analysis", True)
    set_setting(project, "analysis.filters.replicate_outlier", "4")
    # off: the heading, its settings and the tables under it get a # in front
    said = switch_table(project, "analysis", False)
    assert "[analysis]" in said and "[analysis.filters]" in said
    values, tables = live_settings(project.read_text())
    assert "analysis" not in tables and "analysis.filters.replicate_outlier" not in values
    assert values["run.bootstrap.n_boot"] == 50  # the other stage is untouched
    switch_table(project, "run.bootstrap", False)
    assert "run.bootstrap" not in live_settings(project.read_text())[1]
    assert file_problem(project) is None
    assert "# n_boot = 50" in project.read_text()  # kept, switched off
    with pytest.raises(ProjectError, match="cannot be switched"):
        switch_table(project, "run.picker", False)


def test_a_missing_stage_heading_is_added_at_the_end(tmp_path):
    from hxex.project_edit import live_settings, switch_table

    path = tmp_path / "hxex.toml"
    path.write_text('project = "p"\n[analysis]\nreference = "WT"\n')
    switch_table(path, "analysis.filters", True)
    assert path.read_text().endswith("\n\n[analysis.filters]\n")
    assert "analysis.filters" in live_settings(path.read_text())[1]


def test_file_problem_says_what_check_would(project):
    from hxex.project_edit import file_problem

    project.write_text(project.read_text().replace('format = "specexport"', 'format = "csv"'))
    assert "csv" in file_problem(project)
    project.write_text("[input\n")
    assert "not valid TOML" in file_problem(project)


def test_an_unfinished_file_is_written_only_in_the_windows_mode(project):
    """hxex set refuses a change that leaves something else needed; the window (unfinished)
    writes it and file_problem says what is still needed. A bad value of the changed setting
    itself is refused either way."""
    from hxex.project_edit import file_problem, set_settings

    change = [("input.data_dir", None), ("input.format", "mzml")]
    with pytest.raises(ProjectError, match="peptides"):
        set_settings(project, change)
    assert 'format = "specexport"' in project.read_text()
    set_settings(project, change, unfinished=True)
    assert 'format = "mzml"' in project.read_text()
    assert "peptides" in file_problem(project)
    with pytest.raises(ProjectError, match="n_boot"):
        set_settings(project, [("run.bootstrap.n_boot", "many")], unfinished=True)


# --- lists of tables, entry by entry (hxex's window) ---------------------------------------------

@pytest.fixture
def scan_project(tmp_path):
    path = tmp_path / "hxex.toml"
    path.write_text(template("mzml"), encoding="utf-8")
    return path


def test_an_entry_of_a_list_is_changed_in_its_own_lines(scan_project):
    from hxex.project_edit import list_entries, set_entry

    text = scan_project.read_text()
    runs = list_entries(text, "input.runs")
    assert [r["path"] for r in runs] == ["UN_01.mzML", "D_60s_01.mzML"]
    set_entry(scan_project, "input.runs", 1, "rep", "2")
    set_entry(scan_project, "input.runs", 0, "sample", "mutant")  # text, no quotes needed
    runs = list_entries(scan_project.read_text(), "input.runs")
    assert runs[1]["rep"] == 2 and runs[0]["sample"] == "mutant" and runs[1]["sample"] == "WT"
    assert scan_project.read_text().count("# one [[input.runs]] per file.") == 1  # comments kept
    set_entry(scan_project, "input.runs", 0, "rep", None)  # switched off
    assert "rep" not in list_entries(scan_project.read_text(), "input.runs")[0]
    with pytest.raises(ProjectError, match="no entry 3"):
        set_entry(scan_project, "input.runs", 2, "rep", "1")
    with pytest.raises(ProjectError, match="did you mean sample"):
        set_entry(scan_project, "input.runs", 0, "samples", "x")
    with pytest.raises(ProjectError, match="rep"):
        set_entry(scan_project, "input.runs", 0, "rep", "first")


def test_entries_are_added_and_removed(scan_project):
    from hxex.project_edit import add_entry, file_problem, list_entries, remove_entry

    add_entry(scan_project, "input.runs", {"path": "D_5min.mzML", "sample": "WT", "time": "300"})
    runs = list_entries(scan_project.read_text(), "input.runs")
    assert len(runs) == 3 and runs[2] == {"path": "D_5min.mzML", "sample": "WT", "time": 300}
    # a list the file does not have yet: at the end of its table's settings, which stay its
    add_entry(scan_project, "input.exclude", {"sample": "WT", "note": "bad"})
    data = tomllib.loads(scan_project.read_text())
    assert data["input"]["exclude"] == [{"sample": "WT", "note": "bad"}]
    assert data["input"]["format"] == "mzml" and len(data["input"]["runs"]) == 3
    remove_entry(scan_project, "input.runs", 0)
    assert [r["path"] for r in list_entries(scan_project.read_text(), "input.runs")] == [
        "D_60s_01.mzML", "D_5min.mzML"]
    assert file_problem(scan_project) is None or "peptides" in file_problem(scan_project)
    # the last run: hxex set's rule refuses it; the window's writes it and says what is needed
    remove_entry(scan_project, "input.runs", 0)
    with pytest.raises(ProjectError, match=r"input\.runs"):
        remove_entry(scan_project, "input.runs", 0)
    remove_entry(scan_project, "input.runs", 0, unfinished=True)
    assert list_entries(scan_project.read_text(), "input.runs") == []
    with pytest.raises(ProjectError, match="not a list of tables"):
        add_entry(scan_project, "input", {})


def test_a_refused_replacement_is_tried_again_and_written_in_place_at_last(project,
                                                                            monkeypatch):
    """Windows refuses to replace a file another program briefly has open (a virus scanner, an
    indexer, the window's file watcher): tried again, and at last written in place (CI,
    2026-11-03)."""
    import os

    import hxex.project_edit as edit

    monkeypatch.setattr(edit, "REPLACE_TRIES", 3)
    real, calls = os.replace, []

    def refused_twice(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(13, "Access is denied")
        real(src, dst)

    monkeypatch.setattr(os, "replace", refused_twice)
    set_setting(project, "run.controls.d2o_fraction", "0.85")
    assert len(calls) == 3 and "d2o_fraction = 0.85" in project.read_text()

    def always(src, dst):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(os, "replace", always)
    set_setting(project, "run.bootstrap.n_boot", "30")
    assert "n_boot = 30" in project.read_text()  # in place
    assert not list(project.parent.glob("*.hxex-set"))
