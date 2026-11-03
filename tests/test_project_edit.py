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
