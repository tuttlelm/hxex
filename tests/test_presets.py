# SPDX-License-Identifier: GPL-3.0-or-later
"""Presets (hxex.presets, docs/PLAN.md Next steps 28): one top-level setting for how much work
a run and its analysis do, laid under the project file's own settings."""

import tomllib

import pytest

from hxex import cli, presets
from hxex.config import (
    ProjectError,
    describe,
    load_project,
    preset_line,
    preset_warning,
    project_from,
    project_record,
    template,
)
from hxex.io import open_bundle
from hxex.project_edit import set_setting
from test_cli import mzml_project  # noqa: I001

BASE = {"project": "p", "input": {"format": "specexport", "data_dir": "."}}


def _project(tmp_path, **tables):
    return project_from({**BASE, **tables}, tmp_path)


def test_each_preset_sets_the_effort_of_the_run(tmp_path):
    quick = _project(tmp_path, preset="quick").run
    assert (quick.fit.k_min, quick.fit.k_max) == (1, 2) and quick.bootstrap is None
    assert quick.population_test.null == "chi_square"
    bimodal = _project(tmp_path, preset="bimodal").run
    assert bimodal.fit.k_max == 3 and bimodal.population_test.null == "simulated"
    assert bimodal.population_test.decide_up_to == 2
    assert bimodal.bootstrap.ks == (2,) and bimodal.bootstrap.set_k
    assert bimodal.bootstrap.n_boot == 20
    multimodal = _project(tmp_path, preset="multimodal").run
    assert multimodal.population_test.decide_up_to is None
    assert multimodal.bootstrap.ks == "selected" and multimodal.fit.k_max == 3
    maximum = _project(tmp_path, preset="maximum")
    assert maximum.run.bootstrap.n_boot == 100 and maximum.figures.series == "all"
    assert maximum.run.population_test == multimodal.population_test
    # no preset: the defaults, as before presets existed
    none = _project(tmp_path)
    assert none.run.bootstrap is None and none.run.population_test is None
    assert none.preset.name is None and _project(tmp_path, preset=False).preset.name is None


def test_the_files_own_settings_win_and_are_named(tmp_path):
    p = _project(tmp_path, preset="multimodal",
                 run={"fit": {"k_max": 4}, "bootstrap": {"n_boot": 50}})
    assert p.run.fit.k_max == 4 and p.run.bootstrap.n_boot == 50
    assert p.run.bootstrap.set_k  # the preset's, beside the file's n_boot
    assert p.preset.overridden == {"run.fit.k_max": (4, 3)}  # n_boot: the preset has none
    line = preset_line(p)
    assert line.startswith("multimodal: ") and "run.fit.k_max = 4 (preset: 3)" in line
    assert "run.fit.k_max = 4 (the preset's: 3)" in preset_warning(p)
    assert describe(p)["preset"] == line
    # a table the file switches on wins over the preset's off; the file's off over its on
    q = _project(tmp_path, preset="quick", run={"bootstrap": {"n_boot": 5}})
    assert q.run.bootstrap.n_boot == 5 and "run.bootstrap" in q.preset.overridden
    m = _project(tmp_path, preset="maximum", run={"population_test": False})
    assert m.run.population_test is None
    assert preset_warning(_project(tmp_path, preset="quick")) is None


def test_a_preset_never_switches_the_analysis_on(tmp_path):
    assert _project(tmp_path, preset="bimodal").analysis is None
    with_analysis = _project(tmp_path, preset="bimodal", analysis={"reference": "WT"})
    assert with_analysis.analysis.fragments.fit_rates
    assert not _project(tmp_path, preset="quick",
                        analysis={"reference": "WT"}).analysis.fragments.fit_rates


def test_an_unknown_preset_is_refused_with_the_names(tmp_path):
    with pytest.raises(ProjectError, match='"quick", "bimodal", "multimodal", "maximum"'):
        _project(tmp_path, preset="complete")


def test_the_template_leaves_the_presets_settings_to_the_preset(tmp_path):
    text = template()
    assert '# preset = "quick"' in text
    for line in ("k_min = 1", "k_max = 3", 'series = "flagged"'):
        assert f"\n# {line}" in text and f"\n{line}" not in text
    switched = text.replace('# preset = "quick"', 'preset = "quick"', 1)
    p = project_from(tomllib.loads(switched), tmp_path)
    assert p.run.fit.k_max == 2 and p.preset.overridden == {}


def test_hxex_set_says_what_an_unset_setting_takes(tmp_path):
    path = tmp_path / "hxex.toml"
    path.write_text(template())
    (tmp_path / "SpecExport").mkdir()
    assert "not set (default)" in set_setting(path, "preset", "bimodal").describe()
    change = set_setting(path, "run.fit.k_max", "4")
    assert change.describe().startswith('run.fit.k_max: not set (preset "bimodal": 3) -> 4')
    with pytest.raises(ProjectError, match="expected one of"):
        set_setting(path, "preset", "most")


def test_a_quick_run_is_recorded_and_marked_provisional(tmp_path, capsys):
    path = mzml_project(tmp_path)
    text = path.read_text()
    # the fast settings, without the K values and the bootstrap the preset chooses
    text = text.split("[run.bootstrap]")[0].replace("k_min = 1\nk_max = 1\n", "")
    path.write_text('preset = "quick"\n' + text)
    project = load_project(path)
    assert project.run.fit.k_max == 2 and project.run.bootstrap is None
    assert project_record(project)["preset"] == {"name": "quick", "provisional": True,
                                                  "overridden": {}}
    assert cli.main(["-q", "run", str(path)]) == 0
    capsys.readouterr()
    with open_bundle(tmp_path / "out") as b:
        assert b.bootstrap() is None
        recorded = b.manifest["configs"]["project_file"]["preset"]
        assert recorded["name"] == "quick" and recorded["provisional"]
    used = (tmp_path / "out" / "run_settings_used.toml").read_text()
    assert '# The project file chose preset = "quick"' in used and "provisional" in used
    assert "\npreset" not in used  # a record: every setting written out instead
    assert cli.main(["info", str(tmp_path / "out")]) == 0
    info = capsys.readouterr().out
    assert "preset         quick (provisional" in info
    path.write_text(path.read_text() + '\n[figures]\nseries = "all"\nformat = "svg"\n'
                    'kinds = ["uptake"]\noverview = false\n')
    assert cli.main(["-q", "report", str(path), "--figures"]) == 0
    svgs = list((tmp_path / "out_figures").glob("*.svg"))
    assert svgs and all("Provisional" in p.read_text() for p in svgs)


def test_preset_values_by_setting():
    assert presets.preset_value("bimodal", "run.bootstrap.ks") == [2]
    assert presets.preset_value("quick", "run.bootstrap") is False
    assert presets.preset_value("quick", "run.bootstrap.n_boot") is presets.NOT_SET
    assert presets.preset_value(None, "run.fit.k_max") is presets.NOT_SET
    assert "run.fit.k_max" in presets.setting_names()
    assert "figures.series" in presets.setting_names()


def test_the_provisional_note_gets_room_of_its_own_and_joins_one_already_there():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from hxex.report.results import mark_provisional

    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot()
    bottom = ax.get_position().y0 * 3  # inches from the figure's foot
    mark_provisional(fig, "Provisional: a note")
    assert fig.get_size_inches()[1] > 3
    assert ax.get_position().y0 * fig.get_size_inches()[1] > bottom  # room under the axes
    assert [t.get_text() for t in fig.texts] == ["Provisional: a note"]
    mark_provisional(fig, "Provisional: a note")  # the same note once only
    assert len(fig.texts) == 1 and fig.texts[0].get_text().count("\n") == 0
    mark_provisional(fig, "Provisional (B): another")  # a line more of the one note
    assert len(fig.texts) == 1 and fig.texts[0].get_text().endswith("\nProvisional (B): another")
    assert mark_provisional(fig, None) is fig
