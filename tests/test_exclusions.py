# SPDX-License-Identifier: GPL-3.0-or-later
"""Spectra left out by rules of the project file, before fitting ([[input.exclude]]) and after
it ([[analysis.exclude]]); nothing is deleted (docs/PLAN.md, "Excluding spectra")."""

import csv
import tomllib

import numpy as np
import pytest

from hxex import cli
from hxex.config import ProjectError, load_project, project_from, read_input
from hxex.exclusions import Exclusion, matches, read_exclusions
from hxex.io import open_bundle
from test_cli import mzml_project  # noqa: I001


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_rules_are_read_and_checked():
    rules = read_exclusions([{"sample": "WT", "peptide_range": "0045-0060", "charge": 3,
                              "time": "30 s", "rep": 2, "note": "why"}], "input.exclude")
    rule = rules[0]
    assert rule.peptide_range == (45, 60) and rule.time == 30.0 and rule.note == "why"
    assert Exclusion.from_dict(rule.to_dict()) == rule
    assert read_exclusions({"role": "undeut"}, "x")[0].role == "undeut"  # one table: one rule
    for bad, match in (({}, "at least one"), ({"note": "only a note"}, "at least one"),
                       ({"sampel": "WT"}, "unknown key"), ({"role": "control"}, "role"),
                       ({"charge": 0}, "charge"), ({"peptide_range": "early"}, "START-END"),
                       ({"time": "soon"}, "")):
        with pytest.raises(ValueError, match=match):
            read_exclusions([bad], "input.exclude")


def test_a_rule_matches_every_key_it_names(tmp_path):
    project = load_project(mzml_project(tmp_path))
    spectra = read_input(project)
    times = np.asarray(spectra.time, dtype=float)
    one = matches(Exclusion(time=float(times[2])), spectra, "mzml")
    assert one.sum() == 1 and one[2]
    assert matches(Exclusion(sample="WT"), spectra).all()
    assert not matches(Exclusion(sample="WT", charge=3), spectra).any()
    assert matches(Exclusion(file="run1.mzML"), spectra, "mzml").all()
    assert matches(Exclusion(peptide_range=(1, 10), charge=2), spectra).all()
    assert matches(Exclusion(spectrum_id=str(spectra.spectrum_id[0])), spectra).sum() == 1


def test_input_exclusions_are_stored_not_fitted_and_recorded(tmp_path, capsys):
    path = mzml_project(tmp_path)
    spectra = read_input(load_project(path))
    gone = str(spectra.spectrum_id[3])
    text = path.read_text().replace(
        "[output]", f'[[input.exclude]]\nspectrum_id = "{gone}"\nnote = "a bad scan"\n\n'
                    '[[input.exclude]]\nsample = "nobody"\n\n[output]')
    path.write_text(text)
    capsys.readouterr()
    assert cli.main(["check", str(path)]) == 0
    assert "2 rule(s), left out before fitting" in capsys.readouterr().out
    assert cli.main(["run", str(path)]) == 0
    err = capsys.readouterr().err
    assert "matches no spectrum" in err  # the rule for sample "nobody"
    with open_bundle(tmp_path / "out") as b:
        assert gone not in set(b.fit().spectrum_id) and len(b.fit()) == len(spectra) - 1
        assert len(b.spectra()) == len(spectra)  # stored all the same
        ex = b.manifest["excluded"]
        assert ex["spectrum_ids"] == [gone] and ex["rules"][0]["matched"] == 1
        assert ex["rules"][0]["note"] == "a bad scan" and ex["rules"][1]["matched"] == 0
    assert cli.main(["info", str(tmp_path / "out")]) == 0
    assert "1 spectra left out before fitting" in capsys.readouterr().out
    # without the rules, a rerun from the stored spectra fits the excluded one again
    path.write_text(text.split("[[input.exclude]]")[0] + "[output]" + text.split("[output]")[1])
    assert cli.main(["-q", "run", str(path), "--spectra", str(tmp_path / "out"),
                     "--bundle", str(tmp_path / "again")]) == 0
    with open_bundle(tmp_path / "again") as b:
        assert gone in set(b.fit().spectrum_id) and "excluded" not in b.manifest


def test_a_rule_that_leaves_nothing_is_refused(tmp_path, capsys):
    path = mzml_project(tmp_path)
    path.write_text(path.read_text().replace(
        "[output]", '[[input.exclude]]\nfile = "run1.mzML"\n\n[output]'))
    assert cli.main(["run", str(path)]) == 1
    assert "leaves no spectrum" in capsys.readouterr().err


def test_analysis_exclusions_reject_in_memory_only(tmp_path, capsys):
    path = mzml_project(tmp_path)
    ids = list(map(str, read_input(load_project(path)).spectrum_id))
    gone = ids[2]
    path.write_text(path.read_text() + f'\n[analysis]\nks = [1]\noverlay = ["avg"]\n\n'
                    f'[[analysis.exclude]]\nspectrum_id = "{gone}"\nnote = "suspect"\n')
    capsys.readouterr()
    assert cli.main(["check", str(path)]) == 0
    assert "treated as rejected" in capsys.readouterr().out
    assert cli.main(["-q", "run", str(path)]) == 0
    assert cli.main(["-q", "report", str(path), "--tables", str(tmp_path / "t")]) == 0
    status = {r["spectrum_id"]: r["review_status"] for r in _rows(tmp_path / "t" / "spectra.csv")}
    assert status[gone] == "rejected" and list(status.values()).count("rejected") == 1
    assert cli.main(["-q", "analyse", str(path)]) == 0
    uptake = _rows(tmp_path / "out_analysis" / "species_uptake.csv")
    assert gone and all(int(r["n_spectra"]) <= 1 for r in uptake)
    times = {float(r["time"]) for r in uptake if r["time"]}
    assert len(times) == len(ids) - 1  # the excluded time point is gone from the analysis
    with open_bundle(tmp_path / "out") as b:
        assert len(b.review()) == 0 and len(b.fit()) == len(ids)  # the bundle is unchanged


def test_exclusion_tables_take_single_brackets(tmp_path):
    p = project_from(tomllib.loads(
        'project = "p"\n[input]\nformat = "specexport"\ndata_dir = "d"\n'
        '[input.exclude]\nsample = "WT"\n[analysis.exclude]\ncharge = 3\n'), tmp_path)
    assert p.input.exclude[0].sample == "WT" and p.analysis.exclude[0].charge == 3
    with pytest.raises(ProjectError, match=r"input.exclude\]\] #1: unknown key"):
        project_from(tomllib.loads('project = "p"\n[input]\nformat = "specexport"\n'
                                   'data_dir = "d"\n[[input.exclude]]\nsampel = "WT"\n'),
                     tmp_path)


def test_review_csv_lists_the_review_and_the_rules(tmp_path):
    """review.csv shows every review decision (stored in the bundle) and every spectrum an
    [[analysis.exclude]] rule leaves out, with its source and note; spectra.csv's review_K
    shows a K the review chose. With nothing to list, the file goes."""
    path = mzml_project(tmp_path)
    assert cli.main(["-q", "run", str(path)]) == 0
    with open_bundle(tmp_path / "out") as b:
        ids = list(map(str, b.fit().spectrum_id))
        b.set_review(b.review().update(ids[0], status="accepted", k_override=1,
                                       note="checked by eye"))
    path.write_text(path.read_text() + f'\n[analysis]\nks = [1]\noverlay = ["avg"]\n\n'
                    f'[[analysis.exclude]]\nspectrum_id = "{ids[3]}"\nnote = "co-elution"\n')
    tables = tmp_path / "t"
    assert cli.main(["-q", "report", str(path), "--tables", str(tables)]) == 0
    rows = {r["spectrum_id"]: r for r in _rows(tables / "review.csv")}
    assert rows[ids[0]]["source"] == "review" and rows[ids[0]]["note"] == "checked by eye"
    assert rows[ids[0]]["review_K"] == "1" and rows[ids[0]]["sample"] == "WT"
    assert rows[ids[3]]["source"] == "exclusion rule" and "co-elution" in rows[ids[3]]["note"]
    assert rows[ids[3]]["status"] == "rejected" and rows[ids[3]]["stale"] == "False"
    spectra = {r["spectrum_id"]: r for r in _rows(tables / "spectra.csv")}
    assert spectra[ids[0]]["review_K"] == "1" and spectra[ids[1]]["review_K"] == ""
    # no review and no rule: no review.csv
    from hxex.io import ReviewTable

    with open_bundle(tmp_path / "out") as b:
        b.set_review(ReviewTable.empty())
    assert cli.main(["-q", "report", str(tmp_path / "out"), "--tables", str(tables)]) == 0
    assert not (tables / "review.csv").exists()
