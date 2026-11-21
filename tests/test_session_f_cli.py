# SPDX-License-Identifier: GPL-3.0-or-later
"""The new user's path (audit 2026-11-01, Session F): hxex report always writes the tables
(--figures adds the figures), hxex info takes the project file, an unknown key names its line
and the heading still switched off, the version names its commit, and the CSV numbers are
short."""

import csv
from pathlib import Path

import numpy as np
import pytest

import hxex
from hxex import cli
from hxex.config import ProjectError, load_project, template
from hxex.report.tables import _number
from test_cli import mzml_project  # noqa: I001


def test_report_always_writes_the_tables_and_figures_add_to_them(tmp_path, capsys):
    path = mzml_project(tmp_path)
    assert cli.main(["-q", "run", str(path)]) == 0
    tables, figures = tmp_path / "out_tables", tmp_path / "out_figures"
    sid = (tables / "spectra.csv").read_text().splitlines()[1].split(",")[0]
    (tmp_path / "review.csv").write_text(f"spectrum_id,review_status\n{sid},rejected\n")
    path.write_text(path.read_text() + '\n[review]\ntable = "review.csv"\n'
                    '\n[figures]\nkinds = ["uptake"]\nseries = "all"\n')
    capsys.readouterr()
    # right after the run: the tables of the same results are written again without
    # --overwrite, with the review applied, and the closing line names both folders
    assert cli.main(["report", str(path), "--figures"]) == 0
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert out.startswith("5 table(s) in ") and str(tables) in out and str(figures) in out
    assert "(review: 1 rejected); " in out and "figure file(s) in" in out
    assert (tables / "review.csv").is_file()
    with (tables / "spectra.csv").open() as fh:
        assert {r["review_status"] for r in csv.DictReader(fh)} == {"rejected", "unreviewed"}
    from hxex.io import made_from, open_bundle

    # the figures folder is written again (it is hxex's)
    assert cli.main(["-q", "report", str(path), "--figures"]) == 0
    # tables of other results (here: another bundle's) are written again too
    other = tmp_path / "other.toml"
    other.write_text(path.read_text().replace('bundle = "out"', 'bundle = "out2"', 1)
                     .replace("[output]", '[output]\ntables = "out_tables"', 1))
    assert cli.main(["-q", "run", str(other)]) == 0  # its tables: replaced
    capsys.readouterr()
    assert cli.main(["-q", "report", str(path)]) == 0
    with open_bundle(tmp_path / "out") as b:
        assert made_from(tables) == {"": b.manifest["created"]}
    # [output] tables = false: only the figures, and the line says so
    path.write_text(path.read_text().replace("[output]", "[output]\ntables = false", 1))
    capsys.readouterr()
    assert cli.main(["report", str(path), "--figures"]) == 0
    assert capsys.readouterr().out.strip().endswith("no tables ([output] tables = false)")


def test_tables_changed_by_hand_are_not_written_over_without_asking(tmp_path, capsys):
    """The same results' tables are written again in place, unless one of them was changed
    after hxex wrote it (owner, 2026-11-02): then report stops and names it, and
    --keep-previous keeps it in the earlier folder (or the file is moved away)."""
    from hxex.io import edited_outputs

    path = mzml_project(tmp_path)
    assert cli.main(["-q", "run", str(path)]) == 0
    tables = tmp_path / "out_tables"
    assert edited_outputs(tables) == []
    spectra = tables / "spectra.csv"
    edited = spectra.read_text().replace("unreviewed", "rejected", 1)
    spectra.write_text(edited)  # a review typed into the run's own table
    assert edited_outputs(tables) == ["spectra.csv"]
    capsys.readouterr()
    assert cli.main(["-q", "report", str(path)]) == 1
    err = capsys.readouterr().err
    assert "spectra.csv changed after hxex wrote it" in err and "--keep-previous" in err
    assert spectra.read_text() == edited  # nothing written
    assert cli.main(["-q", "report", str(path), "--keep-previous"]) == 0
    assert (tmp_path / "out_tables.1" / "spectra.csv").read_text() == edited
    assert edited_outputs(tables) == []  # the new tables, as written
    spectra.write_text(spectra.read_text().replace("unreviewed", "rejected", 1))
    capsys.readouterr()
    assert cli.main(["-q", "report", str(path)]) == 1  # --overwrite is no longer a way past
    spectra.unlink()  # the edited table deleted: written again
    assert cli.main(["-q", "report", str(path)]) == 0
    assert edited_outputs(tables) == []
    # an unchanged folder is written again in place, as before
    assert cli.main(["-q", "report", str(path)]) == 0


def test_every_command_refuses_to_write_over_a_file_changed_by_hand(tmp_path, capsys):
    """A file hxex wrote that was changed since is never written over (owner, 2026-11-20):
    run, report (tables of earlier results too) and analyse stop before any work and name it,
    and hxex check names it up front."""
    from hxex.io import edited_outputs

    path = mzml_project(tmp_path)
    assert cli.main(["-q", "run", str(path)]) == 0
    assert cli.main(["-q", "analyse", str(path)]) == 0
    tables = tmp_path / "out_tables"
    analysis = next(p for p in tmp_path.iterdir() if p.name.startswith("out_analysis"))
    uptake = tables / "uptake.csv"
    uptake.write_text(uptake.read_text() + "a note\n")
    capsys.readouterr()
    assert cli.main(["check", str(path)]) in (0, 1)
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if "out_tables:" in x)
    assert "1 file changed since hxex wrote it (uptake.csv)" in line
    # run (a fresh start: it would rewrite the tables) stops before reading anything
    assert cli.main(["-q", "run", str(path), "--fresh"]) == 1
    err = capsys.readouterr().err
    assert "uptake.csv changed after hxex wrote it" in err and "--keep-previous moves" in err
    assert "a note" in uptake.read_text()
    # report on tables made from earlier results (a fresh run made new ones) stops too
    uptake.unlink()
    assert cli.main(["-q", "run", str(path), "--fresh"]) == 0
    spectra = tables / "spectra.csv"
    spectra.write_text(spectra.read_text() + "x\n")
    listing = tables / "hxex_listing.json"  # as if made from earlier results
    listing.write_text(listing.read_text().replace('"made_from"', '"made_from_old"'))
    assert cli.main(["-q", "report", str(path)]) == 1
    assert "spectra.csv changed after hxex wrote it" in capsys.readouterr().err
    spectra.unlink()
    # analyse: an analysis table changed by hand
    table = next(p for p in sorted(analysis.glob("*.csv")))
    table.write_text(table.read_text() + "x\n")
    assert edited_outputs(analysis) == [table.name]
    assert cli.main(["-q", "analyse", str(path)]) == 1
    assert f"{table.name} changed after hxex wrote it" in capsys.readouterr().err
    assert table.read_text().endswith("x\n")
    assert cli.main(["-q", "analyse", str(path), "--keep-previous"]) == 0
    assert (analysis.with_name(analysis.name + ".1") / table.name).read_text().endswith("x\n")
    assert edited_outputs(analysis) == []


def test_info_takes_the_project_file_and_tells_its_review_from_the_bundles(tmp_path, capsys):
    path = mzml_project(tmp_path)
    assert cli.main(["-q", "run", str(path)]) == 0
    capsys.readouterr()
    assert cli.main(["info", str(path)]) == 0
    info = capsys.readouterr().out
    assert "review         none stored in the bundle" in info
    assert "review table   none in hxex.toml" in info and "measurements   6" in info
    sid = (tmp_path / "out_tables" / "spectra.csv").read_text().splitlines()[1].split(",")[0]
    (tmp_path / "review.csv").write_text(f"spectrum_id,review_status\n{sid},rejected\n")
    path.write_text(path.read_text() + '\n[review]\ntable = "review.csv"\n')
    assert cli.main(["info", str(path)]) == 0
    info = capsys.readouterr().out
    assert "review         none stored in the bundle" in info
    line = next(x for x in info.splitlines() if x.startswith("review table"))
    assert "review.csv" in line and "1 rejected" in line and "(from hxex.toml" in line
    assert cli.main(["info", str(tmp_path / "out")]) == 0  # the bundle alone: as before
    assert "review table" not in capsys.readouterr().out
    for command in ("info", "report", "analyse"):  # shown with the project file
        with pytest.raises(SystemExit):
            cli.main([command, "-h"])
        assert f"hxex {command} [-h]" in (usage := capsys.readouterr().out)
        assert "hxex.toml|BUNDLE" in usage


def test_an_unknown_key_names_its_line_and_the_heading_still_off(tmp_path):
    """Audit 2026-11-01, Part 2 B1: a setting switched on without its heading lands under the
    live heading above it."""
    text = template()
    lines = text.splitlines()
    at = next(i for i, x in enumerate(lines) if x.startswith('# reference = "WT"'))
    heading = next(i for i in range(at, 0, -1) if lines[i].startswith("# [analysis]"))
    lines[at] = lines[at][2:]
    (tmp_path / "data").mkdir()
    path = tmp_path / "hxex.toml"
    path.write_text("\n".join(lines).replace('data_dir = "SpecExport"', 'data_dir = "data"'))
    with pytest.raises(ProjectError) as info:
        load_project(path)
    assert (f"`reference` on line {at + 1} was read under [output] because the heading above "
            f"it, `# [analysis]` on line {heading + 1}, is still switched off: delete its `# ` "
            f"too, or put the `# ` back in front of line {at + 1}") in str(info.value)
    assert str(info.value).startswith("unknown key(s) in [output]: reference")
    # without a heading switched off: the line, and the did-you-mean kept
    path.write_text('project = "p"\n[input]\nformat = "specexport"\ndata_dir = "data"\n'
                    '[output]\nbundl = "x"\n')
    with pytest.raises(ProjectError, match=r"\(did you mean bundle\?\); `bundl` is on line 6, "
                                           r"under \[output\]"):
        load_project(path)
    path.write_text('project = "p"\n[input]\nformat = "specexport"\ndata_dir = "data"\n'
                    '[run.picker]\nnoize = 1\n')
    with pytest.raises(ProjectError, match=r"`noize` is on line 6, under \[run.picker\]"):
        load_project(path)
    path.write_text('project = "p"\n# [run.controls]\nd2o_fraction = 0.9\n[input]\n'
                    'format = "specexport"\ndata_dir = "data"\n')
    with pytest.raises(ProjectError, match=r"`d2o_fraction` on line 3 was read at the top of "
                                           r"the file \(outside any table\) because the "
                                           r"heading above it, `# \[run.controls\]` on line 2"):
        load_project(path)


def test_an_exclusion_rule_matching_nothing_says_it_may_be_deleted(tmp_path, capsys):
    path = mzml_project(tmp_path)
    path.write_text(path.read_text() + '\n[[input.exclude]]\nsample = "KO"\n')
    assert cli.main(["check", str(path)]) == 1
    assert "delete it if this dataset has no such spectra" in capsys.readouterr().err


def test_the_version_names_its_commit(monkeypatch, tmp_path, capsys):
    """Audit 2026-11-01, Part 2 M8: git archive (GitHub's zip) fills in the commit file; an
    editable install asks git; otherwise only the version."""
    assert hxex._commit_from("3ed39ac 2026-11-02\n") == "3ed39ac, 2026-11-02"
    assert hxex._commit_from("$Format:%h %cs$\n") is None and hxex._commit_from("") is None
    assert hxex._commit_from_git(tmp_path) is None  # not a checkout of hxex
    monkeypatch.setattr(hxex, "_commit", "3ed39ac, 2026-11-02")
    assert hxex.version_text() == f"hxex {hxex.__version__} (3ed39ac, 2026-11-02)"
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert capsys.readouterr().out == f"hxex {hxex.__version__} (3ed39ac, 2026-11-02)\n"
    from hxex.io.bundle import _versions

    assert _versions()["hxex_commit"] == "3ed39ac, 2026-11-02"
    assert _versions()["hxex"] == hxex.__version__  # the rerun check compares the version
    monkeypatch.setattr(hxex, "_commit", "")
    assert hxex.version_text() == f"hxex {hxex.__version__}" and "hxex_commit" not in _versions()


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_git_is_asked_without_a_console_window_on_windows(monkeypatch, platform):
    """git is a console program: asked from a process without a console (the window's
    commands run under pythonw on Windows) it flashed a console window of its own at every
    command's start (Windows test, 2026-11-16). On Windows it is started without one."""
    import subprocess
    import sys

    calls = []

    def run(args, **kw):
        calls.append(kw)
        return subprocess.CompletedProcess(args, 1, "", "")

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(subprocess, "run", run)
    assert hxex._commit_from_git(Path(hxex.__file__).parent) is None
    assert calls  # asked, and every call carries the flag on Windows only
    flags = [kw.get("creationflags") for kw in calls]
    assert flags == ([0x08000000] * len(calls) if platform == "win32" else [None] * len(calls))


def test_a_command_names_the_commit_it_started_with(monkeypatch, tmp_path):
    """A checkout that moves during a long command (another branch merged) is not named in
    what the command writes at its end: the commit is read when the command starts."""
    from hxex.io.bundle import _versions

    checkout = {"now": "aaaaaaa, 2026-11-10"}
    monkeypatch.setattr(hxex, "_commit", None)
    monkeypatch.setattr(hxex, "_commit_from", lambda text: None)
    monkeypatch.setattr(hxex, "_commit_from_git", lambda package: checkout["now"])
    cli.main(["info", str(tmp_path / "missing")])  # fails, but only after it started
    checkout["now"] = "bbbbbbb, 2026-11-10"  # merged while the command ran
    assert _versions()["hxex_commit"] == "aaaaaaa, 2026-11-10"


@pytest.mark.parametrize("value, text", [
    (1800.0, "1800"), (0.4865089041514463, "0.486509"), (0.0036541559618291086, "0.00365416"),
    (523.276543, "523.2765"), (1234.567891, "1234.5679"), (12.3456789, "12.3457"),
    (-2.5, "-2.5"), (0.0, "0"), (-0.0, "0"), (1e-7, "1e-07"), (np.float32(0.1), "0.1"),
    (float("inf"), "inf"), (-float("inf"), "-inf"),
])
def test_csv_numbers_are_short_but_keep_what_matters(value, text):
    assert _number(float(value)) == text


def test_no_progress_thread_is_left_running_after_a_command(tmp_path):
    import threading

    path = mzml_project(tmp_path)
    assert cli.main(["-q", "run", str(path)]) == 0
    for thread in [t for t in threading.enumerate() if t.name == "hxex-progress"]:
        thread.join(2.0)
        assert not thread.is_alive()
