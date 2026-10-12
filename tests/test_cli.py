# SPDX-License-Identifier: GPL-3.0-or-later
"""The command line and the project file: templates, checks, a whole run of a synthetic mzML
project, resuming a stopped bootstrap, reports from a bundle and bundle summaries."""

import tomllib

import pytest

from hxex import cli
from hxex.config import FORMATS, ProjectError, load_project, project_from, template
from hxex.hooks import Cancelled
from hxex.io import open_bundle
from hxex.run import RunConfig, run
from test_mzml import PEPTIDE, time_course, write_mzml  # noqa: I001
from test_run import config as run_config  # noqa: I001
from test_run import raw_spectra

FAST_RUN = """
[run.picker]
zero_filling = 3
noise = 100.0

[run.fit]
k_min = 1
k_max = 1
adam_steps = 60
n_restarts = 1
backend = { device = "cpu", dtype = "float64" }

[run.bootstrap]
n_boot = 3
n_restarts = 1
adam_steps = 30
polish_steps = 0
"""


def mzml_project(tmp_path, *, output="", extra_input=""):
    """A project of one synthetic mzML file (scan times as labelling times) and one peptide."""
    write_mzml(tmp_path / "run1.mzML", time_course(n=6))
    (tmp_path / "peptides.csv").write_text(
        "file,sample,peptide,peptide_range,start_seq,end_seq,charge\n"
        f"x,S,{PEPTIDE},0001-0010,1,10,2\n")
    text = f"""
project = "cli"

[input]
format = "mzml"
peptides = "peptides.csv"
{extra_input}

[[input.runs]]
path = "run1.mzML"
sample = "WT"

[output]
bundle = "out"
figures = false
{output}
{FAST_RUN}
"""
    path = tmp_path / "hxex.toml"
    path.write_text(text)
    return path


def test_every_template_reads_back_as_the_defaults(tmp_path, capsys):
    for fmt in FORMATS:
        p = project_from(tomllib.loads(template(fmt)), tmp_path)
        assert p.run == RunConfig() and p.input.format == fmt
        assert p.output.bundle == tmp_path / "results"
    assert cli.main(["init", "--format", "mzml", "-o", str(tmp_path / "a.toml")]) == 0
    assert load_project(tmp_path / "a.toml").input.runs[1].time == 60.0
    assert cli.main(["init", "-o", str(tmp_path / "a.toml")]) == 1  # exists
    assert cli.main(["init", "--format", "pooled", "-o", "-"]) == 0
    assert 'format = "pooled"' in capsys.readouterr().out


def test_project_files_are_checked(tmp_path):
    def project(text):
        return project_from(tomllib.loads('project = "p"\n' + text), tmp_path)

    p = project('[input]\nformat = "specexport"\ndata_dir = "d"\n'
                "[run.bootstrap]\nn_boot = 7\n[run]\nquality = false\nmax_chunk = 5\n"
                '[run.fit.noise]\nmodel = "poisson"\n[output]\nbundle = "res"\nfigures = "figs"\n')
    assert p.input.data_dir == tmp_path / "d"
    assert p.run.bootstrap.n_boot == 7 and p.run.bootstrap.n_restarts == 3  # the rest default
    assert p.run.quality is None and p.run.max_chunk == 5
    assert p.run.fit.noise.model == "poisson" and p.run.fit.k_max == 3
    assert p.output.figures_dir() == tmp_path / "figs" and p.run.plots is not None
    assert p.output.tables_dir() == tmp_path / "res_tables"
    table = project('[input]\nformat = "hxexpress"\ndata_dir = "d"\npeptides = "metadf.csv"\n')
    assert table.input.peptides == tmp_path / "metadf.csv"  # files listed instead of found
    bad = [
        ('[input]\nformat = "csv"\n', "input.format"),
        ('[input]\nformat = "specexport"\n', "data_dir is needed"),
        ('[input]\nformat = "specexport"\ndata_dir = "d"\nthermo_dir = "t"\n', "mzml"),
        ('[input]\nformat = "mzml"\npeptides = "p.csv"\n', "input.runs"),
        ('[input]\nformat = "specexport"\ndata_dir = "d"\n[run.fit]\nbogus = 1\n', "bogus"),
        ('[input]\nformat = "specexport"\ndata_dir = "d"\n[surprise]\n', "surprise"),
        ('[input]\nformat = "mzml"\npeptides = "p.csv"\nparents = false\n'
         '[[input.runs]]\npath = "a"\nsample = "S"\n', "no target"),
    ]
    for text, match in bad:
        with pytest.raises(ProjectError, match=match):
            project(text)
    with pytest.raises(ProjectError, match="not valid TOML"):
        (tmp_path / "x.toml").write_text("project = ")
        load_project(tmp_path / "x.toml")


def test_check_shows_what_would_be_read(tmp_path, capsys):
    path = mzml_project(tmp_path, extra_input='[input.target]\nfilter_excludes = "sid"')
    assert cli.main(["check", str(path)]) == 0
    out = capsys.readouterr().out
    assert "targets        1 from" in out and "6 scans; 1 of 1 targets have scans" in out
    assert out.strip().endswith("ok")
    (tmp_path / "out").mkdir()  # an existing bundle is reported
    assert cli.main(["check", str(path)]) == 1


def test_run_info_and_report(tmp_path, capsys):
    path = mzml_project(tmp_path, output='notes = "first try"')
    assert cli.main(["-q", "run", str(path)]) == 0
    out = capsys.readouterr().out
    assert "6 measurements; selected K: 1: 6" in out
    with open_bundle(tmp_path / "out") as b:
        assert b.pending == [] and b.bootstrap() is not None and b.has("spectra")
        assert b.spectra_source()["reader"] == "mzml" and b.manifest["notes"] == "first try"
    assert {p.name for p in (tmp_path / "out_tables").iterdir()} == {
        "series.csv", "spectra.csv", "components.csv", "uptake.csv"}
    assert cli.main(["run", str(path)]) == 1  # the bundle exists
    assert cli.main(["info", "--verify", str(tmp_path / "out")]) == 0
    info = capsys.readouterr().out
    for line in ("project        cli", "input          mzml", "measurements   6",
                 "selected K     1: 6", "bootstrap      yes", "pending        nothing",
                 "checksums      ok"):
        assert line in info
    assert cli.main(["report", str(tmp_path / "out"), "--figures", str(tmp_path / "f"),
                     "--series", "all", "--format", "png"]) == 0
    assert any(p.suffix == ".png" for p in (tmp_path / "f").iterdir())
    assert not (tmp_path / "out_tables_again").exists()
    assert cli.main(["report", str(tmp_path / "out"), "--tables",
                     str(tmp_path / "out_tables_again")]) == 0
    assert (tmp_path / "out_tables_again" / "series.csv").is_file()
    assert cli.main(["run", "--overwrite", "--device", "cpu", str(path)]) == 0


def test_resume_completes_a_stopped_bootstrap(tmp_path, capsys):
    raw = raw_spectra(n_series=1)
    stages = []

    class Hook:
        def start_stage(self, name, start, end):
            stages.append(name)

        def __call__(self, done, total):
            pass

    with pytest.raises(Cancelled):
        run(raw, run_config(), out=tmp_path / "b", progress=Hook(),
            cancel=lambda: stages[-1:] == ["bootstrap"])
    assert cli.main(["info", str(tmp_path / "b")]) == 0
    assert "pending        bootstrap" in capsys.readouterr().out
    assert cli.main(["-q", "resume", str(tmp_path / "b"), "--tables"]) == 0
    with open_bundle(tmp_path / "b") as b:
        assert b.pending == [] and b.bootstrap() is not None
    assert (tmp_path / "b_tables" / "uptake.csv").is_file()
    assert cli.main(["resume", str(tmp_path / "b")]) == 1  # nothing pending any more


def test_errors_are_messages_not_tracebacks(tmp_path, capsys):
    assert cli.main(["check", str(tmp_path / "missing.toml")]) == 1
    err = capsys.readouterr().err
    assert "cannot read" in err and "Traceback" not in err
    assert cli.main(["info", str(tmp_path)]) == 1
    p = mzml_project(tmp_path)
    (tmp_path / "run1.mzML").unlink()
    assert cli.main(["check", str(p)]) == 1
