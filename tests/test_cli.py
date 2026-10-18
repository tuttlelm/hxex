# SPDX-License-Identifier: GPL-3.0-or-later
"""The command line and the project file: templates, checks, a whole run of a synthetic mzML
project, resuming a stopped bootstrap, reports from a bundle and bundle summaries."""

import dataclasses
import logging
import tomllib

import pytest

from hxex import cli
from hxex.config import FORMATS, ProjectError, load_project, project_from, template
from hxex.hooks import Cancelled
from hxex.io import ReadError, open_bundle
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


def test_run_time_and_rep_are_checked(tmp_path):
    """A run's time is a number of seconds or absent (false, true or text made a run at 0 s,
    1 s or crashed later); rep is a whole number of 1 or more. The error names the run."""
    def project(run):
        return project_from(tomllib.loads(
            'project = "p"\n[input]\nformat = "mzml"\npeptides = "p.csv"\n'
            f'[[input.runs]]\npath = "a.mzML"\nsample = "S"\n{run}'), tmp_path)

    assert project("").input.runs[0].time is None
    assert project("time = 60\n").input.runs[0].time == 60.0
    assert project("time = 0.0\nrep = 2\n").input.runs[0].rep == 2
    for bad, match in (("time = false", "time must be a number"), ("time = true", "got True"),
                       ('time = "60 s"', "left out"), ("time = inf", "got inf"),
                       ("rep = false", "rep must be"), ("rep = 0", "got 0"),
                       ("rep = 1.5", "rep must be")):
        with pytest.raises(ProjectError, match=match) as info:
            project(bad + "\n")
        assert "input.runs #1 (a.mzML)" in str(info.value)


def test_false_unsets_the_analysis_tables(tmp_path):
    """``false`` for a table-valued [analysis] key means none (it gave a traceback)."""
    def project(text):
        return project_from(tomllib.loads(
            'project = "p"\n[input]\nformat = "specexport"\ndata_dir = "d"\n[analysis]\n'
            + text), tmp_path)

    a = project("filters = false\nfragments = false\ncontrols = false\nmerge = false\n"
                "labelling = false\nruns = false\nfasta = false\n").analysis
    assert a.filters is None and a.fragments is None and a.controls is None
    assert a.merge is None and a.labelling is None and a.runs is None and a.fasta is None
    assert project("").analysis.filters is not None
    for bad in ("choice = false\n", "filters = 3\n", "fragments = 1\n", "choice = [1]\n"):
        with pytest.raises(ProjectError, match=r"\[analysis\]"):
            project(bad)


def test_fragment_scans_are_tied_to_their_parent(tmp_path):
    from hxex.config import scan_targets
    from hxex.fragments import parent_mz_range

    def targets(fragments):
        path = mzml_project(tmp_path, extra_input=f'[input.fragments]\nions = "c:2-3:1"\n'
                                                  f"{fragments}")
        return scan_targets(load_project(path))

    got = targets('ms_level = 3\npreactivation = "HCD"\n')
    frag = got[1]  # after the peptide itself
    assert (frag.ms_level, frag.preactivation) == (3, "HCD") and got[0].parent_mz is None
    picker = load_project(tmp_path / "hxex.toml").run.picker  # the project's bins
    assert frag.parent_mz == parent_mz_range(got[0].entry, picker=picker)
    assert targets("parent_mz = false\n")[1].parent_mz is None
    assert targets("parent_mz = [500.0, 505.5]\n")[1].parent_mz == (500.0, 505.5)
    with pytest.raises(ProjectError, match="parent_mz"):
        targets('parent_mz = "own"\n')


def test_a_peptides_own_retention_window(tmp_path):
    from hxex.config import scan_targets

    path = mzml_project(tmp_path, extra_input='[input.target]\nrt_window = [0.0, 30.0]\n'
                                              '[input.fragments]\nions = "c:2-3:1"\n')
    table = tmp_path / "peptides.csv"
    head = "file,sample,peptide,peptide_range,start_seq,end_seq,charge,rt_start,rt_end\n"
    table.write_text(head + f"x,S,{PEPTIDE},0001-0010,1,10,2,5.5,6.75\n"
                     f"x,S,{PEPTIDE},0001-0010,1,10,3,,\n")
    got = scan_targets(load_project(path))
    first, second = got[:len(got) // 2], got[len(got) // 2:]  # each peptide, then its fragments
    assert first[0].ms_level == 1 and second[0].entry.charge == 3 and len(first) > 1
    assert {t.rt_window for t in first} == {(330.0, 405.0)}  # minutes -> seconds, fragments too
    # without columns: the project's window for the peptide, the fragments' own (none)
    assert second[0].rt_window == (0.0, 30.0)
    assert {t.rt_window for t in second[1:]} == {None}
    for row, match in ((",6.0", "both or neither"), ("7.0,6.0", "after the start"),
                       ("a,6.0", "not numbers")):
        table.write_text(head + f"x,S,{PEPTIDE},0001-0010,1,10,2,{row}\n")
        with pytest.raises(ReadError, match="line 2") as err:
            scan_targets(load_project(path))
        assert match in str(err.value.__cause__)


def test_check_shows_what_would_be_read(tmp_path, capsys):
    path = mzml_project(tmp_path, extra_input='[input.target]\nfilter_excludes = "sid"')
    assert cli.main(["check", str(path)]) == 0
    out = capsys.readouterr().out
    assert "targets        1 from" in out and "6 scans; 1 of 1 targets have scans" in out
    assert out.strip().endswith("ok")
    (tmp_path / "out").mkdir()  # an existing bundle is reported
    assert cli.main(["check", str(path)]) == 1
    err = capsys.readouterr().err
    assert "the bundle" in err and "--keep-previous" in err
    path = mzml_project(tmp_path, output="keep_previous = true")
    assert cli.main(["check", str(path)]) == 0


def test_check_names_every_problem_after_the_summary(tmp_path, capsys):
    """Missing files and folders are named, a peptide that does not tokenise is reported with
    its line and a modX hint, and the summary comes first whatever is wrong."""
    path = mzml_project(tmp_path)
    (tmp_path / "peptides.csv").rename(tmp_path / "gone.csv")
    assert cli.main(["check", str(path)]) == 1
    out, err = capsys.readouterr()
    assert "project        cli" in out and "input.peptides" in err and "gone" not in err
    assert "peptides.csv: no such file" in err and "1 problem(s)" in err
    (tmp_path / "peptides.csv").write_text(
        "file,sample,peptide,peptide_range,start_seq,end_seq,charge\n"
        f"x,S,{PEPTIDE},0001-0010,1,10,2\nx,S,AC(ox)DE,0001-0005,1,5,2\n")
    assert cli.main(["check", str(path)]) == 1
    err = capsys.readouterr().err
    assert "peptides.csv line 3: peptide 'AC(ox)DE'" in err and "modX" in err
    assert "1 problem(s)" in err
    (tmp_path / "out").mkdir()
    folder = tmp_path / "folder.toml"
    folder.write_text('project = "f"\n[input]\nformat = "specexport"\ndata_dir = "nowhere"\n'
                      'retention_times = "rt.csv"\n[output]\nbundle = "out"\n')
    assert cli.main(["check", str(folder)]) == 1
    out, err = capsys.readouterr()
    assert "format         specexport" in out
    assert "nowhere: no such directory" in err and "rt.csv: no such file" in err
    assert "the bundle" in err and "3 problem(s)" in err


def test_run_info_and_report(tmp_path, capsys):
    path = mzml_project(tmp_path, output='notes = "first try"')
    assert cli.main(["-q", "run", str(path)]) == 0
    out = capsys.readouterr().out
    assert "6 measurements; selected K: 1: 6" in out
    with open_bundle(tmp_path / "out") as b:
        assert b.pending == [] and b.bootstrap() is not None and b.has("spectra")
        assert b.spectra_source()["reader"] == "mzml" and b.manifest["notes"] == "first try"
    assert {p.name for p in (tmp_path / "out_tables").iterdir()} == {
        "series.csv", "spectra.csv", "components.csv", "uptake.csv", "hxex_listing.json"}
    assert cli.main(["run", str(path)]) == 1  # the bundle exists
    err = capsys.readouterr().err
    assert "--overwrite" in err and "--keep-previous" in err and "output.bundle" in err
    assert cli.main(["info", "--verify", str(tmp_path / "out")]) == 0
    info = capsys.readouterr().out
    for line in ("project        cli", "input          mzml", "measurements   6",
                 "selected K     1: 6", "bootstrap      yes", "pending        nothing",
                 "review         none", "checksums      ok"):
        assert line in info
    assert "replaced" not in info and "generations" not in info
    assert cli.main(["report", str(tmp_path / "out"), "--figures", str(tmp_path / "f"),
                     "--series", "all", "--format", "png"]) == 0
    assert any(p.suffix == ".png" for p in (tmp_path / "f").iterdir())
    assert not (tmp_path / "out_tables_again").exists()
    assert cli.main(["report", str(tmp_path / "out"), "--tables",
                     str(tmp_path / "out_tables_again")]) == 0
    assert (tmp_path / "out_tables_again" / "series.csv").is_file()
    assert cli.main(["run", "--overwrite", "--device", "cpu", str(path)]) == 0
    assert cli.main(["info", str(tmp_path / "out")]) == 0
    info = capsys.readouterr().out
    assert "replaced       " + str(tmp_path / "out") in info and "changed        nothing" in info
    # --keep-previous moves the bundle and its tables to generation 1 and writes afresh
    (tmp_path / "out_tables" / "mine.txt").write_text("keep")
    assert cli.main(["-q", "run", "--keep-previous", str(path)]) == 0
    assert (tmp_path / "out.1_tables" / "mine.txt").read_text() == "keep"
    assert (tmp_path / "out.1" / "manifest.json").is_file()
    assert (tmp_path / "out_tables" / "series.csv").is_file()
    assert not (tmp_path / "out_tables" / "mine.txt").exists()
    assert cli.main(["info", str(tmp_path / "out")]) == 0
    info = capsys.readouterr().out
    assert "kept as " + str(tmp_path / "out.1") in info and "generations    out.1" in info
    # a file of yours named like a table stops the run before anything is fitted or replaced
    (tmp_path / "out_tables" / "hxex_listing.json").unlink()
    assert cli.main(["-q", "run", "--overwrite", str(path)]) == 1
    assert "series.csv" in capsys.readouterr().err


def test_a_run_from_the_spectra_of_a_bundle(tmp_path, capsys, caplog):
    import numpy as np

    from hxex.io import save_spectra
    from hxex.io.times import DEFAULT_FULLDEUT_TIME

    path = mzml_project(tmp_path)
    assert cli.main(["-q", "run", str(path)]) == 0
    with open_bundle(tmp_path / "out") as b:
        first, binned = b.fit(), b.binned()
        spectra = b.spectra()

    def same_results(bundle, stored="out"):
        with open_bundle(bundle) as b:
            assert b.spectra_source()["reader"] == "mzml"  # the read record is kept
            assert b.spectra_source()["stored_in"].endswith(stored)
            assert np.array_equal(b.binned().intensity, binned.intensity)
            assert np.allclose(b.fit().fits[1].p, first.fits[1].p)

    # the same project file, with the stored spectra in place of the read
    text = path.read_text().replace('peptides = "peptides.csv"',
                                    'peptides = "peptides.csv"\nspectra = "out"')
    (tmp_path / "rerun.toml").write_text(text.replace('bundle = "out"', 'bundle = "out2"'))
    (tmp_path / "run1.mzML").unlink()  # not read again
    capsys.readouterr()
    assert cli.main(["run", str(tmp_path / "rerun.toml")]) == 0
    assert "not used: input peptides, runs" in capsys.readouterr().err
    same_results(tmp_path / "out2")
    # or from the command line, into another bundle
    assert cli.main(["-q", "run", str(path), "--spectra", str(tmp_path / "out"),
                     "--bundle", str(tmp_path / "out3")]) == 0
    same_results(tmp_path / "out3")
    # an [input] of the spectra alone: the format is the stored spectra's
    project_file = tmp_path / "only.toml"
    project_file.write_text(f'project = "again"\n[input]\nspectra = "out"\n'
                            f'[output]\nbundle = "out4"\n{FAST_RUN}')
    assert load_project(project_file).input.format == "mzml"
    assert cli.main(["check", str(project_file)]) == 0
    assert "stored in" in capsys.readouterr().out
    assert cli.main(["-q", "run", str(project_file)]) == 0
    same_results(tmp_path / "out4")
    # a spectra file works as well
    save_spectra(spectra, tmp_path / "s.parquet")
    project_file.write_text(project_file.read_text().replace('"out"', '"s.parquet"'))
    assert cli.main(["-q", "run", str(project_file), "--overwrite"]) == 0
    same_results(tmp_path / "out4", "s.parquet")
    # refused: the bundle it reads from, a missing path, a bundle without raw spectra
    assert cli.main(["run", str(path), "--spectra", str(tmp_path / "out"), "--overwrite"]) == 1
    capsys.readouterr()
    assert cli.main(["run", str(path), "--spectra", str(tmp_path / "nothing")]) == 1
    assert "does not exist" in capsys.readouterr().err
    (tmp_path / "bare").mkdir()
    no_spectra = mzml_project(tmp_path / "bare", output="keep_spectra = false")
    assert cli.main(["-q", "run", str(no_spectra)]) == 0
    assert cli.main(["run", str(path), "--spectra", str(tmp_path / "bare" / "out")]) == 1
    assert "holds no raw spectra" in capsys.readouterr().err
    # the format and the control stamps are the stored record's, not the project file's
    # (which is logged), and the inherited control times follow them
    other = tmp_path / "other.toml"
    other.write_text(f'project = "o"\n[input]\nformat = "specexport"\ndata_dir = "d"\n'
                     f'fulldeut_time = 50.0\nspectra = "out"\n[run.controls]\n'
                     f'[output]\nbundle = "out5"\n{FAST_RUN}')
    with caplog.at_level(logging.WARNING):
        p = load_project(other)
    assert p.input.format == "mzml" and p.input.fulldeut_time == DEFAULT_FULLDEUT_TIME
    assert "read with format = 'mzml'" in caplog.text and "fulldeut_time = 50.0" in caplog.text
    assert p.run.controls.fd_time is None  # a scan format: fd_time is set by hand
    stored = dataclasses.replace(spectra, source={**spectra.source, "reader": "specexport",
                                                  "undeut_time": 0.5, "fulldeut_time": 99.0})
    save_spectra(stored, tmp_path / "s2.parquet")
    other.write_text(other.read_text().replace('spectra = "out"', 'spectra = "s2.parquet"'))
    p = load_project(other)
    assert (p.input.format, p.input.undeut_time, p.input.fulldeut_time) == ("specexport", 0.5,
                                                                            99.0)
    assert (p.run.controls.undeut_time, p.run.controls.fd_time) == (0.5, 99.0)
    p = load_project(other, spectra=tmp_path / "out")  # the command line's --spectra
    assert p.input.format == "mzml" and p.input.spectra == tmp_path / "out"
    assert p.run.controls.fd_time is None
    # a stored path that cannot be read yet leaves the file's own settings until the spectra
    # are read: check reports it as a problem, run refuses it before anything else
    other.write_text(other.read_text().replace('spectra = "s2.parquet"', 'spectra = "gone"'))
    assert load_project(other).input.format == "specexport"
    capsys.readouterr()
    assert cli.main(["check", str(other)]) == 1
    out, err = capsys.readouterr()
    assert "format         specexport" in out and "gone does not exist" in err
    assert cli.main(["run", str(other)]) == 1 and "does not exist" in capsys.readouterr().err
    # a rerun whose picker reaches beyond the m/z the stored spectra were cut to warns
    wide = tmp_path / "wide.toml"
    wide.write_text(text.replace('bundle = "out"', 'bundle = "out6"')
                    .replace("zero_filling = 3", "zero_filling = 12"))
    capsys.readouterr()
    assert cli.main(["run", str(wide)]) == 0
    assert "outside the window the stored spectra were cut to" in capsys.readouterr().err


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
    assert cli.main(["-q", "resume", str(tmp_path / "b"), "--tables", "--figures"]) == 0
    with open_bundle(tmp_path / "b") as b:
        assert b.pending == [] and b.bootstrap() is not None
        assert b.manifest["configs"]["run"]["plots"] is None  # the run's record is unchanged
    assert (tmp_path / "b_tables" / "uptake.csv").is_file()
    assert any((tmp_path / "b_figures").iterdir())  # --figures honoured without run.plots
    assert cli.main(["resume", str(tmp_path / "b")]) == 1  # nothing pending any more


def test_errors_are_messages_not_tracebacks(tmp_path, capsys):
    assert cli.main(["check", str(tmp_path / "missing.toml")]) == 1
    err = capsys.readouterr().err
    assert "cannot read" in err and "Traceback" not in err
    assert cli.main(["info", str(tmp_path)]) == 1
    p = mzml_project(tmp_path)
    (tmp_path / "run1.mzML").unlink()
    assert cli.main(["check", str(p)]) == 1


def test_threads_are_set_for_the_command_only(tmp_path, capsys):
    import torch

    before = torch.get_num_threads()
    path = mzml_project(tmp_path)
    text = path.read_text().replace('project = "cli"', 'project = "cli"\nthreads = 3')
    path.write_text(text)
    assert load_project(path).threads == 3
    assert cli.main(["run", str(path)]) == 0
    assert "using 3 CPU thread(s)" in capsys.readouterr().err
    assert torch.get_num_threads() == before  # restored
    assert cli.main(["run", "--overwrite", "--threads", "2", str(path)]) == 0  # the flag wins
    assert "using 2 CPU thread(s)" in capsys.readouterr().err
    assert cli.main(["report", "--threads", "1", str(tmp_path / "out")]) == 0
    assert torch.get_num_threads() == before
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--threads", "0", str(path)])
    assert exc.value.code == 2
    for bad in ("0", "2.5", "true", '"4"'):
        with pytest.raises(ProjectError, match="threads"):
            project_from(tomllib.loads(f'project = "p"\nthreads = {bad}\n[input]\n'
                                       'format = "specexport"\ndata_dir = "d"\n'), tmp_path)


def _optional_settings(cls, path=()):
    """Every setting of ``cls`` and its nested configs that may be None and is not a switch, as
    (path, class) by the dotted TOML path, the way the project file sees them."""
    from hxex.config import _config_class, _hints, _union_members

    out = {}
    for name, hint in _hints(cls).items():
        members = _union_members(hint)
        if type(None) in members and bool not in members:
            out[(*path, name)] = cls
        sub = _config_class(hint)
        if sub is not None:
            out.update(_optional_settings(sub, (*path, name)))
    return out


# what an optional setting needs beside it to be valid with false
_COMPANIONS = {
    ("fit", "noise", "percent"): "sd = 50.0",
    ("controls", "quench", "initial_fraction"): "ph_read = 2.5\ntemperature = 0.0\n"
                                                 "d2o_fraction = 0.0",
}


def test_false_means_unset_only_where_a_setting_admits_none(tmp_path):
    head = 'project = "p"\n[input]\nformat = "specexport"\ndata_dir = "d"\n'

    def project(text):
        return project_from(tomllib.loads(head + text), tmp_path)

    settings = _optional_settings(RunConfig)
    sections = {p for p in settings if p[0] in ("bootstrap", "controls", "plots", "quality")}
    assert {("fit", "noise", "sd"), ("fit", "noise", "percent"), ("controls", "fd_time"),
            ("controls", "retention_min"), ("controls", "quench"),
            ("quality", "max_mz_offset_ppm"), ("picker", "noise"), ("max_chunk",),
            ("controls", "quench", "initial_fraction")} <= set(settings)
    assert ("plots", "corrected") not in settings  # a switch that may be unset: false is off
    for path in settings:
        if len(path) == 1 and path[0] in sections:
            continue  # the optional stages themselves: false turns them off (tested below)
        table = f"[run.{'.'.join(path[:-1])}]\n" if len(path) > 1 else "[run]\n"
        p = project(table + f"{path[-1]} = false\n{_COMPANIONS.get(path, '')}\n")
        value = p.run
        for name in path:
            value = getattr(value, name)
        assert value is None, path
    # the switches keep an explicit false
    p = project("[run]\ndrop_no_signal = false\nquality = false\n[run.fit]\nnested_init = false\n"
                "[run.plots]\ncorrected = false\noverview = false\n")
    assert p.run.drop_no_signal is False and p.run.quality is None
    assert p.run.fit.nested_init is False
    assert p.run.plots.corrected is False and p.run.plots.overview is False
    # a fixed noise level: sd set, the default percent removed
    p = project("[run.fit.noise]\nsd = 50.0\npercent = false\n")
    assert p.run.fit.noise.sd == 50.0 and p.run.fit.noise.percent is None
    with pytest.raises(ProjectError, match="sd"):
        project("[run.fit.noise]\nsd = 50.0\n")  # both set
    p = project("[run.controls]\nfd_time = false\nquench = false\n[run.quality]\n"
                "max_mz_offset_ppm = false\n")
    assert p.run.controls.fd_time is None and p.run.controls.quench is None
    assert p.run.quality.max_mz_offset_ppm is None


def test_control_times_default_to_the_input_times(tmp_path):
    def project(text, fmt='format = "specexport"\ndata_dir = "d"\nundeut_time = 0.5\n'
                                'fulldeut_time = 99.0\n'):
        return project_from(tomllib.loads(f'project = "p"\n[input]\n{fmt}' + text), tmp_path)

    p = project("[run.controls]\nd2o_fraction = 0.85\n")
    assert (p.run.controls.undeut_time, p.run.controls.fd_time) == (0.5, 99.0)
    assert p.run.reference_time == 0.5
    p = project("[run]\ncontrols = true\n")
    assert (p.run.controls.undeut_time, p.run.controls.fd_time) == (0.5, 99.0)
    p = project("[run.controls]\nundeut_time = 1.0\nfd_time = 100.0\n")  # explicit values win
    assert (p.run.controls.undeut_time, p.run.controls.fd_time) == (1.0, 100.0)
    p = project("[run.controls]\nundeut_time = false\nfd_time = false\n")  # no such controls
    assert (p.run.controls.undeut_time, p.run.controls.fd_time) == (None, None)
    assert project("").run.controls is None
    # a scan format: its runs carry the times; the undeuterated ones are at [run] undeut_time
    scan = ('format = "mzml"\npeptides = "p.csv"\n[[input.runs]]\npath = "a"\nsample = "S"\n'
            "time = 0.0\n")
    p = project("[run]\nundeut_time = 2.0\n[run.controls]\n", scan)
    assert (p.run.controls.undeut_time, p.run.controls.fd_time) == (2.0, None)
    p = project("[run.controls]\nfd_time = 1e6\n", scan)
    assert (p.run.controls.undeut_time, p.run.controls.fd_time) == (0.0, 1e6)


def test_template_unset_lines_are_valid_toml_when_uncommented(tmp_path):
    import re

    from hxex.bootstrap import BootstrapConfig

    for fmt in ("specexport", "mzml"):
        text = template(fmt)
        head, run = text.split("# Run settings", 1)
        run = re.sub(r"^# (\[|[a-z_]+ = )", r"\1", run, flags=re.M)  # every setting on
        assert "(not set)" not in run
        p = project_from(tomllib.loads(head + "# Run settings" + run), tmp_path)
        assert p.run.bootstrap == BootstrapConfig() and p.run.plots is not None
        assert p.run.max_chunk is None and p.run.fit.noise.sd is None
        assert p.run.controls.quench is None and p.run.controls.retention_min is None
        assert p.run.controls.fd_time == 1e6
        assert p.run.controls.undeut_time == 0.0
        if fmt == "mzml":
            assert p.input.scans.min_time is None and p.input.scans.max_spectra is None


def test_runtime_errors_are_messages_too(tmp_path, capsys, monkeypatch):
    def fail(args):
        raise RuntimeError("CUDA is not available")

    monkeypatch.setitem(cli.COMMANDS, "info", fail)
    assert cli.main(["info", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "CUDA is not available" in err and "Traceback" not in err
