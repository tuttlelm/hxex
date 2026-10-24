# SPDX-License-Identifier: GPL-3.0-or-later
"""The command line and the project file: templates, checks, a whole run of a synthetic mzML
project, resuming a stopped bootstrap, reports from a bundle and bundle summaries."""

import dataclasses
import logging
import shutil
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
        f"x,,{PEPTIDE},0001-0010,1,10,2\n")
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


def test_toml_values_are_type_checked(tmp_path, capsys):
    """A value of the wrong form is reported with its table, key, value and the expected form;
    a table's own value errors name the table."""
    def project(text):
        return project_from(tomllib.loads(
            'project = "p"\n[input]\nformat = "specexport"\ndata_dir = "d"\n' + text), tmp_path)

    for text, message in [
        ('[run.fit]\nk_max = "3"\n', '[run.fit] k_max = "3": expected a whole number'),
        ("[run.fit]\nk_max = 2.5\n", "[run.fit] k_max = 2.5: expected a whole number"),
        ("[run]\nn_max_factor = true\n", "[run] n_max_factor = true: expected a number"),
        ("[run.controls]\nd2o_fraction = false\n",
         "[run.controls] d2o_fraction = false: expected a number (false stands for unset"),
        ("[figures]\nseries = 3\n", "[figures] series = 3: expected text or a list"),
        ("[figures]\nk = 0\n", "[figures]: k must be a whole number"),
        ("[run]\nbootstrap = 3\n", "[run] bootstrap = 3: expected a table, true or false"),
        ("[run.fit]\nk_min = 0\n", "[run.fit]: need 1 <= k_min"),
        ("[run.bootstrap]\nci = 0.95\n", "[run.bootstrap]: ci is a percentage"),
        ('[analysis]\nalpha = "x"\n', '[analysis] alpha = "x": expected a number'),
        ('[analysis.filters]\nmin_population = "a"\n',
         '[analysis.filters] min_population = "a": expected a number'),
    ]:
        with pytest.raises(ProjectError) as info:
            project(text)
        assert message in str(info.value), (text, str(info.value))
    assert project("[run.fit]\nscreen = false\nk_max = 2\n").run.fit.k_max == 2
    # a file for hxex analyse only may leave the project name out
    only = tmp_path / "only.toml"
    only.write_text('[analysis]\nreference = "WT"\n[output]\nbundle = "res"\n')
    assert load_project(only, analysis_only=True).project == ""
    with pytest.raises(ProjectError, match="project must be a name"):
        load_project(only)
    # hxex check says what a stage being off means
    assert cli.main(["check", str(only)]) == 0
    out = capsys.readouterr().out
    assert "controls       off: no back-exchange correction" in out and "[run.controls]" in out


def test_template_lines_carry_their_meaning_and_unit():
    """Every generated [run.*] and [input.scans] setting has a trailing comment, sections have
    comment lines, and the [analysis] scalars come before its sub-tables."""
    import re

    text = template("mzml")
    lines = text.split("[run]", 1)[1].splitlines()
    for line, following in zip(lines, lines[1:] + [""], strict=True):
        if re.match(r"^#? ?[a-z_]+ = ", line) and not line.startswith(("label", "base", "delta")):
            # the comment follows at the comment column, or on the next line for a long setting
            assert "  # " in line or re.match(r"^#?\s{20,}# \S", following), line
    assert "half-width (ppm)" in text and "not a resolving power" in text
    assert "Deuterium retention here is the fraction" in text
    assert "# Peak picking" in text and "Modified residues, written in sequences as modX" in text
    assert "TYPE:LENGTHS:CHARGES" in text and "composition decides" in text
    analysis = text.split("# [analysis]", 1)[1].split("# Run settings", 1)[0]
    first_table = analysis.index("# [analysis.")
    assert analysis.index("# protection = true") < first_table
    assert "protection_residues" not in text
    assert "exchange-clock min_time" in text  # the two clocks cross-refer


def test_project_files_are_checked(tmp_path):
    def project(text):
        return project_from(tomllib.loads('project = "p"\n' + text), tmp_path)

    p = project('[input]\nformat = "specexport"\ndata_dir = "d"\n'
                "[run.bootstrap]\nn_boot = 7\n[run]\nquality = false\nmax_chunk = 5\n"
                '[run.fit.noise]\nmodel = "poisson"\n[output]\nbundle = "res"\nfigures = false\n'
                '[figures]\nseries = "all"\nk = 2\n')
    assert p.input.data_dir == tmp_path / "d"
    assert p.run.bootstrap.n_boot == 7 and p.run.bootstrap.n_restarts == 3  # the rest default
    assert p.run.quality is None and p.run.max_chunk == 5
    assert p.run.fit.noise.model == "poisson" and p.run.fit.k_max == 3
    assert p.figures.series == "all" and p.figures.k == 2  # the old figures = false is fine
    # the run draws no figures: the settings that asked it to are refused with the way now
    for moved in ('[output]\nfigures = true\n', '[output]\nfigures = "figs"\n',
                  '[run.plots]\nseries = "all"\n'):
        with pytest.raises(ProjectError, match=r"\[figures\] table now.*hxex report"):
            project('[input]\nformat = "specexport"\ndata_dir = "d"\n' + moved)
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
    assert project('time = "1 min"\n').input.runs[0].time == 60.0
    for bad, match in (("time = false", "time must be a number"), ("time = true", "got True"),
                       ('time = "60 parsecs"', "time must be a number"), ("time = inf", "got inf"),
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
    table.write_text(head + f"x,,{PEPTIDE},0001-0010,1,10,2,5.5,6.75\n"
                     f"x,,{PEPTIDE},0001-0010,1,10,3,,\n")
    got = scan_targets(load_project(path))
    first, second = got[:len(got) // 2], got[len(got) // 2:]  # each peptide, then its fragments
    assert first[0].ms_level == 1 and second[0].entry.charge == 3 and len(first) > 1
    assert {t.rt_window for t in first} == {(330.0, 405.0)}  # minutes -> seconds, fragments too
    # without columns: the project's window for the peptide, the fragments' own (none)
    assert second[0].rt_window == (0.0, 30.0)
    assert {t.rt_window for t in second[1:]} == {None}
    for row, match in ((",6.0", "both or neither"), ("7.0,6.0", "after the start"),
                       ("a,6.0", "not numbers")):
        table.write_text(head + f"x,,{PEPTIDE},0001-0010,1,10,2,{row}\n")
        with pytest.raises(ReadError, match="line 2") as err:
            scan_targets(load_project(path))
        assert match in str(err.value.__cause__)


def test_peptides_with_a_modification_and_fragments_are_refused(tmp_path, capsys):
    """A row is both a peptide and the parent of its fragments: with [[input.fragments]], a
    modification is refused for its MS1 spectra too, every line named by check and run alike;
    without fragments the same table is read."""
    from hxex.config import scan_targets

    head = "file,sample,peptide,peptide_range,start_seq,end_seq,charge,modification\n"
    rows = (f"x,,{PEPTIDE},0001-0010,1,10,2,ion_type:M\n"
            f"x,,{PEPTIDE},0001-0010,1,10,3,undeut_mz:400.2\n"
            f"x,,{PEPTIDE},0001-0010,1,10,4,H:1\n")
    path = mzml_project(tmp_path)
    (tmp_path / "peptides.csv").write_text(head + rows)
    assert len(scan_targets(load_project(path))) == 3
    path = mzml_project(tmp_path, extra_input='[[input.fragments]]\nions = "c:2-3:1"\n')
    (tmp_path / "peptides.csv").write_text(head + rows)
    assert cli.main(["check", str(path)]) == 1
    err = capsys.readouterr().err
    assert "line 3: " in err and "(charge 3) has modification 'undeut_mz:400.2'" in err
    assert "line 4: " in err and "line 2: " not in err
    assert "at MS1 either" in err and "parents = false" in err and "2 problem(s)" in err
    assert cli.main(["run", str(path)]) == 1
    err = capsys.readouterr().err
    assert "line 3: " in err and "line 4: " in err and "at MS1 either" in err
    assert not (tmp_path / "out").exists()


def test_check_shows_what_would_be_read(tmp_path, capsys):
    path = mzml_project(tmp_path, extra_input='[input.target]\nfilter_excludes = "sid"')
    assert cli.main(["check", str(path)]) == 0
    out = capsys.readouterr().out
    assert "targets        1 from" in out and "6 scans; 1 of 1 targets have scans" in out
    assert out.strip().endswith("ok")
    assert "outputs        none yet: hxex run writes out" in out
    # existing results are information, not problems: each command refuses by itself
    path.write_text(path.read_text() + '\n[analysis]\nks = [1]\noverlay = ["avg"]\n')
    assert cli.main(["-q", "run", str(path)]) == 0
    assert cli.main(["-q", "analyse", str(path)]) == 0
    capsys.readouterr()
    assert cli.main(["check", str(path)]) == 0
    out = capsys.readouterr().out
    assert "outputs        out exists: hxex run needs --overwrite or --keep-previous" in out
    assert ("out_tables: up to date; hxex run and hxex report need --overwrite or "
            "--keep-previous") in out
    assert "out_analysis: up to date; hxex analyse needs --overwrite" in out
    assert out.strip().endswith("ok")
    path = mzml_project(tmp_path, output="keep_previous = true")
    assert cli.main(["check", str(path)]) == 0
    assert "out exists: [output] keep_previous moves it to NAME.N first" in \
        capsys.readouterr().out


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
        f"x,,{PEPTIDE},0001-0010,1,10,2\nx,,AC(ox)DE,0001-0005,1,5,2\n")
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
    assert "2 problem(s)" in err and "outputs        out exists" in out


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
    # given the project file, hxex report draws with its [figures] settings (no new run)
    assert not (tmp_path / "out_figures").exists()  # hxex run drew none
    path.write_text(path.read_text() + '\n[figures]\nseries = "all"\nformat = "svg"\n'
                    'kinds = ["uptake"]\n')
    assert cli.main(["-q", "report", str(path), "--figures", "--k", "1"]) == 0
    assert {p.suffix for p in (tmp_path / "out_figures").iterdir()} == {".svg", ".json"}
    capsys.readouterr()
    # drawing again into a folder with figures is refused, as a run into a bundle is
    assert cli.main(["-q", "report", str(path), "--figures"]) == 1
    assert "out_figures holds the figures written before: --overwrite" in capsys.readouterr().err
    assert cli.main(["-q", "report", str(path), "--figures", "--overwrite", "--k", "4"]) == 1
    assert "K = 4 was not fitted" in capsys.readouterr().err
    assert cli.main(["run", "--overwrite", "--device", "cpu", str(path)]) == 0
    assert "out_figures holds figures of other results" in capsys.readouterr().err
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
    # the format is the stored record's, not the project file's (which is logged)
    other = tmp_path / "other.toml"
    other.write_text(f'project = "o"\n[input]\nformat = "specexport"\ndata_dir = "d"\n'
                     f'spectra = "out"\n[run.controls]\n'
                     f'[output]\nbundle = "out5"\n{FAST_RUN}')
    with caplog.at_level(logging.WARNING):
        p = load_project(other)
    assert p.input.format == "mzml"
    assert "read with format = 'mzml'" in caplog.text
    stored = dataclasses.replace(spectra, source={**spectra.source, "reader": "specexport"})
    save_spectra(stored, tmp_path / "s2.parquet")
    other.write_text(other.read_text().replace('spectra = "out"', 'spectra = "s2.parquet"'))
    p = load_project(other)
    assert p.input.format == "specexport"
    p = load_project(other, spectra=tmp_path / "out")  # the command line's --spectra
    assert p.input.format == "mzml" and p.input.spectra == tmp_path / "out"
    # the control times of project files written before the roles are refused by name
    old = other.read_text().replace('data_dir = "d"\n', 'data_dir = "d"\nfulldeut_time = 50.0\n')
    other.write_text(old)
    with pytest.raises(ProjectError, match="fulldeut_time.*no longer identified by a time"):
        load_project(other)
    other.write_text(other.read_text().replace("fulldeut_time = 50.0\n", ""))
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
    assert cli.main(["-q", "resume", str(tmp_path / "b"), "--tables"]) == 0
    with open_bundle(tmp_path / "b") as b:
        assert b.pending == [] and b.bootstrap() is not None
    assert (tmp_path / "b_tables" / "uptake.csv").is_file()
    assert not (tmp_path / "b_figures").exists()  # figures come from hxex report
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
    assert cli.main(["report", "--threads", "1", "--overwrite", str(tmp_path / "out")]) == 0
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
    assert {("fit", "noise", "sd"), ("fit", "noise", "percent"), ("controls", "fulldeut_at"),
            ("controls", "retention_min"), ("controls", "quench"),
            ("quality", "max_mz_offset_ppm"), ("picker", "noise_threshold"), ("max_chunk",),
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
                "[figures]\ncorrected = false\noverview = false\n")
    assert p.run.drop_no_signal is False and p.run.quality is None
    assert p.run.fit.nested_init is False
    assert p.figures.corrected is False and p.figures.overview is False
    # a fixed noise level: sd set, the default percent removed
    p = project("[run.fit.noise]\nsd = 50.0\npercent = false\n")
    assert p.run.fit.noise.sd == 50.0 and p.run.fit.noise.percent is None
    with pytest.raises(ProjectError, match="sd"):
        project("[run.fit.noise]\nsd = 50.0\n")  # both set
    p = project("[run.controls]\nfulldeut_at = false\nquench = false\n[run.quality]\n"
                "max_mz_offset_ppm = false\n")
    assert p.run.controls.fulldeut_at is None and p.run.controls.quench is None
    assert p.run.quality.max_mz_offset_ppm is None


def test_controls_are_roles_and_times_take_units(tmp_path):
    head = 'project = "p"\n[input]\nformat = "specexport"\ndata_dir = "d"\n'

    def project(text):
        return project_from(tomllib.loads(head + text), tmp_path)

    p = project("[run.controls]\nd2o_fraction = 0.85\n")
    assert (p.run.controls.undeut_at, p.run.controls.fulldeut_at) == (None, None)
    p = project('[run.controls]\nfulldeut_at = "20 h"\nundeut_at = 4\n')
    assert (p.run.controls.undeut_at, p.run.controls.fulldeut_at) == (4.0, 72000.0)
    p = project('[run]\nenvelope_fallback_time = "1 min"\n[analysis]\ntimes = ["1 min", 20]\n'
                'exclude_times = ["20 h"]\ncoverage_times = ["4 s"]\n')
    assert p.run.envelope_fallback_time == 60.0 and p.analysis.times == (60.0, 20.0)
    assert p.analysis.exclude_times == (72000.0,) and p.analysis.coverage_times == (4.0,)
    p = project('time_scale = "order"\n')
    assert p.input.time_scale == "order"
    for text, message in [
        ("undeut_time = 0.0\n", "undeut_time.*no longer identified"),
        ("[run]\nundeut_time = 0.0\n", "undeut_time.*no longer identified"),
        ("[run.controls]\nfd_time = 1e6\n", "fd_time.*no longer identified"),
        ("[run.controls]\natol = 1.0\n", "atol.*no longer identified"),
        ("[analysis.controls]\nundeut_time = 0.0\n", "undeut_time.*no longer identified"),
        ('[run.controls]\nfulldeut_at = "20 x"\n', "unknown time unit"),
        ('[run.controls]\nfulldeut_at = true\n', "fulldeut_at"),
        ('time_scale = "minutes"\n', "time_scale must be one of"),
        ('[analysis]\ntimes = ["soon"]\n', "analysis.times"),
    ]:
        with pytest.raises(ProjectError, match=message):
            project(text)


def test_scan_runs_take_roles(tmp_path):
    head = ('project = "p"\n[input]\nformat = "mzml"\npeptides = "p.csv"\n')

    def project(runs):
        return project_from(tomllib.loads(head + runs), tmp_path)

    p = project('[[input.runs]]\npath = "un.mzML"\nsample = "WT"\nrole = "undeut"\n'
                '[[input.runs]]\npath = "td.mzML"\nsample = "WT"\nrole = "fulldeut"\n'
                'time = "27 h"\n[[input.runs]]\npath = "k.mzML"\nsample = "WT"\n'
                'time_offset = "2 s"\nusable_from = "20 s"\n')
    un, td, kin = p.input.runs
    assert (un.role, un.time, un.fixed) == ("undeut", None, True)
    assert (td.role, td.time, td.fixed) == ("fulldeut", 97200.0, True)
    assert (kin.role, kin.time, kin.fixed, kin.time_offset, kin.usable_from) == (
        "", None, False, 2.0, 20.0)
    with pytest.raises(ProjectError, match='role = "undeut"'):
        project('[[input.runs]]\npath = "un.mzML"\nsample = "WT"\ntime = "undeut"\n')
    with pytest.raises(ProjectError, match="role must be"):
        project('[[input.runs]]\npath = "un.mzML"\nsample = "WT"\nrole = "control"\n')
    with pytest.raises(ProjectError, match="time"):
        project('[[input.runs]]\npath = "un.mzML"\nsample = "WT"\ntime = "5 parsecs"\n')


def test_template_unset_lines_are_valid_toml_when_uncommented(tmp_path):
    import re

    from hxex.bootstrap import BootstrapConfig

    for fmt in ("specexport", "mzml"):
        text = template(fmt)
        head, run = text.split("# Run settings", 1)
        run = re.sub(r"^# (\[|[a-z_]+ = )", r"\1", run, flags=re.M)  # every setting on
        assert "(not set)" not in run
        p = project_from(tomllib.loads(head + "# Run settings" + run), tmp_path)
        assert p.run.bootstrap == BootstrapConfig() and p.figures.k == 2
        assert p.run.max_chunk is None and p.run.fit.noise.sd is None
        assert p.run.controls.quench is not None and p.run.controls.fd_mode == "kint"
        assert p.run.controls.retention_min is None
        assert p.run.controls.fulldeut_at == 72000.0  # "20 h" in the template
        assert p.run.controls.undeut_at == 300.0  # "5 min"
        if fmt == "mzml":
            assert [r.role for r in p.input.runs] == ["undeut", ""]
            assert p.input.runs[0].time is None and p.input.runs[1].time == 60.0
        if fmt == "mzml":
            assert p.input.scans.min_time is None and p.input.scans.max_spectra is None


def test_runtime_errors_are_messages_too(tmp_path, capsys, monkeypatch):
    def fail(args):
        raise RuntimeError("CUDA is not available")

    monkeypatch.setitem(cli.COMMANDS, "info", fail)
    assert cli.main(["info", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "CUDA is not available" in err and "Traceback" not in err


def test_check_and_info_report_the_roles_and_the_time_scale(tmp_path, capsys):
    """A scan-format run's role is named by hxex check; after the run, hxex info counts the
    control spectra and says whether the times are real ([input] time_scale)."""
    path = mzml_project(tmp_path, extra_input=(
        'time_scale = "order"\n\n[[input.runs]]\npath = "run1.mzML"\nsample = "WT"\n'
        'role = "undeut"\n'))
    assert cli.main(["check", str(path)]) == 0
    out = capsys.readouterr().out
    assert "run run1.mzML: sample WT, rep 1, undeuterated control, 6 scans" in out
    assert "run run1.mzML: sample WT, rep 1, scan times, 6 scans" in out
    assert "times          order: the times only order the points" in out
    assert cli.main(["-q", "run", str(path)]) == 0
    capsys.readouterr()
    assert cli.main(["info", str(tmp_path / "out")]) == 0
    info = capsys.readouterr().out
    assert "roles          1 undeuterated (1 series)" in info
    assert "times          order: the times only order the points" in info
    with open_bundle(tmp_path / "out") as b:
        assert b.spectra_source()["time_scale"] == "order"
    # the uptake figure of such a run is drawn by index without any setting
    assert cli.main(["report", str(tmp_path / "out"), "--figures", str(tmp_path / "f"),
                     "--series", "all", "--format", "png"]) == 0
    assert any(p.name.endswith("_uptake.png") for p in (tmp_path / "f").iterdir())
    # a rerun from the stored spectra keeps the scale unless the file says otherwise
    rerun = tmp_path / "rerun.toml"
    rerun.write_text('project = "cli"\n[input]\nspectra = "out"\n[output]\nbundle = "again"\n')
    assert load_project(rerun).input.time_scale == "order"
    rerun.write_text('project = "cli"\n[input]\nspectra = "out"\ntime_scale = "seconds"\n'
                     '[output]\nbundle = "again"\n')
    assert load_project(rerun).input.time_scale == "seconds"


def test_check_summarises_the_promoted_controls(tmp_path):
    from hxex.config import describe

    text = ('project = "p"\n[input]\nformat = "specexport"\ndata_dir = "d"\n'
            '[run.controls]\nfulldeut_at = "20 h"\nfulldeut_marked = false\n')
    summary = describe(project_from(tomllib.loads(text), tmp_path))
    assert summary["controls"].startswith("on; fulldeut_at = 20 h (the spectra at that time")
    assert "fulldeut_marked = false (the files marked fulldeut do not)" in summary["controls"]
    assert summary["times"] == "seconds"


def test_check_counts_the_control_files_of_a_specexport_folder(tmp_path, capsys):
    """For a SpecExport folder hxex check counts the control files per sample from their
    names (Non, Full) without reading them."""
    data = tmp_path / "d"
    (data / "WT" / f"0001-0010-{PEPTIDE}").mkdir(parents=True)
    (data / "WT.fasta").write_text(f">WT\n{PEPTIDE}\n")
    for name in ("Non-D-1-z2.csv", "Non-D-2-z2.csv", "Full-D-1-z2.csv", "60s-1-z2.csv",
                 "60s-1-z3.csv"):
        (data / "WT" / f"0001-0010-{PEPTIDE}" / name).write_text("")
    path = tmp_path / "hxex.toml"
    path.write_text('project = "p"\n[input]\nformat = "specexport"\ndata_dir = "d"\n'
                    '[output]\nbundle = "out"\n')
    assert cli.main(["check", str(path)]) == 0
    out = capsys.readouterr().out
    assert "entries        2 in" in out
    assert "  WT: 2 entries, 2 undeuterated and 1 fully deuterated control files" in out


def test_record_paths_are_checked_for_the_format(tmp_path):
    """record_paths = "relative" is for the scan formats; a folder format refused it only when
    the run began, after hxex check had passed."""
    def project(text):
        return project_from(tomllib.loads('project = "p"\n[input]\n' + text), tmp_path)

    with pytest.raises(ProjectError, match="'path', 'name' for specexport.*mzml and thermo"):
        project('format = "specexport"\ndata_dir = "d"\nrecord_paths = "relative"\n')
    assert project('format = "specexport"\ndata_dir = "d"\nrecord_paths = "name"\n')
    runs = '[[input.runs]]\npath = "a.mzML"\nsample = "S"\n'
    assert project(f'format = "mzml"\npeptides = "p.csv"\nrecord_paths = "relative"\n{runs}')
    with pytest.raises(ProjectError, match="'path', 'relative', 'name' for mzml"):
        project(f'format = "mzml"\npeptides = "p.csv"\nrecord_paths = "full"\n{runs}')


def test_a_peptide_is_read_from_the_runs_of_its_sample(tmp_path, capsys):
    """The peptide table's sample ties a row to the runs of that sample (empty: every run);
    file and sample may be left out; a sample no run has, or a row listed twice for one run,
    is refused, naming the line."""
    from hxex.config import scan_targets

    path = mzml_project(tmp_path)
    table = tmp_path / "peptides.csv"
    head = "file,sample,peptide,peptide_range,start_seq,end_seq,charge\n"
    row = f"{PEPTIDE},0001-0010,1,10,2\n"
    table.write_text("peptide,peptide_range,start_seq,end_seq,charge\n" + row)
    assert [t.entry.sample for t in scan_targets(load_project(path))] == [""]
    table.write_text(head + "x,WT," + row)
    assert [t.entry.sample for t in scan_targets(load_project(path))] == ["WT"]
    for rows, match in ((["x,M,"], "line 2.*sample 'M' is not the sample of any.*WT"),
                        (["x,WT,", "x,WT,"], "line 3.*listed again for sample WT"),
                        (["x,WT,", "x,,"], "line 3.*listed again for every run"),
                        (["x,,", "x,WT,"], "line 3.*listed again")):
        table.write_text(head + "".join(r + row for r in rows))
        with pytest.raises(ProjectError, match=match):
            scan_targets(load_project(path))
    # a run of another sample reads none of WT's peptides, and hxex check says so
    text = path.read_text().replace('[output]', '[[input.runs]]\npath = "run1.mzML"\n'
                                    'sample = "M"\n\n[output]')
    path.write_text(text)
    table.write_text(head + "x,WT," + row)
    assert cli.main(["check", str(path)]) == 1
    assert "no peptide of the table is for sample M" in capsys.readouterr().err


def test_a_windows_path_in_double_quotes_gets_a_hint(tmp_path):
    """The commonest mistake of a user new to TOML: backslashes between double quotes."""
    path = tmp_path / "hxex.toml"
    path.write_text('project = "p"\n[input]\nformat = "specexport"\n'
                    'data_dir = "C:\\Users\\me\\data"\n')
    with pytest.raises(ProjectError, match="not valid TOML.*forward slashes.*single quotes"):
        load_project(path)
    path.write_text('project = "p"\n[input]\nformat = "specexport"\n'
                    "data_dir = 'C:\\Users\\me\\data'\n")
    assert str(load_project(path).input.data_dir).endswith("C:\\Users\\me\\data")
    path.write_text('project = "p"\n[input\n')
    with pytest.raises(ProjectError, match="not valid TOML") as info:
        load_project(path)
    assert "backslash" not in str(info.value)


def test_a_bundle_zipped_by_the_operating_system_is_read(tmp_path, capsys):
    """Windows' "Compressed (zipped) folder" and macOS "Compress" put the bundle folder inside
    the zip (macOS adds __MACOSX and .DS_Store); hxex reads it as it reads its own zips. A zip
    with two folders is not a bundle."""
    import shutil
    import zipfile

    path = mzml_project(tmp_path)
    assert cli.main(["-q", "run", str(path)]) == 0
    folder = tmp_path / "out"
    by_os = shutil.make_archive(str(tmp_path / "by_os"), "zip", root_dir=tmp_path,
                                base_dir="out")
    with zipfile.ZipFile(by_os, "a") as zf:
        zf.writestr("__MACOSX/out/._manifest.json", b"junk")
        zf.writestr("out/.DS_Store", b"junk")
    with open_bundle(folder) as a, open_bundle(by_os) as b:
        assert a.manifest == b.manifest and a.fit().spectrum_id.tolist() == \
            b.fit().spectrum_id.tolist()
    capsys.readouterr()
    assert cli.main(["info", by_os]) == 0 and "measurements   6" in capsys.readouterr().out
    assert cli.main(["-q", "report", by_os, "--tables", str(tmp_path / "t")]) == 0
    assert (tmp_path / "t" / "spectra.csv").is_file()
    # a zip of two folders (one a bundle) is not a bundle; made from a folder of its own and
    # written outside it (a zip written into the folder it archives reads itself while it
    # grows: on Windows it filled the disk)
    pair = tmp_path / "pair"
    shutil.copytree(folder, pair / "out")
    (pair / "other").mkdir()
    (pair / "other" / "notes.txt").write_text("not a bundle")
    two = shutil.make_archive(str(tmp_path / "two"), "zip", root_dir=pair)
    assert {n.split("/")[0] for n in zipfile.ZipFile(two).namelist()} == {"out", "other"}
    with pytest.raises(Exception, match="manifest"):
        open_bundle(two)


def test_one_item_of_a_list_may_be_written_with_single_brackets(tmp_path):
    """A list of tables ([[name]]) with one item, written [name] as most headings are, is read
    as that one item; a single-bracket heading written twice is refused with a hint."""
    def project(text):
        return project_from(tomllib.loads('project = "p"\n' + text), tmp_path)

    scan = '[input]\nformat = "mzml"\npeptides = "p.csv"\n'
    for brackets in ("[input.runs]", "[[input.runs]]"):
        p = project(scan + f'{brackets}\npath = "a.mzML"\nsample = "WT"\n')
        assert [r.sample for r in p.input.runs] == ["WT"]
    folder = '[input]\nformat = "specexport"\ndata_dir = "d"\n'
    p = project(folder + '[analysis.regions]\nname = "N"\nstart = 1\nend = 63\n'
                '[analysis.sites]\nresidue = 19\nlabel = "S19"\n')
    assert [r.name for r in p.analysis.annotations.regions] == ["N"]
    assert [s.residue for s in p.analysis.annotations.sites] == [19]
    p = project(folder + '[run.registry.residues]\nlabel = "meK"\nbase = "K"\n'
                'delta = { C = 1, H = 2 }\n')
    assert "meK" in {r["label"] for r in p.run.registry.to_dict()["residues"]}
    path = tmp_path / "hxex.toml"
    path.write_text(scan + '[input.runs]\npath = "a"\nsample = "A"\n'
                    '[input.runs]\npath = "b"\nsample = "B"\n')
    with pytest.raises(ProjectError, match=r"write each under \[\[input.runs\]\]"):
        load_project(path)
    path.write_text(scan + '[input]\nformat = "mzml"\n')
    with pytest.raises(ProjectError, match=r"\[input\] may appear only once"):
        load_project(path)


def test_the_analysis_folder_is_replaced_and_moved_as_the_tables_are(tmp_path, capsys):
    """hxex analyse replaces the files of the earlier analysis (a file it no longer writes
    goes) and keeps the user's own; --keep-previous moves the analysis folder with the bundle
    and its tables (out.1_analysis)."""
    import json

    from hxex.io import output_listing

    path = mzml_project(tmp_path)
    path.write_text(path.read_text() + '\n[analysis]\nks = [1]\noverlay = ["avg"]\n')
    assert cli.main(["-q", "run", str(path)]) == 0
    folder = tmp_path / "out_analysis"
    assert cli.main(["-q", "analyse", str(path)]) == 0
    listed = output_listing(folder)
    assert listed and "species_uptake.csv" in listed
    # the settings used go with the analysis: its own, and a copy of the run's
    assert {"analysis_settings_used.toml", "run_settings_used.toml"} <= set(listed)
    used = load_project(folder / "analysis_settings_used.toml", analysis_only=True)
    assert used.analysis == load_project(path).analysis
    run_copy = (folder / "run_settings_used.toml").read_text()
    assert run_copy == (tmp_path / "out" / "run_settings_used.toml").read_text()
    # a figure an earlier analysis wrote and this one does not, and a file of the user's
    (folder / "woods_old.pdf").write_text("stale")
    (folder / "my_notes.txt").write_text("mine")
    data = json.loads((folder / "hxex_listing.json").read_text())
    data["files"].append("woods_old.pdf")
    (folder / "hxex_listing.json").write_text(json.dumps(data))
    assert cli.main(["-q", "analyse", str(path)]) == 1  # the folder holds an analysis
    assert cli.main(["-q", "analyse", str(path), "--overwrite"]) == 0
    assert not (folder / "woods_old.pdf").exists() and (folder / "my_notes.txt").exists()
    capsys.readouterr()
    assert cli.main(["-q", "run", str(path), "--overwrite"]) == 0
    assert "out_analysis holds an analysis of other results" in capsys.readouterr().err
    # the analysis keeps the settings of the results it was made from
    assert (folder / "run_settings_used.toml").read_text() == run_copy
    with open_bundle(tmp_path / "out") as b:
        assert f"created {b.manifest['created']}" in (
            tmp_path / "out" / "run_settings_used.toml").read_text()
    assert cli.main(["-q", "run", str(path), "--keep-previous"]) == 0
    assert (tmp_path / "out.1_analysis" / "species_uptake.csv").is_file()
    assert (tmp_path / "out.1_tables").is_dir() and not folder.exists()


def test_resume_adds_or_redoes_a_bootstrap_from_the_project_file(tmp_path, capsys):
    """hxex resume PROJECT adds a bootstrap with the project's [run.bootstrap] to a complete
    bundle that has none, or replaces the one there; the fits are kept. Given the bundle alone,
    a complete bundle is refused with the way to do it."""
    path = mzml_project(tmp_path)
    with_boot = path.read_text()
    start = with_boot.index("[run.bootstrap]")
    path.write_text(with_boot[:start])  # the same project without its bootstrap
    assert cli.main(["-q", "run", str(path)]) == 0
    with open_bundle(tmp_path / "out") as b:
        assert b.bootstrap() is None
        ids, ks = list(b.fit().spectrum_id), list(b.fit().selected_K)
    assert cli.main(["resume", str(tmp_path / "out")]) == 1
    assert "hxex resume hxex.toml" in capsys.readouterr().err
    assert cli.main(["resume", str(path)]) == 1  # the project has no [run.bootstrap]
    assert "no [run.bootstrap]" in capsys.readouterr().err
    path.write_text(with_boot)
    assert cli.main(["-q", "resume", str(path)]) == 0
    with open_bundle(tmp_path / "out") as b:
        assert b.bootstrap() is not None and b.manifest["configs"]["bootstrap"]["n_boot"] == 3
        assert list(b.fit().spectrum_id) == ids and list(b.fit().selected_K) == ks
    assert (tmp_path / "out_tables" / "spectra.csv").is_file()
    path.write_text(with_boot.replace("n_boot = 3", "n_boot = 4"))
    assert cli.main(["-q", "resume", str(path)]) == 0
    with open_bundle(tmp_path / "out") as b:
        assert b.manifest["configs"]["bootstrap"]["n_boot"] == 4
        assert b.manifest["configs"]["run"]["bootstrap"]["n_boot"] == 4
    # the same settings again, a list among them (ks): kept, not redone (a list came back from
    # the manifest's JSON and never equalled the project's tuple, so every resume redid it)
    path.write_text(with_boot.replace("n_boot = 3", "n_boot = 4\nks = [1]"))
    assert cli.main(["-q", "resume", str(path)]) == 0
    capsys.readouterr()
    assert cli.main(["resume", str(path)]) == 0
    assert "nothing to add" in capsys.readouterr().out
    # the bundle's settings file follows: the new bootstrap, the project file's input kept
    used = load_project(tmp_path / "out" / "run_settings_used.toml")
    assert used.run.bootstrap.n_boot == 4 and used.input == load_project(path).input


def test_each_reference_gets_an_analysis_folder_of_its_own(tmp_path, capsys):
    """The default analysis folder is named after the reference (results_analysis_WT), so an
    analysis against another reference never replaces it; hxex info says whether each was made
    from the results there now, and hxex analyse warns about the others that were not;
    --keep-previous moves every analysis folder with its bundle; [[analysis]] (several tables)
    is refused with the way to do it."""
    import json

    from hxex.io.bundle import analysis_folder_name

    assert analysis_folder_name("results") == "results_analysis"
    assert analysis_folder_name("results", "WT") == "results_analysis_WT"
    assert analysis_folder_name("results", "A:WT") == "results_analysis_A_WT"
    assert analysis_folder_name("runs", "pS 59/x") == "runs_analysis_pS_59_x"
    path = mzml_project(tmp_path)
    path.write_text(path.read_text() + '\n[analysis]\nks = [1]\noverlay = ["avg"]\n')
    assert cli.main(["-q", "run", str(path)]) == 0
    assert cli.main(["-q", "analyse", str(path)]) == 0
    assert cli.main(["-q", "analyse", str(path), "--reference", "WT"]) == 0
    plain, wt = tmp_path / "out_analysis", tmp_path / "out_analysis_WT"
    assert (plain / "species_uptake.csv").is_file() and (wt / "species_uptake.csv").is_file()
    capsys.readouterr()

    def info():
        assert cli.main(["info", str(tmp_path / "out")]) == 0
        return capsys.readouterr().out

    out = info()
    assert "analyses       out_analysis: up to date" in out and "out_analysis_WT: up to date" in out
    with open_bundle(tmp_path / "out") as b:
        first = b.manifest["created"]
    assert cli.main(["-q", "run", str(path), "--overwrite"]) == 0
    out = info()
    assert f"out_analysis: made from earlier results (created {first})" in out
    assert cli.main(["-q", "analyse", str(path), "--overwrite"]) == 0  # WT is still the old one
    assert f"out_analysis_WT was made from earlier results (created {first})" in \
        capsys.readouterr().err
    out = info()
    assert "out_analysis: up to date" in out and "out_analysis_WT: made from earlier" in out
    listing = json.loads((wt / "hxex_listing.json").read_text())
    del listing["made_from"]  # as a folder written before hxex recorded it
    (wt / "hxex_listing.json").write_text(json.dumps(listing))
    assert "out_analysis_WT: not known" in info()
    assert cli.main(["-q", "run", str(path), "--keep-previous"]) == 0
    assert (tmp_path / "out.1_analysis").is_dir() and (tmp_path / "out.1_analysis_WT").is_dir()
    assert not plain.exists() and not wt.exists()
    with pytest.raises(ProjectError, match=r"one \[analysis\] table.*--reference"):
        project_from(tomllib.loads('project = "p"\n[input]\nformat = "specexport"\n'
                                   'data_dir = "d"\n[[analysis]]\nreference = "WT"\n'
                                   '[[analysis]]\nreference = "Mut"\n'), tmp_path)


def test_the_settings_used_read_back_as_the_project(tmp_path):
    """run_settings_used.toml and analysis_settings_used.toml write out every setting
    (defaults included, a stage that is off as "name = false", a None whose default is not
    None as false) and read back as the same project; with record_paths = "name" no directory
    is recorded; an analysis of several runs gets one run settings file per run."""
    import types

    from hxex.analysis import AnalysisConfig
    from hxex.config import (
        analysis_settings_text,
        project_record,
        run_settings_text,
        settings_records,
    )
    from hxex.report import Settings

    def written(project, name="used.toml"):
        manifest = {"project": project.project, "created": "2026-10-20T00:00:00+00:00",
                    "versions": {"hxex": "0"},
                    "configs": {"run": project.run.to_dict(),
                                "project_file": project_record(project)}}
        text = run_settings_text(manifest)
        (tmp_path / name).write_text(text)
        return text, load_project(tmp_path / name), manifest

    for fmt in FORMATS:
        (tmp_path / "t.toml").write_text(template(fmt))
        project = load_project(tmp_path / "t.toml")
        text, back, _ = written(project)
        assert (back.run, back.input, back.output) == (project.run, project.input,
                                                         project.output), fmt
        assert "bootstrap = false" in text and "uncomment" not in text
    (tmp_path / "rich.toml").write_text("""
project = "demo"
threads = 4
[input]
format = "mzml"
peptides = "lists/peptides.csv"
data_root = "data"
record_paths = "relative"
[input.target]
rt_window = [60.0, 180.0]
[input.scans]
min_time = "1 min"
[[input.fragments]]
ions = "c-1:ALL z-dot:ALL"
label = "gentle"
parent_mz = false
[[input.exclude]]
sample = "WT"
peptide_range = "45-60"
time = "30 s"
[[input.runs]]
path = "sub/UN_01.mzML"
sample = "WT"
role = "undeut"
[[input.runs]]
path = "D_60s_01.mzML"
sample = "WT"
time = "60 s"
precursor_charge = false
[output]
bundle = "results"
[run]
drop_no_signal = false
[run.fit.noise]
sd = 0.5
percent = false
[run.bootstrap]
n_boot = 30
[run.controls]
fulldeut_at = "20 h"
[analysis]
reference = "WT"
protection = true
fragments = false
filters = false
times = ["1 min", 600]
[analysis.labelling]
ph_read = 7.0
temperature = 20.0
d2o_fraction = 0.9
[[analysis.regions]]
name = "N"
start = 1
end = 20
[[review.exclude]]
sample = "WT"
rep = 2
[analysis.controls]
d2o_fraction = 0.85
""")
    project = load_project(tmp_path / "rich.toml")
    text, back, manifest = written(project)
    assert back.run == project.run and back.output == project.output
    assert back.input.runs == project.input.runs and back.input.exclude == project.input.exclude
    assert back.input.fragments == project.input.fragments
    assert back.input.scans == project.input.scans and back.input.target == project.input.target
    # "relative" records the run paths from data_root and only the names of the other paths
    assert 'path = "sub/UN_01.mzML"' in text and 'peptides = "peptides.csv"' in text
    assert str(tmp_path) not in text
    analysis = analysis_settings_text(project.analysis, {"": manifest}, review=project.review)
    (tmp_path / "analysis.toml").write_text(analysis)
    back = load_project(tmp_path / "analysis.toml", analysis_only=True)
    assert back.analysis == project.analysis and back.review == project.review
    (tmp_path / "analysis.toml").write_text(analysis_settings_text(AnalysisConfig(), {}))
    assert load_project(tmp_path / "analysis.toml", analysis_only=True).analysis == \
        AnalysisConfig()
    # an analysis of several runs: one run settings file per run; a run without settings
    # recorded (results from Python) gets none
    runs = {"A": types.SimpleNamespace(settings=Settings.from_manifest(manifest)),
            "B b": types.SimpleNamespace(settings=None)}
    files = settings_records(runs, project.analysis)
    assert set(files) == {"analysis_settings_used.toml", "run_settings_used_A.toml"}
    assert "run A (created 2026-10-20T00:00:00+00:00 by hxex 0)" in \
        files["analysis_settings_used.toml"].replace("\n# ", " ")


def test_check_counts_what_each_exclusion_rule_matches(tmp_path, capsys):
    """hxex check counts each rule's spectra among those the read would make (from the file
    names; nothing is read), and a rule that matches nothing fails the check, saying which of
    its values no spectrum has."""
    data = tmp_path / "d"
    pep = data / "WT" / f"0001-0010-{PEPTIDE}"
    pep.mkdir(parents=True)
    (data / "WT.fasta").write_text(f">WT\n{PEPTIDE}\n")
    for name in ("Non-D-1-z2.csv", "Non-D-2-z2.csv", "Full-D-1-z2.csv", "60s-1-z2.csv",
                 "60s-2-z2.csv", "20h-1-z2.csv"):
        (pep / name).write_text("")
    head = 'project = "p"\n[input]\nformat = "specexport"\ndata_dir = "d"\n'
    rules = ('[[input.exclude]]\nsample = "WT"\npeptide_range = "1-10"\ntime = "60 s"\nrep = 2\n'
             '[[review.exclude]]\nrep = 2\n')
    path = tmp_path / "hxex.toml"
    path.write_text(head + rules + '[output]\nbundle = "out"\n')
    assert cli.main(["check", str(path)]) == 0
    out = capsys.readouterr().out
    assert ("[[input.exclude]] #1 (sample WT, peptide range 1-10, time 1 min, rep 2): 1 of 6 "
            "spectra") in out
    # the review rule counts among the spectra the input rules leave: Non-D-2 only
    assert "[[review.exclude]] #1 (rep 2): 1 of 6 spectra" in out
    # a mistyped value: a problem naming the values the spectra have
    path.write_text(head + '[[input.exclude]]\nsample = "wt"\ncharge = 2\n'
                    '[output]\nbundle = "out"\n')
    assert cli.main(["check", str(path)]) == 1
    err = capsys.readouterr().err
    assert "matches no spectrum: no spectrum has sample 'wt' (samples: WT)" in err
    # each key matches something, the combination nothing
    path.write_text(head + '[[input.exclude]]\nrole = "fulldeut"\nrep = 2\n'
                    '[output]\nbundle = "out"\n')
    assert cli.main(["check", str(path)]) == 1
    assert "no spectrum matches them all" in capsys.readouterr().err
    # a time point promoted to a control matches a role rule, as in the run
    path.write_text(head + '[[input.exclude]]\nrole = "fulldeut"\nrep = 1\n'
                    '[output]\nbundle = "out"\n[run.controls]\nfulldeut_at = "20 h"\n')
    assert cli.main(["check", str(path)]) == 0
    assert "(role fulldeut, rep 1): 2 of 6 spectra" in capsys.readouterr().out


def test_check_counts_exclusions_of_scan_files(tmp_path, capsys):
    path = mzml_project(tmp_path, extra_input='[[input.exclude]]\nsample = "WT"\ntime = 30\n'
                                              '[[input.exclude]]\nsample = "KO"\n')
    assert cli.main(["check", str(path)]) == 1
    out, err = capsys.readouterr()
    assert "[[input.exclude]] #1 (sample WT, time 30 s): 1 of 6 spectra" in out
    assert "#2 (sample KO) matches no spectrum: no spectrum has sample 'KO' (samples: WT)" in err


def test_every_output_folder_follows_the_bundles_rule(tmp_path, capsys):
    """report and analyse refuse a folder that holds earlier output unless told to overwrite it
    or keep it (moved to NAME.N); [output] overwrite or keep_previous says so for every
    command; a run whose bundle was deleted by hand refuses its tables folder, and warns about
    the figures and analysis folders made from the results that are gone (owner, 2026-10-21)."""
    path = mzml_project(tmp_path)
    path.write_text(path.read_text() + '\n[analysis]\nks = [1]\noverlay = ["avg"]\n')
    assert cli.main(["-q", "run", str(path)]) == 0
    assert cli.main(["-q", "report", str(path), "--figures"]) == 0
    assert cli.main(["-q", "analyse", str(path)]) == 0
    figures = tmp_path / "out_figures"
    capsys.readouterr()
    # --keep-previous: the folder moves to NAME.N, numbers never reused
    assert cli.main(["-q", "report", str(path), "--figures", "--keep-previous"]) == 0
    assert (tmp_path / "out_figures.1" / "overview.pdf").is_file()
    assert (figures / "overview.pdf").is_file()
    assert cli.main(["-q", "analyse", str(path), "--keep-previous"]) == 0
    assert (tmp_path / "out_analysis.1" / "species_uptake.csv").is_file()
    assert (tmp_path / "out_analysis" / "species_uptake.csv").is_file()
    assert cli.main(["-q", "report", str(path), "--figures", "--keep-previous"]) == 0
    assert (tmp_path / "out_figures.2").is_dir()
    # the project's [output] overwrite covers every folder
    text = path.read_text()
    path.write_text(text.replace("[output]", "[output]\noverwrite = true", 1))
    assert cli.main(["-q", "report", str(path), "--figures"]) == 0
    assert cli.main(["-q", "analyse", str(path)]) == 0
    path.write_text(text)
    # the bundle deleted by hand: its tables folder is refused before anything is fitted
    shutil.rmtree(tmp_path / "out")
    assert cli.main(["-q", "run", str(path)]) == 1
    assert "out_tables holds the tables of earlier results" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
    assert cli.main(["-q", "run", str(path), "--keep-previous"]) == 0
    err = capsys.readouterr().err
    assert (tmp_path / "out_tables.1" / "series.csv").is_file()
    assert "out_figures holds figures of other results" in err
    assert "out_analysis holds an analysis of other results" in err
    assert cli.main(["info", str(tmp_path / "out")]) == 0
    info = capsys.readouterr().out
    assert "out_tables: up to date" in info and "out_figures: made from earlier results" in info
