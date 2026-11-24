# SPDX-License-Identifier: GPL-3.0-or-later
"""A run that replaces a bundle takes the raw spectra stored there instead of reading the input
again, when the read would give them (hxex.stored_read; owner, 2026-11-16): the same read plan
(settings, peptide table, reader code) and data files with the identities the read recorded.
``--reread`` reads; any difference reads and is named in the log."""

import logging

import numpy as np
import pytest

from hxex import cli
from hxex.config import load_project, read_input
from hxex.io import open_bundle, save_spectra
from hxex.stored_read import PLAN_KEY, stored_spectra_for
from test_cli import PEPTIDE, mzml_project
from test_io import write_spec

TAKEN = "taking the raw spectra stored in"
READ = "reading the mzml input"


def _run(path, capsys, *extra) -> str:
    """The log of ``hxex run`` updating the results (the command line logs to standard error),
    with their bootstrap marked stopped partway so that there is something to do."""
    import json

    manifest = path.parent / "out" / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        data["pending"] = ["bootstrap"]
        manifest.write_text(json.dumps(data))
    capsys.readouterr()
    assert cli.main(["run", str(path), *extra]) == 0
    return capsys.readouterr().err


@pytest.fixture
def project(tmp_path, capsys):
    path = mzml_project(tmp_path)
    assert cli.main(["run", str(path)]) == 0
    capsys.readouterr()
    return path


def test_a_second_run_takes_the_stored_spectra_and_gives_the_same_results(project, capsys):
    out = project.parent / "out"
    with open_bundle(out) as b:
        before = b.binned()
        plan = b.spectra_source()[PLAN_KEY]
    assert plan["reader"] and plan["targets"] and plan["input"]["format"] == "mzml"
    log = _run(project, capsys)
    assert TAKEN in log and READ not in log
    with open_bundle(out) as b:
        after = b.binned()
        assert b.spectra_source()[PLAN_KEY] == plan  # kept for the next run
    assert list(after.spectrum_id) == list(before.spectrum_id)
    np.testing.assert_array_equal(after.intensity, before.intensity)
    assert cli.main(["info", str(project)]) == 0
    out_text = capsys.readouterr().out
    assert "read           not repeated" in out_text
    assert "changed        nothing" in out_text  # where the spectra came from is no setting


def test_reread_reads_the_input(project, capsys):
    log = _run(project, capsys, "--reread")
    assert READ in log and TAKEN not in log
    assert cli.main(["info", str(project)]) == 0
    assert "not repeated" not in capsys.readouterr().out


def test_settings_beside_the_read_keep_the_stored_spectra(project, capsys):
    """A change of the fit (a preset, k_max) needs no read, nor one of the picker's settings the
    read does not use (the bin value, which apex pick); the rest of the picker does for scan
    files (the readers cut the scans by its windows)."""
    project.write_text(project.read_text().replace("k_max = 1", "k_max = 2"))
    assert TAKEN in _run(project, capsys)
    project.write_text(project.read_text().replace(
        "zero_filling = 3", 'zero_filling = 3\nmeasure = "height"\narea_fwhm = 2.0'))
    assert TAKEN in _run(project, capsys)
    project.write_text(project.read_text().replace("zero_filling = 3", "zero_filling = 4"))
    log = _run(project, capsys)
    assert READ in log and "[run.picker] changed" in log


def test_keep_previous_takes_them_from_the_generation_it_made(project, capsys):
    assert cli.main(["run", str(project), "--keep-previous"]) == 0
    log = capsys.readouterr().err
    assert TAKEN in log
    assert "out.1 is used" in log  # the stored picking follows the moved bundle


@pytest.mark.parametrize("change, named", [
    ("data", "run1.mzML"),
    ("table", "the targets"),
    ("input", "[input] scans"),
    ("reader", "the reading code"),
])
def test_a_changed_input_is_read_again(project, capsys, monkeypatch, change, named):
    folder = project.parent
    if change == "data":
        with (folder / "run1.mzML").open("a") as fh:
            fh.write("\n")
    elif change == "table":
        table = folder / "peptides.csv"
        table.write_text(table.read_text().replace("\nx,", "\ny,"))  # the file column
    elif change == "input":
        project.write_text(project.read_text().replace(
            'peptides = "peptides.csv"', 'peptides = "peptides.csv"\n[input.scans]\n'
                                         'mz_margin = 2.5'))
    else:
        import hxex._engine as engine

        monkeypatch.setattr(engine, "reader_fingerprint", lambda: "another")
    log = _run(project, capsys)
    assert READ in log and TAKEN not in log
    assert named in log


def test_a_bundle_without_a_read_plan_or_spectra_is_read(tmp_path, capsys, monkeypatch):
    """Bundles written before the plan was recorded (or with keep_spectra = false) read."""
    import hxex.stored_read as stored_read

    path = mzml_project(tmp_path)
    with monkeypatch.context() as m:
        m.setattr(stored_read, "read_plan", lambda project: None)
        assert cli.main(["run", str(path)]) == 0
    log = _run(path, capsys)
    assert "read before hxex recorded what a read depends on" in log and READ in log
    path.write_text(path.read_text().replace('figures = false', 'figures = false\n'
                                                                'keep_spectra = false'))
    _run(path, capsys)  # stores no spectra
    log = _run(path, capsys)
    assert "holds no raw spectra to take" in log and READ in log


def test_a_folder_format_compares_its_folders_and_fasta_files(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    data = tmp_path / "d"
    pep = data / "WT" / f"0001-0010-{PEPTIDE}"
    pep.mkdir(parents=True)
    fasta = data / "WT.fasta"
    fasta.write_text(f">WT\n{PEPTIDE}\n")
    write_spec(pep / "Non-D-1-z2.csv", [(300.2, 5), (300.3, 4)])
    write_spec(pep / "60s-1-z2.csv", [(300.5, 7), (300.6, 3)])
    path = tmp_path / "hxex.toml"
    path.write_text('project = "p"\n[input]\nformat = "specexport"\ndata_dir = "d"\n'
                    '[output]\nbundle = "out"\n')
    project = load_project(path)
    stored = tmp_path / "spectra.parquet"
    save_spectra(read_input(project), stored)
    taken = stored_spectra_for(project, stored)
    assert taken is not None and len(taken) == 2
    write_spec(pep / "60s-2-z2.csv", [(300.5, 6)])  # a new spectrum in a folder read
    assert stored_spectra_for(project, stored) is None
    assert f"0001-0010-{PEPTIDE}" in caplog.text
    (pep / "60s-2-z2.csv").unlink()
    assert stored_spectra_for(project, stored) is not None
    fasta.write_text(f">WT another name\n{PEPTIDE}\n")  # the same sequence
    assert stored_spectra_for(project, stored) is None
    assert "the FASTA files" in caplog.text


def test_the_reading_code_holds_what_the_readers_import_from_picking():
    """READER (the reading code's fingerprint) names the picking modules only: each name a
    reader imports from hxex.picking must be defined in one of them, or a change there would
    not read the input again."""
    import ast
    from pathlib import Path

    import hxex
    import hxex.picking
    from hxex._engine import READER

    root = Path(hxex.__file__).parent
    readers = [root / name for name in READER if name.startswith("io/")]
    for path in readers:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith(
                    "picking") or node.level != 2:
                continue
            for alias in node.names:
                if node.module == "picking":
                    module = getattr(hxex.picking, alias.name).__module__
                else:
                    module = f"hxex.{node.module}"
                if path.name == "_parallel.py" and module == "hxex.picking.profile":
                    continue  # the processes' thread setting, not what a read gives
                name = module.removeprefix("hxex.").replace(".", "/") + ".py"
                assert name in READER, f"{path.name} imports {alias.name} from {name}"


def test_a_changed_window_row_reads_its_file_again_only(tmp_path, capsys):
    """One row of the elution-window table changed: hxex run reads that file again and takes
    the other file's spectra from the bundle (PLAN Next steps 39 b); the spectra are those of
    a fresh run."""
    from hxex.io.elution_windows import peptide_label
    from hxex.spectra import PeptideEntry
    from test_cli import FAST_RUN
    from test_mzml import time_course, write_mzml

    for k in (1, 2):
        write_mzml(tmp_path / f"run{k}.mzML", time_course(n=6, step=0.5 + 0.01 * k))
    (tmp_path / "peptides.csv").write_text(
        "file,sample,peptide,peptide_range,start_seq,end_seq,charge,rt_start,rt_end\n"
        f"x,,{PEPTIDE},0001-0010,1,10,2,0,10\n")
    label = peptide_label(PeptideEntry("x", "WT", PEPTIDE, "0001-0010", 1, 10, 2))
    windows = tmp_path / "windows.csv"
    windows.write_text(f"sample,file,peptide,rt_start,rt_end\nWT,run1.mzML,{label},0.5,2.0\n")
    path = tmp_path / "hxex.toml"
    path.write_text(f"""
project = "cli"

[input]
format = "mzml"
peptides = "peptides.csv"
rt_elution_windows = "windows.csv"

[[input.runs]]
path = "run1.mzML"
sample = "WT"
time = "60 s"

[[input.runs]]
path = "run2.mzML"
sample = "WT"
time = "120 s"

[output]
bundle = "out"
figures = false
{FAST_RUN}
""")
    assert cli.main(["run", str(path)]) == 0
    windows.write_text(f"sample,file,peptide,rt_start,rt_end\nWT,run1.mzML,{label},1.0,2.0\n")
    assert cli.main(["check", str(path)]) == 0
    assert "reads 1 data file again (run1.mzML)" in capsys.readouterr().out
    log = _run(path, capsys)
    assert "reading again 1 of 2 files (run1.mzML)" in log and TAKEN not in log
    with open_bundle(tmp_path / "out") as b:
        part = b.spectra()
    assert cli.main(["run", str(path), "--fresh"]) == 0
    with open_bundle(tmp_path / "out") as b:
        fresh = b.spectra()
    assert list(part.spectrum_id) == list(fresh.spectrum_id)
    np.testing.assert_array_equal(part.intensity, fresh.intensity)
    np.testing.assert_array_equal(part.mz, fresh.mz)
    assert part.source["elution_windows"] == fresh.source["elution_windows"]
