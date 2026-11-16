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
    """The log of ``hxex run --overwrite`` (the command line logs to standard error)."""
    capsys.readouterr()
    assert cli.main(["run", str(path), "--overwrite", *extra]) == 0
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
    """A change of the fit (a preset, k_max) needs no read; the picker does for scan files (the
    readers cut the scans by its windows)."""
    project.write_text(project.read_text().replace("k_max = 1", "k_max = 2"))
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
