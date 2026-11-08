# SPDX-License-Identifier: GPL-3.0-or-later
"""Retention windows edited for one ion in one file: the ``[input] rt_elution_windows`` table
(hxex.io.elution_windows) and the reader that takes it (PLAN Next steps 26). Synthetic LC runs
of one sample, the second eluting 6 s later."""

import logging

import numpy as np
import pytest

from hxex.config import load_project, read_input
from hxex.io import ReadError, read_mzml_source
from hxex.io.elution_windows import ElutionWindow, read_elution_windows
from test_mzml import write_mzml
from test_retention_search import PEPTIDES, lc_run

PEPTIDE = "VDQIIMAK"
N = len(PEPTIDE)
RT = PEPTIDES[PEPTIDE] / 60.0  # minutes
ION = f"0001-{N:04d}-{PEPTIDE}-z2"  # as the spectrum ids name it


def _project(tmp_path, windows: str | None = None, scans: str = ""):
    tmp_path.mkdir(parents=True, exist_ok=True)
    write_mzml(tmp_path / "un1.mzML", lc_run(1))
    write_mzml(tmp_path / "un2.mzML", lc_run(2, shift=6.0))
    (tmp_path / "peptides.csv").write_text(
        "peptide,peptide_range,start_seq,end_seq,charge,rt,rt_start,rt_end\n"
        f"{PEPTIDE},0001-{N:04d},1,{N},2,{RT:.4f},{RT - 6 / 60:.4f},{RT + 6 / 60:.4f}\n")
    line = ""
    if windows is not None:
        (tmp_path / "rt_elution_windows.csv").write_text(windows)
        line = 'rt_elution_windows = "rt_elution_windows.csv"\n'
    path = tmp_path / "hxex.toml"
    path.write_text(f"""project = "p"

[input]
format = "mzml"
peptides = "peptides.csv"
{line}
[[input.runs]]
path = "un1.mzML"
sample = "S"
role = "undeut"

[[input.runs]]
path = "un2.mzML"
sample = "S"
role = "undeut"
{scans}""")
    return path


def _by_file(spectra):
    return {s.split("/")[1]: i for i, s in enumerate(spectra.spectrum_id)}


def test_the_table_is_read_and_checked(tmp_path):
    path = tmp_path / "w.csv"
    path.write_text("Sample, File ,ION,rt_start,rt_end,drift_start,drift_end\n"
                    "S,a.mzML,1-8-X-z2,1.5,2.0,,\n"
                    "S,a.mzML,2-9-Y-z1,1.0,1.25,3.1,3.3\n")
    rows = read_elution_windows(path)
    assert rows == [ElutionWindow("S", "a.mzML", "1-8-X-z2", 1.5, 2.0),
                    ElutionWindow("S", "a.mzML", "2-9-Y-z1", 1.0, 1.25, 3.1, 3.3)]
    assert rows[1].drift == (3.1, 3.3) and rows[0].drift is None
    assert ElutionWindow.from_dict(rows[1].to_dict()) == rows[1]
    for text, words in (
            ("sample,file,rt_start,rt_end\nS,a,1,2\n", "no ion column"),
            ("sample,file,ion,rt_start,rt_end\nS,a,x,2,1\n", "line 2: rt_end 1 must be after"),
            ("sample,file,ion,rt_start,rt_end,drift_start\nS,a,x,1,2,3\n", "both, or neither"),
            ("sample,file,ion,rt_start,rt_end\nS,a,x,1,2\nS,a,x,1,3\n", "line 3: x in a"),
            ("sample,file,ion,rt_start,rt_end\nS,a,x,early,2\n", "not a number of minutes")):
        path.write_text(text)
        with pytest.raises(ReadError, match=words):
            read_elution_windows(path)


def test_a_row_gives_one_ion_its_own_window_in_one_file(tmp_path):
    plain = read_input(load_project(_project(tmp_path / "a")))
    i = _by_file(plain)["un2.mzML"]
    start, end = plain.rt_start[i] + 0.02, plain.rt_end[i] - 0.02
    windows = ("sample,file,ion,rt_start,rt_end\n"
               f"S,un2.mzML,{ION},{start:.4f},{end:.4f}\n")
    edited = read_input(load_project(_project(tmp_path / "b", windows)))
    j = _by_file(edited)
    # the edited file sums the scans inside its own window; the other file is as it was
    assert start <= edited.rt_start[j["un2.mzML"]] < edited.rt_end[j["un2.mzML"]] <= end
    assert edited.spectrum_id[j["un2.mzML"]] != plain.spectrum_id[i]
    k = _by_file(plain)["un1.mzML"]
    assert edited.spectrum_id[j["un1.mzML"]] == plain.spectrum_id[k]
    np.testing.assert_array_equal(edited.spectrum(j["un1.mzML"])[1], plain.spectrum(k)[1])
    assert len(set(edited.data_id.tolist())) == 1  # still one series
    # recorded, so that a repeat read gives the same spectra
    assert edited.source["elution_windows"][0]["file"] == "un2.mzML"
    again = read_mzml_source(edited.source, data_root=tmp_path / "b")
    assert list(again.spectrum_id) == list(edited.spectrum_id)


def test_an_edited_ion_is_not_searched_and_a_stray_row_is_warned_about(tmp_path, caplog):
    scans = "\n[input.scans]\nfind_peak = true\n"
    found = read_input(load_project(_project(tmp_path / "a", scans=scans)))
    i = _by_file(found)["un2.mzML"]
    start, end = found.rt_start[i] - 0.1, found.rt_end[i] + 0.1  # wider than the peak found
    windows = ("sample,file,ion,rt_start,rt_end\n"
               f"S,un2.mzML,{ION},{start:.4f},{end:.4f}\n"
               f"S,other.mzML,{ION},1.0,2.0\n")
    with caplog.at_level(logging.WARNING):
        edited = read_input(load_project(_project(tmp_path / "b", windows, scans)))
    j = _by_file(edited)["un2.mzML"]
    assert edited.rt_start[j] < found.rt_start[i] and edited.rt_end[j] > found.rt_end[i]
    assert "1 row(s) name no ion read from their file" in caplog.text
    assert "other.mzML" in caplog.text


def test_check_names_rows_the_project_does_not_read(tmp_path, capsys):
    from hxex.cli import main

    good = f"sample,file,ion,rt_start,rt_end\nS,un2.mzML,{ION},0.55,0.75\n"
    path = _project(tmp_path / "a", good)
    assert main(["check", str(path)]) == 0
    assert "1 edited in 1 file(s)" in capsys.readouterr().out
    stray = good + (f"S,gone.mzML,{ION},0.5,0.7\n"
                    "S,un1.mzML,0001-0008-VDQIIMAK-z3,0.5,0.7\n")
    path = _project(tmp_path / "b", stray)
    assert main(["check", str(path)]) != 0
    said = capsys.readouterr().err
    assert "gone.mzML of sample S is not a run of the project" in said
    assert "0001-0008-VDQIIMAK-z3 is not read from un1.mzML" in said
