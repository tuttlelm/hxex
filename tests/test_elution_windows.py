# SPDX-License-Identifier: GPL-3.0-or-later
"""Retention windows edited by hand, for a whole peptide or one ion in one file: the ``[input]
rt_elution_windows`` table (hxex.io.elution_windows) and the reader that takes it (PLAN Next
steps 26). Synthetic LC runs of one sample, the second eluting 6 s later."""

import logging

import numpy as np
import pytest

from hxex.config import load_project, read_input
from hxex.io import ReadError, read_mzml_source
from hxex.io.elution_windows import (
    ElutionWindow,
    by_key,
    by_peptide,
    peptide_label,
    read_elution_windows,
    window_for,
)
from hxex.picking import plan_bins
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
            ("sample,file,rt_start,rt_end\nS,a,1,2\n", "no peptide or ion column"),
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
    assert "1 row(s) name no peptide or ion read from their file" in caplog.text
    assert "other.mzML" in caplog.text


def test_check_names_rows_the_project_does_not_read(tmp_path, capsys):
    from hxex.cli import main

    good = f"sample,file,ion,rt_start,rt_end\nS,un2.mzML,{ION},0.55,0.75\n"
    path = _project(tmp_path / "a", good)
    assert main(["check", str(path)]) == 0
    assert "0 for a whole peptide, 1 for one ion, in 1 file(s)" in capsys.readouterr().out
    stray = good + (f"S,gone.mzML,{ION},0.5,0.7\n"
                    "S,un1.mzML,0001-0008-VDQIIMAK-z3,0.5,0.7\n")
    path = _project(tmp_path / "b", stray)
    assert main(["check", str(path)]) != 0
    said = capsys.readouterr().err
    assert "gone.mzML of sample S is not a run of the project" in said
    assert "0001-0008-VDQIIMAK-z3 is not read from un1.mzML" in said


def test_peptide_rows_are_read_and_checked(tmp_path):
    """A row names a peptide (every ion of it in that file), one ion, or that ion of that
    peptide; a peptide row needs a window and gives no drift range (drift times differ between
    charge states); an ion row may give only a drift range."""
    path = tmp_path / "w.csv"
    path.write_text("sample,file,peptide,ion,rt_start,rt_end,drift_start,drift_end\n"
                    "S,a.mzML,1-8-X,,1.5,2.0,,\n"
                    "S,a.mzML,,1-8-X-z2,,,3.1,3.3\n"
                    "S,a.mzML,,1-8-X-z3,1.6,1.9,,\n"
                    "S,a.mzML,1-8-X,2-4-LK-z1-ion_type:c,1.7,1.8,,\n")
    rows = read_elution_windows(path)
    assert rows[0] == ElutionWindow("S", "a.mzML", None, 1.5, 2.0, peptide="1-8-X")
    assert rows[0].is_peptide and rows[0].key == ("S", "a.mzML", "1-8-X", "")
    assert not rows[3].is_peptide and rows[3].name == "2-4-LK-z1-ion_type:c of peptide 1-8-X"
    assert rows[1].window is None and rows[1].drift == (3.1, 3.3)
    assert ElutionWindow.from_dict(rows[0].to_dict()) == rows[0]
    ions, peptides = by_key(rows), by_peptide(rows)
    # z2 takes the peptide's window and its own drift range; z3 its own window; z4 the
    # peptide's; another peptide nothing
    assert window_for("1-8-X-z2", "1-8-X", "S", "a.mzML", ions, peptides)[:2] == (
        (1.5, 2.0), (3.1, 3.3))
    assert window_for("1-8-X-z3", "1-8-X", "S", "a.mzML", ions, peptides)[:2] == (
        (1.6, 1.9), None)
    assert window_for("1-8-X-z4", "1-8-X", "S", "a.mzML", ions, peptides)[:2] == (
        (1.5, 2.0), None)
    assert window_for("2-9-Y-z2", "2-9-Y", "S", "a.mzML", ions, peptides) == (None, None, [])
    # an ion row naming its peptide: that peptide's ion of that name only (two parents sharing
    # a terminus give c ions of the same name)
    assert window_for("2-4-LK-z1-ion_type:c", "1-8-X", "S", "a.mzML", ions, peptides)[:2] == (
        (1.7, 1.8), None)
    assert window_for("2-4-LK-z1-ion_type:c", "1-9-XY", "S", "a.mzML", ions, peptides) == (
        None, None, [])
    head = "sample,file,peptide,ion,rt_start,rt_end,drift_start,drift_end\n"
    for text, words in (
            ("S,a,,,1,2,,\n", "give a peptide .* or an ion"),
            ("S,a,1-8-X,,1,2,3,4\n", "a peptide row gives no drift range"),
            ("S,a,1-8-X,,,,,\n", "a peptide row needs rt_start and rt_end"),
            ("S,a,,1-8-X-z2,,,,\n", "an ion row needs rt_start and rt_end, or drift"),
            ("S,a,,1-8-X-z2,1,,,\n", "rt_start and rt_end both, or neither"),
            ("S,a,1-8-X,,1,2,,\nS,a,1-8-X,,1,3,,\n", "line 3: peptide 1-8-X in a")):
        path.write_text(head + text)
        with pytest.raises(ReadError, match=words):
            read_elution_windows(path)


def test_a_fragment_belongs_to_its_parent_peptide():
    """A peptide row covers the peptide's own charge states and, as a parent, its fragments:
    the same molecules, which elute together."""
    from hxex.fragments import fragment_entries
    from hxex.spectra import PeptideEntry

    parent = PeptideEntry("", "S", "AGNLGGVVT", "2-10", 2, 10, 3)
    assert peptide_label(parent) == "2-10-AGNLGGVVT"
    assert peptide_label(PeptideEntry("", "S", "AGNLGGVVT", "2-10", 2, 10, 2,
                                      modification="ion_type:M")) == "2-10-AGNLGGVVT"
    assert peptide_label(PeptideEntry("", "S", "AGNLGGVVT", "2-10", 2, 10, 2,
                                      modification="Hex")) == "2-10-AGNLGGVVT-Hex"
    entries = fragment_entries(parent, "c z-dot M")
    assert entries and {peptide_label(e) for e in entries} == {"2-10-AGNLGGVVT"}


def _lc_charges(seed, shift=0.0):
    """MS1 scans every 0.25 s for 60 s: PEPTIDE at charges 2 and 3, eluting together."""
    rng = np.random.default_rng(seed)
    scans = []
    for t in np.arange(0.0, 60.0 + 1e-9, 0.25):
        h = 1e5 * np.exp(-0.5 * ((t - PEPTIDES[PEPTIDE] - shift) / 2.0) ** 2)
        mz, y = [np.array([400.0])], [np.array([0.0])]
        for z in (2, 3):
            plan = plan_bins(PEPTIDE, z)
            shape, offsets = plan.clusters
            for k in range(4):
                centre = plan.undeut_mz + offsets[k] / z
                x = centre + np.arange(-0.016, 0.0161, 0.002)
                mz.append(x)
                y.append(h * shape[k] / shape[0] * np.exp(-0.5 * ((x - centre) / 0.004) ** 2))
        mz, y = np.concatenate(mz), np.concatenate(y)
        order = np.argsort(mz)
        y = np.clip(y[order] * (1 + 0.01 * rng.standard_normal(len(y))), 0, None)
        scans.append({"mz": mz[order], "intensity": y, "time": t, "unit": "s",
                      "filter": "FTMS + p ESI Full ms"})
    return scans


def _charges_project(tmp_path, windows: str | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    write_mzml(tmp_path / "un1.mzML", _lc_charges(1))
    write_mzml(tmp_path / "un2.mzML", _lc_charges(2, shift=6.0))
    (tmp_path / "peptides.csv").write_text(
        "peptide,peptide_range,start_seq,end_seq,charge,rt,rt_start,rt_end\n"
        + "".join(f"{PEPTIDE},0001-{N:04d},1,{N},{z},{RT:.4f},{RT - 6 / 60:.4f},"
                  f"{RT + 6 / 60:.4f}\n" for z in (2, 3)))
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
""")
    return path


def _windows(spectra):
    """(file, charge) -> (rt_start, rt_end) of each spectrum."""
    return {(s.split("/")[1], int(z)): (a, b) for s, z, a, b in zip(
        spectra.spectrum_id, spectra.charge, spectra.rt_start, spectra.rt_end, strict=True)}


def test_a_peptide_row_gives_every_charge_state_its_window(tmp_path, capsys):
    """One peptide row: both charge states in that file sum the scans inside it; the other file
    is as it was. An ion row as well: that ion takes its own window, the other charge the
    peptide's."""
    from hxex.cli import main

    plain = _windows(read_input(load_project(_charges_project(tmp_path / "a"))))
    a, b = plain[("un2.mzML", 2)]
    assert plain[("un2.mzML", 3)] == (a, b)
    start, end = a + 0.02, b - 0.02
    peptide = f"0001-{N:04d}-{PEPTIDE}"
    table = ("sample,file,peptide,ion,rt_start,rt_end\n"
             f"S,un2.mzML,{peptide},,{start:.4f},{end:.4f}\n")
    path = _charges_project(tmp_path / "b", table)
    edited = _windows(read_input(load_project(path)))
    for z in (2, 3):
        lo, hi = edited[("un2.mzML", z)]
        assert start <= lo < hi <= end
        assert edited[("un1.mzML", z)] == plain[("un1.mzML", z)]
    assert main(["check", str(path)]) == 0
    assert "1 for a whole peptide, 0 for one ion, in 1 file(s)" in capsys.readouterr().out
    # the 3+ ion alone over a narrower window still: its own row wins over the peptide's
    narrow = (start + 0.02, end - 0.02)
    table += f"S,un2.mzML,,0001-{N:04d}-{PEPTIDE}-z3,{narrow[0]:.4f},{narrow[1]:.4f}\n"
    both = _windows(read_input(load_project(_charges_project(tmp_path / "c", table))))
    assert both[("un2.mzML", 2)] == edited[("un2.mzML", 2)]
    lo, hi = both[("un2.mzML", 3)]
    assert narrow[0] <= lo < hi <= narrow[1]
    # a peptide the file does not read: named by hxex check
    stray = table + "S,un2.mzML,0002-0009-OTHER,,0.5,0.7\n"
    assert main(["check", str(_charges_project(tmp_path / "d", stray))]) != 0
    assert "peptide 0002-0009-OTHER is not read from un2.mzML" in capsys.readouterr().err
