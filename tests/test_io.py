# SPDX-License-Identifier: GPL-3.0-or-later
"""Reader tests: synthetic files for the logic, and the real test sets when they are available.

The real-data tests read their location from HXEX_BIMODAL_DATA (the pyHXExpress
Bimodal_HDX_Data folder) and HXEX_SPECEXPORT_EXAMPLE (a SpecExport folder with its FASTA file,
plus ``hdxms_analysis_*/alldeutdata_*.csv`` and ``metadf_asrun_*.csv`` from a pyHXExpress run).
HXEX_POOLED_EXAMPLE is a pooled folder with a FASTA and a metadf_*.csv. They are
skipped when the variables are not set.
"""

import csv
import io
import logging
import os
import zipfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from openpyxl import Workbook

from hxex import io as hio
from hxex.hooks import Cancelled
from hxex.spectra import PeptideEntry, SpectrumSet

ENTRY = PeptideEntry("f.xlsx", "S", "PEPTIDE", "0001-0007", 1, 7, 2)


# --- time labels ----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label, seconds, role",
    [
        ("undeut", np.nan, "undeut"), ("Undeut_1", np.nan, "undeut"), ("TD", np.nan, "fulldeut"),
        ("td", np.nan, "fulldeut"), ("60 s", 60.0, ""),
        ("5 min", 300.0, ""), ("2 h", 7200.0, ""), ("1.2 min", 60.0, ""), ("10.3 s", 10.0, ""),
        ("0.5 min", 30.0, ""), ("15 sec", 15.0, ""), ("  3 min ", 180.0, ""),
    ],
)
def test_hxexpress_label(label, seconds, role):
    time, got = hio.parse_hxexpress_label(label)
    assert got == role and (time == seconds or (np.isnan(time) and np.isnan(seconds)))


def test_decimal_times_are_exact():
    for token, expected in [("4.1m", 246.0), ("0.1m", 6.0), ("1.1h", 3960.0), ("0.3s", 0.3)]:
        assert hio.parse_specexport_token(token)[0] == expected
    assert hio.parse_hxexpress_label("0.7 min")[0] == 42.0
    assert hio.parse_hxexpress_label(".5 min")[0] == 30.0  # no leading zero
    assert hio.parse_specexport_token("4.1m") == hio.parse_specexport_token("246s")
    assert hio.parse_time("4.1 min") == 246.0 == hio.parse_time("4.1m")


@pytest.mark.parametrize(
    "label",
    ["60", "5 x", "nan min", "inf s", "abc min", "", 5, None, float("nan")],
)
def test_hxexpress_label_errors(label):
    with pytest.raises(hio.ReadError):
        hio.parse_hxexpress_label(label)


@pytest.mark.parametrize(
    "value, seconds",
    [(60, 60.0), (0.5, 0.5), ("60", 60.0), ("60 s", 60.0), ("5 min", 300.0), ("5min", 300.0),
     ("20 h", 72000.0), ("250 ms", 0.25), ("1.5 min", 90.0), ("1h21m", 4860.0),
     ("10m30s", 630.0), ("2 hours", 7200.0), ("15 SEC", 15.0), (" 3 min ", 180.0)],
)
def test_parse_time(value, seconds):
    assert hio.parse_time(value) == seconds


def test_parse_time_takes_numpy_numbers():
    assert hio.parse_time(np.int64(3)) == 3.0 and hio.parse_time(np.float32(0.5)) == 0.5
    with pytest.raises(hio.ReadError):
        hio.parse_time(np.bool_(True))


@pytest.mark.parametrize(
    "value", ["", "5 x", "nan", "inf s", "abc min", "undeut", "5 min 3", True, None, [60],
              "-1", -5, "-5 min"],
)
def test_parse_time_errors(value):
    with pytest.raises(hio.ReadError, match="setting"):
        hio.parse_time(value, "setting")


def test_negative_times_only_when_asked():
    """A time cannot be negative (a run at -5 s is a mistake), unless the caller allows it."""
    for value in ("-5", -5, np.float64(-0.5)):
        with pytest.raises(hio.ReadError, match="time: a time cannot be negative"):
            hio.parse_time(value, "time")
    assert hio.parse_time("-5", "time", allow_negative=True) == -5.0
    assert hio.parse_time(-5, "time", allow_negative=True) == -5.0


@pytest.mark.parametrize(
    "seconds, text",
    [(4.0, "4 s"), (60.0, "1 min"), (90.0, "1.5 min"), (300.0, "5 min"), (72000.0, "20 h"),
     (5400.0, "1.5 h"), (0.25, "250 ms"), (0.0, "0 s"), (10.3, "10.3 s"), (61.0, "61 s"),
     (float("nan"), ""), (-60.0, "-1 min")],
)
def test_format_time(seconds, text):
    assert hio.format_time(seconds) == text


@pytest.mark.parametrize(
    "token, seconds, role",
    [("Non", np.nan, "undeut"), ("Full", np.nan, "fulldeut"), ("60s", 60.0, ""),
     ("5m", 300.0, ""), ("2h", 7200.0, ""), ("1.5m", 90.0, "")],
)
def test_specexport_token(token, seconds, role):
    time, got = hio.parse_specexport_token(token)
    assert got == role and (time == seconds or (np.isnan(time) and np.isnan(seconds)))


@pytest.mark.parametrize("token", ["60x", "abc", "s", "", "nans", "infs"])
def test_specexport_token_errors(token):
    with pytest.raises(hio.ReadError):
        hio.parse_specexport_token(token)


# --- file names -----------------------------------------------------------------------------

def test_hxexpress_filename():
    e = hio.parse_hxexpress_filename("HSPB1_B1B5_0001-0011-MTERRVPFSLL-z2-allspectra.xlsx")
    assert (e.sample, e.peptide_range, e.start_seq, e.end_seq, e.peptide, e.charge) == (
        "HSPB1_B1B5", "0001-0011", 1, 11, "MTERRVPFSLL", 2,
    )
    assert e.file == "HSPB1_B1B5_0001-0011-MTERRVPFSLL-z2-allspectra.xlsx"


@pytest.mark.parametrize(
    "name",
    ["Angio_2_HI.xlsx", "S_1-8-PEP-2-text.xlsx", "S1-8-PEP-z2-text.xlsx", "S_a-8-PEP-z2-text.xlsx"],
)
def test_hxexpress_filename_errors(name):
    with pytest.raises(hio.ReadError):
        hio.parse_hxexpress_filename(name)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("60s-4-z3.csv", ("60s", 4, 3)),
        ("Non-D-1-z3.csv", ("Non", 1, 3)),
        ("Full-D-2-z12.csv", ("Full", 2, 12)),
        ("data_fits_B4_0053-0071_z3_binom_isotope.tsv", None),
        ("60s-4-z3.pdf", None),
        ("notes.csv", None),
    ],
)
def test_specexport_filename(name, expected):
    assert hio.parse_specexport_filename(name) == expected


# --- HX-Express files -----------------------------------------------------------------------

HEADER = ["undeut", None, "1 min", None, "1.2 min", None, "TD", None]
BODY = [
    [500.2, 10.0, 500.2, 5.0, 500.2, 6.0, 501.0, 1.0],
    [500.1, 20.0, 500.1, 15.0, 500.3, None, 500.0, 2.0],  # unsorted; a missing intensity
    [None, None, 500.3, 25.0, None, 7.0, None, None],  # missing m/z in column 3
    [500.3, 30.0, None, None, 500.4, 8.0, 499.0, 3.0],
]


def write_xlsx(path, header=HEADER, body=BODY):
    wb = Workbook()
    ws = wb.active
    ws.append(header)
    for row in body:
        ws.append(row)
    wb.save(path)


def write_hx_csv(path, header=HEADER, body=BODY):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        for row in [header, *body]:
            w.writerow(["" if c is None else c for c in row])


def test_doubtful_hxexpress_labels_are_warned_about(tmp_path, caplog):
    """A replicate-marked label whose time has no plain label (1.5 min meant as 90 s?) and one
    time written two ways (60 s and 1 min) are warned about, naming the labels and how they are
    read; the usual markers (1 min, 1.2 min), fractions below 1, and units of one size written
    differently (1 m, 1 min) are not."""
    from hxex.io.times import hxexpress_label_doubts

    assert hxexpress_label_doubts(["undeut", "1 min", "1.2 min", "1 m", "0.5 min", "TD"]) == []
    lone, = hxexpress_label_doubts(["undeut", "1.5 min", "5 min"])
    assert lone.startswith("'1.5 min' read as '1 min'") and "90 s for 1.5 min" in lone
    two, = hxexpress_label_doubts(["60 s", "1 min", "1.2 min"])
    assert two == "'1 min', '1.2 min', '60 s' are one time point, 1 min, read as its replicates"
    write_hx_csv(tmp_path / "a.csv", header=["undeut", None, "1.5 min", None, "60 s", None,
                                             "1 min", None])
    with caplog.at_level(logging.WARNING, logger="hxex.io.hxexpress"):
        hio.read_hxexpress_file(tmp_path / "a.csv", ENTRY)
    assert "a.csv: labels '1.5 min' read as '1 min'" not in caplog.text  # 1 min is there
    assert "a.csv: labels '1 min', '1.5 min', '60 s' are one time point" in caplog.text


@pytest.mark.parametrize("writer, name", [(write_xlsx, "a.xlsx"), (write_hx_csv, "a.csv")])
def test_read_hxexpress_file(tmp_path, writer, name):
    writer(tmp_path / name)
    sp = hio.read_hxexpress_file(tmp_path / name, ENTRY, data_id=7)
    assert [(s.label, s.rep, s.role) for s in sp] == [
        ("undeut", 1, "undeut"), ("1 min", 1, ""), ("1.2 min", 2, ""), ("TD", 1, "fulldeut"),
    ]
    assert np.isnan(sp[0].time) and np.isnan(sp[3].time) and [s.time for s in sp[1:3]] == [60.0] * 2
    assert all(s.data_id == 7 and s.entry is ENTRY for s in sp)
    np.testing.assert_array_equal(sp[0].mz, [500.1, 500.2, 500.3])  # sorted
    np.testing.assert_array_equal(sp[0].intensity, [20.0, 10.0, 30.0])
    np.testing.assert_array_equal(sp[1].mz, [500.1, 500.2, 500.3])  # row with empty m/z dropped
    np.testing.assert_array_equal(sp[2].mz, [500.2, 500.4])  # missing intensity dropped
    np.testing.assert_array_equal(sp[3].mz, [499.0, 500.0, 501.0])
    assert sp[0].mz.dtype == np.float64


def test_controls_are_roles_not_times(tmp_path):
    """A reverse experiment: a real 0 s time point next to the undeuterated control. The
    control has a role and no time, so the two are told apart without a user-set time."""
    write_xlsx(tmp_path / "r.xlsx", header=["undeut", None, "0 s", None, "TD", None])
    sp = hio.read_hxexpress_file(tmp_path / "r.xlsx", ENTRY)
    assert [s.role for s in sp] == ["undeut", "", "fulldeut"]
    assert np.isnan(sp[0].time) and sp[1].time == 0.0 and np.isnan(sp[2].time)
    assert [s.rep for s in sp] == [1, 1, 1]
    s = SpectrumSet.from_spectra(sp)
    assert list(s.role) == ["undeut", "", "fulldeut"] and list(s.time_idx()) == [0, 1, 2]
    assert list(s.is_control("undeut")) == [True, False, False]


def test_read_spectra_records_no_control_times(specroot):
    s = hio.read_spectra(hio.scan(specroot, "specexport"), specroot, "specexport",
                         project="proj")
    assert "undeut_time" not in s.source and "fulldeut_time" not in s.source
    assert set(s.role.tolist()) == {"undeut", "", "fulldeut"}
    # a record written before the roles still reads (its times are ignored, logged)
    again = hio.read_spectra_source({**s.source, "undeut_time": 0.0, "fulldeut_time": 1e6})
    assert list(again.role) == list(s.role)


def test_read_hxexpress_errors(tmp_path):
    with pytest.raises(hio.ReadError, match="cannot read"):
        hio.read_hxexpress_file(tmp_path / "missing.xlsx", ENTRY)
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(hio.ReadError, match="unsupported"):
        hio.read_hxexpress_file(tmp_path / "a.txt", ENTRY)
    (tmp_path / "bad.xlsx").write_text("not a workbook")
    with pytest.raises(hio.ReadError):
        hio.read_hxexpress_file(tmp_path / "bad.xlsx", ENTRY)
    write_xlsx(tmp_path / "text.xlsx", body=[[500.0, "abc", None, None, None, None, None, None]])
    with pytest.raises(hio.ReadError, match="not a number"):
        hio.read_hxexpress_file(tmp_path / "text.xlsx", ENTRY)
    write_xlsx(tmp_path / "label.xlsx", header=["undeut", None, "oops", None], body=[[1, 2, 3, 4]])
    with pytest.raises(hio.ReadError, match="column 3"):
        hio.read_hxexpress_file(tmp_path / "label.xlsx", ENTRY)
    write_hx_csv(tmp_path / "empty.csv", header=[None, None], body=[])
    with pytest.raises(hio.ReadError, match="no time labels"):
        hio.read_hxexpress_file(tmp_path / "empty.csv", ENTRY)


def test_scan_hxexpress(tmp_path):
    for name in [
        "B_0003-0009-ABCDEFG-z3-x.xlsx",
        "A_0001-0005-ABCDE-z2-x.xlsx",
        "A_0001-0005-ABCDE-z1-x.xlsx",
        "notes.xlsx",
        "~$A_0001-0005-ABCDE-z2-x.xlsx",
        "readme.txt",
    ]:
        write_xlsx(tmp_path / name)
    entries = hio.scan_hxexpress(tmp_path)
    assert [(e.sample, e.peptide_range, e.charge) for e in entries] == [
        ("A", "0001-0005", 1), ("A", "0001-0005", 2), ("B", "0003-0009", 3),
    ]
    only_b = hio.scan_hxexpress(tmp_path, samples=["B"])
    assert [e.sample for e in only_b] == ["B"]
    assert hio.scan_hxexpress(tmp_path, peptide_ranges=["0001-0005"]) == entries[:2]


# --- SpecExport -----------------------------------------------------------------------------

def write_spec(path, points):
    path.write_text("".join(f"{m},{i}\n" for m, i in points))


@pytest.fixture
def specroot(tmp_path):
    (tmp_path / "S1.fasta").write_text(">S1 test\nMKVLA\nGHEDW\n")
    pep = tmp_path / "S1" / "0002-0005-KVLA"
    pep.mkdir(parents=True)
    write_spec(pep / "Non-D-1-z2.csv", [(300.2, 5), (300.1, 4), (300.3, 0)])
    write_spec(pep / "Full-D-1-z2.csv", [(301.0, 1), (302.0, 2)])
    write_spec(pep / "60s-1-z2.csv", [(300.5, 7)])
    write_spec(pep / "60s-2-z2.csv", [(300.6, 8)])
    write_spec(pep / "4m-1-z3.csv", [(200.5, 9)])
    (pep / "notes.txt").write_text("ignore")
    (pep / "data_fits_S1_0002-0005_z2.tsv").write_text("ignore")
    (tmp_path / "S1" / "analysis").mkdir()
    (tmp_path / "other").mkdir()  # directory without a FASTA file
    (tmp_path / "other.fasta").write_text(">o\nMK\n")
    (tmp_path / "other" / "x").mkdir()
    return tmp_path


def test_scan_specexport(specroot):
    entries = hio.scan_specexport(specroot)
    assert [(e.sample, e.peptide_range, e.charge, e.peptide, e.file) for e in entries] == [
        ("S1", "0002-0005", 2, "KVLA", "0002-0005-KVLA"),
        ("S1", "0002-0005", 3, "KVLA", "0002-0005-KVLA"),
    ]
    assert (entries[0].start_seq, entries[0].end_seq) == (2, 5)
    assert hio.scan_specexport(specroot, samples=["nope"]) == []


def test_scan_specexport_cuts_the_fasta_by_residues(tmp_path):
    """The FASTA is cut by residues (modX: acK is one), a terminal group kept at its end; a
    letter the registry does not know falls back to the cut by letters."""
    for name, fasta, folder, peptide in [
            ("S", "Ac-MAacKLVGS", "0002-0004-AacKL", "AacKL"),
            ("T", "Ac-MAacKLVGS", "0001-0002-MA", "Ac-MA"),
            ("U", "MAKLVGXS", "0002-0004-AKL", "AKL")]:
        (tmp_path / name / folder).mkdir(parents=True)
        (tmp_path / f"{name}.fasta").write_text(f">{name}\n{fasta}\n")
        (tmp_path / name / folder / "60s-1-z2.csv").write_text("")
        (entry,) = hio.scan(tmp_path, "specexport", samples=[name])
        assert entry.peptide == peptide


def test_scan_specexport_range_outside_fasta(specroot):
    (specroot / "S1" / "0004-0099-XXXX").mkdir()
    with pytest.raises(hio.ReadError, match="outside"):
        hio.scan_specexport(specroot)


def test_read_specexport_dir(specroot):
    e = hio.scan_specexport(specroot)[0]
    sp = hio.read_specexport_dir(specroot / "S1" / e.file, e, data_id=3)
    assert [(s.label, s.rep, s.role) for s in sp] == [
        ("Non-D-1-z2.csv", 1, "undeut"), ("60s-1-z2.csv", 1, ""), ("60s-2-z2.csv", 2, ""),
        ("Full-D-1-z2.csv", 1, "fulldeut"),
    ]
    assert np.isnan(sp[0].time) and np.isnan(sp[3].time) and sp[1].time == sp[2].time == 60.0
    np.testing.assert_array_equal(sp[0].mz, [300.1, 300.2, 300.3])
    np.testing.assert_array_equal(sp[0].intensity, [4, 5, 0])
    assert all(s.data_id == 3 for s in sp)
    z3 = hio.read_specexport_dir(specroot / "S1" / e.file, hio.scan_specexport(specroot)[1])
    assert [(s.time, s.rep) for s in z3] == [(240.0, 1)]
    with pytest.raises(hio.ReadError, match="no spectrum files"):
        hio.read_specexport_dir(specroot / "S1" / e.file, replace(e, charge=9))


def test_specexport_bad_rows(tmp_path):
    d = tmp_path / "S" / "0001-0002-AB"
    d.mkdir(parents=True)
    (d / "60s-1-z1.csv").write_text("100.0\n")
    e = PeptideEntry("0001-0002-AB", "S", "AB", "0001-0002", 1, 2, 1)
    with pytest.raises(hio.ReadError, match="two columns"):
        hio.read_specexport_dir(d, e)
    (d / "60s-1-z1.csv").write_text("100.0,abc\n")
    with pytest.raises(hio.ReadError, match="not a number"):
        hio.read_specexport_dir(d, e)


# --- tables, fasta, filters -----------------------------------------------------------------

def test_read_peptide_table(tmp_path):
    p = tmp_path / "meta.csv"
    p.write_text(
        "Index,file,sample,start_seq,end_seq,peptide_range,charge,peptide,modification,polymodal\n"
        "0,a.xlsx,S1,1,5,0001-0005,2.0,PEPTK,H:1 Hex:1,4 60\n"
        "1,b.xlsx,S2,3,9,0003-0009,3,PEPTIDE,,\n"
    )
    a, b = hio.read_peptide_table(p)
    assert (a.charge, a.start_seq, a.peptide_range) == (2, 1, "0001-0005")
    assert a.modification == "H:1 Hex:1"
    assert dict(a.extra) == {"polymodal": "4 60"}
    assert b.modification == "" and b.charge == 3


def test_read_peptide_table_errors(tmp_path):
    p = tmp_path / "meta.csv"
    p.write_text("file,sample\na,b\n")
    with pytest.raises(hio.ReadError, match="missing column"):
        hio.read_peptide_table(p)
    p.write_text("file,sample,peptide,peptide_range,start_seq,end_seq,charge\na,b,P,1-2,x,2,2\n")
    with pytest.raises(hio.ReadError, match="line 2"):
        hio.read_peptide_table(p)
    with pytest.raises(hio.ReadError):
        hio.read_peptide_table(tmp_path / "nope.csv")


def test_read_fasta(tmp_path, caplog):
    p = tmp_path / "a.fasta"
    p.write_text(">one\nMKV\nLA\n>two\nGGG\n")
    with caplog.at_level("WARNING"):
        assert hio.read_fasta_sequence(p) == "GGG"
    assert "2 records" in caplog.text
    p.write_text(">only\nMK V\nLA\n")
    assert hio.read_fasta_sequence(p) == "MKVLA"
    p.write_text("no header\n")
    with pytest.raises(hio.ReadError):
        hio.read_fasta_sequence(p)


# --- read_spectra and SpectrumSet -----------------------------------------------------------

@pytest.fixture
def hxdir(tmp_path):
    for name in ["A_0001-0005-ABCDE-z2-x.xlsx", "B_0003-0009-ABCDEFG-z3-x.xlsx"]:
        write_xlsx(tmp_path / name)
    return tmp_path


def test_read_spectra_set(hxdir):
    entries = hio.scan(hxdir, "hxexpress")
    s = hio.read_spectra(entries, hxdir, "hxexpress", project="proj")
    assert isinstance(s, SpectrumSet) and len(s) == 8
    assert list(s.data_id) == [0] * 4 + [1] * 4
    assert list(s.sample) == ["A"] * 4 + ["B"] * 4
    assert list(s.charge) == [2] * 4 + [3] * 4
    assert list(s.time_idx()) == [0, 1, 1, 2] * 2
    assert s.offsets[-1] == len(s.mz) == len(s.intensity)
    mz, inten = s.spectrum(2)
    np.testing.assert_array_equal(mz, [500.2, 500.4])
    assert set(s.peptide) == {"ABCDE", "ABCDEFG"}


def test_read_spectra_filter_keeps_data_ids(hxdir):
    entries = hio.scan(hxdir, "hxexpress")
    s = hio.read_spectra(entries, hxdir, "hxexpress", project="proj", samples=["B"])
    assert set(s.sample) == {"B"} and set(s.data_id) == {1}


def test_read_spectra_progress_and_cancel(hxdir):
    entries = hio.scan(hxdir, "hxexpress")
    calls = []
    hio.read_spectra(
        entries, hxdir, "hxexpress", project="proj", progress=lambda d, t: calls.append((d, t))
    )
    assert calls == [(1, 2), (2, 2)]
    flags = iter([False, True])
    with pytest.raises(Cancelled):
        hio.read_spectra(entries, hxdir, "hxexpress", project="proj", cancel=lambda: next(flags))
    with pytest.raises(ValueError, match="unknown format"):
        hio.read_spectra(entries, hxdir, "mzml", project="proj")
    with pytest.raises(ValueError, match="unknown format"):
        hio.scan(hxdir, "mzml")


def test_read_spectra_specexport(specroot):
    entries = hio.scan(specroot, "specexport")
    s = hio.read_spectra(entries, specroot, "specexport", project="proj")
    assert len(s) == 5
    assert sorted(set(s.time[np.isfinite(s.time)])) == [60.0, 240.0]
    assert list(s.time_idx()[s.data_id == 0]) == [0, 1, 1, 2]
    assert list(s.role[s.data_id == 0]) == ["undeut", "", "", "fulldeut"]


def same_spectra(a, b):
    assert list(a.spectrum_id) == list(b.spectrum_id) and list(a.data_id) == list(b.data_id)
    np.testing.assert_array_equal(a.mz, b.mz)
    np.testing.assert_array_equal(a.intensity, b.intensity)
    np.testing.assert_array_equal(a.time, b.time)
    np.testing.assert_array_equal(a.role, b.role)


def test_read_spectra_records_the_read_and_repeats_it(hxdir, tmp_path_factory, caplog):
    entries = hio.scan(hxdir, "hxexpress")
    s = hio.read_spectra(entries, hxdir, "hxexpress", project="proj", samples=["B"])
    src = s.source
    assert src["reader"] == "hxexpress" and src["data_dir"] == str(hxdir)
    assert src["samples"] == ["B"]
    assert [PeptideEntry.from_dict(e) for e in src["entries"]] == entries
    name = entries[1].file
    assert list(src["files"]) == [name] and src["files"][name] == hio.file_identity(hxdir / name)
    again = hio.read_spectra_source(src)
    same_spectra(again, s)
    assert again.source == src and hio.read_source(src).source == src
    # moved, recorded by name only, then an input updated between runs
    moved = tmp_path_factory.mktemp("moved")
    for e in entries:
        (moved / e.file).write_bytes((hxdir / e.file).read_bytes())
    named = hio.read_spectra(entries, hxdir, "hxexpress", project="proj", record_paths="name")
    assert named.source["data_dir"] == hxdir.name
    same_spectra(hio.read_spectra_source(named.source, directory=moved), named)
    with pytest.raises(hio.ReadError, match="not found"):
        hio.read_spectra_source(named.source)
    write_xlsx(moved / name, body=[[r[0], (r[1] or 0) * 2 if r[1] else r[1], *r[2:]]
                                   for r in BODY])
    with caplog.at_level(logging.WARNING, logger="hxex.io.identity"):
        changed = hio.read_spectra_source(src, directory=moved)
    assert "has changed since it was read" in caplog.text
    assert changed.source["files"][name]["previous"] == src["files"][name]
    with pytest.raises(hio.ReadError, match="has changed"):
        hio.read_spectra_source(src, directory=moved, on_change="error")
    with pytest.raises(ValueError, match="record_paths"):
        hio.read_spectra(entries, hxdir, "hxexpress", project="proj", record_paths="relative")
    with pytest.raises(ValueError, match="not a read_spectra record"):
        hio.read_spectra_source({"reader": "other"})


def test_a_specexport_folder_is_identified_as_a_whole(specroot):
    entries = hio.scan(specroot, "specexport")
    s = hio.read_spectra(entries, specroot, "specexport", project="proj")
    folder = specroot / "S1" / "0002-0005-KVLA"
    ident = s.source["files"]["S1/0002-0005-KVLA"]
    assert ident == hio.directory_identity(folder) and ident["n_files"] == 7
    assert ident["algorithm"] == "sha256-tree"
    same_spectra(hio.read_spectra_source(s.source), s)
    (folder / "notes.txt").write_text("changed")  # any file in the folder counts
    assert hio.directory_identity(folder)["checksum"] != ident["checksum"]
    (folder / "notes.txt").rename(folder / "renamed.txt")
    assert hio.directory_identity(folder)["n_files"] == 7
    with pytest.raises(hio.ReadError, match="has changed"):
        hio.read_spectra_source(s.source, on_change="error")


def test_empty_spectrum_set():
    s = SpectrumSet.from_spectra([])
    assert len(s) == 0 and len(s.mz) == 0 and len(s.time_idx()) == 0


# --- real data ------------------------------------------------------------------------------

BIMODAL = os.environ.get("HXEX_BIMODAL_DATA")
SPECEXPORT = os.environ.get("HXEX_SPECEXPORT_EXAMPLE")


def pyhxexpress_time(s, i: int) -> float:
    """pyHXExpress's time stamp of spectrum ``i``: 0 for an undeuterated control, 1e6 for a
    fully deuterated one (hxex keeps a role and no time), else the time."""
    return {"undeut": 0.0, "fulldeut": 1e6}.get(str(s.role[i]), float(s.time[i]))


@pytest.mark.skipif(not BIMODAL, reason="HXEX_BIMODAL_DATA not set")
def test_bimodal_raw_spectra_match_pyhxexpress_exactly():
    root = Path(BIMODAL)
    entries = hio.read_peptide_table(root / "hdxms_testsets_metadf.csv")
    s = hio.read_spectra(entries, root, "hxexpress", project="proj")
    zpath = root / "output_fixed1pop_19Feb2024" / "allrawdata_19Feb2024.zip"
    with zipfile.ZipFile(zpath) as z:
        text = z.read(z.namelist()[0]).decode()
    ref = defaultdict(list)
    for r in csv.DictReader(io.StringIO(text)):
        key = (r["file"], float(r["time"]), int(r["rep"]))
        ref[key].append((float(r["mz"]), float(r["Intensity"])))
    assert len(s) == len(ref) == 566
    assert len(s.mz) == 496708
    for i in range(len(s)):
        expected = np.array(ref[(str(s.file[i]), pyhxexpress_time(s, i), int(s.rep[i]))])
        mz, inten = s.spectrum(i)
        np.testing.assert_array_equal(mz, expected[:, 0])
        np.testing.assert_array_equal(inten, expected[:, 1])


@pytest.mark.skipif(not SPECEXPORT, reason="HXEX_SPECEXPORT_EXAMPLE not set")
def test_specexport_example_matches_pyhxexpress_run():
    root = Path(SPECEXPORT)
    out = next(root.glob("hdxms_analysis_*"))
    metadf = next(out.glob("metadf_asrun_*.csv"))
    deut = next(out.glob("alldeutdata_*.csv"))
    entries = hio.scan(root, "specexport")
    ref = {
        (r["sample"], r["peptide_range"], int(float(r["charge"])), r["peptide"])
        for r in csv.DictReader(metadf.open())
    }
    assert {(e.sample, e.peptide_range, e.charge, e.peptide) for e in entries} == ref
    s = hio.read_spectra(entries, root, "specexport", project="proj")
    mine = {
        (str(s.sample[i]), str(s.peptide_range[i]), int(s.charge[i]),
         pyhxexpress_time(s, i), int(s.rep[i]))
        for i in range(len(s))
    }
    theirs = {
        (r["sample"], r["peptide_range"], int(float(r["charge"])), float(r["time"]),
         int(float(r["rep"])))
        for r in csv.DictReader(deut.open())
    }
    assert mine == theirs


# --- pooled folders -------------------------------------------------------------------------

@pytest.fixture
def pooldir(tmp_path):
    for name, files in {
        "0001-0004-MKVL": ["60s-1-z2.csv", "60s-2-z2.csv", "Non-D-1-z2.csv"],
        "0010-0014-ACacKDE": ["60s-1-z2.csv", "60s-1-z3.csv"],
        "0020-0022-GGG": ["4.1m-1-z2.csv"],
        "0030-0031-XXQ": ["60s-1-z2.csv"],  # X is not a residue
        "analysis_output": ["60s-1-z2.csv"],
    }.items():
        (tmp_path / name).mkdir()
        for f in files:
            write_spec(tmp_path / name / f, [(100.0, 1.0), (101.0, 2.0)])
    (tmp_path / "metadf.csv").write_text("x")
    (tmp_path / "pooled.fasta").write_text(
        ">ALPHA1 alpha one\nAAMKVLAA\n>BETA2\nAAAAMKVLGGGAC\n>OTHER\nWWACKDEWW\n"
    )
    return tmp_path


def test_scan_pooled(pooldir):
    entries = hio.scan_specexport_pooled(pooldir, sample="mix")
    assert [(e.file, e.charge, e.peptide, e.sample) for e in entries] == [
        ("0001-0004-MKVL", 2, "MKVL", "mix"),
        ("0010-0014-ACacKDE", 2, "ACacKDE", "mix"),
        ("0010-0014-ACacKDE", 3, "ACacKDE", "mix"),
        ("0020-0022-GGG", 2, "GGG", "mix"),
    ]
    assert (entries[1].start_seq, entries[1].end_seq) == (10, 14)
    assert hio.scan(pooldir, "pooled")[0].sample == ""
    only = hio.scan_specexport_pooled(pooldir, peptide_ranges=["0020-0022"])
    assert [e.file for e in only] == ["0020-0022-GGG"]


def test_scan_pooled_warns_on_residue_count_mismatch(pooldir, caplog):
    (pooldir / "0040-0049-MKV").mkdir()
    write_spec(pooldir / "0040-0049-MKV" / "60s-1-z2.csv", [(1.0, 1.0)])
    with caplog.at_level("WARNING"):
        entries = hio.scan_specexport_pooled(pooldir)
    assert "0040-0049-MKV" in caplog.text and any(e.file == "0040-0049-MKV" for e in entries)


def test_read_spectra_pooled(pooldir):
    entries = hio.scan(pooldir, "pooled", sample="mix")
    s = hio.read_spectra(entries, pooldir, "pooled", project="proj")
    assert set(s.sample) == {"mix"}
    assert len(s) == 3 + 1 + 1 + 1
    assert 246.0 in s.time  # 4.1m


def test_a_pooled_project_reads_its_peptide_table(pooldir):
    """[input] peptides replaces the search of a pooled folder: only its rows are read, with
    the table's sample (not [input] sample); file is the peptide folder in data_dir."""
    from hxex.config import load_project, read_input

    (pooldir / "files.csv").write_text(
        "file,sample,peptide,start_seq,end_seq,charge\n0010-0014-ACacKDE,mix,ACacKDE,10,14,3\n")
    (pooldir / "hxex.toml").write_text(
        'project = "p"\n[input]\nformat = "pooled"\ndata_dir = "."\nsample = "pool"\n'
        'peptides = "files.csv"\n')
    s = read_input(load_project(pooldir / "hxex.toml"))
    assert len(s) == 1 and set(s.sample) == {"mix"} and set(s.charge) == {3}


@pytest.fixture
def statedir(tmp_path):
    """Pooled data as state folders: two conditions of one pool, side by side."""
    root = tmp_path / "states"
    for state, folders in {"alone": ["0001-0004-MKVL", "0010-0012-GGG"],
                           "with partner": ["0001-0004-MKVL"]}.items():
        for name in folders:
            (root / state / name).mkdir(parents=True)
            for f in ("Non-D-1-z2.csv", "4m-1-z2.csv", "4m-2-z2.csv"):
                write_spec(root / state / name / f, [(100.0, 1.0), (101.0, 2.0)])
    (root / "notes").mkdir()  # holds no peptide folders: not a state
    (root / "pool.fasta").write_text(">A1\nMKVLGGG\n")
    return root


def test_a_pooled_folder_of_state_folders_reads_each_state_as_a_sample(statedir, caplog):
    assert hio.pooled_layout(statedir) == "states"
    assert hio.state_folders(statedir) == ["alone", "with partner"]
    with caplog.at_level("WARNING"):
        entries = hio.scan(statedir, "pooled")
    assert not caplog.text
    assert sorted((e.sample, e.file, e.peptide) for e in entries) == [
        ("alone", "alone/0001-0004-MKVL", "MKVL"),
        ("alone", "alone/0010-0012-GGG", "GGG"),
        ("with partner", "with partner/0001-0004-MKVL", "MKVL"),
    ]
    s = hio.read_spectra(entries, statedir, "pooled", project="proj")
    assert len(s) == 9 and set(s.sample) == {"alone", "with partner"}
    assert "proj:with partner/0001-0004-MKVL/4m-2-z2.csv" in set(s.spectrum_id)
    index = hio.spectrum_index(entries, statedir, "pooled", project="proj")
    assert sorted(index.spectrum_id) == sorted(s.spectrum_id)
    only = hio.scan(statedir, "pooled", samples=["with partner"])
    assert {e.sample for e in only} == {"with partner"}


def test_a_pooled_sample_name_with_state_folders_is_not_used_and_says_so(statedir, caplog):
    with caplog.at_level("WARNING"):
        entries = hio.scan(statedir, "pooled", sample="pool")
    assert {e.sample for e in entries} == {"alone", "with partner"}
    assert "input.sample = 'pool' is not used" in caplog.text


def test_a_pooled_folder_with_peptide_and_state_folders_reads_the_peptide_folders(statedir,
                                                                                   caplog):
    """Peptide folders decide the layout (as before state folders were read); the folders of
    peptide folders left unread are named."""
    (statedir / "0020-0022-AAA").mkdir()
    write_spec(statedir / "0020-0022-AAA" / "4m-1-z2.csv", [(100.0, 1.0)])
    with caplog.at_level("WARNING"):
        entries = hio.scan(statedir, "pooled", sample="pool")
    assert [(e.sample, e.file) for e in entries] == [("pool", "0020-0022-AAA")]
    assert "not read (alone, with partner)" in caplog.text


def test_a_pooled_project_of_state_folders_needs_no_sample(statedir):
    """[input] sample is needed only for a folder of peptide folders; a peptide table names
    the peptide folder and its state, as for SpecExport."""
    from hxex.config import ProjectError, load_project, read_input

    toml = statedir.parent / "hxex.toml"
    toml.write_text('project = "p"\n[input]\nformat = "pooled"\ndata_dir = "states"\n')
    s = read_input(load_project(toml))
    assert set(s.sample) == {"alone", "with partner"}
    (statedir.parent / "files.csv").write_text(
        "file,sample,peptide,start_seq,end_seq,charge\n"
        "0001-0004-MKVL,with partner,MKVL,1,4,2\n")
    toml.write_text('project = "p"\n[input]\nformat = "pooled"\ndata_dir = "states"\n'
                    'peptides = "files.csv"\n')
    s = read_input(load_project(toml))
    assert len(s) == 3 and set(s.sample) == {"with partner"}
    one = statedir / "alone"
    toml.write_text(f'project = "p"\n[input]\nformat = "pooled"\ndata_dir = "{one}"\n')
    with pytest.raises(ProjectError, match="input.sample is needed for pooled when data_dir "
                                           "holds peptide folders"):
        load_project(toml)


def test_match_candidates(pooldir):
    entries = hio.scan_specexport_pooled(pooldir)
    m = hio.match_candidates(entries, pooldir / "pooled.fasta", classes=["ALPHA", "BETA", "OTHER"])
    by_file = {x.entry.file: x for x in m if x.entry.charge == 2}
    assert by_file["0001-0004-MKVL"].matches == ("ALPHA1", "BETA2")  # FASTA order
    assert by_file["0001-0004-MKVL"].classes == ("ALPHA", "BETA")
    assert by_file["0010-0014-ACacKDE"].matches == ("OTHER",)  # acK matches K
    assert by_file["0010-0014-ACacKDE"].classes == ("OTHER",)
    assert by_file["0020-0022-GGG"].matches == ("BETA2",)
    assert hio.match_candidates(entries[:1], pooldir / "pooled.fasta")[0].classes == ()
    only_gg = hio.match_candidates(
        [PeptideEntry("f", "", "QQQ", "0001-0003", 1, 3, 2)], pooldir / "pooled.fasta"
    )
    assert only_gg[0].matches == ()


def test_match_candidates_errors(pooldir):
    (pooldir / "empty.fasta").write_text("")
    with pytest.raises(hio.ReadError):
        hio.match_candidates([ENTRY], pooldir / "empty.fasta")
    from hxex.chem import SequenceError

    bad = PeptideEntry("f", "", "PEPzzK", "0001-0006", 1, 6, 2)
    with pytest.raises(SequenceError):
        hio.match_candidates([bad], pooldir / "pooled.fasta")


def test_read_fasta_records(tmp_path):
    p = tmp_path / "a.fasta"
    p.write_text(">ALPHA1 alpha one\nMK\nVL\n\n>empty\n>BETA2\nGG G\n")
    assert hio.read_fasta_records(p) == [("ALPHA1", "MKVL"), ("BETA2", "GGG")]


POOLED = os.environ.get("HXEX_POOLED_EXAMPLE")


@pytest.mark.skipif(not POOLED, reason="HXEX_POOLED_EXAMPLE not set")
def test_pooled_example_matches_pyhxexpress_metadf():
    """A pooled folder with one FASTA and a metadf_*.csv written by the owner's pooled scripts
    (``sample_matches`` lists the matching records; a ``*_class`` column names each peptide's
    protein class, the first part before ``_``)."""
    root = Path(POOLED)
    rows = list(csv.DictReader(next(root.glob("metadf_*.csv")).open()))
    class_column = next(c for c in rows[0] if c.endswith("_class"))
    ref = {(r["file"], int(float(r["charge"]))): (r["sample_matches"], r[class_column])
           for r in rows}
    classes = sorted({cls.split("_")[0] for _, cls in ref.values() if cls})
    entries = hio.scan(root, "pooled")
    found = hio.match_candidates(entries, next(root.glob("*.fasta")), classes=classes)
    assert {(m.entry.file, m.entry.charge) for m in found} == set(ref)
    for m in found:
        matches, cls = ref[(m.entry.file, m.entry.charge)]
        assert ";".join(m.matches) == matches
        assert ";".join(m.classes) == cls.split("_")[0]


# --- spectrum ids ---------------------------------------------------------------------------

def test_spectrum_ids_hxexpress(tmp_path):
    header = ["undeut", None, "1 min", None, "1 min", None]
    write_xlsx(tmp_path / "a.xlsx", header=header, body=[[1, 2, 3, 4, 5, 6]])
    sp = hio.read_hxexpress_file(tmp_path / "a.xlsx", ENTRY)
    # the two "1 min" columns have the same label, time and (different) rep, and distinct ids
    assert [s.spectrum_id for s in sp] == ["f.xlsx#1", "f.xlsx#3", "f.xlsx#5"]
    assert [s.rep for s in sp] == [1, 1, 2]


def test_spectrum_ids_specexport_and_pooled(specroot, pooldir):
    s = hio.read_spectra(hio.scan(specroot, "specexport"), specroot, "specexport", project="proj")
    assert "proj:S1/0002-0005-KVLA/60s-2-z2.csv" in s.spectrum_id
    p = hio.read_spectra(
        hio.scan(pooldir, "pooled", sample="mix"), pooldir, "pooled", project="proj"
    )
    assert "proj:0001-0004-MKVL/60s-2-z2.csv" in p.spectrum_id  # the sample label is not in the id


def test_spectrum_ids_are_unique_and_stable(hxdir, specroot):
    for root, fmt in [(hxdir, "hxexpress"), (specroot, "specexport")]:
        entries = hio.scan(root, fmt)
        s = hio.read_spectra(entries, root, fmt, project="proj")
        assert len(set(s.spectrum_id)) == len(s)
        again = hio.read_spectra(entries[::-1], root, fmt, project="proj")
        assert set(again.spectrum_id) == set(s.spectrum_id)  # independent of order
        i = s.find(str(s.spectrum_id[3]))
        assert i == 3
    with pytest.raises(KeyError):
        s.find("nope")


def test_entries_sharing_a_file_get_distinct_ids(specroot, hxdir):
    """One peptide folder listed twice in the table (two modifications of one peptide): its
    spectra are read for each entry, the ids of the two carry the entry's series number, and
    every other id stays as it was, also when a filter leaves one twin out; spectrum_index
    gives the same ids."""
    import dataclasses

    entries = hio.scan(specroot, "specexport")
    plain = hio.read_spectra(entries, specroot, "specexport", project="p")
    twin = dataclasses.replace(entries[0], modification="H:1")
    both = list(entries) + [twin]
    s = hio.read_spectra(both, specroot, "specexport", project="p")
    assert len(set(s.spectrum_id)) == len(s) == len(plain) + int((plain.data_id == 0).sum())
    shared = s.data_id == 0
    assert all(i.endswith("#series0") for i in s.spectrum_id[shared])
    assert all(i.endswith(f"#series{len(entries)}") for i in s.spectrum_id[s.data_id == len(
        entries)])
    others = ~shared & (s.data_id != len(entries))
    assert set(s.spectrum_id[others]) == set(plain.spectrum_id[plain.data_id != 0])
    assert set(s.modification[s.data_id == len(entries)]) == {"H:1"}
    index = hio.spectrum_index(both, specroot, "specexport", project="p")
    assert list(index.spectrum_id) == list(s.spectrum_id)
    one = hio.read_spectra(both, specroot, "specexport", project="p",
                           peptide_ranges=[entries[0].peptide_range])
    assert set(one.spectrum_id) <= set(s.spectrum_id)
    # an HX-Express file is read whole: two entries of one file share their columns
    hx = hio.scan(hxdir, "hxexpress")
    h = hio.read_spectra(list(hx) + [dataclasses.replace(hx[0], modification="H:1")], hxdir,
                         "hxexpress", project="p")
    assert len(set(h.spectrum_id)) == len(h)


def test_project_name_qualifies_ids(hxdir):
    from hxex.spectra import qualify_spectrum_id, split_spectrum_id

    entries = hio.scan(hxdir, "hxexpress")
    a = hio.read_spectra(entries, hxdir, "hxexpress", project="alpha")
    b = hio.read_spectra(entries, hxdir, "hxexpress", project="beta")
    assert a.spectrum_id[0] == "alpha:A_0001-0005-ABCDE-z2-x.xlsx#1"
    assert not set(a.spectrum_id) & set(b.spectrum_id)
    assert split_spectrum_id(str(a.spectrum_id[0])) == ("alpha", "A_0001-0005-ABCDE-z2-x.xlsx#1")
    assert split_spectrum_id(qualify_spectrum_id("p", "a:b/c")) == ("p", "a:b/c")
    assert a.find("alpha:A_0001-0005-ABCDE-z2-x.xlsx#3") == 1


@pytest.mark.parametrize("project", ["", "a:b"])
def test_bad_project_names(hxdir, project):
    with pytest.raises(ValueError, match="project name"):
        hio.read_spectra(hio.scan(hxdir, "hxexpress"), hxdir, "hxexpress", project=project)


def test_project_is_required(hxdir):
    with pytest.raises(TypeError):
        hio.read_spectra(hio.scan(hxdir, "hxexpress"), hxdir, "hxexpress")


@pytest.mark.parametrize("bad", ["", ":x", "noseparator"])
def test_split_spectrum_id_errors(bad):
    from hxex.spectra import split_spectrum_id

    with pytest.raises(ValueError):
        split_spectrum_id(bad)


# --- audit fixes: units, suffixes, encodings, whole numbers, duplicate files --------------------

@pytest.mark.parametrize(
    "label, seconds",
    [("500 ms", 0.5), ("15 sec", 15.0), ("30 seconds", 30.0), ("3 minutes", 180.0),
     ("5 MIN", 300.0), ("2 hrs", 7200.0), ("1 hour", 3600.0), ("2 Hr", 7200.0)],
)
def test_time_units_are_whole_words(label, seconds):
    assert hio.parse_hxexpress_label(label)[0] == seconds


@pytest.mark.parametrize("label", ["5 mi", "5 minu", "500 msx", "2 hz", "3 sx", "4 m2"])
def test_unknown_time_units_raise(label):
    """A unit is not matched on its first letter ('500 ms' was 30000 s)."""
    with pytest.raises(hio.ReadError, match="unknown time unit"):
        hio.parse_hxexpress_label(label)


def test_scan_hxexpress_takes_csv_and_any_suffix_case(tmp_path):
    write_xlsx(tmp_path / "A_0001-0005-ABCDE-z1-x.XLSX")
    write_hx_csv(tmp_path / "B_0003-0009-ABCDEFG-z3-x.csv")
    (tmp_path / "C_0001-0005-ABCDE-z2-x.txt").write_text("not data")
    (tmp_path / "D_0001-0005-ABCDE-z2-x.csv").mkdir()  # a directory, not a file
    entries = hio.scan_hxexpress(tmp_path)
    assert [(e.sample, e.charge, e.file) for e in entries] == [
        ("A", 1, "A_0001-0005-ABCDE-z1-x.XLSX"), ("B", 3, "B_0003-0009-ABCDEFG-z3-x.csv")]
    for e in entries:  # what the scan lists, the reader reads
        assert len(hio.read_hxexpress_file(tmp_path / e.file, e)) == 4


def test_text_that_is_not_utf8_is_a_read_error(tmp_path, specroot):
    table = tmp_path / "meta.csv"
    table.write_bytes(b"file,sample,peptide,peptide_range,start_seq,end_seq,charge\n"
                      b"a,S\xe9,AB,1-2,1,2,1\n")
    with pytest.raises(hio.ReadError, match="meta.csv"):
        hio.read_peptide_table(table)
    fasta = tmp_path / "x.fasta"
    fasta.write_bytes(b">s\xe9q\nMK\n")
    with pytest.raises(hio.ReadError, match="x.fasta"):
        hio.read_fasta_sequence(fasta)
    hx = tmp_path / "A_0001-0005-ABCDE-z1-x.csv"
    hx.write_bytes(b"undeut,\n\xff,1\n")
    with pytest.raises(hio.ReadError, match="z1-x.csv"):
        hio.read_hxexpress_file(hx, ENTRY)
    pep = specroot / "S1" / "0002-0005-KVLA"
    (pep / "60s-1-z2.csv").write_bytes(b"300.5,7\n\xff\n")
    with pytest.raises(hio.ReadError, match="60s-1-z2.csv"):
        hio.read_specexport_dir(pep, hio.scan_specexport(specroot)[0])


def test_table_integers_must_be_whole(tmp_path):
    p = tmp_path / "meta.csv"
    head = "file,sample,peptide,peptide_range,start_seq,end_seq,charge\n"
    p.write_text(head + "a,S,AB,1-2,1.0,2,2.0\n")
    assert hio.read_peptide_table(p)[0].charge == 2
    for bad in ("a,S,AB,1-2,1,2,2.5\n", "a,S,AB,1-2,1.5,2,2\n", "a,S,AB,1-2,1,nan,2\n"):
        p.write_text(head + bad)
        with pytest.raises(hio.ReadError, match="line 2.*whole number"):
            hio.read_peptide_table(p)


def test_specexport_two_files_for_one_time_and_replicate(specroot, caplog):
    pep = specroot / "S1" / "0002-0005-KVLA"
    e = hio.scan_specexport(specroot)[0]
    write_spec(pep / "1m-1-z2.csv", [(300.5, 7)])  # the same spectrum as 60s-1-z2.csv
    with pytest.raises(hio.ReadError, match="1m-1-z2.csv and 60s-1-z2.csv are both replicate 1 "
                                            "at 1 min"):
        hio.read_specexport_dir(pep, e)
    (pep / "1m-1-z2.csv").unlink()
    # a time point at 0 s next to the undeuterated control: told apart by the role, no warning
    write_spec(pep / "0s-1-z2.csv", [(300.5, 7)])
    with caplog.at_level(logging.WARNING, logger="hxex.io"):
        sp = hio.read_specexport_dir(pep, e)
    assert caplog.text == "" and [s.role for s in sp][:2] == ["undeut", ""]
    assert np.isnan(sp[0].time) and sp[1].time == 0.0 and sp[0].rep == sp[1].rep == 1


# --- points: times and roles ------------------------------------------------------------------

def test_time_index_and_time_point_put_the_controls_at_the_ends():
    from hxex.spectra import check_role, point_order_key, time_index, time_point

    time = np.array([np.nan, 60.0, np.nan, 4.0, 60.0, 72000.0])
    role = np.array(["fulldeut", "", "undeut", "", "", "fulldeut"])
    data_id = np.zeros(6, dtype=int)
    assert list(time_index(data_id, time, role)) == [4, 2, 0, 1, 2, 3]
    assert list(time_index(data_id, time)) == [3, 1, 3, 0, 1, 2]  # without roles: NaN last
    points = time_point(time, role)
    assert list(points) == ["fulldeut", "t=60.0", "undeut", "t=4.0", "t=60.0", "t=72000.0"]
    assert point_order_key(np.nan, "undeut") < point_order_key(0.0, "") < point_order_key(
        np.nan, "fulldeut")
    assert check_role(None) == "" and check_role("undeut") == "undeut"
    with pytest.raises(ValueError, match="role must be"):
        check_role("control")
    with pytest.raises(ValueError, match="where"):
        check_role("TD", "where")


def test_a_pooled_folder_takes_the_projects_modified_residues(pooldir):
    """A pooled folder's sequences may use a [run.registry] label (they were read with the
    default registry only, and such a folder was skipped)."""
    import tomllib

    from hxex.config import folder_entries, project_from

    (pooldir / "0050-0052-AmeKA").mkdir()
    write_spec(pooldir / "0050-0052-AmeKA" / "60s-1-z2.csv", [(100.0, 1.0)])
    project = project_from(tomllib.loads(
        f'project = "p"\n[input]\nformat = "pooled"\ndata_dir = "{pooldir.as_posix()}"\n'
        'sample = "pool"\n[[run.registry.residues]]\nlabel = "meK"\nbase = "K"\n'
        'delta = { C = 1, H = 2 }\n'),
        pooldir)
    assert "AmeKA" in {e.peptide for e in folder_entries(project)}
    assert "AmeKA" not in {e.peptide for e in hio.scan(pooldir, "pooled")}


def test_specexport_times_say_what_they_accept():
    """A SpecExport time is Non, Full or a number with one letter; the message said min and ms
    were units, which file names do not take."""
    assert hio.parse_specexport_token("0.8s") == (0.8, "")
    for bad in ("5min", "250ms", "non", "1.5"):
        with pytest.raises(hio.ReadError, match="one letter, s, m or h"):
            hio.parse_specexport_token(bad)


def test_hxexpress_labels_pair_with_their_own_columns(tmp_path):
    """A sheet with a spacer column: each label reads its own column and the next (pyHXExpress
    would read columns 2i, 2i + 1; a deliberate difference, audit 2026-10-28 question 27)."""
    from hxex.io.hxexpress import hxexpress_columns

    header = ("undeut", None, None, "1 min", None, "TD", None)
    cols = hxexpress_columns(header, tmp_path / "f.csv")
    assert [(c[0], c[1]) for c in cols] == [(0, "undeut"), (3, "1 min"), (5, "TD")]


def test_a_bad_specexport_time_names_its_file(tmp_path):
    from hxex.io.specexport import specexport_points

    (tmp_path / "5min-1-z2.csv").write_text("")
    with pytest.raises(hio.ReadError, match=r"5min-1-z2\.csv: SpecExport time '5min'"):
        specexport_points(tmp_path, ENTRY)


def test_peptide_range_is_optional_and_checked(tmp_path, caplog):
    """peptide_range is the range's label: left out (or empty) it is made from start_seq and
    end_seq, zero-padded as in the file names; one that names other residues is a warning
    naming the line (the residues come from start_seq and end_seq)."""
    p = tmp_path / "t.csv"
    p.write_text("file,sample,peptide,start_seq,end_seq,charge\nf,WT,PEPTIDE,1,7,2\n")
    assert hio.read_peptide_table(p)[0].peptide_range == "0001-0007"
    p.write_text("file,sample,peptide,peptide_range,start_seq,end_seq,charge\n"
                 "f,WT,PEPTIDE,,1,7,2\nf,WT,PEPTIDE,1-7,1,7,3\nf,WT,PEPTIDE,0001-0009,1,7,1\n")
    with caplog.at_level(logging.WARNING):
        entries = hio.read_peptide_table(p)
    assert [e.peptide_range for e in entries] == ["0001-0007", "1-7", "0001-0009"]
    assert "line 4" in caplog.text and "line 3" not in caplog.text
    assert entries[2].start_seq == 1 and entries[2].end_seq == 7


def test_peptides_are_selected_by_their_residues(tmp_path):
    """peptide_ranges compares residues, not text ("1-10" selects 0001-0010); peptide_region
    selects every peptide wholly within it; with both, a peptide that matches either."""
    import tomllib

    from hxex.config import ProjectError, project_from

    entries = [PeptideEntry("f", "WT", "P" * (e - s + 1), f"{s:04d}-{e:04d}", s, e, 2)
               for s, e in ((1, 10), (5, 20), (30, 40), (45, 60))]
    pick = lambda **kw: [(e.start_seq, e.end_seq) for e in hio.filter_entries(entries, **kw)]  # noqa: E731
    assert pick(peptide_ranges=["1-10"]) == pick(peptide_ranges=["0001-0010"]) == [(1, 10)]
    assert pick(peptide_region=[1, 40]) == [(1, 10), (5, 20), (30, 40)]
    assert pick(peptide_ranges=["45-60"], peptide_region=[1, 12]) == [(1, 10), (45, 60)]
    assert len(pick()) == 4
    def project(extra):
        return project_from(tomllib.loads(
            'project = "p"\n[input]\nformat = "specexport"\ndata_dir = "d"\n' + extra), tmp_path)
    assert project('peptide_region = [1, 50]\n').input.peptide_region == (1, 50)
    for bad in ('peptide_ranges = ["one-ten"]', "peptide_region = [50, 1]",
                "peptide_region = [1]"):
        with pytest.raises(ProjectError, match="peptide_r"):
            project(bad + "\n")
    with pytest.raises(ProjectError, match="not used by mzml"):
        project_from(tomllib.loads('project = "p"\n[input]\nformat = "mzml"\npeptides = "p.csv"\n'
                                   'peptide_region = [1, 5]\n[[input.runs]]\npath = "a"\n'
                                   'sample = "S"\n'), tmp_path)


# --- the spectra a read would make, without reading them (hxex check) -------------------------

def same_index(index, spectra):
    """``index`` (SpectrumIndex) lists the spectra of ``spectra`` (SpectrumSet), in order."""
    assert list(index.spectrum_id) == list(spectra.spectrum_id)
    for name in ("sample", "start_seq", "end_seq", "charge", "role", "rep", "label"):
        assert list(getattr(index, name)) == list(getattr(spectra, name)), name
    np.testing.assert_array_equal(index.time, spectra.time)


def test_the_spectrum_index_is_what_the_read_makes(hxdir, specroot):
    entries = hio.scan(hxdir, "hxexpress")
    same_index(hio.spectrum_index(entries, hxdir, "hxexpress", project="proj"),
               hio.read_spectra(entries, hxdir, "hxexpress", project="proj"))
    same_index(hio.spectrum_index(entries, hxdir, "hxexpress", project="proj", samples=["B"]),
               hio.read_spectra(entries, hxdir, "hxexpress", project="proj", samples=["B"]))
    entries = hio.scan(specroot, "specexport")
    same_index(hio.spectrum_index(entries, specroot, "specexport", project="proj"),
               hio.read_spectra(entries, specroot, "specexport", project="proj"))
    # the read's errors: two files for one point and replicate
    write_spec(specroot / "S1" / "0002-0005-KVLA" / "1m-1-z2.csv", [(300.5, 7)])
    with pytest.raises(hio.ReadError, match="both replicate 1"):
        hio.spectrum_index(entries, specroot, "specexport", project="proj")


# --- audit 2026-11-01, Session A: tables and folder readers say what is wrong ----------------

HEAD = "file,sample,peptide,start_seq,end_seq,charge\n"


def test_table_cells_with_spaces_are_stripped(tmp_path):
    """A trailing space in a cell (invisible in Excel) is not part of the value (Readers 11)."""
    p = tmp_path / "t.csv"
    p.write_text("file,sample,peptide,start_seq,end_seq,charge,rt,run\n"
                 " a.xlsx ,WT ,LKDVEHFKVSS , 1, 11 ,2 , 3.5 , D_60s.raw \n")
    (e,) = hio.read_peptide_table(p)
    assert (e.file, e.sample, e.peptide, e.charge) == ("a.xlsx", "WT", "LKDVEHFKVSS", 2)
    assert e.extra == {"rt": "3.5", "run": "D_60s.raw"}
    rt = tmp_path / "rt.csv"
    rt.write_text("sample,peptide,rt\nWT , PEP , 4.5 \n")
    assert hio.read_retention_times(rt) == [{"sample": "WT", "peptide": "PEP", "rt": 4.5}]


def test_table_row_shapes_are_named(tmp_path, caplog):
    """Rows of empty cells (Excel's ``,,,,,``, a blank line) are skipped and counted; a short
    row, an empty number and a charge written ``2+`` or ``2,0`` name the line and the column
    (Readers 16)."""
    p = tmp_path / "t.csv"
    p.write_text(HEAD + "a,S,AB,1,2,1\n,,,,,\n   \n\nb,S,CD,3,4,2\n")
    with caplog.at_level(logging.INFO):
        entries = hio.read_peptide_table(p)
    assert [e.file for e in entries] == ["a", "b"]
    assert "2 empty row(s) skipped" in caplog.text
    for row, message in [
        ("a,S,AB,1,2\n", "line 2: fewer cells than the header has columns"),
        ("a,S,AB,1,2,1,extra\n", "line 2: more cells than the header has columns"),
        ("a,S,AB,1,2,2+\n", r"line 2: charge must be a whole number, not '2\+'"),
        ("a,S,AB,1,2,\"2,0\"\n", "line 2: charge must be a whole number, not '2,0': use a "
                                "decimal point"),
        ("a,S,AB,,2,2\n", "line 2: start_seq is empty"),
        ("a,S,,1,2,2\n", "line 2: peptide is empty"),
    ]:
        p.write_text(HEAD + row)
        with pytest.raises(hio.ReadError, match=message):
            hio.read_peptide_table(p)
    rt = tmp_path / "rt.csv"
    for row, message in [("S,\n", "line 2: rt is empty"),
                         ("S,abc\n", "line 2: rt 'abc' is not a number of minutes"),
                         ("S\n", "line 2: fewer cells")]:
        rt.write_text("peptide,rt\n" + row)
        with pytest.raises(hio.ReadError, match=message):
            hio.read_retention_times(rt)


def test_table_headers_in_any_case_and_other_separators(tmp_path):
    """Header names are matched without regard to case or surrounding spaces (Excel
    capitalises); a table saved with semicolons or tabs is refused saying so, and a missing
    column is reported with the columns found (Readers 17)."""
    p = tmp_path / "t.csv"
    p.write_text("File,Sample ,PEPTIDE, start_seq,End_Seq,Charge,Index,Notes\n"
                 "a,S,AB,1,2,2,0,hello\n")
    (e,) = hio.read_peptide_table(p)
    assert (e.file, e.sample, e.peptide, e.charge) == ("a", "S", "AB", 2)
    assert e.extra == {"Notes": "hello"}  # an unknown column keeps its name; Index is dropped
    p.write_text("file;sample;peptide;start_seq;end_seq;charge\na;S;AB;1;2;2\n")
    with pytest.raises(hio.ReadError, match="separated by ';' .*save the table as CSV with "
                                            "commas"):
        hio.read_peptide_table(p)
    p.write_text("file\tsample\tpeptide\tstart_seq\tend_seq\tcharge\na\tS\tAB\t1\t2\t2\n")
    with pytest.raises(hio.ReadError, match="separated by tabs"):
        hio.read_peptide_table(p)
    p.write_text("file,sample,peptide,start seq,end_seq,charge\na,S,AB,1,2,2\n")
    with pytest.raises(hio.ReadError, match="missing column\\(s\\) start_seq; the columns found "
                                            "are file, sample, peptide, start seq, end_seq, "
                                            "charge"):
        hio.read_peptide_table(p)
    rt = tmp_path / "rt.csv"
    rt.write_text("Peptide;RT\nAB;3.5\n")
    with pytest.raises(hio.ReadError, match="separated by ';'"):
        hio.read_retention_times(rt)
    rt.write_text("Peptide,RT\nAB,3.5\n")
    assert hio.read_retention_times(rt) == [{"peptide": "AB", "rt": 3.5}]
    # a SpecExport file saved with semicolons says so too
    pep = tmp_path / "S" / "0001-0002-AB"
    pep.mkdir(parents=True)
    (pep / "60s-1-z2.csv").write_text("300.5;7\n300.6;8\n")
    entry = PeptideEntry("0001-0002-AB", "S", "AB", "0001-0002", 1, 2, 2)
    with pytest.raises(hio.ReadError, match="line 1: expected two columns.*separated by ';'"):
        hio.read_specexport_dir(pep, entry)


def test_tables_that_are_not_utf8(tmp_path):
    """A Latin-1 table says it is not UTF-8 and how to save it; Excel's "Unicode text"
    (UTF-16 with a byte-order mark) is read (Readers 31)."""
    p = tmp_path / "t.csv"
    p.write_bytes(HEAD.encode() + b"a,S\xe9,AB,1,2,2\n")
    with pytest.raises(hio.ReadError, match="t.csv: the file is not UTF-8 text .*CSV UTF-8"):
        hio.read_peptide_table(p)
    for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
        text = HEAD + "a,Sé,AB,1,2,2\n"
        p.write_bytes(text.encode(encoding) if encoding == "utf-16"
                      else (b"\xff\xfe" if encoding.endswith("le") else b"\xfe\xff")
                      + text.encode(encoding))
        (e,) = hio.read_peptide_table(p)
        assert e.sample == "Sé", encoding
    fasta = tmp_path / "s.fasta"
    fasta.write_bytes(">s\nMKV\n".encode("utf-16"))
    assert hio.read_fasta_sequence(fasta) == "MKV"
    fasta.write_bytes(b">s\xe9\nMKV\n")
    with pytest.raises(hio.ReadError, match="not UTF-8"):
        hio.read_fasta_sequence(fasta)
    hx = tmp_path / "A_0001-0005-ABCDE-z1-x.csv"
    hx.write_bytes("undeut,\n500.1,2\n".encode("utf-16"))
    assert len(hio.read_hxexpress_file(hx, ENTRY)) == 1


def test_a_missing_data_dir_is_named(tmp_path):
    """A data directory that does not exist (or is a file) is refused naming it, before any
    file is opened, by the scan and the read of every folder format (Readers 25)."""
    for fmt in ("hxexpress", "specexport", "pooled"):
        with pytest.raises(hio.ReadError, match="input.data_dir .*nowhere: no such directory"):
            hio.scan(tmp_path / "nowhere", fmt)
        with pytest.raises(hio.ReadError, match="no such directory"):
            hio.read_spectra([ENTRY], tmp_path / "nowhere", fmt, project="p")
    (tmp_path / "afile").write_text("x")
    with pytest.raises(hio.ReadError, match="a file, not a directory"):
        hio.scan(tmp_path / "afile", "specexport")


def test_record_paths_name_of_the_working_directory(tmp_path, monkeypatch, hxdir):
    """``record_paths = "name"`` with ``data_dir = "."`` records the directory's real name,
    not an empty string (Readers 30)."""
    monkeypatch.chdir(hxdir)
    s = hio.read_spectra(hio.scan(".", "hxexpress"), ".", "hxexpress", project="p",
                         record_paths="name")
    assert s.source["data_dir"] == hxdir.name


@pytest.mark.parametrize("name, message", [
    ("Angio_2_HI.xlsx", "expected at least five '-' separated fields, found 1"),
    ("S1-8-PEP-z2-text.xlsx", "the first field 'S1' needs SAMPLE_START"),
    ("S_Ac-8-PEP-z2-text.xlsx", "START 'Ac' is not a residue number"),
    ("S_1-x-PEP-z2-text.xlsx", "END 'x' is not a residue number"),
    ("S_1-8-PEP-2-text.xlsx", "the charge field '2' must be 'z' and the charge"),
    ("S_1-8-PEP-z2-run-1.xlsx", "the last field \\(free text\\) cannot contain '-'"),
    ("S_1-8--z2-text.xlsx", "the PEPTIDE field is empty"),
])
def test_hxexpress_filename_errors_name_the_field(name, message):
    """Each field of an HX-Express file name is tested and the wrong one named (Readers 26)."""
    with pytest.raises(hio.ReadError, match=f"file name {name!r}.*{message}"):
        hio.parse_hxexpress_filename(name)


@pytest.mark.parametrize("label", ["FD", "Full", "60s", 60, 60.0])
def test_hxexpress_labels_name_the_control_words(label):
    """A label that is not a time names the control words, so a user who wrote FD learns to
    write TD (Readers 27)."""
    with pytest.raises(hio.ReadError, match="controls are labelled undeut and TD"):
        hio.parse_hxexpress_label(label)


def test_a_note_below_the_data_of_a_sheet_is_ignored(tmp_path, caplog):
    """Text under the columns (a remark at the bottom of the sheet) ends the data; text with
    numbers below it is still refused (Readers 28)."""
    header = ["undeut", None, "60 s", None]
    body = [[500.1, 10.0, 500.1, 5.0], [500.2, 20.0, 500.2, 15.0], [None, None, None, None],
            ["end of data", None, None, None], [None, "see notes", None, None]]
    write_xlsx(tmp_path / "n.xlsx", header=header, body=body)
    with caplog.at_level(logging.DEBUG, logger="hxex.io.hxexpress"):
        sp = hio.read_hxexpress_file(tmp_path / "n.xlsx", ENTRY)
    assert [len(s.mz) for s in sp] == [2, 2]
    assert "the text from row 5 down is ignored" in caplog.text
    write_hx_csv(tmp_path / "n.csv", header=header, body=body)
    assert [len(s.mz) for s in hio.read_hxexpress_file(tmp_path / "n.csv", ENTRY)] == [2, 2]
    inside = [[500.1, 10.0, 500.1, 5.0], ["oops", 20.0, 500.2, 15.0], [500.3, 30.0, None, None]]
    write_xlsx(tmp_path / "i.xlsx", header=header, body=inside)
    with pytest.raises(hio.ReadError, match="'oops' is not a number"):
        hio.read_hxexpress_file(tmp_path / "i.xlsx", ENTRY)
    first = [["text", None, None, None], [500.1, 10.0, 500.1, 5.0]]  # text above the numbers
    write_xlsx(tmp_path / "f.xlsx", header=header, body=first)
    with pytest.raises(hio.ReadError, match="'text' is not a number"):
        hio.read_hxexpress_file(tmp_path / "f.xlsx", ENTRY)


def test_data_on_a_second_worksheet_are_pointed_out(tmp_path):
    """An empty first sheet with the data on another says so (Readers 29); an empty workbook
    still says it is empty."""
    wb = Workbook()
    wb.active.title = "Cover"
    data = wb.create_sheet("Data")
    data.append(["undeut", None])
    data.append([500.1, 10.0])
    wb.save(tmp_path / "s.xlsx")
    with pytest.raises(hio.ReadError, match="first worksheet is empty and 'Data' holds data"):
        hio.read_hxexpress_file(tmp_path / "s.xlsx", ENTRY)
    from hxex.io.hxexpress import read_hxexpress_header

    with pytest.raises(hio.ReadError, match="first worksheet is empty"):
        read_hxexpress_header(tmp_path / "s.xlsx")
    Workbook().save(tmp_path / "e.xlsx")
    with pytest.raises(hio.ReadError, match="no time labels|file is empty"):
        hio.read_hxexpress_file(tmp_path / "e.xlsx", ENTRY)


# --- pooled identities (docs/ANALYSIS.md "Pooled samples: possible identities") ------------

CANDIDATES = ">ALPHA1 one\nMKVLGGGACK\n>ALPHA2\nMKVLSSGACK\n>BETA1\nWWMKVLPP\n>OTHER9\nQQQ\n"


def test_a_pooled_peptide_is_identified_as_far_as_its_sequence_allows(tmp_path):
    from hxex.io.pooled import candidates_record, identify

    (tmp_path / "pool.fasta").write_text(CANDIDATES)
    record = candidates_record(tmp_path / "pool.fasta", ["ALPHA", "BETA"])
    one = identify("GGGA", 5, record)  # only ALPHA1
    assert (one.identity, one.level, one.candidates, one.offsets) == (
        "ALPHA1", "isoform", ("ALPHA1",), (0,))
    both = identify("GACacK", 7, record)  # modX ignored; both ALPHA records
    assert (both.identity, both.level, both.classes) == ("ALPHA", "class", ("ALPHA",))
    mixed = identify("MKVL", 1, record)  # two ALPHA and one BETA: no one class
    assert (mixed.identity, mixed.level, mixed.candidates) == (
        "ALPHA;BETA", "ambiguous", ("ALPHA1", "ALPHA2", "BETA1"))
    assert mixed.offsets == (0, 0, 2) and mixed.offset_text == "BETA1:+2"
    assert identify("YYY", 1, record).level == "none"
    unclassed = identify("QQQ", 1, record)  # one record, outside every class
    assert (unclassed.identity, unclassed.level) == ("OTHER9", "isoform")
    no_classes = identify("GACK", 7, candidates_record(tmp_path / "pool.fasta"))
    assert (no_classes.identity, no_classes.level) == ("", "ambiguous")


def test_a_class_can_name_its_records_when_its_name_is_not_in_their_ids(tmp_path, caplog):
    """[input.pooled_classes] names each class's records (owner, 2026-11-09: a class's name in
    the record ids is a coincidence of one FASTA, not a rule)."""
    from hxex.io.pooled import candidates_record, identify, record_classes

    (tmp_path / "pool.fasta").write_text(CANDIDATES)
    record = candidates_record(tmp_path / "pool.fasta",
                               {"first": ["ALPHA1", "ALPHA2"], "second": "BETA1"})
    assert record["classes"] == ["first", "second"]
    assert record["class_members"] == {"first": ["ALPHA1", "ALPHA2"], "second": ["BETA1"]}
    both = identify("GACK", 7, record)
    assert (both.identity, both.level, both.classes) == ("first", "class", ("first",))
    mixed = identify("MKVL", 1, record)
    assert (mixed.identity, mixed.level) == ("first;second", "ambiguous")
    assert record_classes(record, "OTHER9") == set()  # a record may be in no class
    # the list form keeps pyHXExpress's rule, text in the ids, and says when a name is in none
    with caplog.at_level("WARNING"):
        listed = candidates_record(tmp_path / "pool.fasta", ["ALPHA", "GAMMA"])
    assert listed["class_members"] == {"ALPHA": ["ALPHA1", "ALPHA2"], "GAMMA": []}
    assert "'GAMMA' is in no record id of pool.fasta" in caplog.text
    # a record kept before the members were stored: its classes from the text in the ids
    old = {k: v for k, v in listed.items() if k != "class_members"}
    assert record_classes(old, "ALPHA2") == {"ALPHA"}
    with pytest.raises(hio.ReadError, match="names record.s. ALPHA3 that pool.fasta does not"):
        candidates_record(tmp_path / "pool.fasta", {"first": ["ALPHA1", "ALPHA3"]})
    with pytest.raises(hio.ReadError, match="class.es. empty name no record"):
        candidates_record(tmp_path / "pool.fasta", {"empty": []})


def test_pooled_candidates_are_refused_when_they_cannot_name_one_record(tmp_path):
    from hxex.io.pooled import candidates_record

    (tmp_path / "twice.fasta").write_text(">A1\nMKV\n>A1 again\nMKL\n")
    with pytest.raises(hio.ReadError, match="record id.s. given twice: A1"):
        candidates_record(tmp_path / "twice.fasta")
    (tmp_path / "pool.fasta").write_text(CANDIDATES)
    with pytest.raises(hio.ReadError, match="each class name must be given once"):
        candidates_record(tmp_path / "pool.fasta", ["ALPHA", "ALPHA"])


def test_pooled_fasta_is_a_pooled_setting_and_lands_in_the_read_record(statedir):
    from hxex.config import ProjectError, load_project, read_input

    (statedir / "pool.fasta").write_text(">A1\nMKVLGGG\n>B1\nMKVLAA\n")
    toml = statedir.parent / "hxex.toml"
    toml.write_text('project = "p"\n[input]\nformat = "pooled"\ndata_dir = "states"\n'
                    'pooled_fasta = "states/pool.fasta"\npooled_classes = ["A", "B"]\n')
    s = read_input(load_project(toml))
    record = s.source["pooled_candidates"]
    assert record["file"] == "pool.fasta" and record["classes"] == ["A", "B"]
    assert record["records"] == [["A1", "MKVLGGG"], ["B1", "MKVLAA"]]
    toml.write_text('project = "p"\n[input]\nformat = "pooled"\ndata_dir = "states"\n'
                    'pooled_fasta = "states/pool.fasta"\n'
                    '[input.pooled_classes]\nfirst = ["A1"]\nsecond = ["B1"]\n')
    project = load_project(toml)
    assert project.input.pooled_classes == {"first": ("A1",), "second": ("B1",)}
    record = read_input(project).source["pooled_candidates"]
    assert record["class_members"] == {"first": ["A1"], "second": ["B1"]}
    toml.write_text('project = "p"\n[input]\nformat = "pooled"\ndata_dir = "states"\n'
                    'pooled_classes = ["A"]\n')
    with pytest.raises(ProjectError, match="pooled_classes needs input.pooled_fasta"):
        load_project(toml)
    toml.write_text('project = "p"\n[input]\nformat = "specexport"\ndata_dir = "states"\n'
                    'pooled_fasta = "states/pool.fasta"\n')
    with pytest.raises(ProjectError, match="pooled_fasta are for pooled data, not specexport"):
        load_project(toml)


def test_pooled_tables_carry_each_series_identity(statedir):
    """series_identities and the identity columns, from the read record's candidates; other
    results get no such columns."""
    from types import SimpleNamespace

    from hxex.io.pooled import candidates_record
    from hxex.report.tables import _with_identities, series_identities

    (statedir / "pool.fasta").write_text(">A1\nMKVLGGG\n>A2\nMKVLSS\n>B1\nPPGGG\n")
    record = candidates_record(statedir / "pool.fasta", ["A", "B"])
    binned = SimpleNamespace(spectrum_id=np.array(["s1", "s2", "s3"]),
                             peptide=np.array(["MKVL", "MKVL", "GGG"]),
                             start_seq=np.array([1, 1, 5]))
    fit = SimpleNamespace(spectrum_id=np.array(["s1", "s2", "s3"]),
                          data_id=np.array([0, 0, 1]))
    settings = SimpleNamespace(spectra_source={"reader": "pooled",
                                               "pooled_candidates": record},
                               registry=hio_registry())
    results = SimpleNamespace(binned=binned, fit=fit, settings=settings)
    known = series_identities(results)
    assert known[0].identity == "A" and known[1].level == "ambiguous"
    cols = _with_identities({"data_id": np.array([0, 1]), "charge": np.array([2, 2]),
                             "mean": np.array([1.0, 2.0])}, results, [0, 1], after="charge")
    assert list(cols) == ["data_id", "charge", "identity", "identity_level", "candidates",
                          "classes", "position_offset", "mean"]
    assert list(cols["candidates"]) == ["A1;A2", "A1;B1"]
    assert list(cols["position_offset"]) == ["", "B1:-2"]
    plain = SimpleNamespace(binned=binned, fit=fit,
                            settings=SimpleNamespace(spectra_source={"reader": "pooled"}))
    assert series_identities(plain) is None
    assert list(_with_identities({"charge": np.array([2])}, plain, [0], after="charge")) == [
        "charge"]


def hio_registry():
    from hxex.chem import DEFAULT_MODIFICATIONS

    return DEFAULT_MODIFICATIONS


def test_hxex_check_counts_pooled_identities(statedir, capsys):
    from hxex import cli

    (statedir / "pool.fasta").write_text(">A1\nMKVLGGG\n>B1\nWWW\n")
    toml = statedir.parent / "hxex.toml"
    toml.write_text('project = "p"\n[input]\nformat = "pooled"\ndata_dir = "states"\n'
                    'pooled_fasta = "states/pool.fasta"\npooled_classes = ["A", "B"]\n')
    assert cli.main(["check", str(toml)]) == 0
    out = capsys.readouterr().out
    assert ("identities     2 peptide sequences against pool.fasta (2 records; classes A, B): "
            "2 isoform, 0 class, 0 ambiguous, 0 none") in out
    assert "numbered differently in a candidate: 1: GGG (10)" in out
    # both peptides fit A1 only: one form, holding both as its own
    assert ("forms          1, the fewest candidates that explain every peptide (own peptides "
            "in brackets): A1 (2)") in out
