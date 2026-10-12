# SPDX-License-Identifier: GPL-3.0-or-later
"""Reader tests: synthetic files for the logic, and the real test sets when they are available.

The real-data tests read their location from HXEX_BIMODAL_DATA (the pyHXExpress
Bimodal_HDX_Data folder) and HXEX_SPECEXPORT_EXAMPLE (a SpecExport folder with its FASTA file,
plus ``hdxms_analysis_*/alldeutdata_*.csv`` and ``metadf_asrun_*.csv`` from a pyHXExpress run).
HXEX_POOLED_EXAMPLE is a pooled folder with All_tubs.fasta and a metadf_*.csv. They are
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
    "label, seconds",
    [
        ("undeut", 0.0), ("Undeut_1", 0.0), ("TD", 1e6), ("td", 1e6), ("60 s", 60.0),
        ("5 min", 300.0), ("2 h", 7200.0), ("1.2 min", 60.0), ("10.3 s", 10.0),
        ("0.5 min", 30.0), ("15 sec", 15.0), ("  3 min ", 180.0),
    ],
)
def test_hxexpress_time(label, seconds):
    assert hio.parse_hxexpress_time(label) == seconds


def test_hxexpress_fulldeut_time_is_user_set():
    assert hio.parse_hxexpress_time("TD", fulldeut_time=86400.0) == 86400.0


def test_decimal_times_are_exact():
    for token, expected in [("4.1m", 246.0), ("0.1m", 6.0), ("1.1h", 3960.0), ("0.3s", 0.3)]:
        assert hio.parse_specexport_time(token) == expected
    assert hio.parse_hxexpress_time("0.7 min") == 42.0
    assert hio.parse_specexport_time("4.1m") == hio.parse_specexport_time("246s")


@pytest.mark.parametrize(
    "label",
    ["60", "5 x", "nan min", "inf s", "abc min", ".5 min", "", 5, None, float("nan")],
)
def test_hxexpress_time_errors(label):
    with pytest.raises(hio.ReadError):
        hio.parse_hxexpress_time(label)


@pytest.mark.parametrize(
    "token, seconds",
    [("Non", 0.0), ("Full", 1e6), ("60s", 60.0), ("5m", 300.0), ("2h", 7200.0), ("1.5m", 90.0)],
)
def test_specexport_time(token, seconds):
    assert hio.parse_specexport_time(token) == seconds


def test_specexport_fulldeut_time_is_user_set():
    assert hio.parse_specexport_time("Full", fulldeut_time=5.0) == 5.0


@pytest.mark.parametrize("token", ["60x", "abc", "s", "", "nans", "infs"])
def test_specexport_time_errors(token):
    with pytest.raises(hio.ReadError):
        hio.parse_specexport_time(token)


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


@pytest.mark.parametrize("writer, name", [(write_xlsx, "a.xlsx"), (write_hx_csv, "a.csv")])
def test_read_hxexpress_file(tmp_path, writer, name):
    writer(tmp_path / name)
    sp = hio.read_hxexpress_file(tmp_path / name, ENTRY, data_id=7)
    assert [(s.label, s.time, s.rep) for s in sp] == [
        ("undeut", 0.0, 1), ("1 min", 60.0, 1), ("1.2 min", 60.0, 2), ("TD", 1e6, 1),
    ]
    assert all(s.data_id == 7 and s.entry is ENTRY for s in sp)
    np.testing.assert_array_equal(sp[0].mz, [500.1, 500.2, 500.3])  # sorted
    np.testing.assert_array_equal(sp[0].intensity, [20.0, 10.0, 30.0])
    np.testing.assert_array_equal(sp[1].mz, [500.1, 500.2, 500.3])  # row with empty m/z dropped
    np.testing.assert_array_equal(sp[2].mz, [500.2, 500.4])  # missing intensity dropped
    np.testing.assert_array_equal(sp[3].mz, [499.0, 500.0, 501.0])
    assert sp[0].mz.dtype == np.float64


def test_read_hxexpress_fulldeut_time(tmp_path):
    write_xlsx(tmp_path / "a.xlsx")
    sp = hio.read_hxexpress_file(tmp_path / "a.xlsx", ENTRY, fulldeut_time=123.0)
    assert sp[-1].time == 123.0


def test_the_undeuterated_label_time_is_user_set(tmp_path, caplog):
    assert hio.parse_hxexpress_time("undeut", undeut_time=-1.0) == -1.0
    assert hio.parse_specexport_time("Non", undeut_time=-1.0) == -1.0
    assert hio.parse_hxexpress_time("undeut") == 0.0 == hio.parse_specexport_time("Non")
    # a reverse experiment: a real 0 s time point next to the undeuterated control
    write_xlsx(tmp_path / "r.xlsx", header=["undeut", None, "0 s", None, "TD", None])
    with caplog.at_level(logging.WARNING, logger="hxex.io.hxexpress"):
        sp = hio.read_hxexpress_file(tmp_path / "r.xlsx", ENTRY)
    assert "cannot be told apart" in caplog.text and [s.time for s in sp][:2] == [0.0, 0.0]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="hxex.io.hxexpress"):
        sp = hio.read_hxexpress_file(tmp_path / "r.xlsx", ENTRY, undeut_time=-1.0)
    assert caplog.text == "" and [s.time for s in sp] == [-1.0, 0.0, 1e6]


def test_read_spectra_refuses_equal_control_times(specroot):
    with pytest.raises(ValueError, match="must differ"):
        hio.read_spectra(hio.scan(specroot, "specexport"), specroot, "specexport",
                         project="proj", undeut_time=5.0, fulldeut_time=5.0)
    s = hio.read_spectra(hio.scan(specroot, "specexport"), specroot, "specexport",
                         project="proj", undeut_time=-1.0)
    assert -1.0 in set(s.time.tolist()) and 0.0 not in set(s.time.tolist())


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


def test_scan_specexport_range_outside_fasta(specroot):
    (specroot / "S1" / "0004-0099-XXXX").mkdir()
    with pytest.raises(hio.ReadError, match="outside"):
        hio.scan_specexport(specroot)


def test_read_specexport_dir(specroot):
    e = hio.scan_specexport(specroot)[0]
    sp = hio.read_specexport_dir(specroot / "S1" / e.file, e, data_id=3, fulldeut_time=999.0)
    assert [(s.label, s.time, s.rep) for s in sp] == [
        ("Non-D-1-z2.csv", 0.0, 1), ("60s-1-z2.csv", 60.0, 1), ("60s-2-z2.csv", 60.0, 2),
        ("Full-D-1-z2.csv", 999.0, 1),
    ]
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
    s = hio.read_spectra(entries, specroot, "specexport", project="proj", fulldeut_time=500.0)
    assert len(s) == 5
    assert sorted(set(s.time)) == [0.0, 60.0, 240.0, 500.0]
    assert list(s.time_idx()[s.data_id == 0]) == [0, 1, 1, 2]


def test_empty_spectrum_set():
    s = SpectrumSet.from_spectra([])
    assert len(s) == 0 and len(s.mz) == 0 and len(s.time_idx()) == 0


# --- real data ------------------------------------------------------------------------------

BIMODAL = os.environ.get("HXEX_BIMODAL_DATA")
SPECEXPORT = os.environ.get("HXEX_SPECEXPORT_EXAMPLE")


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
        expected = np.array(ref[(str(s.file[i]), float(s.time[i]), int(s.rep[i]))])
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
         float(s.time[i]), int(s.rep[i]))
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
        ">TBA1A alpha one\nAAMKVLAA\n>TBB2A\nAAAAMKVLGGGAC\n>HSP\nWWACKDEWW\n"
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


def test_match_candidates(pooldir):
    entries = hio.scan_specexport_pooled(pooldir)
    m = hio.match_candidates(entries, pooldir / "pooled.fasta", classes=["TBA", "TBB", "HSP"])
    by_file = {x.entry.file: x for x in m if x.entry.charge == 2}
    assert by_file["0001-0004-MKVL"].matches == ("TBA1A", "TBB2A")  # FASTA order
    assert by_file["0001-0004-MKVL"].classes == ("TBA", "TBB")
    assert by_file["0010-0014-ACacKDE"].matches == ("HSP",)  # acK matches K
    assert by_file["0010-0014-ACacKDE"].classes == ("HSP",)
    assert by_file["0020-0022-GGG"].matches == ("TBB2A",)
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
    p.write_text(">TBA1A alpha one\nMK\nVL\n\n>empty\n>TBB2A\nGG G\n")
    assert hio.read_fasta_records(p) == [("TBA1A", "MKVL"), ("TBB2A", "GGG")]


POOLED = os.environ.get("HXEX_POOLED_EXAMPLE")


@pytest.mark.skipif(not POOLED, reason="HXEX_POOLED_EXAMPLE not set")
def test_pooled_example_matches_pyhxexpress_metadf():
    """A pooled folder with All_tubs.fasta and a metadf_*.csv written by pyHXExpress's
    get_metadf_pooled and categorize_pooled_peptides (classes TBA and TBB)."""
    root = Path(POOLED)
    ref = {
        (r["file"], int(float(r["charge"]))): (r["sample_matches"], r["tub_class"])
        for r in csv.DictReader(next(root.glob("metadf_*.csv")).open())
    }
    entries = hio.scan(root, "pooled")
    found = hio.match_candidates(entries, root / "All_tubs.fasta", classes=["TBA", "TBB"])
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
        again = hio.read_spectra(entries[::-1], root, fmt, project="proj", fulldeut_time=7.0)
        assert set(again.spectrum_id) == set(s.spectrum_id)  # independent of order and times
        i = s.find(str(s.spectrum_id[3]))
        assert i == 3
    with pytest.raises(KeyError):
        s.find("nope")


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
