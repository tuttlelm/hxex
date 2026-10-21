# SPDX-License-Identifier: GPL-3.0-or-later
"""The user review table: review decisions made in a spreadsheet copy of spectra.csv, read
back, applied in memory, and the chosen K bootstrapped so its errors compare with the others'."""

import csv

import pytest

from hxex import cli
from hxex.io import open_bundle
from hxex.review_table import read_review_table
from test_cli import mzml_project  # noqa: I001


def test_the_table_is_read_as_spreadsheets_write_it(tmp_path):
    """Only spectrum_id, review_status, review_K and note are read; a BOM, a semicolon separator,
    capitals and "2.0" are accepted; rows without a decision are left out; a bad value names its
    row."""
    path = tmp_path / "t.csv"
    path.write_text("﻿spectrum_id;sample;review_status;review_K;note;mean_1\n"
                    "p:a;WT;Rejected;;bad injection;3.1\n"
                    "p:b;WT;;2.0;;4.0\n"
                    "p:c;WT;unreviewed;;just looked;5.0\n"
                    "p:d;WT;accepted;1;;5.5\n", encoding="utf-8")
    sheet = read_review_table(path)
    assert sheet.spectrum_id == ("p:a", "p:b", "p:d")
    assert sheet.status == ("rejected", "", "accepted") and sheet.k == (0, 2, 1)
    assert sheet.note[0] == "bad injection"
    assert sheet.describe() == "3 spectra: 1 rejected, 1 accepted, 2 with a chosen K"
    for text, message in [
        ("spectrum_id,review_K\np:a,two\n", "row 2: review_K 'two'"),
        ("spectrum_id,review_K\np:a,1.5\n", "row 2: review_K '1.5'"),
        ("spectrum_id,review_status\np:a,maybe\n", "row 2: review_status 'maybe'"),
        ("spectrum_id,review_K\np:a,2\np:a,3\n", "row 3: p:a is already in row 2"),
        ("sample,review_K\nWT,2\n", "no spectrum_id column"),
        ("spectrum_id,sample\np:a,WT\n", "neither a review_status nor a review_K"),
    ]:
        path.write_text(text)
        with pytest.raises(ValueError, match=message):
            read_review_table(path)


def _edit(spectra_csv, out, changes):
    """A copy of spectra.csv with ``changes`` (row number -> {column: value}) made."""
    with open(spectra_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    for i, values in changes.items():
        rows[i].update(values)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def test_a_review_table_from_spectra_csv_end_to_end(tmp_path, capsys):
    """Copy spectra.csv, choose a K and reject a spectrum: hxex report refuses the chosen K until
    hxex resume bootstraps it (only it); then the tables show the choices with errors; a new run
    bootstraps the chosen K itself; the analysis honours the rejection."""
    path = mzml_project(tmp_path)
    path.write_text(path.read_text().replace("k_max = 1", "k_max = 2"))
    assert cli.main(["-q", "run", str(path)]) == 0
    tables = tmp_path / "out_tables"
    with open(tables / "spectra.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    one = next(i for i, r in enumerate(rows) if r["K"] == "1" and r["role"] == "")
    other = next(i for i, r in enumerate(rows) if i != one and r["role"] == "")
    chosen, rejected = rows[one]["spectrum_id"], rows[other]["spectrum_id"]
    _edit(tables / "spectra.csv", tmp_path / "user_review_table.csv",
          {one: {"review_K": "2", "note": "two populations"},
           other: {"review_status": "rejected"}})
    path.write_text(path.read_text() + '\n[analysis]\nks = [1]\noverlay = ["avg"]\n'
                    '\n[review]\ntable = "user_review_table.csv"\n')
    cli.main(["check", str(path)])  # (a problem: the bundle exists, for hxex run)
    assert "2 spectra: 1 rejected, 0 accepted, 1 with a chosen K" in capsys.readouterr().out
    # the chosen K has no bootstrap yet: refused, with the way to add it
    assert cli.main(["-q", "report", str(path)]) == 1
    assert "hxex resume" in capsys.readouterr().err
    with open_bundle(tmp_path / "out") as b:
        boot_before = b.bootstrap().mean_se.copy()
    assert cli.main(["-q", "resume", str(path)]) == 0
    with open_bundle(tmp_path / "out") as b:
        fit = b.fit()
        i = list(map(str, fit.spectrum_id)).index(chosen)
        two = b.bootstrap(2)
        assert two.covers(i) and int(two.n_success.astype(bool).sum()) == 1  # only that one
        import numpy as np

        assert np.array_equal(b.bootstrap().mean_se, boot_before, equal_nan=True)  # kept
    assert cli.main(["-q", "report", str(path)]) == 0
    with open(tables / "spectra.csv", newline="") as f:
        out = {r["spectrum_id"]: r for r in csv.DictReader(f)}
    assert out[chosen]["review_K"] == "2" and out[chosen]["K"] == "2"
    assert out[rejected]["review_status"] == "rejected"
    with open(tables / "review.csv", newline="") as f:
        listed = {r["spectrum_id"]: r for r in csv.DictReader(f)}
    assert {listed[chosen]["source"], listed[rejected]["source"]} == {"review table"}
    assert listed[chosen]["note"] == "two populations"
    with open(tables / "components.csv", newline="") as f:
        comp = [r for r in csv.DictReader(f) if r["spectrum_id"] == chosen]
    assert len(comp) == 2 and all(r["mean_se"] not in ("", "nan") for r in comp)
    # nothing more to add: said, not an error
    assert cli.main(["resume", str(path)]) == 0
    assert "nothing to add" in capsys.readouterr().out
    # a new run bootstraps the chosen K itself
    assert cli.main(["-q", "run", str(path), "--overwrite"]) == 0
    with open_bundle(tmp_path / "out") as b:
        assert b.bootstrap(2).covers(list(map(str, b.fit().spectrum_id)).index(chosen))
    assert cli.main(["-q", "report", str(path)]) == 0
    # the analysis leaves the rejected spectrum out (and has no use for a chosen K)
    capsys.readouterr()
    assert cli.main(["analyse", str(path)]) == 0
    assert "user_review_table.csv: 2 spectra: 1 rejected" in capsys.readouterr().err
    # a K the run did not fit is refused, naming the spectrum
    _edit(tables / "spectra.csv", tmp_path / "user_review_table.csv", {one: {"review_K": "5"}})
    assert cli.main(["-q", "report", str(path)]) == 1
    assert "K that was not fitted" in capsys.readouterr().err


def test_spectra_csv_has_the_columns_the_review_table_reads(tmp_path):
    """A copy of spectra.csv is the user review table: it carries every column read back,
    and a note given in the review comes back in it."""
    from hxex.io.bundle import ReviewTable
    from hxex.report import Results, spectra_table
    from hxex.run import run
    from test_run import config, raw_spectra

    res = run(raw_spectra(n_series=1), config())
    results = Results.from_run(res)
    cols = spectra_table(results)
    assert {"spectrum_id", "review_status", "review_K", "note"} <= set(cols)
    sid = str(results.fit.spectrum_id[3])
    noted = results.with_review(ReviewTable.empty().update(sid, status="rejected",
                                                           note="weak spectrum"))
    note = spectra_table(noted)["note"]
    assert note[3] == "weak spectrum" and set(note[:3]) == {""}
