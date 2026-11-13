# SPDX-License-Identifier: GPL-3.0-or-later
"""Exporting a series (docs/PLAN.md Next steps 32): the HX-Express spreadsheet read back by
hxex's own reader, and the plot data agreeing with what the figures draw."""

import csv
import logging

import numpy as np
import pytest

from hxex.io import ReviewTable
from hxex.io.hxexpress import (
    hxexpress_control_label,
    hxexpress_filename,
    hxexpress_time_label,
    parse_hxexpress_filename,
    read_hxexpress_file,
)
from hxex.io.times import parse_hxexpress_label
from hxex.pipeline import run
from hxex.report import Results, spectrum_data
from hxex.report.export import (
    ExportError,
    export_hxexpress,
    hxexpress_name,
    hxexpress_spectra,
    plain_sequence,
    plot_data,
    plot_data_folder,
    write_plot_data,
)
from test_run import PEPTIDE, boot_config, config, raw_spectra


@pytest.fixture(scope="module")
def results():
    raw = raw_spectra(n_series=2)
    res = run(raw, config(bootstrap=boot_config(ks="all", keep_draws=False)))
    return Results.from_run(res, spectra=raw)


def _read(path):
    entry = parse_hxexpress_filename(path.name)
    return entry, read_hxexpress_file(path, entry)


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- labels and names ----------------------------------------------------------------------

@pytest.mark.parametrize(("seconds", "rep", "label"), [
    (30, 1, "30 sec"), (30, 2, "30.01 sec"), (30, 3, "30.02 sec"), (90, 2, "90.01 sec"),
    (60, 1, "1 min"), (1800, 2, "30.01 min"), (3600, 1, "1 h"), (72000, 4, "20.03 h"),
    (5400, 1, "90 min"), (0.5, 1, "500 ms"), (0.5, 2, "500.01 ms"), (1.5, 1, "1500 ms"),
    (0.1, 2, "100.01 ms"), (0, 1, "0 sec"), (0, 2, "0 sec"),
])
def test_time_labels_read_back_as_their_time(seconds, rep, label):
    """Replicates carry HX-Express's marker (the digits after the point) from the second on;
    a time that is not whole in seconds is written in ms, so no label reads as a fraction or
    as another time's replicate."""
    assert hxexpress_time_label(seconds, rep) == label
    assert parse_hxexpress_label(label) == (pytest.approx(seconds), "")


def test_times_that_cannot_be_labelled_are_refused():
    for bad in (-1.0, float("nan"), 0.0001):
        with pytest.raises(ValueError):
            hxexpress_time_label(bad)
    assert hxexpress_control_label("undeut") == "undeut"
    assert hxexpress_control_label("fulldeut", 2) == "TD2"
    assert parse_hxexpress_label("TD2")[1] == "fulldeut"


def test_the_file_name_is_the_one_the_reader_reads():
    name = hxexpress_filename("WT_day1", 12, 20, "PEPKTIDE", 3)
    assert name == "WT_day1_12-20-PEPKTIDE-z3-hxex.xlsx"
    entry = parse_hxexpress_filename(name)
    assert (entry.sample, entry.start_seq, entry.end_seq, entry.peptide, entry.charge) == (
        "WT_day1", 12, 20, "PEPKTIDE", 3)
    for bad in (("W-T", 1, 2, "PE", 1), ("WT", 1, 2, "PEPacK", 1), ("WT", 5, 2, "PE", 1)):
        with pytest.raises(ValueError):
            hxexpress_filename(*bad)


def test_sequences_are_written_in_one_letter_form():
    assert plain_sequence("Ac-PEPacKTIDE-NH2") == "PEPKTIDE"
    assert plain_sequence("PEpSTIDE") == "PESTIDE"
    with pytest.raises(ExportError, match="one-letter"):
        plain_sequence("PEPxyz")


# --- the HX-Express spreadsheet ------------------------------------------------------------

def test_a_series_round_trips_through_the_reader(results, tmp_path):
    """Every spectrum is written raw, in the order undeuterated, times, fully deuterated, and
    hxex's reader gives back the same spectra, times, roles, and replicates."""
    path = export_hxexpress(results, 0, tmp_path)
    assert path.name == f"S_1-10-{PEPTIDE}-z2-hxex.xlsx"
    entry, back = _read(path)
    assert [s.label for s in back] == ["undeut", "undeut2", "undeut3", "1 min", "1.01 min",
                                       "1.02 min", "TD", "TD2"]
    fit = results.fit
    rows = np.flatnonzero(fit.data_id == 0)
    assert len(back) == len(rows)
    by_point = {}
    for i in rows:
        key = (str(fit.role[i]), float(fit.time[i]) if not fit.role[i] else None,
               int(results.replicates[i]))
        by_point[key] = results.raw_spectrum(str(fit.spectrum_id[i]))
    for s in back:
        mz, y = by_point[(s.role, None if s.role else s.time, s.rep)]
        order = np.argsort(mz, kind="stable")
        np.testing.assert_allclose(s.mz, mz[order])
        np.testing.assert_allclose(s.intensity, y[order])


def test_picked_sticks_and_the_first_control_only(results, tmp_path):
    columns = hxexpress_spectra(results, 1, spectra="picked", controls="first")
    assert [c[0] for c in columns] == ["undeut", "1 min", "1.01 min", "1.02 min", "TD"]
    sid = str(results.fit.spectrum_id[np.flatnonzero(results.fit.data_id == 1)][3])
    picked = spectrum_data(sid, results, show=("picked",))["picked"]
    label, mz, y = columns[1]
    np.testing.assert_allclose(mz, picked["mz"])
    np.testing.assert_allclose(y, picked["intensity"])
    path = export_hxexpress(results, 1, tmp_path, spectra="picked", controls="first")
    assert [s.label for s in _read(path)[1]] == [c[0] for c in columns]


def test_rejected_spectra_are_left_out(results, tmp_path, caplog):
    rows = np.flatnonzero(results.fit.data_id == 0)
    gone = str(results.fit.spectrum_id[rows[4]])  # the second replicate at 60 s
    reviewed = results.with_review(ReviewTable.empty().update(gone, status="rejected"))
    with caplog.at_level(logging.INFO, logger="hxex.report.export"):
        labels = [c[0] for c in hxexpress_spectra(reviewed, 0)]
    assert labels == ["undeut", "undeut2", "undeut3", "1 min", "1.01 min", "TD", "TD2"]
    assert "1 rejected spectrum left out" in caplog.text


def test_existing_files_are_kept_unless_overwritten(results, tmp_path):
    path = export_hxexpress(results, 0, tmp_path)
    path.write_bytes(b"mine")
    with pytest.raises(FileExistsError, match="overwrite"):
        export_hxexpress(results, 0, tmp_path)
    assert path.read_bytes() == b"mine"
    export_hxexpress(results, 0, tmp_path, overwrite=True)
    assert len(_read(path)[1]) == 8


def test_a_prefix_and_a_suffix_name_the_file(results, tmp_path):
    name = hxexpress_name(results, 0, prefix="exp2", suffix="raw_v1")
    assert name == f"exp2_S_1-10-{PEPTIDE}-z2-hxex_raw_v1.xlsx"
    assert parse_hxexpress_filename(name).sample == "exp2_S"
    path = export_hxexpress(results, 0, tmp_path, prefix="exp2", suffix="raw_v1")
    assert path.name == name
    for bad in ("a-b", "a/b", "a:b"):
        with pytest.raises(ExportError, match="file name"):
            hxexpress_name(results, 0, suffix=bad)
    assert hxexpress_name(results, 0, suffix="a b").endswith("-hxex_a b.xlsx")
    # a folder name may hold '-': the rule is the HX-Express name's own
    folder = plot_data_folder(results, 0, prefix="talk-2026", suffix="v 2")
    assert folder.startswith("talk-2026_") and folder.endswith("_plot_data_v 2")
    with pytest.raises(ExportError, match="file name"):
        plot_data_folder(results, 0, suffix="a/b")


def test_points_without_real_times_take_their_rank(results, monkeypatch, caplog):
    """time_scale = "order": each point gets its rank in seconds as a dummy time (owner)."""
    monkeypatch.setattr(Results, "time_scale", property(lambda self: "order"))
    with caplog.at_level(logging.INFO, logger="hxex.report.export"):
        labels = [c[0] for c in hxexpress_spectra(results, 0)]
    assert labels[3:6] == ["1 sec", "1.01 sec", "1.02 sec"]
    assert "point 60 -> 1 sec" in caplog.text


def test_raw_spectra_missing_from_the_results_are_named(results):
    bare = Results(measurements=results.measurements, fit=results.fit, binned=results.binned,
                   settings=results.settings)
    with pytest.raises(ExportError, match="picked sticks"):
        hxexpress_spectra(bare, 0)
    assert len(hxexpress_spectra(bare, 0, spectra="picked")) == 8


# --- plot data -----------------------------------------------------------------------------

def test_plot_data_holds_what_the_spectrum_figure_draws(results):
    tables = plot_data(results, data_id=0)
    assert set(tables) == {"bins", "raw", "spectra", "components", "uptake", "about"}
    bins = tables["bins"]
    fit = results.fit
    rows = np.flatnonzero(fit.data_id == 0)
    assert len(tables["spectra"]["spectrum_id"]) == len(rows)
    assert bins["selected"].all() and (bins["k"] == bins["K"]).all()
    for i in rows:
        sid = str(fit.spectrum_id[i])
        drawn = spectrum_data(sid, results)
        mine = bins["spectrum_id"] == sid
        np.testing.assert_allclose(bins["fit"][mine], drawn["fit"]["total"])
        np.testing.assert_allclose(bins["picked"][mine], drawn["picked"]["intensity"])
        np.testing.assert_allclose(bins["residual"][mine],
                                   drawn["picked"]["intensity"] - drawn["fit"]["total"])
        for r in range(drawn["fit"]["k"]):
            np.testing.assert_allclose(bins[f"population_{r + 1}"][mine],
                                       drawn["fit"]["components"][r])
        np.testing.assert_allclose(bins["fit_band_low"][mine], drawn["fit"]["band"][0])
        raw = tables["raw"]["spectrum_id"] == sid
        np.testing.assert_allclose(tables["raw"]["intensity"][raw], drawn["raw"]["intensity"])
    assert set(tables["uptake"]["data_id"]) == {0}


def test_every_fitted_k_and_a_time_point_alone(results):
    fit = results.fit
    rows = np.flatnonzero(fit.data_id == 1)
    point = [str(fit.spectrum_id[i]) for i in rows if fit.time[i] == 60.0]
    tables = plot_data(results, spectrum_ids=point, ks="all")
    bins = tables["bins"]
    assert set(bins["spectrum_id"]) == set(point)
    assert set(bins["k"]) == {int(k) for k in fit.k_values}
    assert (bins["selected"] == (bins["k"] == bins["K"])).all()
    assert set(tables["uptake"]["time"]) == {60.0}
    with pytest.raises(ValueError, match="not both"):
        plot_data(results, data_id=1, spectrum_ids=point)


def test_plot_data_files_and_their_folder(results, tmp_path):
    folder = tmp_path / plot_data_folder(results, 0, prefix="talk", suffix="v2")
    assert folder.name.startswith("talk_") and folder.name.endswith("_plot_data_v2")
    paths = write_plot_data(results, folder, data_id=0)
    assert sorted(p.name for p in paths) == ["about.csv", "bins.csv", "components.csv",
                                             "raw.csv", "spectra.csv", "uptake.csv"]
    about = {r["item"]: r["value"] for r in _rows(folder / "about.csv")}
    assert about["k_written"] == "the K in effect for each spectrum"
    assert about["intensities"] == "as measured (unscaled)"
    assert len(_rows(folder / "bins.csv")) == len(plot_data(results, data_id=0)["bins"]["bin"])
    with pytest.raises(FileExistsError, match="overwrite"):
        write_plot_data(results, folder, data_id=0)
    write_plot_data(results, folder, data_id=0, overwrite=True)
