# SPDX-License-Identifier: GPL-3.0-or-later
"""Review decisions that follow their spectrum when a rerun sums other scans (docs/PLAN.md Next
steps 27): a spectrum id names its scans, so a new retention window gives it a new id; an entry
naming scans the results no longer sum is matched by sample, file, and ion (fixed-time runs
only, exactly one spectrum), in memory for the user review table and stored for the bundle's
own review layer. Synthetic LC runs of one sample (test_elution_windows)."""

import logging

import numpy as np

from hxex.cli import main
from hxex.config import load_project, read_input
from hxex.io import kinetics_spectra, open_bundle
from hxex.io.bundle import ReviewTable
from hxex.pipeline import _review_for
from hxex.review_table import (
    apply_review_table,
    move_entries,
    moved_choices,
    read_review_table,
    scan_stem,
)
from test_cli import FAST_RUN, mzml_project
from test_elution_windows import ION, RT, _by_file, _project
from test_window_edit import _results

A = "p:S/a.mzML/0001-0008-VDQIIMAK-z2"
B = "p:S/b.mzML/0001-0008-VDQIIMAK-z2"


def test_the_stem_is_the_id_without_its_scans():
    assert scan_stem(f"{A}/scans120-131") == A
    assert scan_stem(f"{A}/scan120") == A
    assert scan_stem("p:S/0001-0008-VDQIIMAK-z2/10s/rep1") is None  # names no scans


def test_an_entry_follows_its_one_spectrum():
    moves = move_entries([f"{A}/scans1-5", f"{B}/scans3-9"],
                         [f"{A}/scans2-6", f"{B}/scans3-9"])
    assert moves.moved == {f"{A}/scans1-5": f"{A}/scans2-6"}  # B's id is current: kept
    assert not moves.several and not moves.superseded


def test_an_entry_of_a_kinetics_run_never_moves():
    # every scan group of a kinetics run is its own time point, named by its scans
    moves = move_entries([f"{A}/scans1-5"], [f"{A}/scans2-6"], kinetics=[True])
    assert not moves


def test_entries_matching_several_or_one_taken_are_left_as_they_were():
    current = [f"{A}/scans2-6", f"{A}/scans7-9", f"{B}/scans1-4"]
    moves = move_entries([f"{A}/scans1-5", f"{B}/scans1-3", f"{B}/scans2-5"], current)
    assert moves.several[f"{A}/scans1-5"] == (f"{A}/scans2-6", f"{A}/scans7-9")
    # two old entries for one spectrum: neither is chosen over the other
    assert moves.several[f"{B}/scans1-3"] == moves.several[f"{B}/scans2-5"] == (
        f"{B}/scans1-4",)
    assert not moves.moved


def test_the_new_spectrums_own_entry_wins():
    moves = move_entries([f"{A}/scans1-5", f"{A}/scans2-6"], [f"{A}/scans2-6"])
    assert moves.superseded == {f"{A}/scans1-5": f"{A}/scans2-6"}
    assert not moves.moved


def test_a_kinetics_run_is_told_by_its_read_record(tmp_path):
    spectra = read_input(load_project(mzml_project(tmp_path)))
    ids = list(spectra.spectrum_id)
    assert all(kinetics_spectra(spectra.source, ids))
    old = scan_stem(ids[0]) + "/scans999-1000"
    assert not move_entries([old], ids, kinetics_spectra(spectra.source, ids))


def _rerun_with_window(tmp_path):
    """The two runs read and fitted, then read and fitted again with un1's window narrowed (an
    rt_elution_windows row): the old and new results, and un1's old and new ids."""
    path = _project(tmp_path)
    before = read_input(load_project(path))
    old = _results(before)
    _project(tmp_path, windows="sample,file,ion,rt_start,rt_end\n"
                              f"S,un1.mzML,{ION},{RT - 2 / 60:.4f},{RT + 2 / 60:.4f}\n")
    after = read_input(load_project(path))
    new = _results(after)
    old_id = str(before.spectrum_id[_by_file(before)["un1.mzML"]])
    new_id = str(after.spectrum_id[_by_file(after)["un1.mzML"]])
    assert old_id != new_id and scan_stem(old_id) == scan_stem(new_id)
    return path, old, new, old_id, new_id, after


def test_a_review_table_entry_follows_a_new_window(tmp_path, caplog):
    path, _, new, old_id, new_id, _ = _rerun_with_window(tmp_path)
    table = tmp_path / "user_review_table.csv"
    table.write_text("spectrum_id,review_status,review_K,note\n"
                     f"{old_id},rejected,,a neighbour's peak\n")
    with caplog.at_level(logging.INFO, logger="hxex.review_table"):
        shown = apply_review_table(new, read_review_table(table))
    assert "1 moved to their spectrum's new scans" in caplog.text
    assert "do not have" not in caplog.text  # moved, so not skipped
    entry = shown.review.get(new_id)
    assert entry["status"] == "rejected" and entry["note"] == "a neighbour's peak"
    assert shown.review.get(old_id) is None
    assert table.read_text().count(old_id) == 1  # the file is left as it is
    j = list(shown.review.spectrum_id).index(new_id)
    assert shown.review.source[j] == f"review table (moved from {old_id})"


def test_the_bundles_review_layer_is_stored_under_the_new_id(tmp_path):
    _, _, new, old_id, new_id, after = _rerun_with_window(tmp_path)
    review = ReviewTable.empty().update(old_id, status="rejected", note="checked")
    carried = _review_for(review, new.fit, after.source)
    assert list(carried.spectrum_id) == [new_id]
    assert not carried.stale.any()
    assert carried.get(new_id)["note"] == "checked"
    # without the move it would have gone stale (HISTORY 96)
    assert review.mark_stale(new.fit).stale.all()


def test_a_chosen_k_follows_its_spectrum_into_the_run(tmp_path):
    _, _, new, old_id, new_id, after = _rerun_with_window(tmp_path)
    assert moved_choices({old_id: 2}, new.fit.spectrum_id, after.source) == {new_id: 2}


def _un1(bundle):
    with open_bundle(bundle) as b:
        return next(str(s) for s in b.spectrum_ids() if "/un1.mzML/" in str(s))


def test_check_and_report_after_a_new_window(tmp_path, capsys):
    path = _project(tmp_path)
    text = path.read_text().replace("[input]", '[review]\ntable = "user_review_table.csv"\n\n'
                                    '[output]\nbundle = "results"\n\n[input]', 1)
    path.write_text(text + FAST_RUN)
    table = tmp_path / "user_review_table.csv"
    table.write_text("spectrum_id,review_status\n")
    assert main(["run", str(path)]) == 0
    old_id = _un1(tmp_path / "results")
    table.write_text(f"spectrum_id,review_status,note\n{old_id},rejected,looked at\n")
    windows = (tmp_path / "rt_elution_windows.csv")
    windows.write_text("sample,file,ion,rt_start,rt_end\n"
                       f"S,un1.mzML,{ION},{RT - 2 / 60:.4f},{RT + 2 / 60:.4f}\n")
    path.write_text(path.read_text().replace(
        "[input]\n", '[input]\nrt_elution_windows = "rt_elution_windows.csv"\n', 1))
    assert main(["run", str(path), "--overwrite"]) == 0
    new_id = _un1(tmp_path / "results")
    assert new_id != old_id
    capsys.readouterr()
    assert main(["check", str(path)]) == 0  # a moved row is not a problem
    out = capsys.readouterr().out
    assert "review moves" in out and f"{old_id} -> {new_id}" in out
    assert main(["report", str(path)]) == 0
    rows = (tmp_path / "results_tables" / "spectra.csv").read_text().splitlines()
    header = rows[0].split(",")
    line = next(r.split(",") for r in rows[1:] if r.startswith(new_id + ","))
    assert line[header.index("review_status")] == "rejected"


def test_the_window_rewrites_a_moved_row_when_a_decision_is_written(tmp_path):
    from hxex.gui.review import Decision, write_decisions

    _, _, new, old_id, new_id, _ = _rerun_with_window(tmp_path)
    table = tmp_path / "user_review_table.csv"
    table.write_text("spectrum_id,review_status,review_K,note,extra\n"
                     f"{old_id},rejected,,first look,mine\n")
    write_decisions(table, {new_id: Decision(note="second look")}, results=new)
    rows = [r.split(",") for r in table.read_text().splitlines()]
    assert rows[1:] == [[new_id, "rejected", "", "second look", "mine"]]


def test_the_choices_are_the_same_shape_without_moves():
    assert moved_choices({}, np.array([f"{A}/scans1-2"])) == {}
