# SPDX-License-Identifier: GPL-3.0-or-later
"""Windows saved since the run (hxex.io.elution_windows.pending_windows): the spectra whose
next read sums other scans than the run did, and "Back to the run's window"
(restore_run_windows), which writes the table so that they read as the run did."""

import logging
from types import SimpleNamespace

import pytest

from hxex.config import load_project, read_input
from hxex.io.elution_windows import (
    forget_cached,
    pending_windows,
    peptide_row_shared,
    restore_run_windows,
    save_window,
    undo_window,
)
from test_elution_windows import PEPTIDE, N, _charges_project

PEP = f"0001-{N:04d}-{PEPTIDE}"
Z2, Z3 = f"{PEP}-z2", f"{PEP}-z3"


def _run(path):
    """The project read as hxex run reads it, as results the pending test takes."""
    forget_cached()
    spectra = read_input(load_project(path))
    return SimpleNamespace(settings=SimpleNamespace(spectra_source=spectra.source),
                           binned=spectra)


def _ids(pending):
    """(file, charge) of each pending spectrum."""
    return sorted((p.file, int(p.ion[-1])) for p in pending.by_id.values())


def _summed(results, file, ion):
    b = results.binned
    i = next(k for k, s in enumerate(b.spectrum_id) if f"/{file}/{ion}/" in s)
    return float(b.rt_start[i]), float(b.rt_end[i])


def test_a_window_saved_after_the_run_makes_its_spectra_pending(tmp_path):
    path = _charges_project(tmp_path / "p")
    results = _run(path)
    assert len(pending_windows(results, path)) == 0
    s0, s1 = _summed(results, "un1.mzML", Z2)
    save_window(path, ("S", "un1.mzML", Z2), (s0 + 0.02, s1), peptide=PEP)
    pending = pending_windows(results, path)
    assert _ids(pending) == [("un1.mzML", 2)]
    p = next(iter(pending.by_id.values()))
    assert p.run_window is None and p.window[0] > s0 and p.summed == (s0, s1)


def test_the_runs_own_window_saved_again_is_not_pending(tmp_path):
    """The run used the peptide table: a row selecting the same scans is not pending. The run
    had a row: the same row saved again is not pending either."""
    path = _charges_project(tmp_path / "p")
    results = _run(path)
    s0, s1 = _summed(results, "un1.mzML", Z2)
    save_window(path, ("S", "un1.mzML", Z2), (s0, s1), peptide=PEP)
    assert len(pending_windows(results, path)) == 0
    rerun = _run(path)  # the row in the run's record now
    save_window(path, ("S", "un1.mzML", Z2), (s0, s1), peptide=PEP)
    assert len(pending_windows(rerun, path)) == 0
    save_window(path, ("S", "un1.mzML", Z2), (s0 + 0.02, s1), peptide=PEP)
    assert _ids(pending_windows(rerun, path)) == [("un1.mzML", 2)]


def test_a_peptide_row_marks_every_ion_of_the_peptide_in_that_file(tmp_path):
    path = _charges_project(tmp_path / "p")
    results = _run(path)
    s0, s1 = _summed(results, "un1.mzML", Z2)
    save_window(path, ("S", "un1.mzML", Z2), (s0 + 0.02, s1), peptide=PEP, whole=True,
                siblings=(Z2, Z3))
    assert _ids(pending_windows(results, path)) == [("un1.mzML", 2), ("un1.mzML", 3)]


def test_removing_a_row_the_run_did_not_have_clears_it(tmp_path):
    path = _charges_project(tmp_path / "p")
    results = _run(path)
    s0, s1 = _summed(results, "un2.mzML", Z3)
    saved = save_window(path, ("S", "un2.mzML", Z3), (s0, s1 - 0.02), peptide=PEP)
    assert _ids(pending_windows(results, path)) == [("un2.mzML", 3)]
    undo_window(saved)
    assert len(pending_windows(results, path)) == 0


def test_a_row_the_run_had_and_the_table_lost_is_pending(tmp_path):
    table = f"sample,file,peptide,ion,rt_start,rt_end\nS,un1.mzML,{PEP},,0.6000,0.6600\n"
    path = _charges_project(tmp_path / "p", table)
    results = _run(path)
    (path.parent / "rt_elution_windows.csv").write_text("sample,file,peptide,ion,rt_start,"
                                                        "rt_end\n")
    forget_cached()
    pending = pending_windows(results, path)
    assert _ids(pending) == [("un1.mzML", 2), ("un1.mzML", 3)]
    assert all(p.window is None and p.run_window == (0.6, 0.66)
               for p in pending.by_id.values())


def test_results_without_the_record_mark_nothing_and_say_so_once(tmp_path, caplog):
    path = _charges_project(tmp_path / "p")
    results = _run(path)
    results.settings.spectra_source = {k: v for k, v in results.binned.source.items()
                                       if k != "elution_windows"}
    save_window(path, ("S", "un1.mzML", Z2), (0.61, 0.65), peptide=PEP)
    with caplog.at_level(logging.INFO, logger="hxex.io.elution_windows"):
        first = pending_windows(results, path)
        second = pending_windows(results, path)
    assert len(first) == len(second) == 0 and "does not hold" in first.note
    assert sum("does not hold" in r.message for r in caplog.records) == 1
    assert pending_windows(results, None).note  # no project file: nothing compared


def test_back_to_the_runs_window_removes_the_row_the_save_added(tmp_path):
    path = _charges_project(tmp_path / "p")
    results = _run(path)
    s0, s1 = _summed(results, "un1.mzML", Z2)
    save_window(path, ("S", "un1.mzML", Z2), (s0 + 0.02, s1), peptide=PEP)
    pending = pending_windows(results, path)
    entry = next(iter(pending.by_id.values()))
    assert peptide_row_shared(pending, entry) == []
    restored = restore_run_windows(path, pending, [entry], whole=False, ions=(Z2, Z3))
    assert len(pending_windows(results, path)) == 0
    assert (path.parent / "rt_elution_windows.csv").read_text().splitlines()[1:] == []
    undo_window(restored)  # Undo puts the saved window back
    assert len(pending_windows(results, path)) == 1


def test_back_to_the_runs_rows(tmp_path):
    """The run had a peptide row: putting it back is the whole peptide's; for this ion alone,
    the ion gets a row of its own with the run's window, and the other charge keeps the new
    peptide row (the window asks which: peptide_row_shared names the other ion)."""
    table = f"sample,file,peptide,ion,rt_start,rt_end\nS,un1.mzML,{PEP},,0.6000,0.6600\n"
    path = _charges_project(tmp_path / "p", table)
    results = _run(path)
    save_window(path, ("S", "un1.mzML", Z2), (0.61, 0.65), peptide=PEP, whole=True,
                siblings=(Z2, Z3))
    pending = pending_windows(results, path)
    entry = pending.at("S", "un1.mzML", PEP, Z2)
    assert peptide_row_shared(pending, entry) == [Z3]
    alone = restore_run_windows(path, pending, [entry], whole=False, ions=(Z2, Z3))
    assert _ids(pending_windows(results, path)) == [("un1.mzML", 3)]
    text = (path.parent / "rt_elution_windows.csv").read_text()
    assert f"S,un1.mzML,{PEP},{Z2},0.6,0.66" in text
    undo_window(alone)
    restore_run_windows(path, pending_windows(results, path), [entry], whole=True,
                        ions=(Z2, Z3))
    assert len(pending_windows(results, path)) == 0
    assert len(pending_windows(_run(path), path)) == 0  # read again: as the run read it


def test_a_project_that_names_no_table_is_not_written(tmp_path):
    """The run read a table the project file no longer names: its spectra are pending (the
    next read takes no table), and Back to the run's window says why it cannot write one."""
    table = f"sample,file,peptide,ion,rt_start,rt_end\nS,un1.mzML,{PEP},,0.6000,0.6600\n"
    path = _charges_project(tmp_path / "p", table)
    results = _run(path)
    path.write_text(path.read_text().replace('rt_elution_windows = "rt_elution_windows.csv"',
                                             ""))
    forget_cached()
    pending = pending_windows(results, path)
    assert _ids(pending) == [("un1.mzML", 2), ("un1.mzML", 3)]
    before = (path.parent / "rt_elution_windows.csv").read_bytes()
    with pytest.raises(ValueError, match="names no elution-window table"):
        restore_run_windows(path, pending, list(pending.by_id.values()), whole=True,
                            ions=(Z2, Z3))
    assert (path.parent / "rt_elution_windows.csv").read_bytes() == before
