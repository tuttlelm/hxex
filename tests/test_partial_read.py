# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading again only the files whose part of the read changed (PLAN Next steps 39 b): the
other runs' spectra are taken from the stored ones, and the spectra and record are those of a
full read; where that cannot be, every file is read (logged)."""

import logging

from hxex.io import MzmlConfig, MzmlRun, MzmlTarget, read_mzml
from hxex.io.elution_windows import ElutionWindow, peptide_label
from test_mzml import ENTRY, _drifting, _with_rt, lc_run, write_mzml  # noqa: I001
from test_parallel_read import _same

PART = "reading again 1 of"
FULL = "reading every file"


def _window(file, start, end, sample="WT"):
    return ElutionWindow(sample, file, None, start, end, peptide=peptide_label(ENTRY))


def _ion_mobility(tmp_path):
    lc = [{"time": 1.0 + 0.1 * k, "filter": "Full ms"} for k in range(3)]
    peptide = {5.1: 0.02, 5.2: 0.4, 5.3: 1.0, 5.4: 0.5, 5.5: 0.05}
    other = {5.7: 1.0, 5.8: 0.6}
    files = {name: write_mzml(tmp_path / f"{name}.mzML", _drifting(lc, [
        (centre, height, peptide), (7, 5e4, other)]), indexed=True)
        for name, centre, height in (("un1", 1, 1e5), ("un2", 1, 8e4), ("lab1", 4, 1e5),
                                     ("lab2", 5, 9e4))}
    runs = [MzmlRun(files["un1"], "WT", role="undeut"), MzmlRun(files["un1"], "MUT",
                                                                 role="undeut"),
            MzmlRun(files["un2"], "WT", rep=2, role="undeut"),
            MzmlRun(files["lab1"], "WT", time=60.0), MzmlRun(files["lab2"], "MUT", time=60.0)]
    return runs, [MzmlTarget(ENTRY, rt_window=(60.0, 72.0))]


def test_one_changed_window_reads_its_file_only(tmp_path, caplog):
    """A labelled file's window changed: that file is read again, the other runs (a control
    file shared by two samples among them) taken from the stored spectra, drift ranges and
    all: the spectra and record of a full read."""
    runs, targets = _ion_mobility(tmp_path)
    before = read_mzml(runs, targets, project="p",
                       elution_windows=[_window("lab1.mzML", 1.0, 1.2)])
    rows = [_window("lab1.mzML", 1.1, 1.2)]
    full = read_mzml(runs, targets, project="p", elution_windows=rows)
    with caplog.at_level(logging.INFO, logger="hxex"):
        part = read_mzml(runs, targets, project="p", elution_windows=rows,
                         again=(before, frozenset({"lab1.mzML"})))
    assert PART in caplog.text and FULL not in caplog.text
    _same(full, part)
    assert part.source["drift_gate"]["ranges"] and not (part.intensity == before.intensity).all()


def test_a_changed_control_window_reads_every_file_when_its_drift_range_moves(tmp_path, caplog):
    """An undeuterated control read again gives the drift ranges again: unchanged, only it is
    read; moved (its window now keeps the peptide's scans, not a brighter ion's on the same m/z
    grid at another drift time), every file is: the ion's spectra change in every file."""
    early = [{"time": 1.0 + 0.1 * k, "filter": "Full ms"} for k in range(3)]
    late = [{"time": 1.3 + 0.1 * k, "filter": "Full ms"} for k in range(3)]
    peptide = {5.1: 0.02, 5.2: 0.4, 5.3: 1.0, 5.4: 0.5, 5.5: 0.05}
    other = {5.7: 1.0, 5.8: 0.6}
    un = write_mzml(tmp_path / "un.mzML", _drifting(early, [(1, 1e5, peptide)])
                    + _drifting(late, [(1, 3e5, other)]), indexed=True)
    lab = write_mzml(tmp_path / "lab.mzML", _drifting(early + late, [(4, 1e5, peptide)]))
    runs = [MzmlRun(un, "WT", role="undeut"), MzmlRun(lab, "WT", time=60.0)]
    targets = [MzmlTarget(ENTRY, rt_window=(60.0, 90.0))]
    before = read_mzml(runs, targets, project="p")
    for start, end, moved in ((1.0, 1.5, False), (1.0, 1.2, True)):
        caplog.clear()
        rows = [_window("un.mzML", start, end)]
        full = read_mzml(runs, targets, project="p", elution_windows=rows)
        with caplog.at_level(logging.INFO, logger="hxex"):
            part = read_mzml(runs, targets, project="p", elution_windows=rows,
                             again=(before, frozenset({"un.mzML"})))
        _same(full, part)
        assert (full.source["drift_gate"]["ranges"] != before.source["drift_gate"]["ranges"]
                ) == moved
        assert (FULL in caplog.text) == moved and (PART in caplog.text) != moved


def test_find_peak_keeps_the_controls_suggestions_and_reads_all_for_a_searched_one(
        tmp_path, caplog):
    """With find_peak, the retention search ("suggest") looks at the undeuterated controls:
    a labelled file read again keeps the stored suggestions; a control read again is searched
    again, so every file is read."""
    paths = [write_mzml(tmp_path / f"lc{k}.mzML", lc_run(apex=30.0 + k, seed=k))
             for k in range(3)]
    runs = [MzmlRun(str(paths[0]), "S", role="undeut"),
            MzmlRun(str(paths[1]), "S", time=60.0), MzmlRun(str(paths[2]), "S", time=120.0)]
    targets = [MzmlTarget(_with_rt(ENTRY, 30.5 / 60, z)) for z in (2, 3)]
    config = MzmlConfig(find_peak=True, peak_search=30.0)
    before = read_mzml(runs, targets, project="p", config=config)
    assert "retention_search" in before.source
    for name, full_expected in (("lc1.mzML", False), ("lc0.mzML", True)):
        caplog.clear()
        rows = [ElutionWindow("S", name, None, 0.45, 0.6, peptide=peptide_label(ENTRY))]
        full = read_mzml(runs, targets, project="p", config=config, elution_windows=rows)
        with caplog.at_level(logging.INFO, logger="hxex"):
            part = read_mzml(runs, targets, project="p", config=config, elution_windows=rows,
                             again=(before, frozenset({name})))
        _same(full, part)
        assert (FULL in caplog.text) == full_expected and (PART in caplog.text) != full_expected
