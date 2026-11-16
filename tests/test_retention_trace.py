# SPDX-License-Identifier: GPL-3.0-or-later
"""Retention traces read again from a spectrum's scan files (hxex.io.retention_trace) and drawn
(hxex.report.retention): the review's trace panel (PLAN Next steps 26). Synthetic LC runs of
one sample, the second eluting 6 s later; ion-mobility scans with another ion at a later drift
time."""

import dataclasses

import numpy as np
import pytest

from hxex.io import MzmlRun, MzmlTarget, read_mzml
from hxex.io.retention_trace import TraceReader, readable
from hxex.report.retention import plot_retention
from test_mzml import ENTRY, _drifting, write_mzml
from test_retention_search import PEPTIDES, _entry, lc_run

PEPTIDE = "VDQIIMAK"  # elutes at 38 s
SHIFT = 6.0  # the second run elutes this much later (s)


@pytest.fixture(scope="module")
def lc(tmp_path_factory):
    d = tmp_path_factory.mktemp("lc")
    runs = [MzmlRun(path=str(write_mzml(d / "un1.mzML", lc_run(1))), sample="S", rep=1,
                    role="undeut"),
            MzmlRun(path=str(write_mzml(d / "un2.mzML", lc_run(2, shift=SHIFT))), sample="S",
                    rep=2, role="undeut"),
            MzmlRun(path=str(write_mzml(d / "t60.mzML", lc_run(3))), sample="S", rep=1,
                    time=60.0)]
    rt = PEPTIDES[PEPTIDE]
    target = MzmlTarget(_entry(PEPTIDE), rt_window=(rt - 4.0, rt + 4.0))
    return read_mzml(runs, [target], project="p"), d


def _ids(spectra, file):
    return [s for s in spectra.spectrum_id if f"/{file}/" in s]


def _apex(trace):
    return 60.0 * trace.times[int(np.argmax(trace.intensity))]


def test_the_selected_run_comes_first_with_the_scans_it_summed(lc):
    spectra, _ = lc
    assert readable(spectra.source) and not readable({"reader": "specexport"})
    reader = TraceReader(spectra.source)
    sid = _ids(spectra, "un2.mzML")[0]
    others = [s for s in spectra.spectrum_id if s != sid]
    traces = reader.traces([sid, *others], 0)
    assert traces.note is None and traces.drift is None
    assert [t.file for t in traces.traces] == ["un2.mzML", "un1.mzML", "t60.mzML"]
    first = traces.traces[0]
    assert first.selected and not any(t.selected for t in traces.traces[1:])
    assert first.role == "undeut" and first.rep == 2 and traces.traces[2].time == 60.0
    i = list(spectra.spectrum_id).index(sid)
    assert first.window == pytest.approx((spectra.rt_start[i], spectra.rt_end[i]))
    # each run's trace peaks at its own elution: the second run 6 s later
    assert _apex(first) == pytest.approx(PEPTIDES[PEPTIDE] + SHIFT, abs=0.5)
    assert _apex(traces.traces[1]) == pytest.approx(PEPTIDES[PEPTIDE], abs=0.5)
    # shown around the windows: half a minute either side (within the 60 s run)
    start = min(t.window[0] for t in traces.traces)
    assert first.times.min() == pytest.approx(start - 0.5, abs=0.005)
    assert first.times.max() == pytest.approx(1.0)
    assert all(np.array_equal(t.times, first.times) for t in traces.traces)  # same scan times


def test_ids_that_name_no_run_and_files_that_moved(lc, tmp_path):
    spectra, folder = lc
    reader = TraceReader(spectra.source)
    sid = _ids(spectra, "un1.mzML")[0]
    assert reader.locate(sid, 0)[0] == 0
    assert reader.locate(sid.replace("/un1.mzML/", "/other.mzML/"), 0) is None
    assert reader.locate("p:S/un1.mzML/something", 0) is None
    assert "not named in its id" in reader.traces(["p:S/un1.mzML/x/scans1-2"], 0).note
    moved = TraceReader(spectra.source, directory=tmp_path)  # the files are not there
    out = moved.traces([sid], 0)
    assert out.traces == [] and "un1.mzML not found" in out.note
    # the selected run's file present, another's missing: drawn without it, and said so
    (tmp_path / "un1.mzML").write_bytes((folder / "un1.mzML").read_bytes())
    out = moved.traces([sid, *_ids(spectra, "un2.mzML")], 0)
    assert [t.file for t in out.traces] == ["un1.mzML"] and "un2.mzML not found" in out.note


def test_a_files_mz_offset_moves_the_bins(lc):
    spectra, _ = lc
    sid = _ids(spectra, "un1.mzML")[0]
    plain = TraceReader(spectra.source).traces([sid], 0).traces[0]
    source = dict(spectra.source)
    source["mz_calibration"] = [{"offset_ppm": 2000.0, "n_spectra": 10}] * 3
    moved = TraceReader(source).traces([sid], 0).traces[0]
    assert moved.intensity.max() < 0.01 * plain.intensity.max()  # the bins miss the peaks


def test_ion_mobility_traces_keep_the_drift_range(tmp_path):
    lc = [{"time": 1.0 + 0.05 * k, "filter": "Full ms"} for k in range(7)]
    peptide = {5.1: 0.02, 5.2: 0.4, 5.3: 1.0, 5.4: 0.5, 5.5: 0.05}
    other = {5.7: 1.0, 5.8: 0.6}
    un = write_mzml(tmp_path / "un.mzML", _drifting(lc, [(1, 1e5, peptide), (7, 5e4, other)]),
                    indexed=True)
    runs = [MzmlRun(un, "WT", role="undeut")]
    spectra = read_mzml(runs, [MzmlTarget(ENTRY, rt_window=(63.0, 69.0))], project="p")
    sid = spectra.spectrum_id[0]
    drift = (float(spectra.drift_start[0]), float(spectra.drift_end[0]))
    assert drift == pytest.approx((5.2, 5.4))
    reader = TraceReader(spectra.source)
    gated = reader.traces([sid], 0, drift_ranges={sid: drift})
    summed = reader.traces([sid], 0)
    assert gated.traces[0].drift == pytest.approx(drift) and summed.traces[0].drift is None
    # the other ion (on the same m/z grid, at a later drift time) is left out of the gated trace
    assert (gated.traces[0].intensity < summed.traces[0].intensity).any()
    profile = gated.drift
    assert profile.range == pytest.approx(drift)
    assert profile.drift[int(np.argmax(profile.intensity))] == pytest.approx(5.3)
    fig = plot_retention(gated)
    assert len(fig.axes) == 2 and "drift 5.2-5.4 ms" in fig._suptitle.get_text()
    assert {a.get_gid() for a in fig.axes[1].get_children()} >= {"drift_profile", "drift_range"}
    # a peak a few drift bins wide shows its baseline either side (owner, 2026-11-12: it was
    # drawn cut off), and a range being edited inside the axis does not move it
    from hxex.io.retention_trace import DriftProfile

    bins = np.round(np.arange(1.0, 12.0, 0.07), 3)
    peak = np.where(np.abs(bins - 2.33) < 0.1, 1.0, 0.0)  # three bins
    narrow = dataclasses.replace(gated, drift=DriftProfile(bins, peak, (2.26, 2.40)))
    lo, hi = plot_retention(narrow).axes[1].get_xlim()
    drawn = (bins >= lo) & (bins <= hi)
    assert (peak[drawn] == 0).sum() >= 6 and peak[drawn][0] == 0 and peak[drawn][-1] == 0
    inside = plot_retention(narrow, drift_edit=(2.30, 2.36)).axes[1].get_xlim()
    assert inside == pytest.approx((lo, hi))
    beyond = plot_retention(narrow, drift_edit=(2.30, 4.0)).axes[1].get_xlim()
    assert beyond == pytest.approx((lo, 4.0))


def test_the_figure_draws_every_run_or_says_why_not(lc):
    spectra, _ = lc
    sid = _ids(spectra, "un2.mzML")[0]
    traces = TraceReader(spectra.source).traces(list(spectra.spectrum_id[::-1]), 0)
    assert traces.traces[0].spectrum_id == spectra.spectrum_id[-1]
    traces = TraceReader(spectra.source).traces([sid, *_ids(spectra, "un1.mzML")], 0)
    fig = plot_retention(traces, title="S 1-8 z=2")
    gids = {line.get_gid() for line in fig.axes[0].get_lines()}
    assert {f"trace:{t.spectrum_id}" for t in traces.traces} <= gids
    assert len(fig.axes) == 1 and fig._suptitle.get_text() == "S 1-8 z=2"
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels[0] == "this spectrum: undeuterated, rep 2 (un2.mzML)"
    assert labels[-1] == "the sample's other files (1)"
    empty = plot_retention(dataclasses.replace(traces, traces=[], note="un2.mzML not found"))
    assert "No elution trace: un2.mzML not found" in empty.axes[0].texts[0].get_text()


def test_a_spectrum_summed_again_over_its_own_window_is_the_same(lc):
    spectra, _ = lc
    reader = TraceReader(spectra.source)
    for i, sid in enumerate(spectra.spectrum_id):
        window = (spectra.rt_start[i], spectra.rt_end[i])
        again = reader.spectrum(sid, int(spectra.data_id[i]), window)
        assert again.spectrum_id == sid and again.rep == spectra.rep[i]
        np.testing.assert_allclose(again.mz, spectra.spectrum(i)[0])
        np.testing.assert_allclose(again.intensity, spectra.spectrum(i)[1])
        assert (again.rt_start, again.rt_end) == pytest.approx(window)


def test_another_window_names_its_scans_and_sums_them(lc):
    spectra, _ = lc
    reader = TraceReader(spectra.source)
    i = list(spectra.spectrum_id).index(_ids(spectra, "un1.mzML")[0])
    sid, data_id = spectra.spectrum_id[i], int(spectra.data_id[i])
    start, end = spectra.rt_start[i], spectra.rt_end[i]
    narrow = reader.spectrum(sid, data_id, (start + 0.02, end - 0.02))
    assert narrow.spectrum_id != sid and narrow.spectrum_id.split("/scans")[0] == \
        sid.split("/scans")[0]
    assert start < narrow.rt_start and narrow.rt_end < end
    assert narrow.intensity.sum() < spectra.spectrum(i)[1].sum()
    with pytest.raises(ValueError, match="no scan"):
        reader.spectrum(sid, data_id, (2.0, 2.1))  # after the run's last scan (1 min)
    with pytest.raises(ValueError, match="before its start"):
        reader.spectrum(sid, data_id, (end, start))
    # one scan is a window (the boxes snap to scans: start and end may be the same scan)
    one = reader.spectrum(sid, data_id, (start, start))
    assert one.rt_start == one.rt_end == pytest.approx(start) and "/scan" in one.spectrum_id
    assert "-" not in one.spectrum_id.split("/scan")[-1]
    # the ion's every scan in the file, for the editor's boxes to snap to (F-1)
    traces = reader.traces([sid], data_id)
    times = traces.scan_times
    assert times is not None and np.all(np.diff(times) > 0)
    assert times.min() < start and times.max() > end  # the whole file, not the span drawn
    assert np.isclose(times, start).any() and np.isclose(times, end).any()


def test_the_peptides_other_ions_in_the_run(tmp_path):
    """The peptide's other ions in the selected run (Compare with: this peptide's other ions in
    this run): an MS1 charge state gives a trace of its own; a fragment channel one trace,
    summed over its ions; each peaks with the selected ion (they elute together), and the
    figure draws them scaled to the selected trace's height."""
    from hxex.config import load_project, read_input
    from test_elution_windows import _charges_project

    path = _charges_project(tmp_path / "charges")
    spectra = read_input(load_project(path))
    reader = TraceReader(spectra.source, data_root=path.parent)
    ids = list(spectra.spectrum_id)
    z2 = [i for i, s in enumerate(ids) if "/un2.mzML/" in s and s.split("/")[2].endswith("z2")][0]
    z3 = [i for i, s in enumerate(ids) if "/un2.mzML/" in s and s.split("/")[2].endswith("z3")][0]
    sid, data_id = ids[z2], int(spectra.data_id[z2])
    traces = reader.traces([sid], data_id)
    times = traces.traces[0].times
    ions = reader.ion_traces(sid, data_id, [(ids[z3], int(spectra.data_id[z3]))],
                             (float(times.min()), float(times.max())))
    assert [i.label for i in ions] == ["3+ (MS1)"]
    assert abs(_apex(ions[0]) - _apex(traces.traces[0])) <= 0.5  # s: within two scans
    # a spectrum of another run is not this run's ion
    other = [i for i, s in enumerate(ids) if "/un1.mzML/" in s][0]
    assert reader.ion_traces(sid, data_id, [(ids[other], int(spectra.data_id[other]))],
                             (float(times.min()), float(times.max()))) == []
    fig = plot_retention(dataclasses.replace(traces, ions=ions))
    drawn = [ln for ln in fig.axes[0].lines if ln.get_gid() == "ion:3+ (MS1)"]
    assert len(drawn) == 1
    assert np.max(drawn[0].get_ydata()) == pytest.approx(np.max(traces.traces[0].intensity))


def test_a_fragment_channel_is_one_trace(tmp_path):
    from hxex.fragments import fragment_targets
    from test_mzml import fragment_lc_run

    frags = fragment_targets(ENTRY, "c:2-4:1", activation="ETD", rt_window=(10.0, 60.0))
    path = write_mzml(tmp_path / "frag.mzML", fragment_lc_run(frags, other=None))
    targets = [MzmlTarget(ENTRY, rt_window=(10.0, 60.0)), *frags]
    spectra = read_mzml([MzmlRun(path=str(path), sample="S", time=60.0)], targets, project="p")
    reader = TraceReader(spectra.source, data_root=tmp_path)
    ids = list(spectra.spectrum_id)
    parent = [i for i, m in enumerate(spectra.modification) if "ion_type" not in m][0]
    sid, data_id = ids[parent], int(spectra.data_id[parent])
    traces = reader.traces([sid], data_id)
    times = traces.traces[0].times
    ions = reader.ion_traces(sid, data_id, [(s, int(d)) for i, (s, d) in enumerate(
        zip(ids, spectra.data_id, strict=True)) if i != parent],
        (float(times.min()), float(times.max())))
    assert len(ions) == 1 and ions[0].n_ions == 3
    assert ions[0].label.startswith("c ions: ETD MS2")
    assert abs(_apex(ions[0]) - _apex(traces.traces[0])) <= 1.0  # s



def test_every_trace_reader_reads_one_at_a_time(lc):
    """Two readers (the window made a new one for each review decision) share one lock: two
    threads reading Thermo raw files through .NET at once locked the window (2026-11-09)."""
    spectra, _ = lc
    a, b = TraceReader(spectra.source), TraceReader(spectra.source)
    assert a._lock is b._lock
    sid = _ids(spectra, "un2.mzML")[0]
    with a._lock:  # one reader reading: the other waits
        import threading

        done = []
        t = threading.Thread(target=lambda: done.append(b.traces([sid], 0)))
        t.start()
        t.join(0.5)
        assert not done
    t.join(30)
    assert done and done[0].traces


def test_the_time_axis_is_this_files_whatever_is_read_with_it(lc):
    """The trace panel's retention-time axis is this file's window and margin: reading the
    other files with it (which widens the span every trace is read over) or a file far away
    does not move it (owner, 2026-11-12: it changed between the tab's draws); only a window
    being edited beyond it widens it."""
    from hxex.io.retention_trace import MARGIN

    spectra, _ = lc
    sid = _ids(spectra, "un2.mzML")[0]
    reader = TraceReader(spectra.source)
    alone = reader.traces([sid], 0)
    together = reader.traces([sid, *_ids(spectra, "un1.mzML")], 0)
    far = dataclasses.replace(together.traces[1], times=together.traces[1].times + 10.0)
    distant = dataclasses.replace(together, traces=[together.traces[0], far])
    limits = [plot_retention(t).axes[0].get_xlim() for t in (alone, together, distant)]
    assert limits[0] == pytest.approx(limits[1]) and limits[0] == pytest.approx(limits[2])
    a, b = alone.traces[0].window
    pad = max(MARGIN, b - a)
    assert limits[0] == pytest.approx((a - pad, b + pad))
    # a window being edited beyond the trace's own span does not widen it either (audit of
    # 2026-11-15, F-11: a window at 200 min drew the trace as a sliver): it is marked
    ax = plot_retention(alone, edit=(a - 3 * pad, b - 2 * pad)).axes[0]
    assert ax.get_xlim() == pytest.approx(limits[0])
    assert any(t.get_gid() == "edited_outside" for t in ax.texts)
    ax = plot_retention(alone, edit=(a - 3 * pad, b)).axes[0]  # reaching into the trace
    assert ax.get_xlim() == pytest.approx(limits[0])
    assert not any(t.get_gid() == "edited_outside" for t in ax.texts)


def test_a_match_moves_the_runs_traces_and_changes_none(lc):
    """While a window is edited, the files are matched within the drift range in the boxes,
    but the traces drawn are those the run summed: a new drift range moves the other files by
    their new shifts and changes no trace (owner, 2026-11-12: option A)."""
    from matplotlib.figure import Figure

    from hxex.gui.window_edit import Alignment, FilePlan
    from hxex.report.retention import _aligned_panel, plot_each_file

    spectra, _ = lc
    sid = _ids(spectra, "un2.mzML")[0]
    traces = TraceReader(spectra.source).traces([sid, *_ids(spectra, "un1.mzML")], 0)
    mine, other = traces.traces
    gated = [dataclasses.replace(t, intensity=0.3 * t.intensity) for t in (mine, other)]
    shift = 0.05  # minutes
    aligned = Alignment(sid, "replicates", (1.0, 2.0), gated[0], (
        FilePlan(other.file, "", ("S",), shift, None, 0.99, trace=gated[1]),))
    ax = Figure().add_subplot()
    _aligned_panel(ax, traces, aligned, None)
    lines = {line.get_gid(): line for line in ax.get_lines()}
    assert lines[f"trace:{mine.spectrum_id}"].get_ydata() == pytest.approx(mine.intensity)
    drawn = lines[f"trace:{other.spectrum_id}"]
    assert drawn.get_xdata() == pytest.approx(other.times - shift)
    top = mine.intensity.max() / other.intensity.max()  # scaled to this one's height
    assert drawn.get_ydata() == pytest.approx(other.intensity * top)
    rows = plot_each_file(traces, aligned, edit=mine.window).axes
    assert rows[0].get_lines()[0].get_ydata() == pytest.approx(mine.intensity)
    assert rows[1].get_lines()[0].get_ydata() == pytest.approx(other.intensity)


def test_a_trace_read_stops_when_its_cancel_hook_says_so(lc):
    """Trace reads could not be stopped: a read for the spectrum before held the reader's lock
    and the next spectrum's trace waited for it, and the window's close waited too (audit of
    2026-11-15, F-6)."""
    from hxex.hooks import Cancelled

    spectra, _ = lc
    reader = TraceReader(spectra.source)
    sid = _ids(spectra, "un2.mzML")[0]
    with pytest.raises(Cancelled):
        reader.traces([sid], 0, cancel=lambda: True)
    with pytest.raises(Cancelled):
        reader.ion_traces(sid, 0, [(s, 0) for s in _ids(spectra, "un2.mzML")], (0.0, 10.0),
                          cancel=lambda: True)
    assert reader.traces([sid], 0, cancel=lambda: False).traces  # not stopped: read


def test_a_file_that_is_not_the_one_the_run_read_is_said(lc, tmp_path):
    """The trace reader opens whatever is at the path; a file replaced by another conversion
    (its size or an indexed mzML's own checksum differs from the record) is read all the same,
    with a note in the traces (audit of 2026-11-15, F-14)."""
    spectra, _ = lc
    source = spectra.source
    sid, data_id = spectra.spectrum_id[0], int(spectra.data_id[0])
    assert TraceReader(source).traces([sid], data_id).note is None
    run_i = TraceReader(source).locate(sid, data_id)[0]
    files = list(source["files"])
    files[run_i] = {**files[run_i], "size": files[run_i]["size"] + 1}
    other = {**source, "files": files}
    reader = TraceReader(other)
    got = reader.traces([sid], data_id)
    assert got.traces and "is not the file the run read (its size differs)" in got.note
    assert reader.changed_file(run_i) == got.note
    # a checksum the file carries at its end (an indexed mzML's) is compared when the size is
    # the same; a record without one leaves the file unquestioned
    files[run_i] = {**source["files"][run_i], "algorithm": "mzml-sha1", "checksum": "0" * 40}
    got = TraceReader({**source, "files": files}).traces([sid], data_id)
    assert got.note is None or "checksum" in got.note  # no tail checksum in this file: unknown
    files[run_i] = {k: v for k, v in source["files"][run_i].items() if k != "size"}
    assert TraceReader({**source, "files": files}).traces([sid], data_id).note is None


def test_a_closed_reader_refuses_to_read(lc):
    """A read queued on a reader whose files were closed (other results shown) reopened them,
    and nothing closed them again (audit of 2026-11-15, F-18)."""
    spectra, _ = lc
    reader = TraceReader(spectra.source)
    sid, data_id = spectra.spectrum_id[0], int(spectra.data_id[0])
    assert reader.traces([sid], data_id).traces
    reader.close()
    assert not reader._open
    with pytest.raises(ValueError, match="closed"):
        reader.traces([sid], data_id)
    with pytest.raises(ValueError, match="closed"):
        reader.spectrum(sid, data_id, (spectra.rt_start[0], spectra.rt_end[0]))
    assert not reader._open


def test_the_drift_profile_of_a_new_window(tmp_path):
    """``summed(..., profile=True)`` gives the drift profile of the scans the new window sums
    (the editor's drift panel follows the window being edited: audit of 2026-11-15, F open
    question 1); without ion mobility bins there is none."""
    lc = [{"time": 1.0 + 0.05 * k, "filter": "Full ms"} for k in range(7)]
    peptide = {5.1: 0.02, 5.2: 0.4, 5.3: 1.0, 5.4: 0.5, 5.5: 0.05}
    un = write_mzml(tmp_path / "un.mzML", _drifting(lc, [(1, 1e5, peptide)]), indexed=True)
    spectra = read_mzml([MzmlRun(un, "WT", role="undeut")],
                        [MzmlTarget(ENTRY, rt_window=(63.0, 69.0))], project="p")
    sid = spectra.spectrum_id[0]
    drift = (float(spectra.drift_start[0]), float(spectra.drift_end[0]))
    reader = TraceReader(spectra.source)
    whole = reader.summed(sid, 0, (spectra.rt_start[0], spectra.rt_end[0]), drift=drift,
                          profile=True)
    assert whole.profile is not None and whole.profile.range == pytest.approx(drift)
    run = reader.traces([sid], 0, drift_ranges={sid: drift}).drift
    np.testing.assert_allclose(whole.profile.drift, run.drift)
    np.testing.assert_allclose(whole.profile.intensity, run.intensity)
    part = reader.summed(sid, 0, (spectra.rt_start[0], spectra.rt_start[0] + 0.06), drift=drift,
                         profile=True)
    assert part.profile.intensity.sum() < whole.profile.intensity.sum()
    assert reader.summed(sid, 0, (spectra.rt_start[0], spectra.rt_end[0])).profile is None
    plain = write_mzml(tmp_path / "plain.mzML", lc_run(1))
    flat = read_mzml([MzmlRun(plain, "S", role="undeut")], [MzmlTarget(_entry("VDQIIMAK"))],
                     project="p")
    reader = TraceReader(flat.source)
    got = reader.summed(flat.spectrum_id[0], int(flat.data_id[0]),
                        (flat.rt_start[0], flat.rt_end[0]), profile=True)
    assert got.profile is None and got.spectrum.scans is None


def test_a_fragment_of_a_file_fitted_through_the_cut_keeps_the_scans_record(tmp_path):
    """A fragment ion of a reduced-profile file that records its scans' noise is fitted
    through the instrument's per-scan cut ([input.scans] scan_cut "auto"); summed again over a
    window it carries the same scans' record as the run gave it, so the preview is fitted as
    the run was (audit of 2026-11-15, F-3). The parent (MS1) carries none."""
    from hxex.fragments import fragment_targets
    from hxex.io.mzml import MzmlConfig
    from test_mzml import _array, fragment_lc_run
    from test_thermo import _reduced, _with_analyser

    frags = fragment_targets(ENTRY, "c:2-4:1", activation="ETD", rt_window=(10.0, 60.0))
    targets = [MzmlTarget(ENTRY, rt_window=(10.0, 60.0)), *frags]
    path = write_mzml(tmp_path / "frag.mzML", _reduced(fragment_lc_run(frags, other=None)))
    noise = (_array(np.linspace(100, 2000, 7), "MS:1002743", name="sampled noise m/z array")
             + _array(np.linspace(10, 100, 7), "MS:1002744",
                      name="sampled noise intensity array")
             + _array(np.zeros(7), "MS:1002745", name="sampled noise baseline array"))
    path.write_text(path.read_text().replace("</binaryDataArrayList>",
                                             noise + "</binaryDataArrayList>"))
    _with_analyser(path, "MS:1000484", "orbitrap")
    spectra = read_mzml([MzmlRun(path=str(path), sample="S", time=60.0)], targets,
                        project="p", config=MzmlConfig(find_fragment_peak=True))
    assert spectra.source["scan_cut"]["files"] == ["frag.mzML"]
    reader = TraceReader(spectra.source)
    kinds = ["ion_type" in m for m in spectra.modification]
    frag, parent = kinds.index(True), kinds.index(False)
    for i in (frag, parent):
        sid, data_id = spectra.spectrum_id[i], int(spectra.data_id[i])
        same = reader.spectrum(sid, data_id, (spectra.rt_start[i], spectra.rt_end[i]))
        assert same.spectrum_id == sid
        record = spectra.scans[i]
        if i == parent:
            assert record is None and same.scans is None
            continue
        assert record is not None and same.scans is not None
        np.testing.assert_allclose(same.scans.weights, record.weights)
        np.testing.assert_allclose(same.scans.noise_mz, record.noise_mz)
        np.testing.assert_allclose(same.scans.noise, record.noise)
        # fewer scans: the record follows them (their shares sum to one)
        part = reader.spectrum(sid, data_id, (spectra.rt_start[i], spectra.rt_end[i] - 0.02))
        assert len(part.scans.weights) < len(record.weights)
        assert part.scans.weights.sum() == pytest.approx(1.0)
    # the same file read without the cut: no record
    off = read_mzml([MzmlRun(path=str(path), sample="S", time=60.0)], targets, project="p",
                    config=MzmlConfig(find_fragment_peak=True, scan_cut=False))
    reader = TraceReader(off.source)
    assert reader.spectrum(off.spectrum_id[frag], int(off.data_id[frag]),
                           (off.rt_start[frag], off.rt_end[frag])).scans is None
