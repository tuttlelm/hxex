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
    with pytest.raises(ValueError, match="after its start"):
        reader.spectrum(sid, data_id, (end, start))


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
