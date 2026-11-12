# SPDX-License-Identifier: GPL-3.0-or-later
"""A spectrum's retention window changed in the review (hxex.gui.window_edit; PLAN Next steps
26): the preview picked and fitted with the run's settings, and the window written as the run's
own row of the peptide table, which the next read takes (one series still)."""

import numpy as np
import pytest

from hxex.config import load_project, read_input
from hxex.fit import FitConfig
from hxex.gui.window_edit import (
    compare_words,
    has_fragments,
    own_windows,
    plan_spread,
    preview,
    samples_reading,
    save_window,
    saved_window,
    saved_words,
    scope_words,
    spread_words,
    trace_shift,
    undo_window,
)
from hxex.io.retention_trace import TraceReader
from hxex.model import Backend
from hxex.noise import NoiseConfig
from hxex.picking import PickerConfig
from hxex.pipeline import RunConfig, run
from hxex.report import Results
from test_mzml import write_mzml
from test_retention_search import PEPTIDES, lc_run

PEPTIDE = "VDQIIMAK"
N = len(PEPTIDE)
RT = PEPTIDES[PEPTIDE] / 60.0  # minutes


@pytest.fixture
def project(tmp_path):
    write_mzml(tmp_path / "un1.mzML", lc_run(1))
    write_mzml(tmp_path / "un2.mzML", lc_run(2, shift=6.0))
    (tmp_path / "peptides.csv").write_text(
        "Peptide,start_seq,end_seq,charge,rt_start,rt_end,note\n"
        f"{PEPTIDE},1,{N},2,{RT - 4 / 60:.4f},{RT + 4 / 60:.4f},kept\n")
    path = tmp_path / "hxex.toml"
    path.write_text("""project = "p"

[input]
format = "mzml"
peptides = "peptides.csv"

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


def _read(path):
    """The project's spectra, as hxex run reads them (with its rt_elution_windows table)."""
    return read_input(load_project(path))


def _results(spectra):
    cfg = RunConfig(picker=PickerConfig(noise_threshold=100.0), quality=None,
                    fit=FitConfig(k_min=1, k_max=2, adam_steps=50, n_restarts=1,
                                  backend=Backend("cpu", "float64"),
                                  noise=NoiseConfig("gaussian", percent=3.0, estimate=None)),
                    bootstrap=None, controls=None)
    return Results.from_run(run(spectra, cfg), spectra=spectra)


def test_the_preview_of_the_same_window_is_the_spectrum_the_run_fitted(project):
    spectra = _read(project)
    results = _results(spectra)
    reader = TraceReader(spectra.source, data_root=project.parent)
    i = 0
    sid, data_id = spectra.spectrum_id[i], int(spectra.data_id[i])
    same = reader.spectrum(sid, data_id, (spectra.rt_start[i], spectra.rt_end[i]))
    shown = preview(results, same)
    assert shown.spectrum.spectrum_id == sid and shown.k in (1, 2)
    m = list(results.fit.spectrum_id).index(sid)
    np.testing.assert_allclose(shown.results.measurements.observed[0],
                               results.measurements.observed[m])
    ours = shown.results.fit.fits[1].means[0]
    assert ours == pytest.approx(results.fit.fits[1].means[m], abs=0.02)
    assert any("noise" in n for n in shown.notes)  # this run had no noise curve
    narrow = preview(results, reader.spectrum(sid, data_id, (spectra.rt_start[i] + 0.02,
                                                              spectra.rt_end[i] - 0.02)))
    assert narrow.results.measurements.observed[0].sum() < \
        results.measurements.observed[m].sum()


def test_the_preview_fit_starts_from_the_runs_fit(project):
    """The window's preview starts every K from the run's fit of the spectrum and takes a few
    steps (PREVIEW_STEPS): over the spectrum's own window it gives the run's K and means."""
    spectra = _read(project)
    results = _results(spectra)
    reader = TraceReader(spectra.source, data_root=project.parent)
    for i, sid in enumerate(spectra.spectrum_id):
        same = reader.spectrum(sid, int(spectra.data_id[i]),
                               (spectra.rt_start[i], spectra.rt_end[i]))
        shown = preview(results, same, start_from=sid)
        m = list(results.fit.spectrum_id).index(sid)
        assert shown.k == int(results.fit.selected_K[m])
        for k in shown.results.fit.fits:
            assert np.sort(shown.results.fit.fits[k].means[0]) == pytest.approx(
                np.sort(results.fit.fits[k].means[m]), abs=0.05)
        assert shown.warm and not any("from the run's fit" in n for n in shown.notes)
    # a spectrum the results do not hold: the run's own starts
    assert not preview(results, same, start_from="p:nothing").warm


def test_the_preview_leaves_a_run_fit_that_no_longer_fits(project):
    """A window moved off a co-eluting species: the run's fit, as a start, sits on peaks the new
    spectrum lacks. Each K also starts from the new spectrum itself and keeps the better fit
    (owner, 2026-11-11), so K = 1 is not left on the old peaks."""
    spectra = _read(project)
    results = _results(spectra)
    reader = TraceReader(spectra.source, data_root=project.parent)
    sid = spectra.spectrum_id[0]
    m = list(results.fit.spectrum_id).index(sid)
    right = float(results.fit.fits[1].means[m, 0])
    results.fit.fits[1].n_raw[m] = 5.0  # the run's K = 1 on an envelope at full deuteration
    results.fit.fits[1].logit_p[m] = 6.0  # (from it alone: D 7.0 after the preview's steps)
    same = reader.spectrum(sid, int(spectra.data_id[0]), (spectra.rt_start[0], spectra.rt_end[0]))
    shown = preview(results, same, start_from=sid)
    assert shown.results.fit.fits[1].means[0, 0] == pytest.approx(right, abs=0.05)
    assert shown.warm


def _key(spectra, i):
    """The (sample, file, ion) of spectrum ``i``, as its id names them."""
    sample, file, ion, _scans = spectra.spectrum_id[i].split(":", 1)[1].split("/")
    return sample, file, ion


def test_a_window_is_written_to_the_elution_window_table_and_read_back(project):
    """Save writes the ion's row for its file in [input] rt_elution_windows (made, and named in
    the project file, by the first save); the peptide table is never changed. Read again, the
    file sums the new scans and the runs stay one series; Undo puts both files back."""
    spectra = _read(project)
    peptides = project.parent / "peptides.csv"
    original = {p: p.read_bytes() for p in (peptides, project)}
    reader = TraceReader(spectra.source, data_root=project.parent)
    i = [k for k, s in enumerate(spectra.spectrum_id) if "/un2.mzML/" in s][0]
    sid, data_id = spectra.spectrum_id[i], int(spectra.data_id[i])
    start, end = spectra.rt_start[i] + 0.02, spectra.rt_end[i] - 0.01
    edited = reader.spectrum(sid, data_id, (start, end))
    key = _key(spectra, i)
    saved = save_window(project, key, (edited.rt_start, edited.rt_end))
    table = project.parent / "rt_elution_windows.csv"
    assert saved.added and saved.path == table and len(saved.files) == 2
    assert saved.row["file"] == "un2.mzML" and saved.row["ion"] == key[2]
    assert float(saved.row["rt_start"]) <= edited.rt_start <= edited.rt_end <= \
        float(saved.row["rt_end"])
    assert peptides.read_bytes() == original[peptides]  # never changed
    assert 'rt_elution_windows = "rt_elution_windows.csv"' in project.read_text()
    assert saved_window(project, key) == ((float(saved.row["rt_start"]),
                                           float(saved.row["rt_end"])), None, "ion")
    again = _read(project)
    assert edited.spectrum_id in list(again.spectrum_id)
    assert len(set(again.data_id.tolist())) == 1
    k = list(again.spectrum_id).index(edited.spectrum_id)
    np.testing.assert_allclose(again.spectrum(k)[1], edited.intensity)
    # saved again: the row changes (no second row), the project file is not touched again
    wider = reader.spectrum(sid, data_id, (spectra.rt_start[i], spectra.rt_end[i] + 0.02))
    second = save_window(project, key, (wider.rt_start, wider.rt_end))
    assert not second.added and len(second.files) == 1
    assert len(table.read_text().splitlines()) == 2
    undo_window(second)
    undo_window(saved)
    assert not table.exists() and project.read_bytes() == original[project]
    with pytest.raises(ValueError, match="changed since"):
        undo_window(saved)  # the files are no longer what that save wrote


def test_a_window_is_saved_for_every_sample_that_reads_the_file(project):
    """One file read by two samples (controls shared by the samples of a state) is one
    injection: a window saved from one sample is written for both (the table keys its rows by
    sample), and both read it; the other file of the sample (a replicate) is not touched
    (owner, 2026-11-11)."""
    project.write_text(project.read_text() + """
[[input.runs]]
path = "un1.mzML"
sample = "T"
role = "undeut"
""")
    spectra = _read(project)
    reader = TraceReader(spectra.source, data_root=project.parent)
    i = [k for k, s in enumerate(spectra.spectrum_id) if ":S/un1.mzML/" in s][0]
    sid, data_id = spectra.spectrum_id[i], int(spectra.data_id[i])
    key = _key(spectra, i)
    assert samples_reading(reader, sid, data_id, (key[2],)) == ("S", "T")
    j = [k for k, s in enumerate(spectra.spectrum_id) if ":S/un2.mzML/" in s][0]
    assert samples_reading(reader, spectra.spectrum_id[j], int(spectra.data_id[j]),
                           (key[2],)) == ("S",)  # un2 is read by S alone
    edited = reader.spectrum(sid, data_id, (spectra.rt_start[i] + 0.02, spectra.rt_end[i] - 0.01))
    saved = save_window(project, key, (edited.rt_start, edited.rt_end),
                        samples=samples_reading(reader, sid, data_id, (key[2],)))
    assert saved.samples == ("S", "T")
    lines = saved.path.read_text().splitlines()
    assert len(lines) == 3 and {ln.split(",")[0] for ln in lines[1:]} == {"S", "T"}
    assert "un2.mzML" not in saved.path.read_text()
    words = saved_words(saved, peptide=f"0001-{N:04d}-{PEPTIDE}", ion=key[2], file=key[1],
                        whole=False, fragments=False, drift_written=False)
    assert "z = 2 only" in words and "samples S, T" in words
    again = _read(project)
    for sample in ("S", "T"):  # both samples sum the new scans of un1
        ids = [s for s in again.spectrum_id if f":{sample}/un1.mzML/" in s]
        assert ids and ids[0].endswith(edited.spectrum_id.rsplit("/", 1)[1])
    undo_window(saved)
    assert not saved.path.exists()


def test_the_trace_match_finds_one_files_elution_against_another():
    """The shift of a file's elution, from its trace against the edited file's; refused at the
    search's edge, without signal, or when the traces do not match."""
    t = np.arange(12.0, 14.0, 0.034)  # a scan every 2 s
    peak = np.exp(-0.5 * ((t - 13.0) / 0.05) ** 2)
    later = np.exp(-0.5 * ((t - 13.1) / 0.05) ** 2)
    shift, r, why = trace_shift(t, peak, t, later)
    assert why is None and shift == pytest.approx(0.1, abs=0.004) and r > 0.99
    far = np.exp(-0.5 * ((t - 13.7) / 0.05) ** 2)  # 42 s away: beyond the search
    assert trace_shift(t, peak, t, far)[0] is None
    assert trace_shift(t, peak, t, np.zeros_like(t))[2] == "no signal in its trace"


def test_a_window_spread_to_the_replicates_is_shifted_by_their_elution(project):
    """Saved for the sample's replicates, the window goes to the other file shifted by its
    elution (un2 elutes 6 s later), as the peptide's row; the replicate's own window set before
    is reported (the window asks), and replaced by the spread (owner, 2026-11-11)."""
    spectra = _read(project)
    results = _results(spectra)
    reader = TraceReader(spectra.source, data_root=project.parent)
    i = [k for k, s in enumerate(spectra.spectrum_id) if "/un1.mzML/" in s][0]
    j = [k for k, s in enumerate(spectra.spectrum_id) if "/un2.mzML/" in s][0]
    sid, data_id, key = spectra.spectrum_id[i], int(spectra.data_id[i]), _key(spectra, i)
    peptide = f"0001-{N:04d}-{PEPTIDE}"
    window = (spectra.rt_start[i] + 0.01, spectra.rt_end[i] - 0.01)
    plans = plan_spread(reader, results, sid, data_id, window, peptide=peptide, ion=key[2],
                        ions=(key[2],), level="replicates")
    assert [(p.file, p.samples) for p in plans] == [("un2.mzML", ("S",))]
    assert plans[0].shift * 60 == pytest.approx(6.0, abs=0.5)
    assert plans[0].window[0] == pytest.approx(window[0] + plans[0].shift)
    assert "un2.mzML +6 s" in spread_words(plans)
    # un2 had a window of its own: reported, then replaced
    save_window(project, _key(spectra, j), (spectra.rt_start[j], spectra.rt_end[j]))
    assert own_windows(project, peptide, (key[2],), [("S", "un2.mzML")]) == [("S", "un2.mzML")]
    assert own_windows(project, peptide, (key[2],), [("S", "un1.mzML")]) == []
    saved = save_window(project, key, window, peptide=peptide, whole=True,
                        siblings=(key[2],), more=[("S", "un2.mzML", plans[0].window)])
    rows = [r.split(",") for r in saved.path.read_text().splitlines()[1:]]
    assert sorted((r[1], r[2], r[3]) for r in rows) == [("un1.mzML", peptide, ""),
                                                        ("un2.mzML", peptide, "")]
    again = _read(project)
    k = [n for n, s in enumerate(again.spectrum_id) if "/un2.mzML/" in s][0]
    assert again.rt_start[k] >= plans[0].window[0] - 0.02  # un2 sums the shifted window
    assert again.rt_end[k] <= plans[0].window[1] + 0.02
    assert plan_spread(reader, results, sid, data_id, window, peptide=peptide, ion=key[2],
                       ions=(key[2],), level="peptide") == []


def test_the_save_choices_are_worded_for_the_project():
    """Charge states for a peptide project, ions when fragments are read; files, not runs."""
    words = scope_words("0001-0008-VDQIIMAK-z3", False)
    assert list(words) == ["ion", "peptide", "replicates", "everywhere"]
    assert words["ion"] == "this charge state only (z = 3), in this file"
    assert words["peptide"] == "every charge state of this peptide, in this file"
    assert "replicates (each shifted)" in words["replicates"]
    assert scope_words("0001-0008-VDQIIMAK-c5-pz3-ETD", True)["ion"] == \
        "this ion only, in this file"
    assert compare_words(False)["replicates"] == "this charge state in its replicates"
    assert compare_words(True)["ions"] == "this peptide's other ions in this file"
    assert not has_fragments({"targets": [{"ms_level": 1}]})
    assert has_fragments({"targets": [{"ms_level": 1}, {"ms_level": 2}]})
    assert not has_fragments(None)


def test_a_window_the_project_cannot_read_is_not_written(project):
    before = project.read_bytes()
    with pytest.raises(ValueError, match="not a run of the project"):
        save_window(project, ("S", "elsewhere.mzML", "0001-0008-VDQIIMAK-z2"), (1.0, 2.0))
    with pytest.raises(ValueError, match="is not read from un1.mzML"):
        save_window(project, ("S", "un1.mzML", "0001-0008-VDQIIMAK-z4"), (1.0, 2.0))
    assert project.read_bytes() == before
    assert not (project.parent / "rt_elution_windows.csv").exists()


def test_a_window_saved_for_the_whole_peptide(project):
    """Saved for the whole peptide, the window is the peptide's row in that file (every ion of
    it takes it); the ion's own row gives up its window (kept only for a drift range); other
    ions of the peptide with windows of their own are named."""
    from hxex.io.elution_windows import read_elution_windows

    spectra = _read(project)
    i = [k for k, s in enumerate(spectra.spectrum_id) if "/un2.mzML/" in s][0]
    key = _key(spectra, i)
    peptide = f"0001-{N:04d}-{PEPTIDE}"
    window = (spectra.rt_start[i] + 0.02, spectra.rt_end[i] - 0.01)
    own = save_window(project, key, window)  # first the ion's own window
    assert saved_window(project, key, peptide)[2] == "ion"
    whole = save_window(project, key, window, peptide=peptide, whole=True,
                        siblings=(key[2], "x-z3"))
    assert whole.added and whole.row["peptide"] == peptide and not whole.row["ion"]
    rows = read_elution_windows(whole.path)
    assert [w.is_peptide for w in rows] == [True]  # the ion's row went (nothing left in it)
    got = saved_window(project, key, peptide)
    assert got[2] == "peptide" and got[1] is None
    assert got[0] == (float(whole.row["rt_start"]), float(whole.row["rt_end"]))
    again = _read(project)
    j = [k for k, s in enumerate(again.spectrum_id) if "/un2.mzML/" in s][0]
    assert got[0][0] <= again.rt_start[j] < again.rt_end[j] <= got[0][1]
    undo_window(whole)
    assert saved_window(project, key, peptide)[2] == "ion"
    undo_window(own)
    # for this ion only: the row names its peptide too (two parents can give fragment ions of
    # the same name)
    alone = save_window(project, key, window, peptide=peptide)
    assert alone.row["peptide"] == peptide and alone.row["ion"] == key[2]
    assert saved_window(project, key, peptide)[2] == "ion"
    assert saved_window(project, key, "0002-0009-OTHER") is None
    undo_window(alone)


def test_the_new_windows_axis_does_not_move_when_its_fit_arrives(project):
    """The new window's spectrum is drawn picked, then fitted: its y axis comes from the data
    (raw profile and picked values), so the fit does not move it (owner, 2026-11-12: it
    changed twice after every new drift range)."""
    from hxex.report.retention import plot_window_edit

    spectra = _read(project)
    results = _results(spectra)
    reader = TraceReader(spectra.source, data_root=project.parent)
    sid, did = spectra.spectrum_id[0], int(spectra.data_id[0])
    window = (spectra.rt_start[0], spectra.rt_end[0] - 0.01)
    summed = reader.spectrum(sid, did, window)
    picked = preview(results, summed, fit=False)
    fitted = preview(results, summed, start_from=sid)
    traces = reader.traces([sid], did)

    def new_window_axis(shown):
        fig = plot_window_edit(traces, results, sid, shown, edit=window)
        ax = next(a for a in fig.axes if a.get_title().startswith("new window"))
        return ax.get_ylim()

    # from 0 in both (the picked-only draw had matplotlib's margin below 0: owner, the y-min)
    assert new_window_axis(picked) == pytest.approx(new_window_axis(fitted))
    assert new_window_axis(picked)[0] == 0.0
