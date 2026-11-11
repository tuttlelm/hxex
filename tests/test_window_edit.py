# SPDX-License-Identifier: GPL-3.0-or-later
"""A spectrum's retention window changed in the review (hxex.gui.window_edit; PLAN Next steps
26): the preview picked and fitted with the run's settings, and the window written as the run's
own row of the peptide table, which the next read takes (one series still)."""

import numpy as np
import pytest

from hxex.config import load_project, read_input
from hxex.fit import FitConfig
from hxex.gui.window_edit import preview, save_window, saved_window, undo_window
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
        assert any("from the run's fit" in n for n in shown.notes)
    # a spectrum the results do not hold: the run's own starts
    assert not any("from the run's fit" in n
                   for n in preview(results, same, start_from="p:nothing").notes)


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
    assert any("from the spectrum itself" in n for n in shown.notes)


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
