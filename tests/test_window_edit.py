# SPDX-License-Identifier: GPL-3.0-or-later
"""A spectrum's retention window changed in the review (hxex.gui.window_edit; PLAN Next steps
26): the preview picked and fitted with the run's settings, and the window written as the run's
own row of the peptide table, which the next read takes (one series still)."""

import numpy as np
import pytest

from hxex.config import load_project, scan_targets
from hxex.fit import FitConfig
from hxex.gui.window_edit import preview, save_window, undo_window
from hxex.io import read_mzml
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
    p = load_project(path)
    return read_mzml(p.input.runs, scan_targets(p), project="p", data_root=path.parent)


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


def test_a_window_is_written_as_the_runs_row_and_read_back(project):
    spectra = _read(project)
    table = project.parent / "peptides.csv"
    original = table.read_bytes()
    reader = TraceReader(spectra.source, data_root=project.parent)
    i = [k for k, s in enumerate(spectra.spectrum_id) if "/un2.mzML/" in s][0]
    sid, data_id = spectra.spectrum_id[i], int(spectra.data_id[i])
    start, end = spectra.rt_start[i] + 0.02, spectra.rt_end[i] - 0.01
    edited = reader.spectrum(sid, data_id, (start, end))
    saved = save_window(project, table, edited, "un2.mzML", (edited.rt_start, edited.rt_end))
    assert saved.added and saved.row["run"] == "un2.mzML" and saved.row["note"] == "kept"
    assert float(saved.row["rt_start"]) <= edited.rt_start <= edited.rt_end <= \
        float(saved.row["rt_end"])
    lines = table.read_text().splitlines()
    assert lines[0] == "Peptide,start_seq,end_seq,charge,rt_start,rt_end,note,run"
    assert len(lines) == 3  # the general row and the run's own after it
    # read again: the run sums the new scans, and both runs stay one series
    again = _read(project)
    assert edited.spectrum_id in list(again.spectrum_id)
    assert len(set(again.data_id.tolist())) == 1
    k = list(again.spectrum_id).index(edited.spectrum_id)
    np.testing.assert_allclose(again.spectrum(k)[1], edited.intensity)
    # saved again: the run's row changes, no second row
    wider = reader.spectrum(sid, data_id, (spectra.rt_start[i], spectra.rt_end[i] + 0.02))
    second = save_window(project, table, wider, "un2.mzML", (wider.rt_start, wider.rt_end))
    assert not second.added and len(table.read_text().splitlines()) == 3
    undo_window(second)
    undo_window(saved)
    assert table.read_bytes() == original
    with pytest.raises(ValueError, match="changed since"):
        undo_window(saved)  # the file is no longer what that save wrote


def test_a_window_the_project_cannot_read_is_not_written(project):
    spectra = _read(project)
    table = project.parent / "peptides.csv"
    original = table.read_bytes()
    reader = TraceReader(spectra.source, data_root=project.parent)
    sid, data_id = spectra.spectrum_id[0], int(spectra.data_id[0])
    edited = reader.spectrum(sid, data_id, (spectra.rt_start[0], spectra.rt_end[0]))
    with pytest.raises(ValueError, match="not the file of any"):
        save_window(project, table, edited, "elsewhere.mzML", (1.0, 2.0))
    assert table.read_bytes() == original
