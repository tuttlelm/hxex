# SPDX-License-Identifier: GPL-3.0-or-later
"""Results bundle: exact round trips, integrity, zip, laziness and the review layer."""

import dataclasses
import json
import shutil
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import hxex.io.bundle as bundle_module
from hxex import chem
from hxex.bootstrap import BootstrapConfig, bootstrap
from hxex.fit import FitConfig, Measurements, fit_models
from hxex.io import BundleError, ReviewTable, open_bundle, write_bundle
from hxex.model import Backend
from hxex.noise import NoiseConfig
from hxex.picking import PickerConfig, pick_spectra, plan_bins
from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet
from test_picking import sticks  # noqa: I001

PEPTIDE = "RDKVQKEYAL"
CFG = PickerConfig(zero_filling=3, noise=100.0)


def envelope(plan, centre, width, height):
    bins = np.arange(len(plan.pred_mz))
    return {int(i): float(height * np.exp(-0.5 * ((i - centre) / width) ** 2))
            for i in bins if height * np.exp(-0.5 * ((i - centre) / width) ** 2) > 20}


def natural(plan, height):
    """The peptide's natural isotope envelope (an undeuterated control) with tallest peak
    ``height``: a spectrum the model can describe exactly, unlike a Gaussian near bin 0."""
    shape = chem.isotope_clusters(PEPTIDE)[0]
    shape = shape / shape.max()
    return {int(i): float(height * shape[i]) for i in range(min(len(shape), len(plan.pred_mz)))
            if height * shape[i] > 20}


def build_run(project="proj", n_boot=4):
    plan = plan_bins(PEPTIDE, 2, config=CFG)
    entry = PeptideEntry("f.xlsx", "S", PEPTIDE, "0001-0010", 1, 10, 2)
    layout = [(0.0, 1, 0.5, 0.8), (0.0, 2, 0.5, 0.85), (0.0, 3, 0.5, 0.8), (60.0, 1, 3.0, 1.2),
              (60.0, 2, 3.1, 1.2), (60.0, 3, 2.9, 1.1), (1e6, 1, 8.0, 1.3), (1e6, 2, 8.1, 1.3)]
    spectra = []
    for k, (t, rep, centre, width) in enumerate(layout):
        height = 4e4 * (0.9 + 0.1 * (k % 3))
        peaks = natural(plan, height) if t == 0.0 else envelope(plan, centre, width, height)
        mz, y = sticks(plan, peaks)
        y = y + np.random.default_rng(k).normal(0, 15, len(y))
        spectra.append(Spectrum(mz, y, entry, t, rep, data_id=0, label=f"c{k}",
                                spectrum_id=f"{project}:f.xlsx#{k + 1}"))
    raw = SpectrumSet.from_spectra(spectra)
    binned = pick_spectra(raw, CFG)
    meas = Measurements.from_binned(binned)
    fit = fit_models(meas, FitConfig(
        k_max=2, adam_steps=100, n_restarts=2, backend=Backend("cpu", "float64"),
        noise=NoiseConfig("gaussian", percent=3.0, estimate="replicates", min_replicated=2)))
    boot = bootstrap(meas, fit, BootstrapConfig(n_boot=n_boot, n_restarts=1, adam_steps=40,
                                                polish_steps=0, keep_draws=True))
    return raw, binned, meas, fit, boot


@pytest.fixture(scope="module")
def run():
    return build_run()


@pytest.fixture
def written(tmp_path, run):
    raw, binned, meas, fit, boot = run
    path = write_bundle(tmp_path / "b", measurements=meas, fit=fit, bootstrap=boot, binned=binned,
                        spectra=raw, picker_config=CFG)
    return path


def assert_arrays_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype.kind == b.dtype.kind
    np.testing.assert_array_equal(a, b)  # NaN == NaN here


def assert_fit_equal(a, b):
    assert a.config == b.config and a.k_values == b.k_values
    for name in ("spectrum_id", "data_id", "time", "selected_K", "score", "obs_scale",
                 "sample_size", "n_max", "noise_kind"):
        assert_arrays_equal(getattr(a, name), getattr(b, name))
    assert set(a.fits) == set(b.fits)
    for k in a.fits:
        for name in ("A", "n", "p", "log_A", "n_raw", "logit_p", "nll", "bic", "aic",
                     "passes_min_pop"):
            assert_arrays_equal(getattr(a.fits[k], name), getattr(b.fits[k], name))
    assert (a.noise_estimate is None) == (b.noise_estimate is None)
    if a.noise_estimate is not None:
        for name in ("gain", "estimated", "reason", "score", "outlier", "n_replicated"):
            assert_arrays_equal(getattr(a.noise_estimate, name), getattr(b.noise_estimate, name))


# --- round trips ----------------------------------------------------------------------------

def test_the_run_has_what_the_bundle_must_keep(run):
    raw, binned, meas, fit, boot = run
    assert fit.noise_estimate is not None and fit.noise_estimate.estimated.all()
    assert set(fit.noise_kind) == {"gaussian"} and boot.draws is not None
    assert meas.references[0] == (0, 1, 2)  # the replicate undeuterated controls


def test_everything_round_trips_exactly(written, run):
    raw, binned, meas, fit, boot = run
    with open_bundle(written) as b:
        got = b.measurements()
        for name in ("n_max", "spectrum_id", "data_id", "time"):
            assert_arrays_equal(getattr(got, name), getattr(meas, name))
        assert got.references == meas.references
        for x, y in zip(got.bins + got.observed + got.shapes,
                        meas.bins + meas.observed + meas.shapes, strict=True):
            assert_arrays_equal(x, y)
        assert_fit_equal(b.fit(), fit)
        got_b = b.bootstrap()
        assert got_b.config == boot.config
        for name in ("selected_K", "n_success", "A_se", "n_se", "p_se", "mean_se", "population_se",
                     "A_ci", "n_ci", "p_ci", "mean_ci", "population_ci", "curve_lo", "curve_hi",
                     "curve_offsets"):
            assert_arrays_equal(getattr(got_b, name), getattr(boot, name))
        for name in ("A", "n", "p", "ok"):
            assert_arrays_equal(getattr(got_b.draws, name), getattr(boot.draws, name))
        got_binned = b.binned()
        for name in ("mz", "intensity", "n_deut", "offsets", "env_width", "env_symm",
                     "max_namides", "threshold", "empty", "no_signal", "capped", "spectrum_id",
                     "sample", "peptide", "peptide_range", "file", "label", "modification",
                     "start_seq", "end_seq", "charge", "rep", "data_id", "time"):
            assert_arrays_equal(getattr(got_binned, name), getattr(binned, name))
        got_raw = b.spectra()
        for name in ("mz", "intensity", "offsets", "spectrum_id", "time", "rep"):
            assert_arrays_equal(getattr(got_raw, name), getattr(raw, name))


def test_references_and_missing_references_survive(tmp_path, run):
    raw, binned, meas, fit, boot = run
    refs = tuple((0, 1) if i % 2 == 0 else None for i in range(len(meas)))
    altered = Measurements(meas.bins, meas.observed, meas.shapes, meas.n_max, refs,
                           meas.spectrum_id, meas.data_id, meas.time)
    path = write_bundle(tmp_path / "r", measurements=altered, fit=fit)
    with open_bundle(path) as b:
        assert b.measurements().references == refs


def test_a_zip_loads_like_the_directory(tmp_path, written, run):
    raw, binned, meas, fit, boot = run
    z = write_bundle(tmp_path / "b.zip", measurements=meas, fit=fit, bootstrap=boot,
                     binned=binned, spectra=raw)
    assert z.suffix == ".zip" and zipfile.is_zipfile(z)
    with open_bundle(z) as b, open_bundle(written) as d:
        assert_fit_equal(b.fit(), d.fit())
        assert_arrays_equal(b.spectra().intensity, d.spectra().intensity)
        b.verify()
        with pytest.raises(BundleError, match="read only"):
            b.set_review(ReviewTable.empty())


def test_optional_parts_can_be_left_out(tmp_path, run):
    _, _, meas, fit, _ = run
    path = write_bundle(tmp_path / "min", measurements=meas, fit=fit)
    with open_bundle(path) as b:
        assert b.bootstrap() is None and b.binned() is None and b.spectra() is None
        assert b.tables() == ["components", "fit", "fit_k", "measurements", "noise_estimate"]
        with pytest.raises(BundleError, match="no 'binned' table"):
            b._table("binned")
        assert_fit_equal(b.fit(), fit)


# --- manifest and integrity -----------------------------------------------------------------

def test_the_manifest(written, run):
    raw, binned, meas, fit, boot = run
    with open_bundle(written) as b:
        m = b.manifest
        assert m["kind"] == "hxex-results-bundle" and m["schema_version"] == 2
        assert b.project == "proj" and m["n_measurements"] == len(meas)
        assert m["k_values"] == [1, 2] and set(m["versions"]) >= {"hxex", "torch", "numpy"}
        assert m["configs"]["fit"] == fit.config.to_dict()
        assert m["configs"]["picker"] == CFG.to_dict()
        assert m["configs"]["bootstrap"] == boot.config.to_dict()
        assert m["files"]["components.parquet"]["derived"] is True
        assert all(len(v["sha256"]) == 64 for v in m["files"].values())
        b.verify()
        assert b.tables() == sorted(["measurements", "fit", "fit_k", "noise_estimate", "bootstrap",
                                     "bootstrap_draws", "binned", "spectra", "components"])


def test_verify_detects_damage_and_loss(written):
    path = written / "fit_k.parquet"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with open_bundle(written) as b, pytest.raises(BundleError, match="checksum"):
        b.verify()
    path.unlink()
    with open_bundle(written) as b, pytest.raises(BundleError, match="missing"):
        b.verify()


def test_unreadable_and_newer_bundles(tmp_path, written):
    with pytest.raises(BundleError, match="not a bundle"):
        open_bundle(tmp_path / "nothing")
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "manifest.json").write_text("{not json")
    with pytest.raises(BundleError, match="cannot read the manifest"):
        open_bundle(tmp_path / "junk")
    manifest = json.loads((written / "manifest.json").read_text())
    (written / "manifest.json").write_text(json.dumps({**manifest, "schema_version": 99}))
    with pytest.raises(BundleError, match="schema version 99"):
        open_bundle(written)
    (written / "manifest.json").write_text(json.dumps({**manifest, "kind": "other"}))
    with pytest.raises(BundleError, match="not an hxex results bundle"):
        open_bundle(written)


def test_unknown_tables_are_ignored(written, run):
    shutil.copy(written / "fit.parquet", written / "future.parquet")
    manifest = json.loads((written / "manifest.json").read_text())
    manifest["files"]["future.parquet"] = {"rows": 1, "bytes": 1, "sha256": "0" * 64}
    (written / "manifest.json").write_text(json.dumps(manifest))
    with open_bundle(written) as b:
        assert_fit_equal(b.fit(), run[3])


# --- writing safely -------------------------------------------------------------------------

def test_existing_paths_are_protected(tmp_path, run):
    _, _, meas, fit, _ = run
    path = write_bundle(tmp_path / "b", measurements=meas, fit=fit)
    with pytest.raises(BundleError, match="exists"):
        write_bundle(path, measurements=meas, fit=fit)
    write_bundle(path, measurements=meas, fit=fit, overwrite=True, notes="again")
    with open_bundle(path) as b:
        assert b.manifest["notes"] == "again"
    other = tmp_path / "data"
    other.mkdir()
    (other / "keep.txt").write_text("do not delete")
    with pytest.raises(BundleError, match="not a results bundle"):
        write_bundle(other, measurements=meas, fit=fit, overwrite=True)
    assert (other / "keep.txt").exists()


def test_overwriting_keeps_files_of_your_own(tmp_path, run):
    _, _, meas, fit, boot = run
    path = write_bundle(tmp_path / "b", measurements=meas, fit=fit, bootstrap=boot)
    assert (path / "bootstrap.parquet").exists()
    (path / "summary.md").write_text("my notes")
    (path / "extra").mkdir()
    (path / "extra" / "figure.png").write_bytes(b"png")
    write_bundle(path, measurements=meas, fit=fit, overwrite=True, notes="again")
    assert (path / "summary.md").read_text() == "my notes"
    assert (path / "extra" / "figure.png").exists()
    # a table the new run does not write is gone, not left stale
    assert not (path / "bootstrap.parquet").exists()
    assert not list(tmp_path.glob("b.tmp-*"))
    with open_bundle(path) as b:
        assert b.manifest["notes"] == "again" and not b.has("bootstrap")
        b.verify()
    # a file of your own named like one of the new bundle's is refused, nothing changed
    (path / "bootstrap.parquet").write_text("mine")
    with pytest.raises(BundleError, match="of its own"):
        write_bundle(path, measurements=meas, fit=fit, bootstrap=boot, overwrite=True)
    assert (path / "bootstrap.parquet").read_text() == "mine"
    with open_bundle(path) as b:
        assert b.manifest["notes"] == "again"
    assert not list(tmp_path.glob("b.tmp-*"))


def test_an_interrupted_replacement_is_detected_and_rewritten(tmp_path, run):
    _, _, meas, fit, _ = run
    path = write_bundle(tmp_path / "b", measurements=meas, fit=fit)
    (path / "manifest.json").rename(path / "manifest.json.replacing")
    (path / "fit.parquet").unlink()
    with pytest.raises(BundleError, match="interrupted"):
        open_bundle(path)
    write_bundle(path, measurements=meas, fit=fit, overwrite=True, notes="fixed")
    assert not (path / "manifest.json.replacing").exists()
    with open_bundle(path) as b:
        assert b.manifest["notes"] == "fixed"
        b.verify()


def test_a_failed_write_leaves_nothing(tmp_path, run, monkeypatch):
    _, _, meas, fit, boot = run
    with pytest.raises(BundleError, match="same spectra"):
        write_bundle(tmp_path / "x", measurements=meas.__class__(
            meas.bins[:2], meas.observed[:2], meas.shapes[:2], meas.n_max[:2], meas.references[:2],
            meas.spectrum_id[:2], meas.data_id[:2], meas.time[:2]), fit=fit)
    monkeypatch.setattr(bundle_module, "_components_table",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        write_bundle(tmp_path / "y", measurements=meas, fit=fit, bootstrap=boot)
    assert list(tmp_path.iterdir()) == []  # neither the bundle nor a temporary directory


def test_inconsistent_parts_are_refused(tmp_path, run):
    raw, binned, meas, fit, boot = run
    with pytest.raises(BundleError, match="not made from this fit"):
        write_bundle(tmp_path / "a", measurements=meas, fit=fit, bootstrap=type(boot)(
            **{**boot.__dict__, "selected_K": boot.selected_K + 1}))
    with pytest.raises(BundleError, match="must be in the binned table"):
        write_bundle(tmp_path / "b", measurements=meas, fit=fit, binned=binned.select(
            np.arange(len(binned)) < 3))


# --- laziness, types and the tidy table -----------------------------------------------------

def test_spectra_can_be_read_by_id(written, run, monkeypatch):
    raw = run[0]
    ids = [str(raw.spectrum_id[5]), str(raw.spectrum_id[1])]
    with open_bundle(written) as b:
        sub = b.spectra(ids)
        assert sorted(sub.spectrum_id) == sorted(ids)
        for sid in ids:
            i, j = raw.find(sid), sub.find(sid)
            assert_arrays_equal(sub.spectrum(j)[0], raw.spectrum(i)[0])
            assert_arrays_equal(sub.spectrum(j)[1], raw.spectrum(i)[1])
        assert len(b.spectra([])) == 0


def test_many_row_groups(tmp_path, run, monkeypatch):
    raw, binned, meas, fit, boot = run
    monkeypatch.setattr(bundle_module, "ROW_GROUP", 3)
    path = write_bundle(tmp_path / "rg", measurements=meas, fit=fit, spectra=raw, binned=binned)
    assert pq.ParquetFile(path / "spectra.parquet").num_row_groups == 3  # 8 spectra, 3 per group
    with open_bundle(path) as b:
        assert_arrays_equal(b.spectra().intensity, raw.intensity)
        assert list(b.spectra([str(raw.spectrum_id[7])]).spectrum_id) == [str(raw.spectrum_id[7])]


def test_the_files_have_plain_arrow_types(written):
    types = lambda name: {f.name: str(f.type) for f in pq.read_schema(written / name)}  # noqa: E731
    t = types("measurements.parquet")
    assert t["bins"] == "list<element: double>" and t["spectrum_id"] == "string"
    assert t["references"] == "list<element: int64>" and t["measurement"] == "int64"
    f = types("fit_k.parquet")
    assert f["A"] == "list<element: double>" and f["fit_K"] == "int64"
    assert f["passes_min_pop"] == "bool"
    assert types("binned.parquet")["n_deut"] == "list<element: int64>"
    column = pq.ParquetFile(written / "fit.parquet").metadata.row_group(0).column(0)
    assert column.compression == "ZSTD"


def test_the_tidy_table_matches_the_run(written, run):
    raw, binned, meas, fit, boot = run
    expected = boot.table(fit, selected_only=False)
    with open_bundle(written) as b:
        got = b.components_table()
    assert set(got) == set(expected)
    for name, values in expected.items():
        if values.dtype.kind in "US":
            assert list(got[name]) == list(values)
        else:
            np.testing.assert_array_equal(np.asarray(got[name], dtype=float),
                                          np.asarray(values, dtype=float))


def test_project_name(tmp_path, run):
    _, _, meas, fit, _ = run
    path = write_bundle(tmp_path / "p", measurements=meas, fit=fit, project="myproj")
    with open_bundle(path) as b:
        assert b.project == "myproj"


# --- the review layer -----------------------------------------------------------------------

def test_review_table_operations(run):
    fit = run[3]
    ids = [str(s) for s in fit.spectrum_id]
    r = ReviewTable.empty()
    assert len(r) == 0 and r.get(ids[0]) is None
    r = r.update(ids[0], status="rejected", note="spike at bin 3")
    r = r.update(ids[1], k_override=1)
    r = r.update(ids[0], note="spike at bin 3, contaminant")  # only the note changes
    assert r.get(ids[0])["status"] == "rejected" and r.get(ids[0])["note"].endswith("contaminant")
    assert r.get(ids[1])["k_override"] == 1
    assert list(r.effective_K(fit))[:2] == [fit.selected_K[0], 1]
    status = r.status_of(fit)
    assert status[0] == "rejected" and status[1] == "unreviewed" and status[2] == "unreviewed"
    cleared = r.update(ids[0], status="", note="").update(ids[1], k_override=0)
    assert len(cleared) == 0
    with pytest.raises(ValueError, match="status"):
        r.update(ids[0], status="maybe")
    with pytest.raises(ValueError, match="k_override"):
        r.update(ids[0], k_override=-1)
    with pytest.raises(ValueError, match="was not fitted"):
        r.update(ids[2], k_override=7).effective_K(fit)


def test_review_is_stored_with_the_bundle_and_needs_no_refit(written, run):
    fit = run[3]
    ids = [str(s) for s in fit.spectrum_id]
    with open_bundle(written) as b:
        assert len(b.review()) == 0
        before = (written / "fit_k.parquet").read_bytes()
        review = b.review().update(ids[3], status="accepted", note="checked by eye").update(
            ids[4], k_override=1)
        b.set_review(review)
        b.verify()  # the manifest was updated with the new table
    assert (written / "fit_k.parquet").read_bytes() == before  # the fit is untouched
    with open_bundle(written) as b:
        got = b.review()
        assert got.get(ids[3])["status"] == "accepted" and got.get(ids[4])["k_override"] == 1
        assert got.effective_K(b.fit())[4] == 1
        assert_fit_equal(b.fit(), fit)
        b.verify()


def test_review_is_validated_against_the_fit(written, run):
    with open_bundle(written) as b:
        with pytest.raises(BundleError, match="unknown spectra"):
            b.set_review(ReviewTable.empty().update("proj:not-there", status="rejected"))
        bad = ReviewTable.empty().update(str(b.fit().spectrum_id[0]), k_override=9)
        with pytest.raises(ValueError, match="was not fitted"):
            b.set_review(bad)


def test_a_review_given_at_writing(tmp_path, run):
    _, _, meas, fit, _ = run
    r = ReviewTable.empty().update(str(fit.spectrum_id[2]), status="rejected")
    with open_bundle(write_bundle(tmp_path / "rv", measurements=meas, fit=fit, review=r)) as b:
        assert b.review().get(str(fit.spectrum_id[2]))["status"] == "rejected"
    # checked as set_review checks: unknown spectra and a K that was not fitted are refused
    with pytest.raises(BundleError, match="unknown spectra"):
        write_bundle(tmp_path / "rv2", measurements=meas, fit=fit,
                     review=ReviewTable.empty().update("proj:not-there", status="rejected"))
    bad = ReviewTable.empty().update(str(fit.spectrum_id[2]), k_override=max(fit.k_values) + 1)
    with pytest.raises(ValueError, match="was not fitted"):
        write_bundle(tmp_path / "rv3", measurements=meas, fit=fit, review=bad)
    assert not (tmp_path / "rv3").exists()


# --- controls -------------------------------------------------------------------------------

def test_controls_and_corrected_means_are_kept(tmp_path, run):
    from hxex.controls import ControlConfig, fit_controls
    raw, binned, meas, fit, boot = run
    cfg = ControlConfig(undeut_time=0.0, fd_time=1e6)
    ctrl = fit_controls(fit, binned.max_namides, cfg)
    path = write_bundle(tmp_path / "c", measurements=meas, fit=fit, bootstrap=boot,
                        controls=ctrl)
    with open_bundle(path) as b:
        got = b.controls()
        assert got.config == cfg
        for name in ("data_id", "n_amides", "n_undeut", "n_fd", "m0_measured", "mfd_measured",
                     "m0", "m0_se", "mfd", "mfd_se", "retention", "fallback", "flags"):
            assert_arrays_equal(getattr(got, name), getattr(ctrl, name))
        comp = b.components_table()
        i = np.flatnonzero(comp["is_selected"])
        assert np.all(comp["mean_corrected"] >= 0) and np.isfinite(comp["mean_corrected"][i]).all()
        np.testing.assert_allclose(
            comp["mean_corrected"][i],
            np.maximum((comp["mean"][i] - ctrl.m0[0]) / (ctrl.mfd[0] - ctrl.m0[0])
                       * ctrl.n_amides[0], 0))
        assert "mean" in comp and "mean_corrected_ci_lo" in comp  # raw values stay
        assert (comp["mean_corrected_ci_lo"][i] <= comp["mean_corrected"][i] + 1e-9).all()
    with open_bundle(written_no_controls(tmp_path, run)) as b:
        assert b.controls() is None and "mean_corrected" not in b.components_table()


def written_no_controls(tmp_path, run):
    raw, binned, meas, fit, boot = run
    return write_bundle(tmp_path / "n", measurements=meas, fit=fit, bootstrap=boot)


def test_quality_flags_are_kept(tmp_path, run):
    from hxex.quality import QualityConfig, assess_quality
    raw, binned, meas, fit, boot = run
    q = assess_quality(meas, fit, QualityConfig(min_intensity=0.2))
    path = write_bundle(tmp_path / "q", measurements=meas, fit=fit, quality=q)
    with open_bundle(path) as b:
        got = b.quality()
        assert got.config == q.config
        for name in ("spectrum_id", "deviance", "deviance_rel", "misfit", "misfit_rel",
                     "intensity", "intensity_rel",
                     "poor_fit", "low_intensity", "noise_outlier", "flagged"):
            assert_arrays_equal(getattr(got, name), getattr(q, name))
    with open_bundle(written_no_controls(tmp_path, run)) as b:
        assert b.quality() is None


# --- audit fixes: what overwrite may remove, the review write, list column limits -------------

def test_overwrite_never_removes_what_is_not_a_bundle(tmp_path, run):
    _, _, meas, fit, _ = run
    plain = tmp_path / "b"
    plain.write_text("my data")
    with pytest.raises(BundleError, match="not a results bundle"):
        write_bundle(plain, measurements=meas, fit=fit, overwrite=True)
    assert plain.read_text() == "my data"
    other_zip = tmp_path / "b.zip"
    with zipfile.ZipFile(other_zip, "w") as zf:
        zf.writestr("notes.txt", "mine")
    with pytest.raises(BundleError, match="not a results bundle"):
        write_bundle(other_zip, measurements=meas, fit=fit, overwrite=True)
    with zipfile.ZipFile(other_zip) as zf:
        assert zf.namelist() == ["notes.txt"]
    # a zipped bundle is replaced
    z = write_bundle(tmp_path / "c.zip", measurements=meas, fit=fit)
    write_bundle(z, measurements=meas, fit=fit, overwrite=True, notes="again")
    with open_bundle(z) as b:
        assert b.manifest["notes"] == "again"
    # a bundle directory is not replaced by a zip of the same name
    d = write_bundle(tmp_path / "d", measurements=meas, fit=fit)
    d.rename(tmp_path / "d.zip")
    with pytest.raises(BundleError, match="bundle directory"):
        write_bundle(tmp_path / "d.zip", measurements=meas, fit=fit, overwrite=True)
    assert (tmp_path / "d.zip" / "manifest.json").exists()
    assert not list(tmp_path.glob("*.tmp-*"))


def test_review_is_replaced_whole_or_not_at_all(written, run, monkeypatch):
    ids = [str(s) for s in run[3].spectrum_id]
    with open_bundle(written) as b:
        b.set_review(ReviewTable.empty().update(ids[0], status="accepted"))
        before = (written / "review.parquet").read_bytes()
        manifest = (written / "manifest.json").read_bytes()

        def fail(table, where, **kwargs):
            where.write_bytes(b"half a table")
            raise OSError("disk full")

        monkeypatch.setattr(bundle_module.pq, "write_table", fail)
        with pytest.raises(OSError, match="disk full"):
            b.set_review(ReviewTable.empty().update(ids[1], status="rejected"))
    assert (written / "review.parquet").read_bytes() == before
    assert (written / "manifest.json").read_bytes() == manifest
    assert not list(written.glob("*.tmp"))
    with open_bundle(written) as b:
        assert b.review().get(ids[0])["status"] == "accepted" and b.review().get(ids[1]) is None
        b.verify()


def test_list_columns_refuse_more_values_than_int32_offsets_hold():
    from hxex.io._arrow import _flat_ragged, _ragged

    class Huge:
        def __len__(self):
            return 2**31

    with pytest.raises(ValueError, match="list column"):
        _flat_ragged(np.zeros(2), np.array([0, 2**31]), pa.float64())
    with pytest.raises(ValueError, match="list column"):
        _ragged([np.zeros(1), Huge()], pa.float64())
    assert _flat_ragged(np.zeros(2), np.array([0, 2]), pa.float64()).to_pylist() == [[0.0, 0.0]]
    assert _ragged([np.ones(1), None], pa.float64()).to_pylist() == [[1.0], None]


# --- the overwrite contract: stale review entries, provenance, generations -------------------

def test_stale_review_entries_are_kept_on_record_and_their_k_ignored(tmp_path, run):
    from hxex.io.bundle import _read_review, _review_table

    raw, binned, meas, fit, boot = run
    ids = [str(s) for s in fit.spectrum_id]
    r = ReviewTable.empty().update(ids[0], status="rejected", k_override=2, note="two")
    r = r.update("proj:gone#9", status="accepted", note="was fine").update(ids[1], k_override=2)
    assert not r.stale.any() and r.n_stale == 0
    with pytest.raises(BundleError, match="unknown spectra"):
        write_bundle(tmp_path / "a", measurements=meas, fit=fit, review=r)
    marked = r.mark_stale(fit)
    assert marked.get("proj:gone#9")["stale"] and not marked.get(ids[0])["stale"]
    assert marked.n_stale == 1
    # a chosen K that a K = 1 only fit cannot hold is stale too, and ignored
    one = fit_models(meas, dataclasses.replace(fit.config, k_max=1))
    assert one.k_values == (1,)
    marked_one = r.mark_stale(one)
    assert marked_one.n_stale == 3 and (marked_one.effective_K(one) == 1).all()
    assert (marked_one.status_of(one) == "rejected").sum() == 1  # the status still counts
    # stored and read back with the marker; the current entry is checked as before
    path = write_bundle(tmp_path / "b", measurements=meas, fit=fit, review=marked)
    with open_bundle(path) as b:
        got = b.review()
        assert got.stale.tolist() == marked.stale.tolist() and got.n_stale == 1
        assert (got.effective_K(fit)[0] == 2) and got.get(ids[1])["k_override"] == 2
        # a change to a stale entry makes it current: it must then apply, or is refused
        with pytest.raises(BundleError, match="unknown spectra"):
            b.set_review(got.update("proj:gone#9", note="looked again"))
        b.set_review(got.update("proj:gone#9", status="", note=""))  # cleared: gone
        assert b.review().get("proj:gone#9") is None
    # a review table written before the marker existed reads as all current
    table = _review_table(marked).drop_columns(["stale"])
    old = _read_review(table)
    assert not old.stale.any() and len(old) == 3
    with pytest.raises(ValueError, match="one value per entry"):
        ReviewTable(old.spectrum_id, old.status, old.k_override, old.note, old.updated,
                    np.array([True]))


def test_provenance_is_recorded_as_given_and_settings_compared(tmp_path, run):
    from hxex.io import bundle_manifest, provenance_of
    from hxex.io.bundle import settings_changed

    raw, binned, meas, fit, boot = run
    path = write_bundle(tmp_path / "p", measurements=meas, fit=fit)
    old = bundle_manifest(path)
    assert old["created"] and old["provenance"] is None
    assert bundle_manifest(tmp_path / "nothing") is None
    (tmp_path / "plain").write_text("x")
    assert bundle_manifest(tmp_path / "plain") is None
    review = ReviewTable.empty().update(str(fit.spectrum_id[0]), status="accepted")
    review = review.update("proj:gone#1", note="n").mark_stale(fit)
    new_configs = {"run": {"fit": {"k_max": 3, "screen": True}, "picker": {"noise": 100.0}},
                   "spectra_source": {"files": ["a"], "hxex.created": "later"}}
    prov = provenance_of({**old, "configs": {
        "run": {"fit": {"k_max": 2, "screen": True}, "picker": {"noise": 100.0},
                "quality": {"max_ppm": 20.0}},
        "spectra_source": {"files": ["a"], "hxex.created": "earlier"}}},
        path, new_configs=new_configs, review=review, moved_to=None)
    assert prov["replaced"] == str(path) and prov["replaced_created"] == old["created"]
    assert prov["settings_changed"] == ["fit.k_max", "quality"]
    assert prov["review"] == {"carried": 2, "stale": 1} and prov["moved_to"] is None
    # without run settings on one side the stage settings are compared; the spectra source
    # counts once, whatever differs in it
    assert settings_changed({"fit": {"k_max": 2}, "spectra_source": {"files": ["a"]}},
                            {"fit": {"k_max": 2}, "spectra_source": {"files": ["b"]}}) == [
        "spectra_source"]
    assert settings_changed({"fit": {"k_max": 2}}, {"fit": {"k_max": 2, "screen": True},
                                                    "bootstrap": {"n_boot": 2}}) == [
        "bootstrap", "fit.screen"]
    assert settings_changed(None, None) == []
    again = write_bundle(tmp_path / "p", measurements=meas, fit=fit, overwrite=True,
                         provenance=prov)
    with open_bundle(again) as b:
        assert b.manifest["provenance"] == prov
    z = write_bundle(tmp_path / "p.zip", measurements=meas, fit=fit, provenance=prov)
    assert bundle_manifest(z)["provenance"] == prov


def test_generations_are_numbered_once_and_found(tmp_path, run):
    from hxex.io import generations, keep_previous

    raw, binned, meas, fit, boot = run
    assert keep_previous(tmp_path / "none") is None
    path = write_bundle(tmp_path / "g", measurements=meas, fit=fit, notes="one")
    (tmp_path / "g_figures").mkdir()
    (tmp_path / "g_figures" / "f.pdf").write_bytes(b"pdf")
    assert keep_previous(path) == tmp_path / "g.1"
    assert (tmp_path / "g.1_figures" / "f.pdf").exists() and not path.exists()
    write_bundle(path, measurements=meas, fit=fit, notes="two")
    (tmp_path / "g.7").write_text("a file of yours with a generation's name")
    assert keep_previous(path) == tmp_path / "g.8"  # after every number used, bundle or not
    assert generations(path) == [(1, tmp_path / "g.1"), (8, tmp_path / "g.8")]
    with open_bundle(tmp_path / "g.8") as b:
        assert b.manifest["notes"] == "two"
    # a zip bundle: name.N.zip, and its folders name.N.zip_tables (the bundle name plus suffix)
    z = write_bundle(tmp_path / "z.zip", measurements=meas, fit=fit)
    (tmp_path / "z.zip_tables").mkdir()
    assert keep_previous(z) == tmp_path / "z.1.zip"
    assert (tmp_path / "z.1.zip_tables").is_dir() and generations(z) == [(1, tmp_path / "z.1.zip")]
    (tmp_path / "plain").write_text("x")
    with pytest.raises(BundleError, match="not a results bundle"):
        keep_previous(tmp_path / "plain")
