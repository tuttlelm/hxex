# SPDX-License-Identifier: GPL-3.0-or-later
"""The run stage: dropping flagged spectra, chunking (which must not change the result),
progress and cancellation, and the bundle it writes."""

import logging

import numpy as np
import pytest

from hxex.bootstrap import BootstrapConfig, BootstrapResult
from hxex.controls import ControlConfig
from hxex.fit import FitConfig, FitResult, Measurements
from hxex.hooks import Cancelled
from hxex.io import open_bundle
from hxex.model import Backend
from hxex.noise import NoiseConfig
from hxex.picking import PickerConfig, pick_spectra, plan_bins
from hxex.quality import QualityConfig
from hxex.run import RunConfig, chunk_bytes, plan_chunks, resume_bootstrap, run
from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet
from test_bundle import envelope, natural
from test_picking import sticks

PEPTIDE = "RDKVQKEYAL"
PICK = PickerConfig(zero_filling=3, noise=100.0)
CPU = Backend("cpu", "float64")
LAYOUT = [(0.0, 1, 0.5, 0.8), (0.0, 2, 0.5, 0.85), (0.0, 3, 0.5, 0.8), (60.0, 1, 3.0, 1.2),
          (60.0, 2, 3.1, 1.2), (60.0, 3, 2.9, 1.1), (1e6, 1, 8.0, 1.3), (1e6, 2, 8.1, 1.3)]


def raw_spectra(n_series=3, project="proj", empty=None):
    plan = plan_bins(PEPTIDE, 2, config=PICK)
    spectra = []
    for s in range(n_series):
        entry = PeptideEntry(f"f{s}.xlsx", "S", PEPTIDE, "0001-0010", 1, 10, 2)
        for k, (t, rep, centre, width) in enumerate(LAYOUT):
            height = 4e4 * (0.9 + 0.1 * (k % 3))
            peaks = (natural(plan, height) if t == 0.0
                     else envelope(plan, centre + 0.1 * s, width, height))
            mz, y = sticks(plan, peaks)
            y = y + np.random.default_rng(100 * s + k).normal(0, 15, len(y))
            if empty == (s, k):
                y = np.zeros_like(y)
            spectra.append(Spectrum(mz, y, entry, t, rep, data_id=s, label=f"s{s}c{k}",
                                    spectrum_id=f"{project}:f{s}.xlsx#{k + 1}"))
    return SpectrumSet.from_spectra(spectra)


def config(**kw):
    fit = FitConfig(k_max=2, adam_steps=60, n_restarts=2, backend=CPU,
                    noise=NoiseConfig("gaussian", percent=3.0, estimate="replicates",
                                      min_replicated=2))
    boot = BootstrapConfig(n_boot=3, n_restarts=1, adam_steps=30, polish_steps=0, keep_draws=True)
    base = dict(picker=PICK, fit=fit, bootstrap=boot,
                controls=ControlConfig(undeut_time=0.0, fd_time=1e6))
    return RunConfig(**{**base, **kw})


def measurements(raw):
    return Measurements.from_binned(pick_spectra(raw, PICK), nterm_subtract=PICK.nterm_subtract)


@pytest.fixture(scope="module")
def whole():
    raw = raw_spectra()
    return raw, run(raw, config())


def test_run_gives_every_stage(whole):
    raw, res = whole
    assert res.kept.all() and len(res.fit) == len(raw) == len(res.measurements)
    assert res.bootstrap is not None and len(res.bootstrap) == len(raw)
    assert list(res.controls.data_id) == [0, 1, 2] and res.quality is not None
    assert not res.controls.fallback.any()
    assert (res.controls.n_amides == res.measurements.n_max[0] / 1.2).all()


def test_flagged_spectra_are_dropped_and_recorded():
    raw = raw_spectra(empty=(1, 4))
    res = run(raw, config(bootstrap=None))
    assert res.kept.sum() == len(raw) - 1 and len(res.fit) == len(raw) - 1
    assert list(res.dropped_ids) == ["proj:f1.xlsx#5"]
    assert list(res.dropped_reason[~res.kept]) == ["empty"]
    assert len(res.binned) == len(raw)  # the picked table keeps every spectrum and its flags
    assert "proj:f1.xlsx#5" not in set(res.measurements.spectrum_id)


def test_nothing_left_is_an_error():
    raw = raw_spectra(n_series=1)
    zero = SpectrumSet.from_spectra([
        Spectrum(raw.spectrum(i)[0], np.zeros(len(raw.spectrum(i)[1])),
                 PeptideEntry("f0.xlsx", "S", PEPTIDE, "0001-0010", 1, 10, 2), float(raw.time[i]),
                 int(raw.rep[i]), data_id=0, label="z", spectrum_id=str(raw.spectrum_id[i]))
        for i in range(len(raw))])
    with pytest.raises(ValueError, match="every spectrum was dropped"):
        run(zero, config())


def test_chunking_does_not_change_the_result(whole):
    raw, res = whole
    small = run(raw, config(max_chunk=1))  # one series per chunk: three chunks
    a, b = res.fit, small.fit
    assert np.array_equal(a.spectrum_id, b.spectrum_id)
    assert np.array_equal(a.selected_K, b.selected_K)
    for k in a.k_values:
        np.testing.assert_allclose(b.fits[k].nll, a.fits[k].nll, rtol=1e-4, atol=1e-4)
        # padding to another width changes the rounding, and a short fit amplifies it (1e-3 D)
        np.testing.assert_allclose(b.fits[k].means, a.fits[k].means, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(b.obs_scale, a.obs_scale, rtol=1e-9)
    ba, bb = res.bootstrap, small.bootstrap
    np.testing.assert_allclose(bb.mean_se, ba.mean_se, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(bb.curve_offsets, ba.curve_offsets)
    np.testing.assert_allclose(bb.curve_hi, ba.curve_hi, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(small.controls.m0, res.controls.m0, atol=1e-5)
    np.testing.assert_array_equal(small.quality.poor_fit, res.quality.poor_fit)


def test_plan_chunks_cover_whole_series_within_the_budget():
    raw = raw_spectra(n_series=5)
    meas = measurements(raw)
    one = chunk_bytes(max(len(b) for b in meas.bins), max(len(s) for s in meas.shapes), 2, 16, 8)
    chunks = plan_chunks(meas, 2, 8, budget=one)
    assert np.array_equal(np.sort(np.concatenate(chunks)), np.arange(len(meas)))
    for rows in chunks:
        assert len(rows) in (8, 16)  # whole series of 8
        assert len(set(meas.data_id[rows])) == len(rows) // 8
    assert len(chunks) == 3  # five series, two per chunk
    assert len(plan_chunks(meas, 2, 8, budget=1e12)) == 1
    assert len(plan_chunks(meas, 2, 8, budget=1e12, max_chunk=8)) == 5


def test_a_series_over_the_budget_is_its_own_chunk_with_a_warning(caplog):
    raw = raw_spectra(n_series=2)
    meas = measurements(raw)
    with caplog.at_level(logging.WARNING, logger="hxex.run"):
        chunks = plan_chunks(meas, 2, 8, budget=1.0)
    assert len(chunks) == 2 and "cannot be split" in caplog.text


def test_progress_is_monotone_to_the_end_and_cancel_stops_cleanly(tmp_path):
    raw = raw_spectra(n_series=2)
    seen = []
    run(raw, config(max_chunk=1), progress=lambda d, t: seen.append((d, t)))
    done = [d for d, _ in seen]
    assert done == sorted(done) and seen[-1] == (1000, 1000) and {t for _, t in seen} == {1000}
    stop = []
    out = tmp_path / "never"
    with pytest.raises(Cancelled):
        run(raw, config(), out=out, cancel=lambda: len(stop) > 0 or stop.append(1) or False)
    assert not out.exists()


def test_the_stages_are_named_in_order_and_cover_the_units(tmp_path):
    class Recorder:
        def __init__(self):
            self.calls = []

        def start_stage(self, name, start, end):
            self.calls.append(("stage", name, start, end))

        def __call__(self, done, total):
            self.calls.append(("done", done, total))

    rec = Recorder()
    run(raw_spectra(n_series=2), config(max_chunk=1), out=tmp_path / "b", progress=rec)
    stages = [c for c in rec.calls if c[0] == "stage"]
    assert [c[1] for c in stages] == ["pick", "fit", "controls and quality", "write fits",
                                      "bootstrap", "write"]
    ends = [(c[2], c[3]) for c in stages]
    assert ends[0][0] == 0 and all(a <= b for a, b in ends)
    assert all(e[1] <= f[0] + 1e-9 for e, f in zip(ends, ends[1:], strict=False))
    # every report lies in the stage it belongs to (to the rounding of the units), and only the
    # very last one reaches the total (so a display does not finish before writing)
    stage = None
    for c in rec.calls:
        if c[0] == "stage":
            stage = c
        else:
            assert stage[2] - 1 <= c[1] <= stage[3] + 1
    done = [c[1] for c in rec.calls if c[0] == "done"]
    assert done.count(1000) == 1 and rec.calls[-1] == ("done", 1000, 1000)


def test_the_bundle_is_written_and_records_the_run(tmp_path):
    raw = raw_spectra(n_series=2, empty=(0, 7))
    cfg = config()
    path = tmp_path / "bundle"
    res = run(raw, cfg, out=path, project="proj", notes="test")
    with open_bundle(path) as b:
        assert b.has("spectra") and b.has("controls") and b.has("quality") and b.has("bootstrap")
        assert RunConfig.from_dict(b.manifest["configs"]["run"]) == cfg
        assert len(b.binned()) == len(raw) and len(b.fit()) == len(raw) - 1
        np.testing.assert_array_equal(b.quality().poor_fit, res.quality.poor_fit)
        assert b.controls().config == cfg.controls
        # where the peaks were found, and the m/z screen, go with the bundle
        np.testing.assert_array_equal(b.binned().peak_mz, res.binned.peak_mz)
        np.testing.assert_array_equal(b.binned().mz_offset_ppm, res.binned.mz_offset_ppm)
        assert np.isfinite(res.binned.peak_mz).any()
        np.testing.assert_array_equal(b.quality().mz_offset, res.quality.mz_offset)
        np.testing.assert_array_equal(b.quality().mz_shift, res.quality.mz_shift)
        assert np.isfinite(res.quality.mz_offset).all()
    plain = run(raw, config(controls=None, quality=None, bootstrap=None), out=tmp_path / "p",
                keep_spectra=False)
    with open_bundle(tmp_path / "p") as b:
        assert not (b.has("spectra") or b.has("controls") or b.has("quality") or b.has("bootstrap"))
    assert plain.controls is None and plain.quality is None


def test_a_picked_set_can_be_run_directly():
    raw = raw_spectra(n_series=1)
    res = run(pick_spectra(raw, PICK), config(bootstrap=None))
    assert len(res.fit) == len(raw)


def test_config_checks_and_round_trip():
    with pytest.raises(ValueError, match="k_min"):
        RunConfig(fit=FitConfig(k_min=2), controls=ControlConfig(undeut_time=0.0, fd_time=1e6))
    with pytest.raises(ValueError):
        RunConfig(max_chunk=0)
    with pytest.raises(ValueError, match="unknown"):
        RunConfig.from_dict({"bogus": 1})
    c = RunConfig(quality=QualityConfig(min_intensity=0.2), undeut_time=5.0)
    assert RunConfig.from_dict(c.to_dict()) == c and c.reference_time == 5.0
    assert config().reference_time == 0.0


def test_select_and_merge_are_checked(whole):
    raw, res = whole
    m = res.measurements
    sub = m.select([8, 9, 10, 11, 12, 13, 14, 15])  # series 1
    assert sub.references[0] == (0, 1, 2) and sub.spectrum_id[0] == m.spectrum_id[8]
    with pytest.raises(ValueError, match="reference"):
        m.select([3, 4])
    with pytest.raises(ValueError, match="distinct"):
        m.select([0, 0])
    n = len(res.fit)
    with pytest.raises(ValueError, match="one row position"):
        FitResult.merge([res.fit], [[0, 1]])
    with pytest.raises(ValueError, match="exactly once"):
        FitResult.merge([res.fit], [[0] * n])
    with pytest.raises(ValueError, match="exactly once"):
        BootstrapResult.merge([res.bootstrap], [[0] * n])
    assert isinstance(sub, Measurements)


# --- the bootstrap at fixed K ---------------------------------------------------------------

def boot_config(**kw):
    return BootstrapConfig(n_boot=3, n_restarts=1, adam_steps=30, polish_steps=0, **kw)


def test_ks_setting_is_checked_and_round_trips():
    for bad in ("some", (), (0,), (1, 1)):
        with pytest.raises(ValueError, match="ks must be"):
            BootstrapConfig(ks=bad)
    c = BootstrapConfig(ks=[1, 2])
    assert c.ks == (1, 2) and BootstrapConfig.from_dict(c.to_dict()) == c
    assert BootstrapConfig().ks == "selected" and BootstrapConfig(ks="all").ks == "all"
    with pytest.raises(ValueError, match="outside the fitted"):
        config(bootstrap=boot_config(ks=(1, 5)))


def test_fixed_k_bootstrap_agrees_with_the_selected_k_bootstrap(whole):
    from hxex.bootstrap import bootstrap, bootstrap_by_k, selected_from
    raw, res = whole
    cfg = boot_config()
    by_k = bootstrap_by_k(res.measurements, res.fit, (1, 2), cfg)
    assert set(by_k) == {1, 2} and all(len(b) == len(res.fit) for b in by_k.values())
    assert all((by_k[k].selected_K == k).all() for k in by_k)
    plain = bootstrap(res.measurements, res.fit, cfg)
    again = selected_from(res.fit, by_k)
    for name in ("mean_se", "population_se", "mean_ci", "n_success", "curve_hi", "curve_offsets"):
        np.testing.assert_allclose(getattr(again, name), getattr(plain, name), rtol=1e-6,
                                   atol=1e-8, equal_nan=True)
    with pytest.raises(ValueError, match="not fitted"):
        bootstrap_by_k(res.measurements, res.fit, (3,), cfg)
    with pytest.raises(ValueError, match="cover"):
        selected_from(res.fit, {k: b for k, b in by_k.items() if k != res.fit.selected_K[0]})


def test_run_with_ks_gives_every_k_and_chunking_does_not_matter():
    raw = raw_spectra()
    cfg = config(bootstrap=boot_config(ks="all", keep_draws=True))
    one = run(raw, cfg)
    assert set(one.bootstrap_by_k) == {1, 2} and one.bootstrap is not None
    for k, b in one.bootstrap_by_k.items():
        assert (b.selected_K == k).all() and b.draws is not None
    # the selected-K bootstrap is made of the rows of the fixed-K ones
    for i, k in enumerate(one.fit.selected_K):
        np.testing.assert_array_equal(one.bootstrap.mean_se[i], one.bootstrap_by_k[k].mean_se[i])
    small = run(raw, config(bootstrap=boot_config(ks="all", keep_draws=True), max_chunk=1))
    for k in (1, 2):
        np.testing.assert_allclose(small.bootstrap_by_k[k].mean_se, one.bootstrap_by_k[k].mean_se,
                                   rtol=1e-3, atol=1e-3, equal_nan=True)
    only = run(raw, config(bootstrap=boot_config(ks=(1,))))
    assert set(only.bootstrap_by_k) == {1} and len(only.bootstrap) == len(raw)


def test_the_bundle_keeps_the_fixed_k_bootstrap(tmp_path):
    raw = raw_spectra(n_series=2)
    res = run(raw, config(bootstrap=boot_config(ks="all", keep_draws=True)),
              out=tmp_path / "b")
    with open_bundle(tmp_path / "b") as b:
        assert b.bootstrap_ks() == [1, 2] and b.has("bootstrap_k_draws")
        for k in (1, 2):
            got, want = b.bootstrap(k), res.bootstrap_by_k[k]
            for name in ("selected_K", "n_success", "mean_se", "mean_ci", "population_se",
                         "curve_lo", "curve_offsets"):
                np.testing.assert_array_equal(getattr(got, name), getattr(want, name))
            np.testing.assert_array_equal(got.draws.n, want.draws.n)
        np.testing.assert_array_equal(b.bootstrap().mean_se, res.bootstrap.mean_se)
        assert b.bootstrap(3) is None
        comp = b.components_table()
        for k in (1, 2):  # a fixed-K result fills the rows of its K, selected or not
            rows = comp["fit_K"] == k
            assert np.isfinite(comp["mean_se"][rows]).all() and np.isfinite(
                comp["mean_corrected_se"][rows]).all()
    res_plain = run(raw, config(bootstrap=boot_config()), out=tmp_path / "p")
    with open_bundle(tmp_path / "p") as b:
        assert b.bootstrap_ks() == [] and b.bootstrap(1) is None
        comp = b.components_table()
        assert np.isnan(comp["mean_se"][~comp["is_selected"]]).all()
    assert res_plain.bootstrap_by_k is None


def test_write_bundle_refuses_a_wrong_fixed_k_result(tmp_path, whole):
    from hxex.io import BundleError, write_bundle
    raw, res = whole
    with pytest.raises(BundleError, match="fixed-K"):
        write_bundle(tmp_path / "x", measurements=res.measurements, fit=res.fit,
                     bootstrap_by_k={2: res.bootstrap})  # the selected-K result is not at K=2


def test_the_fits_are_written_before_the_bootstrap_and_resumed(tmp_path, whole, caplog):
    raw, res = whole
    out = tmp_path / "b"
    in_bootstrap = []

    class StopInBootstrap:
        def start_stage(self, name, start, end):
            in_bootstrap.append(name == "bootstrap")

        def __call__(self, done, total):
            pass

    hook = StopInBootstrap()
    with caplog.at_level(logging.WARNING, logger="hxex.run"), pytest.raises(Cancelled):
        run(raw, config(), out=out, progress=hook,
            cancel=lambda: bool(in_bootstrap) and in_bootstrap[-1])
    assert "resume_bootstrap" in caplog.text
    with open_bundle(out) as b:  # the fits, controls and flags, waiting for the bootstrap
        assert b.pending == ["bootstrap"] and b.bootstrap() is None
        assert b.has("spectra") and b.controls() is not None and b.quality() is not None
        np.testing.assert_array_equal(b.fit().selected_K, res.fit.selected_K)
    done = resume_bootstrap(out)
    with open_bundle(out) as b:
        assert b.pending == [] and b.has("spectra")
        boot = b.bootstrap()
    for got in (done.bootstrap, boot):  # as if the run had not been stopped
        np.testing.assert_allclose(got.mean_se, res.bootstrap.mean_se, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(got.curve_hi, res.bootstrap.curve_hi, rtol=1e-6, atol=1e-8)
    assert list(done.dropped_reason) == list(res.dropped_reason) and done.kept.all()
    with pytest.raises(ValueError, match="no bootstrap is pending"):
        resume_bootstrap(out)


def test_a_finished_run_and_a_run_without_bootstrap_have_nothing_pending(tmp_path):
    raw = raw_spectra(n_series=1)
    run(raw, config(bootstrap=None), out=tmp_path / "a")
    with open_bundle(tmp_path / "a") as b:
        assert b.pending == [] and b.bootstrap() is None
    with pytest.raises(ValueError, match="no bootstrap is pending"):
        resume_bootstrap(tmp_path / "a")
    stop = []
    with pytest.raises(Cancelled):  # stopped during the fit: nothing is written
        run(raw, config(), out=tmp_path / "c", cancel=lambda: len(stop) > 0 or stop.append(1))
    assert not (tmp_path / "c").exists()


def test_controls_can_fall_back_to_theoretical_back_exchange(tmp_path):
    from hxex.analysis.species import analysis_controls
    from hxex.io import open_bundle
    from hxex.report import Results

    quench = {"ph_read": 2.5, "temperature": 0.0, "d2o_fraction": 0.0}
    ctrl = ControlConfig(undeut_time=0.0, fd_time=1e6, fd_fallback="kint", quench=quench,
                         kint_min_series=3)
    res = run(raw_spectra(), config(bootstrap=None, controls=ctrl), out=tmp_path / "b")
    m = res.controls.fd_model
    assert m is not None and m["n_series"] == 3 and 0 < m["a"] <= 1.5
    with open_bundle(tmp_path / "b") as b:
        assert b.controls().fd_model == m and b.controls().config == ctrl
        results = Results.from_bundle(b)
    # recomputed at analysis time from the bundle's picked spectra and settings
    again = analysis_controls(results, {"kint_tolerance": 0.2}, quiet=True)
    assert again.fd_model["n_series"] == 3 and again.config.kint_tolerance == 0.2
