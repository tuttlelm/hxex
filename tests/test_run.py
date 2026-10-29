# SPDX-License-Identifier: GPL-3.0-or-later
"""The run stage: dropping flagged spectra, chunking (which must not change the result),
progress and cancellation, and the bundle it writes."""

import dataclasses
import logging

import numpy as np
import pytest

from hxex import chem
from hxex.bootstrap import BootstrapConfig, BootstrapResult
from hxex.controls import ControlConfig
from hxex.fit import FitConfig, FitResult, Measurements
from hxex.hooks import Cancelled
from hxex.io import open_bundle
from hxex.model import Backend
from hxex.noise import NoiseConfig
from hxex.picking import PickerConfig, pick_spectra, plan_bins
from hxex.picking.screen import SCREEN_SCORES
from hxex.quality import QualityConfig
from hxex.run import RunConfig, chunk_bytes, plan_chunks, resume_bootstrap, run
from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet
from test_bundle import envelope, natural
from test_picking import sticks

PEPTIDE = "RDKVQKEYAL"
PICK = PickerConfig(zero_filling=3, noise_threshold=100.0)
CPU = Backend("cpu", "float64")
# (time, rep, centre, width): time 0 is an undeuterated control and 1e6 a fully deuterated one
# (the role is set from them; a control's time is NaN, as the readers leave it)
LAYOUT = [(0.0, 1, 0.5, 0.8), (0.0, 2, 0.5, 0.85), (0.0, 3, 0.5, 0.8), (60.0, 1, 3.0, 1.2),
          (60.0, 2, 3.1, 1.2), (60.0, 3, 2.9, 1.1), (1e6, 1, 8.0, 1.3), (1e6, 2, 8.1, 1.3)]
ROLES = {0.0: "undeut", 1e6: "fulldeut"}


def control(t: float) -> tuple[float, str]:
    """``(time, role)`` of a layout time: the controls have a role and no time."""
    return (np.nan, ROLES[t]) if t in ROLES else (t, "")


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
            time, role = control(t)
            spectra.append(Spectrum(mz, y, entry, time, rep, data_id=s, label=f"s{s}c{k}",
                                    spectrum_id=f"{project}:f{s}.xlsx#{k + 1}", role=role))
    return SpectrumSet.from_spectra(spectra)


def config(**kw):
    fit = FitConfig(k_max=2, adam_steps=60, n_restarts=2, backend=CPU,
                    noise=NoiseConfig("gaussian", percent=3.0, estimate="replicates",
                                      min_replicated=2))
    boot = BootstrapConfig(n_boot=3, n_restarts=1, adam_steps=30, polish_steps=0, keep_draws=True)
    base = dict(picker=PICK, fit=fit, bootstrap=boot, controls=ControlConfig())
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
    # the amide count of the controls is the peptide's real site count (not n_max / factor)
    sites = chem.exchangeable_sites(PEPTIDE, nterm_subtract=PICK.nterm_subtract)
    assert sites == chem.count_amides(PEPTIDE, nterm_subtract=PICK.nterm_subtract) == 9
    assert res.controls.n_amides.tolist() == [sites] * 3


def test_flagged_spectra_are_dropped_and_recorded():
    raw = raw_spectra(empty=(1, 4))
    res = run(raw, config(bootstrap=None))
    assert res.kept.sum() == len(raw) - 1 and len(res.fit) == len(raw) - 1
    assert list(res.dropped_ids) == ["proj:f1.xlsx#5"]
    assert list(res.dropped_reason[~res.kept]) == ["empty"]
    assert len(res.binned) == len(raw)  # the picked table keeps every spectrum and its flags
    assert "proj:f1.xlsx#5" not in set(res.measurements.spectrum_id)
    # a spectrum without a bin is not "without signal"
    from types import SimpleNamespace

    from hxex.run import _drop_reasons

    binned = SimpleNamespace(empty=[True, False, False, False], offsets=np.array([0, 3, 3, 5, 8]))
    reasons = _drop_reasons(binned, np.array([False, False, False, True]))
    assert list(reasons) == ["empty", "no_bins", "no_signal", ""]


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


def test_a_series_without_its_undeuterated_control_is_warned_about(tmp_path, caplog):
    from hxex import load_results
    from hxex.noise import noise_reference_kinds
    from hxex.report import series_table

    raw = raw_spectra(n_series=2)
    keep = ~((raw.data_id == 1) & (raw.role == "undeut"))
    with caplog.at_level(logging.WARNING, logger="hxex.run"):
        run(raw.select(keep), config(bootstrap=None), out=tmp_path / "b")
    assert "1 of 2 series have no undeuterated control" in caplog.text
    assert "S 0001-0010 2+" in caplog.text
    table = series_table(load_results(tmp_path / "b"))
    assert list(table["noise_reference"]) == ["undeuterated control", "own spectrum"]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="hxex.run"):  # a fixed sd needs no reference
        run(raw.select(keep), config(bootstrap=None, fit=dataclasses.replace(
            config().fit, noise=NoiseConfig("gaussian", sd=100.0, percent=None))))
    assert "undeuterated control" not in caplog.text
    assert noise_reference_kinds([0, 0, 1, 2], [(0,), (0,), None, (3,)],
                                 ["undeut", "", "", ""]) == {
        0: "undeuterated control", 1: "own spectrum", 2: "run's scans"}


def test_random_numbers_follow_what_a_spectrum_is_not_the_project_name(tmp_path):
    raw = raw_spectra(n_series=2)
    other = dataclasses.replace(raw, spectrum_id=np.array(
        [str(i).replace("proj:", "other:") for i in raw.spectrum_id]))
    a = run(raw, config(), out=tmp_path / "a")
    b = run(other, config())
    assert not set(a.measurements.spectrum_id) & set(b.measurements.spectrum_id)
    np.testing.assert_array_equal(a.measurements.random_id, b.measurements.random_id)
    for k in a.fit.k_values:  # the same random starts, so the same fits
        np.testing.assert_array_equal(a.fit.fits[k].means, b.fit.fits[k].means)
    np.testing.assert_array_equal(a.bootstrap.mean_se, b.bootstrap.mean_se)  # and draws
    with open_bundle(tmp_path / "a") as bundle:  # read back as fitted (resume, population test)
        np.testing.assert_array_equal(bundle.measurements().random_id, a.measurements.random_id)


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
        for name in SCREEN_SCORES:  # the spectrum quality screens too
            np.testing.assert_array_equal(getattr(b.binned(), name), getattr(res.binned, name))
        assert np.isfinite(res.binned.screen_beyond_reach[~res.binned.empty]).all()
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
    # the controls are fitted at K = 1 whatever k_min is (fit_models adds it for them)
    assert RunConfig(fit=FitConfig(k_min=2), controls=ControlConfig()).fit.k_min == 2
    with pytest.raises(ValueError):
        RunConfig(max_chunk=0)
    with pytest.raises(ValueError, match="unknown"):
        RunConfig.from_dict({"bogus": 1})
    c = RunConfig(quality=QualityConfig(min_intensity=0.2), envelope_fallback_time=5.0)
    assert RunConfig.from_dict(c.to_dict()) == c
    with pytest.raises(ValueError, match="undeut_time no longer exists"):
        RunConfig.from_dict({**c.to_dict(), "undeut_time": 0.0})
    with pytest.raises(ValueError, match="no longer exist"):
        ControlConfig.from_dict({"undeut_time": 0.0, "fd_time": 1e6})


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
    done = resume_bootstrap(out, device="cpu")  # the run's device replaced (same here)
    assert done.config.fit.backend.device == "cpu" and done.fit.config.backend.device == "cpu"
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
    ctrl = ControlConfig(fd_fallback="kint", quench=quench,
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


def test_site_counts_add_hex_sites_and_give_zero_for_an_ion_without_any():
    from hxex.run import site_counts

    binned = pick_spectra(raw_spectra(n_series=1), PICK)
    cfg = config()
    n = len(binned)
    assert site_counts(binned, cfg).tolist() == [9] * n
    # owner: Hex sites stand for exchangeable amides on a modification and count
    hexed = dataclasses.replace(binned, modification=np.array(["Hex:2"] * n))
    assert site_counts(hexed, cfg).tolist() == [11] * n
    bare = dataclasses.replace(binned, peptide=np.array(["L"] * n),
                               modification=np.array(["ion_type:z"] * n))
    assert site_counts(bare, cfg).tolist() == [0] * n


def test_an_existing_output_is_refused_before_anything_is_fitted(tmp_path, monkeypatch):
    from hxex.io.bundle import BundleError

    raw = raw_spectra(n_series=1)
    path = tmp_path / "taken"
    path.write_text("not a bundle")
    monkeypatch.setattr("hxex.run.pick_spectra", lambda *a, **k: pytest.fail("picked anyway"))
    with pytest.raises(BundleError, match="exists"):
        run(raw, config(bootstrap=None), out=path)


def test_overwriting_a_reviewed_bundle_keeps_its_review(tmp_path, caplog):
    from hxex.io import ReviewTable
    from hxex.io.bundle import BundleError

    cfg = config(bootstrap=None)
    path = tmp_path / "b"
    res = run(raw_spectra(n_series=1), cfg, out=path)
    kept, dropped = str(res.fit.spectrum_id[3]), "proj:f0.xlsx#6"
    review = ReviewTable.empty().update(kept, status="rejected", note="noisy")
    review = review.update(dropped, status="accepted", note="fine")
    with open_bundle(path) as b:
        b.set_review(review)
        assert b.manifest["provenance"] is None
    with pytest.raises(BundleError, match="exists"):
        run(raw_spectra(n_series=1), cfg, out=path)
    # the new run drops spectrum 6 (empty): its entry is carried but stale, the other current
    with caplog.at_level(logging.INFO, logger="hxex.run"):
        again = run(raw_spectra(n_series=1, empty=(0, 5)), cfg, out=path, overwrite=True)
    assert dropped in again.dropped_ids
    assert "carrying 2 review entries over to the new bundle, 1 stale" in caplog.text
    with open_bundle(path) as b:
        r = b.review()
        assert sorted(r.spectrum_id.tolist()) == sorted([kept, dropped])
        assert r.get(kept) == {**r.get(kept), "note": "noisy", "stale": False}
        assert r.get(dropped) == {**r.get(dropped), "note": "fine", "stale": True}
        assert (r.status_of(b.fit()) == "rejected").sum() == 1
        prov = b.manifest["provenance"]
        assert prov["replaced"] == str(path) and prov["moved_to"] is None
        assert prov["review"] == {"carried": 2, "stale": 1}
        assert prov["settings_changed"] == [] and prov["replaced_created"]
    # the run's result carries it too, so tables and figures made from the result agree
    # with those made from the bundle
    from hxex.report import Results

    assert again.review is not None and len(again.review) == 2
    assert (Results.from_run(again).review_status == "rejected").sum() == 1
    assert (Results.from_run(res).review_status == "rejected").sum() == 0
    # a third run with other settings records what changed, and the review goes on
    cfg2 = dataclasses.replace(cfg, fit=dataclasses.replace(cfg.fit, k_max=1))
    third = run(raw_spectra(n_series=1), cfg2, out=path, overwrite=True)
    with open_bundle(path) as b:
        prov = b.manifest["provenance"]
        assert prov["settings_changed"] == ["fit.k_max"]
        assert prov["review"] == {"carried": 2, "stale": 0}  # spectrum 6 is back
    assert third.review.n_stale == 0


def test_keep_previous_moves_the_bundle_and_its_folders(tmp_path, caplog):
    from hxex.io import generations
    from hxex.io.bundle import BundleError
    from hxex.report import Results, write_tables
    from hxex.report.figures import PlotConfig

    cfg = config(bootstrap=None)
    path = tmp_path / "res"
    first = run(raw_spectra(n_series=1), cfg, out=path)
    write_tables(Results.from_run(first), tmp_path / "res_tables")
    first.write_figures(tmp_path / "res_figures", plots=PlotConfig(series="all",
                                                                   kinds=("uptake",)))
    (tmp_path / "res_tables" / "mine.txt").write_text("keep me")
    (tmp_path / "res.3_tables").mkdir()  # a number used before is never reused
    with pytest.raises(BundleError, match="exists"):
        run(raw_spectra(n_series=1), cfg, out=path)
    with caplog.at_level(logging.INFO, logger="hxex.io.bundle"):
        second = run(raw_spectra(n_series=1), cfg, out=path, keep_previous=True)
    assert "kept the previous results: moved res (" in caplog.text
    assert "to res.4" in caplog.text and "res_figures" in caplog.text
    for name in ("res.4", "res.4_tables", "res.4_figures", "res"):
        assert (tmp_path / name).is_dir(), name
    # moved with the bundle; a run writes neither the tables nor the figures itself
    assert not (tmp_path / "res_tables").exists() and not (tmp_path / "res_figures").exists()
    assert (tmp_path / "res.4_tables" / "mine.txt").read_text() == "keep me"
    with open_bundle(tmp_path / "res.4") as old, open_bundle(path) as new:
        assert old.manifest["provenance"] is None
        prov = new.manifest["provenance"]
        assert prov["moved_to"] == str(tmp_path / "res.4")
        assert prov["replaced_created"] == old.manifest["created"]
    assert generations(path) == [(4, tmp_path / "res.4")]
    assert second.review is None
    # nothing to keep: a fresh path is written as usual; something else there is refused
    run(raw_spectra(n_series=1), cfg, out=tmp_path / "fresh", keep_previous=True)
    (tmp_path / "plain").write_text("mine")
    with pytest.raises(BundleError, match="not a results bundle"):
        run(raw_spectra(n_series=1), cfg, out=tmp_path / "plain", keep_previous=True)
    assert (tmp_path / "plain").read_text() == "mine"


def test_a_screened_run_keeps_the_selected_k_and_its_bundle_reads_back(whole, tmp_path, caplog):
    raw, res = whole
    cfg = config(fit=dataclasses.replace(config().fit, screen=True),
                 bootstrap=dataclasses.replace(config().bootstrap, ks=(1, 2)))
    with caplog.at_level(logging.WARNING, logger="hxex.run"):
        screened = run(raw, cfg, out=tmp_path / "s")
    np.testing.assert_array_equal(screened.fit.selected_K, res.fit.selected_K)
    skipped = ~screened.fit.fitted(2)
    assert skipped.any() and res.fit.fitted(2).all()
    assert "fixed-K bootstrap has no values" in caplog.text
    assert np.all(screened.bootstrap_by_k[2].n_success[skipped] == 0)
    with open_bundle(tmp_path / "s") as b:
        np.testing.assert_array_equal(b.fit().fitted(2), screened.fit.fitted(2))
        np.testing.assert_array_equal(b.fit().selected_K, screened.fit.selected_K)
        assert RunConfig.from_dict(b.manifest["configs"]["run"]).fit.screen


def test_a_k_chosen_in_review_survives_the_screen(whole, tmp_path, caplog):
    """A K the screen skips: a review table's choice of it is fitted by the run anyway (the
    selected K unchanged) and bootstrapped; a bundle fitted without it gets it from resume,
    as the run would have fitted it; a carried-over bundle review choosing it is marked stale
    (no crash at write, with the bootstrap's checkpoint)."""
    from hxex.io import ReviewTable
    from hxex.run import resume_bootstrap

    raw, res = whole
    cfg = config(fit=dataclasses.replace(config().fit, screen=True))
    plain = run(raw, cfg, out=tmp_path / "s")
    skipped = np.flatnonzero(~plain.fit.fitted(2))
    assert len(skipped) >= 2
    a, b = (str(plain.fit.spectrum_id[i]) for i in skipped[:2])
    # the run with the table's choice
    chosen = run(raw, cfg, chosen_k={a: 2})
    i = skipped[0]
    assert chosen.fit.fitted(2)[i] and not chosen.fit.fitted(2)[skipped[1]]
    np.testing.assert_array_equal(chosen.fit.selected_K, plain.fit.selected_K)
    assert chosen.bootstrap_by_k[2].covers(i)
    # the bundle fitted without it: resume fits it, as the run did, and bootstraps it
    resumed = resume_bootstrap(tmp_path / "s", chosen_k={a: 2})
    assert resumed.fit.fitted(2)[i] and resumed.bootstrap_by_k[2].covers(i)
    assert resumed.fit.fits[2].nll[i] == pytest.approx(chosen.fit.fits[2].nll[i], rel=1e-4)
    others = np.ones(len(plain.fit), dtype=bool)
    others[i] = False
    np.testing.assert_array_equal(resumed.fit.fitted(2)[others], plain.fit.fitted(2)[others])
    np.testing.assert_array_equal(resumed.fit.selected_K, plain.fit.selected_K)
    with open_bundle(tmp_path / "s") as bundle:
        assert bundle.fit().fitted(2)[i]
    # a bundle review choosing K = 2 (fitted there) carried to a run whose screen skips it:
    # stale, and the run completes (it failed at the checkpoint write)
    run(raw, config(), out=tmp_path / "r")
    with open_bundle(tmp_path / "r") as bundle:
        bundle.set_review(ReviewTable.empty().update(b, k_override=2, note="two?"))
    with caplog.at_level(logging.INFO, logger="hxex.run"):
        again = run(raw, cfg, out=tmp_path / "r", overwrite=True)
    assert "1 stale" in caplog.text
    assert again.review.get(b)["stale"] and again.review.get(b)["note"] == "two?"
    with open_bundle(tmp_path / "r") as bundle:
        assert bundle.review().get(b)["stale"] and not bundle.pending


def test_a_late_time_point_serves_as_the_fully_deuterated_control(whole):
    """[run.controls] fulldeut_at: the spectra at that time are the control as well."""
    raw, _ = whole
    cfg = config(controls=ControlConfig(fulldeut_at=60.0), bootstrap=None)
    res = run(raw, cfg)
    assert set(res.fit.role[np.isclose(res.fit.time, 60.0)]) == {"fulldeut"}
    assert np.isfinite(res.fit.time[res.fit.role == "fulldeut"]).sum() == 3 * 3
    assert (res.controls.n_fd == 5).all()  # the two marked files plus the three at 60 s
    with pytest.raises(ValueError, match="fulldeut_at = 61 s: no spectrum"):
        run(raw, config(controls=ControlConfig(fulldeut_at=61.0), bootstrap=None))


def test_stored_spectra_keep_the_readers_roles(whole, tmp_path):
    """Audit 2026-10-28 High 2: the bundle stores the spectra with the roles the readers gave
    them; fulldeut_at is applied again from the settings of every run, so a rerun from the
    stored spectra without it has the time point back in its time course."""
    from hxex.config import load_stored_spectra

    raw, _ = whole
    run(raw, config(controls=ControlConfig(fulldeut_at=60.0), bootstrap=None),
        out=tmp_path / "b")
    stored = load_stored_spectra(tmp_path / "b")
    assert np.asarray(stored.role).tolist() == np.asarray(raw.role).tolist()
    again = run(stored, config(controls=ControlConfig(), bootstrap=None))
    assert (again.controls.n_fd == 2).all()
    assert ((again.fit.role == "") & np.isclose(again.fit.time, 60.0)).any()


def test_a_rerun_from_stored_spectra_reuses_their_picking(whole, tmp_path, caplog):
    """A rerun from a bundle's stored spectra takes the bundle's picked table when picking again
    would give it (the same spectra, roles, picking settings and hxex version): the results
    equal a run that picks; a change of any of them picks again, saying what differed."""
    from hxex.config import load_stored_spectra

    raw, _ = whole
    cfg = config(bootstrap=None)
    first = run(raw, cfg, out=tmp_path / "b")
    stored = load_stored_spectra(tmp_path / "b")
    with caplog.at_level(logging.INFO, logger="hxex.run"):
        again = run(stored, cfg)
    assert "the picking stored in" in caplog.text
    for a, b in zip(again.binned.intensity, first.binned.intensity, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(again.fit.selected_K, first.fit.selected_K)
    np.testing.assert_allclose(again.fit.fits[1].nll, first.fit.fits[1].nll)
    caplog.clear()
    wider = dataclasses.replace(cfg, picker=dataclasses.replace(cfg.picker, resolution_ppm=40.0))
    with caplog.at_level(logging.INFO, logger="hxex.run"):
        run(stored, wider)
    assert "picking the stored spectra again: [run.picker] differ(s)" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="hxex.run"):
        run(stored, config(controls=ControlConfig(fulldeut_at=60.0), bootstrap=None))
    assert "the control roles" in caplog.text


def test_the_controls_are_fitted_at_k1_whatever_k_min(tmp_path):
    """With k_min above 1 the controls still get the K = 1 fit the back-exchange correction
    uses (and are fitted from k_min to k_max like the rest); the other spectra are not fitted
    at K = 1, and no spectrum is chosen at it. The controls equal those of a k_min = 1 run."""
    raw = raw_spectra()
    one = run(raw, config(controls=ControlConfig()))
    two = run(raw, config(controls=ControlConfig(),
                          fit=dataclasses.replace(config().fit, k_min=2, k_max=2)))
    fit = two.fit
    role = np.asarray(fit.role, dtype=object)
    is_control = (role != "") & (role != None)  # noqa: E711
    assert is_control.any() and (~is_control).any()
    assert 1 in fit.k_values and fit.config.k_values == (2,)
    assert (fit.fitted(1) == is_control).all() and fit.fitted(2).all()
    assert (fit.selected_K == 2).all()
    np.testing.assert_allclose(two.controls.m0, one.controls.m0, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(two.controls.mfd, one.controls.mfd, rtol=1e-6, equal_nan=True)


def test_a_control_promoted_after_a_run_without_its_k1_fit_is_refused(tmp_path):
    """A time point promoted to a control only in the analysis ([analysis.controls]
    fulldeut_at) after a run with k_min above 1 has no K = 1 fit: refused with the reason
    (its missing mean made the series' means NaN), not used."""
    from hxex.analysis.species import analysis_controls
    from hxex.report import Results

    raw = raw_spectra()
    two = run(raw, config(fit=dataclasses.replace(config().fit, k_min=2, k_max=2)))
    results = Results.from_run(two)
    t = float(np.nanmax(np.asarray(results.fit.time, dtype=float)))
    with pytest.raises(ValueError, match="no K=1 fit.*k_min = 1"):
        analysis_controls(results, {"fulldeut_at": t, "fulldeut_marked": False})
    assert analysis_controls(results, {"d2o_fraction": 0.85}) is not None  # the marked ones
