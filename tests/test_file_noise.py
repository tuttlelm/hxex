# SPDX-License-Identifier: GPL-3.0-or-later
"""The noise of a run from its own scans (PLAN Next steps 43, HISTORY 449): the halves of each
spectrum's scans kept at read time, their bin values at the pick's positions, one noise curve
per scan file (the halves, or a kinetics run's neighbouring scans), its use in the fit and its
record in the bundle."""

import dataclasses
import json

import numpy as np
import pytest

from hxex.cli import _noise_curve_text
from hxex.fit import Measurements
from hxex.io import open_bundle
from hxex.io.mzml import (
    MzmlConfig,
    MzmlRun,
    MzmlTarget,
    read_mzml,
    scan_file_names,
    scan_spectrum_keys,
)
from hxex.noise import (
    ControlsNoise,
    FileNoise,
    NoiseConfig,
    bin_noise,
    bin_variances,
    file_noise,
    median_b1,
)
from hxex.picking import PickerConfig, pick_spectra, plan_bins
from hxex.picking.measure import PeakMeasure
from hxex.picking.profile import PeakShape, fit_spectrum, ion_moments
from hxex.pipeline import run
from hxex.reuse import kept_curve
from hxex.spectra import PeptideEntry
from test_mzml import ENTRY, PEPTIDE, write_mzml
from test_run import config

PICK = PickerConfig(zero_filling=3, noise_threshold=100.0)
SHAPE = np.array([0.1, 0.3, 0.6, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3, 0.15, 0.08, 0.04])


def halves_with_noise(a2, b1, c2, n=2000, lo=3.5, hi=6.5, seed=1, file="f", charges=(1, 2, 3)):
    """Spectra whose two halves carry the variance ``a2/2 + z b1 I + 2 c2 I^2`` each, so the
    full sum's is ``a2 + z b1 I + c2 I^2``."""
    rng = np.random.default_rng(seed)
    obs, odd, even, ch, files = [], [], [], [], []
    for s in range(n):
        z = charges[s % len(charges)]
        half = 10 ** rng.uniform(lo, hi) * SHAPE / 2
        var = a2 / 2 + z * b1 * half + 2 * c2 * half ** 2
        o = half + rng.standard_normal(len(SHAPE)) * np.sqrt(var)
        e = half + rng.standard_normal(len(SHAPE)) * np.sqrt(var)
        obs.append(o + e)
        odd.append(o)
        even.append(e)
        ch.append(z)
        files.append(file)
    return obs, odd, even, ch, files


def test_the_split_route_recovers_the_curve_of_each_file():
    a, b, c = halves_with_noise(0.0, 8.0, 0.0, file="x"), \
        halves_with_noise(1e4, 8.0, 0.02 ** 2, file="y", seed=2), \
        halves_with_noise(0.0, 800.0, 0.0, lo=5, hi=7.5, file="z", seed=3)
    args = [sum((part[k] for part in (a, b, c)), []) for k in range(5)]
    curves, why = file_noise(*args, n_resample=4)
    assert why == {} and [f.file for f in curves] == ["x", "y", "z"]
    by = {f.file: f for f in curves}
    assert all(f.route == "split" and f.n_spectra == 2000 for f in curves)
    assert abs(by["x"].b1 - 8.0) < 0.6 and by["x"].c2 < 1e-6
    assert abs(by["y"].b1 - 8.0) < 0.8 and abs(np.sqrt(by["y"].c2) - 0.02) < 0.004
    assert abs(np.sqrt(by["y"].a2) - 100) < 40
    assert abs(by["z"].b1 - 800) / 800 < 0.08
    assert np.isfinite(by["x"].b1_se) and by["x"].b1_se < 1.0
    # the bundle's b1: the median over the files, with an error from their spread
    b1, se = median_b1(curves)
    assert b1 == pytest.approx(sorted(f.b1 for f in curves)[1]) and np.isfinite(se)


def test_the_neighbouring_scans_route_for_kinetics_runs():
    rng = np.random.default_rng(2)
    obs, files, target, time, kin, ch = [], [], [], [], [], []
    for t in range(1500):
        mean = 10 ** (4 + 2.5 * (t % 50) / 50) * SHAPE * (1 + 0.002 * t)
        obs.append(mean + rng.standard_normal(len(SHAPE)) * np.sqrt(2 * 5.0 * mean))
        files.append("k")
        target.append(f"T{t % 50}")
        time.append(float(t // 50))
        kin.append(True)
        ch.append(2)
    none = [None] * len(obs)
    # scans picked to different lengths are aligned by bin number
    bins = [np.arange(len(SHAPE) - (t % 3), dtype=float) for t in range(len(obs))]
    obs = [o[:len(b)] for o, b in zip(obs, bins, strict=True)]
    curves, why = file_noise(obs, none, none, ch, files, bins=bins, kinetics=kin,
                             target=target, time=time, n_resample=4)
    assert why == {} and len(curves) == 1 and curves[0].route == "scans"
    assert abs(curves[0].b1 - 5.0) < 0.5 and curves[0].n_spectra == 1500
    with pytest.raises(ValueError, match="bins, target and time"):
        file_noise(obs, none, none, ch, files, kinetics=kin)


def test_files_without_enough_bins_or_halves_are_listed_and_copies_count_once():
    obs, odd, even, ch, files = halves_with_noise(0.0, 8.0, 0.0, n=40)
    curves, why = file_noise(obs[:3], odd[:3], even[:3], ch[:3], ["a"] * 3, n_resample=2)
    assert curves == [] and why == {"a": "too few bins with signal in both halves"}
    none = [None] * 3
    curves, why = file_noise(obs[:3], none, none, ch[:3], ["a"] * 3)
    assert why == {"a": "no spectrum summed from two or more scans"}
    # not from a scan file: left out, nothing said
    curves, why = file_noise(obs, odd, even, ch, [""] * len(obs))
    assert curves == [] and why == {}
    # a copy (the same measurement read twice) counts once
    twice = file_noise(obs + obs, odd + odd, even + even, ch + ch, files + files,
                       measurement=list(range(40)) * 2, n_resample=2)[0]
    once = file_noise(obs, odd, even, ch, files, n_resample=2)[0]
    assert twice[0].n_spectra == once[0].n_spectra == 40 and twice[0].b1 == once[0].b1


def test_a_reduced_file_takes_the_recorded_noise_as_its_floor():
    obs, odd, even, ch, files = halves_with_noise(0.0, 8.0, 0.0, n=600)
    recorded = np.full(len(obs), 300.0)
    plain = file_noise(obs, odd, even, ch, files, recorded_noise=recorded, n_resample=2)[0][0]
    cut = file_noise(obs, odd, even, ch, files, recorded_noise=recorded, reduced={"f"},
                     n_resample=2)[0][0]
    assert not plain.floor_recorded and plain.a2 < 300.0 ** 2
    assert cut.floor_recorded and cut.a2 == 300.0 ** 2 and np.isnan(cut.a2_se)
    assert cut.b1 == plain.b1


def test_bin_noise_takes_each_measurements_file_curve():
    curve = ControlsNoise(b1=10.0, model_error=0.09, onset=float("inf"), n_controls=0,
                          n_series=0, by_file=(FileNoise("a", "split", 100.0, 5.0, 1e-4, 3, 30),))
    y = np.array([1e3, 1e4, 1e3])
    noise = bin_noise([y, y, y], [2.0, 2.0, 2.0], curve, files=["a", "b", ""])
    smooth = np.convolve(y, np.ones(3), "same") / np.convolve(np.ones(3), np.ones(3), "same")
    np.testing.assert_allclose(noise.base[0], 100.0 + 2 * 5.0 * smooth + 1e-4 * smooth ** 2)
    np.testing.assert_allclose(noise.base[1], 2 * 10.0 * smooth)
    np.testing.assert_allclose(noise.base[2], 2 * 10.0 * smooth)
    assert noise.model_error == 0.09
    var = bin_variances([y], [2.0], curve, files=["a"])[0]
    np.testing.assert_allclose(var, np.maximum(100.0 + 10.0 * smooth + 1e-4 * smooth ** 2
                                               + (0.09 * smooth) ** 2, (0.005 * 1e4) ** 2))
    with pytest.raises(ValueError, match="one entry per measurement"):
        bin_noise([y], [2.0], curve, files=["a", "b"])
    # the record round-trips, and routes say what measured it
    back = ControlsNoise.from_dict(json.loads(json.dumps(curve.to_dict())))
    assert back == curve and back.routes == "files" and back.curve_of("a").b1 == 5.0
    # a file is named when its b1 is more than three errors and 10 % from the bundle's
    files = (FileNoise("a", "split", 0.0, 10.5, 0.0, 9, 90, b1_se=0.1),
             FileNoise("b", "split", 0.0, 12.0, 0.0, 9, 90, b1_se=0.1),
             FileNoise("c", "split", 0.0, 14.0, 0.0, 9, 90, b1_se=3.0))
    named = dataclasses.replace(curve, b1_se=0.1, by_file=files).disagreeing_files()
    assert [f.file for f in named] == ["b"]
    assert dataclasses.replace(curve, n_controls=3).routes == "mixed"
    assert ControlsNoise.from_dict({"b1": 1.0, "model_error": 0.09, "onset": None,
                                    "n_controls": 5, "n_series": 5}).routes == "controls"


def test_same_curve_compares_per_file_and_the_kept_curve_rule_is_per_file(tmp_path):
    a = FileNoise("a", "split", 100.0, 5.0, 1e-4, 3, 30)
    b = FileNoise("b", "split", 0.0, 7.0, 0.0, 3, 30)
    old = ControlsNoise(b1=6.0, model_error=0.09, onset=float("inf"), n_controls=0, n_series=0,
                        by_file=(a, b))
    moved_b = dataclasses.replace(b, b1=7.5)
    new = dataclasses.replace(old, b1=6.25, by_file=(dataclasses.replace(a, b1=5.001), moved_b))
    same = new.same_curve(old, ["a", "b", "", "c"])
    assert list(same) == [False, False, False, False]  # curves differ as measured
    # within the share of HISTORY 433 the earlier curve is kept, per file
    bundle = tmp_path / "results"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"kind": "hxex-results-bundle",
                                                      "noise_curve": old.to_dict()}))
    kept = kept_curve(bundle, new)
    assert kept.curve_of("a") == a and kept.curve_of("b") == moved_b and kept.b1 == 6.25
    assert list(kept.same_curve(old, ["a", "b", "", "c"])) == [True, False, False, False]
    # another model error: nothing kept
    other = dataclasses.replace(new, model_error=0.1)
    assert kept_curve(bundle, other) == other
    assert not new.same_curve(other, ["a"]).any()


def _noisy_scans(rng, ions, level, width, n_scans=8, top=3e4):
    """``n_scans`` profile scans of ``ions`` (peptide, charge) at a deuteration ``level``, with
    counting noise (b1 = 20 per unit charge) in every point."""
    plans = {ion: plan_bins(ion[0], ion[1], config=PICK) for ion in ions}
    scans = []
    heights = {ion: top * 10 ** rng.uniform(0, 1) for ion in ions}
    for k in range(n_scans):
        mz, y = [], []
        for (pep, z), plan in plans.items():
            for b, pmz in enumerate(plan.pred_mz):
                h = heights[(pep, z)] * np.exp(-0.5 * ((b - level) / width) ** 2)
                if h < 1:
                    continue
                x = pmz + np.arange(-0.012, 0.0121, 0.002)
                v = h * np.exp(-0.5 * ((x - pmz) / 0.004) ** 2)
                v = v + rng.standard_normal(len(x)) * np.sqrt(z * 20.0 * v)
                mz.append(x)
                y.append(np.clip(v, 0, None))
        mz, y = np.concatenate(mz), np.concatenate(y)
        order = np.argsort(mz)
        scans.append({"mz": mz[order], "intensity": y[order], "time": 5.0 + 0.01 * k,
                      "filter": "FTMS + p ESI Full ms"})
    return scans


IONS = [(p, z) for p in ("RDKVQKEYAL", "PEPTIDEKKW", "ALKSDFGHQR", "MNVTWREELK", "GGHHKKLLMF")
        for z in (2, 3)]


@pytest.fixture(scope="module")
def scan_project(tmp_path_factory):
    """Six scan files (undeuterated, 60 s and fully deuterated, two replicates each) of ten
    ions, read into raw spectra with their halves."""
    d = tmp_path_factory.mktemp("scans")
    rng = np.random.default_rng(0)
    runs = []
    for name, (level, width, t, role) in {"un": (0.3, 0.8, None, "undeut"),
                                          "t60": (3.0, 1.2, 60.0, ""),
                                          "fd": (8.0, 1.3, None, "fulldeut")}.items():
        for rep in (1, 2):
            path = write_mzml(d / f"{name}_{rep}.mzML", _noisy_scans(rng, IONS, level, width))
            runs.append(MzmlRun(path, "S", rep=rep, time=t, role=role))
    targets = [MzmlTarget(PeptideEntry("", "S", pep, f"{10 * i + 1:04d}-{10 * i + 10:04d}",
                                       10 * i + 1, 10 * i + 10, z,
                                       extra={"rt": "5.04", "rt_start": "5.0",
                                              "rt_end": "5.08"}))
               for i, (pep, z) in enumerate(IONS)]
    raw = read_mzml(runs, targets, project="p", config=MzmlConfig(), picker=PICK)
    return d, raw


def test_the_reader_keeps_the_halves_of_every_summed_spectrum(scan_project, tmp_path):
    d, raw = scan_project
    assert len(raw) == 60 and (raw.n_scans == 8).all() and np.isnan(raw.recorded_noise).all()
    for i in range(len(raw)):
        odd, even = raw.halves(i)
        np.testing.assert_allclose(odd + even, raw.spectrum(i)[1], rtol=1e-9)
        assert (odd >= 0).all() and (even >= 0).all() and odd.sum() > 0 and even.sum() > 0
    files, targets = scan_spectrum_keys(raw.source, raw.spectrum_id)
    assert sorted(set(files)) == sorted(f"{n}_{r}.mzML" for n in ("un", "t60", "fd")
                                        for r in (1, 2))
    assert set(targets) == {f"{10 * i + 1:04d}-{10 * i + 10:04d}-{pep}-z{z}"
                            for i, (pep, z) in enumerate(IONS)}
    assert list(scan_file_names(None, raw.spectrum_id)) == [""] * 60
    # a kinetics run (each scan its own spectrum) has no halves
    one = read_mzml([MzmlRun(d / "un_1.mzML", "S")], [MzmlTarget(ENTRY)], project="p",
                    picker=PICK)
    assert (one.n_scans == 1).all() and one.halves(0) is None
    # the bins of the halves are read where the pick found the full spectrum's peaks
    binned = pick_spectra(raw, PICK)
    assert (binned.n_scans == 8).all()
    np.testing.assert_allclose(binned.intensity_odd + binned.intensity_even, binned.intensity,
                               rtol=1e-9)
    empty = binned.intensity == 0
    assert (binned.intensity_odd[empty] == 0).all()
    # with areas the halves are areas too (their sum is the full area)
    area = pick_spectra(raw, PICK, measure=PeakMeasure("area", "measured", 15.0, 0.0))
    np.testing.assert_allclose(area.intensity_odd + area.intensity_even, area.intensity,
                               rtol=1e-6)
    assert not (len(area.intensity) == len(binned.intensity)
                and np.allclose(area.intensity, binned.intensity))


def test_the_profile_fit_gives_the_halves_their_values_with_the_spectrums_design(scan_project):
    _, raw = scan_project
    plan = plan_bins(PEPTIDE, 2, config=PICK)
    moments = ion_moments(PEPTIDE)
    law = PeakMeasure("area", "auto", 15.0, 0.0)
    i = int(np.flatnonzero((raw.peptide == PEPTIDE) & (raw.charge == 2))[0])
    mz, y = raw.spectrum(i)
    sticks = pick_spectra(raw.select(np.arange(len(raw)) == i), PICK).intensity
    fit = fit_spectrum(mz, y, sticks, plan, moments, law.fwhm_mz, PeakShape.gaussian(),
                       centre=0.0, limit=5.0, step=1.0, coarse=False, halves=raw.halves(i))
    assert fit is not None and fit.halves is not None
    odd, even = fit.halves
    np.testing.assert_allclose(odd + even, fit.values, rtol=0.05, atol=1e-6 * fit.values.max())


def test_a_run_measures_one_curve_per_scan_file_and_keeps_it(scan_project, tmp_path, caplog):
    _, raw = scan_project
    cfg = config(picker=PICK, fit=dataclasses.replace(
        config().fit, noise=NoiseConfig("gaussian", percent=3.0, estimate="controls")))
    out = tmp_path / "results"
    res = run(raw, cfg, out=out)
    curve = res.fit.noise_curve
    assert curve is not None and curve.routes == "files" and curve.n_controls == 0
    measured = {f.file for f in curve.by_file}
    assert measured and all(f.route == "split" and f.n_spectra == 10 for f in curve.by_file)
    assert measured | {name for name, _ in curve.unmeasured} == set(
        f"{n}_{r}.mzML" for n in ("un", "t60", "fd") for r in (1, 2))
    assert curve.b1 == pytest.approx(median_b1(curve.by_file)[0])
    text = _noise_curve_text(res.fit)
    assert text.startswith("per bin, a2 + z x b1 x I + c2 x I^2 + (9 % of the fit)^2, one curve "
                           "per scan file") and "b1 median" in text
    with open_bundle(out) as b:
        assert b.fit().noise_curve == curve
        assert b.manifest["noise_curve"]["files"][0]["route"] == "split"
        m = b.measurements()
        assert set(m.files()) == {f"{n}_{r}.mzML" for n in ("un", "t60", "fd") for r in (1, 2)}
        stored = b.spectra()
        assert stored.halves(0) is not None and (stored.n_scans == 8).all()
        binned = b.binned()
        np.testing.assert_allclose(binned.intensity_odd + binned.intensity_even,
                                   binned.intensity, rtol=1e-6)
    # the same data again: the curves are the same, so every fit is kept
    caplog.set_level("INFO")
    again = run(raw, cfg, out=out, overwrite=True)
    assert again.fit.noise_curve == curve
    assert any("keep their fit" in r.message and "60 of 60" in r.message
               for r in caplog.records)
    np.testing.assert_array_equal(again.fit.selected_K, res.fit.selected_K)
    # the stored spectra (a rerun without the files) carry the halves: the same curves
    from_stored = Measurements.from_binned(binned, nterm_subtract=PICK.nterm_subtract,
                                           file=scan_file_names(stored.source,
                                                                binned.spectrum_id))
    assert set(from_stored.files()) == set(m.files())
