# SPDX-License-Identifier: GPL-3.0-or-later
"""A run that replaces a bundle keeps the fits, population test and bootstrap of the series
whose input is unchanged (hxex.reuse; docs/PLAN.md Next steps 26, "Applying"): the result is
the one a full run gives, and only the changed series are fitted again."""

import dataclasses
import logging

import numpy as np
import pytest

from hxex.pipeline import run
from hxex.population_test import PopulationTestConfig
from hxex.spectra import SpectrumSet
from test_run import config, raw_spectra

TEST = PopulationTestConfig(simulations=4)


def _config():
    base = config(population_test=TEST)
    return dataclasses.replace(base, bootstrap=dataclasses.replace(base.bootstrap, ks=(1, 2)))


def _changed(raw: SpectrumSet, index: int, factor: float = 1.3,
             whole: bool = False) -> SpectrumSet:
    """``raw`` with the tallest point of spectrum ``index`` (``whole``: every point) scaled by
    ``factor``."""
    a, b = raw.offsets[index], raw.offsets[index + 1]
    y = np.asarray(raw.intensity, dtype=float).copy()
    if whole:
        y[a:b] *= factor
    else:
        y[a + int(np.argmax(y[a:b]))] *= factor
    return dataclasses.replace(raw, intensity=y)


def _same(a, b, rows=None, close=False, m=None, atol=1e-8):
    """The per-measurement arrays (``m`` rows; default ``len(a)``) of two results (fit, test,
    bootstrap, controls or flags) equal, at ``rows`` (all: None); with ``close``, to 1e-4
    relative or ``atol``."""
    m = len(a) if m is None else m
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if not isinstance(x, np.ndarray) or x.ndim == 0 or len(x) != m:
            continue
        if rows is not None:
            x, y = x[rows], y[rows]
        if x.dtype.kind in "fc" and close:
            # a series fitted in another batch rounds differently (HISTORY 46, 219): on
            # macOS a population test's null mean moved by 4e-6 relative (HISTORY 330)
            np.testing.assert_allclose(x, y, rtol=1e-4, atol=atol, err_msg=f.name)
        else:
            np.testing.assert_array_equal(x, y, err_msg=f.name)


def _same_fit(a, b, rows=None, close=False):
    np.testing.assert_array_equal(a.selected_K if rows is None else a.selected_K[rows],
                                  b.selected_K if rows is None else b.selected_K[rows])
    for k in a.k_values:
        _same(a.fits[k], b.fits[k], rows, close, m=len(a))


@pytest.fixture(scope="module")
def first(tmp_path_factory):
    out = tmp_path_factory.mktemp("reuse") / "results"
    raw = raw_spectra(n_series=3)
    return raw, out, run(raw, _config(), out=out)


def test_a_second_run_keeps_every_fit(first, caplog):
    raw, out, before = first
    with caplog.at_level(logging.INFO, logger="hxex.pipeline"):
        again = run(raw, _config(), out=out, overwrite=True)
    n = len(before.fit)
    assert (f"{n} of {n} spectra keep their fit, population test and bootstrap from the "
            "previous results (the same input); 0 fitted") in caplog.text
    _same_fit(again.fit, before.fit)
    _same(again.population_test, before.population_test)
    _same(again.bootstrap, before.bootstrap)
    for k in (1, 2):
        _same(again.bootstrap_by_k[k], before.bootstrap_by_k[k])
    _same(again.controls, before.controls)
    _same(again.quality, before.quality)


def test_a_changed_spectrum_refits_its_series_only(first, tmp_path, caplog):
    raw = first[0]
    changed = _changed(raw, index=3)  # series 0, a labelled time point
    sid = str(raw.spectrum_id[3])
    target = tmp_path / "results"
    before = run(raw, _config(), out=target)
    with caplog.at_level(logging.INFO, logger="hxex.pipeline"):
        partial = run(changed, _config(), out=target, overwrite=True)
    fresh = run(changed, _config())  # a full run, nothing reused
    n = len(raw)
    series = np.asarray(partial.measurements.data_id)
    own = series == series[list(map(str, partial.fit.spectrum_id)).index(sid)]
    assert (f"{n - own.sum()} of {n} spectra keep their fit") in caplog.text
    # the unchanged series are the previous results' (exactly: two full runs differ in the last
    # bits), and the whole a full run's
    kept = np.flatnonzero(~own)
    _same_fit(partial.fit, before.fit, kept)
    _same_fit(partial.fit, fresh.fit, close=True)
    # the test's statistics are differences of two fitted losses in the hundreds: their
    # rounding is absolute, 1e-5 for a statistic of 0.08 on macOS (HISTORY 332)
    _same(partial.population_test, fresh.population_test, close=True, atol=1e-3)
    _same(partial.bootstrap, fresh.bootstrap, close=True)
    _same(partial.controls, fresh.controls, close=True)
    _same(partial.quality, fresh.quality, close=True)


def test_other_settings_refit_everything(first, tmp_path, caplog):
    raw, _, _ = first
    target = tmp_path / "results"
    run(raw, _config(), out=target)
    other = _config()
    other = dataclasses.replace(other, bootstrap=dataclasses.replace(other.bootstrap, n_boot=4))
    with caplog.at_level(logging.INFO, logger="hxex.reuse"):
        run(raw, other, out=target, overwrite=True)
    assert "fitting every spectrum" in caplog.text and "[run.bootstrap] changed" in caplog.text


def test_reuse_can_be_turned_off(first, tmp_path, caplog):
    raw, _, _ = first
    target = tmp_path / "results"
    run(raw, _config(), out=target)
    with caplog.at_level(logging.INFO):
        run(raw, _config(), out=target, overwrite=True, reuse=False)
    assert "keep their fit" not in caplog.text


def test_a_changed_undeuterated_control_refits_its_series(first, tmp_path, caplog):
    raw, _, _ = first
    target = tmp_path / "results"
    run(raw, _config(), out=target)
    index = int(np.flatnonzero((raw.role == "undeut") & (raw.data_id == 1))[0])
    with caplog.at_level(logging.INFO, logger="hxex.pipeline"):
        run(_changed(raw, index), _config(), out=target, overwrite=True)
    per_series = len(raw) // 3
    assert f"{len(raw) - per_series} of {len(raw)} spectra keep their fit" in caplog.text


def test_the_command_line_reuses_and_refit_does_not(tmp_path, capsys):
    from hxex.cli import main
    from test_cli import mzml_project

    path = mzml_project(tmp_path)
    assert main(["run", str(path)]) == 0
    capsys.readouterr()
    assert main(["run", str(path), "--overwrite"]) == 0
    assert "6 of 6 spectra keep their fit" in capsys.readouterr().err
    assert main(["info", str(path)]) == 0
    assert "6 of 6 spectra kept the replaced bundle's fits" in capsys.readouterr().out
    assert main(["run", str(path), "--overwrite", "--refit"]) == 0
    assert "keep their fit" not in capsys.readouterr().err
    assert main(["info", str(path)]) == 0
    assert "0 of 6 spectra kept" in capsys.readouterr().out


def test_the_software_that_made_the_fits_is_kept_by_a_resume(first, tmp_path, caplog):
    import json

    from hxex.io.bundle import _versions
    from hxex.pipeline import resume_bootstrap

    raw = first[0]
    target = tmp_path / "results"
    run(raw, _config(), out=target)
    resume_bootstrap(target, add=True, bootstrap=_config().bootstrap)
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["fitted_with"] == json.loads(json.dumps(_versions()))
    with caplog.at_level(logging.INFO, logger="hxex.pipeline"):
        run(raw, _config(), out=target, overwrite=True)
    assert f"{len(raw)} of {len(raw)} spectra keep their fit" in caplog.text
    # a bundle resumed before the record: not known, so nothing is reused
    manifest = json.loads((target / "manifest.json").read_text())
    del manifest["fitted_with"]
    manifest["provenance"] = {"resumed": [{"added": "bootstrap"}]}
    (target / "manifest.json").write_text(json.dumps(manifest))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="hxex.reuse"):
        run(raw, _config(), out=target, overwrite=True)
    assert "it is not known which version of hxex made them" in caplog.text


def test_a_noise_curve_that_hardly_moved_is_kept(tmp_path, caplog):
    import json

    from hxex.noise import NoiseConfig

    raw = raw_spectra(n_series=6)
    base = _config()
    cfg = dataclasses.replace(base, fit=dataclasses.replace(
        base.fit, noise=NoiseConfig("gaussian", percent=3.0, estimate="controls")))
    target = tmp_path / "results"
    run(raw, cfg, out=target)
    before = json.loads((target / "manifest.json").read_text())["noise_curve"]
    index = int(np.flatnonzero((raw.role == "undeut") & (raw.data_id == 2))[0])
    with caplog.at_level(logging.INFO):
        again = run(_changed(raw, index, factor=1.000001), cfg, out=target, overwrite=True)
    assert "the earlier curve is kept" in caplog.text
    per_series = len(raw) // 6
    assert f"{len(raw) - per_series} of {len(raw)} spectra keep their fit" in caplog.text
    assert again.fit.noise_curve.b1 == before["b1"]
    # a control gone (no signal left): the curve rests on other controls, nothing is kept
    caplog.clear()
    with caplog.at_level(logging.INFO):
        run(_changed(raw, index, factor=0.0, whole=True), cfg, out=target, overwrite=True)
    assert "the earlier curve is kept" not in caplog.text
    assert "per-bin noise measured from the undeuterated controls changed" in caplog.text


def test_a_k_chosen_in_the_review_keeps_the_fits_and_is_bootstrapped(tmp_path, caplog):
    """A K the review chooses that the run fitted: every fit is kept, and the bootstrap at the
    chosen K is made for that spectrum (``[run.bootstrap] ks`` does not hold it), as a full run
    with the same review makes it."""
    raw = raw_spectra(n_series=3)
    target = tmp_path / "results"
    first = run(raw, config(), out=target)
    sid = next(str(s) for s, k in zip(first.fit.spectrum_id, first.fit.selected_K, strict=True)
               if k == 1)
    with caplog.at_level(logging.INFO, logger="hxex.pipeline"):
        again = run(raw, config(), out=target, overwrite=True, chosen_k={sid: 2})
    n = len(raw)
    assert f"{n} of {n} spectra keep their fit" in caplog.text
    assert "bootstrapping 1 spectra at the K chosen for them in the review" in caplog.text
    fresh = run(raw, config(), chosen_k={sid: 2})
    i = list(map(str, again.fit.spectrum_id)).index(sid)
    assert again.bootstrap_by_k[2].covers(i)
    _same_fit(again.fit, first.fit)
    _same(again.bootstrap_by_k[2], fresh.bootstrap_by_k[2], close=True)


def test_a_k_chosen_in_the_review_that_the_screen_skipped_refits_its_series(tmp_path, caplog):
    """A K the review chooses that the screen skipped for that spectrum: its series is fitted
    again (with the K chosen fitted whatever the screen says), every other series is kept, and
    the whole is a full run's with the same review."""
    base = config()
    cfg = dataclasses.replace(base, fit=dataclasses.replace(base.fit, screen=True))
    raw = raw_spectra(n_series=3)
    target = tmp_path / "results"
    first = run(raw, cfg, out=target)
    skipped = np.flatnonzero(~first.fit.fitted(2))
    assert len(skipped)
    sid = str(first.fit.spectrum_id[skipped[0]])
    with caplog.at_level(logging.INFO, logger="hxex.pipeline"):
        again = run(raw, cfg, out=target, overwrite=True, chosen_k={sid: 2})
    fresh = run(raw, cfg, chosen_k={sid: 2})
    series = np.asarray(again.fit.data_id)
    i = list(map(str, again.fit.spectrum_id)).index(sid)
    own = series == series[i]
    n = len(raw)
    assert f"{n - own.sum()} of {n} spectra keep their fit" in caplog.text
    assert again.fit.fitted(2)[i] and again.bootstrap_by_k[2].covers(i)
    _same_fit(again.fit, first.fit, np.flatnonzero(~own))
    _same_fit(again.fit, fresh.fit, close=True)
    _same(again.bootstrap, fresh.bootstrap, close=True)


def test_fits_are_kept_across_package_builds_with_the_same_fitting_code(first, tmp_path,
                                                                        caplog, capsys):
    """Fits made on a graphics card are kept on a laptop without one, and across a commit that
    changed nothing of the fitting code: the engine's fingerprint decides, not the package
    versions or the commit; what differed is logged and recorded (owner, 2026-11-11). Another
    fingerprint, or a bundle made before the fingerprint was recorded by other software, is
    fitted again, with the differences named."""
    import json

    from hxex.reuse import differences, same_engine

    now = {"hxex_commit": "1d33325", "torch": "2.14.1+cpu", "numpy": "2.4.6", "engine": "abc"}
    gpu = dict(now, hxex_commit="9c660e4", torch="2.14.1+cu126")
    assert same_engine(gpu, now) == (True, None)
    assert differences(gpu, now) == ["hxex_commit 9c660e4 -> 1d33325",
                                     "torch 2.14.1+cu126 -> 2.14.1+cpu"]
    ok, why = same_engine(dict(gpu, engine="xyz"), now)
    assert not ok and "fitting code differs" in why and "torch 2.14.1+cu126" in why
    old = {k: v for k, v in gpu.items() if k != "engine"}  # made before the fingerprint
    ok, why = same_engine(old, now)
    assert not ok and "before hxex recorded its fitting code" in why
    assert same_engine({k: v for k, v in now.items() if k != "engine"}, now)[0]

    raw = first[0]
    target = tmp_path / "results"
    run(raw, _config(), out=target)
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["fitted_with"]["engine"]
    manifest["fitted_with"]["torch"] = "0.0+gpu-build"  # made on another computer
    (target / "manifest.json").write_text(json.dumps(manifest))
    with caplog.at_level(logging.INFO):
        run(raw, _config(), out=target, overwrite=True)
    assert f"{len(raw)} of {len(raw)} spectra keep their fit" in caplog.text
    assert "made with other software (torch 0.0+gpu-build ->" in caplog.text
    prov = json.loads((target / "manifest.json").read_text())["provenance"]
    assert prov["fits_reused_made_with"][0].startswith("torch 0.0+gpu-build")
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["fitted_with"]["engine"] = "another"
    (target / "manifest.json").write_text(json.dumps(manifest))
    caplog.clear()
    with caplog.at_level(logging.INFO):
        run(raw, _config(), out=target, overwrite=True)
    assert "hxex's fitting code differs" in caplog.text and "keep their fit" not in caplog.text


def test_the_fitting_code_fingerprint_follows_the_engine_files_only():
    from hxex._engine import ENGINE, engine_fingerprint

    assert len(engine_fingerprint()) == 16 and "fit" in ENGINE and "noise.py" in ENGINE
    assert "gui" not in ENGINE and "report" not in ENGINE
