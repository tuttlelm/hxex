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


def _with(base, **changes):
    """``base`` with settings of its fit, population test or bootstrap changed
    (``fit__k_max=3``: the table, two underscores, the key)."""
    for key, value in changes.items():
        table, name = key.split("__")
        part = getattr(base, table)
        base = dataclasses.replace(base, **{table: (dataclasses.replace(part, **{name: value})
                                                    if part is not None else value)})
    return base


def _base_without_test():
    return dataclasses.replace(_config(), population_test=None)


UPDATES = {  # name: (settings before, settings after, what is redone, every fit kept)
    "n_boot": (_config, lambda: _with(_config(), bootstrap__n_boot=4), "bootstrap", True),
    "a fixed K added": (
        lambda: _with(_config(), bootstrap__ks=(1,)), _config, "bootstrap at K = 2", True),
    "the test's settings": (
        _config, lambda: _with(_config(), population_test__simulations=6), "population test",
        True),
    "the test added": (_base_without_test, _config, "population test", True),
    "the test taken away": (_config, _base_without_test, "population test taken away", True),
    "k_max": (lambda: _with(_base_without_test(), bootstrap__ks=(1,)),
              lambda: _with(_base_without_test(), fit__k_max=3, bootstrap__ks=(1,)),
              "fits to K = 3", True),
    "d2o_fraction": (_config, lambda: _with(_config(), controls__d2o_fraction=0.8),
                     "population test", True),
    "other fit settings": (_config, lambda: _with(_config(), fit__n_restarts=3), None, False),
}


@pytest.mark.parametrize("name", list(UPDATES))
def test_an_update_gives_what_a_fresh_run_gives(first, tmp_path, caplog, name):
    """Each setting a run updates the results for (docs/PLAN.md Next steps 40): the fits kept
    where they can be, what the change touches made again, and the whole what a fresh run with
    the new settings gives."""
    raw = first[0]
    before, after, redone, kept = UPDATES[name]
    target = tmp_path / "results"
    run(raw, before(), out=target)
    with caplog.at_level(logging.INFO):
        updated = run(raw, after(), out=target, overwrite=True)
    fresh = run(raw, after())
    n = len(raw)
    if kept:
        assert f"{n} of {n} spectra keep their fit" in caplog.text
    else:
        assert "fitting every spectrum" in caplog.text and "[run.fit] changed" in caplog.text
    from hxex.io import open_bundle

    with open_bundle(target) as b:
        assert b.manifest["provenance"]["redone"] == ([redone] if redone else [])
    _same_fit(updated.fit, fresh.fit, close=True)
    if fresh.population_test is None:
        assert updated.population_test is None
    else:
        _same(updated.population_test, fresh.population_test, close=True, atol=1e-3)
    _same(updated.bootstrap, fresh.bootstrap, close=True)
    assert sorted(updated.bootstrap_by_k or {}) == sorted(fresh.bootstrap_by_k or {})
    for k, boot in (fresh.bootstrap_by_k or {}).items():
        _same(updated.bootstrap_by_k[k], boot, close=True)
    _same(updated.controls, fresh.controls, close=True)


def test_a_stopped_bootstrap_is_finished(first, tmp_path, caplog):
    """A bundle whose bootstrap stopped partway: the next run keeps the fits and the test and
    makes the bootstrap."""
    import json

    raw = first[0]
    target = tmp_path / "results"
    done = run(raw, _config(), out=target)
    manifest = target / "manifest.json"
    data = json.loads(manifest.read_text())
    data["pending"] = ["bootstrap"]
    manifest.write_text(json.dumps(data))
    with caplog.at_level(logging.INFO):
        again = run(raw, _config(), out=target, overwrite=True)
    n = len(raw)
    assert f"{n} of {n} spectra keep their fit" in caplog.text
    _same_fit(again.fit, done.fit)
    _same(again.bootstrap, done.bootstrap, close=True)


def test_a_changed_spectrum_and_changed_settings_together(first, tmp_path, caplog):
    """A changed spectrum (its series fitted again with the new settings) and a changed
    bootstrap (the kept series bootstrapped again): what a fresh run gives."""
    raw = first[0]
    target = tmp_path / "results"
    run(raw, _config(), out=target)
    changed = _changed(raw, index=3)
    other = _with(_config(), bootstrap__n_boot=4)
    with caplog.at_level(logging.INFO):
        updated = run(changed, other, out=target, overwrite=True)
    fresh = run(changed, other)
    assert "spectra keep their fit" in caplog.text
    _same_fit(updated.fit, fresh.fit, close=True)
    _same(updated.population_test, fresh.population_test, close=True, atol=1e-3)
    _same(updated.bootstrap, fresh.bootstrap, close=True)


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


def test_the_command_line_updates_and_fresh_starts_again(tmp_path, capsys):
    from hxex.cli import main
    from test_cli import mzml_project

    path = mzml_project(tmp_path)
    assert main(["run", str(path)]) == 0
    capsys.readouterr()
    # nothing changed: nothing to do, nothing written
    created = (tmp_path / "out" / "manifest.json").read_text()
    assert main(["run", str(path)]) == 0
    assert "the results are up to date, nothing to do" in capsys.readouterr().out
    assert (tmp_path / "out" / "manifest.json").read_text() == created
    # a change: an update, keeping the fits
    path.write_text(path.read_text().replace("n_boot = 3", "n_boot = 4"))
    assert main(["run", str(path)]) == 0
    err = capsys.readouterr().err
    assert "6 of 6 spectra keep their fit" in err and "bootstrap: out of date" in err
    assert main(["info", str(path)]) == 0
    out = capsys.readouterr().out
    assert "6 of 6 spectra kept the replaced bundle's fits" in out
    assert "redone         bootstrap" in out
    # --fresh: everything anew; --overwrite and --refit said to be retired
    assert main(["run", str(path), "--fresh"]) == 0
    assert "keep their fit" not in capsys.readouterr().err
    assert main(["info", str(path)]) == 0
    out = capsys.readouterr().out
    assert "0 of 6 spectra kept" in out and "started again" in out and "earlier" in out
    assert main(["run", str(path), "--overwrite"]) == 0
    assert "--overwrite is no longer needed" in capsys.readouterr().err
    assert main(["run", str(path), "--refit"]) == 0
    assert "--refit is replaced by --fresh" in capsys.readouterr().err
