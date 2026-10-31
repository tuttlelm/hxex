# SPDX-License-Identifier: GPL-3.0-or-later
"""Finding a series' retention time from its exact m/z (hxex.io.retention_search; HISTORY 201).
Synthetic LC runs: five peptides at their table times; one of them has another species 30 ppm
off at its table time, ten times taller, and elutes itself 15 s later at its exact m/z (the
situation of the Orbitrap digest, HISTORY 200)."""

import dataclasses
import logging

import numpy as np
import pytest

from hxex.io import MzmlConfig, MzmlRun, MzmlTarget, read_mzml
from hxex.io.retention_search import scan_tops, suggestions_table
from hxex.picking import plan_bins
from hxex.spectra import PeptideEntry
from test_mzml import write_mzml

PEPTIDES = {  # peptide -> table time (s)
    "AGNLGGGVVTIER": 12.0, "LNGTDPEDVIR": 20.0, "VDQIIMAK": 38.0, "GTGIVSAPVPK": 52.0,
    "IWDLEGK": 30.0,
}
OFF = "IWDLEGK"  # read at 30 s: another species 30 ppm off; the peptide elutes at 45 s
TRUE_RT = 45.0
Z = 2


def _entry(peptide):
    n = len(peptide)
    return PeptideEntry("", "", peptide, f"0001-{n:04d}", 1, n, Z,
                        extra={"rt": str(PEPTIDES[peptide] / 60)})


def _envelope(peptide, height, ppm=0.0):
    """The peptide's undeuterated profile envelope (natural abundance), moved by ``ppm``."""
    plan = plan_bins(peptide, Z)
    shape, offsets = plan.clusters
    mz, y = [], []
    for k in range(4):
        centre = (plan.undeut_mz + offsets[k] / Z) * (1 + ppm * 1e-6)
        x = centre + np.arange(-0.016, 0.0161, 0.002)
        mz.append(x)
        y.append(height * shape[k] / shape[0] * np.exp(-0.5 * ((x - centre) / 0.004) ** 2))
    return np.concatenate(mz), np.concatenate(y)


def lc_run(seed, shift=0.0):
    """MS1 scans every 0.25 s for 60 s (``shift``: every elution later by this, s)."""
    rng = np.random.default_rng(seed)
    scans = []
    species = [(p, rt, 1e5, 0.0) for p, rt in PEPTIDES.items() if p != OFF]
    species += [(OFF, PEPTIDES[OFF], 1e6, 30.0),  # the other species, at the table's time
                (OFF, TRUE_RT, 1e5, 0.0)]  # the peptide itself
    for t in np.arange(0.0, 60.0 + 1e-9, 0.25):
        parts = []
        for peptide, apex, height, ppm in species:
            h = height * np.exp(-0.5 * ((t - apex - shift) / 2.0) ** 2)
            if h > 1:
                parts.append(_envelope(peptide, h, ppm))
        mz = np.concatenate([p[0] for p in parts]) if parts else np.array([400.0])
        y = np.concatenate([p[1] for p in parts]) if parts else np.array([0.0])
        order = np.argsort(mz)
        y = np.clip(y[order] * (1 + 0.01 * rng.standard_normal(len(y))), 0, None)
        scans.append({"mz": mz[order], "intensity": y, "time": t, "unit": "s",
                      "filter": "FTMS + p ESI Full ms"})
    return scans


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("lc")
    return [MzmlRun(path=str(write_mzml(d / "un1.mzML", lc_run(1))), sample="S", rep=1,
                    role="undeut"),
            MzmlRun(path=str(write_mzml(d / "un2.mzML", lc_run(2, shift=0.3))), sample="S",
                    rep=2, role="undeut"),
            MzmlRun(path=str(write_mzml(d / "t60.mzML", lc_run(3))), sample="S", rep=1,
                    time=60.0)]


def _read(runs, mode, **kw):
    targets = [MzmlTarget(_entry(p)) for p in PEPTIDES]
    return read_mzml(runs, targets, project="p",
                     config=MzmlConfig(find_peak=True, find_retention_time=mode, **kw))


def _rt(spectra, peptide):
    return 60 * np.asarray(spectra.rt)[np.asarray(spectra.peptide) == peptide]


def test_scan_tops_finds_profile_and_centroided_tops():
    mz, y = _envelope("VDQIIMAK", 1e5, ppm=5.0)
    x, h = scan_tops(mz, y, centroid=False)
    plan = plan_bins("VDQIIMAK", Z)
    assert len(x) == 4
    assert (x[0] / plan.undeut_mz - 1) * 1e6 == pytest.approx(5.0, abs=0.5)
    assert h[0] == pytest.approx(1e5, rel=0.01)
    c = scan_tops(np.array([500.0, 500.5, 501.0]), np.array([1.0, 0.0, 2.0]), centroid=True)
    assert list(c[0]) == [500.0, 501.0]


def test_suggest_finds_the_peptide_elsewhere_and_changes_nothing(runs, caplog):
    with caplog.at_level(logging.INFO, logger="hxex.io.retention_search"):
        suggested = _read(runs, "suggest")
    off = _read(runs, "off")
    # nothing read differently: the series is still read at the other species
    assert np.array_equal(suggested.intensity, off.intensity)
    assert np.all(np.abs(_rt(suggested, OFF) - PEPTIDES[OFF]) < 2.0)
    record = suggested.source["retention_search"]
    assert "retention_search" not in off.source
    assert record["mode"] == "suggest"
    rows = {r["peptide"]: r for r in record["series"]}
    assert set(rows) == {OFF}  # the others pass the m/z check: not searched
    row = rows[OFF]
    assert row["searched_because"] == "off_mz" and row["verdict"] == "elsewhere"
    assert 60 * row["proposed_rt"] == pytest.approx(TRUE_RT + 0.15, abs=1.0)
    assert row["n_injections"] == 2 and not row["applied"]
    assert 60 * row["read_rt"] == pytest.approx(PEPTIDES[OFF], abs=2.0)
    assert record["rt_tolerance_s"] > 0 and len(record["files"]) == 2
    assert all(f["tolerance_ppm"] == pytest.approx(1.25 * f["limit_ppm"])
               for f in record["files"])
    assert "exact m/z elutes at another time" in caplog.text
    table = suggestions_table(record)
    assert table["verdict"] == ["elsewhere"] and table["applied"] == [False]


def test_automatic_reads_the_series_at_the_time_found(runs):
    moved = _read(runs, "automatic")
    off = _read(runs, "off")
    rt = _rt(moved, OFF)
    assert len(rt) == 3 and np.all(np.abs(rt - TRUE_RT) < 2.0)  # every run, the controls too
    for p in PEPTIDES:  # the others are read as before
        if p != OFF:
            assert np.allclose(_rt(moved, p), _rt(off, p))
    row = moved.source["retention_search"]["series"][0]
    assert row["applied"] and row["data_id"] is not None
    assert set(np.asarray(moved.data_id)[np.asarray(moved.peptide) == OFF]) == {row["data_id"]}


def test_a_series_read_at_its_own_time_is_not_searched(runs):
    """With the table's time right, the m/z check passes every series and nothing is searched;
    rt_search_tolerance replaces the measured retention tolerance."""
    targets = [MzmlTarget(_entry(p) if p != OFF else dataclasses.replace(
        _entry(p), extra={"rt": str(TRUE_RT / 60)})) for p in PEPTIDES]
    read = read_mzml(runs, targets, project="p",
                     config=MzmlConfig(find_peak=True, rt_search_tolerance=7.0))
    assert np.all(np.abs(_rt(read, OFF) - TRUE_RT) < 2.0)
    record = read.source.get("retention_search")
    assert record is not None and record["series"] == []
    assert record["rt_tolerance_s"] == 7.0


def test_settings_are_checked():
    with pytest.raises(ValueError, match="find_retention_time"):
        MzmlConfig(find_retention_time="sometimes")
    with pytest.raises(ValueError, match="needs find_peak"):
        MzmlConfig(find_retention_time="automatic")
    with pytest.raises(ValueError, match="rt_search_ppm"):
        MzmlConfig(find_peak=True, rt_search_ppm=0)
    assert MzmlConfig().find_retention_time == "suggest"
    cfg = MzmlConfig(find_peak=True, find_retention_time="automatic", rt_search_ppm=4.0,
                     rt_search_tolerance=6.0)
    assert MzmlConfig.from_dict(cfg.to_dict()) == cfg


def test_the_run_flags_rt_found_and_writes_the_suggestions(runs, tmp_path):
    from hxex import load_results
    from hxex.fit import FitConfig
    from hxex.model import Backend
    from hxex.noise import NoiseConfig
    from hxex.report.tables import write_tables
    from hxex.run import RunConfig, run

    fit = FitConfig(k_max=1, adam_steps=40, n_restarts=1, backend=Backend("cpu", "float64"),
                    noise=NoiseConfig("gaussian", percent=3.0))
    result = run(_read(runs, "automatic"), RunConfig(fit=fit), out=tmp_path / "b")
    off = np.asarray(result.binned.peptide)[result.kept] == OFF
    assert np.all(result.quality.rt_found[off]) and not np.any(result.quality.rt_found[~off])
    loaded = load_results(tmp_path / "b")
    assert np.array_equal(loaded.quality.rt_found, result.quality.rt_found)
    written = {p.name for p in write_tables(loaded, tmp_path / "t")}
    assert "retention_suggestions.csv" in written
    text = (tmp_path / "t" / "retention_suggestions.csv").read_text()
    assert OFF in text and "elsewhere" in text
    # without a search there is no table
    plain = run(_read(runs, "off"), RunConfig(fit=fit), out=tmp_path / "c")
    assert not np.any(plain.quality.rt_found)
    assert "retention_suggestions.csv" not in {
        p.name for p in write_tables(load_results(tmp_path / "c"), tmp_path / "u")}
