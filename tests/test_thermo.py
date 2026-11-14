# SPDX-License-Identifier: GPL-3.0-or-later
"""Thermo raw reader: how the RawFileReader's values become scans (activations, precursors,
isolation windows), and, with a stand-in for the RawFileReader, that a raw file is read exactly
as the same scans in mzML. A real file is compared with its ThermoRawFileParser mzML when
HXEX_THERMO_DIR (Thermo's assemblies) and HXEX_THERMO_EXAMPLE (a .raw file with a .mzML of the
same name beside it) are set."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hxex.io.thermo as thermo
from hxex.io import (
    MzmlFile,
    MzmlRun,
    MzmlTarget,
    ScanTable,
    ThermoError,
    ThermoFile,
    mzml_series_targets,
    read_mzml,
    read_mzml_source,
    read_source,
    read_thermo,
)
from hxex.io.thermo import Reaction, ScanInfo, scan_from_info
from test_mzml import ENTRY, time_course, write_mzml  # noqa: I001

ETD, HCD, CID = ("ElectronTransferDissociation", "HigherEnergyCollisionalDissociation",
                 "CollisionInducedDissociation")


def test_scan_values_as_thermorawfileparser_writes_them():
    ms1 = scan_from_info(ScanInfo(5, 1, 0.5, "FTMS + p ESI Full ms", False, charge=3), first=5)
    assert (ms1.index, ms1.id, ms1.ms_level, ms1.time) == (
        0, "controllerType=0 controllerNumber=1 scan=5", 1, 30.0)
    assert np.isnan(ms1.precursor_mz) and ms1.precursor_charge == 0 and ms1.activation == ()
    assert np.isnan(ms1.isolation[0]) and ms1.centroid is False

    etd = scan_from_info(ScanInfo(2, 2, 0.1, "ms2", False, (Reaction(1012.0, 8.0, 0.0, ETD,
                                                                     False),), charge=20))
    assert etd.activation == ("electron transfer dissociation",)
    assert etd.precursor_mz == 1012.0 and etd.precursor_charge == 20
    assert etd.isolation == (1008.0, 1016.0)
    # the instrument's monoisotopic m/z when it found one; an isolation offset moves the window
    mono = scan_from_info(ScanInfo(2, 2, 0.1, "", True, (Reaction(500.0, 2.0, 0.5, HCD,
                                                                      False),), mono_mz=499.75))
    assert mono.precursor_mz == 499.75 and mono.isolation == (499.5, 501.5) and mono.centroid
    assert mono.activation == ("beam-type collision-induced dissociation",)

    ethcd = (Reaction(777.4, 10.0, 0.0, ETD, False), Reaction(777.4, 10.0, 0.0, HCD, True))
    s = scan_from_info(ScanInfo(3, 2, 0.1, "sa", False, ethcd, supplemental=True))
    assert s.activation == ("electron transfer dissociation",
                            "supplemental beam-type collision-induced dissociation")
    s = scan_from_info(ScanInfo(3, 2, 0.1, "", False, (ethcd[0], Reaction(777.4, 10.0, 0.0, CID,
                                                                           True))))
    assert s.activation == ("collision-induced dissociation", "electron transfer dissociation")
    # MS3: the last stage made the scan (HCD of the precursor, then ETD of the product)
    ms3 = (Reaction(493.26, 2.0, 0.0, HCD, False), Reaction(600.1, 2.0, 0.0, ETD, False),
           Reaction(600.1, 2.0, 0.0, HCD, True))
    s = scan_from_info(ScanInfo(4, 3, 0.1, "sa", False, ms3, supplemental=True))
    assert s.ms_level == 3 and s.precursor_mz == 600.1
    # every stage is kept: the parent (HCD of 493.26) and the last (EThcD of 600.1)
    assert [st.mz for st in s.stages] == [493.26, 600.1]
    t = ScanTable.from_scans([s])
    assert t.parent_mz[0] == 493.26 and (t.parent_lower[0], t.parent_upper[0]) == (492.26, 494.26)
    assert t.preactivation[0] == "beam-type collision-induced dissociation"
    assert s.activation == ("electron transfer dissociation",
                            "supplemental beam-type collision-induced dissociation")
    # parent, neutral-loss and neutral-gain scans (negative orders) count as MS2
    assert scan_from_info(ScanInfo(1, -1, 0.1, "", False, ethcd[:1])).ms_level == 2
    assert scan_from_info(ScanInfo(1, 2, 0.1, "", False, (Reaction(1.0, 0.0, 0.0, "Other",
                                                                   False),))).activation == (
        "other",)


class FakeReader:
    """Stands in for Thermo's RawFileReader: the scans of a test_mzml scan list."""

    version = "RawFileReader test"

    def __init__(self, scans, first=1):
        self.scans, self.first, self.last = scans, first, first + len(scans) - 1
        self.closed = False

    def info(self, number):
        s = self.scans[number - self.first]
        reactions = ()
        if "precursor" in s:  # the isolation window test_mzml writes: 2.5 below, 1.5 above
            pmz = s["precursor"][0]
            reactions = (Reaction(pmz, 4.0, -0.5, {"MS:1000598": ETD}.get(s["precursor"][2], HCD),
                                  False),)
        return ScanInfo(number, s.get("ms_level", 1), s["time"], s.get("filter", ""),
                        bool(s.get("centroid")), reactions,
                        charge=s["precursor"][1] if "precursor" in s else 0,
                        resolution=s.get("resolution", float("nan")))

    def instrument(self):
        return "Orbitrap Fake"

    def arrays(self, number):
        s = self.scans[number - self.first]
        return np.asarray(s["mz"], float), np.asarray(s["intensity"], float)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_raw(tmp_path, monkeypatch):
    """A .raw file of the time course with ETD scans, and the same scans as mzML."""
    scans = time_course(n=6, ms2_every=2)
    raw = tmp_path / "run.raw"
    raw.write_bytes(b"not really a raw file")
    mzml = write_mzml(tmp_path / "run.mzML", scans)
    opened = []

    def open_reader(path, thermo_dir):
        assert thermo_dir == "dlls"
        opened.append(FakeReader(scans))
        return opened[-1]

    monkeypatch.setattr(thermo, "_open_reader", open_reader)
    return raw, mzml, opened


def test_a_raw_file_reads_as_its_mzml(fake_raw):
    raw, mzml, opened = fake_raw
    with ThermoFile(raw, thermo_dir="dlls") as f:
        a, b = f.scans(), MzmlFile(mzml).scans()
        assert f.reader_version == "RawFileReader test"
        for name in ("index", "ms_level", "time", "filter", "precursor_mz", "precursor_charge",
                     "activation", "centroid", "isolation_lower", "isolation_upper"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
        assert list(a.select(activation="ETD")) == list(b.select(activation="ETD")) == [1, 4, 7]
        one, ref = f.scan(4, (600.0, 640.0)), MzmlFile(mzml).scan(4, (600.0, 640.0))
        np.testing.assert_array_equal(one.mz, ref.mz)
        np.testing.assert_array_equal(one.intensity, ref.intensity)
        assert len(one.mz) and one.mz.min() >= 600.0 and one.mz.max() <= 640.0
        with pytest.raises(ThermoError, match="no scan 99"):
            f.scan(99)
    assert opened[0].closed
    for call in (lambda: f.scan(4), lambda: f.reader_version, lambda: f.instrument):
        with pytest.raises(ThermoError, match="closed"):
            call()
    with ThermoFile(raw, thermo_dir="dlls") as g:
        pass
    with pytest.raises(ThermoError, match="closed"):
        g.scans()  # never read before the close
    with pytest.raises(ThermoError, match="no such file"):
        ThermoFile(raw.with_name("missing.raw"), thermo_dir="dlls")


def test_read_thermo_gives_the_spectra_of_read_mzml_and_repeats(fake_raw, tmp_path):
    raw, mzml, opened = fake_raw
    targets = [MzmlTarget(ENTRY), MzmlTarget(ENTRY, ms_level=2, activation="ETD")]
    a = read_thermo([MzmlRun(raw, "WT", time_offset=10.0)], targets, project="p",
                    thermo_dir="dlls")
    b = read_mzml([MzmlRun(mzml, "WT", time_offset=10.0)], targets, project="p")
    assert len(a) == len(b) == 9 and all(r.closed for r in opened)
    np.testing.assert_array_equal(a.mz, b.mz)
    np.testing.assert_array_equal(a.intensity, b.intensity)
    np.testing.assert_array_equal(a.time, b.time)
    np.testing.assert_array_equal(a.data_id, b.data_id)
    assert [s.replace("run.raw", "run.mzML") for s in a.spectrum_id] == list(b.spectrum_id)
    src = a.source
    assert src["reader"] == "thermo" and src["thermo_reader"] == "RawFileReader test"
    skip = ("reader", "thermo_reader", "runs", "files", "instruments", "profile")
    assert {k: v for k, v in src.items() if k not in skip} == {
        k: v for k, v in b.source.items() if k not in skip}
    # the raw file's FTMS scans are Fourier transform; the test mzML names no analyser
    assert [p["fourier_transform"] for p in (src["profile"][0], b.source["profile"][0])] == [
        True, None]
    assert src["instruments"][0]["model"] == "Orbitrap Fake" and b.source["instruments"] == [None]
    assert src["files"][0]["name"] == "run.raw" and src["files"][0]["algorithm"] == "sha256"
    assert mzml_series_targets(src)[1] == targets[1]

    again = read_source(src, thermo_dir="dlls")
    assert list(again.spectrum_id) == list(a.spectrum_id) and again.source == src
    np.testing.assert_array_equal(again.intensity, a.intensity)
    raw.write_bytes(b"changed")
    with pytest.raises(ThermoError, match="changed"):
        read_source(src, thermo_dir="dlls", on_change="error")
    with pytest.raises(ValueError, match="not an mzML read"):
        read_mzml_source(src)
    with pytest.raises(ValueError, match="not a Thermo raw read"):
        thermo.read_thermo_source(b.source)


def _reduced(scans):
    """The scans as an instrument's reduced profile stores them: each peak's points framed by
    zeros (here two zeros per point), so that two thirds of the points are 0."""
    out = []
    for s in scans:
        mz, y = np.asarray(s["mz"], float), np.asarray(s["intensity"], float)
        pad = np.r_[mz - 0.03, mz + 0.03]
        order = np.argsort(np.r_[mz, pad], kind="stable")
        out.append({**s, "mz": np.r_[mz, pad][order], "intensity": np.r_[y, 0 * pad][order]})
    return out


def _with_analyser(path, accession, name):
    """The mzML file with an instrument configuration naming one analyser (before its run)."""
    text = path.read_text()
    config = ('<instrumentConfigurationList count="1"><instrumentConfiguration id="IC1">'
              '<componentList count="1"><analyzer order="1">'
              f'<cvParam cvRef="MS" accession="{accession}" name="{name}"/></analyzer>'
              '</componentList></instrumentConfiguration></instrumentConfigurationList>')
    path.write_text(text.replace("<run", config + "<run", 1))
    return path


class NoiseReader(FakeReader):
    """A FakeReader whose scans record a constant noise at their centroids."""

    noise_level = 1e4

    def noise(self, number):
        mz = np.asarray(self.scans[number - self.first]["mz"], float)
        return mz, np.full(len(mz), self.noise_level)


def test_reduced_profile_scans_are_noticed(tmp_path, monkeypatch, caplog):
    """Fourier-transform profile scans with many of their points 0 are reduced (the instrument
    deleted the peaks below its noise threshold): each file's record says so per MS level, a
    Thermo file's with the weakest kept peak against its recorded noise, ion-trap scans left
    out; a read
    warns once for the dataset and records it; full profile scans are not reduced, nor are the
    zeros of an mzML file that names no Fourier-transform analyser (a Q-TOF's empty channels)."""
    from hxex.cli import _profile_text, _reduced_profile_info
    from hxex.io.mzml import profile_record, reduced_profile_text

    full = time_course(n=6, ms2_every=2)
    reduced = _reduced(full)
    trap = {**reduced[1], "filter": "ITMS + p ESI Full ms2 1012.0@etd20.00", "time": 9.0}
    with MzmlFile(write_mzml(tmp_path / "full.mzML", full)) as f:
        plain = profile_record(f)
    assert plain["reduced"] is False and set(plain["levels"]) == {"1", "2"}
    assert plain["levels"]["1"]["zero_share"] == 0 and "cut" not in plain["levels"]["1"]
    assert reduced_profile_text([plain]) is None
    assert _reduced_profile_info({"profile": [plain]}) is None
    raw = tmp_path / "run.raw"
    raw.write_bytes(b"not really a raw file")
    monkeypatch.setattr(thermo, "_open_reader", lambda path, thermo_dir: NoiseReader(
        reduced + [trap]))
    with ThermoFile(raw, thermo_dir="dlls") as f:
        record = profile_record(f)
    smallest = min(np.max(s["intensity"]) for s in full if s.get("ms_level", 1) == 1)
    assert record["reduced"] is True and record["levels"]["1"]["zero_share"] == 0.667
    assert record["levels"]["1"]["scans"] == 6 and record["levels"]["2"]["scans"] == 3  # no trap
    assert record["levels"]["1"]["cut"] <= smallest / 1e4  # the weakest peaks' tops / noise
    text = reduced_profile_text([record, plain, None])
    assert text.startswith("1 of 3 files hold reduced profile scans (MS1/2: 67% of the points")
    assert "x the noise the files record" in text and "read low or 0" in text
    assert "MS1 reduced (67% of the points 0, weakest peaks kept" in _profile_text(record)
    assert "MS1 not reduced (0% of the points 0)" in _profile_text(plain)
    targets = [MzmlTarget(ENTRY)]
    with caplog.at_level("WARNING", logger="hxex.io.mzml"):
        a = read_thermo([MzmlRun(raw, "WT")], targets, project="p", thermo_dir="dlls")
    assert a.source["profile"] == [record]
    assert sum("reduced profile scans" in r.getMessage() for r in caplog.records) == 1
    caplog.clear()
    orbitrap = _with_analyser(write_mzml(tmp_path / "ot.mzML", reduced), "MS:1000484", "orbitrap")
    unnamed = write_mzml(tmp_path / "unnamed.mzML", reduced)
    tof = _with_analyser(write_mzml(tmp_path / "tof.mzML", reduced), "MS:1000084",
                         "time-of-flight")
    assert [MzmlFile(x).fourier_transform for x in (orbitrap, unnamed, tof)] == [True, None, False]
    with caplog.at_level("WARNING", logger="hxex.io.mzml"):
        b = read_mzml([MzmlRun(tmp_path / "full.mzML", "WT")], targets, project="p")
        c = read_mzml([MzmlRun(orbitrap, "WT")], targets, project="p")
        d = read_mzml([MzmlRun(unnamed, "WT"), MzmlRun(tof, "WT", rep=2)], targets, project="p")
    assert b.source["profile"] == [plain] and c.source["profile"][0]["reduced"] is True
    assert "cut" not in c.source["profile"][0]["levels"]["1"]  # mzML records no noise
    assert [r["reduced"] for r in d.source["profile"]] == [False, False]
    assert [r["fourier_transform"] for r in d.source["profile"]] == [None, False]
    assert d.source["profile"][0]["levels"]["1"]["zero_share"] == 0.667  # recorded all the same
    assert sum("reduced profile scans" in r.getMessage() for r in caplog.records) == 1


def test_scan_cut_auto_acts_on_reduced_files_that_record_noise(tmp_path, monkeypatch, caplog):
    """[input.scans] scan_cut = "auto" (the default): fragment spectra of a reduced profile file
    that records its scans' noise carry the scans' record; a reduced mzML file without noise is
    read without it and named in the warning; full profile files and scan_cut = false record
    nothing; the read record lists the files cut; hxex info says what was done."""
    import dataclasses

    from hxex.cli import _reduced_profile_info
    from hxex.fragments import fragment_targets
    from hxex.io.mzml import MzmlConfig
    from test_mzml import _with_rt, fragment_lc_run

    parent = dataclasses.replace(_with_rt(ENTRY, 30.0 / 60), extra={
        **dict(_with_rt(ENTRY, 30.0 / 60).extra), "rt_start": str(20.0 / 60),
        "rt_end": str(40.0 / 60)})
    frags = fragment_targets(parent, "c:2-4:1", activation="ETD", rt_window=(20.0, 40.0))
    targets = [MzmlTarget(parent, rt_window=(20.0, 40.0))] + frags
    lc = fragment_lc_run(frags, other=None)
    scans = [{**{k: v for k, v in s.items() if k != "precursors"}, "time": s["time"] / 60,
              **({"precursor": s["precursors"][0][:3],
                  "filter": f"FTMS + p ESI d Full ms2 {s['precursors'][0][0]:.4f}@etd20.00"}
                 if "precursors" in s else {})} for s in lc]

    class Noise500(NoiseReader):
        noise_level = 500.0

    raw = tmp_path / "run.raw"
    raw.write_bytes(b"not really a raw file")
    runs = [MzmlRun(path=str(raw), sample="S", time=60.0)]
    config = MzmlConfig(find_fragment_peak=True)
    assert config.scan_cut == "auto"
    monkeypatch.setattr(thermo, "_open_reader", lambda path, thermo_dir: Noise500(
        _reduced(scans)))
    with caplog.at_level("WARNING", logger="hxex.io.mzml"):
        read = read_thermo(runs, targets, project="p", config=config, thermo_dir="dlls")
    is_frag = np.array(["ion_type" in m for m in read.modification])
    assert all(r is not None for r in read.scans[is_frag])
    assert all(r is None for r in read.scans[~is_frag])
    assert read.source["scan_cut"] == {"files": ["run.raw"], "no_noise": []}
    text = next(r.getMessage() for r in caplog.records if "reduced profile" in r.getMessage())
    assert "fits the fragment spectra of the reduced file through this cut" in text
    assert "fragment spectra without allowing" not in text
    assert "through this cut" in _reduced_profile_info(read.source)
    off = read_thermo(runs, targets, project="p", thermo_dir="dlls",
                      config=MzmlConfig(find_fragment_peak=True, scan_cut=False))
    assert all(r is None for r in off.scans) and off.source["scan_cut"]["files"] == []
    assert "scan_cut = false" in _reduced_profile_info(off.source)
    # full profile scans: nothing to correct, nothing recorded
    monkeypatch.setattr(thermo, "_open_reader", lambda path, thermo_dir: Noise500(scans))
    full = read_thermo(runs, targets, project="p", config=config, thermo_dir="dlls")
    assert all(r is None for r in full.scans) and full.source["scan_cut"]["files"] == []
    # a reduced Orbitrap mzML without -N: read without the record, named in the warning
    mzml = _with_analyser(write_mzml(tmp_path / "ot.mzML", _reduced(lc)), "MS:1000484",
                          "orbitrap")
    caplog.clear()
    with caplog.at_level("WARNING", logger="hxex.io.mzml"):
        plain = read_mzml([MzmlRun(path=str(mzml), sample="S", time=60.0)], targets,
                          project="p", config=config)
    assert all(r is None for r in plain.scans)
    assert plain.source["scan_cut"] == {"files": [], "no_noise": ["ot.mzML"]}
    text = next(r.getMessage() for r in caplog.records if "reduced profile" in r.getMessage())
    assert ("the reduced file records no noise, so the fragment spectra of those files are "
            "fitted without allowing for it") in text
    assert "ThermoRawFileParser's -N" in text


def test_scan_cut_on_a_run_without_a_fixed_time_read_in_windows(tmp_path, monkeypatch):
    """A run without a fixed time read with group = "windows" reads only the scans inside its
    windows; scan_cut (here "auto", the files reduced and recording noise) gives each window's
    fragment spectrum its scans' record from those scans alone (it once looked up the scans
    outside the windows and failed)."""
    import dataclasses

    from hxex.fragments import fragment_targets
    from hxex.io.mzml import MzmlConfig
    from test_mzml import _with_rt, fragment_lc_run

    parent = dataclasses.replace(_with_rt(ENTRY, 30.0 / 60), extra={
        **dict(_with_rt(ENTRY, 30.0 / 60).extra), "rt_start": str(20.0 / 60),
        "rt_end": str(40.0 / 60)})
    frags = fragment_targets(parent, "c:2-4:1", activation="ETD", rt_window=(20.0, 40.0))
    targets = [MzmlTarget(parent, rt_window=(20.0, 40.0))] + frags
    scans = [{**{k: v for k, v in s.items() if k != "precursors"}, "time": s["time"] / 60,
              **({"precursor": s["precursors"][0][:3],
                  "filter": f"FTMS + p ESI d Full ms2 {s['precursors'][0][0]:.4f}@etd20.00"}
                 if "precursors" in s else {})} for s in fragment_lc_run(frags, other=None)]

    class Noise500(NoiseReader):
        noise_level = 500.0

    monkeypatch.setattr(thermo, "_open_reader", lambda path, thermo_dir: Noise500(
        _reduced(scans)))
    raw = tmp_path / "run.raw"
    raw.write_bytes(b"not really a raw file")
    read = read_thermo([MzmlRun(path=str(raw), sample="S")], targets, project="p",
                       thermo_dir="dlls",
                       config=MzmlConfig(group="windows", windows=((27.0, 33.0),)))
    is_frag = np.array(["ion_type" in m for m in read.modification])
    assert is_frag.sum() == 3 and read.source["scan_cut"]["files"] == ["run.raw"]
    for i in np.flatnonzero(is_frag):
        rec = read.scans[i]
        assert rec is not None and rec.weights.sum() == pytest.approx(1.0)


def test_the_recorded_resolving_power_travels_with_the_spectra(tmp_path, monkeypatch, capsys):
    """Thermo raw files record the resolving power per scan and the instrument model; the
    spectra carry them (mzML has none), the read record lists them and hxex check shows them."""
    from hxex.cli import _instrument_text
    from hxex.io.mzml import instrument_record

    scans = time_course(n=6, ms2_every=2)
    for s in scans:
        s["resolution"] = 120000.0 if s.get("ms_level", 1) == 1 else 30000.0
    raw = tmp_path / "run.raw"
    raw.write_bytes(b"not really a raw file")
    mzml = write_mzml(tmp_path / "run.mzML", scans)
    monkeypatch.setattr(thermo, "_open_reader", lambda path, thermo_dir: FakeReader(scans))
    targets = [MzmlTarget(ENTRY), MzmlTarget(ENTRY, ms_level=2, activation="ETD")]
    with ThermoFile(raw, thermo_dir="dlls") as f:
        table = f.scans()
        assert f.instrument == "Orbitrap Fake" and f.resolving_power_mz == 200.0
        assert set(table.resolution[table.ms_level == 1]) == {120000.0}
        record = instrument_record(f)
    assert record == {"model": "Orbitrap Fake", "resolving_power_mz": 200.0,
                      "resolving_power": {"1": 120000.0, "2": 30000.0}}
    text = _instrument_text(record)
    assert "instrument Orbitrap Fake" in text and "120000 at MS1, 30000 at MS2" in text
    assert "quoted at m/z 200" in text and "resolving_power" not in text  # no setting to suggest
    a = read_thermo([MzmlRun(raw, "WT")], targets, project="p", thermo_dir="dlls")
    ms1 = a.data_id == 0
    assert set(a.resolving_power[ms1]) == {120000.0} and set(a.resolving_power[~ms1]) == {30000.0}
    assert set(a.resolving_power_mz) == {200.0} and a.source["instruments"] == [record]
    b = read_mzml([MzmlRun(mzml, "WT")], targets, project="p")
    assert np.isnan(b.resolving_power).all() and b.source["instruments"] == [None]
    assert np.isnan(MzmlFile(mzml).scans().resolution).all()


def test_what_reading_needs_is_named(tmp_path, monkeypatch):
    monkeypatch.delenv(thermo.THERMO_DIR_ENV, raising=False)
    raw = tmp_path / "x.raw"
    raw.write_bytes(b"x")
    with pytest.raises(ThermoError, match="HXEX_THERMO_DIR"):
        ThermoFile(raw)
    with pytest.raises(ThermoError, match="no ThermoFisher.CommonCore.Data.dll"):
        ThermoFile(raw, thermo_dir=tmp_path)
    monkeypatch.setenv(thermo.THERMO_DIR_ENV, str(tmp_path))
    with pytest.raises(ThermoError, match="RawFileReader.dll"):
        ThermoFile(raw)


# --- a real file --------------------------------------------------------------------------------

REAL = os.environ.get("HXEX_THERMO_EXAMPLE")
DLLS = os.environ.get(thermo.THERMO_DIR_ENV)


@pytest.mark.skipif(not (REAL and DLLS), reason="HXEX_THERMO_EXAMPLE or HXEX_THERMO_DIR not set")
def test_a_real_raw_file_matches_its_thermorawfileparser_mzml():
    raw, mzml = Path(REAL), Path(REAL).with_suffix(".mzML")
    with ThermoFile(raw) as f:
        a, b = f.scans(), MzmlFile(mzml).scans()
        assert len(a) == len(b) > 0
        for name in ("index", "id", "ms_level", "filter", "precursor_charge", "centroid"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
        for name in ("time", "precursor_mz", "isolation_lower", "isolation_upper"):  # as text
            np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=1e-12,
                                       err_msg=name)
        # MS1 and MS2 activations as ThermoRawFileParser's (it is not consistent for MS3)
        low = a.ms_level <= 2
        np.testing.assert_array_equal(a.activation[low], b.activation[low])
        picks = np.unique(np.linspace(0, len(a) - 1, 20).astype(int))
        scans, m = f.read_scans(picks), MzmlFile(mzml)
        for i in picks:
            ref = m.scan(int(i))
            np.testing.assert_array_equal(scans[i].mz, ref.mz)
            np.testing.assert_allclose(scans[i].intensity, ref.intensity, rtol=1e-6)
    run_raw, run_mzml = MzmlRun(raw, "S", time=0.0), MzmlRun(mzml, "S", time=0.0)
    x = read_thermo([run_raw], [MzmlTarget(ENTRY)], project="p")
    y = read_mzml([run_mzml], [MzmlTarget(ENTRY)], project="p")
    np.testing.assert_array_equal(x.mz, y.mz)
    np.testing.assert_allclose(x.intensity, y.intensity, rtol=1e-6)


_TRACE_IN_A_THREAD = """
import json, sys, threading
from hxex.io.retention_trace import TraceReader
record = json.loads(open(sys.argv[1]).read())
reader = TraceReader(record["source"])  # made in the main thread, as the window makes it
out = []
thread = threading.Thread(
    target=lambda: out.append(reader.traces([record["id"]], record["data_id"])))
thread.start()
thread.join()
reader.close()  # in the main thread, as the window closes it
print("traces", len(out[0].traces))
"""


@pytest.mark.skipif(not (REAL and DLLS), reason="HXEX_THERMO_EXAMPLE or HXEX_THERMO_DIR not set")
def test_a_trace_read_in_a_worker_thread_leaves_the_process_whole(tmp_path):
    """The elution tab reads Thermo files in a worker thread, in a window whose results came
    from a bundle (its main thread had not used .NET): .NET must be started in the thread that
    makes the reader, or the process aborts when the main thread next touches .NET (here at
    exit; in the window as the tab was drawn)."""
    import json
    import subprocess
    import sys

    spectra = read_thermo([MzmlRun(Path(REAL), "S", time=0.0)], [MzmlTarget(ENTRY)],
                          project="p")
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"source": spectra.source, "id": str(spectra.spectrum_id[0]),
                                  "data_id": int(spectra.data_id[0])}))
    script = tmp_path / "trace.py"
    script.write_text(_TRACE_IN_A_THREAD)
    import hxex  # the subprocess imports this same package

    env = dict(os.environ, PYTHONPATH=str(Path(hxex.__file__).resolve().parents[1]))
    done = subprocess.run([sys.executable, str(script), str(record)], capture_output=True,
                          text=True, timeout=600, cwd=tmp_path, env=env)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "traces 1" in done.stdout and "Native Crash" not in done.stderr


# --- the RawFileReader handle is released when opening fails ----------------------------------

class FakeRaw:
    """What the RawFileReader's FileFactory returns, as far as opening uses it."""

    def __init__(self, fail=None, ms_count=1):
        self.IsError, self.InAcquisition, self.disposed = False, False, False
        self.IncludeReferenceAndExceptionData = False
        self._fail, self._count = fail, ms_count

    def GetInstrumentCountOfType(self, kind):
        if self._fail is not None:
            raise self._fail
        return self._count

    def SelectInstrument(self, kind, number):
        pass

    @property
    def RunHeaderEx(self):
        return SimpleNamespace(FirstSpectrum=1, LastSpectrum=3)

    def GetTrailerExtraHeaderInformation(self):
        return SimpleNamespace(Length=0)

    def Dispose(self):
        self.disposed = True


@pytest.mark.parametrize("raw, error", [
    (FakeRaw(fail=RuntimeError("the .NET call failed")), RuntimeError),
    (FakeRaw(ms_count=0), ThermoError),
    (FakeRaw(), None),
])
def test_opening_releases_the_file_when_a_call_fails(tmp_path, monkeypatch, raw, error):
    api = SimpleNamespace(adapter=SimpleNamespace(FileFactory=lambda path: raw),
                          device=SimpleNamespace(MS=1), version="test")
    monkeypatch.setattr(thermo, "_thermo_api", lambda thermo_dir: api)
    path = tmp_path / "run.raw"
    path.write_bytes(b"not really a raw file")
    if error is not None:
        with pytest.raises(error):
            thermo._RawFileReader(path, None)
        assert raw.disposed
        return
    reader = thermo._RawFileReader(path, None)
    assert not raw.disposed and (reader.first, reader.last) == (1, 3)
    assert raw.IncludeReferenceAndExceptionData
    reader.close()
    assert raw.disposed


def test_trailer_values_written_as_decimals():
    """A trailer ``Charge State:`` written ``2.0`` reads as 2, not as the default 0
    (Readers 35)."""
    from hxex.io.thermo import trailer_value

    assert trailer_value("2.0", 0) == 2 and trailer_value("2", 0) == 2
    assert trailer_value(2.0, 0) == 2 and trailer_value("", 0) == 0
    assert trailer_value("abc", 0) == 0 and trailer_value(None, 0) == 0
    assert trailer_value("523.25", 0.0) == 523.25 and trailer_value("", 0.0) == 0.0


def test_scan_cut_records_each_fragment_scans_noise_and_share(tmp_path, monkeypatch):
    """[input.scans] scan_cut (PLAN Next steps 24): every fragment spectrum carries its scans'
    shares of the signal (from the channel's whole ladder: they follow the elution and sum to
    1; scans without signal left out) and the noise each scan recorded across its window; the
    parent's MS1 spectrum carries none, nor does a read without the setting; the record goes
    through a spectra file and is cut down to the bins when picked. mzML files, which record no
    noise unless converted with -N, are refused; converted with -N they give the same record."""
    import dataclasses

    from hxex.fragments import fragment_targets
    from hxex.io import load_spectra, save_spectra
    from hxex.io.mzml import MzmlConfig, MzmlError
    from hxex.picking import pick_spectra
    from test_mzml import _with_rt, fragment_lc_run

    parent = dataclasses.replace(_with_rt(ENTRY, 30.0 / 60), extra={
        **dict(_with_rt(ENTRY, 30.0 / 60).extra), "rt_start": str(20.0 / 60),
        "rt_end": str(40.0 / 60)})
    frags = fragment_targets(parent, "c:2-4:1", activation="ETD", rt_window=(20.0, 40.0))
    targets = [MzmlTarget(parent, rt_window=(20.0, 40.0))] + frags
    scans = [{**{k: v for k, v in s.items() if k != "precursors"}, "time": s["time"] / 60,
              # as a raw file has them (times in minutes, the precursor in the filter)
              **({"precursor": s["precursors"][0][:3],
                  "filter": f"FTMS + p ESI d Full ms2 {s['precursors'][0][0]:.4f}@etd20.00"}
                 if "precursors" in s else {})}
             for s in fragment_lc_run(frags, other=None)]
    raw = tmp_path / "run.raw"
    raw.write_bytes(b"not really a raw file")
    class Noise500(NoiseReader):
        noise_level = 500.0

    monkeypatch.setattr(thermo, "_open_reader", lambda path, thermo_dir: Noise500(scans))
    runs = [MzmlRun(path=str(raw), sample="S", time=60.0)]
    config = MzmlConfig(find_fragment_peak=True, scan_cut=True)
    read = read_thermo(runs, targets, project="p", config=config, thermo_dir="dlls")
    is_frag = np.array(["ion_type" in m for m in read.modification])
    assert is_frag.sum() == 3 and all(r is None for r in read.scans[~is_frag])
    for i in np.flatnonzero(is_frag):
        rec = read.scans[i]
        assert rec.weights.sum() == pytest.approx(1.0) and np.all(rec.weights > 0)
        assert rec.noise.shape == (len(rec.weights), len(rec.noise_mz)) == (len(rec.weights), 5)
        np.testing.assert_allclose(rec.noise, 500.0)
        mz, _ = read.spectrum(i)
        assert rec.noise_mz[0] <= mz.min() and rec.noise_mz[-1] >= mz.max()
        # the shares follow the elution: the middle scans hold the most
        assert np.argmax(rec.weights) in range(len(rec.weights) // 3, 2 * len(rec.weights) // 3 + 1)
    assert read.source["config"]["scan_cut"] is True
    # through a spectra file, and down to the bins
    save_spectra(read, tmp_path / "spectra.parquet")
    again = load_spectra(tmp_path / "spectra.parquet")
    for a, b in zip(read.scans, again.scans, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            np.testing.assert_array_equal(a.noise, b.noise)
            np.testing.assert_array_equal(a.weights, b.weights)
    binned = pick_spectra(read)
    for i in range(len(binned)):
        cut = binned.scan_cut[i]
        assert (cut is None) == (not is_frag[i])
        if cut is not None:
            assert cut.noise.shape == (len(binned.spectrum(i)[0]), len(cut.weights))
    # without the setting nothing is recorded
    plain = read_thermo(runs, targets, project="p", thermo_dir="dlls",
                        config=MzmlConfig(find_fragment_peak=True))
    assert all(r is None for r in plain.scans)
    # an mzML file records no noise, unless converted with -N
    from test_mzml import _array

    mzml = write_mzml(tmp_path / "run.mzML", fragment_lc_run(frags, other=None))
    with pytest.raises(MzmlError, match="scan_cut needs the noise"):
        read_mzml([MzmlRun(path=str(mzml), sample="S", time=60.0)], targets, project="p",
                  config=config)
    noise = (_array(np.linspace(100, 2000, 400), "MS:1002743", name="sampled noise m/z array")
             + _array(np.full(400, 500.0), "MS:1002744", name="sampled noise intensity array"))
    converted = tmp_path / "converted.mzML"
    converted.write_text(mzml.read_text().replace("</binaryDataArrayList>",
                                                  noise + "</binaryDataArrayList>"))
    from_mzml = read_mzml([MzmlRun(path=str(converted), sample="S", time=60.0)], targets,
                          project="p", config=config)
    for a, b in zip(read.scans, from_mzml.scans, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            np.testing.assert_allclose(b.noise, a.noise)
            np.testing.assert_allclose(b.weights, a.weights)
