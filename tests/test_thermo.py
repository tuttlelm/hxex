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
    skip = ("reader", "thermo_reader", "runs", "files", "instruments")
    assert {k: v for k, v in src.items() if k not in skip} == {
        k: v for k, v in b.source.items() if k not in skip}
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


def test_the_recorded_resolving_power_travels_with_the_spectra(tmp_path, monkeypatch, capsys):
    """Thermo raw files record the resolving power per scan and the instrument model; the
    spectra carry them (mzML has none), the read record lists them and hxex check shows them."""
    from hxex.cli import _instrument_text
    from hxex.io.mzml import instrument_record
    from hxex.picking import PickerConfig

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
    text = _instrument_text(record, PickerConfig())
    assert "instrument Orbitrap Fake" in text and "120000 at MS1, 30000 at MS2" in text
    assert "quoted at m/z 200" in text and 'resolving_power = "recorded"' in text
    set_r = PickerConfig(pick="window_max", resolving_power=60000.0)
    assert "recorded" not in _instrument_text(record, set_r)
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
