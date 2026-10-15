# SPDX-License-Identifier: GPL-3.0-or-later
"""mzML reader: synthetic files written here (every encoding the reader supports), summing,
both experiment layouts, and an end-to-end run. A real file can be checked with
HXEX_MZML_EXAMPLE (an mzML file written by ThermoRawFileParser)."""

import base64
import dataclasses
import hashlib
import json
import logging
import os
import zlib

import numpy as np
import pytest

from hxex.io import (
    MzmlConfig,
    MzmlError,
    MzmlFile,
    MzmlRun,
    MzmlTarget,
    file_identity,
    iter_scans,
    read_mzml,
    sum_centroid,
    sum_profile,
)
from hxex.picking import plan_bins
from hxex.spectra import PeptideEntry

NS = "http://psi.hupo.org/ms/mzml"
PEPTIDE, CHARGE = "RDKVQKEYAL", 2
ENTRY = PeptideEntry("", "", PEPTIDE, "0001-0010", 1, 10, CHARGE)


def _array(values, accession, *, float64=True, compress=True, numpress=False):
    raw = np.asarray(values, dtype="<f8" if float64 else "<f4").tobytes()
    if compress:
        raw = zlib.compress(raw)
    params = [f'<cvParam cvRef="MS" accession="{accession}" name="array" value=""/>',
              f'<cvParam cvRef="MS" accession="{"MS:1000523" if float64 else "MS:1000521"}" '
              'name="float" value=""/>',
              f'<cvParam cvRef="MS" accession="{"MS:1000574" if compress else "MS:1000576"}" '
              'name="compression" value=""/>']
    if numpress:
        params.append('<cvParam cvRef="MS" accession="MS:1002312" name="numpress" value=""/>')
    text = base64.b64encode(raw).decode()
    return (f'<binaryDataArray encodedLength="{len(text)}">{"".join(params)}'
            f"<binary>{text}</binary></binaryDataArray>")


def write_mzml(path, scans, *, indexed=False, float64=True, compress=True, numpress=False):
    """scans: dicts with mz, intensity, time (minutes unless unit='s'), and optionally ms_level,
    filter, centroid (bool), precursor (m/z, charge, activation accession and name) or
    precursors (several, first stage to last)."""
    parts = []
    for i, s in enumerate(scans):
        unit = ("UO:0000010", "second") if s.get("unit") == "s" else ("UO:0000031", "minute")
        kind = "MS:1000127" if s.get("centroid") else "MS:1000128"
        precursors = s.get("precursors") or ([s["precursor"]] if "precursor" in s else [])
        prec = ""
        for pmz, pz, acc, name in precursors:
            prec += (
                '<precursor><isolationWindow>'
                f'<cvParam cvRef="MS" accession="MS:1000827" value="{pmz}"/>'
                '<cvParam cvRef="MS" accession="MS:1000828" value="2.5"/>'
                '<cvParam cvRef="MS" accession="MS:1000829" value="1.5"/></isolationWindow>'
                '<selectedIonList count="1"><selectedIon>'
                '<cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" '
                f'value="{pmz}"/>'
                f'<cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="{pz}"/>'
                "</selectedIon></selectedIonList><activation>"
                f'<cvParam cvRef="MS" accession="{acc}" name="{name}" value=""/>'
                '<cvParam cvRef="MS" accession="MS:1000045" name="collision energy" value="20"/>'
                "</activation></precursor>"
            )
        if precursors:
            prec = f'<precursorList count="{len(precursors)}">{prec}</precursorList>'
        parts.append(
            f'<spectrum id="scan={i + 1}" index="{i}" defaultArrayLength="{len(s["mz"])}">'
            '<cvParam cvRef="MS" accession="MS:1000511" name="ms level" '
            f'value="{s.get("ms_level", 1)}"/>'
            f'<cvParam cvRef="MS" accession="{kind}" name="spectrum type" value=""/>'
            '<scanList count="1"><scan>'
            '<cvParam cvRef="MS" accession="MS:1000016" name="scan start time" '
            f'value="{s["time"]}" '
            f'unitAccession="{unit[0]}" unitName="{unit[1]}"/>'
            '<cvParam cvRef="MS" accession="MS:1000512" name="filter string" '
            f'value="{s.get("filter", "")}"/>'
            "</scan></scanList>" + prec +
            '<binaryDataArrayList count="2">'
            + _array(s["mz"], "MS:1000514", float64=float64, compress=compress, numpress=numpress)
            + _array(s["intensity"], "MS:1000515", float64=float64, compress=compress,
                     numpress=numpress)
            + "</binaryDataArrayList></spectrum>"
        )
    body = (f'<mzML xmlns="{NS}" version="1.1.0"><run id="r"><spectrumList count="{len(scans)}">'
            + "".join(parts) + '</spectrumList><chromatogramList count="1"><chromatogram id="TIC" '
            'index="0" defaultArrayLength="0"/></chromatogramList></run></mzML>')
    if not indexed:
        path.write_text('<?xml version="1.0" encoding="utf-8"?>' + body)
        return path
    head = f'<?xml version="1.0" encoding="utf-8"?><indexedmzML xmlns="{NS}">'
    doc = (head + body).encode()
    offsets, start = [], 0
    while (pos := doc.find(b"<spectrum ", start)) >= 0:
        offsets.append(pos)
        start = pos + 1
    index = ('<indexList count="1"><index name="spectrum">'
             + "".join(f'<offset idRef="scan={i + 1}">{o}</offset>' for i, o in enumerate(offsets))
             + "</index></indexList>")
    index_offset = len(doc)
    # the file's own SHA-1, of everything up to and including the opening checksum tag
    doc += (index + f"<indexListOffset>{index_offset}</indexListOffset><fileChecksum>").encode()
    sha1 = hashlib.sha1(doc).hexdigest()
    path.write_bytes(doc + f"{sha1}</fileChecksum></indexedmzML>".encode())
    return path


def profile_envelope(center_bin, width, height, *, shift=0.0, gap_points=True):
    """A profile spectrum of the test peptide: a narrow Gaussian peak at each predicted bin, with
    nothing written between peaks (as Thermo leaves out zero stretches)."""
    plan = plan_bins(PEPTIDE, CHARGE)
    mz, y = [], []
    for b, pmz in enumerate(plan.pred_mz):
        h = height * np.exp(-0.5 * ((b - center_bin) / width) ** 2)
        if h < 1:
            continue
        x = pmz + shift + np.arange(-0.012, 0.0121, 0.002)
        mz.append(x)
        y.append(h * np.exp(-0.5 * ((x - pmz - shift) / 0.004) ** 2))
    return np.concatenate(mz), np.concatenate(y)


def time_course(n=6, start=0.5, step=0.5, ms2_every=0):
    """Scans whose deuteration grows with the scan time; optional ETD scans in between."""
    scans = []
    for k in range(n):
        mz, y = profile_envelope(1 + k, 1.0, 1e5, shift=1e-4 * k)
        scans.append({"mz": mz, "intensity": y, "time": start + step * k,
                      "filter": "FTMS + p ESI Full ms"})
        if ms2_every and k % ms2_every == 0:
            fmz, fy = profile_envelope(1 + k, 1.0, 5e4)
            scans.append({"mz": fmz, "intensity": fy, "time": start + step * k + 0.01,
                          "ms_level": 2, "filter": "FTMS + p ESI Full ms2 1012.0@etd20.00",
                          "precursor": (1012.0, 20, "MS:1000598",
                                        "electron transfer dissociation")})
    return scans


# --- the file ---------------------------------------------------------------------------------

def test_scan_table(tmp_path):
    scans = time_course(n=4, ms2_every=2)
    scans[0]["unit"] = "s"
    scans[0]["time"] = 30.0
    f = MzmlFile(write_mzml(tmp_path / "a.mzML", scans))
    t = f.scans()
    assert len(t) == 6 and list(t.ms_level) == [1, 2, 1, 1, 2, 1]
    assert t.time[0] == 30.0 and t.time[1] == pytest.approx((0.5 + 0.01) * 60)  # s and minutes
    assert t.precursor_charge[1] == 20 and t.precursor_mz[1] == 1012.0
    assert np.isnan(t.precursor_mz[0]) and t.precursor_charge[0] == 0
    assert (t.isolation_lower[1], t.isolation_upper[1]) == (1009.5, 1013.5)
    assert np.isnan(t.isolation_lower[0]) and np.isnan(t.isolation_upper[0])  # MS1: no window
    assert t.activation[1] == "electron transfer dissociation"  # collision energy is not one
    assert list(t.centroid) == [0] * 6 and "ms2" in t.filter[1]
    assert list(t.select(ms_level=2)) == [1, 4]
    assert list(t.select(activation="ETD")) == [1, 4] and list(t.select(activation="HCD")) == []
    assert list(t.select(activation="transfer")) == [1, 4]  # free text: part of a name
    assert list(t.select(precursor_mz=1012.4, precursor_tol=0.5, precursor_charge=20)) == [1, 4]
    assert list(t.select(precursor_mz=1013.0, precursor_tol=0.5)) == []
    assert list(t.select(ms_level=1, time_range=(0, 60))) == [0, 2]
    assert list(t.select(filter_contains="@etd")) == [1, 4]


def test_filter_string_as_a_user_param(tmp_path):
    """OpenMS writes the filter string as a userParam of the spectrum."""
    path = write_mzml(tmp_path / "o.mzML", time_course(n=1))
    cv = ('<cvParam cvRef="MS" accession="MS:1000512" name="filter string" '
          'value="FTMS + p ESI Full ms"/>')
    text = path.read_text().replace(cv, "")
    text = text.replace("<scanList", '<userParam name="filter string" type="xsd:string" '
                        'value="FTMS + p ESI Full ms"/><scanList', 1)
    path.write_text(text)
    assert MzmlFile(path).scans().filter[0] == "FTMS + p ESI Full ms"


@pytest.mark.parametrize("float64", [True, False])
@pytest.mark.parametrize("compress", [True, False])
def test_arrays_in_every_supported_encoding(tmp_path, float64, compress):
    scans = time_course(n=2)
    path = write_mzml(tmp_path / "e.mzML", scans, float64=float64, compress=compress)
    got = list(iter_scans(path))
    for s, want in zip(got, scans, strict=True):
        tol = 0 if float64 else 1e-6
        np.testing.assert_allclose(s.mz, want["mz"], rtol=tol)
        np.testing.assert_allclose(s.intensity, want["intensity"], rtol=tol)
        assert s.mz.dtype == np.float64


def test_unsupported_and_broken_files(tmp_path):
    with pytest.raises(MzmlError, match="numpress"):
        list(iter_scans(write_mzml(tmp_path / "n.mzML", time_course(n=1), numpress=True)))
    bad = tmp_path / "bad.mzML"
    bad.write_text("<mzML><run><spectrumList>")
    with pytest.raises(MzmlError, match="not a readable mzML"):
        list(iter_scans(bad))
    with pytest.raises(MzmlError, match="no such file"):
        MzmlFile(tmp_path / "missing.mzML")


@pytest.mark.parametrize("indexed", [True, False])
def test_single_scans_by_index(tmp_path, indexed):
    scans = time_course(n=5, ms2_every=2)
    f = MzmlFile(write_mzml(tmp_path / "i.mzML", scans, indexed=indexed))
    assert bool(f._index_offsets()) == indexed
    every = list(iter_scans(f.path))
    for i in (0, 3, len(scans) - 1):
        one = f.scan(i)
        assert one.index == i and one.time == every[i].time
        np.testing.assert_array_equal(one.mz, every[i].mz)
    lo, hi = every[0].mz[3], every[0].mz[10]
    cut = f.scan(0, mz_window=(lo, hi))
    assert cut.mz.min() >= lo and cut.mz.max() <= hi and len(cut.mz) == 8
    with pytest.raises(MzmlError, match="no scan"):
        MzmlFile(write_mzml(tmp_path / "j.mzML", scans)).scan(99)


# --- summing ----------------------------------------------------------------------------------

def test_profile_sum_keeps_gaps_empty():
    a = profile_envelope(2, 1.0, 1e4)
    b = profile_envelope(2, 1.0, 1e4, shift=0.001)  # the same peaks, sampled half a step off
    mz, y = sum_profile([a, b])
    # one grid with the scans' spacing, not the union of points: the shifted scan only adds a few
    # points at the end of each peak, where the first scan has no data
    n_peaks = len(a[0]) // 13
    assert len(a[0]) < len(mz) <= len(a[0]) + n_peaks < len(np.union1d(a[0], b[0]))
    assert y.max() == pytest.approx(2 * a[1].max(), rel=0.05)
    plan = plan_bins(PEPTIDE, CHARGE)
    for pmz in plan.pred_mz:  # every peak of the sum is about twice the peak of one scan
        near_a, near = np.abs(a[0] - pmz) < 0.01, np.abs(mz - pmz) < 0.01
        if near_a.any() and a[1][near_a].max() > 100:
            assert y[near].max() == pytest.approx(2 * a[1][near_a].max(), rel=0.05)
    # between two peaks (a stretch neither scan wrote) the sum stays zero
    middle = (plan.pred_mz[1] + plan.pred_mz[2]) / 2
    grid = np.linspace(middle - 0.1, middle + 0.1, 50)
    from hxex.io.mzml import _resample
    assert np.all(_resample(a[0], a[1], grid, 3.0) == 0)
    np.testing.assert_array_equal(sum_profile([a])[1], a[1])
    assert sum_profile([])[0].size == 0
    # where the scan with most points has no data, the other scan's points join the grid
    c = (a[0][: len(a[0]) // 2], a[1][: len(a[0]) // 2])
    d = (a[0] + 0.001, a[1])
    mz2, _ = sum_profile([d, c])
    assert len(d[0]) <= len(mz2) <= len(d[0]) + n_peaks  # c lies inside d's data (but its ends)
    outside = (a[0][-20:] + 5, a[1][-20:])
    assert sum_profile([c, outside])[0].size == len(c[0]) + 20  # beyond c's range: all added


def test_centroid_sum_merges_within_ppm():
    a = (np.array([500.0, 600.0]), np.array([10.0, 20.0]))
    b = (np.array([500.002, 600.5]), np.array([30.0, 5.0]))  # 4 ppm away, and a new peak
    mz, y = sum_centroid([a, b], ppm=10)
    np.testing.assert_allclose(mz, [500.0015, 600.0, 600.5])
    np.testing.assert_allclose(y, [40.0, 20.0, 5.0])
    assert len(sum_centroid([a, b], ppm=1)[0]) == 4


# --- spectra from runs ------------------------------------------------------------------------

def test_scan_time_layout_each_every_and_windows(tmp_path):
    path = write_mzml(tmp_path / "c.mzML", time_course(n=6, ms2_every=2))
    target = MzmlTarget(ENTRY)
    runs = [MzmlRun(path, "WT", rep=1, time_offset=5.0)]
    each = read_mzml(runs, [target], project="p")
    assert len(each) == 6 and list(each.time) == [5.0 + 60 * (0.5 + 0.5 * k) for k in range(6)]
    assert set(each.sample) == {"WT"} and set(each.rep) == {1} and set(each.data_id) == {0}
    assert len(set(each.spectrum_id)) == 6 and each.spectrum_id[0].startswith("p:WT/c.mzML/")
    plan = plan_bins(PEPTIDE, CHARGE)
    mz0, _ = each.spectrum(0)
    assert mz0.min() >= plan.pred_mz.min() - 2 and mz0.max() <= plan.pred_mz.max() + 2
    every = read_mzml(runs, [target], project="p", config=MzmlConfig(group="every", every=4))
    assert len(every) == 2
    assert every.time[0] == pytest.approx(5.0 + 60 * np.mean([0.5, 1, 1.5, 2]))
    # the file positions of the first and the last summed MS1 scan (ETD scans are in between)
    assert every.spectrum_id[0].endswith("scans0-5")
    windows = read_mzml(runs, [target], project="p",
                        config=MzmlConfig(group="windows", windows=[(0, 70), (80, 200)]))
    assert len(windows) == 2


def test_kinetics_spectra_from_the_read_record(tmp_path):
    from hxex.io import kinetics_spectra

    kin = write_mzml(tmp_path / "kin.mzML", time_course(n=4, ms2_every=2))
    un = write_mzml(tmp_path / "un.mzML", time_course(n=2, ms2_every=2))
    runs = [MzmlRun(un, "WT", time=0.0), MzmlRun(kin, "WT")]
    s = read_mzml(runs, [MzmlTarget(ENTRY)], project="p")
    flags = kinetics_spectra(s.source, s.spectrum_id)
    assert flags.tolist() == [t != 0.0 for t in s.time] and flags.sum() == 4
    assert not kinetics_spectra(None, s.spectrum_id).any()
    assert not kinetics_spectra({"reader": "specexport"}, s.spectrum_id).any()


def test_fragment_targets_need_their_scans(tmp_path, caplog):
    path = write_mzml(tmp_path / "f.mzML", time_course(n=6, ms2_every=2))
    frag = MzmlTarget(ENTRY, ms_level=2, activation="ETD", precursor_mz=1012.0,
                      precursor_charge=20)
    s = read_mzml([MzmlRun(path, "WT")], [frag, MzmlTarget(ENTRY)], project="p")
    assert sorted(set(s.data_id)) == [0, 1]
    assert (s.data_id == 0).sum() == 3 and (s.data_id == 1).sum() == 6
    # the precursor and activation stay with every fragment spectrum and series
    assert all("@ETD-p1012-pz20/scan" in i for i in s.spectrum_id[s.data_id == 0])
    assert not any("@" in i for i in s.spectrum_id[s.data_id == 1])
    from hxex.io import load_spectra, mzml_series_targets, save_spectra
    saved = load_spectra(save_spectra(s, tmp_path / "f.parquet"))
    assert mzml_series_targets(saved.source) == {0: frag, 1: MzmlTarget(ENTRY)}
    hcd = MzmlTarget(ENTRY, ms_level=2, activation="HCD")
    with caplog.at_level(logging.WARNING, logger="hxex.io.mzml"):
        s2 = read_mzml([MzmlRun(path, "WT")], [hcd, MzmlTarget(ENTRY)], project="p")
    assert "no scan for target" in caplog.text and len(s2) == 6
    with pytest.raises(MzmlError, match="no spectra"):
        read_mzml([MzmlRun(path, "WT")], [hcd], project="p")


def test_fixed_time_layout_sums_the_retention_window(tmp_path):
    lc = time_course(n=6)  # an LC-MS file: the scans are an elution profile, not a time course
    runs = [MzmlRun(write_mzml(tmp_path / "t60.mzML", lc), "WT", rep=1, time=60.0),
            MzmlRun(write_mzml(tmp_path / "t60b.mzML", lc), "WT", rep=2, time=60.0),
            MzmlRun(write_mzml(tmp_path / "un.mzML", lc), "WT", rep=1, time=0.0)]
    target = MzmlTarget(ENTRY, rt_window=(60.0, 120.0))  # scans at 60, 90 and 120 s
    s = read_mzml(runs, [target], project="p")
    assert len(s) == 3 and list(s.time) == [60.0, 60.0, 0.0] and list(s.rep) == [1, 2, 1]
    assert set(s.data_id) == {0}  # one series: the same sample and target
    assert s.spectrum_id[0].endswith("scans1-3")
    one = read_mzml(runs[:1], [MzmlTarget(ENTRY, rt_window=(60.0, 60.0))], project="p")
    whole = read_mzml(runs[:1], [MzmlTarget(ENTRY)], project="p")  # no window: every scan
    assert one.spectrum(0)[1].max() < s.spectrum(0)[1].max() < whole.spectrum(0)[1].max()


def test_config_and_arguments_are_checked(tmp_path):
    for bad in ({"group": "some"}, {"every": 0}, {"group": "windows"}, {"windows": [(5, 1)]},
                {"gap_factor": 1.0}, {"sum_ppm": 0}):
        with pytest.raises(ValueError):
            MzmlConfig(**bad)
    path = write_mzml(tmp_path / "x.mzML", time_course(n=1))
    with pytest.raises(ValueError, match="at least one"):
        read_mzml([], [MzmlTarget(ENTRY)], project="p")
    with pytest.raises(ValueError):
        read_mzml([MzmlRun(path, "WT")], [MzmlTarget(ENTRY)], project="bad:name")


def test_mzml_spectra_run_end_to_end(tmp_path):
    """Read, pick and fit: the fitted deuteration must grow along the scan time course."""
    from hxex.fit import FitConfig
    from hxex.model import Backend
    from hxex.noise import NoiseConfig
    from hxex.picking import PickerConfig
    from hxex.run import RunConfig, run

    path = write_mzml(tmp_path / "e2e.mzML", time_course(n=6))
    spectra = read_mzml([MzmlRun(path, "WT")], [MzmlTarget(ENTRY)], project="p")
    cfg = RunConfig(picker=PickerConfig(zero_filling=3, noise=100.0),
                    fit=FitConfig(k_min=1, k_max=1, adam_steps=150, n_restarts=2,
                                  backend=Backend("cpu", "float64"),
                                  noise=NoiseConfig("gaussian", percent=3.0)),
                    quality=None)
    res = run(spectra, cfg, out=tmp_path / "b", keep_spectra=True)
    means = res.fit.fits[1].means[:, 0]
    assert len(means) == 6 and np.all(np.diff(means) > 0.5)
    # the spectra read from mzML survive a spectra file and a bundle, and run again the same way
    from hxex.io import load_spectra, open_bundle, save_spectra
    saved = load_spectra(save_spectra(spectra, tmp_path / "s.parquet"))
    with open_bundle(tmp_path / "b") as b:
        stored = b.spectra()
    for again in (saved, stored):
        assert list(again.spectrum_id) == list(spectra.spectrum_id)
        np.testing.assert_array_equal(again.mz, spectra.mz)
        np.testing.assert_array_equal(again.intensity, spectra.intensity)
        np.testing.assert_array_equal(again.time, spectra.time)
        np.testing.assert_allclose(run(again, cfg).fit.fits[1].means[:, 0], means, atol=1e-6)
        assert again.source == spectra.source


def test_the_read_is_recorded_and_repeated(tmp_path):
    """The runs, targets and settings of a read go with the spectra (and into spectra files and
    bundles, stored spectra or not), and read_mzml_source repeats the read from them."""
    from hxex.fit import FitConfig
    from hxex.io import load_spectra, open_bundle, read_mzml_source, save_spectra
    from hxex.model import Backend
    from hxex.noise import NoiseConfig
    from hxex.picking import PickerConfig
    from hxex.run import RunConfig, run

    path = write_mzml(tmp_path / "a.mzML", time_course(n=6))
    picker = PickerConfig(zero_filling=3, noise=100.0)
    target = MzmlTarget(ENTRY, ms_level=1, filter_excludes="sid", rt_window=None)
    config = MzmlConfig(group="every", every=2, mz_margin=1.5)
    runs = [MzmlRun(path, "WT", rep=1, time_offset=30.0)]
    spectra = read_mzml(runs, [target], project="p", config=config, picker=picker)
    src = spectra.source
    assert src["reader"] == "mzml" and src["project"] == "p"
    assert MzmlTarget.from_dict(src["targets"][0]) == target
    assert MzmlConfig.from_dict(src["config"]) == config
    assert MzmlRun.from_dict(src["runs"][0]) == MzmlRun(str(path), "WT", time_offset=30.0)
    assert PickerConfig.from_dict(src["picker"]) == picker
    assert spectra.select(spectra.time > spectra.time.min()).source == src

    def same(a, b):
        assert list(a.spectrum_id) == list(b.spectrum_id) and a.source == b.source
        np.testing.assert_array_equal(a.mz, b.mz)
        np.testing.assert_array_equal(a.intensity, b.intensity)
        np.testing.assert_array_equal(a.time, b.time)

    same(read_mzml_source(src), spectra)
    same(read_mzml_source(load_spectra(save_spectra(spectra, tmp_path / "s.parquet")).source),
         spectra)
    moved = tmp_path / "moved"
    moved.mkdir()
    path.rename(moved / path.name)
    again = read_mzml_source(src, directory=moved)
    np.testing.assert_array_equal(again.intensity, spectra.intensity)

    cfg = RunConfig(picker=picker, quality=None,
                    fit=FitConfig(k_min=1, k_max=1, adam_steps=50, n_restarts=1,
                                  backend=Backend("cpu", "float64"),
                                  noise=NoiseConfig("gaussian", percent=3.0)))
    res = run(again, cfg, out=tmp_path / "b", keep_spectra=False)
    assert res.spectra_source == again.source
    with open_bundle(tmp_path / "b") as b:
        assert b.spectra() is None and b.spectra_source() == again.source
        same(read_mzml_source(b.spectra_source(), directory=moved), again)
    with pytest.raises(ValueError, match="not an mzML read"):
        read_mzml_source({"reader": "specexport"})
    with pytest.raises(ValueError, match="unknown"):
        MzmlConfig.from_dict({"bogus": 1})
    with pytest.raises(ValueError, match="unknown"):
        MzmlTarget.from_dict({**target.to_dict(), "bogus": 1})
    rt = MzmlTarget(ENTRY, rt_window=(60.0, 120.0), activation="ETD", precursor_charge=3)
    assert MzmlTarget.from_dict(json.loads(json.dumps(rt.to_dict()))) == rt
    w = MzmlConfig(group="windows", windows=((0, 10), (10, 30)), max_spectra=5)
    assert MzmlConfig.from_dict(json.loads(json.dumps(w.to_dict()))) == w


# --- a real file --------------------------------------------------------------------------------

REAL = os.environ.get("HXEX_MZML_EXAMPLE")


@pytest.mark.skipif(not REAL, reason="HXEX_MZML_EXAMPLE not set")
def test_a_real_thermo_file():
    f = MzmlFile(REAL)
    t = f.scans()
    assert len(t) > 0 and set(t.ms_level) <= {1, 2} and np.all(np.diff(t.time) >= 0)
    if (t.ms_level == 2).any():
        assert any("dissociation" in a for a in t.activation[t.ms_level == 2])
    one = f.scan(len(t) // 2)
    assert one.index == len(t) // 2 and len(one.mz) == len(one.intensity) > 0
    assert np.all(np.diff(one.mz) > 0)


def test_filter_exclusion_earliest_time_and_log_sampling(tmp_path):
    scans = time_course(n=12, start=0.1, step=0.25, ms2_every=1)
    for k, s in enumerate(scans):
        if s.get("ms_level") == 2 and k % 4 == 1:
            s["filter"] = s["filter"] + " sid=35.00"  # a second kind of ETD scan
    path = write_mzml(tmp_path / "k.mzML", scans)
    t = MzmlFile(path).scans()
    with_sid = t.select(filter_contains="sid")
    without = t.select(ms_level=2, filter_excludes="sid")
    assert len(with_sid) and len(without) and not set(with_sid) & set(without)
    assert len(with_sid) + len(without) == int((t.ms_level == 2).sum())
    frag = MzmlTarget(ENTRY, ms_level=2, activation="ETD", filter_excludes="sid")
    s = read_mzml([MzmlRun(path, "WT")], [frag], project="p")
    assert len(s) == len(without)
    # the earliest time: scans before it are left out
    late = read_mzml([MzmlRun(path, "WT")], [MzmlTarget(ENTRY)], project="p",
                     config=MzmlConfig(min_time=60.0))
    assert late.time.min() >= 60.0 and len(late) < 12
    # sampling: at most N spectra, the first and the last kept, denser early than late
    every = read_mzml([MzmlRun(path, "WT")], [MzmlTarget(ENTRY)], project="p")
    few = read_mzml([MzmlRun(path, "WT")], [MzmlTarget(ENTRY)], project="p",
                    config=MzmlConfig(max_spectra=5))
    assert 2 <= len(few) <= 5 and few.time[0] == every.time[0] and few.time[-1] == every.time[-1]
    gaps = np.diff(few.time)
    assert gaps[0] <= gaps[-1]
    with pytest.raises(ValueError, match="max_spectra"):
        MzmlConfig(max_spectra=1)
    # a run with a fixed time is not sampled (it gives one sum per target)
    fixed = read_mzml([MzmlRun(path, "WT", time=60.0)], [MzmlTarget(ENTRY)], project="p",
                      config=MzmlConfig(max_spectra=2))
    assert len(fixed) == 1


def test_activation_short_names_are_exact(tmp_path):
    """Thermo writes EThcD as ETD plus supplemental beam-type CID: 'ETD' must not select it."""
    etd = ("MS:1000598", "electron transfer dissociation")
    scans = time_course(n=2)
    mz, y = profile_envelope(2, 1.0, 1e4)
    scans += [{"mz": mz, "intensity": y, "time": 2.0, "ms_level": 2, "filter": "etd",
               "precursor": (777.4, 2, *etd)},
              {"mz": mz, "intensity": y, "time": 2.1, "ms_level": 2, "filter": "ethcd",
               "precursor": (777.4, 2, *etd)},
              {"mz": mz, "intensity": y, "time": 2.2, "ms_level": 2, "filter": "hcd",
               "precursor": (777.4, 2, "MS:1000422", "beam-type collision-induced dissociation")}]
    path = write_mzml(tmp_path / "a.mzML", scans)
    text = path.read_text()
    # give the second ETD scan the supplemental HCD that Thermo writes for EThcD
    extra = ('<cvParam cvRef="MS" accession="MS:1002678" '
             'name="supplemental beam-type collision-induced dissociation" value=""/>')
    first = text.index('name="electron transfer dissociation"', text.index('id="scan=4"'))
    end = text.index("/>", first) + 2
    path.write_text(text[:end] + extra + text[end:])
    t = MzmlFile(path).scans()
    assert list(t.select(activation="ETD")) == [2]
    assert list(t.select(activation="EThcD")) == [3]
    assert list(t.select(activation="HCD")) == [4]
    assert list(t.select(activation="dissociation")) == [2, 3, 4]


def test_files_are_identified_and_paths_recorded_as_asked(tmp_path, caplog):
    """Every file's size and checksum go with the spectra; a repeated read finds the files
    (by recorded path, relative to a data root, or by name in a directory) and flags a file
    that changed (data updated between runs)."""
    from hxex.io import read_mzml_source

    data = tmp_path / "data"
    data.mkdir()
    plain = write_mzml(data / "plain.mzML", time_course(n=4))
    indexed = write_mzml(data / "indexed.mzML", time_course(n=4), indexed=True)
    ident = file_identity(plain)
    assert ident == {"name": "plain.mzML", "size": plain.stat().st_size, "algorithm": "sha256",
                     "checksum": hashlib.sha256(plain.read_bytes()).hexdigest()}
    own = file_identity(indexed)  # the converter's checksum, read from the end of the file
    assert own["algorithm"] == "mzml-sha1" and len(own["checksum"]) == 40
    raw = indexed.read_bytes()
    assert own["checksum"] == hashlib.sha1(raw[: raw.index(b"<fileChecksum>") + 14]).hexdigest()

    runs = [MzmlRun("plain.mzML", "WT"), MzmlRun("indexed.mzML", "WT", rep=2)]
    s = read_mzml(runs, [MzmlTarget(ENTRY)], project="p", data_root=data,
                  record_paths="relative")
    assert [r["path"] for r in s.source["runs"]] == ["plain.mzML", "indexed.mzML"]
    assert s.source["files"] == [ident, own] and s.source["record_paths"] == "relative"
    assert read_mzml_source(s.source, data_root=data).source == s.source
    with pytest.raises(MzmlError, match="not found"):
        read_mzml_source(s.source)  # relative paths need the data root

    named = read_mzml([MzmlRun(plain, "WT")], [MzmlTarget(ENTRY)], project="p",
                      record_paths="name")
    assert named.source["runs"][0]["path"] == "plain.mzML"
    assert str(tmp_path) not in json.dumps(named.source)  # nothing about where the data lives
    np.testing.assert_array_equal(read_mzml_source(named.source, directory=data).intensity,
                                  named.intensity)

    full = read_mzml([MzmlRun(plain, "WT")], [MzmlTarget(ENTRY)], project="p")
    assert full.source["runs"][0]["path"] == str(plain)
    plain.write_bytes(plain.read_bytes().replace(b"</mzML>", b"</mzML>\n"))  # updated data
    with caplog.at_level(logging.WARNING, logger="hxex.io.mzml"):
        again = read_mzml_source(full.source)
    assert "has changed since it was read" in caplog.text
    assert again.source["files"] == [{**file_identity(plain), "previous": ident}]
    with pytest.raises(MzmlError, match="has changed"):
        read_mzml_source(full.source, on_change="error")
    with pytest.raises(ValueError, match="on_change"):
        read_mzml_source(full.source, on_change="ignore")
    with pytest.raises(ValueError, match="needs a data_root"):
        read_mzml(runs, [MzmlTarget(ENTRY)], project="p", record_paths="relative")
    with pytest.raises(ValueError, match="not inside the data root"):
        read_mzml([MzmlRun(plain, "WT")], [MzmlTarget(ENTRY)], project="p",
                  data_root=tmp_path / "elsewhere", record_paths="relative")
    with pytest.raises(ValueError, match="record_paths"):
        read_mzml(runs, [MzmlTarget(ENTRY)], project="p", record_paths="absolute")


def test_a_run_can_replace_the_targets_scan_conditions(tmp_path):
    """A control file acquired with sid=1.00 throughout and a wrong precursor charge, read with
    kinetics files that mark in-source activation with sid=: the run sets its own conditions."""
    etd = ("MS:1000598", "electron transfer dissociation")
    mz, y = profile_envelope(2, 1.0, 1e4)

    def ms2(t, filt, charge):
        return {"mz": mz, "intensity": y, "time": t, "ms_level": 2, "filter": filt,
                "precursor": (777.4, charge, *etd)}

    un = write_mzml(tmp_path / "un.mzML", [ms2(0.1 * k, "sid=1.00 etd", 3) for k in range(3)])
    kin = write_mzml(tmp_path / "kin.mzML", [ms2(0.5 + 0.1 * k, "sid=20.00 etd" if k % 2 else
                                                 "etd", 2) for k in range(6)])
    target = MzmlTarget(ENTRY, ms_level=2, activation="ETD", precursor_charge=2,
                        filter_excludes="sid")
    with pytest.raises(MzmlError, match="no spectra"):  # the target's conditions miss the control
        read_mzml([MzmlRun(un, "S", time=0.0)], [target], project="p")
    control = MzmlRun(un, "S", time=0.0, filter_excludes=None, precursor_charge=None)
    s = read_mzml([control, MzmlRun(kin, "S")], [target], project="p")
    assert list(s.time[:1]) == [0.0] and s.spectrum_id[0].endswith("scans0-2")  # all 3 summed
    assert len(s) == 1 + 3  # the kinetics file keeps the target's conditions: no sid scans
    assert control.conditions(target) == {
        "activation": "ETD", "precursor_mz": None, "precursor_charge": None,
        "filter_contains": None, "filter_excludes": None}
    # recorded with the read, and only the conditions that were set
    assert s.source["runs"][0] == {**MzmlRun(str(un), "S", time=0.0).to_dict(),
                                   "filter_excludes": None, "precursor_charge": None}
    assert "filter_excludes" not in s.source["runs"][1]
    assert MzmlRun.from_dict(s.source["runs"][0]) == dataclasses.replace(control, path=str(un))
    from hxex.io import read_mzml_source
    np.testing.assert_array_equal(read_mzml_source(s.source).intensity, s.intensity)
    with pytest.raises(ValueError, match="precursor_charge"):
        MzmlRun(un, "S", precursor_charge="any")


def test_result_tables_name_the_fragment_and_precursor(tmp_path):
    """Results of an mzML read carry how each measurement's scans were chosen."""
    from hxex.fit import FitConfig
    from hxex.io import open_bundle
    from hxex.model import Backend
    from hxex.noise import NoiseConfig
    from hxex.picking import PickerConfig
    from hxex.report import Results, spectra_table
    from hxex.run import RunConfig, run

    path = write_mzml(tmp_path / "f.mzML", time_course(n=6, ms2_every=2))
    frag = MzmlTarget(dataclasses.replace(ENTRY, extra={"fragment": "c:10"}), ms_level=2,
                      activation="ETD", precursor_mz=1012.0, precursor_charge=20)
    spectra = read_mzml([MzmlRun(path, "WT")], [frag, MzmlTarget(ENTRY)], project="p")
    cfg = RunConfig(picker=PickerConfig(zero_filling=3, noise=100.0), quality=None,
                    fit=FitConfig(k_min=1, k_max=1, adam_steps=50, n_restarts=1,
                                  backend=Backend("cpu", "float64"),
                                  noise=NoiseConfig("gaussian", percent=3.0)))
    res = run(spectra, cfg, out=tmp_path / "b", keep_spectra=False)
    with open_bundle(tmp_path / "b") as b:
        from_bundle = Results.from_bundle(b)
    for results in (Results.from_run(res), from_bundle):
        t = spectra_table(results)
        ms2 = t["ms_level"] == 2
        assert ms2.sum() == 3 and (~ms2).sum() == 6
        assert set(t["fragment"][ms2]) == {"c:10"} and set(t["fragment"][~ms2]) == {""}
        assert set(t["activation"][ms2]) == {"ETD"} and set(t["precursor_charge"][ms2]) == {20}
        assert set(t["precursor_mz"][ms2]) == {1012.0} and np.isnan(t["precursor_mz"][~ms2]).all()


ETD = ("MS:1000598", "electron transfer dissociation")
HCD = ("MS:1000422", "beam-type collision-induced dissociation")


def test_msn_stages_parent_and_preactivation(tmp_path):
    """Every stage of an MS3 scan: from several precursors, or from the filter string when only
    the last stage is written (as ThermoRawFileParser does)."""
    def scan(t, level, precursors, filt=""):
        return {"mz": [100.0, 101.0], "intensity": [1.0, 2.0], "time": t, "ms_level": level,
                "precursors": precursors, "filter": filt}

    path = write_mzml(tmp_path / "ms3.mzML", [
        scan(0.1, 2, [(500.0, 2, *ETD)], "FTMS + p ESI Full ms2 500.0000@etd25.00 [100-900]"),
        scan(0.2, 3, [(500.0, 2, *ETD)],
             "FTMS + p ESI Full ms3 500.0100@hcd15.00 500.0000@etd25.00 [100-900]"),
        scan(0.3, 3, [(600.0, 2, "MS:1000598", "electron transfer dissociation")],
             "FTMS + p ESI sa Full ms3 600.0100@hcd20.00 600.0000@etd50.00@hcd25.00 [100-900]"),
        scan(0.4, 3, [(700.0, 2, *HCD), (650.0, 2, *ETD)]),  # written stage by stage
        scan(0.5, 3, [(500.0, 2, *ETD)]),  # earlier stage not stated
        {"mz": [100.0], "intensity": [1.0], "time": 0.6},
    ])
    t = MzmlFile(path).scans()
    assert list(t.parent_mz[:4]) == [500.0, 500.01, 600.01, 700.0]
    assert np.isnan(t.parent_mz[4:]).all() and list(t.preactivation[4:]) == ["?", ""]
    assert list(t.preactivation[:4]) == ["", HCD[1], HCD[1], HCD[1]]
    assert list(t.preactivation_energy[1:4]) == [15.0, 20.0, 20.0]
    assert list(t.energy[:2]) == [20.0, 20.0]  # the last stage's, as the precursor states it
    # the parent window: written for MS2 and for stage-by-stage MS3, not in a filter string
    assert (t.parent_lower[0], t.parent_upper[0]) == (497.5, 501.5)
    assert np.isnan(t.parent_lower[1]) and t.parent_lower[3] == 697.5
    assert list(t.select(ms_level=3, preactivation="HCD")) == [1, 2, 3]
    assert list(t.select(preactivation="none", ms_level=2)) == [0]
    assert list(t.select(preactivation="collision")) == [1, 2, 3]  # free text, as activation
    assert list(t.select(parent_mz=(499.5, 500.2))) == [0, 1]  # overlap, else within tolerance
    assert list(t.select(parent_mz=(699.0, 701.0))) == [3]
    assert list(t.select(parent_mz=(640.0, 660.0))) == []  # the last stage is not the parent
    assert MzmlFile(path).scan(3).stages[0].mz == 700.0


def test_filter_stages():
    from hxex.io.mzml import filter_stages

    st = filter_stages("FTMS + p ESI sa Full ms3 468.9100@hcd20.00 468.9000@etd50.00@hcd25.00 "
                       "[129.0000-948.0000]")
    assert [s.mz for s in st] == [468.91, 468.9] and [s.energy for s in st] == [20.0, 50.0]
    assert st[0].activation == (HCD[1],)
    assert st[1].activation == (ETD[1], "supplemental " + HCD[1])
    assert filter_stages("FTMS + p ESI Full ms [100.0000-2000.0000]") == ()
    assert len(filter_stages("ITMS + c ESI d Full ms2 445.12@cid35.00 [110.00-900.00]")) == 1


def test_fragments_take_the_scans_of_their_own_parent(tmp_path, caplog):
    """A bottom-up file with several precursors of the same charge: each fragment target selects
    the scans whose parent isolation overlaps its own parent's m/z range."""
    from hxex.fragments import fragment_targets, parent_mz_range
    from hxex.io.mzml import target_scans

    lo, hi = parent_mz_range(ENTRY)
    other = hi + 30.0

    def ms(t, level, pmz, filt=""):
        prec = [(pmz, CHARGE, *ETD)]
        return {"mz": [100.0], "intensity": [1.0], "time": t, "ms_level": level,
                "precursors": prec, "filter": filt}

    path = write_mzml(tmp_path / "bu.mzML", [
        ms(0.1, 2, lo + 0.5), ms(0.2, 2, other),
        ms(0.3, 3, lo + 0.5, f"FTMS + p ESI Full ms3 {lo + 0.51:.4f}@hcd10.00 "
                             f"{lo + 0.5:.4f}@etd25.00 [100-900]"),
        ms(0.4, 3, other, f"FTMS + p ESI Full ms3 {other:.4f}@hcd10.00 {other:.4f}@etd25.00 "
                          "[100-900]"),
        ms(0.5, 3, lo + 0.5),  # earlier stage not stated
    ])
    table, run, config = MzmlFile(path).scans(), MzmlRun(path, "S"), MzmlConfig()
    ms2 = fragment_targets(ENTRY, "c:3:1", activation="ETD")[0]
    assert ms2.parent_mz == (lo, hi)
    assert list(target_scans(table, run, ms2, config)) == [0]
    anyp = fragment_targets(ENTRY, "c:3:1", activation="ETD", parent_mz=None)[0]
    assert list(target_scans(table, run, anyp, config)) == [0, 1]  # any precursor of charge 2
    ms3 = fragment_targets(ENTRY, "c:3:1", ms_level=3, preactivation="HCD")[0]
    with caplog.at_level(logging.WARNING):
        assert list(target_scans(table, run, ms3, config)) == [2]
    assert "1 MS3 scans do not state their earlier stages" in caplog.text
    from hxex.io.mzml import _target_label

    assert _target_label(ms3).endswith("-ms3-pre:HCD-pz2") and "ms" not in _target_label(ms2)
    assert MzmlTarget.from_dict(ms3.to_dict()) == ms3
    assert "parent_mz" not in MzmlTarget(ENTRY).to_dict()  # older records stay as they were
    with pytest.raises(ValueError, match="parent_mz"):
        fragment_targets(ENTRY, "c:3:1", parent_mz="own")
    with pytest.raises(ValueError, match="range"):
        MzmlTarget(ENTRY, parent_mz=(5.0, 1.0))


def test_usable_from_is_on_the_acquisition_clock(tmp_path):
    """A run's usable_from leaves out its scans before that scan time; time_offset moves only
    the spectra's times (the exchange clock); MzmlConfig.min_time is a scan time too."""
    path = write_mzml(tmp_path / "k.mzML", time_course(n=6))  # scans at 30 to 180 s
    target = MzmlTarget(ENTRY)
    run = MzmlRun(path, "WT", time_offset=100.0, usable_from=75.0)
    got = read_mzml([run], [target], project="p")
    assert list(got.time) == [190.0, 220.0, 250.0, 280.0]  # scans 90-180 s, plus 100 s
    both = read_mzml([run], [target], project="p", config=MzmlConfig(min_time=125.0))
    assert list(both.time) == [250.0, 280.0]  # the later of the two starts, both scan times
    assert got.source["runs"][0]["usable_from"] == 75.0
    assert MzmlRun.from_dict(run.to_dict()) == MzmlRun(str(path), "WT", time_offset=100.0,
                                                       usable_from=75.0)
    assert "usable_from" not in MzmlRun(path, "WT").to_dict()  # older records stay as they were
    # a run with a fixed time: the start applies within the retention window as well
    fixed = MzmlRun(path, "WT", time=60.0, usable_from=100.0)
    summed = read_mzml([fixed], [MzmlTarget(ENTRY, rt_window=(50.0, 160.0))], project="p")
    assert len(summed) == 1 and summed.spectrum_id[0].endswith("scans3-4")  # 120 and 150 s


# --- audit fixes: arrays, ids, recorded names, charges, sampling ------------------------------

def test_empty_and_damaged_binary_arrays(tmp_path):
    scans = time_course(n=2)
    scans[1] = {"mz": [], "intensity": [], "time": 1.0}
    path = write_mzml(tmp_path / "e.mzML", scans)
    # an empty <binary></binary> with zlib declared (as some writers write an empty scan)
    empty = base64.b64encode(zlib.compress(b"")).decode()
    text = path.read_text()
    assert text.count(f"<binary>{empty}</binary>") == 2
    path.write_text(text.replace(f"<binary>{empty}</binary>", "<binary></binary>"))
    got = list(iter_scans(path))
    assert len(got) == 2 and len(got[1].mz) == 0 and len(got[1].intensity) == 0
    assert got[1].mz.dtype == np.float64
    # damaged payloads name the scan instead of escaping as zlib or base64 errors
    first = base64.b64encode(zlib.compress(np.asarray(scans[0]["mz"]).tobytes())).decode()
    assert text.count(first) == 1
    for bad in ("not base64!!", base64.b64encode(b"not zlib data").decode()):
        path.write_text(text.replace(first, bad))
        with pytest.raises(MzmlError, match=r"scan 0 \(scan=1\): an array cannot be decoded"):
            list(iter_scans(path))


def test_spectrum_ids_must_be_unique(tmp_path):
    """Two runs with one sample and file name (replicates in sibling directories) would give
    the same ids, as would time windows that select the same scans twice."""
    scans = time_course(n=3)
    (tmp_path / "rep1").mkdir()
    (tmp_path / "rep2").mkdir()
    a = write_mzml(tmp_path / "rep1" / "run.mzML", scans)
    b = write_mzml(tmp_path / "rep2" / "run.mzML", scans)
    with pytest.raises(MzmlError, match="not unique") as info:
        read_mzml([MzmlRun(a, "WT"), MzmlRun(b, "WT", rep=2)], [MzmlTarget(ENTRY)], project="p")
    message = str(info.value)
    assert "run 0 (WT, rep 1, " in message and "run 1 (WT, rep 2, " in message
    assert str(b) in message and "same sample and file name" in message
    ok = read_mzml([MzmlRun(a, "WT"), MzmlRun(b, "WT2")], [MzmlTarget(ENTRY)], project="p")
    assert len(ok) == 6 and len(set(ok.spectrum_id)) == 6
    twice = MzmlConfig(group="windows", windows=[(0, 1000), (0, 1000)])
    with pytest.raises(MzmlError, match="not unique.*run 0.*selected more than once"):
        read_mzml([MzmlRun(a, "WT")], [MzmlTarget(ENTRY)], project="p", config=twice)


def test_run_paths_recorded_on_windows_are_found_by_name(tmp_path):
    from hxex.io import kinetics_spectra, read_mzml_source

    kin = write_mzml(tmp_path / "kin.mzML", time_course(n=3))
    s = read_mzml([MzmlRun(kin, "WT")], [MzmlTarget(ENTRY)], project="p")
    src = json.loads(json.dumps(s.source))
    src["runs"][0]["path"] = r"C:\data\hdx\kin.mzML"  # the record of a read made on Windows
    assert kinetics_spectra(src, s.spectrum_id).all()
    again = read_mzml_source(src, directory=tmp_path)
    np.testing.assert_array_equal(again.intensity, s.intensity)
    assert again.source["runs"][0]["path"] == r"C:\data\hdx\kin.mzML"
    del src["files"]  # an older record without the file names: the name comes from the path
    assert kinetics_spectra(src, s.spectrum_id).all()
    assert len(read_mzml_source(src, directory=tmp_path)) == len(s)


def test_precursor_charge_is_that_of_its_own_stage(tmp_path):
    hcd = ("MS:1000422", "beam-type collision-induced dissociation")
    etd = ("MS:1000598", "electron transfer dissociation")
    scans = time_course(n=1)
    scans.append({"mz": scans[0]["mz"], "intensity": scans[0]["intensity"], "time": 0.6,
                  "ms_level": 3, "precursors": [(1012.0, 3, *hcd), (1012.0, 7, *etd)]})
    path = write_mzml(tmp_path / "ms3.mzML", scans)
    assert list(iter_scans(path))[1].precursor_charge == 7
    charge7 = '<cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="7"/>'
    text = path.read_text()
    assert text.count(charge7) == 1
    path.write_text(text.replace(charge7, ""))  # the last stage states no charge
    scan = list(iter_scans(path))[1]
    assert scan.precursor_charge == 0 and scan.precursor_mz == 1012.0 and len(scan.stages) == 2


def test_sampling_scans_that_share_one_time(tmp_path):
    from hxex.io.mzml import _sample_log

    picks = _sample_log(np.full(10, 5.0), 4)
    assert picks[0] == 0 and picks[-1] == 9 and len(picks) <= 4 and len(set(picks)) == len(picks)
    assert list(_sample_log(np.full(3, 5.0), 4)) == [0, 1, 2]
    path = write_mzml(tmp_path / "same.mzML", time_course(n=6, step=0.0))
    s = read_mzml([MzmlRun(path, "WT")], [MzmlTarget(ENTRY)], project="p",
                  config=MzmlConfig(max_spectra=3))
    assert 2 <= len(s) <= 3 and s.spectrum_id[0].endswith("scan0")
    assert s.spectrum_id[-1].endswith("scan5")
