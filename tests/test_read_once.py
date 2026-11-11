# SPDX-License-Identifier: GPL-3.0-or-later
"""A scan file listed by several samples (a pooled project's control files) is read once for
all of them: its scans are decoded once per pass (drift ranges, spectra, retention search),
however many samples list it, and the spectra and their record are those of a read that opens
and decodes the file again for every run."""

import collections
import dataclasses
import json

import numpy as np
import pytest

from hxex.io import MzmlConfig, MzmlFile, MzmlRun, MzmlTarget, read_mzml
from hxex.io import mzml as mzml_module
from test_mzml import ENTRY, _drifting, write_mzml
from test_retention_search import PEPTIDES, _entry, lc_run

PEPTIDE_DRIFT = {5.1: 0.02, 5.2: 0.4, 5.3: 1.0, 5.4: 0.5, 5.5: 0.05}
OTHER_DRIFT = {5.7: 1.0, 5.8: 0.6}


def _assert_identical(a, b):
    """Every field of two spectra sets equal, floats bit for bit, and the same record."""
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if f.name == "source":
            assert json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)
        elif f.name == "scans":  # the fragments' scan records
            assert (x is None) == (y is None)
            for u, v in zip(x if x is not None else [], y if y is not None else [], strict=True):
                assert (u is None) == (v is None)
                if u is not None:
                    for part in ("weights", "noise_mz", "noise"):
                        assert getattr(u, part).tobytes() == getattr(v, part).tobytes()
        elif x is None or y is None:
            assert x is None and y is None, f.name
        else:
            x, y = np.asarray(x), np.asarray(y)
            assert x.dtype == y.dtype and x.shape == y.shape, f.name
            if x.dtype.kind == "f":
                assert x.tobytes() == y.tobytes(), f.name
            else:
                np.testing.assert_array_equal(x, y, err_msg=f.name)


def _run_by_run(monkeypatch):
    """Read every run on its own, as before: no file shared between runs."""
    monkeypatch.setattr(mzml_module, "_same_file", lambda numbers, opened: [[n] for n in numbers])


def _counting(monkeypatch):
    """Count the files opened and the scans decoded (with their arrays), per file name."""
    opened, decoded = collections.Counter(), collections.Counter()
    init, read = MzmlFile.__init__, MzmlFile.read_scans

    def counted_init(self, path):
        init(self, path)
        opened[self.path.name] += 1

    def counted_read(self, indices, mz_windows=None, drift=False):
        out = read(self, indices, mz_windows, drift)
        decoded[self.path.name] += len(out)
        return out

    monkeypatch.setattr(MzmlFile, "__init__", counted_init)
    monkeypatch.setattr(MzmlFile, "read_scans", counted_read)
    return opened, decoded


def _pooled(tmp_path, n_samples):
    """Ion mobility files: an undeuterated and a fully deuterated control listed by every
    sample, and one labelled file per sample; one target per sample (a pooled project's
    peptide table has a row per sample), the second sample's with a narrower window."""
    lc = [{"time": 1.0 + 0.1 * k, "filter": "Full ms"} for k in range(4)]
    un = write_mzml(tmp_path / "un.mzML", _drifting(lc, [(1, 1e5, PEPTIDE_DRIFT),
                                                         (7, 5e4, OTHER_DRIFT)]), indexed=True)
    fd = write_mzml(tmp_path / "fd.mzML", _drifting(lc, [(8, 1e5, PEPTIDE_DRIFT),
                                                         (7, 5e4, OTHER_DRIFT)]))
    runs, targets = [], []
    for k in range(n_samples):
        sample = f"S{k}"
        lab = write_mzml(tmp_path / f"lab{k}.mzML", _drifting(lc, [
            (3 + k, 1e5, PEPTIDE_DRIFT), (7, 5e4, OTHER_DRIFT)]), indexed=True)
        runs += [MzmlRun(lab, sample, time=60.0), MzmlRun(fd, sample, role="fulldeut"),
                 MzmlRun(un, sample, role="undeut")]
        window = (60.0, 70.0) if k == 1 else (60.0, 80.0)
        targets.append(MzmlTarget(dataclasses.replace(ENTRY, sample=sample), rt_window=window))
    return runs, targets


def test_a_control_file_listed_by_several_samples_is_read_once(tmp_path, monkeypatch):
    runs, targets = _pooled(tmp_path, 3)
    with monkeypatch.context() as m:
        _run_by_run(m)
        before = read_mzml(runs, targets, project="p")
    with monkeypatch.context() as m:
        opened, decoded = _counting(m)
        shared = read_mzml(runs, targets, project="p")
    _assert_identical(shared, before)
    assert len(shared) == 9 and set(shared.sample) == {"S0", "S1", "S2"}
    assert shared.source["drift_gate"]["ranges"][0]["n_files"] == 1
    # as much work as when one sample lists the files
    (tmp_path / "one").mkdir()
    one_runs, one_targets = _pooled(tmp_path / "one", 1)
    with monkeypatch.context() as m:
        one_opened, one_decoded = _counting(m)
        read_mzml(one_runs, one_targets, project="p")
    for name in ("un.mzML", "fd.mzML"):
        assert opened[name] == one_opened[name], name
        assert decoded[name] == one_decoded[name], name
    # the undeuterated control: once for the drift ranges, once for the spectra
    assert decoded["un.mzML"] == 2 * 4
    with monkeypatch.context() as m:
        _run_by_run(m)
        _, before_decoded = _counting(m)
        read_mzml(runs, targets, project="p")
    assert before_decoded["un.mzML"] > decoded["un.mzML"]


@pytest.mark.parametrize("mode", ["suggest", "automatic"])
def test_the_retention_search_reads_a_shared_control_once(tmp_path, monkeypatch, mode):
    """Two samples list the same two undeuterated controls; the second reads one peptide
    fewer, so its search windows are not the first's: each control's scans are read once, over
    the union of the windows, and each run's verdicts are those of a read of its own."""
    un1 = write_mzml(tmp_path / "un1.mzML", lc_run(1))
    un2 = write_mzml(tmp_path / "un2.mzML", lc_run(2, shift=0.3))
    t60 = write_mzml(tmp_path / "t60.mzML", lc_run(3))
    t60b = write_mzml(tmp_path / "t60b.mzML", lc_run(4))
    runs = [MzmlRun(str(un1), "S", rep=1, role="undeut"),
            MzmlRun(str(un2), "S", rep=2, role="undeut"),
            MzmlRun(str(un1), "T", rep=1, role="undeut"),
            MzmlRun(str(un2), "T", rep=2, role="undeut"),
            MzmlRun(str(t60), "S", rep=1, time=60.0),
            MzmlRun(str(t60b), "T", rep=1, time=60.0)]
    peptides = list(PEPTIDES)
    targets = ([MzmlTarget(dataclasses.replace(_entry(p), sample="S")) for p in peptides]
               + [MzmlTarget(dataclasses.replace(_entry(p), sample="T"))
                  for p in peptides[1:]])
    config = MzmlConfig(find_peak=True, find_retention_time=mode)
    with monkeypatch.context() as m:
        _run_by_run(m)
        before = read_mzml(runs, targets, project="p", config=config)
    with monkeypatch.context() as m:
        opened, decoded = _counting(m)
        shared = read_mzml(runs, targets, project="p", config=config)
    _assert_identical(shared, before)
    record = shared.source["retention_search"]
    assert len(record["files"]) == 4 and {r["sample"] for r in record["series"]} == {"S", "T"}
    if mode == "automatic":
        assert any(r["applied"] for r in record["series"])
    # the spectra (find_peak's search range: every scan) and the search, each once; the
    # automatic search reads the series it moves again
    if mode == "suggest":
        assert decoded["un1.mzML"] == 2 * len(MzmlFile(un1).scans())
    # opened once (and once to look for ion mobility); the automatic search's reread once more
    assert opened["un1.mzML"] == (2 if mode == "suggest" else 3)
    with monkeypatch.context() as m:
        _run_by_run(m)
        before_opened, before_decoded = _counting(m)
        read_mzml(runs, targets, project="p", config=config)
    for name in ("un1.mzML", "un2.mzML"):  # run by run: twice the work
        assert before_decoded[name] == 2 * decoded[name], name
        assert before_opened[name] - 1 == 2 * (opened[name] - 1), name


def test_a_thermo_file_listed_by_two_samples_is_opened_once(tmp_path, monkeypatch):
    """A Thermo raw file of a parent and its fragments (reduced profile, recording its noise:
    the fragment spectra carry their scans' record; fragment peaks found) listed by two
    samples: opened once, the same spectra and record as opening it for each run."""
    from hxex.fragments import fragment_targets
    from hxex.io import read_thermo, thermo
    from test_mzml import _with_rt, fragment_lc_run
    from test_thermo import NoiseReader, _reduced

    parent = dataclasses.replace(_with_rt(ENTRY, 30.0 / 60), extra={
        **dict(_with_rt(ENTRY, 30.0 / 60).extra), "rt_start": str(20.0 / 60),
        "rt_end": str(40.0 / 60)})
    frags = fragment_targets(parent, "c:2-4:1", activation="ETD", rt_window=(20.0, 40.0))
    lc = fragment_lc_run(frags, other=None)
    scans = _reduced([{**{k: v for k, v in s.items() if k != "precursors"},
                       "time": s["time"] / 60,
                       **({"precursor": s["precursors"][0][:3],
                           "filter": f"FTMS + p ESI d Full ms2 {s['precursors'][0][0]:.4f}"
                                     "@etd20.00"} if "precursors" in s else {})} for s in lc])
    raw = tmp_path / "run.raw"
    raw.write_bytes(b"not really a raw file")
    runs = [MzmlRun(path=str(raw), sample=sample, time=60.0) for sample in ("S", "T")]
    targets = [dataclasses.replace(t, entry=dataclasses.replace(t.entry, sample=sample))
               for sample in ("S", "T")
               for t in [MzmlTarget(parent, rt_window=(20.0, 40.0)), *frags]]
    opens = []

    def open_reader(path, thermo_dir):
        opens.append(path)
        return NoiseReader(scans)

    monkeypatch.setattr(thermo, "_open_reader", open_reader)
    config = MzmlConfig(find_fragment_peak=True)
    with monkeypatch.context() as m:
        _run_by_run(m)
        before = read_thermo(runs, targets, project="p", config=config, thermo_dir="dlls")
    assert len(opens) == 2
    opens.clear()
    shared = read_thermo(runs, targets, project="p", config=config, thermo_dir="dlls")
    assert len(opens) == 1
    _assert_identical(shared, before)
    assert set(shared.sample) == {"S", "T"} and any(r is not None for r in shared.scans)
    assert shared.source["scan_cut"]["files"] == ["run.raw", "run.raw"]
