# SPDX-License-Identifier: GPL-3.0-or-later
"""Scan files read in parallel (hxex.io._parallel): the same spectra, record, and log lines as
a read in one process; how many processes; errors and scripts without a main guard."""

import dataclasses
import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from hxex.io import MzmlConfig, MzmlError, MzmlRun, MzmlTarget, _parallel, read_mzml
from test_mzml import ENTRY, _drifting, _with_rt, lc_run, write_mzml  # noqa: I001

SRC = Path(__file__).resolve().parents[1] / "src"


def _same(a, b):
    """Two SpectrumSets field by field (NaN equal to NaN; the record as stored)."""
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if f.name == "scans":
            assert [s is None for s in x] == [s is None for s in y]
        elif isinstance(x, np.ndarray) and x.dtype.kind == "f":
            np.testing.assert_array_equal(x, y)
        elif isinstance(x, np.ndarray):
            assert x.tolist() == y.tolist(), f.name
        else:
            assert x == y, f.name


def _both(caplog, *args, **kwargs):
    """The read in this process and with two processes, with the log lines of each."""
    out = []
    for workers in (1, 2):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="hxex"):
            spectra = read_mzml(*args, workers=workers, **kwargs)
        out.append((spectra, [(r.name, r.levelno, r.getMessage()) for r in caplog.records
                              if r.name.startswith("hxex")]))
    return out


def test_ion_mobility_files_read_in_parallel_give_the_same_spectra_and_lines(tmp_path, caplog):
    """Drift ranges from two undeuterated controls (one file listed by two samples) and four
    files read: two processes give what one does, record and log lines in the same order."""
    lc = [{"time": 1.0 + 0.1 * k, "filter": "Full ms"} for k in range(3)]
    peptide = {5.1: 0.02, 5.2: 0.4, 5.3: 1.0, 5.4: 0.5, 5.5: 0.05}
    other = {5.7: 1.0, 5.8: 0.6}
    un1 = write_mzml(tmp_path / "un1.mzML", _drifting(lc, [(1, 1e5, peptide), (7, 5e4, other)]),
                     indexed=True)
    un2 = write_mzml(tmp_path / "un2.mzML", _drifting(lc, [(1, 8e4, peptide), (7, 6e4, other)]))
    lab1 = write_mzml(tmp_path / "lab1.mzML", _drifting(lc, [(4, 1e5, peptide), (7, 5e4, other)]))
    lab2 = write_mzml(tmp_path / "lab2.mzML", _drifting(lc, [(5, 9e4, peptide), (7, 4e4, other)]))
    runs = [MzmlRun(un1, "WT", role="undeut"), MzmlRun(un1, "MUT", role="undeut"),
            MzmlRun(un2, "WT", rep=2, role="undeut"), MzmlRun(lab1, "WT", time=60.0),
            MzmlRun(lab2, "MUT", time=60.0)]
    target = MzmlTarget(ENTRY, rt_window=(60.0, 72.0))
    (one, lines_one), (two, lines_two) = _both(caplog, runs, [target], project="p")
    _same(one, two)
    assert len(one) == 5 and one.source["drift_gate"]["ranges"]
    assert lines_one == lines_two and any("drift ranges found" in m for _, _, m in lines_one)


def test_find_peak_in_parallel_gives_the_same_spectra(tmp_path, caplog):
    """find_peak's search, per file in its own process, and the progress of every run."""
    paths = [write_mzml(tmp_path / f"lc{k}.mzML", lc_run(apex=30.0 + k, seed=k))
             for k in range(3)]
    runs = [MzmlRun(str(p), "S", rep=k + 1, time=60.0) for k, p in enumerate(paths)]
    targets = [MzmlTarget(_with_rt(ENTRY, 30.5 / 60, z)) for z in (2, 3)]
    config = MzmlConfig(find_peak=True, peak_search=30.0)
    (one, lines_one), (two, lines_two) = _both(caplog, runs, targets, project="p",
                                               config=config)
    _same(one, two)
    assert lines_one == lines_two
    seen = []
    read_mzml(runs, targets, project="p", config=config, workers=2,
              progress=lambda done, total: seen.append((done, total)))
    assert seen[-1] == (3, 3)


def test_the_retention_search_of_controls_read_in_parallel_finds_the_same(tmp_path, caplog):
    """With find_peak the retention search ("suggest") looks at the undeuterated controls: each
    control file is read and searched in its own process, with the spectra, record (the
    search's suggestions included), and log lines of one process (a file's search lines come
    after its read lines, so lines may be ordered otherwise)."""
    paths = [write_mzml(tmp_path / f"lc{k}.mzML", lc_run(apex=30.0 + k, seed=k))
             for k in range(4)]
    runs = [MzmlRun(str(paths[0]), "S", role="undeut"),
            MzmlRun(str(paths[1]), "S", rep=2, role="undeut"),
            MzmlRun(str(paths[2]), "S", time=60.0), MzmlRun(str(paths[3]), "S", time=120.0)]
    targets = [MzmlTarget(_with_rt(ENTRY, 30.5 / 60, z)) for z in (2, 3)]
    config = MzmlConfig(find_peak=True, peak_search=30.0)
    (one, lines_one), (two, lines_two) = _both(caplog, runs, targets, project="p",
                                               config=config)
    _same(one, two)
    assert len(one.source["retention_search"]["files"]) == 2  # both controls searched
    assert sum("retention search: 2 series" in m for _, _, m in lines_two) == 2
    assert sorted(lines_one) == sorted(lines_two)


def test_how_many_processes_read(tmp_path, monkeypatch):
    """One file, a format that cannot be opened elsewhere, or small files: this process; never
    more processes than files; as many as the memory holds."""
    small = [tmp_path / f"{k}.mzML" for k in range(4)]
    for p in small:
        p.write_bytes(b"x" * 10)
    assert _parallel.read_workers(small[:1], 4) == 1
    assert _parallel.read_workers(small, 4, can=False) == 1
    assert _parallel.read_workers(small, None) == 1  # too small to be worth it
    assert _parallel.read_workers(small, 3) == 3 and _parallel.read_workers(small, 9) == 4
    monkeypatch.setattr(_parallel, "PARALLEL_BYTES", 1)
    monkeypatch.setattr(_parallel, "usable_cpus", lambda: 16)
    monkeypatch.setattr(_parallel, "available_memory", lambda: None)
    assert _parallel.read_workers(small, None) == 4
    each = _parallel.MEMORY_PER_BYTE * 10 + _parallel.MEMORY_START
    monkeypatch.setattr(_parallel, "available_memory",
                        lambda: int(2.5 * each / _parallel.MEMORY_SHARE))
    assert _parallel.read_workers(small, None) == 2
    monkeypatch.setattr(_parallel, "available_memory", lambda: 0)
    assert _parallel.read_workers(small, None) == 1


def test_an_error_in_a_process_is_the_reads_error(tmp_path):
    """A damaged file read in another process stops the read with its own error."""
    good = write_mzml(tmp_path / "lc0.mzML", lc_run(seed=0))
    bad = write_mzml(tmp_path / "lc1.mzML", [{**scan, "mz": scan["mz"][::-1]}
                                             for scan in lc_run(seed=1)])
    runs = [MzmlRun(str(good), "S", time=60.0), MzmlRun(str(bad), "S", time=120.0)]
    with pytest.raises(MzmlError, match="not in increasing order"):
        read_mzml(runs, [MzmlTarget(_with_rt(ENTRY, 0.5))], project="p",
                  config=MzmlConfig(find_peak=True), workers=2)


def test_a_script_without_a_main_guard_reads_in_parallel(tmp_path):
    """A script that reads at its top (no ``if __name__ == "__main__":``) is not run again by
    the processes (they would each read and start their own processes)."""
    for k in range(2):
        write_mzml(tmp_path / f"lc{k}.mzML", lc_run(seed=k))
    script = tmp_path / "read.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path[:0] = [{str(SRC)!r}, {str(Path(__file__).parent)!r}]
        from test_mzml import ENTRY, _with_rt
        from hxex.io import MzmlConfig, MzmlRun, MzmlTarget, read_mzml
        runs = [MzmlRun("lc0.mzML", "S", time=60.0), MzmlRun("lc1.mzML", "S", time=120.0)]
        s = read_mzml(runs, [MzmlTarget(_with_rt(ENTRY, 0.5))], project="p",
                      config=MzmlConfig(find_peak=True), workers=2)
        print("spectra", len(s))
    """))
    done = subprocess.run([sys.executable, script.name], cwd=tmp_path, capture_output=True,
                          text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    # the last line: an old brainpy says on its import that its compiled part is missing
    assert done.stdout.strip().splitlines()[-1] == "spectra 2"
