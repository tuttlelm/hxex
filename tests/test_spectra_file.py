# SPDX-License-Identifier: GPL-3.0-or-later
"""Raw spectra saved to and loaded from a single Parquet file, without a fit."""

import os
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import hxex.io.spectra_file as spectra_file
from hxex.io import ReadError, load_spectra, save_spectra, spectra_info
from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet


def make_spectra(n=7, project="proj", seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        entry = PeptideEntry(f"f{i % 2}.xlsx", "sample é", "PEPacKTIDE", "0001-0008", 1, 8,
                             2 + i % 2, modification="H:1 Hex:1" if i % 3 == 0 else "")
        size = int(rng.integers(0, 60)) if i == 3 else int(rng.integers(20, 200))  # one empty
        mz = np.sort(rng.uniform(500, 900, size))
        out.append(Spectrum(mz, rng.uniform(0, 1e6, size), entry, time=float(i * 60.5),
                            rep=1 + i % 3, data_id=i % 2, label=f"1.{i} min",
                            spectrum_id=f"{project}:f{i % 2}.xlsx#{i + 1}"))
    return SpectrumSet.from_spectra(out)


def assert_same(a: SpectrumSet, b: SpectrumSet):
    for name in ("mz", "intensity", "offsets", "spectrum_id", "sample", "peptide", "peptide_range",
                 "file", "label", "modification", "start_seq", "end_seq", "charge", "rep",
                 "data_id", "time"):
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        assert x.shape == y.shape and x.dtype.kind == y.dtype.kind, name
        np.testing.assert_array_equal(x, y)


# --- round trips ----------------------------------------------------------------------------

def test_exact_round_trip(tmp_path):
    s = make_spectra()
    path = save_spectra(s, tmp_path / "spectra.parquet")
    assert path.exists() and not list(tmp_path.glob("*.tmp-*"))
    assert_same(load_spectra(path), s)


def test_empty_and_single_spectrum_sets(tmp_path):
    empty = SpectrumSet.from_spectra([])
    assert_same(load_spectra(save_spectra(empty, tmp_path / "e.parquet")), empty)
    one = make_spectra(1)
    assert_same(load_spectra(save_spectra(one, tmp_path / "o.parquet")), one)


def test_metadata_and_project(tmp_path):
    s = make_spectra()
    info = spectra_info(save_spectra(s, tmp_path / "a.parquet"))
    assert info["kind"] == "hxex-spectra" and info["schema_version"] == "1"
    assert info["project"] == "proj" and info["n_spectra"] == str(len(s))
    assert len(info["content_sha256"]) == 64 and info["created"].endswith("+00:00")
    named = save_spectra(s, tmp_path / "b.parquet", project="tubB5")
    assert spectra_info(named)["project"] == "tubB5"
    mixed = SpectrumSet.from_spectra(
        [Spectrum(np.array([1.0]), np.array([2.0]), PeptideEntry("f", "s", "AK", "1-2", 1, 2, 1),
                  0.0, 1, spectrum_id=i) for i in ("a:1", "b:2")]
    )
    assert spectra_info(save_spectra(mixed, tmp_path / "m.parquet"))["project"] == ""


def test_the_file_has_plain_arrow_types(tmp_path):
    path = save_spectra(make_spectra(), tmp_path / "t.parquet")
    types = {f.name: str(f.type) for f in pq.read_schema(path)}
    assert types["mz"] == "list<element: double>" and types["intensity"] == "list<element: double>"
    assert types["spectrum_id"] == "string" and types["charge"] == "int64"
    assert types["time"] == "double"
    column = pq.ParquetFile(path).metadata.row_group(0).column(0)
    assert column.compression == "ZSTD"


# --- reading part of a file -----------------------------------------------------------------

def test_spectra_by_id(tmp_path):
    s = make_spectra(12)
    path = save_spectra(s, tmp_path / "s.parquet")
    ids = [str(s.spectrum_id[9]), str(s.spectrum_id[2]), "proj:not-there"]
    sub = load_spectra(path, ids)
    assert list(sub.spectrum_id) == [str(s.spectrum_id[2]), str(s.spectrum_id[9])]  # file order
    for sid in sub.spectrum_id:
        np.testing.assert_array_equal(sub.spectrum(sub.find(sid))[1], s.spectrum(s.find(sid))[1])
    assert len(load_spectra(path, [])) == 0
    assert len(load_spectra(path, ["proj:not-there"])) == 0


def test_many_row_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(spectra_file, "ROW_GROUP", 3)
    s = make_spectra(10)
    path = save_spectra(s, tmp_path / "g.parquet")
    assert pq.ParquetFile(path).num_row_groups == 4
    assert_same(load_spectra(path), s)
    assert list(load_spectra(path, [str(s.spectrum_id[8])]).spectrum_id) == [str(s.spectrum_id[8])]


# --- integrity ------------------------------------------------------------------------------

def test_a_changed_value_is_caught_by_the_checksum(tmp_path):
    s = make_spectra()
    path = save_spectra(s, tmp_path / "d.parquet")
    table = pq.read_table(path)
    mz = table.column("mz").combine_chunks()
    values = mz.flatten().to_numpy().copy()
    values[0] += 1e-6  # one point of one spectrum
    altered = table.set_column(
        table.schema.get_field_index("mz"), "mz",
        pa.ListArray.from_arrays(mz.offsets, pa.array(values)),
    ).replace_schema_metadata(table.schema.metadata)
    pq.write_table(altered, path)
    with pytest.raises(ReadError, match="checksum"):
        load_spectra(path)
    np.testing.assert_array_equal(load_spectra(path, verify=False).mz[1:], s.mz[1:])


def test_content_digest_depends_on_the_content():
    s = make_spectra()
    assert spectra_file.content_digest(s) == spectra_file.content_digest(make_spectra())
    assert spectra_file.content_digest(s) != spectra_file.content_digest(make_spectra(seed=1))
    assert spectra_file.content_digest(s) != spectra_file.content_digest(make_spectra(project="x"))


# --- safe writing ---------------------------------------------------------------------------

def test_existing_files_are_protected(tmp_path):
    s = make_spectra()
    path = save_spectra(s, tmp_path / "s.parquet")
    with pytest.raises(FileExistsError, match="overwrite"):
        save_spectra(s, path)
    save_spectra(make_spectra(3), path, overwrite=True)
    assert len(load_spectra(path)) == 3
    (tmp_path / "folder").mkdir()
    with pytest.raises(FileExistsError, match="directory"):
        save_spectra(s, tmp_path / "folder", overwrite=True)


def test_a_failed_write_leaves_nothing_and_keeps_the_old_file(tmp_path, monkeypatch):
    s = make_spectra()
    path = save_spectra(s, tmp_path / "s.parquet")
    before = path.read_bytes()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pq, "write_table", boom)
    with pytest.raises(RuntimeError, match="boom"):
        save_spectra(make_spectra(2), path, overwrite=True)
    assert path.read_bytes() == before and [p.name for p in tmp_path.iterdir()] == ["s.parquet"]
    with pytest.raises(RuntimeError, match="boom"):
        save_spectra(s, tmp_path / "new.parquet")
    assert [p.name for p in tmp_path.iterdir()] == ["s.parquet"]


def test_the_directory_is_created(tmp_path):
    path = save_spectra(make_spectra(2), tmp_path / "a" / "b" / "s.parquet")
    assert len(load_spectra(path)) == 2


# --- files that are not spectra files -------------------------------------------------------

def test_unreadable_files(tmp_path):
    with pytest.raises(ReadError, match="cannot read"):
        load_spectra(tmp_path / "missing.parquet")
    (tmp_path / "junk.parquet").write_bytes(b"not parquet")
    with pytest.raises(ReadError, match="cannot read"):
        load_spectra(tmp_path / "junk.parquet")
    pq.write_table(pa.table({"x": [1, 2]}), tmp_path / "other.parquet")
    with pytest.raises(ReadError, match="missing column"):
        load_spectra(tmp_path / "other.parquet")
    assert spectra_info(tmp_path / "other.parquet") == {}


def test_other_and_newer_formats(tmp_path):
    table = pq.read_table(save_spectra(make_spectra(2), tmp_path / "s.parquet"))
    meta = dict(table.schema.metadata)
    other = table.replace_schema_metadata({**meta, b"hxex.kind": b"other"})
    pq.write_table(other, tmp_path / "k.parquet")
    with pytest.raises(ReadError, match="not a spectra file"):
        load_spectra(tmp_path / "k.parquet")
    newer = table.replace_schema_metadata({**meta, b"hxex.schema_version": b"9"})
    pq.write_table(newer, tmp_path / "v.parquet")
    with pytest.raises(ReadError, match="format version 9"):
        load_spectra(tmp_path / "v.parquet")


# --- the bundle's own spectra table ----------------------------------------------------------

def test_the_spectra_table_of_a_bundle_is_the_same_format(tmp_path):
    from hxex.io import open_bundle, write_bundle
    from test_bundle import build_run  # noqa: I001

    raw, binned, meas, fit, boot = build_run()
    bundle = write_bundle(tmp_path / "b", measurements=meas, fit=fit, binned=binned, spectra=raw)
    assert_same(load_spectra(bundle / "spectra.parquet"), raw)  # no hxex metadata: no checksum
    assert spectra_info(bundle / "spectra.parquet") == {}
    mine = save_spectra(raw, tmp_path / "mine.parquet")
    assert [(f.name, str(f.type)) for f in pq.read_schema(mine)] == [
        (f.name, str(f.type)) for f in pq.read_schema(bundle / "spectra.parquet")]
    with open_bundle(bundle) as b:
        assert_same(b.spectra(), load_spectra(mine))


# --- no torch, and real data ----------------------------------------------------------------

def test_saving_and_loading_do_not_import_torch(tmp_path):
    target = tmp_path / "x.parquet"
    code = "\n".join([
        "import sys, numpy as np",
        "from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet",
        "from hxex.io import save_spectra, load_spectra",
        "e = PeptideEntry('f', 's', 'AK', '1-2', 1, 2, 1)",
        "sp = Spectrum(np.array([1.0, 2.0]), np.array([3.0, 4.0]), e, 0.0, 1, spectrum_id='p:1')",
        f"save_spectra(SpectrumSet.from_spectra([sp]), r'{target}'); t = load_spectra(r'{target}')",
        "assert len(t) == 1 and 'torch' not in sys.modules",
    ])
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


@pytest.mark.skipif(not os.environ.get("HXEX_BIMODAL_DATA"), reason="HXEX_BIMODAL_DATA not set")
def test_the_real_raw_spectra(tmp_path):
    from hxex import io as hio

    root = os.environ["HXEX_BIMODAL_DATA"]
    entries = hio.read_peptide_table(os.path.join(root, "hdxms_testsets_metadf.csv"))
    raw = hio.read_spectra(entries, root, "hxexpress", project="bimodal")
    path = save_spectra(raw, tmp_path / "raw.parquet")
    assert path.stat().st_size < 1.3e6  # 566 spectra, 496,708 points: about 1.07 MB
    t = time.perf_counter()
    got = load_spectra(path)
    assert time.perf_counter() - t < 2.0
    assert_same(got, raw)
