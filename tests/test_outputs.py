# SPDX-License-Identifier: GPL-3.0-or-later
"""Output folders (tables, figures) keep a listing of hxex's files: those are replaced, files of
your own are kept, and one named like a new file is refused before anything changes."""
import json

import pytest

from hxex.io import OutputError, output_listing, replace_outputs
from hxex.io.outputs import LISTING
from hxex.report import Results, write_tables
from hxex.report.figures import PlotConfig, write_figures
from test_run import config, raw_spectra, run


@pytest.fixture(scope="module")
def results():
    return Results.from_run(run(raw_spectra(n_series=2), config(bootstrap=None)))


def test_replace_outputs_keeps_your_files_and_replaces_its_own(tmp_path):
    out = tmp_path / "o"
    assert output_listing(out) is None
    with replace_outputs(out, ["a.txt", "b.txt"]) as written:
        for name in ("a.txt", "b.txt"):
            (out / name).write_text(name)
            written.append(out / name)
    assert output_listing(out) == ["a.txt", "b.txt"]
    (out / "mine.txt").write_text("keep")
    with replace_outputs(out, ["a.txt", "c.txt"]) as written:
        assert not (out / "a.txt").exists() and not (out / "b.txt").exists()  # listed: gone
        assert (out / "mine.txt").exists()
        (out / "c.txt").write_text("c")
        written.append(out / "c.txt")
    assert sorted(p.name for p in out.iterdir()) == ["c.txt", LISTING, "mine.txt"]
    # a file of yours named like a new file: refused before anything changes
    with pytest.raises(OutputError, match="mine.txt"):
        with replace_outputs(out, ["c.txt", "mine.txt"]):
            pytest.fail("entered the block")
    assert (out / "c.txt").read_text() == "c" and output_listing(out) == ["c.txt"]
    # a failure part-way still leaves the listing of what was written, so the next write
    # knows those files are hxex's
    with pytest.raises(RuntimeError):
        with replace_outputs(out, ["d.txt", "e.txt"]) as written:
            (out / "d.txt").write_text("d")
            written.append(out / "d.txt")
            raise RuntimeError("stopped")
    assert output_listing(out) == ["d.txt"]
    with replace_outputs(out, ["d.txt"]):
        pass
    assert output_listing(out) == [] and not (out / "d.txt").exists()
    # files in a sub-folder are listed with "/" (a listing written on Windows with "\\" is
    # read the same); a path outside the folder is left out, not raised over the real error
    with pytest.raises(RuntimeError, match="stopped"):
        with replace_outputs(out, ["sub/f.txt"]) as written:
            (out / "sub").mkdir()
            (out / "sub" / "f.txt").write_text("f")
            written += [out / "sub" / "f.txt", tmp_path / "elsewhere.txt"]
            raise RuntimeError("stopped")
    assert output_listing(out) == ["sub/f.txt"]
    text = (out / LISTING).read_text().replace("sub/f.txt", "sub\\\\f.txt")
    (out / LISTING).write_text(text)
    assert output_listing(out) == ["sub/f.txt"]
    with replace_outputs(out, ["sub/f.txt"]):
        pass
    assert not (out / "sub" / "f.txt").exists()
    # a folder with files but no listing (written before, or yours): only a clash is refused
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "series.csv").write_text("old")
    with pytest.raises(OutputError, match="before hxex kept a listing"):
        with replace_outputs(legacy, ["series.csv"]):
            pytest.fail("entered the block")
    with replace_outputs(legacy, ["other.csv"]):
        pass
    assert (legacy / "series.csv").read_text() == "old"
    (legacy / LISTING).write_text(json.dumps({"kind": "something else"}))
    assert output_listing(legacy) is None


def test_tables_and_figures_are_listed_and_replaced(tmp_path, results):
    tables = tmp_path / "t"
    first = write_tables(results, tables)
    assert output_listing(tables) == sorted(p.name for p in first)
    (tables / "notes.md").write_text("mine")
    write_tables(results, tables)
    assert (tables / "notes.md").read_text() == "mine"
    (tables / "uptake.csv").unlink()
    (tables / LISTING).unlink()
    with pytest.raises(OutputError, match="series.csv"):
        write_tables(results, tables)
    figures = tmp_path / "f"
    png = PlotConfig(series="all", kinds=("uptake",), format="png", overview=False)
    first = write_figures(results, figures, png)
    assert len(first) == 2 and output_listing(figures) == sorted(p.name for p in first)
    pdf = PlotConfig(series=(0,), kinds=("uptake",), format="pdf", overview=False)
    second = write_figures(results, figures, pdf)
    assert len(second) == 1  # the png files of the earlier call are gone
    assert sorted(p.name for p in figures.iterdir()) == sorted([LISTING, second[0].name])
