# SPDX-License-Identifier: GPL-3.0-or-later
"""The tutorial (docs/guide/tutorial.md) and its notebook (docs/tutorial) say the same thing,
and the download is built as the page describes. Working through it on the real data is
opt-in: HXEX_TUTORIAL_DATA = the published B5 SpecExport folder (about 20 minutes)."""

import ast
import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "guide" / "tutorial.md"
TUTORIAL = ROOT / "docs" / "tutorial"
sys.path.insert(0, str(TUTORIAL))

import make_tutorial  # noqa: E402


def page_blocks(kind: str) -> list[str]:
    return re.findall(rf"```{kind}\n(.*?)```", PAGE.read_text(encoding="utf-8"), re.S)


def notebook_code() -> list[str]:
    nb = json.loads(make_tutorial.NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_the_page_and_the_notebook_give_the_same_commands():
    page = [line.strip() for block in page_blocks("bash") for line in block.splitlines()
            if line.startswith("hxex ")]
    notebook = [line.strip()[1:] for cell in notebook_code() for line in cell.splitlines()
                if line.strip().startswith("!hxex ")]
    assert page == notebook
    assert page[0] == "hxex init --format specexport" and page[-1] == "hxex analyse hxex.toml"


def test_the_notebook_makes_the_changes_the_page_shows():
    """Every line the notebook writes into hxex.toml is in one of the page's TOML blocks."""
    shown = {line.strip() for block in page_blocks("toml") for line in block.splitlines()}
    written = []
    for cell in notebook_code():
        source = "\n".join(line for line in cell.splitlines()
                           if not line.startswith(("!", "%")))  # shell commands and magics
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "change":
                written.append(ast.literal_eval(node.args[1]))
            elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "add_before":
                written += [x for x in ast.literal_eval(node.args[1]).splitlines() if x]
    assert len(written) > 15
    missing = [w for w in written if not any(s.startswith(w) for s in shown)]
    assert not missing


def test_the_zip_holds_what_the_page_lists(tmp_path):
    import zipfile

    data = tmp_path / "SpecExport"
    for sample in make_tutorial.SAMPLES:
        (data / sample / "0001-0009-MDIAIHHPW").mkdir(parents=True)
        (data / sample / "0001-0009-MDIAIHHPW" / "4s-1-z2.csv").write_text("1,2\n")
        (data / f"{sample}.fasta").write_text(">x\nMDIAIHHPW\n")
    (data / "other_sample").mkdir()  # not part of the tutorial
    archive = make_tutorial.build_zip(data, tmp_path / "out")
    names = set(zipfile.ZipFile(archive).namelist())
    top = "hxex_b5_tutorial/"
    assert {top + n for n in ("hxex_b5_tutorial.ipynb", "README.txt", "LICENSE.txt",
                              "SpecExport/WT.fasta",
                              "SpecExport/R120G/0001-0009-MDIAIHHPW/4s-1-z2.csv")} <= names
    assert not any("other_sample" in n for n in names)


@pytest.mark.skipif(not os.environ.get("HXEX_TUTORIAL_DATA"), reason="HXEX_TUTORIAL_DATA not set")
def test_working_through_the_tutorial(tmp_path):
    pytest.importorskip("nbclient")
    import hxex
    from hxex.report import find_series

    archive = make_tutorial.build_zip(Path(os.environ["HXEX_TUTORIAL_DATA"]), tmp_path)
    folder = make_tutorial.unpack(archive, tmp_path / "work")
    make_tutorial.execute_notebook(folder)
    make_tutorial.write_images(folder, tmp_path / "images")
    results = hxex.load_results(folder / "hxex.toml")
    # the data_id numbers the page gives
    for rng, data_id in (("87-96", 140), ("42-54", 56), ("68-75", 107)):
        assert find_series(results, samples=["WT"], peptide_ranges=[rng], charges=[2]) == [data_id]
    assert len(list((tmp_path / "images").glob("tutorial_*.png"))) == 7
