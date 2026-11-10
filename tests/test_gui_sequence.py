# SPDX-License-Identifier: GPL-3.0-or-later
"""The Project workspace's sequence view (hxex.gui.sequence_view; owner, 2026-11-04): each
sample's sequence as the analysis numbers it, rows shared by samples with the same sequence,
differences from the reference highlighted, regions and sites marked, a residue described."""
import os

import pytest

from hxex.gui.sequence_view import describe, project_sequences, residues_of, sequence_html


def _project(tmp_path, fasta: str, analysis: str = "") -> object:
    (tmp_path / "SpecExport").mkdir()
    for name, seq in (("WT", "MKPLAAGDKE"), ("M1", "MKPLAHGDKE"), ("M2", "MKPLAAGDKE")):
        (tmp_path / f"{name}.fasta").write_text(f">{name}{fasta}\n{seq}\n", encoding="utf-8")
    path = tmp_path / "hxex.toml"
    path.write_text('project = "seq"\n\n[input]\nformat = "specexport"\ndata_dir = "SpecExport"\n\n'
                    '[analysis]\nreference = "WT"\nfasta = { WT = "WT.fasta", M1 = "M1.fasta", '
                    'M2 = "M2.fasta" }\n' + analysis, encoding="utf-8")
    return path


def test_the_sequences_are_read_and_grouped(tmp_path):
    marks = ('[[analysis.regions]]\nname = "loop"\nstart = 5\nend = 7\n'
             '[[analysis.sites]]\nresidue = 8\nlabel = "S8"\n')
    project = _project(tmp_path, "/3-12", marks)
    rows, annotations, message = project_sequences(project)
    assert message == ""
    assert [r.samples for r in rows] == [["WT", "M2"], ["M1"]]  # the reference first; same: one
    assert rows[0].first == 3 and rows[0].last == 12  # numbered by the FASTA header
    assert rows[1].differs == {8} and rows[1].at(8) == "H"  # residue 8 = 6th letter
    text = sequence_html(rows, annotations)
    assert "res:1:8" in text and "loop (5–7)" in text and "site" in text
    said = describe(rows, annotations, 1, 8)
    assert said == "M1: residue 8, H; differs: WT, M2 A; site S8"
    assert "a proline" in describe(rows, annotations, 0, 5)  # P at residue 5
    assert "region loop (5–7)" in describe(rows, annotations, 0, 6)


def test_no_sequence_says_where_to_put_one(tmp_path):
    (tmp_path / "SpecExport").mkdir()  # SpecExport, no FASTA beside the folders, none named
    path = tmp_path / "hxex.toml"
    path.write_text('project = "seq"\n\n[input]\nformat = "specexport"\n'
                    'data_dir = "SpecExport"\n', encoding="utf-8")
    rows, _, message = project_sequences(path)
    assert rows == [] and "[analysis] fasta" in message
    assert residues_of("PEPacKX") == ("P", "E", "P", "acK", "X")  # by residues, X kept


def test_the_view_shows_the_project_and_describes_a_click(qtbot, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QUrl

    from hxex.gui.sequence_view import SequenceView

    project = _project(tmp_path, "")
    view = SequenceView()
    qtbot.addWidget(view)
    view.show_project(project)
    assert "3 sample(s), 2 distinct sequence(s)" in view.about.text()
    view._clicked(QUrl("res:1:6"))
    assert view.about.text().startswith("M1: residue 6, H")
    view.show_project(None)
    assert "No project file open" in view.text.toPlainText()


def test_the_sequences_a_run_stored_stand_in_for_missing_files(tmp_path):
    """A SpecExport run keeps each sample's sequence in its bundle (owner, 2026-11-04): the
    files come first, a file that no longer matches is said, and the stored copies are shown
    when the files are gone."""
    import json

    from hxex.io.bundle import KIND, MANIFEST

    data = tmp_path / "SpecExport"
    data.mkdir()
    (data / "WT.fasta").write_text(">WT/3-12\nMKPLAAGDKE\n", encoding="utf-8")
    (data / "M1.fasta").write_text(">M1/3-12\nMKPLAHGDKE\n", encoding="utf-8")
    (tmp_path / "out").mkdir()
    stored = {"WT": {"sequence": "MKPLAAGDKE", "first": 3, "file": "WT.fasta"},
              "M1": {"sequence": "MKPLAAGDKE", "first": 3, "file": "M1.fasta"}}
    (tmp_path / "out" / MANIFEST).write_text(json.dumps(
        {"kind": KIND, "configs": {"spectra_source": {"reader": "specexport",
                                                      "sequences": stored}}}))
    path = tmp_path / "hxex.toml"
    path.write_text('project = "seq"\n\n[input]\nformat = "specexport"\ndata_dir = "SpecExport"\n'
                    '\n[output]\nbundle = "out"\n', encoding="utf-8")
    rows, _, message = project_sequences(path)
    assert [r.samples for r in rows] == [["M1"], ["WT"]]  # the files, as they are now
    assert message.startswith("M1.fasta differs from the copy the run stored")
    for f in data.iterdir():
        f.unlink()
    rows, _, message = project_sequences(path)
    assert [r.samples for r in rows] == [["M1", "WT"]] and rows[0].first == 3
    assert "the sequences the run stored in its bundle are shown" in message


def test_coverage_runs_under_each_samples_sequence(tmp_path):
    """With results open, each sample's coverage runs under its row (owner, 2026-11-05): a block
    as high as the peptides covering a residue, a dot where none does; a click lists them; a
    peptide whose every spectrum the review rejected does not cover."""
    from types import SimpleNamespace

    import numpy as np

    from hxex.gui.sequence_view import coverage_of, depth

    project = _project(tmp_path, "/3-12")
    rows, annotations, _ = project_sequences(project)
    binned = SimpleNamespace(
        spectrum_id=np.array(["a", "b", "c", "d"]), sample=np.array(["WT", "WT", "M1", "WT"]),
        start_seq=np.array([3, 5, 3, 9]), end_seq=np.array([7, 8, 12, 12]),
        peptide_range=np.array(["0003-0007", "0005-0008", "0003-0012", "0009-0012"]))
    results = SimpleNamespace(binned=binned, fit=SimpleNamespace(spectrum_id=binned.spectrum_id),
                              review_status=np.array(["", "", "", "rejected"]))
    coverage = coverage_of(results)
    assert coverage == {"M1": [(3, 12, "0003-0012")],
                        "WT": [(3, 7, "0003-0007"), (5, 8, "0005-0008")]}  # 9-12 rejected
    assert [depth(coverage["WT"], n) for n in (3, 5, 9)] == [1, 2, 0]
    text = sequence_html(rows, annotations, coverage=coverage)
    assert "peptides covering a residue (1 to 2)" in text and "·" in text
    assert text.count("█") >= 1  # residues 5-7 of WT, at the most
    said = describe(rows, annotations, 0, 6, coverage)
    assert "WT: 2 peptide(s), 3-7, 5-8" in said
    assert "WT: not covered" in describe(rows, annotations, 0, 10, coverage)
    assert "covering" not in sequence_html(rows, annotations)  # without results: none


def _pooled_project(tmp_path, fasta: str | None) -> object:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "pool").mkdir()
    lines = ['project = "pool"', "", "[input]", 'format = "pooled"', 'data_dir = "pool"',
             'sample = "S"']
    if fasta is not None:
        (tmp_path / "pool.fasta").write_text(fasta, encoding="utf-8")
        lines += ['pooled_fasta = "pool.fasta"', 'pooled_classes = ["A", "B"]']
    path = tmp_path / "hxex.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


POOL = ">A1\nMKVLGGGPEK\n>B1\nQQMKVLGGG\n>A2\nMKVLSSGPEK\n>A3\nMKVLGGGPEK\n"


def test_a_pooled_project_shows_its_records_by_class(tmp_path):
    """Pooled data (owner, 2026-11-10): the records of [input] pooled_fasta, grouped by class
    in the order the classes are named, FASTA order within a class, a record identical to
    another of its class on its row, differences counted from the first of its class."""
    rows, annotations, message = project_sequences(_pooled_project(tmp_path, POOL))
    assert [(r.samples, r.group) for r in rows] == [(["A1", "A3"], "A"), (["A2"], "A"),
                                                   (["B1"], "B")]
    assert rows[1].differs == {5, 6} and rows[2].differs == set()  # B1 against no other B
    assert "pool.fasta, in 2 class(es)" in message
    said = describe(rows, annotations, 1, 5)
    assert said == "A2 (class A): residue 5, S; differs: A1, A3 G"  # B1 not compared
    text = sequence_html(rows, annotations)
    assert "A2" in text and "\n\nB1" in text  # a line between the classes
    rows, _, message = project_sequences(_pooled_project(tmp_path / "bare", None))
    assert rows == [] and "[input] pooled_fasta" in message


def test_pooled_coverage_is_each_records_in_its_own_numbering(tmp_path):
    """With pooled results open, each record's coverage is that of the peptides it may hold,
    moved to its own numbering, a peptide only that record may hold drawn in black."""
    from types import SimpleNamespace

    import numpy as np

    from hxex.chem import DEFAULT_MODIFICATIONS
    from hxex.gui.sequence_view import OWN, coverage_of, pooled_coverage_of
    from hxex.io.pooled import candidates_record

    (tmp_path / "pool.fasta").write_text(POOL, encoding="utf-8")
    record = candidates_record(tmp_path / "pool.fasta", ["A", "B"])
    binned = SimpleNamespace(
        spectrum_id=np.array(["a", "b", "c"]), sample=np.array(["S", "S", "S"]),
        peptide=np.array(["MKVL", "SSGPEK", "MKVL"]), start_seq=np.array([1, 5, 1]),
        end_seq=np.array([4, 10, 4]),
        peptide_range=np.array(["0001-0004", "0005-0010", "0001-0004"]))
    results = SimpleNamespace(
        binned=binned, fit=SimpleNamespace(spectrum_id=binned.spectrum_id,
                                           data_id=np.array([0, 1, 0])),
        review_status=np.array(["", "", "rejected"]),
        settings=SimpleNamespace(spectra_source={"reader": "pooled",
                                                 "pooled_candidates": record},
                                 registry=DEFAULT_MODIFICATIONS))
    coverage = coverage_of(results)
    assert coverage == pooled_coverage_of(results)
    assert coverage["B1"] == [(3, 6, "0001-0004", False)]  # MKVL starts at B1's residue 3
    assert coverage["A2"] == [(1, 4, "0001-0004", False), (5, 10, "0005-0010", True)]
    assert coverage["A1"] == coverage["A3"] == [(1, 4, "0001-0004", False)]
    rows, annotations, _ = project_sequences(_pooled_project(tmp_path / "project", POOL))
    text = sequence_html(rows, annotations, coverage=coverage)
    assert f"color:{OWN};" in text and "only this record may hold" in text
    assert "(only this record)" in describe(rows, annotations, 1, 6, coverage)


def test_the_residues_a_line_are_chosen_in_the_view(qtbot, tmp_path):
    """Owner, 2026-11-10: how many residues a line holds is a choice of the view (blocks of
    ten), kept by the window's session only when it is not the default."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from hxex.gui.sequence_view import PER_LINE, SequenceView

    view = SequenceView()
    qtbot.addWidget(view)
    view.show_project(_project(tmp_path, ""))
    assert view.per_line == PER_LINE and view.view_settings() == {}
    view.set_per_line(80)
    assert view.view_settings() == {"residues_per_line": 80}
    long_rows = [type(r)(r.samples, r.residues * 20, r.first) for r in view.rows]
    lines = [line for line in sequence_html(long_rows, None, per_line=80).splitlines()
             if line.startswith("WT, M2")]
    assert len(lines) == 3  # 200 residues at 80 a line
    view.set_per_line(33)  # not a choice: the default
    assert view.per_line == PER_LINE
    view.set_per_line(None)
    assert view.per_line == PER_LINE

