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
