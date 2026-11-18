# SPDX-License-Identifier: GPL-3.0-or-later
"""The Project workspace's sequence view (hxex.gui.sequence_view; owner, 2026-11-04): each
sample's sequence as the analysis numbers it, rows shared by samples with the same sequence,
differences from the reference highlighted, regions and sites marked, a residue described."""

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
    (data / "WT").mkdir()
    (data / "M1").mkdir()
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
    for f in data.glob("*.fasta"):
        f.unlink()
    rows, _, message = project_sequences(path)
    assert [r.samples for r in rows] == [["M1", "WT"]] and rows[0].first == 3
    assert "the sequences the run stored in its bundle are shown" in message


def test_only_the_samples_the_project_reads(tmp_path):
    """[input] samples limits the view (owner, 2026-11-13: projects split by sample from one
    data folder): the FASTA files beside the folders, matched in any case as the reader
    pairs them, and the per-sample [analysis.fasta] table."""
    data = tmp_path / "SpecExport"
    data.mkdir()
    (data / "WT.fasta").write_text(">WT/3-12\nMKPLAAGDKE\n", encoding="utf-8")
    (data / "M1.fasta").write_text(">M1/3-12\nMKPLAHGDKE\n", encoding="utf-8")
    (data / "M2.FASTA").write_text(">M2/3-12\nMKPLAWGDKE\n", encoding="utf-8")
    for name in ("WT", "M1", "M2"):
        (data / name).mkdir()
    path = tmp_path / "hxex.toml"
    head = 'project = "seq"\n\n[input]\nformat = "specexport"\ndata_dir = "SpecExport"\n'
    path.write_text(head, encoding="utf-8")
    assert sorted(s for r in project_sequences(path)[0] for s in r.samples) == ["M1", "M2", "WT"]
    path.write_text(head + 'samples = ["WT", "m2"]\n', encoding="utf-8")
    assert sorted(s for r in project_sequences(path)[0] for s in r.samples) == ["M2", "WT"]
    path.write_text(head + 'samples = ["WT"]\n\n[analysis.fasta]\nWT = "SpecExport/WT.fasta"\n'
                    'M1 = "SpecExport/M1.fasta"\n', encoding="utf-8")
    assert [r.samples for r in project_sequences(path)[0]] == [["WT"]]


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
    assert "MS1 peptides covering a residue (1 to 2;" in text and "·" in text
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



def test_a_project_without_sequences_leaves_nothing_of_the_one_before(qtbot, tmp_path):
    """E-13: a project without sequences (or none open) after one with: the line under the
    text no longer gives the previous project's counts, nor its message."""
    pytest.importorskip("PySide6")
    from hxex.gui.sequence_view import ABOUT, SequenceView

    first = tmp_path / "first"
    first.mkdir()
    view = SequenceView()
    qtbot.addWidget(view)
    view.show_project(_project(first, ""))
    assert "distinct sequence(s)" in view.about.text()
    view._message = "a message about the first project"
    bare = tmp_path / "bare"
    (bare / "SpecExport").mkdir(parents=True)
    path = bare / "hxex.toml"
    path.write_text('project = "seq"\n\n[input]\nformat = "specexport"\n'
                    'data_dir = "SpecExport"\n', encoding="utf-8")
    view.show_project(path)
    assert "[analysis] fasta" in view.text.toPlainText()
    assert view.about.text() == ABOUT and view._message == ""
    view.show_project(first / "hxex.toml")
    assert "distinct sequence(s)" in view.about.text()
    view.show_project(None)
    assert view.text.toPlainText() == "No project file open."
    assert view.about.text() == ABOUT and view._message == ""


def _stored_project(tmp_path, fasta_line: str = "") -> object:
    """A SpecExport project whose run stored WT's and M1's sequences in its bundle, with
    WT.fasta and M1.fasta beside their sample folders."""
    import json

    from hxex.io.bundle import KIND, MANIFEST

    data = tmp_path / "SpecExport"
    data.mkdir()
    for name, seq in (("WT", "MKPLAAGDKE"), ("M1", "MKPLAHGDKE")):
        (data / f"{name}.fasta").write_text(f">{name}/3-12\n{seq}\n", encoding="utf-8")
        (data / name).mkdir()
    (tmp_path / "out").mkdir()
    stored = {"WT": {"sequence": "MKPLAAGDKE", "first": 3, "file": "WT.fasta"},
              "M1": {"sequence": "MKPLAHGDKE", "first": 3, "file": "M1.fasta"}}
    (tmp_path / "out" / MANIFEST).write_text(json.dumps(
        {"kind": KIND, "configs": {"spectra_source": {"reader": "specexport",
                                                      "sequences": stored}}}))
    path = tmp_path / "hxex.toml"
    path.write_text('project = "seq"\n\n[input]\nformat = "specexport"\ndata_dir = "SpecExport"\n'
                    '\n[output]\nbundle = "out"\n' + fasta_line, encoding="utf-8")
    return path


def test_one_missing_fasta_takes_the_stored_copy_with_a_note(tmp_path):
    """E-14: one sample's FASTA deleted: that sample is shown with the copy the run stored in
    its bundle, and a note names the file (it no longer drops out without a word)."""
    path = _stored_project(tmp_path)
    (tmp_path / "SpecExport" / "M1.fasta").unlink()
    rows, _, message = project_sequences(path)
    assert sorted(s for r in rows for s in r.samples) == ["M1", "WT"]
    m1 = next(r for r in rows if "M1" in r.samples)
    assert m1.first == 3 and m1.at(8) == "H"  # the stored copy, numbered as stored
    assert message == ("M1.fasta is not found: the sequence the run stored in its bundle is "
                       "shown for M1.")


def test_a_missing_analysis_fasta_file_takes_the_stored_copy(tmp_path):
    """Open question 6: an [analysis] fasta entry naming a missing file shows the sequence the
    run stored, with a note naming the file, not an error; a sample the run stored nothing
    for is said."""
    path = _stored_project(tmp_path, '\n[analysis.fasta]\nWT = "SpecExport/WT.fasta"\n'
                                     'M1 = "gone/M1.fasta"\n')
    rows, _, message = project_sequences(path)
    assert sorted(s for r in rows for s in r.samples) == ["M1", "WT"]
    assert "does not read" not in message
    assert message == ("M1.fasta is not found: the sequence the run stored in its bundle is "
                       "shown for M1.")
    path.write_text(path.read_text(encoding="utf-8").replace("M1 =", "M9 ="), encoding="utf-8")
    rows, _, message = project_sequences(path)
    assert [r.samples for r in rows] == [["WT"]]
    assert "M1.fasta ([analysis] fasta, M9) is not found, and the run stored no copy" in message
    # one file for every sample, missing: each sample's stored copy
    text = path.read_text(encoding="utf-8")
    path.write_text(text[:text.index("\n[analysis.fasta]")]
                    + '\n[analysis]\nfasta = "gone/all.fasta"\n', encoding="utf-8")
    rows, _, message = project_sequences(path)
    assert sorted(s for r in rows for s in r.samples) == ["M1", "WT"]
    assert message == ("all.fasta is not found: the sequences the run stored in its bundle are "
                       "shown.")


def test_a_fasta_without_its_sample_folder_is_left_out(tmp_path):
    """Open question 7: a SAMPLE.fasta with no sample folder of its name beside it (the reader
    does not read it) is left out of the view, with a note."""
    data = tmp_path / "SpecExport"
    data.mkdir()
    (data / "WT.fasta").write_text(">WT/3-12\nMKPLAAGDKE\n", encoding="utf-8")
    (data / "WT").mkdir()
    (data / "M1.fasta").write_text(">M1/3-12\nMKPLAHGDKE\n", encoding="utf-8")
    (data / "m1").mkdir()  # another case: not M1's folder for the reader
    path = tmp_path / "hxex.toml"
    path.write_text('project = "seq"\n\n[input]\nformat = "specexport"\n'
                    'data_dir = "SpecExport"\n', encoding="utf-8")
    rows, _, message = project_sequences(path)
    assert [r.samples for r in rows] == [["WT"]]
    assert message == ("M1.fasta has no sample folder of its name beside it, so hxex does not "
                       "read it: left out.")


def test_the_coverage_counts_ms1_peptides_only(tmp_path):
    """E-21: fragment ions (an ion type other than M) do not count as peptides covering a
    residue; the legend says the coverage is of MS1 peptides."""
    from types import SimpleNamespace

    import numpy as np

    from hxex.gui.sequence_view import coverage_of

    binned = SimpleNamespace(
        spectrum_id=np.array(["a", "b", "c"]), sample=np.array(["WT", "WT", "WT"]),
        start_seq=np.array([3, 3, 3]), end_seq=np.array([12, 4, 5]),
        peptide_range=np.array(["0003-0012", "0003-0004", "0003-0005"]),
        data_id=np.array([0, 1, 2]),
        modification=np.array(["", "ion_type:c", "ion_type:z"]))
    results = SimpleNamespace(binned=binned, fit=SimpleNamespace(spectrum_id=binned.spectrum_id),
                              review_status=np.array(["", "", ""]), settings=None)
    assert coverage_of(results) == {"WT": [(3, 12, "0003-0012")]}
    rows, annotations, _ = project_sequences(_project(tmp_path, "/3-12"))
    text = sequence_html(rows, annotations, coverage=coverage_of(results))
    assert "MS1 peptides covering a residue (1 to 1; fragment ions not counted)" in text



def test_the_sequence_is_drawn_in_a_real_fixed_width_font(qapp, tmp_path):
    """``monospace`` is a font alias on Linux only: on Windows Qt gave Tahoma and nothing
    aligned (2026-11-16). The view names an installed fixed-width family instead, and draws
    the coverage with glyphs that family draws at a letter's width."""
    from PySide6.QtGui import QFontInfo

    from hxex.gui import fonts
    from hxex.gui.sequence_view import BAR, BAR_SAFE, BLOCKS, BLOCKS_SAFE

    font = fonts.fixed_font()
    assert font.family() == fonts.fixed_family() and font.fixedPitch()
    assert QFontInfo(font).fixedPitch() or fonts.fixed_family() not in fonts.FAMILIES
    marks = '[[analysis.regions]]\nname = "loop"\nstart = 5\nend = 7\n'
    rows, annotations, _ = project_sequences(_project(tmp_path, "/3-12", marks))
    text = sequence_html(rows, annotations, family=fonts.fixed_family(),
                         blocks=BLOCKS_SAFE, bar=BAR_SAFE)
    assert f"font-family: {fonts.fixed_family()}, monospace" in text
    assert BAR_SAFE in text and BAR not in text  # the region's bar: the safe glyph
    assert BAR in sequence_html(rows, annotations)  # the default: the full set
    assert fonts.even_width(font, "AB") and len(BLOCKS_SAFE) == 4 and len(BLOCKS) == 8


GROUPS = {"A_alone": "MKPLAAGDKE", "A_in_AB": "MKPLAHGDKE", "A_pH5": "MKPLAAGDKW",
          "B_alone": "GSTWVLLPRS", "B_in_AB": "GSTWVQLPRS", "C_extra": "MKPLAAGDKE"}


def _groups_project(tmp_path) -> object:
    """Three groups (owner, 2026-11-18): A with its own regions and sites, B taking the shared
    ones, and a pH group sharing A_alone with A; C_extra is in no group."""
    (tmp_path / "SpecExport").mkdir()
    for name, seq in GROUPS.items():
        (tmp_path / f"{name}.fasta").write_text(f">{name}\n{seq}\n", encoding="utf-8")
    fasta = ", ".join(f'{n} = "{n}.fasta"' for n in GROUPS)
    path = tmp_path / "hxex.toml"
    path.write_text(
        'project = "seq"\n\n[input]\nformat = "specexport"\ndata_dir = "SpecExport"\n\n'
        f"[analysis]\nfasta = {{ {fasta} }}\n\n"
        '[[analysis.regions]]\nname = "shared"\nstart = 2\nend = 4\n\n'
        '[[analysis.sites]]\nresidue = 9\nlabel = "S9"\n\n'
        '[analysis.groups.A]\nreference = "A_alone"\n'
        'samples = { "A+B" = "A_in_AB", alone = "A_alone" }\n\n'
        '[[analysis.groups.A.regions]]\nname = "loopA"\nstart = 5\nend = 7\n\n'
        '[[analysis.groups.A.sites]]\nresidue = 6\nlabel = "H6"\n\n'
        '[analysis.groups.B]\nreference = "B_alone"\nsamples = ["B_alone", "B_in_AB"]\n\n'
        '[analysis.groups.pH]\nreference = "A_pH5"\nsamples = ["A_alone", "A_pH5"]\n',
        encoding="utf-8")
    return path


def _columns(line: str, label: int, char: str) -> list[int]:
    """The residue numbers (a line from residue 1) at which ``char`` stands in a line of
    :func:`sequence_html`, its markup taken out: the name column, two spaces, then a cell per
    residue with a space after every ten."""
    import re

    text = re.sub(r"<[^>]+>", "", line)[label + 2:]
    out, n = [], 0
    for i, ch in enumerate(text):
        if (i + 1) % 11 == 0:  # the space after a block of ten
            continue
        n += 1
        if ch == char:
            out.append(n)
    return out


def test_groups_show_their_samples_regions_and_sites(tmp_path):
    """With [analysis.groups] (owner, 2026-11-18): rows by group in the file's order, the
    reference first; differences within a group only; each group's regions under its rows and
    its sites in its rows only; a sample in two groups shown in each; one in no group left out
    with a note; a click names the group and its regions."""
    import re

    rows, marks, message = project_sequences(_groups_project(tmp_path))
    assert [(r.group, r.samples) for r in rows] == [
        ("A", ["A_alone"]), ("A", ["A_in_AB"]), ("B", ["B_alone"]), ("B", ["B_in_AB"]),
        ("pH", ["A_pH5"]), ("pH", ["A_alone"])]
    assert [r.differs for r in rows] == [set(), {6}, set(), {6}, set(), {10}]
    assert "C_extra is in no group of [analysis.groups]" in message
    assert [r.name for r in marks["A"].regions] == ["loopA"]  # its own replace the shared
    assert [r.name for r in marks["B"].regions] == ["shared"]  # none of its own: the shared

    text = sequence_html(rows, marks)
    assert "<b>A</b>: " in text and "loopA (5–7)" in text and "<b>B</b>: " in text
    lines = text.split("\n")
    label = len("A_alone")  # the longest name; the label column is at least 6 wide

    def row_line(name: str, after: int = 0) -> int:
        return next(i for i, line in enumerate(lines) if i >= after
                    and re.sub(r"<[^>]+>", "", line).startswith(f"{name:<{label}}  "))

    a_last, b_first = row_line("A_in_AB"), row_line("B_alone")
    a_bar = row_line("A", a_last + 1)
    assert a_last < a_bar < b_first  # A's regions under A's rows, before B's
    assert _columns(lines[a_bar], label, "━") == [5, 6, 7]
    b_bar = row_line("B", b_first + 1)
    assert _columns(lines[b_bar], label, "━") == [2, 3, 4]
    assert lines[a_bar + 1] == ""  # the gap between the groups
    ph_bar = row_line("pH", row_line("A_pH5"))
    assert _columns(lines[ph_bar], label, "━") == [2, 3, 4]

    def underlined(k: int) -> list[int]:
        return sorted(int(n) for n, style in re.findall(
            rf'res:{k}:(\d+)"[^>]*><span style="([^"]*)"', text) if "underline" in style)

    assert underlined(0) == underlined(1) == [6]  # group A's own site
    assert underlined(2) == underlined(3) == underlined(5) == [9]  # the shared site

    said = describe(rows, marks, 1, 6)
    assert said == ("A_in_AB (group A): residue 6, H; differs: A_alone A; region loopA (5–7); "
                    "site H6")
    assert "region" not in describe(rows, marks, 0, 3)  # A does not take the shared region
    assert "region shared (2–4)" in describe(rows, marks, 2, 3)
    assert describe(rows, marks, 5, 10).startswith("A_alone (group pH): residue 10, E; "
                                                   "differs: A_pH5 W")


def test_the_view_counts_the_groups_samples(qtbot, tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QUrl

    from hxex.gui.sequence_view import SequenceView

    view = SequenceView()
    qtbot.addWidget(view)
    view.show_project(_groups_project(tmp_path))
    about = view.about.text()
    assert about.startswith("5 sample(s) in 3 group(s), 5 distinct sequence(s). ")
    assert "C_extra is in no group" in about
    view._clicked(QUrl("res:3:6"))
    assert view.about.text().startswith("B_in_AB (group B): residue 6, Q; differs: B_alone L")
