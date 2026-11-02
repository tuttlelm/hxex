# SPDX-License-Identifier: GPL-3.0-or-later
"""The folder formats' inputs checked against the data: SpecExport folder names against their
FASTA (and its Pfam-style numbering), folders and FASTA files that do not pair up, empty reads,
duplicated table rows, and peptide tables that contradict the folder names."""

import logging

import pytest

from hxex import io as hio
from hxex.spectra import PeptideEntry


def sample(root, name, fasta, folders, header=None, files=("Non-1-z2.csv", "60s-1-z2.csv")):
    """A SpecExport sample: ``name.fasta`` and one folder per name in ``folders``."""
    (root / f"{name}.fasta").write_text(f">{header or name}\n{fasta}\n")
    for folder in folders:
        (root / name / folder).mkdir(parents=True)
        for f in files:
            (root / name / folder / f).write_text("300.1,5\n300.2,4\n")


# a four-residue tag before residue 1 of the sequence HDExaminer numbered
TAG = "GGGM" + "KVLAGHEDWRSTPQ"


def test_a_fasta_numbered_otherwise_than_the_folders_is_refused(tmp_path):
    sample(tmp_path, "WT", TAG, ["0002-0005-KVLA", "0006-0010-GHEDW"])
    with pytest.raises(hio.ReadError) as err:
        hio.scan(tmp_path, "specexport")
    text = str(err.value)
    assert "2 peptide folder(s) do not match" in text
    assert ("WT/0002-0005-KVLA: the folder names KVLA, WT.fasta gives GGMK at residues 2-5"
            in text)
    assert "differs from HDExaminer's" in text
    # every folder is found with the FASTA's first residue numbered -2: the header to write
    assert "making >WT/-2-15 the header line" in text
    # with that header, the folders' numbers and the FASTA's agree
    (tmp_path / "WT.fasta").write_text(f">WT/-2-15 tagged\n{TAG}\n")
    entries = hio.scan(tmp_path, "specexport")
    assert [(e.peptide, e.start_seq, e.end_seq) for e in entries] == [
        ("KVLA", 2, 5), ("GHEDW", 6, 10)]


def test_the_header_keeps_the_records_name_and_numbers_the_ends(tmp_path):
    """The terminal groups stay at the FASTA's ends under its own numbering; the suggested
    header keeps the record's name."""
    sample(tmp_path, "WT", "Ac-GMKVLA-NH2", ["0000-0002-MKV", "0002-0004-VLA"],
           header="construct_7/1-6 his-tagged")
    with pytest.raises(hio.ReadError, match=r"making >construct_7/-1-4 the header line"):
        hio.scan(tmp_path, "specexport")
    (tmp_path / "WT.fasta").write_text(">construct_7/-1-4\nAc-GMKVLA-NH2\n")
    assert [e.peptide for e in hio.scan(tmp_path, "specexport")] == ["MKV", "VLA-NH2"]
    (tmp_path / "WT" / "-001-0000-GM").mkdir()  # not a START-END folder name: skipped
    assert len(hio.scan(tmp_path, "specexport")) == 2


def test_read_fasta_numbered(tmp_path):
    p = tmp_path / "a.fasta"
    p.write_text(">a\nMKVLA\n")
    assert hio.read_fasta_numbered(p) == ("MKVLA", 1)
    p.write_text(">sp|P1|X/0-4 description\nMKVLA\n")
    assert hio.read_fasta_numbered(p) == ("MKVLA", 0)
    p.write_text(">a/-2-2\nMacKVLA-NH2\n")  # modX residues counted as residues
    assert hio.read_fasta_numbered(p) == ("MacKVLA-NH2", -2)
    p.write_text(">a/5-12\nMKVLA\n")
    with pytest.raises(hio.ReadError, match=r"a.fasta: the header numbers the residues 5 to 12 "
                                            r"\(8 residues\), but the sequence has 5"):
        hio.read_fasta_numbered(p)
    assert hio.read_fasta_sequence(p) == "MKVLA"  # the sequence alone takes no numbering


def test_a_mutants_substitutions_are_named_and_its_fasta_used(tmp_path, caplog):
    """HDExaminer names a mutant's folders from the reference sequence: a few residues that
    differ (a third of those covered at most) are the mutant's substitutions, not a refusal."""
    sample(tmp_path, "D7Q", "MKVLAGQEDW", ["0002-0005-KVLA", "0006-0010-GDEDW",
                                           "0007-0009-DED"])
    with caplog.at_level(logging.WARNING):
        entries = hio.scan(tmp_path, "specexport")
    assert {e.peptide for e in entries} == {"KVLA", "GQEDW", "QED"}
    assert ("D7Q.fasta differs from the names of 2 of its peptide folders at residue 7 (D in "
            "the folder names, Q in the FASTA), as a mutant's FASTA does") in caplog.text
    assert "or a FASTA that differs from the sequence HDExaminer searched against" in caplog.text


def test_a_difference_every_sample_shares_is_not_called_a_mutation(tmp_path, caplog):
    """Folder names that differ from every sample's FASTA at the same residues point at the
    FASTA files (or HDExaminer's sequence), not at a mutation: one warning for all samples; a
    sample's own substitution keeps its own warning, counting only its folders."""
    folders = ["0002-0005-KVLA", "0006-0010-GDEDW", "0007-0009-DED"]
    sample(tmp_path, "WT", "MKVLAGHEDW", folders)
    sample(tmp_path, "K3E", "MKELAGHEDW", folders)
    with caplog.at_level(logging.WARNING):
        hio.scan(tmp_path, "specexport")
    assert ("every sample's FASTA (K3E.fasta, WT.fasta) differs from the peptide folder names "
            "at residue 7 (D in the folder names, H in the FASTA)") in caplog.text
    assert ("K3E.fasta differs from the names of 1 of its peptide folders at residue 3 (V in "
            "the folder names, E in the FASTA), as a mutant's FASTA does") in caplog.text
    assert "WT.fasta differs" not in caplog.text


def test_another_construct_is_refused_as_such(tmp_path):
    sample(tmp_path, "WT", "MKVLAGHEDW", ["0002-0005-KVLA", "0006-0010-PSTRY"])
    with pytest.raises(hio.ReadError) as err:
        hio.scan(tmp_path, "specexport")
    text = str(err.value)
    assert text.startswith("1 peptide folder(s)")
    assert "WT/0006-0010-PSTRY: the folder names PSTRY, WT.fasta gives GHEDW" in text
    assert "another construct or a mutant" in text and "making" not in text


def test_a_refusal_names_ten_folders_and_counts_the_rest(tmp_path):
    folders = [f"{i:04d}-{i + 1:04d}-{TAG[i - 1 + 4:i + 1 + 4]}" for i in range(1, 13)]
    sample(tmp_path, "WT", TAG, folders)
    with pytest.raises(hio.ReadError) as err:
        hio.scan(tmp_path, "specexport")
    assert "12 peptide folder(s)" in str(err.value) and "and 2 more" in str(err.value)


def test_ranges_outside_the_fasta_are_refused_with_the_others(tmp_path):
    sample(tmp_path, "WT", "KVLAGHEDW", ["0005-0008-KVLA", "0011-0013-EDW"])
    with pytest.raises(hio.ReadError) as err:
        hio.scan(tmp_path, "specexport")
    text = str(err.value)
    assert "WT/0011-0013-EDW: residues 11-13 are outside WT.fasta (residues 1-9)" in text
    assert "WT/0005-0008-KVLA: the folder names KVLA, WT.fasta gives GHED" in text
    assert "making >WT/5-13 the header line" in text


def test_modified_residues_and_termini_are_compared_as_base_residues(tmp_path):
    sample(tmp_path, "WT", "Ac-MAacKLVGpS", ["0002-0004-AKL", "0001-0003-Ac-MAacK",
                                             "0005-0007-VGS"])
    assert {e.peptide for e in hio.scan(tmp_path, "specexport")} == {"AacKL", "Ac-MAacK", "VGpS"}


def test_a_folder_name_that_cannot_be_read_is_not_checked(tmp_path, caplog):
    """A folder name whose sequence cannot be read is warned about and not checked (the
    character is one Windows allows in a file name: ? is not)."""
    sample(tmp_path, "WT", "MKVLAGHEDW", ["0002-0005-KV#A"])
    with caplog.at_level(logging.WARNING):
        (entry,) = hio.scan(tmp_path, "specexport")[:1]
    assert entry.peptide == "KVLA"
    assert "0002-0005-KV#A: the sequence in the folder's name cannot be read" in caplog.text
    assert "not checked against WT.fasta" in caplog.text


def test_a_lower_case_fasta_is_read_in_capitals(tmp_path, caplog):
    sample(tmp_path, "WT", "mkvla\nghedw", ["0002-0005-KVLA"])
    with caplog.at_level(logging.INFO):
        assert hio.scan(tmp_path, "specexport")[0].peptide == "KVLA"
    assert "WT.fasta: the sequence is in lower case; read in capitals" in caplog.text
    assert hio.read_fasta_sequence(tmp_path / "WT.fasta") == "MKVLAGHEDW"


def test_a_fasta_cut_by_letters_says_why(tmp_path, caplog):
    sample(tmp_path, "WT", "MKVXAGHEDW", ["0006-0010-GHEDW"])
    with caplog.at_level(logging.WARNING):
        assert hio.scan(tmp_path, "specexport")[0].peptide == "GHEDW"
    assert "WT.fasta: unknown residue 'X'" in caplog.text
    assert "cut by its letters" in caplog.text


# --- folders and FASTA files that do not pair up ----------------------------------------------

def test_folders_and_fasta_files_that_do_not_pair_up_are_named(tmp_path, caplog):
    sample(tmp_path, "WT", "MKVLA", ["0001-0003-MKV"])
    (tmp_path / "mutant" / "0001-0003-MKV").mkdir(parents=True)  # no FASTA
    (tmp_path / "D2" / "0001-0003-MKV").mkdir(parents=True)
    (tmp_path / "d2.fasta").write_text(">d2\nMKVLA\n")  # case differs
    (tmp_path / "D3" / "0001-0003-MKV").mkdir(parents=True)
    (tmp_path / "D3" / "D3.fasta").write_text(">D3\nMKVLA\n")  # inside its folder
    (tmp_path / "spare.fasta").write_text(">spare\nMKVLA\n")  # no folder
    (tmp_path / "analysis").mkdir()  # holds no peptide folders: not a sample
    with caplog.at_level(logging.WARNING):
        entries = hio.scan(tmp_path, "specexport")
    assert {e.sample for e in entries} == {"WT"}
    text = caplog.text
    assert "sample folder mutant has no mutant.fasta beside it, so it is not read" in text
    assert "names are case-sensitive: d2.fasta and folder D2" in text
    assert "d2.fasta has no folder" not in text  # said once, beside the folder
    assert "D3.fasta: a sample's FASTA file must sit beside its folder" in text
    assert "spare.fasta has no folder spare beside it, so it is not used" in text
    assert "analysis" not in text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        hio.scan(tmp_path, "specexport", samples=["WT", "d2"])
    assert "names are case-sensitive: d2.fasta and folder D2" in caplog.text
    assert "mutant" not in caplog.text and "spare" not in caplog.text
    assert "samples: d2 is not a sample" in caplog.text


def test_a_sample_folder_given_as_data_dir_is_pointed_out(tmp_path, caplog):
    sample(tmp_path, "WT", "MKVLA", ["0001-0003-MKV"])
    with caplog.at_level(logging.WARNING):
        assert hio.scan(tmp_path / "WT", "specexport") == []
    assert "holds peptide folders (START-END-SEQUENCE) itself: this looks like a sample folder " \
           'or a pooled folder: give the folder above it as data_dir, or format = "pooled"' \
           in caplog.text


# --- empty reads ------------------------------------------------------------------------------

def test_an_empty_read_is_refused_with_its_cause(tmp_path):
    sample(tmp_path, "WT", "MKVLAGHEDW", ["0002-0005-KVLA"])
    entries = hio.scan(tmp_path, "specexport")
    read = dict(project="p")
    with pytest.raises(hio.ReadError, match=r"nothing to read: no samples \(a SAMPLE.fasta"):
        hio.read_spectra([], tmp_path, "specexport", **read)
    with pytest.raises(hio.ReadError, match=r"samples = \['wt'\] matches none of the samples "
                                            r"found \('WT': the FASTA files with their folders; "
                                            r"names are case-sensitive\)"):
        hio.read_spectra(entries, tmp_path, "specexport", samples=["wt"], **read)
    with pytest.raises(hio.ReadError, match=r"peptide_ranges = \['1-4'\] matches none of the 1 "
                                            r"peptides found \(residues 2 to 5\)"):
        hio.read_spectra(entries, tmp_path, "specexport", peptide_ranges=["1-4"], **read)
    with pytest.raises(hio.ReadError, match=r"peptide_region = \[6, 9\] matches none"):
        hio.spectrum_index(entries, tmp_path, "specexport", project="p",
                           peptide_region=[6, 9])
    # a recorded read that finds nothing is refused the same way
    source = hio.read_spectra(entries, tmp_path, "specexport", **read).source
    with pytest.raises(hio.ReadError, match="peptide_ranges = \\['1-4'\\] matches none"):
        hio.read_spectra_source({**source, "peptide_ranges": ["1-4"]})


def test_an_hxexpress_sample_without_files_is_named(tmp_path, caplog):
    (tmp_path / "WT_0001-0005-ABCDE-z2-x.csv").write_text("undeut,\n1,2\n")
    with caplog.at_level(logging.WARNING):
        entries = hio.scan_hxexpress(tmp_path, samples=["wt", "WT"])
    assert len(entries) == 1
    assert "samples: no file of sample wt" in caplog.text
    assert "the samples found, the start of the file names, are WT; names are case-sensitive" \
           in caplog.text
    with pytest.raises(hio.ReadError, match="the start of the file names"):
        hio.read_spectra(entries, tmp_path, "hxexpress", project="p", samples=["wt"])


# --- peptide tables ---------------------------------------------------------------------------

HEADER = "file,sample,peptide,start_seq,end_seq,charge,modification\n"


def test_a_folder_table_row_listed_twice_is_refused(tmp_path):
    p = tmp_path / "files.csv"
    row = "0002-0005-KVLA,WT,KVLA,2,5,2,\n"
    p.write_text(HEADER + row + "0002-0005-KVLA,WT,KVLA,2,5,3,\n" + row)
    with pytest.raises(hio.ReadError, match=r"files.csv line 4: listed again \(the same file, "
                                            r"sample, peptide, residues, charge and modification "
                                            r"as line 2\); delete one of the two lines"):
        hio.read_peptide_table(p)
    # a row that differs only in its modification is a series of its own
    p.write_text(HEADER + row + "0002-0005-KVLA,WT,KVLA,2,5,2,Hex:1\n")
    assert len(hio.read_peptide_table(p)) == 2
    # the scan formats' tables are checked by the project (listed again for a sample)
    p.write_text("peptide,start_seq,end_seq,charge\nKVLA,2,5,2\nKVLA,2,5,2\n")
    assert len(hio.read_peptide_table(p, required=hio.entries.SCAN_REQUIRED)) == 2


def test_a_table_that_contradicts_the_folder_names_is_named(tmp_path, caplog):
    sample(tmp_path, "WT", "MKVLAGHEDW", ["0002-0005-KVLA"],
           files=("Non-1-z2.csv", "Non-1-z3.csv"))
    entries = [PeptideEntry("0002-0005-KVLA", "WT", "KVLE", "0002-0005", 2, 5, 2),
               PeptideEntry("0002-0005-KVLA", "WT", "KVLE", "0002-0005", 2, 5, 3),
               PeptideEntry("0002-0005-KVLA", "WT", "Ac-KVacKA", "0003-0006", 3, 6, 2),
               PeptideEntry("0002-0005-KVLA", "WT", "KV?A", "0002-0005", 2, 5, 2)]
    assert hio.folder_name_disagreements(entries, "specexport") == [
        "WT/0002-0005-KVLA: the peptide table's sequence KVLE differs from the folder's name: "
        "the table's sequence is used",
        "WT/0002-0005-KVLA: the peptide table's residues 3-6 differ from the folder's name "
        "(2-5): the table's residues are used",
        "WT/0002-0005-KVLA: the peptide table's sequence Ac-KVacKA differs from the folder's "
        "name: the table's sequence is used",
    ]
    assert hio.folder_name_disagreements(entries, "hxexpress") == []
    pooled = [PeptideEntry("0002-0005-KVacKA", "", "KVKA", "0002-0005", 2, 5, 2)]
    assert hio.folder_name_disagreements(pooled, "pooled") == []  # base residues agree
    with caplog.at_level(logging.WARNING):
        hio.read_spectra(entries[:1], tmp_path, "specexport", project="p")
    assert "the peptide table's sequence KVLE differs from the folder's name" in caplog.text
    # entries found by scanning always agree
    assert hio.folder_name_disagreements(hio.scan(tmp_path, "specexport"), "specexport") == []


def test_a_missing_charge_names_the_charges_found(tmp_path):
    sample(tmp_path, "WT", "MKVLAGHEDW", ["0002-0005-KVLA"],
           files=("Non-1-z2.csv", "60s-1-z4.csv"))
    entry = PeptideEntry("0002-0005-KVLA", "WT", "KVLA", "0002-0005", 2, 5, 3)
    folder = tmp_path / "WT" / "0002-0005-KVLA"
    with pytest.raises(hio.ReadError, match=r"no spectrum files for charge 3 \(files found "
                                            r"for charges 2, 4\)"):
        hio.read_specexport_dir(folder, entry)
    for f in folder.iterdir():
        f.unlink()
    with pytest.raises(hio.ReadError, match=r"\(no spectrum files of any charge\)"):
        hio.read_specexport_dir(folder, entry)
