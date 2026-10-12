# SPDX-License-Identifier: GPL-3.0-or-later
"""Fragment-ion entries from a parent peptide: lists, residues as tokens, terminal groups,
charges, and the mzML targets."""

import logging

import pytest

from hxex import chem
from hxex.fragments import (
    FragmentSpec,
    fragment_entries,
    fragment_targets,
    ion_terminus,
    parse_fragments,
)
from hxex.io import MzmlTarget
from hxex.spectra import PeptideEntry

SUBP = PeptideEntry("f.mzML", "S", "RPKPQQFFGLM-NH2", "1-11", 1, 11, 3)


def rows(entries):
    return [(e.peptide, e.peptide_range, e.charge, e.modification) for e in entries]


def test_parse_fragments():
    specs = parse_fragments("c-1:ALL z-dot:2-4,9 y:1,2:1,2 M")
    assert specs == (FragmentSpec("c-1"), FragmentSpec("z-dot", "2-4,9"),
                     FragmentSpec("y", "1,2", "1,2"), FragmentSpec("M"))
    assert [str(s) for s in specs] == ["c-1", "z-dot:2-4,9", "y:1,2:1,2", "M"]
    assert parse_fragments(" ".join(str(s) for s in specs)) == specs
    for bad in ("", "q:1", "c:0", "c:5-2", "c:x", "c:1:0"):
        with pytest.raises(ValueError):
            parse_fragments(bad)
    assert [ion_terminus(i) for i in ("a", "b-NH3", "c-1", "x", "y-H2O", "z+1", "M-NH3")] == [
        "N", "N", "N", "C", "C", "C", "M"]


def test_n_and_c_terminal_fragments_keep_their_own_terminal_group():
    got = rows(fragment_entries(SUBP, "c-1:2,11 z-dot:4 M:ALL:3"))
    assert got == [
        ("RP", "1-2", 1, "ion_type:c-1"), ("RPKPQQFFGLM", "1-11", 1, "ion_type:c-1"),
        ("RP", "1-2", 2, "ion_type:c-1"), ("RPKPQQFFGLM", "1-11", 2, "ion_type:c-1"),
        ("FGLM-NH2", "8-11", 1, "ion_type:z-dot"), ("FGLM-NH2", "8-11", 2, "ion_type:z-dot"),
        ("RPKPQQFFGLM-NH2", "1-11", 3, "ion_type:M"),
    ]
    e = fragment_entries(SUBP, "z-dot:4:1")[0]
    assert dict(e.extra) == {"parent_peptide": "RPKPQQFFGLM-NH2", "parent_range": "1-11",
                             "parent_charge": "3", "fragment": "z-dot:4"}
    assert e.file == SUBP.file and e.sample == SUBP.sample
    # the m/z of a C-terminal fragment includes the amide
    assert chem.mz(e.peptide, 1, "z-dot") == pytest.approx(
        chem.mz("FGLM", 1, "z-dot") - 0.984016, abs=1e-5)


def test_residues_are_tokens_and_ranges_follow_the_parent():
    parent = PeptideEntry("f", "S", "Ac-PEPacKTIDE", "21-28", 21, 28, 2)
    got = rows(fragment_entries(parent, "b:4 y:5 M"))
    # b4 ends with the whole acK; y5 starts with it; N-terminal ions keep the acetyl
    assert got == [("Ac-PEPacK", "21-24", 1, "ion_type:b"), ("acKTIDE", "24-28", 1, "ion_type:y"),
                   ("Ac-PEPacKTIDE", "21-28", 1, "ion_type:M"),
                   ("Ac-PEPacKTIDE", "21-28", 2, "ion_type:M")]


def test_lengths_and_charges():
    p1 = PeptideEntry("f", "S", "ACDEFGHIKLMN", "1-12", 1, 12, 3)
    every = fragment_entries(p1, "c")
    assert len(every) == 12 * 2  # lengths 1-12, charges 1 and 2 (the parent's minus one)
    assert {e.charge for e in fragment_entries(p1, "c", limit_charge=False)} == {1, 2, 3}
    assert [e.end_seq for e in fragment_entries(p1, "c:10-:1")] == [10, 11, 12]
    assert [e.end_seq for e in fragment_entries(p1, "c:-3:1")] == [1, 2, 3]
    assert {e.charge for e in fragment_entries(p1, "c:2:4")} == {4}  # explicit charges as given
    one = PeptideEntry("f", "S", "PEPTIDE", "1-7", 1, 7, 1)
    assert fragment_entries(one, "c") == [] and len(fragment_entries(one, "M")) == 1


def test_too_long_and_parent_modifications(caplog):
    with caplog.at_level(logging.WARNING, logger="hxex.fragments"):
        got = fragment_entries(SUBP, "c:10,12,15:1")
    assert [e.end_seq for e in got] == [10] and "longer than the parent" in caplog.text
    for mod in ("H:1", "Hex:1", "undeut_mz:500.2", "ion_type:c"):
        with pytest.raises(ValueError, match="cannot be passed"):
            fragment_entries(PeptideEntry("f", "S", "PEPTIDE", "1-7", 1, 7, 2, mod), "c")
    fine = PeptideEntry("f", "S", "PEPTIDE", "1-7", 1, 7, 2, "ion_type:M")
    assert len(fragment_entries(fine, "c:2")) == 1


def test_fragment_targets():
    targets = fragment_targets(SUBP, "c-1:2 z-dot:4:1", activation="ETD", filter_excludes="sid")
    assert all(isinstance(t, MzmlTarget) for t in targets)
    assert [(t.ms_level, t.activation, t.precursor_charge, t.filter_excludes) for t in targets] \
        == [(2, "ETD", 3, "sid")] * 3
    assert [t.entry.peptide for t in targets] == ["RP", "RP", "FGLM-NH2"]
    # a file that records a wrong precursor charge: select by the other conditions only
    anyz = fragment_targets(SUBP, "c-1:2:1", activation="EThcD", precursor_charge=None)
    assert anyz[0].precursor_charge is None
    assert fragment_targets(SUBP, "c-1:2:1", precursor_charge=2)[0].precursor_charge == 2
    with pytest.raises(ValueError, match="precursor_charge"):
        fragment_targets(SUBP, "c-1:2:1", precursor_charge="any")
    assert MzmlTarget.from_dict(targets[2].to_dict()) == targets[2]
    assert dict(MzmlTarget.from_dict(targets[2].to_dict()).entry.extra)["fragment"] == "z-dot:4"
