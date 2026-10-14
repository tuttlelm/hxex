# SPDX-License-Identifier: GPL-3.0-or-later
"""Fragment-ion entries from a parent peptide: lists, residues as tokens, terminal groups,
charges, and the mzML targets."""

import logging

import pytest

from hxex import chem
from hxex.fragments import (
    FragmentSpec,
    amide_difference,
    fragment_amides,
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
    got = rows(fragment_entries(SUBP, "c-1:2,10 z-dot:4 M:ALL:3"))
    z4 = "ion_type:z-dot parent_pos:8"  # C-terminal ions record where they start
    assert got == [
        ("RP", "1-2", 1, "ion_type:c-1"), ("RPKPQQFFGL", "1-10", 1, "ion_type:c-1"),
        ("RP", "1-2", 2, "ion_type:c-1"), ("RPKPQQFFGL", "1-10", 2, "ion_type:c-1"),
        ("FGLM-NH2", "8-11", 1, z4), ("FGLM-NH2", "8-11", 2, z4),
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
    assert got == [("Ac-PEPacK", "21-24", 1, "ion_type:b"),
                   ("acKTIDE", "24-28", 1, "ion_type:y parent_pos:4"),
                   ("Ac-PEPacKTIDE", "21-28", 1, "ion_type:M"),
                   ("Ac-PEPacKTIDE", "21-28", 2, "ion_type:M")]


def test_lengths_and_charges():
    p1 = PeptideEntry("f", "S", "ACDEFGHIKLMN", "1-12", 1, 12, 3)
    every = fragment_entries(p1, "c")
    assert len(every) == 11 * 2  # lengths 1-11, charges 1 and 2 (the parent's minus one)
    assert {e.charge for e in fragment_entries(p1, "c", limit_charge=False)} == {1, 2, 3}
    assert [e.end_seq for e in fragment_entries(p1, "c:10-:1")] == [10, 11]
    assert [e.end_seq for e in fragment_entries(p1, "c:-3:1")] == [1, 2, 3]
    assert {e.charge for e in fragment_entries(p1, "c:2:4")} == {4}  # explicit charges as given
    one = PeptideEntry("f", "S", "PEPTIDE", "1-7", 1, 7, 1)
    assert fragment_entries(one, "c") == [] and len(fragment_entries(one, "M")) == 1


def test_too_long_and_parent_modifications(caplog):
    with caplog.at_level(logging.WARNING, logger="hxex.fragments"):
        got = fragment_entries(SUBP, "c:10,11,15:1")
    assert [e.end_seq for e in got] == [10] and "not shorter than the parent" in caplog.text
    for mod in ("H:1", "Hex:1", "undeut_mz:500.2", "ion_type:c"):
        with pytest.raises(ValueError, match="cannot be passed"):
            fragment_entries(PeptideEntry("f", "S", "PEPTIDE", "1-7", 1, 7, 2, mod), "c")
    fine = PeptideEntry("f", "S", "PEPTIDE", "1-7", 1, 7, 2, "ion_type:M")
    assert len(fragment_entries(fine, "c:3")) == 1


def test_no_c_or_z_ions_at_a_proline():
    """Electron-based dissociation does not cut the N-Calpha bond of proline: no c ion before
    one and no z ion starting with one; b and y ions there are made."""
    got = fragment_entries(SUBP, "c z y:7,9:1", limit_charge=False)
    assert {int(e.extra["fragment"].split(":")[1]) for e in got if e.extra["fragment"][0] == "c"} \
        == set(range(1, 11)) - {1, 3}  # RPKP: c1 and c3 come before a proline
    assert {e.extra["fragment"] for e in got if e.extra["fragment"][0] == "z"} == {
        f"z:{n}" for n in range(1, 11)} - {"z:8", "z:10"}  # z8 = PQQ..., z10 = PKP...
    assert {e.extra["fragment"] for e in got if e.extra["fragment"][0] == "y"} == {"y:7", "y:9"}


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


# side-chain nitrogens of the plain residues used below (pyteomics compositions)
_SIDE_N = {"K": 1, "R": 3, "H": 2, "N": 1, "Q": 1, "W": 1}


@pytest.mark.filterwarnings("ignore:The compositions of z")
@pytest.mark.parametrize("parent", ["GSHDLVFK", "RPKPQQFFGLM", "HHGHHGLLKVLR", "PAPWNQRPS"])
@pytest.mark.parametrize("nterm", [0, 1, 2])
def test_amides_are_the_backbone_nitrogens_of_the_composition(parent, nterm):
    """For every ion type and length: the amides of a fragment are its backbone nitrogens (from
    the pyteomics composition, less the side chains'), placed at the cleavage end, less prolines
    and the parent's first ``nterm`` residues. The count of a fragment entry, from its own
    sequence and modification, agrees."""
    from pyteomics import mass

    n = len(parent)
    entry = PeptideEntry("f", "S", parent, f"1-{n}", 1, n, 2)
    for ion in chem.ion_types():
        side = ion_terminus(ion)
        # neutral-loss variants follow their base ion (the lost NH3 taken to be a side chain's),
        # except c-NH3, which has lost its cleavage-site NH2 (as its composition says)
        neutral_loss = ("H2O" in ion or "NH3" in ion) and ion != "c-NH3"
        base = ion.split("-")[0] if neutral_loss else ion
        for length in [n] if side == "M" else range(1, n):
            part = parent[:length] if side != "C" else parent[n - length:]
            backbone = mass.Composition(sequence=part, ion_type=base)["N"] - sum(
                _SIDE_N.get(r, 0) for r in part)
            held = range(1, backbone + 1) if side != "C" else range(n - backbone + 1, n + 1)
            want = tuple(i for i in held if i > nterm and parent[i - 1] != "P")
            got = chem.amide_positions(parent, ion, None if side == "M" else length,
                                       nterm_subtract=nterm)
            assert got == want, (ion, length)
            if side == "M":
                assert got == chem.amide_positions(parent, nterm_subtract=nterm)
    for e in fragment_entries(entry, " ".join(chem.ion_types()), limit_charge=False):
        spec = chem.parse_modification(e.modification)
        assert chem.exchangeable_sites(e.peptide, mod=spec, ion_type=spec.ion_type,
                                       nterm_subtract=nterm) == len(fragment_amides(
            entry, e.extra["fragment"], nterm_subtract=nterm)), e.extra["fragment"]


def test_differences_map_to_residues():
    p = PeptideEntry("f", "S", "GSHDLVFK", "21-28", 21, 28, 3)  # residues 21 to 28

    def diff(a, b, **kw):
        return amide_difference(p, a, b, **kw)

    assert fragment_amides(p, "c:5") == (22, 23, 24, 25, 26)  # residues 2-6 (c_k: 2 to k+1)
    assert fragment_amides(p, "z-dot:3") == (27, 28)  # residues 7-8 (z: not its first)
    assert fragment_amides(p, "y:3") == (26, 27, 28)
    assert fragment_amides(p, "M") == tuple(range(22, 29))
    assert diff("c:6", "c:5") == (27,) and diff("c-1:2", "c-1:1") == (23,)
    assert diff("b:5", "b:4") == (25,) and diff("a:5", "a:4") == (25,)
    assert diff("z-dot:6", "z-dot:3") == (24, 25, 26)
    assert diff("y:4", "y:3") == (25,) and diff("y:4", "z-dot:4") == (25,)
    # the parent minus a c ion is its complementary z ion
    for k in range(1, 7):
        assert diff("M", f"c:{k}") == fragment_amides(p, f"z-dot:{8 - k}")
    assert diff("c:3", "c:1", nterm_subtract=2) == (23, 24)  # residue 2 does not count
    for bad in [("c:5", "z-dot:3"), ("c:5", "c:6"), ("z:3", "c:2")]:
        with pytest.raises(ValueError, match="not a set of residues"):
            diff(*bad)
    for label in ("c:8", "c:0", "q:2", "c"):
        with pytest.raises((ValueError, chem.ChemistryError)):
            fragment_amides(p, label)


def test_amide_rules_set_the_boundary_amide():
    rules = chem.AmideRules({"c-1": False})
    assert chem.count_amides("GSHDL", ion_type="c-1") == 5
    assert chem.count_amides("GSHDL", ion_type="c-1", rules=rules) == 4
    assert chem.count_amides("GSHDL", ion_type="c", rules=rules) == 5  # c itself unchanged
    assert chem.AmideRules.from_dict(rules.to_dict()) == rules
    assert rules.with_boundary("y-NH3", False).keeps_boundary("y-NH3") is False
    with pytest.raises(chem.ChemistryError):
        chem.AmideRules({"M": True})
    with pytest.raises(chem.ChemistryError):
        chem.AmideRules({"c": 1})
    # a C-terminal ion's count depends on where it starts only near the parent's N-terminus
    assert chem.count_amides("HHDIVIK", ion_type="y", nterm_subtract=2) == 7
    assert chem.count_amides("HHDIVIK", ion_type="y", nterm_subtract=2, parent_pos=2) == 6
    with pytest.raises(chem.ChemistryError, match="parent_pos"):
        chem.count_amides("AHH", ion_type="c", parent_pos=3)
