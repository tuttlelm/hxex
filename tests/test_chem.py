# SPDX-License-Identifier: GPL-3.0-or-later
"""Chemistry tests: golden values from hxex_updating.py, pyteomics, and PTM handling."""

import json
from pathlib import Path

import numpy as np
import pytest
from pyteomics import mass

from hxex import chem

GOLDEN = json.loads((Path(__file__).parent / "golden" / "chem_reference.json").read_text())

# Documented deliberate differences (docs/HISTORY.md, "Sequence syntax and PTMs", item 1): the
# reference slices by characters, so a modified residue among the first residues shifts the
# N-terminal proline window when nterm_subtract = 2.
AMIDE_DIFFERENCES = {("acKPEPTIDE", 2): 5, ("pTPEpYPEPK", 2): 4}


@pytest.mark.parametrize("row", GOLDEN["amides"], ids=lambda r: f"{r['seq']}-n{r['nterm']}")
def test_amides_match_reference(row):
    expected = AMIDE_DIFFERENCES.get((row["seq"], row["nterm"]), row["n"])
    assert chem.count_amides(row["seq"], nterm_subtract=row["nterm"]) == expected


def test_amide_differences_are_exactly_the_documented_ones():
    differing = {
        (r["seq"], r["nterm"])
        for r in GOLDEN["amides"]
        if chem.count_amides(r["seq"], nterm_subtract=r["nterm"]) != r["n"]
    }
    assert differing == set(AMIDE_DIFFERENCES)


@pytest.mark.parametrize("row", GOLDEN["mz"], ids=lambda r: f"{r['seq']}-z{r['z']}")
def test_mz_matches_reference(row):
    assert chem.mz(row["seq"], row["z"]) == pytest.approx(row["mz"], abs=1e-9)


@pytest.mark.parametrize("row", GOLDEN["shape"], ids=lambda r: r["seq"])
def test_shape_matches_reference(row):
    shape = chem.natural_abundance(row["seq"])
    assert len(shape) == len(row["shape"])
    assert shape.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(shape, np.array(row["shape"]) / sum(row["shape"]), atol=1e-12)


# --- pyteomics -----------------------------------------------------------------------------

@pytest.mark.parametrize("ion_type", chem.ion_types())
@pytest.mark.parametrize("charge", [1, 2, 3])
def test_mz_matches_pyteomics_for_every_ion_type(ion_type, charge):
    seq = "RDKVQKEYAL"
    assert chem.mz(seq, charge, ion_type) == pytest.approx(
        mass.calculate_mass(sequence=seq, ion_type=ion_type, charge=charge), abs=1e-9
    )
    assert chem.neutral_mass(seq, ion_type) == pytest.approx(
        mass.calculate_mass(sequence=seq, ion_type=ion_type), abs=1e-9
    )


@pytest.mark.parametrize("ion_type", chem.ion_types())
def test_composition_matches_pyteomics(ion_type):
    expected = mass.Composition(sequence="PEPTIDEK", ion_type=ion_type)
    assert chem.composition("PEPTIDEK", ion_type) == {k: v for k, v in expected.items() if v}


@pytest.mark.parametrize("ion_type", chem.ion_types())
def test_shape_follows_ion_composition(ion_type):
    """Independent check: build the shape directly from pyteomics' composition."""
    from brainpy import isotopic_variants

    comp = dict(mass.Composition(sequence="RDKVQKEYAL", ion_type=ion_type))
    comp["H"] -= chem.count_amides("RDKVQKEYAL", ion_type=ion_type)
    expected = np.array([p.intensity for p in isotopic_variants(comp, charge=1)])
    np.testing.assert_allclose(
        chem.natural_abundance("RDKVQKEYAL", ion_type), expected / expected.sum(), atol=1e-12
    )


def test_ion_type_changes_shape_slightly():
    m, a = chem.natural_abundance("RDKVQKEYAL"), chem.natural_abundance("RDKVQKEYAL", "a")
    centroid = lambda s: float(np.arange(len(s)) @ s)  # noqa: E731
    assert 0 < abs(centroid(m) - centroid(a)) < 0.1


def test_unknown_ion_type():
    with pytest.raises(chem.IonTypeError):
        chem.mz("PEPTIDE", 2, "q")
    with pytest.raises(chem.IonTypeError):
        chem.count_amides("PEPTIDE", ion_type="q")


# --- PTMs and tokenisation ------------------------------------------------------------------

def test_tokenisation():
    assert chem.parse_sequence("PEPacKTIDE") == ("P", "E", "P", "acK", "T", "I", "D", "E")
    assert chem.parse_sequence("pSpTpYK") == ("pS", "pT", "pY", "K")


@pytest.mark.parametrize("bad", ["", "PEP TIDE", "PEPzzKTIDE", "pEPTIDE", "PEPTIDE1", "peptide",
                                 "PEPTIDE-NH3", "Xx-PEPTIDE", "A-B-C-D", "PEPTIDE-", "-NH2"])
def test_bad_sequences(bad):
    with pytest.raises(chem.SequenceError):
        chem.parse_sequence(bad)


def test_terminal_groups():
    assert chem.split_termini("RPKPQQFFGLM-NH2") == ("H-", "RPKPQQFFGLM", "-NH2")
    assert chem.split_termini("Ac-PEacKTIDE") == ("Ac-", "PEacKTIDE", "-OH")
    assert chem.split_termini("Ac-PEPTIDE-NH2") == ("Ac-", "PEPTIDE", "-NH2")
    assert chem.split_termini("PEPTIDE") == ("H-", "PEPTIDE", "-OH")
    assert chem.join_termini("H-", ("F", "acK"), "-NH2") == "FacK-NH2"
    assert chem.join_termini("Ac-", ("F",), "-OH") == "Ac-F"
    assert chem.parse_sequence("Ac-PEacK-NH2") == ("P", "E", "acK")
    # terminal groups are not residues: the amide count is unchanged
    assert chem.count_amides("RPKPQQFFGLM-NH2") == chem.count_amides("RPKPQQFFGLM")


@pytest.mark.parametrize("ion_type", ["M", "b", "c-1", "y", "z-dot", "x"])
def test_c_terminal_amide_matches_pyteomics(ion_type):
    seq = "RPKPQQFFGLM-NH2"  # substance P
    # the group given to pyteomics explicitly (older pyteomics cannot read it from the sequence)
    table = {**mass.std_aa_comp, "-NH2": mass.Composition(formula="NH2")}
    parsed = ["H-", *"RPKPQQFFGLM", "-NH2"]
    assert chem.mz(seq, 2, ion_type) == pytest.approx(mass.calculate_mass(
        parsed_sequence=parsed, ion_type=ion_type, charge=2, aa_comp=table), abs=1e-6)
    # -OH replaced by -NH2: one O fewer, one N and one H more
    d = {k: chem.composition(seq, ion_type).get(k, 0) - chem.composition(seq[:-4], ion_type)
         .get(k, 0) for k in "CHNO"}
    assert d == {"C": 0, "H": 1, "N": 1, "O": -1}


def test_n_terminal_acetyl():
    # pyteomics reads "Ac" as the element actinium, so compare with the composition instead
    d = {k: chem.composition("Ac-PEPTIDE", "b").get(k, 0) - chem.composition("PEPTIDE", "b")
         .get(k, 0) for k in "CHNO"}
    assert d == {"C": 2, "H": 2, "N": 0, "O": 1}
    assert chem.mz("Ac-PEPTIDE", 1, "b") - chem.mz("PEPTIDE", 1, "b") == pytest.approx(
        42.010565, abs=1e-5)


@pytest.mark.parametrize(
    "mod_seq, base_seq, delta",
    [
        ("PEPacKTIDE", "PEPKTIDE", {"C": 2, "H": 2, "O": 1}),
        ("PEPpSTIDE", "PEPSTIDE", {"H": 1, "P": 1, "O": 3}),
        ("PEPpTIDE".replace("pT", "pTT"), "PEPTTIDE", {"H": 1, "P": 1, "O": 3}),
        ("PEPpYTIDE", "PEPYTIDE", {"H": 1, "P": 1, "O": 3}),
    ],
)
def test_ptm_composition(mod_seq, base_seq, delta):
    got, base = chem.composition(mod_seq), chem.composition(base_seq)
    diff = {k: got.get(k, 0) - base.get(k, 0) for k in set(got) | set(base)}
    assert {k: v for k, v in diff.items() if v} == delta


def test_ptm_does_not_change_amide_count_by_default():
    assert chem.count_amides("PEPacKTIDE") == chem.count_amides("PEPKTIDE")


def test_modified_residue_counts_as_one_residue_for_nterm_and_proline_rules():
    # 8 residues, nterm_subtract 2 removes acK and P; the remaining P at index 2 is a proline.
    assert chem.count_amides("acKPEPTIDE", nterm_subtract=2) == 8 - 2 - 1
    assert chem.count_amides("PEPacKTIDE") == 8 - 1 - 1


def test_side_chain_term_uses_base_residue():
    assert chem.count_amides("acKA", count_sc=1.0) == chem.count_amides("KA", count_sc=1.0)
    assert chem.count_amides("pSA", count_sc=1.0) == chem.count_amides("SA", count_sc=1.0)


def test_no_global_pyteomics_state():
    chem.composition("PEPacKTIDE")
    chem.mz("pSPEPTIDE", 2)
    assert not {"acK", "pS", "pT", "pY"} & set(mass.std_aa_comp)


def test_registry_is_extensible_and_immutable():
    reg = chem.DEFAULT_MODIFICATIONS.with_modification("meK", "K", "CH2")
    assert "meK" in reg and "meK" not in chem.DEFAULT_MODIFICATIONS
    got = chem.composition("PEPmeKTIDE", registry=reg)
    base = chem.composition("PEPKTIDE")
    assert got["C"] - base["C"] == 1 and got["H"] - base["H"] == 2
    with pytest.raises(chem.SequenceError):
        chem.composition("PEPmeKTIDE")
    with pytest.raises(chem.ChemistryError):
        reg.with_modification("meK", "K", "CH2")
    with pytest.raises(chem.ChemistryError):
        reg.with_modification("meS", "K", "CH2")


# --- modification strings -------------------------------------------------------------------

def test_parse_modification():
    spec = chem.parse_modification("H:1 Hex:2 ion_type:c undeut_mz:523.25")
    assert dict(spec.composition) == {"H": 1}
    assert (spec.hex, spec.ion_type, spec.undeut_mz) == (2, "c", 523.25)
    assert chem.parse_modification(None) == chem.parse_modification("")
    for bad in ["H", "H:x", "ion_type:q", "H:"]:
        with pytest.raises(chem.ChemistryError):
            chem.parse_modification(bad)


def test_modification_composition_is_additive_in_mz():
    """pyHXExpress adds the modification's mass plus charge protons again (an extra ~1.007);
    here m/z is the m/z of the modified ion. Deliberate difference (docs/PLAN.md)."""
    mod = chem.parse_modification("H:1")
    expected = (mass.Composition("PEPTIDEK") + mass.Composition({"H": 1})).mass(charge=2)
    assert chem.mz("PEPTIDEK", 2, mod=mod) == pytest.approx(expected, abs=1e-9)


def test_modification_composition_is_additive_in_shape():
    """pyHXExpress replaces H by the modification's H count in the shape (a bug); here the
    change is added. Deliberate difference (docs/PLAN.md)."""
    plain = chem.natural_abundance("PEPTIDEK")
    modded = chem.natural_abundance("PEPTIDEK", mod=chem.parse_modification("H:1"))
    assert len(modded) == len(plain)
    assert 0 < np.abs(modded - plain).max() < 0.01


def test_hex_removes_extra_exchangeable_hydrogens():
    plain = chem.natural_abundance("PEPTIDEK")
    hexed = chem.natural_abundance("PEPTIDEK", mod=chem.parse_modification("Hex:1"))
    comp = chem.composition("PEPTIDEK")
    comp["H"] -= chem.count_amides("PEPTIDEK") + 1
    from brainpy import isotopic_variants

    ref = np.array([p.intensity for p in isotopic_variants(comp, charge=1)])
    np.testing.assert_allclose(hexed, ref / ref.sum(), atol=1e-12)
    assert not np.allclose(hexed[: len(plain)], plain[: len(hexed)])


def test_undeut_mz_override():
    assert chem.mz("PEPTIDE", 2, mod=chem.parse_modification("undeut_mz:400.5")) == 400.5


def test_bad_charge():
    with pytest.raises(chem.ChemistryError):
        chem.mz("PEPTIDE", 0)


def test_exchangeable_sites_adds_hex():
    spec = chem.parse_modification("Hex:3 H:1")
    assert chem.exchangeable_sites("PEPTIDEK") == chem.count_amides("PEPTIDEK")
    assert chem.exchangeable_sites("PEPTIDEK", mod=spec) == chem.count_amides("PEPTIDEK") + 3
    assert chem.exchangeable_sites("acKPEPTIDE", mod=spec, nterm_subtract=2) == 5 + 3


def test_amide_counts_by_ion_type():
    """AEPTIDEK (proline 3; with nterm_subtract 1, residues 2, 4-8 of the peptide): a c ion also
    holds the next residue's amide, x and y their first residue's (a C-terminal ion's first
    residue is not the parent's), z not (see test_fragments.py for every ion type against the
    compositions). PEPTIDEK starts with a proline, which has no amide H."""
    assert chem.count_amides("AEPTIDEK") == 6
    for ion_type in chem.ion_types():
        want = {"c": 7, "x": 7, "y": 7}.get(ion_type[0], 6)
        assert chem.count_amides("AEPTIDEK", ion_type=ion_type) == want, ion_type
        want = {"c": 7}.get(ion_type[0], 6)
        assert chem.count_amides("PEPTIDEK", ion_type=ion_type) == want, ion_type


def test_amide_rules_set_the_boundary_for_one_ion_type():
    rules = chem.AmideRules.from_dict({"c": False, "z": True})
    assert chem.count_amides("AEPTIDEK", ion_type="c", rules=rules) == 6
    assert chem.count_amides("AEPTIDEK", ion_type="z", rules=rules) == 7
    assert chem.count_amides("AEPTIDEK", ion_type="c-1", rules=rules) == 7  # its own default
    assert chem.count_amides("AEPTIDEK", rules=rules) == 6


def test_amide_rules_flow_into_sites_and_shape():
    rules = chem.AmideRules({"c": False})
    spec = chem.parse_modification("Hex:1")
    assert chem.exchangeable_sites("PEPTIDEK", mod=spec, ion_type="c") == 7 + 1
    assert chem.exchangeable_sites("PEPTIDEK", mod=spec, ion_type="c", rules=rules) == 6 + 1
    default = chem.natural_abundance("PEPTIDEK", "c")
    other = chem.natural_abundance("PEPTIDEK", "c", rules=rules)
    assert not np.allclose(default, other[: len(default)])
    # the fragment's place in its parent travels in the modification string
    spec = chem.parse_modification("ion_type:y parent_pos:2")
    assert spec.parent_pos == 2 and chem.exchangeable_sites(
        "EPTIDEK", mod=spec, ion_type="y", nterm_subtract=2) == 5
    with pytest.raises(chem.ChemistryError):
        chem.parse_modification("parent_pos:0")


def test_amide_rules_validation_and_serialisation():
    rules = chem.AmideRules().with_boundary("c", False)
    assert chem.AmideRules.from_dict(rules.to_dict()) == rules
    assert chem.DEFAULT_AMIDE_RULES.to_dict() == {}
    with pytest.raises(chem.IonTypeError):
        chem.AmideRules({"q": True})
    with pytest.raises(chem.ChemistryError):
        chem.AmideRules({"c": 1})  # the integer offsets of earlier versions
    with pytest.raises(chem.IonTypeError):
        chem.count_amides("PEPTIDE", ion_type="q")


def test_polluted_pyteomics_table_is_ignored(monkeypatch):
    """pyHXExpress adds acK etc. to pyteomics' global table; hxex must not depend on that."""
    polluted = dict(mass.std_aa_comp)
    polluted["acK"] = polluted["K"] + mass.Composition("C2H2O")
    polluted["xxK"] = polluted["K"]
    monkeypatch.setattr(mass, "std_aa_comp", polluted)
    reg = chem.ModificationRegistry().with_modification("acK", "K", "C2H2O")  # no clash
    assert "xxK" not in reg.aa_comp and "acK" in reg.aa_comp
    assert chem.composition("PEPacKTIDE", registry=reg) == chem.composition("PEPacKTIDE")
    with pytest.raises(chem.SequenceError):
        chem.parse_sequence("PEPxxKTIDE", reg)
    with pytest.raises(chem.SequenceError):
        chem.parse_sequence("PEPacKTIDE", chem.ModificationRegistry())
