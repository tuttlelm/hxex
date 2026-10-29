# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path

import numpy as np
import pytest

from hxex import kint
from hxex.kint import ExchangeConditions, expected_uptake, intrinsic_rates

GOLDEN = Path(__file__).parent / "golden" / "kint_reference.json"


def test_rates_match_the_reference_code():
    """pacilab/hdx-rates-mixtures, its rounding stripped (tests/golden/make_kint_golden.py)."""
    data = json.loads(GOLDEN.read_text())
    assert not data["rounded"] and len(data["cases"]) > 100
    for case in data["cases"]:
        cond = ExchangeConditions(case["ph_read"], case["temperature_k"] - 273.15,
                                  case["d2o_fraction"], case["reference"])
        rates = intrinsic_rates(case["sequence"], cond)
        for name in ("kforw", "kback"):
            ref = np.array(case[name])
            mine = getattr(rates, name)
            amide = np.isfinite(mine)
            # the reference gives 0 where there is no amide hydrogen (residue 1, prolines)
            assert np.all(ref[~amide] == 0)
            assert np.allclose(mine[amide], ref[amide], rtol=1e-9, atol=0), case
        assert not np.isfinite(rates.kforw[0])
        assert not np.any(np.isfinite(rates.kforw[rates.aa == "P"]))


def test_pure_solvents_and_the_mixture_equilibrium():
    seq = "GAVLEDHKS"
    assert np.all(intrinsic_rates(seq, ExchangeConditions(7, 20, 1.0)).kback[1:] == 0)
    assert np.all(intrinsic_rates(seq, ExchangeConditions(7, 20, 0.0)).kforw[1:] == 0)
    r = intrinsic_rates(seq, ExchangeConditions(2.5, 0, 0.1))
    # an amide favours D slightly (fractionation): its equilibrium is a little above x
    assert np.all((r.equilibrium[1:] > 0.1) & (r.equilibrium[1:] < 0.13))
    assert np.allclose(r.rate, r.kforw + r.kback, equal_nan=True)
    # slowest near pH 2.5 to 3, faster when warm
    alanine = [intrinsic_rates("AAAAA", ExchangeConditions(ph, 0, 0.1)).rate[2]
               for ph in (1.5, 2.8, 5.0)]
    assert alanine[1] < alanine[0] and alanine[1] < alanine[2]
    cold, warm = (intrinsic_rates("AAAAA", ExchangeConditions(2.5, t, 0.1)).rate[2]
                  for t in (0, 25))
    assert warm > 5 * cold
    oligo = intrinsic_rates(seq, ExchangeConditions(7, 20, 0.9, "3Ala"))
    assert np.all(oligo.rate[1:] > intrinsic_rates(seq, ExchangeConditions(7, 20, 0.9)).rate[1:])


def test_modified_residues_and_termini():
    cond = ExchangeConditions(7, 20, 0.9)
    plain = intrinsic_rates("PEPSTIDE", cond, start=11)
    phos = intrinsic_rates("PEPpSTIDE", cond, start=11)
    assert phos.residue.tolist() == list(range(11, 19)) and phos.aa[3] == "pS"
    # no published factors: the standard residue's, marked on the amide and the next one
    assert np.allclose(phos.rate, plain.rate, equal_nan=True)
    assert phos.approximate.tolist() == [False] * 3 + [True, True] + [False] * 3
    assert not plain.approximate.any()
    own = intrinsic_rates("PEPpSTIDE", ExchangeConditions(7, 20, 0.9, factors={
        "pS": (-0.9, -0.12, 0.69, 0.6)}))
    assert not own.approximate.any() and own.rate[3] != plain.rate[3]
    # an acetylated N-terminus gives residue 1 an amide and takes residue 2's N-terminal factor
    free = intrinsic_rates("AAAAA", cond)
    acetyl = intrinsic_rates("Ac-AAAAA", cond)
    assert np.isnan(free.rate[0]) and np.isfinite(acetyl.rate[0])
    assert acetyl.rate[1] < free.rate[1] / 10  # the free amine speeds up residue 2 (base)
    # a C-terminal amide has no carboxyl factor; its last residue is approximate
    amide = intrinsic_rates("AAAAA-NH2", cond)
    assert amide.rate[-1] > free.rate[-1] and amide.approximate.tolist()[-1]
    assert np.allclose(amide.rate[1:-1], free.rate[1:-1])
    # both terminal factors on a dipeptide's one amide (the reference applies only the first)
    dipeptide = intrinsic_rates("AA", cond).rate[1]
    assert dipeptide != pytest.approx(free.rate[1])
    assert dipeptide == pytest.approx(intrinsic_rates("AA-NH2", cond).rate[1]
                                      * free.rate[-1] / amide.rate[-1], rel=0.5)


def test_terminal_groups_take_the_published_factors():
    """Golden case from the published factors (audit 2026-10-28 question 23): Bai et al. 1993
    Table II for the free N-terminus (AR -1.32, BR 1.62 on residue 2); the Englander lab's
    rate spreadsheet (HXrates, row "H3CCO-") for the acetyl group (AR 0.293, BR -0.197 on
    residue 1, rounded here to 0.29 and -0.20; it is in no paper); no factor for a C-terminal
    amide (the published C-terminal factor is the carboxyl's) and no N-terminal factor after
    an acetyl group (Bai's factor was measured against an acetyl-blocked dipeptide). For
    residues without a charged side chain each rate is alpha 10^a + beta 10^b, a and b the
    summed acid and base factors, alpha and beta set by the conditions alone: solved here from
    a free peptide's residues 2 and 3, then used to predict the others."""
    cond = ExchangeConditions(5.0, 20, 0.9)
    free = intrinsic_rates("AAAA", cond)
    acetyl = intrinsic_rates("Ac-AAAA-NH2", cond)
    gly = intrinsic_rates("Ac-GG-NH2", cond)
    for rate in ("kforw", "kback"):
        k = getattr(free, rate)
        alpha, beta = np.linalg.solve([[10**-1.32, 10**1.62], [1.0, 1.0]], [k[1], k[2]])

        def predict(a, b, alpha=alpha, beta=beta):
            return alpha * 10**a + beta * 10**b

        got = getattr(acetyl, rate)
        assert got[0] == pytest.approx(predict(0.29, -0.20), rel=1e-9)  # the acetyl's factors
        assert got[1] == pytest.approx(predict(0.0, 0.0), rel=1e-9)  # no N-terminal factor
        assert got[3] == pytest.approx(predict(0.0, 0.0), rel=1e-9)  # no carboxyl factor
        got = getattr(gly, rate)
        assert got[0] == pytest.approx(predict(-0.22 + 0.29, -0.03 - 0.20), rel=1e-9)
        assert got[1] == pytest.approx(predict(-0.22 + 0.22, -0.03 + 0.17), rel=1e-9)
    assert acetyl.approximate.tolist() == [False, False, False, True]  # -NH2: the last


def test_expected_uptake():
    cond = ExchangeConditions(2.5, 0, 0.05)
    rates = intrinsic_rates("PEPTIDEKAL", cond, start=5)
    residues = [8, 9, 10]  # T, I, D (5 and 7 are prolines)
    assert expected_uptake(rates, residues, 0.0, 0.9) == pytest.approx(2.7)
    late = expected_uptake(rates, residues, 1e9, 0.9)
    assert late == pytest.approx(rates.equilibrium[3:6].sum())
    t = np.array([0.0, 60.0, 600.0])
    trace = expected_uptake(rates, residues, t, [0.9, 0.9, 0.0])
    assert trace.shape == (3,) and trace[0] == pytest.approx(1.8)
    k = rates.rate[3:6]
    eq = rates.equilibrium[3:6]
    want = np.sum(eq + (np.array([0.9, 0.9, 0.0]) - eq) * np.exp(-np.outer(t, k)), axis=1)
    assert np.allclose(trace, want)
    with pytest.raises(ValueError, match="no amide"):
        expected_uptake(rates, [7], 0.0, 0.9)  # residue 7 is a proline
    with pytest.raises(ValueError, match="not all in"):
        expected_uptake(rates, [20], 0.0, 0.9)


def test_conditions():
    cond = ExchangeConditions(2.5, 0, 0.05, "3Ala", factors={"acK": [0, 0, 0, 0]})
    assert ExchangeConditions.from_dict(cond.to_dict()) == cond
    for bad in ({"ph_read": 15}, {"temperature": 293.0}, {"d2o_fraction": 90},
                {"reference": "poly"}, {"factors": {"pS": (1, 2)}}):
        with pytest.raises(ValueError):
            ExchangeConditions(**{"ph_read": 7, "temperature": 20, "d2o_fraction": 0.9, **bad})
    with pytest.raises(ValueError, match="unknown"):
        ExchangeConditions.from_dict({"ph_read": 7, "temperature": 20, "d2o_fraction": 1,
                                      "pD": 7.4})
    assert kint.REFERENCES == ("PDLA", "3Ala")


def test_conditions_per_sample():
    from hxex.kint import conditions_dict, read_conditions, sample_conditions

    one = {"ph_read": 2.8, "temperature": 20, "d2o_fraction": 0.05, "initial_fraction": 0.9}
    assert read_conditions(one) == ExchangeConditions(2.8, 20, 0.05, initial_fraction=0.9)
    per = read_conditions({"WT": one, "V": {**one, "ph_read": 3.0}})
    assert sample_conditions(per, "V").ph_read == 3.0 and sample_conditions(per, "X") is None
    assert read_conditions(conditions_dict(per)) == per
    # a run's own conditions under its name (a table of per-sample tables)
    nested = read_conditions({"WT": one, "A": {"WT": {**one, "ph_read": 3.5}}})
    assert sample_conditions(nested, "WT", "A").ph_read == 3.5
    assert sample_conditions(nested, "WT", "B").ph_read == 2.8
    assert sample_conditions(nested, "WT").ph_read == 2.8
    assert read_conditions(conditions_dict(nested)) == nested
    with pytest.raises(ValueError, match="one table per sample"):
        read_conditions({"A": {"WT": {"X": one}}})
    assert sample_conditions(per["WT"], "anything") == per["WT"]
    with pytest.raises(ValueError, match="initial_fraction"):
        ExchangeConditions(2.8, 20, 0.05, initial_fraction=90)
    with pytest.raises(ValueError, match="must be a table"):
        read_conditions({"WT": 2.8})
