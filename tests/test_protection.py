# SPDX-License-Identifier: GPL-3.0-or-later
"""MS1 protection factors: intrinsic rates in the protein, the per-time and fitted factors,
bounds, and the differences between samples, on uptake made from known factors."""

import logging

import numpy as np
import pytest

from hxex import chem
from hxex.analysis import SpeciesUptake
from hxex.analysis.protection import compare_protection, peptide_protection, peptide_rates
from hxex.kint import ExchangeConditions, intrinsic_rates

PROTEIN = "MAKLVGSTVEDRWQAFKLTH"
LABELLING = ExchangeConditions(7.0, 20.0, 0.9)
TIMES = (10.0, 100.0, 1000.0, 10000.0)


def rates_in_protein(start, end, nterm=1):
    """The rates of the counted amides of PROTEIN[start..end], from the whole protein."""
    full = intrinsic_rates(PROTEIN, LABELLING)
    return full.rate[full.at(range(start + nterm, end + 1))]


def uptake(entries, *, nterm=1, times=TIMES, se=0.01, corrected=True):
    """A species table from (sample, start, end, fD at each time); one row per time."""
    cols = {name: [] for name in ("sample", "peptide", "modification", "kind", "start", "end",
                                  "time", "K", "rank", "mean", "mean_se", "fD", "fD_se",
                                  "population", "population_se", "n_amides", "n_spectra",
                                  "n_charges", "corrected", "charge_trend")}
    for sample, start, end, fds in entries:
        peptide = PROTEIN[start - 1:end]
        n = len(peptide) - nterm
        for t, fd in zip(times, fds, strict=True):
            values = {"sample": sample, "peptide": peptide, "modification": "",
                      "kind": "peptide", "start": start, "end": end, "time": t, "K": 1,
                      "rank": 1, "mean": fd * n, "mean_se": se * n, "fD": fd, "fD_se": se,
                      "population": 1.0, "population_se": 0.0, "n_amides": n, "n_spectra": 3,
                      "n_charges": 1, "corrected": corrected, "charge_trend": False}
            for name, value in values.items():
                cols[name].append(value)
    return SpeciesUptake(**{k: np.array(v) for k, v in cols.items()}, nterm_subtract=nterm)


def model(k, protection, times=TIMES):
    k = np.asarray(k)[None, :] / np.asarray(protection)
    return np.mean(1 - np.exp(-np.asarray(times)[:, None] * k), axis=1)


def test_rates_are_those_of_the_protein():
    """A peptide's rates: the residue before it and no termini inside the protein."""
    for start, end in ((4, 12), (12, 20)):
        k, approx = peptide_rates(PROTEIN[start - 1:end], start, LABELLING,
                                  before=PROTEIN[start - 2], protein_end=end == len(PROTEIN))
        assert np.allclose(k, rates_in_protein(start, end)) and not approx.any()
    # without the residue before, the first amide has no rate (counted with nterm_subtract 0)
    k, _ = peptide_rates(PROTEIN[3:12], 4, LABELLING, nterm_subtract=0)
    assert np.isnan(k[0]) and np.allclose(k[1:], rates_in_protein(4, 12))
    # the protein's N-terminus: a counted free amine exchanges at once
    k, _ = peptide_rates(PROTEIN[:8], 1, LABELLING, nterm_subtract=0)
    assert np.isinf(k[0]) and np.allclose(k[1:], rates_in_protein(1, 8))
    # the peptide's own termini would be wrong: its last residue is not a carboxyl here
    alone = intrinsic_rates(PROTEIN[3:12], LABELLING, start=4)
    assert not np.isclose(alone.rate[-1], rates_in_protein(4, 12)[-1])


def test_protection_factors_are_recovered():
    k1, k2 = rates_in_protein(2, 9), rates_in_protein(10, 20)
    up = uptake([("A", 2, 9, model(k1, 10.0)), ("A", 10, 20, model(k2, 1000.0))])
    table, by_time = peptide_protection(up, LABELLING, sequences=PROTEIN)
    assert table.log10_protection.tolist() == pytest.approx([1.0, 3.0], abs=1e-4)
    assert np.allclose(table.protection, [10.0, 1000.0], rtol=1e-3)
    assert (table.bound == "").all() and table.one_factor_fits.all()
    assert np.all(table.log10_protection_se < 0.05)
    # drift needs two times between the limits: protection 10 has one at these times
    assert np.isnan(table.protection_drift[0]) and abs(table.protection_drift[1]) < 1e-3
    # each time on its own gives the same factor where fD is between the limits
    inside = by_time.bound == ""
    first = by_time.where(start=2)
    assert np.allclose(by_time.log10_protection[first & inside], 1.0, atol=1e-4)
    assert (by_time.bound[first] != "").any()  # 10000 s: fully exchanged at protection 10
    # without the protein sequence: the residue before from the other peptides, same values
    # (the second peptide ends at the protein's C-terminus, which only the sequence tells)
    other, _ = peptide_protection(up, {"A": LABELLING})
    assert other.log10_protection[0] == pytest.approx(1.0, abs=1e-4)
    assert abs(other.log10_protection[1] - 3.0) > 0.05


def test_unequal_amides_make_the_factor_drift():
    """Half the amides exposed, half buried: the apparent factor rises with time and one factor
    does not fit."""
    k = rates_in_protein(2, 13)
    half = len(k) // 2
    fd = (model(k[:half], 1.0) * half + model(k[half:], 1e4) * (len(k) - half)) / len(k)
    table, _ = peptide_protection(uptake([("A", 2, 13, fd)]), LABELLING, sequences=PROTEIN)
    assert table.protection_drift[0] > 1.0 and not table.one_factor_fits[0]
    assert table.misfit[0] > 10


def test_bounds_instead_of_values():
    up = uptake([("A", 2, 9, [0.98, 0.99, 1.0, 1.0]), ("A", 10, 20, [0.0, 0.01, 0.02, 0.03])])
    table, _ = peptide_protection(up, LABELLING, sequences=PROTEIN)
    assert table.bound.tolist() == ["upper", "lower"]
    assert np.isnan(table.log10_protection).all() and np.isfinite(table.limit).all()
    # the limits are the factors that give the thresholds at the most telling time
    k1, k2 = rates_in_protein(2, 9), rates_in_protein(10, 20)
    assert model(k1, table.limit[0], times=(10.0,))[0] == pytest.approx(0.95, abs=1e-4)
    assert model(k2, table.limit[1], times=(10000.0,))[0] == pytest.approx(0.05, abs=1e-4)


def test_samples_left_out_say_why(caplog):
    k = rates_in_protein(2, 9)
    up = uptake([("A", 2, 9, model(k, 10.0))])
    raw = uptake([("B", 2, 9, model(k, 10.0))], corrected=False)
    both = SpeciesUptake(**{name: np.concatenate([a, b]) for (name, a), b in zip(
        up.columns().items(), raw.columns().values(), strict=True)}, nterm_subtract=1)
    with caplog.at_level(logging.WARNING):
        table, _ = peptide_protection(both, {"A": LABELLING, "B": LABELLING}, sequences=PROTEIN)
    assert set(table.sample) == {"A"} and "not corrected" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        table, _ = peptide_protection(up, {"other": LABELLING}, sequences=PROTEIN)
    assert len(table) == 0 and "no labelling conditions" in caplog.text


def test_differences_between_samples():
    k1, k2 = rates_in_protein(2, 9), rates_in_protein(10, 20)
    up = uptake([("WT", 2, 9, model(k1, 10.0)), ("WT", 10, 20, model(k2, 100.0)),
                 ("M", 2, 9, model(k1, 1000.0)), ("M", 10, 20, [1.0, 1.0, 1.0, 1.0])])
    table, _ = peptide_protection(up, LABELLING, sequences=PROTEIN)
    cmp = compare_protection(table, "WT")
    assert cmp.sample.tolist() == ["M", "M"]
    assert cmp.delta[0] == pytest.approx(2.0, abs=1e-3) and cmp.significant[0]
    assert np.isnan(cmp.delta[1]) and cmp.note[1] == "sample upper bound"
    with pytest.raises(ValueError, match="match"):
        compare_protection(table, "WT", match="other")


def test_modified_residue_counts_as_approximate():
    peptide = "AKLVGpSTVE"  # residues 2-11 with S7 phosphorylated
    k, approx = peptide_rates(peptide, 2, LABELLING, before="M")
    assert approx.sum() == 2 and np.isfinite(k).all()
    assert len(k) == len(chem.parse_sequence(peptide)) - 1


def test_protection_figures_and_settings(tmp_path):
    from hxex.analysis import AnalysisConfig
    from hxex.config import load_project
    from hxex.report import protection_difference_figure, protection_figure

    k1, k2 = rates_in_protein(2, 9), rates_in_protein(10, 20)
    up = uptake([("WT", 2, 9, model(k1, 10.0)), ("WT", 10, 20, model(k2, 100.0)),
                 ("M", 2, 9, model(k1, 1000.0)), ("M", 10, 20, [1.0, 1.0, 1.0, 1.0])])
    table, _ = peptide_protection(up, LABELLING, sequences=PROTEIN)
    fig = protection_figure(table, "M", length=20)
    (ax,) = [a for a in fig.axes if a.get_title()]
    assert ax.get_title() == "M, average (K=1)"
    hatched = [p.get_hatch() for p in ax.patches]
    assert sorted(hatched, key=str) == ["////", None]
    fig = protection_difference_figure(compare_protection(table, "WT"), "M")
    assert [a.get_title() for a in fig.axes if a.get_title()] == ["M - WT, average (K=1)"]
    with pytest.raises(ValueError, match="no protection"):
        protection_figure(table, "other")

    # settings: opt-in, refused without the labelling conditions
    assert not AnalysisConfig().protection
    with pytest.raises(ValueError, match="labelling"):
        AnalysisConfig(protection=True)
    cfg = AnalysisConfig(protection=True, labelling={"ph_read": 7.0, "temperature": 20.0,
                                                     "d2o_fraction": 0.9})
    assert cfg.labelling == LABELLING
    again = AnalysisConfig.from_dict(cfg.to_dict())
    assert again.labelling == LABELLING and again.protection
    project = tmp_path / "hxex.toml"
    project.write_text('project = "p"\n[input]\nformat = "specexport"\ndata_dir = "d"\n'
                       '[analysis]\nprotection = true\n'
                       '[analysis.labelling.WT]\nph_read = 7.0\ntemperature = 20.0\n'
                       'd2o_fraction = 0.9\n')
    loaded = load_project(project).analysis
    assert loaded.protection and loaded.labelling == {"WT": LABELLING}
