# SPDX-License-Identifier: GPL-3.0-or-later
"""MS1 protection factors: intrinsic rates in the protein, the per-time and fitted factors,
bounds, and the differences between samples, on uptake made from known factors."""

import dataclasses
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


def uptake(entries, *, nterm=1, times=TIMES, se=0.01, corrected=True, fallback=False):
    """A species table from (sample, start, end, fD at each time); one row per time."""
    cols = {name: [] for name in ("sample", "peptide", "modification", "kind", "start", "end",
                                  "time", "K", "rank", "mean", "mean_se", "fD", "fD_se",
                                  "population", "population_se", "n_amides", "n_spectra",
                                  "n_charges", "corrected", "charge_trend", "control_fallback")}
    for sample, start, end, fds in entries:
        peptide = PROTEIN[start - 1:end]
        n = len(peptide) - nterm
        for t, fd in zip(times, fds, strict=True):
            values = {"sample": sample, "peptide": peptide, "modification": "",
                      "kind": "peptide", "start": start, "end": end, "time": t, "K": 1,
                      "rank": 1, "mean": fd * n, "mean_se": se * n, "fD": fd, "fD_se": se,
                      "population": 1.0, "population_se": 0.0, "n_amides": n, "n_spectra": 3,
                      "n_charges": 1, "corrected": corrected, "charge_trend": False,
                      "control_fallback": fallback}
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


def test_unknown_errors_take_the_median_of_the_known_ones():
    """A NaN standard error (no bootstrap) must not get the largest weight: it takes the
    median of the peptide's known errors; with none known the fit is unweighted."""
    from hxex.analysis.protection import _known_errors

    assert _known_errors(np.array([np.nan, np.nan])) is None
    assert _known_errors(np.array([0.02, np.nan, 0.04, 0.0])).tolist() == [0.02, 0.03, 0.04, 0.03]
    k = rates_in_protein(2, 9)
    up = uptake([("A", 2, 9, model(k, 10.0) + np.array([0.0, 0.02, -0.02, 0.0]))])
    known, _ = peptide_protection(up, LABELLING, sequences=PROTEIN)
    half = dataclasses.replace(up, fD_se=np.where(up.time == 100.0, np.nan, up.fD_se))
    table, _ = peptide_protection(half, LABELLING, sequences=PROTEIN)
    assert np.isfinite(table.misfit[0]) and table.misfit[0] == pytest.approx(known.misfit[0])
    assert table.log10_protection[0] == pytest.approx(known.log10_protection[0], abs=1e-6)
    none = dataclasses.replace(up, fD_se=np.full(len(up), np.nan))
    table, _ = peptide_protection(none, LABELLING, sequences=PROTEIN)
    assert np.isnan(table.misfit[0]) and np.isfinite(table.log10_protection_se[0])


def test_a_peptide_without_a_finite_fd_has_no_times():
    k = rates_in_protein(2, 9)
    up = uptake([("A", 2, 9, model(k, 10.0)), ("A", 10, 20, [np.nan] * 4)])
    table, _ = peptide_protection(up, LABELLING, sequences=PROTEIN)
    assert table.n_times[table.where(start=10)].tolist() == [0]
    assert np.isnan(table.log10_protection[table.where(start=10)]).all()


def test_two_reference_forms_over_one_range_do_not_collapse():
    k1 = rates_in_protein(2, 9)
    up = uptake([("WT", 2, 9, model(k1, 10.0)), ("M", 2, 9, model(k1, 100.0))])
    mod = uptake([("WT", 2, 9, model(k1, 1000.0)), ("M", 2, 9, model(k1, 1000.0))])
    mod = dataclasses.replace(mod, peptide=np.array([p.replace("S", "pS", 1) for p in
                                                     mod.peptide]))
    both = SpeciesUptake(**{name: np.concatenate([x, y]) for (name, x), y in zip(
        up.columns().items(), mod.columns().values(), strict=True)}, nterm_subtract=1)
    table, _ = peptide_protection(both, LABELLING, sequences=PROTEIN)
    cmp = compare_protection(table, "WT")  # range matching: identical sequences pair
    assert len(cmp) == 2 and sorted(cmp.peptide.tolist()) == sorted(cmp.peptide_reference)
    deltas = dict(zip(cmp.peptide.tolist(), cmp.delta.tolist(), strict=True))
    assert deltas[PROTEIN[1:9]] == pytest.approx(1.0, abs=1e-3)
    assert deltas[PROTEIN[1:9].replace("S", "pS", 1)] == pytest.approx(0.0, abs=1e-3)
    assert len(compare_protection(table, "WT", match="sequence")) == 2


def test_a_series_against_its_direction_is_refused():
    """Falling fD with forward conditions, or rising fD with back-exchange ones, is refused."""
    k = rates_in_protein(2, 9)
    up = uptake([("A", 2, 9, model(k, 10.0)[::-1]), ("A", 10, 20, [0.9, 0.6, 0.3, 0.1])])
    with pytest.raises(ValueError, match="falls with time.*set initial_fraction"):
        peptide_protection(up, LABELLING, sequences=PROTEIN)
    mixed = uptake([("A", 2, 9, model(k, 10.0)), ("A", 10, 20, [0.9, 0.6, 0.3, 0.1])])
    peptide_protection(mixed, LABELLING, sequences=PROTEIN)  # not most of them: accepted
    rising = uptake([("A", 2, 9, model(k, 10.0)), ("A", 10, 20, [0.1, 0.3, 0.6, 0.9])])
    with pytest.raises(ValueError, match="rises with time.*leave initial_fraction out"):
        peptide_protection(rising, DILUTION, sequences=PROTEIN)
    # a sample without conditions is not judged (it is left out)
    peptide_protection(up, {"other": LABELLING}, sequences=PROTEIN)


DILUTION = ExchangeConditions(7.0, 20.0, 0.1, initial_fraction=0.9)  # 90 % D2O, diluted 9-fold
BACK_TIMES = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)


def back_model(start, end, protection, f0, *, scale=0.9, times=BACK_TIMES):
    """fD of PROTEIN[start..end] deuterated to ``f0`` and exchanging in DILUTION; ``scale``:
    the fully deuterated reference's D fraction (None: the absolute scale)."""
    full = intrinsic_rates(PROTEIN, DILUTION)
    at = full.at(range(start + 1, end + 1))
    k, eq = full.rate[at], full.equilibrium[at]
    e = eq / scale if scale else eq
    decay = np.exp(-np.asarray(times)[:, None] * k[None, :] / protection)
    return np.mean(e + (f0 - e) * decay, axis=1), float(np.mean(e))


def test_back_exchange_from_a_measured_fully_deuterated_control():
    """Corrected with a measured control: fD starts at 1 and relaxes to the solvent's
    equilibrium over the labelling's D fraction."""
    fd1, e1 = back_model(2, 9, 1e3, 1.0)
    fd2, _ = back_model(10, 20, 1e4, 1.0)
    up = uptake([("A", 2, 9, fd1), ("A", 10, 20, fd2)], times=BACK_TIMES)
    table, by_time = peptide_protection(up, DILUTION, sequences=PROTEIN)
    assert table.log10_protection.tolist() == pytest.approx([3.0, 4.0], abs=1e-3)
    assert (table.direction == "back").all() and table.corrected.all()
    assert table.fD0.tolist() == [1.0, 1.0] and np.isnan(table.fD0_se).all()
    assert table.equilibrium[0] == pytest.approx(e1) and 0.1 < e1 < 0.15
    assert table.one_factor_fits.all()
    inside = by_time.bound == ""
    assert np.allclose(by_time.log10_protection[inside & (by_time.peptide == PROTEIN[1:9])],
                       3.0, atol=1e-3)
    # forward rows say so
    k = rates_in_protein(2, 9)
    fwd, _ = peptide_protection(uptake([("A", 2, 9, model(k, 10.0))]), LABELLING,
                                sequences=PROTEIN)
    assert fwd.direction.tolist() == ["forward"] and fwd.fD0.tolist() == [0.0]
    assert fwd.equilibrium.tolist() == [1.0]


def test_back_exchange_start_is_fitted_without_a_measured_control():
    fd, _ = back_model(10, 20, 100.0, 0.8)
    up = uptake([("A", 10, 20, fd)], times=BACK_TIMES, fallback=True)
    table, by_time = peptide_protection(up, DILUTION, sequences=PROTEIN)
    assert table.log10_protection[0] == pytest.approx(2.0, abs=1e-3)
    assert table.fD0[0] == pytest.approx(0.8, abs=1e-3) and 0 < table.fD0_se[0] < 0.2
    assert by_time.progress[0] > 0 and by_time.progress[-1] == pytest.approx(
        (0.8 - fd[-1]) / (0.8 - table.equilibrium[0]))
    # with noise, the errors cover the truth
    rng = np.random.default_rng(3)
    noisy = uptake([("A", 10, 20, fd + rng.normal(0, 0.01, len(fd)))], times=BACK_TIMES,
                   fallback=True)
    t2, _ = peptide_protection(noisy, DILUTION, sequences=PROTEIN)
    assert abs(t2.log10_protection[0] - 2.0) < 3 * t2.log10_protection_se[0]
    assert abs(t2.fD0[0] - 0.8) < 3 * t2.fD0_se[0]
    # a fully deuterated control from the other charge states counts as measured (HISTORY 130)
    fd1, _ = back_model(10, 20, 100.0, 1.0)
    up = uptake([("A", 10, 20, fd1)], times=BACK_TIMES, fallback=True)
    borrowed = dataclasses.replace(up, fd_other_charge=np.ones(len(up), dtype=bool))
    table, _ = peptide_protection(borrowed, DILUTION, sequences=PROTEIN)
    assert table.fD0.tolist() == [1.0] and np.isnan(table.fD0_se[0])
    assert table.log10_protection[0] == pytest.approx(2.0, abs=1e-3)


def test_uncorrected_back_exchange_is_fitted_on_the_absolute_scale(caplog):
    fd, e = back_model(10, 20, 100.0, 0.6, scale=None)
    up = uptake([("A", 10, 20, fd)], times=BACK_TIMES, corrected=False)
    with caplog.at_level(logging.INFO):
        table, _ = peptide_protection(up, DILUTION, sequences=PROTEIN)
    assert "absolute scale" in caplog.text
    assert table.log10_protection[0] == pytest.approx(2.0, abs=1e-3)
    assert table.fD0[0] == pytest.approx(0.6, abs=1e-3) and not table.corrected[0]
    assert table.equilibrium[0] == pytest.approx(e)
    # uncorrected forward labelling is still left out
    k = rates_in_protein(2, 9)
    out, _ = peptide_protection(uptake([("A", 2, 9, model(k, 10.0))], corrected=False),
                                LABELLING, sequences=PROTEIN)
    assert len(out) == 0


def test_back_exchange_bounds_and_too_few_times(caplog):
    done, e = back_model(10, 20, 1e-3, 0.9, times=BACK_TIMES)  # over before the first time
    slow, _ = back_model(2, 9, 1e9, 0.9, times=BACK_TIMES)  # nothing exchanged
    up = uptake([("A", 10, 20, done), ("A", 2, 9, slow + np.linspace(0, -0.001, 7))],
                times=BACK_TIMES, fallback=True)
    table, _ = peptide_protection(up, DILUTION, sequences=PROTEIN)
    bound = dict(zip(table.start.tolist(), table.bound.tolist(), strict=True))
    assert bound == {10: "upper", 2: "lower"}
    assert np.isnan(table.log10_protection).all()
    fd0 = dict(zip(table.start.tolist(), table.fD0.tolist(), strict=True))
    assert np.isnan(fd0[10]) and fd0[2] == pytest.approx(0.9, abs=0.01)
    fd, _ = back_model(10, 20, 100.0, 0.8, times=(30.0, 300.0))
    two = uptake([("A", 10, 20, fd)], times=(30.0, 300.0), fallback=True)
    with caplog.at_level(logging.WARNING):
        table, _ = peptide_protection(two, DILUTION, sequences=PROTEIN)
    assert np.isnan(table.log10_protection[0]) and "fewer than 3 times" in caplog.text


def test_modified_residue_counts_as_approximate():
    peptide = "AKLVGpSTVE"  # residues 2-11 with S7 phosphorylated
    k, approx = peptide_rates(peptide, 2, LABELLING, before="M")
    assert approx.sum() == 2 and np.isfinite(k).all()
    assert len(k) == len(chem.parse_sequence(peptide)) - 1


def test_protection_figures_and_settings(tmp_path):
    from hxex.analysis import AnalysisConfig
    from hxex.config import ProjectError, load_project
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
    # the residue level from peptides was removed: its settings are refused with the reason
    with pytest.raises(TypeError):
        AnalysisConfig(protection_residues=True)
    project = tmp_path / "hxex.toml"
    project.write_text('project = "p"\n[input]\nformat = "specexport"\ndata_dir = "d"\n'
                       '[analysis]\nprotection = true\n'
                       '[analysis.labelling.WT]\nph_read = 7.0\ntemperature = 20.0\n'
                       'd2o_fraction = 0.9\n')
    loaded = load_project(project).analysis
    assert loaded.protection and loaded.labelling == {"WT": LABELLING}
    # per run of [analysis.runs]: [analysis.labelling.<run>.<sample>]
    project.write_text(project.read_text() + '[analysis.runs]\nA = "a.zip"\n'
                       '[analysis.labelling.A.WT]\nph_read = 7.4\ntemperature = 20.0\n'
                       'd2o_fraction = 0.9\n')
    loaded = load_project(project).analysis
    assert loaded.labelling["A"]["WT"].ph_read == 7.4
    _rewrite(project, '[analysis.runs]\nA = "a.zip"', '[analysis.runs]\nB = "a.zip"')
    with pytest.raises(ProjectError, match="'A' is not a run"):
        load_project(project)
    _rewrite(project, "protection = true\n", "protection = true\nprotection_residues = true\n")
    with pytest.raises(ProjectError, match="protection_residues: residue-level protection "
                       "factors from peptides were removed.*fragment ladders"):
        load_project(project)


def _rewrite(path, old, new):
    path.write_text(path.read_text().replace(old, new))
    return path
