# SPDX-License-Identifier: GPL-3.0-or-later
"""The population test (hxex.population_test): the p-value of a null's moments, the steps, the
width check, the test of each spectrum and its place in the run and the bundle."""

import dataclasses

import numpy as np
import pytest
from scipy.stats import binom, chi2

from hxex.fit import FitConfig, Measurements, fit_models
from hxex.fit.result import KFit
from hxex.io import open_bundle
from hxex.model import Backend
from hxex.noise import NoiseConfig
from hxex.population_test import (
    CHI_SQUARE,
    SCREENED,
    SIMULATED,
    PopulationTest,
    PopulationTestConfig,
    gamma_tail,
    narrow_components,
    population_test,
    steps_K,
)
from hxex.run import RunConfig, resume_bootstrap, run

CPU = Backend("cpu", "float64")
FIT = FitConfig(k_max=2, adam_steps=300, n_restarts=4, backend=CPU,
                noise=NoiseConfig("gaussian", sd=0.01, percent=None))
TEST = PopulationTestConfig(simulations=8)
N_MAX = 20


def spectrum(populations, noise=0.01, seed=0):
    """A deuterium distribution on 0..N_MAX: a mixture of binomials ((share, n, p), ...) with
    Gaussian noise, the natural envelope a single peak."""
    x = np.arange(N_MAX + 1, dtype=float)
    y = sum(a * binom.pmf(x, n, p) for a, n, p in populations)
    return x, y + np.random.default_rng(seed).normal(0.0, noise, len(x))


@pytest.fixture(scope="module")
def mixed():
    """Six spectra of one population and six of two well separated ones."""
    one = [spectrum([(1.0, 18, 0.3)], seed=s) for s in range(6)]
    two = [spectrum([(0.5, 18, 0.15), (0.5, 18, 0.8)], seed=10 + s) for s in range(6)]
    pairs = one + two
    data = Measurements.from_arrays([x for x, _ in pairs], [y for _, y in pairs],
                                    [None] * len(pairs), [N_MAX] * len(pairs),
                                    spectrum_id=[f"s{i}" for i in range(len(pairs))])
    fit = fit_models(data, FIT)
    return data, fit, population_test(data, fit, TEST)


def test_the_gamma_tail_of_the_chi_square_moments_is_the_chi_square_tail():
    t = np.array([0.0, 1.0, 3.0, 7.8, 20.0])
    np.testing.assert_allclose(gamma_tail(t, 3.0, 6.0), chi2.sf(t, 3), rtol=1e-10)
    # a null without spread: 1 at or below its mean, 0 above; NaN moments give NaN
    np.testing.assert_array_equal(gamma_tail([1.0, 2.0, 3.0], 2.0, 0.0), [1.0, 1.0, 0.0])
    assert np.isnan(gamma_tail(1.0, np.nan, 1.0))


def test_steps_stop_at_the_first_that_does_not_add_a_population():
    cfg = PopulationTestConfig(significance=0.01, min_population=0.1)
    p = np.array([[0.001, 0.001], [0.001, 0.5], [0.5, 0.001], [0.001, 0.001], [0.001, 0.001]])
    added = np.array([[0.3, 0.2], [0.3, 0.2], [0.3, 0.2], [0.05, 0.2], [0.3, 0.2]])
    usable = np.ones_like(p, dtype=bool)
    usable[4, 1] = False  # the width check left the second step
    np.testing.assert_array_equal(steps_K(p, added, usable, cfg), [3, 2, 1, 1, 2])


def test_the_width_check_finds_a_component_narrower_than_deuterium_allows():
    # component 2: mean 7 D at p = 0.995 (sd 0.19 D) against at least 0.3 sqrt(7 x 0.1) = 0.25
    n = np.array([[20.0, 7.035], [20.0, 20.0], [20.0, 0.5]])
    p = np.array([[0.2, 0.995], [0.2, 0.6], [0.2, 0.999]])
    a = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
    nan = np.full(3, np.nan)
    kf = KFit(k=2, A=a, n=n, p=p, log_A=a, n_raw=n, logit_p=p, nll=nan, bic=nan, aic=nan,
              passes_min_pop=np.ones(3, dtype=bool))

    class Fit:
        fits = {2: kf}

    # the third is narrow too, but under 1 D: a tail, not a species
    np.testing.assert_array_equal(narrow_components(Fit, 2, 0.9), [True, False, False])
    # with a lower D2O fraction the bound is wider, so the same component is narrower still
    assert narrow_components(Fit, 2, 0.5)[0]


def test_one_and_two_populations_are_told_apart(mixed):
    _, fit, test = mixed
    assert test.steps == (1,)
    assert (test.null_kind[:, 0] == SIMULATED).all()
    np.testing.assert_array_equal(test.selected_K, [1] * 6 + [2] * 6)
    assert (test.p[6:, 0] < 1e-6).all() and (test.p[:6, 0] > 0.01).all()
    np.testing.assert_array_equal(test.criterion_K, fit.selected_K)
    assert (test.null_draws[:, 0] > -1e-6).sum() >= 0.9 * test.null_draws[:, 0].size
    # the moments are those of the draws, clipped at 0
    np.testing.assert_allclose(test.null_mean[:, 0],
                               np.clip(test.null_draws[:, 0], 0, None).mean(axis=1), rtol=1e-5)


def test_a_spectrum_is_tested_alone_whatever_else_is_in_the_batch(mixed):
    data, fit, test = mixed
    rows = np.array([7, 2, 9])
    part = population_test(data.select(rows), fit.select(rows), TEST)
    np.testing.assert_allclose(part.null_draws, test.null_draws[rows], rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(part.selected_K, test.selected_K[rows])
    merged = PopulationTest.merge([test.select([0, 1]), test.select(np.arange(2, 12))],
                                  [[0, 1], list(range(2, 12))])
    np.testing.assert_array_equal(merged.null_draws, test.null_draws)


def test_the_chi_square_null_needs_no_simulation(mixed):
    data, fit, _ = mixed
    quick = population_test(data, fit, dataclasses.replace(TEST, null="chi_square"))
    assert (quick.null_kind == CHI_SQUARE).all() and np.isnan(quick.null_draws).all()
    np.testing.assert_allclose(quick.null_mean, 3.0)
    np.testing.assert_array_equal(quick.selected_K, [1] * 6 + [2] * 6)


def test_the_test_screen_skips_only_what_cannot_reach_the_bound():
    # one population almost without noise: no mixture can gain 1 on it; two populations can
    pairs = ([spectrum([(1.0, 18, 0.3)], noise=1e-4, seed=s) for s in range(3)]
             + [spectrum([(0.5, 18, 0.15), (0.5, 18, 0.8)], seed=10 + s) for s in range(3)])
    data = Measurements.from_arrays([x for x, _ in pairs], [y for _, y in pairs], [None] * 6,
                                    [N_MAX] * 6, spectrum_id=[f"s{i}" for i in range(6)])
    full = fit_models(data, FIT)
    screened = fit_models(data, dataclasses.replace(FIT, screen=True), test_screen=True)
    skipped = ~screened.fitted(2)
    np.testing.assert_array_equal(skipped, [True] * 3 + [False] * 3)
    # where it fitted, the fits are those of fitting everything
    np.testing.assert_allclose(screened.fits[2].nll[~skipped], full.fits[2].nll[~skipped],
                               rtol=1e-6)
    # a skipped spectrum's statistic was below 1 in the full fit
    assert (2 * (full.fits[1].nll - full.fits[2].nll))[skipped].max() < 1.0
    with pytest.raises(ValueError, match="criterion's screen"):
        population_test(data, screened, TEST)
    test = population_test(data, screened, TEST, test_screen=True)
    assert (test.null_kind[skipped, 0] == SCREENED).all()
    assert (test.statistic[skipped, 0] == 0).all()
    np.testing.assert_array_equal(test.selected_K, [1] * 3 + [2] * 3)


def test_settings_are_checked_and_round_trip():
    with pytest.raises(ValueError, match="k_min = 1"):
        RunConfig(fit=FitConfig(k_min=2, k_max=3), population_test=TEST)
    with pytest.raises(ValueError, match="k_max of 2"):
        RunConfig(fit=FitConfig(k_max=1), population_test=TEST)
    with pytest.raises(ValueError, match="Gaussian"):
        RunConfig(fit=FitConfig(k_max=2, noise=NoiseConfig("poisson", percent=3.0)),
                  population_test=TEST)
    with pytest.raises(ValueError, match="null must be"):
        PopulationTestConfig(null="exact")
    with pytest.raises(ValueError, match="unknown"):
        PopulationTestConfig.from_dict({"sims": 3})
    c = RunConfig(fit=FitConfig(k_max=3), population_test=TEST)
    assert RunConfig.from_dict(c.to_dict()) == c


def test_the_run_takes_each_spectrum_k_from_the_test_and_stores_it(tmp_path):
    from test_run import config, raw_spectra

    raw = raw_spectra(n_series=2)
    cfg = config(population_test=PopulationTestConfig(simulations=4))
    res = run(raw, cfg, out=tmp_path / "b")
    test = res.population_test
    assert test is not None and len(test) == len(res.fit)
    np.testing.assert_array_equal(res.fit.selected_K, test.selected_K)
    np.testing.assert_array_equal(res.bootstrap.selected_K, test.selected_K)
    with open_bundle(tmp_path / "b") as b:
        stored = b.population_test()
        assert b.manifest["configs"]["population_test"] == cfg.population_test.to_dict()
    for name in ("statistic", "null_mean", "null_var", "added_population", "p", "null_draws"):
        np.testing.assert_allclose(getattr(stored, name), getattr(test, name), rtol=1e-6,
                                   equal_nan=True)
    for name in ("null_kind", "narrow", "selected_K", "criterion_K"):
        np.testing.assert_array_equal(getattr(stored, name), getattr(test, name))
    assert stored.steps == test.steps and stored.d2o_fraction == cfg.controls.d2o_fraction
    # chunking does not change it
    small = run(raw, dataclasses.replace(cfg, max_chunk=1))
    np.testing.assert_allclose(small.population_test.null_draws, test.null_draws, rtol=1e-3,
                               atol=1e-3)
    np.testing.assert_array_equal(small.population_test.selected_K, test.selected_K)
    # a resumed bootstrap keeps the test
    again = resume_bootstrap(tmp_path / "b", bootstrap=cfg.bootstrap, add=True)
    np.testing.assert_array_equal(again.population_test.selected_K, test.selected_K)
    with open_bundle(tmp_path / "b") as b:
        assert b.population_test() is not None


def test_the_project_file_turns_the_test_on(tmp_path):
    from hxex.config import run_config_from, template

    cfg = run_config_from({"fit": {"k_max": 2}, "population_test": {"simulations": 10}})
    assert cfg.population_test == PopulationTestConfig(simulations=10)
    assert run_config_from({"fit": {"k_max": 2}, "population_test": True}).population_test \
        == PopulationTestConfig()
    assert run_config_from({}).population_test is None
    assert "# [run.population_test]" in template("specexport")


@pytest.fixture(scope="module")
def tested_run():
    from test_run import config, raw_spectra

    raw = raw_spectra(n_series=2)
    return run(raw, config(population_test=PopulationTestConfig(simulations=4)))


def _forced(results, rows, statistic=1000.0, added=0.3, narrow=False):
    """``results`` with the test of ``rows`` (fit rows) reading two populations, strongly."""
    test = results.population_test
    st, ad, nw = test.statistic.copy(), test.added_population.copy(), test.narrow.copy()
    st[rows, 0], ad[rows, 0], nw[rows, 0] = statistic, added, narrow
    test = dataclasses.replace(test, statistic=st, added_population=ad, narrow=nw)
    return dataclasses.replace(results, population_test=test)


def _set_k(up, role, time=None):
    rows = up.where(role=role, K=1, rank=1)
    if time is not None:
        rows &= up.time == time
    return up.best_set_K[rows], up.best_set_reason[rows]


def test_the_sets_take_their_k_from_the_test(tested_run):
    from hxex.analysis import species_uptake
    from hxex.analysis.settings import population_rule
    from hxex.report import Results

    res = Results.from_run(tested_run)
    assert population_rule(None, res) == "population_test"
    up = species_uptake(res, populations="population_test")
    assert np.isfinite(up.set_p_1).all() and np.isnan(up.set_p_2).all()
    assert (up.best_set_K == 1).all()
    assert all(r.startswith("set test: one") for r in up.best_set_reason)
    # the labelled spectra reading two make their sets two
    labelled = np.flatnonzero(res.fit.role == "")
    up = species_uptake(_forced(res, labelled), populations="population_test")
    k, reason = _set_k(up, "", 60.0)
    assert (k == 2).all() and all(r.startswith("set test: two") for r in reason)
    assert (_set_k(up, "undeut")[0] == 1).all()
    # the width check is a flag only: the sets keep their two
    up = species_uptake(_forced(res, labelled, narrow=True), populations="population_test")
    assert (_set_k(up, "", 60.0)[0] == 2).all()
    # without the test, best_set_K is the criterion's and the rule says so
    bare = dataclasses.replace(res, population_test=None)
    assert population_rule(None, bare) == "best_set"
    with pytest.raises(ValueError, match=r"\[run.population_test\]"):
        population_rule("population_test", bare)


def test_a_peptide_whose_undeuterated_set_reads_two_is_flagged_or_capped(tested_run):
    from hxex.analysis import SpeciesFilters, species_uptake
    from hxex.report import Results

    res = Results.from_run(tested_run)
    role = np.asarray(res.fit.role)
    # series 0's undeuterated spectra and every labelled spectrum read two
    series0 = np.asarray(res.fit.data_id) == 0
    forced = _forced(res, np.flatnonzero((role == "undeut") & series0 | (role == "")))
    up = species_uptake(forced, populations="population_test")  # by default: flagged, not capped
    assert (_set_k(up, "undeut")[0] == 2).all()  # one peptide, both series in its sets
    assert (_set_k(up, "", 60.0)[0] == 2).all() and up.undeuterated_two.all()
    capped = species_uptake(forced, populations="population_test",
                            filters=SpeciesFilters(undeuterated_cap=True))
    k, reason = _set_k(capped, "", 60.0)
    assert (k == 1).all() and all("undeuterated_cap" in r for r in reason)
    assert not species_uptake(Results.from_run(tested_run),
                              populations="population_test").undeuterated_two.any()


def test_the_residue_tiers_follow_the_rule():
    from hxex.analysis.residues import population_tiers

    s2 = np.array([0.001, 0.001, 0.001, 0.5, 0.5, 0.001, 0.001, np.nan])
    a2 = np.array([0.6, 0.4, 0.1, 0.4, 0.1, 0.6, 0.6, 0.5])
    s3 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.001, 0.001, 0.5])
    a3 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.7, 0.0])
    assert list(population_tiers(s2, a2, s3, a3)) == [
        "two, localised", "two", "undecided", "undecided", "no sign", "three",
        "three, localised", ""]
    # three needs two first
    assert population_tiers([0.5], [0.1], [0.001], [0.7])[0] == "no sign"


def test_the_residues_carry_the_tiers(tested_run):
    from hxex.analysis import residue_map, species_uptake
    from hxex.report import Results

    res = Results.from_run(tested_run)
    labelled = np.flatnonzero(res.fit.role == "")
    up = species_uptake(_forced(res, labelled), populations="population_test")
    rm = residue_map(up, population_test=True)
    one = rm.where(K=1, time=60.0) & ~rm.no_amide
    assert set(rm.population_tier[one]) == {"two, localised"}
    assert (rm.test_agreement_2[one] == 1).all() and (rm.test_strength_2[one] < 0.01).all()
    un = rm.where(K=1, role="undeut") & ~rm.no_amide
    assert set(rm.population_tier[un]) == {"no sign"}
    assert (rm.population_tier[rm.K == 2] == "").all()
    assert set(residue_map(up).population_tier) == {""}


def test_the_analysis_uses_the_test_and_draws_the_tiers(tested_run, tmp_path):
    import csv

    from hxex.analysis import AnalysisConfig
    from hxex.report import Results, write_analysis

    files = write_analysis(Results.from_run(tested_run), AnalysisConfig(fragments=None),
                           tmp_path / "a")
    names = {p.name for p in files}
    assert "population_tiers.pdf" in names
    with open(tmp_path / "a" / "species_uptake.csv") as fh:
        row = next(csv.DictReader(fh))
    assert row["set_p_1"] != "" and row["best_set_reason"].startswith("set test")
    with open(tmp_path / "a" / "residues.csv") as fh:
        tiers = {r["population_tier"] for r in csv.DictReader(fh) if r["K"] == "1"}
    assert "no sign" in tiers
    settings = (tmp_path / "a" / "analysis_settings_used.toml").read_text()
    assert 'populations = "population_test"' in settings


def test_a_second_step_tests_two_against_three(mixed, tmp_path):
    from hxex.io import write_bundle

    data, _, _ = mixed
    rows = np.array([0, 6, 7])
    sub = data.select(rows)
    fit = fit_models(sub, dataclasses.replace(FIT, k_max=3))
    test = population_test(sub, fit, dataclasses.replace(TEST, simulations=3))
    assert test.steps == (1, 2) and test.null_draws.shape == (3, 2, 3)
    assert (test.null_kind == SIMULATED).all() and np.isfinite(test.p).all()
    assert test.selected_K[0] == 1 and (test.selected_K[1:] >= 2).all()
    fit = dataclasses.replace(fit, selected_K=test.selected_K.copy())
    write_bundle(tmp_path / "b", measurements=sub, fit=fit, population_test=test)
    with open_bundle(tmp_path / "b") as b:
        stored = b.population_test()
    np.testing.assert_allclose(stored.null_draws, test.null_draws, rtol=1e-6)
    np.testing.assert_array_equal(stored.null_kind, test.null_kind)


def test_a_set_at_three_keeps_its_k_beside_the_described_two(tested_run):
    from hxex.analysis import species_uptake
    from hxex.analysis.set_test import SetTest
    from hxex.analysis.species import _set_key, test_set
    from hxex.report import Results

    up = species_uptake(Results.from_run(tested_run))
    cols = up.columns()
    keys = {_set_key(*(cols[n][i] for n in ("run", "sample", "peptide", "modification", "start",
                                             "end", "time", "role"))) for i in range(len(up))}
    three = SetTest(K=3, p=(1e-9, 1e-5), added=(0.3, 0.2), n_spectra=3, n_tested=3,
                    undeuterated_two=False, reason="set test: three")
    out = test_set(cols, {key: three for key in keys})
    assert (out["set_test_K"] == 3).all() and (out["best_set_K"] == 2).all()
    assert all("K = 3 not kept" in r for r in out["best_set_reason"])


def test_borderline_follows_the_test_when_it_is_on():
    from hxex.analysis.compare import compare_population_counts

    key = ("PEPTIDE", "", 1, 7, 60.0)
    best = {("", "WT", *key): (1, 5.0), ("", "A", *key): (2, -50.0),
            ("", "B", *key): (2, -50.0)}
    # A: clear by the criterion and by the test; B: clear by the criterion, borderline by p
    set_p = {("", "WT", *key): (0.5, np.nan), ("", "A", *key): (1e-6, np.nan),
             ("", "B", *key): (0.005, np.nan)}
    by_criterion = compare_population_counts(best, "WT")
    assert not by_criterion.borderline.any() and np.isnan(by_criterion.p).all()
    by_test = compare_population_counts(best, "WT", set_p=set_p, significance=0.01)
    order = list(by_test.sample)
    assert list(by_test.borderline[[order.index("A"), order.index("B")]]) == [False, True]
    assert by_test.p_reference[0] == 0.5


def test_a_statistic_below_the_loss_precision_is_zero():
    from hxex.population_test import statistic_of

    # an exact fit at K and K + 1: rounding noise is no evidence (it gave arbitrary p-values)
    np.testing.assert_array_equal(statistic_of([1.8e-4, 20.0, 20.0, np.nan],
                                               [1.8e-4 - 1e-17, 19.0, 20.5, 1.0]),
                                  [0.0, 2.0, 0.0, np.nan])
    assert statistic_of([1000.0], [1000.0 - 0.05])[0] == 0.0  # below 1e-4 of the loss


def test_the_residue_choice_follows_the_tiers(tested_run):
    from hxex.analysis import residue_map, species_uptake
    from hxex.analysis.populations import ChoiceConfig, choose_populations
    from hxex.report import Results

    res = Results.from_run(tested_run)
    labelled = np.flatnonzero(res.fit.role == "")
    up = species_uptake(_forced(res, labelled), populations="population_test")
    rm = residue_map(up, two_population_peptides=True, population_test=True)
    choice = choose_populations(rm, tiers=True)
    at = choice.where(time=60.0) & np.isfinite(choice.fD_low)
    assert at.any() and choice.two_populations[at].all()
    assert (choose_populations(rm, ChoiceConfig(tier="localised"), tiers=True)
            .two_populations[at].all())
    plain = choose_populations(residue_map(species_uptake(res, populations="population_test"),
                                           two_population_peptides=True,
                                           population_test=True), tiers=True)
    some = np.isfinite(plain.fD_avg) & np.isfinite(plain.fD_low)
    assert not plain.two_populations.any()
    assert set(plain.reason[some]) <= {"tier: no sign", "tier: undecided"}
    with pytest.raises(ValueError, match="tier"):
        ChoiceConfig(tier="three")


def test_spectra_above_full_deuteration_leave_the_test(tested_run):
    from hxex.analysis import SpeciesFilters, left_out_summary, species_uptake
    from hxex.report import Results

    res = Results.from_run(tested_run)
    forced = _forced(res, np.flatnonzero(res.fit.role == ""))
    assert (_set_k(species_uptake(forced, populations="population_test"), "", 60.0)[0] == 2).all()
    # every two-population fit above a (deliberately low) limit: the sets fall back to one
    low = SpeciesFilters(max_fD=0.01)
    k, reason = _set_k(species_uptake(forced, populations="population_test", filters=low),
                       "", 60.0)
    assert (k == 1).all() and all(r == "no spectrum to test: one population" for r in reason)
    summary = left_out_summary(forced, low, population_test=True)
    row = [i for i, w in enumerate(summary["what"]) if w.startswith("held out of the test")]
    assert len(row) == 1 and summary["count"][row[0]] > 0
