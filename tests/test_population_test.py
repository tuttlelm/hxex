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
from hxex.pipeline import RunConfig, resume_bootstrap, run
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
    np.testing.assert_allclose(test.null_var[:, 0],  # the sample variance (audit 2026-10-28)
                               np.clip(test.null_draws[:, 0], 0, None).var(axis=1, ddof=1),
                               rtol=1e-4)


def test_a_null_without_spread_leaves_the_spectrum_untested(mixed, monkeypatch):
    """Audit 2026-10-28 Medium 21: when every draw of a spectrum's null is the same (0), the
    null cannot give a p-value (it read 0 for any statistic above it): the spectrum is not
    tested at that step."""
    import hxex.population_test as module

    data, fit, test = mixed
    real = module.simulate_null

    def flat_first(data, fit, rows, *args, **kwargs):
        out = real(data, fit, rows, *args, **kwargs)
        out[rows == 7] = 0.0
        return out

    monkeypatch.setattr(module, "simulate_null", flat_first)
    again = population_test(data, fit, TEST)
    # named apart from an untested spectrum (audit 2026-11-01 Low 58), and still not tested
    assert again.null_kind[7, 0] == "flat" and np.isnan(again.p[7, 0]) and again.selected_K[7] == 1
    assert not again.tested()[7, 0]
    others = np.arange(len(fit)) != 7
    np.testing.assert_array_equal(again.selected_K[others], test.selected_K[others])


def test_a_failed_null_draw_is_left_out_not_the_run(mixed, monkeypatch, caplog):
    """Audit 2026-11-01 Engine 29: a null simulation whose refits all fail (FitError) leaves
    that draw NaN, with a warning; the other draws stand. Low 63: few simulations warn."""
    import logging

    import hxex.population_test as module
    from hxex.fit import FitError

    data, fit, _ = mixed
    real, calls = module.fit_models, []

    def fail_first(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise FitError("all restarts were non-finite for K=1")
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "fit_models", fail_first)
    rows = np.arange(3)
    with caplog.at_level(logging.WARNING):
        sim = module.simulate_null(data, fit, rows, 1, 4)
    assert np.isnan(sim[:, 0]).all() and np.isfinite(sim[:, 1:]).all()
    assert "null simulation 1 at K = 1 failed for 3 spectra" in caplog.text
    monkeypatch.undo()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        population_test(data, fit, dataclasses.replace(TEST, simulations=3))
    assert "3 simulations per spectrum" in caplog.text


def test_a_null_draw_without_signal_is_left_out_for_that_spectrum(mixed, monkeypatch):
    """A draw of a nearly empty spectrum may come out with no bin above 0 (with the noise at the
    fitted curve, HISTORY 291, it did on the benchmark): no K can be judged on it, so it is NaN
    for that spectrum; the others are fitted, and the run goes on."""
    import hxex.population_test as module

    data, fit, _ = mixed
    real = module.scaled_prediction

    def empty_first(*args, **kwargs):
        pred = real(*args, **kwargs)
        return [np.full(len(pred[0]), -1e9), *pred[1:]]  # every draw below 0

    monkeypatch.setattr(module, "scaled_prediction", empty_first)
    sim = module.simulate_null(data, fit, np.arange(3), 1, 3)
    assert np.isnan(sim[0]).all() and np.isfinite(sim[1:]).all()


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
        RunConfig(fit=FitConfig(k_max=2, noise=NoiseConfig("poisson", percent=3.0, estimate=None)),
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

    class Stages:  # a progress hook that follows the stages
        def __init__(self):
            self.names, self.spans = [], []

        def start_stage(self, name, start, end):
            self.names.append(name)
            self.spans.append((start, end))

        def __call__(self, done, total):
            pass

    stages = Stages()
    res = run(raw, cfg, out=tmp_path / "b", progress=stages)
    # the fits and the test's null are stages of their own, one after the other (one stage,
    # the fits were its first few percent: 0 % for minutes)
    at = stages.names.index("fit")
    assert stages.names[at + 1] == "population test"
    assert stages.spans[at][1] == stages.spans[at + 1][0]
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
    stages = Stages()
    small = run(raw, dataclasses.replace(cfg, max_chunk=1), progress=stages)
    chunks = [x for x in stages.names if x.startswith(("fit, chunk", "population test, chunk"))]
    n = len(chunks) // 2
    assert n > 1 and chunks[:2] == [f"fit, chunk 1 of {n}", f"population test, chunk 1 of {n}"]
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


def test_a_time_point_promoted_to_a_control_keeps_its_test(tested_run):
    """Audit 2026-11-01 Analysis 34: [analysis.controls] fulldeut_at / undeut_at promote a time
    point; its sets are keyed as the tables key them, so they keep the test's verdict."""
    from hxex.analysis import SpeciesFilters, species_uptake
    from hxex.report import Results

    res = Results.from_run(tested_run)
    labelled = np.flatnonzero(res.fit.role == "")
    forced = _forced(res, labelled)
    # a labelled point as the undeuterated control reads high: keep the series to look at it
    keep = SpeciesFilters(stand_in_controls="include")
    # the undeuterated control is then the point alone (undeut_marked = false): beside the
    # marked ones, its spectra read above m0_max and are left out of it (HISTORY 286)
    alone = {"undeut_at": 60.0, "undeut_marked": False}
    for change, role in (({"fulldeut_at": 60.0}, "fulldeut"), (alone, "undeut")):
        up = species_uptake(forced, populations="population_test", controls=change,
                            filters=keep)
        rows = up.where(K=1, rank=1) & (up.time == 60.0)
        assert rows.any() and set(up.role[rows]) == {role}
        assert np.isfinite(up.set_p_1[rows]).all()
        assert all(r.startswith("set test: two") for r in up.best_set_reason[rows])
    # a promoted undeuterated set reading two is the peptide's undeuterated_two
    up = species_uptake(forced, populations="population_test", controls=alone, filters=keep)
    assert up.undeuterated_two.all()


def test_narrow_component_is_the_flag_of_the_fit_at_the_spectrums_k():
    from types import SimpleNamespace

    from hxex.report.tables import _test_columns

    # steps one and two: the K = 2 and K = 3 fits' flags
    narrow = np.array([[True, False], [True, False], [False, True], [False, True]])
    test = SimpleNamespace(criterion_K=np.ones(4), steps=(1, 2), p=np.zeros((4, 2)),
                           narrow=narrow, null_kind=np.full((4, 2), "simulated"))
    cols = _test_columns(SimpleNamespace(population_test=test,
                                         effective_K=np.array([1, 2, 2, 3])))
    # K = 1 never; K = 2 the first step's flag; K = 3 the second's
    np.testing.assert_array_equal(cols["narrow_component"], [False, True, False, True])


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


def test_sets_never_tested_take_no_part_in_the_residue_levels():
    """Audit 2026-10-28 question 16: a set without a p-value at step one (never tested) leaves
    the residue's strength and agreement; one tested at step one that read one still enters
    step two with z = 0."""
    from scipy.stats import norm

    from hxex.analysis.residues import _test_levels

    z = np.ones((3, 1))  # three sets covering one residue with equal weight
    p1 = np.array([0.002, 0.5, np.nan])
    tested = np.isfinite(p1)
    s, a = _test_levels(z, p1, np.array([True, False, False]), 1.0, tested)
    zs = norm.isf([0.002, 0.5])
    assert a[0] == pytest.approx(0.5)
    assert s[0] == pytest.approx(norm.sf(zs.sum() / np.sqrt(2)))
    p2 = np.array([0.2, np.nan, np.nan])  # the second set read one: z = 0 at step two
    s, a = _test_levels(z, p2, np.array([False, False, False]), 1.0, tested)
    assert a[0] == 0 and s[0] == pytest.approx(norm.sf(norm.isf(0.2) / np.sqrt(2)))


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
    assert set(residue_map(up, population_test=False).population_tier) == {""}
    # the rule travels with the tables: the species table records it, the residue map and the
    # choice follow it, and the results' own rule is the default (as in hxex analyse)
    assert up.populations == "population_test"
    assert list(residue_map(up).population_tier) == list(rm.population_tier)
    from hxex.analysis import choose_populations

    assert np.array_equal(choose_populations(rm).two_populations,
                          choose_populations(rm, tiers=True).two_populations)
    assert species_uptake(res).populations == "population_test"
    assert species_uptake(res, populations="best_set").populations == "best_set"


def test_the_analysis_uses_the_test_and_draws_the_tiers(tested_run, tmp_path):
    import csv

    from hxex.analysis import AnalysisConfig
    from hxex.report import Results, write_analysis

    files = write_analysis(Results.from_run(tested_run), AnalysisConfig(fragments=None),
                           tmp_path / "a")
    names = {p.name for p in files}
    assert "population_tiers.pdf" in names
    # the tier map's data table: a row per sample, point and residue with a tier
    from hxex.analysis import residue_map, species_uptake
    from hxex.gui.figures import figure_rows
    from hxex.report.tiers import plot_population_tiers

    rm = residue_map(species_uptake(Results.from_run(tested_run),
                                    populations="population_test"))
    columns, table = figure_rows(plot_population_tiers(rm))
    assert columns[:7] == ["sample", "time", "time_label", "residue", "population_tier",
                           "test_strength_2", "test_agreement_2"]
    assert table and all(r["population_tier"] for r in table)
    # a page of a long time course: NaN stands for the controls' rows, else the times given
    labels = [t.get_text() for t in plot_population_tiers(rm, times=[np.nan]).axes[0]
              .get_yticklabels()]
    assert labels and all("deuterated" in label for label in labels)
    later = plot_population_tiers(rm, times=[60.0])
    assert {r["time"] for r in figure_rows(later)[1]} == {60.0}
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
    # a set the test never saw reads 0, as without the test (audit 2026-11-01 Low 68)
    none = test_set(cols, {})
    assert (none["set_test_K"] == 0).all() and (none["best_set_K"] == 1).all()
    assert set(none["best_set_reason"]) == {"not tested: one population"}


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
    # audit 2026-11-01 Low 65: a tier the localised setting does not draw says so
    for tier in ("two", "three"):
        forced = dataclasses.replace(rm, population_tier=np.where(
            np.asarray(rm.population_tier, dtype=object) != "", tier, ""))
        local = choose_populations(forced, ChoiceConfig(tier="localised"), tiers=True)
        assert not local.two_populations[at].any()
        assert set(local.reason[at]) == {f"tier: {tier}, not localised: one drawn"}


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


def _same_arrays(a, b, names=None):
    """Every array field of two results (or those ``names``) equal, NaN where NaN (numbers to
    rounding: a bootstrap of a few spectra adds up in another order than one of the chunk)."""
    for name in names or [f.name for f in dataclasses.fields(a)]:
        x, y = getattr(a, name), getattr(b, name)
        if isinstance(x, np.ndarray) and x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=1e-9, atol=0, equal_nan=True, err_msg=name)
        elif isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y, err_msg=name)


# a permissive test, so that the test's K differs from the criterion's (and the bootstrap of
# the spectra whose K changed is patched)
LOOSE = PopulationTestConfig(simulations=4, significance=0.99, min_population=0.0)


@pytest.mark.parametrize("screen", [False, True])
def test_resume_adds_the_test_as_a_run_with_it_would(tmp_path, screen):
    """hxex resume with [run.population_test] keeps the bundle's fits and gives the test, the
    K, the quality flags and the bootstrap of a run with the test: from the fixed-K bootstrap
    where there is one, else a bootstrap of the spectra whose K changed. A fit made with the
    criterion's screen has the series the test's screen fits differently fitted again."""
    from test_run import config, raw_spectra

    raw = raw_spectra(n_series=2)
    base = config()
    base = dataclasses.replace(
        base, fit=dataclasses.replace(base.fit, screen=screen),
        bootstrap=dataclasses.replace(base.bootstrap, ks="selected" if screen else (1, 2)))
    plain = run(raw, base, out=tmp_path / "b")
    fresh = run(raw, dataclasses.replace(base, population_test=LOOSE))
    assert (fresh.fit.selected_K != plain.fit.selected_K).any()
    res = resume_bootstrap(tmp_path / "b", population_test=LOOSE, project_file={"project": "p"})
    np.testing.assert_array_equal(res.fit.selected_K, fresh.fit.selected_K)
    _same_arrays(res.population_test, fresh.population_test)
    _same_arrays(res.bootstrap, fresh.bootstrap)
    _same_arrays(res.quality, fresh.quality)
    for k in fresh.fit.k_values:
        _same_arrays(res.fit.fits[k], fresh.fit.fits[k])
    with open_bundle(tmp_path / "b") as b:
        np.testing.assert_array_equal(b.population_test().selected_K, fresh.fit.selected_K)
        assert b.manifest["configs"]["run"]["population_test"] == LOOSE.to_dict()
        assert b.manifest["configs"]["project_file"] == {"project": "p"}
        [entry] = b.manifest["provenance"]["resumed"]
        assert entry["added"] == "population test" and entry["project_file_replaced"]
    if screen:
        return
    # a test with other settings is made again, as a run with them makes it
    other = PopulationTestConfig(simulations=4)
    again = resume_bootstrap(tmp_path / "b", population_test=other)
    fresh = run(raw, dataclasses.replace(base, population_test=other))
    np.testing.assert_array_equal(again.fit.selected_K, fresh.fit.selected_K)
    _same_arrays(again.population_test, fresh.population_test)
    _same_arrays(again.bootstrap, fresh.bootstrap)
    with open_bundle(tmp_path / "b") as b:
        assert len(b.manifest["provenance"]["resumed"]) == 2


def test_resume_refuses_the_test_for_fits_that_cannot_carry_it(tmp_path):
    from test_run import config, raw_spectra

    base = config(bootstrap=None)
    run(raw_spectra(n_series=1), dataclasses.replace(
        base, fit=dataclasses.replace(base.fit, k_max=1)), out=tmp_path / "b")
    with pytest.raises(ValueError, match="cannot carry the population test.*k_max.*run again"):
        resume_bootstrap(tmp_path / "b", population_test=TEST)
