# SPDX-License-Identifier: GPL-3.0-or-later
"""The population test of sets of spectra (docs/ANALYSIS.md "The population test").

A set is a species at one time point or control (run, sample, peptide, modification, residues,
point): its replicates and charge states, the spectra of the molecules at that point. The run
tested every spectrum (:mod:`hxex.population_test`); a set adds up its spectra's statistics and
the moments of their nulls, step by step, and has K + 1 populations when the step's p-value is
below ``significance`` and the added population's mean share is at least ``min_population``
(the run's settings), up to the first step that does not reject.

Which spectra count: the ones the analysis keeps (not rejected, not left out by
``[analysis.filters]``, :func:`hxex.analysis.left_out_spectra`). Spectra kept at one
population leave the test (they neither add nor take away evidence): those the spectrum quality
screens keep at one population, those the review set to ``review_K = 1``, and, from the step
whose K + 1 fit it is, a spectrum with a population above full deuteration (corrected fD above
``[analysis.filters] max_fD``, 1.1: about the 99th to 99.5th percentile of genuine fully
deuterated single populations; such a population cannot be a deuterated form of the peptide,
HISTORY 178-179), the review's accepted spectra excepted. A set left with no spectrum to test
has one population.

A flag by default (owner, 2026-10-26, HISTORY 175): ``undeuterated_two``, every set of a peptide
(in every sample of the run) whose undeuterated set has two populations or more. An
undeuterated sample has one, so the extra one is another species; whether the labelled spectra
share it is not known (it can move out of the bins as they take up deuterium), so their tests
are not changed. With ``undeuterated_cap`` (off by default) the labelled and fully deuterated
spectra of such a peptide leave the test as well (kept at one population), except those the
review accepted.
"""
# evidence: docs/HISTORY.md items 139, 161-162, 168-175

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..population_test import PopulationTestConfig, gamma_tail, steps_K
from ..spectra import UNDEUT
from .screening import Screening, screening

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SetTest:
    """The test of one set: ``K``, the p-value and the mean added population of each step
    (``p``, ``added``; NaN where the step was not tested), ``n_spectra`` (the set's spectra),
    ``n_tested`` (those tested at the first step), ``undeuterated_two`` (a flag: the peptide's
    undeuterated set, in some sample of the run, has two populations or more) and ``reason``
    (for people)."""

    K: int
    p: tuple[float, ...]
    added: tuple[float, ...]
    n_spectra: int
    n_tested: int
    undeuterated_two: bool
    reason: str


def run_tests(results) -> list:
    """The population tests of ``results`` (one run, or a :class:`hxex.analysis.RunSet`), None
    for a run without one."""
    from .runs import RunSet

    runs = results.runs.values() if isinstance(results, RunSet) else [results]
    return [getattr(r, "population_test", None) for r in runs]


def has_test(results) -> bool:
    """Whether every run of ``results`` has the population test."""
    tests = run_tests(results)
    return bool(tests) and all(t is not None for t in tests)


@dataclass
class _Run:
    """One run's spectra for the sets: the test, the rows used, the rows kept at one
    population (all steps), and each row's set key and peptide."""

    name: str
    test: Any
    use: np.ndarray
    one: np.ndarray
    accepted: np.ndarray
    unphysical: np.ndarray
    keys: list
    peptide: list
    undeut: np.ndarray


def _run_rows(name: str, r, state, filters, exclude_flagged: bool, controls) -> _Run:
    from .species import _kept, _points, left_out_spectra, review_one

    fit, b = r.fit, r.binned
    if b is None:
        raise ValueError("the population test of sets needs the picked table (results.binned)")
    test = r.population_test
    if test is None:
        raise ValueError(f"run {name or '(the run)'} has no population test: add "
                         "[run.population_test] and run again, or set [analysis] populations "
                         "= \"best_set\"")
    use = _kept(r, exclude_flagged)
    if filters is not None:
        gone = set(map(str, left_out_spectra(r, filters, exclude_flagged=exclude_flagged,
                                             controls=controls)["spectrum_id"]))
        use &= np.array([str(s) not in gone for s in fit.spectrum_id], dtype=bool)
    screen = (screening(r, filters.spectrum_quality_filtering, filters.off_mz)
              if filters is not None else Screening.nothing(len(fit)))
    one = screen.one_population | review_one(r)
    index = {str(s): j for j, s in enumerate(b.spectrum_id)}
    rows = [index[str(s)] for s in fit.spectrum_id]
    points = _points(fit)
    keys, peptide = [], []
    for i, j in enumerate(rows):
        run, sample = state(name, str(b.sample[j]))
        keys.append((run, sample, str(b.peptide[j]), str(b.modification[j]),
                     int(b.start_seq[j]), int(b.end_seq[j]), points[i]))
        peptide.append((name, str(b.peptide[j]), str(b.modification[j]), int(b.start_seq[j]),
                        int(b.end_seq[j])))
    unphysical = (above_max_fd(r, filters, controls, screen, _kept(r, exclude_flagged))
                  if filters is not None
                  else np.zeros(test.statistic.shape, dtype=bool))
    return _Run(name, test, use, one, np.asarray(r.review_status) == "accepted", unphysical,
                keys, peptide, np.asarray(fit.role, dtype=str) == UNDEUT)


def above_max_fd(r, filters, controls=None, screen=None, use=None) -> np.ndarray:
    """``(M, S)``: whether the K + 1 fit of each step has a population above ``filters.max_fD``
    (fD corrected for back-exchange as the tables correct it, measured deuterons over amides
    for an uncorrected series), carried to the later steps."""
    from ..controls import correct_means
    from .species import _kept, _screened_controls, _site_counts

    fit, test = r.fit, r.population_test
    if screen is None:
        screen = screening(r, filters.spectrum_quality_filtering, filters.off_mz)
    if use is None:
        use = _kept(r, False)
    ctl, _ = _screened_controls(r, controls, screen, use, quiet=True)
    out = np.zeros(test.statistic.shape, dtype=bool)
    for s, k in enumerate(test.steps):
        means = np.asarray(fit.fits[k + 1].means, dtype=float)
        if ctl is not None:
            means = correct_means(ctl, fit.data_id, means).mean
            amides = np.asarray(ctl.n_amides, dtype=float)[ctl.rows(fit.data_id)]
        else:
            amides = _site_counts(r)
        with np.errstate(invalid="ignore", divide="ignore"):
            fd = np.where(amides[:, None] > 0, means / amides[:, None], np.nan)
            out[:, s] = np.nan_to_num(fd, nan=-np.inf).max(axis=1) > filters.max_fD
    return np.logical_or.accumulate(out, axis=1)


def set_tests(results, *, filters=None, exclude_flagged: bool = False,
              controls: Mapping[str, Any] | None = None, undeuterated_cap: bool = False,
              quiet: bool = False) -> dict[tuple, SetTest]:
    """The test of every set of ``results`` (one run, or a :class:`hxex.analysis.RunSet`;
    merged samples pool their runs' spectra), by ``(run, sample, peptide, modification, start,
    end, point)`` as the species table's sets (:func:`hxex.analysis.species.best_set`).
    ``filters`` (:class:`hxex.analysis.SpeciesFilters`) and ``controls`` decide the spectra
    kept, as in the tables; ``undeuterated_cap`` as in the module docstring. Every run must have
    the population test, made with the same settings."""
    from .runs import RunSet

    if isinstance(results, RunSet):
        named = list(results.runs.items())
        state = results.state
    else:
        named = [("", results)]

        def state(_run, sample):
            return ("", sample)

    runs = [_run_rows(name, r, state, filters, exclude_flagged, controls) for name, r in named]
    first = runs[0].test
    for run in runs[1:]:
        if run.test.config != first.config or run.test.steps != first.steps:
            raise ValueError("the runs' population tests were made with different settings "
                             "([run.population_test], [run.fit] k_max): their sets cannot be "
                             "pooled")
    config, steps = first.config, first.steps

    members: dict[tuple, list[tuple[int, int]]] = {}
    for g, run in enumerate(runs):
        for i in np.flatnonzero(run.use):
            members.setdefault(run.keys[i], []).append((g, int(i)))
    result = {key: _one_set(rows, runs, config, steps) for key, rows in members.items()}
    # the flag: peptides whose undeuterated set, in some sample, has two populations or more
    flagged = {runs[g].peptide[i] for key, rows in members.items() if result[key].K >= 2
               for g, i in rows if runs[g].undeut[i]}
    if flagged and undeuterated_cap:
        result = {key: _one_set(rows, runs, config, steps, capped=flagged)
                  for key, rows in members.items()}
    if flagged:
        for key, rows in members.items():
            if any(runs[g].peptide[i] in flagged for g, i in rows):
                result[key] = dataclasses.replace(result[key], undeuterated_two=True)
    if not quiet:
        n_two = sum(t.K >= 2 for t in result.values())
        log.info("population test: %d of %d sets have two populations or more", n_two,
                 len(result))
        if flagged:
            log.warning("population test: %d peptides have two populations or more in an "
                        "undeuterated set (another species: undeuterated_two in "
                        "species_uptake.csv); %s", len(flagged),
                        "their labelled and fully deuterated spectra are kept at one "
                        "population ([analysis.filters] undeuterated_cap)" if undeuterated_cap
                        else "their other sets are tested as they are")
    return result


def _one_set(rows, runs, config: PopulationTestConfig, steps, capped=frozenset()) -> SetTest:
    n_steps = len(steps)
    t_sum, m_sum, v_sum = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    added: list[list[float]] = [[] for _ in range(n_steps)]
    n_live = np.zeros(n_steps, dtype=np.int64)
    is_capped = False
    for g, i in rows:
        run = runs[g]
        test = run.test
        if run.one[i]:
            continue
        if run.peptide[i] in capped and not run.undeut[i] and not run.accepted[i]:
            is_capped = True
            continue
        for s in range(n_steps):
            if test.null_kind[i, s] == "" or (run.unphysical[i, s] and not run.accepted[i]):
                break  # not tested at this step (or a population above max_fD): nor after
            t_sum[s] += test.statistic[i, s]
            m_sum[s] += test.null_mean[i, s]
            v_sum[s] += test.null_var[i, s]
            if np.isfinite(test.added_population[i, s]):
                added[s].append(float(test.added_population[i, s]))
            n_live[s] += 1
    tested = n_live > 0
    p = np.where(tested, gamma_tail(t_sum, m_sum, v_sum), np.nan)
    mean_added = np.array([np.mean(a) if a else np.nan for a in added])
    k = int(steps_K(p[None, :], mean_added[None, :], tested[None, :], config)[0])
    words = {1: "one", 2: "two", 3: "three"}
    if not tested[0]:
        reason = ("kept at one population: the peptide's undeuterated set has two or more "
                  "(undeuterated_cap)" if is_capped else "no spectrum to test: one population")
    else:
        at = max(k - 2, 0)  # the last step that added a population, else the first
        reason = (f"set test: {words.get(k, str(k))} (p = {p[at]:.2g}, added population "
                  f"{mean_added[at]:.2f})")
    return SetTest(K=k, p=tuple(float(x) for x in p), added=tuple(float(x) for x in mean_added),
                   n_spectra=len(rows), n_tested=int(n_live[0]), undeuterated_two=False,
                   reason=reason)
