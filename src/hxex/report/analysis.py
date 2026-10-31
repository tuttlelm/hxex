# SPDX-License-Identifier: GPL-3.0-or-later
"""Write a whole post-fitting analysis (docs/ANALYSIS.md) from results: the tables as CSV files and
the figures, as :class:`hxex.analysis.AnalysisConfig` says (what ``hxex analyse`` runs).

Tables: ``species_uptake.csv``, ``residues.csv``, ``coverage.csv``, ``left_out_spectra.csv``
(the spectra ``[analysis.filters]`` left out whole, with the reason), ``populations.csv`` (the
choice between one and two populations; with K=2), and with a reference sample
``compare_peptides.csv`` (K=1 fD, and with K=2 each rank's fD and the rank-1 population),
``compare_residues.csv``, ``compare_populations.csv``, ``compare_choices.csv`` and
``compare_population_counts.csv`` (each sample's number of populations against the
reference's), with regions ``regions.csv``, and for several runs (a
:class:`hxex.analysis.RunSet`) ``run_settings.csv``
(where their settings differ) and ``run_identity.csv`` (species whose identity differs).
Figures: ``coverage_<sample>``, ``redundancy_resolution``, ``choice_<sample>_<time>`` (or
``profile_<sample>_<time>`` without K=2), ``population_tiers`` (with the population test: the
residue tiers, :func:`hxex.report.plot_population_tiers`), and with a reference
``compare_residues_<time>`` (the values of ``config.overlay``,
:func:`hxex.report.residue_overlay_figure`), ``compare_populations_<time>``,
``residue_significance`` (which residues differ from the reference,
:func:`hxex.report.plot_residue_significance`), ``residue_significance_K2``
(:func:`hxex.report.plot_residue_significance_k2`), ``woods_<sample>_<time>`` (K=1, and where the
number of populations differs), ``woods_K2_<sample>_<time>`` (the K=2 comparisons),
``butterfly_<sample>`` and ``regions_<time>``. For a run with fragment series
(:class:`hxex.analysis.FragmentAnalysis`;
those series are left out of the above): ``fragment_uptake.csv``, ``ladder_rates.csv``,
``ladder_residues.csv``, ``closure.csv`` and per parent
``ladder_<sample>_<range>_z<charge>_<ion>_z<fragment charge>``,
``closure_..._<n ion>+<c ion>`` and ``residue_rates_...``. With ``protection`` (opt-in,
:mod:`hxex.analysis.protection`): ``protection.csv``, ``protection_by_time.csv`` and the
figures ``protection_<sample>``, and with a reference ``compare_protection.csv`` and
``protection_difference_<sample>``. The ``<sample>`` of a figure is its display label
(:func:`hxex.analysis.display_labels`: ``run:sample`` where the sample's name is in several
runs of a :class:`hxex.analysis.RunSet`); characters a file name cannot hold (that ``:``)
become ``_``. The tables name the run in a ``run`` column instead.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

import numpy as np

from ..analysis import (
    AnalysisConfig,
    RunSet,
    best_sets,
    check_identity,
    choose_populations,
    compare_choices,
    compare_peptides,
    compare_population_counts,
    compare_populations,
    compare_residues,
    compare_settings,
    concat_tables,
    coverage_table,
    judged_on_shared,
    left_out_spectra,
    left_out_summary,
    paired_residue_maps,
    region_summary,
    residue_map,
    set_pvalues,
    species_uptake,
)
from ..analysis.residues import NoPeptideAtTime
from ..analysis.residues import states as _states
from ..analysis.runs import display_labels, label, resolve_sample
from ..analysis.settings import OVERLAY_VALUES, UNDEUTERATED, population_rule
from ..hooks import CancelHook, ProgressHook, check_cancel, report
from ..io.outputs import replace_outputs
from ..io.times import format_time
from .compare import (
    plot_butterfly,
    plot_region_bars,
    residue_overlay_figure,
    woods_figure,
    woods_populations_figure,
)
from .coverage import coverage_figure, plot_redundancy_resolution
from .fragments import fragment_outputs
from .protection import (
    protection_difference_figure,
    protection_figure,
)
from .residues import plot_population_choice, plot_residue_profile
from .significance import plot_residue_significance, plot_residue_significance_k2
from .tables import write_csv
from .tiers import plot_population_tiers

log = logging.getLogger(__name__)


def _columns(table) -> dict[str, np.ndarray]:
    cols = {f.name: getattr(table, f.name) for f in fields(table)
            if isinstance(getattr(table, f.name), np.ndarray) and f.name != "run"}
    return {"run": table.run, **cols}


def _display(table, labels: dict, reference=None):
    """A copy of a table for the figures, its ``sample`` replaced by the display label of each
    row's ``(run, sample)`` (``labels``, :func:`hxex.analysis.display_labels`), so that every
    state has one name, and its ``reference`` by the reference's label if given."""
    shown = np.array([labels.get((str(r), str(s)), label(str(r), str(s))) for r, s in
                      zip(table.run, table.sample, strict=True)], dtype=object)
    changes = {"sample": shown}
    if reference is not None:
        changes["reference"] = reference
    return dataclasses.replace(table, **changes)


def _time_scale(results) -> str:
    """:attr:`hxex.report.Results.time_scale` of a run or a RunSet (seconds for results that
    do not say, such as the fragment readers' stand-ins)."""
    return getattr(results, "time_scale", "seconds")


def _time_label(time: float, scale: str = "seconds") -> str:
    """A time for a file name: ``5min``, ``20h``, ``4s`` (``undeuterated`` for a control
    without a time, NaN; the number alone when the times only order the points)."""
    return ("undeuterated" if not np.isfinite(time)
            else format_time(time, scale).replace(" ", ""))


def _stem(name: str) -> str:
    """``name`` as a file name on every system."""
    return re.sub(r'[<>:"/\\|?*\s]', "_", name)


def _ranked(by_rank: dict) -> dict[str, np.ndarray]:
    """One table of residue comparisons at K = 2 (rank -> comparison), with a ``rank``
    column (1: the low-D population's position, 2: the high-D one) after ``residue``."""
    out: dict[str, list] = {}
    for rank, comparison in by_rank.items():
        for name, values in comparison.columns().items():
            out.setdefault(name, []).append(values)
            if name == "residue":
                out.setdefault("rank", []).append(np.full(len(comparison), rank))
    return {name: np.concatenate(values) for name, values in out.items()}


def _has_overlay(residues, sample: str, time: float, show, choice=None) -> bool:
    """Whether ``sample`` has a residue value at ``time`` among the overlay values ``show``
    (the K=2 ones only where ``choice``, if given, keeps two populations; the populations
    from ``choice``)."""
    for value in show:
        if value.startswith("pop_"):  # the populations come from the choice
            if choice is not None and (choice.where(sample=sample, time=time)
                                       & np.asarray(choice.two_populations, dtype=bool)
                                       & np.isfinite(choice.population_low)).any():
                return True
            continue
        k, rank = OVERLAY_VALUES[value]
        rows = residues.where(sample=sample, time=time, K=k, rank=rank) & np.isfinite(residues.fD)
        if choice is not None and k >= 2:
            two = choice.where(sample=sample, time=time) & np.asarray(choice.two_populations,
                                                                      dtype=bool)
            rows &= np.isin(residues.residue, choice.residue[two])
        if rows.any():
            return True
    return False


def _without_times(results, times):
    """``results`` (a run or a RunSet) with ``times`` left out of the analysis
    (:meth:`hxex.report.Results.without_times`); the time of a control that is only a control
    (a time point serving as a control as well may be left out) is refused."""
    runs = results.runs if isinstance(results, RunSet) else {"": results}
    clash = set()
    for r in runs.values():
        fit = r.fit
        only = (np.asarray(fit.role, dtype=str) != "") & np.isfinite(np.asarray(fit.time, float))
        for t in times:
            if np.any(only & np.isclose(np.asarray(fit.time, float), t)):
                clash.add(float(t))
    scale = _time_scale(results)
    if clash:
        raise ValueError(f"exclude_times {', '.join(format_time(t, scale) for t in sorted(clash))}"
                         ": a control's time (the controls are set by the run)")
    out = {name: r.without_times(times) for name, r in runs.items()}
    n = sum(int(r.excluded.sum()) for r in out.values())
    if n:
        log.info("[analysis] exclude_times: %d measurement(s) at %s left out of the analysis", n,
                 ", ".join(format_time(t, scale) for t in times))
    else:
        log.warning("[analysis] exclude_times: no measurement at %s",
                    ", ".join(format_time(t, scale) for t in times))
    return RunSet(out, merge=results.merge) if isinstance(results, RunSet) else out[""]


def _fragment_series(results) -> tuple[set[int], bool]:
    """The series (``data_id``) read from fragment lists (the precursor M read with them
    included), and whether every series is one."""
    from ..io.mzml import SCAN_READERS, mzml_series_targets

    source = results.settings.spectra_source if results.settings is not None else None
    if not source or source.get("reader") not in SCAN_READERS or "series" not in source:
        return set(), False
    targets = mzml_series_targets(source)
    series = {d for d, t in targets.items() if "fragment" in t.entry.extra}
    return series, bool(targets) and len(series) == len(targets)


def _peptide_outputs(results, config: AnalysisConfig, base, up, cancel=None):
    """The tables (name -> columns) and figures (stem -> drawing function) of the peptide
    analysis of the species table ``up``."""
    # the samples are states (run, sample); the settings name them by sample (run:sample for
    # one run's), and the figures by their display label
    present = _states(up)
    wanted = ([resolve_sample(up.run, up.sample, s) for s in config.samples]
              if config.samples else present)
    ref_state = (resolve_sample(up.run, up.sample, config.reference, "reference sample")
                 if config.reference else None)
    if ref_state is not None:
        wanted = [ref_state] + [s for s in wanted if s != ref_state]
    pairs = list(zip(up.run.astype(str), up.sample.astype(str), strict=True))
    up = up.select(np.array([p in set(wanted) for p in pairs], dtype=bool))
    sequences = config.sequences(base)
    # K=2 residue values from the peptides whose sets are best described by two populations,
    # with the share of the weight they carry (HISTORY 137)
    # else the bootstrap rule (HISTORY 139)
    gate = config.populations in ("best_set", "population_test")
    residues = residue_map(up, sequences=sequences, weight_exponent=config.weight_exponent,
                           modified=config.modified_peptides,
                           two_population_peptides=gate and 2 in config.ks,
                           population_test=config.populations == "population_test")
    # modified forms kept apart are states of their own (e.g. 'WT pS59'), after their sample
    state_pairs = [s for s in _states(residues) if s not in wanted]
    all_states = wanted + state_pairs
    labels = display_labels([s[0] for s in all_states], [s[1] for s in all_states])
    samples = [labels[s] for s in wanted]  # display labels, the reference first
    states_shown = [labels[s] for s in state_pairs]
    reference = labels[ref_state] if ref_state is not None else ""
    ref_name = label(*ref_state) if ref_state is not None else ""  # names it for the analyses
    coverage_time = None if config.coverage_times == UNDEUTERATED else config.coverage_times
    try:
        coverage = coverage_table(up, sequences=sequences, count=config.redundancy,
                                  time=coverage_time)
    except NoPeptideAtTime as exc:
        log.warning("%s: coverage_times = \"all\" is used instead (set [analysis] "
                    "coverage_times to choose the times coverage is judged on)", exc)
        coverage = coverage_table(up, sequences=sequences, count=config.redundancy, time="all")
    choice = (choose_populations(residues, config.choice_limits,
                                 tiers=config.populations == "population_test")
              if 2 in config.ks else None)
    length = int(residues.residue.max())
    # the per-time figures: every time point (the controls are not times; a control that is
    # also a time point is one)
    times = list(config.times) or sorted({float(t) for t in up.time if np.isfinite(t)})
    scale = _time_scale(results)
    when = {"time_scale": scale}  # the figures label their time by the scale
    ann = config.annotations
    marks = ann.marks()
    tables: dict[str, dict] = {"species_uptake": up.columns(), "residues": residues.columns(),
                               "coverage": coverage.columns()}
    if choice is not None:
        tables["populations"] = choice.columns()
    if config.filters is not None:  # the spectra [analysis.filters] left out whole, and why
        tables["left_out_spectra"] = left_out_spectra(
            results, config.filters, exclude_flagged=config.exclude_flagged,
            controls=config.controls)
    # the figures take the tables with each state's display label as its sample
    up_d, residues_d, coverage_d = (_display(t, labels) for t in (up, residues, coverage))
    choice_d = _display(choice, labels) if choice is not None else None
    if isinstance(results, RunSet):
        tables["run_settings"] = compare_settings(results).columns()
        # the identity check compares the runs, so merged samples are looked at run by run
        unmerged = up if not results.merge else species_uptake(
            results.unmerged(), ks=config.ks, exclude_flagged=config.exclude_flagged,
            filters=config.filters, controls=config.controls,
            min_population=config.choice_limits.min_population, populations="best_set")
        tables["run_identity"] = check_identity(unmerged).columns()
    comparisons = {}
    pop_cmp = counts_d = None
    choices_cmp = k2_positions = None
    k2_tested = False
    if config.reference:
        ref = ref_name
        best = best_sets(up)  # each species' number of populations (best_set_K)
        kw = {"exclude_flagged": config.exclude_flagged, "filters": config.filters,
              "alpha": config.alpha, "samples": [label(*s) for s in wanted],
              "match": config.peptide_match, "controls": config.controls, "best_sets": best,
              "two_population_sets": gate}
        comparisons["K=1 fD"] = compare_peptides(results, ref, **kw)
        if 2 in config.ks:
            comparisons["K=2 rank 1 (low D) fD"] = compare_peptides(results, ref, K=2, rank=1,
                                                                    **kw)
            comparisons["K=2 rank 2 (high D) fD"] = compare_peptides(results, ref, K=2, rank=2,
                                                                     **kw)
            comparisons["K=2 rank 1 (low D) population"] = compare_peptides(
                results, ref, K=2, rank=1, quantity="population",
                choice=config.choice_limits, **kw)
        rows = {}
        for title, c in comparisons.items():
            cols = c.columns()
            cols = {"comparison": np.array([title] * len(c)), **cols}
            for name, values in cols.items():
                rows.setdefault(name, []).append(values)
        tables["compare_peptides"] = {name: np.concatenate(v) for name, v in rows.items()}
        if 2 in config.ks and gate:
            test_p = None
            if config.populations == "population_test":
                from ..analysis.set_test import run_tests

                test_p = dict(significance=run_tests(results)[0].config.significance,
                              set_p=set_pvalues(up))
            counts = compare_population_counts(best, ref, samples=kw["samples"],
                                               match=config.peptide_match,
                                               evidence=config.borderline_evidence,
                                               **(test_p or {}))
            tables["compare_population_counts"] = counts.columns()
            counts_d = _display(counts, labels, reference)
        # each sample's own values; whether a difference is significant is judged from the
        # peptides both samples measured at that time (owner, 2026-10-31; HISTORY 202), so a
        # peptide only one of them has cannot make one
        crit = comparisons["K=1 fD"].delta_crit
        pairs = paired_residue_maps(
            up, ref, match=config.peptide_match, sequences=sequences,
            weight_exponent=config.weight_exponent, modified=config.modified_peptides,
            two_population_peptides=gate and 2 in config.ks,
            population_test=config.populations == "population_test")
        pair_choice = ({s: choose_populations(m, config.choice_limits,
                                              tiers=config.populations == "population_test")
                        for s, m in pairs.items()} if choice is not None else {})
        res_cmp = judged_on_shared(
            compare_residues(residues, ref, alpha=config.alpha, delta_crit=crit, choice=choice),
            concat_tables([compare_residues(m, ref, alpha=config.alpha, delta_crit=crit,
                                            choice=pair_choice.get(s))
                           for s, m in pairs.items()]))
        tables["compare_residues"] = res_cmp.columns()
        if choice is not None:
            # K = 2 differences between samples are tested only with a bootstrap at K = 2 in
            # every run: without one most residues have no error and the few calls do not
            # reproduce the bootstrapped ones (owner, 2026-10-31; HISTORY 204)
            k2_tested = all(r.bootstrap_by_k.get(2) is not None for r in (
                results.runs.values() if isinstance(results, RunSet) else [results]))
            pop_cmp = judged_on_shared(  # the K=1 threshold, divided by each separation
                compare_populations(choice, ref, alpha=config.alpha, delta_crit=crit),
                concat_tables([compare_populations(pair_choice[s], ref, alpha=config.alpha,
                                                   delta_crit=crit) for s in pairs]))
            pop_cols = pop_cmp.columns()
            if not k2_tested:
                pop_cols["z"] = np.full(len(pop_cmp), np.nan)
                pop_cols["significant"] = np.full(len(pop_cmp), np.nan, dtype=object)
                log.info("K=2 differences between samples (population share, positions) not "
                         "tested: no bootstrap at K=2 ([run.bootstrap] ks = [1, 2], or hxex "
                         "resume); residue_significance_K2 shows where the number of "
                         "populations differs")
            tables["compare_populations"] = pop_cols
            choices_cmp = compare_choices(choice, ref)
            tables["compare_choices"] = choices_cmp.columns()
            if k2_tested:  # the two positions, each with its own K=2 threshold
                k2_positions = {}
                for rank, title in ((1, "K=2 rank 1 (low D) fD"), (2, "K=2 rank 2 (high D) fD")):
                    k2_crit = comparisons[title].delta_crit
                    k2_positions[rank] = judged_on_shared(
                        compare_residues(residues, ref, K=2, rank=rank, alpha=config.alpha,
                                         delta_crit=k2_crit, choice=choice),
                        concat_tables([compare_residues(m, ref, K=2, rank=rank,
                                                        alpha=config.alpha,
                                                        delta_crit=k2_crit,
                                                        choice=pair_choice[s])
                                       for s, m in pairs.items()]))
                tables["compare_residues_K2"] = _ranked(k2_positions)
        comparisons = {title: _display(c, labels, reference) for title, c in comparisons.items()}
        res_cmp = _display(res_cmp, labels, reference)
        if choices_cmp is not None:
            choices_cmp = _display(choices_cmp, labels, reference)
            pop_cmp = _display(pop_cmp, labels, reference) if k2_tested else None
            if k2_positions is not None:
                k2_positions = {r: _display(c, labels, reference)
                                for r, c in k2_positions.items()}
    protection = prot_cmp = None
    if config.protection:
        if scale == "order":
            raise ValueError("[analysis] protection needs real labelling times, and [input] "
                             "time_scale = \"order\" says the times only order the points")
        from ..analysis.protection import (
            compare_protection,
            peptide_protection,
        )

        protection, by_time = peptide_protection(up, config.labelling, sequences=sequences,
                                                 exclude_times=(), alpha=config.alpha)
        tables["protection"] = protection.columns()
        tables["protection_by_time"] = by_time.columns()
        if config.reference:
            prot_cmp = compare_protection(protection, ref_name, alpha=config.alpha,
                                          match=config.peptide_match)
            tables["compare_protection"] = prot_cmp.columns()
            prot_cmp = _display(prot_cmp, labels, reference)
        protection = _display(protection, labels)
    regions = None
    if ann.regions:
        regions = region_summary(residues, ann)
        tables["regions"] = _columns(regions)
        regions = _display(regions, labels)
    up, residues, coverage, choice = up_d, residues_d, coverage_d, choice_d
    states = states_shown

    jobs = []  # (file stem, function drawing the figure)
    for s in samples:
        jobs.append((f"coverage_{s}", lambda s=s: coverage_figure(
            up, samples=[s], times=times, length=length, **when)))
    jobs.append(("redundancy_resolution", lambda: plot_redundancy_resolution(
        coverage, samples=samples, marks=marks)))
    runs_read = results.runs.values() if isinstance(results, RunSet) else [results]
    booted = all(r.bootstrap is not None or r.bootstrap_by_k for r in runs_read)
    for s in samples + states:
        for t in times:
            when_ = _time_label(t, scale)
            if choice is not None and choice.where(sample=s, time=t).any():
                jobs.append((f"choice_{s}_{when_}", lambda s=s, t=t: plot_population_choice(
                    choice, s, t, marks=marks, length=length, bootstrap=booted, **when)))
            elif residues.where(sample=s, time=t).any():
                jobs.append((f"profile_{s}_{when_}", lambda s=s, t=t: plot_residue_profile(
                    residues, s, t, marks=marks, length=length, bootstrap=booted, **when)))
    if config.populations == "population_test" and (residues.population_tier != "").any():
        jobs.append(("population_tiers", lambda: plot_population_tiers(
            residues, samples + states, length=length, **when)))
    if config.reference:
        others = [s for s in samples if s != reference]
        residue_others = others + states  # the residue comparisons include the states
        # the Woods plot: page 1, K=1 and where the number of populations differs; page 2,
        # the K=2 comparisons (made only where both have two populations), the population in
        # fD units: its difference times the separation, judged by the K=1 delta_crit
        # (HISTORY 135, 137)
        short = {"K=2 rank 1 (low D) fD": "K=2 low-D fD", "K=2 rank 2 (high D) fD":
                 "K=2 high-D fD", "K=2 rank 1 (low D) population": "K=2 low-D population"}
        k2_stack = [(short.get(title, title), c.scaled_by_separation())
                    for title, c in comparisons.items() if title != "K=1 fD"]
        for t in times:
            when_ = _time_label(t, scale)
            only = choice if config.overlay_two_only else None
            shown = [s for s in residue_others
                     if _has_overlay(residues, s, t, config.overlay, only)]
            if shown and _has_overlay(residues, reference, t, config.overlay, only):
                jobs.append((f"compare_residues_{when_}", lambda t=t, shown=shown, only=only:
                             residue_overlay_figure(
                                 residues, reference, shown, t, show=config.overlay,
                                 choice=only, annotations=ann, length=length, **when)))
            # the populations, drawn as the K=2 values are: each sample where it keeps two
            ref_pop = choice is not None and _has_overlay(residues, reference, t, ("pop_low",),
                                                          choice)
            with_pop = [s for s in residue_others if choice is not None and (
                ref_pop or _has_overlay(residues, s, t, ("pop_low",), choice))]
            if with_pop:
                jobs.append((f"compare_populations_{when_}",
                             lambda t=t, with_pop=with_pop: residue_overlay_figure(
                                 residues, reference, with_pop, t, show=("pop_low",),
                                 choice=choice, annotations=ann, length=length, **when)))
            for s in others:
                k1_rows = comparisons["K=1 fD"].where(sample=s, time=t).any()
                if k1_rows and counts_d is not None:
                    jobs.append((f"woods_{s}_{when_}", lambda s=s, t=t: woods_populations_figure(
                        comparisons["K=1 fD"], counts_d, s, t, annotations=ann,
                        length=length, **when)))
                elif k1_rows:
                    jobs.append((f"woods_{s}_{when_}", lambda s=s, t=t: woods_figure(
                        [("K=1 fD", comparisons["K=1 fD"])], s, t, annotations=ann,
                        length=length, **when)))
                if k2_stack and all(c.where(sample=s, time=t).any() for _, c in k2_stack):
                    jobs.append((f"woods_K2_{s}_{when_}", lambda s=s, t=t: woods_figure(
                        k2_stack, s, t, annotations=ann, length=length,
                        note=("compared where both samples have two populations" if gate
                              else "every K=2 description compared"), **when)))
        # which residues differ from the reference, judged on the shared peptides (HISTORY 203)
        compared = [s for s in residue_others if res_cmp.where(sample=s).any()]
        if compared:
            jobs.append(("residue_significance", lambda compared=compared:
                         plot_residue_significance(res_cmp, compared, length=length, **when)))
        with_two = [s for s in residue_others
                    if choices_cmp is not None and choices_cmp.where(sample=s).any()]
        if with_two:
            positions = k2_positions or {}
            jobs.append(("residue_significance_K2", lambda with_two=with_two, positions=positions:
                         plot_residue_significance_k2(
                             choices_cmp, pop_cmp, positions.get(1), positions.get(2),
                             with_two, length=length, **when)))
        k1 = comparisons["K=1 fD"]
        drawn = list(times)
        for s in others:
            if drawn and k1.where(sample=s, time=drawn).any():
                jobs.append((f"butterfly_{s}", lambda s=s: plot_butterfly(
                    k1, s, times=[t for t in drawn if k1.where(sample=s, time=t).any()],
                    **when)))
    if protection is not None:
        for s in samples:
            if protection.where(sample=s).any():
                jobs.append((f"protection_{s}", lambda s=s: protection_figure(
                    protection, s, length=length)))
    if prot_cmp is not None:
        for s in samples:
            if prot_cmp.where(sample=s).any():
                jobs.append((f"protection_difference_{s}", lambda s=s:
                             protection_difference_figure(prot_cmp, s, length=length)))
    if regions is not None:
        for t in times:
            if (regions.time == t).any():
                jobs.append((f"regions_{_time_label(t, scale)}", lambda t=t: plot_region_bars(
                    regions, t, samples=samples, **when)))

    return tables, jobs


def write_analysis(
    results,
    config: AnalysisConfig,
    directory: str | Path,
    *,
    base: str | Path = ".",
    record_paths: str = "path",
    review=None,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
    before_writing=None,
) -> list[Path]:
    """Compute the analysis of ``results`` (:class:`hxex.report.Results`, or a
    :class:`hxex.analysis.RunSet` to compare separate runs) and write its tables and figures into
    ``directory``; returns the files. Relative FASTA paths are taken from
    ``base``. ``progress(done, total)`` counts the files.

    ``review`` (:class:`hxex.review_table.ReviewSettings`, a project's ``[review]`` table: the
    user review table and its exclusion rules) and ``config.exclude`` are applied first, in
    memory: their rejected spectra are left out.

    With them go the settings used, for people (not read back): ``analysis_settings_used.toml``
    (every setting of ``config`` written out; its paths recorded as ``record_paths`` says, as
    ``[input] record_paths``) and the run settings of the results read
    (``run_settings_used.toml``, or ``run_settings_used_RUN.toml`` per run of a
    :class:`~hxex.analysis.RunSet`), written from their bundles' records, so the folder keeps
    them when a bundle is replaced. Nothing is written when there is no table or figure. The
    folder's listing records the results read (:func:`hxex.io.made_from`), so ``hxex info``
    can tell when the folder no longer matches its bundle. ``before_writing`` (no arguments) is
    called once everything is computed, just before the first file is written (``hxex
    analyse --keep-previous`` moves the earlier folder then, so that a refusal moves
    nothing)."""
    from ..config import settings_records

    directory = Path(directory)
    read = results.runs if isinstance(results, RunSet) else results  # as read, for the records
    if config.exclude or review:  # rejected in memory, the bundle's fits kept
        from ..exclusions import exclude_from_results
        from ..review_table import apply_view

        def view(r):
            return exclude_from_results(apply_view(r, review), config.exclude)

        if isinstance(results, RunSet):
            results = RunSet({name: view(r) for name, r in results.runs.items()},
                             merge=results.merge)
        else:
            results = view(results)
    if config.exclude_times:
        results = _without_times(results, config.exclude_times)
    rule = population_rule(config.populations, results)
    if config.populations is None:
        log.info("the number of populations: %s", "the population test of the sets"
                 if rule == "population_test" else "best_set_K (the run has no population test)")
    config = dataclasses.replace(config, populations=rule)
    frag_tables, frag_jobs = {}, []
    # fragment series (and precursor ions read with them) are not peptides: the peptide
    # branch leaves them out by series (a peptide read from MS1 too has the same sequence)
    runs = results.runs if isinstance(results, RunSet) else {"": results}
    split = {name: _fragment_series(r) for name, r in runs.items()}
    any_fragments = any(series for series, _ in split.values())
    only_fragments = all(only for _, only in split.values())
    if config.fragments is not None and any_fragments:
        if _time_scale(results) == "order":
            raise ValueError("the fragment analysis (ladder kinetics) needs real times, and "
                             "[input] time_scale = \"order\" says the times only order the "
                             "points (fragments = false leaves it out)")
        frag_tables, frag_jobs = fragment_outputs(results, config.fragments)
    tables, jobs = {}, []
    if not only_fragments:
        if any_fragments:
            peptides = {name: r.without_series(split[name][0]) if split[name][0] else r
                        for name, r in runs.items()}
            results = (RunSet(peptides, merge=results.merge) if isinstance(results, RunSet)
                       else peptides[""])
        up = species_uptake(results, ks=config.ks, exclude_flagged=config.exclude_flagged,
                            filters=config.filters, controls=config.controls,
                            min_population=config.choice_limits.min_population,
                            populations=config.populations)
        if len(up):
            tables, jobs = _peptide_outputs(results, config, base, up, cancel=cancel)
            # what was left out on the way, step by step, for people
            tables["left_out_summary"] = left_out_summary(
                results, config.filters, ks=config.ks, exclude_flagged=config.exclude_flagged,
                controls=config.controls, left_out=tables.get("left_out_spectra"),
                exclude_times=config.exclude_times,
                population_test=config.populations == "population_test")
    tables.update(frag_tables)
    jobs += frag_jobs

    total = len(tables) + len(jobs)
    names = [f"{name}.csv" for name in tables] + [f"{_stem(stem)}.{config.format}"
                                                   for stem, _ in jobs]
    records = settings_records(read, config, record_paths, review) if total else {}
    names += list(records)
    total += len(records)
    # as the tables and figures folders: the files of the earlier analysis go, yours stay
    made = {name: str(getattr(getattr(r, "settings", None), "created", "") or "")
            for name, r in (read.items() if isinstance(read, Mapping) else [("", read)])}
    if total and before_writing is not None:
        before_writing()
    with replace_outputs(directory, names, made=made) as written:
        for n, (name, cols) in enumerate(tables.items()):
            check_cancel(cancel)
            written.append(write_csv(cols, directory / f"{name}.csv"))
            report(progress, n + 1, total)
        for n, (stem, draw) in enumerate(jobs):
            check_cancel(cancel)
            path = directory / f"{_stem(stem)}.{config.format}"
            draw().savefig(path, dpi=150, bbox_inches="tight")
            written.append(path)
            report(progress, len(tables) + n + 1, total)
        for n, (name, text) in enumerate(records.items()):
            (directory / name).write_text(text, encoding="utf-8")
            written.append(directory / name)
            report(progress, len(tables) + len(jobs) + n + 1, total)
    log.info("analysis: %d tables and %d figures in %s", len(tables), len(jobs), directory)
    return written
