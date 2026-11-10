# SPDX-License-Identifier: GPL-3.0-or-later
"""Write a whole post-fitting analysis (docs/ANALYSIS.md) from results: the tables as CSV files and
the figures, as :class:`hxex.analysis.AnalysisConfig` says (what ``hxex analyse`` runs).

Tables: ``species_uptake.csv``, ``residues.csv``, ``coverage.csv``, ``left_out_spectra.csv``
(the spectra ``[analysis.filters]`` left out whole, with the reason), ``populations.csv`` (the
choice between one and two populations; with K=2), and with a reference sample
``compare_peptides.csv`` (K=1 fD, and with K=2 each rank's fD and the rank-1 population),
``compare_residues.csv``, ``compare_populations.csv``, ``compare_choices.csv`` and
``compare_population_counts.csv`` (each sample's number of populations against the
reference's), with regions ``regions.csv`` and ``compare_regions.csv``, and for several runs (a
:class:`hxex.analysis.RunSet`) ``run_settings.csv``
(where their settings differ) and ``run_identity.csv`` (species whose identity differs).
Figures: ``coverage_<sample>``, ``redundancy_resolution``, ``choice_<sample>_<time>`` (or
``profile_<sample>_<time>`` without K=2), ``population_tiers`` (with the population test: the
residue tiers, :func:`hxex.report.plot_population_tiers`), and with a reference
``overlay_residues_<time>`` (the values of ``config.overlay``,
:func:`hxex.report.residue_overlay_figure`), ``overlay_populations_<time>``,
``residue_significance`` (which residues differ from the reference,
:func:`hxex.report.plot_residue_significance`), ``region_significance`` (with regions,
:func:`hxex.report.plot_region_significance`), ``residue_significance_K2_<sample>``
(:func:`hxex.report.plot_residue_significance_k2`, one per sample),
``woods_<sample>_<time>`` (K=1, and where the number of populations differs),
``woods_K2_<sample>_<time>`` (the K=2 comparisons), ``butterfly_<sample>`` and
``regions_<time>``. For a run with fragment series (:class:`hxex.analysis.FragmentAnalysis`;
those series are left out of the above): ``fragment_uptake.csv``, ``closure.csv`` and per
parent ``ladder_<sample>_<range>_z<charge>_<ion>_z<fragment charge>`` and
``closure_..._<n ion>+<c ion>``; with ``fit_rates`` (opt-in) also ``ladder_rates.csv``,
``ladder_residues.csv``, ``residue_rates_...`` and the fits on the ladder figures. With
``protection`` (opt-in, :mod:`hxex.analysis.protection`): ``protection.csv``,
``protection_by_time.csv`` and the figures ``protection_<sample>``, and with a reference
``compare_protection.csv`` and ``protection_difference_<sample>``. The ``<sample>`` of a
figure is its display label (:func:`hxex.analysis.display_labels`: ``run:sample`` where the
sample's name is in several runs of a :class:`hxex.analysis.RunSet`); characters a file name
cannot hold (that ``:``) become ``_``. The tables name the run in a ``run`` column instead.
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
    compare_regions,
    compare_residues,
    compare_settings,
    concat_tables,
    coverage_table,
    gated_by_peptides,
    judged_on_shared,
    left_out_spectra,
    left_out_summary,
    paired_residue_maps,
    region_summary,
    residue_map,
    set_average_ks,
    set_pvalues,
    species_uptake,
)
from ..analysis.residues import NoPeptideAtTime, residue_level_states
from ..analysis.residues import states as _states
from ..analysis.runs import display_labels, label, resolve_sample
from ..analysis.settings import (
    ALL_REFERENCES,
    OVERLAY_VALUES,
    UNDEUTERATED,
    population_rule,
    resolve_match,
)
from ..hooks import CancelHook, ProgressHook, check_cancel, report
from ..io.outputs import replace_outputs
from ..io.times import format_time
from ..spectra import UNDEUT
from .compare import (
    plot_butterfly,
    plot_region_bars,
    plot_residue_samples,
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
from .significance import (
    plot_region_significance,
    plot_residue_significance,
    plot_residue_significance_k2,
)
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


MAP_ROW = 0.2  # inches per row of a significance or tier map
OVERLAY_PANEL = 3.0  # inches per sample of an overlay figure
COVERAGE_CELL = (4.2, 3.2)  # inches per panel of a coverage figure


def _pages(items: list, per_page: int) -> list[list]:
    """``items`` in pages of ``per_page`` (at least one each; one empty page for none)."""
    per_page = max(1, int(per_page))
    return [items[i:i + per_page] for i in range(0, len(items), per_page)] or [[]]


def _page_stem(stem: str, n: int) -> str:
    """Page ``n`` (from 0) of a figure: its own name first, then ``NAME_2``, ``NAME_3``."""
    return stem if n == 0 else f"{stem}_{n + 1}"


def _map_pages(page: tuple[float, float], times: list, rows_per_time: int,
               overhead: float) -> list[list]:
    """The times of a map in pages of ``page`` (width, height in inches): each time takes
    ``rows_per_time`` rows of :data:`MAP_ROW`, the title, axis and legend ``overhead``."""
    rows = max(1, int(round((page[1] - overhead) / MAP_ROW, 6)))
    return _pages(times, rows // max(1, rows_per_time))


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


def _fitted_ks(config: AnalysisConfig, runs: Mapping) -> AnalysisConfig:
    """``config`` with ``ks`` cut to the Ks every run fitted, when ``ks`` is left at its
    default (a run fitted with ``[run.fit] k_max = 1`` is analysed at K = 1, and a default
    ``overlay`` keeps only the K = 1 average); a K set in ``ks`` that a run did not fit is
    refused, naming the settings."""
    fitted = set.intersection(*(set(r.fit.fits) for r in runs.values()))
    missing = [k for k in config.ks if k not in fitted]
    if not missing:
        return config
    default = AnalysisConfig()
    most = max(fitted) if fitted else 0
    if config.ks != default.ks:
        raise ValueError(f"[analysis] ks = {list(config.ks)}: K = {missing} were not fitted "
                         f"(the run tried K up to {most}, [run.fit] k_max): leave ks out, "
                         "remove those Ks, or run again with a larger k_max")
    ks = tuple(k for k in config.ks if k in fitted)
    overlay = (tuple(v for v in config.overlay if OVERLAY_VALUES[v][0] in ks)
               if config.overlay == default.overlay else config.overlay)
    log.info("the run tried K up to %d ([run.fit] k_max), so the analysis uses K = %s",
             most, ", ".join(map(str, ks)))
    return dataclasses.replace(config, ks=ks, overlay=overlay)


def _sample_states(results) -> list[tuple[str, str]] | None:
    """Every ``(run, sample)`` of the results as the analysis names them (merged samples
    under their label), sorted as the analysis orders them; None when the results do not say
    (no binned spectra)."""
    runs = results.runs if isinstance(results, RunSet) else {"": results}
    states = []
    for name, r in runs.items():
        if r.binned is None:
            return None
        for sample in dict.fromkeys(np.asarray(r.binned.sample).astype(str)):
            states.append(results.state(name, sample) if isinstance(results, RunSet)
                          else ("", sample))
    return sorted(dict.fromkeys(states))


def _check_names(results, config: AnalysisConfig) -> None:
    """Refuse a ``reference`` (each of several) or a ``samples`` entry that names no sample of
    the results, before anything is computed (so the message is not lost after the filters'
    lines)."""
    states = _sample_states(results)
    if states is None:
        return  # nothing to check against (the tables say what they have)
    run_names, sample_names = [s[0] for s in states], [s[1] for s in states]
    for ref in config.references:
        if ref != ALL_REFERENCES:
            resolve_sample(run_names, sample_names, ref, "reference sample")
    for s in config.samples or ():
        resolve_sample(run_names, sample_names, s)


def analysis_references(results, config: AnalysisConfig) -> list[str]:
    """The references ``config`` makes analyses against, one analysis each: ``['']`` with
    none, the one or the several named, or with ``'all'`` every sample of the results in the
    order of ``[analysis] samples`` (else the analysis's own order), named as the settings name
    them (``run:sample`` for one run's sample of several runs)."""
    refs = config.references
    if not refs:
        return [""]
    if ALL_REFERENCES not in refs:
        _check_names(results, config)
        return list(refs)
    if config.samples:
        return list(config.samples)
    states = _sample_states(results)
    if states is None:
        raise ValueError('[analysis] reference = "all": these results do not list their samples; '
                         "name the references")
    return [label(*state) for state in states]


def _sample_styles(up, labels: dict, shown: list, colours) -> dict:
    """Each sample's style for every figure of the analysis (by display label), so a sample
    keeps its colour whichever samples a figure shows, and whichever sample is the reference:
    ``[analysis.colours]`` (a sample's name, ``run:sample``, or a modified state's label) for
    those it names, the defaults for the rest, in the order ``shown`` (``[analysis] samples``'
    order: the first black). A name that is no sample of the analysis is left out with a
    warning."""
    from .coverage import sample_styles

    chosen = {}
    for name, colour in (colours or {}).items():
        if name in shown:
            chosen[name] = colour
            continue
        try:
            state = resolve_sample(up.run, up.sample, name, "sample")
        except ValueError as exc:
            if "several runs" in str(exc):
                raise ValueError(f"[analysis.colours] {exc}") from None
            log.warning("[analysis.colours] %s: no such sample in this analysis; left out",
                        name)
            continue
        chosen[labels[state]] = colour
    return sample_styles(shown, None, chosen)  # not the reference first (owner, 2026-11-04)


def _labelled(up, wanted: list, reference) -> list:
    """``wanted`` without the samples that have no labelled time point in the species table
    ``up`` (controls only, or every labelled spectrum left out), each named in a warning; a
    reference without one is refused."""
    timed = {(str(r), str(s)) for r, s, role in zip(up.run, up.sample, up.role, strict=True)
             if not role}
    labels = display_labels([s[0] for s in wanted], [s[1] for s in wanted])
    if reference is not None and reference not in timed:
        raise ValueError(f"the reference sample {labels[reference]!r} has no labelled time "
                         "point (controls only, or every labelled spectrum left out): set "
                         "[analysis] reference to a sample with labelled time points")
    for s in wanted:
        if s not in timed:
            log.warning("%s has no labelled time points (controls only, or every labelled "
                        "spectrum left out): left out of the analysis", labels[s])
    return [s for s in wanted if s in timed]


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
    listed = list(wanted)  # the settings' order, for the figures that compare nothing
    if ref_state is not None:
        wanted = [ref_state] + [s for s in wanted if s != ref_state]
    wanted = _labelled(up, wanted, ref_state)
    pairs = list(zip(up.run.astype(str), up.sample.astype(str), strict=True))
    up = up.select(np.array([p in set(wanted) for p in pairs], dtype=bool))
    sequences = config.sequences(base)
    # residue-level outputs only for a sample whose peptides make one protein (owner,
    # 2026-11-08: pooled unrelated peptides have no residues to resolve)
    level = residue_level_states(up, config.residue_level, sequences=sequences)
    # a pooled sample with candidates is several proteins: its residue-level outputs are those
    # of each form of the pool's minimum set (owner, 2026-11-09), never one map of the mixture
    pools = _pool_forms(results, up)
    pooled = {s for s in level if s[0] in pools}
    forms_on = config.residue_level is not False
    for state, why in level.items():
        if why and config.residue_level == "auto" and state not in pooled:
            log.info("%s: no residue-level outputs (%s); [analysis] residue_level = true makes "
                     "them", label(*state), why)
    if pooled and forms_on:
        log.info("pooled samples (%s): residue-level outputs for each form of the pool "
                 "(pooled_forms.csv), not for the mixture", ", ".join(label(*s) for s in
                                                                         sorted(pooled)))
    on = {state for state, why in level.items() if not why and state not in pooled}
    res_up = up.select(np.array([p in on for p in zip(
        up.run.astype(str), up.sample.astype(str), strict=True)], dtype=bool))
    # K=2 residue values from the peptides whose sets are best described by two populations,
    # with the share of the weight they carry (HISTORY 137)
    # else the bootstrap rule (HISTORY 139)
    gate = config.populations in ("best_set", "population_test")
    residues = (residue_map(res_up, sequences=sequences,
                            weight_exponent=config.weight_exponent,
                            modified=config.modified_peptides,
                            two_population_peptides=gate and 2 in config.ks,
                            population_test=config.populations == "population_test")
                if on else None)
    # modified forms kept apart are states of their own (e.g. 'WT pS12'), after their sample
    state_pairs = ([s for s in _states(residues) if s not in wanted]
                   if residues is not None else [])
    all_states = wanted + state_pairs
    labels = display_labels([s[0] for s in all_states], [s[1] for s in all_states])
    samples = [labels[s] for s in wanted]  # display labels, the reference first
    # the figures that compare nothing keep [analysis] samples' order (the reference where it
    # is listed, first when it is not), so they are the same against every reference (owner)
    in_order = [labels[s] for s in wanted if s not in listed] + [
        labels[s] for s in listed if s in set(wanted)]
    states_shown = [labels[s] for s in state_pairs]
    reference = labels[ref_state] if ref_state is not None else ""
    ref_name = label(*ref_state) if ref_state is not None else ""  # names it for the analyses
    styles = _sample_styles(up, labels, in_order + states_shown, config.colours)
    # the display labels of the samples with residue-level outputs, in the figures' order
    res_order = [labels[s] for s in wanted if s in on]
    res_order = [x for x in in_order if x in set(res_order)]
    coverage = choice = None
    length = None
    if residues is not None:
        coverage_time = (None if config.coverage_times == UNDEUTERATED
                         else config.coverage_times)
        try:
            coverage = coverage_table(res_up, sequences=sequences, count=config.redundancy,
                                      time=coverage_time)
        except NoPeptideAtTime as exc:
            log.warning("%s: coverage_times = \"all\" is used instead (set [analysis] "
                        "coverage_times to choose the times coverage is judged on)", exc)
            coverage = coverage_table(res_up, sequences=sequences, count=config.redundancy,
                                      time="all")
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
    tables: dict[str, dict] = {"species_uptake": up.columns()}
    if residues is not None:
        tables["residues"] = residues.columns()
        tables["coverage"] = coverage.columns()
    if choice is not None:
        tables["populations"] = choice.columns()
    if config.filters is not None:  # the spectra [analysis.filters] left out whole, and why
        tables["left_out_spectra"] = left_out_spectra(
            results, config.filters, exclude_flagged=config.exclude_flagged,
            controls=config.controls)
    # the figures take the tables with each state's display label as its sample
    up_d = _display(up, labels)
    residues_d = _display(residues, labels) if residues is not None else None
    coverage_d = _display(coverage, labels) if coverage is not None else None
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
    k1_raw = None
    pop_cmp = counts_d = None
    choices_cmp = shared_choices = k2_positions = reg_cmp = None
    k2_tested = False
    if config.reference:
        ref = ref_name
        best = best_sets(up)  # each species' number of populations (best_set_K)
        kw = {"exclude_flagged": config.exclude_flagged, "filters": config.filters,
              "alpha": config.alpha, "samples": [label(*s) for s in wanted],
              "match": config.peptide_match, "controls": config.controls, "best_sets": best,
              "two_population_sets": gate,
              # the averages of sets with two populations far apart (HISTORY 210) or three
              # and more
              "set_averages": set_average_ks(up)}
        comparisons["K=1 fD"] = compare_peptides(results, ref, **kw)
        k1_raw = comparisons["K=1 fD"]
        if 2 in config.ks:
            comparisons["K=2 rank 1 (low D) fD"] = compare_peptides(results, ref, K=2, rank=1,
                                                                    **kw)
            comparisons["K=2 rank 2 (high D) fD"] = compare_peptides(results, ref, K=2, rank=2,
                                                                     **kw)
            comparisons["K=2 rank 1 (low D) population"] = compare_peptides(
                results, ref, K=2, rank=1, quantity="population",
                choice=config.choice_limits, comparisons=list(comparisons.values()), **kw)
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
        res_cmp = None
        # the residues compared only where the reference has residue-level outputs
        if residues is not None and ref_state in on:
            # each sample's own values; whether a difference is significant is judged from the
            # peptides both samples measured at that time (owner, 2026-10-31; HISTORY 202), so a
            # peptide only one of them has cannot make one
            crit = comparisons["K=1 fD"].delta_crit
            pairs = paired_residue_maps(
                res_up, ref, match=config.peptide_match, sequences=sequences,
                weight_exponent=config.weight_exponent, modified=config.modified_peptides,
                two_population_peptides=gate and 2 in config.ks,
                population_test=config.populations == "population_test")
            pair_choice = ({s: choose_populations(m, config.choice_limits,
                                                  tiers=config.populations == "population_test")
                            for s, m in pairs.items()} if choice is not None else {})
            # and only where a covering peptide carries it: a safeguard against wrong error bars;
            # the residue test's other calls stay visible as residue_test_only (HISTORY 206)
            nterm = up.nterm_subtract
            res_cmp = gated_by_peptides(judged_on_shared(
                compare_residues(residues, ref, alpha=config.alpha, delta_crit=crit, choice=choice),
                concat_tables([compare_residues(m, ref, alpha=config.alpha, delta_crit=crit,
                                                choice=pair_choice.get(s))
                               for s, m in pairs.items()])), comparisons["K=1 fD"],
                nterm_subtract=nterm, modified=config.modified_peptides)
            tables["compare_residues"] = res_cmp.columns()
            if ann.regions:  # the regions compared from their residues, with the same gate
                reg_cmp = compare_regions(res_cmp, ann, comparisons["K=1 fD"], nterm_subtract=nterm,
                                          modified=config.modified_peptides)
                tables["compare_regions"] = reg_cmp.columns()
            if choice is not None:
                # K = 2 differences between samples are tested only with a bootstrap at K = 2 in
                # every run: without one most residues have no error and the few calls do not
                # reproduce the bootstrapped ones (owner, 2026-10-31; HISTORY 204)
                k2_tested = all(r.bootstrap_by_k.get(2) is not None for r in (
                    results.runs.values() if isinstance(results, RunSet) else [results]))
                if k2_tested:  # the two positions, each with its own K=2 threshold
                    k2_positions = {}
                    for rank, title in ((1, "K=2 rank 1 (low D) fD"),
                                        (2, "K=2 rank 2 (high D) fD")):
                        k2_crit = comparisons[title].delta_crit
                        k2_positions[rank] = gated_by_peptides(judged_on_shared(
                            compare_residues(residues, ref, K=2, rank=rank, alpha=config.alpha,
                                             delta_crit=k2_crit, choice=choice),
                            concat_tables([compare_residues(m, ref, K=2, rank=rank,
                                                            alpha=config.alpha,
                                                            delta_crit=k2_crit,
                                                            choice=pair_choice[s])
                                           for s, m in pairs.items()])), comparisons[title],
                            nterm_subtract=nterm, modified=config.modified_peptides)
                    tables["compare_residues_K2"] = _ranked(k2_positions)
                # where the map shows no position difference the states are the same, and the
                # populations are compared on positions shared by both samples (HISTORY 281)
                shown = list(k2_positions.values()) if k2_positions is not None else None
                pop_cmp = judged_on_shared(  # the K=1 threshold, divided by each separation
                    compare_populations(choice, ref, alpha=config.alpha, delta_crit=crit,
                                        positions=shown),
                    concat_tables([compare_populations(pair_choice[s], ref, alpha=config.alpha,
                                                       delta_crit=crit, positions=shown)
                                   for s in pairs]))
                if k2_tested:
                    pop_cmp = gated_by_peptides(
                        pop_cmp, comparisons["K=2 rank 1 (low D) population"],
                        nterm_subtract=nterm, modified=config.modified_peptides)
                pop_cols = pop_cmp.columns()
                if not k2_tested:  # the same columns as the tested case, empty
                    pop_cols["z"] = np.full(len(pop_cmp), np.nan)
                    pop_cols["significant"] = np.full(len(pop_cmp), np.nan, dtype=object)
                    pop_cols["peptide_support"] = np.full(len(pop_cmp), np.nan)
                    pop_cols["residue_test_only"] = np.full(len(pop_cmp), np.nan, dtype=object)
                    log.info("K=2 differences between samples (population share, positions) not "
                             "tested: no bootstrap at K=2 ([run.bootstrap] ks = [1, 2], or hxex "
                             "resume); residue_significance_K2 shows where the number of "
                             "populations differs")
                tables["compare_populations"] = pop_cols
                choices_cmp = compare_choices(choice, ref)
                tables["compare_choices"] = choices_cmp.columns()
                # the K = 2 map decides one or two on the shared peptides too, as its test does
                shared_choices = concat_tables([compare_choices(pair_choice[s], ref)
                                                for s in pairs])
        comparisons = {title: _display(c, labels, reference) for title, c in comparisons.items()}
        if res_cmp is not None:
            res_cmp = _display(res_cmp, labels, reference)
        if reg_cmp is not None:
            reg_cmp = _display(reg_cmp, labels, reference)
        if choices_cmp is not None:
            choices_cmp = _display(choices_cmp, labels, reference)
            if shared_choices is not None:
                shared_choices = _display(shared_choices, labels, reference)
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
    if ann.regions and residues is not None:
        regions = region_summary(residues, ann)
        tables["regions"] = _columns(regions)
        regions = _display(regions, labels)
    pooled_order = [s for s in in_order if any(labels[p] == s for p in pooled)]
    # the forms pages and the head-to-head pages share one y axis (owner, 2026-11-09): up to
    # the highest residue or peptide fD with its error
    on_pooled = np.array([(str(u), str(s)) in pooled for u, s in zip(up.run, up.sample,
                                                                       strict=True)], dtype=bool)
    peptide_top = (_highest(up_d.select(on_pooled & (up.role == "")), times)
                   if pooled and config.head_to_head is not False else None)
    form_jobs, form_top = (_pooled_form_outputs(
        pools, up, pooled, tables, config, labels=labels, styles=styles, order=pooled_order,
        ref_state=ref_state, k1=k1_raw, times=times, scale=scale, floor=peptide_top)
        if pooled and forms_on else ([], peptide_top))
    if pooled:  # a coverage map per form of the pool: peptide-level, whatever residue_level
        form_jobs += _pooled_coverage_jobs(pools, up, pooled, config, labels=labels,
                                           order=pooled_order, times=times, scale=scale)
    if pooled and config.head_to_head is not False:
        form_jobs += _head_to_head_outputs(
            results, pools, up, up_d, pooled, tables, config, styles=styles,
            order=pooled_order, times=times, scale=scale, top=form_top)
    up, residues, coverage, choice = up_d, residues_d, coverage_d, choice_d
    states = states_shown

    jobs = []  # (file stem, function drawing the figure)
    page = config.page_size
    per_row = max(1, int((page[0] - 0.8) // COVERAGE_CELL[0]))
    per_page = per_row * max(1, int(page[1] // COVERAGE_CELL[1]))
    # coverage: peptide-level, so every sample (owner, 2026-11-09), but a pooled sample, whose
    # peptides belong to several proteins: those get a map per form (pooled_coverage)
    pooled_shown = {labels[s] for s in pooled}
    for s in [x for x in in_order if x not in pooled_shown]:
        # the undeuterated control, then the times, wrapped and paged
        points = (["UN"] if up.where(sample=s, role=UNDEUT).any() else []) + list(times)
        for n, chunk in enumerate(_pages(points, per_page)):
            jobs.append((_page_stem(f"coverage_{s}", n), lambda s=s, chunk=chunk: coverage_figure(
                up, samples=[s], times=[t for t in chunk if t != "UN"], length=length,
                columns=per_row, undeuterated="UN" in chunk, cell=COVERAGE_CELL, **when)))
    if coverage is not None:
        jobs.append(("redundancy_resolution", lambda: plot_redundancy_resolution(
            coverage, samples=res_order, marks=marks, styles=styles)))
    runs_read = results.runs.values() if isinstance(results, RunSet) else [results]
    booted = all(r.bootstrap is not None or r.bootstrap_by_k for r in runs_read)
    for s in (res_order + states if residues is not None else []):
        for t in times:
            when_ = _time_label(t, scale)
            if choice is not None and choice.where(sample=s, time=t).any():
                jobs.append((f"choice_{s}_{when_}", lambda s=s, t=t: plot_population_choice(
                    choice, s, t, marks=marks, length=length, bootstrap=booted, **when)))
            elif residues.where(sample=s, time=t).any():
                jobs.append((f"profile_{s}_{when_}", lambda s=s, t=t: plot_residue_profile(
                    residues, s, t, marks=marks, length=length, bootstrap=booted, **when)))
    # the bands: +-delta_crit / 2 of the peptide comparisons (one value for every pair of
    # samples), so two samples' bands part where they differ by more (owner); without a
    # reference, +-1 standard error
    k1_crit = comparisons["K=1 fD"].delta_crit if "K=1 fD" in comparisons else None
    # every sample's fD by residue on one axis, none a reference (owner), one y axis for every
    # time so the figures compare (owner, 2026-11-04)
    k1 = (residues.where(K=1, rank=1) & np.isin(residues.time, list(times))
          if residues is not None else np.zeros(0, dtype=bool))
    k1 &= np.isfinite(residues.fD) if residues is not None else k1
    highest = (float(np.max(residues.fD[k1] + np.nan_to_num(residues.fD_se[k1])))
               if k1.any() else None)
    for t in (times if residues is not None else []):
        with_values = [s for s in res_order + states
                       if residues.where(sample=s, time=t, K=1, rank=1).any()]
        if with_values:
            jobs.append((f"residues_all_{_time_label(t, scale)}",
                         lambda t=t, w=with_values: plot_residue_samples(
                             residues, w, t, styles=styles, annotations=ann, length=length,
                             top=highest, delta_crit=k1_crit, **when)))
    if (config.populations == "population_test" and residues is not None
            and (residues.population_tier != "").any()):
        tiered = (residues.K == 1) & (residues.population_tier != "")
        points = [np.nan] + sorted({float(t) for t in residues.time[tiered]
                                    if np.isfinite(t)})  # NaN: the controls' rows
        for n, chunk in enumerate(_map_pages(config.page_size, points,
                                             len(res_order + states), 2.2)):
            jobs.append((_page_stem("population_tiers", n), lambda chunk=chunk:
                         plot_population_tiers(residues, res_order + states, length=length,
                                               times=chunk, **when)))
    if config.reference:
        others = [s for s in samples if s != reference]
        # the residue comparisons: the samples with residue-level outputs, and the states,
        # against a reference that has them
        residue_others = ([s for s in others if s in set(res_order)] + states
                          if res_cmp is not None else [])
        # the Woods plot: page 1, K=1 and where the number of populations differs; page 2,
        # the K=2 comparisons (made only where both have two populations), the population in
        # fD units: its difference times the separation, judged by the K=1 delta_crit
        # (HISTORY 135, 137), drawn as the high-D share so that above zero (red) is more D,
        # as on the fD panels (HISTORY 221)
        short = {"K=2 rank 1 (low D) fD": "K=2 low-D fD", "K=2 rank 2 (high D) fD":
                 "K=2 high-D fD", "K=2 rank 1 (low D) population": "K=2 high-D population"}
        k2_stack = [(short.get(title, title), c.scaled_by_separation().as_high_share())
                    for title, c in comparisons.items() if title != "K=1 fD"]
        # the population first, then the positions: the order of the K=2 significance map
        k2_stack.sort(key=lambda panel: not panel[0].endswith("population"))
        for t in times:
            when_ = _time_label(t, scale)
            only = choice if config.overlay_two_only else None
            shown = [s for s in residue_others
                     if _has_overlay(residues, s, t, config.overlay, only)]
            per_page = int(config.page_size[1] // OVERLAY_PANEL)  # samples on a page
            if shown and _has_overlay(residues, reference, t, config.overlay, only):
                for n, part in enumerate(_pages(shown, per_page)):
                    jobs.append((_page_stem(f"overlay_residues_{when_}", n),
                                 lambda t=t, part=part, only=only: residue_overlay_figure(
                                     residues, reference, part, t, show=config.overlay,
                                     choice=only, annotations=ann, length=length,
                                     styles=styles, **when)))
            # the populations, drawn as the K=2 values are: each sample where it keeps two; the
            # high-D share, as the Woods plot draws it (above = more D; owner, 2026-11-02)
            ref_pop = choice is not None and _has_overlay(residues, reference, t, ("pop_high",),
                                                          choice)
            with_pop = [s for s in residue_others if choice is not None and (
                ref_pop or _has_overlay(residues, s, t, ("pop_high",), choice))]
            for n, part in enumerate(_pages(with_pop, per_page) if with_pop else []):
                jobs.append((_page_stem(f"overlay_populations_{when_}", n),
                             lambda t=t, part=part: residue_overlay_figure(
                                 residues, reference, part, t, show=("pop_high",),
                                 choice=choice, annotations=ann, length=length, styles=styles,
                                 **when)))
            for s in (o for o in others if o in residue_others):  # Woods: on the residues
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
            map_times = sorted({float(t) for t in res_cmp.time[res_cmp.where(sample=compared)]})
            for n, chunk in enumerate(_map_pages(config.page_size, map_times, len(compared),
                                                 2.4)):
                jobs.append((_page_stem("residue_significance", n),
                             lambda compared=compared, chunk=chunk: plot_residue_significance(
                                 res_cmp, compared, length=length, times=chunk, **when)))
        if reg_cmp is not None and len(reg_cmp):
            in_regions = [s for s in residue_others if reg_cmp.where(sample=s).any()]
            region_times = sorted({float(t) for t in reg_cmp.time[reg_cmp.where(
                sample=in_regions)]})
            for n, chunk in enumerate(_map_pages(config.page_size, region_times,
                                                 len(in_regions), 2.8)):
                jobs.append((_page_stem("region_significance", n),
                             lambda in_regions=in_regions, chunk=chunk: plot_region_significance(
                                 reg_cmp, in_regions, times=chunk, **when)))
        with_two = [s for s in residue_others
                    if shared_choices is not None and shared_choices.where(sample=s).any()]
        positions = k2_positions or {}
        sections = 1 + sum(positions.get(r) is not None and len(positions[r]) > 0
                           for r in (1, 2))
        for s in with_two:  # one figure per sample: five samples on one page overwhelm (owner)
            k2_times = sorted({float(t) for t in shared_choices.time[
                shared_choices.where(sample=s)]})
            for n, chunk in enumerate(_map_pages(config.page_size, k2_times, sections,
                                                 2.4 + MAP_ROW * sections)):
                jobs.append((_page_stem(f"residue_significance_K2_{s}", n),
                             lambda s=s, positions=positions, chunk=chunk:
                             plot_residue_significance_k2(
                                 shared_choices, pop_cmp, positions.get(1), positions.get(2),
                                 [s], length=length, blocks=res_cmp, own_choices=choices_cmp,
                                 times=chunk, **when)))
        k1 = comparisons["K=1 fD"]
        drawn = list(times)
        for s in others:
            if drawn and k1.where(sample=s, time=drawn).any():
                jobs.append((f"butterfly_{s}", lambda s=s: plot_butterfly(
                    k1, s, times=[t for t in drawn if k1.where(sample=s, time=t).any()],
                    **when)))
    if protection is not None:
        for s in in_order:
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
                    regions, t, samples=in_order, styles=styles, **when)))
    jobs.extend(form_jobs)

    return tables, jobs


def _pool_forms(results, up) -> dict:
    """The minimum set of every pooled run with candidates (``[input] pooled_fasta``) whose
    samples are in ``up``: run -> :class:`hxex.analysis.pooled.PooledForms`, from the peptides
    of all its samples."""
    from ..analysis.pooled import pooled_forms
    from ..io.pooled import CANDIDATES_RECORD

    runs = results.runs if isinstance(results, RunSet) else {"": results}
    out = {}
    pep = up.kind == "peptide"
    for name, r in runs.items():
        settings = getattr(r, "settings", None)
        record = ((settings.spectra_source or {}).get(CANDIDATES_RECORD)
                  if settings is not None else None)
        rows = np.flatnonzero(pep & (up.run.astype(str) == name))
        if not record or not len(rows):
            continue
        keys = list(zip(up.peptide[rows].astype(str), up.start[rows].astype(int),
                        up.end[rows].astype(int), up.modification[rows].astype(str),
                        strict=True))
        forms = pooled_forms(keys, record, run=name, registry=settings.registry)
        if forms.alternatives:
            log.warning("%s: the pool's minimum set is not unique (also: %s); the forms of the "
                        "first are used", label(name, "") or "the run",
                        "; ".join(", ".join(a) for a in forms.alternatives))
        if not forms.exact:
            log.warning("%s: too many candidates to search every minimum set; the set was "
                        "found one form at a time and may not be the smallest",
                        label(name, "") or "the run")
        out[name] = (forms, record, settings.registry)
    return out


def _joined(first: dict | None, second: dict) -> dict:
    """Two tables' columns one after the other (a column one lacks: empty there)."""
    if not first:
        return second
    n1, n2 = len(next(iter(first.values()))), len(next(iter(second.values())))
    out = {}
    for name in dict.fromkeys([*first, *second]):
        a = first.get(name, np.full(n1, np.nan, dtype=object))
        b = second.get(name, np.full(n2, np.nan, dtype=object))
        out[name] = np.concatenate([np.asarray(a, dtype=object) if a.dtype != b.dtype else a,
                                    np.asarray(b, dtype=object) if a.dtype != b.dtype else b])
    return out


def _highest(table, times) -> float | None:
    """The highest K=1 fD plus its error of ``table`` (a species table or a residue map) at
    ``times``; None without any."""
    k1 = ((table.K == 1) & (table.rank == 1) & np.isfinite(table.fD)
          & np.isin(table.time, list(times)))
    return (float(np.max(table.fD[k1] + np.nan_to_num(table.fD_se[k1]))) if k1.any()
            else None)


def _pooled_form_outputs(pools: dict, up, pooled: set, tables: dict, config: AnalysisConfig, *,
                         labels: dict, styles: dict, order: list, ref_state, k1, times,
                         scale: str, floor: float | None = None) -> tuple[list, float | None]:
    """The residue-level outputs of the pooled samples (owner, 2026-11-09): the forms of each
    pool's minimum set (``pooled_forms`` table), each form's residue values (rows of the
    ``residues`` table, the sample named ``'<sample> <form>'``), each form compared with the
    same form of the reference (rows of ``compare_residues``: judged on the peptides both
    measured, then gated by the peptide comparison of the form's peptides), and one figure
    page per class of forms and time (:func:`hxex.report.pooled.plot_pooled_forms`). Adds
    the tables to ``tables``; returns the figure jobs and the top of their y axis (the
    head-to-head figure shares it)."""
    from ..analysis.pooled import form_class, form_comparison, form_sequences, form_uptake
    from .pooled import FORM_PANEL, plot_pooled_forms

    forms = {run: f for run, (f, _, _) in pools.items()}
    rows = np.array([(str(u), str(s)) in pooled for u, s in zip(up.run, up.sample,
                                                                  strict=True)], dtype=bool)
    fu, base = form_uptake(up.select(rows & (up.kind == "peptide")), forms)
    sequences: dict[str, str] = {}
    for f, record, registry in pools.values():
        sequences.update(form_sequences(f, record, base, registry))
    gate = config.populations in ("best_set", "population_test")
    rm = residue_map(fu, sequences=sequences or None, weight_exponent=config.weight_exponent,
                     modified="combined", two_population_peptides=gate and 2 in config.ks,
                     population_test=config.populations == "population_test")
    tables["pooled_forms"] = _joined(None, {})
    for f, _, _ in pools.values():
        tables["pooled_forms"] = _joined(tables["pooled_forms"], f.columns())
    tables["residues"] = _joined(tables.get("residues"), rm.columns())
    # the display label of each form state: its sample's, then the form
    form_labels = {state: f"{labels.get((state[0], s), label(state[0], s))} "
                          f"{state[1][len(s) + 1:]}" for state, s in base.items()}
    if k1 is not None and ref_state in pooled:
        crit = k1.delta_crit
        k1_forms = form_comparison(k1, forms)
        nterm = up.nterm_subtract
        found = []
        ref_forms = forms[ref_state[0]].forms
        for f in ref_forms:
            ref_f = (ref_state[0], f"{ref_state[1]} {f.name}")
            mine = [s for s in base if s[1][len(base[s]) + 1:] == f.name]
            if ref_f not in mine or len(mine) < 2:
                continue
            keep = np.array([(str(u), str(s)) in set(mine) for u, s in zip(fu.run, fu.sample,
                                                                            strict=True)])
            fu_f = fu.select(keep)
            in_map = np.array([(str(u), str(s)) in set(mine) for u, s in zip(rm.run, rm.sample,
                                                                              strict=True)])
            rm_f = dataclasses.replace(rm, **{x.name: getattr(rm, x.name)[in_map]
                                              for x in fields(rm)})
            name = label(*ref_f)
            pairs = paired_residue_maps(
                fu_f, name, match="sequence", modified="combined", sequences=sequences or None,
                weight_exponent=config.weight_exponent,
                two_population_peptides=gate and 2 in config.ks,
                population_test=config.populations == "population_test")
            found.append(gated_by_peptides(judged_on_shared(
                compare_residues(rm_f, name, alpha=config.alpha, delta_crit=crit),
                concat_tables([compare_residues(m, name, alpha=config.alpha, delta_crit=crit)
                               for m in pairs.values()])), k1_forms, nterm_subtract=nterm,
                modified="combined"))
        if found:
            tables["compare_residues"] = _joined(tables.get("compare_residues"),
                                                 concat_tables(found).columns())
    # the figures: one page per class of forms (forms of no single class last) and time
    shown = _display(rm, form_labels)
    tops = [x for x in (_highest(shown, times), floor) if x is not None]
    top = max(tops) if tops else None
    first, record, registry = next(iter(pools.values()))
    groups: dict[str, list] = {}
    for f in first.forms:
        groups.setdefault(form_class(f, record), []).append(f)
    groups = dict(sorted(groups.items(), key=lambda g: (g[0] == "", g[0])))
    per_page = max(1, int((config.page_size[1] - 0.7) // FORM_PANEL))
    crit = k1.delta_crit if k1 is not None else None
    jobs = []
    for t in times:
        when_ = _time_label(t, scale)
        for cls, members in groups.items():
            with_values = [f for f in members if any(
                shown.where(sample=f"{s} {f.name}", time=t, K=1, rank=1).any() for s in order)]
            for n, part in enumerate(_pages(with_values, per_page)):
                stem = _page_stem(f"pooled_forms_{_stem(cls or 'other')}_{when_}", n)
                jobs.append((stem, lambda t=t, part=part: plot_pooled_forms(
                    shown, part, order, t, styles=styles, nterm_subtract=up.nterm_subtract,
                    top=top, delta_crit=crit, time_scale=scale, registry=registry)))
    return jobs, top


def _pooled_coverage_jobs(pools: dict, up, pooled: set, config: AnalysisConfig, *,
                          labels: dict, order: list, times, scale: str) -> list:
    """One coverage map per form of each pool's minimum set (owner, 2026-11-09): the peptides
    the form may have given, every pooled sample a row, the undeuterated control and the
    times as columns (paged as the samples' maps), the form's own peptides outlined in black
    and white;
    file ``pooled_coverage_<form>``."""
    from ..analysis.pooled import form_uptake

    forms = {run: f for run, (f, _, _) in pools.items()}
    rows = np.array([(str(u), str(s)) in pooled for u, s in zip(up.run, up.sample,
                                                                  strict=True)], dtype=bool)
    fu, base = form_uptake(up.select(rows & (up.kind == "peptide")), forms)
    form_labels = {state: f"{labels.get((state[0], s), label(state[0], s))} "
                          f"{state[1][len(s) + 1:]}" for state, s in base.items()}
    shown = _display(fu, form_labels)
    page = config.page_size
    per_row = max(1, int((page[0] - 0.8) // COVERAGE_CELL[0]))
    per_page = per_row * max(1, int(page[1] // (COVERAGE_CELL[1] * max(len(order), 1))))
    jobs = []
    first = next(iter(pools.values()))[0]
    for form in first.forms:
        samples = [f"{s} {form.name}" for s in order
                   if shown.where(sample=f"{s} {form.name}").any()]
        if not samples:
            continue
        has_un = any(shown.where(sample=s, role=UNDEUT).any() for s in samples)
        points = (["UN"] if has_un else []) + list(times)
        own = set(form.own)
        for n, chunk in enumerate(_pages(points, per_page)):
            jobs.append((_page_stem(f"pooled_coverage_{_stem(form.name)}", n),
                         lambda samples=samples, chunk=chunk, own=own, form=form:
                         coverage_figure(
                             shown, samples=samples, times=[t for t in chunk if t != "UN"],
                             columns=per_row, undeuterated="UN" in chunk, cell=COVERAGE_CELL,
                             outlined=own, time_scale=scale,
                             title=f"{form.name}: its own peptides outlined in black and white")))
    return jobs


def _head_to_head_outputs(results, pools: dict, up, up_d, pooled: set, tables: dict,
                          config: AnalysisConfig, *, styles: dict, order: list, times,
                          scale: str, top: float | None) -> list:
    """The pooled sequences measured over one range, head to head (owner, 2026-11-09):
    ``compare_sequences`` (each panel's sequences against its first, per sample and time)
    and figure pages ``head_to_head_<time>`` (:func:`hxex.report.pooled.plot_head_to_head`),
    their y axis from 0 to ``top`` (shared with the forms pages). ``[analysis] head_to_head``
    chooses the ranges. Adds the table to ``tables``; returns the figure jobs."""
    from ..analysis.compare import compare_sequences
    from ..analysis.pooled import head_to_head_panels
    from .pooled import HEAD_TO_HEAD_COLUMNS, HEAD_TO_HEAD_PANEL, plot_head_to_head

    _, record, registry = next(iter(pools.values()))
    rows = np.flatnonzero([(str(u), str(s)) in pooled and k == "peptide" and not r
                           for u, s, k, r in zip(up.run, up.sample, up.kind, up.role,
                                                 strict=True)])
    peptides = [(str(up.peptide[i]), int(up.start[i]), int(up.end[i])) for i in rows
                if not str(up.modification[i])]
    panels = head_to_head_panels(peptides, record, config.head_to_head, registry=registry)
    if not panels:
        log.info("head to head: no range with sequences to show ([analysis] head_to_head = "
                 "%r)", config.head_to_head)
        return []
    comparison = compare_sequences(
        results, panels, alpha=config.alpha, exclude_flagged=config.exclude_flagged,
        filters=config.filters, controls=config.controls, set_averages=set_average_ks(up))
    tables["compare_sequences"] = comparison.columns()
    rows_per_page = max(1, int((config.page_size[1] - 0.4) // HEAD_TO_HEAD_PANEL[1]))
    per_page = HEAD_TO_HEAD_COLUMNS * rows_per_page
    jobs = []
    for t in times:
        when_ = _time_label(t, scale)
        for n, part in enumerate(_pages(panels, per_page)):
            jobs.append((_page_stem(f"head_to_head_{when_}", n), lambda t=t, part=part:
                         plot_head_to_head(up_d, part, order, t, styles=styles, top=top,
                                           time_scale=scale)))
    return jobs


@dataclasses.dataclass(frozen=True)
class AnalysisOutputs:
    """What an analysis computes before anything is written: its ``tables`` (name -> columns),
    its ``figures`` (file stem -> a function drawing the figure, called when wanted), and the
    ``config`` used (``ks`` cut to the K values fitted, the populations rule decided)."""

    tables: dict
    figures: list
    config: AnalysisConfig


def analysis_outputs(results, config: AnalysisConfig, *, base: str | Path = ".", review=None,
                     cancel: CancelHook | None = None) -> AnalysisOutputs:
    """Compute the analysis :func:`write_analysis` writes, without writing it (the window draws
    its figures one at a time, from the same list): ``review`` and ``config.exclude`` applied
    first, in memory, as there."""
    if not config.single_reference:
        raise ValueError(f"one reference at a time: {', '.join(config.references)} make an "
                         "analysis each (analysis_references, AnalysisConfig.with_reference; "
                         "hxex analyse writes a folder for each)")
    if config.exclude or review:  # rejected in memory, the bundle's fits kept
        from ..exclusions import exclude_from_results
        from ..review_table import apply_view, refuse_unfitted

        def view(r):
            r = exclude_from_results(apply_view(r, review), config.exclude)
            refuse_unfitted(r)
            return r

        if isinstance(results, RunSet):
            results = RunSet({name: view(r) for name, r in results.runs.items()},
                             merge=results.merge)
        else:
            results = view(results)
    if config.exclude_times:
        results = _without_times(results, config.exclude_times)
    config = _fitted_ks(config, results.runs if isinstance(results, RunSet) else {"": results})
    _check_names(results, config)
    rule = population_rule(config.populations, results)
    if config.populations is None:
        log.info("the number of populations: %s", "the population test of the sets"
                 if rule == "population_test" else "best_set_K (the run has no population test)")
    config = dataclasses.replace(config, populations=rule)
    runs = results.runs if isinstance(results, RunSet) else {"": results}
    match = resolve_match(config.peptide_match, runs)
    if config.peptide_match == "auto" and match == "sequence":
        log.info("peptides compared by sequence: pooled data, where two sequences at one range "
                 "are different proteins ([analysis] peptide_match)")
    config = dataclasses.replace(config, peptide_match=match)
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
        frag_tables, frag_jobs = fragment_outputs(results, config.fragments,
                                                  page_size=config.page_size)
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
    return AnalysisOutputs(_with_pooled_identities(tables, runs), jobs, config)


def _with_pooled_identities(tables: dict, runs: Mapping) -> dict:
    """The analysis tables of pooled results with candidates (``[input] pooled_fasta``), each
    table naming peptides (``peptide`` and ``start``) gaining the identity columns after
    ``end`` (:data:`hxex.report.tables.IDENTITY_COLUMNS`), from its run's candidates (a merged
    row, with an empty run, from the first run that has them); other tables unchanged."""
    from ..io.pooled import identities

    records = {name: (r.settings.spectra_source or {}) for name, r in runs.items()
               if getattr(r, "settings", None) is not None
               and (r.settings.spectra_source or {}).get("pooled_candidates")}
    if not records:
        return tables
    registry = next(iter(runs.values())).settings.registry
    fallback = next(iter(records))
    out = {}
    for name, cols in tables.items():
        if not ({"peptide", "start", "end"} <= set(cols)) or not len(cols["peptide"]):
            out[name] = cols
            continue
        run = (cols["run"].astype(str) if "run" in cols
               else np.full(len(cols["peptide"]), "", dtype=object))
        keys = [(str(r) if str(r) in records else fallback, str(p), int(s))
                for r, p, s in zip(run, cols["peptide"], cols["start"], strict=True)]
        found = {name_: identities(records[name_], {(p, s) for n, p, s in keys if n == name_},
                                   registry=registry) or {}
                 for name_ in {k[0] for k in keys}}
        rows = [found[n].get((p, s)) for n, p, s in keys]
        new = {
            "identity": [x.identity if x else "" for x in rows],
            "identity_level": [x.level if x else "" for x in rows],
            "candidates": [";".join(x.candidates) if x else "" for x in rows],
            "classes": [";".join(x.classes) if x else "" for x in rows],
            "position_offset": [x.offset_text if x else "" for x in rows],
        }
        table = {}
        for col, values in cols.items():
            table[col] = values
            if col == "end":
                table.update({n: np.array(v, dtype=str) for n, v in new.items()})
        out[name] = table
    return out


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
    outputs = analysis_outputs(results, config, base=base, review=review, cancel=cancel)
    tables, jobs, config = outputs.tables, outputs.figures, outputs.config

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
