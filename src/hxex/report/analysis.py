# SPDX-License-Identifier: GPL-3.0-or-later
"""Write a whole post-fitting analysis (docs/ANALYSIS.md) from results: the tables as CSV files and
the figures, as :class:`hxex.analysis.AnalysisConfig` says (what ``hxex analyse`` runs).

Tables: ``species_uptake.csv``, ``residues.csv``, ``coverage.csv``, ``populations.csv`` (the
choice between one and two populations; with K=2), and with a reference sample
``compare_peptides.csv`` (K=1 fD, and with K=2 each rank's fD and the rank-1 population),
``compare_residues.csv``, ``compare_populations.csv`` and ``compare_choices.csv``, with
regions ``regions.csv``, and for several runs (a :class:`hxex.analysis.RunSet`) ``run_settings.csv``
(where their settings differ) and ``run_identity.csv`` (species whose identity differs).
Figures: ``coverage_<sample>``, ``redundancy_resolution``, ``choice_<sample>_<time>`` (or
``profile_<sample>_<time>`` without K=2), and with a reference ``compare_residues_<time>`` (the
values of ``config.overlay``, :func:`hxex.report.residue_overlay_figure`),
``compare_populations_<time>``, ``woods_<sample>_<time>``, ``butterfly_<sample>`` and
``regions_<time>``. For a run with fragment series (:class:`hxex.analysis.FragmentAnalysis`;
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
from dataclasses import fields
from pathlib import Path

import numpy as np

from ..analysis import (
    AnalysisConfig,
    RunSet,
    check_identity,
    choose_populations,
    compare_choices,
    compare_peptides,
    compare_populations,
    compare_residues,
    compare_settings,
    coverage_table,
    region_summary,
    residue_map,
    species_uptake,
)
from ..analysis.residues import NoPeptideAtTime
from ..analysis.residues import states as _states
from ..analysis.runs import display_labels, label, resolve_sample
from ..analysis.settings import OVERLAY_VALUES, UNDEUTERATED
from ..hooks import CancelHook, ProgressHook, check_cancel, report
from ..io.outputs import replace_outputs
from ..io.times import format_time
from .compare import (
    plot_butterfly,
    plot_region_bars,
    residue_comparison_figure,
    residue_overlay_figure,
    woods_figure,
)
from .coverage import coverage_figure, plot_redundancy_resolution
from .fragments import fragment_outputs
from .protection import (
    protection_difference_figure,
    protection_figure,
)
from .residues import plot_population_choice, plot_residue_profile
from .tables import write_csv

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


def _has_overlay(residues, sample: str, time: float, show, choice=None) -> bool:
    """Whether ``sample`` has a residue value at ``time`` among the overlay values ``show``
    (the K=2 ones only where ``choice``, if given, keeps two populations)."""
    for k, rank in (OVERLAY_VALUES[v] for v in show):
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
    residues = residue_map(up, sequences=sequences, weight_exponent=config.weight_exponent,
                           modified=config.modified_peptides)
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
    choice = choose_populations(residues, config.choice) if 2 in config.ks else None
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
    # the figures take the tables with each state's display label as its sample
    up_d, residues_d, coverage_d = (_display(t, labels) for t in (up, residues, coverage))
    choice_d = _display(choice, labels) if choice is not None else None
    if isinstance(results, RunSet):
        tables["run_settings"] = compare_settings(results).columns()
        # the identity check compares the runs, so merged samples are looked at run by run
        unmerged = up if not results.merge else species_uptake(
            results.unmerged(), ks=config.ks, exclude_flagged=config.exclude_flagged,
            filters=config.filters, controls=config.controls)
        tables["run_identity"] = check_identity(unmerged).columns()
    comparisons = {}
    pop_cmp = None
    if config.reference:
        ref = ref_name
        kw = {"exclude_flagged": config.exclude_flagged, "filters": config.filters,
              "alpha": config.alpha, "samples": [label(*s) for s in wanted],
              "match": config.peptide_match, "controls": config.controls}
        comparisons["K=1 fD"] = compare_peptides(results, ref, **kw)
        if 2 in config.ks:
            comparisons["K=2 rank 1 (low D) fD"] = compare_peptides(results, ref, K=2, rank=1,
                                                                    **kw)
            comparisons["K=2 rank 2 (high D) fD"] = compare_peptides(results, ref, K=2, rank=2,
                                                                     **kw)
            comparisons["K=2 rank 1 (low D) population"] = compare_peptides(
                results, ref, K=2, rank=1, quantity="population", **kw)
        rows = {}
        for title, c in comparisons.items():
            cols = c.columns()
            n = len(c)
            cols = {"comparison": np.array([title] * n), **cols,
                    "delta_crit": np.full(n, c.delta_crit)}
            for name, values in cols.items():
                rows.setdefault(name, []).append(values)
        tables["compare_peptides"] = {name: np.concatenate(v) for name, v in rows.items()}
        res_cmp = compare_residues(residues, ref, alpha=config.alpha,
                                   delta_crit=comparisons["K=1 fD"].delta_crit, choice=choice)
        tables["compare_residues"] = _columns(res_cmp)
        if choice is not None:
            pop_cmp = compare_populations(
                choice, ref, alpha=config.alpha,
                delta_crit=comparisons["K=2 rank 1 (low D) population"].delta_crit)
            tables["compare_populations"] = _columns(pop_cmp)
            tables["compare_choices"] = compare_choices(choice, ref).columns()
            pop_cmp = _display(pop_cmp, labels, reference)
        comparisons = {title: _display(c, labels, reference) for title, c in comparisons.items()}
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
    for s in samples + states:
        for t in times:
            when_ = _time_label(t, scale)
            if choice is not None and choice.where(sample=s, time=t).any():
                jobs.append((f"choice_{s}_{when_}", lambda s=s, t=t: plot_population_choice(
                    choice, s, t, marks=marks, length=length, **when)))
            elif residues.where(sample=s, time=t).any():
                jobs.append((f"profile_{s}_{when_}", lambda s=s, t=t: plot_residue_profile(
                    residues, s, t, marks=marks, length=length, **when)))
    if config.reference:
        others = [s for s in samples if s != reference]
        residue_others = others + states  # the residue comparisons include the states
        stack = list(comparisons.items())
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
            with_pop = [s for s in residue_others
                        if pop_cmp is not None and pop_cmp.where(sample=s, time=t).any()]
            if with_pop:
                jobs.append((f"compare_populations_{when_}",
                             lambda t=t, with_pop=with_pop: residue_comparison_figure(
                                 pop_cmp, with_pop, t, annotations=ann, length=length,
                                 **when)))
            for s in others:
                if all(c.where(sample=s, time=t).any() for _, c in stack):
                    jobs.append((f"woods_{s}_{when_}", lambda s=s, t=t: woods_figure(
                        stack, s, t, annotations=ann, length=length, **when)))
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
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> list[Path]:
    """Compute the analysis of ``results`` (:class:`hxex.report.Results`, or a
    :class:`hxex.analysis.RunSet` to compare separate runs) and write its tables and figures into
    ``directory``; returns the files. Relative FASTA paths are taken from
    ``base``. ``progress(done, total)`` counts the files."""
    directory = Path(directory)
    if config.exclude_times:
        results = _without_times(results, config.exclude_times)
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
                            filters=config.filters, controls=config.controls)
        if len(up):
            tables, jobs = _peptide_outputs(results, config, base, up, cancel=cancel)
    tables.update(frag_tables)
    jobs += frag_jobs

    total = len(tables) + len(jobs)
    names = [f"{name}.csv" for name in tables] + [f"{_stem(stem)}.{config.format}"
                                                   for stem, _ in jobs]
    # as the tables and figures folders: the files of the earlier analysis go, yours stay
    with replace_outputs(directory, names) as written:
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
    log.info("analysis: %d tables and %d figures in %s", len(tables), len(jobs), directory)
    return written
