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
``closure.csv`` and per parent ``ladder_<sample>_<range>_z<charge>_<ion>_z<fragment charge>``,
``closure_..._<n ion>+<c ion>`` and ``residue_rates_...``. With ``protection`` (opt-in,
:mod:`hxex.analysis.protection`): ``protection.csv``, ``protection_by_time.csv`` and the
figures ``protection_<sample>``, and with a reference ``compare_protection.csv`` and
``protection_difference_<sample>``; with ``protection_residues`` (opt-in) the residue level:
``protection_residues.csv``, the
figure ``protection_residues`` and with a reference ``compare_protection_residues.csv`` and
``protection_residue_comparison_<sample>``. Characters a file name cannot
hold (the ``:`` of ``run:sample``) become ``_``.
"""

from __future__ import annotations

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
from ..analysis.settings import OVERLAY_VALUES, UNDEUTERATED
from ..hooks import CancelHook, ProgressHook, check_cancel, report
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
    residue_protection_comparison_figure,
    residue_protection_figure,
)
from .residues import plot_population_choice, plot_residue_profile
from .tables import write_csv

log = logging.getLogger(__name__)


def _columns(table) -> dict[str, np.ndarray]:
    return {f.name: getattr(table, f.name) for f in fields(table)
            if isinstance(getattr(table, f.name), np.ndarray)}


def _time_label(time: float) -> str:
    """A time for a file name: ``5min``, ``20h``, ``4s`` (``undeuterated`` for a control
    without a time, NaN)."""
    return "undeuterated" if not np.isfinite(time) else format_time(time).replace(" ", "")


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
    if clash:
        raise ValueError(f"exclude_times {', '.join(f'{t:g}' for t in sorted(clash))}: a "
                         "control's time (the controls are set by the run)")
    out = {name: r.without_times(times) for name, r in runs.items()}
    n = sum(int(r.excluded.sum()) for r in out.values())
    if n:
        log.info("[analysis] exclude_times: %d measurement(s) at %s s left out of the analysis", n,
                 ", ".join(f"{t:g}" for t in times))
    else:
        log.warning("[analysis] exclude_times: no measurement at %s s",
                    ", ".join(f"{t:g}" for t in times))
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
    present = list(dict.fromkeys(up.sample.tolist()))
    samples = list(config.samples) if config.samples else present
    missing = [s for s in samples if s not in present]
    if missing:
        raise ValueError(f"no sample(s) {', '.join(missing)} in the results")
    if config.reference:
        if config.reference not in present:
            raise ValueError(f"no reference sample {config.reference!r} in the results "
                             f"(samples: {', '.join(present)})")
        samples = [config.reference] + [s for s in samples if s != config.reference]
    up = up.select(np.isin(up.sample, samples))
    sequences = config.sequences(base)
    residues = residue_map(up, sequences=sequences, weight_exponent=config.weight_exponent,
                           modified=config.modified_peptides)
    # modified forms kept apart are states of their own (e.g. 'WT pS59'), after their sample
    states = [s for s in dict.fromkeys(residues.sample.tolist()) if s not in samples]
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
    ann = config.annotations
    marks = ann.marks()
    tables: dict[str, dict] = {"species_uptake": up.columns(), "residues": residues.columns(),
                               "coverage": coverage.columns()}
    if choice is not None:
        tables["populations"] = choice.columns()
    if isinstance(results, RunSet):
        tables["run_settings"] = compare_settings(results).columns()
        # the identity check compares the runs, so merged samples are looked at run by run
        unmerged = up if not results.merge else species_uptake(
            results.unmerged(), ks=config.ks, exclude_flagged=config.exclude_flagged,
            filters=config.filters, controls=config.controls)
        tables["run_identity"] = check_identity(unmerged).columns()
    comparisons = {}
    if config.reference:
        ref = config.reference
        kw = {"exclude_flagged": config.exclude_flagged, "filters": config.filters,
              "alpha": config.alpha, "samples": samples, "match": config.peptide_match,
              "controls": config.controls}
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
        pop_cmp = None
        if choice is not None:
            pop_cmp = compare_populations(
                choice, ref, alpha=config.alpha,
                delta_crit=comparisons["K=2 rank 1 (low D) population"].delta_crit)
            tables["compare_populations"] = _columns(pop_cmp)
            tables["compare_choices"] = compare_choices(choice, ref).columns()
    protection = prot_cmp = res_prot = res_cmp = None
    if config.protection:
        from ..analysis.protection import (
            compare_protection,
            compare_residue_protection,
            peptide_protection,
            residue_protection,
        )

        protection, by_time = peptide_protection(up, config.labelling, sequences=sequences,
                                                 exclude_times=(), alpha=config.alpha)
        tables["protection"] = protection.columns()
        tables["protection_by_time"] = by_time.columns()
        if config.protection_residues:
            res_prot = residue_protection(up, config.labelling, sequences=sequences,
                                          exclude_times=(), r1=config.protection_r1,
                                          peptides=protection,
                                          modified=config.modified_peptides, cancel=cancel)
            tables["protection_residues"] = res_prot.columns()
        if config.reference:
            prot_cmp = compare_protection(protection, config.reference, alpha=config.alpha,
                                          match=config.peptide_match)
            tables["compare_protection"] = prot_cmp.columns()
            if res_prot is not None:
                res_cmp = compare_residue_protection(res_prot, config.reference,
                                                     alpha=config.alpha)
                tables["compare_protection_residues"] = res_cmp.columns()
    regions = None
    if ann.regions:
        regions = region_summary(residues, ann)
        tables["regions"] = _columns(regions)

    jobs = []  # (file stem, function drawing the figure)
    for s in samples:
        jobs.append((f"coverage_{s}", lambda s=s: coverage_figure(
            up, samples=[s], times=times, length=length)))
    jobs.append(("redundancy_resolution", lambda: plot_redundancy_resolution(
        coverage, samples=samples, marks=marks)))
    for s in samples + states:
        for t in times:
            label = _time_label(t)
            if choice is not None and choice.where(sample=s, time=t).any():
                jobs.append((f"choice_{s}_{label}", lambda s=s, t=t: plot_population_choice(
                    choice, s, t, marks=marks, length=length)))
            elif residues.where(sample=s, time=t).any():
                jobs.append((f"profile_{s}_{label}", lambda s=s, t=t: plot_residue_profile(
                    residues, s, t, marks=marks, length=length)))
    if config.reference:
        others = [s for s in samples if s != config.reference]
        residue_others = others + states  # the residue comparisons include the states
        stack = list(comparisons.items())
        for t in times:
            label = _time_label(t)
            only = choice if config.overlay_two_only else None
            shown = [s for s in residue_others
                     if _has_overlay(residues, s, t, config.overlay, only)]
            if shown and _has_overlay(residues, config.reference, t, config.overlay, only):
                jobs.append((f"compare_residues_{label}", lambda t=t, shown=shown, only=only:
                             residue_overlay_figure(
                                 residues, config.reference, shown, t, show=config.overlay,
                                 choice=only, annotations=ann, length=length)))
            with_pop = [s for s in residue_others
                        if pop_cmp is not None and pop_cmp.where(sample=s, time=t).any()]
            if with_pop:
                jobs.append((f"compare_populations_{label}",
                             lambda t=t, with_pop=with_pop: residue_comparison_figure(
                                 pop_cmp, with_pop, t, annotations=ann, length=length)))
            for s in others:
                if all(c.where(sample=s, time=t).any() for _, c in stack):
                    jobs.append((f"woods_{s}_{label}", lambda s=s, t=t: woods_figure(
                        stack, s, t, annotations=ann, length=length)))
        k1 = comparisons["K=1 fD"]
        drawn = list(times)
        for s in others:
            if drawn and k1.where(sample=s, time=drawn).any():
                jobs.append((f"butterfly_{s}", lambda s=s: plot_butterfly(
                    k1, s, times=[t for t in drawn if k1.where(sample=s, time=t).any()])))
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
    if res_prot is not None and len(res_prot):
        shown = [s for s in samples if res_prot.where(sample=s).any()]
        jobs.append(("protection_residues", lambda: residue_protection_figure(
            res_prot, shown, annotations=ann, length=length,
            max_se=config.protection_max_se)))
    if res_cmp is not None:
        for n, s in enumerate(samples):
            if res_cmp.where(sample=s).any():
                jobs.append((f"protection_residue_comparison_{s}", lambda s=s, n=n:
                             residue_protection_comparison_figure(
                                 res_prot, res_cmp, s, style_index=n, annotations=ann,
                                 length=length, max_se=config.protection_max_se)))
    if regions is not None:
        for t in times:
            if (regions.time == t).any():
                jobs.append((f"regions_{_time_label(t)}", lambda t=t: plot_region_bars(
                    regions, t, samples=samples)))

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
    directory.mkdir(parents=True, exist_ok=True)
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

    written = []
    total = len(tables) + len(jobs)
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
