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
``profile_<sample>_<time>`` without K=2), and with a reference ``compare_residues_<time>``,
``compare_populations_<time>``, ``woods_<sample>_<time>``, ``butterfly_<sample>`` and
``regions_<time>``. Characters a file name cannot hold (the ``:`` of ``run:sample``) become
``_``.
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
from ..analysis.settings import UNDEUTERATED
from ..hooks import CancelHook, ProgressHook, check_cancel, report
from .compare import (
    plot_butterfly,
    plot_region_bars,
    residue_comparison_figure,
    woods_figure,
)
from .coverage import coverage_figure, plot_redundancy_resolution
from .residues import plot_population_choice, plot_residue_profile
from .tables import write_csv

log = logging.getLogger(__name__)


def _columns(table) -> dict[str, np.ndarray]:
    return {f.name: getattr(table, f.name) for f in fields(table)
            if isinstance(getattr(table, f.name), np.ndarray)}


def _time_label(time: float, undeut: float) -> str:
    return "undeuterated" if np.isclose(time, undeut) else f"{time:g}s"


def _stem(name: str) -> str:
    """``name`` as a file name on every system."""
    return re.sub(r'[<>:"/\\|?*\s]', "_", name)


def _fd_times(results) -> list[float]:
    """The fully deuterated control times of the run (or of every run of a RunSet)."""
    out = []
    for r in results.runs.values() if isinstance(results, RunSet) else [results]:
        run = r.settings.run if r.settings is not None else None
        if run is not None and run.controls is not None and run.controls.fd_time is not None:
            out.append(float(run.controls.fd_time))
    return out


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
    up = species_uptake(results, ks=config.ks, exclude_flagged=config.exclude_flagged,
                        filters=config.filters)
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
    residues = residue_map(up, sequences=sequences, weight_exponent=config.weight_exponent)
    coverage_time = None if config.coverage_times == UNDEUTERATED else config.coverage_times
    coverage = coverage_table(up, sequences=sequences, count=config.redundancy,
                              time=coverage_time)
    choice = choose_populations(residues, config.choice) if 2 in config.ks else None
    length = int(residues.residue.max())
    fd_times = _fd_times(results)
    times = list(config.times) or [t for t in sorted(set(up.time.tolist()))
                                   if not any(np.isclose(t, f) for f in fd_times)]
    undeut = up.undeut_time
    ann = config.annotations
    marks = ann.marks()
    tables: dict[str, dict] = {"species_uptake": up.columns(), "residues": residues.columns(),
                               "coverage": coverage.columns()}
    if choice is not None:
        tables["populations"] = choice.columns()
    if isinstance(results, RunSet):
        tables["run_settings"] = compare_settings(results).columns()
        tables["run_identity"] = check_identity(up).columns()
    comparisons = {}
    if config.reference:
        ref = config.reference
        kw = {"exclude_flagged": config.exclude_flagged, "filters": config.filters,
              "alpha": config.alpha, "samples": samples}
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
                                   delta_crit=comparisons["K=1 fD"].delta_crit)
        tables["compare_residues"] = _columns(res_cmp)
        pop_cmp = None
        if choice is not None:
            pop_cmp = compare_populations(
                choice, ref, alpha=config.alpha,
                delta_crit=comparisons["K=2 rank 1 (low D) population"].delta_crit)
            tables["compare_populations"] = _columns(pop_cmp)
            tables["compare_choices"] = compare_choices(choice, ref).columns()
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
    for s in samples:
        for t in times:
            label = _time_label(t, undeut)
            if choice is not None and choice.where(sample=s, time=t).any():
                jobs.append((f"choice_{s}_{label}", lambda s=s, t=t: plot_population_choice(
                    choice, s, t, marks=marks, length=length, undeut_time=undeut)))
            elif residues.where(sample=s, time=t).any():
                jobs.append((f"profile_{s}_{label}", lambda s=s, t=t: plot_residue_profile(
                    residues, s, t, marks=marks, length=length, undeut_time=undeut)))
    if config.reference:
        others = [s for s in samples if s != config.reference]
        stack = list(comparisons.items())
        for t in times:
            label = _time_label(t, undeut)
            if others and res_cmp.where(time=t).any():
                jobs.append((f"compare_residues_{label}", lambda t=t: residue_comparison_figure(
                    res_cmp, others, t, annotations=ann, length=length, undeut_time=undeut)))
            if others and pop_cmp is not None and pop_cmp.where(time=t).any():
                jobs.append((f"compare_populations_{label}",
                             lambda t=t: residue_comparison_figure(
                                 pop_cmp, others, t, annotations=ann, length=length,
                                 undeut_time=undeut)))
            for s in others:
                if all(c.where(sample=s, time=t).any() for _, c in stack):
                    jobs.append((f"woods_{s}_{label}", lambda s=s, t=t: woods_figure(
                        stack, s, t, annotations=ann, length=length, undeut_time=undeut)))
        k1 = comparisons["K=1 fD"]
        drawn = [t for t in times if not np.isclose(t, undeut)]
        for s in others:
            if drawn and k1.where(sample=s, time=drawn).any():
                jobs.append((f"butterfly_{s}", lambda s=s: plot_butterfly(
                    k1, s, times=[t for t in drawn if k1.where(sample=s, time=t).any()],
                    undeut_time=undeut)))
    if regions is not None:
        for t in times:
            if (regions.time == t).any():
                jobs.append((f"regions_{_time_label(t, undeut)}", lambda t=t: plot_region_bars(
                    regions, t, samples=samples)))

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
