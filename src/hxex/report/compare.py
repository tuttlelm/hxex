# SPDX-License-Identifier: GPL-3.0-or-later
"""Plots of comparisons (docs/ANALYSIS.md, phases 3 and 4; between samples of a run or between
runs): a reference sample against the others residue by residue, peptide differences (Woods and
butterfly plots) and region bar charts. Samples keep
one colour and marker each (:data:`hxex.report.coverage.SAMPLE_STYLES`, the reference first, in
black). Functions return figures and never show them."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FuncFormatter

from ..analysis import (
    Annotations,
    PeptideComparison,
    PopulationChoice,
    PopulationCountComparison,
    RegionTable,
    ResidueComparison,
    ResidueMap,
)
from ..analysis.settings import OVERLAY_VALUES
from .coverage import MARK_LINE, SAMPLE_STYLES
from .residues import _when

INCREASE, DECREASE, NEUTRAL = "#c0392b", "#1f5fa8", "#b5b5b5"
FILL = 0.65  # residue figures: marker fill and K=2 error bars; outlines solid (owner, 2026-10-21)
K2_MARKERS = {"low": "^", "high": "v", "pop_low": "^", "pop_high": "v"}  # rank 1 up, 2 down
COUNT_GREY = "#7f7f7f"


def _style(n: int):
    return SAMPLE_STYLES[n % len(SAMPLE_STYLES)]


def _marks(ax: Axes, annotations: Annotations | None, label_sites: bool = True) -> None:
    if annotations is None:
        return
    for x in annotations.marks():
        ax.axvline(x, **MARK_LINE)
    if label_sites:
        for site in annotations.sites:
            if site.label:
                ax.text(site.residue, 1.0, site.label, transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=7, color="#4d4d4d")


DIFFERENT_NOTE = "* corrected differently (control fallback)"


def _corrected_differently(ax: Axes, xs, *, y: float = 0.995, note: bool = True) -> int:
    """A ``*`` just inside the top of ``ax`` at each x of ``xs`` (residues or positions where
    the two compared values were corrected differently: one with a fallback control), and, with
    ``note``, a note in the lower left; returns how many were drawn."""
    xs = sorted({float(x) for x in xs})
    for x in xs:
        ax.text(x, y, "*", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=11, color="#4d4d4d", gid=f"corrected_differently:{x:g}")
    if xs and note:
        ax.text(0.005, 0.01, DIFFERENT_NOTE, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7, color="#4d4d4d")
    return len(xs)


def _threshold_lines(ax, comparison, rows, left, right) -> None:
    """``+-threshold`` dashed: one line across the panel when every row shares it (the fD
    comparisons' ``delta_crit``), else a short dashed line over each row's place (``left`` to
    ``right``; a population's threshold depends on the separation of its K=2 positions)."""
    # above the grid (zorder 1.5), below the bars: a threshold on a gridline stays visible
    style = {"color": "#555555", "linestyle": "--", "linewidth": 0.9, "zorder": 1.6}
    th = np.asarray(comparison.threshold, dtype=float)[rows]
    finite = np.isfinite(th)
    if not finite.any():
        return
    if np.allclose(th[finite], th[finite][0]):
        for y in (th[finite][0], -th[finite][0]):
            ax.axhline(y, **style)
        return
    for x0, x1, y in zip(np.asarray(left)[finite], np.asarray(right)[finite], th[finite],
                         strict=True):
        ax.plot([x0, x1], [y, y], **style)
        ax.plot([x0, x1], [-y, -y], **style)


def _label(quantity: str) -> str:
    return "population (low D)" if quantity == "population" else "fD"


def _rank_points(ax: Axes, x, y, se, colour: str, marker: str, zorder: float, gid: str) -> None:
    """Points with capped standard-error bars, as every residue figure draws them: a 65 %
    fill with a solid outline, error bars at the same opacity."""
    small = marker == "o"
    bars = ax.errorbar(x, y, yerr=np.nan_to_num(se), fmt=marker,
                       markersize=3.5 if small else 5.0, color=colour,
                       markerfacecolor=to_rgba(colour, FILL), markeredgecolor=colour,
                       markeredgewidth=0.7, ecolor=to_rgba(colour, FILL), elinewidth=0.9,
                       capsize=2.0, capthick=0.9, zorder=zorder)
    bars.lines[0].set_gid(gid)


def _top(value, se=None) -> float:
    """The highest point drawn: ``value`` plus its error bar (-inf when nothing is finite)."""
    value = np.asarray(value, dtype=float)
    upper = value if se is None else value + np.nan_to_num(np.asarray(se, dtype=float))
    upper = upper[np.isfinite(upper)]
    return float(upper.max()) if len(upper) else -np.inf


def _upper_limit(top: float) -> float:
    """The top of a deuteration axis: 1.05, or above the highest point drawn (fD is not
    clipped at 1: values to ``max_fD`` are kept), as :func:`plot_residue_profile`."""
    return max(1.05, 1.05 * top)


def _residue_axes(ax: Axes, *, length: int, ylabel: str, reference: str, sample: str,
                  style_index: int, n_diff: int, time: float, time_scale: str,
                  annotations: Annotations | None, top: float = -np.inf) -> None:
    """The axes, legend (the samples' colours, and the corrected-differently mark when drawn)
    and time label every residue figure shares; ``top`` is the highest point drawn."""
    _marks(ax, annotations, label_sites=False)
    ax.set_xlim(0, length)
    ax.set_ylim(-0.05, _upper_limit(top))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.set_xlabel("Residue")
    ax.set_ylabel(ylabel)
    ax.grid(color="#e5e5e5", linewidth=0.8)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=5,
                      color=_style(i)[0], markerfacecolor=to_rgba(_style(i)[0], FILL),
                      label=name)
               for name, i in ((reference, 0), (sample, style_index))]
    if n_diff:
        handles.append(Line2D([], [], linestyle="none", marker="$*$", markersize=7,
                              color="#4d4d4d", label="corrected differently"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.98),
              ncol=len(handles), frameon=False, fontsize=9, handletextpad=0.2)
    ax.text(1.0, 1.02, _when(time, time_scale), transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9)


def _overlay_label(show: Sequence[str]) -> str:
    """``fD_low, fD_avg, fD_high`` and ``pop_low, pop_high`` for the values shown."""
    parts = [rf"fD$_{{\rm {v}}}$" for v in ("low", "avg", "high") if v in show]
    parts += [rf"pop$_{{\rm {v[4:]}}}$" for v in ("pop_low", "pop_high") if v in show]
    return ", ".join(parts)


def plot_residue_comparison(
    comparison: ResidueComparison,
    sample: str,
    time: float,
    *,
    style_index: int = 1,
    annotations: Annotations | None = None,
    length: int | None = None,
    ax: Axes | None = None,
    time_scale: str = "seconds",
    mark_corrected_differently: bool = True,
) -> Figure:
    """The reference (black) and ``sample`` (its colour) residue by residue at ``time``, in the
    comparison's quantity (``fD``, or the low-D population from
    :func:`hxex.analysis.compare_populations`), with the markers, fills, legend and axes of
    :func:`plot_residue_overlay` (a K=1 value a circle, rank 1 an up and rank 2 a down
    triangle); residues whose two values were corrected differently (one with a fallback
    control) get a ``*`` unless ``mark_corrected_differently`` is False. Which differences are
    significant is in the comparison table, not drawn.

    ``time_scale`` (:attr:`hxex.report.Results.time_scale`): with ``'order'`` the time is
    labelled as a point number, not a time."""
    rows = comparison.where(sample=sample, time=time)
    if not rows.any():
        raise ValueError(f"no comparison of {sample!r} at time {time:g}")
    if ax is None:
        fig = Figure(figsize=(7.5, 3.0))
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    order = np.argsort(comparison.residue[rows])
    x = comparison.residue[rows][order]
    marker = "o" if comparison.K == 1 else "^" if comparison.rank == 1 else "v"
    top = -np.inf
    for n, (name, value, se) in enumerate((
            (comparison.reference, comparison.value_reference, comparison.value_reference_se),
            (sample, comparison.value, comparison.value_se))):
        colour = _style(0 if n == 0 else style_index)[0]
        _rank_points(ax, x, value[rows][order], se[rows][order], colour, marker, 3 + n,
                     f"{comparison.quantity}:{name}")
        top = max(top, _top(value[rows], se[rows]))
    n_diff = (_corrected_differently(
        ax, comparison.residue[rows & comparison.corrected_differently], note=False)
        if mark_corrected_differently else 0)
    _residue_axes(ax, length=length if length is not None else int(comparison.residue.max()),
                  ylabel=_label(comparison.quantity), reference=comparison.reference,
                  sample=sample, style_index=style_index, n_diff=n_diff, time=time,
                  time_scale=time_scale, annotations=annotations, top=top)
    return fig


def residue_comparison_figure(
    comparison: ResidueComparison,
    samples: Sequence[str],
    time: float,
    *,
    annotations: Annotations | None = None,
    length: int | None = None,
    time_scale: str = "seconds",
    mark_corrected_differently: bool = True,
) -> Figure:
    """One :func:`plot_residue_comparison` panel per sample of ``samples`` against the
    comparison's reference (laid out as :func:`residue_overlay_figure`); each sample keeps its
    colour.

    ``time_scale`` (:attr:`hxex.report.Results.time_scale`): with ``'order'`` the time is
    labelled as a point number, not a time."""
    samples = [s for s in samples if s != comparison.reference]
    if not samples:
        raise ValueError("no sample to compare with the reference")
    fig = Figure(figsize=(6.5, 3.0 * len(samples)), layout="constrained")
    axes = fig.subplots(len(samples), 1, squeeze=False)[:, 0]
    for n, (ax, sample) in enumerate(zip(axes, samples, strict=True)):
        plot_residue_comparison(comparison, sample, time, style_index=n + 1,
                                annotations=annotations, length=length, ax=ax,
                                time_scale=time_scale,
                                mark_corrected_differently=mark_corrected_differently)
    return fig


def plot_residue_overlay(
    residues: ResidueMap,
    reference: str,
    sample: str,
    time: float,
    *,
    show: Sequence[str] = ("low", "high"),
    choice: PopulationChoice | None = None,
    style_index: int = 1,
    annotations: Annotations | None = None,
    length: int | None = None,
    ax: Axes | None = None,
    time_scale: str = "seconds",
    mark_corrected_differently: bool = True,
) -> Figure:
    """``reference`` (black) and ``sample`` (its colour) residue by residue at ``time``, the
    sample drawn over the reference. ``show`` is any of ``'low'`` and ``'high'`` (the K=2 ranks
    1 and 2: up and down triangles with capped standard-error bars), ``'pop_low'`` and
    ``'pop_high'`` (the populations of those ranks, from ``choice``, which they need: the same
    triangles) and ``'avg'`` (K=1: small
    circles on a line with a +-1 standard-error band, joined only between residues one peptide
    measures together, ``ResidueMap.linked_next``). Fills and K=2 error bars are 65 % opaque
    with solid outlines, so overlapping markers stay apart; the legend names the samples' colours.
    With ``choice``
    (:func:`hxex.analysis.choose_populations`), the K=2 values are drawn only where the choice
    keeps two populations for that sample (each sample on its own); the average is always drawn.
    A ``*`` at the top marks residues drawn for both whose values were corrected differently (one
    with a fallback control), unless ``mark_corrected_differently`` is False. Point groups have
    the gid ``overlay:<sample>:<value>``.

    ``time_scale`` (:attr:`hxex.report.Results.time_scale`): with ``'order'`` the time is
    labelled as a point number, not a time."""
    unknown = [v for v in show if v not in OVERLAY_VALUES]
    if unknown or not show:
        raise ValueError(f"show must be some of {', '.join(OVERLAY_VALUES)}")
    if choice is None and any(v.startswith("pop_") for v in show):
        raise ValueError("the populations (pop_low, pop_high) need the choice "
                         "(hxex.analysis.choose_populations)")
    if ax is None:
        fig = Figure(figsize=(7.5, 3.0))
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    drawn, top = 0, -np.inf
    shown: dict[str, set] = {reference: set(), sample: set()}
    fell_back: dict[str, set] = {reference: set(), sample: set()}
    for n, (name, index) in enumerate(((reference, 0), (sample, style_index))):
        colour = _style(index)[0]
        for value in show:
            if value.startswith("pop_"):  # a population: the sample's choice, where it keeps two
                rows = (choice.where(sample=name, time=time)
                        & np.asarray(choice.two_populations, dtype=bool)
                        & np.isfinite(choice.population_low))
                if not rows.any():
                    continue
                drawn += 1
                shown[name].update(choice.residue[rows].tolist())
                fell_back[name].update(choice.residue[rows & choice.control_fallback].tolist())
                pop = choice.population_low[rows]
                pop = pop if value == "pop_low" else 1 - pop
                _rank_points(ax, choice.residue[rows], pop, choice.population_low_se[rows],
                             colour, K2_MARKERS[value], 3 + n, f"overlay:{name}:{value}")
                top = max(top, _top(pop, choice.population_low_se[rows]))
                continue
            k, rank = OVERLAY_VALUES[value]
            rows = residues.where(sample=name, time=time, K=k, rank=rank) & np.isfinite(
                residues.fD)
            if choice is not None and k >= 2:
                rows &= np.isin(residues.residue, _two_population_residues(choice, name, time))
            if not rows.any():
                continue
            drawn += 1
            shown[name].update(residues.residue[rows].tolist())
            fell_back[name].update(residues.residue[rows & residues.control_fallback].tolist())
            top = max(top, _top(residues.fD[rows], residues.fD_se[rows]))
            if value == "avg":
                _average_line(ax, residues, name, time, colour, zorder=1.7 + 0.1 * n)
                (line,) = ax.plot(residues.residue[rows], residues.fD[rows], linestyle="none",
                                  marker="o", markersize=2.6, color=colour,
                                  markerfacecolor=to_rgba(colour, FILL),
                                  markeredgecolor=colour, markeredgewidth=0.6,
                                  zorder=1.9 + 0.1 * n)
                line.set_gid(f"overlay:{name}:avg")
                continue
            _rank_points(ax, residues.residue[rows], residues.fD[rows], residues.fD_se[rows],
                         colour, K2_MARKERS[value], 3 + n, f"overlay:{name}:{value}")
    if not drawn:
        raise ValueError(f"no residue values of {reference!r} or {sample!r} at time {time:g}")
    both = shown[reference] & shown[sample]
    n_diff = (_corrected_differently(
        ax, [r for r in both if (r in fell_back[reference]) != (r in fell_back[sample])],
        note=False) if mark_corrected_differently else 0)
    _residue_axes(ax, length=length if length is not None else int(residues.residue.max()),
                  ylabel=_overlay_label(show), reference=reference, sample=sample,
                  style_index=style_index, n_diff=n_diff, time=time, time_scale=time_scale,
                  annotations=annotations, top=top)
    return fig


def _average_line(ax: Axes, residues: ResidueMap, sample: str, time: float, colour: str,
                  zorder: float) -> None:
    """The K=1 residue values of ``sample`` at ``time`` as a line with a +-1 standard-error
    band in ``colour``, broken between residues no peptide measures together; a residue without
    a value (a proline) is bridged when one peptide spans both its neighbours."""
    rows = residues.where(sample=sample, time=time, K=1, rank=1)
    res = np.asarray(residues.residue[rows], dtype=int)
    if not len(res):
        return
    n = int(res.max()) + 2
    y, e = np.full(n, np.nan), np.full(n, np.nan)
    linked = np.zeros(n, dtype=bool)
    y[res], e[res] = residues.fD[rows], np.nan_to_num(residues.fD_se[rows])
    linked[res] = np.asarray(residues.linked_next[rows], dtype=bool)
    for r in range(1, n - 1):  # a missing value inside one peptide's span
        if np.isnan(y[r]) and linked[r - 1] and linked[r] and np.isfinite(y[r - 1]) \
                and np.isfinite(y[r + 1]):
            y[r], e[r] = (y[r - 1] + y[r + 1]) / 2, (e[r - 1] + e[r + 1]) / 2
    xs, ys, es = [], [], []
    for r in range(1, n):
        xs.append(r)
        ys.append(y[r])
        es.append(e[r])
        if not linked[r]:  # no peptide measures r and r + 1 together: break the line
            xs.append(r + 0.5)
            ys.append(np.nan)
            es.append(np.nan)
    xs, ys, es = np.array(xs, dtype=float), np.array(ys), np.array(es)
    ax.fill_between(xs, ys - es, ys + es, color=colour, alpha=0.3, linewidth=0, zorder=zorder)
    ax.plot(xs, ys, color=colour, linewidth=1.0, alpha=0.9, zorder=zorder + 0.05)


def _two_population_residues(choice: PopulationChoice, sample: str, time: float) -> np.ndarray:
    """The residues where ``choice`` keeps two populations for ``sample`` at ``time``."""
    rows = choice.where(sample=sample, time=time) & np.asarray(choice.two_populations, dtype=bool)
    return choice.residue[rows]


def residue_overlay_figure(
    residues: ResidueMap,
    reference: str,
    samples: Sequence[str],
    time: float,
    *,
    show: Sequence[str] = ("low", "high"),
    choice: PopulationChoice | None = None,
    annotations: Annotations | None = None,
    length: int | None = None,
    time_scale: str = "seconds",
    mark_corrected_differently: bool = True,
) -> Figure:
    """One :func:`plot_residue_overlay` panel per sample of ``samples`` against ``reference``;
    each sample keeps its colour. ``choice`` filters the K=2 values as in
    :func:`plot_residue_overlay`.

    ``time_scale`` (:attr:`hxex.report.Results.time_scale`): with ``'order'`` the time is
    labelled as a point number, not a time."""
    samples = [s for s in samples if s != reference]
    if not samples:
        raise ValueError("no sample to compare with the reference")
    fig = Figure(figsize=(6.5, 3.0 * len(samples)), layout="constrained")
    axes = fig.subplots(len(samples), 1, squeeze=False)[:, 0]
    for n, (ax, sample) in enumerate(zip(axes, samples, strict=True)):
        plot_residue_overlay(residues, reference, sample, time, show=show, choice=choice,
                             style_index=n + 1, time_scale=time_scale,
                             annotations=annotations, length=length, ax=ax,
                             mark_corrected_differently=mark_corrected_differently)
    return fig


def plot_peptide_differences(
    comparison: PeptideComparison,
    sample: str,
    time: float,
    *,
    annotations: Annotations | None = None,
    length: int | None = None,
    ax: Axes | None = None,
    time_scale: str = "seconds",
) -> Figure:
    """The Woods plot of ``sample`` against the reference at ``time``: each peptide a horizontal
    bar over its residues at its difference in ``fD``, red if significantly higher, blue if lower,
    grey otherwise, with the standard error as a vertical bar and the threshold dashed (one
    line, ``+-delta_crit``; for a population as it is, a short line over each peptide, and one
    line for :meth:`~hxex.analysis.PeptideComparison.scaled_by_separation`'s copy); a
    ``*`` at the top over the middle of each peptide whose two values were corrected differently
    (one with a fallback control). Bars have the gid ``peptide:<start>-<end>``.

    ``time_scale`` (:attr:`hxex.report.Results.time_scale`): with ``'order'`` the time is
    labelled as a point number, not a time."""
    rows = np.flatnonzero(comparison.where(sample=sample, time=time))
    if not len(rows):
        raise ValueError(f"no comparison of {sample!r} at time {time:g}")
    if ax is None:
        fig = Figure(figsize=(7.5, 3.2))
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    for i in rows:
        d = comparison.delta[i]
        colour = (INCREASE if d > 0 else DECREASE) if comparison.significant[i] else NEUTRAL
        (line,) = ax.plot([comparison.start[i] - 0.5, comparison.end[i] + 0.5], [d, d],
                          color=colour, linewidth=2.2, solid_capstyle="butt",
                          zorder=3 if comparison.significant[i] else 2)
        line.set_gid(f"peptide:{comparison.start[i]}-{comparison.end[i]}")
        mid = (comparison.start[i] + comparison.end[i]) / 2
        ax.plot([mid, mid], [d - comparison.delta_se[i], d + comparison.delta_se[i]],
                color=colour, linewidth=0.7, zorder=2)
    _threshold_lines(ax, comparison, rows, comparison.start[rows] - 0.5,
                     comparison.end[rows] + 0.5)
    ax.axhline(0, color="#999999", linewidth=0.8, zorder=1.6)  # above the grid, grey: not the data
    differ = rows[comparison.corrected_differently[rows]]
    _corrected_differently(ax, (comparison.start[differ] + comparison.end[differ]) / 2)
    _marks(ax, annotations)
    top = length if length is not None else int(comparison.end[rows].max())
    ax.set_xlim(0, top)
    extent = np.abs(comparison.delta[rows]) + np.nan_to_num(comparison.delta_se[rows])
    finite = extent[np.isfinite(extent)]
    span = float(finite.max()) if len(finite) else 0.2  # every delta NaN: a default range
    ax.set_ylim(-max(span, 0.2) * 1.1, max(span, 0.2) * 1.1)
    ax.set_xlabel("residue")
    what = ("ΔfD" if comparison.quantity == "fD" else "Δpop × sep"
            if comparison.scaled else "Δpopulation")
    ax.set_ylabel(f"{what} ({sample} − {comparison.reference})")
    ax.grid(color="#e5e5e5", linewidth=0.8)
    n_sig = int(comparison.significant[rows].sum())
    n_compared = int(np.isfinite(comparison.delta[rows]).sum())  # a population may not be
    ax.text(0.99, 0.03, f"{_when(time, time_scale)}; {n_sig} of {n_compared} significant",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    return fig


def _lanes(spans: Sequence[tuple[int, int]]) -> list[int]:
    """Stack overlapping residue spans in lanes (the first lane that is free)."""
    ends: list[float] = []
    out = []
    for start, end in spans:
        for j, last in enumerate(ends):
            if start > last + 0.5:
                ends[j] = end
                out.append(j)
                break
        else:
            ends.append(end)
            out.append(len(ends) - 1)
    return out


def _count_groups(counts: PopulationCountComparison, sample: str, time: float):
    """The peptides of ``sample`` at ``time``: two populations in the sample only, in the
    reference only, and in both, each a sorted list of ``(start, end, borderline)``."""
    rows = np.flatnonzero(counts.where(sample=sample, time=time))
    groups: tuple[list, list, list] = ([], [], [])
    for i in rows:
        k, kr = int(counts.K[i]), int(counts.K_reference[i])
        span = (int(counts.start[i]), int(counts.end[i]), bool(counts.borderline[i]))
        if k == 2 and kr == 2:
            groups[2].append(span)
        elif counts.differs[i]:
            groups[0 if k > kr else 1].append(span)
    return tuple(sorted(g) for g in groups)


COUNT_BAR, COUNT_STEP, COUNT_GAP = 0.2, 0.26, 0.18  # bar height, lane spacing, band gap


def plot_population_counts(
    counts: PopulationCountComparison,
    sample: str,
    time: float,
    *,
    length: int | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Where ``sample`` and the reference differ in the number of populations their peptides
    are best described by (owner, 2026-10-21): a grey bar over each peptide, in the band of the
    side with two populations (``<sample> only`` above the line, ``<reference> only`` below), and
    the peptides where both have two in a band of their own (``both``); outlined, not filled,
    where the difference is borderline (``PopulationCountComparison.borderline``). Overlapping
    peptides are stacked in lanes. Bars have the gid ``count:<band>:<start>-<end>``."""
    if ax is None:
        fig = Figure(figsize=(7.5, 1.8))
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    up, down, both = _count_groups(counts, sample, time)
    n_up, n_dn, n_both = (max(_lanes([(a, b) for a, b, _ in g])) + 1 if g else 1
                          for g in (up, down, both))
    base = -COUNT_STEP * n_dn - COUNT_GAP  # the "both" band begins below the reference's
    for band, group, y_of in (
            ("sample", up, lambda lane: COUNT_STEP * (lane + 0.5)),
            ("reference", down, lambda lane: -COUNT_STEP * (lane + 0.5)),
            ("both", both, lambda lane: base - COUNT_STEP * (lane + 0.5))):
        for (a, b, borderline), lane in zip(group, _lanes([(a, b) for a, b, _ in group]),
                                            strict=True):
            bars = ax.broken_barh([(a - 0.5, b - a + 1)], (y_of(lane) - COUNT_BAR / 2,
                                                         COUNT_BAR),
                                  facecolors="none" if borderline else COUNT_GREY,
                                  edgecolors=COUNT_GREY, linewidth=0.8)
            bars.set_gid(f"count:{band}:{a}-{b}")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.axhline(base + COUNT_GAP / 2, color="#cccccc", linewidth=0.8)
    ax.set_ylim(base - COUNT_STEP * n_both - 0.04, COUNT_STEP * n_up + 0.04)
    ax.set_yticks([COUNT_STEP * n_up / 2, -COUNT_STEP * n_dn / 2,
                   base - COUNT_STEP * n_both / 2],
                  [f"{sample} only", f"{counts.reference} only", "both"], fontsize=8)
    ax.annotate("K=2", xy=(0, 1), xycoords="axes fraction", xytext=(-6, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=9,
                weight="bold")
    top = length if length is not None else int(max(counts.end)) if len(counts) else 1
    ax.set_xlim(0, top)
    ax.set_title("Two Populations Selected (outline: borderline)", fontsize=10, loc="left")
    ax.set_xlabel("residue")
    return fig


def woods_populations_figure(
    k1: PeptideComparison,
    counts: PopulationCountComparison,
    sample: str,
    time: float,
    *,
    annotations: Annotations | None = None,
    length: int | None = None,
    time_scale: str = "seconds",
) -> Figure:
    """The first page of the Woods plot of ``sample`` at ``time``: the K=1 differences
    (:func:`plot_peptide_differences`) above where the number of populations differs
    (:func:`plot_population_counts`); the K=2 comparisons are the second page
    (:func:`woods_figure`)."""
    up, down, both = _count_groups(counts, sample, time)
    lanes = sum(max(_lanes([(a, b) for a, b, _ in g])) + 1 if g else 1
                for g in (up, down, both))
    height = 0.5 + 0.13 * lanes  # inches: the panel fits its lanes
    fig = Figure(figsize=(7.5, 3.2 + height), layout="constrained")
    top_ax, bottom_ax = fig.subplots(2, 1, sharex=True,
                                     gridspec_kw={"height_ratios": [2.7, height]})
    plot_peptide_differences(k1, sample, time, annotations=annotations, length=length,
                             ax=top_ax, time_scale=time_scale)
    top_ax.set_title("K=1 fD", fontsize=10, loc="left")
    top_ax.set_xlabel("")
    plot_population_counts(counts, sample, time, length=length or int(max(k1.end)),
                           ax=bottom_ax)
    return fig


def plot_region_bars(
    regions: RegionTable,
    time: float,
    *,
    samples: Sequence[str] | None = None,
    region_order: Sequence[str] | None = None,
    ax: Axes | None = None,
    time_scale: str = "seconds",
) -> Figure:
    """Grouped bars of the mean ``fD`` of every region, one bar per
    sample in its colour, with standard-error bars; regions without a value are left empty. Each
    bar has the gid ``region:<sample>:<region>``.

    ``time_scale`` (:attr:`hxex.report.Results.time_scale`): with ``'order'`` the time is
    labelled as a point number, not a time."""
    sel = regions.time == time
    if not sel.any():
        raise ValueError(f"no regions at time {time:g}")
    samples = list(samples) if samples is not None else list(dict.fromkeys(regions.sample[sel]))
    names = list(region_order) if region_order is not None else list(
        dict.fromkeys(regions.region[sel]))
    if ax is None:
        fig = Figure(figsize=(max(5.0, 0.7 * len(names) + 1.5), 3.0))
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    width = 0.8 / max(len(samples), 1)
    x = np.arange(len(names))
    top = -np.inf
    for n, sample in enumerate(samples):
        colour, _, _ = _style(n)
        values, errors = [], []
        for name in names:
            row = sel & (regions.sample == sample) & (regions.region == name)
            values.append(float(regions.fD[row][0]) if row.any() else np.nan)
            errors.append(float(regions.fD_se[row][0]) if row.any() else np.nan)
        bars = ax.bar(x + (n - (len(samples) - 1) / 2) * width, values, width=width,
                      color=colour, label=sample, yerr=np.nan_to_num(errors),
                      error_kw={"elinewidth": 0.7, "ecolor": "#555555"}, alpha=0.9)
        for patch, name in zip(bars.patches, names, strict=True):
            patch.set_gid(f"region:{sample}:{name}")
        top = max(top, _top(values, errors))
    labels = [textwrap.fill(name.replace(",", ", "), 10).replace(", ", ",") for name in names]
    ax.set_xticks(x, labels, rotation=0, fontsize=8)
    ax.set_ylim(0, _upper_limit(top))
    ax.set_ylabel("fD")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(samples), frameon=False,
              fontsize=8)
    return fig


def plot_butterfly(
    comparison: PeptideComparison,
    sample: str,
    *,
    times: Sequence[float] | None = None,
    max_labels: int = 60,
    time_scale: str = "seconds",
) -> Figure:
    """The butterfly plot of ``sample`` against the reference: every peptide in order of its
    position, the reference's value upwards and the sample's downwards, one line per time (a
    sequential colour per time), with the differences below (filled where significant,
    ``+-delta_crit`` dashed); a ``*`` at the top over each peptide whose two values were corrected
    differently at a time drawn (one with a fallback control; named in the legend). ``times``
    default to every time point of the comparison (the controls are not times). Lines have the
    gids ``butterfly:<time>:reference``, ``butterfly:<time>:sample`` and
    ``difference:<time>``.

    ``time_scale`` (:attr:`hxex.report.Results.time_scale`): with ``'order'`` the time is
    labelled as a point number, not a time."""
    rows = comparison.where(sample=sample)
    if not rows.any():
        raise ValueError(f"no comparison of {sample!r}")
    if times is None:
        times = sorted({float(t) for t in comparison.time[rows] if np.isfinite(t)})
    times = [float(t) for t in times]
    if not times:
        raise ValueError(f"no time to draw for {sample!r}")
    def key(i):
        return (int(comparison.start[i]), int(comparison.end[i]), str(comparison.peptide[i]),
                str(comparison.modification[i]))

    keys = sorted({key(i) for i in np.flatnonzero(rows)})
    position = {k: n for n, k in enumerate(keys)}
    fig = Figure(figsize=(max(6.0, min(0.12 * len(keys) + 2.0, 16.0)), 5.5),
                 layout="constrained")
    top, bottom = fig.subplots(2, 1, sharex=True, height_ratios=(2, 1))
    cmap = matplotlib.colormaps["viridis"]
    for n, t in enumerate(times):
        here = np.flatnonzero(rows & (comparison.time == t))
        if not len(here):
            continue
        x = np.array([position[key(i)] for i in here])
        colour = cmap(n / max(len(times) - 1, 1) * 0.9)

        def spread(values, x=x):  # every peptide's place, NaN where not measured (a gap)
            out = np.full(len(keys), np.nan)
            out[x] = values
            return out

        line = {"color": colour, "marker": "o", "markersize": 2.5, "linewidth": 1.0}
        (up,) = top.plot(spread(comparison.value_reference[here]), label=_when(t, time_scale),
                         **line)
        (down,) = top.plot(spread(-comparison.value[here]), **line)
        up.set_gid(f"butterfly:{t:g}:reference")
        down.set_gid(f"butterfly:{t:g}:sample")
        (diff,) = bottom.plot(spread(comparison.delta[here]), color=colour, linewidth=0.8)
        diff.set_gid(f"difference:{t:g}")
        sig = comparison.significant[here].astype(bool)
        bottom.scatter(x[sig], comparison.delta[here][sig], color=colour, s=14, zorder=3)
        bottom.scatter(x[~sig], comparison.delta[here][~sig], facecolors="none",
                       edgecolors=colour, s=10, linewidths=0.7, zorder=3)
    drawn_rows = np.flatnonzero(rows & np.isin(comparison.time, times)
                                & comparison.corrected_differently)
    n_diff = _corrected_differently(top, [position[key(i)] for i in drawn_rows], note=False)
    top.axhline(0, color="#000000", linewidth=0.8)
    top.set_ylim(-1.1, 1.1)
    top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}"))
    top.set_ylabel(_label(comparison.quantity))
    top.text(0.005, 0.98, comparison.reference, transform=top.transAxes, ha="left", va="top",
             fontsize=9)
    top.text(0.005, 0.02, sample, transform=top.transAxes, ha="left", va="bottom", fontsize=9)
    top.grid(color="#e5e5e5", linewidth=0.8)
    handles, _ = top.get_legend_handles_labels()
    if n_diff:  # in the legend, not over the panel (owner)
        handles.append(Line2D([], [], linestyle="none", marker="$*$", markersize=7,
                              color="#4d4d4d", label="corrected differently"))
    top.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False,
               fontsize=7, ncol=len(handles))
    drawn = np.flatnonzero(rows & np.isin(comparison.time, times))
    x = np.array([position[key(i)] for i in drawn], dtype=float)
    _threshold_lines(bottom, comparison, drawn, x - 0.4, x + 0.4)
    bottom.axhline(0, color="#000000", linewidth=0.8, zorder=1.6)  # above the grid
    what = "ΔfD" if comparison.quantity == "fD" else "Δpopulation"
    bottom.set_ylabel(what)
    bottom.grid(color="#e5e5e5", linewidth=0.8)
    step = max(1, int(np.ceil(len(keys) / max_labels)))
    ticks = np.arange(0, len(keys), step)
    bottom.set_xticks(ticks, [f"{keys[i][0]}-{keys[i][1]}" for i in ticks], rotation=90,
                      fontsize=6)
    bottom.set_xlim(-0.5, len(keys) - 0.5)
    bottom.set_xlabel("peptide (residues)")
    return fig


def woods_figure(
    comparisons: Sequence[tuple[str, PeptideComparison]],
    sample: str,
    time: float,
    *,
    annotations: Annotations | None = None,
    length: int | None = None,
    time_scale: str = "seconds",
    note: str = "",
) -> Figure:
    """Woods plots of ``sample`` stacked, one per ``(title, comparison)`` (for example K=1, the
    K=2 ranks and the population), sharing the residue axis; the fD panels and a population
    scaled by its separation (in fD units) share one y range. ``note``: a line at the top of
    the page.

    ``time_scale`` (:attr:`hxex.report.Results.time_scale`): with ``'order'`` the time is
    labelled as a point number, not a time."""
    fig = Figure(figsize=(7.5, 2.7 * len(comparisons) + 0.3), layout="constrained")
    axes = fig.subplots(len(comparisons), 1, sharex=True, squeeze=False)[:, 0]
    for n, (ax, (title, comparison)) in enumerate(zip(axes, comparisons, strict=True)):
        plot_peptide_differences(comparison, sample, time, annotations=annotations,
                                 length=length, ax=ax, time_scale=time_scale)
        ax.set_title(title, fontsize=10, loc="left")
        if n < len(comparisons) - 1:
            ax.set_xlabel("")
    if note:  # one line for the whole page, e.g. which peptides were compared
        fig.suptitle(note, fontsize=9, x=0.99, ha="right", color="#4d4d4d")
    fd = [ax for ax, (_, c) in zip(axes, comparisons, strict=True)
          if c.quantity == "fD" or c.scaled]  # in fD units: one y range
    if fd:
        lim = max(abs(ax.get_ylim()[0]) for ax in fd)
        for ax in fd:
            ax.set_ylim(-lim, lim)
    return fig
