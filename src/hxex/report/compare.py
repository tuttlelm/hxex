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
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, FuncFormatter

from ..analysis import (
    Annotations,
    PeptideComparison,
    PopulationChoice,
    RegionTable,
    ResidueComparison,
    ResidueMap,
)
from ..analysis.settings import OVERLAY_VALUES
from .coverage import MARK_LINE, SAMPLE_STYLES
from .residues import _when

INCREASE, DECREASE, NEUTRAL = "#c0392b", "#1f5fa8", "#b5b5b5"


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


def _label(quantity: str) -> str:
    return "population (low D)" if quantity == "population" else "fD"


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
) -> Figure:
    """The reference (black) and ``sample`` residue by residue at ``time``, in the comparison's
    quantity (``fD``, or the low-D population from :func:`hxex.analysis.compare_populations`),
    with standard-error bars; residues with a significant difference get a band at the top of
    the axes (red up, blue down), and residues whose two values were corrected differently (one
    with a fallback control) a ``*`` below it.

    ``time_scale`` (:attr:`hxex.report.Results.time_scale`): with ``'order'`` the time is
    labelled as a point number, not a time."""
    rows = comparison.where(sample=sample, time=time)
    if not rows.any():
        raise ValueError(f"no comparison of {sample!r} at time {time:g}")
    if ax is None:
        fig = Figure(figsize=(7.5, 2.8))
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    order = np.argsort(comparison.residue[rows])
    x = comparison.residue[rows][order]
    for n, (name, value, se) in enumerate((
            (comparison.reference, comparison.value_reference, comparison.value_reference_se),
            (sample, comparison.value, comparison.value_se))):
        colour, marker, size = _style(0 if n == 0 else style_index)
        bars = ax.errorbar(x, value[rows][order], yerr=np.nan_to_num(se[rows][order]),
                           color=colour, marker=marker, markersize=size * 0.8, linewidth=1.0,
                           elinewidth=0.8, capsize=0, label=name, zorder=4 if n == 0 else 3,
                           alpha=0.9)
        bars.lines[0].set_gid(f"{comparison.quantity}:{name}")
    sig = rows & comparison.significant
    for res, d in zip(comparison.residue[sig], comparison.delta[sig], strict=True):
        ax.add_patch(Rectangle((res - 0.5, 0.96), 1.0, 0.04, transform=ax.get_xaxis_transform(),
                               color=INCREASE if d > 0 else DECREASE, linewidth=0,
                               gid=f"significant:{res}"))
    _corrected_differently(ax, comparison.residue[rows & comparison.corrected_differently],
                           y=0.955)
    _marks(ax, annotations, label_sites=False)
    ax.set_xlim(0, length if length is not None else int(comparison.residue.max()))
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(_label(comparison.quantity))
    ax.grid(color="#e5e5e5", linewidth=0.8)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8)
    ax.text(0.99, 0.04, _when(time, time_scale), transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9)
    return fig


def residue_comparison_figure(
    comparison: ResidueComparison,
    samples: Sequence[str],
    time: float,
    *,
    annotations: Annotations | None = None,
    length: int | None = None,
    time_scale: str = "seconds",
) -> Figure:
    """One panel per sample of ``samples`` against the comparison's reference, sharing the
    residue axis; each sample keeps its colour.

    ``time_scale`` (:attr:`hxex.report.Results.time_scale`): with ``'order'`` the time is
    labelled as a point number, not a time."""
    samples = [s for s in samples if s != comparison.reference]
    fig = Figure(figsize=(7.5, 2.3 * len(samples) + 0.4), layout="constrained")
    axes = fig.subplots(len(samples), 1, sharex=True, squeeze=False)[:, 0]
    for n, (ax, sample) in enumerate(zip(axes, samples, strict=True)):
        plot_residue_comparison(comparison, sample, time, style_index=n + 1,
                                annotations=annotations, length=length, ax=ax,
                                time_scale=time_scale)
        if n < len(samples) - 1:
            ax.set_xlabel("")
    axes[-1].set_xlabel("residue")
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
) -> Figure:
    """``reference`` (black) and ``sample`` (its colour) residue by residue at ``time``, as
    points with capped standard-error bars, no lines, the
    sample drawn over the reference. ``show`` is any of ``'low'`` and ``'high'`` (the K=2 ranks
    1 and 2, filled points) and ``'avg'`` (K=1, open points). With ``choice``
    (:func:`hxex.analysis.choose_populations`), the K=2 values are drawn only where the choice
    keeps two populations; the average is always drawn. A ``*`` at the
    top marks residues drawn for both whose values were corrected differently (one with a
    fallback control). Point groups have the gid ``overlay:<sample>:<value>``.

    ``time_scale`` (:attr:`hxex.report.Results.time_scale`): with ``'order'`` the time is
    labelled as a point number, not a time."""
    unknown = [v for v in show if v not in OVERLAY_VALUES]
    if unknown or not show:
        raise ValueError(f"show must be some of {', '.join(OVERLAY_VALUES)}")
    if ax is None:
        fig = Figure(figsize=(7.5, 3.0))
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    drawn = 0
    shown: dict[str, set] = {reference: set(), sample: set()}
    fell_back: dict[str, set] = {reference: set(), sample: set()}
    for n, (name, index) in enumerate(((reference, 0), (sample, style_index))):
        colour = _style(index)[0]
        for value in show:
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
            bars = ax.errorbar(
                residues.residue[rows], residues.fD[rows],
                yerr=np.nan_to_num(residues.fD_se[rows]), fmt="o",
                markersize=3.0 if n == 0 else 4.0, color=colour,
                markerfacecolor="white" if value == "avg" else colour,
                markeredgewidth=0.8, ecolor=colour, elinewidth=0.9, capsize=2.0,
                capthick=0.9, alpha=0.85, zorder=3 + n)
            bars.lines[0].set_gid(f"overlay:{name}:{value}")
    if not drawn:
        raise ValueError(f"no residue values of {reference!r} or {sample!r} at time {time:g}")
    both = shown[reference] & shown[sample]
    n_diff = _corrected_differently(
        ax, [r for r in both if (r in fell_back[reference]) != (r in fell_back[sample])],
        note=False)
    _marks(ax, annotations, label_sites=False)
    ax.set_xlim(0, length if length is not None else int(residues.residue.max()))
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.set_xlabel("Residue")
    ax.set_ylabel(", ".join(rf"fD$_{{\rm {v}}}$" for v in show))
    ax.grid(color="#e5e5e5", linewidth=0.8)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=6,
                      color=_style(i)[0], label=name)
               for name, i in ((reference, 0), (sample, style_index))]
    if "avg" in show and len(show) > 1:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=6,
                              color="#555555", markerfacecolor="white", label="avg (K=1)"))
    if n_diff:
        handles.append(Line2D([], [], linestyle="none", marker="$*$", markersize=7,
                              color="#4d4d4d", label="corrected differently"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.98),
              ncol=len(handles), frameon=False, fontsize=9, handletextpad=0.2)
    ax.text(1.0, 1.02, _when(time, time_scale), transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9)
    return fig


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
                             annotations=annotations, length=length,
                             ax=ax)
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
    grey otherwise, with the standard error as a vertical bar and ``+-delta_crit`` dashed; a
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
    if np.isfinite(comparison.delta_crit):
        for y in (comparison.delta_crit, -comparison.delta_crit):
            ax.axhline(y, color="#555555", linestyle="--", linewidth=0.9, zorder=1)
    ax.axhline(0, color="#000000", linewidth=0.8, zorder=1)
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
    what = "ΔfD" if comparison.quantity == "fD" else "Δpopulation"
    ax.set_ylabel(f"{what} ({sample} − {comparison.reference})")
    ax.grid(color="#e5e5e5", linewidth=0.8)
    n_sig = int(comparison.significant[rows].sum())
    ax.text(0.99, 0.03, f"{_when(time, time_scale)}; {n_sig} of {len(rows)} significant",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
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
    labels = [textwrap.fill(name.replace(",", ", "), 10).replace(", ", ",") for name in names]
    ax.set_xticks(x, labels, rotation=0, fontsize=8)
    ax.set_ylim(0, 1.05)
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
    if np.isfinite(comparison.delta_crit):
        for y in (comparison.delta_crit, -comparison.delta_crit):
            bottom.axhline(y, color="#555555", linestyle="--", linewidth=0.9, zorder=1)
    bottom.axhline(0, color="#000000", linewidth=0.8, zorder=1)
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
) -> Figure:
    """Woods plots of ``sample`` stacked, one per ``(title, comparison)`` (for example K=1, the
    K=2 ranks and the population), sharing the residue axis; the fD panels share one y range.

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
    fd = [ax for ax, (_, c) in zip(axes, comparisons, strict=True) if c.quantity == "fD"]
    if fd:
        lim = max(abs(ax.get_ylim()[0]) for ax in fd)
        for ax in fd:
            ax.set_ylim(-lim, lim)
    return fig
