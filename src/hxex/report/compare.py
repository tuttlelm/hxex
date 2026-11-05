# SPDX-License-Identifier: GPL-3.0-or-later
"""Plots of comparisons (docs/ANALYSIS.md, phases 3 and 4; between samples of a run or between
runs): a reference sample against the others residue by residue, peptide differences (Woods and
butterfly plots) and region bar charts. Samples keep
one colour and marker each (the analysis gives every figure the same ``styles``, in the order
of ``[analysis] samples``; without them, :func:`hxex.report.coverage.sample_style`, the
reference first, in black). Functions return figures and never show them."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence

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
from . import hover
from .coverage import MARK_LINE, SampleStyle, sample_hatch, sample_style
from .residues import _when

INCREASE, DECREASE, NEUTRAL = "#c0392b", "#1f5fa8", "#b5b5b5"
FILL = 0.65  # residue figures: marker fill and K=2 error bars; outlines solid (owner, 2026-10-21)
K2_MARKERS = {"low": "^", "high": "v", "pop_low": "^", "pop_high": "v"}  # rank 1 up, 2 down
COUNT_GREY = "#7f7f7f"
KEY_GREY = "#4d4d4d"  # legend entries for shapes and lines: not a sample's colour
# what each value's mark is called in the residue figures' legends
# (the residue values' own names, as the axes write them: "population" is kept for the
# population figures, owner 2026-11-02)
SHAPE_LABELS = {"avg": r"fD$_{\rm avg}$", "low": r"fD$_{\rm low}$", "high": r"fD$_{\rm high}$",
                "pop_low": r"pop$_{\rm low}$", "pop_high": r"pop$_{\rm high}$"}
THRESHOLD_LABEL = "critical difference (±)"


def _style(n: int):
    """The style of the sample at position ``n`` (the reference at 0; :func:`sample_style`)."""
    return sample_style(n)


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


def _difference_key(quantity: str, reference: str, share: str = "low") -> list:
    """The legend of the Woods plot's colours and dashed lines: what red and blue mean for
    ``quantity`` (sample minus ``reference``; for a population, the ``share`` compared), grey,
    and the critical difference."""
    from matplotlib.patches import Patch

    if quantity == "fD":
        more = f"significantly more D than {reference}"
        less = f"significantly less D than {reference}"
    elif share == "high":  # the high-D share (times the separation when scaled: same sign)
        more = f"significantly larger high-D share than {reference} (more D)"
        less = f"significantly smaller high-D share than {reference} (less D)"
    else:  # the low-D population's share
        more = f"significantly larger low-D share than {reference}"
        less = f"significantly smaller low-D share than {reference}"
    return [Patch(color=INCREASE, label=more), Patch(color=DECREASE, label=less),
            Patch(color=NEUTRAL, label="not significant"),
            Line2D([], [], color="#555555", linestyle="--", linewidth=0.9,
                   label=THRESHOLD_LABEL)]


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


def _rank_points(ax: Axes, x, y, se, colour: str, marker: str, zorder: float, gid: str):
    """Points with capped standard-error bars, as every residue figure draws them: a 65 %
    fill with a solid outline, error bars at the same opacity. Returns the points (for their
    data-table rows)."""
    small = marker == "o"
    bars = ax.errorbar(x, y, yerr=np.nan_to_num(se), fmt=marker,
                       markersize=3.5 if small else 5.0, color=colour,
                       markerfacecolor=to_rgba(colour, FILL), markeredgecolor=colour,
                       markeredgewidth=0.7, ecolor=to_rgba(colour, FILL), elinewidth=0.9,
                       capsize=2.0, capthick=0.9, zorder=zorder)
    bars.lines[0].set_gid(gid)
    return bars.lines[0]


def _value_rows(artist, sample: str, reference: str, time: float, time_scale: str,
                measure: str, x, y, se) -> None:
    """Give ``artist`` (a sample's points of a residue overlay) one data-table row per residue:
    the sample, the figure's reference, the time, what is drawn (``measure``), the residue and
    its value with its standard error."""
    hover.set_rows(artist, [
        {"sample": sample, "reference": reference, "time": float(time),
         "time_label": hover.time_label(time, time_scale), "measure": measure,
         "residue": int(r), "value": float(v), "value_se": float(e)}
        for r, v, e in zip(np.asarray(x), np.asarray(y, dtype=float),
                           np.asarray(se, dtype=float), strict=True)])


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
                  colours: tuple[str, str], n_diff: int, time: float, time_scale: str,
                  annotations: Annotations | None, top: float = -np.inf,
                  shapes: Sequence[str] = ()) -> None:
    """The axes, legend (the samples' colours, the shapes of ``shapes``, keys of
    :data:`SHAPE_LABELS`, in grey when more than a circle is drawn, and the
    corrected-differently mark when drawn; a single triangle shape, as on the population
    overlay, is drawn in each sample's colour instead) and time label every residue figure
    shares; ``top`` is the highest point drawn."""
    _marks(ax, annotations, label_sites=False)
    ax.set_xlim(0, length)
    ax.set_ylim(-0.05, _upper_limit(top))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.set_xlabel("Residue")
    ax.set_ylabel(ylabel)
    ax.grid(color="#e5e5e5", linewidth=0.8)
    named = [v for v in shapes if v != "avg"]  # a circle alone needs no name
    # one triangle shape and nothing else (the population overlay): the samples' keys are that
    # triangle in their colours, which the axis label names; else squares and grey shapes
    alone = K2_MARKERS[named[0]] if len(shapes) == 1 and named else None
    handles = [Line2D([], [], linestyle="none",
                      marker=alone or ("s" if named else "o"), markersize=5,
                      color=colour, markerfacecolor=to_rgba(colour, FILL), label=name)
               for name, colour in zip((reference, sample), colours, strict=True)]
    if named and alone is None:  # the shapes, in grey: the colours are the samples'
        for value in shapes:
            avg = value == "avg"
            handles.append(Line2D([], [], linestyle="-" if avg else "none",
                                  linewidth=1.0, marker="o" if avg else K2_MARKERS[value],
                                  markersize=3 if avg else 5, color=KEY_GREY,
                                  markerfacecolor=to_rgba(KEY_GREY, FILL),
                                  label=SHAPE_LABELS[value]))
    if n_diff:
        handles.append(Line2D([], [], linestyle="none", marker="$*$", markersize=7,
                              color="#4d4d4d", label="corrected differently"))
    many = len(handles) > 3  # from the left, clear of the time label on the right
    ax.legend(handles=handles, loc="lower left" if many else "lower center",
              bbox_to_anchor=(0.0 if many else 0.5, 0.98), ncol=len(handles), frameon=False,
              fontsize=7.5 if many else 9, handletextpad=0.2, columnspacing=1.0,
              borderaxespad=0.0 if many else 0.5, handlelength=1.3)  # fDavg's line: short
    ax.text(1.0, 1.02, _when(time, time_scale), transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9)


def _overlay_label(show: Sequence[str]) -> str:
    """``fD`` for deuteration values (the legend names which), and ``pop_low, pop_high`` for
    the populations shown."""
    parts = ["fD"] if any(v in show for v in ("low", "avg", "high")) else []
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
    style_index = max(1, int(style_index))  # the reference's style (0) is the reference's only
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
    shape = ("avg" if comparison.K == 1 else
             ("pop_low" if comparison.rank == 1 else "pop_high")
             if comparison.quantity == "population" else
             "low" if comparison.rank == 1 else "high")
    _residue_axes(ax, length=length if length is not None else int(comparison.residue.max()),
                  ylabel=_label(comparison.quantity), reference=comparison.reference,
                  sample=sample, colours=(_style(0)[0], _style(style_index)[0]),
                  n_diff=n_diff, time=time,
                  time_scale=time_scale, annotations=annotations, top=top, shapes=(shape,))
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
    styles: Mapping[str, SampleStyle] | None = None,
) -> Figure:
    """``reference`` and ``sample`` in their colours (``styles``, else black for the reference
    and the style at ``style_index`` for the sample; :func:`hxex.report.coverage.sample_styles`)
    residue by residue at
    ``time``, the sample drawn over the reference. ``show`` is any of ``'low'`` and ``'high'``
    (the K=2 ranks 1 and 2: up and down triangles with capped standard-error bars),
    ``'pop_low'`` and ``'pop_high'`` (the populations of those ranks, from ``choice``, which they
    need: the same triangles) and ``'avg'`` (K=1: small
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
    style_index = max(1, int(style_index))  # the reference's style (0) is the reference's only
    if ax is None:
        fig = Figure(figsize=(7.5, 3.0))
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    drawn, top = 0, -np.inf
    drawn_values: set[str] = set()  # for the legend
    shown: dict[str, set] = {reference: set(), sample: set()}
    fell_back: dict[str, set] = {reference: set(), sample: set()}
    colours = tuple(styles[name].colour if styles and name in styles else _style(index)[0]
                    for name, index in ((reference, 0), (sample, style_index)))
    for n, (name, colour) in enumerate(zip((reference, sample), colours, strict=True)):
        for value in show:
            if value.startswith("pop_"):  # a population: the sample's choice, where it keeps two
                rows = (choice.where(sample=name, time=time)
                        & np.asarray(choice.two_populations, dtype=bool)
                        & np.isfinite(choice.population_low))
                if not rows.any():
                    continue
                drawn += 1
                drawn_values.add(value)
                shown[name].update(choice.residue[rows].tolist())
                fell_back[name].update(choice.residue[rows & choice.control_fallback].tolist())
                pop = choice.population_low[rows]
                pop = pop if value == "pop_low" else 1 - pop
                points = _rank_points(ax, choice.residue[rows], pop,
                                      choice.population_low_se[rows], colour, K2_MARKERS[value],
                                      3 + n, f"overlay:{name}:{value}")
                _value_rows(points, name, reference, time, time_scale,
                            "low-D share" if value == "pop_low" else "high-D share",
                            choice.residue[rows], pop, choice.population_low_se[rows])
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
            drawn_values.add(value)
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
                _value_rows(line, name, reference, time, time_scale, "fD (K=1)",
                            residues.residue[rows], residues.fD[rows], residues.fD_se[rows])
                continue
            points = _rank_points(ax, residues.residue[rows], residues.fD[rows],
                                  residues.fD_se[rows], colour, K2_MARKERS[value], 3 + n,
                                  f"overlay:{name}:{value}")
            _value_rows(points, name, reference, time, time_scale,
                        f"fD ({hover.population(k, rank)})", residues.residue[rows],
                        residues.fD[rows], residues.fD_se[rows])
    if not drawn:
        raise ValueError(f"no residue values of {reference!r} or {sample!r} at time {time:g}")
    both = shown[reference] & shown[sample]
    n_diff = (_corrected_differently(
        ax, [r for r in both if (r in fell_back[reference]) != (r in fell_back[sample])],
        note=False) if mark_corrected_differently else 0)
    _residue_axes(ax, length=length if length is not None else int(residues.residue.max()),
                  ylabel=_overlay_label(show), reference=reference, sample=sample,
                  colours=colours, n_diff=n_diff, time=time, time_scale=time_scale,
                  annotations=annotations, top=top,
                  shapes=[v for v in SHAPE_LABELS if v in drawn_values])
    return fig


def _average_line(ax: Axes, residues: ResidueMap, sample: str, time: float, colour: str,
                  zorder: float, band_alpha: float = 0.3, half_width: float | None = None) -> None:
    """The K=1 residue values of ``sample`` at ``time`` as a line with a +-1 standard-error
    band (``half_width``: a band of that half-width instead) in ``colour``, broken between
    residues no peptide measures together; a residue without
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
    if half_width is not None:
        es = np.where(np.isfinite(ys), half_width, np.nan)
    ax.fill_between(xs, ys - es, ys + es, color=colour, alpha=band_alpha, linewidth=0,
                    zorder=zorder)
    ax.plot(xs, ys, color=colour, linewidth=1.0, alpha=0.9, zorder=zorder + 0.05)


def plot_residue_samples(
    residues: ResidueMap,
    samples: Sequence[str],
    time: float,
    *,
    styles: Mapping[str, SampleStyle] | None = None,
    annotations: Annotations | None = None,
    length: int | None = None,
    time_scale: str = "seconds",
    ax: Axes | None = None,
    top: float | None = None,
    delta_crit: float | None = None,
) -> Figure:
    """Every sample's average deuteration (fDavg: the K=1 fit's, or, for a set with two
    populations at least 2 D apart, the K = 2 population-weighted mean; ``average_from_k2``,
    HISTORY 210) residue by residue at ``time`` on one axis,
    none of them a reference (owner, 2026-11-04: "the all samples fDavg overlay, so not head to
    head"; drawn as redundancy and resolution draw the samples): each sample in its colour and
    marker (``styles``, :func:`hxex.report.coverage.sample_styles`; else the default style of
    its place), a line broken between residues no peptide measures together, with a light
    +-1 standard-error band; the samples named above the panel, the time on its right; sites
    and region boundaries dashed. Each sample's line carries its residues as data-table rows
    (``hover.set_rows``). ``top``: the highest value the y axis must hold (the same for every
    time of an analysis, so the figures compare; owner, 2026-11-04); else this time's own.
    ``delta_crit`` (the K=1 threshold of the analysis's peptide comparisons, one value for every
    pair of samples): each band is +-delta_crit / 2 instead of +-1 standard error, so two
    samples' bands part where their difference exceeds it, and its value is said above the
    plot, on the left (owner, 2026-11-04)."""
    if ax is None:
        fig = Figure(figsize=(8.0, 3.4), layout="constrained")  # the axis label kept in
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    fixed, top, handles = top, -np.inf, []
    crit = (float(delta_crit) if delta_crit is not None and np.isfinite(delta_crit)
            and delta_crit > 0 else None)
    for n, name in enumerate(samples):
        rows = residues.where(sample=name, time=time, K=1, rank=1) & np.isfinite(residues.fD)
        if not rows.any():
            continue
        colour, marker, size = ((styles[name].colour, styles[name].marker, styles[name].size)
                                if styles and name in styles else sample_style(n))
        _average_line(ax, residues, name, time, colour, zorder=1.7 + 0.1 * n,
                      band_alpha=0.12 if crit is None else 0.18,
                      half_width=None if crit is None else crit / 2)
        res, fd, se = residues.residue[rows], residues.fD[rows], residues.fD_se[rows]
        (points,) = ax.plot(res, fd, linestyle="none", marker=marker, markersize=size * 0.8,
                            color=colour, markerfacecolor=to_rgba(colour, FILL),
                            markeredgecolor=colour, markeredgewidth=0.6, zorder=2.5 + 0.1 * n)
        points.set_gid(f"samples:{name}")
        hover.set_rows(points, [
            {"sample": name, "time": float(time), "time_label": hover.time_label(time, time_scale),
             "residue": int(r), "fD": float(v), "fD_se": float(e)}
            for r, v, e in zip(res, fd, se, strict=True)])
        top = max(top, _top(fd, se))
        handles.append(Line2D([], [], color=colour, marker=marker, markersize=size * 0.8,
                              markerfacecolor=to_rgba(colour, FILL), linewidth=1.0, label=name))
    if not handles:
        raise ValueError(f"no residue values at time {time:g}")
    _marks(ax, annotations, label_sites=False)
    ax.set_xlim(0, length if length is not None else int(residues.residue.max()))
    ax.set_ylim(-0.05, _upper_limit(top if fixed is None else fixed))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.set_xlabel("Residue")
    ax.set_ylabel(r"fD$_{\rm avg}$")  # the K=1 fit, or the K=2 average of separated twos
    ax.grid(color="#e5e5e5", linewidth=0.8)
    many = len(handles) > 4
    ax.legend(handles=handles, loc="lower left" if many else "lower center",
              bbox_to_anchor=(0.0 if many else 0.5, 0.98), ncol=len(handles), frameon=False,
              fontsize=7.5 if many else 9, handletextpad=0.3, columnspacing=1.0,
              borderaxespad=0.0 if many else 0.5, handlelength=1.6)
    ax.text(1.0, 1.02, _when(time, time_scale), transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9)
    many_note = len(handles) > 4  # the legend runs from the left: the note inside instead
    note = (f"±½ δcrit (δcrit = {crit:.3f} fD)" if crit is not None  # short: beside the legend
            else "±1 standard error")
    if many_note:
        ax.text(0.01, 0.97, note, transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
    else:
        ax.set_title(note, loc="left", fontsize=8)
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
    mark_corrected_differently: bool = True,
    styles: Mapping[str, SampleStyle] | None = None,
) -> Figure:
    """One :func:`plot_residue_overlay` panel per sample of ``samples`` against ``reference``;
    each sample keeps its colour (``styles``: the analysis's, so it is the same in every
    figure; else by its place among ``samples``). ``choice`` filters the K=2 values as in
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
                             mark_corrected_differently=mark_corrected_differently,
                             styles=styles)
    return fig


def _difference_hover(artist, comparison: PeptideComparison, i: int, sample: str,
                      time_scale: str = "seconds") -> None:
    """A Woods bar's tooltip (the peptide, both values, the difference and the verdict) and its
    data-table row (the columns of ``compare_peptides.csv``)."""
    name = (f"{comparison.share}-D share" if comparison.quantity == "population"
            else "fD" if comparison.K == 1 else
            f"fD ({hover.population(comparison.K, comparison.rank)})")
    reason = str(comparison.reason[i])
    hover.set_hover(
        artist, hover.peptide(comparison.peptide[i], comparison.modification[i]),
        hover.residues(comparison.start[i], comparison.end[i]),
        hover.value(f"{name}, {sample}", comparison.value[i], comparison.value_se[i]),
        hover.value(f"{name}, {comparison.reference}", comparison.value_reference[i],
                    comparison.value_reference_se[i]),
        hover.value("difference", comparison.delta[i], comparison.delta_se[i], signed=True),
        reason if reason else ("significant" if comparison.significant[i]
                               else "not significant"),
        "corrected differently (one side with a fallback control)"
        if comparison.corrected_differently[i] else "")
    time = float(comparison.time[i])
    hover.set_row(artist, {
        "sample": sample, "reference": str(comparison.reference), "time": time,
        "time_label": hover.time_label(time, time_scale), "measure": name,
        "start": int(comparison.start[i]), "end": int(comparison.end[i]),
        "peptide": str(comparison.peptide[i]), "modification": str(comparison.modification[i]),
        "value": float(comparison.value[i]), "value_se": float(comparison.value_se[i]),
        "value_reference": float(comparison.value_reference[i]),
        "value_reference_se": float(comparison.value_reference_se[i]),
        "delta": float(comparison.delta[i]), "delta_se": float(comparison.delta_se[i]),
        "significant": bool(comparison.significant[i]), "reason": reason,
        "corrected_differently": bool(comparison.corrected_differently[i])})


def plot_peptide_differences(
    comparison: PeptideComparison,
    sample: str,
    time: float,
    *,
    annotations: Annotations | None = None,
    length: int | None = None,
    ax: Axes | None = None,
    time_scale: str = "seconds",
    legend: bool = True,
) -> Figure:
    """The Woods plot of ``sample`` against the reference at ``time``: each peptide a horizontal
    bar over its residues at its difference in ``fD``, red if significantly higher, blue if lower,
    grey otherwise, with the standard error as a vertical bar and the threshold dashed (one
    line, ``+-delta_crit``; for a population as it is, a short line over each peptide, and one
    line for :meth:`~hxex.analysis.PeptideComparison.scaled_by_separation`'s copy); a
    ``*`` at the top over the middle of each peptide whose two values were corrected differently
    (one with a fallback control). With ``legend``, the colours and the dashed line are named
    above the panel, on the right. Bars have the gid ``peptide:<start>-<end>``.

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
        _difference_hover(line, comparison, i, sample, time_scale)
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
    pop = "Δhigh-D share" if comparison.share == "high" else "Δpop"
    what = ("ΔfD" if comparison.quantity == "fD" else f"{pop} × sep"
            if comparison.scaled else pop if comparison.share == "high" else "Δpopulation")
    ax.set_ylabel(f"{what} ({sample} − {comparison.reference})")
    ax.grid(color="#e5e5e5", linewidth=0.8)
    n_sig = int(comparison.significant[rows].sum())
    n_compared = int(np.isfinite(comparison.delta[rows]).sum())  # a population may not be
    ax.text(0.99, 0.03, f"{_when(time, time_scale)}; {n_sig} of {n_compared} significant",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    if legend:
        ax.legend(handles=_difference_key(comparison.quantity, comparison.reference,
                                          comparison.share),
                  loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False,
                  fontsize=7, handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)
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
            hover.set_hover(bars, hover.residues(a, b), {
                "sample": f"two populations in {sample} only",
                "reference": f"two populations in {counts.reference} only",
                "both": f"two populations in {sample} and {counts.reference}"}[band],
                "borderline" if borderline else "")
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
    styles: Mapping[str, SampleStyle] | None = None,
) -> Figure:
    """Grouped bars of the mean ``fD`` of every region, one bar per
    sample in its colour (``styles``, else by its place among ``samples``), with standard-error
    bars; regions without a value are left empty. Each bar has the gid
    ``region:<sample>:<region>``.

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
        if styles and sample in styles:
            colour, hatch = styles[sample].colour, styles[sample].hatch
        else:  # past the palette the colours repeat, with a hatch
            colour, hatch = _style(n)[0], sample_hatch(n)
        values, errors = [], []
        for name in names:
            row = sel & (regions.sample == sample) & (regions.region == name)
            values.append(float(regions.fD[row][0]) if row.any() else np.nan)
            errors.append(float(regions.fD_se[row][0]) if row.any() else np.nan)
        bars = ax.bar(x + (n - (len(samples) - 1) / 2) * width, values, width=width,
                      color=colour, label=sample, yerr=np.nan_to_num(errors), hatch=hatch,
                      edgecolor="white" if hatch else None,
                      error_kw={"elinewidth": 0.7, "ecolor": "#555555"}, alpha=0.9)
        table = regions.columns()
        for patch, name, v, se in zip(bars.patches, names, values, errors, strict=True):
            patch.set_gid(f"region:{sample}:{name}")
            hover.set_hover(patch, f"{sample}, region {name}", hover.value("fD", v, se))
            hover.set_table(patch, table, sel & (regions.sample == sample)
                            & (regions.region == name),
                            time_label=hover.time_label(time, time_scale))
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
    sequential colour per time), with the differences below (a filled circle where significant,
    an ``x`` where not, ``+-delta_crit`` dashed; all three named in a legend above that
    panel); a ``*`` at the top over each peptide whose two values were corrected differently at
    a time drawn (one with a fallback control; named in the legend). ``times``
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
        bottom.scatter(x[~sig], comparison.delta[here][~sig], marker="x", color=colour, s=14,
                       linewidths=1.0, zorder=3)
    drawn_rows = np.flatnonzero(rows & np.isin(comparison.time, times)
                                & comparison.corrected_differently)
    n_diff = _corrected_differently(top, [position[key(i)] for i in drawn_rows], note=False)
    top.axhline(0, color="#000000", linewidth=0.8)
    drawn = np.concatenate([comparison.value_reference[rows], comparison.value[rows]])
    span = 1.05 * max(1.05, float(np.nanmax(np.abs(drawn))) if np.isfinite(drawn).any() else 0)
    top.set_ylim(-span, span)
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
    key = [Line2D([], [], linestyle="none", marker="o", markersize=4, color=KEY_GREY,
                  label="significant difference"),
           Line2D([], [], linestyle="none", marker="x", markersize=3.6, color=KEY_GREY,
                  markeredgewidth=1.0, label="not significant"),
           Line2D([], [], color="#555555", linestyle="--", linewidth=0.9,
                  label=THRESHOLD_LABEL)]
    bottom.legend(handles=key, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(key),
                  frameon=False, fontsize=7, borderaxespad=0.2)
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
        # the colours named where their meaning changes: the first panel, then the first
        # population panel
        new = n == 0 or comparison.quantity != comparisons[n - 1][1].quantity
        plot_peptide_differences(comparison, sample, time, annotations=annotations,
                                 length=length, ax=ax, time_scale=time_scale, legend=new)
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
