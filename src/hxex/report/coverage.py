# SPDX-License-Identifier: GPL-3.0-or-later
"""Coverage maps and redundancy/resolution plots of the post-fitting analysis (docs/ANALYSIS.md).

:func:`plot_coverage` draws every peptide of one sample at one time as a thin bar over its
residues, coloured by its fractional deuteration (``fD``) at a fixed K and rank, the bars ordered
by their first residue and stacked downwards, wrapping back to the top when the stack is full (the
layout of pyHDX's peptide coverage plot). :func:`coverage_figure` puts several samples and times in
a grid. :func:`plot_redundancy_resolution` overlays the samples' redundancy and resolution per
residue (from :func:`hxex.analysis.coverage_table`). Functions return figures and never show
them.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

from ..analysis import CoverageTable, SpeciesUptake
from ..io.times import format_time
from ..spectra import UNDEUT

# CET-L20 "gouldian", a perceptually uniform map by Peter Kovesi (colorcet.com, CC BY 4.0),
# sampled at 17 points: the map pyHDX uses for relative uptake
FD_COLORS = (
    "#303030", "#3c3659", "#423d7f", "#4345a2", "#3f51bc", "#3761c9", "#2977b7", "#258b95",
    "#429a76", "#6ca557", "#96ad3a", "#beb31d", "#e6b713", "#f8c214", "#fed313", "#fee60f",
    "#f9f90a",
)
FD_CMAP = LinearSegmentedColormap.from_list("hxex_fD", FD_COLORS)
FD_CMAP.set_bad("#d0d0d0")
# one colour, marker and marker size per sample (the reference sample first, in black)
# the dashed grey lines at region boundaries and sites, the same in every residue plot (owner)
MARK_LINE = {"color": "#8c8c8c", "linewidth": 1.0, "linestyle": "--", "zorder": 1}

SAMPLE_STYLES = (
    ("#000000", "s", 3.5), ("#1f77b4", "o", 4.5), ("#d62728", "o", 6.0), ("#17becf", "o", 7.0),
    ("#9467bd", "v", 5.0), ("#ff7f0e", "^", 5.0), ("#2ca02c", "D", 4.0), ("#8c564b", "P", 5.0),
)


def autowrap(start: Sequence[int], end: Sequence[int], margin: int = 4, step: int = 5) -> int:
    """The number of rows of a coverage stack: the smallest multiple of ``step`` for which no two
    peptides (in the order given, filling rows top to bottom and wrapping) share a row within
    ``margin`` residues (pyHDX's rule)."""
    start = np.asarray(start, dtype=np.int64)
    end = np.asarray(end, dtype=np.int64)
    if len(start) == 0:
        return step
    wrap = step
    while wrap < len(start):
        last = np.full(wrap, -np.inf)
        ok = True
        for row, s, e in zip(itertools.cycle(range(wrap)), start, end, strict=False):
            if s <= last[row] + margin:
                ok = False
                break
            last[row] = e
        if ok:
            return wrap
        wrap += step
    return max(wrap, len(start))


def _title(sample: str, time: float, k: int, rank: int) -> str:
    when = "undeuterated" if not np.isfinite(time) else f"t = {format_time(time)}"
    which = "" if k == 1 else f", K={k} rank {rank}"
    return f"{sample}{which}, {when}"


def plot_coverage(
    uptake: SpeciesUptake,
    sample: str,
    time: float,
    *,
    K: int = 1,
    rank: int = 1,
    length: int | None = None,
    wrap: int | None = None,
    ax: Axes | None = None,
    colorbar: bool = True,
    title: str | None = None,
) -> Figure:
    """The coverage map of ``sample`` at ``time`` (NaN: its undeuterated control; peptides
    only), coloured by ``fD`` at ``K`` and
    ``rank`` (0 to 1; grey where it is not known). ``length`` is the protein's (the x axis runs
    from 0 to it; else to the last covered residue); ``wrap`` the rows of the stack (else
    :func:`autowrap`). Each bar has the gid ``peptide:<start>-<end>``."""
    at = ({"role": UNDEUT} if not np.isfinite(time) else {"time": time, "role": ""})
    rows = np.flatnonzero(uptake.where(sample=sample, K=K, rank=rank, kind="peptide", **at))
    if not len(rows):
        raise ValueError(f"no peptides of {sample!r} at time {time:g} (K={K}, rank {rank})")
    order = rows[np.lexsort((uptake.end[rows], uptake.start[rows]))]
    start, end = uptake.start[order], uptake.end[order]
    wrap = wrap or autowrap(start, end)
    if ax is None:
        fig = Figure(figsize=(4.6, 3.4))
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    norm = Normalize(0.0, 1.0, clip=True)
    for n, i in enumerate(order):
        value = uptake.fD[i]
        colour = FD_CMAP(norm(value)) if np.isfinite(value) else FD_CMAP.get_bad()
        y = -(n % wrap) - 1
        rect = Rectangle((uptake.start[i] - 0.5, y), uptake.end[i] - uptake.start[i] + 1, 1,
                         facecolor=colour, edgecolor="k", linewidth=0.3)
        rect.set_gid(f"peptide:{uptake.start[i]}-{uptake.end[i]}")
        ax.add_patch(rect)
    top = length if length is not None else int(end.max())
    ax.set_xlim(0, top)
    ax.set_ylim(-wrap, 0)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.grid(axis="x", color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("residue")
    ax.set_title(title if title is not None else _title(sample, time, K, rank), fontsize=10)
    if colorbar:
        from matplotlib.cm import ScalarMappable

        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=FD_CMAP), ax=ax, pad=0.01)
        bar.set_label("fD")
    return fig


def coverage_figure(
    uptake: SpeciesUptake,
    *,
    samples: Sequence[str] | None = None,
    times: Sequence[float] | None = None,
    K: int = 1,
    rank: int = 1,
    length: int | None = None,
    cell: tuple[float, float] = (4.2, 3.2),
) -> Figure:
    """Coverage maps of several samples (rows) and points (columns: the undeuterated control
    when there is one, then the ``times``, by default every time point), all with one colour
    scale and the same number of rows per stack. Empty cells (no peptide of that sample at
    that point) are left blank."""
    pep = uptake.select(uptake.where(K=K, rank=rank, kind="peptide"))
    samples = list(samples) if samples is not None else sorted(set(pep.sample.tolist()))
    times = (list(times) if times is not None
             else sorted({float(t) for t in pep.time if np.isfinite(t)}))
    if (pep.role == UNDEUT).any():
        times = [np.nan, *times]  # the undeuterated control's column (its time is NaN)

    def at(s, t):
        return pep.where(sample=s, role=UNDEUT) if not np.isfinite(t) else pep.where(
            sample=s, time=t, role="")

    wraps = []
    for s in samples:
        for t in times:
            sel = np.flatnonzero(at(s, t))
            if len(sel):
                o = sel[np.lexsort((pep.end[sel], pep.start[sel]))]
                wraps.append(autowrap(pep.start[o], pep.end[o]))
    wrap = max(wraps) if wraps else None
    top = length if length is not None else (int(pep.end.max()) if len(pep) else None)
    fig = Figure(figsize=(cell[0] * len(times) + 0.8, cell[1] * len(samples)),
                 layout="constrained")
    axes = fig.subplots(len(samples), len(times), squeeze=False)
    for r, s in enumerate(samples):
        for c, t in enumerate(times):
            ax = axes[r][c]
            if not at(s, t).any():
                ax.set_axis_off()
                continue
            plot_coverage(uptake, s, t, K=K, rank=rank, length=top, wrap=wrap, ax=ax,
                          colorbar=False)
            if r < len(samples) - 1:
                ax.set_xlabel("")
    from matplotlib.cm import ScalarMappable

    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=FD_CMAP), ax=axes, pad=0.01,
                       fraction=0.03, aspect=40)
    bar.set_label("fD")
    return fig


def plot_redundancy_resolution(
    coverage: CoverageTable,
    *,
    samples: Sequence[str] | None = None,
    marks: Sequence[float] = (),
    axes: Sequence[Axes] | None = None,
) -> Figure:
    """Redundancy (top) and resolution (bottom) per residue, one line with markers per sample
    (the first in black), gaps where a residue is not covered or has no amide (a proline, or one
    of the first residues of every peptide). Prolines are marked ``P`` along the bottom of both
    panels; ``marks`` adds dashed lines at those residues (sites, region boundaries). ``axes`` is
    a pair to draw into."""
    samples = list(samples) if samples is not None else list(dict.fromkeys(coverage.sample))
    if axes is None:
        fig = Figure(figsize=(8.0, 5.2))
        top, bottom = fig.subplots(2, 1, sharex=True)
    else:
        top, bottom = axes
        fig = top.figure
    prolines: set[int] = set()
    n_res = 0
    for n, sample in enumerate(samples):
        rows = coverage.sample == sample
        if not rows.any():
            raise ValueError(f"no sample {sample!r} in the coverage table")
        colour, marker, size = SAMPLE_STYLES[n % len(SAMPLE_STYLES)]
        res = coverage.residue[rows]
        n_res = max(n_res, int(res.max()))
        prolines |= set(res[coverage.proline[rows]].tolist())
        shown = coverage.covered[rows]
        for ax, values, name in ((top, coverage.redundancy[rows], "redundancy"),
                                 (bottom, coverage.resolution[rows], "resolution")):
            y = np.where(shown, values, np.nan).astype(float)
            (line,) = ax.plot(res, y, color=colour, marker=marker, markersize=size,
                              linewidth=1.2, alpha=0.9, label=sample, zorder=10 - size / 2)
            line.set_gid(f"{name}:{sample}")
    for ax, label in ((top, "redundancy"), (bottom, "resolution")):
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(color="#e5e5e5", linewidth=0.8)
        for x in marks:
            ax.axvline(x, **MARK_LINE)
        for x in sorted(prolines):
            ax.text(x, 0.01, "P", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=7, fontweight="bold")
    top.set_ylabel("redundancy" if coverage.count == "peptides" else "redundancy (spectra)")
    bottom.set_xlabel("residue")
    bottom.set_xlim(0.5, n_res + 0.5)
    top.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(samples), frameon=False)
    return fig
