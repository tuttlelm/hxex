# SPDX-License-Identifier: GPL-3.0-or-later
"""Protection-factor figures (:mod:`hxex.analysis.protection`), in the layout of the coverage
map (:func:`hxex.report.coverage.plot_coverage`): each peptide a bar over its residues,
stacked downwards, one panel per K and rank (the average, then the K = 2 ranks).

:func:`protection_figure` colours the bars by ``log10_protection`` (the uptake map's colours
reversed, so that protected peptides are dark as on the uptake maps); a peptide with only a
bound is hatched, coloured at its limit. :func:`protection_difference_figure` colours them by
the difference from the reference sample (a diverging map, red more protected); differences
that are not significant are drawn pale, and peptides without a difference (a bound) grey and
hatched. Functions return figures and never show them.
"""

from __future__ import annotations

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

from .coverage import FD_CMAP, autowrap

PROTECTION_CMAP = FD_CMAP.reversed().with_extremes(bad="#d0d0d0")
_PANELS = ((1, 1, "average (K=1)"), (2, 1, "K=2 rank 1 (low D)"), (2, 2, "K=2 rank 2 (high D)"))


def _panels(table, sample):
    return [(k, r, name) for k, r, name in _PANELS if table.where(sample=sample, K=k, rank=r).any()]


def _bars(ax, table, rows, colour, hatch, length, wrap):
    order = rows[np.lexsort((table.end[rows], table.start[rows]))]
    for n, i in enumerate(order):
        rect = Rectangle((table.start[i] - 0.5, -(n % wrap) - 1), table.end[i] - table.start[i] + 1,
                         1, facecolor=colour(i), edgecolor="k", linewidth=0.3, hatch=hatch(i))
        rect.set_gid(f"peptide:{table.start[i]}-{table.end[i]}")
        ax.add_patch(rect)
    ax.set_xlim(0, length)
    ax.set_ylim(-wrap, 0)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.grid(axis="x", color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _layout(table, sample, panels, length, cell):
    rows = [np.flatnonzero(table.where(sample=sample, K=k, rank=r)) for k, r, _ in panels]
    wraps = [autowrap(table.start[o], table.end[o])
             for o in (x[np.lexsort((table.end[x], table.start[x]))] for x in rows)]
    top = length if length is not None else int(max(table.end[x].max() for x in rows))
    fig = Figure(figsize=(cell[0] * len(panels) + 0.8, cell[1]), layout="constrained")
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    return fig, axes, rows, max(wraps), top


def protection_figure(
    table,
    sample: str,
    *,
    length: int | None = None,
    cell: tuple[float, float] = (4.6, 3.4),
) -> Figure:
    """The protection factors of ``sample``'s peptides (:class:`hxex.analysis.protection.
    ProtectionTable`): see the module docstring. ``length`` is the protein's (else the last
    covered residue)."""
    panels = _panels(table, sample)
    if not panels:
        raise ValueError(f"no protection factors of {sample!r}")
    fig, axes, rows, wrap, top = _layout(table, sample, panels, length, cell)
    pick = table.where(sample=sample)
    shown = np.where(table.bound == "", table.log10_protection, np.log10(table.limit))[pick]
    shown = shown[np.isfinite(shown)]
    lo = min(0.0, float(np.floor(shown.min()))) if len(shown) else 0.0
    hi = max(lo + 1.0, float(np.ceil(shown.max()))) if len(shown) else 1.0
    norm = Normalize(lo, hi, clip=True)

    def colour(i):
        value = table.log10_protection[i] if not table.bound[i] else np.log10(table.limit[i])
        return PROTECTION_CMAP(norm(value)) if np.isfinite(value) else PROTECTION_CMAP.get_bad()

    def hatch(i):
        return "////" if table.bound[i] else None

    for ax, sel, (_, _, name) in zip(axes, rows, panels, strict=True):
        _bars(ax, table, sel, colour, hatch, top, wrap)
        ax.set_title(f"{sample}, {name}", fontsize=10)
        ax.set_xlabel("residue")
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=PROTECTION_CMAP), ax=axes, pad=0.01,
                       fraction=0.03, aspect=40)
    bar.set_label("log10 protection factor (hatched: a bound)")
    return fig


def protection_difference_figure(
    comparison,
    sample: str,
    *,
    length: int | None = None,
    cell: tuple[float, float] = (4.6, 3.4),
) -> Figure:
    """The differences in ``log10_protection`` between ``sample`` and the reference
    (:class:`hxex.analysis.protection.ProtectionComparison`): see the module docstring."""
    import matplotlib

    panels = _panels(comparison, sample)
    if not panels:
        raise ValueError(f"no protection-factor differences of {sample!r}")
    fig, axes, rows, wrap, top = _layout(comparison, sample, panels, length, cell)
    pick = comparison.where(sample=sample)
    finite = np.abs(comparison.delta[pick][np.isfinite(comparison.delta[pick])])
    span = max(0.5, float(np.ceil(2 * finite.max()) / 2)) if len(finite) else 1.0
    norm = Normalize(-span, span, clip=True)
    cmap = matplotlib.colormaps["RdBu_r"]

    def colour(i):
        d = comparison.delta[i]
        if not np.isfinite(d):
            return "#d0d0d0"
        rgba = list(cmap(norm(d)))
        if not comparison.significant[i]:
            rgba[3] = 0.3
        return tuple(rgba)

    def hatch(i):
        return "////" if comparison.note[i] else None

    for ax, sel, (_, _, name) in zip(axes, rows, panels, strict=True):
        _bars(ax, comparison, sel, colour, hatch, top, wrap)
        ax.set_title(f"{sample} - {comparison.reference}, {name}", fontsize=10)
        ax.set_xlabel("residue")
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, pad=0.01, fraction=0.03,
                       aspect=40)
    bar.set_label("Δ log10 protection factor (pale: not significant)")
    return fig
