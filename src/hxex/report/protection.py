# SPDX-License-Identifier: GPL-3.0-or-later
"""Protection-factor figures (:mod:`hxex.analysis.protection`), in the layout of the coverage
map (:func:`hxex.report.coverage.plot_coverage`): each peptide a bar over its residues,
stacked downwards, one panel per K and rank (the average, then the K = 2 ranks).

:func:`protection_figure` colours the bars by ``log10_protection`` (the uptake map's colours
reversed, so that protected peptides are dark as on the uptake maps); a peptide with only a
bound is hatched, coloured at its limit. :func:`protection_difference_figure` colours them by
the difference from the reference sample (a diverging map, red more protected); differences
that are not significant are drawn pale, and peptides without a difference (a bound) grey and
hatched.

At the residue level (:func:`hxex.analysis.protection.residue_protection`), in the layout of
the residue plots: :func:`residue_protection_figure` overlays the samples' fitted
``log10_protection`` residue by residue (a panel per K and rank; filled where the peptides
resolve the residue, open where the value is the regulariser's share-out; bounds, when there
are any, as triangles at their limit, pointing the way the value lies; the legend names only
what is drawn), and :func:`residue_protection_comparison_figure`
draws a sample against the reference with a band at the top where the difference is
significant (red: more protected than the reference, blue: less). Functions return figures and
never show them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

from .compare import DECREASE, INCREASE, _marks, _style
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


def _residue_panel(ax, table, sample, k, rank, style, max_se):
    """One sample's residue values on ``ax``, those whose standard error is above ``max_se``
    (or unknown) left out; returns (how many were left out, the y values drawn, what was
    drawn: ``{'values', 'open', 'upper', 'lower'}``)."""
    rows = np.flatnonzero(table.where(sample=sample, K=k, rank=rank))
    if not len(rows):
        return 0, np.empty(0), set()
    colour, marker, size = style
    x = table.residue[rows]
    y = table.log10_protection[rows]
    raw_se = table.log10_protection_se[rows]
    se = np.nan_to_num(raw_se)
    finite = np.isfinite(y)
    ok = finite & np.isfinite(raw_se) & (raw_se <= max_se)
    hidden = int((finite & ~ok).sum())
    res = table.resolved[rows]
    # as the residue comparison plots: a line through the residues with standard-error bars,
    # broken where a residue is missing or left out; open markers where not resolved
    order = np.argsort(x)
    xs, ys, es, ks = x[order], np.where(ok, y, np.nan)[order], se[order], ok[order]
    gaps = np.flatnonzero(np.diff(xs) > 1) + 1
    xs_l = np.insert(xs.astype(float), gaps, np.nan)
    ys_l = np.insert(ys, gaps, np.nan)
    es_l = np.insert(np.where(ks, es, 0.0), gaps, 0.0)
    bars = ax.errorbar(xs_l, ys_l, yerr=es_l, color=colour, marker=None, linewidth=1.0,
                       elinewidth=0.8, capsize=0, zorder=3, alpha=0.9)
    bars.lines[0].set_gid(f"protection:{sample}")
    for keep, face in ((ok & res, colour), (ok & ~res, "white")):
        if keep.any():
            ax.scatter(x[keep], y[keep], marker=marker, s=(size * 0.8) ** 2, color=colour,
                       facecolors=face, linewidths=0.8, zorder=4,
                       gid=f"protection:{sample}:{'resolved' if face is colour else 'open'}")
    drawn = [y[ok] - se[ok], y[ok] + se[ok]]
    kinds = ({"values"} if ok.any() else set()) | ({"open"} if (ok & ~res).any() else set())
    for bound, sign in (("upper", "v"), ("lower", "^")):
        b = table.bound[rows] == bound
        if b.any():
            limits = np.log10(table.limit[rows][b])
            ax.scatter(x[b], limits, marker=sign, color=colour, s=18, zorder=3,
                       gid=f"bound:{sample}:{bound}")
            drawn.append(limits)
            kinds.add(bound)
    return hidden, np.concatenate(drawn), kinds


def _legend(ax, entries, kinds, ncol):
    """Each sample drawn with its marker and line, then the keys of what else was drawn."""
    handles = [Line2D([], [], color=colour, marker=marker, markersize=size * 0.8,
                      markerfacecolor=colour, linewidth=1.0, label=name)
               for name, (colour, marker, size) in entries]
    grey = "#4d4d4d"
    if "open" in kinds:
        handles.append(Line2D([], [], color=grey, marker="o", markerfacecolor="white",
                              linestyle="none", markersize=5, label="not resolved"))
    for bound, sign, text in (("upper", "v", "upper bound"), ("lower", "^", "lower bound")):
        if bound in kinds:
            handles.append(Line2D([], [], color=grey, marker=sign, linestyle="none",
                                  markersize=5, label=text))
    if handles:
        ax.legend(handles=handles, loc="upper left", ncol=ncol, frameon=False, fontsize=8)


def _finish_panel(ax, values, hidden, max_se):
    """The y range of what was drawn, and how many residues were left out."""
    values = values[np.isfinite(values)]
    if len(values):
        lo, hi = float(values.min()), float(values.max())
        pad = max(0.3, 0.08 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
    if hidden:
        ax.text(0.99, 0.02, f"{hidden} residue(s) with se > {max_se:g} not shown",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#4d4d4d")


def residue_protection_figure(
    table,
    samples: Sequence[str] | None = None,
    *,
    annotations=None,
    length: int | None = None,
    max_se: float = 5.0,
) -> Figure:
    """Fitted residue protection factors of ``samples`` (default: all, in table order), a panel
    per K and rank: see the module docstring. Residues whose standard error is above ``max_se``
    (log10 units) or unknown are not drawn (the panel says how many); the table keeps them."""
    samples = list(samples) if samples is not None else list(dict.fromkeys(table.sample.tolist()))
    panels = [(k, r, name) for k, r, name in _PANELS
              if table.where(sample=samples, K=k, rank=r).any()]
    if not panels:
        raise ValueError("no residue protection factors of those samples")
    fig = Figure(figsize=(7.5, 2.8 * len(panels) + 0.3), layout="constrained")
    axes = fig.subplots(len(panels), 1, sharex=True, squeeze=False)[:, 0]
    for ax, (k, r, name) in zip(axes, panels, strict=True):
        hidden, values, entries, kinds = 0, [], [], set()
        for n, s in enumerate(samples):
            h, v, drawn = _residue_panel(ax, table, s, k, r, _style(n), max_se)
            hidden += h
            values.append(v)
            kinds |= drawn
            if drawn:
                entries.append((s, _style(n)))
        _finish_panel(ax, np.concatenate(values), hidden, max_se)
        _marks(ax, annotations, label_sites=False)
        ax.set_ylabel("log10 protection factor")
        ax.grid(color="#e5e5e5", linewidth=0.8)
        ax.set_title(name, fontsize=9)
        _legend(ax, entries, kinds, ncol=min(4, len(entries) + 2))
    axes[-1].set_xlim(0, length if length is not None else int(table.residue.max()) + 1)
    axes[-1].set_xlabel("Residue")
    return fig


def residue_protection_comparison_figure(
    table,
    comparison,
    sample: str,
    *,
    style_index: int = 1,
    annotations=None,
    length: int | None = None,
    max_se: float = 5.0,
) -> Figure:
    """``sample`` against the comparison's reference, residue by residue, a panel per K and
    rank: see the module docstring; ``max_se`` as for :func:`residue_protection_figure`."""
    ref = comparison.reference
    panels = [(k, r, name) for k, r, name in _PANELS
              if comparison.where(sample=sample, K=k, rank=r).any()]
    if not panels:
        raise ValueError(f"no residue protection comparison of {sample!r}")
    fig = Figure(figsize=(7.5, 2.8 * len(panels) + 0.3), layout="constrained")
    axes = fig.subplots(len(panels), 1, sharex=True, squeeze=False)[:, 0]
    for ax, (k, r, name) in zip(axes, panels, strict=True):
        h0, v0, d0 = _residue_panel(ax, table, ref, k, r, _style(0), max_se)
        h1, v1, d1 = _residue_panel(ax, table, sample, k, r, _style(style_index), max_se)
        _finish_panel(ax, np.concatenate([v0, v1]), h0 + h1, max_se)
        sig = comparison.where(sample=sample, K=k, rank=r) & comparison.significant
        for res, d in zip(comparison.residue[sig], comparison.delta[sig], strict=True):
            ax.add_patch(Rectangle((res - 0.5, 0.96), 1.0, 0.04,
                                   transform=ax.get_xaxis_transform(),
                                   color=INCREASE if d > 0 else DECREASE, linewidth=0,
                                   gid=f"significant:{res}"))
        _marks(ax, annotations, label_sites=False)
        ax.set_ylabel("log10 protection factor")
        ax.grid(color="#e5e5e5", linewidth=0.8)
        ax.set_title(f"{sample} against {ref}, {name}", fontsize=9)
        entries = [(name_, st) for name_, st, d in ((ref, _style(0), d0),
                                                     (sample, _style(style_index), d1)) if d]
        _legend(ax, entries, d0 | d1, ncol=4)
    axes[-1].set_xlim(0, length if length is not None else int(table.residue.max()) + 1)
    axes[-1].set_xlabel("Residue")
    return fig
