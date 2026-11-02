# SPDX-License-Identifier: GPL-3.0-or-later
"""The residue significance maps (owner, 2026-10-31; HISTORY 203, 204, 206): one row per compared
sample and time, one cell per residue, laid out as the population tier map
(:mod:`hxex.report.tiers`).

K = 1 (:func:`plot_residue_significance`): whether the residue's difference from the reference is
significant (judged from the peptides both samples measured,
:func:`hxex.analysis.judged_on_shared`) and, where it is, the direction and size of that shared
difference in multiples of the threshold (``delta_crit``). Size, not z, grades the cells: the
residue errors come from the fits, shrunk by combining overlapping peptides, so z would paint the
deepest colours where they are most understated (HISTORY 202).

K = 2 (:func:`plot_residue_significance_k2`): where the number of populations differs (two in the
sample only, two in the reference only: textures; needs no bootstrap), and, with a bootstrap at
K = 2, the significant differences in the population share and in the two populations' positions,
coloured as at K = 1. Without one most residues have no K = 2 error and the few differences
called do not reproduce the bootstrapped ones (HISTORY 204), so they are not tested (owner,
2026-10-31).

Colours: a diverging red (more deuterium than the reference) and blue (less), three steps each,
and a grey for "not significant", every pair apart under simulated colour-vision deficiency
(OKLab difference at least 10); residues not tested (no shared peptide) are cross-hatched,
residues the residue test calls but no covering peptide carries
(:func:`hxex.analysis.gated_by_peptides`; not significant, HISTORY 206) are grey with dots,
uncovered ones white. Functions return figures and never show them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.ticker import MultipleLocator

from ..analysis import ChoiceComparison, RegionComparison, ResidueComparison
from ..io.times import format_time

STEPS = (2.0, 4.0)  # the size steps, in multiples of the threshold: 1-2, 2-4, above 4
# not significant; less deuterium (light to dark); more deuterium (light to dark)
SIGNIFICANCE_COLOURS = ("#d4d4d4", "#86b9dc", "#2f7bbf", "#08306b", "#fb9d7d", "#de3a27",
                        "#7f0010")
NOT_TESTED_HATCH = "xxxx"
# the residue (or region) test called it, but no significant peptide confirms it: dots in the
# colour of its direction on grey (not significant)
UNCONFIRMED_MORE = "more...."
UNCONFIRMED_LESS = "less...."
UNCONFIRMED_LABEL = "significance not confirmed at peptide level"
BLOCK_OUTLINE = "0.15"  # the residue blocks the peptides resolve (HISTORY 207)
BLOCK_LINE_WIDTH = 0.6
SAMPLE_GAP = 0.35  # the empty space between samples, in rows
DIFFERS = "differs"  # the samples' own peptides decide one or two differently (a triangle)
DIFFERS_COLOUR = "black"  # the triangle's edge
DIFFERS_FILL = "#e8b928"  # the tier map's "undecided" yellow (hxex.report.tiers)
TRIANGLE_WIDTH = 1.8  # residues
TRIANGLE_HEIGHT = 0.55  # of a row
SAMPLE_ONLY_HATCH = "////"  # two populations in the sample only
REFERENCE_ONLY_HATCH = "\\\\\\\\"  # two populations in the reference only
_EDGE = "0.45"
_SIZES = ("1-2x", "2-4x", "over 4x")


def significance_classes(comparison: ResidueComparison, *, more_d_positive: bool = True
                         ) -> np.ndarray:
    """Each row's class: 0 not significant, 1-3 less deuterium than the reference and 4-6 more
    (by the shared difference over the threshold: below 2, 2 to 4, above 4), -1 not tested
    (no peptide measured in both samples covers the residue: ``delta_shared`` NaN), -2 and -3
    the residue test only (``residue_test_only``: no significant peptide confirms it), more and
    less deuterium than the reference. Without
    ``delta_shared`` (a comparison not judged on shared peptides) the row's own ``delta``.
    ``more_d_positive``: a positive difference means more deuterium (an fD); False for the
    low-D population's share, where more of it means less deuterium."""
    shared = comparison.delta_shared if comparison.delta_shared is not None else comparison.delta
    shared = np.asarray(shared, dtype=float)
    threshold = np.asarray(comparison.threshold, dtype=float)
    out = np.zeros(len(comparison), dtype=np.int64)
    out[np.isnan(shared)] = -1
    sig = np.asarray(comparison.significant, dtype=bool) & np.isfinite(shared)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.abs(shared) / np.where(threshold > 0, threshold, np.nan)
    step = 1 + (ratio >= STEPS[0]).astype(int) + (ratio >= STEPS[1]).astype(int)
    more = (shared > 0) if more_d_positive else (shared < 0)
    out[sig] = np.where(more[sig], 3 + step[sig], step[sig])
    if comparison.residue_test_only is not None:
        only = np.asarray(comparison.residue_test_only, dtype=bool) & np.isfinite(shared)
        out[only] = np.where(more[only], -2, -3)
    return out


def _when(t: float, time_scale: str) -> str:
    return f"point {format_time(t, time_scale)}" if time_scale == "order" else format_time(t)


def _blocks_of(table) -> dict:
    """``(sample, time)`` -> the ``(first, last)`` residue blocks of a comparison (its
    ``block_start``, ``block_end``; empty without them)."""
    out: dict[tuple, set] = {}
    starts = getattr(table, "block_start", None)
    if starts is None:
        return out
    for s, t, b0, b1 in zip(table.sample, table.time, starts, table.block_end, strict=True):
        if np.isfinite(b0) and np.isfinite(b1):
            out.setdefault((str(s), float(t)), set()).add((int(b0), int(b1)))
    return out


def _cells(table, samples, n, code_of, time_scale, blocks=None, mark_of=None):
    """One grid row per sample and time of ``table``: (labels, first-of-sample flags, codes
    (NaN: white), per hatch the residues to hatch, and the residue blocks to outline),
    ``code_of(i)`` giving row ``i``'s code (a number) or hatch (a string), or None for white;
    ``blocks`` (:func:`_blocks_of`) default the table's own; ``mark_of(i)`` True marks row
    ``i``'s residue with the :data:`DIFFERS` stroke."""
    blocks = _blocks_of(table) if blocks is None else blocks
    labels, first, rows, hatches, outlines = [], [], [], [], []
    for s in samples:
        here = table.sample == s
        for j, t in enumerate(sorted({float(t) for t in table.time[here]})):
            line = np.full(n, np.nan)
            marks: dict[str, np.ndarray] = {}
            for i in np.flatnonzero(here & (table.time == t)):
                r = int(table.residue[i])
                if mark_of is not None and 1 <= r <= n and mark_of(i):
                    marks.setdefault(DIFFERS, np.zeros(n, dtype=bool))[r - 1] = True
                code = code_of(i)
                if not 1 <= r <= n or code is None:
                    continue
                if isinstance(code, str):
                    marks.setdefault(code, np.zeros(n, dtype=bool))[r - 1] = True
                else:
                    line[r - 1] = code
            labels.append(f"{s}  {_when(t, time_scale)}")
            first.append(j == 0)
            rows.append(line)
            hatches.append(marks)
            drawn = np.isfinite(line)
            for where in marks.values():  # a stroke over a white cell is boxed too
                drawn |= where
            outlines.append(sorted(b for b in blocks.get((str(s), t), ())
                                   if drawn[max(b[0], 1) - 1:min(b[1], n)].any()))
    return labels, first, rows, hatches, outlines


def _split(first: int, last: int, blocks) -> list[tuple[int, int]]:
    """Residues ``first`` to ``last`` cut at the edges of ``blocks`` (``(first, last)`` pairs):
    the pieces, each inside one block (or outside every block)."""
    cuts = sorted({b0 for b0, _ in blocks if first < b0 <= last}
                  | {b1 + 1 for _, b1 in blocks if first <= b1 < last})
    starts = [first, *cuts]
    ends = [c - 1 for c in cuts] + [last]
    return list(zip(starts, ends, strict=True))


LEGEND_COLUMNS = 3
LEGEND_ROW = 0.19  # inches per row of the legend (7 pt text and its spacing)


def _draw(sections, n: int, legend: list, title: str, block_outline: str | None = BLOCK_OUTLINE,
          columns: Sequence[str] | None = None) -> Figure:
    """The map: ``sections`` is a list of (heading or '', labels, first flags, rows, hatches,
    block outlines), stacked, each heading on a row of its own above its section and an empty
    gap between samples; each row's residue blocks outlined in ``block_outline`` (None: not
    drawn)."""
    labels, rows, hatches, outlines, headings, heights = [], [], [], [], [], []

    def add(label, row, hatch, outline, height):
        labels.append(label)
        rows.append(row)
        hatches.append(hatch)
        outlines.append(outline)
        heights.append(height)

    empty = np.full(n, np.nan)
    for heading, lab, fir, row, hat, out in sections:
        if not row:
            continue
        if heading:
            headings.append((len(rows), heading))
            add("", empty, {}, [], 1.0)
        for k in range(len(row)):
            if fir[k] and k:  # a true gap between samples, so every block is outlined whole
                add("", empty, {}, [], SAMPLE_GAP)
            add(lab[k], row[k], hat[k], out[k], 1.0)
    if not rows:
        raise ValueError("nothing to draw")
    edges = np.concatenate([[0.0], np.cumsum(heights)])  # row k spans edges[k] to edges[k + 1]
    image = np.ma.masked_invalid(np.array(rows))
    cmap = ListedColormap(list(SIGNIFICANCE_COLOURS)).with_extremes(bad="white")
    width = (max(8.0, min(16.0, 0.06 * n + 3.0)) if columns is None
             else max(6.0, min(16.0, 0.32 * n + 3.0)))
    # the legend has a band of its own, sized in inches by its rows, so that it never meets
    # the axis label and the tick labels however few rows the map has
    legend_rows = math.ceil(len(legend) / LEGEND_COLUMNS)
    legend_height = LEGEND_ROW * legend_rows + 0.15
    height = 0.2 * edges[-1] + (1.3 if columns is None else 2.0) + legend_height
    fig = Figure(figsize=(width, height))
    ax = fig.add_subplot()
    ax.pcolormesh(np.arange(n + 1) + 0.5, edges, image, cmap=cmap, vmin=-0.5,
                  vmax=len(SIGNIFICANCE_COLOURS) - 0.5, shading="flat")
    for y, marks in enumerate(hatches):  # runs of hatched residues
        for hatch, where in marks.items():
            r = 0
            while r < n:
                if where[r]:
                    start = r
                    while r < n and where[r]:
                        r += 1
                    if hatch == DIFFERS:  # a corner triangle in each block of the run
                        for a, b in _split(start + 1, r, outlines[y]):
                            right, top = b + 0.5, edges[y]
                            width = min(TRIANGLE_WIDTH, b - a + 1.0)
                            ax.add_patch(Polygon(
                                [(right, top), (right - width, top),
                                 (right, top + TRIANGLE_HEIGHT * heights[y])],
                                closed=True, facecolor=DIFFERS_FILL, edgecolor=DIFFERS_COLOUR,
                                lw=0.6, zorder=4, gid=f"{DIFFERS}:{labels[y]}:{a}-{b}"))
                        r += 1
                        continue
                    pattern, face, edge = _hatch_style(hatch)
                    ax.add_patch(Rectangle((start + 0.5, edges[y]), r - start, heights[y],
                                           facecolor=face, edgecolor=edge, hatch=pattern,
                                           lw=0.0, gid=f"{hatch}:{labels[y]}:{start + 1}-{r}"))
                r += 1
    if block_outline is not None:  # the resolution: the blocks the peptides can tell apart
        for y, blocks in enumerate(outlines):
            for b0, b1 in blocks:
                ax.add_patch(Rectangle((b0 - 0.5, edges[y]), b1 - b0 + 1, heights[y],
                                       fill=False, edgecolor=block_outline,
                                       lw=BLOCK_LINE_WIDTH, zorder=3,
                                       clip_on=False,  # the edges on the plot's border too
                                       gid=f"block:{labels[y]}:{b0}-{b1}"))
    centres = 0.5 * (edges[:-1] + edges[1:])
    ticks = [y for y, name in enumerate(labels) if name]
    ax.set_yticks(centres[ticks])
    ax.set_yticklabels([labels[y] for y in ticks], fontsize=7)
    ax.tick_params(axis="y", length=2)
    for y, heading in headings:
        ax.text(1.0, centres[y], heading, fontsize=7, fontweight="bold", va="center", ha="left")
    ax.set_ylim(edges[-1], 0.0)
    ax.set_xlim(0.5, n + 0.5)
    if columns is None:  # residues
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.set_xlabel("residue")
    else:  # one named column each (regions)
        ax.set_xticks(np.arange(1, n + 1))
        ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("region")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.legend(handles=legend, frameon=False, loc="lower center", ncol=LEGEND_COLUMNS,
               fontsize=7, bbox_to_anchor=(0.5, 0.05 / height),
               handler_map={_TriangleHandle: _TriangleHandler()})
    fig.suptitle(title, fontsize=8, x=0.01, ha="left")
    fig.tight_layout(rect=(0, legend_height / height, 1, 1 - 0.25 / height))
    return fig


def _graded_legend(more: str, less: str) -> list:
    return ([Patch(color=SIGNIFICANCE_COLOURS[4 + j], label=f"{more}, {_SIZES[j]}")
             for j in range(3)]
            + [Patch(color=SIGNIFICANCE_COLOURS[1 + j], label=f"{less}, {_SIZES[j]}")
               for j in range(3)])


def _hatch_style(key: str) -> tuple[str, str, str]:
    """A mark's (hatch pattern, face, edge): the unconfirmed differences are dots in the colour
    of their direction on grey; the others grey hatching on white."""
    if key == UNCONFIRMED_MORE:
        return "....", SIGNIFICANCE_COLOURS[0], SIGNIFICANCE_COLOURS[5]
    if key == UNCONFIRMED_LESS:
        return "....", SIGNIFICANCE_COLOURS[0], SIGNIFICANCE_COLOURS[2]
    return key, "white", _EDGE


def _hatch_patch(hatch: str, label: str) -> Patch:
    pattern, face, edge = _hatch_style(hatch)
    return Patch(facecolor=face, edgecolor=edge, hatch=pattern, label=label)


def _unconfirmed_legend(more: str = "more D", less: str = "less D") -> list:
    return [_hatch_patch(UNCONFIRMED_MORE, f"{more}: {UNCONFIRMED_LABEL}"),
            _hatch_patch(UNCONFIRMED_LESS, f"{less}: {UNCONFIRMED_LABEL}")]


class _TriangleHandle(Patch):
    """A legend entry drawn as the corner triangle."""

    def __init__(self, label: str):
        super().__init__(facecolor=DIFFERS_FILL, edgecolor=DIFFERS_COLOUR, lw=0.6, label=label)


class _TriangleHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        w, h = handlebox.width, handlebox.height
        box = Rectangle((x0, y0), w, h, facecolor=SIGNIFICANCE_COLOURS[0], edgecolor="none",
                        transform=handlebox.get_transform())
        tri = Polygon([(x0 + w, y0 + h), (x0 + w - 0.45 * w, y0 + h), (x0 + w, y0 + 0.35 * h)],
                      closed=True, facecolor=DIFFERS_FILL, edgecolor=DIFFERS_COLOUR, lw=0.6,
                      transform=handlebox.get_transform())
        handlebox.add_artist(box)
        handlebox.add_artist(tri)
        return tri


def _block_patch(colour: str) -> Patch:
    return Patch(fill=False, edgecolor=colour, lw=BLOCK_LINE_WIDTH,
                 label="a block the peptides cannot divide (resolution)")


def _code(c: int):
    """A class (:func:`significance_classes`) as a cell code or hatch."""
    return {-1: NOT_TESTED_HATCH, -2: UNCONFIRMED_MORE, -3: UNCONFIRMED_LESS}.get(c, int(c))


def _graded(comparison: ResidueComparison, samples, n, time_scale, more_d_positive=True):
    classes = significance_classes(comparison, more_d_positive=more_d_positive)
    return _cells(comparison, samples, n, lambda i: _code(classes[i]), time_scale)


def _samples(table, samples):
    return list(dict.fromkeys(str(s) for s in table.sample)) if samples is None else list(samples)


def plot_residue_significance(
    comparison: ResidueComparison,
    samples: Sequence[str] | None = None,
    *,
    length: int | None = None,
    time_scale: str = "seconds",
    block_outline: str | None = BLOCK_OUTLINE,
) -> Figure:
    """The K = 1 significance map of ``comparison`` (a residue comparison, usually from
    :func:`hxex.analysis.judged_on_shared`) for ``samples`` (default: all, in table order): one
    row per sample and time, one cell per residue (see the module docstring). White cells: a
    residue not compared (uncovered in either sample, or without an amide). Each row's residue
    blocks (``block_start``, ``block_end``: the stretches the shared peptides cannot divide, the
    resolution) are outlined in ``block_outline`` (a colour; None: not drawn)."""
    if not len(comparison):
        raise ValueError("the residue comparison is empty")
    samples = _samples(comparison, samples)
    n = int(length if length is not None else comparison.residue.max())
    section = _graded(comparison, samples, n, time_scale)
    if not section[2]:
        raise ValueError(f"no comparison of {', '.join(samples)}")
    legend = _graded_legend("more D", "less D") + [
        Patch(color=SIGNIFICANCE_COLOURS[0], label="not significant"),
        *_unconfirmed_legend(),
        _hatch_patch(NOT_TESTED_HATCH, "not tested (no shared peptide)"),
        Patch(facecolor="white", edgecolor="0.6", label="not compared")]
    if block_outline is not None:
        legend.append(_block_patch(block_outline))
    reference = comparison.reference or "the reference"
    quantity = "fD" if comparison.quantity == "fD" else "low-D population"
    return _draw([("", *section)], n, legend,
                 f"Residues different from {reference} ({quantity}): significant on the "
                 "peptides both samples measured; colour = direction and size of that "
                 "difference in multiples of the threshold", block_outline)


def plot_residue_significance_k2(
    choices: ChoiceComparison,
    population: ResidueComparison | None = None,
    low: ResidueComparison | None = None,
    high: ResidueComparison | None = None,
    samples: Sequence[str] | None = None,
    *,
    length: int | None = None,
    time_scale: str = "seconds",
    blocks: ResidueComparison | None = None,
    block_outline: str | None = BLOCK_OUTLINE,
    own_choices: ChoiceComparison | None = None,
) -> Figure:
    """The K = 2 significance map: section 1 from ``choices`` (:func:`hxex.analysis.compare_choices`
    of the choices made on the peptides both samples measured, so that they agree with the test
    and the blocks; ``own_choices``, the same from each sample's own peptides, marks with a
    stroke the residues they decide differently): where only the sample or only the reference keeps
    two populations (textures); with ``population`` (:func:`hxex.analysis.compare_populations`,
    judged on shared peptides; given only with a bootstrap at K = 2), where both keep two, the
    significant differences in the low-D population's share (blue: more of the low-D
    population, so less deuterium; red: less of it), else grey. Sections 2 and 3 (with ``low``
    and ``high``, the K = 2 rank 1 and rank 2 residue comparisons judged on shared peptides):
    the significant differences in the low-D and high-D positions, as at K = 1, drawn where
    both samples keep two populations (by ``choices``). White: one population in both, or not
    compared. Residue
    blocks are outlined as in :func:`plot_residue_significance`: the position sections' own, the
    populations section's from ``blocks`` (the K = 1 comparison; none without it)."""
    if not len(choices):
        raise ValueError("the choice comparison is empty")
    samples = _samples(choices, samples)
    n = int(length if length is not None else choices.residue.max())
    pop_class = {}
    if population is not None:
        classes = significance_classes(population, more_d_positive=False)
        keys = zip(population.sample, population.time, population.residue, strict=True)
        pop_class = dict(zip(keys, classes, strict=True))
    two_ref = np.asarray(choices.two_reference, dtype=bool)
    two = np.asarray(choices.two, dtype=bool)
    keys = list(zip(choices.sample, choices.time, choices.residue, strict=True))
    shared_both = {k: bool(a and b) for k, a, b in zip(keys, two_ref, two, strict=True)}
    own = ({} if own_choices is None else
           {k: (bool(a), bool(b)) for k, a, b in zip(
               zip(own_choices.sample, own_choices.time, own_choices.residue, strict=True),
               own_choices.two_reference, own_choices.two, strict=True)})

    def differs(i):
        mine = own.get(keys[i])
        return mine is not None and mine != (bool(two_ref[i]), bool(two[i]))

    def counts(i):
        if two[i] and not two_ref[i]:
            return SAMPLE_ONLY_HATCH
        if two_ref[i] and not two[i]:
            return REFERENCE_ONLY_HATCH
        if not two[i]:
            return None
        return _code(pop_class.get((choices.sample[i], choices.time[i], choices.residue[i]), 0))

    heading = ("populations: red = a smaller low-D share than the reference (so more D), "
               "blue = a larger one" if population is not None else "populations")
    sections = [(heading, *_cells(choices, samples, n, counts, time_scale,
                                  blocks=_blocks_of(blocks) if blocks is not None else {},
                                  mark_of=differs if own else None))]
    for name, comparison in (("low-D position", low), ("high-D position", high)):
        if comparison is not None and len(comparison):
            both = np.array([shared_both.get(k, False) for k in zip(
                comparison.sample, comparison.time, comparison.residue, strict=True)], dtype=bool)
            classes = significance_classes(comparison)
            sections.append((name, *_cells(
                comparison, samples, n,
                lambda i, c=classes, b=both: None if not b[i] else _code(c[i]), time_scale)))
    legend = [_hatch_patch(SAMPLE_ONLY_HATCH, "two populations in the sample only"),
              _hatch_patch(REFERENCE_ONLY_HATCH, "two populations in the reference only"),
              Patch(facecolor="white", edgecolor="0.6", label="one population in both")]
    tested = population is not None
    if tested:
        legend += _graded_legend("more D than the reference", "less D than the reference")
        legend += [Patch(color=SIGNIFICANCE_COLOURS[0], label="two in both, not significant"),
                   *_unconfirmed_legend("more D than the reference", "less D than the reference"),
                   _hatch_patch(NOT_TESTED_HATCH, "not tested (no shared peptide)")]
    else:
        legend.append(Patch(color=SIGNIFICANCE_COLOURS[0], label="two populations in both"))
    if own:
        legend.append(_TriangleHandle(label="one or two decided differently on each sample's "
                                            "own peptides"))
    if block_outline is not None:
        legend.append(_block_patch(block_outline))
    reference = choices.reference or "the reference"
    title = (f"Two populations against {reference}: "
             + ("significant differences on the peptides both samples measured; colour = "
                "direction and size in multiples of the threshold" if tested else
                "where the number differs (differences in share and position need a bootstrap "
                "at K = 2: [run.bootstrap] ks = [1, 2])"))
    return _draw(sections, n, legend, title, block_outline)


def plot_region_significance(
    comparison: RegionComparison,
    samples: Sequence[str] | None = None,
    *,
    time_scale: str = "seconds",
) -> Figure:
    """The region significance map of ``comparison`` (:func:`hxex.analysis.compare_regions`):
    one row per sample and time, one column per region (in order of their start), coloured as
    the residue map (:func:`plot_residue_significance`) by the region's shared difference over
    the threshold where it is significant; grey not significant, grey with dots the region test
    only (no peptide inside the region carries the difference), cross-hatched not tested (no
    residue of the region compared)."""
    if not len(comparison):
        raise ValueError("the region comparison is empty")
    samples = _samples(comparison, samples)
    order = sorted({(int(a), int(b), str(r)) for a, b, r in zip(
        comparison.start, comparison.end, comparison.region, strict=True)})
    column = {r: k for k, (_, _, r) in enumerate(order)}
    n = len(order)
    shared = np.asarray(comparison.delta_shared, dtype=float)
    threshold = np.asarray(comparison.threshold, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.abs(shared) / np.where(threshold > 0, threshold, np.nan)
    step = 1 + (ratio >= STEPS[0]).astype(int) + (ratio >= STEPS[1]).astype(int)
    sig = np.asarray(comparison.significant, dtype=bool) & np.isfinite(shared)
    only = np.asarray(comparison.region_test_only, dtype=bool)
    labels, first, rows, hatches, outlines = [], [], [], [], []
    for s in samples:
        here = comparison.sample == s
        for j, t in enumerate(sorted({float(t) for t in comparison.time[here]})):
            line = np.full(n, np.nan)
            marks = {h: np.zeros(n, dtype=bool)
                     for h in (NOT_TESTED_HATCH, UNCONFIRMED_MORE, UNCONFIRMED_LESS)}
            for i in np.flatnonzero(here & (comparison.time == t)):
                k = column[str(comparison.region[i])]
                if not np.isfinite(shared[i]):
                    marks[NOT_TESTED_HATCH][k] = True
                elif only[i]:
                    marks[UNCONFIRMED_MORE if shared[i] > 0 else UNCONFIRMED_LESS][k] = True
                elif sig[i]:
                    line[k] = 3 + step[i] if shared[i] > 0 else step[i]
                else:
                    line[k] = 0
            labels.append(f"{s}  {_when(t, time_scale)}")
            first.append(j == 0)
            rows.append(line)
            hatches.append({h: m for h, m in marks.items() if m.any()})
            outlines.append([(k + 1, k + 1) for k in range(n)])  # every region boxed
    if not rows:
        raise ValueError(f"no comparison of {', '.join(samples)}")
    legend = _graded_legend("more D", "less D") + [
        Patch(color=SIGNIFICANCE_COLOURS[0], label="not significant"),
        *_unconfirmed_legend(),
        _hatch_patch(NOT_TESTED_HATCH, "not tested (no residue of the region compared)")]
    reference = comparison.reference or "the reference"
    names = [f"{r} ({a}-{b})" for a, b, r in order]
    return _draw([("", labels, first, rows, hatches, outlines)], n, legend,
                 f"Regions different from {reference} (fD): the mean of their residues on the "
                 "peptides both samples measured; colour = direction and size in multiples of "
                 "the threshold", columns=names)

