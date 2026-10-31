# SPDX-License-Identifier: GPL-3.0-or-later
"""The residue significance maps (owner, 2026-10-31; HISTORY 203-204): one row per compared
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
uncovered ones white. Functions return figures and never show them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator

from ..analysis import ChoiceComparison, ResidueComparison
from ..io.times import format_time

STEPS = (2.0, 4.0)  # the size steps, in multiples of the threshold: 1-2, 2-4, above 4
# not significant; less deuterium (light to dark); more deuterium (light to dark)
SIGNIFICANCE_COLOURS = ("#d4d4d4", "#86b9dc", "#2f7bbf", "#08306b", "#fb9d7d", "#de3a27",
                        "#7f0010")
NOT_TESTED_HATCH = "xxxx"
SAMPLE_ONLY_HATCH = "////"  # two populations in the sample only
REFERENCE_ONLY_HATCH = "\\\\\\\\"  # two populations in the reference only
_EDGE = "0.45"
_SIZES = ("1-2x", "2-4x", "over 4x")


def significance_classes(comparison: ResidueComparison, *, more_d_positive: bool = True
                         ) -> np.ndarray:
    """Each row's class: 0 not significant, 1-3 less deuterium than the reference and 4-6 more
    (by the shared difference over the threshold: below 2, 2 to 4, above 4), -1 not tested
    (no peptide measured in both samples covers the residue: ``delta_shared`` NaN). Without
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
    return out


def _when(t: float, time_scale: str) -> str:
    return f"point {format_time(t, time_scale)}" if time_scale == "order" else format_time(t)


def _cells(table, samples, n, code_of, time_scale):
    """One grid row per sample and time of ``table``: (labels, first-of-sample flags, codes
    (NaN: white), and per hatch the residues to hatch), ``code_of(i)`` giving row ``i``'s code
    (a number) or hatch (a string), or None for white."""
    labels, first, rows, hatches = [], [], [], []
    for s in samples:
        here = table.sample == s
        for j, t in enumerate(sorted({float(t) for t in table.time[here]})):
            line = np.full(n, np.nan)
            marks: dict[str, np.ndarray] = {}
            for i in np.flatnonzero(here & (table.time == t)):
                r = int(table.residue[i])
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
    return labels, first, rows, hatches


def _draw(sections, n: int, legend: list, title: str) -> Figure:
    """The map: ``sections`` is a list of (heading or '', labels, first flags, rows, hatches),
    stacked, each heading on an empty row of its own above its section."""
    labels, first, rows, hatches, headings = [], [], [], [], []
    for heading, lab, fir, row, hat in sections:
        if not row:
            continue
        if heading:
            headings.append((len(rows), heading))
            labels.append("")
            first.append(False)
            rows.append(np.full(n, np.nan))
            hatches.append({})
        labels += lab
        first += fir
        rows += row
        hatches += hat
    if not rows:
        raise ValueError("nothing to draw")
    image = np.ma.masked_invalid(np.array(rows))
    cmap = ListedColormap(list(SIGNIFICANCE_COLOURS)).with_extremes(bad="white")
    fig = Figure(figsize=(max(8.0, min(16.0, 0.06 * n + 3.0)), 0.2 * len(rows) + 1.9))
    ax = fig.add_subplot()
    ax.imshow(image, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(SIGNIFICANCE_COLOURS) - 0.5,
              interpolation="nearest", extent=(0.5, n + 0.5, len(rows) - 0.5, -0.5))
    for y, marks in enumerate(hatches):  # runs of hatched residues
        for hatch, where in marks.items():
            r = 0
            while r < n:
                if where[r]:
                    start = r
                    while r < n and where[r]:
                        r += 1
                    ax.add_patch(Rectangle((start + 0.5, y - 0.5), r - start, 1.0,
                                           facecolor="white", edgecolor=_EDGE, hatch=hatch,
                                           lw=0.0, gid=f"{hatch}:{labels[y]}:{start + 1}-{r}"))
                r += 1
    ticks = [y for y, name in enumerate(labels) if name]
    ax.set_yticks(ticks)
    ax.set_yticklabels([labels[y] for y in ticks], fontsize=7)
    ax.tick_params(axis="y", length=2)
    for j in range(1, len(rows)):
        if first[j]:
            ax.axhline(j - 0.5, color="white", lw=3)
    for y, heading in headings:
        ax.text(1.0, y, heading, fontsize=7, fontweight="bold", va="center", ha="left")
    ax.set_xlim(0.5, n + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel("residue")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.legend(handles=legend, frameon=False, loc="lower center", ncol=3, fontsize=7)
    fig.suptitle(title, fontsize=8, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    return fig


def _graded_legend(more: str, less: str) -> list:
    return ([Patch(color=SIGNIFICANCE_COLOURS[4 + j], label=f"{more}, {_SIZES[j]}")
             for j in range(3)]
            + [Patch(color=SIGNIFICANCE_COLOURS[1 + j], label=f"{less}, {_SIZES[j]}")
               for j in range(3)])


def _hatch_patch(hatch: str, label: str) -> Patch:
    return Patch(facecolor="white", edgecolor=_EDGE, hatch=hatch, label=label)


def _graded(comparison: ResidueComparison, samples, n, time_scale, more_d_positive=True):
    classes = significance_classes(comparison, more_d_positive=more_d_positive)
    return _cells(comparison, samples, n,
                  lambda i: NOT_TESTED_HATCH if classes[i] < 0 else int(classes[i]), time_scale)


def _samples(table, samples):
    return list(dict.fromkeys(str(s) for s in table.sample)) if samples is None else list(samples)


def plot_residue_significance(
    comparison: ResidueComparison,
    samples: Sequence[str] | None = None,
    *,
    length: int | None = None,
    time_scale: str = "seconds",
) -> Figure:
    """The K = 1 significance map of ``comparison`` (a residue comparison, usually from
    :func:`hxex.analysis.judged_on_shared`) for ``samples`` (default: all, in table order): one
    row per sample and time, one cell per residue (see the module docstring). White cells: a
    residue not compared (uncovered in either sample, or without an amide)."""
    if not len(comparison):
        raise ValueError("the residue comparison is empty")
    samples = _samples(comparison, samples)
    n = int(length if length is not None else comparison.residue.max())
    labels, first, rows, hatches = _graded(comparison, samples, n, time_scale)
    if not rows:
        raise ValueError(f"no comparison of {', '.join(samples)}")
    legend = _graded_legend("more D", "less D") + [
        Patch(color=SIGNIFICANCE_COLOURS[0], label="not significant"),
        _hatch_patch(NOT_TESTED_HATCH, "not tested (no shared peptide)"),
        Patch(facecolor="white", edgecolor="0.6", label="not compared")]
    reference = comparison.reference or "the reference"
    quantity = "fD" if comparison.quantity == "fD" else "low-D population"
    return _draw([("", labels, first, rows, hatches)], n, legend,
                 f"Residues different from {reference} ({quantity}): significant on the "
                 "peptides both samples measured; colour = direction and size of that "
                 "difference in multiples of the threshold")


def plot_residue_significance_k2(
    choices: ChoiceComparison,
    population: ResidueComparison | None = None,
    low: ResidueComparison | None = None,
    high: ResidueComparison | None = None,
    samples: Sequence[str] | None = None,
    *,
    length: int | None = None,
    time_scale: str = "seconds",
) -> Figure:
    """The K = 2 significance map: section 1 from ``choices``
    (:func:`hxex.analysis.compare_choices`): where only the sample or only the reference keeps
    two populations (textures); with ``population`` (:func:`hxex.analysis.compare_populations`,
    judged on shared peptides; given only with a bootstrap at K = 2), where both keep two, the
    significant differences in the low-D population's share (blue: more of the low-D
    population, so less deuterium; red: less of it), else grey. Sections 2 and 3 (with ``low``
    and ``high``, the K = 2 rank 1 and rank 2 residue comparisons judged on shared peptides):
    the significant differences in the low-D and high-D positions, as at K = 1, drawn where
    both samples keep two populations. White: one population in both, or not compared."""
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

    def counts(i):
        if two[i] and not two_ref[i]:
            return SAMPLE_ONLY_HATCH
        if two_ref[i] and not two[i]:
            return REFERENCE_ONLY_HATCH
        if not two[i]:
            return None
        c = pop_class.get((choices.sample[i], choices.time[i], choices.residue[i]), 0)
        return NOT_TESTED_HATCH if c < 0 else int(c)

    heading = ("populations: red = a smaller low-D share than the reference (so more D), "
               "blue = a larger one" if population is not None else "populations")
    sections = [(heading, *_cells(choices, samples, n, counts, time_scale))]
    for name, comparison in (("low-D position", low), ("high-D position", high)):
        if comparison is not None and len(comparison):
            both = (np.ones(len(comparison), dtype=bool) if comparison.two is None else
                    np.asarray(comparison.two_reference, dtype=bool)
                    & np.asarray(comparison.two, dtype=bool))
            classes = significance_classes(comparison)
            sections.append((name, *_cells(
                comparison, samples, n,
                lambda i, c=classes, b=both: (None if not b[i] else NOT_TESTED_HATCH
                                              if c[i] < 0 else int(c[i])), time_scale)))
    legend = [_hatch_patch(SAMPLE_ONLY_HATCH, "two populations in the sample only"),
              _hatch_patch(REFERENCE_ONLY_HATCH, "two populations in the reference only"),
              Patch(facecolor="white", edgecolor="0.6", label="one population in both")]
    tested = population is not None
    if tested:
        legend += _graded_legend("more D than the reference", "less D than the reference")
        legend += [Patch(color=SIGNIFICANCE_COLOURS[0], label="two in both, not significant"),
                   _hatch_patch(NOT_TESTED_HATCH, "not tested (no shared peptide)")]
    else:
        legend.append(Patch(color=SIGNIFICANCE_COLOURS[0], label="two populations in both"))
    reference = choices.reference or "the reference"
    title = (f"Two populations against {reference}: "
             + ("significant differences on the peptides both samples measured; colour = "
                "direction and size in multiples of the threshold" if tested else
                "where the number differs (differences in share and position need a bootstrap "
                "at K = 2: [run.bootstrap] ks = [1, 2])"))
    return _draw(sections, n, legend, title)
