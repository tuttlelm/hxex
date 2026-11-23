# SPDX-License-Identifier: GPL-3.0-or-later
"""Where each figure goes in the window's Analysis tree (owner, 2026-11-22), without Qt:

- **Summary**: a group's samples together, no reference: the coverage maps, redundancy and
  resolution (each form's of a pool too), then the residue fDavg, the regions, the population
  tiers (a pool's: per class of its forms);
- **Single samples**: group, then sample: every figure of one sample, coverage, redundancy
  and resolution first, the summary figures drawn for it alone among them
  (:attr:`hxex.report.analysis.AnalysisOutputs.window` ``"single"``), its fragment figures
  last (the parent chosen in the lists above the figure; owner, 2026-11-23);
- **Comparisons**: against the group's reference: the significance maps of the whole group,
  then each sample's figures, among them its panel of the overlays, its rows of the
  significance map, and the population tiers of the reference and that sample
  (``window`` ``"compared"``; owner, 2026-11-23).

A pool's classes stand for groups and its forms (each sample's form, ``"<sample> <form>"``)
for samples, the forms of the minimum set taking a sample's place. Without
``[analysis.groups]`` there is no group level (one analysis, its samples directly). Kinds the
settings leave out or the results give nothing for are listed under **Not drawn**, once each.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..report.catalogue import PAGE, TIME, figure_choices, kind_named, kind_of
from .order import natural_key

SUMMARY, SINGLE, HEAD_TO_HEAD, COMPARISONS, NOT_DRAWN = (
    "Summary", "Single samples", "Head to head (sequences)", "Comparisons", "Not drawn")
TOPS = (SUMMARY, SINGLE, HEAD_TO_HEAD, COMPARISONS)


def comparisons_top(reference: str) -> str:
    """The part of the tree holding the comparisons against ``reference`` (owner, 2026-11-23:
    one part per reference ticked), ``Comparisons`` without one."""
    return f"{COMPARISONS} (vs. {reference})" if reference else COMPARISONS


def top_of(name: str) -> str:
    """The part ``name`` is one of (``Comparisons (vs. X)``: :data:`COMPARISONS`)."""
    return COMPARISONS if name.startswith(COMPARISONS) else name
LEVELS: tuple[str, ...] = ()  # rows between a sample and its kinds (none: owner, 2026-11-23)

TOP_WORDS = {
    SUMMARY: "a group's samples together, none of them a reference: the coverage maps (one "
             "bar per peptide, coloured by fD), redundancy and resolution, the residue fDavg, "
             "the regions, the population tiers",
    SINGLE: "every figure of one sample: its coverage, redundancy and resolution, residue "
            "fDavg, population tiers, regions, and its fragment ions",
    HEAD_TO_HEAD: "a pool's sequences compared with each other, range by range, in each sample "
                  "(no reference sample: the same whichever is ticked)",
    COMPARISONS: "each sample against the reference: the significance maps of the whole group, "
                 "then each sample's (Woods, butterfly, its significance maps, its panel of "
                 "the overlays, the population tiers of the reference and that sample)",
    NOT_DRAWN: "the kinds of figure the settings leave out, or these results give nothing for: "
               "select one to see why",
}

# where each kind goes, in the order each part lists them (owner, 2026-11-23)
_SUMMARY_KINDS = ("coverage", "pooled_coverage", "redundancy_resolution", "residues_all",
                  "regions", "population_tiers", "pooled_forms", "cross_group_regions")
_SINGLE_ORDER = ("coverage", "redundancy_resolution", "residues_all", "population_tiers",
                 "choice", "profile", "regions", "protection")
_FRAGMENT_KINDS = ("fragment_coverage", "fragment_traces", "ladder", "closure",
                   "residue_rates")
_GROUP_COMPARISONS = ("residue_significance", "region_significance", "region_composition",
                      "overlay_residues", "overlay_populations")
_SAMPLE_COMPARISONS = ("woods", "woods_K2", "butterfly", "residue_significance_K2",
                       "region_composition_K2", "protection_difference")
_TOP_COMPARISONS = ("head_to_head",)
# the figures of every sample the window leaves to each sample's panel (owner, 2026-11-23);
# a pool's (one per form) stay
_EACH_SAMPLE_ONLY = ("overlay_residues", "overlay_populations")
HOME = {**{k: SUMMARY for k in _SUMMARY_KINDS},
        **{k: COMPARISONS for k in (*_GROUP_COMPARISONS, *_SAMPLE_COMPARISONS)},
        **{k: HEAD_TO_HEAD for k in _TOP_COMPARISONS},
        "protection": SINGLE, "choice": SINGLE, "profile": SINGLE,
        **{k: SINGLE for k in _FRAGMENT_KINDS}}
# the order within each part; under a compared sample its own kinds, then its part of the
# group's figures (window "compared": its significance rows, overlay panels, tiers)
_ORDERS = {SUMMARY: _SUMMARY_KINDS, SINGLE: (*_SINGLE_ORDER, *_FRAGMENT_KINDS),
           HEAD_TO_HEAD: _TOP_COMPARISONS,
           COMPARISONS: ("woods", "woods_K2", "butterfly",
                         "residue_significance", "residue_significance_K2",
                         "region_significance", "region_composition", "region_composition_K2",
                         "overlay_residues", "overlay_populations", "population_tiers",
                         "protection_difference")}


def _order(top: str, kind: str) -> int:
    kinds = _ORDERS.get(top_of(top), ())
    return kinds.index(kind) if kind in kinds else 999


@dataclass
class Entry:
    """A kind of figure at one place of the tree: ``path`` the names of the rows above it
    (the top first), ``figures`` its figures there (stem, drawing function), in order."""

    path: tuple[str, ...]
    kind: str
    figures: list[tuple[str, Callable]] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return (*self.path, self.kind)


@dataclass
class Placed:
    """One analysis to place: its group (None without ``[analysis.groups]``), the outputs the
    figures of its levels come from, and those its comparisons come from (None: not
    computed)."""

    group: str | None
    levels: object | None  # hxex.report.analysis.AnalysisOutputs
    compared: object | None
    reference: str = ""
    comparisons: str = COMPARISONS  # the part its comparisons go in (comparisons_top)


def _by_kind(figures) -> dict[str, list]:
    out: dict[str, list] = {}
    for stem, draw in figures:
        kind = kind_of(stem)
        if kind is not None:
            out.setdefault(kind.name, []).append((stem, draw))
    return out


def _first_value(kind: str, figures: list) -> dict[str, str]:
    """stem -> the value before the time and page (sample, form, class) of ``kind``'s
    figures ('' for a figure of every sample)."""
    choices = figure_choices(kind_named(kind), [s for s, _ in figures])
    first = next((lab for lab in choices.labels if lab not in (TIME, PAGE)), None)
    return {s: (choices.of(s).get(first, "") if first else "") for s, _ in figures}


_PARENT_SAMPLE = re.compile(r"_-?\d+-\d+(?:_z\d+)?\Z")


def _fragment_sample(parent: str, samples: Sequence[str]) -> str:
    """The sample of a fragment figure's parent (``NAME_START-END_zN``): the longest sample
    the parent's name begins with, else the name itself."""
    for s in sorted(samples, key=len, reverse=True):
        if parent.startswith(f"{s}_"):
            return s
    return _PARENT_SAMPLE.sub("", parent)


def entries(placed: Sequence[Placed], across=None) -> list[Entry]:
    """Every figure of the analyses ``placed`` at its place in the tree (``across``: the
    outputs of the regions across the groups, None without them)."""
    found: dict[tuple, Entry] = {}

    def add(path, kind, figure):
        entry = found.setdefault((*path, kind), Entry(tuple(path), kind))
        entry.figures.append(figure)

    for p in placed:
        g = (p.group,) if p.group else ()
        if p.levels is not None:
            window = getattr(p.levels, "window", {}) or {}
            pools = window.get("pools") or []
            samples = list(window.get("samples") or [])
            pool_of = {st: pool["class"] or "other" for pool in pools for st in pool["states"]}
            class_of_stem = {pool["stem"]: pool["class"] or "other" for pool in pools}
            class_of_form = {fs: pool["class"] or "other" for pool in pools
                             for fs in pool["forms"].values()}
            by_kind = _by_kind(p.levels.figures)
            fragment_samples = list(samples) + list(pool_of)

            def sample_path(s, *level, g=g, pool_of=pool_of):
                where = (pool_of[s],) if s in pool_of else ()
                return (SINGLE, *g, *where, s, *level)

            for kind, figures in by_kind.items():
                home = HOME.get(kind)
                if home == SUMMARY and kind == "cross_group_regions":
                    continue  # the across outputs' own
                elif home == SUMMARY:
                    values = _first_value(kind, figures)
                    for f in figures:
                        part = values[f[0]]
                        cls = class_of_stem.get(part)
                        add((SUMMARY, *g, *((cls,) if cls else ())), kind, f)
                    if kind == "coverage":  # also each sample's own, first under it
                        for f in figures:
                            add(sample_path(values[f[0]]), kind, f)
                elif kind in _FRAGMENT_KINDS:
                    choices = figure_choices(kind_named(kind), [s for s, _ in figures])
                    for f in figures:
                        parent = choices.of(f[0]).get("parent", "")
                        add(sample_path(_fragment_sample(parent, fragment_samples)), kind, f)
                elif home == SINGLE:  # protection, the population choice, the profile
                    for f, s in zip(figures, _first_value(kind, figures).values(), strict=True):
                        add(sample_path(s), kind, f)
            for s, figures in (window.get("single") or {}).items():
                for f in figures:
                    kind = kind_of(f[0])
                    if kind is not None:
                        add(sample_path(s), kind.name, f)
        if p.compared is not None:
            window = getattr(p.compared, "window", {}) or {}
            pools = window.get("pools") or []
            pool_of = {st: pool["class"] or "other" for pool in pools for st in pool["states"]}
            class_of_form = {fs: pool["class"] or "other" for pool in pools
                             for fs in pool["forms"].values()}
            head = (p.comparisons, *g)
            for kind, figures in _by_kind(p.compared.figures).items():
                if kind in _TOP_COMPARISONS:  # the same against every reference: once
                    if (HEAD_TO_HEAD, kind) not in found:
                        for f in figures:
                            add((HEAD_TO_HEAD,), kind, f)
                elif kind in _GROUP_COMPARISONS:
                    values = _first_value(kind, figures)
                    for f in figures:
                        cls = class_of_form.get(values[f[0]])
                        if kind in _EACH_SAMPLE_ONLY and not cls:
                            continue  # every sample's panels: each is under its sample
                        add((*head, *((cls,) if cls else ())), kind, f)
                elif kind in _SAMPLE_COMPARISONS:
                    for f, s in zip(figures, _first_value(kind, figures).values(), strict=True):
                        add((*head, *((pool_of[s],) if s in pool_of else ()), s), kind, f)
            for s, figures in (window.get("compared") or {}).items():
                for f in figures:
                    kind = kind_of(f[0])
                    if kind is not None:
                        add((*head, *((pool_of[s],) if s in pool_of else ()), s), kind.name, f)
    if across is not None:
        for f in getattr(across, "figures", []):
            add((SUMMARY,), "cross_group_regions", f)
    for entry in found.values():  # fragment figures by parent position, ranges as numbers
        if entry.kind in _FRAGMENT_KINDS:
            entry.figures.sort(key=lambda f: natural_key(f[0]))
    return sorted(found.values(), key=lambda e: _sort_key(e, placed))


def _sort_key(entry: Entry, placed: Sequence[Placed]) -> tuple:
    """Tops in order, groups as configured, samples as the analysis orders them (classes and
    their forms after the samples), levels in order, kinds in the order of :data:`HOME`."""
    top = TOPS.index(top_of(entry.path[0]))
    # the comparisons parts in the order of the references
    parts = [p.comparisons for p in placed]
    top = (top, parts.index(entry.path[0]) if entry.path[0] in parts else 0)
    groups = [p.group for p in placed]
    order: dict[str, int] = {}
    for p in placed:
        for outputs in (p.levels, p.compared):
            window = getattr(outputs, "window", {}) or {}
            for s in window.get("samples") or []:
                order.setdefault(s, len(order))
            for pool in window.get("pools") or []:
                order.setdefault(pool["class"] or "other", len(order) + 1000)
                for st in pool["states"]:
                    order.setdefault(st, len(order) + 1000)
    parts = []
    for name in entry.path[1:]:
        if name in groups:
            parts.append((0, groups.index(name), ""))
        elif name in LEVELS:
            parts.append((2, LEVELS.index(name), ""))
        else:
            parts.append((1, order.get(name, 10_000), name))
    # a kind directly under a row comes before the rows under it (a group's own figures
    # before its samples)
    return (top, *[(0, *x) for x in parts], (-1, _order(entry.path[0], entry.kind), ""))


def planned_entries(planned, has_figures: Callable[[str], bool] | None = None
                    ) -> tuple[list[Entry], list[Entry]]:
    """Before an analysis is computed: each kind the settings allow under its top (no
    figures yet), and the kinds they leave out under :data:`NOT_DRAWN`. With
    ``has_figures`` (an analysis computed), only the kinds with no figure go under Not drawn
    and none is listed at its top."""
    shown, left = [], []
    for p in planned:
        name = p.kind.name
        if name not in HOME:
            continue
        if p.state == "off" or (has_figures is not None and not has_figures(name)):
            left.append(Entry((NOT_DRAWN,), name))
        elif has_figures is None:
            top = HOME[name]
            shown.append(Entry((top,), name))
    shown.sort(key=lambda e: (TOPS.index(top_of(e.path[0])), _order(e.path[0], e.kind)))
    left.sort(key=lambda e: (TOPS.index(HOME[e.kind]), _order(HOME[e.kind], e.kind)))
    return shown, left


# the summary figures the window also draws for one sample (AnalysisOutputs.window "single")
ONE_SAMPLE_KINDS = ("residues_all", "redundancy_resolution", "population_tiers", "regions")
_SINGLE_TITLES = {"residues_all": "Residue fDavg"}  # drawn for one sample: not "every sample"


def title_at(entry: Entry, title: str) -> str:
    """A kind's title at its place in the tree (the summary figures, drawn for one sample
    under Single samples, lose their "every sample")."""
    if entry.path and entry.path[0] == SINGLE:
        return _SINGLE_TITLES.get(entry.kind, title)
    return title
