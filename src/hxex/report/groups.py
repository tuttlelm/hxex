# SPDX-License-Identifier: GPL-3.0-or-later
"""Across the groups of ``[analysis.groups]`` (docs/ANALYSIS.md "Groups of samples"): each
group's regions, matched by name between groups (each over its own protein's residues), in one
table, ``cross_group_regions.csv``, and one figure per time, ``cross_group_regions_<time>``,
written by ``hxex analyse`` into ``results_analysis_groups`` once every group is analysed."""

from __future__ import annotations

import logging
import textwrap
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from ..io.outputs import replace_outputs
from ..io.times import format_time
from . import hover
from .tables import write_csv

log = logging.getLogger(__name__)

CROSS_GROUPS = "groups"  # the folder: the bundle's name + _analysis_groups
CROSS_TABLE = "cross_group_regions"
COLUMNS = ("label", "group", "sample", "reference", "region", "start", "end", "time", "fD",
           "fD_se", "n_residues", "coverage", "delta", "delta_se", "significant")


def shared_region_names(config) -> list[str]:
    """The region names two groups or more of ``config`` (with ``[analysis.groups]``) have,
    each group's own regions or the shared ones (:meth:`hxex.analysis.AnalysisConfig.one`), in
    the order of their first appearance."""
    seen: dict[str, set[str]] = {}
    for name in config.groups or {}:
        for region in config.one(name).annotations.regions:
            seen.setdefault(region.name, set()).add(name)
    return [r for r, groups in seen.items() if len(groups) > 1]


def cross_group_regions(analyses: Mapping[str, tuple]) -> dict[str, np.ndarray]:
    """``cross_group_regions.csv``: one row per vial label, group, region, and time, from each
    group's analysis (group -> ``(config, tables)``; the config of :meth:`hxex.analysis.
    AnalysisConfig.with_group`, the tables of its :class:`hxex.report.AnalysisOutputs`): the
    group's sample of that label and its reference, the region's residues in that group
    (``start``, ``end``: they differ between proteins), ``fD``, ``fD_se``, ``n_residues`` and
    ``coverage`` (``regions.csv``), and the group's own region test against its reference
    (``compare_regions.csv``): ``delta`` and ``delta_se`` (the difference and error the test
    used) and ``significant`` (empty for the reference). Regions are matched by name; a group
    without a region of that name has no row for it. Rows in the order of the groups, each
    group's reference first, then its labels in order."""
    rows: list[dict] = []
    for name, (config, tables) in analyses.items():
        group = config.groups[name]
        regions = tables.get("regions")
        if not regions or not len(regions.get("region", ())):
            continue
        tests = tables.get("compare_regions") or {}
        test_of = {}
        for i in range(len(tests.get("region", ()))):
            key = (str(tests["sample"][i]), str(tests["region"][i]), float(tests["time"][i]))
            test_of[key] = (float(tests["delta_shared"][i]), float(tests["delta_shared_se"][i]),
                            bool(tests["significant"][i]))
        order = [group.reference] + [s for s in group.samples if s != group.reference]
        by_sample: dict[str, list[int]] = {}
        for i, sample in enumerate(np.asarray(regions["sample"]).astype(str)):
            by_sample.setdefault(sample, []).append(i)
        for sample in order:
            for i in by_sample.get(sample, ()):
                region, time = str(regions["region"][i]), float(regions["time"][i])
                delta, delta_se, significant = test_of.get(
                    (sample, region, time), (np.nan, np.nan, None))
                rows.append({
                    "label": group.label_of(sample), "group": name, "sample": sample,
                    "reference": group.reference, "region": region,
                    "start": int(regions["start"][i]), "end": int(regions["end"][i]),
                    "time": time, "fD": float(regions["fD"][i]),
                    "fD_se": float(regions["fD_se"][i]),
                    "n_residues": int(regions["n_residues"][i]),
                    "coverage": float(regions["coverage"][i]),
                    "delta": delta, "delta_se": delta_se,
                    "significant": "" if significant is None else str(significant)})
    out = {}
    for column in COLUMNS:
        values = [r[column] for r in rows]
        out[column] = np.array(values, dtype=object if column in (
            "label", "group", "sample", "reference", "region", "significant") else float
            if column not in ("start", "end", "n_residues") else np.int64)
    return out


# one sequential colour map per group: its reference darkest, its other vials lighter (owner,
# 2026-11-18); past the list the maps repeat
GROUP_MAPS = ("Blues", "Oranges", "Greens", "Purples", "Reds", "Greys", "YlOrBr", "BuPu")
BAR_WIDTH = 0.16  # inches per bar
REGION_GAP = 0.35  # inches between two regions' bars
ROW_HEIGHT = 2.6  # inches per row of panels


def _can_underline() -> bool:
    """Whether this matplotlib's mathtext has ``\\underline`` (3.9 and later; 3.8 does not)."""
    from matplotlib.mathtext import MathTextParser

    try:
        MathTextParser("path").parse(r"$\underline{x}$")
    except ValueError:
        return False
    return True


def _math_text(text: str) -> str:
    """``text`` as upright mathtext (its spaces and special characters kept)."""
    out = []
    for ch in text:
        if ch == " ":
            out.append(r"\ ")
        elif ch in "\\{}$_^%#&~":
            out.append("\\" + ch if ch in "{}$_%#&" else {"\\": r"\backslash", "^": r"\hat{}",
                                                         "~": r"\sim"}[ch])
        else:
            out.append(ch)
    return r"\mathrm{" + "".join(out) + "}"


def vial_label(group: str, label: str, underline: bool | None = None) -> str:
    """A bar's legend entry: the vial ``label`` with the measured protein (``group``) underlined
    (bold where matplotlib cannot underline, 3.8): its part of the label split at ``+``, or the
    group's name alone for a label that does not name it (``alone``)."""
    from ..analysis.shared import vial_groups

    underline = _can_underline() if underline is None else underline

    def mark(text: str) -> str:
        inner = _math_text(text)
        return f"$\\underline{{{inner}}}$" if underline else f"$\\mathbf{{{inner[8:-1]}}}$"

    parts = [p.strip() for p in str(label).split("+")]
    if group not in vial_groups(label):
        return mark(group)
    return "+".join(mark(p) if p == group else p.replace("$", r"\$") for p in parts)


def _bars(table: Mapping[str, np.ndarray]) -> list[tuple[str, str, str]]:
    """``(group, label, sample)`` of every bar, in the table's order (by group, each group's
    reference first)."""
    return list(dict.fromkeys(zip(np.asarray(table["group"]).astype(str),
                                  np.asarray(table["label"]).astype(str),
                                  np.asarray(table["sample"]).astype(str), strict=True)))


def bar_colours(bars: list[tuple[str, str, str]]) -> dict[tuple[str, str, str], tuple]:
    """Each bar's colour: one colour map per group (:data:`GROUP_MAPS`), the group's first bar
    (its reference) darkest, its others lighter."""
    groups = list(dict.fromkeys(g for g, _, _ in bars))
    out = {}
    for n, group in enumerate(groups):
        mine = [b for b in bars if b[0] == group]
        cmap = colormaps[GROUP_MAPS[n % len(GROUP_MAPS)]]
        for i, bar in zip(np.linspace(0.92, 0.45, max(len(mine), 2))[:len(mine)], mine,
                          strict=True):
            out[bar] = cmap(float(i))
    return out


def plot_cross_group_regions(table: Mapping[str, np.ndarray], time: float, *,
                             page_width: float = 16.0, time_scale: str = "seconds") -> Figure:
    """The Regions figure across the groups at ``time`` (owner, 2026-11-18, from drafts): grouped
    bars of each region's ``fD`` (``table``: :func:`cross_group_regions`), the region names
    along x, each region's bars from every group that has it, each over its own residues; the
    bars ordered by group, each group's reference first, one colour map per group (the
    reference darkest); the legend names each bar by its vial label with the measured protein
    underlined (:func:`vial_label`). Regions go on further rows of panels within
    ``page_width`` inches, never one very wide row. Each bar has the gid
    ``region:<group>:<sample>:<region>`` and its residues in its hover."""
    sel = np.asarray(table["time"], dtype=float) == float(time)
    if not sel.any():
        raise ValueError(f"no regions across the groups at time {time:g}")
    rows = {k: np.asarray(v)[sel] for k, v in table.items()}
    bars = _bars(rows)
    colours = bar_colours(bars)
    names = list(dict.fromkeys(np.asarray(rows["region"]).astype(str)))
    region_width = BAR_WIDTH * len(bars) + REGION_GAP
    per_row = max(1, int((page_width - 1.0) // region_width))
    n_rows = -(-len(names) // per_row)
    per_row = -(-len(names) // n_rows)  # balanced over the rows
    width = max(6.0, min(page_width, region_width * per_row + 1.0))
    fig = Figure(figsize=(width, ROW_HEIGHT * n_rows + 0.9), layout="constrained")
    axes = fig.subplots(n_rows, 1, squeeze=False)[:, 0]
    index = {}
    for i in range(len(rows["region"])):
        index[(str(rows["group"][i]), str(rows["sample"][i]), str(rows["region"][i]))] = i
    top = 1.05
    step = 0.86 / len(bars)
    labels = {b: vial_label(b[0], b[1]) for b in bars}
    for r, ax in enumerate(axes):
        shown = names[r * per_row:(r + 1) * per_row]
        x = np.arange(len(shown))
        for n, bar in enumerate(bars):
            group, label, sample = bar
            found = [index.get((group, sample, name)) for name in shown]
            values = np.array([np.nan if i is None else float(rows["fD"][i]) for i in found])
            errors = np.array([np.nan if i is None else float(rows["fD_se"][i]) for i in found])
            drawn = ax.bar(x + (n - (len(bars) - 1) / 2) * step, values, width=step,
                           color=colours[bar], yerr=np.nan_to_num(errors),
                           label=labels[bar] if r == 0 else None,
                           error_kw={"elinewidth": 0.6, "ecolor": "#555555"})
            for patch, name, i in zip(drawn.patches, shown, found, strict=True):
                patch.set_gid(f"region:{group}:{sample}:{name}")
                if i is not None:
                    hover.set_hover(patch, f"{label} ({group}: {sample}), region {name}, "
                                    f"residues {rows['start'][i]}-{rows['end'][i]}",
                                    hover.value("fD", rows["fD"][i], rows["fD_se"][i]))
            finite = values + np.nan_to_num(errors)
            finite = finite[np.isfinite(finite)]
            if len(finite):
                top = max(top, 1.05 * float(finite.max()))
        ax.set_xticks(x, [textwrap.fill(name.replace(",", ", "), 12).replace(", ", ",")
                          for name in shown], fontsize=8)
        ax.set_xlim(-0.5, per_row - 0.5)
        ax.set_ylabel("fD")
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
    for ax in axes:
        ax.set_ylim(0, top)
    handles, texts = axes[0].get_legend_handles_labels()
    underline = "underlined" if _can_underline() else "in bold"
    # the caption as the legend's title: above its entries, where nothing else is drawn (under
    # the panels it ran into the region names or the provisional note)
    legend = fig.legend(
        handles, texts, loc="outside upper center", frameon=False, fontsize=9,
        ncol=min(len(texts), max(1, int((width - 1.0) // 1.4))),
        title=(f"{format_time(time, time_scale)}; {underline}: the protein measured; each "
               "region over that protein's own residues"), title_fontsize=8)
    legend.get_title().set_color("#4d4d4d")
    legend.get_title().set_style("italic")
    return fig


def write_cross_groups(analyses: Mapping[str, tuple], directory: str | Path, *,
                       made: Mapping[str, str] | None = None, figure_format: str = "pdf",
                       time_scale: str = "seconds", page_width: float = 16.0,
                       note: str | None = None) -> list[Path]:
    """Write ``cross_group_regions.csv`` (:func:`cross_group_regions`) and one figure per time
    (:func:`plot_cross_group_regions`, ``cross_group_regions_<time>``, marked provisional with
    ``note``) into ``directory``, as the analysis folders are written (the files of an earlier
    write go, yours stay; ``made`` records the results read); returns the files (none when no
    group has a region)."""
    from .analysis import _time_label
    from .results import mark_provisional

    table = cross_group_regions(analyses)
    if not len(table["region"]):
        log.info("across the groups: no region in any group's analysis, nothing written")
        return []
    directory = Path(directory)
    times = sorted({float(t) for t in table["time"]})
    figures = {f"{CROSS_TABLE}_{_time_label(t, time_scale)}.{figure_format}": t for t in times}
    with replace_outputs(directory, [f"{CROSS_TABLE}.csv", *figures], made=made) as written:
        written.append(write_csv(table, directory / f"{CROSS_TABLE}.csv"))
        for name, t in figures.items():
            fig = plot_cross_group_regions(table, t, page_width=page_width,
                                           time_scale=time_scale)
            mark_provisional(fig, note).savefig(directory / name, dpi=150, bbox_inches="tight")
            written.append(directory / name)
    return list(written)
