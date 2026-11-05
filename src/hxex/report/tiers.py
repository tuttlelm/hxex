# SPDX-License-Identifier: GPL-3.0-or-later
"""The residue tiers of the population test (:func:`hxex.analysis.residues.population_tiers`):
one row per sample and point (the undeuterated control, the fully deuterated control, then the
times), one cell per residue, coloured by its tier. The owner's reference is the tier map of
HISTORY 148; the two tiers beyond two take a rose pair, apart from the blues under every
simulated colour-vision deficiency (OKLab difference at least 9.9 between any two tiers).
Functions return figures and never show them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from ..analysis import ResidueMap
from ..analysis.residues import TIERS
from ..io.times import format_time
from ..spectra import FULLDEUT, UNDEUT
from . import hover

# no sign, undecided, two, two localised (the owner's), three, three localised
TIER_COLOURS = ("#c3c8d0", "#e8b928", "#6b9be6", "#173f86", "#d77fa1", "#8e1f4f")
TIER_LABELS = ("no sign", "undecided (one criterion)", "two (agreement 1/3 to 1/2)",
               "two, localised (agreement 1/2 or more)", "three (agreement 1/3 to 1/2)",
               "three, localised (agreement 1/2 or more)")


def _point_label(time: float, role: str, scale: str) -> str:
    if role == UNDEUT:
        return "undeuterated"
    if role == FULLDEUT:
        return "fully deuterated"
    return f"point {format_time(time, scale)}" if scale == "order" else format_time(time)


def plot_population_tiers(
    residues: ResidueMap,
    samples: Sequence[str] | None = None,
    *,
    length: int | None = None,
    time_scale: str = "seconds",
    times: Sequence[float] | None = None,
) -> Figure:
    """The tier map of ``residues`` (a residue map made with the population test) for
    ``samples`` (default: all, in table order). White cells: a residue no peptide covers, or
    one without an amide. ``times``: only those points (a page of a long time course; NaN
    stands for the controls' rows)."""
    k1 = (residues.K == 1) & (residues.population_tier != "")
    if not k1.any():
        raise ValueError("the residue map has no population tiers (made without the "
                         "population test)")
    if samples is None:
        samples = list(dict.fromkeys(str(s) for s in residues.sample[k1]))
    code_of = {name: j for j, name in enumerate(TIERS)}
    n = int(length if length is not None else residues.residue[k1].max())
    rows, labels, first_of_sample, table = [], [], [], []
    three_columns = residues.test_strength_3 is not None
    for s in samples:
        here = k1 & (residues.sample == s)
        points = {}
        for i in np.flatnonzero(here):
            t = float(residues.time[i])
            key = (str(residues.role[i]), None if not np.isfinite(t) else t)
            points.setdefault(key, i)
        order = sorted(points, key=lambda k: ({UNDEUT: 0, FULLDEUT: 1}.get(k[0], 2),
                                               -1.0 if k[1] is None else k[1]))
        if times is not None:
            wanted = [float(x) for x in times]
            controls = any(not np.isfinite(x) for x in wanted)
            order = [(role, t) for role, t in order
                     if (controls if t is None or role in (UNDEUT, FULLDEUT) else t in wanted)]
        for j, (role, t) in enumerate(order):
            sel = here & (residues.role == role)
            sel &= (np.isnan(residues.time) if t is None else residues.time == t)
            line = np.full(n, np.nan)
            point = _point_label(np.nan if t is None else t, role, time_scale)
            for i in np.flatnonzero(sel):
                r = int(residues.residue[i])
                if 1 <= r <= n and not residues.no_amide[i] and residues.population_tier[i]:
                    line[r - 1] = code_of[str(residues.population_tier[i])]
                    row = {"sample": s, "time": np.nan if t is None else t,
                           "time_label": point, "residue": r,
                           "population_tier": str(residues.population_tier[i]),
                           "test_strength_2": float(residues.test_strength_2[i]),
                           "test_agreement_2": float(residues.test_agreement_2[i])}
                    if three_columns:
                        row["test_strength_3"] = float(residues.test_strength_3[i])
                        row["test_agreement_3"] = float(residues.test_agreement_3[i])
                    table.append(row)
            rows.append(line)
            labels.append(f"{s}  {point}")
            first_of_sample.append(j == 0)
    image = np.ma.masked_invalid(np.array(rows))
    three = bool(np.isfinite(residues.test_strength_3[k1]).any())
    shown = len(TIERS) if three else 4
    cmap = ListedColormap(list(TIER_COLOURS)).with_extremes(bad="white")
    fig = Figure(figsize=(max(8.0, min(16.0, 0.06 * n + 3.0)), 0.2 * len(rows) + 1.7))
    ax = fig.add_subplot()
    cells = ax.imshow(image, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(TIERS) - 0.5,
                      interpolation="nearest", extent=(0.5, n + 0.5, len(rows) - 0.5, -0.5))
    hover.set_rows(cells, table)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis="y", length=2)
    for j in range(1, len(rows)):
        if first_of_sample[j]:
            ax.axhline(j - 0.5, color="white", lw=3)
    ax.set_xlim(0.5, n + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel("residue")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Patch(color=TIER_COLOURS[j], label=TIER_LABELS[j]) for j in range(shown)][::-1]
    handles.append(Patch(facecolor="white", edgecolor="0.6", label="uncovered or no amide"))
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=3 if three else 5,
               fontsize=7)
    ax.set_title("Population test by residue: strong = combined p of the covering peptides "
                 "below 0.01; agreement = share of their weight with the populations "
                 "(undeuterated rows should show none)", fontsize=8, loc="left")
    fig.tight_layout(rect=(0, 0.07 if three else 0.05, 1, 1))
    return fig
