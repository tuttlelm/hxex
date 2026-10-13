# SPDX-License-Identifier: GPL-3.0-or-later
"""Residue profiles of the post-fitting analysis (docs/ANALYSIS.md): the fractional deuteration
(``fD``) of every residue along the sequence for one sample and time, from
:func:`hxex.analysis.residue_map`.

:func:`plot_residue_profile` draws the one-population value (K = 1) and the two populations of
the K = 2 fits (rank 1, the lower deuteration, and rank 2), each with an uncertainty band, gaps
where a residue has no value. The choice between one and two populations (phase 2) adds markers
and a population panel to the same figure. Functions return figures and never show them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import norm

from ..analysis import ResidueMap

# (K, rank) -> colour of its line and band, and its legend label (the owner's figures)
PROFILE_STYLES = {
    (1, 1): ("#6b6b6b", "K=1"),
    (2, 1): ("#33c3d6", "K=2 rank 1"),
    (2, 2): ("#f01ee0", "K=2 rank 2"),
}


def _when(time: float, undeut_time: float | None) -> str:
    if undeut_time is not None and np.isclose(time, undeut_time):
        return "undeuterated"
    return f"{time:g} s"


def plot_residue_profile(
    residues: ResidueMap,
    sample: str,
    time: float,
    *,
    show: Sequence[tuple[int, int]] = ((1, 1), (2, 1), (2, 2)),
    band: float = 0.5,
    marks: Sequence[float] = (),
    length: int | None = None,
    undeut_time: float | None = None,
    ax: Axes | None = None,
    legend: bool = True,
) -> Figure:
    """The ``fD`` of every residue of ``sample`` at ``time`` for each ``(K, rank)`` in ``show``
    (those present in ``residues``), each with a band of the ``band`` confidence interval
    (0.5 by default, as pyHXExpress's ``choose_fits``; 0 for none) from the residue standard
    errors. ``marks`` draws vertical lines at those residues (user-defined sites or region
    boundaries); ``length`` sets the x range (else the residues of the map). Each line has the
    gid ``fD:K<k>:rank<r>``."""
    if not 0 <= band < 1:
        raise ValueError("band must be a confidence level in [0, 1)")
    z = norm.ppf(0.5 + band / 2) if band > 0 else 0.0
    if ax is None:
        fig = Figure(figsize=(8.0, 3.4))
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    drawn = 0
    top = 1.0
    n_res = 0
    for k, rank in show:
        rows = residues.where(sample=sample, time=time, K=k, rank=rank)
        if not rows.any():
            continue
        colour, label = PROFILE_STYLES.get((k, rank), ("#1f77b4", f"K={k} rank {rank}"))
        order = np.argsort(residues.residue[rows])
        x = residues.residue[rows][order]
        y = residues.fD[rows][order]
        se = residues.fD_se[rows][order]
        n_res = max(n_res, int(x.max()))
        if z > 0:
            ax.fill_between(x, y - z * se, y + z * se, color=colour, alpha=0.25, linewidth=0,
                            step=None, zorder=2)
        (line,) = ax.plot(x, y, color=colour, linewidth=2.0, label=label, zorder=3)
        line.set_gid(f"fD:K{k}:rank{rank}")
        finite = y[np.isfinite(y)]
        if len(finite):
            top = max(top, float(finite.max()))
        drawn += 1
    if not drawn:
        raise ValueError(f"no residue values of {sample!r} at time {time:g}")
    for x0 in marks:
        ax.axvline(x0, color="#8f86c4", linewidth=1.2, zorder=1)
    ax.set_xlim(0, length if length is not None else n_res)
    ax.set_ylim(0, max(1.1, top * 1.05))
    ax.set_xlabel("residue")
    ax.set_ylabel("fD (fractional deuteration)")
    ax.grid(color="#e5e5e5", linewidth=0.8)
    ax.text(0.99, 0.04, f"{sample} {_when(time, undeut_time)}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10)
    if legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=drawn, frameon=False)
    return fig
