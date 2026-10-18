# SPDX-License-Identifier: GPL-3.0-or-later
"""Residue profiles of the post-fitting analysis (docs/ANALYSIS.md): the fractional deuteration
(``fD``) of every residue along the sequence for one sample and time, from
:func:`hxex.analysis.residue_map`.

:func:`plot_residue_profile` draws the one-population value (K = 1) and the two populations of
the K = 2 fits (rank 1, the lower deuteration, and rank 2), each with an uncertainty band, gaps
where a residue has no value. :func:`plot_population_choice` adds the choice between one and two
populations (:func:`hxex.analysis.choose_populations`): solid and dashed lines, markers for the
chosen values and a population panel. Functions return figures and never show them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import norm

from ..analysis import ResidueMap
from .coverage import MARK_LINE

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
    (0.5 by default; 0 for none) from the residue standard
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
        ax.axvline(x0, **MARK_LINE)
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


# markers of the chosen description (the owner's figures): K=2 low and high, and K=1
CHOSEN_STYLES = {
    "low": dict(color="#1f8a3a", edgecolor="#0f5a22"),
    "high": dict(color="#8c0a12", edgecolor="#5a0008"),
    "avg": dict(color="none", edgecolor="#000000"),
}


def _segments(x: np.ndarray, y: np.ndarray, chosen: np.ndarray):
    """``y`` where ``chosen`` and where not (NaN elsewhere), each keeping the point next to a
    switch so that the solid and dashed parts meet."""
    solid = np.where(chosen, y, np.nan)
    dashed = np.where(~chosen, y, np.nan)
    for s in np.flatnonzero(chosen[1:] != chosen[:-1]):  # the dashed part reaches the solid one
        dashed[s : s + 2] = y[s : s + 2]
    return solid, dashed


def plot_population_choice(
    choice,
    sample: str,
    time: float,
    *,
    band: float = 0.5,
    marks: Sequence[float] = (),
    length: int | None = None,
    undeut_time: float | None = None,
    fig: Figure | None = None,
) -> Figure:
    """The choice between one and two populations along the sequence
    (:func:`hxex.analysis.choose_populations`): the K=1 and
    K=2 curves with bands, solid where their description was chosen and dashed elsewhere;
    markers for the chosen values, K=2 sized by population and K=1 as open circles; below, the
    population of each component where two were chosen. Lines have the gid ``fD:<name>``,
    markers ``chosen:<name>``."""
    rows = choice.where(sample=sample, time=time)
    if not rows.any():
        raise ValueError(f"no choice for {sample!r} at time {time:g}")
    if not 0 <= band < 1:
        raise ValueError("band must be a confidence level in [0, 1)")
    z = norm.ppf(0.5 + band / 2) if band > 0 else 0.0
    order = np.argsort(choice.residue[rows])
    x = choice.residue[rows][order]
    two = choice.two_populations[rows][order]
    pop = choice.population_low[rows][order]
    if fig is None:
        fig = Figure(figsize=(8.0, 4.6))
    top, bottom = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": (2.6, 1)})
    curves = (("avg", (1, 1), ~two), ("low", (2, 1), two), ("high", (2, 2), two))
    for name, style_key, chosen in curves:
        y = getattr(choice, f"fD_{name}")[rows][order]
        se = getattr(choice, f"fD_{name}_se")[rows][order]
        colour, label = PROFILE_STYLES[style_key]
        if z > 0:
            top.fill_between(x, y - z * se, y + z * se, color=colour, alpha=0.2, linewidth=0,
                             zorder=2)
        solid, dashed = _segments(x, y, chosen & np.isfinite(y))
        (line,) = top.plot(x, solid, color=colour, linewidth=2.0, label=label, zorder=3)
        line.set_gid(f"fD:{name}")
        top.plot(x, dashed, color=colour, linewidth=2.0, linestyle=(0, (4, 3)), zorder=3)
    avg = choice.fD_avg[rows][order]
    pts = top.scatter(x[~two], avg[~two], s=40, facecolors="none", linewidths=1.6, zorder=4,
                      label="K=1 chosen", **{k: v for k, v in CHOSEN_STYLES["avg"].items()
                                             if k != "color"})
    pts.set_gid("chosen:avg")
    for name, share in (("low", pop), ("high", 1 - pop)):
        y = getattr(choice, f"fD_{name}")[rows][order]
        pts = top.scatter(x[two], y[two], s=12 + 70 * np.nan_to_num(share[two]),
                          color=CHOSEN_STYLES[name]["color"],
                          edgecolors=CHOSEN_STYLES[name]["edgecolor"], linewidths=0.6, zorder=5,
                          label=f"{name}-D chosen")
        pts.set_gid(f"chosen:{name}")
        b = bottom.scatter(x[two], share[two], s=18, color=CHOSEN_STYLES[name]["color"],
                           edgecolors=CHOSEN_STYLES[name]["edgecolor"], linewidths=0.6, zorder=3)
        b.set_gid(f"population:{name}")
    n_res = int(x.max())
    for ax in (top, bottom):
        for x0 in marks:
            ax.axvline(x0, **MARK_LINE)
        ax.grid(color="#e5e5e5", linewidth=0.8)
    finite = np.concatenate([getattr(choice, f"fD_{n}")[rows] for n in ("avg", "low", "high")])
    finite = finite[np.isfinite(finite)]
    top.set_ylim(0, max(1.1, float(finite.max()) * 1.05 if len(finite) else 1.1))
    top.set_ylabel("fD (fractional deuteration)")
    top.text(0.99, 0.04, f"{sample} {_when(time, undeut_time)}", transform=top.transAxes,
             ha="right", va="bottom", fontsize=10)
    top.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False,
               fontsize=8)
    bottom.set_ylim(0, 1.05)
    bottom.set_ylabel("population")
    bottom.set_xlabel("residue")
    bottom.set_xlim(0, length if length is not None else n_res)
    return fig
