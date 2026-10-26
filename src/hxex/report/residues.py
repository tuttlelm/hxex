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
from ..io.times import format_time
from .coverage import MARK_LINE

# (K, rank) -> colour of its line and band, and its legend label (the owner's figures)
PROFILE_STYLES = {
    (1, 1): ("#6b6b6b", "K=1"),
    (2, 1): ("#33c3d6", "K=2 rank 1"),
    (2, 2): ("#f01ee0", "K=2 rank 2"),
}


def _when(time: float, scale: str = "seconds") -> str:
    """A time for a figure: in a readable unit, or ``undeuterated`` for a control without
    one (NaN); ``point 2`` when the times only order the points (``scale='order'``; the
    figure functions' ``time_scale``, :attr:`hxex.report.Results.time_scale`)."""
    if not np.isfinite(time):
        return "undeuterated"
    return f"point {format_time(time, scale)}" if scale == "order" else format_time(time)


NO_ERRORS = "no error bands: they need a bootstrap ([run.bootstrap], ks = [1, 2])"
SCATTER_ONLY = ("no bootstrap: error bands from the replicates' scatter only, too small "
                "([run.bootstrap], ks = [1, 2])")


def _no_errors_note(ax, values, ses, bootstrap: bool | None = None) -> None:
    """Say on ``ax`` that the values drawn have no error bands, when none of them has one, or
    that their bands come from the replicates' scatter only, when ``bootstrap`` is False."""
    values, ses = np.asarray(values, dtype=float), np.asarray(ses, dtype=float)
    drawn = np.isfinite(values)
    if not drawn.any():
        return
    text = (NO_ERRORS if not np.isfinite(ses[drawn]).any()
            else SCATTER_ONLY if bootstrap is False else None)
    if text:
        ax.text(0.01, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=8,
                color="#555555")


def plot_residue_profile(
    residues: ResidueMap,
    sample: str,
    time: float,
    *,
    show: Sequence[tuple[int, int]] = ((1, 1), (2, 1), (2, 2)),
    band: float = 0.5,
    marks: Sequence[float] = (),
    length: int | None = None,
    ax: Axes | None = None,
    legend: bool = True,
    time_scale: str = "seconds",
    bootstrap: bool | None = None,
) -> Figure:
    """The ``fD`` of every residue of ``sample`` at ``time`` for each ``(K, rank)`` in ``show``
    (those present in ``residues``), each with a band of the ``band`` confidence interval
    (0.5 by default; 0 for none) from the residue standard
    errors. ``marks`` draws vertical lines at those residues (user-defined sites or region
    boundaries); ``length`` sets the x range (else the residues of the map). Each line has the
    gid ``fD:K<k>:rank<r>``.

    ``time_scale`` (:attr:`hxex.report.Results.time_scale`): with ``'order'`` the time is
    labelled as a point number, not a time."""
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
    all_y, all_se = [], []
    for k, rank in show:
        rows = residues.where(sample=sample, time=time, K=k, rank=rank)
        if not rows.any():
            continue
        colour, label = PROFILE_STYLES.get((k, rank), ("#1f77b4", f"K={k} rank {rank}"))
        order = np.argsort(residues.residue[rows])
        x = residues.residue[rows][order]
        y = residues.fD[rows][order]
        se = residues.fD_se[rows][order]
        all_y.append(y)
        all_se.append(se)
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
    ax.text(0.99, 0.04, f"{sample} {_when(time, time_scale)}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10)
    _no_errors_note(ax, np.concatenate(all_y), np.concatenate(all_se), bootstrap)
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
    fig: Figure | None = None,
    time_scale: str = "seconds",
    bootstrap: bool | None = None,
) -> Figure:
    """The choice between one and two populations along the sequence
    (:func:`hxex.analysis.choose_populations`): the K=1 and
    K=2 curves with bands, solid where their description was chosen and dashed elsewhere;
    markers for the chosen values, K=2 sized by population and K=1 as open circles; below, the
    population of each component where two were chosen. Lines have the gid ``fD:<name>``,
    markers ``chosen:<name>``.

    ``time_scale`` (:attr:`hxex.report.Results.time_scale`): with ``'order'`` the time is
    labelled as a point number, not a time. ``bootstrap`` False (the results have none) adds a
    note that the error bands are the replicates' scatter only; with no error at all the
    figure says so whatever ``bootstrap`` is."""
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
    top.text(0.99, 0.04, f"{sample} {_when(time, time_scale)}", transform=top.transAxes,
             ha="right", va="bottom", fontsize=10)
    shown = [(getattr(choice, f"fD_{n}")[rows], getattr(choice, f"fD_{n}_se")[rows])
             for n in ("avg", "low", "high")]
    _no_errors_note(top, np.concatenate([v for v, _ in shown]),
                    np.concatenate([e for _, e in shown]), bootstrap)
    top.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False,
               fontsize=8)
    bottom.set_ylim(0, 1.05)
    bottom.set_ylabel("population")
    bottom.set_xlabel("residue")
    bottom.set_xlim(0, length if length is not None else n_res)
    return fig
