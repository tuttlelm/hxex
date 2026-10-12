# SPDX-License-Identifier: GPL-3.0-or-later
"""Plots of one series (a peptide at one charge): spectra by time and replicate, the populations
of every spectrum, and the uptake along the time course.

Each plot has a ``*_data`` function that returns the arrays it draws; the plot only draws them.
Populations are identified by the rank of their mean within each spectrum (1 = lowest), as
chosen for every time point independently.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..controls import correct_means
from ..spectra import time_index
from .results import Results
from .spectrum import LAYERS, plot_spectrum, population_style

# replicate shapes in the overlay of the population map (there the colour is the population)
REP_MARKERS = ("o", "s", "^", "D", "v", "P")

# replicates (uptake plot): Okabe-Ito colours, the first two differing most
REP_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")


@dataclass(frozen=True)
class Reference:
    """Known populations of a test data set, to draw next to the fitted ones: for each time, the
    known means as a fraction of the number of exchangeable amides and the known fractions."""

    times: np.ndarray
    level: np.ndarray  # (T, P) fraction of n_amides, NaN where the population is absent
    fraction: np.ndarray  # (T, P)

    def at(self, time: float, n_amides: float, atol: float = 1e-6):
        """(means in D, fractions) of the populations known at ``time``, or None."""
        hit = np.flatnonzero(np.abs(self.times - time) <= atol)
        if not len(hit):
            return None
        lv, fr = self.level[hit[0]], self.fraction[hit[0]]
        keep = np.isfinite(lv) & np.isfinite(fr) & (fr > 0)
        return lv[keep] * n_amides, fr[keep]


def load_reference(path) -> Reference:
    """Read a tab-separated table of known populations: ``time``, ``fd1``..., ``p1``... (the
    layout of the Bimodal test set; ``fd`` in percent of the amides, ``#N/A`` for none)."""
    from pathlib import Path

    rows = [r.split("\t") for r in Path(path).read_text().splitlines() if r.strip()]
    head = [h.strip().lower() for h in rows[0]]
    fd_cols = [j for j, h in enumerate(head) if h.startswith("fd")]
    p_cols = [j for j, h in enumerate(head) if h.startswith("p") and h[1:].isdigit()]

    def num(text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return float("nan")

    times = np.array([num(r[0]) for r in rows[1:]])
    level = np.array([[num(r[j]) / 100.0 for j in fd_cols] for r in rows[1:]])
    frac = np.array([[num(r[j]) for j in p_cols] for r in rows[1:]])
    return Reference(times, level, frac)


def series_rows(results: Results, data_id: int) -> np.ndarray:
    """Positions of the series' spectra ordered by time, then replicate (the order the picked
    table gives within a time; without it, the order in the results)."""
    rows = np.flatnonzero(results.fit.data_id == data_id)
    if not len(rows):
        raise KeyError(data_id)
    rep = _reps(results)[rows]
    return rows[np.lexsort((np.arange(len(rows)), rep, results.fit.time[rows]))]


def _reps(results: Results) -> np.ndarray:
    """Replicate number of every spectrum (from the picked table; else 1, 2, ... within a time)."""
    fit = results.fit
    if results.binned is not None:
        index = {str(s): j for j, s in enumerate(results.binned.spectrum_id)}
        if all(str(s) in index for s in fit.spectrum_id):
            return np.array([int(results.binned.rep[index[str(s)]]) for s in fit.spectrum_id])
    out = np.ones(len(fit), dtype=np.int64)
    seen: dict[tuple[int, float], int] = {}
    for i in range(len(fit)):
        key = (int(fit.data_id[i]), float(fit.time[i]))
        seen[key] = seen.get(key, 0) + 1
        out[i] = seen[key]
    return out


def series_info(results: Results, data_id: int) -> dict:
    """Description of a series: ``rows``, ``times``, ``reps``, ``time_index`` of each spectrum,
    ``label`` (sample, range, peptide, charge when the picked table is there) and ``n_amides``
    (from the controls, else None)."""
    rows = series_rows(results, data_id)
    fit = results.fit
    label = f"series {data_id}"
    if results.binned is not None:
        j = results.binned.find(str(fit.spectrum_id[rows[0]]))
        b = results.binned
        label = f"{b.sample[j]} {b.peptide_range[j]} {b.peptide[j]} z={int(b.charge[j])}".strip()
    row = _control_row(results, data_id)
    n_amides = None if row is None else float(results.controls.n_amides[row])
    return {
        "data_id": data_id, "rows": rows, "label": label, "n_amides": n_amides,
        "times": fit.time[rows], "reps": _reps(results)[rows],
        "time_index": time_index(fit.data_id[rows], fit.time[rows]),
    }


def _layout(info: dict, columns: int | None) -> tuple[int, int, list[tuple[int, int]], bool]:
    """Rows, columns and the (row, column) of every spectrum of a series (in ``info`` order).

    With several replicates (and ``columns`` not given) a row is a time and a column a replicate.
    Otherwise the spectra, in time order, fill a grid of ``columns`` columns (default: a near
    square of at most 4 columns), so that a long single-replicate time course is not one tall
    column. The last value says whether the layout is time by replicate."""
    n = len(info["rows"])
    times, reps = np.unique(info["times"]), np.unique(info["reps"])
    if columns is None and len(reps) > 1:
        pos = [(int(np.searchsorted(times, t)), int(np.searchsorted(reps, r)))
               for t, r in zip(info["times"], info["reps"], strict=True)]
        return len(times), len(reps), pos, True
    if columns is not None and columns < 1:
        raise ValueError("columns must be at least 1")
    ncols = columns or max(1, min(4, math.ceil(math.sqrt(n))))
    ncols = min(ncols, n)
    return math.ceil(n / ncols), ncols, [(i // ncols, i % ncols) for i in range(n)], False


def _grid(info: dict, size: tuple[float, float], columns: int | None):
    nrows, ncols, positions, _ = _layout(info, columns)
    fig = Figure(figsize=(size[0] * ncols, size[1] * nrows))
    axes = fig.subplots(nrows, ncols, squeeze=False)
    cells = {int(info["rows"][i]): axes[r, c] for i, (r, c) in enumerate(positions)}
    for ax in axes.ravel():
        if ax not in cells.values():
            ax.set_visible(False)
    return fig, cells


def plot_series_spectra(
    results: Results,
    data_id: int,
    *,
    show: Sequence[str] = LAYERS,
    components: Sequence[int] | None = None,
    k: int | None = None,
    columns: int | None = None,
    cell: tuple[float, float] = (4.5, 3.0),
    **kwargs,
) -> Figure:
    """A grid of spectra: with replicates a row per time and a column per replicate, otherwise
    the time course wrapped into ``columns`` columns (see :func:`_layout`; see
    :func:`plot_spectrum` for the keyword arguments, which are passed on). Spectra whose layers
    cannot be drawn are left blank."""
    info = series_info(results, data_id)
    fig, cells = _grid(info, cell, columns)
    for i, ax in cells.items():
        sid = str(results.fit.spectrum_id[i])
        try:
            plot_spectrum(sid, results, show=show, k=k, components=components, ax=ax,
                          legend=False, **kwargs)
        except ValueError as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", fontsize=6, wrap=True,
                    transform=ax.transAxes)
        pos = list(info["rows"]).index(i)
        ax.set_title(f"{_time_text(results, info['times'][pos])}, rep {info['reps'][pos]}",
                     fontsize=8)
    fig.suptitle(info["label"], fontsize=10)
    fig.tight_layout()
    return fig


def _err(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _control_role(results: Results, t: float) -> str | None:
    """``'FD'`` or ``'UN'`` if ``t`` is the fully deuterated or undeuterated control time (matched
    within the controls' ``atol``, so the controls may be at any time, as in an experiment that
    starts deuterated and exchanges into H2O), else None."""
    ctrl = results.controls
    if ctrl is None:
        return None
    cfg = ctrl.config
    if cfg.fd_time is not None and abs(t - cfg.fd_time) <= cfg.atol:
        return "FD"
    if cfg.undeut_time is not None and abs(t - cfg.undeut_time) <= cfg.atol:
        return "UN"
    return None


def _time_text(results: Results, t: float) -> str:
    """A time for a title: ``FD`` or ``UN`` for a control, else seconds."""
    return _control_role(results, t) or f"{t:g} s"


def _control_row(results: Results, data_id: int) -> int | None:
    ctrl = results.controls
    if ctrl is None or data_id not in set(ctrl.data_id.tolist()):
        return None
    return int(ctrl.rows([data_id])[0])


def population_data(
    results: Results,
    spectrum_id: str,
    *,
    k: int | None = None,
    corrected: bool | None = None,
    components: Sequence[int] | None = None,
) -> dict:
    """Populations of one spectrum for the population map: ``mean`` and ``fraction`` of every
    population by rank (``mean_se``, ``fraction_se``), the ``draws`` of the bootstrap as
    ``(n, K, 2)`` (mean, fraction; None if not kept), ``centroid`` (fraction-weighted mean),
    ``corrected`` (whether the means are back-exchange corrected), ``n_amides`` and ``shown``."""
    i = results.find(spectrum_id)
    k = int(results.effective_K[i] if k is None else k)
    kf = results.fit.fits[k]
    order = np.argsort(kf.means[i])
    mean, frac = kf.means[i][order], kf.fractions[i][order]
    boot = results.bootstrap_for(i, k)
    nan = np.full(k, np.nan)
    mean_se = boot.mean_se[i, :k][order] if boot is not None else nan
    frac_se = boot.frac_se[i, :k][order] if boot is not None else nan
    draws = None
    if boot is not None and boot.draws is not None:
        d = boot.draws
        ok = d.ok[i]
        a = d.A[i][ok][:, :k][:, order]
        dm = (d.n[i][ok][:, :k] * d.p[i][ok][:, :k])[:, order]
        df = a / np.maximum(a.sum(axis=1, keepdims=True), 1e-10)
        draws = np.stack([dm, df], axis=-1)
    use = results.controls is not None if corrected is None else corrected
    n_amides = None
    if use and results.controls is not None:
        row = _control_row(results, int(results.fit.data_id[i]))
        n_amides = float(results.controls.n_amides[row])
        c = correct_means(results.controls, [results.fit.data_id[i]], mean[None, :],
                          se=mean_se[None, :])
        mean, mean_se = c.mean[0], c.se[0]
        if draws is not None:
            draws = np.stack([
                correct_means(results.controls, [results.fit.data_id[i]] * len(draws),
                              draws[:, :, 0]).mean, draws[:, :, 1]], axis=-1)
    else:
        use = False
    shown = list(range(1, k + 1)) if components is None else sorted(int(c) for c in components)
    return {
        "spectrum_id": spectrum_id, "index": i, "k": k, "mean": mean, "fraction": frac,
        "mean_se": mean_se, "fraction_se": frac_se, "draws": draws,
        "centroid": float((mean * frac).sum()), "corrected": use, "n_amides": n_amides,
        "shown": shown,
    }


def _limits(ax: Axes, n_amides: float | None) -> None:
    """The dashed guides at 0 and at the number of amides (once per axes) and the x range."""
    if n_amides is None:
        return
    if not any(line.get_gid() == "limits" for line in ax.collections):
        ax.vlines([0, n_amides], 0, 1, color="grey", ls=":", lw=1, gid="limits")
    ax.set_xlim(-1, n_amides + 1)


def _draw_population_map(ax: Axes, d: dict, title: str | None) -> None:
    for rank in d["shown"]:
        c = rank - 1
        style = population_style(rank)
        color = style["color"]
        if d["draws"] is not None:
            ax.scatter(d["draws"][:, c, 0], d["draws"][:, c, 1], s=8, color=color, alpha=0.3,
                       lw=0, gid=f"draws:{rank}")
        ax.errorbar(d["mean"][c], d["fraction"][c], xerr=_err(d["mean_se"][c]),
                    yerr=_err(d["fraction_se"][c]),
                    color=color, marker="o", ms=5, lw=1.5, fillstyle=style["fillstyle"],
                    label=f"pop {rank}", gid=f"pop:{rank}")
    ax.scatter([d["centroid"]], [1.0], marker="*", s=90, color="darkorange", zorder=1,
               label="centroid", gid="centroid")
    _limits(ax, d["n_amides"])
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("deuterium (back-exchange corrected)" if d["corrected"] else "deuterium")
    ax.set_ylabel("population")
    if title:
        ax.set_title(title, fontsize=8)


def plot_populations(
    results: Results,
    data_id: int,
    *,
    k: int | None = None,
    corrected: bool | None = None,
    components: Sequence[int] | None = None,
    overlay: bool = True,
    reference: Reference | None = None,
    columns: int | None = None,
    cell: tuple[float, float] = (4.0, 3.0),
) -> Figure:
    """The population map of a series, laid out like :func:`plot_series_spectra`: each cell the
    fraction against the deuterium level of every population (bootstrap draws when kept, mean
    with standard errors, the centroid as a star), and, in the time by replicate layout with
    ``overlay``, a last column that superimposes the replicates. ``reference`` marks the known
    populations (open circles)."""
    info = series_info(results, data_id)
    nrows, ncols_cells, positions, by_rep = _layout(info, columns)
    reps = np.unique(info["reps"])
    ncols = ncols_cells + (1 if overlay and by_rep else 0)
    fig = Figure(figsize=(cell[0] * ncols, cell[1] * nrows))
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for pos, i in enumerate(info["rows"]):
        sid = str(results.fit.spectrum_id[i])
        r_idx = int(np.searchsorted(reps, info["reps"][pos]))
        t_idx = positions[pos][0]
        d = population_data(results, sid, k=k, corrected=corrected, components=components)
        ax = axes[positions[pos]]
        ax.set_visible(True)
        label = f"{_time_text(results, info['times'][pos])}, rep {info['reps'][pos]}"
        _draw_population_map(ax, d, label)
        if reference is not None and d["n_amides"] is not None:
            known = reference.at(float(info["times"][pos]), d["n_amides"])
            if known is not None:
                ax.scatter(known[0], known[1], s=70, facecolors="none", edgecolors="darkred",
                           lw=1.5, zorder=6, gid="reference")
        if ncols > ncols_cells:
            ov = axes[t_idx, -1]
            ov.set_visible(True)
            for rank in d["shown"]:
                c = rank - 1
                # the colour is the population's (as in the single cells), the shape the replicate
                ov.errorbar(d["mean"][c], d["fraction"][c], xerr=_err(d["mean_se"][c]),
                            yerr=_err(d["fraction_se"][c]), marker=REP_MARKERS[r_idx % 6], ms=5,
                            lw=1, color=population_style(rank)["color"],
                            fillstyle=population_style(rank)["fillstyle"],
                            gid=f"overlay:{info['reps'][pos]}")
            ov.set_ylim(-0.05, 1.1)
            ov.set_title(f"{_time_text(results, info['times'][pos])}, replicates", fontsize=8)
            _limits(ov, d["n_amides"])
            if reference is not None and d["n_amides"] is not None:
                known = reference.at(float(info["times"][pos]), d["n_amides"])
                drawn = any(c.get_gid() == "reference" for c in ov.collections)
                if known is not None and not drawn:
                    ov.scatter(known[0], known[1], s=70, facecolors="none", edgecolors="darkred",
                               lw=1.5, zorder=6, gid="reference")
    fig.suptitle(info["label"], fontsize=10)
    fig.tight_layout()
    return fig


def uptake_data(
    results: Results,
    data_id: int,
    *,
    corrected: bool | None = None,
    components: Sequence[int] | None = None,
    k: int | None = None,
) -> dict:
    """The populations of every spectrum of a series along the time course, one row per
    spectrum and population: ``index`` (position in the results), ``time``, ``time_index``,
    ``rep``, ``rank``, ``mean`` (corrected if asked and possible), ``mean_se``, ``fraction``,
    ``fraction_se``, plus ``corrected``, ``n_amides``, ``m0`` and ``mfd`` (None without
    controls), ``role`` of every row (``'FD'``, ``'UN'`` or ``''``: whether the spectrum is a
    control, matched to the control times) and ``fd_time`` (the fully deuterated time of the
    controls, or None)."""
    info = series_info(results, data_id)
    cols: dict[str, list] = {n: [] for n in (
        "index", "time", "time_index", "rep", "rank", "mean", "mean_se", "fraction",
        "fraction_se")}
    use = False
    for pos, i in enumerate(info["rows"]):
        d = population_data(results, str(results.fit.spectrum_id[i]), k=k, corrected=corrected,
                            components=components)
        use = d["corrected"]
        for rank in d["shown"]:
            c = rank - 1
            for name, value in (
                ("index", i), ("time", info["times"][pos]), ("time_index", info["time_index"][pos]),
                ("rep", info["reps"][pos]), ("rank", rank), ("mean", d["mean"][c]),
                ("mean_se", d["mean_se"][c]), ("fraction", d["fraction"][c]),
                ("fraction_se", d["fraction_se"][c]),
            ):
                cols[name].append(value)
    out: dict = {n: np.array(v) for n, v in cols.items()}
    ctrl = results.controls
    row = _control_row(results, data_id)
    out.update({
        "label": info["label"], "corrected": use, "n_amides": info["n_amides"],
        "m0": None if row is None else float(ctrl.m0[row]),
        "mfd": None if row is None else float(ctrl.mfd[row]),
        "fd_time": None if ctrl is None else ctrl.config.fd_time,
        "role": np.array([_control_role(results, float(t)) or "" for t in out["time"]], dtype=str),
        "control_fallback": None if row is None else bool(ctrl.fallback[row]),
    })
    return out


def plot_uptake(
    results: Results,
    data_id: int,
    *,
    x: str = "time",
    scale: str = "linear",
    corrected: bool | None = None,
    components: Sequence[int] | None = None,
    k: int | None = None,
    reference: Reference | None = None,
    ax: Axes | None = None,
    legend: bool = True,
) -> Figure:
    """Deuterium uptake of every population along the time course (the markers have an area that
    grows with the population's fraction, error bars are the bootstrap standard errors of the
    mean, and each replicate has its own colour).

    ``x='time'`` plots against the measured time (``scale='linear'`` or ``'log'``; a log axis
    leaves out times of 0 or less, and the fully deuterated control, recognised by its time
    whether it comes first or last, is drawn as a line, not a
    point). ``x='index'`` plots against the position of the time point in the series (0, 1, 2,
    ..., labelled with the times), with every time point including the controls, as for the
    test data sets. ``corrected`` (default: if the results have controls) uses the back-exchange
    corrected means, and the limits drawn are then 0 and the number of amides; otherwise the
    measured control levels. ``reference`` marks the known populations (open circles).
    """
    if x not in ("time", "index"):
        raise ValueError("x must be 'time' or 'index'")
    if scale not in ("linear", "log"):
        raise ValueError("scale must be 'linear' or 'log'")
    d = uptake_data(results, data_id, corrected=corrected, components=components, k=k)
    fig = ax.figure if ax is not None else Figure(figsize=(6.5, 4.5))
    ax = ax if ax is not None else fig.subplots()
    is_fd = d["role"] == "FD"
    keep = np.ones(len(d["time"]), dtype=bool) if x == "index" else ~is_fd
    if x == "time" and scale == "log":
        keep &= d["time"] > 0
    xs = d["time_index"] if x == "index" else d["time"]
    reps = np.unique(d["rep"])
    for r in reps:
        sel = keep & (d["rep"] == r)
        color = REP_COLORS[(int(r) - 1) % len(REP_COLORS)]
        ax.errorbar(xs[sel], d["mean"][sel],
                    yerr=np.where(np.isfinite(d["mean_se"][sel]), d["mean_se"][sel], 0.0),
                    fmt="none", ecolor=color, elinewidth=1.2, capsize=0, gid=f"errors:rep{r}")
        ax.scatter(xs[sel], d["mean"][sel], s=20 + 160 * np.nan_to_num(d["fraction"][sel]),
                   color=color, marker="v", alpha=0.9, label=f"rep {r}", gid=f"rep:{r}")
    if reference is not None and d["n_amides"] is not None:
        for t in np.unique(d["time"][keep]):
            known = reference.at(float(t), d["n_amides"])
            if known is not None:
                pos = d["time_index"][d["time"] == t][0] if x == "index" else t
                ax.scatter([pos] * len(known[0]), known[0], s=20 + 160 * known[1],
                           facecolors="none", edgecolors="darkred", lw=1.5, zorder=6,
                           gid="reference")
    lo, hi = (0.0, d["n_amides"]) if d["corrected"] else (d["m0"], d["mfd"])
    for level in (lo, hi):
        if level is not None:
            ax.axhline(level, color="grey", ls=":", lw=1, alpha=0.7, gid="limits")
    if x == "index":
        ticks = np.unique(d["time_index"])
        labels = []
        for t in ticks:
            time = d["time"][d["time_index"] == t][0]
            role = d["role"][d["time_index"] == t][0]
            labels.append(role or f"{time:g}")
        ax.set_xticks(ticks, labels, rotation=45 if len(ticks) > 8 else 0, fontsize=8)
        ax.set_xlabel("time point (s)")
    else:
        ax.set_xlabel("time (s)")
        ax.set_xscale(scale)
    ax.set_ylabel("deuterium (back-exchange corrected)" if d["corrected"] else "deuterium")
    ax.set_title(d["label"], fontsize=9)
    if legend and len(reps) > 1:
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    try:
        fig.tight_layout()
    except Exception:  # noqa: BLE001 - layout is cosmetic
        pass
    return fig


__all__ = [
    "Reference", "load_reference", "plot_populations", "plot_series_spectra", "plot_uptake",
    "population_data", "series_info", "series_rows", "uptake_data",
]
