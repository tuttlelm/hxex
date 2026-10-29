# SPDX-License-Identifier: GPL-3.0-or-later
"""Plots of one series (a peptide at one charge): spectra by time and replicate, the populations
of every spectrum, and the uptake along the time course.

Each plot has a ``*_data`` function that returns the arrays it draws; the plot only draws them.
Populations are identified by the rank of their mean within each spectrum (1 = lowest), as
chosen for every time point independently.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..controls import correct_means
from ..io.times import format_time
from ..spectra import FULLDEUT, UNDEUT, time_index
from .results import Results
from .spectrum import LAYERS, plot_spectrum, population_style

log = logging.getLogger(__name__)

# replicate shapes in the overlay of the population map (there the colour is the population)
REP_MARKERS = ("o", "s", "^", "D", "v", "P")

# replicates (uptake plot): Okabe-Ito colours, the first two differing most
REP_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")


@dataclass(frozen=True)
class Reference:
    """Known populations of a test data set, to draw next to the fitted ones: for each time, the
    known means as fractional deuteration (per exchangeable amide) and the known populations."""

    times: np.ndarray
    level: np.ndarray  # (T, P) fractional deuteration, NaN where the component is absent
    population: np.ndarray  # (T, P)

    def at(self, time: float, n_amides: float, atol: float = 1e-6):
        """(means in D, populations) of the components known at ``time``, or None."""
        hit = np.flatnonzero(np.abs(self.times - time) <= atol)
        if not len(hit):
            return None
        lv, fr = self.level[hit[0]], self.population[hit[0]]
        keep = np.isfinite(lv) & np.isfinite(fr) & (fr > 0)
        return lv[keep] * n_amides, fr[keep]


def load_reference(path) -> Reference:
    """Read a tab-separated table of known populations: ``time``, ``fd1``..., ``p1``... (the
    layout of the Bimodal test set; ``fd`` in percent of the amides, ``#N/A`` for none)."""
    from pathlib import Path

    rows = [r.split("\t") for r in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if r.strip()]
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
    fit = results.fit
    rank = time_index(fit.data_id[rows], fit.time[rows], fit.role[rows])
    return rows[np.lexsort((np.arange(len(rows)), rep, rank))]


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


def _fragment_of(results: Results, data_id: int) -> dict | None:
    """The fragment ion of a series read from scan files (its ``fragment`` label such as
    ``c-1:6``, the ``parent_range``, the ``parent_charge`` and the ``condition``), or None for
    a peptide or other data."""
    from ..io.mzml import SCAN_READERS, MzmlTarget

    source = results.settings.spectra_source if results.settings is not None else None
    if not source or source.get("reader") not in SCAN_READERS or "series" not in source:
        return None
    for s in source["series"]:
        if int(s["data_id"]) == int(data_id):
            target = MzmlTarget.from_dict(source["targets"][int(s["target"])])
            extra = dict(target.entry.extra)
            if "fragment" not in extra:
                return None
            return {"fragment": str(extra["fragment"]),
                    "parent_range": str(extra.get("parent_range", "")),
                    "parent_charge": int(extra.get("parent_charge", 0) or 0),
                    "condition": str(extra.get("condition") or target.activation or "")}
    return None


def series_info(results: Results, data_id: int) -> dict:
    """Description of a series: ``rows``, ``times``, ``reps``, ``time_index`` of each spectrum,
    ``label`` (sample, range, peptide, charge when the picked table is there; for a fragment
    ion its range and sequence, the fragment label such as ``c-1:6`` with its charge, the
    parent's range and charge and the condition), ``file_label`` (the label for file names:
    a fragment's without the sequence, so that the fragment is kept when names are
    shortened) and ``n_amides`` (from the controls, else None)."""
    rows = series_rows(results, data_id)
    fit = results.fit
    label = file_label = f"series {data_id}"
    if results.binned is not None:
        j = results.binned.find(str(fit.spectrum_id[rows[0]]))
        b = results.binned
        z = int(b.charge[j])
        label = file_label = f"{b.sample[j]} {b.peptide_range[j]} {b.peptide[j]} z={z}".strip()
        frag = _fragment_of(results, data_id)
        if frag is not None:
            pz, cond = frag["parent_charge"], frag["condition"]
            label = (f"{b.sample[j]} {b.peptide_range[j]} {b.peptide[j]} {frag['fragment']} "
                     f"z={z}, parent {frag['parent_range']} z={pz}"
                     + (f", {cond}" if cond else "")).strip()
            file_label = (f"{b.sample[j]} {b.peptide_range[j]} {frag['fragment']} z{z} "
                          f"parent z{pz} {cond}").strip()
    row = _control_row(results, data_id)
    n_amides = None if row is None else float(results.controls.n_amides[row])
    return {
        "data_id": data_id, "rows": rows, "label": label, "file_label": file_label,
        "n_amides": n_amides,
        "times": fit.time[rows], "roles": fit.role[rows], "reps": _reps(results)[rows],
        "time_index": time_index(fit.data_id[rows], fit.time[rows], fit.role[rows]),
    }


def _layout(info: dict, columns: int | None) -> tuple[int, int, list[tuple[int, int]], bool]:
    """Rows, columns and the (row, column) of every spectrum of a series (in ``info`` order).

    With several replicates (and ``columns`` not given) a row is a time and a column a replicate,
    unless two spectra share a time and a replicate number (warned). Otherwise the spectra, in
    time order, fill a grid of ``columns`` columns (default: a near square of at most 4
    columns), so that a long single-replicate time course is not one tall column. The last
    value says whether the layout is time by replicate."""
    n = len(info["rows"])
    # a point is a time or a control role; ``time_index`` ranks them (the controls at the ends)
    points, reps = np.unique(info["time_index"]), np.unique(info["reps"])
    pairs = list(zip(info["time_index"].tolist(), info["reps"].tolist(), strict=True))
    if columns is None and len(reps) > 1 and len(set(pairs)) < len(pairs):
        log.warning("%s: two spectra share a time and a replicate number; drawn in time "
                    "order, not by replicate", info.get("label", "a series"))
    elif columns is None and len(reps) > 1:
        pos = [(int(np.searchsorted(points, t)), int(np.searchsorted(reps, r)))
               for t, r in zip(info["time_index"], info["reps"], strict=True)]
        return len(points), len(reps), pos, True
    if columns is not None and columns < 1:
        raise ValueError("columns must be at least 1")
    ncols = columns or max(1, min(4, math.ceil(math.sqrt(n))))
    ncols = min(ncols, n)
    return math.ceil(n / ncols), ncols, [(i // ncols, i % ncols) for i in range(n)], False


def _grid(info: dict, size: tuple[float, float], columns: int | None, residual: bool = False):
    """The figure and, for each spectrum, its axes (and its residual axes below it, or None)."""
    nrows, ncols, positions, _ = _layout(info, columns)
    height = size[1] * (1.3 if residual else 1.0)
    fig = Figure(figsize=(size[0] * ncols, height * nrows))
    grid = fig.add_gridspec(nrows, ncols)
    cells = {}
    for i, (r, c) in enumerate(positions):
        if residual:
            sub = grid[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
            ax = fig.add_subplot(sub[0])
            cells[int(info["rows"][i])] = (ax, fig.add_subplot(sub[1], sharex=ax))
        else:
            cells[int(info["rows"][i])] = (fig.add_subplot(grid[r, c]), None)
    return fig, cells


def _series_range(results: Results, spectrum_ids: Sequence[str], axes) -> tuple[float, float]:
    """The expected m/z range of a series' spectra: the lowest and highest predicted m/z of
    their picked bins, or without a picked table the union of the ``axes``' current ranges."""
    lo, hi = np.inf, -np.inf
    binned = results.binned
    if binned is not None:
        for sid in spectrum_ids:
            try:
                mz = np.asarray(binned.spectrum(binned.find(sid))[0], dtype=float)
            except KeyError:
                continue
            mz = mz[np.isfinite(mz)]
            if len(mz):
                lo, hi = min(lo, float(mz.min())), max(hi, float(mz.max()))
    if not np.isfinite(lo):
        for ax in axes:
            if ax.has_data():
                a, b = ax.get_xlim()
                lo, hi = min(lo, a), max(hi, b)
    return lo, hi


def plot_series_spectra(
    results: Results,
    data_id: int,
    *,
    show: Sequence[str] = LAYERS,
    components: Sequence[int] | None = None,
    k: int | None = None,
    columns: int | None = None,
    residual: bool = False,
    cell: tuple[float, float] = (4.5, 3.0),
    padding: float = 0.1,
    **kwargs,
) -> Figure:
    """A grid of spectra: with replicates a row per time and a column per replicate, otherwise
    the time course wrapped into ``columns`` columns (see :func:`_layout`; see
    :func:`plot_spectrum` for the keyword arguments, which are passed on). ``residual`` puts the
    residuals of each spectrum (picked minus fit) in a small panel under it; it needs the picked
    and fit layers. Spectra whose layers cannot be drawn are left blank. Every panel has the
    same m/z range: the series' expected range (the predicted m/z of the picked bins of all its
    spectra; without a picked table, the panels' own ranges together) with ``padding`` times its
    width on each side."""
    info = series_info(results, data_id)
    fig, cells = _grid(info, cell, columns, residual)
    unread = None
    if "raw" in show and results.spectra is None and results.loader is not None:
        # the series' raw spectra in one read, not one read per panel
        ids = [str(results.fit.spectrum_id[i]) for i in cells]
        try:
            results = dataclasses.replace(results, spectra=results.loader(ids))
        except Exception as exc:  # noqa: BLE001 (the other layers can still be drawn)
            log.warning("the raw spectra of series %d could not be read: %s", data_id, exc)
            unread = f"raw spectra not read: {exc}"
            results = dataclasses.replace(results, loader=None)
    for i, (ax, rax) in cells.items():
        sid = str(results.fit.spectrum_id[i])
        try:
            plot_spectrum(sid, results, show=show, k=k, components=components, ax=ax, rax=rax,
                          residual=residual, legend=False, **kwargs)
            if rax is not None:
                ax.tick_params(labelbottom=False)
                ax.set_xlabel("")
        except ValueError as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", fontsize=6, wrap=True,
                    transform=ax.transAxes)
        if unread is not None:
            ax.text(0.02, 0.98, unread, ha="left", va="top", fontsize=6, wrap=True,
                    color="darkred", transform=ax.transAxes, gid="raw_unread")
        pos = list(info["rows"]).index(i)
        when = _time_text(info["times"][pos], info["roles"][pos], results.time_scale)
        ax.set_title(f"{when}, rep {info['reps'][pos]}", fontsize=8)
    lo, hi = _series_range(results, [str(results.fit.spectrum_id[i]) for i in cells],
                           [ax for ax, _ in cells.values()])
    if np.isfinite(lo) and hi > lo:
        pad = padding * (hi - lo)
        for ax, _ in cells.values():  # the residual panels share their spectrum's x axis
            ax.set_xlim(lo - pad, hi + pad)
    # the series title in a band at the top, placed in inches (a fraction of the height would
    # drop it into the first row of a tall figure; the layout does not count it in nested grids)
    height = fig.get_figheight()
    fig.suptitle(info["label"], fontsize=10, y=1 - 0.1 / height, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / height))
    return fig


def _err(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


ROLE_SHORT = {UNDEUT: "UN", FULLDEUT: "FD"}


def _control_role(role: str) -> str | None:
    """``'FD'`` or ``'UN'`` for a spectrum with the fully deuterated or undeuterated role (the
    controls are known by their role, so they may be at any time, as in an experiment that
    starts deuterated and exchanges into H2O), else None."""
    return ROLE_SHORT.get(str(role))


def _time_text(t: float, role: str = "", scale: str = "seconds") -> str:
    """A point for a title: ``FD`` or ``UN`` for a control (with its time when it has one),
    else the time in a readable unit (:func:`hxex.io.format_time`); ``point 3`` when the times
    only order the points (``scale='order'``, :attr:`Results.time_scale`)."""
    when = format_time(t, scale)
    if scale == "order" and when:
        when = f"point {when}"
    short = _control_role(role)
    if short is not None:
        return f"{short} ({when})" if np.isfinite(t) else short
    return when


def _control_row(results: Results, data_id: int) -> int | None:
    """The series' row of the controls table; None without controls and for a series that is
    not corrected (fragment ions)."""
    ctrl = results.controls
    row = None if ctrl is None else ctrl.row_of(data_id)
    return None if row is None or ctrl.uncorrected[row] else row


def _uncorrected(results: Results, data_id: int) -> bool:
    """Whether the series has a controls row that leaves it uncorrected (fragment ions)."""
    ctrl = results.controls
    row = None if ctrl is None else ctrl.row_of(data_id)
    return row is not None and bool(ctrl.uncorrected[row])


def population_data(
    results: Results,
    spectrum_id: str,
    *,
    k: int | None = None,
    corrected: bool | None = None,
    components: Sequence[int] | None = None,
) -> dict:
    """Populations of one spectrum for the population map: ``mean`` and ``population`` of every
    population by rank (``mean_se``, ``population_se``), the ``draws`` of the bootstrap as
    ``(n, K, 2)`` (mean, population; None if not kept), ``centroid`` (population-weighted mean),
    ``corrected`` (whether the means are back-exchange corrected), ``n_amides`` and ``shown``.
    ``corrected`` None (the default) corrects when the results have controls for the spectrum's
    series; True raises KeyError when they have none for it, and draws a series the controls
    leave uncorrected (fragment ions) uncorrected, with ``uncorrected_series`` True so that the
    panel says so."""
    i = results.find(spectrum_id)
    k = int(results.effective_K[i] if k is None else k)
    results.fit.require_fitted(i, k)
    kf = results.fit.fits[k]
    order = np.argsort(kf.means[i])
    mean, frac = kf.means[i][order], kf.populations[i][order]
    boot = results.bootstrap_for(i, k)
    nan = np.full(k, np.nan)
    mean_se = boot.mean_se[i, :k][order] if boot is not None else nan
    population_se = boot.population_se[i, :k][order] if boot is not None else nan
    draws = None
    if boot is not None and boot.draws is not None:
        d = boot.draws
        ok = d.ok[i]
        a = d.A[i][ok][:, :k][:, order]
        dm = (d.n[i][ok][:, :k] * d.p[i][ok][:, :k])[:, order]
        df = a / np.maximum(a.sum(axis=1, keepdims=True), 1e-10)
        draws = np.stack([dm, df], axis=-1)
    data_id = int(results.fit.data_id[i])
    row = _control_row(results, data_id)
    if corrected and row is None and not _uncorrected(results, data_id):
        raise KeyError(f"no controls for series {data_id} (spectrum {spectrum_id})")
    use = row is not None and (corrected is None or bool(corrected))
    n_amides = None
    if use:
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
        "spectrum_id": spectrum_id, "index": i, "k": k, "mean": mean, "population": frac,
        "mean_se": mean_se, "population_se": population_se, "draws": draws,
        "centroid": float((mean * frac).sum()), "corrected": use, "n_amides": n_amides,
        "shown": shown, "uncorrected_series": bool(corrected) and row is None,
    }


def _limits(ax: Axes, n_amides: float | None) -> None:
    """The dashed guides at 0 and at the number of amides (once per axes) and the x range."""
    if n_amides is None:
        return
    if not any(line.get_gid() == "limits" for line in ax.collections):
        ax.vlines([0, n_amides], 0, 1, color="grey", ls=":", lw=1, gid="limits")
    ax.set_xlim(-1, n_amides + 1)


def _deuterium_label(corrected: bool, uncorrected_series: bool) -> str:
    """The deuterium axis label; a series drawn uncorrected although corrected values were
    asked for (one the controls leave uncorrected, such as fragment ions) says so."""
    if corrected:
        return "deuterium (back-exchange corrected)"
    return "deuterium (uncorrected series)" if uncorrected_series else "deuterium"


def _draw_population_map(ax: Axes, d: dict, title: str | None) -> None:
    for rank in d["shown"]:
        c = rank - 1
        style = population_style(rank)
        color = style["color"]
        if d["draws"] is not None:
            ax.scatter(d["draws"][:, c, 0], d["draws"][:, c, 1], s=8, color=color, alpha=0.3,
                       lw=0, gid=f"draws:{rank}")
        ax.errorbar(d["mean"][c], d["population"][c], xerr=_err(d["mean_se"][c]),
                    yerr=_err(d["population_se"][c]),
                    color=color, marker="o", ms=5, lw=1.5, fillstyle=style["fillstyle"],
                    label=f"pop {rank}", gid=f"pop:{rank}")
    ax.scatter([d["centroid"]], [1.0], marker="*", s=90, color="darkorange", zorder=1,
               label="centroid", gid="centroid")
    _limits(ax, d["n_amides"])
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel(_deuterium_label(d["corrected"], d["uncorrected_series"]))
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
    population against the deuterium level of every component (bootstrap draws when kept, mean
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
        if k is not None and not results.fit.fitted(k)[i]:
            continue  # the screen skipped this K for the spectrum: its panel stays empty
        d = population_data(results, sid, k=k, corrected=corrected, components=components)
        ax = axes[positions[pos]]
        ax.set_visible(True)
        rejected = results.review_status[i] == "rejected"
        when = _time_text(info["times"][pos], info["roles"][pos], results.time_scale)
        label = f"{when}, rep {info['reps'][pos]}" + (" (rejected)" if rejected else "")
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
                # the colour is the population's (as in the single cells), the shape the replicate;
                # a rejected or excluded spectrum hollow and grey
                style = ({"color": "grey", "fillstyle": "none", "alpha": 0.6} if rejected else
                         {"color": population_style(rank)["color"],
                          "fillstyle": population_style(rank)["fillstyle"]})
                ov.errorbar(d["mean"][c], d["population"][c], xerr=_err(d["mean_se"][c]),
                            yerr=_err(d["population_se"][c]), marker=REP_MARKERS[r_idx % 6], ms=5,
                            lw=1, gid=f"overlay:{info['reps'][pos]}"
                            + (":rejected" if rejected else ""), **style)
            ov.set_ylim(-0.05, 1.1)
            when = _time_text(info["times"][pos], info["roles"][pos], results.time_scale)
            ov.set_title(f"{when}, replicates", fontsize=8)
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
    ``rep``, ``rank``, ``mean`` (corrected if asked and possible), ``mean_se``, ``population``,
    ``population_se``, plus ``corrected``, ``n_amides``, ``m0`` and ``mfd`` (None without
    controls) and ``role`` of every row (``'FD'``, ``'UN'`` or ``''``: whether the spectrum is a
    control, by its role; a control's ``time`` is NaN unless it was also a time point), and
    ``rejected`` (the review rejected the spectrum, or an [[review.exclude]] rule excluded
    it: drawn hollow and grey)."""
    info = series_info(results, data_id)
    cols: dict[str, list] = {n: [] for n in (
        "index", "time", "time_index", "rep", "rank", "mean", "mean_se", "population",
        "population_se", "role")}
    use, uncorrected_series = False, False
    for pos, i in enumerate(info["rows"]):
        if k is not None and not results.fit.fitted(k)[i]:
            continue  # the screen skipped this K for the spectrum
        d = population_data(results, str(results.fit.spectrum_id[i]), k=k, corrected=corrected,
                            components=components)
        use, uncorrected_series = d["corrected"], d["uncorrected_series"]
        for rank in d["shown"]:
            c = rank - 1
            for name, value in (
                ("index", i), ("time", info["times"][pos]), ("time_index", info["time_index"][pos]),
                ("role", _control_role(info["roles"][pos]) or ""),
                ("rep", info["reps"][pos]), ("rank", rank), ("mean", d["mean"][c]),
                ("mean_se", d["mean_se"][c]), ("population", d["population"][c]),
                ("population_se", d["population_se"][c]),
            ):
                cols[name].append(value)
    out: dict = {n: np.array(v) for n, v in cols.items()}
    out["role"] = np.asarray(out["role"], dtype=str)
    status = results.review_status
    out["rejected"] = np.array([status[int(i)] == "rejected" for i in out["index"]], dtype=bool)
    ctrl = results.controls
    row = _control_row(results, data_id)
    out.update({
        "label": info["label"], "corrected": use, "uncorrected_series": uncorrected_series,
        "n_amides": info["n_amides"],
        "m0": None if row is None else float(ctrl.m0[row]),
        "mfd": None if row is None else float(ctrl.mfd[row]),
        "control_fallback": None if row is None else bool(ctrl.fallback[row]),
    })
    return out


def plot_uptake(
    results: Results,
    data_id: int,
    *,
    x: str | None = None,
    scale: str = "linear",
    corrected: bool | None = None,
    components: Sequence[int] | None = None,
    k: int | None = None,
    reference: Reference | None = None,
    ax: Axes | None = None,
    legend: bool = True,
) -> Figure:
    """Deuterium uptake of every population along the time course (the markers have an area that
    grows with the population, error bars are the bootstrap standard errors of the
    mean, and each replicate has its own colour).

    ``x='time'`` plots against the measured time (``scale='linear'`` or ``'log'``; a log axis
    leaves out times of 0 or less; the controls, known by their role, are drawn as lines, not
    points, unless they are time points as well). ``x='index'`` plots against the position of
    the point in the series (0, 1, 2, ..., labelled with the times), with every point including
    the controls (the undeuterated first, the fully deuterated last), as for the test data
    sets. ``x=None`` (the default) is ``'time'``, or ``'index'`` when the times only order the
    points (:attr:`Results.time_scale` ``'order'``: the labels are then the numbers given, and
    ``x='time'`` is refused). ``corrected`` (default: if the results have controls) uses the
    back-exchange corrected means, and the limits drawn are then 0 and the number of amides;
    otherwise the measured control levels. ``reference`` marks the known populations (open
    circles; in the measured units, ``m0 + level * (mfd - m0)``, when the plot is not
    corrected).
    """
    if x not in ("time", "index", None):
        raise ValueError("x must be 'time' or 'index'")
    if scale not in ("linear", "log"):
        raise ValueError("scale must be 'linear' or 'log'")
    time_scale = results.time_scale
    if x is None:
        x = "index" if time_scale == "order" else "time"
    elif x == "time" and time_scale == "order":
        raise ValueError("the times of these results only order the points ([input] "
                         "time_scale = \"order\"): the uptake is drawn by index (x='index')")
    d = uptake_data(results, data_id, corrected=corrected, components=components, k=k)
    fig = ax.figure if ax is not None else Figure(figsize=(6.5, 4.5))
    ax = ax if ax is not None else fig.subplots()
    keep = np.ones(len(d["time"]), dtype=bool) if x == "index" else np.isfinite(d["time"])
    if x == "time" and scale == "log":
        keep &= d["time"] > 0
    xs = d["time_index"] if x == "index" else d["time"]
    reps = np.unique(d["rep"])
    rejected = d["rejected"]
    for r in reps:
        sel = keep & (d["rep"] == r) & ~rejected
        color = REP_COLORS[(int(r) - 1) % len(REP_COLORS)]
        ax.errorbar(xs[sel], d["mean"][sel],
                    yerr=np.where(np.isfinite(d["mean_se"][sel]), d["mean_se"][sel], 0.0),
                    fmt="none", ecolor=color, elinewidth=1.2, capsize=0, gid=f"errors:rep{r}")
        ax.scatter(xs[sel], d["mean"][sel], s=20 + 160 * np.nan_to_num(d["population"][sel]),
                   color=color, marker="v", alpha=0.9, label=f"rep {r}", gid=f"rep:{r}")
    out = keep & rejected  # rejected or excluded: visible, but hollow and grey
    if out.any():
        ax.errorbar(xs[out], d["mean"][out],
                    yerr=np.where(np.isfinite(d["mean_se"][out]), d["mean_se"][out], 0.0),
                    fmt="none", ecolor="grey", elinewidth=1.0, alpha=0.6, capsize=0,
                    gid="errors:rejected")
        ax.scatter(xs[out], d["mean"][out], s=20 + 160 * np.nan_to_num(d["population"][out]),
                   facecolors="none", edgecolors="grey", marker="v", lw=1.2,
                   label="rejected / excluded", gid="rejected")
    if reference is not None and d["n_amides"] is not None:
        for t in np.unique(d["time"][keep]):
            known = reference.at(float(t), d["n_amides"])
            if known is not None:
                pos = d["time_index"][d["time"] == t][0] if x == "index" else t
                ys = known[0]  # corrected units; else back to the measured scale
                if not d["corrected"]:
                    ys = d["m0"] + ys / d["n_amides"] * (d["mfd"] - d["m0"])
                ax.scatter([pos] * len(ys), ys, s=20 + 160 * known[1],
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
            labels.append(role or format_time(time, time_scale))
        ax.set_xticks(ticks, labels, rotation=45 if len(ticks) > 8 else 0, fontsize=8)
        ax.set_xlabel("time point")
    else:
        ax.set_xlabel("time (s)")
        ax.set_xscale(scale)
    ax.set_ylabel(_deuterium_label(d["corrected"], d["uncorrected_series"]))
    ax.set_title(d["label"], fontsize=9)
    if legend and (len(reps) > 1 or out.any()):
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
