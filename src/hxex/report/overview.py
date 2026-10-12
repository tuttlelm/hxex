# SPDX-License-Identifier: GPL-3.0-or-later
"""The K comparison of one spectrum, and overviews of a whole run: controls, quality flags and
the selected K. As elsewhere in :mod:`hxex.report`, every plot has a ``*_data`` function that
returns what it draws, and the plots return figures without showing or saving them."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from ..quality import FLAG_NAMES
from .results import Results
from .series import _control_role
from .spectrum import COLORS, FIT_COLOR, plot_spectrum
from .tables import _index_columns

FALLBACK_REASONS = ("m0_missing", "m0_high", "fd_missing", "retention_low", "retention_high",
                    "no_amides")
QUALITY_FLAGS = FLAG_NAMES
GOOD = "#999999"  # unflagged points and measured controls


# --- K comparison of one spectrum -----------------------------------------------------------

def k_comparison_data(spectrum_id: str, results: Results) -> dict:
    """The fit of one spectrum at every K tried: ``k_values``, ``nll``, ``bic``, ``aic``, the
    selection ``score`` (the configured criterion, inf where the minimum-population rule rejected
    that K), ``delta`` (score minus the best score), ``passes`` (minimum population),
    ``criterion``, ``selected`` and ``effective`` (the reviewer's K, else the selected one)."""
    i = results.find(spectrum_id)
    fit = results.fit
    ks = tuple(fit.k_values)
    score = np.asarray(fit.score[i], dtype=float)
    finite = score[np.isfinite(score)]
    return {
        "spectrum_id": spectrum_id, "index": i, "k_values": ks,
        "nll": np.array([fit.fits[k].nll[i] for k in ks]),
        "bic": np.array([fit.fits[k].bic[i] for k in ks]),
        "aic": np.array([fit.fits[k].aic[i] for k in ks]),
        "score": score, "delta": score - (finite.min() if len(finite) else np.nan),
        "passes": np.array([bool(fit.fits[k].passes_min_pop[i]) for k in ks]),
        "criterion": fit.config.criterion, "selected": int(fit.selected_K[i]),
        "effective": int(results.effective_K[i]),
    }


def plot_k_comparison(
    spectrum_id: str,
    results: Results,
    *,
    ks: Sequence[int] | None = None,
    show: Sequence[str] = ("picked", "fit"),
    cell: tuple[float, float] = (4.6, 3.6),
    **kwargs,
) -> Figure:
    """One spectrum fitted at each K side by side (default: every K tried), each panel with its
    loss, its selection score relative to the best and whether the minimum-population rule
    rejected it; the selected K is marked, and the reviewer's K if it differs. Keyword arguments
    go to :func:`plot_spectrum`."""
    d = k_comparison_data(spectrum_id, results)
    ks = list(d["k_values"] if ks is None else ks)
    bad = [k for k in ks if k not in d["k_values"]]
    if bad:
        raise ValueError(f"K = {bad} was not fitted (tried {d['k_values']})")
    fig = Figure(figsize=(cell[0] * len(ks), cell[1]))
    axes = np.atleast_1d(fig.subplots(1, len(ks), sharey=True))
    for ax, k in zip(axes, ks, strict=True):
        j = d["k_values"].index(k)
        plot_spectrum(spectrum_id, results, k=k, show=show, ax=ax, title=False, **kwargs)
        marks = []
        if k == d["selected"]:
            marks.append("selected")
        if k == d["effective"] and d["effective"] != d["selected"]:
            marks.append("reviewer's choice")
        if not d["passes"][j]:
            marks.append("population below minimum")
        delta = d["delta"][j]
        name = d["criterion"].upper()
        score_text = "rejected" if not np.isfinite(delta) else f"Δ{name} {delta:+.1f}"
        ax.set_title(f"K={k}   loss {d['nll'][j]:.4g}   {score_text}"
                     + (f"\n{', '.join(marks)}" if marks else ""), fontsize=8,
                     fontweight="bold" if k == d["effective"] else "normal")
        if k == d["effective"]:
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
    fig.suptitle(spectrum_id, fontsize=9)
    fig.tight_layout()
    return fig


# --- controls overview ----------------------------------------------------------------------

def controls_overview_data(results: Results) -> dict:
    """For every series with controls: ``retention`` of the measured fully deuterated control
    against the undeuterated level used (NaN without one), ``m0`` (the measured undeuterated
    mean, NaN without one), the counts of each fallback ``reasons`` and of series using both
    measured controls (``measured``), and the configured limits."""
    c = results.controls
    if c is None:
        raise ValueError("these results have no controls")
    has_fd = c.n_fd > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        retention = np.where(has_fd & (c.n_amides > 0),
                             (c.mfd_measured - c.m0) / c.n_amides, np.nan)
    reasons = {r: int(sum(r in str(f).split(";") for f in c.flags)) for r in FALLBACK_REASONS}
    cfg = c.config
    return {
        "data_id": c.data_id, "retention": retention, "m0": c.m0_measured,
        "fallback": c.fallback.copy(), "reasons": reasons, "measured": int((~c.fallback).sum()),
        "retention_min": cfg.retention_min, "retention_max": cfg.retention_max,
        "m0_max": cfg.m0_max, "n_series": len(c),
    }


def plot_controls_overview(results: Results) -> Figure:
    """The controls of a run: the distribution of the measured retention with the limits, the
    distribution of the undeuterated control means with ``m0_max``, and why series fell back to
    theoretical values."""
    d = controls_overview_data(results)
    fig = Figure(figsize=(14, 3.8))
    a1, a2, a3 = fig.subplots(1, 3)
    ret = d["retention"][np.isfinite(d["retention"])]
    if len(ret):
        a1.hist(ret, bins=40, color=GOOD, gid="retention")
    for x in (d["retention_min"], d["retention_max"]):
        a1.axvline(x, color=FIT_COLOR, ls="--", lw=1, gid="limits")
    median = f", median {np.median(ret):.2f}" if len(ret) else ""
    a1.set_title(f"retention of the measured controls ({len(ret)} series{median})", fontsize=9)
    a1.set_xlabel("(mFD − m0) / exchangeable amides")
    a1.set_ylabel("series")
    m0 = d["m0"][np.isfinite(d["m0"])]
    if len(m0):
        a2.hist(m0, bins=40, color=GOOD, gid="m0")
    a2.axvline(d["m0_max"], color=FIT_COLOR, ls="--", lw=1, gid="limits")
    a2.set_title(f"undeuterated control means ({len(m0)} series)", fontsize=9)
    a2.set_xlabel("m0 (deuterons beyond natural abundance)")
    names = ["both measured", *d["reasons"]]
    counts = [d["measured"], *d["reasons"].values()]
    colors = [GOOD] + [FIT_COLOR] * len(d["reasons"])
    a3.barh(names[::-1], counts[::-1], color=colors[::-1], gid="reasons")
    for y, n in enumerate(counts[::-1]):
        a3.text(n, y, f" {n}", va="center", fontsize=8)
    a3.set_title(f"control status of {d['n_series']} series "
                 f"({int(d['fallback'].sum())} fell back)", fontsize=9)
    a3.set_xlabel("series (a series can have several reasons)")
    fig.tight_layout()
    return fig


# --- quality overview -----------------------------------------------------------------------

def quality_overview_data(results: Results) -> dict:
    """The quality flags of a run: the count of each flag and of flagged spectra, the
    ``deviance_rel`` and ``intensity_rel`` of every spectrum with ``flagged``, the thresholds,
    and the share of flagged spectra per sample (``samples``, ``flagged_share``; one group
    ``''`` without the picked table)."""
    q = results.quality
    if q is None:
        raise ValueError("these results have no quality flags")
    sample = _index_columns(results)["sample"]
    samples = sorted(set(sample.tolist()))
    share = np.array([q.flagged[sample == s].mean() for s in samples])
    return {
        "counts": {n: int(getattr(q, n).sum()) for n in QUALITY_FLAGS},
        "n_flagged": int(q.flagged.sum()), "n": len(q),
        "deviance_rel": q.deviance_rel, "intensity_rel": q.intensity_rel, "flagged": q.flagged,
        "poor_fit_factor": q.config.poor_fit_factor, "min_intensity": q.config.min_intensity,
        "samples": samples, "flagged_share": share,
    }


def flagged_share_grid(x, y, flagged, bins: int = 40, min_count: int = 5) -> dict | None:
    """The share of flagged spectra on a grid of ``bins`` x ``bins`` cells, equal in log space,
    over the spectra where ``x`` and ``y`` are both positive and finite: ``x_edges``,
    ``y_edges``, ``count`` and ``share`` (``(bins, bins)``, NaN in empty cells) and the spectra
    of the cells with fewer than ``min_count`` (``sparse_x``, ``sparse_y``, ``sparse_flagged``;
    ``sparse_in_grid`` counts them; they are also in ``count``). None if no spectrum has both
    values."""
    x, y, f = np.asarray(x, float), np.asarray(y, float), np.asarray(flagged, bool)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if not ok.any():
        return None
    lx, ly, f = np.log10(x[ok]), np.log10(y[ok]), f[ok]
    pad = 1e-9
    x_edges = np.linspace(lx.min() - pad, lx.max() + pad, bins + 1)
    y_edges = np.linspace(ly.min() - pad, ly.max() + pad, bins + 1)
    count, _, _ = np.histogram2d(lx, ly, bins=[x_edges, y_edges])
    hits, _, _ = np.histogram2d(lx[f], ly[f], bins=[x_edges, y_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(count > 0, hits / count, np.nan)
    ix = np.clip(np.searchsorted(x_edges, lx, side="right") - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(y_edges, ly, side="right") - 1, 0, bins - 1)
    sparse = count[ix, iy] < min_count
    return {
        "x_edges": 10.0 ** x_edges, "y_edges": 10.0 ** y_edges, "count": count, "share": share,
        "sparse_x": x[ok][sparse], "sparse_y": y[ok][sparse], "sparse_flagged": f[sparse],
        "sparse_in_grid": int(sparse.sum()),
    }


def plot_quality_overview(results: Results, *, bins: int = 40, min_count: int = 5) -> Figure:
    """The quality flags of a run: how many spectra carry each flag; each spectrum's fit
    (deviance relative to its series) against its intensity (relative to its replicates), as the
    share of flagged spectra in each cell of a ``bins`` x ``bins`` grid (cells with fewer than
    ``min_count`` spectra show the spectra as points instead), with the thresholds; and the
    share of flagged spectra per sample."""
    d = quality_overview_data(results)
    fig = Figure(figsize=(14, 3.8))
    a1, a2, a3 = fig.subplots(1, 3)
    names = [*d["counts"], "any flag"]
    counts = [*d["counts"].values(), d["n_flagged"]]
    a1.barh(names[::-1], counts[::-1], color=FIT_COLOR, gid="counts")
    for y, n in enumerate(counts[::-1]):
        a1.text(n, y, f" {n} ({100 * n / max(d['n'], 1):.1f} %)", va="center", fontsize=8)
    a1.set_title(f"quality flags of {d['n']:,} spectra", fontsize=9)
    a1.set_xlabel("spectra")
    g = flagged_share_grid(d["intensity_rel"], d["deviance_rel"], d["flagged"], bins=bins,
                           min_count=min_count)
    if g is not None:
        share = np.ma.masked_where(g["count"] < min_count, 100 * g["share"])
        mesh = a2.pcolormesh(g["x_edges"], g["y_edges"], share.T, cmap="cividis", vmin=0,
                             vmax=100, gid="share")
        fig.colorbar(mesh, ax=a2, label=f"flagged (% of spectra in the cell, ≥{min_count})")
        # cells with few spectra: the spectra themselves, so isolated ones stay visible
        sx, sy, sf = g["sparse_x"], g["sparse_y"], g["sparse_flagged"]
        a2.scatter(sx[~sf], sy[~sf], s=6, color=GOOD, lw=0, gid="spectra")
        a2.scatter(sx[sf], sy[sf], s=14, facecolors="none", edgecolors=FIT_COLOR, lw=0.7,
                   gid="flagged")
    a2.axhline(d["poor_fit_factor"], color=FIT_COLOR, ls="--", lw=1, gid="limits")
    a2.axvline(d["min_intensity"], color=FIT_COLOR, ls="--", lw=1, gid="limits")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("intensity / mean of the other replicates")
    a2.set_ylabel("deviance / series median")
    n_both = 0 if g is None else int(g["count"].sum())
    a2.set_title(f"fit against intensity ({n_both:,} spectra with both); colour: share\n"
                 f"flagged; sparse cells as points (grey: kept, open: flagged)", fontsize=9)
    labels = [s or "all" for s in d["samples"]]
    a3.bar(labels, 100 * d["flagged_share"], color=FIT_COLOR, gid="samples")
    a3.set_ylabel("flagged spectra (%)")
    a3.set_title("flagged share per sample", fontsize=9)
    a3.tick_params(axis="x", rotation=45 if len(labels) > 6 else 0)
    fig.tight_layout()
    return fig


# --- K selection overview -------------------------------------------------------------------

def k_selection_data(results: Results) -> dict:
    """The K of a run: counts of the selected K and of the K in effect (with the reviewer's
    choices) for each K tried, and the share of each K in effect at every distinct time
    (``times``, ``time_labels`` with ``FD``/``UN`` for the controls, ``share`` ``(T, nK)``)."""
    fit = results.fit
    ks = tuple(fit.k_values)
    eff = results.effective_K
    times = np.unique(fit.time)
    share = np.array([[np.mean(eff[fit.time == t] == k) for k in ks] for t in times])
    labels = [_control_role(results, float(t)) or f"{t:g}" for t in times]
    return {
        "k_values": ks,
        "selected": np.array([int((fit.selected_K == k).sum()) for k in ks]),
        "effective": np.array([int((eff == k).sum()) for k in ks]),
        "times": times, "time_labels": labels, "share": share,
        "reviewed": bool(results.review is not None and len(results.review)),
    }


def plot_k_selection(results: Results) -> Figure:
    """The K of a run: how many spectra selected each K (and, after review, how many use each),
    and the share of each K at every time point."""
    d = k_selection_data(results)
    fig = Figure(figsize=(12, 3.8))
    a1, a2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 2]})
    x = np.arange(len(d["k_values"]))
    width = 0.4 if d["reviewed"] else 0.7
    a1.bar(x - (width / 2 if d["reviewed"] else 0), d["selected"], width,
           color=[COLORS[i % len(COLORS)] for i in range(len(x))], label="selected",
           gid="selected")
    if d["reviewed"]:
        a1.bar(x + width / 2, d["effective"], width, color="none",
               edgecolor=[COLORS[i % len(COLORS)] for i in range(len(x))], hatch="//",
               label="after review", gid="effective")
        a1.legend(fontsize=7, frameon=False)
    for xi, n in zip(x, d["selected"], strict=True):
        a1.text(xi, n, f"{n:,}", ha="center", va="bottom", fontsize=8)
    a1.set_xticks(x, [f"K={k}" for k in d["k_values"]])
    a1.set_ylabel("spectra")
    a1.set_title("selected K", fontsize=9)
    bottom = np.zeros(len(d["times"]))
    pos = np.arange(len(d["times"]))
    for j, k in enumerate(d["k_values"]):
        a2.bar(pos, 100 * d["share"][:, j], bottom=100 * bottom,
               color=COLORS[j % len(COLORS)], label=f"K={k}", gid=f"share:{k}")
        bottom += d["share"][:, j]
    a2.set_xticks(pos, d["time_labels"], rotation=45 if len(pos) > 8 else 0, fontsize=8)
    a2.set_xlabel("time point (s)")
    a2.set_ylabel("spectra (%)")
    a2.set_title("K in effect at each time point", fontsize=9)
    a2.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return fig


# --- all overviews ---------------------------------------------------------------------------

def overview_figures(results: Results) -> dict[str, Figure]:
    """The overviews the results allow: ``k_selection`` always, ``controls`` and ``quality``
    when the results have them."""
    out = {"k_selection": plot_k_selection(results)}
    if results.controls is not None:
        out["controls"] = plot_controls_overview(results)
    if results.quality is not None:
        out["quality"] = plot_quality_overview(results)
    return out


def write_overview(results: Results, path, dpi: int = 200) -> list:
    """Write the overviews: one multipage PDF if ``path`` ends in ``.pdf``, else one file per
    overview named ``<stem>_<kind><suffix>`` (SVG or PNG). Returns the files."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figures = overview_figures(results)
    if path.suffix == ".pdf":
        with PdfPages(path) as pdf:
            for fig in figures.values():
                pdf.savefig(fig, dpi=dpi)
        return [path]
    files = []
    for kind, fig in figures.items():
        out = path.with_name(f"{path.stem}_{kind}{path.suffix}")
        fig.savefig(out, dpi=dpi)
        files.append(out)
    return files
