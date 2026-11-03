# SPDX-License-Identifier: GPL-3.0-or-later
"""The K comparison of one spectrum, and overviews of a whole run: controls, quality flags and
the selected K. As elsewhere in :mod:`hxex.report`, every plot has a ``*_data`` function that
returns what it draws, and the plots return figures without showing or saving them."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from ..controls import CONTROL_NOTES
from ..io.times import format_time
from ..quality import FLAG_NAMES
from ..spectra import point_order_key, time_point
from .results import Results
from .series import _control_role
from .spectrum import COLORS, FIT_COLOR, plot_spectrum
from .tables import _index_columns

FALLBACK_REASONS = ("m0_missing", "m0_high", "fd_missing", "retention_low",
                    "retention_outlier", "retention_high", "fd_below_data", "fd_off_model",
                    "no_amides")
QUALITY_FLAGS = FLAG_NAMES
GOOD = "#999999"  # unflagged points and measured controls
NOTE = "#c4c4c4"  # control notes: the control stays in use
LIMIT_TEXT = {"color": FIT_COLOR, "fontsize": 7, "rotation": 90, "va": "top"}

# What the overview's bars are called on the figure: short words close to the flag names of the
# tables (owner, 2026-11-02: plain, not verbose). Every name a bar can show has an entry;
# tests/test_report.py checks it.
CONTROL_LABELS = {
    "measured": "both measured",
    "uncorrected": "uncorrected (fragments)",
    # why a control was missing or not used
    "m0_missing": "no UN control",
    "m0_high": "UN too heavy",
    "fd_missing": "no FD control",
    "retention_low": "FD retention too low",
    "retention_outlier": "FD retention outlier",
    "retention_high": "FD retention too high",
    "fd_below_data": "FD below the data",
    "fd_off_model": "FD off the prediction",
    "no_amides": "no amides",
    # notes: the control stays in use
    "fd_disagrees": "FD differs from other samples (note)",
    "fd_multimodal": "FD two populations (note)",
    "m0_multimodal": "UN two populations (note)",
    "data_below_m0": "data below UN (note)",
    "fd_retention_agrees": "agrees with other samples (note)",
    "fd_charges_disagree": "FD differs between charges (note)",
    "m0_replicate_left_out": "UN replicate left out (note)",
    "fd_replicate_left_out": "FD replicate left out (note)",
    "fd_charges_agree": "agrees across charges (note)",
    "fd_deamidated": "FD deamidation corrected (note)",
}
QUALITY_LABELS = {
    "poor_fit": "poor fit",
    "low_intensity": "low intensity",
    "noise_outlier": "noise outlier",
    "mz_shift": "m/z shift",
    "off_mz": "off m/z",
    "poor_envelope": "poor envelope",
    "saturated": "saturated",
    "tails_low": "tails low",
    "bright": "bright",
    "rt_found": "RT found",
    "any": "any flag",
}


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


LEGEND_BELOW = 1.6  # inches added to a K comparison's height for the legends under the panels


LEGEND_BELOW_ROWS = 0.9  # inches per row when only the populations are under each panel


def _legend_below(ax, only_populations: bool = False) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if only_populations:
        pairs = [(h, lab) for h, lab in zip(handles, labels, strict=True) if lab.startswith("pop ")]
        handles, labels = [h for h, _ in pairs], [lab for _, lab in pairs]
    if handles:
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  fontsize=6.5, frameon=False)


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
    rejected it; the selected K is marked, and the reviewer's K if it differs. A K the screen
    skipped for the spectrum shows the picked spectrum alone. Keyword arguments go to
    :func:`plot_spectrum`."""
    fig = plot_k_comparisons([spectrum_id], results, ks=ks, show=show, cell=cell, **kwargs)
    fig.suptitle(spectrum_id, fontsize=9)  # constrained layout places it with the legends
    return fig


def plot_k_comparisons(
    spectrum_ids: Sequence[str],
    results: Results,
    *,
    labels: Sequence[str] | None = None,
    ks: Sequence[int] | None = None,
    show: Sequence[str] = ("picked", "fit"),
    cell: tuple[float, float] = (4.6, 3.6),
    title: str | None = None,
    **kwargs,
) -> Figure:
    """Several spectra fitted at each K: a row of :func:`plot_k_comparison` per spectrum (the
    replicates of a time point, say), each row's titles starting with its label (``labels``,
    default the spectrum id) and its chosen K framed. ``ks`` default: every K tried."""
    if not len(spectrum_ids):
        raise ValueError("no spectra to compare")
    rows = [k_comparison_data(sid, results) for sid in spectrum_ids]
    ks = list(rows[0]["k_values"] if ks is None else ks)
    bad = [k for k in ks if k not in rows[0]["k_values"]]
    if bad:
        raise ValueError(f"K = {bad} was not fitted (tried {rows[0]['k_values']})")
    labels = list(spectrum_ids if labels is None else labels)
    # each panel's legend (its own populations) under it, covering none of the spectrum
    # (owner, 2026-11-03); constrained layout makes room for it
    many = len(rows) > 1
    # several rows: the layers named once at the top, only each fit's populations under it
    below = LEGEND_BELOW_ROWS if many else LEGEND_BELOW
    height = (cell[1] + below) * len(rows) + (0.6 if many else 0.0)
    fig = Figure(figsize=(cell[0] * len(ks), height), layout="constrained")
    axes = np.atleast_2d(fig.subplots(len(rows), len(ks), sharey="row", squeeze=False))
    kwargs.setdefault("legend", False)
    for r, (sid, d) in enumerate(zip(spectrum_ids, rows, strict=True)):
        fitted = {k: bool(results.fit.fitted(k)[d["index"]]) for k in ks}
        lead = f"{labels[r]}: " if many else ""
        for ax, k in zip(axes[r], ks, strict=True):
            j = d["k_values"].index(k)
            if not fitted[k]:  # the screen skipped it: the spectrum alone
                plot_spectrum(sid, results, show=("picked",), ax=ax, title=False, **kwargs)
                ax.set_title(f"{lead}K={k} not fitted (screened)", fontsize=8)
                _legend_below(ax, many)
                continue
            plot_spectrum(sid, results, k=k, show=show, ax=ax, title=False, **kwargs)
            _legend_below(ax, many)
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
            ax.set_title(f"{lead}K={k}   loss {d['nll'][j]:.4g}   {score_text}"
                         + (f"\n{', '.join(marks)}" if marks else ""), fontsize=8,
                         fontweight="bold" if k == d["effective"] else "normal")
            if k == d["effective"]:
                for spine in ax.spines.values():
                    spine.set_linewidth(2.0)
    if title is not None:
        fig.suptitle(title, fontsize=9)
    if many:  # the layers, once for the page (from the panel naming the most)
        named = max((ax.get_legend_handles_labels() for ax in axes.flat), key=lambda h: len(h[1]))
        layers = [(h, lab) for h, lab in zip(*named, strict=True) if not lab.startswith("pop ")]
        if layers:
            fig.legend([h for h, _ in layers], [lab for _, lab in layers],
                       loc="outside lower center", ncol=min(len(layers), 5), fontsize=7,
                       frameon=False)
    return fig


# --- controls overview ----------------------------------------------------------------------

def controls_overview_data(results: Results) -> dict:
    """For every series with controls: ``retention`` of the measured fully deuterated control
    against the undeuterated level used (NaN without one), ``m0`` (the measured undeuterated
    mean, NaN without one), the counts of each fallback ``reasons``, of series using both
    measured controls (``measured``) and of series the controls leave uncorrected
    (``uncorrected``: fragment ions), and the configured limits."""
    c = results.controls
    if c is None:
        raise ValueError("these results have no controls")
    has_fd = c.n_fd > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        retention = np.where(has_fd & (c.n_amides > 0),
                             (c.mfd_measured - c.m0) / c.n_amides, np.nan)
    reasons = {r: int(sum(r in str(f).split(";") for f in c.flags)) for r in FALLBACK_REASONS}
    notes = {r: int(sum(r in str(f).split(";") for f in c.flags)) for r in CONTROL_NOTES}
    cfg = c.config
    return {
        "data_id": c.data_id, "retention": retention, "m0": c.m0_measured,
        "fallback": c.fallback.copy(), "reasons": reasons, "notes": notes,
        "measured": int((~c.fallback & ~c.uncorrected).sum()),
        "uncorrected": int(c.uncorrected.sum()),
        "retention_min": retention_floor(c), "retention_max": cfg.retention_max,
        "m0_max": cfg.m0_max, "n_series": len(c),
    }


def retention_limit_labels(controls) -> tuple[str, str]:
    """What the lower and upper retention limits of :func:`plot_controls_overview` are, in
    words: how the lower one was set (:func:`retention_floor`) and the ``retention_max``."""
    from ..controls import OUTLIER_MIN_SERIES

    cfg = controls.config
    floor = retention_floor(controls)
    upper = f"max {cfg.retention_max:g}"
    if cfg.retention_min is not None:
        return f"min {floor:.2f}", upper
    limits = controls.limits
    if limits is not None and limits.get("median") is not None:
        return f"outlier limit {floor:.2f}", upper
    if limits is not None:
        return f"no outlier limit (< {OUTLIER_MIN_SERIES} controls)", upper
    if cfg.fd_mode == "kint":
        return "no lower limit (kint)", upper
    return f"min {floor:.2f}", upper


def _limit_text(ax, x: float, text: str, *, right: bool = True) -> None:
    """``text`` beside the vertical limit line at ``x``, read upwards, at the top of ``ax``."""
    ax.annotate(text, (x, 1.0), xycoords=ax.get_xaxis_transform(),
                xytext=(2 if right else -2, -3), textcoords="offset points",
                ha="left" if right else "right", gid="limit_label", **LIMIT_TEXT)


def _counts_text(ax, counts, total: int | None = None) -> None:
    """Each bar's count at its end (with its share of ``total`` when given)."""
    for y, n in enumerate(counts[::-1]):
        share = f" ({100 * n / max(total, 1):.1f} %)" if total is not None else ""
        ax.annotate(f" {n:,}{share}", (n, y), va="center", fontsize=8)
    ax.set_xlim(0, 1.3 * max(max(counts, default=0), 1))


def retention_floor(controls) -> float:
    """The lower retention limit a run's controls were judged by: the set ``retention_min``,
    the run-relative limit (:attr:`hxex.controls.ControlTable.limits`), or, in bundles written
    before it, half of ``d2o_fraction`` (0 with the theoretical fallback)."""
    cfg = controls.config
    if cfg.retention_min is not None:
        return cfg.retention_min
    if controls.limits is not None:
        return float(controls.limits["retention_floor"])
    return 0.0 if cfg.fd_mode == "kint" else cfg.d2o_fraction / 2


def plot_controls_overview(results: Results) -> Figure:
    """The controls of a run: the distribution of the measured retention with the limits it
    was judged by (:func:`retention_floor`), the distribution of the undeuterated control means
    with ``m0_max``, and the control status of the series: how many use both measured controls,
    why controls were missing or not used, and the notes that leave the control in use. The
    limits are named beside their lines (:func:`retention_limit_labels`), the bars in short
    words (:data:`CONTROL_LABELS`)."""
    d = controls_overview_data(results)
    fig = Figure(figsize=(15, 4.2), layout="constrained")
    a1, a2, a3 = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 1.25]})
    ret = d["retention"][np.isfinite(d["retention"])]
    if len(ret):  # fixed edges over the drawn range, so that a small run shows its bars
        a1.hist(ret, bins=np.linspace(min(0.0, float(ret.min())),
                                      max(1.1, float(ret.max()), d["retention_max"]), 45),
                color=GOOD, gid="retention")
    low, high = retention_limit_labels(results.controls)
    for x, text in ((d["retention_min"], low), (d["retention_max"], high)):
        a1.axvline(x, color=FIT_COLOR, ls="--", lw=1, gid="limits")
        _limit_text(a1, x, text, right=x < d["retention_max"])
    median = f", median {np.median(ret):.2f}" if len(ret) else ""
    a1.set_title(f"FD control retention ({len(ret)} series{median})", fontsize=9)
    a1.set_xlabel("retention")
    a1.set_ylabel("series")
    m0 = d["m0"][np.isfinite(d["m0"])]
    if len(m0):
        a2.hist(m0, bins=np.linspace(min(0.0, float(m0.min())),
                                     max(d["m0_max"], float(m0.max())) * 1.1, 45),
                color=GOOD, gid="m0")
    a2.axvline(d["m0_max"], color=FIT_COLOR, ls="--", lw=1, gid="limits")
    _limit_text(a2, d["m0_max"], f"max {d['m0_max']:g} D", right=False)
    a2.set_title(f"UN control means ({len(m0)} series)", fontsize=9)
    a2.set_xlabel("UN mean (D)")
    # the reasons that occurred (orange), then the notes (light grey, the control used)
    reasons = [k for k, v in d["reasons"].items() if v]
    notes = [k for k, v in d["notes"].items() if v]
    # series the controls leave uncorrected (fragment ions): their own bar, in neither group
    plain = ["uncorrected"] if d["uncorrected"] else []
    keys = ["measured", *plain, *reasons, *notes]
    counts = [d["measured"], *([d["uncorrected"]] if plain else []),
              *(d["reasons"][k] for k in reasons), *(d["notes"][k] for k in notes)]
    colors = ([GOOD] + ["#6b8fb3"] * len(plain) + [FIT_COLOR] * len(reasons)
              + [NOTE] * len(notes))
    a3.barh([CONTROL_LABELS[k] for k in keys][::-1], counts[::-1], color=colors[::-1],
            gid="reasons")
    _counts_text(a3, counts)
    a3.tick_params(axis="y", labelsize=8)
    a3.set_title(f"control status of {d['n_series']} series "
                 f"({int(d['fallback'].sum())} with a stand-in control)", fontsize=9)
    a3.set_xlabel("series")
    return fig


# --- quality overview -----------------------------------------------------------------------

def quality_overview_data(results: Results) -> dict:
    """The quality flags of a run: the count of each flag and of flagged spectra, the
    ``fit_rel`` (the ``poor_fit`` statistic: ``misfit_rel``, or ``deviance_rel`` for flags written
    before it; ``fit_label`` names it) and ``intensity_rel`` of every spectrum with ``flagged``,
    the thresholds,
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
        "fit_rel": q.misfit_rel if np.isfinite(q.misfit_rel).any() else q.deviance_rel,
        "fit_label": ("misfit / series median" if np.isfinite(q.misfit_rel).any()
                      else "deviance / series median"),
        "intensity_rel": q.intensity_rel, "flagged": q.flagged,
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
    """The quality flags of a run: how many spectra carry each flag (in words,
    :data:`QUALITY_LABELS`); each spectrum's fit
    (misfit relative to its series) against its intensity (relative to its replicates), as the
    share of flagged spectra in each cell of a ``bins`` x ``bins`` grid (cells with fewer than
    ``min_count`` spectra show the spectra as points instead), with the two flag limits named
    beside their lines; and the share of flagged spectra per sample."""
    d = quality_overview_data(results)
    fig = Figure(figsize=(15, 4.4), layout="constrained")
    fig.get_layout_engine().set(wspace=0.08)  # room between the colour bar and the next panel
    a1, a2, a3 = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.15, 1.1, 0.8]})
    keys = [*d["counts"], "any"]
    counts = [*d["counts"].values(), d["n_flagged"]]
    a1.barh([QUALITY_LABELS[k] for k in keys][::-1], counts[::-1], color=FIT_COLOR,
            gid="counts")
    _counts_text(a1, counts, total=d["n"])
    a1.tick_params(axis="y", labelsize=8)
    a1.set_title(f"quality flags of {d['n']:,} spectra", fontsize=9)
    a1.set_xlabel("spectra")
    g = flagged_share_grid(d["intensity_rel"], d["fit_rel"], d["flagged"], bins=bins,
                           min_count=min_count)
    if g is not None:
        share = np.ma.masked_where(g["count"] < min_count, 100 * g["share"])
        mesh = a2.pcolormesh(g["x_edges"], g["y_edges"], share.T, cmap="cividis", vmin=0,
                             vmax=100, gid="share")
        bar = fig.colorbar(mesh, ax=a2)
        # read top to bottom, apart from the next panel's axis label, which reads upwards
        bar.set_label("flagged (% per cell)", rotation=270, labelpad=12)
        # cells with few spectra: the spectra themselves, so isolated ones stay visible
        sx, sy, sf = g["sparse_x"], g["sparse_y"], g["sparse_flagged"]
        a2.scatter(sx[~sf], sy[~sf], s=6, color=GOOD, lw=0, gid="spectra",
                   label="spectrum (sparse cell)")
        a2.scatter(sx[sf], sy[sf], s=14, facecolors="none", edgecolors=FIT_COLOR, lw=0.7,
                   gid="flagged", label="flagged (sparse cell)")
        a2.legend(loc="lower right", fontsize=7, frameon=False, handletextpad=0.2)
    a2.axhline(d["poor_fit_factor"], color=FIT_COLOR, ls="--", lw=1, gid="limits")
    a2.axvline(d["min_intensity"], color=FIT_COLOR, ls="--", lw=1, gid="limits")
    # the limits named on the panel: the fit limit at its right end (on a pale box, over the
    # points), the intensity limit above the frame
    a2.annotate(f"poor fit {d['poor_fit_factor']:g}×",
                (1.0, d["poor_fit_factor"]), xycoords=("axes fraction", "data"),
                xytext=(-3, 3), textcoords="offset points", ha="right", va="bottom",
                color=FIT_COLOR, fontsize=7, gid="limit_label",
                bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "alpha": 0.8,
                      "linewidth": 0})
    a2.annotate(f"low intensity {d['min_intensity']:g}×", (d["min_intensity"], 1.0),
                xycoords=("data", "axes fraction"), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", color=FIT_COLOR, fontsize=7, gid="limit_label")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("intensity / replicates' mean")
    a2.set_ylabel(d["fit_label"])
    n_both = 0 if g is None else int(g["count"].sum())
    a2.set_title(f"misfit against intensity ({n_both:,} spectra)", fontsize=9, pad=14)
    import textwrap

    labels = [textwrap.shorten(s or "all", 24, placeholder="...") for s in d["samples"]]
    a3.bar(labels, 100 * d["flagged_share"], color=FIT_COLOR, gid="samples")
    a3.set_ylabel("flagged spectra (%)")
    a3.set_title("flagged share per sample", fontsize=9)
    a3.tick_params(axis="x", rotation=45 if len(labels) > 6 or any(
        len(s) > 10 for s in labels) else 0)
    return fig


# --- K selection overview -------------------------------------------------------------------

def k_selection_data(results: Results) -> dict:
    """The K of a run: counts of the selected K and of the K in effect (with the reviewer's
    choices) for each K tried, and the share of each K in effect at every distinct time
    (``times``, ``time_labels`` with ``FD``/``UN`` for the controls, ``share`` ``(T, nK)``)."""
    fit = results.fit
    ks = tuple(fit.k_values)
    eff = results.effective_K
    kept = np.asarray(results.review_status) != "rejected"  # the analysis uses only these
    points = time_point(fit.time, fit.role)
    order = {}
    for i, key in enumerate(points):  # the undeuterated control first, the fully deuterated last
        order.setdefault(key, (point_order_key(float(fit.time[i]), str(fit.role[i])), i))
    keys = sorted(order, key=lambda key: order[key][0])
    times = np.array([float(fit.time[order[key][1]]) for key in keys])
    share = np.array([[np.mean(eff[(points == key) & kept] == k) if (
        (points == key) & kept).any() else 0.0 for k in ks] for key in keys])
    labels = [_control_role(fit.role[order[key][1]])
              or format_time(fit.time[order[key][1]], results.time_scale) for key in keys]
    return {
        "k_values": ks,
        "selected": np.array([int((fit.selected_K == k).sum()) for k in ks]),
        "effective": np.array([int(((eff == k) & kept).sum()) for k in ks]),
        "rejected": int((~kept).sum()),
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
               label="after review" + (f" ({d['rejected']} rejected left out)"
                                       if d["rejected"] else ""), gid="effective")
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
    a2.set_xlabel("time point")
    a2.set_ylabel("spectra (%)")
    a2.set_title("K in effect at each time point"
                 + (" (rejected spectra left out)" if d["rejected"] else ""), fontsize=9)
    a2.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return fig


# --- all overviews ---------------------------------------------------------------------------

def overview_kinds(results: Results) -> list[str]:
    """The overviews the results allow, without drawing them: ``k_selection`` always,
    ``controls`` and ``quality`` when the results have them."""
    return ["k_selection", *(["controls"] if results.controls is not None else []),
            *(["quality"] if results.quality is not None else [])]


def overview_files(results: Results, path) -> list[str]:
    """The names :func:`write_overview` gives the overviews at ``path``: ``path`` itself for a
    PDF, else ``<stem>_<kind><suffix>`` per overview."""
    from pathlib import PurePath

    path = PurePath(path)
    if path.suffix == ".pdf":
        return [path.name]
    return [f"{path.stem}_{kind}{path.suffix}" for kind in overview_kinds(results)]


def overview_figures(results: Results) -> dict[str, Figure]:
    """The overviews the results allow (:func:`overview_kinds`), drawn."""
    draw = {"k_selection": plot_k_selection, "controls": plot_controls_overview,
            "quality": plot_quality_overview}
    return {kind: draw[kind](results) for kind in overview_kinds(results)}


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
    for name, fig in zip(overview_files(results, path), figures.values(), strict=True):
        out = path.with_name(name)
        fig.savefig(out, dpi=dpi)
        files.append(out)
    return files
