# SPDX-License-Identifier: GPL-3.0-or-later
"""The K comparison of one spectrum, and overviews of a whole run: controls, quality flags and
the selected K. As elsewhere in :mod:`hxex.report`, every plot has a ``*_data`` function that
returns what it draws, and the plots return figures without showing or saving them."""

from __future__ import annotations

import math
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
from .spectrum import COLORS, FIT_COLOR, plot_spectrum, wrap_title
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
    "fd_same_measurement": "FD same file as another sample (note)",
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
    "drift_unclear": "drift unclear",
    "elution_cut": "elution cut",
    "precursor_off": "precursor off",
    "overlapping_ion": "overlapping ion",
    "below_noise": "below the noise",
    "spectrum_off_mz": "spectrum off m/z",
    "undeut_high": "UN too heavy",
    "any": "any flag",
}


# --- K comparison of one spectrum -----------------------------------------------------------

def k_comparison_data(spectrum_id: str, results: Results, *, analysis=None,
                      sets="compute") -> dict:
    """The fit of one spectrum at every K tried: ``k_values``, ``nll``, ``bic``, ``aic``, the
    selection ``score`` (the configured criterion, inf where the minimum-population rule rejected
    that K), ``delta`` (score minus the best score), ``criterion_delta`` (the criterion minus the
    best score, also where the rule rejected the K), ``passes`` (minimum population),
    ``smallest`` (each K's smallest population), ``fit_min`` (``[run.fit] min_population``),
    ``criterion``, ``selected`` and ``effective`` (the reviewer's K, else the selected one), and
    ``test``: None without the population test, else ``by`` (True: the test chose
    ``selected``), ``k`` (the test's K), ``significance``, ``min_population`` and, per K above
    one, ``steps[K]``: the step from K - 1 (``p``, ``added`` (the smallest population of the K
    fit), ``kind`` (how its null was made), ``decides`` (False above ``decide_up_to``), ``adds``
    (its p and share pass) and ``reached`` (every step below it adds a population)), and
    ``set``: the population test of the spectrum's set (:class:`hxex.analysis.set_test.SetTest`:
    every replicate and charge state of its peptide in its sample at its time, as ``hxex
    analyse`` decides the sets with the analysis settings ``analysis``, its defaults when None;
    ``sets``: those already worked out, one per spectrum, :func:`~hxex.analysis.set_test.
    spectrum_sets`), None without the test or for a spectrum in no set."""
    from ..analysis.set_test import spectrum_sets

    i = results.find(spectrum_id)
    fit = results.fit
    ks = tuple(fit.k_values)
    score = np.asarray(fit.score[i], dtype=float)
    finite = score[np.isfinite(score)]
    best = finite.min() if len(finite) else np.nan
    values = np.array([getattr(fit.fits[k], fit.config.criterion)[i] for k in ks], dtype=float)
    return {
        "spectrum_id": spectrum_id, "index": i, "k_values": ks,
        "nll": np.array([fit.fits[k].nll[i] for k in ks]),
        "bic": np.array([fit.fits[k].bic[i] for k in ks]),
        "aic": np.array([fit.fits[k].aic[i] for k in ks]),
        "score": score, "delta": score - best, "criterion_delta": values - best,
        "passes": np.array([bool(fit.fits[k].passes_min_pop[i]) for k in ks]),
        "smallest": np.array([float(np.min(fit.fits[k].populations[i])) for k in ks]),
        "fit_min": float(fit.config.min_population),
        "criterion": fit.config.criterion, "selected": int(fit.selected_K[i]),
        "effective": int(results.effective_K[i]),
        "test": _test_steps(results, i),
        "set": _set_of(results, i, analysis, sets, spectrum_sets),
    }


def _set_of(results: Results, i: int, analysis, sets, spectrum_sets):
    """The set test of fit row ``i`` (:func:`k_comparison_data`), or None."""
    if results.population_test is None:
        return None
    if isinstance(sets, str):  # "compute"
        sets = spectrum_sets(results, analysis)
    return sets[i] if sets is not None else None


def set_note(d: dict) -> str | None:
    """The line a K comparison adds where the spectrum's K (the K in effect) differs from its
    set's (:func:`k_comparison_data`): ``population test (set) p-value 6.9e-04 < 0.01: K = 1
    (self), K = 2 (set)``; None where they agree or there is no set. "(set)": the test of
    every replicate and charge state of the peptide in its sample at that time, as against
    the spectrum's own, "(self)" (owner, 2026-11-18). Information only (owner, 2026-11-17):
    the spectrum keeps its own K; the analysis uses the set's."""
    from ..analysis.set_test import set_step

    t = d.get("set")
    if t is None or int(t.K) == int(d["effective"]):
        return None
    tail = f"K = {int(d['effective'])} (self), K = {int(t.K)} (set)"
    p, added = set_step(t)
    if not np.isfinite(p):
        return f"population test (set): not tested; {tail}"
    test = d.get("test") or {}
    sig = test.get("significance", 0.01)
    text = f"population test (set) p-value {_p_text(p)} {'<' if p < sig else '≥'} {sig:g}"
    least = test.get("min_population")
    if int(t.K) == 1 and p < sig and np.isfinite(added) and least is not None:
        text += f", population {_share(added)} < {_share(least)}"
    return f"{text}: {tail}"


def _test_steps(results: Results, i: int) -> dict | None:
    """The population test of fit row ``i`` step by step (:func:`k_comparison_data`)."""
    from ..population_test import NOT_TESTED, decided_steps

    test = results.population_test
    if test is None:
        return None
    cfg = test.config
    n_decided = decided_steps(cfg, test.steps)
    steps, reached = {}, True
    for s, k in enumerate(test.steps):
        p = float(test.p[i, s])
        added = float(test.added_population[i, s])
        kind = str(test.null_kind[i, s])
        adds = (kind not in NOT_TESTED and np.isfinite(p) and p < cfg.significance
                and added >= cfg.min_population)
        steps[k + 1] = {"p": p, "added": added, "kind": kind, "decides": s < n_decided,
                        "adds": bool(adds), "reached": reached}
        reached = reached and bool(adds)
    return {"by": int(test.selected_K[i]) == int(results.fit.selected_K[i]),
            "k": int(test.selected_K[i]),
            "significance": float(cfg.significance),
            "min_population": float(cfg.min_population), "steps": steps}


def _share(x: float) -> str:
    return f"{100 * x:.0f} %" if x >= 0.01 else f"{100 * x:.1f} %"


def _p_text(p: float) -> str:
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def k_comparison_notes(d: dict, k: int) -> list[str]:
    """Why K = ``k`` was or was not chosen, in words, from :func:`k_comparison_data`: a smallest
    population below the fit's or the population test's minimum, and the test's step from
    ``k - 1`` to ``k``."""
    from ..population_test import FLAT, SCREENED, UNTESTED

    j = d["k_values"].index(k)
    test = d["test"]
    step = None if test is None else test["steps"].get(k)
    tested = step is not None and step["kind"] not in (UNTESTED, FLAT, SCREENED)
    below = []  # (share, the minimum it is below)
    if not d["passes"][j]:
        below.append((d["smallest"][j], f"fit minimum {_share(d['fit_min'])}"))
    if tested and step["added"] < test["min_population"]:
        below.append((step["added"], f"test minimum {_share(test['min_population'])}"))
    notes = []
    if below:
        shares = {_share(x) for x, _ in below}
        if len(shares) == 1:
            notes.append(f"population {shares.pop()} < " + ", ".join(m for _, m in below))
        else:
            notes.extend(f"population {_share(x)} < {m}" for x, m in below)
    if step is None:
        return notes
    if not tested:
        notes.append("population test (self): screened" if step["kind"] == SCREENED
                     else "population test (self): not tested")
        return notes
    sig = test["significance"]
    if not step["reached"]:
        outcome = f"not selected (K={k - 1} not selected)"
    elif not step["decides"]:
        outcome = ("flag only (above decide_up_to)" if step["adds"]
                   else "not selected")
    elif not step["adds"]:
        outcome = "not selected"
    else:  # a step below the chosen K passed; only the chosen K says selected
        outcome = "selected" if k == test["k"] else "passed"
    notes.append(f"population test (self) p-value {_p_text(step['p'])} "
                 f"{'<' if step['p'] < sig else '≥'} {sig:g} {outcome}")
    return notes


LEGEND_BELOW = 1.6  # inches added to a K comparison's height for the legends under the panels


LEGEND_BELOW_ROWS = 0.9  # inches per row when only the populations are under each panel


def _legend_below(ax, only_populations: bool = False, under=None) -> None:
    """The legend of ``ax`` under it, or under ``under`` (its residual panel)."""
    handles, labels = ax.get_legend_handles_labels()
    if only_populations:
        pairs = [(h, lab) for h, lab in zip(handles, labels, strict=True) if lab.startswith("pop ")]
        handles, labels = [h for h, _ in pairs], [lab for _, lab in pairs]
    if handles:
        target = ax if under is None else under
        # the same distance under a residual panel (a quarter of the height) as under a spectrum
        target.legend(handles, labels, loc="upper center",
                      bbox_to_anchor=(0.5, -0.18 if under is None else -0.54),
                      fontsize=6.5, frameon=False)


RESIDUAL_HEIGHT = 1.3  # a panel with its residuals under it, as on the series' spectra pages


def _residual_grid(fig: Figure, nrows: int, ncols: int):
    """Axes for ``nrows`` spectra at ``ncols`` K, each with a residual panel under it: the
    spectra of a row share their intensity axis, the residuals of a row theirs, and each
    residual panel its spectrum's m/z axis."""
    # one grid, spectra and residuals in alternate rows, so that the panels of a row line up
    # whatever the height of their titles (a grid of its own per panel did not)
    grid = fig.add_gridspec(2 * nrows, ncols, height_ratios=[3, 1] * nrows)
    axes = np.empty((nrows, ncols), dtype=object)
    raxes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            ax = fig.add_subplot(grid[2 * r, c], sharey=axes[r, 0] if c else None)
            axes[r, c] = ax
            raxes[r, c] = fig.add_subplot(grid[2 * r + 1, c], sharex=ax,
                                          sharey=raxes[r, 0] if c else None)
    return axes, raxes


def plot_k_comparison(
    spectrum_id: str,
    results: Results,
    *,
    ks: Sequence[int] | None = None,
    show: Sequence[str] = ("picked", "fit"),
    cell: tuple[float, float] = (4.6, 3.6),
    analysis=None,
    sets="compute",
    residual: bool = False,
    **kwargs,
) -> Figure:
    """One spectrum fitted at each K side by side (default: every K tried), each panel with its
    loss, its selection score relative to the best and whether the minimum-population rule
    rejected it; the selected K is marked, and the reviewer's K if it differs. A K the screen
    skipped for the spectrum shows the picked spectrum alone. Where the spectrum's set (with
    the population test) has another number of populations than the spectrum's K, the
    panel at the set's number says so (:func:`set_note`; ``analysis`` and ``sets`` as in
    :func:`k_comparison_data`). ``residual`` puts each fit's residuals (picked minus fit) in a
    small panel under it, on one scale for every K. Keyword arguments go to
    :func:`plot_spectrum`."""
    fig = plot_k_comparisons([spectrum_id], results, ks=ks, show=show, cell=cell,
                             analysis=analysis, sets=sets, residual=residual, **kwargs)
    # constrained layout places it with the legends; a fragment's long id on two lines
    fig.suptitle(wrap_title(spectrum_id, fig.get_figwidth() - 0.4, 9), fontsize=9)
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
    analysis=None,
    sets="compute",
    residual: bool = False,
    **kwargs,
) -> Figure:
    """Several spectra fitted at each K: a row of :func:`plot_k_comparison` per spectrum (the
    replicates of a time point, say), each row's titles starting with its label (``labels``,
    default the spectrum id) and its chosen K framed. ``ks`` default: every K tried.
    ``analysis`` and ``sets``: as in :func:`k_comparison_data` (the set's line,
    :func:`set_note`). ``residual``: each fit's residuals (picked minus fit) in a small panel
    under it, the residual panels of one spectrum on one scale, so that the K's misfits
    compare directly."""
    if not len(spectrum_ids):
        raise ValueError("no spectra to compare")
    if isinstance(sets, str) and results.population_test is not None:  # worked out once
        from ..analysis.set_test import spectrum_sets

        sets = spectrum_sets(results, analysis)
    rows = [k_comparison_data(sid, results, analysis=analysis, sets=sets)
            for sid in spectrum_ids]
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
    residual = residual and "picked" in show and "fit" in show  # residuals need both layers
    plot = cell[1] * (RESIDUAL_HEIGHT if residual else 1.0)
    height = (plot + below) * len(rows) + (0.6 if many else 0.0)
    fig = Figure(figsize=(cell[0] * len(ks), height), layout="constrained")
    if residual:
        axes, raxes = _residual_grid(fig, len(rows), len(ks))
    else:
        axes = np.atleast_2d(fig.subplots(len(rows), len(ks), sharey="row", squeeze=False))
        raxes = np.full(axes.shape, None, dtype=object)
    kwargs.setdefault("legend", False)
    room = max(1.0, cell[0] - 0.8)  # the width a panel's title may take (a long label wraps)
    for r, (sid, d) in enumerate(zip(spectrum_ids, rows, strict=True)):
        fitted = {k: bool(results.fit.fitted(k)[d["index"]]) for k in ks}
        lead = f"{labels[r]}: " if many else ""
        # the set's line: on the panel at the set's number of populations, else on the K in
        # effect's (a set at a K the panels do not show, or the screen skipped)
        set_line = set_note(d)
        set_at = (int(d["set"].K) if set_line and fitted.get(int(d["set"].K))
                  else int(d["effective"]))
        for ax, rax, k in zip(axes[r], raxes[r], ks, strict=True):
            j = d["k_values"].index(k)
            if not fitted[k]:  # the screen skipped it: the spectrum alone
                plot_spectrum(sid, results, show=("picked",), ax=ax, title=False, **kwargs)
                ax.set_title(wrap_title(f"{lead}K={k} not fitted (screened)", room, 8),
                             fontsize=8)
                if rax is not None:  # no fit, no residuals: the m/z axis stays under it
                    rax.text(0.5, 0.5, "no fit", transform=rax.transAxes, ha="center",
                             va="center", fontsize=7, color="grey")
                    rax.set_xlabel(ax.get_xlabel())
                    ax.set_xlabel("")
                _legend_below(ax, many, rax)
                continue
            plot_spectrum(sid, results, k=k, show=show, ax=ax, title=False,
                          residual=rax is not None, rax=rax, **kwargs)
            _legend_below(ax, many, rax)
            marks = []
            if k == d["selected"]:
                if d["test"] is not None and d["test"]["by"]:
                    marks.append("selected by the population test (self)")
                else:
                    marks.append(f"selected (lowest Δ{d['criterion'].upper()})")
            if k == d["effective"] and d["effective"] != d["selected"]:
                marks.append("reviewer's choice")
            name = d["criterion"].upper()
            raw = d["criterion_delta"][j]
            score_text = f"Δ{name} {raw:+.1f}" if np.isfinite(raw) else f"Δ{name} –"
            if not np.isfinite(d["delta"][j]) and np.isfinite(raw):
                score_text += " (ruled out)"
            notes = k_comparison_notes(d, k)
            if set_line and k == set_at:  # the test on a line, the two numbers on the next
                notes.extend(set_line.rsplit(": ", 1))
            # the marks ("selected", ...) and the reasons above the K line, so every panel's K
            # line sits at the same height (a title grows upwards from its axes; owner)
            ax.set_title(wrap_title("".join(f"{line}\n" for line in
                                            ([", ".join(marks)] if marks else []) + notes)
                                    + f"{lead}K={k}   loss {d['nll'][j]:.4g}   {score_text}",
                                    room, 8), fontsize=8,
                         fontweight="bold" if k == d["effective"] else "normal")
            if k == d["effective"]:
                for spine in ax.spines.values():
                    spine.set_linewidth(2.0)
    if residual:  # one axis label per row, as the shared spectra have (fig.subplots' rule)
        for ax, rax in zip(axes[:, 1:].flat, raxes[:, 1:].flat, strict=True):
            for a in (ax, rax):
                a.tick_params(labelleft=False)
                a.set_ylabel("")
        for ax in axes.flat:  # the residuals under it carry the m/z labels
            ax.tick_params(labelbottom=False)
    if title is not None:
        fig.suptitle(wrap_title(title, fig.get_figwidth() - 0.4, 9), fontsize=9)
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
        "retention_min": retention_floor(c), "retention_max": cfg.max_retention,
        "m0_max": cfg.m0_max, "n_series": len(c),
    }


def retention_limit_labels(controls) -> tuple[str, str]:
    """What the lower and upper retention limits of :func:`plot_controls_overview` are, in
    words: how the lower one was set (:func:`retention_floor`) and the upper one
    (``retention_max``, or ``d2o_fraction`` when unset)."""
    from ..controls import OUTLIER_MIN_SERIES

    cfg = controls.config
    floor = retention_floor(controls)
    upper = (f"max {cfg.retention_max:g}" if cfg.retention_max is not None
             else f"max {cfg.max_retention:g} (d2o_fraction)")
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

def k_selection_data(results: Results, max_points: int | None = 48) -> dict:
    """The K of a run: counts of the selected K and of the K in effect (with the reviewer's
    choices) for each K tried, and the share of each K in effect at every distinct time
    (``times``, ``time_labels`` with ``FD``/``UN`` for the controls, ``share`` ``(T, nK)``).
    With more than ``max_points`` times (a kinetics run's hundreds), they are grouped into at
    most that many, spread over log time as ``[figures] max_points`` thins a series' spectra:
    the share is then over every spectrum of a group, labelled by its first and last time
    (``points``: the times before grouping; None: every time its own bar)."""
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
    n_points = len(keys)
    timed = [n for n, key in enumerate(keys) if not _control_role(fit.role[order[key][1]])]
    if max_points is not None and len(timed) > max_points:
        from ..io.mzml import _sample_log

        starts = [timed[i] for i in _sample_log(times[timed], max_points)]
        groups = [[n for n in timed if a <= n < b] for a, b in
                  zip(starts, [*starts[1:], timed[-1] + 1], strict=True)]
        groups = [g for g in groups if g]
        controls = [n for n in range(len(keys)) if n not in set(timed)]
        before = [n for n in controls if n < timed[0]]
        after = [n for n in controls if n > timed[-1]]
        bars = [[n] for n in before] + groups + [[n] for n in after]
        share = np.array([[np.mean(eff[np.isin(points, [keys[n] for n in g]) & kept] == k)
                           if (np.isin(points, [keys[n] for n in g]) & kept).any() else 0.0
                           for k in ks] for g in bars])
        def short(t: float) -> str:  # a scan's time to three figures: 42.4 s, not 42.3934 s
            return format_time(float(f"{t:.3g}"), results.time_scale)

        labels = [labels[g[0]] if g[0] not in set(timed) else short(times[g[0]])
                  if len(g) == 1 else f"{short(times[g[0]])}–{short(times[g[-1]])}"
                  for g in bars]
        times = np.array([times[g[0]] for g in bars])
    return {
        "points": n_points,
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
        # above the bars, in the room left for it (it covered the tallest bar: owner)
        a1.legend(fontsize=7, frameon=False, loc="upper right")
    for xi, n in zip(x, d["selected"], strict=True):
        a1.text(xi, n, f"{n:,}", ha="center", va="bottom", fontsize=8)
    tallest = max([*d["selected"], *(d["effective"] if d["reviewed"] else [])], default=0)
    a1.set_ylim(0, max(tallest, 1) * 1.35)  # room for the counts and the legend
    a1.set_xticks(x, [f"K={k}" for k in d["k_values"]])
    a1.set_ylabel("spectra")
    a1.set_title("selected K", fontsize=9)
    bottom = np.zeros(len(d["times"]))
    pos = np.arange(len(d["times"]))
    grouped = len(d["times"]) < d["points"]
    for j, k in enumerate(d["k_values"]):
        a2.bar(pos, 100 * d["share"][:, j], bottom=100 * bottom,
               color=COLORS[j % len(COLORS)], label=f"K={k}", gid=f"share:{k}")
        bottom += d["share"][:, j]
    every = max(1, math.ceil(len(pos) / 16))  # a readable number of labels (every bar ticked)
    a2.set_xticks(pos, [t if n % every == 0 or n == len(pos) - 1 else ""
                        for n, t in enumerate(d["time_labels"])],
                  rotation=45 if len(pos) > 8 else 0, fontsize=8)
    a2.set_xlabel("time" if grouped else "time point")
    a2.set_ylabel("spectra (%)")
    a2.set_title(("K in effect over time" + f" ({d['points']} time points in "
                  f"{len(d['times'])} groups)" if grouped else "K in effect at each time point")
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
    from .results import mark_provisional, provisional_note

    note = provisional_note(results)
    figures = {k: mark_provisional(f, note) for k, f in overview_figures(results).items()}
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
