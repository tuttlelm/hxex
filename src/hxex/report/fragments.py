# SPDX-License-Identifier: GPL-3.0-or-later
"""Fragment-ladder kinetics figure: D content against time, one row per ladder step.

Layout: the top row shows the precursor ions (``M``, ``M-NH3``)
of each condition on the left and the precursor's decay fits on the right; each further row is a
step of one ion series, in residue order: on the left the two fragments of the step in each
condition and their difference, on the right the difference with a fit of
``A exp(-k t) + C`` and its rate. A step whose difference holds several exchanging residues
(``block``) is drawn without a fit; a residue that no step covers keeps an empty row that says
why. With ``fit`` False (``hxex analyse`` unless ``[analysis.fragments] fit_rates = true``)
nothing is fitted and the right column shows the points only. Time is on a log axis in
seconds. With intrinsic rates (:mod:`hxex.kint`) and the
fraction the amides started from, the right column also shows, dashed in black, the
theoretical curve of the unprotected amides: the step's residue, or all the parent's amides in
the precursor row.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure

from .. import chem
from ..analysis.fragments import (
    FragmentUptake,
    LadderStep,
    Trace,
    _parent_starts,
    difference_trace,
    fit_decay,
    fragment_trace,
    ladder_steps,
)
from ..analysis.runs import display_labels
from ..analysis.runs import label as _label
from ..kint import IntrinsicRates, expected_uptake, intrinsic_rates
from . import hover

# per condition: (shorter fragment, longer fragment) and the difference
_PAIR_COLOURS = [("tab:green", "tab:red"), ("tab:purple", "tab:brown"),
                 ("tab:pink", "tab:olive"), ("tab:gray", "tab:cyan")]
_DIFF_COLOURS = ["tab:blue", "tab:orange", "tab:cyan", "tab:gray"]
# the residue-rate figure: per condition (and precursor charge)
_SOURCE_COLOURS = [*_DIFF_COLOURS, "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink",
                   "tab:olive"]
_EXTRA_COLOURS = ["#08306b", "#67000d", "#00441b", "#3f007d", "#7f2704"]
_MARKER = {"marker": "v", "s": 9, "linewidths": 0}
_ERRORS = {"fmt": "none", "elinewidth": 0.6, "alpha": 0.35, "capsize": 0}  # +-1 se, light


def _points(ax, trace, colour, label=None, alpha=None, rows: dict | None = None,
            **style) -> None:
    """``trace`` as points with light +-1 se error bars (where the error is known); with
    ``rows`` (the columns that name the trace), each point is a row of the data tab."""
    known = np.isfinite(trace.mean_se)
    if known.any():
        ax.errorbar(trace.time[known], trace.mean[known], yerr=trace.mean_se[known],
                    ecolor=colour, **{**_ERRORS, "alpha": _ERRORS["alpha"] * (alpha or 1.0)})
    dots = ax.scatter(trace.time, trace.mean, color=colour, label=label, alpha=alpha,
                      **(style or _MARKER))
    if rows is not None:
        hover.set_rows(dots, [{**rows, "time": float(t), "mean": float(m), "mean_se": float(e)}
                              for t, m, e in zip(trace.time, trace.mean, trace.mean_se,
                                                 strict=True)])
# closure splits: a colour and a marker each (10 colours, 7 markers: 70 distinct pairs)
_SPLIT_COLOURS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown",
                  "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
_SPLIT_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _span(start: int, parent: str) -> str:
    """The parent's residue range, ``11-18``, as the figures' file names give it: every figure
    names the parent it shows (owner, 2026-11-17)."""
    return f"{start}-{start + len(chem.parse_sequence(parent)) - 1}"


def _title_residues(residues: Sequence[str], first: int, last: int, side: str) -> str:
    """The parent's sequence up to (N-terminal ions) or from (C-terminal ions) the residues
    ``first``..``last`` (1-based), which are bracketed."""
    inner = "".join(residues[first - 1:last])
    if side == "N":
        return "".join(residues[:first - 1]) + f"({inner})"
    return f"({inner})" + "".join(residues[last:])


LADDER_PANEL = (3.6, 2.8)  # inches per panel of a ladder figure (two a row)


def ladder_kinetics_figure(
    uptake: FragmentUptake,
    ion_type: str,
    *,
    charge: int = 1,
    sample: str | None = None,
    parent: str | None = None,
    parent_charge: int | None = None,
    run: str | None = None,
    conditions: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    precursors: Sequence[str] = ("M", "M-NH3"),
    precursor_charge: int | None = None,
    min_time: float | None = None,
    fit: bool = True,
    panel: tuple[float, float] = LADDER_PANEL,
    right_traces: Callable[[LadderStep | None], Sequence[tuple[str, Trace, bool]]] | None = None,
    right_defaults: bool = True,
    kint: IntrinsicRates | None = None,
    steps_shown: tuple[int, int] | None = None,
    note: str = "",
) -> Figure:
    """The ladder of ``ion_type`` (fragment ``charge``) of one parent: see the module docstring.

    ``steps_shown`` (first, end): only those ladder steps (in order, from 0), under the
    precursor row, which every page repeats (a long ladder on pages of ``[analysis]
    page_size``, :func:`ladder_pages`); ``note`` is said in the precursor row's title
    (``page 2 of 3``).

    ``sample``, ``parent`` (its sequence), ``parent_charge`` and ``run`` (the state's run,
    :attr:`hxex.analysis.FragmentUptake.state_run`: '' for a merged sample) choose the series
    when the table holds several (each must be unique otherwise). ``conditions`` (default: all,
    sorted) are drawn in that order, named by ``labels`` (for example ``{'gentle': 'G'}``); a
    merged sample's runs are separate series, named ``run:condition`` where a condition occurs
    in several of them. ``precursors``
    are the ion types of the top row at ``precursor_charge`` (default: ``charge``). Points before
    ``min_time`` (s, exchange clock: the spectra's times) are drawn but not fitted.

    ``right_traces`` adds traces of your own to the right column: called with each row's
    :class:`~hxex.analysis.LadderStep` (None for the precursor row), it returns ``(label, trace,
    fit)`` tuples, each drawn in its own colour and, when ``fit``, with its decay fit. With
    ``right_defaults`` False the right column shows only those (not the precursor's fits and
    the differences).

    ``kint``: the parent's intrinsic rates (:func:`hxex.kint.intrinsic_rates`) in the sample's
    solvent; with its ``conditions.initial_fraction`` set, the theoretical curves are drawn
    (amides without a rate, such as a free N-terminal amine, are left out of them)."""
    pick, state, sample, parent, pz = _one(uptake, sample, parent, parent_charge, run)
    table, entry, steps = _ladder(uptake, pick, ion_type, charge)
    shown = display_labels(uptake.state_run, uptake.sample)[(state, sample)]
    conditions, names = _series(table, conditions, labels)
    residues = chem.parse_sequence(parent)
    side = chem.ion_terminus(ion_type)
    series = table.where(ion_type=ion_type, charge=charge)
    if steps_shown is not None:
        steps = steps[steps_shown[0]:steps_shown[1]]
    activation = ", ".join(sorted(set(table.activation[series].tolist()))) or ion_type
    named = f"{shown} {_span(int(table.parent_start[0]), parent)} (Z{pz})"
    head = f"{named}  {activation} {ion_type}_z{charge}"
    pre_z = charge if precursor_charge is None else precursor_charge

    fig = Figure(figsize=(2 * panel[0], panel[1] * (len(steps) + 1)), layout="constrained")
    axes = fig.subplots(len(steps) + 1, 2, squeeze=False)
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.grid(True, which="major", alpha=0.4)
        ax.set_ylabel("D-content")
    for ax in axes[-1]:
        ax.set_xlabel("Time, sec")

    # the precursor ions
    left, right = axes[0]
    left.set_title(f"{named}  {activation} M_z{pre_z}" + (f"  ({note})" if note else ""),
                   fontsize=9)
    # the data tab: each point (time, D, its error) named by step, trace and condition, and
    # each fitted curve's rate, amplitude and offset (owner, 2026-11-23)
    base = {"sample": shown, "parent": parent, "parent_charge": int(pz)}
    for j, cond in enumerate(conditions):
        colours = _PAIR_COLOURS[j % len(_PAIR_COLOURS)]
        for m, ion in enumerate(precursors):
            label = f"{ion}:{len(residues)}"
            tr = fragment_trace(table, label, run=cond[0], condition=cond[1], charge=pre_z)
            if len(tr):
                _points(left, tr, colours[m % 2], f"{ion} {names[cond]}", rows={
                    **base, "condition": names[cond], "step": "precursor", "trace": label,
                    "charge": int(pre_z)})
        tr = fragment_trace(table, f"M:{len(residues)}", run=cond[0], condition=cond[1],
                            charge=pre_z)
        if len(tr) and right_defaults:
            _fitted(right, tr, _DIFF_COLOURS[j % len(_DIFF_COLOURS)], f"M {names[cond]}",
                    min_time, fit, rows={**base, "condition": names[cond], "step": "precursor",
                                         "trace": f"M:{len(residues)} fit",
                                         "charge": int(pre_z)})
    _extras(right, right_traces, None, min_time)
    t_axis = table.time[table.time > 0]
    if kint is not None and kint.conditions.initial_fraction is not None:
        from ..fragments import fragment_amides

        held = fragment_amides(entry, "M", nterm_subtract=table.nterm_subtract,
                               rules=table.amide_rules)
        _intrinsic(right, kint, held, t_axis, "intrinsic, all amides")
    _legend(left)
    _legend(right)

    # the ladder steps
    for row, step in zip(axes[1:], steps, strict=True):
        left, right = row
        # the residues measured (all the step covers when it measures none)
        lo, hi = (min(step.residues), max(step.residues)) if step.residues else step.span
        first, last = lo - entry.start_seq + 1, hi - entry.start_seq + 1
        title = f"{head}  {_title_residues(residues, first, last, side)}-" + (
            f"{lo}" if lo == hi else f"{lo}–{hi}")
        left.set_title(title, fontsize=9)
        if step.larger is None:
            for ax in row:
                ax.text(0.5, 0.5, step.note, ha="center", va="center", transform=ax.transAxes,
                        color="0.4")
            continue
        where = {**base, "step": f"{lo}" if lo == hi else f"{lo}-{hi}",
                 "residues": ", ".join(map(str, step.residues)), "ion_type": ion_type,
                 "charge": int(charge)}
        for j, cond in enumerate(conditions):
            colours = _PAIR_COLOURS[j % len(_PAIR_COLOURS)]
            for m, frag in enumerate((step.smaller, step.larger)):
                if frag is None:
                    continue
                tr = fragment_trace(table, frag, run=cond[0], condition=cond[1], charge=charge)
                if len(tr):
                    _points(left, tr, colours[m],
                            f"{names[cond]}:{frag.replace(':', '_')}_z{charge}",
                            rows={**where, "condition": names[cond], "trace": frag})
            if step.smaller is not None:
                tr = difference_trace(table, step.larger, step.smaller, run=cond[0],
                                      condition=cond[1], charge=charge)
                colour = _DIFF_COLOURS[j % len(_DIFF_COLOURS)]
                if len(tr):
                    diff = f"Δ {step.larger} − {step.smaller}"
                    _points(left, tr, colour, f"Δ {names[cond]}",
                            rows={**where, "condition": names[cond], "trace": diff})
                    if right_defaults:
                        _fitted(right, tr, colour, f"Δ {names[cond]}", min_time,
                                fit and step.single,
                                rows={**where, "condition": names[cond],
                                      "trace": f"{diff} fit"})
        _extras(right, right_traces, step, min_time)
        if (step.single and kint is not None
                and kint.conditions.initial_fraction is not None):
            (residue,) = step.residues
            k = float(kint.rate[kint.at([residue])[0]])
            _intrinsic(right, kint, step.residues, t_axis, f"intrinsic, k = {k:.2e} s⁻¹")
        if not step.single:
            right.text(0.5, 0.5, f"{step.note or 'block'}: residues "
                       f"{', '.join(map(str, step.residues)) or 'none'}", ha="center",
                       va="center", transform=right.transAxes, color="0.4")
        _legend(left)
        _legend(right)
    positive = table.time[table.time > 0]
    if len(positive):  # one time axis for every panel, the empty ones too
        lo, hi = float(positive.min()), float(positive.max())
        for ax in axes.ravel():
            ax.set_xlim(lo / 1.15, hi * 1.15)
    return fig


def _ladder(uptake: FragmentUptake, pick, ion_type: str, charge: int):
    """The rows of one parent after time 0, its entry, and the ladder steps of ``ion_type`` at
    fragment ``charge``."""
    table = uptake.select(pick & (uptake.time > 0))  # not the undeuterated control (log axis)
    if not len(table):
        raise ValueError("no fragment spectra after time 0")
    entry = table.parent_entry()
    series = table.where(ion_type=ion_type, charge=charge)
    lengths = sorted(set(table.length[series].tolist()))
    return table, entry, ladder_steps(entry, ion_type, lengths,
                                      nterm_subtract=table.nterm_subtract,
                                      rules=table.amide_rules)



def ladder_pages(uptake: FragmentUptake, ion_type: str, *, charge: int = 1,
                 page_height: float | None = None, sample: str | None = None,
                 parent: str | None = None, parent_charge: int | None = None,
                 run: str | None = None) -> list[tuple[int, int]]:
    """The pages of a ladder figure (owner, 2026-11-04: "break fragment ladder figure into
    pages"): (first, end) steps of each, as many as fit under the precursor row in
    ``page_height`` inches (None: one page)."""
    pick = _one(uptake, sample, parent, parent_charge, run)[0]
    n = len(_ladder(uptake, pick, ion_type, charge)[2])
    per_page = n if page_height is None else max(1, int(page_height // LADDER_PANEL[1]) - 1)
    return [(a, min(a + per_page, n)) for a in range(0, n, max(1, per_page))] or [(0, 0)]


def _extras(ax, right_traces, step, min_time) -> None:
    """The caller's own traces of a row (see ``right_traces``)."""
    if right_traces is None:
        return
    for n, (label, trace, fit) in enumerate(right_traces(step) or ()):
        if len(trace):
            _fitted(ax, trace.after(1e-12), _EXTRA_COLOURS[n % len(_EXTRA_COLOURS)], label,
                    min_time, fit)


def _intrinsic(ax, kint, residues, times, label) -> None:
    """The theoretical curve of the unprotected amides of ``residues`` (dashed, black)."""
    pos = kint.at(residues)
    residues = [r for r, k in zip(residues, kint.rate[pos], strict=True) if np.isfinite(k)]
    if not residues or not len(times):
        return
    t = np.geomspace(float(times.min()), float(times.max()), 200)
    ax.plot(t, expected_uptake(kint, residues, t, kint.conditions.initial_fraction),
            color="black", ls="--", lw=1.0, label=label)


def _fitted(ax, trace, colour, label, min_time, fit, rows: dict | None = None) -> None:
    """``trace`` as points, with its decay fit (from ``min_time``) when ``fit`` (else the
    points carry the label): a resolved fit with its +-1 se band, an unresolved one dashed.
    With ``rows``, the fit is a row of the data tab too (its rate, amplitude, and offset)."""
    _points(ax, trace, colour, None if fit else label)  # its rows are the left panel's
    if not fit:
        return
    f = fit_decay(trace, min_time=min_time)
    if not f.ok:
        return
    t = np.geomspace(max(trace.after(min_time).time.min(), 1e-6), trace.time.max(), 200)
    fit_row = None if rows is None else {
        **rows, "rate": f.rate, "rate_se": f.rate_se, "amplitude": f.amplitude,
        "amplitude_se": f.amplitude_se, "offset": f.offset, "n": f.n, "resolved": f.resolved}
    if not f.resolved:
        (line,) = ax.plot(t, f.curve(t), color=colour, lw=1.0, ls="--",
                          label=f"{label}, rate not resolved")
    else:
        curve, se = f.curve(t), f.curve_se(t)
        if np.isfinite(se).all():  # +-1 se of the fitted curve
            ax.fill_between(t, curve - se, curve + se, color=colour, alpha=0.2, lw=0)
        (line,) = ax.plot(t, curve, color=colour, lw=1.5,
                          label=f"{label}, rate: {f.rate:.2e}±{f.rate_se:.2e} s⁻¹")
    if fit_row is not None:
        hover.set_row(line, fit_row)


def _legend(ax) -> None:
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6, loc="upper right", frameon=True, markerscale=1.2)


# --- every fragment of a ladder on one panel ---------------------------------------------------

TRACES_PANEL = (5.6, 3.4)  # inches: the panel and its legend
_FEW_POINTS = 30  # a trace with this many points or fewer is drawn with markers too
_TRACE_COLOURS = "viridis_r"  # shortest fragment light, longest dark
SERIES_GID = "series:"  # a legend name's gid: the fragment's series (``series:<data_id>``)


def _trace_panel(ax, table: FragmentUptake, shown: str, pair: tuple[str, str], name: str,
                 ion_type: str, charge: int, *, left_out: np.ndarray | None = None,
                 highlight: str | None = None, at: np.ndarray | None = None) -> int:
    """Every ``ion_type`` fragment at ``charge`` of the series ``pair`` (run, condition) of
    ``table`` (one parent, after time 0) on ``ax``; ``shown`` is the sample's display name,
    ``name`` the series'. ``at`` (the page's time points, sorted: :func:`_time_points`) places
    each time at its time point's position (evenly spaced) instead of at its time.
    ``left_out`` (why each row of ``table`` is left out, '' for kept) draws those rows as a
    faded dashed line without a band (owner, 2026-11-17: as the grey outlines of left-out
    points elsewhere); ``highlight`` (a fragment, ``c-1:4``) is drawn thicker. Each legend
    name's gid is ``series:<data_id>`` (the window selects that series on a click). The number
    of fragments drawn."""
    from matplotlib import colormaps

    rows = (table.where(ion_type=ion_type, charge=charge, condition=pair[1])
            & (np.asarray(table.run) == pair[0]))
    left = (np.zeros(len(table), dtype=bool) if left_out is None
            else np.asarray(left_out).astype(str) != "")
    lengths = sorted(set(table.length[rows].tolist()))
    if not lengths:
        return 0
    cmap = colormaps[_TRACE_COLOURS]
    colours = (cmap(np.linspace(0.15, 1.0, len(lengths))) if len(lengths) > 1
               else [cmap(0.6)])
    kept, gone = table.select(~left), table.select(left)
    reasons = np.asarray(left_out, dtype=object)[left] if left_out is not None else None
    ids = []

    def x(times):
        if at is None:
            return times
        # the nearest time point (a trace's averaged times need not equal the table's bits)
        return np.abs(np.asarray(times, float)[:, None] - at[None, :]).argmin(axis=1).astype(float)

    for length, colour in zip(lengths, colours, strict=True):
        fragment = f"{ion_type}:{length}"
        mine = rows & (table.fragment == fragment)
        ids.append(int(table.data_id[mine][0]))
        label = f"{ion_type}_{length}_z{charge}"
        wide = 2.8 if fragment == highlight else 1.4
        base = {"sample": shown, "condition": name, "parent": str(table.parent[0]),
                "parent_charge": int(table.parent_charge[0]), "fragment": fragment,
                "ion_type": ion_type, "length": int(length), "charge": int(charge)}
        tr = fragment_trace(kept, fragment, run=pair[0], condition=pair[1], charge=charge)
        if len(tr):
            known = np.isfinite(tr.mean_se)
            if known.any():
                ax.fill_between(x(tr.time), tr.mean - tr.mean_se, tr.mean + tr.mean_se,
                                where=known, color=colour, alpha=0.2, lw=0)
            (line,) = ax.plot(x(tr.time), tr.mean, color=colour, lw=wide, label=label,
                              marker="o" if len(tr) <= _FEW_POINTS else None, ms=3)
            hover.set_rows(line, [{**base, "time": float(t), "mean": float(m),
                                   "mean_se": float(e), "left_out": ""}
                                  for t, m, e in zip(tr.time, tr.mean, tr.mean_se, strict=True)])
        off = fragment_trace(gone, fragment, run=pair[0], condition=pair[1], charge=charge)
        if len(off):
            why = reasons[np.flatnonzero((gone.fragment == fragment) & (gone.charge == charge)
                                         & (gone.condition == pair[1]))]
            (dashed,) = ax.plot(x(off.time), off.mean, color=colour, lw=max(1.0, wide - 0.4),
                                ls="--", alpha=0.45, label=None if len(tr) else label,
                                marker="o" if len(off) <= _FEW_POINTS else None, ms=3,
                                mfc="none")
            hover.set_rows(dashed, [{**base, "time": float(t), "mean": float(m),
                                     "mean_se": float("nan"),
                                     "left_out": ", ".join(sorted(set(map(str, why))))}
                                    for t, m in zip(off.time, off.mean, strict=True)])
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7,
                       frameon=False, ncol=2 if len(lengths) > 16 else 1, borderaxespad=0)
    for text, data_id, length in zip(legend.get_texts(), ids, lengths, strict=True):
        text.set_gid(f"{SERIES_GID}{data_id}")
        hover.set_hover(text, f"{ion_type}_{length}_z{charge}: click to show this fragment's "
                        "series")
        if f"{ion_type}:{length}" == highlight:
            text.set_fontweight("bold")
    ax.grid(True, which="major", alpha=0.4)
    ax.set_xlabel("Time, sec" if at is None else "time point")
    ax.set_ylabel("D-content")
    return len(lengths)


def _check_scale(scale: str) -> None:
    if scale not in ("linear", "log", "index"):
        raise ValueError("scale must be 'linear', 'log' or 'index'")


def _time_points(table: FragmentUptake, scale: str) -> np.ndarray | None:
    """For ``scale`` ``'index'``, the times of ``table``, sorted: each drawn at its position
    among them (the time points evenly spaced, as the uptake page draws them); None for a
    time axis."""
    return np.unique(np.asarray(table.time, float)) if scale == "index" else None


def _set_time_axis(ax, scale: str, at: np.ndarray | None) -> None:
    """The time axis of a traces panel: ``scale``, or the time points by name (``at``)."""
    from ..io.times import format_time

    if at is None:
        ax.set_xscale(scale)
        return
    ax.set_xticks(range(len(at)), [format_time(float(t)) for t in at],
                  rotation=45 if len(at) > 8 else 0, fontsize=7)


def _pick_series(table, labels, condition, series_run):
    """The one series ``(run, condition)`` of ``table`` that ``condition`` and ``series_run``
    choose, and its display name."""
    pairs, names = _series(table, None, labels)
    if condition is not None:
        pairs = [p for p in pairs if p[1] == str(condition)]
    if series_run is not None:
        pairs = [p for p in pairs if p[0] == str(series_run)]
    if len(pairs) != 1:
        what = ", ".join(names[p] for p in pairs) or "none"
        raise ValueError(f"choose one condition of the parent ({what})")
    return pairs[0], names[pairs[0]]


def _traces_title(shown, table, pz, name, ion_type, charge) -> str:
    activation = ", ".join(sorted(set(table.activation[table.where(
        ion_type=ion_type, charge=charge)].tolist()))) or ion_type
    said = name if activation in name else f"{name}  {activation}"
    named = f"{shown} {_span(int(table.parent_start[0]), str(table.parent[0]))} (Z{pz})"
    return f"{named}  {said} {ion_type}_z{charge}"


def fragment_traces_figure(
    uptake: FragmentUptake,
    ion_type: str,
    *,
    charge: int = 1,
    condition: str | None = None,
    series_run: str | None = None,
    sample: str | None = None,
    parent: str | None = None,
    parent_charge: int | None = None,
    run: str | None = None,
    labels: Mapping[str, str] | None = None,
    scale: str = "log",
    panel: tuple[float, float] = TRACES_PANEL,
) -> Figure:
    """Every fragment of one ion series on one panel: deuterium against time, a line per
    fragment (light to dark from the shortest to the longest) with a pale +-1 se band, for one
    condition (gentle or harsh, say) of one parent at one precursor charge (owner, 2026-11-17,
    from their own figure of the gentle and harsh scrambling standards). Replicates at one time
    are averaged as in the ladder figure; the undeuterated control (time 0) is not drawn.

    ``sample``, ``parent``, ``parent_charge`` and ``run`` (the state's run) choose the parent as
    in :func:`ladder_kinetics_figure`; ``condition`` (and ``series_run``, the run of the series
    in a merged sample) the series, each needed only when the parent has several. ``labels``
    names the condition (``{'gentle': 'G'}``). ``scale``: the time axis, ``'log'`` or
    ``'linear'`` (``hxex analyse`` draws a kinetics run's on a linear axis: every scan has its
    own time), or ``'index'``, the time points evenly spaced and named. Each line carries its
    points as rows of the window's data table."""
    _check_scale(scale)
    pick, state, sample, parent, pz = _one(uptake, sample, parent, parent_charge, run)
    table = uptake.select(pick & (uptake.time > 0))
    if not len(table):
        raise ValueError("no fragment spectra after time 0")
    at = _time_points(table, scale)
    shown = display_labels(uptake.state_run, uptake.sample)[(state, sample)]
    pair, name = _pick_series(table, labels, condition, series_run)
    fig = Figure(figsize=panel, layout="constrained")
    ax = fig.add_subplot()
    if not _trace_panel(ax, table, shown, pair, name, ion_type, charge, at=at):
        raise ValueError(f"no {ion_type} fragments at charge {charge} in {name}")
    _set_time_axis(ax, scale, at)
    ax.set_title(_traces_title(shown, table, pz, name, ion_type, charge), fontsize=9)
    return fig


def fragment_traces_page(
    uptake: FragmentUptake,
    *,
    left_out: np.ndarray | None = None,
    highlight: str | None = None,
    sample: str | None = None,
    parent: str | None = None,
    parent_charge: int | None = None,
    run: str | None = None,
    labels: Mapping[str, str] | None = None,
    scale: str = "log",
    panel: tuple[float, float] = TRACES_PANEL,
) -> Figure:
    """Every fragment of one parent at one precursor charge, on panels of
    :func:`fragment_traces_figure`: a row per ion type and fragment charge, a column per
    condition (the window's first look at which fragments a ladder should use; owner,
    2026-11-17). ``left_out`` (why each row of ``uptake`` is left out, '' for kept;
    :func:`hxex.analysis.fragment_uptake_with_left_out`) draws those rows as faded dashed
    lines; ``highlight`` (a fragment, ``c-1:4``) is drawn thicker, its legend name bold.
    ``scale`` as in :func:`fragment_traces_figure`; with ``'index'`` every panel places the
    page's time points alike."""
    from ..analysis.fragments import fragment_ion_types

    _check_scale(scale)
    pick, state, sample, parent, pz = _one(uptake, sample, parent, parent_charge, run)
    later = pick & (uptake.time > 0)
    table = uptake.select(later)
    if not len(table):
        raise ValueError("no fragment spectra after time 0")
    at = _time_points(table, scale)
    left = None if left_out is None else np.asarray(left_out, dtype=object)[later]
    shown = display_labels(uptake.state_run, uptake.sample)[(state, sample)]
    pairs, names = _series(table, None, labels)
    series = [(ion, int(z)) for ion in fragment_ion_types(table)
              for z in sorted(set(table.charge[table.ion_type == ion].tolist()))]
    if not series:
        raise ValueError("no fragment ions of the parent after time 0")
    fig = Figure(figsize=(panel[0] * len(pairs), panel[1] * len(series)), layout="constrained")
    axes = fig.subplots(len(series), len(pairs), squeeze=False)
    for (ion, z), row in zip(series, axes, strict=True):
        for pair, ax in zip(pairs, row, strict=True):
            if _trace_panel(ax, table, shown, pair, names[pair], ion, z, left_out=left,
                            highlight=highlight, at=at):
                _set_time_axis(ax, scale, at)
                ax.set_title(_traces_title(shown, table, pz, names[pair], ion, z), fontsize=9)
            else:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"no {ion}_z{z} fragments in {names[pair]}", ha="center",
                        va="center", transform=ax.transAxes, color="0.4")
    return fig


# --- which fragments have data to fit -----------------------------------------------------------

# the states of FragmentCoverage, best first: (legend text, face, edge, hatch, text colour)
_COVERAGE_STYLE = {
    "all": ("every point kept", "#1f4e79", "#1f4e79", None, "white"),
    "enough": ("some left out, ≥ {n} kept", "#6fa3d2", "#1f4e79", None, "black"),
    "too_few": ("1-{m} points kept (too few to fit rates)", "#d9e6f2", "#1f4e79", None,
                "black"),
    "none": ("read, every point left out", "white", "#1f4e79", None, "#444444"),
    "not_read": ("targeted, never read", "white", "#999999", "////", "#444444"),
}
_COVERAGE_GAP = 0.12  # half the gap drawn at a cut, so two empty bars still show where it is


def _coverage_pairs(ion_types: Sequence[str]) -> list[tuple[str | None, str | None]]:
    """The N-terminal ion types of a parent with their complementary C-terminal ones, one pair
    per row of a block (``c-1`` with ``z-dot``): in name order, each N type with the first
    unpaired C type its partner letter allows; types without a partner alone."""
    from ..analysis.fragments import _PARTNERS

    n_types = sorted(t for t in ion_types if chem.ion_terminus(t) == "N")
    c_types = sorted(t for t in ion_types if chem.ion_terminus(t) == "C")
    pairs: list[tuple[str | None, str | None]] = []
    for a in n_types:
        b = next((t for t in c_types if _PARTNERS.get(a[0]) == t[0]), None)
        if b is not None:
            c_types.remove(b)
        pairs.append((a, b))
    return pairs + [(None, b) for b in c_types]


def _coverage_text(table, i: int) -> str:
    """What a bar says: the time points kept, as a share in a kinetics run."""
    if table.state[i] == "not_read":
        return ""
    if table.kinetics[i]:
        return f"{100 * int(table.kept[i]) / max(1, int(table.points[i])):.0f}%"
    return str(int(table.kept[i]))


def _coverage_bar(ax, table, i: int, left: float, right: float, y: float, side: str,
                  name: str, shown: str) -> None:
    """One fragment series' bar from ``left`` to ``right`` (data units) at row ``y``, its
    count at the cut end (``side``: the bar's ``'left'`` or ``'right'`` of the cut; ``'M'``
    at its right end), its hover lines and data-table row."""
    from matplotlib.patches import Rectangle

    _, face, edge, hatch, colour = _COVERAGE_STYLE[str(table.state[i])]
    rect = Rectangle((left, y - 0.4), right - left, 0.8, facecolor=face, edgecolor=edge,
                     hatch=hatch, linewidth=0.6)
    ax.add_patch(rect)
    text = _coverage_text(table, i)
    if text:
        at, align = ((right - 0.08, "right") if side in ("left", "M") else (left + 0.08, "left"))
        ax.text(at, y, text, ha=align, va="center", fontsize=5, color=colour, clip_on=True)
    label = f"{table.fragment[i]} {int(table.charge[i])}+"
    why = {k: int(v[i]) for k, v in table.reasons.items() if int(v[i])}
    lines = [label, name,
             "never read" if table.state[i] == "not_read" else
             f"{int(table.kept[i])} of {int(table.points[i])} time points kept"]
    if why:
        lines.append("left out: " + ", ".join(f"{k} {n}" for k, n in why.items()))
    hover.set_hover(rect, *lines)
    hover.set_row(rect, {
        "sample": shown, "parent": str(table.parent[i]), "parent_charge": int(
            table.parent_charge[i]), "condition": name, "fragment": str(table.fragment[i]),
        "charge": int(table.charge[i]), "points": int(table.points[i]),
        "kept": int(table.kept[i]), "state": str(table.state[i]),
        **{f"left_out_{k}": int(v[i]) for k, v in table.reasons.items()}})


def fragment_coverage_figure(
    coverage,
    *,
    sample: str | None = None,
    parent: str | None = None,
    parent_charge: int | None = None,
    run: str | None = None,
    labels: Mapping[str, str] | None = None,
) -> Figure:
    """Which fragment ions of one parent at one precursor charge kept enough spectra to fit
    (:func:`hxex.analysis.fragment_coverage`), an overall quality assessment (owner,
    2026-11-23; PLAN Next steps 41). A panel per condition, side by side. The x axis is the
    parent's residues: its sequence below, the protein's numbers above, marks at the residue
    boundaries. First the precursor ion at each charge (a bar over the parent); then a block
    per fragment charge with a row per cut: the N-terminal ion of that length left of the cut
    and its complementary C-terminal ion (the rest of the parent, at the same fragment charge)
    right of it, a small gap at the cut. The left axis names the N-terminal ions, the right
    axis the C-terminal ones. Every cut has its row, so what was never targeted or never read
    shows; a cut before a proline (never targeted) carries a red P on the proline. Each bar is
    coloured by its state and says how many time points it kept (a share in a kinetics run).

    ``sample``, ``parent``, ``parent_charge`` and ``run`` (the state's run) choose the parent
    as in :func:`ladder_kinetics_figure`; ``labels`` names the conditions."""
    from matplotlib.patches import Patch

    from ..analysis.fragments import COVERAGE_MIN_POINTS

    pick, state, sample, parent, pz = _one(coverage, sample, parent, parent_charge, run)
    table = coverage.select(pick)
    shown = display_labels(coverage.state_run, coverage.sample)[(state, sample)]
    residues = chem.parse_sequence(parent)
    n = len(residues)
    start = int(table.parent_start[0])
    pairs, names = _series(table, None, labels)
    index = {(str(r), str(c), str(f), int(z)): i for i, (r, c, f, z) in enumerate(zip(
        table.run, table.condition, table.fragment, table.charge, strict=True))}
    m_charges = sorted({int(z) for z, ion in zip(table.charge, table.ion_type, strict=True)
                        if chem.ion_terminus(str(ion)) == "M"})
    ions = sorted({str(t) for t in table.ion_type if chem.ion_terminus(str(t)) != "M"})
    blocks = [(pair, z) for pair in _coverage_pairs(ions)
              for z in sorted({int(c) for c, t in zip(table.charge, table.ion_type, strict=True)
                               if str(t) in pair})]
    rows: list[tuple] = [("M", z) for z in m_charges]
    rows += [("cut", pair, z, k) for pair, z in blocks for k in range(1, n)]
    if len(rows) == len(m_charges):
        raise ValueError(f"no fragment ions of {parent}")
    height = len(rows)
    fig = Figure(figsize=(1.6 + 3.0 * len(pairs), 1.3 + 0.15 * height), layout="constrained")
    axes = fig.subplots(1, len(pairs), sharey=True, squeeze=False)[0]
    edges = [x + 0.5 for x in range(n + 1)]
    for ax, (series_run, condition) in zip(axes, pairs, strict=True):
        name = names[(series_run, condition)]
        for y, row in enumerate(rows):
            if row[0] == "M":
                i = index.get((series_run, condition, f"M:{n}", row[1]))
                if i is not None:
                    _coverage_bar(ax, table, i, 0.5, n + 0.5, -y, "M", name, shown)
                continue
            _, (n_ion, c_ion), z, k = row
            if residues[k] == "P":  # the cut before a proline: neither ion is made
                ax.text(k + 1, -y, "P", ha="center", va="center", fontsize=6, color="#aa0000")
            for ion, length, left, right, side in (
                    (n_ion, k, 0.5, k + 0.5 - _COVERAGE_GAP, "left"),
                    (c_ion, n - k, k + 0.5 + _COVERAGE_GAP, n + 0.5, "right")):
                i = (index.get((series_run, condition, f"{ion}:{length}", z))
                     if ion is not None else None)
                if i is not None:
                    _coverage_bar(ax, table, i, left, right, -y, side, name, shown)
            following = rows[y + 1] if y + 1 < height else None
            if following is not None and following[1:3] != row[1:3]:
                ax.axhline(-y - 0.5, color="#999999", linewidth=0.6)  # the next block
        ax.set_xlim(0.5, n + 0.5)
        ax.set_xticks(range(1, n + 1), residues, fontsize=7)
        ax.set_xticks(edges, minor=True)
        ax.tick_params(axis="x", which="major", length=0)
        ax.tick_params(axis="x", which="minor", length=3, width=0.6, direction="out")
        # every residue numbered over its letter, without a mark (a mark at a residue's
        # centre read as a bar's end); the marks at the residue boundaries
        top = ax.secondary_xaxis("top")
        top.set_xticks(range(1, n + 1), [str(start + j) for j in range(n)],
                       fontsize=5.5 if n <= 16 else 5, rotation=0 if n <= 16 else 90)
        top.set_xticks(edges, minor=True)
        top.tick_params(which="major", length=0, pad=2)
        top.tick_params(which="minor", length=3, width=0.6, direction="out")
        for x in edges[1:-1]:
            ax.axvline(x, color="#f0f0f0", linewidth=0.5, zorder=0)
        points = sorted({int(p) for p, kin, c, r in zip(
            table.points, table.kinetics, table.condition, table.run, strict=True)
            if str(c) == condition and str(r) == series_run and not kin})
        said = (f"{points[-1]} time points" if points else "kinetics: share of scans kept")
        ax.set_title(f"{name}  ({said})", fontsize=8)

    def label(row, side: str) -> str:
        if row[0] == "M":
            return f"M {row[1]}+" if side == "left" else ""
        _, (n_ion, c_ion), z, k = row
        ion, length = (n_ion, k) if side == "left" else (c_ion, n - k)
        return f"{ion}:{length} {z}+" if ion is not None else ""

    axes[0].set_ylim(-height + 0.4, 0.6)
    axes[0].set_yticks([-y for y in range(height)], [label(r, "left") for r in rows],
                       fontsize=5.5)
    right = axes[-1].secondary_yaxis("right")
    right.set_yticks([-y for y in range(height)], [label(r, "right") for r in rows],
                     fontsize=5.5)
    handles = [Patch(facecolor=face, edgecolor=edge, hatch=hatch,
                     label=text.format(n=COVERAGE_MIN_POINTS, m=COVERAGE_MIN_POINTS - 1))
               for text, face, edge, hatch, _ in _COVERAGE_STYLE.values()]
    handles.append(Patch(facecolor="none", edgecolor="none",
                         label="P: no cut before a proline; numbers: time points kept"))
    fig.legend(handles=handles, loc="outside lower center", ncols=2, fontsize=7,
               frameon=False)
    fig.suptitle(f"{shown}  {parent} ({_span(start, parent)}), precursor {pz}+", fontsize=9)
    return fig


def _kinetics_rows(results, uptake: FragmentUptake) -> np.ndarray:
    """Whether each row of ``uptake`` came from a kinetics run (every scan its own time:
    :func:`hxex.io.mzml.kinetics_spectra`)."""
    from ..analysis.runs import RunSet
    from ..io.mzml import kinetics_spectra

    runs = results.runs if isinstance(results, RunSet) else {"": results}
    out = np.zeros(len(uptake), dtype=bool)
    for name, run_results in runs.items():
        rows = (np.asarray(uptake.run) == name) if isinstance(results, RunSet) else (
            np.ones(len(uptake), dtype=bool))
        if not rows.any():
            continue
        settings = getattr(run_results, "settings", None)
        source = getattr(settings, "spectra_source", None)
        out[rows] = kinetics_spectra(source, [str(s) for s in uptake.spectrum_id[rows]])
    return out


def _file_part(text: str) -> str:
    """``text`` as part of a file name."""
    import re

    return re.sub(r"[^\w.+-]+", "_", str(text)).strip("_") or "condition"


# --- closure and per-residue rates -------------------------------------------------------------

_ION_MARKERS = {"c": "v", "z": "^", "a": "s", "x": "D", "b": "<", "y": ">"}


def _one(table, sample, parent, parent_charge, run=None):
    """The rows of one state (``run``: its ``state_run``, '' for a merged sample), ``sample``,
    ``parent`` and ``parent_charge`` of ``table``, each unique: ``(rows, state run, sample,
    parent, parent charge)``."""
    pick = table.where(sample=sample, parent=parent, parent_charge=parent_charge)
    if run is not None:
        pick &= table.state_run == run
    out = []
    for name in ("state_run", "sample", "parent", "parent_charge"):
        values = np.unique(getattr(table, name)[pick])
        if len(values) != 1:
            what = "runs" if name == "state_run" else f"{name} values"
            raise ValueError(f"the table holds {len(values)} {what}; choose one "
                             f"({', '.join(map(str, values[:5]))})")
        out.append(values[0])
    return pick, str(out[0]), str(out[1]), str(out[2]), int(out[3])


class _Picked:
    """The ``run`` and ``condition`` columns of some rows, for :func:`_series`."""

    def __init__(self, run, condition):
        self.run, self.condition = run, condition


def _series(table, conditions, labels):
    """The ``(run, condition)`` series of one state's rows, in the order of ``conditions`` (a
    list of condition names; default: all, sorted), and their display names: the run's own
    label of the condition when ``labels`` has one (``{'etd': {'gentle': 'ETD G'}}``), else the
    condition's label (or the condition), prefixed ``run:`` where the condition occurs in
    several runs."""
    from ..analysis.runs import sample_setting

    pairs = sorted({(str(r), str(c)) for r, c in zip(table.run, table.condition, strict=True)})
    if conditions is not None:
        wanted = [str(c) for c in conditions]
        pairs = [p for c in wanted for p in pairs if p[1] == c]
    shown = display_labels([p[0] for p in pairs], [p[1] for p in pairs])
    names = {}
    for run, cond in pairs:
        own = (labels or {}).get(run) if run else None
        if isinstance(own, Mapping) and cond in own:
            names[(run, cond)] = str(own[cond])
            continue
        name = sample_setting(labels, "", cond) or cond
        names[(run, cond)] = _label(run, name) if shown[(run, cond)] != cond else name
    return pairs, names


def closure_figure(
    uptake: FragmentUptake,
    n_ion: str = "c-1",
    c_ion: str = "z-dot",
    *,
    sample: str | None = None,
    parent: str | None = None,
    parent_charge: int | None = None,
    run: str | None = None,
    conditions: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    n_charge: int = 1,
    c_charge: int = 1,
    precursor_charge: int | None = None,
    min_time: float | None = None,
    tolerance: float = 0.25,
    panel: tuple[float, float] = (4.2, 3.0),
    parents=None,
) -> Figure:
    """The closure test against time (:func:`hxex.analysis.closure_trace`): for every split
    ``k``, ``D(n_ion k) + D(c_ion n-k) - D(precursor)``, one panel per condition, a colour and
    a marker per split, with the band of ``tolerance`` around 0. Points before
    ``min_time`` are drawn fainter (they are left out of :func:`hxex.analysis.closure_table`).
    The precursor is the parent's MS1 spectrum where ``parents``
    (:func:`hxex.analysis.parent_uptake`) holds it, else its ion in the fragment scans; the
    y axis says which."""
    from ..analysis.fragments import closure_reference, closure_trace

    pick, state, sample, parent, pz = _one(uptake, sample, parent, parent_charge, run)
    table = uptake.select(pick & (uptake.time > 0))
    shown = display_labels(uptake.state_run, uptake.sample)[(state, sample)]
    conditions, names = _series(table, conditions, labels)
    n = len(chem.parse_sequence(parent))
    have = set(table.fragment.tolist())
    splits = [k for k in range(1, n) if f"{n_ion}:{k}" in have and f"{c_ion}:{n - k}" in have]
    if not splits:
        raise ValueError(f"no {n_ion} and {c_ion} pairs of {parent}")
    fig = Figure(figsize=(panel[0] * len(conditions), panel[1]), layout="constrained")
    axes = fig.subplots(1, len(conditions), squeeze=False, sharey=True)[0]
    mz = pz if precursor_charge is None else precursor_charge
    references = set()
    for ax, cond in zip(axes, conditions, strict=True):
        references.add(closure_reference(table, parents, run=cond[0], condition=cond[1]))
        ax.set_xscale("log")
        ax.axhspan(-tolerance, tolerance, color="0.92", zorder=0)
        ax.axhline(0.0, color="0.5", lw=0.8)
        for m, k in enumerate(splits):
            tr = closure_trace(table, f"{n_ion}:{k}", f"{c_ion}:{n - k}", n_charge=n_charge,
                               c_charge=c_charge, precursor_charge=precursor_charge,
                               parents=parents, run=cond[0], condition=cond[1])
            if not len(tr):
                continue
            colour = _SPLIT_COLOURS[m % len(_SPLIT_COLOURS)]
            style = {"marker": _SPLIT_MARKERS[m % len(_SPLIT_MARKERS)], "s": 10,
                     "linewidths": 0}
            early = tr.time < (min_time if min_time is not None else -np.inf)
            split = {"sample": shown, "parent": parent, "parent_charge": int(pz),
                     "condition": names[cond], "split": f"{n_ion}_{k} + {c_ion}_{n - k}",
                     "n_fragment": f"{n_ion}:{k}", "c_fragment": f"{c_ion}:{n - k}"}
            _points(ax, Trace(tr.time[~early], tr.mean[~early], tr.mean_se[~early]), colour,
                    f"{n_ion}_{k} + {c_ion}_{n - k}", 0.8,
                    rows={**split, "before_min_time": False}, **style)
            _points(ax, Trace(tr.time[early], tr.mean[early], tr.mean_se[early]), colour,
                    None, 0.25, rows={**split, "before_min_time": True}, **style)
        ax.set_title(f"{shown} {_span(int(table.parent_start[0]), parent)} (Z{pz})  "
                     f"{names[cond]}", fontsize=9)
        ax.set_xlabel("Time, sec")
        ax.grid(True, which="major", alpha=0.4)
    handles, names = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, loc="outside right center", fontsize=7, markerscale=1.5)
    reference = (f"MS1_z{pz}" if references == {"MS1"} else f"M_z{mz}" if references == {"M"}
                 else f"MS1 or M_z{pz}")
    axes[0].set_ylabel(f"{n_ion} + {c_ion} − {reference} (D)")
    return fig


def residue_rates_figure(
    rates,
    *,
    sample: str | None = None,
    parent: str | None = None,
    parent_charge: int | None = None,
    run: str | None = None,
    labels: Mapping[str, str] | None = None,
    panel: tuple[float, float] = (7.0, 2.6),
) -> Figure:
    """The rates of every single-residue ladder step (:func:`hxex.analysis.ladder_rates`) by
    residue: on top the rate (log axis, with its standard error), below the fitted amplitude
    (the deuterium that exchanges over the time range, with its standard error). Ion types are
    told apart by marker, conditions by colour; a fragment charge other than 1 is named in the
    legend. Every precursor charge of the parent is drawn unless ``parent_charge`` chooses one;
    with several, each condition and precursor charge has a colour of its own (``Z3`` in the
    legend), so that runs of one sample on different precursors (two activations, say) are
    compared in one figure. Unresolved fits (the decay not shown by the time range) are drawn as
    open markers along the bottom of the rate panel and have no amplitude; the rate axis spans
    the rates whose standard error is below the rate (others may reach beyond it). When the
    rates carry intrinsic rates (``kint``, :func:`hxex.analysis.ladder_rates` with exchange
    conditions), each residue's is drawn as a black bar on the rate panel and a third panel
    shows the protection factors (``kint / rate``, log axis, with the +-1 se range of its log10;
    unresolved fits left out)."""
    pick = rates.where(sample=sample, parent=parent, parent_charge=parent_charge)
    if run is not None:
        pick &= rates.state_run == run
    for name in ("state_run", "sample", "parent"):
        values = np.unique(getattr(rates, name)[pick])
        if len(values) != 1:
            what = "runs" if name == "state_run" else f"{name} values"
            raise ValueError(f"the table holds {len(values)} {what}; choose one "
                             f"({', '.join(map(str, values[:5]))})")
    state, sample = str(rates.state_run[pick][0]), str(rates.sample[pick][0])
    parent = str(rates.parent[pick][0])
    shown = display_labels(rates.state_run, rates.sample)[(state, sample)]
    charges = sorted(set(rates.parent_charge[pick].tolist()))
    residues = chem.parse_sequence(parent)
    start = int(rates.parent_start[pick][0])
    with_kint = rates.kint is not None and bool(np.isfinite(rates.kint[pick]).any())
    n_panels = 3 if with_kint else 2
    fig = Figure(figsize=(panel[0], n_panels * panel[1]), layout="constrained")
    axes = fig.subplots(n_panels, 1, sharex=True)
    top, bottom = axes[0], axes[1]
    series = sorted({(str(r), str(c), int(pz), str(i), int(z)) for r, c, pz, i, z in zip(
        rates.run[pick], rates.condition[pick], rates.parent_charge[pick], rates.ion_type[pick],
        rates.charge[pick], strict=True)},
        key=lambda s: (s[0], s[1], s[2], chem.ion_terminus(s[3]) != "N", s[3], s[4]))
    _, names = _series(_Picked(rates.run[pick], rates.condition[pick]), None, labels)
    sources = sorted({s[:3] for s in series})
    width = 0.6 / max(1, len(series))
    for m, (run_, cond, pz, ion, z) in enumerate(series):
        rows = pick & rates.where(run=run_, condition=cond, parent_charge=pz, ion_type=ion,
                                  charge=z)
        x = rates.residue[rows] + (m - (len(series) - 1) / 2) * width
        colour = _SOURCE_COLOURS[sources.index((run_, cond, pz)) % len(_SOURCE_COLOURS)]
        marker = _ION_MARKERS.get(ion[0], "o")
        name = (names[(run_, cond)] + (f" Z{pz}" if len(charges) > 1 else "")
                + f" {ion}" + (f" z{z}" if z != 1 else ""))
        ok = rates.resolved[rows]
        top.errorbar(x[ok], rates.rate[rows][ok], yerr=rates.rate_se[rows][ok], fmt=marker,
                     color=colour, ms=5, capsize=2, lw=0.8, label=name)
        top.scatter(x[~ok], np.zeros(int((~ok).sum())), marker=marker, facecolors="none",
                    edgecolors=colour, s=25, transform=top.get_xaxis_transform(), clip_on=False)
        bottom.errorbar(x[ok], rates.amplitude[rows][ok], yerr=rates.amplitude_se[rows][ok],
                        fmt=marker, color=colour, ms=5, capsize=2, lw=0.8)
        if with_kint:
            p = rates.protection[rows][ok]
            spread = 10.0 ** rates.log10_protection_se[rows][ok]  # +-1 se in log10
            axes[2].errorbar(x[ok], p, yerr=[p - p / spread, p * spread - p], fmt=marker,
                             color=colour, ms=5, capsize=2, lw=0.8)
    if with_kint:
        known = pick & np.isfinite(rates.kint)
        ks = sorted({(int(r), float(k)) for r, k in zip(rates.residue[known], rates.kint[known],
                                                         strict=True)})
        top.hlines([k for _, k in ks], [r - 0.4 for r, _ in ks], [r + 0.4 for r, _ in ks],
                   color="black", lw=1.5, label="intrinsic")
        pf = axes[2]
        pf.set_yscale("log")
        pf.axhline(1.0, color="0.5", lw=0.8)
        pf.set_ylabel("protection factor")
        pf.grid(True, axis="y", alpha=0.4)
    # the data tab: the rows of ladder_rates.csv this figure draws (owner, 2026-11-23)
    hover.set_table(top, rates.columns(), keep=pick)
    top.set_yscale("log")
    # the axis follows the rates known to better than their own size
    good = pick & rates.resolved & (rates.rate_se < rates.rate)
    if good.any():
        values = rates.rate[good]
        if with_kint:
            values = np.concatenate([values, rates.kint[pick][np.isfinite(rates.kint[pick])]])
        lo, hi = float(values.min()), float(values.max())
        top.set_ylim(lo / 3, hi * 3)
    top.set_ylabel("rate (s⁻¹)")
    bottom.set_ylabel("amplitude (D)")
    bottom.axhline(0.0, color="0.5", lw=0.8)
    ticks = list(range(start, start + len(residues)))
    last = axes[-1]
    last.set_xticks(ticks, [f"{residues[t - start]}{t}" for t in ticks],
                      fontsize=7 if len(ticks) > 30 else 8,
                      rotation=90 if len(ticks) > 30 else 0)
    last.set_xlim(start - 0.6, start + len(residues) - 0.4)
    for ax in (top, bottom):
        ax.grid(True, axis="y", alpha=0.4)
    top.set_title(f"{shown} {_span(start, parent)} {parent} (Z{'/'.join(map(str, charges))})  "
                  "single-residue rates"
                  + ("  (open: not resolved)" if not rates.resolved[pick].all() else ""),
                  fontsize=9)
    if top.get_legend_handles_labels()[0]:
        top.legend(fontsize=7, loc="best", ncol=max(1, len(series) // 4))
    return fig


def _has_fragment_series(results) -> bool:
    """Whether ``results`` were read from scan files with fragment targets."""
    from ..io.mzml import SCAN_READERS, mzml_series_targets

    settings = getattr(results, "settings", None)
    source = getattr(settings, "spectra_source", None)
    if not source or source.get("reader") not in SCAN_READERS or "series" not in source:
        return False
    return any("fragment" in t.entry.extra for t in mzml_series_targets(source).values())


def fragment_outputs(results, config, *, quiet: bool = False,
                     page_size: tuple[float, float] | None = None, cancel=None):
    """The fragment part of an analysis (:class:`hxex.analysis.FragmentAnalysis` ``config``):
    tables (name -> columns) and figures (file stem -> function drawing it), or empty ones when
    the run has no fragment series. Per parent (``<sample>_<range>_z<charge>``):
    ``fragment_coverage_...`` (which fragments kept enough spectra to fit),
    ``ladder_..._<ion>_z<charge>``, ``fragments_..._<condition>_<ion>_z<charge>`` (every
    fragment of the series on one panel), ``closure_..._<n ion>+<c ion>`` and
    ``residue_rates_...``;
    tables ``fragment_uptake``, ``fragment_coverage``, ``fragment_left_out`` (the spectra
    ``config.leave_out_flags`` left out, with the flags), ``ladder_rates``,
    ``ladder_residues`` and ``closure``. A parent read at several precursor charges has one
    ``residue_rates_<sample>_<range>`` (no charge) for all of them. The rate fits are opt-in
    (``config.fit_rates``): without them there are no ``ladder_rates``, ``ladder_residues``
    or ``residue_rates_...``, and the ladder figures draw no fitted curves (no decay is
    fitted at all).
    ``results`` may be a
    :class:`hxex.analysis.RunSet` (:func:`hxex.analysis.fragment_uptake`): the tables' ``run``
    column then names each row's run, and the figures of an unmerged sample found in several
    runs are named ``<run>_<sample>_...``. ``page_size`` (``[analysis] page_size``): a long
    ladder goes on over pages ``ladder_..._2``, each under its own precursor row. ``cancel``
    (:data:`hxex.hooks.CancelHook`) is asked before the uptake, between the parents of the
    rate fits and between the closure tables, so the window's Stop ends the minutes of ladder
    fits (audit of 2026-11-15, E open question 2)."""
    from ..analysis.fragments import (
        _PARTNERS,
        closure_table,
        fragment_coverage,
        fragment_ion_types,
        fragment_left_out,
        fragment_uptake,
        ladder_rates,
        ladder_residues,
        parent_uptake,
    )
    from ..analysis.runs import RunSet

    runs = results.runs if isinstance(results, RunSet) else {"": results}
    with_fragments = [name for name, r in runs.items() if _has_fragment_series(r)]
    if not with_fragments:
        return {}, []  # no run read from scan files with fragment targets
    without_k1 = [name for name in with_fragments if 1 not in runs[name].fit.fits]
    if without_k1:
        import logging

        logging.getLogger(__name__).warning(
            "no fragment outputs: fragment_uptake needs the K = 1 fits (fit with k_min = 1)%s",
            f"; run(s) {', '.join(without_k1)}" if any(without_k1) else "")
        return {}, []
    from ..hooks import check_cancel

    check_cancel(cancel)
    flags = config.leave_out_flags  # [analysis.fragments] leave_out_flags
    uptake = fragment_uptake(results, leave_out_flags=flags)  # any other error is a real one
    if not len(uptake):
        return {}, []
    ions = list(config.ion_types) or fragment_ion_types(uptake)
    fit_rates = config.fit_rates
    check_cancel(cancel)
    rates = ladder_rates(uptake, ion_types=ions, charges=config.charges,
                         min_time=config.min_time,
                         conditions=config.exchange if config.kint else None,
                         max_rate_error=config.max_rate_error,
                         cancel=cancel) if fit_rates else None
    pairs = [(a, b) for a in ions for b in ions
             if chem.ion_terminus(a) == "N" and chem.ion_terminus(b) == "C"
             and _PARTNERS[a[0]] == b[0]]
    parents = parent_uptake(results)  # the closure's reference where the run read the parents
    closures = []
    for a, b in pairs:
        check_cancel(cancel)
        closures.append(closure_table(uptake, a, b, precursor_charge=config.precursor_charge,
                                      min_time=config.min_time, tolerance=config.tolerance,
                                      parents=parents))
    tables = {"fragment_uptake": uptake.columns()}
    coverage = fragment_coverage(results, leave_out_flags=flags)
    tables["fragment_coverage"] = coverage.columns()
    left = fragment_left_out(results, leave_out_flags=flags)
    if left and len(left["reason"]):  # the spectra left out, and why (not the rejected)
        tables["fragment_left_out"] = left
        n_flagged = int(np.sum(left["reason"] != "excluded"))
        if n_flagged and not quiet:
            import logging

            logging.getLogger(__name__).info(
                "%d fragment spectra left out for their quality flags (%s; [analysis.fragments] "
                "leave_out_flags): fragment_left_out.csv", n_flagged, ", ".join(flags))
    if rates is not None:
        tables |= {"ladder_rates": rates.columns(),
                   "ladder_residues": ladder_residues(rates).columns()}
    if closures:
        cols = [c.columns() for c in closures]
        tables["closure"] = {name: np.concatenate([c[name] for c in cols]) for name in cols[0]}
    flagged = int(sum(c.flagged.sum() for c in closures)) if closures else 0
    if flagged and not quiet:
        import logging

        logging.getLogger(__name__).warning(
            "%d fragment pair(s) miss the precursor by more than %g D (closure.csv)",
            flagged, config.tolerance)
    from ..analysis.runs import sample_setting

    jobs = []
    kinetics = _kinetics_rows(results, uptake)  # linear time for their traces
    state_run = uptake.state_run
    shown = display_labels(state_run, uptake.sample)
    first = _parent_starts(uptake)
    groups = sorted({(str(u), str(s), str(p), int(z)) for u, s, p, z in zip(
        state_run, uptake.sample, uptake.parent, uptake.parent_charge, strict=True)},
        key=lambda g: (g[0], g[1], first[g[2]], g[2:]))  # parents by position
    for state, sample, parent, pz in groups:
        rows = uptake.where(sample=sample, parent=parent, parent_charge=pz) & (state_run == state)
        start = int(uptake.parent_start[rows][0])
        name = shown[(state, sample)]
        stem = f"{name}_{start}-{start + len(chem.parse_sequence(parent)) - 1}_z{pz}"
        sel = {"run": state, "sample": sample, "parent": parent, "parent_charge": pz}
        jobs.append((f"fragment_coverage_{stem}", lambda sel=sel: fragment_coverage_figure(
            coverage, labels=config.labels, **sel)))  # which fragments have data to fit
        solvent = sample_setting(config.exchange, state, sample) if config.kint else None
        kint = (intrinsic_rates(parent, solvent, start=start) if solvent is not None
                else None)
        for ion in ions:
            for z in config.charges:
                if (rows & uptake.where(ion_type=ion, charge=z) & (uptake.time > 0)).any():
                    # every fragment of the series on one panel, one figure per condition
                    # (the first look, before the ladder)
                    have = rows & uptake.where(ion_type=ion, charge=z) & (uptake.time > 0)
                    for r, c in sorted({(str(a), str(b)) for a, b in zip(
                            uptake.run[have], uptake.condition[have], strict=True)}):
                        these = have & (uptake.run == r) & (uptake.condition == c)
                        scale = "linear" if kinetics[these].all() else "log"
                        tag = _file_part(c if state or not r else f"{r}_{c}")
                        jobs.append((
                            f"fragments_{stem}_{tag}_{ion}_z{z}",
                            lambda ion=ion, z=z, sel=sel, r=r, c=c, scale=scale:
                            fragment_traces_figure(uptake, ion, charge=z, condition=c,
                                                   series_run=r, labels=config.labels,
                                                   scale=scale, **sel)))
                    pages = ladder_pages(uptake, ion, charge=z, page_height=(
                        None if page_size is None else page_size[1]), **sel)
                    for n, shown_steps in enumerate(pages):
                        note = f"page {n + 1} of {len(pages)}" if len(pages) > 1 else ""
                        jobs.append((
                            f"ladder_{stem}_{ion}_z{z}" + (f"_{n + 1}" if n else ""),
                            lambda ion=ion, z=z, sel=sel, kint=kint, shown_steps=shown_steps,
                            note=note: ladder_kinetics_figure(
                                uptake, ion, charge=z, labels=config.labels,
                                min_time=config.min_time, kint=kint, steps_shown=shown_steps,
                                note=note, fit=fit_rates, **sel)))
        for a, b in pairs:  # pairs at labelled times: the closure figure draws only those
            have = set(uptake.fragment[rows & (uptake.time > 0)].tolist())
            n = len(chem.parse_sequence(parent))
            if any(f"{a}:{k}" in have and f"{b}:{n - k}" in have for k in range(1, n)):
                jobs.append((f"closure_{stem}_{a}+{b}", lambda a=a, b=b, sel=sel: closure_figure(
                    uptake, a, b, labels=config.labels, min_time=config.min_time,
                    tolerance=config.tolerance, precursor_charge=config.precursor_charge,
                    parents=parents, **sel)))
    # the rates: one figure per parent, its precursor charges together
    for state, sample, parent in (sorted({g[:3] for g in groups},
                                         key=lambda g: (g[0], g[1], first[g[2]], g[2]))
                                  if rates is not None else ()):
        have = rates.where(sample=sample, parent=parent) & (rates.state_run == state)
        if not have.any():
            continue
        start = int(rates.parent_start[have][0])
        charges = sorted(set(rates.parent_charge[have].tolist()))
        name = shown[(state, sample)]
        stem = f"{name}_{start}-{start + len(chem.parse_sequence(parent)) - 1}" + (
            f"_z{charges[0]}" if len(charges) == 1 else "")
        jobs.append((f"residue_rates_{stem}",
                     lambda sel=(state, sample, parent): residue_rates_figure(
                         rates, labels=config.labels, run=sel[0], sample=sel[1], parent=sel[2])))
    return tables, jobs
