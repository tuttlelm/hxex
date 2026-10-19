# SPDX-License-Identifier: GPL-3.0-or-later
"""Fragment-ladder kinetics figure: D content against time, one row per ladder step.

Layout: the top row shows the precursor ions (``M``, ``M-NH3``)
of each condition on the left and the precursor's decay fits on the right; each further row is a
step of one ion series, in residue order: on the left the two fragments of the step in each
condition and their difference, on the right the difference with a fit of
``A exp(-k t) + C`` and its rate. A step whose difference holds several exchanging residues
(``block``) is drawn without a fit; a residue that no step covers keeps an empty row that says
why. Time is on a log axis in seconds. With intrinsic rates (:mod:`hxex.kint`) and the
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
    difference_trace,
    fit_decay,
    fragment_trace,
    ladder_steps,
)
from ..analysis.runs import display_labels
from ..analysis.runs import label as _label
from ..kint import IntrinsicRates, expected_uptake, intrinsic_rates

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


def _points(ax, trace, colour, label=None, alpha=None, **style) -> None:
    """``trace`` as points with light +-1 se error bars (where the error is known)."""
    known = np.isfinite(trace.mean_se)
    if known.any():
        ax.errorbar(trace.time[known], trace.mean[known], yerr=trace.mean_se[known],
                    ecolor=colour, **{**_ERRORS, "alpha": _ERRORS["alpha"] * (alpha or 1.0)})
    ax.scatter(trace.time, trace.mean, color=colour, label=label, alpha=alpha,
               **(style or _MARKER))
# closure splits: a colour and a marker each (10 colours, 7 markers: 70 distinct pairs)
_SPLIT_COLOURS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown",
                  "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
_SPLIT_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _title_residues(residues: Sequence[str], first: int, last: int, side: str) -> str:
    """The parent's sequence up to (N-terminal ions) or from (C-terminal ions) the residues
    ``first``..``last`` (1-based), which are bracketed."""
    inner = "".join(residues[first - 1:last])
    if side == "N":
        return "".join(residues[:first - 1]) + f"({inner})"
    return f"({inner})" + "".join(residues[last:])


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
    panel: tuple[float, float] = (3.6, 2.8),
    right_traces: Callable[[LadderStep | None], Sequence[tuple[str, Trace, bool]]] | None = None,
    right_defaults: bool = True,
    kint: IntrinsicRates | None = None,
) -> Figure:
    """The ladder of ``ion_type`` (fragment ``charge``) of one parent: see the module docstring.

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
    table = uptake.select(pick & (uptake.time > 0))  # not the undeuterated control (log axis)
    if not len(table):
        raise ValueError("no fragment spectra after time 0")
    shown = display_labels(uptake.state_run, uptake.sample)[(state, sample)]
    conditions, names = _series(table, conditions, labels)
    entry = table.parent_entry()
    residues = chem.parse_sequence(parent)
    side = chem.ion_terminus(ion_type)
    series = table.where(ion_type=ion_type, charge=charge)
    lengths = sorted(set(table.length[series].tolist()))
    steps = ladder_steps(entry, ion_type, lengths, nterm_subtract=table.nterm_subtract,
                         rules=table.amide_rules)
    activation = ", ".join(sorted(set(table.activation[series].tolist()))) or ion_type
    head = f"{shown} (Z{pz})  {activation} {ion_type}_z{charge}"
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
    left.set_title(f"{shown} (Z{pz})  {activation} M_z{pre_z}", fontsize=9)
    for j, cond in enumerate(conditions):
        colours = _PAIR_COLOURS[j % len(_PAIR_COLOURS)]
        for m, ion in enumerate(precursors):
            label = f"{ion}:{len(residues)}"
            tr = fragment_trace(table, label, run=cond[0], condition=cond[1], charge=pre_z)
            if len(tr):
                _points(left, tr, colours[m % 2], f"{ion} {names[cond]}")
        tr = fragment_trace(table, f"M:{len(residues)}", run=cond[0], condition=cond[1],
                            charge=pre_z)
        if len(tr) and right_defaults:
            _fitted(right, tr, _DIFF_COLOURS[j % len(_DIFF_COLOURS)], f"M {names[cond]}",
                    min_time, fit)
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
        for j, cond in enumerate(conditions):
            colours = _PAIR_COLOURS[j % len(_PAIR_COLOURS)]
            for m, frag in enumerate((step.smaller, step.larger)):
                if frag is None:
                    continue
                tr = fragment_trace(table, frag, run=cond[0], condition=cond[1], charge=charge)
                if len(tr):
                    _points(left, tr, colours[m],
                            f"{names[cond]}:{frag.replace(':', '_')}_z{charge}")
            if step.smaller is not None:
                tr = difference_trace(table, step.larger, step.smaller, run=cond[0],
                                      condition=cond[1], charge=charge)
                colour = _DIFF_COLOURS[j % len(_DIFF_COLOURS)]
                if len(tr):
                    _points(left, tr, colour, f"D {names[cond]}")
                    if right_defaults:
                        _fitted(right, tr, colour, f"D {names[cond]}", min_time,
                                fit and step.single)
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


def _fitted(ax, trace, colour, label, min_time, fit) -> None:
    """``trace`` as points, with its decay fit (from ``min_time``) when ``fit`` (else the
    points carry the label)."""
    _points(ax, trace, colour, None if fit else label)
    if not fit:
        return
    f = fit_decay(trace, min_time=min_time)
    if not f.ok:
        return
    t = np.geomspace(max(trace.after(min_time).time.min(), 1e-6), trace.time.max(), 200)
    if not f.resolved:
        ax.plot(t, f.curve(t), color=colour, lw=1.0, ls="--",
                label=f"{label}, rate not resolved")
        return
    ax.plot(t, f.curve(t), color=colour, lw=1.5,
            label=f"{label}, rate: {f.rate:.2e}±{f.rate_se:.2e} s⁻¹")


def _legend(ax) -> None:
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6, loc="upper right", frameon=True, markerscale=1.2)


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
) -> Figure:
    """The closure test against time (:func:`hxex.analysis.closure_trace`): for every split
    ``k``, ``D(n_ion k) + D(c_ion n-k) - D(precursor)``, one panel per condition, a colour and
    a marker per split, with the band of ``tolerance`` around 0. Points before
    ``min_time`` are drawn fainter (they are left out of :func:`hxex.analysis.closure_table`)."""
    from ..analysis.fragments import closure_trace

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
    for ax, cond in zip(axes, conditions, strict=True):
        ax.set_xscale("log")
        ax.axhspan(-tolerance, tolerance, color="0.92", zorder=0)
        ax.axhline(0.0, color="0.5", lw=0.8)
        for m, k in enumerate(splits):
            tr = closure_trace(table, f"{n_ion}:{k}", f"{c_ion}:{n - k}", n_charge=n_charge,
                               c_charge=c_charge, precursor_charge=precursor_charge,
                               run=cond[0], condition=cond[1])
            if not len(tr):
                continue
            colour = _SPLIT_COLOURS[m % len(_SPLIT_COLOURS)]
            style = {"marker": _SPLIT_MARKERS[m % len(_SPLIT_MARKERS)], "s": 10,
                     "linewidths": 0}
            early = tr.time < (min_time if min_time is not None else -np.inf)
            _points(ax, Trace(tr.time[~early], tr.mean[~early], tr.mean_se[~early]), colour,
                    f"{n_ion}_{k} + {c_ion}_{n - k}", 0.8, **style)
            _points(ax, Trace(tr.time[early], tr.mean[early], tr.mean_se[early]), colour,
                    None, 0.25, **style)
        ax.set_title(f"{shown} (Z{pz})  {names[cond]}", fontsize=9)
        ax.set_xlabel("Time, sec")
        ax.grid(True, which="major", alpha=0.4)
    handles, names = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, loc="outside right center", fontsize=7, markerscale=1.5)
    axes[0].set_ylabel(f"{n_ion} + {c_ion} − M_z{mz} (D)")
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
    top.set_title(f"{shown} {parent} (Z{'/'.join(map(str, charges))})  single-residue rates"
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


def fragment_outputs(results, config, *, quiet: bool = False):
    """The fragment part of an analysis (:class:`hxex.analysis.FragmentAnalysis` ``config``):
    tables (name -> columns) and figures (file stem -> function drawing it), or empty ones when
    the run has no fragment series. Per parent (``<sample>_<range>_z<charge>``):
    ``ladder_..._<ion>_z<charge>``, ``closure_..._<n ion>+<c ion>`` and ``residue_rates_...``;
    tables ``fragment_uptake``, ``ladder_rates``, ``ladder_residues`` and ``closure``. A parent
    read at several precursor charges has one ``residue_rates_<sample>_<range>`` (no charge)
    for all of them.
    ``results`` may be a
    :class:`hxex.analysis.RunSet` (:func:`hxex.analysis.fragment_uptake`): the tables' ``run``
    column then names each row's run, and the figures of an unmerged sample found in several
    runs are named ``<run>_<sample>_...``."""
    from ..analysis.fragments import (
        _PARTNERS,
        closure_table,
        fragment_ion_types,
        fragment_uptake,
        ladder_rates,
        ladder_residues,
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
    uptake = fragment_uptake(results)  # any other error is a real one
    if not len(uptake):
        return {}, []
    ions = list(config.ion_types) or fragment_ion_types(uptake)
    rates = ladder_rates(uptake, ion_types=ions, charges=config.charges,
                         min_time=config.min_time,
                         conditions=config.exchange if config.kint else None,
                         max_rate_error=config.max_rate_error)
    pairs = [(a, b) for a in ions for b in ions
             if chem.ion_terminus(a) == "N" and chem.ion_terminus(b) == "C"
             and _PARTNERS[a[0]] == b[0]]
    closures = [closure_table(uptake, a, b, precursor_charge=config.precursor_charge,
                              min_time=config.min_time, tolerance=config.tolerance)
                for a, b in pairs]
    tables = {"fragment_uptake": uptake.columns(), "ladder_rates": rates.columns(),
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
    state_run = uptake.state_run
    shown = display_labels(state_run, uptake.sample)
    groups = sorted({(str(u), str(s), str(p), int(z)) for u, s, p, z in zip(
        state_run, uptake.sample, uptake.parent, uptake.parent_charge, strict=True)})
    for state, sample, parent, pz in groups:
        rows = uptake.where(sample=sample, parent=parent, parent_charge=pz) & (state_run == state)
        start = int(uptake.parent_start[rows][0])
        name = shown[(state, sample)]
        stem = f"{name}_{start}-{start + len(chem.parse_sequence(parent)) - 1}_z{pz}"
        sel = {"run": state, "sample": sample, "parent": parent, "parent_charge": pz}
        solvent = sample_setting(config.exchange, state, sample) if config.kint else None
        kint = (intrinsic_rates(parent, solvent, start=start) if solvent is not None
                else None)
        for ion in ions:
            for z in config.charges:
                if (rows & uptake.where(ion_type=ion, charge=z) & (uptake.time > 0)).any():
                    jobs.append((f"ladder_{stem}_{ion}_z{z}",
                                 lambda ion=ion, z=z, sel=sel, kint=kint: ladder_kinetics_figure(
                                     uptake, ion, charge=z, labels=config.labels,
                                     min_time=config.min_time, kint=kint, **sel)))
        for a, b in pairs:
            have = set(uptake.fragment[rows].tolist())
            n = len(chem.parse_sequence(parent))
            if any(f"{a}:{k}" in have and f"{b}:{n - k}" in have for k in range(1, n)):
                jobs.append((f"closure_{stem}_{a}+{b}", lambda a=a, b=b, sel=sel: closure_figure(
                    uptake, a, b, labels=config.labels, min_time=config.min_time,
                    tolerance=config.tolerance, precursor_charge=config.precursor_charge,
                    **sel)))
    # the rates: one figure per parent, its precursor charges together
    for state, sample, parent in sorted({g[:3] for g in groups}):
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
