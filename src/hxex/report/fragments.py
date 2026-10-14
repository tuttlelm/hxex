# SPDX-License-Identifier: GPL-3.0-or-later
"""Fragment-ladder kinetics figure: D content against time, one row per ladder step.

Layout (after the owner's reference): the top row shows the precursor ions (``M``, ``M-NH3``)
of each condition on the left and the precursor's decay fits on the right; each further row is a
step of one ion series, in residue order: on the left the two fragments of the step in each
condition and their difference, on the right the difference with a fit of
``A exp(-k t) + C`` and its rate. A step whose difference holds several exchanging residues
(``block``) is drawn without a fit; a residue that no step covers keeps an empty row that says
why. Time is on a log axis in seconds.
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

# per condition: (shorter fragment, longer fragment) and the difference
_PAIR_COLOURS = [("tab:green", "tab:red"), ("tab:purple", "tab:brown"),
                 ("tab:pink", "tab:olive"), ("tab:gray", "tab:cyan")]
_DIFF_COLOURS = ["tab:blue", "tab:orange", "tab:cyan", "tab:gray"]
_EXTRA_COLOURS = ["#08306b", "#67000d", "#00441b", "#3f007d", "#7f2704"]
_MARKER = {"marker": "v", "s": 9, "linewidths": 0}


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
    conditions: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    precursors: Sequence[str] = ("M", "M-NH3"),
    precursor_charge: int | None = None,
    min_time: float | None = None,
    fit: bool = True,
    panel: tuple[float, float] = (3.6, 2.8),
    right_traces: Callable[[LadderStep | None], Sequence[tuple[str, Trace, bool]]] | None = None,
    right_defaults: bool = True,
) -> Figure:
    """The ladder of ``ion_type`` (fragment ``charge``) of one parent: see the module docstring.

    ``sample``, ``parent`` (its sequence) and ``parent_charge`` choose the series when the table
    holds several (each must be unique otherwise). ``conditions`` (default: all, sorted) are
    drawn in that order, named by ``labels`` (for example ``{'gentle': 'G'}``). ``precursors``
    are the ion types of the top row at ``precursor_charge`` (default: ``charge``). Points before
    ``min_time`` (s) are drawn but not fitted (spurious early points).

    ``right_traces`` adds traces of your own to the right column: called with each row's
    :class:`~hxex.analysis.LadderStep` (None for the precursor row), it returns ``(label, trace,
    fit)`` tuples, each drawn in its own colour and, when ``fit``, with its decay fit. With
    ``right_defaults`` False the right column shows only those (not the precursor's fits and
    the differences)."""
    pick = uptake.where(sample=sample, parent=parent, parent_charge=parent_charge)
    for name in ("sample", "parent", "parent_charge"):
        values = np.unique(getattr(uptake, name)[pick])
        if len(values) != 1:
            raise ValueError(f"the table holds {len(values)} {name} values; choose one "
                             f"({', '.join(map(str, values[:5]))})")
    table = uptake.select(pick & (uptake.time > 0))  # not the undeuterated control (log axis)
    if not len(table):
        raise ValueError("no fragment spectra after time 0")
    sample, parent = str(table.sample[0]), str(table.parent[0])
    pz = int(table.parent_charge[0])
    conditions = list(conditions) if conditions is not None else sorted(
        np.unique(table.condition).tolist())
    names = {c: (labels or {}).get(c, c) for c in conditions}
    entry = table.parent_entry()
    residues = chem.parse_sequence(parent)
    side = chem.ion_terminus(ion_type)
    series = table.where(ion_type=ion_type, charge=charge)
    lengths = sorted(set(table.length[series].tolist()))
    steps = ladder_steps(entry, ion_type, lengths, nterm_subtract=table.nterm_subtract,
                         rules=table.amide_rules)
    activation = ", ".join(sorted(set(table.activation[series].tolist()))) or ion_type
    head = f"{sample} (Z{pz})  {activation} {ion_type}_z{charge}"
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
    left.set_title(f"{sample} (Z{pz})  {activation} M_z{pre_z}", fontsize=9)
    for j, cond in enumerate(conditions):
        colours = _PAIR_COLOURS[j % len(_PAIR_COLOURS)]
        for m, ion in enumerate(precursors):
            label = f"{ion}:{len(residues)}"
            tr = fragment_trace(table, label, condition=cond, charge=pre_z)
            if len(tr):
                left.scatter(tr.time, tr.mean, color=colours[m % 2],
                             label=f"{ion} {names[cond]}", **_MARKER)
        tr = fragment_trace(table, f"M:{len(residues)}", condition=cond, charge=pre_z)
        if len(tr) and right_defaults:
            _fitted(right, tr, _DIFF_COLOURS[j % len(_DIFF_COLOURS)], f"M {names[cond]}",
                    min_time, fit)
    _extras(right, right_traces, None, min_time)
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
                tr = fragment_trace(table, frag, condition=cond, charge=charge)
                if len(tr):
                    left.scatter(tr.time, tr.mean, color=colours[m],
                                 label=f"{names[cond]}:{frag.replace(':', '_')}_z{charge}",
                                 **_MARKER)
            if step.smaller is not None:
                tr = difference_trace(table, step.larger, step.smaller, condition=cond,
                                      charge=charge)
                colour = _DIFF_COLOURS[j % len(_DIFF_COLOURS)]
                if len(tr):
                    left.scatter(tr.time, tr.mean, color=colour,
                                 label=f"D {names[cond]}", **_MARKER)
                    if right_defaults:
                        _fitted(right, tr, colour, f"D {names[cond]}", min_time,
                                fit and step.single)
        _extras(right, right_traces, step, min_time)
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


def _fitted(ax, trace, colour, label, min_time, fit) -> None:
    """``trace`` as points, with its decay fit (from ``min_time``) when ``fit`` (else the
    points carry the label)."""
    ax.scatter(trace.time, trace.mean, color=colour, label=None if fit else label, **_MARKER)
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


def _one(table, sample, parent, parent_charge):
    """``sample``, ``parent`` and ``parent_charge`` of ``table``, each unique."""
    pick = table.where(sample=sample, parent=parent, parent_charge=parent_charge)
    out = []
    for name in ("sample", "parent", "parent_charge"):
        values = np.unique(getattr(table, name)[pick])
        if len(values) != 1:
            raise ValueError(f"the table holds {len(values)} {name} values; choose one "
                             f"({', '.join(map(str, values[:5]))})")
        out.append(values[0])
    return pick, str(out[0]), str(out[1]), int(out[2])


def closure_figure(
    uptake: FragmentUptake,
    n_ion: str = "c-1",
    c_ion: str = "z-dot",
    *,
    sample: str | None = None,
    parent: str | None = None,
    parent_charge: int | None = None,
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
    ``k``, ``D(n_ion k) + D(c_ion n-k) - D(precursor)``, one panel per condition, one colour per
    split (light to dark with ``k``), with the band of ``tolerance`` around 0. Points before
    ``min_time`` are drawn fainter (they are left out of :func:`hxex.analysis.closure_table`)."""
    from ..analysis.fragments import closure_trace

    pick, sample, parent, pz = _one(uptake, sample, parent, parent_charge)
    table = uptake.select(pick & (uptake.time > 0))
    conditions = list(conditions) if conditions is not None else sorted(
        np.unique(table.condition).tolist())
    n = len(chem.parse_sequence(parent))
    have = set(table.fragment.tolist())
    splits = [k for k in range(1, n) if f"{n_ion}:{k}" in have and f"{c_ion}:{n - k}" in have]
    if not splits:
        raise ValueError(f"no {n_ion} and {c_ion} pairs of {parent}")
    import matplotlib

    cmap = matplotlib.colormaps["viridis"]
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
                               condition=cond)
            if not len(tr):
                continue
            colour = cmap(0.1 + 0.8 * m / max(1, len(splits) - 1))
            early = tr.time < (min_time if min_time is not None else -np.inf)
            ax.scatter(tr.time[~early], tr.mean[~early], color=colour,
                       label=f"{n_ion}_{k} + {c_ion}_{n - k}", **_MARKER)
            ax.scatter(tr.time[early], tr.mean[early], color=colour, alpha=0.3, **_MARKER)
        ax.set_title(f"{sample} (Z{pz})  {(labels or {}).get(cond, cond)}", fontsize=9)
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
    labels: Mapping[str, str] | None = None,
    panel: tuple[float, float] = (7.0, 2.6),
) -> Figure:
    """The rates of every single-residue ladder step (:func:`hxex.analysis.ladder_rates`) by
    residue: on top the rate (log axis, with its standard error), below the fitted amplitude
    (the deuterium that exchanges over the time range). Ion types are told apart by marker,
    conditions by colour; a fragment charge other than 1 is named in the legend. Unresolved
    fits (the decay not shown by the time range) are drawn as open markers along the bottom of
    the rate panel and have no amplitude; the rate axis spans the rates whose standard error is
    below the rate (others may reach beyond it)."""
    pick, sample, parent, pz = _one(rates, sample, parent, parent_charge)
    residues = chem.parse_sequence(parent)
    start = int(rates.parent_start[pick][0])
    fig = Figure(figsize=(panel[0], 2 * panel[1]), layout="constrained")
    top, bottom = fig.subplots(2, 1, sharex=True)
    series = sorted({(str(c), str(i), int(z)) for c, i, z in zip(
        rates.condition[pick], rates.ion_type[pick], rates.charge[pick], strict=True)},
        key=lambda s: (s[0], chem.ion_terminus(s[1]) != "N", s[1], s[2]))
    conditions = sorted({s[0] for s in series})
    width = 0.6 / max(1, len(series))
    for m, (cond, ion, z) in enumerate(series):
        rows = pick & rates.where(condition=cond, ion_type=ion, charge=z)
        x = rates.residue[rows] + (m - (len(series) - 1) / 2) * width
        colour = _DIFF_COLOURS[conditions.index(cond) % len(_DIFF_COLOURS)]
        marker = _ION_MARKERS.get(ion[0], "o")
        name = f"{(labels or {}).get(cond, cond)} {ion}" + (f" z{z}" if z != 1 else "")
        ok = rates.resolved[rows]
        top.errorbar(x[ok], rates.rate[rows][ok], yerr=rates.rate_se[rows][ok], fmt=marker,
                     color=colour, ms=5, capsize=2, lw=0.8, label=name)
        top.scatter(x[~ok], np.zeros(int((~ok).sum())), marker=marker, facecolors="none",
                    edgecolors=colour, s=25, transform=top.get_xaxis_transform(), clip_on=False)
        bottom.scatter(x[ok], rates.amplitude[rows][ok], marker=marker, color=colour, s=25)
    top.set_yscale("log")
    # the axis follows the rates known to better than their own size
    good = pick & rates.resolved & (rates.rate_se < rates.rate)
    if good.any():
        lo, hi = float(rates.rate[good].min()), float(rates.rate[good].max())
        top.set_ylim(lo / 3, hi * 3)
    top.set_ylabel("rate (s⁻¹)")
    bottom.set_ylabel("amplitude (D)")
    bottom.axhline(0.0, color="0.5", lw=0.8)
    ticks = list(range(start, start + len(residues)))
    bottom.set_xticks(ticks, [f"{residues[t - start]}{t}" for t in ticks],
                      fontsize=7 if len(ticks) > 30 else 8,
                      rotation=90 if len(ticks) > 30 else 0)
    bottom.set_xlim(start - 0.6, start + len(residues) - 0.4)
    for ax in (top, bottom):
        ax.grid(True, axis="y", alpha=0.4)
    top.set_title(f"{sample} {parent} (Z{pz})  single-residue rates"
                  + ("  (open: not resolved)" if not rates.resolved[pick].all() else ""),
                  fontsize=9)
    if top.get_legend_handles_labels()[0]:
        top.legend(fontsize=7, loc="best", ncol=max(1, len(series) // 4))
    return fig


def fragment_outputs(results, config, *, quiet: bool = False):
    """The fragment part of an analysis (:class:`hxex.analysis.FragmentAnalysis` ``config``):
    tables (name -> columns) and figures (file stem -> function drawing it), or empty ones when
    the run has no fragment series. Per parent (``<sample>_<range>_z<charge>``):
    ``ladder_..._<ion>_z<charge>``, ``closure_..._<n ion>+<c ion>`` and ``residue_rates_...``;
    tables ``fragment_uptake``, ``ladder_rates`` and ``closure``."""
    from ..analysis.fragments import (
        _PARTNERS,
        closure_table,
        fragment_ion_types,
        fragment_uptake,
        ladder_rates,
    )

    try:
        uptake = fragment_uptake(results)
    except ValueError:  # not read from scan files
        return {}, []
    if not len(uptake):
        return {}, []
    ions = list(config.ion_types) or fragment_ion_types(uptake)
    rates = ladder_rates(uptake, ion_types=ions, charges=config.charges,
                         min_time=config.min_time)
    pairs = [(a, b) for a in ions for b in ions
             if chem.ion_terminus(a) == "N" and chem.ion_terminus(b) == "C"
             and _PARTNERS[a[0]] == b[0]]
    closures = [closure_table(uptake, a, b, precursor_charge=config.precursor_charge,
                              min_time=config.min_time, tolerance=config.tolerance)
                for a, b in pairs]
    tables = {"fragment_uptake": uptake.columns(), "ladder_rates": rates.columns()}
    if closures:
        cols = [c.columns() for c in closures]
        tables["closure"] = {name: np.concatenate([c[name] for c in cols]) for name in cols[0]}
        flagged = int(sum(c.flagged.sum() for c in closures))
        if flagged and not quiet:
            import logging

            logging.getLogger(__name__).warning(
                "%d fragment pair(s) miss the precursor by more than %g D (closure.csv)",
                flagged, config.tolerance)
    jobs = []
    groups = sorted({(str(s), str(p), int(z)) for s, p, z in zip(
        uptake.sample, uptake.parent, uptake.parent_charge, strict=True)})
    for sample, parent, pz in groups:
        rows = uptake.where(sample=sample, parent=parent, parent_charge=pz)
        start = int(uptake.parent_start[rows][0])
        stem = f"{sample}_{start}-{start + len(chem.parse_sequence(parent)) - 1}_z{pz}"
        sel = {"sample": sample, "parent": parent, "parent_charge": pz}
        for ion in ions:
            for z in config.charges:
                if (rows & uptake.where(ion_type=ion, charge=z) & (uptake.time > 0)).any():
                    jobs.append((f"ladder_{stem}_{ion}_z{z}",
                                 lambda ion=ion, z=z, sel=sel: ladder_kinetics_figure(
                                     uptake, ion, charge=z, labels=config.labels,
                                     min_time=config.min_time, **sel)))
        for a, b in pairs:
            have = set(uptake.fragment[rows].tolist())
            n = len(chem.parse_sequence(parent))
            if any(f"{a}:{k}" in have and f"{b}:{n - k}" in have for k in range(1, n)):
                jobs.append((f"closure_{stem}_{a}+{b}", lambda a=a, b=b, sel=sel: closure_figure(
                    uptake, a, b, labels=config.labels, min_time=config.min_time,
                    tolerance=config.tolerance, precursor_charge=config.precursor_charge,
                    **sel)))
        if rates.where(**sel).any():
            jobs.append((f"residue_rates_{stem}", lambda sel=sel: residue_rates_figure(
                rates, labels=config.labels, **sel)))
    return tables, jobs
