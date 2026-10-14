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

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure

from .. import chem
from ..analysis.fragments import (
    FragmentUptake,
    difference_trace,
    fit_decay,
    fragment_trace,
    ladder_steps,
)

# per condition: (shorter fragment, longer fragment) and the difference
_PAIR_COLOURS = [("tab:green", "tab:red"), ("tab:purple", "tab:brown"),
                 ("tab:pink", "tab:olive"), ("tab:gray", "tab:cyan")]
_DIFF_COLOURS = ["tab:blue", "tab:orange", "tab:cyan", "tab:gray"]
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
) -> Figure:
    """The ladder of ``ion_type`` (fragment ``charge``) of one parent: see the module docstring.

    ``sample``, ``parent`` (its sequence) and ``parent_charge`` choose the series when the table
    holds several (each must be unique otherwise). ``conditions`` (default: all, sorted) are
    drawn in that order, named by ``labels`` (for example ``{'gentle': 'G'}``). ``precursors``
    are the ion types of the top row at ``precursor_charge`` (default: ``charge``). Points before
    ``min_time`` (s) are drawn but not fitted (spurious early points)."""
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
        if len(tr):
            _fitted(right, tr, _DIFF_COLOURS[j % len(_DIFF_COLOURS)], f"M {names[cond]}",
                    min_time, fit)
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
                    _fitted(right, tr, colour, f"D {names[cond]}", min_time,
                            fit and step.single)
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


def _fitted(ax, trace, colour, label, min_time, fit) -> None:
    """``trace`` as points, with its decay fit (from ``min_time``) when ``fit``."""
    ax.scatter(trace.time, trace.mean, color=colour, **_MARKER)
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
