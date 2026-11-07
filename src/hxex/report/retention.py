# SPDX-License-Identifier: GPL-3.0-or-later
"""The retention trace of one spectrum (PLAN Next steps 26: the review's trace panel).

:func:`plot_retention` draws what :func:`hxex.io.retention_trace.retention_traces` reads: the
selected spectrum's retention trace in its run, the scans it summed shaded, the same series'
other runs in the sample as faint lines behind it (retention moves between files, so a window
on a neighbour's peak shows at once), and, for files with ion mobility bins, the spectrum's
drift profile with the drift range it was summed over.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..io.retention_trace import RetentionTraces, Trace
from ..io.times import format_time
from .spectrum import RAW_DARK

OTHER = "#b3b3b3"  # the other runs' traces
SUMMED = "#d9d9d9"  # the scans (or drift times) the spectrum summed
_SHOWN_SHARE = 0.02  # the drift panel shows the profile down to this share of its apex


def run_text(trace: Trace) -> str:
    """A run as the figures name it: its time (or control role), replicate and file."""
    if trace.time is not None:
        when = format_time(trace.time)
    elif trace.role == "undeut":
        when = "undeuterated"
    elif trace.role == "fulldeut":
        when = "fully deuterated"
    else:
        when = "kinetics"
    return f"{when}, rep {trace.rep} ({trace.file})"


def plot_retention(traces: RetentionTraces, *, title: str | None = None) -> Figure:
    """Draw a spectrum's retention traces (:class:`RetentionTraces`, the selected run first).
    Returns the figure; nothing is shown or saved. Without traces the figure says why (the
    traces' ``note``). Every artist drawn for a run has the ``gid`` ``trace:<spectrum id>``."""
    with_drift = traces.drift is not None
    fig = Figure(figsize=(9.6 if with_drift else 7.2, 3.6), layout="constrained")
    if not traces.traces:
        ax = fig.add_subplot()
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"No elution trace: {traces.note or 'nothing read'}",
                ha="center", va="center", wrap=True, transform=ax.transAxes)
        return fig
    if with_drift:
        ax, dax = fig.subplots(1, 2, width_ratios=(2.2, 1))
    else:
        ax, dax = fig.add_subplot(), None
    selected, others = traces.traces[0], traces.traces[1:]
    for t in others:
        ax.plot(t.times, t.intensity, color=OTHER, lw=0.9, zorder=2,
                gid=f"trace:{t.spectrum_id}")
    ax.axvspan(*selected.window, color=SUMMED, lw=0, zorder=1, gid="summed")
    ax.plot(selected.times, selected.intensity, color=RAW_DARK, lw=1.6, zorder=4,
            marker="o", ms=2.5, gid=f"trace:{selected.spectrum_id}")
    ax.set_xlabel("retention time (min)")
    ax.set_ylabel("intensity at the ion's bins")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    handles = [Line2D([], [], color=RAW_DARK, lw=1.6, marker="o", ms=2.5,
                      label=f"this spectrum: {run_text(selected)}"),
               Patch(color=SUMMED, lw=0, label=f"scans summed ({selected.window[0]:.2f}-"
                                                f"{selected.window[1]:.2f} min)")]
    if others:
        handles.append(Line2D([], [], color=OTHER, lw=0.9,
                              label=f"the sample's other runs ({len(others)})"))
    ax.legend(handles=handles, loc="upper left", fontsize="small", frameon=False)
    top = max(float(np.nanmax(t.intensity)) if len(t.intensity) else 0.0
              for t in traces.traces)
    if top > 0:
        ax.set_ylim(top=top * 1.35)  # room for the legend
    if dax is not None:
        _drift_panel(dax, traces)
    head = title if title is not None else traces.label
    if selected.drift is not None:
        head += f"; drift {selected.drift[0]:.3g}-{selected.drift[1]:.3g} ms"
    fig.suptitle(head, fontsize="medium")
    if traces.note:
        fig.text(0.01, 0.01, traces.note, fontsize="x-small", color="#7a1f1f")
    return fig


def _drift_panel(ax, traces: RetentionTraces) -> None:
    """The selected spectrum's drift profile over its window, with its drift range shaded."""
    p = traces.drift
    drift, y = np.asarray(p.drift, dtype=float), np.asarray(p.intensity, dtype=float)
    if p.range is not None:
        ax.axvspan(*p.range, color=SUMMED, lw=0, zorder=1, gid="drift_range")
    ax.plot(drift, y, color=RAW_DARK, lw=1.4, zorder=3, gid="drift_profile")
    ax.set_xlabel("drift time (ms)")
    ax.set_ylabel("drift profile on the isotope peaks")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    if len(y) and y.max() > 0:
        shown = drift[y >= _SHOWN_SHARE * y.max()]
        lo, hi = float(shown.min()), float(shown.max())
        if p.range is not None:
            lo, hi = min(lo, p.range[0]), max(hi, p.range[1])
        pad = 0.1 * max(hi - lo, 1e-3)
        ax.set_xlim(lo - pad, hi + pad)
