# SPDX-License-Identifier: GPL-3.0-or-later
"""The retention trace of one spectrum (PLAN Next steps 26: the review's trace panel).

:func:`plot_retention` draws what :func:`hxex.io.retention_trace.retention_traces` reads: the
selected spectrum's retention trace in its run, the scans it summed shaded, behind it either the
same series' other runs in the sample as faint lines (retention moves between files, so a window
on a neighbour's peak shows at once) or the peptide's other ions in the same run, each scaled to
the selected trace's height (its charge states and fragment channels elute together, so an ion
whose peak sits apart has interference at its own m/z), and, for files with ion mobility bins,
the spectrum's drift profile with the drift range it was summed over.
"""

from __future__ import annotations

import re

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..io.retention_trace import RetentionTraces, Trace
from ..io.times import format_time
from .spectrum import RAW_DARK

OTHER = "#b3b3b3"  # the other runs' traces
SUMMED = "#d9d9d9"  # the scans (or drift times) the spectrum summed
EDITED = "#4d4d4d"  # the outline of a window being edited
HATCH = "#c4c4c4"  # its hatching: light, behind the traces (owner: dark, it hid the other runs)
UNMATCHED = "#7f7f7f"  # a file whose trace did not match: dashed, where it is (not a sample colour)
_SHOWN_SHARE = 0.02  # the drift panel shows the profile down to this share of its apex
# the peptide's other ions in the run (not the samples' colours: one sample is drawn here)
ION_COLOURS = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d")


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


def plot_retention(traces: RetentionTraces, *, title: str | None = None,
                   edit: tuple[float, float] | None = None,
                   drift_edit: tuple[float, float] | None = None) -> Figure:
    """Draw a spectrum's retention traces (:class:`RetentionTraces`, the selected run first).
    Returns the figure; nothing is shown or saved. Without traces the figure says why (the
    traces' ``note``). Every artist drawn for a run has the ``gid`` ``trace:<spectrum id>``.
    ``edit`` (minutes) and ``drift_edit`` (ms) mark a window being edited (hatched, ``gid``
    ``edited`` and ``drift_edited``)."""
    with_drift = traces.drift is not None
    fig = Figure(figsize=(9.6 if with_drift else 7.2, 3.6), layout="constrained")
    if not traces.traces:
        _nothing(fig.add_subplot(), traces)
        return fig
    if with_drift:
        ax, dax = fig.subplots(1, 2, width_ratios=(2.2, 1))
    else:
        ax, dax = fig.add_subplot(), None
    _trace_panel(ax, traces, edit)
    if dax is not None:
        _drift_panel(dax, traces, drift_edit)
    _heading(fig, traces, title)
    return fig


def plot_window_edit(traces: RetentionTraces, results, spectrum_id: str, preview, *,
                     edit: tuple[float, float], drift_edit: tuple[float, float] | None = None,
                     each_k: bool = False, title: str | None = None,
                     mz_range: tuple[float, float] | None = None,
                     waiting: str | None = None, aligned=None) -> Figure:
    """The retention trace with a window being edited (``edit``, minutes; ``drift_edit``, ms),
    and under it the spectrum as ``results`` hold it (``spectrum_id``) beside the spectrum
    summed over the new window (``preview``, :class:`hxex.gui.window_edit.Preview`, or None
    while it is not ready; picked only while its fit runs), each at its K; with ``each_k``
    the new window's spectrum at every K the run tried instead. ``mz_range`` is the series'
    m/z axis; ``waiting`` says why there is no preview or fit (default: it is being made).
    The preview's notes (``preview.notes``) are not drawn: the window shows a line above the
    figure (:func:`preview_text`) with them as its tooltip (:func:`preview_notes`). With
    ``aligned`` (:class:`hxex.gui.window_edit.Alignment`) the trace panel draws the other
    files moved onto this file's clock by their match (:func:`_aligned_panel`) instead of
    where they are."""
    from .spectrum import plot_spectrum

    with_drift = traces.drift is not None
    fitted = preview is not None and preview.results is not None
    ks = sorted(preview.results.fit.fits) if each_k and fitted else [None]
    columns = 1 + len(ks)
    fig = Figure(figsize=(max(9.6, 3.2 * columns), 7.6), layout="constrained")
    rows = fig.add_gridspec(2, 1, height_ratios=(1, 1.3))
    if not traces.traces:
        _nothing(fig.add_subplot(rows[0]), traces)
    else:
        top = rows[0].subgridspec(1, 2, width_ratios=(2.2, 1)) if with_drift else [rows[0]]
        if aligned is not None:
            _aligned_panel(fig.add_subplot(top[0]), traces, aligned, edit)
        else:
            _trace_panel(fig.add_subplot(top[0]), traces, edit)
        if with_drift:
            _drift_panel(fig.add_subplot(top[1]), traces, drift_edit)
    bottom = rows[1].subgridspec(1, columns)
    axes = [fig.add_subplot(bottom[c]) for c in range(columns)]
    i = list(results.fit.spectrum_id).index(spectrum_id)
    k_now = int(results.effective_K[i])
    plot_spectrum(spectrum_id, results, ax=axes[0], show=("raw", "picked", "fit"), k=k_now,
                  mz_range=mz_range, legend=False, title=False)
    _data_height(axes[0])
    window = traces.traces[0].window if traces.traces else None
    axes[0].set_title(f"now\n{_panel_text(spectrum_id, window)}\n"
                      f"{_k_text(results.fit, i, k_now)}", fontsize="small")
    if preview is None:
        axes[1].set_axis_off()
        axes[1].text(0.5, 0.5, f"No preview: {waiting}" if waiting
                     else "reading the new window's scans…", ha="center", va="center", wrap=True,
                     transform=axes[1].transAxes)
    else:
        spectrum = preview.spectrum
        sid = spectrum.spectrum_id
        new_window = (spectrum.rt_start, spectrum.rt_end)
        for ax_k, k in zip(axes[1:], ks, strict=True):
            if not fitted:  # picked, the fit still running
                plot_spectrum(sid, spectra=preview.spectra, binned=preview.binned, ax=ax_k,
                              show=("raw", "picked"), mz_range=mz_range, legend=False,
                              title=False)
                _data_height(ax_k)  # the same axis as when it is fitted
                ax_k.set_title(f"new window\n{_panel_text(sid, new_window)}\n"
                               f"{waiting or 'fitting its spectrum…'}", fontsize="small")
                continue
            k = preview.k if k is None else k
            plot_spectrum(sid, preview.results, ax=ax_k, show=("raw", "picked", "fit"), k=k,
                          mz_range=mz_range, legend=False, title=False)
            _data_height(ax_k)
            head = ("new window" if not each_k else f"new window, K = {k}"
                    + (" (chosen)" if k == preview.k else ""))
            ax_k.set_title(f"{head}\n{_panel_text(sid, new_window)}\n"
                           f"{_k_text(preview.results.fit, 0, k, with_k=not each_k)}",
                           fontsize="small",
                           fontweight="bold" if each_k and k == preview.k else "normal")
    _heading(fig, traces, title)
    return fig


def plot_each_file(traces: RetentionTraces, aligned, *, edit: tuple[float, float],
                   title: str | None = None) -> Figure:
    """One row per file a window spread at ``aligned.level`` reaches
    (:class:`hxex.gui.window_edit.Alignment`), each on its own clock, this file first: the
    file's trace of the charge state (as the run summed it: :func:`_as_summed`), the scans its
    spectrum sums now (shaded) and the window being edited (``edit``, minutes) shifted into it
    (hatched), with the shift, or why the file was not matched (no window). Every row's trace
    has the ``gid`` ``trace:<spectrum id>``; only the first row (this file, axes label
    ``this file``) takes a drag across it as the window."""
    rows = [(_as_summed(traces, aligned.reference),
             traces.traces[0].window if traces.traces else None, edit, "this file")]
    for p in aligned.for_window(edit):
        if p.trace is None:
            continue
        label = (f"{p.shift * 60:+.0f} s (r {p.correlation:.2f})" if p.shift is not None
                 else f"not matched: {p.why}")
        if aligned.level == "everywhere" and p.samples:  # whose file (a control: several)
            label += "; " + (p.samples[0] if len(p.samples) == 1
                             else f"{len(p.samples)} samples")
        t = _as_summed(traces, p.trace)
        rows.append((t, t.window, p.window, label))
    fig = Figure(figsize=(8.4, 0.62 * len(rows) + 1.1), layout="constrained")
    axes = np.atleast_1d(fig.subplots(len(rows), 1, sharex=True))
    for n, (ax, (t, summed, window, label)) in enumerate(zip(axes, rows, strict=True)):
        if summed is not None:
            ax.axvspan(*summed, color=SUMMED, lw=0, zorder=1, gid="summed")
        if window is not None:
            ax.axvspan(*window, facecolor="none", edgecolor=HATCH, hatch="///", lw=0,
                       zorder=1.5, gid="edited")
            ax.axvspan(*window, facecolor="none", edgecolor=EDITED, lw=0.6, zorder=1.6)
        ax.plot(t.times, t.intensity, color=RAW_DARK if n == 0 else EDITED,
                lw=1.3 if n == 0 else 0.9, zorder=3, gid=f"trace:{t.spectrum_id}")
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.text(1.01, 0.5, f"{run_text(t)}\n{label}", transform=ax.transAxes,
                fontsize="x-small", va="center", fontweight="bold" if n == 0 else "normal",
                color="#000000" if window is not None else UNMATCHED)
        if n == 0:
            ax.set_label("this file")
    axes[-1].set_xlabel("each file's own clock: retention time (min)")
    fig.suptitle((title if title is not None else traces.label)
                 + f"\neach file's window: the scans summed now (shaded) and the new window "
                   f"shifted into it (hatched); {len(rows) - 1} other file(s)",
                 fontsize="medium")
    return fig


PREVIEW_TEXT = "Preview only: results will change upon rerun with overwrite enabled."


def preview_text(preview) -> str:
    """What the window says about a preview, above its figure: one fixed line, so that it does
    not change length (and the figure's size) as the preview is read and fitted (owner,
    2026-11-12); its notes go in :func:`preview_notes`."""
    return PREVIEW_TEXT


def preview_notes(preview) -> str:
    """How the preview differs from a run (``preview.notes``), as one sentence, or empty:
    the line's tooltip (:func:`preview_text`)."""
    notes = "; ".join(preview.notes) if preview is not None and preview.notes else ""
    return f"{notes[0].upper()}{notes[1:]}." if notes else ""


def _data_height(ax) -> None:
    """A spectrum panel's y axis from its data alone (the raw profile within the m/z axis and
    the picked values), from 0 with a tenth above, so that the fit, drawn when it is ready, does not
    move it (owner, 2026-11-12: the new window's axis changed twice after each new drift
    range, picked and then fitted)."""
    from matplotlib.collections import PathCollection

    lo, hi = ax.get_xlim()
    top = 0.0
    for a in ax.get_children():
        gid = a.get_gid() or ""
        if gid == "raw" and isinstance(a, Line2D):
            x, y = (np.asarray(v, dtype=float) for v in a.get_data())
            y = y[(x >= lo) & (x <= hi) & np.isfinite(y)]
        elif gid == "picked" and isinstance(a, PathCollection):
            y = np.asarray(a.get_offsets(), dtype=float).reshape(-1, 2)[:, 1]
            y = y[np.isfinite(y)]
        else:
            continue
        if len(y):
            top = max(top, float(y.max()))
    if top > 0:  # from 0, as the fitted spectrum's picking-window bars put it (the picked-only
        ax.set_ylim(0.0, 1.1 * top)  # draw had matplotlib's margin below 0: y-min jumped)


def _panel_text(spectrum_id: str, window: tuple[float, float] | None) -> str:
    """A spectrum panel's scans and minutes, from its id and window."""
    m = re.search(r"/scans?(\d+)(?:-(\d+))?$", spectrum_id)
    scans = ("" if m is None else
             f"scan {m.group(1)}" if m.group(2) is None else f"scans {m.group(1)}-{m.group(2)}")
    if window is not None and np.all(np.isfinite(window)):
        scans += f", {window[0]:.2f}-{window[1]:.2f} min"
    return scans


def _k_text(fit, i: int, k: int, with_k: bool = True) -> str:
    """K and the populations' deuteration (uncorrected, D) of measurement ``i``."""
    from .spectrum import d_values

    return ", ".join(t for t in ((f"K = {k}" if with_k else ""), d_values(fit, i, k)) if t)


def _nothing(ax, traces: RetentionTraces) -> None:
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"No elution trace: {traces.note or 'nothing read'}",
            ha="center", va="center", wrap=True, transform=ax.transAxes)


def _heading(fig, traces: RetentionTraces, title: str | None) -> None:
    head = title if title is not None else traces.label
    if traces.traces and traces.traces[0].drift is not None:
        drift = traces.traces[0].drift
        head += f"; drift {drift[0]:.3g}-{drift[1]:.3g} ms"
    fig.suptitle(head, fontsize="medium")
    if traces.note and traces.traces:
        fig.text(0.01, 0.01, traces.note, fontsize="x-small", color="#7a1f1f")


def _own_range(ax, selected, edit: tuple[float, float] | None) -> None:
    """The retention-time axis of this file's trace: its window and the reader's margin either
    side (:data:`hxex.io.retention_trace.MARGIN`, or the window's width), widened only to show
    the window being edited. Every draw of the spectrum keeps it, whatever other files are read
    with it or matched to it (the reader reads every trace around all of their windows, and the
    axis moved as they arrived; owner, 2026-11-12)."""
    from ..io.retention_trace import MARGIN

    a, b = (float(v) for v in selected.window)
    if not (np.isfinite(a) and np.isfinite(b)):
        return
    pad = max(MARGIN, b - a)
    lo, hi = a - pad, b + pad
    if edit is not None and np.all(np.isfinite(edit)):
        lo, hi = min(lo, float(edit[0])), max(hi, float(edit[1]))
    ax.set_xlim(lo, hi)


def _trace_panel(ax, traces: RetentionTraces, edit: tuple[float, float] | None) -> None:
    """The selected run's trace over its window (shaded), the others behind it (other runs, or
    the peptide's other ions in this run, each scaled to the selected trace's height), and the
    window being edited (hatched)."""
    selected, others = traces.traces[0], traces.traces[1:]
    for t in others:
        ax.plot(t.times, t.intensity, color=OTHER, lw=0.9, zorder=2,
                gid=f"trace:{t.spectrum_id}")
    own_top = float(np.nanmax(selected.intensity)) if len(selected.intensity) else 0.0
    ion_handles = []
    for n, ion in enumerate(traces.ions):
        colour = ION_COLOURS[n % len(ION_COLOURS)]
        top = float(np.nanmax(ion.intensity)) if len(ion.intensity) else 0.0
        y = ion.intensity * (own_top / top) if top > 0 and own_top > 0 else ion.intensity
        ax.plot(ion.times, y, color=colour, lw=1.1, zorder=3, gid=f"ion:{ion.label}")
        ion_handles.append(Line2D([], [], color=colour, lw=1.1, label=ion.label))
    ax.axvspan(*selected.window, color=SUMMED, lw=0, zorder=1, gid="summed")
    if edit is not None:  # behind every trace (owner: over them it hid the other runs)
        ax.axvspan(*edit, facecolor="none", edgecolor=HATCH, hatch="///", lw=0, zorder=1.5,
                   gid="edited")
        ax.axvspan(*edit, facecolor="none", edgecolor=EDITED, lw=0.8, zorder=1.6,
                   gid="edited_outline")
    ax.plot(selected.times, selected.intensity, color=RAW_DARK, lw=1.6, zorder=4,
            marker="o", ms=2.5, gid=f"trace:{selected.spectrum_id}")
    _own_range(ax, selected, edit)
    ax.set_xlabel("retention time (min)")
    ax.set_ylabel("intensity at the ion's bins")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    handles = [Line2D([], [], color=RAW_DARK, lw=1.6, marker="o", ms=2.5,
                      label=f"this spectrum: {run_text(selected)}"),
               Patch(color=SUMMED, lw=0, label=f"scans summed ({selected.window[0]:.2f}-"
                                                f"{selected.window[1]:.2f} min)")]
    if edit is not None:
        handles.append(Patch(facecolor="none", edgecolor=HATCH, hatch="///", lw=0.8,
                             label=f"new window ({edit[0]:.2f}-{edit[1]:.2f} min)"))
    if others:
        handles.append(Line2D([], [], color=OTHER, lw=0.9,
                              label=f"{traces.others_label} ({len(others)})"))
    first = ax.legend(handles=handles, loc="upper left", fontsize="small", frameon=False)
    if ion_handles:
        ax.add_artist(first)  # kept beside the ions' own legend
        ax.legend(handles=ion_handles, loc="upper right", fontsize="x-small", frameon=False,
                  title=f"{traces.ions_label}\n(each scaled to this one's height)",
                  title_fontsize="x-small")
    top = max(float(np.nanmax(t.intensity)) if len(t.intensity) else 0.0
              for t in traces.traces)
    if top > 0:
        room = max(1.45 if edit is not None else 1.35, 1.25 + 0.09 * (len(ion_handles) + 2))
        ax.set_ylim(top=top * room)  # room for the legends


def _as_summed(traces: RetentionTraces, trace):
    """``trace`` (one a match read, within the drift range being edited) as the run summed it:
    the tab's own trace of that spectrum (its run's drift range), when the tab read one. A new
    drift range then moves the other files by their new shifts but changes no trace (owner,
    2026-11-12: the traces changed with every drift range dragged)."""
    for t in traces.traces:
        if t.spectrum_id == trace.spectrum_id:
            return t
    return trace


def _aligned_panel(ax, traces: RetentionTraces, aligned,
                   edit: tuple[float, float] | None) -> None:
    """The other files a window spread reaches moved onto this file's clock by their match and
    scaled to this trace's height (those not matched dashed, where they are), this file's trace
    within the drift range matched, the scans it sums now (shaded), and the window being edited
    (hatched): whether the match lined the peaks up, at a glance."""
    selected = traces.traces[0]
    ref = _as_summed(traces, aligned.reference)
    own_top = float(np.nanmax(ref.intensity)) if len(ref.intensity) else 0.0
    matched = [p for p in aligned.plans if p.trace is not None and p.shift is not None]
    unmatched = [p for p in aligned.plans if p.trace is not None and p.shift is None]
    for p, dashed in [(p, False) for p in matched] + [(p, True) for p in unmatched]:
        t = _as_summed(traces, p.trace)
        top = float(np.nanmax(t.intensity)) if len(t.intensity) else 0.0
        y = t.intensity * (own_top / top) if top > 0 and own_top > 0 else t.intensity
        ax.plot(t.times - (0.0 if dashed else p.shift), y, color=UNMATCHED if dashed else OTHER,
                lw=0.9, ls="--" if dashed else "-", zorder=2, gid=f"trace:{t.spectrum_id}")
    ax.axvspan(*selected.window, color=SUMMED, lw=0, zorder=1, gid="summed")
    if edit is not None:
        ax.axvspan(*edit, facecolor="none", edgecolor=HATCH, hatch="///", lw=0, zorder=1.5,
                   gid="edited")
        ax.axvspan(*edit, facecolor="none", edgecolor=EDITED, lw=0.8, zorder=1.6,
                   gid="edited_outline")
    ax.plot(ref.times, ref.intensity, color=RAW_DARK, lw=1.6, zorder=4, marker="o", ms=2.5,
            gid=f"trace:{ref.spectrum_id}")
    _own_range(ax, selected, edit)
    ax.set_xlabel("retention time (min, this file's clock)")
    ax.set_ylabel("intensity at the ion's bins")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    handles = [Line2D([], [], color=RAW_DARK, lw=1.6, marker="o", ms=2.5,
                      label=f"this spectrum: {run_text(selected)}"),
               Patch(color=SUMMED, lw=0, label=f"scans summed ({selected.window[0]:.2f}-"
                                                f"{selected.window[1]:.2f} min)")]
    if edit is not None:
        handles.append(Patch(facecolor="none", edgecolor=HATCH, hatch="///", lw=0.8,
                             label=f"new window ({edit[0]:.2f}-{edit[1]:.2f} min)"))
    if matched:
        shifts = [60 * p.shift for p in matched]
        handles.append(Line2D([], [], color=OTHER, lw=0.9, label=(
            f"{traces.others_label} ({len(matched)}), each moved by its match "
            f"({min(shifts):+.0f} to {max(shifts):+.0f} s)\nand scaled to this one's height")))
    if unmatched:
        handles.append(Line2D([], [], color=UNMATCHED, lw=0.9, ls="--",
                              label=f"not matched ({len(unmatched)}): where they are"))
    ax.legend(handles=handles, loc="upper left", fontsize="small", frameon=False)
    if own_top > 0:
        ax.set_ylim(top=own_top * max(1.45, 1.25 + 0.09 * (len(handles) + 1)))


def _drift_panel(ax, traces: RetentionTraces,
                 edit: tuple[float, float] | None = None) -> None:
    """The selected spectrum's drift profile over its window, with its drift range shaded (and
    the range being edited hatched)."""
    p = traces.drift
    drift, y = np.asarray(p.drift, dtype=float), np.asarray(p.intensity, dtype=float)
    if p.range is not None:
        ax.axvspan(*p.range, color=SUMMED, lw=0, zorder=1, gid="drift_range")
    if edit is not None:
        ax.axvspan(*edit, facecolor="none", edgecolor=EDITED, hatch="///", lw=0.8, zorder=2,
                   gid="drift_edited")
    ax.plot(drift, y, color=RAW_DARK, lw=1.4, zorder=3, gid="drift_profile")
    ax.set_xlabel("drift time (ms)")
    ax.set_ylabel("drift profile on the isotope peaks")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    if len(y) and y.max() > 0:
        # the profile down to a small share of its apex and the range the run summed, with its
        # own width either side (at least three drift bins): a narrow peak shows its baseline
        # too (owner, 2026-11-12: a peak 3-4 bins wide was drawn cut off at its sides); a
        # range being edited only widens it, so the axis stays put between draws
        shown = drift[y >= _SHOWN_SHARE * y.max()]
        lo, hi = float(shown.min()), float(shown.max())
        if p.range is not None:
            lo, hi = min(lo, p.range[0]), max(hi, p.range[1])
        steps = np.diff(drift)
        step = float(np.median(steps[steps > 0])) if (steps > 0).any() else 0.0
        pad = max(hi - lo, 3 * step, 1e-3)
        lo, hi = lo - pad, hi + pad
        if edit is not None:
            lo, hi = min(lo, edit[0]), max(hi, edit[1])
        ax.set_xlim(lo, hi)
