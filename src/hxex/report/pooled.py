# SPDX-License-Identifier: GPL-3.0-or-later
"""The figures of pooled samples: each form of the pool's minimum set
(:mod:`hxex.analysis.pooled`) residue by residue, every sample on one axis; and the sequences
measured over one range side by side (head to head)."""

from __future__ import annotations

import textwrap
from collections.abc import Collection, Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import chem
from ..analysis.pooled import PooledForm
from ..analysis.residues import ResidueMap
from . import hover
from .compare import plot_residue_samples
from .coverage import SampleStyle, legend_columns

FORM_PANEL = 2.6  # inches of page height per form
OWN_FILL, OWN_EDGE = "0.91", "black"  # the strip of a form's own residues: no sample's colour
OWN_LABEL = "measured by this form's own peptides"


def own_spans(form: PooledForm, nterm_subtract: int,
              registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS
              ) -> list[tuple[float, float]]:
    """The stretches of residues ``form``'s own peptides measure (each from its first
    exchanging amide to its last residue), overlapping ones joined, as x limits."""
    spans = sorted((s + chem.effective_nterm_subtract(p, nterm_subtract) - 0.5, e + 0.5)
                   for p, s, e in form.own)
    out: list[list[float]] = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


NOTE_HEIGHT = 0.32  # inches under a figure for the provisional note


def _spans_text(spans: Sequence[tuple[float, float]]) -> str:
    return ", ".join(f"{int(a + 0.5)}-{int(b - 0.5)}" for a, b in spans)


def provisional(fig: Figure, spans: Sequence[tuple[float, float]] | None = None, *,
                strip: bool = True) -> Figure:
    """``fig``, a residue figure of a form of a pool (owner, 2026-11-10), marked provisional
    by a note under the figure (the figure grows to hold it). ``spans``: the residues the
    form's own peptides measure (:func:`own_spans`; None for a figure of several forms):
    with ``strip``, drawn as on the forms pages, a grey strip along the bottom of every panel
    with residues on its x axis; otherwise (maps of cells) named in the note."""
    if spans is None:
        where = ("outside the residues of each form's own peptides (the grey strip of the "
                 "forms pages)")
    elif not spans:
        where = "the form has no peptides of its own, so everywhere"
    elif strip:
        where = "outside the grey strip (the residues of its own peptides)"
        for ax in fig.axes:
            if any("residue" in a.get_xlabel().lower()
                   for a in ax.get_shared_x_axes().get_siblings(ax)):
                for a, b in spans:
                    ax.axvspan(a, b, ymin=0.0, ymax=0.04, facecolor=OWN_FILL,
                               edgecolor=OWN_EDGE, lw=0.6, zorder=0.8)
    else:
        where = f"outside residues {_spans_text(spans)} (those of its own peptides)"
    note = (f"Provisional, a form of a pool: {where} the values mix the forms that share "
            "the peptides, and a mixture of forms can look like two populations.")
    height = fig.get_figheight()
    fig.set_figheight(height + NOTE_HEIGHT)
    share = NOTE_HEIGHT / (height + NOTE_HEIGHT)
    engine = fig.get_layout_engine()
    if isinstance(engine, ConstrainedLayoutEngine):
        engine.set(rect=(0.0, share, 1.0, 1.0 - share))
    else:  # the axes keep their size in inches: the old fractions scaled to the new height
        pars = fig.subplotpars
        scale = height / (height + NOTE_HEIGHT)
        fig.subplots_adjust(bottom=share + pars.bottom * scale, top=share + pars.top * scale)
        for legend in fig.legends:  # a legend placed in the figure moves up with the axes
            legend.set_bbox_to_anchor((0.0, share, 1.0, 1.0 - share), transform=fig.transFigure)
    per_line = max(40, int(fig.get_figwidth() * 17))
    fig.text(0.5, 0.004, "\n".join(textwrap.wrap(note, per_line)), ha="center", va="bottom",
             fontsize=7, color="0.35", style="italic")
    return fig


def name_forms(fig: Figure, forms: Mapping[str, tuple[str, str]]) -> Figure:
    """``fig``, a figure of the forms of a pool, with the rows its artists carry (the hover
    rows, :mod:`hxex.report.hover`) naming the sample the results know: a form state's
    ``sample`` (``'fresh ALPHA1'``, as the form tables name it) becomes its sample, the form a
    ``form`` entry of its own, so a peptide clicked in the figure leads to its spectra (owner,
    2026-11-10). ``forms``: each form state's (sample, form); a ``reference`` named so
    becomes its sample too."""
    for artist in fig.findobj():
        for attr in (hover._ROW, hover._ROWS):
            value = getattr(artist, attr, None)
            if not value:
                continue
            rows = [value] if isinstance(value, dict) else value
            for row in rows:
                state = row.get("sample")
                if state in forms:
                    row["sample"], row["form"] = forms[state]
                if row.get("reference") in forms:
                    row["reference"] = forms[row["reference"]][0]
    return fig


OWN_NOTE = "outlined: the form's own peptides"


def outline_own(fig: Figure, own: Collection[tuple[str, int, int]]) -> Figure:
    """``fig``, a Woods plot of a form of a pool, with the bars of the form's ``own`` peptides
    (``(peptide, start, end)``) outlined in black, as on the form's coverage map, and a note
    saying so in each panel that has one."""
    for ax in fig.axes:
        marked = False
        for line in list(ax.get_lines()):
            row = hover.row_of(line)
            if not str(line.get_gid() or "").startswith("peptide:") or not row:
                continue
            if (row.get("peptide"), row.get("start"), row.get("end")) in own:
                ax.plot(line.get_xdata(), line.get_ydata(), color=OWN_EDGE,
                        linewidth=line.get_linewidth() + 2.0, solid_capstyle="butt",
                        zorder=line.get_zorder() - 0.05)
                marked = True
        if marked:
            ax.text(0.01, 0.97, OWN_NOTE, transform=ax.transAxes, ha="left", va="top",
                    fontsize=7, color="0.3")
    return fig


def plot_pooled_forms(
    residues: ResidueMap,
    forms: Sequence[PooledForm],
    samples: Sequence[str],
    time: float,
    *,
    names: Mapping[str, str] | None = None,
    styles: Mapping[str, SampleStyle] | None = None,
    nterm_subtract: int = 1,
    length: int | None = None,
    top: float | None = None,
    delta_crit: float | None = None,
    time_scale: str = "seconds",
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
) -> Figure:
    """One panel per form of a pool (owner, 2026-11-09): every sample's fDavg residue by
    residue at ``time``, as :func:`hxex.report.plot_residue_samples` draws it, from the
    peptides the form may have given; a grey strip, outlined in black, along the bottom marks
    the residues its own peptides measure (no other form of the set may have given them).
    Elsewhere the value is the mixture of the forms that share the peptides. ``names``: the
    state each sample's form has in ``residues`` (``(sample, form name)`` joined by a space
    when not given); ``styles``: each sample's style; a modified form whose peptides fit
    more records than its members names them in its title; ``top``: the highest value the y
    axis must hold."""
    n = max(len(forms), 1)
    fig = Figure(figsize=(9.0, 0.7 + FORM_PANEL * n), layout="constrained")
    axes = fig.subplots(n, 1, sharex=True, squeeze=False)[:, 0]
    styles = dict(styles or {})
    for ax, form in zip(axes, forms, strict=False):
        states = [(names or {}).get((s, form.name), f"{s} {form.name}") for s in samples]
        panel_styles = {state: styles[s] for state, s in zip(states, samples, strict=True)
                        if s in styles}
        plot_residue_samples(residues, states, time, styles=panel_styles, ax=ax, length=length,
                             top=top, delta_crit=delta_crit, time_scale=time_scale)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        for a, b in own_spans(form, nterm_subtract, registry):
            ax.axvspan(a, b, ymin=0.0, ymax=0.05, facecolor=OWN_FILL, edgecolor=OWN_EDGE,
                       lw=0.8, zorder=0.8)
        also = [c for c in form.candidates if c not in form.members]
        title = f"{form.name}: {len(form.own)} of its {len(form.peptides)} peptides are its own"
        if also:
            title += f" (the modified peptides also fit {', '.join(also)})"
        ax.set_title(title, loc="left", fontsize=9)
        if ax is not axes[-1]:
            ax.set_xlabel("")
    handles = [Line2D([], [], color=styles[s].colour, marker=styles[s].marker, label=s)
               if s in styles else Line2D([], [], label=s) for s in samples]
    handles.append(Patch(facecolor=OWN_FILL, edgecolor=OWN_EDGE, lw=0.8, label=OWN_LABEL))
    labels = [h.get_label() for h in handles]
    fig.legend(handles=handles, loc="outside upper center", fontsize=8, frameon=False,
               ncol=legend_columns(axes[0], labels, fontsize=8))
    return fig


HEAD_TO_HEAD_PANEL = (3.6, 3.4)  # inches per panel, width and height
HEAD_TO_HEAD_COLUMNS = 3
# values whose fully deuterated control is a stand-in, on the head-to-head page
STAND_IN_ALPHA = 0.3
SEQUENCE_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def _tick(name: str) -> str:
    """A sample name as an x tick: long names broken at their first space or underscore."""
    if len(name) > 8:
        for sep in (" ", "_"):
            if sep in name:
                head, tail = name.split(sep, 1)
                return f"{head}\n{tail}"
    return name


def plot_head_to_head(
    uptake,
    panels: Sequence,
    samples: Sequence[str],
    time: float,
    *,
    styles: Mapping[str, SampleStyle] | None = None,
    top: float | None = None,
    columns: int = HEAD_TO_HEAD_COLUMNS,
    time_scale: str = "seconds",
    stand_ins: Collection[tuple] = (),
) -> Figure:
    """The pooled sequences of each panel (:class:`hxex.analysis.pooled.HeadToHead`) side by
    side at ``time`` (owner, 2026-11-09, layout A): one panel per range, the samples along
    x, each sequence its own marker shape beside the others (the first, which the others are
    compared with in ``compare_sequences.csv``, filled; the others hollow), each sample in its
    colour, ±1 standard error bars; the K=1 fD of the species table ``uptake`` (its samples
    named as ``samples``). ``top``: the y axis's upper end, shared by every panel and page (the
    forms pages' own), from 0. ``stand_ins``: (sample, peptide, start, end) whose correction
    is not from controls of their own sequence, drawn pale: their rows in
    ``compare_sequences.csv`` are not tested."""
    from .hover import set_row, time_label

    styles = dict(styles or {})
    n = max(len(panels), 1)
    cols = min(columns, n)
    rows = -(-n // cols)
    fig = Figure(figsize=(HEAD_TO_HEAD_PANEL[0] * cols, 0.4 + HEAD_TO_HEAD_PANEL[1] * rows),
                 layout="constrained")
    axes = fig.subplots(rows, cols, squeeze=False).ravel()
    sel = uptake.where(K=1, rank=1, role="", time=time)
    value = {(str(uptake.sample[i]), str(uptake.peptide[i]), int(uptake.start[i]),
              int(uptake.end[i])): (float(uptake.fD[i]), float(uptake.fD_se[i]))
             for i in np.flatnonzero(sel)}
    x = np.arange(len(samples))
    when = time_label(time, time_scale)
    stand_ins = set(stand_ins)
    for ax, panel in zip(axes, panels, strict=False):
        shown_pale = False
        k = len(panel.sequences)
        width = 0.18 if k > 1 else 0.0
        for j, (seq, label) in enumerate(zip(panel.sequences, panel.labels, strict=True)):
            off = (j - (k - 1) / 2) * width
            marker = SEQUENCE_MARKERS[j % len(SEQUENCE_MARKERS)]
            for i, s in enumerate(samples):
                got = value.get((s, seq, panel.start, panel.end))
                if got is None or not np.isfinite(got[0]):
                    continue
                colour = styles[s].colour if s in styles else "0.3"
                pale = (s, seq, panel.start, panel.end) in stand_ins
                bars = ax.errorbar(x[i] + off, got[0], got[1] if np.isfinite(got[1]) else None,
                                   fmt=marker, color=colour, ms=6, capsize=2,
                                   mfc=colour if j == 0 else "white", zorder=2,
                                   alpha=STAND_IN_ALPHA if pale else 1.0)
                shown_pale |= pale
                set_row(bars[0], {"sample": s, "range": panel.name, "peptide": seq,
                                  "sequence": label, "time": time, "fD": got[0],
                                  "fD_se": got[1],
                                  "controls": "stand-in: not compared" if pale else "own"})
        handles = [Line2D([], [], marker=SEQUENCE_MARKERS[j % len(SEQUENCE_MARKERS)],
                          color="0.3", ls="none", mfc="0.3" if j == 0 else "white", label=lab)
                   for j, lab in enumerate(panel.labels)]
        if shown_pale:
            handles.append(Line2D([], [], marker="o", color="0.3", ls="none",
                                  alpha=STAND_IN_ALPHA, label="pale: stand-in control, not "
                                                              "compared"))
        ax.legend(handles=handles, fontsize=6.5, loc="best", frameon=False)
        ax.set_xticks(x, [_tick(s) for s in samples], fontsize=7)
        ax.set_xlim(-0.6, len(samples) - 0.4)
        if top is not None and np.isfinite(top):
            ax.set_ylim(0, top)
        ax.set_ylabel(r"fD$_\mathrm{avg}$")
        ax.set_title(panel.name + (f" ({panel.group})" if panel.group else ""), loc="left",
                     fontsize=9)
        ax.set_title(when, loc="right", fontsize=8)
        ax.grid(axis="y", color="0.9")
        ax.set_axisbelow(True)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    return fig
