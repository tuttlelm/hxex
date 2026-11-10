# SPDX-License-Identifier: GPL-3.0-or-later
"""The figures of pooled samples: each form of the pool's minimum set
(:mod:`hxex.analysis.pooled`) residue by residue, every sample on one axis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import chem
from ..analysis.pooled import PooledForm
from ..analysis.residues import ResidueMap
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
