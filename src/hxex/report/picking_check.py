# SPDX-License-Identifier: GPL-3.0-or-later
"""The picking check of a spectrum read with the profile fit (``[run.picker] pick =
"profile_fit"``; docs/PLAN.md Next steps 33, step 1): the raw profile, the peptide's fitted
peaks, the other species' (stray) peaks, and the bins' values, with the spectrum's purity and
stray share. A figure of its own, not drawn over the populations' fit (owner, 2026-11-14: too
busy on a fit plot)."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .results import Results
from .spectrum import OWN_WIDTH, STICK

STRAY_COLOR = "#E69F00"  # orange: not the peptide (owner, 2026-11-14); no population curve here
# the raw profile lighter than in the spectrum figure: the peptide's fitted peaks (dark) lie on
# it in a clean spectrum, and both must stay visible
RAW_LIGHT = "#a0a0a0"


def picking_check_data(spectrum_id: str, results: Results) -> dict:
    """What :func:`plot_picking_check` draws: the region's ``x``, the raw profile ``y``, the
    peptide's fitted peaks (``bins``), the strays' curve (``strays``), the bins' fitted m/z and
    values (``where``, ``values``), ``offset_ppm``, ``purity`` and ``stray_share`` (of this
    fit, which is the run's: same seed, same offset), and ``label``. ValueError when the
    spectrum was not read with the profile fit."""
    from ..picking.profile import model_spectrum

    b = results.binned
    if b is None:
        raise ValueError("the results hold no picked table")
    i = b.find(spectrum_id)
    offset = float(np.asarray(b.fit_offset_ppm, dtype=float)[i]) \
        if b.fit_offset_ppm is not None else float("nan")
    if not np.isfinite(offset):
        raise ValueError(f"{spectrum_id} was not read with the profile fit (it kept the pick: a "
                         "fragment, another ion type, its own envelope, or centroided data)")
    raw = results.raw_spectrum(spectrum_id)
    if raw is None:
        raise ValueError(f"no raw spectrum stored for {spectrum_id}")
    s = results.settings
    fit = model_spectrum(raw[0], raw[1], str(b.peptide[i]), int(b.charge[i]), offset_ppm=offset,
                         measure=s.measure, config=s.picker, registry=s.registry,
                         modification=str(b.modification[i]),
                         resolving_power=_recorded(b, "resolving_power", i),
                         resolving_power_mz=_recorded(b, "resolving_power_mz", i))
    if fit is None:
        raise ValueError(f"nothing to fit in {spectrum_id}")
    return {"x": fit.x, "y": fit.y, "bins": fit.bins_curve, "strays": fit.stray_curve,
            "where": fit.where, "values": fit.values, "offset_ppm": offset,
            "purity": fit.purity, "stray_share": fit.stray_share,
            "label": str(spectrum_id)}


def _recorded(b, name: str, i: int) -> float | None:
    v = getattr(b, name, None)
    return None if v is None else float(np.asarray(v, dtype=float)[i])


def plot_picking_check(spectrum_id: str, results: Results, *, ax: Axes | None = None,
                       legend: bool = True, title: bool = True,
                       mz_range: tuple[float, float] | None = None) -> Figure:
    """Draw the profile fit of one spectrum (:func:`picking_check_data`): the raw profile (grey),
    the peptide's fitted peaks (dark, on top), the stray peaks of other species (orange, dashed),
    and
    the bins' values as up-pointing triangles (areas as height-equivalents, so they need not
    sit on the profile's apexes). The title gives the purity, the stray share, and the m/z
    offset the fit took. Returns the figure; nothing is shown or saved. Artists' ``gid``:
    ``raw``, ``bins``, ``strays``, ``values``."""
    d = picking_check_data(spectrum_id, results)
    own = ax is None
    if own:
        fig = Figure(figsize=(OWN_WIDTH + (3.2 if legend else 0.0), 3.5),
                     layout="constrained" if legend else None)
        ax = fig.subplots()
    else:
        fig = ax.figure
    ax.plot(d["x"], d["y"], color=RAW_LIGHT, lw=2.2, label="raw profile", gid="raw")
    ax.plot(d["x"], d["bins"], color=STICK, lw=1.4, label="the peptide's fitted peaks",
            gid="bins")
    if np.any(d["strays"] > 0):
        ax.plot(d["x"], d["strays"], color=STRAY_COLOR, lw=1.4, ls="--",
                label="other species (stray peaks)", gid="strays")
    ok = np.isfinite(d["where"]) & (d["values"] > 0)
    ax.scatter(d["where"][ok], d["values"][ok], marker="^", s=26, color="k", zorder=4, lw=0,
               label="bin values (areas)", gid="values")
    ax.set_xlabel("m/z")
    ax.set_ylabel("intensity")
    if mz_range is not None:
        ax.set_xlim(*mz_range)
    if title:
        ax.set_title(f"{d['label']}\npurity {d['purity']:.2f}   stray share "
                     f"{d['stray_share']:.2f}   offset {d['offset_ppm']:+.1f} ppm",
                     fontsize=9, loc="left")
    if legend:
        if own:
            ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        else:
            ax.legend(fontsize=7, frameon=False)
    return fig
