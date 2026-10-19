# SPDX-License-Identifier: GPL-3.0-or-later
"""Writing the series figures to files: one multipage PDF (or one file per page for SVG and PNG)
for each selected series. Figures are made, saved and released one at a time, so a large run
does not hold them in memory."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from ..hooks import CancelHook, ProgressHook, check_cancel, report
from ..io.outputs import replace_outputs
from .results import Results
from .series import (
    Reference,
    plot_populations,
    plot_series_spectra,
    plot_uptake,
    series_info,
    series_rows,
)
from .spectrum import LAYERS

log = logging.getLogger(__name__)
KINDS = ("spectra", "populations", "uptake")
FORMATS = ("pdf", "svg", "png")


@dataclass(frozen=True)
class PlotConfig:
    """Which figures :func:`write_figures` writes (the ``[run.plots]`` table).

    - ``series``: ``'flagged'`` (series with a flagged or rejected spectrum, or a control that
      fell back; the default, because a large run has too many series to draw), ``'all'`` or a
      list of ``data_id``.
    - ``kinds``: the pages of a series' figure, any of ``spectra``, ``populations`` and
      ``uptake``. The ``populations`` page is left out of a series in which every spectrum has
      one population.
    - ``show``: the layers of the spectra page, any of ``raw``, ``picked`` and ``fit``;
      ``residual`` adds a residual panel under every spectrum.
    - ``x`` (``'time'`` or ``'index'``; None, the default: ``'time'``, or ``'index'`` when the
      times only order the points, ``[input] time_scale = "order"``, which also turns an
      explicit ``'time'`` into ``'index'`` with a warning) and ``scale`` (``'linear'`` or
      ``'log'``): the uptake axis (:func:`plot_uptake`).
    - ``corrected``: draw back-exchange corrected values; None: when the results have controls.
    - ``format``: ``'pdf'`` (one file per series, one page per kind), ``'svg'`` or ``'png'`` (one
      file per kind); ``dpi``: the resolution of PNG files and of raster images inside PDF or
      SVG files.
    - ``overview``: also write the run's overviews (K selection, controls, quality) as
      ``overview``.
    """

    series: str | tuple[int, ...] = "flagged"
    kinds: tuple[str, ...] = KINDS
    show: tuple[str, ...] = LAYERS
    x: str | None = None
    scale: str = "linear"
    corrected: bool | None = None
    format: str = "pdf"
    dpi: int = 300
    overview: bool = True
    residual: bool = False

    def __post_init__(self):
        if not isinstance(self.series, str):
            object.__setattr__(self, "series", tuple(int(s) for s in self.series))
        elif self.series not in ("flagged", "all"):
            raise ValueError("series must be 'flagged', 'all' or a tuple of data_id")
        object.__setattr__(self, "kinds", tuple(self.kinds))
        object.__setattr__(self, "show", tuple(self.show))
        if not self.kinds or any(k not in KINDS for k in self.kinds):
            raise ValueError(f"kinds must be a non-empty subset of {KINDS}")
        if not self.show or any(s not in LAYERS for s in self.show):
            raise ValueError(f"show must be a non-empty subset of {LAYERS}")
        if self.x not in ("time", "index", None) or self.scale not in ("linear", "log"):
            raise ValueError("x must be 'time' or 'index' and scale 'linear' or 'log'")
        if self.format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}")
        if self.dpi < 1:
            raise ValueError("dpi must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlotConfig:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown plot setting(s): {', '.join(sorted(unknown))}")
        return cls(**data)


def select_series(results: Results, which: str | Sequence[int] = "flagged") -> list[int]:
    """The ``data_id`` of the series to draw: all of them, the listed ones, or the flagged ones
    (a spectrum with a quality flag, a rejected spectrum or a control that fell back)."""
    every = [int(d) for d in np.unique(results.fit.data_id)]
    if not isinstance(which, str):
        missing = [int(d) for d in which if int(d) not in every]
        if missing:
            raise KeyError(missing)
        return [int(d) for d in which]
    if which == "all":
        return every
    if which != "flagged":
        raise ValueError("which must be 'flagged', 'all' or a list of data_id")
    bad = np.zeros(len(results.fit), dtype=bool)
    if results.quality is not None:
        bad |= results.quality.flagged
    if results.review is not None:
        bad |= results.review_status == "rejected"
    flagged = {int(d) for d in results.fit.data_id[bad]}
    if results.controls is not None:
        flagged |= {int(d) for d, f in zip(results.controls.data_id, results.controls.fallback,
                                           strict=True) if f}
    return [d for d in every if d in flagged]


def series_figures(
    results: Results, data_id: int, config: PlotConfig = PlotConfig(),  # noqa: B008
    reference: Reference | None = None,
) -> dict[str, Figure]:
    """The figures of one series asked for by ``config``, by kind (:func:`series_kinds`)."""
    out: dict[str, Figure] = {}
    for kind in series_kinds(results, data_id, config):
        if kind == "spectra":
            out[kind] = plot_series_spectra(results, data_id, show=config.show,
                                            corrected=config.corrected,
                                            residual=config.residual and "fit" in config.show
                                            and "picked" in config.show)
        elif kind == "populations":
            out[kind] = plot_populations(results, data_id, corrected=config.corrected,
                                         reference=reference)
        else:
            out[kind] = plot_uptake(results, data_id, x=uptake_axis(results, config),
                                    scale=config.scale, corrected=config.corrected,
                                    reference=reference)
    return out


def uptake_axis(results: Results, config: PlotConfig) -> str | None:
    """``config.x`` for ``results``: ``'time'`` becomes None (an index axis) when the times only
    order the points (:attr:`Results.time_scale` ``'order'``)."""
    if config.x == "time" and results.time_scale == "order":
        return None
    return config.x


def series_kinds(results: Results, data_id: int, config: PlotConfig) -> tuple[str, ...]:
    """The kinds of ``config`` drawn for a series: all of them, except the ``populations`` page
    when every spectrum of the series has one population at the K in effect (a map of one
    population shows nothing)."""
    kinds = config.kinds
    if "populations" in kinds:
        rows = series_rows(results, data_id)
        if not (results.effective_K[rows] > 1).any():
            kinds = tuple(k for k in kinds if k != "populations")
    return kinds


def figure_stems(results: Results, data_ids: Sequence[int]) -> dict[int, str]:
    """The file stem of each series' figure: its label (sample, range, peptide or fragment,
    charge, condition) as a file name; the ``data_id`` is added where two series would share a
    stem."""
    stems = {int(d): _slug(series_info(results, int(d))["file_label"]) for d in data_ids}
    counts = Counter(stems.values())
    return {d: stem if counts[stem] == 1 else f"{stem}_{d}" for d, stem in stems.items()}


def write_figures(
    results: Results,
    directory: str | Path,
    config: PlotConfig = PlotConfig(),  # noqa: B008
    *,
    reference: Reference | None = None,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> list[Path]:
    """Write the run's overviews (if ``config.overview``, as ``overview``) and the figures of
    the series selected by ``config`` into ``directory`` (named by :func:`figure_stems`), and
    return the files. A series' figures are made and released one by one. The figures hxex
    wrote there before are replaced (its listing, :func:`hxex.io.replace_outputs`), never a
    file of your own."""
    directory = Path(directory)
    chosen = select_series(results, config.series)
    stems = figure_stems(results, chosen)
    if chosen and "uptake" in config.kinds and config.x == "time" and \
            results.time_scale == "order":
        log.warning("plots: x = \"time\" is drawn by index: the times only order the points "
                    "([input] time_scale = \"order\")")
    names = [f"overview.{config.format}"] if config.overview else []
    for data_id, stem in stems.items():  # the files a series gives, known before it is drawn
        if config.format == "pdf":
            names.append(f"{stem}.pdf")
        else:
            names += [f"{stem}_{kind}.{config.format}"
                      for kind in series_kinds(results, data_id, config)]
    with replace_outputs(directory, names) as written:
        if config.overview:
            from .overview import write_overview

            written += write_overview(results, directory / f"overview.{config.format}",
                                      dpi=config.dpi)
        for n, data_id in enumerate(chosen):
            check_cancel(cancel)
            stem = stems[data_id]
            figures = series_figures(results, data_id, config, reference)
            if config.format == "pdf":
                path = directory / f"{stem}.pdf"
                with PdfPages(path) as pdf:
                    for fig in figures.values():
                        pdf.savefig(fig, dpi=config.dpi)
                written.append(path)
            else:
                for kind, fig in figures.items():
                    path = directory / f"{stem}_{kind}.{config.format}"
                    fig.savefig(path, dpi=config.dpi)
                    written.append(path)
            report(progress, n + 1, len(chosen))
    log.info("wrote %d figure file(s) for %d series in %s", len(written), len(chosen), directory)
    return list(written)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60] or "series"
