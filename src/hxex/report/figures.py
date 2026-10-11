# SPDX-License-Identifier: GPL-3.0-or-later
"""Writing the series figures to files: one multipage PDF (or one file per page for SVG and PNG)
for each selected series. Figures are made, saved and released one at a time, so a large run
does not hold them in memory."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from ..hooks import CancelHook, ProgressHook, check_cancel, report
from .results import Results
from .series import (
    Reference,
    plot_populations,
    plot_series_spectra,
    plot_uptake,
    series_info,
)
from .spectrum import LAYERS

log = logging.getLogger(__name__)
KINDS = ("spectra", "populations", "uptake")
FORMATS = ("pdf", "svg", "png")


@dataclass(frozen=True)
class PlotConfig:
    """Which figures to write. ``series`` is ``'flagged'`` (series with a flagged or rejected
    spectrum, or a control that fell back; the default, because a large run has too many series
    to draw), ``'all'`` or a tuple of ``data_id``. ``kinds`` is a subset of ``spectra``,
    ``populations`` and ``uptake``; ``show`` the layers of the spectra; ``x`` and ``scale`` the
    uptake axis; ``corrected`` the back-exchange correction (None: when the results have
    controls); ``format`` is ``pdf`` (one file per series, one page per kind), ``svg`` or ``png``
    (one file per kind)."""

    series: str | tuple[int, ...] = "flagged"
    kinds: tuple[str, ...] = KINDS
    show: tuple[str, ...] = LAYERS
    x: str = "time"
    scale: str = "linear"
    corrected: bool | None = None
    format: str = "pdf"
    dpi: int = 200

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
        if self.x not in ("time", "index") or self.scale not in ("linear", "log"):
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
        bad |= results.review.status_of(results.fit) == "rejected"
    flagged = {int(d) for d in results.fit.data_id[bad]}
    if results.controls is not None:
        flagged |= {int(d) for d, f in zip(results.controls.data_id, results.controls.fallback,
                                           strict=True) if f}
    return [d for d in every if d in flagged]


def series_figures(
    results: Results, data_id: int, config: PlotConfig = PlotConfig(),  # noqa: B008
    reference: Reference | None = None,
) -> dict[str, Figure]:
    """The figures of one series asked for by ``config``, by kind."""
    out: dict[str, Figure] = {}
    for kind in config.kinds:
        if kind == "spectra":
            out[kind] = plot_series_spectra(results, data_id, show=config.show,
                                            corrected=config.corrected)
        elif kind == "populations":
            out[kind] = plot_populations(results, data_id, corrected=config.corrected,
                                         reference=reference)
        else:
            out[kind] = plot_uptake(results, data_id, x=config.x, scale=config.scale,
                                    corrected=config.corrected, reference=reference)
    return out


def write_figures(
    results: Results,
    directory: str | Path,
    config: PlotConfig = PlotConfig(),  # noqa: B008
    *,
    reference: Reference | None = None,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> list[Path]:
    """Write the figures of the series selected by ``config`` into ``directory`` and return the
    files. A series' figures are made and released one by one."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    chosen = select_series(results, config.series)
    written: list[Path] = []
    for n, data_id in enumerate(chosen):
        check_cancel(cancel)
        stem = f"hxex_{data_id}_{_slug(series_info(results, data_id)['label'])}"
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
    log.info("wrote %d figure file(s) for %d series", len(written), len(chosen))
    return written


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60] or "series"
