# SPDX-License-Identifier: GPL-3.0-or-later
"""Results in a notebook, in as few lines as possible (the tutorial notebook uses these):
:func:`load_results` loads a project's results as ``hxex report`` sees them, and
:func:`show_series` draws one peptide's figures."""

from __future__ import annotations

import dataclasses
import io
import logging
from collections.abc import Sequence
from pathlib import Path

from matplotlib.figure import Figure

from .figures import PlotConfig, series_figures
from .results import Results
from .tables import find_series

log = logging.getLogger(__name__)


def load_results(path: str | Path = "hxex.toml") -> Results:
    """The results of a project file (its bundle with its ``[review]`` applied, as ``hxex report
    hxex.toml`` uses them, and its ``[figures]`` settings, which :func:`show_series` draws
    with) or of a bundle folder or zip (its own review layer only). Load them again after
    changing the project file."""
    from ..io import open_bundle
    from ..review_table import apply_view

    bundle, review, figures = Path(path), None, None
    if bundle.suffix == ".toml":
        from ..config import load_project

        project = load_project(bundle, analysis_only=True)
        if project.output.bundle is None:
            raise ValueError(f"{bundle} names no [output] bundle")
        bundle, review = project.output.bundle, project.review
        figures = project.figures or PlotConfig()
    with open_bundle(bundle) as b:
        if b.pending:
            log.warning("%s is waiting for its %s ('hxex resume' adds it)", bundle,
                        ", ".join(b.pending))
        results = dataclasses.replace(apply_view(Results.from_bundle(b), review), figures=figures)
    if results.loader is not None:  # the bundle is closed now (a zip cannot be read closed)
        results = dataclasses.replace(results, loader=_reopening(bundle))
    return results


def _reopening(bundle: Path):
    """A raw-spectra loader that opens the bundle for each request (the results outlive the
    bundle they were loaded from)."""
    from ..io import open_bundle

    def load(ids):
        with open_bundle(bundle) as b:
            return b.spectra(ids)

    return load


def show_series(
    results: Results,
    sample: str,
    peptide_range: str,
    charge: int | None = None,
    *,
    k: int | None = None,
    kinds: Sequence[str] | None = None,
    scale: str | None = None,
    format: str = "png",  # noqa: A002 (as the format of [figures])
    dpi: int = 80,
) -> list[Figure] | None:
    """Draw the figures of one peptide (``peptide_range`` by residues, ``"13-24"``) of
    ``sample``, every charge state unless ``charge`` is given: the pages of ``hxex report
    --figures``, drawn with the project's ``[figures]`` settings (:attr:`Results.figures`;
    the defaults for results not loaded from a project file). ``kinds`` (any of ``spectra``,
    ``populations`` and ``uptake``), ``k`` (every spectrum at this number of populations) and
    ``scale`` (of the uptake's time axis, ``'linear'`` or ``'log'``) change them for this
    call. In a notebook they are shown and nothing is returned; elsewhere the figures are
    returned. ``format`` ``'png'`` shows a sharp image (drawn at twice ``dpi`` and shown at
    the size of ``dpi``, as for a high-resolution screen); ``'svg'`` a vector image, sharp at
    any zoom but slow to show for a large spectra page."""
    if format not in ("png", "svg"):
        raise ValueError("format must be 'png' or 'svg'")
    ids = find_series(results, samples=[sample], peptide_ranges=[peptide_range],
                      charges=None if charge is None else [charge])
    if not ids:
        raise ValueError(f"no series of {sample} {peptide_range}"
                         + ("" if charge is None else f" at charge {charge}"))
    config = results.figures or PlotConfig()
    changes = {"k": k, "kinds": None if kinds is None else tuple(kinds), "scale": scale}
    config = dataclasses.replace(config, **{n: v for n, v in changes.items() if v is not None})
    figures = [f for d in ids for f in series_figures(results, d, config).values()]
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return figures
    if get_ipython() is None:
        return figures
    for figure in figures:
        display(notebook_image(figure, format, dpi))
    return None


def notebook_image(figure: Figure, format: str = "png", dpi: int = 80):  # noqa: A002
    """``figure`` as an IPython image (hxex draws without pyplot, so a notebook's inline
    backend never sees its figures): a PNG drawn at twice ``dpi`` and shown at the size of
    ``dpi`` (sharp on any screen), or an SVG."""
    from IPython.display import SVG, Image

    buffer = io.BytesIO()
    if format == "svg":
        figure.savefig(buffer, format="svg")
        return SVG(data=buffer.getvalue())
    figure.savefig(buffer, format="png", dpi=2 * dpi)
    return Image(data=buffer.getvalue(), format="png", retina=True)
