# SPDX-License-Identifier: GPL-3.0-or-later
"""Plots of one series (a peptide at one charge): spectra by time and replicate, the populations
of every spectrum, and the uptake along the time course.

Each plot has a ``*_data`` function that returns the arrays it draws; the plot only draws them.
Populations are identified by the rank of their mean within each spectrum (1 = lowest), as
chosen for every time point independently.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..controls import correct_means
from ..io.times import format_time
from ..spectra import FULLDEUT, UNDEUT, time_index
from .results import Results
from .spectrum import (
    LAYERS,
    drawn_layers,
    page_legend_handles,
    peak_measure,
    plot_spectrum,
    population_style,
    spectrum_notes,
    wrap_title,
)

log = logging.getLogger(__name__)

# replicate shapes in the overlay of the population map (there the colour is the population);
# twelve, so that a pooled run's replicates stay apart (they repeat after twelve)
REP_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h", "p", "<", ">", "*")

# replicates (uptake plot): Okabe-Ito colours, the first two differing most; past the six
# colours the marker changes too (UPTAKE_MARKERS), so that no two replicates look alike
REP_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")
UPTAKE_MARKERS = ("v", "s", "D", "P", "X", "h")
REJECTED_STYLE = {"color": "grey", "fillstyle": "none", "alpha": 0.6}


def rep_style(rep: int) -> tuple[str, str]:
    """(colour, marker) of replicate ``rep`` (from 1) on the uptake page: the six colours of
    ``REP_COLORS`` with a down triangle, then the same colours with the next marker of
    ``UPTAKE_MARKERS``, so that replicates differ in colour or marker up to 36."""
    i = max(int(rep) - 1, 0)
    return (REP_COLORS[i % len(REP_COLORS)],
            UPTAKE_MARKERS[(i // len(REP_COLORS)) % len(UPTAKE_MARKERS)])


def _page_header(fig: Figure, label: str, handles: list) -> None:
    """The series title in a band at the top with the page's one legend under it, both placed
    in inches (a fraction of the height would drop them into the first row of a tall figure),
    and the panels laid out below them. The band is added to the figure's height, so the panels
    keep their size however many rows there are (taken from it, a one-row page lost a third of
    its height and its panels were squashed: owner, 2026-11-03)."""
    width = fig.get_figwidth()
    # a long title (a fragment's) wrapped to the page, every piece kept; each line more gives
    # the band one line's height (10 points and the line spacing)
    label = wrap_title(label, width - 0.4, 10)
    extra = 0.17 * label.count("\n")
    top = 0.45 + extra  # the title's band (inches)
    ncol = rows = 0
    if handles:
        ncol = max(1, min(len(handles), int(width / 2.0)))
        rows = math.ceil(len(handles) / ncol)
        top = 0.4 + extra + 0.17 * rows + 0.08  # the legend under the title, and a small gap
    fig.set_figheight(fig.get_figheight() + top)
    height = fig.get_figheight()
    fig.suptitle(label, fontsize=10, y=1 - 0.1 / height, va="top")
    if handles:
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1 - (0.38 + extra) / height),
                   ncol=ncol, frameon=False, fontsize=7, handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 1 - top / height))


@dataclass(frozen=True)
class Reference:
    """Known populations of a test data set, to draw next to the fitted ones: for each time, the
    known means as fractional deuteration (per exchangeable amide) and the known populations."""

    times: np.ndarray
    level: np.ndarray  # (T, P) fractional deuteration, NaN where the component is absent
    population: np.ndarray  # (T, P)

    def at(self, time: float, n_amides: float, atol: float = 1e-6):
        """(means in D, populations) of the components known at ``time``, or None."""
        hit = np.flatnonzero(np.abs(self.times - time) <= atol)
        if not len(hit):
            return None
        lv, fr = self.level[hit[0]], self.population[hit[0]]
        keep = np.isfinite(lv) & np.isfinite(fr) & (fr > 0)
        return lv[keep] * n_amides, fr[keep]


def load_reference(path) -> Reference:
    """Read a tab-separated table of known populations: ``time``, ``fd1``..., ``p1``... (the
    layout of the Bimodal test set; ``fd`` in percent of the amides, ``#N/A`` for none)."""
    from pathlib import Path

    rows = [r.split("\t") for r in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if r.strip()]
    head = [h.strip().lower() for h in rows[0]]
    fd_cols = [j for j, h in enumerate(head) if h.startswith("fd")]
    p_cols = [j for j, h in enumerate(head) if h.startswith("p") and h[1:].isdigit()]

    def num(text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return float("nan")

    times = np.array([num(r[0]) for r in rows[1:]])
    level = np.array([[num(r[j]) / 100.0 for j in fd_cols] for r in rows[1:]])
    frac = np.array([[num(r[j]) for j in p_cols] for r in rows[1:]])
    return Reference(times, level, frac)


def series_rows(results: Results, data_id: int) -> np.ndarray:
    """Positions of the series' spectra ordered by time, then replicate (the order the picked
    table gives within a time; without it, the order in the results)."""
    rows = np.flatnonzero(results.fit.data_id == data_id)
    if not len(rows):
        raise KeyError(data_id)
    rep = _reps(results)[rows]
    fit = results.fit
    rank = time_index(fit.data_id[rows], fit.time[rows], fit.role[rows])
    return rows[np.lexsort((np.arange(len(rows)), rep, rank))]


def _reps(results: Results) -> np.ndarray:
    """Replicate number of every spectrum (:attr:`Results.replicates`, computed once)."""
    return results.replicates


def replicate_numbers(results: Results) -> np.ndarray:
    """Replicate number of every spectrum (from the picked table; else 1, 2, ... within a time)."""
    fit = results.fit
    if results.binned is not None:
        index = {str(s): j for j, s in enumerate(results.binned.spectrum_id)}
        if all(str(s) in index for s in fit.spectrum_id):
            return np.array([int(results.binned.rep[index[str(s)]]) for s in fit.spectrum_id])
    out = np.ones(len(fit), dtype=np.int64)
    seen: dict[tuple[int, float], int] = {}
    for i in range(len(fit)):
        key = (int(fit.data_id[i]), float(fit.time[i]))
        seen[key] = seen.get(key, 0) + 1
        out[i] = seen[key]
    return out


def _fragment_of(results: Results, data_id: int) -> dict | None:
    """The fragment ion of a series read from scan files (its ``fragment`` label such as
    ``c-1:6``, the ``parent_range``, the ``parent_charge`` and the ``condition``), or None for
    a peptide or other data."""
    from ..io.mzml import SCAN_READERS, MzmlTarget

    source = results.settings.spectra_source if results.settings is not None else None
    if not source or source.get("reader") not in SCAN_READERS or "series" not in source:
        return None
    for s in source["series"]:
        if int(s["data_id"]) == int(data_id):
            target = MzmlTarget.from_dict(source["targets"][int(s["target"])])
            extra = dict(target.entry.extra)
            if "fragment" not in extra:
                return None
            return {"fragment": str(extra["fragment"]),
                    "parent_range": str(extra.get("parent_range", "")),
                    "parent_charge": int(extra.get("parent_charge", 0) or 0),
                    "condition": str(extra.get("condition") or target.activation or "")}
    return None


def series_info(results: Results, data_id: int) -> dict:
    """Description of a series: ``rows``, ``times``, ``reps``, ``time_index`` of each spectrum,
    ``label`` (sample, range, peptide, charge when the picked table is there; for a fragment
    ion its range and sequence, the fragment label such as ``c-1:6`` with its charge, the
    parent's range and charge and the condition), ``file_label`` (the label for file names:
    a fragment's without the sequence, so that the fragment is kept when names are
    shortened) and ``n_amides`` (from the controls, else None)."""
    rows = series_rows(results, data_id)
    fit = results.fit
    label = file_label = f"series {data_id}"
    sample = ""
    if results.binned is not None:
        j = results.binned.find(str(fit.spectrum_id[rows[0]]))
        b = results.binned
        sample = str(b.sample[j])
        z = int(b.charge[j])
        label = file_label = f"{b.sample[j]} {b.peptide_range[j]} {b.peptide[j]} z={z}".strip()
        frag = _fragment_of(results, data_id)
        if frag is not None:
            pz, cond = frag["parent_charge"], frag["condition"]
            label = (f"{b.sample[j]} {b.peptide_range[j]} {b.peptide[j]} {frag['fragment']} "
                     f"z={z}, parent {frag['parent_range']} z={pz}"
                     + (f", {cond}" if cond else "")).strip()
            file_label = (f"{b.sample[j]} {b.peptide_range[j]} {frag['fragment']} z{z} "
                          f"parent z{pz} {cond}").strip()
    row = _control_row(results, data_id)
    n_amides = None if row is None else float(results.controls.n_amides[row])
    return {
        "data_id": data_id, "rows": rows, "label": label, "file_label": file_label,
        "sample": sample, "n_amides": n_amides,
        "times": fit.time[rows], "roles": fit.role[rows], "reps": _reps(results)[rows],
        "time_index": time_index(fit.data_id[rows], fit.time[rows], fit.role[rows]),
    }


PAGE_ROWS, PAGE_COLUMNS = 8, 6  # the most rows and columns of a spectra page (owner)
MAX_POINTS = 48  # the most time points a series' spectra pages draw (a kinetics run thinned)


def _clashing(info: dict) -> bool:
    """Whether two spectra of a series share a time point and a replicate number."""
    pairs = list(zip(info["time_index"].tolist(), info["reps"].tolist(), strict=True))
    return len(set(pairs)) < len(pairs)


def _by_replicate(info: dict) -> bool:
    """Whether a series' spectra are laid out a time point per row, a replicate per column: a
    labelled time point (not a control) has several replicates, and no two spectra share a
    time and a replicate number. Replicated controls alone do not count: their series fills
    the grid (owner, 2026-11-04: a course of single spectra with replicated controls)."""
    if _clashing(info):
        return False
    roles = info.get("roles", [""] * len(info["rows"]))  # none given: no controls
    labelled = np.array([_control_role(r) is None for r in roles], dtype=bool)
    if not labelled.any():
        return len(np.unique(info["reps"])) > 1  # controls only: by replicate as before
    _, counts = np.unique(np.asarray(info["time_index"])[labelled], return_counts=True)
    return bool((counts > 1).any())


def _layout(info: dict, columns: int | None, max_columns: int = PAGE_COLUMNS,
            fixed: tuple | None = None) -> tuple[int, int, list[tuple[int, int]], bool]:
    """Rows, columns and the (row, column) of every spectrum of a series (in ``info`` order).

    With several replicates (and ``columns`` not given) a row is a time and a column a replicate,
    unless two spectra share a time and a replicate number (warned); a time point with more
    than ``max_columns`` replicates goes on over the next row(s). Otherwise the spectra, in time
    order, fill a grid of ``columns`` columns (default: a near square of at most
    ``max_columns``), so that a long single-replicate time course is not one tall column. The
    last value says whether the layout is time by replicate. ``fixed`` (a page of
    :func:`spectra_pages`): the series' layout, the same on every page:
    ``('replicates', reps)`` (every replicate number of the series) or ``('grid', columns)``."""
    n = len(info["rows"])
    # a point is a time or a control role; ``time_index`` ranks them (the controls at the ends)
    points, reps = np.unique(info["time_index"]), np.unique(info["reps"])
    if fixed is not None and fixed[0] == "grid":
        columns = int(fixed[1])
    replicates = fixed is not None and fixed[0] == "replicates"
    if replicates:
        reps = np.asarray(fixed[1])
    elif fixed is None and columns is None and len(reps) > 1 and _clashing(info):
        log.warning("%s: two spectra share a time and a replicate number; drawn in time "
                    "order, not by replicate", info.get("label", "a series"))
    if replicates or (fixed is None and columns is None and _by_replicate(info)):
        ncols = min(len(reps), max(1, max_columns))
        lines = math.ceil(len(reps) / ncols)  # rows a time point takes
        pos = []
        for t, r in zip(info["time_index"], info["reps"], strict=True):
            at = int(np.searchsorted(reps, r))
            pos.append((int(np.searchsorted(points, t)) * lines + at // ncols, at % ncols))
        return len(points) * lines, ncols, pos, True
    if columns is not None and columns < 1:
        raise ValueError("columns must be at least 1")
    ncols = columns or max(1, min(max_columns, math.ceil(math.sqrt(n))))
    if fixed is None:
        ncols = min(ncols, n)  # a page of a series keeps the series' width
    return math.ceil(n / ncols), ncols, [(i // ncols, i % ncols) for i in range(n)], False


@dataclasses.dataclass(frozen=True)
class SpectraPage:
    """One page of a series' spectra (:func:`spectra_pages`): ``rows``, the spectra drawn
    (positions in the results), ``number`` and ``of`` (its place among the pages), and the time
    points ``shown`` of the series' ``points`` (fewer when ``max_points`` thinned them)."""

    rows: tuple[int, ...]
    number: int
    of: int
    shown: int
    points: int
    layout: tuple = ()  # the series' layout, for :func:`_layout`'s ``fixed``

    @property
    def note(self) -> str:
        """What the page header says of it ('' for a single page of every time point)."""
        parts = []
        if self.of > 1:
            parts.append(f"page {self.number} of {self.of}")
        if self.shown < self.points:
            parts.append(f"{self.shown} of {self.points} time points shown, every one fitted "
                         "([figures] max_points)")
        return "; ".join(parts)


def spectra_pages(results: Results, data_id: int, *, page_rows: int = PAGE_ROWS,
                  page_columns: int = PAGE_COLUMNS,
                  max_points: int | None = MAX_POINTS) -> list[SpectraPage]:
    """The pages a series' spectra are drawn on (owner, 2026-11-04: a kinetics run made one page
    of 281 panels, 214 inches tall). First, more than ``max_points`` time points (not the
    controls, always kept) are thinned to that many, spread over log time with the first and
    last kept, as ``[input.scans] max_spectra`` thins a read (None: every one). Then the points
    fill pages of ``page_rows`` rows of ``page_columns``: with replicates, a time point per row
    (more replicates than columns go on over the next row; a point is never split between
    pages); without, ``page_rows`` x ``page_columns`` time points a page."""
    from ..io.mzml import _sample_log

    if page_rows < 1 or page_columns < 1:
        raise ValueError("page_rows and page_columns must be at least 1")
    info = series_info(results, data_id)
    order = np.argsort(info["time_index"], kind="stable")
    rows, index = np.asarray(info["rows"])[order], np.asarray(info["time_index"])[order]
    roles = np.asarray(info["roles"]).astype(str)[order]
    times = np.asarray(info["times"], dtype=float)[order]
    points = list(dict.fromkeys(index.tolist()))  # in order, controls at the ends
    control = {p for p in points if _control_role(roles[index == p][0]) is not None}
    timed = [p for p in points if p not in control]
    keep = set(points)
    if max_points is not None and len(timed) > max_points:
        at = np.array([times[index == p][0] for p in timed])
        keep = control | {timed[i] for i in _sample_log(at, max_points)}
    chosen = [p for p in points if p in keep]
    by_point = {p: [int(r) for r in rows[index == p]] for p in chosen}
    pages: list[list[int]] = []
    if _by_replicate(info):  # a time point per row: whole points to a page
        reps = tuple(int(r) for r in np.unique(info["reps"]))
        layout = ("replicates", reps)
        lines = math.ceil(len(reps) / page_columns)
        per_page = max(1, page_rows // lines)
        for start in range(0, len(chosen), per_page):
            pages.append([r for p in chosen[start:start + per_page] for r in by_point[p]])
    else:  # the grid filled, in time order: the same width on every page
        drawn = [r for p in chosen for r in by_point[p]]
        columns = max(1, min(page_columns, math.ceil(math.sqrt(len(drawn)))))
        layout = ("grid", columns)
        per_page = page_rows * columns
        pages = [drawn[start:start + per_page] for start in range(0, len(drawn), per_page)]
    pages = pages or [[]]
    return [SpectraPage(tuple(page), n + 1, len(pages), len(chosen), len(points), layout)
            for n, page in enumerate(pages)]


def _grid(info: dict, size: tuple[float, float], columns: int | None, residual: bool = False,
          max_columns: int = PAGE_COLUMNS, fixed: tuple | None = None):
    """The figure and, for each spectrum, its axes (and its residual axes below it, or None)."""
    nrows, ncols, positions, _ = _layout(info, columns, max_columns, fixed)
    height = size[1] * (1.3 if residual else 1.0)
    fig = Figure(figsize=(size[0] * ncols, height * nrows))
    grid = fig.add_gridspec(nrows, ncols)
    cells = {}
    for i, (r, c) in enumerate(positions):
        if residual:
            sub = grid[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
            ax = fig.add_subplot(sub[0])
            cells[int(info["rows"][i])] = (ax, fig.add_subplot(sub[1], sharex=ax))
        else:
            cells[int(info["rows"][i])] = (fig.add_subplot(grid[r, c]), None)
    return fig, cells


def _cell_room(fig: Figure, cells: dict) -> float:
    """The width (inches) a panel's title may take on a page of ``cells``: the page's width
    over its columns, less the intensity axis's labels."""
    columns = max(1, len({round(ax.get_position().x0, 6) for ax, _ in cells.values()}))
    return max(1.0, fig.get_figwidth() / columns - 0.8)


def _series_range(results: Results, spectrum_ids: Sequence[str], axes) -> tuple[float, float]:
    """The expected m/z range of a series' spectra: the lowest and highest predicted m/z of
    their picked bins, or without a picked table the union of the ``axes``' current ranges."""
    lo, hi = np.inf, -np.inf
    binned = results.binned
    if binned is not None:
        for sid in spectrum_ids:
            try:
                mz = np.asarray(binned.spectrum(binned.find(sid))[0], dtype=float)
            except KeyError:
                continue
            mz = mz[np.isfinite(mz)]
            if len(mz):
                lo, hi = min(lo, float(mz.min())), max(hi, float(mz.max()))
    if not np.isfinite(lo):
        for ax in axes:
            if ax.has_data():
                a, b = ax.get_xlim()
                lo, hi = min(lo, a), max(hi, b)
    return lo, hi


def series_mz_range(results: Results, data_id: int,
                    padding: float = 0.1) -> tuple[float, float] | None:
    """The m/z range a series' spectra pages draw (:func:`plot_series_spectra`): the series'
    expected range with ``padding`` times its width on each side; None without a picked table.
    Given to a single spectrum's figures, they keep the series' axis (owner, 2026-11-04)."""
    ids = [str(results.fit.spectrum_id[i]) for i in series_rows(results, data_id)]
    lo, hi = _series_range(results, ids, [])
    if not (np.isfinite(lo) and hi > lo):
        return None
    pad = padding * (hi - lo)
    return lo - pad, hi + pad


def plot_series_spectra(
    results: Results,
    data_id: int,
    *,
    show: Sequence[str] = LAYERS,
    components: Sequence[int] | None = None,
    k: int | None = None,
    columns: int | None = None,
    residual: bool = False,
    cell: tuple[float, float] = (4.5, 3.0),
    padding: float = 0.1,
    rows: Sequence[int] | None = None,
    max_columns: int = PAGE_COLUMNS,
    note: str = "",
    layout: tuple | None = None,
    **kwargs,
) -> Figure:
    """A grid of spectra: with replicates a row per time and a column per replicate, otherwise
    the time course wrapped into ``columns`` columns (see :func:`_layout`; see
    :func:`plot_spectrum` for the keyword arguments, which are passed on). ``residual`` puts the
    residuals of each spectrum (picked minus fit) in a small panel under it; it needs the picked
    and fit layers. Spectra whose layers cannot be drawn are left blank. Every panel has the
    same m/z range: the series' expected range (the predicted m/z of the picked bins of all its
    spectra; without a picked table, the panels' own ranges together) with ``padding`` times its
    width on each side. ``rows`` (the spectra's positions in the results) draws only those
    spectra of the series, one time point's replicates or one page of
    :func:`spectra_pages` say, on the m/z range of the whole series, so that they compare with
    the full page; ``max_columns``: the most columns (:func:`_layout`); ``note``: said after the
    series' name in the page header (which page, how many time points); ``layout``: a page's
    :attr:`SpectraPage.layout`, so every page of a series is laid out alike."""
    info = series_info(results, data_id)
    every = [str(results.fit.spectrum_id[i]) for i in info["rows"]]
    if rows is not None:
        keep = np.isin(info["rows"], np.asarray(list(rows), dtype=int))
        if not keep.any():
            raise ValueError(f"none of the rows {list(rows)} is a spectrum of series {data_id}")
        info = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v)
                for k, v in info.items()}
    fig, cells = _grid(info, cell, columns, residual, max_columns, layout or None)
    unread = None
    if "raw" in show and results.spectra is None and results.loader is not None:
        # the series' raw spectra in one read, not one read per panel
        ids = [str(results.fit.spectrum_id[i]) for i in cells]
        try:
            results = dataclasses.replace(results, spectra=results.loader(ids))
        except Exception as exc:  # noqa: BLE001 (the other layers can still be drawn)
            log.warning("the raw spectra of series %d could not be read: %s", data_id, exc)
            unread = f"raw spectra not read: {exc}"
            results = dataclasses.replace(results, loader=None)
    room = _cell_room(fig, cells)  # the width a panel's title may take
    for i, (ax, rax) in cells.items():
        sid = str(results.fit.spectrum_id[i])
        try:
            plot_spectrum(sid, results, show=show, k=k, components=components, ax=ax, rax=rax,
                          residual=residual, legend=False, **kwargs)
            if rax is not None:
                ax.tick_params(labelbottom=False)
                ax.set_xlabel("")
        except ValueError as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", fontsize=6, wrap=True,
                    transform=ax.transAxes)
        if unread is not None:
            ax.text(0.02, 0.98, unread, ha="left", va="top", fontsize=6, wrap=True,
                    color="darkred", transform=ax.transAxes, gid="raw_unread")
        pos = list(info["rows"]).index(i)
        when = _time_text(info["times"][pos], info["roles"][pos], results.time_scale)
        # the panel keeps what plot_spectrum says of the spectrum: its K, status and flags
        title = f"{when}, rep {info['reps'][pos]} ({spectrum_notes(results, i, k)})"
        ax.set_title(wrap_title(title, room, 8), fontsize=8)
    lo, hi = _series_range(results, every, [ax for ax, _ in cells.values()])
    if np.isfinite(lo) and hi > lo:
        pad = padding * (hi - lo)
        for ax, _ in cells.values():  # the residual panels share their spectrum's x axis
            ax.set_xlim(lo - pad, hi + pad)
    # one legend for the page, naming every layer drawn (the panels have none)
    handles = page_legend_handles(drawn_layers(ax for ax, _ in cells.values()),
                                  peak_measure(results))
    _page_header(fig, f"{info['label']} ({note})" if note else info["label"], handles)
    return fig


def _err(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


ROLE_SHORT = {UNDEUT: "UN", FULLDEUT: "FD"}


def _control_role(role: str) -> str | None:
    """``'FD'`` or ``'UN'`` for a spectrum with the fully deuterated or undeuterated role (the
    controls are known by their role, so they may be at any time, as in an experiment that
    starts deuterated and exchanges into H2O), else None."""
    return ROLE_SHORT.get(str(role))


def _time_text(t: float, role: str = "", scale: str = "seconds") -> str:
    """A point for a title: ``FD`` or ``UN`` for a control (with its time when it has one),
    else the time in a readable unit (:func:`hxex.io.format_time`); ``point 3`` when the times
    only order the points (``scale='order'``, :attr:`Results.time_scale`)."""
    when = format_time(t, scale)
    if scale == "order" and when:
        when = f"point {when}"
    short = _control_role(role)
    if short is not None:
        return f"{short} ({when})" if np.isfinite(t) else short
    return when


def _control_row(results: Results, data_id: int) -> int | None:
    """The series' row of the controls table; None without controls and for a series that is
    not corrected (fragment ions)."""
    ctrl = results.controls
    row = None if ctrl is None else ctrl.row_of(data_id)
    return None if row is None or ctrl.uncorrected[row] else row


def _uncorrected(results: Results, data_id: int) -> bool:
    """Whether the series has a controls row that leaves it uncorrected (fragment ions)."""
    ctrl = results.controls
    row = None if ctrl is None else ctrl.row_of(data_id)
    return row is not None and bool(ctrl.uncorrected[row])


def population_data(
    results: Results,
    spectrum_id: str,
    *,
    k: int | None = None,
    corrected: bool | None = None,
    components: Sequence[int] | None = None,
) -> dict:
    """Populations of one spectrum for the population map: ``mean`` and ``population`` of every
    population by rank (``mean_se``, ``population_se``), the ``draws`` of the bootstrap as
    ``(n, K, 2)`` (mean, population; None if not kept), ``centroid`` (population-weighted mean),
    ``corrected`` (whether the means are back-exchange corrected), ``n_amides`` and ``shown``.
    ``corrected`` None (the default) corrects when the results have controls for the spectrum's
    series; True raises KeyError when they have none for it, and draws a series the controls
    leave uncorrected (fragment ions) uncorrected, with ``uncorrected_series`` True so that the
    panel says so."""
    i = results.find(spectrum_id)
    k = int(results.effective_K[i] if k is None else k)
    results.fit.require_fitted(i, k)
    kf = results.fit.fits[k]
    order = np.argsort(kf.means[i])
    mean, frac = kf.means[i][order], kf.populations[i][order]
    boot = results.bootstrap_for(i, k)
    nan = np.full(k, np.nan)
    mean_se = boot.mean_se[i, :k][order] if boot is not None else nan
    population_se = boot.population_se[i, :k][order] if boot is not None else nan
    draws = None
    if boot is not None and boot.draws is not None:
        d = boot.draws
        ok = d.ok[i]
        a = d.A[i][ok][:, :k][:, order]
        dm = (d.n[i][ok][:, :k] * d.p[i][ok][:, :k])[:, order]
        df = a / np.maximum(a.sum(axis=1, keepdims=True), 1e-10)
        draws = np.stack([dm, df], axis=-1)
    data_id = int(results.fit.data_id[i])
    row = _control_row(results, data_id)
    if corrected and row is None and not _uncorrected(results, data_id):
        raise ValueError(f"no controls for series {data_id} (spectrum {spectrum_id}), so no "
                         "corrected values: leave corrected unset (or false)")
    use = row is not None and (corrected is None or bool(corrected))
    n_amides = None
    if use:
        n_amides = float(results.controls.n_amides[row])
        c = correct_means(results.controls, [results.fit.data_id[i]], mean[None, :],
                          se=mean_se[None, :])
        mean, mean_se = c.mean[0], c.se[0]
        if draws is not None:
            draws = np.stack([
                correct_means(results.controls, [results.fit.data_id[i]] * len(draws),
                              draws[:, :, 0]).mean, draws[:, :, 1]], axis=-1)
    else:
        use = False
    shown = list(range(1, k + 1)) if components is None else sorted(int(c) for c in components)
    return {
        "spectrum_id": spectrum_id, "index": i, "k": k, "mean": mean, "population": frac,
        "mean_se": mean_se, "population_se": population_se, "draws": draws,
        "centroid": float((mean * frac).sum()), "corrected": use, "n_amides": n_amides,
        "shown": shown, "uncorrected_series": bool(corrected) and row is None,
    }


def _top(d: dict) -> float:
    """The largest deuterium value a population map of ``d`` draws: a component's mean plus
    its standard error, or a bootstrap draw (-inf when none is finite)."""
    values = [d["mean"][r - 1] + (d["mean_se"][r - 1] if np.isfinite(d["mean_se"][r - 1]) else 0)
              for r in d["shown"]]
    if d["draws"] is not None and d["draws"].size:
        values.append(np.nanmax(d["draws"][:, [r - 1 for r in d["shown"]], 0], initial=-np.inf))
    values = [v for v in values if np.isfinite(v)]
    return max(values) if values else -np.inf


def _limits(ax: Axes, n_amides: float | None, top: float = -np.inf) -> None:
    """The dashed guides at 0 and at the number of amides (once per axes) and the x range: to
    the amide count + 1, wider when a component (``top``: its mean plus its error) lies beyond
    it, so that none is clipped (an axes drawn again keeps its widest range)."""
    if n_amides is None:
        return
    right = max(n_amides + 1, top + 0.3)
    if any(line.get_gid() == "limits" for line in ax.collections):
        right = max(right, ax.get_xlim()[1])
    else:
        ax.vlines([0, n_amides], 0, 1, color="grey", ls=":", lw=1, gid="limits")
    ax.set_xlim(-1, right)


def _deuterium_label(corrected: bool, uncorrected_series: bool) -> str:
    """The deuterium axis label; a series drawn uncorrected although corrected values were
    asked for (one the controls leave uncorrected, such as fragment ions) says so."""
    if corrected:
        return "deuterium (back-exchange corrected)"
    return "deuterium (uncorrected series)" if uncorrected_series else "deuterium"


def _draw_population_map(ax: Axes, d: dict, title: str | None) -> None:
    for rank in d["shown"]:
        c = rank - 1
        style = population_style(rank)
        color = style["color"]
        if d["draws"] is not None:
            ax.scatter(d["draws"][:, c, 0], d["draws"][:, c, 1], s=8, color=color, alpha=0.3,
                       lw=0, gid=f"draws:{rank}")
        ax.errorbar(d["mean"][c], d["population"][c], xerr=_err(d["mean_se"][c]),
                    yerr=_err(d["population_se"][c]),
                    color=color, marker="o", ms=5, lw=1.5, fillstyle=style["fillstyle"],
                    label=f"pop {rank}", gid=f"pop:{rank}")
    ax.scatter([d["centroid"]], [1.0], marker="*", s=90, color="darkorange", zorder=1,
               label="centroid", gid="centroid")
    _limits(ax, d["n_amides"], _top(d))
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel(_deuterium_label(d["corrected"], d["uncorrected_series"]))
    ax.set_ylabel("population")
    if title:
        ax.set_title(title, fontsize=8)


def plot_populations(
    results: Results,
    data_id: int,
    *,
    k: int | None = None,
    corrected: bool | None = None,
    components: Sequence[int] | None = None,
    overlay: bool = True,
    reference: Reference | None = None,
    columns: int | None = None,
    cell: tuple[float, float] = (4.0, 3.0),
    rows: Sequence[int] | None = None,
    max_columns: int = PAGE_COLUMNS,
    note: str = "",
    layout: tuple | None = None,
) -> Figure:
    """The population map of a series, laid out like :func:`plot_series_spectra`: each cell the
    population against the deuterium level of every component (bootstrap draws when kept, mean
    with standard errors, the centroid as a star), and, in the time by replicate layout with
    ``overlay``, a last column that superimposes the replicates. ``reference`` marks the known
    populations (open circles). ``rows``, ``max_columns``, ``note`` and ``layout``: a page of
    :func:`spectra_pages`, as :func:`plot_series_spectra` takes them (the populations page goes
    on over the same pages as the spectra; owner, 2026-11-04)."""
    info = series_info(results, data_id)
    if rows is not None:
        keep = np.isin(info["rows"], np.asarray(list(rows), dtype=int))
        if not keep.any():
            raise ValueError(f"none of the rows {list(rows)} is a spectrum of series {data_id}")
        info = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v)
                for k, v in info.items()}
    nrows, ncols_cells, positions, by_rep = _layout(info, columns, max_columns, layout or None)
    reps = (np.asarray(layout[1]) if layout and layout[0] == "replicates"  # the series' own
            else np.unique(info["reps"]))
    ncols = ncols_cells + (1 if overlay and by_rep else 0)
    fig = Figure(figsize=(cell[0] * ncols, cell[1] * nrows))
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for pos, i in enumerate(info["rows"]):
        sid = str(results.fit.spectrum_id[i])
        r_idx = int(np.searchsorted(reps, info["reps"][pos]))
        t_idx = positions[pos][0]
        if k is not None and not results.fit.fitted(k)[i]:
            continue  # the screen skipped this K for the spectrum: its panel stays empty
        d = population_data(results, sid, k=k, corrected=corrected, components=components)
        ax = axes[positions[pos]]
        ax.set_visible(True)
        rejected = results.review_status[i] == "rejected"
        when = _time_text(info["times"][pos], info["roles"][pos], results.time_scale)
        label = (f"{when}, rep {info['reps'][pos]} "  # the map's axis is D: not repeated
                 f"({spectrum_notes(results, i, k, with_d=False)})")
        _draw_population_map(ax, d, "\n".join(textwrap.wrap(label, 58)))
        if reference is not None and d["n_amides"] is not None:
            known = reference.at(float(info["times"][pos]), d["n_amides"])
            if known is not None:
                ax.scatter(known[0], known[1], s=70, facecolors="none", edgecolors="darkred",
                           lw=1.5, zorder=6, gid="reference")
        if ncols > ncols_cells:
            ov = axes[t_idx, -1]
            ov.set_visible(True)
            for rank in d["shown"]:
                c = rank - 1
                # the colour is the population's (as in the single cells), the shape the replicate;
                # a rejected or excluded spectrum hollow and grey
                style = (REJECTED_STYLE if rejected else
                         {"color": population_style(rank)["color"],
                          "fillstyle": population_style(rank)["fillstyle"]})
                ov.errorbar(d["mean"][c], d["population"][c], xerr=_err(d["mean_se"][c]),
                            yerr=_err(d["population_se"][c]),
                            marker=REP_MARKERS[r_idx % len(REP_MARKERS)], ms=5,
                            lw=1, gid=f"overlay:{info['reps'][pos]}"
                            + (":rejected" if rejected else ""), **style)
            ov.set_ylim(-0.05, 1.1)
            when = _time_text(info["times"][pos], info["roles"][pos], results.time_scale)
            ov.set_title(f"{when}, replicates", fontsize=8)
            _limits(ov, d["n_amides"], _top(d))
            if reference is not None and d["n_amides"] is not None:
                known = reference.at(float(info["times"][pos]), d["n_amides"])
                drawn = any(c.get_gid() == "reference" for c in ov.collections)
                if known is not None and not drawn:
                    ov.scatter(known[0], known[1], s=70, facecolors="none", edgecolors="darkred",
                               lw=1.5, zorder=6, gid="reference")
    _page_header(fig, f"{info['label']} ({note})" if note else info["label"],
                 _population_legend(fig, reps if ncols > ncols_cells else ()))
    return fig


def _population_legend(fig: Figure, reps) -> list:
    """The populations page's legend: every symbol drawn on its panels (the populations by
    colour, the population-weighted mean, bootstrap draws, the replicates' shapes in the
    replicates column, rejected spectra and known populations)."""
    from matplotlib.lines import Line2D

    gids = {a.get_gid() or "" for ax in fig.axes if ax.get_visible() for a in ax.get_children()}
    out = []
    for rank in sorted(int(g.split(":")[1]) for g in gids if g.startswith("pop:")):
        style = population_style(rank)
        out.append(Line2D([], [], color=style["color"], marker="o", ms=5,
                          fillstyle=style["fillstyle"], label=f"population {rank} (± se)"))
    if "centroid" in gids:
        out.append(Line2D([], [], ls="none", marker="*", ms=9, color="darkorange",
                          label="population-weighted mean"))
    if any(g.startswith("draws:") for g in gids):
        out.append(Line2D([], [], ls="none", marker="o", ms=3, color="grey", alpha=0.4,
                          label="bootstrap draws"))
    for j, r in enumerate(reps):
        out.append(Line2D([], [], ls="none", marker=REP_MARKERS[j % len(REP_MARKERS)], ms=5,
                          color="black", fillstyle="none", label=f"rep {r} (replicates column)"))
    if any(g.endswith(":rejected") for g in gids):
        out.append(Line2D([], [], ls="none", marker="o", ms=5, label="rejected / excluded",
                          **REJECTED_STYLE))
    if "reference" in gids:
        out.append(Line2D([], [], ls="none", marker="o", ms=8, markerfacecolor="none",
                          markeredgecolor="darkred", label="known populations"))
    if "limits" in gids:
        # in the uptake page's words (UN and FD, as everywhere else)
        out.append(Line2D([], [], color="grey", ls=":", lw=1,
                          label="UN level (0)\nFD level (amide count)"))
    return out


def uptake_data(
    results: Results,
    data_id: int,
    *,
    corrected: bool | None = None,
    components: Sequence[int] | None = None,
    k: int | None = None,
) -> dict:
    """The populations of every spectrum of a series along the time course, one row per
    spectrum and population: ``index`` (position in the results), ``time``, ``time_index``,
    ``rep``, ``rank``, ``mean`` (corrected if asked and possible), ``mean_se``, ``population``,
    ``population_se``, plus ``corrected``, ``n_amides``, ``m0`` and ``mfd`` (None without
    controls) and ``role`` of every row (``'FD'``, ``'UN'`` or ``''``: whether the spectrum is a
    control, by its role; a control's ``time`` is NaN unless it was also a time point), and
    ``rejected`` (the review rejected the spectrum, or an [[review.exclude]] rule excluded
    it: drawn hollow and grey)."""
    info = series_info(results, data_id)
    cols: dict[str, list] = {n: [] for n in (
        "index", "time", "time_index", "rep", "rank", "mean", "mean_se", "population",
        "population_se", "role")}
    use, uncorrected_series = False, False
    for pos, i in enumerate(info["rows"]):
        if k is not None and not results.fit.fitted(k)[i]:
            continue  # the screen skipped this K for the spectrum
        d = population_data(results, str(results.fit.spectrum_id[i]), k=k, corrected=corrected,
                            components=components)
        use, uncorrected_series = d["corrected"], d["uncorrected_series"]
        for rank in d["shown"]:
            c = rank - 1
            for name, value in (
                ("index", i), ("time", info["times"][pos]), ("time_index", info["time_index"][pos]),
                ("role", _control_role(info["roles"][pos]) or ""),
                ("rep", info["reps"][pos]), ("rank", rank), ("mean", d["mean"][c]),
                ("mean_se", d["mean_se"][c]), ("population", d["population"][c]),
                ("population_se", d["population_se"][c]),
            ):
                cols[name].append(value)
    out: dict = {n: np.array(v) for n, v in cols.items()}
    out["role"] = np.asarray(out["role"], dtype=str)
    status = results.review_status
    ctrl = results.controls
    # rejected or excluded by the review, or a control spectrum the quality screens left out
    screened = set(ctrl.screened or ()) if ctrl is not None else set()
    ids = results.fit.spectrum_id
    out["rejected"] = np.array([status[int(i)] == "rejected" or str(ids[int(i)]) in screened
                                for i in out["index"]], dtype=bool)
    row = _control_row(results, data_id)
    un_source, fd_source = control_sources(ctrl, row) if row is not None else (None, None)
    out.update({
        "label": info["label"], "corrected": use, "uncorrected_series": uncorrected_series,
        "n_amides": info["n_amides"],
        "m0": None if row is None else float(ctrl.m0[row]),
        "mfd": None if row is None else float(ctrl.mfd[row]),
        "control_fallback": None if row is None else bool(ctrl.fallback[row]),
        "un_source": un_source, "fd_source": fd_source,
    })
    return out


# the fully deuterated stand-ins by their flag (hxex.controls), in the words of the legend
FD_STAND_INS = (("fd_other_charge", "other charge stand-in"),
                ("fd_other_sample", "other sample stand-in"),
                ("fd_map", "retention map stand-in"), ("fd_kint", "kint stand-in"),
                ("fd_median", "median stand-in"))


def control_sources(ctrl, row: int) -> tuple[str, str]:
    """Where a series' undeuterated and fully deuterated levels came from, in words:
    ``measured``, or the stand-in that replaced a missing or rejected control (``theoretical``
    for the undeuterated level; ``other charge stand-in``, ``other sample stand-in``,
    ``retention map stand-in``, ``kint stand-in``, ``median stand-in`` or ``D2O fraction
    stand-in`` for the fully deuterated one; :mod:`hxex.controls`)."""
    from ..controls import FD_REJECTED

    tokens = set(str(ctrl.flags[row]).split(";"))
    un = "theoretical (0)" if tokens & {"m0_missing", "m0_high"} else "measured"
    fd = next((text for flag, text in FD_STAND_INS if flag in tokens), None)
    if fd is None:
        fd = "D2O fraction stand-in" if tokens & set(FD_REJECTED) else "measured"
    return un, fd


def plot_uptake(
    results: Results,
    data_id: int,
    *,
    x: str | None = None,
    scale: str = "linear",
    corrected: bool | None = None,
    components: Sequence[int] | None = None,
    k: int | None = None,
    reference: Reference | None = None,
    ax: Axes | None = None,
    legend: bool = True,
) -> Figure:
    """Deuterium uptake of every population along the time course (the markers have an area that
    grows with the population, error bars are the bootstrap standard errors of the
    mean, and each replicate has its own colour).

    ``x='time'`` plots against the measured time (``scale='linear'`` or ``'log'``; a log axis
    leaves out times of 0 or less); every control spectrum of the series (known by its role)
    that is not a time point as well is drawn at an end of the axis, the undeuterated ones
    before the first time and the fully deuterated ones after the last, labelled ``UN`` and
    ``FD`` on the axis (no time tick beyond the times drawn), in their replicate's colour
    (hollow and grey when rejected, excluded, or left out by the quality screens).
    ``x='index'`` plots against the position of the point in the series (0, 1, 2, ...,
    labelled with the times), with every point including
    the controls (the undeuterated first, the fully deuterated last), as for the test data
    sets. ``x=None`` (the default) is ``'time'``, or ``'index'`` when the times only order the
    points (:attr:`Results.time_scale` ``'order'``: the labels are then the numbers given, and
    ``x='time'`` is refused). ``corrected`` (default: if the results have controls) uses the
    back-exchange corrected means, and the limits drawn are then 0 and the number of amides;
    otherwise the measured control levels. ``reference`` marks the known populations (open
    circles; in the measured units, ``m0 + level * (mfd - m0)``, when the plot is not
    corrected). The dotted lines are the levels used for the correction; the legend, outside
    the axes on the right, names each replicate's colour and marker (:func:`rep_style`; the
    marker's area grows with the population) and says when a level came from a stand-in
    (:func:`control_sources`).
    """
    if x not in ("time", "index", None):
        raise ValueError("x must be 'time' or 'index'")
    if scale not in ("linear", "log"):
        raise ValueError("scale must be 'linear' or 'log'")
    time_scale = results.time_scale
    if x is None:
        x = "index" if time_scale == "order" else "time"
    elif x == "time" and time_scale == "order":
        raise ValueError("the times of these results only order the points ([input] "
                         "time_scale = \"order\"): the uptake is drawn by index (x='index')")
    d = uptake_data(results, data_id, corrected=corrected, components=components, k=k)
    # a figure of its own is laid out by constrained layout, which leaves room for the legend
    # outside the axes
    own = ax is None
    fig = ax.figure if ax is not None else Figure(figsize=(8.0, 4.5), layout="constrained")
    ax = ax if ax is not None else fig.subplots()
    keep = np.ones(len(d["time"]), dtype=bool) if x == "index" else np.isfinite(d["time"])
    if x == "time" and scale == "log":
        keep &= d["time"] > 0
    xs = (d["time_index"] if x == "index" else d["time"]).astype(float)
    # x = "time": the series' own controls that are not time points as well, at the axis ends
    # (undeuterated left, fully deuterated right), so that the levels behind the correction show
    ends = (~keep & (d["role"] != "")) if x == "time" else np.zeros(len(xs), dtype=bool)
    control_at: dict[str, float] = {}  # where the controls are drawn, labelled on the axis
    if ends.any():
        xs = xs.copy()
        left, right = _end_positions(xs[keep], scale)
        for role, at in (("UN", left), ("FD", right)):
            if (ends & (d["role"] == role)).any():
                xs[ends & (d["role"] == role)] = at
                control_at[role] = at
    drawn = keep | ends
    reps = np.unique(d["rep"][drawn]) if drawn.any() else np.unique(d["rep"])
    rejected = d["rejected"]
    for r in reps:
        sel = drawn & (d["rep"] == r) & ~rejected
        color, marker = rep_style(int(r))
        ax.errorbar(xs[sel], d["mean"][sel],
                    yerr=np.where(np.isfinite(d["mean_se"][sel]), d["mean_se"][sel], 0.0),
                    fmt="none", ecolor=color, elinewidth=1.2, capsize=0, gid=f"errors:rep{r}")
        ax.scatter(xs[sel], d["mean"][sel], s=20 + 160 * np.nan_to_num(d["population"][sel]),
                   color=color, marker=marker, alpha=0.9, label=f"rep {r}", gid=f"rep:{r}")
    out = drawn & rejected  # rejected, excluded or screened: visible, but hollow and grey
    if out.any():
        ax.errorbar(xs[out], d["mean"][out],
                    yerr=np.where(np.isfinite(d["mean_se"][out]), d["mean_se"][out], 0.0),
                    fmt="none", ecolor="grey", elinewidth=1.0, alpha=0.6, capsize=0,
                    gid="errors:rejected")
        ax.scatter(xs[out], d["mean"][out], s=20 + 160 * np.nan_to_num(d["population"][out]),
                   facecolors="none", edgecolors="grey", marker="v", lw=1.2,
                   label="rejected / excluded", gid="rejected")
    if reference is not None and d["n_amides"] is not None:
        for t in np.unique(d["time"][keep]):
            known = reference.at(float(t), d["n_amides"])
            if known is not None:
                pos = d["time_index"][d["time"] == t][0] if x == "index" else t
                ys = known[0]  # corrected units; else back to the measured scale
                if not d["corrected"]:
                    ys = d["m0"] + ys / d["n_amides"] * (d["mfd"] - d["m0"])
                ax.scatter([pos] * len(ys), ys, s=20 + 160 * known[1],
                           facecolors="none", edgecolors="darkred", lw=1.5, zorder=6,
                           gid="reference")
    lo, hi = (0.0, d["n_amides"]) if d["corrected"] else (d["m0"], d["mfd"])
    for level in (lo, hi):
        if level is not None:
            ax.axhline(level, color="grey", ls=":", lw=1, alpha=0.7, gid="limits")
    if x == "index":
        ticks = np.unique(d["time_index"])
        labels = []
        for t in ticks:
            time = d["time"][d["time_index"] == t][0]
            role = d["role"][d["time_index"] == t][0]
            labels.append(role or format_time(time, time_scale))
        ax.set_xticks(ticks, labels, rotation=45 if len(ticks) > 8 else 0, fontsize=8)
        ax.set_xlabel("time point")
    else:
        ax.set_xlabel("time (s)")
        ax.set_xscale(scale)
        if control_at:
            _label_controls(ax, control_at, xs[keep], scale)
            d["control_at"] = control_at
    ax.set_ylabel(_deuterium_label(d["corrected"], d["uncorrected_series"]))
    ax.set_title(d["label"], fontsize=9)
    if legend:  # outside the axes, on the right
        kw = {"handles": _uptake_legend(ax, d, lo, hi, reference is not None), "fontsize": 7,
              "frameon": False, "title": "marker area: population", "title_fontsize": 7}
        if own:  # constrained layout makes room for a figure legend placed "outside"
            fig.legend(loc="outside right upper", **kw)
        else:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), **kw)
    return fig


def _label_controls(ax: Axes, control_at: dict[str, float], times: np.ndarray,
                    scale: str) -> None:
    """On a time axis, the controls' positions labelled ``UN`` and ``FD`` on the axis itself (as
    the index axis labels them), and no time tick beyond the times drawn: a number out there
    would read as the controls' time (owner, 2026-11-03)."""
    from matplotlib.ticker import Formatter, Locator

    times = times[np.isfinite(times) & ((times > 0) if scale == "log" else True)]
    if not len(times):
        lo, hi = -np.inf, np.inf
    else:
        lo, hi = float(times.min()), float(times.max())

    def middle(a, b):  # halfway between a data time and a control, on the axis' scale
        return float(np.sqrt(a * b)) if scale == "log" else (a + b) / 2

    lo = middle(lo, control_at["UN"]) if "UN" in control_at and np.isfinite(lo) else -np.inf
    hi = middle(hi, control_at["FD"]) if "FD" in control_at and np.isfinite(hi) else np.inf
    names = {round(v, 9): k for k, v in control_at.items()}

    class Ticks(Locator):
        def __init__(self, base, add):
            self.base, self.add = base, add

        def __call__(self):
            kept = [v for v in self.base() if lo <= v <= hi]
            return sorted(kept + (list(control_at.values()) if self.add else []))

    class Labels(Formatter):
        def __init__(self, base):
            self.base = base

        def set_locs(self, locs):
            self.base.set_locs([v for v in locs if round(v, 9) not in names])

        def __call__(self, value, pos=None):
            return names.get(round(value, 9)) or self.base(value, pos)

        def get_offset(self):
            return self.base.get_offset()

    xaxis = ax.xaxis
    xaxis.set_major_locator(Ticks(xaxis.get_major_locator(), True))
    xaxis.set_major_formatter(Labels(xaxis.get_major_formatter()))
    xaxis.set_minor_locator(Ticks(xaxis.get_minor_locator(), False))


def _end_positions(times: np.ndarray, scale: str) -> tuple[float, float]:
    """Where the controls go on a time axis: a little before the first and after the last time
    drawn (by a share of the range, in decades on a log axis)."""
    times = times[np.isfinite(times) & ((times > 0) if scale == "log" else True)]
    if scale == "log":
        if not len(times):
            return 1.0, 10.0
        a, b = np.log10(times.min()), np.log10(times.max())
        gap = max(0.12 * (b - a), 0.3)
        return float(10 ** (a - gap)), float(10 ** (b + gap))
    if not len(times):
        return -1.0, 1.0
    a, b = float(times.min()), float(times.max())
    gap = 0.1 * (b - a) if b > a else max(abs(a), 1.0) * 0.5
    return a - gap, b + gap


def _uptake_legend(ax: Axes, d: dict, lo, hi, reference: bool) -> list:
    """The uptake page's legend: each replicate's colour and marker, rejected points, the
    controls at the axis ends, known populations and the dotted levels used for the
    correction, saying where each level came from (a stand-in by name)."""
    from matplotlib.lines import Line2D

    handles, _ = ax.get_legend_handles_labels()
    handles = [h for h in handles if h.get_gid() and h.get_gid().startswith(("rep:", "rejected"))]
    out = []
    for h in handles:
        gid = h.get_gid()
        if gid == "rejected":
            out.append(Line2D([], [], ls="none", marker="v", ms=7, markerfacecolor="none",
                              markeredgecolor="grey", label="rejected / excluded"))
        else:
            color, marker = rep_style(int(gid.split(":")[1]))
            out.append(Line2D([], [], ls="none", marker=marker, ms=7, color=color,
                              label=f"rep {gid.split(':')[1]}"))
    if d.get("control_at"):
        out.append(Line2D([], [], ls="none", label="UN, FD: control spectra"))
    if reference:
        out.append(Line2D([], [], ls="none", marker="o", ms=8, markerfacecolor="none",
                          markeredgecolor="darkred", label="known populations"))
    if lo is not None or hi is not None:
        if d["corrected"]:
            un, fd = "UN level (0)", "FD level (amide count)"
        else:
            un, fd = "UN level", "FD level"
        if d.get("un_source") not in (None, "measured"):
            un += f": {d['un_source']}"
        if d.get("fd_source") not in (None, "measured"):
            fd += f": {d['fd_source']}"
        out.append(Line2D([], [], color="grey", ls=":", lw=1, label=f"{un}\n{fd}"))
    return out


__all__ = [
    "Reference", "control_sources", "load_reference", "plot_populations", "plot_series_spectra",
    "plot_uptake", "population_data", "rep_style", "series_info", "series_rows", "uptake_data",
]
