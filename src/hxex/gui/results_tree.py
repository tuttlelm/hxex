# SPDX-License-Identifier: GPL-3.0-or-later
"""The tree of the Results workspace, without Qt (docs/PLAN.md, "GUI", "Results and review"):
each peptide of each sample, its charges (the series), each series' time points (the
undeuterated controls first, the fully deuterated last, labelled as the figures label them),
and each point's replicates, which are the spectra. Every level carries the counts the window
shows, from the same tables ``hxex report`` writes (``series.csv``, ``spectra.csv``), with the
review applied, and the spectra the analysis settings leave out (:func:`left_out`, as
``hxex analyse`` lists them in ``left_out_spectra.csv``).
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

# the filters of the tree: (the list's words, what they mean)
FILTERS = {"all": ("all series", "every series"),
           "flagged": ("flagged", "series with a spectrum that has a quality flag"),
           "multi": ("K above 1", "series with a spectrum of more than one population"),
           "rejected": ("rejected", "series with a rejected spectrum"),
           "left_out": ("left out", "series with a spectrum the analysis settings leave out"),
           "unreviewed": ("not reviewed", "series with no spectrum accepted or rejected yet")}


@dataclass
class Spectrum:
    index: int  # the spectrum's position in the results
    spectrum_id: str
    rep: int
    K: int
    status: str  # review status: "", "accepted" or "rejected"
    flags: str  # quality flags, ";"-separated
    left_out: str = ""  # why the analysis settings leave it out ('' when they keep it)


@dataclass
class Point:
    label: str  # "UN", "30 s", "FD", "point 3", ...
    order: int  # its place in the series (controls at the ends)
    spectra: list[Spectrum] = field(default_factory=list)

    @property
    def ks(self) -> str:
        return " ".join(str(s.K) for s in self.spectra)


@dataclass
class Series:
    data_id: int
    charge: int
    n_spectra: int
    n_flagged: int
    n_rejected: int
    n_multi: int
    n_reviewed: int
    n_left_out: int = 0


@dataclass
class Peptide:
    sample: str
    peptide_range: str
    peptide: str
    series: list[Series] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {name: sum(getattr(s, name) for s in self.series)
                for name in ("n_spectra", "n_flagged", "n_rejected", "n_multi", "n_left_out")}


def peptides(results, left: list[str] | None = None) -> list[Peptide]:
    """Every peptide of every sample with its series, in the order of the samples and then of
    the residues and charges; ``left``: :func:`left_out` of the results (none left out when
    not given)."""
    from ..report.tables import series_table

    rows = series_table(results)
    reviewed = _reviewed(results)
    n_left = left_out_counts(results, left)
    out: dict[tuple, Peptide] = {}
    for r in range(len(rows["data_id"])):
        key = (str(rows["sample"][r]), str(rows["peptide_range"][r]), str(rows["peptide"][r]))
        pep = out.setdefault(key, Peptide(*key))
        data_id = int(rows["data_id"][r])
        pep.series.append(Series(
            data_id, int(rows["charge"][r] or 0), int(rows["n_spectra"][r]),
            int(rows["n_flagged"][r]), int(rows["n_rejected"][r]), int(rows["n_multi"][r]),
            reviewed.get(data_id, 0), n_left.get(data_id, 0)))
    for pep in out.values():
        pep.series.sort(key=lambda s: s.charge)
    return [out[k] for k in sorted(out, key=lambda k: (k[0], _start(k[1]), k[1], k[2]))]


def _start(peptide_range: str) -> int:
    try:
        return int(peptide_range.split("-")[0])
    except ValueError:
        return 0


def _reviewed(results) -> dict[int, int]:
    status = np.asarray(results.review_status).astype(str)
    ids = results.fit.data_id
    out: dict[int, int] = {}
    for d, s in zip(ids, status, strict=True):
        if s in ("accepted", "rejected"):
            out[int(d)] = out.get(int(d), 0) + 1
    return out


def left_out(results, config=None) -> list[str]:
    """Why the analysis settings ``config`` (:class:`hxex.analysis.AnalysisConfig`; its
    defaults when None) leave each spectrum of ``results`` out, '' for the ones kept: the
    reasons of ``left_out_spectra.csv`` (``[analysis.filters]``), ``exclude_flagged`` and
    ``exclude_times``, and the spectra kept at one population (``spectrum_quality_filtering``).
    Rejected spectra are not listed (the review left them out)."""
    with _quiet_here():  # the controls' messages, said when the results were read (owner)
        return _left_out(results, config)


@contextmanager
def _quiet_here():
    """No log lines from this thread while the block runs (other threads' still pass): the
    left-out column computes the controls again after every review decision, which repeated
    their messages each time."""
    me = threading.get_ident()

    def elsewhere(record: logging.LogRecord) -> bool:
        return record.thread != me

    handlers = {*logging.getLogger().handlers, *logging.getLogger("hxex").handlers}
    for handler in handlers:
        handler.addFilter(elsewhere)
    try:
        yield
    finally:
        for handler in handlers:
            handler.removeFilter(elsewhere)


def _left_out(results, config) -> list[str]:
    from ..analysis import AnalysisConfig, left_out_spectra
    from ..analysis.screening import screening

    config = config or AnalysisConfig()
    n = len(results.fit)
    why = np.full(n, "", dtype=object)
    status = np.asarray(results.review_status).astype(str)
    if config.filters is not None:
        screen = screening(results, config.filters.spectrum_quality_filtering,
                           config.filters.off_mz)
        one = np.asarray(screen.one_population, dtype=bool) & (status == "unreviewed")
        why[one] = "kept at K = 1 (spectrum quality)"
        table = left_out_spectra(results, config.filters, exclude_flagged=config.exclude_flagged,
                                 controls=config.controls)
        at = {str(sid): i for i, sid in enumerate(results.fit.spectrum_id)}
        for sid, reason in zip(table["spectrum_id"], table["reason"], strict=True):
            why[at[str(sid)]] = str(reason)
    if config.exclude_flagged and results.quality is not None:
        why[np.asarray(results.quality.flagged, dtype=bool)] = "quality flag (exclude_flagged)"
    time = np.asarray(results.fit.time, dtype=float)
    for t in config.exclude_times:
        why[np.isclose(time, t, rtol=1e-9, atol=1e-6)] = "time left out (exclude_times)"
    why[status == "rejected"] = ""
    return [str(w) for w in why]


def left_out_counts(results, left: list[str] | None) -> dict[int, int]:
    """How many spectra of each series (by ``data_id``) ``left`` leaves out (those kept at one
    population are not counted)."""
    out: dict[int, int] = {}
    if left is None:
        return out
    for d, why in zip(results.fit.data_id, left, strict=True):
        if why and not why.startswith("kept"):
            out[int(d)] = out.get(int(d), 0) + 1
    return out


def points(results, data_id: int, left: list[str] | None = None) -> list[Point]:
    """The time points of a series, in its order, each with its spectra by replicate
    (``left``: :func:`left_out` of the results)."""
    from ..report.series import _time_text, series_info

    info = series_info(results, data_id)
    status = np.asarray(results.review_status).astype(str)
    ks = results.effective_K
    flags = _flags(results)
    by_point: dict[int, Point] = {}
    for pos, i in enumerate(info["rows"]):
        order = int(info["time_index"][pos])
        point = by_point.get(order)
        if point is None:
            label = _time_text(float(info["times"][pos]), str(info["roles"][pos]),
                               results.time_scale)
            point = by_point[order] = Point(label, order)
        point.spectra.append(Spectrum(
            int(i), str(results.fit.spectrum_id[i]), int(info["reps"][pos]), int(ks[i]),
            str(status[i]) if status[i] not in ("", "None", "unreviewed") else "",
            flags[i], left[i] if left is not None else ""))
    for point in by_point.values():
        point.spectra.sort(key=lambda s: s.rep)
    return [by_point[k] for k in sorted(by_point)]


def _flags(results) -> list[str]:
    from ..quality import FLAG_NAMES

    q = results.quality
    m = len(results.fit)
    if q is None:
        return [""] * m
    return [";".join(n for n in FLAG_NAMES if getattr(q, n)[i]) for i in range(m)]


def shown(pep: Peptide, series: Series, how: str = "all", text: str = "") -> bool:
    """Whether a series is shown under the filter ``how`` (:data:`FILTERS`) and the search
    ``text`` (in the sample, residues or peptide, any case)."""
    if text:
        words = f"{pep.sample} {pep.peptide_range} {pep.peptide} z{series.charge}".lower()
        if text.strip().lower() not in words:
            return False
    if how == "flagged":
        return series.n_flagged > 0
    if how == "multi":
        return series.n_multi > 0
    if how == "rejected":
        return series.n_rejected > 0
    if how == "left_out":
        return series.n_left_out > 0
    if how == "unreviewed":
        return series.n_reviewed == 0
    if how != "all":
        raise ValueError(f"unknown filter {how!r}")
    return True
