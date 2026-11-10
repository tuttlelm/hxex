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
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from .order import natural_key

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
    # how the series' scans were chosen (spectra read from mzML or Thermo files; empty or 0
    # otherwise): the fragment ('c-1:4'; '' for a peptide), its ion type, the fragment list's
    # label (or the scan conditions), the MS level and the parent ('1-12 z3')
    fragment: str = ""
    ion_type: str = ""
    condition: str = ""
    ms_level: int = 0
    parent: str = ""

    @property
    def name(self) -> str:
        """The series as the tree names it under its peptide: the charge, and for a fragment
        its ion, condition and MS level (``z=1  c-1:4  ETD, MS2``)."""
        if not self.fragment:
            return f"z={self.charge}"
        return f"z={self.charge}  {self.fragment}  " + ", ".join(
            p for p in (self.condition, f"MS{self.ms_level}" if self.ms_level else "") if p)


@dataclass
class Peptide:
    sample: str
    peptide_range: str
    peptide: str
    series: list[Series] = field(default_factory=list)
    # what a pooled peptide may be ([input] pooled_fasta; empty otherwise), as the tables give
    # it (:func:`hxex.report.tables.series_identities`): the record, the class or the classes
    # (";"); its level (isoform, class, ambiguous or none); the classes and the candidate
    # records, ";"-joined
    identity: str = ""
    identity_level: str = ""
    classes: str = ""
    candidates: str = ""

    def counts(self) -> dict[str, int]:
        return {name: sum(getattr(s, name) for s in self.series)
                for name in ("n_spectra", "n_flagged", "n_rejected", "n_multi", "n_left_out")}


def peptides(results, left: list[str] | None = None) -> list[Peptide]:
    """Every peptide of every sample with its series, in the order of the samples and then of
    the residues and charges; ``left``: :func:`left_out` of the results (none left out when
    not given)."""
    from ..report.tables import series_identities, series_table

    rows = series_table(results)
    known = series_identities(results) or {}
    reviewed = _reviewed(results)
    n_left = left_out_counts(results, left)
    targets = series_targets(results)
    out: dict[tuple, Peptide] = {}
    for r in range(len(rows["data_id"])):
        key = (str(rows["sample"][r]), str(rows["peptide_range"][r]), str(rows["peptide"][r]))
        pep = out.setdefault(key, Peptide(*key))
        data_id = int(rows["data_id"][r])
        found = known.get(data_id)
        if found is not None and not pep.identity_level:
            pep.identity, pep.identity_level = found.identity, found.level
            pep.classes, pep.candidates = ";".join(found.classes), ";".join(found.candidates)
        pep.series.append(Series(
            data_id, int(rows["charge"][r] or 0), int(rows["n_spectra"][r]),
            int(rows["n_flagged"][r]), int(rows["n_rejected"][r]), int(rows["n_multi"][r]),
            reviewed.get(data_id, 0), n_left.get(data_id, 0), **targets.get(data_id, {})))
    for pep in out.values():
        pep.series.sort(key=lambda s: (s.charge, s.ms_level, natural_key(s.condition),
                                       natural_key(s.fragment)))
    return [out[k] for k in sorted(out, key=lambda k: (natural_key(k[0]), natural_key(k[1]),
                                                       k[2]))]


def series_targets(results) -> dict[int, dict]:
    """Each series' (by ``data_id``) fragment, ion type, condition and MS level, as
    ``spectra.csv`` gives them (:func:`hxex.report.tables._target_columns`); empty for results
    not read from mzML or Thermo files."""
    from ..io.mzml import mzml_series_targets
    from ..report.tables import _target_columns

    cols = _target_columns(results)
    if not cols:
        return {}
    targets = mzml_series_targets(results.settings.spectra_source)
    out: dict[int, dict] = {}
    for i, d in enumerate(results.fit.data_id):
        if int(d) not in out:
            target = targets.get(int(d))
            z = dict(target.entry.extra).get("parent_charge", "") if target is not None else ""
            parent = str(cols["parent_range"][i])
            out[int(d)] = {"fragment": str(cols["fragment"][i]),
                           "ion_type": str(cols["ion_type"][i]),
                           "condition": str(cols["condition"][i]),
                           "ms_level": int(cols["ms_level"][i]),
                           "parent": f"{parent} z{z}" if parent and z != "" else parent}
    return out


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


def search_matches(text: str, sample: str, peptide_range: str, peptide: str,
                   extra: str = "") -> bool:
    """Whether a row matches the search ``text``: every word of it must, a number as the residue
    it names (the peptide covers it: ``20``), two numbers as the peptide's residues (``12-23``,
    however the range is padded), anything else as text in the sample, residues, peptide or
    ``extra`` (any case). Owner, 2026-11-04: ``42-54`` did not find ``0042-0054``."""
    words = f"{sample} {peptide_range} {peptide} {extra}".lower()
    span = _span(peptide_range)
    for word in text.lower().split():
        both = re.fullmatch(r"(\d+)-(\d+)", word)
        if both:
            if span != (int(both[1]), int(both[2])):
                return False
        elif word.isdigit():
            if span is None or not span[0] <= int(word) <= span[1]:
                return False
        elif word not in words:
            return False
    return True


def _span(peptide_range: str) -> tuple[int, int] | None:
    """``'0042-0054'`` as ``(42, 54)`` (None when it does not read so)."""
    both = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(peptide_range))
    return (int(both[1]), int(both[2])) if both else None


def shown(pep: Peptide, series: Series, how: str = "all", text: str = "",
          ms_level: int | None = None, ion_type: str | None = None,
          identity_level: str | None = None, pool_class: str | None = None) -> bool:
    """Whether a series is shown under the filter ``how`` (:data:`FILTERS`), the search
    ``text`` (:func:`search_matches`: residues by number, else the sample, residues, peptide,
    charge, fragment, condition, or a pooled peptide's identity and candidates as text), the
    MS level ``ms_level`` and the ion type ``ion_type`` (None: any; ``''``: peptides, not
    fragments), and for pooled peptides the identity level ``identity_level`` (isoform,
    class, ambiguous or none) and the class ``pool_class`` (one of the peptide's classes;
    None: any)."""
    if text and not search_matches(text, pep.sample, pep.peptide_range, pep.peptide,
                                   f"z{series.charge} {series.fragment} {series.condition} "
                                   f"{series.parent} {pep.identity} {pep.candidates}"):
        return False
    if ms_level is not None and series.ms_level != ms_level:
        return False
    if ion_type is not None and series.ion_type != ion_type:
        return False
    if identity_level is not None and pep.identity_level != identity_level:
        return False
    if pool_class is not None and pool_class not in pep.classes.split(";"):
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
