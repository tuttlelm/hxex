# SPDX-License-Identifier: GPL-3.0-or-later
"""The tree of the Results workspace, without Qt (docs/PLAN.md, "GUI", "Results and review"):
each peptide of each sample, its charges (the series), each series' time points (the
undeuterated controls first, the fully deuterated last, labelled as the figures label them),
and each point's replicates, which are the spectra. Results with fragment ions are grouped by
precursor instead (owner, 2026-11-17): a top row is a parent at one precursor charge, with its
MS1 series at that charge, its precursor ions (M), and its fragments under it. Every level
carries the counts the window shows, from the same tables ``hxex report`` writes
(``series.csv``, ``spectra.csv``), with the review applied, and the spectra the analysis
settings leave out (:func:`left_out`, as ``hxex analyse`` lists them in
``left_out_spectra.csv``).
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
    # its set's number of populations by the population test of the set (every replicate and
    # charge state at this time, :func:`hxex.analysis.set_test.spectrum_sets`), 0 for none,
    # and the set's verdict in words ('2 populations (p 1.0e-85)')
    set_K: int = 0
    set_words: str = ""

    @property
    def differs(self) -> bool:
        """Whether its set has another number of populations than its K (owner, 2026-11-17:
        shown, the K unchanged)."""
        return bool(self.set_K) and self.set_K != self.K

    @property
    def k_text(self) -> str:
        """The K column: ``1``, or ``K = 1 (self), K = 2 (set)`` where the set's differs."""
        return k_text(self.K, self.set_K)

    @property
    def k_tip(self) -> str:
        """The K cell's tooltip where the set has another number (empty otherwise)."""
        return set_tip(self.K, self.set_words) if self.differs else ""


SET_MARK_TIP = ("population test (self), this spectrum alone: K = {k}; population test (set), "
                "every replicate and charge state of this peptide in this sample at this "
                "time: {words}, as hxex analyse decides it. The analysis uses the set's; the "
                "spectrum keeps its own fit and K")


def k_text(k: int, set_k: int) -> str:
    """A spectrum's K as the tree writes it: ``1``, or ``K = 1 (self), K = 2 (set)`` where its set
    (by the population test of the set) has another number of populations (owner,
    2026-11-17: the set's evidence can support a K the spectrum alone cannot)."""
    return f"K = {k} (self), K = {set_k} (set)" if set_k and set_k != k else str(k)


def set_tip(k: int, words: str) -> str:
    """Why a K is marked ``K = 1 (self), K = N (set)``: the spectrum's K and the set's verdict."""
    return SET_MARK_TIP.format(words=words, k=k)


@dataclass
class Point:
    label: str  # "UN", "30 s", "FD", "point 3", ...
    order: int  # its place in the series (controls at the ends)
    spectra: list[Spectrum] = field(default_factory=list)

    @property
    def ks(self) -> str:
        """Its replicates' K, and ``(set N)`` when one of them differs from their set."""
        return point_ks([s.K for s in self.spectra], self.set_K if self.differs else 0)

    @property
    def set_K(self) -> int:
        """The set's number of populations (one set holds a time point's spectra), 0 for none."""
        return next((s.set_K for s in self.spectra if s.set_K), 0)

    @property
    def differs(self) -> bool:
        return any(s.differs for s in self.spectra)


def point_ks(ks, set_k: int = 0) -> str:
    """A time point's K column: its replicates' K (``1 2 2 2``), then ``(set N)`` when one of
    them differs from their set's ``set_k`` (0: none differs)."""
    text = " ".join(str(k) for k in ks)
    return f"{text} (set {set_k})" if set_k else text


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
    # the parent of a fragment (or precursor ion): its residues, sequence and the precursor's
    # charge (empty and 0 for a peptide read at MS1)
    parent_range: str = ""
    parent_peptide: str = ""
    parent_charge: int = 0
    own_range: str = ""  # the series' own residues ('2-6' for c-1:5 of 2-12)
    by_precursor: bool = False  # named under a precursor row (results with fragments)

    @property
    def name(self) -> str:
        """The series as the tree names it under its peptide: the charge, and for a fragment
        its ion, condition and MS level (``z=1  c-1:4  ETD, MS2``). Under a precursor row the
        precursor's charge is the row's, so the fragment comes first (``c-1:4  z=1  ETD, MS2``),
        a precursor ion goes without its length (``M  z=1  ETD, MS2``), and the parent's own
        series is ``MS1``."""
        how = ", ".join(p for p in (self.condition,
                                    f"MS{self.ms_level}" if self.ms_level else "") if p)
        if not self.fragment:
            return "MS1" if self.by_precursor else f"z={self.charge}"
        if not self.by_precursor:
            return f"z={self.charge}  {self.fragment}  {how}"
        ion = self.ion_type if self.ion_type.startswith("M") else self.fragment
        return f"{ion}  z={self.charge}  {how}"


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
    charge: int = 0  # a precursor row's precursor charge (0: a peptide with its charges)

    @property
    def label(self) -> str:
        """The row as the tree and the titles name it: ``2-12 ALKRINKELSD`` (a precursor row
        adds its charge: ``2-12 ALKRINKELSD 3+``)."""
        return f"{self.peptide_range} {self.peptide}" + (f" {self.charge}+" if self.charge
                                                         else "")

    def counts(self) -> dict[str, int]:
        return {name: sum(getattr(s, name) for s in self.series)
                for name in ("n_spectra", "n_flagged", "n_rejected", "n_multi", "n_left_out")}


def peptides(results, left: list[str] | None = None) -> list[Peptide]:
    """Every peptide of every sample with its series, in the order of the samples and then of
    the residues and charges; ``left``: :func:`left_out` of the results (none left out when
    not given). Results with fragment ions are grouped by precursor (:func:`_precursor_key`):
    a row per parent and precursor charge, its MS1 series first, then its precursor ions, then
    its fragments by ion series, length, charge, MS level and condition (owner, 2026-11-17)."""
    from ..report.tables import series_identities, series_table

    rows = series_table(results)
    known = series_identities(results) or {}
    reviewed = _reviewed(results)
    n_left = left_out_counts(results, left)
    targets = series_targets(results)
    grouped = any(t.get("fragment") for t in targets.values())
    out: dict[tuple, Peptide] = {}
    for r in range(len(rows["data_id"])):
        data_id = int(rows["data_id"][r])
        target = targets.get(data_id, {})
        own = (str(rows["sample"][r]), str(rows["peptide_range"][r]), str(rows["peptide"][r]))
        charge = int(rows["charge"][r] or 0)
        if grouped:
            key = _precursor_key(own, charge, target)
            pep = out.setdefault(key, Peptide(*key[:3], charge=key[3]))
        else:
            pep = out.setdefault(own, Peptide(*own))
        found = known.get(data_id)
        if found is not None and not pep.identity_level:
            pep.identity, pep.identity_level = found.identity, found.level
            pep.classes, pep.candidates = ";".join(found.classes), ";".join(found.candidates)
        pep.series.append(Series(
            data_id, charge, int(rows["n_spectra"][r]),
            int(rows["n_flagged"][r]), int(rows["n_rejected"][r]), int(rows["n_multi"][r]),
            reviewed.get(data_id, 0), n_left.get(data_id, 0), **target, own_range=own[1],
            by_precursor=grouped))
    for pep in out.values():
        pep.series.sort(key=_order_under_precursor if grouped else (
            lambda s: (s.charge, s.ms_level, natural_key(s.condition), natural_key(s.fragment))))
    return [out[k] for k in sorted(out, key=lambda k: (natural_key(k[0]), natural_key(k[1]),
                                                       k[2], *k[3:]))]


def _precursor_key(own: tuple[str, str, str], charge: int, target: dict) -> tuple:
    """The precursor row of a series: (sample, the parent's residues and sequence, the
    precursor's charge); a series read at MS1 is a parent at its own charge."""
    if target.get("fragment") and target.get("parent_range"):
        return (own[0], target["parent_range"], target["parent_peptide"],
                int(target["parent_charge"]))
    return (*own, charge)


def _order_under_precursor(s: Series) -> tuple:
    """A precursor row's series in order: the MS1 series, the precursor ions, then the
    fragments by ion series (N-terminal types first), length, charge, MS level and
    condition, so that each fragment's conditions are together."""
    if not s.fragment:
        group, ion = 0, ""
    elif s.ion_type.startswith("M"):
        group, ion = 1, s.ion_type
    else:
        group, ion = (2 if s.ion_type[:1] in "abc" else 3), s.ion_type
    length = s.fragment.rpartition(":")[2]
    return (group, natural_key(ion), int(length) if length.isdigit() else 0, s.charge,
            s.ms_level, natural_key(s.condition), s.data_id)


def series_targets(results) -> dict[int, dict]:
    """Each series' (by ``data_id``) fragment, ion type, condition and MS level, as
    ``spectra.csv`` gives them (:func:`hxex.report.tables.target_columns`); empty for results
    not read from mzML or Thermo files."""
    from ..io.mzml import mzml_series_targets
    from ..report.tables import target_columns

    cols = target_columns(results)
    if not cols:
        return {}
    targets = mzml_series_targets(results.settings.spectra_source)
    out: dict[int, dict] = {}
    for i, d in enumerate(results.fit.data_id):
        if int(d) not in out:
            target = targets.get(int(d))
            extra = dict(target.entry.extra) if target is not None else {}
            z = extra.get("parent_charge", "")
            parent = str(cols["parent_range"][i])
            out[int(d)] = {"fragment": str(cols["fragment"][i]),
                           "ion_type": str(cols["ion_type"][i]),
                           "condition": str(cols["condition"][i]),
                           "ms_level": int(cols["ms_level"][i]),
                           "parent": f"{parent} z{z}" if parent and z != "" else parent,
                           "parent_range": parent,
                           "parent_peptide": str(extra.get("parent_peptide", "")),
                           "parent_charge": int(z) if str(z).isdigit() else 0}
    return out


def _reviewed(results) -> dict[int, int]:
    status = np.asarray(results.review_status).astype(str)
    ids = results.fit.data_id
    out: dict[int, int] = {}
    for d, s in zip(ids, status, strict=True):
        if s in ("accepted", "rejected"):
            out[int(d)] = out.get(int(d), 0) + 1
    return out


def spectrum_sets(results, config=None) -> list | None:
    """The test of each spectrum's set (:func:`hxex.analysis.set_test.spectrum_sets`) with the
    analysis settings ``config`` (its defaults when None), worked out with the log quiet; None
    without the population test."""
    from ..analysis.set_test import spectrum_sets as sets_of

    if getattr(results, "population_test", None) is None:
        return None
    with _quiet_here():
        return sets_of(results, config)


def left_out(results, config=None) -> list[str]:
    """Why the analysis settings ``config`` (:class:`hxex.analysis.AnalysisConfig`; its
    defaults when None) leave each spectrum of ``results`` out, '' for the ones kept: the
    reasons of ``left_out_spectra.csv`` (``[analysis.filters]``), ``exclude_flagged`` and
    ``exclude_times``, the fragment ions ``[analysis.fragments] leave_out_flags`` leaves out,
    and the spectra kept at one population (``spectrum_quality_filtering``).
    Rejected spectra are not listed (the review left them out). The reasons are the library's
    (:func:`hxex.analysis.left_out_reasons`; moved there, audit of 2026-11-15, Q-4); this
    wrapper only keeps the log quiet."""
    from ..analysis import left_out_reasons

    with _quiet_here():  # the controls' messages, said when the results were read (owner)
        return left_out_reasons(results, config)


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


def points(results, data_id: int, left: list[str] | None = None,
           sets: list | None = None) -> list[Point]:
    """The time points of a series, in its order, each with its spectra by replicate
    (``left``: :func:`left_out` of the results; ``sets``: :func:`spectrum_sets` of the
    results, the test of each spectrum's set, or None)."""
    from ..analysis.set_test import set_words
    from ..report.series import series_info, time_text

    info = series_info(results, data_id)
    status = np.asarray(results.review_status).astype(str)
    ks = results.effective_K
    flags = _flags(results)
    by_point: dict[int, Point] = {}
    for pos, i in enumerate(info["rows"]):
        order = int(info["time_index"][pos])
        point = by_point.get(order)
        if point is None:
            label = time_text(float(info["times"][pos]), str(info["roles"][pos]),
                               results.time_scale)
            point = by_point[order] = Point(label, order)
        found = sets[int(i)] if sets is not None else None
        sig = (results.population_test.config.significance
               if results.population_test is not None else None)
        point.spectra.append(Spectrum(
            int(i), str(results.fit.spectrum_id[i]), int(info["reps"][pos]), int(ks[i]),
            str(status[i]) if status[i] not in ("", "None", "unreviewed") else "",
            flags[i], left[i] if left is not None else "",
            int(found.K) if found is not None else 0,
            set_words(found, sig) if found is not None else ""))
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
    ``text`` (:func:`search_matches`: residues by number, the series' own (a fragment's, under
    a precursor row), else the sample, residues, peptide,
    charge, fragment, condition, or a pooled peptide's identity and candidates as text), the
    MS level ``ms_level`` and the ion type ``ion_type`` (None: any; ``''``: peptides, not
    fragments), and for pooled peptides the identity level ``identity_level`` (isoform,
    class, ambiguous or none) and the class ``pool_class`` (one of the peptide's classes;
    None: any)."""
    if text and not search_matches(text, pep.sample, series.own_range or pep.peptide_range,
                                   pep.peptide,
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
