# SPDX-License-Identifier: GPL-3.0-or-later
"""The tree of the Results workspace, without Qt (docs/PLAN.md, "GUI", "Results and review"):
each peptide of each sample, its charges (the series), each series' time points (the
undeuterated controls first, the fully deuterated last, labelled as the figures label them),
and each point's replicates, which are the spectra. Every level carries the counts the window
shows, from the same tables ``hxex report`` writes (``series.csv``, ``spectra.csv``), with the
review applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# the filters of the tree: (the list's words, what they mean)
FILTERS = {"all": ("all series", "every series"),
           "flagged": ("flagged", "series with a spectrum that has a quality flag"),
           "multi": ("K above 1", "series with a spectrum of more than one population"),
           "rejected": ("rejected", "series with a rejected spectrum"),
           "unreviewed": ("not reviewed", "series with no spectrum accepted or rejected yet")}


@dataclass
class Spectrum:
    index: int  # the spectrum's position in the results
    spectrum_id: str
    rep: int
    K: int
    status: str  # review status: "", "accepted" or "rejected"
    flags: str  # quality flags, ";"-separated


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


@dataclass
class Peptide:
    sample: str
    peptide_range: str
    peptide: str
    series: list[Series] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {name: sum(getattr(s, name) for s in self.series)
                for name in ("n_spectra", "n_flagged", "n_rejected", "n_multi")}


def peptides(results) -> list[Peptide]:
    """Every peptide of every sample with its series, in the order of the samples and then of
    the residues and charges."""
    from ..report.tables import series_table

    rows = series_table(results)
    reviewed = _reviewed(results)
    out: dict[tuple, Peptide] = {}
    for r in range(len(rows["data_id"])):
        key = (str(rows["sample"][r]), str(rows["peptide_range"][r]), str(rows["peptide"][r]))
        pep = out.setdefault(key, Peptide(*key))
        data_id = int(rows["data_id"][r])
        pep.series.append(Series(
            data_id, int(rows["charge"][r] or 0), int(rows["n_spectra"][r]),
            int(rows["n_flagged"][r]), int(rows["n_rejected"][r]), int(rows["n_multi"][r]),
            reviewed.get(data_id, 0)))
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


def points(results, data_id: int) -> list[Point]:
    """The time points of a series, in its order, each with its spectra by replicate."""
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
            flags[i]))
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
    if how == "unreviewed":
        return series.n_reviewed == 0
    if how != "all":
        raise ValueError(f"unknown filter {how!r}")
    return True
