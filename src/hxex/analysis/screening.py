# SPDX-License-Identifier: GPL-3.0-or-later
"""What the analysis does with the spectrum quality screens (``[analysis.filters]
spectrum_quality_filtering``; HISTORY 149, 152).

The scores (:mod:`hxex.picking.screen`) are measured at picking; here a level turns them into
one of three actions per spectrum, at thresholds agreed with the owner from fit evidence
(counterfactual refits with the interference taken out, and replicate agreement):

- remove (the spectrum is left out whole, at every K): ``screen_below_mono`` (biases the K = 1
  mean), ``screen_off_position`` and ``screen_between`` (heavy replicate tails);
- one population only (its K >= 2 descriptions are left out; the K = 1 mean is unharmed):
  ``screen_between_separate`` (creates false populations);
- warn (logged; kept): ``screen_beyond_reach`` (no measurable effect on the fits).

A removed control counts as rejected: the back-exchange correction is recomputed without it
(:func:`hxex.analysis.species.analysis_controls`). The run's controls already leave out those
the loose level removes (:func:`control_removal`; audit 2026-10-28), so the stored controls are
the default analysis's. A spectrum the review accepted is never
touched; neither are fragment series (the levels were set on intact peptides) nor spectra
that were not assessed (NaN scores).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEVELS = ("none", "loose", "strict")

# level -> action -> ((score, threshold), ...); a spectrum at or above a threshold takes the
# action (the first removing score names the reason)
THRESHOLDS: dict[str, dict[str, tuple[tuple[str, float], ...]]] = {
    "loose": {
        "remove": (("screen_below_mono", 0.10), ("screen_off_position", 0.20),
                   ("screen_between", 0.40)),
        "one_population": (("screen_between_separate", 0.20),),
        "warn": (("screen_beyond_reach", 0.10),),
    },
    "strict": {
        "remove": (("screen_below_mono", 0.05), ("screen_off_position", 0.10),
                   ("screen_between", 0.20)),
        "one_population": (("screen_between_separate", 0.10),),
        "warn": (("screen_beyond_reach", 0.10),),
    },
}


@dataclass(frozen=True)
class Screening:
    """One entry per row of the fit: ``remove`` (left out whole) with its ``reason``,
    ``one_population`` (K >= 2 descriptions left out) and ``warn``."""

    remove: np.ndarray
    reason: np.ndarray
    one_population: np.ndarray
    warn: np.ndarray

    @classmethod
    def nothing(cls, n: int) -> Screening:
        no = np.zeros(n, dtype=bool)
        return cls(no, np.full(n, "", dtype=object), no.copy(), no.copy())


def spectra_source(results) -> dict | None:
    """How the spectra of ``results`` were read: a run result's own record, else the one in
    its settings (:class:`hxex.report.Results`)."""
    source = getattr(results, "spectra_source", None)
    settings = getattr(results, "settings", None)
    if source is None and settings is not None:
        source = settings.spectra_source
    return source


def screening(results, level: str) -> Screening:
    """The action the level ``level`` (:data:`LEVELS`) takes on every spectrum of the fit."""
    if level not in LEVELS:
        raise ValueError(f"spectrum_quality_filtering must be one of {', '.join(LEVELS)}")
    fit, binned = results.fit, results.binned
    n = len(fit.spectrum_id)
    if level == "none" or binned is None:
        return Screening.nothing(n)
    from ..controls import fragment_ids
    from ..io.mzml import fragment_series

    index = {str(s): j for j, s in enumerate(binned.spectrum_id)}
    rows = np.array([index[str(s)] for s in fit.spectrum_id], dtype=np.int64)
    fragments = fragment_ids(binned.modification, binned.data_id,
                             fragment_series(spectra_source(results)))
    eligible = ~np.isin(np.asarray(fit.data_id), list(fragments))
    eligible &= np.asarray(results.review_status) != "accepted"
    return _actions(binned, rows, eligible, level)


def control_removal(picked, fragments) -> np.ndarray:
    """``(M,)`` for every spectrum of ``picked`` (a :class:`hxex.picking.BinnedSet`): whether it
    is a control the loose level removes (the run's controls leave those out, so that
    series.csv shows the state the default analysis uses; fragment series, ``fragments``
    (data_id), are not screened)."""
    n = len(picked.spectrum_id)
    eligible = np.asarray(picked.role, dtype=str) != ""
    eligible &= ~np.isin(np.asarray(picked.data_id), list(fragments))
    return _actions(picked, np.arange(n), eligible, "loose").remove


def _actions(binned, rows: np.ndarray, eligible: np.ndarray, level: str) -> Screening:
    """The actions of ``level`` on the rows ``rows`` of ``binned`` (``eligible``: those that
    may be screened)."""
    n = len(rows)

    def over(score: str, threshold: float) -> np.ndarray:
        values = np.asarray(getattr(binned, score), dtype=float)[rows]
        with np.errstate(invalid="ignore"):
            return eligible & (values >= threshold)  # NaN (not assessed) never is

    rules = THRESHOLDS[level]
    remove = np.zeros(n, dtype=bool)
    reason = np.full(n, "", dtype=object)
    for score, threshold in rules["remove"]:
        hit = over(score, threshold) & ~remove
        reason[hit] = f"spectrum quality ({score})"
        remove |= hit
    one = np.zeros(n, dtype=bool)
    for score, threshold in rules["one_population"]:
        one |= over(score, threshold)
    warn = np.zeros(n, dtype=bool)
    for score, threshold in rules["warn"]:
        warn |= over(score, threshold)
    return Screening(remove, reason, one & ~remove, warn & ~remove)
