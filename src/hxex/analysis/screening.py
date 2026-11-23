# SPDX-License-Identifier: GPL-3.0-or-later
"""What the analysis does with the spectrum quality screens (``[analysis.filters]
spectrum_quality_filtering``; HISTORY 149, 152), the picker's m/z check (``[analysis.filters]
off_mz``; HISTORY 200) and its claims (``[analysis.filters] other_ion``; PLAN 34).

The scores (:mod:`hxex.picking.screen`) are measured at picking; here a level turns them into
one of three actions per spectrum, at thresholds agreed with the owner from fit evidence
(counterfactual refits with the interference taken out, and replicate agreement):

- remove (the spectrum is left out whole, at every K): ``screen_below_mono`` (biases the K = 1
  mean), ``screen_off_position`` and ``screen_between`` (heavy replicate tails), and a high
  ``screen_beyond_reach`` (most of the picked intensity cannot be the peptide; such spectra
  disagree with their replicates or the predicted uptake, HISTORY 199);
- one population only (its K >= 2 descriptions are left out; the K = 1 mean is unharmed):
  ``screen_between_separate`` (creates false populations);
- warn (logged; kept): ``screen_beyond_reach`` below its removal level (no measurable effect
  on the fits at low values).

Every spectrum of a series the m/z check flags ``off_mz`` (:mod:`hxex.picking.mz_check`) is
removed too (``off its m/z``), controls included: its undeuterated envelope sits away from the
peptide's m/z, so the series most likely reads another species (on the Orbitrap digest of
HISTORY 200 the retention time pointed at another peptide in most such series). So is every
spectrum of a series whose undeuterated cluster another listed ion claims and it does not
(``other_ion``, :mod:`hxex.picking.claims`): it reads that ion; the reason names it and both
offsets (``on another ion's peaks (...)``).

A removed control counts as rejected: the back-exchange correction is recomputed without it
(:func:`hxex.analysis.species.analysis_controls`). The run's controls already leave out those the
loose level, ``off_mz`` and ``other_ion`` remove (:func:`control_removal`; audit 2026-10-28), so the
stored controls are the default analysis's. A spectrum the review accepted is never touched; neither
are fragment series (the levels were set on intact peptides) nor spectra that were not assessed (NaN
scores).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEVELS = ("none", "loose", "strict")
OFF_MZ_REASON = "off its m/z"
OTHER_ION_REASON = "on another ion's peaks"
# the window's "left out" column for a spectrum the screens keep at one population (not left out)
KEPT_AT_ONE_REASON = "kept at K = 1 (spectrum quality)"


def other_ion_reason(other: str, other_ppm: float, own_ppm: float, by_position: bool) -> str:
    """The reason a series on another ion's peaks is left out for: :data:`OTHER_ION_REASON`
    with the other ion and the two offsets in brackets (no ``;`` or ``, ``, which separate
    names in a cell)."""
    note = f"{other} at {other_ppm:+.1f} ppm, own grid at {own_ppm:+.1f} ppm"
    if by_position:
        note += " - by position alone: retention not known"
    return f"{OTHER_ION_REASON} ({note})"


def screen_reason(score: str) -> str:
    """The reason a spectrum the score ``score`` removes is left out for."""
    return f"spectrum quality ({score})"

# level -> action -> ((score, threshold), ...); a spectrum at or above a threshold takes the
# action (the first removing score names the reason)
THRESHOLDS: dict[str, dict[str, tuple[tuple[str, float], ...]]] = {
    "loose": {
        "remove": (("screen_below_mono", 0.10), ("screen_off_position", 0.20),
                   ("screen_between", 0.40), ("screen_beyond_reach", 0.50)),
        "one_population": (("screen_between_separate", 0.20),),
        "warn": (("screen_beyond_reach", 0.10),),
    },
    "strict": {
        "remove": (("screen_below_mono", 0.05), ("screen_off_position", 0.10),
                   ("screen_between", 0.20), ("screen_beyond_reach", 0.30)),
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


def screening(results, level: str, off_mz: bool = True, other_ion: bool = True) -> Screening:
    """The action the level ``level`` (:data:`LEVELS`) takes on every spectrum of the fit, with
    ``off_mz`` the removal of the series off their m/z, and with ``other_ion`` of those on
    another listed ion's peaks."""
    if level not in LEVELS:
        raise ValueError(f"spectrum_quality_filtering must be one of {', '.join(LEVELS)}")
    fit, binned = results.fit, results.binned
    n = len(fit.spectrum_id)
    if (level == "none" and not off_mz and not other_ion) or binned is None:
        return Screening.nothing(n)
    from ..controls import fragment_ids
    from ..io.mzml import fragment_series

    index = {str(s): j for j, s in enumerate(binned.spectrum_id)}
    rows = np.array([index[str(s)] for s in fit.spectrum_id], dtype=np.int64)
    fragments = fragment_ids(binned.modification, binned.data_id,
                             fragment_series(spectra_source(results)))
    eligible = ~np.isin(np.asarray(fit.data_id), list(fragments))
    eligible &= np.asarray(results.review_status) != "accepted"
    return _actions(binned, rows, eligible, level, off_mz, other_ion)


def control_removal(picked, fragments) -> np.ndarray:
    """``(M,)`` for every spectrum of ``picked`` (a :class:`hxex.picking.BinnedSet`): whether it
    is a control the default analysis removes (the loose level, ``off_mz`` and ``other_ion``; the
    run's controls leave those out, so that series.csv shows the state the default analysis uses;
    fragment series, ``fragments`` (data_id), are not screened)."""
    n = len(picked.spectrum_id)
    eligible = np.asarray(picked.role, dtype=str) != ""
    eligible &= ~np.isin(np.asarray(picked.data_id), list(fragments))
    return _actions(picked, np.arange(n), eligible, "loose", True, True).remove


def off_mz_spectra(binned) -> np.ndarray:
    """``(M,)`` for every spectrum of ``binned``: whether the m/z check flagged its series
    ``off_mz`` (False where not assessed, or in a bundle without the check)."""
    offset = np.asarray(binned.envelope_offset_ppm, dtype=float)
    limit = np.asarray(binned.envelope_offset_limit_ppm, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.abs(offset) > limit  # NaN (not assessed) never is


def other_ion_spectra(binned) -> np.ndarray:
    """``(M,)`` for every spectrum of ``binned``: whether its series' undeuterated cluster is
    another listed ion's (claim ``lost``: flag ``other_ion``; False where not assessed)."""
    return np.asarray(binned.claim, dtype=object) == "lost"


def _actions(binned, rows: np.ndarray, eligible: np.ndarray, level: str,
             off_mz: bool = False, other_ion: bool = False) -> Screening:
    """The actions of ``level`` (and with ``off_mz`` the removal of the series off their m/z,
    with ``other_ion`` of those on another ion's peaks) on the rows ``rows`` of ``binned``
    (``eligible``: those that may be removed)."""
    n = len(rows)

    def over(score: str, threshold: float) -> np.ndarray:
        values = np.asarray(getattr(binned, score), dtype=float)[rows]
        with np.errstate(invalid="ignore"):
            return eligible & (values >= threshold)  # NaN (not assessed) never is

    rules = THRESHOLDS.get(level, {"remove": (), "one_population": (), "warn": ()})
    remove = np.zeros(n, dtype=bool)
    reason = np.full(n, "", dtype=object)
    if other_ion:  # first: its reason names the other ion, which off_mz's does not
        lost = eligible & other_ion_spectra(binned)[rows]
        for j in np.flatnonzero(lost):
            r = rows[j]
            reason[j] = other_ion_reason(str(binned.claim_other[r]),
                                         float(binned.claim_other_offset_ppm[r]),
                                         float(binned.claim_offset_ppm[r]),
                                         bool(binned.claim_by_position[r]))
        remove |= lost
    if off_mz:
        hit = eligible & off_mz_spectra(binned)[rows] & ~remove
        reason[hit] = OFF_MZ_REASON
        remove |= hit
    for score, threshold in rules["remove"]:
        hit = over(score, threshold) & ~remove
        reason[hit] = screen_reason(score)
        remove |= hit
    one = np.zeros(n, dtype=bool)
    for score, threshold in rules["one_population"]:
        one |= over(score, threshold)
    warn = np.zeros(n, dtype=bool)
    for score, threshold in rules["warn"]:
        warn |= over(score, threshold)
    return Screening(remove, reason, one & ~remove, warn & ~remove)
