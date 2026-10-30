# SPDX-License-Identifier: GPL-3.0-or-later
"""The species uptake table: fractional deuteration per sample, species, time and rank.

A species is a peptide (or the intact protein, or a fragment ion): its sequence with
modifications and its position on the protein. Its values come from the fits at a fixed number
of populations ``K`` (every measurement is fitted at every K; the standard errors come from the
fixed-K bootstrap, or from the selected-K one where that K was selected), so that the populations
of overlapping peptides can be compared rank by rank. Components are ranked by their
fractional deuteration, 1 the lowest.

Two measures of deuteration, as in the fit: ``mean``, the absolute deuterium (deuterons beyond
natural abundance, from 0 to about ``n_amides``), back-exchange corrected when the run has
controls (``corrected``; never for fragment ions, :mod:`hxex.controls`), and ``fD``, the
**fractional deuteration**, ``mean / n_amides`` (0 to 1). Both mean the same for forward
labelling and for reverse exchange (docs/ANALYSIS.md).

Spectra the review rejected are left out; flagged ones are kept unless ``exclude_flagged``. The
spectra of a species at one time are combined in two steps. The charge states of each replicate
are averaged first (they share the replicate's sample, labelling, quench, and injection, so they
are not independent replicates; the mean's error is ``sqrt(sum se_i^2) / c`` of their own
errors); the replicates are then combined: the value is their mean and its standard error
``sqrt(max(mean(se_i^2), s^2) / n)`` over the ``n`` replicates, the larger of their own
(bootstrap) variance and their observed scatter ``s^2``, so that neither is counted twice; with
one replicate it is that replicate's standard error. ``charge_trend`` flags a species whose charge
states disagree systematically (see :func:`species_uptake`).
"""

from __future__ import annotations

import dataclasses
import logging
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy import stats

from .. import chem
from ..controls import ControlConfig, apply_control_roles, correct_means, fit_controls
from ..io.times import format_time
from ..spectra import point_order_key
from .screening import LEVELS as SCREEN_LEVELS
from .screening import Screening, screening, spectra_source

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeciesFilters:
    """Which spectra and which K >= 2 descriptions the analysis trusts (the
    ``[analysis.filters]`` table; docs/ANALYSIS.md). None (``[analysis.filters] = false``)
    keeps everything.

    Whole spectra, at every K (:func:`left_out_spectra`; a spectrum the review accepted is
    always kept; owner, 2026-10-22, HISTORY 140):

    - ``replicate_outlier``: at an exposure time, a spectrum whose K=1 fD lies more than this
      many typical replicate standard deviations from the median of its series' replicates at
      that time (the same peptide and charge state; at least three replicates) is left out
      (``replicate outlier``). The typical deviation is the run's own: the median of the
      replicate variances of all its time points (each scaled by its chi-square median),
      robust to the outliers it looks for. None turns it off. Charge states are not compared
      with each other here (``charge_trend`` does that).
    - ``above_amides``: a spectrum whose K=1 mean exceeds the peptide's exchangeable amides
      before the back-exchange correction is left out (``above the amide count``): no
      correction makes it uptake.
    - ``stand_in_controls``: ``'exclude'`` (the default) leaves out the series whose fully
      deuterated control is a stand-in (the run's median retention, the kint prediction or
      ``d2o_fraction``; not one measured at the peptide's other charge states), except in a
      sample with no measured fully deuterated control at all, whose series all use one
      (``stand-in control``); ``'other_sample'`` leaves them out too but keeps the series that
      took the same peptide's retention in the run's other samples (``fd_other_sample``, median
      error 0.006-0.012: as close as the peptide's charge states; owner, 2026-10-30), not the
      retention map or the run-level stand-ins; ``'include'`` keeps them all. A series whose
      measured controls were all removed by ``spectrum_quality_filtering`` says so
      (``stand-in control (screened)``). ``'exclude'`` (and ``'other_sample'``) also leaves
      out the series whose undeuterated control read above ``[run.controls] m0_max``
      (``m0_high``: the windows hold another species, which the labelled spectra share;
      ``stand-in control (undeuterated)``), with no exception for a
      sample (a warning when a sample loses half its series or more). A missing undeuterated
      control (``m0_missing``) is not a reason: 0 stands in for it fairly.
    - ``spectrum_quality_filtering``: what the spectrum quality screens
      (:mod:`hxex.picking.screen`) do, at ``'loose'`` (the default) or ``'strict'``
      thresholds (:mod:`hxex.analysis.screening`): a spectrum showing another species is left
      out (``spectrum quality (<score>)``; a control is then left out of the back-exchange
      correction too), one with a separate cluster between the isotope positions keeps only
      its K = 1 description, and intensity beyond the reachable range is a warning.
      ``'none'`` turns it off. Fragment series are not screened.
    - ``undeuterated_cap``: with the population test (``[analysis] populations =
      "population_test"``), keep the labelled and fully deuterated spectra of a peptide whose
      undeuterated set has two populations or more, in any sample of the run, at one population
      (off by default: such peptides are flagged, ``undeuterated_two``;
      :mod:`hxex.analysis.set_test`).

    Two-population descriptions, per spectrum, before replicates and charge states are
    combined:

    - ``min_population``: every population within ``[min_population, 1 - min_population]`` (the
      fit has a limit of the same name on the fitted K, and the choice on the lever-rule
      population).
    - ``max_fD``: no component deuterated beyond this (its ``n * p`` over the exchangeable
      amides, back-exchange corrected when the run has controls, as reported: more than full
      deuteration is not physical; the fit itself allows ``n`` up to the run's ``n_max_factor``
      times the amides).

    Per species and time, after combining:

    - ``max_fD_se``: every rank's ``fD`` standard error below this (unknown errors pass).
    - ``k1_between_ranks``: the K = 1 value between the lowest and the highest rank (``fD_low <
      fD_avg < fD_high``; needs K = 1 among the K analysed): a split whose average does not lie
      between its populations does not describe the same molecules.
    """

    min_population: float = 0.05
    max_fD_se: float = 0.25
    max_fD: float = 1.1
    k1_between_ranks: bool = True
    replicate_outlier: float | None = 6.0
    above_amides: bool = True
    stand_in_controls: str = "exclude"
    spectrum_quality_filtering: str = "loose"
    undeuterated_cap: bool = False

    def __post_init__(self):
        if not 0 <= self.min_population < 0.5:
            raise ValueError("min_population must be in [0, 0.5)")
        if self.max_fD_se <= 0 or self.max_fD <= 0:
            raise ValueError("max_fD_se and max_fD must be positive")
        if self.replicate_outlier is not None and (isinstance(self.replicate_outlier, bool)
                                                   or not self.replicate_outlier > 0):
            raise ValueError("replicate_outlier must be a positive number of standard "
                             "deviations (or unset)")
        if not isinstance(self.above_amides, bool):
            raise ValueError("above_amides must be true or false")
        if self.stand_in_controls not in STAND_IN_CONTROLS:
            raise ValueError(f"stand_in_controls must be one of {', '.join(STAND_IN_CONTROLS)}")
        if not isinstance(self.undeuterated_cap, bool):
            raise ValueError("undeuterated_cap must be true or false")
        if self.spectrum_quality_filtering not in SCREEN_LEVELS:
            raise ValueError("spectrum_quality_filtering must be one of "
                             f"{', '.join(SCREEN_LEVELS)}")


STAND_IN_CONTROLS = ("exclude", "other_sample", "include")


# a charge trend: the lowest and highest charge differ by more than this many combined standard
# errors, in the same direction, at this share of the times both were measured (and at least two)
TREND_Z = 2.0
TREND_SHARE = 0.5


_SCALARS = ("nterm_subtract", "populations")  # settings of the table, not columns


@dataclass(frozen=True)
class SpeciesUptake:
    """One row per run, sample, species, time, ``K`` and ``rank`` (columns of equal length).

    Identity: ``run`` (the run the row's spectra come from: '' for one run, the run's name in a
    :class:`hxex.analysis.RunSet`, and '' for a merged sample, whose spectra pool several
    runs; a species is identified by ``(run, sample, ...)``, and ``run:sample`` is a display
    label only, :func:`hxex.analysis.display_labels`), ``sample``, ``peptide`` (modX),
    ``modification``, ``kind`` (``'peptide'``, or
    ``'fragment'`` for an ion type other than ``M``), ``start`` and ``end`` (residues on the
    protein, inclusive). Values: ``time``, ``K``, ``rank``, ``mean`` and ``mean_se`` (deuterons),
    ``fD`` and ``fD_se`` (``mean / n_amides``),
    ``population`` and ``population_se`` (the component's share), ``n_amides`` (exchangeable
    amides),
    ``n_spectra``, ``n_charges``, ``corrected`` and ``charge_trend``; ``control_fallback``: the
    back-exchange correction of at least one of its spectra used a fallback control (a missing
    or rejected control, :mod:`hxex.controls`); ``fd_other_charge``: every such fallback was a
    fully deuterated control taken from the same peptide's other charge states, which counts as
    measured (:attr:`hxex.controls.ControlTable.measured`). From a
    :class:`hxex.analysis.RunSet`: ``runs`` (the runs its spectra come from, e.g. ``'A,B'``
    for a merged sample; '' for one run),
    ``n_spectra_by_run`` (``'4,4'``) and ``run_offset`` (a merged species whose runs disagree
    systematically; see :func:`species_uptake`); ``best_set_K``, ``best_set``,
    ``best_set_reason`` and ``set_delta_criterion``: the number of populations chosen for the
    species at that time from its whole set of spectra (:func:`best_set`, or the population
    test, :mod:`hxex.analysis.set_test`, whose p-values are ``set_p_1``, one against two, and
    ``set_p_2``, two against three, NaN without the test; ``set_test_K`` is the test's own
    number, which ``best_set_K`` can only follow up to the largest K analysed; 0 without the
    test; ``undeuterated_two``, a flag on every row of a peptide whose undeuterated set, in
    some sample of the run, has two populations or more by the test: another species); ``rt``:
    the median
    retention time of its
    spectra (minutes; NaN: unknown); ``role``: ``'undeut'`` or ``'fulldeut'`` for a control
    (whose ``time`` is NaN unless it was also a time point), ``''`` for a time point: coverage,
    redundancy and resolution refer to the undeuterated controls. ``nterm_subtract`` (the
    N-terminal residues that do not count) is the run's; ``populations`` is the rule the table
    was made with (:data:`hxex.analysis.settings.POPULATION_RULES`), which
    :func:`hxex.analysis.residue_map` follows.
    """

    sample: np.ndarray
    peptide: np.ndarray
    modification: np.ndarray
    kind: np.ndarray
    start: np.ndarray
    end: np.ndarray
    time: np.ndarray
    K: np.ndarray
    rank: np.ndarray
    mean: np.ndarray
    mean_se: np.ndarray
    fD: np.ndarray
    fD_se: np.ndarray
    population: np.ndarray
    population_se: np.ndarray
    n_amides: np.ndarray
    n_spectra: np.ndarray
    n_charges: np.ndarray
    corrected: np.ndarray
    charge_trend: np.ndarray
    control_fallback: np.ndarray | None = None
    runs: np.ndarray | None = None
    n_spectra_by_run: np.ndarray | None = None
    run_offset: np.ndarray | None = None
    fd_other_charge: np.ndarray | None = None
    rt: np.ndarray | None = None
    role: np.ndarray | None = None
    run: np.ndarray | None = None
    best_set_K: np.ndarray | None = None
    best_set: np.ndarray | None = None
    best_set_reason: np.ndarray | None = None
    set_delta_criterion: np.ndarray | None = None
    set_p_1: np.ndarray | None = None
    set_p_2: np.ndarray | None = None
    set_test_K: np.ndarray | None = None
    undeuterated_two: np.ndarray | None = None
    nterm_subtract: int = 1
    populations: str = "best_set"

    def __post_init__(self):  # built without them: no fallback, one run, no controls
        n = len(self.sample)
        if self.run is None:
            object.__setattr__(self, "run", np.full(n, "", dtype=object))
        else:
            object.__setattr__(self, "run", np.asarray(self.run, dtype=object))
        if self.role is None:
            object.__setattr__(self, "role", np.full(n, "", dtype="<U8"))
        else:
            object.__setattr__(self, "role", np.asarray(self.role, dtype=str))
        for name, empty in (("control_fallback", np.zeros(n, dtype=bool)),
                            ("runs", np.full(n, "", dtype=object)),
                            ("n_spectra_by_run", np.full(n, "", dtype=object)),
                            ("run_offset", np.zeros(n, dtype=bool)),
                            ("fd_other_charge", np.zeros(n, dtype=bool)),
                            ("rt", np.full(n, np.nan)),
                            ("best_set_K", np.zeros(n, dtype=np.int64)),
                            ("best_set", np.zeros(n, dtype=bool)),
                            ("best_set_reason", np.full(n, "", dtype=object)),
                            ("set_delta_criterion", np.full(n, np.nan)),
                            ("set_p_1", np.full(n, np.nan)), ("set_p_2", np.full(n, np.nan)),
                            ("set_test_K", np.zeros(n, dtype=np.int64)),
                            ("undeuterated_two", np.zeros(n, dtype=bool))):
            if getattr(self, name) is None:
                object.__setattr__(self, name, empty)

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        """The table as columns (for :func:`hxex.report.write_csv`), ``run`` first."""
        return {"run": self.run, **{f.name: getattr(self, f.name) for f in fields(self)
                                    if f.name not in _SCALARS and f.name != "run"}}

    def select(self, mask) -> SpeciesUptake:
        mask = np.asarray(mask)
        return SpeciesUptake(**{f.name: getattr(self, f.name)[mask] for f in fields(self)
                                if f.name not in _SCALARS},
                             **{name: getattr(self, name) for name in _SCALARS})

    def where(self, *, sample=None, time=None, K=None, rank=None, kind=None,
              role=None, run=None) -> np.ndarray:
        """Rows matching every condition given (a value, or a list of values); ``role``
        ``'undeut'``/``'fulldeut'`` selects the controls, ``''`` the time points; ``sample``
        alone matches the sample in every run."""
        keep = np.ones(len(self), dtype=bool)
        if role is not None:
            keep &= np.isin(self.role, np.atleast_1d(role))
        for name, value in (("run", run), ("sample", sample), ("time", time), ("K", K),
                            ("rank", rank), ("kind", kind)):
            if value is not None:
                values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
                keep &= np.isin(getattr(self, name), values)
        return keep


def _by_replicate(recs, at: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The values (record field ``at``: 0 fD, 2 population) and standard errors of a species'
    spectra with the charge states of each replicate (its run and ``rep``) averaged: one value
    per replicate, in the order the replicates first appear. A replicate's error is
    ``sqrt(sum se_i^2) / c`` over its ``c`` charge states with a known error (NaN without
    any)."""
    reps: dict[tuple, list] = {}
    for r in recs:
        reps.setdefault((r[8], r[6]), []).append(r)
    values, ses = [], []
    for members in reps.values():
        values.append(float(np.mean([m[at] for m in members])))
        own = np.array([m[at + 1] for m in members], dtype=float)
        own = own[np.isfinite(own)]
        ses.append(float(np.sqrt(np.sum(own**2)) / len(own)) if len(own) else np.nan)
    return np.array(values), np.array(ses)


def _combine_replicates(recs, at: int = 0) -> tuple[float, float]:
    """:func:`_combine` of :func:`_by_replicate`: a species' value and standard error."""
    return _combine(*_by_replicate(recs, at))


def _combine(values: np.ndarray, ses: np.ndarray) -> tuple[float, float]:
    """Mean of ``values`` and its standard error (see the module docstring)."""
    n = len(values)
    mean = float(np.mean(values))
    finite = ses[np.isfinite(ses)]
    own = float(np.mean(finite**2)) if len(finite) else np.nan
    if n == 1:
        return mean, float(ses[0])
    scatter = float(np.var(values, ddof=1))
    var = scatter if not np.isfinite(own) else max(own, scatter)
    return mean, float(np.sqrt(var / n))


# recomputed controls by results (weakly held) and changes: analysis_controls computes them once
_RECOMPUTED: dict[tuple, tuple[weakref.ref, Any]] = {}


def analysis_controls(results, changes: Mapping[str, Any] | None = None, *, quiet: bool = False,
                      screened: np.ndarray | None = None):
    """The controls the analysis uses (:class:`hxex.controls.ControlTable`, or None without
    controls): those of the run, recomputed from its K=1 fits when the review rejected a control
    spectrum (so that a rejected control does not count), when ``screened`` (one entry per row
    of the fit: the spectra the spectrum quality screens remove; None: those the run's
    controls left out, ``ControlTable.screened``) holds other control spectra than the run's
    controls left out, or when ``changes`` (control settings to change, e.g.
    ``{'d2o_fraction': 0.85}``) are given. Recomputed controls leave out the rejected and
    screened spectra and the noise outliers."""
    stored = results.controls
    run = results.settings.run if results.settings is not None else None
    base = stored.config if stored is not None else (run.controls if run is not None else None)
    if base is None:
        if not changes:
            return None
        base = ControlConfig()
    fit = results.fit
    rejected = np.asarray(results.review_status == "rejected", dtype=bool)
    config = dataclasses.replace(base, **dict(changes or {}))
    if config.undeut_at is not None or config.fulldeut_at is not None:
        # a time point promoted to a control for the analysis (or by the run)
        fit = dataclasses.replace(fit, role=apply_control_roles(
            fit.role, fit.time, config, where="[analysis.controls]"))
    at_control = np.asarray(fit.role, dtype=str) != ""
    n_rejected = int((rejected & at_control).sum())
    by_run = set(stored.screened or ()) if stored is not None else set()
    if screened is None:  # as the run's controls
        screened = np.isin(np.asarray(fit.spectrum_id).astype(str), list(by_run))
    screened = np.asarray(screened, dtype=bool) & at_control & ~rejected
    n_screened = int(screened.sum())
    same = set(np.asarray(fit.spectrum_id).astype(str)[screened]) == by_run
    if stored is not None and not changes and not n_rejected and same:
        return stored
    # one analysis asks for the same recomputed controls once per table: compute them once
    key = (id(results), tuple(sorted((str(k), repr(v)) for k, v in (changes or {}).items())),
           np.flatnonzero(screened).tobytes())
    hit = _RECOMPUTED.get(key)
    if hit is not None and hit[0]() is results:
        return hit[1]
    usable = ~rejected
    if fit.noise_estimate is not None:
        usable &= ~np.asarray(fit.noise_estimate.outlier, dtype=bool)
    rates = rt = None
    if config.fd_mode == "kint":
        if results.binned is None or run is None:
            raise ValueError("fd_fallback 'kint' needs the run's picked spectra and settings "
                             "(the bundle's binned table)")
        from ..controls import series_rt
        from ..run import control_rates

        keep = np.isin(results.binned.spectrum_id, fit.spectrum_id)
        picked = results.binned.select(keep)
        rates = control_rates(picked, dataclasses.replace(run, controls=config))
        rt = series_rt(picked.data_id, picked.rt)
    groups = charges = residues = None
    fragments: set[int] = set()
    if results.binned is not None:
        from ..controls import charge_groups, fragment_ids, series_groups, series_residues
        from ..io.mzml import fragment_series

        b = results.binned
        groups = series_groups(b.peptide, b.modification, b.charge, b.data_id)
        scan_fragments = fragment_series(spectra_source(results))
        charges = charge_groups(b.sample, b.peptide, b.modification, b.charge, b.data_id,
                                scan_fragments)
        fragments = fragment_ids(b.modification, b.data_id, scan_fragments)
        picker = run.picker if run is not None else None
        residues = series_residues(
            b.peptide, b.modification, b.start_seq, b.data_id,
            nterm_subtract=picker.nterm_subtract if picker is not None else 1,
            rules=picker.amide_rules if picker is not None else None,
            registry=results.settings.run.registry if run is not None else None,
            fragments=scan_fragments)
    deamidation = None
    if config.deamidation:
        if charges is None:
            log.warning("deamidation: not assessed (the bundle has no picked table)")
        else:
            from ..deamidation import deamidation_from_picked
            deamidation = deamidation_from_picked(
                results.binned, fit.spectrum_id, fit.role, usable & ~screened, charges,
                residues, min_share=config.deamidation_min_share,
                registry=results.settings.run.registry if run is not None else None)
    out = fit_controls(fit, _site_counts(results), config, usable=usable, rates=rates, rt=rt,
                       groups=groups, charges=charges, uncorrected=fragments,
                       screened=screened if n_screened else None, residues=residues,
                       deamidation=deamidation)
    for k in [k for k, (ref, _) in _RECOMPUTED.items() if ref() is None]:
        del _RECOMPUTED[k]
    _RECOMPUTED[key] = (weakref.ref(results), out)
    if not quiet:
        log.info("controls recomputed (%s): %d of %d series fall back%s",
                 "; ".join(x for x in (
                     f"{n_rejected} rejected control spectra left out" if n_rejected else "",
                     f"{n_screened} control spectra left out by spectrum_quality_filtering"
                     if n_screened else "",
                     f"settings changed: {', '.join(sorted(changes))}" if changes else "") if x),
                 int(out.fallback.sum()), len(out),
                 f" (stored: {int(stored.fallback.sum())})" if stored is not None else "")
    return out


def _site_counts(results) -> np.ndarray:
    """The exchangeable sites of every measurement: from the stored controls, else counted on
    the picked table (:func:`hxex.run.site_counts`), else read back from ``n_max`` (rounded)."""
    fit = results.fit
    if results.controls is not None:
        try:
            return np.asarray(results.controls.n_amides, dtype=float)[
                results.controls.rows(fit.data_id)]
        except KeyError:
            pass
    run = results.settings.run if results.settings is not None else None
    if results.binned is not None and run is not None:
        from ..run import site_counts

        keep = np.isin(results.binned.spectrum_id, fit.spectrum_id)
        sub = results.binned.select(keep)
        index = {str(s): j for j, s in enumerate(sub.spectrum_id)}
        counts = site_counts(sub, run)
        return np.array([counts[index[str(s)]] for s in fit.spectrum_id], dtype=float)
    factor = run.n_max_factor if run is not None else 1.2
    return np.rint(np.asarray(results.measurements.n_max, dtype=float) / factor)


def _records(results, ks, exclude_flagged, filters, quiet=False, controls_changes=None,
             counts=None, evidence=None):
    """The filtered values of every spectrum, grouped by (run, sample, peptide, modification,
    start, end, time, K, rank): lists of ``(fD, fD_se, population, population_se, charge,
    n_amides, rep, control_fallback, run, not_measured, uncorrected)`` (run: '' for one run, the
    run's name in a
    :class:`hxex.analysis.RunSet`; the group's run is '' for a merged sample); with whether
    each sample's values are corrected ((run, sample) -> bool) and ``nterm_subtract``. The
    point of a group is a :class:`Point` (``(time, role)``: a time, or a control role).
    ``results`` may be a :class:`hxex.analysis.RunSet`. ``counts`` (a dict) receives what the
    K >= 2 filters left out (:func:`_filter_species`; for :func:`left_out_summary`);
    ``evidence`` (a dict) the summed criterion of each set (:func:`_add_evidence`)."""
    from .runs import RunSet

    if isinstance(results, RunSet):
        return results._records(ks, exclude_flagged, filters, quiet, controls_changes,
                                counts=counts, evidence=evidence)
    groups, corrected, nterm, dropped = _spectrum_records(
        results, ks, exclude_flagged, filters, quiet, controls_changes, evidence=evidence)
    groups = _filter_species(groups, ks, filters, dropped, quiet)
    if counts is not None:
        counts.update(dropped)
    return groups, corrected, nterm


def _kept(results, exclude_flagged: bool) -> np.ndarray:
    """The rows of the fit the tables use: not rejected by the review, not left out of the
    analysis (``exclude_times``, another branch's series) and, with ``exclude_flagged``, not
    flagged by the quality checks."""
    keep = np.ones(len(results.fit), dtype=bool)
    keep &= results.review_status != "rejected"
    if getattr(results, "excluded_times", ()) or getattr(results, "excluded_series", ()):
        keep &= ~results.excluded
    if exclude_flagged and results.quality is not None:
        keep &= ~np.asarray(results.quality.flagged, dtype=bool)
    return keep


def _screened_controls(results, changes, screen: Screening, keep: np.ndarray, quiet: bool):
    """The controls the analysis uses with the screened controls left out, and those before
    (the same object when the screens left none out)."""
    removed = screen.remove & keep & (np.asarray(results.review_status) != "accepted")
    removed &= np.asarray(results.fit.role, dtype=str) != ""
    unscreened = analysis_controls(results, changes, quiet=quiet or bool(removed.any()),
                                   screened=np.zeros(len(removed), dtype=bool))
    if not removed.any() or unscreened is None:
        return unscreened, unscreened
    return analysis_controls(results, changes, quiet=quiet, screened=removed), unscreened


def left_out_spectra(results, filters: SpeciesFilters | None = SpeciesFilters(),  # noqa: B008
                     *, exclude_flagged: bool = False,
                     controls: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
    """The spectra the analysis leaves out whole (:class:`SpeciesFilters`:
    ``replicate_outlier``, ``above_amides``, ``stand_in_controls``), with the reason:
    columns ``run`` ('' for one run), ``spectrum_id``, ``sample``, ``peptide_range``,
    ``charge``, ``rep``, ``time``, ``role``, ``fD`` (K=1, corrected as the analysis corrects
    it) and ``reason``. Spectra the review rejected are not listed (they are out already)."""
    from .runs import RunSet

    if isinstance(results, RunSet):
        parts = []
        for name, r in results.runs.items():
            part = left_out_spectra(r, filters, exclude_flagged=exclude_flagged,
                                    controls=controls)
            part["run"] = np.full(len(part["spectrum_id"]), name, dtype=object).astype(str)
            parts.append(part)
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    fit, binned = results.fit, results.binned
    keep = _kept(results, exclude_flagged)
    n_amides = _site_counts(results)
    if filters is None:
        out, why, fd = np.zeros(len(fit), dtype=bool), np.full(len(fit), "", dtype=object), None
    else:
        screen = screening(results, filters.spectrum_quality_filtering)
        ctl, unscreened = _screened_controls(results, controls, screen, keep, quiet=True)
        out, why, fd = _left_out(results, filters, ctl, n_amides, keep, values=True,
                                 screen=screen, unscreened=unscreened)
    index = {str(s): j for j, s in enumerate(binned.spectrum_id)}
    rows = np.array([index[str(s)] for s in fit.spectrum_id], dtype=np.int64)
    i = np.flatnonzero(out)
    b = rows[i]
    return {
        "run": np.full(len(i), "", dtype=object).astype(str),
        "spectrum_id": np.asarray(fit.spectrum_id)[i].astype(str),
        "sample": np.asarray(binned.sample)[b].astype(str),
        "peptide_range": np.asarray(binned.peptide_range)[b].astype(str),
        "charge": np.asarray(binned.charge)[b].astype(np.int64),
        "rep": np.asarray(binned.rep)[b],
        "time": np.asarray(fit.time, dtype=float)[i],
        "role": np.asarray(fit.role).astype(str)[i],
        "fD": (fd[i] if fd is not None else np.full(len(i), np.nan)),
        "reason": why[i].astype(str),
    }


# control flags that mean a fully deuterated control was measured but not used
_FD_REJECTED = ("retention_low", "retention_high", "retention_outlier", "fd_below_data",
                "fd_off_model")
# the reasons the spectrum quality screens remove a spectrum, in the order of the summary
_SCREEN_REMOVE = ("screen_below_mono", "screen_off_position", "screen_between")


def left_out_summary(results, filters: SpeciesFilters | None = SpeciesFilters(),  # noqa: B008
                     *, ks: Sequence[int] = (1, 2), exclude_flagged: bool = False,
                     controls: Mapping[str, Any] | None = None,
                     left_out: Mapping[str, np.ndarray] | None = None,
                     exclude_times: Sequence[float] = (),
                     population_test: bool = False) -> dict[str, np.ndarray]:
    """What was left out on the way from the spectra read to the analysis, one row per step,
    for people (``left_out_summary.csv``): columns ``run`` ('' for one run), ``stage``
    (``before fitting``, ``review``, ``controls``, ``analysis: whole spectra``, ``analysis:
    two-population descriptions``), ``what``, ``count``, ``unit`` (spectra, series, or
    peptides and times) and ``setting`` (where it is set). Every check that ran has a row, a
    count of 0 included. ``left_out``: the :func:`left_out_spectra` table if already made.
    ``population_test``: the analysis decides by the population test (a row for the spectra
    held out of it)."""
    from .runs import RunSet

    rows: list[tuple] = []

    def add(run, stage, what, count, unit, setting=""):
        rows.append((run, stage, what, int(count), unit, setting))

    runs = results.runs.items() if isinstance(results, RunSet) else [("", results)]
    if left_out is None and filters is not None:
        left_out = left_out_spectra(results, filters, exclude_flagged=exclude_flagged,
                                    controls=controls)
    for name, r in runs:
        fit = r.fit
        record = r.settings.excluded if r.settings is not None else {}
        n_excluded = len(record.get("spectrum_ids", ()))
        picked = len(r.binned) if r.binned is not None else len(fit)
        stage = "before fitting"
        add(name, stage, "spectra read", picked + n_excluded, "spectra")
        rules = record.get("rules", ())
        for rule in rules:
            keys = ", ".join(f"{k} {format_time(v) if k == 'time' else v}"
                             for k, v in rule.items() if k not in ("note", "matched"))
            what = f"{keys} ({rule['note']})" if rule.get("note") else keys
            add(name, stage, f"matched by an [[input.exclude]] rule: {what}; stored, not "
                "fitted", rule.get("matched", 0), "spectra", "[[input.exclude]]")
        if len(rules) != 1:
            add(name, stage, "left out by [[input.exclude]] in all" if rules else
                "left out by [[input.exclude]] (no rules set)", n_excluded, "spectra",
                "[[input.exclude]]")
        add(name, stage, "no peaks to fit (empty, or without signal)", picked - len(fit),
            "spectra", "[run.picker]")
        add(name, stage, "fitted", len(fit), "spectra")
        stage = "review"
        status = np.asarray(r.review_status)
        add(name, stage, "rejected (the review table, [[review.exclude]], [analysis] exclude)",
            np.sum(status == "rejected"), "spectra", "[review]")
        chosen = (np.asarray(r.effective_K) != np.asarray(fit.selected_K)) & (status != "rejected")
        add(name, stage, "kept, with the number of populations chosen in the review (review_K)",
            np.sum(chosen), "spectra", "[review] table")
        if exclude_times:
            add(name, stage, "at a time left out by exclude_times", np.sum(r.excluded),
                "spectra", "[analysis] exclude_times")
        keep = _kept(r, exclude_flagged)
        if exclude_flagged:
            add(name, stage, "flagged by a quality check (exclude_flagged)",
                np.sum(~keep & (status != "rejected")), "spectra", "[analysis] exclude_flagged")
        level = filters.spectrum_quality_filtering if filters is not None else "none"
        screen = screening(r, level) if filters is not None else Screening.nothing(len(fit))
        ctl, _ = _screened_controls(r, controls, screen, keep, quiet=True)
        stage = "controls"
        if ctl is None:
            add(name, stage, "no back-exchange correction (no [run.controls] table)", 0,
                "series", "[run.controls]")
        else:
            corrected = ~np.asarray(ctl.uncorrected, dtype=bool)
            tokens = [set(str(f).split(";")) for f in ctl.flags]

            def series_with(*names, tokens=tokens, corrected=corrected):
                return sum(1 for t, c in zip(tokens, corrected, strict=True)
                           if c and t & set(names))

            role = np.asarray(fit.role, dtype=str)
            screened = (screen.remove & keep & (status != "accepted") & (role != ""))
            add(name, stage, "series corrected", np.sum(corrected), "series")
            add(name, stage, "control spectra left out by the spectrum quality screens",
                np.sum(screened), "spectra", "[analysis.filters] spectrum_quality_filtering")
            add(name, stage, "a control replicate left out (far from the others, above the "
                "amide count, or a noise outlier)",
                series_with("m0_replicate_left_out", "fd_replicate_left_out"), "series",
                "[run.controls] replicate_outlier")
            add(name, stage, "undeuterated control missing or rejected: natural abundance used",
                series_with("m0_missing", "m0_high"), "series", "[run.controls] m0_max")
            add(name, stage, "fully deuterated control missing", series_with("fd_missing"),
                "series", "")
            add(name, stage, "fully deuterated control rejected (retention, or below the "
                "series' own time points)", series_with(*_FD_REJECTED), "series",
                "[run.controls]")
            add(name, stage, "corrected with a stand-in fully deuterated control",
                np.sum(~np.asarray(ctl.fd_measured, dtype=bool) & corrected), "series",
                "[run.controls] fd_fallback")
        if filters is None:
            continue
        stage = "analysis: whole spectra"
        mine = (np.asarray(left_out["run"]).astype(str) == name) if left_out is not None else []
        reasons = (np.asarray(left_out["reason"]).astype(str)[mine] if left_out is not None
                   else np.array([], dtype=str))
        listed = [f"spectrum quality ({x})" for x in _SCREEN_REMOVE] if level != "none" else []
        if filters.stand_in_controls != "include":
            listed += ["stand-in control", "stand-in control (screened)",
                       "stand-in control (undeuterated)"]
        if filters.above_amides:
            listed.append("above the amide count")
        if filters.replicate_outlier is not None:
            listed.append("replicate outlier")
        listed += sorted(set(reasons) - set(listed))
        where = {"spectrum quality": f"spectrum_quality_filtering = \"{level}\"",
                 "stand-in control": f"stand_in_controls = \"{filters.stand_in_controls}\"",
                 "above the amide count": "above_amides = true",
                 "replicate outlier": f"replicate_outlier = {filters.replicate_outlier:g}"
                 if filters.replicate_outlier is not None else ""}
        for reason in listed:
            setting = next((v for k, v in where.items() if reason.startswith(k)), "")
            add(name, stage, f"left out: {reason}", np.sum(reasons == reason), "spectra",
                f"[analysis.filters] {setting}" if setting else "[analysis.filters]")
        if level != "none":
            gone = set(np.asarray(left_out["spectrum_id"]).astype(str)[mine]) if left_out else set()
            out_now = np.array([str(x) in gone for x in fit.spectrum_id], dtype=bool)
            add(name, stage, "kept, with a warning: picked intensity beyond the peptide's reach "
                "(screen_beyond_reach)", np.sum(screen.warn & keep & ~out_now), "spectra",
                f"[analysis.filters] spectrum_quality_filtering = \"{level}\"")
    if filters is not None and any(k >= 2 for k in ks):
        dropped: dict[str, int] = {}
        _records(results, ks, exclude_flagged, filters, quiet=True, controls_changes=controls,
                 counts=dropped)
        stage = "analysis: two-population descriptions (the spectrum stays, at one population)"
        level = filters.spectrum_quality_filtering
        for key, what, unit, setting in (
                ("one_population", "a separate cluster inside the peptide's range "
                 "(screen_between_separate)", "spectra",
                 f"spectrum_quality_filtering = \"{level}\""),
                ("population", f"a population below {filters.min_population:g} or above "
                 f"{1 - filters.min_population:g}", "spectra",
                 f"min_population = {filters.min_population:g}"),
                ("fD", f"a population above fD {filters.max_fD:g}", "spectra",
                 f"max_fD = {filters.max_fD:g}"),
                ("fD_se", f"an fD error of {filters.max_fD_se:g} or more", "peptides and times",
                 f"max_fD_se = {filters.max_fD_se:g}"),
                ("k1_between_ranks", "the one-population value outside the two populations",
                 "peptides and times", "k1_between_ranks = true"),
                ("review_one", "the review set one population", "spectra", "")):
            if key == "k1_between_ranks" and not filters.k1_between_ranks:
                continue
            add("", stage, what, dropped.get(key, 0), unit,
                f"[analysis.filters] {setting}" if setting else "review_K = 1")
    if filters is not None and population_test:
        from .set_test import above_max_fd

        stage = "analysis: population test (the spectrum stays, at one population)"
        for name, r in runs:
            if getattr(r, "population_test", None) is None:
                continue
            held = (above_max_fd(r, filters, controls)[:, 0] & _kept(r, exclude_flagged)
                    & (np.asarray(r.review_status) != "accepted"))
            add(name, stage, f"held out of the test: a population above fD {filters.max_fD:g} "
                "in the two-population fit", np.sum(held), "spectra",
                f"[analysis.filters] max_fD = {filters.max_fD:g}")
    names = ("run", "stage", "what", "count", "unit", "setting")
    return {n: np.array([row[i] for row in rows],
                        dtype=np.int64 if n == "count" else object).astype(
                            np.int64 if n == "count" else str)
            for i, n in enumerate(names)}


def _left_out(results, filters: SpeciesFilters, controls, n_amides: np.ndarray,
              keep: np.ndarray, values: bool = False, screen: Screening | None = None,
              unscreened=None):
    """Which rows of the fit :class:`SpeciesFilters` leaves out whole, and why (the first
    reason that applies: the spectrum quality screens (``screen``, :func:`screening`), the
    stand-in control, then the amide count, then the replicate outlier); with ``values`` also
    each row's K=1 fD. ``unscreened`` (the controls before the screens left any out) tells a
    stand-in the screens made from one there was already."""
    fit, binned = results.fit, results.binned
    n = len(fit.spectrum_id)
    out = np.zeros(n, dtype=bool)
    why = np.full(n, "", dtype=object)
    accepted = np.asarray(results.review_status) == "accepted"
    index = {str(s): j for j, s in enumerate(binned.spectrum_id)}
    rows = np.array([index[str(s)] for s in fit.spectrum_id], dtype=np.int64)
    sample = np.asarray(binned.sample)[rows].astype(str)

    def mark(mask: np.ndarray, reason: str) -> None:
        new = mask & keep & ~out & ~accepted
        out[new] = True
        why[new] = reason

    screened = np.zeros(n, dtype=bool)
    if screen is not None:
        screened = screen.remove & keep & ~accepted
        out |= screened
        why[screened] = screen.reason[screened]
    raw = np.zeros(n, dtype=bool)
    if controls is not None:
        at = controls.rows(fit.data_id)
        raw = np.asarray(controls.uncorrected, dtype=bool)[at]
        n_amides = np.asarray(controls.n_amides, dtype=float)[at]
        if filters.stand_in_controls != "include":
            measured = np.asarray(controls.fd_measured, dtype=bool)[at]
            if filters.stand_in_controls == "other_sample":  # as close as the charge states
                measured |= np.char.find(np.asarray(controls.flags, dtype=str)[at],
                                         "fd_other_sample") >= 0
            stand_in = ~measured & ~raw
            # a sample without any measured fully deuterated control keeps its stand-ins
            has = {s for s in np.unique(sample) if (measured & ~raw & (sample == s)).any()}
            lost = np.zeros(n, dtype=bool)
            if unscreened is not None and unscreened is not controls:
                lost = np.asarray(unscreened.fd_measured, dtype=bool)[
                    unscreened.rows(fit.data_id)] & stand_in
            mark(lost & np.isin(sample, list(has)), "stand-in control (screened)")
            mark(stand_in & np.isin(sample, list(has)), "stand-in control")
            # an undeuterated control above m0_max holds another species: no sample exception
            heavy = np.char.find(np.asarray(controls.flags, dtype=str)[at], "m0_high") >= 0
            heavy &= ~raw
            mark(heavy, "stand-in control (undeuterated)")
            data_id = np.asarray(fit.data_id)
            for s in (() if values else np.unique(sample)):  # once per analysis
                mine = sample == s
                n_series = len(np.unique(data_id[mine]))
                n_heavy = len(np.unique(data_id[mine & heavy]))
                if n_series and 2 * n_heavy >= n_series:
                    log.warning("sample %s: %d of %d series have an undeuterated control above "
                                "m0_max and are left out (stand_in_controls = \"%s\"); "
                                "is a labelled file marked as undeuterated?", s, n_heavy,
                                n_series, filters.stand_in_controls)
    k1 = fit.fits[1].means[:, 0] if 1 in fit.fits else np.full(n, np.nan)
    if filters.above_amides:
        mark(np.isfinite(k1) & (k1 > n_amides + 1e-9) & ~raw, "above the amide count")
    fd = k1.copy()
    if controls is not None:
        fd = correct_means(controls, fit.data_id, k1[:, None]).mean[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        fd = fd / n_amides
    if filters.replicate_outlier is not None:
        role = _analysis_roles(fit, controls)
        time = np.asarray(fit.time, dtype=float)
        at_time = (role != "undeut") & (role != "fulldeut") & np.isfinite(time)
        usable = at_time & keep & ~screened & np.isfinite(fd)
        groups: dict[tuple, list[int]] = {}
        data_id = np.asarray(fit.data_id)
        for i in np.flatnonzero(usable):  # the replicates of one series (peptide and charge)
            groups.setdefault((int(data_id[i]), float(time[i])), []).append(i)
        scaled = [float(np.var(fd[g], ddof=1)) * (len(g) - 1) / stats.chi2.median(len(g) - 1)
                  for g in groups.values() if len(g) >= 2]
        if scaled:
            limit = filters.replicate_outlier * float(np.sqrt(np.median(scaled)))
            far = np.zeros(n, dtype=bool)
            for g in groups.values():
                if len(g) < 3:
                    continue  # two replicates that disagree: the data cannot say which is off
                # from the median of all of them: with three or more, one outlier cannot move
                # it far (the median of the others would be pulled halfway to it)
                centre = float(np.median(fd[g]))
                for i in g:
                    far[i] = abs(fd[i] - centre) > limit
            mark(far, "replicate outlier")
    if values:
        return out, why, fd
    return out, why


def _spectrum_records(results, ks, exclude_flagged, filters, quiet=False,
                      controls_changes=None, evidence=None):
    """:func:`_records` of one run with only the per-spectrum filters applied, and the counts
    of what they dropped (the per-species filters, :func:`_filter_species`, come after merged
    samples are pooled). ``evidence`` (a dict, keyed as the groups without K and rank)
    receives each set's summed criterion over the same spectra (:func:`_add_evidence`)."""
    fit, binned = results.fit, results.binned
    if binned is None:
        raise ValueError("the species uptake table needs the picked table (results.binned)")
    missing = [k for k in ks if k not in fit.fits]
    if missing:
        raise ValueError(f"K = {missing} were not fitted (fitted: {list(fit.k_values)})")
    picker = results.settings.picker if results.settings is not None else None
    nterm = picker.nterm_subtract if picker is not None else 1
    index = {str(s): j for j, s in enumerate(binned.spectrum_id)}
    rows = np.array([index[str(s)] for s in fit.spectrum_id])
    keep = _kept(results, exclude_flagged)
    screen = (screening(results, filters.spectrum_quality_filtering) if filters is not None
              else Screening.nothing(len(fit)))
    controls, unscreened = _screened_controls(results, controls_changes, screen, keep, quiet)
    n_amides = _site_counts(results)
    if filters is not None:
        out, why = _left_out(results, filters, controls, n_amides, keep, screen=screen,
                             unscreened=unscreened)
        warned = screen.warn & keep & ~out
        if warned.any() and not quiet:
            log.warning("%d spectra have picked intensity beyond the range the peptide can reach "
                        "(screen_beyond_reach in spectra.csv): kept; worth a look",
                        int(warned.sum()))
        keep &= ~out
        if out.any() and not quiet:
            counts: dict[str, int] = {}
            for w in why[out]:
                counts[str(w)] = counts.get(str(w), 0) + 1
            log.info("spectra left out by [analysis.filters]: %s",
                     "; ".join(f"{n} {w}" for w, n in sorted(counts.items())))
    fell_back = np.zeros(len(fit), dtype=bool)
    not_measured = np.zeros(len(fit), dtype=bool)
    raw = np.zeros(len(fit), dtype=bool)
    if controls is not None:
        at = controls.rows(fit.data_id)
        raw = controls.uncorrected[at]
        n_amides = np.asarray(controls.n_amides, dtype=float)[at]
        fell_back = np.asarray(controls.fallback, dtype=bool)[at]
        not_measured = ~controls.measured[at]

    # one record per spectrum and rank, then grouped by species, point (a time, or a control
    # role), K and rank
    points = _points(fit, _analysis_roles(fit, controls))
    groups: dict[tuple, list[tuple[float, float, float, float, int]]] = {}
    dropped = {"population": 0, "fD": 0, "fD_se": 0, "k1_between_ranks": 0,
               "one_population": 0, "review_one": 0}
    reviewed_one = review_one(results)
    described = {}  # K -> the spectra whose K description the table keeps
    for k in ks:
        kf = fit.fits[k]
        order = np.argsort(kf.means, axis=1)
        means = np.take_along_axis(kf.means, order, axis=1)
        fracs = np.take_along_axis(kf.populations, order, axis=1)
        mean_se = np.full_like(means, np.nan)
        frac_se = np.full_like(means, np.nan)
        for i in range(len(fit)):
            boot = results.bootstrap_for(i, k)
            if boot is not None:
                mean_se[i] = boot.mean_se[i, :k][order[i]]
                frac_se[i] = boot.population_se[i, :k][order[i]]
        fitted = fit.fitted(k)
        if not quiet and (keep & ~fitted).any():
            log.info("K=%d was not fitted for %d spectra (the screen showed it cannot be "
                     "selected): they have no K=%d description", k, int((keep & ~fitted).sum()), k)
        no_boot = int(np.isnan(mean_se[keep & fitted, 0]).sum())
        if k >= 2 and no_boot and not quiet:
            log.warning("K=%d: %d of %d spectra have no bootstrap, so their standard errors come "
                        "only from the scatter of replicates (or are unknown): too small for the "
                        "choice between one and two populations (bootstrap at K=1 and K=2: "
                        "[run.bootstrap] ks = [1, 2])", k, no_boot, int((keep & fitted).sum()))
        if controls is not None:
            c = correct_means(controls, fit.data_id, means, se=mean_se)
            means, mean_se = c.mean, c.se
        usable = keep & fitted
        if k >= 2:  # the review's one population: the K = 1 description only
            one = reviewed_one & usable
            dropped["review_one"] += int(one.sum())
            usable &= ~one
        if filters is not None and k >= 2:
            one = screen.one_population & usable
            dropped["one_population"] += int(one.sum())
            usable &= ~one
            lo = filters.min_population
            bad_pop = np.any((fracs < lo) | (fracs > 1 - lo), axis=1) & usable
            bad_fd = np.any(means / n_amides[:, None] > filters.max_fD, axis=1) & usable
            dropped["population"] += int(bad_pop.sum())
            dropped["fD"] += int((bad_fd & ~bad_pop).sum())
            usable &= ~(bad_pop | bad_fd)
        described[k] = usable
        for i in np.flatnonzero(usable):
            j = rows[i]
            key = ("", str(binned.sample[j]), str(binned.peptide[j]),
                   str(binned.modification[j]), int(binned.start_seq[j]),
                   int(binned.end_seq[j]), points[i], k)
            for r in range(k):
                groups.setdefault(key + (r + 1,), []).append((
                    means[i, r] / n_amides[i], mean_se[i, r] / n_amides[i], fracs[i, r],
                    frac_se[i, r], int(binned.charge[j]), float(n_amides[i]),
                    int(binned.rep[j]), bool(fell_back[i]), "", bool(not_measured[i]),
                    bool(raw[i]),
                ))

    if evidence is not None:
        _add_evidence(evidence, results, ks, keep, described, rows, points)
    corrected = {("", str(s)): controls is not None for s in np.unique(binned.sample)}
    return groups, corrected, int(nterm), dropped


class Point(tuple):
    """A measurement point as a grouping key: ``(time, role)`` with NaN as None, so that the
    controls without a time group by role (``('undeut',)``-like keys compare equal, which NaN
    floats do not). Sorts the undeuterated control first, times in order, the fully deuterated
    control last."""

    __slots__ = ()

    @property
    def time(self) -> float:
        return float("nan") if self[0] is None else float(self[0])

    @property
    def role(self) -> str:
        return str(self[1])

    def __lt__(self, other):
        return point_order_key(self.time, self.role) < point_order_key(other.time, other.role)


def _analysis_roles(fit, controls) -> np.ndarray:
    """The control role of each spectrum as the analysis uses it: the fit's, with the time
    points ``[analysis.controls] undeut_at``/``fulldeut_at`` promote (``controls``: the
    analysis's control table), so a promoted point leaves the labelled time points too."""
    role = np.asarray(fit.role, dtype=str)
    config = getattr(controls, "config", None)
    if config is not None and (config.undeut_at is not None or config.fulldeut_at is not None):
        role = apply_control_roles(role, fit.time, config, where="[analysis.controls]")
    return role


def _points(fit, role=None) -> list[Point]:
    time = np.asarray(fit.time, dtype=float)
    role = np.asarray(fit.role if role is None else role, dtype=str)
    return [Point((None if not np.isfinite(t) else float(t), str(r)))
            for t, r in zip(time, role, strict=True)]


def _filter_species(groups: dict, ks, filters, dropped: dict, quiet: bool = False) -> dict:
    """The per-species filters of :class:`SpeciesFilters` (``max_fD_se``,
    ``k1_between_ranks``) on grouped records, once per species and time, after merged samples
    are pooled; ``dropped`` (the per-spectrum counts) is completed and logged."""
    if filters is not None:  # a K >= 2 description whose fD error is too large is not kept
        worst: dict[tuple, float] = {}
        for key, recs in groups.items():
            if key[7] >= 2:
                _, se = _combine_replicates(recs)
                worst[key[:8]] = max(worst.get(key[:8], 0.0), se if np.isfinite(se) else 0.0)
        too_wide = {key for key, se in worst.items() if se >= filters.max_fD_se}
        dropped["fD_se"] = len(too_wide)
        groups = {key: recs for key, recs in groups.items() if key[:8] not in too_wide}
        if filters.k1_between_ranks and 1 in ks:  # the K=1 value between the K >= 2 ranks
            value = {key: _combine_replicates(recs)[0] for key, recs in groups.items()}
            unordered = set()
            for key in value:
                base, k, rank = key[:7], key[7], key[8]
                if k < 2 or rank != 1 or base + (1, 1) not in value or base + (k, k) not in value:
                    continue
                if not value[key] < value[base + (1, 1)] < value[base + (k, k)]:
                    unordered.add(base + (k,))
            dropped["k1_between_ranks"] = len(unordered)
            groups = {key: recs for key, recs in groups.items() if key[:8] not in unordered}
        if any(dropped.values()) and not quiet:
            reasons = {
                "population": (f"{dropped['population']} spectra with a population outside "
                               f"[{filters.min_population:g}, "
                               f"{1 - filters.min_population:g}] (min_population)"),
                "fD": f"{dropped['fD']} spectra with a component above fD {filters.max_fD:g} "
                      "(max_fD)",
                "fD_se": (f"{dropped['fD_se']} species and times with an fD error of "
                          f"{filters.max_fD_se:g} or more (max_fD_se)"),
                "k1_between_ranks": (f"{dropped['k1_between_ranks']} with the K=1 value "
                                     "outside the ranks (k1_between_ranks)"),
                "one_population": (f"{dropped['one_population']} spectra kept at one population "
                                   "(screen_between_separate, spectrum_quality_filtering)"),
                "review_one": (f"{dropped['review_one']} spectra the review set to one "
                               "population (review_K = 1)"),
            }
            log.info("K >= 2 descriptions left out by [analysis.filters]: %s",
                     "; ".join(text for key, text in reasons.items() if dropped.get(key)))
    return groups


def species_uptake(
    results,
    ks: Sequence[int] = (1, 2),
    *,
    exclude_flagged: bool = False,
    filters: SpeciesFilters | None = SpeciesFilters(),  # noqa: B008
    controls: Mapping[str, Any] | None = None,
    min_population: float = 0.05,
    populations: str | None = None,
) -> SpeciesUptake:
    """The species uptake table of ``results`` (:class:`hxex.report.Results`, or a
    :class:`hxex.analysis.RunSet` of several runs) at each fixed K in ``ks`` (K values the run
    fitted). ``filters`` (:class:`SpeciesFilters`; None for none) decides
    which K >= 2 descriptions are kept; a species and time whose description is not kept has no
    rows at that K. The back-exchange correction uses :func:`analysis_controls` (``controls``:
    control settings to change).

    ``charge_trend`` is True for every row of a species (sample, peptide and modification, K and
    rank) whose lowest and highest charge states, wherever both were measured, differ by more
    than ``TREND_Z`` combined standard errors in the same direction at more than ``TREND_SHARE``
    of those times, and at two times at least: more likely a systematic effect (an overlapping
    species, a wrong charge assignment) than noise.

    ``run_offset`` (merged runs, :class:`hxex.analysis.RunSet` ``merge``) is True for every row
    of a species whose spectra come from several runs and one of whose runs differs from the
    first by more than ``TREND_Z`` combined standard errors in the same direction at more than
    ``TREND_SHARE`` of the times both were measured, and at two times at least (the charge-trend
    rule): a difference between runs (a batch effect) that pooling would otherwise hide.

    ``best_set_K`` (with ``best_set``, ``best_set_reason`` and ``set_delta_criterion``) is the
    number of populations chosen for each species and time from all of its spectra together
    (:func:`best_set`; ``min_population``: the population limit of that choice, the
    ``[analysis.choice]`` one); with ``populations = 'population_test'`` they come from the
    population test of the sets instead (:func:`hxex.analysis.set_test.set_tests`), and
    ``set_p_1`` and ``set_p_2`` are its p-values. ``populations`` None takes the rule of the
    results, as ``hxex analyse`` does (:func:`hxex.analysis.settings.population_rule`: the
    population test when the run has it), and the table records the rule.
    """
    from .settings import population_rule

    populations = population_rule(populations, results)
    evidence: dict[tuple, dict] = {}
    groups, corrected, nterm = _records(results, ks, exclude_flagged, filters,
                                        controls_changes=controls, evidence=evidence)
    cols: dict[str, list] = {f.name: [] for f in fields(SpeciesUptake)
                             if f.name not in _SCALARS
                             and f.name not in ("set_p_1", "set_p_2", "set_test_K",
                                                "undeuterated_two")}
    by_charge: dict[tuple, dict[int, list[tuple[float, float]]]] = {}
    by_run: dict[tuple, list[dict]] = {}
    for key, recs in sorted(groups.items()):
        run, sample, peptide, mod, start, end, point, k, rank = key
        time, role = point.time, point.role
        d, d_se = _combine_replicates(recs, 0)
        fr, fr_se = _combine_replicates(recs, 2)
        charges = sorted({r[4] for r in recs})
        amides = float(np.mean([r[5] for r in recs]))  # the same for every spectrum of a species
        run_names = [x for x in dict.fromkeys(r[8] for r in recs) if x] or [""]  # RunSet order
        ion = chem.parse_modification(mod).ion_type
        for name, value in (
            ("run", run), ("sample", sample), ("peptide", peptide), ("modification", mod),
            ("kind", "peptide" if ion == "M" else "fragment"), ("start", start), ("end", end),
            ("time", time), ("role", role), ("K", k), ("rank", rank), ("mean", d * amides),
            ("mean_se", d_se * amides), ("fD", d), ("fD_se", d_se),
            ("population", fr), ("population_se", fr_se), ("n_amides", amides),
            ("n_spectra", len(recs)), ("n_charges", len(charges)),
            ("corrected", corrected[(run, sample)] and not any(r[10] for r in recs)),
            ("charge_trend", False),
            ("control_fallback", any(r[7] for r in recs)),
            ("runs", ",".join(run_names)),
            ("n_spectra_by_run", ",".join(str(sum(r[8] == x for r in recs)) for x in run_names)
             if any(run_names) else ""),
            ("run_offset", False),
            ("fd_other_charge", any(r[7] for r in recs) and not any(r[9] for r in recs)),
        ):
            cols[name].append(value)
        if len(run_names) > 1:  # each run against the first, at this time
            per_run = {x: _combine_replicates([r for r in recs if r[8] == x])
                       for x in run_names}
            by_run.setdefault((run, sample, peptide, mod, k, rank), []).append(per_run)
        if len(charges) > 1:
            per = by_charge.setdefault((run, sample, peptide, mod, k, rank), {})
            low = [(r[0], r[1]) for r in recs if r[4] == charges[0]]
            high = [(r[0], r[1]) for r in recs if r[4] == charges[-1]]
            a, b = _combine(*map(np.array, zip(*low, strict=True))), _combine(
                *map(np.array, zip(*high, strict=True)))
            per[len(per)] = [a, b]
    trend = {key for key, times in by_charge.items() if _is_trend(list(times.values()))}
    out = {name: np.array(values) for name, values in cols.items()}
    species = list(zip(out["run"], out["sample"], out["peptide"], out["modification"], out["K"],
                       out["rank"], strict=True))
    out["charge_trend"] = np.array([(str(a), str(b), str(c), str(d), int(e), int(f)) in trend
                                    for a, b, c, d, e, f in species], dtype=bool)
    offset = set()
    for key, times in by_run.items():
        first = next(iter(times[0]))  # the first run is the same at every time if present
        others = {x for per in times for x in per if x != first}
        for other in others:
            pairs = [[per[first], per[other]] for per in times if first in per and other in per]
            if _is_trend(pairs):
                offset.add(key)
    out["run_offset"] = np.array([(str(a), str(b), str(c), str(d), int(e), int(f)) in offset
                                  for a, b, c, d, e, f in species], dtype=bool)
    if offset:
        log.warning("%d species merged by [analysis.merge] differ systematically between runs "
                    "(run_offset in the species table)",
                    len({key[:4] for key in offset}))
    for name in ("start", "end", "K", "rank", "n_spectra", "n_charges"):
        out[name] = out[name].astype(np.int64)
    for name in ("time", "mean", "mean_se", "fD", "fD_se", "population", "population_se",
                 "n_amides"):
        out[name] = out[name].astype(np.float64)
    out["corrected"] = out["corrected"].astype(bool)
    out["control_fallback"] = out["control_fallback"].astype(bool)
    out["fd_other_charge"] = out["fd_other_charge"].astype(bool)
    for name in ("runs", "n_spectra_by_run", "role"):
        out[name] = out[name].astype(str)
    out["run"] = out["run"].astype(object)
    rts = _species_rt(results, exclude_flagged)
    out["rt"] = np.array([rts.get((str(a), str(b), str(c), str(d)), np.nan) for a, b, c, d in
                          zip(out["run"], out["sample"], out["peptide"], out["modification"],
                              strict=True)], dtype=np.float64)
    out.update(best_set(out, evidence, ks, min_population))
    if populations == "population_test":
        from .set_test import set_tests

        tests = set_tests(results, filters=filters, exclude_flagged=exclude_flagged,
                          controls=controls,
                          undeuterated_cap=filters is not None and filters.undeuterated_cap)
        out.update(test_set(out, tests))
    return SpeciesUptake(**out, nterm_subtract=nterm, populations=populations)


def _set_key(run, sample, peptide, mod, start, end, time, role) -> tuple:
    t = float(time)
    return (str(run), str(sample), str(peptide), str(mod), int(start), int(end),
            Point((None if not np.isfinite(t) else t, str(role))))


def review_one(results) -> np.ndarray:
    """The spectra the review set to one population (``review_K = 1``, current entries): the
    analysis keeps their K = 1 description only, as the run and the population test do."""
    out = np.zeros(len(results.fit), dtype=bool)
    review = results.review
    if review is None or not len(review):
        return out
    index = {str(s): i for i, s in enumerate(results.fit.spectrum_id)}
    for sid, k, stale in zip(review.spectrum_id, review.k_override, review.stale, strict=True):
        if k == 1 and not stale and str(sid) in index:
            out[index[str(sid)]] = True
    return out


def _add_evidence(evidence: dict, results, ks, keep: np.ndarray,
                  described: Mapping[int, np.ndarray], rows: np.ndarray, points) -> None:
    """Add to ``evidence`` ((run, sample, peptide, modification, start, end, point) -> the
    criterion of each K of ``ks`` summed over the set's spectra, with ``n``, the number of
    spectra summed) the spectra of one run the species table uses (``keep``: not rejected and
    not left out whole by the filters). A spectrum whose K description the table does not keep
    (``described``: not fitted, left out by the K >= 2 filters, kept at one population by the
    screens or the review) adds its criterion at the K below instead, so it cannot vote for
    more populations than the table gives it. The fits' own criterion (BIC or AIC,
    ``[run.fit] criterion``) is used, not the run's selection score, which the run's population
    rule sets to infinity: one such spectrum would otherwise decide the whole set."""
    fit, b = results.fit, results.binned
    criterion = fit.config.criterion
    order = sorted(ks)
    values = {k: np.asarray(getattr(fit.fits[k], criterion), dtype=float) for k in order}
    used: dict[int, np.ndarray] = {}
    usable = keep & described[order[0]] & np.isfinite(values[order[0]])
    for n, k in enumerate(order):
        own = described[k] & np.isfinite(values[k])
        used[k] = values[k] if n == 0 else np.where(own, values[k], used[order[n - 1]])
    for i in np.flatnonzero(usable):
        j = rows[i]
        key = ("", str(b.sample[j]), str(b.peptide[j]), str(b.modification[j]),
               int(b.start_seq[j]), int(b.end_seq[j]), points[i])
        entry = evidence.setdefault(key, {"n": 0, **{k: 0.0 for k in order}})
        entry["n"] += 1
        for k in order:
            entry[k] += float(used[k][i])


def best_sets(uptake: SpeciesUptake) -> dict[tuple, tuple[int, float]]:
    """``(run, sample, peptide, modification, start, end, time) -> (best_set_K,
    set_delta_criterion)`` for every species and time point of a species table (the controls
    left out): the number of populations each set of spectra supports, and the evidence
    (summed criterion at K = 2 minus K = 1; negative favours two)."""
    out: dict[tuple, tuple[int, float]] = {}
    role = np.asarray(uptake.role, dtype=str)
    for i in np.flatnonzero(role == ""):
        key = (str(uptake.run[i]), str(uptake.sample[i]), str(uptake.peptide[i]),
               str(uptake.modification[i]), int(uptake.start[i]), int(uptake.end[i]),
               float(uptake.time[i]))
        out.setdefault(key, (int(uptake.best_set_K[i]), float(uptake.set_delta_criterion[i])))
    return out


def set_pvalues(uptake: SpeciesUptake) -> dict[tuple, tuple[float, float]]:
    """``(run, sample, peptide, modification, start, end, time) -> (set_p_1, set_p_2)`` for
    every species and time point of a species table made with the population test (the
    controls left out; the keys of :func:`best_sets`)."""
    out: dict[tuple, tuple[float, float]] = {}
    role = np.asarray(uptake.role, dtype=str)
    for i in np.flatnonzero(role == ""):
        key = (str(uptake.run[i]), str(uptake.sample[i]), str(uptake.peptide[i]),
               str(uptake.modification[i]), int(uptake.start[i]), int(uptake.end[i]),
               float(uptake.time[i]))
        out.setdefault(key, (float(uptake.set_p_1[i]), float(uptake.set_p_2[i])))
    return out


def best_set(cols: Mapping[str, np.ndarray], evidence: Mapping[tuple, dict], ks,
             min_population: float = 0.05) -> dict[str, np.ndarray]:
    """The columns ``best_set_K``, ``best_set``, ``best_set_reason`` and
    ``set_delta_criterion`` of a species table (``cols``): for each species and time, the
    number of populations its whole set of spectra supports. Replicates and charge states are
    spectra of the same molecules, so their evidence adds up: the criterion of each K summed
    over the set (``evidence``, :func:`_add_evidence`) chooses K, the smallest sum winning. A K
    is passed over when the table has no description at it (the filters left it out) or one
    of its combined populations lies outside ``[min_population, 1 - min_population]``; the
    next best K is taken. ``set_delta_criterion`` is the summed criterion at K = 2 minus K = 1
    (negative: the set favours two populations). ``best_set`` marks the rows of the chosen K."""
    n = len(cols["K"])
    out = {"best_set_K": np.zeros(n, dtype=np.int64), "best_set": np.zeros(n, dtype=bool),
           "best_set_reason": np.full(n, "", dtype=object),
           "set_delta_criterion": np.full(n, np.nan)}
    rows: dict[tuple, list[int]] = {}
    for i in range(n):
        key = _set_key(cols["run"][i], cols["sample"][i], cols["peptide"][i],
                       cols["modification"][i], cols["start"][i], cols["end"][i],
                       cols["time"][i], cols["role"][i])
        rows.setdefault(key, []).append(i)
    lo = float(min_population)
    for key, idx in rows.items():
        by_k: dict[int, list[int]] = {}
        for i in idx:
            by_k.setdefault(int(cols["K"][i]), []).append(i)
        ev = evidence.get(key)
        if ev is not None and 1 in ev and 2 in ev:
            out["set_delta_criterion"][idx] = ev[2] - ev[1]
        if len([k for k in ks if k >= 1]) < 2 or max(ks) < 2:
            chosen, reason = (1 if 1 in by_k else min(by_k)), "only one K analysed"
        elif ev is None:
            chosen, reason = (1 if 1 in by_k else min(by_k)), "no spectrum fitted at every K"
        else:
            order = sorted((k for k in ks if k in ev), key=lambda k: (ev[k], k))
            chosen, reason, passed = None, "", []
            for k in order:
                if k not in by_k:
                    passed.append(f"K = {k} not kept by the filters")
                    continue
                pops = [float(cols["population"][i]) for i in by_k[k]]
                if k >= 2 and any(p < lo or p > 1 - lo for p in pops):
                    passed.append(f"K = {k} has a population below {lo:g}")
                    continue
                chosen = k
                break
            words = {1: "one", 2: "two", 3: "three"}
            if chosen is None:
                chosen, reason = min(by_k), "; ".join(passed)
            elif passed:  # the set favoured a K that could not be used
                best = words.get(order[0], str(order[0]))
                reason = (f"set evidence for {best}, but {'; '.join(passed)}: "
                          f"{words.get(chosen, str(chosen))} used")
            else:
                reason = f"set evidence for {words.get(chosen, str(chosen))}"
        out["best_set_K"][idx] = chosen
        out["best_set_reason"][idx] = reason
        out["best_set"][by_k.get(chosen, [])] = True
    return out


def test_set(cols: Mapping[str, np.ndarray], tests: Mapping[tuple, Any]
             ) -> dict[str, np.ndarray]:
    """The columns ``best_set_K``, ``best_set``, ``best_set_reason``, ``set_p_1``, ``set_p_2``,
    ``set_test_K`` and ``undeuterated_two`` of a species table (``cols``) from the population
    test of its sets (``tests``, :func:`hxex.analysis.set_test.set_tests`). Where the table has
    no description at the test's K (not analysed, or left out by the filters), the largest K
    below it that it has is used, and the reason says so."""
    n = len(cols["K"])
    out = {"best_set_K": np.zeros(n, dtype=np.int64), "best_set": np.zeros(n, dtype=bool),
           "best_set_reason": np.full(n, "", dtype=object), "set_p_1": np.full(n, np.nan),
           "set_p_2": np.full(n, np.nan), "set_test_K": np.ones(n, dtype=np.int64),
           "undeuterated_two": np.zeros(n, dtype=bool)}
    rows: dict[tuple, list[int]] = {}
    for i in range(n):
        key = _set_key(cols["run"][i], cols["sample"][i], cols["peptide"][i],
                       cols["modification"][i], cols["start"][i], cols["end"][i],
                       cols["time"][i], cols["role"][i])
        rows.setdefault(key, []).append(i)
    words = {1: "one", 2: "two", 3: "three"}
    for key, idx in rows.items():
        by_k: dict[int, list[int]] = {}
        for i in idx:
            by_k.setdefault(int(cols["K"][i]), []).append(i)
        t = tests.get(key)
        if t is None:  # no spectrum of the set kept for the test (the filters left them out)
            chosen, reason = (1 if 1 in by_k else min(by_k)), "not tested: one population"
        else:
            for s, p in enumerate(t.p[:2]):
                out[f"set_p_{s + 1}"][idx] = p
            out["set_test_K"][idx] = t.K
            out["undeuterated_two"][idx] = t.undeuterated_two
            below = [k for k in by_k if k <= t.K]
            chosen = max(below) if below else min(by_k)
            reason = t.reason
            if chosen != t.K:
                reason += (f"; K = {t.K} not kept by the filters or not analysed: "
                           f"{words.get(chosen, str(chosen))} used")
        out["best_set_K"][idx] = chosen
        out["best_set_reason"][idx] = reason
        out["best_set"][by_k.get(chosen, [])] = True
    return out


def _species_rt(results, exclude_flagged: bool = False
                ) -> dict[tuple[str, str, str, str], float]:
    """(run, sample, peptide, modification) -> the median retention time (minutes) of its
    spectra (those the tables use, :func:`_kept`; run and sample as in the table of a
    :class:`hxex.analysis.RunSet`); species without one are absent."""
    from .runs import RunSet

    runs = results.runs.items() if isinstance(results, RunSet) else [("", results)]
    values: dict[tuple[str, str, str, str], list[float]] = {}
    for name, r in runs:
        b = r.binned
        if b is None or getattr(b, "rt", None) is None:
            continue
        index = {str(s): j for j, s in enumerate(b.spectrum_id)}
        used = np.zeros(len(b.spectrum_id), dtype=bool)
        used[[index[str(s)] for s in r.fit.spectrum_id[_kept(r, exclude_flagged)]]] = True
        for i in np.flatnonzero(np.isfinite(b.rt) & used):
            run, sample = results.state(name, str(b.sample[i])) if name else ("", str(b.sample[i]))
            values.setdefault((run, sample, str(b.peptide[i]), str(b.modification[i])),
                              []).append(float(b.rt[i]))
    return {key: float(np.median(v)) for key, v in values.items()}


def _is_trend(pairs: list) -> bool:
    """Whether the (lowest charge, highest charge) values at each time differ systematically."""
    diffs = []
    for (a, sa), (b, sb) in pairs:
        se = np.hypot(sa, sb) if np.isfinite(sa) and np.isfinite(sb) else np.nan
        if np.isfinite(se) and se > 0:
            diffs.append((b - a) / se)
    if len(diffs) < 2:
        return False
    diffs = np.array(diffs)
    up, down = int(np.sum(diffs > TREND_Z)), int(np.sum(diffs < -TREND_Z))
    hits = max(up, down)
    return hits >= 2 and hits > TREND_SHARE * len(diffs)
