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
controls (``corrected``), and ``fD``, the **fractional deuteration**, ``mean / n_amides`` (0 to
1). Both mean the same for forward labelling and for reverse exchange (docs/ANALYSIS.md).

Spectra the review rejected are left out; flagged ones are kept unless ``exclude_flagged``. The
replicates and charge states of a species at one time are combined: the value is their mean and
its standard error ``sqrt(max(mean(se_i^2), s^2) / n)``, the larger of the measurements' own
(bootstrap) variance and their observed scatter ``s^2``, so that neither is counted twice; with one
measurement it is that measurement's standard error. ``charge_trend`` flags a species whose charge
states disagree systematically (see :func:`species_uptake`).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from .. import chem
from ..controls import ControlConfig, correct_means, fit_controls

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeciesFilters:
    """Which K >= 2 descriptions the analysis trusts (K = 1 is never filtered), following the
    owner's pyHXExpress practice (docs/ANALYSIS.md).

    Per spectrum, before replicates and charge states are combined: every population within
    ``[min_population, 1 - min_population]``, and no component deuterated beyond ``max_fD``
    (its ``n * p`` over the exchangeable amides, back-exchange corrected when the run has
    controls, as reported: more than full deuteration is not physical; the fit itself allows
    ``n`` up to the run's ``n_max_factor`` times the amides). Per species and time, after
    combining: every rank's ``fD`` standard error below ``max_fD_se`` (unknown errors pass),
    and, with ``k1_between_ranks``, the K = 1 value between the lowest and the highest rank
    (``fD_low < fD_avg < fD_high``; needs K = 1 among the K analysed): a split whose average
    does not lie between its populations does not describe the same molecules."""

    min_population: float = 0.05
    max_fD_se: float = 0.25
    max_fD: float = 1.1
    k1_between_ranks: bool = True

    def __post_init__(self):
        if not 0 <= self.min_population < 0.5:
            raise ValueError("min_population must be in [0, 0.5)")
        if self.max_fD_se <= 0 or self.max_fD <= 0:
            raise ValueError("max_fD_se and max_fD must be positive")


# a charge trend: the lowest and highest charge differ by more than this many combined standard
# errors, in the same direction, at this share of the times both were measured (and at least two)
TREND_Z = 2.0
TREND_SHARE = 0.5


_SCALARS = ("nterm_subtract", "undeut_time")  # settings of the table, not columns


@dataclass(frozen=True)
class SpeciesUptake:
    """One row per sample, species, time, ``K`` and ``rank`` (columns of equal length).

    Identity: ``sample``, ``peptide`` (modX), ``modification``, ``kind`` (``'peptide'``, or
    ``'fragment'`` for an ion type other than ``M``), ``start`` and ``end`` (residues on the
    protein, inclusive). Values: ``time``, ``K``, ``rank``, ``mean`` and ``mean_se`` (deuterons),
    ``fD`` and ``fD_se`` (``mean / n_amides``),
    ``population`` and ``population_se`` (the component's share), ``n_amides`` (exchangeable
    amides),
    ``n_spectra``, ``n_charges``, ``corrected`` and ``charge_trend``; ``control_fallback``: the
    back-exchange correction of at least one of its spectra used a fallback control (a missing
    or rejected control, :mod:`hxex.controls`). From a :class:`hxex.analysis.RunSet`: ``runs``
    (the runs its spectra come from, e.g. ``'A,B'`` for a merged sample; '' for one run),
    ``n_spectra_by_run`` (``'4,4'``) and ``run_offset`` (a merged species whose runs disagree
    systematically; see :func:`species_uptake`). ``nterm_subtract`` (the
    N-terminal residues that do not count) and ``undeut_time`` (the undeuterated control's time,
    which coverage, redundancy and resolution refer to) are the run's.
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
    nterm_subtract: int = 1
    undeut_time: float = 0.0

    def __post_init__(self):  # built without them: no fallback, one run
        n = len(self.sample)
        for name, empty in (("control_fallback", np.zeros(n, dtype=bool)),
                            ("runs", np.full(n, "", dtype=object)),
                            ("n_spectra_by_run", np.full(n, "", dtype=object)),
                            ("run_offset", np.zeros(n, dtype=bool))):
            if getattr(self, name) is None:
                object.__setattr__(self, name, empty)

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        """The table as columns (for :func:`hxex.report.write_csv`)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in _SCALARS}

    def select(self, mask) -> SpeciesUptake:
        mask = np.asarray(mask)
        return SpeciesUptake(**{f.name: getattr(self, f.name)[mask] for f in fields(self)
                                if f.name not in _SCALARS},
                             nterm_subtract=self.nterm_subtract, undeut_time=self.undeut_time)

    def where(self, *, sample=None, time=None, K=None, rank=None, kind=None) -> np.ndarray:
        """Rows matching every condition given (a value, or a list of values)."""
        keep = np.ones(len(self), dtype=bool)
        for name, value in (("sample", sample), ("time", time), ("K", K), ("rank", rank),
                            ("kind", kind)):
            if value is not None:
                values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
                keep &= np.isin(getattr(self, name), values)
        return keep


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


def analysis_controls(results, changes: Mapping[str, Any] | None = None, *, quiet: bool = False):
    """The controls the analysis uses (:class:`hxex.controls.ControlTable`, or None without
    controls): those of the run, recomputed from its K=1 fits when the review rejected a control
    spectrum (so that a rejected control does not count) or when ``changes`` (control settings
    to change, e.g. ``{'d2o_fraction': 0.85}``) are given. Recomputed controls leave out the
    rejected spectra and the noise outliers."""
    stored = results.controls
    run = results.settings.run if results.settings is not None else None
    base = stored.config if stored is not None else (run.controls if run is not None else None)
    if base is None:
        if not changes:
            return None
        base = ControlConfig()
    fit = results.fit
    rejected = np.asarray(results.review_status == "rejected", dtype=bool)
    time = np.asarray(fit.time, dtype=float)
    at_control = np.zeros(len(fit), dtype=bool)
    for t in (base.undeut_time, base.fd_time):
        if t is not None:
            at_control |= np.abs(time - t) <= base.atol
    n_rejected = int((rejected & at_control).sum())
    if stored is not None and not changes and not n_rejected:
        return stored
    config = dataclasses.replace(base, **dict(changes or {}))
    usable = ~rejected
    if fit.noise_estimate is not None:
        usable &= ~np.asarray(fit.noise_estimate.outlier, dtype=bool)
    rates = None
    if config.fd_mode == "kint":
        if results.binned is None or run is None:
            raise ValueError("fd_fallback 'kint' needs the run's picked spectra and settings "
                             "(the bundle's binned table)")
        from ..run import control_rates

        keep = np.isin(results.binned.spectrum_id, fit.spectrum_id)
        rates = control_rates(results.binned.select(keep),
                              dataclasses.replace(run, controls=config))
    out = fit_controls(fit, _site_counts(results), config, usable=usable, rates=rates)
    if not quiet:
        log.info("controls recomputed (%s): %d of %d series fall back%s",
                 "; ".join(x for x in (
                     f"{n_rejected} rejected control spectra left out" if n_rejected else "",
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


def _records(results, ks, exclude_flagged, filters, quiet=False, controls_changes=None):
    """The filtered values of every spectrum, grouped by (sample, peptide, modification, start,
    end, time, K, rank): lists of ``(fD, fD_se, population, population_se, charge, n_amides,
    rep, control_fallback, run)`` (run: '' for one run, the run's name in a
    :class:`hxex.analysis.RunSet`); with whether each sample's values are corrected
    (sample -> bool), ``nterm_subtract`` and the undeuterated time. ``results`` may be a
    :class:`hxex.analysis.RunSet` (its samples are then labelled ``run:sample``)."""
    from .runs import RunSet

    if isinstance(results, RunSet):
        return results._records(ks, exclude_flagged, filters, quiet, controls_changes)
    groups, corrected, nterm, undeut, dropped = _spectrum_records(
        results, ks, exclude_flagged, filters, quiet, controls_changes)
    groups = _filter_species(groups, ks, filters, dropped, quiet)
    return groups, corrected, nterm, undeut


def _spectrum_records(results, ks, exclude_flagged, filters, quiet=False,
                      controls_changes=None):
    """:func:`_records` of one run with only the per-spectrum filters applied, and the counts
    of what they dropped (the per-species filters, :func:`_filter_species`, come after merged
    samples are pooled)."""
    fit, binned = results.fit, results.binned
    if binned is None:
        raise ValueError("the species uptake table needs the picked table (results.binned)")
    missing = [k for k in ks if k not in fit.fits]
    if missing:
        raise ValueError(f"K = {missing} were not fitted (fitted: {list(fit.k_values)})")
    run = results.settings.run if results.settings is not None else None
    picker = results.settings.picker if results.settings is not None else None
    nterm = picker.nterm_subtract if picker is not None else 1
    index = {str(s): j for j, s in enumerate(binned.spectrum_id)}
    rows = np.array([index[str(s)] for s in fit.spectrum_id])
    keep = np.ones(len(fit), dtype=bool)
    keep &= results.review_status != "rejected"
    if getattr(results, "excluded_times", ()):  # left out of the analysis (exclude_times)
        keep &= ~results.excluded
    if exclude_flagged and results.quality is not None:
        keep &= ~np.asarray(results.quality.flagged, dtype=bool)
    controls = analysis_controls(results, controls_changes, quiet=quiet)
    n_amides = _site_counts(results)
    fell_back = np.zeros(len(fit), dtype=bool)
    if controls is not None:
        n_amides = np.asarray(controls.n_amides, dtype=float)[controls.rows(fit.data_id)]
        fell_back = np.asarray(controls.fallback, dtype=bool)[controls.rows(fit.data_id)]

    # one record per spectrum and rank, then grouped by species, time, K and rank
    groups: dict[tuple, list[tuple[float, float, float, float, int]]] = {}
    dropped = {"population": 0, "fD": 0, "fD_se": 0, "k1_between_ranks": 0}
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
        no_boot = int(np.isnan(mean_se[keep, 0]).sum())
        if k >= 2 and no_boot and not quiet:
            log.warning("K=%d: %d of %d spectra have no bootstrap, so their standard errors come "
                        "only from the scatter of replicates (or are unknown): too small for the "
                        "choice between one and two populations (run the bootstrap at K=1 and "
                        "K=2, BootstrapConfig.ks)", k, no_boot, int(keep.sum()))
        if controls is not None:
            c = correct_means(controls, fit.data_id, means, se=mean_se)
            means, mean_se = c.mean, c.se
        usable = keep.copy()
        if filters is not None and k >= 2:
            lo = filters.min_population
            bad_pop = np.any((fracs < lo) | (fracs > 1 - lo), axis=1) & keep
            bad_fd = np.any(means / n_amides[:, None] > filters.max_fD, axis=1) & keep
            dropped["population"] += int(bad_pop.sum())
            dropped["fD"] += int((bad_fd & ~bad_pop).sum())
            usable &= ~(bad_pop | bad_fd)
        for i in np.flatnonzero(usable):
            j = rows[i]
            key = (str(binned.sample[j]), str(binned.peptide[j]), str(binned.modification[j]),
                   int(binned.start_seq[j]), int(binned.end_seq[j]), float(fit.time[i]), k)
            for r in range(k):
                groups.setdefault(key + (r + 1,), []).append((
                    means[i, r] / n_amides[i], mean_se[i, r] / n_amides[i], fracs[i, r],
                    frac_se[i, r], int(binned.charge[j]), float(n_amides[i]),
                    int(binned.rep[j]), bool(fell_back[i]), "",
                ))

    undeut = run.reference_time if run is not None else float(np.min(fit.time))
    corrected = {str(s): controls is not None for s in np.unique(binned.sample)}
    return groups, corrected, int(nterm), float(undeut), dropped


def _filter_species(groups: dict, ks, filters, dropped: dict, quiet: bool = False) -> dict:
    """The per-species filters of :class:`SpeciesFilters` (``max_fD_se``,
    ``k1_between_ranks``) on grouped records, once per species and time, after merged samples
    are pooled; ``dropped`` (the per-spectrum counts) is completed and logged."""
    if filters is not None:  # a K >= 2 description whose fD error is too large is not kept
        worst: dict[tuple, float] = {}
        for key, recs in groups.items():
            if key[6] >= 2:
                _, se = _combine(np.array([r[0] for r in recs]), np.array([r[1] for r in recs]))
                worst[key[:7]] = max(worst.get(key[:7], 0.0), se if np.isfinite(se) else 0.0)
        too_wide = {key for key, se in worst.items() if se >= filters.max_fD_se}
        dropped["fD_se"] = len(too_wide)
        groups = {key: recs for key, recs in groups.items() if key[:7] not in too_wide}
        if filters.k1_between_ranks and 1 in ks:  # the K=1 value between the K >= 2 ranks
            value = {key: _combine(np.array([r[0] for r in recs]),
                                   np.array([r[1] for r in recs]))[0]
                     for key, recs in groups.items()}
            unordered = set()
            for key in value:
                base, k, rank = key[:6], key[6], key[7]
                if k < 2 or rank != 1 or base + (1, 1) not in value or base + (k, k) not in value:
                    continue
                if not value[key] < value[base + (1, 1)] < value[base + (k, k)]:
                    unordered.add(base + (k,))
            dropped["k1_between_ranks"] = len(unordered)
            groups = {key: recs for key, recs in groups.items() if key[:7] not in unordered}
        if any(dropped.values()) and not quiet:
            log.info("K >= 2 descriptions left out: %d spectra with a population outside "
                     "[%g, %g], %d with a component above fD %g; %d species and times with "
                     "an fD error of %g or more, %d with the K=1 value outside the ranks",
                     dropped["population"], filters.min_population, 1 - filters.min_population,
                     dropped["fD"], filters.max_fD, dropped["fD_se"], filters.max_fD_se,
                     dropped["k1_between_ranks"])
    return groups


def species_uptake(
    results,
    ks: Sequence[int] = (1, 2),
    *,
    exclude_flagged: bool = False,
    filters: SpeciesFilters | None = SpeciesFilters(),  # noqa: B008
    controls: Mapping[str, Any] | None = None,
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
    """
    groups, corrected, nterm, undeut = _records(results, ks, exclude_flagged, filters,
                                                controls_changes=controls)
    cols: dict[str, list] = {f.name: [] for f in fields(SpeciesUptake) if f.name not in _SCALARS}
    by_charge: dict[tuple, dict[int, list[tuple[float, float]]]] = {}
    by_run: dict[tuple, list[dict]] = {}
    for key, recs in sorted(groups.items()):
        sample, peptide, mod, start, end, time, k, rank = key
        v = np.array([r[0] for r in recs])
        s = np.array([r[1] for r in recs])
        f = np.array([r[2] for r in recs])
        fs = np.array([r[3] for r in recs])
        d, d_se = _combine(v, s)
        fr, fr_se = _combine(f, fs)
        charges = sorted({r[4] for r in recs})
        amides = float(np.mean([r[5] for r in recs]))  # the same for every spectrum of a species
        run_names = [x for x in dict.fromkeys(r[8] for r in recs) if x] or [""]  # RunSet order
        ion = chem.parse_modification(mod).ion_type
        for name, value in (
            ("sample", sample), ("peptide", peptide), ("modification", mod),
            ("kind", "peptide" if ion == "M" else "fragment"), ("start", start), ("end", end),
            ("time", time), ("K", k), ("rank", rank), ("mean", d * amides),
            ("mean_se", d_se * amides), ("fD", d), ("fD_se", d_se),
            ("population", fr), ("population_se", fr_se), ("n_amides", amides),
            ("n_spectra", len(recs)), ("n_charges", len(charges)),
            ("corrected", corrected[sample]), ("charge_trend", False),
            ("control_fallback", any(r[7] for r in recs)),
            ("runs", ",".join(run_names)),
            ("n_spectra_by_run", ",".join(str(sum(r[8] == x for r in recs)) for x in run_names)
             if any(run_names) else ""),
            ("run_offset", False),
        ):
            cols[name].append(value)
        if len(run_names) > 1:  # each run against the first, at this time
            per_run = {x: _combine(*map(np.array, zip(*[(r[0], r[1]) for r in recs
                                                          if r[8] == x], strict=True)))
                       for x in run_names}
            by_run.setdefault((sample, peptide, mod, k, rank), []).append(per_run)
        if len(charges) > 1:
            per = by_charge.setdefault((sample, peptide, mod, k, rank), {})
            low = [(r[0], r[1]) for r in recs if r[4] == charges[0]]
            high = [(r[0], r[1]) for r in recs if r[4] == charges[-1]]
            a, b = _combine(*map(np.array, zip(*low, strict=True))), _combine(
                *map(np.array, zip(*high, strict=True)))
            per[len(per)] = [a, b]
    trend = {key for key, times in by_charge.items() if _is_trend(list(times.values()))}
    out = {name: np.array(values) for name, values in cols.items()}
    species = list(zip(out["sample"], out["peptide"], out["modification"], out["K"], out["rank"],
                       strict=True))
    out["charge_trend"] = np.array([(str(a), str(b), str(c), int(d), int(e)) in trend
                                    for a, b, c, d, e in species], dtype=bool)
    offset = set()
    for key, times in by_run.items():
        first = next(iter(times[0]))  # the first run is the same at every time if present
        others = {x for per in times for x in per if x != first}
        for other in others:
            pairs = [[per[first], per[other]] for per in times if first in per and other in per]
            if _is_trend(pairs):
                offset.add(key)
    out["run_offset"] = np.array([(str(a), str(b), str(c), int(d), int(e)) in offset
                                  for a, b, c, d, e in species], dtype=bool)
    if offset:
        log.warning("%d merged species differ systematically between runs (run_offset)",
                    len({key[:3] for key in offset}))
    for name in ("start", "end", "K", "rank", "n_spectra", "n_charges"):
        out[name] = out[name].astype(np.int64)
    for name in ("time", "mean", "mean_se", "fD", "fD_se", "population", "population_se",
                 "n_amides"):
        out[name] = out[name].astype(np.float64)
    out["corrected"] = out["corrected"].astype(bool)
    out["control_fallback"] = out["control_fallback"].astype(bool)
    for name in ("runs", "n_spectra_by_run"):
        out[name] = out[name].astype(str)
    return SpeciesUptake(**out, nterm_subtract=nterm, undeut_time=undeut)


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
