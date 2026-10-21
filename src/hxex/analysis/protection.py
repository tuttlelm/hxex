# SPDX-License-Identifier: GPL-3.0-or-later
"""Peptide protection factors from MS1 uptake (docs/ANALYSIS.md, "MS1 protection factors").

Opt-in, with the labelling conditions (:class:`hxex.kint.ExchangeConditions`). An amide with
intrinsic rate ``kint`` (``kforw + kback`` in the labelling solvent, :mod:`hxex.kint`)
exchanges at ``kint / protection`` (EX2), so a peptide's back-exchange-corrected fraction
deuterated is modelled as ``mean_i(1 - exp(-kint_i t / protection))`` over the amides it
counts, with one protection factor per peptide. The name is always ``protection``, never
``P`` or ``p`` (the binomial parameter and the p-value).

:func:`peptide_protection` gives, per sample, peptide, K and rank, the apparent protection
factor at each time (the value that gives the measured ``fD``) and one factor fitted over all
times in ``log10``, with bounds where the data hold only a limit. K = 2 ranks get apparent
values too: two populations come from EX1 or mixed exchange, where a protection factor has no
EX2 meaning. :func:`compare_protection` gives the differences between samples in
``log10_protection``.

Both directions (docs/ANALYSIS.md, "Direction"): with ``initial_fraction`` in the conditions
above their ``d2o_fraction`` the amides began deuterated and lose deuterium (back-exchange
kinetics), and the model is ``mean_i(e_i + (fD0 - e_i) exp(-kint_i t / protection))``: ``e_i``
the amide's equilibrium in the solvent (``kforw / (kforw + kback)``, over ``initial_fraction``
on the corrected scale), ``fD0`` fitted unless a measured fully deuterated control fixes it
at 1. Forward labelling is the same model with ``fD0 = 0`` and ``e_i = 1``.

Intrinsic rates are those of the protein labelled whole: a peptide's ends are not termini
unless they are the protein's (or written, ``Ac-``, ``-NH2``), and its first amide's rate
depends on the residue before it (from the protein sequence, else from the other peptides).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np

from .. import chem
from ..kint import ExchangeConditions, intrinsic_rates
from .compare import _pairs
from .residues import states
from .runs import label as _label
from .runs import resolve_sample, sample_setting
from .species import SpeciesUptake

log = logging.getLogger(__name__)

# fD beyond these at every time gives only a limit (owner's design)
LOW, HIGH = 0.05, 0.95
_LOG_RANGE = (-6.0, 15.0)  # log10 protection searched


def _columns(obj) -> dict[str, np.ndarray]:
    return {"run": obj.run, **{f.name: getattr(obj, f.name) for f in fields(obj)
                               if isinstance(getattr(obj, f.name), np.ndarray)
                               and f.name != "run"}}


def _runs(obj) -> None:
    """Fill a table's ``run`` column (blank: one run) and make it an object array."""
    run = obj.run if obj.run is not None else np.full(len(obj), "", dtype=object)
    object.__setattr__(obj, "run", np.asarray(run, dtype=object))


def _where(obj, **conditions) -> np.ndarray:
    keep = np.ones(len(obj), dtype=bool)
    for name, value in conditions.items():
        if value is not None:
            values = value if isinstance(value, (list, tuple, set, np.ndarray)) else [value]
            keep &= np.isin(getattr(obj, name), list(values))
    return keep


@dataclass(frozen=True)
class ProtectionTable:
    """One row per run, sample, peptide (``peptide``, ``modification``, ``start``, ``end``),
    ``K`` and ``rank``: ``log10_protection`` and ``log10_protection_se`` (one factor fitted over
    every time; NaN for a bound), ``protection`` (``10**log10_protection``), ``bound``
    (``'upper'``: exchanged at every time, the factor is at most ``limit``; ``'lower'``: barely
    exchanged at every time, at least ``limit``; ``''``: a value), ``limit``, ``n_times``,
    ``protection_drift`` (log10 of the last apparent factor over the first, of the times
    between the limits; NaN with fewer than two: the amides of one peptide protected
    differently make it rise), ``misfit`` (reduced chi-square of the fit; NaN without errors
    or with one time) and ``one_factor_fits`` (``misfit`` within the chi-square limit at
    ``alpha``; False where it cannot be tested), ``n_amides`` and ``n_approximate`` (amides
    whose intrinsic rate borrows a factor: modified residues), ``direction`` (``'forward'``,
    labelling from 0 towards the plateau 1; ``'back'``, the amides began deuterated: the
    conditions' ``initial_fraction`` is above their ``d2o_fraction``), ``corrected`` (for
    back-exchange without correction, ``fD`` is ``mean / n_amides``, the absolute scale),
    ``fD0`` (the ``fD`` at time 0: 0 forward, 1 for back-exchange from a measured fully
    deuterated control, otherwise fitted, with ``fD0_se``, NaN when fixed or unknown) and
    ``equilibrium`` (the ``fD`` the peptide relaxes to, the mean of its amides': 1 forward)."""

    sample: np.ndarray
    peptide: np.ndarray
    modification: np.ndarray
    start: np.ndarray
    end: np.ndarray
    K: np.ndarray
    rank: np.ndarray
    log10_protection: np.ndarray
    log10_protection_se: np.ndarray
    protection: np.ndarray
    bound: np.ndarray
    limit: np.ndarray
    n_times: np.ndarray
    protection_drift: np.ndarray
    misfit: np.ndarray
    one_factor_fits: np.ndarray
    n_amides: np.ndarray
    n_approximate: np.ndarray
    direction: np.ndarray
    corrected: np.ndarray
    fD0: np.ndarray
    fD0_se: np.ndarray
    equilibrium: np.ndarray
    run: np.ndarray | None = None

    def __post_init__(self):
        _runs(self)

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return _columns(self)

    def where(self, **conditions) -> np.ndarray:
        return _where(self, **conditions)


@dataclass(frozen=True)
class ProtectionByTime:
    """The apparent protection factor of every run, sample, peptide, ``K``, ``rank`` and ``time``:
    the measured ``fD`` and ``fD_se``, the ``progress`` of the exchange (``(fD0 - fD) / (fD0 -
    equilibrium)`` with the values of :class:`ProtectionTable`: ``fD`` itself in forward
    labelling; NaN where ``fD0`` is unknown), ``log10_protection`` and its
    ``log10_protection_se`` (from ``fD_se``), ``bound`` and ``limit`` as in
    :class:`ProtectionTable` (at that time)."""

    sample: np.ndarray
    peptide: np.ndarray
    modification: np.ndarray
    start: np.ndarray
    end: np.ndarray
    K: np.ndarray
    rank: np.ndarray
    time: np.ndarray
    fD: np.ndarray
    fD_se: np.ndarray
    progress: np.ndarray
    log10_protection: np.ndarray
    log10_protection_se: np.ndarray
    bound: np.ndarray
    limit: np.ndarray
    run: np.ndarray | None = None

    def __post_init__(self):
        _runs(self)

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return _columns(self)

    def where(self, **conditions) -> np.ndarray:
        return _where(self, **conditions)


# --- intrinsic rates in the protein --------------------------------------------------------------

def _known_residues(uptake: SpeciesUptake, state: tuple[str, str], registry) -> dict[int, str]:
    """Residue number -> modX label, from the peptides of ``state`` (run, sample)."""
    known: dict[int, str] = {}
    rows = np.flatnonzero(uptake.where(run=state[0], sample=state[1]))
    for pep, start in {(str(uptake.peptide[i]), int(uptake.start[i])) for i in rows}:
        for j, label in enumerate(chem.parse_sequence(pep, registry)):
            known.setdefault(start + j, label)
    return known


def peptide_rates(
    peptide: str,
    start: int,
    conditions: ExchangeConditions,
    *,
    before: str | None = None,
    protein_end: bool = False,
    nterm_subtract: int = 1,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """The intrinsic rates (1/s) and ``approximate`` marks of the amides a peptide counts
    (:func:`hxex.chem.amide_positions`), as a stretch of a protein labelled whole: ``before``
    is the residue before it (None: unknown, or ``start`` 1, the protein's N-terminus);
    ``protein_end``: its last residue is the protein's (a free carboxyl). A counted N-terminal
    amine of the protein exchanges at once (infinite rate); an amide whose rate is unknown (the
    residue before is) is NaN."""
    k, _, approx = _peptide_exchange(peptide, start, conditions, before=before,
                                     protein_end=protein_end, nterm_subtract=nterm_subtract,
                                     registry=registry)
    return k, approx


def _peptide_exchange(peptide, start, conditions, *, before, protein_end, nterm_subtract,
                      registry):
    """:func:`peptide_rates` with the amides' equilibrium D fractions in the solvent
    (:attr:`hxex.kint.IntrinsicRates.equilibrium`; NaN where the rate is not finite):
    ``(rates, equilibria, approximate)``."""
    nterm, core, cterm = chem.split_termini(peptide)
    n_free = start == 1 or nterm != "H-"
    seq, first = peptide, start
    if not n_free and before is not None:
        seq, first = (f"{before}{core}{cterm}" if cterm != "-OH" else f"{before}{core}"), start - 1
    rates = intrinsic_rates(seq, conditions, start=first, registry=registry,
                            free_termini=(n_free, protein_end))
    pos = np.array(chem.amide_positions(peptide, "M", None, nterm_subtract=nterm_subtract,
                                        registry=registry), dtype=np.int64)
    at = rates.at(start + pos - 1)
    k = rates.rate[at]
    if n_free and nterm == "H-":
        k = np.where((pos == 1) & ~np.isfinite(k), np.inf, k)  # the free amine
    return k, rates.equilibrium[at], rates.approximate[at]


# --- the fits ------------------------------------------------------------------------------------

F0_MAX = 2.0  # the largest fitted fD0 (corrected values can exceed 1 a little)


def _model(x: float, k: np.ndarray, t, f0: float = 0.0, e=1.0) -> np.ndarray:
    """``mean_i(e_i + (f0 - e_i) exp(-k_i t / 10**x))`` at the times ``t``: forward labelling
    (``mean_i(1 - exp(...))``) with the defaults."""
    rise = -np.expm1(-np.multiply.outer(np.atleast_1d(t), k / 10.0**x))
    return np.mean(f0 + (e - f0) * rise, axis=-1)


def _slope(x: float, k: np.ndarray, t: float) -> float:
    """d model / d x at one time."""
    u = k * t / 10.0**x
    with np.errstate(invalid="ignore"):
        terms = np.where(np.isfinite(u), u * np.exp(-u), 0.0)  # an infinite rate: no slope
    return float(-np.log(10) * np.mean(terms))


def _solve(k: np.ndarray, t: float, y: float) -> float:
    """The log10 protection that gives ``y`` at ``t`` (clipped to the searched range)."""
    from scipy.optimize import brentq

    lo, hi = _LOG_RANGE
    f_lo, f_hi = float(_model(lo, k, t)[0]) - y, float(_model(hi, k, t)[0]) - y
    if f_lo <= 0:
        return lo
    if f_hi >= 0:
        return hi
    return float(brentq(lambda x: float(_model(x, k, t)[0]) - y, lo, hi, xtol=1e-6))


def _apparent(k, t, y, se):
    """(log10 protection, its se, bound, limit) at one time."""
    if not np.isfinite(y):
        return np.nan, np.nan, "", np.nan
    if y >= HIGH:
        return np.nan, np.nan, "upper", 10.0 ** _solve(k, t, HIGH)
    if y <= LOW:
        return np.nan, np.nan, "lower", 10.0 ** _solve(k, t, LOW)
    x = _solve(k, t, y)
    slope = _slope(x, k, t)
    return x, (abs(se / slope) if np.isfinite(se) and slope != 0 else np.nan), "", np.nan


def _known_errors(se: np.ndarray) -> np.ndarray | None:
    """``se`` with every unknown error (NaN, or not above 0) replaced by the median of the
    known ones; None when none is known (an unweighted fit). An unknown error must not get
    the largest weight."""
    # decided in docs/AUDIT_2026-10-17.md
    se = np.asarray(se, dtype=float)
    known = np.isfinite(se) & (se > 0)
    if not known.any():
        return None
    return np.where(known, se, float(np.median(se[known])))


def _fit(k, t, y, se, apparent, alpha, f0=0.0, e=1.0, free=False):
    """(log10 protection, its se, fD0, its se, misfit, one factor fits) over every time;
    weighted by the errors when any is known (:func:`_known_errors`). ``f0`` is fixed unless
    ``free`` (then fitted, from a few starting factors, between the largest equilibrium
    ``e`` and :data:`F0_MAX`; its se is NaN when fixed)."""
    from scipy import stats
    from scipy.optimize import least_squares

    se = _known_errors(se)
    weighted = se is not None
    w = 1.0 / se if weighted else np.ones_like(y)
    finite = np.isfinite(apparent)
    x0 = float(np.median(apparent[finite])) if finite.any() else 0.0
    lo, hi = _LOG_RANGE
    if free:
        f_lo = float(np.max(e))
        f_start = float(np.clip(np.max(y), f_lo + 1e-3, F0_MAX - 1e-3))
        best = None
        for dx in (-2.0, 0.0, 2.0):
            start = [float(np.clip(x0 + dx, lo + 1e-3, hi - 1e-3)), f_start]
            res = least_squares(lambda p: (_model(p[0], k, t, p[1], e) - y) * w, start,
                                bounds=([lo, f_lo], [hi, F0_MAX]))
            if best is None or res.cost < best.cost:
                best = res
        res = best
        f0 = float(res.x[1])
    else:
        res = least_squares(lambda x: (_model(x[0], k, t, f0, e) - y) * w, [x0], bounds=_LOG_RANGE)
    x = float(res.x[0])
    n, n_par = len(y), res.jac.shape[1]
    dof = n - n_par
    chi2 = float(np.sum(res.fun**2))
    misfit = chi2 / dof if dof > 0 and weighted else np.nan
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac)
    except np.linalg.LinAlgError:
        cov = np.full((n_par, n_par), np.nan)
    if weighted:
        scale = max(1.0, misfit) if np.isfinite(misfit) else 1.0
    else:  # no errors: the residuals' scatter
        scale = chi2 / dof if dof > 0 else np.nan
    var = np.diag(cov) * scale
    var = np.where(var > 0, var, np.nan)
    fits = bool(np.isfinite(misfit) and chi2 <= stats.chi2.ppf(1 - alpha, dof))
    return (x, float(np.sqrt(var[0])), f0, float(np.sqrt(var[1])) if free else np.nan, misfit,
            fits)


def _back(conditions) -> bool:
    """Whether ``conditions`` describe a back-exchange: the amides began with more deuterium
    (``initial_fraction``) than the solvent holds (``d2o_fraction``)."""
    return (conditions is not None and conditions.initial_fraction is not None
            and conditions.initial_fraction > conditions.d2o_fraction)


def _check_direction(pep, back: set, judged: set) -> None:
    """Refuse a sample of ``judged`` (the (run, sample) pairs with conditions) whose peptides
    mostly go the other way from its conditions' direction (``back``: the back-exchange
    pairs). A peptide falls (rises) when its last finite ``fD`` is below (above) its first by
    more than twice their combined error, or by 0.05 without one."""
    # the two-direction model: docs/AUDIT_2026-10-17.md, design item 3
    falling: dict[tuple, int] = {}
    rising: dict[tuple, int] = {}
    groups: dict[tuple, list[int]] = {}
    for i in np.argsort(pep.time, kind="stable"):
        key = ((str(pep.run[i]), str(pep.sample[i])), str(pep.peptide[i]),
               str(pep.modification[i]), int(pep.start[i]), int(pep.end[i]), int(pep.K[i]),
               int(pep.rank[i]))
        if np.isfinite(pep.fD[i]):
            groups.setdefault(key, []).append(int(i))
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        first, last = rows[0], rows[-1]
        step = 2.0 * float(np.hypot(pep.fD_se[first], pep.fD_se[last]))
        step = step if np.isfinite(step) and step > 0 else 0.05
        change = float(pep.fD[last] - pep.fD[first])
        if change < -step:
            falling[key[0]] = falling.get(key[0], 0) + 1
        elif change > step:
            rising[key[0]] = rising.get(key[0], 0) + 1

    def counts(states):
        return ", ".join(f"{_label(*s)}: {falling.get(s, 0)} falling, {rising.get(s, 0)} rising"
                         for s in states)

    found = (set(falling) | set(rising)) & judged
    down = sorted(s for s in found - back if falling.get(s, 0) > rising.get(s, 0))
    up = sorted(s for s in found & back if rising.get(s, 0) > falling.get(s, 0))
    if down:
        raise ValueError(
            f"sample(s) {', '.join(_label(*s) for s in down)}: fD falls with time in most "
            f"peptides ({counts(down)}), but the labelling conditions describe forward "
            "labelling. For back-exchange kinetics (deuterated, then exchanged in a solvent "
            "with less deuterium) set initial_fraction in [analysis.labelling] to the D "
            "fraction the amides began with, above the d2o_fraction of the solvent the "
            "exchange happens in")
    if up:
        raise ValueError(
            f"sample(s) {', '.join(_label(*s) for s in up)}: fD rises with time in most "
            f"peptides ({counts(up)}), but initial_fraction above d2o_fraction in "
            "[analysis.labelling] describes back-exchange: leave initial_fraction out for "
            "forward labelling")


def _prepare(uptake, conditions, sequences, exclude_times, registry):
    """The peptides of ``uptake`` that get protection factors: (the table, its rows grouped by
    (run, sample), peptide, modification, start, end, K and rank, the rates of each peptide by
    its first five keys (``(kint of the counted amides, how many approximate, their residue
    numbers, their equilibria on the state's scale)``, None where not all known), and the
    back-exchange states with the ``fD`` their amides began with when it is not fitted from
    the data: 1 corrected, ``initial_fraction`` on the absolute scale). ``conditions`` and
    ``sequences`` are one value for every sample, or by sample with per-run entries
    (:func:`hxex.analysis.sample_setting`). Modified forms stay in their sample."""
    pep = uptake.select(uptake.where(kind="peptide"))
    times = np.asarray(pep.time, dtype=float)
    keep = (pep.role == "") & np.isfinite(times)  # the time points, not the controls
    for t in exclude_times:
        keep &= ~np.isclose(times, t)
    pep = pep.select(keep)
    judged = {st for st in states(pep) if sample_setting(conditions, *st)}
    _check_direction(pep, {st for st in judged if _back(sample_setting(conditions, *st))},
                     judged)
    def setting_of(setting, state):
        return sample_setting(setting, state[0], state[1])

    all_states = states(pep)
    state_of = list(zip(pep.run.astype(str), pep.sample.astype(str), strict=True))
    back = {st for st in all_states if _back(setting_of(conditions, st))}
    raw = sorted({st for st, c in zip(state_of, pep.corrected, strict=True) if not c})
    uncorrected = [st for st in raw if st not in back]
    if uncorrected:
        log.warning("no protection factors for sample(s) %s: not corrected for back-exchange "
                    "(the run had no controls, [run.controls])",
                    ", ".join(_label(*st) for st in uncorrected))
    if set(raw) & back:
        log.info("back-exchange sample(s) %s not corrected: fitted on the absolute scale (fD = "
                 "mean / n_amides, relaxing to the solvent's equilibrium)",
                 ", ".join(_label(*st) for st in sorted(set(raw) & back)))
    start_of = {st: (float(setting_of(conditions, st).initial_fraction) if st in raw else 1.0)
                for st in back}
    no_conditions = sorted(set(all_states) - set(uncorrected)
                           - {st for st in all_states if setting_of(conditions, st)})
    if no_conditions:
        log.warning("no labelling conditions for sample(s) %s ([analysis.labelling]): no "
                    "protection factors for them", ", ".join(_label(*st) for st in no_conditions))
    left_out = set(uncorrected + no_conditions)
    pep = pep.select(np.array([st not in left_out for st in state_of], dtype=bool))
    groups: dict[tuple, list[int]] = {}
    for i in range(len(pep)):
        key = ((str(pep.run[i]), str(pep.sample[i])), str(pep.peptide[i]),
               str(pep.modification[i]), int(pep.start[i]), int(pep.end[i]), int(pep.K[i]),
               int(pep.rank[i]))
        groups.setdefault(key, []).append(i)
    labels: dict[tuple[str, str], dict[int, str]] = {}
    for state in states(pep):
        seq = setting_of(sequences, state)
        labels[state] = ({j + 1: lab for j, lab in enumerate(chem.parse_sequence(seq, registry))}
                         if seq else _known_residues(pep, state, registry))
    rates: dict[tuple, tuple | None] = {}
    skipped: dict[tuple, int] = {}
    for key, rows in groups.items():
        state, peptide, mod, start, end = rkey = key[:5]
        if rkey in rates:
            continue
        rates[rkey] = None
        spec = chem.parse_modification(mod)
        if spec.ion_type == "M" and not spec.hex:
            seq = setting_of(sequences, state)
            last = bool(seq) and end == len(labels[state])
            cond = setting_of(conditions, state)
            k, eq, approx = _peptide_exchange(peptide, start, cond,
                                              before=labels[state].get(start - 1),
                                              protein_end=last, nterm_subtract=pep.nterm_subtract,
                                              registry=registry)
            pos = np.array(chem.amide_positions(peptide, "M", None,
                                                nterm_subtract=pep.nterm_subtract,
                                                registry=registry), dtype=np.int64)
            if len(k) == int(pep.n_amides[rows[0]]) and len(k) and not np.isnan(k).any():
                if state in back:  # a free amine (infinite rate) takes the others' equilibrium
                    known = np.isfinite(eq)
                    eq = np.where(known, eq, np.mean(eq[known]) if known.any()
                                  else cond.d2o_fraction)
                    e = eq / cond.initial_fraction if start_of[state] == 1.0 else eq
                else:
                    e = np.ones(len(k))
                rates[rkey] = (k, int(approx.sum()), start + pos - 1, e)
        if rates[rkey] is None:
            skipped[state] = skipped.get(state, 0) + 1
    if skipped:
        log.warning("protection factors: %s peptide(s) left out (rates not all known: a modified "
                    "ion type, or the residue before unknown)",
                    ", ".join(f"{_label(*s)}: {n}" for s, n in skipped.items()))
    return pep, groups, rates, start_of


def _progress(y, se, f0, equilibrium):
    """The exchange's progress ``(f0 - y) / (f0 - equilibrium)`` at each time and its se
    (``y`` itself in forward labelling); NaN without a start."""
    span = f0 - equilibrium
    if not np.isfinite(span) or abs(span) < 1e-6:
        return np.full(len(y), np.nan), np.full(len(y), np.nan)
    return (f0 - y) / span, se / abs(span)


def _start(pep, rows, state, start_of) -> tuple[float, bool, float]:
    """(fD0, whether it is fitted, the largest fD0 the data can have) of one peptide: 0 in
    forward labelling; for back-exchange 1 from measured controls (the run's corrected values,
    no control fallback; a fully deuterated control from the peptide's other charge states
    counts as measured, HISTORY 130), otherwise fitted."""
    if state not in start_of:
        return 0.0, False, 0.0
    fell_back = (np.asarray(pep.control_fallback, dtype=bool)
                 & ~np.asarray(pep.fd_other_charge, dtype=bool))
    if start_of[state] == 1.0 and not fell_back[rows].any():
        return 1.0, False, 1.0
    return np.nan, True, start_of[state]


def peptide_protection(
    uptake: SpeciesUptake,
    conditions: ExchangeConditions | Mapping[str, ExchangeConditions],
    *,
    sequences: str | Mapping[str, str] | None = None,
    exclude_times=(),
    alpha: float = 0.05,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
) -> tuple[ProtectionTable, ProtectionByTime]:
    """The protection factors of the peptides of ``uptake`` (every K and rank it holds) in the
    labelling ``conditions`` (one set for every sample, or sample -> set): see the module
    docstring. The controls (the rows with a role) and the times in ``exclude_times`` are
    left out. ``sequences`` (the protein's, one or per
    sample; :meth:`hxex.analysis.AnalysisConfig.sequences`) gives the residue before each
    peptide and the protein's C-terminus; without it the residue before comes from the other
    peptides, and no peptide ends at a free carboxyl. Forward-labelling samples without
    back-exchange correction, and samples without conditions, are left out with a warning,
    and so are peptides whose rates are not all known (a modified ion type, or the residue
    before unknown). A sample whose peptides
    lose deuterium with time is fitted in the other direction when its conditions say so
    (``initial_fraction`` above ``d2o_fraction``; see the module docstring), and refused
    otherwise, as is a back-exchange sample whose peptides gain deuterium. Back-exchange
    samples without correction are fitted on the absolute scale. The fit over all times is
    weighted by ``fD_se`` where it is known; an unknown error takes the median of the
    peptide's known ones (none known: unweighted)."""
    pep, groups, rates, start_of = _prepare(uptake, conditions, sequences, exclude_times,
                                            registry)
    too_few = 0
    out = {f.name: [] for f in fields(ProtectionTable)}
    by_time = {f.name: [] for f in fields(ProtectionByTime)}
    for key, rows in groups.items():
        if rates[key[:5]] is None:
            continue
        k, n_approx, _, e = rates[key[:5]]
        f0, free, f0_ref = _start(pep, rows, key[0], start_of)
        e_mean = float(np.mean(e))
        key = (*key[0], *key[1:])  # (run, sample, peptide, ...)
        rows = sorted(rows, key=lambda i: pep.time[i])
        t = pep.time[rows].astype(float)
        y = pep.fD[rows].astype(float)
        se = pep.fD_se[rows].astype(float)
        finite = np.isfinite(y)
        x = x_se = misfit = f0_se = np.nan
        fits, bound, limit = False, "", np.nan

        if free:
            # every amide at equilibrium by the first time (from the largest start the data
            # can have): neither the start nor the speed is known, the factor is a bound
            u_ref, _ = _progress(y, se, f0_ref, e_mean)
            if finite.any() and np.all(u_ref[finite] >= HIGH):
                f0, basis = np.nan, f0_ref
            elif finite.sum() >= 3:
                x_ref = np.array([_apparent(k, ti, ui, np.nan)[0]
                                  for ti, ui in zip(t, u_ref, strict=True)])
                x, x_se, f0, f0_se, misfit, fits = _fit(
                    k, t[finite], y[finite], se[finite], x_ref[finite], alpha, e=e, free=True)
                basis = f0
            else:
                f0 = basis = np.nan
                too_few += 1
        else:
            basis = f0
        u, u_se = _progress(y, se, basis, e_mean)
        app = [_apparent(k, ti, ui, si) for ti, ui, si in zip(t, u, u_se, strict=True)]
        x_t = np.array([a[0] for a in app])
        for j, (ti, a) in enumerate(zip(t, app, strict=True)):
            for name, value in zip(("run", "sample", "peptide", "modification", "start", "end",
                                    "K", "rank"), key, strict=True):
                by_time[name].append(value)
            for name, value in (("time", ti), ("fD", y[j]), ("fD_se", se[j]),
                                ("progress", u[j]),
                                ("log10_protection", a[0]), ("log10_protection_se", a[1]),
                                ("bound", a[2]), ("limit", a[3])):
                by_time[name].append(value)
        bounds = [a[2] for a, f in zip(app, finite, strict=True) if f]
        if bounds and all(b == "upper" for b in bounds):
            bound, limit = "upper", min(a[3] for a in app if a[2] == "upper")
            x = x_se = misfit = np.nan
            fits = False
        elif bounds and all(b == "lower" for b in bounds):
            bound, limit = "lower", max(a[3] for a in app if a[2] == "lower")
            x = x_se = misfit = np.nan
            fits = False
        elif finite.any() and not free:
            x, x_se, _, _, misfit, fits = _fit(k, t[finite], y[finite], se[finite],
                                               x_t[finite], alpha, f0=f0, e=e)
        inside = x_t[np.isfinite(x_t)]
        drift = float(inside[-1] - inside[0]) if len(inside) > 1 else np.nan
        for name, value in zip(("run", "sample", "peptide", "modification", "start", "end", "K",
                                "rank"), key, strict=True):
            out[name].append(value)
        for name, value in (("log10_protection", x), ("log10_protection_se", x_se),
                            ("protection", 10.0**x), ("bound", bound), ("limit", limit),
                            ("n_times", int(finite.sum())), ("protection_drift", drift),
                            ("misfit", misfit), ("one_factor_fits", fits),
                            ("n_amides", len(k)), ("n_approximate", n_approx),
                            ("direction", "back" if f0_ref else "forward"),
                            ("corrected", bool(np.asarray(pep.corrected)[rows].all())),
                            ("fD0", f0), ("fD0_se", f0_se), ("equilibrium", e_mean)):
            out[name].append(value)
    if too_few:
        log.warning("protection factors: %d back-exchange peptide(s) with fewer than 3 times "
                    "left without a value (their starting fD is fitted)", too_few)
    ints = ("start", "end", "K", "rank", "n_times", "n_amides", "n_approximate")
    text = ("sample", "peptide", "modification", "bound", "direction")

    def array(name, values):
        if name == "run":
            return np.array(values, dtype=object)
        if name in text:
            return np.array(values, dtype=str)
        if name in ("one_factor_fits", "corrected"):
            return np.array(values, dtype=bool)
        return np.array(values, dtype=np.int64 if name in ints else float)

    return (ProtectionTable(**{n: array(n, v) for n, v in out.items()}),
            ProtectionByTime(**{n: array(n, v) for n, v in by_time.items()}))


# --- differences between samples -----------------------------------------------------------------

@dataclass(frozen=True)
class ProtectionComparison:
    """``log10_protection`` of each sample's peptides against the ``reference`` sample's (the
    same residues with ``match='range'``, the same sequence with ``'sequence'``), per ``K`` and
    ``rank``: ``delta`` (sample minus reference, log10 units), ``delta_se``, ``z`` and
    ``significant`` (``|z|`` beyond the normal quantile of ``alpha``, two-sided); ``note`` says
    why there is no value (a bound on either side). ``peptide_reference`` is the reference's
    peptide (a mutant's differs)."""

    sample: np.ndarray
    peptide: np.ndarray
    modification: np.ndarray
    start: np.ndarray
    end: np.ndarray
    K: np.ndarray
    rank: np.ndarray
    peptide_reference: np.ndarray
    delta: np.ndarray
    delta_se: np.ndarray
    z: np.ndarray
    significant: np.ndarray
    note: np.ndarray
    reference: str = ""
    alpha: float = 0.05
    run: np.ndarray | None = None
    reference_run: str = ""

    def __post_init__(self):
        _runs(self)

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return _columns(self)

    def where(self, **conditions) -> np.ndarray:
        return _where(self, **conditions)


def compare_protection(
    table: ProtectionTable,
    reference: str,
    *,
    alpha: float = 0.05,
    match: str = "range",
) -> ProtectionComparison:
    """The differences in ``log10_protection`` between each sample and ``reference``
    (:class:`ProtectionComparison`)."""
    from scipy import stats

    if match not in ("range", "sequence"):
        raise ValueError("match must be 'range' or 'sequence'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    def ident(i):  # as compare_peptides keys species: _pairs matches on the range, [2:]
        return (str(table.peptide[i]), str(table.modification[i]), int(table.start[i]),
                int(table.end[i]), int(table.K[i]), int(table.rank[i]))

    reference = resolve_sample(table.run, table.sample, reference, "reference sample")
    ref = {ident(i): i for i in np.flatnonzero(table.where(run=reference[0],
                                                           sample=reference[1]))}
    crit = float(stats.norm.ppf(1 - alpha / 2))
    cols = {f.name: [] for f in fields(ProtectionComparison)
            if f.name not in ("reference", "reference_run", "alpha")}
    pairs = []
    for state in states(table):
        if state == reference:
            continue
        other = {ident(i): i for i in np.flatnonzero(table.where(run=state[0],
                                                                 sample=state[1]))}
        pairs += [(ref[rkey], other[key]) for rkey, key in _pairs(ref, other, match)]
    for j, i in sorted(pairs, key=lambda p: p[1]):
        note = ""
        if table.bound[i] or table.bound[j]:
            note = "; ".join(f"{who} {table.bound[r]} bound" for who, r in
                             (("sample", i), ("reference", j)) if table.bound[r])
        d = float(table.log10_protection[i] - table.log10_protection[j])
        se = float(np.hypot(table.log10_protection_se[i], table.log10_protection_se[j]))
        z = d / se if np.isfinite(se) and se > 0 else np.nan
        for name, value in (("run", table.run[i]), ("sample", table.sample[i]),
                            ("peptide", table.peptide[i]),
                            ("modification", table.modification[i]), ("start", table.start[i]),
                            ("end", table.end[i]), ("K", table.K[i]), ("rank", table.rank[i]),
                            ("peptide_reference", table.peptide[j]), ("delta", d),
                            ("delta_se", se), ("z", z),
                            ("significant", bool(np.isfinite(z) and abs(z) > crit)),
                            ("note", note)):
            cols[name].append(value)
    kinds = {"start": np.int64, "end": np.int64, "K": np.int64, "rank": np.int64,
             "delta": float, "delta_se": float, "z": float, "significant": bool, "run": object}
    return ProtectionComparison(**{n: np.array(v, dtype=kinds.get(n, str))
                                   for n, v in cols.items()}, reference=reference[1],
                                reference_run=reference[0], alpha=alpha)
