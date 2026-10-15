# SPDX-License-Identifier: GPL-3.0-or-later
"""Peptide protection factors from MS1 uptake (docs/ANALYSIS.md, "MS1 protection factors").

Opt-in, with the labelling conditions (:class:`hxex.kint.ExchangeConditions`). An amide with
intrinsic rate ``kint`` (``kforw + kback`` in the labelling solvent, :mod:`hxex.kint`)
exchanges at ``kint / protection`` (EX2), so a peptide's back-exchange-corrected fraction
deuterated is modelled as ``mean_i(1 - exp(-kint_i t / protection))`` over the amides it
counts, with one protection factor per peptide. The name is always ``protection``, never
``P`` or ``p`` (the binomial parameter and the p-value; owner).

:func:`peptide_protection` gives, per sample, peptide, K and rank, the apparent protection
factor at each time (the value that gives the measured ``fD``) and one factor fitted over all
times in ``log10``, with bounds where the data hold only a limit. K = 2 ranks get apparent
values too: two populations come from EX1 or mixed exchange, where a protection factor has no
EX2 meaning. :func:`compare_protection` gives the differences between samples in
``log10_protection``.

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
from ..kint import ExchangeConditions, intrinsic_rates, sample_conditions
from .species import SpeciesUptake

log = logging.getLogger(__name__)

# fD beyond these at every time gives only a limit (owner's design)
LOW, HIGH = 0.05, 0.95
_LOG_RANGE = (-6.0, 15.0)  # log10 protection searched


def _columns(obj) -> dict[str, np.ndarray]:
    return {f.name: getattr(obj, f.name) for f in fields(obj)
            if isinstance(getattr(obj, f.name), np.ndarray)}


def _where(obj, **conditions) -> np.ndarray:
    keep = np.ones(len(obj), dtype=bool)
    for name, value in conditions.items():
        if value is not None:
            values = value if isinstance(value, (list, tuple, set, np.ndarray)) else [value]
            keep &= np.isin(getattr(obj, name), list(values))
    return keep


@dataclass(frozen=True)
class ProtectionTable:
    """One row per sample, peptide (``peptide``, ``modification``, ``start``, ``end``), ``K``
    and ``rank``: ``log10_protection`` and ``log10_protection_se`` (one factor fitted over
    every time; NaN for a bound), ``protection`` (``10**log10_protection``), ``bound``
    (``'upper'``: exchanged at every time, the factor is at most ``limit``; ``'lower'``: barely
    exchanged at every time, at least ``limit``; ``''``: a value), ``limit``, ``n_times``,
    ``protection_drift`` (log10 of the last apparent factor over the first, of the times
    between the limits; NaN with fewer than two: the amides of one peptide protected
    differently make it rise), ``misfit`` (reduced chi-square of the fit; NaN without errors
    or with one time) and ``one_factor_fits`` (``misfit`` within the chi-square limit at
    ``alpha``; False where it cannot be tested), ``n_amides`` and ``n_approximate`` (amides
    whose intrinsic rate borrows a factor: modified residues)."""

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

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return _columns(self)

    def where(self, **conditions) -> np.ndarray:
        return _where(self, **conditions)


@dataclass(frozen=True)
class ProtectionByTime:
    """The apparent protection factor of every sample, peptide, ``K``, ``rank`` and ``time``:
    the measured ``fD`` and ``fD_se``, ``log10_protection`` and its ``log10_protection_se``
    (from ``fD_se``), ``bound`` and ``limit`` as in :class:`ProtectionTable` (at that time)."""

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
    log10_protection: np.ndarray
    log10_protection_se: np.ndarray
    bound: np.ndarray
    limit: np.ndarray

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return _columns(self)

    def where(self, **conditions) -> np.ndarray:
        return _where(self, **conditions)


# --- intrinsic rates in the protein --------------------------------------------------------------

def _known_residues(uptake: SpeciesUptake, sample: str, registry) -> dict[int, str]:
    """Residue number -> modX label, from the sample's peptides."""
    known: dict[int, str] = {}
    rows = np.flatnonzero(uptake.where(sample=sample))
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
    return k, rates.approximate[at]


# --- the fits ------------------------------------------------------------------------------------

def _model(x: float, k: np.ndarray, t) -> np.ndarray:
    """``mean_i(1 - exp(-k_i t / 10**x))`` at the times ``t``."""
    return np.mean(-np.expm1(-np.multiply.outer(np.atleast_1d(t), k / 10.0**x)), axis=-1)


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


def _fit(k, t, y, se, apparent, alpha):
    """(log10 protection, its se, misfit, one factor fits) over every time."""
    from scipy import stats
    from scipy.optimize import least_squares

    weighted = np.all(np.isfinite(se) & (se > 0))
    w = 1.0 / se if weighted else np.ones_like(y)
    finite = np.isfinite(apparent)
    x0 = float(np.median(apparent[finite])) if finite.any() else 0.0
    res = least_squares(lambda x: (_model(x[0], k, t) - y) * w, [x0], bounds=_LOG_RANGE)
    x = float(res.x[0])
    n = len(y)
    jtj = float(res.jac[:, 0] @ res.jac[:, 0])
    chi2 = float(np.sum(res.fun**2))
    misfit = chi2 / (n - 1) if n > 1 and weighted else np.nan
    if jtj <= 0:
        return x, np.nan, misfit, False
    if weighted:
        scale = max(1.0, misfit) if np.isfinite(misfit) else 1.0
    else:  # no errors: the residuals' scatter
        scale = chi2 / (n - 1) if n > 1 else np.nan
    fits = bool(np.isfinite(misfit) and chi2 <= stats.chi2.ppf(1 - alpha, n - 1))
    return x, float(np.sqrt(scale / jtj)), misfit, fits


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
    docstring. Times at or below the undeuterated control's and those in ``exclude_times``
    (the fully deuterated controls') are left out. ``sequences`` (the protein's, one or per
    sample; :meth:`hxex.analysis.AnalysisConfig.sequences`) gives the residue before each
    peptide and the protein's C-terminus; without it the residue before comes from the other
    peptides, and no peptide ends at a free carboxyl. Samples without back-exchange correction,
    or without conditions, are left out with a warning, and so are peptides whose rates are not
    all known (a modified ion type, or the residue before unknown)."""
    pep = uptake.select(uptake.where(kind="peptide"))
    times = np.asarray(pep.time, dtype=float)
    keep = times > pep.undeut_time
    for t in exclude_times:
        keep &= ~np.isclose(times, t)
    pep = pep.select(keep)
    uncorrected = sorted({str(s) for s, c in zip(pep.sample, pep.corrected, strict=True)
                          if not c})
    if uncorrected:
        log.warning("no protection factors for sample(s) %s: not corrected for back-exchange",
                    ", ".join(uncorrected))
    no_conditions = sorted({str(s) for s in pep.sample} - set(uncorrected)
                           - {str(s) for s in pep.sample if sample_conditions(conditions, s)})
    if no_conditions:
        log.warning("no labelling conditions for sample(s) %s: no protection factors",
                    ", ".join(no_conditions))
    pep = pep.select(~np.isin(pep.sample, uncorrected + no_conditions))

    out = {f.name: [] for f in fields(ProtectionTable)}
    by_time = {f.name: [] for f in fields(ProtectionByTime)}
    skipped: dict[str, int] = {}
    groups: dict[tuple, list[int]] = {}
    for i in range(len(pep)):
        key = (str(pep.sample[i]), str(pep.peptide[i]), str(pep.modification[i]),
               int(pep.start[i]), int(pep.end[i]), int(pep.K[i]), int(pep.rank[i]))
        groups.setdefault(key, []).append(i)
    known_cache: dict[str, dict[int, str]] = {}
    rate_cache: dict[tuple, tuple | None] = {}
    for key, rows in groups.items():
        sample, peptide, mod, start, end = key[:5]
        rkey = key[:5]
        if rkey not in rate_cache:
            spec = chem.parse_modification(mod)
            if spec.ion_type != "M" or spec.hex:
                rate_cache[rkey] = None
            else:
                seq = (sequences.get(sample) if isinstance(sequences, Mapping) else sequences)
                if seq:
                    residues = chem.parse_sequence(seq, registry)
                    before = residues[start - 2] if start > 1 else None
                    last = end == len(residues)
                else:
                    if sample not in known_cache:
                        known_cache[sample] = _known_residues(pep, sample, registry)
                    before, last = known_cache[sample].get(start - 1), False
                k, approx = peptide_rates(peptide, start, sample_conditions(conditions, sample),
                                          before=before, protein_end=last,
                                          nterm_subtract=pep.nterm_subtract, registry=registry)
                n_amides = int(pep.n_amides[rows[0]])
                ok = len(k) == n_amides and len(k) > 0 and not np.isnan(k).any()
                rate_cache[rkey] = (k, int(approx.sum())) if ok else None
        if rate_cache[rkey] is None:
            skipped[sample] = skipped.get(sample, 0) + 1
            continue
        k, n_approx = rate_cache[rkey]
        rows = sorted(rows, key=lambda i: pep.time[i])
        t = pep.time[rows].astype(float)
        y = pep.fD[rows].astype(float)
        se = pep.fD_se[rows].astype(float)
        app = [_apparent(k, ti, yi, si) for ti, yi, si in zip(t, y, se, strict=True)]
        x_t = np.array([a[0] for a in app])
        for j, (ti, a) in enumerate(zip(t, app, strict=True)):
            for name, value in zip(("sample", "peptide", "modification", "start", "end", "K",
                                    "rank"), key, strict=True):
                by_time[name].append(value)
            for name, value in (("time", ti), ("fD", y[j]), ("fD_se", se[j]),
                                ("log10_protection", a[0]), ("log10_protection_se", a[1]),
                                ("bound", a[2]), ("limit", a[3])):
                by_time[name].append(value)
        finite = np.isfinite(y)
        bounds = [a[2] for a, f in zip(app, finite, strict=True) if f]
        x = x_se = misfit = np.nan
        fits, bound, limit = False, "", np.nan
        if bounds and all(b == "upper" for b in bounds):
            bound, limit = "upper", min(a[3] for a in app if a[2] == "upper")
        elif bounds and all(b == "lower" for b in bounds):
            bound, limit = "lower", max(a[3] for a in app if a[2] == "lower")
        elif finite.any():
            x, x_se, misfit, fits = _fit(k, t[finite], y[finite], se[finite], x_t[finite], alpha)
        inside = x_t[np.isfinite(x_t)]
        drift = float(inside[-1] - inside[0]) if len(inside) > 1 else np.nan
        for name, value in zip(("sample", "peptide", "modification", "start", "end", "K", "rank"),
                               key, strict=True):
            out[name].append(value)
        for name, value in (("log10_protection", x), ("log10_protection_se", x_se),
                            ("protection", 10.0**x), ("bound", bound), ("limit", limit),
                            ("n_times", int(finite.sum())), ("protection_drift", drift),
                            ("misfit", misfit), ("one_factor_fits", fits),
                            ("n_amides", len(k)), ("n_approximate", n_approx)):
            out[name].append(value)
    if skipped:
        log.warning("protection factors: %s peptide(s) left out (rates not all known: a modified "
                    "ion type, or the residue before unknown)",
                    ", ".join(f"{s}: {n}" for s, n in skipped.items()))
    ints = ("start", "end", "K", "rank", "n_times", "n_amides", "n_approximate")
    text = ("sample", "peptide", "modification", "bound")

    def array(name, values):
        if name in text:
            return np.array(values, dtype=str)
        if name == "one_factor_fits":
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

    def ident(i):
        pep = ((int(table.start[i]), int(table.end[i])) if match == "range"
               else (str(table.peptide[i]), str(table.modification[i])))
        return (*pep, int(table.K[i]), int(table.rank[i]))

    ref = {ident(i): i for i in np.flatnonzero(table.where(sample=reference))}
    crit = float(stats.norm.ppf(1 - alpha / 2))
    cols = {f.name: [] for f in fields(ProtectionComparison) if f.name not in ("reference",
                                                                              "alpha")}
    for i in np.flatnonzero(~table.where(sample=reference)):
        j = ref.get(ident(i))
        if j is None:
            continue
        note = ""
        if table.bound[i] or table.bound[j]:
            note = "; ".join(f"{who} {table.bound[r]} bound" for who, r in
                             (("sample", i), ("reference", j)) if table.bound[r])
        d = float(table.log10_protection[i] - table.log10_protection[j])
        se = float(np.hypot(table.log10_protection_se[i], table.log10_protection_se[j]))
        z = d / se if np.isfinite(se) and se > 0 else np.nan
        for name, value in (("sample", table.sample[i]), ("peptide", table.peptide[i]),
                            ("modification", table.modification[i]), ("start", table.start[i]),
                            ("end", table.end[i]), ("K", table.K[i]), ("rank", table.rank[i]),
                            ("peptide_reference", table.peptide[j]), ("delta", d),
                            ("delta_se", se), ("z", z),
                            ("significant", bool(np.isfinite(z) and abs(z) > crit)),
                            ("note", note)):
            cols[name].append(value)
    kinds = {"start": np.int64, "end": np.int64, "K": np.int64, "rank": np.int64,
             "delta": float, "delta_se": float, "z": float, "significant": bool}
    return ProtectionComparison(**{n: np.array(v, dtype=kinds.get(n, str))
                                   for n, v in cols.items()}, reference=reference, alpha=alpha)
