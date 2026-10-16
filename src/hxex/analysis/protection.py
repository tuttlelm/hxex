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
from ..hooks import CancelHook, ProgressHook, check_cancel, report
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


def _prepare(uptake, conditions, sequences, exclude_times, registry):
    """The peptides of ``uptake`` that get protection factors: (the table, its rows grouped by
    sample, peptide, modification, start, end, K and rank, the rates of each peptide by its
    first five keys (``(kint of the counted amides, how many approximate, their residue
    numbers)``, None where not all known), each sample's residue labels by number)."""
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
    groups: dict[tuple, list[int]] = {}
    for i in range(len(pep)):
        key = (str(pep.sample[i]), str(pep.peptide[i]), str(pep.modification[i]),
               int(pep.start[i]), int(pep.end[i]), int(pep.K[i]), int(pep.rank[i]))
        groups.setdefault(key, []).append(i)
    labels: dict[str, dict[int, str]] = {}
    for sample in dict.fromkeys(pep.sample.tolist()):
        seq = sequences.get(sample) if isinstance(sequences, Mapping) else sequences
        labels[sample] = ({j + 1: lab for j, lab in enumerate(chem.parse_sequence(seq, registry))}
                          if seq else _known_residues(pep, sample, registry))
    rates: dict[tuple, tuple | None] = {}
    skipped: dict[str, int] = {}
    for key, rows in groups.items():
        sample, peptide, mod, start, end = rkey = key[:5]
        if rkey in rates:
            continue
        rates[rkey] = None
        spec = chem.parse_modification(mod)
        if spec.ion_type == "M" and not spec.hex:
            seq = sequences.get(sample) if isinstance(sequences, Mapping) else sequences
            last = bool(seq) and end == len(labels[sample])
            k, approx = peptide_rates(peptide, start, sample_conditions(conditions, sample),
                                      before=labels[sample].get(start - 1), protein_end=last,
                                      nterm_subtract=pep.nterm_subtract, registry=registry)
            pos = np.array(chem.amide_positions(peptide, "M", None,
                                                nterm_subtract=pep.nterm_subtract,
                                                registry=registry), dtype=np.int64)
            if len(k) == int(pep.n_amides[rows[0]]) and len(k) and not np.isnan(k).any():
                rates[rkey] = (k, int(approx.sum()), start + pos - 1)
        if rates[rkey] is None:
            skipped[sample] = skipped.get(sample, 0) + 1
    if skipped:
        log.warning("protection factors: %s peptide(s) left out (rates not all known: a modified "
                    "ion type, or the residue before unknown)",
                    ", ".join(f"{s}: {n}" for s, n in skipped.items()))
    return pep, groups, rates, labels


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
    pep, groups, rates, _ = _prepare(uptake, conditions, sequences, exclude_times, registry)
    out = {f.name: [] for f in fields(ProtectionTable)}
    by_time = {f.name: [] for f in fields(ProtectionByTime)}
    for key, rows in groups.items():
        if rates[key[:5]] is None:
            continue
        k, n_approx, _ = rates[key[:5]]
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


# --- residue level -------------------------------------------------------------------------------

R1 = 0.1  # default regulariser weight (log10 units; chosen on synthetic data, HISTORY item 82)
SE_FLOOR = 0.01  # fD standard errors below this are raised to it
HOP_PATIENCE = 40  # basin hops without improvement before the search stops


@dataclass(frozen=True)
class ResidueProtection:
    """One row per state (``sample``, ``K``, ``rank``) and fitted residue
    (:func:`residue_protection`): ``residue``, ``aa``, ``log10_protection`` and
    ``log10_protection_se`` (for the regulariser weight ``r1`` used; NaN for a bound),
    ``protection``, ``bound`` and ``limit`` (every covering peptide bounded the same way),
    ``resolved`` (the peptides alone separate this residue from its neighbours; elsewhere the
    value is the regulariser's share-out) and ``n_peptides`` (covering it). ``misfit`` is the
    state's reduced chi-square, repeated on its rows."""

    sample: np.ndarray
    K: np.ndarray
    rank: np.ndarray
    residue: np.ndarray
    aa: np.ndarray
    log10_protection: np.ndarray
    log10_protection_se: np.ndarray
    protection: np.ndarray
    bound: np.ndarray
    limit: np.ndarray
    resolved: np.ndarray
    n_peptides: np.ndarray
    misfit: np.ndarray
    r1: float = R1

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return _columns(self)

    def where(self, **conditions) -> np.ndarray:
        return _where(self, **conditions)


def _resolved(member: np.ndarray) -> np.ndarray:
    """Whether each residue's unit vector lies in the span of the peptides' coverage rows."""
    _, sv, vt = np.linalg.svd(member, full_matrices=False)
    keep = sv > sv.max() * 1e-9 if len(sv) else sv > 0
    return np.sum(vt[keep] ** 2, axis=0) > 0.999


def _fit_residues(peps, obs, k, pairs, r1, x0, cancel=None):
    """Minimise the loss of the design (docs/ANALYSIS.md); returns (x, se, misfit)."""
    import torch

    n_res = len(k)
    member = torch.zeros((len(peps), n_res), dtype=torch.float64)
    for j, (idx, _, _) in enumerate(peps):
        member[j, idx] = 1.0
    const = torch.tensor([c for _, c, _ in peps], dtype=torch.float64)
    n_am = torch.tensor([n for _, _, n in peps], dtype=torch.float64)
    which = torch.tensor(obs[0], dtype=torch.long)
    t = torch.tensor(obs[1], dtype=torch.float64)
    y = torch.tensor(obs[2], dtype=torch.float64)
    w = 1.0 / torch.tensor(np.maximum(np.nan_to_num(obs[3], nan=SE_FLOOR), SE_FLOOR),
                           dtype=torch.float64)
    kk = torch.tensor(k, dtype=torch.float64)
    a = torch.tensor(pairs[:, 0], dtype=torch.long) if len(pairs) else None
    b = torch.tensor(pairs[:, 1], dtype=torch.long) if len(pairs) else None

    def parts(x):
        rate = kk / 10.0**x
        held = (1.0 - torch.exp(-t[:, None] * rate[None, :])) * member[which]
        pred = (held.sum(1) + const[which]) / n_am[which]
        chi2 = torch.sum(((pred - y) * w) ** 2)
        reg = (torch.sqrt((x[a] - x[b]) ** 2 + 1e-4).mean() if a is not None
               else torch.zeros((), dtype=torch.float64))
        return chi2, reg

    floor, ceiling = _LOG_RANGE[0] + 3.0, _LOG_RANGE[1] - 1.0  # -3 to 14: a wall beyond

    def loss(x):
        chi2, reg = parts(x)
        wall = torch.sum(torch.relu(x - ceiling) ** 2 + torch.relu(floor - x) ** 2)
        return chi2 / len(y) + r1 * reg + wall

    def polish(start, iters):
        """L-BFGS from ``start``; never worse than ``start`` (a failed line search can be)."""
        x0_ = torch.tensor(start, dtype=torch.float64)
        with torch.no_grad():
            start_value = float(loss(x0_))
        x = x0_.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([x], lr=1.0, max_iter=iters, tolerance_grad=1e-10,
                                tolerance_change=1e-14, history_size=30,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            value = loss(x)
            value.backward()
            return value

        opt.step(closure)
        x = x.detach()
        with torch.no_grad():
            value = float(loss(x))
        if not np.isfinite(value) or value > start_value:
            return x0_, start_value
        return x, value

    # the loss is not convex (saturating exponentials; neighbours can trade values), so after a
    # first descent, windows of neighbouring residues are given new values and the descent is
    # kept when it ends lower (basin hopping; seeded, so the result is reproducible)
    rng = np.random.default_rng(0)
    x, best = polish(np.asarray(x0, dtype=float), 2000)
    stale, hops = 0, max(60, 2 * n_res)
    for _ in range(hops):
        if stale >= HOP_PATIENCE:
            break
        check_cancel(cancel)
        width = int(rng.integers(2, 6))
        if rng.random() < 0.5:  # around the worst-fitting peptide's residues
            with torch.no_grad():
                rate = kk / 10.0**x
                held = (1.0 - torch.exp(-t[:, None] * rate[None, :])) * member[which]
                resid = (((held.sum(1) + const[which]) / n_am[which] - y) * w) ** 2
                per_pep = torch.zeros(len(peps), dtype=torch.float64).index_add_(0, which, resid)
            idx = peps[int(torch.argmax(per_pep))][0]
            centre = int(rng.choice(idx)) if len(idx) else int(rng.integers(0, n_res))
        else:
            centre = int(rng.integers(0, n_res))
        first = max(0, min(centre - width // 2, n_res - width))
        trial = x.clone()
        trial[first:first + width] = torch.tensor(rng.uniform(-1.0, 8.0, min(width, n_res)))
        trial, value = polish(trial.numpy(), 30)
        if value < best - 1e-9:
            x, best, stale = trial, value, 0
        else:
            stale += 1
    x, best = polish(x.numpy(), 2000)
    x = x.clone().requires_grad_(False)
    xf = x
    chi2 = float(parts(xf)[0])
    misfit = chi2 / max(1, len(y) - n_res)
    hess = torch.autograd.functional.hessian(lambda v: len(y) * loss(v), xf)
    if not bool(torch.isfinite(hess).all()):
        return xf.numpy(), np.full(n_res, np.nan), misfit
    try:
        cov = 2.0 * torch.linalg.inv(hess)
    except RuntimeError:
        cov = 2.0 * torch.linalg.pinv(hess)
    var = torch.diagonal(cov).numpy() * max(1.0, misfit)
    se = np.where(var > 0, np.sqrt(np.abs(var)), np.nan)
    return xf.numpy(), se, misfit


def residue_protection(
    uptake: SpeciesUptake,
    conditions: ExchangeConditions | Mapping[str, ExchangeConditions],
    *,
    sequences: str | Mapping[str, str] | None = None,
    exclude_times=(),
    r1: float = R1,
    peptides: ProtectionTable | None = None,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> ResidueProtection:
    """One protection factor per exchanging residue of every state (a sample at K = 1, or a
    K = 2 rank: apparent values), fitted to all its peptides at once with a total-variation
    regulariser of weight ``r1`` (docs/ANALYSIS.md, "Residue level"; pyHDX's
    ``fit_gibbs_global`` is the reference). Inputs and the peptides used are those of
    :func:`peptide_protection`; ``peptides`` (its table, computed if not given) gives the
    starting values and the bounds. ``progress(done, total)`` counts the states; ``cancel``
    stops the fit (:class:`hxex.hooks.Cancelled`)."""
    if not r1 >= 0:
        raise ValueError("r1 must be >= 0")
    if peptides is None:
        peptides = peptide_protection(uptake, conditions, sequences=sequences,
                                      exclude_times=exclude_times, registry=registry)[0]
    pep, groups, rates, labels = _prepare(uptake, conditions, sequences, exclude_times,
                                          registry)
    start_value = {}
    for i in range(len(peptides)):
        key = (str(peptides.sample[i]), str(peptides.peptide[i]), str(peptides.modification[i]),
               int(peptides.start[i]), int(peptides.end[i]), int(peptides.K[i]),
               int(peptides.rank[i]))
        x = peptides.log10_protection[i]
        start_value[key] = ((float(x) if np.isfinite(x) else
                             float(np.log10(peptides.limit[i])) if peptides.bound[i] else np.nan),
                            str(peptides.bound[i]), float(peptides.limit[i]))
    states = sorted({(k[0], k[5], k[6]) for k in groups if rates[k[:5]] is not None},
                    key=lambda s: (list(labels).index(s[0]), s[1], s[2]))
    cols = {f.name: [] for f in fields(ResidueProtection) if f.name != "r1"}
    for n_done, (sample, kk_, rank) in enumerate(states):
        report(progress, n_done, len(states))
        check_cancel(cancel)
        keys = [k for k in groups if (k[0], k[5], k[6]) == (sample, kk_, rank)
                and rates[k[:5]] is not None]
        rate_of: dict[int, float] = {}
        for key in keys:
            k, _, res = rates[key[:5]]
            for r, kr in zip(res.tolist(), k.tolist(), strict=True):
                if np.isfinite(kr):
                    rate_of.setdefault(r, kr)
        residues = sorted(rate_of)
        if not residues:
            continue
        col = {r: j for j, r in enumerate(residues)}
        peps, obs = [], ([], [], [], [])
        member = np.zeros((len(keys), len(residues)))
        covering: dict[int, list[tuple]] = {r: [] for r in residues}
        for j, key in enumerate(keys):
            k, _, res = rates[key[:5]]
            idx = [col[r] for r, kr in zip(res.tolist(), k.tolist(), strict=True)
                   if np.isfinite(kr)]
            member[j, idx] = 1.0
            peps.append((idx, float(np.sum(~np.isfinite(k))), float(len(k))))
            for r in idx:
                covering[residues[r]].append(start_value.get(key, (np.nan, "", np.nan)))
            for i in groups[key]:
                if np.isfinite(pep.fD[i]):
                    for target, value in zip(obs, (j, pep.time[i], pep.fD[i], pep.fD_se[i]),
                                             strict=True):
                        target.append(float(value))
        if not obs[0]:
            continue
        pairs = []
        for r0, r1_ in zip(residues[:-1], residues[1:], strict=True):
            gap = range(r0 + 1, r1_)
            if all(labels[sample].get(g) is not None
                   and registry.base_residue(labels[sample][g]) == "P" for g in gap):
                pairs.append((col[r0], col[r1_]))
        x0 = []
        for r in residues:
            vals = [v[0] for v in covering[r] if np.isfinite(v[0])]
            x0.append(float(np.mean(vals)) if vals else 2.0)
        x, se, misfit = _fit_residues(peps, (np.array(obs[0], int), *map(np.array, obs[1:])),
                                      np.array([rate_of[r] for r in residues]),
                                      np.array(pairs, dtype=np.int64).reshape(-1, 2), r1, x0,
                                      cancel)
        resolved = _resolved(member)
        for j, r in enumerate(residues):
            bounds = {v[1] for v in covering[r]}
            bound, limit, value, err = "", np.nan, x[j], se[j]
            if len(bounds) == 1 and "" not in bounds:
                bound = bounds.pop()
                limits = [v[2] for v in covering[r]]
                limit = min(limits) if bound == "upper" else max(limits)
                value = err = np.nan
            for name, v in (("sample", sample), ("K", kk_), ("rank", rank), ("residue", r),
                            ("aa", labels[sample].get(r, "")), ("log10_protection", value),
                            ("log10_protection_se", err), ("protection", 10.0**value),
                            ("bound", bound), ("limit", limit),
                            ("resolved", bool(resolved[j])),
                            ("n_peptides", len(covering[r])), ("misfit", misfit)):
                cols[name].append(v)
        log.info("residue protection %s K=%d rank %d: %d residues, %d resolved, misfit %.3g "
                 "(r1 %g)", sample, kk_, rank, len(residues), int(resolved.sum()), misfit, r1)
    report(progress, len(states), len(states))
    kinds = {"K": np.int64, "rank": np.int64, "residue": np.int64, "log10_protection": float,
             "log10_protection_se": float, "protection": float, "limit": float,
             "resolved": bool, "n_peptides": np.int64, "misfit": float}
    return ResidueProtection(**{n: np.array(v, dtype=kinds.get(n, str)) for n, v in cols.items()},
                             r1=r1)


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


@dataclass(frozen=True)
class ResidueProtectionComparison:
    """Fitted residue ``log10_protection`` of each sample against the ``reference`` sample's,
    per ``K``, ``rank`` and ``residue``: ``delta`` (log10 units), ``delta_se``, ``z`` and
    ``significant`` (beyond the normal quantile of ``alpha``, two-sided; never where either
    side is not resolved: that value is the regulariser's share-out), ``resolved`` (both
    sides) and ``note`` (a bound, or not resolved)."""

    sample: np.ndarray
    K: np.ndarray
    rank: np.ndarray
    residue: np.ndarray
    aa: np.ndarray
    delta: np.ndarray
    delta_se: np.ndarray
    z: np.ndarray
    significant: np.ndarray
    resolved: np.ndarray
    note: np.ndarray
    reference: str = ""
    alpha: float = 0.05

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return _columns(self)

    def where(self, **conditions) -> np.ndarray:
        return _where(self, **conditions)


def compare_residue_protection(
    table: ResidueProtection,
    reference: str,
    *,
    alpha: float = 0.05,
) -> ResidueProtectionComparison:
    """The differences in fitted residue ``log10_protection`` between each sample and
    ``reference`` (:class:`ResidueProtectionComparison`)."""
    from scipy import stats

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    ref = {(int(table.K[i]), int(table.rank[i]), int(table.residue[i])): i
           for i in np.flatnonzero(table.where(sample=reference))}
    crit = float(stats.norm.ppf(1 - alpha / 2))
    cols = {f.name: [] for f in fields(ResidueProtectionComparison)
            if f.name not in ("reference", "alpha")}
    for i in np.flatnonzero(~table.where(sample=reference)):
        j = ref.get((int(table.K[i]), int(table.rank[i]), int(table.residue[i])))
        if j is None:
            continue
        notes = [f"{who} {table.bound[r]} bound" for who, r in (("sample", i), ("reference", j))
                 if table.bound[r]]
        notes += [f"{who} not resolved" for who, r in (("sample", i), ("reference", j))
                  if not table.resolved[r]]
        resolved = bool(table.resolved[i] and table.resolved[j])
        d = float(table.log10_protection[i] - table.log10_protection[j])
        se = float(np.hypot(table.log10_protection_se[i], table.log10_protection_se[j]))
        z = d / se if np.isfinite(se) and se > 0 else np.nan
        for name, value in (("sample", table.sample[i]), ("K", table.K[i]),
                            ("rank", table.rank[i]), ("residue", table.residue[i]),
                            ("aa", table.aa[i]), ("delta", d), ("delta_se", se), ("z", z),
                            ("significant", bool(resolved and np.isfinite(z) and abs(z) > crit)),
                            ("resolved", resolved), ("note", "; ".join(notes))):
            cols[name].append(value)
    kinds = {"K": np.int64, "rank": np.int64, "residue": np.int64, "delta": float,
             "delta_se": float, "z": float, "significant": bool, "resolved": bool}
    return ResidueProtectionComparison(**{n: np.array(v, dtype=kinds.get(n, str))
                                          for n, v in cols.items()},
                                       reference=reference, alpha=alpha)
