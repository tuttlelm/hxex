# SPDX-License-Identifier: GPL-3.0-or-later
"""Undeuterated and fully deuterated controls and the back-exchange correction.

A series (one peptide at one charge, ``data_id``) has its controls among its spectra by their
role, ``undeut`` and ``fulldeut`` (:data:`hxex.spectra.ROLES`; set by the readers from the
formats' control labels, by ``role`` in ``[[input.runs]]``, or by :func:`apply_control_roles`
for a time point that also serves as a control, ``undeut_at`` and ``fulldeut_at``); replicate
controls are averaged. The control means ``m0`` and
``mFD`` are the means of the K=1 fits of the control spectra, in the model's units: deuterons
beyond natural abundance (the natural-abundance shape is convolved separately, so an undeuterated
spectrum has ``m0`` near 0, and the centre of the raw envelope must not be used instead).

Fragment ions (the series of ``uncorrected``: an ion type other than ``M``, or a fragment
scan target) are not corrected: their row keeps the measured control means, uses ``m0 = 0``
and ``mFD = n_amides`` (so the correction leaves their values as they are; :func:`correct_means`
passes them through), is flagged ``uncorrected`` and takes no part in the run's limits,
median or theoretical fit (owner, 2026-10-21: the fragments' correction may belong at the
parent level, still to be designed).

Each control falls back to its theoretical value, and the series is flagged, on its own:

- ``m0`` falls back to 0 when it is missing (``m0_missing``) or above ``m0_max`` (``m0_high``);
- ``mFD`` falls back to ``m0 + r * n_amides`` when it is missing (``fd_missing``), when the
  retention ``(mFD - m0) / n_amides`` is 0 or less or below a set ``retention_min``
  (``retention_low``), above ``retention_max`` (``retention_high``), below the highest time
  point of its series (``fd_below_data``, beyond ``time_course_tolerance``), or, without a set
  ``retention_min``, an outlier below the run's controls (``retention_outlier``,
  ``outlier_sd``; with ``fd_fallback='kint'`` only for the series the prediction cannot judge,
  those without rates: fragment ions and ``Hex`` series, or for every series when the
  prediction could not be fitted), with ``m0`` as it is used. With ``combine_charges`` (the
  default) every charge state of a peptide in one sample takes the plain mean retention of
  the peptide's accepted measured controls over its charge states, whether its own control
  is good or not (``fd_charges_combined``, ``fd_other_charge``; a series without its own
  then counts as measured); without it, a series whose control is missing or rejected
  takes the mean of the other charge states' (``fd_other_charge``, still a fallback;
  ``fd_other_charge = False`` turns it off). Never for fragment ions, whose conditions are
  never pooled. Otherwise the fallback
  retention ``r`` is ``d2o_fraction`` (``fd_fallback='d2o'``, the deuterium fraction of the
  labelling buffer, a user setting) or the median retention of the run's accepted controls
  (``fd_fallback='median'``, the default without ``quench`` conditions; flagged ``fd_median``;
  ``d2o_fraction`` when no control is accepted), which includes the back-exchange the buffer's
  fraction does not.

With ``fd_fallback='kint'`` (the default when the ``quench`` conditions are given, which it
needs) the fully deuterated value comes from theoretical back-exchange instead: every amide of
the peptide keeps ``a * exp(-k_i * t0)`` of a deuteron, ``k_i`` its intrinsic rate at the quench
conditions (:mod:`hxex.kint`), with the labelling fraction ``a``
and an effective time ``t0`` fitted to the run's measured fully deuterated controls (robustly,
then again without the rejected ones). A measured control is rejected when it misses that
prediction by more than ``kint_tolerance`` deuterons per amide (``fd_off_model``), in place of
the ``retention_min`` floor (which still applies when set); a missing or rejected control falls
back to ``m0 + prediction`` (flagged ``fd_kint``) with the fit's scatter per amide times
``n_amides`` as its standard error. Series without rates (extra ``Hex``
sites) take the median retention of the run's accepted controls (``fd_median``) instead, and
so does every series when fewer than ``kint_min_series`` controls are usable (or agree with
the fit) and nothing is fitted (logged; owner, 2026-10-21: the median predicts held-out
controls far better than ``d2o_fraction``, HISTORY 131). When every
control in the fit has a retention
time (``RT``, minutes; :func:`series_rt`) for at least ``kint_min_series`` of the controls, the
effective time is ``t0 + b RT`` (``b`` fitted, not below 0), and a
series without one is taken at the median RT of those that have one.
The fitted model is :attr:`ControlTable.fd_model`.

A measured control with a single replicate has no uncertainty (flagged ``m0_unreplicated`` or
``fd_unreplicated``); a theoretical value is exact. Two flags only warn and leave the control
in use: ``fd_disagrees`` (its retention differs from the same peptide's in the other samples
by more than ``sample_tolerance``; :func:`series_groups`) and ``fd_multimodal`` or
``m0_multimodal`` (most of a control's replicate spectra chose more than one population: an
incompletely deuterated or a mixed sample), and ``data_below_m0`` (a time point of the series
reads below its undeuterated control by more than ``time_course_tolerance``: that time point
is suspect). The corrected mean of a population is
``(mean - m0) / (mFD - m0) * n_amides``, not below 0 (:func:`correct_means`); the uncertainty of
the controls is added to the uncertainty of the mean.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import norm

from .fit.result import FitResult
from .io.times import format_time
from .kint import ExchangeConditions
from .spectra import FULLDEUT, UNDEUT, time_point

log = logging.getLogger(__name__)

FD_FALLBACKS = ("d2o", "median", "kint")
# flags that only warn (the control stays in use)
CONTROL_NOTES = ("fd_disagrees", "fd_multimodal", "m0_multimodal", "data_below_m0",
                 "fd_retention_agrees", "fd_charges_disagree", "m0_replicate_left_out",
                 "fd_replicate_left_out", "fd_charges_agree")


@dataclass(frozen=True)
class ControlConfig:
    """The limits for a good control and the fallbacks (the ``[run.controls]`` table).
    "Retention" here is deuterium retention, the fraction of the label a fully deuterated
    control kept, not a retention time.

    - ``undeut_at``, ``fulldeut_at`` (s): the spectra measured at this time also serve as the
      undeuterated or the fully deuterated control (a plateaued late time point as the fully
      deuterated reference, the equilibrium end point of a back-exchange run as the
      undeuterated one); None: only the spectra the readers marked. The controls themselves
      are the spectra with that role (:mod:`hxex.spectra`), never a time.
    - ``undeut_marked``, ``fulldeut_marked``: whether the spectra marked as that control by
      the readers or ``[[input.runs]]`` serve as it (the default). False, with the matching
      ``*_at``, leaves them out: the control is then the spectra at that time alone (a
      measured fully deuterated file that reads low, replaced by the plateau of the time
      course). Those spectra stay fitted and keep their role in the tables.
    - ``m0_max`` (deuterons): the largest mean of a good undeuterated control.
    - ``retention_min``, ``retention_max`` (fractions): the range of a good fully deuterated
      control's retention ``(mFD - m0) / n_amides``. ``retention_min`` None (the default) sets
      no fixed floor: a control is then judged against the run (``outlier_sd``) or, with
      ``fd_fallback = 'kint'``, against the prediction.
    - ``outlier_sd``: with ``retention_min`` unset, a fully deuterated control whose retention
      is more than this many robust standard deviations (1.4826 times the median absolute
      deviation) below the median of the run's controls is rejected (``retention_outlier``);
      judged only when at least five controls pass the other checks. Retention depends on the
      workflow and on the peptide (its sequence, its time in the LC), so a fixed floor would
      reject good controls of a workflow with much back-exchange.
    - ``time_course_tolerance`` (deuterons per amide): a fully deuterated control below the
      highest time point of its series (the median of its replicates' K=1 means, uncorrected)
      by more than this times ``n_amides`` is rejected (``fd_below_data``: it did not fully
      deuterate); an undeuterated control above the lowest time point by more than this is
      only flagged (``data_below_m0``: the early, fast time points are the hard ones to
      measure, so the time point is suspect, not the control).
    - ``sample_tolerance`` (fraction): a measured fully deuterated control whose retention
      differs from the median of the same peptide's controls in the other samples (or
      conditions) by more than this is flagged ``fd_disagrees``, not rejected.
    - ``combine_charges`` (the default): every charge state of a peptide in one sample takes
      the plain mean retention of the peptide's accepted measured fully deuterated controls
      over its charge states (flagged ``fd_charges_combined`` where two or more contribute):
      retention is a property of the peptide, not of its charge (HISTORY 130 and 140). A
      charge state without a good control of its own takes it too, flagged
      ``fd_other_charge``, and counts as measured (not ``fallback``); ``fd_fallback`` applies
      only to a peptide with no accepted measured control at any charge. Contributing controls
      whose retentions span more than ``sample_tolerance`` are combined and flagged
      ``fd_charges_disagree`` (a warning; owner, 2026-10-22). The undeuterated control stays
      per charge state. Peptides only (not fragment ions), and only with
      :func:`charge_groups` given to :func:`fit_controls`.
    - ``replicate_outlier``: a control replicate whose K=1 mean lies more than this many
      typical replicate standard deviations (in deuterons per amide, from the run's time-point
      replicates: the median of their chi-square-scaled variances) from the median of the
      series' replicates of that control (three or more) is left out of its mean, flagged
      ``m0_replicate_left_out`` or ``fd_replicate_left_out`` (a warning); None turns it off. A
      control replicate whose K=1 mean is above the amide count is always left out (the same
      flags): no deuterium level explains it (owner, 2026-10-22, HISTORY 143).
    - ``fd_other_charge`` (with ``combine_charges = false``): a series whose fully deuterated
      control is missing or rejected takes the mean retention of the same peptide's accepted
      measured controls at its other charge states in the same sample (flagged
      ``fd_other_charge``, still ``fallback``), before ``fd_fallback``; the others keep their
      own.
    - ``d2o_fraction``: the deuterium atom fraction of the labelling buffer (often 0.8 to 0.9);
      the analysis may change it for its own controls (``[analysis.controls]``).
    - ``fd_fallback``: the retention used when a series' fully deuterated control is missing or
      rejected: ``'d2o'`` (``d2o_fraction``), ``'median'`` (the median of the run's accepted
      controls; ``d2o_fraction`` if there are none) or ``'kint'`` (theoretical back-exchange
      fitted to the run's controls, see the module docstring; needs ``quench``). None (the
      default) means ``'kint'`` when ``quench`` is given and ``'median'`` otherwise
      (:attr:`fd_mode`).
    - ``quench``: the conditions of the back-exchange (:class:`hxex.kint.ExchangeConditions`):
      pH read and temperature of the quench and LC, and the ``d2o_fraction`` of their solvent
      (about 0).
    - ``kint_tolerance`` (deuterons per amide): with ``'kint'``, a measured control this far
      from the prediction is rejected; ``kint_min_series``: the fewest usable controls to fit
      the prediction to.
    """
    # the limits' defaults, the fallbacks and the kint model: docs/HISTORY.md items 27, 59, 61, 71,
    # 73 and 126

    undeut_at: float | None = None
    fulldeut_at: float | None = None
    undeut_marked: bool = True
    fulldeut_marked: bool = True
    m0_max: float = 0.5
    retention_min: float | None = None
    retention_max: float = 1.1
    outlier_sd: float = 4.0
    time_course_tolerance: float = 0.1
    sample_tolerance: float = 0.1
    combine_charges: bool = True
    fd_other_charge: bool = True
    replicate_outlier: float | None = 6.0
    d2o_fraction: float = 0.9
    fd_fallback: str | None = None
    quench: ExchangeConditions | None = None
    kint_tolerance: float = 0.15
    kint_min_series: int = 10

    def __post_init__(self):
        if isinstance(self.quench, Mapping):
            object.__setattr__(self, "quench", ExchangeConditions.from_dict(self.quench))
        if self.fd_mode == "kint" and self.quench is None:
            raise ValueError("fd_fallback 'kint' needs the quench conditions (quench: ph_read, "
                             "temperature, d2o_fraction)")
        if not self.kint_tolerance > 0:
            raise ValueError("kint_tolerance must be positive")
        if self.kint_min_series < 3:
            raise ValueError("kint_min_series must be at least 3")
        if self.m0_max < 0:
            raise ValueError("m0_max must be >= 0")
        for name in ("outlier_sd", "time_course_tolerance", "sample_tolerance"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.min_retention < self.retention_max:
            raise ValueError("need 0 <= retention_min < retention_max")
        if self.fd_fallback is not None and self.fd_fallback not in FD_FALLBACKS:
            raise ValueError(f"fd_fallback must be one of {', '.join(FD_FALLBACKS)}")
        for name in ("undeut_at", "fulldeut_at"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not np.isfinite(float(value)):
                    raise ValueError(f"{name} must be a time in seconds or None")
                object.__setattr__(self, name, float(value))
        if (self.undeut_at is not None and self.fulldeut_at is not None
                and self.undeut_at == self.fulldeut_at):
            raise ValueError("undeut_at and fulldeut_at cannot be the same time")
        for name in ("undeut", "fulldeut"):
            if not getattr(self, f"{name}_marked") and getattr(self, f"{name}_at") is None:
                raise ValueError(f"{name}_marked = false needs {name}_at (the spectra that "
                                 "serve as the control instead of the marked ones)")
        if not 0 < self.d2o_fraction <= 1:
            raise ValueError("d2o_fraction must be in (0, 1]")
        if not isinstance(self.fd_other_charge, bool):
            raise ValueError("fd_other_charge must be true or false")
        if not isinstance(self.combine_charges, bool):
            raise ValueError("combine_charges must be true or false")
        if self.replicate_outlier is not None and (isinstance(self.replicate_outlier, bool)
                                                   or not self.replicate_outlier > 0):
            raise ValueError("replicate_outlier must be a positive number of standard "
                             "deviations (or unset)")

    @property
    def fd_mode(self) -> str:
        """The fallback in use: ``fd_fallback``, or when unset ``'kint'`` with quench conditions
        and ``'median'`` without."""
        if self.fd_fallback is not None:
            return self.fd_fallback
        return "kint" if self.quench is not None else "median"

    @property
    def min_retention(self) -> float:
        """The fixed lower retention limit (``retention_min``; unset: 0, the controls being
        judged against the run, ``outlier_sd``, or the prediction; the limit a run used is in
        :attr:`ControlTable.limits`)."""
        return 0.0 if self.retention_min is None else self.retention_min

    def to_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["quench"] = None if self.quench is None else self.quench.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ControlConfig:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown & set(LEGACY_SETTINGS):
            raise ValueError(
                f"control setting(s) {', '.join(sorted(unknown & set(LEGACY_SETTINGS)))} no "
                "longer exist: the controls are the spectra with a role (undeut, fulldeut), "
                "set by the formats' control labels or role in [[input.runs]]; undeut_at and "
                "fulldeut_at make the spectra at a time serve as a control as well")
        if unknown:
            raise ValueError(f"unknown control setting(s): {', '.join(sorted(unknown))}")
        return cls(**data)


LEGACY_SETTINGS = ("undeut_time", "fd_time", "atol")  # the control times before the roles


def apply_control_roles(role, time, config: ControlConfig, *, where: str = "") -> np.ndarray:
    """``role`` (one per spectrum) with the spectra at ``config.undeut_at`` marked
    ``'undeut'`` and those at ``config.fulldeut_at`` marked ``'fulldeut'`` (they keep their
    time). The match is exact (times are compared as :func:`hxex.spectra.time_point` keys,
    to the nanosecond). A time no spectrum has is an error listing the times present."""
    role = np.array(np.asarray(role, dtype=str), dtype="<U8")
    time = np.asarray(time, dtype=float)
    if len(role) != len(time):
        raise ValueError("role and time must have one entry per spectrum")
    points = time_point(time)
    for name, label in (("undeut_at", UNDEUT), ("fulldeut_at", FULLDEUT)):
        at = getattr(config, name)
        if at is None:
            continue
        hit = points == time_point([at])[0]
        if not hit.any():
            present = sorted({float(t) for t in time[np.isfinite(time)]})
            raise ValueError(
                f"{where + ': ' if where else ''}{name} = {format_time(at)}: no spectrum was "
                f"measured at that time (times present: "
                f"{', '.join(format_time(t) for t in present[:20])}"
                f"{', ...' if len(present) > 20 else ''})")
        other = hit & (role != "") & (role != label)
        if other.any():
            raise ValueError(f"{name} = {format_time(at)}: {int(other.sum())} spectra at that "
                             f"time are already {role[other][0]} controls")
        role[hit] = label
    return role


@dataclass(frozen=True)
class ControlTable:
    """The controls of every series, arrays with one row per series (ascending ``data_id``).

    ``m0`` and ``mfd`` are the values used (measured, or theoretical after a fallback) with their
    standard errors ``m0_se`` and ``mfd_se`` (NaN: single replicate; 0: theoretical).
    ``m0_measured`` and ``mfd_measured`` are the replicate means (NaN: no control), ``n_undeut``
    and ``n_fd`` the replicates that were used. ``retention`` is ``(mfd - m0) / n_amides`` of the
    values used. ``fallback`` is True where either control is theoretical, and ``flags`` lists
    the reasons, separated by ``;`` (see the module docstring). ``fd_model`` is the fitted
    theoretical back-exchange with ``fd_fallback='kint'`` (``a``, ``t0`` in s, ``n_series``
    fitted, ``n_rejected``, ``sd_per_amide`` and the ``quench`` conditions; None otherwise or
    when too few controls could be fitted). ``limits`` is the run-relative retention limit
    when ``retention_min`` was unset (``retention_floor``, with the ``median`` and ``robust_sd``
    of the controls it came from, None with too few of them for a limit (the floor is then 0),
    and their number ``n_series``); None with a set ``retention_min`` and in bundles written
    before it (judged by half of ``d2o_fraction``). ``screened`` is the spectrum ids of the
    control spectra the spectrum quality screens left out (the run: the loose level's
    removals; the analysis: its own level's); None when nothing was screened (bundles written
    before the run screened its controls).
    """

    config: ControlConfig
    data_id: np.ndarray
    n_amides: np.ndarray
    n_undeut: np.ndarray
    n_fd: np.ndarray
    m0_measured: np.ndarray
    mfd_measured: np.ndarray
    m0: np.ndarray
    m0_se: np.ndarray
    mfd: np.ndarray
    mfd_se: np.ndarray
    retention: np.ndarray
    fallback: np.ndarray
    flags: np.ndarray
    fd_model: Mapping[str, Any] | None = None
    limits: Mapping[str, Any] | None = None
    screened: tuple[str, ...] | None = None

    def __len__(self) -> int:
        return len(self.data_id)

    @property
    def uncorrected(self) -> np.ndarray:
        """True for the series that are not corrected (fragment ions, flagged
        ``uncorrected``): their values pass through :func:`correct_means` unchanged."""
        return np.array(["uncorrected" in str(f).split(";") for f in self.flags], dtype=bool)

    @property
    def measured(self) -> np.ndarray:
        """True where both controls are measured: the series' own, or for the fully deuterated
        one the same peptide's at its other charge states (``fd_other_charge``; the charge
        state does not change the deuterium level, HISTORY 130). Such a series is not
        ``fallback`` with ``combine_charges`` and keeps ``fallback`` True without it."""
        other = np.array(["fd_other_charge" in str(f).split(";") for f in self.flags], dtype=bool)
        m0_ok = (np.asarray(self.n_undeut) > 0) & np.array(
            ["m0_high" not in str(f).split(";") for f in self.flags], dtype=bool)
        return (~np.asarray(self.fallback, dtype=bool) | (other & m0_ok)) & ~self.uncorrected

    @property
    def fd_measured(self) -> np.ndarray:
        """True where the fully deuterated control is measured: the series' own, accepted, or
        the same peptide's at its other charge states (``fd_other_charge``); False for a
        stand-in (the run's median retention, the kint prediction or ``d2o_fraction``) and for
        fragment ions (not corrected)."""
        def own_or_other(flags: str) -> bool:
            tokens = set(str(flags).split(";"))
            return "fd_other_charge" in tokens or not tokens & set(FD_REJECTED)
        return (np.array([own_or_other(f) for f in self.flags], dtype=bool)
                & (np.asarray(self.n_amides) > 0) & ~self.uncorrected)

    def rows(self, data_id) -> np.ndarray:
        """Row of each series id in ``data_id`` (KeyError for a series without a row)."""
        index = {int(d): i for i, d in enumerate(self.data_id)}
        return np.array([index[int(d)] for d in np.asarray(data_id)], dtype=np.int64)


def _mean_se(values: np.ndarray) -> tuple[float, float, int]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return float(values.mean()), se, n


def series_rates(
    peptide,
    modification,
    data_id,
    conditions: ExchangeConditions,
    *,
    nterm_subtract: int = 1,
    rules=None,
    registry=None,
) -> dict[int, np.ndarray]:
    """The intrinsic rates (1/s) at ``conditions`` of the amides each series counts
    (``data_id`` -> array, in :func:`hxex.chem.amide_positions` order), from per-spectrum
    ``peptide``, ``modification`` and ``data_id`` arrays (a picked set's). Only parent-type
    ions (``M``) without extra ``Hex`` sites get rates; a counted N-terminal amine (with
    ``nterm_subtract`` 0) has an infinite rate (it keeps nothing)."""
    from . import chem
    from .kint import intrinsic_rates

    rules = chem.DEFAULT_AMIDE_RULES if rules is None else rules
    registry = chem.DEFAULT_MODIFICATIONS if registry is None else registry
    out: dict[int, np.ndarray] = {}
    cache: dict[tuple[str, str], np.ndarray | None] = {}
    for pep, mod, d in zip(np.asarray(peptide), np.asarray(modification), np.asarray(data_id),
                           strict=True):
        d = int(d)
        if d in out:
            continue
        key = (str(pep), str(mod))
        if key not in cache:
            spec = chem.parse_modification(key[1])
            if spec.ion_type != "M" or spec.hex:
                cache[key] = None
            else:
                pos = np.array(chem.amide_positions(key[0], "M", None,
                                                    nterm_subtract=nterm_subtract, rules=rules,
                                                    registry=registry), dtype=np.int64)
                k = intrinsic_rates(key[0], conditions, registry=registry).rate[pos - 1]
                cache[key] = np.where(np.isfinite(k), k, np.inf)
        if cache[key] is not None:
            out[d] = cache[key]
    return out


def series_rt(data_id, rt) -> dict[int, float]:
    """``data_id`` -> the median retention time (minutes) of the series' spectra, for the
    series that have one (per-spectrum ``data_id`` and ``rt``, a picked set's)."""
    data_id = np.asarray(data_id)
    rt = np.asarray(rt, dtype=float)
    out = {}
    for d in np.unique(data_id[np.isfinite(rt)]):
        out[int(d)] = float(np.median(rt[(data_id == d) & np.isfinite(rt)]))
    return out


def _kint_predict(a: float, t0: float, rates, rt=None, b: float = 0.0) -> np.ndarray:
    """``a sum exp(-k t)`` of each series, ``t = t0 + b * RT`` (RT in minutes, as seconds)."""
    rt = np.zeros(len(rates)) if rt is None else np.asarray(rt, dtype=float)
    return np.array([a * float(np.sum(np.exp(-k * (t0 + b * 60.0 * r))))
                     for k, r in zip(rates, rt, strict=True)])


def _fit_kint(measured: np.ndarray, rates: list, n: np.ndarray,
              rt: np.ndarray | None = None) -> tuple[float, float, float, float]:
    """``a``, ``t0`` and ``b`` of ``measured ~ a * sum exp(-k (t0 + b RT))`` (robust; ``b`` 0
    without retention times), and the scatter per amide of the residuals."""
    from scipy.optimize import least_squares

    with_rt = rt is not None

    def resid(p):
        return _kint_predict(p[0], float(np.exp(p[1])), rates, rt,
                             p[2] if with_rt else 0.0) - measured

    best = None
    for t_start in (60.0, 600.0, 3600.0, 36000.0):
        start = [0.9, np.log(t_start)] + ([0.0] if with_rt else [])
        lower = [0.0, np.log(1e-3)] + ([0.0] if with_rt else [])
        upper = [1.5, np.log(1e7)] + ([1e3] if with_rt else [])
        r = least_squares(resid, start, loss="soft_l1", f_scale=0.5, bounds=(lower, upper))
        if best is None or r.cost < best.cost:
            best = r
    a, t0 = float(best.x[0]), float(np.exp(best.x[1]))
    b = float(best.x[2]) if with_rt else 0.0
    free = 3 if with_rt else 2
    sd = (float(np.std((_kint_predict(a, t0, rates, rt, b) - measured) / n, ddof=free))
          if len(n) > free else float("nan"))
    return a, t0, b, sd


def fit_controls(
    fit: FitResult,
    n_amides,
    config: ControlConfig,
    usable: np.ndarray | None = None,
    rates: Mapping[int, np.ndarray] | None = None,
    rt: Mapping[int, float] | None = None,
    groups: Mapping[int, Any] | None = None,
    charges: Mapping[int, tuple] | None = None,
    uncorrected: Collection[int] = (),
    screened: np.ndarray | None = None,
) -> ControlTable:
    """The control table of a fit that includes K=1 (``fit.fits[1]``).

    The controls are the measurements with a role (``fit.role``: ``'undeut'``, ``'fulldeut'``).
    ``n_amides`` is the number of exchangeable amides of each measurement's peptide, ``(M,)``
    (constant within a series). ``usable`` (``(M,)`` bool) marks the spectra that may serve as
    controls; by default every spectrum that the noise estimate did not flag as an outlier.
    ``rates`` (``data_id`` -> the intrinsic rates at the quench conditions of the amides the
    series counts, :func:`series_rates`) is needed with ``fd_fallback='kint'``; ``rt``
    (``data_id`` -> retention time in minutes, :func:`series_rt`) adds the retention time to
    that model when at least ``kint_min_series`` of its controls have one. ``groups``
    (``data_id`` -> the molecule a series measures, :func:`series_groups`) compares each fully
    deuterated control with the same molecule's in the other samples (``fd_disagrees``).
    ``charges`` (``data_id`` -> (the peptide in its sample, charge), :func:`charge_groups`)
    lets a series without a good fully deuterated control take its other charge states'
    (``fd_other_charge``). The series of ``uncorrected`` (``data_id``; fragment ions,
    :func:`fragment_ids`) are not corrected and take no part in the others' checks.
    ``screened`` (``(M,)`` bool) marks the control spectra the spectrum quality screens remove
    (:mod:`hxex.analysis.screening`): left out like unusable ones, and recorded by id
    (``ControlTable.screened``).
    """
    uncorrected = {int(d) for d in uncorrected}
    if 1 not in fit.fits:
        raise ValueError("the controls come from the K=1 fits: fit with k_min=1")
    if config.fd_mode == "kint" and rates is None:
        raise ValueError("fd_fallback 'kint' needs the amides' intrinsic rates (series_rates)")
    if rates is not None and uncorrected:  # not in the back-exchange model of the others
        rates = {d: r for d, r in rates.items() if int(d) not in uncorrected}
    m = len(fit)
    n_amides = np.asarray(n_amides, dtype=float)
    if n_amides.shape != (m,):
        raise ValueError("n_amides must have one value per measurement")
    if usable is None:
        usable = (
            ~np.asarray(fit.noise_estimate.outlier, dtype=bool)
            if fit.noise_estimate is not None else np.ones(m, dtype=bool)
        )
    usable = np.asarray(usable, dtype=bool)
    if usable.shape != (m,):
        raise ValueError("usable must have one value per measurement")
    screened_ids = None
    if screened is not None:
        screened = np.asarray(screened, dtype=bool) & (np.asarray(fit.role, dtype=str) != "")
        if screened.shape != (m,):
            raise ValueError("screened must have one value per measurement")
        screened_ids = tuple(str(x) for x in np.asarray(fit.spectrum_id)[screened])
        usable = usable & ~screened
    means = fit.fits[1].means[:, 0]
    role = np.asarray(fit.role, dtype=str)
    series = np.unique(fit.data_id)
    serves = {UNDEUT: role == UNDEUT, FULLDEUT: role == FULLDEUT}
    for label, at, marked in ((UNDEUT, config.undeut_at, config.undeut_marked),
                              (FULLDEUT, config.fulldeut_at, config.fulldeut_marked)):
        if not marked:  # only the spectra at the given time serve (apply_control_roles)
            serves[label] &= time_point(fit.time) == time_point([at])[0]

    # every spectrum serving as a control needs its K = 1 fit: the run makes one for the
    # spectra with a role, but not for a time point promoted only afterwards (an
    # [analysis.controls] undeut_at or fulldeut_at, after a run with k_min above 1); its
    # missing mean would otherwise pass the limits and make the series' means NaN
    unfitted = (serves[UNDEUT] | serves[FULLDEUT]) & usable & ~np.isfinite(means)
    if unfitted.any() and not fit.fitted(1)[unfitted].all():
        raise ValueError(
            f"{int((unfitted & ~fit.fitted(1)).sum())} spectra serving as controls have no K=1 "
            "fit: the run fitted K from k_min above 1, and they were not controls then (a time "
            "point promoted by undeut_at or fulldeut_at after the run). Promote them in "
            "[run.controls] and run again, or run with k_min = 1")

    # a fit that failed in every restart keeps placeholder parameters (a mean of 0.3 n): never a
    # control or a point of the time course
    failed = fit.fitted(1) & ~np.isfinite(np.asarray(fit.fits[1].nll, dtype=float)) & usable
    if failed.any():
        log.warning("%d spectra whose K=1 fit failed are not used by the controls (%d of them "
                    "controls)", int(failed.sum()),
                    int((failed & (serves[UNDEUT] | serves[FULLDEUT])).sum()))
        usable = usable & ~failed
    left_out = _control_replicates_left_out(fit, means, n_amides, usable, serves, uncorrected,
                                            config)

    def pick(rows: np.ndarray, label: str) -> np.ndarray:
        return means[rows[usable[rows] & ~left_out[rows] & serves[label][rows]]]

    # the time course of each series (its spectra without a role, K=1 means, the median of
    # each time point's replicates, so that one bad replicate does not decide): the fully
    # deuterated control must not be below it, and an undeuterated control above it is noted
    # (time_course_tolerance)
    plain = (role == "") & ~serves[UNDEUT] & ~serves[FULLDEUT] & usable & np.isfinite(means)
    points = time_point(fit.time)
    multi = np.asarray(fit.selected_K) > 1

    def course(rows: np.ndarray) -> tuple[float, float]:
        rows = rows[plain[rows]]
        if not len(rows):
            return float("nan"), float("nan")
        level = [np.median(means[rows[points[rows] == p]]) for p in np.unique(points[rows])]
        return float(min(level)), float(max(level))

    cols: dict[str, list] = {name: [] for name in (
        "n_amides", "n_undeut", "n_fd", "m0_measured", "mfd_measured", "m0", "m0_se", "mfd",
        "mfd_se", "retention", "fallback", "flags",
        "course_high", "course_tol")}  # the last two for the stand-in checks only
    candidates: list[int] = []  # rows whose measured mFD passed the checks so far
    pending: list[int] = []  # rows whose mFD falls back, filled in after the loop
    for s in series:
        rows = np.flatnonzero(fit.data_id == s)
        n_am = np.unique(n_amides[rows])
        if len(n_am) != 1:
            raise ValueError(f"series {s} has more than one amide count: {n_am}")
        n_am = float(n_am[0])
        low, high = course(rows)
        tol = config.time_course_tolerance * n_am
        m0_meas, m0_se, n_u = _mean_se(pick(rows, UNDEUT))
        mfd_meas, mfd_se, n_f = _mean_se(pick(rows, FULLDEUT))
        if int(s) in uncorrected:  # values as measured: m0 0, mFD n_amides
            values = {
                "n_amides": n_am, "n_undeut": n_u, "n_fd": n_f, "m0_measured": m0_meas,
                "mfd_measured": mfd_meas, "m0": 0.0, "m0_se": 0.0, "mfd": n_am, "mfd_se": 0.0,
                "retention": float("nan"), "fallback": False, "flags": "uncorrected",
                "course_high": high, "course_tol": tol,
            }
            for name, value in values.items():
                cols[name].append(value)
            continue
        flags: list[str] = []
        for label, name in ((UNDEUT, "m0_replicate_left_out"), (FULLDEUT, "fd_replicate_left_out")):
            if (left_out[rows] & serves[label][rows]).any():
                flags.append(name)
        if n_u == 0:
            flags.append("m0_missing")
        elif m0_meas > config.m0_max:
            flags.append("m0_high")
        m0_ok = n_u > 0 and "m0_high" not in flags
        m0, m0_err = (m0_meas, m0_se) if m0_ok else (0.0, 0.0)
        if n_u > 0 and m0_ok and n_u == 1:
            flags.append("m0_unreplicated")
        mfd_ok = False
        if n_f == 0:
            flags.append("fd_missing")
        elif n_am > 0:
            ret = (mfd_meas - m0) / n_am
            # a fully deuterated level at or below the undeuterated one cannot scale anything,
            # whatever the limits
            if ret <= 0 or ret < config.min_retention:
                flags.append("retention_low")
            elif ret > config.retention_max:
                flags.append("retention_high")
            elif np.isfinite(high) and mfd_meas < high - tol:
                flags.append("fd_below_data")
            else:
                mfd_ok = True
        else:
            flags.append("no_amides")
        if mfd_ok:
            mfd, mfd_err = mfd_meas, mfd_se
            candidates.append(len(cols["mfd"]))
        else:
            mfd, mfd_err = m0 + config.d2o_fraction * n_am, 0.0
            pending.append(len(cols["mfd"]))
        if mfd_ok and n_f == 1:
            flags.append("fd_unreplicated")
        # a time point below the undeuterated control: the early, fast points are the hard
        # ones to measure, so the time point is suspect, not the control (a note only)
        if n_u > 0 and np.isfinite(low) and m0_meas > low + tol:
            flags.append("data_below_m0")
        for label, name in ((UNDEUT, "m0_multimodal"), (FULLDEUT, "fd_multimodal")):
            used = rows[usable[rows] & serves[label][rows]]
            if len(used) and 2 * multi[used].sum() > len(used):  # most replicates
                flags.append(name)
        fallback = not (m0_ok and mfd_ok)
        values = {
            "n_amides": n_am, "n_undeut": n_u, "n_fd": n_f, "m0_measured": m0_meas,
            "mfd_measured": mfd_meas, "m0": m0, "m0_se": m0_err, "mfd": mfd, "mfd_se": mfd_err,
            "retention": (mfd - m0) / n_am if n_am > 0 else float("nan"),
            "fallback": fallback, "flags": ";".join(flags), "course_high": high,
            "course_tol": tol,
        }
        for name, value in values.items():
            cols[name].append(value)

    # without a fixed floor the controls are judged against the run: those the prediction
    # judges (fd_fallback 'kint', series with rates) are left to it
    limits = None
    if config.retention_min is None:
        limits = _run_limits(cols, candidates, config)
        judged = (candidates if config.fd_mode != "kint" else
                  [i for i in candidates if rates is None or int(series[i]) not in rates])
        pending += _reject_outliers(cols, judged, limits, config)
    fd_model = None
    if config.fd_mode == "kint":
        fd_model = _kint_fallback(cols, series, sorted(pending), rates, config, rt)
        if fd_model is None and config.retention_min is None:
            # nothing judged the rated series' controls: the run's limit judges them as well
            pending_set = set(pending)
            rated = [i for i in candidates if i not in pending_set
                     and int(series[i]) in rates]
            pending += _reject_outliers(cols, rated, limits, config)
    # what the prediction did not fill (the median mode, series without rates, or no model)
    # takes the median retention of the run's accepted controls; fd_fallback 'd2o' keeps
    # d2o_fraction
    pending = sorted(i for i in set(pending) if config.fd_mode != "d2o"
                     and "fd_kint" not in cols["flags"][i].split(";"))
    accepted = [cols["retention"][i] for i in candidates
                if not set(cols["flags"][i].split(";")) & set(FD_REJECTED)]
    if pending and not accepted:
        log.warning("no fully deuterated control was accepted, so the %d series without one "
                    "take d2o_fraction (%g) as their retention: no back-exchange correction",
                    len(pending), config.d2o_fraction)
    if pending and accepted:
        r = float(np.median(accepted))
        for i in pending:
            n_am, m0 = cols["n_amides"][i], cols["m0"][i]
            cols["mfd"][i] = m0 + r * n_am
            cols["retention"][i] = r if n_am > 0 else float("nan")
            cols["flags"][i] = ";".join(f for f in (cols["flags"][i], "fd_median") if f)
    median = float(np.median(accepted)) if accepted else None
    if config.combine_charges and charges:
        _combine_charges(cols, series, charges, config, median)
    elif config.fd_other_charge and charges:
        _other_charge_fallback(cols, series, charges, config, median)
    _flag_disagreement(cols, series, groups, config, charges)
    return ControlTable(
        config=config, data_id=series.astype(np.int64),
        n_amides=np.array(cols["n_amides"], dtype=np.float64),
        n_undeut=np.array(cols["n_undeut"], dtype=np.int64),
        n_fd=np.array(cols["n_fd"], dtype=np.int64),
        m0_measured=np.array(cols["m0_measured"], dtype=np.float64),
        mfd_measured=np.array(cols["mfd_measured"], dtype=np.float64),
        m0=np.array(cols["m0"], dtype=np.float64),
        m0_se=np.array(cols["m0_se"], dtype=np.float64),
        mfd=np.array(cols["mfd"], dtype=np.float64),
        mfd_se=np.array(cols["mfd_se"], dtype=np.float64),
        retention=np.array(cols["retention"], dtype=np.float64),
        fallback=np.array(cols["fallback"], dtype=bool),
        flags=np.array(cols["flags"], dtype=str),
        fd_model=fd_model, limits=limits, screened=screened_ids,
    )


def _control_replicates_left_out(fit, means: np.ndarray, n_amides: np.ndarray,
                                 usable: np.ndarray, serves: Mapping, uncorrected,
                                 config: ControlConfig) -> np.ndarray:
    """The control spectra left out of their control's mean (``replicate_outlier``): a K=1 mean
    above the amide count, or a replicate far from the median of its series' replicates of
    that control (three or more), measured in typical replicate standard deviations of the
    run's time points (deuterons per amide, uncorrected)."""
    role_rows = (serves[UNDEUT] | serves[FULLDEUT]) & usable & np.isfinite(means)
    skip = np.isin(np.asarray(fit.data_id), list(uncorrected))
    out = role_rows & ~skip & (means > n_amides + 1e-9)
    if config.replicate_outlier is None:
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        per_amide = np.where(n_amides > 0, means / n_amides, np.nan)
    plain = (np.asarray(fit.role, dtype=str) == "") & usable & np.isfinite(per_amide)
    points = time_point(fit.time)
    groups: dict[tuple, list[int]] = {}
    for i in np.flatnonzero(plain):
        groups.setdefault((int(fit.data_id[i]), points[i]), []).append(i)
    scaled = [float(np.var(per_amide[g], ddof=1)) * (len(g) - 1) / stats.chi2.median(len(g) - 1)
              for g in groups.values() if len(g) >= 2]
    if not scaled:
        return out
    limit = config.replicate_outlier * float(np.sqrt(np.median(scaled)))
    ids = np.asarray(fit.data_id)
    for label in (UNDEUT, FULLDEUT):
        rows_of: dict[int, list[int]] = {}
        for i in np.flatnonzero(role_rows & ~skip & ~out & serves[label]):
            rows_of.setdefault(int(ids[i]), []).append(i)
        for g in rows_of.values():
            if len(g) < 3:
                continue
            centre = float(np.median(per_amide[g]))
            for i in g:
                if abs(per_amide[i] - centre) > limit:
                    out[i] = True
    return out


OUTLIER_MIN_SERIES = 5  # the fewest controls the run-relative limit is computed from


def _run_limits(cols: dict, candidates: list[int], config: ControlConfig) -> dict:
    """The run-relative retention limit (``outlier_sd`` robust standard deviations below the
    median of the controls that passed the other checks); 0, with ``median`` and
    ``robust_sd`` None, with fewer than :data:`OUTLIER_MIN_SERIES` of them."""
    ret = np.array([cols["retention"][i] for i in candidates], dtype=float)
    if len(ret) < OUTLIER_MIN_SERIES:
        if len(ret):
            log.info("the run-relative retention limit needs %d controls, %d passed the other "
                     "checks: no lower limit", OUTLIER_MIN_SERIES, len(ret))
        return {"retention_floor": 0.0, "median": None, "robust_sd": None,
                "n_series": len(ret)}
    median = float(np.median(ret))
    sd = float(1.4826 * np.median(np.abs(ret - median)))
    return {"retention_floor": max(median - config.outlier_sd * sd, 0.0), "median": median,
            "robust_sd": sd, "n_series": len(ret)}


def _reject_outliers(cols: dict, rows: list[int], limits: dict | None,
                     config: ControlConfig) -> list[int]:
    """Reject the fully deuterated controls of ``rows`` below ``limits`` (``retention_outlier``,
    retention ``d2o_fraction`` until a fallback fills it in) and return them."""
    if limits is None or limits["median"] is None:
        return []
    floor = limits["retention_floor"]
    low = [i for i in rows if cols["retention"][i] < floor]
    for i in low:
        n_am, m0 = cols["n_amides"][i], cols["m0"][i]
        cols["mfd"][i], cols["mfd_se"][i] = m0 + config.d2o_fraction * n_am, 0.0
        cols["retention"][i] = config.d2o_fraction
        cols["fallback"][i] = True
        cols["flags"][i] = ";".join([*(f for f in cols["flags"][i].split(";")
                                       if f and f != "fd_unreplicated"), "retention_outlier"])
    if low:
        log.warning("%d fully deuterated control(s) below the run's retention limit %.2f "
                    "(median %.2f - %g x %.3f) are rejected", len(low), floor,
                    limits["median"], config.outlier_sd, limits["robust_sd"])
    return low


def _flag_disagreement(cols: dict, series: np.ndarray, groups: Mapping | None,
                       config: ControlConfig, charges: Mapping | None = None) -> None:
    """Flag ``fd_disagrees`` on the measured fully deuterated controls whose retention differs
    by more than ``sample_tolerance`` from the median of their group (the same peptide in the
    other samples, :func:`series_groups`; with two controls, from the other one: both are
    flagged). A control flagged ``fd_multimodal`` whose retention agrees gets
    ``fd_retention_agrees``: its envelope is wide or split, but its retention, which the
    correction uses, matches the other samples' (owner, 2026-10-22). One that no other sample
    can check is compared with the same peptide's measured controls at its other charge
    states in the same sample (``charges``): ``fd_charges_agree`` when its retention is within
    ``sample_tolerance`` of their median (HISTORY 145)."""
    if groups:
        _flag_sample_disagreement(cols, series, groups, config)
    if not charges:
        return
    members: dict[tuple, list[int]] = {}
    for i, s in enumerate(series):
        if int(s) in charges and cols["n_fd"][i] > 0 and cols["n_amides"][i] > 0:
            members.setdefault(charges[int(s)][0], []).append(i)
    for rows in members.values():
        measured = {i: (cols["mfd_measured"][i] - cols["m0"][i]) / cols["n_amides"][i]
                    for i in rows}
        for i in rows:
            tokens = cols["flags"][i].split(";")
            if ("fd_multimodal" not in tokens or "fd_retention_agrees" in tokens
                    or "fd_disagrees" in tokens or not np.isfinite(measured[i])):
                continue  # not multimodal, or already checked against the other samples
            others = [r for j, r in measured.items() if j != i and np.isfinite(r)]
            if others and abs(measured[i] - float(np.median(others))) <= config.sample_tolerance:
                cols["flags"][i] += ";fd_charges_agree"


def _flag_sample_disagreement(cols: dict, series: np.ndarray, groups: Mapping,
                              config: ControlConfig) -> None:
    """The other-samples part of :func:`_flag_disagreement`."""
    members: dict[Any, list[int]] = {}
    for i, s in enumerate(series):
        if (int(s) in groups and cols["n_fd"][i] > 0 and cols["n_amides"][i] > 0
                and "uncorrected" not in cols["flags"][i].split(";")):
            members.setdefault(groups[int(s)], []).append(i)
    n = 0
    for rows in members.values():
        measured = {i: (cols["mfd_measured"][i] - cols["m0"][i]) / cols["n_amides"][i]
                    for i in rows}
        finite = [r for r in measured.values() if np.isfinite(r)]
        for i in rows:
            if len(finite) < 2 or not np.isfinite(measured[i]):
                continue
            # the group's median (robust from three on), or the other control of a pair
            ref = (float(np.median(finite)) if len(finite) > 2
                   else next(r for j, r in measured.items() if j != i))
            if abs(measured[i] - ref) > config.sample_tolerance:
                cols["flags"][i] = ";".join(f for f in (cols["flags"][i], "fd_disagrees") if f)
                n += 1
            elif "fd_multimodal" in cols["flags"][i].split(";"):
                # a wide or split envelope whose retention agrees with the same peptide's in the
                # other samples: the correction, which uses the retention, stays sound
                cols["flags"][i] += ";fd_retention_agrees"
    if n:
        log.warning("%d fully deuterated control(s) differ from the same peptide's in the other "
                    "samples by more than %g in retention (flagged fd_disagrees)", n,
                    config.sample_tolerance)


def series_groups(peptide, modification, charge, data_id) -> dict[int, tuple]:
    """``data_id`` -> the molecule a series measures (peptide, modification and charge), from
    per-spectrum arrays (a picked set's): series with the same key in different samples or
    conditions should have the same retention."""
    out: dict[int, tuple] = {}
    for p, mod, z, d in zip(np.asarray(peptide), np.asarray(modification), np.asarray(charge),
                            np.asarray(data_id), strict=True):
        out.setdefault(int(d), (str(p), str(mod), int(z)))
    return out


def fragment_ids(modification, data_id, fragments=()) -> set[int]:
    """The series (``data_id``) of fragment ions, from per-spectrum arrays: those of
    ``fragments`` (ids of fragment scan targets, :func:`hxex.io.mzml.fragment_series`) and every
    series whose ion type is not ``M``."""
    from .chem import parse_modification

    out = {int(d) for d in fragments}
    for mod, d in zip(np.asarray(modification), np.asarray(data_id), strict=True):
        if int(d) not in out and parse_modification(str(mod)).ion_type != "M":
            out.add(int(d))
    return out


def charge_groups(sample, peptide, modification, charge, data_id,
                  fragments=()) -> dict[int, tuple[tuple, int]]:
    """``data_id`` -> ((sample, peptide, modification), charge), from per-spectrum arrays: the
    series with the same first part are one peptide in one sample at different charges, whose
    fully deuterated controls should agree (``fd_other_charge``). ``undeut_mz``, which differs
    by charge, is left out of the modification. Fragment ions are left out: the series of
    ``fragments`` (ids), and every series whose ion type is not ``M``."""
    skip = fragment_ids(modification, data_id, fragments)
    out: dict[int, tuple[tuple, int]] = {}
    for s, p, mod, z, d in zip(np.asarray(sample), np.asarray(peptide),
                               np.asarray(modification), np.asarray(charge),
                               np.asarray(data_id), strict=True):
        d = int(d)
        if d in skip or d in out:
            continue
        kept = " ".join(x for x in str(mod).split() if not x.startswith("undeut_mz:"))
        out[d] = ((str(s), str(p), kept), int(z))
    return out


# the flags of a fully deuterated control that is not used (the series falls back)
FD_REJECTED = ("fd_missing", "retention_low", "retention_high", "fd_below_data",
               "retention_outlier", "fd_off_model")


def _fd_used(cols: dict, i: int) -> bool:
    """Whether row ``i``'s own fully deuterated control is measured and accepted."""
    return (cols["n_fd"][i] > 0 and cols["n_amides"][i] > 0
            and not set(cols["flags"][i].split(";")) & set(FD_REJECTED))


def _below_its_data(cols: dict, i: int, r: float) -> bool:
    """Whether the retention ``r`` taken from other charge states would put row ``i``'s fully
    deuterated level below its own time course (as ``fd_below_data`` judges a measured one)."""
    high, tol = cols["course_high"][i], cols["course_tol"][i]
    return bool(np.isfinite(high) and cols["m0"][i] + r * cols["n_amides"][i] < high - tol)


def _stand_in_rejected(cols: dict, i: int, median: float | None,
                       config: ControlConfig) -> None:
    """Row ``i``, whose stand-in from other charge states fails its own time course: treated as
    ``fd_below_data``, with the run's median retention (``d2o_fraction`` without accepted
    controls)."""
    r, extra = (median, ["fd_median"]) if median is not None else (config.d2o_fraction, [])
    n_am = cols["n_amides"][i]
    cols["mfd"][i], cols["mfd_se"][i] = cols["m0"][i] + r * n_am, 0.0
    cols["retention"][i] = r
    cols["fallback"][i] = True
    kept = [f for f in cols["flags"][i].split(";") if f and f not in (
        "fd_median", "fd_kint", "fd_below_data", "fd_unreplicated")]
    cols["flags"][i] = ";".join([*kept, "fd_below_data", *extra])


def _combine_charges(cols: dict, series: np.ndarray, charges: Mapping,
                     config: ControlConfig, median: float | None = None) -> None:
    """``combine_charges``: give every series of a peptide (one sample) with an accepted
    measured fully deuterated control at any charge state the plain mean retention of those
    controls (in ``cols``), replacing a fallback already filled in. Where the mean would put
    a series below its own time course, a series with an accepted control of its own keeps
    it; one without is ``fd_below_data`` and takes the run's ``median`` retention instead."""
    members: dict[tuple, list[int]] = {}
    for i, s in enumerate(series):
        if int(s) in charges:
            members.setdefault(charges[int(s)][0], []).append(i)
    n_other = n_combined = n_disagree = n_below = n_own = 0
    for rows in members.values():
        donors = [j for j in rows if _fd_used(cols, j)]
        if not donors:
            continue
        rets = [cols["retention"][j] for j in donors]
        r = float(np.mean(rets))
        se = np.array([cols["mfd_se"][j] / cols["n_amides"][j] for j in donors])
        r_se = (float(np.sqrt(np.sum(np.nan_to_num(se) ** 2))) / len(donors)
                if np.isfinite(se).any() else float("nan"))
        disagree = len(donors) >= 2 and max(rets) - min(rets) > config.sample_tolerance
        n_disagree += disagree
        for i in rows:
            n_am = cols["n_amides"][i]
            if n_am <= 0:
                continue
            own = i in donors
            if own and len(donors) == 1:
                continue  # its own control alone: nothing changes
            if _below_its_data(cols, i, r):
                if own:  # its own control passed its checks: kept (owner, 2026-10-29)
                    n_own += 1
                    continue
                _stand_in_rejected(cols, i, median, config)
                n_below += 1
                continue
            cols["mfd"][i] = cols["m0"][i] + r * n_am
            cols["mfd_se"][i] = n_am * r_se
            cols["retention"][i] = r
            kept = [f for f in cols["flags"][i].split(";")
                    if f and f not in ("fd_median", "fd_kint")]
            if own:
                kept.append("fd_charges_combined")
                n_combined += 1
            else:
                kept.append("fd_other_charge")
                # measured at the peptide's other charges: only the undeuterated control can
                # still make the series a fallback
                cols["fallback"][i] = bool({"m0_missing", "m0_high"} & set(kept))
                n_other += 1
            if disagree:
                kept.append("fd_charges_disagree")
            cols["flags"][i] = ";".join(kept)
    if n_other or n_combined:
        log.info("fully deuterated controls combined over charge states (combine_charges): %d "
                 "series take the mean of two or more, %d without a good control of their own "
                 "take the peptide's at its other charges", n_combined, n_other)
    if n_below:
        log.info("%d series without a good control of their own would be below their own "
                 "time course with the combined control: fd_below_data, the run's median "
                 "retention instead", n_below)
    if n_own:
        log.info("%d series keep their own fully deuterated control: the combined one would "
                 "put it below their own time course", n_own)
    if n_disagree:
        log.warning("%d peptides have fully deuterated controls at different charge states "
                    "whose retentions differ by more than %g; combined anyway "
                    "(fd_charges_disagree)", n_disagree, config.sample_tolerance)


def _other_charge_fallback(cols: dict, series: np.ndarray, charges: Mapping,
                           config: ControlConfig, median: float | None = None) -> None:
    """Give every series without a good fully deuterated control the mean retention of the
    same peptide's accepted measured controls at its other charges, in place of the fallback
    already filled in (``fd_median``, ``fd_kint``, or ``d2o_fraction``). A series that it
    would put below its own time course is ``fd_below_data`` and takes the run's ``median``
    retention instead."""
    def used(i: int) -> bool:
        return _fd_used(cols, i)

    members: dict[tuple, list[int]] = {}
    for i, s in enumerate(series):
        if int(s) in charges:
            members.setdefault(charges[int(s)][0], []).append(i)
    n = 0
    for rows in members.values():
        for i in rows:
            if used(i) or cols["n_amides"][i] <= 0:
                continue
            z = charges[int(series[i])][1]
            donors = [j for j in rows if used(j) and charges[int(series[j])][1] != z]
            if not donors:
                continue
            r = float(np.mean([cols["retention"][j] for j in donors]))
            if _below_its_data(cols, i, r):
                _stand_in_rejected(cols, i, median, config)
                continue
            se = np.array([cols["mfd_se"][j] / cols["n_amides"][j] for j in donors])
            n_am, m0 = cols["n_amides"][i], cols["m0"][i]
            cols["mfd"][i] = m0 + r * n_am
            cols["mfd_se"][i] = (n_am * float(np.sqrt(np.sum(np.nan_to_num(se) ** 2)))
                                 / len(donors) if np.isfinite(se).any() else float("nan"))
            cols["retention"][i] = r
            cols["flags"][i] = ";".join([*(f for f in cols["flags"][i].split(";")
                                           if f and f not in ("fd_median", "fd_kint")),
                                         "fd_other_charge"])
            n += 1
    if n:
        log.info("%d series without a good fully deuterated control take the same peptide's "
                 "at its other charge states (fd_other_charge)", n)


def _kint_fallback(cols: dict, series: np.ndarray, pending: list[int], rates: Mapping,
                   config: ControlConfig, rt: Mapping[int, float] | None = None) -> dict | None:
    """Fit theoretical back-exchange to the accepted controls with rates, reject the controls
    far from it, refit, and fill in every series that falls back and has rates (in ``cols``)."""
    def flag(i: int, name: str) -> None:
        cols["flags"][i] = ";".join(f for f in (cols["flags"][i], name) if f)

    pending_set = set(pending)
    fitted = [i for i, s in enumerate(series)
              if i not in pending_set and int(s) in rates and cols["n_amides"][i] > 0]
    if len(fitted) < config.kint_min_series:
        log.warning("theoretical back-exchange not fitted: %d usable fully deuterated controls "
                    "(kint_min_series %d); missing ones take the run's median retention",
                    len(fitted), config.kint_min_series)
        return None

    def measured(rows):
        return np.array([cols["mfd"][i] - cols["m0"][i] for i in rows])

    def amides(rows):
        return np.array([cols["n_amides"][i] for i in rows])

    rt = dict(rt or {})
    with_rt = [i for i in fitted if int(series[i]) in rt]
    use_rt = len(with_rt) >= config.kint_min_series
    median_rt = float(np.median([rt[int(series[i])] for i in with_rt])) if use_rt else 0.0
    if rt and not use_rt:
        log.info("theoretical back-exchange without retention times: %d of %d controls have "
                 "one (kint_min_series %d)", len(with_rt), len(fitted), config.kint_min_series)
    elif use_rt and len(with_rt) < len(fitted):
        log.info("theoretical back-exchange: %d of %d controls have no retention time and are "
                 "taken at the median, %.2f min", len(fitted) - len(with_rt), len(fitted),
                 median_rt)

    def times(rows):
        return (np.array([rt.get(int(series[i]), median_rt) for i in rows]) if use_rt
                else None)

    a, t0, b, _ = _fit_kint(measured(fitted), [rates[int(series[i])] for i in fitted],
                            amides(fitted), times(fitted))
    off = (np.abs(_kint_predict(a, t0, [rates[int(series[i])] for i in fitted], times(fitted), b)
                  - measured(fitted)) / amides(fitted) > config.kint_tolerance)
    rejected = [i for i, o in zip(fitted, off, strict=True) if o]
    kept = [i for i, o in zip(fitted, off, strict=True) if not o]
    if len(kept) < config.kint_min_series:
        log.warning("theoretical back-exchange not fitted: %d fully deuterated controls agree "
                    "with it (kint_min_series %d); missing ones take the run's median "
                    "retention", len(kept), config.kint_min_series)
        return None
    a, t0, b, sd = _fit_kint(measured(kept), [rates[int(series[i])] for i in kept],
                             amides(kept), times(kept))
    for i in rejected:
        cols["fallback"][i] = True
        cols["flags"][i] = ";".join(f for f in cols["flags"][i].split(";")
                                    if f and f != "fd_unreplicated")
        flag(i, "fd_off_model")
    for i in sorted(set(pending) | set(rejected)):
        k = rates.get(int(series[i]))
        if k is None or cols["n_amides"][i] <= 0:
            continue
        n_am, m0 = cols["n_amides"][i], cols["m0"][i]
        own = rt.get(int(series[i]), median_rt) if use_rt else 0.0
        cols["mfd"][i] = m0 + float(_kint_predict(a, t0, [k], [own], b)[0])
        cols["mfd_se"][i] = sd * n_am if np.isfinite(sd) else 0.0
        cols["retention"][i] = (cols["mfd"][i] - m0) / n_am
        flag(i, "fd_kint")
    log.info("theoretical back-exchange: a = %.3f, t0 = %.0f s%s from %d controls (%d "
             "rejected, scatter %.3f D per amide)", a, t0,
             f" + {b:.3g} x RT" if use_rt else "", len(kept), len(rejected), sd)
    return {"a": a, "t0": t0, "b_rt": b if use_rt else None, "median_rt": median_rt
            if use_rt else None, "n_series": len(kept), "n_rejected": len(rejected),
            "sd_per_amide": sd, "quench": config.quench.to_dict()}


@dataclass(frozen=True)
class CorrectedMeans:
    """Back-exchange-corrected means, with the shape of the input means ``(M, ...)``.

    ``se`` and ``ci`` (``(..., 2)``) are None when none were given. ``control_se`` is the part of
    the uncertainty that comes from the controls alone.
    """

    mean: np.ndarray
    control_se: np.ndarray
    se: np.ndarray | None
    ci: np.ndarray | None


def correct_means(
    controls: ControlTable,
    data_id,
    mean: np.ndarray,
    se: np.ndarray | None = None,
    ci: np.ndarray | None = None,
    ci_level: float = 0.95,
) -> CorrectedMeans:
    """Correct means for back-exchange, ``(mean - m0) / (mFD - m0) * n_amides`` (not below 0);
    the series that are not corrected (:attr:`ControlTable.uncorrected`) keep their values.

    ``mean`` is ``(M, ...)`` with the series of each row in ``data_id`` (``(M,)``); NaN stays NaN.
    A series whose ``mfd`` is not above its ``m0`` cannot be corrected: its values become NaN,
    with a warning (unless the series has no exchangeable site, where NaN is expected).
    A standard error ``se`` (same shape) gets the control uncertainty added in quadrature (delta
    method). An interval ``ci`` (``(..., 2)``) is transformed at its ends, which is exact because
    the correction increases with the mean, and each side is then widened in quadrature by the
    control uncertainty at ``ci_level``.
    """
    mean = np.asarray(mean, dtype=float)
    rows = controls.rows(data_id)
    if mean.shape[0] != len(rows):
        raise ValueError("data_id must have one value per row of mean")
    pad = (slice(None),) + (None,) * (mean.ndim - 1)
    m0, f, n = (getattr(controls, k)[rows][pad] for k in ("m0", "mfd", "n_amides"))
    raw = controls.uncorrected[rows]  # fragment ions: passed through unchanged
    bad = (controls.mfd[rows] <= controls.m0[rows]) & (controls.n_amides[rows] > 0) & ~raw
    if bad.any():
        ids = np.unique(np.asarray(data_id)[bad])
        log.warning("%d series (data_id %s) have a fully deuterated level at or below the "
                    "undeuterated one: their corrected means are NaN", len(ids),
                    ", ".join(str(int(d)) for d in ids[:5]) + (", ..." if len(ids) > 5 else ""))
    s0 = np.nan_to_num(controls.m0_se[rows])[pad]
    sf = np.nan_to_num(controls.mfd_se[rows])[pad]
    span = np.where(f - m0 > 0, f - m0, np.nan)

    def transform(x):
        return np.maximum((x - m0) / span * n, 0.0)

    corrected = transform(mean)
    d0 = n * (mean - f) / span**2
    df = -n * (mean - m0) / span**2
    inside = (mean - m0) / span * n > 0  # clipped values do not depend on the controls
    control_se = np.where(inside, np.hypot(d0 * s0, df * sf), 0.0)
    out_se = out_ci = None
    if se is not None:
        se = np.asarray(se, dtype=float)
        out_se = np.hypot(n / span * se, control_se)
    if ci is not None:
        ci = np.asarray(ci, dtype=float)
        z = norm.ppf(0.5 + ci_level / 2.0)
        half = z * control_se
        lo, hi = transform(ci[..., 0]), transform(ci[..., 1])
        out_ci = np.stack([
            np.maximum(corrected - np.hypot(corrected - lo, half), 0.0),
            corrected + np.hypot(hi - corrected, half),
        ], axis=-1)
    if raw.any():
        keep = np.broadcast_to(raw[pad], corrected.shape)
        corrected = np.where(keep, mean, corrected)
        control_se = np.where(keep, 0.0, control_se)
        if out_se is not None:
            out_se = np.where(keep, se, out_se)
        if out_ci is not None:
            out_ci = np.where(keep[..., None], ci, out_ci)
    return CorrectedMeans(mean=corrected, control_se=control_se, se=out_se, ci=out_ci)
