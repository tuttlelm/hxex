# SPDX-License-Identifier: GPL-3.0-or-later
"""Undeuterated and fully deuterated controls and the back-exchange correction.

A series (one peptide at one charge, ``data_id``) has its controls at the user-set times
``undeut_time`` and ``fd_time``; replicate controls are averaged. The control means ``m0`` and
``mFD`` are the means of the K=1 fits of the control spectra, in the model's units: deuterons
beyond natural abundance (the natural-abundance shape is convolved separately, so an undeuterated
spectrum has ``m0`` near 0, and the centre of the raw envelope must not be used instead).

Each control falls back to its theoretical value, and the series is flagged, on its own:

- ``m0`` falls back to 0 when it is missing (``m0_missing``) or above ``m0_max`` (``m0_high``);
- ``mFD`` falls back to ``m0 + r * n_amides`` when it is missing (``fd_missing``) or when the
  retention ``(mFD - m0) / n_amides`` is below ``retention_min`` (``retention_low``; by default
  half of ``d2o_fraction``, also with ``fd_fallback='kint'`` for the series the prediction
  cannot judge, those without rates: fragment ions and ``Hex`` series; a retention of 0 or
  less is always rejected, whatever the floor) or
  above ``retention_max`` (``retention_high``), with ``m0`` as it is used. The fallback
  retention ``r`` is ``d2o_fraction`` (``fd_fallback='d2o'``, the deuterium fraction of the
  labelling buffer, a user setting) or the median retention of the run's accepted controls
  (``fd_fallback='median'``; flagged ``fd_median``), which includes the back-exchange the
  buffer's fraction does not.

With ``fd_fallback='kint'`` (the default when the ``quench`` conditions are given, which it
needs) the fully deuterated value comes from theoretical back-exchange instead (HISTORY items 71
and 73): every amide of the peptide keeps ``a * exp(-k_i * t0)`` of a deuteron, ``k_i`` its
intrinsic rate at the quench conditions (:mod:`hxex.kint`), with the labelling fraction ``a``
and an effective time ``t0`` fitted to the run's measured fully deuterated controls (robustly,
then again without the rejected ones). A measured control is rejected when it misses that
prediction by more than ``kint_tolerance`` deuterons per amide (``fd_off_model``), in place of
the ``retention_min`` floor (which still applies when set); a missing or rejected control falls
back to ``m0 + prediction`` (flagged ``fd_kint``) with the fit's scatter per amide times
``n_amides`` as its standard error. Series without rates (fragment ions, extra ``Hex`` sites)
keep the ``d2o_fraction`` fallback; with fewer than ``kint_min_series`` usable (or agreeing)
controls nothing is fitted and every series keeps it, with its floor (logged). When every
control in the fit has a retention
time (``RT``, minutes; :func:`series_rt`) for at least ``kint_min_series`` of the controls, the
effective time is ``t0 + b RT`` (``b`` fitted, not below 0; HISTORY items 71 and 83), and a
series without one is taken at the median RT of those that have one.
The fitted model is :attr:`ControlTable.fd_model`.

A measured control with a single replicate has no uncertainty (flagged ``m0_unreplicated`` or
``fd_unreplicated``); a theoretical value is exact. The corrected mean of a population is
``(mean - m0) / (mFD - m0) * n_amides``, not below 0 (:func:`correct_means`); the uncertainty of
the controls is added to the uncertainty of the mean.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy.stats import norm

from .fit.result import FitResult
from .kint import ExchangeConditions

log = logging.getLogger(__name__)

FD_FALLBACKS = ("d2o", "median", "kint")


@dataclass(frozen=True)
class ControlConfig:
    """Control times (in the units of the measurement times) and the limits for a good control.

    A time of None means that the data has no control of that kind. The defaults of the limits are
    those of the notebook the method comes from; on the Bimodal data they hardly matter (see
    docs/HISTORY.md item 27). ``d2o_fraction`` is the deuterium fraction of the labelling buffer
    (often 0.8 to 0.9; the default is pyHXExpress's ``Dfrac``). ``retention_min`` None (the
    default) means half of it. ``fd_fallback`` is the retention used when a series' fully
    deuterated control is missing or rejected: ``'d2o'`` (``d2o_fraction``), ``'median'`` (the
    median of the run's accepted controls; ``d2o_fraction`` if there are none) or ``'kint'``
    (theoretical back-exchange fitted to the run's controls; see the module docstring), which
    needs ``quench``; None (the default) means ``'kint'`` when ``quench`` is given and ``'d2o'``
    otherwise (:attr:`fd_mode`; owner, HISTORY item 73). ``quench`` holds the conditions of the
    back-exchange (:class:`hxex.kint.ExchangeConditions`:
    pH_read and temperature of the quench and LC, ``d2o_fraction`` of their solvent, about 0).
    ``kint_tolerance`` (deuterons per amide) rejects a measured control that far from the
    prediction, and ``kint_min_series`` is the fewest usable controls to fit. With ``'kint'``
    and ``retention_min`` None there is no retention floor. See docs/HISTORY.md items 59, 61
    and 73.
    """

    undeut_time: float | None = None
    fd_time: float | None = None
    m0_max: float = 0.5
    retention_min: float | None = None
    retention_max: float = 1.1
    d2o_fraction: float = 0.9
    fd_fallback: str | None = None
    atol: float = 1e-6
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
        if not 0 <= self.min_retention < self.retention_max:
            raise ValueError("need 0 <= retention_min < retention_max")
        if self.fd_fallback is not None and self.fd_fallback not in FD_FALLBACKS:
            raise ValueError(f"fd_fallback must be one of {', '.join(FD_FALLBACKS)}")
        if self.atol < 0:
            raise ValueError("atol must be >= 0")
        if not 0 < self.d2o_fraction <= 1:
            raise ValueError("d2o_fraction must be in (0, 1]")

    @property
    def fd_mode(self) -> str:
        """The fallback in use: ``fd_fallback``, or when unset ``'kint'`` with quench conditions
        and ``'d2o'`` without."""
        if self.fd_fallback is not None:
            return self.fd_fallback
        return "kint" if self.quench is not None else "d2o"

    @property
    def min_retention(self) -> float:
        """The lower retention limit in use (``retention_min``; unset: half of ``d2o_fraction``,
        or none with ``fd_fallback='kint'``, which judges controls by the prediction; series
        without rates keep the ``d2o_fraction`` floor there, see :func:`fit_controls`)."""
        if self.retention_min is not None:
            return self.retention_min
        return 0.0 if self.fd_mode == "kint" else self.d2o_fraction / 2

    def to_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["quench"] = None if self.quench is None else self.quench.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ControlConfig:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown control setting(s): {', '.join(sorted(unknown))}")
        return cls(**data)


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
    when too few controls could be fitted).
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

    def __len__(self) -> int:
        return len(self.data_id)

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
) -> ControlTable:
    """The control table of a fit that includes K=1 (``fit.fits[1]``).

    ``n_amides`` is the number of exchangeable amides of each measurement's peptide, ``(M,)``
    (constant within a series). ``usable`` (``(M,)`` bool) marks the spectra that may serve as
    controls; by default every spectrum that the noise estimate did not flag as an outlier.
    ``rates`` (``data_id`` -> the intrinsic rates at the quench conditions of the amides the
    series counts, :func:`series_rates`) is needed with ``fd_fallback='kint'``; ``rt``
    (``data_id`` -> retention time in minutes, :func:`series_rt`) adds the retention time to
    that model when at least ``kint_min_series`` of its controls have one.
    """
    if 1 not in fit.fits:
        raise ValueError("the controls come from the K=1 fits: fit with k_min=1")
    if config.fd_mode == "kint" and rates is None:
        raise ValueError("fd_fallback 'kint' needs the amides' intrinsic rates (series_rates)")
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
    means = fit.fits[1].means[:, 0]
    time = np.asarray(fit.time, dtype=float)
    series = np.unique(fit.data_id)

    def pick(rows: np.ndarray, t: float | None) -> np.ndarray:
        if t is None:
            return means[:0]
        return means[rows[usable[rows] & (np.abs(time[rows] - t) <= config.atol)]]

    cols: dict[str, list] = {name: [] for name in (
        "n_amides", "n_undeut", "n_fd", "m0_measured", "mfd_measured", "m0", "m0_se", "mfd",
        "mfd_se", "retention", "fallback", "flags")}
    accepted: list[float] = []  # the retention of every accepted control (for 'median')
    pending: list[int] = []  # rows whose mFD falls back, filled in after the loop
    for s in series:
        rows = np.flatnonzero(fit.data_id == s)
        n_am = np.unique(n_amides[rows])
        if len(n_am) != 1:
            raise ValueError(f"series {s} has more than one amide count: {n_am}")
        n_am = float(n_am[0])
        m0_meas, m0_se, n_u = _mean_se(pick(rows, config.undeut_time))
        mfd_meas, mfd_se, n_f = _mean_se(pick(rows, config.fd_time))
        flags: list[str] = []
        if n_u == 0:
            flags.append("m0_missing")
        elif m0_meas > config.m0_max:
            flags.append("m0_high")
        m0_ok = n_u > 0 and m0_meas <= config.m0_max
        m0, m0_err = (m0_meas, m0_se) if m0_ok else (0.0, 0.0)
        if n_u > 0 and m0_ok and n_u == 1:
            flags.append("m0_unreplicated")
        mfd_ok = False
        floor = config.min_retention
        if config.fd_mode == "kint" and config.retention_min is None and (
                rates is None or int(s) not in rates):
            floor = config.d2o_fraction / 2  # no rates: the prediction cannot judge the control
        if n_f == 0:
            flags.append("fd_missing")
        elif n_am > 0:
            ret = (mfd_meas - m0) / n_am
            # a fully deuterated level at or below the undeuterated one cannot scale anything,
            # whatever the floor (none with fd_fallback='kint' for the series with rates)
            if ret <= 0 or ret < floor:
                flags.append("retention_low")
            elif ret > config.retention_max:
                flags.append("retention_high")
            else:
                mfd_ok = True
        else:
            flags.append("no_amides")
        if mfd_ok:
            mfd, mfd_err = mfd_meas, mfd_se
            accepted.append((mfd - m0) / n_am)
        else:
            mfd, mfd_err = m0 + config.d2o_fraction * n_am, 0.0
            pending.append(len(cols["mfd"]))
        if mfd_ok and n_f == 1:
            flags.append("fd_unreplicated")
        fallback = not (m0_ok and mfd_ok)
        values = {
            "n_amides": n_am, "n_undeut": n_u, "n_fd": n_f, "m0_measured": m0_meas,
            "mfd_measured": mfd_meas, "m0": m0, "m0_se": m0_err, "mfd": mfd, "mfd_se": mfd_err,
            "retention": (mfd - m0) / n_am if n_am > 0 else float("nan"),
            "fallback": fallback, "flags": ";".join(flags),
        }
        for name, value in values.items():
            cols[name].append(value)
    if config.fd_mode == "median" and accepted and pending:
        r = float(np.median(accepted))
        for i in pending:
            n_am, m0 = cols["n_amides"][i], cols["m0"][i]
            cols["mfd"][i] = m0 + r * n_am
            cols["retention"][i] = r if n_am > 0 else float("nan")
            cols["flags"][i] = ";".join(f for f in (cols["flags"][i], "fd_median") if f)
    fd_model = None
    if config.fd_mode == "kint":
        fd_model = _kint_fallback(cols, series, pending, rates, config, rt)
        if fd_model is None and config.retention_min is None:
            # nothing judges the rated series' controls: they get the d2o floor as well
            _apply_floor(cols, series, pending, rates, config)
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
        fd_model=fd_model,
    )


def _apply_floor(cols: dict, series: np.ndarray, pending: list[int], rates: Mapping,
                 config: ControlConfig) -> None:
    """Reject the accepted fully deuterated controls of the series with rates whose retention
    is below half of ``d2o_fraction`` (``retention_low``, the fallback ``d2o_fraction``), as
    the series without rates were (in ``cols``). For ``fd_fallback='kint'`` when the
    theoretical model was not fitted, so that no control is judged."""
    floor = config.d2o_fraction / 2
    low = [i for i, s in enumerate(series) if i not in set(pending) and int(s) in rates
           and cols["n_amides"][i] > 0 and cols["retention"][i] < floor]
    for i in low:
        n_am, m0 = cols["n_amides"][i], cols["m0"][i]
        cols["mfd"][i], cols["mfd_se"][i] = m0 + config.d2o_fraction * n_am, 0.0
        cols["retention"][i] = config.d2o_fraction
        cols["fallback"][i] = True
        cols["flags"][i] = ";".join([*(f for f in cols["flags"][i].split(";")
                                       if f and f != "fd_unreplicated"), "retention_low"])
    if low:
        log.warning("%d fully deuterated control(s) below the retention floor %.2f fall back to "
                    "d2o_fraction (no theoretical model to judge them)", len(low), floor)


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
                    "(kint_min_series %d); missing ones fall back to d2o_fraction",
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
                    "with it (kint_min_series %d)", len(kept), config.kint_min_series)
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
    """Correct means for back-exchange, ``(mean - m0) / (mFD - m0) * n_amides`` (not below 0).

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
    bad = (controls.mfd[rows] <= controls.m0[rows]) & (controls.n_amides[rows] > 0)
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
    return CorrectedMeans(mean=corrected, control_se=control_se, se=out_se, ci=out_ci)
