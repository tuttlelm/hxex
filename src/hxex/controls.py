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
  half of ``d2o_fraction``) or above ``retention_max`` (``retention_high``), with ``m0`` as it is
  used. The fallback retention ``r`` is ``d2o_fraction`` (``fd_fallback='d2o'``, the deuterium
  fraction of the labelling buffer, a user setting) or the median retention of the run's
  accepted controls (``fd_fallback='median'``; flagged ``fd_median``), which includes the
  back-exchange the buffer's fraction does not.

A measured control with a single replicate has no uncertainty (flagged ``m0_unreplicated`` or
``fd_unreplicated``); a theoretical value is exact. The corrected mean of a population is
``(mean - m0) / (mFD - m0) * n_amides``, not below 0 (:func:`correct_means`); the uncertainty of
the controls is added to the uncertainty of the mean.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy.stats import norm

from .fit.result import FitResult

FD_FALLBACKS = ("d2o", "median")


@dataclass(frozen=True)
class ControlConfig:
    """Control times (in the units of the measurement times) and the limits for a good control.

    A time of None means that the data has no control of that kind. The defaults of the limits are
    those of the notebook the method comes from; on the Bimodal data they hardly matter (see
    docs/HISTORY.md item 27). ``d2o_fraction`` is the deuterium fraction of the labelling buffer
    (often 0.8 to 0.9; the default is pyHXExpress's ``Dfrac``). ``retention_min`` None (the
    default) means half of it. ``fd_fallback`` is the retention used when a series' fully
    deuterated control is missing or rejected: ``'d2o'`` (``d2o_fraction``) or ``'median'`` (the
    median of the run's accepted controls; ``d2o_fraction`` if there are none). See
    docs/HISTORY.md items 59 and 61.
    """

    undeut_time: float | None = None
    fd_time: float | None = None
    m0_max: float = 0.5
    retention_min: float | None = None
    retention_max: float = 1.1
    d2o_fraction: float = 0.9
    fd_fallback: str = "d2o"
    atol: float = 1e-6

    def __post_init__(self):
        if self.m0_max < 0:
            raise ValueError("m0_max must be >= 0")
        if not 0 <= self.min_retention < self.retention_max:
            raise ValueError("need 0 <= retention_min < retention_max")
        if self.fd_fallback not in FD_FALLBACKS:
            raise ValueError(f"fd_fallback must be one of {', '.join(FD_FALLBACKS)}")
        if self.atol < 0:
            raise ValueError("atol must be >= 0")
        if not 0 < self.d2o_fraction <= 1:
            raise ValueError("d2o_fraction must be in (0, 1]")

    @property
    def min_retention(self) -> float:
        """The lower retention limit in use (``retention_min``, or half of ``d2o_fraction``)."""
        return self.d2o_fraction / 2 if self.retention_min is None else self.retention_min

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

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
    the reasons, separated by ``;`` (see the module docstring).
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


def fit_controls(
    fit: FitResult,
    n_amides,
    config: ControlConfig,
    usable: np.ndarray | None = None,
) -> ControlTable:
    """The control table of a fit that includes K=1 (``fit.fits[1]``).

    ``n_amides`` is the number of exchangeable amides of each measurement's peptide, ``(M,)``
    (constant within a series). ``usable`` (``(M,)`` bool) marks the spectra that may serve as
    controls; by default every spectrum that the noise estimate did not flag as an outlier.
    """
    if 1 not in fit.fits:
        raise ValueError("the controls come from the K=1 fits: fit with k_min=1")
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
        if n_f == 0:
            flags.append("fd_missing")
        elif n_am > 0:
            ret = (mfd_meas - m0) / n_am
            if ret < config.min_retention:
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
    if config.fd_fallback == "median" and accepted and pending:
        r = float(np.median(accepted))
        for i in pending:
            n_am, m0 = cols["n_amides"][i], cols["m0"][i]
            cols["mfd"][i] = m0 + r * n_am
            cols["retention"][i] = r if n_am > 0 else float("nan")
            cols["flags"][i] = ";".join(f for f in (cols["flags"][i], "fd_median") if f)
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
    )


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
