# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-spectrum quality flags: poor fit, low intensity and replicate outliers.

The flags only mark spectra; nothing is dropped here (see :mod:`hxex.run`, and the review layer
of the results bundle for the manual decision). Spectra that the picker marked empty or without
signal never reach the fit and are handled before it.

- ``poor_fit``: the fit is much worse than the others of its series. The statistic is the
  misfit of the selected-K fit relative to the spectrum's own intensity, ``sum((obs - fit)^2) /
  sum(obs^2)`` (raw intensities; it does not depend on the noise model), compared with the
  median of its own series (needs ``min_series_fits`` assessed spectra in the series). A
  spectrum's reduced deviance (``sum((obs - fit)^2)`` for Gaussian noise, the Poisson deviance
  for Poisson noise, in the noise-scaled units, divided by ``bins - 3K``) is not compared within
  the series, because at one noise level per series it grows with the height of the peaks: the
  undeuterated controls, the most peaked spectra, were flagged for the same relative misfit.
  It must still exceed ``min_deviance`` (a fit that is better than
  the noise model implies is not poor, however clean the rest of its series is: on the Bimodal
  data 11 of 12 relative-only flags were fits with deviance below 0.6).
- ``low_intensity``: the total intensity of the spectrum is below ``min_intensity`` times the
  mean of the other replicates at the same time point (needs one other replicate; spectra without
  replicates are not assessed, because the total legitimately changes along a time course).
- ``noise_outlier``: the replicate-consistency score of the noise estimate flagged the spectrum
  (:class:`hxex.noise.NoiseEstimate`); False when the noise was not estimated.
- ``mz_shift``: the picked peaks of the spectrum's series are, as a rule, further than
  ``max_mz_offset_ppm`` from their predicted m/z, compared with the run as a whole. The statistic
  (``mz_offset``) is the median over the series of each spectrum's offset (the picker's
  ``mz_offset_ppm``) minus the median of those series medians, which takes out the instrument's
  calibration. A series-wide shift means the peaks are not where the composition puts them: a
  wrong sequence, terminal group, modification or ion type, or a different species. On fragment
  data with the right composition the series are within about 3 ppm; a wrong C-terminus moved
  them by 23 to 41 ppm; on an intact-peptide dataset 2 % of series were beyond 15 ppm. Not
  assessed without the picker's offsets (a binned table read from
  elsewhere), nor for a series the picker's m/z check assessed (``off_mz`` replaces it there:
  the picker's offsets are measured inside its window, which a narrow window clips).
- ``off_mz``: the series' undeuterated controls put its envelope further from the run's
  calibration than ``[run.picker] off_mz_sd`` times the run's scatter
  (:mod:`hxex.picking.mz_check`): most likely the series reads another species (the
  retention time points at another peptide, or one shares its peaks). The statistic is
  ``envelope_offset`` (ppm). Peptide ions with undeuterated controls only; the analysis leaves
  such series out (``[analysis.filters] off_mz``).
- ``rt_found``: the series was read at the retention time found for its exact m/z in the
  undeuterated controls, not the peptide table's (``[input.scans] find_retention_time =
  "automatic"``, :mod:`hxex.io.retention_search`): check the time
  (``retention_suggestions.csv``) and correct the table.
- ``poor_envelope``: the series' ion-type envelope (``mixed`` or ``infer``) was not measured
  well (:attr:`hxex.picking.EnvelopeTable.poor`): left with the theory of one ion type for want
  of usable reference spectra, not describing the spectra it was fitted to, or, shared between
  conditions, not describing the series' own references. A fragment's envelope is a mixture of
  ion types (``c-1`` and ``c``, say) that the theory cannot predict, so its deuteration is then
  off by up to a hydrogen. False with the theory envelope.
- ``saturated`` and ``tails_low``: the isotope pattern of an undeuterated control, whose true
  shape is the natural abundance of its composition, compared with that shape (the height
  fixed at the tallest predicted peak; :func:`control_shape_ratios`). ``saturated``: the peaks
  beside the tallest one (predicted at 30 to 100 % of it) read above ``saturated_ratio`` times
  their prediction, that is the tallest peak reads low, the sign of detector saturation.
  ``tails_low``: the small peaks (predicted at 1 to 10 % of the top, at least two of them) read
  below ``tails_low_ratio`` times their prediction; seen in bright spectra on every dataset
  examined, following the height relative to the top, its cause unknown (HISTORY 156). A
  control is flagged only when most of its series' control replicates show the sign (a
  single replicate's tail is noisy), and only the brighter half of the run's undeuterated
  controls (tallest stick at or above their median) is judged: the signs belong to bright
  spectra, and a weak control's peak ratios are noise.
- ``bright``: a labelled spectrum or fully deuterated control whose tallest stick is at least as
  high as that of a control of its series flagged ``saturated`` or ``tails_low``: likely
  distorted the same way. Their own shape cannot be checked, since it is not known in advance.
"""
# evidence for the screens: docs/HISTORY.md items 37, 57 and 87

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from .fit.measurements import Measurements
from .fit.result import FitResult
from .model import Backend
from .model.mixture import MixtureBatch, PaddedBatch, shape_weights
from .spectra import UNDEUT, time_point

FLAG_NAMES = ("poor_fit", "low_intensity", "noise_outlier", "mz_shift", "off_mz",
              "poor_envelope", "saturated", "tails_low", "bright", "rt_found")


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds of the quality screens (the ``[run.quality]`` table; see the module docstring
    for each flag).

    - ``poor_fit_factor``: ``poor_fit`` when the misfit is this many times the series' median,
      and ``min_deviance``: the reduced deviance must also exceed this; ``min_series_fits``:
      the fewest fitted spectra in a series for the screen to judge.
    - ``min_intensity``: ``low_intensity`` when the total intensity is below this share of the
      other replicates' mean at the same time.
    - ``max_mz_offset_ppm``: ``mz_shift`` when a series' m/z offset against the run is beyond
      this; None: screen off.
    - ``saturated_ratio``: ``saturated`` when an undeuterated control's peaks beside the
      tallest read above this times natural abundance; None: off.
    - ``tails_low_ratio``: ``tails_low`` when an undeuterated control's tail peaks read below
      this times natural abundance; None: off.
    """
    # defaults set on the private pooled dataset: docs/HISTORY.md item 28

    poor_fit_factor: float = 10.0
    min_deviance: float = 1.0
    min_series_fits: int = 5
    min_intensity: float = 0.3
    max_mz_offset_ppm: float | None = 15.0
    saturated_ratio: float | None = 1.15
    tails_low_ratio: float | None = 0.7

    def __post_init__(self):
        if not self.poor_fit_factor > 1:
            raise ValueError("poor_fit_factor must be > 1")
        if self.min_deviance < 0:
            raise ValueError("min_deviance must be >= 0")
        if self.min_series_fits < 2:
            raise ValueError("min_series_fits must be >= 2")
        if not 0 < self.min_intensity < 1:
            raise ValueError("min_intensity must be between 0 and 1")
        if self.max_mz_offset_ppm is not None and not self.max_mz_offset_ppm > 0:
            raise ValueError("max_mz_offset_ppm must be > 0 (or None to skip the screen)")
        if self.saturated_ratio is not None and not self.saturated_ratio > 1:
            raise ValueError("saturated_ratio must be > 1 (or None to skip the screen)")
        if self.tails_low_ratio is not None and not 0 < self.tails_low_ratio < 1:
            raise ValueError("tails_low_ratio must be between 0 and 1 (or None to skip the screen)")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QualityConfig:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown quality setting(s): {', '.join(sorted(unknown))}")
        return cls(**data)


@dataclass(frozen=True)
class QualityFlags:
    """The statistics and flags of every measurement (arrays of one entry per measurement).

    ``deviance`` is the reduced deviance of the selected-K fit and ``deviance_rel`` its ratio to the
    median of the series; ``misfit`` is ``sum((obs - fit)^2) / sum(obs^2)`` of that fit and
    ``misfit_rel`` its ratio to the median of the series (the ``poor_fit`` statistic; NaN in flags
    read from a bundle written before it); ``intensity`` is the total intensity and
    ``intensity_rel`` its ratio to the mean of the other replicates at the same time; ``mz_offset``
    is the series' m/z offset relative to the run, in ppm, and ``envelope_offset`` the picker's
    m/z check of the series (``off_mz``), in ppm. NaN where not assessed (the flag is False
    there). ``flagged`` is True where any flag is set. Flags read from a bundle written before
    them are False.
    """

    config: QualityConfig
    spectrum_id: np.ndarray
    deviance: np.ndarray
    deviance_rel: np.ndarray
    intensity: np.ndarray
    intensity_rel: np.ndarray
    poor_fit: np.ndarray
    low_intensity: np.ndarray
    noise_outlier: np.ndarray
    mz_offset: np.ndarray | None = None
    mz_shift: np.ndarray | None = None
    misfit: np.ndarray | None = None
    misfit_rel: np.ndarray | None = None
    poor_envelope: np.ndarray | None = None
    saturated: np.ndarray | None = None
    tails_low: np.ndarray | None = None
    bright: np.ndarray | None = None
    envelope_offset: np.ndarray | None = None
    off_mz: np.ndarray | None = None
    rt_found: np.ndarray | None = None

    def __post_init__(self):  # flags read from a bundle written before the m/z screen, misfit
        for name in ("mz_offset", "misfit", "misfit_rel", "envelope_offset"):  # or a flag
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.full(len(self.spectrum_id), np.nan))
        for name in ("mz_shift", "off_mz", "poor_envelope", "saturated", "tails_low", "bright",
                     "rt_found"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.zeros(len(self.spectrum_id), dtype=bool))

    def __len__(self) -> int:
        return len(self.spectrum_id)

    @property
    def flagged(self) -> np.ndarray:
        out = np.zeros(len(self), dtype=bool)
        for name in FLAG_NAMES:
            out |= np.asarray(getattr(self, name), dtype=bool)
        return out


def reduced_deviance(meas: Measurements, fit: FitResult) -> np.ndarray:
    """The reduced deviance of every measurement at its selected K (NaN where ``bins <= 3K``)."""
    return fit_statistics(meas, fit)[0]


def fit_statistics(meas: Measurements, fit: FitResult) -> tuple[np.ndarray, np.ndarray]:
    """The reduced deviance and the misfit ``sum((obs - fit)^2) / sum(obs^2)`` (raw
    intensities) of every measurement at its selected K (NaN where ``bins <= 3K``)."""
    backend = Backend("cpu", "float64")
    out = np.full(len(fit), np.nan)
    misfit = np.full(len(fit), np.nan)
    for k in fit.k_values:
        idx = np.flatnonzero(fit.selected_K == k)
        if not len(idx):
            continue
        kf = fit.fits[k]
        bins = [meas.bins[i] for i in idx]
        batch = PaddedBatch.from_arrays(
            bins, [np.zeros(len(b)) for b in bins], meas.n_max[idx], backend
        )
        model = MixtureBatch(
            batch, shape_weights([meas.shapes[i] for i in idx], k, backend),
            backend.tensor(kf.log_A[idx]), backend.tensor(kf.n_raw[idx]),
            backend.tensor(kf.logit_p[idx]), "gaussian",
        )
        pred = model.predict().detach().cpu().numpy()
        for j, i in enumerate(idx):
            obs = np.asarray(meas.observed[i], dtype=float) / fit.obs_scale[i]
            p = np.maximum(pred[: len(obs), j], 1e-12)
            dof = len(obs) - 3 * k
            if dof <= 0:
                continue
            if fit.noise_kind[i] == "poisson":
                pos = obs > 0
                term = np.zeros_like(obs)
                term[pos] = obs[pos] * np.log(obs[pos] / p[pos])
                total = 2.0 * (term - (obs - p)).sum()
            else:
                total = ((obs - p) ** 2).sum()
            out[i] = total / dof
            power = float((obs**2).sum())
            if power > 0:  # the same in raw units: the scale cancels
                misfit[i] = float(((obs - p) ** 2).sum()) / power
    return out, misfit


def series_mz_offset(data_id: np.ndarray, offset_ppm: np.ndarray) -> np.ndarray:
    """Each spectrum's series median of ``offset_ppm`` minus the median of the series medians
    (NaN for a series without a finite offset)."""
    data_id = np.asarray(data_id)
    offset_ppm = np.asarray(offset_ppm, dtype=float)
    out = np.full(len(data_id), np.nan)
    medians = {}
    for s in np.unique(data_id):
        rows = data_id == s
        v = offset_ppm[rows & np.isfinite(offset_ppm)]
        if len(v):
            medians[s] = float(np.median(v))
    if not medians:
        return out
    ref = float(np.median(list(medians.values())))
    for s, m in medians.items():
        out[data_id == s] = m - ref
    return out


def control_shape_ratios(meas: Measurements) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each undeuterated control against the natural-abundance shape of its composition, the
    height fixed at the tallest predicted peak: the median observed / predicted of the peaks
    beside it (predicted at 30 to 100 % of the top), of the tail peaks (1 to 10 %; NaN with fewer
    than two), and the spectrum's tallest stick. NaN for every other spectrum, and for a control
    whose tallest predicted peak was not picked."""
    m = len(meas)
    near, tail, top = np.full(m, np.nan), np.full(m, np.nan), np.full(m, np.nan)
    role = np.asarray(meas.role, dtype=str) if meas.role is not None else np.full(m, "")
    for i in np.flatnonzero(role == UNDEUT):
        y = np.asarray(meas.observed[i], dtype=np.float64)
        k = np.asarray(meas.bins[i]).astype(int)
        shape = np.asarray(meas.shapes[i], dtype=np.float64)
        ok = (k >= 0) & (k < len(shape))
        if not len(y) or y.max() <= 0 or not ok.any():
            continue
        rel = shape[k[ok]] / shape.max()
        t = int(np.argmax(rel))
        if rel[t] < 1 or y[ok][t] <= 0:
            continue
        ratio = y[ok] / (y[ok][t] * shape[k[ok]] / shape.max())
        beside = (rel >= 0.3) & (rel < 1)
        tails = (rel >= 0.01) & (rel < 0.1)
        if beside.any():
            near[i] = np.median(ratio[beside])
        if tails.sum() >= 2:
            tail[i] = np.median(ratio[tails])
        top[i] = y.max()
    return near, tail, top


def _brightness_flags(meas: Measurements, data_id: np.ndarray, config: QualityConfig):
    """``saturated``, ``tails_low`` (undeuterated controls, when most of the series' control
    replicates show the sign) and ``bright`` (the series' labelled spectra and fully deuterated
    controls at least as bright as a flagged control), see the module docstring."""
    m = len(meas)
    flags = {name: np.zeros(m, dtype=bool) for name in ("saturated", "tails_low", "bright")}
    if config.saturated_ratio is None and config.tails_low_ratio is None:
        return flags
    near, tail, top = control_shape_ratios(meas)
    role = np.asarray(meas.role, dtype=str) if meas.role is not None else np.full(m, "")
    peak = np.array([float(np.max(o)) if len(o) else 0.0 for o in meas.observed])
    tests = (("saturated", near, lambda r: r > config.saturated_ratio, config.saturated_ratio),
             ("tails_low", tail, lambda r: r < config.tails_low_ratio, config.tails_low_ratio))
    # only the brighter half of the run's controls is judged: the signs belong to bright
    # spectra, and a weak control's peak ratios are noise (6 % of the controls below 5,000
    # read 'saturated' on two datasets without this)
    assessed = np.isfinite(top)
    level = float(np.median(top[assessed])) if assessed.any() else np.inf
    for s in np.unique(data_id):
        rows = np.flatnonzero(data_id == s)
        controls = rows[role[rows] == UNDEUT]
        others = rows[role[rows] != UNDEUT]  # labelled spectra and fully deuterated controls
        for name, stat, test, limit in tests:
            if limit is None:
                continue
            judged = controls[np.isfinite(stat[controls]) & (top[controls] >= level)]
            if not len(judged):
                continue
            hit = judged[test(stat[judged])]
            if 2 * len(hit) > len(judged):  # most replicates
                flags[name][hit] = True
                flags["bright"][others[peak[others] >= top[hit].min()]] = True
    return flags


def assess_quality(
    meas: Measurements,
    fit: FitResult,
    config: QualityConfig = QualityConfig(),  # noqa: B008
    mz_offset_ppm: np.ndarray | None = None,
    poor_envelope_series: np.ndarray | None = None,
    envelope_offset_ppm: np.ndarray | None = None,
    envelope_offset_limit_ppm: np.ndarray | None = None,
    rt_found_series: np.ndarray | None = None,
) -> QualityFlags:
    """Quality statistics and flags of a fit (see the module docstring). ``mz_offset_ppm`` is
    the picker's offset of every measurement (:attr:`~hxex.picking.BinnedSet.mz_offset_ppm`);
    without it the m/z screen is not assessed. ``envelope_offset_ppm`` and
    ``envelope_offset_limit_ppm`` are the picker's m/z check of every measurement's series
    (:attr:`~hxex.picking.BinnedSet.envelope_offset_ppm`; NaN or None: not assessed).
    ``poor_envelope_series`` lists the series (``data_id``) whose envelope is poor
    (:attr:`hxex.picking.EnvelopeTable.poor`); ``rt_found_series`` those read at the
    retention time found for their exact m/z (:mod:`hxex.io.retention_search`)."""
    if not np.array_equal(meas.spectrum_id, fit.spectrum_id):
        raise ValueError("the measurements and the fit must be for the same spectra")
    m = len(fit)
    deviance, misfit = fit_statistics(meas, fit)
    total = np.array([float(np.sum(o)) for o in meas.observed])
    deviance_rel = np.full(m, np.nan)
    misfit_rel = np.full(m, np.nan)
    intensity_rel = np.full(m, np.nan)
    time = time_point(fit.time, fit.role)  # replicates share a point: a time, or a control role
    for s in np.unique(fit.data_id):
        rows = np.flatnonzero(fit.data_id == s)
        assessed = rows[np.isfinite(deviance[rows])]
        if len(assessed) >= config.min_series_fits:
            median = np.median(deviance[assessed])
            if median > 0:
                deviance_rel[assessed] = deviance[assessed] / median
            median = np.median(misfit[assessed])
            if median > 0:
                misfit_rel[assessed] = misfit[assessed] / median
        for t in np.unique(time[rows]):
            same = rows[time[rows] == t]
            if len(same) < 2:
                continue
            for i in same:
                others = total[same[same != i]].mean()
                if others > 0:
                    intensity_rel[i] = total[i] / others
    noise_outlier = (
        np.asarray(fit.noise_estimate.outlier, dtype=bool)
        if fit.noise_estimate is not None else np.zeros(m, dtype=bool)
    )
    mz_offset = np.full(m, np.nan)
    if mz_offset_ppm is not None:
        if len(mz_offset_ppm) != m:
            raise ValueError("mz_offset_ppm must have one value per measurement")
        mz_offset = series_mz_offset(fit.data_id, mz_offset_ppm)
    envelope_offset = np.full(m, np.nan)
    envelope_limit = np.full(m, np.nan)
    for name, given, out in (("envelope_offset_ppm", envelope_offset_ppm, envelope_offset),
                             ("envelope_offset_limit_ppm", envelope_offset_limit_ppm,
                              envelope_limit)):
        if given is not None:
            if len(given) != m:
                raise ValueError(f"{name} must have one value per measurement")
            out[:] = np.asarray(given, dtype=float)
    checked = np.isfinite(envelope_offset) & np.isfinite(envelope_limit)
    off_mz = checked & (np.abs(np.nan_to_num(envelope_offset)) > np.nan_to_num(envelope_limit))
    limit = config.max_mz_offset_ppm
    mz_shift = (np.zeros(m, dtype=bool) if limit is None
                else (np.abs(np.nan_to_num(mz_offset)) > limit) & ~checked)
    brightness = _brightness_flags(meas, np.asarray(fit.data_id), config)
    return QualityFlags(
        config=config, spectrum_id=fit.spectrum_id, deviance=deviance,
        deviance_rel=deviance_rel, intensity=total, intensity_rel=intensity_rel,
        poor_fit=(np.nan_to_num(misfit_rel) > config.poor_fit_factor)
        & (np.nan_to_num(deviance) > config.min_deviance),
        low_intensity=np.nan_to_num(intensity_rel, nan=1.0) < config.min_intensity,
        noise_outlier=noise_outlier, mz_offset=mz_offset, mz_shift=mz_shift,
        envelope_offset=envelope_offset, off_mz=off_mz,
        misfit=misfit, misfit_rel=misfit_rel,
        poor_envelope=np.isin(fit.data_id, np.asarray(
            [] if poor_envelope_series is None else poor_envelope_series, dtype=np.int64)),
        rt_found=np.isin(fit.data_id, np.asarray(
            [] if rt_found_series is None else rt_found_series, dtype=np.int64)),
        **brightness,
    )
