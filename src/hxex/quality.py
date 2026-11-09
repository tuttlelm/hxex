# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-spectrum quality flags: poor fit, low intensity and replicate outliers.

The flags only mark spectra; nothing is dropped here (see :mod:`hxex.pipeline`, and the review layer
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
- ``drift_unclear``: the series was read from scans with ion mobility bins without a clear drift
  range (``[input.scans] drift_gate``, :mod:`hxex.io.drift_gate`): no undeuterated control with
  signal, so every drift time was summed, or its undeuterated drift profile has a second peak
  outside the range (another ion at that m/z, or the peptide in two shapes): check its
  spectra, or give the peptide its own ``drift_start``, ``drift_end``.
- ``precursor_off``: the precursor's own ion in the fragment scans (``M``, unreacted: at the
  parent's charge) reads more than ``max_precursor_offset`` deuterons (K = 1 means) from its
  parent's MS1 spectrum in the same file: most likely another species in the isolation window
  shares its m/z. Measured on a targeted dataset (HISTORY 307): 3+ and 4+ precursors within
  0.14 D (90th percentile; 2 of 204 spectra beyond 1 D), 2+ precursors 2.1 D (median; 57 of 85
  beyond 1 D). The statistic is ``precursor_offset`` (D, M minus MS1); not assessed without the
  parents' MS1 spectra or the K = 1 fits. The closure test takes the MS1 spectra as its
  reference where they were read (:func:`hxex.analysis.parent_uptake`).
- ``overlapping_ion``: another ion read from the same scans (a c-1 or z-dot fragment of the same
  parent with its mixed components, or the precursor ``M``) puts at least ``max_ion_overlap`` of
  this fragment's picked intensity on its bins: its peaks fall inside the picker's windows. The
  other ion's fitted curve is scaled by its own bins away from this fragment's strong bins (those
  holding at least a fifth of the fragment's own fitted curve), so an ion inside the fragment's
  reachable range, every bin of which lies near one of the fragment's, still counts (an ion
  that coincides with every strong bin cannot be told apart and is not counted). The statistic is
  ``ion_overlap`` (the share), with the other ion in ``ion_overlap_with``. Measured on a targeted
  dataset (2026-11-06): 7-8 % of fragment spectra at 0.2 or more, at every time point; ions
  of one ladder whose masses differ by close to whole isotope steps (a c and a z ion lacking
  different end residues), two charges of one ion type, and the bright precursor at 3+ or 4+.
  Ions that are not read (a, b and y ions) are not assessed.
- ``below_noise``: a fragment spectrum fitted through the instrument's per-scan cut
  (``[input.scans] scan_cut``) whose tallest peak, in the scan where it is largest, is below
  ``min_signal_to_noise`` (4) times the noise that scan recorded there: in reduced profile scans
  so many of its peaks were deleted scan by scan that even the cut-aware fit cannot recover its
  deuteration (HISTORY 308: below about 4 x at the elution's top, no fit does). The statistic is
  ``signal_to_noise`` (the tallest bin's reading times its scan's share of the signal, over that
  scan's noise there, at the scan where this is largest; 0 for a spectrum with nothing read).
  Measured on a targeted dataset (2026-11-07), fits through the cut: 38 % of the labelled and
  fully deuterated fragment spectra below 4 x the noise (23 % below 1); the closure (c + z - MS1
  parent) missed by a median -0.37 D at 1-4 x and -0.11 D at 4-16 x, and MS2 and MS3 readings of
  one fragment scattered 0.50 D at 1-4 x, 0.29 D at 4-16 x and 0.19 D above. Not assessed (NaN)
  without the scans' record.
- ``elution_cut``: a fragment spectrum whose elution peak, found per run (``[input.scans]
  find_fragment_peak``), reaches the first or last scan the instrument recorded for its
  precursor: the scheduled acquisition cut the peak (retention moved in that run), so part of
  the signal was never recorded and the spectrum is dimmer than its elution. Per spectrum.
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
- ``undeut_high``: an undeuterated control whose K=1 mean is above ``[run.controls] m0_max``
  (deuterons): an undeuterated peptide sits near 0, so the spectrum holds another species or
  a wrong pick. Judged one spectrum at a time, so a series without replicates is flagged too
  (owner, 2026-11-05). The controls leave it out of their mean when the series has a good
  undeuterated spectrum, and the analysis then leaves it out of its undeuterated rows (an
  ``accepted`` review keeps it there); a series whose every undeuterated spectrum is above it is
  ``m0_high``, which ``[analysis.filters] stand_in_controls`` decides. Not assessed for series
  the controls leave uncorrected (fragment ions) nor without controls.
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
              "poor_envelope", "saturated", "tails_low", "bright", "rt_found", "undeut_high",
              "drift_unclear", "elution_cut", "precursor_off", "overlapping_ion",
              "below_noise")


# The version of the checks behind the flags: add one whenever a flag or statistic is added or
# what one tests changes, so that a bundle whose flags an earlier version assessed says so and
# `hxex resume` assesses them again (owner, 2026-11-08: a bundle fitted before overlapping_ion
# existed kept it empty through later resumes). 0: flags read from a bundle that does not record
# it. 1: every flag up to overlapping_ion and below_noise (HISTORY 309, 314). 2: overlapping_ion
# scales the other ion by its bins away from the fragment's strong bins (was: all its bins).
QUALITY_VERSION = 2


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
    - ``max_precursor_offset`` (deuterons): ``precursor_off`` when a precursor ion in the
      fragment scans reads further than this from its parent's MS1 spectrum; None: off.
    - ``max_ion_overlap``: ``overlapping_ion`` when another ion of the same scans puts at least
      this share of a fragment's picked intensity on its bins; None: off.
    - ``min_signal_to_noise``: ``below_noise`` when a fragment spectrum fitted through the
      per-scan cut has its tallest peak below this times its scan's recorded noise; None: off.
    """
    # defaults set on the private pooled dataset: docs/HISTORY.md item 28

    poor_fit_factor: float = 10.0
    min_deviance: float = 1.0
    min_series_fits: int = 5
    min_intensity: float = 0.3
    max_mz_offset_ppm: float | None = 15.0
    saturated_ratio: float | None = 1.15
    tails_low_ratio: float | None = 0.7
    max_precursor_offset: float | None = 1.0
    max_ion_overlap: float | None = 0.2
    min_signal_to_noise: float | None = 4.0

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
        if self.max_precursor_offset is not None and not self.max_precursor_offset > 0:
            raise ValueError("max_precursor_offset must be > 0 (or None to skip the screen)")
        if self.max_ion_overlap is not None and not 0 < self.max_ion_overlap <= 1:
            raise ValueError("max_ion_overlap must be between 0 and 1 (or None to skip the screen)")
        if self.tails_low_ratio is not None and not 0 < self.tails_low_ratio < 1:
            raise ValueError("tails_low_ratio must be between 0 and 1 (or None to skip the screen)")
        if self.min_signal_to_noise is not None and not self.min_signal_to_noise > 0:
            raise ValueError("min_signal_to_noise must be > 0 (or None to skip the screen)")

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
    undeut_high: np.ndarray | None = None
    drift_unclear: np.ndarray | None = None
    elution_cut: np.ndarray | None = None
    precursor_offset: np.ndarray | None = None
    precursor_off: np.ndarray | None = None
    ion_overlap: np.ndarray | None = None
    ion_overlap_with: np.ndarray | None = None
    overlapping_ion: np.ndarray | None = None
    signal_to_noise: np.ndarray | None = None
    below_noise: np.ndarray | None = None
    version: int = QUALITY_VERSION  # of the checks that assessed them (QUALITY_VERSION)

    @property
    def outdated(self) -> bool:
        """Whether an earlier version of the checks assessed these flags (a flag added since is
        False and its statistic NaN, not assessed)."""
        return self.version < QUALITY_VERSION

    def __post_init__(self):  # flags read from a bundle written before the m/z screen, misfit
        for name in ("mz_offset", "misfit", "misfit_rel", "envelope_offset",  # or a flag
                     "precursor_offset", "ion_overlap", "signal_to_noise"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.full(len(self.spectrum_id), np.nan))
        for name in ("mz_shift", "off_mz", "poor_envelope", "saturated", "tails_low", "bright",
                     "rt_found", "undeut_high", "drift_unclear", "elution_cut",
                     "precursor_off", "overlapping_ion", "below_noise"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.zeros(len(self.spectrum_id), dtype=bool))
        if self.ion_overlap_with is None:
            object.__setattr__(self, "ion_overlap_with",
                               np.full(len(self.spectrum_id), "", dtype=object))

    def __len__(self) -> int:
        return len(self.spectrum_id)

    @property
    def flagged(self) -> np.ndarray:
        out = np.zeros(len(self), dtype=bool)
        for name in FLAG_NAMES:
            out |= np.asarray(getattr(self, name), dtype=bool)
        return out


def undeut_high(fit: FitResult, controls) -> np.ndarray:
    """The ``undeut_high`` flag of every measurement: an undeuterated control (``fit.role``)
    whose K=1 mean is above ``controls.config.m0_max``; False without controls or a K=1 fit,
    and for the series ``controls`` leaves uncorrected."""
    m = len(fit)
    if controls is None or 1 not in fit.fits or fit.role is None:
        return np.zeros(m, dtype=bool)
    means = np.asarray(fit.fits[1].means, dtype=float)[:, 0]
    raw = np.asarray(controls.uncorrected, dtype=bool)[controls.rows(fit.data_id)]
    with np.errstate(invalid="ignore"):
        return ((np.asarray(fit.role, dtype=str) == UNDEUT) & ~raw & np.isfinite(means)
                & (means > controls.config.m0_max))


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
        # a picked bin can lie where the composition predicts nothing (rel 0): no ratio there
        ratio = np.full(len(rel), np.nan)
        ratio[rel > 0] = y[ok][rel > 0] / (y[ok][t] * rel[rel > 0])
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
    controls=None,
    drift_unclear_series: np.ndarray | None = None,
    elution_cut_spectra=None,
    precursor_offset: np.ndarray | None = None,
    ion_overlap: tuple[np.ndarray, np.ndarray] | None = None,
) -> QualityFlags:
    """Quality statistics and flags of a fit (see the module docstring). ``mz_offset_ppm`` is
    the picker's offset of every measurement (:attr:`~hxex.picking.BinnedSet.mz_offset_ppm`);
    without it the m/z screen is not assessed. ``envelope_offset_ppm`` and
    ``envelope_offset_limit_ppm`` are the picker's m/z check of every measurement's series
    (:attr:`~hxex.picking.BinnedSet.envelope_offset_ppm`; NaN or None: not assessed).
    ``poor_envelope_series`` lists the series (``data_id``) whose envelope is poor
    (:attr:`hxex.picking.EnvelopeTable.poor`); ``rt_found_series`` those read at the
    retention time found for their exact m/z (:mod:`hxex.io.retention_search`);
    ``drift_unclear_series`` those read without a clear drift range
    (:mod:`hxex.io.drift_gate`); ``elution_cut_spectra`` the spectra (ids) whose found
    fragment peak the acquisition cut (:func:`hxex.io.mzml.find_fragment_peaks`);
    ``precursor_offset`` every measurement's :func:`precursor_offsets` (None: not assessed).
    ``ion_overlap``: :func:`ion_overlaps` (the share and the other ion; None: not assessed).
    ``controls``
    (:class:`hxex.controls.ControlTable`) gives ``undeut_high`` its limit (None: not
    assessed)."""
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
    offset = (np.full(m, np.nan) if precursor_offset is None
              else np.asarray(precursor_offset, dtype=float))
    if len(offset) != m:
        raise ValueError("precursor_offset must have one value per measurement")
    limit_d = config.max_precursor_offset
    precursor_off = (np.zeros(m, dtype=bool) if limit_d is None
                     else np.abs(np.nan_to_num(offset)) > limit_d)
    overlap, overlap_with = ((np.full(m, np.nan), np.full(m, "", dtype=object))
                             if ion_overlap is None else
                             (np.asarray(ion_overlap[0], dtype=float),
                              np.asarray(ion_overlap[1], dtype=object)))
    if len(overlap) != m or len(overlap_with) != m:
        raise ValueError("ion_overlap must have one value per measurement")
    overlapping = (np.zeros(m, dtype=bool) if config.max_ion_overlap is None
                   else np.nan_to_num(overlap) >= config.max_ion_overlap)
    snr = signal_to_noise(meas)
    below = (np.zeros(m, dtype=bool) if config.min_signal_to_noise is None
             else np.nan_to_num(snr, nan=np.inf) < config.min_signal_to_noise)
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
        drift_unclear=np.isin(fit.data_id, np.asarray(
            [] if drift_unclear_series is None else drift_unclear_series, dtype=np.int64)),
        elution_cut=np.isin(np.asarray(fit.spectrum_id, dtype=str), np.asarray(
            [] if elution_cut_spectra is None else list(elution_cut_spectra), dtype=str)),
        undeut_high=undeut_high(fit, controls),
        precursor_offset=offset, precursor_off=precursor_off,
        ion_overlap=overlap, ion_overlap_with=np.where(overlapping, overlap_with, ""),
        overlapping_ion=overlapping, signal_to_noise=snr, below_noise=below,
        **brightness,
    )


def signal_to_noise(meas: Measurements) -> np.ndarray:
    """Every measurement's ``signal_to_noise`` (``below_noise``): for a spectrum with its scans'
    record (fitted through the per-scan cut), the largest over its bins and scans of the bin's
    reading times the scan's share of the signal over the noise that scan recorded at the bin
    (the size of the peak in that scan against the instrument's threshold; 0 when nothing was
    read); NaN without the record."""
    out = np.full(len(meas.bins), np.nan)
    if meas.scan_cut is None:
        return out
    for i, cut in enumerate(meas.scan_cut):
        if cut is None:
            continue
        obs = np.clip(np.asarray(meas.observed[i], dtype=float), 0.0, None)
        w = np.asarray(cut.weights, dtype=float)
        noise = np.asarray(cut.noise, dtype=float).reshape(len(obs), len(w))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(noise > 0, obs[:, None] * w[None, :] / noise, np.nan)
        out[i] = 0.0 if not obs.any() else (float(np.nanmax(ratio)) if np.isfinite(ratio).any()
                                            else np.nan)
    return out


def precursor_offsets(fit: FitResult, source, labels) -> np.ndarray:
    """Every measurement's precursor offset (deuterons): for a precursor ion read in fragment
    scans (ion type ``M`` at its parent's charge: the unreacted precursor), its K = 1 mean
    minus that of its parent's MS1 spectrum in the same file (``labels``: the spectra's labels,
    ``'<file> scans...'``); NaN for every other measurement and where the file has no MS1
    spectrum of the parent, or without the K = 1 fits or a scan-file read record
    (``source``)."""
    from .io.mzml import SCAN_READERS, mzml_series_targets

    m = len(fit)
    out = np.full(m, np.nan)
    if not source or source.get("reader") not in SCAN_READERS or 1 not in fit.fits:
        return out
    targets = mzml_series_targets(source)
    means = fit.fits[1].means[:, 0]
    files = [str(label).rsplit(" ", 1)[0] for label in labels]
    ms1, precursors = {}, []
    for i in range(m):
        t = targets.get(int(fit.data_id[i]))
        if t is None:
            continue
        e = t.entry
        extra = dict(e.extra)
        if "fragment" not in extra:
            if t.ms_level == 1:
                ms1[(files[i], e.peptide, str(e.peptide_range), int(e.charge))] = means[i]
        elif str(extra["fragment"]).startswith("M:") and int(e.charge) == int(
                extra["parent_charge"]):
            precursors.append((i, (files[i], extra["parent_peptide"],
                                   str(extra["parent_range"]), int(extra["parent_charge"]))))
    for i, key in precursors:
        if key in ms1:
            out[i] = means[i] - ms1[key]
    return out


def ion_overlaps(data: Measurements, fit: FitResult, source, labels, mz, picker
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Every fragment measurement's overlap with the other ions read from the same scans
    (``overlapping_ion``): the largest share of its picked intensity (all its bins) that one other
    measurement of its channel (same file, ``labels`` ``'<file> scans...'``, parent, condition,
    MS level and scan conditions: :func:`hxex.io.mzml._channel`) puts on its bins, and that ion
    (``'c-1 13-29 1+'``). The other ion's selected-K curve (read through the per-scan cut where
    it was fitted so) counts at this fragment's bins within the picker's window (``picker``:
    :meth:`~hxex.picking.PickerConfig.half_width`; ``mz``, the bins' m/z of every measurement) or
    within one peak width at half maximum, from the resolving power the file records (two peaks
    closer than that read as one, whichever the window takes),
    scaled by the other ion's own bins that are not near this fragment's strong bins (where its
    own fitted curve is at least 20 % of its tallest; all its bins when it has no fit) and hold at
    least 20 % of the other ion's curve: a fit of a coinciding ion is made of the same peaks. An
    ion with no such bins is not counted. The strong bins, not all of them: a fragment's bins
    cover its whole reachable range, so an ion anywhere inside it lies near one of them, and
    counted by all its bins would never count (including a fragment whose fit sits on that ion).
    NaN for measurements that are not fragments, and without a scan-file read record
    (``source``)."""
    from .io.mzml import SCAN_READERS, _channel, mzml_series_targets
    from .report.curves import component_curves, cut_curves

    m = len(fit)
    share = np.full(m, np.nan)
    other = np.full(m, "", dtype=object)
    if not source or source.get("reader") not in SCAN_READERS:
        return share, other
    targets = mzml_series_targets(source)
    files = [str(label).rsplit(" ", 1)[0] for label in labels]
    recorded = _recorded_resolution(source)
    channels: dict[tuple, list[int]] = {}
    for i in range(m):
        t = targets.get(int(fit.data_id[i]))
        if t is not None and "fragment" in dict(t.entry.extra) and t.ms_level >= 2:
            channels.setdefault((files[i], _channel(t)), []).append(i)

    def curve(i):
        k = int(fit.selected_K[i])
        kf = fit.fits.get(k)
        if kf is None or not np.isfinite(kf.nll[i]):
            return None
        c = component_curves(data.bins[i], kf.A[i], kf.n[i], kf.p[i], data.shapes[i])
        cut = None if data.scan_cut is None else data.scan_cut[i]
        return (cut_curves(c, cut) if cut is not None else c).sum(axis=0)

    for members in channels.values():
        curves = {i: curve(i) for i in members}
        bins_mz = {i: np.asarray(mz[i], dtype=float) for i in members}
        for f in members:
            e = targets[int(fit.data_id[f])].entry
            if str(dict(e.extra).get("fragment", "")).startswith("M:"):
                continue  # the precursor is only another ion
            obs = np.asarray(data.observed[f], dtype=float)
            share[f] = 0.0
            if obs.sum() <= 0:
                continue
            fmz = bins_mz[f]
            half = np.broadcast_to(np.asarray(picker.half_width(fmz, int(e.charge)), float),
                                   fmz.shape)
            r0, r_mz = recorded.get((files[f], int(targets[int(fit.data_id[f])].ms_level)),
                                    (np.nan, np.nan))
            if np.isfinite(r0) and np.isfinite(r_mz):  # the peak width at half maximum
                width = fmz / (r0 * (r_mz / fmz) ** picker.resolving_power_exponent)
                half = np.maximum(half, width)
            own = curves[f]
            strong = (own >= 0.2 * own.max() if own is not None and own.max() > 0
                      else np.ones(len(fmz), dtype=bool))
            lo, hi = float(np.min(fmz - half)), float(np.max(fmz + half))
            for o in members:
                pred = curves[o]
                if o == f or pred is None:
                    continue
                omz = bins_mz[o]
                if omz.max() < lo or omz.min() > hi:
                    continue
                near = np.abs(fmz[:, None] - omz[None, :]) <= half[:, None]
                if not near.any():
                    continue
                free = ~near[strong].any(axis=0) & (pred >= 0.2 * pred.max())
                if not free.any():
                    continue
                oobs = np.asarray(data.observed[o], dtype=float)
                scale = float(np.sum(oobs[free] * pred[free]) / np.sum(pred[free] ** 2))
                contrib = (near * (scale * pred)[None, :]).sum(axis=1)
                value = float(np.minimum(contrib, obs).sum() / obs.sum())
                if value > share[f]:
                    oe = targets[int(fit.data_id[o])].entry
                    ion = chem_label(oe)
                    share[f], other[f] = value, f"{ion} {oe.peptide_range} {oe.charge}+"
    return share, other


def _recorded_resolution(source) -> dict[tuple[str, int], tuple[float, float]]:
    """(file name, MS level) -> (resolving power, the m/z it is quoted at) from a read record's
    ``instruments`` (the lowest where the scans of a level differ); empty where none is
    recorded (mzML files)."""
    out: dict[tuple[str, int], tuple[float, float]] = {}
    instruments = source.get("instruments") or []
    names = [str(f.get("name", "")) for f in source.get("files") or []]
    for name, inst in zip(names, instruments, strict=False):
        if not inst or not np.isfinite(float(inst.get("resolving_power_mz") or np.nan)):
            continue
        for level, value in (inst.get("resolving_power") or {}).items():
            try:
                r0 = min(float(v) for v in str(value).split("/"))
            except ValueError:
                continue
            out[(name, int(level))] = (r0, float(inst["resolving_power_mz"]))
    return out


def chem_label(entry) -> str:
    """A fragment entry's ion type as the fragment list names it (``c-1``, ``z-dot``, ``M``)."""
    from .chem import parse_modification

    return parse_modification(entry.modification).ion_type

