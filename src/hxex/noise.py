# SPDX-License-Identifier: GPL-3.0-or-later
"""Noise level of the likelihood, set from a reference spectrum or given directly.

Each spectrum is divided by a scale ``c`` before it is fitted. ``c`` is the same for every
measurement that shares a reference spectrum (normally the undeuterated control of its series), so
spectra keep their relative intensities and a weaker spectrum is relatively noisier. The spectra
of a kinetics run take theirs from the run's own scans instead (:func:`kinetics_references`).

``model='gaussian'``: baseline-type noise with a constant sd ``sigma`` that does not depend on
the signal. ``c = sigma``, given directly (``sd``) or as ``percent`` of the tallest peak of the
reference spectrum.

``model='poisson'``: after dividing by ``c`` the intensities are treated as counts, so the noise
sd grows as ``sqrt(intensity)``. ``c = sum(reference) / n_eff`` (the reference is worth ``n_eff``
effective counts), or ``c = (percent / 100)^2 * max(reference)`` (relative noise of ``percent`` at
the tallest peak); with neither, ``c = 1`` (raw intensities taken as counts, which is
overconfident for mass-spectrometry data and tends to over-select K).

A reference is given per measurement as the indices of its reference spectra (replicates are
averaged), or None for the measurement's own spectrum.

``estimate='controls'`` (opt-in) gives every bin its own noise variance instead of one sd per
spectrum: ``z b1 I + (m I)^2`` at a bin of intensity ``I`` in a spectrum of charge ``z``
(:class:`ControlsNoise`), at least (0.5 % of the spectrum's tallest bin)^2. ``b1`` (the counting
noise of the instrument, which grows with the ion's charge) is measured once per bundle from the
undeuterated controls
(:func:`controls_noise`); ``m`` (``model_error``, percent) is a fixed allowance for how far real
envelopes depart from the model's binomial mixtures, the same for every dataset. The scale above
then only fixes the units. Measurements: docs/HISTORY.md items 187 to 191.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy.stats import chi2

from .spectra import UNDEUT, time_point

MODELS = ("gaussian", "poisson")
ESTIMATES = (None, "controls", "replicates")


@dataclass(frozen=True)
class NoiseConfig:
    """Noise model and level of the fit (the ``[run.fit.noise]`` table): the loss scale, which
    also sets the model selection. The default is Gaussian at 3 % of the tallest peak of the
    series' reference spectrum, the setting of the regression baseline, not a recommendation.

    - ``model``: ``'gaussian'`` or ``'poisson'``.
    - ``sd``: the Gaussian sd in intensity units, one value or one per measurement; exactly one
      of ``sd`` and ``percent`` is given for ``gaussian``.
    - ``percent``: the sd (or the Poisson scale) as percent of the tallest peak of the series'
      undeuterated reference spectrum.
    - ``n_eff``: for ``poisson`` only, the effective counts of the tallest peak (alone, or
      without ``percent``).
    - ``estimate``: ``'controls'`` gives every bin its own noise variance, ``z b1 I +
      (m I)^2``, with ``b1`` measured from the run's undeuterated controls
      (:func:`controls_noise`) and ``m`` = ``model_error`` percent (needs the gaussian model;
      ``sd`` or ``percent`` then only fix the units). ``'replicates'`` estimates the noise of
      each series (one peptide at one charge, with its controls and time points) from its
      replicate spectra
      (:func:`estimate_noise`): a variance equal to a gain times the intensity. The loss stays
      Gaussian (needs the gaussian model): each spectrum of the series gets the sd ``sqrt(gain x
      its mean bin intensity)``. A series with fewer than ``min_replicated`` time points of two
      or more replicates falls back to the level above. ``min_share`` is the smallest share of a
      spectrum's total intensity a bin must have to be used, and a replicate whose score exceeds
      ``outlier_score`` is left out of the estimate and flagged ``noise_outlier``.
    - ``model_error``: with ``estimate='controls'``, the allowance for model error as percent of
      each bin's intensity (default 9: the value that calibrated the errors on three Q-TOF
      datasets, docs/HISTORY.md item 191). Lower: errors smaller and more spectra split into
      populations; higher: errors larger and fewer splits.
    """
    # why the loss stays Gaussian: docs/HISTORY.md items 49 and 55 to 57

    model: str = "gaussian"
    sd: float | tuple[float, ...] | None = None
    percent: float | None = 3.0
    n_eff: float | None = None
    estimate: str | None = None
    min_replicated: int = 3
    min_share: float = 0.005
    outlier_score: float = 4.0
    model_error: float = 9.0

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        if self.sd is not None and not np.isscalar(self.sd):
            object.__setattr__(self, "sd", tuple(float(x) for x in self.sd))
        if self.model == "gaussian":
            if self.n_eff is not None:
                raise ValueError("n_eff is not used with the gaussian model; use sd or percent")
            if (self.sd is None) == (self.percent is None):
                raise ValueError("the gaussian model needs exactly one of sd and percent")
            if self.sd is not None and np.any(np.asarray(self.sd, dtype=float) <= 0):
                raise ValueError("sd must be > 0")
        else:
            if self.sd is not None:
                raise ValueError("sd is only used with the gaussian model")
            if self.n_eff is not None and self.percent is not None:
                raise ValueError("specify n_eff or percent, not both")
            if self.n_eff is not None and self.n_eff <= 0:
                raise ValueError("n_eff must be > 0")
        if self.percent is not None and self.percent <= 0:
            raise ValueError("percent must be > 0")
        if self.estimate not in ESTIMATES:
            raise ValueError("estimate must be None, 'controls' or 'replicates'")
        if self.estimate and self.model != "gaussian":
            raise ValueError(f"estimate={self.estimate!r} sets a Gaussian noise level: use the "
                             "gaussian model")
        if not (np.isfinite(self.model_error) and self.model_error >= 0):
            raise ValueError("model_error must be 0 or more (percent)")
        if self.min_replicated < 1:
            raise ValueError("min_replicated must be at least 1")
        if not 0.0 < self.min_share < 1.0:
            raise ValueError("min_share must be between 0 and 1")
        if not self.outlier_score > 0:
            raise ValueError("outlier_score must be positive")

    @property
    def needs_reference(self) -> bool:
        """True if the scale is taken from reference spectra (not given as ``sd`` or 1)."""
        if self.model == "gaussian":
            return self.sd is None
        return self.n_eff is not None or self.percent is not None

    def to_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        if isinstance(out["sd"], tuple):
            out["sd"] = list(out["sd"])
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NoiseConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown noise setting(s): {', '.join(sorted(unknown))}")
        return cls(**data)


Reference = Sequence[int] | None


def control_references(
    data_id: Sequence[int], role: Sequence[str]
) -> list[tuple[int, ...] | None]:
    """The reference spectra of every measurement: the undeuterated controls (``role``
    ``'undeut'``, :data:`hxex.spectra.UNDEUT`) of the same series (``data_id``). A series
    without one gives None (the measurement is its own reference). Replicate controls are all
    returned."""
    data_id = np.asarray(data_id)
    undeut = np.asarray(role, dtype=str) == UNDEUT
    if len(undeut) != len(data_id):
        raise ValueError("data_id and role must have one entry per measurement")
    out: list[tuple[int, ...] | None] = []
    cache: dict[int, tuple[int, ...] | None] = {}
    for series in data_id:
        s = int(series)
        if s not in cache:
            idx = np.flatnonzero((data_id == series) & undeut)
            cache[s] = tuple(int(i) for i in idx) or None
        out.append(cache[s])
    return out


def noise_reference_kinds(
    data_id: Sequence[int], references: Sequence[Reference], role: Sequence[str]
) -> dict[int, str]:
    """What each series' noise level comes from: ``'undeuterated control'`` (its own controls),
    ``"run's scans"`` (a kinetics run's own scans, :func:`kinetics_references`) or
    ``'own spectrum'`` when one of its measurements has no reference (no undeuterated control of
    the series was fitted, so the spectrum sets its own level from its tallest peak). A series
    with several kinds takes the first of own spectrum, run's scans, undeuterated control."""
    undeut = np.asarray(role, dtype=str) == UNDEUT
    order = ("own spectrum", "run's scans", "undeuterated control")
    out: dict[int, str] = {}
    for series, ref in zip(np.asarray(data_id), references, strict=True):
        kind = ("own spectrum" if ref is None
                else "undeuterated control" if all(undeut[j] for j in ref) else "run's scans")
        s = int(series)
        if s not in out or order.index(kind) < order.index(out[s]):
            out[s] = kind
    return out


def kinetics_references(
    data_id: Sequence[int],
    rep: Sequence[int],
    observed: Sequence[np.ndarray],
    kinetics: Sequence[bool],
) -> list[tuple[int, ...] | None]:
    """The reference of every spectrum of a kinetics run (``kinetics``: each scan its own time,
    from one infusion file): the one spectrum of the same series (``data_id``) and replicate
    (``rep``) among the kinetics spectra whose tallest peak is the median of theirs (the lower
    one for an even count), so the series shares one noise level at the kinetics file's
    intensity. A separate undeuterated control can be far weaker than the kinetics scans (about
    2 % on the private fragment dataset), and its scale would make them look almost noiseless.
    None for the other spectra."""
    data_id, rep = np.asarray(data_id), np.asarray(rep)
    kinetics = np.asarray(kinetics, dtype=bool)
    if not (len(data_id) == len(rep) == len(observed) == len(kinetics)):
        raise ValueError("data_id, rep, observed and kinetics must have one entry per spectrum")
    out: list[tuple[int, ...] | None] = [None] * len(data_id)
    for series, r in sorted({(int(d), int(r)) for d, r in
                             zip(data_id[kinetics], rep[kinetics], strict=True)}):
        idx = np.flatnonzero(kinetics & (data_id == series) & (rep == r))
        tallest = np.array([np.max(observed[i]) if len(observed[i]) else 0.0 for i in idx])
        ref = (int(idx[np.argsort(tallest, kind="stable")[(len(idx) - 1) // 2]]),)
        for i in idx:
            out[int(i)] = ref
    return out


def noise_scale(
    observed: Sequence[np.ndarray],
    config: NoiseConfig = NoiseConfig(),  # noqa: B008
    references: Sequence[Reference] | None = None,
) -> np.ndarray:
    """The scale ``c`` of every measurement (see the module docstring), shape ``(M,)``.

    ``references`` has one entry per measurement (indices into ``observed``, or None for the
    measurement itself); it is required whenever the scale comes from a reference spectrum.
    """
    m = len(observed)
    if config.model == "gaussian" and config.sd is not None:
        sd = np.broadcast_to(np.asarray(config.sd, dtype=float), (m,)).copy()
        return sd
    if not config.needs_reference:
        return np.ones(m)
    if references is None:
        raise ValueError(
            "references are required: the noise level is set from a reference spectrum "
            "(for example the undeuterated control of each series); use None entries for "
            "'the measurement itself'"
        )
    if len(references) != m:
        raise ValueError("references must have one entry per measurement")
    refs = [(i,) if r is None else tuple(int(j) for j in r) for i, r in enumerate(references)]

    def stat(entry: tuple[int, ...], fn) -> float:
        if not entry:
            raise ValueError("empty reference entry")
        return float(np.mean([fn(np.asarray(observed[j], dtype=float)) for j in entry]))

    if config.model == "poisson" and config.n_eff is not None:
        return np.array([stat(r, np.sum) / config.n_eff for r in refs])
    pct = config.percent / 100.0  # type: ignore[operator]
    if config.model == "gaussian":
        return np.array([pct * stat(r, np.max) for r in refs])
    return np.array([pct**2 * stat(r, np.max) for r in refs])


def sample_size(
    scaled: Sequence[np.ndarray], kinds: NoiseConfig | str | Sequence[str]
) -> np.ndarray:
    """The ``N`` of the BIC penalty for spectra already divided by ``c``: the number of bins
    (gaussian) or the total effective counts (poisson). ``kinds`` is the noise model of each
    measurement, one name for all, or a :class:`NoiseConfig`."""
    if isinstance(kinds, NoiseConfig):
        kinds = kinds.model
    names = [kinds] * len(scaled) if isinstance(kinds, str) else list(kinds)
    return np.array(
        [
            float(len(o)) if k == "gaussian" else float(np.sum(o))
            for o, k in zip(scaled, names, strict=True)
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class NoiseEstimate:
    """The noise of each measurement estimated from replicates (see :func:`estimate_noise`).

    All arrays have one entry per measurement. ``gain`` is the variance per unit intensity of the
    measurement's series, NaN where ``estimated`` is False; ``reason`` says why not. ``score`` is
    the consistency of a spectrum with its replicates (about 1 if it behaves as the series' noise
    says; NaN where the time point has fewer than 3 replicates) and ``outlier`` flags scores above
    the configured limit. ``n_replicated`` counts the replicated time points of the series.
    """

    gain: np.ndarray
    estimated: np.ndarray
    reason: np.ndarray
    score: np.ndarray
    outlier: np.ndarray
    n_replicated: np.ndarray


def _cell_matrix(bins, observed, members):
    """Intensities of replicate spectra on the union of their bins (absent bins are 0), each
    rescaled to the mean total intensity of the time point."""
    union = np.unique(np.concatenate([np.asarray(bins[i]) for i in members]))
    x = np.zeros((len(members), len(union)))
    for r, i in enumerate(members):
        x[r, np.searchsorted(union, np.asarray(bins[i]))] = observed[i]
    totals = x.sum(axis=1)
    good = totals > 0
    mean_total = totals[good].mean() if good.any() else 0.0
    return x * (mean_total / np.where(good, totals, 1.0))[:, None], mean_total, good


def _cell_gain(x: np.ndarray, mean_total: float, min_share: float) -> float:
    """The gain of one time point: the median over bins of (replicate variance / mean), divided by
    the median of the chi-square distribution of that ratio, so that a bad bin cannot move it."""
    n = x.shape[0]
    mean = x.mean(axis=0)
    use = mean >= min_share * mean_total
    if n < 2 or use.sum() < 3:
        return float("nan")
    ratio = x.var(axis=0, ddof=1)[use] / mean[use]
    return float(np.median(ratio) / (chi2.ppf(0.5, n - 1) / (n - 1)))


def _score_cell(x, good, total, g, min_share, limit):
    """Scores of the replicates of one time point and the set of outliers (indices into ``x``).

    The score of a replicate is the mean over bins of its squared difference from the mean of the
    other active replicates divided by the variance the gain predicts for that difference,
    ``g * mean * (1 + 1 / k)`` for ``k`` others. While three or more replicates are active, the
    one with the highest score is removed if that score exceeds ``limit`` and the rest are
    scored again, so one bad spectrum does not make its good replicates look bad. Removed
    replicates are scored against the mean of the active ones. NaN where a score cannot be
    formed (a pair, or too few usable bins).
    """
    n = len(x)
    scores = np.full(n, np.nan)
    active = [r for r in range(n) if good[r]]

    def score_of(r, pool):
        others = x[[j for j in pool if j != r]]
        if len(others) < 1:
            return np.nan
        mean = others.mean(axis=0)
        use = mean >= min_share * total
        if use.sum() < 3:
            return np.nan
        k = len(others)
        return float(np.mean((x[r] - mean)[use] ** 2 / (g * mean[use] * (1.0 + 1.0 / k))))

    removed: set[int] = set()
    while len(active) >= 3:
        current = {r: score_of(r, active) for r in active}
        valid = {r: v for r, v in current.items() if np.isfinite(v)}
        if not valid:
            break
        worst = max(valid, key=valid.get)
        if valid[worst] <= limit:
            break
        active.remove(worst)
        removed.add(worst)
    if n >= 3:  # a pair cannot say which of the two is off
        for r in range(n):
            if good[r]:
                scores[r] = score_of(r, active + ([r] if r in removed else []))
    return scores, removed


def estimate_noise(
    bins: Sequence[np.ndarray],
    observed: Sequence[np.ndarray],
    data_id: Sequence[int],
    time: Sequence[float],
    config: NoiseConfig,
    role: Sequence[str] | None = None,
) -> NoiseEstimate:
    """Estimate the noise of every series from its replicate spectra.

    A series is one ``data_id`` (a peptide at one charge); replicates are the spectra of a series
    at the same point: the same time, or the same control ``role`` (:func:`hxex.spectra.time_point`;
    without roles the controls are known by their times). For each point with two or more
    replicates the
    replicates are rescaled to a common total (the amplitude of a spectrum is a free parameter of
    the fit, so only the shape and the noise matter), and the variance of every bin of enough
    intensity is compared with its mean: the noise is taken as variance = gain x intensity, which
    within a series was found to hold with an exponent of about 1.1. The gain of the series is the
    median over its time points of the median over bins of variance / mean (each corrected for
    the chi-square distribution), a robust estimate that a bad spectrum cannot inflate.

    Each spectrum of a time point with three or more replicates also gets a score: the mean over
    bins of its squared difference from the mean of the others, divided by the variance the gain
    predicts for that difference (``gain x mean x (1 + 1 / (n - 1))``). It is about 1 for a
    consistent spectrum; above ``config.outlier_score`` the spectrum is flagged and left out of a
    second estimate of the gain. The flag is a warning (the spectrum may also differ for a real
    reason, such as its deuteration); nothing is dropped here.

    A series needs ``config.min_replicated`` time points with replicates, else it is not
    estimated (``reason`` says why) and the caller uses the fallback noise settings.
    """
    m = len(observed)
    data_id = np.asarray(data_id)
    time = time_point(time, role)
    gain = np.full(m, np.nan)
    estimated = np.zeros(m, dtype=bool)
    reason = np.full(m, "", dtype=object)
    score = np.full(m, np.nan)
    outlier = np.zeros(m, dtype=bool)
    n_rep = np.zeros(m, dtype=np.int64)
    min_share = config.min_share

    for series in np.unique(data_id):
        rows = np.flatnonzero(data_id == series)
        cells = [rows[time[rows] == t] for t in np.unique(time[rows])]
        replicated = [c for c in cells if len(c) >= 2]
        n_rep[rows] = len(replicated)
        if len(replicated) < config.min_replicated:
            reason[rows] = (
                "no replicates" if not replicated else
                f"{len(replicated)} replicated time points (need {config.min_replicated})"
            )
            continue
        matrices = [_cell_matrix(bins, observed, c) for c in replicated]
        g, removed = float("nan"), [set() for _ in replicated]
        for _ in range(2):  # estimate the gain, find the outliers, estimate again without them
            cell_gains = []
            for (x, total, good), drop in zip(matrices, removed, strict=True):
                keep = [r for r in range(len(x)) if good[r] and r not in drop]
                if len(keep) >= 2:
                    cell_gains.append(_cell_gain(x[keep], total, min_share))
            cell_gains = [v for v in cell_gains if np.isfinite(v) and v > 0]
            if not cell_gains:
                g = float("nan")
                break
            g = float(np.median(cell_gains))
            new_removed = []
            for c, (x, total, good) in zip(replicated, matrices, strict=True):
                scores, drop = _score_cell(x, good, total, g, min_share, config.outlier_score)
                for r, i in enumerate(c):
                    score[i] = scores[r]
                new_removed.append(drop)
            if new_removed == removed:
                break
            removed = new_removed
        if not np.isfinite(g):
            reason[rows] = "too few usable bins"
            score[rows] = np.nan
            continue
        gain[rows] = g
        estimated[rows] = True
        for c, drop in zip(replicated, removed, strict=True):
            for r in drop:
                outlier[c[r]] = True
    return NoiseEstimate(gain, estimated, reason.astype(str), score, outlier, n_rep)


# --- per-bin noise from the undeuterated controls (estimate='controls') -------------------------

_ONSET_BANDS = (0.0, 3e4, 1e5, 2e5, 4e5, 6e5, 8e5, 1.2e6, 2e6, 4e6, np.inf)
_FLOOR_SHARE = 0.005  # every bin's variance is at least (this share of its spectrum's top)^2


@dataclass(frozen=True)
class SampleNoise:
    """The counting scale ``b1`` measured from one sample's undeuterated controls alone, its
    standard error (from resampling the sample's series) and the series it rests on; NaN
    (not measured) with fewer than 5 series or when its controls show no counting term."""

    sample: str
    b1: float
    b1_se: float
    n_series: int


@dataclass(frozen=True)
class ControlsNoise:
    """A bundle's per-bin noise (``estimate='controls'``): variance ``z b1 I + (m I)^2`` at a
    bin of intensity ``I`` (raw units) in a spectrum of charge ``z``, with ``m`` =
    ``model_error`` (a share, 0.09 for 9 %), at least (0.5 % of the spectrum's tallest bin)^2
    (:func:`bin_variances`). ``b1`` comes from ``n_controls`` undeuterated controls of
    ``n_series`` series below ``onset`` (the tallest peak above which the run's controls depart
    from their natural-abundance shape; ``inf``: none found); ``b1_se`` is its standard error
    and ``by_sample`` the same measured per sample. No constant floor is fitted: the controls
    put it at about 0 (docs/HISTORY.md item 187), and with few controls it trades off against
    ``b1``."""

    b1: float
    model_error: float
    onset: float
    n_controls: int
    n_series: int
    b1_se: float = float("nan")
    by_sample: tuple[SampleNoise, ...] = ()

    def variance(self, intensity: np.ndarray, charge: float = 1.0) -> np.ndarray:
        i = np.clip(np.asarray(intensity, dtype=np.float64), 0.0, None)
        return charge * self.b1 * i + (self.model_error * i) ** 2

    def disagreeing_samples(self, z: float = 3.0) -> list[SampleNoise]:
        """The samples whose own ``b1`` differs from the bundle's by more than ``z`` combined
        standard errors: a sign that the bundle mixes instruments or settings."""
        out = []
        for s in self.by_sample:
            se = np.hypot(s.b1_se, self.b1_se if np.isfinite(self.b1_se) else 0.0)
            if np.isfinite(s.b1) and np.isfinite(se) and se > 0 and abs(s.b1 - self.b1) > z * se:
                out.append(s)
        return out

    def to_dict(self) -> dict[str, Any]:
        def num(x: float) -> float | None:
            return float(x) if np.isfinite(x) else None

        return {"b1": self.b1, "model_error": self.model_error, "onset": num(self.onset),
                "n_controls": self.n_controls, "n_series": self.n_series,
                "b1_se": num(self.b1_se),
                "by_sample": [{"sample": s.sample, "b1": num(s.b1), "b1_se": num(s.b1_se),
                               "n_series": s.n_series} for s in self.by_sample]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ControlsNoise:
        def num(x) -> float:
            return float("nan") if x is None else float(x)

        return cls(b1=float(data["b1"]), model_error=float(data["model_error"]),
                   onset=float("inf") if data.get("onset") is None else float(data["onset"]),
                   n_controls=int(data["n_controls"]), n_series=int(data["n_series"]),
                   b1_se=num(data.get("b1_se")),
                   by_sample=tuple(SampleNoise(str(s["sample"]), num(s["b1"]), num(s["b1_se"]),
                                               int(s["n_series"]))
                                   for s in data.get("by_sample", ())))


def _natural_fit(bins, observed, shape) -> tuple[np.ndarray, np.ndarray] | None:
    """An undeuterated control's intensities and its natural-abundance shape at its bins (zero
    beyond the shape), or None without signal."""
    k = np.asarray(bins).astype(np.int64)
    y = np.asarray(observed, dtype=np.float64)
    sh = np.asarray(shape, dtype=np.float64)
    ok = (k >= 0) & (k < len(sh))
    if not ok.any() or y.max(initial=0.0) <= 0 or sh.max(initial=0.0) <= 0:
        return None
    p = np.zeros(len(y))
    p[ok] = sh[k[ok]]
    if p.max() <= 0:
        return None
    return y, p


def distortion_onset(bins, observed, shapes, role) -> float:
    """The tallest peak above which a run's undeuterated controls stop matching their
    natural-abundance shape: the lower edge of the first band of tops (30,000, 100,000, ...
    4,000,000) whose controls' median flattening index (the median observed/predicted ratio of
    the peaks predicted at 3-30 % of the top on the heavy side, over that ratio at the tallest
    peak, one least-squares height) is outside 0.9-1.1, with at least 10 controls in the band,
    above a dimmer band whose controls match (within 0.9-1.1, at least 10 of them); ``inf``
    when no band departs. Detector compression reads above 1, a heavy-side deficit below
    (docs/HISTORY.md items 181 and 187). Dim controls can read above 1 too, from picked noise
    in their small heavy-side peaks: that is no onset, so a departure counts only once the
    controls have matched (item 191)."""
    tops, flat = [], []
    for i, r in enumerate(role):
        if r != UNDEUT:
            continue
        got = _natural_fit(bins[i], observed[i], shapes[i])
        if got is None:
            continue
        y, p = got
        p = p / p.max()
        t = int(np.argmax(p))
        u = p >= 0.03
        s = float(np.sum(y[u] * p[u]) / np.sum(p[u] ** 2))
        heavy = (np.arange(len(p)) > t) & (p >= 0.03) & (p < 0.3)
        if s <= 0 or y[t] <= 0 or not heavy.any():
            continue
        tops.append(float(y.max()))
        flat.append(float(np.median(y[heavy] / (s * p[heavy])) / (y[t] / (s * p[t]))))
    tops_a, flat_a = np.asarray(tops), np.asarray(flat)
    matched = False
    for lo, hi in zip(_ONSET_BANDS[:-1], _ONSET_BANDS[1:], strict=True):
        band = (tops_a >= lo) & (tops_a < hi)
        if band.sum() < 10:
            continue
        if abs(float(np.median(flat_a[band])) - 1.0) <= 0.1:
            matched = True
        elif matched:
            return float(lo)
    return float("inf")


def _residual_design(shape: np.ndarray, intensity: np.ndarray, charge: float) -> np.ndarray:
    """Each residual's expected variance per unit of ``b1`` and of an ``I^2`` term when one
    least-squares height of ``shape`` is fitted: the residuals are ``(1 - H) y`` with ``H`` the
    projection on the shape, so residual ``i`` has variance ``sum_j (delta_ij - H_ij)^2 v_j``.
    The tallest peak carries most of the height's fit (its residual is far smaller than its
    noise), which a flat ``n / (n - 1)`` correction would not undo. ``(n, 2)``."""
    p = np.asarray(shape, dtype=np.float64)
    h = np.outer(p, p) / float(p @ p)
    m = (np.eye(len(p)) - h) ** 2
    return m @ np.column_stack([charge * intensity, intensity ** 2])


def _curve_fit(intensity, resid, charge, design) -> float | None:
    """``b1`` from residuals: grouped by intensity within each charge (up to 14 log-quantile
    groups of at least 8 bins per charge), each group's robust variance fitted by non-negative
    least squares on the group's mean ``design`` row (:func:`_residual_design`) in relative
    terms (each group to within a share of itself, so the brightest groups do not decide it).
    The bins of a group differ in variance, which makes a robust spread understate their mean:
    the residuals are standardised by their expected variance under the current fit and the
    fit repeated. The ``I^2`` term takes the controls' own departure from their shape and is
    dropped. None with fewer than three groups."""
    from scipy.optimize import nnls
    i = np.asarray(intensity, dtype=np.float64)
    keep = i > 0
    i = i[keep]
    d = np.asarray(resid, dtype=np.float64)[keep]
    z = np.asarray(charge, dtype=np.float64)[keep]
    x = np.asarray(design, dtype=np.float64)[keep]
    groups = []
    for zz in np.unique(z):
        at = np.flatnonzero(z == zz)
        if len(at) < 24:
            continue
        log_i = np.log(i[at])
        n_groups = max(3, min(14, len(at) // 20))
        edges = np.unique(np.quantile(log_i, np.linspace(0, 1, n_groups + 1)))
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            g = at[(log_i >= lo) & (log_i <= hi)]
            if len(g) >= 8:
                groups.append(g)
    if len(groups) < 3:
        return None
    a = np.array([x[g].mean(axis=0) for g in groups])
    w = np.array([np.sqrt(len(g)) for g in groups])
    norm = np.maximum(np.median(a, axis=0), 1e-300)
    coef = None
    for _ in range(6):
        if coef is None:
            y = np.array([(1.4826 * np.median(np.abs(d[g]))) ** 2 for g in groups])
            scale = y
        else:
            v = np.maximum(x @ coef, 1e-300)
            y = np.array([(1.4826 * np.median(np.abs(d[g] / np.sqrt(v[g])))) ** 2
                          * float(np.mean(v[g])) for g in groups])
            scale = a @ coef  # each group's variance as fitted, not as observed: dividing by
            # the observed one would favour the groups that happen to read low
        scale = np.maximum(scale, 1e-300)
        sol, _ = nnls(a / norm / scale[:, None] * w[:, None], y / scale * w)
        coef = sol / norm
    return float(coef[0])


def controls_noise(
    bins: Sequence[np.ndarray],
    observed: Sequence[np.ndarray],
    shapes: Sequence[np.ndarray],
    role: Sequence[str],
    charge: Sequence[float],
    data_id: Sequence[int],
    *,
    model_error: float,
    usable: Sequence[bool] | None = None,
    sample: Sequence[str] | None = None,
    n_resample: int = 20,
    seed: int = 0,
) -> ControlsNoise | None:
    """The bundle's per-bin noise from its undeuterated controls (``role`` ``'undeut'``).

    Each usable control (``usable``: for example the ones every picker screen passes; default
    all) whose tallest peak is below the run's :func:`distortion_onset` is fitted with its
    natural-abundance shape (one least-squares height over the bins predicted at 5 % of the top
    or more); its residuals there, with their expected variance under that fit
    (:func:`_residual_design`), give ``b1`` (:func:`_curve_fit`). ``model_error`` is a share
    (0.09 for 9 %). The standard error of ``b1`` comes from ``n_resample`` resamplings of the
    series (``data_id``), and with ``sample`` the same is measured for each sample alone. None
    when fewer than 5 series give residuals or no positive ``b1`` is found. Replicate pairs
    of controls give the same counting scale on the Q-TOF data (docs/HISTORY.md items 187 and
    190)."""
    m = len(bins)
    role = np.asarray(role, dtype=str)
    charge = np.asarray(charge, dtype=np.float64)
    data_id = np.asarray(data_id)
    use = np.ones(m, dtype=bool) if usable is None else np.asarray(usable, dtype=bool)
    onset = distortion_onset(bins, observed, shapes, role)
    parts: dict[int, list[tuple[np.ndarray, ...]]] = {}
    n_controls = 0
    for i in np.flatnonzero((role == UNDEUT) & use):
        got = _natural_fit(bins[i], observed[i], shapes[i])
        if got is None:
            continue
        y, p = got
        if y.max() >= onset:
            continue
        sel = p >= 0.05 * p.max()
        if int(sel.sum()) < 3:
            continue
        h = float(np.sum(y[sel] * p[sel]) / np.sum(p[sel] ** 2))
        pred = h * p[sel]
        parts.setdefault(int(data_id[i]), []).append(
            (pred, y[sel] - pred, np.full(int(sel.sum()), charge[i]),
             _residual_design(p[sel], pred, float(charge[i]))))
        n_controls += 1

    def fit(keys) -> float | None:
        blocks = [b for k in keys for b in parts[k]]
        if not blocks:
            return None
        return _curve_fit(*(np.concatenate([b[j] for b in blocks]) for j in range(4)))

    def with_se(keys) -> tuple[float | None, float]:
        keys = sorted(keys)
        got = fit(keys) if len(keys) >= 5 else None
        if got is None:
            return None, float("nan")
        rng = np.random.default_rng(seed)
        draws = [fit(list(rng.choice(keys, len(keys)))) for _ in range(n_resample)]
        b = [d for d in draws if d is not None]
        return got, (float(np.std(b, ddof=1)) if len(b) >= 2 else float("nan"))

    b1, se = with_se(parts)
    if b1 is None or not b1 > 0:
        return None
    by_sample: list[SampleNoise] = []
    if sample is not None:
        names = np.asarray(sample, dtype=str)
        series_sample = {int(d): str(s) for d, s in zip(data_id, names, strict=True)}
        for name in sorted({series_sample[k] for k in parts}):
            keys = [k for k in parts if series_sample[k] == name]
            s_b1, s_se = with_se(keys)
            if s_b1 is None or not s_b1 > 0:  # too few controls, or no counting term found
                s_b1, s_se = float("nan"), float("nan")
            by_sample.append(SampleNoise(name, s_b1, s_se, len(keys)))
    return ControlsNoise(b1=b1, model_error=model_error, onset=onset, n_controls=n_controls,
                         n_series=len(parts), b1_se=se, by_sample=tuple(by_sample))


def bin_variances(
    observed: Sequence[np.ndarray], charge: Sequence[float], curve: ControlsNoise
) -> list[np.ndarray]:
    """Each measurement's noise variance per bin (raw units) from ``curve``, at its observed
    intensity smoothed over three bins (so a bin that reads high by chance does not raise its own
    variance alone), floored at (0.5 % of the spectrum's tallest bin)^2 so empty bins keep a
    finite weight."""
    out = []
    for o, z in zip(observed, np.asarray(charge, dtype=np.float64), strict=True):
        y = np.clip(np.asarray(o, dtype=np.float64), 0.0, None)
        at = np.convolve(y, np.ones(3) / 3.0, mode="same") if len(y) >= 3 else y
        top = max(float(y.max(initial=0.0)), 1e-12)
        out.append(np.maximum(curve.variance(at, float(z)), (_FLOOR_SHARE * top) ** 2))
    return out
