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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy.stats import chi2

MODELS = ("gaussian", "poisson")


@dataclass(frozen=True)
class NoiseConfig:
    """Noise model and level. The default is the setting used for the regression baseline:
    Gaussian, 3 % of the tallest peak of the reference spectrum (not a general recommendation).

    ``sd`` is a scalar or one value per measurement. Exactly one of ``sd`` and ``percent`` is
    needed for ``gaussian``; ``n_eff`` is only for ``poisson`` (alone or without ``percent``).

    ``estimate='replicates'`` estimates the noise of each series (one peptide at one charge, with
    its controls and time points) from its replicate spectra (:func:`estimate_noise`): variance
    equal to a gain times the intensity. The loss stays Gaussian: each spectrum of the series gets
    the sd ``sqrt(gain x its mean bin intensity)`` (the Poisson loss split two populations badly on
    real data, docs/HISTORY.md items 49, 55 to 57); it needs the gaussian model. A series needs at
    least ``min_replicated`` time points with two or more replicates; one without that falls back to
    the settings above, as if ``estimate`` were not set. ``min_share`` is the smallest share of a
    spectrum's total intensity a bin must have to be used, and a replicate whose score exceeds
    ``outlier_score`` is flagged and left out of the estimate.
    """

    model: str = "gaussian"
    sd: float | tuple[float, ...] | None = None
    percent: float | None = 3.0
    n_eff: float | None = None
    estimate: str | None = None
    min_replicated: int = 3
    min_share: float = 0.005
    outlier_score: float = 4.0

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
        if self.estimate not in (None, "replicates"):
            raise ValueError("estimate must be None or 'replicates'")
        if self.estimate and self.model != "gaussian":
            raise ValueError("estimate='replicates' sets a Gaussian noise level: use the "
                             "gaussian model")
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
    data_id: Sequence[int], time: Sequence[float], undeut_time: float, atol: float = 1e-6
) -> list[tuple[int, ...] | None]:
    """The reference spectra of every measurement: those of the same series (``data_id``) at the
    user-set undeuterated time. A series without one gives None (the measurement is its own
    reference). Replicate controls are all returned."""
    data_id = np.asarray(data_id)
    time = np.asarray(time, dtype=float)
    out: list[tuple[int, ...] | None] = []
    cache: dict[int, tuple[int, ...] | None] = {}
    for series in data_id:
        s = int(series)
        if s not in cache:
            idx = np.flatnonzero((data_id == series) & (np.abs(time - undeut_time) <= atol))
            cache[s] = tuple(int(i) for i in idx) or None
        out.append(cache[s])
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
    2 % on the private fragment dataset), and its scale would make them look almost noiseless
    (HISTORY item 77). None for the other spectra."""
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
) -> NoiseEstimate:
    """Estimate the noise of every series from its replicate spectra.

    A series is one ``data_id`` (a peptide at one charge); replicates are the spectra of a series
    at the same time (the controls included). For each time point with two or more replicates the
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
    time = np.round(np.asarray(time, dtype=float), 9)
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
