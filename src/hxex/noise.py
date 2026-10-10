# SPDX-License-Identifier: GPL-3.0-or-later
"""Noise level of the likelihood, set from a reference spectrum or given directly.

Each spectrum is divided by a scale ``c`` before it is fitted. ``c`` is the same for every
measurement that shares a reference spectrum (normally the undeuterated control of its series), so
spectra keep their relative intensities and a weaker spectrum is relatively noisier.

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

MODELS = ("gaussian", "poisson")


@dataclass(frozen=True)
class NoiseConfig:
    """Noise model and level. The default is the setting used for the regression baseline:
    Gaussian, 3 % of the tallest peak of the reference spectrum (not a general recommendation).

    ``sd`` is a scalar or one value per measurement. Exactly one of ``sd`` and ``percent`` is
    needed for ``gaussian``; ``n_eff`` is only for ``poisson`` (alone or without ``percent``).
    """

    model: str = "gaussian"
    sd: float | tuple[float, ...] | None = None
    percent: float | None = 3.0
    n_eff: float | None = None

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


def sample_size(scaled: Sequence[np.ndarray], config: NoiseConfig) -> np.ndarray:
    """The ``N`` of the BIC penalty for spectra already divided by ``c``: the number of bins
    (gaussian) or the total effective counts (poisson)."""
    if config.model == "gaussian":
        return np.array([len(o) for o in scaled], dtype=float)
    return np.array([float(np.sum(o)) for o in scaled])
