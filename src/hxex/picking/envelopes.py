# SPDX-License-Identifier: GPL-3.0-or-later
"""The isotope envelope of an ion: theoretical, an ion-type mixture, or inferred from a reference.

Fragment spectra can hold several ion types of one fragment that differ by a hydrogen (for
example ``c-1``, ``c`` and ``c+1`` in ETD), whose envelopes overlap. The envelope used to place the
picker's bins and to fit the spectra is then not the natural isotope envelope of one ion type.
``EnvelopeConfig.mode``:

- ``'theory'``: the natural isotope clusters of the ion type (the default);
- ``'mixed'``: ``components`` ion types, the given one (the lightest) plus 0, 1, ... hydrogens,
  with weights fitted (non-negative least squares) to the picked reference spectra of the series
  (normally the undeuterated control); the mixture sets both the bin positions
  (:func:`hxex.chem.mixed_clusters`) and the envelope of the fit;
- ``'infer'``: the fit uses the picked reference envelope itself (averaged over the references),
  and the bins are placed as for ``'mixed'`` (or the theory with one component).

With ``share`` the conditions of one ion share its envelope: the series of the same sample, ion,
charge, activation and precursor that differ only in their scan filters or the fragment list's
label (gentle and harsh in-source activation, say) get one envelope, fitted to all their references
together (:func:`hxex.io.envelope_groups` gives the groups). The ion-type mixture is set by the
activation, not by the in-source energy: on the owner's comparison of undeuterated fragments at low
and high energy in ETD, EThcD and UVPD the envelopes agreed unless something else was wrong with
the data (docs/HISTORY.md item 87). So the conditions are first compared: two agree when the
envelope fitted to the references of each describes those of the other nearly as well as their own
does (:func:`reference_residual` at most ``max_residual`` above the residual of their own envelope,
both ways; how well the mixture fits at all is ``good``). The shared envelope is fitted to the
references of the conditions that agree with the most others; the rest (``agrees`` False) are left
out of it and their series marked ``poor``. When none agrees with another (two conditions that
disagree, say) there is no telling which is right: the envelope is fitted to all of them and all
are marked. A condition without references of its own takes the shared envelope.

This follows pyHXExpress's ``NA_envelope`` options (``mixedN`` and ``infer``), with two
differences: the components are shifted by the hydrogen mass rather than by one ExMS step, and the
mixture also sets the bin positions. Like pyHXExpress it assumes the proportions of the ion types
do not change with deuteration.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy.optimize import nnls

from .. import chem
from ..hooks import CancelHook, ProgressHook, check_cancel, report
from .config import DEFAULT_PICKER_CONFIG, PickerConfig
from .pick import pick_spectrum, plan_bins

MODES = ("theory", "mixed", "infer")


@dataclass(frozen=True)
class EnvelopeConfig:
    """How the envelope of each series is found (see the module docstring). ``components`` is the
    number of ion types in a mixture; ``max_residual`` the largest relative residual (RMS of the
    mixture's misfit over the RMS of the reference envelope) for which a reference counts as
    good; ``iterations`` the rounds of picking and fitting the weights; ``share`` gives the
    conditions of one ion one envelope (see the module docstring)."""

    mode: str = "theory"
    components: int = 2
    max_residual: float = 0.15
    iterations: int = 3
    share: bool = False

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.components < 1:
            raise ValueError("components must be at least 1")
        if not self.max_residual > 0:
            raise ValueError("max_residual must be positive")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EnvelopeConfig:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown envelope setting(s): {', '.join(sorted(unknown))}")
        return cls(**data)


@dataclass(frozen=True)
class Envelope:
    """The envelope found for one ion: ``mode``, the ion-type ``weights`` (lightest first, summing
    to 1; ``[1]`` for the theory), ``clusters`` (abundance and mass offset of each cluster, for
    the bin positions), ``fit_shape`` (the envelope of the fit, summing to 1), ``residual`` (of the
    mixture fit to the references; NaN for the theory), ``n_reference`` (reference spectra used)
    and ``good`` (``residual <= max_residual``, or the theory)."""

    mode: str
    weights: np.ndarray
    clusters: tuple[np.ndarray, np.ndarray]
    fit_shape: np.ndarray
    residual: float
    n_reference: int
    good: bool


def _normalised(y: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length)
    k = min(length, len(y))
    out[:k] = y[:k]
    total = out.sum()
    return out / total if total > 0 else out


def estimate_envelope(
    references: Sequence[tuple[np.ndarray, np.ndarray]],
    peptide: str,
    charge: int,
    *,
    modification: str = "",
    config: EnvelopeConfig = EnvelopeConfig(),  # noqa: B008
    picker: PickerConfig = DEFAULT_PICKER_CONFIG,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
) -> Envelope:
    """The envelope of one ion from its reference spectra (``(mz, intensity)`` pairs, normally the
    undeuterated controls of the series; ignored for ``'theory'``).

    For ``'mixed'`` and ``'infer'``: start from equal weights, then ``iterations`` times pick the
    references with the bins of the current mixture, average their envelopes (each scaled to sum
    to 1) and fit the weights by non-negative least squares to that average. Each set of weights
    (the starting ones, every fitted one) is scored by its misfit (at the best scale) to the
    references picked with its own bins, those a run would use, and the best is kept: the bins
    move with the weights, so the rounds need not settle (on a weak reference they can cycle,
    docs/HISTORY.md item 88). ``residual`` is that misfit, and with ``'infer'`` the average picked
    with the kept weights' bins is the envelope of the fit. Without a usable reference (none given,
    or nothing picked) the result is the theory with ``good`` False.
    """
    spec = chem.parse_modification(modification)
    base = chem.isotope_clusters(
        peptide, spec.ion_type, mod=spec, nterm_subtract=picker.nterm_subtract,
        rules=picker.amide_rules, registry=registry,
    )
    theory = Envelope("theory", np.array([1.0]), base, base[0], float("nan"), 0, True)
    if config.mode == "theory":
        return theory
    shape = base[0]
    k = config.components
    weights = np.full(k, 1.0 / k)
    length = len(shape) + k - 1
    columns = np.zeros((length, k))
    for i in range(k):
        columns[i:i + len(shape), i] = shape
    def average(clusters):
        observed, n = _picked_average(references, clusters, length, peptide, charge,
                                      modification=modification, picker=picker,
                                      registry=registry)
        if observed is None:
            raise _NoReference(n)
        return observed, n

    best = None  # (residual, weights, clusters, observed, n_used)
    try:
        for i in range(config.iterations + 1):
            clusters = chem.mixed_clusters(*base, weights)
            observed, n_used = average(clusters)
            residual = _residual(columns @ weights, observed)
            if best is None or residual < best[0]:
                best = (residual, weights, clusters, observed, n_used)
            if i == config.iterations:
                break
            fitted, _ = nnls(columns, observed)
            if fitted.sum() <= 0:
                raise _NoReference(n_used)
            weights = fitted / fitted.sum()
    except _NoReference as none:
        return Envelope(config.mode, theory.weights, base, base[0], float("nan"), none.n_used,
                        False)
    residual, weights, clusters, observed, n_used = best
    fit_shape = clusters[0] if config.mode == "mixed" else observed / observed.sum()
    return Envelope(config.mode, weights, clusters, fit_shape, residual, n_used,
                    residual <= config.max_residual)


class _NoReference(Exception):
    """No usable reference: nothing picked, or no positive weights (``n_used`` picked)."""

    def __init__(self, n_used: int):
        super().__init__(n_used)
        self.n_used = n_used


def _picked_average(references, clusters, length, peptide, charge, *, modification, picker,
                    registry) -> tuple[np.ndarray | None, int]:
    """The references picked with the bins of ``clusters``, each scaled to sum to 1, averaged
    over those with signal (None without any), and how many those are."""
    plan = plan_bins(peptide, charge, modification=modification, config=picker,
                     registry=registry, clusters=clusters)
    picked = [pick_spectrum(mz, y, peptide, charge, modification=modification, config=picker,
                            registry=registry, plan=plan).intensity for mz, y in references]
    picked = [p for p in picked if p.sum() > 0]
    if not picked:
        return None, 0
    return np.mean([_normalised(p, length) for p in picked], axis=0), len(picked)


def _residual(model: np.ndarray, observed: np.ndarray) -> float:
    """The RMS misfit of ``model`` at its best scale to ``observed``, over the RMS of
    ``observed``."""
    scale = float(model @ observed) / float(model @ model)
    return float(np.sqrt(np.mean((scale * model - observed) ** 2))
                 / np.sqrt(np.mean(observed ** 2)))


def reference_residual(
    references: Sequence[tuple[np.ndarray, np.ndarray]],
    envelope: Envelope,
    peptide: str,
    charge: int,
    *,
    modification: str = "",
    picker: PickerConfig = DEFAULT_PICKER_CONFIG,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
) -> float:
    """How well ``envelope`` describes ``references``: the references picked with its bins,
    averaged (each scaled to sum to 1), against its fit envelope at the best scale; the RMS of
    the misfit over the RMS of the average, as :attr:`Envelope.residual`. NaN without a usable
    reference."""
    model = np.asarray(envelope.fit_shape, dtype=float)
    observed, _ = _picked_average(references, envelope.clusters, len(model), peptide, charge,
                                  modification=modification, picker=picker, registry=registry)
    return float("nan") if observed is None else _residual(model, observed)


@dataclass(frozen=True)
class EnvelopeTable:
    """The envelope of every series of a run, one row per series (ascending ``data_id``):
    ``mode``, ``source`` (``reference``: the series' reference spectra, normally the undeuterated
    control; ``shared``: the references of every condition of the ion, with ``share``;
    ``fallback``: the spectra at the fallback time, because the reference was missing or poor;
    ``theory``: no usable spectra), the ion-type ``weights`` (``(n_series, components)``,
    lightest first), ``residual``, ``n_reference`` (the spectra the envelope was fitted to;
    with ``share``, the series' own references), ``good``, the
    envelope of the fit (``fit_shape``, flat, series ``i`` in
    ``shape_offsets[i]:shape_offsets[i + 1]``) and, with ``share``, ``own_residual``: how well
    the envelope describes the series' own references (:func:`reference_residual`; NaN without
    any, or without ``share``) and ``agrees``: False where the series' references disagree with
    the other conditions' (see the module docstring; True where not assessed)."""

    config: EnvelopeConfig
    data_id: np.ndarray
    mode: np.ndarray
    source: np.ndarray
    weights: np.ndarray
    residual: np.ndarray
    n_reference: np.ndarray
    good: np.ndarray
    fit_shape: np.ndarray
    shape_offsets: np.ndarray
    own_residual: np.ndarray | None = None
    agrees: np.ndarray | None = None

    def __post_init__(self):  # read from a bundle written before share
        if self.own_residual is None:
            object.__setattr__(self, "own_residual", np.full(len(self.data_id), np.nan))
        if self.agrees is None:
            object.__setattr__(self, "agrees", np.ones(len(self.data_id), dtype=bool))

    def __len__(self) -> int:
        return len(self.data_id)

    @property
    def poor(self) -> np.ndarray:
        """Series whose ion-type envelope was not measured well: with ``mixed`` or ``infer``,
        those left with the theory (no usable spectra), those whose envelope does not describe
        the spectra it was fitted to (``good`` False) and, with ``share``, those whose own
        references disagree with the other conditions' or are not described by the shared
        envelope."""
        measured = self.mode != "theory"
        own = np.nan_to_num(np.asarray(self.own_residual, dtype=float), nan=0.0)
        return measured & ((self.source == "theory") | ~np.asarray(self.good, dtype=bool)
                           | ~np.asarray(self.agrees, dtype=bool)
                           | (own > self.config.max_residual))

    def shape(self, i: int) -> np.ndarray:
        return self.fit_shape[self.shape_offsets[i]:self.shape_offsets[i + 1]]


def series_envelopes(
    spectra,
    *,
    config: EnvelopeConfig,
    reference_time: float,
    fallback_time: float | None = None,
    picker: PickerConfig = DEFAULT_PICKER_CONFIG,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
    groups: Mapping[int, Hashable] | None = None,
    atol: float = 1e-6,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> tuple[EnvelopeTable, dict[int, Envelope]]:
    """The envelope of every series (``data_id``) of a :class:`~hxex.spectra.SpectrumSet` from
    its spectra at ``reference_time`` (the undeuterated control). If those are missing or the
    mixture does not describe them (``good`` False), the spectra at ``fallback_time`` are used
    (for example a fully exchanged end-state run), and the source is recorded as ``fallback``;
    with neither, the theory. With ``config.share``, the series of one group (``groups``:
    ``data_id`` to a key, :func:`hxex.io.envelope_groups`; a series not in it is a group of its
    own) share one envelope (see the module docstring). ``progress(done, total)`` counts the
    series; ``cancel`` is polled before each group (:class:`hxex.hooks.Cancelled`). Returns the
    table and the envelopes by series."""
    by_series: dict[int, Envelope] = {}
    sources: dict[int, str] = {}
    own_count: dict[int, int] = {}
    own_residual: dict[int, float] = {}
    agrees: dict[int, bool] = {}
    ids_all = [int(d) for d in np.unique(spectra.data_id)]
    if config.share and groups:
        keyed: dict[Hashable, list[int]] = {}
        for d in ids_all:
            keyed.setdefault(groups.get(d, ("series", d)), []).append(d)
        members = list(keyed.values())
    else:
        members = [[d] for d in ids_all]
    n_done = 0
    for group in members:
        check_cancel(cancel)
        report(progress, n_done, len(ids_all))
        n_done += len(group)
        rows = {d: np.flatnonzero(spectra.data_id == d) for d in group}
        i0 = rows[group[0]][0]
        peptide, charge = str(spectra.peptide[i0]), int(spectra.charge[i0])
        modification = str(spectra.modification[i0])

        def at(t, d, rows=rows):
            sel = rows[d][np.abs(spectra.time[rows[d]] - t) <= atol]
            return [spectra.spectrum(int(i)) for i in sel]

        def estimate(refs, peptide=peptide, charge=charge, modification=modification):
            return estimate_envelope(refs, peptide, charge, modification=modification,
                                     config=config, picker=picker, registry=registry)

        def residual(refs, envelope, peptide=peptide, charge=charge,
                     modification=modification):
            return reference_residual(refs, envelope, peptide, charge,
                                      modification=modification, picker=picker,
                                      registry=registry)

        own = {d: at(reference_time, d) for d in group}
        chosen, source = None, "theory"
        refs = [r for d in group for r in own[d]]
        if refs:
            chosen = estimate(refs)
            source = "reference" if len(group) == 1 else "shared"
        res = {d: float("nan") for d in group}
        agree = {d: True for d in group}
        alone: dict[int, Envelope] = {}
        if chosen is not None and len(group) > 1 and config.mode != "theory":
            have = [d for d in group if own[d]]
            if len(have) > 1:  # which conditions agree with which (see the module docstring)
                alone = {d: estimate(own[d]) for d in have}
                self_fit = {d: residual(own[d], alone[d]) for d in have}
                fits = {(d, e): residual(own[d], alone[e]) - self_fit[d] <= config.max_residual
                        for d in have for e in have if d != e}
                count = {d: sum(fits[d, e] and fits[e, d] for e in have if e != d)
                         for d in have}
                best = max(count.values())
                agree.update({d: best > 0 and count[d] == best for d in have})
                core = [r for d in have if agree[d] for r in own[d]]
                if core and len(core) < len(refs):
                    chosen = estimate(core)
                    # conditions outside the core that it describes as well as their own
                    # envelope does are admitted (three or more conditions: two may agree
                    # with the core but not with each other), and the core is refitted once
                    back = [d for d in have if not agree[d]
                            and residual(own[d], chosen) - self_fit[d] <= config.max_residual]
                    if back:
                        agree.update(dict.fromkeys(back, True))
                        core += [r for d in back for r in own[d]]
                        chosen = estimate(core)
            res = {d: residual(own[d], chosen) if own[d] else float("nan") for d in group}
        if (chosen is None or not chosen.good) and fallback_time is not None:
            back = [r for d in group for r in at(fallback_time, d)]
            if back:
                other = estimate(back)
                if chosen is None or other.good or other.residual < chosen.residual:
                    chosen, source = other, "fallback"
        if chosen is None or chosen.n_reference == 0:
            base = estimate([])  # the theory (no reference spectra)
            chosen = Envelope(config.mode, np.array([1.0]), base.clusters, base.fit_shape,
                              float("nan"), 0, config.mode == "theory")
            source = "theory"
        for d in group:
            by_series[d] = chosen
            sources[d] = source
            if len(group) == 1:
                own_count[d] = chosen.n_reference
            else:  # the series' own references with signal, as for a series on its own
                own_count[d] = (alone[d].n_reference if d in alone
                                else estimate(own[d]).n_reference if own[d] else 0)
            # the pairwise comparison stays on record when the group then falls back
            own_residual[d] = res[d]
            agrees[d] = agree[d]
    report(progress, len(ids_all), len(ids_all))
    ids = sorted(by_series)
    width = max(config.components, 1)
    weights = np.full((len(ids), width), np.nan)
    shapes, offsets = [], [0]
    for r, d in enumerate(ids):
        e = by_series[d]
        weights[r, : len(e.weights)] = e.weights
        shapes.append(np.asarray(e.fit_shape, dtype=float))
        offsets.append(offsets[-1] + len(e.fit_shape))
    table = EnvelopeTable(
        config=config, data_id=np.array(ids, dtype=np.int64),
        mode=np.array([by_series[d].mode for d in ids], dtype=str),
        source=np.array([sources[d] for d in ids], dtype=str), weights=weights,
        residual=np.array([by_series[d].residual for d in ids], dtype=float),
        n_reference=np.array([own_count[d] for d in ids], dtype=np.int64),
        good=np.array([by_series[d].good for d in ids], dtype=bool),
        fit_shape=np.concatenate(shapes) if shapes else np.empty(0),
        shape_offsets=np.asarray(offsets, dtype=np.int64),
        own_residual=np.array([own_residual[d] for d in ids], dtype=float),
        agrees=np.array([agrees[d] for d in ids], dtype=bool),
    )
    return table, by_series
