# SPDX-License-Identifier: GPL-3.0-or-later
"""The engine's input: binned spectra with their shapes, bounds on n and noise references."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .. import chem
from ..noise import Reference, control_references
from ..picking import BinnedSet


@dataclass(frozen=True)
class Measurements:
    """M binned spectra ready to fit; list entries have one array per measurement.

    ``bins`` are the deuteration bins (``n_deut``), ``observed`` the intensities, ``shapes`` the
    natural-abundance envelope (normalised when fitted), ``n_max`` the upper bound of the number
    of exchanging sites ``n`` of every population, and ``references`` the reference spectra of
    each measurement's noise level (see :mod:`hxex.noise`). ``spectrum_id``, ``data_id`` and
    ``time`` identify the measurements and are carried into the results.
    """

    bins: tuple[np.ndarray, ...]
    observed: tuple[np.ndarray, ...]
    shapes: tuple[np.ndarray, ...]
    n_max: np.ndarray
    references: tuple[Reference, ...]
    spectrum_id: np.ndarray
    data_id: np.ndarray
    time: np.ndarray

    def __post_init__(self):
        m = len(self.bins)
        sizes = (len(self.observed), len(self.shapes), len(self.n_max), len(self.references),
                 len(self.spectrum_id), len(self.data_id), len(self.time))
        if any(s != m for s in sizes):
            raise ValueError("every field needs one entry per measurement")
        for i, (b, o) in enumerate(zip(self.bins, self.observed, strict=True)):
            if len(b) != len(o) or len(b) == 0:
                raise ValueError(
                    f"measurement {i}: bins and intensities must match and not be empty"
                )
        if np.any(np.asarray(self.n_max) <= 0):
            raise ValueError("n_max must be positive")

    def __len__(self) -> int:
        return len(self.bins)

    def select(self, index) -> Measurements:
        """The measurements at ``index`` (positions, in that order). The reference spectra of each
        selected measurement must be selected too (select whole series); they are renumbered."""
        index = np.asarray(index, dtype=np.int64)
        if index.ndim != 1 or len(set(index.tolist())) != len(index):
            raise ValueError("index must be a one-dimensional list of distinct positions")
        new = {int(old): j for j, old in enumerate(index)}
        refs = []
        for i in index:
            ref = self.references[int(i)]
            if ref is not None and any(r not in new for r in ref):
                raise ValueError("a measurement was selected without its reference spectra")
            refs.append(None if ref is None else tuple(new[r] for r in ref))
        return Measurements(
            bins=tuple(self.bins[i] for i in index),
            observed=tuple(self.observed[i] for i in index),
            shapes=tuple(self.shapes[i] for i in index), n_max=self.n_max[index],
            references=tuple(refs), spectrum_id=self.spectrum_id[index],
            data_id=self.data_id[index], time=self.time[index],
        )

    @classmethod
    def from_arrays(
        cls,
        bins: Sequence[np.ndarray],
        observed: Sequence[np.ndarray],
        shapes: Sequence[np.ndarray | None],
        n_max: Sequence[float],
        references: Sequence[Reference] | None = None,
        spectrum_id: Sequence[str] | None = None,
        data_id: Sequence[int] | None = None,
        time: Sequence[float] | None = None,
    ) -> Measurements:
        """Measurements from plain arrays (for tests and for data that did not come from the
        picker). Missing identifiers become ``m0, m1, ...``, series 0 and time 0."""
        m = len(bins)
        return cls(
            bins=tuple(np.asarray(b, dtype=np.float64) for b in bins),
            observed=tuple(np.asarray(o, dtype=np.float64) for o in observed),
            shapes=tuple(
                np.array([1.0]) if s is None else np.asarray(s, dtype=np.float64) for s in shapes
            ),
            n_max=np.asarray(n_max, dtype=np.float64),
            references=tuple(references) if references is not None else (None,) * m,
            spectrum_id=np.array(
                spectrum_id if spectrum_id is not None else [f"m{i}" for i in range(m)], dtype=str
            ),
            data_id=np.asarray(data_id if data_id is not None else [0] * m, dtype=np.int64),
            time=np.asarray(time if time is not None else [0.0] * m, dtype=np.float64),
        )

    @classmethod
    def from_binned(
        cls,
        binned: BinnedSet,
        *,
        n_max_factor: float = 1.2,
        undeut_time: float = 0.0,
        nterm_subtract: int = 1,
        amide_rules: chem.AmideRules = chem.DEFAULT_AMIDE_RULES,
        registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
    ) -> Measurements:
        """Measurements from a picked set. Drop flagged spectra first (``BinnedSet.select``).

        ``n_max = n_max_factor * exchangeable sites`` (amides plus user ``Hex``); the shape is
        the natural-abundance envelope of the peptide and ion type (``chem.natural_abundance``);
        the references are the spectra of the same series (``data_id``) at ``undeut_time``.
        ``nterm_subtract`` and ``amide_rules`` should be those used for picking.
        """
        if not n_max_factor > 0:
            raise ValueError("n_max_factor must be positive")
        shape_cache: dict[tuple[str, str], np.ndarray] = {}
        sites_cache: dict[tuple[str, str], int] = {}
        bins, observed, shapes, n_max = [], [], [], []
        for i in range(len(binned)):
            _, y, nd = binned.spectrum(i)
            peptide, modification = str(binned.peptide[i]), str(binned.modification[i])
            key = (peptide, modification)
            if key not in shape_cache:
                spec = chem.parse_modification(modification)
                shape_cache[key] = chem.natural_abundance(
                    peptide, spec.ion_type, mod=spec, nterm_subtract=nterm_subtract,
                    rules=amide_rules, registry=registry,
                )
                sites_cache[key] = chem.exchangeable_sites(
                    peptide, mod=spec, ion_type=spec.ion_type, nterm_subtract=nterm_subtract,
                    rules=amide_rules, registry=registry,
                )
            bins.append(nd.astype(np.float64))
            observed.append(y.astype(np.float64))
            shapes.append(shape_cache[key])
            n_max.append(n_max_factor * sites_cache[key])
        return cls(
            bins=tuple(bins), observed=tuple(observed), shapes=tuple(shapes),
            n_max=np.asarray(n_max, dtype=np.float64),
            references=tuple(control_references(binned.data_id, binned.time, undeut_time)),
            spectrum_id=binned.spectrum_id, data_id=binned.data_id, time=binned.time,
        )
