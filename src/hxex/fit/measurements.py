# SPDX-License-Identifier: GPL-3.0-or-later
"""The engine's input: binned spectra with their shapes, bounds on n and noise references."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .. import chem
from ..noise import Reference, control_references, kinetics_references
from ..picking import BinnedSet


@dataclass(frozen=True)
class Measurements:
    """M binned spectra ready to fit; list entries have one array per measurement.

    ``bins`` are the deuteration bins (``n_deut``), ``observed`` the intensities, ``shapes`` the
    natural-abundance envelope (normalised when fitted), ``n_max`` the upper bound of the number
    of exchanging sites ``n`` of every population, and ``references`` the reference spectra of
    each measurement's noise level (see :mod:`hxex.noise`). ``spectrum_id``, ``data_id``,
    ``time`` and ``role`` (``'undeut'``, ``'fulldeut'`` or ``''``: the controls of a series are
    the measurements with a role; all ``''`` when not given) identify the measurements and are
    carried into the results. ``random_id`` (default the spectrum id) is what each measurement's
    random numbers are keyed by (:meth:`random_keys`): from a picked set, what the spectrum is
    (:func:`measurement_identity`), so its random starts, bootstrap draws and null simulations do
    not depend on the project's name, the file layout or the other spectra.
    """

    bins: tuple[np.ndarray, ...]
    observed: tuple[np.ndarray, ...]
    shapes: tuple[np.ndarray, ...]
    n_max: np.ndarray
    references: tuple[Reference, ...]
    spectrum_id: np.ndarray
    data_id: np.ndarray
    time: np.ndarray
    role: np.ndarray | None = None
    random_id: np.ndarray | None = None

    def __post_init__(self):
        m = len(self.bins)
        if self.role is None:  # built without roles: time points only
            object.__setattr__(self, "role", np.full(m, "", dtype="<U8"))
        else:
            object.__setattr__(self, "role", np.asarray(self.role, dtype=str))
        object.__setattr__(self, "random_id", np.asarray(
            self.spectrum_id if self.random_id is None else self.random_id, dtype=str))
        sizes = (len(self.observed), len(self.shapes), len(self.n_max), len(self.references),
                 len(self.spectrum_id), len(self.data_id), len(self.time), len(self.role),
                 len(self.random_id))
        if any(s != m for s in sizes):
            raise ValueError("every field needs one entry per measurement")
        for i, (b, o) in enumerate(zip(self.bins, self.observed, strict=True)):
            if len(b) != len(o) or len(b) == 0:
                raise ValueError(
                    f"measurement {i}: bins and intensities must match and not be empty"
                )
            if not np.array_equal(b, np.round(b)) or np.any(b < 0):
                raise ValueError(f"measurement {i}: bins must be whole numbers of 0 or more "
                                 "(deuterons beyond natural abundance)")
        if np.any(np.asarray(self.n_max) <= 0):
            raise ValueError("n_max must be positive")

    def __len__(self) -> int:
        return len(self.bins)

    def random_keys(self) -> np.ndarray:
        """One stable integer per measurement, from its ``random_id``: with the stage's seed it
        seeds every random number drawn for the measurement."""
        from .optimise import stable_keys
        return stable_keys(self.random_id)

    def select(self, index, *, drop_references: bool = False) -> Measurements:
        """The measurements at ``index`` (positions, in that order). The reference spectra of each
        selected measurement must be selected too (select whole series); they are renumbered.
        With ``drop_references`` the selection has none (for the bootstrap, which resamples
        each measurement from its own fit and does not use them)."""
        index = np.asarray(index, dtype=np.int64)
        if index.ndim != 1 or len(set(index.tolist())) != len(index):
            raise ValueError("index must be a one-dimensional list of distinct positions")
        new = {int(old): j for j, old in enumerate(index)}
        refs = []
        for i in index:
            ref = None if drop_references else self.references[int(i)]
            if ref is not None and any(r not in new for r in ref):
                raise ValueError("a measurement was selected without its reference spectra")
            refs.append(None if ref is None else tuple(new[r] for r in ref))
        return Measurements(
            bins=tuple(self.bins[i] for i in index),
            observed=tuple(self.observed[i] for i in index),
            shapes=tuple(self.shapes[i] for i in index), n_max=self.n_max[index],
            references=tuple(refs), spectrum_id=self.spectrum_id[index],
            data_id=self.data_id[index], time=self.time[index], role=self.role[index],
            random_id=self.random_id[index],
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
        role: Sequence[str] | None = None,
        random_id: Sequence[str] | None = None,
    ) -> Measurements:
        """Measurements from plain arrays (for tests and for data that did not come from the
        picker). Missing identifiers become ``m0, m1, ...``, series 0, time 0 and no role."""
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
            role=None if role is None else np.asarray(role, dtype=str),
            random_id=None if random_id is None else np.asarray(random_id, dtype=str),
        )

    @classmethod
    def from_binned(
        cls,
        binned: BinnedSet,
        *,
        n_max_factor: float = 1.2,
        nterm_subtract: int = 1,
        amide_rules: chem.AmideRules = chem.DEFAULT_AMIDE_RULES,
        registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
        shapes: Mapping[int, np.ndarray] | None = None,
        kinetics: Sequence[bool] | None = None,
    ) -> Measurements:
        """Measurements from a picked set. Drop flagged spectra first (``BinnedSet.select``).
        ``shapes`` replaces the envelope of some series (``data_id``), for example an ion-type
        mixture or an inferred envelope (:mod:`hxex.picking.envelope`). ``kinetics`` marks the
        spectra of kinetics runs (:func:`hxex.io.kinetics_spectra`), whose reference is taken
        from the run's own scans (:func:`hxex.noise.kinetics_references`).

        ``n_max = n_max_factor * exchangeable sites`` (amides plus user ``Hex``; at least one site,
        for fragment ions without any); the shape is
        the natural-abundance envelope of the peptide and ion type (``chem.natural_abundance``);
        the references are the undeuterated controls of the same series (``data_id``, role
        ``'undeut'``; :func:`hxex.noise.control_references`).
        ``nterm_subtract`` and ``amide_rules`` should be those used for picking.
        """
        if not n_max_factor > 0:
            raise ValueError("n_max_factor must be positive")
        shape_cache: dict[tuple[str, str], np.ndarray] = {}
        sites_cache: dict[tuple[str, str], int] = {}
        shapes = shapes or {}
        bins, observed, shape_list, n_max = [], [], [], []
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
            series = int(binned.data_id[i])
            shape_list.append(np.asarray(shapes[series], dtype=np.float64)
                              if series in shapes else shape_cache[key])
            # an ion without exchangeable sites (a z ion of one residue) is fitted with the cap of
            # one: its mass shift can still be measured (scrambling)
            n_max.append(n_max_factor * max(sites_cache[key], 1))
        references = control_references(binned.data_id, binned.role)
        if kinetics is not None:
            if len(kinetics) != len(binned):
                raise ValueError("kinetics must have one entry per spectrum")
            from_run = kinetics_references(binned.data_id, binned.rep, observed, kinetics)
            references = [k if k is not None else c
                          for k, c in zip(from_run, references, strict=True)]
        return cls(
            bins=tuple(bins), observed=tuple(observed), shapes=tuple(shape_list),
            n_max=np.asarray(n_max, dtype=np.float64),
            references=tuple(references),
            spectrum_id=binned.spectrum_id, data_id=binned.data_id, time=binned.time,
            role=binned.role, random_id=measurement_identity(binned),
        )


def measurement_identity(binned: BinnedSet) -> np.ndarray:
    """What each spectrum of ``binned`` is: sample, peptide, modification, residue range, charge,
    replicate, time, role and label, joined into one text. Unlike the spectrum id it does not
    contain the project's name or depend on how the files are laid out, so the random numbers
    keyed by it (:attr:`Measurements.random_id`) are the same whichever project or batch the
    spectrum is fitted in. Spectra that are the same in every one of these get ``#1``, ``#2``,
    ... after the first, in the order of their spectrum ids."""
    def text(value) -> str:
        if isinstance(value, (float, np.floating)):
            return repr(float(value))
        return str(value)

    fields = (binned.sample, binned.peptide, binned.modification, binned.start_seq,
              binned.end_seq, binned.charge, binned.rep, binned.time, binned.role, binned.label)
    ids = ["\x1f".join(text(f[i]) for f in fields) for i in range(len(binned))]
    groups: dict[str, list[int]] = {}
    for i, key in enumerate(ids):
        groups.setdefault(key, []).append(i)
    for key, members in groups.items():
        if len(members) > 1:
            for n, i in enumerate(sorted(members, key=lambda j: str(binned.spectrum_id[j]))):
                if n:
                    ids[i] = f"{key}#{n}"
    return np.asarray(ids, dtype=str)
