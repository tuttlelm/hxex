# SPDX-License-Identifier: GPL-3.0-or-later
"""Comparing separate runs (docs/ANALYSIS.md, phase 4): the comparisons within a run, across
results bundles.

A :class:`RunSet` holds several runs under names of their own; its samples are labelled
``run:sample`` (:data:`SEPARATOR`), and everything that takes the results of one run
(:func:`hxex.analysis.species_uptake`, :func:`hxex.analysis.compare_peptides`, and so the
residue map, the population choice, the comparisons and :func:`hxex.report.write_analysis`) takes
a ``RunSet`` the same way. Peptides are matched by their residues on the protein (or, on request,
by identical sequence with modifications; :func:`hxex.analysis.compare_peptides`, charge states
combined), never by ``data_id``, a position within its run; :func:`check_identity` lists the
species whose position or amide count differs between runs. :func:`compare_settings` lists
where the runs' settings differ and which differences matter for comparing them (picker, noise
model, K values, bootstrap, controls and their limits).

One population in one run and two in another: every spectrum is fitted at every K, so the K=1
values (the average) are always compared; a K=2 rank or population is compared only where both
sides keep a two-population description (:class:`hxex.analysis.SpeciesFilters`), and where the
choice between one and two populations differs (:func:`compare_choices`) the difference is
reported, never resolved by imposing one side's description on the other. The same rule holds
between samples of one run.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import numpy as np

from .species import SpeciesUptake

log = logging.getLogger(__name__)

SEPARATOR = ":"

# settings whose difference matters when runs are compared (a setting or a section, dotted)
MATTERS = ("picker", "fit.noise", "fit.k_min", "fit.k_max", "fit.criterion",
           "fit.min_population", "bootstrap", "controls", "envelope", "envelope_fallback_time",
           "undeut_time", "n_max_factor", "drop_no_signal", "registry")
# settings that do not change the results (resources and drawing)
IGNORED = ("plots", "memory_budget_bytes", "max_chunk", "fit.backend", "bootstrap.keep_draws",
           "bootstrap.curve_band")
ABSENT = "(none)"


def _under(name: str, prefixes) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


@dataclass(frozen=True)
class RunSet:
    """Several runs (:class:`hxex.report.Results`) by name, analysed together; the first is the
    reference run of :func:`check_identity`. Run names must not contain :data:`SEPARATOR`.

    ``merge`` combines runs (pyHXExpress ``combine_batch_data``): it gives samples a label of
    their own (``{'A:WT': 'WT', 'B:WT': 'WT'}``), and samples with the same label are one sample,
    their replicate spectra pooled; the others keep ``run:sample``."""

    runs: Mapping[str, object] = field(default_factory=dict)
    merge: Mapping[str, str] | None = None

    def __post_init__(self):
        if not self.runs:
            raise ValueError("a RunSet needs at least one run")
        for name in self.runs:
            if not name or SEPARATOR in name:
                raise ValueError(f"run name {name!r}: must be non-empty without {SEPARATOR!r}")
        object.__setattr__(self, "runs", dict(self.runs))
        merge = {str(k): str(v) for k, v in (self.merge or {}).items()}
        for key, label in merge.items():
            run, sample = self.split(key)
            if run not in self.runs or not sample:
                raise ValueError(f"merge: {key!r} is not run:sample of a run in the set")
            if not label:
                raise ValueError(f"merge: {key!r} needs a non-empty label")
        object.__setattr__(self, "merge", merge)

    def unmerged(self) -> RunSet:
        """The same runs without ``merge``."""
        return RunSet(self.runs)

    def sample_label(self, run: str, sample: str) -> str:
        """The label of ``sample`` of ``run`` in this set (``merge``, else ``run:sample``)."""
        key = self.label(run, sample)
        return self.merge.get(key, key)

    @property
    def names(self) -> list[str]:
        return list(self.runs)

    @staticmethod
    def label(run: str, sample: str) -> str:
        """The label of ``sample`` of ``run`` (``run:sample``)."""
        return f"{run}{SEPARATOR}{sample}"

    @staticmethod
    def split(label: str) -> tuple[str, str]:
        """``(run, sample)`` of a label."""
        run, _, sample = str(label).partition(SEPARATOR)
        return run, sample

    def _records(self, ks, exclude_flagged, filters, quiet, controls_changes=None):
        """:func:`hxex.analysis.species._records` of every run, the samples relabelled: the
        per-spectrum filters are applied within each run, the per-species ones once, after
        merged samples have pooled their spectra."""
        from .species import _filter_species, _spectrum_records

        groups, corrected, nterm, undeut = {}, {}, {}, {}
        dropped: dict[str, int] = {}
        for name, results in self.runs.items():
            g, c, nterm[name], undeut[name], d = _spectrum_records(
                results, ks, exclude_flagged, filters, quiet, controls_changes)
            for key, n in d.items():
                dropped[key] = dropped.get(key, 0) + n
            for key, recs in g.items():  # merged samples pool their spectra, each tagged by run
                groups.setdefault((self.sample_label(name, key[0]), *key[1:]), []).extend(
                    (*r[:8], name) for r in recs)
            for s, v in c.items():
                label = self.sample_label(name, s)
                if corrected.get(label, v) != v and not quiet:
                    log.warning("%s pools runs with and without back-exchange correction", label)
                corrected[label] = corrected.get(label, v) and v
        if len(set(nterm.values())) > 1:
            raise ValueError("the runs subtract different numbers of N-terminal residues "
                             f"({_listing(nterm)}): their residues cannot be compared")
        if len(set(undeut.values())) > 1 and not quiet:
            log.warning("the runs' undeuterated times differ (%s); coverage and the figures use "
                        "the first run's", _listing(undeut))
        if len(set(corrected.values())) > 1 and not quiet:
            log.warning("back-exchange correction differs between runs (corrected: %s)",
                        ", ".join(s for s, v in corrected.items() if v) or "none")
        first = self.names[0]
        groups = _filter_species(groups, ks, filters, dropped, quiet)
        return groups, corrected, nterm[first], undeut[first]


def _listing(values: Mapping) -> str:
    return ", ".join(f"{k}: {v}" for k, v in values.items())


def _flatten(data, prefix: str = "") -> dict[str, str]:
    out = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(_flatten(value, name + ".") if value else {name: "{}"})
        else:
            out[name] = ABSENT if value is None else repr(value) if not isinstance(
                value, str) else value
    return out


@dataclass(frozen=True)
class SettingsComparison:
    """The run settings that differ between runs: one row per ``setting`` (dotted, as in
    :class:`hxex.run.RunConfig`), with ``matters`` (whether the difference matters for comparing
    the runs, :data:`MATTERS`) and the value in each run (``values``, one column per run of
    ``runs``; :data:`ABSENT` where a run does not have the setting or section)."""

    runs: tuple[str, ...]
    setting: np.ndarray
    matters: np.ndarray
    values: np.ndarray

    def __len__(self) -> int:
        return len(self.setting)

    def columns(self) -> dict[str, np.ndarray]:
        return {"setting": self.setting, "matters": self.matters,
                **{run: self.values[:, j] for j, run in enumerate(self.runs)}}


def compare_settings(runs: RunSet | Mapping[str, object], *, quiet: bool = False
                     ) -> SettingsComparison:
    """Where the runs' settings differ (the settings with defaults filled in, so that runs written
    by different versions compare fairly; :data:`IGNORED` left out). A warning names every
    difference that matters, unless ``quiet``; the package versions are compared too."""
    runs = runs if isinstance(runs, RunSet) else RunSet(runs)
    flat = {}
    for name, results in runs.runs.items():
        settings = getattr(results, "settings", None)
        config = settings.run if settings is not None else None
        if config is None:
            log.warning("run %s recorded no settings", name)
        values = _flatten(config.to_dict()) if config is not None else {}
        if settings is not None:
            values.update(_flatten({"versions": settings.versions}))
        flat[name] = values
    names = sorted({k for v in flat.values() for k in v if not _under(k, IGNORED)})
    rows = []
    for setting in names:
        values = [flat[run].get(setting, ABSENT) for run in runs.names]
        if len(set(values)) > 1:
            rows.append((setting, _under(setting, MATTERS), values))
    if not quiet:
        for setting, matters, values in rows:
            if matters:
                log.warning("the runs differ in %s (%s)", setting,
                            ", ".join(f"{r}: {v}" for r, v in zip(runs.names, values,
                                                                  strict=True)))
    return SettingsComparison(
        runs=tuple(runs.names), setting=np.array([r[0] for r in rows], dtype=object),
        matters=np.array([r[1] for r in rows], dtype=bool),
        values=np.array([r[2] for r in rows], dtype=object).reshape(len(rows), len(runs.names)))


@dataclass(frozen=True)
class IdentityCheck:
    """Species (``peptide`` and ``modification``) found in several runs whose position or
    amide count differs between them: one row per such species and run, with that run's
    ``start``, ``end`` and ``n_amides`` (positions joined by ``;`` if a run has several) and the
    ``issue`` (``'position'``, ``'amides'`` or both). Such species are not matched between the
    runs whose positions differ."""

    peptide: np.ndarray
    modification: np.ndarray
    run: np.ndarray
    start: np.ndarray
    end: np.ndarray
    n_amides: np.ndarray
    issue: np.ndarray

    def __len__(self) -> int:
        return len(self.peptide)

    def columns(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def check_identity(uptake: SpeciesUptake, *, quiet: bool = False) -> IdentityCheck:
    """The species of ``uptake`` (the table of a :class:`RunSet`) whose identity does not agree
    between runs (see :class:`IdentityCheck`); a warning counts them unless ``quiet``. A row's
    runs are those in its ``runs`` column (a merged sample), else the run part of its label;
    a table whose labels have no run part (one run's) is refused."""
    per: dict[tuple[str, str], dict[str, tuple[set, set]]] = {}
    bare: list[str] = []
    for label, peptide, mod, start, end, amides, runs in zip(
            uptake.sample, uptake.peptide, uptake.modification, uptake.start, uptake.end,
            uptake.n_amides, uptake.runs, strict=True):
        names = [r for r in str(runs).split(",") if r]
        if not names:
            run, sample = RunSet.split(label)
            if not sample:
                bare.append(str(label))
                continue
            names = [run]
        for run in names:
            where, count = per.setdefault((str(peptide), str(mod)), {}).setdefault(
                run, (set(), set()))
            where.add((int(start), int(end)))
            count.add(round(float(amides), 6))
    if bare:
        shown = ", ".join(sorted(set(bare))[:5])
        raise ValueError(f"check_identity needs the table of a RunSet (samples labelled "
                         f"run:sample); these samples name no run: {shown}")
    rows = []
    for (peptide, mod), by_run in sorted(per.items()):
        if len(by_run) < 2:
            continue
        issues = [name for name, n in (("position", 0), ("amides", 1))
                  if len({frozenset(v[n]) for v in by_run.values()}) > 1]
        for run, (where, count) in by_run.items():
            if issues:
                where = sorted(where)
                rows.append({"peptide": peptide, "modification": mod, "run": run,
                             "start": ";".join(str(s) for s, _ in where),
                             "end": ";".join(str(e) for _, e in where),
                             "n_amides": ";".join(f"{c:g}" for c in sorted(count)),
                             "issue": "+".join(issues)})
    if rows and not quiet:
        n = len({(r["peptide"], r["modification"]) for r in rows})
        log.warning("%d species differ in position or amide count between runs (see "
                    "check_identity); they are not matched where their positions differ", n)
    cols = {f.name: np.array([r[f.name] for r in rows], dtype=object) for f in
            fields(IdentityCheck)}
    return IdentityCheck(**cols)


@dataclass(frozen=True)
class ChoiceComparison:
    """One row per sample (not the reference), time and residue where both have a K=1 value:
    whether each keeps two populations (``two_reference``, ``two``), the reasons it does not
    (``reason_reference``, ``reason``; see :data:`hxex.analysis.populations.REASONS`) and
    ``agree``."""

    sample: np.ndarray
    time: np.ndarray
    residue: np.ndarray
    two_reference: np.ndarray
    two: np.ndarray
    reason_reference: np.ndarray
    reason: np.ndarray
    agree: np.ndarray
    reference: str = ""

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "reference"}

    def where(self, *, sample=None, time=None) -> np.ndarray:
        keep = np.ones(len(self), dtype=bool)
        for name, value in (("sample", sample), ("time", time)):
            if value is not None:
                values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
                keep &= np.isin(getattr(self, name), values)
        return keep


def compare_choices(choice, reference: str) -> ChoiceComparison:
    """Every sample against ``reference`` in the choice between one and two populations
    (:func:`hxex.analysis.choose_populations`), residue by residue: where one keeps two
    populations and the other one, only the K=1 values are compared between them (see the module
    docstring)."""
    value = {(str(s), float(t), int(r)): (bool(two), str(why)) for s, t, r, avg, two, why in zip(
        choice.sample, choice.time, choice.residue, choice.fD_avg, choice.two_populations,
        choice.reason, strict=True) if np.isfinite(avg)}
    if not any(k[0] == reference for k in value):
        raise ValueError(f"no reference sample {reference!r} in the choice")
    rows = []
    for (sample, time, res), (two, why) in sorted(value.items()):
        if sample == reference or (reference, time, res) not in value:
            continue
        two_ref, why_ref = value[(reference, time, res)]
        rows.append((sample, time, res, two_ref, two, why_ref, why, two_ref == two))
    names = [f.name for f in fields(ChoiceComparison) if f.name != "reference"]
    cols = {name: np.array([r[i] for r in rows]) for i, name in enumerate(names)}
    if not rows:
        cols = {name: np.array([]) for name in names}
    return ChoiceComparison(**cols, reference=reference)
