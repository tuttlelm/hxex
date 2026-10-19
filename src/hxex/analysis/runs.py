# SPDX-License-Identifier: GPL-3.0-or-later
"""Comparing separate runs (docs/ANALYSIS.md, phase 4): the comparisons within a run, across
results bundles.

A :class:`RunSet` holds several runs under names of their own; the run is a column (``run``)
beside ``sample`` in every analysis table, so a species is identified by ``(run, sample, ...)``,
and everything that takes the results of one run (:func:`hxex.analysis.species_uptake`,
:func:`hxex.analysis.compare_peptides`, and so the residue map, the population choice, the
comparisons and :func:`hxex.report.write_analysis`) takes a ``RunSet`` the same way. A merged
sample (``merge``) pools its runs' spectra and has a blank ``run``. ``run:sample``
(:data:`SEPARATOR`) is a display label (:func:`display_labels`: used only where the same sample
name occurs in several runs of a table) and the way to name one run's sample in a setting
(:func:`resolve_sample`; ``reference``, ``samples``, the keys of ``merge``). Peptides are
matched by their residues on the protein (or, on request,
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
           "n_max_factor", "drop_no_signal", "registry")
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

    ``merge`` combines runs: it gives samples a label of
    their own (``{'A:WT': 'WT', 'B:WT': 'WT'}``), and samples with the same label are one sample,
    their replicate spectra pooled, with a blank ``run``; the others keep their run and name
    (:meth:`state`)."""

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

    @property
    def time_scale(self) -> str:
        """The runs' time scale (:attr:`hxex.report.Results.time_scale`), which must agree
        (times only ordering the points in one run cannot be compared with real times)."""
        scales = {name: getattr(r, "time_scale", "seconds") for name, r in self.runs.items()}
        if len(set(scales.values())) > 1:
            raise ValueError("the runs' time scales differ ([input] time_scale): "
                             + ", ".join(f"{name}: {s}" for name, s in scales.items()))
        return next(iter(scales.values()))

    def unmerged(self) -> RunSet:
        """The same runs without ``merge``."""
        return RunSet(self.runs)

    def state(self, run: str, sample: str) -> tuple[str, str]:
        """The ``(run, sample)`` of ``sample`` of ``run`` in this set's tables: ``('', label)``
        for a merged sample, else ``(run, sample)``."""
        key = self.label(run, sample)
        return ("", self.merge[key]) if key in self.merge else (run, sample)

    def is_merged(self, run: str, sample: str) -> bool:
        return self.label(run, sample) in self.merge

    @property
    def names(self) -> list[str]:
        return list(self.runs)

    @staticmethod
    def label(run: str, sample: str) -> str:
        """The label of ``sample`` of ``run`` (``run:sample``)."""
        return f"{run}{SEPARATOR}{sample}"

    @staticmethod
    def split(label: str) -> tuple[str, str]:
        """``(run, sample)`` of a ``run:sample`` label."""
        run, _, sample = str(label).partition(SEPARATOR)
        return run, sample

    def _records(self, ks, exclude_flagged, filters, quiet, controls_changes=None):
        """:func:`hxex.analysis.species._records` of every run, each group keyed by its
        :meth:`state`: the per-spectrum filters are applied within each run, the per-species
        ones once, after merged samples have pooled their spectra."""
        from .species import _filter_species, _spectrum_records

        groups, corrected, nterm = {}, {}, {}
        dropped: dict[str, int] = {}
        for name, results in self.runs.items():
            g, c, nterm[name], d = _spectrum_records(
                results, ks, exclude_flagged, filters, quiet, controls_changes)
            for key, n in d.items():
                dropped[key] = dropped.get(key, 0) + n
            for key, recs in g.items():  # merged samples pool their spectra, each tagged by run
                groups.setdefault((*self.state(name, key[1]), *key[2:]), []).extend(
                    (*r[:8], name) for r in recs)
            for (_, s), v in c.items():
                state = self.state(name, s)
                if corrected.get(state, v) != v and not quiet:
                    log.warning("%s pools runs with and without back-exchange correction",
                                state[1])
                corrected[state] = corrected.get(state, v) and v
        if len(set(nterm.values())) > 1:
            raise ValueError("the runs subtract different numbers of N-terminal residues "
                             f"({_listing(nterm)}): their residues cannot be compared")
        if len(set(corrected.values())) > 1 and not quiet:
            log.warning("back-exchange correction differs between the runs of [analysis.runs] "
                        "(corrected: %s)",
                        ", ".join(label(*s) for s, v in corrected.items() if v) or "none")
        first = self.names[0]
        groups = _filter_species(groups, ks, filters, dropped, quiet)
        return groups, corrected, nterm[first]


def _listing(values: Mapping) -> str:
    return ", ".join(f"{k}: {v}" for k, v in values.items())


def label(run: str, sample: str) -> str:
    """``run:sample``, or ``sample`` alone when ``run`` is blank."""
    return f"{run}{SEPARATOR}{sample}" if run else str(sample)


def sample_setting(setting, run: str, sample: str):
    """The value of a per-sample setting for ``sample`` of ``run``: ``setting`` itself when it
    is one value for every sample (not a mapping), else its entry for the sample under the run's
    name (``{'A': {'WT': ...}}``, the run-specific value) when there is one, else its entry for
    the sample (``{'WT': ...}``, for every run); None when the sample has none."""
    if setting is None or not isinstance(setting, Mapping):
        return setting
    if run:
        own = setting.get(run)
        if isinstance(own, Mapping) and own.get(sample) is not None:
            return own[sample]
    value = setting.get(sample)
    return None if isinstance(value, Mapping) else value


def display_labels(runs, samples) -> dict[tuple[str, str], str]:
    """The display label of every ``(run, sample)`` in the columns given: ``run:sample`` where
    the same sample name occurs in several runs of the table (an unmerged sample in a
    :class:`RunSet`), the sample's name alone otherwise (one run, or a merged sample)."""
    pairs = dict.fromkeys(zip((str(r) for r in runs), (str(s) for s in samples), strict=True))
    by_sample: dict[str, set[str]] = {}
    for run, sample in pairs:
        by_sample.setdefault(sample, set()).add(run)
    return {(run, sample): label(run, sample) if len(by_sample[sample]) > 1 else sample
            for run, sample in pairs}


def resolve_sample(runs, samples, name: str, what: str = "sample") -> tuple[str, str]:
    """The ``(run, sample)`` of a table that ``name`` (a setting such as ``reference``) names:
    a sample's name, when it is in one run of the table, or ``run:sample`` naming one run's.
    An ambiguous name lists the choices; an unknown one the samples present."""
    pairs = list(dict.fromkeys(zip((str(r) for r in runs), (str(s) for s in samples),
                                   strict=True)))
    name = str(name)
    if ("", name) in pairs:  # one run's sample, or a merged one: named by its name alone
        return "", name
    found = [p for p in pairs if p[1] == name]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ValueError(f"{what} {name!r} is in several runs: name one of "
                         + ", ".join(label(*p) for p in found) + " (or merge them)")
    if SEPARATOR in name:
        run, sample = RunSet.split(name)
        if (run, sample) in pairs:
            return run, sample
    present = sorted({display_labels(runs, samples)[p] for p in pairs})
    raise ValueError(f"no {what} {name!r} in the results (samples: {', '.join(present)})")


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
    runs are those in its ``runs`` column (a merged sample), else its ``run``; a table whose
    rows name no run (one run's) is refused."""
    per: dict[tuple[str, str], dict[str, tuple[set, set]]] = {}
    bare: list[str] = []
    for run, sample, peptide, mod, start, end, amides, runs in zip(
            uptake.run, uptake.sample, uptake.peptide, uptake.modification, uptake.start,
            uptake.end, uptake.n_amides, uptake.runs, strict=True):
        names = [r for r in str(runs).split(",") if r] or ([str(run)] if run else [])
        if not names:
            bare.append(str(sample))
            continue
        for run in names:
            where, count = per.setdefault((str(peptide), str(mod)), {}).setdefault(
                run, (set(), set()))
            where.add((int(start), int(end)))
            count.add(round(float(amides), 6))
    if bare:
        shown = ", ".join(sorted(set(bare))[:5])
        raise ValueError(f"check_identity needs the table of a RunSet (rows with a run); "
                         f"these samples name no run: {shown}")
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
    """One row per run and sample (not the reference), time and residue where both have a K=1
    value: whether each keeps two populations (``two_reference``, ``two``), the reasons it does
    not (``reason_reference``, ``reason``; see :data:`hxex.analysis.populations.REASONS`) and
    ``agree``. The reference is ``(reference_run, reference)``."""

    run: np.ndarray
    sample: np.ndarray
    time: np.ndarray
    residue: np.ndarray
    two_reference: np.ndarray
    two: np.ndarray
    reason_reference: np.ndarray
    reason: np.ndarray
    agree: np.ndarray
    reference: str = ""
    reference_run: str = ""

    def __len__(self) -> int:
        return len(self.sample)

    def columns(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name not in ("reference", "reference_run")}

    def where(self, *, sample=None, time=None, run=None) -> np.ndarray:
        keep = np.ones(len(self), dtype=bool)
        for name, value in (("run", run), ("sample", sample), ("time", time)):
            if value is not None:
                values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
                keep &= np.isin(getattr(self, name), values)
        return keep


def compare_choices(choice, reference: str) -> ChoiceComparison:
    """Every sample against ``reference`` (a sample's name, or ``run:sample``;
    :func:`resolve_sample`) in the choice between one and two populations
    (:func:`hxex.analysis.choose_populations`), residue by residue: where one keeps two
    populations and the other one, only the K=1 values are compared between them (see the module
    docstring)."""
    ref = resolve_sample(choice.run, choice.sample, reference, "reference sample")
    value = {(str(u), str(s), float(t), int(r)): (bool(two), str(why))
             for u, s, t, r, avg, two, why in zip(
                 choice.run, choice.sample, choice.time, choice.residue, choice.fD_avg,
                 choice.two_populations, choice.reason, strict=True) if np.isfinite(avg)}
    rows = []
    for (run, sample, time, res), (two, why) in sorted(value.items()):
        if (run, sample) == ref or (*ref, time, res) not in value:
            continue
        two_ref, why_ref = value[(*ref, time, res)]
        rows.append((run, sample, time, res, two_ref, two, why_ref, why, two_ref == two))
    names = [f.name for f in fields(ChoiceComparison)
             if f.name not in ("reference", "reference_run")]
    cols = {name: np.array([r[i] for r in rows]) for i, name in enumerate(names)}
    if not rows:
        cols = {name: np.array([]) for name in names}
    return ChoiceComparison(**cols, reference=ref[1], reference_run=ref[0])
