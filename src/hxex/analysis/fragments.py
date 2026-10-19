# SPDX-License-Identifier: GPL-3.0-or-later
"""Fragment ions: their deuteration over time, ladder steps and single-amide kinetics.

The simple ladder approach (docs/ANALYSIS.md, "Fragment branch"): consecutive fragments of one
ion series differ by the amides at the cleavage site (:func:`hxex.fragments.amide_difference`),
so their difference, scan by scan, follows those amides. :func:`fragment_uptake` gives the K = 1
deuteration of every fragment-ion spectrum of a run read from mzML or Thermo files;
:func:`ladder_steps` lists the steps of one ion series and the residues each measures;
:func:`difference_trace` gives the difference of two fragments over time, matched scan by scan;
:func:`closure_trace` and :func:`closure_table` test that complementary ions (c and z, a and x,
b and y) add up to the precursor, a check of the fragments' fits;
:func:`fit_decay` fits ``A exp(-k t) + C`` to a trace; :func:`fit_residues` fits per-residue
deuteration (0 to 1) to all fragments of a parent at one time, by bounded least squares.

Fragment charges, precursor charges and conditions (for example with and without in-source
activation) are kept apart: they may give different exchange. Nothing is corrected for
back-exchange.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np

from .. import chem
from ..kint import ExchangeConditions, conditions_dict
from ..spectra import FULLDEUT, PeptideEntry

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FragmentUptake:
    """K = 1 deuteration of every fragment-ion (and precursor-ion) spectrum, one row each.

    Identity: ``sample``, ``condition`` (the fragment list's label, or the scan conditions:
    activation, pre-activation, filter), ``parent`` (its sequence), ``parent_start`` (the
    parent's first residue on the protein), ``parent_charge`` (the precursor charge),
    ``fragment`` (``'c-1:4'``; ``'M:8'`` for the precursor ion), ``ion_type``, ``length``
    (residues) and ``charge`` (of the fragment), ``activation``, ``data_id`` (the series) and
    ``rep``. Values: ``time`` (s), ``mean`` (deuterons beyond natural abundance) and ``mean_se``
    (from the K = 1 bootstrap, NaN without one), ``n_amides`` (the exchangeable amides the
    fragment holds; :func:`hxex.fragments.fragment_amides`), ``flagged`` (quality flags) and
    ``spectrum_id``. ``nterm_subtract`` and ``amide_rules`` are those of the run's picker.
    ``run``: the run each row comes from ('' for one run, the run's name in a
    :class:`hxex.analysis.RunSet`; fragment rows always keep their run, as fragment series are
    never pooled) and ``merged``: whether the row's sample is a merged one (``merge`` of the
    set: its runs are drawn side by side in the sample's figures). The figures group rows by
    :attr:`state_run` and ``sample``: a merged sample is one state (``state_run`` ''), an
    unmerged one is a state per run; within a state the series are ``(run, condition)``."""

    sample: np.ndarray
    condition: np.ndarray
    parent: np.ndarray
    parent_start: np.ndarray
    parent_charge: np.ndarray
    fragment: np.ndarray
    ion_type: np.ndarray
    length: np.ndarray
    charge: np.ndarray
    activation: np.ndarray
    data_id: np.ndarray
    rep: np.ndarray
    time: np.ndarray
    mean: np.ndarray
    mean_se: np.ndarray
    n_amides: np.ndarray
    flagged: np.ndarray
    spectrum_id: np.ndarray
    nterm_subtract: int = 1
    amide_rules: chem.AmideRules = chem.DEFAULT_AMIDE_RULES
    run: np.ndarray | None = None
    merged: np.ndarray | None = None

    def __post_init__(self):  # built without them: one run, nothing merged
        _fill_runs(self)

    def __len__(self) -> int:
        return len(self.time)

    @property
    def state_run(self) -> np.ndarray:
        """The run of each row's state: its ``run``, or '' for a merged sample."""
        return _state_run(self)

    def states(self) -> list[tuple[str, str]]:
        """The distinct ``(state_run, sample)`` of the table, in order of first appearance."""
        return list(dict.fromkeys(zip(self.state_run.tolist(), self.sample.astype(str).tolist(),
                                      strict=True)))

    def columns(self) -> dict[str, np.ndarray]:
        return _columns(self)

    def select(self, mask) -> FragmentUptake:
        mask = np.asarray(mask)
        return FragmentUptake(**{name: col[mask] for name, col in self.columns().items()},
                              nterm_subtract=self.nterm_subtract, amide_rules=self.amide_rules)

    def where(self, **conditions) -> np.ndarray:
        """Rows whose columns equal the values given (``fragment='c-1:4', charge=1``); a value
        that is a list or tuple matches any of its items."""
        keep = np.ones(len(self), dtype=bool)
        for name, value in conditions.items():
            if value is None:
                continue
            col = getattr(self, name)
            values = value if isinstance(value, (list, tuple, set)) else [value]
            keep &= np.isin(col, list(values))
        return keep

    def parent_entry(self, i: int = 0) -> PeptideEntry:
        """The parent of row ``i`` as an entry (for :mod:`hxex.fragments`)."""
        start = int(self.parent_start[i])
        peptide = str(self.parent[i])
        n = len(chem.parse_sequence(peptide))
        return PeptideEntry("", str(self.sample[i]), peptide, f"{start}-{start + n - 1}", start,
                            start + n - 1, int(self.parent_charge[i]))


def _condition(target) -> str:
    """A target's condition: the fragment list's label, else its scan conditions."""
    label = dict(target.entry.extra).get("condition")
    if label:
        return str(label)
    parts = [target.activation or ""]
    if target.preactivation is not None:
        parts.append(f"pre:{target.preactivation}")
    if target.filter_contains:
        parts.append(f"+{target.filter_contains}")
    if target.filter_excludes:
        parts.append(f"-{target.filter_excludes}")
    return " ".join(p for p in parts if p)


def _fill_runs(table) -> None:
    n = len(table)
    run = table.run if table.run is not None else np.full(n, "", dtype=object)
    object.__setattr__(table, "run", np.asarray(run, dtype=object))
    merged = table.merged if table.merged is not None else np.zeros(n, dtype=bool)
    object.__setattr__(table, "merged", np.asarray(merged, dtype=bool))


def _state_run(table) -> np.ndarray:
    return np.where(np.asarray(table.merged, dtype=bool), "", table.run.astype(str)).astype(str)


def _columns(table) -> dict[str, np.ndarray]:
    """The table's columns, ``run`` first and ``merged`` last."""
    cols = {f.name: getattr(table, f.name) for f in fields(table)
            if isinstance(getattr(table, f.name), np.ndarray) and f.name not in ("run", "merged")}
    return {"run": table.run, **cols, "merged": table.merged}


def fragment_uptake(results, *, exclude_flagged: bool = False) -> FragmentUptake:
    """The K = 1 deuteration of every fragment-ion spectrum of ``results`` (a run read from mzML
    or Thermo files with fragment targets, :func:`hxex.fragments.fragment_targets`). Spectra
    the review rejected are left out, flagged ones too with ``exclude_flagged``; series that
    are not fragments of a parent (peptides read from MS1) are left out, and so are the fully
    deuterated controls (the spectra with the role ``'fulldeut'``): they are not a point of
    the kinetics.

    From a :class:`hxex.analysis.RunSet`, the fragments of every run read from scan files, in
    one table whose ``run`` column names each row's run; a merged sample (``merge``) keeps the
    merge label in ``sample`` with ``merged`` True, so that the runs of one sample (two
    activations, say) are drawn side by side in its figures, as series ``(run, condition)``,
    and never pooled (each run has an exchange clock of its own)."""
    from ..fragments import fragment_amides
    from ..io.mzml import mzml_series_targets
    from .runs import RunSet

    if isinstance(results, RunSet):
        return _runset_uptake(results, exclude_flagged)

    settings = results.settings
    source = settings.spectra_source if settings is not None else None
    if not source:
        raise ValueError("fragment_uptake needs a run read from mzML or Thermo files")
    targets = mzml_series_targets(source)
    fit, binned = results.fit, results.binned
    if 1 not in fit.fits:
        raise ValueError("fragment_uptake needs the K = 1 fits")
    picker = settings.picker
    nterm = picker.nterm_subtract if picker is not None else 1
    rules = picker.amide_rules if picker is not None else chem.DEFAULT_AMIDE_RULES
    index = {str(s): j for j, s in enumerate(binned.spectrum_id)}
    keep = np.ones(len(fit), dtype=bool)
    if results.review is not None:
        keep &= results.review.status_of(fit) != "rejected"
    if getattr(results, "excluded_times", ()) or getattr(results, "excluded_series", ()):
        keep &= ~results.excluded  # left out of the analysis (exclude_times, other branch)
    keep &= np.asarray(fit.role, dtype=str) != FULLDEUT
    flagged = (np.asarray(results.quality.flagged, dtype=bool) if results.quality is not None
               else np.zeros(len(fit), dtype=bool))
    if exclude_flagged:
        keep &= ~flagged
    means = fit.fits[1].means[:, 0]
    amides: dict[tuple, int] = {}
    rows: dict[str, list] = {f.name: [] for f in fields(FragmentUptake)
                             if f.name not in ("nterm_subtract", "amide_rules", "run", "merged")}
    skipped = 0
    for i in np.flatnonzero(keep):
        target = targets.get(int(fit.data_id[i]))
        extra = dict(target.entry.extra) if target is not None else {}
        if "fragment" not in extra:
            skipped += 1
            continue
        start = int(str(extra["parent_range"]).split("-")[0])
        parent = PeptideEntry("", "", extra["parent_peptide"], extra["parent_range"], start,
                              start, int(extra["parent_charge"]))
        label = extra["fragment"]
        key = (parent.peptide, label)
        if key not in amides:
            amides[key] = len(fragment_amides(parent, label, nterm_subtract=nterm, rules=rules))
        boot = results.bootstrap_for(i, 1)
        j = index[str(fit.spectrum_id[i])]
        ion, _, length = label.rpartition(":")
        values = {
            "sample": str(binned.sample[j]), "condition": _condition(target),
            "parent": parent.peptide, "parent_start": start, "parent_charge": parent.charge,
            "fragment": label, "ion_type": ion,
            "length": int(length) if chem.ion_terminus(ion) != "M" else len(
                chem.parse_sequence(parent.peptide)),
            "charge": int(binned.charge[j]), "activation": target.activation or "",
            "data_id": int(fit.data_id[i]), "rep": int(binned.rep[j]),
            "time": float(fit.time[i]), "mean": float(means[i]),
            "mean_se": float(boot.mean_se[i, 0]) if boot is not None else np.nan,
            "n_amides": amides[key], "flagged": bool(flagged[i]),
            "spectrum_id": str(fit.spectrum_id[i]),
        }
        for name, value in values.items():
            rows[name].append(value)
    if skipped:
        log.info("fragment_uptake: %d spectra of series that are not fragments left out", skipped)
    kinds = {"parent_start": np.int64, "parent_charge": np.int64, "length": np.int64,
             "charge": np.int64, "data_id": np.int64, "rep": np.int64, "time": float,
             "mean": float, "mean_se": float, "n_amides": np.int64, "flagged": bool}
    return FragmentUptake(
        **{name: np.array(values, dtype=kinds.get(name, str)) for name, values in rows.items()},
        nterm_subtract=int(nterm), amide_rules=rules)


def _runset_uptake(runs, exclude_flagged: bool) -> FragmentUptake:
    """:func:`fragment_uptake` of the runs of a :class:`hxex.analysis.RunSet`."""
    from ..io.mzml import SCAN_READERS

    tables = {}
    for name, results in runs.runs.items():
        settings = results.settings
        source = settings.spectra_source if settings is not None else None
        if not source or source.get("reader") not in SCAN_READERS:
            continue  # a folder format (MS1 tables): no fragments, nothing to read
        try:
            u = fragment_uptake(results, exclude_flagged=exclude_flagged)
        except ValueError as exc:
            raise ValueError(f"run {name}: {exc}") from exc
        if len(u):
            tables[name] = u
    if not tables:
        raise ValueError("fragment_uptake needs a run read from mzML or Thermo files")
    first = next(iter(tables.values()))
    if any((u.nterm_subtract, u.amide_rules) != (first.nterm_subtract, first.amide_rules)
           for u in tables.values()):
        raise ValueError("the runs count fragment amides differently (nterm_subtract or amide "
                         "rules): their fragments cannot be compared")
    parts = []
    for name, u in tables.items():
        cols = u.columns()
        cols["merged"] = np.array([runs.is_merged(name, s) for s in cols["sample"]], bool)
        cols["sample"] = np.array([runs.state(name, s)[1] for s in cols["sample"]], str)
        cols["run"] = np.full(len(u), name, dtype=object)
        parts.append(cols)
    return FragmentUptake(**{key: np.concatenate([c[key] for c in parts]) for key in parts[0]},
                          nterm_subtract=first.nterm_subtract, amide_rules=first.amide_rules)


# --- ladder steps -------------------------------------------------------------------------------

@dataclass(frozen=True)
class LadderStep:
    """One row of a ladder: the fragment ``larger`` minus ``smaller`` (labels as in
    :func:`hxex.fragments.fragment_amides`; ``smaller`` None for the shortest fragment alone,
    ``larger`` None for a residue no step covers). ``span`` is the first and last parent
    position (1-based) whose backbone nitrogen the step adds, ``residues`` the exchanging ones
    among them (protein numbering), and ``note`` why a row has no single residue: ``'block'``
    (several exchanging residues), ``'proline'``, ``'N-terminal'`` (among the parent's first
    ``nterm_subtract``) or ``'no ion'`` (no fragment of the series covers it)."""

    larger: str | None
    smaller: str | None
    span: tuple[int, int]
    residues: tuple[int, ...]
    note: str = ""

    @property
    def single(self) -> bool:
        return len(self.residues) == 1 and self.larger is not None


def _nitrogens(n: int, ion_type: str, length: int, rules: chem.AmideRules) -> set[int]:
    """Parent positions (1-based) of the backbone nitrogens a fragment holds."""
    side = chem.ion_terminus(ion_type)
    if side == "M":
        return set(range(1, n + 1))
    boundary = rules.keeps_boundary(ion_type)
    if side == "N":
        return set(range(1, length + 1 + (1 if boundary else 0)))
    first = n - length + 1
    return set(range(first if boundary else first + 1, n + 1))


def ladder_steps(
    parent: PeptideEntry,
    ion_type: str,
    lengths: Sequence[int],
    *,
    nterm_subtract: int = 1,
    rules: chem.AmideRules = chem.DEFAULT_AMIDE_RULES,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
) -> list[LadderStep]:
    """The rows of the ``ion_type`` ladder of ``parent`` with the fragment ``lengths`` measured,
    in the order of the residues: one step per pair of consecutive lengths (the shortest alone
    first), and one row for each parent position that no step covers (``note`` says why).
    Positions are numbered on the protein (the parent's ``start_seq``)."""
    from ..fragments import amide_difference, fragment_amides

    residues = chem.parse_sequence(parent.peptide, registry)
    n = len(residues)
    side = chem.ion_terminus(ion_type)
    if side == "M":
        raise ValueError("a ladder needs a fragment ion type, not a parent-type ion")
    lengths = sorted({int(x) for x in lengths if 1 <= int(x) < n})
    kw = {"nterm_subtract": nterm_subtract, "rules": rules, "registry": registry}
    offset = parent.start_seq - 1
    steps: list[LadderStep] = []
    covered: set[int] = set()
    previous = None
    for length in lengths:
        label = f"{ion_type}:{length}"
        new = _nitrogens(n, ion_type, length, rules)
        if previous is None:
            res = fragment_amides(parent, label, **kw)
        else:
            new -= _nitrogens(n, ion_type, previous, rules)
            res = amide_difference(parent, label, f"{ion_type}:{previous}", **kw)
        if new:
            span = (min(new) + offset, max(new) + offset)
            note = "" if len(res) == 1 else "block" if res else _reason(
                residues, sorted(new), nterm_subtract, registry)
            steps.append(LadderStep(label, None if previous is None else
                                    f"{ion_type}:{previous}", span, res, note))
            covered |= new
        previous = length
    for p in range(1, n + 1):
        if p not in covered:
            steps.append(LadderStep(None, None, (p + offset, p + offset), (),
                                    _reason(residues, [p], nterm_subtract, registry)
                                    or "no ion"))
    return sorted(steps, key=lambda s: s.span[0])


def _reason(residues, positions, nterm_subtract, registry) -> str:
    """Why the parent ``positions`` (1-based) hold no exchanging amide ('' if one does)."""
    if all(p <= nterm_subtract for p in positions):
        return "N-terminal"
    if all(p <= nterm_subtract or registry.base_residue(residues[p - 1]) == "P"
           for p in positions):
        return "proline"
    return ""


# --- traces and fits ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Trace:
    """Deuterium against time: ``time`` (s), ``mean`` and ``mean_se`` (NaN when unknown)."""

    time: np.ndarray
    mean: np.ndarray
    mean_se: np.ndarray

    def __len__(self) -> int:
        return len(self.time)

    def after(self, min_time: float | None) -> Trace:
        """The points at ``min_time`` and later (all with None)."""
        if min_time is None:
            return self
        keep = self.time >= min_time
        return Trace(self.time[keep], self.mean[keep], self.mean_se[keep])


def fragment_trace(uptake: FragmentUptake, fragment: str, **conditions) -> Trace:
    """The deuteration of one fragment over time (the rows matching ``conditions``, for example
    ``condition='gentle', charge=1, sample='S'``), in time order; replicates at the same time
    are averaged."""
    rows = np.flatnonzero(uptake.where(fragment=fragment, **conditions))
    return _average(uptake.time[rows], uptake.mean[rows], uptake.mean_se[rows])


def _average(t, y, se) -> Trace:
    times = np.unique(np.round(t, 6))
    key = np.round(t, 6)
    mean = np.array([np.mean(y[key == x]) for x in times])
    err = np.array([np.sqrt(np.mean(se[key == x] ** 2)) / np.sqrt(np.sum(key == x))
                    for x in times])
    return Trace(times, mean, err)


def difference_trace(uptake: FragmentUptake, larger: str, smaller: str | None,
                     **conditions) -> Trace:
    """``larger`` minus ``smaller`` over time (``smaller`` None: ``larger`` alone), matched
    scan by scan: the two fragments' spectra of one replicate at the same time come from the
    same scans. Standard errors add in quadrature. ``conditions`` select the rows as in
    :func:`fragment_trace`; give the condition, charge and sample when the table holds several."""
    a = np.flatnonzero(uptake.where(fragment=larger, **conditions))
    if smaller is None:
        return _average(uptake.time[a], uptake.mean[a], uptake.mean_se[a])
    b = np.flatnonzero(uptake.where(fragment=smaller, **conditions))
    other = {(str(uptake.run[j]), str(uptake.sample[j]), str(uptake.condition[j]),
              int(uptake.charge[j]), int(uptake.parent_charge[j]), int(uptake.rep[j]),
              round(float(uptake.time[j]), 6)): j for j in b}
    t, y, se = [], [], []
    for i in a:
        j = other.get((str(uptake.run[i]), str(uptake.sample[i]), str(uptake.condition[i]),
                       int(uptake.charge[i]), int(uptake.parent_charge[i]), int(uptake.rep[i]),
                       round(float(uptake.time[i]), 6)))
        if j is None:
            continue
        t.append(uptake.time[i])
        y.append(uptake.mean[i] - uptake.mean[j])
        se.append(np.hypot(uptake.mean_se[i], uptake.mean_se[j]))
    return _average(np.array(t, float), np.array(y, float), np.array(se, float))


# --- closure: an N-terminal ion and its complement hold the whole precursor ---------------------

_PARTNERS = {"a": "x", "b": "y", "c": "z"}


def _check_partners(n_ion: str, c_ion: str) -> None:
    if chem.ion_terminus(n_ion) != "N" or chem.ion_terminus(c_ion) != "C" or \
            _PARTNERS[n_ion[0]] != c_ion[0]:
        raise ValueError(f"{n_ion} and {c_ion} are not complementary ions (a and x, b and y, "
                         "c and z)")


def closure_trace(
    uptake: FragmentUptake,
    n_fragment: str,
    c_fragment: str,
    *,
    precursor: str = "M",
    n_charge: int = 1,
    c_charge: int = 1,
    precursor_charge: int | None = None,
    **conditions,
) -> Trace:
    """``D(n_fragment) + D(c_fragment) - D(precursor)`` over time, matched scan by scan (the
    closure test). An N-terminal ion of ``k`` residues and the complementary
    C-terminal ion of ``n - k`` (``'c-1:3'`` and ``'z-dot:5'`` of an 8-residue parent) share the
    precursor's atoms, so their deuterium adds up to the precursor's whatever scrambling or
    hydrogen transfer happened; a miss points at the fragments' fits (envelope, ion types,
    interference). ``precursor`` is the ion type of the precursor (its label is ``'M:n'``) at
    ``precursor_charge`` (default: the unreacted precursor, the parent's charge); the fragments
    are taken at ``n_charge`` and ``c_charge``. ``conditions`` select the rows as in
    :func:`fragment_trace` (give the condition when the table holds several). Points at time 0
    (the undeuterated control) are left out."""
    n_ion, _, k = n_fragment.rpartition(":")
    c_ion, _, j = c_fragment.rpartition(":")
    _check_partners(n_ion, c_ion)
    rows = uptake.where(**conditions) & (uptake.time > 0)
    parents = np.unique(uptake.parent[rows])
    if len(parents) > 1:
        raise ValueError(f"the rows hold {len(parents)} parents; choose one (parent=...)")
    if not len(parents):
        return Trace(np.empty(0), np.empty(0), np.empty(0))
    n = len(chem.parse_sequence(str(parents[0])))
    if int(k) + int(j) != n:
        raise ValueError(f"{n_fragment} and {c_fragment} do not make up the {n}-residue parent")

    def key(i):
        return (str(uptake.run[i]), str(uptake.sample[i]), str(uptake.condition[i]),
                int(uptake.parent_charge[i]), int(uptake.rep[i]), round(float(uptake.time[i]), 6))

    def index(label, charge):
        sel = np.flatnonzero(rows & uptake.where(fragment=label, charge=charge))
        return {key(i): i for i in sel}

    c_rows = index(c_fragment, c_charge)
    m_sel = np.flatnonzero(rows & uptake.where(fragment=f"{precursor}:{n}"))
    m_charge = (np.full(len(m_sel), precursor_charge) if precursor_charge is not None
                else uptake.parent_charge[m_sel])
    m_rows = {key(i): i for i, z in zip(m_sel, m_charge, strict=True)
              if int(uptake.charge[i]) == int(z)}
    t, y, se = [], [], []
    for kk, i in index(n_fragment, n_charge).items():
        jc, jm = c_rows.get(kk), m_rows.get(kk)
        if jc is None or jm is None:
            continue
        t.append(uptake.time[i])
        y.append(uptake.mean[i] + uptake.mean[jc] - uptake.mean[jm])
        se.append(np.sqrt(uptake.mean_se[i] ** 2 + uptake.mean_se[jc] ** 2
                          + uptake.mean_se[jm] ** 2))
    return _average(np.array(t, float), np.array(y, float), np.array(se, float))


@dataclass(frozen=True)
class ClosureTable:
    """The closure test of every complementary pair (:func:`closure_table`), one row per run,
    sample, condition, parent, precursor charge and split ``k`` (residues of the N-terminal
    ion): the two fragments, the number of matched scans ``n``, the ``median`` miss (deuterons),
    its ``spread`` (half the interquartile range), the ``mean`` and ``flagged`` (the median
    beyond ``tolerance``). ``run`` and ``merged`` as in :class:`FragmentUptake`."""

    sample: np.ndarray
    condition: np.ndarray
    parent: np.ndarray
    parent_charge: np.ndarray
    k: np.ndarray
    n_fragment: np.ndarray
    c_fragment: np.ndarray
    n: np.ndarray
    median: np.ndarray
    spread: np.ndarray
    mean: np.ndarray
    flagged: np.ndarray
    tolerance: float = 0.25
    run: np.ndarray | None = None
    merged: np.ndarray | None = None

    def __post_init__(self):
        _fill_runs(self)

    def __len__(self) -> int:
        return len(self.k)

    @property
    def state_run(self) -> np.ndarray:
        return _state_run(self)

    def columns(self) -> dict[str, np.ndarray]:
        return _columns(self)


def closure_table(
    uptake: FragmentUptake,
    n_ion: str = "c-1",
    c_ion: str = "z-dot",
    *,
    precursor: str = "M",
    n_charge: int = 1,
    c_charge: int = 1,
    precursor_charge: int | None = None,
    min_time: float | None = None,
    tolerance: float = 0.25,
) -> ClosureTable:
    """The closure test (:func:`closure_trace`) of every split ``k`` with both ions measured,
    for each sample, condition, parent and precursor charge, from ``min_time`` (s; spurious
    early points). A pair whose median miss is beyond ``tolerance`` deuterons is ``flagged``.
    ``n_ion`` and ``c_ion`` are complementary (c and z for ETD and ECD, a and x for UVPD, b and
    y for collisional activation)."""
    _check_partners(n_ion, c_ion)
    cols: dict[str, list] = {f.name: [] for f in fields(ClosureTable) if f.name != "tolerance"}
    groups = sorted({(str(r), str(s), str(c), str(p), int(z), bool(m)) for r, s, c, p, z, m in zip(
        uptake.run, uptake.sample, uptake.condition, uptake.parent, uptake.parent_charge,
        uptake.merged, strict=True)})
    for run, sample, condition, parent, pz, merged in groups:
        n = len(chem.parse_sequence(parent))
        sel = dict(run=run, sample=sample, condition=condition, parent=parent, parent_charge=pz)
        have = set(uptake.fragment[uptake.where(**sel)].tolist())
        for k in range(1, n):
            a, b = f"{n_ion}:{k}", f"{c_ion}:{n - k}"
            if a not in have or b not in have:
                continue
            tr = closure_trace(uptake, a, b, precursor=precursor, n_charge=n_charge,
                               c_charge=c_charge, precursor_charge=precursor_charge,
                               **sel).after(min_time)
            if not len(tr):
                continue
            q1, med, q3 = np.percentile(tr.mean, [25, 50, 75])
            for name, value in (("run", run), ("merged", merged), ("sample", sample),
                                ("condition", condition),
                                ("parent", parent), ("parent_charge", pz), ("k", k),
                                ("n_fragment", a), ("c_fragment", b), ("n", len(tr)),
                                ("median", med), ("spread", (q3 - q1) / 2),
                                ("mean", float(np.mean(tr.mean))),
                                ("flagged", bool(abs(med) > tolerance))):
                cols[name].append(value)
    kinds = {"parent_charge": np.int64, "k": np.int64, "n": np.int64, "median": float,
             "spread": float, "mean": float, "flagged": bool, "run": object, "merged": bool}
    return ClosureTable(**{name: np.array(v, dtype=kinds.get(name, str))
                           for name, v in cols.items()}, tolerance=tolerance)


@dataclass(frozen=True)
class DecayFit:
    """``D(t) = amplitude * exp(-rate * t) + offset`` fitted to a trace: ``rate`` (1/s) with its
    standard error ``rate_se`` (from the fit's covariance, scaled by the residual scatter), the
    ``amplitude`` and ``offset`` (deuterons), the number of points ``n``, ``ok`` (False when
    the fit failed or had too few points) and ``resolved``: whether the time range shows the
    decay (``rate * t_last >= 0.1`` and ``rate * t_first <= 3``); a decay too slow for the
    range fits a straight line as a tiny rate with huge amplitude and offset, and one over
    before the first point has no shape, so their rates mean nothing. ``amplitude_se``: the
    amplitude's standard error, from the same covariance."""

    rate: float
    rate_se: float
    amplitude: float
    offset: float
    n: int
    ok: bool
    resolved: bool = False
    amplitude_se: float = np.nan

    def curve(self, t) -> np.ndarray:
        return self.amplitude * np.exp(-self.rate * np.asarray(t, float)) + self.offset


def fit_decay(trace: Trace, *, min_time: float | None = None) -> DecayFit:
    """Fit ``A exp(-k t) + C`` (a decay for back-exchange, a rise when ``A`` is negative) to the
    points of ``trace`` at ``min_time`` and later, by least squares from several starting rates
    spread over the time range (``0.1`` to ``1000`` decays across it); ``k`` is kept positive.
    The fit is made as ``A' exp(-k (t - t_first)) + C`` (the same curve, conditioned on the
    first fitted point rather than on time 0) and ``A = A' exp(k t_first)`` reported. Times
    here are on the exchange clock (spectrum times: scan time plus the run's ``time_offset``);
    to leave a file's spurious early scans out of everything, set the run's ``usable_from``
    instead (:class:`hxex.io.MzmlRun`, acquisition clock)."""
    from scipy.optimize import curve_fit

    tr = trace.after(min_time)
    ok = np.isfinite(tr.mean) & np.isfinite(tr.time)
    t, y = tr.time[ok], tr.mean[ok]
    if len(t) < 4 or np.ptp(t) <= 0:
        return DecayFit(np.nan, np.nan, np.nan, np.nan, len(t), False)
    t_first, t_last = float(t.min()), float(t.max())
    span = float(np.ptp(t))

    def model(x, a, k, c):
        return a * np.exp(-k * (x - t_first)) + c

    tail = max(1, len(t) // 10)
    order = np.argsort(t)
    c0 = float(np.median(y[order[-tail:]]))
    a0 = float(np.median(y[order[:tail]])) - c0
    best = None
    for k0 in np.geomspace(0.1 / span, 1000.0 / span, 7):
        try:
            p, cov = curve_fit(model, t, y, p0=(a0, k0, c0),
                               bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
                               maxfev=5000)
        except (RuntimeError, ValueError):
            continue
        sse = float(np.sum((model(t, *p) - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, p, cov)
    if best is None:
        return DecayFit(np.nan, np.nan, np.nan, np.nan, len(t), False)
    _, p, cov = best
    rate_se = float(np.sqrt(cov[1, 1])) if np.all(np.isfinite(cov)) else np.nan
    rate = float(p[1])
    with np.errstate(over="ignore", invalid="ignore"):  # an unresolved fast decay's amplitude
        grow = np.exp(rate * t_first)
        amplitude = float(p[0] * grow)
        # amplitude = A' exp(k t_first): its gradient in (A', k, C)
        g = np.array([grow, p[0] * t_first * grow, 0.0])
        var = float(g @ cov @ g) if np.all(np.isfinite(cov)) else np.nan
    amplitude_se = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else np.nan
    resolved = rate * t_last >= 0.1 and rate * t_first <= 3.0
    return DecayFit(rate, rate_se, amplitude, float(p[2]), len(t), True, resolved, amplitude_se)


# --- per-residue values from all fragments ------------------------------------------------------

@dataclass(frozen=True)
class ResidueFit:
    """Per-residue deuteration from fragments: ``residue`` (protein numbering), ``value``
    (deuterons per amide, 0 to 1), ``identifiable`` (whether the fragments determine it on its
    own: its amide is a combination of the fragments; otherwise only sums over blocks are) and
    the ``residual`` of the fit (fragments' deuteration minus the fitted sums, per fragment
    in ``fragments`` order)."""

    residue: np.ndarray
    value: np.ndarray
    identifiable: np.ndarray
    fragments: tuple[str, ...]
    residual: np.ndarray


def fit_residues(
    parent: PeptideEntry,
    values: Mapping[str, float],
    *,
    nterm_subtract: int = 1,
    rules: chem.AmideRules = chem.DEFAULT_AMIDE_RULES,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
) -> ResidueFit:
    """Per-residue deuteration (between 0 and 1) whose sums over each fragment's amides best
    match ``values`` (fragment label -> deuterons, at one time; any ion types, ``'M:n'`` for the
    precursor), by bounded linear least squares. Residues that no combination of the fragments
    isolates are marked not ``identifiable``: their values share the block's sum arbitrarily."""
    from scipy.optimize import lsq_linear

    from ..fragments import fragment_amides

    kw = {"nterm_subtract": nterm_subtract, "rules": rules, "registry": registry}
    labels = tuple(values)
    held = [fragment_amides(parent, label, **kw) for label in labels]
    residues = sorted({r for h in held for r in h})
    if not residues:
        raise ValueError("the fragments hold no exchangeable amide")
    col = {r: c for c, r in enumerate(residues)}
    a = np.zeros((len(labels), len(residues)))
    for row, h in enumerate(held):
        a[row, [col[r] for r in h]] = 1.0
    y = np.array([float(values[label]) for label in labels])
    sol = lsq_linear(a, y, bounds=(0.0, 1.0))
    # a residue is identifiable when its unit vector lies in the row space of the design
    pinv = np.linalg.pinv(a)
    projector = pinv @ a
    identifiable = np.isclose(np.diag(projector), 1.0, atol=1e-8)
    return ResidueFit(np.array(residues), sol.x, identifiable, labels, y - a @ sol.x)


# --- the rates of every ladder step, and the analysis settings ----------------------------------

@dataclass(frozen=True)
class LadderRates:
    """The decay fit (:func:`fit_decay`) of every single-residue ladder step
    (:func:`ladder_rates`), one row per run, sample, parent, precursor charge, condition, ion
    type, fragment charge and step (``run`` and ``merged`` as in :class:`FragmentUptake`):
    the ``residue`` (protein numbering) and its amino acid ``aa``,
    the fragments (``larger`` minus ``smaller``; ``smaller`` empty for the shortest fragment
    alone), ``rate`` and ``rate_se`` (1/s), ``amplitude``, its ``amplitude_se``, and
    ``offset`` (deuterons), the
    number of points ``n`` and whether the decay is ``resolved`` in the time range (an
    unresolved rate means nothing). Only when computed with exchange conditions (otherwise
    None, and not among the columns): ``kint``, the residue's intrinsic rate in the sample's
    solvent (1/s; ``kforw + kback``, :mod:`hxex.kint`, from the parent's sequence; NaN for a
    sample without conditions) and ``protection``, ``kint / rate`` (NaN where the decay is not
    resolved: such a rate means nothing), with ``log10_protection`` and its
    ``log10_protection_se`` (from ``rate_se``; ``kint`` is exact). ``precise``: the decay is
    resolved and ``rate_se / rate`` is at most ``max_rate_error``, the steps
    :func:`ladder_residues` combines."""

    sample: np.ndarray
    parent: np.ndarray
    parent_start: np.ndarray
    parent_charge: np.ndarray
    condition: np.ndarray
    ion_type: np.ndarray
    charge: np.ndarray
    residue: np.ndarray
    aa: np.ndarray
    larger: np.ndarray
    smaller: np.ndarray
    rate: np.ndarray
    rate_se: np.ndarray
    amplitude: np.ndarray
    amplitude_se: np.ndarray
    offset: np.ndarray
    n: np.ndarray
    resolved: np.ndarray
    precise: np.ndarray | None = None
    kint: np.ndarray | None = None
    protection: np.ndarray | None = None
    log10_protection: np.ndarray | None = None
    log10_protection_se: np.ndarray | None = None
    min_time: float | None = None
    max_rate_error: float | None = None
    run: np.ndarray | None = None
    merged: np.ndarray | None = None

    def __post_init__(self):
        _fill_runs(self)
        if self.precise is None:
            object.__setattr__(self, "precise", np.asarray(self.resolved, dtype=bool).copy())

    def __len__(self) -> int:
        return len(self.residue)

    @property
    def state_run(self) -> np.ndarray:
        return _state_run(self)

    def states(self) -> list[tuple[str, str]]:
        return list(dict.fromkeys(zip(self.state_run.tolist(), self.sample.astype(str).tolist(),
                                      strict=True)))

    def columns(self) -> dict[str, np.ndarray]:
        return _columns(self)

    def where(self, **conditions) -> np.ndarray:
        keep = np.ones(len(self), dtype=bool)
        for name, value in conditions.items():
            if value is not None:
                values = value if isinstance(value, (list, tuple, set)) else [value]
                keep &= np.isin(getattr(self, name), list(values))
        return keep


MAX_RATE_ERROR = 0.5  # rate_se / rate of a step ladder_residues uses (log10 se about 0.22)


def fragment_ion_types(uptake: FragmentUptake) -> list[str]:
    """The fragment ion types of the table (not the precursor ions), N-terminal first."""
    ions = {str(i) for i in np.unique(uptake.ion_type) if chem.ion_terminus(str(i)) != "M"}
    return sorted(ions, key=lambda i: (chem.ion_terminus(i) != "N", i))


def ladder_rates(
    uptake: FragmentUptake,
    *,
    ion_types: Sequence[str] | None = None,
    charges: Sequence[int] = (1,),
    min_time: float | None = None,
    conditions: ExchangeConditions | Mapping[str, ExchangeConditions] | None = None,
    max_rate_error: float = MAX_RATE_ERROR,
) -> LadderRates:
    """Fit ``A exp(-k t) + C`` (:func:`fit_decay`, from ``min_time``) to every single-residue
    step (:func:`ladder_steps`) of the ``ion_types`` (default: every fragment ion type of the
    table) at the fragment ``charges``, in every sample, parent, precursor charge and
    condition (a merged sample's runs separately). The shortest fragment counts as a step when
    it holds one amide. With the exchange ``conditions`` (the solvent of the kinetics: one set
    for every sample, or by sample with per-run entries, :func:`hxex.analysis.sample_setting`),
    each step also gets its residue's intrinsic rate and protection factor (NaN for a sample
    without conditions); without them there are no such columns. ``precise`` marks the
    resolved steps whose ``rate_se / rate`` is at most ``max_rate_error``."""
    if not max_rate_error > 0:
        raise ValueError("max_rate_error must be positive")
    from ..kint import intrinsic_rates
    from .runs import sample_setting

    ion_types = list(ion_types) if ion_types is not None else fragment_ion_types(uptake)
    cols: dict[str, list] = {f.name: [] for f in fields(LadderRates)
                             if f.name not in ("min_time", "max_rate_error")}
    state_run = uptake.state_run
    groups = sorted({(str(u), str(s), str(p), int(z)) for u, s, p, z in zip(
        state_run, uptake.sample, uptake.parent, uptake.parent_charge, strict=True)})
    for state, sample, parent, pz in groups:
        rows = uptake.where(sample=sample, parent=parent, parent_charge=pz) & (state_run == state)
        entry = uptake.parent_entry(int(np.flatnonzero(rows)[0]))
        residues = chem.parse_sequence(parent)
        solvent = sample_setting(conditions, state, sample)
        kint = (intrinsic_rates(parent, solvent, start=entry.start_seq)
                if solvent is not None else None)
        merged = bool(uptake.merged[rows][0])
        scan_conditions = sorted({(str(r), str(c)) for r, c in zip(
            uptake.run[rows], uptake.condition[rows], strict=True)})
        for ion in ion_types:
            for z in charges:
                series = rows & uptake.where(ion_type=ion, charge=z)
                if not series.any():
                    continue
                lengths = sorted(set(uptake.length[series].tolist()))
                for step in ladder_steps(entry, ion, lengths,
                                         nterm_subtract=uptake.nterm_subtract,
                                         rules=uptake.amide_rules):
                    if not step.single:
                        continue
                    (residue,) = step.residues
                    k_int = (float(kint.rate[kint.at([residue])[0]]) if kint is not None
                             else np.nan)
                    for run, cond in scan_conditions:
                        tr = difference_trace(uptake, step.larger, step.smaller, run=run,
                                              sample=sample, parent=parent, parent_charge=pz,
                                              condition=cond, charge=z).after(1e-9)
                        if not len(tr):
                            continue
                        f = fit_decay(tr, min_time=min_time)
                        relative = f.rate_se / f.rate if f.rate > 0 else np.nan
                        protection = (k_int / f.rate if f.resolved and f.rate > 0 else np.nan)
                        for name, value in (
                                ("run", run), ("merged", merged),
                                ("sample", sample), ("parent", parent),
                                ("parent_start", entry.start_seq), ("parent_charge", pz),
                                ("condition", cond), ("ion_type", ion), ("charge", z),
                                ("residue", residue),
                                ("aa", residues[residue - entry.start_seq]),
                                ("larger", step.larger), ("smaller", step.smaller or ""),
                                ("rate", f.rate), ("rate_se", f.rate_se),
                                ("amplitude", f.amplitude), ("amplitude_se", f.amplitude_se),
                                ("offset", f.offset), ("n", f.n),
                                ("resolved", f.resolved),
                                ("precise", bool(f.resolved and relative <= max_rate_error)),
                                ("kint", k_int), ("protection", protection),
                                ("log10_protection", np.log10(protection)),
                                ("log10_protection_se", relative / np.log(10)
                                 if np.isfinite(protection) else np.nan)):
                            cols[name].append(value)
    kinds = {"parent_start": np.int64, "parent_charge": np.int64, "charge": np.int64,
             "residue": np.int64, "rate": float, "rate_se": float, "amplitude": float,
             "amplitude_se": float, "offset": float, "n": np.int64, "resolved": bool,
             "precise": bool, "kint": float,
             "protection": float, "log10_protection": float, "log10_protection_se": float,
             "run": object, "merged": bool}
    if conditions is None:
        for name in ("kint", "protection", "log10_protection", "log10_protection_se"):
            del cols[name]
    else:
        from .runs import label as _label

        missing = sorted({(u, s) for u, s, *_ in groups
                          if sample_setting(conditions, u, s) is None})
        if missing:
            log.warning("no exchange conditions for sample(s) %s ([analysis.fragments.exchange]):"
                        " no intrinsic rates for them", ", ".join(_label(*m) for m in missing))
    with np.errstate(divide="ignore", invalid="ignore"):  # log10 of NaN protection factors
        arrays = {name: np.array(v, dtype=kinds.get(name, str)) for name, v in cols.items()}
    return LadderRates(**arrays, min_time=min_time, max_rate_error=max_rate_error)


@dataclass(frozen=True)
class LadderResidues:
    """One rate per residue from the precise ladder steps (:attr:`LadderRates.precise`), the
    N-terminal ladder's estimate combined with the C-terminal ladder's: one row per run,
    sample, parent, precursor charge, condition, fragment ``charge``, residue and pair of ion
    types (``n_ion``, ``c_ion``; blank where that side has no precise step). Different
    conditions (gentle and harsh), precursor charges, fragment charges and runs are never
    combined, and neither are ion types of one terminus (c and c-1 are read from the same
    peaks, not independent): a residue with c, c-1 and z steps has the rows (c, z) and (c-1,
    z), alternatives, not to be averaged. ``n`` estimates (1 or 2), ``log10_rate`` (their mean
    weighted by ``1 / se**2`` in log10, each with ``rate_se / (rate ln 10)``) and
    ``log10_rate_se`` (scaled by ``sqrt(misfit)`` when the misfit is above 1), ``rate``,
    ``misfit`` (chi-square of the two about their mean; NaN for one) and ``agreement``:
    ``'agree'`` (within its limit at ``alpha``), ``'disagree'`` (the two ladders give different
    rates: the mean is then a poor summary) or ``'single'``. With exchange conditions (as
    :class:`LadderRates`): ``kint``, ``log10_protection`` (``log10(kint) - log10_rate``), its
    se and ``protection``."""

    sample: np.ndarray
    parent: np.ndarray
    parent_start: np.ndarray
    parent_charge: np.ndarray
    condition: np.ndarray
    charge: np.ndarray
    residue: np.ndarray
    aa: np.ndarray
    n_ion: np.ndarray
    c_ion: np.ndarray
    n: np.ndarray
    log10_rate: np.ndarray
    log10_rate_se: np.ndarray
    rate: np.ndarray
    misfit: np.ndarray
    agreement: np.ndarray
    kint: np.ndarray | None = None
    log10_protection: np.ndarray | None = None
    log10_protection_se: np.ndarray | None = None
    protection: np.ndarray | None = None
    run: np.ndarray | None = None
    merged: np.ndarray | None = None

    def __post_init__(self):
        _fill_runs(self)

    def __len__(self) -> int:
        return len(self.residue)

    def columns(self) -> dict[str, np.ndarray]:
        return _columns(self)

    def where(self, **conditions) -> np.ndarray:
        keep = np.ones(len(self), dtype=bool)
        for name, value in conditions.items():
            if value is not None:
                values = value if isinstance(value, (list, tuple, set)) else [value]
                keep &= np.isin(getattr(self, name), list(values))
        return keep


def ladder_residues(rates: LadderRates, *, alpha: float = 0.05) -> LadderResidues:
    """Combine the precise steps of :func:`ladder_rates` per residue, the N-terminal ladder
    with the C-terminal one (see :class:`LadderResidues`); residues without a precise step
    have no row."""
    from itertools import product

    from scipy import stats

    with_kint = rates.kint is not None
    keys = ("run", "merged", "sample", "parent", "parent_start", "parent_charge", "condition",
            "charge", "residue", "aa")
    groups: dict[tuple, dict[str, dict[str, int]]] = {}
    for i in np.flatnonzero(rates.precise):
        key = tuple(np.asarray(getattr(rates, k))[i] for k in keys)
        key = tuple(v.item() if isinstance(v, np.generic) else v for v in key)
        side = "N" if chem.ion_terminus(str(rates.ion_type[i])) == "N" else "C"
        groups.setdefault(key, {"N": {}, "C": {}})[side][str(rates.ion_type[i])] = int(i)
    cols: dict[str, list] = {f.name: [] for f in fields(LadderResidues)}
    for key in sorted(groups, key=lambda k: tuple(map(str, k))):
        sides = groups[key]
        for n_ion, c_ion in product(sorted(sides["N"]) or [""], sorted(sides["C"]) or [""]):
            rows = [sides["N"][n_ion]] if n_ion else []
            rows += [sides["C"][c_ion]] if c_ion else []
            rate, rate_se = rates.rate[rows], rates.rate_se[rows]
            x = np.log10(rate)
            w = 1.0 / (rate_se / (rate * np.log(10))) ** 2
            mean = float(np.sum(w * x) / np.sum(w))
            mean_se = float(np.sqrt(1.0 / np.sum(w)))
            n = len(rows)
            chi2 = float(np.sum(w * (x - mean) ** 2))
            misfit = chi2 / (n - 1) if n > 1 else np.nan
            if n > 1:
                mean_se *= float(np.sqrt(max(1.0, misfit)))
                agreement = "agree" if chi2 <= stats.chi2.ppf(1 - alpha, n - 1) else "disagree"
            else:
                agreement = "single"
            for k, v in zip(keys, key, strict=True):
                cols[k].append(v)
            for name, v in (("n_ion", n_ion), ("c_ion", c_ion), ("n", n),
                            ("log10_rate", mean), ("log10_rate_se", mean_se),
                            ("rate", 10.0**mean), ("misfit", misfit), ("agreement", agreement)):
                cols[name].append(v)
            if with_kint:
                k_int = float(rates.kint[rows[0]])
                lp = np.log10(k_int) - mean if k_int > 0 else np.nan
                for name, v in (("kint", k_int), ("log10_protection", lp),
                                ("log10_protection_se", mean_se if np.isfinite(lp) else np.nan),
                                ("protection", 10.0**lp)):
                    cols[name].append(v)
    kinds = {"parent_start": np.int64, "parent_charge": np.int64, "charge": np.int64,
             "residue": np.int64, "n": np.int64, "log10_rate": float, "log10_rate_se": float,
             "rate": float, "misfit": float, "kint": float, "log10_protection": float,
             "log10_protection_se": float, "protection": float, "run": object, "merged": bool}
    if not with_kint:
        for name in ("kint", "log10_protection", "log10_protection_se", "protection"):
            del cols[name]
    return LadderResidues(**{name: np.array(v, dtype=kinds.get(name, str))
                             for name, v in cols.items()})


@dataclass(frozen=True)
class FragmentAnalysis:
    """The fragment part of a post-fitting analysis (the ``[analysis.fragments]`` table; written
    by :func:`hxex.report.write_analysis` for runs with fragment series).

    - ``ion_types``: the fragment ion types analysed (empty: every one of the run).
    - ``charges``: the fragment charges analysed.
    - ``min_time`` (s, on the exchange clock, the spectra's times): earlier points are drawn but
      not fitted; a file's unusable early scans are better left out with its ``usable_from``.
      (The ``min_time`` of ``[input.scans]`` is on the scan clock.)
    - ``labels``: short condition names for the figures (``{'gentle': 'G'}``; a run's own
      under its name, ``{'etd': {'gentle': 'ETD G'}}``, for a merged sample's runs).
    - ``tolerance`` (deuterons): a closure ``c + z - M`` beyond it is flagged.
    - ``precursor_charge``: the closure's precursor; None: the unreacted precursor.
    - ``max_rate_error``: a ladder step's ``rate_se / rate`` at most this is ``precise`` and
      enters the per-residue rates (:func:`ladder_residues`, ``ladder_residues.csv``: the N-
      and C-terminal ladders combined, fragment charges and conditions kept apart).
    - ``kint``: compute intrinsic rates, protection factors and theoretical curves (opt-in;
      needs ``exchange``); ``exchange``: the samples' exchange conditions
      (:class:`hxex.kint.ExchangeConditions`, one set for every sample, or by sample with a
      run's own under its name, ``[analysis.fragments.exchange.<run>.<sample>]``), whose
      ``d2o_fraction`` is the kinetics' solvent, not the labelling buffer's.
    """

    ion_types: tuple[str, ...] = ()
    charges: tuple[int, ...] = (1,)
    min_time: float | None = None
    labels: Mapping[str, str] | None = None
    tolerance: float = 0.25
    precursor_charge: int | None = None
    max_rate_error: float = MAX_RATE_ERROR
    kint: bool = False
    exchange: ExchangeConditions | Mapping[str, ExchangeConditions] | None = None

    def __post_init__(self):
        from ..kint import read_conditions

        object.__setattr__(self, "exchange", read_conditions(self.exchange))
        if not isinstance(self.kint, bool):
            raise ValueError("kint must be true or false")
        if self.kint and not self.exchange:
            raise ValueError("kint needs the samples' exchange conditions (exchange: ph_read, "
                             "temperature, d2o_fraction)")
        object.__setattr__(self, "ion_types", tuple(str(i) for i in self.ion_types))
        for ion in self.ion_types:
            if chem.ion_terminus(ion) == "M":
                raise ValueError(f"{ion!r} is not a fragment ion type")
        object.__setattr__(self, "charges", tuple(int(z) for z in self.charges))
        if not self.charges or min(self.charges) < 1:
            raise ValueError("charges must be one or more positive charges")
        if self.labels is not None:
            object.__setattr__(self, "labels", {
                str(k): ({str(c): str(n) for c, n in v.items()} if isinstance(v, Mapping)
                         else str(v)) for k, v in self.labels.items()})
        if not self.tolerance > 0:
            raise ValueError("tolerance must be positive")
        object.__setattr__(self, "max_rate_error", float(self.max_rate_error))
        if not self.max_rate_error > 0:
            raise ValueError("max_rate_error must be positive")

    def to_dict(self) -> dict:
        return {"ion_types": list(self.ion_types), "charges": list(self.charges),
                "min_time": self.min_time, "labels": dict(self.labels or {}) or None,
                "tolerance": self.tolerance, "precursor_charge": self.precursor_charge,
                "max_rate_error": self.max_rate_error, "kint": self.kint,
                "exchange": conditions_dict(self.exchange)}

    @classmethod
    def from_dict(cls, data: Mapping) -> FragmentAnalysis:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown fragment analysis setting(s): {', '.join(sorted(unknown))}")
        return cls(**dict(data))
