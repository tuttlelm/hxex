# SPDX-License-Identifier: GPL-3.0-or-later
"""The forms a pooled sample must hold: the fewest candidates of the pool FASTA that explain
every peptide measured (protein inference by parsimony; owner, 2026-11-09), each with the
peptides it may have given, and a residue map per form.

A pooled sample (``[input] format = "pooled"`` with ``pooled_fasta``) holds peptides that may
come from several of the FASTA's records (:func:`hxex.io.identify`). A record with a peptide no
other record holds must be in the mix; the fewest records that, with those, hold every peptide
are the *minimum set*. Records whose peptides are the same (none of the measured peptides tells
them apart) are one form, named ``'A or B'``. A modified form (a peptide with a modified
residue, such as ``acK40``) is a form of its own, on the form of the set it fits
(``'ALPHA1 acK40'`` style names: the base form's name, then the sites); its map takes the base
form's unmodified peptides that do not cover a modified site, and the modified peptides.

A form's residue map uses every peptide it may have given. A peptide that another form of the
set may also have given measures their mixture: its value is the same in both maps, so each
form's residues are marked *specific* (a covering peptide only this form may have given) or
*shared*.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .. import chem
from .residues import modified_state

log = logging.getLogger(__name__)

EXACT_LIMIT = 20  # more undecided candidate groups than this: the set is found greedily

FORM_COLUMNS = ("run", "form", "members", "candidates", "base", "sites", "n_peptides",
                "n_own", "own_peptides", "shared_peptides", "minimum_set_unique")


@dataclass(frozen=True)
class PooledForm:
    """One form of the minimum set. ``name``: the members joined by ``' or '``, then the
    modified sites for a modified form; ``members``: the candidate record ids it may be (in
    FASTA order); ``sites``: the modified sites (:func:`hxex.analysis.modified_state`), ''
    for an unmodified form; ``base``: the unmodified form a modified one is on ('' for an
    unmodified form); ``peptides``: the ``(peptide, start, end)`` it may have given;
    ``own``: those no other form of the set may have given; ``candidates``: for a modified
    form, every record its modified peptides fit (the set's reading names the members only).
    """

    name: str
    members: tuple[str, ...]
    sites: str
    base: str
    peptides: tuple[tuple[str, int, int], ...]
    own: tuple[tuple[str, int, int], ...]
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class PooledForms:
    """The minimum set of one run's pooled samples (every sample of the run together: they are
    the same pool under different conditions). ``alternatives``: the other minimum sets of
    unmodified forms when the set is not unique (each a tuple of form names), empty when it
    is; ``exact``: False when the set was found greedily (:data:`EXACT_LIMIT`)."""

    run: str
    forms: tuple[PooledForm, ...]
    alternatives: tuple[tuple[str, ...], ...] = ()
    exact: bool = True

    def form(self, name: str) -> PooledForm:
        for f in self.forms:
            if f.name == name:
                return f
        raise KeyError(name)

    def columns(self) -> dict[str, np.ndarray]:
        def text(peptides):
            return ";".join(f"{s}-{e} {p}" for p, s, e in peptides)

        rows = [(self.run, f.name, ";".join(f.members), ";".join(f.candidates or f.members),
                 f.base, f.sites, len(f.peptides), len(f.own), text(f.own),
                 text([p for p in f.peptides if p not in set(f.own)]),
                 not self.alternatives) for f in self.forms]
        numbers = {"n_peptides": np.int64, "n_own": np.int64, "minimum_set_unique": bool}
        return {name: np.array([r[i] for r in rows], dtype=numbers.get(name, object))
                for i, name in enumerate(FORM_COLUMNS)}


def minimum_set(candidates: Mapping[object, Sequence[str]], order: Sequence[str] = ()
                ) -> tuple[list[tuple[str, ...]], list[list[tuple[str, ...]]], bool]:
    """The fewest groups of candidates that hold every key of ``candidates`` (a peptide ->
    the record ids that hold it; keys with no candidate are ignored). Candidates holding the
    same keys are one group (in ``order``, else by name). Returns the groups of one minimum
    set (required groups first: those that alone hold a key), the other minimum sets, and
    whether the search was exact."""
    rank = {c: i for i, c in enumerate(order)}

    def ordered(names):
        return tuple(sorted(names, key=lambda c: (rank.get(c, len(rank)), c)))

    held: dict[str, set] = {}
    for key, cands in candidates.items():
        for c in cands:
            held.setdefault(c, set()).add(key)
    by_keys: dict[frozenset, list[str]] = {}
    for c, keys in held.items():
        by_keys.setdefault(frozenset(keys), []).append(c)
    groups = {ordered(cs): keys for keys, cs in by_keys.items()}
    covers: dict[object, list[tuple[str, ...]]] = {}
    for g, keys in groups.items():
        for k in keys:
            covers.setdefault(k, []).append(g)
    required = sorted({gs[0] for gs in covers.values() if len(gs) == 1},
                      key=lambda g: (rank.get(g[0], len(rank)), g))
    left = set(covers) - set().union(*(groups[g] for g in required)) if required else set(covers)
    others = sorted((g for g in groups if g not in required and groups[g] & left),
                    key=lambda g: (rank.get(g[0], len(rank)), g))
    if not left:
        return required, [], True
    if len(others) > EXACT_LIMIT:  # greedy: the group holding most of what is left, each time
        chosen, rest = [], set(left)
        while rest:
            best = max(others, key=lambda g: len(groups[g] & rest))
            chosen.append(best)
            rest -= groups[best]
        return required + chosen, [], False
    for size in range(1, len(others) + 1):
        found = [list(combo) for combo in itertools.combinations(others, size)
                 if left <= set().union(*(groups[g] for g in combo))]
        if found:
            return required + found[0], [required + f for f in found[1:]], True
    raise AssertionError("every key has a candidate")  # pragma: no cover


def pooled_forms(peptides: Sequence[tuple[str, int, int, str]], record: dict, *, run: str = "",
                 registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS
                 ) -> PooledForms:
    """The minimum set of the pooled peptides ``(peptide, start, end, modification)`` (every
    sample of one run together) against the candidates ``record``
    (:func:`hxex.io.candidates_record`). Peptides no candidate holds are left out (they are
    named ``none`` in the identity columns)."""
    from ..io.pooled import identify

    peptides = list(dict.fromkeys(peptides))
    names = [name for name, _ in record["records"]]
    found: dict[tuple[str, int, int], tuple[str, ...]] = {}
    sites_of: dict[tuple[str, int, int], str] = {}
    for p, s, e, m in peptides:
        ident = identify(p, s, record, registry=registry)
        if ident.candidates:
            found[(p, s, e)] = ident.candidates
            sites_of[(p, s, e)] = modified_state(p, s, m, registry)
    # every peptide, modified or not, is evidence that its candidates are in the mix
    groups, alternatives, exact = minimum_set(found, names)
    plain = []
    for g in groups:
        keep = tuple(k for k, cands in found.items() if not sites_of[k] and set(cands) & set(g))
        plain.append(PooledForm(" or ".join(g), g, "", "", keep, ()))
    modified = []
    for sites in dict.fromkeys(v for v in sites_of.values() if v):
        mine = [k for k, v in sites_of.items() if v == sites]
        cands = set.intersection(*(set(found[k]) for k in mine))
        bases = [f for f in plain if set(f.members) & cands] or [
            f for f in plain if set(f.members) & set().union(*(set(found[k]) for k in mine))]
        members = tuple(c for f in bases for c in f.members if c in cands) or tuple(
            c for f in bases for c in f.members)
        base_name = " or ".join(f.name for f in bases)
        site_residues = _site_numbers(mine, registry)
        kept = [k for k in (bases[0].peptides if len(bases) == 1 else
                            set.intersection(*(set(f.peptides) for f in bases)))
                if not site_residues & set(range(k[1], k[2] + 1))]
        modified.append(PooledForm(f"{base_name} {sites}", members, sites, base_name,
                                   tuple(sorted(kept, key=lambda k: (k[1], k[2], k[0])))
                                   + tuple(sorted(mine, key=lambda k: (k[1], k[2], k[0]))), (),
                                   tuple(c for c in names if c in cands)))
    forms = plain + modified
    holders: dict[tuple[str, int, int], int] = {}
    for f in forms:
        for k in f.peptides:
            holders[k] = holders.get(k, 0) + 1
    forms = [dataclasses.replace(f, own=tuple(k for k in f.peptides if holders[k] == 1))
             for f in forms]
    unexplained = set(found) - {k for f in forms for k in f.peptides}
    if unexplained:  # pragma: no cover - every peptide's candidates meet a group
        log.warning("%d pooled peptides fit no form of the minimum set", len(unexplained))
    return PooledForms(run, tuple(forms),
                       tuple(tuple(" or ".join(g) for g in alt) for alt in alternatives),
                       exact)


def _site_numbers(peptides, registry) -> set[int]:
    """The residue numbers of the modified residues of ``peptides`` ``(peptide, start, end)``."""
    out = set()
    for p, s, _ in peptides:
        for i, label in enumerate(chem.parse_sequence(p, registry)):
            if registry.base_residue(label) != label:
                out.add(s + i)
    return out


def form_class(form: PooledForm, record: dict) -> str:
    """The class every member of ``form`` falls in (``[input] pooled_classes``), '' when they
    fall in none or in several."""
    from ..io.pooled import record_classes

    found = set.intersection(*(record_classes(record, m) for m in form.members))
    return next(iter(found)) if len(found) == 1 else ""


def form_sequences(forms: PooledForms, record: dict, states: Mapping[tuple[str, str], str],
                   registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS
                   ) -> dict[str, str]:
    """The protein sequence of each form state ``(run, '<sample> <form>')`` of ``states`` (->
    its sample), keyed by the state's name: the record of a form with one member, when every
    peptide of the form lies on it at its own residue numbers; none otherwise (the residues
    are then known only where a peptide covers them)."""
    from .residues import SequenceMismatch, check_sequence, protein_letters

    seqs = dict(record["records"])
    letters: dict[str, list[str] | None] = {}
    for f in forms.forms:
        if len(f.members) != 1:
            letters[f.name] = None
            continue
        known = protein_letters(seqs[f.members[0]], registry)
        try:
            for p, s, _ in f.peptides:
                check_sequence(p, s, known, registry)
        except SequenceMismatch:
            log.info("%s: its peptides are not all at their own residue numbers on %s; its "
                     "residues are known only where a peptide covers them", f.name, f.members[0])
            known = None
        letters[f.name] = known
    out = {}
    for (run, state), sample in states.items():
        if run != forms.run:
            continue
        name = state[len(sample) + 1:]
        if letters.get(name) is not None:
            out[state] = seqs[forms.form(name).members[0]]
    return out


def _form_rows(table, forms: Mapping[str, PooledForms]):
    """The rows of ``table`` (``run``, ``sample``, ``peptide``, ``start``, ``end`` columns),
    each repeated for every form of its run's minimum set that may have given its peptide,
    with the new state names and their samples."""
    idx, labels, base = [], [], {}
    for i, (u, s, p, a, e) in enumerate(zip(table.run, table.sample, table.peptide,
                                            table.start, table.end, strict=True)):
        pf = forms.get(str(u))
        if pf is None:
            continue
        key = (str(p), int(a), int(e))
        for f in pf.forms:
            if key in f.peptides:
                idx.append(i)
                label = f"{s} {f.name}"
                labels.append(label)
                base[(str(u), label)] = str(s)
    return np.asarray(idx, dtype=int), np.array(labels, dtype=str), base


def form_uptake(uptake, forms: Mapping[str, PooledForms]):
    """``uptake`` (a :class:`hxex.analysis.SpeciesUptake`, peptides) with every peptide row
    repeated for each form of its run's minimum set (``forms``: run -> :class:`PooledForms`)
    that may have given it, its sample named ``'<sample> <form>'``; rows of peptides no form
    holds are left out. Returns the table and the sample of every new state ``(run,
    state)``."""
    idx, labels, base = _form_rows(uptake, forms)
    out = uptake.select(idx)
    return dataclasses.replace(out, sample=labels), base


def form_comparison(comparison, forms: Mapping[str, PooledForms]):
    """A peptide comparison (:class:`hxex.analysis.PeptideComparison`) with its rows repeated
    for each form that may have given the peptide, the sample named as :func:`form_uptake`
    names it, so a form's residue comparison is gated by its own peptides
    (:func:`hxex.analysis.gated_by_peptides`)."""
    idx, labels, _ = _form_rows(comparison, forms)
    picked = dataclasses.replace(comparison, **{
        f.name: getattr(comparison, f.name)[idx] for f in dataclasses.fields(comparison)
        if isinstance(getattr(comparison, f.name), np.ndarray)
        and len(getattr(comparison, f.name)) == len(comparison.sample)})
    return dataclasses.replace(picked, sample=labels)
