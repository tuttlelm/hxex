# SPDX-License-Identifier: GPL-3.0-or-later
"""Peptides another protein in the same vial also has (docs/ANALYSIS.md "Groups of samples").

With ``[analysis.groups]``, a vial (a label shared by several groups) holds several proteins,
each exported as a sample of its own group. A peptide measured in one of them whose sequence
also occurs in another protein of the vial is the sum of both proteins' copies there, not like
with like against its group's reference: :func:`shared_peptides` finds them, the analysis marks
them (``shared_with``), and ``[analysis] shared_sequences = "leave_out"`` leaves their spectra
out (:data:`hxex.analysis.species.SHARED_REASON`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from .. import chem

log = logging.getLogger(__name__)

SHARED_SEQUENCES = ("keep", "leave_out")  # [analysis] shared_sequences


@dataclass(frozen=True)
class SharedPeptides:
    """What :func:`shared_peptides` found for one group's analysis.

    - ``shared``: ``(run, sample, peptide)`` -> the other groups whose protein in the same vial
      also has the peptide's sequence (modifications ignored).
    - ``unknown``: ``(sample, other group, other sample)`` of each partner in a vial whose
      protein sequence is not known (no ``[analysis.fasta]`` entry and none stored with the
      results), so its peptides could not be checked against it.
    """

    shared: Mapping[tuple[str, str, str], tuple[str, ...]] = field(default_factory=dict)
    unknown: tuple[tuple[str, str, str], ...] = ()

    def groups_of(self, run: str, sample: str, peptide: str) -> str:
        """The other groups, by name and comma separated ('' for a peptide not shared)."""
        return ", ".join(self.shared.get((str(run), str(sample), str(peptide)), ()))

    def left_out(self) -> frozenset[tuple[str, str]]:
        """``(sample, peptide)`` of every shared peptide (for
        :attr:`hxex.analysis.SpeciesFilters.shared`)."""
        return frozenset((s, p) for (_, s, p) in self.shared)


def vial_groups(label: str) -> list[str]:
    """The groups a vial label names: its parts split at ``+`` (``"A+B"`` -> A and B). Two
    groups' samples of one label are one vial only when the label names both (owner,
    2026-11-18): ``"alone"`` in every group names none, so it is each group's own vial."""
    return [part.strip() for part in str(label).split("+") if part.strip()]


def _letters(sequence: str, registry) -> str:
    """The base residues of ``sequence`` as one string (``pS`` reads as ``S``)."""
    return "".join(registry.base_residue(label) for label in chem.parse_sequence(sequence,
                                                                                  registry))


def _stored_sequence(results, run: str, sample: str) -> str | None:
    """The protein sequence the run's reader stored for ``sample`` (a SpecExport's FASTA)."""
    from .runs import RunSet

    r = results.runs.get(run) if isinstance(results, RunSet) else results
    settings = getattr(r, "settings", None)
    source = (getattr(settings, "spectra_source", None) or {}) if settings is not None else {}
    entry = (source.get("sequences") or {}).get(sample)
    return str(entry["sequence"]) if isinstance(entry, Mapping) and entry.get("sequence") \
        else None


def shared_peptides(results, config, sequences=None) -> SharedPeptides:
    """The peptides of the analysis ``config`` (one group's, :meth:`hxex.analysis.AnalysisConfig.
    with_group`) that another group's protein in the same vial also has: for each of the
    group's samples, the other groups' samples with the same label where the label names both
    groups (:func:`vial_groups`: ``"A+B"`` is one vial in groups A and B), their protein sequence
    from ``sequences`` (``config.sequences(base)``: ``[analysis.fasta]``) or else the one the
    results stored, and every peptide the sample measured whose residues occur in it. Empty
    for an analysis that is not a group's."""
    from .runs import RunSet, resolve_sample, sample_setting

    if not config.group or not config.groups:
        return SharedPeptides()
    group = config.groups[config.group]
    runs = results.runs if isinstance(results, RunSet) else {"": results}
    first = next(iter(runs.values()))
    registry = getattr(getattr(first, "settings", None), "registry", None) \
        or chem.DEFAULT_MODIFICATIONS
    states = []
    for name, r in runs.items():
        if r.binned is None:
            return SharedPeptides()
        for sample in dict.fromkeys(str(s) for s in r.binned.sample):
            states.append((name, sample))
    run_names, sample_names = [s[0] for s in states], [s[1] for s in states]

    def state(name: str) -> tuple[str, str] | None:
        try:
            return resolve_sample(run_names, sample_names, name)
        except ValueError:
            return None

    def peptides(run: str, sample: str) -> list[str]:
        binned = runs[run].binned
        return list(dict.fromkeys(str(p) for s, p in zip(binned.sample, binned.peptide,
                                                         strict=True) if str(s) == sample))

    shared: dict[tuple[str, str, str], list[str]] = {}
    unknown = []
    cache: dict[tuple[str, str], str | None] = {}
    for sample, label in zip(group.samples, group.labels, strict=True):
        own = state(sample)
        if own is None:
            continue
        for other_name, other in config.groups.items():
            if other_name == config.group or label not in other.labels \
                    or not {config.group, other_name} <= set(vial_groups(label)):
                continue
            partner = other.samples[other.labels.index(label)]
            where = state(partner)
            if where is None or where == own:
                continue
            if where not in cache:
                seq = sample_setting(sequences, *where)
                seq = str(seq) if seq is not None else _stored_sequence(results, *where)
                cache[where] = _letters(seq, registry) if seq else None
            protein = cache[where]
            if protein is None:
                unknown.append((sample, other_name, partner))
                continue
            for peptide in peptides(*own):
                if _letters(peptide, registry) in protein:
                    names = shared.setdefault((own[0], own[1], peptide), [])
                    if other_name not in names:
                        names.append(other_name)
    return SharedPeptides({k: tuple(v) for k, v in shared.items()}, tuple(unknown))
