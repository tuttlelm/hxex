# SPDX-License-Identifier: GPL-3.0-or-later
"""Spectra left out by rules of the project file (docs/PLAN.md, "Excluding spectra").

A rule (:class:`Exclusion`, one ``[[input.exclude]]`` or ``[[review.exclude]]`` table) names
any of a spectrum's sample, peptide (by its residues), charge, time, control role, replicate,
data file and id, and a note; a spectrum is excluded when it matches every key of one rule.
Before fitting (``[[input.exclude]]``, :func:`hxex.run.run`) the matched spectra are stored but
not fitted; after it (``[[review.exclude]]``, :func:`exclude_from_results`) they are treated
as rejected, in memory, by every consumer of the review layer. Nothing is deleted.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .io.entries import parse_range
from .io.times import format_time, parse_time

log = logging.getLogger(__name__)

ROLES = ("undeut", "fulldeut")
_SCAN_READERS = ("mzml", "thermo")


@dataclass(frozen=True)
class Exclusion:
    """One rule: the spectra matching every key given (None: any). ``peptide_range`` is
    ``(start, end)`` residues; ``time`` in seconds (a time point; a control has none, so a rule
    with a time never matches one); ``file`` the name of the data file (see
    :func:`data_file_names`); ``note`` says why."""

    sample: str | None = None
    peptide_range: tuple[int, int] | None = None
    charge: int | None = None
    time: float | None = None
    role: str | None = None
    rep: int | None = None
    file: str | None = None
    spectrum_id: str | None = None
    note: str = ""

    KEYS = ("sample", "peptide_range", "charge", "time", "role", "rep", "file", "spectrum_id")

    def __post_init__(self):
        if all(getattr(self, k) is None for k in self.KEYS):
            raise ValueError("a rule must name at least one of " + ", ".join(self.KEYS))
        if self.role is not None and self.role not in ROLES:
            raise ValueError(f"role must be one of {', '.join(ROLES)}, got {self.role!r}")

    def describe(self) -> str:
        """The rule in words, as written (``sample WT, peptide 45-60, charge 3``)."""
        parts = []
        for key in self.KEYS:
            value = getattr(self, key)
            if value is None:
                continue
            if key == "peptide_range":
                value = f"{value[0]}-{value[1]}"
            elif key == "time":
                value = format_time(value)
            parts.append(f"{key.replace('_', ' ')} {value}")
        return ", ".join(parts) + (f" ({self.note})" if self.note else "")

    def to_dict(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in self.KEYS if getattr(self, k) is not None}
        if "peptide_range" in out:
            out["peptide_range"] = f"{out['peptide_range'][0]}-{out['peptide_range'][1]}"
        if self.note:
            out["note"] = self.note
        return out

    @classmethod
    def from_dict(cls, data: Mapping) -> Exclusion:
        """A rule from a project-file table (or :meth:`to_dict`); unknown keys and values of
        the wrong form raise :class:`ValueError` naming them."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown key(s) {', '.join(unknown)} (a rule takes "
                             f"{', '.join(cls.KEYS)} and note)")
        values: dict[str, Any] = {}
        for key, value in data.items():
            if value is None or value is False and key != "note":
                continue
            if key in ("sample", "file", "spectrum_id", "role", "note"):
                if not isinstance(value, str):
                    raise ValueError(f"{key} must be text, got {value!r}")
                values[key] = value
            elif key in ("charge", "rep"):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{key} must be a whole number of 1 or more, got {value!r}")
                values[key] = value
            elif key == "peptide_range":
                values[key] = parse_range(value)
            elif key == "time":
                values[key] = parse_time(value, "time")
        return cls(**values)


@dataclass(frozen=True)
class SpectrumIndex:
    """What a rule can match of each spectrum a read would make, without the spectra (for
    ``hxex check``): ``spectrum_id``, ``sample``, ``start_seq``, ``end_seq``, ``charge``,
    ``time`` (NaN for a control), ``role``, ``rep``, ``label`` and ``file`` (the entry's file),
    as on a :class:`~hxex.spectra.SpectrumSet`; built by :func:`hxex.io.spectrum_index` and
    :func:`hxex.io.mzml.scan_index_rows`."""

    spectrum_id: np.ndarray
    sample: np.ndarray
    start_seq: np.ndarray
    end_seq: np.ndarray
    charge: np.ndarray
    time: np.ndarray
    role: np.ndarray
    rep: np.ndarray
    label: np.ndarray
    file: np.ndarray

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping]) -> SpectrumIndex:
        """From one mapping per spectrum with every field."""
        kinds = {"start_seq": np.int64, "end_seq": np.int64, "charge": np.int64,
                 "rep": np.int64, "time": float}
        return cls(**{f.name: np.array([r[f.name] for r in rows],
                                       dtype=kinds.get(f.name, object))
                      for f in fields(cls)})

    def select(self, keep) -> SpectrumIndex:
        keep = np.asarray(keep)
        return SpectrumIndex(**{f.name: getattr(self, f.name)[keep] for f in fields(self)})

    def __len__(self) -> int:
        return len(self.spectrum_id)


def why_nothing(rule: Exclusion, index, reader: str | None = None) -> str:
    """For a rule that matches nothing: the keys that alone match nothing, with the values the
    spectra have (``sample 'wt' (samples: D109H, R120G, WT)``); empty when only the
    combination of its keys matches nothing."""
    out = []
    for key in Exclusion.KEYS:
        value = getattr(rule, key)
        if value is None or matches(Exclusion(**{key: value}), index, reader).any():
            continue
        if key == "peptide_range":
            have = sorted({f"{a}-{b}" for a, b in zip(index.start_seq, index.end_seq,
                                                         strict=True)},
                          key=lambda t: tuple(int(x) for x in t.split("-")))
            shown = f"{value[0]}-{value[1]}"
        elif key == "time":
            t = np.asarray(index.time, dtype=float)
            have = [format_time(x) for x in sorted(set(t[np.isfinite(t)].tolist()))]
            shown = format_time(value)
        elif key == "file":
            have = sorted(set(data_file_names(index, reader).tolist()))
            shown = value
        elif key == "spectrum_id":
            out.append(f"no spectrum has the id {value!r}")
            continue
        elif key == "role":
            have = sorted({str(r) for r in index.role if r})
            shown = value
        else:
            have = sorted({v.item() if hasattr(v, "item") else v
                           for v in np.asarray(getattr(index, key))})
            shown = value
        listed = ", ".join(str(h) for h in have[:12]) + (", ..." if len(have) > 12 else "")
        out.append(f"no spectrum has {key.replace('_', ' ')} {shown!r} "
                   f"({key.replace('_', ' ')}s: {listed or 'none'})")
    return "; ".join(out)


def read_exclusions(value, where: str) -> tuple[Exclusion, ...]:
    """The rules of ``[[where]]`` (a list of tables, or one table): ValueError naming the rule
    on a mistake."""
    if value is None:
        return ()
    items = [value] if isinstance(value, Mapping) else value
    if not isinstance(items, list):
        raise ValueError(f"[[{where}]] must be tables of rules")
    rules = []
    for i, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"[[{where}]] #{i}: expected a table")
        try:
            rules.append(Exclusion.from_dict(item))
        except ValueError as exc:
            raise ValueError(f"[[{where}]] #{i}: {exc}") from None
    return tuple(rules)


def data_file_names(index, reader: str | None) -> np.ndarray:
    """The name of each spectrum's data file, as a rule's ``file`` names it: a scan format's run
    file (the first word of the label), an HX-Express spreadsheet (the entry's file) or a
    SpecExport spectrum file (the label). ``index`` is a :class:`~hxex.spectra.SpectrumSet` or
    :class:`~hxex.picking.BinnedSet`."""
    labels = np.asarray(index.label, dtype=str)
    if reader in _SCAN_READERS:
        return np.array([label.split(" ", 1)[0] for label in labels], dtype=str)
    if reader == "hxexpress":
        return np.asarray(index.file, dtype=str)
    return labels


def matches(rule: Exclusion, index, reader: str | None = None) -> np.ndarray:
    """Which spectra of ``index`` (a spectrum or binned set) ``rule`` matches."""
    n = len(index.spectrum_id)
    keep = np.ones(n, dtype=bool)
    if rule.sample is not None:
        keep &= np.asarray(index.sample, dtype=str) == rule.sample
    if rule.peptide_range is not None:
        keep &= (np.asarray(index.start_seq) == rule.peptide_range[0]) & (
            np.asarray(index.end_seq) == rule.peptide_range[1])
    if rule.charge is not None:
        keep &= np.asarray(index.charge) == rule.charge
    if rule.time is not None:
        time = np.asarray(index.time, dtype=float)
        keep &= np.isfinite(time) & np.isclose(time, rule.time, rtol=1e-9, atol=1e-6)
    if rule.role is not None:
        role = (np.full(n, "", dtype=object) if getattr(index, "role", None) is None
                else np.asarray(index.role, dtype=object))
        keep &= role == rule.role
    if rule.rep is not None:
        keep &= np.asarray(index.rep) == rule.rep
    if rule.file is not None:
        keep &= data_file_names(index, reader) == rule.file
    if rule.spectrum_id is not None:
        keep &= np.asarray(index.spectrum_id, dtype=str) == rule.spectrum_id
    return keep


def exclusion_mask(rules: Sequence[Exclusion], index, reader: str | None = None,
                   where: str = "exclude") -> tuple[np.ndarray, list[int]]:
    """The spectra any of ``rules`` excludes, and the count of each rule (one log line per
    rule; a rule that matches nothing is a warning)."""
    mask = np.zeros(len(index.spectrum_id), dtype=bool)
    counts = []
    for i, rule in enumerate(rules, start=1):
        hit = matches(rule, index, reader)
        counts.append(int(hit.sum()))
        if hit.any():
            log.info("[[%s]] #%d (%s): %d spectra left out", where, i, rule.describe(),
                     int(hit.sum()))
        else:
            log.warning("[[%s]] #%d (%s) matches no spectrum: check its values", where, i,
                        rule.describe())
        mask |= hit
    return mask, counts


def exclusion_record(rules: Sequence[Exclusion], counts: Sequence[int],
                     spectrum_ids: Sequence[str]) -> dict:
    """What a bundle's manifest records of the run-level exclusions (``excluded``)."""
    return {"rules": [dict(r.to_dict(), matched=int(c)) for r, c in zip(rules, counts,
                                                                     strict=True)],
            "spectrum_ids": [str(s) for s in spectrum_ids]}


def exclude_from_results(results, rules: Sequence[Exclusion]):
    """``results`` (:class:`hxex.report.Results`) with the spectra ``rules`` matches rejected in
    its review layer, in memory only (the bundle is not changed): every consumer of the review
    (tables, figures, the analysis) then leaves them out. The spectra are matched on the
    results' picked table (its index); without one, only ``spectrum_id`` rules apply."""
    from .io.bundle import ReviewTable

    if not rules:
        return results
    if results.binned is None:
        raise ValueError("[[review.exclude]] needs the results' picked table (the bundle's "
                         "binned spectra) to match its rules")
    index = results.binned
    source = results.settings.spectra_source if results.settings is not None else None
    reader = (source or {}).get("reader")
    ids_all = np.asarray(index.spectrum_id, dtype=str)
    fitted = np.isin(ids_all, np.asarray(results.fit.spectrum_id, dtype=str))
    notes: dict[str, str] = {}
    for i, rule in enumerate(rules, start=1):
        hit = matches(rule, index, reader) & fitted
        if not hit.any():
            log.warning("[[review.exclude]] #%d (%s) matches no fitted spectrum: check its "
                        "values", i, rule.describe())
            continue
        log.info("[[review.exclude]] #%d (%s): %d spectra left out", i, rule.describe(),
                 int(hit.sum()))
        text = "excluded by [[review.exclude]]" + (f": {rule.note}" if rule.note else "")
        for sid in ids_all[hit]:
            notes.setdefault(str(sid), text)
    if not notes:
        return results
    review = results.review if results.review is not None else ReviewTable.empty()
    ids = list(notes)
    have = set(map(str, review.spectrum_id))
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    status = review.status.astype(object).copy()
    note = review.note.astype(object).copy()
    k_override = review.k_override.copy()
    for j, sid in enumerate(map(str, review.spectrum_id)):
        if sid in notes:
            status[j] = "rejected"
            note[j] = notes[sid] + (f"; {note[j]}" if note[j] else "")
    new = [s for s in ids if s not in have]
    merged = ReviewTable(
        spectrum_id=np.concatenate([review.spectrum_id.astype(object),
                                    np.array(new, dtype=object)]),
        status=np.concatenate([status, np.full(len(new), "rejected", dtype=object)]),
        k_override=np.concatenate([k_override, np.zeros(len(new), dtype=np.int64)]),
        note=np.concatenate([note, np.array([notes[s] for s in new], dtype=object)]),
        updated=np.concatenate([review.updated.astype(object),
                                np.full(len(new), stamp, dtype=object)]),
        stale=np.concatenate([review.stale, np.zeros(len(new), dtype=bool)]),
        source=None if review.source is None else np.concatenate(
            [review.source, np.full(len(new), "exclusion rule", dtype=object)]),
    )
    return results.with_review(merged)
