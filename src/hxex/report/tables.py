# SPDX-License-Identifier: GPL-3.0-or-later
"""Result tables as plain columns (dicts of arrays) and CSV files; no pandas.

Populations are ranked by their mean within each spectrum (1 = lowest), at the K in effect (the
reviewer's choice, else the selected K). Means are in deuterons beyond natural abundance;
``*_corrected`` are the back-exchange-corrected means (empty without controls). The ``centroid``
of a spectrum is the population-weighted mean of its components' means, the deuterium level of
the whole spectrum (what a single-population analysis would report). ``review_status`` is the
reviewer's decision on a spectrum (``accepted``, ``rejected`` or empty); ``control_fallback``
says a series' back-exchange correction used a fallback instead of its own measured controls.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..controls import CONTROL_NOTES, correct_means
from ..io.entries import parse_range
from ..io.outputs import replace_outputs
from ..noise import noise_reference_kinds
from ..picking.binned import MZ_CHECK_COLUMNS
from ..picking.screen import SCREEN_SCORES
from ..quality import FLAG_NAMES
from ..spectra import point_order_key, time_point
from .results import Results
from .series import _control_row, _reps


def _index_columns(results: Results) -> dict[str, np.ndarray]:
    fit = results.fit
    m = len(fit)
    cols = {"spectrum_id": fit.spectrum_id, "data_id": fit.data_id, "time": fit.time,
            "role": fit.role, "rep": _reps(results)}
    b = results.binned
    names = ("sample", "peptide_range", "peptide", "charge")
    if b is not None:
        index = {str(s): j for j, s in enumerate(b.spectrum_id)}
        rows = np.array([index.get(str(s), -1) for s in fit.spectrum_id])
        for name in names:
            values = getattr(b, name)
            cols[name] = np.array([values[j] if j >= 0 else "" for j in rows])
        for name in ("rt", "rt_start", "rt_end"):  # minutes: the spectrum's, and its scans'
            values = getattr(b, name, None)
            cols[name] = np.array([float(values[j]) if j >= 0 and values is not None else np.nan
                                   for j in rows])
    else:
        for name in names:
            cols[name] = np.array([""] * m)
        for name in ("rt", "rt_start", "rt_end"):
            cols[name] = np.full(m, np.nan)
    cols.update(_target_columns(results))
    cols["review_status"] = np.asarray(results.review_status).astype(str)
    cols["review_K"] = _review_k(results)
    cols["note"] = _review_note(results)
    if results.quality is not None:
        q = results.quality
        cols["flags"] = np.array([
            ";".join(n for n in FLAG_NAMES
                     if getattr(q, n)[i]) for i in range(m)
        ])
    return cols


def _review_k(results: Results) -> np.ndarray:
    """The K the review chose for each spectrum (empty where it chose none or the entry is
    stale): the ``K`` of the tables is then this one, not the fit's choice."""
    out = np.full(len(results.fit), "", dtype=object)  # a whole number, or empty
    review = results.review
    if review is None or not len(review):
        return out
    index = {str(s): i for i, s in enumerate(results.fit.spectrum_id)}
    for sid, k, stale in zip(review.spectrum_id, review.k_override, review.stale, strict=True):
        if k and not stale and str(sid) in index:
            out[index[str(sid)]] = int(k)
    return out


def _review_note(results: Results) -> np.ndarray:
    """The review's note on each spectrum (empty without one): the ``note`` column a user
    fills in a copy of ``spectra.csv`` (the user review table) comes back here."""
    out = np.full(len(results.fit), "", dtype=object)
    review = results.review
    if review is None or not len(review):
        return out
    index = {str(s): i for i, s in enumerate(results.fit.spectrum_id)}
    for sid, note in zip(review.spectrum_id, review.note, strict=True):
        if note and str(sid) in index:
            out[index[str(sid)]] = str(note)
    return out


# the note exclusion rules give their entries (hxex.exclusions.exclude_from_results)
_RULE_NOTE = "excluded by [[review.exclude]]"


def review_table(results: Results) -> dict[str, np.ndarray]:
    """One row per entry of the review layer, with the spectrum's sample, peptide, charge,
    time, role and replicate (empty for a stale entry whose spectrum the run no longer has):
    ``status``, ``review_K`` (the chosen K, empty for none), ``note``, ``updated``, ``stale``,
    and ``source``: ``review`` (stored in the bundle), ``review table`` (the user review table,
    :mod:`hxex.review_table`) or ``exclusion rule`` (an ``[[review.exclude]]`` rule; both
    applied for this table only)."""
    review = results.review
    index = _index_columns(results)
    rows = {str(s): i for i, s in enumerate(results.fit.spectrum_id)}
    names = ("sample", "peptide_range", "peptide", "charge", "time", "role", "rep")
    cols: dict[str, list] = {n: [] for n in ("spectrum_id", *names, "status", "review_K",
                                             "note", "updated", "stale", "source")}
    for j, sid in enumerate(map(str, review.spectrum_id)):
        i = rows.get(sid)
        cols["spectrum_id"].append(sid)
        for name in names:
            value = index[name][i] if i is not None else ""
            cols[name].append(np.nan if name == "time" and i is None else value)
        k = int(review.k_override[j])
        note = str(review.note[j])
        cols["status"].append(str(review.status[j]))
        cols["review_K"].append(k if k else "")
        cols["note"].append(note)
        cols["updated"].append(str(review.updated[j]))
        cols["stale"].append(bool(review.stale[j]))
        source = "review" if review.source is None else str(review.source[j])
        cols["source"].append("exclusion rule" if note.startswith(_RULE_NOTE) else source)
    return {name: np.array(values, dtype=object) for name, values in cols.items()}


TARGET_COLUMNS = ("fragment", "ms_level", "preactivation", "activation", "precursor_mz",
                  "precursor_charge")


def _target_columns(results: Results) -> dict[str, np.ndarray]:
    """For spectra read from mzML or Thermo raw files, how each measurement's scans were
    chosen: the fragment (for example ``c-1:4``, empty for a peptide), the MS level, the
    activation before the last stage (MS3), the activation and the precursor (empty or NaN
    where the target set none). Nothing for other data."""
    source = results.settings.spectra_source if results.settings is not None else None
    from ..io.mzml import SCAN_READERS, mzml_series_targets

    if not source or source.get("reader") not in SCAN_READERS or "series" not in source:
        return {}

    targets = mzml_series_targets(source)
    rows = [targets.get(int(d)) for d in results.fit.data_id]
    return {
        "fragment": np.array([dict(t.entry.extra).get("fragment", "") if t else "" for t in rows]),
        "ms_level": np.array([t.ms_level if t else 0 for t in rows], dtype=np.int64),
        "preactivation": np.array([(t.preactivation or "") if t else "" for t in rows]),
        "activation": np.array([(t.activation or "") if t else "" for t in rows]),
        "precursor_mz": np.array([t.precursor_mz if t and t.precursor_mz is not None
                                  else np.nan for t in rows], dtype=float),
        "precursor_charge": np.array([t.precursor_charge if t and t.precursor_charge is not None
                                      else 0 for t in rows], dtype=np.int64),
    }


def _populations(results: Results, i: int, k: int):
    """Ranked means, populations and their standard errors, and the corrected means (or None:
    no controls, or none for the spectrum's series)."""
    results.fit.require_fitted(i, k)
    kf = results.fit.fits[k]
    order = np.argsort(kf.means[i])
    mean, frac = kf.means[i][order], kf.populations[i][order]
    boot = results.bootstrap_for(i, k)
    nan = np.full(k, np.nan)
    mean_se = boot.mean_se[i, :k][order] if boot is not None else nan
    population_se = boot.population_se[i, :k][order] if boot is not None else nan
    corrected = corrected_se = None
    if _control_row(results, int(results.fit.data_id[i])) is not None:
        c = correct_means(results.controls, [results.fit.data_id[i]], mean[None, :],
                          se=mean_se[None, :])
        corrected, corrected_se = c.mean[0], c.se[0]
    return mean, frac, mean_se, population_se, corrected, corrected_se


def components_table(results: Results) -> dict[str, np.ndarray]:
    """One row per spectrum and population at the K in effect: the index columns (ids, time,
    replicate, sample, peptide, charge, the scan target, ``review_status``, ``flags``), ``K``,
    ``rank``, ``mean``, ``mean_se``, ``mean_corrected``, ``mean_corrected_se``, ``population``,
    ``population_se``."""
    index = _index_columns(results)
    ks = results.effective_K
    pick: list[int] = []
    cols: dict[str, list] = {n: [] for n in (
        "K", "rank", "mean", "mean_se", "mean_corrected", "mean_corrected_se", "population",
        "population_se")}
    for i, k in enumerate(ks):
        mean, frac, mean_se, population_se, cor, cor_se = _populations(results, i, int(k))
        for c in range(int(k)):
            pick.append(i)
            for name, value in (
                ("K", int(k)), ("rank", c + 1), ("mean", mean[c]), ("mean_se", mean_se[c]),
                ("mean_corrected", np.nan if cor is None else cor[c]),
                ("mean_corrected_se", np.nan if cor_se is None else cor_se[c]),
                ("population", frac[c]), ("population_se", population_se[c]),
            ):
                cols[name].append(value)
    out = {name: values[pick] for name, values in index.items()}
    out.update({name: np.array(values) for name, values in cols.items()})
    return out


def spectra_table(results: Results) -> dict[str, np.ndarray]:
    """One row per spectrum: the index columns,
    ``K``, ``centroid`` and ``centroid_corrected`` (the population-weighted mean of the
    components) and ``mean_<r>``, ``mean_corrected_<r>``, ``population_<r>``,
    ``population_se_<r>`` for each rank ``r`` up to the largest K (empty where the spectrum has
    fewer components); with the population test (:mod:`hxex.population_test`), the K the
    fit's criterion chose (``criterion_K``; ``K`` is then the test's), each step's p-value
    (``test_p_<K>``, K against K + 1; empty where untested) and ``narrow_component`` (the width
    check left the spectrum at fewer populations); then the spectrum quality screens' scores
    (``screen_*``, :mod:`hxex.picking.screen`; empty where not assessed) and the m/z check of
    the spectrum's series (``envelope_offset_ppm``, ``envelope_offset_limit_ppm``,
    ``window_shift_ppm``, :mod:`hxex.picking.mz_check`; the first two empty where not
    assessed, the last 0 except in bundles picked with the former ``recentre``)."""
    out = dict(_index_columns(results))
    ks = results.effective_K
    k_max = int(ks.max())
    m = len(ks)
    out["K"] = ks.copy()
    centroid = np.full(m, np.nan)
    centroid_cor = np.full(m, np.nan)
    wide = {f"{n}_{r}": np.full(m, np.nan) for r in range(1, k_max + 1)
            for n in ("mean", "mean_corrected", "population", "population_se")}
    for i, k in enumerate(ks):
        mean, frac, _, population_se, cor, _ = _populations(results, i, int(k))
        centroid[i] = (mean * frac).sum()
        if cor is not None:
            centroid_cor[i] = (cor * frac).sum()
        for c in range(int(k)):
            r = c + 1
            wide[f"mean_{r}"][i], wide[f"population_{r}"][i] = mean[c], frac[c]
            wide[f"population_se_{r}"][i] = population_se[c]
            if cor is not None:
                wide[f"mean_corrected_{r}"][i] = cor[c]
    out["centroid"], out["centroid_corrected"] = centroid, centroid_cor
    out.update(wide)
    out.update(_test_columns(results))
    out.update(_screen_columns(results))
    return out


def _test_columns(results: Results) -> dict[str, np.ndarray]:
    test = results.population_test
    if test is None:
        return {}
    out = {"criterion_K": np.asarray(test.criterion_K, dtype=np.int64)}
    for s, k in enumerate(test.steps):
        out[f"test_p_{k}"] = np.asarray(test.p[:, s], dtype=float)
    # the fit at the K in effect: the step that added its last population (none at K = 1)
    step = {int(k) + 1: s for s, k in enumerate(test.steps)}
    narrow = np.asarray(test.narrow, dtype=bool)
    out["narrow_component"] = np.array(
        [s is not None and bool(narrow[i, s]) for i, s in
         enumerate(step.get(int(k)) for k in results.effective_K)], dtype=bool)
    return out


def _screen_columns(results: Results) -> dict[str, np.ndarray]:
    m = len(results.fit)
    b = results.binned
    names = (*SCREEN_SCORES, *MZ_CHECK_COLUMNS)
    if b is None:
        return {name: np.full(m, np.nan) for name in names}
    index = {str(s): j for j, s in enumerate(b.spectrum_id)}
    rows = np.array([index.get(str(s), -1) for s in results.fit.spectrum_id], dtype=np.int64)
    return {name: np.where(rows >= 0, np.asarray(getattr(b, name))[rows], np.nan)
            for name in names}


def uptake_table(results: Results) -> dict[str, np.ndarray]:
    """Replicate-averaged populations: one row per series, time, ``K`` (the spectra's K in
    effect: spectra fitted at different K are never averaged together) and population rank,
    with the number of replicates averaged (``n_replicates``), the ``mean`` of the means over them
    (back-exchange corrected for a series the controls correct, as measured for the others) and
    its standard deviation (``replicate_sd``; NaN with one replicate), and the mean
    ``population``. Spectra are the ones accepted by the reviewer or not yet reviewed."""
    comp = components_table(results)
    ctrl = results.controls
    corrected = (np.zeros(len(comp["data_id"]), dtype=bool) if ctrl is None else
                 np.isin(comp["data_id"], ctrl.data_id[~np.asarray(ctrl.uncorrected, bool)]))
    mean = np.where(corrected, comp["mean_corrected"], comp["mean"])  # per series
    ok = comp["review_status"] != "rejected"
    rows = []
    role = comp["role"] if "role" in comp else np.full(len(comp["time"]), "", dtype="<U8")
    points = time_point(comp["time"], role)  # replicates share a point: a time, or a control
    keys = {}
    for i in np.flatnonzero(ok):
        key = (int(comp["data_id"][i]), points[i], int(comp["K"][i]), int(comp["rank"][i]))
        keys.setdefault(key, (key[0], *point_order_key(float(comp["time"][i]), str(role[i])),
                              *key[2:]))
    for d, point, k, r in sorted(keys, key=lambda k: keys[k]):
        sel = (ok & (comp["data_id"] == d) & (points == point) & (comp["K"] == k)
               & (comp["rank"] == r))
        values, fractions = mean[sel], comp["population"][sel]
        first = np.flatnonzero(sel)[0]
        rows.append((d, float(comp["time"][first]), str(role[first]), k, r,
                     str(comp["sample"][first]), str(comp["peptide"][first]),
                     str(comp["peptide_range"][first]), int(len(values)), float(values.mean()),
                     float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                     float(fractions.mean())))
    names = ("data_id", "time", "role", "K", "rank", "sample", "peptide", "peptide_range",
             "n_replicates", "mean", "replicate_sd", "population")
    return {name: np.array([row[j] for row in rows]) for j, name in enumerate(names)}


def _median_rt(binned, where, ids) -> float:
    rt = getattr(binned, "rt", None)
    if rt is None:
        return np.nan
    values = [float(rt[where[str(i)]]) for i in ids if str(i) in where]
    values = [v for v in values if np.isfinite(v)]
    return float(np.median(values)) if values else np.nan


def series_table(results: Results) -> dict[str, np.ndarray]:
    """One row per series (a peptide at one charge in one sample), ordered by ``data_id``:
    ``data_id``, ``sample``, ``peptide_range``, ``peptide``, ``charge`` (empty or 0 without the
    picked table), ``n_spectra`` (fitted), ``n_flagged`` (with a quality flag), ``n_rejected``
    (by the reviewer), ``n_multi`` (more than one population at the K in effect), ``controls``
    (``measured``, ``uncorrected`` for a series the controls leave as measured, such as a
    fragment ion, or the fallback reasons; empty without controls), ``control_notes`` (the
    control flags that only warn, :data:`hxex.controls.CONTROL_NOTES`), ``retention`` (the
    deuterium retention used for the back-exchange correction, a fraction; NaN without
    controls), ``deamidated_share`` (with ``[run.controls] deamidation``, for a series holding
    an Asn-Gly site: its sample's deamidated share of the fully deuterated controls,
    :mod:`hxex.deamidation`; NaN otherwise or when not assessed), ``control_fallback`` (the
    correction used a fallback, not the series' own
    controls), ``noise_reference`` (what its noise level comes from: ``undeuterated control``,
    ``own spectrum`` when no undeuterated control of the series was fitted, or ``run's scans``
    for kinetics; :func:`hxex.noise.noise_reference_kinds`) and ``rt`` (the median
    chromatographic retention time of its spectra, minutes; NaN: unknown)."""
    fit = results.fit
    ids = np.unique(fit.data_id)
    b = results.binned
    where = {str(s): j for j, s in enumerate(b.spectrum_id)} if b is not None else {}
    flagged = results.quality.flagged if results.quality is not None else np.zeros(len(fit), bool)
    rejected = np.asarray(results.review_status == "rejected", dtype=bool)
    multi = results.effective_K > 1
    cols: dict[str, list] = {n: [] for n in (
        "data_id", "sample", "peptide_range", "peptide", "charge", "n_spectra", "n_flagged",
        "n_rejected", "n_multi", "controls", "control_notes", "retention", "deamidated_share",
        "control_fallback", "noise_reference", "rt")}
    ctrl = results.controls
    ctrl_rows = {int(d): r for r, d in enumerate(ctrl.data_id)} if ctrl is not None else {}
    uncorrected = ctrl.uncorrected if ctrl is not None else np.zeros(0, dtype=bool)
    meas = results.measurements
    refs = noise_reference_kinds(meas.data_id, meas.references, meas.role)
    for d in ids:
        members = np.flatnonzero(fit.data_id == d)
        j = where.get(str(fit.spectrum_id[members[0]]))
        r = ctrl_rows.get(int(d))
        values = {
            "data_id": int(d),
            "sample": "" if j is None else str(b.sample[j]),
            "peptide_range": "" if j is None else str(b.peptide_range[j]),
            "peptide": "" if j is None else str(b.peptide[j]),
            "charge": 0 if j is None else int(b.charge[j]),
            "n_spectra": len(members), "n_flagged": int(flagged[members].sum()),
            "n_rejected": int(rejected[members].sum()), "n_multi": int(multi[members].sum()),
            "controls": "" if r is None else ("uncorrected" if uncorrected[r]
                                              else "measured" if not ctrl.fallback[r]
                                              else str(ctrl.flags[r])),
            "control_notes": "" if r is None else _control_notes(str(ctrl.flags[r])),
            "retention": np.nan if r is None else float(ctrl.retention[r]),
            "deamidated_share": _deamidated_share(ctrl, int(d)),
            "control_fallback": False if r is None else bool(ctrl.fallback[r]),
            "noise_reference": refs.get(int(d), ""),
            "rt": _median_rt(b, where, fit.spectrum_id[members]),
        }
        for name, value in values.items():
            cols[name].append(value)
    out = {name: np.array(values) for name, values in cols.items()}
    for name in ("sample", "peptide_range", "peptide", "controls", "noise_reference"):
        out[name] = out[name].astype(str)
    return out


def _range_of(label) -> tuple[int, int] | None:
    """The residues a range label names, or None for one that is not ``START-END``."""
    try:
        return parse_range(label)
    except ValueError:
        return None


def find_series(
    results: Results,
    *,
    samples: Sequence[str] | None = None,
    peptide_ranges: Sequence[str] | None = None,
    charges: Sequence[int] | None = None,
    flagged: bool = False,
) -> list[int]:
    """The ``data_id`` of the series that match every filter given (None: no filter).
    ``flagged`` keeps only series with a flagged or rejected spectrum or a control that fell
    back (the same rule as ``select_series(results, 'flagged')``)."""
    t = series_table(results)
    keep = np.ones(len(t["data_id"]), dtype=bool)
    if samples is not None:
        keep &= np.isin(t["sample"], [str(x) for x in samples])
    if peptide_ranges is not None:  # by residues: "1-10" is "0001-0010"
        wanted = {parse_range(r) for r in peptide_ranges}
        keep &= np.array([_range_of(r) in wanted for r in t["peptide_range"]], dtype=bool)
    if charges is not None:
        keep &= np.isin(t["charge"], [int(x) for x in charges])
    if flagged:
        keep &= (t["n_flagged"] > 0) | (t["n_rejected"] > 0) | t["control_fallback"]
    return [int(d) for d in t["data_id"][keep]]


def format_table(
    columns: dict[str, np.ndarray], rows: Sequence[int] | None = None, limit: int | None = 40,
    width: int = 30,
) -> str:
    """Columns as aligned plain text (for a notebook or a terminal): the ``rows`` given (default
    all), at most ``limit`` of them (None: no limit), each cell cut at ``width`` characters."""
    names = list(columns)
    n = len(columns[names[0]]) if names else 0
    order = list(range(n)) if rows is None else [int(r) for r in rows]
    shown = order if limit is None else order[:limit]

    def cell(value) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{float(value):.3g}"
        return str(value)[:width]

    text = [[cell(columns[c][i]) for c in names] for i in shown]
    widths = [max([len(c), *(len(r[k]) for r in text)]) for k, c in enumerate(names)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(names, widths, strict=True))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)) for r in text]
    if len(shown) < len(order):
        lines.append(f"... and {len(order) - len(shown)} more")
    return "\n".join(lines)


def write_csv(columns: dict[str, np.ndarray], path: str | Path) -> Path:
    """Write columns of equal length as a CSV file with a header (UTF-8); NaN is written as
    empty, an infinite value as ``inf`` or ``-inf``."""
    path = Path(path)
    names = list(columns)
    n = len(columns[names[0]]) if names else 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for i in range(n):
            writer.writerow([_cell(columns[name][i]) for name in names])
    return path


def _cell(value) -> str:
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else repr(float(value))  # inf stays 'inf'
    return str(value)


def _deamidated_share(ctrl, data_id: int) -> float:
    """The deamidated share of the series' sample when it holds an Asn-Gly site (NaN)."""
    found = None if ctrl is None or ctrl.deamidation is None else ctrl.deamidation.of(data_id)
    return float("nan") if found is None else float(found.share)


def _control_notes(flags: str) -> str:
    """The control flags that only warn (:data:`hxex.controls.CONTROL_NOTES`), for
    ``series.csv``; ``fd_multimodal`` says whether its retention was checked against the same
    peptide's controls in the other samples: ``(retention agrees)``, ``(retention differs)`` (with
    ``fd_disagrees``), or, with no other sample to compare, against its other charge states
    (``(retention agrees with its other charge states)``), else ``(retention unchecked)``."""
    tokens = [f for f in flags.split(";") if f]
    notes = []
    for f in tokens:
        if f == "fd_multimodal":
            check = ("retention differs" if "fd_disagrees" in tokens
                     else "retention agrees" if "fd_retention_agrees" in tokens
                     else "retention agrees with its other charge states"
                     if "fd_charges_agree" in tokens
                     else "retention unchecked")
            notes.append(f"fd_multimodal ({check})")
        elif f in CONTROL_NOTES and f not in ("fd_retention_agrees", "fd_charges_agree"):
            notes.append(f)
    return ";".join(notes)


def retention_suggestions_table(results: Results) -> dict | None:
    """The retention-time search of the read (:mod:`hxex.io.retention_search`), one row per
    series searched: ``sample``, ``peptide_range``, ``peptide``, ``charge``, ``modification``,
    why it was searched (``off_mz``, ``no undeuterated offset``, ``no undeuterated
    spectrum``), the peptide table's ``table_rt`` and the time it was read at (``read_rt``,
    minutes), the ``verdict`` (``here``, ``elsewhere``, ``uncertain``, ``ambiguous``, ``not
    found``), the ``proposed_rt``, whether it was ``applied`` (``automatic``), the number of
    undeuterated injections searched and every ``candidates`` time with its height; None
    when the read did not search."""
    from ..analysis.screening import spectra_source
    from ..io.retention_search import suggestions_table

    source = spectra_source(results)
    return suggestions_table(None if source is None else source.get("retention_search"))


def write_tables(results: Results, directory: str | Path, *, created: str | None = None
                 ) -> list[Path]:
    """Write ``series.csv``, ``spectra.csv``, ``components.csv`` and ``uptake.csv`` into
    ``directory``, ``review.csv`` when the review layer (or an exclusion rule) has entries and
    ``retention_suggestions.csv`` when the read searched for retention times
    (:func:`retention_suggestions_table`), replacing the tables hxex wrote there before (its
    listing, :func:`hxex.io.replace_outputs`: a ``review.csv`` with nothing left to list goes)
    and never a file of your own. The listing
    records the results they were made from (``created``, the bundle's stamp; default the
    results' own), so ``hxex info`` can tell when they are out of date."""
    directory = Path(directory)
    tables = {"series.csv": series_table, "spectra.csv": spectra_table,
              "components.csv": components_table, "uptake.csv": uptake_table}
    if results.review is not None and len(results.review):
        tables["review.csv"] = review_table
    if retention_suggestions_table(results) is not None:
        tables["retention_suggestions.csv"] = retention_suggestions_table
    made = {"": created if created is not None else results.settings.created
            if results.settings is not None else ""}
    with replace_outputs(directory, tables, made=made) as written:
        for name, make in tables.items():
            written.append(write_csv(make(results), directory / name))
    return list(written)
