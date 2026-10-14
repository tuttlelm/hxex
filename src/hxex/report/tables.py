# SPDX-License-Identifier: GPL-3.0-or-later
"""Result tables as plain columns (dicts of arrays) and CSV files; no pandas.

Populations are ranked by their mean within each spectrum (1 = lowest), at the K in effect (the
reviewer's choice, else the selected K). Means are in deuterons beyond natural abundance;
``*_corrected`` are the back-exchange-corrected means (empty without controls).
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..controls import correct_means
from ..quality import FLAG_NAMES
from .results import Results
from .series import _reps


def _index_columns(results: Results) -> dict[str, np.ndarray]:
    fit = results.fit
    m = len(fit)
    cols = {"spectrum_id": fit.spectrum_id, "data_id": fit.data_id, "time": fit.time,
            "rep": _reps(results)}
    b = results.binned
    names = ("sample", "peptide_range", "peptide", "charge")
    if b is not None:
        index = {str(s): j for j, s in enumerate(b.spectrum_id)}
        rows = np.array([index.get(str(s), -1) for s in fit.spectrum_id])
        for name in names:
            values = getattr(b, name)
            cols[name] = np.array([values[j] if j >= 0 else "" for j in rows])
    else:
        for name in names:
            cols[name] = np.array([""] * m)
    cols.update(_target_columns(results))
    status = results.review.status_of(fit) if results.review is not None else None
    cols["status"] = status if status is not None else np.array(["unreviewed"] * m)
    if results.quality is not None:
        q = results.quality
        cols["flags"] = np.array([
            ";".join(n for n in FLAG_NAMES
                     if getattr(q, n)[i]) for i in range(m)
        ])
    return cols


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
    """Ranked means, populations and their standard errors, and the corrected means (or None)."""
    kf = results.fit.fits[k]
    order = np.argsort(kf.means[i])
    mean, frac = kf.means[i][order], kf.populations[i][order]
    boot = results.bootstrap_for(i, k)
    nan = np.full(k, np.nan)
    mean_se = boot.mean_se[i, :k][order] if boot is not None else nan
    population_se = boot.population_se[i, :k][order] if boot is not None else nan
    corrected = corrected_se = None
    if results.controls is not None:
        c = correct_means(results.controls, [results.fit.data_id[i]], mean[None, :],
                          se=mean_se[None, :])
        corrected, corrected_se = c.mean[0], c.se[0]
    return mean, frac, mean_se, population_se, corrected, corrected_se


def components_table(results: Results) -> dict[str, np.ndarray]:
    """One row per spectrum and population at the K in effect: the index columns, ``K``,
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
    """One row per spectrum, in the spirit of pyHXExpress's ``data_fits``: the index columns,
    ``K``, ``centroid`` and ``centroid_corrected`` (the population-weighted mean of the
    components) and ``mean_<r>``, ``mean_corrected_<r>``, ``population_<r>``,
    ``population_se_<r>`` for each rank ``r`` up to the largest K (empty where the spectrum has
    fewer components)."""
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
    return out


def uptake_table(results: Results) -> dict[str, np.ndarray]:
    """Replicate-averaged populations: one row per series, time and population rank with the
    number of replicates ``n``, the mean of the (corrected) means and its standard deviation over
    replicates (NaN with one replicate), and the mean population. Spectra are the ones accepted by
    the reviewer or not yet reviewed."""
    comp = components_table(results)
    mean = comp["mean_corrected"] if results.controls is not None else comp["mean"]
    ok = comp["status"] != "rejected"
    rows = []
    keys = sorted({(int(d), float(t), int(r)) for d, t, r in
                   zip(comp["data_id"][ok], comp["time"][ok], comp["rank"][ok], strict=True)})
    for d, t, r in keys:
        sel = ok & (comp["data_id"] == d) & (comp["time"] == t) & (comp["rank"] == r)
        values, fractions = mean[sel], comp["population"][sel]
        first = np.flatnonzero(sel)[0]
        rows.append((d, t, r, str(comp["sample"][first]), str(comp["peptide"][first]),
                     str(comp["peptide_range"][first]), int(len(values)), float(values.mean()),
                     float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                     float(fractions.mean())))
    names = ("data_id", "time", "rank", "sample", "peptide", "peptide_range", "n", "mean",
             "mean_sd", "population")
    return {name: np.array([row[j] for row in rows]) for j, name in enumerate(names)}


def series_table(results: Results) -> dict[str, np.ndarray]:
    """One row per series (a peptide at one charge in one sample), ordered by ``data_id``:
    ``data_id``, ``sample``, ``peptide_range``, ``peptide``, ``charge`` (empty or 0 without the
    picked table), ``n_spectra`` (fitted), ``n_flagged`` (with a quality flag), ``n_rejected``
    (by the reviewer), ``n_multi`` (more than one population at the K in effect), ``controls``
    (``measured``, or the fallback reasons; empty without controls), ``retention`` (the one used;
    NaN without controls) and ``fallback``."""
    fit = results.fit
    ids = np.unique(fit.data_id)
    b = results.binned
    where = {str(s): j for j, s in enumerate(b.spectrum_id)} if b is not None else {}
    flagged = results.quality.flagged if results.quality is not None else np.zeros(len(fit), bool)
    rejected = (results.review.status_of(fit) == "rejected" if results.review is not None
                else np.zeros(len(fit), bool))
    multi = results.effective_K > 1
    cols: dict[str, list] = {n: [] for n in (
        "data_id", "sample", "peptide_range", "peptide", "charge", "n_spectra", "n_flagged",
        "n_rejected", "n_multi", "controls", "retention", "fallback")}
    ctrl = results.controls
    ctrl_rows = {int(d): r for r, d in enumerate(ctrl.data_id)} if ctrl is not None else {}
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
            "controls": "" if r is None else ("measured" if not ctrl.fallback[r]
                                              else str(ctrl.flags[r])),
            "retention": np.nan if r is None else float(ctrl.retention[r]),
            "fallback": False if r is None else bool(ctrl.fallback[r]),
        }
        for name, value in values.items():
            cols[name].append(value)
    out = {name: np.array(values) for name, values in cols.items()}
    for name in ("sample", "peptide_range", "peptide", "controls"):
        out[name] = out[name].astype(str)
    return out


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
    if peptide_ranges is not None:
        keep &= np.isin(t["peptide_range"], [str(x) for x in peptide_ranges])
    if charges is not None:
        keep &= np.isin(t["charge"], [int(x) for x in charges])
    if flagged:
        keep &= (t["n_flagged"] > 0) | (t["n_rejected"] > 0) | t["fallback"]
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
            return "" if not np.isfinite(value) else f"{float(value):.3g}"
        return str(value)[:width]

    text = [[cell(columns[c][i]) for c in names] for i in shown]
    widths = [max([len(c), *(len(r[k]) for r in text)]) for k, c in enumerate(names)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(names, widths, strict=True))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)) for r in text]
    if len(shown) < len(order):
        lines.append(f"... and {len(order) - len(shown)} more")
    return "\n".join(lines)


def write_csv(columns: dict[str, np.ndarray], path: str | Path) -> Path:
    """Write columns of equal length as a CSV file with a header; NaN is written as empty."""
    path = Path(path)
    names = list(columns)
    n = len(columns[names[0]]) if names else 0
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for i in range(n):
            writer.writerow([_cell(columns[name][i]) for name in names])
    return path


def _cell(value) -> str:
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(value) else repr(float(value))
    return str(value)


def write_tables(results: Results, directory: str | Path) -> list[Path]:
    """Write ``series.csv``, ``spectra.csv``, ``components.csv`` and ``uptake.csv`` into
    ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return [
        write_csv(series_table(results), directory / "series.csv"),
        write_csv(spectra_table(results), directory / "spectra.csv"),
        write_csv(components_table(results), directory / "components.csv"),
        write_csv(uptake_table(results), directory / "uptake.csv"),
    ]
