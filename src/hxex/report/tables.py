# SPDX-License-Identifier: GPL-3.0-or-later
"""Result tables as plain columns (dicts of arrays) and CSV files; no pandas.

Populations are ranked by their mean within each spectrum (1 = lowest), at the K in effect (the
reviewer's choice, else the selected K). Means are in deuterons beyond natural abundance;
``*_corrected`` are the back-exchange-corrected means (empty without controls).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..controls import correct_means
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
    status = results.review.status_of(fit) if results.review is not None else None
    cols["status"] = status if status is not None else np.array(["unreviewed"] * m)
    if results.quality is not None:
        q = results.quality
        cols["flags"] = np.array([
            ";".join(n for n in ("poor_fit", "low_intensity", "noise_outlier")
                     if getattr(q, n)[i]) for i in range(m)
        ])
    return cols


def _populations(results: Results, i: int, k: int):
    """Ranked means, fractions and their standard errors, and the corrected means (or None)."""
    kf = results.fit.fits[k]
    order = np.argsort(kf.means[i])
    mean, frac = kf.means[i][order], kf.fractions[i][order]
    boot = results.bootstrap_for(i, k)
    nan = np.full(k, np.nan)
    mean_se = boot.mean_se[i, :k][order] if boot is not None else nan
    frac_se = boot.frac_se[i, :k][order] if boot is not None else nan
    corrected = corrected_se = None
    if results.controls is not None:
        c = correct_means(results.controls, [results.fit.data_id[i]], mean[None, :],
                          se=mean_se[None, :])
        corrected, corrected_se = c.mean[0], c.se[0]
    return mean, frac, mean_se, frac_se, corrected, corrected_se


def components_table(results: Results) -> dict[str, np.ndarray]:
    """One row per spectrum and population at the K in effect: the index columns, ``K``,
    ``rank``, ``mean``, ``mean_se``, ``mean_corrected``, ``mean_corrected_se``, ``fraction``,
    ``fraction_se``."""
    index = _index_columns(results)
    ks = results.effective_K
    pick: list[int] = []
    cols: dict[str, list] = {n: [] for n in (
        "K", "rank", "mean", "mean_se", "mean_corrected", "mean_corrected_se", "fraction",
        "fraction_se")}
    for i, k in enumerate(ks):
        mean, frac, mean_se, frac_se, cor, cor_se = _populations(results, i, int(k))
        for c in range(int(k)):
            pick.append(i)
            for name, value in (
                ("K", int(k)), ("rank", c + 1), ("mean", mean[c]), ("mean_se", mean_se[c]),
                ("mean_corrected", np.nan if cor is None else cor[c]),
                ("mean_corrected_se", np.nan if cor_se is None else cor_se[c]),
                ("fraction", frac[c]), ("fraction_se", frac_se[c]),
            ):
                cols[name].append(value)
    out = {name: values[pick] for name, values in index.items()}
    out.update({name: np.array(values) for name, values in cols.items()})
    return out


def spectra_table(results: Results) -> dict[str, np.ndarray]:
    """One row per spectrum, in the spirit of pyHXExpress's ``data_fits``: the index columns,
    ``K``, ``centroid`` and ``centroid_corrected`` (the fraction-weighted mean of the populations)
    and ``mean_<r>``, ``mean_corrected_<r>``, ``fraction_<r>``, ``fraction_se_<r>`` for each rank
    ``r`` up to the largest K (empty where the spectrum has fewer populations)."""
    out = dict(_index_columns(results))
    ks = results.effective_K
    k_max = int(ks.max())
    m = len(ks)
    out["K"] = ks.copy()
    centroid = np.full(m, np.nan)
    centroid_cor = np.full(m, np.nan)
    wide = {f"{n}_{r}": np.full(m, np.nan) for r in range(1, k_max + 1)
            for n in ("mean", "mean_corrected", "fraction", "fraction_se")}
    for i, k in enumerate(ks):
        mean, frac, _, frac_se, cor, _ = _populations(results, i, int(k))
        centroid[i] = (mean * frac).sum()
        if cor is not None:
            centroid_cor[i] = (cor * frac).sum()
        for c in range(int(k)):
            r = c + 1
            wide[f"mean_{r}"][i], wide[f"fraction_{r}"][i] = mean[c], frac[c]
            wide[f"fraction_se_{r}"][i] = frac_se[c]
            if cor is not None:
                wide[f"mean_corrected_{r}"][i] = cor[c]
    out["centroid"], out["centroid_corrected"] = centroid, centroid_cor
    out.update(wide)
    return out


def uptake_table(results: Results) -> dict[str, np.ndarray]:
    """Replicate-averaged populations: one row per series, time and population rank with the
    number of replicates ``n``, the mean of the (corrected) means and its standard deviation over
    replicates (NaN with one replicate), and the mean fraction. Spectra are the ones accepted by
    the reviewer or not yet reviewed."""
    comp = components_table(results)
    mean = comp["mean_corrected"] if results.controls is not None else comp["mean"]
    ok = comp["status"] != "rejected"
    rows = []
    keys = sorted({(int(d), float(t), int(r)) for d, t, r in
                   zip(comp["data_id"][ok], comp["time"][ok], comp["rank"][ok], strict=True)})
    for d, t, r in keys:
        sel = ok & (comp["data_id"] == d) & (comp["time"] == t) & (comp["rank"] == r)
        values, fractions = mean[sel], comp["fraction"][sel]
        first = np.flatnonzero(sel)[0]
        rows.append((d, t, r, str(comp["sample"][first]), str(comp["peptide"][first]),
                     str(comp["peptide_range"][first]), int(len(values)), float(values.mean()),
                     float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                     float(fractions.mean())))
    names = ("data_id", "time", "rank", "sample", "peptide", "peptide_range", "n", "mean",
             "mean_sd", "fraction")
    return {name: np.array([row[j] for row in rows]) for j, name in enumerate(names)}


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
    """Write ``spectra.csv``, ``components.csv`` and ``uptake.csv`` into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return [
        write_csv(spectra_table(results), directory / "spectra.csv"),
        write_csv(components_table(results), directory / "components.csv"),
        write_csv(uptake_table(results), directory / "uptake.csv"),
    ]
