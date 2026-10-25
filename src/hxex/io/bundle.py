# SPDX-License-Identifier: GPL-3.0-or-later
"""The results bundle: everything needed to replot and review a run, without refitting.

A bundle is a directory (or a zip of one, read only) holding ``manifest.json`` and Parquet
tables, readable from Python, R and Julia (see docs/BUNDLE.md for the schema). Spectra, binned
tables and fits are stored as exact float64, int64, bool and string columns, one row per
spectrum (or per spectrum and K), with the spectrum id as the key, so loading gives back the
objects the run produced, bit for bit. The review layer (accept or reject, a chosen K, notes) is
the only part that changes after writing; it needs no refit.

Written tables: ``measurements`` and ``fit`` (always), ``fit_k``, ``noise_estimate``,
``bootstrap``, ``bootstrap_draws``, ``bootstrap_k`` and ``bootstrap_k_draws`` (fixed-K
results), ``binned``, ``spectra`` (raw), ``review``, ``controls``,
``quality``, ``envelopes``, and a derived tidy
``components`` table for other tools. The loader ignores tables it does not know.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..bootstrap import BootstrapConfig, BootstrapDraws, BootstrapResult, interval_table
from ..controls import ControlConfig, ControlTable, correct_means
from ..fit import FitConfig, FitResult, KFit, Measurements
from ..noise import NoiseEstimate
from ..picking import BinnedSet, EnvelopeConfig, EnvelopeTable, PickerConfig
from ..picking.binned import MZ_CHECK_COLUMNS
from ..picking.screen import SCREEN_SCORES
from ..quality import FLAG_NAMES, QualityConfig, QualityFlags
from ..spectra import FULLDEUT, UNDEUT, SpectrumSet
from ._arrow import (
    ROLE,
    _col,
    _flat_ragged,
    _index_arrays,
    _index_table,
    _matrix,
    _ragged,
    _ragged_read,
    _rows,
    _spectra_from_table,
    _spectra_table,
    _split,
    _str,
    _strs,
)

log = logging.getLogger(__name__)
SCHEMA_VERSION = 2  # 2: a component's share is "population" (was "frac"/"fraction")
KIND = "hxex-results-bundle"
MANIFEST = "manifest.json"
PENDING_STAGES = ("bootstrap",)  # stages a bundle can be written without, to add later
ROW_GROUP = 1000  # rows per Parquet row group (lazy loading reads whole groups)
STATUSES = ("accepted", "rejected")


class BundleError(Exception):
    """A bundle could not be written, read or verified."""


# --- stores: a directory, or a zip (read only) ------------------------------------------------

class _DirStore:
    writable = True

    def __init__(self, path: Path):
        self.path = path

    def exists(self, name: str) -> bool:
        return (self.path / name).is_file()

    def read(self, name: str) -> bytes:
        return (self.path / name).read_bytes()

    def table(self, name: str, columns=None, filters=None) -> pa.Table:
        return pq.read_table(self.path / name, columns=columns, filters=filters)

    def close(self) -> None:
        pass


def _zip_root(zf: zipfile.ZipFile) -> str:
    """Where the bundle is inside a zip: ``''`` when ``manifest.json`` is at the top (as hxex
    writes it), or ``'NAME/'`` when the zip holds one folder with the bundle in it (as Windows'
    "Compressed (zipped) folder" and macOS "Compress" make it; the ``__MACOSX`` folder and
    ``.DS_Store`` files macOS adds are ignored). ``''`` when neither fits (no bundle there)."""
    names = [n for n in zf.namelist() if not n.startswith("__MACOSX/")
             and n.rsplit("/", 1)[-1] != ".DS_Store"]
    if MANIFEST in names:
        return ""
    roots = {n.split("/", 1)[0] + "/" for n in names}
    if len(roots) == 1 and (root := roots.pop()) + MANIFEST in names:
        return root
    return ""


class _ZipStore:
    writable = False

    def __init__(self, path: Path):
        self.zf = zipfile.ZipFile(path)
        self.root = _zip_root(self.zf)

    def exists(self, name: str) -> bool:
        return self.root + name in self.zf.namelist()

    def read(self, name: str) -> bytes:
        return self.zf.read(self.root + name)

    def table(self, name: str, columns=None, filters=None) -> pa.Table:
        return pq.read_table(pa.BufferReader(self.read(name)), columns=columns, filters=filters)

    def close(self) -> None:
        self.zf.close()


# --- writing --------------------------------------------------------------------------------



def _binned_table(b: BinnedSet) -> pa.Table:
    cols = _index_table(b)
    cols["mz"] = _flat_ragged(b.mz, b.offsets, pa.float64())
    cols["intensity"] = _flat_ragged(b.intensity, b.offsets, pa.float64())
    cols["n_deut"] = _flat_ragged(b.n_deut, b.offsets, pa.int64())
    cols["peak_mz"] = _flat_ragged(b.peak_mz, b.offsets, pa.float64())
    per_spectrum = (
        ("mz_offset_ppm", np.float64),
        ("env_width", np.float64), ("env_symm", np.float64), ("threshold", np.float64),
        ("max_namides", np.int64), ("empty", bool), ("no_signal", bool), ("capped", bool),
        *((name, np.float64) for name in SCREEN_SCORES),
        *((name, np.float64) for name in MZ_CHECK_COLUMNS),
    )
    for name, typ in per_spectrum:
        cols[name] = pa.array(np.asarray(getattr(b, name), dtype=typ))
    return pa.table(cols)


def _measurements_table(m: Measurements) -> pa.Table:
    n = len(m)
    refs = [None if r is None else np.asarray(r, dtype=np.int64) for r in m.references]
    return pa.table({
        "measurement": pa.array(np.arange(n, dtype=np.int64)),
        "spectrum_id": _str(m.spectrum_id),
        "data_id": pa.array(np.asarray(m.data_id, dtype=np.int64)),
        "time": pa.array(np.asarray(m.time, dtype=np.float64)),
        ROLE: _str(m.role),
        "n_max": pa.array(np.asarray(m.n_max, dtype=np.float64)),
        "bins": _ragged(list(m.bins), pa.float64()),
        "observed": _ragged(list(m.observed), pa.float64()),
        "shape": _ragged(list(m.shapes), pa.float64()),
        "references": _ragged(refs, pa.int64()),
    })


def _fit_tables(fit: FitResult) -> dict[str, pa.Table]:
    n = len(fit)
    tables = {"fit": pa.table({
        "measurement": pa.array(np.arange(n, dtype=np.int64)),
        "spectrum_id": _str(fit.spectrum_id),
        "data_id": pa.array(np.asarray(fit.data_id, dtype=np.int64)),
        "time": pa.array(np.asarray(fit.time, dtype=np.float64)),
        ROLE: _str(fit.role),
        "selected_K": pa.array(np.asarray(fit.selected_K, dtype=np.int64)),
        "obs_scale": pa.array(np.asarray(fit.obs_scale, dtype=np.float64)),
        "sample_size": pa.array(np.asarray(fit.sample_size, dtype=np.float64)),
        "n_max": pa.array(np.asarray(fit.n_max, dtype=np.float64)),
        "noise_kind": _str(fit.noise_kind),
        "score": _ragged(_rows(fit.score), pa.float64()),
    })}
    rows: dict[str, list] = {k: [] for k in (
        "measurement", "fit_K", "A", "n", "p", "log_A", "n_raw", "logit_p", "nll", "bic", "aic",
        "passes_min_pop")}
    for k in fit.k_values:
        f = fit.fits[k]
        rows["measurement"].append(np.arange(n, dtype=np.int64))
        rows["fit_K"].append(np.full(n, k, dtype=np.int64))
        for name in ("nll", "bic", "aic"):
            rows[name].append(np.asarray(getattr(f, name), dtype=np.float64))
        rows["passes_min_pop"].append(np.asarray(f.passes_min_pop, dtype=bool))
        for name in ("A", "n", "p", "log_A", "n_raw", "logit_p"):
            rows[name].extend(_rows(getattr(f, name)))
    cols: dict[str, pa.Array] = {}
    for name in ("measurement", "fit_K"):
        cols[name] = pa.array(np.concatenate(rows[name]))
    for name in ("A", "n", "p", "log_A", "n_raw", "logit_p"):
        cols[name] = _ragged(rows[name], pa.float64())
    for name in ("nll", "bic", "aic"):
        cols[name] = pa.array(np.concatenate(rows[name]))
    cols["passes_min_pop"] = pa.array(np.concatenate(rows["passes_min_pop"]))
    tables["fit_k"] = pa.table(cols)
    if fit.noise_estimate is not None:
        e = fit.noise_estimate
        tables["noise_estimate"] = pa.table({
            "measurement": pa.array(np.arange(n, dtype=np.int64)),
            "gain": pa.array(np.asarray(e.gain, dtype=np.float64)),
            "estimated": pa.array(np.asarray(e.estimated, dtype=bool)),
            "reason": _str(e.reason),
            "score": pa.array(np.asarray(e.score, dtype=np.float64)),
            "outlier": pa.array(np.asarray(e.outlier, dtype=bool)),
            "n_replicated": pa.array(np.asarray(e.n_replicated, dtype=np.int64)),
        })
    return tables


def _bootstrap_tables(b: BootstrapResult, k: int | None = None) -> dict[str, pa.Table]:
    """The tables of a bootstrap: ``bootstrap`` (and ``bootstrap_draws``), or, for a fixed-K
    result (``k`` given), ``bootstrap_k`` (and ``bootstrap_k_draws``) with a ``fit_K`` column."""
    n = len(b)
    suffix = "" if k is None else "_k"
    extra = {} if k is None else {"fit_K": pa.array(np.full(n, k, dtype=np.int64))}
    cols: dict[str, pa.Array] = {
        **extra,
        "measurement": pa.array(np.arange(n, dtype=np.int64)),
        "selected_K": pa.array(np.asarray(b.selected_K, dtype=np.int64)),
        "n_success": pa.array(np.asarray(b.n_success, dtype=np.int64)),
    }
    for name in ("A", "n", "p", "mean", "population"):
        cols[f"{name}_se"] = _ragged(_rows(getattr(b, f"{name}_se")), pa.float64())
        cols[f"{name}_ci"] = _ragged(_rows(getattr(b, f"{name}_ci")), pa.float64())
    cols["curve_lo"] = _flat_ragged(b.curve_lo, b.curve_offsets, pa.float64())
    cols["curve_hi"] = _flat_ragged(b.curve_hi, b.curve_offsets, pa.float64())
    if b.component_lo is not None:
        cols["component_lo"] = _flat_ragged(b.component_lo, b.component_offsets, pa.float64())
        cols["component_hi"] = _flat_ragged(b.component_hi, b.component_offsets, pa.float64())
    tables = {f"bootstrap{suffix}": pa.table(cols)}
    if b.draws is not None:
        d = b.draws
        tables[f"bootstrap{suffix}_draws"] = pa.table({
            **extra,
            "measurement": pa.array(np.arange(n, dtype=np.int64)),
            "A": _ragged(_rows(d.A), pa.float64()),
            "n": _ragged(_rows(d.n), pa.float64()),
            "p": _ragged(_rows(d.p), pa.float64()),
            "ok": _ragged(_rows(d.ok), pa.bool_()),
        })
    return tables


def _envelopes_table(e: EnvelopeTable) -> pa.Table:
    return pa.table({
        "data_id": pa.array(np.asarray(e.data_id, dtype=np.int64)),
        "mode": _str(e.mode), "source": _str(e.source),
        "weights": _ragged(_rows(e.weights), pa.float64()),
        "residual": pa.array(np.asarray(e.residual, dtype=np.float64)),
        "n_reference": pa.array(np.asarray(e.n_reference, dtype=np.int64)),
        "good": pa.array(np.asarray(e.good, dtype=bool)),
        "fit_shape": _flat_ragged(e.fit_shape, e.shape_offsets, pa.float64()),
        "own_residual": pa.array(np.asarray(e.own_residual, dtype=np.float64)),
        "agrees": pa.array(np.asarray(e.agrees, dtype=bool)),
    })


def _read_envelopes(table: pa.Table, config: dict) -> EnvelopeTable:
    cfg = EnvelopeConfig.from_dict(config)
    shape, offsets = _ragged_read(table, "fit_shape")
    return EnvelopeTable(
        config=cfg, data_id=np.asarray(_col(table, "data_id"), dtype=np.int64),
        mode=_strs(table, "mode"), source=_strs(table, "source"),
        weights=_matrix(table, "weights", max(cfg.components, 1)),
        residual=np.asarray(_col(table, "residual"), dtype=np.float64),
        n_reference=np.asarray(_col(table, "n_reference"), dtype=np.int64),
        good=np.asarray(_col(table, "good"), dtype=bool),
        fit_shape=np.asarray(shape, dtype=np.float64), shape_offsets=offsets,
        own_residual=(np.asarray(_col(table, "own_residual"), dtype=np.float64)
                      if "own_residual" in table.column_names else None),
        agrees=(np.asarray(_col(table, "agrees"), dtype=bool)
                if "agrees" in table.column_names else None),
    )


def _quality_table(q: QualityFlags) -> pa.Table:
    cols: dict[str, pa.Array] = {"measurement": pa.array(np.arange(len(q), dtype=np.int64))}
    for name in ("deviance", "deviance_rel", "misfit", "misfit_rel", "intensity",
                 "intensity_rel", "mz_offset", "envelope_offset"):
        cols[name] = pa.array(np.asarray(getattr(q, name), dtype=np.float64))
    for name in FLAG_NAMES:
        cols[name] = pa.array(np.asarray(getattr(q, name), dtype=bool))
    cols["flagged"] = pa.array(np.asarray(q.flagged, dtype=bool))
    return pa.table(cols)


def _read_quality(table: pa.Table, config: dict, spectrum_id: np.ndarray) -> QualityFlags:
    f = lambda name: np.asarray(_col(table, name), dtype=np.float64)  # noqa: E731
    b = lambda name: np.asarray(_col(table, name), dtype=bool)  # noqa: E731
    return QualityFlags(
        config=QualityConfig.from_dict(config), spectrum_id=spectrum_id,
        deviance=f("deviance"), deviance_rel=f("deviance_rel"), intensity=f("intensity"),
        intensity_rel=f("intensity_rel"), poor_fit=b("poor_fit"),
        low_intensity=b("low_intensity"), noise_outlier=b("noise_outlier"),
        # written before the m/z screen: left to the defaults (not assessed)
        mz_offset=f("mz_offset") if "mz_offset" in table.column_names else None,
        mz_shift=b("mz_shift") if "mz_shift" in table.column_names else None,
        # written before the misfit statistic: NaN
        misfit=f("misfit") if "misfit" in table.column_names else None,
        misfit_rel=f("misfit_rel") if "misfit_rel" in table.column_names else None,
        # written before the envelope flag: False
        poor_envelope=(b("poor_envelope") if "poor_envelope" in table.column_names else None),
        # written before the brightness flags: False
        **{name: (b(name) if name in table.column_names else None)
           for name in ("saturated", "tails_low", "bright")},
        # written before the m/z check: not assessed
        envelope_offset=(f("envelope_offset") if "envelope_offset" in table.column_names
                         else None),
        off_mz=b("off_mz") if "off_mz" in table.column_names else None,
    )


def _controls_table(c: ControlTable) -> pa.Table:
    cols: dict[str, pa.Array] = {"data_id": pa.array(c.data_id)}
    for name in ("n_amides", "m0_measured", "mfd_measured", "m0", "m0_se", "mfd", "mfd_se",
                 "retention"):
        cols[name] = pa.array(np.asarray(getattr(c, name), dtype=np.float64))
    cols["n_undeut"] = pa.array(np.asarray(c.n_undeut, dtype=np.int64))
    cols["n_fd"] = pa.array(np.asarray(c.n_fd, dtype=np.int64))
    cols["fallback"] = pa.array(np.asarray(c.fallback, dtype=bool))
    cols["flags"] = _str(c.flags)
    table = pa.table(cols)
    meta = {}
    if c.fd_model is not None:  # the fitted theoretical back-exchange (fd_fallback 'kint')
        meta[b"fd_model"] = json.dumps(dict(c.fd_model))
    if c.limits is not None:  # the run-relative retention limit
        meta[b"limits"] = json.dumps(dict(c.limits))
    if meta:
        table = table.replace_schema_metadata(meta)
    return table


def _read_controls(table: pa.Table, config: dict) -> ControlTable:
    f = lambda name: np.asarray(_col(table, name), dtype=np.float64)  # noqa: E731
    return ControlTable(
        config=ControlConfig.from_dict(config),
        data_id=np.asarray(_col(table, "data_id"), dtype=np.int64),
        n_amides=f("n_amides"),
        n_undeut=np.asarray(_col(table, "n_undeut"), dtype=np.int64),
        n_fd=np.asarray(_col(table, "n_fd"), dtype=np.int64),
        m0_measured=f("m0_measured"), mfd_measured=f("mfd_measured"), m0=f("m0"),
        m0_se=f("m0_se"), mfd=f("mfd"), mfd_se=f("mfd_se"), retention=f("retention"),
        fallback=np.asarray(_col(table, "fallback"), dtype=bool), flags=_strs(table, "flags"),
        fd_model=json.loads(meta[b"fd_model"]) if (meta := table.schema.metadata or {}).get(
            b"fd_model") else None,
        limits=json.loads(meta[b"limits"]) if meta.get(b"limits") else None,
    )


def _corrected_columns(
    data: dict[str, np.ndarray], boot: BootstrapResult | None, controls: ControlTable
) -> dict[str, np.ndarray]:
    """Back-exchange-corrected mean (and its intervals) for every row of a components table."""
    rows = len(data["measurement"])
    level = boot.config.ci / 100.0 if boot is not None else 0.95
    se = data.get("mean_se")
    ci = None
    if "mean_ci_lo" in data:
        ci = np.stack([data["mean_ci_lo"], data["mean_ci_hi"]], axis=-1)
    out = correct_means(controls, data["data_id"], data["mean"], se=se, ci=ci, ci_level=level)
    cols = {"mean_corrected": out.mean, "mean_corrected_control_se": out.control_se}
    if out.se is not None:
        cols["mean_corrected_se"] = out.se
    if out.ci is not None:
        cols["mean_corrected_ci_lo"], cols["mean_corrected_ci_hi"] = out.ci[:, 0], out.ci[:, 1]
    rows_ctrl = controls.rows(data["data_id"])
    raw = controls.uncorrected[rows_ctrl]  # fragment ions: not corrected
    for name in cols:
        cols[name] = np.where(raw, np.nan, cols[name])
    cols["control_fallback"] = controls.fallback[rows_ctrl]
    assert all(len(v) == rows for v in cols.values())
    return cols


def _components_table(
    fit: FitResult,
    boot: BootstrapResult | None,
    controls: ControlTable | None = None,
    by_k: Mapping[int, BootstrapResult] | None = None,
) -> pa.Table:
    """The tidy table of every fit (and its intervals), for other tools; derived, not read back.
    With controls it also has the back-exchange-corrected means."""
    results = ([boot] if boot is not None else []) + list((by_k or {}).values())
    data = interval_table(fit, results) if results else fit.table()
    if controls is not None:
        data = {**data, **_corrected_columns(data, boot, controls)}
    cols = {}
    for name, values in data.items():
        if values.dtype.kind in "US" or values.dtype == object:
            cols[name] = _str(values)
        else:
            cols[name] = pa.array(values)
    return pa.table(cols)


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("hxex", "numpy", "scipy", "torch", "pyarrow", "pyteomics"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_table(directory: Path, name: str, table: pa.Table, files: dict, **extra) -> None:
    """Write ``table`` as ``name.parquet`` in ``directory`` (through a temporary file, so that
    a table replaced in place, like the review layer, is whole or untouched) and record it."""
    path = directory / f"{name}.parquet"
    tmp = directory / f"{name}.parquet.tmp"
    try:
        pq.write_table(table, tmp, compression="zstd", row_group_size=ROW_GROUP)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    files[f"{name}.parquet"] = {
        "rows": table.num_rows, "bytes": path.stat().st_size, "sha256": _sha256(path), **extra,
    }


# the manifest of a bundle directory while its files are being replaced: present only when
# that replacement was interrupted (the directory is then not a bundle until rewritten)
REPLACING = MANIFEST + ".replacing"


def _old_manifest(path: Path) -> dict | None:
    """The manifest of the bundle directory ``path`` (or of its interrupted replacement)."""
    for name in (MANIFEST, REPLACING):
        try:
            manifest = json.loads((path / name).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(manifest, dict) and manifest.get("kind") == KIND:
            return manifest
    return None


def _is_bundle_dir(path: Path) -> bool:
    return _old_manifest(path) is not None


def _is_bundle_zip(path: Path) -> bool:
    """Whether the file ``path`` is a zipped results bundle (a zip with a bundle manifest)."""
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read(_zip_root(zf) + MANIFEST).decode())
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return False
    return isinstance(manifest, dict) and manifest.get("kind") == KIND


def _is_bundle(path: Path) -> bool:
    """Whether ``path`` is a results bundle (a directory or a zip) that may be replaced."""
    if path.is_dir():
        return _is_bundle_dir(path)
    return path.is_file() and _is_bundle_zip(path)


def bundle_manifest(path: str | Path) -> dict | None:
    """The manifest of the bundle at ``path`` (a directory, its interrupted replacement, or a
    zip), or None when there is no bundle there."""
    path = Path(path)
    if path.is_dir():
        return _old_manifest(path)
    if path.is_file() and zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path) as zf:
                manifest = json.loads(zf.read(_zip_root(zf) + MANIFEST).decode())
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            return None
        if isinstance(manifest, dict) and manifest.get("kind") == KIND:
            return manifest
    return None


def settings_changed(old: Mapping | None, new: Mapping | None) -> list[str]:
    """The dotted paths of the settings that differ between the ``configs`` of two manifests:
    the run settings leaf by leaf (``fit.k_max``), a stage's settings where a bundle has no
    run settings, and ``spectra_source`` as one item (which files were read, how)."""
    old, new = dict(old or {}), dict(new or {})
    changed: list[str] = []
    if old.get("run") is not None and new.get("run") is not None:
        changed += _diff(old["run"], new["run"], "")
    else:
        for key in ("fit", "bootstrap", "picker", "controls", "quality", "envelope"):
            changed += _diff(old.get(key), new.get(key), key)
    if _without_stamps(old.get("spectra_source")) != _without_stamps(new.get("spectra_source")):
        changed.append("spectra_source")
    return sorted(changed)


def _diff(old, new, prefix: str) -> list[str]:
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        out: list[str] = []
        for key in sorted(set(old) | set(new), key=str):
            name = f"{prefix}.{key}" if prefix else str(key)
            out += _diff(old.get(key), new.get(key), name)
        return out
    return [] if old == new else [prefix or "(all)"]


def _without_stamps(value):
    """``value`` with the keys that record when something was read or written removed, so
    that reading the same files again does not count as a change."""
    if isinstance(value, Mapping):
        return {k: _without_stamps(v) for k, v in value.items()
                if not str(k).endswith(("created", "read", "written"))}
    if isinstance(value, list):
        return [_without_stamps(v) for v in value]
    return value


def provenance_of(
    old: Mapping,
    path: str | Path,
    *,
    new_configs: Mapping | None,
    review: ReviewTable | None,
    moved_to: str | Path | None = None,
) -> dict[str, Any]:
    """The provenance block of a bundle replacing the one whose manifest is ``old`` at
    ``path``: when it was replaced, where it went (``moved_to``, with ``keep_previous``), which
    settings changed (:func:`settings_changed`) and how many review entries were carried and
    are stale. ``new_configs`` is the new bundle's ``configs`` (the run settings and the
    spectra source are what is compared)."""
    return {
        "replaced": str(path),
        "replaced_created": old.get("created"),
        "replaced_hxex": (old.get("versions") or {}).get("hxex"),
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "moved_to": None if moved_to is None else str(moved_to),
        "settings_changed": settings_changed(old.get("configs"), new_configs),
        "review": {"carried": 0 if review is None else len(review),
                   "stale": 0 if review is None else review.n_stale},
    }


# --- generations: a bundle kept beside its replacement ---------------------------------------

SIBLINGS = ("_tables", "_figures", "_analysis")  # the output folders named after a bundle


def analysis_folder_name(name: str, reference: str = "") -> str:
    """The default analysis folder of the bundle ``name``: ``name_analysis``, with the reference
    the samples are compared with when there is one (``name_analysis_WT``), so analyses against
    different references never write into one folder. Characters a folder name should not hold
    become ``_`` (``A:WT`` gives ``name_analysis_A_WT``)."""
    if not reference:
        return f"{name}_analysis"
    return f"{name}_analysis_{re.sub(r'[^A-Za-z0-9._-]+', '_', reference).strip('_') or 'ref'}"


def analysis_folders(path: Path) -> list[Path]:
    """The analysis folders named after the bundle at ``path``: ``name_analysis`` and every
    ``name_analysis_REFERENCE``."""
    path = Path(path)
    if not path.parent.is_dir():
        return []
    return sorted(p for p in path.parent.iterdir() if p.is_dir() and (
        p.name == f"{path.name}_analysis" or p.name.startswith(f"{path.name}_analysis_")))


def analysis_states(path: str | Path) -> list[tuple[Path, str | None]]:
    """Whether each analysis folder of the bundle at ``path`` (:func:`analysis_folders`) was
    made from the results there now: ``(folder, state)`` with state None (it was), the
    ``created`` stamp of the results it was made from (earlier ones: ``hxex analyse`` brings
    it up to date), or ``""`` (not known: written before hxex recorded it)."""
    from .outputs import made_from

    path = Path(path)
    created = (bundle_manifest(path) or {}).get("created", "")
    out = []
    for folder in analysis_folders(path):
        made = made_from(folder)
        stamp = None if made is None else made.get("")
        out.append((folder, "" if stamp is None or not stamp
                    else None if stamp == created else stamp))
    return out


def _generation_stem(path: Path) -> tuple[str, str]:
    """``(stem, ext)`` so that generation N of the bundle is ``stem.N`` + ``ext``."""
    return (path.stem, ".zip") if path.suffix == ".zip" else (path.name, "")


def generations(path: str | Path) -> list[tuple[int, Path]]:
    """The earlier generations kept beside the bundle ``path`` (``name.N``, or ``name.N.zip``
    for a zip; see :func:`keep_previous`), oldest first."""
    path = Path(path)
    stem, ext = _generation_stem(path)
    found: dict[int, Path] = {}
    if path.parent.is_dir():
        for item in path.parent.iterdir():
            m = re.fullmatch(re.escape(stem) + r"\.(\d+)" + re.escape(ext), item.name)
            if m and _is_bundle(item):
                found[int(m.group(1))] = item
    return sorted(found.items())


def _next_generation(path: Path) -> int:
    """One more than every number ever used beside ``path`` (``name.N`` and anything that
    starts like it: its folders, a bundle that was deleted but whose tables remain)."""
    stem, _ = _generation_stem(path)
    used = [0]
    for item in path.parent.iterdir():
        m = re.match(re.escape(stem) + r"\.(\d+)(\.|_|$)", item.name)
        if m:
            used.append(int(m.group(1)))
    return max(used) + 1


def _size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def keep_previous(path: str | Path) -> Path | None:
    """Move the bundle at ``path`` to its next generation, ``name.N`` (``name.N.zip`` for a
    zip), with its ``name_tables``, ``name_figures``, and analysis folders (``name_analysis``,
    ``name_analysis_REFERENCE``) moved to ``name.N_tables``, ``name.N_figures``, and
    ``name.N_analysis...``, and return the generation's
    path (None when there is nothing at ``path``). N is one more than any number used beside the
    bundle before, so a generation's name is never reused. A path that holds something other
    than a bundle is refused. Sizes are logged."""
    path = Path(path)
    if not path.exists():
        return None
    if not _is_bundle(path):
        raise BundleError(f"{path} exists but is not a results bundle; not moving it")
    stem, ext = _generation_stem(path)
    n = _next_generation(path)
    target = path.with_name(f"{stem}.{n}{ext}")
    moves = [(path, target)]
    for suffix in SIBLINGS[:-1]:
        sibling = path.with_name(path.name + suffix)
        if sibling.exists():
            moves.append((sibling, target.with_name(target.name + suffix)))
    for folder in analysis_folders(path):  # name_analysis and name_analysis_REFERENCE
        moves.append((folder, target.with_name(target.name + folder.name[len(path.name):])))
    for src, dst in moves:
        if dst.exists():  # cannot happen (N is new); never replace anything
            raise BundleError(f"{dst} exists; not moving {src} over it")
    done = []
    for src, dst in moves:
        size = _size_of(src)
        os.replace(src, dst)
        done.append(f"{src.name} ({size / 1e6:.1f} MB) to {dst.name}")
    log.info("kept the previous results: moved %s", "; ".join(done))
    return target


def write_bundle(
    path: str | Path,
    *,
    measurements: Measurements,
    fit: FitResult,
    bootstrap: BootstrapResult | None = None,
    bootstrap_by_k: Mapping[int, BootstrapResult] | None = None,
    binned: BinnedSet | None = None,
    spectra: SpectrumSet | None = None,
    picker_config: PickerConfig | None = None,
    review: ReviewTable | None = None,
    controls: ControlTable | None = None,
    quality: QualityFlags | None = None,
    envelopes: EnvelopeTable | None = None,
    run_config: dict | None = None,
    spectra_source: Mapping | None = None,
    project: str | None = None,
    notes: str = "",
    pending: Sequence[str] = (),
    overwrite: bool = False,
    provenance: Mapping | None = None,
    excluded: Mapping | None = None,
    project_file: Mapping | None = None,
) -> Path:
    """Write a results bundle and return its path.

    ``path`` ending in ``.zip`` gives a zip (read only); anything else a directory. The
    bundle is written next to its final place and moved there when complete, so a failed write
    leaves nothing behind. An existing bundle is only replaced with ``overwrite=True`` (a
    directory that is not a bundle is never touched). A bundle directory is replaced file by
    file: the files its manifest lists go, the new ones come in and the manifest is written
    last; anything else in the directory (notes of your own) stays. A file of yours named like
    one of the new bundle's files is refused before anything changes. An interrupted
    replacement leaves a directory that does not open (its manifest is
    ``manifest.json.replacing``) until it is written again. ``project`` is recorded in the manifest.
    ``spectra_source`` (how the raw spectra were read, see
    :attr:`~hxex.spectra.SpectrumSet.source`; by default that of ``spectra``) is recorded in the
    manifest whether or not the spectra are stored. ``pending`` names the stages the run has
    still to add (``'bootstrap'`` for the bundle a run writes before its bootstrap, see
    :func:`hxex.run.resume_bootstrap`). ``provenance`` is recorded as given (see
    :func:`provenance_of`: the bundle this one replaces, what changed, the review carried), and
    ``excluded`` too (the run-level exclusions: :func:`hxex.exclusions.exclusion_record`).
    ``project_file`` is the project file's other settings, as used
    (:func:`hxex.config.project_record`; None for a run from Python). Beside the manifest, the
    bundle holds ``run_settings_used.toml``, every setting of the run written out for people
    (:func:`hxex.config.run_settings_text`); it is replaced with the bundle and not read back.
    """
    path = Path(path)
    unknown = set(pending) - set(PENDING_STAGES)
    if unknown:
        raise BundleError(f"unknown pending stage(s): {', '.join(sorted(unknown))}")
    if spectra_source is None and spectra is not None:
        spectra_source = spectra.source
    if len(measurements) != len(fit) or not np.array_equal(
        measurements.spectrum_id, fit.spectrum_id
    ):
        raise BundleError("the measurements and the fit must be for the same spectra")
    if bootstrap is not None and not np.array_equal(bootstrap.selected_K, fit.selected_K):
        raise BundleError("the bootstrap was not made from this fit")
    for k, b in (bootstrap_by_k or {}).items():
        if k not in fit.fits or len(b) != len(fit) or not (b.selected_K == k).all():
            raise BundleError(f"bootstrap_by_k[{k}] is not a fixed-K bootstrap of this fit")
    if picker_config is None:
        log.warning("no picker settings were given: the bundle will not record how the spectra "
                    "were picked (plots cannot show the picking window)")
    if run_config is None:
        log.warning("no run settings were given: the bundle will record the stage settings but "
                    "not the settings of the run as a whole")
    if binned is not None and not set(measurements.spectrum_id) <= set(binned.spectrum_id):
        raise BundleError("every measurement must be in the binned table")
    if path.exists():
        if not overwrite:
            raise BundleError(f"{path} exists (use overwrite=True to replace a bundle)")
        if not _is_bundle(path):
            raise BundleError(f"{path} exists but is not a results bundle; not removing it")
        if path.is_dir() and path.suffix == ".zip":
            raise BundleError(f"{path} is a bundle directory; it cannot be replaced by a zip "
                              "(remove it first)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=path.name + ".tmp-", dir=path.parent))
    try:
        files: dict[str, dict] = {}
        _write_table(tmp, "measurements", _measurements_table(measurements), files)
        for name, table in _fit_tables(fit).items():
            _write_table(tmp, name, table, files)
        if bootstrap is not None:
            for name, table in _bootstrap_tables(bootstrap).items():
                _write_table(tmp, name, table, files)
        if bootstrap_by_k:
            by_table: dict[str, list[pa.Table]] = {}
            for k in sorted(bootstrap_by_k):
                for name, table in _bootstrap_tables(bootstrap_by_k[k], k).items():
                    by_table.setdefault(name, []).append(table)
            for name, parts in by_table.items():
                _write_table(tmp, name, pa.concat_tables(parts), files)
        if binned is not None:
            _write_table(tmp, "binned", _binned_table(binned), files)
        if spectra is not None:
            _write_table(tmp, "spectra", _spectra_table(spectra), files)
        if quality is not None:
            if not np.array_equal(quality.spectrum_id, fit.spectrum_id):
                raise BundleError("the quality flags were not made from this fit")
            _write_table(tmp, "quality", _quality_table(quality), files)
        if envelopes is not None:
            _write_table(tmp, "envelopes", _envelopes_table(envelopes), files)
        if controls is not None:
            _write_table(tmp, "controls", _controls_table(controls), files)
        components = _components_table(fit, bootstrap, controls, bootstrap_by_k)
        _write_table(tmp, "components", components, files, derived=True)
        if review is not None:
            _check_review(review, fit)
            _write_table(tmp, "review", _review_table(review), files)
        manifest = {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "project": project if project is not None else _project_of(fit.spectrum_id),
            "notes": notes,
            "versions": _versions(),
            "k_values": list(fit.k_values),
            "n_measurements": len(fit),
            "pending": list(pending),
            "provenance": None if provenance is None else dict(provenance),
            **({} if not excluded else {"excluded": dict(excluded)}),
            "configs": {
                "fit": fit.config.to_dict(),
                "bootstrap": None if bootstrap is None else bootstrap.config.to_dict(),
                "picker": None if picker_config is None else picker_config.to_dict(),
                "controls": None if controls is None else controls.config.to_dict(),
                "quality": None if quality is None else quality.config.to_dict(),
                "envelope": None if envelopes is None else envelopes.config.to_dict(),
                "run": run_config,
                "spectra_source": None if spectra_source is None else dict(spectra_source),
                "project_file": None if project_file is None else dict(project_file),
            },
            "files": files,
        }
        from ..config import RUN_SETTINGS_FILE, run_settings_text

        (tmp / RUN_SETTINGS_FILE).write_text(run_settings_text(manifest), encoding="utf-8")
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        _publish(tmp, path, overwrite)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return path


def _project_of(ids) -> str:
    heads = {str(i).partition(":")[0] for i in ids if ":" in str(i)}
    return heads.pop() if len(heads) == 1 else ""


def _publish(tmp: Path, path: Path, overwrite: bool) -> None:
    """Move the finished bundle to ``path`` (as a directory, or zipped if ``path`` is .zip)."""
    if path.suffix == ".zip":
        if path.exists() and not (path.is_file() and _is_bundle_zip(path)):
            raise BundleError(f"{path} exists but is not a zipped results bundle; not removing it")
        archive = tmp.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
            for item in sorted(tmp.iterdir()):
                zf.write(item, item.name)
        shutil.rmtree(tmp)
        if path.exists():
            path.unlink()
        os.replace(archive, path)
        return
    if not path.exists():
        os.replace(tmp, path)
        return
    if path.is_file():
        if not _is_bundle_zip(path):  # checked by write_bundle; never remove what is not a bundle
            raise BundleError(f"{path} exists but is not a results bundle; not removing it")
        path.unlink()
        os.replace(tmp, path)
        return
    old = _old_manifest(path)
    if old is None:  # checked by write_bundle; never remove what is not a bundle
        raise BundleError(f"{path} is a directory but not a results bundle; not removing it")
    from ..config import RUN_SETTINGS_FILE

    owned = set(old.get("files", {})) | {MANIFEST, REPLACING, RUN_SETTINGS_FILE}
    new = [item.name for item in tmp.iterdir() if item.name != MANIFEST]
    clash = sorted(n for n in new if n not in owned and (path / n).exists())
    if clash:
        raise BundleError(f"{path} holds file(s) of its own named like the bundle's "
                          f"({', '.join(clash)}); move them and write again")
    if (path / MANIFEST).exists():
        os.replace(path / MANIFEST, path / REPLACING)
    for name in owned - {MANIFEST, REPLACING}:
        (path / name).unlink(missing_ok=True)
    for name in new:
        os.replace(tmp / name, path / name)
    os.replace(tmp / MANIFEST, path / MANIFEST)
    (path / REPLACING).unlink(missing_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)


# --- the review layer -----------------------------------------------------------------------

@dataclass(frozen=True)
class ReviewTable:
    """Per-spectrum review: ``status`` (``accepted`` or ``rejected``), a chosen ``k_override``
    (0 = none) and a ``note``, with the time of the last change. Spectra not listed are
    unreviewed. It changes nothing in the fit: :meth:`effective_K` and :meth:`status_of` give
    the reviewed view. ``stale`` marks an entry a rerun could not apply (its spectrum was
    dropped, or its chosen K was not fitted): the status and note are kept on record, the
    chosen K is ignored while stale, and :meth:`update` makes the entry current again (it is
    then checked like any other when stored)."""

    spectrum_id: np.ndarray
    status: np.ndarray
    k_override: np.ndarray
    note: np.ndarray
    updated: np.ndarray
    stale: np.ndarray | None = None
    # where an entry came from, in memory only (not stored): "review" (the bundle's), or
    # "review table" (hxex.review_table); None: every entry is the bundle's
    source: np.ndarray | None = None

    def __post_init__(self) -> None:
        stale = (np.zeros(len(self.spectrum_id), dtype=bool) if self.stale is None
                 else np.asarray(self.stale, dtype=bool))
        if stale.shape != (len(self.spectrum_id),):
            raise ValueError("stale must have one value per entry")
        object.__setattr__(self, "stale", stale)
        if self.source is not None and len(self.source) != len(self.spectrum_id):
            raise ValueError("source must have one value per entry")

    @classmethod
    def empty(cls) -> ReviewTable:
        z = np.array([], dtype=str)
        return cls(z, z.copy(), np.array([], dtype=np.int64), z.copy(), z.copy())

    def __len__(self) -> int:
        return len(self.spectrum_id)

    @property
    def n_stale(self) -> int:
        return int(self.stale.sum())

    def get(self, spectrum_id: str) -> dict[str, Any] | None:
        hits = np.flatnonzero(self.spectrum_id == spectrum_id)
        if len(hits) == 0:
            return None
        i = hits[0]
        return {"status": str(self.status[i]), "k_override": int(self.k_override[i]),
                "note": str(self.note[i]), "updated": str(self.updated[i]),
                "stale": bool(self.stale[i])}

    def with_stale(self, stale: np.ndarray) -> ReviewTable:
        """The same entries with ``stale`` (one bool per entry) in place of the marker."""
        return dataclasses.replace(self, stale=np.asarray(stale, dtype=bool))

    def mark_stale(self, fit: FitResult) -> ReviewTable:
        """The entries ``fit`` cannot hold (a spectrum it does not have, a chosen K it did not
        fit) marked stale and the others current, so the table can be stored with ``fit``."""
        known = set(map(str, fit.spectrum_id))
        stale = [str(s) not in known or (int(k) != 0 and int(k) not in fit.k_values)
                 for s, k in zip(self.spectrum_id, self.k_override, strict=True)]
        return self.with_stale(np.array(stale, dtype=bool))

    def update(
        self,
        spectrum_id: str,
        *,
        status: str | None = None,
        k_override: int | None = None,
        note: str | None = None,
        now: datetime | None = None,
    ) -> ReviewTable:
        """A new table with the entry for ``spectrum_id`` changed (created if missing). Only the
        arguments given are changed; ``status=''`` and ``k_override=0`` clear them."""
        if status not in (None, "", *STATUSES):
            raise ValueError(f"status must be one of {STATUSES} (or '' to clear)")
        if k_override is not None and k_override < 0:
            raise ValueError("k_override must be 0 (none) or a positive K")
        stamp = (now or datetime.now(UTC)).isoformat(timespec="seconds")
        old = self.get(spectrum_id) or {"status": "", "k_override": 0, "note": ""}
        new = {
            "status": old["status"] if status is None else status,
            "k_override": old["k_override"] if k_override is None else int(k_override),
            "note": old["note"] if note is None else note,
        }
        keep = self.spectrum_id != spectrum_id
        if not new["status"] and not new["k_override"] and not new["note"]:
            return ReviewTable(self.spectrum_id[keep], self.status[keep], self.k_override[keep],
                               self.note[keep], self.updated[keep], self.stale[keep])
        return ReviewTable(
            np.append(self.spectrum_id[keep], spectrum_id).astype(str),
            np.append(self.status[keep], new["status"]).astype(str),
            np.append(self.k_override[keep], new["k_override"]).astype(np.int64),
            np.append(self.note[keep], new["note"]).astype(str),
            np.append(self.updated[keep], stamp).astype(str),
            np.append(self.stale[keep], False),  # a change makes the entry current again
        )

    def effective_K(self, fit: FitResult) -> np.ndarray:
        """The K of every measurement with the reviewer's choices applied (stale ones not)."""
        out = fit.selected_K.copy()
        index = {str(s): i for i, s in enumerate(fit.spectrum_id)}
        for sid, k, stale in zip(self.spectrum_id, self.k_override, self.stale, strict=True):
            if k and not stale:
                if int(k) not in fit.k_values:
                    raise ValueError(f"{sid}: K = {int(k)} was not fitted (tried {fit.k_values})")
                fit.require_fitted(index[str(sid)], int(k))
                out[index[str(sid)]] = int(k)
        return out

    def status_of(self, fit: FitResult) -> np.ndarray:
        """``accepted``, ``rejected`` or ``unreviewed`` for every measurement (a stale entry
        still counts when its spectrum is in ``fit``)."""
        out = np.full(len(fit), "unreviewed", dtype=object)
        index = {str(s): i for i, s in enumerate(fit.spectrum_id)}
        for sid, status in zip(self.spectrum_id, self.status, strict=True):
            if status and str(sid) in index:
                out[index[str(sid)]] = str(status)
        return out.astype(str)


def _check_review(review: ReviewTable, fit: FitResult) -> None:
    """Refuse a review layer that ``fit`` cannot hold: current entries for spectra it does not
    have (:class:`BundleError`) or with a chosen K it did not fit (ValueError); stale entries
    are on record only and pass. One check for the review given at writing and for
    :meth:`Bundle.set_review`."""
    known = set(map(str, fit.spectrum_id))
    unknown = [str(s) for s, stale in zip(review.spectrum_id, review.stale, strict=True)
               if not stale and str(s) not in known]
    if unknown:
        raise BundleError(f"review entries for unknown spectra: {unknown[:3]}")
    review.effective_K(fit)  # raises if a chosen K was not fitted


def _review_table(r: ReviewTable) -> pa.Table:
    return pa.table({
        "spectrum_id": _str(r.spectrum_id), "status": _str(r.status),
        "k_override": pa.array(np.asarray(r.k_override, dtype=np.int64)),
        "note": _str(r.note), "updated": _str(r.updated),
        "stale": pa.array(np.asarray(r.stale, dtype=bool)),
    })


def _read_review(table: pa.Table) -> ReviewTable:
    stale = (np.asarray(_col(table, "stale"), dtype=bool) if "stale" in table.column_names
             else None)  # bundles written before the marker: every entry current
    return ReviewTable(_strs(table, "spectrum_id"), _strs(table, "status"),
                       np.asarray(_col(table, "k_override"), dtype=np.int64),
                       _strs(table, "note"), _strs(table, "updated"), stale)


# --- reading --------------------------------------------------------------------------------

class Bundle:
    """An open results bundle. Tables are read when asked for (``spectra_by_id`` reads only the
    rows needed), and the objects returned are the ones the run produced."""

    def __init__(self, path: Path, store, manifest: dict[str, Any]):
        self.path, self._store = path, store
        self.manifest, self._legacy_times = modernise_manifest(manifest)
        self._cache: dict[str, Any] = {}

    @property
    def project(self) -> str:
        return self.manifest.get("project", "")

    @property
    def schema_version(self) -> int:
        return int(self.manifest["schema_version"])

    def _roles(self, t: pa.Table) -> np.ndarray:
        """The ``role`` column of a per-spectrum table, or, for a bundle written before the
        roles, the roles its control times imply (:func:`legacy_roles`)."""
        if ROLE in t.column_names:
            return _strs(t, ROLE)
        return legacy_roles(np.asarray(_col(t, "time"), dtype=np.float64), self._legacy_times)

    @property
    def pending(self) -> list[str]:
        """The stages the run has still to add (``['bootstrap']`` for a bundle written before
        the bootstrap, which :func:`hxex.run.resume_bootstrap` completes); empty when complete."""
        return list(self.manifest.get("pending", []))

    def has(self, table: str) -> bool:
        return f"{table}.parquet" in self.manifest["files"]

    def tables(self) -> list[str]:
        return sorted(n[: -len(".parquet")] for n in self.manifest["files"])

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> Bundle:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _table(self, name: str, **kw) -> pa.Table:
        if not self.has(name):
            raise BundleError(f"this bundle has no {name!r} table")
        return self._store.table(f"{name}.parquet", **kw)

    def verify(self) -> None:
        """Check the size and SHA-256 of every file against the manifest."""
        for name, info in self.manifest["files"].items():
            if not self._store.exists(name):
                raise BundleError(f"{name} is listed in the manifest but missing")
            data = self._store.read(name)
            if hashlib.sha256(data).hexdigest() != info["sha256"] or len(data) != info["bytes"]:
                raise BundleError(f"{name} does not match its checksum: the bundle is damaged")

    def measurements(self) -> Measurements:
        if "measurements" not in self._cache:
            t = self._table("measurements")
            bins = _split(*_ragged_read(t, "bins"))
            observed = _split(*_ragged_read(t, "observed"))
            shapes = _split(*_ragged_read(t, "shape"))
            ref_flat, ref_off = _ragged_read(t, "references")
            ref_null = np.asarray(t.column("references").is_null().to_numpy(zero_copy_only=False))
            refs = tuple(
                None if ref_null[i] else tuple(int(x) for x in ref_flat[ref_off[i]:ref_off[i + 1]])
                for i in range(t.num_rows)
            )
            self._cache["measurements"] = Measurements(
                bins=tuple(bins), observed=tuple(observed), shapes=tuple(shapes),
                n_max=np.asarray(_col(t, "n_max"), dtype=np.float64), references=refs,
                spectrum_id=_strs(t, "spectrum_id"),
                data_id=np.asarray(_col(t, "data_id"), dtype=np.int64),
                time=np.asarray(_col(t, "time"), dtype=np.float64),
                role=self._roles(t),
            )
        return self._cache["measurements"]

    def fit(self) -> FitResult:
        if "fit" not in self._cache:
            t = self._table("fit")
            m = t.num_rows
            k_values = tuple(int(k) for k in self.manifest["k_values"])
            fk = self._table("fit_k")
            kcol = np.asarray(_col(fk, "fit_K"), dtype=np.int64)
            fits = {}
            for k in k_values:
                sel = np.flatnonzero(kcol == k)
                sub = fk.take(pa.array(sel))
                order = np.argsort(np.asarray(_col(sub, "measurement")))
                sub = sub.take(pa.array(order))
                fits[k] = KFit(
                    k=k, A=_matrix(sub, "A", k), n=_matrix(sub, "n", k), p=_matrix(sub, "p", k),
                    log_A=_matrix(sub, "log_A", k), n_raw=_matrix(sub, "n_raw", k),
                    logit_p=_matrix(sub, "logit_p", k),
                    nll=np.asarray(_col(sub, "nll"), dtype=np.float64),
                    bic=np.asarray(_col(sub, "bic"), dtype=np.float64),
                    aic=np.asarray(_col(sub, "aic"), dtype=np.float64),
                    passes_min_pop=np.asarray(_col(sub, "passes_min_pop"), dtype=bool),
                )
            estimate = None
            if self.has("noise_estimate"):
                e = self._table("noise_estimate")
                estimate = NoiseEstimate(
                    gain=np.asarray(_col(e, "gain"), dtype=np.float64),
                    estimated=np.asarray(_col(e, "estimated"), dtype=bool),
                    reason=_strs(e, "reason"),
                    score=np.asarray(_col(e, "score"), dtype=np.float64),
                    outlier=np.asarray(_col(e, "outlier"), dtype=bool),
                    n_replicated=np.asarray(_col(e, "n_replicated"), dtype=np.int64),
                )
            self._cache["fit"] = FitResult(
                config=FitConfig.from_dict(self.manifest["configs"]["fit"]),
                spectrum_id=_strs(t, "spectrum_id"),
                data_id=np.asarray(_col(t, "data_id"), dtype=np.int64),
                time=np.asarray(_col(t, "time"), dtype=np.float64),
                role=self._roles(t),
                k_values=k_values, fits=fits,
                selected_K=np.asarray(_col(t, "selected_K"), dtype=np.int64),
                score=_matrix(t, "score", len(k_values)),
                obs_scale=np.asarray(_col(t, "obs_scale"), dtype=np.float64),
                sample_size=np.asarray(_col(t, "sample_size"), dtype=np.float64),
                n_max=np.asarray(_col(t, "n_max"), dtype=np.float64),
                noise_kind=_strs(t, "noise_kind"), noise_estimate=estimate,
            )
            assert m == len(self._cache["fit"])
        return self._cache["fit"]

    def bootstrap_ks(self) -> list[int]:
        """The K values that have a fixed-K bootstrap (``bootstrap(k)``)."""
        if not self.has("bootstrap_k"):
            return []
        return sorted(set(int(k) for k in _col(self._table("bootstrap_k"), "fit_K")))

    def bootstrap(self, k: int | None = None) -> BootstrapResult | None:
        """The bootstrap at each measurement's selected K, or, with ``k``, the fixed-K
        bootstrap of every measurement at that K; None if there is none."""
        name = "bootstrap" if k is None else "bootstrap_k"
        if not self.has(name) or (k is not None and k not in self.bootstrap_ks()):
            return None
        if (name, k) in self._cache:
            return self._cache[name, k]
        k_max = max(self.manifest["k_values"])
        cfg = BootstrapConfig.from_dict(self.manifest["configs"]["bootstrap"])

        def part(table_name: str) -> pa.Table:
            t = self._table(table_name)
            if k is None:
                return t
            t = t.take(pa.array(np.flatnonzero(np.asarray(_col(t, "fit_K")) == k)))
            return t.take(pa.array(np.argsort(np.asarray(_col(t, "measurement")))))

        t = part(name)
        arrays: dict[str, np.ndarray] = {}
        for stat in ("A", "n", "p", "mean", "population"):
            # schema 1 called the population "frac"
            col = stat if stat != "population" or "population_se" in t.column_names else "frac"
            arrays[f"{stat}_se"] = _matrix(t, f"{col}_se", k_max)
            arrays[f"{stat}_ci"] = _matrix(t, f"{col}_ci", 2 * k_max).reshape(-1, k_max, 2)
        lo, offsets = _ragged_read(t, "curve_lo")
        hi, _ = _ragged_read(t, "curve_hi")
        comp: dict[str, np.ndarray] = {}
        if "component_lo" in t.column_names:  # absent in bundles written before they were stored
            clo, coff = _ragged_read(t, "component_lo")
            chi, _ = _ragged_read(t, "component_hi")
            comp = {"component_lo": np.asarray(clo, dtype=np.float64),
                    "component_hi": np.asarray(chi, dtype=np.float64), "component_offsets": coff}
        draws = None
        if self.has(f"{name}_draws"):
            d = part(f"{name}_draws")
            m = d.num_rows
            draws = BootstrapDraws(
                A=_matrix(d, "A", cfg.n_boot * k_max).reshape(m, cfg.n_boot, k_max),
                n=_matrix(d, "n", cfg.n_boot * k_max).reshape(m, cfg.n_boot, k_max),
                p=_matrix(d, "p", cfg.n_boot * k_max).reshape(m, cfg.n_boot, k_max),
                ok=_matrix(d, "ok", cfg.n_boot, dtype=bool),
            )
        result = BootstrapResult(
            config=cfg, selected_K=np.asarray(_col(t, "selected_K"), dtype=np.int64),
            n_success=np.asarray(_col(t, "n_success"), dtype=np.int64),
            curve_lo=np.asarray(lo, dtype=np.float64), curve_hi=np.asarray(hi, dtype=np.float64),
            curve_offsets=offsets, draws=draws, **comp, **arrays,
        )
        self._cache[name, k] = result
        return result

    def binned(self) -> BinnedSet | None:
        if not self.has("binned"):
            return None
        t = self._table("binned")
        mz, offsets = _ragged_read(t, "mz")
        intensity, _ = _ragged_read(t, "intensity")
        n_deut, _ = _ragged_read(t, "n_deut")
        # bundles written before the picker recorded peak positions have neither column
        peak_mz = (np.asarray(_ragged_read(t, "peak_mz")[0], dtype=np.float64)
                   if "peak_mz" in t.column_names else None)
        offset = (np.asarray(_col(t, "mz_offset_ppm"), dtype=np.float64)
                  if "mz_offset_ppm" in t.column_names else None)
        return BinnedSet(
            mz=np.asarray(mz, dtype=np.float64), intensity=np.asarray(intensity, dtype=np.float64),
            n_deut=np.asarray(n_deut, dtype=np.int64), offsets=offsets,
            env_width=np.asarray(_col(t, "env_width"), dtype=np.float64),
            env_symm=np.asarray(_col(t, "env_symm"), dtype=np.float64),
            max_namides=np.asarray(_col(t, "max_namides"), dtype=np.int64),
            threshold=np.asarray(_col(t, "threshold"), dtype=np.float64),
            empty=np.asarray(_col(t, "empty"), dtype=bool),
            no_signal=np.asarray(_col(t, "no_signal"), dtype=bool),
            capped=np.asarray(_col(t, "capped"), dtype=bool),
            **{**_index_arrays(t, ("rt", "rt_start", "rt_end")), ROLE: self._roles(t)},
            peak_mz=peak_mz, mz_offset_ppm=offset,
            **{name: np.asarray(_col(t, name), dtype=np.float64)  # older bundles: not assessed
               for name in SCREEN_SCORES if name in t.column_names},
            **{name: np.asarray(_col(t, name), dtype=np.float64)  # older bundles: not checked
               for name in MZ_CHECK_COLUMNS if name in t.column_names},
        )

    def spectra(self, ids: Sequence[str] | None = None) -> SpectrumSet | None:
        """The raw spectra (all, or only the rows with these ids), or None if not stored."""
        if not self.has("spectra"):
            return None
        if ids is not None and len(ids) == 0:
            return SpectrumSet.from_spectra([], source=self.spectra_source())
        filters = None if ids is None else [("spectrum_id", "in", [str(i) for i in ids])]
        t = self._table("spectra", filters=filters)
        return dataclasses.replace(_spectra_from_table(t), source=self.spectra_source(),
                                   role=self._roles(t))

    def spectra_source(self) -> dict | None:
        """How the raw spectra were read (for example the runs, targets and settings of
        :func:`~hxex.io.read_mzml`, which :func:`~hxex.io.read_mzml_source` repeats), or None if
        it was not recorded. Present even when the spectra themselves were not stored."""
        return self.manifest["configs"].get("spectra_source")

    def components_table(self) -> dict[str, np.ndarray]:
        """The tidy table of every fit with its intervals (as written for other tools)."""
        t = self._table("components")
        return {name: np.asarray(t.column(name).to_pylist()) if t.schema.field(name).type ==
                pa.string() else _col(t, name) for name in t.column_names}

    def controls(self) -> ControlTable | None:
        """The control table, or None if the run had no controls."""
        if not self.has("controls"):
            return None
        if "controls" not in self._cache:
            config = self.manifest["configs"]["controls"]
            self._cache["controls"] = _read_controls(self._table("controls"), config)
        return self._cache["controls"]

    def envelopes(self) -> EnvelopeTable | None:
        """The envelope of every series, or None if the run used the theory."""
        if not self.has("envelopes"):
            return None
        if "envelopes" not in self._cache:
            config = self.manifest["configs"]["envelope"]
            self._cache["envelopes"] = _read_envelopes(self._table("envelopes"), config)
        return self._cache["envelopes"]

    def quality(self) -> QualityFlags | None:
        """The quality flags, or None if the run did not assess them."""
        if not self.has("quality"):
            return None
        if "quality" not in self._cache:
            config = self.manifest["configs"]["quality"]
            ids = self.fit().spectrum_id
            self._cache["quality"] = _read_quality(self._table("quality"), config, ids)
        return self._cache["quality"]

    def review(self) -> ReviewTable:
        return _read_review(self._table("review")) if self.has("review") else ReviewTable.empty()

    def set_review(self, review: ReviewTable) -> None:
        """Replace the review layer (directory bundles only); no fit is touched."""
        if not self._store.writable:
            raise BundleError("a zip bundle is read only; unpack it to edit the review layer")
        _check_review(review, self.fit())
        files = dict(self.manifest["files"])
        _write_table(self.path, "review", _review_table(review), files)
        manifest = {**self.manifest, "files": files}
        tmp = self.path / (MANIFEST + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(tmp, self.path / MANIFEST)
        self.manifest = manifest


LEGACY_TIME_KEYS = ("undeut_time", "fd_time", "atol")


def modernise_manifest(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """``manifest`` with the settings of bundles written before the roles (control times in
    ``configs.controls`` and ``configs.run``, which the config classes no longer know) taken
    out, and those times (``undeut_time``, ``fd_time``, ``atol``) for :func:`legacy_roles`.
    A current manifest comes back as it is, with an empty record."""
    configs = manifest.get("configs") or {}
    controls = configs.get("controls")
    run = configs.get("run")
    run_controls = (run or {}).get("controls") if isinstance(run, dict) else None
    legacy: dict[str, float] = {}
    for table in (controls, run_controls):
        if isinstance(table, dict):
            for key in LEGACY_TIME_KEYS:
                if key in table and table[key] is not None and key not in legacy:
                    legacy[key] = float(table[key])
    if isinstance(run, dict) and run.get("undeut_time") is not None and "undeut_time" not in legacy:
        legacy["undeut_time"] = float(run["undeut_time"])
    stale = (isinstance(controls, dict) and any(k in controls for k in LEGACY_TIME_KEYS)) or (
        isinstance(run_controls, dict) and any(k in run_controls for k in LEGACY_TIME_KEYS)) or (
        isinstance(run, dict) and "undeut_time" in run)
    if not stale:
        return manifest, {}
    manifest = json.loads(json.dumps(manifest))  # a copy: the file is left as written
    configs = manifest["configs"]
    for table in (configs.get("controls"), (configs.get("run") or {}).get("controls")):
        if isinstance(table, dict):
            for key in LEGACY_TIME_KEYS:
                table.pop(key, None)
    if isinstance(configs.get("run"), dict):
        configs["run"].pop("undeut_time", None)
    log.info("bundle written before the roles: its controls are known by their recorded times "
             "(%s)", ", ".join(f"{k} = {v:g}" for k, v in legacy.items()))
    return manifest, legacy


def legacy_roles(time: np.ndarray, legacy: Mapping[str, float]) -> np.ndarray:
    """The roles of spectra of a bundle written before the roles, from the control times it
    recorded (``undeut_time``, ``fd_time``, within ``atol``); no roles when it recorded none."""
    time = np.asarray(time, dtype=float)
    role = np.full(len(time), "", dtype="<U8")
    atol = float(legacy.get("atol", 1e-6))
    for key, label in (("undeut_time", UNDEUT), ("fd_time", FULLDEUT)):
        if key in legacy:
            role[np.abs(time - legacy[key]) <= atol] = label
    return role


def open_bundle(path: str | Path) -> Bundle:
    """Open a bundle (a directory or a ``.zip``: the bundle's files at its top, or in its one
    folder, as a zipped bundle folder holds them). Raises :class:`BundleError` if it is not one,
    or was written by a newer schema."""
    path = Path(path)
    if path.is_dir() and not (path / MANIFEST).exists() and (path / REPLACING).exists():
        raise BundleError(f"{path}: writing it was interrupted (it holds {REPLACING}); write the "
                          f"bundle again")
    try:
        if path.is_dir():
            store: Any = _DirStore(path)
        elif path.is_file() and zipfile.is_zipfile(path):
            store = _ZipStore(path)
        else:
            raise BundleError(f"{path} is not a bundle directory or zip")
        manifest = json.loads(store.read(MANIFEST).decode())
    except (OSError, KeyError, ValueError) as exc:
        raise BundleError(f"cannot read the manifest of {path}: {exc}") from exc
    if manifest.get("kind") != KIND:
        raise BundleError(f"{path} is not an hxex results bundle")
    if int(manifest.get("schema_version", 0)) > SCHEMA_VERSION:
        raise BundleError(
            f"{path} has schema version {manifest['schema_version']}; this hxex reads up to "
            f"{SCHEMA_VERSION}. Upgrade hxex to open it."
        )
    return Bundle(path, store, manifest)
