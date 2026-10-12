# SPDX-License-Identifier: GPL-3.0-or-later
"""Raw spectra in a single Parquet file, without a fit or a bundle.

``save_spectra`` writes a :class:`~hxex.spectra.SpectrumSet` and ``load_spectra`` reads it back
exactly. The layout is the one of ``spectra.parquet`` in a results bundle (one row per
spectrum, the index columns and the lists ``mz`` and ``intensity``; see docs/BUNDLE.md), so the
two are interchangeable and the file is readable from R and Julia. The file metadata records the
format, the project, the time of writing, the hxex version, a SHA-256 of the contents (checked
on a full load) and, when the spectra have one, their ``source`` (how they were read, as JSON).
This module needs numpy and pyarrow only, not torch.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..spectra import SpectrumSet
from ._arrow import _INDEX, _INDEX_INT, _spectra_from_table, _spectra_table
from .errors import ReadError

KIND = "hxex-spectra"
SCHEMA_VERSION = 1
ROW_GROUP = 1000  # spectra per Parquet row group
_PREFIX = "hxex."
_REQUIRED = (*_INDEX, *_INDEX_INT, "time", "mz", "intensity")


def content_digest(spectra: SpectrumSet) -> str:
    """SHA-256 of what defines the spectra: the ids, the times and every m/z and intensity."""
    h = hashlib.sha256()
    for arr, dtype in ((spectra.offsets, "<i8"), (spectra.mz, "<f8"),
                       (spectra.intensity, "<f8"), (spectra.time, "<f8")):
        h.update(np.ascontiguousarray(arr, dtype=dtype).tobytes())
    h.update("\x00".join(str(i) for i in spectra.spectrum_id).encode("utf-8"))
    return h.hexdigest()


def _project_of(ids) -> str:
    heads = {str(i).partition(":")[0] for i in ids if ":" in str(i)}
    return heads.pop() if len(heads) == 1 else ""


def _hxex_version() -> str:
    try:
        return version("hxex")
    except PackageNotFoundError:
        return "unknown"


def save_spectra(
    spectra: SpectrumSet,
    path: str | Path,
    *,
    project: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write ``spectra`` to the Parquet file ``path`` and return the path.

    The file is written beside its final place and moved there when complete, so a failed write
    leaves nothing behind. An existing file is only replaced with ``overwrite=True`` (a directory
    is never replaced). ``project`` defaults to the project of the spectrum ids
    (``PROJECT:LOCAL_ID``) if they share one.
    """
    path = Path(path)
    if path.exists() and (path.is_dir() or not overwrite):
        raise FileExistsError(
            f"{path} exists" + (" and is a directory" if path.is_dir() else
                                " (use overwrite=True to replace it)")
        )
    meta = {
        _PREFIX + "kind": KIND,
        _PREFIX + "schema_version": str(SCHEMA_VERSION),
        _PREFIX + "project": project if project is not None else _project_of(spectra.spectrum_id),
        _PREFIX + "created": datetime.now(UTC).isoformat(timespec="seconds"),
        _PREFIX + "hxex_version": _hxex_version(),
        _PREFIX + "n_spectra": str(len(spectra)),
        _PREFIX + "content_sha256": content_digest(spectra),
    }
    if spectra.source is not None:
        meta[_PREFIX + "source"] = json.dumps(spectra.source)
    table = _spectra_table(spectra).replace_schema_metadata(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        pq.write_table(table, tmp, compression="zstd", row_group_size=ROW_GROUP)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def spectra_info(path: str | Path) -> dict[str, str]:
    """The metadata of a spectra file (empty for a Parquet file not written by hxex, such as the
    ``spectra.parquet`` of a bundle)."""
    try:
        meta = pq.read_schema(Path(path)).metadata or {}
    except (OSError, pa.ArrowException) as exc:
        raise ReadError(f"cannot read {path}: {exc}") from exc
    return {k.decode()[len(_PREFIX):]: v.decode() for k, v in meta.items()
            if k.decode().startswith(_PREFIX)}


def load_spectra(
    path: str | Path, ids: Sequence[str] | None = None, *, verify: bool = True
) -> SpectrumSet:
    """Read a spectra file (or the ``spectra.parquet`` of a bundle) into a SpectrumSet.

    With ``ids`` only the rows with those spectrum ids are read, in the order stored (ids that
    are not in the file are ignored). A full load checks the contents against the digest in the
    file metadata when there is one (``verify=False`` skips it); a partial load cannot.
    Raises :class:`~hxex.io.ReadError` for a file that is not a spectra file, is of a newer
    format or does not match its digest.
    """
    path = Path(path)
    info = spectra_info(path)
    if info.get("kind", KIND) != KIND:
        raise ReadError(f"{path} is not a spectra file (kind {info['kind']!r})")
    if int(info.get("schema_version", SCHEMA_VERSION)) > SCHEMA_VERSION:
        raise ReadError(
            f"{path} has format version {info['schema_version']}; this hxex reads up to "
            f"{SCHEMA_VERSION}. Upgrade hxex to open it."
        )
    try:
        schema = pq.read_schema(path)
        missing = [c for c in _REQUIRED if c not in schema.names]
        if missing:
            raise ReadError(f"{path} is not a spectra file: missing column(s) {', '.join(missing)}")
        source = json.loads(info["source"]) if "source" in info else None
        if ids is not None and len(ids) == 0:
            return SpectrumSet.from_spectra([], source=source)
        filters = None if ids is None else [("spectrum_id", "in", [str(i) for i in ids])]
        spectra = _spectra_from_table(pq.read_table(path, filters=filters))
    except (OSError, pa.ArrowException) as exc:
        raise ReadError(f"cannot read {path}: {exc}") from exc
    if verify and ids is None and "content_sha256" in info:
        if content_digest(spectra) != info["content_sha256"]:
            raise ReadError(f"{path} does not match its checksum: the file is damaged")
    return dataclasses.replace(spectra, source=source)
