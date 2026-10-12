# SPDX-License-Identifier: GPL-3.0-or-later
"""What identifies an input regardless of where it is, so that a read can be repeated and a
changed input noticed: a file's name, size and checksum, or a directory's name, number of files,
total size and one checksum over its files' names and contents."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Mapping
from pathlib import Path

log = logging.getLogger(__name__)

# what to do when an input differs from its record
CHANGE_ACTIONS = ("warn", "error")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict:
    """A file's ``name``, ``size`` (bytes) and a ``checksum`` with its ``algorithm``. An indexed
    mzML file's own checksum (``<fileChecksum>``, SHA-1, written by the converter) is used when
    there is one, as it costs nothing to read; otherwise the SHA-256 of the whole file."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as fh:
        fh.seek(max(0, size - 4096))
        m = re.search(rb"<fileChecksum>\s*([0-9a-fA-F]{40})\s*</fileChecksum>", fh.read())
    if m:
        return {"name": path.name, "size": size, "algorithm": "mzml-sha1",
                "checksum": m.group(1).decode().lower()}
    return {"name": path.name, "size": size, "algorithm": "sha256", "checksum": _sha256(path)}


def directory_identity(path: str | Path) -> dict:
    """A directory's ``name``, ``n_files``, total ``size`` and one SHA-256 over every file below
    it (its path relative to the directory and the SHA-256 of its contents, in sorted order), so
    that renaming, adding, removing or changing any file changes the checksum."""
    path = Path(path)
    digest = hashlib.sha256()
    n = size = 0
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = f.relative_to(path).as_posix()
        digest.update(f"{rel}\0{_sha256(f)}\n".encode())
        n += 1
        size += f.stat().st_size
    return {"name": path.name, "n_files": n, "size": size, "algorithm": "sha256-tree",
            "checksum": digest.hexdigest()}


def identity(path: str | Path) -> dict:
    """:func:`file_identity` of a file, :func:`directory_identity` of a directory."""
    path = Path(path)
    return directory_identity(path) if path.is_dir() else file_identity(path)


def compare_identity(path: Path, known: Mapping | None, on_change: str, error: type[Exception]
                     ) -> dict:
    """The identity of ``path`` now, checked against ``known`` (its record, or None). Unchanged:
    the record as it was. Changed: a warning (``on_change='warn'``) and the identity now with
    the recorded one under ``previous``, or ``error`` raised (``'error'``)."""
    if on_change not in CHANGE_ACTIONS:
        raise ValueError(f"on_change must be one of {', '.join(CHANGE_ACTIONS)}")
    now = identity(path)
    if known is None:
        return now
    keys = ("size", "algorithm", "checksum", "n_files")
    if all(now.get(k) == known.get(k) for k in keys):
        return dict(known)
    if on_change == "error":
        raise error(f"{path} has changed since it was read (its size or checksum differs)")
    log.warning("%s has changed since it was read (its size or checksum differs); reading it "
                "as it is now", path)
    return {**now, "previous": {k: v for k, v in known.items() if k != "previous"}}
