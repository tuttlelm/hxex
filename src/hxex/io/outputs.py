# SPDX-License-Identifier: GPL-3.0-or-later
"""The listing of an output folder (tables, figures): which files hxex wrote there.

A folder of tables or figures beside a bundle is written again by every ``hxex run``,
``resume`` and ``report``. As a bundle directory is replaced file by file (the files its
manifest lists go, see :mod:`hxex.io.bundle`), an output folder keeps a listing,
``hxex_listing.json``, of the files hxex wrote: writing again removes the listed files, refuses
to replace a file of yours named like one of the new files before anything changes, and leaves
every other file alone. A folder with files but no listing (one written before the listing
existed, or your own) is only written when no file of yours would be replaced.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

LISTING = "hxex_listing.json"
KIND = "hxex-output-listing"


class OutputError(Exception):
    """An output folder cannot be written without replacing a file that is not hxex's."""


def output_listing(directory: str | Path) -> list[str] | None:
    """The files hxex wrote into ``directory`` (its listing), or None when it has none."""
    try:
        data = json.loads((Path(directory) / LISTING).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("kind") != KIND:
        return None
    return [str(f) for f in data.get("files", [])]


def _write_listing(directory: Path, files: Iterable[str]) -> None:
    data = {"kind": KIND, "written": datetime.now(UTC).isoformat(timespec="seconds"),
            "files": sorted(set(files))}
    tmp = directory / (LISTING + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(directory / LISTING)


@contextmanager
def replace_outputs(directory: str | Path, names: Iterable[str]) -> Iterator[list[Path]]:
    """Prepare ``directory`` for the files ``names`` (relative to it) and yield the list the
    caller appends every written path to; the listing is written from that list when the
    block ends, also when it fails part-way, so the next write knows what is hxex's. Before
    anything changes: a file in ``directory`` named like one of ``names`` that the listing
    does not cover is refused (:class:`OutputError`); then the listed files are removed, so
    the folder holds only what this write produces (and your own files)."""
    directory = Path(directory)
    names = sorted(set(names))
    listed = output_listing(directory)
    owned = set(listed or ()) | {LISTING}
    clash = [n for n in names if n not in owned and (directory / n).exists()]
    if clash:
        hint = ("written before hxex kept a listing of its files, or " if listed is None
                else "")
        raise OutputError(
            f"{directory} holds file(s) {hint}of your own named like the new ones "
            f"({', '.join(clash[:5])}{', ...' if len(clash) > 5 else ''}); move them, or write "
            f"to another directory")
    directory.mkdir(parents=True, exist_ok=True)
    for name in listed or ():
        target = directory / name
        if target.is_file():
            target.unlink()
    written: list[Path] = []
    try:
        yield written
    finally:
        _write_listing(directory, (str(p.relative_to(directory)) for p in written))
