# SPDX-License-Identifier: GPL-3.0-or-later
"""The listing of an output folder (tables, figures): which files hxex wrote there.

A folder of tables or figures beside a bundle is written again by every ``hxex run``,
``resume`` and ``report``. As a bundle directory is replaced file by file (the files its
manifest lists go, see :mod:`hxex.io.bundle`), an output folder keeps a listing,
``hxex_listing.json``, of the files hxex wrote: writing again removes the listed files, refuses
to replace a file of yours named like one of the new files before anything changes, and leaves
every other file alone. A folder with files but no listing (one written before the listing
existed, or your own) is only written when no file of yours would be replaced.

Every command that writes an output folder follows the bundle's rule (:func:`prepare_output`;
owner, 2026-10-21): a folder that already holds hxex output is refused unless the command is
told to overwrite it (its files are replaced) or to keep it (it is moved to ``name.N`` first);
``[output] overwrite`` and ``keep_previous`` in a project file say so for every folder.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from ._json import dumps as strict_json

log = logging.getLogger(__name__)

LISTING = "hxex_listing.json"
KIND = "hxex-output-listing"


class OutputError(Exception):
    """An output folder cannot be written without replacing a file that is not hxex's."""


def _listing(directory: str | Path) -> dict | None:
    try:
        data = json.loads((Path(directory) / LISTING).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("kind") == KIND else None


def output_listing(directory: str | Path) -> list[str] | None:
    """The files hxex wrote into ``directory`` (its listing), or None when it has none."""
    data = _listing(directory)  # "/" between folders (older Windows listings wrote "\\")
    return None if data is None else [str(f).replace("\\", "/") for f in data.get("files", [])]


def made_from(directory: str | Path) -> dict[str, str] | None:
    """The results an analysis folder was made from, as its listing records them (run name,
    empty for a single bundle -> the bundle's ``created`` stamp; empty when not recorded), or
    None for a folder whose listing does not say (written before hxex recorded it)."""
    data = _listing(directory)
    if data is None or not isinstance(data.get("made_from"), dict):
        return None
    return {str(k): str(v) for k, v in data["made_from"].items()}


def _write_listing(directory: Path, files: Iterable[str],
                   made: Mapping[str, str] | None = None) -> None:
    data = {"kind": KIND, "written": datetime.now(UTC).isoformat(timespec="seconds"),
            "files": sorted(set(files)),
            **({} if made is None else {"made_from": dict(made)})}
    tmp = directory / (LISTING + ".tmp")
    tmp.write_text(strict_json(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(directory / LISTING)


def holds_output(directory: str | Path) -> bool:
    """Whether ``directory`` holds files hxex wrote (its listing names at least one)."""
    return Path(directory).is_dir() and bool(output_listing(directory))


def _next_number(directory: Path) -> int:
    """One more than every ``name.N`` used beside ``directory`` (never reused)."""
    used = [0]
    for item in directory.parent.iterdir():
        m = re.match(re.escape(directory.name) + r"\.(\d+)(\.|_|$)", item.name)
        if m:
            used.append(int(m.group(1)))
    return max(used) + 1


def keep_previous_folder(directory: str | Path) -> Path:
    """Move the output folder ``directory`` to ``name.N`` (N one more than any used beside it)
    and return where it went."""
    directory = Path(directory)
    target = directory.with_name(f"{directory.name}.{_next_number(directory)}")
    size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    os.replace(directory, target)
    log.info("kept the previous %s as %s (%.1f MB)", directory.name, target.name, size / 1e6)
    return target


def prepare_output(directory: str | Path, *, overwrite: bool = False,
                   keep_previous: bool = False, what: str = "files", move: bool = True,
                   elsewhere: str = "another folder") -> Path | None:
    """Apply the rule of an output folder that already holds hxex output (the bundle's rule):
    refused (:class:`OutputError`, saying how to go on) unless ``overwrite`` (its hxex files are
    replaced by the write that follows) or ``keep_previous`` (it is moved to ``name.N`` now, and
    that path is returned; with ``move`` False only checked). A folder without hxex output
    passes. ``what`` names its content and ``elsewhere`` the way to write somewhere else, for
    the message."""
    directory = Path(directory)
    if not holds_output(directory):
        return None
    if keep_previous:
        return keep_previous_folder(directory) if move else None
    if overwrite:
        return None
    raise OutputError(
        f"{directory} holds the {what} written before: --overwrite replaces them, "
        f"--keep-previous moves the folder to {directory.name}.N first, or write to "
        f"{elsewhere}. In a project file, [output] overwrite = true or keep_previous = true "
        "does either for every command")


@contextmanager
def replace_outputs(directory: str | Path, names: Iterable[str], *,
                    made: Mapping[str, str] | None = None) -> Iterator[list[Path]]:
    """Prepare ``directory`` for the files ``names`` (relative to it) and yield the list the
    caller appends every written path to; the listing is written from that list when the
    block ends, also when it fails part-way, so the next write knows what is hxex's. Before
    anything changes: a file in ``directory`` named like one of ``names`` that the listing
    does not cover is refused (:class:`OutputError`); then the listed files are removed, so
    the folder holds only what this write produces (and your own files). ``made`` (run name
    -> the ``created`` stamp of the bundle read; :func:`made_from`) is recorded in the listing
    of an analysis folder, so a later command can tell whether it is out of date."""
    directory = Path(directory)
    names = sorted({Path(n).as_posix() for n in names})  # compared with the listing's form
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
        _write_listing(directory, _relative(directory, written), made)


def _relative(directory: Path, paths: Iterable[Path]) -> list[str]:
    """``paths`` relative to ``directory``, with "/" on every system (a listing moved between
    systems stays valid); a path outside it is logged and left out (never raised from the
    ``finally`` that writes the listing, which would hide the error that ended the write)."""
    out = []
    for p in paths:
        try:
            out.append(Path(p).relative_to(directory).as_posix())
        except ValueError:
            log.warning("%s is not in %s: left out of its listing", p, directory)
    return out
