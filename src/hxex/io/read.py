# SPDX-License-Identifier: GPL-3.0-or-later
"""One entry point over both input formats."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal

from ..hooks import CancelHook, ProgressHook, check_cancel, report
from ..spectra import PeptideEntry, Spectrum, SpectrumSet, qualify_spectrum_id
from .entries import filter_entries
from .hxexpress import read_hxexpress_file, scan_hxexpress
from .pooled import scan_specexport_pooled
from .specexport import read_specexport_dir, scan_specexport
from .times import DEFAULT_FULLDEUT_TIME, DEFAULT_UNDEUT_TIME

log = logging.getLogger(__name__)

Format = Literal["hxexpress", "specexport", "pooled"]


def scan(
    data_dir: str | Path,
    fmt: Format,
    *,
    samples: Sequence[str] | None = None,
    peptide_ranges: Sequence[str] | None = None,
    sample: str = "",
) -> list[PeptideEntry]:
    """List the entries in a data directory of the given format.

    ``fmt='pooled'`` is a SpecExport folder with peptides from several proteins; ``sample`` is
    the label given to its entries (the other formats take sample names from the data).
    """
    if fmt == "hxexpress":
        return scan_hxexpress(data_dir, samples=samples, peptide_ranges=peptide_ranges)
    if fmt == "specexport":
        return scan_specexport(data_dir, samples=samples, peptide_ranges=peptide_ranges)
    if fmt == "pooled":
        return scan_specexport_pooled(data_dir, sample=sample, peptide_ranges=peptide_ranges)
    raise ValueError(f"unknown format {fmt!r}; use 'hxexpress', 'specexport' or 'pooled'")


def read_spectra(
    entries: Sequence[PeptideEntry],
    data_dir: str | Path,
    fmt: Format,
    *,
    project: str,
    fulldeut_time: float = DEFAULT_FULLDEUT_TIME,
    undeut_time: float = DEFAULT_UNDEUT_TIME,
    samples: Sequence[str] | None = None,
    peptide_ranges: Sequence[str] | None = None,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> SpectrumSet:
    """Read the raw spectra of ``entries`` into one :class:`SpectrumSet`.

    ``project`` names the data set (non-empty, no ``:``); it prefixes every ``spectrum_id``
    (``PROJECT:LOCAL_ID``) so ids stay unique when sets from different directories are combined.

    ``data_id`` is each entry's position in ``entries`` (before filtering by ``samples`` and
    ``peptide_ranges``, so ids stay stable when a filter is applied). ``undeut_time`` and
    ``fulldeut_time`` are the times given to the control labels (``undeut``/``Non`` and
    ``TD``/``Full``); they must differ. Raises
    :class:`~hxex.hooks.Cancelled` if ``cancel`` returns True between entries.
    """
    if fmt not in ("hxexpress", "specexport", "pooled"):
        raise ValueError(f"unknown format {fmt!r}; use 'hxexpress', 'specexport' or 'pooled'")
    qualify_spectrum_id(project, "")  # validate the name before any reading
    if undeut_time == fulldeut_time:
        raise ValueError("undeut_time and fulldeut_time must differ")
    root = Path(data_dir)
    kept = filter_entries(entries, samples=samples, peptide_ranges=peptide_ranges)
    wanted = {id(e) for e in kept}
    jobs = [(i, e) for i, e in enumerate(entries) if id(e) in wanted]
    spectra: list[Spectrum] = []
    for done, (data_id, entry) in enumerate(jobs):
        check_cancel(cancel)
        log.info("reading %s %s z=%d", entry.sample, entry.peptide_range, entry.charge)
        if fmt == "hxexpress":
            spectra += read_hxexpress_file(
                root / entry.file, entry, data_id=data_id, fulldeut_time=fulldeut_time,
                undeut_time=undeut_time,
            )
        else:
            rel = entry.file if fmt == "pooled" else f"{entry.sample}/{entry.file}"
            spectra += read_specexport_dir(
                root / rel, entry, data_id=data_id, fulldeut_time=fulldeut_time,
                undeut_time=undeut_time, id_prefix=rel,
            )
        report(progress, done + 1, len(jobs))
    return SpectrumSet.from_spectra(
        replace(s, spectrum_id=qualify_spectrum_id(project, s.spectrum_id)) for s in spectra
    )
