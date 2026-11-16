# SPDX-License-Identifier: GPL-3.0-or-later
"""Exporting one series: an HX-Express spreadsheet, and the data behind its figures.

A series is a peptide (or fragment ion) at one charge in one sample, as the window's Results
tree lists it: its time points, replicates, and controls. :func:`export_hxexpress` writes the
spreadsheet HX-Express reads (one sheet with every time point and replicate of one peptide and
charge); :func:`write_plot_data` writes CSV tables with what the series' figures draw, so they
can be drawn again in one's own style. docs/PLAN.md Next steps 32.
"""

from __future__ import annotations

import logging
import re
import weakref
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..chem import SequenceError, parse_sequence
from ..io.hxexpress import (
    HXEXPRESS_TEXT,
    hxexpress_control_label,
    hxexpress_filename,
    hxexpress_time_label,
    write_hxexpress_file,
)
from ..io.times import format_time
from ..spectra import FULLDEUT, UNDEUT, point_order_key
from .results import Results
from .series import _control_row, _reps, series_rows
from .spectrum import _charge_of, _fit_data, _picked, _recorded, peak_measure
from .tables import (
    _index_columns,
    components_table,
    spectra_table,
    uptake_table,
    write_csv,
)

log = logging.getLogger(__name__)

SPECTRA_CHOICES = ("raw", "picked")  # what an HX-Express export writes
CONTROLS_CHOICES = ("all", "first")  # every control replicate, or the first of each
KS_CHOICES = ("selected", "all")  # the plot data's K: the one in effect, or every fitted K


class ExportError(ValueError):
    """A series that cannot be exported as asked (the message says why and what to do)."""


# --- HX-Express ----------------------------------------------------------------------------

def _first_row(results: Results, data_id: int) -> int:
    try:
        return int(series_rows(results, data_id)[0])
    except KeyError:
        raise ExportError(f"no series {data_id} in the results") from None


def plain_sequence(sequence: str, results: Results | None = None) -> str:
    """The one-letter (FASTA) form of a modX sequence: each residue's own letter, modifications
    and terminal groups left out (``Ac-PEPacKTIDE`` -> ``PEPKTIDE``)."""
    registry = (results.settings.registry if results is not None
                and results.settings is not None else None)
    try:
        residues = parse_sequence(sequence) if registry is None else parse_sequence(sequence,
                                                                                    registry)
    except SequenceError as exc:
        raise ExportError(f"cannot write the sequence {sequence!r} in one-letter form: {exc}"
                          ) from exc
    return "".join(r[-1] for r in residues)


def _affix(value: str, what: str, *, hxexpress: bool = False) -> str:
    """A user's prefix or suffix for a file or folder name: nothing a name cannot hold on every
    system, and, in an HX-Express file name (``hxexpress``), no ``-`` (it separates the name's
    fields)."""
    value = str(value or "").strip()
    bad = sorted({c for c in value if c in '/\\:*?"<>|' or ord(c) < 32
                  or (hxexpress and c == "-")})
    if bad:
        why = ("; '-' separates the HX-Express name's fields, use '_'" if "-" in bad else "")
        raise ExportError(f"the {what} {value!r} cannot be used in a file name: it holds "
                          f"{' '.join(repr(c) for c in bad)}{why}")
    return value


NAME_LIMIT = 255  # the longest file name (in bytes, UTF-8) most file systems take

# what tells two series of one HX-Express name apart, in the order tried: the modified form the
# one-letter sequence leaves out, the fragment list's label or scan conditions, the parent
QUALIFIERS = ("form", "condition", "parent")


def _text_part(value: str) -> str:
    """``value`` as part of an HX-Express name's free-text field: letters, digits, and ``_``
    only (``+`` written ``p``, as the ion types; anything else ``_``)."""
    text = "".join(c if c.isalnum() else "_" for c in str(value).replace("+", "p"))
    return re.sub("_+", "_", text).strip("_")


def _residue_text(number: int) -> str:
    return f"minus{-number}" if number < 0 else str(number)


def _parent_text(parent_range: str) -> str:
    """A parent's residues in the free-text field (``12-20`` -> ``parent_12_20``; a residue
    below 0 is written ``minus3``, since ``-`` separates the name's fields)."""
    m = re.fullmatch(r"\s*(-?\d+)\s*-\s*(-?\d+)\s*", str(parent_range))
    if m is None:
        return _text_part(f"parent_{parent_range}") if parent_range else ""
    return f"parent_{_residue_text(int(m[1]))}_{_residue_text(int(m[2]))}"


def _check_length(name: str, what: str = "file name") -> None:
    size = len(name.encode("utf-8"))
    if size > NAME_LIMIT:
        raise ExportError(f"the {what} {name[:40]}… is {size} characters long, more than the "
                          f"{NAME_LIMIT} a file system takes: shorten the prefix or suffix")


def _held(cache: dict, results: Results, make):
    """``make(results)``, kept per results (by identity: Results holds arrays, so it cannot be
    hashed) while the results live; the dialog asks again at every key pressed."""
    held = cache.get(id(results))
    if held is not None and held[0]() is results:
        return held[1]
    value = make(results)
    for key in [k for k, (ref, _) in cache.items() if ref() is None]:
        del cache[key]
    cache[id(results)] = (weakref.ref(results), value)
    return value


_NAMES: dict[int, tuple[weakref.ref, dict]] = {}  # id(results) -> (results, name fields)


def _hxexpress_fields(results: Results) -> dict[int, dict | str]:
    """Every series' HX-Express name fields (``sample``, ``written``, ``start``, ``end``,
    ``sequence``, ``charge``, ``text``: the free-text field's parts), made unique over the whole
    results (:func:`hxexpress_name`), or the reason a series cannot be named."""
    return _held(_NAMES, results, _make_hxexpress_fields)


def _make_hxexpress_fields(results: Results) -> dict[int, dict | str]:
    from ..analysis.fragments import _condition
    from ..io.mzml import SCAN_READERS, mzml_series_targets

    b = results.binned
    if b is None:
        raise ExportError("the results have no picked table, which names the series' sample, "
                          "peptide, and charge")
    fit = results.fit
    source = results.settings.spectra_source if results.settings is not None else None
    targets = (mzml_series_targets(source) if source and source.get("reader") in SCAN_READERS
               and "series" in source else {})
    where = {str(s): j for j, s in enumerate(b.spectrum_id)}
    ids, first = np.unique(np.asarray(fit.data_id), return_index=True)
    out: dict[int, dict | str] = {}
    for d, i in zip(ids.tolist(), first.tolist(), strict=True):
        j = where.get(str(fit.spectrum_id[i]))
        if j is None:
            out[d] = f"series {d} is not in the picked table, which names its peptide"
            continue
        peptide = str(b.peptide[j])
        try:
            plain = plain_sequence(peptide, results)
        except ExportError as exc:
            out[d] = str(exc)
            continue
        sample = str(b.sample[j])
        written = sample
        for char in ("-", "/", "\\"):
            written = written.replace(char, "_")
        target = targets.get(d)
        extra = dict(target.entry.extra) if target is not None else {}
        fragment = str(extra.get("fragment", "") or "")
        if fragment:  # the ion type and the channel it was read in (c_1_MS3, M_MS2)
            ion = fragment.rpartition(":")[0] or fragment
            text = [_text_part(ion), f"MS{target.ms_level}"]
        else:
            text = [HXEXPRESS_TEXT]
        modification = str(b.modification[j] or "") if b.modification is not None else ""
        form = "_".join(p for p in (_text_part(peptide) if peptide != plain else "",
                                    _text_part(modification)) if p)
        out[d] = {
            "sample": sample, "written": written, "start": int(b.start_seq[j]),
            "end": int(b.end_seq[j]), "sequence": plain, "charge": int(b.charge[j]),
            "text": text, "qualifiers": {
                "form": form,
                "condition": _text_part(_condition(target)) if fragment else "",
                "parent": _parent_text(str(extra.get("parent_range", "") or "")) if fragment
                else ""}}
    _make_unique({d: f for d, f in out.items() if isinstance(f, dict)})
    return out


def _make_unique(fields: dict[int, dict]) -> None:
    """Add to the free-text field of the series that would share a name what tells them apart
    (:data:`QUALIFIERS`, then ``id<data_id>``), so no two series of the results share a file
    name, in any case (Windows and macOS do not tell ``WT`` from ``wt``)."""

    def key(f: dict) -> tuple:
        return (f["written"].lower(), f["start"], f["end"], f["sequence"], f["charge"],
                "_".join(f["text"]).lower())

    def clashes() -> list[list[int]]:
        groups: dict[tuple, list[int]] = {}
        for d, f in fields.items():
            groups.setdefault(key(f), []).append(d)
        return [g for g in groups.values() if len(g) > 1]

    for kind in QUALIFIERS:
        for group in clashes():
            values = {fields[d]["qualifiers"][kind] for d in group}
            if len(values) > 1:
                for d in group:
                    if fields[d]["qualifiers"][kind]:
                        fields[d]["text"].append(fields[d]["qualifiers"][kind])
    for group in clashes():
        for d in group:
            fields[d]["text"].append(f"id{d}")


def hxexpress_name(results: Results, data_id: int, *, prefix: str = "", suffix: str = ""
                   ) -> str:
    """The HX-Express file name of a series: ``SAMPLE_START-END-SEQUENCE-zCHARGE-TEXT.xlsx``,
    the sequence in one-letter form (:func:`plain_sequence`; HX-Express takes modifications by
    manual entry). ``TEXT``, the free-text field HX-Express does not read, is ``hxex`` for a
    peptide and, for a fragment ion, its ion type and the channel it was read in (``c_1_MS3``;
    ``M_MS2`` for the precursor read with the fragments). The name is unique over the whole
    results: where two series would share one, ``TEXT`` also says what tells them apart, in
    readable form: the modified form the one-letter sequence leaves out (``PEPacKTIDE``), the
    fragment list's label or scan conditions (``ETD``), the parent's residues
    (``parent_12_20``), and ``id<data_id>`` only when they still share it. A ``-`` (or path
    separator) in the sample name is written ``_``, with a warning naming both. ``suffix`` is
    added to ``TEXT`` (``hxex_SUFFIX``); ``prefix`` goes before the sample
    (``PREFIX_SAMPLE``), so a reader of the file takes it as part of the sample's name.
    Neither may hold ``-``. A name longer than a file system takes is refused
    (:class:`ExportError`); for a series starting below residue 0 see
    :func:`hxexpress_warning`."""
    prefix = _affix(prefix, "prefix", hxexpress=True)
    suffix = _affix(suffix, "suffix", hxexpress=True)
    _first_row(results, data_id)
    fields = _hxexpress_fields(results).get(int(data_id))
    if fields is None:
        raise ExportError(f"no series {data_id} in the results")
    if isinstance(fields, str):
        raise ExportError(fields)
    sample, written = fields["sample"], fields["written"]
    if written != sample:
        log.warning("the sample %r is written %r in the HX-Express file name ('-' separates its "
                    "fields)", sample, written)
    if prefix:
        written = f"{prefix}_{written}"
    text = "_".join([*fields["text"], suffix] if suffix else fields["text"])
    try:
        name = hxexpress_filename(written, fields["start"], fields["end"], fields["sequence"],
                                  fields["charge"], text)
    except ValueError as exc:
        raise ExportError(f"series {data_id}: {exc}") from exc
    _check_length(name)
    return name


def hxexpress_warning(results: Results, data_id: int) -> str:
    """Why hxex's HX-Express reader will not take the series' peptide from its file name by
    itself ('' when it will): a series starting below residue 0 (numbered from a tag) has a
    ``-`` in its range, the name's field separator. The file is written all the same
    (HX-Express itself does not mind); give the peptide in a table when reading it back."""
    fields = _hxexpress_fields(results).get(int(data_id))
    if not isinstance(fields, dict) or fields["start"] >= 0:
        return ""
    return (f"series {data_id} ({fields['sample']} {fields['start']}-{fields['end']}) starts "
            "below residue 0: hxex's HX-Express reader cannot take its peptide from the file "
            "name by itself ('-' separates the name's fields); list the file and its peptide "
            "in a peptide table when reading it back (HX-Express itself does not mind)")


def _order_label(value: float) -> str:
    try:
        return hxexpress_time_label(value)
    except ValueError:
        return "(cannot be written)"


def hxexpress_spectra(results: Results, data_id: int, *, spectra: str = "raw",
                      controls: str = "all") -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The ``(label, mz, intensity)`` of every spectrum of a series as
    :func:`export_hxexpress` writes them, in order: the undeuterated controls, the times
    ascending, the fully deuterated controls, each point's replicates in order. Spectra the
    review rejected are left out (an ``[[review.exclude]]`` rule rejects; ``[[input.exclude]]``
    spectra were never fitted). ``spectra``: ``raw`` (the spectra as read) or ``picked`` (the
    sticks: each bin's predicted m/z and value). ``controls``: ``all`` replicates or the
    ``first`` of each control. With ``time_scale = "order"`` each point's own value (as the
    results store it) is written as that many seconds, so every series labels a point alike."""
    if spectra not in SPECTRA_CHOICES:
        raise ValueError(f"spectra must be one of {SPECTRA_CHOICES}, got {spectra!r}")
    if controls not in CONTROLS_CHOICES:
        raise ValueError(f"controls must be one of {CONTROLS_CHOICES}, got {controls!r}")
    fit = results.fit
    try:
        rows = series_rows(results, data_id)
    except KeyError:
        raise ExportError(f"no series {data_id} in the results") from None
    status = np.asarray(results.review_status)
    left = [i for i in rows if status[i] == "rejected"]
    rows = np.array([i for i in rows if status[i] != "rejected"], dtype=np.int64)
    if left:
        log.info("series %d: %d rejected spectr%s left out", data_id, len(left),
                 "um" if len(left) == 1 else "a")
    if not len(rows):
        raise ExportError(f"series {data_id}: every spectrum is rejected; nothing to write")
    reps = _reps(results)
    rows = np.array(sorted(rows, key=lambda i: (
        point_order_key(np.nan if fit.role[i] else float(fit.time[i]), str(fit.role[i])),
        int(reps[i]), int(i))), dtype=np.int64)
    order = results.time_scale == "order"
    if order:  # the points' own values, never ranks: the same in every series (owner)
        points = sorted({float(fit.time[i]) for i in rows
                         if not fit.role[i] and np.isfinite(fit.time[i])})
        log.info("series %d: the points have no real times (time_scale = \"order\"); each "
                 "point's own value is written as that many seconds: %s", data_id,
                 ", ".join(f"point {format_time(t, 'order')} -> {_order_label(t)}"
                           for t in points))
    ids = [str(fit.spectrum_id[i]) for i in rows]
    raw = _raw_spectra(results, ids) if spectra == "raw" else {}
    if spectra == "raw" and len(raw) < len(ids):
        missing = [s for s in ids if s not in raw]
        raise ExportError(
            f"series {data_id}: the raw spectra of {len(missing)} of its {len(ids)} spectra are "
            f"not in the results (first: {missing[0]}); export the picked sticks instead")
    out, count = [], {}
    for i, sid in zip(rows, ids, strict=True):
        role = str(fit.role[i])
        t = float(fit.time[i])
        key = role or t
        count[key] = count.get(key, 0) + 1
        n = count[key]
        if role in (UNDEUT, FULLDEUT):
            if controls == "first" and n > 1:
                continue
            label = hxexpress_control_label(role, n)
        else:
            try:
                label = hxexpress_time_label(t, n)
            except ValueError as exc:
                why = (" (time_scale = \"order\": each point's own value is written as that "
                       "many seconds)" if order else "")
                raise ExportError(f"series {data_id}, spectrum {sid}: {exc}{why}") from exc
        if spectra == "raw":
            mz, y = raw[sid]
        else:
            picked = _picked(sid, results, None)
            if picked is None or picked[0] is None:
                raise ExportError(f"series {data_id}: the results have no picked m/z for "
                                  f"{sid} (the picked table is missing)")
            mz, y = picked[0], picked[2]
        out.append((label, np.asarray(mz, float), np.asarray(y, float)))
    return out


def _raw_spectra(results: Results, ids: Sequence[str]) -> dict[str, tuple]:
    """The raw spectra of ``ids`` that the results hold (read from the bundle in one go)."""
    source = results.spectra
    if source is None and results.loader is not None:
        source = results.loader(list(ids))
    if source is None:
        return {}
    out = {}
    for sid in ids:
        try:
            out[sid] = source.spectrum(source.find(sid))
        except KeyError:
            continue
    return out


def export_hxexpress(results: Results, data_id: int, directory: str | Path, *,
                     spectra: str = "raw", controls: str = "all", prefix: str = "",
                     suffix: str = "", overwrite: bool = False) -> Path:
    """Write a series as an HX-Express spreadsheet in ``directory``, named by
    :func:`hxexpress_name` (with the user's ``prefix`` and ``suffix``), and return its path.
    What is written: :func:`hxexpress_spectra`. An existing file is refused unless
    ``overwrite``. A series hxex's reader cannot name from the file name alone
    (:func:`hxexpress_warning`) is written with a warning."""
    directory = Path(directory)
    path = directory / hxexpress_name(results, data_id, prefix=prefix, suffix=suffix)
    if _exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists (overwrite=True replaces it)")
    columns = hxexpress_spectra(results, data_id, spectra=spectra, controls=controls)
    warning = hxexpress_warning(results, data_id)
    if warning:
        log.warning("%s: %s", path.name, warning)
    directory.mkdir(parents=True, exist_ok=True)
    write_hxexpress_file(path, columns)
    log.info("wrote %s (%d spectra, %s)", path, len(columns), spectra)
    return path


def _exists(path: Path) -> bool:
    """Whether ``path`` is there; a path the system cannot take (a name too long) is an
    :class:`ExportError` naming it."""
    try:
        return path.exists()
    except OSError as exc:
        raise ExportError(f"cannot use {path}: {exc.strerror or exc}") from exc


# --- the data behind the figures -----------------------------------------------------------

def _rows_of(results: Results, data_id: int | None, spectrum_ids: Sequence[str] | None
             ) -> np.ndarray:
    if (data_id is None) == (spectrum_ids is None):
        raise ValueError("give a series (data_id) or spectra (spectrum_ids), not both")
    if data_id is not None:
        try:
            return series_rows(results, int(data_id))
        except KeyError:
            raise ExportError(f"no series {data_id} in the results") from None
    rows = []
    for sid in spectrum_ids:
        try:
            rows.append(results.find(str(sid)))
        except KeyError:
            raise ExportError(f"no spectrum {sid!r} in the results") from None
    if not rows:
        raise ExportError("no spectra to export")
    return np.array(rows, dtype=np.int64)


def _bin_windows(results: Results, sid: str, picked, source) -> dict[float, tuple]:
    """Each picked bin's window as ``{bin: (centre m/z, half-width)}``: the window the spectrum
    figure shades (the predicted m/z, moved where the bundle moved it). Empty when the results
    hold no picked m/z; the half-width is NaN when the picker settings are unknown."""
    if picked is None or picked[0] is None:
        return {}
    centre = np.asarray(picked[0], float)
    half = np.full(len(centre), np.nan)
    cfg = results.settings.picker if results.settings is not None else None
    if cfg is not None:
        half = np.asarray(cfg.half_width(centre, _charge_of(sid, results, None),
                                         **_recorded(sid, results, source)), float)
    return {float(b): (float(c), float(h))
            for b, c, h in zip(np.asarray(picked[1], float), centre, half, strict=True)}


def _bins_table(results: Results, rows: np.ndarray, ks: str) -> dict[str, np.ndarray]:
    """One row per spectrum, K, and deuterium bin: what a spectrum panel draws."""
    fit = results.fit
    b = results.binned
    where = {} if b is None else {str(s): j for j, s in enumerate(b.spectrum_id)}
    effective = results.effective_K
    status = np.asarray(results.review_status)
    reps = _reps(results)
    fitted = sorted(int(k) for k in fit.k_values)
    blocks: list[dict] = []
    k_top = 0
    cfg = results.settings.picker if results.settings is not None else None
    source = results.spectra
    if source is None and cfg is not None and cfg.uses_recorded and results.loader is not None:
        source = results.loader([str(fit.spectrum_id[i]) for i in rows])  # recorded R
    for i in rows:
        sid = str(fit.spectrum_id[i])
        chosen = int(effective[i])
        picked = _picked(sid, results, None)
        windows = _bin_windows(results, sid, picked, source)
        for k in ([chosen] if ks == "selected" else fitted):
            try:
                d = _fit_data(results, sid, k, None)
            except (KeyError, ValueError):  # a K the screen skipped for this spectrum
                continue
            k_top = max(k_top, k)
            blocks.append({"i": int(i), "sid": sid, "k": k, "selected": k == chosen, "d": d,
                           "picked": picked, "windows": windows})
    names = ["spectrum_id", "data_id", "sample", "peptide_range", "peptide", "charge", "time",
             "time_label", "role", "rep", "review_status", "K", "k", "selected", "bin", "mz",
             "mz_low", "mz_high", "picked", "fit", "fit_band_low", "fit_band_high", "residual"]
    for r in range(1, k_top + 1):
        names += [f"population_{r}", f"population_{r}_band_low", f"population_{r}_band_high"]
    cols: dict[str, list] = {n: [] for n in names}
    scale = results.time_scale
    for blk in blocks:
        i, d = blk["i"], blk["d"]
        bins = np.asarray(d["bins"], float)
        n = len(bins)
        j = where.get(blk["sid"])
        time, role = float(fit.time[i]), str(fit.role[i])
        label = {UNDEUT: "UN", FULLDEUT: "FD"}.get(role) or format_time(time, scale)
        ident = {
            "spectrum_id": blk["sid"], "data_id": int(fit.data_id[i]),
            "sample": "" if j is None else str(b.sample[j]),
            "peptide_range": "" if j is None else str(b.peptide_range[j]),
            "peptide": "" if j is None else str(b.peptide[j]),
            "charge": np.nan if j is None else float(b.charge[j]),
            "time": time, "time_label": label, "role": role, "rep": int(reps[i]),
            "review_status": str(status[i]), "K": int(effective[i]), "k": blk["k"],
            "selected": blk["selected"]}
        for name, value in ident.items():
            cols[name] += [value] * n
        cols["bin"] += bins.tolist()
        picked = blk["picked"]
        window = np.array([blk["windows"].get(float(x), (np.nan, np.nan)) for x in bins],
                          dtype=float).reshape(n, 2)
        cols["mz"] += window[:, 0].tolist()
        cols["mz_low"] += (window[:, 0] - window[:, 1]).tolist()
        cols["mz_high"] += (window[:, 0] + window[:, 1]).tolist()
        observed = (np.full(n, np.nan) if picked is None or len(picked[2]) != n
                    else np.asarray(picked[2], float))
        cols["picked"] += observed.tolist()
        total = np.asarray(d["total"], float)
        cols["fit"] += total.tolist()
        band = d["band"]
        cols["fit_band_low"] += (np.full(n, np.nan) if band is None else band[0]).tolist()
        cols["fit_band_high"] += (np.full(n, np.nan) if band is None else band[1]).tolist()
        cols["residual"] += (observed - total).tolist()
        parts, part_band = d["components"], d["component_band"]
        for r in range(1, k_top + 1):
            have = r <= d["k"]
            cols[f"population_{r}"] += (parts[r - 1] if have else np.full(n, np.nan)).tolist()
            for side, name in ((0, "low"), (1, "high")):
                value = (part_band[r - 1, side] if have and part_band is not None
                         else np.full(n, np.nan))
                cols[f"population_{r}_band_{name}"] += np.asarray(value, float).tolist()
    out = {name: np.array(values, dtype=object) for name, values in cols.items()}
    for name in names:
        if name not in ("spectrum_id", "sample", "peptide_range", "peptide", "time_label",
                        "role", "review_status", "selected"):
            out[name] = np.array(cols[name], dtype=float)
    for name in ("data_id", "rep", "K", "k"):
        out[name] = out[name].astype(np.int64)
    return out


def _components_bootstrap_table(results: Results, rows: np.ndarray
                                ) -> dict[str, np.ndarray] | None:
    """One row per spectrum, population, and bootstrap resample at the K in effect: the cloud
    the populations page draws behind each population (None when the results kept no
    resamples: ``[run.bootstrap] keep_draws``). ``rank`` is the population's in
    ``components.csv`` (the fit's, lowest mean first), ``resample`` the resample's number."""
    from .series import population_data

    fit = results.fit
    index = _index_columns(results)
    names = ("spectrum_id", "data_id", "sample", "peptide_range", "peptide", "charge", "time",
             "role", "rep")
    cols: dict[str, list] = {n: [] for n in (*names, "review_status", "K", "rank", "resample",
                                             "mean", "mean_corrected", "population")}
    status = np.asarray(results.review_status)
    for i in rows:
        i, k = int(i), int(results.effective_K[i])
        if not _resampled(results, i):
            continue
        sid = str(fit.spectrum_id[i])
        plain = population_data(results, sid, k=k, corrected=False)
        if plain["draws"] is None or not len(plain["draws"]):
            continue
        shown = population_data(results, sid, k=k)  # corrected where the series is
        resample = np.flatnonzero(results.bootstrap_for(i, k).draws.ok[i]) + 1
        n = len(resample)
        for c in range(k):
            for name in names:
                cols[name] += [index[name][i]] * n
            cols["review_status"] += [str(status[i])] * n
            cols["K"] += [k] * n
            cols["rank"] += [c + 1] * n
            cols["resample"] += resample.tolist()
            cols["mean"] += plain["draws"][:, c, 0].tolist()
            cols["mean_corrected"] += (shown["draws"][:, c, 0] if shown["corrected"]
                                       else np.full(n, np.nan)).tolist()
            cols["population"] += plain["draws"][:, c, 1].tolist()
    if not cols["spectrum_id"]:
        return None
    out = {name: np.asarray(values) for name, values in cols.items()}
    for name in ("time", "mean", "mean_corrected", "population"):
        out[name] = np.asarray(cols[name], dtype=float)
    return out


def _resampled(results: Results, i: int) -> bool:
    """Whether the bootstrap kept resamples of spectrum ``i`` at the K in effect (so
    ``components_bootstrap`` has rows for it)."""
    k = int(results.effective_K[i])
    boot = results.bootstrap_for(i, k)
    return (boot is not None and boot.draws is not None and bool(results.fit.fitted(k)[i])
            and bool(np.asarray(boot.draws.ok[i]).any()))


def kept_resamples(results: Results) -> bool:
    """Whether the results kept the bootstrap's resamples (``[run.bootstrap] keep_draws``), so
    that :func:`plot_data` can write ``components_bootstrap``."""
    boots = [results.bootstrap, *results.bootstrap_by_k.values()]
    return any(b is not None and b.draws is not None for b in boots)


def plot_tables(results: Results) -> tuple[str, ...]:
    """The tables :func:`write_plot_data` may write for some series of these results: ``raw``
    only when they hold the raw spectra, ``components_bootstrap`` only when they kept the
    resamples. What it writes for given spectra: :func:`plot_data_tables`."""
    raw = results.spectra is not None or results.loader is not None
    return tuple(t for t in PLOT_TABLES
                 if (t != "raw" or raw) and (t != "components_bootstrap"
                                             or kept_resamples(results)))


def _holds_raw(results: Results, ids: Sequence[str]) -> bool:
    """Whether the results hold the raw spectrum of any of ``ids``: those in memory are looked
    up; a bundle stores the raw spectrum of every spectrum it fitted."""
    if results.spectra is not None:
        for sid in ids:
            try:
                results.spectra.find(sid)
            except KeyError:
                continue
            return True
        return False
    return results.loader is not None


def plot_data_tables(results: Results, *, data_id: int | None = None,
                     spectrum_ids: Sequence[str] | None = None) -> tuple[str, ...]:
    """The tables :func:`plot_data` makes (and :func:`write_plot_data` writes) for a series
    (``data_id``) or some of its spectra (``spectrum_ids``), without making them: ``raw`` only
    when the results hold their raw spectra, ``components_bootstrap`` only when the bootstrap
    kept resamples of one of them at its K in effect."""
    rows = _rows_of(results, data_id, spectrum_ids)
    ids = [str(results.fit.spectrum_id[i]) for i in rows]
    raw = _holds_raw(results, ids)
    resamples = any(_resampled(results, int(i)) for i in rows)
    return tuple(t for t in PLOT_TABLES
                 if (t != "raw" or raw) and (t != "components_bootstrap" or resamples))


def _raw_table(results: Results, rows: np.ndarray) -> dict[str, np.ndarray] | None:
    """One row per raw point of each spectrum (None when the results hold no raw spectra)."""
    fit = results.fit
    ids = [str(fit.spectrum_id[i]) for i in rows]
    raw = _raw_spectra(results, ids) if _holds_raw(results, ids) else {}
    if not raw:
        return None
    cols: dict[str, list] = {n: [] for n in ("spectrum_id", "data_id", "mz", "intensity")}
    for i, sid in zip(rows, ids, strict=True):
        if sid not in raw:
            continue
        mz, y = raw[sid]
        cols["spectrum_id"] += [sid] * len(mz)
        cols["data_id"] += [int(fit.data_id[i])] * len(mz)
        cols["mz"] += np.asarray(mz, float).tolist()
        cols["intensity"] += np.asarray(y, float).tolist()
    return {"spectrum_id": np.array(cols["spectrum_id"], dtype=object),
            "data_id": np.array(cols["data_id"], dtype=np.int64),
            "mz": np.array(cols["mz"], dtype=float),
            "intensity": np.array(cols["intensity"], dtype=float)}


def plot_data(results: Results, *, data_id: int | None = None,
              spectrum_ids: Sequence[str] | None = None, ks: str = "selected"
              ) -> dict[str, dict[str, np.ndarray]]:
    """The data behind a series' figures (``data_id``), or behind some of its spectra
    (``spectrum_ids``: a time point, or one spectrum), as tables of columns:

    - ``bins``: one row per spectrum, K, and deuterium bin: the spectrum's identity, ``K`` (in
      effect) and ``k`` (of the row; ``selected`` when they agree), ``bin``, ``mz`` (the bin's
      window centre: its predicted m/z), ``mz_low``/``mz_high`` (the picking window's edges,
      as the figure shades it: select the ``raw`` points of the same ``spectrum_id`` between
      them), ``picked`` (the bin's value), ``fit`` (the whole fit) with
      ``fit_band_low``/``high`` (the bootstrap band; empty without one), ``residual`` (picked
      minus fit), and ``population_<r>`` (each population's curve, ascending mean) with its
      band. Unscaled. ``ks``: ``selected`` (the K in effect) or ``all`` (every fitted K);
    - ``raw``: one row per point of each raw spectrum (left out when the results hold none);
    - ``spectra``, ``components``, ``uptake``: the rows of ``spectra.csv``, ``components.csv``,
      and ``uptake.csv`` for these spectra (uptake: their series at their points), as the run's
      tables have them;
    - ``components_bootstrap``: one row per spectrum, population, and bootstrap resample, the
      cloud the populations page draws behind each population of ``components`` (left out when
      the results kept no resamples: ``[run.bootstrap] keep_draws``);
    - ``about``: ``item``, ``value`` pairs (hxex version, the run, the peak measure, the
      correction, the K written).

    Rejected spectra are included, marked by ``review_status``, as the figures draw them."""
    if ks not in KS_CHOICES:
        raise ValueError(f"ks must be one of {KS_CHOICES}, got {ks!r}")
    rows = _rows_of(results, data_id, spectrum_ids)
    fit = results.fit
    series = sorted({int(d) for d in fit.data_id[rows]})
    out: dict[str, dict[str, np.ndarray]] = {"bins": _bins_table(results, rows, ks)}
    raw = _raw_table(results, rows)
    if raw is not None:
        out["raw"] = raw
    out["spectra"] = spectra_table(results, rows=rows)
    out["components"] = components_table(results, rows=rows)
    resamples = _components_bootstrap_table(results, rows)
    if resamples is not None:
        out["components_bootstrap"] = resamples
    uptake = uptake_table(results, data_ids=series)
    points = {(int(fit.data_id[i]), str(fit.role[i]),
               round(float(fit.time[i]), 9) if np.isfinite(fit.time[i]) else None) for i in rows}
    keep = np.array([
        (int(d), str(r), round(float(t), 9) if np.isfinite(t) else None) in points
        for d, r, t in zip(uptake["data_id"], uptake["role"], uptake["time"], strict=True)],
        dtype=bool)
    out["uptake"] = {name: np.asarray(values)[keep] for name, values in uptake.items()}
    out["about"] = _about(results, series, ks, raw is not None, resamples is not None)
    return out


def _about(results: Results, series: list[int], ks: str, raw: bool, resamples: bool
           ) -> dict[str, np.ndarray]:
    import hxex

    settings = results.settings
    corrected = [d for d in series if _control_row(results, d) is not None]
    items = [
        ("exported", datetime.now(UTC).isoformat(timespec="seconds")),
        ("hxex_version", hxex.__version__),
        ("hxex_commit", hxex.commit()),
        ("project", "" if settings is None else settings.project),
        ("results_created", "" if settings is None else settings.created),
        ("peak_measure", peak_measure(results)),
        ("time_scale", results.time_scale),
        ("back_exchange_corrected", "yes" if corrected and len(corrected) == len(series)
         else "no" if not corrected else "some series"),
        ("k_written", "the K in effect for each spectrum" if ks == "selected"
         else "every fitted K"),
        ("raw_spectra", "raw.csv" if raw else "not in the results"),
        ("bootstrap_resamples", "components_bootstrap.csv" if resamples
         else "not kept (results made before 2026-11-13, or keep_draws = false)"),
        ("intensities", "as measured (unscaled)"),
    ]
    return {"item": np.array([k for k, _ in items], dtype=object),
            "value": np.array([v for _, v in items], dtype=object)}


PLOT_TABLES = ("bins", "raw", "spectra", "components", "components_bootstrap", "uptake",
               "about")  # write_plot_data


def plot_data_name(results: Results, data_id: int, *, part: str = "", prefix: str = "",
                   suffix: str = "") -> str:
    """The start of a series' plot-data file names, ``[PREFIX_]STEM[_PART][_SUFFIX]``, each
    file then ending ``_bins.csv``, ``_raw.csv`` and so on (owner, 2026-11-12: files named by
    the series in the chosen folder, not a folder of their own). ``STEM`` is the series' figure
    name (:func:`hxex.report.figures.figure_stems`) and ``PART`` what of it is exported when
    not all of it (a time point ``30 s``, a spectrum ``30 s rep 2``), written compactly
    (:func:`hxex.report.figures.name_part`: ``30s``, ``30s_rep2``, ``1p5min``)."""
    from .figures import name_part

    prefix, suffix = _affix(prefix, "prefix"), _affix(suffix, "suffix")
    _first_row(results, data_id)
    stem = _series_stems(results)[int(data_id)]
    name = "_".join(p for p in (prefix, stem, name_part(part), suffix) if p)
    _check_length(f"{name}_{max(PLOT_TABLES, key=len)}.csv")
    return name


_STEMS: dict[int, tuple[weakref.ref, dict[int, str]]] = {}  # id(results) -> (results, stems)


def _series_stems(results: Results) -> dict[int, str]:
    """Every series' figure name, made unique over the whole results (``_id<data_id>`` where two
    series share a label), so that exports of separate series never share file names; kept
    per results (the dialog asks again at every key pressed)."""
    from .figures import figure_stems

    return _held(_STEMS, results, lambda r: figure_stems(
        r, [int(d) for d in np.unique(r.fit.data_id)]))


def plot_data_files(directory: str | Path, name: str, tables: Sequence[str] = PLOT_TABLES
                    ) -> list[Path]:
    """The files of ``tables`` (:func:`plot_data_tables`; by default every table
    :func:`write_plot_data` may write) in ``directory`` for ``name`` (:func:`plot_data_name`)."""
    return [Path(directory) / f"{name}_{table}.csv" for table in tables]


def plot_data_leftovers(directory: str | Path, name: str, tables: Sequence[str]
                        ) -> list[Path]:
    """The files of ``name`` in ``directory`` that an earlier export wrote and this one, of
    ``tables``, does not (an earlier ``_components_bootstrap.csv`` or ``_raw.csv``): left as
    they are, never deleted (owner, 2026-11-15: the user may have changed them)."""
    return [p for p in plot_data_files(directory, name, [t for t in PLOT_TABLES
                                                          if t not in tables]) if _exists(p)]


def write_plot_data(results: Results, directory: str | Path, *, name: str | None = None,
                    data_id: int | None = None, spectrum_ids: Sequence[str] | None = None,
                    ks: str = "selected", overwrite: bool = False) -> list[Path]:
    """Write :func:`plot_data` as CSV files into ``directory`` (made if missing), named
    ``NAME_bins.csv``, ``NAME_raw.csv``, ``NAME_spectra.csv``, ``NAME_components.csv``,
    ``NAME_components_bootstrap.csv``, ``NAME_uptake.csv``, and ``NAME_about.csv``
    (:func:`plot_tables`: raw and bootstrap only when the results hold them); ``name``
    defaults to the series' :func:`plot_data_name`. Refused, before anything is written, when
    any of them is there, unless ``overwrite``. ``NAME_about.csv`` lists the files of this
    export (``files``); a file of the same name an earlier export wrote and this one does not
    (:func:`plot_data_leftovers`) is left as it is, and named in the messages."""
    tables = plot_data(results, data_id=data_id, spectrum_ids=spectrum_ids, ks=ks)
    if name is None:
        series = sorted({int(d) for d in tables["spectra"]["data_id"]})
        if len(series) != 1:
            raise ExportError(f"the spectra are of {len(series)} series: give the files a "
                              "name (name=...)")
        name = plot_data_name(results, series[0])
    directory = Path(directory)
    paths = {table: directory / f"{name}_{table}.csv" for table in tables}
    there = [p.name for p in paths.values() if _exists(p)]
    if there and not overwrite:
        raise FileExistsError(f"{directory} already holds {', '.join(there)} "
                              "(overwrite=True replaces them)")
    about = tables["about"]
    tables["about"] = {
        "item": np.append(about["item"], "files").astype(object),
        "value": np.append(about["value"], "; ".join(p.name for p in paths.values())
                           ).astype(object)}
    left = plot_data_leftovers(directory, name, list(tables))
    if left:
        log.warning("%s: left as %s from an earlier export, not part of this one (not "
                    "written now): %s", directory, "it is" if len(left) == 1 else "they are",
                    ", ".join(p.name for p in left))
    directory.mkdir(parents=True, exist_ok=True)
    written = [write_csv(columns, paths[table]) for table, columns in tables.items()]
    n = len(tables["spectra"]["spectrum_id"])
    log.info("wrote the plot data of %d spectr%s into %s (%s_*.csv)", n,
             "um" if n == 1 else "a", directory, name)
    return written
