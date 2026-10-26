# SPDX-License-Identifier: GPL-3.0-or-later
"""The project file: one TOML file with everything a run needs (what ``hxex run`` reads).

::

    project = "name"            # prefixes every spectrum id
    threads = 12                # optional: CPU threads for the command (default: PyTorch's)

    [input]                     # what to read, by format
    format = "specexport"       # specexport, hxexpress, pooled, mzml or thermo
    ...

    [run]                       # the run settings (hxex.run.RunConfig), section by section
    [run.fit]
    k_max = 3
    [run.bootstrap]             # a section turns its stage on; "bootstrap = false" turns it off

    [output]
    bundle = "results"          # the folder (or .zip) hxex creates for the results

    [analysis]                  # the post-fitting analysis (hxex analyse; hxex.analysis)
    reference = "WT"

A key left out keeps its default. TOML has no null: ``false`` stands for None where a setting
can be None (an optional stage, a scan condition a run removes, an unset limit) and is not a
switch; a switch (``drop_no_signal``, ``overview``) set to ``false`` is off. This holds in every
``[run]`` table and sub-table, by the type of each setting. Times are in seconds, written as a
number or as text with a unit (``"5 min"``, ``"20 h"``; :func:`hxex.io.parse_time`), in every
time setting (``[[input.runs]]`` ``time``, ``[analysis]`` ``times``, ...); the loader converts
them, so nothing after it sees a unit. The controls are the spectra with a role (``undeut``,
``fulldeut``): the folder formats' control labels, ``role`` in ``[[input.runs]]``, or the
spectra at ``undeut_at`` and ``fulldeut_at`` of ``[run.controls]``. Relative paths are taken
relative to the project file's directory.
:func:`load_project` reads and checks the file, :func:`read_input` reads the spectra it names,
and :func:`template` writes a project file with every default.
"""

from __future__ import annotations

import logging
import math
import re
import textwrap
import tomllib
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

from .hooks import CancelHook, ProgressHook
from .io.errors import ReadError
from .io.mzml import MzmlConfig, MzmlRun, MzmlTarget
from .io.times import TIME_SCALES, parse_time
from .run import RunConfig
from .spectra import SpectrumSet, check_role

log = logging.getLogger(__name__)

FORMATS = ("specexport", "hxexpress", "pooled", "mzml", "thermo")
FOLDER_FORMATS = ("specexport", "hxexpress", "pooled")
SCAN_FORMATS = ("mzml", "thermo")
# run settings that are None unless the file gives them, and their classes
_OPTIONAL_SECTIONS = ("bootstrap", "population_test", "controls", "quality")
# the figures moved out of the run (owner, 2026-10-20): hxex report draws them
FIGURES_MOVED = ("the figure settings are the [figures] table now, and hxex run draws no "
                 "figures: hxex report hxex.toml --figures draws them with those settings")


class ProjectError(ValueError):
    """A project file that cannot be used."""


def _check_keys(data: Mapping, allowed, where: str) -> None:
    unknown = set(data) - set(allowed)
    if unknown:
        raise ProjectError(f"unknown key(s) in {where}: {', '.join(sorted(unknown))}")


def _items(value) -> list:
    """The items of a list of tables (``[[name]]``): a single table (``[name]``, one item
    written with single brackets) as a list of one."""
    if value is None:
        return []
    return [value] if isinstance(value, Mapping) else value


def _none(value):
    """TOML's stand-in for None (``false``) as None."""
    return None if value is False else value


def _hints(cls) -> dict[str, Any]:
    """The resolved field annotations of a config class."""
    from .report.figures import PlotConfig  # RunConfig names it for type checking only

    return typing.get_type_hints(cls, localns={"PlotConfig": PlotConfig})


def _union_members(hint) -> tuple:
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        return typing.get_args(hint)
    return (hint,)


def _config_class(hint) -> type | None:
    """The config dataclass a field's annotation names (``X`` or ``X | None``), else None."""
    for member in _union_members(hint):
        if isinstance(member, type) and is_dataclass(member):
            return member
    return None


def _false_to_none(data: Mapping, cls) -> dict:
    """``data`` (a TOML table of the config class ``cls``) with ``false`` as None in the keys
    whose field admits None and is not a switch (no bool in its type; the module's convention),
    recursively into the sub-tables of nested config classes. Keys the class does not know are
    kept as they are (its ``from_dict`` reports them)."""
    hints = _hints(cls)
    out = {}
    for key, value in data.items():
        members = _union_members(hints[key]) if key in hints else ()
        if value is False and type(None) in members and bool not in members:
            out[key] = None
        elif isinstance(value, Mapping) and (sub := _config_class(hints.get(key))) is not None:
            out[key] = _false_to_none(value, sub)
        else:
            out[key] = value
    return out


def _form(hint) -> tuple[str, Any] | None:
    """What a TOML value of a field annotated ``hint`` must look like: ``(words, accepts)``
    with ``accepts(value)`` True for a value of that form; None when any value will do (an
    untyped or ``Any`` field). A None (``false`` in the file) is checked by the caller."""
    forms: list[tuple[str, Any]] = []
    for member in _union_members(hint):
        if member is type(None):
            continue
        origin = typing.get_origin(member) or member
        if member is Any or origin is object or isinstance(origin, typing.TypeVar):
            return None
        if origin is bool:
            forms.append(("true or false", lambda v: isinstance(v, bool)))
        elif origin is int:
            forms.append(("a whole number", lambda v: isinstance(v, int)
                          and not isinstance(v, bool)))
        elif origin is float:
            forms.append(("a number", lambda v: isinstance(v, (int, float))
                          and not isinstance(v, bool)))
        elif origin is str or origin is Path:
            forms.append(("text", lambda v: isinstance(v, str)))
        elif isinstance(origin, type) and issubclass(origin, (tuple, list)) or (
                isinstance(origin, type) and issubclass(origin, typing.Sequence)
                and not issubclass(origin, str)):
            forms.append(("a list", lambda v: isinstance(v, list)))
        elif isinstance(origin, type) and (issubclass(origin, Mapping) or is_dataclass(origin)):
            forms.append(("a table", lambda v: isinstance(v, Mapping)))
        else:
            return None
    if not forms:
        return None
    words = " or ".join(dict.fromkeys(w for w, _ in forms))
    return words, lambda v: any(accepts(v) for _, accepts in forms)


def _check_types(data: Mapping, cls, where: str) -> None:
    """Every key of ``data`` (a TOML table of the config class ``cls``, ``false`` already
    read as None where the field admits it) has a value of the form its field's type says,
    recursively through the sub-tables of nested config classes; else a :class:`ProjectError`
    naming the table, the key, the value and the expected form. Unknown keys are left to the
    class (its ``from_dict`` reports them)."""
    hints = _hints(cls)
    for key, value in data.items():
        if key not in hints:
            continue
        hint = hints[key]
        form = _form(hint)
        if value is None:
            if form is not None and type(None) not in _union_members(hint):
                raise ProjectError(f"[{where}] {key} cannot be false (it has no unset state): "
                                   f"leave it out for the default")
            continue
        if form is not None and not form[1](value):
            shown = _toml_value(value) if not isinstance(value, Mapping) else "a table"
            hint_text = (" (false stands for unset only where a setting can be unset)"
                         if value is False else "")
            raise ProjectError(f"[{where}] {key} = {shown}: expected {form[0]}{hint_text}")
        if isinstance(value, Mapping) and (sub := _config_class(hint)) is not None:
            _check_types(value, sub, f"{where}.{key}")


def _merge(base: dict, update: Mapping, where: str) -> dict:
    """``base`` (a config's ``to_dict``) with ``update`` laid over it, key by key into tables."""
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            out[key] = _merge(base[key], value, f"{where}.{key}")
        else:
            out[key] = value
    return out


def _section_class(name: str) -> type:
    """The config class of a ``[run.NAME]`` table."""
    return _config_class(_hints(RunConfig)[name])


def _section_defaults(name: str) -> dict:
    return _section_class(name)().to_dict()


def run_config_from(data: Mapping) -> RunConfig:
    """The :class:`~hxex.run.RunConfig` of a project file's ``[run]`` table (``false`` as None
    where a setting admits it; see the module docstring)."""
    merged = RunConfig().to_dict()
    if "plots" in data:
        raise ProjectError(f"[run.plots]: {FIGURES_MOVED}. Rename [run.plots] to [figures]")
    if "undeut_time" in data:
        raise ProjectError("[run] undeut_time: the controls are no longer identified by a time; "
                           "the undeuterated controls (the noise references) are the spectra "
                           "with the role undeut (the control labels, or role in "
                           "[[input.runs]])")
    if isinstance(data.get("controls"), Mapping):
        _refuse_control_times(data["controls"], "[run.controls]",
                              "the controls are the spectra with a role (the control labels, "
                              "or role in [[input.runs]]); undeut_at and fulldeut_at make the "
                              "spectra at a time serve as a control as well")
        data = {**data, "controls": _times_in(data["controls"], ("undeut_at", "fulldeut_at"),
                                              "run.controls")}
    data = _times_in(data, ("envelope_fallback_time",), "run")
    registry = data.get("registry")
    if isinstance(registry, Mapping) and isinstance(registry.get("residues"), Mapping):
        data = {**data, "registry": {**registry, "residues": _items(registry["residues"])}}
    given = _false_to_none(data, RunConfig)
    for key, value in given.items():
        if key in _OPTIONAL_SECTIONS:
            if value is None:
                merged[key] = None
            elif value is True:
                merged[key] = _section_defaults(key)
            elif isinstance(value, Mapping):
                merged[key] = _merge(_section_defaults(key), value, f"run.{key}")
            else:
                raise ProjectError(f"[run] {key} = {_toml_value(value)}: expected a table, "
                                   "true or false")
        elif key in merged and isinstance(value, Mapping) and isinstance(merged[key], Mapping):
            merged[key] = _merge(merged[key], value, f"run.{key}")
        else:
            merged[key] = value
    _check_types({k: v for k, v in given.items() if not (k in _OPTIONAL_SECTIONS
                                                          and v is True)}, RunConfig, "run")
    for key, value in merged.items():  # each table's own errors name it
        if isinstance(value, Mapping) and (cls := _section_class(key)) is not None:
            try:
                cls.from_dict(value)
            except (TypeError, ValueError) as exc:
                raise ProjectError(f"[run.{key}]: {exc}") from exc
    try:
        return RunConfig.from_dict(merged)
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"[run]: {exc}") from exc


@dataclass(frozen=True)
class FragmentSettings:
    """Fragment targets made from every peptide of the table (one ``[[input.fragments]]`` table;
    a project may have several, for example the same ions with and without in-source
    activation).

    - ``ions``: the fragment list, space-separated ``TYPE:LENGTHS:CHARGES`` items such as
      ``'c-1:ALL z-dot:2-6,9'`` (:mod:`hxex.fragments`; ``ALL`` is the default for both).
    - ``label``: names the condition (``'gentle'``; kept with each entry as ``condition``).
    - ``ms_level``: 2, or 3 for MS3 scans; ``activation``: of the last stage;
      ``preactivation``: for MS3, the activation before the last stage (``'none'`` for none).
    - ``parent_mz``: ``'parent'`` (scans whose isolation overlaps the peptide's m/z range), an
      explicit ``[low, high]`` (m/z) or None (any parent); ``precursor_mz`` (m/z) and
      ``precursor_charge`` (``'parent'``, a charge, or None for any) match the scan's precursor.
    - ``filter_contains``, ``filter_excludes``: text the scan filter must or must not contain.
    - ``rt_window`` (s): ``[start, end]`` of the scans used, for runs with a fixed time (a
      peptide's own ``rt_start``, ``rt_end`` columns, in minutes, take its place).
    - ``limit_charge``: ``ALL`` fragment charges run from 1 to the parent's charge minus one
      (False: to the parent's charge); charges written out are used as given.
    """

    ions: str
    label: str | None = None
    ms_level: int = 2
    activation: str | None = None
    preactivation: str | None = None
    parent_mz: tuple[float, float] | str | None = "parent"
    precursor_mz: float | None = None
    precursor_charge: int | str | None = "parent"
    filter_contains: str | None = None
    filter_excludes: str | None = None
    rt_window: tuple[float, float] | None = None
    limit_charge: bool = True


# the scan conditions of the peptides themselves (MzmlTarget without its entry)
_TARGET_KEYS = ("ms_level", "activation", "precursor_mz", "precursor_charge", "rt_window",
                "filter_contains", "filter_excludes", "parent_mz", "preactivation")


@dataclass(frozen=True)
class InputSettings:
    """What to read (the ``[input]`` table); paths are absolute here.

    - ``format``: ``specexport``, ``hxexpress``, ``pooled`` (folder formats), ``mzml`` or
      ``thermo`` (scan formats).
    - Folder formats: ``data_dir``; ``samples`` and ``peptide_ranges`` (only these; a range
      ``START-END`` selects the peptide with those residues, however it is written) and
      ``peptide_region`` (``[first, last]``: every peptide wholly within those residues; with
      ``peptide_ranges`` too, a peptide that matches either);
      ``sample`` (the name given to a pooled folder's entries); ``peptides``, a table of the
      files instead of finding them by name. The control files (``Non``/``Full``,
      ``undeut``/``TD``) get their role from their label.
    - Scan formats: ``peptides`` (the peptide table, CSV), ``runs`` (one :class:`MzmlRun` per
      file), ``parents`` (read each peptide itself), ``target`` (how the peptides' scans are
      chosen), ``fragments`` (fragment targets), ``scans`` (how scans become spectra,
      :class:`MzmlConfig`), ``data_root`` (relative run paths start here) and ``thermo_dir``
      (Thermo's DLLs).
    - Every format: ``record_paths`` (``'path'``, ``'relative'`` or ``'name'``: how file paths
      are recorded), ``retention_times`` (a table of retention times, minutes), ``spectra``
      (take the raw spectra stored by an earlier run, a bundle or a spectra file, instead of
      reading the input) and ``time_scale``: ``'seconds'`` (the times are real labelling times)
      or ``'order'`` (the times only order the points, as with dummy labels: the uptake is drawn
      by index and the kinetics, decay and protection analyses refuse).
    """

    format: str
    # folder formats
    data_dir: Path | None = None
    samples: tuple[str, ...] | None = None
    peptide_ranges: tuple[str, ...] | None = None
    peptide_region: tuple[int, int] | None = None
    sample: str = ""
    record_paths: str = "path"
    time_scale: str = "seconds"
    # mzML and Thermo raw files
    peptides: Path | None = None
    runs: tuple[MzmlRun, ...] = ()
    parents: bool = True
    target: dict = field(default_factory=dict)
    fragments: tuple[FragmentSettings, ...] = ()
    scans: MzmlConfig = MzmlConfig()  # noqa: RUF009
    data_root: Path | None = None
    thermo_dir: Path | None = None
    # every format: a table of retention times (minutes)
    retention_times: Path | None = None
    # every format: take the raw spectra stored by an earlier run (a bundle, or a spectra file)
    # instead of reading the input
    spectra: Path | None = None
    # every format: rules of spectra left out before fitting (stored, not fitted)
    exclude: tuple = ()


@dataclass(frozen=True)
class OutputSettings:
    """Where results go (the ``[output]`` table): the bundle and the tables (a directory, or
    True for the bundle's name plus ``_tables``). The bundle and the folders named after it
    (its tables, figures and analyses, and their earlier generations, ``name.N``) are hxex's: a
    later command replaces them (``overwrite``) or moves them first (``keep_previous``: the
    bundle with its folders to ``name.N``, see :func:`hxex.run.run`; a figures, tables or
    analysis folder written alone to its own ``NAME.N``, :func:`hxex.io.prepare_output`), for
    ``hxex run``, ``report`` and ``analyse`` alike. A run draws no figures (``hxex report``
    does, into ``name_figures``)."""

    bundle: Path | None = None
    overwrite: bool = False
    keep_previous: bool = False
    keep_spectra: bool = True
    tables: Path | bool = True
    notes: str = ""

    def tables_dir(self) -> Path | None:
        return _beside(self.bundle, self.tables, "_tables")


def _beside(bundle: Path | None, value: Path | bool, suffix: str) -> Path | None:
    if value is False:
        return None
    if value is True:
        if bundle is None:
            raise ProjectError(f"output: give a directory or a bundle for the {suffix[1:]}")
        return bundle.with_name(bundle.name + suffix)
    return value


@dataclass(frozen=True)
class Project:
    """A project file, checked: ``project`` (the spectrum-id prefix), ``input`` (None in a
    file read for analysis only, :func:`load_project`), ``run`` and ``output``; ``threads``,
    the CPU threads of a command that runs it (None: PyTorch's default; a setting of the
    process, so the command line applies it, not the library); ``path`` is the file itself."""

    project: str
    input: InputSettings | None
    run: RunConfig
    output: OutputSettings
    path: Path | None = None
    threads: int | None = None
    analysis: object = None  # hxex.analysis.AnalysisConfig, or None without an [analysis]
    figures: object = None  # hxex.report.PlotConfig of the [figures] table (None: defaults)
    review: object = None  # hxex.review_table.ReviewSettings of the [review] table (None: none)


def _path(base: Path, value, where: str) -> Path | None:
    if value is None or value is False:
        return None
    if not isinstance(value, str):
        raise ProjectError(f"{where} must be a path (text)")
    p = Path(value).expanduser()
    return p if p.is_absolute() else base / p


def _strings(value, where: str) -> tuple[str, ...] | None:
    value = _none(value)
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ProjectError(f"{where} must be text or a list of text")
    return tuple(value)


def _exclude(data: Mapping) -> tuple:
    """``[[input.exclude]]``: the rules of spectra left out before fitting."""
    from .exclusions import read_exclusions

    try:
        return read_exclusions(data.get("exclude"), "input.exclude")
    except ValueError as exc:
        raise ProjectError(str(exc)) from None


def _ranges(value) -> tuple[str, ...] | None:
    """``[input] peptide_ranges``: each ``START-END`` (checked here, so hxex check reports it)."""
    from .io.entries import parse_range

    ranges = _strings(value, "input.peptide_ranges")
    for r in ranges or ():
        try:
            parse_range(r)
        except ValueError as exc:
            raise ProjectError(f"input.peptide_ranges: {exc}") from None
    return ranges


def _region(value) -> tuple[int, int] | None:
    """``[input] peptide_region``: ``[first, last]``, residue numbers, first <= last."""
    if value is None:
        return None
    if (not isinstance(value, list) or len(value) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)
            or not 1 <= value[0] <= value[1]):
        raise ProjectError(f"input.peptide_region = {value!r}: expected [first, last], two "
                           "residue numbers with first <= last (for example [1, 50])")
    return int(value[0]), int(value[1])


def _window(value, where: str, unit: str = "[start, end] in seconds"):
    value = _none(value)
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ProjectError(f"{where} must be {unit}")
    if unit.endswith("seconds"):  # a time window: units allowed
        return (_time(value[0], where), _time(value[1], where))
    return (float(value[0]), float(value[1]))


def _time(value, where: str) -> float:
    """Seconds of a time setting written as a number or as text with units
    (:func:`hxex.io.parse_time`); a :class:`ProjectError` names ``where`` otherwise."""
    try:
        return parse_time(value, where)
    except ReadError as exc:
        raise ProjectError(str(exc)) from exc


def _times_in(data: Mapping, keys, where: str) -> dict:
    """``data`` with the time settings ``keys`` (a time, or a list of times) in seconds."""
    out = dict(data)
    for key in keys:
        value = out.get(key)
        if value is None or value is False:  # false: unset (the false rule, applied later)
            continue
        if isinstance(value, list):
            out[key] = [_time(v, f"{where}.{key}") for v in value]
        else:
            out[key] = _time(value, f"{where}.{key}")
    return out


CONTROL_WORDS = {"undeut": "undeut", "un": "undeut", "non": "undeut", "undeuterated": "undeut",
                 "td": "fulldeut", "fd": "fulldeut", "full": "fulldeut", "fulldeut": "fulldeut"}
_OLD_CONTROL_KEYS = ("undeut_time", "fulldeut_time", "fd_time", "atol")


def _time_scale(data: Mapping, record: Mapping | None = None) -> str:
    """``[input] time_scale``; when the file does not set it and stored spectra are taken,
    the scale their read record carries (``record``)."""
    scale = data.get("time_scale", (record or {}).get("time_scale", "seconds"))
    if scale not in TIME_SCALES:
        raise ProjectError(f"input.time_scale must be one of {', '.join(TIME_SCALES)}")
    return scale


def _refuse_control_times(data: Mapping, where: str, instead: str) -> None:
    """The control-time keys of project files written before the roles are refused with a
    message that names what took their place."""
    old = [k for k in _OLD_CONTROL_KEYS if k in data]
    if old:
        raise ProjectError(f"{where} {', '.join(old)}: the controls are no longer identified "
                           f"by a time; {instead}")


def _fragment_settings(block) -> FragmentSettings:
    """One ``[input.fragments]`` table."""
    if not isinstance(block, Mapping):
        raise ProjectError("input.fragments must be a table or an array of tables")
    frag = dict(block)
    _check_keys(frag, {f.name for f in fields(FragmentSettings)}, "[input.fragments]")
    if "ions" not in frag:
        raise ProjectError("input.fragments needs ions (for example \"c-1:ALL z-dot:ALL\")")
    frag = {k: _none(v) for k, v in frag.items()}
    if "rt_window" in frag:
        frag["rt_window"] = _window(frag["rt_window"], "input.fragments.rt_window")
    if frag.get("parent_mz") not in (None, "parent"):
        frag["parent_mz"] = _window(frag["parent_mz"], "input.fragments.parent_mz",
                                    "\"parent\", [low, high] (m/z) or false")
    return FragmentSettings(**frag)


def _input_from(data: Mapping, base: Path) -> InputSettings:
    """The ``[input]`` table. With ``spectra`` (stored raw spectra) the rest of the table is
    checked as usual when it is there (the same project file can read the input or rerun from a
    bundle), but the format is that of the stored spectra's read record (logged when the file
    says otherwise); with ``spectra`` alone (and at most ``retention_times`` and
    ``time_scale``) it comes from the record only."""
    names = {f.name for f in fields(InputSettings)}
    _refuse_control_times(data, "[input]", "the control labels (Non/Full, undeut/TD) give "
                          "their spectra a role; for a scan format set role in [[input.runs]]")
    _check_keys(data, names, "[input]")
    if "spectra" not in data:
        return _read_settings(data, base)
    data = dict(data)
    stored = _path(base, data.pop("spectra"), "input.spectra")
    try:  # the record only (the manifest); a missing or unusable path is reported when read
        source = stored_spectra_source(stored) or {}
    except ProjectError as exc:
        log.debug("input.spectra: %s (resolved when the spectra are read)", exc)
        source = None
    reader = (source or {}).get("reader")
    record = {"format": reader if reader in FORMATS else "stored"}
    if not (set(data) - {"retention_times", "time_scale", "exclude"}):
        return InputSettings(
            retention_times=_path(base, data.get("retention_times"), "input.retention_times"),
            time_scale=_time_scale(data, source), spectra=stored, exclude=_exclude(data),
            **record)
    inp = replace(_read_settings(data, base), time_scale=_time_scale(data, source))
    if source is None:  # the file's own settings until the record can be read
        return replace(inp, spectra=stored)
    differs = {k: getattr(inp, k) for k, v in record.items() if getattr(inp, k) != v
               and (k == "format" or inp.format in FOLDER_FORMATS)}
    if differs:
        log.warning("input.spectra: the stored spectra were read with %s; the project file's "
                    "%s do(es) not apply to them", ", ".join(
                        f"{k} = {v!r}" for k, v in record.items() if k in differs),
                    ", ".join(f"{k} = {v!r}" for k, v in differs.items()))
    return replace(inp, spectra=stored, **record)


def _record_paths(data: Mapping, allowed: tuple[str, ...], fmt: str) -> None:
    value = data.get("record_paths", "path")
    if value not in allowed:
        extra = ' ("relative" is for mzml and thermo, whose run paths it records relative to '\
                'data_root)' if value == "relative" else ""
        raise ProjectError(f"input.record_paths = {value!r}: expected "
                           f"{', '.join(repr(a) for a in allowed)} for {fmt}{extra}")


def _read_settings(data: Mapping, base: Path) -> InputSettings:
    fmt = data.get("format")
    if fmt not in FORMATS:
        raise ProjectError(f"input.format must be one of {', '.join(FORMATS)}")
    if fmt in FOLDER_FORMATS:
        extra = set(data) & {"runs", "parents", "target", "fragments", "scans", "data_root",
                             "thermo_dir"}
        if extra:
            raise ProjectError(f"input key(s) {', '.join(sorted(extra))} are for mzml and "
                               f"thermo, not {fmt}")
        if "data_dir" not in data:
            raise ProjectError(f"input.data_dir is needed for {fmt}")
        _record_paths(data, ("path", "name"), fmt)
        return InputSettings(
            format=fmt, data_dir=_path(base, data["data_dir"], "input.data_dir"),
            samples=_strings(data.get("samples"), "input.samples"),
            peptide_ranges=_ranges(data.get("peptide_ranges")),
            peptide_region=_region(data.get("peptide_region")),
            sample=str(data.get("sample", "")),
            record_paths=str(data.get("record_paths", "path")),
            time_scale=_time_scale(data),
            peptides=_path(base, data.get("peptides"), "input.peptides"),
            retention_times=_path(base, data.get("retention_times"), "input.retention_times"),
            exclude=_exclude(data),
        )
    extra = set(data) & {"data_dir", "samples", "peptide_ranges", "peptide_region", "sample"}
    if extra:
        raise ProjectError(f"input key(s) {', '.join(sorted(extra))} are not used by {fmt} "
                           "(the runs give the samples and times)")
    if "peptides" not in data:
        raise ProjectError("input.peptides (a peptide table, CSV) is needed for mzml and thermo")
    _record_paths(data, ("path", "relative", "name"), fmt)
    runs = _items(data.get("runs"))
    if not isinstance(runs, list) or not runs:
        raise ProjectError("give at least one [[input.runs]] (path and sample)")
    run_names = {f.name for f in fields(MzmlRun)}
    parsed_runs = []
    for i, r in enumerate(runs, start=1):
        where = f"input.runs #{i}" + (f" ({r['path']})" if isinstance(r.get("path"), str)
                                      else "")
        _check_keys(r, run_names, where)
        r = dict(r)
        time = r.get("time")
        if isinstance(time, str) and time.strip().lower() in CONTROL_WORDS:
            raise ProjectError(f"{where}: time = {_toml_value(time)} names a control; write "
                               f"role = \"{CONTROL_WORDS[time.strip().lower()]}\" (and leave "
                               "time out, or give the file's labelling time as well)")
        if time is not None:
            try:
                r["time"] = parse_time(time, f"{where} time")
            except ReadError as exc:
                raise ProjectError(
                    f"{where}: time must be a number of seconds or text with a unit (\"60 s\", "
                    f"\"5 min\"), the labelling time of the whole file, or left out (each scan "
                    f"has its own time); got {time!r} ({exc})") from None
        for key in ("time_offset", "usable_from"):
            if r.get(key) is not None:
                r[key] = _time(r[key], f"{where} {key}")
        if "role" in r:
            try:
                r["role"] = check_role(_none(r["role"]), f"{where} role")
            except ValueError as exc:
                raise ProjectError(str(exc)) from exc
        rep = r.get("rep", 1)
        if isinstance(rep, bool) or not isinstance(rep, int) or rep < 1:
            raise ProjectError(f"{where}: rep must be a whole number of 1 or more, got {rep!r}")
        try:
            parsed_runs.append(MzmlRun.from_dict({k: _none(v) if k != "time" else v
                                                  for k, v in r.items()}))
        except (TypeError, ValueError) as exc:
            raise ProjectError(f"{where}: {exc}") from exc
    target = dict(data.get("target", {}))
    _check_keys(target, _TARGET_KEYS, "[input.target]")
    target = {k: _none(v) for k, v in target.items()}
    if "rt_window" in target:
        target["rt_window"] = _window(target["rt_window"], "input.target.rt_window")
    if target.get("parent_mz") is not None:
        target["parent_mz"] = _window(target["parent_mz"], "input.target.parent_mz",
                                      "[low, high] (m/z) or false")
    fragments = tuple(_fragment_settings(b) for b in _items(data.get("fragments")))
    parents = bool(data.get("parents", True))
    if not parents and not fragments:
        raise ProjectError("parents = false needs [input.fragments]: there would be no target")
    scans = data.get("scans", {})
    if not isinstance(scans, Mapping):
        raise ProjectError(f"[input] scans = {_toml_value(scans)}: expected a table")
    scans = _false_to_none(scans, MzmlConfig)
    scans = _times_in(scans, ("min_time", "peak_search", "peak_smoothing"), "input.scans")
    if isinstance(scans.get("windows"), list):
        scans["windows"] = [_window(w, "input.scans.windows") for w in scans["windows"]]
    _check_types(scans, MzmlConfig, "input.scans")
    try:
        scans = MzmlConfig.from_dict(_merge(MzmlConfig().to_dict(), scans, "input.scans"))
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"[input.scans]: {exc}") from exc
    return InputSettings(
        format=fmt, peptides=_path(base, data["peptides"], "input.peptides"),
        runs=tuple(parsed_runs), parents=parents, target=target, fragments=fragments,
        scans=scans, record_paths=str(data.get("record_paths", "path")),
        time_scale=_time_scale(data),
        data_root=_path(base, data.get("data_root", "."), "input.data_root"),
        thermo_dir=_path(base, data.get("thermo_dir"), "input.thermo_dir"),
        retention_times=_path(base, data.get("retention_times"), "input.retention_times"),
        exclude=_exclude(data),
    )


def _output_from(data: Mapping, base: Path) -> OutputSettings:
    if "figures" in data:  # every template before the move wrote figures = false
        if data["figures"] is not False:
            raise ProjectError(f"[output] figures: {FIGURES_MOVED}. Delete the line")
        log.info("[output] figures = false is no longer needed (hxex run draws no figures): "
                 "the line can go")
        data = {k: v for k, v in data.items() if k != "figures"}
    _check_keys(data, {f.name for f in fields(OutputSettings)}, "[output]")

    def where(key, default):
        value = data.get(key, default)
        return value if isinstance(value, bool) else _path(base, value, f"output.{key}")

    return OutputSettings(
        bundle=_path(base, data.get("bundle"), "output.bundle"),
        overwrite=bool(data.get("overwrite", False)),
        keep_previous=bool(data.get("keep_previous", False)),
        keep_spectra=bool(data.get("keep_spectra", True)),
        tables=where("tables", True), notes=str(data.get("notes", "")),
    )


def project_from(data: Mapping, base: str | Path = ".", path: Path | None = None, *,
                 analysis_only: bool = False, spectra: str | Path | None = None) -> Project:
    """A :class:`Project` from the parsed TOML ``data``; relative paths are taken relative to
    ``base``. With ``analysis_only`` the ``[input]`` table may be left out when there is an
    ``[analysis]`` one (comparing bundles needs no input): ``input`` is then None. ``spectra``
    (a path from the working directory) takes the place of ``[input] spectra``: the run takes
    the raw spectra stored there."""
    base = Path(base)
    if spectra is not None:
        data = {**data, "input": {**data.get("input", {}),
                                  "spectra": str(Path(spectra).resolve())}}
    _check_keys(data, {"project", "threads", "input", "run", "output", "analysis", "figures",
                       "review"}, "the project file")
    data, review_data = _review_moved(data)
    only_analysis = analysis_only and "input" not in data and "analysis" in data
    name = data.get("project", "" if only_analysis else None)
    if not isinstance(name, str) or (not name and not only_analysis) or ":" in name:
        raise ProjectError("project must be a name (non-empty text without ':')"
                           + ("; a file with only an [analysis] table may leave it out"
                              if analysis_only else ""))
    if "input" not in data and not (analysis_only and "analysis" in data):
        raise ProjectError("the project file needs an [input] table" + (
            " (or, for hxex analyse only, an [analysis] table)" if analysis_only else
            "; a file with only an [analysis] table is for hxex analyse"))
    run_table = data.get("run", {})
    if not isinstance(run_table, Mapping):
        raise ProjectError("[run] must be a table")
    run = run_config_from(run_table)
    inp = None
    if "input" in data:
        inp = _input_from(data["input"], base)
    output = _output_from(data.get("output", {}), base)
    figures = _figures_from(data["figures"]) if "figures" in data else None
    review = _review_from(review_data, base) if review_data is not None else None
    threads = data.get("threads")
    if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int)
                                or threads < 1):
        raise ProjectError("threads must be a whole number of at least 1")
    if isinstance(data.get("analysis"), list):
        raise ProjectError(
            "[[analysis]]: a project has one [analysis] table (single brackets). To compare "
            "with another reference too, run hxex analyse --reference NAME: each reference "
            "gets a folder of its own (results_analysis_NAME)")
    analysis = _analysis_from(data["analysis"], base) if "analysis" in data else None
    return Project(name, inp, run, output, path, threads, analysis, figures, review)


def _review_moved(data: Mapping) -> tuple[Mapping, Mapping | None]:
    """``data`` without ``[[analysis.exclude]]``, and the ``[review]`` table with those rules in
    it: they were the after-fitting rules before ``[review]`` held them (owner, 2026-10-20)."""
    review = data.get("review")
    if review is not None and not isinstance(review, Mapping):
        raise ProjectError("[review] must be a table")
    analysis = data.get("analysis")
    if not isinstance(analysis, Mapping) or "exclude" not in analysis:
        return data, review
    if review is not None and "exclude" in review:
        raise ProjectError("[[analysis.exclude]] and [[review.exclude]]: the rules are "
                           "[[review.exclude]] now; move the [[analysis.exclude]] ones there")
    log.warning("[[analysis.exclude]] is called [[review.exclude]] now (the decisions made "
                "after looking at the results have a [review] table of their own); the old "
                "name still works: rename it to stop this message")
    data = {**data, "analysis": {k: v for k, v in analysis.items() if k != "exclude"}}
    return data, {**(review or {}), "exclude": analysis["exclude"]}


def _review_from(data: Mapping, base: Path):
    """The :class:`hxex.review_table.ReviewSettings` of the ``[review]`` table (its ``table``
    taken relative to ``base``)."""
    from .exclusions import read_exclusions
    from .review_table import ReviewSettings

    _check_keys(data, {"table", "exclude"}, "[review]")
    table = data.get("table")
    if table is not None and table is not False and not isinstance(table, str):
        raise ProjectError("[review] table must be the review table's file name (text)")
    try:
        rules = read_exclusions(data.get("exclude"), "review.exclude")
    except ValueError as exc:
        raise ProjectError(str(exc)) from None
    path = _path(base, table, "review.table")
    return ReviewSettings(table=None if path is None else str(path), exclude=rules)


def _figures_from(data) -> Any:
    """The :class:`hxex.report.PlotConfig` of the ``[figures]`` table."""
    from .report.figures import PlotConfig

    if not isinstance(data, Mapping):
        raise ProjectError("[figures] must be a table")
    values = _false_to_none(data, PlotConfig)
    _check_types(values, PlotConfig, "figures")
    try:
        return PlotConfig.from_dict(_merge(PlotConfig().to_dict(), values, "figures"))
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"[figures]: {exc}") from exc


ANALYSIS_KEYS = ("reference", "samples", "times", "exclude_times", "ks", "fasta",
                 "coverage_times", "redundancy", "weight_exponent", "filters", "choice", "alpha",
                 "regions", "sites",
                 "exclude_flagged", "format", "out", "runs", "overlay",
                 "overlay_two_only", "peptide_match", "populations", "borderline_evidence",
                 "modified_peptides",
                 "controls", "merge",
                 "fragments", "protection", "labelling")
# removed (owner, 2026-10-19; docs/HISTORY.md item 106): refused with the reason
_REMOVED_ANALYSIS_KEYS = ("protection_residues", "protection_r1", "protection_max_se")


def _analysis_from(data: Mapping, base: Path):
    """The :class:`hxex.analysis.AnalysisConfig` of an ``[analysis]`` table (relative paths from
    ``base``)."""
    from .analysis import AnalysisConfig

    removed = [k for k in _REMOVED_ANALYSIS_KEYS if k in data]
    if removed:
        raise ProjectError(
            f"[analysis] {', '.join(removed)}: residue-level protection factors from peptides "
            "were removed, because the factors that overlapping peptides give for single "
            "residues are unreliable. Peptide-level protection factors stay (protection = "
            "true); residue-level rates and protection factors come from fragment ladders "
            "([analysis.fragments], ladder_residues.csv). Delete the setting")
    _check_keys(data, ANALYSIS_KEYS, "[analysis]")
    data = _times_in(data, ("times", "exclude_times"), "analysis")
    if isinstance(data.get("coverage_times"), list):
        data = _times_in(data, ("coverage_times",), "analysis")
    if isinstance(data.get("controls"), Mapping):
        _refuse_control_times(data["controls"], "[analysis.controls]",
                              "the controls are the spectra with a role; undeut_at and "
                              "fulldeut_at make the spectra at a time serve as a control as "
                              "well")
        data = {**data, "controls": _times_in(data["controls"], ("undeut_at", "fulldeut_at"),
                                              "analysis.controls")}
    if isinstance(data.get("fragments"), Mapping):
        data = {**data, "fragments": _times_in(data["fragments"], ("min_time",),
                                               "analysis.fragments")}
    values = _false_to_none({k: v for k, v in data.items() if k not in ("regions", "sites")},
                            AnalysisConfig)
    if values.get("labelling") is False:  # typed Any: the helper cannot tell
        values["labelling"] = None
    if isinstance(values.get("controls"), Mapping):  # changes to [run.controls]: its false rule
        from .controls import ControlConfig

        values["controls"] = _false_to_none(values["controls"], ControlConfig)
    _check_types(values, AnalysisConfig, "analysis")
    fasta = values.get("fasta")
    if isinstance(fasta, str):
        values["fasta"] = str(_path(base, fasta, "analysis.fasta"))
    elif isinstance(fasta, Mapping):  # per sample, or per run then sample
        values["fasta"] = {
            str(k): ({str(s): str(_path(base, w, f"analysis.fasta.{k}.{s}"))
                      for s, w in v.items()} if isinstance(v, Mapping)
                     else str(_path(base, v, f"analysis.fasta.{k}")))
            for k, v in fasta.items()}
    if "out" in values:
        values["out"] = str(_path(base, values["out"], "analysis.out"))
    if values.get("runs") is not None:
        if not isinstance(values["runs"], Mapping):
            raise ProjectError("[analysis.runs] must be a table of run name = bundle")
        values["runs"] = {str(k): str(_path(base, v, f"analysis.runs.{k}"))
                          for k, v in values["runs"].items()}
    values["annotations"] = {"regions": list(_items(data.get("regions"))),
                             "sites": list(_items(data.get("sites")))}
    try:
        if isinstance(values.get("filters"), Mapping):
            from .analysis import SpeciesFilters

            values["filters"] = {**SpeciesFilters().__dict__, **values["filters"]}
        if "choice" in values:
            from .analysis.settings import choice_defaults

            if not isinstance(values["choice"], Mapping):
                raise TypeError("choice must be a table")
            # the keys not given take the defaults of the populations rule
            base = choice_defaults(values.get("populations", "best_set")).to_dict()
            values["choice"] = {**base, **values["choice"]}
        return AnalysisConfig.from_dict(values)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ProjectError(f"[analysis]: {exc}") from exc


# the tables that hold a list of items (``[[name]]``; a single ``[name]`` is read as one item)
LIST_TABLES = ("input.runs", "input.fragments", "input.exclude", "analysis.regions",
               "analysis.sites", "analysis.exclude", "review.exclude", "run.registry.residues")


def _toml_hint(text: str, exc: Exception) -> str:
    """A hint for the commonest mistakes in a project file: a Windows path with backslashes
    between double quotes, where a backslash starts an escape ("C:\\Users" is an error), and
    a heading in single brackets written twice (several items of a list need double
    brackets; any other table's settings go under one heading)."""
    import re

    found = re.search(r"at line (\d+)", str(exc))
    lines = text.splitlines()
    if found is None or not 0 < int(found[1]) <= len(lines):
        return ""
    line = lines[int(found[1]) - 1]
    twice = re.search(r"Cannot declare \(([^)]*)\) twice", str(exc))
    if twice:
        name = ".".join(p for part in twice[1].split(",") if (p := part.strip(" '\"")))
        if name in LIST_TABLES:
            return (f"; a heading in single brackets, [{name}], may appear only once: for "
                    f"several items, write each under [[{name}]], with double brackets")
        return f"; [{name}] may appear only once: put all its settings under one heading"
    if re.search(r'"[^"]*\\[^"]*"', line):
        return ("; a backslash between double quotes starts an escape: write the path with "
                "forward slashes (\"C:/Users/me/data\") or between single quotes "
                "('C:\\Users\\me\\data')")
    return ""


def load_project(path: str | Path, *, analysis_only: bool = False,
                 spectra: str | Path | None = None) -> Project:
    """Read and check a project file (``analysis_only``: one that may have only an
    ``[analysis]`` table, for :func:`hxex.report.write_analysis`; ``spectra``: stored raw
    spectra in place of the input, :func:`project_from`)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        data = tomllib.loads(text)
    except OSError as exc:
        raise ProjectError(f"cannot read {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ProjectError(f"{path}: not valid TOML ({exc}){_toml_hint(text, exc)}") from exc
    return project_from(data, path.resolve().parent, path, analysis_only=analysis_only,
                        spectra=spectra)


# --- reading the input --------------------------------------------------------------------------

def scan_peptides(project: Project):
    """The peptide table of an mzML or Thermo project, checked against its runs: a row's
    ``sample`` (empty: every run) must be the sample of a run, and a peptide (with its charge and
    modification) may be listed once per sample, or once with an empty sample. A row may also
    name one ``run`` (the file name of an ``[[input.runs]]`` path): it is read from that run
    only, and once per run; a run with a row of its own does not read the peptide's general row
    (the general row's entry gets ``not_runs``), so a run's row overrides it (its own
    ``rt_start``, ``rt_end``, say)."""
    from .io.entries import SCAN_REQUIRED, read_peptide_table
    from .io.mzml import run_name

    inp = project.input
    entries = read_peptide_table(inp.peptides, required=SCAN_REQUIRED)
    samples = sorted({r.sample for r in inp.runs})
    run_samples = {run_name(r): r.sample for r in inp.runs}
    seen: dict[tuple, list[str]] = {}
    own: dict[tuple, set[str]] = {}  # peptide key -> the runs with a row of their own
    for line, entry in enumerate(entries, start=2):
        name = f"{inp.peptides} line {line}: {entry.peptide} (charge {entry.charge})"
        if entry.sample and entry.sample not in samples:
            raise ProjectError(f"{name}: sample {entry.sample!r} is not the sample of any "
                               f"[[input.runs]] ({', '.join(samples)}); a peptide is read from "
                               "the runs of its sample, or from every run when its sample is "
                               "empty")
        key = (entry.peptide, entry.start_seq, entry.end_seq, entry.charge, entry.modification)
        run = str(dict(entry.extra).get("run", "") or "").strip()
        if run:
            if run not in run_samples:
                raise ProjectError(f"{name}: run {run!r} is not the file of any [[input.runs]]; "
                                   "the run column names a run by its file name")
            if entry.sample and run_samples[run] != entry.sample:
                raise ProjectError(f"{name}: run {run!r} is a run of sample "
                                   f"{run_samples[run]!r}, not {entry.sample!r}")
            if run in own.setdefault(key, set()):
                raise ProjectError(f"{name}: listed again for run {run!r}; list a peptide once "
                                   "per run")
            own[key].add(run)
            continue
        before = seen.setdefault(key, [])
        if entry.sample in before or (before and "" in (entry.sample, *before)):
            raise ProjectError(f"{name}: listed again for "
                               f"{'every run' if not entry.sample else 'sample ' + entry.sample}"
                               "; list a peptide once per sample (or once with an empty "
                               "sample for every run)")
        before.append(entry.sample)
    problems = fragment_parent_problems(project, entries)
    if problems:
        raise ProjectError("; ".join(problems))
    if own:  # the general rows step aside for the runs that have their own
        entries = [e if str(dict(e.extra).get("run", "") or "").strip() or not own.get(
            (e.peptide, e.start_seq, e.end_seq, e.charge, e.modification)) else
            replace(e, extra=tuple(dict(e.extra, not_runs=";".join(sorted(own[(
                e.peptide, e.start_seq, e.end_seq, e.charge, e.modification)]))).items()))
            for e in entries]
    return entries


def fragment_parent_problems(project: Project, entries) -> list[str]:
    """With ``[[input.fragments]]``: one problem per peptide of the table whose
    ``modification`` its fragments cannot take (:func:`hxex.fragments.passable_to_fragments`),
    the last followed by why and what to do; none without fragments. A row is both the peptide
    and the parent of its fragments, so the refusal covers its MS1 spectra too."""
    from .fragments import passable_to_fragments

    inp = project.input
    if not inp.fragments:
        return []
    problems = [f"{inp.peptides} line {line}: {e.peptide} (charge {e.charge}) has "
                f"modification {e.modification!r}"
                for line, e in enumerate(entries, start=2)
                if not passable_to_fragments(e.modification)]
    if problems:
        problems[-1] += (
            "\n  a project with [[input.fragments]] cannot read peptides with a modification "
            "(composition, Hex, undeut_mz or an ion type), at MS1 either: its fragments cannot "
            "take it. Write modified residues into the sequence as modX labels, or use two "
            "projects: one without [[input.fragments]] for the MS1 spectra with the "
            "modification, and one with [[input.fragments]] and parents = false, listing the "
            "peptide without it, for the fragments")
    return problems


def scan_targets(project: Project) -> list[MzmlTarget]:
    """The targets of an mzML or Thermo project: each peptide of the table (if ``parents``),
    and its fragments (one set per ``[input.fragments]`` table), each read from the runs of its
    sample (every run when the table gives none; :func:`scan_peptides`). A peptide's
    ``rt_start`` and ``rt_end`` columns set the retention window of its targets, in place of
    ``rt_window``."""
    from .fragments import fragment_targets
    from .spectra import entry_rt, entry_rt_window

    inp = project.input
    targets = []
    for line, entry in enumerate(scan_peptides(project), start=2):
        if (inp.scans.find_peak and inp.parents and entry_rt_window(entry) is None
                and not math.isfinite(entry_rt(entry))):
            raise ProjectError(f"{inp.peptides} line {line}: {entry.peptide} (charge "
                               f"{entry.charge}): [input.scans] find_peak needs each peptide's "
                               "rt (minutes) to look near, or its own rt_start, rt_end")
        # the peptide's own retention window (rt_start, rt_end) overrides the project's
        window = entry_rt_window(entry)
        if inp.parents:
            target = inp.target if window is None else {**inp.target, "rt_window": window}
            targets.append(MzmlTarget(entry, **target))
        for f in inp.fragments:
            targets += fragment_targets(
                entry, f.ions, ms_level=f.ms_level, activation=f.activation,
                preactivation=f.preactivation, parent_mz=f.parent_mz,
                precursor_mz=f.precursor_mz, precursor_charge=f.precursor_charge,
                filter_contains=f.filter_contains, filter_excludes=f.filter_excludes,
                rt_window=f.rt_window if window is None else window,
                limit_charge=f.limit_charge, label=f.label,
                picker=project.run.picker, registry=project.run.registry,
            )
    if not targets:
        raise ProjectError(f"{inp.peptides}: no peptides")
    return targets


def folder_entries(project: Project):
    """The entries of a folder-format project: those of the peptide table when it names one
    (pyHXExpress's metadf: file, sample, peptide, ...), else those found in ``data_dir`` by file
    name."""
    from .io import read_peptide_table, scan

    inp = project.input
    if inp.peptides is not None:
        return read_peptide_table(inp.peptides)
    return scan(inp.data_dir, inp.format, samples=inp.samples,
                peptide_ranges=inp.peptide_ranges, peptide_region=inp.peptide_region,
                sample=inp.sample,
                registry=project.run.registry)


def _is_spectra_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".parquet"


def stored_spectra_source(path: str | Path) -> dict | None:
    """The read record of the raw spectra stored in ``path`` (a bundle, as a directory or
    ``.zip``, or a spectra file of :func:`hxex.io.save_spectra`), without reading the
    spectra; None when they have none. A bundle without raw spectra is an error."""
    import json

    from .io import BundleError, open_bundle, spectra_info

    path = Path(path)
    if not path.exists():
        raise ProjectError(f"input.spectra: {path} does not exist")
    try:
        if _is_spectra_file(path):
            source = spectra_info(path).get("source")
            return None if not source else json.loads(source)
        with open_bundle(path) as b:
            if not b.has("spectra"):
                raise ProjectError(f"input.spectra: the bundle {path} holds no raw spectra "
                                   "(it was written with keep_spectra = false)")
            return b.spectra_source()
    except BundleError as exc:
        raise ProjectError(f"input.spectra: {exc}") from exc


def load_stored_spectra(path: str | Path) -> SpectrumSet:
    """The raw spectra stored in ``path`` (see :func:`stored_spectra_source`), with their read
    record, to which ``stored_in`` (the path they were taken from) is added."""
    from .io import BundleError, load_spectra, open_bundle

    path = Path(path)
    stored_spectra_source(path)  # exists and holds spectra
    try:
        if _is_spectra_file(path):
            spectra = load_spectra(path)
        else:
            with open_bundle(path) as b:
                spectra = b.spectra()
    except BundleError as exc:
        raise ProjectError(f"input.spectra: {exc}") from exc
    return replace(spectra, source={**(spectra.source or {}), "stored_in": str(path)})


def project_with_spectra(project: Project, path: str | Path) -> Project:
    """``project`` taking the raw spectra stored in ``path`` (as ``[input] spectra``; a
    relative path is taken from the working directory). The project is read again from its
    file, so the format, the control times and everything that follows from them are those of
    the stored spectra (:func:`load_project` with ``spectra``)."""
    if project.path is None:
        raise ProjectError("the project was not read from a file: give spectra in [input]")
    return load_project(project.path, spectra=path)


_READ_STAGE = ("format", "data_dir", "samples", "peptide_ranges", "peptide_region", "sample",
               "peptides",
               "runs", "parents", "target", "fragments", "scans", "data_root", "thermo_dir")


def read_input(
    project: Project,
    *,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> SpectrumSet:
    """The raw spectra the project's ``[input]`` names (recorded, so the read can be
    repeated), with the retention times of ``retention_times`` when it names a table. With
    ``spectra`` they are the spectra stored by an earlier run (:func:`load_stored_spectra`):
    the settings of the read itself (files, targets, scans) do not apply then. The read record
    (``source``) carries ``time_scale``, so the results know whether the times are real."""
    inp = project.input
    if inp.spectra is not None:
        default = InputSettings(inp.format)
        base = project.path.resolve().parent if project.path is not None else Path.cwd()
        unused = [k for k in _READ_STAGE if k != "format"
                  and getattr(inp, k) != getattr(default, k)
                  and not (k == "data_root" and inp.data_root == base)]
        log.info("taking the raw spectra stored in %s%s", inp.spectra,
                 f"; not used: input {', '.join(unused)}" if unused else "")
        spectra = load_stored_spectra(inp.spectra)
    else:
        spectra = _read_input(project, progress=progress, cancel=cancel)
    if (spectra.source or {}).get("time_scale", "seconds") != inp.time_scale:
        if inp.spectra is not None:
            log.info("input.time_scale = %r replaces the stored spectra's %r", inp.time_scale,
                     (spectra.source or {}).get("time_scale", "seconds"))
        spectra = replace(spectra, source={**(spectra.source or {}),
                                           "time_scale": inp.time_scale})
    if project.input.retention_times is not None:
        from .io import apply_retention_times, read_retention_times

        spectra = apply_retention_times(spectra, read_retention_times(
            project.input.retention_times))
    return spectra


def _read_input(project: Project, *, progress=None, cancel=None) -> SpectrumSet:
    inp = project.input
    if inp.format in FOLDER_FORMATS:
        from .io import read_spectra

        return read_spectra(
            folder_entries(project), inp.data_dir, inp.format, project=project.project,
            samples=inp.samples,
            peptide_ranges=inp.peptide_ranges, peptide_region=inp.peptide_region,
            record_paths=inp.record_paths,
            progress=progress, cancel=cancel,
        )
    common = dict(project=project.project, config=inp.scans, picker=project.run.picker,
                  registry=project.run.registry, data_root=inp.data_root,
                  record_paths=inp.record_paths, progress=progress, cancel=cancel)
    targets = scan_targets(project)
    if inp.format == "mzml":
        from .io import read_mzml

        return read_mzml(list(inp.runs), targets, **common)
    from .io import read_thermo

    return read_thermo(list(inp.runs), targets, thermo_dir=inp.thermo_dir, **common)


# --- the template -------------------------------------------------------------------------------

def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, Mapping):
        return "{ " + ", ".join(f"{k} = {_toml_value(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"cannot write {value!r} as TOML")


_COMMENT_COLUMN = 34  # where a setting's trailing comment starts
_LINE_WIDTH = 96

# What each generated template line means, with its unit, keyed table.key. A setting whose
# default is None says what false means. Settings with the same name elsewhere cross-refer.
_COMMENTS: dict[str, str] = {
    "input.scans.group": 'one spectrum per scan ("each"), per "every" scans summed, or per '
                         'time window ("windows"); runs with a fixed time sum all their scans',
    "input.scans.every": 'scans summed per spectrum with group = "every"',
    "input.scans.windows": 'with group = "windows": [[start, end], ...] in s of scan time',
    "input.scans.min_time": "s of scan time: leave out every run's scans before it (false: "
                            "none); not the exchange-clock min_time of [analysis.fragments]",
    "input.scans.max_spectra": "most spectra per target in a run without a fixed time, spread "
                               "over log time (false: all)",
    "input.scans.mz_margin": "m/z kept on either side of a target's predicted bins",
    "input.scans.sum_ppm": "centroided peaks closer than this (ppm) are summed",
    "input.scans.gap_factor": "profile scans: a point spacing this many times its neighbours' "
                              "is a stretch the instrument left out",
    "input.scans.precursor_tol": "m/z tolerance matching a scan's precursor to a target's",
    "input.scans.find_peak": "LC-MS runs with a fixed time: find each peptide's elution peak "
                             "near its rt (peptide table, minutes) and sum that peak's "
                             "scans; rows with rt_start, rt_end keep them",
    "input.scans.peak_search": "s either side of a peptide's rt that its peak is looked for in",
    "input.scans.peak_smoothing": "s: the chromatogram's running mean (false: half the run's "
                                  "typical peak width, measured)",
    "input.scans.peak_width": "the window: the apex +- this many peak widths at half maximum",
    "run.envelope_fallback_time": "s: spectra at this time are the envelope references when a "
                                  "series' undeuterated controls are missing or poor (false: "
                                  "none)",
    "run.n_max_factor": "n (exchangeable sites) may be up to this times the amide count",
    "run.drop_no_signal": "drop spectra the picker found no signal in (empty ones always are)",
    "run.memory_budget_bytes": "memory of one chunk of the fit, bytes (false: automatic)",
    "run.max_chunk": "most measurements in one chunk (false: no cap)",
    "run.picker.resolution_ppm": "half-width (ppm) of the window a peak is picked in around "
                                 "its predicted m/z (never above half the isotope spacing); "
                                 "not a resolving power",
    "run.picker.zero_filling": "bins picked after the last one above the noise threshold",
    "run.picker.noise_threshold": "absolute intensity below which a point is noise (false: "
                                  "noise_percent of the tallest point instead)",
    "run.picker.noise_percent": "noise threshold as percent of the spectrum's tallest point",
    "run.picker.env_threshold": "fraction of the tallest bin at which the envelope width is "
                                "measured",
    "run.picker.nterm_subtract": "N-terminal residues without an exchanging amide (the amide "
                                 "count bounds the bins)",
    "run.picker.pick": 'what a bin takes: "peak" (the tallest peak whose apex is in the window) '
                       'or "window_max" (the tallest point in the window, as pyHXExpress)',
    "run.picker.shoulder_min_points": "with pick = \"window_max\": points in a window from which "
                                      "a maximum at its edge counts as a neighbour's shoulder "
                                      "(bin 0)",
    "run.picker.ladder": 'bin positions: "composition" (isotope clusters of the composition) '
                         'or "exms" (pyHXExpress\'s fixed steps)',
    "run.picker.resolving_power": "window from the instrument instead of resolution_ppm: R "
                                  'quoted at resolving_power_mz, or "recorded" (each '
                                  "spectrum's value from Thermo raw files; hxex check shows "
                                  "them); false: the resolution_ppm window",
    "run.picker.resolving_power_mz": "m/z the resolving power is quoted at (Thermo: 200; 400 "
                                     "on the first Orbitrap generation)",
    "run.picker.resolving_power_exponent": "R falls as (m0/m)^e: 0.5 Orbitrap, 0 TOF, 1 FT-ICR",
    "run.picker.window_fwhm": "half-width as a multiple of the peak width at half maximum, "
                              "m/R(m); never below mass_accuracy_ppm, never above half the "
                              "isotope spacing",
    "run.picker.mass_accuracy_ppm": "floor (ppm) of the resolving-power window",
    "run.picker.off_mz_sd": "off_mz: a series whose undeuterated controls put its envelope "
                            "further than this many times the run's scatter from the run's "
                            "calibration (at least 3 ppm; peptide ions only; false: not "
                            "checked)",
    "run.picker.recentre": "move the windows of an off_mz series onto its undeuterated "
                           "envelope's peaks (false: flag only)",
    "run.fit.k_min": "fewest populations tried (must be 1 with screen; the controls are also "
                     "fitted at K = 1 for the back-exchange correction, whatever k_min)",
    "run.fit.k_max": "most populations tried (equal to k_min: that number only, no choice)",
    "run.fit.criterion": '"bic" or "aic": chooses K among those tried',
    "run.fit.min_population": "a K whose smallest population is below this is rejected (0: "
                              "off); [analysis.filters] and [analysis.choice] have their own",
    "run.fit.n_restarts": "starting points per K",
    "run.fit.adam_steps": "Adam steps per restart",
    "run.fit.k1_adam_steps": "Adam steps of the K=1 fit only (false: adam_steps); 200 is "
                             "enough and faster",
    "run.fit.lbfgs_steps": "L-BFGS iterations after Adam (0: none)",
    "run.fit.lr_adam": "Adam learning rate at the start",
    "run.fit.lr_adam_end": "Adam learning rate at the end (cosine schedule)",
    "run.fit.nested_init": "start each K from the K-1 fit with its heaviest population split",
    "run.fit.kmeans_after_warm": "also start from k-means when the nested start is used",
    "run.fit.split_delta": "logit shift between the halves of a split population",
    "run.fit.warm_noise": "sd of the jitter added to a warm start",
    "run.fit.grad_clip": '"per_measurement" (each fit on its own) or "global" gradient '
                         "clipping",
    "run.fit.polish_steps": "Levenberg-Marquardt iterations per measurement after the restarts "
                            "(0: none)",
    "run.fit.seed": "seed of everything random (each measurement is also seeded by its id)",
    "run.fit.screen": "fit K > 1 only where it could be selected: about a third of the time, "
                      "but the skipped fits are missing from fixed-K analyses (with "
                      "[run.population_test]: only where the test could find it)",
    "run.fit.noise.model": '"gaussian" or "poisson"',
    "run.fit.noise.sd": "gaussian: the sd in intensity units (false: percent instead)",
    "run.fit.noise.percent": "sd (or poisson scale) as percent of the tallest peak of the "
                             "series' undeuterated reference spectrum",
    "run.fit.noise.n_eff": "poisson: effective counts in the tallest peak (false: percent)",
    "run.fit.noise.estimate": '"replicates": estimate each series\' noise from its replicate '
                              "spectra (false: the level above)",
    "run.fit.noise.min_replicated": "estimate: fewest time points with two or more replicates",
    "run.fit.noise.min_share": "estimate: smallest share of a spectrum's intensity a bin needs",
    "run.fit.noise.outlier_score": "estimate: a replicate scoring above this is left out and "
                                   "flagged noise_outlier",
    "run.fit.backend.device": '"auto" (CUDA if available, else CPU), "cpu", "cuda" or "cuda:N"',
    "run.fit.backend.dtype": '"auto" (float32), "float32" or "float64" (the last digits)',
    "run.bootstrap.n_boot": "resamples of every measurement: 20 gives usable standard errors; "
                            "raise it (100 or more) when the 95 % intervals matter",
    "run.bootstrap.ci": "interval width in percent (95.0), not a fraction",
    "run.bootstrap.n_restarts": "starting points per refit (the first is the original fit)",
    "run.bootstrap.adam_steps": "Adam steps per refit",
    "run.bootstrap.lr_adam": "Adam learning rate at the start of a refit",
    "run.bootstrap.lr_adam_end": "Adam learning rate at the end of a refit",
    "run.bootstrap.warm_noise": "sd of the jitter on the original fit",
    "run.bootstrap.polish_steps": "Levenberg-Marquardt iterations per refit (0: none)",
    "run.bootstrap.grad_clip": "as in [run.fit]",
    "run.bootstrap.seed": "seed of the resampling and the restarts",
    "run.bootstrap.keep_draws": "keep every resample's fit in the bundle",
    "run.bootstrap.curve_band": "also the band of the total uptake curve",
    "run.population_test.simulations": "simulated spectra per spectrum and step for the test's "
                                       "reference (20; time grows with it)",
    "run.population_test.significance": "a further population counts when the chance of "
                                        "seeing the evidence without it is below this",
    "run.population_test.min_population": "... and the further population is at least this "
                                          "share of the molecules",
    "run.population_test.null": '"simulated", or "chi_square" for a quick first look without '
                                "simulations (less exact)",
    "run.population_test.seed": "seed of the simulations",
    "run.bootstrap.ks": '"selected" (each measurement at its own K), "all" fitted K, or a '
                        "list of K (fixed-K bootstraps for the analysis)",
    "run.controls.m0_max": "largest mean (D) of a good undeuterated control",
    "run.controls.retention_min": "a fixed smallest deuterium retention of a good fully "
                                  "deuterated control, as a fraction (false: none; the "
                                  "controls are judged against the run, outlier_sd, or the "
                                  'kint prediction); not a retention time',
    "run.controls.retention_max": "largest deuterium retention of a good fully deuterated "
                                  "control, as a fraction",
    "run.controls.outlier_sd": "a fully deuterated control this many robust standard "
                               "deviations below the median retention of the run's controls "
                               "is rejected (without retention_min)",
    "run.controls.time_course_tolerance": "D per amide: a fully deuterated control below its "
                                          "series' highest time point by more than this is "
                                          "rejected (an undeuterated one above its lowest "
                                          "is flagged data_below_m0)",
    "run.controls.sample_tolerance": "a fully deuterated control whose retention differs "
                                     "this much from the same peptide's in the other "
                                     "samples is flagged fd_disagrees (still used)",
    "run.controls.combine_charges": "every charge state of a peptide (in one sample) takes the "
                                    "mean retention of its measured fully deuterated controls "
                                    "over all charge states; fd_fallback only when none is "
                                    "good (peptides only, not fragment ions)",
    "run.controls.replicate_outlier": "a control replicate this many typical replicate sd from "
                                      "its replicates' median (three or more) is left out of "
                                      "the control's mean (one above the amide count always "
                                      "is); unset: off",
    "run.controls.fd_other_charge": "with combine_charges = false: a series without a good "
                                    "fully deuterated control takes the retention of the same "
                                    "peptide's at its other charge states, before fd_fallback",
    "run.controls.d2o_fraction": "D atom fraction of the labelling buffer (0 to 1); "
                                 "[analysis.controls] may change it for the analysis",
    "run.controls.fd_fallback": 'retention of a series without a good fully deuterated control: '
                                '"d2o" (d2o_fraction), "median" (of the run\'s accepted '
                                'controls) or "kint" (predicted back-exchange; needs quench; '
                                'series it cannot predict take the median); false: "kint" with '
                                'quench, else "median"',
    "run.controls.undeut_at": 's (or "5 min"): the spectra at this time also serve as the '
                              "undeuterated control (false: only the files marked undeut)",
    "run.controls.fulldeut_at": 's (or "20 h"): the spectra at this time also serve as the '
                                "fully deuterated control, e.g. a plateaued late time point "
                                "(false: only the files marked fulldeut)",
    "run.controls.undeut_marked": "the files marked undeut serve as the control (false: only "
                                  "the spectra at undeut_at do)",
    "run.controls.fulldeut_marked": "the files marked fulldeut serve as the control (false: "
                                    "only the spectra at fulldeut_at do, e.g. when the "
                                    "measured file reads low)",
    "input.time_scale": '"seconds" (real times) or "order" (the times only order the points: '
                        "uptake drawn by index, kinetics and protection analyses refuse)",
    "run.controls.kint_tolerance": "kint: a measured control this far (D per amide) from the "
                                   "prediction is rejected",
    "run.controls.kint_min_series": "kint: fewest usable controls to fit the prediction to "
                                    "(fewer: every series takes the median)",
    "run.quality.poor_fit_factor": "poor_fit: misfit this many times the series' median",
    "run.quality.min_deviance": "poor_fit: and a reduced deviance above this",
    "run.quality.min_series_fits": "poor_fit: fewest fitted spectra in a series to judge",
    "run.quality.min_intensity": "low_intensity: total intensity below this share of the other "
                                 "replicates' mean",
    "run.quality.max_mz_offset_ppm": "mz_shift: a series' m/z offset (ppm) beyond this, for "
                                     "series [run.picker] off_mz_sd does not check (fragment "
                                     "ions; false: screen off)",
    "run.quality.saturated_ratio": "saturated: an undeuterated control whose peaks beside the "
                                   "tallest read above this times natural abundance (false: off)",
    "run.quality.tails_low_ratio": "tails_low: an undeuterated control whose tail peaks (1-10 % "
                                   "of the top) read below this times natural abundance "
                                   "(false: off)",
    "figures.series": '"flagged" (series with a flag, a rejection or a control fallback), '
                      '"all" or a list of data_id',
    "figures.kinds": "pages of a series' figure: spectra, populations, uptake",
    "figures.show": "layers of the spectra page: raw, picked, fit",
    "figures.x": 'uptake axis: "time" or "index" (unset: "time", or "index" when the '
                 'times only order the points, input.time_scale = "order")',
    "figures.scale": 'uptake axis: "linear" or "log"',
    "figures.corrected": "draw back-exchange corrected values (unset: when there are "
                         "controls)",
    "figures.format": '"pdf" (one file per series, a page per kind), "png" or "svg" (one '
                      "file per kind)",
    "figures.dpi": "resolution of PNG files and of raster images in PDF or SVG",
    "figures.overview": "also the run's overview (K selection, controls, quality flags)",
    "figures.residual": "a residual panel under every spectrum",
    "figures.k": "draw every series at this number of populations (unset: each spectrum at "
               "its own K, the fit's or the review's choice)",
    "run.envelope.mode": '"theory" (the ion type\'s isotope clusters), "mixed" (ion types a '
                         'hydrogen apart, weights fitted to the undeuterated spectra) or '
                         '"infer" (the undeuterated envelope itself)',
    "run.envelope.components": "mixed: number of ion types",
    "run.envelope.max_residual": "largest relative residual of a mixture that describes its "
                                 "references (else poor_envelope)",
    "run.envelope.iterations": "rounds of picking and fitting the weights",
    "run.envelope.share": "conditions of one ion that differ only in scan filters or label "
                          "share an envelope",
}

# comment lines written before a table's header
_SECTION_COMMENTS: dict[str, str] = {
    "run": "Run settings (hxex.run.RunConfig); times are in s, as the measurement times. The "
           "bootstrap, the population test, and the controls are stages that their table "
           "turns on: uncomment the commented-out run.bootstrap, run.population_test, or "
           "run.controls table below to use it; a stage's \"name = false\" turns it off.",
    "run.picker": "Peak picking: the spectra become sticks at the predicted isotope positions "
                  "(hxex.picking.PickerConfig; docs/PEAK_PICKING.md).",
    "run.picker.amide_rules": "Per fragment ion type, whether it keeps the amide N-H at its "
                              'cleavage site, e.g. "y-NH3" = false. Empty: the ion type\'s '
                              "composition decides (c, x, y keep it; a, b, z do not).",
    "run.fit": "The fit of k_min to k_max populations to every spectrum, the number chosen by "
               "the criterion; with k_min and k_max equal, that number only "
               "(hxex.fit.FitConfig).",
    "run.fit.noise": "Noise model and level of the fit (hxex.noise.NoiseConfig). The level "
                     "sets the loss scale and the model selection.",
    "run.fit.backend": "Where the fit runs (hxex.model.Backend); nothing is set globally.",
    "run.bootstrap": "Standard errors and intervals from resampled refits of every "
                     "measurement (hxex.bootstrap.BootstrapConfig). Slow: most of a run's time.",
    "run.population_test": "Tests how many populations each spectrum, and each set of "
                           "replicates and charge states, holds against simulated spectra "
                           "(hxex.population_test.PopulationTestConfig). Each spectrum's K is "
                           "then the test's; needs k_min = 1. Slow: ten to twenty times the "
                           "fit per step.",
    "run.controls": "Back-exchange correction from the undeuterated and fully deuterated "
                    "controls, fitted at K=1 (hxex.controls.ControlConfig). Deuterium "
                    "retention here is the fraction of deuterium kept, not a retention time.",
    "run.quality": "Quality flags on every spectrum (hxex.quality.QualityConfig): poor_fit, "
                   "low_intensity, noise_outlier, mz_shift, off_mz (thresholds in "
                   "[run.picker]), poor_envelope, saturated, tails_low, bright. Nothing is "
                   "dropped; the review layer of the bundle takes the decisions.",
    "figures": "Figures, drawn by hxex report hxex.toml --figures from the results (hxex run "
               "draws none; hxex.report.PlotConfig). Changing these needs no new run.",
    "run.registry": "Modified residues, written in sequences as modX labels (a lowercase "
                    "prefix plus the residue letter, e.g. acK, pS; a sequence is read residue "
                    "by residue). Each [[run.registry.residues]] adds delta to the base "
                    "residue's composition. The table itself stays empty.",
    "run.envelope": "The isotope envelope that places the bins and shapes the fit "
                    "(hxex.picking.EnvelopeConfig). Fragment spectra can hold several ion "
                    "types of one fragment a hydrogen apart (c-1, c, c+1 in ETD).",
}


def _commented(line: str, comment: str | None, prefix: str = "") -> list[str]:
    """``line`` with ``comment`` after it at the comment column, wrapped onto continuation lines
    that start with ``prefix`` (``'# '`` inside a commented-out table)."""
    if not comment:
        return [line]
    parts = textwrap.wrap(comment, _LINE_WIDTH - _COMMENT_COLUMN - 2)
    pad = (prefix.rstrip() or "").ljust(_COMMENT_COLUMN)
    if len(line) > _COMMENT_COLUMN - 2:  # too long for a trailing comment: comment lines after
        return [line] + [pad + "# " + p for p in parts]
    return [line.ljust(_COMMENT_COLUMN) + "# " + parts[0]] + [pad + "# " + p for p in parts[1:]]


def _section_comment(name: str) -> list[str]:
    text = _SECTION_COMMENTS.get(name)
    if not text:
        return []
    return textwrap.wrap(text, _LINE_WIDTH, initial_indent="# ", subsequent_indent="# ")


def _unset_line(key: str, hint, note: tuple[str, str] | None = None, table: str = "",
                prefix: str = "# ") -> list[str]:
    """The commented line of a setting at None: valid TOML once uncommented (``false``, the
    stand-in for None, or ``true`` for a switch that may also be unset) with a comment saying
    so, or ``note``'s value and comment instead."""
    if note is None:
        comment = _COMMENTS.get(f"{table}.{key}")
        value = "true" if bool in _union_members(hint) else "false"
        if comment is None:
            comment = ("not set; true or false sets it" if value == "true"
                       else "not set; false means unset")
        note = (value, comment)
    return _commented(f"# {key} = {note[0]}", note[1], prefix)


def _toml_table(name: str, data: Mapping, comment: bool = False, cls: type | None = None,
                notes: Mapping[str, tuple[str, str]] | None = None) -> list[str]:
    """``[name]`` with ``data``: scalars first (None as a commented line, it is the default),
    then sub-tables and arrays of tables, each line with its comment from ``_COMMENTS`` and
    the table preceded by its ``_SECTION_COMMENTS`` entry. ``cls`` is the table's config class
    (its types say how a None is written; sub-tables take theirs from it) and ``notes`` gives
    the value and comment written for the None settings it names."""
    c = "# " if comment else ""
    hints = _hints(cls) if cls is not None else {}
    notes = notes or {}
    lines = _section_comment(name) + [f"{c}[{name}]"]
    later: list[str] = []
    for key, value in data.items():
        if isinstance(value, Mapping):
            later += [""] + _toml_table(f"{name}.{key}", value, comment,
                                        _config_class(hints.get(key)))
        elif isinstance(value, list) and value and all(isinstance(v, Mapping) for v in value):
            for item in value:
                later += ["", f"{c}[[{name}.{key}]]"]
                later += [f"{c}{k} = {_toml_value(v)}" for k, v in item.items()]
        elif value is None:
            lines += _unset_line(key, hints.get(key), notes.get(key), name, c)
        else:
            lines += _commented(f"{c}{key} = {_toml_value(value)}",
                                _COMMENTS.get(f"{name}.{key}"), c)
    return lines + later


_INPUT_TEMPLATES = {
    "specexport": """[input]
format = "specexport"
# spectra = "old/results"        # take the raw spectra stored in a bundle (or a spectra
#                                # file) instead of reading the files below
data_dir = "SpecExport"          # the HDExaminer SpecExport folder
# samples = ["WT", "mutant"]     # only these samples
# peptide_ranges = ["1-10"]      # only these peptides (by residues: "0001-0010" is the same)
# peptide_region = [1, 50]       # only peptides wholly within these residues
record_paths = "path"            # or "name": record file names only
time_scale = "seconds"           # or "order": the times only order the points
# retention_times = "rt.csv"     # retention times (minutes): rt and sample, start_seq,
#                                # end_seq, peptide, charge (or an rt column in peptides)""",
    "hxexpress": """[input]
format = "hxexpress"
# spectra = "old/results"        # take the raw spectra stored in a bundle (or a spectra
#                                # file) instead of reading the files below
data_dir = "data"                # the folder of HX-Express spreadsheets
# peptides = "metadf.csv"        # a table of the files (pyHXExpress's metadf) instead of
#                                # finding them by name (SAMPLE_START-END-PEPTIDE-zCHARGE-...)
# samples = ["WT"]
# peptide_ranges = ["1-10"]
# peptide_region = [1, 50]
record_paths = "path"
time_scale = "seconds\"""",
    "pooled": """[input]
format = "pooled"
# spectra = "old/results"        # take the raw spectra stored in a bundle (or a spectra
#                                # file) instead of reading the files below
data_dir = "pooled"              # a SpecExport folder with peptides of several proteins
sample = "pool"                  # the sample name given to its entries
# peptide_ranges = ["1-10"]
# peptide_region = [1, 50]
record_paths = "path"
time_scale = "seconds\"""",
}

_SCAN_TEMPLATE = """[input]
format = "{fmt}"
# spectra = "old/results"        # take the raw spectra stored in a bundle (or a spectra
#                                # file) instead of reading the files below
peptides = "peptides.csv"        # the peptide table (columns as pyHXExpress's metadf)
data_root = "."                  # relative run paths are taken from here
# retention_times = "rt.csv"     # retention times (minutes): rt and sample, start_seq,
#                                # end_seq, peptide, charge (else from the peptide table's
#                                # rt column, else from the scans of fixed-time runs)
record_paths = "path"            # or "relative" (to data_root) or "name"{thermo}
time_scale = "seconds"           # or "order": the times only order the points
parents = true                   # read each peptide itself (set false for fragments only)

[input.target]                   # how the peptides' scans are chosen
ms_level = 1
# rt_window = [60.0, 180.0]      # seconds, for runs with a fixed time
#                                # (a peptide's own: rt_start, rt_end columns of the table, min)
# filter_excludes = "sid"        # leave out scans whose filter contains this

# [[input.fragments]]            # fragment ions of every peptide (repeat the table for
#                                # other conditions, e.g. with and without in-source activation)
# ions = "c-1:ALL z-dot:ALL"     # space-separated TYPE:LENGTHS:CHARGES; LENGTHS and CHARGES
#                                # are ALL (the default) or numbers and ranges ("2-6,9", "3-");
#                                # types as c, c-1, z-dot, y-NH3, M (the peptide itself)
# label = "gentle"               # names the condition in tables and figures
# ms_level = 2                   # 3 for MS3 scans
# activation = "ETD"             # of the last stage
# preactivation = "HCD"          # MS3: the activation before the last stage ("none" for none)
# parent_mz = "parent"           # scans whose parent isolation overlaps the peptide's m/z range
#                                # ([low, high] to set it, false for any parent)
# precursor_charge = "parent"    # or a charge, or false for any charge
# filter_excludes = "sid"

# How scans become spectra (hxex.io.MzmlConfig).
{scans}

# one [[input.runs]] per file. "role" marks a control file: "undeut" (undeuterated) or
# "fulldeut" (fully deuterated); its scans are summed into one spectrum without a time. "time"
# is the labelling time of the whole file (seconds, or text with a unit: "60 s", "5 min",
# "20 h"); leave it out when each scan has its own time (continuous labelling). A file with
# both role and time is a time point that also serves as the control. Two clocks: the scan
# times (the acquisition clock) and the exchange clock, scan time + "time_offset" (when
# exchange began relative to acquisition). "usable_from" (acquisition clock) leaves out this
# file's scans before it (e.g. spurious early points): it does not move the time axis,
# time_offset does.
[[input.runs]]
path = "UN_01.{ext}"
sample = "WT"
rep = 1
role = "undeut"

[[input.runs]]
path = "D_60s_01.{ext}"
sample = "WT"
rep = 1
time = "60 s\""""


def template(fmt: str = "specexport", project: str = "my_project") -> str:
    """A project file for the format ``fmt`` with every run setting at its default (optional
    stages commented out), to edit."""
    if fmt not in FORMATS:
        raise ProjectError(f"format must be one of {', '.join(FORMATS)}")
    if fmt in SCAN_FORMATS:
        scans = "\n".join(_toml_table("input.scans", MzmlConfig().to_dict(), cls=MzmlConfig))
        thermo = ('\n# thermo_dir = "/path/to/ThermoRawFileParser"   # Thermo\'s DLLs, or set '
                  'HXEX_THERMO_DIR' if fmt == "thermo" else "")
        input_text = _SCAN_TEMPLATE.format(fmt=fmt, scans=scans, thermo=thermo,
                                           ext="raw" if fmt == "thermo" else "mzML")
    else:
        input_text = _INPUT_TEMPLATES[fmt]
    run = RunConfig().to_dict()
    lines = [
        "# hxex project file (TOML). A key left out keeps its default; false stands for",
        "# \"not set\" where a setting can be unset (a switch set to false is off). The",
        "# control times of [run.controls] left out are the input's. Relative paths are",
        "# relative to this file.",
        "",
        f'project = "{project}"              # prefixes every spectrum id',
        "# threads = 12                    # CPU threads (default: PyTorch's, the physical",
        "#                                 # cores); 12 was fastest on a 24-core CPU",
        "",
        input_text,
        "",
        "# [[input.exclude]]               # spectra left out before fitting (stored, not",
        '# sample = "WT"                   # fitted): a spectrum matching every key is left out;',
        '# peptide_range = "45-60"         # repeat the table for each rule (keys: sample,',
        "# charge = 3                      # peptide_range, charge, time, role, rep, file,",
        '# note = "co-eluting species"     # spectrum_id, note)',
        "",
        "[output]",
        "# The bundle and the folders named after it (results_tables, results_figures,",
        "# results_analysis_..., and the earlier ones results.1, results_figures.1, ...) belong",
        "# to hxex: run, report and analyse stop at a folder that holds results unless",
        "# overwrite or keep_previous (here, or --overwrite / --keep-previous) says what to do.",
        "# Keep your own files elsewhere (files of yours inside them are kept, but one named",
        "# like an hxex file stops the command).",
        'bundle = "results"               # name of the folder (or .zip archive) hxex creates',
        "#                                 # for the results: it must not exist yet",
        "overwrite = false                # replace earlier results in the bundle (its review",
        "#                                 # is carried over), tables, figures and analysis",
        "keep_previous = false            # move earlier results first: the bundle with its",
        "#                                 # folders to results.1, a figures or analysis",
        "#                                 # folder alone to NAME.1 (report, analyse)",
        "keep_spectra = true              # store the raw spectra in the bundle",
        "tables = true                    # CSV tables beside the bundle, or a directory",
        'notes = ""',
        "",
        "# Decisions made after looking at the results, used by hxex run (its tables), hxex",
        "# report, and hxex analyse; the fits are kept. Uncomment to use.",
        "# [review]",
        '# table = "user_review_table.csv"  # a copy of spectra.csv with review_status,',
        "#                                 # review_K and note filled in (other columns are",
        "#                                 # ignored); a chosen review_K needs a bootstrap at",
        "#                                 # it: hxex resume hxex.toml adds the missing ones",
        "# [[review.exclude]]              # spectra treated as rejected (their fits are kept);",
        '# sample = "WT"                   # the keys of [[input.exclude]]',
        '# time = "30 s"',
        "# rep = 2",
        "",
        "# The post-fitting analysis (hxex analyse): uncomment to use. Regions and sites are",
        "# yours to define for the protein; nothing is derived.",
        "# [analysis]",
        '# reference = "WT"                # the sample the others are compared with (the',
        '#                                 # reference of exchange and labelling is a rate set)',
        '# samples = ["WT", "variant"]     # which samples, in this order (default: all)',
        "# times = [4.0, 60.0]             # time points of the per-time figures (default: all",
        "#                                 # but the fully deuterated control)",
        "# exclude_times = [0.1, 0.8]      # times left out of the whole analysis (unreliable;",
        "#                                 # the fits are kept)",
        '# fasta = "protein.fasta"         # or one per sample: [analysis.fasta] WT = "wt.fasta"',
        '#                                 # (for one run of [analysis.runs]: [analysis.fasta.A])',
        '# coverage_times = "undeuterated" # or "all", or a list of times',
        '# redundancy = "peptides"         # or "spectra"',
        "# alpha = 0.05                    # significance of the comparisons",
        '# overlay = ["low", "high", "avg"]  # by-residue comparisons: K=2 low, high; K=1 avg',
        "# overlay_two_only = true         # K=2 values only where two populations are chosen",
        '# peptide_match = "range"         # compare peptides over the same residues ("range",',
        '#                                 # mutated or not) or identical sequences ("sequence")',
        '# populations = "best_set"        # how many populations: "population_test" (the',
        '#                                 # default with [run.population_test]), "best_set"',
        '#                                 # (each peptide\'s spectra together; the default',
        '#                                 # without), or "bootstrap" (the residue\'s K=2 values',
        '#                                 # and their errors, [analysis.choice])',
        '# modified_peptides = "separate"  # modified forms as their own states ("WT pS59") or',
        '#                                 # averaged with the sample ("combined")',
        '# format = "pdf"                  # or "png", "svg"',
        "# protection = true               # opt in to the peptides' protection factors",
        "#                                 # (hxex.analysis.protection); needs the labelling table",
        "# [analysis.controls]             # control settings to change for the analysis (the",
        "# d2o_fraction = 0.85             # controls are recomputed from the run's K=1 fits)",
        "# [analysis.filters]              # which spectra and K=2 descriptions are kept",
        '# spectrum_quality_filtering = "loose"  # or "strict", "none": spectra showing another',
        "#                                 # species (spectra.csv screen_* columns)",
        "# replicate_outlier = 6           # typical replicate sd from the other replicates",
        "# above_amides = true             # leave out a K=1 mean above the amide count",
        '# stand_in_controls = "exclude"   # series without a measured fully deuterated control,',
        "#                                 # or with an undeuterated control above m0_max",
        "# min_population = 0.05",
        "# max_fD_se = 0.25",
        "# max_fD = 1.1",
        "# k1_between_ranks = true",
        "# undeuterated_cap = false        # population test: keep a peptide whose undeuterated",
        "#                                 # set has two populations at one in every sample",
        "#                                 # (off: flagged only, undeuterated_two)",
        "# [analysis.choice]               # the choice between one and two populations",
        '# confidence = 0.5                # 0.95 with populations = "bootstrap"',
        "# max_fD_se = 0.25",
        '# min_population = 0.05           # 0.10 with populations = "bootstrap"',
        '# tier = "two"                    # population_test: the least residue tier with two',
        '#                                 # populations ("two": a third of the weight agrees;',
        '#                                 # "localised": half); the three limits above unused',
        "# [[analysis.regions]]",
        '# name = "N-terminal region"',
        "# start = 1",
        "# end = 63",
        "# [[analysis.sites]]",
        "# residue = 19",
        '# label = "S19"',
        "# [analysis.runs]                 # compare separate runs: the tables get a run column;",
        '# A = "runA.zip"                  # reference = "WT", or "A:WT" when WT is in several',
        '#                                 # runs; default: this project\'s bundle',
        '# B = "../other/runB.zip"',
        "# [analysis.merge]                # combine runs: these samples become one, pooled",
        '# "A:WT" = "WT"',
        '# "B:WT" = "WT"',
        "# [analysis.fragments]            # runs with fragment ions (written whenever there are",
        '# ion_types = ["c-1", "z-dot"]    # some; fragments = false turns them off); default:',
        "# charges = [1]                   # every fragment ion type, fragment charge 1",
        "# min_time = 20.0                 # s, exchange clock: earlier points drawn, not fitted",
        "#                                 # (not [input.scans] min_time, on the scan clock)",
        "# tolerance = 0.25                # closure: c + z - M beyond this is flagged (D)",
        "# max_rate_error = 0.5            # ladder steps with rate_se / rate at most this",
        "#                                 # give the per-residue rates (ladder_residues.csv)",
        "# kint = true                     # opt in: intrinsic rates, protection factors and",
        "#                                 # theoretical curves (hxex.kint); needs exchange",
        '# [analysis.fragments.labels]     # short condition names in the figures',
        '# gentle = "G"',
        "# [analysis.fragments.exchange]   # the kinetics' solvent, for every sample (or one",
        "# ph_read = 3.0                   # table per sample: [analysis.fragments.exchange.S],",
        "#                                 # or per run: [analysis.fragments.exchange.A.S])",
        "# temperature = 20.0              # degrees Celsius",
        "# d2o_fraction = 0.05             # D atom fraction of the kinetics' solvent (0 to 1;",
        "#                                 # the labelling buffer's is [run.controls])",
        '# reference = "PDLA"              # intrinsic-rate set: "PDLA" or "3Ala"',
        "# initial_fraction = 0.9          # D fraction when the exchange began (curves)",
        "# [analysis.labelling]            # the labelling solvent, for every sample (or one",
        "# ph_read = 7.0                   # table per sample: [analysis.labelling.WT], or per",
        "#                                 # run of [analysis.runs]: [analysis.labelling.A.WT])",
        "# temperature = 20.0              # degrees Celsius",
        "# d2o_fraction = 0.9              # D atom fraction of the labelling buffer (as",
        "#                                 # [run.controls] d2o_fraction)",
        '# reference = "PDLA"              # intrinsic-rate set: "PDLA" or "3Ala"',
        "# initial_fraction = 0.9          # back-exchange only: the D fraction the amides",
        "#                                 # began with (above d2o_fraction: fD falls with",
        "#                                 # time); leave it out for labelling",
    ]
    scalars = {k: v for k, v in run.items()
               if not isinstance(v, Mapping) and k not in _OPTIONAL_SECTIONS}
    lines += [""] + _toml_table("run", scalars, cls=RunConfig)
    control_times = {
        "undeut_at": ('"5 min"', "the spectra at this time also serve as the undeuterated "
                                 "control (false: only the files marked undeut)"),
        "fulldeut_at": ('"20 h"', "the spectra at this time also serve as the fully "
                                  "deuterated control (false: only the files marked fulldeut)"),
        "quench": ('{ ph_read = 2.5, temperature = 0.0, d2o_fraction = 0.0 }',
                   "pH read, degrees Celsius and D fraction of the quench and LC, for "
                   'fd_fallback "kint" (false: none)'),
    }
    for key, value in run.items():
        if key in _OPTIONAL_SECTIONS and value is None:
            lines += [""] + _toml_table(f"run.{key}", _section_defaults(key), comment=True,
                                        cls=_section_class(key),
                                        notes={"controls": control_times}.get(key))
        elif isinstance(value, Mapping):
            lines += [""] + _toml_table(f"run.{key}", value, cls=_section_class(key))
    from .report.figures import PlotConfig

    lines += [""] + _toml_table("figures", PlotConfig().to_dict(), cls=PlotConfig,
                                notes={"x": ('"time"', _COMMENTS["figures.x"]),
                                       "k": ("2", _COMMENTS["figures.k"])})
    return "\n".join(lines) + "\n"


# --- the settings used: run_settings_used.toml and analysis_settings_used.toml --------------

RUN_SETTINGS_FILE = "run_settings_used.toml"  # in a bundle (and copied into its analyses)
ANALYSIS_SETTINGS_FILE = "analysis_settings_used.toml"  # in an analysis folder


def _recorded(path, mode: str) -> str | None:
    """``path`` as ``[input] record_paths`` records it: whole (``"path"``) or its name only."""
    if path is None:
        return None
    return str(path) if mode == "path" else Path(path).name


def _toml_ready(value):
    """``value`` for a TOML table: tuples as lists, None as ``false`` (the loader's None)."""
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return [_toml_ready(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _toml_ready(v) for k, v in value.items()}
    return value


def _input_record(inp: InputSettings, mode: str) -> dict[str, Any]:
    """The ``[input]`` table of ``inp`` with every setting written out (None where unset)."""
    data: dict[str, Any] = {"format": inp.format}
    if inp.spectra is not None:  # stored spectra: the read settings do not apply to them
        return {"spectra": _recorded(inp.spectra, mode), "time_scale": inp.time_scale,
                "retention_times": _recorded(inp.retention_times, mode),
                **({"exclude": [r.to_dict() for r in inp.exclude]} if inp.exclude else {})}
    if inp.format in FOLDER_FORMATS:
        data.update(
            data_dir=_recorded(inp.data_dir, mode),
            samples=None if inp.samples is None else list(inp.samples),
            peptide_ranges=None if inp.peptide_ranges is None else list(inp.peptide_ranges),
            peptide_region=None if inp.peptide_region is None else list(inp.peptide_region),
            sample=inp.sample, peptides=_recorded(inp.peptides, mode))
    else:
        from .io.mzml import _recorded_path

        root = inp.data_root
        runs = []
        for r in inp.runs:
            item = r.to_dict()
            path = Path(r.path) if root is None else root / r.path
            item["path"] = (str(r.path) if mode == "path"
                            else _recorded_path(path, root, mode))
            if item.get("time") is None:
                item.pop("time", None)  # each scan has its own time
            runs.append(_toml_ready(item))
        data.update(
            peptides=_recorded(inp.peptides, mode), data_root=_recorded(root, mode),
            thermo_dir=_recorded(inp.thermo_dir, mode), parents=inp.parents,
            target=_toml_ready(inp.target), scans=inp.scans.to_dict(), runs=runs,
            fragments=[_toml_ready({f.name: getattr(frag, f.name) for f in fields(frag)})
                       for frag in inp.fragments])
    data.update(record_paths=inp.record_paths, time_scale=inp.time_scale,
                retention_times=_recorded(inp.retention_times, mode))
    if inp.exclude:
        data["exclude"] = [r.to_dict() for r in inp.exclude]
    return data


def project_record(project: Project) -> dict[str, Any]:
    """The settings of ``project`` beside its run settings, as used: the project's name and
    threads, its ``[input]`` table (every setting written out) and its ``[output]`` table. Paths
    are recorded as ``[input] record_paths`` says (``"name"`` and ``"relative"``: file names,
    and the run paths relative to ``data_root``), as the bundle records the files it read. A run
    keeps it in its bundle (the manifest's ``configs.project_file``), and the bundle's
    ``run_settings_used.toml`` is written from it (:func:`run_settings_text`)."""
    inp = project.input
    mode = "path" if inp is None else inp.record_paths
    out = project.output

    def folder(value):
        return value if isinstance(value, bool) else _recorded(value, mode)

    return {
        "project": project.project, "threads": project.threads,
        "input": None if inp is None else _input_record(inp, mode),
        "output": {"bundle": _recorded(out.bundle, mode), "overwrite": out.overwrite,
                   "keep_previous": out.keep_previous, "keep_spectra": out.keep_spectra,
                   "tables": folder(out.tables), "notes": out.notes},
    }


def _falsified(data: Mapping, defaults: Mapping | None) -> dict:
    """``data`` with ``false`` for a None whose default is not None (a table written out
    leaves a None default as a commented line; a setting turned off must be written)."""
    defaults = defaults or {}
    out = {}
    for key, value in data.items():
        default = defaults.get(key)
        if value is None and default is not None:
            value = False
        elif isinstance(value, Mapping) and isinstance(default, Mapping):
            value = _falsified(value, default)
        out[key] = value
    return out


def _record_header(title: str, lines: list[str]) -> list[str]:
    return [f"# {title}"] + textwrap.wrap(" ".join(lines), _LINE_WIDTH - 2, initial_indent="# ",
                                          subsequent_indent="# ") + [""]


def _made(manifest: Mapping) -> str:
    """When and with which hxex a bundle was written (for a header)."""
    hxex_version = (manifest.get("versions") or {}).get("hxex", "unknown")
    return f"created {manifest.get('created') or 'at an unknown time'} by hxex {hxex_version}"


def run_settings_text(manifest: Mapping) -> str:
    """``run_settings_used.toml`` of the bundle whose manifest (or :class:`hxex.report.Settings`
    as a mapping) is ``manifest``: every setting of the run written out, defaults included, as
    a project file. The ``[input]`` and ``[output]`` tables are those recorded with the bundle
    (``configs.project_file``); a bundle written from Python, or before they were recorded, has
    the run settings only."""
    configs = manifest.get("configs") or {}
    record = configs.get("project_file") or {}
    run = configs.get("run")
    name = record.get("project") or manifest.get("project") or ""
    lines = _record_header(f"{RUN_SETTINGS_FILE}: the settings of the run that wrote these "
                           "results", [
        f"Results {_made(manifest)}.",
        "Every setting is written out: one you did not set shows its default. hxex writes",
        "this file with the results and replaces it with them; it is a record, not read back",
        "(the bundle's manifest.json holds the same settings for hxex). Copied elsewhere, it",
        "is a project file that repeats the run. Paths are recorded as [input] record_paths",
        "says (\"name\": file names only).",
    ])
    if name:
        lines.append(f"project = {_toml_value(name)}")
    if record.get("threads") is not None:
        lines.append(f"threads = {record['threads']}")
    if record.get("input"):
        lines += [""] + _toml_table("input", record["input"], cls=InputSettings)
    elif record:
        lines += ["", "# [input]: none (the run took its spectra from Python)"]
    else:
        lines += ["", "# [input] and [output] were not recorded (results written from Python,",
                  "# or before hxex recorded them); manifest.json says how the spectra were "
                  "read",
                  "# (configs.spectra_source)."]
    if record.get("output"):
        lines += [""] + _toml_table("output", record["output"], cls=OutputSettings)
    if run is None:
        lines += ["", "# [run]: the results do not record their run settings."]
    else:
        defaults = RunConfig().to_dict()
        data = {k: (False if v is None and k in _OPTIONAL_SECTIONS else v)
                for k, v in run.items()}
        table = _toml_table("run", _falsified(data, defaults), cls=RunConfig)
        intro = textwrap.wrap(  # the template's says how to turn the stages on
            "Run settings (hxex.run.RunConfig); times are in s. A stage written \"name = "
            "false\" (bootstrap, population_test, controls, quality) was off.", _LINE_WIDTH,
            initial_indent="# ", subsequent_indent="# ")
        lines += [""] + intro + table[len(_section_comment("run")):]
    return "\n".join(lines) + "\n"


def _analysis_record(config, mode: str) -> dict[str, Any]:
    """The ``[analysis]`` table of ``config`` with every setting written out."""
    data = config.to_dict()
    data.pop("exclude")  # a project file gives the rules in [review] (analysis_settings_text)
    annotations = data.pop("annotations")
    data["regions"], data["sites"] = annotations["regions"], annotations["sites"]
    fasta = data.get("fasta")
    if isinstance(fasta, str):
        data["fasta"] = _recorded(fasta, mode)
    elif isinstance(fasta, Mapping):
        data["fasta"] = {k: ({s: _recorded(p, mode) for s, p in v.items()}
                             if isinstance(v, Mapping) else _recorded(v, mode))
                         for k, v in fasta.items()}
    data["out"] = _recorded(data.get("out"), mode)
    if data.get("runs"):
        data["runs"] = {k: _recorded(v, mode) for k, v in data["runs"].items()}
    return data


def analysis_settings_text(config, results: Mapping[str, Mapping], *,
                           record_paths: str = "path", review=None) -> str:
    """``analysis_settings_used.toml``: every setting of the analysis ``config``
    (:class:`hxex.analysis.AnalysisConfig`) written out, defaults included, as the
    ``[analysis]`` table of a project file. ``results`` names the results it read (run name,
    empty for a single bundle -> their manifest, or :class:`hxex.report.Settings` as a
    mapping), for the header. ``record_paths`` records its paths as ``[input] record_paths``
    does. ``review`` (:class:`hxex.review_table.ReviewSettings`) adds the ``[review]`` table the
    analysis applied."""
    from .analysis import AnalysisConfig

    read = [f"{'the results' if not run else f'run {run}'} ("
            + (_made(m) if m.get("configs") else "its settings not recorded") + ")"
            for run, m in results.items()]
    copies = (f"{RUN_SETTINGS_FILE}" if list(results) == [""] else
              "run_settings_used_RUN.toml, one per run")
    lines = _record_header(f"{ANALYSIS_SETTINGS_FILE}: the settings of this analysis", [
        f"It read {', '.join(read) or 'no results'}.",
        "Every setting is written out: one you did not set shows its default. The run",
        f"settings of the results it read are beside this file ({copies}). hxex writes this",
        "file with the analysis and replaces it with it; it is a record, not read back. To",
        "repeat the analysis, put the [analysis] table in a project file, or run",
        f"hxex analyse {ANALYSIS_SETTINGS_FILE} --bundle BUNDLE. Paths are recorded as",
        "[input] record_paths says (\"name\": file names only).",
    ])
    defaults = _analysis_record(AnalysisConfig(), "path")
    data = _falsified(_analysis_record(config, record_paths), defaults)
    lines += _toml_table("analysis", data, cls=AnalysisConfig)
    rules = [*(review.exclude if review else ()), *config.exclude]
    if rules or (review and review.table):
        table = {"table": _recorded(review.table if review else None, record_paths),
                 "exclude": [r.to_dict() for r in rules]}
        lines += ["", "# The review decisions the analysis applied (its rejections)."]
        lines += _toml_table("review", {k: v for k, v in table.items() if v})
    return "\n".join(lines) + "\n"


def run_settings_name(run: str = "") -> str:
    """The name of a run's settings file in an analysis folder: ``run_settings_used.toml`` for
    the one bundle read, ``run_settings_used_RUN.toml`` for each run of several."""
    if not run:
        return RUN_SETTINGS_FILE
    return f"run_settings_used_{re.sub(r'[^A-Za-z0-9._-]+', '_', run).strip('_') or 'run'}.toml"


def settings_records(results, config, record_paths: str = "path",
                     review=None) -> dict[str, str]:
    """The settings files of an analysis (file name -> text): ``analysis_settings_used.toml``
    for ``config`` and the run settings of each results read (``results``: a
    :class:`hxex.report.Results`, or run name -> Results), from what their bundles recorded."""
    import dataclasses

    def manifest(r) -> dict:
        settings = getattr(r, "settings", None)
        if dataclasses.is_dataclass(settings):
            return dataclasses.asdict(settings)
        return dict(settings) if isinstance(settings, Mapping) else {}

    runs = results if isinstance(results, Mapping) else {"": results}
    manifests = {name: manifest(r) for name, r in runs.items()}
    out = {ANALYSIS_SETTINGS_FILE: analysis_settings_text(config, manifests,
                                                          record_paths=record_paths,
                                                          review=review)}
    for name, m in manifests.items():
        if m.get("configs"):
            out[run_settings_name(name)] = run_settings_text(m)
    return out


def describe(project: Project) -> dict[str, Any]:
    """A short summary of a project's settings (for ``hxex check``)."""
    run = project.run
    inp = project.input
    return {
        "project": project.project,
        "format": "(none: analysis only)" if inp is None else inp.format + (
            "" if inp.spectra is None else f" (stored in {inp.spectra})"),
        "K": f"{run.fit.k_min} to {run.fit.k_max}",
        "device": run.fit.backend.device,
        "bootstrap": ("off: no standard errors or intervals (a [run.bootstrap] table turns "
                      "it on)" if run.bootstrap is None
                      else f"{run.bootstrap.n_boot} resamples"),
        "population test": ("off: K by the fit's criterion (a [run.population_test] table "
                            "turns it on)" if run.population_test is None
                            else f"on, {run.population_test.null} null"
                            + (f", {run.population_test.simulations} simulations"
                               if run.population_test.null == "simulated" else "")),
        "controls": ("off: no back-exchange correction, means in deuterons measured (a "
                     "[run.controls] table turns it on)" if run.controls is None
                     else "on" + _promotions(run.controls)),
        "quality flags": "off (quality = false)" if run.quality is None else "on",
        "times": ("seconds" if inp is None or inp.time_scale == "seconds"
                  else "order: the times only order the points (uptake drawn by index; the "
                       "kinetics and protection analyses refuse)"),
        "bundle": str(project.output.bundle) if project.output.bundle else "(none)",
        **_exclusion_lines("excluded", () if inp is None else inp.exclude,
                           "left out before fitting (stored, not fitted)"),
        **_exclusion_lines("excluded later", () if project.review is None
                           else project.review.exclude,
                           "treated as rejected in the tables, figures and analysis"),
        **_review_table_line(project.review),
    }


def _review_table_line(review) -> dict[str, str]:
    """The summary line of the review table (none without one); a table that cannot be read is
    a problem of the project (ProjectError)."""
    if review is None or not review.table:
        return {}
    from .review_table import read_review_table

    try:
        sheet = read_review_table(review.table)
    except ValueError as exc:
        raise ProjectError(f"[review] table: {exc}") from None
    return {"review table": f"{review.table}: {sheet.describe()}"}


def _exclusion_lines(key: str, rules, what: str) -> dict[str, str]:
    """The summary lines of exclusion rules (none without rules)."""
    if not rules:
        return {}
    return {key: f"{len(rules)} rule(s), {what}: "
                 + "; ".join(r.describe() for r in rules)}


def _promotions(controls) -> str:
    """The time points a control config promotes to controls (``undeut_at``, ``fulldeut_at``)
    and the marked files it sets aside (``*_marked = false``), for a summary line."""
    from .io.times import format_time

    parts = [f"{name} = {format_time(getattr(controls, name))} (the spectra at that time "
             f"serve as the {label} control)"
             for name, label in (("undeut_at", "undeuterated"), ("fulldeut_at", "fully deuterated"))
             if getattr(controls, name) is not None]
    parts += [f"{name} = false (the files marked {label} do not)"
              for name, label in (("undeut_marked", "undeut"), ("fulldeut_marked", "fulldeut"))
              if not getattr(controls, name)]
    return "; " + "; ".join(parts) if parts else ""
