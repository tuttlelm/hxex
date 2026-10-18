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
    bundle = "results"          # the results bundle

    [analysis]                  # the post-fitting analysis (hxex analyse; hxex.analysis)
    reference = "WT"

A key left out keeps its default. TOML has no null: ``false`` stands for None where a setting
can be None (an optional stage, a scan condition a run removes, an unset limit) and is not a
switch; a switch (``drop_no_signal``, ``overview``) set to ``false`` is off. This holds in every
``[run]`` table and sub-table, by the type of each setting. The control times of
``[run.controls]`` (``undeut_time``, ``fd_time``) left out are those the readers give the
controls (``[input] undeut_time`` and ``fulldeut_time`` of a folder format; ``[run]
undeut_time`` for the undeuterated runs of a scan format); ``false`` says the data has no such
control. Relative paths are taken relative to the project file's directory.
:func:`load_project` reads and checks the file, :func:`read_input` reads the spectra it names,
and :func:`template` writes a project file with every default.
"""

from __future__ import annotations

import logging
import math
import textwrap
import tomllib
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

from .hooks import CancelHook, ProgressHook
from .io.mzml import MzmlConfig, MzmlRun, MzmlTarget
from .io.times import DEFAULT_FULLDEUT_TIME, DEFAULT_UNDEUT_TIME
from .run import RunConfig
from .spectra import SpectrumSet

log = logging.getLogger(__name__)

FORMATS = ("specexport", "hxexpress", "pooled", "mzml", "thermo")
FOLDER_FORMATS = ("specexport", "hxexpress", "pooled")
SCAN_FORMATS = ("mzml", "thermo")
# run settings that are None unless the file gives them, and their classes
_OPTIONAL_SECTIONS = ("bootstrap", "controls", "plots", "quality")


class ProjectError(ValueError):
    """A project file that cannot be used."""


def _check_keys(data: Mapping, allowed, where: str) -> None:
    unknown = set(data) - set(allowed)
    if unknown:
        raise ProjectError(f"unknown key(s) in {where}: {', '.join(sorted(unknown))}")


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


def _control_times(run: RunConfig, given, inp: InputSettings) -> RunConfig:
    """``run`` with the control times its ``[run.controls]`` (``given``, the table as written)
    leaves out taken from where the readers put the controls: the undeuterated at the input's
    ``undeut_time`` (folder formats) or at ``[run] undeut_time`` (scan formats, whose runs carry
    their own times); the fully deuterated at the input's ``fulldeut_time`` (folder formats
    only: a scan format's fully deuterated runs have whatever time their ``[[input.runs]]``
    give, so ``fd_time`` is set by hand). A time given as ``false`` (no control of that kind)
    stays None."""
    if run.controls is None:
        return run
    given = given if isinstance(given, Mapping) else {}
    folder = inp.format in FOLDER_FORMATS
    times = {}
    if "undeut_time" not in given:
        times["undeut_time"] = inp.undeut_time if folder else run.undeut_time
    if "fd_time" not in given and folder:
        times["fd_time"] = inp.fulldeut_time
    return replace(run, controls=replace(run.controls, **times)) if times else run


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
    - ``limit_charge``: fragment charges stay at or below the parent's.
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
    - Folder formats: ``data_dir``; ``samples`` and ``peptide_ranges`` (only these);
      ``sample`` (the name given to a pooled folder's entries); ``undeut_time`` and
      ``fulldeut_time`` (s), the times recorded for the undeuterated and fully deuterated
      control files (the controls' default times); ``peptides``, a table of the files instead of
      finding them by name.
    - Scan formats: ``peptides`` (the peptide table, CSV), ``runs`` (one :class:`MzmlRun` per
      file), ``parents`` (read each peptide itself), ``target`` (how the peptides' scans are
      chosen), ``fragments`` (fragment targets), ``scans`` (how scans become spectra,
      :class:`MzmlConfig`), ``data_root`` (relative run paths start here) and ``thermo_dir``
      (Thermo's DLLs).
    - Every format: ``record_paths`` (``'path'``, ``'relative'`` or ``'name'``: how file paths
      are recorded), ``retention_times`` (a table of retention times, minutes) and ``spectra``
      (take the raw spectra stored by an earlier run, a bundle or a spectra file, instead of
      reading the input).
    """

    format: str
    # folder formats
    data_dir: Path | None = None
    samples: tuple[str, ...] | None = None
    peptide_ranges: tuple[str, ...] | None = None
    sample: str = ""
    undeut_time: float = DEFAULT_UNDEUT_TIME
    fulldeut_time: float = DEFAULT_FULLDEUT_TIME
    record_paths: str = "path"
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


@dataclass(frozen=True)
class OutputSettings:
    """Where results go (the ``[output]`` table): the bundle, the tables (a directory, or True
    for the bundle's name plus ``_tables``) and the figures (likewise, ``_figures``). The
    bundle and these two folders (and their earlier generations, ``name.N``) are hxex's: a
    later run replaces them (``overwrite``) or moves them to the next generation first
    (``keep_previous``); see :func:`hxex.run.run`."""

    bundle: Path | None = None
    overwrite: bool = False
    keep_previous: bool = False
    keep_spectra: bool = True
    tables: Path | bool = True
    figures: Path | bool = False
    notes: str = ""

    def tables_dir(self) -> Path | None:
        return _beside(self.bundle, self.tables, "_tables")

    def figures_dir(self) -> Path | None:
        return _beside(self.bundle, self.figures, "_figures")


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


def _window(value, where: str, unit: str = "[start, end] in seconds"):
    value = _none(value)
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ProjectError(f"{where} must be {unit}")
    return (float(value[0]), float(value[1]))


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
    bundle), but the format and the folder formats' control times are those of the stored
    spectra's read record (logged when the file says otherwise); with ``spectra`` alone (and
    at most ``retention_times``) they come from the record only."""
    names = {f.name for f in fields(InputSettings)}
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
    record = {
        "format": reader if reader in FORMATS else "stored",
        "undeut_time": float((source or {}).get("undeut_time", DEFAULT_UNDEUT_TIME)),
        "fulldeut_time": float((source or {}).get("fulldeut_time", DEFAULT_FULLDEUT_TIME)),
    }
    if not (set(data) - {"retention_times"}):
        return InputSettings(
            retention_times=_path(base, data.get("retention_times"), "input.retention_times"),
            spectra=stored, **record)
    inp = _read_settings(data, base)
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
        return InputSettings(
            format=fmt, data_dir=_path(base, data["data_dir"], "input.data_dir"),
            samples=_strings(data.get("samples"), "input.samples"),
            peptide_ranges=_strings(data.get("peptide_ranges"), "input.peptide_ranges"),
            sample=str(data.get("sample", "")),
            undeut_time=float(data.get("undeut_time", DEFAULT_UNDEUT_TIME)),
            fulldeut_time=float(data.get("fulldeut_time", DEFAULT_FULLDEUT_TIME)),
            record_paths=str(data.get("record_paths", "path")),
            peptides=_path(base, data.get("peptides"), "input.peptides"),
            retention_times=_path(base, data.get("retention_times"), "input.retention_times"),
        )
    extra = set(data) & {"data_dir", "samples", "peptide_ranges", "sample", "undeut_time",
                         "fulldeut_time"}
    if extra:
        raise ProjectError(f"input key(s) {', '.join(sorted(extra))} are not used by {fmt} "
                           "(the runs give the samples and times)")
    if "peptides" not in data:
        raise ProjectError("input.peptides (a peptide table, CSV) is needed for mzml and thermo")
    runs = data.get("runs") or []
    if not isinstance(runs, list) or not runs:
        raise ProjectError("give at least one [[input.runs]] (path and sample)")
    run_names = {f.name for f in fields(MzmlRun)}
    parsed_runs = []
    for i, r in enumerate(runs, start=1):
        where = f"input.runs #{i}" + (f" ({r['path']})" if isinstance(r.get("path"), str)
                                      else "")
        _check_keys(r, run_names, where)
        time = r.get("time")
        if time is not None and (isinstance(time, bool) or not isinstance(time, (int, float))
                                 or not math.isfinite(time)):
            raise ProjectError(f"{where}: time must be a number of seconds (the labelling time "
                               "of the whole file) or left out (each scan has its own time); "
                               f"got {time!r}")
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
    blocks = data.get("fragments") or []
    fragments = tuple(_fragment_settings(b) for b in
                      ([blocks] if isinstance(blocks, Mapping) else blocks))
    parents = bool(data.get("parents", True))
    if not parents and not fragments:
        raise ProjectError("parents = false needs [input.fragments]: there would be no target")
    scans = data.get("scans", {})
    if not isinstance(scans, Mapping):
        raise ProjectError(f"[input] scans = {_toml_value(scans)}: expected a table")
    scans = _false_to_none(scans, MzmlConfig)
    _check_types(scans, MzmlConfig, "input.scans")
    try:
        scans = MzmlConfig.from_dict(_merge(MzmlConfig().to_dict(), scans, "input.scans"))
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"[input.scans]: {exc}") from exc
    return InputSettings(
        format=fmt, peptides=_path(base, data["peptides"], "input.peptides"),
        runs=tuple(parsed_runs), parents=parents, target=target, fragments=fragments,
        scans=scans, record_paths=str(data.get("record_paths", "path")),
        data_root=_path(base, data.get("data_root", "."), "input.data_root"),
        thermo_dir=_path(base, data.get("thermo_dir"), "input.thermo_dir"),
        retention_times=_path(base, data.get("retention_times"), "input.retention_times"),
    )


def _output_from(data: Mapping, base: Path) -> OutputSettings:
    _check_keys(data, {f.name for f in fields(OutputSettings)}, "[output]")

    def where(key, default):
        value = data.get(key, default)
        return value if isinstance(value, bool) else _path(base, value, f"output.{key}")

    return OutputSettings(
        bundle=_path(base, data.get("bundle"), "output.bundle"),
        overwrite=bool(data.get("overwrite", False)),
        keep_previous=bool(data.get("keep_previous", False)),
        keep_spectra=bool(data.get("keep_spectra", True)),
        tables=where("tables", True), figures=where("figures", False),
        notes=str(data.get("notes", "")),
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
    _check_keys(data, {"project", "threads", "input", "run", "output", "analysis"},
                "the project file")
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
        run = _control_times(run, run_table.get("controls"), inp)
    output = _output_from(data.get("output", {}), base)
    if output.figures is not False and run.plots is None:  # figures asked for: default plots
        from .report.figures import PlotConfig

        run = replace(run, plots=PlotConfig())
    threads = data.get("threads")
    if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int)
                                or threads < 1):
        raise ProjectError("threads must be a whole number of at least 1")
    analysis = _analysis_from(data["analysis"], base) if "analysis" in data else None
    return Project(name, inp, run, output, path, threads, analysis)


ANALYSIS_KEYS = ("reference", "samples", "times", "exclude_times", "ks", "fasta",
                 "coverage_times", "redundancy", "weight_exponent", "filters", "choice", "alpha",
                 "regions", "sites",
                 "exclude_flagged", "format", "out", "runs", "overlay",
                 "overlay_two_only", "peptide_match", "modified_peptides", "controls", "merge",
                 "fragments", "protection", "labelling", "protection_residues", "protection_r1",
                 "protection_max_se")


def _analysis_from(data: Mapping, base: Path):
    """The :class:`hxex.analysis.AnalysisConfig` of an ``[analysis]`` table (relative paths from
    ``base``)."""
    from .analysis import AnalysisConfig

    _check_keys(data, ANALYSIS_KEYS, "[analysis]")
    values = _false_to_none({k: v for k, v in data.items() if k not in ("regions", "sites")},
                            AnalysisConfig)
    if values.get("labelling") is False:  # typed Any: the helper cannot tell
        values["labelling"] = None
    _check_types(values, AnalysisConfig, "analysis")
    fasta = values.get("fasta")
    if isinstance(fasta, str):
        values["fasta"] = str(_path(base, fasta, "analysis.fasta"))
    elif isinstance(fasta, Mapping):
        values["fasta"] = {str(k): str(_path(base, v, f"analysis.fasta.{k}"))
                           for k, v in fasta.items()}
    if "out" in values:
        values["out"] = str(_path(base, values["out"], "analysis.out"))
    if values.get("runs") is not None:
        if not isinstance(values["runs"], Mapping):
            raise ProjectError("[analysis.runs] must be a table of run name = bundle")
        values["runs"] = {str(k): str(_path(base, v, f"analysis.runs.{k}"))
                          for k, v in values["runs"].items()}
    values["annotations"] = {"regions": list(data.get("regions", [])),
                             "sites": list(data.get("sites", []))}
    try:
        if isinstance(values.get("filters"), Mapping):
            from .analysis import SpeciesFilters

            values["filters"] = {**SpeciesFilters().__dict__, **values["filters"]}
        if "choice" in values:
            from .analysis import ChoiceConfig

            if not isinstance(values["choice"], Mapping):
                raise TypeError("choice must be a table")
            values["choice"] = {**ChoiceConfig().to_dict(), **values["choice"]}
        return AnalysisConfig.from_dict(values)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ProjectError(f"[analysis]: {exc}") from exc


def load_project(path: str | Path, *, analysis_only: bool = False,
                 spectra: str | Path | None = None) -> Project:
    """Read and check a project file (``analysis_only``: one that may have only an
    ``[analysis]`` table, for :func:`hxex.report.write_analysis`; ``spectra``: stored raw
    spectra in place of the input, :func:`project_from`)."""
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProjectError(f"cannot read {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ProjectError(f"{path}: not valid TOML ({exc})") from exc
    return project_from(data, path.resolve().parent, path, analysis_only=analysis_only,
                        spectra=spectra)


# --- reading the input --------------------------------------------------------------------------

def scan_targets(project: Project) -> list[MzmlTarget]:
    """The targets of an mzML or Thermo project: each peptide of the table (if ``parents``),
    and its fragments (one set per ``[input.fragments]`` table). A peptide's ``rt_start`` and
    ``rt_end`` columns set the retention window of its targets, in place of ``rt_window``."""
    from .fragments import fragment_targets
    from .io.entries import read_peptide_table
    from .spectra import entry_rt_window

    inp = project.input
    targets = []
    for entry in read_peptide_table(inp.peptides):
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
                peptide_ranges=inp.peptide_ranges, sample=inp.sample)


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


_READ_STAGE = ("format", "data_dir", "samples", "peptide_ranges", "sample", "undeut_time",
               "fulldeut_time", "peptides", "runs", "parents", "target", "fragments", "scans",
               "data_root", "thermo_dir")


def read_input(
    project: Project,
    *,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> SpectrumSet:
    """The raw spectra the project's ``[input]`` names (recorded, so the read can be
    repeated), with the retention times of ``retention_times`` when it names a table. With
    ``spectra`` they are the spectra stored by an earlier run (:func:`load_stored_spectra`):
    the settings of the read itself (files, targets, scans) do not apply then."""
    inp = project.input
    if inp.spectra is not None:
        default = InputSettings(inp.format)
        base = project.path.resolve().parent if project.path is not None else Path.cwd()
        unused = [k for k in _READ_STAGE if k not in ("format", "undeut_time", "fulldeut_time")
                  and getattr(inp, k) != getattr(default, k)
                  and not (k == "data_root" and inp.data_root == base)]
        log.info("taking the raw spectra stored in %s%s", inp.spectra,
                 f"; not used: input {', '.join(unused)}" if unused else "")
        spectra = load_stored_spectra(inp.spectra)
    else:
        spectra = _read_input(project, progress=progress, cancel=cancel)
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
            fulldeut_time=inp.fulldeut_time, undeut_time=inp.undeut_time, samples=inp.samples,
            peptide_ranges=inp.peptide_ranges, record_paths=inp.record_paths,
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
    "run.envelope_fallback_time": "s: spectra at this time are the envelope references when a "
                                  "series' own are missing or poor (false: none)",
    "run.undeut_time": "s: time of the undeuterated spectra, the noise references (scan "
                       "formats: also the controls' undeut_time unless set)",
    "run.n_max_factor": "n (exchangeable sites) may be up to this times the amide count",
    "run.drop_no_signal": "drop spectra the picker found no signal in (empty ones always are)",
    "run.memory_budget_bytes": "memory of one chunk of the fit, bytes (false: automatic)",
    "run.max_chunk": "most measurements in one chunk (false: no cap)",
    "run.picker.resolution_ppm": "half-width (ppm) of the window a peak is picked in around "
                                 "its predicted m/z; not a resolving power",
    "run.picker.zero_filling": "bins picked after the last one above the noise threshold",
    "run.picker.noise_threshold": "absolute intensity below which a point is noise (false: "
                                  "noise_percent of the tallest point instead)",
    "run.picker.noise_percent": "noise threshold as percent of the spectrum's tallest point",
    "run.picker.env_threshold": "fraction of the tallest bin at which the envelope width is "
                                "measured",
    "run.picker.nterm_subtract": "N-terminal residues without an exchanging amide (the amide "
                                 "count bounds the bins)",
    "run.picker.shoulder_min_points": "points in a window from which a maximum at the window's "
                                      "edge counts as a neighbour's shoulder (bin 0)",
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
    "run.fit.k_min": "fewest populations tried",
    "run.fit.k_max": "most populations tried (one K: no model selection)",
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
                      "but the skipped fits are missing from fixed-K analyses",
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
    "run.bootstrap.n_boot": "resamples of every measurement",
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
    "run.bootstrap.ks": '"selected" (each measurement at its own K), "all" fitted K, or a '
                        "list of K (fixed-K bootstraps for the analysis)",
    "run.controls.m0_max": "largest mean (D) of a good undeuterated control",
    "run.controls.retention_min": "smallest deuterium retention of a good fully deuterated "
                                  "control, as a fraction (false: half of d2o_fraction; none "
                                  'with fd_fallback "kint"); not a retention time',
    "run.controls.retention_max": "largest deuterium retention of a good fully deuterated "
                                  "control, as a fraction",
    "run.controls.d2o_fraction": "D atom fraction of the labelling buffer (0 to 1); "
                                 "[analysis.controls] may change it for the analysis",
    "run.controls.fd_fallback": 'retention of a series without a good fully deuterated control: '
                                '"d2o" (d2o_fraction), "median" (of the run\'s accepted '
                                'controls) or "kint" (predicted back-exchange; needs quench); '
                                'false: "kint" with quench, else "d2o"',
    "run.controls.atol": "s: a spectrum this close to a control time is that control",
    "run.controls.kint_tolerance": "kint: a measured control this far (D per amide) from the "
                                   "prediction is rejected",
    "run.controls.kint_min_series": "kint: fewest usable controls to fit the prediction to",
    "run.quality.poor_fit_factor": "poor_fit: misfit this many times the series' median",
    "run.quality.min_deviance": "poor_fit: and a reduced deviance above this",
    "run.quality.min_series_fits": "poor_fit: fewest fitted spectra in a series to judge",
    "run.quality.min_intensity": "low_intensity: total intensity below this share of the other "
                                 "replicates' mean",
    "run.quality.max_mz_offset_ppm": "mz_shift: a series' m/z offset (ppm) beyond this (false: "
                                     "screen off)",
    "run.plots.series": '"flagged" (series with a flag, a rejection or a control fallback), '
                        '"all" or a list of data_id',
    "run.plots.kinds": "pages of a series' figure: spectra, populations, uptake",
    "run.plots.show": "layers of the spectra page: raw, picked, fit",
    "run.plots.x": 'uptake axis: "time" or "index"',
    "run.plots.scale": 'uptake axis: "linear" or "log"',
    "run.plots.corrected": "draw back-exchange corrected values (unset: when there are "
                           "controls)",
    "run.plots.format": '"pdf" (one file per series, a page per kind), "png" or "svg" (one '
                        "file per kind)",
    "run.plots.dpi": "resolution of PNG files and of raster images in PDF or SVG",
    "run.plots.overview": "also the run's overview (K selection, controls, quality flags)",
    "run.plots.residual": "a residual panel under every spectrum",
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
           "bootstrap, the controls and the plots are stages that their table turns on: "
           "uncomment the commented-out run.bootstrap, run.controls or run.plots table below to "
           "use it; a stage's \"name = false\" turns it off.",
    "run.picker": "Peak picking: the spectra become sticks at the predicted isotope positions "
                  "(hxex.picking.PickerConfig; docs/PEAK_PICKING.md).",
    "run.picker.amide_rules": "Per fragment ion type, whether it keeps the amide N-H at its "
                              'cleavage site, e.g. "y-NH3" = false. Empty: the ion type\'s '
                              "composition decides (c, x, y keep it; a, b, z do not).",
    "run.fit": "The fit of 1 to k_max populations to every spectrum (hxex.fit.FitConfig).",
    "run.fit.noise": "Noise model and level of the fit (hxex.noise.NoiseConfig). The level "
                     "sets the loss scale and the model selection.",
    "run.fit.backend": "Where the fit runs (hxex.model.Backend); nothing is set globally.",
    "run.bootstrap": "Standard errors and intervals from resampled refits of every "
                     "measurement (hxex.bootstrap.BootstrapConfig). Slow: most of a run's time.",
    "run.controls": "Back-exchange correction from the undeuterated and fully deuterated "
                    "controls, fitted at K=1 (hxex.controls.ControlConfig). Deuterium "
                    "retention here is the fraction of deuterium kept, not a retention time.",
    "run.quality": "Quality flags on every spectrum (hxex.quality.QualityConfig): poor_fit, "
                   "low_intensity, noise_outlier, mz_shift, poor_envelope. Nothing is dropped; "
                   "the review layer of the bundle takes the decisions.",
    "run.plots": "Figures written by hxex run and hxex report (hxex.report.PlotConfig).",
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
# peptide_ranges = ["0001-0010"] # only these peptides
undeut_time = 0.0                # the time given to undeuterated controls (s)
fulldeut_time = 1000000.0        # the time given to fully deuterated controls (s)
record_paths = "path"            # or "name": record file names only
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
# peptide_ranges = ["0001-0010"]
undeut_time = 0.0
fulldeut_time = 1000000.0
record_paths = "path\"""",
    "pooled": """[input]
format = "pooled"
# spectra = "old/results"        # take the raw spectra stored in a bundle (or a spectra
#                                # file) instead of reading the files below
data_dir = "pooled"              # a SpecExport folder with peptides of several proteins
sample = "pool"                  # the sample name given to its entries
# peptide_ranges = ["0001-0010"]
undeut_time = 0.0
fulldeut_time = 1000000.0
record_paths = "path\"""",
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

# one [[input.runs]] per file; "time" is the labelling time of the whole file (s), or leave it
# out when each scan has its own time (continuous labelling). Two clocks: the scan times (the
# acquisition clock) and the exchange clock, scan time + "time_offset" (s; when exchange began
# relative to acquisition). "usable_from" (s, acquisition clock) leaves out this file's scans
# before it (e.g. spurious early points): it does not move the time axis, time_offset does.
[[input.runs]]
path = "UN_01.{ext}"
sample = "WT"
rep = 1
time = 0.0

[[input.runs]]
path = "D_60s_01.{ext}"
sample = "WT"
rep = 1
time = 60.0"""


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
        "[output]",
        "# The bundle and the folders named after it (results_tables, results_figures and",
        "# the generations results.1, results.1_tables, ...) belong to hxex: a later run",
        "# replaces or renames them. Keep your own files elsewhere (files of yours inside",
        "# them are kept, but one named like an hxex file stops the run).",
        'bundle = "results"               # the results bundle (a directory, or .zip)',
        "overwrite = false                # replace an existing bundle (its review is carried",
        "#                                 # over) and its tables and figures",
        "keep_previous = false            # move the existing bundle, tables and figures to",
        "#                                 # the next generation (results.1, ...) first",
        "keep_spectra = true              # store the raw spectra in the bundle",
        "tables = true                    # CSV tables beside the bundle, or a directory",
        "figures = false                  # figures beside the bundle, or a directory",
        'notes = ""',
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
        '# coverage_times = "undeuterated" # or "all", or a list of times',
        '# redundancy = "peptides"         # or "spectra"',
        "# alpha = 0.05                    # significance of the comparisons",
        '# overlay = ["low", "high"]       # by-residue comparisons: K=2 low/high and/or "avg"',
        "# overlay_two_only = true         # K=2 values only where two populations are chosen",
        '# peptide_match = "range"         # compare peptides over the same residues ("range",',
        '#                                 # mutated or not) or identical sequences ("sequence")',
        '# modified_peptides = "separate"  # modified forms as their own states ("WT pS59") or',
        '#                                 # averaged with the sample ("combined")',
        '# format = "pdf"                  # or "png", "svg"',
        "# protection = true               # opt in to the peptides' protection factors",
        "#                                 # (hxex.analysis.protection); needs the labelling table",
        "# protection_residues = true      # also one factor per residue (opt-in)",
        "# protection_r1 = 0.1             # smoothing of the residue-level fit",
        "# protection_max_se = 5.0         # residues with a larger error left out of figures",
        "# [analysis.controls]             # control settings to change for the analysis (the",
        "# d2o_fraction = 0.85             # controls are recomputed from the run's K=1 fits)",
        "# [analysis.filters]              # which K=2 descriptions are kept (false: none)",
        "# min_population = 0.05",
        "# max_fD_se = 0.25",
        "# max_fD = 1.1",
        "# k1_between_ranks = true",
        "# [analysis.choice]               # the choice between one and two populations",
        "# confidence = 0.5",
        "# max_fD_se = 0.25",
        "# min_population = 0.05",
        "# [[analysis.regions]]",
        '# name = "N-terminal region"',
        "# start = 1",
        "# end = 63",
        "# [[analysis.sites]]",
        "# residue = 19",
        '# label = "S19"',
        "# [analysis.runs]                 # compare separate runs: samples become run:sample",
        '# A = "runA.zip"                  # (reference = "A:WT"); default: this project\'s bundle',
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
        "# kint = true                     # opt in: intrinsic rates, protection factors and",
        "#                                 # theoretical curves (hxex.kint); needs exchange",
        '# [analysis.fragments.labels]     # short condition names in the figures',
        '# gentle = "G"',
        "# [analysis.fragments.exchange]   # the kinetics' solvent, for every sample (or one",
        "# ph_read = 3.0                   # table per sample: [analysis.fragments.exchange.S])",
        "# temperature = 20.0              # degrees Celsius",
        "# d2o_fraction = 0.05             # D atom fraction of the kinetics' solvent (0 to 1;",
        "#                                 # the labelling buffer's is [run.controls])",
        '# reference = "PDLA"              # intrinsic-rate set: "PDLA" or "3Ala"',
        "# initial_fraction = 0.9          # D fraction when the exchange began (curves)",
        "# [analysis.labelling]            # the labelling solvent, for every sample (or one",
        "# ph_read = 7.0                   # table per sample: [analysis.labelling.WT])",
        "# temperature = 20.0              # degrees Celsius",
        "# d2o_fraction = 0.9              # D atom fraction of the labelling buffer (as",
        "#                                 # [run.controls] d2o_fraction)",
        '# reference = "PDLA"              # intrinsic-rate set: "PDLA" or "3Ala"',
    ]
    scalars = {k: v for k, v in run.items()
               if not isinstance(v, Mapping) and k not in _OPTIONAL_SECTIONS}
    lines += [""] + _toml_table("run", scalars, cls=RunConfig)
    if fmt in SCAN_FORMATS:
        control_times = {
            "undeut_time": ("0.0", "s: time of the undeuterated controls (default: [run] "
                                   "undeut_time; false: the data has none)"),
            "fd_time": ("1000000.0", "s: time given to the fully deuterated runs in "
                                     "[[input.runs]] (false: the data has none)"),
            "quench": ('{ ph_read = 2.5, temperature = 0.0, d2o_fraction = 0.0 }',
                       "pH read, degrees Celsius and D fraction of the quench and LC, for "
                       'fd_fallback "kint" (false: none)'),
        }
    else:
        control_times = {
            "undeut_time": ("0.0", "s: time of the undeuterated controls (default: [input] "
                                   "undeut_time; false: the data has none)"),
            "fd_time": ("1000000.0", "s: time of the fully deuterated controls (default: "
                                     "[input] fulldeut_time; false: the data has none)"),
            "quench": ('{ ph_read = 2.5, temperature = 0.0, d2o_fraction = 0.0 }',
                       "pH read, degrees Celsius and D fraction of the quench and LC, for "
                       'fd_fallback "kint" (false: none)'),
        }
    for key, value in run.items():
        if key in _OPTIONAL_SECTIONS and value is None:
            lines += [""] + _toml_table(f"run.{key}", _section_defaults(key), comment=True,
                                        cls=_section_class(key),
                                        notes=control_times if key == "controls" else None)
        elif isinstance(value, Mapping):
            lines += [""] + _toml_table(f"run.{key}", value, cls=_section_class(key))
    return "\n".join(lines) + "\n"


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
        "controls": ("off: no back-exchange correction, means in deuterons measured (a "
                     "[run.controls] table turns it on)" if run.controls is None else "on"),
        "quality flags": "off (quality = false)" if run.quality is None else "on",
        "bundle": str(project.output.bundle) if project.output.bundle else "(none)",
    }
