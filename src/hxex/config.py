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
can be None (an optional stage, a scan condition a run removes, an unset limit). Relative paths
are taken relative to the project file's directory. :func:`load_project` reads and checks the
file, :func:`read_input` reads the spectra it names, and :func:`template` writes a project file
with every default.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .hooks import CancelHook, ProgressHook
from .io.mzml import MzmlConfig, MzmlRun, MzmlTarget
from .io.times import DEFAULT_FULLDEUT_TIME, DEFAULT_UNDEUT_TIME
from .run import RunConfig
from .spectra import SpectrumSet

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


def _merge(base: dict, update: Mapping, where: str) -> dict:
    """``base`` (a config's ``to_dict``) with ``update`` laid over it, key by key into tables."""
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            out[key] = _merge(base[key], value, f"{where}.{key}")
        else:
            out[key] = value
    return out


def _section_defaults(name: str) -> dict:
    from .bootstrap import BootstrapConfig
    from .controls import ControlConfig
    from .quality import QualityConfig
    from .report.figures import PlotConfig

    cls = {"bootstrap": BootstrapConfig, "controls": ControlConfig, "plots": PlotConfig,
           "quality": QualityConfig}[name]
    return cls().to_dict()


def run_config_from(data: Mapping) -> RunConfig:
    """The :class:`~hxex.run.RunConfig` of a project file's ``[run]`` table."""
    merged = RunConfig().to_dict()
    for key, value in data.items():
        if key in _OPTIONAL_SECTIONS:
            if value is False:
                merged[key] = None
            elif value is True:
                merged[key] = _section_defaults(key)
            elif isinstance(value, Mapping):
                merged[key] = _merge(_section_defaults(key), value, f"run.{key}")
            else:
                raise ProjectError(f"run.{key} must be a table, true or false")
        elif key in merged and isinstance(value, Mapping) and isinstance(merged[key], Mapping):
            merged[key] = _merge(merged[key], value, f"run.{key}")
        else:
            merged[key] = _none(value)
    try:
        return RunConfig.from_dict(merged)
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"[run]: {exc}") from exc


@dataclass(frozen=True)
class FragmentSettings:
    """Fragment targets made from every peptide of the table (``ions`` in the syntax of
    :mod:`hxex.fragments`, for example ``'c-1:ALL z-dot:ALL'``), with their scan conditions."""

    ions: str
    activation: str | None = None
    precursor_mz: float | None = None
    precursor_charge: int | str | None = "parent"
    filter_contains: str | None = None
    filter_excludes: str | None = None
    rt_window: tuple[float, float] | None = None
    limit_charge: bool = True


# the scan conditions of the peptides themselves (MzmlTarget without its entry)
_TARGET_KEYS = ("ms_level", "activation", "precursor_mz", "precursor_charge", "rt_window",
                "filter_contains", "filter_excludes")


@dataclass(frozen=True)
class InputSettings:
    """What to read (the ``[input]`` table); paths are absolute here."""

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
    fragments: FragmentSettings | None = None
    scans: MzmlConfig = MzmlConfig()  # noqa: RUF009
    data_root: Path | None = None
    thermo_dir: Path | None = None


@dataclass(frozen=True)
class OutputSettings:
    """Where results go (the ``[output]`` table): the bundle, the tables (a directory, or True
    for the bundle's name plus ``_tables``) and the figures (likewise, ``_figures``)."""

    bundle: Path | None = None
    overwrite: bool = False
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
    """A project file, checked: ``project`` (the spectrum-id prefix), ``input``, ``run`` and
    ``output``; ``threads``, the CPU threads of a command that runs it (None: PyTorch's
    default; a setting of the process, so the command line applies it, not the library);
    ``path`` is the file itself."""

    project: str
    input: InputSettings
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


def _window(value, where: str):
    value = _none(value)
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ProjectError(f"{where} must be [start, end] in seconds")
    return (float(value[0]), float(value[1]))


def _input_from(data: Mapping, base: Path) -> InputSettings:
    names = {f.name for f in fields(InputSettings)}
    _check_keys(data, names, "[input]")
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
        _check_keys(r, run_names, f"input.runs #{i}")
        try:
            parsed_runs.append(MzmlRun.from_dict({k: _none(v) if k != "time" else v
                                                  for k, v in r.items()}))
        except (TypeError, ValueError) as exc:
            raise ProjectError(f"input.runs #{i}: {exc}") from exc
    target = dict(data.get("target", {}))
    _check_keys(target, _TARGET_KEYS, "[input.target]")
    target = {k: _none(v) for k, v in target.items()}
    if "rt_window" in target:
        target["rt_window"] = _window(target["rt_window"], "input.target.rt_window")
    fragments = None
    if data.get("fragments"):
        frag = dict(data["fragments"])
        _check_keys(frag, {f.name for f in fields(FragmentSettings)}, "[input.fragments]")
        if "ions" not in frag:
            raise ProjectError("input.fragments needs ions (for example \"c-1:ALL z-dot:ALL\")")
        frag = {k: _none(v) for k, v in frag.items()}
        if "rt_window" in frag:
            frag["rt_window"] = _window(frag["rt_window"], "input.fragments.rt_window")
        fragments = FragmentSettings(**frag)
    parents = bool(data.get("parents", True))
    if not parents and fragments is None:
        raise ProjectError("parents = false needs [input.fragments]: there would be no target")
    scans = data.get("scans", {})
    try:
        scans = MzmlConfig.from_dict(_merge(MzmlConfig().to_dict(), {
            k: _none(v) for k, v in scans.items()}, "input.scans"))
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"[input.scans]: {exc}") from exc
    return InputSettings(
        format=fmt, peptides=_path(base, data["peptides"], "input.peptides"),
        runs=tuple(parsed_runs), parents=parents, target=target, fragments=fragments,
        scans=scans, record_paths=str(data.get("record_paths", "path")),
        data_root=_path(base, data.get("data_root", "."), "input.data_root"),
        thermo_dir=_path(base, data.get("thermo_dir"), "input.thermo_dir"),
    )


def _output_from(data: Mapping, base: Path) -> OutputSettings:
    _check_keys(data, {f.name for f in fields(OutputSettings)}, "[output]")

    def where(key, default):
        value = data.get(key, default)
        return value if isinstance(value, bool) else _path(base, value, f"output.{key}")

    return OutputSettings(
        bundle=_path(base, data.get("bundle"), "output.bundle"),
        overwrite=bool(data.get("overwrite", False)),
        keep_spectra=bool(data.get("keep_spectra", True)),
        tables=where("tables", True), figures=where("figures", False),
        notes=str(data.get("notes", "")),
    )


def project_from(data: Mapping, base: str | Path = ".", path: Path | None = None) -> Project:
    """A :class:`Project` from the parsed TOML ``data``; relative paths are taken relative to
    ``base``."""
    base = Path(base)
    _check_keys(data, {"project", "threads", "input", "run", "output", "analysis"},
                "the project file")
    name = data.get("project")
    if not isinstance(name, str) or not name or ":" in name:
        raise ProjectError("project must be a name (non-empty text without ':')")
    if "input" not in data:
        raise ProjectError("the project file needs an [input] table")
    run = run_config_from(data.get("run", {}))
    output = _output_from(data.get("output", {}), base)
    if output.figures is not False and run.plots is None:  # figures asked for: default plots
        from dataclasses import replace

        from .report.figures import PlotConfig

        run = replace(run, plots=PlotConfig())
    threads = data.get("threads")
    if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int)
                                or threads < 1):
        raise ProjectError("threads must be a whole number of at least 1")
    analysis = _analysis_from(data["analysis"], base) if "analysis" in data else None
    return Project(name, _input_from(data["input"], base), run, output, path, threads, analysis)


ANALYSIS_KEYS = ("reference", "samples", "times", "ks", "fasta", "coverage_times", "redundancy",
                 "weight_exponent", "filters", "choice", "alpha", "regions", "sites",
                 "exclude_flagged", "format", "out", "runs", "overlay",
                 "overlay_two_only", "peptide_match", "modified_peptides")


def _analysis_from(data: Mapping, base: Path):
    """The :class:`hxex.analysis.AnalysisConfig` of an ``[analysis]`` table (relative paths from
    ``base``)."""
    from .analysis import AnalysisConfig

    _check_keys(data, ANALYSIS_KEYS, "[analysis]")
    values = {k: v for k, v in data.items() if k not in ("regions", "sites")}
    fasta = values.get("fasta")
    if isinstance(fasta, str):
        values["fasta"] = str(_path(base, fasta, "analysis.fasta"))
    elif isinstance(fasta, Mapping):
        values["fasta"] = {str(k): str(_path(base, v, f"analysis.fasta.{k}"))
                           for k, v in fasta.items()}
    if "out" in values:
        values["out"] = str(_path(base, values["out"], "analysis.out"))
    if "runs" in values:
        if not isinstance(values["runs"], Mapping):
            raise ProjectError("[analysis.runs] must be a table of run name = bundle")
        values["runs"] = {str(k): str(_path(base, v, f"analysis.runs.{k}"))
                          for k, v in values["runs"].items()}
    if "filters" in values and isinstance(values["filters"], Mapping):
        from .analysis import SpeciesFilters

        values["filters"] = {**SpeciesFilters().__dict__, **values["filters"]}
    if "choice" in values:
        from .analysis import ChoiceConfig

        values["choice"] = {**ChoiceConfig().to_dict(), **values["choice"]}
    values["annotations"] = {"regions": list(data.get("regions", [])),
                             "sites": list(data.get("sites", []))}
    try:
        return AnalysisConfig.from_dict(values)
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"[analysis]: {exc}") from exc


def load_project(path: str | Path) -> Project:
    """Read and check a project file."""
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProjectError(f"cannot read {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ProjectError(f"{path}: not valid TOML ({exc})") from exc
    return project_from(data, path.resolve().parent, path)


# --- reading the input --------------------------------------------------------------------------

def scan_targets(project: Project) -> list[MzmlTarget]:
    """The targets of an mzML or Thermo project: each peptide of the table (if ``parents``),
    and its fragments (if ``[input.fragments]``)."""
    from .fragments import fragment_targets
    from .io.entries import read_peptide_table

    inp = project.input
    targets = []
    for entry in read_peptide_table(inp.peptides):
        if inp.parents:
            targets.append(MzmlTarget(entry, **inp.target))
        if inp.fragments is not None:
            f = inp.fragments
            targets += fragment_targets(
                entry, f.ions, activation=f.activation, precursor_mz=f.precursor_mz,
                precursor_charge=f.precursor_charge, filter_contains=f.filter_contains,
                filter_excludes=f.filter_excludes, rt_window=f.rt_window,
                limit_charge=f.limit_charge, registry=project.run.registry,
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


def read_input(
    project: Project,
    *,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> SpectrumSet:
    """The raw spectra the project's ``[input]`` names (recorded, so the read can be
    repeated)."""
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


def _toml_table(name: str, data: Mapping, comment: bool = False) -> list[str]:
    """``[name]`` with ``data``: scalars first (None as a commented line, it is the default),
    then sub-tables and arrays of tables."""
    c = "# " if comment else ""
    lines = [f"{c}[{name}]"]
    later: list[str] = []
    for key, value in data.items():
        if isinstance(value, Mapping):
            later += [""] + _toml_table(f"{name}.{key}", value, comment)
        elif isinstance(value, list) and value and all(isinstance(v, Mapping) for v in value):
            for item in value:
                later += ["", f"{c}[[{name}.{key}]]"]
                later += [f"{c}{k} = {_toml_value(v)}" for k, v in item.items()]
        elif value is None:
            lines.append(f"# {key} =  (not set)")
        else:
            lines.append(f"{c}{key} = {_toml_value(value)}")
    return lines + later


_INPUT_TEMPLATES = {
    "specexport": """[input]
format = "specexport"
data_dir = "SpecExport"          # the HDExaminer SpecExport folder
# samples = ["WT", "mutant"]     # only these samples
# peptide_ranges = ["0001-0010"] # only these peptides
undeut_time = 0.0                # the time given to undeuterated controls (s)
fulldeut_time = 1000000.0        # the time given to fully deuterated controls (s)
record_paths = "path"            # or "name": record file names only""",
    "hxexpress": """[input]
format = "hxexpress"
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
data_dir = "pooled"              # a SpecExport folder with peptides of several proteins
sample = "pool"                  # the sample name given to its entries
# peptide_ranges = ["0001-0010"]
undeut_time = 0.0
fulldeut_time = 1000000.0
record_paths = "path\"""",
}

_SCAN_TEMPLATE = """[input]
format = "{fmt}"
peptides = "peptides.csv"        # the peptide table (columns as pyHXExpress's metadf)
data_root = "."                  # relative run paths are taken from here
record_paths = "path"            # or "relative" (to data_root) or "name"{thermo}
parents = true                   # read each peptide itself (set false for fragments only)

[input.target]                   # how the peptides' scans are chosen
ms_level = 1
# rt_window = [60.0, 180.0]      # seconds, for runs with a fixed time
# filter_excludes = "sid"        # leave out scans whose filter contains this

# [input.fragments]              # fragment ions of every peptide
# ions = "c-1:ALL z-dot:ALL"
# activation = "ETD"
# precursor_charge = "parent"    # or a charge, or false for any charge
# filter_excludes = "sid"

[input.scans]                    # how scans become spectra (hxex.io.MzmlConfig)
{scans}

# one [[input.runs]] per file; "time" is the labelling time of the whole file (s), or leave it
# out when each scan has its own time (continuous labelling; then "time_offset")
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
        scans = "\n".join(_toml_table("x", MzmlConfig().to_dict())[1:])
        thermo = ('\n# thermo_dir = "/path/to/ThermoRawFileParser"   # Thermo\'s DLLs, or set '
                  'HXEX_THERMO_DIR' if fmt == "thermo" else "")
        input_text = _SCAN_TEMPLATE.format(fmt=fmt, scans=scans, thermo=thermo,
                                           ext="raw" if fmt == "thermo" else "mzML")
    else:
        input_text = _INPUT_TEMPLATES[fmt]
    run = RunConfig().to_dict()
    lines = [
        "# hxex project file (TOML). A key left out keeps its default; false stands for",
        "# \"not set\" where a setting can be unset. Relative paths are relative to this file.",
        "",
        f'project = "{project}"              # prefixes every spectrum id',
        "# threads = 12                    # CPU threads (default: PyTorch's, the physical",
        "#                                 # cores); 12 was fastest on a 24-core CPU",
        "",
        input_text,
        "",
        "[output]",
        'bundle = "results"               # the results bundle (a directory, or .zip)',
        "overwrite = false",
        "keep_spectra = true              # store the raw spectra in the bundle",
        "tables = true                    # CSV tables beside the bundle, or a directory",
        "figures = false                  # figures beside the bundle, or a directory",
        'notes = ""',
        "",
        "# The post-fitting analysis (hxex analyse): uncomment to use. Regions and sites are",
        "# yours to define for the protein; nothing is derived.",
        "# [analysis]",
        '# reference = "WT"                # the sample the others are compared with',
        '# samples = ["WT", "variant"]     # which samples, in this order (default: all)',
        "# times = [4.0, 60.0]             # time points of the per-time figures (default: all",
        "#                                 # but the fully deuterated control)",
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
        "",
        "# Run settings (hxex.run.RunConfig). A [run.bootstrap], [run.controls] or [run.plots]",
        "# section turns that stage on; uncomment one to use it.",
    ]
    scalars = {k: v for k, v in run.items()
               if not isinstance(v, Mapping) and k not in _OPTIONAL_SECTIONS}
    lines += [""] + _toml_table("run", scalars)
    for key, value in run.items():
        if key in _OPTIONAL_SECTIONS and value is None:
            lines += [""] + _toml_table(f"run.{key}", _section_defaults(key), comment=True)
        elif isinstance(value, Mapping):
            lines += [""] + _toml_table(f"run.{key}", value)
    return "\n".join(lines) + "\n"


def describe(project: Project) -> dict[str, Any]:
    """A short summary of a project's settings (for ``hxex check``)."""
    run = project.run
    return {
        "project": project.project,
        "format": project.input.format,
        "K": f"{run.fit.k_min} to {run.fit.k_max}",
        "device": run.fit.backend.device,
        "bootstrap": "off" if run.bootstrap is None else f"{run.bootstrap.n_boot} resamples",
        "controls": "off" if run.controls is None else "on",
        "quality flags": "off" if run.quality is None else "on",
        "bundle": str(project.output.bundle) if project.output.bundle else "(none)",
    }
