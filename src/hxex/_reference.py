# SPDX-License-Identifier: GPL-3.0-or-later
"""The reference pages of the user guide, generated from the code: ``docs/guide/settings.md``
from the config classes, so that every key a project file accepts is listed with its type,
default and meaning (a new setting without a description fails the tests), and
``docs/guide/commands.md`` from the command-line parser (:func:`hxex.cli.build_parser`).

The descriptions are those of the project-file template (:data:`hxex.config._COMMENTS`,
keyed ``table.key``), and :data:`_DESCRIPTIONS` here for the tables the template writes by hand
(``[input]``, ``[output]``, ``[analysis]``). Table introductions come from
:data:`hxex.config._SECTION_COMMENTS`, or :data:`_INTROS` here.

Regenerate both with ``python -m hxex._reference docs/guide``.
"""

from __future__ import annotations

import re
import sys
import types
import typing
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path

from . import config
from .analysis import AnalysisConfig, ChoiceConfig, FragmentAnalysis, SpeciesFilters
from .analysis.compare import Region, Site
from .config import (
    _COMMENTS,
    _SECTION_COMMENTS,
    ANALYSIS_KEYS,
    FragmentSettings,
    InputSettings,
    OutputSettings,
    _hints,
    _section_class,
)
from .io.mzml import MzmlConfig, MzmlRun, MzmlTarget
from .kint import ExchangeConditions
from .run import RunConfig

REQUIRED = "required"
UNSET = "unset"

_FOLDER = "folder"
_SCAN = "scan"
# [input] keys by the formats they apply to (the rest apply to every format)
_INPUT_FORMATS = {
    "data_dir": _FOLDER, "samples": _FOLDER, "peptide_ranges": _FOLDER,
    "peptide_region": _FOLDER, "sample": "pooled",
    "runs": _SCAN, "parents": _SCAN, "target": _SCAN, "fragments": _SCAN, "scans": _SCAN,
    "data_root": _SCAN, "thermo_dir": "thermo",
}

_CONDITIONS = {
    "ph_read": "the pH meter reading in the solvent (glass electrode, uncorrected)",
    "temperature": "degrees Celsius",
    "d2o_fraction": "deuterium atom fraction of the solvent, 0 to 1",
    "reference": 'intrinsic-rate set: "PDLA" (poly-DL-alanine) or "3Ala"',
    "factors": "log10 Bai factors [AL, AR, BL, BR] for modified residues (label = list), "
               "replacing those of their standard residue",
}

_DESCRIPTIONS: dict[str, str] = {
    ".project": "the project's name (text without a colon, `:`); prefixes every spectrum id",
    ".threads": "CPU threads for the command (unset: PyTorch's default, the physical cores); "
                "about half the physical cores was fastest on a 24-core machine",
    "input.format": "specexport, hxexpress, pooled (folder formats), mzml or thermo (scan "
                    "formats)",
    "input.data_dir": "the folder to read (needed for the folder formats)",
    "input.samples": "only these samples (unset: all)",
    "input.peptide_ranges": 'only these peptides, by their residues: "1-10" selects the peptide '
                            'from residue 1 to 10 ("0001-0010" is the same); unset: all',
    "input.peptide_region": "[first, last]: only peptides wholly within these residues; with "
                            "peptide_ranges too, a peptide that matches either; unset: all",
    "input.sample": "the sample name given to the entries of a pooled folder",
    "input.record_paths": 'how the files read are recorded in the bundle: "path" (as given), '
                          '"name" (file names only), or for the scan formats "relative" (to '
                          "data_root)",
    "input.peptides": "scan formats: the peptide table (CSV), needed; each row is read from "
                      "the runs of its sample (every run when it has none); folder formats: a "
                      "table of the files instead of finding them by name",
    "input.runs": "one [[input.runs]] table per file (below)",
    "input.parents": "read each peptide itself; false reads only its fragments (needs "
                     "[[input.fragments]])",
    "input.target": "how the peptides' scans are chosen: the [input.target] table (below)",
    "input.fragments": "fragment ions of every peptide: [[input.fragments]] tables (below)",
    "input.scans": "how scans become spectra: the [input.scans] table (below)",
    "input.data_root": "relative run paths are taken from here",
    "input.thermo_dir": "the folder of Thermo's RawFileReader DLLs (else the HXEX_THERMO_DIR "
                        "environment variable)",
    "input.retention_times": "a table of retention times in minutes (columns rt and sample, "
                             "start_seq, end_seq, peptide, charge)",
    "input.exclude": "spectra left out before fitting: [[input.exclude]] rules (below)",
    "review.table": "the user review table: a copy of spectra.csv with review_status "
                    "(rejected or accepted), review_K (the number of populations to use), and "
                    "note filled in; other columns are ignored. Any name (the guide suggests "
                    "\"user_review_table.csv\"); a name alone is a file beside the project file",
    "input.spectra": "take the raw spectra stored by an earlier run (a bundle or a spectra "
                     "file) instead of reading the input",
    "input.target.ms_level": "MS level of the peptides' scans",
    "input.target.activation": 'activation of the scans (e.g. "HCD"; unset: any)',
    "input.target.precursor_mz": "m/z of the scans' precursor (unset: any)",
    "input.target.precursor_charge": "charge of the scans' precursor (unset: any)",
    "input.target.rt_window": "[start, end] in s of scan time: the scans used in runs with a "
                              "fixed time (a peptide's rt_start, rt_end columns, in minutes, "
                              "take its place)",
    "input.target.filter_contains": "text the scan filter must contain",
    "input.target.filter_excludes": 'text the scan filter must not contain (e.g. "sid")',
    "input.target.parent_mz": "[low, high] m/z the scans' parent isolation must overlap "
                              "(unset: any)",
    "input.target.preactivation": 'MS3: the activation before the last stage ("none" for '
                                  "none)",
    "input.runs.path": "the file (relative paths from data_root)",
    "input.runs.sample": "the sample the file belongs to",
    "input.runs.rep": "replicate number (1 or more)",
    "input.runs.time": 'labelling time of the whole file (s, or text with a unit: "60 s", '
                       '"5 min"); leave it out when each scan has its own time (kinetics)',
    "input.runs.time_offset": "s: the exchange clock is the scan time plus this (when exchange "
                              "began relative to acquisition)",
    "input.runs.usable_from": "s of scan time: this file's scans before it are left out (it "
                              "does not move the time axis; time_offset does)",
    "input.runs.activation": '"target" keeps the targets\' condition; a value replaces it for '
                             "this file only, false removes it",
    "input.runs.precursor_mz": 'as activation: "target", a value, or false (any)',
    "input.runs.precursor_charge": 'as activation: "target", a charge, or false (any; for a '
                                   "file that records a wrong precursor charge)",
    "input.runs.filter_contains": 'as activation: "target", a text, or false',
    "input.runs.filter_excludes": 'as activation: "target", a text, or false (e.g. a control '
                                  "file acquired with in-source activation throughout)",
    "input.runs.role": '"undeut" or "fulldeut": a control file, its scans summed into one '
                       "spectrum without a time (with time as well: a time point that also "
                       "serves as the control)",
    "input.fragments.ions": 'space-separated TYPE:LENGTHS:CHARGES items ("c-1:ALL z-dot:2-6,9"); '
                            'LENGTHS and CHARGES are ALL (the default) or numbers and ranges; '
                            "types such as c, c-1, z-dot, y-NH3, M (the peptide itself)",
    "input.fragments.label": 'names the condition in tables and figures ("gentle")',
    "input.fragments.ms_level": "2, or 3 for MS3 scans",
    "input.fragments.activation": "activation of the last stage",
    "input.fragments.preactivation": 'MS3: the activation before the last stage ("none" for '
                                     "none)",
    "input.fragments.parent_mz": '"parent" (scans whose isolation overlaps the peptide\'s m/z '
                                 "range), [low, high] in m/z, or false (any parent)",
    "input.fragments.precursor_mz": "m/z of the scans' precursor (unset: any)",
    "input.fragments.precursor_charge": '"parent", a charge, or false (any)',
    "input.fragments.filter_contains": "text the scan filter must contain",
    "input.fragments.filter_excludes": "text the scan filter must not contain",
    "input.fragments.rt_window": "[start, end] in s of scan time, for runs with a fixed time "
                                 "(a peptide's rt_start, rt_end columns take its place)",
    "input.fragments.limit_charge": "ALL charges run from 1 to the parent's charge - 1 (false: "
                                    "to the parent's charge); charges written out are used as "
                                    "given",
    "output.bundle": "name of the folder (or .zip archive, a name ending in .zip) that `hxex run` "
                     "creates for the results (the bundle); it must not exist yet unless "
                     "overwrite or keep_previous is set",
    "output.overwrite": "replace an existing bundle (its review is carried over) and its tables "
                        "(its figures and analyses stay until redrawn)",
    "output.keep_previous": "move the existing bundle, tables and figures to the next "
                            "generation (results.1, ...) first",
    "output.keep_spectra": "store the raw spectra in the bundle (needed to rerun from it)",
    "output.tables": "CSV tables: true beside the bundle (<bundle>_tables), a directory, or "
                     "false",
    "output.notes": "free text kept in the bundle's manifest",
    "analysis.reference": "the sample the others are compared with (empty: no comparisons); "
                          '`"run:sample"` names one run\'s sample',
    "analysis.samples": "which samples, in this order (empty: all, the reference first)",
    "analysis.times": "time points of the per-time figures (empty: all)",
    "analysis.exclude_times": "time points left out of the whole analysis (unreliable ones; "
                              "the fits are kept)",
    "analysis.ks": "the fixed K values analysed (1 is needed)",
    "analysis.fasta": "one sequence file for every sample, or a table of sample = file (a "
                      "run's own under its name); unset: the sequence comes from the peptides",
    "analysis.coverage_times": '"undeuterated" (judged on the undeuterated controls), "all", '
                               "or a list of times",
    "analysis.redundancy": 'what the redundancy counts: "peptides" or "spectra"',
    "analysis.weight_exponent": "exponent of the residue averaging's weights",
    "analysis.filters": "which spectra and K=2 descriptions are kept: [analysis.filters] "
                        "(below), or false for all",
    "analysis.choice": "the choice between one and two populations: [analysis.choice] (below)",
    "analysis.alpha": "significance level of the comparisons",
    "analysis.regions": "named stretches of the protein to mark: [[analysis.regions]] (below)",
    "analysis.sites": "single residues to mark: [[analysis.sites]] (below)",
    "analysis.exclude_flagged": "leave out spectra with a quality flag",
    "analysis.format": 'of the figure files: "pdf", "png" or "svg"',
    "analysis.out": "the output directory (unset: beside the bundle, named after it and the "
                    "reference: results_analysis_WT, or results_analysis without a reference)",
    "analysis.runs": "separate runs to compare: [analysis.runs] (below)",
    "analysis.overlay": 'the values of the by-residue comparisons: "low" and "high" (the K=2 '
                        'populations\' fD), "avg" (K=1), and/or "pop_low", "pop_high" (the '
                        'populations\' shares)',
    "analysis.overlay_two_only": "draw K=2 values only where the choice keeps two populations",
    "analysis.borderline_evidence": "where the number of populations differs between samples: "
                                    "borderline unless the two-population side's summed "
                                    "criterion is at least this far below 0 and the other's "
                                    "above 0",
    "analysis.populations": 'how one or two populations are decided: "best_set" (where a '
                            'peptide\'s spectra together are best described by two; a residue '
                            'where such peptides carry half its weight; K=2 compared where both '
                            'samples have two) or "bootstrap" (the residue\'s K=2 positions '
                            'and their bootstrap errors, with [analysis.choice]\'s limits)',
    "analysis.peptide_match": 'which peptides are compared: "range" (the same residues, mutated '
                              'or not) or "sequence" (identical sequences)',
    "analysis.modified_peptides": '"separate" (modified forms as states of their own) or '
                                  '"combined" (averaged with the sample)',
    "analysis.controls": "control settings to change for the analysis: [analysis.controls] "
                         "(below)",
    "analysis.merge": "samples of several runs combined into one: [analysis.merge] (below)",
    "analysis.fragments": "the fragment outputs: [analysis.fragments] (below), or false to "
                          "turn them off",
    "analysis.protection": "opt in to the peptides' protection factors (needs "
                           "[analysis.labelling])",
    "analysis.labelling": "the labelling solvent: [analysis.labelling] (below)",
    "analysis.filters.min_population": "every population of a spectrum within [min_population, "
                                       "1 - min_population]",
    "analysis.filters.max_fD": "no population deuterated beyond this fraction of the amides",
    "analysis.filters.max_fD_se": "every population's fD standard error below this",
    "analysis.filters.k1_between_ranks": "the K=1 value lies between the two populations' values",
    "analysis.filters.replicate_outlier": "leave out a spectrum whose K=1 fD lies more than "
                                          "this many typical replicate standard deviations "
                                          "from the median of its replicates (same charge "
                                          "state, three or more); unset: off",
    "analysis.filters.above_amides": "leave out a spectrum whose K=1 mean exceeds the amide "
                                     "count before the back-exchange correction",
    "analysis.filters.stand_in_controls": '"exclude" or "include" the series whose fully '
                                          "deuterated control is a stand-in (a sample with "
                                          "no measured control keeps them) or whose "
                                          "undeuterated control read above [run.controls] "
                                          "m0_max (m0_high: another species)",
    "analysis.filters.spectrum_quality_filtering": '"loose", "strict", or "none": what the '
                                                   "spectrum quality screens do: leave out "
                                                   "spectra showing another species, keep one "
                                                   "population only, or warn (fragments are "
                                                   "not screened)",
    "analysis.choice.confidence": "a fraction (0.5: 50 %): the interval that must separate the "
                                  "two K=2 values; default 0.5 with populations = "
                                  '"best_set", 0.95 with "bootstrap"',
    "analysis.choice.max_fD_se": "both K=2 standard errors below this",
    "analysis.choice.min_population": "the low-D population (lever rule) within "
                                      "[min_population, 1 - min_population]; default 0.05 "
                                      'with populations = "best_set", 0.10 with "bootstrap"',
    "analysis.regions.name": "the region's name",
    "analysis.regions.start": "first residue (1-based, inclusive)",
    "analysis.regions.end": "last residue (inclusive)",
    "analysis.sites.residue": "the residue number",
    "analysis.sites.label": 'the label drawn ("S19")',
    "analysis.fragments.ion_types": "the fragment ion types analysed (empty: every one of the "
                                    "run)",
    "analysis.fragments.charges": "the fragment charges analysed",
    "analysis.fragments.min_time": "s on the exchange clock: earlier points are drawn but not "
                                   "fitted (not [input.scans] min_time, on the scan clock)",
    "analysis.fragments.labels": "short condition names for the figures: "
                                 "[analysis.fragments.labels] (below)",
    "analysis.fragments.tolerance": "deuterons: a closure c + z - M beyond this is flagged",
    "analysis.fragments.precursor_charge": "the closure's precursor charge (unset: the "
                                           "unreacted precursor)",
    "analysis.fragments.max_rate_error": "ladder steps with rate_se / rate at most this give "
                                         "the per-residue rates (ladder_residues.csv)",
    "analysis.fragments.kint": "opt in: intrinsic rates, protection factors and theoretical "
                               "curves (needs [analysis.fragments.exchange])",
    "analysis.fragments.exchange": "the kinetics' solvent: [analysis.fragments.exchange] "
                                   "(below)",
    **{f"analysis.fragments.exchange.{k}": v for k, v in _CONDITIONS.items()},
    "analysis.fragments.exchange.initial_fraction": "D fraction of the amides when the exchange "
                                                    "began (the start of the theoretical "
                                                    "curves; unset: no curves)",
    **{f"analysis.labelling.{k}": v for k, v in _CONDITIONS.items()},
    "analysis.labelling.initial_fraction": "back-exchange only: the D fraction the amides began "
                                           "with (above d2o_fraction: fD falls with time); "
                                           "leave it out for labelling",
    "run.controls.quench": "pH read, degrees Celsius and D fraction of the quench and LC "
                           "({ ph_read = 2.5, temperature = 0.0, d2o_fraction = 0.0 }), for "
                           'fd_fallback "kint" (unset: none)',
    "run.registry.residues.label": 'the modX label used in sequences ("pS": a lowercase prefix '
                                   "plus the residue letter)",
    "run.registry.residues.base": "the standard residue it modifies",
    "run.registry.residues.delta": 'composition added to the base residue ({ H = 1, P = 1, '
                                   "O = 3 })",
}

_INTROS: dict[str, str] = {
    "input.exclude": "Spectra left out before fitting: they are stored with the raw spectra "
                     "but not picked or fitted, and the bundle records them. A spectrum is "
                     "excluded when it matches every key one rule names; repeat the heading "
                     "for each rule. See [Excluding spectra](excluding-spectra.md).",
    "review": "Decisions made after looking at the results: the user review table and the "
              "exclusion rules below. hxex run (its tables), hxex report and hxex analyse, given "
              "the project file, all apply them; the fits stay in the bundle. See "
              "[Reviewing results](project-file.md#reviewing-results).",
    "review.exclude": "Spectra treated as rejected, in the tables, figures, and analysis; their "
                      "fits stay in the bundle. The same keys as [[input.exclude]]. (Called "
                      "[[analysis.exclude]] before; that name still works.) See "
                      "[Excluding spectra](excluding-spectra.md).",
    "": "Top-level keys, before the first table.",
    "input": "What to read. Which keys apply depends on the format (the Formats column): "
             "folder = the folder formats (specexport, hxexpress, pooled), scan = the scan "
             "formats (mzml, thermo), all = every format. See [Input formats](inputs.md).",
    "input.target": "Scan formats: how the scans of the peptides themselves are chosen.",
    "input.runs": "Scan formats: one table per file. A file with a time is one time point, a "
                  "file with a role is a control, and a file with neither is a kinetics run "
                  "whose every scan has its own time. Two clocks: the scan time (acquisition "
                  "clock) and the exchange clock, scan time + time_offset.",
    "input.fragments": "Scan formats: fragment ions made from every peptide of the table. "
                       "Repeat the table for other conditions (e.g. with and without in-source "
                       "activation).",
    "input.scans": "Scan formats: how scans become spectra (hxex.io.MzmlConfig).",
    "output": "Where results go. The bundle and the folders named after it (<bundle>_tables, "
              "<bundle>_figures, <bundle>_analysis_<reference>, and the generations "
              "<bundle>.1, ...) belong to *hxex*: a later run replaces or renames them.",
    "analysis": "The post-fitting analysis that `hxex analyse` writes (`hxex.analysis`). See "
                "docs/ANALYSIS.md for the methods.",
    "analysis.filters": "Which spectra and K=2 descriptions the analysis trusts. Whole "
                        "spectra: spectrum_quality_filtering, replicate_outlier, "
                        "above_amides, stand_in_controls (the "
                        "spectra left out are listed in left_out_spectra.csv; one you accept "
                        "in the review is kept). K=2 descriptions per spectrum: "
                        "min_population, max_fD; per species and time: max_fD_se, "
                        "k1_between_ranks.",
    "analysis.choice": "The limits of the choice between one and two populations. With "
                       'populations = "bootstrap" they are the rule itself, so their defaults '
                       'are strict (0.95, 10 %); with "best_set" they only check the result '
                       "of the fits' comparison (0.5, 5 %). A key you set applies in either "
                       "mode. A residue whose K=2 standard errors are unknown keeps the "
                       "one-population description.",
    "analysis.controls": "Any key of [run.controls]; the controls are then recomputed from the "
                         "run's K=1 fits.",
    "analysis.runs": "Separate runs to compare: run name = bundle path (default: this "
                     "project's bundle). The tables then get a run column.",
    "analysis.merge": 'Samples combined across runs: `"run:sample" = label`; the same label pools '
                      "them (fragment series are drawn side by side, not pooled).",
    "analysis.regions": "Repeat the table for every region (regions may overlap).",
    "analysis.sites": "Repeat the table for every site (a modification, a mutation).",
    "analysis.fragments": "The fragment outputs of a run with fragment series (ladders, rates, "
                          "closure, per-residue rates); written whenever the run has fragments.",
    "analysis.fragments.labels": 'Short condition names: condition = name ("gentle" = "G"), '
                                 "or a run's own under its name ([analysis.fragments.labels."
                                 "RUN]).",
    "analysis.fragments.exchange": "The solvent the fragment kinetics happen in, for every "
                                   "sample, or one table per sample ([analysis.fragments."
                                   "exchange.SAMPLE]) or per run and sample. Its d2o_fraction is "
                                   "the kinetics' solvent, not the labelling buffer's.",
    "analysis.labelling": "The labelling solvent of the peptide protection factors, for every "
                          "sample, or one table per sample ([analysis.labelling.SAMPLE]) or per "
                          "run and sample.",
    "run.picker.amide_rules": _SECTION_COMMENTS["run.picker.amide_rules"] + " Keys are ion "
                              "types, values true or false.",
    "run.registry.residues": "One table per modified residue; the default registry holds acK, "
                             "pS, pT and pY.",
}


@dataclass(frozen=True)
class Setting:
    key: str
    kind: str
    default: str
    text: str | None
    formats: str = ""


@dataclass(frozen=True)
class Table:
    name: str  # dotted, "" for the top level
    array: bool  # [[name]]
    settings: tuple[Setting, ...]
    intro: str = ""

    @property
    def heading(self) -> str:
        if not self.name:
            return "Top level"
        return f"[[{self.name}]]" if self.array else f"[{self.name}]"


def _kind(hint) -> str:
    """A field's type in the words of a TOML file."""
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        members = [m for m in typing.get_args(hint) if m is not type(None)]
    else:
        members = [hint]
    words = []
    for member in members:
        origin = typing.get_origin(member) or member
        if member is typing.Any:
            words.append("table")
        elif origin is bool:
            words.append("true/false")
        elif origin is int:
            words.append("integer")
        elif origin is float:
            words.append("number")
        elif origin is str:
            words.append("text")
        elif origin is Path:
            words.append("path")
        elif isinstance(origin, type) and issubclass(origin, (tuple, list)):
            words.append("list")
        else:
            words.append("table")
    return " or ".join(dict.fromkeys(words))


def _value(value) -> str:
    if value is None:
        return UNSET
    if isinstance(value, Path):
        value = str(value)
    if is_dataclass(value):
        return "(table)"
    if isinstance(value, Mapping) and not value:
        return "{}"
    if isinstance(value, Mapping):
        return "(table)"
    if isinstance(value, (list, tuple)) and value and is_dataclass(value[0]):
        return "(tables)"
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, Mapping) for v in value):
        return "(tables)"
    return f"`{config._toml_value(list(value) if isinstance(value, tuple) else value)}`"


def _field_default(f):
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return MISSING


def _settings(cls, table: str, *, keys=None, skip=(), defaults: Mapping | None = None,
              formats: Mapping | None = None) -> tuple[Setting, ...]:
    """The settings of ``table``, one per field of ``cls`` (or per name of ``keys``, typed and
    defaulted by ``cls``'s fields), with the field default unless ``defaults`` gives one."""
    hints = _hints(cls)
    by_name = {f.name: f for f in fields(cls)}
    instance_defaults = defaults or {}
    out = []
    for name in (by_name if keys is None else keys):
        if name in skip:
            continue
        f = by_name[name]
        default = instance_defaults.get(name, _field_default(f))
        shown = REQUIRED if default is MISSING else _value(default)
        text = _COMMENTS.get(f"{table}.{name}") or _DESCRIPTIONS.get(f"{table}.{name}")
        out.append(Setting(name, _kind(hints[name]), shown, text,
                           (formats or {}).get(name, "all" if formats is not None else "")))
    return tuple(out)


def _run_tables(name: str, cls, values: Mapping) -> list[Table]:
    """``[name]`` (a section of [run]) and its nested sections, scalars first."""
    hints = _hints(cls)
    scalar = [k for k in values if not isinstance(values[k], Mapping)
              and not (isinstance(values[k], list) and values[k]
                       and all(isinstance(v, Mapping) for v in values[k]))]
    tables = [Table(name, False, _settings(cls, name, keys=scalar, defaults=values),
                    _SECTION_COMMENTS.get(name, _INTROS.get(name, "")))]
    for key, value in values.items():
        sub = config._config_class(hints.get(key))
        if key in scalar:
            continue
        if isinstance(value, Mapping) and sub is not None:
            tables += _run_tables(f"{name}.{key}", sub, value)
        elif isinstance(value, Mapping):  # a free-form table (amide_rules)
            tables.append(Table(f"{name}.{key}", False, (), _INTROS.get(f"{name}.{key}", "")))
        else:  # an array of tables (the registry's residues)
            first = value[0]
            tables.append(Table(f"{name}.{key}", True, tuple(
                Setting(k, "table" if isinstance(v, Mapping) else "text", REQUIRED,
                        _DESCRIPTIONS.get(f"{name}.{key}.{k}")) for k, v in first.items()),
                _INTROS.get(f"{name}.{key}", "")))
    return tables


# the keys of an exclusion rule ([[input.exclude]], [[review.exclude]]; hxex.exclusions)
_RULE = (
    Setting("sample", "text", UNSET, "the sample"),
    Setting("peptide_range", "text", UNSET, 'the peptide, by its residues ("45-60" is '
            '"0045-0060")'),
    Setting("charge", "integer", UNSET, "the charge state"),
    Setting("time", "number or text", UNSET, 'a time point (s, or text with a unit: "30 s"); '
            "a control has no time"),
    Setting("role", "text", UNSET, '"undeut" or "fulldeut": the controls'),
    Setting("rep", "integer", UNSET, "the replicate"),
    Setting("file", "text", UNSET, "the data file's name: a run's mzML or raw file, an "
            "HX-Express spreadsheet, or a SpecExport spectrum file"),
    Setting("spectrum_id", "text", UNSET, "one spectrum, by its id (spectra.csv)"),
    Setting("note", "text", '`""`', "why: kept with the record, and shown by hxex check and "
            "hxex info"),
)


def tables() -> list[Table]:
    """Every table a project file accepts, in the order of the reference."""
    out = [Table("", False, (
        Setting("project", "text", REQUIRED, _DESCRIPTIONS[".project"]),
        Setting("threads", "integer", UNSET, _DESCRIPTIONS[".threads"])), _INTROS[""])]
    out.append(Table("input", False, _settings(
        InputSettings, "input", defaults={"data_root": Path(".")}, formats=_INPUT_FORMATS),
        _INTROS["input"]))
    out.append(Table("input.target", False, _settings(
        MzmlTarget, "input.target", keys=config._TARGET_KEYS), _INTROS["input.target"]))
    out.append(Table("input.runs", True, _settings(MzmlRun, "input.runs"),
                     _INTROS["input.runs"]))
    out.append(Table("input.fragments", True, _settings(FragmentSettings, "input.fragments"),
                     _INTROS["input.fragments"]))
    out.append(Table("input.exclude", True, _RULE, _INTROS["input.exclude"]))
    out.append(Table("input.scans", False, _settings(MzmlConfig, "input.scans"),
                     _INTROS["input.scans"]))
    out.append(Table("output", False, _settings(OutputSettings, "output"), _INTROS["output"]))
    run = RunConfig().to_dict()
    scalars = [k for k, v in run.items()
               if not isinstance(v, Mapping) and k not in config._OPTIONAL_SECTIONS]
    out.append(Table("run", False, _settings(RunConfig, "run", keys=scalars, defaults=run),
                     _SECTION_COMMENTS["run"]))
    for key, value in run.items():
        if key in scalars:
            continue
        values = value if isinstance(value, Mapping) else config._section_defaults(key)
        out += _run_tables(f"run.{key}", _section_class(key), values)
    out.append(Table("review", False, (
        Setting("table", "path", UNSET, _DESCRIPTIONS["review.table"]),), _INTROS["review"]))
    out.append(Table("review.exclude", True, _RULE, _INTROS["review.exclude"]))
    nested = ("filters", "choice", "regions", "sites", "runs", "merge", "controls", "fragments",
              "labelling")
    out.append(Table("analysis", False, _settings(
        AnalysisConfig, "analysis", keys=[k for k in ANALYSIS_KEYS if k not in nested]),
        _INTROS["analysis"]))
    out.append(Table("analysis.filters", False, _settings(SpeciesFilters, "analysis.filters"),
                     _INTROS["analysis.filters"]))
    out.append(Table("analysis.choice", False, _settings(ChoiceConfig, "analysis.choice"),
                     _INTROS["analysis.choice"]))
    for name in ("controls", "runs", "merge"):
        out.append(Table(f"analysis.{name}", False, (), _INTROS[f"analysis.{name}"]))
    out.append(Table("analysis.regions", True, _settings(Region, "analysis.regions"),
                     _INTROS["analysis.regions"]))
    out.append(Table("analysis.sites", True, _settings(Site, "analysis.sites"),
                     _INTROS["analysis.sites"]))
    out.append(Table("analysis.fragments", False, _settings(
        FragmentAnalysis, "analysis.fragments", skip=("labels", "exchange")),
        _INTROS["analysis.fragments"]))
    out.append(Table("analysis.fragments.labels", False, (), _INTROS["analysis.fragments.labels"]))
    for name in ("analysis.fragments.exchange", "analysis.labelling"):
        out.append(Table(name, False, _settings(ExchangeConditions, name), _INTROS[name]))
    from .report.figures import PlotConfig

    out.append(Table("figures", False, _settings(PlotConfig, "figures"),
                     _SECTION_COMMENTS["figures"] + " See [Outputs](outputs.md)."))
    return out


def missing() -> list[str]:
    """``table.key`` of every setting without a description, and every [analysis] key the
    reference does not list."""
    gaps = [f"{t.name}.{s.key}" for t in tables() for s in t.settings if not s.text]
    listed = {s.key for t in tables() if t.name == "analysis" for s in t.settings}
    listed |= {t.name.split(".")[1] for t in tables() if t.name.count(".") == 1
               and t.name.startswith("analysis.")}
    gaps += [f"analysis.{k} (not listed)" for k in ANALYSIS_KEYS if k not in listed]
    return gaps


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _anchor(table: Table) -> str:
    return "top-level" if not table.name else table.name.replace(".", "")


def markdown() -> str:
    """The settings reference page."""
    all_tables = tables()
    lines = [
        "<!-- Generated by `python -m hxex._reference docs/guide`; do not edit. "
        "The descriptions live in src/hxex/config.py (_COMMENTS) and src/hxex/_reference.py. -->",
        "",
        "# Settings reference",
        "",
        "Every key a project file (`hxex.toml`) accepts, table by table, with its type, its "
        "default and what it does. `hxex init --format FORMAT` writes a file with these "
        "defaults and shorter comments; [The project file](project-file.md) explains how the "
        "file works.",
        "",
        "Conventions: a key left out keeps its default. TOML has no null, so `false` stands for "
        "*unset* where a setting can be unset (an optional stage, a limit); a switch set to "
        "`false` is off. Times are in seconds, written as a number or as text with a unit "
        '(`"5 min"`, `"20 h"`). Relative paths are relative to the project file. '
        "*required* means the table needs the key. A heading in double brackets "
        "(`[[input.runs]]`) is for something there can be several of: write the heading once "
        "for each, each followed by its own settings (with only one, single brackets work "
        "too).",
        "",
        "## Tables",
        "",
    ]
    lines += [f"- [`{t.heading}`](#{_anchor(t)})" if t.name else "- [Top level](#top-level)"
              for t in all_tables]  # GitHub's heading anchors drop the brackets and dots
    for t in all_tables:
        lines += ["", "## " + (f"`{t.heading}`" if t.name else "Top level"), ""]
        if t.intro:
            lines += [t.intro, ""]
        if not t.settings:
            continue
        with_formats = any(s.formats for s in t.settings)
        head = "| Key | Type | Default |" + (" Formats |" if with_formats else "") + " Meaning |"
        lines += [head, "|" + "---|" * (head.count("|") - 1)]
        for s in t.settings:
            row = f"| `{s.key}` | {s.kind} | {s.default} |"
            if with_formats:
                row += f" {s.formats} |"
            lines.append(row + f" {_cell(s.text or '')} |")
    return "\n".join(lines) + "\n"


def _flags(action) -> str:
    if action.option_strings:
        names = ", ".join(f"`{o}`" for o in action.option_strings)
        if action.nargs == 0:
            return names
        metavar = action.metavar or (
            "{" + ",".join(action.choices) + "}" if action.choices else action.dest.upper())
        return names + (f" `[{metavar}]`" if action.nargs == "?" else f" `{metavar}`")
    return f"`{_positional(action)}`"


def _positional(action) -> str:
    return (action.metavar or action.dest) + (" ..." if action.nargs == "+" else "")


def _arguments(parser) -> list[str]:
    """The table of a parser's arguments (help and sub-commands left out)."""
    import argparse

    rows = []
    for action in parser._actions:  # argparse keeps no public list of its arguments
        if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
            continue
        if isinstance(action, argparse._VersionAction):
            text = "print the version and exit"
        else:
            text = action.help or ""
        # a NAME:name pair (RUN:sample) is something to type: code, which also keeps
        # emoji-converting previews from reading ":s" as a face
        text = re.sub(r"(?<![`\w])([A-Za-z_]+:[A-Za-z_]+)(?![`\w])", r"`\1`", text)
        rows.append(f"| {_cell(_flags(action))} | {_cell(text)} |")
    if not rows:
        return []
    return ["| Argument | Meaning |", "|---|---|", *rows]


def undescribed_arguments() -> list[str]:
    """``command argument`` of every command-line argument without a help text."""
    import argparse

    from .cli import build_parser

    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    out = []
    for name, p in [("hxex", parser), *sub.choices.items()]:
        out += [f"{name} {a.dest}" for a in p._actions
                if not a.help and not isinstance(a, (argparse._HelpAction,
                                                     argparse._SubParsersAction,
                                                     argparse._VersionAction))]
    return out


def commands_markdown() -> str:
    """The command-line reference page."""
    import argparse

    from .cli import build_parser

    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    helps = {a.dest: a.help for a in sub._choices_actions}
    names: dict[int, list[str]] = {}
    for name, p in sub.choices.items():  # aliases share their command's parser
        names.setdefault(id(p), []).append(name)
    lines = [
        "<!-- Generated by `python -m hxex._reference docs/guide`; do not edit. The text is "
        "the help of src/hxex/cli.py. -->",
        "",
        "# Command-line reference",
        "",
        "Every command and option of *hxex*, from the program's own definitions (`hxex "
        "COMMAND --help` prints the same). [The project file and the commands]"
        "(project-file.md) shows how they fit together.",
        "",
        "Messages and progress go to the terminal's error stream, results and summaries to its "
        "output. Ctrl-C stops a run cleanly after the current step (a second Ctrl-C aborts at "
        "once). Exit status: 0 done, 1 an error (its message is printed), 2 a usage error, 130 "
        "stopped.",
        "",
        "## Options of every command",
        "",
        "Given before the command: `hxex -v run hxex.toml`.",
        "",
        *_arguments(parser),
    ]
    for p in dict.fromkeys(sub.choices.values()):
        first, *aliases = names[id(p)]
        positional = " ".join(_positional(a) for a in p._actions if not a.option_strings)
        lines += ["", f"## hxex {first}", "", helps.get(first, "").capitalize() + ".", ""]
        lines += [f"`hxex {' '.join(filter(None, (first, positional, '[options]')))}`", ""]
        if aliases:
            lines += [f"Also spelt `hxex {', '.join(aliases)}`.", ""]
        lines += _arguments(p)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write ``settings.md`` and ``commands.md`` into the folder named by ``argv[0]`` (the
    settings page to stdout when none is named)."""
    argv = sys.argv[1:] if argv is None else argv
    gaps = missing()
    if gaps:
        print("settings without a description:\n  " + "\n  ".join(gaps), file=sys.stderr)
        return 1
    if argv:
        folder = Path(argv[0])
        (folder / "settings.md").write_text(markdown(), encoding="utf-8")
        (folder / "commands.md").write_text(commands_markdown(), encoding="utf-8")
    else:
        sys.stdout.write(markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
