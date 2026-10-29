# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line entry point (``hxex``).

::

    hxex init [--format F] [-o hxex.toml]   write a project file with every default
    hxex check PROJECT                      check it and show what would be read
    hxex run PROJECT                        read, pick, fit, bootstrap; bundle, tables, figures
    hxex resume BUNDLE                      add the bootstrap to a run that stopped during it
    hxex report BUNDLE                      tables and figures from a saved bundle
    hxex info BUNDLE                        what a bundle holds
    hxex analyse PROJECT|BUNDLE...          the post-fitting analysis: tables and figures

The project file is described in :mod:`hxex.config`. Progress and messages go to stderr through
``logging`` (``-q`` for warnings only, ``-v`` for more); results and summaries go to stdout.
Ctrl-C stops a run cleanly after the current step (a second Ctrl-C aborts at once); a run
stopped during its bootstrap keeps the bundle of its fits, which ``hxex resume`` completes.
Exit status: 0 done, 1 an error (its message is logged), 2 a usage error, 130 stopped.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import signal
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import __version__

log = logging.getLogger("hxex.cli")

EXIT_ERROR, EXIT_STOPPED = 1, 130


def build_parser() -> argparse.ArgumentParser:
    from .config import FORMATS

    parser = argparse.ArgumentParser(
        prog="hxex",
        description="Multimodal binomial-mixture analysis of HDX-MS spectra.",
    )
    parser.add_argument("--version", action="version", version=f"hxex {__version__}")
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    noise.add_argument("-v", "--verbose", action="store_true", help="more messages")
    parser.add_argument("--progress-every", type=float, default=30.0, metavar="SECONDS",
                        help="seconds between progress lines (default 30)")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("init", help="write a project file with every default")
    p.add_argument("--format", choices=FORMATS, default="specexport",
                   help="the input format (default specexport)")
    p.add_argument("--project", default="my_project", help="the project name")
    p.add_argument("-o", "--output", default="hxex.toml",
                   help="the file to write (default hxex.toml; '-' for stdout)")
    p.add_argument("--force", action="store_true",
                   help="replace an existing file (without it, init refuses to write over one)")

    p = sub.add_parser("check", help="check a project file and show what would be read")
    p.add_argument("project_file", metavar="PROJECT", help="the project file (TOML)")
    _spectra_option(p)

    p = sub.add_parser("run", help="run a project: read, pick, fit, bootstrap, write the "
                                   "bundle and tables (figures: hxex report)")
    p.add_argument("project_file", metavar="PROJECT", help="the project file (TOML)")
    p.add_argument("--bundle", help="the results bundle (instead of output.bundle)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing bundle (and its tables), carrying its review "
                        "over; also a tables folder left from a bundle that is gone")
    p.add_argument("--keep-previous", action="store_true",
                   help="move an existing bundle (with its tables, figures and analysis "
                        "folders) to the next generation, NAME.N, before writing")
    p.add_argument("--device", help="cpu, cuda, cuda:1, ... (instead of run.fit.backend)")
    _spectra_option(p)
    _threads_option(p)

    p = sub.add_parser("resume", help="complete a bundle's bootstrap: finish a stopped one, add "
                                      "or redo one, or bootstrap the review's chosen K")
    p.add_argument("bundle", metavar="BUNDLE|PROJECT",
                   help="a results bundle (a folder, or .zip) waiting for its bootstrap, or a "
                        "project file: its [output] bundle gets a bootstrap with its "
                        "[run.bootstrap] settings (redone only if they changed) and one at the "
                        "K its [review] table chooses where that is missing; the fits kept")
    p.add_argument("--device", help="cpu, cuda, cuda:1, ... (instead of the run's device)")
    _threads_option(p)
    p.add_argument("--tables", nargs="?", const=True, default=None, metavar="DIR",
                   help="also write the tables (default: beside the bundle)")

    p = sub.add_parser("report", help="draw the figures, and write the tables again, from a "
                                      "bundle; fits nothing")
    p.add_argument("bundle", metavar="BUNDLE|PROJECT",
                   help="a results bundle (a folder, or .zip), or a project file: its [output] "
                        "bundle, with its [review] decisions applied and its [figures] "
                        "settings")
    _threads_option(p)
    p.add_argument("--tables", nargs="?", const=True, default=None, metavar="DIR",
                   help="write the tables (default: beside the bundle, or a project file's "
                        "[output] tables)")
    p.add_argument("--figures", nargs="?", const=True, default=None, metavar="DIR",
                   help="write the figures (default: beside the bundle)")
    _output_options(p, "the tables or figures folder")
    p.add_argument("--series", choices=("flagged", "all"),
                   help="which series to draw (default: [figures] series, else flagged)")
    p.add_argument("--k", type=int, metavar="K",
                   help="draw every series at this number of populations (default: [figures] "
                        "k, else each spectrum's own)")
    p.add_argument("--format", dest="fig_format", choices=("pdf", "png", "svg"),
                   help="figure file format")

    p = sub.add_parser("analyse", aliases=["analyze"],
                       help="the post-fitting analysis of a bundle: tables and figures")
    p.add_argument("source", metavar="PROJECT|BUNDLE", nargs="+",
                   help="a project file (its [analysis] settings, and [analysis.runs] or "
                        "output.bundle; a file with only [analysis] is enough), a bundle, or "
                        "several bundles to compare as separate "
                        "runs (NAME=BUNDLE, or BUNDLE named after its file; the tables get a "
                        "run column)")
    p.add_argument("--bundle", help="the bundle (instead of the project's output.bundle)")
    p.add_argument("--out", help="the output directory (default: [analysis] out, else the "
                                 "bundle's name + _analysis + _REFERENCE when there is a "
                                 "reference, e.g. results_analysis_WT)")
    _output_options(p, "the analysis folder")
    p.add_argument("--reference", help="the reference sample (instead of [analysis] reference; "
                                       "RUN:sample when the sample is in several runs)")
    p.add_argument("--format", dest="fig_format", choices=("pdf", "png", "svg"),
                   help="figure file format")
    _threads_option(p)

    p = sub.add_parser("info", help="what a results bundle holds")
    p.add_argument("bundle", metavar="BUNDLE", help="a results bundle (a folder, or .zip)")
    p.add_argument("--verify", action="store_true", help="also check every file's checksum")
    return parser


def _spectra_option(p) -> None:
    p.add_argument("--spectra", metavar="PATH",
                   help="take the raw spectra stored in this bundle (or spectra file) instead "
                        "of reading the input (as input.spectra)")


def _output_options(p, folder: str) -> None:
    """--overwrite and --keep-previous for a command that writes an output folder."""
    p.add_argument("--overwrite", action="store_true",
                   help=f"replace hxex's files in {folder} when it holds earlier output "
                        "(or [output] overwrite = true)")
    p.add_argument("--keep-previous", action="store_true",
                   help=f"move {folder} to NAME.N first when it holds earlier output (or "
                        "[output] keep_previous = true)")


def _threads_option(p) -> None:
    p.add_argument("--threads", type=_positive_int, metavar="N",
                   help="CPU threads (default: the project file's threads, else PyTorch's)")


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


@contextlib.contextmanager
def _threads(n: int | None):
    """PyTorch's CPU threads set to ``n`` while a command runs (restored afterwards; None leaves
    PyTorch's choice). The setting is the process's, which the command line owns."""
    if n is None:
        yield
        return
    import torch

    before = torch.get_num_threads()
    torch.set_num_threads(n)
    log.info("using %d CPU thread(s)", n)
    try:
        yield
    finally:
        torch.set_num_threads(before)


@contextlib.contextmanager
def _logging_to_stderr(args):
    """hxex's messages on stderr while a command runs (the logger is restored afterwards, for
    programs and tests that call :func:`main`)."""
    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logger = logging.getLogger("hxex")
    saved = logger.handlers[:], logger.level, logger.propagate
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False
    try:
        yield
    finally:
        logger.handlers[:], logger.level, logger.propagate = saved


def _say(text: str = "") -> None:
    """A line of a result or summary, on stdout."""
    sys.stdout.write(text + "\n")


class _Stop:
    """The cancel hook of a command: True after Ctrl-C (a second Ctrl-C aborts at once)."""

    def __init__(self):
        self.asked = False

    def __call__(self) -> bool:
        return self.asked

    def _on_interrupt(self, signum, frame):
        self.asked = True
        log.warning("stopping after the current step (Ctrl-C again to abort now)")
        signal.signal(signal.SIGINT, signal.default_int_handler)

    @contextlib.contextmanager
    def installed(self):
        try:
            previous = signal.signal(signal.SIGINT, self._on_interrupt)
        except ValueError:  # not the main thread: no handler, Ctrl-C raises as usual
            yield self
            return
        try:
            yield self
        finally:
            signal.signal(signal.SIGINT, previous)


def _beside(bundle: Path, value, suffix: str) -> Path | None:
    if value is None:
        return None
    return bundle.with_name(bundle.name + suffix) if value is True else Path(value)


# --- commands -----------------------------------------------------------------------------------

def cmd_init(args) -> int:
    from .config import template

    text = template(args.format, args.project)
    if args.output == "-":
        sys.stdout.write(text)
        return 0
    out = Path(args.output)
    if out.exists() and not args.force:
        log.error("%s exists (use --force to replace it)", out)
        return EXIT_ERROR
    out.write_text(text, encoding="utf-8")
    log.info("wrote %s (format %s): edit the input paths, then 'hxex check %s'. The output "
             "bundle and the folders named after it (NAME_tables, NAME_figures, NAME_analysis_..., "
             "NAME.N) are "
             "hxex's: later runs replace or rename them", out, args.format, out)
    return 0


def _project(args, *, analysis_only: bool = False):
    """The project of ``args.project_file``, with ``--spectra`` in place of input.spectra
    (``analysis_only``: a file with only an ``[analysis]`` table is accepted)."""
    from .config import load_project

    return load_project(args.project_file, analysis_only=analysis_only,
                        spectra=getattr(args, "spectra", None) or None)


def cmd_check(args) -> int:
    """Print what the project would read, then every problem found (each a line; the exit
    status says whether there were any)."""
    from .config import (
        FOLDER_FORMATS,
        ProjectError,
        describe,
        folder_entries,
        stored_spectra_source,
    )

    project = _project(args, analysis_only=True)
    for key, value in describe(project).items():
        _say(f"{key:14s} {value}")
    inp = project.input
    problems: list[str] = []
    if inp is None:
        _say(f"{'input':14s} none (a project file for hxex analyse)")
    elif inp.spectra is not None:
        try:
            source = stored_spectra_source(inp.spectra) or {}
        except ProjectError as exc:
            problems.append(str(exc))
        else:
            _say(f"{'spectra':14s} stored in {inp.spectra} (read by "
                 f"{source.get('reader', 'unknown')}, project {source.get('project', 'unknown')}"
                 "); the input files are not read")
    elif inp.format in FOLDER_FORMATS:
        if inp.peptides is not None and not inp.peptides.is_file():
            problems.append(f"input.peptides {inp.peptides}: no such file")
        if not inp.data_dir.is_dir():
            problems.append(f"input.data_dir {inp.data_dir}: no such directory")
        if not problems:
            entries = folder_entries(project)
            samples = Counter(e.sample for e in entries)
            _say(f"{'entries':14s} {len(entries)} in {inp.data_dir}")
            roles = _folder_control_files(inp, entries)
            for s, n in sorted(samples.items()):
                _say(f"  {s or '(no sample)'}: {n}" + _control_file_text(roles, s))
            if roles is None:
                _say(f"{'controls':14s} the column labels of each sheet mark them (undeut, TD)")
            if not entries:
                problems.append(f"no spectra of format {inp.format} in {inp.data_dir}")
            problems += _check_peptides(project, entries)
            if not problems and _has_rules(project):
                from .io import ReadError, spectrum_index

                try:
                    index = spectrum_index(
                        entries, inp.data_dir, inp.format, project=project.project,
                        samples=inp.samples, peptide_ranges=inp.peptide_ranges,
                        peptide_region=inp.peptide_region)
                except ReadError as exc:
                    problems.append(str(exc))
                else:
                    problems += _check_exclusions(project, index, inp.format)
    else:
        rows: list[dict] = []
        problems += _check_scan_files(project, rows)
        if not problems and _has_rules(project):
            from .exclusions import SpectrumIndex

            problems += _check_exclusions(project, SpectrumIndex.from_rows(rows), inp.format)
    if inp is not None and inp.retention_times is not None and not inp.retention_times.is_file():
        problems.append(f"input.retention_times {inp.retention_times}: no such file")
    for n, line in enumerate(_output_lines(project)):
        _say(f"{'' if n else 'outputs':14s} {line}")
    if problems:
        for p in problems:
            log.error("%s", p)
        log.warning("%d problem(s) found", len(problems))
        return EXIT_ERROR
    _say("ok")
    return 0


def _output_lines(project) -> list[str]:
    """What ``hxex check`` says of the project's output folders: which exist, which commands
    would then need ``--overwrite`` or ``--keep-previous`` (or what ``[output] overwrite`` /
    ``keep_previous`` will do), and whether each was made from the results in the bundle now.
    Information only: each command refuses by itself, before doing any work."""
    from .io import holds_output, made_from
    from .io.bundle import analysis_folder_name, analysis_states, bundle_manifest

    out = project.output
    bundle = out.bundle
    if bundle is None:
        return ["no [output] bundle"]
    flags = ("[output] overwrite replaces it" if out.overwrite else
             "[output] keep_previous moves it to NAME.N first" if out.keep_previous else "")

    def needs(command: str) -> str:
        verb = "need" if " and " in command else "needs"
        return flags or f"{command} {verb} --overwrite or --keep-previous"

    created = (bundle_manifest(bundle) or {}).get("created", "") if bundle.exists() else ""
    lines = []
    if bundle.exists():
        lines.append(f"{bundle.name} exists: {needs('hxex run')}")
    tables = _beside(bundle, out.tables, "_tables") if out.tables is not False else None
    for folder, command in ((tables, "hxex run and hxex report"),
                            (bundle.with_name(bundle.name + "_figures"),
                             "hxex report --figures")):
        if folder is None or not holds_output(folder):
            continue
        stamp = (made_from(folder) or {}).get("")
        state = ("left from a bundle that is gone" if not bundle.exists()
                 else "made from results not recorded" if not stamp
                 else "up to date" if stamp == created else "made from earlier results")
        lines.append(f"{folder.name}: {state}; {needs(command)}")
    analysis = project.analysis  # none: hxex analyse uses the default settings
    wanted = (Path(analysis.out) if analysis is not None and analysis.out else
              bundle.with_name(analysis_folder_name(
                  bundle.name, analysis.reference if analysis is not None else None)))
    folders = dict(analysis_states(bundle))
    if wanted is not None and wanted not in folders and holds_output(wanted):
        folders[wanted] = ""
    for folder, state in sorted(folders.items()):
        text = ("up to date" if state is None else "made from earlier results" if state
                else "made from results not recorded")
        if wanted is not None and folder.resolve() == wanted.resolve():
            text += f"; {needs('hxex analyse')}"
        lines.append(f"{folder.name}: {text}")
    return lines or [f"none yet: hxex run writes {bundle.name}"]


def _has_rules(project) -> bool:
    """Whether the project has exclusion rules for ``hxex check`` to count."""
    return bool(project.input.exclude or _view_exclude(project))


def _check_exclusions(project, index, reader: str) -> list[str]:
    """Count the spectra each exclusion rule matches among those the read would make
    (``index``, :class:`hxex.exclusions.SpectrumIndex`; the control roles of ``undeut_at`` and
    ``fulldeut_at`` applied first, as the run does): one line per rule, and a problem for each
    rule that matches nothing. ``[[review.exclude]]`` rules count among the spectra the
    ``[[input.exclude]]`` rules leave."""
    import dataclasses

    from .controls import apply_control_roles
    from .exclusions import matches, why_nothing

    controls = project.run.controls
    if controls is not None and (controls.undeut_at is not None
                                 or controls.fulldeut_at is not None):
        try:
            role = apply_control_roles(index.role, index.time, controls,
                                       where="[run.controls]")
        except ValueError as exc:
            return [str(exc)]
        index = dataclasses.replace(index, role=role.astype(object))
    problems = []
    left = np.ones(len(index), dtype=bool)
    for where, rules, within in (("input.exclude", project.input.exclude, None),
                                 ("review.exclude", _view_exclude(project), left)):
        for i, rule in enumerate(rules, start=1):
            hit = matches(rule, index, reader)
            if within is not None:
                hit &= within
            n = int(hit.sum())
            if where == "input.exclude":
                left &= ~hit
            text = f"[[{where}]] #{i} ({rule.describe()})"
            if n:
                _say(f"{'matched':14s} {text}: {n} of {len(index)} spectra")
                continue
            why = why_nothing(rule, index, reader) or (
                "each key matches some spectrum, but no spectrum matches them all")
            problems.append(f"{text} matches no spectrum: {why}")
    return problems


def _check_peptides(project, entries) -> list[str]:
    """One problem per peptide of ``entries`` (in table order) that does not tokenise with the
    project's registry: file and line (of the peptide table) or file name (entries found by
    scanning), the sequence and the reason, with a reminder of the modX labels."""
    from .chem import ChemistryError, parse_sequence

    inp = project.input
    problems = []
    for n, entry in enumerate(entries):
        try:
            parse_sequence(entry.peptide, project.run.registry)
        except ChemistryError as exc:
            where = (f"{inp.peptides} line {n + 2}" if inp.peptides is not None
                     else f"{entry.file}")
            problems.append(f"{where}: peptide {entry.peptide!r}: {exc} (modified residues "
                            "are modX labels such as acK or pS; unknown ones go in "
                            "[run.registry])")
    return problems


def _instrument_text(record: dict, picker) -> str:
    """One line on a scan file's instrument and recorded resolving power, for ``hxex check``."""
    parts = [f"instrument {record['model']}" if record.get("model") else "instrument unknown"]
    powers = record.get("resolving_power") or {}
    if powers:
        at = record.get("resolving_power_mz")
        at_text = f" (quoted at m/z {at:g})" if at is not None and np.isfinite(at) else ""
        parts.append("resolving power " + ", ".join(
            f"{v if isinstance(v, str) else f'{v:g}'} at MS{level}" for level, v in powers.items())
            + at_text)
        if picker.resolving_power is None:
            parts.append('[run.picker] resolving_power = "recorded" would use it')
    return "; ".join(parts)


def _check_scan_files(project, index_rows: list | None = None) -> list[str]:
    """List each run's scans per target; the problems found. ``index_rows`` (a list) receives
    the spectra the read would make (:func:`hxex.io.mzml.scan_index_rows`), for the exclusion
    counts."""
    from .config import fragment_parent_problems, scan_targets
    from .io import read_peptide_table
    from .io.entries import SCAN_REQUIRED
    from .io.mzml import (
        MzmlFile,
        _target_label,
        instrument_record,
        scan_index_rows,
        target_applies,
        target_scans,
    )

    inp = project.input
    if not inp.peptides.is_file():
        return [f"input.peptides {inp.peptides}: no such file"]
    if inp.format == "thermo" and inp.thermo_dir is not None and not inp.thermo_dir.is_dir():
        return [f"input.thermo_dir {inp.thermo_dir}: no such directory"]
    entries = read_peptide_table(inp.peptides, required=SCAN_REQUIRED)
    problems = _check_peptides(project, entries)
    if problems:
        return problems
    problems = fragment_parent_problems(project, entries)
    if problems:
        return problems
    targets = scan_targets(project)
    _say(f"{'targets':14s} {len(targets)} from {inp.peptides}")
    for run in inp.runs:
        path = Path(run.path)
        path = path if path.is_absolute() else inp.data_root / path
        if not path.is_file():
            problems.append(f"run {path}: no such file")
            continue
        if inp.format == "thermo":
            from .io import ThermoFile

            opened = ThermoFile(path, thermo_dir=inp.thermo_dir)
        else:
            opened = MzmlFile(path)
        with opened as f:
            table = f.scans()
            mine = [t for t in targets if target_applies(t, run)]  # its sample's, and blank
            counts = [len(target_scans(table, run, t, inp.scans)) for t in mine]
            instrument = instrument_record(f)
            if index_rows is not None and _has_rules(project):
                index_rows += scan_index_rows(table, path.name, run, targets,
                                              project=project.project, config=inp.scans)
        none = [_target_label(t) for t, n in zip(mine, counts, strict=True) if n == 0]
        _say(f"run {path.name}: sample {run.sample}, rep {run.rep}, {_run_time_text(run)}, "
             f"{len(table)} scans; {len(mine) - len(none)} of {len(mine)} targets have "
             "scans")
        if instrument is not None:
            _say("  " + _instrument_text(instrument, project.run.picker))
        for label in none[:10]:
            _say(f"  no scans: {label}")
        if len(none) > 10:
            _say(f"  ... and {len(none) - 10} more")
        if not mine:
            problems.append(f"run {path.name}: no peptide of the table is for sample "
                            f"{run.sample} (a peptide is read from the runs of its sample, or "
                            "from every run when its sample is empty)")
        elif len(none) == len(mine):
            problems.append(f"run {path.name}: no target has scans")
    return problems


_ROLE_NAMES = {"undeut": "undeuterated", "fulldeut": "fully deuterated"}


def _run_time_text(run) -> str:
    """What a scan-format run's spectra are: its time, its control role, both, or scan times."""
    from .io import format_time

    parts = [] if run.time is None else [f"time {format_time(run.time)}"]
    if run.role:
        parts.append(f"{_ROLE_NAMES[run.role]} control")
    return ", ".join(parts) if parts else "scan times"


def _folder_control_files(inp, entries) -> dict | None:
    """The control files of a SpecExport or pooled folder by (sample, role), counted from the
    file names (the data are not read); None for HX-Express sheets, whose labels are inside."""
    if inp.format == "hxexpress":
        return None
    from .io.read import _input_path
    from .io.specexport import parse_specexport_filename, parse_specexport_token

    counts: Counter = Counter()
    for entry in entries:
        folder = inp.data_dir / _input_path(inp.format, entry)
        if not folder.is_dir():
            continue
        for path in folder.glob("*.csv"):
            parsed = parse_specexport_filename(path.name)
            if parsed is not None and parsed[2] == entry.charge:
                role = parse_specexport_token(parsed[0])[1]
                if role:
                    counts[(entry.sample, role)] += 1
    return counts


def _control_file_text(roles: dict | None, sample: str) -> str:
    """``, N undeuterated and M fully deuterated control files`` for a sample's line."""
    if roles is None:
        return ""
    n_un, n_fd = roles.get((sample, "undeut"), 0), roles.get((sample, "fulldeut"), 0)
    return (f" entries, {n_un} undeuterated and {n_fd} fully deuterated control files"
            if n_un or n_fd else " entries, no control files")


def cmd_run(args) -> int:
    import dataclasses

    from .config import check_table_choices, project_record, read_input
    from .hooks import ProgressLog
    from .model import Backend
    from .review_table import table_choices
    from .run import run

    project = _project(args)
    out = project.output
    bundle = Path(args.bundle) if args.bundle else out.bundle
    if bundle is None:
        log.error("no bundle: set output.bundle or give --bundle")
        return EXIT_ERROR
    stored = project.input.spectra
    if stored is not None and bundle.resolve() == stored.resolve():
        log.error("%s holds the spectra this run takes: write the results to another bundle",
                  bundle)
        return EXIT_ERROR
    if stored is not None:
        from .config import stored_spectra_source

        stored_spectra_source(stored)  # a path that cannot be read: refused before anything
    overwrite = args.overwrite or out.overwrite
    keep_previous = args.keep_previous or out.keep_previous
    if bundle.exists() and not (overwrite or keep_previous):
        log.error("%s exists: --overwrite replaces it (and its tables, carrying its review "
                  "over), --keep-previous moves it to the next generation first "
                  "(%s.N), or write to another output.bundle (the bundle and the folders "
                  "named after it are hxex's)", bundle, bundle.name)
        return EXIT_ERROR
    from .io import prepare_output

    tables = _beside(bundle, out.tables, "_tables") if out.tables is not False else None
    # the tables folder follows the bundle's rule (refused before anything is fitted); a
    # bundle kept as its previous generation takes its own tables folder along
    moves_along = (keep_previous and bundle.exists() and tables is not None
                   and tables == bundle.with_name(bundle.name + "_tables"))
    if tables is not None and not moves_along:
        prepare_output(tables, overwrite=overwrite, keep_previous=keep_previous, move=False,
                       what="tables of earlier results",
                       elsewhere="another output.bundle (or output.tables)")
    config = project.run
    chosen_k = table_choices(project.review)  # before anything is read
    check_table_choices(chosen_k, config.fit.k_values)
    if args.device:
        b = config.fit.backend
        config = dataclasses.replace(config, fit=dataclasses.replace(
            config.fit, backend=Backend(args.device, b.dtype)))
    stop = _Stop()
    threads = args.threads if args.threads is not None else project.threads
    with stop.installed(), _threads(threads):
        if stored is None:
            log.info("reading the %s input of %s", project.input.format, project.project)
        spectra = read_input(project, progress=ProgressLog(name="read",
                                                           interval=args.progress_every),
                             cancel=stop)
        log.info("%d spectra read", len(spectra))
        result = run(
            spectra, config, out=bundle, project=project.project, notes=out.notes,
            overwrite=overwrite, keep_previous=keep_previous, keep_spectra=out.keep_spectra,
            exclude=project.input.exclude, view_exclude=_view_exclude(project),
            chosen_k=chosen_k,
            project_file=project_record(dataclasses.replace(
                project, threads=threads, output=dataclasses.replace(
                    out, bundle=bundle, overwrite=overwrite, keep_previous=keep_previous))),
            progress=ProgressLog(interval=args.progress_every), cancel=stop,
        )
    if tables is not None:
        from .io.bundle import bundle_manifest
        from .report import Results, write_tables
        from .review_table import apply_view

        if keep_previous:  # not moved along with a bundle: kept on its own
            prepare_output(tables, keep_previous=True)
        write_tables(apply_view(Results.from_run(result), project.review), tables,
                     created=(bundle_manifest(bundle) or {}).get("created", ""))
    _summary(result.fit, bundle, tables=tables)
    _warn_stale_outputs(bundle, args.project_file)
    return 0


def _warn_stale_outputs(bundle: Path, project_file) -> None:
    """Warn about the bundle's figures and analysis folders made from other results than the
    ones there now (after a run, however the earlier bundle went, or a resume), naming the
    command that brings each up to date."""
    from .io import holds_output, made_from
    from .io.bundle import analysis_states, bundle_manifest

    created = (bundle_manifest(bundle) or {}).get("created", "")
    figures = bundle.with_name(bundle.name + "_figures")
    if holds_output(figures) and (made_from(figures) or {}).get("") != created:
        log.warning("%s holds figures of other results than the ones in %s now: 'hxex report "
                    "%s --figures --overwrite' draws them again", figures, bundle, project_file)
    for folder, state in analysis_states(bundle):
        if state is not None:
            log.warning("%s holds an analysis of other results than the ones in %s now: "
                        "'hxex analyse %s --overwrite' brings it up to date", folder, bundle,
                        project_file)


def _view_exclude(project) -> tuple:
    """The project's [[review.exclude]] rules (none without a [review] table)."""
    return tuple(project.review.exclude) if project.review is not None else ()


def _summary(fit, bundle: Path, *, tables: Path | None = None) -> None:
    ks = Counter(int(k) for k in fit.selected_K)
    _say(f"{len(fit)} measurements; selected K: "
         + ", ".join(f"{k}: {n}" for k, n in sorted(ks.items())))
    _say(f"bundle: {bundle}")
    if tables is not None:
        _say(f"tables: {tables}")


def cmd_resume(args) -> int:
    from .hooks import ProgressLog
    from .io.bundle import bundle_manifest
    from .run import NothingToResume, resume_bootstrap

    bundle, extra = Path(args.bundle), {}
    tables_arg, threads = args.tables, args.threads
    review = None
    if bundle.suffix == ".toml" and bundle.is_file():  # a project file: add or redo a bootstrap
        from .config import check_table_choices, load_project
        from .review_table import table_choices

        project = load_project(bundle, analysis_only=True)
        if project.output.bundle is None:
            log.error("%s names no [output] bundle", bundle)
            return EXIT_ERROR
        chosen = table_choices(project.review)
        check_table_choices(chosen, project.run.fit.k_values)
        if project.run.bootstrap is None and not chosen:
            log.error("%s has no [run.bootstrap] table: add one with the bootstrap's settings",
                      bundle)
            return EXIT_ERROR
        bundle, review = project.output.bundle, project.review
        held = ((bundle_manifest(bundle) or {}).get("configs") or {}).get("bootstrap")
        # the bootstrap there is kept when it has the project's settings: only what the
        # review's chosen K lacks is added. Compared as the manifest stores them (JSON:
        # a tuple such as ks = (1, 2) comes back a list, and never equalled the project's)
        # A key the stored settings lack (one added to hxex after that bootstrap) takes its
        # default, so only a real change of settings redoes the bootstrap
        from .bootstrap import BootstrapConfig

        if isinstance(held, dict):
            defaults = json.loads(json.dumps(BootstrapConfig().to_dict()))
            if filled := sorted(set(defaults) - set(held)):
                log.info("%s: the stored bootstrap settings lack %s; taken as the defaults",
                         bundle, ", ".join(filled))
                held = {**{k: defaults[k] for k in filled}, **held}
        # without [run.bootstrap] (a review table's K to fit) the bundle's bootstrap is kept
        redo = (project.run.bootstrap is not None
                and held != json.loads(json.dumps(project.run.bootstrap.to_dict())))
        extra = {"bootstrap": project.run.bootstrap, "add": redo,
                 "view_exclude": _view_exclude(project), "chosen_k": chosen}
        out = project.output
        tables_arg = tables_arg if tables_arg is not None else (out.tables or None)
        threads = threads if threads is not None else project.threads
    stop = _Stop()
    with stop.installed(), _threads(threads):
        try:
            result = resume_bootstrap(
                bundle, device=args.device, progress=ProgressLog(interval=args.progress_every),
                cancel=stop, **extra)
        except NothingToResume:
            if not extra:  # the bundle alone: say how to add or redo a bootstrap
                raise
            _say(f"{bundle}: the bootstrap has the project's settings and every K chosen in "
                 "the review has a bootstrap: nothing to add")
            return 0
    tables = _beside(bundle, tables_arg, "_tables")
    if tables is not None:
        from .report import Results, write_tables
        from .review_table import apply_view

        write_tables(apply_view(Results.from_run(result), review), tables,
                     created=(bundle_manifest(bundle) or {}).get("created", ""))
        log.info("tables in %s", tables)
    _summary(result.fit, bundle)
    _warn_stale_outputs(Path(bundle), args.bundle)
    return 0


def cmd_report(args) -> int:
    import dataclasses

    from .hooks import ProgressLog
    from .io import open_bundle, prepare_output
    from .report import Results, write_tables
    from .report.figures import PlotConfig, write_figures

    bundle, review, plots = Path(args.bundle), None, PlotConfig()
    overwrite, keep_previous = args.overwrite, args.keep_previous
    own_tables: Path | bool = True  # the tables folder when neither --tables nor --figures
    if bundle.suffix == ".toml" and bundle.is_file():  # a project file: its bundle, rules, figures
        from .config import load_project

        project = load_project(bundle, analysis_only=True)
        if project.output.bundle is None:
            log.error("%s names no [output] bundle to report on", bundle)
            return EXIT_ERROR
        bundle, review = project.output.bundle, project.review
        plots = project.figures or plots
        overwrite = overwrite or project.output.overwrite
        keep_previous = keep_previous or project.output.keep_previous
        own_tables = project.output.tables
    tables = _beside(bundle, args.tables, "_tables")
    figures = _beside(bundle, args.figures, "_figures")
    if tables is None and figures is None:
        if own_tables is False:
            _say(f"nothing to write: {args.bundle} has [output] tables = false (--tables writes "
                 "the tables all the same, --figures draws the figures)")
            return 0
        tables = _beside(bundle, own_tables, "_tables")
    outputs = [(folder, what, option) for folder, what, option in (
        (tables, "tables", "--tables DIR"), (figures, "figures", "--figures DIR"))
        if folder is not None]
    for folder, what, option in outputs:  # the bundle's rule, before anything is opened
        prepare_output(folder, overwrite=overwrite, keep_previous=keep_previous, move=False,
                       what=what, elsewhere=f"another folder with {option}")
    stop = _Stop()
    with open_bundle(bundle) as b, stop.installed(), _threads(args.threads):
        if b.pending:
            log.warning("%s is waiting for its %s ('hxex resume' adds it)", bundle,
                        ", ".join(b.pending))
        from .review_table import apply_view, missing_bootstraps

        results = apply_view(Results.from_bundle(b), review)  # a K not fitted: ValueError
        missing = missing_bootstraps(results)
        if missing:
            log.error("%d spectra are shown at a K chosen in the review that has no bootstrap "
                      "(for example %s), so their errors would not compare with the others': "
                      "hxex resume %s bootstraps them (only those), then report again",
                      len(missing), ", ".join(f"{s} at K = {k}"
                                              for s, k in list(missing.items())[:2]),
                      args.bundle)
            return EXIT_ERROR
        if keep_previous:  # moved only now that nothing more can refuse the report
            for folder, _, _ in outputs:
                prepare_output(folder, keep_previous=True)
        if tables is not None:
            written = write_tables(results, tables, created=b.manifest.get("created", ""))
            _say(f"{len(written)} table(s) in {tables}")
        if figures is not None:
            if args.k is not None:
                plots = dataclasses.replace(plots, k=args.k)
            if args.series:
                plots = dataclasses.replace(plots, series=args.series)
            if args.fig_format:
                plots = dataclasses.replace(plots, format=args.fig_format)
            written = write_figures(results, figures, plots,
                                    progress=ProgressLog(name="figures",
                                                         interval=args.progress_every),
                                    cancel=stop)
            _say(f"{len(written)} figure file(s) in {figures}")
    return 0


def _peak_measure_text(measure) -> str:
    """What the bins took (heights or areas) and why, for ``hxex info``."""
    if measure is None:
        return "heights (written before hxex chose between heights and areas)"
    return measure.describe()


def _noise_curve_text(fit) -> str:
    """The per-bin noise of the fit, with every sample's counting scale (``hxex info``)."""
    curve, noise = fit.noise_curve, fit.config.noise
    if curve is None:
        if noise.model == "poisson":
            text = "poisson"
        elif noise.sd is not None:
            text = "one sd per spectrum, set directly"
        else:
            text = f"one sd per spectrum, {noise.percent:g} % of its reference's tallest peak"
        if noise.estimate == "replicates":
            text += " (estimated from replicates where a series has enough)"
        elif noise.estimate == "controls":
            text += " (too few undeuterated controls for the per-bin noise)"
        return text
    onset = ("" if not np.isfinite(curve.onset)
             else f", controls with tops of {curve.onset:,.0f} or more left out")
    text = (f"per bin, z x b1 x I + ({100 * curve.model_error:g} % of I)^2 with b1 = "
            f"{curve.b1:.3g} +- {curve.b1_se:.2g} (from {curve.n_controls} undeuterated "
            f"controls of {curve.n_series} series{onset})")
    if curve.by_sample:
        odd = {s.sample for s in curve.disagreeing_samples()}
        text += "; b1 by sample: " + ", ".join(
            (f"{s.sample} {s.b1:.3g} +- {s.b1_se:.2g}{' (differs)' if s.sample in odd else ''}"
             if np.isfinite(s.b1) else f"{s.sample} not measured ({s.n_series} series)")
            for s in curve.by_sample)
    return text


def _noise_reference_text(meas) -> str:
    """What the series' noise levels come from, counted (``hxex info``)."""
    from .noise import noise_reference_kinds

    kinds = Counter(noise_reference_kinds(meas.data_id, meas.references, meas.role).values())
    parts = [f"{kinds[k]} series {name}" for k, name in (
        ("undeuterated control", "from their undeuterated controls"),
        ("run's scans", "from the run's own scans"),
        ("own spectrum", "from their own spectra (no undeuterated control fitted: see series.csv "
                         "noise_reference)")) if kinds[k]]
    return "; ".join(parts) or "none"


def _role_counts_text(fit) -> str:
    """``N undeuterated (S series), M fully deuterated (T series)`` from the fit's roles."""
    from .io import format_time

    if fit.role is None:
        return "none"
    role = np.asarray(fit.role, dtype=str)
    parts = []
    for key, name in _ROLE_NAMES.items():
        sel = role == key
        if sel.any():
            series = len(set(fit.data_id[sel].tolist()))
            timed = np.asarray(fit.time, dtype=float)[sel]
            timed = timed[np.isfinite(timed)]
            also = (f", at {', '.join(format_time(t) for t in sorted(set(timed.tolist())))}"
                    if len(timed) else "")
            parts.append(f"{int(sel.sum())} {name} ({series} series{also})")
    return "; ".join(parts) if parts else "none"


def _promotion_text(controls: dict | None) -> str:
    """``; undeut_at = 5 min`` and the like from a run's recorded control settings."""
    from .io import format_time

    if not controls:
        return ""
    parts = [f"{name} = {format_time(float(controls[name]))}"
             for name in ("undeut_at", "fulldeut_at") if controls.get(name) is not None]
    parts += [f"{name} = false" for name in ("undeut_marked", "fulldeut_marked")
              if controls.get(name) is False]
    return "; " + ", ".join(parts) if parts else ""


def cmd_info(args) -> int:
    from .io import generations, open_bundle

    with open_bundle(args.bundle) as b:
        m = b.manifest
        fit = b.fit()
        ks = Counter(int(k) for k in fit.selected_K)
        source = b.spectra_source() or {}
        review = b.review()
        run_config = m.get("configs", {}).get("run") or {}
        rows = [
            ("bundle", str(b.path)),
            ("project", b.project or "(none)"),
            ("created", m.get("created", "")),
            ("hxex", m.get("versions", {}).get("hxex", "")),
            ("input", source.get("reader", source.get("format", "(not recorded)"))),
            ("bin values", _peak_measure_text(b.peak_measure())),
            ("measurements", str(len(fit))),
            ("series", str(len(set(fit.data_id.tolist())))),
            ("K tried", ", ".join(str(k) for k in fit.config.k_values)
             + (" (and K = 1 for the controls, for the back-exchange correction)"
                if 1 in fit.k_values and 1 not in fit.config.k_values else "")),
            ("selected K", ", ".join(f"{k}: {n}" for k, n in sorted(ks.items()))),
            ("bootstrap", "yes" if b.bootstrap() is not None else "no"),
            ("controls", ("yes" if b.has("controls") else "no")
             + _promotion_text(run_config.get("controls"))),
            ("roles", _role_counts_text(fit)),
            ("noise reference", _noise_reference_text(b.measurements())),
            ("noise", _noise_curve_text(fit)),
            ("times", "seconds" if source.get("time_scale", "seconds") == "seconds"
             else "order: the times only order the points"),
            ("quality flags", "yes" if b.has("quality") else "no"),
            ("raw spectra", "stored" if b.has("spectra") else "not stored"),
            ("pending", ", ".join(b.pending) or "nothing (complete)"),
            ("review", f"{len(review)} entries" + (f" ({review.n_stale} stale)"
                                                   if review.n_stale else "")
             if len(review) else "none"),
        ]
        if m.get("notes"):
            rows.append(("notes", m["notes"]))
        if m.get("excluded"):
            ex = m["excluded"]
            rows.append(("excluded", f"{len(ex.get('spectrum_ids', []))} spectra left out "
                                     "before fitting ([[input.exclude]], stored, not fitted)"))
            for rule in ex.get("rules", []):
                matched = rule.get("matched", 0)
                text = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in rule.items()
                                 if k not in ("note", "matched"))
                rows.append(("", f"  {text}: {matched}"
                             + (f" ({rule['note']})" if rule.get("note") else "")))
        rows += _provenance_rows(m.get("provenance"))
        earlier = generations(b.path)
        if earlier:
            rows.append(("generations", ", ".join(p.name for _, p in earlier)))
        rows += _output_rows(b.path)
        rows += _analysis_rows(b.path)
        for key, value in rows:
            _say(f"{key:14s} {value}")
        if args.verify:
            b.verify()
            _say(f"{'checksums':14s} ok")
    return 0


def _output_rows(bundle: Path) -> list[tuple[str, str]]:
    """The ``hxex info`` lines of the bundle's tables and figures folders: whether each was made
    from the results there now (none without such folders)."""
    from .io import holds_output, made_from
    from .io.bundle import bundle_manifest

    created = (bundle_manifest(bundle) or {}).get("created", "")
    rows = []
    for suffix, command in (("_tables", "report"), ("_figures", "report --figures")):
        folder = bundle.with_name(bundle.name + suffix)
        if not holds_output(folder):
            continue
        stamp = (made_from(folder) or {}).get("")
        text = (f"{folder.name}: up to date" if stamp == created else
                f"{folder.name}: not known (written before hxex recorded it)" if not stamp else
                f"{folder.name}: made from earlier results (created {stamp}); hxex {command} "
                "--overwrite writes it again")
        rows.append(("" if rows else "outputs", text))
    return rows


def _analysis_rows(bundle: Path) -> list[tuple[str, str]]:
    """The ``hxex info`` lines of the bundle's analysis folders: whether each was made from the
    results there now (none without analysis folders)."""
    from .io.bundle import analysis_states

    rows = []
    for folder, state in analysis_states(bundle):
        text = (f"{folder.name}: up to date" if state is None else
                f"{folder.name}: not known (written before hxex recorded it)" if not state else
                f"{folder.name}: made from earlier results (created {state}); run hxex "
                "analyse again")
        rows.append(("" if rows else "analyses", text))
    return rows


def _stale_analyses(bundle: Path, written: Path) -> None:
    """Warn about the bundle's other analysis folders made from earlier results (``hxex
    analyse`` has just brought ``written`` up to date)."""
    from .io.bundle import analysis_states

    for folder, state in analysis_states(bundle):
        if state and folder.resolve() != written.resolve():
            log.warning("%s was made from earlier results (created %s): run hxex analyse for "
                        "it again", folder, state)


def _provenance_rows(prov) -> list[tuple[str, str]]:
    """The ``hxex info`` lines of a manifest's provenance block (none without one)."""
    if not prov:
        return []
    created = prov.get("replaced_created")
    where = prov.get("moved_to")
    rows = [("replaced", f"{prov.get('replaced')} (created {created or 'unknown'}) on "
                         f"{prov.get('at', '')}" + (f", kept as {where}" if where else ""))]
    changed = prov.get("settings_changed") or []
    rows.append(("changed", "nothing" if not changed else
                 f"{len(changed)} setting(s): " + ", ".join(changed[:8])
                 + (", ..." if len(changed) > 8 else "")))
    review = prov.get("review") or {}
    if review.get("carried"):
        rows.append(("carried", f"{review['carried']} review entries, "
                                f"{review.get('stale', 0)} stale"))
    return rows


def _is_bundle(path: Path) -> bool:
    return path.suffix == ".zip" or (path.is_dir() and (path / "manifest.json").is_file())


def _run_source(text: str) -> tuple[str, Path]:
    """``(name, bundle)`` of a ``NAME=BUNDLE`` or ``BUNDLE`` argument."""
    name, sep, path = text.partition("=")
    if sep and name and not Path(text).exists():
        return name, Path(path)
    path = Path(text)
    return (path.name if path.is_dir() else path.stem), path


def cmd_analyse(args) -> int:
    import contextlib
    import dataclasses

    from .analysis import AnalysisConfig, RunSet
    from .hooks import ProgressLog
    from .io import open_bundle
    from .report import Results, write_analysis

    threads = args.threads
    record_paths = "path"  # the analysis settings file records its paths as the input does
    review = None  # the project's [review] decisions
    overwrite, keep_previous = args.overwrite, args.keep_previous
    first = Path(_run_source(args.source[0])[1])
    if len(args.source) > 1 or _is_bundle(first):
        runs = dict(_run_source(s) for s in args.source)
        if len(runs) < len(args.source):
            log.error("two runs have the same name: give NAME=BUNDLE")
            return EXIT_ERROR
        if len(args.source) == 1 and _is_bundle(Path(args.source[0])):
            runs = {"": first}  # one bundle, not named: its samples keep their names
        config, base = AnalysisConfig(), Path(".")
        if args.bundle:
            if len(runs) > 1:
                log.error("--bundle takes the place of a single bundle")
                return EXIT_ERROR
            runs = {"": Path(args.bundle)}
    else:
        from .config import load_project

        project = load_project(first, analysis_only=True)
        config = project.analysis if project.analysis is not None else AnalysisConfig()
        base = first.resolve().parent
        threads = threads if threads is not None else project.threads
        if project.input is not None:
            record_paths = project.input.record_paths
        review = project.review
        overwrite = overwrite or project.output.overwrite
        keep_previous = keep_previous or project.output.keep_previous
        if config.runs and not args.bundle:
            runs = {name: Path(path) for name, path in config.runs.items()}
        else:
            bundle = Path(args.bundle) if args.bundle else project.output.bundle
            if bundle is None:
                log.error("no bundle: set output.bundle, [analysis.runs] or give --bundle")
                return EXIT_ERROR
            runs = {"": bundle}
    changes = {}
    if args.reference:
        changes["reference"] = args.reference
    if args.fig_format:
        changes["format"] = args.fig_format
    if changes:
        config = dataclasses.replace(config, **changes)
    several = len(runs) > 1 or "" not in runs
    lead = next(iter(runs.values()))
    from .io.bundle import analysis_folder_name

    # the default folder is named after the reference, so analyses against different
    # references never write into one folder
    out = Path(args.out) if args.out else Path(config.out) if config.out else lead.with_name(
        analysis_folder_name("runs" if several else lead.name, config.reference))
    from .io import prepare_output

    prepare_output(out, overwrite=overwrite, keep_previous=keep_previous, what="analysis",
                   elsewhere="another folder with --out DIR (or [analysis] out)", move=False)
    stop = _Stop()
    with contextlib.ExitStack() as stack:
        stack.enter_context(stop.installed())
        stack.enter_context(_threads(threads))
        opened = {}
        for name, path in runs.items():
            b = stack.enter_context(open_bundle(path))
            if b.pending:
                log.warning("%s is waiting for its %s; the analysis uses what it holds",
                            path, ", ".join(b.pending))
            opened[name] = Results.from_bundle(b)
        results = RunSet(opened, merge=config.merge) if several else opened[""]
        written = write_analysis(results, config, out, base=base, record_paths=record_paths,
                                 review=review,
                                 progress=ProgressLog(name="analysis",
                                                      interval=args.progress_every),
                                 cancel=stop,
                                 # moved only once nothing more can refuse the analysis
                                 before_writing=(lambda: prepare_output(out, keep_previous=True))
                                 if keep_previous else None)
    if not several:
        _stale_analyses(lead, out)
    tables = sum(1 for p in written if p.suffix == ".csv")
    figures = sum(1 for p in written if p.suffix == f".{config.format}")
    _say(f"{tables} table(s) and {figures} figure(s) in {out}, with the settings used")
    return 0


COMMANDS = {"init": cmd_init, "check": cmd_check, "run": cmd_run, "resume": cmd_resume,
            "report": cmd_report, "info": cmd_info, "analyse": cmd_analyse,
            "analyze": cmd_analyse}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    with _logging_to_stderr(args):
        return _dispatch(args)


def _dispatch(args) -> int:
    from .hooks import Cancelled
    from .io import OutputError, ReadError
    from .io.bundle import BundleError

    try:
        return COMMANDS[args.command](args)
    except Cancelled:
        log.warning("stopped")
        return EXIT_STOPPED
    except KeyboardInterrupt:
        log.warning("aborted")
        return EXIT_STOPPED
    except (ReadError, BundleError, OutputError, ValueError, OSError, RuntimeError) as exc:
        if args.verbose:
            log.exception("failed")
        else:
            log.error("%s", exc)
        return EXIT_ERROR
