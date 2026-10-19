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
    p.add_argument("--force", action="store_true", help="replace an existing file")

    p = sub.add_parser("check", help="check a project file and show what would be read")
    p.add_argument("project_file", metavar="PROJECT")
    _spectra_option(p)

    p = sub.add_parser("run", help="run a project: read, pick, fit, bootstrap, write")
    p.add_argument("project_file", metavar="PROJECT")
    p.add_argument("--bundle", help="the results bundle (instead of output.bundle)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing bundle (and its tables and figures), carrying "
                        "its review over")
    p.add_argument("--keep-previous", action="store_true",
                   help="move an existing bundle (with its tables and figures) to the next "
                        "generation, NAME.N, before writing")
    p.add_argument("--device", help="cpu, cuda, cuda:1, ... (instead of run.fit.backend)")
    _spectra_option(p)
    _threads_option(p)

    p = sub.add_parser("resume", help="add the bootstrap to a bundle that is waiting for it")
    p.add_argument("bundle")
    p.add_argument("--device", help="cpu, cuda, cuda:1, ... (instead of the run's device)")
    _threads_option(p)
    p.add_argument("--tables", nargs="?", const=True, default=None, metavar="DIR",
                   help="also write the tables (default: beside the bundle)")
    p.add_argument("--figures", nargs="?", const=True, default=None, metavar="DIR",
                   help="also write the figures (default: beside the bundle)")

    p = sub.add_parser("report", help="write tables and figures from a saved bundle")
    p.add_argument("bundle")
    _threads_option(p)
    p.add_argument("--tables", nargs="?", const=True, default=None, metavar="DIR",
                   help="write the tables (default: beside the bundle)")
    p.add_argument("--figures", nargs="?", const=True, default=None, metavar="DIR",
                   help="write the figures (default: beside the bundle)")
    p.add_argument("--series", choices=("flagged", "all"),
                   help="which series to draw (default: the run's plot setting, else flagged)")
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
                                 "bundle's name + _analysis)")
    p.add_argument("--reference", help="the reference sample (instead of [analysis] reference; "
                                       "RUN:sample when the sample is in several runs)")
    p.add_argument("--format", dest="fig_format", choices=("pdf", "png", "svg"),
                   help="figure file format")
    _threads_option(p)

    p = sub.add_parser("info", help="what a results bundle holds")
    p.add_argument("bundle")
    p.add_argument("--verify", action="store_true", help="also check every file's checksum")
    return parser


def _spectra_option(p) -> None:
    p.add_argument("--spectra", metavar="PATH",
                   help="take the raw spectra stored in this bundle (or spectra file) instead "
                        "of reading the input (as input.spectra)")


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
             "bundle and the folders named after it (NAME_tables, NAME_figures, NAME.N) are "
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
    else:
        problems += _check_scan_files(project)
    if inp is not None and inp.retention_times is not None and not inp.retention_times.is_file():
        problems.append(f"input.retention_times {inp.retention_times}: no such file")
    out = project.output
    if (out.bundle is not None and out.bundle.exists() and not out.overwrite
            and not out.keep_previous):
        problems.append(f"the bundle {out.bundle} exists: 'hxex run' needs --overwrite or "
                        "--keep-previous (output.overwrite or output.keep_previous), or "
                        "another output.bundle")
    if problems:
        for p in problems:
            log.error("%s", p)
        log.warning("%d problem(s) found", len(problems))
        return EXIT_ERROR
    _say("ok")
    return 0


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


def _check_scan_files(project) -> list[str]:
    """List each run's scans per target; the problems found."""
    from .config import scan_targets
    from .io import read_peptide_table
    from .io.mzml import MzmlFile, _target_label, instrument_record, target_scans

    inp = project.input
    if not inp.peptides.is_file():
        return [f"input.peptides {inp.peptides}: no such file"]
    if inp.format == "thermo" and inp.thermo_dir is not None and not inp.thermo_dir.is_dir():
        return [f"input.thermo_dir {inp.thermo_dir}: no such directory"]
    problems = _check_peptides(project, read_peptide_table(inp.peptides))
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
            counts = [len(target_scans(table, run, t, inp.scans)) for t in targets]
            instrument = instrument_record(f)
        none = [_target_label(t) for t, n in zip(targets, counts, strict=True) if n == 0]
        _say(f"run {path.name}: sample {run.sample}, rep {run.rep}, {_run_time_text(run)}, "
             f"{len(table)} scans; {len(targets) - len(none)} of {len(targets)} targets have "
             "scans")
        if instrument is not None:
            _say("  " + _instrument_text(instrument, project.run.picker))
        for label in none[:10]:
            _say(f"  no scans: {label}")
        if len(none) > 10:
            _say(f"  ... and {len(none) - 10} more")
        if len(none) == len(targets):
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

    from .config import read_input
    from .hooks import ProgressLog
    from .model import Backend
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
        log.error("%s exists: --overwrite replaces it (and its tables and figures, carrying "
                  "its review over), --keep-previous moves it to the next generation first "
                  "(%s.N), or write to another output.bundle (the bundle and the folders "
                  "named after it are hxex's)", bundle, bundle.name)
        return EXIT_ERROR
    if keep_previous and bundle.exists():
        for key, value in (("tables", out.tables), ("figures", out.figures)):
            if isinstance(value, Path):
                log.info("output.%s %s is a directory of its own: not moved with the previous "
                         "bundle (its hxex files are replaced)", key, value)
    config = project.run
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
            figures_dir=_beside(bundle, out.figures, "_figures")
            if out.figures is not False else None,
            progress=ProgressLog(interval=args.progress_every), cancel=stop,
        )
    tables = _beside(bundle, out.tables, "_tables") if out.tables is not False else None
    if tables is not None:
        from .report import Results, write_tables

        write_tables(Results.from_run(result), tables)
    _summary(result.fit, bundle, tables=tables,
             figures=_beside(bundle, out.figures, "_figures") if out.figures is not False
             else None)
    return 0


def _summary(fit, bundle: Path, *, tables: Path | None = None,
             figures: Path | None = None) -> None:
    ks = Counter(int(k) for k in fit.selected_K)
    _say(f"{len(fit)} measurements; selected K: "
         + ", ".join(f"{k}: {n}" for k, n in sorted(ks.items())))
    _say(f"bundle: {bundle}")
    if tables is not None:
        _say(f"tables: {tables}")
    if figures is not None:
        _say(f"figures: {figures}")


def cmd_resume(args) -> int:
    from .hooks import ProgressLog
    from .run import resume_bootstrap

    bundle = Path(args.bundle)
    stop = _Stop()
    with stop.installed(), _threads(args.threads):
        result = resume_bootstrap(
            bundle, device=args.device, figures_dir=_beside(bundle, args.figures, "_figures"),
            progress=ProgressLog(interval=args.progress_every), cancel=stop)
    tables = _beside(bundle, args.tables, "_tables")
    if tables is not None:
        from .report import Results, write_tables

        write_tables(Results.from_run(result), tables)
        log.info("tables in %s", tables)
    _summary(result.fit, bundle)
    return 0


def cmd_report(args) -> int:
    import dataclasses

    from .hooks import ProgressLog
    from .io import open_bundle
    from .report import Results, write_tables
    from .report.figures import PlotConfig, write_figures

    bundle = Path(args.bundle)
    tables = _beside(bundle, args.tables, "_tables")
    figures = _beside(bundle, args.figures, "_figures")
    if tables is None and figures is None:
        tables = _beside(bundle, True, "_tables")
    stop = _Stop()
    with open_bundle(bundle) as b, stop.installed(), _threads(args.threads):
        if b.pending:
            log.warning("%s is waiting for its %s ('hxex resume' adds it)", bundle,
                        ", ".join(b.pending))
        results = Results.from_bundle(b)
        if tables is not None:
            written = write_tables(results, tables)
            _say(f"{len(written)} table(s) in {tables}")
        if figures is not None:
            run = b.manifest["configs"].get("run") or {}
            plots = PlotConfig.from_dict(run["plots"]) if run.get("plots") else PlotConfig()
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
            ("measurements", str(len(fit))),
            ("series", str(len(set(fit.data_id.tolist())))),
            ("K tried", ", ".join(str(k) for k in fit.k_values)),
            ("selected K", ", ".join(f"{k}: {n}" for k, n in sorted(ks.items()))),
            ("bootstrap", "yes" if b.bootstrap() is not None else "no"),
            ("controls", ("yes" if b.has("controls") else "no")
             + _promotion_text(run_config.get("controls"))),
            ("roles", _role_counts_text(fit)),
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
        rows += _provenance_rows(m.get("provenance"))
        earlier = generations(b.path)
        if earlier:
            rows.append(("generations", ", ".join(p.name for _, p in earlier)))
        for key, value in rows:
            _say(f"{key:14s} {value}")
        if args.verify:
            b.verify()
            _say(f"{'checksums':14s} ok")
    return 0


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
    out = Path(args.out) if args.out else Path(config.out) if config.out else lead.with_name(
        "runs_analysis" if several else lead.name + "_analysis")
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
        written = write_analysis(results, config, out, base=base,
                                 progress=ProgressLog(name="analysis",
                                                      interval=args.progress_every),
                                 cancel=stop)
    tables = sum(1 for p in written if p.suffix == ".csv")
    _say(f"{tables} table(s) and {len(written) - tables} figure(s) in {out}")
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
