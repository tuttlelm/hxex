# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line entry point (``hxex``).

::

    hxex init [--format F] [-o hxex.toml]   write a project file with every default
    hxex set PROJECT SETTING VALUE          change one setting of the project file
    hxex check PROJECT                      check it and show what would be read
    hxex run PROJECT                        read, pick, fit, bootstrap; bundle, tables, figures
    hxex resume PROJECT|BUNDLE              complete a bundle's bootstrap, add the population test
    hxex report PROJECT|BUNDLE [--figures]  the tables (and the figures) from a saved bundle
    hxex info PROJECT|BUNDLE                what a bundle holds
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
import os
import signal
import sys
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

log = logging.getLogger("hxex.cli")

EXIT_ERROR, EXIT_STOPPED = 1, 130


def build_parser() -> argparse.ArgumentParser:
    from .config import FORMATS

    parser = argparse.ArgumentParser(
        prog="hxex",
        description="Multimodal binomial-mixture analysis of HDX-MS spectra.",
        epilog="Paths given on the command line are relative to the folder you run the command "
               "from; paths in the project file are relative to the project file's folder.",
    )
    parser.add_argument("--version", action=_Version, nargs=0,
                        help="show hxex's version, with the commit it was installed from")
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    noise.add_argument("-v", "--verbose", action="store_true", help="more messages")
    parser.add_argument("--progress-every", type=_seconds, default=30.0, metavar="SECONDS",
                        help="seconds between progress lines (default 30)")
    parser.add_argument("--progress-format", choices=("text", "json"), default="text",
                        help="json: also a line '@hxex-progress {...}' with the stage, the share "
                             "done and the time left, for a program to read (hxex's window "
                             "draws its progress bar from them); text (default): the progress "
                             "lines only")
    parser.add_argument("--stop-from-input", action="store_true",
                        help="for a program running hxex (hxex's window): a line 'stop' on the "
                             "input stops after the current step, as Ctrl-C does, a second "
                             "aborts at once; the input closing stops too")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("init", help="write a project file with every default")
    p.add_argument("--format", choices=FORMATS, default="specexport",
                   help="the input format (default specexport)")
    p.add_argument("--project", default="my_project", help="the project name")
    p.add_argument("-o", "--output", default="hxex.toml",
                   help="the file to write (default hxex.toml; '-' for stdout)")
    p.add_argument("--force", action="store_true",
                   help="replace an existing file (without it, init refuses to write over one)")

    p = sub.add_parser("set", help="change one setting of a project file (no text editor "
                       "needed)")
    p.add_argument("project_file", metavar="hxex.toml",
                   help="the project file (hxex.toml, written by hxex init)")
    p.add_argument("setting", metavar="SETTING",
                   help="the setting as table.name, e.g. run.controls.d2o_fraction (the "
                        "settings page of the guide lists them)")
    p.add_argument("value", nargs="*", metavar="VALUE",
                   help='its new value: a number, true or false, text (no quotes needed: 20 '
                        'min, controls), or a list in quotes ("[1, 2]")')
    p.add_argument("--default", action="store_true",
                   help="switch the setting off again (a # in front), so that it takes its "
                        "default; give no value")

    p = sub.add_parser("check", help="check a project file and show what would be read")
    p.add_argument("project_file", metavar="hxex.toml",
                   help="the project file (hxex.toml, written by hxex init)")
    _spectra_option(p)

    p = sub.add_parser("run", help="run a project: read, pick, fit, bootstrap, write the "
                                   "bundle and tables (figures: hxex report)")
    p.add_argument("project_file", metavar="hxex.toml",
                   help="the project file (hxex.toml, written by hxex init)")
    p.add_argument("--bundle", help="the results bundle (instead of output.bundle; relative "
                                    "to the current folder, not to the project file's)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing bundle (and its tables), carrying its review "
                        "over; also a tables folder left from a bundle that is gone")
    p.add_argument("--keep-previous", action="store_true",
                   help="move an existing bundle (with its tables, figures and analysis "
                        "folders) to the next generation, NAME.N, before writing")
    p.add_argument("--device", help="cpu, cuda, cuda:1, ... (instead of run.fit.backend)")
    _spectra_option(p)
    _threads_option(p)

    p = sub.add_parser("resume", help="complete a bundle's bootstrap (finish a stopped one, add "
                                      "or redo one, or bootstrap the review's chosen K), add "
                                      "the population test, or work the controls out again")
    p.add_argument("bundle", metavar="hxex.toml|BUNDLE",
                   help="the project file (hxex.toml), or a results bundle (a folder, or .zip) "
                        "waiting for its bootstrap. With the project file, its [output] bundle "
                        "gets a bootstrap with its "
                        "[run.bootstrap] settings (redone only if they changed), one at the "
                        "K its [review] table chooses where that is missing, and the population "
                        "test with its [run.population_test] settings (made again only if they "
                        "changed), and its controls are worked out again when its "
                        "[run.controls] changed; the fits kept")
    p.add_argument("--device", help="cpu, cuda, cuda:1, ... (instead of the run's device)")
    _threads_option(p)
    p.add_argument("--tables", nargs="?", const=True, default=None, metavar="DIR",
                   help="where the tables go (default: beside the bundle, or a project "
                        "file's [output] tables)")

    p = sub.add_parser("report", help="write the tables again (with the review applied), and "
                                      "with --figures draw the figures, from the results; fits "
                                      "nothing")
    p.add_argument("bundle", metavar="hxex.toml|BUNDLE",
                   help="the project file (hxex.toml): its [output] bundle, with its [review] "
                        "decisions applied and its [figures] settings; or a results bundle (a "
                        "folder, or .zip)")
    _threads_option(p)
    p.add_argument("--tables", nargs="?", const=True, default=None, metavar="DIR",
                   help="where the tables go (default: beside the bundle, NAME_tables, or the "
                        "project file's [output] tables); they are always written, and with "
                        "[output] tables = false only when --tables is given")
    p.add_argument("--figures", nargs="?", const=True, default=None, metavar="DIR",
                   help="also draw the figures (default: beside the bundle, NAME_figures)")
    _output_options(p, "the tables and figures folders")
    p.add_argument("--series", choices=("flagged", "all"),
                   help="which series to draw (default: [figures] series, else flagged)")
    p.add_argument("--k", type=int, metavar="K",
                   help="draw every series at this number of populations (default: [figures] "
                        "k, else each spectrum's own)")
    p.add_argument("--format", dest="fig_format", choices=("pdf", "png", "svg"),
                   help="figure file format")

    p = sub.add_parser("analyse", aliases=["analyze"],
                       help="the post-fitting analysis of a bundle: tables and figures")
    p.add_argument("source", metavar="hxex.toml|BUNDLE", nargs="+",
                   help="the project file, hxex.toml (its [analysis] settings, and "
                        "[analysis.runs] or output.bundle; a file with only [analysis] is "
                        "enough), a bundle, or several bundles to compare as separate runs "
                        "(NAME=BUNDLE, or BUNDLE named after its file; the tables get a run "
                        "column)")
    p.add_argument("--bundle", help="the bundle (instead of the project's output.bundle; "
                                    "relative to the current folder, not to the project file's)")
    p.add_argument("--out", help="the output directory (default: [analysis] out, else the "
                                 "bundle's name + _analysis + _REFERENCE when there is a "
                                 "reference, e.g. results_analysis_WT)")
    _output_options(p, "the analysis folder")
    p.add_argument("--reference", help="the reference sample (instead of [analysis] reference; "
                                       "RUN:sample when the sample is in several runs); "
                                       "several, separated by commas, or all (every sample): "
                                       "a folder each")
    p.add_argument("--format", dest="fig_format", choices=("pdf", "png", "svg"),
                   help="figure file format")
    _threads_option(p)

    p = sub.add_parser("info", help="what the results hold")
    p.add_argument("bundle", metavar="hxex.toml|BUNDLE",
                   help="the project file (hxex.toml): its [output] bundle, and what its "
                        "[review] table decides; or a results bundle (a folder, or .zip)")
    p.add_argument("--verify", action="store_true", help="also check every file's checksum")

    p = sub.add_parser("figures", help="which figures report --figures and analyse would draw "
                                       "with the project's settings, and why the others not")
    p.add_argument("project", metavar="hxex.toml",
                   help="the project file; its results are read when they exist (counts, and "
                        "what only the results can tell)")
    p.add_argument("--settings-only", action="store_true",
                   help="judge by the settings alone, without reading the results")

    p = sub.add_parser("gui", help="open hxex's window (needs the gui extra: pip install "
                                   "\"hxex[gui]\")")
    p.add_argument("path", nargs="?", metavar="PATH",
                   help="what to open: a GUI session file (.hxexgui), a project file "
                        "(hxex.toml, with its results), or a results bundle")
    return parser


class _Version(argparse.Action):
    """``--version``: the version and the commit (asked of git only when wanted)."""

    def __call__(self, parser, namespace, values, option_string=None):
        from . import version_text

        sys.stdout.write(version_text() + "\n")
        parser.exit()


def _spectra_option(p) -> None:
    p.add_argument("--spectra", metavar="PATH",
                   help="take the raw spectra stored in this bundle (or spectra file) instead "
                        "of reading the input (as input.spectra)")


def _output_options(p, folder: str) -> None:
    """--overwrite and --keep-previous for a command that writes an output folder."""
    holds = "they hold" if folder.endswith("folders") else "it holds"
    p.add_argument("--overwrite", action="store_true",
                   help=f"replace hxex's files in {folder} when {holds} earlier output "
                        "(or [output] overwrite = true)")
    p.add_argument("--keep-previous", action="store_true",
                   help=f"move {folder} to NAME.N first when {holds} earlier output (or "
                        "[output] keep_previous = true)")


def _threads_option(p) -> None:
    p.add_argument("--threads", type=_positive_int, metavar="N",
                   help="CPU threads (default: the project file's threads, else PyTorch's)")


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a whole number of at least 1, not "
                                         f"'{text}'") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number of seconds, not '{text}'") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
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
    """The cancel hook of a command: True after Ctrl-C (a second Ctrl-C aborts at once), or
    after a line 'stop' on the input with ``--stop-from-input`` (:func:`_stop_from_input`)."""

    active: _Stop | None = None  # the hook of the command running, for the input's stop
    early = False  # a stop from the input before the command was ready: taken when it is

    def __init__(self):
        self.asked = False

    def __call__(self) -> bool:
        return self.asked

    def _on_interrupt(self, signum, frame):
        if self.asked:  # a second stop (from the input, which cannot reset this handler)
            raise KeyboardInterrupt
        self.asked = True
        log.warning("stopping after the current step (Ctrl-C again to abort now)")
        signal.signal(signal.SIGINT, signal.default_int_handler)

    @contextlib.contextmanager
    def installed(self):
        _Stop.active = self
        if _Stop.early:
            self.asked = True
            log.warning("stopping after the current step (Stop again to abort now)")
        try:
            try:
                previous = signal.signal(signal.SIGINT, self._on_interrupt)
            except ValueError:  # not the main thread: no handler, Ctrl-C raises as usual
                yield self
                return
            try:
                yield self
            finally:
                signal.signal(signal.SIGINT, previous)
        finally:
            _Stop.active = None


def _input_stop() -> None:
    """A 'stop' from the input: the running command stops after its current step; a second
    (or a command without a cancel hook) aborts at once, as a second Ctrl-C does."""
    import _thread

    active = _Stop.active
    if active is not None and not active.asked:
        active.asked = True
        log.warning("stopping after the current step (Stop again to abort now)")
    elif active is None and not _Stop.early:  # not ready yet: stop when it is
        _Stop.early = True
    else:
        _thread.interrupt_main()


def _stop_from_input(stream=None) -> threading.Thread:
    """``--stop-from-input``: a thread that reads the input line by line; 'stop' stops the
    command (:func:`_input_stop`), and so does the input closing (the program that started
    hxex is gone), once."""
    stream = stream if stream is not None else sys.stdin
    lines = _windows_pipe_lines(stream) if os.name == "nt" else None

    def watch():
        try:
            for line in (lines if lines is not None else stream):
                if line.strip().lower() == "stop":
                    _input_stop()
        except (OSError, ValueError):
            pass
        active = _Stop.active
        if not _Stop.early and (active is None or not active.asked):
            _input_stop()

    thread = threading.Thread(target=watch, name="hxex-stop-from-input", daemon=True)
    thread.start()
    return thread


def _windows_pipe_lines(stream):
    """The input's lines on Windows, read only once they have arrived: a thread blocked reading
    a pipe there blocks other uses of the same pipe, and a command run by a program stalled so
    (CI, 2026-11-03). None when the input is not a pipe (a console: read as usual)."""
    import ctypes
    import msvcrt
    import time
    from ctypes import wintypes

    try:
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
    except (OSError, ValueError, AttributeError):
        return None
    peek = ctypes.windll.kernel32.PeekNamedPipe
    available = wintypes.DWORD()
    if not peek(handle, None, 0, None, ctypes.byref(available), None):
        return None  # not a pipe

    def lines():
        fd, pending = stream.fileno(), b""
        while True:
            if not peek(handle, None, 0, None, ctypes.byref(available), None):
                return  # the pipe closed: the end of the input
            if available.value:
                pending += os.read(fd, available.value)
                *done, pending = pending.split(b"\n")
                for line in done:
                    yield line.decode("utf-8", errors="replace")
            else:
                time.sleep(0.1)

    return lines()


def _progress(args, **kwargs):
    """The progress hook of a command: lines every ``--progress-every`` seconds, and with
    ``--progress-format json`` a line for a program as well (:func:`_progress_json`)."""
    from .hooks import ProgressLog

    report = _progress_json if getattr(args, "progress_format", "text") == "json" else None
    return ProgressLog(interval=args.progress_every, report=report, **kwargs)


PROGRESS_PREFIX = "@hxex-progress "


def _progress_json(record: dict) -> None:
    """One progress record as a line on stderr: ``@hxex-progress`` and the record as JSON."""
    import json

    sys.stderr.write(PROGRESS_PREFIX + json.dumps(record) + "\n")
    sys.stderr.flush()


def _beside(bundle: Path, value, suffix: str) -> Path | None:
    if value is None:
        return None
    return bundle.with_name(bundle.name + suffix) if value is True else Path(value)


# --- commands -----------------------------------------------------------------------------------

def cmd_init(args) -> int:
    from .config import template

    if not args.project or ":" in args.project:
        log.error("--project must be a name (non-empty text without ':'), not %r", args.project)
        return EXIT_ERROR
    text = template(args.format, args.project)
    if args.output == "-":
        sys.stdout.write(text)
        return 0
    out = Path(args.output)
    if out.exists() and not args.force:
        log.error("%s exists (use --force to replace it)", out)
        return EXIT_ERROR
    out.write_text(text, encoding="utf-8")
    log.info("wrote %s (format %s): edit the input paths (in a text editor, or with 'hxex set "
             "%s SETTING VALUE'), then 'hxex check %s'. The output "
             "bundle and the folders named after it (NAME_tables, NAME_figures, NAME_analysis_..., "
             "NAME.N) are "
             "hxex's: later runs replace or rename them", out, args.format, out, out)
    return 0


def cmd_set(args) -> int:
    from .project_edit import set_setting

    if args.default == bool(args.value):
        log.error("give the new value of %s, or --default to return it to its default (not "
                  "both)", args.setting)
        return 2
    change = set_setting(args.project_file, args.setting,
                         None if args.default else " ".join(args.value))
    _say(change.describe())
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
    import dataclasses

    from .config import (
        FOLDER_FORMATS,
        ProjectError,
        describe,
        folder_entries,
        stored_spectra_source,
    )

    project = _project(args, analysis_only=True)
    lines = describe(project)
    problems: list[str] = _check_review_ids(project, lines)
    for key, value in lines.items():
        _say(f"{key:14s} {value}")
    inp = project.input
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
        from .io import ReadError
        from .io.entries import filter_entries

        before = len(problems)
        if inp.peptides is not None and not inp.peptides.is_file():
            problems.append(f"input.peptides {inp.peptides}: no such file")
        if not inp.data_dir.is_dir():
            problems.append(f"input.data_dir {inp.data_dir}: no such directory")
        if len(problems) == before:
            # as the read takes them: a peptide table's rows filtered by samples,
            # peptide_ranges and peptide_region (scan filters by itself)
            try:
                found = folder_entries(project if not inp.samples else dataclasses.replace(
                    project, input=dataclasses.replace(inp, samples=None)))
            except ReadError as exc:  # the FASTA against the folder names, a repeated row
                problems.append(str(exc))
        if len(problems) == before:
            entries = filter_entries(found, samples=inp.samples,
                                     peptide_ranges=inp.peptide_ranges,
                                     peptide_region=inp.peptide_region)
            samples = Counter(e.sample for e in entries)
            _say(f"{'entries':14s} {len(entries)} in {inp.data_dir}")
            roles = _folder_control_files(inp, entries)
            for s, n in sorted(samples.items()):
                _say(f"  {s or '(no sample)'}: {n}" + _control_file_text(roles, s))
            if roles is None:
                _say(f"{'controls':14s} the column labels of each sheet mark them (undeut, TD)")
            have = sorted({e.sample for e in found if e.sample})
            for s in inp.samples or ():
                if s not in have:  # a misspelt sample is otherwise silently absent
                    problems.append(f"[input] samples: no sample {s!r} in {inp.data_dir} "
                                    f"(samples found: {', '.join(have) or 'none'})")
            if not entries:
                problems.append(f"no spectra of format {inp.format} in {inp.data_dir}"
                                + _format_hint(project))
            if inp.format == "specexport" and (project.analysis is None
                                               or project.analysis.fasta is None):
                from .io.specexport import _fasta_files

                _say(f"{'sequences':14s} "
                     + _fasta_note(inp.data_dir, ({"file": p.name}
                                                  for p in _fasta_files(inp.data_dir))))
            problems += _check_peptides(project, entries)
            problems += _check_folder_files(project, entries)
            problems += _check_controls_present(project, roles)
            if len(problems) == before and _has_rules(project):
                from .io import spectrum_index

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
        before = len(problems)
        problems += _check_scan_files(project, rows)
        problems += _check_controls_present(project, None)
        if len(problems) == before and _has_rules(project):
            from .exclusions import SpectrumIndex

            problems += _check_exclusions(project, SpectrumIndex.from_rows(rows), inp.format)
    if inp is not None and inp.retention_times is not None and not inp.retention_times.is_file():
        problems.append(f"input.retention_times {inp.retention_times}: no such file")
    problems += _check_ks(project)
    for n, line in enumerate(_output_lines(project)):
        _say(f"{'' if n else 'outputs':14s} {line}")
    if problems:  # after the status block, and the verdict last, whatever the terminal buffers
        sys.stdout.flush()
        for p in problems:
            log.error("%s", p)
        sys.stderr.flush()
        _say(f"not ok: {len(problems)} problem(s) found (listed above)")
        return EXIT_ERROR
    _say("ok")
    return 0


def _format_hint(project) -> str:
    """When a folder format finds nothing, whether another folder format's layout fits the
    folder: ``; the folder holds ... (format = "specexport"?)``, else empty."""
    import dataclasses

    from .config import FOLDER_FORMATS, folder_entries

    layouts = {"specexport": "sample folders with peptide folders",
               "hxexpress": "HX-Express spreadsheets", "pooled": "spectra of one sample"}
    for other in FOLDER_FORMATS:
        if other == project.input.format:
            continue
        try:
            n = len(folder_entries(dataclasses.replace(
                project, input=dataclasses.replace(project.input, format=other,
                                                   peptides=None))))
        except Exception:  # noqa: BLE001 - a layout that does not fit: no hint
            continue
        if n:
            return (f" (the folder holds {layouts[other]}, {n} spectra: format = "
                    f"\"{other}\"?)")
    return ""


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

    def writes(command: str) -> str:
        return f"({command} {'write' if ' and ' in command else 'writes'} it)"

    created = (bundle_manifest(bundle) or {}).get("created", "") if bundle.exists() else ""
    lines = []
    if bundle.exists():
        lines.append(f"{bundle.name} exists {writes('hxex run')}")
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
        lines.append(f"{folder.name}: {state} {writes(command)}")
    analysis = project.analysis  # none: hxex analyse uses the default settings
    refs = [r for r in (analysis.references if analysis is not None else ()) or ("",)]
    every = "all" in refs  # every sample in turn: each of the bundle's analysis folders
    if analysis is not None and analysis.out:
        wanted = [Path(analysis.out)]
    elif analysis is not None and analysis.runs:  # as hxex analyse names it: beside the first
        first = Path(next(iter(analysis.runs.values())))
        wanted = [first.with_name(analysis_folder_name("runs", r)) for r in refs if r != "all"]
    else:
        wanted = [bundle.with_name(analysis_folder_name(bundle.name, r))
                  for r in refs if r != "all"]
    folders = dict(analysis_states(bundle))
    for one in wanted:
        if analysis is not None and analysis.runs and not analysis.out and holds_output(one):
            # an analysis of several runs records each run's results by its name
            made = made_from(one) or {}
            stamps = {name: (bundle_manifest(Path(p)) or {}).get("created", "")
                      for name, p in analysis.runs.items()}
            folders[one] = ("" if any(not made.get(n) for n in stamps) else None
                            if all(made[n] == c for n, c in stamps.items()) else "earlier")
        elif one not in folders and holds_output(one):
            folders[one] = ""
    analysed = {w.resolve() for w in wanted}
    for folder, state in sorted(folders.items()):
        text = ("up to date" if state is None else "made from earlier results" if state
                else "made from results not recorded")
        if every or folder.resolve() in analysed:
            text += f" {writes('hxex analyse')}"
        lines.append(f"{folder.name}: {text}")
    if not lines:
        return [f"none yet: hxex run writes {bundle.name}"]
    # said once for the block: each command refuses by itself, before doing any work
    lines.append("[output] overwrite = true: each command replaces its folder" if out.overwrite
                 else "[output] keep_previous = true: each command moves its folder to NAME.N "
                      "first" if out.keep_previous
                 else "to write one again, give its command --overwrite (replace) or "
                      "--keep-previous (keep the earlier one as NAME.N)")
    return lines


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
            problems.append(f"{text} matches no spectrum: {why}. Correct the rule, or delete "
                            "it if this dataset has no such spectra")
    return problems


def _choices_not_fitted(chosen, bundle: Path) -> str | None:
    """Why ``hxex resume`` cannot bootstrap review-table choices (``chosen``) of a K the bundle
    was not fitted with (the project file may try it now; only ``hxex run`` fits it), or
    None."""
    from .fit import FitConfig
    from .io.bundle import bundle_manifest
    from .review_table import choices_outside

    held = ((bundle_manifest(bundle) or {}).get("configs") or {}).get("fit")
    if not chosen or held is None:
        return None
    fitted = FitConfig.from_dict(held).k_values
    if not (outside := choices_outside(chosen, fitted)):
        return None
    return (f"the review table chooses a K that {bundle} was not fitted with for "
            f"{len(outside)} spectra ({', '.join(outside[:3])}); the bundle was fitted with "
            f"K = {', '.join(map(str, fitted))} and resume fits no new K: run again with "
            "hxex run --overwrite (the review table is applied to the new fits)")


def _check_review_ids(project, lines: dict) -> list[str]:
    """A problem when the project's review table names spectra the results in its bundle do not
    have (``hxex report`` and the analysis would skip those rows), and the ``review table`` line
    of ``lines`` counting only the rows they have; nothing without a review table or a bundle
    (a table that cannot be read is :func:`hxex.config.describe`'s problem)."""
    import dataclasses

    from .io import open_bundle
    from .io.bundle import bundle_manifest
    from .review_table import read_review_table, table_choices

    review, bundle = project.review, project.output.bundle
    if (review is None or not review.table or bundle is None or not bundle.exists()
            or bundle_manifest(bundle) is None):
        return []
    problems = [p for p in [_choices_not_fitted(table_choices(review), bundle)] if p]
    sheet = read_review_table(review.table)
    with open_bundle(bundle) as b:
        known = set(map(str, b.spectrum_ids()))
    keep = [i for i, sid in enumerate(sheet.spectrum_id) if sid in known]
    unknown = [sid for sid in sheet.spectrum_id if sid not in known]
    if not unknown:
        return problems
    matched = dataclasses.replace(sheet, **{name: tuple(getattr(sheet, name)[i] for i in keep)
                                            for name in ("spectrum_id", "status", "k", "note")})
    lines["review table"] = (f"{review.table}: {matched.describe()} (and {len(unknown)} row(s) "
                             f"naming spectra {bundle.name} does not have)")
    more = f", and {len(unknown) - 3} more" if len(unknown) > 3 else ""
    return problems + [
        f"[review] table {review.table}: {len(unknown)} row(s) name spectra the results in "
        f"{bundle.name} do not have ({', '.join(unknown[:3])}{more}); those rows would be "
        "skipped: copy each spectrum_id from spectra.csv, or delete the rows"]


def _check_folder_files(project, entries) -> list[str]:
    """One problem per entry of a SpecExport or pooled folder that the read would refuse from
    its file names (no file of the entry's charge, two files for one point; no file is
    opened), and one per peptide-table row whose sequence or residues differ from its folder's
    name (the run uses the table's and warns)."""
    inp = project.input
    if inp.format == "hxexpress":
        return []
    from .io import ReadError
    from .io.read import _input_path, folder_name_disagreements
    from .io.specexport import specexport_points

    problems = []
    for entry in entries:
        folder = inp.data_dir / _input_path(inp.format, entry)
        if not folder.is_dir():
            problems.append(f"{folder} not found (the peptide table's file column names the "
                            "peptide folder)")
            continue
        try:
            specexport_points(folder, entry)
        except ReadError as exc:
            problems.append(str(exc))
    if inp.peptides is not None:
        problems += folder_name_disagreements(entries, inp.format,
                                              registry=project.run.registry,
                                              data_dir=inp.data_dir)
    return problems


def _check_controls_present(project, roles: dict | None) -> list[str]:
    """A problem when ``[run.controls]`` is on but the input has no control of either kind:
    no control file counted (``roles``, SpecExport and pooled; None for the others), no run
    with a role (mzML and Thermo), and neither ``undeut_at`` nor ``fulldeut_at`` set. The run
    would take 0 D and d2o_fraction as every series' ends: no correction worth the name.
    HX-Express sheets (their labels are inside) and stored spectra are not assessed."""
    controls, inp = project.run.controls, project.input
    if controls is None or inp.spectra is not None or inp.format == "hxexpress":
        return []
    if controls.undeut_at is not None or controls.fulldeut_at is not None:
        return []
    if roles is not None:
        if any(roles.values()):
            return []
        where = "no Non- or Full- spectrum file was found"
    else:
        if any(run.role for run in inp.runs):
            return []
        where = "no [[input.runs]] has a role"
    return [f"[run.controls] is on, but the input has no control: {where}, and neither "
            "undeut_at nor fulldeut_at is set; every series would take 0 D and d2o_fraction "
            "as its ends (no back-exchange correction). Mark the control runs (role = "
            "\"undeut\" or \"fulldeut\"), set undeut_at / fulldeut_at, or delete "
            "[run.controls]"]


def _check_ks(project) -> list[str]:
    """A problem when ``[analysis] ks`` sets a K that ``[run.fit]`` does not try (``hxex
    analyse`` would refuse); left at its default, the analysis uses the Ks the run fits, which
    is said in a line."""
    from .analysis import AnalysisConfig

    analysis = project.analysis
    if analysis is None or project.input is None:
        return []
    tried = project.run.fit.k_values
    missing = [k for k in analysis.ks if k not in tried]
    if not missing:
        return []
    if analysis.ks == AnalysisConfig().ks:
        _say(f"{'analysis':14s} K = {', '.join(str(k) for k in analysis.ks if k in tried)} "
             f"(the run tries K up to {max(tried)})")
        return []
    return [f"[analysis] ks = {list(analysis.ks)}: K = {missing} are not tried by the run "
            f"([run.fit] k_max = {project.run.fit.k_max}): remove them from ks, or raise k_max"]


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
            parts.append('[run.picker] resolving_power = "recorded" would use it with '
                         'pick = "window_max" (pick = "peak" keeps the resolution_ppm window)')
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
    from .model import Backend
    from .pipeline import run
    from .review_table import table_choices

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
    from .io.bundle import bundle_manifest

    if bundle.exists() and bundle_manifest(bundle) is None:  # refused before the read
        log.error("%s exists but is not a results bundle; it is never replaced or moved: "
                  "write to another output.bundle", bundle)
        return EXIT_ERROR
    if bundle.exists() and not (overwrite or keep_previous):
        log.error("%s exists: --overwrite replaces it (and its tables, carrying its review "
                  "over), --keep-previous moves it to the next generation first "
                  "(%s.N), or write to another output.bundle (the bundle and the folders "
                  "named after it are hxex's)", bundle, bundle.name)
        return EXIT_ERROR
    from .io import check_output, holds_output, prepare_output
    from .report.tables import TABLE_NAMES

    tables = _beside(bundle, out.tables, "_tables") if out.tables is not False else None
    if tables is not None and _same_folder(tables, bundle):
        log.error("[output] tables (%s) is the results bundle itself (--bundle %s): the tables "
                  "go in a folder of their own", tables, bundle)
        return EXIT_ERROR
    # the tables folder follows the bundle's rule (refused before anything is fitted); a
    # bundle kept as its previous generation takes its own tables folder along
    moves_along = (keep_previous and bundle.exists() and tables is not None
                   and tables == bundle.with_name(bundle.name + "_tables"))
    if tables is not None and not moves_along:
        prepare_output(tables, overwrite=overwrite, keep_previous=keep_previous, move=False,
                       what="tables of earlier results",
                       elsewhere="another output.bundle (or output.tables)")
        if not (keep_previous and holds_output(tables)):  # stays: its files must be hxex's
            check_output(tables, TABLE_NAMES)
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
        with _progress(args, name="read") as progress:
            spectra = read_input(project, progress=progress, cancel=stop)
        log.info("%d spectra read", len(spectra))
        with _progress(args) as progress:
            result = run(
                spectra, config, out=bundle, project=project.project, notes=out.notes,
                overwrite=overwrite, keep_previous=keep_previous, keep_spectra=out.keep_spectra,
                exclude=project.input.exclude, view_exclude=_view_exclude(project),
                chosen_k=chosen_k,
                project_file=project_record(dataclasses.replace(
                    project, threads=threads, output=dataclasses.replace(
                        out, bundle=bundle, overwrite=overwrite, keep_previous=keep_previous))),
                progress=progress, cancel=stop,
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
    _warn_stale_outputs(bundle, args.project_file, out.overwrite)
    return 0


def _warn_stale_outputs(bundle: Path, project_file, overwrite: bool = False) -> None:
    """Warn about the bundle's figures and analysis folders made from other results than the
    ones there now (after a run, however the earlier bundle went, or a resume), naming the
    command that brings each up to date (without ``--overwrite`` when the project file's
    ``[output] overwrite`` is on)."""
    from .io import holds_output, made_from
    from .io.bundle import analysis_states, bundle_manifest

    created = (bundle_manifest(bundle) or {}).get("created", "")
    flag = "" if overwrite else " --overwrite"
    figures = bundle.with_name(bundle.name + "_figures")
    if holds_output(figures) and (made_from(figures) or {}).get("") != created:
        log.warning("%s holds figures of other results than the ones in %s now: 'hxex report "
                    "%s --figures%s' draws them again", figures, bundle, project_file, flag)
    for folder, state in analysis_states(bundle):
        if state is not None:
            log.warning("%s holds an analysis of other results than the ones in %s now: "
                        "'hxex analyse %s%s' brings it up to date", folder, bundle,
                        project_file, flag)


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


def _changed_controls(project, bundle):
    """The project's ``[run.controls]`` when they differ from those the bundle's controls were
    worked out with (``hxex resume`` works them out again; no refit), else None."""
    from .controls import ControlConfig
    from .io.bundle import bundle_manifest

    if project.run.controls is None:
        return None
    stored = (((bundle_manifest(bundle) or {}).get("configs") or {}).get("run") or {}).get(
        "controls")
    if not stored:
        return None  # a run without controls: run again to have them
    return None if ControlConfig.from_dict(stored) == project.run.controls else \
        project.run.controls


def cmd_resume(args) -> int:
    from .io.bundle import bundle_manifest
    from .pipeline import NothingToResume, resume_bootstrap

    bundle, extra = Path(args.bundle), {}
    tables_arg, threads = args.tables, args.threads
    review = project = None
    if problem := _bundle_problem(bundle, "resume"):
        log.error("%s", problem)
        return EXIT_ERROR
    # a project file: add or redo a bootstrap or the population test
    if bundle.suffix == ".toml" and bundle.is_file():
        import dataclasses

        from .config import check_table_choices, load_project, project_record
        from .review_table import table_choices

        project = load_project(bundle, analysis_only=True)
        if project.output.bundle is None:
            log.error("%s names no [output] bundle", bundle)
            return EXIT_ERROR
        chosen = table_choices(project.review)
        check_table_choices(chosen, project.run.fit.k_values)
        if problem := _choices_not_fitted(chosen, project.output.bundle):
            log.error("%s", problem)
            return EXIT_ERROR
        changed = _changed_controls(project, project.output.bundle)
        if (project.run.bootstrap is None and project.run.population_test is None
                and not chosen and changed is None):
            log.error("%s has no [run.bootstrap] or [run.population_test] table, and its "
                      "[run.controls] are the bundle's: add one with its settings", bundle)
            return EXIT_ERROR
        bundle, review = project.output.bundle, project.review
        threads = threads if threads is not None else project.threads
        test = _test_to_resume(project, bundle)
        if isinstance(test, int):
            return test
        if test is not None or changed is not None:
            record = project_record(dataclasses.replace(project, threads=threads))
            if project.input is None:  # a file for the analysis: the bundle's input stays
                held_file = ((bundle_manifest(bundle) or {}).get("configs") or {}).get(
                    "project_file") or {}
                if held_file.get("input"):
                    record["input"] = held_file["input"]
            extra.update(project_file=record)
        if test is not None:
            extra["population_test"] = test
        if changed is not None:
            extra["run_controls"] = changed
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
        extra.update(bootstrap=project.run.bootstrap, add=redo,
                     view_exclude=_view_exclude(project), chosen_k=chosen)
        out = project.output
        tables_arg = tables_arg if tables_arg is not None else (out.tables or None)
    from .io import check_output, holds_output, made_from, prepare_output
    from .report.tables import TABLE_NAMES

    created = (bundle_manifest(bundle) or {}).get("created", "")
    if project is None and tables_arg is None:
        # the bundle's own tables beside it are brought up to date (audit 2026-11-01 Medium
        # 21); a folder there made from other results is left as it is, with a warning
        own = Path(bundle).with_name(Path(bundle).name + "_tables")
        if holds_output(own) and (made_from(own) or {}).get("") != created:
            log.warning("%s holds tables of other results: not written (hxex report %s "
                        "--overwrite writes them)", own, bundle)
        else:
            tables_arg = True
    tables = _beside(bundle, tables_arg, "_tables")
    # a tables folder holding other results' tables follows the bundle's rule (refused before
    # anything is fitted)
    others = (tables is not None and holds_output(tables)
              and (made_from(tables) or {}).get("") != created)
    flags = {} if project is None else {"overwrite": project.output.overwrite,
                                        "keep_previous": project.output.keep_previous}
    if others:
        prepare_output(tables, move=False, what="tables of other results",
                       elsewhere="another folder with --tables DIR", **flags)
    if tables is not None and not (others and flags.get("keep_previous")):
        check_output(tables, TABLE_NAMES)  # a file of your own there: refused before the fit
    stop = _Stop()
    with stop.installed(), _threads(threads), \
            _progress(args) as progress:
        try:
            result = resume_bootstrap(bundle, device=args.device, progress=progress,
                                      cancel=stop, **extra)
        except NothingToResume:
            if not extra:  # the bundle alone: say how to add or redo a bootstrap
                raise
            _say(f"{bundle}: the bootstrap and the population test have the project's "
                 "settings and every K chosen in the review has a bootstrap: nothing to add")
            return 0
    if others:
        prepare_output(tables, what="tables of other results", **flags)
    if tables is not None:
        from .report import Results, write_tables
        from .review_table import apply_view

        write_tables(apply_view(Results.from_run(result), review), tables,
                     created=(bundle_manifest(bundle) or {}).get("created", ""))
        log.info("tables in %s", tables)
    _summary(result.fit, bundle)
    _warn_stale_outputs(Path(bundle), args.bundle, bool(flags.get("overwrite")))
    return 0


def _test_to_resume(project, bundle):
    """The population test settings ``hxex resume`` gives the bundle of ``project``: the
    project's ``[run.population_test]`` when the bundle has no test or one with other settings
    (None: nothing to do), or an exit code when the bundle cannot take it (its fits differ from
    the project's ``[run.fit]``: run again). The other ``[run]`` settings that differ from the
    bundle's are named in the log, and are not applied."""
    from .io.bundle import bundle_manifest, settings_changed
    from .pipeline import RunConfig

    stored = ((bundle_manifest(bundle) or {}).get("configs") or {}).get("run")
    if stored is None:
        return None  # resume itself refuses a bundle without its run settings
    normal = lambda c: json.loads(json.dumps(c.to_dict()))  # noqa: E731 - as the manifest
    held = normal(RunConfig.from_dict(stored))  # a key added since takes its default
    wanted = normal(project.run)
    changed = [c for c in settings_changed({"run": held}, {"run": wanted})
               if c != "fit.backend.device"]
    test = project.run.population_test
    if test is None:
        if held.get("population_test") is not None:
            log.info("%s: the project has no [run.population_test]; the bundle's population test "
                     "stays", bundle)
    elif held.get("population_test") == wanted["population_test"]:
        test = None
    else:
        fits = [c for c in changed if c.startswith("fit.")]
        if fits:
            log.error("%s: the bundle's fits were made with other [run.fit] settings (%s): the "
                      "population test is made from the bundle's fits; run again with "
                      "'hxex run --overwrite' (or --keep-previous) for the project's", bundle,
                      ", ".join(fits))
            return EXIT_ERROR
    other = [c for c in changed if not c.startswith(("bootstrap", "population_test"))]
    if other:
        log.info("%s: %d [run] setting(s) differ from the bundle's and are not applied by "
                 "resume (they need 'hxex run'): %s", bundle, len(other),
                 ", ".join(other[:8]) + (", ..." if len(other) > 8 else ""))
    return test


def cmd_report(args) -> int:
    """The tables (always, unless a project file's ``[output] tables = false``; ``--tables DIR``
    chooses the folder) and, with ``--figures``, the figures too, from a bundle."""
    import dataclasses

    from .io import (
        check_output,
        edited_outputs,
        holds_output,
        made_from,
        open_bundle,
        prepare_output,
    )
    from .io.bundle import bundle_manifest
    from .report import Results, write_tables
    from .report.figures import PlotConfig, write_figures
    from .report.tables import TABLE_NAMES

    bundle, review, plots = Path(args.bundle), None, PlotConfig()
    overwrite, keep_previous = args.overwrite, args.keep_previous
    own_tables: Path | bool = True  # the project's [output] tables (a bundle alone: beside it)
    if problem := _bundle_problem(bundle, "report"):
        log.error("%s", problem)
        return EXIT_ERROR
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
        if problem := _bundle_problem(bundle, "report"):  # the bundle itself, before its folders
            log.error("%s", problem)
            return EXIT_ERROR
        if _changed_controls(project, bundle) is not None:
            log.warning("%s: [run.controls] differ from the settings the bundle's controls were "
                        "worked out with, so the corrected values follow the bundle's; 'hxex "
                        "resume %s' works them out again with the project's (no refit)",
                        args.bundle, args.bundle)
    figures = _beside(bundle, args.figures, "_figures")
    tables = _beside(bundle, args.tables if args.tables is not None
                     else None if own_tables is False else own_tables, "_tables")
    if tables is None and figures is None:
        _say(f"nothing to write: {args.bundle} has [output] tables = false (--tables writes "
             "the tables all the same, --figures draws the figures)")
        return 0
    outputs = [(folder, what, option) for folder, what, option in (
        (tables, "tables", "--tables DIR"), (figures, "figures", "--figures DIR"))
        if folder is not None]
    for folder, what, option in outputs:
        if _same_folder(folder, bundle):
            log.error("%s is the results bundle itself: the %s go in a folder of their own "
                      "(%s)", folder, what, option)
            return EXIT_ERROR
    # tables made from these same results (hxex run wrote them, or an earlier report) hold
    # nothing the bundle and the project file do not: written again in place, so that
    # 'hxex report hxex.toml --figures' after a run needs no --overwrite for them
    created = (bundle_manifest(bundle) or {}).get("created", "")
    same = (tables is not None and bool(created) and holds_output(tables)
            and (made_from(tables) or {}).get("") == created)
    # ... unless one was changed after hxex wrote it (opened and saved in a spreadsheet, say):
    # then the rule applies after all, and the refusal names the files (owner, 2026-11-02)
    edited = edited_outputs(tables) if same else []
    if edited and not (overwrite or keep_previous):
        log.error("%s: %s changed after hxex wrote %s, and writing the tables again would "
                  "replace %s. To go on:\n"
                  "  copy what you changed out of the folder first (a review table belongs "
                  "next to hxex.toml), then report again;\n"
                  "  or --overwrite replaces %s;\n"
                  "  or --keep-previous moves the folder to %s.N first",
                  tables, ", ".join(edited[:5]) + (", ..." if len(edited) > 5 else ""),
                  "it" if len(edited) == 1 else "them", "it" if len(edited) == 1 else "them",
                  "it" if len(edited) == 1 else "them", tables.name)
        return EXIT_ERROR
    same = same and not edited
    ruled = [(folder, what, option) for folder, what, option in outputs
             if not (what == "tables" and same)]
    for folder, what, option in ruled:  # the bundle's rule, before anything is opened
        prepare_output(folder, overwrite=overwrite, keep_previous=keep_previous, move=False,
                       what=what, elsewhere=f"another folder with {option}")
    if tables is not None and not (keep_previous and not same and holds_output(tables)):
        check_output(tables, TABLE_NAMES)  # a file of your own there: refused before drawing
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
            for folder, _, _ in ruled:
                prepare_output(folder, keep_previous=True)
        said = []
        if tables is not None:
            written = write_tables(results, tables, created=b.manifest.get("created", ""))
            said.append(f"{len(written)} table(s) in {tables}{_review_text(results)}")
        if figures is not None:
            if args.k is not None:
                plots = dataclasses.replace(plots, k=args.k)
            if args.series:
                plots = dataclasses.replace(plots, series=args.series)
            if args.fig_format:
                plots = dataclasses.replace(plots, format=args.fig_format)
            with _progress(args, name="figures") as progress:
                written = write_figures(results, figures, plots, progress=progress, cancel=stop)
            said.append(f"{len(written)} figure file(s) in {figures}")
            if tables is None:
                said.append("no tables ([output] tables = false)")
        _say("; ".join(said))
    return 0


def _review_text(results) -> str:
    """`` (review: 1 rejected, 1 accepted, 2 with a chosen K)`` for the closing line of a
    report, or empty without review decisions."""
    review = results.review
    if review is None or not len(review):
        return ""
    status = np.asarray(results.review_status)
    rejected, accepted = int((status == "rejected").sum()), int((status == "accepted").sum())
    chosen = sum(int(k) > 0 and not stale for k, stale in zip(review.k_override, review.stale,
                                                               strict=True))
    parts = [f"{rejected} rejected"] if rejected else []
    parts += [f"{accepted} accepted"] if accepted else []
    parts += [f"{chosen} with a chosen K"] if chosen else []
    return f" (review: {', '.join(parts)})" if parts else ""


def _bundle_problem(path: Path, command: str) -> str | None:
    """Why ``path`` cannot be the results ``command`` reads, before anything is opened: a
    project file that does not exist, no results yet, or a folder that is not a bundle. None
    when it may be one (or is a project file to read)."""
    if path.suffix == ".toml":
        if not path.is_file():
            return f"no such project file: {path}"
        return None
    if not path.exists():
        return (f"no results yet at {path}: run 'hxex run hxex.toml' first (it writes the "
                f"bundle this command reads)")
    if path.is_dir() and not (path / "manifest.json").is_file():
        from .io.bundle import REPLACING

        if (path / REPLACING).exists():
            return None  # open_bundle says that its writing was interrupted
        return (f"{path} is not a results bundle (it has no manifest.json): a bundle is the "
                "whole folder hxex run wrote")
    return None


def _same_folder(a: Path, b: Path) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return Path(a) == Path(b)


def _fasta_note(folder, sequences) -> str:
    """The one line saying that a SpecExport project's analysis has no sequence of its own:
    the reader's FASTA files are used to read the peptides, the analysis only those named in
    ``[analysis] fasta`` (owner, 2026-11-04: the analysis stays explicit)."""
    files = [str(v["file"]) for v in sequences if v.get("file")]
    where = Path(str(folder)).name or str(folder)
    example = f'fasta = {{ {Path(files[0]).stem} = "{where}/{files[0]}", ... }}' if files else ""
    return ("[analysis] fasta is not set: the analysis takes the sequence from the peptides"
            + (f"; to give it the sequences the peptides were read against, name "
               f"{', '.join(files)} there ({example})" if files else ""))


def _sequences_text(sequences: Mapping) -> str:
    """The ``hxex info`` line of the sequences a SpecExport read recorded: each sample's
    residue range and FASTA file, samples with the same sequence on one entry."""
    import re

    groups: dict[tuple, list[str]] = {}
    for sample, v in sequences.items():
        n = len(re.findall(r"[a-z]*[A-Z]", str(v["sequence"])))  # by residues (modX labels)
        first = int(v.get("first", 1))
        groups.setdefault((str(v["sequence"]), first, first + n - 1), []).append(
            f"{sample} ({v['file']})" if v.get("file") else str(sample))
    return "; ".join(f"{', '.join(names)}: residues {lo}-{hi}"
                     for (_, lo, hi), names in groups.items())


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


def _noise_reference_text(meas, fit) -> str:
    """What the series' noise levels come from, counted (``hxex info``)."""
    from .noise import noise_reference_kinds

    if fit.noise_curve is not None:
        return "none needed: every bin has its own noise (see noise)"
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


def _bootstrap_text(bundle, m: int) -> str:
    """What the bundle's bootstraps cover (owner, 2026-11-04: "yes" did not say whether a K was
    bootstrapped for every spectrum, as the residue level's K = 2 comparisons need): the one
    at each spectrum's chosen K, and each fixed K with the spectra it covers."""
    configs = bundle.manifest.get("configs", {})
    n_boot = (configs.get("bootstrap") or {}).get("n_boot")
    parts = []
    chosen = bundle.bootstrap()
    if chosen is not None:
        covered = int((np.asarray(chosen.n_success) > 0).sum())
        parts.append("at each spectrum's chosen K" + (f", {n_boot} resamples" if n_boot else "")
                     + ("" if covered == m else f" ({covered} of {m} spectra)"))
    for k in bundle.bootstrap_ks():
        fixed = bundle.bootstrap(k)
        covered = int((np.asarray(fixed.n_success) > 0).sum()) if fixed is not None else 0
        parts.append(f"at K = {k} for " + ("every spectrum" if covered == m else
                                            f"{covered} of {m} spectra (chosen in the review)"))
    if not parts:
        return "none"
    if not bundle.bootstrap_ks():
        parts.append("none at a fixed K ([run.bootstrap] ks; the K = 2 comparisons of residues "
                     "need K = 2)")
    return "; ".join(parts)


def cmd_info(args) -> int:
    from .io import generations, open_bundle

    bundle, project = Path(args.bundle), None
    if problem := _bundle_problem(bundle, "info"):
        log.error("%s", problem)
        return EXIT_ERROR
    if bundle.suffix == ".toml":  # a project file: its bundle, and its own review decisions
        from .config import load_project

        project = load_project(bundle, analysis_only=True)
        if project.output.bundle is None:
            log.error("%s names no [output] bundle", bundle)
            return EXIT_ERROR
        bundle = project.output.bundle
        if problem := _bundle_problem(bundle, "info"):
            log.error("%s", problem)
            return EXIT_ERROR
    with open_bundle(bundle) as b:
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
            ("hxex", m.get("versions", {}).get("hxex", "")
             + (f" ({m['versions']['hxex_commit']})"
                if m.get("versions", {}).get("hxex_commit") else "")),
            ("input", source.get("reader", source.get("format", "(not recorded)"))),
            *([("sequences", _sequences_text(source["sequences"]))]
              if source.get("sequences") else []),
            ("bin values", _peak_measure_text(b.peak_measure())),
            ("measurements", str(len(fit))),
            ("series", str(len(set(fit.data_id.tolist())))),
            ("K tried", ", ".join(str(k) for k in fit.config.k_values)
             + (" (and K = 1 for the controls, for the back-exchange correction)"
                if 1 in fit.k_values and 1 not in fit.config.k_values else "")),
            ("selected K", ", ".join(f"{k}: {n}" for k, n in sorted(ks.items()))),
            ("bootstrap", _bootstrap_text(b, len(fit))),
            ("controls", ("yes" if b.has("controls") else "no")
             + _promotion_text(run_config.get("controls"))),
            ("roles", _role_counts_text(fit)),
            ("noise reference", _noise_reference_text(b.measurements(), fit)),
            ("noise", _noise_curve_text(fit)),
            ("times", "seconds" if source.get("time_scale", "seconds") == "seconds"
             else "order: the times only order the points"),
            ("quality flags", "yes" if b.has("quality") else "no"),
            ("raw spectra", "stored" if b.has("spectra") else "not stored"),
            ("pending", ", ".join(b.pending) or "nothing (complete)"),
            ("review", (f"{len(review)} entries" + (f" ({review.n_stale} stale)"
                                                    if review.n_stale else "")
                        if len(review) else "none") + " stored in the bundle"),
        ]
        if project is not None:
            rows += _project_review_rows(project, Path(args.bundle))
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


def _project_review_rows(project, path: Path) -> list[tuple[str, str]]:
    """The ``hxex info`` lines of a project file's own review decisions (its ``[review]``
    table and rules, counted as ``hxex check`` counts them), which hxex report, hxex analyse
    and the run's tables apply, apart from the review stored in the bundle."""
    from .config import describe

    lines = describe(project)
    _check_review_ids(project, lines)  # counts only the rows the bundle has
    rows = []
    if "review table" in lines:
        rows.append(("review table", f"{lines['review table']} (from {path.name}; applied by "
                                     "hxex report and hxex analyse)"))
    if "excluded later" in lines:
        rows.append(("excluded later", f"{lines['excluded later']} (from {path.name})"))
    return rows or [("review table", f"none in {path.name}")]


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


def _stale_analyses(bundle: Path, out: Path, written: Sequence[Path] = ()) -> None:
    """Warn about the bundle's other analysis folders made from earlier results (``hxex
    analyse`` has just brought ``out``, and the folders ``written`` with it, up to date)."""
    from .io.bundle import analysis_states

    done = {p.resolve() for p in [out, *written]}
    for folder, state in analysis_states(bundle):
        if state and folder.resolve() not in done:
            log.warning("%s was made from earlier results (created %s): run hxex analyse for "
                        "it again", folder, state)


def _provenance_rows(prov) -> list[tuple[str, str]]:
    """The ``hxex info`` lines of a manifest's provenance block (none without one)."""
    if not prov:
        return []
    rows = []
    if prov.get("replaced"):
        created = prov.get("replaced_created")
        where = prov.get("moved_to")
        rows.append(("replaced", f"{prov.get('replaced')} (created {created or 'unknown'}) on "
                                 f"{prov.get('at', '')}" + (f", kept as {where}" if where else "")))
        changed = prov.get("settings_changed") or []
        rows.append(("changed", "nothing" if not changed else
                     f"{len(changed)} setting(s): " + ", ".join(changed[:8])
                     + (", ..." if len(changed) > 8 else "")))
        review = prov.get("review") or {}
        if review.get("carried"):
            rows.append(("carried", f"{review['carried']} review entries, "
                                    f"{review.get('stale', 0)} stale"))
    for entry in prov.get("resumed") or ():
        rows.append(("resumed", f"{entry.get('added')} added by hxex resume on "
                                f"{entry.get('at', '')}"
                     + (", with the project file's settings" if entry.get("project_file_replaced")
                        else "")))
    return rows


def _is_bundle(path: Path) -> bool:
    return path.suffix == ".zip" or (path.is_dir() and (path / "manifest.json").is_file())


def _named(text: str) -> bool:
    """Whether a source argument is ``NAME=BUNDLE`` (not a path that holds ``=``)."""
    name, sep, _ = text.partition("=")
    return bool(sep and name) and not Path(text).exists()


def _run_source(text: str) -> tuple[str, Path]:
    """``(name, bundle)`` of a ``NAME=BUNDLE`` or ``BUNDLE`` argument."""
    if _named(text):
        name, _, path = text.partition("=")
        return name, Path(path)
    path = Path(text)
    return (path.name if path.is_dir() else path.stem), path


def cmd_analyse(args) -> int:
    import contextlib
    import dataclasses

    from .analysis import AnalysisConfig, RunSet
    from .io import open_bundle
    from .report import Results, write_analysis

    threads = args.threads
    record_paths = "path"  # the analysis settings file records its paths as the input does
    review = None  # the project's [review] decisions
    overwrite, keep_previous = args.overwrite, args.keep_previous
    first = Path(_run_source(args.source[0])[1])
    if len(args.source) == 1 and first.suffix != ".toml" and (
            problem := _bundle_problem(first, "analyse")):
        log.error("%s", problem)
        return EXIT_ERROR
    if len(args.source) > 1 or _is_bundle(first):
        runs = dict(_run_source(s) for s in args.source)
        if len(runs) < len(args.source):
            log.error("two runs have the same name: give NAME=BUNDLE")
            return EXIT_ERROR
        if len(args.source) == 1 and not _named(args.source[0]):
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
    if args.reference:  # one, several (commas), or all
        names = [n.strip() for n in args.reference.split(",") if n.strip()]
        changes["reference"] = names[0] if len(names) == 1 else tuple(names)
    if args.fig_format:
        changes["format"] = args.fig_format
    if changes:
        config = dataclasses.replace(config, **changes)
    several = len(runs) > 1 or "" not in runs
    lead = next(iter(runs.values()))
    from .analysis.settings import ALL_REFERENCES
    from .io import prepare_output
    from .io.bundle import analysis_folder_name

    for path in runs.values():
        if problem := _bundle_problem(Path(path), "analyse"):
            log.error("%s", problem)
            return EXIT_ERROR
    named = Path(args.out) if args.out else Path(config.out) if config.out else None
    if named is not None and not config.single_reference:
        log.error("%s: several references (%s) make a folder each, named after the reference "
                  "(results_analysis_NAME); leave out --out and [analysis] out, or give one "
                  "reference with --reference NAME", named, ", ".join(config.references))
        return EXIT_ERROR

    def folder(ref: str) -> Path:
        # the default folder is named after the reference, so analyses against different
        # references never write into one folder
        return named if named is not None else lead.with_name(
            analysis_folder_name("runs" if several else lead.name, ref))

    def prepared(outs: dict[str, Path]) -> bool:
        """The folders checked and made ready, before anything is computed."""
        for out in outs.values():
            for name, path in runs.items():
                if _same_folder(out, path):
                    log.error("%s is the results bundle itself%s: the analysis goes in a "
                              "folder of its own (--out DIR, or [analysis] out)", out,
                              f" ({name})" if name else "")
                    return False
            prepare_output(out, overwrite=overwrite, keep_previous=keep_previous,
                           what="analysis",
                           elsewhere="another folder with --out DIR (or [analysis] out)",
                           move=False)
        return True

    # with "all" the references are the results' samples, known once they are open
    outs = ({} if ALL_REFERENCES in config.references
            else {ref: folder(ref) for ref in (config.references or ("",))})
    if outs and not prepared(outs):
        return EXIT_ERROR
    stop = _Stop()
    said = []
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
        if config.fasta is None:
            for r in opened.values():
                if (src := r.settings.spectra_source or {}).get("reader") == "specexport":
                    log.info("%s", _fasta_note(src.get("data_dir", "the data folder"),
                                               (src.get("sequences") or {}).values()))
                    break
        results = RunSet(opened, merge=config.merge) if several else opened[""]
        if not outs:
            from .report import analysis_references

            outs = {ref: folder(ref) for ref in analysis_references(results, config)}
            log.info('[analysis] reference = "all": %d analyses, against %s', len(outs),
                     ", ".join(outs))
            if not prepared(outs):
                return EXIT_ERROR
        for ref, out in outs.items():
            one = config if config.single_reference else config.with_reference(ref)
            if len(outs) > 1:
                log.info("the analysis against %s", ref)
            with _progress(args, name=f"analysis against {ref}" if len(outs) > 1
                           else "analysis") as progress:
                written = write_analysis(
                    results, one, out, base=base, record_paths=record_paths, review=review,
                    progress=progress, cancel=stop,
                    # moved only once nothing more can refuse the analysis
                    before_writing=(lambda out=out: prepare_output(out, keep_previous=True))
                    if keep_previous else None)
            tables = sum(1 for p in written if p.suffix == ".csv")
            figures = sum(1 for p in written if p.suffix == f".{config.format}")
            said.append(f"{tables} table(s) and {figures} figure(s) in {out}")
    if not several:  # said once, for every folder not just written
        _stale_analyses(lead, next(iter(outs.values())), written=list(outs.values()))
    for line in said:
        _say(line + ", with the settings used")
    return 0


def cmd_figures(args) -> int:
    """Which figures ``hxex report --figures`` and ``hxex analyse`` would draw with the project's
    settings (:func:`hxex.report.catalogue.plan`), and the setting that rules out each of the
    others; nothing is drawn."""
    from .config import load_project
    from .io import open_bundle
    from .report import Results, analysis_references
    from .report.catalogue import ANALYSE, REPORT, Situation, plan

    path = Path(args.project)
    if path.suffix != ".toml" or not path.exists():
        log.error("%s: give the project file (hxex.toml)", path)
        return EXIT_ERROR
    project = load_project(path, analysis_only=True)
    bundle = project.output.bundle
    read = (not args.settings_only and bundle is not None
            and _bundle_problem(Path(bundle), "figures") is None)
    counts: dict[str, tuple[str, str]] = {}
    try:
        if read:
            from .review_table import apply_view

            with open_bundle(bundle) as b:
                results = apply_view(Results.from_bundle(b), project.review)
                situation = Situation.of(project.figures, project.analysis, results)
                counts = _figure_counts(results, situation.figures)
        else:
            situation = Situation.of(project.figures, project.analysis)
    except ValueError as exc:  # settings these results refuse: the commands would refuse too
        log.error("%s", exc)
        return EXIT_ERROR
    _say(f"Figures with the settings of {path}"
         + (f" and the results in {bundle}" if read else
            " (settings only: " + ("--settings-only" if args.settings_only else
                                   "no results yet") + ")"))
    analysis = situation.analysis
    if not analysis.single_reference:  # one analysis per reference, the same figures each
        names = (analysis_references(results, analysis) if read else analysis.references)
        _say(f"hxex analyse makes one analysis per reference, a folder each: "
             f"{', '.join(names)}" + ("" if read or "all" not in names else
                                      " (every sample, named once there are results)"))
    planned = plan(situation)
    for command, heading in ((REPORT, f"hxex report {path.name} --figures  ([figures])"),
                             (ANALYSE, f"hxex analyse {path.name}  ([analysis])")):
        _say()
        _say(heading)
        width = max(len(p.kind.title) for p in planned if p.kind.command == command)
        for p in planned:
            if p.kind.command != command:
                continue
            state = p.state
            if state == "off":
                detail = p.reason or "needs " + "; ".join(
                    f"{n.setting} {n.words}" for n in p.missing)
            elif p.kind.name in counts:  # the results tell
                state, detail = counts[p.kind.name]
            else:
                detail = p.reason
            _say(f"  {state:8s} {p.kind.title:{width}s}  {detail}".rstrip())
    _say()
    _say("off: a setting rules it out; depends: the results decide where it is drawn. "
         "What each figure shows: docs/guide/figures.md")
    return 0


def _figure_counts(results, figures) -> dict[str, tuple[str, str]]:
    """What the results say of the run's figures, as (state, detail): the overview's pages,
    and how many series ``[figures] series`` chooses for each page."""
    from .report.figures import select_series, series_kinds
    from .report.overview import overview_kinds

    total = len(np.unique(results.fit.data_id))
    chosen = select_series(results, figures.series)
    which = (f'"{figures.series}"' if isinstance(figures.series, str)
             else "a list of data_id")
    out = {"overview": ("drawn", "pages: " + ", ".join(overview_kinds(results)))}
    for page in ("spectra", "populations", "uptake"):
        n = sum(page in series_kinds(results, d, figures) for d in chosen)
        detail = f"{n} of {total} series (series = {which})"
        if page == "populations" and n < len(chosen):
            detail += f"; {len(chosen) - n} with one population in every spectrum left out"
        out[f"series_{page}"] = ("drawn" if n else "none", detail)
    return out


def cmd_gui(args) -> int:
    from . import gui

    if not gui.available():
        log.error("%s", gui.MISSING)
        return EXIT_ERROR
    return gui.launch(args.path)


COMMANDS = {"init": cmd_init, "set": cmd_set, "check": cmd_check, "run": cmd_run,
            "resume": cmd_resume, "report": cmd_report, "info": cmd_info, "figures": cmd_figures,
            "analyse": cmd_analyse, "analyze": cmd_analyse, "gui": cmd_gui}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    if not args.stop_from_input:
        with _logging_to_stderr(args):
            return _dispatch(args)
    _stop_from_input()
    with _logging_to_stderr(args):
        code = _dispatch(args)
    # the thread waiting on the input would hold the process open at shutdown on Windows (a
    # blocked read is not interrupted): end it here, once everything is written
    sys.stdout.flush()
    sys.stderr.flush()
    logging.shutdown()
    os._exit(code)


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
    except Exception as exc:  # noqa: BLE001 - the boundary: the user never sees a traceback
        log.debug("internal error", exc_info=True)  # the traceback, shown with -v
        log.error("hxex stopped with an internal error (%s: %s); please report it with the "
                  "messages above and the output of the same command run with -v (which shows "
                  "where it stopped)", type(exc).__name__, exc)
        return EXIT_ERROR
