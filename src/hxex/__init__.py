# SPDX-License-Identifier: GPL-3.0-or-later
"""hxex: multimodal binomial-mixture analysis of HDX-MS spectra.

From spectra (exported or raw) to binned deuterium sticks to fitted populations with
uncertainties, controls, quality flags, tables, figures and a post-fitting analysis. A run is
described by a TOML project file (:mod:`hxex.config`) and driven from the command line
(``hxex init``, ``check``, ``run``, ``report``, ``analyse``; :mod:`hxex.cli`) or from Python::

    import hxex

    project = hxex.load_project("hxex.toml")
    spectra = hxex.read_input(project)
    result = hxex.run(spectra, project.run, out="results", project=project.project)

    with hxex.open_bundle("results") as bundle:
        results = hxex.Results.from_bundle(bundle)
    hxex.write_tables(results, "results_tables")

The entry points below are imported on first use (they load PyTorch); the subpackages are
:mod:`hxex.io` (readers and the results bundle), :mod:`hxex.picking`, :mod:`hxex.fit`,
:mod:`hxex.bootstrap`, :mod:`hxex.controls`, :mod:`hxex.quality`, :mod:`hxex.report` (tables
and figures) and :mod:`hxex.analysis`.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("hxex")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"

_COMMIT_FILE = "_commit.txt"  # "$Format:%h %cs$", filled in by git archive (.gitattributes)
_commit: str | None = None


def _commit_from(text: str) -> str | None:
    """``'3ed39ac, 2026-11-02'`` from the commit file's text once git archive has filled it in
    (``'3ed39ac 2026-11-02'``); None while it still holds the ``$Format`` placeholder."""
    parts = text.split()
    if len(parts) != 2 or "$" in text or "%" in text:
        return None
    return f"{parts[0]}, {parts[1]}"


def _commit_from_git(package: Path) -> str | None:
    """The commit of the git checkout this package runs from (an editable install), with
    ``+dirty`` when tracked files differ from it; None when it is not one, or git is missing or
    slow."""
    import subprocess

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", "-C", str(package), *args], capture_output=True,
                                  text=True, timeout=3, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    top = git("rev-parse", "--show-toplevel")
    try:  # the checkout must be this package's own, not a folder that happens to hold it
        if not top or (Path(top) / "src" / "hxex").resolve() != package.resolve():
            return None
    except OSError:
        return None
    found = git("log", "-1", "--format=%h %cs")
    commit = _commit_from(found or "")
    if commit is None:
        return None
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        name, date = commit.split(", ")
        commit = f"{name}+dirty, {date}"
    return commit


def commit() -> str:
    """The commit hxex was installed from and its date (``'3ed39ac, 2026-11-02'``), or empty
    when it is not known: from a download made by git archive (GitHub's zip), or from the git
    checkout of an editable install."""
    global _commit
    if _commit is None:
        package = Path(__file__).parent
        try:
            text = (package / _COMMIT_FILE).read_text(encoding="utf-8")
        except OSError:
            text = ""
        _commit = _commit_from(text) or _commit_from_git(package) or ""
    return _commit


def version_text() -> str:
    """``'hxex 0.0.1.dev0 (3ed39ac, 2026-11-02)'``, without the parentheses when the commit
    is not known."""
    known = commit()
    return f"hxex {__version__}" + (f" ({known})" if known else "")


# entry point -> the module that defines it
_ENTRY_POINTS = {
    "load_project": "hxex.config", "read_input": "hxex.config", "template": "hxex.config",
    "Project": "hxex.config", "ProjectError": "hxex.config",
    "run": "hxex.pipeline", "resume_bootstrap": "hxex.pipeline", "RunConfig": "hxex.pipeline",
    "RunResult": "hxex.pipeline",
    "open_bundle": "hxex.io", "write_bundle": "hxex.io", "Bundle": "hxex.io",
    "read_spectra": "hxex.io", "read_mzml": "hxex.io", "read_thermo": "hxex.io",
    "scan": "hxex.io", "read_peptide_table": "hxex.io",
    "Results": "hxex.report", "write_tables": "hxex.report", "write_figures": "hxex.report",
    "write_analysis": "hxex.report", "PlotConfig": "hxex.report",
    "load_results": "hxex.report", "show_series": "hxex.report",
    "AnalysisConfig": "hxex.analysis",
}

__all__ = ["__version__", "commit", "version_text", *_ENTRY_POINTS]


def __getattr__(name: str):
    module = _ENTRY_POINTS.get(name)
    if module is None:
        raise AttributeError(f"module 'hxex' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_ENTRY_POINTS))
