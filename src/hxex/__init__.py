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

try:
    __version__ = version("hxex")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"

# entry point -> the module that defines it
_ENTRY_POINTS = {
    "load_project": "hxex.config", "read_input": "hxex.config", "template": "hxex.config",
    "Project": "hxex.config", "ProjectError": "hxex.config",
    "run": "hxex.run", "resume_bootstrap": "hxex.run", "RunConfig": "hxex.run",
    "RunResult": "hxex.run",
    "open_bundle": "hxex.io", "write_bundle": "hxex.io", "Bundle": "hxex.io",
    "read_spectra": "hxex.io", "read_mzml": "hxex.io", "read_thermo": "hxex.io",
    "scan": "hxex.io", "read_peptide_table": "hxex.io",
    "Results": "hxex.report", "write_tables": "hxex.report", "write_figures": "hxex.report",
    "write_analysis": "hxex.report", "PlotConfig": "hxex.report",
    "AnalysisConfig": "hxex.analysis",
}

__all__ = ["__version__", *_ENTRY_POINTS]


def __getattr__(name: str):
    module = _ENTRY_POINTS.get(name)
    if module is None:
        raise AttributeError(f"module 'hxex' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_ENTRY_POINTS))
