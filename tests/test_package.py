"""Smoke tests for the scaffold: the package installs, imports and runs on CPU."""

import importlib
import sys
from importlib.metadata import version

import pytest

import hxex
from hxex import cli

SUBMODULES = [
    "hxex.chem",
    "hxex.io",
    "hxex.picking",
    "hxex.model",
    "hxex.fit",
    "hxex.report",
    "hxex.noise",
    "hxex.controls",
    "hxex.bootstrap",
    "hxex.run",
    "hxex.cli",
]


def test_python_version_floor():
    assert sys.version_info >= (3, 11)


def test_version_matches_installed_metadata():
    assert hxex.__version__ == version("hxex")


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_imports(name):
    assert importlib.import_module(name) is not None


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert hxex.__version__ in capsys.readouterr().out


def test_cli_no_arguments_runs():
    assert cli.main([]) == 0


def test_core_dependencies_import():
    """The declared runtime dependencies are importable in this environment."""
    for name in ("numpy", "scipy", "torch", "pyarrow", "pyteomics", "brainpy", "matplotlib"):
        assert importlib.import_module(name) is not None


def test_torch_runs_on_cpu():
    """Phase 0 exit criterion: the numerical backend works without a GPU."""
    import torch

    x = torch.arange(6, dtype=torch.float64, device="cpu").reshape(2, 3)
    # rows [0, 1, 2] and [3, 4, 5]: x @ x.T = [[5, 14], [14, 50]], whose sum is 83
    assert float((x @ x.T).sum()) == pytest.approx(83.0)
