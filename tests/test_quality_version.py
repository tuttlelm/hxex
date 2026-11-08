# SPDX-License-Identifier: GPL-3.0-or-later
"""Quality flags assessed by an earlier version of the checks (hxex.quality.QUALITY_VERSION):
the bundle records the version, loading such a bundle says so, hxex info says so, and hxex
resume assesses them again, also on a bundle with nothing else to do (owner, 2026-11-08: a
bundle fitted before overlapping_ion existed kept that flag empty through later resumes)."""

import json
import logging

import numpy as np
import pytest

from hxex.io import open_bundle
from hxex.pipeline import NothingToResume, resume_bootstrap, run
from hxex.quality import QUALITY_VERSION
from hxex.report.results import Results
from test_run import config, raw_spectra


@pytest.fixture
def bundle(tmp_path):
    out = tmp_path / "results"
    run(raw_spectra(), config(), out=out)
    return out


def _age(path, version=None):
    """The bundle as an earlier version wrote it: no quality version (or ``version``), and a
    flag the earlier checks did not have left empty."""
    manifest = json.loads((path / "manifest.json").read_text())
    if version is None:
        manifest.pop("quality_version", None)
    else:
        manifest["quality_version"] = version
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_the_bundle_records_the_checks_version(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["quality_version"] == QUALITY_VERSION
    with open_bundle(bundle) as b:
        assert b.quality().version == QUALITY_VERSION and not b.quality().outdated
    # a complete bundle with current flags: nothing to resume
    with pytest.raises(NothingToResume):
        resume_bootstrap(bundle)


def test_outdated_flags_are_said_and_assessed_again_by_resume(bundle, caplog, capsys):
    from hxex.cli import main

    with open_bundle(bundle) as b:
        before = b.quality()
    _age(bundle)
    with open_bundle(bundle) as b:
        assert b.quality().version == 0 and b.quality().outdated
        with caplog.at_level(logging.WARNING):
            Results.from_bundle(b)
    assert "assessed by an earlier version of hxex" in caplog.text
    assert main(["info", str(bundle)]) == 0
    assert "assessed by an earlier version of hxex" in capsys.readouterr().out
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="hxex.pipeline"):
        resume_bootstrap(bundle)  # nothing else to do: the flags alone
    assert "assessing the quality flags again" in caplog.text
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["quality_version"] == QUALITY_VERSION
    assert manifest["provenance"]["resumed"][-1]["added"] == "quality flags"
    with open_bundle(bundle) as b:
        after = b.quality()
    assert not after.outdated
    np.testing.assert_array_equal(after.flagged, before.flagged)  # the same checks here
