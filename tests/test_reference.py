# SPDX-License-Identifier: GPL-3.0-or-later
"""The settings reference of the user guide (docs/guide/settings.md)."""

from pathlib import Path

from hxex import _reference

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "guide"
PAGE = GUIDE / "settings.md"


def test_every_setting_has_a_description():
    assert _reference.missing() == [], (
        "describe these settings in hxex.config._COMMENTS (the template's tables) or "
        "hxex._reference._DESCRIPTIONS ([input], [output], [analysis])")


def test_the_settings_page_is_up_to_date():
    assert PAGE.read_text(encoding="utf-8") == _reference.markdown(), (
        "regenerate it: python -m hxex._reference docs/guide")


def test_the_commands_page_is_up_to_date_and_every_argument_described():
    assert _reference.undescribed_arguments() == [], "give these arguments a help text"
    page = (GUIDE / "commands.md").read_text(encoding="utf-8")
    assert page == _reference.commands_markdown(), (
        "regenerate it: python -m hxex._reference docs/guide")
    for command in ("init", "check", "run", "resume", "report", "analyse", "info"):
        assert f"## hxex {command}\n" in page
    assert "`--keep-previous`" in page and "Also spelt `hxex analyze`" in page


def test_the_reference_lists_the_tables_a_project_file_accepts():
    names = {t.name for t in _reference.tables()}
    assert {"", "input", "input.runs", "input.fragments", "input.scans", "input.target",
            "output", "run", "run.picker", "run.fit", "run.fit.noise", "run.bootstrap",
            "run.controls", "run.quality", "run.plots", "run.envelope", "analysis",
            "analysis.fragments", "analysis.labelling"} <= names
    keys = {s.key for t in _reference.tables() if t.name == "run.controls" for s in t.settings}
    assert {"d2o_fraction", "undeut_at", "fulldeut_at", "quench"} <= keys
