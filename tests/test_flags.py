# SPDX-License-Identifier: GPL-3.0-or-later
"""The dictionary of flags, control notes, population-test flags, spectrum quality screens, and
left-out reasons (hxex.flags): complete against the names the code writes, both ways; its
thresholds filled in from the settings; ``hxex explain``; ``flags.csv``; the window's
tooltips."""

import re
from pathlib import Path

import numpy as np
import pytest

from hxex import cli, flags
from hxex.pipeline import RunConfig, run
from hxex.quality import QualityConfig
from hxex.report import Results
from hxex.report.tables import write_tables
from test_run import config, raw_spectra  # noqa: I001

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "guide"


def test_every_name_the_code_writes_has_an_entry_and_every_entry_is_written():
    produced = flags.produced()
    assert set(produced) == set(flags.FAMILIES)
    for family, names in produced.items():
        known = [e.name for e in flags.of_family(family)]
        assert sorted(set(names) - set(known)) == [], f"{family}: give these an entry"
        assert sorted(set(known) - set(names)) == [], f"{family}: no code writes these"
    assert len({e.name for e in flags.ENTRIES}) == len(flags.ENTRIES)


# names in hxex/controls.py that look like control flags but are columns or settings
_NOT_CONTROL_FLAGS = {"fd_time", "m0_se", "m0_measured", "retention_floor", "fd_retention_map"}


def test_every_control_flag_written_in_the_controls_module_is_listed():
    """CONTROL_FLAGS is kept by hand: every flag-like name the module writes must be in it,
    so that a new one cannot reach series.csv without an entry here."""
    from hxex.controls import CONTROL_FLAGS

    text = (ROOT / "src" / "hxex" / "controls.py").read_text(encoding="utf-8")
    found = set(re.findall(r"\"((?:fd|m0|retention|data|no)_[a-z_0-9]+)\"", text))
    assert sorted(found - _NOT_CONTROL_FLAGS - set(CONTROL_FLAGS)) == []


def test_every_left_out_reason_written_in_the_analysis_is_listed():
    from hxex.analysis.species import LEFT_OUT_REASONS

    text = (ROOT / "src" / "hxex" / "analysis" / "species.py").read_text(encoding="utf-8")
    # the reasons are named constants; no reason is written as a bare string to mark()
    assert not re.findall(r"mark\([^\n]*\"", text)
    assert "stand-in control (screened)" in LEFT_OUT_REASONS


def _slug(heading: str) -> str:
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_entries_name_real_settings_and_guide_sections_and_render_at_the_defaults():
    from hxex import _reference

    tables = {t.name: {s.key for s in t.settings} for t in _reference.tables()}
    for e in flags.ENTRIES:
        assert e.short and e.action and e.level and e.where, e.name
        for setting in e.settings:
            table, key = re.fullmatch(r"\[([a-z_.]+)\] ([a-z_0-9]+)", setting).groups()
            assert table in tables and key in tables[table], f"{e.name}: no setting {setting}"
        d = flags.describe(e)  # every placeholder resolves
        assert "{" not in "".join(d.values()), e.name
        if e.guide:
            page, _, anchor = e.guide.partition("#")
            text = (GUIDE / page).read_text(encoding="utf-8")
            headings = {_slug(h) for h in re.findall(r"^#+ (.+)$", text, flags=re.MULTILINE)}
            assert anchor in headings, f"{e.name}: no heading {anchor} in {page}"


def test_thresholds_follow_the_settings_given():
    e = flags.entry("low_intensity")
    assert "`min_intensity` (0.3)" in flags.describe(e)["meaning"]
    values = flags.setting_values(RunConfig(quality=QualityConfig(min_intensity=0.5)))
    assert "`min_intensity` (0.5)" in flags.describe(e, values)["meaning"]
    # a table not set gives its defaults; a setting switched off reads "off"
    assert "`m0_max` (0.5)" in flags.describe(flags.entry("m0_high"))["short"]
    off = flags.setting_values(RunConfig(quality=QualityConfig(max_mz_offset_ppm=None)))
    assert "`max_mz_offset_ppm` (off)" in flags.describe(flags.entry("mz_shift"), off)["meaning"]
    # a placeholder inside backticks is a setting written out
    assert '`stand_in_controls = "exclude"`' in flags.describe(flags.entry("fd_median"))["action"]
    # the screens' thresholds at each level, the level in use marked
    action = flags.describe(flags.entry("screen_below_mono"), level="strict")["action"]
    assert "0.1 at `loose` and 0.05 at `strict` (the level used)" in action


def test_names_are_found_as_the_tables_write_them():
    assert flags.entry("OFF_MZ").name == "off_mz"
    assert flags.entry(" fd_multimodal (retention agrees) ").name == "fd_multimodal"
    assert flags.entry("stand-in control (screened)").name == "stand-in control (screened)"
    assert flags.entry("off_mx") is None and flags.near("off_mx") == ["off_mz"]
    assert flags.names_in("off_mz;poor_fit;nonsense") == ["off_mz", "poor_fit"]
    tip = flags.tooltip(["poor_fit", "nonsense"])
    assert tip.startswith("**poor_fit**: The fit is much worse") and "nonsense" not in tip


def test_explain_gives_one_in_full_or_lists_them_all(tmp_path, capsys):
    assert cli.main(["explain", "off_mz"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("off_mz (quality flag, per series")
    for label in ("where", "meaning", "hxex does", "look at", "settings", "guide"):
        assert f"  {label}" in out
    assert "off_mz_sd (6)" in out and "`" not in out
    assert cli.main(["explain", "stand-in", "control", "(screened)"]) == 0
    assert "left-out reason" in capsys.readouterr().out
    assert cli.main(["explain", "off_mx"]) == 1
    assert "did you mean off_mz?" in capsys.readouterr().err
    assert cli.main(["explain"]) == 0
    listing = capsys.readouterr().out
    for title, _, _ in flags.FAMILIES.values():
        assert title in listing
    assert all(f"    {e.name}: " in listing for e in flags.ENTRIES)
    # with a project: its thresholds
    project = tmp_path / "hxex.toml"
    project.write_text("[analysis]\n[analysis.filters]\nspectrum_quality_filtering = \"strict\"\n"
                       "[run.quality]\nmin_intensity = 0.4\n", encoding="utf-8")
    assert cli.main(["explain", "low_intensity", "--project", str(project)]) == 0
    out = " ".join(capsys.readouterr().out.split())  # as one line: the text is wrapped
    assert "min_intensity (0.4)" in out and f"(from {project})" in out
    assert cli.main(["explain", "screen_between", "--project", str(project)]) == 0
    assert "0.2 at strict (the level used)" in " ".join(capsys.readouterr().out.split())


def test_the_flags_page_is_generated_from_the_dictionary():
    from hxex import _reference

    page = (GUIDE / "flags.md").read_text(encoding="utf-8")
    assert page == _reference.flags_markdown(), (
        "regenerate it: python -m hxex._reference docs/guide")
    for e in flags.ENTRIES:
        assert f"`{e.name}`" in page
    assert "flags.md" in (GUIDE / "outputs.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def results():
    return Results.from_run(run(raw_spectra(n_series=2), config(bootstrap=None)))


def test_the_tables_explain_the_flags_that_occur(tmp_path, results):
    counts, units = flags.run_counts(results)
    found = {n for n, c in counts.items() if c}
    written = {p.name for p in write_tables(results, tmp_path / "t")}
    if not found:
        assert "flags.csv" not in written
        return
    assert "flags.csv" in written
    import csv

    with open(tmp_path / "t" / "flags.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["name"] for r in rows} == found
    for r in rows:
        assert int(r["count"]) == counts[r["name"]] and r["meaning"] and "`" not in r["meaning"]
        assert r["unit"] == units.get(r["name"], "spectra")


def test_the_flags_table_counts_and_orders_by_the_dictionary():
    t = flags.table({"poor_fit": 2, "fd_median": 3, "off_mz": 0, "something new": 1},
                    units={"fd_median": "series"})
    assert list(t["name"]) == ["poor_fit", "fd_median", "something new"]
    assert list(t["count"]) == [2, 3, 1] and list(t["unit"]) == ["spectra", "series", "spectra"]
    assert t["family"][2] == "" and t["meaning"][2] == ""
    assert "control_fallback" in t["action"][1] and "`" not in t["action"][1]


def test_an_analysis_counts_its_left_out_reasons_and_population_flags():
    tables = {
        "left_out_spectra": {"reason": np.array(["replicate outlier", "replicate outlier",
                                                 "stand-in control", ""])},
        "species_uptake": {
            "run": np.array(["", "", "", ""]), "sample": np.array(["S", "S", "S", "V"]),
            "peptide": np.array(["PEP"] * 4), "modification": np.array([""] * 4),
            "start": np.array([1] * 4), "end": np.array([3] * 4),
            "time": np.array([10.0, 60.0, 60.0, 60.0]),
            "undeuterated_two": np.array([True, True, True, False]),
            "more_populations_possible": np.array([False, True, True, True])}}
    counts, units = flags.analysis_counts(tables)
    assert counts == {"replicate outlier": 2, "stand-in control": 1, "undeuterated_two": 1,
                      "more_populations_possible": 2}
    assert units["undeuterated_two"] == "peptides"
