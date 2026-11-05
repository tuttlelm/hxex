# SPDX-License-Identifier: GPL-3.0-or-later
"""The figure catalogue (hxex.report.catalogue): every file hxex draws has its entry, and the
plan of a project's settings agrees with what is drawn."""

import dataclasses
import re
from pathlib import Path

import pytest

from hxex.analysis import AnalysisConfig, Annotations, FragmentAnalysis, Region
from hxex.io import open_bundle
from hxex.report import PlotConfig, Results, write_analysis, write_figures
from hxex.report.catalogue import (
    ANALYSE,
    KINDS,
    REPORT,
    Situation,
    kind_named,
    kind_of,
    plan,
)
from test_analysis import two_sample_bundle  # noqa: F401 - the fixture
from test_fragment_kinetics import stub_run

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "guide"


def _drawn(directory: Path, command: str) -> dict[str, set[str]]:
    """The kinds of the figures written in ``directory``, each with its files."""
    out: dict[str, set[str]] = {}
    for path in directory.iterdir():
        if path.suffix in (".png", ".svg"):
            kind = kind_of(path.stem, command)
            assert kind is not None, f"{path.name} has no entry in the catalogue"
            out.setdefault(kind.name, set()).add(path.name)
    return out


def _agrees(situation: Situation, drawn: dict[str, set[str]], command: str) -> None:
    for p in plan(situation):
        if p.kind.command != command:
            continue
        if p.state == "off":
            assert p.kind.name not in drawn, f"{p.kind.name} is drawn, though ruled out"
        elif p.state == "drawn":
            assert p.kind.name in drawn, f"{p.kind.name} is not drawn"


@pytest.mark.parametrize("changes", [
    {"reference": "S", "annotations": Annotations(regions=(Region("all", 1, 10),))},
    {"reference": "S", "ks": (1,)},
    {},
], ids=["reference with regions", "K=1 only", "no reference"])
def test_every_analysis_figure_has_an_entry_and_the_plan_agrees(two_sample_bundle, tmp_path,  # noqa: F811
                                                                 changes):
    cfg = AnalysisConfig(format="png", **changes)
    with open_bundle(two_sample_bundle) as b:
        results = Results.from_bundle(b)
        write_analysis(results, cfg, tmp_path)
        situation = Situation.of(analysis=cfg, results=results)
    drawn = _drawn(tmp_path, ANALYSE)
    _agrees(situation, drawn, ANALYSE)
    if cfg.reference:
        assert {"woods", "butterfly", "residue_significance"} <= drawn.keys()
    else:
        assert not {"woods", "butterfly", "overlay_residues"} & drawn.keys()
    if cfg.ks == (1,):
        assert "profile" in drawn and "choice" not in drawn


def test_fragment_figures_have_their_entries(tmp_path):
    run = stub_run()
    cfg = AnalysisConfig(format="png", fragments=FragmentAnalysis(min_time=10.0))
    write_analysis(run, cfg, tmp_path)
    drawn = _drawn(tmp_path, ANALYSE)
    assert {"ladder", "closure", "residue_rates"} <= drawn.keys()
    _agrees(Situation.of(analysis=cfg, results=run), drawn, ANALYSE)
    off = Situation.of(analysis=dataclasses.replace(cfg, fragments=None), results=run)
    assert {p.kind.name for p in plan(off) if p.state == "off"} >= {"ladder", "closure"}


def test_every_run_figure_has_an_entry_and_the_plan_agrees(two_sample_bundle, tmp_path):  # noqa: F811
    figures = PlotConfig(series="all", format="svg")
    with open_bundle(two_sample_bundle) as b:
        results = Results.from_bundle(b)
        write_figures(results, tmp_path, figures)
    drawn = _drawn(tmp_path, REPORT)
    assert {"overview", "series_spectra", "series_uptake"} <= drawn.keys()
    _agrees(Situation.of(figures=figures, results=results), drawn, REPORT)
    only = PlotConfig(series="all", format="svg", kinds=("uptake",), overview=False)
    with open_bundle(two_sample_bundle) as b:
        write_figures(Results.from_bundle(b), tmp_path / "only", only)
    drawn = _drawn(tmp_path / "only", REPORT)
    assert drawn.keys() == {"series_uptake"}
    _agrees(Situation.of(figures=only), drawn, REPORT)


def test_the_plan_names_the_setting_that_rules_a_figure_out():
    planned = {p.kind.name: p for p in plan(Situation.of())}
    woods = planned["woods"]
    assert woods.state == "off" and [n.setting for n in woods.missing] == ["[analysis] reference"]
    assert planned["coverage"].state == "drawn"
    # without the results, whether the run has the population test is not known
    assert planned["population_tiers"].state == "depends"
    assert "[analysis] populations" in planned["population_tiers"].reason
    with_ref = {p.kind.name: p for p in plan(Situation.of(analysis=AnalysisConfig(
        reference="WT", populations="best_set")))}
    assert with_ref["woods"].state == "drawn"
    assert with_ref["population_tiers"].state == "off"
    assert with_ref["woods_K2"].state == "depends"


def test_kinds_are_told_apart_by_their_longest_name():
    assert kind_of("woods_K2_V_30s").name == "woods_K2"
    assert kind_of("woods_V_30s").name == "woods"
    assert kind_of("protection_difference_V").name == "protection_difference"
    assert kind_of("residue_significance_K2_WT").name == "residue_significance_K2"
    assert kind_of("residue_significance").name == "residue_significance"
    assert kind_of("WT_0001_0010_PEPTIDE_z_2_uptake", REPORT).name == "series_uptake"
    assert kind_of("something_else") is None
    assert kind_named("woods").title == "Woods plot"


def _slug(heading: str) -> str:
    """A heading's anchor as GitHub makes it."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def test_every_entry_names_real_settings_and_guide_pages():
    from hxex import _reference

    tables = {t.name: {s.key for s in t.settings} for t in _reference.tables()}
    names = {kind.name for kind in KINDS}
    assert len(names) == len(KINDS)
    for kind in KINDS:
        page, _, anchor = kind.guide.partition("#")
        text = (GUIDE / page).read_text(encoding="utf-8")
        if anchor:
            headings = {_slug(h) for h in re.findall(r"^#+ (.+)$", text, flags=re.MULTILINE)}
            assert anchor in headings, f"{kind.name}: no heading {anchor} in {page}"
        for setting in [n.setting for n in kind.needs] + list(kind.changed_by):
            match = re.fullmatch(r"\[\[?([a-z_.]+)\]\]?(?: ([a-z_0-9]+))?", setting)
            assert match, f"{kind.name}: {setting!r} is not written as the guide writes it"
            table, key = match.groups()
            assert table in tables, f"{kind.name}: no table [{table}]"
            assert key is None or key in tables[table], f"{kind.name}: no setting {setting}"
