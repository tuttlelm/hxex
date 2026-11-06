# SPDX-License-Identifier: GPL-3.0-or-later
"""The Analysis workspace (hxex.gui.analysis_workspace): the figures of hxex analyse for the
results shown, the catalogue's reasons for those left out, and an analysis marked out of date
when the results or [analysis] change. Runs offscreen."""
import dataclasses
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402

from hxex.gui.analysis_workspace import setting_table  # noqa: E402
from test_analysis import two_sample_bundle  # noqa: E402, F401 - the fixture
from test_gui import _window  # noqa: E402


@pytest.fixture
def project(tmp_path, two_sample_bundle):  # noqa: F811
    path = tmp_path / "hxex.toml"
    path.write_text(f'project = "gui"\n\n[output]\nbundle = "{two_sample_bundle.as_posix()}"\n\n'
                    '[analysis]\nreference = "S"\n', encoding="utf-8")
    return path


def _items(page) -> dict:
    return {page.list.topLevelItem(i).data(0, 256): page.list.topLevelItem(i)
            for i in range(page.list.topLevelItemCount())}


def test_the_analysis_is_computed_shown_and_marked_out_of_date(qtbot, tmp_path, project):
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    window.tabs.setCurrentWidget(page)
    window.show()
    # shown: nothing computed until asked (owner: half a minute on a large run)
    assert page.outputs is None and not page.computing
    assert page.status.text().startswith("Not computed yet")
    assert page.compute_button.text() == "Compute" and page.compute_button.isEnabled()
    page.compute_button.click()
    qtbot.waitUntil(lambda: page.outputs is not None, timeout=120000)
    items = _items(page)
    assert items["woods"].childCount() >= 1 and "figure(s)" in items["woods"].text(1)
    assert items["regions"].text(1) == "off"
    assert "ladder" not in items  # no fragment series: no "Fragment ions (MS2/MS3)" part
    # the first figure is drawn, under the name hxex analyse gives it
    qtbot.waitUntil(lambda: page.panel.view.figure is not None, timeout=60000)
    assert page.panel.save_name in dict(page.outputs.figures)
    # a figure left out says why, and the button goes to its setting
    page.list.setCurrentItem(items["regions"])
    assert "Off: needs [[analysis.regions]]" in page.about.text()
    assert page.change.isVisibleTo(page) and page.change.text() == "Change [[analysis.regions]]…"
    page.change.click()
    assert window.tabs.currentWidget() is window.project_page
    # a review decision (new results) marks it out of date, keeping the figures
    window.tabs.setCurrentWidget(page)
    shown = page.outputs
    window.results_page.results = dataclasses.replace(window.results_page.results)
    page.refresh()
    assert "Out of date: the results changed" in page.status.text()
    assert page.outputs is shown and page.compute_button.text() == "Recompute"
    woods = _items(page)["woods"].child(0)
    with qtbot.waitSignal(page.panel.drawn, timeout=60000):
        page.list.setCurrentItem(woods)
    before, stem = page.panel.view.figure, woods.data(0, 257)
    assert before is not None
    page.compute()
    qtbot.waitUntil(lambda: not page.computing and page.outputs is not shown, timeout=120000)
    assert "as hxex analyse would write them" in page.status.text()
    # the figure shown is drawn again from the new analysis (owner: it was not)
    qtbot.waitUntil(lambda: page.panel.view.figure not in (None, before), timeout=60000)
    assert page.panel.save_name == stem
    # so does a change of [analysis]
    project.write_text(project.read_text() + "alpha = 0.01\n", encoding="utf-8")
    page.refresh()
    assert "[analysis] settings changed" in page.status.text()


def test_settings_the_results_refuse_say_so(qtbot, tmp_path, project):
    project.write_text(project.read_text().replace('"S"', '"nothing"'), encoding="utf-8")
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    window.tabs.setCurrentWidget(page)
    window.show()
    qtbot.waitUntil(lambda: "cannot be made" in page.status.text(), timeout=30000)
    assert "nothing" in page.status.text() and page.outputs is None


def test_several_references_are_chosen_and_kept(qtbot, tmp_path, project):
    project.write_text(project.read_text().replace('reference = "S"', 'reference = "all"'),
                       encoding="utf-8")
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    window.tabs.setCurrentWidget(page)
    window.show()
    page.compute()
    qtbot.waitUntil(lambda: page.outputs is not None, timeout=120000)
    assert page.reference.isVisibleTo(page) and page.reference.count() == 2
    assert page.shown_reference == "S" and "against S" in page.status.text()
    first = page.outputs
    # two sections: the figures the same for every reference first, then the comparisons
    heads = [page.list.topLevelItem(i) for i in range(page.list.topLevelItemCount())
             if page.list.topLevelItem(i).data(0, 256) is None]
    assert [h.text(0) for h in heads] == ["Peptides (MS1): same for every reference",
                                          "Peptides (MS1): compared with the reference S"]
    # the results hold no fragment series: no fragment section
    names = [page.list.topLevelItem(i).data(0, 256)
             for i in range(page.list.topLevelItemCount())]
    assert names.index("coverage") < names.index(None, 1) < names.index("woods")
    assert not heads[0].flags() & Qt.ItemFlag.ItemIsSelectable
    coverage = _items(page)["coverage"].child(0)
    with qtbot.waitSignal(page.panel.drawn, timeout=60000):
        page.list.setCurrentItem(coverage)
    drawn = page.panel.view.figure
    page.reference.setCurrentIndex(1)  # V: computed when asked
    assert page.outputs is None and "Not computed yet against V" in page.status.text()
    assert "shown from the analysis against S" in page.status.text()
    # a figure the same for every reference stays, without computing or drawing again
    assert _items(page)["coverage"].childCount() >= 1 and _items(page)["woods"].childCount() == 0
    assert page.panel.view.figure is drawn and not page.panel.drawing
    page.compute()
    qtbot.waitUntil(lambda: page.outputs is not None and not page.computing, timeout=120000)
    assert page.shown_reference == "V" and page.outputs is not first
    assert any(stem.startswith("woods_S_") for stem, _ in page.outputs.figures)
    # the latest analysis now gives the common figures (a Recompute renews them)
    assert page.shared_reference == "V" and page.shared_outputs is page.outputs
    page.reference.setCurrentIndex(0)  # back to S: kept, not computed again
    assert page.outputs is first and not page.computing
    assert page.shared_reference == "V"


def test_a_changed_review_table_marks_the_analysis_and_is_read_before_recomputing(
        qtbot, tmp_path, project):
    """A review table changed outside the review panel (the Tables view, a spreadsheet) after
    the analysis was computed: the analysis says it may change, and Recompute reads the results
    again with the table first."""
    from hxex.gui.review import make_review_table

    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    table = make_review_table(project, window.results_page.results)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)  # [review] table: read again
    window._project_file_changed()
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    window.tabs.setCurrentWidget(page)
    window.show()
    page.compute()
    qtbot.waitUntil(lambda: page.outputs is not None and not page.computing, timeout=120000)
    assert page.why_out_of_date() == ""
    results = window.results_page.results
    lines = table.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    row = lines[1].split(",")
    row[header.index("review_status")] = "rejected"
    table.write_text("\n".join([lines[0], ",".join(row), *lines[2:]]) + "\n", encoding="utf-8")
    page.refresh()
    assert "review table changed since it was computed" in page.status.text()
    assert "may change" in page.status.text()
    page.compute()
    qtbot.waitUntil(lambda: window.results_page.results is not results and not window.loading
                    and page.outputs is not None and not page.computing, timeout=120000)
    assert page.why_out_of_date() == ""
    assert (window.results_page.results.review_status == "rejected").sum() == 1


def test_setting_tables():
    assert setting_table("[analysis] reference") == "analysis"
    assert setting_table("[[analysis.regions]]") == "analysis.regions"
    assert setting_table("[analysis.fragments]") == "analysis.fragments"


def test_the_compute_time_hint_says_minutes_for_fragment_ladders():
    from hxex.analysis import AnalysisConfig, FragmentAnalysis
    from hxex.gui.analysis_workspace import compute_time
    from test_fragment_kinetics import stub_run

    run = stub_run(conditions=("gentle",))
    fits = AnalysisConfig(fragments=FragmentAnalysis(fit_rates=True))
    hint = compute_time(run, fits)
    assert "ladder rate fits" in hint and f"{len(run.fit):,} fragment spectra" in hint
    # without the rate fits (the default), the fragment analysis, or for peptides read from
    # MS1: the peptides' time only
    quick = "about half a minute for 25,000 spectra"
    assert compute_time(run, AnalysisConfig()) == quick
    assert compute_time(run, AnalysisConfig(fragments=None)) == quick
    run.settings.spectra_source = None
    assert compute_time(run, AnalysisConfig()) == quick


def test_the_fit_rates_box_writes_the_project_file(qtbot, tmp_path, project):
    """Results with fragment series show a 'fit ladder rates' box; ticking it writes
    [analysis.fragments] fit_rates = true to the project file (hxex analyse reads the same),
    unticking writes false (the default). Peptide results have no box."""
    import dataclasses as dc

    from hxex.config import load_project
    from hxex.fit import FitConfig
    from hxex.io.mzml import MzmlRun, MzmlTarget, read_mzml
    from hxex.model import Backend
    from hxex.noise import NoiseConfig
    from hxex.picking import PickerConfig
    from hxex.pipeline import RunConfig, run
    from test_mzml import ENTRY, time_course, write_mzml

    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    page.refresh()
    assert page.fit_rates.isHidden()  # peptides only

    path = write_mzml(tmp_path / "f.mzML", time_course(n=6, ms2_every=2))
    frag = MzmlTarget(dc.replace(ENTRY, extra={"fragment": "c:10"}), ms_level=2,
                      activation="ETD", precursor_mz=1012.0, precursor_charge=20)
    spectra = read_mzml([MzmlRun(path, "WT")], [frag, MzmlTarget(ENTRY)], project="p")
    cfg = RunConfig(picker=PickerConfig(zero_filling=3, noise_threshold=100.0), quality=None,
                    fit=FitConfig(k_min=1, k_max=1, adam_steps=50, n_restarts=1,
                                  backend=Backend("cpu", "float64"),
                                  noise=NoiseConfig("gaussian", percent=3.0)))
    run(spectra, cfg, out=tmp_path / "frag", keep_spectra=False)
    frag_project = tmp_path / "frag.toml"
    bundle = (tmp_path / "frag").as_posix()
    frag_project.write_text(f'project = "p"\n\n[output]\nbundle = "{bundle}"\n\n[analysis]\n',
                            encoding="utf-8")
    window.open_project(frag_project)
    qtbot.waitUntil(lambda: not window.loading and page.results is not None, timeout=30000)
    page.refresh()
    assert page.fit_rates.isVisibleTo(page) and not page.fit_rates.isChecked()
    assert page.fit_rates.isEnabled()
    page.fit_rates.click()
    analysis = load_project(frag_project, analysis_only=True).analysis
    assert analysis.fragments.fit_rates and "fit_rates = true" in frag_project.read_text()
    page.fit_rates.click()
    assert "fit_rates = true" not in frag_project.read_text()
    assert not load_project(frag_project, analysis_only=True).analysis.fragments.fit_rates


def test_the_analysis_list_puts_the_fragment_figures_apart():
    """The Analysis list: the peptides' figures (MS1) in two parts, then the fragment ions'
    (MS2/MS3) in a part of their own, which results without fragment series do not show."""
    from hxex.analysis import AnalysisConfig
    from hxex.gui.analysis_workspace import analysis_sections
    from hxex.report.catalogue import ANALYSE, Situation, plan

    situation = dataclasses.replace(Situation.of(analysis=AnalysisConfig(reference="S")),
                                   fragments=True)
    kinds = [p for p in plan(situation) if p.kind.command == ANALYSE]
    heads = analysis_sections(kinds, True, "S", "S")
    assert [h[0] for h in heads] == ["Peptides (MS1): same for every reference",
                                     "Peptides (MS1): compared with the reference S",
                                     "Fragment ions (MS2/MS3)"]
    names = [[p.kind.name for p in h[2]] for h in heads]
    assert names[2] == ["ladder", "closure", "residue_rates"]
    assert "coverage" in names[0] and "woods" in names[1]
    assert not {"ladder", "closure"} & set(names[0] + names[1])
    assert analysis_sections(kinds, False, "S", "S")[2][2] == []

