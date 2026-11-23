# SPDX-License-Identifier: GPL-3.0-or-later
"""The Analysis workspace (hxex.gui.analysis_workspace): the figures of hxex analyse for the
results shown, the catalogue's reasons for those left out, and an analysis marked out of date
when the results or [analysis] change. Runs offscreen."""
import dataclasses

import pytest

pytest.importorskip("PySide6")


from hxex.gui.analysis_workspace import KIND, setting_table
from test_analysis import two_sample_bundle  # noqa: F401 - the fixture
from test_gui import _window


@pytest.fixture
def project(tmp_path, two_sample_bundle):  # noqa: F811
    path = tmp_path / "hxex.toml"
    path.write_text(f'project = "gui"\n\n[output]\nbundle = "{two_sample_bundle.as_posix()}"\n\n'
                    '[analysis]\nreference = "S"\nresidue_level = true\n', encoding="utf-8")
    return path


def _items(page) -> dict:
    """Each kind's first row in the tree (in the tree's order: under its top before Not
    drawn)."""
    found = {}
    for item in page._leaf_items.values():
        found.setdefault(item.data(0, 256), item)
    return found


def _at(page, *path):
    """The row of a kind at ``path`` (the rows above it, then the kind)."""
    return page._leaf_items["\x1f".join(path)]


def _tops(page) -> list[str]:
    return [page.list.topLevelItem(i).text(0) for i in range(page.list.topLevelItemCount())]


def _under(page, top: str) -> set:
    """The kinds anywhere under the top ``top``."""
    return {e.kind for e in page._nodes.values() if e.path[0] == top}


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
    # one row per kind, saying how many figures it has (chosen above the figure, not listed)
    assert items["woods"].childCount() == 0 and "figure" in items["woods"].text(1)
    assert items["regions"].text(1) == "off" and _under(page, "Not drawn") >= {"regions"}
    # no fragment series: no fragment figures in the tree, the kinds listed as not drawn
    assert "ladder" in _under(page, "Not drawn")
    assert not {"ladder", "regions"} & set().union(*(_under(page, t) for t in _tops(page)
                                                    if t != "Not drawn"))
    # the first figure is drawn, under the name hxex analyse gives it
    qtbot.waitUntil(lambda: page.panel.view.figure is not None, timeout=60000)
    assert page.panel.save_name in dict(page.outputs.figures)
    # a figure left out says why, and the button goes to its setting
    page.list.setCurrentItem(items["regions"])
    assert "Off: needs [[analysis.regions]]" in page.about.text()
    assert page.change.isVisibleTo(page) and page.change.text() == "Change [[analysis.regions]]…"
    page.change.click()
    assert window.tabs.currentWidget() is window.project_page
    # the same results read again (a [figures] change) are no change (E-9) ...
    window.tabs.setCurrentWidget(page)
    shown = page.outputs
    results = window.results_page.results
    window.results_page.results = dataclasses.replace(results)
    page.refresh()
    assert page.why_out_of_date() == "" and "as hxex analyse would write them" in page.status.text()
    # ... and the analysis keeps no Results object (E-22)
    assert not isinstance(page._analyses["S"][1][0], type(results))
    # a review decision (new results) marks it out of date, keeping the figures
    from hxex.io.bundle import ReviewTable

    review = results.review if results.review is not None else ReviewTable.empty()
    window.results_page.results = results.with_review(
        review.update(str(results.fit.spectrum_id[0]), status="rejected"))
    page.refresh()
    assert "Out of date: the results changed" in page.status.text()
    assert page.outputs is shown and page.compute_button.text() == "Recompute"
    woods = _items(page)["woods"]
    with qtbot.waitSignal(page.panel.drawn, timeout=60000):
        page.list.setCurrentItem(woods)
    before, stem = page.panel.view.figure, page.choices.stem
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


def test_several_references_are_ticked_computed_and_kept(qtbot, tmp_path, project):
    """The references are ticked in a list written to [analysis] reference (owner,
    2026-11-23): Compute makes the analysis against each, the tree has a Comparisons part per
    reference, and unticking one writes the file and keeps what was computed."""
    project.write_text(project.read_text().replace('reference = "S"', 'reference = "all"'),
                       encoding="utf-8")
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    window.tabs.setCurrentWidget(page)
    window.show()
    menu = page.references.menu()
    assert page.references.isVisibleTo(page) and not page.reference.isVisibleTo(page)
    assert [(a.text(), a.isChecked()) for a in menu.actions()] == [("S", True), ("V", True)]
    words = page.references.toolTip().replace("\n", " ")
    assert "[analysis] reference in the project file" in words
    assert "Not computed yet against S, V" in page.status.text()
    page.compute()
    qtbot.waitUntil(lambda: set(page._analyses) == {"S", "V"} and not page.computing,
                    timeout=240000)
    assert _tops(page) == ["Summary", "Single samples", "Comparisons (vs. S)",
                           "Comparisons (vs. V)", "Not drawn"]
    assert "figure(s) against S, V" in page.status.text()
    on_v = page._leaf_items["\x1f".join(("Comparisons (vs. V)", "S", "woods"))]
    page.list.setCurrentItem(on_v)  # the reference of the part selected is the one shown
    assert page.shown_reference == "V" and page.outputs is page._analyses["V"][0]
    first = page._analyses["S"][0]
    # the list stays open while boxes are ticked (owner, 2026-11-23), written when it closes
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    menu.popup(page.references.mapToGlobal(QPoint(0, page.references.height())))
    qtbot.waitUntil(menu.isVisible, timeout=5000)
    v = menu.actions()[1]
    at = menu.actionGeometry(v).center()
    QTest.mouseMove(menu, at)
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=at)
    assert menu.isVisible() and not v.isChecked()  # V unticked, the list still open
    assert 'reference = "all"' in project.read_text()  # nothing written yet
    menu.close()  # V unticked: written, its analysis kept
    qtbot.waitUntil(lambda: 'reference = "S"' in project.read_text(), timeout=5000)
    qtbot.waitUntil(lambda: _tops(page)[2:4] == ["Comparisons (vs. S)", "Not drawn"],
                    timeout=30000)
    assert page._analyses["S"][0] is first and not page.computing


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


def test_the_tree_places_each_figure(qtbot, tmp_path, project):
    """The tree (owner, 2026-11-22, 2026-11-23): Summary (coverage, redundancy, then the
    samples together), Single samples (each sample's figures, coverage first: the summary
    figures drawn for one sample among them), Comparisons (the whole group's significance
    maps, then each sample's figures, its panel of the overlays and its tiers among them)."""
    window, page = _computed_window(qtbot, tmp_path, project)
    rows = set(page._row_items)
    assert {("Single samples", "S"), ("Single samples", "V"), ("Comparisons (vs. S)", "V")} <= rows
    assert ("Comparisons (vs. S)", "S") not in rows  # the reference compares with nothing
    summary = [page.list.topLevelItem(0).child(i).data(0, KIND)
               for i in range(page.list.topLevelItem(0).childCount())]
    assert summary[:2] == ["coverage", "redundancy_resolution"]
    v = _at(page, "Single samples", "V", "coverage").parent()
    assert [v.child(i).data(0, KIND) for i in range(2)] == ["coverage",
                                                            "redundancy_resolution"]
    assert "Residue fDavg" == _at(page, "Single samples", "V", "residues_all").text(0)
    one = page._nodes["\x1f".join(("Single samples", "V", "residues_all"))]
    assert all(stem.startswith("residues_all_V_") for stem, _ in one.figures)
    assert not set(dict(one.figures)) & set(dict(page.outputs.figures))  # the window's own
    assert _under(page, "Comparisons (vs. S)") >= {"woods", "butterfly"}
    _at(page, "Comparisons (vs. S)", "V", "woods")  # a sample's comparisons under the sample
    # the overlays of every sample are each sample's panel instead (owner, 2026-11-23)
    mine = page._nodes["\x1f".join(("Comparisons (vs. S)", "V", "overlay_residues"))]
    assert all(stem.startswith("overlay_residues_V_") for stem, _ in mine.figures)
    keys = {n.key[:2] for n in page._nodes.values()}
    assert ("Comparisons (vs. S)", "overlay_residues") not in keys
    assert "drawn for this sample alone" not in page.about.text()
    with qtbot.waitSignal(page.panel.drawn, timeout=60000):
        page.list.setCurrentItem(_at(page, "Single samples", "V", "residues_all"))
    assert "drawn for this sample alone" in page.about.text()



def test_without_a_reference_the_window_says_so_and_offers_one(qtbot, tmp_path,
                                                               two_sample_bundle):  # noqa: F811
    """Two samples and no [analysis] reference: a warning above the list that the samples are
    not compared, with a list to choose the reference, written to the project file (owner,
    2026-11-13: the comparisons were off and only the list's "off" lines said why)."""
    import tomllib

    path = tmp_path / "hxex.toml"
    path.write_text(f'project = "gui"\n\n[output]\nbundle = "{two_sample_bundle.as_posix()}"\n'
                    '\n[analysis]\nresidue_level = true\n', encoding="utf-8")
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(path)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    window.tabs.setCurrentWidget(page)
    window.show()
    assert page.no_reference.isVisibleTo(page)
    assert page.no_reference_text.text() == (
        "Reference needed for comparison plots: tick one under comparisons against")
    # the list above chooses it (owner, 2026-11-23: no second choice beside the warning)
    menu = page.references.menu()
    assert page.references.isVisibleTo(page)
    assert [(a.text(), a.isChecked()) for a in menu.actions()] == [("S", False), ("V", False)]
    menu.actions()[0].setChecked(True)
    page._references_ticked()  # as when the list closes
    assert tomllib.loads(path.read_text(encoding="utf-8"))["analysis"]["reference"] == "S"
    assert not page.no_reference.isVisibleTo(page)  # compared now: the warning goes
    # every sample, ticked: a list of them
    path.write_text(f'project = "gui"\n\n[output]\nbundle = "{two_sample_bundle.as_posix()}"\n'
                    '\n[analysis]\nresidue_level = true\n', encoding="utf-8")
    page.refresh()
    assert page.no_reference.isVisibleTo(page)
    for action in menu.actions():
        action.setChecked(True)
    page._references_ticked()
    assert tomllib.loads(path.read_text(encoding="utf-8"))["analysis"]["reference"] == ["S", "V"]


def test_a_reference_set_shows_no_warning(qtbot, tmp_path, project):
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    assert not window.analysis_page.no_reference.isVisibleTo(window.analysis_page)


def _held_analysis(monkeypatch, *, listens: bool = True):
    """hxex.report.analysis_outputs (as the window imports it) held until released; returns
    (started, release, outcome), outcome["end"] = "done" / "cancelled" once it ended."""
    import threading

    import hxex.report
    from hxex.hooks import Cancelled

    real = hxex.report.analysis_outputs
    started, release, outcome = threading.Event(), threading.Event(), {}

    def held(results, config, **kw):
        started.set()
        release.wait(60)
        if not listens:  # a computation that does not ask its hook
            kw["cancel"] = None
        try:
            out = real(results, config, **kw)
        except Cancelled:
            outcome["end"] = "cancelled"
            raise
        outcome["end"] = "done"
        return out

    monkeypatch.setattr(hxex.report, "analysis_outputs", held)
    return started, release, outcome


def _settled() -> None:
    """Every background job of the window finished and what it sent back delivered: the
    evidence that something did not happen is that nothing is left to make it happen, not a
    fixed wait a loaded machine could outrun (audit of 2026-11-15, Q-29)."""
    from PySide6.QtCore import QCoreApplication, QThreadPool

    from hxex.gui.tasks import drawing_pool

    pools = (drawing_pool(), QThreadPool.globalInstance())
    for _ in range(10):
        for pool in pools:
            assert pool.waitForDone(60000)
        QCoreApplication.processEvents()
        if not any(pool.activeThreadCount() for pool in pools):
            break


def _nothing_computed(page) -> None:
    assert page.outputs is None and page.shared_outputs is None and not page.computing
    assert not page._analyses and not page.busy.isVisibleTo(page)
    assert not any(e.figures for e in page._nodes.values())
    assert page.status.text().startswith("Not computed yet")
    assert page.compute_button.isEnabled()


def test_another_project_opened_during_a_compute_shows_nothing_of_the_first(
        qtbot, tmp_path, project, two_sample_bundle, monkeypatch):  # noqa: F811
    """E-1: project A's analysis, computed while project B was opened, is stopped and never
    shown as B's (both compare against a reference, the usual case)."""
    other = tmp_path / "b"
    other.mkdir()
    b = other / "hxex.toml"
    b.write_text(f'project = "b"\n\n[output]\nbundle = "{two_sample_bundle.as_posix()}"\n\n'
                 '[analysis]\nreference = "V"\nresidue_level = true\n', encoding="utf-8")
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    window.tabs.setCurrentWidget(page)
    window.show()
    started, release, outcome = _held_analysis(monkeypatch)
    page.compute()
    assert started.wait(30) and page.computing
    assert window.open_asking(b)
    qtbot.waitUntil(lambda: not window.loading and window.results_path == b.absolute(),
                    timeout=30000)
    release.set()
    qtbot.waitUntil(lambda: "end" in outcome, timeout=120000)
    _settled()  # anything the computation sent back has arrived
    assert outcome["end"] == "cancelled"  # the library stopped at its next step
    _nothing_computed(page)
    assert page.shown_reference in ("", "V")


def test_a_bundle_opened_during_the_first_compute_shows_nothing_of_the_project(
        qtbot, tmp_path, project, two_sample_bundle, monkeypatch):  # noqa: F811
    """E-3: nothing computed yet, a bundle opened while the first Compute runs: the Compute is
    dropped (here one that does not ask its hook: it runs to its end, unseen)."""
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    window.tabs.setCurrentWidget(page)
    window.show()
    started, release, outcome = _held_analysis(monkeypatch, listens=False)
    page.compute()
    assert started.wait(30) and page.computing and not page._analyses
    window.open_results(two_sample_bundle)
    qtbot.waitUntil(lambda: not window.loading
                    and window.results_path == two_sample_bundle.absolute(), timeout=30000)
    assert not page.computing  # dropped as soon as the bundle is shown
    release.set()
    qtbot.waitUntil(lambda: "end" in outcome, timeout=120000)
    _settled()
    assert outcome["end"] == "done"
    _nothing_computed(page)
    # and an analysis of other results arriving is never stored as this one's
    page._computed(object(), page.results, None, project.absolute(), "S", None)
    assert page.outputs is None and not page._analyses


def test_a_recompute_waiting_for_its_results_does_not_start_for_others(
        qtbot, tmp_path, project, two_sample_bundle):  # noqa: F811
    """E-23: a Recompute reads the project's results again first; a bundle opened meanwhile
    is shown, and nothing is computed for it."""
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    window.tabs.setCurrentWidget(page)
    window.show()
    window.results_review_stamp = ("a review table", "changed since")  # read again first
    page.compute()
    assert window.loading and not page.computing  # the project's results read again first
    window.open_results(two_sample_bundle)  # before the project's results are read again
    qtbot.waitUntil(lambda: not window.loading
                    and window.results_path == two_sample_bundle.absolute(), timeout=30000)
    _settled()
    _nothing_computed(page)  # not started for the bundle


# --- fix Session 5 (audit of 2026-11-15) ---------------------------------------------------------

def _computed_window(qtbot, tmp_path, project):
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    window.tabs.setCurrentWidget(page)
    window.show()
    page.compute()
    qtbot.waitUntil(lambda: page.outputs is not None and not page.computing, timeout=120000)
    qtbot.waitUntil(lambda: page.panel.view.figure is not None and not page.panel.drawing,
                    timeout=60000)
    return window, page


def test_a_reference_named_like_a_number_or_a_date_is_written_as_text(qtbot, tmp_path, project):
    """E-8: the reference ticked in the list is written quoted, so a sample named "37" or
    "2026-11-01" is read as text, not refused or taken for a number."""
    from hxex.config import load_project

    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    menu = page.references.menu()
    for name in ("37", "2026-11-01"):
        menu.clear()
        action = menu.addAction(name)
        action.setData(name)
        action.setCheckable(True)
        action.setChecked(True)
        page._ticked = []
        page._references_ticked()
        assert load_project(project, analysis_only=True).analysis.reference == name
        assert f'reference = "{name}"' in project.read_text(encoding="utf-8")


def test_a_recompute_draws_the_figure_shown_once(qtbot, tmp_path, project):
    """E-15: a Recompute drew the figure shown twice (refresh, then _computed again)."""
    window, page = _computed_window(qtbot, tmp_path, project)
    draws = []
    page.panel.drawn.connect(lambda: draws.append(1))
    first = page.outputs
    page.compute()
    qtbot.waitUntil(lambda: page.outputs is not first and not page.computing, timeout=120000)
    qtbot.waitUntil(lambda: not page.panel.drawing and bool(draws), timeout=60000)
    _settled()
    assert len(draws) == 1


def _show(qtbot, page, item) -> None:
    """Select a kind and wait for its figure (it may be shown already: no signal then)."""
    page.list.setCurrentItem(item)
    qtbot.waitUntil(lambda: not page.panel.drawing and page.panel.view.figure is not None
                    and page.panel.save_name == page.choices.stem, timeout=60000)


def test_a_refresh_keeps_the_figure_chosen(qtbot, tmp_path, project):
    """E-16: a refresh kept neither the place in the list nor the figure chosen."""
    window, page = _computed_window(qtbot, tmp_path, project)
    _show(qtbot, page, _at(page, "Summary", "coverage"))
    with qtbot.waitSignal(page.panel.drawn, timeout=60000):
        page.choices.box("sample").setCurrentIndex(1)
    chosen = page.choices.stem
    page.list.collapseItem(page.list.topLevelItem(2))  # Comparisons closed, a sample opened
    _at(page, "Single samples", "V", "residues_all").parent().setExpanded(True)
    page.refresh()
    assert page.list.currentItem() is _at(page, "Summary", "coverage")
    assert page.choices.stem == chosen
    assert not page.list.topLevelItem(2).isExpanded()
    assert _at(page, "Single samples", "V", "residues_all").parent().isExpanded()


def test_the_figure_of_a_kind_is_chosen_above_it(qtbot, tmp_path, project):
    """A row per kind at each place (owner, 2026-11-22): what varies among its figures there
    is chosen in lists above the figure, each among the figures the lists before it allow;
    another kind keeps the values where it has them, and a list with one value is hidden (the
    tree's row names the sample)."""
    window, page = _computed_window(qtbot, tmp_path, project)
    _show(qtbot, page, _at(page, "Summary", "coverage"))
    assert page.choices.isVisibleTo(page)
    sample = page.choices.box("sample")
    assert sample.count() == 2
    with qtbot.waitSignal(page.panel.drawn, timeout=60000):
        sample.setCurrentIndex(1)
    chosen = sample.currentData()
    assert page.panel.save_name == page.choices.stem == f"coverage_{chosen}"
    # a kind under one sample: no sample list; one time point: no time list either
    page.list.setCurrentItem(_at(page, "Comparisons (vs. S)", "V", "woods"))
    shown = [b for b in page.choices.findChildren(type(sample)) if b.isVisibleTo(page)]
    assert shown == []  # hidden at once, not when Qt deletes the old lists
    _show(qtbot, page, _at(page, "Comparisons (vs. S)", "V", "woods"))
    assert page.choices.values()["time"] == page.panel.save_name.rsplit("_", 1)[1]


def test_a_refresh_while_the_results_are_read_again_keeps_compute_busy(qtbot, tmp_path, project):
    """E-19: during the re-read before a Recompute, a refresh enabled Compute and replaced the
    line."""
    window, page = _computed_window(qtbot, tmp_path, project)
    window.results_review_stamp = ("a review table", "changed since")  # read again first
    page.compute()
    assert window.loading and not page.computing
    page.refresh()
    assert not page.compute_button.isEnabled()
    assert "Reading the results again" in page.status.text()
    qtbot.waitUntil(lambda: not window.loading and not page.computing, timeout=120000)
    assert page.compute_button.isEnabled()


def test_relative_analysis_paths_count_from_the_project_files_real_folder(qtbot, tmp_path,
                                                                        project):
    """E-25: a project file reached through a link takes its relative [analysis] paths from
    the real folder, as hxex analyse does."""
    link = tmp_path / "elsewhere"
    link.mkdir()
    linked = link / "hxex.toml"
    try:
        linked.symlink_to(project)
    except (OSError, NotImplementedError):
        pytest.skip("no symbolic links here")
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(linked)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    assert window.analysis_page._settings()[1] == project.parent


def test_stop_ends_the_computation_and_keeps_the_figures(qtbot, tmp_path, project, monkeypatch):
    """E open question 2: a Stop beside Compute; the library stops at its next step, nothing
    of it is shown, and the figures of the earlier analysis stay."""
    window, page = _computed_window(qtbot, tmp_path, project)
    first = page.outputs
    started, release, outcome = _held_analysis(monkeypatch)
    assert not page.stop_button.isVisibleTo(page)
    page.compute()
    assert started.wait(30) and page.computing
    assert page.stop_button.isVisibleTo(page) and not page.compute_button.isEnabled()
    page.stop()
    assert not page.computing and not page.stop_button.isVisibleTo(page)
    assert page.status.text().startswith("The computation was stopped")
    assert page.compute_button.isEnabled() and page.outputs is first
    release.set()
    qtbot.waitUntil(lambda: "end" in outcome, timeout=120000)
    _settled()
    assert outcome["end"] == "cancelled" and page.outputs is first


def test_groups_are_branches_of_the_tree(qtbot, tmp_path, project):
    """With [analysis.groups] every group is a branch under each top (owner, 2026-11-22),
    computed together; each group's figures are its own (the groups hold different samples),
    its comparisons against its own reference."""
    project.write_text(project.read_text().replace('reference = "S"\n', "") +
                       '[analysis.groups.G1]\nreference = "S"\nsamples = ["S", "V"]\n'
                       '[analysis.groups.G2]\nreference = "V"\nsamples = ["V"]\n',
                       encoding="utf-8")
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    window.tabs.setCurrentWidget(page)
    window.show()
    assert not page.reference.isVisibleTo(page)  # no choice: every group in the tree
    assert page.no_reference.isHidden()  # the groups name their references
    page.compute()
    qtbot.waitUntil(lambda: page.outputs is not None and not page.computing, timeout=180000)
    assert "of every group" in page.status.text()
    assert {"G1", "G2"} <= set(page._analyses)
    rows = page._row_items
    assert ("Summary", "G1") in rows and ("Single samples", "G2", "V") in rows
    assert rows[("Comparisons", "G1")].text(1) == "against S"
    assert ("Comparisons", "G2") not in rows  # G2 holds its reference alone
    g1 = page._nodes["\x1f".join(("Summary", "G1", "coverage"))]
    g2 = page._nodes["\x1f".join(("Summary", "G2", "coverage"))]
    assert len(g1.figures) == 2 and len(g2.figures) == 1  # each group's samples only


def test_the_regions_across_the_groups_are_in_the_summary(qtbot, tmp_path, project):
    """With two groups sharing a region name, Compute makes each group's analysis, then the
    table and the figures hxex analyse writes in results_analysis_groups, under Summary."""
    project.write_text(project.read_text().replace('reference = "S"\n', "") +
                       '[[analysis.regions]]\nname = "all"\nstart = 1\nend = 30\n'
                       '[analysis.groups.G1]\nreference = "S"\nsamples = ["S"]\n'
                       '[analysis.groups.G2]\nreference = "V"\nsamples = ["V"]\n',
                       encoding="utf-8")
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    window.tabs.setCurrentWidget(page)
    window.show()
    assert "Not computed yet of every group" in page.status.text()
    page.compute()
    qtbot.waitUntil(lambda: page.outputs is not None and not page.computing, timeout=180000)
    table = page.outputs.tables["cross_group_regions"]
    assert set(table["group"]) == {"G1", "G2"}
    node = page._nodes["\x1f".join(("Summary", "cross_group_regions"))]
    assert len(node.figures) == len(set(table["time"])) >= 1
    fig = node.figures[0][1]()
    assert fig.axes and fig.legends  # drawn as hxex analyse draws it


def test_a_groups_regions_changed_after_computing_put_it_out_of_date(qtbot, tmp_path, project):
    """A region's residues changed in a group's own list (owner, 2026-11-18: Compute kept the
    old regions): the group's analysis is out of date, and "(across the groups)" computes that
    group again instead of reusing it."""
    from hxex.gui.analysis_workspace import ACROSS

    project.write_text(project.read_text().replace('reference = "S"\n', "") +
                       '[analysis.groups.G1]\nreference = "S"\nsamples = ["S"]\n'
                       '[[analysis.groups.G1.regions]]\nname = "all"\nstart = 1\nend = 30\n'
                       '[analysis.groups.G2]\nreference = "V"\nsamples = ["V"]\n'
                       '[[analysis.groups.G2.regions]]\nname = "all"\nstart = 1\nend = 30\n',
                       encoding="utf-8")
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    window.tabs.setCurrentWidget(page)
    window.show()
    page.compute()  # every group
    qtbot.waitUntil(lambda: page.outputs is not None and not page.computing, timeout=180000)
    assert page.why_out_of_date() == ""
    project.write_text(project.read_text().replace("end = 30\n[analysis.groups.G2]",
                                                   "end = 8\n[analysis.groups.G2]"),
                       encoding="utf-8")
    qtbot.waitUntil(lambda: "settings changed" in page.why_out_of_date(), timeout=10000)
    assert page.shown_reference == ACROSS
    page.compute()
    qtbot.waitUntil(lambda: page.outputs is not None and not page.computing, timeout=180000)
    table = page.outputs.tables["cross_group_regions"]
    ends = {(g, int(e)) for g, e in zip(table["group"], table["end"], strict=True)}
    assert ("G1", 8) in ends and ("G1", 30) not in ends  # G1 computed again


def test_a_fragment_parent_is_kept_between_the_rates_and_the_traces(qtbot):
    """The residue rates of a parent read at several precursor charges name it without the
    charge (``S_2-12``), the traces with it (``S_2-12_z3``): moving between them kept no
    parent and showed the first (owner, 2026-11-23)."""
    from hxex.gui.figure_choices import ChoicesRow
    from hxex.report.catalogue import figure_choices, kind_named

    row = ChoicesRow()
    qtbot.addWidget(row)
    traces = figure_choices(kind_named("fragment_traces"), [
        "fragments_S_1-9_z2_etd_c_z1", "fragments_S_2-12_z2_etd_c_z1",
        "fragments_S_2-12_z3_etd_c_z1"])
    rates = figure_choices(kind_named("residue_rates"), ["residue_rates_S_1-9_z2",
                                                         "residue_rates_S_2-12"])
    assert row.set_figures(traces, stem="fragments_S_2-12_z3_etd_c_z1")
    assert row.set_figures(rates, keep=row.values()) == "residue_rates_S_2-12"
    # back to the charge last chosen
    assert row.set_figures(traces, keep=row.values()) == "fragments_S_2-12_z3_etd_c_z1"
    assert row.set_figures(rates, keep=row.values()) == "residue_rates_S_2-12"


def test_a_parent_a_kind_lacks_is_said_and_comes_back_after(qtbot):
    """A parent with no closure figure showed the first parent there, which the next kind then
    kept (owner, 2026-11-23): the closure says so beside its lists, and the next kind with the
    parent shows it again."""
    from hxex.gui.figure_choices import ChoicesRow
    from hxex.report.catalogue import figure_choices, kind_named

    row = ChoicesRow()
    qtbot.addWidget(row)
    row.show()
    rates = figure_choices(kind_named("residue_rates"), ["residue_rates_S_2-12_z3",
                                                         "residue_rates_S_31-36_z2"])
    closure = figure_choices(kind_named("closure"), ["closure_S_2-12_z3_c-1+z-dot",
                                                     "closure_S_51-72_z4_c-1+z-dot"])
    traces = figure_choices(kind_named("fragment_traces"), [
        "fragments_S_2-12_z3_EThcD_c-1_z1", "fragments_S_31-36_z2_EThcD_c-1_z1"])
    assert row.set_figures(rates, stem="residue_rates_S_2-12_z3")
    row.box("parent").setCurrentIndex(1)  # 31-36, chosen
    assert row.set_figures(closure, keep=row.values()) == "closure_S_2-12_z3_c-1+z-dot"
    assert row.note.isVisible() and "S_31-36_z2" in row.note.text()
    assert row.set_figures(traces, keep=row.values()) == "fragments_S_31-36_z2_EThcD_c-1_z1"
    assert not row.note.isVisible()
    assert row.set_figures(rates, keep=row.values()) == "residue_rates_S_31-36_z2"


def test_the_head_to_head_figure_is_in_its_own_part_once():
    """The pooled sequences' head to head compares sequences, not samples with a reference:
    the same whichever reference is ticked, so it has a part of its own, listed once (owner,
    2026-11-23)."""
    from types import SimpleNamespace

    from hxex.gui.analysis_tree import HEAD_TO_HEAD, Placed, comparisons_top, entries

    def outputs(ref):
        return SimpleNamespace(figures=[("head_to_head_30min", lambda: None),
                                        (f"woods_X_30min_{ref}", lambda: None)],
                               window={"samples": ["X"]})

    placed = [Placed(None, None, outputs(r), r, comparisons_top(r)) for r in ("A", "B")]
    found = entries(placed)
    h2h = [e for e in found if e.kind == "head_to_head"]
    assert [e.path for e in h2h] == [(HEAD_TO_HEAD,)] and len(h2h[0].figures) == 1
    assert {e.path[0] for e in found if e.kind == "woods"} == {"Comparisons (vs. A)",
                                                              "Comparisons (vs. B)"}
