# SPDX-License-Identifier: GPL-3.0-or-later
"""The Analysis workspace (hxex.gui.analysis_workspace): the figures of hxex analyse for the
results shown, the catalogue's reasons for those left out, and an analysis marked out of date
when the results or [analysis] change. Runs offscreen."""
import dataclasses

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from hxex.gui.analysis_workspace import setting_table
from test_analysis import two_sample_bundle  # noqa: F401 - the fixture
from test_gui import _window


@pytest.fixture
def project(tmp_path, two_sample_bundle):  # noqa: F811
    path = tmp_path / "hxex.toml"
    path.write_text(f'project = "gui"\n\n[output]\nbundle = "{two_sample_bundle.as_posix()}"\n\n'
                    '[analysis]\nreference = "S"\nresidue_level = true\n', encoding="utf-8")
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
    page.list.setCurrentItem(coverage)  # the first figure: drawn by the Compute already (E-20)
    qtbot.waitUntil(lambda: page.panel.view.figure is not None and not page.panel.drawing,
                    timeout=60000)
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
    assert names[2] == ["fragment_traces", "ladder", "closure", "residue_rates"]
    assert "coverage" in names[0] and "woods" in names[1]
    assert not {"ladder", "fragment_traces", "closure"} & set(names[0] + names[1])
    assert analysis_sections(kinds, False, "S", "S")[2][2] == []



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
    assert page.no_reference_text.text() == "Reference needed for comparison plots"
    choices = [page.choose_reference.itemText(i) for i in range(page.choose_reference.count())]
    assert choices == ["choose a reference…", "S", "V", "every sample in turn"]
    page.choose_reference.activated.emit(choices.index("S"))
    assert tomllib.loads(path.read_text(encoding="utf-8"))["analysis"]["reference"] == "S"
    assert not page.no_reference.isVisibleTo(page)  # compared now: the warning goes
    # every sample in turn
    path.write_text(f'project = "gui"\n\n[output]\nbundle = "{two_sample_bundle.as_posix()}"\n'
                    '\n[analysis]\nresidue_level = true\n', encoding="utf-8")
    page.refresh()
    assert page.no_reference.isVisibleTo(page)
    page.choose_reference.activated.emit(page.choose_reference.count() - 1)
    assert tomllib.loads(path.read_text(encoding="utf-8"))["analysis"]["reference"] == "all"


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
    assert all(page.list.topLevelItem(i).childCount() == 0
               for i in range(page.list.topLevelItemCount()))
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
    """E-8: the reference chosen in the bar is written quoted, so a sample named "37" or
    "2026-11-01" is read as text, not refused or taken for a number."""
    from hxex.config import load_project

    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    page = window.analysis_page
    for name in ("37", "2026-11-01"):
        page.choose_reference.addItem(name, name)
        page._reference_chosen(page.choose_reference.count() - 1)
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


def test_a_refresh_keeps_the_kinds_opened(qtbot, tmp_path, project):
    """E-16: every refresh closed the kinds the user had opened."""
    window, page = _computed_window(qtbot, tmp_path, project)
    items = _items(page)
    for kind in ("coverage", "woods"):
        items[kind].setExpanded(True)
    assert items["coverage"].isExpanded() and items["woods"].isExpanded()
    page.refresh()
    items = _items(page)
    assert items["coverage"].isExpanded() and items["woods"].isExpanded()
    assert not items["butterfly"].isExpanded()


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
