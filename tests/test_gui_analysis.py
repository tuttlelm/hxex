# SPDX-License-Identifier: GPL-3.0-or-later
"""The Analysis workspace (hxex.gui.analysis_workspace): the figures of hxex analyse for the
results shown, the catalogue's reasons for those left out, and an analysis marked out of date
when the results or [analysis] change. Runs offscreen."""
import dataclasses
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

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
    assert items["regions"].text(1) == "off" and items["ladder"].text(1) == "off"
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
    page.reference.setCurrentIndex(1)  # V: computed when asked
    assert page.outputs is None and "Not computed yet against V" in page.status.text()
    page.compute()
    qtbot.waitUntil(lambda: page.outputs is not None and not page.computing, timeout=120000)
    assert page.shown_reference == "V" and page.outputs is not first
    assert any(stem.startswith("woods_S_") for stem, _ in page.outputs.figures)
    page.reference.setCurrentIndex(0)  # back to S: kept, not computed again
    assert page.outputs is first and not page.computing


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
