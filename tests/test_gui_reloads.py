# SPDX-License-Identifier: GPL-3.0-or-later
"""The window reading its results again (hxex.gui.window): a change to the review table or
the project file made while the results load is seen when they arrive; the first decision of a
project, which names its review table in the project file, does not read the results again;
and a run or resume of the project shown says that the results will be replaced. Runs
offscreen."""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from hxex.cli import main  # noqa: E402
from hxex.gui.review import table_stamp  # noqa: E402
from test_cli import mzml_project  # noqa: E402


@pytest.fixture
def window_on(qtbot, tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox

    from hxex.gui.window import MainWindow

    project = mzml_project(tmp_path)
    assert main(["-q", "run", str(project)]) == 0
    window = MainWindow(settings=QSettings(str(tmp_path / "g.ini"), QSettings.Format.IniFormat))
    window.ask_to_save = lambda: QMessageBox.StandardButton.Discard
    qtbot.addWidget(window)

    def open_(path):
        window.open_asking(path)
        qtbot.waitUntil(lambda: not window.loading and window.results_path is not None,
                        timeout=60000)
        return window.results_page

    return window, project, open_


def _first_spectrum(page):
    series = page.tree.topLevelItem(0).child(0)
    page.tree.expandItem(series)
    return series.child(0).child(0)


def test_the_first_decision_does_not_read_the_results_again(qtbot, window_on, monkeypatch):
    """The first decision names the review table in the project file, which is a [review]
    change; the decision is in the results shown already, so they are not read again (audit
    of 2026-11-15, D open question 7)."""
    window, project, open_ = window_on
    page = open_(project)
    page.tree.setCurrentItem(_first_spectrum(page))
    loads = []
    monkeypatch.setattr(window._loading, "submit", lambda *a, **k: loads.append(a))
    assert "review.table" not in (window._view_settings or {}).get("values", {})
    assert page.review.decide(status="rejected")
    assert (project.parent / "user_review_table.csv").exists()
    window.project_page.editor._file_changed(str(project))  # as the watcher reports it
    qtbot.wait(100)
    assert loads == []
    assert "review.table" in window._view_settings["values"]
    # a later [review] change still reads them again
    from hxex.project_edit import set_setting

    set_setting(project, "figures.scale", "log")
    window.project_page.editor._file_changed(str(project))
    assert len(loads) == 1


def test_a_table_saved_while_the_results_load_is_applied(qtbot, window_on, monkeypatch):
    """The review table saved (from a spreadsheet, say) while the results are being read:
    read again once they arrive, rather than at the next change (audit of 2026-11-15, A-14 and
    D-6)."""
    from hxex.gui.review import Decision, write_decisions
    from hxex.report import notebook

    window, project, open_ = window_on
    page = open_(project)
    page.tree.setCurrentItem(_first_spectrum(page))
    assert page.review.decide(status="accepted")  # names the table
    table = project.parent / "user_review_table.csv"
    sid = _first_spectrum(page).toolTip(0)
    real = notebook.load_results

    def slowly(path):
        time.sleep(1.5)
        return real(path)

    monkeypatch.setattr(notebook, "load_results", slowly)
    window._pending_selection = page.selection()  # as the window's own rereads keep the place
    window.open_results(project)
    assert window.loading
    qtbot.wait(300)
    write_decisions(table, {sid: Decision(status="rejected")}, {}, results=page.results)
    changed = table_stamp(project)
    assert changed != window.results_review_stamp
    qtbot.waitUntil(lambda: not window.loading, timeout=60000)  # the first read ...
    qtbot.waitUntil(lambda: window.results_review_stamp == changed and not window.loading,
                    timeout=60000)  # ... then once more, with the table as saved meanwhile
    page = window.results_page
    i = int(__import__("numpy").flatnonzero(page.results.fit.spectrum_id == sid)[0])
    assert str(page.results.review_status[i]) == "rejected"
    assert page.selection().get("spectrum_id") == sid  # the place kept


def test_a_run_of_the_project_shown_says_the_results_are_replaced(qtbot, window_on,
                                                                   monkeypatch):
    """While a run or resume of the project whose results are shown runs, the Results panel
    and the status bar say that the results shown are replaced when it finishes; nothing is
    blocked (audit of 2026-11-15, A open question 1)."""
    window, project, open_ = window_on
    page = open_(project)
    run_page = window.run_page
    run_page.args = ["run", project.name]
    run_page._ran_on = project
    monkeypatch.setattr(type(run_page.runner), "running", property(lambda self: True))
    window._run_activity("hxex run hxex.toml: fit")
    assert page.notice.isVisibleTo(page) and "replaced when it finishes" in page.notice.text()
    assert window.replacing.isVisibleTo(window)
    run_page.args = ["check", project.name]  # a check changes no results
    window._run_activity("hxex check hxex.toml")
    assert page.notice.isHidden() and window.replacing.isHidden()
    run_page.args = ["run", project.name]
    window._run_activity("hxex run hxex.toml: fit")
    assert page.notice.isVisibleTo(page)
    monkeypatch.setattr(type(run_page.runner), "running", property(lambda self: False))
    window._run_ended("done")
    assert page.notice.isHidden() and window.replacing.isHidden()
