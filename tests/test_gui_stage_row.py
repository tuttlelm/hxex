# SPDX-License-Identifier: GPL-3.0-or-later
"""The Run tab's stage row (hxex.gui.stage_row; docs/PLAN.md Next steps 40, step 4): one mark per
stage and per output, the reasons in their tooltips, the run buttons' words, the check off the
window's thread and quiet, and the status bar's line outside the Run tab. Runs offscreen."""
import logging
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402

from hxex import cli  # noqa: E402
from hxex.gui import stage_row as stage_row_module  # noqa: E402
from hxex.gui.stage_row import RUN, UPDATE, RowState, StageRow, work_out  # noqa: E402
from hxex.stages import DONE, NOT_DONE, OFF, OUT_OF_DATE, STAGES, Stage  # noqa: E402
from test_cli import mzml_project  # noqa: E402


def _state(path, **changes) -> RowState:
    stages = [Stage(name, DONE) for name in STAGES]
    state = RowState(path, stages=stages, outputs=[Stage("tables", DONE)],
                     plan=[(s, f"keeps the {s}") for s in STAGES], run="keeps everything",
                     has_results=True, up_to_date=True)
    for key, value in changes.items():
        setattr(state, key, value)
    return state


@pytest.fixture
def row(qtbot, tmp_path, monkeypatch):
    widget = StageRow()
    qtbot.addWidget(widget)
    started = []
    monkeypatch.setattr(widget, "_start", lambda slow=False: started.append(slow))
    widget.started = started
    widget.project = tmp_path / "hxex.toml"
    return widget


def test_each_mark_says_its_state_and_its_reasons(row, tmp_path):
    stages = [Stage(name, DONE) for name in STAGES]
    stages[3] = Stage("fit", OUT_OF_DATE, ["[run.fit] k_max: 2, was 1"])
    stages[4] = Stage("population_test", OFF)
    stages[6] = Stage("bootstrap", NOT_DONE, ["stopped partway: hxex run continues it"])
    row.state = _state(row.project, stages=stages, up_to_date=False,
                       plan=[("fit", "keeps fits"), ("fit", "fits missing K (k_max 2)")],
                       run="keeps fits, fits missing K (k_max 2)")
    row._show()
    fit = row.marks["fit"]
    assert "<b>! fit</b>" in fit.text()
    assert "[run.fit] k_max: 2, was 1" in fit.toolTip()
    assert "Run: keeps fits, fits missing K (k_max 2)" in fit.toolTip()
    assert "✓ read" in row.marks["read"].text()
    assert "– population test" in row.marks["population_test"].text()
    assert "○ bootstrap" in row.marks["bootstrap"].text()
    assert "stopped partway" in row.marks["bootstrap"].toolTip()
    assert row.marks["tables"].isVisibleTo(row) and not row.marks["figures"].isVisibleTo(row)
    assert row.run_label == UPDATE and row.run_enabled() and row.fresh_enabled()
    assert row.plan.text() == "Run: keeps fits, fits missing K (k_max 2)"
    assert row.summary() == "results out of date (2 stages)"


def test_the_buttons_follow_where_the_results_stand(row):
    row.state = _state(row.project)  # up to date
    row._show()
    assert row.run_label == RUN and not row.run_enabled() and row.fresh_enabled()
    assert row.summary() == "" and "Nothing to do" in row.plan.text()
    row.state = _state(row.project, has_results=False, up_to_date=False,
                       stages=[Stage(s, NOT_DONE, ["no results yet"]) for s in STAGES])
    row._show()
    assert row.run_label == RUN and row.run_enabled() and not row.fresh_enabled()
    assert row.summary() == ""  # no results: nothing is out of date
    row.state = _state(row.project, can_run=False)  # no [input]: it cannot run
    row._show()
    assert not row.run_enabled() and not row.fresh_enabled()
    # the button keeps its width whatever its words, so the line beside it stays put
    from PySide6.QtWidgets import QPushButton

    assert row.run_button.minimumWidth() >= QPushButton(UPDATE).sizeHint().width()
    layout = row.layout().itemAt(2).layout()  # the button on the left (owner, 2026-11-21)
    assert layout.itemAt(0).widget() is row.run_button


def test_a_project_file_that_does_not_load_is_said_and_runs_nothing(qtbot, tmp_path):
    path = mzml_project(tmp_path)
    path.write_text(path.read_text().replace("[output]", "[output]\noverwrite = true", 1))
    state = work_out(path, slow=False)
    assert "overwrite" in state.problem
    widget = StageRow()
    qtbot.addWidget(widget)
    widget.project, widget.state = path, state
    widget._show()
    assert widget.problem.isVisibleTo(widget) and not widget.marks_widget.isVisibleTo(widget)
    assert widget.problem.text().startswith("The project file does not load: ")
    assert not widget.run_enabled() and not widget.fresh_enabled()


def test_changes_are_gathered_and_nothing_is_checked_during_a_command(qtbot, row):
    row.refresh()
    row.refresh()
    qtbot.waitUntil(lambda: bool(row.started), timeout=5000)
    qtbot.wait(stage_row_module.SETTLE_MS + 100)
    assert row.started == [False]  # one check for two changes close together
    row.pause(True)
    row.refresh(now=True)
    assert row.started == [False] and "A command is running" in row.plan.text()
    row.pause(False)  # ended: worked out again at once
    assert row.started == [False, False]


def test_the_check_runs_off_the_windows_thread_in_two_passes_and_quietly(qtbot, tmp_path,
                                                                         caplog):
    path = mzml_project(tmp_path)
    assert cli.main(["-q", "run", str(path)]) == 0
    path.write_text(path.read_text().replace("n_boot = 3", "n_boot = 4"))
    widget = StageRow()
    qtbot.addWidget(widget)
    passes = []
    import hxex.stages

    real = hxex.stages.standing

    def counted(project, bundle, slow=True):
        passes.append(slow)
        logging.getLogger("hxex.stages").warning("a note the check gives")
        return real(project, bundle, slow=slow)

    handler = logging.Handler()
    seen = []
    handler.emit = lambda record: seen.append(record.getMessage())
    stage_row_module.keep_quiet(handler)
    logging.getLogger("hxex").addHandler(handler)
    try:
        hxex.stages.standing = counted
        widget.show_project(path)
        qtbot.waitUntil(lambda: widget.state is not None and not widget.checking,
                        timeout=120000)
    finally:
        hxex.stages.standing = real
        logging.getLogger("hxex").removeHandler(handler)
    assert passes == [False, True] and widget.state.complete
    assert "a note the check gives" not in seen
    assert [s.name for s in widget.state.late] == ["bootstrap"]
    assert widget.run_label == UPDATE


def test_the_status_bar_says_when_the_results_are_out_of_date(qtbot, tmp_path):
    from PySide6.QtWidgets import QMessageBox

    from hxex.gui.session import WORKSPACES
    from hxex.gui.window import MainWindow

    path = mzml_project(tmp_path)
    assert cli.main(["-q", "run", str(path)]) == 0
    path.write_text(path.read_text().replace("n_boot = 3", "n_boot = 4"))
    window = MainWindow(settings=QSettings(str(tmp_path / "gui.ini"),
                                           QSettings.Format.IniFormat))
    window.ask_to_save = lambda: QMessageBox.StandardButton.Discard  # no dialog in a test
    qtbot.addWidget(window)
    window.open_project(path, load_results=False)
    row = window.run_page.stage_row
    qtbot.waitUntil(lambda: row.state is not None and not row.checking, timeout=120000)
    assert window.out_of_date.isVisibleTo(window)
    assert window.out_of_date.text() == "results out of date (1 stage): Run tab"
    window.out_of_date.click()
    assert window.tabs.currentIndex() == WORKSPACES.index("run")
    assert not window.out_of_date.isVisibleTo(window)  # the row is in view
    assert window.run_page.buttons["run"].text() == UPDATE
