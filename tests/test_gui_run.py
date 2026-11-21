# SPDX-License-Identifier: GPL-3.0-or-later
"""Running hxex's commands from the window (hxex.gui.runner, hxex.gui.run_workspace): real
commands on a synthetic project, and a stand-in command to test Stop, the abort and the kill.
Runs offscreen."""
import os
import sys

import pytest

pytest.importorskip("PySide6")

from hxex.cli import EXIT_STOPPED
from hxex.gui import runner as runner_module
from hxex.gui.run_workspace import RunWorkspace
from hxex.gui.runner import CommandRun, outcome, shown_command
from test_cli import mzml_project

# a stand-in for hxex: a line on stdout, a progress record, then waits; obeys "stop" lines
# unless told to be stubborn
STAND_IN = r'''
import json, os, sys, threading, time
stubborn = "stubborn" in sys.argv
print("working", flush=True)
sys.stderr.write("@hxex-progress " + json.dumps({"event": "progress", "stage": "fit",
    "fraction": 0.5, "overall": 0.25, "elapsed": 1.0, "left": 3.0}) + "\n")
sys.stderr.flush()
stops = []
def watch():
    for line in sys.stdin:
        if line.strip() == "stop":
            stops.append(1)
            print(f"stop {len(stops)} received", flush=True)
threading.Thread(target=watch, daemon=True).start()
for _ in range(1200):
    time.sleep(0.05)
    if stops and not stubborn:
        print("stopped", flush=True)
        os._exit(130)  # as hxex does: a thread waiting on the input holds sys.exit on Windows
os._exit(0)
'''


@pytest.fixture
def stand_in(tmp_path):
    script = tmp_path / "stand_in.py"
    script.write_text(STAND_IN)
    return script


def test_the_command_reads_as_it_would_be_typed():
    assert shown_command(["run", "hxex.toml", "--device", "cuda"]) == \
        "hxex run hxex.toml --device cuda"
    assert shown_command(["check", "my project.toml"]) == 'hxex check "my project.toml"'
    assert outcome(0) == "done" and outcome(EXIT_STOPPED) == "stopped"
    assert "killed" in outcome(9, crashed=True)


def test_stop_stops_the_command_after_its_step(qtbot, stand_in, tmp_path):
    run = CommandRun(program=[sys.executable, str(stand_in)])
    lines, records = [], []
    run.line.connect(lambda text, which: lines.append(text))
    run.progress.connect(records.append)
    run.start(["run", "hxex.toml"], tmp_path)
    qtbot.waitUntil(lambda: "working" in lines, timeout=20000)
    # the progress record comes on the error stream, which may arrive after the output's line
    qtbot.waitUntil(lambda: bool(records), timeout=20000)
    assert records[0]["stage"] == "fit"  # not shown as a line
    assert not any(line.startswith("@hxex-progress") for line in lines)
    with qtbot.waitSignal(run.finished, timeout=20000) as done:
        assert run.stop() == 1
    assert done.args == [EXIT_STOPPED, "stopped"] and "stop 1 received" in lines


def test_a_second_stop_aborts_and_a_stubborn_command_is_killed(qtbot, stand_in, tmp_path,
                                                                monkeypatch):
    monkeypatch.setattr(runner_module, "GRACE", 0.5)
    run = CommandRun(program=[sys.executable, str(stand_in), "stubborn"])
    lines = []
    run.line.connect(lambda text, which: lines.append(text))
    run.start(["run", "hxex.toml"], tmp_path)
    qtbot.waitUntil(lambda: "working" in lines, timeout=20000)
    with qtbot.waitSignal(run.finished, timeout=20000) as done:
        run.stop()
        qtbot.waitUntil(lambda: "stop 1 received" in lines, timeout=10000)
        assert run.stop() == 2  # the abort; it ignores both, so it is killed
    assert not run.running and "killed" in done.args[1]


@pytest.fixture
def workspace(qtbot, tmp_path):
    page = RunWorkspace()
    qtbot.addWidget(page)
    page.show_project(mzml_project(tmp_path))
    return page


def test_the_run_workspace_runs_hxex_and_asks_how_to_run_again(qtbot, workspace,
                                                                            tmp_path):
    page = workspace
    assert all(b.isEnabled() for b in page.buttons.values()) and not page.stop_button.isEnabled()
    with qtbot.waitSignal(page.runner.finished, timeout=120000):
        page.buttons["check"].click()
        assert page.command.text() == "hxex check hxex.toml"
        assert not page.buttons["run"].isEnabled()  # one command at a time
    assert "hxex check hxex.toml: done" in page.console.toPlainText()
    # the whole output in a log beside the project, with the command and how it ended
    log = page.log_path
    assert log.parent == tmp_path / "hxex_logs" and log.name.startswith("hxex_check_")
    text = log.read_text()
    assert text.startswith("$ hxex check hxex.toml\nin ") and "hxex " in text.splitlines()[2]
    assert "[hxex check hxex.toml: done, after " in text and "@hxex-progress" not in text
    with qtbot.waitSignal(page.results_changed, timeout=300000):
        page.buttons["run"].click()
    assert page.bar.value() == 1000 and (tmp_path / "out").exists()
    assert "@hxex-progress" not in page.console.toPlainText()
    # the total run time beside the bar, kept when the command has ended (owner, 2026-11-05)
    assert page.elapsed.text().startswith("total ") and not page._clock.isActive()
    # again: the window shows where the results stand and asks (docs/PLAN.md Next steps 40)
    asked = []

    def answer(rows, up_to_date):
        asked.append((rows, up_to_date))
        return "fresh"

    page.ask_run = answer
    with qtbot.waitSignal(page.results_changed, timeout=300000):
        page.buttons["run"].click()
    assert asked[0][1] and ("stages", "read: done") in asked[0][0]
    assert page.command.text() == "hxex run hxex.toml --fresh"
    # a change: Bring up to date runs hxex run alone
    page.project.write_text(page.project.read_text().replace("n_boot = 3", "n_boot = 4"))
    page.ask_run = lambda rows, up_to_date: asked.append((rows, up_to_date)) or "update"
    with qtbot.waitSignal(page.results_changed, timeout=300000):
        page.buttons["run"].click()
    assert not asked[-1][1] and any("bootstrap: out of date" in text for _, text in asked[-1][0])
    assert page.command.text() == "hxex run hxex.toml"
    # Cancel: nothing runs; a table changed by hand is named in the question (the run would
    # stop at it: owner, 2026-11-20)
    uptake = tmp_path / "out_tables" / "uptake.csv"
    uptake.write_text(uptake.read_text() + "a note\n")
    page.ask_run = lambda rows, up_to_date: asked.append((rows, up_to_date)) and None
    page.buttons["run"].click()
    assert not page.runner.running
    assert any(key == "outputs" and "uptake.csv" in text and "changed since hxex wrote" in text
               for key, text in asked[-1][0])


def test_closing_during_a_command_asks_and_stops_it(qtbot, stand_in, tmp_path):
    page = RunWorkspace(program=[sys.executable, str(stand_in)])
    qtbot.addWidget(page)
    project = tmp_path / "hxex.toml"
    project.write_text("")
    page.show_project(project)
    page.run_command("run")
    qtbot.waitUntil(lambda: "working" in page.console.toPlainText(), timeout=20000)
    # the progress line comes on the error stream, which may arrive after the output's line
    qtbot.waitUntil(lambda: page.state.text() == "fit: 50%, about 3s left", timeout=20000)
    assert page.stop_button.isEnabled()
    page.ask_stop_to_close = lambda: False
    assert not page.may_close() and page.runner.running
    page.ask_stop_to_close = lambda: True
    assert page.may_close() and not page.runner.running
    # the log ends with how the command ended
    assert "stopped" in page.log_path.read_text().splitlines()[-1]


def test_the_device_list_names_the_project_setting_and_opens_it(qtbot, workspace, tmp_path):
    """The first entry says where the device is set and what it is (owner: 'as the project
    says' did not say where); Change… opens that setting."""
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox

    from hxex.gui.window import MainWindow

    page = workspace
    assert page.device.itemText(0) == "[run.fit.backend] device: cpu"  # the project's
    window = MainWindow(settings=QSettings(str(tmp_path / "g.ini"), QSettings.Format.IniFormat))
    window.ask_to_save = lambda: QMessageBox.StandardButton.Discard
    qtbot.addWidget(window)
    project = page.project
    window.open_asking(project)
    run = window.run_page
    assert run.device.itemText(0) == "[run.fit.backend] device: cpu"
    text = project.read_text().replace('device = "cpu"', 'device = "cuda:1"')
    project.write_text(text)
    window.project_page.editor._file_changed(str(project))  # as the watcher reports it
    assert run.device.itemText(0) == "[run.fit.backend] device: cuda:1"
    window.tabs.setCurrentIndex(1)
    run.show_setting.emit("run.fit.backend")
    assert window.tabs.currentIndex() == 0
    assert window.project_page.editor.list.currentItem().text() == "[run.fit.backend]"


def test_the_console_wraps_long_lines_unless_told_not_to(workspace):
    from PySide6.QtWidgets import QPlainTextEdit

    page = workspace
    assert page.wrap.isChecked()
    assert page.console.lineWrapMode() == QPlainTextEdit.LineWrapMode.WidgetWidth
    page.wrap.setChecked(False)
    assert page.console.lineWrapMode() == QPlainTextEdit.LineWrapMode.NoWrap


def test_reopening_a_project_shows_its_last_commands_log(qtbot, stand_in, tmp_path):
    """The console shows the project's last command, and the earlier ones are listed (owner,
    2026-11-08: reopened, the Run tab showed nothing of how the last command went); another
    project file's logs in the same folder are not."""
    project = tmp_path / "hxex.toml"
    project.write_text("")
    other = tmp_path / "hxex_other.toml"
    other.write_text("")
    folder = tmp_path / "hxex_logs"
    folder.mkdir()
    # a log named before logs were named by project: its first line tells the project
    (folder / "run_2026-11-07_101500.log").write_text(
        "$ hxex run hxex.toml\nin x\nstarted 2026-11-07 10:15:00, hxex 0\n\nfitting\n\n"
        "[hxex run hxex.toml: stopped, after 31m 17s, 2026-11-07 10:46:17]\n")
    (folder / "hxex_other_check_2026-11-07_120000.log").write_text(
        "$ hxex check hxex_other.toml\nin x\nstarted 2026-11-07 12:00:00, hxex 0\n\n")
    page = RunWorkspace(program=[sys.executable, str(stand_in)])
    qtbot.addWidget(page)
    page.show_project(project)
    text = page.console.toPlainText()
    assert "an earlier command: hxex run hxex.toml, started 7 Nov 2026 10:15: stopped after " \
        "31m 17s" in text and "fitting" in text
    assert [page.history.itemText(i) for i in range(page.history.count())] == [
        "run, 7 Nov 10:15: stopped after 31m 17s"]
    page.run_command("check")
    assert not page.history.isEnabled()  # not while a command runs
    qtbot.waitUntil(lambda: "working" in page.console.toPlainText(), timeout=20000)
    assert "fitting" not in page.console.toPlainText()  # the new command's output alone
    with qtbot.waitSignal(page.runner.finished, timeout=20000):
        page.stop()
    assert page.log_path.name.startswith("hxex_check_")
    assert page.history.count() == 2 and page.history.itemText(0).startswith("check, ")
    assert "stopped after" in page.history.itemText(0) and page.history.isEnabled()
    page.show_project(other)
    assert "hxex_other.toml" in page.console.toPlainText() and page.history.count() == 1
    assert page.history.itemText(0).endswith("no end recorded (the window closed during it?)")


def test_a_running_command_shows_outside_the_run_tab(qtbot, stand_in, tmp_path):
    """The Run tab's title and colour, and a line in the status bar that opens the Run tab,
    while a command runs; outside the Run tab its title says how it ended until it is opened
    (owner, 2026-11-09)."""
    from hxex.gui.session import WORKSPACES
    from test_gui import _window

    w = _window(qtbot, tmp_path / "gui.ini")
    page = w.run_page
    page.runner.program = [sys.executable, str(stand_in)]
    project = tmp_path / "hxex.toml"
    project.write_text("")
    page.show_project(project)
    run_tab = WORKSPACES.index("run")
    w.tabs.setCurrentIndex(WORKSPACES.index("results"))
    page.run_command("run")
    qtbot.waitUntil(lambda: "fit: 50%" in w.running.text(), timeout=20000)
    assert w.tabs.tabText(run_tab) == "Running" and not w.running.isHidden()
    assert w.running.text().startswith("running: hxex run hxex.toml: fit: 50%")
    assert w.tabs.tabBar().tabTextColor(run_tab).isValid()
    w.running.click()  # opens the Run tab
    assert w.tabs.currentIndex() == run_tab
    w.tabs.setCurrentIndex(WORKSPACES.index("results"))
    with qtbot.waitSignal(page.runner.finished, timeout=20000):
        page.stop()
    assert w.running.isHidden() and w.tabs.tabText(run_tab) == "Run (stopped)"
    assert not w.tabs.tabBar().tabTextColor(run_tab).isValid()
    w.tabs.setCurrentIndex(run_tab)  # seen: the title is the usual one again
    assert w.tabs.tabText(run_tab) == "Run"


def test_a_command_that_ends_for_another_project_asks_nothing(qtbot, stand_in, tmp_path):
    """New session or New project file... switched the project under a running command, and
    its refusal to overwrite then asked to rerun it on the new project (audit of 2026-11-15,
    A-3)."""
    page = RunWorkspace(program=[sys.executable, str(stand_in)])
    qtbot.addWidget(page)
    a, b = tmp_path / "a" / "hxex.toml", tmp_path / "b" / "hxex.toml"
    for p in (a, b):
        p.parent.mkdir()
        p.write_text("")
    page.show_project(a)
    page.run_command("run")
    qtbot.waitUntil(lambda: "working" in page.console.toPlainText(), timeout=20000)
    page.show_project(b)
    asked, changed = [], []
    page.ask_output_exists = lambda *a: (asked.append(1), "overwrite")[1]
    page.results_changed.connect(lambda: changed.append(1))
    page._tail.append("run again with 'hxex run --overwrite' (or --keep-previous)")
    with qtbot.waitSignal(page.runner.finished, timeout=20000):
        page.stop()
    assert not asked and not changed and not page.runner.running
    assert page.command.text() == "hxex run hxex.toml"  # not run again on b


def test_closing_with_a_stubborn_command_keeps_the_window_alive_and_kills_it(
        qtbot, stand_in, monkeypatch, tmp_path):
    """The close waited 45 s in the window's thread with nothing painted, then aborted and
    killed a command whose step had not ended, which the question did not say (audit of
    2026-11-15, A-5): the window's events are handled while it waits, and the question says
    what follows."""
    import time

    from PySide6.QtCore import QTimer

    from hxex.gui import run_workspace as module

    monkeypatch.setattr(module, "STOP_WAIT_MS", 400)
    monkeypatch.setattr(module, "ABORT_WAIT_MS", 400)
    monkeypatch.setattr(module, "KILL_WAIT_MS", 4000)
    monkeypatch.setattr(runner_module, "GRACE", 0.3)
    page = RunWorkspace(program=[sys.executable, str(stand_in), "stubborn"])
    qtbot.addWidget(page)
    project = tmp_path / "hxex.toml"
    project.write_text("")
    page.show_project(project)
    page.run_command("run")
    qtbot.waitUntil(lambda: "working" in page.console.toPlainText(), timeout=20000)
    ticks = []
    timer = QTimer(page)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(20)
    page.ask_stop_to_close = lambda: True
    began = time.monotonic()
    assert page.may_close()
    assert time.monotonic() - began < 8 and not page.runner.running
    assert len(ticks) > 5  # the window's events were handled while it waited
    assert "killed" in page.console.toPlainText() and not page._closing
    page.ask_stop_to_close = RunWorkspace.ask_stop_to_close.__get__(page)
    text = page.ask_stop_to_close.__func__.__code__.co_consts
    assert any("is aborted, and killed" in str(c) for c in text)


# --- fix Session 5 (audit of 2026-11-15) ---------------------------------------------------------

# a stand-in that prints the message in message.txt beside it and fails
REFUSER = r'''
import pathlib, sys
print(pathlib.Path(__file__).with_name("message.txt").read_text(), flush=True)
sys.exit(1)
'''


def test_the_replace_question_comes_only_for_the_refusal_of_a_replaceable_command(
        qtbot, tmp_path):
    """A-4: a check printing its outputs block names --keep-previous and must not get the
    question; only run, report and analyse refusing to write over output edited by hand ask
    (keep it as NAME.N; owner, 2026-11-20), with the refusal's lines naming the files, and
    after the finished slot has returned (A-16). Kept, the command runs again with its own
    options and --keep-previous."""
    from hxex.io.outputs import _write_listing, edited_refusal

    folder = tmp_path / "out_tables"
    folder.mkdir()
    (folder / "spectra.csv").write_text("a\n")
    _write_listing(folder, ["spectra.csv"])
    (folder / "spectra.csv").write_text("b\n")
    refusal = "hxex: error: " + edited_refusal(folder)
    script = tmp_path / "refuser.py"
    script.write_text(REFUSER)
    message = tmp_path / "message.txt"
    page = RunWorkspace(program=[sys.executable, str(script)])
    qtbot.addWidget(page)
    project = tmp_path / "hxex.toml"
    project.write_text('project = "p"\n')
    page.show_project(project)
    asked = []
    page.ask_output_exists = lambda *a: asked.append(a) or None
    cases = [
        ("check", "each command writes its folder again (hxex run brings the bundle up to "
                  "date); --keep-previous keeps the earlier one as NAME.N", False),
        ("report", refusal, True),
        ("analyse", refusal, True),
        ("run", refusal, True),
    ]
    for command, text, asks in cases:
        message.write_text(text)
        asked.clear()
        with qtbot.waitSignal(page.runner.finished, timeout=30000):
            page.run_command(command, ("--fresh",) if command == "run" else ())
        assert asked == [], command  # never inside the finished slot (A-16)
        if asks:
            qtbot.waitUntil(lambda: len(asked) == 1, timeout=5000)
            asked_command, detail = asked[0]
            assert asked_command == command
            assert "spectra.csv changed after hxex wrote it" in detail
        else:  # the question is asked from a zero timer: delivered, not waited for (Q-29)
            from PySide6.QtWidgets import QApplication

            QApplication.processEvents()
            QApplication.processEvents()
            assert asked == [], command
    # kept as NAME.N: the run again, with its own options and --keep-previous
    answers = ["keep-previous"]
    page.ask_output_exists = lambda *a: answers.pop() if answers else None
    with qtbot.waitSignal(page.runner.finished, timeout=30000):
        page._ask_replace("run")
    assert page.args[-2:] == ["--fresh", "--keep-previous"]
    # not asked while the window is closing or the project changed under it
    asked.clear()
    page.ask_output_exists = lambda *a: asked.append(a) or None
    page._closing = True
    page._ask_replace("report")
    page._closing = False
    page.show_project(tmp_path / "other.toml")
    page._ask_replace("report")
    assert asked == []


def test_a_finished_command_lets_its_process_go(qtbot, stand_in, tmp_path):
    """A-8: one QProcess per command accumulated as a child of the runner; now the next
    command lets the finished one go (outside its own signals: a deferred deletion crashed a
    runner collected before it ran)."""
    from PySide6.QtCore import QProcess

    page = RunWorkspace(program=[sys.executable, str(stand_in)])
    qtbot.addWidget(page)
    project = tmp_path / "hxex.toml"
    project.write_text('project = "p"\n')
    page.show_project(project)
    for _ in range(3):
        with qtbot.waitSignal(page.runner.finished, timeout=30000):
            page.run_command("check")
            qtbot.waitUntil(lambda: "working" in page.console.toPlainText(), timeout=10000)
            page.stop()
    assert len(page.runner.findChildren(QProcess)) == 1  # the last one only


@pytest.mark.skipif(os.name == "nt", reason="Windows keeps an open log's folder")
def test_a_log_folder_removed_during_a_command_is_said(qtbot, stand_in, tmp_path):
    """A-9: the lines went to the unlinked file and the console still named the log."""
    import shutil

    page = RunWorkspace(program=[sys.executable, str(stand_in)])
    qtbot.addWidget(page)
    project = tmp_path / "hxex.toml"
    project.write_text('project = "p"\n')
    page.show_project(project)
    with qtbot.waitSignal(page.runner.finished, timeout=30000):
        page.run_command("check")
        qtbot.waitUntil(lambda: "working" in page.console.toPlainText(), timeout=10000)
        shutil.rmtree(tmp_path / "hxex_logs")
        page.stop()
    assert "is gone" in page.console.toPlainText() and page.log_path is None
    assert page.history.itemText(0) == "none yet"


def test_a_log_that_cannot_be_written_any_more_is_said_once(qtbot, stand_in, tmp_path):
    """A-17: a disk that fills during a command raised in the slot on every line."""
    page = RunWorkspace(program=[sys.executable, str(stand_in)])
    qtbot.addWidget(page)
    project = tmp_path / "hxex.toml"
    project.write_text('project = "p"\n')
    page.show_project(project)

    class Full:
        def write(self, text):
            raise OSError(28, "No space left on device")

        def flush(self):
            pass

        def close(self):
            pass

    with qtbot.waitSignal(page.runner.finished, timeout=30000):
        page.run_command("check")
        qtbot.waitUntil(lambda: "working" in page.console.toPlainText(), timeout=10000)
        page.log_file = Full()
        page.stop()
    text = page.console.toPlainText()
    assert text.count("cannot be written any more") == 1 and "stopped" in text
    assert page.log_file is None and page.log_path is None
