# SPDX-License-Identifier: GPL-3.0-or-later
"""Running hxex's commands from the window (hxex.gui.runner, hxex.gui.run_workspace): real
commands on a synthetic project, and a stand-in command to test Stop, the abort and the kill.
Runs offscreen."""
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from hxex.cli import EXIT_STOPPED  # noqa: E402
from hxex.gui import runner as runner_module  # noqa: E402
from hxex.gui.run_workspace import RunWorkspace  # noqa: E402
from hxex.gui.runner import CommandRun, outcome, shown_command  # noqa: E402
from test_cli import mzml_project  # noqa: E402

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


def test_the_run_workspace_runs_hxex_and_offers_to_replace_earlier_results(qtbot, workspace,
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
    # again: hxex refuses to write over the bundle; the window offers to replace it
    asked = []

    def answer():
        asked.append(1)
        return "overwrite"

    page.ask_output_exists = answer
    with qtbot.waitSignal(page.results_changed, timeout=300000):
        page.buttons["run"].click()
    assert asked == [1] and page.command.text() == "hxex run hxex.toml --overwrite"


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
    assert w.tabs.tabText(run_tab) == "Run (running)" and not w.running.isHidden()
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
