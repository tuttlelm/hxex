# SPDX-License-Identifier: GPL-3.0-or-later
"""The window's one parsed project state (audit of 2026-11-15, Q-2 and Q-3), the decision and
file-format logic moved to the library (Q-4, Q-6: the window's Qt-free modules import and work
without Qt), the session's reserved fields (Q-21), and a closed window letting its results go
(M-5). Runs offscreen."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import weakref
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from hxex.gui.project_state import ProjectState
from test_analysis import two_sample_bundle  # noqa: F401 - the fixture
from test_gui import _window

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project(tmp_path, two_sample_bundle):  # noqa: F811
    path = tmp_path / "hxex.toml"
    path.write_text(f'project = "gui"\n\n[output]\nbundle = "{two_sample_bundle.as_posix()}"\n\n'
                    '[analysis]\nreference = "S"\n', encoding="utf-8")
    return path


# --- Q-2: one parsed state ---------------------------------------------------------------------

def test_the_project_state_holds_both_readings_of_the_file(project):
    state = ProjectState.read(project)
    assert state.path == project and state.parsed and state.problem is None
    assert state.value("analysis.reference") == "S" and "analysis" in state.tables
    assert state.loaded is not None and state.analysis.reference == "S"
    assert state.bundle is not None and state.bundle.exists()
    assert state.input is None  # an analysis-only project file
    assert state.of_tables("analysis") == {"values": {"analysis.reference": "S"},
                                           "tables": ["analysis"], "lists": {}}
    # a list of tables' entries count too: a region's residues changed is a change
    project.write_text(project.read_text() + '[[analysis.regions]]\nname = "N"\nstart = 1\n'
                       'end = 5\n', encoding="utf-8")
    regions = ProjectState.read(project).of_tables("analysis")["lists"]
    assert regions == {"analysis.regions": [{"name": "N", "start": 1, "end": 5}]}
    project.write_text(project.read_text().replace("end = 5", "end = 6"), encoding="utf-8")
    assert ProjectState.read(project).of_tables("analysis")["lists"] != regions
    assert state.signature is not None
    assert ProjectState.read(None) == ProjectState()


def test_a_file_that_does_not_load_gives_the_defaults_and_says_so_once(tmp_path, caplog):
    path = tmp_path / "hxex.toml"
    path.write_text('project = "x"\n\n[analysis]\nreference = 3\n', encoding="utf-8")
    with caplog.at_level("WARNING", logger="hxex.gui.project_state"):
        state = ProjectState.read(path)
    assert state.parsed and state.loaded is None and state.problem
    assert state.analysis.reference == ""  # the defaults
    assert state.bundle is None and state.input is None
    assert sum("takes the settings it needs as their defaults" in r.message
               for r in caplog.records) == 1
    missing = ProjectState.read(tmp_path / "none.toml")
    assert missing.text is None and not missing.parsed and "cannot read" in missing.problem


def test_the_window_keeps_one_state_renewed_when_the_file_changes(qtbot, tmp_path, project):
    window = _window(qtbot, tmp_path / "gui.ini")
    assert window.project_state == ProjectState()
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    state = window.project_state
    assert state.path == project and state.analysis.reference == "S"
    # the panels read the window's state, not the file: the same object everywhere
    assert window.analysis_page._settings()[0].reference == "S"
    assert window.results_page._project_state(project) is state
    assert window.project_results
    # the file changed on disk: a panel answering at once reads it again (fresh), and the
    # watcher renews the window's state when it fires
    project.write_text(project.read_text().replace('reference = "S"', 'reference = "V"'))
    fresh = window.fresh_project_state()
    assert fresh is not state and fresh.analysis.reference == "V"
    assert window.project_state is fresh
    qtbot.waitUntil(lambda: window.project_page.editor.values.get("analysis.reference") == "V",
                    timeout=10000)
    qtbot.waitUntil(lambda: window.project_state.value("analysis.reference") == "V",
                    timeout=10000)
    # the run tab's device line comes from the same state
    assert "auto" in window.run_page.device.itemText(0)


def test_project_results_says_whether_the_results_are_the_projects_own(qtbot, tmp_path, project,
                                                                        two_sample_bundle):  # noqa: F811
    window = _window(qtbot, tmp_path / "gui.ini")
    assert not window.project_results
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    assert window.project_results and window.results_page.review.project == project
    window.open_results(two_sample_bundle)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    assert not window.project_results and window.results_page.review.project is None
    assert window.session().results == two_sample_bundle


# --- Q-4 and Q-6: the logic in the library, the window's modules without Qt -------------------

def test_the_left_out_column_is_the_librarys(two_sample_bundle):  # noqa: F811
    from hxex.analysis import AnalysisConfig, left_out_reasons
    from hxex.gui.results_tree import left_out
    from hxex.report.notebook import load_results

    results = load_results(two_sample_bundle)
    labelled = [float(t) for t in results.fit.time if t == t and 0 < t < 1e6]
    config = AnalysisConfig(exclude_times=(labelled[0],))
    reasons = left_out_reasons(results, config)
    assert len(reasons) == len(results.fit)
    assert "time left out (exclude_times)" in reasons
    assert left_out(results, config) == reasons


def test_the_window_uses_the_library_writers():
    from hxex.gui import review, user_tables, window_edit
    from hxex.io import elution_windows
    from hxex.io import user_tables as lib_tables
    from hxex.review_table import make_review_table, review_match, write_decisions

    assert window_edit.save_window is elution_windows.save_window
    assert window_edit.undo_window is elution_windows.undo_window
    assert window_edit.saved_window is elution_windows.saved_window
    assert review.write_decisions is write_decisions
    assert review.make_review_table is make_review_table
    assert user_tables.review_match is review_match
    assert user_tables.read_table is lib_tables.read_table


def test_the_qt_free_modules_work_without_qt(tmp_path):
    """The modules the window shares with Python import and work with PySide6 blocked (audit
    of 2026-11-15, Q-6: ``make_review_table`` needed Qt through the project editor)."""
    (tmp_path / "hxex.toml").write_text('project = "x"\n\n[review]\ntable = "r.csv"\n',
                                        encoding="utf-8")
    (tmp_path / "r.csv").write_text("spectrum_id,review_status,review_K,note\nS/a,,,\n",
                                    encoding="utf-8")
    script = textwrap.dedent(f"""
        import sys
        sys.modules["PySide6"] = None  # import PySide6 raises ImportError
        from pathlib import Path
        from hxex.gui import figures, project_state, results_tree, review, session  # noqa
        from hxex.gui import user_tables, window_edit  # noqa
        from hxex.gui.review import Decision, table_path, write_decisions
        from hxex.gui.user_tables import read_table
        folder = Path({str(tmp_path)!r})
        project = folder / "hxex.toml"
        assert table_path(project) == folder / "r.csv"
        write_decisions(folder / "r.csv", {{"S/a": Decision(status="rejected", note="x")}})
        rows = read_table(folder / "r.csv").rows
        assert rows == [["S/a", "rejected", "", "x"]], rows
        state = project_state.ProjectState.read(project)
        assert state.value("review.table") == "r.csv", state
        try:
            import PySide6  # noqa: F401
        except ImportError:
            print("no qt, as intended")
        else:
            raise SystemExit("PySide6 was imported")
    """)
    # the whole environment, with the source first: a stripped one broke torch's import on
    # Windows (Winsock needs SYSTEMROOT) and matplotlib's cache without HOME (CI, 2026-11-16)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])])}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env=env, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "no qt, as intended" in out.stdout


# --- Q-21: the session's reserved fields ------------------------------------------------------

def test_the_session_leaves_its_reserved_fields_empty(qtbot, tmp_path, project):
    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    session = window.session()
    assert session.analysis is None and session.figures == []  # reserved, never written


# --- M-5: a closed window lets its results go --------------------------------------------------

def test_a_closed_window_lets_its_results_and_figures_go(qtbot, tmp_path, project):
    from PySide6.QtWidgets import QApplication

    window = _window(qtbot, tmp_path / "gui.ini")
    window.open_project(project)
    qtbot.waitUntil(lambda: not window.loading, timeout=30000)
    window.show()
    results = window.results_page.results
    assert results is not None
    page = window.results_page
    page.tree.setCurrentItem(page.tree.topLevelItem(0))  # a series: its figures drawn
    name = page.figures.tabText(page.figures.currentIndex()).lower() if page.figures.count() \
        else ""
    panel = page.panels.get(name)
    held = [weakref.ref(results)]
    if panel is not None:
        qtbot.waitUntil(lambda: panel.view.figure is not None, timeout=60000)
        held.append(weakref.ref(panel.view.figure))
    results = page = panel = None
    assert window.close()
    QApplication.processEvents()
    # background work the window started (a drawing, Qt's global pool) holds the results in
    # its arguments until it ends: waited for, then its callbacks delivered (macOS CI,
    # 2026-11-23: the results still held once, the figure gone)
    from PySide6.QtCore import QThreadPool

    from hxex.gui import tasks

    for pool in (tasks._DRAWING, QThreadPool.globalInstance()):
        if pool is not None:
            assert pool.waitForDone(120000)
    QApplication.processEvents()
    gc.collect()
    alive = [ref() for ref in held if ref() is not None]
    if alive:  # once on macOS and once on Ubuntu CI (2026-11-23/24), never here: say who holds
        report = "\n".join(f"{type(obj).__name__} still held by:\n{_holders(obj)}"
                            for obj in alive)
        del alive
        for _ in range(3):  # a delivery that posted another event: released by more rounds?
            QApplication.processEvents()
        gc.collect()
        later = [type(ref()).__name__ for ref in held if ref() is not None]
        pytest.fail(f"{report}\nafter three more rounds of events still held: {later or 'none'}")


def _holders(obj, depth: int = 6) -> str:
    """Who refers to ``obj``, a few levels up: types, dict keys and their owners, closures
    with their functions, frames with their place (test frames left out)."""
    import types

    lines, seen = [], set()

    def walk(target, level):
        if level == depth:
            return
        for r in gc.get_referrers(target):
            if id(r) in seen or isinstance(r, types.FrameType) and r.f_code.co_name in (
                    "walk", "_holders", "test_a_closed_window_lets_its_results_and_figures_go"):
                continue
            seen.add(id(r))
            text = type(r).__name__
            if isinstance(r, dict):
                owners = [o for o in gc.get_referrers(r) if getattr(o, "__dict__", None) is r]
                text += f" {[k for k, v in r.items() if v is target]!r}" + (
                    f" of {type(owners[0]).__name__}" if owners else "")
            elif isinstance(r, types.CellType):
                text += " of " + ", ".join(
                    f"{f.__qualname__} ({f.__code__.co_filename}:{f.__code__.co_firstlineno})"
                    for f in gc.get_referrers(r) if isinstance(f, types.FunctionType))
            elif isinstance(r, types.FrameType):
                text += f" {r.f_code.co_name} ({r.f_code.co_filename}:{r.f_lineno})"
            elif isinstance(r, types.FunctionType):
                text += f" {r.__qualname__}"
            elif isinstance(r, types.MethodType):
                text += f" {r.__func__.__qualname__} of {type(r.__self__).__name__}"
            elif isinstance(r, (list, tuple)):
                text += f" of {len(r)}"
            lines.append("  " * level + "<- " + text)
            walk(r, level + 1)
            if len(lines) > 60:
                return

    walk(obj, 0)
    return "\n".join(lines[:60])
