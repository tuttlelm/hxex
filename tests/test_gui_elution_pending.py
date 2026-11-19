# SPDX-License-Identifier: GPL-3.0-or-later
"""Windows saved since the run, in the window: red rows in the Results tree and the List, a
line under them, and the elution tab's Back to the run's window. Runs offscreen."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from hxex.gui.results_workspace import ResultsWorkspace
from hxex.io.elution_windows import save_window
from test_elution_windows import _charges_project
from test_window_edit import PEPTIDE, N, _read, _results, project  # noqa: F401

PEP = f"0001-{N:04d}-{PEPTIDE}"


def _red(item) -> bool:
    return item.data(0, Qt.ItemDataRole.ForegroundRole) is not None


def _leaves(w):
    from hxex.gui.results_workspace import PAYLOAD

    out = {}
    for t in range(w.tree.topLevelItemCount()):
        top = w.tree.topLevelItem(t)
        for j in range(top.childCount()):
            series = top.child(j)
            w.tree.expandItem(series)
            for k in range(series.childCount()):
                point = series.child(k)
                for m in range(point.childCount()):
                    leaf = point.child(m)
                    out[leaf.data(0, PAYLOAD)[3].spectrum_id] = leaf
    return out


def test_a_saved_window_is_red_until_it_is_put_back(qtbot, project):  # noqa: F811
    spectra = _read(project)
    results = _results(spectra)
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, project)
    assert w.pending_label.isHidden() and not _red(w.tree.topLevelItem(0))
    leaves = _leaves(w)
    sid = next(s for s in leaves if "/un2.mzML/" in s)
    i = list(spectra.spectrum_id).index(sid)
    sample, file, ion, _scans = sid.split(":", 1)[1].split("/")
    save_window(project, (sample, file, ion), (spectra.rt_start[i] + 0.02, spectra.rt_end[i]),
                peptide=PEP)
    w.refresh_pending()  # the table watcher's call (a save outside the elution tab)
    top = w.tree.topLevelItem(0)
    series = top.child(0)
    assert _red(top) and _red(series)
    leaves = _leaves(w)
    assert _red(leaves[sid]) and _red(leaves[sid].parent())
    assert not any(_red(leaf) for s, leaf in leaves.items() if s != sid)
    assert "window saved since the run" in leaves[sid].toolTip(0)
    assert not w.pending_label.isHidden()
    assert w.pending_label.text().startswith("1 spectrum has a window saved since the run")
    assert w.list.model.red == {sid}
    # the elution tab puts it back
    w.tree.setCurrentItem(leaves[sid])
    tabs = [w.figures.tabText(i) for i in range(w.figures.count())]
    panel = w.panel("elution")
    w.figures.setCurrentIndex(tabs.index("elution"))
    qtbot.waitUntil(lambda: w._shown_traces is not None and w._shown_traces[1] == sid
                    and not panel.drawing, timeout=30000)
    editor = w.window_editor
    qtbot.waitUntil(editor.back.isEnabled, timeout=30000)
    assert not editor.back.isHidden() and not editor.active  # without editing too
    editor.back.click()
    assert not w.pending and w.pending_label.isHidden() and not _red(w.tree.topLevelItem(0))
    assert not w.list.model.red and editor.undo.isEnabled() and not editor.back.isEnabled()
    editor.undo.click()  # the saved window again
    assert sid in w.pending and editor.back.isEnabled()


def test_a_changed_peptide_row_asks_whole_peptide_or_this_one(qtbot, tmp_path, monkeypatch):
    table = f"sample,file,peptide,ion,rt_start,rt_end\nS,un1.mzML,{PEP},,0.6000,0.6600\n"
    path = _charges_project(tmp_path / "p", table)
    results = _results(_read(path))
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, path)
    z2, z3 = f"{PEP}-z2", f"{PEP}-z3"
    save_window(path, ("S", "un1.mzML", z2), (0.61, 0.65), peptide=PEP, whole=True,
                siblings=(z2, z3))
    w.refresh_pending()
    leaves = _leaves(w)
    pending = sorted(s for s in leaves if _red(leaves[s]))
    assert len(pending) == 2 and all("/un1.mzML/" in s for s in pending)
    sid = next(s for s in pending if f"/{z2}/" in s)
    asked = []

    def answer(followers, fragments):
        asked.append(followers)
        return "ion"

    w.tree.setCurrentItem(leaves[sid])
    tabs = [w.figures.tabText(i) for i in range(w.figures.count())]
    panel = w.panel("elution")
    w.figures.setCurrentIndex(tabs.index("elution"))
    qtbot.waitUntil(lambda: w._shown_traces is not None and w._shown_traces[1] == sid
                    and not panel.drawing, timeout=30000)
    editor = w.window_editor
    monkeypatch.setattr(editor, "_whole_or_ion", answer)
    qtbot.waitUntil(editor.back.isEnabled, timeout=30000)
    editor.back.click()
    assert asked == [[z3]]
    assert sorted(w.pending.by_id) == [s for s in pending if f"/{z3}/" in s]
    editor.undo.click()
    monkeypatch.setattr(editor, "_whole_or_ion", lambda *_a: "whole")
    qtbot.waitUntil(editor.back.isEnabled, timeout=30000)
    editor.back.click()
    assert not w.pending
