# SPDX-License-Identifier: GPL-3.0-or-later
"""The elution tab's window editing (hxex.gui.window_editor; PLAN Next steps 26): the switch,
the boxes, a drag across the trace, the preview beside the spectrum, Save to the elution-window
table and Undo. Runs offscreen."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from hxex.gui.results_workspace import ResultsWorkspace  # noqa: E402
from test_window_edit import _read, _results, project  # noqa: E402, F401


def _titles(panel):
    fig = panel.view.figure
    return [] if fig is None else [ax.get_title() for ax in fig.axes]


def test_a_window_is_edited_previewed_saved_and_undone(qtbot, project):  # noqa: F811
    spectra = _read(project)
    results = _results(spectra)
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, project)
    series = w.tree.topLevelItem(0).child(0)
    w.tree.expandItem(series)
    with qtbot.waitSignal(w.panel("fits at each K").drawn, timeout=30000):
        w.tree.setCurrentItem(series.child(0).child(1))  # UN, rep 2
    tabs = [w.figures.tabText(i) for i in range(w.figures.count())]
    panel = w.panel("elution")
    w.figures.setCurrentIndex(tabs.index("elution"))
    qtbot.waitUntil(lambda: w._shown_traces is not None and not panel.drawing
                    and len(w._shown_traces[3].traces) == 2, timeout=30000)
    editor = w.window_editor
    assert not editor.save.isVisible()  # the controls show with the switch
    selected = w._shown_traces[3].traces[0]
    editor.edit.setChecked(True)
    assert editor.window() == pytest.approx(selected.window, abs=1e-3)
    assert editor.drift() is None  # no ion mobility here
    qtbot.waitUntil(lambda: any(t.startswith("new window") for t in _titles(panel))
                    and not panel.drawing, timeout=30000)
    # a drag across the trace sets the window (figure fractions, as the view gives them)
    fig = panel.view.figure
    ax = next(a for a in fig.axes if a.get_xlabel().startswith("retention time"))
    start, end = selected.window[0] + 0.02, selected.window[1] - 0.01
    to_fig = fig.transFigure.inverted()
    (x0, y0), (x1, y1) = (to_fig.transform(ax.transData.transform((t, 0.0)))
                          for t in (start, end))
    panel.spanned.emit(x0, y0 + 0.01, x1, y1 + 0.01)
    assert editor.window() == pytest.approx((start, end), abs=2e-3)
    qtbot.waitUntil(lambda: editor.failure is not None or (
        editor._previewed is not None and editor._previewed[1] == editor.window()),
        timeout=60000)
    assert editor.failure is None
    assert editor.preview.k is not None  # fitted (it was drawn picked first)
    # what the preview is: above the figure in the window's font, not inside the figure
    assert editor.notes.text().startswith("Preview: the results change only")
    assert not any("preview" in t.get_text().lower() for t in panel.view.figure.texts)
    assert editor.preview.spectrum.spectrum_id != w._shown_traces[1]
    table = project.parent / "rt_elution_windows.csv"
    before = project.read_bytes()
    assert editor.save.isEnabled() and editor.save.text() == "Save window"
    messages = []
    editor.message.connect(messages.append)
    editor.save.click()
    assert "un2.mzML" in table.read_text() and "hxex run --overwrite" in messages[-1]
    assert "rt_elution_windows" in project.read_text()
    assert editor.undo.isEnabled()
    editor.undo.click()
    assert not table.exists() and project.read_bytes() == before
    assert not editor.undo.isEnabled()
    # switched off: the plain trace again, dragging moves the figure
    editor.edit.setChecked(False)
    assert not panel.span_mode
    qtbot.waitUntil(lambda: not panel.drawing and len(panel.view.figure.axes) == 1,
                    timeout=30000)
