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


def test_a_window_is_edited_previewed_saved_and_undone(qtbot, project, monkeypatch):  # noqa: F811
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
    # the editing controls are rows of their own: in the buttons' row they made the window
    # wider than the screen (2,061 px; owner, 2026-11-08); and each row wraps, so the panel is
    # narrower than the editing row on one line (a fixed 1,000 px failed with Windows' fonts:
    # 1,102; widths compared within the window)
    one_line = sum(x.sizeHint().width() for x in (editor.edit, *editor.controls))
    assert panel.minimumSizeHint().width() < one_line
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
    # the raw file is read, then the spectrum fitted: said on the status line with the busy
    # bar, through the redraws in between (owner, 2026-11-08)
    assert panel._task == "Reading the new window's scans…" and not panel.busy.isHidden()
    qtbot.waitUntil(lambda: editor.failure is not None or (
        editor._previewed is not None and editor._previewed[1] == editor.window()),
        timeout=60000)
    assert editor.failure is None
    qtbot.waitUntil(lambda: not panel.drawing, timeout=30000)
    assert panel._task is None and panel.status.full() == "" and panel.busy.isHidden()
    assert editor.preview.k is not None  # fitted (it was drawn picked first)
    # what the preview is: above the figure in the window's font, not inside the figure
    assert editor.notes.text() == ("Preview only: results will change upon rerun with "
                                   "overwrite enabled.")  # one line, the same in every draw
    assert "from the run's fit" not in editor.notes.toolTip()  # how it was fitted: not said
    layout = panel.layout()  # the line above the figure's title (owner, 2026-11-12)
    assert layout.indexOf(editor.notes) + 1 == layout.indexOf(panel.title)
    assert not any("preview" in t.get_text().lower() for t in panel.view.figure.texts)
    assert editor.preview.spectrum.spectrum_id != w._shown_traces[1]
    table = project.parent / "rt_elution_windows.csv"
    before = project.read_bytes()
    assert editor.save.isEnabled() and editor.save.text() == "Save window"
    messages = []
    editor.message.connect(messages.append)
    # saved for the whole peptide by default (its ions elute together), or for the ion only
    assert editor.scope.currentData() == "peptide"
    editor.save.click()
    assert "un2.mzML" in table.read_text() and "hxex run --overwrite" in messages[-1]
    assert "with overwrite enabled" in messages[-1]
    assert "(every charge state)" in messages[-1] and "for sample S" in messages[-1]
    assert editor.scope.currentText() == "every charge state of this peptide, in this file"
    row = table.read_text().splitlines()[1].split(",")
    assert row[2] == "0001-0008-VDQIIMAK" and row[3] == ""  # the peptide's row
    assert "rt_elution_windows" in project.read_text()
    editor.scope.setCurrentIndex(editor.scope.findData("ion"))
    editor.save.click()
    assert "z = 2 only" in messages[-1]
    assert "0001-0008-VDQIIMAK-z2" in table.read_text()
    assert editor.undo.isEnabled()
    editor.undo.click()
    editor.undo.click()
    assert not table.exists() and project.read_bytes() == before
    assert not editor.undo.isEnabled()
    # spread to the replicates (the sample's other file, un1, shifted by its elution): un1 had
    # a window set before, so the window asks; kept, un1 is left as it was (owner, 2026-11-11)
    from hxex.gui import window_editor
    from hxex.gui.window_edit import save_window

    other = next(s for s in spectra.spectrum_id if "/un1.mzML/" in s)
    k = list(spectra.spectrum_id).index(other)
    own = save_window(project, ("S", "un1.mzML", other.split("/")[2]),
                      (spectra.rt_start[k], spectra.rt_end[k]))
    asked = []

    class Box:  # the replace-or-keep question, answered "Keep them"
        Icon = window_editor.QMessageBox.Icon
        ButtonRole = window_editor.QMessageBox.ButtonRole
        StandardButton = window_editor.QMessageBox.StandardButton

        def __init__(self, _icon, _title, text, parent=None):
            asked.append(text)
            self.buttons = []

        def addButton(self, label, _role=None):
            self.buttons.append(label)
            return label

        def exec(self):
            return 0

        def clickedButton(self):
            return next(b for b in self.buttons if b == "Keep them")

    monkeypatch.setattr(window_editor, "QMessageBox", Box)
    editor.scope.setCurrentIndex(editor.scope.findData("replicates"))
    assert "replicates (each shifted)" in editor.scope.currentText()
    unchanged = table.read_bytes()
    editor.save.click()
    qtbot.waitUntil(lambda: messages and "kept their own window" in messages[-1], timeout=30000)
    assert asked and "un1.mzML" in asked[0]
    assert "un1.mzML" in messages[-1] and editor.save.isEnabled()
    rows = table.read_text().splitlines()
    assert len(rows) == 3 and table.read_bytes() != unchanged  # un2's peptide row added only
    editor.undo.click()
    assert table.read_bytes() == unchanged
    from hxex.gui.window_edit import undo_window

    undo_window(own)
    # switched off: the plain trace again, dragging moves the figure
    editor.edit.setChecked(False)
    assert not panel.span_mode
    qtbot.waitUntil(lambda: not panel.drawing and len(panel.view.figure.axes) == 1,
                    timeout=30000)


def test_compare_with_the_peptides_other_charge_states_in_this_file(qtbot, tmp_path):
    """Compare with: this peptide's other charge states in this file draws its other charge
    state behind the selected trace; switched back, its replicates. With Edit window on,
    choosing how to save switches Compare with to that level's view (one way only)."""
    from test_elution_windows import _charges_project

    path = _charges_project(tmp_path / "charges")
    results = _results(_read(path))
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, path)
    series = w.tree.topLevelItem(0).child(0)
    w.tree.expandItem(series)
    with qtbot.waitSignal(w.panel("fits at each K").drawn, timeout=30000):
        w.tree.setCurrentItem(series.child(0).child(1))  # UN, rep 2
    tabs = [w.figures.tabText(i) for i in range(w.figures.count())]
    panel = w.panel("elution")
    w.figures.setCurrentIndex(tabs.index("elution"))
    qtbot.waitUntil(lambda: w._shown_traces is not None and not panel.drawing
                    and len(w._shown_traces[3].traces) == 2, timeout=30000)
    w.compare.setCurrentIndex(w.compare.findData("ions"))
    qtbot.waitUntil(lambda: w._shown_traces is not None and not panel.drawing
                    and len(w._shown_traces[3].ions) == 1, timeout=30000)
    shown = w._shown_traces[3]
    assert len(shown.traces) == 1 and shown.ions[0].label.endswith("(MS1)")
    assert any(ln.get_gid() and ln.get_gid().startswith("ion:")
               for ln in panel.view.figure.axes[0].lines)
    assert w.compare.currentText() == "this peptide's other charge states in this file"
    w.compare.setCurrentIndex(w.compare.findData("replicates"))
    qtbot.waitUntil(lambda: not panel.drawing and len(w._shown_traces[3].traces) == 2
                    and not w._shown_traces[3].ions, timeout=30000)
    assert w._shown_traces[3].others_label == "its replicates"
    editor = w.window_editor
    editor.edit.setChecked(True)
    for level, view in (("peptide", "ions"), ("everywhere", "everywhere"),
                        ("replicates", "replicates")):
        editor.scope.setCurrentIndex(editor.scope.findData(level))
        editor.scope.activated.emit(editor.scope.currentIndex())  # as a user's choice
        assert w.compare.currentData() == view
    editor.scope.setCurrentIndex(editor.scope.findData("ion"))
    editor.scope.activated.emit(editor.scope.currentIndex())
    assert w.compare.currentData() == "replicates"  # this charge state alone: left as it was
    w.compare.setCurrentIndex(w.compare.findData("everywhere"))
    assert editor.scope.currentData() == "ion"  # browsing never changes how it is saved
    editor.edit.setChecked(False)


def test_the_window_is_edited_again_after_the_results_are_read_again(qtbot, project):  # noqa: F811
    # results read again (a resume finished) with the switch still on: the figure was left
    # empty (the editor compared the two Results by value; owner, 2026-11-08)
    spectra = _read(project)
    first, second = _results(spectra), _results(spectra)  # fitted before any preview
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    panel = w.panel("elution")

    def show_elution(results):
        w.show_results(results, project)
        series = w.tree.topLevelItem(0).child(0)
        w.tree.expandItem(series)
        w.tree.setCurrentItem(series.child(0).child(1))
        tabs = [w.figures.tabText(i) for i in range(w.figures.count())]
        w.figures.setCurrentIndex(tabs.index("elution"))

    show_elution(first)
    qtbot.waitUntil(lambda: w._shown_traces is not None and not panel.drawing, timeout=30000)
    w.window_editor.edit.setChecked(True)
    qtbot.waitUntil(lambda: w.window_editor._previewed is not None and not panel.drawing,
                    timeout=60000)
    show_elution(second)
    qtbot.waitUntil(lambda: w.window_editor._previewed is not None
                    and w.window_editor._target is not None  # None while the trace is read
                    and w.window_editor._target[0] is second and not panel.drawing,
                    timeout=60000)
    assert any(t.startswith("new window") for t in _titles(panel))



def _left_edge(ax, patch):
    """A span's left edge in data units: axvspan gives a Polygon before matplotlib 3.9 and a
    Rectangle after, so read the outline it draws."""
    drawn = patch.get_transform().transform(patch.get_path().vertices)
    return ax.transData.inverted().transform(drawn)[:, 0].min()

def test_the_replicates_are_aligned_while_editing_and_each_file_is_a_row(qtbot, project):  # noqa: F811
    """Compare with on the replicates while editing: the replicate's trace is moved onto this
    file's clock by the trace match (owner's choice, 2026-11-11: draft A); Each file's window
    draws one row per file with the new window shifted into it (draft B); a save at that
    level uses the same match."""
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
    assert w.compare.currentData() == "replicates"
    qtbot.waitUntil(lambda: w._shown_traces is not None and not panel.drawing
                    and len(w._shown_traces[3].traces) == 2, timeout=30000)
    editor = w.window_editor
    editor.edit.setChecked(True)
    assert editor.each_file.isEnabled() and not editor.each_file.isChecked()
    qtbot.waitUntil(lambda: editor.alignment is not None and not panel.drawing, timeout=30000)
    (plan,) = editor.alignment.plans
    assert plan.file == "un1.mzML" and plan.shift is not None
    other = plan.trace.spectrum_id
    fig = panel.view.figure
    ax = next(a for a in fig.axes if a.get_xlabel().startswith("retention time"))
    line = next(x for x in ax.lines if x.get_gid() == f"trace:{other}")
    assert line.get_xdata() == pytest.approx(plan.trace.times - plan.shift)
    assert any("moved by its match" in t.get_text() for t in ax.get_legend().get_texts())
    # one row per file, this file first; the window shifted into the other
    editor.each_file.setChecked(True)
    qtbot.waitUntil(lambda: not panel.drawing and len(panel.view.figure.axes) == 2,
                    timeout=30000)
    first, second = panel.view.figure.axes
    assert first.get_label() == "this file"
    hatched = [p for p in second.patches if p.get_gid() == "edited"]
    window = editor.window()
    assert _left_edge(second, hatched[0]) == pytest.approx(window[0] + plan.shift)
    # matched once: the window moves, the match stays
    editor._set_boxes((window[0] + 0.01, window[1]), None)
    editor.draw()
    assert editor._aligning is None
    # a match made within another drift range (the boxes' range dragged since): drawn until
    # the new one lands, so the files keep their places; a save waits for the new one
    ctx = editor.context()
    key = editor._aligned_key
    editor._aligned_key = (*key[:2], (1.0, 2.0))
    assert editor._aligned_now(ctx) is editor.alignment
    editor._aligned_key = ("p:another spectrum", *key[1:])
    assert editor._aligned_now(ctx) is None
    editor._aligned_key = key
    # a drag across the first row sets the window; across another row it does not
    fig = panel.view.figure
    to_fig = fig.transFigure.inverted()
    start, end = window[0] + 0.02, window[1] - 0.01
    (x0, y0), (x1, y1) = (to_fig.transform(first.transData.transform((t, 0.0)))
                          for t in (start, end))
    panel.spanned.emit(x0, y0 + 0.01, x1, y1 + 0.01)
    assert editor.window() == pytest.approx((start, end), abs=2e-3)
    (x0, y0), (x1, y1) = (to_fig.transform(second.transData.transform((t, 0.0)))
                          for t in (start - 0.01, end - 0.01))
    panel.spanned.emit(x0, y0 + 0.01, x1, y1 + 0.01)
    assert editor.window() == pytest.approx((start, end), abs=2e-3)
    # Compare with on this peptide's other ions: no other file, so no rows
    w.compare.setCurrentIndex(w.compare.findData("ions"))
    qtbot.waitUntil(lambda: not panel.drawing and not editor.each_file.isEnabled(),
                    timeout=30000)
    # the preview's fit done before the next test fits (two fits at once share PyTorch's
    # forward-AD levels: window_edit._FIT_LOCK)
    qtbot.waitUntil(lambda: editor.failure is not None or (
        editor._previewed is not None and editor._previewed[1] == editor.window()),
        timeout=60000)


def test_another_spectrum_never_draws_the_one_before(qtbot, project, monkeypatch):  # noqa: F811
    """Editing a window, then selecting another spectrum: the previous spectrum's preview
    (still being read) is dropped, so the tab never draws the previous spectrum again after it
    was cleared for the new one (owner, 2026-11-12: it went blank, then showed the previous
    view, then the new one)."""
    import threading

    from hxex.gui import window_edit
    from hxex.report import retention

    spectra = _read(project)
    results = _results(spectra)
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, project)
    series = w.tree.topLevelItem(0).child(0)
    w.tree.expandItem(series)
    first, second = series.child(0).child(1), series.child(0).child(0)  # UN rep 2, rep 1
    with qtbot.waitSignal(w.panel("fits at each K").drawn, timeout=30000):
        w.tree.setCurrentItem(first)
    tabs = [w.figures.tabText(i) for i in range(w.figures.count())]
    panel = w.panel("elution")
    w.figures.setCurrentIndex(tabs.index("elution"))
    qtbot.waitUntil(lambda: w._shown_traces is not None and not panel.drawing
                    and len(w._shown_traces[3].traces) == 2, timeout=30000)
    editor = w.window_editor
    editor.edit.setChecked(True)
    qtbot.waitUntil(lambda: editor.preview is not None and editor.preview.results is not None
                    and not editor._task.running and not panel.drawing, timeout=60000)
    old = w._shown_traces[1]
    # the next preview of this spectrum waits until released: it is still being read when
    # the other spectrum is selected
    gate, original = threading.Event(), window_edit.preview
    monkeypatch.setattr(window_edit, "preview",
                        lambda *a, **k: (gate.wait(30), original(*a, **k))[1])
    plot = retention.plot_window_edit

    def tagged(traces, res, sid, *a, **k):  # each figure knows its spectrum
        fig = plot(traces, res, sid, *a, **k)
        fig.hxex_spectrum = sid
        return fig

    monkeypatch.setattr(retention, "plot_window_edit", tagged)
    drawn = []  # the spectra of the figures the tab showed (a figure drawn and dropped is not)
    show = panel.show_figure
    monkeypatch.setattr(panel, "show_figure", lambda fig, *a, **k: (
        drawn.append(getattr(fig, "hxex_spectrum", None)), show(fig, *a, **k))[1])
    start, end = editor.window()
    editor.start.setValue(start + 0.01 * (end - start))  # a new window: read in the background
    qtbot.waitUntil(lambda: editor._task.running, timeout=10000)
    # the new spectrum's trace is read slowly (as from large scan files): the previous
    # spectrum's preview is ready first
    reader, slow = w._reader(), threading.Event()
    read = reader.traces
    monkeypatch.setattr(reader, "traces", lambda *a, **k: (slow.wait(30), read(*a, **k))[1])
    w.tree.setCurrentItem(second)
    drawn.clear()  # from here on, only the new spectrum may be drawn
    gate.set()
    qtbot.wait(1500)  # the previous preview read, fitted and (before the fix) drawn
    qtbot.waitUntil(lambda: not panel.drawing, timeout=30000)
    slow.set()
    qtbot.waitUntil(lambda: w._shown_traces is not None and w._shown_traces[1] != old
                    and editor.preview is not None and editor.preview.results is not None
                    and not editor._task.running and not panel.drawing, timeout=60000)
    new = w._shown_traces[1]
    assert drawn and old not in drawn and set(drawn) - {None} == {new}


def test_the_replicates_are_drawn_only_once_matched(qtbot, project, monkeypatch):  # noqa: F811
    """While editing, the replicates' traces are left out until they are matched to this
    file's trace: drawn where they are and then moved, the figure jumped (owner,
    2026-11-12)."""
    import threading

    from hxex.gui import window_edit

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
    other = w._shown_traces[3].traces[1].spectrum_id
    gate, original = threading.Event(), window_edit.align_files
    monkeypatch.setattr(window_edit, "align_files",
                        lambda *a, **k: (gate.wait(30), original(*a, **k))[1])

    def traces_drawn():
        fig = panel.view.figure
        return {line.get_gid() for ax in (fig.axes if fig else ()) for line in ax.lines
                if (line.get_gid() or "").startswith("trace:")}

    shown = []
    show = panel.show_figure
    monkeypatch.setattr(panel, "show_figure", lambda *a, **k: (
        show(*a, **k), shown.append(traces_drawn()))[0])
    editor = w.window_editor
    editor.edit.setChecked(True)
    # the preview read and fitted while the match waits: the replicate is not drawn
    qtbot.waitUntil(lambda: editor.preview is not None and editor.preview.results is not None
                    and not panel.drawing, timeout=60000)
    assert shown and all(f"trace:{other}" not in s for s in shown)
    gate.set()
    qtbot.waitUntil(lambda: editor.alignment is not None and not panel.drawing, timeout=30000)
    assert f"trace:{other}" in shown[-1]  # drawn once, matched
