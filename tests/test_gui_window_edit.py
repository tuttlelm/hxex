# SPDX-License-Identifier: GPL-3.0-or-later
"""The elution tab's window editing (hxex.gui.window_editor; PLAN Next steps 26): the switch,
the boxes, a drag across the trace, the preview beside the spectrum, Save to the elution-window
table and Undo. Runs offscreen."""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from hxex.gui.results_workspace import ResultsWorkspace
from test_window_edit import PEPTIDE, RT, N, _read, _results, project  # noqa: F401


def _titles(panel):
    fig = panel.view.figure
    return [] if fig is None else [ax.get_title() for ax in fig.axes]


def _pictured(panel) -> bool:
    """Whether the figure set is the one whose picture is on screen (a drag is mapped with
    the pictured figure: F-19)."""
    return not panel.drawing and getattr(panel.view, "ready", True)


def _scan_times(w):
    return w._shown_traces[3].scan_times


def _sent(task):
    """The tickets whose results ``task``'s background jobs have sent back, delivered to the
    window or ignored (a job replaced meanwhile)."""
    sent = []
    task._signals.done.connect(lambda ticket, _result: sent.append(ticket))
    task._signals.failed.connect(lambda ticket, _message: sent.append(ticket))
    return sent


def _wait_delivered(qtbot, ticket, sent):
    """Waits until the job ``ticket`` sent its result back and the window had the chance to
    act on it: the evidence that something did not happen is the event that would have made
    it happen, not a fixed wait a loaded machine could outrun (audit of 2026-11-15, Q-29)."""
    from PySide6.QtWidgets import QApplication

    qtbot.waitUntil(lambda: ticket in sent, timeout=30000)
    QApplication.processEvents()  # the window's own slot, queued before ours, has run


def test_a_window_is_edited_previewed_saved_and_undone(qtbot, project, monkeypatch):  # noqa: F811
    spectra = _read(project)
    results = _results(spectra)
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, project)
    series = w.tree.topLevelItem(0).child(0)
    w.tree.expandItem(series)
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
    # the unchanged window is previewed over the same scans: the boxes hold the scans' own
    # times, whatever their decimals show (audit of 2026-11-15, F-1)
    qtbot.waitUntil(lambda: editor.preview is not None, timeout=30000)
    assert editor.preview.spectrum.spectrum_id == w._shown_traces[1]
    times = _scan_times(w)
    assert editor.start.decimals() == 4 and np.isclose(times, editor.window()[0]).any()
    # a drag across the trace sets the window (figure fractions, as the view gives them),
    # mapped with the figure whose picture is on screen
    qtbot.waitUntil(lambda: _pictured(panel), timeout=30000)
    fig = panel.view.figure
    ax = next(a for a in fig.axes if a.get_xlabel().startswith("retention time"))
    start, end = selected.window[0] + 0.02, selected.window[1] - 0.01
    to_fig = fig.transFigure.inverted()
    (x0, y0), (x1, y1) = (to_fig.transform(ax.transData.transform((t, 0.0)))
                          for t in (start, end))
    panel.spanned.emit(x0, y0 + 0.01, x1, y1 + 0.01)
    assert editor.window() == pytest.approx((start, end), abs=3e-3)  # the nearest scans
    assert all(np.isclose(times, v).any() for v in editor.window())
    # the arrows step scan by scan, never back onto the same scan
    at = editor.window()[0]
    editor.start.stepBy(1)
    assert editor.window()[0] == pytest.approx(times[np.searchsorted(times, at) + 1])
    editor.start.stepBy(-1)
    assert editor.window()[0] == pytest.approx(at)
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
    assert editor.scope.currentText() == "2. every charge state of this peptide, in this file"
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
    """Show elution traces for, choice 2 (every charge state of this peptide, in this file),
    draws its other charge state behind the selected trace; choice 1 the trace alone; switched
    back, its replicates. With Edit window on, choosing how to save leaves the traces as they
    are (it redrew them: owner, 2026-11-12)."""
    from test_elution_windows import _charges_project

    path = _charges_project(tmp_path / "charges")
    results = _results(_read(path))
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, path)
    series = w.tree.topLevelItem(0).child(0)
    w.tree.expandItem(series)
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
    assert w.compare.currentText() == "2. every charge state of this peptide, in this file"
    w.compare.setCurrentIndex(w.compare.findData("replicates"))
    qtbot.waitUntil(lambda: not panel.drawing and len(w._shown_traces[3].traces) == 2
                    and not w._shown_traces[3].ions, timeout=30000)
    assert w._shown_traces[3].others_label == "its replicates"
    w.compare.setCurrentIndex(w.compare.findData("ion"))  # view 1: this file's trace alone
    qtbot.waitUntil(lambda: not panel.drawing and len(w._shown_traces[3].traces) == 1
                    and not w._shown_traces[3].ions, timeout=30000)
    w.compare.setCurrentIndex(w.compare.findData("replicates"))
    editor = w.window_editor
    editor.edit.setChecked(True)
    for level in ("peptide", "everywhere", "ion"):
        editor.scope.setCurrentIndex(editor.scope.findData(level))
        editor.scope.activated.emit(editor.scope.currentIndex())  # as a user's choice
        assert w.compare.currentData() == "replicates"  # the view stays (no redraw)
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
                    and w.window_editor._target[0] is w._reader()  # the new record's reader
                    and w._shown_traces[0] is second and not panel.drawing,
                    timeout=60000)
    assert any(t.startswith("new window") for t in _titles(panel))



def _left_edge(ax, patch):
    """A span's left edge in data units: axvspan gives a Polygon before matplotlib 3.9 and a
    Rectangle after, so read the outline it draws."""
    drawn = patch.get_transform().transform(patch.get_path().vertices)
    return ax.transData.inverted().transform(drawn)[:, 0].min()

def test_the_replicates_are_aligned_while_editing_and_each_file_is_a_row(qtbot, project):  # noqa: F811
    """Show elution traces for on the replicates (3) while editing: the replicate's trace is moved
    onto this file's clock by the trace match (owner's choice, 2026-11-11: draft A); Each
    file's window draws one row per file with the new window shifted into it (draft B); a save
    at that level uses the same match."""
    spectra = _read(project)
    results = _results(spectra)
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, project)
    series = w.tree.topLevelItem(0).child(0)
    w.tree.expandItem(series)
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
    # the picture drawn before the patches' transforms are read: the matplotlib view draws
    # the picture at the view's dpi in its worker (the figure's dpi changed meanwhile), which
    # failed the lowest-versions CI job of 5731ad2 once
    qtbot.waitUntil(lambda: _pictured(panel), timeout=30000)
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
    qtbot.waitUntil(lambda: _pictured(panel), timeout=30000)
    fig = panel.view.figure
    to_fig = fig.transFigure.inverted()
    start, end = window[0] + 0.02, window[1] - 0.01
    (x0, y0), (x1, y1) = (to_fig.transform(first.transData.transform((t, 0.0)))
                          for t in (start, end))
    panel.spanned.emit(x0, y0 + 0.01, x1, y1 + 0.01)
    assert editor.window() == pytest.approx((start, end), abs=3e-3)
    (x0, y0), (x1, y1) = (to_fig.transform(second.transData.transform((t, 0.0)))
                          for t in (start - 0.01, end - 0.01))
    panel.spanned.emit(x0, y0 + 0.01, x1, y1 + 0.01)
    assert editor.window() == pytest.approx((start, end), abs=3e-3)
    # view 2, this peptide's other ions: no other file, so no rows
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
    editor.start.stepBy(1)  # a new window (one scan later): read in the background
    qtbot.waitUntil(lambda: editor._task.running, timeout=10000)
    held, sent = editor._task._ticket, _sent(editor._task)
    # the new spectrum's trace is read slowly (as from large scan files): the previous
    # spectrum's preview is ready first
    reader, slow = w._reader(), threading.Event()
    read = reader.traces
    monkeypatch.setattr(reader, "traces", lambda *a, **k: (slow.wait(30), read(*a, **k))[1])
    w.tree.setCurrentItem(second)
    drawn.clear()  # from here on, only the new spectrum may be drawn
    gate.set()
    # the previous preview read and its result sent back (before the fix: drawn); waited
    # for, not a fixed time (audit of 2026-11-15, Q-29)
    _wait_delivered(qtbot, held, sent)
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


def test_undo_stays_with_the_results_the_window_was_saved_in(qtbot, project, tmp_path):  # noqa: F811
    """Undo after opening another project rewrote the previous project's elution-window table
    and project file, naming only the file's name (audit of 2026-11-15, D-5 and F-4): the
    history belongs to the results shown, and a save of another project is refused."""
    from hxex.gui.window_edit import save_window

    spectra = _read(project)
    results = _results(spectra)
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, project)
    editor = w.panel("elution") and w.window_editor  # made with its panel
    sid = next(s for s in spectra.spectrum_id if "/un2.mzML/" in s)
    k = list(spectra.spectrum_id).index(sid)
    saved = save_window(project, ("S", "un2.mzML", sid.split("/")[2]),
                        (spectra.rt_start[k] + 0.01, spectra.rt_end[k]))
    table, before, project_text = saved.path, saved.path.read_bytes(), project.read_bytes()
    assert saved.project == project.absolute()
    editor.history.append(saved)
    editor.undo.setEnabled(True)
    w.show_results(results, project)  # the same results read again: the history stays
    assert editor.history and editor.undo.isEnabled()
    other = tmp_path / "other" / "hxex.toml"
    other.parent.mkdir()
    other.write_text(project.read_text())
    w.show_results(results, other)
    assert not editor.history and not editor.undo.isEnabled()
    messages = []
    editor.message.connect(messages.append)
    editor.history.append(saved)  # as if it had survived the switch
    editor._undo()
    assert messages[-1].startswith("not undone") and str(project) in messages[-1]
    assert table.read_bytes() == before and project.read_bytes() == project_text
    w.show_results(results, project.parent / "out")  # the same project's bundle on its own
    editor.history.append(saved)
    editor._undo()
    assert messages[-1].startswith("not undone") and table.read_bytes() == before


def test_a_save_for_the_peptide_asks_before_replacing_this_files_charge_states(
        qtbot, tmp_path, monkeypatch):
    """Save choice 2 with the file's other charge state holding a window of its own: the window
    asks, naming this file; kept, that row keeps its window (audit of 2026-11-15, D-8)."""
    from hxex.gui import window_editor
    from test_elution_windows import PEPTIDE, N, _charges_project

    peptide = f"0001-{N:04d}-{PEPTIDE}"
    rows = "".join(f"S,un2.mzML,{peptide},{peptide}-z{z},0.6000,0.6600\n" for z in (2, 3))
    path = _charges_project(tmp_path / "charges",
                            "sample,file,peptide,ion,rt_start,rt_end\n" + rows)
    results = _results(_read(path))
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, path)
    series = w.tree.topLevelItem(0).child(0)
    w.tree.expandItem(series)
    w.tree.setCurrentItem(series.child(0).child(1))  # UN, rep 2: un2.mzML
    tabs = [w.figures.tabText(i) for i in range(w.figures.count())]
    panel = w.panel("elution")
    w.figures.setCurrentIndex(tabs.index("elution"))
    qtbot.waitUntil(lambda: w._shown_traces is not None and not panel.drawing, timeout=30000)
    editor = w.window_editor
    editor.edit.setChecked(True)
    qtbot.waitUntil(lambda: editor.failure is not None or (
        editor._previewed is not None and editor._previewed[1] == editor.window()
        and editor.preview is not None), timeout=60000)
    assert editor.failure is None
    qtbot.waitUntil(lambda: not panel.drawing, timeout=30000)
    asked, messages = [], []
    editor.message.connect(messages.append)

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
    editor.scope.setCurrentIndex(editor.scope.findData("peptide"))  # its own row set "ion"
    editor.save.click()
    assert asked and "un2.mzML (this file: 1 other charge state(s))" in asked[0]
    assert "keep a window of their own" in messages[-1]
    lines = (path.parent / "rt_elution_windows.csv").read_text().splitlines()
    names = lines[0].split(",")
    rows = [dict(zip(names, r.split(","), strict=True)) for r in lines[1:]]
    ions = [r["ion"] for r in rows if r["ion"] and r["rt_start"]]
    assert len(ions) == 1  # the edited charge state gave its window up, the other kept its own
    assert any(not r["ion"] and r["peptide"] == peptide for r in rows)  # the peptide's row
    editor.edit.setChecked(False)


# --- the audit of 2026-11-15, fix Session 4 ---------------------------------------------------


def _run_project(tmp_path):
    """The two-file project of test_window_edit, run with hxex run (a bundle the window opens
    through its project file)."""
    from hxex.cli import main
    from test_cli import FAST_RUN
    from test_mzml import write_mzml
    from test_retention_search import lc_run

    write_mzml(tmp_path / "un1.mzML", lc_run(1))
    write_mzml(tmp_path / "un2.mzML", lc_run(2, shift=6.0))
    (tmp_path / "peptides.csv").write_text(
        "Peptide,start_seq,end_seq,charge,rt_start,rt_end,note\n"
        f"{PEPTIDE},1,{N},2,{RT - 4 / 60:.4f},{RT + 4 / 60:.4f},kept\n")
    path = tmp_path / "hxex.toml"
    path.write_text('project = "p"\n\n[input]\nformat = "mzml"\npeptides = "peptides.csv"\n'
                    '\n[[input.runs]]\npath = "un1.mzML"\nsample = "S"\nrole = "undeut"\n'
                    '\n[[input.runs]]\npath = "un2.mzML"\nsample = "S"\nrole = "undeut"\n'
                    '\n[output]\nbundle = "out"\nfigures = false\n' + FAST_RUN)
    assert main(["-q", "run", str(path)]) == 0
    return path


def _show_elution(qtbot, page, which=1):
    series = page.tree.topLevelItem(0).child(0)
    page.tree.expandItem(series)
    page.tree.setCurrentItem(series.child(0).child(which))
    tabs = [page.figures.tabText(i) for i in range(page.figures.count())]
    panel = page.panel("elution")
    page.figures.setCurrentIndex(tabs.index("elution"))
    qtbot.waitUntil(lambda: page._shown_traces is not None and not panel.drawing
                    and len(page._shown_traces[3].traces) == 2, timeout=30000)
    return panel


def test_a_review_decision_keeps_the_window_being_edited(qtbot, tmp_path, monkeypatch):
    """A decision (status, K, or a note) on the spectrum being edited gives new Results; the
    typed window and its preview stay, and the trace is not read again (F-2: the editor took
    the new Results for another spectrum and threw the editing away)."""
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox

    from hxex.gui.window import MainWindow

    path = _run_project(tmp_path)
    window = MainWindow(settings=QSettings(str(tmp_path / "g.ini"), QSettings.Format.IniFormat))
    window.ask_to_save = lambda: QMessageBox.StandardButton.Discard
    qtbot.addWidget(window)
    window.open_asking(path)
    qtbot.waitUntil(lambda: not window.loading and window.results_path is not None,
                    timeout=60000)
    page = window.results_page
    panel = _show_elution(qtbot, page)
    editor = page.window_editor
    editor.edit.setChecked(True)
    qtbot.waitUntil(lambda: editor._previewed is not None and not panel.drawing, timeout=60000)
    editor.start.stepBy(2)  # typed: two scans later
    typed = editor.window()
    qtbot.waitUntil(lambda: editor._previewed is not None and editor._previewed[1] == typed
                    and not panel.drawing, timeout=60000)
    shown = editor.preview
    before = page.results
    reader = page._reader()
    reads = []
    monkeypatch.setattr(reader, "traces", lambda *a, **k: (reads.append(a), None)[1])
    assert page.review.decide(status="accepted", note="a note about the window")
    qtbot.waitUntil(lambda: page.results is not before and page._shown_traces[0] is page.results
                    and not panel.drawing, timeout=30000)
    assert editor.window() == typed and editor.preview is shown  # kept, not read again
    assert editor._previewed[1] == typed and not reads
    assert page._shown_traces[1] == page._selection[1][3].spectrum_id
    # results read again from disk are another record: the boxes are set afresh
    window.close()


def test_a_save_still_matching_is_dropped_when_another_spectrum_is_selected(
        qtbot, project, monkeypatch):  # noqa: F811
    """A save at level 3 reads the drift range once, at the click, and a save still matching
    the other files when another spectrum is selected is dropped and said, instead of asking
    its question under the new spectrum (F-7, D-11, F-16)."""
    import threading

    from hxex.gui import window_edit

    spectra = _read(project)
    results = _results(spectra)
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, project)
    w.compare.setCurrentIndex(w.compare.findData("ion"))  # view 1: nothing matched beforehand
    series = w.tree.topLevelItem(0).child(0)
    w.tree.expandItem(series)
    w.tree.setCurrentItem(series.child(0).child(1))
    tabs = [w.figures.tabText(i) for i in range(w.figures.count())]
    panel = w.panel("elution")
    w.figures.setCurrentIndex(tabs.index("elution"))
    qtbot.waitUntil(lambda: w._shown_traces is not None and not panel.drawing, timeout=30000)
    editor = w.window_editor
    editor.edit.setChecked(True)
    qtbot.waitUntil(lambda: editor._previewed is not None and not panel.drawing, timeout=60000)
    gate, started, original = threading.Event(), threading.Event(), window_edit.plan_spread
    monkeypatch.setattr(window_edit, "plan_spread",
                        lambda *a, **k: (started.set(), gate.wait(30), original(*a, **k))[2])
    messages = []
    editor.message.connect(messages.append)
    editor.scope.setCurrentIndex(editor.scope.findData("replicates"))
    editor.save.click()
    assert editor._saving == w._shown_traces[1] and not editor.save.isEnabled()
    held, sent = editor._spread_task._ticket, _sent(editor._spread_task)
    # the match must be running when the other spectrum is selected: a job replaced before
    # its work started never runs (C-12), so nothing would come back to wait for
    qtbot.waitUntil(started.is_set, timeout=30000)
    w.tree.setCurrentItem(series.child(0).child(0))  # another spectrum while matching
    assert editor._saving is None and editor.save.isEnabled()
    assert messages and messages[-1].startswith("not saved: another spectrum was selected")
    gate.set()
    _wait_delivered(qtbot, held, sent)  # the match's result back (before the fix: saved)
    assert not (project.parent / "rt_elution_windows.csv").exists()
    editor.edit.setChecked(False)


def test_a_table_that_does_not_read_is_said_when_the_boxes_are_set(qtbot, project):  # noqa: F811
    """An elution-window table the project cannot read (two rows for one key, a hand edit)
    is said on the status bar when the boxes are set, not passed over in silence (F-9)."""
    spectra = _read(project)
    results = _results(spectra)
    sid = next(s for s in spectra.spectrum_id if "/un2.mzML/" in s)
    ion = sid.split("/")[2]
    (project.parent / "rt_elution_windows.csv").write_text(
        "sample,file,peptide,ion,rt_start,rt_end\n"
        f"S,un2.mzML,,{ion},0.6000,0.6600\nS,un2.mzML,,{ion},0.6100,0.6700\n")
    project.write_text(project.read_text().replace(
        'peptides = "peptides.csv"', 'peptides = "peptides.csv"\n'
                                    'rt_elution_windows = "rt_elution_windows.csv"'))
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, project)
    panel = _show_elution(qtbot, w)
    editor = w.window_editor
    messages = []
    editor.message.connect(messages.append)
    editor.edit.setChecked(True)
    assert any(m.startswith("the elution-window table was not read:") for m in messages)
    assert editor.window() == pytest.approx(w._shown_traces[3].traces[0].window, abs=1e-6)
    qtbot.waitUntil(lambda: editor._previewed is not None and not panel.drawing, timeout=60000)
    editor.edit.setChecked(False)


def test_a_failed_read_is_given_when_saving(qtbot, project):  # noqa: F811
    """When the new window's spectrum could not be made (no signal), Save gives that failure
    instead of saying to wait for it (F-15)."""
    spectra = _read(project)
    results = _results(spectra)
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, project)
    panel = _show_elution(qtbot, w)
    editor = w.window_editor
    editor.edit.setChecked(True)
    qtbot.waitUntil(lambda: editor._previewed is not None and not panel.drawing, timeout=60000)
    times = _scan_times(w)
    editor._set_boxes((times[-1], times[-1]), None)  # the file's last scan: no peptide there
    editor.refresh()
    qtbot.waitUntil(lambda: editor.failure is not None and not panel.drawing, timeout=60000)
    assert "no signal" in editor.failure
    messages = []
    editor.message.connect(messages.append)
    editor.save.click()
    assert messages[-1] == f"not saved: {editor.failure}"
    editor.edit.setChecked(False)


def test_each_files_window_is_off_when_no_other_file_has_the_ion(qtbot, tmp_path):
    """With traces 3 or 4 and no other file of the ion, Each file's window did nothing; it is
    disabled once the match found no file (F-17)."""
    from test_mzml import write_mzml
    from test_retention_search import lc_run

    write_mzml(tmp_path / "un1.mzML", lc_run(1))
    (tmp_path / "peptides.csv").write_text(
        "Peptide,start_seq,end_seq,charge,rt_start,rt_end\n"
        f"VDQIIMAK,1,8,2,{RT - 4 / 60:.4f},{RT + 4 / 60:.4f}\n")
    path = tmp_path / "hxex.toml"
    path.write_text('project = "p"\n\n[input]\nformat = "mzml"\npeptides = "peptides.csv"\n'
                    '\n[[input.runs]]\npath = "un1.mzML"\nsample = "S"\nrole = "undeut"\n')
    results = _results(_read(path))
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, path)
    series = w.tree.topLevelItem(0).child(0)
    w.tree.expandItem(series)
    w.tree.setCurrentItem(series.child(0).child(0))
    tabs = [w.figures.tabText(i) for i in range(w.figures.count())]
    panel = w.panel("elution")
    w.figures.setCurrentIndex(tabs.index("elution"))
    qtbot.waitUntil(lambda: w._shown_traces is not None and not panel.drawing, timeout=30000)
    editor = w.window_editor
    assert w.compare.currentData() == "replicates"
    editor.edit.setChecked(True)
    qtbot.waitUntil(lambda: editor._aligned_key is not None and not panel.drawing,
                    timeout=30000)
    assert editor.alignment is None and not editor.each_file.isEnabled()
    qtbot.waitUntil(lambda: editor._previewed is not None and not panel.drawing, timeout=60000)
    editor.edit.setChecked(False)


def test_the_project_file_is_loaded_once_for_the_spectra_shown(qtbot, project, monkeypatch):  # noqa: F811
    """Each spectrum shown with Edit window on loaded the project file three times and read
    the whole table (F-12); now the file is loaded once and kept while unchanged."""
    import hxex.config as config
    from hxex.gui.window_edit import forget_cached

    forget_cached()
    spectra = _read(project)
    results = _results(spectra)
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, project)
    panel = _show_elution(qtbot, w)
    editor = w.window_editor
    editor.edit.setChecked(True)
    qtbot.waitUntil(lambda: editor._previewed is not None and not panel.drawing, timeout=60000)
    loads = []
    load = config.load_project
    monkeypatch.setattr(config, "load_project", lambda *a, **k: (loads.append(a), load(*a, **k))[1])
    series = w.tree.topLevelItem(0).child(0)
    w.tree.setCurrentItem(series.child(0).child(0))  # the other replicate
    qtbot.waitUntil(lambda: w._shown_traces is not None and w._shown_traces[1].count("un1")
                    and editor._previewed is not None and not panel.drawing, timeout=60000)
    assert loads == []
    # the project file broken after the results opened: Save says so, no traceback (D-10)
    project.write_text("not = [toml\n")
    messages = []
    editor.message.connect(messages.append)
    editor.save.click()
    assert messages and messages[-1].startswith("not saved:")
    editor.edit.setChecked(False)
