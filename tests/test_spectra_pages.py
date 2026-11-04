# SPDX-License-Identifier: GPL-3.0-or-later
"""A series' spectra on pages (hxex.report.series.spectra_pages; owner, 2026-11-04: a kinetics
run drew one page of 281 panels, 214 inches tall): at most [figures] page_rows by page_columns
a page, more replicates than columns over the next row, a long time course thinned to
max_points over log time (every spectrum still fitted), and the files and window pages."""
import dataclasses

import numpy as np
import pytest

import hxex.report.series as series
from hxex.report.figures import PlotConfig, series_figures, write_figures
from hxex.report.series import _layout, spectra_pages


def _course(monkeypatch, times, reps, controls=("undeut",), control_reps=None):
    """A made-up series: each time with ``reps`` replicates, and the controls (with
    ``control_reps`` replicates; default: ``reps``)."""
    rows, t, role, rep, index = [], [], [], [], []
    points = [(c, np.nan if c == "undeut" else 1e6) for c in controls if c == "undeut"]
    points += [("", x) for x in times]
    points += [(c, 1e6) for c in controls if c == "fulldeut"]
    for n, (r, x) in enumerate(points):
        for k in range(1, (control_reps or reps) + 1 if r else reps + 1):
            rows.append(len(rows))
            t.append(x)
            role.append(r)
            rep.append(k)
            index.append(n)
    info = {"rows": np.array(rows), "times": np.array(t, dtype=float),
            "roles": np.array(role, dtype=object), "reps": np.array(rep),
            "time_index": np.array(index), "label": "S 0001-0010 PEPTIDE z=2"}
    monkeypatch.setattr(series, "series_info", lambda results, data_id: info)
    return info


def test_a_kinetics_course_is_thinned_and_paged(monkeypatch):
    times = np.linspace(22.0, 900.0, 281)
    _course(monkeypatch, times, reps=1)
    pages = spectra_pages(None, 0)
    assert len(pages) == 1 and 2 <= pages[0].shown <= 49 and pages[0].points == 282
    drawn = set(pages[0].rows)
    assert {0, 1, 281} <= drawn  # the control, the first and the last time point
    assert "time points shown, every one fitted" in pages[0].note
    every = spectra_pages(None, 0, max_points=None)  # all of them: pages of 8 x 6
    assert [len(p.rows) for p in every] == [48] * 5 + [42] and every[-1].note == "page 6 of 6"
    assert sorted(r for p in every for r in p.rows) == list(range(282))


def test_replicates_make_a_row_each_and_wrap_beyond_the_columns(monkeypatch):
    info = _course(monkeypatch, [4.0 * 2 ** n for n in range(12)], reps=3,
                   controls=("undeut", "fulldeut"))
    pages = spectra_pages(None, 0)
    assert [len(p.rows) // 3 for p in pages] == [8, 6]  # time points per page: 14 in all
    assert pages[0].note == "page 1 of 2"
    info = _course(monkeypatch, [4.0, 60.0, 1800.0], reps=8)
    nrows, ncols, pos, by_rep = _layout(info, None, max_columns=6)
    assert by_rep and (nrows, ncols) == (8, 6)  # 4 points, each over two rows of 6 then 2
    assert pos[:8] == [(0, c) for c in range(6)] + [(1, 0), (1, 1)]
    assert len(spectra_pages(None, 0)) == 1  # 4 points x 2 rows = 8 rows: one page
    assert len(spectra_pages(None, 0, page_rows=4)) == 2


def test_the_settings_are_checked():
    assert PlotConfig(max_points=False).max_points is None
    for bad in ({"page_rows": 0}, {"page_columns": 1.5}, {"max_points": 1}, {"max_points": True}):
        with pytest.raises(ValueError):
            PlotConfig(**bad)


@pytest.fixture(scope="module")
def results():
    from hxex.report import Results
    from test_run import config, raw_spectra, run

    return Results.from_run(run(raw_spectra(n_series=1), config(bootstrap=None)))


def test_the_pages_are_drawn_and_written(results, tmp_path):
    cfg = PlotConfig(page_rows=1, kinds=("spectra", "uptake"), format="png", overview=False,
                     series="all", dpi=40)
    figures = series_figures(results, 0, cfg)
    assert list(figures) == ["spectra_1", "spectra_2", "spectra_3", "uptake"]
    assert figures["spectra_2"]._suptitle.get_text().endswith("(page 2 of 3)")
    written = sorted(p.name for p in write_figures(results, tmp_path / "png", cfg))
    pages = [n for n in written if "_spectra_" in n]
    assert len(pages) == 3 and [n.rsplit("_", 1)[1] for n in pages] == ["1.png", "2.png",
                                                                         "3.png"]
    pdf = write_figures(results, tmp_path / "pdf", dataclasses.replace(cfg, format="pdf"))
    assert len(pdf) == 1 and pdf[0].suffix == ".pdf"


def test_the_window_offers_the_pages(qtbot, results):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from hxex.gui.results_workspace import ResultsWorkspace

    page = ResultsWorkspace()
    qtbot.addWidget(page)
    page.show_results(dataclasses.replace(results, figures=PlotConfig(page_rows=1)), None)
    series_item = page.tree.topLevelItem(0).child(0)
    spectra = page.panel("spectra")
    with qtbot.waitSignal(spectra.drawn, timeout=30000):
        page.tree.setCurrentItem(series_item)
    box = page.spectra_page
    assert not box.isHidden()  # several pages: the list in the tab's bar
    assert [box.itemText(i) for i in range(box.count())] == [f"page {n} of 3" for n in (1, 2, 3)]
    with qtbot.waitSignal(spectra.drawn, timeout=30000):
        box.setCurrentIndex(2)
    assert spectra.view.figure._suptitle.get_text().endswith("(page 3 of 3)")
    # one page (the defaults): no list
    page.show_results(results, None)
    with qtbot.waitSignal(spectra.drawn, timeout=30000):
        page.tree.setCurrentItem(page.tree.topLevelItem(0).child(0))
    assert box.isHidden()


def test_replicated_controls_alone_fill_the_grid_on_every_page(monkeypatch):
    """Only the controls replicated (owner, on a targeted LC-MS run): the series fills the grid,
    and every page keeps the series' layout and width (the first page was drawn by replicate,
    the second as a grid)."""
    _course(monkeypatch, [4.0 * 2 ** n for n in range(14)], reps=1,
            controls=("undeut", "fulldeut"), control_reps=2)
    pages = spectra_pages(None, 0)
    assert len(pages) == 1 and pages[0].layout == ("grid", 5)  # 18 spectra: 4 rows of 5
    many = spectra_pages(None, 0, page_rows=2)
    assert len(many) == 2 and {p.layout for p in many} == {("grid", 5)}
    info = series.series_info(None, 0)
    sub = {k: (v[list(many[1].rows)] if isinstance(v, np.ndarray) else v) for k, v in info.items()}
    nrows, ncols, _, by_rep = _layout(sub, None, fixed=many[1].layout)
    assert ncols == 5 and not by_rep  # the last page as wide as the first
    # a replicated time course keeps its replicate columns on every page
    _course(monkeypatch, [4.0 * 2 ** n for n in range(12)], reps=3)
    pages = spectra_pages(None, 0)
    assert {p.layout for p in pages} == {("replicates", (1, 2, 3))}


def test_a_spectrum_keeps_its_series_axis_and_plot_area(qtbot, results):
    """Stepping from a series to a time point and a spectrum keeps the series' m/z range
    (owner: it changed), and intensity labels of one decimal keep the plot area in place."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from hxex.gui.results_workspace import ResultsWorkspace
    from hxex.report.series import series_mz_range
    from hxex.report.spectrum import plot_spectrum

    sid = str(results.fit.spectrum_id[1])
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = plot_spectrum(sid, results)
    FigureCanvasAgg(fig).draw()
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels() if t.get_text()]
    assert labels and all(t.count(".") == 1 for t in labels)  # 0.0, 0.5, 1.0: one width
    page = ResultsWorkspace()
    qtbot.addWidget(page)
    page.show_results(results, None)
    wanted = series_mz_range(results, 0)
    series_item = page.tree.topLevelItem(0).child(0)
    page.tree.expandItem(series_item)
    leaf = series_item.child(1).child(0)
    with qtbot.waitSignal(page.panel("spectrum").drawn, timeout=30000):
        page.tree.setCurrentItem(leaf)
    shown = page.panel("spectrum").view.figure.axes[0].get_xlim()
    assert np.allclose(shown, wanted)
    qtbot.waitUntil(lambda: page.panel("fits at each K").view.figure is not None, timeout=30000)
    assert np.allclose(page.panel("fits at each K").view.figure.axes[0].get_xlim(), wanted)


def test_the_arrows_go_through_a_series_pages_before_the_next_series(qtbot):
    """Stepping by series with the spectra in view (owner: pages 2, 3 ... were missed):
    forward through the pages, then the next series' first page; back from a first page, the
    previous series' last page."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from hxex.gui.results_workspace import ResultsWorkspace
    from hxex.report import Results
    from test_run import config, raw_spectra, run

    two = Results.from_run(run(raw_spectra(n_series=2), config(bootstrap=None)))
    page = ResultsWorkspace()
    qtbot.addWidget(page)
    page.show_results(dataclasses.replace(two, figures=PlotConfig(page_rows=1)), None)
    top = page.tree.topLevelItem(0)
    page.tree.setCurrentItem(top.child(0))
    page.figures.setCurrentWidget(page.panel("spectra"))
    box = page.spectra_page

    def where():
        return page.tree.currentItem().data(0, 257)[1].data_id, box.currentIndex() + 1

    assert where() == (0, 1)
    seen = [where()]
    for _ in range(3):
        assert page.step_spectrum(1, "series")
        seen.append(where())
    assert seen == [(0, 1), (0, 2), (0, 3), (1, 1)]
    assert page.step_spectrum(-1, "series") and where() == (0, 3)  # back: the last page
    assert page.step_spectrum(-1, "series") and where() == (0, 2)
    # on another tab the arrows go from series to series
    page.figures.setCurrentWidget(page.panel("uptake"))
    assert page.step_spectrum(1, "series") and where()[0] == 1
