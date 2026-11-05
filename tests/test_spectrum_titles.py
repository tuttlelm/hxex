# SPDX-License-Identifier: GPL-3.0-or-later
"""Spectrum figures' titles and m/z ticks: long titles (a fragment's spectrum id) wrapped at
natural places with nothing lost, page headers that make room for them, and m/z tick labels
that never collide however narrow the panel (owner, 2026-11-05)."""

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from hxex.pipeline import run
from hxex.report import Results, plot_series_spectra, plot_spectrum
from hxex.report.series import _page_header
from hxex.report.spectrum import MzLocator, text_width, wrap_title
from test_run import config, raw_spectra

# synthetic ids in the readers' forms (a peptide's and a fragment's, MS3 with pre-activation)
PEPTIDE = "proj:sample_a/run_01.raw/10-17-PEPTIDEK-z2/scans1200-1700"
FRAGMENT = ("proj:sample_a/run_01.raw/12-19-PTIDEKAR-z1-ion_type:z-dot_parent_pos:4-ms3-pre:"
            "HCD@ETD-ETD_pre/scans1250-1716")


def test_a_title_that_fits_stays_on_one_line():
    assert wrap_title(PEPTIDE, 6.0, 9) == PEPTIDE
    assert wrap_title("UN, rep 1 (K=1)", 3.0, 8) == "UN, rep 1 (K=1)"


def test_a_fragment_id_goes_on_two_lines_at_its_ion_type_with_nothing_lost():
    lines = wrap_title(FRAGMENT, 5.4, 9).split("\n")
    assert len(lines) == 2
    assert lines[0] == "proj:sample_a/run_01.raw/12-19-PTIDEKAR-z1"
    assert lines[1].startswith("-ion_type:")
    assert "".join(lines) == FRAGMENT  # a break inside an id adds and drops nothing
    assert all(text_width(line, 9) <= 5.4 for line in lines)


@pytest.mark.parametrize("width", [1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
def test_a_narrow_title_keeps_every_character(width):
    wrapped = wrap_title(FRAGMENT, width, 9)
    assert "".join(wrapped.split("\n")) == FRAGMENT
    # every line fits unless it is a piece with no place left to break
    for line in wrapped.split("\n"):
        assert text_width(line, 9) <= width or not any(c in line[1:] for c in "-_:/@ ")


def test_notes_break_between_notes_first():
    title = "K=2, rejected; poor_fit, bright  showing pop 2 (total is the whole fit)"
    lines = wrap_title(title, 3.2, 9).split("\n")
    assert lines == ["K=2, rejected; poor_fit, bright", "showing pop 2 (total is the whole fit)"]


def test_lines_already_in_a_title_are_kept_and_wrapped_alone():
    wrapped = wrap_title(f"{FRAGMENT}\nK=1; mz_shift", 5.4, 9)
    assert wrapped.split("\n")[-1] == "K=1; mz_shift" and wrapped.count("\n") == 2


def test_a_piece_without_a_break_is_never_cut():
    word = "x" * 200
    assert wrap_title(word, 1.0, 9) == word


def _gaps(fig, ax) -> list[float]:
    """Gaps (pixels) between neighbouring visible x tick labels of ``ax``."""
    FigureCanvasAgg(fig).draw()
    lo, hi = ax.get_xlim()
    boxes = [t.get_window_extent() for t in ax.get_xticklabels()
             if t.get_text() and lo <= t.get_position()[0] <= hi]
    return [b.x0 - a.x1 for a, b in zip(boxes, boxes[1:], strict=False)]


@pytest.mark.parametrize("columns", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("limits", [(1112.4, 1133.6), (1287.0, 1306.0), (485.1, 491.9),
                                    (1964.0, 2004.0), (512.0, 512.9)])
def test_mz_tick_labels_never_collide(columns, limits):
    # a page row of ``columns`` panels of the default cell (4.5 inches each)
    fig = Figure(figsize=(4.5 * columns, 3.0))
    axes = fig.subplots(1, columns, squeeze=False)[0]
    for ax in axes:
        ax.set_xlim(*limits)
        ax.xaxis.set_major_locator(MzLocator())
    fig.tight_layout()
    for ax in axes:
        gaps = _gaps(fig, ax)
        assert gaps and min(gaps) > 0  # at least two labels, none touching


def test_a_narrow_panel_gets_fewer_ticks_than_a_wide_one():
    counts = []
    for width in (2.0, 8.0):
        fig = Figure(figsize=(width, 2.5))
        ax = fig.add_subplot()
        ax.set_xlim(1112.4, 1133.6)
        ax.xaxis.set_major_locator(MzLocator())
        FigureCanvasAgg(fig).draw()
        counts.append(sum(1 for t in ax.get_xticklabels() if t.get_text()))
    assert 2 <= counts[0] < counts[1]


def test_a_long_page_title_wraps_above_the_legend_and_the_panels():
    label = ("sample_a 12-19 PTIDEKAR z-dot:8 z=1, parent 10-25 PEPTIDEKARPTIDEKAR z=4, "
             "ETD_pre (page 1 of 4)")
    fig = Figure(figsize=(4.5, 3.0))
    ax = fig.add_subplot()
    ax.set_title("UN, rep 1 (K=1)", fontsize=8)
    handles = [Line2D([], [], label=name) for name in ("fit", "population 1", "raw profile")]
    _page_header(fig, label, handles)
    text = fig._suptitle.get_text()
    assert "\n" in text and text.replace("\n", " ") == label  # wrapped at spaces, all kept
    renderer = FigureCanvasAgg(fig).get_renderer()
    title = fig._suptitle.get_window_extent(renderer)
    legend = fig.legends[0].get_window_extent(renderer)
    panel = ax.get_tightbbox(renderer)
    assert title.x0 >= 0 and title.x1 <= fig.get_figwidth() * fig.dpi  # inside the page
    assert title.y0 > legend.y1 and legend.y0 > panel.y1  # title, legend, then the panels


def test_a_short_page_title_keeps_its_band():
    fig = Figure(figsize=(9.0, 3.0))
    fig.add_subplot()
    _page_header(fig, "sample_a 10-17 PEPTIDEK z=2", [])
    assert fig._suptitle.get_text() == "sample_a 10-17 PEPTIDEK z=2"
    assert fig.get_figheight() == pytest.approx(3.45)


@pytest.fixture(scope="module")
def results():
    raw = raw_spectra(n_series=1)
    return Results.from_run(run(raw, config()), spectra=raw)


def test_a_peptide_spectrum_keeps_its_one_line_id_and_gets_the_mz_ticks(results):
    sid = str(results.fit.spectrum_id[0])
    fig = plot_spectrum(sid, results, show=("picked", "fit"))
    ax = fig.axes[0]
    assert ax.get_title().split("\n")[0] == sid
    assert isinstance(ax.xaxis.get_major_locator(), MzLocator)
    assert min(_gaps(fig, ax)) > 0
    page = plot_series_spectra(results, 0, show=("picked", "fit"), residual=True)
    for panel in page.axes:  # the residual panels share their spectrum's ticks
        assert isinstance(panel.xaxis.get_major_locator(), MzLocator)
