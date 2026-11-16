# SPDX-License-Identifier: GPL-3.0-or-later
"""The Results workspace's List (hxex.gui.spectra_list) on its own: the row chosen kept
through filters and sorts whichever table it shows, stepping by row, the row found again after
the list is cleared and filled, and the columns sized from a sample of rows. Runs offscreen."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from hxex.gui.spectra_list import SpectraList

SERIES = [{"data_id": d, "sample": "S", "peptide_range": "0001-0010", "charge": 2,
           "n_spectra": 8} for d in range(3)]
COMPONENTS = [{"spectrum_id": f"s{i}", "data_id": 0, "sample": "S", "peptide_range": "0001-0010",
               "peptide": "RDKVQKEYAL", "charge": 2, "time": 60.0, "role": "", "K": k, "rank": r,
               "mean": 2.0 * r, "_index": i}
              for i, k in ((0, 1), (1, 2), (2, 1)) for r in range(1, k + 1)]


@pytest.fixture
def lst(qtbot):
    w = SpectraList()
    qtbot.addWidget(w)
    return w


def test_find_and_the_arrows_work_on_tables_without_a_spectrum_id(lst, qtbot):
    """series.csv and uptake.csv rows have no spectrum_id: Find, the sample list and a column
    filter keep the row chosen, and ◀ ▶ step by row (audit of 2026-11-15, C-5)."""
    lst.set_rows(list(SERIES[0]), SERIES, table="series")
    chosen = []
    lst.chosen.connect(chosen.append)
    assert lst.step(1) and lst.current_row()["data_id"] == 0
    lst.search.setText("S")  # found; the row stays chosen, quietly
    assert lst.current_row()["data_id"] == 0 and len(chosen) == 1
    lst.set_column_filter("data_id", ("range", 0, 1))
    assert lst.model.rowCount() == 2 and lst.current_row()["data_id"] == 0
    assert lst.step(1) and lst.current_row()["data_id"] == 1 and len(chosen) == 2
    assert not lst.step(1)  # the end
    lst.search.setText("nothing")
    assert lst.model.rowCount() == 0 and lst.current_row() is None


def test_the_arrows_pass_a_spectrum_with_two_populations(lst):
    """components.csv has one row per population: stepping goes row by row, past both rows of a
    spectrum with two (audit of 2026-11-15, C-6: it stayed on its first row)."""
    lst.set_rows(list(COMPONENTS[0]), COMPONENTS, table="components")
    lst.select_index(0)
    seen = []
    while lst.step(1):
        seen.append((lst.current_row()["spectrum_id"], lst.current_row()["rank"]))
    assert seen == [("s1", 1), ("s1", 2), ("s2", 1)]


def test_a_sort_keeps_the_chosen_row(lst):
    """After a sort the row highlighted is still the spectrum chosen (audit of 2026-11-15,
    C-4), and the next step goes from it."""
    lst.set_rows(list(COMPONENTS[0]), COMPONENTS, table="components")
    lst.select_index(0)
    assert lst.current_row()["spectrum_id"] == "s0"
    lst.grid.sortByColumn(lst.model.columns.index("mean"), Qt.SortOrder.DescendingOrder)
    assert lst.model.rows[0]["mean"] == 4.0 and lst.current_row()["spectrum_id"] == "s0"
    at = lst.grid.selectionModel().currentIndex().row()
    assert lst.model.rows[at] is lst.current_row()


def test_the_row_chosen_is_found_again_after_a_reread(lst):
    """The list cleared for a reread of the results and filled again: the row chosen before is
    chosen again, quietly (audit of 2026-11-15, C open question 5)."""
    lst.set_rows(list(COMPONENTS[0]), COMPONENTS, table="components")
    lst.select_index(2)
    chosen = []
    lst.chosen.connect(chosen.append)
    lst.clear("Making components.csv…")
    assert lst.current_row() is None
    again = [dict(r) for r in COMPONENTS]
    lst.set_rows(list(again[0]), again, table="components")
    assert lst.current_row() == COMPONENTS[2] and chosen == []
    lst.clear()
    lst.set_rows(list(SERIES[0]), SERIES, table="series")  # another table: nothing to find
    assert lst.current_row() is None


def test_the_columns_are_sized_from_a_sample_of_rows(lst):
    """The columns take their widths from the header and the first rows, not from every row
    (audit of 2026-11-15, M-2: 1.75 s on a list of 26,000 spectra); a wide value late in the
    list does not widen its column, a long one early does, within a cap."""
    from hxex.gui.data_table import size_columns

    rows = [{"spectrum_id": f"s{i}", "note": ""} for i in range(400)]
    rows[0]["note"] = "a note of some length here"
    rows[399]["note"] = "x" * 500
    lst.set_rows(["spectrum_id", "note"], rows)
    note = lst.grid.columnWidth(1)
    metrics = lst.grid.fontMetrics()
    assert note < metrics.horizontalAdvance("x" * 500)
    assert note > metrics.horizontalAdvance("a note of some length here")
    rows[0]["note"] = ""
    lst.set_rows(["spectrum_id", "note"], rows)
    assert lst.grid.columnWidth(1) < note  # only the header and the empty cells now
    size_columns(lst.grid, sample=400)  # every row: the long value, capped
    assert note < lst.grid.columnWidth(1) <= 60 * metrics.averageCharWidth() + 1
