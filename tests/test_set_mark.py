# SPDX-License-Identifier: GPL-3.0-or-later
"""A spectrum whose K differs from its set's (owner, 2026-11-17): with the population test, the
set's number of populations (every replicate and charge state of a peptide in one sample at one
time, :func:`hxex.analysis.set_test.spectrum_sets`) is shown beside the spectrum's own K, in
``spectra.csv`` (``set_K``), the fits at each K, and the window's tree; nothing else changes."""

import dataclasses

import numpy as np
import pytest

from hxex.analysis import AnalysisConfig, SpeciesFilters
from hxex.analysis.set_test import SetTest, set_ks, set_words, spectrum_sets
from hxex.population_test import PopulationTestConfig
from hxex.report import Results
from hxex.report.overview import k_comparison_data, plot_k_comparison, set_note
from hxex.report.tables import spectra_table, write_tables
from test_population_test import _forced
from test_run import config, raw_spectra, run


@pytest.fixture(scope="module")
def tested():
    """A run with the population test whose labelled spectra each read one population while
    their sets read two (the test's statistics forced, the spectra's own K left as they are)."""
    res = Results.from_run(run(raw_spectra(n_series=2),
                               config(population_test=PopulationTestConfig(simulations=4))))
    labelled = np.flatnonzero(res.fit.role == "")
    return _forced(res, labelled)


def _labelled(res):
    return np.flatnonzero(np.asarray(res.fit.role) == "")


def test_each_spectrum_has_its_sets_test_and_its_own_k_stays(tested):
    sets = spectrum_sets(tested)
    assert len(sets) == len(tested.fit) and all(t is not None for t in sets)
    ks = set_ks(sets, len(tested.fit))
    labelled = _labelled(tested)
    assert (ks[labelled] == 2).all() and (np.delete(ks, labelled) == 1).all()
    assert (tested.effective_K == 1).all()  # information only: no K changes
    assert spectrum_sets(tested) is sets  # worked out once per results and settings
    # a time the analysis leaves out has no sets; a run without the test has none at all
    t = float(tested.fit.time[labelled[0]])
    left = spectrum_sets(tested, AnalysisConfig(exclude_times=(t,)))
    assert all(left[i] is None for i in labelled)
    assert spectrum_sets(dataclasses.replace(tested, population_test=None)) is None
    assert spectrum_sets(tested, AnalysisConfig(populations="best_set")) is None
    # a rejected spectrum is in no set
    from hxex.io.bundle import ReviewTable

    sid = str(tested.fit.spectrum_id[labelled[0]])
    rejected = tested.with_review(ReviewTable.empty().update(sid, status="rejected"))
    assert spectrum_sets(rejected)[int(labelled[0])] is None
    # the analysis's filters decide the sets: undeuterated_cap keeps a peptide whose
    # undeuterated set reads two at one population
    undeut = np.flatnonzero(np.asarray(tested.fit.role) == "undeut")
    both = _forced(tested, np.concatenate([undeut, labelled]))
    capped = spectrum_sets(both, AnalysisConfig(filters=SpeciesFilters(undeuterated_cap=True)))
    assert all(capped[i].K == 1 for i in labelled)


def test_a_sets_words_name_its_populations_and_p():
    two = SetTest(K=2, p=(1e-85, 0.4), added=(0.2, 0.01), n_spectra=4, n_tested=4,
                  undeuterated_two=False, reason="")
    assert set_words(two) == "2 populations (p 1.0e-85)"
    one = SetTest(K=1, p=(0.3,), added=(0.05,), n_spectra=4, n_tested=4,
                  undeuterated_two=False, reason="")
    assert set_words(one, 0.01) == "1 population (p 0.3)"
    small = dataclasses.replace(one, p=(1e-5,))  # the p passes, the share does not
    assert set_words(small, 0.01) == "1 population (p 1.0e-05, population 5 %)"
    untested = dataclasses.replace(one, p=(np.nan,))
    assert set_words(untested) == "1 population (not tested)"


def test_spectra_csv_has_the_sets_k_beside_the_spectrums(tested, tmp_path):
    table = spectra_table(tested)
    columns = list(table)
    assert columns[columns.index("K") + 1] == "set_K"
    labelled = _labelled(tested)
    assert all(table["set_K"][i] == 2 for i in labelled)
    assert (table["K"] == 1).all()
    # a few rows give the same values; sets worked out before are taken as they are
    part = spectra_table(tested, rows=labelled[:2])
    assert list(part["set_K"]) == [2, 2]
    assert list(spectra_table(tested, rows=labelled[:2], sets=None)["set_K"]) == ["", ""]
    # empty without the population test
    bare = spectra_table(dataclasses.replace(tested, population_test=None))
    assert set(bare["set_K"]) == {""}
    # hxex report: the analysis settings decide the sets
    write_tables(tested, tmp_path / "default")
    write_tables(tested, tmp_path / "other", analysis=AnalysisConfig(populations="best_set"))
    import csv

    def column(folder):
        with open(tmp_path / folder / "spectra.csv", newline="") as f:
            return [row["set_K"] for row in csv.DictReader(f)]

    assert sorted(set(column("default"))) == ["1", "2"]
    assert set(column("other")) == {""}


def test_the_fits_at_each_k_say_the_sets_number_where_it_differs(tested):
    labelled = _labelled(tested)
    sid = str(tested.fit.spectrum_id[labelled[0]])
    d = k_comparison_data(sid, tested)
    assert d["set"].K == 2
    note = set_note(d)
    assert note.startswith("population test (set) p-value ") and note.endswith(
        ": K = 1 (self), K = 2 (set)")
    fig = plot_k_comparison(sid, tested)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    k1, k2 = titles[0], titles[1]
    assert "(set)" not in k1
    lines = k2.splitlines()
    assert any(line.startswith("population test (set)") for line in lines)
    assert lines[-1].startswith("K=2")  # the K line stays last
    # where they agree: nothing said
    undeut = int(np.flatnonzero(np.asarray(tested.fit.role) == "undeut")[0])
    agree = k_comparison_data(str(tested.fit.spectrum_id[undeut]), tested)
    assert agree["set"].K == 1 and set_note(agree) is None
    # sets given (the window's), or none
    assert k_comparison_data(sid, tested, sets=None)["set"] is None


def test_the_tree_marks_a_spectrum_that_differs_from_its_set(qtbot, tested):
    pytest.importorskip("PySide6")
    from hxex.gui.results_tree import points
    from hxex.gui.results_workspace import COLUMN, PAYLOAD, ResultsWorkspace

    sets = spectrum_sets(tested)
    labelled = points(tested, 0, sets=sets)[1]
    assert labelled.ks == "1 1 1 (set 2)"
    assert labelled.spectra[0].k_text == "K = 1 (self), K = 2 (set)"
    assert "2 populations" in labelled.spectra[0].k_tip
    assert points(tested, 0, sets=sets)[0].ks == "1 1 1"  # the undeuterated set agrees
    assert points(tested, 0)[1].ks == "1 1 1"  # no sets given: no mark

    page = ResultsWorkspace()
    qtbot.addWidget(page)
    page.show_results(tested, None)
    series = page.tree.topLevelItem(0).child(0)
    page.tree.expandItem(series)
    point = series.child(1)
    assert point.text(COLUMN["K"]) == "1 1 1 (set 2)"
    leaf = point.child(0)
    assert leaf.text(COLUMN["K"]) == "K = 1 (self), K = 2 (set)"
    assert "population test (set)" in leaf.toolTip(COLUMN["K"])
    assert series.child(0).child(0).text(COLUMN["K"]) == "1"
    assert leaf.toolTip(COLUMN["K"]) and not series.child(0).child(0).toolTip(COLUMN["K"])
    # a decision that gives the spectrum its set's K: the mark goes, the set's K stays
    from hxex.io.bundle import ReviewTable

    spectrum = leaf.data(0, PAYLOAD)[3]
    decided = tested.with_review(ReviewTable.empty().update(spectrum.spectrum_id,
                                                           k_override=2))
    page.results = decided
    page._reviewed([spectrum.index])
    assert leaf.text(COLUMN["K"]) == "2" and point.text(COLUMN["K"]) == "2 1 1 (set 2)"
    # the data tab's row of spectra.csv names the set's K
    from hxex.gui.spectra_list import list_rows

    columns, rows = list_rows(tested, sets=page.sets)
    assert columns[columns.index("K") + 1] == "set_K"
    assert {r["set_K"] for r in rows if r["role"] == ""} == {2}
