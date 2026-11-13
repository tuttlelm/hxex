# SPDX-License-Identifier: GPL-3.0-or-later
"""Exporting from the Results tree (hxex.gui.export_dialog; docs/PLAN.md Next steps 32): what
each level exports, the names the dialog shows, and replacing only after asking. Offscreen."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from hxex.gui.export_dialog import HXEXPRESS, PLOT_DATA, ExportDialog, Target  # noqa: E402
from hxex.gui.results_workspace import ResultsWorkspace  # noqa: E402
from hxex.io.hxexpress import parse_hxexpress_filename, read_hxexpress_file  # noqa: E402
from hxex.report import Results  # noqa: E402
from test_run import PEPTIDE, config, raw_spectra, run  # noqa: E402


@pytest.fixture(scope="module")
def results():
    raw = raw_spectra(n_series=2)
    return Results.from_run(run(raw, config(bootstrap=None)), spectra=raw)


@pytest.fixture
def page(qtbot, results):
    w = ResultsWorkspace()
    qtbot.addWidget(w)
    w.show_results(results, None)
    return w


def _dialog(qtbot, results, kind, targets, folder):
    d = ExportDialog(results, kind, targets, folder)
    qtbot.addWidget(d)
    return d


def test_each_level_exports_its_part(page):
    peptide = page.tree.topLevelItem(0)
    targets = page.export_targets(peptide)
    assert [t.data_id for t in targets[HXEXPRESS]] == [0, 1]  # each charge shown
    series = peptide.child(0)
    page.tree.expandItem(series)
    point = series.child(1)  # 1 min, three replicates
    spectrum = point.child(1)
    some = page.export_targets(point)[PLOT_DATA][0]
    assert some.part == "1 min" and len(some.spectrum_ids) == 3
    one = page.export_targets(spectrum)[PLOT_DATA][0]
    assert one.part == "1 min rep 2" and len(one.spectrum_ids) == 1
    assert [t.data_id for t in page.export_targets(spectrum)[HXEXPRESS]] == [0]


def test_the_dialog_names_what_it_writes_and_refuses_bad_affixes(qtbot, results, tmp_path):
    d = _dialog(qtbot, results, HXEXPRESS, [Target(0)], tmp_path)
    assert f"S_1-10-{PEPTIDE}-z2-hxex.xlsx" in d.names.text()
    d.suffix.setText("v-2")
    assert not d.export_button.isEnabled() and "'-'" in d.names.text()
    d.suffix.setText("v2")
    assert d.export_button.isEnabled() and "hxex_v2.xlsx" in d.names.text()


def test_an_export_replaces_files_only_when_the_user_says_so(qtbot, results, tmp_path):
    d = _dialog(qtbot, results, HXEXPRESS, [Target(0), Target(1)], tmp_path)
    assert d.export() == [] and len(d.written) == 2
    path = d.written[0]
    entry = parse_hxexpress_filename(path.name)
    assert len(read_hxexpress_file(path, entry)) == 8
    path.write_bytes(b"mine")
    again = _dialog(qtbot, results, HXEXPRESS, [Target(0)], tmp_path)
    asked = []
    again.ask_replace = lambda there: asked.append(there) or False
    assert again.export() == ["not exported: nothing was replaced"]
    assert asked == [[path]] and path.read_bytes() == b"mine"
    again.ask_replace = lambda there: True
    assert again.export() == [] and len(read_hxexpress_file(path, entry)) == 8


def test_a_time_points_plot_data_go_to_their_own_folder(qtbot, page, results, tmp_path):
    series = page.tree.topLevelItem(0).child(0)
    page.tree.expandItem(series)
    target = page.export_targets(series.child(1))[PLOT_DATA][0]
    d = _dialog(qtbot, results, PLOT_DATA, [target], tmp_path)
    d.ks.setCurrentIndex(d.ks.findData("all"))
    assert d.export() == []
    folder, = d.written
    assert folder.name.endswith("_1_min_plot_data")
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        ["about.csv", "bins.csv", "components.csv", "raw.csv", "spectra.csv", "uptake.csv"])
    assert d.existing() == [folder]


def test_the_right_click_menu_names_each_levels_export(page):
    peptide = page.tree.topLevelItem(0)
    series = peptide.child(0)
    page.tree.expandItem(series)
    seen = []
    for item in (peptide, series, series.child(1), series.child(1).child(0)):
        menu, actions = page.export_menu(item)
        seen.append([a.text() for a in menu.actions()])
        assert [kind for kind, _ in actions.values()] == [HXEXPRESS, PLOT_DATA]
    assert seen == [
        ["Export its 2 charge states for HX-Express…", "Export plot data…"],
        ["Export for HX-Express…", "Export plot data…"],
        ["Export the series for HX-Express…", "Export this time point's plot data…"],
        ["Export the series for HX-Express…", "Export this spectrum's plot data…"]]
