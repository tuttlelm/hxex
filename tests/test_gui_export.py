# SPDX-License-Identifier: GPL-3.0-or-later
"""Exporting from the Results view (hxex.gui.export_dialog; docs/PLAN.md Next steps 32): what
each level exports, the names the dialog shows, and replacing only after asking. Offscreen."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from hxex.gui.export_dialog import HXEXPRESS, PLOT_DATA, ExportDialog, Target  # noqa: E402
from hxex.gui.results_workspace import ResultsWorkspace  # noqa: E402
from hxex.io.hxexpress import parse_hxexpress_filename, read_hxexpress_file  # noqa: E402
from hxex.report import Results  # noqa: E402
from test_run import PEPTIDE, boot_config, config, raw_spectra, run  # noqa: E402


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
    # the two test series share their label: the series' number keeps their files apart
    assert d.names.text() == f"Writes 1 file: S_1-10-{PEPTIDE}-z2-hxex_id0.xlsx"
    d.suffix.setText("v-2")
    assert not d.export_button.isEnabled() and "'-'" in d.names.text()
    d.suffix.setText("v2")
    assert d.export_button.isEnabled() and "hxex_id0_v2.xlsx" in d.names.text()
    d.suffix.setText("x" * 260)  # a name too long for the file system (audit E-7)
    assert not d.export_button.isEnabled() and "characters long" in d.names.text()
    assert d.export() and not d.written


def test_an_export_replaces_files_only_when_the_user_says_so(qtbot, results, tmp_path):
    d = _dialog(qtbot, results, HXEXPRESS, [Target(0), Target(1)], tmp_path)
    assert d.export() == [] and len(d.written) == 2
    path = d.written[0]
    entry = parse_hxexpress_filename(path.name)
    assert len(read_hxexpress_file(path, entry)) == 8
    path.write_bytes(b"mine")
    again = _dialog(qtbot, results, HXEXPRESS, [Target(0)], tmp_path)
    asked = []
    again.ask_replace = lambda there, left=(): asked.append(there) or False
    assert again.export() == ["not exported: nothing was replaced"]
    assert asked == [[path]] and path.read_bytes() == b"mine"
    again.ask_replace = lambda there, left=(): True
    assert again.export() == [] and len(read_hxexpress_file(path, entry)) == 8


def test_a_time_points_plot_data_are_named_by_it(qtbot, page, results, tmp_path):
    series = page.tree.topLevelItem(0).child(0)
    page.tree.expandItem(series)
    target = page.export_targets(series.child(1))[PLOT_DATA][0]
    d = _dialog(qtbot, results, PLOT_DATA, [target], tmp_path)
    d.ks.setCurrentIndex(d.ks.findData("all"))
    name = d.name_of(target)
    assert name.endswith("_z2_id0_1min")
    assert d.names.text().startswith(f"Writes 6 CSV files named {name}")
    assert d.export() == []
    assert sorted(p.name for p in d.written) == sorted(
        f"{name}_{t}.csv" for t in ("about", "bins", "components", "raw", "spectra", "uptake"))
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(p.name for p in d.written)
    assert sorted(d.existing()) == sorted(d.written)


def test_the_export_button_names_each_levels_export(page, qtbot, monkeypatch):
    """**Export data** sits beside **Save figure** on the selection's figures (not the run's
    overview), and its menu names what the selected row exports; the tree has no right-click."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QMenu

    assert page.tree.contextMenuPolicy() != Qt.ContextMenuPolicy.CustomContextMenu
    panel = page.panel("spectra")
    bar = panel._bar_layout
    button = bar.itemAt(bar.indexOf(panel.save_button) + 1).widget()
    assert button.text() == "Export data" and button.menu() is not None
    assert all(w.text() != "Export data" for w in page.panel("overview")._options
               if hasattr(w, "text"))
    menu = QMenu()
    page.tree.setCurrentItem(None)
    assert page.fill_export_menu(menu) == {} and not menu.actions()[0].isEnabled()
    peptide = page.tree.topLevelItem(0)
    series = peptide.child(0)
    page.tree.expandItem(series)
    seen = []
    for item in (peptide, series, series.child(1), series.child(1).child(0)):
        page.tree.setCurrentItem(item)
        actions = page.fill_export_menu(menu)
        seen.append([a.text() for a in menu.actions()])
        assert [kind for kind, _ in actions.values()] == [HXEXPRESS, PLOT_DATA]
    assert seen == [
        ["Export its 2 charge states for HX-Express…", "Export plot data…"],
        ["Export for HX-Express…", "Export plot data…"],
        ["Export the series for HX-Express…", "Export this time point's plot data…"],
        ["Export the series for HX-Express…", "Export this spectrum's plot data…"]]
    opened = []
    monkeypatch.setattr(page, "open_export", lambda kind, targets: opened.append(kind))
    menu.actions()[1].trigger()  # the spectrum's plot data
    assert opened == [PLOT_DATA]


def test_two_targets_of_one_name_are_refused(qtbot, results, tmp_path):
    """Audit E-2: an export never writes two things into one file."""
    d = _dialog(qtbot, results, HXEXPRESS, [Target(0), Target(1), Target(0)], tmp_path)
    assert not d.export_button.isEnabled()
    assert "would be written as" in d.names.text() and "hxex_id0.xlsx" in d.names.text()
    assert d.export() and not d.written and not list(tmp_path.iterdir())


def test_a_series_below_residue_0_is_exported_with_a_warning(qtbot, results, tmp_path):
    """Audit E-7: the dialog opens, warns that hxex's reader cannot take the peptide from the
    file name, and writes the file all the same."""
    import dataclasses

    import numpy as np

    b = results.binned
    tagged = dataclasses.replace(results, binned=dataclasses.replace(
        b, start_seq=np.where(b.data_id == 0, -3, b.start_seq),
        end_seq=np.where(b.data_id == 0, 6, b.end_seq)))
    d = _dialog(qtbot, tagged, HXEXPRESS, [Target(0)], tmp_path)
    assert d.export_button.isEnabled()
    assert f"S_-3-6-{PEPTIDE}-z2-hxex.xlsx" in d.names.text()
    assert "Warning:" in d.names.text() and "below residue 0" in d.names.text()
    assert d.export() == [] and [p.name for p in d.written] == [
        f"S_-3-6-{PEPTIDE}-z2-hxex.xlsx"]


def test_a_relative_folder_is_counted_from_the_results_folder(qtbot, results, tmp_path,
                                                             monkeypatch):
    """Audit E-18: a folder typed without its full location is the results' folder's, as the
    suggested one is, never the folder the window was started in; the dialog shows where."""
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "project"
    d = ExportDialog(results, HXEXPRESS, [Target(0)], home / "hxex_exports", base=home)
    qtbot.addWidget(d)
    d.folder.setText("exports")
    assert d.folder_path() == home / "exports"
    assert d.where.text() == f"→ {home / 'exports'}"
    assert d.export() == [] and d.written[0].parent == home / "exports"
    assert not (tmp_path / "exports").exists()
    d.folder.setText(str(tmp_path / "full"))
    assert d.folder_path() == tmp_path / "full" and d.where.text() == ""


def test_the_replace_question_names_this_exports_files_and_the_earlier_ones_left(
        qtbot, results, tmp_path):
    """Audit E-5 and E-12: the count and the question are of the files this export writes; a
    file of an earlier export of the same name that this one does not write is named as left
    from it, and nothing is deleted."""
    target = Target(0)
    d = _dialog(qtbot, results, PLOT_DATA, [target], tmp_path)
    name = d.name_of(target)
    assert d.names.text().startswith(f"Writes 6 CSV files named {name}")
    assert d.export() == [] and len(d.written) == 6
    earlier = tmp_path / f"{name}_components_bootstrap.csv"
    earlier.write_text("mine")
    again = _dialog(qtbot, results, PLOT_DATA, [target], tmp_path)
    asked = []
    again.ask_replace = lambda there, left=(): asked.append((there, left)) or True
    assert again.export() == []
    (there, left), = asked
    assert sorted(there) == sorted(d.written) and left == [earlier]
    text = again.replace_text(there, left)
    assert text.startswith("6 of the files this export writes already exist")
    assert f"{name}_components_bootstrap.csv" in text and "earlier export" in text
    assert earlier.read_text() == "mine"


def test_the_count_is_of_the_spectrums_own_tables(qtbot, tmp_path):
    """Audit E-12: a spectrum whose K in effect has no bootstrap has no bootstrap cloud, so the
    dialog says six files where the results kept resamples of others, and writes six."""
    from hxex.io import ReviewTable

    raw = raw_spectra(n_series=1)
    kept = Results.from_run(run(raw, config(bootstrap=boot_config(keep_draws=True))),
                            spectra=raw)
    fit = kept.fit
    i = 3
    other = next(k for k in (1, 2) if k != int(kept.effective_K[i]) and fit.fitted(k)[i])
    sid = str(fit.spectrum_id[i])
    reviewed = kept.with_review(ReviewTable.empty().update(sid, k_override=other))
    target = Target(0, [sid], "1 min rep 1")
    d = _dialog(qtbot, reviewed, PLOT_DATA, [target], tmp_path)
    assert d.names.text().startswith("Writes 6 CSV files named")
    assert d.export() == [] and len(d.written) == 6
    whole = _dialog(qtbot, reviewed, PLOT_DATA, [Target(0)], tmp_path)
    assert whole.names.text().startswith("Writes 7 CSV files named")
