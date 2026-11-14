# SPDX-License-Identifier: GPL-3.0-or-later
"""The Results workspace (docs/PLAN.md, "GUI", "Results and review"): the results as a tree,
peptide, then its charges (the series), then the time points, then the replicates (the
spectra), with filters and a search; and the figures of what is selected:

- a peptide or a series: the series' pages (spectra, uptake, populations), as ``hxex report``
  draws them;
- a time point: its replicates side by side (the spectra page cut to that time, on the
  series' m/z range), with the series' uptake;
- a spectrum: the spectrum itself, its fits at each K, and the series' uptake.

The children of a series are made when it is opened, so a large run opens quickly.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .data_table import DataPanel, cell_text, flag_tip
from .figure_view import FigurePanel
from .order import natural_key
from .results_tree import FILTERS, Peptide, left_out, left_out_counts, peptides, points, shown
from .review_panel import ReviewPanel
from .spectra_list import SpectraList, list_rows
from .tasks import Latest
from .tips import tip
from .window_edit import has_fragments, ions_legend, spread_files, view_tips, view_words
from .window_editor import set_choice_tips

log = logging.getLogger(__name__)

TIME_AXES = {"log": "log time axis", "linear": "linear time axis",
             "index": "time points evenly spaced"}
COLUMNS = ("peptide, charge, time, replicate", "sample", "identity", "ion", "MS level",
           "condition", "parent", "spectra", "flagged", "rejected", "left out", "K > 1", "K",
           "review", "flags")
# the columns of fragment series, shown only for results read from mzML or Thermo files
TARGET_COLUMNS = ("ion", "MS level", "condition", "parent")
COLUMN = {name: i for i, name in enumerate(COLUMNS)}
# the identity levels of pooled peptides, in the order the filter lists them
IDENTITY_LEVELS = {"isoform": "peptides only one record of the pooled FASTA holds",
                   "class": "peptides several records of one class hold",
                   "ambiguous": "peptides records of several classes hold",
                   "none": "peptides no record holds"}
# what each column holds, as its header's tooltip
COLUMN_TIPS = {
    "identity": "what a pooled peptide may be ([input] pooled_fasta): the one record whose "
                "sequence holds it, else the class all its records fall in, else the classes; "
                "its level and every candidate in the tooltip",
    "ion": "the fragment ion (c-1:4: ion type and length; M: the precursor), empty for a "
           "peptide read at MS1",
    "MS level": "the MS level of the series' scans (1 for peptides, 2 or 3 for fragments)",
    "condition": "the fragment list's label ([[input.fragments]] label), else its scan "
                 "conditions",
    "parent": "the parent peptide the fragment was taken from: its residues and charge (one "
              "fragment can come from two parents)",
    "spectra": "how many spectra (replicates) were fitted",
    "flagged": "spectra with a quality flag (named in the last column); a flag alone leaves "
               "nothing out",
    "rejected": "spectra rejected in the review (the review table, [[review.exclude]] rules, "
                "or the bundle's own review); the analysis leaves them out",
    "left out": "spectra the analysis settings leave out, and why ([analysis.filters], "
                "exclude_flagged, exclude_times; the project file's settings, else the "
                "defaults), as hxex analyse lists them in left_out_spectra.csv; accepting one "
                "in the review keeps it against [analysis.filters]. 'kept at K = 1' is not counted",
    "K > 1": "spectra fitted with more than one population (rejected ones not counted)",
    "K": "the number of populations of each replicate's fit",
    "review": "the review's decision: accepted, rejected or none yet",
    "flags": "the quality flags of the spectrum (hover one for its meaning; the guide's "
             "flags.md, or hxex explain NAME, has them all)"}
KIND, PAYLOAD = Qt.ItemDataRole.UserRole, Qt.ItemDataRole.UserRole + 1
DEPTH = {"peptide": 0, "series": 1, "point": 2, "spectrum": 3}  # a kind of row's level


class ReviewTree(QTreeWidget):
    """The results tree, whose keys A, R and U are review decisions (in its own key handling: an
    event filter is given a plain event by PySide6 6.7, without the key's modifiers)."""

    review_key = Signal(str)

    def keyPressEvent(self, event):  # noqa: N802 - Qt's name
        if event.modifiers() in (Qt.KeyboardModifier.NoModifier,
                                 Qt.KeyboardModifier.ShiftModifier) \
                and event.text().lower() in ("a", "r", "u"):
            self.review_key.emit(event.text())
            event.accept()
            return
        super().keyPressEvent(event)
# the figure tabs of each kind of selection, in their order
SERIES_TABLES = ("uptake.csv", "components.csv", "spectra.csv")  # a series' rows of these
SERIES_PAGES = ("spectra", "uptake", "populations")  # a series' pages, as hxex report draws
TABS = {"start": ("overview",),  # results open, nothing selected yet: the first look
        "series": (*SERIES_PAGES, "data", "overview"),
        "point": ("spectra", "fits at each K", "uptake", "data", "overview"),
        "spectrum": ("spectrum", "fits at each K", "elution", "uptake", "data", "overview")}
# shown only for spectra read from scan files (mzML, Thermo raw): read again when in view
TRACE_TABS = ("elution",)
# the figure tabs' tooltips (owner, 2026-11-13): a series is a peptide at one charge in one
# sample, with its time points, replicates, and controls; a spectrum is one replicate
TAB_TIPS = {
    "spectra": "the raw spectrum, the picked peaks, and the fit of each spectrum of the selection",
    "spectrum": "this spectrum's raw data, picked peaks, and fit",
    "fits at each K": "the same spectra fitted with 1, 2, 3 … populations, side by side",
    "elution": "where this spectrum was summed in retention time (and drift); change its window "
               "here",
    "uptake": "the series' deuterium against time (or order), every replicate and the controls",
    "populations": "the components: each population's deuterium level and share, one panel per "
                   "spectrum",
    "data": "the rows behind the figures: spectra, populations, or uptake for a series; the "
            "spectra of a time point; one spectrum's values",
    "overview": "the whole results at a glance: populations chosen, controls, quality flags (it "
                "doesn't follow the selection)",
}
# the pages of the overview tab, as hxex report --figures writes them (overview_<kind>)
OVERVIEW_PAGES = {"k_selection": "K selection", "controls": "controls",
                  "quality": "quality flags"}


class ResultsWorkspace(QWidget):
    """The results of a run as a tree, and the figures of the selection."""

    selection_changed = Signal()

    def __init__(self, window=None):
        super().__init__()
        self.window = window
        self.results = None
        self.left: list[str] | None = None  # why the analysis settings leave each spectrum out
        self.analysis_config = None  # the settings that says so
        self._left_task = Latest(self)
        self.tree_data: list[Peptide] = []
        self.path_label = QLabel("No results open.")
        open_button = QPushButton("Open results…")
        open_button.setToolTip(tip("a results bundle: a folder, or a .zip"))
        if window is not None:
            open_button.clicked.connect(window.open_results_dialog)
        top = QHBoxLayout()
        top.addWidget(self.path_label, 1)
        top.addWidget(open_button)

        self.filter = QComboBox()
        for i, (key, (text, meaning)) in enumerate(FILTERS.items()):
            self.filter.addItem(text, key)
            self.filter.setItemData(i, tip(meaning), Qt.ItemDataRole.ToolTipRole)
        self.filter.setToolTip(tip("which series the tree shows"))
        self.filter.currentIndexChanged.connect(self._fill)
        # MS level and ion type: shown for results with fragment series
        self.ms_filter = QComboBox()
        self.ms_filter.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.ms_filter.setToolTip(tip("the MS level of the series the tree shows"))
        self.ms_filter.currentIndexChanged.connect(self._fill)
        self.ion_filter = QComboBox()
        self.ion_filter.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.ion_filter.setToolTip(tip("the ion type of the series the tree shows (peptides: the "
                                       "peptides read at MS1)"))
        self.ion_filter.currentIndexChanged.connect(self._fill)
        # identity level and class: shown for pooled results with candidates
        self.level_filter = QComboBox()
        self.level_filter.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.level_filter.setToolTip(tip("what the pooled peptides the tree shows may be: one "
                                         "isoform, one class, several classes (ambiguous), or "
                                         "nothing in the pooled FASTA"))
        self.level_filter.currentIndexChanged.connect(self._fill)
        self.class_filter = QComboBox()
        self.class_filter.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.class_filter.setToolTip(tip("the class of the pooled peptides the tree shows ([input] "
                                         "pooled_classes; an ambiguous peptide is in each of its "
                                         "classes)"))
        self.class_filter.currentIndexChanged.connect(self._fill)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a peptide, residues, sample, or fragment")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._fill)
        self.count = QLabel()  # x of y series: on the status line under the tree
        # the choices wrap onto further lines when the tree is narrow, the search box taking
        # the rest of its line (owner, 2026-11-08)
        from .flow_layout import FlowLayout

        self.filters = QWidget()
        filters = FlowLayout(self.filters)
        filters.addWidget(self.filter)
        filters.addWidget(self.ms_filter)
        filters.addWidget(self.ion_filter)
        filters.addWidget(self.level_filter)
        filters.addWidget(self.class_filter)
        self.search.setMinimumWidth(200)
        filters.add_filling(self.search)

        self.tree = ReviewTree()
        self.tree.setColumnCount(len(COLUMNS))
        self.tree.setHeaderLabels(list(COLUMNS))
        for name, words in COLUMN_TIPS.items():
            self.tree.headerItem().setToolTip(COLUMN[name], tip(words))
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemExpanded.connect(self._expanded)
        self.tree.currentItemChanged.connect(lambda item, _previous: self._selected(
            item, again=False))
        self.export_folder: Path | None = None  # the folder the last export went to
        self.review = ReviewPanel(self)
        self.review.decided.connect(self._reviewed)
        self.review.step.connect(self.step_spectrum)
        if window is not None:
            self.review.resume_asked.connect(window.run_resume)
        # review from the tree: A accepted, R rejected, U not reviewed (the tree's own keys, so
        # they work whenever it has the keyboard)
        self.tree.review_key.connect(self.review.key)
        self._leaves: dict[int, QTreeWidgetItem] = {}  # a spectrum's row in the tree
        self._series_items: dict[int, QTreeWidgetItem] = {}
        # the tree, and beside it every spectrum as a list (owner, 2026-11-03): a row chosen in
        # the list selects that spectrum in the tree, so the figures and the review are the same
        tree_page = QWidget()
        tree_side = QVBoxLayout(tree_page)
        tree_side.setContentsMargins(0, 0, 0, 0)
        tree_side.addWidget(self.filters)
        tree_side.addWidget(self.tree, 1)
        self.list = SpectraList()
        self.list.chosen.connect(self._list_chosen)
        self.list.table_asked.connect(self._list_table)
        self.list.update_asked.connect(lambda: self._list_table(self.list.table_name, True))
        self._list_tables: dict[str, tuple] = {}  # table -> (results, columns, rows) made
        # the other tables made apart from the spectra's: one slot kept only the latest
        # request, so a reread of the results dropped an Update under way (owner: stuck)
        self._table_task = Latest(self)
        self.list.grid.review_key.connect(self.review.key)
        self._list_task = Latest(self)
        self.left_tabs = QTabWidget()
        self.left_tabs.addTab(tree_page, "Tree")
        self.left_tabs.addTab(self.list, "List")
        self.left_tabs.setTabToolTip(0, tip("peptides, their charges, time points and replicates"))
        self.left_tabs.setTabToolTip(1, tip("every spectrum, one row each, with the columns of "
                                            "spectra.csv: sort by a column, filter, and step "
                                            "through them in that order"))
        # a status line under the list: how many series (Tree) or spectra (List) are shown
        # (owner, 2026-11-08: in the rows of choices they kept the panel wide)
        self.left_status = QWidget()
        status = QHBoxLayout(self.left_status)
        status.setContentsMargins(4, 0, 4, 0)
        status.addWidget(self.count)
        status.addWidget(self.list.count)
        status.addStretch()
        self.left_tabs.currentChanged.connect(self._left_tab_changed)
        left = QWidget()
        side = QVBoxLayout(left)
        side.setContentsMargins(0, 0, 0, 0)
        side.addWidget(self.left_tabs, 1)
        side.addWidget(self.left_status)
        self._left_tab_changed(0)
        # the "left out" column is worked out again only when asked (owner: no recalculation
        # in the middle of a review session)
        self.left_note = QLabel()
        self.left_note.setWordWrap(True)
        self.left_update = QPushButton("Update")
        self.left_update.setToolTip(tip("work out the 'left out' column again with your decisions "
                                        "and the project's [analysis] settings (a few seconds on "
                                        "a large run)"))
        self.left_update.clicked.connect(self.update_left_out)
        stale = QHBoxLayout()
        stale.setContentsMargins(0, 0, 0, 0)
        stale.addWidget(self.left_note, 1)
        stale.addWidget(self.left_update)
        self.left_bar = QWidget()
        self.left_bar.setLayout(stale)
        self.left_bar.hide()
        side.addWidget(self.left_bar)
        side.addWidget(self.review)

        # the uptake page's time axis, for the window only (owner: the project's linear default
        # is not how uptake is usually read); the project's [figures] setting first
        self.time_axis = QComboBox()
        self.time_axis.setToolTip(tip("how the uptake page spreads the times: the project file's "
                                      "[figures] scale and x, or for this window only log, linear, "
                                      "or one even step per time point"))
        self.time_axis.currentIndexChanged.connect(self._redraw)
        # the overview tab's page: the run's summary pages, one at a time (owner, 2026-11-04)
        self.overview_page = QComboBox()
        self.overview_page.setToolTip(tip("which page of the overview: as hxex report --figures "
                                          "writes them (overview)"))
        self.overview_page.currentIndexChanged.connect(lambda _i: self._draw_overview())
        # the spectra tab's page, for a series on several pages ([figures] page_rows,
        # page_columns, max_points; owner, 2026-11-04: a kinetics run made one 214-inch page)
        self.spectra_page = QComboBox()
        self.spectra_page.setToolTip(tip("which page of the series' spectra ([figures] page_rows "
                                         "and page_columns; max_points thins a long time course)"))
        self.spectra_page.currentIndexChanged.connect(lambda _i: self._draw_spectra_page())
        self.spectra_page.hide()
        # the populations tab goes on over the same pages: its own list, kept in step
        self.population_page = QComboBox()
        self.population_page.setToolTip(tip("which page of the series' populations (the same pages "
                                            "as its spectra)"))
        self.population_page.currentIndexChanged.connect(self.spectra_page.setCurrentIndex)
        self.population_page.hide()
        self._populations_paged = False  # the series' populations page is drawn per page
        self._spectra_job = None  # (results, data_id, config, pages) of the series shown
        self._land_on_last_page = False  # stepping back into a series: its last page
        for box in (self.time_axis, self.overview_page, self.spectra_page,
                    self.population_page):  # as wide as needed
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._overview_figures: dict[str, object] = {}  # page -> figure, for _overview_of
        self._overview_of = None  # the results those figures were drawn from
        self.figures = QTabWidget()
        self.figures.currentChanged.connect(lambda _i: self._draw_overview())
        # the selection's rows of the report's tables (owner, 2026-11-04): a spectrum's
        # spectra.csv row, a time point's replicates', a series' uptake.csv rows
        self.table_view = DataPanel()
        self.table_view.activated.connect(self._table_row_chosen)
        # a series' rows of which table: uptake.csv (its uptake page), components.csv (its
        # populations page), or spectra.csv (its spectra)
        self.series_table = QComboBox()
        self.series_table.addItems(SERIES_TABLES)
        self.series_table.setToolTip(
            tip("which of the report's tables to list this series' rows of"))
        self.series_table.currentTextChanged.connect(self._series_table_changed)
        self.series_table.hide()
        self.table_view.add_option(self.series_table)
        self._table_for = None  # (results, kind, payload) the table shows
        self.figures.currentChanged.connect(lambda _i: self._fill_table())
        # the selected spectrum's elution trace, read from its scan files when in view
        # (PLAN Next steps 26): the file first, then the files compared behind it
        self._trace_task = Latest(self)
        self._trace_reader = None  # (read record, TraceReader) for the results shown
        self._trace_for = None  # (results, spectrum id) the elution tab shows
        self._shown_traces = None  # (results, spectrum id, data id, traces, title) drawn there
        self.window_editor = None  # the elution tab's window editing (made with its panel)
        # what the elution tab draws ("Show elution traces for", numbered as the save levels;
        # owner, 2026-11-12: the two lists are independent): this file's trace alone, this
        # peptide's other charge states (ions) in this file (do they elute together: a window
        # for the whole peptide, or for this one only?), or this charge state (ion) in the
        # sample's other files (does this file need a window of its own?); worded for the
        # project (window_edit.view_words)
        self.compare = QComboBox()
        for data, text in view_words(False).items():
            self.compare.addItem(text, data)
        self.compare.setCurrentIndex(self.compare.findData("replicates"))
        set_choice_tips(self.compare, view_tips(False))
        self.compare.currentIndexChanged.connect(self._compare_changed)
        self.compare_box = QWidget()
        row = QHBoxLayout(self.compare_box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Show elution traces for"))
        row.addWidget(self.compare)
        row.addStretch(1)
        self._drawn = None  # (kind, series, time point or spectrum) the figures show
        self.figures.currentChanged.connect(lambda _i: self._draw_retention())
        self.panels: dict[str, FigurePanel] = {}
        self.render = getattr(window, "render", "matplotlib")
        self.kind = None
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(self.figures)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)  # a panel dragged to nothing left no trace
        split.setSizes([560, 900])
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(split, 1)
        self._show_tabs("start")

    # --- the figures' tabs -----------------------------------------------------------------

    def panel(self, name: str) -> FigurePanel:
        if name not in self.panels:
            panel = FigurePanel(render=self.render)
            if name != "overview":  # the overview is the whole run's, not the selection's
                panel.add_option(self._export_button())
            if name == "uptake":  # the only page with a time axis: its choice in its own bar
                panel.add_option(self.time_axis)
            if name == "overview":
                panel.add_option(self.overview_page)
            if name == "spectra":
                panel.add_option(self.spectra_page)
            if name == "populations":
                panel.add_option(self.population_page)
            if name == "elution":  # its window can be changed there (PLAN Next steps 26)
                from .window_editor import WindowEditor

                panel.add_row(self.compare_box)
                self.window_editor = WindowEditor(panel, self._trace_context,
                                                  self._draw_plain_trace, self)
                if self.window is not None:
                    self.window_editor.message.connect(
                        lambda text: self.window.statusBar().showMessage(text, 15000))
            if self.window is not None:
                panel.clicked.connect(
                    lambda gid: self.window.statusBar().showMessage(gid, 5000))
            self.panels[name] = panel
        return self.panels[name]

    def set_render(self, render: str) -> None:
        """How the figures are drawn: ``'svg'`` (Qt's SVG renderer) or ``'matplotlib'``."""
        self.render = render
        for panel in self.panels.values():
            panel.set_render(render)

    def _show_tabs(self, kind: str) -> None:
        if kind == self.kind:
            return
        keys = getattr(self, "_tab_keys", ())
        at = self.figures.currentIndex()
        current = keys[at] if 0 <= at < len(keys) else None
        self.figures.blockSignals(True)
        while self.figures.count():
            self.figures.removeTab(0)
        names = tuple(n for n in TABS[kind] if n not in TRACE_TABS or self._traceable())
        for name in names:
            at = self.figures.addTab(self.table_view if name == "data" else self.panel(name), name)
            self.figures.setTabToolTip(at, tip(TAB_TIPS[name]))
        self._tab_keys = names  # the data tab's text carries its count
        if current in names:
            self.figures.setCurrentIndex(names.index(current))
        self.figures.blockSignals(False)
        self.kind = kind

    def _data_count(self, n: int | None) -> None:
        """The data tab named with its rows, ``data (12)``, as in Analysis (owner, 2026-11-05);
        ``data`` before it is filled."""
        at = self.figures.indexOf(self.table_view)
        if at >= 0:
            self.figures.setTabText(at, f"data ({n})" if n else "data")

    # --- the tree --------------------------------------------------------------------------

    def show_results(self, results, path: Path | None, left: list[str] | None = None,
                     analysis=None) -> None:
        """Show ``results`` (from ``path``); ``left``: why the analysis settings ``analysis``
        leave each spectrum out (:func:`~hxex.gui.results_tree.left_out`; worked out here when
        not given)."""
        self.results = results
        self.analysis_config = analysis
        source = None if results is None or results.settings is None else \
            results.settings.spectra_source
        for i, text in enumerate(view_words(has_fragments(source)).values()):
            self.compare.setItemText(i, text)
        set_choice_tips(self.compare, view_tips(has_fragments(source)))
        if results is not None and left is None:
            left = _left_out_or_none(results, analysis)
        self.left = left
        self.left_bar.hide()  # worked out as the results were read
        self._selection, self._table_for = (None, None), None  # the table: nothing selected
        self._trace_for, self._drawn, self._shown_traces = None, None, None
        self._drop_reader()  # the files of the results shown before
        self.results_path = path
        self.kind = None  # the tabs again: the elution tab follows the results
        self.table_view.clear()
        for panel in self.panels.values():
            panel.clear()
        self.tree_data = [] if results is None else peptides(results, left)
        self._fill_target_filters()
        self._fill_time_axis()
        self.review.show_spectra([], "Select a spectrum, a time point or a series.")
        self.path_label.setText("No results open." if results is None else str(path))
        self._fill()
        self._build_list()
        self._fill_overview_pages()
        self._show_tabs("start")  # the first look: the run's overview
        self._draw_overview()

    # --- the elution tab -----------------------------------------------------------------------

    def _traceable(self) -> bool:
        """Whether the results shown were read from scan files (an elution trace can be read)."""
        from ..io.retention_trace import readable

        settings = getattr(self.results, "settings", None)
        return readable(None if settings is None else settings.spectra_source)

    def _reader(self):
        """The trace reader of the results shown: their read record, with the data folder of
        the project they came from (relative run paths start there), else the folder that holds
        the results."""
        from ..io.retention_trace import TraceReader

        # one reader for the read record shown: a review decision gives new results with the
        # same record, and a second reader then read the same files at once (2026-11-09)
        source = self.results.settings.spectra_source
        if self._trace_reader is not None and self._trace_reader[0] is source:
            return self._trace_reader[1]
        self._drop_reader()
        data_root, thermo_dir = None, None
        path = getattr(self, "results_path", None)
        if path is not None and Path(path).suffix == ".toml":
            from ..config import load_project

            try:
                inp = load_project(path, analysis_only=True).input
            except Exception as exc:  # noqa: BLE001 (the project file changed: its folder then)
                log.info("retention traces: %s (%s)", path, exc)
                inp = None
            if inp is not None:
                data_root, thermo_dir = inp.data_root, inp.thermo_dir
            data_root = data_root or Path(path).parent
        elif path is not None:
            data_root = Path(path).parent
        reader = TraceReader(self.results.settings.spectra_source, data_root=data_root,
                             thermo_dir=thermo_dir)
        self._trace_reader = (source, reader)
        return reader

    def _drop_reader(self) -> None:
        """Close the trace reader's files, off the window's thread: a read still running holds
        them, and the window does not wait for it."""
        if self._trace_reader is None:
            return
        reader, self._trace_reader = self._trace_reader[1], None
        threading.Thread(target=reader.close, name="close trace files", daemon=True).start()

    def _compare_changed(self, _index: int) -> None:
        self._trace_for = None  # read again with the other comparison
        self._draw_retention()

    def _peptide_ions(self, reader, results, sid: str, data_id: int) -> list[tuple[str, int]]:
        """The spectra (id, series) of the selected spectrum's peptide's other ions in its run:
        its other charge states and, as a parent, its fragments (or, for a fragment, its
        parent and the parent's other fragments)."""
        from ..io.elution_windows import peptide_label

        where = reader.locate(sid, data_id)
        if where is None:
            return []
        peptide = peptide_label(reader.targets[reader.of_series[data_id]].entry)
        head = sid.split("/", 2)[:2]  # project:sample, file
        ids = np.asarray(results.fit.spectrum_id).astype(str)
        series = np.asarray(results.fit.data_id)
        out = []
        for s, d in zip(ids, series, strict=True):
            d = int(d)
            if s == sid or d == data_id or s.split("/", 2)[:2] != head:
                continue
            t_i = reader.of_series.get(d)
            if t_i is not None and peptide_label(reader.targets[t_i].entry) == peptide:
                out.append((s, d))
        return out

    def _draw_retention(self) -> None:
        """The retention tab, when it is in view: the selected spectrum's trace in its run
        (drawn first), then with what "Show elution traces for" chooses behind it (its
        replicates, or every file of its sample, or the peptide's other ions in this file;
        choice 1: nothing)."""

        keys = getattr(self, "_tab_keys", ())
        at = self.figures.currentIndex()
        if self.kind != "spectrum" or not (0 <= at < len(keys)) or keys[at] != "elution":
            return
        kind, payload = self._selection
        if kind != "spectrum" or self.results is None:
            return
        results, sid = self.results, payload[3].spectrum_id
        if self._trace_for is not None and self._trace_for[0] is results \
                and self._trace_for[1] == sid:  # the results by identity, not by value
            return
        self._trace_for = (results, sid)
        shown = self._shown_traces
        if shown is None or shown[0] is not results or shown[1] != sid:
            # another spectrum: the one before is forgotten, and the editor's work for it
            # dropped, so nothing draws it again while this one's trace is read (the same
            # spectrum read again, for another view, keeps the window being edited)
            self._shown_traces = None
            if self.window_editor is not None:
                self.window_editor.forget()
        panel = self.panel("elution")
        try:
            reader = self._reader()
        except (ValueError, KeyError, TypeError) as exc:
            panel.clear(f"No elution trace: {exc}")
            return
        data_id = int(np.asarray(results.fit.data_id)[payload[3].index])
        view = self.compare.currentData()
        fragments = has_fragments(reader)
        others_label = {"replicates": "its replicates",
                        "everywhere": "the sample's other files"}.get(view, "")
        groups: dict[int, list[str]] = {}  # the files compared, one spectrum each, by series
        if view in ("replicates", "everywhere"):
            from ..io.elution_windows import peptide_label

            target = reader.targets[reader.of_series[data_id]]
            ion = sid.split(":", 1)[-1].split("/")[2] if sid.count("/") >= 3 else ""
            for s, d in spread_files(reader, results, sid, data_id,
                                     peptide=peptide_label(target.entry), ion=ion, level=view):
                groups.setdefault(d, []).append(s)
        same = groups.pop(data_id, [])
        drift = {}
        binned = results.binned
        if binned is not None and binned.drift_start is not None:
            for s in (sid, *same, *(x for g in groups.values() for x in g)):
                try:
                    j = binned.find(s)
                except KeyError:
                    continue
                drift[s] = (float(binned.drift_start[j]), float(binned.drift_end[j]))
        title = f"{payload[0].sample} {payload[0].peptide_range} {payload[0].peptide} " \
                f"{payload[1].name}, {payload[2].label} rep {payload[3].rep}"  # which file

        def failed(message: str) -> None:
            self._trace_for = None  # tried again when next in view
            panel.clear(f"No elution trace: {message}")

        def show(traces, suffix: str = "") -> None:
            traces = dataclasses.replace(
                traces, others_label=others_label or traces.others_label,
                ions_label=ions_legend(fragments))
            self._shown_traces = (results, sid, data_id, traces, title + suffix)
            if self.window_editor is not None and self.window_editor.active:
                self.window_editor.update()
            else:
                self._draw_plain_trace()

        def others(first) -> None:
            if self.compare.currentData() == "ions":
                ions = self._peptide_ions(reader, results, sid, data_id)
                if not ions or not first.traces:
                    show(first, "; no other ion of this peptide in this run" if first.traces
                         else "")
                    return
                ranges = dict(drift)
                if binned is not None and binned.drift_start is not None:
                    for s, _d in ions:
                        try:
                            j = binned.find(s)
                        except KeyError:
                            continue
                        ranges[s] = (float(binned.drift_start[j]), float(binned.drift_end[j]))
                times = first.traces[0].times
                span = (float(np.min(times)), float(np.max(times)))
                show(first, "; reading this peptide's other ions in this run…")
                self._trace_task.submit(
                    lambda: dataclasses.replace(first, ions=reader.ion_traces(
                        sid, data_id, ions, span, drift_ranges=ranges)), show, failed)
                return
            if not (same or groups):
                show(first, f"; no other file: {others_label} hold no spectrum of it"
                     if first.traces and others_label else "")
                return
            show(first, f"; reading {others_label}…")

            def read_all():  # this series' files with this one; the sample's other series after
                got = reader.traces([sid, *same], data_id, drift_ranges=drift)
                extra = []
                for d, ids_ in groups.items():
                    more = reader.traces(ids_, d, drift_ranges=drift).traces
                    extra += [dataclasses.replace(t, selected=False) for t in more]
                return dataclasses.replace(got, traces=[*got.traces, *extra])

            self._trace_task.submit(read_all, show, failed)

        panel.clear("reading the elution trace…")
        self._trace_task.submit(lambda: reader.traces([sid], data_id, drift_ranges=drift),
                                others, failed)

    def _draw_plain_trace(self) -> None:
        """The elution tab without editing: the traces last read."""
        from ..report.retention import plot_retention

        if self._shown_traces is None:
            return
        *_, traces, title = self._shown_traces
        self.panel("elution").draw(lambda: plot_retention(traces, title=title))

    def _trace_context(self):
        """What the elution tab shows, for its window editing (None: nothing yet)."""
        from ..report.series import series_mz_range
        from .window_editor import TraceContext

        shown = self._shown_traces
        if shown is None or shown[0] is not self.results:
            return None
        results, sid, data_id, traces, title = shown
        try:
            mz_range = series_mz_range(results, data_id)
        except (ValueError, KeyError):
            mz_range = None
        return TraceContext(results, getattr(self, "results_path", None), sid, data_id, traces,
                            title, mz_range, self._reader(), compare=self.compare.currentData())

    # --- the data tab ----------------------------------------------------------------------------

    def _fill_table(self) -> None:
        """The table tab, when it is in view: the selection's rows of the report's tables, as
        ``hxex report`` writes them (made again when the selection or the results change)."""
        from ..report.tables import components_table, series_table, spectra_table, uptake_table

        if self.figures.currentWidget() is not self.table_view or self.results is None:
            return
        kind, payload = getattr(self, "_selection", (None, None))
        self.series_table.setVisible(kind not in (None, "spectrum", "point"))
        if kind is None:
            return
        key = (id(self.results), kind, id(payload))  # by identity: reset with new results
        if self._table_for == key:
            return
        self._table_for = key
        results, series = self.results, payload[1]
        if kind == "spectrum":
            table = spectra_table(results, rows=[payload[3].index])
            rows = [{"column": name, "value": _plain_value(values[0])}
                    for name, values in table.items()]
            self.table_view.show_rows(["column", "value"], rows,
                                      f"{payload[3].spectrum_id}_spectrum")
            self._data_count(1)
            self.table_view.set_note("spectra.csv: this spectrum's row")
        elif kind == "point":
            table = spectra_table(results, rows=[s.index for s in payload[2].spectra])
            self._show_columns(table, f"{payload[0].peptide_range}_z{series.charge}_"
                                      f"{payload[2].label}_spectra")
            self.table_view.set_note("spectra.csv: this time point's spectra")
        else:
            row = series_table(results)
            at = int(np.flatnonzero(np.asarray(row["data_id"]) == series.data_id)[0])
            about = ", ".join(f"{name} {_cell(row[name][at])}" for name in row
                              if name not in ("data_id", "sample", "peptide_range", "peptide",
                                              "charge") and _cell(row[name][at]))
            chosen = self.series_table.currentText()
            stem = f"{payload[0].sample}_{payload[0].peptide_range}_z{series.charge}"
            if chosen == "components.csv":
                spectra = np.flatnonzero(results.fit.data_id == series.data_id)
                self._show_columns(components_table(results, rows=spectra),
                                   f"{stem}_components")
                what = "this series' populations, one row per spectrum and population"
            elif chosen == "spectra.csv":
                spectra = np.flatnonzero(results.fit.data_id == series.data_id)
                self._show_columns(spectra_table(results, rows=spectra), f"{stem}_spectra")
                what = "this series' spectra"
            else:
                self._show_columns(uptake_table(results, data_ids=[series.data_id]),
                                   f"{stem}_uptake")
                what = "this series' rows"
            self.table_view.set_note(f"{chosen}: {what}. series.csv: {about}")

    def _series_table_changed(self, _name: str) -> None:
        self._table_for = None
        self._fill_table()

    def _show_columns(self, table: dict, name: str) -> None:
        columns = list(table)
        n = len(next(iter(table.values()))) if table else 0
        rows = [{c: _plain_value(table[c][i]) for c in columns} for i in range(n)]
        self.table_view.show_rows(columns, rows, name)
        self._data_count(n)

    def _table_row_chosen(self, row: dict) -> None:
        """A row of the table double-clicked: its spectrum, when it names one."""
        sid = row.get("spectrum_id")
        if sid and self.results is not None:
            i = int(np.flatnonzero(self.results.fit.spectrum_id == sid)[0])
            leaf = self._leaves.get(i)
            if leaf is not None:
                self.tree.setCurrentItem(leaf)

    # --- the spectra pages of a series ---------------------------------------------------------

    def _step_page(self, step: int) -> bool:
        """With a series' spectra in view on several pages: its next (``step`` 1) or previous
        (-1) page; False at either end (the step then goes to the next series)."""
        item = self.tree.currentItem()
        if self._spectra_job is None or item is None or item.data(0, KIND) not in (
                "series", "peptide") or self.figures.currentWidget() not in (
                self.panels.get("spectra"), self.panels.get("populations")):
            return False
        at = self.spectra_page.currentIndex() + step
        if not 0 <= at < self.spectra_page.count():
            return False
        self.spectra_page.setCurrentIndex(at)
        return True

    def _show_spectra_pages(self, results, data_id: int, config,
                            populations: bool = False) -> None:
        """The series' spectra pages: the list in the tab's bar (shown when there are several;
        the page kept when the same series is drawn again; the last one when stepping back
        into the series), and the chosen page drawn; with ``populations``, its populations page
        too, on the same pages."""
        from ..report.figures import series_spectra_pages

        pages = series_spectra_pages(results, data_id, config)
        same = self._spectra_job is not None and self._spectra_job[1] == data_id
        keep = self.spectra_page.currentIndex() if same else 0
        if not same and self._land_on_last_page:
            keep = len(pages) - 1
        self._land_on_last_page = False
        self._spectra_job = (results, data_id, config, pages)
        self._populations_paged = populations
        for box in (self.spectra_page, self.population_page):
            box.blockSignals(True)
            box.clear()
            for page in pages:
                box.addItem(f"page {page.number} of {page.of}")
            box.setCurrentIndex(min(max(keep, 0), len(pages) - 1))
            box.blockSignals(False)
        self.spectra_page.setVisible(len(pages) > 1)
        self.population_page.setVisible(len(pages) > 1 and populations)
        self._draw_spectra_page()

    def _draw_spectra_page(self) -> None:
        from ..report.figures import plot_spectra_page

        if self._spectra_job is None:
            return
        results, data_id, config, pages = self._spectra_job
        at = max(self.spectra_page.currentIndex(), 0)
        page = pages[at]
        self.panel("spectra").draw(
            lambda: plot_spectra_page(results, data_id, config, page))
        if self._populations_paged:  # the same page of the populations
            from ..report.series import plot_populations

            self.population_page.blockSignals(True)
            self.population_page.setCurrentIndex(at)
            self.population_page.blockSignals(False)
            self.panel("populations").draw(lambda: plot_populations(
                results, data_id, k=config.k, corrected=config.corrected, rows=page.rows,
                max_columns=config.page_columns, note=page.note, layout=page.layout))

    # --- the list of spectra -------------------------------------------------------------------

    def _build_list(self) -> None:
        """The list's rows, made in the background (seconds on a large run; the tree is ready
        meanwhile)."""
        results, left = self.results, self.left
        self._list_tables = {}
        if results is None:
            self.list.clear("No results open.")
            return
        self.list.clear("Building the list…")
        self._list_task.submit(lambda: list_rows(results, left),
                               lambda made: self._list_built(results, "spectra", made),
                               lambda message: self.list.clear(
                                   f"The list could not be made: {message}"))
        shown = self.list.tables.currentData()
        if shown != "spectra":  # the table chosen, made again for these results
            self._list_table(shown)

    def _list_built(self, results, table: str, made) -> None:
        if results is not self.results:  # other results opened meanwhile
            return
        self._list_tables[table] = (results, *made)
        if self.list.tables.currentData() == table:
            self.list.set_rows(*made, table=table)

    def _list_table(self, table: str, again: bool = False) -> None:
        """Another of the report's tables in the list: made once per results, in the
        background (seconds for components.csv and uptake.csv on a large run); ``again``: made
        anew, with the latest decisions."""
        results = self.results
        if results is None:
            return
        kept = self._list_tables.get(table)
        if kept is not None and not again:
            self.list.set_rows(kept[1], kept[2], table=table)
            if kept[0] is not results:  # made before the latest decisions
                self.list.mark_stale()
            return
        self.list.clear(f"Making {table}.csv…")
        left = self.left
        task = self._list_task if table == "spectra" else self._table_task
        task.submit(lambda: _table_rows(results, table, left),
                               lambda made: self._list_built(results, table, made),
                               lambda message: self.list.clear(
                                   f"{table}.csv could not be made: {message}"))

    def _list_chosen(self, row: dict) -> None:
        """A row chosen in the list: its spectrum selected in the tree (its figures, the
        review panel), or its series for a table of series (series.csv, uptake.csv)."""
        where = {"sample": str(row.get("sample")), "peptide_range": str(row.get("peptide_range")),
                 "charge": int(row.get("charge") or 0)}
        if row.get("spectrum_id"):
            where["spectrum_id"] = str(row.get("spectrum_id"))
        elif row.get("data_id") is not None:  # one series of several with the same name
            where["data_id"] = int(row["data_id"])
        self.select(where)

    # --- the overview -------------------------------------------------------------------------

    def _fill_overview_pages(self) -> None:
        """The overview's pages the results allow (controls and quality only when they have
        them), the page chosen before kept."""
        from ..report.overview import overview_kinds

        keep = self.overview_page.currentData()
        kinds = overview_kinds(self.results) if self.results is not None else []
        self.overview_page.blockSignals(True)
        self.overview_page.clear()
        for kind in kinds:
            self.overview_page.addItem(OVERVIEW_PAGES.get(kind, kind), kind)
        index = self.overview_page.findData(keep)
        self.overview_page.setCurrentIndex(max(index, 0))
        self.overview_page.blockSignals(False)

    def _draw_overview(self) -> None:
        """The overview page chosen, when its tab is in view: drawn once per results (a review
        decision makes new results, so the K counts follow) and kept."""
        from ..report.overview import (
            plot_controls_overview,
            plot_k_selection,
            plot_quality_overview,
        )

        results = self.results
        if results is None or self.figures.currentWidget() is not self.panels.get("overview"):
            return
        page = self.overview_page.currentData()
        if page is None:
            return
        if self._overview_of is not results:
            self._overview_of, self._overview_figures = results, {}
        panel = self.panel("overview")
        panel.save_name = f"overview_{page}"
        kept = self._overview_figures.get(page)
        if kept is not None:
            if panel.view.figure is not kept:
                panel.show_figure(kept)
            return
        draw = {"k_selection": plot_k_selection, "controls": plot_controls_overview,
                "quality": plot_quality_overview}[page]
        figures = self._overview_figures

        def work():
            fig = draw(results)
            figures[page] = fig  # kept for these results (a dict of their own)
            return fig

        panel.draw(work)

    def _fill_time_axis(self) -> None:
        """The time axis list: the project's setting (by name and value) first."""
        from ..report.figures import PlotConfig

        keep = self.time_axis.currentData() if self.time_axis.count() else None
        config = (self.results.figures if self.results is not None else None) or PlotConfig()
        own = "by time point" if config.x == "index" else config.scale
        self.time_axis.blockSignals(True)
        self.time_axis.clear()
        self.time_axis.addItem(f"[figures] scale: {own}", None)
        for key, text in TIME_AXES.items():
            self.time_axis.addItem(text, key)
        index = self.time_axis.findData(keep) if keep is not None else 0
        self.time_axis.setCurrentIndex(max(index, 0))
        self.time_axis.blockSignals(False)

    def set_time_axis(self, key: str | None) -> None:
        index = self.time_axis.findData(key) if key else 0
        self.time_axis.setCurrentIndex(max(index, 0))

    def _plot_config(self):
        """The project's figure settings, with the time axis chosen in the window."""
        from ..report.figures import PlotConfig

        config = self.results.figures or PlotConfig()
        key = self.time_axis.currentData()
        if key == "index":
            return dataclasses.replace(config, x="index")
        if key in ("log", "linear"):
            x = "index" if self.results.time_scale == "order" else "time"
            return dataclasses.replace(config, x=x, scale=key)
        return config

    def _redraw(self, *_) -> None:
        self._selected(self.tree.currentItem())

    def _fill_target_filters(self) -> None:
        """The MS level and ion type lists, from the series of the results (hidden, with the
        tree's ion, MS level and condition columns, when no series has them)."""
        every = [s for p in self.tree_data for s in p.series]
        levels = sorted({s.ms_level for s in every if s.ms_level})
        ions = sorted({s.ion_type for s in every if s.ms_level})
        targeted = bool(levels)
        for box, items in ((self.ms_filter, [("every MS level", None)]
                            + [(f"MS{m}", m) for m in levels]),
                           (self.ion_filter, [("every ion", None)]
                            + [(i or "peptides", i) for i in ions])):
            box.blockSignals(True)
            box.clear()
            for text, value in items:
                box.addItem(text, value)
            box.blockSignals(False)
            box.setVisible(targeted)
        for name in TARGET_COLUMNS:
            self.tree.setColumnHidden(COLUMN[name], not targeted)
        self._fill_pool_filters()

    def _fill_pool_filters(self) -> None:
        """The identity level and class lists, from the pooled peptides of the results
        (hidden, with the tree's identity column, when no peptide has an identity)."""
        levels = {p.identity_level for p in self.tree_data if p.identity_level}
        classes = sorted({c for p in self.tree_data for c in p.classes.split(";") if c},
                         key=natural_key)
        pooled = bool(levels)
        for box, items, meanings in (
                (self.level_filter, [("every identity", None)]
                 + [(f"{x} level", x) for x in IDENTITY_LEVELS if x in levels], IDENTITY_LEVELS),
                (self.class_filter, [("every class", None)] + [(c, c) for c in classes], {})):
            box.blockSignals(True)
            box.clear()
            for i, (text, value) in enumerate(items):
                box.addItem(text, value)
                if value in meanings:
                    box.setItemData(i, tip(meanings[value]), Qt.ItemDataRole.ToolTipRole)
            box.blockSignals(False)
        self.level_filter.setVisible(pooled)
        self.class_filter.setVisible(pooled and bool(classes))
        self.tree.setColumnHidden(COLUMN["identity"], not pooled)

    def _shown(self, pep: Peptide, s) -> bool:
        """Whether the tree shows the series ``s`` of ``pep`` under the filters and search."""
        return shown(pep, s, self.filter.currentData() or "all", self.search.text(),
                     self.ms_filter.currentData(), self.ion_filter.currentData(),
                     self.level_filter.currentData(), self.class_filter.currentData())

    def _fill(self, *_) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        self._leaves, self._series_items = {}, {}
        n_series = 0
        for pep in self.tree_data:
            series = [s for s in pep.series if self._shown(pep, s)]
            if not series:
                continue
            counts = {n: sum(getattr(s, n) for s in series)
                      for n in ("n_spectra", "n_flagged", "n_rejected", "n_left_out", "n_multi")}
            top = QTreeWidgetItem([f"{pep.peptide_range} {pep.peptide}", pep.sample,
                                   pep.identity, "", "", "", "", *_counts(counts), "", "", ""])
            if pep.identity_level:
                top.setToolTip(COLUMN["identity"], tip(
                    f"{pep.identity_level} level; candidates: "
                    + (pep.candidates.replace(";", ", ") or "none")))
            top.setData(0, KIND, "peptide")
            top.setData(0, PAYLOAD, pep)
            top.setToolTip(0, tip(f"{pep.sample} {pep.peptide_range} {pep.peptide}"))
            for s, name in zip(series, _series_names(series), strict=True):
                child = QTreeWidgetItem([name, "", "", s.fragment, str(s.ms_level or ""),
                                         s.condition, s.parent, *_counts(dataclasses.asdict(s)),
                                         "", "", ""])
                child.setData(0, KIND, "series")
                child.setData(0, PAYLOAD, (pep, s))
                child.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
                top.addChild(child)
                self._series_items[s.data_id] = child
                n_series += 1
            self.tree.addTopLevelItem(top)
        self.tree.blockSignals(False)
        for c in range(1, len(COLUMNS)):
            self.tree.resizeColumnToContents(c)
        # the series rows are folded, so the sizing above did not see their text: the fragment
        # columns and the names take the widest of every series
        metrics = self.tree.fontMetrics()
        every = [s for p in self.tree_data for s in p.series]
        pad = 3 * metrics.horizontalAdvance("M")
        for name, value in (("ion", "fragment"), ("condition", "condition"),
                            ("parent", "parent")):
            widest = max((metrics.horizontalAdvance(getattr(s, value)) for s in every), default=0)
            self.tree.setColumnWidth(COLUMN[name], max(self.tree.columnWidth(COLUMN[name]),
                                                       widest + pad))
        names = max((metrics.horizontalAdvance(n) for p in self.tree_data
                     for n in _series_names(p.series)), default=0)
        self.tree.setColumnWidth(0, max(self.tree.columnWidth(0), 260,
                                        min(names + 4 * pad, 480)))
        total = sum(len(p.series) for p in self.tree_data)
        self.count.setText(f"{n_series} of {total} series" if self.results is not None else "")

    def _expanded(self, item: QTreeWidgetItem) -> None:
        """A series opened: its time points and their spectra, made now."""
        if item.data(0, KIND) != "series" or item.childCount():
            return
        pep, series = item.data(0, PAYLOAD)
        for point in points(self.results, series.data_id, self.left):
            child = QTreeWidgetItem([""] * len(COLUMNS))
            child.setText(0, point.label)
            child.setText(COLUMN["spectra"], str(len(point.spectra)))
            child.setText(COLUMN["K"], point.ks)
            child.setData(0, KIND, "point")
            child.setData(0, PAYLOAD, (pep, series, point))
            for spectrum in point.spectra:
                leaf = QTreeWidgetItem([f"rep {spectrum.rep}"] + [""] * (len(COLUMNS) - 1))
                leaf.setText(COLUMN["flagged"], "yes" if spectrum.flags else "")
                leaf.setText(COLUMN["K"], str(spectrum.K))
                leaf.setText(COLUMN["flags"], spectrum.flags.replace(";", ", "))
                leaf.setToolTip(COLUMN["flags"], flag_tip(spectrum.flags))
                leaf.setData(0, KIND, "spectrum")
                leaf.setData(0, PAYLOAD, (pep, series, point, spectrum))
                leaf.setToolTip(0, tip(spectrum.spectrum_id))
                child.addChild(leaf)
                self._leaves[spectrum.index] = leaf
                self._show_status(leaf, spectrum.status)
                self._show_left_out(leaf, spectrum.left_out)
            self._sum_children(child)
            item.addChild(child)

    # --- exporting -------------------------------------------------------------------------

    def _export_button(self) -> QPushButton:
        """**Export data** beside **Save figure** (owner, 2026-11-12: the right-click in the tree
        was easy to miss): a menu of what the selected tree row exports, made when it opens."""
        from PySide6.QtWidgets import QMenu

        button = QPushButton("Export data")
        button.setToolTip(tip("the selection's data: an HX-Express spreadsheet of the series, "
                              "or the CSV files behind its figures"))
        menu = QMenu(button)
        menu.setToolTipsVisible(True)
        menu.aboutToShow.connect(lambda m=menu: self.fill_export_menu(m))
        button.setMenu(menu)
        return button

    def export_targets(self, item: QTreeWidgetItem) -> dict[str, list]:
        """What **Export data** exports for the tree row ``item`` (docs/PLAN.md Next steps 32):
        for HX-Express, the series (a peptide: each of its charges shown); for the plot data, the
        series, the time point's spectra, or the spectrum."""
        from .export_dialog import HXEXPRESS, PLOT_DATA, Target

        kind, payload = item.data(0, KIND), item.data(0, PAYLOAD)
        if kind == "peptide":
            series = [s for s in payload.series if self._shown(payload, s)]
            whole = [Target(s.data_id) for s in series]
            return {HXEXPRESS: whole, PLOT_DATA: whole}
        data_id = payload[1].data_id
        if kind == "series":
            return {HXEXPRESS: [Target(data_id)], PLOT_DATA: [Target(data_id)]}
        point = payload[2]
        if kind == "point":
            some = Target(data_id, [s.spectrum_id for s in point.spectra], point.label)
        else:
            spectrum = payload[3]
            some = Target(data_id, [spectrum.spectrum_id], f"{point.label} rep {spectrum.rep}")
        return {HXEXPRESS: [Target(data_id)], PLOT_DATA: [some]}

    def fill_export_menu(self, menu) -> dict:
        """Fill **Export data**'s ``menu`` for the selected tree row; returns
        ``{action: (kind, targets)}`` (empty, with one greyed line, when nothing is selected)."""
        from .export_dialog import HXEXPRESS, PLOT_DATA

        menu.clear()
        item = self.tree.currentItem()
        if item is None or self.results is None or item.data(0, KIND) not in DEPTH:
            menu.addAction("Select a peptide, charge, time point, or spectrum").setEnabled(False)
            return {}
        kind = item.data(0, KIND)
        targets = self.export_targets(item)
        n = len(targets[HXEXPRESS])
        hx = menu.addAction(
            f"Export its {n} charge states for HX-Express…" if kind == "peptide" and n > 1
            else "Export for HX-Express…" if kind in ("peptide", "series")
            else "Export the series for HX-Express…")
        hx.setToolTip(tip("one spreadsheet per peptide and charge, with every time point and "
                          "replicate, as HX-Express reads it"))
        what = {"point": "this time point's ", "spectrum": "this spectrum's "}.get(kind, "")
        plot = menu.addAction(f"Export {what}plot data…")
        plot.setToolTip(tip("CSV files with what the figures draw: picked values, fits, "
                            "populations, raw spectra, and the table rows"))
        actions = {hx: (HXEXPRESS, targets[HXEXPRESS]), plot: (PLOT_DATA, targets[PLOT_DATA])}
        for action, (what, chosen) in actions.items():
            action.triggered.connect(lambda _c=False, w=what, t=chosen: self.open_export(w, t))
        return actions

    def default_export_folder(self) -> Path:
        """The last export's folder, else ``hxex_exports`` beside the results (not inside the
        bundle, whose folders later runs replace)."""
        if self.export_folder is not None:
            return self.export_folder
        path = getattr(self, "results_path", None)
        return (Path(path).parent if path is not None else Path.cwd()) / "hxex_exports"

    def open_export(self, kind: str, targets: list):
        """The export dialog for ``targets`` (returned once closed, for the tests)."""
        from .export_dialog import ExportDialog

        dialog = ExportDialog(self.results, kind, targets, self.default_export_folder(), self)
        if dialog.exec() and dialog.written:
            self.export_folder = Path(dialog.folder.text().strip()).expanduser()
        return dialog

    # --- selection -------------------------------------------------------------------------

    def _selected(self, item: QTreeWidgetItem | None, again: bool = True) -> None:
        """Draw what ``item`` shows. Moving to an item that shows the same figures (a
        peptide and its first charge) draws nothing again unless ``again`` (settings or
        decisions changed)."""
        from ..report.figures import series_figures, series_kinds
        from ..report.overview import plot_k_comparison, plot_k_comparisons
        from ..report.series import plot_series_spectra, series_mz_range
        from ..report.spectrum import plot_spectrum

        if item is None or self.results is None:
            return
        kind = item.data(0, KIND)
        payload = item.data(0, PAYLOAD)
        if kind == "peptide":  # its first charge's pages
            pep = payload
            series = next((s for s in pep.series if self._shown(pep, s)), pep.series[0])
            kind, payload = "series", (pep, series)
        if kind == "point" and len(payload[2].spectra) == 1:
            # one replicate: its own view, so stepping into it draws nothing new (owner)
            kind, payload = "spectrum", (*payload, payload[2].spectra[0])
        results = self.results
        series = payload[1]
        data_id = series.data_id
        shown = (payload[3].spectrum_id if kind == "spectrum"
                 else payload[2].order if kind == "point" else None)
        drawn = (kind, data_id, shown)
        if not again and drawn == self._drawn:  # the item moved to, not what it shows
            self.selection_changed.emit()
            if self.window is not None:
                self.window.changed()
            return
        self._drawn = drawn
        config = self._plot_config()
        self._show_tabs(kind)
        self._show_review(kind, payload)
        # one m/z axis for the series at every level: its pages, a time point, a spectrum
        mz_range = series_mz_range(results, data_id) if kind != "series" else None

        def uptake():
            one = dataclasses.replace(config, kinds=("uptake",))
            return series_figures(results, data_id, one)["uptake"]

        if kind == "series":
            drawn = series_kinds(results, data_id,
                                 dataclasses.replace(config, kinds=SERIES_PAGES))
            for name in SERIES_PAGES:
                panel = self.panel(name)
                if name not in drawn:
                    panel.clear("one population in every spectrum: no population map")
                    continue
                if name == "spectra":  # on pages: the one chosen (the first for a new series),
                    # the populations on the same pages
                    self._show_spectra_pages(results, data_id, config,
                                             populations="populations" in drawn)
                    continue
                if name == "populations":  # drawn with its spectra's page
                    continue
                one = dataclasses.replace(config, kinds=(name,))
                panel.draw(lambda one=one, name=name: series_figures(results, data_id,
                                                                     one)[name])
        elif kind == "point":
            self.spectra_page.hide()
            self.population_page.hide()
            self._spectra_job = None
            point = payload[2]
            rows = [s.index for s in point.spectra]
            # a near square of panels (2 x 2 for four replicates): one row of four was a wide,
            # short page, tiny when fitted to the window (owner, 2026-11-03)
            columns = max(1, math.ceil(math.sqrt(len(rows))))
            self.panel("spectra").draw(lambda: plot_series_spectra(
                results, data_id, rows=rows, columns=columns, show=config.show, k=config.k,
                corrected=config.corrected))
            point = payload[2]
            ids = [s.spectrum_id for s in point.spectra]
            labels = [f"rep {s.rep}" for s in point.spectra]
            title = f"{payload[0].sample} {payload[0].peptide_range} {payload[0].peptide} " \
                    f"{series.name}, {point.label}"
            self.panel("fits at each K").draw(lambda: plot_k_comparisons(
                ids, results, labels=labels, title=title, mz_range=mz_range))
            self.panel("uptake").draw(uptake)
        else:
            spectrum = payload[3]
            sid = spectrum.spectrum_id
            self.panel("spectrum").draw(lambda: plot_spectrum(
                sid, results, show=config.show, k=config.k, corrected=config.corrected,
                mz_range=mz_range))
            self.panel("fits at each K").draw(lambda: plot_k_comparison(sid, results,
                                                                        mz_range=mz_range))
            self.panel("uptake").draw(uptake)
        self._draw_overview()  # in view: kept, or drawn again for new results (a decision)
        self._selection = (kind, payload)
        self._data_count(None)  # until the tab is filled for this selection
        self._fill_table()
        self._draw_retention()
        self.selection_changed.emit()
        if self.window is not None:
            self.window.changed()

    # --- stepping through the spectra ---------------------------------------------------------

    def step_spectrum(self, step: int, level: str | None = None) -> bool:
        """The review panel's arrows: :meth:`_step`, through a series' spectra pages first when
        stepping by series with them in view."""
        try:
            return self._step(step, level)
        finally:
            self._land_on_last_page = False  # only for the step that set it

    def _step(self, step: int, level: str | None = None) -> bool:
        """Select the next (``step`` 1) or previous (-1) row of ``level`` (``'spectrum'``,
        ``'point'``, ``'series'`` or ``'peptide'``; default: the review panel's choice) in the
        tree's order, opening series on the way. From a row below ``level``, the next after the
        row's own ``level`` row; from a row above it, the first (or last) one inside it. False
        at either end."""
        if self.left_tabs.currentWidget() is self.list:  # in the list's order, as sorted
            return self.list.step(step)
        level = level or self.review.level.currentData() or "spectrum"
        if level in ("series", "peptide") and self._step_page(step):
            return True  # a series' next (or previous) page before the next series (owner)
        self._land_on_last_page = step < 0 and level in ("series", "peptide")  # its last page
        depth = DEPTH[level]
        item = self.tree.currentItem()
        if item is None:
            if not self.tree.topLevelItemCount():
                return False
            start = self.tree.topLevelItem(0 if step > 0 else self.tree.topLevelItemCount() - 1)
            target = self._edge_at(start, step, depth)
        else:
            while DEPTH[item.data(0, KIND)] > depth:  # a spectrum, stepping by time point, say
                item = item.parent()
            target = (self._after_at(item, step, depth) if DEPTH[item.data(0, KIND)] == depth
                      else self._edge_at(item, step, depth))
        if target is None:
            return False
        parent = target.parent()
        while parent is not None:
            parent.setExpanded(True)
            parent = parent.parent()
        self.tree.setCurrentItem(target)
        self.tree.scrollToItem(target)
        return True

    def _children(self, item: QTreeWidgetItem) -> list[QTreeWidgetItem]:
        if item.data(0, KIND) == "series" and not item.childCount():
            self._expanded(item)  # its time points and spectra, made now
        return [item.child(i) for i in range(item.childCount())]

    def _edge_at(self, item: QTreeWidgetItem, step: int, depth: int) -> QTreeWidgetItem | None:
        """The first (``step`` > 0) or last row of ``depth`` under ``item`` (itself when of it)."""
        if DEPTH[item.data(0, KIND)] == depth:
            return item
        children = self._children(item)
        for child in (children if step > 0 else reversed(children)):
            found = self._edge_at(child, step, depth)
            if found is not None:
                return found
        return None

    def _after_at(self, item: QTreeWidgetItem, step: int, depth: int) -> QTreeWidgetItem | None:
        """The row of ``depth`` after (or before) ``item``: its sibling, else the edge row of
        the next sibling of the nearest level that has one."""
        while item is not None:
            parent = item.parent()
            siblings = (self._children(parent) if parent is not None else
                        [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())])
            at = siblings.index(item) + step
            while 0 <= at < len(siblings):
                found = self._edge_at(siblings[at], step, depth)
                if found is not None:
                    return found
                at += step
            item = parent
        return None

    def _show_review(self, kind: str, payload) -> None:
        pep, series = payload[0], payload[1]
        name = f"{pep.sample} {pep.peptide_range} {pep.peptide} {series.name}"
        if kind == "series":
            rows = list(np.flatnonzero(self.results.fit.data_id == series.data_id))
        elif kind == "point":
            rows = [s.index for s in payload[2].spectra]
            name += f", {payload[2].label}"
        else:
            rows = [payload[3].index]
            name += f", {payload[2].label}, rep {payload[3].rep}"
        self.review.show_spectra(rows, name)

    def _reviewed(self, changed: list) -> None:
        """Decisions were made: the rows of the spectra they touch, their time points, series
        and peptides show them, and the figures are drawn again."""
        from .review_panel import counts

        results = self.results
        status = np.asarray(results.review_status).astype(str)
        ks = results.effective_K
        series_ids = {int(results.fit.data_id[i]) for i in changed}
        k = COLUMN["K"]
        for i in changed:
            leaf = self._leaves.get(int(i))
            if leaf is not None:
                leaf.setText(k, str(int(ks[i])))
                self._show_status(leaf, status[i])
                point = leaf.parent()
                point.setText(k, " ".join(point.child(j).text(k)
                                          for j in range(point.childCount())))
                self._sum_children(point)
        for data_id in series_ids:
            item = self._series_items.get(data_id)
            if item is None:
                continue
            rejected, multi = counts(results, data_id)
            item.setText(COLUMN["rejected"], str(rejected or ""))
            item.setText(COLUMN["K > 1"], str(multi or ""))
            self._sum_children(item.parent(), ("rejected", "K > 1"))
        self._redraw()
        kept = self._list_tables.get("spectra")
        if kept is not None:  # the decided spectra's rows, made again (K, populations, review)
            fresh = list_rows(results, self.left, rows=sorted(int(i) for i in changed))[1]
            by_index = {r["_index"]: r for r in fresh}
            for row in kept[2]:
                row.update(by_index.get(row["_index"], {}))
            self._list_tables["spectra"] = (results, kept[1], kept[2])
            if self.list.table_name == "spectra":
                self.list.update_rows(fresh)
        if self.list.table_name != "spectra" and self.list.rows:
            self.list.mark_stale()  # the other tables: made again only when asked
        # a decision can change what the analysis leaves out (an accepted spectrum is kept; a
        # rejected one changes its replicates' median): said, and worked out again when asked
        self._left_stale("your latest review decisions")

    def analysis_changed(self, config) -> None:
        """The project's [analysis] settings changed: the "left out" column is from the earlier
        ones until Update."""
        self.analysis_config = config
        if self.results is not None:
            self._left_stale("the [analysis] settings changed")

    def _left_stale(self, why: str) -> None:
        self.left_note.setText(f"The 'left out' column is from before {why}." if why.startswith(
            "your") else f"The 'left out' column is out of date: {why}.")
        self.left_update.setEnabled(True)
        self.left_bar.show()

    def _left_tab_changed(self, _index: int) -> None:
        """The status line's count and the review panel's stepping follow the tab shown."""
        on_list = self.left_tabs.currentWidget() is self.list
        self.count.setVisible(not on_list)
        self.list.count.setVisible(on_list)
        self.review.set_list_mode(on_list)

    def update_left_out(self) -> None:
        """Work out the "left out" column again (in the background), as asked."""
        results, config = self.results, self.analysis_config
        if results is None:
            return
        self.left_note.setText("Working out the 'left out' column…")
        self.left_update.setEnabled(False)
        self._left_task.submit(lambda: _left_out_or_none(results, config),
                               lambda left: self._left_out_known(results, left),
                               self._left_failed)

    def _left_failed(self, message: str) -> None:
        log.warning("what the analysis leaves out is not known: %s", message)
        self.left_note.setText(f"The 'left out' column could not be worked out: {message}")
        self.left_update.setEnabled(True)

    def _left_out_known(self, results, left: list[str] | None) -> None:
        """What the analysis leaves out, worked out again after a decision: shown on every row
        made so far."""
        if results is not self.results:  # newer decisions since: still out of date
            self._left_stale("your latest review decisions")
            return
        self.left_bar.hide()
        self.left = left
        if self.list.rows:
            self.list.update_rows([{**row, "left_out": left[row["_index"]] if left else ""}
                                   for row in self.list.rows])
        n_left = left_out_counts(results, left)
        for pep in self.tree_data:
            for series in pep.series:
                series.n_left_out = n_left.get(series.data_id, 0)
        for i, leaf in self._leaves.items():
            spectrum = leaf.data(0, PAYLOAD)[3]
            spectrum.left_out = left[i] if left is not None else ""
            self._show_left_out(leaf, spectrum.left_out)
        points = {id(leaf.parent()): leaf.parent() for leaf in self._leaves.values()}
        for point in points.values():
            self._sum_children(point, ("left out",))
        for data_id, item in self._series_items.items():
            item.setText(COLUMN["left out"], str(n_left.get(data_id, 0) or ""))
        for t in range(self.tree.topLevelItemCount()):
            self._sum_children(self.tree.topLevelItem(t), ("left out",))

    @staticmethod
    def _show_status(leaf: QTreeWidgetItem, status: str) -> None:
        status = "" if status in ("", "unreviewed", "None") else status
        leaf.setText(COLUMN["review"], status)
        leaf.setText(COLUMN["rejected"], "yes" if status == "rejected" else "")

    @staticmethod
    def _show_left_out(leaf: QTreeWidgetItem, why: str) -> None:
        leaf.setText(COLUMN["left out"], why)
        leaf.setToolTip(COLUMN["left out"], flag_tip(why) or tip(why))

    @staticmethod
    def _sum_children(item: QTreeWidgetItem,
                      names: tuple[str, ...] = ("flagged", "rejected", "left out")) -> None:
        """``item``'s counts in the columns ``names``: its rows' counts added up (a
        replicate's row counts as one when it says anything; 'kept at K = 1' does not)."""
        for name in names:
            col = COLUMN[name]
            total = 0
            for j in range(item.childCount()):
                text = item.child(j).text(col)
                total += (int(text) if text.isdigit() else
                          int(bool(text) and not text.startswith("kept")))
            item.setText(col, str(total or ""))

    def selection(self) -> dict:
        """What is selected, by what outlasts a rerun: sample, residues and charge (and for a
        fragment its ion, condition and MS level), and the time point's label or the spectrum's
        id."""
        item = self.tree.currentItem()
        if item is None:
            return {}
        kind, payload = item.data(0, KIND), item.data(0, PAYLOAD)
        if kind == "peptide":
            return {"sample": payload.sample, "peptide_range": payload.peptide_range}
        pep, series = payload[0], payload[1]
        out = {"sample": pep.sample, "peptide_range": pep.peptide_range,
               "charge": series.charge}
        if series.fragment:
            out.update(fragment=series.fragment, condition=series.condition,
                       ms_level=series.ms_level)
        if kind == "point":
            out["point"] = payload[2].label
        elif kind == "spectrum":
            out["spectrum_id"] = payload[3].spectrum_id
        return out

    def view_settings(self) -> dict:
        """How the figures are shown, for the session (only what differs from the project)."""
        key = self.time_axis.currentData()
        return {"time_axis": key} if key else {}

    def select_peptide(self, sample: str, start: int, end: int, peptide: str = "") -> bool:
        """Select the peptide ``start``-``end`` of ``sample`` (its sequence ``peptide`` first,
        when two share the residues, a modified one say); False when the tree lacks it."""
        found = [p for p in self.tree_data
                 if p.sample == sample and _residues(p.peptide_range) == (start, end)]
        if not found:
            return False
        pep = next((p for p in found if p.peptide == peptide), found[0])
        return self.select({"sample": pep.sample, "peptide_range": pep.peptide_range})

    def select(self, selection: dict) -> bool:
        """Select what ``selection`` names (as :meth:`selection` gives it); False when the
        results do not have it (the filter and search are cleared to find it)."""
        if not selection:
            return False
        boxes = (self.filter, self.ms_filter, self.ion_filter, self.level_filter,
                 self.class_filter)
        if self.search.text() or any(box.currentIndex() > 0 for box in boxes):
            for box in boxes:
                box.setCurrentIndex(0 if box.count() else -1)
            self.search.clear()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            pep = top.data(0, PAYLOAD)
            if (pep.sample, pep.peptide_range) != (selection.get("sample"),
                                                   selection.get("peptide_range")):
                continue
            if "charge" not in selection:
                self.tree.setCurrentItem(top)
                return True
            for j in range(top.childCount()):
                child = top.child(j)
                found = child.data(0, PAYLOAD)[1]
                if found.charge != selection["charge"] or (
                        "data_id" in selection and found.data_id != selection["data_id"]) or any(
                        name in selection and getattr(found, name) != selection[name]
                        for name in ("fragment", "condition", "ms_level")):
                    continue
                if "point" not in selection and "spectrum_id" not in selection:
                    self.tree.setCurrentItem(child)
                    return True
                top.setExpanded(True)
                child.setExpanded(True)
                for k in range(child.childCount()):
                    point = child.child(k)
                    if point.data(0, PAYLOAD)[2].label == selection.get("point"):
                        self.tree.setCurrentItem(point)
                        return True
                    for m in range(point.childCount()):
                        leaf = point.child(m)
                        if leaf.data(0, PAYLOAD)[3].spectrum_id == selection.get("spectrum_id"):
                            point.setExpanded(True)
                            self.tree.setCurrentItem(leaf)
                            return True
        return False


def _residues(peptide_range: str) -> tuple[int, int] | None:
    """``'0001-0013'`` as ``(1, 13)`` (None when it does not read so)."""
    try:
        first, last = peptide_range.split("-")[:2]
        return int(first), int(last)
    except ValueError:
        return None


def _table_rows(results, table: str, left) -> tuple[list[str], list[dict]]:
    """A report table's columns and rows for the list (``spectra``: :func:`list_rows`)."""
    from ..report.tables import components_table, series_table, uptake_table
    from .spectra_list import _plain

    if table == "spectra":
        return list_rows(results, left)
    made = {"series": series_table, "components": components_table,
            "uptake": uptake_table}[table](results)
    columns = list(made)
    n = len(next(iter(made.values()))) if made else 0
    return columns, [{c: _plain(made[c][i]) for c in columns} for i in range(n)]


def _plain_value(value):
    """A table value as Python's own (numbers, text, booleans)."""
    from .spectra_list import _plain

    return _plain(value)


def _cell(value) -> str:
    return cell_text(_plain_value(value))


def _series_names(series: list) -> list[str]:
    """The tree's names of a peptide's series (:attr:`Series.name`); one fragment from two
    parents adds its parent, and two series still of one name (a peptide read from two lists,
    say) the series number, as spectra.csv's data_id."""
    names = [s.name for s in series]
    names = [n + f"  from {s.parent}" if s.parent and names.count(n) > 1 else n
             for s, n in zip(series, names, strict=True)]
    return [n + (f" (series {s.data_id})" if names.count(n) > 1 else "")
            for s, n in zip(series, names, strict=True)]


def _counts(c: dict) -> list[str]:
    return [str(c["n_spectra"]), str(c["n_flagged"] or ""), str(c["n_rejected"] or ""),
            str(c["n_left_out"] or ""), str(c["n_multi"] or "")]


def _left_out_or_none(results, config) -> list[str] | None:
    """:func:`~hxex.gui.results_tree.left_out`, or None (said in the log) when the results do
    not allow it (no picked table, say)."""
    try:
        return left_out(results, config)
    except Exception as exc:  # noqa: BLE001 - the tree is still useful without the column
        log.warning("what the analysis leaves out is not known: %s", exc)
        return None


__all__ = ["ResultsWorkspace"]
