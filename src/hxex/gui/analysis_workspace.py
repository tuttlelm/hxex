# SPDX-License-Identifier: GPL-3.0-or-later
"""The Analysis workspace (docs/PLAN.md, "GUI", "Analysis"): exactly what ``hxex analyse``
writes with the project's settings, shown rather than written.

On the left, every kind of figure ``hxex analyse`` draws (:mod:`hxex.report.catalogue`), one
row per kind under the catalogue's groups: drawn, left to the results, or off, with the reason
and a button to the setting that rules it out. On the right, a figure of the kind selected,
its sample, time, form, and page chosen in the lists above it (:mod:`.figure_choices`; owner,
2026-11-22: a row per figure made the list long and hard to navigate), drawn from ``hxex
analyse``'s own list of figures (:func:`hxex.report.analysis_outputs`), so the window and the
files cannot differ, and saved under the name the command gives it.

The analysis is computed in the background when asked (Compute, beside the reference list: half
a minute for a large run of peptides, minutes with fragment ladder fits), and kept: a change of
the results (a new run, a review decision), of the review table (in the Tables view or another
program) or of ``[analysis]`` marks it out of date, keeping the figures, and Recompute computes
it again, reading the review table again first when it changed. With several references
(``[analysis] reference`` a list, or ``"all"``), a list chooses which one is shown; each is
computed when asked and kept, as ``hxex analyse`` writes a folder for each.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .analysis_tree import ONE_SAMPLE_KINDS, title_at
from .data_table import DataPanel, spectra_menu
from .figure_choices import ChoicesRow
from .figure_view import HINT, SPECTRA_HINT, FigurePanel
from .review import table_stamp
from .summary_panel import SummaryPanel
from .tasks import Latest, drawing_pool
from .tips import ERROR_COLOUR, tip

log = logging.getLogger(__name__)

KIND, NODE = Qt.ItemDataRole.UserRole, Qt.ItemDataRole.UserRole + 1
# before Compute: nothing beside a kind the settings draw (kinds they rule out are under Not
# drawn), "data permitting" beside one the results decide (owner, 2026-11-23)
STATE_WORDS = {"drawn": "", "depends": "data permitting", "off": "off"}


# the group list's entry for the figure across the groups (results_analysis_groups); never a
# group's name (a TOML key cannot hold this character unescaped, and the editor refuses it)
ACROSS = "\x1facross the groups"


class _UnderParent(QStyledItemDelegate):
    """A row with nothing under it (a kind of figure) starts its name under its parent's name,
    in the room its own ▸ would take, at every level; a row with rows under it keeps its ▸
    there, so each level steps in by one ▸ (owner, 2026-11-23)."""

    def paint(self, painter, option, index):
        if index.parent().isValid() and not index.model().hasChildren(index):
            option = QStyleOptionViewItem(option)
            option.rect = option.rect.adjusted(-self.parent().indentation(), 0, 0, 0)
        super().paint(painter, option, index)


class _StayingMenu(QMenu):
    """A list of check boxes that stays open while they are ticked (owner, 2026-11-23: several
    references at once); it closes on a click outside it or Escape."""

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt's name
        action = self.activeAction()
        if action is not None and action.isCheckable() and action.isEnabled():
            action.setChecked(not action.isChecked())
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):  # noqa: N802 - Qt's name
        action = self.activeAction()
        if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space)
                and action is not None and action.isCheckable()):
            action.setChecked(not action.isChecked())
            return
        super().keyPressEvent(event)


def compute_time(results, config) -> str:
    """How long Compute takes, for the status line: the peptide analysis is quick (about half
    a minute for 25,000 spectra); the fragment ladder rate fits (``[analysis.fragments]
    fit_rates`` on a run with fragment series) take minutes (115 s for 17,000 fragment
    spectra, measured 2026-11-05 on 48 cores; drawing a ladder figure then refits its steps)."""
    from ..report.analysis import fragment_series_ids

    quick = "about half a minute for 25,000 spectra"
    try:
        fits = config.fragments is not None and config.fragments.fit_rates
        series = fragment_series_ids(results)[0] if fits else set()
    except Exception:  # noqa: BLE001 - a hint only
        series = set()
    if not series:
        return quick
    n = int(np.isin(np.asarray(results.fit.data_id), list(series)).sum())
    return (f"{quick}; the ladder rate fits of its {n:,} fragment spectra take minutes: "
            "about 2 for 17,000")


def setting_table(setting: str) -> str:
    """The table of a setting as the catalogue writes it: ``[analysis] reference`` ->
    ``analysis``, ``[[analysis.regions]]`` -> ``analysis.regions``."""
    return setting.split("]")[0].strip("[")


class AnalysisWorkspace(QWidget):
    """``hxex analyse``'s figures for the results shown, with the project's settings."""

    def __init__(self, window=None):
        super().__init__()
        self.window = window
        self._error: str | None = None  # why the last computation failed (None: it did not)
        # the tree (hxex.gui.analysis_tree): each kind at its place (key -> Entry), the rows
        # above the kinds (path -> item), the kinds' rows (key -> item), the rows opened (None
        # before the first fill: the tops open)
        self._nodes: dict = {}
        self._row_items: dict = {}
        self._leaf_items: dict = {}
        self._expanded: set | None = None
        self._figure_count = 0
        self._drawn_draw = None  # the drawing function of the figure shown
        self._keep_stem: str | None = None  # the figure to choose again after a refill
        # each reference's analysis: (hxex.report.AnalysisOutputs, (the results' key
        # (results_key), [analysis] text, results path, review table's stamp) it was computed
        # from)
        self._analyses: dict[str, tuple] = {}
        self._cancel: threading.Event | None = None
        self._computing = Latest(self)
        self._computing_path = None  # the results path the running computation is of
        # Recompute waits for these results (a project file) read again (None: nothing waits)
        self._compute_when_loaded: Path | None = None
        self.render = getattr(window, "render", "matplotlib")

        self.status = QLabel("No results open.")
        self.status.setWordWrap(True)
        self.compute_button = QPushButton("Compute")
        self.compute_button.setToolTip(tip("compute the analysis as hxex analyse does (nothing is "
                                           "written)"))
        self.compute_button.clicked.connect(self.compute)
        # a Stop for the computation (the ladder rate fits take minutes; audit of 2026-11-15,
        # E open question 2): shown while one runs
        self.stop_button = QPushButton("Stop")
        self.stop_button.setToolTip(tip("stop the computation (the figures shown stay as they "
                                        "are)"))
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.hide()
        # a review K the run's screen skipped: hxex analyse refuses, so does Compute, and
        # Run fits it (E-4)
        self.resume = QPushButton("Fit the chosen K (Run)…")
        self.resume.setToolTip(tip("hxex run: brings the results up to date, which fits (and "
                                   "bootstraps) the K your review chose where the run's quick "
                                   "screen skipped it; until then the analysis cannot be made, "
                                   "as hxex analyse refuses it"))
        self.resume.clicked.connect(lambda: self._resume())
        self.resume.hide()
        # [analysis.fragments] fit_rates, for results with fragment series: written to the
        # project file, so hxex analyse does the same (owner, 2026-11-05)
        self.fit_rates = QCheckBox("fit ladder rates")
        self.fit_rates.setToolTip(
            tip("[analysis.fragments] fit_rates: fit each ladder step's decay rate (ladder and "
                "residue rates, the fitted curves); minutes on a large run"))
        self.fit_rates.hide()
        self.fit_rates.toggled.connect(self._set_fit_rates)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)  # computing: no count to show
        self.busy.setMaximumWidth(120)
        self.busy.hide()
        self.reference = QComboBox()
        self.reference.setToolTip(tip("which reference's analysis is shown: [analysis] reference "
                                      "names several (or all), and hxex analyse writes a folder "
                                      "for each"))
        self.reference.currentIndexChanged.connect(lambda _i: self.refresh())
        self.reference_label = QLabel("against")
        # the references, ticked in a list (owner, 2026-11-23): written to the project file's
        # [analysis] reference; Compute makes the analysis against each, and the tree has a
        # Comparisons part per reference
        self.references = QToolButton()
        self.references.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        # Qt's own small arrow in the corner left out: the text ends in ▾ (owner, 2026-11-23)
        self.references.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0; }")
        self.references.setMenu(_StayingMenu(self.references))
        self.references.menu().aboutToHide.connect(self._references_ticked)
        self.references.setToolTip(tip(
            "the samples the others are compared with: tick as many as you like; when the list "
            "closes they are written to [analysis] reference in the project file, so hxex "
            "analyse makes the same comparisons; Compute makes the analysis against each one "
            "ticked, and the tree has a Comparisons part for each"))
        self.references.hide()
        self._ticked: list[str] = []  # the references ticked (the project's [analysis] reference)
        self._compute_note = (0, 0, "")  # the reference being computed, of how many
        self._compute_timer = QTimer(self)
        self._compute_timer.setInterval(500)
        self._compute_timer.timeout.connect(self._show_compute_progress)
        top = QHBoxLayout()
        top.addWidget(self.reference_label)
        top.addWidget(self.reference)
        top.addWidget(self.references)
        top.addWidget(self.compute_button)
        top.addWidget(self.stop_button)
        top.addWidget(self.resume)
        top.addWidget(self.fit_rates)
        top.addWidget(self.busy)
        top.addWidget(self.status, 1)

        # no reference with two samples or more: said loudly, with the choice at hand (owner,
        # 2026-11-13: the comparisons were off and only the list's "off" lines said why)
        self.no_reference = QWidget()
        bar = QHBoxLayout(self.no_reference)
        bar.setContentsMargins(0, 0, 0, 0)
        self.no_reference_text = QLabel()
        self.no_reference_text.setWordWrap(True)
        self.no_reference_text.setStyleSheet(f"color: {ERROR_COLOUR};")
        self.choose_reference = QComboBox()
        self.choose_reference.setToolTip(tip(
            "writes [analysis] reference to the project file: the sample the others are "
            "compared with; \"every sample in turn\" writes reference = \"all\" (an analysis "
            "for each)"))
        self.choose_reference.activated.connect(self._reference_chosen)
        bar.addWidget(self.choose_reference)
        bar.addWidget(self.no_reference_text, 1)
        self.no_reference.hide()

        self.list = QTreeWidget()
        self.list.setColumnCount(2)
        self.list.setHeaderLabels(["figure", ""])
        self.list.setUniformRowHeights(True)
        self.list.setItemDelegateForColumn(0, _UnderParent(self.list))
        header = self.list.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.list.currentItemChanged.connect(self._selected)
        self.about = QLabel()
        self.about.setWordWrap(True)
        self.about.setTextFormat(Qt.TextFormat.PlainText)
        self.change = QPushButton()
        self.change.clicked.connect(self._change)
        self.change.hide()
        left = QWidget()
        side = QVBoxLayout(left)
        side.setContentsMargins(0, 0, 0, 0)
        side.addWidget(self.list, 1)
        side.addWidget(self.about)
        side.addWidget(self.change)

        self.panel = FigurePanel(render=self.render, hint=HINT + SPECTRA_HINT)
        # the figure of the kind: its sample, time, form, page (one row per kind in the list)
        self.choices = ChoicesRow()
        self.choices.chosen.connect(self._draw_stem)
        self.panel.add_row(self.choices)
        self.panel.drawn.connect(self._show_data)
        self.panel.clicked_row.connect(self._clicked_row)
        self.panel.rows_asked.connect(self._rows_asked)
        # beside the figure, its values: what it draws, one row per bar (owner, 2026-11-04)
        self.data = DataPanel()
        # the rows of every figure of the kind, not only of the one shown (owner, 2026-11-04)
        self.every_figure = QCheckBox("every figure of this kind")
        self.every_figure.setToolTip(tip(
            "list the rows of all the kind's figures, a figure column naming each one's file "
            "(gathered when this tab is shown: it draws every figure of the kind)"))
        self.every_figure.toggled.connect(self._every_toggled)
        self.data.add_option(self.every_figure)
        self._kind_rows = None  # the kind whose figures' rows are all shown (every_figure)
        self._rows_task = Latest(self, drawing_pool())
        # the rows gathered of each kind and place, kept until the analysis changes (gathering
        # draws every figure: a minute for a project's ladders; owner, 2026-11-23)
        self._row_cache: dict[str, tuple[list, list]] = {}
        self._cache_owners: list = []  # the analyses the cached rows and figures come from
        self._gather_done = (0, 0)  # figures gathered so far, of how many
        self._progress = QTimer(self)
        self._progress.setInterval(300)
        self._progress.timeout.connect(self._show_progress)
        self._rows_wanted = None  # (kind, stems, outputs) to gather when the data tab is shown
        self._gathering = 0  # which gathering is current; a stale one stops between figures
        self.data.activated.connect(self.go_to_spectra)
        self.views = QTabWidget()
        self.views.addTab(self.panel, "figure")
        self.views.addTab(self.data, "data")
        self.views.setTabToolTip(1, tip("the values of what the figure draws, one row per bar: "
                                        "sort, copy, or save them as CSV"))
        # the HDX data summary of the analysis shown (owner, 2026-11-13: a tab beside these)
        self.summary = SummaryPanel()
        self._summary_shown = None  # the summary columns shown (redrawn only when they change)
        self.views.addTab(self.summary, "HDX Summary Table")
        self.views.setTabToolTip(2, tip("the HDX data summary of Masson et al. 2019, one column "
                                        "per sample, as hxex analyse writes summary.csv and "
                                        "summary.md; your own items: [analysis.summary]"))
        self.views.currentChanged.connect(lambda _i: self._gather_if_shown())
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.views)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)  # a panel dragged to nothing left no trace
        split.setSizes([400, 800])
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.no_reference)
        layout.addWidget(split, 1)
        self._change_table = ""
        self._planned: dict[str, object] = {}
        self._latest: str | None = None  # the reference of the latest analysis computed
        self._by_group = False  # [analysis.groups]: one analysis per group, nothing shared
        self._key_cache: tuple = (None, None)  # (results object, its key): E-9
        self.refresh()

    # --- what the analysis is made from ---------------------------------------------------

    @property
    def shown_reference(self) -> str:
        """The reference whose analysis is shown ('' with none): the one whose Comparisons part
        holds the row selected, else the first ticked; or with ``[analysis.groups]`` the
        group."""
        if not self._by_group and self._ticked:
            from .analysis_tree import comparisons_top

            item = self.list.currentItem()
            while item is not None and item.parent() is not None:
                item = item.parent()
            top = item.text(0) if item is not None else ""
            return next((r for r in self._ticked if comparisons_top(r) == top), self._ticked[0])
        data = self.reference.currentData()
        return data if isinstance(data, str) else ""

    @property
    def outputs(self):
        """The analysis shown (None: not computed yet)."""
        entry = self._analyses.get(self.shown_reference)
        return entry[0] if entry else None

    @property
    def _made_from(self):
        entry = self._analyses.get(self.shown_reference)
        return entry[1] if entry else None

    @property
    def shared_reference(self) -> str | None:
        """The reference of the analysis the figures common to every reference come from: the
        latest computed (so a Recompute, after a review decision say, renews them too); None
        before any. With ``[analysis.groups]`` the groups hold different samples, so nothing
        is shared: the group shown, once computed."""
        if self._by_group:
            shown = self.shown_reference
            return shown if shown in self._analyses else None
        return self._latest if self._latest in self._analyses else None

    def _of(self, ref: str) -> str:
        """`` against REF`` (or `` of group NAME``) for the status line; empty without one."""
        if not ref:
            return ""
        if ref == ACROSS:
            return " of every group"
        return f" of group {ref}" if self._by_group else f" against {ref}"

    @property
    def shared_outputs(self):
        """The analysis the figures common to every reference come from (None: none yet)."""
        ref = self.shared_reference
        return self._analyses[ref][0] if ref is not None else None

    def _fill_references(self, config) -> None:
        """The references to choose from (shown only when there are several)."""
        from ..report import analysis_references

        refs = analysis_references(self.results, config)
        if config.by_group:  # every group in the tree, computed together (owner, 2026-11-22)
            refs = [ACROSS]
        now = [self.reference.itemData(i) for i in range(self.reference.count())]
        if now != refs:
            keep = self.shown_reference
            self.reference.blockSignals(True)
            self.reference.clear()
            for ref in refs:
                self.reference.addItem("(across the groups)" if ref == ACROSS else
                                       ref or "(no reference)", ref)
            index = self.reference.findData(keep)
            self.reference.setCurrentIndex(max(index, 0))
            self.reference.blockSignals(False)
        self._by_group = config.by_group
        self.reference_label.setText("comparisons against")
        if config.by_group:  # each group's reference is on its own page
            self._ticked = []
            self.references.hide()
            self.reference.hide()
            self.reference_label.hide()
            return
        self.reference.hide()  # the tree has a part per reference
        self._ticked = [r for r in refs if r]
        samples = list(config.samples) or self.results.samples()
        menu = self.references.menu()
        if [a.data() for a in menu.actions()] != samples or \
                [a.isChecked() for a in menu.actions()] != [s in self._ticked for s in samples]:
            menu.clear()
            for sample in samples:
                action = menu.addAction(sample)
                action.setData(sample)
                action.setCheckable(True)
                action.setChecked(sample in self._ticked)
        # one reference by name, several as a count (owner, 2026-11-23): the tree names each
        shown = (self._ticked[0] if len(self._ticked) == 1 else
                 f"multiple ({len(self._ticked)})" if self._ticked else "none")
        self.references.setText(f"{shown} ▾")
        self.references.setEnabled(self._project() is not None)
        visible = len(samples) > 1
        self.references.setVisible(visible)
        self.reference_label.setVisible(visible)

    def _references_ticked(self) -> None:
        """The samples ticked in the references list written to [analysis] reference (none
        ticked: the line taken out); the analysis then compares with each."""
        from ..project_edit import ProjectError
        from .project_editor import _quoted

        project = self._project()
        if project is None:
            return
        ticked = [a.data() for a in self.references.menu().actions() if a.isChecked()]
        if ticked == self._ticked:  # the list closed without a change
            return
        value = (None if not ticked else _quoted(ticked[0]) if len(ticked) == 1
                 else "[" + ", ".join(_quoted(t) for t in ticked) + "]")
        problem = None
        try:
            self._write(project, [("analysis.reference", value)],
                        f"[analysis] reference: {', '.join(ticked) or 'none'}")
            log.info("[analysis] reference set to %s", ", ".join(ticked) or "none")
        except (ProjectError, OSError, ValueError, TypeError) as exc:
            problem = str(exc)
            log.error("the references were not set: %s", exc)
        self.refresh()
        if problem is not None:
            self.status.setText(f"The references were not set: {problem}")

    @property
    def results(self):
        page = getattr(self.window, "results_page", None)
        return getattr(page, "results", None)

    def _project(self) -> Path | None:
        """The project file whose [analysis] applies: only results opened through it."""
        window = self.window
        if window is None or not window.project_results:
            return None
        return window.project

    def _settings(self):
        """(the [analysis] settings, the folder relative paths are taken from, their text as
        the project file has them, for telling a change)."""
        from ..analysis import AnalysisConfig

        project = self._project()
        if project is None:
            path = getattr(self.window, "results_path", None)
            return AnalysisConfig(), Path(path).parent if path else Path("."), None
        state = self.window.fresh_project_state()  # the window's one parsed state (Q-2)
        config = state.analysis
        of = state.of_tables("analysis")
        text = (of["values"], of["tables"], of["lists"])  # lists: regions, sites, groups' own
        # relative paths from the project file's real folder, as hxex analyse takes them (a
        # project file reached through a link: audit of 2026-11-15, E-25)
        return config, project.resolve().parent, repr(text)

    def results_key(self) -> tuple | None:
        """What the analysis reads of the results shown, to tell a change of them: the
        bundle's stamp, the review layer's content, and the project's ``[review]`` settings
        (the exclusion rules). Reading the same results again (a ``[figures]`` change) gives
        the same key, so the analysis stays up to date (audit of 2026-11-15, E-9); keeping the
        key, not the Results object, lets old results go (E-22)."""
        results = self.results
        if results is None:
            return None
        if self._key_cache[0] is results:
            return self._key_cache[1]
        created = results.settings.created if results.settings is not None else ""
        review = results.review
        layer = None if review is None else (
            tuple(map(str, review.spectrum_id)), tuple(map(str, review.status)),
            tuple(int(k) for k in review.k_override), tuple(map(str, review.note)))
        rules = getattr(self.window, "_figure_and_review_settings", None)
        settings = (repr(rules(("review",))) if rules is not None and self._project() is not None
                    else None)
        key = (created, layer, settings)
        self._key_cache = (results, key)
        return key

    def why_out_of_date(self, made_from=None) -> str:
        """Why the analysis shown (or the one ``made_from`` describes) no longer matches the
        results and settings (empty: it does)."""
        made_from = made_from or self._made_from
        if made_from is None:
            return ""
        key, text, _, stamp = made_from
        if self.results_key() != key:
            return "the results changed (a new run, or a review decision)"
        try:
            now = self._settings()[2]
        except Exception:  # noqa: BLE001 - a project file that does not read now
            return "the project file does not read"
        if now != text:
            return "the project's [analysis] settings changed"
        if table_stamp(self._project()) != stamp:
            return ("the review table changed since it was computed, so the analysis may "
                    "change")
        return ""

    # --- the list ---------------------------------------------------------------------------

    def refresh(self) -> None:
        """Show the plan of the current results and settings, the figures computed, and whether
        they are out of date (computed only when asked: Compute)."""
        self._fill()
        self._show_fit_rates()
        self._show_no_reference()
        self._show_summary()
        self.resume.setVisible(self.results is not None and bool(self.results.unfitted_choices))
        if self.results is None:
            self.status.setText("No results open: open a project file whose results exist, or "
                                "a results bundle.")
            self.compute_button.setEnabled(False)
            return
        # busy: computing, or reading the results again for a Recompute (a refresh meanwhile
        # enabled Compute and replaced the line: audit of 2026-11-15, E-19)
        self.compute_button.setEnabled(not self._busy)
        if self._busy:
            return
        if not self._by_group and len(self._ticked) > 1:
            self._status_of_several()
            return
        why = self.why_out_of_date()
        if self.outputs is None:
            self.compute_button.setText("Compute")
            if self._error is None:
                ref = self.shown_reference
                shared = self.shared_reference
                stale = (self.why_out_of_date(self._analyses[shared][1])
                         if shared is not None else "")
                try:
                    hint = compute_time(self.results, self._settings()[0])
                except Exception:  # noqa: BLE001 - the project file does not read: no hint
                    hint = "about half a minute for 25,000 spectra"
                self.status.setText(
                    "Not computed yet" + self._of(ref) + ": Compute "
                    f"makes the figures hxex analyse would write for {len(self.results.fit):,} "
                    f"spectra ({hint}; nothing is written)."
                    + (f" The figures the same for every reference are shown from the analysis "
                       f"against {shared}" + (f" (out of date: {stale})" if stale else "") + "."
                       if shared is not None and not self._by_group else "")
                    + self._settings_note())
                self._redraw_if_another()
        elif why:
            self.status.setText(f"Out of date: {why}. The figures shown are the earlier "
                                "analysis's; Recompute to see the new one.")
            self.compute_button.setText("Recompute")
        else:
            n = self._figure_count
            ref = self.shown_reference
            self.status.setText(f"{n} figure(s)" + self._of(ref)
                                + ", as hxex analyse would write them" + self._settings_note())
            self._redraw_if_another()
            self.compute_button.setText("Recompute")

    def _status_of_several(self) -> None:
        """The status line with several references ticked: which are not computed yet or out
        of date (Compute makes them), else the figures against all of them."""
        made = {r: self._analyses.get(r) for r in self._ticked}
        missing = [r for r, m in made.items() if m is None]
        stale = {r: self.why_out_of_date(m[1]) for r, m in made.items() if m is not None}
        stale = {r: why for r, why in stale.items() if why}
        if missing or stale:
            self.compute_button.setText("Compute")
            parts = [f"not computed yet against {', '.join(missing)}"] if missing else []
            if stale:
                parts.append(f"out of date against {', '.join(stale)} "
                             f"({next(iter(stale.values()))})")
            text = "; ".join(parts)
            try:
                hint = compute_time(self.results, self._settings()[0])
            except Exception:  # noqa: BLE001 - the project file does not read: no hint
                hint = "about half a minute for 25,000 spectra"
            self.status.setText(f"{text[0].upper()}{text[1:]}: Compute makes them ({hint} "
                                "each; nothing is written)." + self._settings_note())
        else:
            self.compute_button.setText("Recompute")
            self.status.setText(f"{self._figure_count} figure(s) against "
                                f"{', '.join(self._ticked)}, as hxex analyse would write them"
                                + self._settings_note())
        self._redraw_if_another()

    def _show_compute_progress(self) -> None:
        if not self._computing.running:
            self._compute_timer.stop()
            return
        i, n, ref = self._compute_note
        self.status.setText(f"Computing the analysis against {ref} ({i + 1} of {n})…")

    @property
    def _busy(self) -> bool:
        return self._computing.running or self._compute_when_loaded is not None

    def _show_summary(self) -> None:
        """The summary tab: the summary of the analysis shown (filled again only when that
        analysis changes), and whether it is out of date. Kept while an analysis computes."""
        if self._computing.running:
            return
        outputs = self.outputs
        columns = outputs.tables.get("summary") if outputs is not None else None
        if columns is not self._summary_shown:
            self._summary_shown = columns
            self.summary.show_summary(columns)
        if columns is not None:
            self.summary.set_out_of_date(self.why_out_of_date())

    def _show_fit_rates(self) -> None:
        """The 'fit ladder rates' box: shown for results with fragment series whose fragment
        analysis is on, ticked as the project file says, and changeable only with the project
        file open (a bundle opened alone has no file to write it to) and nothing computing."""
        from ..report.analysis import fragment_series_ids

        config = None
        if self.results is not None:
            try:
                config = self._settings()[0]
            except Exception:  # noqa: BLE001 - the project file does not read: no box
                config = None
        try:
            fragments = bool(config is not None and config.fragments is not None
                             and fragment_series_ids(self.results)[0])
        except Exception:  # noqa: BLE001 - results without scan records
            fragments = False
        self.fit_rates.setVisible(fragments)
        if not fragments:
            return
        self.fit_rates.blockSignals(True)
        self.fit_rates.setChecked(bool(config.fragments.fit_rates))
        self.fit_rates.blockSignals(False)
        self.fit_rates.setEnabled(self._project() is not None and not self._busy)

    def _show_no_reference(self) -> None:
        """The warning that the samples are not compared: results with two samples or more and
        no ``[analysis] reference``; with the project file open, a list to choose one."""
        samples: list[str] = []
        if self.results is not None:
            try:
                config = self._settings()[0]
            except Exception:  # noqa: BLE001 - the project file does not read: said elsewhere
                config = None
            if config is not None and not config.reference and not config.runs \
                    and not config.groups:
                samples = list(config.samples) or self.results.samples()
        if len(samples) < 2:
            self.no_reference.hide()
            return
        project = self._project() is not None
        self.no_reference_text.setText(
            "Reference needed for comparison plots" if project else
            "Reference needed for comparison plots: open the project file to choose one")
        self.choose_reference.setVisible(project)
        items = ["choose a reference…", *samples, "every sample in turn"]
        if [self.choose_reference.itemText(i)
                for i in range(self.choose_reference.count())] != items:
            self.choose_reference.blockSignals(True)
            self.choose_reference.clear()
            self.choose_reference.addItem(items[0], None)
            for name in samples:
                self.choose_reference.addItem(name, name)
            self.choose_reference.addItem(items[-1], "all")
            self.choose_reference.blockSignals(False)
        self.choose_reference.setCurrentIndex(0)
        self.no_reference.show()

    def _write(self, project, changes, what: str) -> None:
        """A change to the project file through the window's one route (the Project editor's
        Undo records it; audit of 2026-11-15, Q-1), else directly."""
        from ..project_edit import set_settings

        writer = getattr(self.window, "write_settings", None)
        if writer is not None:
            writer(changes, what)
        else:
            set_settings(project, changes, unfinished=True)

    def _reference_chosen(self, index: int) -> None:
        """Write the reference chosen to the project file; the analysis then compares."""
        from ..project_edit import ProjectError
        from .project_editor import _quoted

        name = self.choose_reference.itemData(index)
        project = self._project()
        if name is None or project is None:
            return
        problem = None
        try:
            # as TOML text, quoted: a sample named like a number or a date would be read as one,
            # or refused (audit of 2026-11-15, E-8)
            self._write(project, [("analysis.reference", _quoted(str(name)))],
                        f"[analysis] reference: {name}")
            log.info("[analysis] reference set to %s", name)
        except (ProjectError, OSError, ValueError, TypeError) as exc:
            problem = str(exc)
            log.error("the reference was not set: %s", exc)
        self.refresh()
        if problem is not None:  # after the refresh, which writes the line too
            self.status.setText(f"The reference was not set: {problem}")

    def _set_fit_rates(self, on: bool) -> None:
        """Write ``[analysis.fragments] fit_rates`` (true or false) to the project file; the
        analysis shown is then out of date."""
        from ..project_edit import ProjectError

        project = self._project()
        if project is None:
            return
        try:
            self._write(project, [("analysis.fragments.fit_rates", "true" if on else "false")],
                        f"[analysis.fragments] fit_rates: {'true' if on else 'false'}")
        except (ProjectError, OSError, ValueError) as exc:
            log.error("fit_rates was not changed: %s", exc)
            self.status.setText(f"fit_rates was not changed: {exc}")
        self.refresh()

    def _redraw_if_another(self) -> None:
        """The figures changed (another reference, a new analysis): the figure selected drawn
        again, unless the drawing shown is still its own (a figure the same for every
        reference stays as it is)."""
        node = self._current_node()
        if node is None:
            return
        draw = dict(node.figures).get(self.choices.stem or "")
        if draw is None or draw is not self._drawn_draw:
            self._selected(self.list.currentItem())

    def _current_node(self):
        item = self.list.currentItem()
        return self._nodes.get(item.data(0, NODE)) if item is not None else None

    def _settings_note(self) -> str:
        return ("" if self._project() is not None else
                " (a bundle opened alone: the default [analysis] settings; open the project "
                "file for its own)")

    def _placed(self, config):
        """The analyses the tree shows: with groups each group's (and the regions across them),
        else the shown reference's comparisons with the figures of the levels from the latest
        analysis (the same against every reference)."""
        from .analysis_tree import Placed

        if config.by_group:
            placed = []
            for g in config.groups:
                outputs = self._analyses.get(g, (None,))[0]
                placed.append(Placed(g, outputs, outputs, config.one(g).reference or ""))
            return placed, self._analyses.get(ACROSS, (None,))[0]
        if not self._ticked:
            return [Placed(None, self.shared_outputs, self.outputs, "")], None
        # the levels from the latest analysis (the same against every reference), then a
        # Comparisons part per reference ticked (owner, 2026-11-23)
        from .analysis_tree import comparisons_top

        return ([Placed(None, self.shared_outputs, None)]
                + [Placed(None, None, self._analyses.get(r, (None,))[0], r, comparisons_top(r))
                   for r in self._ticked]), None

    def _fill(self) -> None:
        """The tree: Coverage, Summary, Single samples, Comparisons (:mod:`.analysis_tree`),
        each kind of figure at its place once computed, the kinds left out under Not drawn."""
        from ..report.catalogue import Situation, plan
        from . import analysis_tree as at

        current = self.list.currentItem()
        keep = current.data(0, NODE) if current is not None else None
        # the figure chosen, the rows opened, and the place in the list, kept across the
        # rebuild (audit of 2026-11-15, E-16)
        self._keep_stem = self.choices.stem
        if self.list.topLevelItemCount():
            self._expanded = {key for key, item in self._row_items.items() if item.isExpanded()}
        scrolled = self.list.verticalScrollBar().value()
        self.list.blockSignals(True)
        self.list.clear()
        self._planned, self._nodes, self._row_items, self._leaf_items = {}, {}, {}, {}
        self._figure_count = 0
        if self.results is not None:
            try:
                config = self._settings()[0]
                self._fill_references(config)
                one = (config if config.single_reference or config.by_group
                       else config.one(self.shown_reference))
                planned = plan(Situation.of(analysis=one, results=self.results))
            except Exception as exc:  # noqa: BLE001 - settings these results refuse
                self.list.blockSignals(False)
                self._error = str(exc)
                self.status.setText(f"The analysis cannot be made: {exc}")
                return
            self._error = None
            self._planned = {p.kind.name: p for p in planned}
            placed, across = self._placed(config)
            computed = any(p.levels is not None or p.compared is not None for p in placed)
            found = at.entries(placed, across) if computed else []
            have = {e.kind for e in found}
            shown, left = at.planned_entries(planned, (lambda k: k in have) if computed
                                             else None)
            if not config.by_group and self._ticked:
                # each reference not computed yet: its comparisons named in its part, not drawn
                waiting = [p for p in placed if p.reference and p.compared is None]
                named = [e for e in at.planned_entries(planned)[0]
                         if e.path[0] == at.COMPARISONS]
                shown = [e for e in shown if e.path[0] != at.COMPARISONS]
                if computed and waiting:
                    left = [e for e in left if at.HOME[e.kind] != at.COMPARISONS]
                shown += [at.Entry((p.comparisons,), e.kind) for p in waiting for e in named]
            # the figures hxex analyse writes (not the window's own of one sample)
            written = {id(o): o for p in placed for o in (p.levels, p.compared)
                       if o is not None}
            self._figure_count = sum(len(o.figures) for o in written.values()) + (
                len(across.figures) if across is not None else 0)
            refs = {p.group: p.reference for p in placed}
            parts = [p.comparisons for p in placed]

            def place(entry):  # the parts in order, each reference's in the list's order
                t = entry.path[0]
                return (at.TOPS.index(at.top_of(t)) if t in at.TOPS or t in parts else 99,
                        parts.index(t) if t in parts else 0)

            for entry in sorted(found + shown + left, key=place):
                self._add_entry(entry, computed, refs)
            if keep in self._leaf_items:
                self.list.setCurrentItem(self._leaf_items[keep])
        self.list.blockSignals(False)
        self.list.verticalScrollBar().setValue(scrolled)
        self._describe(self.list.currentItem())

    def _row(self, path: tuple, refs: dict) -> QTreeWidgetItem:
        """The row of ``path`` (a top, a group or class, a sample or form, a level), made with
        the rows above it when new."""
        from . import analysis_tree as at

        if path in self._row_items:
            return self._row_items[path]
        item = QTreeWidgetItem([path[-1], ""])
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        font = item.font(0)
        if len(path) == 1:
            font.setBold(True)
            item.setToolTip(0, tip(at.TOP_WORDS.get(at.top_of(path[0]), "")))
        elif path[-1] in at.LEVELS:
            font.setItalic(True)
        elif path[0] == at.COMPARISONS and len(path) == 2 and refs.get(path[1]):
            item.setText(1, f"against {refs[path[1]]}")  # a group's own reference
        item.setFont(0, font)
        if path[0] == at.NOT_DRAWN:  # grey, as the kinds under it (owner, 2026-11-23)
            for col in (0, 1):
                item.setForeground(col, self.palette().placeholderText())
        if len(path) == 1:
            self.list.addTopLevelItem(item)
        else:
            self._row(path[:-1], refs).addChild(item)
        self._row_items[path] = item
        expanded = self._expanded if self._expanded is not None else {
            (t,) for t in at.TOPS}  # at first the tops open
        item.setExpanded(path in expanded)
        return item

    def _add_entry(self, entry, computed: bool, refs: dict) -> None:
        """One kind of figure at its place in the tree (its figures chosen above the figure)."""
        from . import analysis_tree as at

        p = self._planned.get(entry.kind)
        if p is None:
            return
        key = "\x1f".join(entry.key)
        self._nodes[key] = entry
        n = len(entry.figures)
        text = ((f"{n} figure" + "s" * (n != 1)) if n else
                self._state_text(p, computed))
        item = QTreeWidgetItem([title_at(entry, p.kind.title), text])
        item.setData(0, KIND, entry.kind)
        item.setData(0, NODE, key)
        item.setToolTip(0, tip(p.kind.shows))
        if not n and (computed or p.state == "off" or entry.path[0] == at.NOT_DRAWN):
            for col in (0, 1):
                item.setForeground(col, self.palette().placeholderText())
        self._row(entry.path, refs).addChild(item)
        self._leaf_items[key] = item

    def _state_text(self, planned, computed: bool) -> str:
        if planned.state == "off":
            return "off"
        if not computed:
            return STATE_WORDS[planned.state]
        return "none for these results"

    def _selected(self, item, _previous=None) -> None:
        """A kind selected: its figure chosen in the lists above the figure (the one chosen
        before, after a refill; else the sample, time, and form of the kind shown before, where
        this kind has them; else its first), and drawn. A row above the kinds (a group, a
        sample) changes nothing."""
        from ..report.catalogue import figure_choices, kind_named

        self._describe(item)
        node = self._nodes.get(item.data(0, NODE)) if item is not None else None
        keep, self._keep_stem = self._keep_stem, None
        if node is None:
            return
        self._kind_rows = None
        self._rows_wanted = None
        self._gathering += 1  # a gathering under way stops at its next figure
        if not node.figures:  # a kind with no figure here: why, in the figure's place
            self.choices.set_figures(None)
            self.panel.save_name = ""
            self._drawn_draw = None
            self._clear_figure(self._off_text(self._planned.get(node.kind))
                               or ("not computed yet: Compute makes the figures"
                                   if not self._computed_any() else
                                   "no figure of this kind for these results"))
            return
        stem = self.choices.set_figures(figure_choices(kind_named(node.kind),
                                                       [s for s, _ in node.figures]),
                                        keep=self.choices.values(), stem=keep)
        self._draw_stem(stem)

    def _computed_any(self) -> bool:
        return any(e.figures for e in self._nodes.values())

    def _draw_stem(self, stem: str) -> None:
        """Draw the figure ``stem`` of the kind selected (chosen in the lists above it)."""
        node = self._current_node()
        if node is None:
            return
        draw = dict(node.figures).get(stem)
        self._drawn_draw = draw
        if self.every_figure.isChecked():
            self._want_kind_rows(node)
        if draw is None:  # a figure of an earlier analysis this one does not draw
            self.panel.save_name = ""
            self._clear_figure("this figure is not in the analysis computed now")
            return
        self.panel.save_name = stem
        self._check_caches()
        self.panel.draw(draw, key=(node.key, stem))

    def _want_kind_rows(self, node) -> None:
        """The rows of every figure of the kind selected in the data tab, gathered when it is
        shown (once per kind selected)."""
        key = "\x1f".join(node.key)
        if self._kind_rows == key:
            return
        self._kind_rows = key
        self._rows_wanted = (key, node.kind, [s for s, _ in node.figures], dict(node.figures))
        self.data.show_rows([], [], node.kind)
        self.data.set_note(f"The rows of the {len(node.figures)} figures of this kind are "
                           "gathered when this tab is shown.")
        self.views.setTabText(1, "data")
        self._gather_if_shown()

    def _every_toggled(self, on: bool) -> None:
        node = self._current_node()
        self._gathering += 1
        self._kind_rows = None
        self._rows_wanted = None
        if on and node is not None and node.figures:
            self._want_kind_rows(node)
        elif not on and self.panel.view.figure is not None:
            self._show_data()

    def _clear_figure(self, message: str = "") -> None:
        self.panel.clear(message)
        self.data.clear()
        self.views.setTabText(1, "data")

    def _show_data(self) -> None:
        """The figure drawn: its data table (a kind selected: every figure's, gathered apart)."""
        from .figures import figure_rows

        if self.every_figure.isChecked() and self._kind_rows is not None:
            return
        columns, rows = figure_rows(self.panel.view.figure)
        self.data.show_rows(columns, rows, self.panel.save_name)
        self.data.set_note("")
        self.views.setTabText(1, f"data ({len(rows)})" if rows else "data")

    def _gather_if_shown(self) -> None:
        """A kind selected and the data tab in view: gather its figures' rows (only then: it
        draws every figure of the kind, minutes for a run with hundreds)."""
        if self._rows_wanted is None or self.views.currentWidget() is not self.data:
            return
        key, kind, stems, figures = self._rows_wanted
        self._rows_wanted = None
        self._gather_rows(key, kind, stems, figures)

    def _gather_rows(self, key: str, kind: str, stems: list[str], figures: dict) -> None:
        """A kind selected: the rows of every one of its figures in the data tab, a ``figure``
        column naming each one's file (owner, 2026-11-04: the first time point alone was
        confusing). Made in the drawing thread (matplotlib is used by one thread at a time);
        a new selection stops it between figures (a figure chosen meanwhile waits for one
        figure, not for all), and a kind whose first figure carries no rows has none."""
        from .figures import figure_rows

        self._kind_rows = key
        self._gathering += 1
        mine = self._gathering
        self._check_caches()
        if key in self._row_cache:  # gathered before, from the same analysis
            self._rows_gathered(key, self._row_cache[key], kind)
            return
        self.data.show_rows([], [], kind)
        self._gather_done = (0, len(stems))
        self._show_progress()
        self._progress.start()

        def work():
            columns, rows = ["figure"], []
            for n, stem in enumerate(stems):
                if self._gathering != mine:  # another selection since: stop
                    return None
                self._gather_done = (n, len(stems))
                draw = figures.get(stem)
                if draw is None:
                    continue
                names, found = figure_rows(draw())
                if n == 0 and not found:  # the kind's figures carry no rows
                    break
                columns += [n for n in names if n not in columns]
                rows += [{"figure": stem, **r} for r in found]
            return columns, rows

        self._rows_task.submit(work, lambda made: self._rows_gathered(key, made, kind),
                               lambda message: self.data.set_note(
                                   f"The rows could not be gathered: {message}"))

    def _show_progress(self) -> None:
        """How far a gathering has come, on the data tab's note line."""
        if not self._rows_task.running:
            self._progress.stop()
            return
        n, total = self._gather_done
        self.data.set_note(f"Gathering the rows: {n} of {total} figures of this kind (each is "
                           "drawn once; kept until the analysis changes)…")

    def _check_caches(self) -> None:
        """Forget the rows and figures kept when the analyses shown are other ones (a Compute,
        other results)."""
        owners = [made for made, _ in self._analyses.values()]
        if len(owners) != len(self._cache_owners) or any(
                a is not b for a, b in zip(owners, self._cache_owners, strict=True)):
            self._row_cache.clear()
            self.panel.forget_kept()
            self._cache_owners = owners

    def _rows_gathered(self, key: str, made, kind: str = "") -> None:
        self._progress.stop()
        if made is None or self._kind_rows != key:  # another selection since
            return
        self._row_cache[key] = made
        columns, rows = made
        if not rows:
            columns = []
        self.data.show_rows(columns, rows, kind)
        self.data.set_note(f"Every figure of this kind ({len({r['figure'] for r in rows})} with "
                           "rows); choose one figure for its rows alone." if rows else
                           "No data table for this kind of figure.")
        self.views.setTabText(1, f"data ({len(rows)})" if rows else "data")

    def _clicked_row(self, row: dict) -> None:
        """A peptide clicked in the figure: its row selected in the data tab (its spectra are a
        right click away)."""
        self.data.select_row(row)

    def rows_menu(self, rows: list[dict]):
        """The right-click menu of the peptides ``rows`` under the pointer: each one's spectra
        (None when there is nothing to offer)."""
        return spectra_menu(rows, self.go_to_spectra, self)

    def _rows_asked(self, rows: list, where) -> None:
        menu = self.rows_menu(rows)
        if menu is not None:
            menu.exec(where)
            menu.deleteLater()

    def go_to_spectra(self, row: dict) -> bool:
        """Show the spectra of the peptide of ``row`` (a data-table row with ``sample``,
        ``start`` and ``end``) in Results; False (said in the status line) when it has none."""
        if self.window is None or not {"sample", "start", "end"} <= set(row):
            return False
        found = self.window.show_peptide(str(row["sample"]), int(row["start"]), int(row["end"]),
                                         str(row.get("peptide", "")))
        if not found:
            self.status.setText(f"{row['sample']} {row['start']}-{row['end']} is not in the "
                                "results tree.")
        return found

    def _describe(self, item) -> None:
        """What the selected kind shows, and, when it is off, why, with its setting."""
        self.change.hide()
        self._change_table = ""
        if item is None:
            self.about.setText("")
            return
        p = self._planned.get(item.data(0, KIND))
        if p is None:  # a row above the kinds: a top says what it holds
            from .analysis_tree import TOP_WORDS, top_of

            path = next((k for k, v in self._row_items.items() if v is item), ())
            words = TOP_WORDS.get(top_of(path[0]), "") if len(path) == 1 else ""
            self.about.setText(f"{path[0]}: {words}." if words else "")
            return
        lines = [f"{p.kind.title}: {p.kind.shows}."]
        node = self._nodes.get(item.data(0, NODE))
        if node is not None and node.figures and node.kind in ONE_SAMPLE_KINDS \
                and node.path[0] == "Single samples":
            lines = [f"{title_at(node, p.kind.title)}: drawn for this sample alone (the "
                     f"window's own; hxex analyse writes the figure of every sample: "
                     f"{p.kind.shows})."]
        if p.state == "off" and not (node is not None and node.figures):
            lines.append(self._off_text(p))
            if p.missing:
                need = p.missing[0]
                self._change_table = setting_table(need.setting)
                self.change.setText(f"Change {need.setting}…")
                self.change.show()
        elif p.reason and p.state != "off":
            lines.append(f"Drawn {p.reason}.")
        self.about.setText("\n".join(lines))

    @staticmethod
    def _off_text(p) -> str:
        """Why a kind of figure is off (empty when it is not)."""
        if p is None or p.state != "off":
            return ""
        if p.missing:
            return "Off: needs " + "; ".join(f"{n.setting} {n.words}" for n in p.missing) + "."
        return f"Off: {p.reason}."

    def _change(self) -> None:
        if self.window is not None and self._change_table:
            self.window.show_setting(self._change_table)

    # --- computing --------------------------------------------------------------------------

    def compute(self) -> None:
        """Compute the analysis of the results shown in the background (nothing is written)."""
        from ..report import analysis_outputs

        results = self.results
        if results is None:
            return
        try:
            config, base, text = self._settings()
        except Exception as exc:  # noqa: BLE001 - the project file does not read
            self.status.setText(f"The analysis cannot be made: {exc}")
            return
        if config.runs:
            self.status.setText("This project compares separate runs ([analysis.runs]): the "
                                "window does not show that yet; hxex analyse writes it.")
            return
        ref = self.shown_reference
        several = (not config.by_group and len(self._ticked) > 1)
        full = config
        if not config.single_reference and ref != ACROSS:
            config = config.one(ref)
        stamp = table_stamp(self._project())
        if self._project() is not None and self.window is not None \
                and stamp != getattr(self.window, "results_review_stamp", stamp):
            # the review table changed since the results were read: read them again first
            log.info("the review table changed: the results are read again before the analysis")
            self._compute_when_loaded = self._project()
            self.status.setText("Reading the results again with the review table's changes…")
            self.busy.show()
            self.compute_button.setEnabled(False)
            self.window.open_results(self._project())
            return
        from ..review_table import refuse_unfitted

        try:  # a review K the screen skipped: refused as hxex analyse refuses it (E-4)
            refuse_unfitted(results)
        except ValueError as exc:
            self.status.setText(f"The analysis cannot be made, as hxex analyse refuses it: "
                                f"{exc} (the Run button beside Compute fits it).")
            log.error("the analysis cannot be made: %s", exc)
            return
        if self._cancel is not None:
            self._cancel.set()  # an older computation still running stops
        cancel = self._cancel = threading.Event()
        n = len(results.fit)
        path = getattr(self.window, "results_path", None)
        key = self.results_key()
        self.status.setText(f"Computing the analysis of {n:,} spectra"
                            + self._of(ref) + f" ({compute_time(results, config)})…")
        self.busy.show()
        self.stop_button.show()
        self.compute_button.setEnabled(False)
        self.fit_rates.setEnabled(False)  # the settings stay as computed until it is done
        self._computing_path = path
        log.info("computing the analysis (as hxex analyse, nothing written)")
        if several:  # every reference ticked not computed yet or out of date, in turn
            todo = [r for r in self._ticked if r not in self._analyses
                    or self.why_out_of_date(self._analyses[r][1])] or list(self._ticked)
            made_each: dict = {}
            self._compute_note = (0, len(todo), todo[0])
            self._show_compute_progress()
            self._compute_timer.start()

            def each():
                for i, r in enumerate(todo):
                    self._compute_note = (i, len(todo), r)
                    made_each[r] = analysis_outputs(results, full.one(r), base=base,
                                                    cancel=cancel.is_set)
                return made_each[todo[-1]]

            self._computing.submit(
                each, lambda outputs: self._computed(outputs, key, text, path, todo[-1], stamp,
                                                     made_each), self._failed)
            return
        if ref == ACROSS:  # every group's analysis (those up to date kept), then across them
            ready = {g: entry[0] for g, entry in self._analyses.items()
                     if g in (config.groups or {}) and not self.why_out_of_date(entry[1])}
            made: dict = {}
            self._computing.submit(
                lambda: across_outputs(results, config, base, ready, made, cancel.is_set),
                lambda outputs: self._computed(outputs, key, text, path, ref, stamp, made),
                self._failed)
            return
        self._computing.submit(
            lambda: analysis_outputs(results, config, base=base, cancel=cancel.is_set),
            lambda outputs: self._computed(outputs, key, text, path, ref, stamp),
            self._failed)

    @property
    def computing(self) -> bool:
        return self._computing.running

    def release(self) -> None:
        """Let go of every analysis computed and the figure shown (the window closes; audit
        of 2026-11-15, M-5)."""
        self.cancel()
        self._analyses = {}  # `outputs` reads from it
        self._key_cache = (None, None)
        self.panel.clear()
        self.panel.forget_kept()  # the figures kept to show again
        self._row_cache.clear()
        self._cache_owners = []

    def cancel(self) -> None:
        """Stop the computation running, if any (other results are shown, or the window
        closes): the library stops at its next step, and its result is never shown."""
        if self._cancel is not None:
            self._cancel.set()
        if self._computing.running:
            log.info("the analysis being computed is stopped")
        self._computing.drop()
        self._computing_path = None
        self.busy.hide()
        self.stop_button.hide()
        self.compute_button.setEnabled(self.results is not None)
        self.fit_rates.setEnabled(self._project() is not None)

    def stop(self) -> None:
        """The Stop button: the computation ends at its next step (between the tables, the
        figures, and each parent's ladder rate fits) and nothing of it is shown; the figures
        of the earlier analysis stay."""
        if not self._computing.running:
            return
        self.cancel()
        self.refresh()
        self.status.setText("The computation was stopped. " + self.status.text())

    def _resume(self) -> None:
        run = getattr(self.window, "run_resume", None)
        if run is not None:
            run()

    def _computed(self, outputs, key, text, path, ref, stamp, groups=None) -> None:
        if self.results is None or path != getattr(self.window, "results_path", None):
            # an analysis of results no longer shown (another project or bundle opened while it
            # was computed): never shown as this one's
            log.info("the analysis of %s is not shown: other results are open now", path)
            return
        self.busy.hide()
        self.stop_button.hide()
        self._analyses[ref] = (outputs, (key, text, path, stamp))
        for name, made in (groups or {}).items():  # across the groups: each group's too
            self._analyses[name] = (made, (key, text, path, stamp))
        self._latest = ref  # the figures the same for every reference come from it now
        log.info("analysis: %d tables and %d figures", len(outputs.tables), len(outputs.figures))
        self.refresh()  # draws the figure shown again from the new analysis, once (E-15)
        if self.list.currentItem() is not None:
            self._redraw_if_another()  # nothing to do when refresh drew it
        else:  # the first kind with figures selected: its first figure drawn
            first = next((k for k, e in self._nodes.items() if e.figures), None)
            if first is not None:
                self.list.setCurrentItem(self._leaf_items[first])

    def _failed(self, message: str) -> None:
        self._compute_when_loaded = None
        self.busy.hide()
        self.stop_button.hide()
        if self._cancel is not None and self._cancel.is_set():  # stopped as the window asked
            log.info("the analysis was stopped")
            self.refresh()
            return
        self.compute_button.setEnabled(True)
        self.fit_rates.setEnabled(self._project() is not None)
        self._error = message
        self.status.setText(f"The analysis could not be made: {message}")
        log.error("the analysis could not be made: %s", message)

    # --- the window --------------------------------------------------------------------------

    def results_changed(self) -> None:
        """Other results are shown: an analysis of other results altogether (another bundle or
        project) is dropped; one of the same results, made again, is marked out of date."""
        path = getattr(self.window, "results_path", None)
        made = {entry[1][2] for entry in self._analyses.values()}
        # the computation running is of other results (also the first, with nothing made yet)
        elsewhere = self._computing.running and path != self._computing_path
        if self.results is None or (made and path not in made) or elsewhere:
            self.cancel()
            self._analyses.clear()
            self._clear_figure()
        self._error = None
        # a Recompute asked for its own results read again, not for whatever opens next
        wanted, self._compute_when_loaded = self._compute_when_loaded, None
        if wanted is not None and path != wanted:
            log.info("the Recompute asked for %s is not made: other results are open now",
                     wanted)
        if not self.computing:
            self.busy.hide()
        self.refresh()
        if wanted is not None and path == wanted and self.results is not None:
            self.compute()

    def load_failed(self) -> None:
        """The results could not be read again (for a Recompute): nothing is computed."""
        if self._compute_when_loaded is not None:
            self._compute_when_loaded = None
            self.busy.hide()
            self.refresh()

    def set_render(self, render: str) -> None:
        self.render = render
        self.panel.set_render(render)

    def showEvent(self, event):  # noqa: N802 - Qt's name
        super().showEvent(event)
        self.refresh()



def across_outputs(results, config, base, ready: dict, made: dict, cancel=None):
    """Every group's analysis as the window shows it (those in ``ready`` reused, the others
    computed and put in ``made``), then, when two groups or more share a region name, the
    table and figures hxex analyse writes in results_analysis_groups
    (``cross_group_regions``), as an :class:`hxex.report.analysis.AnalysisOutputs` (empty
    without them)."""
    from ..hooks import check_cancel
    from ..report import analysis_outputs
    from ..report.analysis import AnalysisOutputs, _time_label, _time_scale
    from ..report.groups import CROSS_TABLE, cross_group_regions, plot_cross_group_regions
    from ..report.results import mark_provisional, provisional_note

    analyses = {}
    for name in config.groups:
        check_cancel(cancel)
        outputs = ready.get(name)
        if outputs is None:
            outputs = made[name] = analysis_outputs(results, config.one(name), base=base,
                                                    cancel=cancel)
        analyses[name] = (config.one(name), outputs.tables)
    from ..report.groups import shared_region_names

    if len(config.groups) < 2 or not shared_region_names(config):  # every group, no regions
        return AnalysisOutputs(tables={}, figures=[], config=config)
    table = cross_group_regions(analyses)
    scale = _time_scale(results)
    note = provisional_note(results)
    times = sorted({float(t) for t in table["time"]})

    def draw(t):
        return lambda: mark_provisional(plot_cross_group_regions(
            table, t, page_width=config.page_size[0], time_scale=scale), note)

    figures = [(f"{CROSS_TABLE}_{_time_label(t, scale)}", draw(t)) for t in times]
    return AnalysisOutputs(tables={CROSS_TABLE: table}, figures=figures, config=config)
