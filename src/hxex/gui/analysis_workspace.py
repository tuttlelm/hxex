# SPDX-License-Identifier: GPL-3.0-or-later
"""The Analysis workspace (docs/PLAN.md, "GUI", "Analysis"): exactly what ``hxex analyse``
writes with the project's settings, shown rather than written.

On the left, every kind of figure ``hxex analyse`` draws (:mod:`hxex.report.catalogue`): drawn,
left to the results, or off, with the reason and a button to the setting that rules it out;
under each kind its figures, once computed. On the right, the figure selected, drawn from
``hxex analyse``'s own list of figures (:func:`hxex.report.analysis_outputs`), so the window and
the files cannot differ, and saved under the name the command gives it.

The analysis is computed in the background when asked (Compute, beside the reference list:
half a minute for a large run), and kept: a change of the results (a new run, a review
decision), of the review table (in the Tables view or another program) or of ``[analysis]``
marks it out of date, keeping the figures, and Recompute computes it again, reading the review
table again first when it changed. With several references (``[analysis] reference`` a list,
or ``"all"``), a list chooses which one is shown; each is computed when asked and kept, as
``hxex analyse`` writes a folder for each.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .data_table import DataPanel, spectra_menu
from .figure_view import HINT, SPECTRA_HINT, FigurePanel
from .review import table_stamp
from .tasks import Latest, drawing_pool

log = logging.getLogger(__name__)

KIND, STEM = Qt.ItemDataRole.UserRole, Qt.ItemDataRole.UserRole + 1
STATE_WORDS = {"drawn": "drawn", "depends": "if the results have it", "off": "off"}


def setting_table(setting: str) -> str:
    """The table of a setting as the catalogue writes it: ``[analysis] reference`` ->
    ``analysis``, ``[[analysis.regions]]`` -> ``analysis.regions``."""
    return setting.split("]")[0].strip("[")


class AnalysisWorkspace(QWidget):
    """``hxex analyse``'s figures for the results shown, with the project's settings."""

    def __init__(self, window=None):
        super().__init__()
        self.window = window
        # each reference's analysis: (hxex.report.AnalysisOutputs, (results object, [analysis]
        # text, results path, review table's stamp) it was computed from)
        self._analyses: dict[str, tuple] = {}
        self._cancel: threading.Event | None = None
        self._computing = Latest(self)
        self._compute_when_loaded = False  # Recompute waits for results read again
        self.render = getattr(window, "render", "matplotlib")

        self.status = QLabel("No results open.")
        self.status.setWordWrap(True)
        self.compute_button = QPushButton("Compute")
        self.compute_button.setToolTip("compute the analysis as hxex analyse does (nothing is "
                                       "written)")
        self.compute_button.clicked.connect(self.compute)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)  # computing: no count to show
        self.busy.setMaximumWidth(120)
        self.busy.hide()
        self.reference = QComboBox()
        self.reference.setToolTip("which reference's analysis is shown: [analysis] reference "
                                  "names several (or all), and hxex analyse writes a folder "
                                  "for each")
        self.reference.currentIndexChanged.connect(lambda _i: self.refresh())
        self.reference_label = QLabel("against")
        top = QHBoxLayout()
        top.addWidget(self.reference_label)
        top.addWidget(self.reference)
        top.addWidget(self.compute_button)
        top.addWidget(self.busy)
        top.addWidget(self.status, 1)

        self.list = QTreeWidget()
        self.list.setColumnCount(2)
        self.list.setHeaderLabels(["figure", ""])
        self.list.setUniformRowHeights(True)
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
        self.panel.drawn.connect(self._show_data)
        self.panel.clicked_row.connect(self._clicked_row)
        self.panel.rows_asked.connect(self._rows_asked)
        # beside the figure, its values: what it draws, one row per bar (owner, 2026-11-04)
        self.data = DataPanel()
        self._kind_rows = None  # a kind selected: the rows of all its figures are shown
        self._rows_task = Latest(self, drawing_pool())
        self.data.activated.connect(self.go_to_spectra)
        self.views = QTabWidget()
        self.views.addTab(self.panel, "figure")
        self.views.addTab(self.data, "data")
        self.views.setTabToolTip(1, "the values of what the figure draws, one row per bar: "
                                    "sort, copy or save them as CSV")
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.views)
        split.setStretchFactor(1, 1)
        split.setSizes([400, 800])
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(split, 1)
        self._change_table = ""
        self._planned: dict[str, object] = {}
        self._drawn_from = None  # the analysis the figure shown was drawn from
        self._latest: str | None = None  # the reference of the latest analysis computed
        self.refresh()

    # --- what the analysis is made from ---------------------------------------------------

    @property
    def shown_reference(self) -> str:
        """The reference whose analysis is shown ('' with none)."""
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
        before any."""
        return self._latest if self._latest in self._analyses else None

    @property
    def shared_outputs(self):
        """The analysis the figures common to every reference come from (None: none yet)."""
        ref = self.shared_reference
        return self._analyses[ref][0] if ref is not None else None

    def outputs_for(self, kind: str):
        """The analysis a kind of figure is drawn from: the shown reference's for a comparison,
        the latest for a figure the same against every reference."""
        from ..report.catalogue import kind_named

        try:
            compares = kind_named(kind).compares
        except (KeyError, ValueError, StopIteration):
            compares = True
        return self.outputs if compares else self.shared_outputs

    def _fill_references(self, config) -> None:
        """The references to choose from (shown only when there are several)."""
        from ..report import analysis_references

        refs = analysis_references(self.results, config)
        now = [self.reference.itemData(i) for i in range(self.reference.count())]
        if now != refs:
            keep = self.shown_reference
            self.reference.blockSignals(True)
            self.reference.clear()
            for ref in refs:
                self.reference.addItem(ref or "(no reference)", ref)
            index = self.reference.findData(keep)
            self.reference.setCurrentIndex(max(index, 0))
            self.reference.blockSignals(False)
        several = len(refs) > 1
        self.reference.setVisible(several)
        self.reference_label.setVisible(several)

    @property
    def results(self):
        page = getattr(self.window, "results_page", None)
        return getattr(page, "results", None)

    def _project(self) -> Path | None:
        """The project file whose [analysis] applies: only results opened through it."""
        window = self.window
        if window is None or window.project is None or window.results_path != window.project:
            return None
        return window.project

    def _settings(self):
        """(the [analysis] settings, the folder relative paths are taken from, their text as
        the project file has them, for telling a change)."""
        from ..analysis import AnalysisConfig
        from ..config import load_project
        from ..project_edit import live_settings

        project = self._project()
        if project is None:
            path = getattr(self.window, "results_path", None)
            return AnalysisConfig(), Path(path).parent if path else Path("."), None
        config = load_project(project, analysis_only=True).analysis or AnalysisConfig()
        values, tables = live_settings(project.read_text(encoding="utf-8-sig"))
        text = ({k: v for k, v in values.items() if k.split(".")[0] == "analysis"},
                sorted(t for t in tables if t.split(".")[0] == "analysis"))
        return config, project.parent, repr(text)

    def why_out_of_date(self, made_from=None) -> str:
        """Why the analysis shown (or the one ``made_from`` describes) no longer matches the
        results and settings (empty: it does)."""
        made_from = made_from or self._made_from
        if made_from is None:
            return ""
        results, text, _, stamp = made_from
        if self.results is not results:
            return "the results changed (a new run or resume, or a review decision)"
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
        if self.results is None:
            self.status.setText("No results open: open a project file whose results exist, or "
                                "a results bundle.")
            self.compute_button.setEnabled(False)
            return
        self.compute_button.setEnabled(not self._computing.running)
        if self._computing.running:
            return
        why = self.why_out_of_date()
        if self.outputs is None:
            self.compute_button.setText("Compute")
            if self._error is None:
                ref = self.shown_reference
                shared = self.shared_reference
                stale = (self.why_out_of_date(self._analyses[shared][1])
                         if shared is not None else "")
                self.status.setText(
                    "Not computed yet" + (f" against {ref}" if ref else "") + ": Compute "
                    f"makes the figures hxex analyse would write for {len(self.results.fit):,} "
                    "spectra (about half a minute for 25,000; nothing is written)."
                    + (f" The figures the same for every reference are shown from the analysis "
                       f"against {shared}" + (f" (out of date: {stale})" if stale else "") + "."
                       if shared is not None else "")
                    + self._settings_note())
                self._redraw_if_another()
        elif why:
            self.status.setText(f"Out of date: {why}. The figures shown are the earlier "
                                "analysis's; Recompute to see the new one.")
            self.compute_button.setText("Recompute")
        else:
            n = len(self.outputs.figures)
            ref = self.shown_reference
            self.status.setText(f"{n} figure(s)" + (f" against {ref}" if ref else "")
                                + ", as hxex analyse would write them" + self._settings_note())
            self._redraw_if_another()
            self.compute_button.setText("Recompute")

    _error: str | None = None

    def _redraw_if_another(self) -> None:
        """Another reference chosen: the figure selected again, unless it is drawn from the same
        analysis (a figure the same for every reference stays as it is)."""
        item = self.list.currentItem()
        if item is None:
            return
        outputs = self.outputs_for(item.data(0, KIND))
        if outputs is not self._drawn_from:
            self._selected(item)

    def _settings_note(self) -> str:
        return ("" if self._project() is not None else
                " (a bundle opened alone: the default [analysis] settings; open the project "
                "file for its own)")

    def _fill(self) -> None:
        """The kinds of figure and, once computed, their figures."""
        from ..report.catalogue import ANALYSE, Situation, kind_of, plan

        current = self.list.currentItem()
        keep = (current.data(0, KIND), current.data(0, STEM)) if current is not None else None
        self.list.blockSignals(True)
        self.list.clear()
        self._planned = {}
        if self.results is not None:
            try:
                config = self._settings()[0]
                self._fill_references(config)
                if not config.single_reference:
                    config = config.with_reference(self.shown_reference)
                planned = plan(Situation.of(analysis=config, results=self.results))
            except Exception as exc:  # noqa: BLE001 - settings these results refuse
                self.list.blockSignals(False)
                self._error = str(exc)
                self.status.setText(f"The analysis cannot be made: {exc}")
                return
            self._error = None
            by_kind: dict[str, list[str]] = {}
            for outputs, compares in ((self.shared_outputs, False), (self.outputs, True)):
                for stem, _ in (outputs.figures if outputs is not None else []):
                    kind = kind_of(stem)
                    if kind is not None and kind.compares == compares:
                        by_kind.setdefault(kind.name, []).append(stem)
            kinds = [p for p in planned if p.kind.command == ANALYSE]
            shared = self.shared_reference
            sections = (
                ("Same for every reference",
                 "these figures do not compare samples with the reference: they are drawn once, "
                 "from the latest analysis computed (against "
                 + (shared if shared else "the reference computed") + "), and shown whichever "
                 "reference is chosen",
                 [p for p in kinds if not p.kind.compares]),
                ("Compared with the reference" + (f" {self.shown_reference}"
                                                  if self.shown_reference else ""),
                 "these figures compare each sample with the reference chosen above",
                 [p for p in kinds if p.kind.compares]))
            for heading, tip, group in sections:
                if not group:
                    continue
                head = QTreeWidgetItem([heading, ""])
                head.setFlags(Qt.ItemFlag.ItemIsEnabled)  # a heading: not selectable
                font = head.font(0)
                font.setBold(True)
                head.setFont(0, font)
                head.setToolTip(0, tip)
                head.setFirstColumnSpanned(True)
                self.list.addTopLevelItem(head)
                head.setFirstColumnSpanned(True)
                for p in group:
                    self._add_kind(p, by_kind.get(p.kind.name, []), keep)
        self.list.blockSignals(False)
        self._describe(self.list.currentItem())

    def _add_kind(self, p, stems: list[str], keep) -> None:
        """One kind of figure in the list, with its figures under it (``keep``: the kind and
        figure selected before, selected again)."""
        outputs = self.outputs_for(p.kind.name)
        self._planned[p.kind.name] = p
        item = QTreeWidgetItem([p.kind.title, self._state_text(p, stems)])
        item.setData(0, KIND, p.kind.name)
        item.setToolTip(0, p.kind.shows)
        if p.state == "off" or (outputs is not None and not stems):
            for col in (0, 1):
                item.setForeground(col, self.palette().placeholderText())
        for stem in stems:
            child = QTreeWidgetItem([stem, ""])
            child.setData(0, KIND, p.kind.name)
            child.setData(0, STEM, stem)
            item.addChild(child)
        self.list.addTopLevelItem(item)
        if keep is not None and keep[0] == p.kind.name:
            target = item
            for i in range(item.childCount()):
                if item.child(i).data(0, STEM) == keep[1]:
                    target = item.child(i)
            self.list.setCurrentItem(target)
            item.setExpanded(True)

    def _state_text(self, planned, stems) -> str:
        if planned.state == "off":
            return "off"
        if self.outputs_for(planned.kind.name) is None:
            return STATE_WORDS[planned.state]
        return f"{len(stems)} figure(s)" if stems else "none for these results"

    def _selected(self, item, _previous=None) -> None:
        self._describe(item)
        outputs = self.outputs_for(item.data(0, KIND)) if item is not None else None
        if item is None or item.data(0, KIND) is None or outputs is None:
            return
        stem = item.data(0, STEM)
        self._kind_rows = None
        if stem is None and item.childCount():  # a kind: its first figure, every figure's rows
            item.setExpanded(True)
            stem = item.child(0).data(0, STEM)
            stems = [item.child(i).data(0, STEM) for i in range(item.childCount())]
            self._gather_rows(item.data(0, KIND), stems, outputs)
        if stem is None:  # a kind with no figure: why, in the figure's place
            self.panel.save_name = ""
            self._clear_figure(self._off_text(self._planned.get(item.data(0, KIND)))
                             or "no figure of this kind for these results")
            return
        draw = dict(outputs.figures).get(stem)
        self._drawn_from = outputs
        if draw is None:  # a figure of an earlier analysis this one does not draw
            self.panel.save_name = ""
            self._clear_figure("this figure is not in the analysis computed now")
            return
        self.panel.save_name = stem
        self.panel.draw(draw)

    def _clear_figure(self, message: str = "") -> None:
        self.panel.clear(message)
        self.data.clear()
        self.views.setTabText(1, "data")

    def _show_data(self) -> None:
        """The figure drawn: its data table (a kind selected: every figure's, gathered apart)."""
        from .figures import figure_rows

        if self._kind_rows is not None:
            return
        columns, rows = figure_rows(self.panel.view.figure)
        self.data.show_rows(columns, rows, self.panel.save_name)
        self.data.set_note("")
        self.views.setTabText(1, f"data ({len(rows)})" if rows else "data")

    def _gather_rows(self, kind: str, stems: list[str], outputs) -> None:
        """A kind selected: the rows of every one of its figures in the data tab, a ``figure``
        column naming each one's file (owner, 2026-11-04: the first time point alone was
        confusing). Made in the drawing thread (matplotlib is used by one thread at a time)."""
        from .figures import figure_rows

        figures = dict(outputs.figures)
        self._kind_rows = kind
        self.data.show_rows([], [], kind)
        self.data.set_note(f"Gathering the rows of the {len(stems)} figures of this kind…")

        def work():
            columns, rows = ["figure"], []
            for stem in stems:
                draw = figures.get(stem)
                if draw is None:
                    continue
                names, found = figure_rows(draw())
                columns += [n for n in names if n not in columns]
                rows += [{"figure": stem, **r} for r in found]
            return columns, rows

        self._rows_task.submit(work, lambda made: self._rows_gathered(kind, made),
                               lambda message: self.data.set_note(
                                   f"The rows could not be gathered: {message}"))

    def _rows_gathered(self, kind: str, made) -> None:
        if self._kind_rows != kind:  # another selection since
            return
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
        if p is None:
            self.about.setText("")
            return
        lines = [f"{p.kind.title}: {p.kind.shows}."]
        if p.state == "off":
            lines.append(self._off_text(p))
            if p.missing:
                need = p.missing[0]
                self._change_table = setting_table(need.setting)
                self.change.setText(f"Change {need.setting}…")
                self.change.show()
        elif p.reason:
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
        if not config.single_reference:
            config = config.with_reference(ref)
        stamp = table_stamp(self._project())
        if self._project() is not None and self.window is not None \
                and stamp != getattr(self.window, "results_review_stamp", stamp):
            # the review table changed since the results were read: read them again first
            log.info("the review table changed: the results are read again before the analysis")
            self._compute_when_loaded = True
            self.status.setText("Reading the results again with the review table's changes…")
            self.busy.show()
            self.compute_button.setEnabled(False)
            self.window.open_results(self._project())
            return
        if self._cancel is not None:
            self._cancel.set()  # an older computation still running stops
        cancel = self._cancel = threading.Event()
        n = len(results.fit)
        path = getattr(self.window, "results_path", None)
        self.status.setText(f"Computing the analysis of {n:,} spectra"
                            + (f" against {ref}" if ref else "")
                            + " (about half a minute for 25,000 spectra)…")
        self.busy.show()
        self.compute_button.setEnabled(False)
        log.info("computing the analysis (as hxex analyse, nothing written)")
        self._computing.submit(
            lambda: analysis_outputs(results, config, base=base, cancel=cancel.is_set),
            lambda outputs: self._computed(outputs, results, text, path, ref, stamp),
            self._failed)

    @property
    def computing(self) -> bool:
        return self._computing.running

    def _computed(self, outputs, results, text, path, ref, stamp) -> None:
        self.busy.hide()
        self._analyses[ref] = (outputs, (results, text, path, stamp))
        self._latest = ref  # the figures the same for every reference come from it now
        log.info("analysis: %d tables and %d figures", len(outputs.tables), len(outputs.figures))
        self.refresh()
        if self.list.currentItem() is not None:  # the figure shown, from the new analysis
            self._selected(self.list.currentItem())
        else:  # the first figure drawn
            for i in range(self.list.topLevelItemCount()):
                item = self.list.topLevelItem(i)
                if item.childCount():
                    item.setExpanded(True)
                    self.list.setCurrentItem(item.child(0))
                    break

    def _failed(self, message: str) -> None:
        self._compute_when_loaded = False
        self.busy.hide()
        self.compute_button.setEnabled(True)
        self._error = message
        self.status.setText(f"The analysis could not be made: {message}")
        log.error("the analysis could not be made: %s", message)

    # --- the window --------------------------------------------------------------------------

    def results_changed(self) -> None:
        """Other results are shown: an analysis of other results altogether (another bundle or
        project) is dropped; one of the same results, made again, is marked out of date."""
        path = getattr(self.window, "results_path", None)
        made = {entry[1][2] for entry in self._analyses.values()}
        if self.results is None or (made and path not in made):
            if self._cancel is not None:
                self._cancel.set()
            self._analyses.clear()
            self._clear_figure()
        self._error = None
        if not self.computing:
            self.busy.hide()
        self.refresh()
        if self._compute_when_loaded:  # Recompute asked for the results read again
            self._compute_when_loaded = False
            if self.results is not None:
                self.compute()

    def load_failed(self) -> None:
        """The results could not be read again (for a Recompute): nothing is computed."""
        if self._compute_when_loaded:
            self._compute_when_loaded = False
            self.busy.hide()
            self.refresh()

    def set_render(self, render: str) -> None:
        self.render = render
        self.panel.set_render(render)

    def showEvent(self, event):  # noqa: N802 - Qt's name
        super().showEvent(event)
        self.refresh()
