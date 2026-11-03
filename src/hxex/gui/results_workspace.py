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
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
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

from .figure_view import FigurePanel
from .results_tree import FILTERS, Peptide, peptides, points, shown
from .review_panel import ReviewPanel

log = logging.getLogger(__name__)

TIME_AXES = {"log": "log time axis", "linear": "linear time axis",
             "index": "time points evenly spaced"}
COLUMNS = ("peptide, charge, time, replicate", "sample", "spectra", "flagged", "rejected",
           "K > 1", "K", "review", "flags")
KIND, PAYLOAD = Qt.ItemDataRole.UserRole, Qt.ItemDataRole.UserRole + 1
DEPTH = {"peptide": 0, "series": 1, "point": 2, "spectrum": 3}  # a kind of row's level
# the figure tabs of each kind of selection, in their order
TABS = {"series": ("spectra", "uptake", "populations"),
        "point": ("spectra", "fits at each K", "uptake"),
        "spectrum": ("spectrum", "fits at each K", "uptake")}


class ResultsWorkspace(QWidget):
    """The results of a run as a tree, and the figures of the selection."""

    selection_changed = Signal()

    def __init__(self, window=None):
        super().__init__()
        self.window = window
        self.results = None
        self.tree_data: list[Peptide] = []
        self.path_label = QLabel("No results open.")
        open_button = QPushButton("Open results…")
        open_button.setToolTip("a results bundle: a folder, or a .zip")
        if window is not None:
            open_button.clicked.connect(window.open_results_dialog)
        top = QHBoxLayout()
        top.addWidget(self.path_label, 1)
        top.addWidget(open_button)

        self.filter = QComboBox()
        for i, (key, (text, meaning)) in enumerate(FILTERS.items()):
            self.filter.addItem(text, key)
            self.filter.setItemData(i, meaning, Qt.ItemDataRole.ToolTipRole)
        self.filter.setToolTip("which series the tree shows")
        self.filter.currentIndexChanged.connect(self._fill)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a peptide, residues or sample")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._fill)
        self.count = QLabel()
        filters = QHBoxLayout()
        filters.addWidget(self.filter)
        filters.addWidget(self.search, 1)
        filters.addWidget(self.count)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(COLUMNS))
        self.tree.setHeaderLabels(list(COLUMNS))
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemExpanded.connect(self._expanded)
        self.tree.currentItemChanged.connect(self._selected)
        self.review = ReviewPanel(self)
        self.review.decided.connect(self._reviewed)
        self.review.step.connect(self.step_spectrum)
        if window is not None:
            self.review.resume_asked.connect(window.run_resume)
        # review from the tree: A accepted, R rejected, U not reviewed (the tree's own keys, so
        # they work whenever it has the keyboard)
        self.tree.installEventFilter(self)
        self._leaves: dict[int, QTreeWidgetItem] = {}  # a spectrum's row in the tree
        self._series_items: dict[int, QTreeWidgetItem] = {}
        left = QWidget()
        side = QVBoxLayout(left)
        side.setContentsMargins(0, 0, 0, 0)
        side.addLayout(filters)
        side.addWidget(self.tree, 1)
        side.addWidget(self.review)

        # the uptake page's time axis, for the window only (owner: the project's linear default
        # is not how uptake is usually read); the project's [figures] setting first
        self.time_axis = QComboBox()
        self.time_axis.setToolTip("how the uptake page spreads the times: the project file's "
                                  "[figures] scale and x, or for this window only log, linear, or "
                                  "one even step per time point")
        self.time_axis.currentIndexChanged.connect(self._redraw)
        self.figures = QTabWidget()
        self.panels: dict[str, FigurePanel] = {}
        self.render = getattr(window, "render", "matplotlib")
        self.kind = None
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(self.figures)
        split.setStretchFactor(1, 1)
        split.setSizes([560, 900])
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(split, 1)
        self._show_tabs("series")

    # --- the figures' tabs -----------------------------------------------------------------

    def panel(self, name: str) -> FigurePanel:
        if name not in self.panels:
            panel = FigurePanel(render=self.render)
            if name == "uptake":  # the only page with a time axis: its choice in its own bar
                panel.add_option(self.time_axis)
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
        current = self.figures.tabText(self.figures.currentIndex()) \
            if self.figures.count() else None
        self.figures.blockSignals(True)
        while self.figures.count():
            self.figures.removeTab(0)
        for name in TABS[kind]:
            self.figures.addTab(self.panel(name), name)
        if current in TABS[kind]:
            self.figures.setCurrentIndex(TABS[kind].index(current))
        self.figures.blockSignals(False)
        self.kind = kind

    # --- the tree --------------------------------------------------------------------------

    def show_results(self, results, path: Path | None) -> None:
        self.results = results
        for panel in self.panels.values():
            panel.clear()
        self.tree_data = [] if results is None else peptides(results)
        self._fill_time_axis()
        self.review.show_spectra([], "Select a spectrum, a time point or a series.")
        self.path_label.setText("No results open." if results is None else str(path))
        self._fill()

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

    def _fill(self, *_) -> None:
        how = self.filter.currentData() or "all"
        text = self.search.text()
        self.tree.blockSignals(True)
        self.tree.clear()
        self._leaves, self._series_items = {}, {}
        n_series = 0
        for pep in self.tree_data:
            series = [s for s in pep.series if shown(pep, s, how, text)]
            if not series:
                continue
            counts = {n: sum(getattr(s, n) for s in series)
                      for n in ("n_spectra", "n_flagged", "n_rejected", "n_multi")}
            top = QTreeWidgetItem([f"{pep.peptide_range} {pep.peptide}", pep.sample,
                                   *_counts(counts), "", "", ""])
            top.setData(0, KIND, "peptide")
            top.setData(0, PAYLOAD, pep)
            top.setToolTip(0, f"{pep.sample} {pep.peptide_range} {pep.peptide}")
            charges = [s.charge for s in series]
            for s in series:
                name = f"z={s.charge}" + (f" (series {s.data_id})" if charges.count(s.charge) > 1
                                          else "")
                child = QTreeWidgetItem([name, "", *_counts(dataclasses.asdict(s)), "", "", ""])
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
        self.tree.setColumnWidth(0, max(self.tree.columnWidth(0), 260))
        total = sum(len(p.series) for p in self.tree_data)
        self.count.setText(f"{n_series} of {total} series" if self.results is not None else "")

    def _expanded(self, item: QTreeWidgetItem) -> None:
        """A series opened: its time points and their spectra, made now."""
        if item.data(0, KIND) != "series" or item.childCount():
            return
        pep, series = item.data(0, PAYLOAD)
        for point in points(self.results, series.data_id):
            flagged = sum(bool(s.flags) for s in point.spectra)
            child = QTreeWidgetItem([point.label, "", str(len(point.spectra)),
                                     str(flagged) if flagged else "", "", "", point.ks, "", ""])
            child.setData(0, KIND, "point")
            child.setData(0, PAYLOAD, (pep, series, point))
            for spectrum in point.spectra:
                leaf = QTreeWidgetItem([f"rep {spectrum.rep}", "", "", "", "", "",
                                        str(spectrum.K), spectrum.status,
                                        spectrum.flags.replace(";", ", ")])
                leaf.setData(0, KIND, "spectrum")
                leaf.setData(0, PAYLOAD, (pep, series, point, spectrum))
                leaf.setToolTip(0, spectrum.spectrum_id)
                child.addChild(leaf)
                self._leaves[spectrum.index] = leaf
            item.addChild(child)

    # --- selection -------------------------------------------------------------------------

    def _selected(self, item: QTreeWidgetItem | None, _previous=None) -> None:
        from ..report.figures import series_figures, series_kinds
        from ..report.overview import plot_k_comparison, plot_k_comparisons
        from ..report.series import plot_series_spectra
        from ..report.spectrum import plot_spectrum

        if item is None or self.results is None:
            return
        kind = item.data(0, KIND)
        payload = item.data(0, PAYLOAD)
        if kind == "peptide":  # its first charge's pages
            pep = payload
            series = next((s for s in pep.series if shown(pep, s, self.filter.currentData()
                                                          or "all", self.search.text())),
                          pep.series[0])
            kind, payload = "series", (pep, series)
        if kind == "point" and len(payload[2].spectra) == 1:
            # one replicate: its own view, so stepping into it draws nothing new (owner)
            kind, payload = "spectrum", (*payload, payload[2].spectra[0])
        results = self.results
        config = self._plot_config()
        series = payload[1]
        data_id = series.data_id
        self._show_tabs(kind)
        self._show_review(kind, payload)

        def uptake():
            one = dataclasses.replace(config, kinds=("uptake",))
            return series_figures(results, data_id, one)["uptake"]

        if kind == "series":
            drawn = series_kinds(results, data_id,
                                 dataclasses.replace(config, kinds=TABS["series"]))
            for name in TABS["series"]:
                panel = self.panel(name)
                if name not in drawn:
                    panel.clear("one population in every spectrum: no population map")
                    continue
                one = dataclasses.replace(config, kinds=(name,))
                panel.draw(lambda one=one, name=name: series_figures(results, data_id,
                                                                     one)[name])
        elif kind == "point":
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
                    f"z={series.charge}, {point.label}"
            self.panel("fits at each K").draw(lambda: plot_k_comparisons(
                ids, results, labels=labels, title=title))
            self.panel("uptake").draw(uptake)
        else:
            spectrum = payload[3]
            sid = spectrum.spectrum_id
            self.panel("spectrum").draw(lambda: plot_spectrum(
                sid, results, show=config.show, k=config.k, corrected=config.corrected))
            self.panel("fits at each K").draw(lambda: plot_k_comparison(sid, results))
            self.panel("uptake").draw(uptake)
        self.selection_changed.emit()
        if self.window is not None:
            self.window.changed()

    # --- stepping through the spectra ---------------------------------------------------------

    def step_spectrum(self, step: int, level: str | None = None) -> bool:
        """Select the next (``step`` 1) or previous (-1) row of ``level`` (``'spectrum'``,
        ``'point'``, ``'series'`` or ``'peptide'``; default: the review panel's choice) in the
        tree's order, opening series on the way. From a row below ``level``, the next after the
        row's own ``level`` row; from a row above it, the first (or last) one inside it. False
        at either end."""
        level = level or self.review.level.currentData() or "spectrum"
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

    def eventFilter(self, watched, event):  # noqa: N802 - Qt's name
        if watched is self.tree and event.type() == QEvent.Type.KeyPress \
                and event.modifiers() in (Qt.KeyboardModifier.NoModifier,
                                          Qt.KeyboardModifier.ShiftModifier) \
                and event.text().lower() in ("a", "r", "u"):
            self.review.key(event.text())
            return True
        return super().eventFilter(watched, event)

    def _show_review(self, kind: str, payload) -> None:
        pep, series = payload[0], payload[1]
        name = f"{pep.sample} {pep.peptide_range} {pep.peptide} z={series.charge}"
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
        for i in changed:
            leaf = self._leaves.get(int(i))
            if leaf is not None:
                leaf.setText(6, str(int(ks[i])))
                leaf.setText(7, "" if status[i] in ("unreviewed", "None") else status[i])
                point = leaf.parent()
                point.setText(6, " ".join(point.child(j).text(6)
                                          for j in range(point.childCount())))
        for data_id in series_ids:
            item = self._series_items.get(data_id)
            if item is None:
                continue
            rejected, multi = counts(results, data_id)
            item.setText(4, str(rejected or ""))
            item.setText(5, str(multi or ""))
            top = item.parent()
            for col in (4, 5):
                total = sum(int(top.child(j).text(col) or 0) for j in range(top.childCount()))
                top.setText(col, str(total or ""))
        self._redraw()

    def selection(self) -> dict:
        """What is selected, by what outlasts a rerun: sample, residues and charge, and the
        time point's label or the spectrum's id."""
        item = self.tree.currentItem()
        if item is None:
            return {}
        kind, payload = item.data(0, KIND), item.data(0, PAYLOAD)
        if kind == "peptide":
            return {"sample": payload.sample, "peptide_range": payload.peptide_range}
        pep, series = payload[0], payload[1]
        out = {"sample": pep.sample, "peptide_range": pep.peptide_range,
               "charge": series.charge}
        if kind == "point":
            out["point"] = payload[2].label
        elif kind == "spectrum":
            out["spectrum_id"] = payload[3].spectrum_id
        return out

    def view_settings(self) -> dict:
        """How the figures are shown, for the session (only what differs from the project)."""
        key = self.time_axis.currentData()
        return {"time_axis": key} if key else {}

    def select(self, selection: dict) -> bool:
        """Select what ``selection`` names (as :meth:`selection` gives it); False when the
        results do not have it (the filter and search are cleared to find it)."""
        if not selection:
            return False
        if self.filter.currentIndex() or self.search.text():
            self.filter.setCurrentIndex(0)
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
                if child.data(0, PAYLOAD)[1].charge != selection["charge"]:
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


def _counts(c: dict) -> list[str]:
    return [str(c["n_spectra"]), str(c["n_flagged"] or ""), str(c["n_rejected"] or ""),
            str(c["n_multi"] or "")]


__all__ = ["ResultsWorkspace"]
