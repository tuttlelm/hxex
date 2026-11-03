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

from .figure_view import FigurePanel
from .results_tree import FILTERS, Peptide, peptides, points, shown

log = logging.getLogger(__name__)

COLUMNS = ("peptide, charge, time, replicate", "sample", "spectra", "flagged", "rejected",
           "K > 1", "K", "review", "flags")
KIND, PAYLOAD = Qt.ItemDataRole.UserRole, Qt.ItemDataRole.UserRole + 1
# the figure tabs of each kind of selection, in their order
TABS = {"series": ("spectra", "uptake", "populations"),
        "point": ("spectra", "uptake"),
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
        left = QWidget()
        side = QVBoxLayout(left)
        side.setContentsMargins(0, 0, 0, 0)
        side.addLayout(filters)
        side.addWidget(self.tree, 1)

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
        self.path_label.setText("No results open." if results is None else str(path))
        self._fill()

    def _fill(self, *_) -> None:
        how = self.filter.currentData() or "all"
        text = self.search.text()
        self.tree.blockSignals(True)
        self.tree.clear()
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
            item.addChild(child)

    # --- selection -------------------------------------------------------------------------

    def _selected(self, item: QTreeWidgetItem | None, _previous=None) -> None:
        from ..report.figures import PlotConfig, series_figures, series_kinds
        from ..report.overview import plot_k_comparison
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
        results = self.results
        config = results.figures or PlotConfig()
        series = payload[1]
        data_id = series.data_id
        self._show_tabs(kind)

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
