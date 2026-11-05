# SPDX-License-Identifier: GPL-3.0-or-later
"""Every kind of figure hxex draws, in one list (docs/PLAN.md, "GUI", the figure catalogue):
the command that draws it, its file names, what it shows, the settings it needs and the
settings that change it, and the guide page that explains it.

:func:`plan` says, for a project's settings (and, when given, its results), which kinds are
drawn, which are left to the data, and which a setting rules out, naming that setting. The user
guide's figure page (``docs/guide/figures.md``, from :mod:`hxex._reference`), ``hxex figures``
and the window's Analysis list are made from it. :func:`kind_of` names the kind of a written
file; a test ties every file ``hxex report --figures`` and ``hxex analyse`` write to its entry
here, so the list cannot fall behind the drawing code.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from dataclasses import dataclass, field

REPORT = "report --figures"
ANALYSE = "analyse"

_RUN_PAGE = "outputs.md#figures-from-hxex-report---figures"
_ANALYSIS_PAGE = "outputs.md#figures-from-hxex-analyse"


@dataclass(frozen=True)
class Situation:
    """What decides which figures are drawn: the ``[figures]`` and ``[analysis]`` settings
    (their classes' defaults when the project has none), and what is known of the results (None
    where the results were not read)."""

    figures: object  # hxex.report.figures.PlotConfig
    analysis: object  # hxex.analysis.AnalysisConfig
    ks: tuple[int, ...]  # the K values the analysis uses (cut to those fitted, when known)
    populations: str | None  # the analysis's populations rule (None: not known without results)
    fragments: bool | None = None  # whether the results hold fragment series
    peptides: bool | None = None  # whether they hold peptide series (not only fragments)

    @classmethod
    def of(cls, figures=None, analysis=None, results=None) -> Situation:
        """The situation of these settings and, when given, these results (a
        :class:`hxex.report.Results` or a :class:`hxex.analysis.RunSet`). Settings the results
        refuse, as ``hxex analyse`` would (a reference naming no sample, a K not fitted), raise
        ValueError."""
        from ..analysis import AnalysisConfig
        from .figures import PlotConfig

        figures = figures if figures is not None else PlotConfig()
        analysis = analysis if analysis is not None else AnalysisConfig()
        if results is None:
            return cls(figures, analysis, tuple(analysis.ks), analysis.populations)
        from ..analysis import RunSet
        from ..analysis.settings import population_rule
        from .analysis import _check_names, _fitted_ks, _fragment_series

        _check_names(results, analysis)  # a reference that names no sample: refused
        runs = results.runs if isinstance(results, RunSet) else {"": results}
        fitted = _fitted_ks(analysis, runs)
        split = [_fragment_series(r) for r in runs.values()]
        return cls(figures, analysis, tuple(fitted.ks),
                   population_rule(analysis.populations, results),
                   any(series for series, _ in split), not all(only for _, only in split))


@dataclass(frozen=True)
class Need:
    """A setting a kind of figure needs: ``setting`` as the guide names it, ``words`` what it
    must be, and ``met``: True or False, or None when only the results can tell."""

    setting: str
    words: str
    met: Callable[[Situation], bool | None]


@dataclass(frozen=True)
class FigureKind:
    """One kind of figure. ``files`` are its file names (without the extension), ``{...}``
    standing for what varies; ``data`` says in words when the results decide too (empty: the
    settings alone decide); ``changed_by`` names the settings that change it."""

    name: str
    title: str
    command: str
    files: tuple[str, ...]
    shows: str
    guide: str
    needs: tuple[Need, ...] = ()
    data: str = ""
    changed_by: tuple[str, ...] = ()
    peptides: bool = False
    _patterns: tuple[re.Pattern, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_patterns", tuple(
            re.compile("".join(".+" if part.startswith("{") else re.escape(part)
                               for part in re.split(r"(\{[^}]+\})", f)) + r"\Z")
            for f in self.files))

    @property
    def compares(self) -> bool:
        """Whether its figures compare the samples with the reference (they change with
        ``[analysis] reference``); the others are the same against every reference, but for
        the reference drawn first, in black."""
        return _REFERENCE in self.needs

    def matches(self, stem: str) -> int:
        """How much of ``stem`` is fixed text of one of this kind's file names (0: no match);
        the longest wins, so ``woods_K2_...`` is not taken for ``woods_...``."""
        best = 0
        for f, pattern in zip(self.files, self._patterns, strict=True):
            if pattern.match(stem):
                best = max(best, len(re.sub(r"\{[^}]+\}", "", f)))
        return best


@dataclass(frozen=True)
class Planned:
    """A kind of figure under a :class:`Situation`: ``state`` ``'drawn'`` (the settings draw
    it), ``'depends'`` (the settings allow it; the results decide, see ``reason``) or ``'off'``
    (``missing`` names the settings that rule it out; or ``reason`` says what the results
    lack)."""

    kind: FigureKind
    state: str
    missing: tuple[Need, ...] = ()
    reason: str = ""


# --- the needs ---------------------------------------------------------------------------------

def _kinds_include(page: str) -> Need:
    return Need("[figures] kinds", f"includes \"{page}\"",
                lambda s: page in s.figures.kinds)


_OVERVIEW = Need("[figures] overview", "true", lambda s: bool(s.figures.overview))
_REFERENCE = Need("[analysis] reference", "names a sample to compare the others with",
                  lambda s: bool(s.analysis.reference))
_K2 = Need("[analysis] ks", "includes 2 (and the run fitted K = 2)", lambda s: 2 in s.ks)
_TEST = Need("[analysis] populations", "\"population_test\" (the default when the run has "
             "[run.population_test])",
             lambda s: None if s.populations is None else s.populations == "population_test")
_REGIONS = Need("[[analysis.regions]]", "at least one region",
                lambda s: bool(s.analysis.annotations.regions))
_PROTECTION = Need("[analysis] protection", "true (with [analysis.labelling])",
                   lambda s: bool(s.analysis.protection))
_FRAGMENTS = Need("[analysis.fragments]", "not false",
                  lambda s: s.analysis.fragments is not None)

_SERIES_DATA = "for the series [figures] series chooses (\"flagged\" by default)"


KINDS: tuple[FigureKind, ...] = (
    # hxex report --figures: [figures]
    FigureKind(
        "overview", "Overview", REPORT, ("overview", "overview_{page}"),
        "which K was selected and how often, the controls, and the quality flags of the run",
        _RUN_PAGE, (_OVERVIEW,),
        data="the controls and quality pages only when the results have controls and flags"),
    FigureKind(
        "series_spectra", "Spectra of a series", REPORT,
        ("{series}", "{series}_spectra", "{series}_spectra_{page}"),
        "raw data, picked values and the fit of every spectrum of a series, with the picking "
        "window shaded, on pages of [figures] page_rows by page_columns (more time points than "
        "max_points thinned)", _RUN_PAGE, (_kinds_include("spectra"),), data=_SERIES_DATA,
        changed_by=("[figures] series", "[figures] show", "[figures] residual",
                    "[figures] k", "[figures] corrected", "[figures] page_rows",
                    "[figures] page_columns", "[figures] max_points")),
    FigureKind(
        "series_populations", "Populations of a series", REPORT,
        ("{series}", "{series}_populations", "{series}_populations_{page}"),
        "each population's mean and share, one panel per spectrum, and the replicates overlaid, "
        "on the same pages as the series' spectra",
        _RUN_PAGE, (_kinds_include("populations"),),
        data=_SERIES_DATA + "; left out when every spectrum has one population",
        changed_by=("[figures] series", "[figures] k", "[figures] corrected",
                    "[figures] page_rows", "[figures] page_columns", "[figures] max_points")),
    FigureKind(
        "series_uptake", "Uptake of a series", REPORT, ("{series}", "{series}_uptake"),
        "deuteration against time, a marker per population sized by its share, the controls "
        "at the ends of the axis", _RUN_PAGE, (_kinds_include("uptake"),), data=_SERIES_DATA,
        changed_by=("[figures] series", "[figures] x", "[figures] scale", "[figures] k",
                    "[figures] corrected")),
    # hxex analyse: [analysis]
    FigureKind(
        "coverage", "Coverage", ANALYSE, ("coverage_{sample}",),
        "one bar per peptide, coloured by fD, for every time and rank", _ANALYSIS_PAGE,
        changed_by=("[analysis.filters]", "[analysis] page_size")),
    FigureKind(
        "redundancy_resolution", "Redundancy and resolution", ANALYSE,
        ("redundancy_resolution",), "redundancy and resolution per residue", _ANALYSIS_PAGE,
        changed_by=("[analysis] redundancy", "[analysis] coverage_times",
                    "[analysis.colours]")),
    FigureKind(
        "residues_all", "Residue fD, every sample", ANALYSE, ("residues_all_{time}",),
        "every sample's average deuteration (fDavg: the K=1 fit, or the K=2 average where two "
        "populations lie at least 2 D apart) by residue on one axis, none of them a reference, "
        "with its standard error", _ANALYSIS_PAGE,
        data="at the times where a sample has residue values",
        changed_by=("[analysis] samples", "[analysis.colours]", "[analysis] weight_exponent",
                    "[analysis] times")),
    FigureKind(
        "choice", "Residue profile with the population choice", ANALYSE,
        ("choice_{sample}_{time}",),
        "the residue profile: K=1, the K=2 ranks, the chosen description and the population",
        _ANALYSIS_PAGE, (_K2,), data="where the sample has K=2 values at that time",
        changed_by=("[analysis] populations", "[analysis.choice]", "[analysis] weight_exponent")),
    FigureKind(
        "profile", "Residue profile", ANALYSE, ("profile_{sample}_{time}",),
        "the residue profile at K=1, where there is no population choice to draw",
        _ANALYSIS_PAGE, data="where no population choice is drawn for that sample and time "
                             "(no K=2 values there, or [analysis] ks without 2)",
        changed_by=("[analysis] weight_exponent",)),
    FigureKind(
        "population_tiers", "Population tiers", ANALYSE,
        ("population_tiers", "population_tiers_{page}"),
        "the residue tiers of the population test, one row per sample and time point",
        "population-test.md#the-residue-tiers", (_TEST,),
        data="where a residue has a tier", changed_by=("[analysis] page_size",)),
    FigureKind(
        "overlay_residues", "Residue overlay", ANALYSE, ("overlay_residues_{time}",),
        "each sample's residue values drawn over the reference's (values only, no "
        "significance)", _ANALYSIS_PAGE, (_REFERENCE,),
        data="where the reference and a sample have the values to draw at that time",
        changed_by=("[analysis] overlay", "[analysis] overlay_two_only",
                    "[analysis.colours]", "[analysis] page_size")),
    FigureKind(
        "overlay_populations", "Population overlay", ANALYSE, ("overlay_populations_{time}",),
        "the share of the high-D population by residue, each sample over the reference",
        _ANALYSIS_PAGE, (_REFERENCE, _K2),
        data="where a sample has two populations at that time",
        changed_by=("[analysis] populations", "[analysis.choice]", "[analysis.colours]",
                    "[analysis] page_size")),
    FigureKind(
        "woods", "Woods plot", ANALYSE, ("woods_{sample}_{time}",),
        "peptide differences from the reference at K=1, and where the number of populations "
        "differs", _ANALYSIS_PAGE, (_REFERENCE,),
        changed_by=("[analysis] alpha", "[analysis] peptide_match", "[analysis.filters]",
                    "[analysis] populations")),
    FigureKind(
        "woods_K2", "Woods plot at K=2", ANALYSE, ("woods_K2_{sample}_{time}",),
        "the two populations compared with the reference's, where both samples have two",
        _ANALYSIS_PAGE, (_REFERENCE, _K2), data="where both samples have two populations",
        changed_by=("[analysis] alpha", "[analysis] peptide_match", "[analysis] populations")),
    FigureKind(
        "residue_significance", "Residue significance map", ANALYSE,
        ("residue_significance", "residue_significance_{page}"),
        "which residues differ significantly from the reference: one row per sample and "
        "time, one cell per residue", _ANALYSIS_PAGE, (_REFERENCE,),
        data="where a residue could be compared",
        changed_by=("[analysis] alpha", "[analysis] peptide_match", "[analysis] page_size")),
    FigureKind(
        "residue_significance_K2", "Residue significance map at K=2", ANALYSE,
        ("residue_significance_K2_{sample}",),
        "one figure per sample: where the number of populations differs from the reference's, "
        "and, with a bootstrap at K=2, the differences in the share and in the two "
        "populations' positions",
        _ANALYSIS_PAGE, (_REFERENCE, _K2),
        data="where a sample has two populations; the share and position sections need a "
             "bootstrap at K=2 ([run.bootstrap] ks)",
        changed_by=("[analysis] populations", "[analysis.choice]", "[analysis] page_size")),
    FigureKind(
        "region_significance", "Region significance map", ANALYSE,
        ("region_significance", "region_significance_{page}"),
        "which regions differ significantly from the reference: one row per sample and time, "
        "one column per region", _ANALYSIS_PAGE, (_REFERENCE, _REGIONS),
        changed_by=("[analysis] alpha", "[[analysis.regions]]", "[analysis] page_size")),
    FigureKind(
        "butterfly", "Butterfly plot", ANALYSE, ("butterfly_{sample}",),
        "the reference above, the sample below, a line per time, the differences underneath",
        _ANALYSIS_PAGE, (_REFERENCE,),
        changed_by=("[analysis] alpha", "[analysis] peptide_match")),
    FigureKind(
        "regions", "Regions", ANALYSE, ("regions_{time}",), "each region's fD, per sample",
        _ANALYSIS_PAGE, (_REGIONS,), changed_by=("[[analysis.regions]]", "[analysis.colours]")),
    FigureKind(
        "protection", "Protection factors", ANALYSE, ("protection_{sample}",),
        "the peptides' protection factors", "protection-factors.md", (_PROTECTION,),
        changed_by=("[analysis.labelling]",)),
    FigureKind(
        "protection_difference", "Protection factor differences", ANALYSE,
        ("protection_difference_{sample}",),
        "the peptides' protection factors compared with the reference's",
        "protection-factors.md", (_PROTECTION, _REFERENCE),
        changed_by=("[analysis.labelling]", "[analysis] alpha")),
    FigureKind(
        "ladder", "Fragment ladder", ANALYSE, ("ladder_{parent}_{ion}_z{charge}",),
        "the precursor, then for every ladder step the two fragments and their difference "
        "with its fit", "fragment-residues.md", (_FRAGMENTS,),
        changed_by=("[analysis.fragments]",)),
    FigureKind(
        "closure", "Ladder closure", ANALYSE, ("closure_{parent}_{ions}",),
        "the closure miss over time for each split of the parent into an N- and a C-terminal "
        "fragment", "fragment-residues.md", (_FRAGMENTS,),
        data="where both fragments of a split were measured",
        changed_by=("[analysis.fragments]",)),
    FigureKind(
        "residue_rates", "Residue rates", ANALYSE, ("residue_rates_{parent}",),
        "per-residue rates, amplitudes and, with kint, protection, from the fragment ladders",
        "fragment-residues.md", (_FRAGMENTS,),
        changed_by=("[analysis.fragments]", "[analysis.fragments.exchange]")),
)


KINDS = tuple(dataclasses.replace(k, peptides=True)
              if k.command == ANALYSE and _FRAGMENTS not in k.needs else k for k in KINDS)


def kind_named(name: str) -> FigureKind:
    """The kind of figure called ``name``."""
    for kind in KINDS:
        if kind.name == name:
            return kind
    raise KeyError(f"no figure kind {name!r}")


def kind_of(stem: str, command: str = ANALYSE) -> FigureKind | None:
    """The kind of figure a file written by ``command`` is (``stem``: its name without the
    extension); None when none matches. A series' file of ``hxex report --figures`` is named
    after the series, so only its page suffix (``_spectra``, ...) can tell its kind."""
    best, found = 0, None
    for kind in KINDS:
        if kind.command != command:
            continue
        n = kind.matches(stem)
        if n > best:
            best, found = n, kind
    return found


def plan(situation: Situation) -> list[Planned]:
    """Every kind of figure under ``situation``, in the order of :data:`KINDS`."""
    out = []
    for kind in KINDS:
        results = [(need, need.met(situation)) for need in kind.needs]
        missing = tuple(need for need, met in results if met is False)
        unknown = [need for need, met in results if met is None]
        fragments = _FRAGMENTS in kind.needs
        if missing:
            out.append(Planned(kind, "off", missing))
        elif kind.peptides and situation.peptides is False:
            out.append(Planned(kind, "off", reason="the results hold only fragment series"))
        elif fragments and situation.fragments is False:
            out.append(Planned(kind, "off", reason="the results hold no fragment series"))
        elif unknown or kind.data or (fragments and situation.fragments is None):
            reason = "; ".join([kind.data] * bool(kind.data) + [
                f"{need.setting}: {need.words}" for need in unknown]
                + ["where the results hold fragment series"]
                * (fragments and situation.fragments is None))
            out.append(Planned(kind, "depends", reason=reason))
        else:
            out.append(Planned(kind, "drawn"))
    return out
