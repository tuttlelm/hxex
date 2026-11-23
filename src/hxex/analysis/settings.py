# SPDX-License-Identifier: GPL-3.0-or-later
"""The settings of a post-fitting analysis (the ``[analysis]`` section of a project file; see
:mod:`hxex.config` and :func:`hxex.report.write_analysis`)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from ..io.times import parse_time
from .compare import MATCHES, Annotations
from .fragments import FragmentAnalysis
from .populations import ChoiceConfig
from .residues import COUNTS, MODIFIED
from .runs import SEPARATOR
from .species import SpeciesFilters

POPULATION_RULES = ("population_test", "best_set", "bootstrap")  # how the populations are decided


def population_rule(populations: str | None, results) -> str:
    """The ``populations`` rule an analysis of ``results`` uses: the one set, else
    ``'population_test'`` when every run has the population test (:mod:`hxex.population_test`)
    and ``'best_set'`` when not. ``'population_test'`` on results without the test is refused
    with the fix."""
    from .set_test import has_test

    if populations is None:
        return "population_test" if has_test(results) else "best_set"
    if populations == "population_test" and not has_test(results):
        raise ValueError("[analysis] populations = \"population_test\" needs the population "
                         "test in the run: add [run.population_test] to the project file and "
                         "run again (hxex run), or use populations = \"best_set\"")
    return populations


def choice_defaults(populations: str = "best_set") -> ChoiceConfig:
    """The ``[analysis.choice]`` defaults of a ``populations`` rule (owner, 2026-10-22, HISTORY
    142): with ``'bootstrap'`` the separation and population tests are the rule itself, so
    they are strict (confidence 0.95, populations of at least 10 %: the calibration found the
    0.5 / 5 % limits calling two populations in most single ones); with ``'best_set'`` the
    comparison of the fits decides and the tests only check its result (0.5 / 5 %, the
    minimum population also that of ``best_set_K``'s own check)."""
    if populations == "bootstrap":
        return ChoiceConfig(confidence=0.95, min_population=0.10)
    return ChoiceConfig()

FORMATS = ("pdf", "png", "svg")
UNDEUTERATED = "undeuterated"
# the values a by-residue comparison can show: name -> (K, rank)
# the values the by-residue comparison can draw: K and rank (the populations from the choice)
OVERLAY_VALUES = {"low": (2, 1), "high": (2, 2), "avg": (1, 1), "pop_low": (2, 1),
                  "pop_high": (2, 2)}


class NumberedSequence(str):
    """A protein sequence whose first residue is residue ``first`` (a FASTA numbered by its
    header, ``>NAME/START-END``), not residue 1."""

    first: int

    def __new__(cls, sequence: str, first: int):
        obj = super().__new__(cls, sequence)
        obj.first = int(first)
        return obj


DEFAULT_OVERLAY = ("low", "high", "avg")  # without K = 2 in ks: ("avg",)


ALL_REFERENCES = "all"  # [analysis] reference = "all": every sample in turn


def _same_sample(a: str, b: str) -> bool:
    """Whether two sample names can name the same sample: equal, or one is ``run:sample`` and
    the other that sample's bare name (which the results resolve where it is unambiguous)."""
    from .runs import SEPARATOR

    def bare(name: str) -> str:
        return name.partition(SEPARATOR)[2] if SEPARATOR in name else name

    return a == b or (SEPARATOR in a) != (SEPARATOR in b) and bare(a) == bare(b)


@dataclass(frozen=True)
class AnalysisGroup:
    """One group of ``[analysis.groups]`` (docs/ANALYSIS.md "Groups of samples"): samples
    analysed together against their own ``reference``, apart from the other groups' (one
    protein of a heterooligomer seen in each vial, a pH series of one protein).

    - ``reference``: one of ``samples`` (its sample name).
    - ``samples``: the group's samples, in their order in the figures and tables.
    - ``labels``: one per sample, the vial (condition) it was measured in: a label names the
      same vial seen from each group that holds it, which lines the groups up across the
      analysis and tells which proteins share a vial. Without labels each sample is its own.
    - ``annotations``: the group's own regions and sites, each kind replacing the shared
      ``[[analysis.regions]]`` or ``[[analysis.sites]]`` (None for a kind the group does not
      give: it takes the shared ones; never merged).
    """

    reference: str
    samples: tuple[str, ...]
    labels: tuple[str, ...] = ()
    regions: tuple | None = None  # hxex.analysis.Region items, or None: the shared ones
    sites: tuple | None = None  # hxex.analysis.Site items, or None: the shared ones

    def __post_init__(self):
        from .compare import Region, Site

        samples = tuple(str(s) for s in self.samples)
        labels = tuple(str(v) for v in self.labels) or samples
        if not samples:
            raise ValueError("samples: name the group's samples")
        if len(labels) != len(samples):
            raise ValueError("one label per sample")
        for what, names in (("sample", samples), ("label", labels)):
            twice = sorted({n for n in names if names.count(n) > 1})
            if twice:
                raise ValueError(f"the {what} {', '.join(map(repr, twice))} is named twice")
        reference = str(self.reference or "")
        if not reference:
            raise ValueError("reference: name the sample the group's others are compared with")
        if not any(_same_sample(reference, s) for s in samples):
            raise ValueError(f"the reference {reference!r} is not among the group's samples")
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "labels", labels)
        for name, cls in (("regions", Region), ("sites", Site)):
            items = getattr(self, name)
            if items is not None:
                object.__setattr__(self, name, tuple(
                    i if isinstance(i, cls) else cls(**i) for i in items))

    def label_of(self, sample: str) -> str:
        """The vial label of one of the group's samples."""
        return self.labels[self.samples.index(sample)]

    def annotations(self, shared: Annotations) -> Annotations:
        """The regions and sites of the group's analysis: its own of each kind, else
        ``shared``'s."""
        return Annotations(
            regions=shared.regions if self.regions is None else self.regions,
            sites=shared.sites if self.sites is None else self.sites)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"reference": self.reference,
                               "samples": (list(self.samples) if self.labels == self.samples
                                           else dict(zip(self.labels, self.samples, strict=True)))}
        for name in ("regions", "sites"):
            items = getattr(self, name)
            if items is not None:
                out[name] = Annotations(**{name: items}).to_dict()[name]
        return out

    @classmethod
    def from_dict(cls, data: Mapping) -> AnalysisGroup:
        """A group from its table: ``samples`` a list of samples, or a table of label =
        sample."""
        unknown = set(data) - {"reference", "samples", "regions", "sites"}
        if unknown:
            raise ValueError(f"unknown key(s) {', '.join(sorted(unknown))}: a group has "
                             "reference, samples, and its own regions and sites")
        samples = data.get("samples")
        if isinstance(samples, Mapping):
            labels, samples = tuple(samples), tuple(samples.values())
        elif isinstance(samples, (list, tuple)):
            labels, samples = (), tuple(samples)
        else:
            raise ValueError("samples must be a list of samples, or a table of label = sample")
        return cls(reference=data.get("reference", ""), samples=samples, labels=labels,
                   regions=data.get("regions"), sites=data.get("sites"))


@dataclass(frozen=True)
class AnalysisConfig:
    """What :func:`hxex.report.write_analysis` computes and draws (the ``[analysis]`` table).

    Which data:

    - ``reference``: the sample the others are compared with (empty: no comparisons); a list of
      samples, or ``'all'`` (every sample in turn), makes one analysis per reference
      (:func:`hxex.report.analysis_references`, :meth:`with_reference`; ``hxex analyse``
      writes a folder for each), and an analysis itself takes one. (The ``reference`` of the
      exchange and labelling conditions is an intrinsic-rate set.)
    - ``samples``: the samples, in this order (empty: all, the reference first).
    - ``groups``: groups of samples analysed apart (name -> :class:`AnalysisGroup`; docs/
      ANALYSIS.md "Groups of samples"): each its own reference, samples and regions and sites,
      one analysis each (:attr:`analysis_names`, :meth:`one`; ``hxex analyse`` writes a folder
      for each, named after the group). ``reference`` and ``samples`` are then the groups' own
      (set together with ``groups`` is refused: one way in); every other setting applies to
      every group. ``group`` names the group an analysis is of (:meth:`with_group`; empty
      otherwise).
    - ``shared_sequences``: what a group's analysis does with a peptide whose sequence another
      group's protein in the same vial (a sample of the same label) also has
      (:mod:`hxex.analysis.shared`; always marked, ``shared_with`` in the peptide tables):
      ``'keep'`` (the default) compares it like any other, ``'leave_out'`` leaves its spectra
      out of the sample's analysis (reason in ``left_out_spectra.csv``).
    - ``times`` (s, or text with a unit such as ``"5 min"``): the time points of the per-time
      figures (empty: all); ``exclude_times``: time points left out of the whole analysis
      (unreliable ones; the fits in the bundle are kept; the controls are not times).
    - ``ks``: the fixed K values analysed (1 is needed: the average, and the choice). Left
      at the default, :func:`hxex.report.write_analysis` drops a K the runs did not fit (and
      the K = 2 values from a default ``overlay``); set, such a K is refused.
    - ``exclude_flagged``: leave out spectra with a quality flag.
    - ``filters``: which K=2 descriptions are kept (:class:`SpeciesFilters`; None: all).
    - ``controls``: control settings to change for the analysis (``{'d2o_fraction': 0.85}``;
      the controls are then recomputed from the run's K=1 fits; they are also recomputed, with
      the run's settings, when the review rejected a control spectrum).
    - ``runs``: separate runs to compare (run name -> bundle; empty: the one run). The run is
      then a column (``run``) beside ``sample`` in every table; ``reference`` and ``samples``
      name a sample (in one run of the set) or ``run:sample`` (one run's sample), and the
      per-sample settings (``fasta``, ``labelling``, ``fragments.exchange``) take a run's own
      entries under its name (``{'WT': ..., 'A': {'WT': ...}}``;
      :func:`hxex.analysis.sample_setting`). ``merge``: with ``runs``, samples to combine
      across runs (``'run:sample'`` -> label; the same label pools them, :class:`RunSet`;
      fragment series are not pooled but drawn side by side).

    Residues:

    - ``residue_level``: which samples have residue-level outputs (redundancy and resolution,
      residue values and their comparisons, Woods plots; coverage maps are drawn regardless):
      ``'auto'`` (the default) those whose peptides make one protein
      (:func:`hxex.analysis.residues.residue_level_states`: two of its peptides overlap, and
      they agree on the residues they share), ``True`` every sample, ``False`` none (the
      peptide-level tables and comparisons stay).
    - ``fasta``: one sequence file for every sample, or one per sample (sample -> path, with a
      run's own under its name); empty: the sequence comes from the peptides.
    - ``coverage_times``: ``'undeuterated'`` (coverage judged on the undeuterated controls; when
      there are none, ``'all'`` with a warning), ``'all'`` or a list of times.
    - ``redundancy``: what it counts, ``'peptides'`` or ``'spectra'``.
    - ``weight_exponent``: of the residue averaging (:func:`residue_map`).
    - ``choice``: the limits of the choice between one and two populations (:class:`ChoiceConfig`;
      None, the default: :func:`choice_defaults` of ``populations``).
    - ``alpha``: the significance level of the comparisons.
    - ``annotations``: regions and sites to mark (``[[analysis.regions]]``, ``[[analysis.sites]]``).
    - ``overlay``: the values the by-residue comparison figures show, any of ``'low'`` and
      ``'high'`` (the K=2 ranks) and ``'avg'`` (K=1); the default, all three, keeps ``'avg'``
      alone when ``ks`` has no 2 (a K=2 value set explicitly is then refused);
      ``overlay_two_only``: draw the K=2 values
      only where the choice keeps two populations (the default; False draws every K=2 value
      the filters keep).
    - ``colours``: a sample's colour in every figure that colours samples (sample, or
      ``run:sample``, -> any colour matplotlib knows: ``'orange'``, ``'#1f77b4'``); a sample not
      named keeps its default (:func:`hxex.report.coverage.sample_styles`: the reference black,
      the others in their order), the same in every figure.
    - ``populations``: how the number of populations is decided (HISTORY 137, 139):
      ``'population_test'`` (the population test of the sets, :mod:`hxex.analysis.set_test`, in
      place of ``best_set_K``; otherwise as ``'best_set'``), ``'best_set'`` (a peptide has two where
      its set of spectra is best described by two, ``best_set_K``; a residue where such
      peptides carry at least half of its weight; the K=2 comparisons need both samples to
      have two) or ``'bootstrap'`` (the residue's K=2 positions tested with their bootstrap
      errors, ``choice``'s confidence and population limits, every peptide's K=2 values used,
      no ``best_set_K`` gate). None (the default): ``'population_test'`` when the run has the
      test, else ``'best_set'`` (:func:`population_rule`).
    - ``borderline_evidence``: in the comparison of the number of populations
      (:func:`compare_population_counts`), a difference is ``borderline`` unless the
      two-population side's summed criterion is at least this far below 0 and the other side's
      above 0 (HISTORY 137, 144).
    - ``peptide_match``: which peptides the peptide comparisons pair, ``'range'`` (the same
      residues, mutated or not) or ``'sequence'`` (identical sequences only;
      :func:`compare_peptides`); ``'auto'`` (the default): ``'sequence'`` for pooled data,
      where two sequences at one range are different proteins, else by range where the two
      peptides are one protein's (mutants: a third of positions or fewer differ), never two
      different proteins' by position (:func:`resolve_match`); ``'range'`` set by hand pairs
      those too.
    - ``head_to_head``: pooled data only: which ranges the head-to-head figure and
      ``compare_sequences`` show, ``'auto'`` (the default: every range with several sequences
      of one class, and every isoform's sequence alone), a list of ranges (``'27-52'``), or
      ``False`` (none; :func:`hxex.analysis.pooled.head_to_head_panels`).
    - ``modified_peptides``: ``'separate'`` (modified forms as states of their own at the
      residue level, ``'WT pS12'``) or ``'combined'`` (averaged with the sample's other
      peptides; :func:`residue_map`).

    Outputs and options:

    - ``format``: of the figure files (``'pdf'``, ``'png'``, ``'svg'``); ``out``: the output
      directory (None: beside the bundle).
    - ``page_size``: the largest page of a figure, ``(width, height)`` in inches (16 by 12).
      A figure that grows with the data (a coverage map's time points, a significance or tier
      map's rows, an overlay's samples) wraps and goes on over further pages, ``NAME_2``,
      ``NAME_3``, each filled with as many of its own panels as fit (owner, 2026-11-04).
    - ``fragments``: the fragment outputs of a run with fragment series
      (:class:`FragmentAnalysis`; ladder figures and rates, closure, per-residue rates; written
      whenever the run has fragments; None turns them off; fragment series are left out of the
      peptide analysis).
    - ``exclude``: rules (:class:`hxex.exclusions.Exclusion`) whose spectra are treated as
      rejected, in memory (the bundle's fits are kept). A project file gives them in
      ``[[review.exclude]]`` instead, with the review table (:mod:`hxex.review_table`), which
      every command applies; :func:`hxex.report.write_analysis` takes those as ``review``.
    - ``protection``: the peptides' protection factors (:mod:`hxex.analysis.protection`; opt-in;
      needs ``labelling``, the labelling conditions, :class:`hxex.kint.ExchangeConditions`, one
      set for every sample or sample -> set). There are no residue-level factors from
      peptides (removed: unreliable); fragment ladders give them (``fragments``).
    - ``summary``: what only the experimenter knows, for the HDX data summary table
      (``summary.csv`` and ``summary.md``, :mod:`hxex.report.summary`; Masson et al. 2019):
      item -> text, one text for every sample or a table of sample -> text. An item named like
      one of Masson's rows (e.g. ``"HDX reaction details"``, ``"Replicates"``) is added to that
      row, any other item is listed first.
    """

    reference: str | tuple[str, ...] = ""
    samples: tuple[str, ...] = ()
    times: tuple[float, ...] = ()
    exclude_times: tuple[float, ...] = ()
    ks: tuple[int, ...] = (1, 2)
    residue_level: str | bool = "auto"
    fasta: str | Mapping[str, str] | None = None
    coverage_times: str | tuple[float, ...] = UNDEUTERATED
    redundancy: str = "peptides"
    weight_exponent: float = 1.0
    filters: SpeciesFilters | None = SpeciesFilters()  # noqa: RUF009
    choice: ChoiceConfig | None = None
    alpha: float = 0.05
    annotations: Annotations = Annotations()  # noqa: RUF009
    exclude_flagged: bool = False
    format: str = "pdf"
    page_size: tuple[float, float] = (16.0, 12.0)
    out: str | None = None
    runs: Mapping[str, str] | None = None
    overlay: tuple[str, ...] = DEFAULT_OVERLAY
    overlay_two_only: bool = True
    peptide_match: str = "auto"
    head_to_head: str | bool | tuple[str, ...] = "auto"
    populations: str | None = None
    borderline_evidence: float = 10.0
    modified_peptides: str = "separate"
    controls: Mapping[str, Any] | None = None
    merge: Mapping[str, str] | None = None
    colours: Mapping[str, str] | None = None
    fragments: FragmentAnalysis | None = FragmentAnalysis()  # noqa: RUF009
    protection: bool = False
    labelling: Any = None  # ExchangeConditions, or sample -> ExchangeConditions
    exclude: tuple = ()  # hxex.exclusions.Exclusion rules: spectra treated as rejected
    summary: Mapping[str, Any] | None = None  # item -> text, or sample -> text
    groups: Mapping[str, AnalysisGroup] | None = None
    group: str = ""  # the group this analysis is of (with_group), empty otherwise
    shared_sequences: str = "keep"

    def __post_init__(self):
        from ..kint import read_conditions

        self._check_groups()

        if isinstance(self.reference, (list, tuple)):  # several: one name stays a name
            names = tuple(dict.fromkeys(str(r) for r in self.reference))
            object.__setattr__(self, "reference", names[0] if len(names) == 1 else
                               names or "")

        object.__setattr__(self, "labelling", read_conditions(self.labelling))
        if not isinstance(self.protection, bool):
            raise ValueError("protection must be true or false")
        if self.protection and not self.labelling:
            raise ValueError("protection needs the labelling conditions (labelling: ph_read, "
                             "temperature, d2o_fraction)")
        object.__setattr__(self, "overlay", tuple(str(v) for v in self.overlay))
        if not self.overlay or any(v not in OVERLAY_VALUES for v in self.overlay):
            raise ValueError(f"overlay must be some of {', '.join(OVERLAY_VALUES)}")
        if self.overlay == DEFAULT_OVERLAY:  # the default follows ks: K=1 alone shows avg
            object.__setattr__(self, "overlay", tuple(
                v for v in self.overlay if OVERLAY_VALUES[v][0] in self.ks) or ("avg",))
        if any(OVERLAY_VALUES[v][0] not in self.ks for v in self.overlay):
            raise ValueError("overlay shows K=2 values ('low', 'high'): include 2 in ks, or "
                             "set overlay = ['avg']")
        if self.runs is not None:
            object.__setattr__(self, "runs", {str(k): str(v) for k, v in self.runs.items()})
            for name in self.runs:
                if not name or SEPARATOR in name:
                    raise ValueError(f"run name {name!r}: must be non-empty without "
                                     f"{SEPARATOR!r}")
        for what, setting in (("fasta", self.fasta), ("labelling", self.labelling)):
            self._check_nested(what, setting)
        if self.summary is not None:
            object.__setattr__(self, "summary", _summary_items(self.summary))
            for item, value in self.summary.items():
                self._check_nested(f"summary.{item}", value)
        if self.fragments is not None:
            self._check_nested("fragments.exchange", self.fragments.exchange)
            self._check_nested("fragments.labels", self.fragments.labels)
        object.__setattr__(self, "samples", tuple(str(s) for s in self.samples))
        object.__setattr__(self, "times", tuple(parse_time(t, "times") for t in self.times))
        object.__setattr__(self, "exclude_times",
                           tuple(parse_time(t, "exclude_times") for t in self.exclude_times))
        object.__setattr__(self, "ks", tuple(int(k) for k in self.ks))
        if not self.ks or 1 not in self.ks:
            raise ValueError("ks must include 1 (the average, and the choice needs it)")
        if isinstance(self.coverage_times, str):
            if self.coverage_times not in (UNDEUTERATED, "all"):
                raise ValueError("coverage_times must be 'undeuterated', 'all' or a list of times")
        else:
            object.__setattr__(self, "coverage_times",
                               tuple(parse_time(t, "coverage_times") for t in self.coverage_times))
        if not (self.residue_level is True or self.residue_level is False
                or self.residue_level == "auto"):
            raise ValueError(f'residue_level must be "auto", true or false, not '
                             f"{self.residue_level!r}")
        if self.redundancy not in COUNTS:
            raise ValueError(f"redundancy must be one of {', '.join(COUNTS)}")
        if self.format not in FORMATS:
            raise ValueError(f"format must be one of {', '.join(FORMATS)}")
        try:
            width, height = (float(v) for v in self.page_size)
        except (TypeError, ValueError):
            raise ValueError(f"page_size = {self.page_size!r}: give the width and height in "
                             "inches, [16, 12]") from None
        if width < 6 or height < 4:
            raise ValueError(f"page_size = {self.page_size!r}: at least 6 inches wide and 4 "
                             "high (a page holds one panel at least)")
        object.__setattr__(self, "page_size", (width, height))
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha = {self.alpha}: write the significance level as a number "
                             "between 0 and 1 (5 % is 0.05)")
        if self.weight_exponent < 0:
            raise ValueError("weight_exponent must be >= 0")
        if self.merge is not None:
            object.__setattr__(self, "merge", {str(k): str(v) for k, v in self.merge.items()})
            if not self.runs:
                raise ValueError("merge combines samples of separate runs: set runs too")
        if self.colours is not None:
            from matplotlib.colors import is_color_like

            object.__setattr__(self, "colours", {str(k): str(v) for k, v in self.colours.items()})
            bad = [f"{k} = {v!r}" for k, v in self.colours.items() if not is_color_like(v)]
            if bad:
                raise ValueError(f"[analysis.colours] {', '.join(bad)}: not a colour (a name such "
                                 "as \"orange\", or a hex code such as \"#1f77b4\")")
        if self.controls is not None:
            from ..controls import ControlConfig

            object.__setattr__(self, "controls", dict(self.controls))
            unknown = set(self.controls) - {f.name for f in fields(ControlConfig)}
            if unknown:
                raise ValueError(f"unknown control setting(s): {', '.join(sorted(unknown))}")
            ControlConfig(**self.controls)  # the values must be valid on their own
        if self.modified_peptides not in MODIFIED:
            raise ValueError(f"modified_peptides must be one of {', '.join(MODIFIED)}")
        # "similar" is what "auto" resolves to (resolve_match), not a value to write
        if self.peptide_match not in ("auto", *MATCHES, "similar"):
            raise ValueError(f"peptide_match must be one of auto, {', '.join(MATCHES)}")
        if self.head_to_head is True:
            object.__setattr__(self, "head_to_head", "auto")
        elif self.head_to_head not in ("auto", False):
            from .pooled import head_to_head_ranges

            given = ([self.head_to_head] if isinstance(self.head_to_head, str)
                     else list(self.head_to_head))
            head_to_head_ranges(given)  # refused with the reason when not ranges
            object.__setattr__(self, "head_to_head", tuple(str(g) for g in given))
        if self.populations is not None and self.populations not in POPULATION_RULES:
            raise ValueError(f"populations must be one of {', '.join(POPULATION_RULES)}")
        from .shared import SHARED_SEQUENCES

        if self.shared_sequences not in SHARED_SEQUENCES:
            raise ValueError(f"shared_sequences must be one of {', '.join(SHARED_SEQUENCES)}")
        if self.shared_sequences == "leave_out" and self.filters is None:
            raise ValueError('shared_sequences = "leave_out" leaves spectra out as '
                             "[analysis.filters] does: filters = false turns that off")
        if self.shared_sequences == "leave_out" and self.runs:
            raise ValueError('shared_sequences = "leave_out" is not made for separate runs '
                             '([analysis.runs]) yet: use "keep" (the peptides are marked, '
                             "shared_with)")
        if isinstance(self.borderline_evidence, bool) or not self.borderline_evidence >= 0:
            raise ValueError("borderline_evidence must be a number >= 0")

        if self.samples:
            for ref in self.references:
                if ref != ALL_REFERENCES and not any(_same_sample(ref, s) for s in self.samples):
                    raise ValueError(f"the reference {ref!r} is not among the samples")

    def _check_groups(self) -> None:
        """``groups`` as :class:`AnalysisGroup` (from tables too), and the one way in."""
        if not self.groups:
            object.__setattr__(self, "groups", None)
            if self.group:
                raise ValueError(f"group {self.group!r}: there are no groups")
            return
        groups = {}
        for name, value in self.groups.items():
            name = str(name)
            if not name.strip():
                raise ValueError("a group needs a name")
            if name.lower() == "groups":  # results_analysis_groups is the folder across them
                raise ValueError('[analysis.groups.groups]: "groups" names the folder across '
                                 "the groups (results_analysis_groups); give the group another "
                                 "name")
            try:
                groups[name] = (value if isinstance(value, AnalysisGroup)
                                else AnalysisGroup.from_dict(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"[analysis.groups.{name}] {exc}") from None
        object.__setattr__(self, "groups", groups)
        if self.group:
            if self.group not in groups:
                raise ValueError(f"group {self.group!r} is not one of the groups "
                                 f"({', '.join(groups)})")
        elif self.reference or self.samples:
            raise ValueError("[analysis] reference and samples go in each group with "
                             "[analysis.groups] (its reference and samples): delete them from "
                             "[analysis]")

    @property
    def analysis_names(self) -> tuple[str, ...]:
        """The analyses these settings make, by name: the groups (until one is chosen,
        :meth:`with_group`), else the references as written (:attr:`references`; none: one
        analysis without a reference)."""
        if self.groups and not self.group:
            return tuple(self.groups)
        return self.references

    @property
    def by_group(self) -> bool:
        """Whether the analyses are the groups' (``[analysis.groups]``), one per group."""
        return bool(self.groups) and not self.group

    def one(self, name: str) -> AnalysisConfig:
        """The analysis ``name`` of :attr:`analysis_names`: a group's (:meth:`with_group`), or
        against one reference (:meth:`with_reference`); itself when it is one analysis."""
        if self.by_group:
            return self.with_group(name)
        return self if self.single_reference else self.with_reference(name)

    def with_group(self, name: str) -> AnalysisConfig:
        """The analysis of the group ``name``: its reference and samples, and its own regions
        and sites (each kind it gives replacing the shared ones)."""
        if not self.groups or name not in self.groups:
            raise ValueError(f"no group {name!r}" + (f" (the groups: {', '.join(self.groups)})"
                                                     if self.groups else ""))
        g = self.groups[name]
        return replace(self, reference=g.reference, samples=g.samples,
                       annotations=g.annotations(self.annotations), group=name)

    @property
    def references(self) -> tuple[str, ...]:
        """The references as written: none, one, several, or ``('all',)``."""
        if not self.reference:
            return ()
        return self.reference if isinstance(self.reference, tuple) else (self.reference,)

    @property
    def single_reference(self) -> bool:
        """Whether this is one analysis: no reference, or one sample (not a list or ``'all'``),
        and not the groups' analyses before one is chosen."""
        return (not self.by_group and isinstance(self.reference, str)
                and self.reference != ALL_REFERENCES)

    def with_reference(self, reference: str) -> AnalysisConfig:
        """This analysis against one ``reference`` (a sample of :attr:`references`, or of every
        sample for ``'all'``)."""
        return replace(self, reference=reference)

    def _check_nested(self, what: str, setting) -> None:
        """A per-sample setting's nested tables (a run's own entries) must be named after runs
        of ``runs``."""
        if not isinstance(setting, Mapping):
            return
        for key, value in setting.items():
            if isinstance(value, Mapping) and key not in (self.runs or {}):
                raise ValueError(f"{what}.{key} is a table of samples, but {key!r} is not a run "
                                 "of runs ([analysis.runs]); a sample's entry is one value")

    def sequences(self, base: str | Path = ".") -> str | dict | None:
        """The protein sequence(s) of ``fasta`` (relative paths from ``base``): one, or by
        sample with a run's own under its name (:func:`hxex.analysis.sample_setting`). A FASTA
        numbered by its header (``>NAME/START-END``, :func:`hxex.io.read_fasta_numbered`)
        gives a :class:`NumberedSequence`."""
        from ..io import read_fasta_numbered

        def read(path: str) -> str:
            p = Path(path).expanduser()
            sequence, first = read_fasta_numbered(p if p.is_absolute() else Path(base) / p)
            return sequence if first == 1 else NumberedSequence(sequence, first)

        if self.fasta is None:
            return None
        if isinstance(self.fasta, str):
            return read(self.fasta)
        return {str(key): ({str(s): read(p) for s, p in value.items()}
                           if isinstance(value, Mapping) else read(value))
                for key, value in self.fasta.items()}

    @property
    def choice_limits(self) -> ChoiceConfig:
        """The limits of the choice in use: ``choice``, or :func:`choice_defaults` of
        ``populations`` when it is unset (so changing ``populations`` changes them)."""
        return self.choice if self.choice is not None else choice_defaults(self.populations)

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("filters", "choice", "fragments"):
                hidden = getattr(value, "NOT_SETTINGS", ())  # set by the analysis itself
                value = None if value is None else value.to_dict() if hasattr(
                    value, "to_dict") else {g.name: getattr(value, g.name)
                                            for g in fields(value) if g.name not in hidden}
            elif f.name == "annotations":
                value = value.to_dict()
            elif f.name == "groups":
                value = None if value is None else {k: g.to_dict() for k, g in value.items()}
            elif f.name == "exclude":
                value = [r.to_dict() for r in value]
            elif f.name == "labelling":
                from ..kint import conditions_dict

                value = conditions_dict(value)
            elif isinstance(value, tuple):
                value = list(value)
            elif isinstance(value, Mapping):
                value = dict(value)
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: Mapping) -> AnalysisConfig:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown analysis setting(s): {', '.join(sorted(unknown))}")
        values = dict(data)
        if "filters" in values:
            f = values["filters"]
            values["filters"] = None if f is None or f is False else (
                SpeciesFilters() if f is True else SpeciesFilters(**f))
        if values.get("choice") is not None:
            values["choice"] = ChoiceConfig.from_dict(values["choice"])
        if "annotations" in values:
            values["annotations"] = Annotations.from_dict(values["annotations"])
        if "exclude" in values:
            from ..exclusions import Exclusion, read_exclusions

            values["exclude"] = tuple(
                r if isinstance(r, Exclusion) else read_exclusions(r, "analysis.exclude")[0]
                for r in values["exclude"])
        if "fragments" in values:
            f = values["fragments"]
            values["fragments"] = None if f is None or f is False else (
                FragmentAnalysis() if f is True else FragmentAnalysis.from_dict(f))
        return cls(**values)



def resolve_match(match: str, runs) -> str:
    """The peptide matching an analysis uses: ``match`` itself unless ``'auto'``, which is
    ``'sequence'`` when any of ``runs`` (Results, by run name) was read as pooled data or with a
    pool's candidates (``[input] pooled_fasta``, also from scan files): two sequences at one
    range are then different proteins, never to be paired silently (docs/ANALYSIS.md "Pooled
    samples: possible identities"); else ``'similar'``: by range where the two peptides are one
    protein's, differing at a third of positions or fewer
    (:func:`hxex.analysis.compare.different_sequences`)."""
    from ..io.pooled import CANDIDATES_RECORD

    if match != "auto":
        return match
    sources = [getattr(r.settings, "spectra_source", None) or {} for r in runs.values()
               if getattr(r, "settings", None) is not None]
    pooled = any(s.get("reader") == "pooled" or s.get(CANDIDATES_RECORD) for s in sources)
    # else by range where the two peptides are one protein's (mutants), never two different
    # proteins' by position (owner, 2026-11-23; "range" set by hand still pairs those)
    return "sequence" if pooled else "similar"


def _summary_items(items) -> dict:
    """``[analysis.summary]`` checked: item -> text, or a table of sample (or run, then
    sample) -> text; numbers are written as text."""
    if not isinstance(items, Mapping):
        raise ValueError("summary must be a table of item = text (or of item = a table of "
                         "sample = text)")

    def text(value, where: str):
        if isinstance(value, Mapping):
            return {str(k): text(v, f"{where}.{k}") for k, v in value.items()}
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError(f"summary.{where} = {value!r}: write the text in quotes")
        return str(value)

    return {str(item): text(value, str(item)) for item, value in items.items()}

