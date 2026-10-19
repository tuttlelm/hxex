# SPDX-License-Identifier: GPL-3.0-or-later
"""The settings of a post-fitting analysis (the ``[analysis]`` section of a project file; see
:mod:`hxex.config` and :func:`hxex.report.write_analysis`)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from ..io.times import parse_time
from .compare import MATCHES, Annotations
from .fragments import FragmentAnalysis
from .populations import ChoiceConfig
from .residues import COUNTS, MODIFIED
from .runs import SEPARATOR
from .species import SpeciesFilters

FORMATS = ("pdf", "png", "svg")
UNDEUTERATED = "undeuterated"
# the values a by-residue comparison can show: name -> (K, rank)
OVERLAY_VALUES = {"low": (2, 1), "high": (2, 2), "avg": (1, 1)}


@dataclass(frozen=True)
class AnalysisConfig:
    """What :func:`hxex.report.write_analysis` computes and draws (the ``[analysis]`` table).

    Which data:

    - ``reference``: the sample the others are compared with (empty: no comparisons). (The
      ``reference`` of the exchange and labelling conditions is an intrinsic-rate set.)
    - ``samples``: the samples, in this order (empty: all, the reference first).
    - ``times`` (s, or text with a unit such as ``"5 min"``): the time points of the per-time
      figures (empty: all); ``exclude_times``: time points left out of the whole analysis
      (unreliable ones; the fits in the bundle are kept; the controls are not times).
    - ``ks``: the fixed K values analysed (1 is needed: the average, and the choice).
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

    - ``fasta``: one sequence file for every sample, or one per sample (sample -> path, with a
      run's own under its name); empty: the sequence comes from the peptides.
    - ``coverage_times``: ``'undeuterated'`` (coverage judged on the undeuterated controls; when
      there are none, ``'all'`` with a warning), ``'all'`` or a list of times.
    - ``redundancy``: what it counts, ``'peptides'`` or ``'spectra'``.
    - ``weight_exponent``: of the residue averaging (:func:`residue_map`).
    - ``choice``: the limits of the choice between one and two populations (:class:`ChoiceConfig`).
    - ``alpha``: the significance level of the comparisons.
    - ``annotations``: regions and sites to mark (``[[analysis.regions]]``, ``[[analysis.sites]]``).
    - ``overlay``: the values the by-residue comparison figures show, any of ``'low'`` and
      ``'high'`` (the K=2 ranks) and ``'avg'`` (K=1); ``overlay_two_only``: draw the K=2 values
      only where the choice keeps two populations (the default; False draws every K=2 value
      the filters keep).
    - ``peptide_match``: which peptides the peptide comparisons pair, ``'range'`` (the same
      residues, mutated or not) or ``'sequence'`` (identical sequences only;
      :func:`compare_peptides`).
    - ``modified_peptides``: ``'separate'`` (modified forms as states of their own at the
      residue level, ``'WT pS59'``) or ``'combined'`` (averaged with the sample's other
      peptides; :func:`residue_map`).

    Outputs and options:

    - ``format``: of the figure files (``'pdf'``, ``'png'``, ``'svg'``); ``out``: the output
      directory (None: beside the bundle).
    - ``fragments``: the fragment outputs of a run with fragment series
      (:class:`FragmentAnalysis`; ladder figures and rates, closure, per-residue rates; written
      whenever the run has fragments; None turns them off; fragment series are left out of the
      peptide analysis).
    - ``protection``: the peptides' protection factors (:mod:`hxex.analysis.protection`; opt-in;
      needs ``labelling``, the labelling conditions, :class:`hxex.kint.ExchangeConditions`, one
      set for every sample or sample -> set). There are no residue-level factors from
      peptides (removed: unreliable); fragment ladders give them (``fragments``).
    """

    reference: str = ""
    samples: tuple[str, ...] = ()
    times: tuple[float, ...] = ()
    exclude_times: tuple[float, ...] = ()
    ks: tuple[int, ...] = (1, 2)
    fasta: str | Mapping[str, str] | None = None
    coverage_times: str | tuple[float, ...] = UNDEUTERATED
    redundancy: str = "peptides"
    weight_exponent: float = 1.0
    filters: SpeciesFilters | None = SpeciesFilters()  # noqa: RUF009
    choice: ChoiceConfig = ChoiceConfig()  # noqa: RUF009
    alpha: float = 0.05
    annotations: Annotations = Annotations()  # noqa: RUF009
    exclude_flagged: bool = False
    format: str = "pdf"
    out: str | None = None
    runs: Mapping[str, str] | None = None
    overlay: tuple[str, ...] = ("low", "high")
    overlay_two_only: bool = True
    peptide_match: str = "range"
    modified_peptides: str = "separate"
    controls: Mapping[str, Any] | None = None
    merge: Mapping[str, str] | None = None
    fragments: FragmentAnalysis | None = FragmentAnalysis()  # noqa: RUF009
    protection: bool = False
    labelling: Any = None  # ExchangeConditions, or sample -> ExchangeConditions

    def __post_init__(self):
        from ..kint import read_conditions

        object.__setattr__(self, "labelling", read_conditions(self.labelling))
        if not isinstance(self.protection, bool):
            raise ValueError("protection must be true or false")
        if self.protection and not self.labelling:
            raise ValueError("protection needs the labelling conditions (labelling: ph_read, "
                             "temperature, d2o_fraction)")
        object.__setattr__(self, "overlay", tuple(str(v) for v in self.overlay))
        if not self.overlay or any(v not in OVERLAY_VALUES for v in self.overlay):
            raise ValueError(f"overlay must be some of {', '.join(OVERLAY_VALUES)}")
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
        if self.redundancy not in COUNTS:
            raise ValueError(f"redundancy must be one of {', '.join(COUNTS)}")
        if self.format not in FORMATS:
            raise ValueError(f"format must be one of {', '.join(FORMATS)}")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self.weight_exponent < 0:
            raise ValueError("weight_exponent must be >= 0")
        if self.merge is not None:
            object.__setattr__(self, "merge", {str(k): str(v) for k, v in self.merge.items()})
            if not self.runs:
                raise ValueError("merge combines samples of separate runs: set runs too")
        if self.controls is not None:
            from ..controls import ControlConfig

            object.__setattr__(self, "controls", dict(self.controls))
            unknown = set(self.controls) - {f.name for f in fields(ControlConfig)}
            if unknown:
                raise ValueError(f"unknown control setting(s): {', '.join(sorted(unknown))}")
            ControlConfig(**self.controls)  # the values must be valid on their own
        if self.modified_peptides not in MODIFIED:
            raise ValueError(f"modified_peptides must be one of {', '.join(MODIFIED)}")
        if self.peptide_match not in MATCHES:
            raise ValueError(f"peptide_match must be one of {', '.join(MATCHES)}")
        if self.reference and self.samples and self.reference not in self.samples:
            raise ValueError(f"the reference {self.reference!r} is not among the samples")

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
        sample with a run's own under its name (:func:`hxex.analysis.sample_setting`)."""
        from ..io import read_fasta_sequence

        def read(path: str) -> str:
            p = Path(path).expanduser()
            return read_fasta_sequence(p if p.is_absolute() else Path(base) / p)

        if self.fasta is None:
            return None
        if isinstance(self.fasta, str):
            return read(self.fasta)
        return {str(key): ({str(s): read(p) for s, p in value.items()}
                           if isinstance(value, Mapping) else read(value))
                for key, value in self.fasta.items()}

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("filters", "choice", "fragments"):
                value = None if value is None else value.to_dict() if hasattr(
                    value, "to_dict") else {g.name: getattr(value, g.name)
                                            for g in fields(value)}
            elif f.name == "annotations":
                value = value.to_dict()
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
        if "choice" in values:
            values["choice"] = ChoiceConfig.from_dict(values["choice"])
        if "annotations" in values:
            values["annotations"] = Annotations.from_dict(values["annotations"])
        if "fragments" in values:
            f = values["fragments"]
            values["fragments"] = None if f is None or f is False else (
                FragmentAnalysis() if f is True else FragmentAnalysis.from_dict(f))
        return cls(**values)

