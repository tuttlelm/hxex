# SPDX-License-Identifier: GPL-3.0-or-later
"""The settings of a post-fitting analysis (the ``[analysis]`` section of a project file; see
:mod:`hxex.config` and :func:`hxex.report.write_analysis`)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from .compare import Annotations
from .populations import ChoiceConfig
from .residues import COUNTS
from .runs import SEPARATOR
from .species import SpeciesFilters

FORMATS = ("pdf", "png", "svg")
UNDEUTERATED = "undeuterated"
# the values a by-residue comparison can show: name -> (K, rank)
OVERLAY_VALUES = {"low": (2, 1), "high": (2, 2), "avg": (1, 1)}


@dataclass(frozen=True)
class AnalysisConfig:
    """What :func:`hxex.report.write_analysis` computes and draws.

    ``reference``: the sample the others are compared with (empty: no comparisons);
    ``samples``: the samples, in this order (empty: all, the reference first); ``times``: the
    time points of the per-time figures (empty: all but the fully deuterated control); ``ks``:
    the fixed K values analysed; ``fasta``: one sequence file for every sample, or one per sample
    (sample -> path; empty: the sequence comes from the peptides); ``coverage_times``:
    ``'undeuterated'``, ``'all'`` or a list of times; ``redundancy``: ``'peptides'`` or
    ``'spectra'``; ``weight_exponent`` of the residue averaging; ``filters`` (None: none) and
    ``choice`` limits; ``alpha`` of the comparisons; ``annotations`` (regions, sites);
    ``exclude_flagged``; ``format`` of the figure files; ``out``, the output directory (None:
    beside the bundle); ``runs``: separate runs to compare (run name -> bundle; empty: the one
    run), whose samples are then named ``run:sample`` everywhere (``reference``, ``samples``,
    ``fasta``); ``overlay``: the values the by-residue comparison figures show, any of
    ``'low'`` and ``'high'`` (the K=2 ranks) and ``'avg'`` (K=1)."""

    reference: str = ""
    samples: tuple[str, ...] = ()
    times: tuple[float, ...] = ()
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

    def __post_init__(self):
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
        object.__setattr__(self, "samples", tuple(str(s) for s in self.samples))
        object.__setattr__(self, "times", tuple(float(t) for t in self.times))
        object.__setattr__(self, "ks", tuple(int(k) for k in self.ks))
        if not self.ks or 1 not in self.ks:
            raise ValueError("ks must include 1 (the average, and the choice needs it)")
        if isinstance(self.coverage_times, str):
            if self.coverage_times not in (UNDEUTERATED, "all"):
                raise ValueError("coverage_times must be 'undeuterated', 'all' or a list of times")
        else:
            object.__setattr__(self, "coverage_times",
                               tuple(float(t) for t in self.coverage_times))
        if self.redundancy not in COUNTS:
            raise ValueError(f"redundancy must be one of {', '.join(COUNTS)}")
        if self.format not in FORMATS:
            raise ValueError(f"format must be one of {', '.join(FORMATS)}")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self.weight_exponent < 0:
            raise ValueError("weight_exponent must be >= 0")
        if self.reference and self.samples and self.reference not in self.samples:
            raise ValueError(f"the reference {self.reference!r} is not among the samples")

    def sequences(self, base: str | Path = ".") -> str | dict[str, str] | None:
        """The protein sequence(s) of ``fasta`` (relative paths from ``base``)."""
        from ..io import read_fasta_sequence

        def read(path: str) -> str:
            p = Path(path).expanduser()
            return read_fasta_sequence(p if p.is_absolute() else Path(base) / p)

        if self.fasta is None:
            return None
        if isinstance(self.fasta, str):
            return read(self.fasta)
        return {str(sample): read(path) for sample, path in self.fasta.items()}

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("filters", "choice"):
                value = None if value is None else value.to_dict() if hasattr(
                    value, "to_dict") else {g.name: getattr(value, g.name)
                                            for g in fields(value)}
            elif f.name == "annotations":
                value = value.to_dict()
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
        return cls(**values)

