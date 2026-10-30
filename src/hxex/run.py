# SPDX-License-Identifier: GPL-3.0-or-later
"""Running a whole data set: pick, drop, fit, bootstrap, controls, quality flags, bundle.

:func:`run` takes raw spectra (or already picked ones), drops the spectra the picker flagged,
fits the rest in chunks of whole series, and returns a :class:`RunResult` (and writes a results
bundle if asked). A chunk holds whole series because the noise estimate and the control
references work within a series; each measurement is seeded from its own id, so the result does not
depend on how the data was split (tests/test_run.py). The chunk size is set by a memory budget
(by default a share of the free GPU memory, or a fixed amount on the CPU) and can be capped.

Progress is reported as ``progress(done, 1000)``: picking, the fit and the bootstrap each take a
share of the 1000 units (about 5, 30 and 65 with a bootstrap; 5 and 95 without; the last 10 are
the controls, the quality flags, writing and figures), divided between chunks by their number of
measurements. It is an estimate of the time, not a count of work.
A hook with a ``start_stage`` method, such as :class:`~hxex.hooks.ProgressLog`, is also told the
stages and their units (see :func:`~hxex.hooks.start_stage`): ``pick``, ``fit``, ``bootstrap``,
``controls and quality``, ``write fits``, ``write`` and ``figures``. ``cancel`` is polled
throughout and raises :class:`~hxex.hooks.Cancelled`.

With a bundle (``out``) and a bootstrap, the bundle is first written as soon as the fits, the
controls and the quality flags are done, marked as waiting for its bootstrap
(:attr:`~hxex.io.Bundle.pending`), and written again, complete, at the end. A run stopped or
failed during the bootstrap leaves that bundle, and :func:`resume_bootstrap` adds the bootstrap
without fitting again. Stopped before, nothing is written.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import chem
from .bootstrap import (
    BootstrapConfig,
    BootstrapResult,
    bootstrap,
    bootstrap_by_k,
    scatter,
    selected_from,
)
from .controls import (
    ControlConfig,
    ControlTable,
    apply_control_roles,
    charge_groups,
    fit_controls,
    fragment_ids,
    series_groups,
    series_rates,
    series_residues,
    series_rt,
)
from .fit import FitConfig, FitResult, Measurements, fit_models
from .fit.result import _KFIT_ARRAYS
from .hooks import CancelHook, ProgressHook, check_cancel, report, start_stage
from .picking import (
    DEFAULT_PICKER_CONFIG,
    BinnedSet,
    EnvelopeConfig,
    EnvelopeTable,
    PeakMeasure,
    PickerConfig,
    choose_measure,
    pick_spectra,
    series_envelopes,
)
from .population_test import (
    DEFAULT_D2O_FRACTION,
    PopulationTest,
    PopulationTestConfig,
    check_fit_config,
    population_test,
)
from .quality import QualityConfig, QualityFlags, assess_quality
from .review_table import merge_choices
from .spectra import SpectrumSet

if TYPE_CHECKING:
    from .report.figures import PlotConfig
    from .report.series import Reference

log = logging.getLogger(__name__)

UNITS = 1000  # the ``total`` of the progress hook
_FINISH = 10  # the last units: controls, quality flags, writing
CPU_BUDGET = 4 * 2**30  # bytes per chunk on the CPU unless set
GPU_SHARE = 0.5  # share of the free GPU memory used by one chunk unless set
# memory of the fit (the bootstrap needs less) in units of one pmf array (shape points x bins x
# components per measurement). Bins that are whole numbers (every picked spectrum) use the
# integer convolution: peaks of 7.6 to 9.2 arrays measured on the CPU and the GPU for 945 and
# 3,727 measurements (docs/HISTORY.md item 46), so 12. Other bins evaluate the pmf at every
# shifted bin: 5.2 to 5.9 times 12 arrays on the GPU for 94 to 542 Bimodal measurements, so 75.
WORK_FACTOR = 12.0


@dataclass(frozen=True)
class RunConfig:
    """Settings of :func:`run`: the ``[run]`` table of a project file, one stage per sub-table.

    - ``picker``: peak picking (:class:`~hxex.picking.PickerConfig`).
    - ``fit``: the fit of one to several populations (:class:`~hxex.fit.FitConfig`).
    - ``bootstrap``: standard errors and intervals (:class:`~hxex.bootstrap.BootstrapConfig`);
      None: no bootstrap.
    - ``population_test``: the population test
      (:class:`~hxex.population_test.PopulationTestConfig`); None: none, each spectrum's K is
      the fit criterion's. With it, each spectrum's K is the test's, and ``fit.screen`` is the
      test's screen (:func:`hxex.fit.screen.can_exceed`).
    - ``controls``: the back-exchange correction from the undeuterated and fully deuterated
      controls (:class:`~hxex.controls.ControlConfig`); None: no correction, the means stay in
      measured deuterons. The controls are the K=1 fits, made for the control spectra whatever
      ``fit.k_min`` is (:func:`hxex.fit.fit_models`).
    - ``quality``: the quality flags (:class:`~hxex.quality.QualityConfig`); None: none.
    - ``registry``: the modified residues a sequence may name, as modX labels
      (:class:`~hxex.chem.ModificationRegistry`).
    - ``envelope``: the isotope envelope that places the bins and shapes the fit
      (:class:`~hxex.picking.EnvelopeConfig`).
    - ``envelope_fallback_time`` (s): the spectra at this time are the envelope references of a
      series whose own (its undeuterated controls) are missing or poor; None: none.
    - ``n_max_factor``: the fit's ``n`` (exchangeable sites) may be up to this times the amide
      count.
    - ``drop_no_signal``: drop the spectra the picker found no signal in (``empty`` ones are
      always dropped).
    - ``memory_budget_bytes``: the memory of one chunk of the fit; None: automatic (see the
      module docstring).
    - ``max_chunk``: the most measurements in one chunk; None: no cap.
    """

    picker: PickerConfig = DEFAULT_PICKER_CONFIG
    fit: FitConfig = FitConfig()  # noqa: RUF009
    bootstrap: BootstrapConfig | None = None
    population_test: PopulationTestConfig | None = None
    controls: ControlConfig | None = None
    quality: QualityConfig | None = QualityConfig()  # noqa: RUF009
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS
    envelope: EnvelopeConfig = EnvelopeConfig()  # noqa: RUF009
    envelope_fallback_time: float | None = None
    n_max_factor: float = 1.2
    drop_no_signal: bool = True
    memory_budget_bytes: int | None = None
    max_chunk: int | None = None

    def __post_init__(self):
        if not self.n_max_factor > 0:
            raise ValueError("n_max_factor must be positive")
        if self.memory_budget_bytes is not None and self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive")
        if self.max_chunk is not None and self.max_chunk < 1:
            raise ValueError("max_chunk must be at least 1")
        if self.population_test is not None:
            check_fit_config(self.fit)
        if self.bootstrap is not None and not isinstance(self.bootstrap.ks, str):
            missing = [k for k in self.bootstrap.ks if k not in self.fit.k_values]
            if missing:
                raise ValueError(f"bootstrap.ks {missing} are outside the fitted K values")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = v.to_dict() if hasattr(v, "to_dict") else v
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunConfig:
        unknown = set(data) - {f.name for f in fields(cls)}
        if "undeut_time" in unknown:
            raise ValueError("run setting undeut_time no longer exists: the undeuterated "
                             "controls (the noise references) are the spectra with the role "
                             "undeut, set by the formats' control labels or role in "
                             "[[input.runs]]")
        unknown.discard("plots")  # bundles from before the run stopped drawing figures
        if unknown:
            raise ValueError(f"unknown run setting(s): {', '.join(sorted(unknown))}")
        parts = {
            "picker": PickerConfig, "fit": FitConfig,
            "bootstrap": BootstrapConfig, "population_test": PopulationTestConfig,
            "controls": ControlConfig, "quality": QualityConfig,
            "registry": chem.ModificationRegistry, "envelope": EnvelopeConfig,
        }
        kwargs = {
            k: (parts[k].from_dict(v) if k in parts and v is not None else v)
            for k, v in data.items() if k != "plots"
        }
        return cls(**kwargs)


@dataclass(frozen=True)
class RunResult:
    """What a run produced. ``binned`` is every picked spectrum (flags included); the spectra
    where ``kept`` is False were dropped and are not in ``measurements`` or ``fit``;
    ``dropped_reason`` says why (``empty``, ``no_signal``, or ``no_bins``: nothing to fit, a
    peptide without a bin). ``bootstrap`` is at each
    measurement's selected K; ``bootstrap_by_k`` (when ``BootstrapConfig.ks`` asks for it) has
    the fixed-K bootstrap of every measurement at each K requested. ``population_test`` is the
    test of every measurement (with it, ``fit.selected_K`` is the test's K).
    ``spectra_source`` is how the raw spectra were read (their
    :attr:`~hxex.spectra.SpectrumSet.source`), written to the bundle even when the spectra are
    not. ``peak_measure`` is whether the bins took heights or areas, and why
    (:class:`~hxex.picking.PeakMeasure`; None for a set picked before the run)."""

    config: RunConfig
    binned: BinnedSet
    kept: np.ndarray
    dropped_reason: np.ndarray
    measurements: Measurements
    fit: FitResult
    bootstrap: BootstrapResult | None
    bootstrap_by_k: dict[int, BootstrapResult] | None
    controls: ControlTable | None
    quality: QualityFlags | None
    envelopes: EnvelopeTable | None = None
    spectra_source: dict | None = None
    review: Any = None  # the hxex.io.bundle.ReviewTable carried over to the bundle, if any
    excluded: dict | None = None  # the run-level exclusions (hxex.exclusions.exclusion_record)
    view_exclude: tuple = ()  # [[review.exclude]] rules applied to the run's figures
    project_file: dict | None = None  # the project file's other settings (config.project_record)
    population_test: PopulationTest | None = None
    peak_measure: PeakMeasure | None = None

    @property
    def dropped_ids(self) -> np.ndarray:
        return self.binned.spectrum_id[~self.kept]

    def write_figures(
        self,
        directory: str | Path,
        *,
        spectra: SpectrumSet | None = None,
        reference: Reference | None = None,
        progress: ProgressHook | None = None,
        cancel: CancelHook | None = None,
        plots: PlotConfig | None = None,
    ) -> list[Path]:
        """Write the series figures described by ``plots`` (default: the default settings) with
        the run's review applied; ``spectra`` adds the raw layer. Returns the files. (A run
        draws no figures itself; this is the run's equivalent of ``hxex report --figures``.)"""
        from .exclusions import exclude_from_results
        from .report import Results
        from .report.figures import PlotConfig, write_figures

        return write_figures(
            exclude_from_results(Results.from_run(self, spectra=spectra), self.view_exclude),
            directory,
            plots or PlotConfig(),
            reference=reference, progress=progress, cancel=cancel,
        )

    def write(
        self,
        path: str | Path,
        *,
        spectra: SpectrumSet | None = None,
        project: str | None = None,
        notes: str = "",
        overwrite: bool = False,
        pending: Sequence[str] = (),
        review=None,
        provenance=None,
    ) -> Path:
        """Write the results bundle (see :mod:`hxex.io.bundle`); ``spectra`` adds the raw data,
        ``pending``, ``review`` and ``provenance`` are as for :func:`~hxex.io.write_bundle`."""
        from .io.bundle import write_bundle

        return write_bundle(
            path, measurements=self.measurements, fit=self.fit, bootstrap=self.bootstrap,
            bootstrap_by_k=self.bootstrap_by_k, binned=self.binned, spectra=spectra,
            picker_config=self.config.picker,
            controls=self.controls, quality=self.quality, envelopes=self.envelopes,
            population_test=self.population_test, run_config=self.config.to_dict(),
            spectra_source=self.spectra_source if spectra is None or spectra.source is None
            else spectra.source,
            project=project, notes=notes, overwrite=overwrite, pending=pending, review=review,
            provenance=provenance, excluded=self.excluded, project_file=self.project_file,
            peak_measure=self.peak_measure,
        )


def chunk_bytes(n_bins: int, shape_len: int, k_max: int, n: int, itemsize: int,
                factor: float = WORK_FACTOR) -> float:
    """Estimated memory of a chunk of ``n`` measurements with ``n_bins`` bins and shapes of
    ``shape_len`` entries: the array of the binomial pmf of every shape position, bin and
    population, times the number of such arrays alive at once (``factor``: ``WORK_FACTOR``,
    a measured peak, which grows linearly with the chunk)."""
    return float(itemsize) * factor * n_bins * shape_len * k_max * n


def default_budget(config: RunConfig) -> int:
    """The chunk memory budget of a run (see the module docstring)."""
    if config.memory_budget_bytes is not None:
        return config.memory_budget_bytes
    device = config.fit.backend.torch_device()
    if device.type == "cuda":
        import torch

        free, _ = torch.cuda.mem_get_info(device)
        return int(GPU_SHARE * free)
    return CPU_BUDGET


def plan_chunks(
    data: Measurements, k_max: int, itemsize: int, budget: float, max_chunk: int | None = None
) -> list[np.ndarray]:
    """Groups of measurement positions, each made of whole series (``data_id``), that fit in
    ``budget`` bytes by :func:`chunk_bytes` (a series larger than that is a chunk of its own).
    Series are ordered by their widest spectrum so that a chunk has similar widths, which keeps
    the padding small; the groups cover every measurement once."""
    shape_len = max(len(s) for s in data.shapes)
    factor = WORK_FACTOR
    widths = np.array([len(b) for b in data.bins])
    series = [np.flatnonzero(data.data_id == s) for s in np.unique(data.data_id)]
    series.sort(key=lambda rows: -int(widths[rows].max()))
    chunks: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    count = 0
    for rows in series:
        if current:
            width = max(int(widths[r].max()) for r in [*current, rows])
            fits = chunk_bytes(width, shape_len, k_max, count + len(rows), itemsize,
                               factor) <= budget
            if max_chunk is not None and count + len(rows) > max_chunk:
                fits = False
            if not fits:
                chunks.append(current)
                current, count = [], 0
        current.append(rows)
        count += len(rows)
    if current:
        chunks.append(current)
    out = [np.sort(np.concatenate(group)) for group in chunks]
    for rows in out:
        need = chunk_bytes(int(widths[rows].max()), shape_len, k_max, len(rows), itemsize,
                           factor)
        if need > budget:
            log.warning(
                "a chunk of %d measurements needs about %.1f GiB, over the budget of %.1f GiB "
                "(it is one series that cannot be split)", len(rows), need / 2**30, budget / 2**30,
            )
    return out


class _Span:
    """A progress hook that maps (done, total) of one step onto a range of the overall units."""

    def __init__(self, hook: ProgressHook | None, start: float, end: float):
        self.hook, self.start, self.end = hook, start, end

    def __call__(self, done: int, total: int) -> None:
        if self.hook is not None:
            frac = done / total if total else 1.0
            report(self.hook, int(round(self.start + (self.end - self.start) * frac)), UNITS)


def _bootstrap_ks(config: RunConfig, fit: FitResult) -> list[int]:
    """The Ks of the fixed-K bootstrap the run was asked for (none for ``'selected'``)."""
    if config.bootstrap is None or config.bootstrap.ks == "selected":
        return []
    # "all": the Ks chosen from, not the controls' extra K = 1 (fit.k_values may hold it)
    return (list(config.fit.k_values) if config.bootstrap.ks == "all"
            else list(config.bootstrap.ks))


@dataclass
class _Plan:
    """The chunks of an analysis and its progress units: the share of ``progress_range`` of
    the fit and the population test (``fit_share``), the test's share of that (``test_share``)
    and, per chunk, its edges (fraction of the measurements before it)."""

    chunks: list[np.ndarray]
    subs: list[Measurements]
    edges: np.ndarray
    lo: float
    hi: float
    fit_share: float
    test_share: float = 0.0


# time of the population test's null and of a bootstrap, against the fit's: at 20 simulations
# step K took 11 + 7 (K - 1) times the fits on a CPU (the tutorial data, HISTORY 174)
_TEST_WEIGHT, _TEST_WEIGHT_STEP = 11.0, 7.0
_BOOTSTRAP_WEIGHT = 7.0 / 3.0


def bundle_noise(picked, meas: Measurements, config: RunConfig):
    """The bundle's per-bin noise (``[run.fit.noise] estimate = "controls"``,
    :func:`hxex.noise.controls_noise`) from the undeuterated controls that every picker screen
    passes at the loose level (a control not assessed counts as passing), with ``b1`` per
    sample; a warning names the samples whose own ``b1`` differs from the bundle's (more than
    three standard errors), which a bundle mixing instruments or settings would show. None when
    the setting is off or there are too few controls (the fit then uses one sd per
    spectrum)."""
    from .analysis.screening import THRESHOLDS
    from .noise import controls_noise
    noise = config.fit.noise
    if noise.estimate != "controls":
        return None
    usable = np.ones(len(meas), dtype=bool)
    for rules in THRESHOLDS["loose"].values():
        for name, limit in rules:
            score = getattr(picked, name, None)
            if score is not None:
                usable &= ~(np.asarray(score, dtype=np.float64) >= limit)
    curve = controls_noise(meas.bins, meas.observed, meas.shapes, meas.role, meas.charges(),
                           meas.data_id, model_error=noise.model_error / 100.0, usable=usable,
                           sample=picked.sample)
    if curve is None:
        return None
    for s in curve.disagreeing_samples():
        log.warning("per-bin noise: sample %s's own counting scale (%.3g +- %.2g, %d series) "
                    "differs from the bundle's (%.3g +- %.2g): are its spectra from another "
                    "instrument or setting? (hxex info shows every sample's)", s.sample, s.b1,
                    s.b1_se, s.n_series, curve.b1, curve.b1_se)
    return curve


def _warn_own_reference(meas: Measurements, picked, config: RunConfig) -> None:
    """Warn when series are fitted without an undeuterated control: each of their spectra is
    then its own noise reference, so its noise is usually set lower than its series' and the
    number of populations can be over-estimated (guide: Analysing part of a dataset)."""
    from .noise import noise_reference_kinds

    if config.fit.noise.sd is not None:  # a fixed sd: the reference is not used
        return
    kinds = noise_reference_kinds(meas.data_id, meas.references, meas.role)
    own = sorted(s for s, k in kinds.items() if k == "own spectrum")
    if not own:
        return
    if len(own) == len(kinds):
        log.warning("no series has an undeuterated control in this run, so every spectrum is its "
                    "own noise reference (its noise set from its own tallest peak): the number of "
                    "populations can be over-estimated (guide: Analysing part of a dataset)")
        return
    first = {int(d): i for i, d in reversed(list(enumerate(picked.data_id)))}
    names = [f"{picked.sample[first[s]]} {picked.peptide_range[first[s]]} "
             f"{int(picked.charge[first[s]])}+" for s in own[:5]]
    log.warning("%d of %d series have no undeuterated control in this run, so their spectra are "
                "their own noise reference (noise set from their own tallest peak; the number of "
                "populations can be over-estimated): %s%s; series.csv noise_reference lists them",
                len(own), len(kinds), ", ".join(names), ", ..." if len(own) > 5 else "")


def _plan(data: Measurements, config: RunConfig, progress_range) -> _Plan:
    k_max = config.fit.k_max
    itemsize = config.fit.backend.torch_dtype().itemsize
    chunks = plan_chunks(data, k_max, itemsize, default_budget(config), config.max_chunk)
    log.info("%d measurements in %d chunk(s)", len(data), len(chunks))
    sizes = np.array([len(c) for c in chunks], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(sizes) / sizes.sum()])
    lo, hi = progress_range
    test = config.population_test
    w_test = (0.0 if test is None or test.null != "simulated"
              else sum(_TEST_WEIGHT + _TEST_WEIGHT_STEP * (k - 1)
                       for k in range(1, config.fit.k_max)) * test.simulations / 20.0)
    w_boot = _BOOTSTRAP_WEIGHT if config.bootstrap is not None else 0.0
    return _Plan(chunks, [data.select(rows) for rows in chunks], edges, lo, hi,
                 (1.0 + w_test) / (1.0 + w_test + w_boot), w_test / (1.0 + w_test))


def _d2o_fraction(config: RunConfig) -> float:
    """The labelling D2O fraction of the population test's width check."""
    return (config.controls.d2o_fraction if config.controls is not None
            else DEFAULT_D2O_FRACTION)


def _fit_part(
    data: Measurements, config: RunConfig, plan: _Plan, progress, cancel, mz_offset_ppm,
    rates=None, n_sites=None, rt=None, poor_envelope=None, groups=None, charges=None,
    fragments=(), envelope_offset=(None, None), noise_curve=None, screened=None, chosen=None,
    residues=None, deamidation=None,
) -> tuple[list[FitResult], FitResult, ControlTable | None, QualityFlags | None,
           PopulationTest | None]:
    """Fit every chunk (and test it, with the population test: each measurement's K is then
    the test's), then the controls and the quality flags of the merged fit (which do not
    need the bootstrap). ``n_sites`` is the exchangeable sites of every measurement
    (:func:`site_counts`); without it they are read back from ``data.n_max`` (rounded; an ion
    without any site then counts one). ``poor_envelope``: the series whose envelope is poor;
    ``groups``: the molecule of each series (:func:`hxex.controls.series_groups`);
    ``charges``: the peptide of each series and its charge (:func:`hxex.controls.charge_groups`);
    ``envelope_offset``: the picker's m/z check of every measurement, (offset, limit) in ppm
    (:attr:`hxex.picking.BinnedSet.envelope_offset_ppm`). ``noise_curve``: the bundle's per-bin
    noise (:func:`bundle_noise`); with ``estimate = "controls"`` and none given it is measured
    here from every undeuterated control of ``data``, once for all the chunks. ``screened``:
    the control spectra the spectrum quality screens remove at the loose level
    (:func:`hxex.analysis.screening.control_removal`), left out of the controls. ``chosen``
    (spectrum id -> K, a review table's choices): fitted whatever the screen says.
    ``deamidation``: the fully deuterated controls' deamidation
    (:func:`hxex.deamidation.deamidation_from_picked`), with ``[run.controls] deamidation``."""
    lo, hi, share, edges = plan.lo, plan.hi, plan.fit_share, plan.edges
    if noise_curve is None and config.fit.noise.estimate == "controls":
        from .noise import controls_noise
        noise_curve = controls_noise(data.bins, data.observed, data.shapes, data.role,
                                     data.charges(), data.data_id,
                                     model_error=config.fit.noise.model_error / 100.0)
    fits: list[FitResult] = []
    tests: list[PopulationTest] = []
    testing = config.population_test is not None
    start_stage(progress, "fit and population test" if testing else "fit", lo,
                lo + (hi - lo) * share)
    for j, sub in enumerate(plan.subs):
        check_cancel(cancel)
        a = lo + (hi - lo) * share * edges[j]
        b = lo + (hi - lo) * share * edges[j + 1]
        mid = a + (b - a) * (1.0 - plan.test_share)
        required = (np.array([int(chosen.get(str(i), 0)) for i in sub.spectrum_id])
                    if chosen else None)
        part = fit_models(sub, config.fit, progress=_Span(progress, a, mid), cancel=cancel,
                          test_screen=testing, noise_curve=noise_curve, required_k=required)
        if testing:
            test = population_test(sub, part, config.population_test,
                                   d2o_fraction=_d2o_fraction(config), test_screen=True,
                                   progress=_Span(progress, mid, b), cancel=cancel)
            part = dataclasses.replace(part, selected_K=test.selected_K.copy())
            tests.append(test)
        fits.append(part)
    fit = fits[0] if len(fits) == 1 else FitResult.merge(fits, plan.chunks)
    test = None
    if tests:
        test = tests[0] if len(tests) == 1 else PopulationTest.merge(tests, plan.chunks)
    check_cancel(cancel)
    at = lo + (hi - lo) * share
    start_stage(progress, "controls and quality", at, at)
    controls = None
    if config.controls is not None:
        if n_sites is None:
            n_sites = np.rint(np.asarray(data.n_max, dtype=float) / config.n_max_factor)
        controls = fit_controls(fit, n_sites, config.controls, rates=rates, rt=rt,
                                groups=groups, charges=charges, uncorrected=fragments,
                                screened=screened, residues=residues,
                                deamidation=deamidation)
        if screened is not None and screened.any():
            log.info("controls: %d control spectra left out by the spectrum quality screens "
                     "(the loose level of [analysis.filters] spectrum_quality_filtering)",
                     int(screened.sum()))
    quality = None
    if config.quality is not None:
        quality = assess_quality(data, fit, config.quality, mz_offset_ppm=mz_offset_ppm,
                                 poor_envelope_series=poor_envelope,
                                 envelope_offset_ppm=envelope_offset[0],
                                 envelope_offset_limit_ppm=envelope_offset[1])
    return fits, fit, controls, quality, test


def _bootstrap_part(
    config: RunConfig, plan: _Plan, fits: Sequence[FitResult], progress, cancel
) -> tuple[BootstrapResult | None, dict[int, BootstrapResult] | None]:
    """Bootstrap every chunk from its fit (``fits``, one per chunk) and merge."""
    if config.bootstrap is None:
        return None, None
    lo, hi, share, edges = plan.lo, plan.hi, plan.fit_share, plan.edges
    boots: list[BootstrapResult] = []
    fixed: list[dict[int, BootstrapResult]] = []
    ks = _bootstrap_ks(config, fits[0])
    skipped = [k for k in ks if not all(f.fitted(k).all() for f in fits)]
    if skipped:
        log.warning("the screen skipped K = %s for some spectra: the fixed-K bootstrap has no "
                    "values for them (fit without FitConfig.screen for a fixed-K analysis)",
                    ", ".join(map(str, skipped)))
    start_stage(progress, "bootstrap", lo + (hi - lo) * share, hi)
    for j, (sub, fit) in enumerate(zip(plan.subs, fits, strict=True)):
        check_cancel(cancel)
        a = lo + (hi - lo) * (share + (1 - share) * edges[j])
        b = lo + (hi - lo) * (share + (1 - share) * edges[j + 1])
        hook = _Span(progress, a, b)
        if not ks:
            boots.append(bootstrap(sub, fit, config.bootstrap, progress=hook, cancel=cancel))
            continue
        # every selected K must be covered: add the missing ones (not requested for output)
        extra = sorted(set(int(k) for k in fit.selected_K) - set(ks))
        done = bootstrap_by_k(sub, fit, [*ks, *extra], config.bootstrap, progress=hook,
                              cancel=cancel)
        boots.append(selected_from(fit, done))
        fixed.append({k: done[k] for k in ks})
    boot = boots[0] if len(boots) == 1 else BootstrapResult.merge(boots, plan.chunks)
    by_k = None
    if fixed:
        by_k = {k: fixed[0][k] if len(fixed) == 1
                else BootstrapResult.merge([f[k] for f in fixed], plan.chunks) for k in ks}
    return boot, by_k


def _chosen_part(config: RunConfig, data: Measurements, fit: FitResult,
                 by_k: dict[int, BootstrapResult] | None, chosen: Mapping[str, int],
                 progress, cancel) -> dict[int, BootstrapResult] | None:
    """``by_k`` with a bootstrap added, at their chosen K, for the spectra of ``chosen``
    (spectrum id -> K: a review's chosen K) that have none there: a partial fixed-K bootstrap
    (:func:`hxex.bootstrap.scatter`), so a K chosen in the review is shown with errors made
    like every other spectrum's. Spectra the fit does not have, at their selected K, or at a K
    not fitted for them are passed over."""
    if config.bootstrap is None or not chosen:
        return by_k
    wanted = _missing_choices(fit, by_k, chosen)
    if not wanted:
        return by_k
    by_k = dict(by_k or {})
    total = sum(len(r) for r in wanted.values())
    log.info("bootstrapping %d spectra at the K chosen for them in the review", total)
    start_stage(progress, "bootstrap chosen K", UNITS - _FINISH, UNITS - _FINISH)
    for k, rows in sorted(wanted.items()):
        check_cancel(cancel)
        rows = np.sort(np.asarray(rows, dtype=np.int64))
        part = bootstrap_by_k(data.select(rows, drop_references=True), fit.select(rows), [k],
                              config.bootstrap, cancel=cancel)[k]
        by_k[k] = scatter(part, rows, len(fit), k, base=by_k.get(k))
    return by_k


def _missing_choices(fit: FitResult, by_k: Mapping[int, BootstrapResult] | None,
                     chosen: Mapping[str, int]) -> dict[int, list[int]]:
    """The rows of ``fit`` whose chosen K (``chosen``, spectrum id -> K) has no bootstrap in
    ``by_k``, by K (see :func:`_chosen_part`)."""
    by_k = by_k or {}
    index = {str(s): i for i, s in enumerate(fit.spectrum_id)}
    wanted: dict[int, list[int]] = {}
    for sid, k in chosen.items():
        i = index.get(str(sid))
        if (i is None or not k or k == fit.selected_K[i] or k not in fit.k_values
                or not fit.fitted(k)[i] or (k in by_k and by_k[k].covers(i))):
            continue
        wanted.setdefault(int(k), []).append(i)
    return wanted


def _unfitted_choices(fit: FitResult, chosen: Mapping[str, int]) -> np.ndarray:
    """For each row of ``fit``, the K chosen for it (``chosen``, spectrum id -> K) when that K,
    or a K between 2 and it, was not fitted for it (the screen skipped it); else 0. A K the
    run never tried is not counted (it cannot be fitted without the run's settings)."""
    index = {str(s): i for i, s in enumerate(fit.spectrum_id)}
    need = np.zeros(len(fit), dtype=np.int64)
    for sid, k in chosen.items():
        i = index.get(str(sid))
        if i is None or not k or k not in fit.fits:
            continue
        if any(not fit.fitted(j)[i] for j in fit.k_values if 1 < j <= k):
            need[i] = int(k)
    return need


def _fit_chosen(config: RunConfig, data: Measurements, fit: FitResult,
                chosen: Mapping[str, int], cancel) -> FitResult:
    """``fit`` with the K chosen in a review table (``chosen``) fitted for the spectra the screen
    skipped it for, and every K between 2 and it, as a run with the table would have fitted
    them: their whole series are fitted again without the screen (each measurement is seeded
    from its own id) and only the missing fits are taken; every other fit and the selected K
    stay as they were."""
    need = _unfitted_choices(fit, chosen)
    if not need.any():
        return fit
    rows = np.flatnonzero(np.isin(fit.data_id, np.unique(fit.data_id[need > 0])))
    log.info("fitting the K chosen in the review table for %d spectra the screen skipped it "
             "for (their %d series fitted again without the screen)", int((need > 0).sum()),
             len(np.unique(fit.data_id[rows])))
    sub = dataclasses.replace(config.fit, screen=False, k_max=int(need.max()))
    refit = fit_models(data.select(rows), sub, cancel=cancel, noise_curve=fit.noise_curve)
    fits, score = dict(fit.fits), fit.score.copy()
    for k in fit.k_values:
        if k < 2 or k > need.max():
            continue
        cells = np.flatnonzero((k <= need[rows]) & ~fit.fitted(k)[rows])
        if not len(cells):
            continue
        new, at = refit.fits[k], rows[cells]
        arrays = {name: getattr(fits[k], name).copy() for name in _KFIT_ARRAYS}
        for name, values in arrays.items():
            values[at] = getattr(new, name)[cells]
        fits[k] = dataclasses.replace(fits[k], **arrays)
        criterion = getattr(new, fit.config.criterion)[cells]
        score[at, fit.k_values.index(k)] = np.where(new.passes_min_pop[cells], criterion,
                                                    np.inf)
    return dataclasses.replace(fit, fits=fits, score=score)


def _review_choices(review) -> dict[str, int]:
    """The K a (carried-over or stored) review layer chooses, by spectrum (current entries)."""
    if review is None:
        return {}
    return {str(s): int(k) for s, k, stale in zip(review.spectrum_id, review.k_override,
                                                  review.stale, strict=True)
            if k and not stale}


def analyse(
    data: Measurements,
    config: RunConfig,
    *,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
    progress_range: tuple[float, float] = (0.0, float(UNITS)),
    mz_offset_ppm: np.ndarray | None = None,
    envelope_offset_ppm: np.ndarray | None = None,
    envelope_offset_limit_ppm: np.ndarray | None = None,
) -> tuple[
    FitResult,
    BootstrapResult | None,
    dict[int, BootstrapResult] | None,
    ControlTable | None,
    QualityFlags | None,
]:
    """Fit (and bootstrap) ``data`` in chunks of whole series, then the controls and the quality
    flags of the merged result (``mz_offset_ppm``, the picker's offset of every measurement,
    for the m/z screen; ``envelope_offset_ppm`` and ``envelope_offset_limit_ppm``, the
    picker's m/z check, for ``off_mz``). The chunking does not change the result."""
    plan = _plan(data, config, progress_range)
    fits, fit, controls, quality, _ = _fit_part(
        data, config, plan, progress, cancel, mz_offset_ppm,
        envelope_offset=(envelope_offset_ppm, envelope_offset_limit_ppm))
    boot, by_k = _bootstrap_part(config, plan, fits, progress, cancel)
    return fit, boot, by_k, controls, quality


def with_control_roles(data, config: RunConfig):
    """``data`` (a :class:`~hxex.spectra.SpectrumSet` or a :class:`~hxex.picking.BinnedSet`)
    with the spectra at the controls' ``undeut_at`` and ``fulldeut_at`` given that role as well
    (:func:`hxex.controls.apply_control_roles`); unchanged when neither is set."""
    c = config.controls
    if c is None or (c.undeut_at is None and c.fulldeut_at is None):
        return data
    return dataclasses.replace(data, role=apply_control_roles(data.role, data.time, c,
                                                              where="[run.controls]"))


def site_counts(binned: BinnedSet, config: RunConfig) -> np.ndarray:
    """The exchangeable sites of every spectrum of ``binned``, ``(M,)`` integers: the backbone
    amides of its peptide and ion type plus its user ``Hex`` sites (they stand for exchangeable
    amides on a modification), as :meth:`hxex.fit.Measurements.from_binned` counts
    them for ``n_max`` (0 for an ion without any site, which the controls flag ``no_amides``)."""
    cache: dict[tuple[str, str], int] = {}
    out = np.zeros(len(binned), dtype=np.int64)
    for i, (pep, mod) in enumerate(zip(binned.peptide, binned.modification, strict=True)):
        key = (str(pep), str(mod))
        if key not in cache:
            spec = chem.parse_modification(key[1])
            cache[key] = chem.exchangeable_sites(
                key[0], mod=spec, ion_type=spec.ion_type,
                nterm_subtract=config.picker.nterm_subtract, rules=config.picker.amide_rules,
                registry=config.registry)
        out[i] = cache[key]
    return out


def control_rates(binned: BinnedSet, config: RunConfig) -> dict | None:
    """The intrinsic rates at the quench conditions of every series' amides, when the controls
    fall back to theoretical back-exchange (``fd_fallback='kint'``); else None."""
    c = config.controls
    if c is None or c.fd_mode != "kint":
        return None
    return series_rates(binned.peptide, binned.modification, binned.data_id, c.quench,
                        nterm_subtract=config.picker.nterm_subtract,
                        rules=config.picker.amide_rules, registry=config.registry)


def run(
    data: SpectrumSet | BinnedSet,
    config: RunConfig = RunConfig(),  # noqa: B008
    *,
    out: str | Path | None = None,
    project: str | None = None,
    notes: str = "",
    overwrite: bool = False,
    keep_previous: bool = False,
    keep_spectra: bool = True,
    exclude: Sequence = (),
    view_exclude: Sequence = (),
    project_file: Mapping | None = None,
    chosen_k: Mapping[str, int] | None = None,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> RunResult:
    """Run the whole analysis on raw spectra (picked here) or on an already picked set.

    ``exclude`` (:class:`hxex.exclusions.Exclusion` rules, ``[[input.exclude]]``) leaves the
    spectra it matches out before anything else: they are stored with the raw spectra but not
    picked or fitted, and the bundle records the rules and the spectra they left out.
    ``view_exclude`` (``[[review.exclude]]``) treats its spectra as rejected in the figures
    of :meth:`RunResult.write_figures` (in memory; the fits are kept). A run draws no figures.

    Spectra flagged ``empty`` (and ``no_signal`` if ``config.drop_no_signal``), and spectra without
    a bin, are dropped and recorded. With ``out`` the results bundle is written there (with the raw
    spectra if the input was a :class:`SpectrumSet` and ``keep_spectra``). An existing ``out`` is
    refused before anything is fitted unless ``overwrite`` or ``keep_previous``. With
    ``overwrite`` the bundle there is replaced (its files; files of your own in the directory
    stay) and its review layer carried over, with the entries this run cannot apply marked
    stale (see :class:`~hxex.io.ReviewTable`); with ``keep_previous`` it is first moved to its next
    generation (``name.N``, with its ``name_tables`` and ``name_figures`` folders; see
    :func:`hxex.io.keep_previous`) just before the new bundle is written. Either way the new
    manifest records what was replaced, what changed and the review carried (its
    ``provenance``). ``project_file`` (:func:`hxex.config.project_record`) records the
    project file's other settings in the bundle, beside the run settings. With a bootstrap, the
    spectra shown at a K other than the fit's choice (``chosen_k``, spectrum id -> K, from a
    review table, and the carried-over review's chosen K) are bootstrapped at that K as well,
    so their errors are made like every other's. Raises ValueError if nothing is left.
    """
    previous, review = (_previous(out, overwrite or keep_previous) if out is not None
                        else (None, None))
    # every spectrum, for the bundle, with the readers' roles: the promotion of [run.controls]
    # undeut_at/fulldeut_at is applied again from the settings of every run
    stored = data if isinstance(data, SpectrumSet) else None
    data = with_control_roles(data, config)
    excluded = None
    if exclude:
        from .exclusions import exclusion_mask, exclusion_record

        reader = (data.source or {}).get("reader") if isinstance(data, SpectrumSet) else None
        mask, counts = exclusion_mask(exclude, data, reader, "input.exclude")
        excluded = exclusion_record(exclude, counts, np.asarray(data.spectrum_id)[mask])
        if mask.all():
            raise ValueError("[[input.exclude]] leaves no spectrum to fit")
        if mask.any():
            log.info("[[input.exclude]]: %d of %d spectra left out (stored, not fitted)",
                     int(mask.sum()), len(mask))
            data = data.select(~mask)
    raw = data if isinstance(data, SpectrumSet) else None
    if raw is not None and raw.source and raw.source.get("stored_in"):
        _check_stored_windows(raw.source, config)
    if config.envelope.mode != "theory" and raw is None:
        raise ValueError(f"envelope mode {config.envelope.mode!r} needs the raw spectra "
                         "(a SpectrumSet), not a picked set")
    pick_share = 0.05 if raw is not None else 0.0
    envelope_share = 0.03 if raw is not None and config.envelope.mode != "theory" else 0.0
    envelopes, clusters, shapes = None, None, None
    measure = None
    stored_picking = _stored_picking(raw, config) if raw is not None else None
    if stored_picking is not None:  # the bundle's picking, as picking again would give it
        binned, measure, envelopes = stored_picking
        if envelopes is not None:
            shapes = {int(d): envelopes.fit_shape[envelopes.shape_offsets[i]:
                                                  envelopes.shape_offsets[i + 1]]
                      for i, d in enumerate(envelopes.data_id)}
        raw_for_picking = None
    else:
        raw_for_picking = raw
    if raw_for_picking is not None:
        measure = choose_measure(raw, config.picker, registry=config.registry)
        log.info("picking: the bins take %s", measure.describe())
    if raw_for_picking is not None and config.envelope.mode != "theory":
        groups = None
        if config.envelope.share:
            from .io.mzml import envelope_groups

            groups = envelope_groups(raw.source)
            if not groups:
                log.warning("envelope share: the spectra were not read from scan files, so "
                            "every series keeps its own envelope")
        start_stage(progress, "envelope", 0.0, UNITS * envelope_share)
        envelopes, by_series = series_envelopes(
            raw, config=config.envelope,
            fallback_time=config.envelope_fallback_time, picker=config.picker,
            registry=config.registry, groups=groups,
            progress=_Span(progress, 0.0, UNITS * envelope_share), cancel=cancel,
            measure=measure,
        )
        clusters = {d: e.clusters for d, e in by_series.items()}
        shapes = {d: e.fit_shape for d, e in by_series.items()}
        n_back = int(np.isin(envelopes.source, ("fallback", "theory")).sum())
        if n_back:
            log.warning("%d of %d series took their envelope from the fallback or the theory",
                        n_back, len(envelopes))
        if config.envelope.share:
            shared = envelopes.source == "shared"
            log.info("%d series share an envelope with another condition, %d without "
                     "references of their own", int(shared.sum()),
                     int((shared & (envelopes.n_reference == 0)).sum()))
        n_poor = int(envelopes.poor.sum())
        if n_poor:
            log.warning("%d of %d series have a poor envelope (flagged poor_envelope)", n_poor,
                        len(envelopes))
    if raw_for_picking is not None:
        start_stage(progress, "pick", UNITS * envelope_share,
                    UNITS * (envelope_share + pick_share))
        binned = pick_spectra(
            raw, config.picker, registry=config.registry, clusters=clusters,
            progress=_Span(progress, UNITS * envelope_share,
                           UNITS * (envelope_share + pick_share)), cancel=cancel,
            measure=measure,
        )
        pick_share += envelope_share
    elif stored_picking is None:
        binned = data
    drop_empty = np.asarray(binned.empty, dtype=bool)
    drop_nosig = np.asarray(binned.no_signal, dtype=bool) & config.drop_no_signal
    has_bins = np.diff(binned.offsets) > 0
    kept = ~drop_empty & ~drop_nosig & has_bins
    reason = _drop_reasons(binned, kept)
    if not kept.any():
        raise ValueError("every spectrum was dropped (empty, without signal or without bins)")
    if (~kept).any():
        log.warning("dropped %d of %d spectra: %d empty, %d without signal, %d without bins",
                    int((~kept).sum()), len(kept), int((reason == "empty").sum()),
                    int((reason == "no_signal").sum()), int((reason == "no_bins").sum()))
    picked = binned.select(kept)
    from .io.mzml import kinetics_spectra

    measurements = Measurements.from_binned(
        picked, n_max_factor=config.n_max_factor,
        nterm_subtract=config.picker.nterm_subtract,
        amide_rules=config.picker.amide_rules, registry=config.registry, shapes=shapes,
        kinetics=None if raw is None else kinetics_spectra(raw.source, picked.spectrum_id),
    )
    _warn_own_reference(measurements, picked, config)
    noise_curve = bundle_noise(picked, measurements, config)
    plan = _plan(measurements, config, (UNITS * pick_share, float(UNITS - _FINISH)))
    rates = control_rates(picked, config)
    rt = series_rt(picked.data_id, picked.rt) if rates is not None else None
    groups = series_groups(picked.peptide, picked.modification, picked.charge, picked.data_id)
    from .io.mzml import fragment_series

    scan_fragments = fragment_series(None if raw is None else raw.source)
    charges = charge_groups(picked.sample, picked.peptide, picked.modification, picked.charge,
                            picked.data_id, scan_fragments)
    fragments = fragment_ids(picked.modification, picked.data_id, scan_fragments)
    residues = series_residues(picked.peptide, picked.modification, picked.start_seq,
                               picked.data_id, nterm_subtract=config.picker.nterm_subtract,
                               rules=config.picker.amide_rules, registry=config.registry,
                               fragments=scan_fragments)
    from .analysis.screening import control_removal

    screened = control_removal(picked, fragments)
    deamidation = None
    if config.controls is not None and config.controls.deamidation:
        from .deamidation import deamidation_from_picked
        deamidation = deamidation_from_picked(
            picked, measurements.spectrum_id, measurements.role, ~screened, charges, residues,
            min_share=config.controls.deamidation_min_share, registry=config.registry)
    poor = None if envelopes is None else envelopes.data_id[envelopes.poor]
    fits, fit, controls, quality, test = _fit_part(measurements, config, plan, progress, cancel,
                                             binned.mz_offset_ppm[kept], rates,
                                             site_counts(picked, config), rt=rt,
                                             poor_envelope=poor, groups=groups,
                                             charges=charges, fragments=fragments,
                                             residues=residues, deamidation=deamidation,
                                             screened=screened,
                                             envelope_offset=(
                                                 binned.envelope_offset_ppm[kept],
                                                 binned.envelope_offset_limit_ppm[kept]),
                                             noise_curve=noise_curve,
                                             chosen=merge_choices(chosen_k))
    result = RunResult(
        config=config, binned=binned, kept=kept, dropped_reason=reason,
        measurements=measurements, fit=fit, bootstrap=None, bootstrap_by_k=None,
        controls=controls, quality=quality, envelopes=envelopes,
        spectra_source=None if raw is None or raw.source is None else dict(raw.source),
        excluded=excluded, view_exclude=tuple(view_exclude),
        project_file=None if project_file is None else dict(project_file),
        population_test=test, peak_measure=measure,
    )
    spectra = stored if keep_spectra else None
    review = _review_for(review, fit)
    provenance = None
    if previous is not None:
        from .io.bundle import keep_previous as move_previous
        from .io.bundle import provenance_of

        moved = move_previous(out) if keep_previous else None
        provenance = provenance_of(
            previous, out, moved_to=moved, review=review,
            new_configs={"run": config.to_dict(), "spectra_source": result.spectra_source})
    checkpoint = out is not None and config.bootstrap is not None
    if checkpoint:  # the fits are kept if the bootstrap is stopped or fails
        check_cancel(cancel)
        at = plan.lo + (plan.hi - plan.lo) * plan.fit_share
        start_stage(progress, "write fits", at, at)
        result.write(out, spectra=spectra, project=project, notes=notes, overwrite=overwrite,
                     pending=("bootstrap",), review=review, provenance=provenance)
        overwrite = True
    try:
        boot, by_k = _bootstrap_part(config, plan, fits, progress, cancel)
    except BaseException:
        if checkpoint:
            log.warning("the bootstrap did not finish: the fits are in %s; 'hxex resume' (or "
                        "hxex.run.resume_bootstrap) adds the bootstrap to them", out)
        raise
    by_k = _chosen_part(config, measurements, fit, by_k,
                        merge_choices(_review_choices(review), chosen_k), progress, cancel)
    result = dataclasses.replace(result, bootstrap=boot, bootstrap_by_k=by_k)
    return _finish(result, out, spectra=spectra, project=project, notes=notes,
                   overwrite=overwrite, progress=progress, cancel=cancel, review=review,
                   provenance=provenance)


def _stored_picking(raw: SpectrumSet, config: RunConfig):
    """The picking of the bundle the raw spectra were taken from (``[input] spectra``), when
    picking them again would give it: ``(binned, peak measure, envelopes)``, or None. It is
    taken when the bundle's ``[run.picker]``, ``[run.envelope]`` (with its fallback time) and
    modification registry are the run's, its picked spectra are these (after
    ``[[input.exclude]]``, in this order) with these control roles (after ``undeut_at`` and
    ``fulldeut_at``), and it was written by this version of hxex; otherwise the log says what
    differs and everything is picked again."""
    from .io.bundle import BundleError, _versions, open_bundle

    where = (raw.source or {}).get("stored_in")
    if not where or not Path(where).exists():
        return None
    try:
        with open_bundle(where) as b:
            if not b.has("binned"):
                return None
            stored = (b.manifest.get("configs") or {}).get("run") or {}
            now = json.loads(json.dumps(config.to_dict()))
            differ = [label for key, label in (
                ("picker", "[run.picker]"), ("envelope", "[run.envelope]"),
                ("envelope_fallback_time", "[run] envelope_fallback_time"),
                ("registry", "the modification registry"))
                if now.get(key) != stored.get(key)]
            written = (b.manifest.get("versions") or {}).get("hxex")
            if written != _versions()["hxex"]:
                differ.append(f"the hxex version ({written} then)")
            binned = b.binned()
            if list(map(str, binned.spectrum_id)) != list(map(str, raw.spectrum_id)):
                differ.append("the spectra picked")
            elif list(map(str, binned.role)) != list(map(str, raw.role)):
                differ.append("the control roles")
            if differ:
                log.info("picking the stored spectra again: %s differ(s) from the picking "
                         "stored in %s", ", ".join(differ), where)
                return None
            measure, envelopes = b.peak_measure(), b.envelopes()
    except BundleError:
        return None
    if config.envelope.mode != "theory" and envelopes is None:
        return None
    log.info("the picking stored in %s is used (the same spectra and picking settings)",
             where)
    return binned, measure, envelopes


def _drop_reasons(binned, kept: np.ndarray) -> np.ndarray:
    """Why each spectrum not ``kept`` was dropped: ``empty`` (nothing picked), ``no_bins`` (no
    bin to fit: the peptide's plan has none), else ``no_signal``; empty for those kept."""
    empty = np.asarray(binned.empty, dtype=bool)
    no_bins = np.diff(binned.offsets) == 0
    return np.where(kept, "", np.where(empty, "empty", np.where(no_bins, "no_bins",
                                                                  "no_signal"))).astype(str)


def _previous(out, replace: bool):
    """Refuse an existing ``out`` unless ``replace`` (before anything is fitted, as the CLI
    does); with it, ``(manifest, review)`` of the bundle there (``(None, None)`` when there is
    nothing there; a bundle without a review gives ``(manifest, None)``). Something that is
    not a bundle is refused: it is never replaced or moved."""
    from .io.bundle import BundleError, bundle_manifest, open_bundle

    path = Path(out)
    if not path.exists():
        return None, None
    if not replace:
        raise BundleError(f"{path} exists (use overwrite=True to replace a bundle, or "
                          "keep_previous=True to keep it as an earlier generation)")
    manifest = bundle_manifest(path)
    if manifest is None:
        raise BundleError(f"{path} exists but is not a results bundle; not replacing it")
    try:
        with open_bundle(path) as b:
            review = b.review() if b.has("review") else None
    except BundleError:
        review = None
    return manifest, review


def _review_for(review, fit: FitResult):
    """``review`` with the entries the new run cannot apply (spectra it does not have, a chosen
    K it did not fit) marked stale: their status and note stay on record, the chosen K is
    ignored (see :class:`~hxex.io.ReviewTable`). One log line with the counts; None stays
    None."""
    if review is None or not len(review):
        return review
    review = review.mark_stale(fit)
    log.info("carrying %d review entries over to the new bundle, %d stale (their spectrum was "
             "dropped or their K not fitted: status and note kept, chosen K ignored)",
             len(review), review.n_stale)
    return review


def _check_stored_windows(source: Mapping, config: RunConfig) -> None:
    """Warn when the picker's bins reach beyond the m/z the stored spectra were cut to."""
    from .io.mzml import stored_window_excess

    outside = stored_window_excess(source, config.picker, config.registry)
    if outside:
        log.warning("%d target(s) need m/z outside the window the stored spectra were cut to "
                    "(the picker's bins or margin grew since they were read): points outside "
                    "it cannot be recovered; read the files again for them, for example %s",
                    len(outside), outside[0])


def _finish(result: RunResult, out, *, spectra, project, notes, overwrite, progress, cancel,
            review=None, provenance=None) -> RunResult:
    """Write the bundle (if ``out``) with the carried-over ``review`` (kept on the result);
    ``provenance`` is recorded in the bundle."""
    result = dataclasses.replace(result, review=review)
    if out is not None:
        check_cancel(cancel)
        start_stage(progress, "write", UNITS - _FINISH, UNITS)
        result.write(out, spectra=spectra, project=project, notes=notes, overwrite=overwrite,
                     review=review, provenance=provenance)
    report(progress, UNITS, UNITS)
    return result


class NothingToResume(ValueError):
    """A complete bundle with every chosen K bootstrapped: :func:`resume_bootstrap` has nothing
    to add."""


def resume_bootstrap(
    path: str | Path,
    *,
    bootstrap: BootstrapConfig | None = None,
    device: str | None = None,
    add: bool = False,
    view_exclude: Sequence = (),
    chosen_k: Mapping[str, int] | None = None,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> RunResult:
    """Add the bootstrap to the bundle a run wrote before its bootstrap (its ``pending`` is
    ``['bootstrap']``: the run was stopped or failed during the bootstrap), and rewrite it
    complete. The fits are not repeated, and the result is the one the run would have given
    (each measurement is seeded from its own id). The bootstrap settings are the run's unless
    ``bootstrap`` is given; ``device`` (``'cpu'``, ``'cuda'``, ...) replaces the run's, so a
    bundle fitted on a GPU can be resumed where there is none. The stored spectra, notes and
    review are kept. No figures are drawn; ``view_exclude`` (``[[review.exclude]]`` rules) is
    kept on the result for :meth:`RunResult.write_figures`.

    With ``add`` (and ``bootstrap``, its settings) a complete bundle takes a bootstrap too: one
    run without a bootstrap gets one, and a bundle that has one has it replaced (more
    resamples, a fixed-K bootstrap for the analysis); the fits are kept and the bundle's run
    settings record the new bootstrap.

    Either way, and on its own for a complete bundle, the spectra shown at a K other than the
    fit's choice (``chosen_k``, spectrum id -> K, from a review table, and the bundle's own
    review) are bootstrapped at that K where they have no bootstrap there (:func:`run` does
    the same); a K a review table chose that the run's screen skipped for its spectrum is
    fitted first (as a run with the table fits it). A complete bundle with nothing to add is
    refused (ValueError)."""
    from .io.bundle import open_bundle

    path = Path(path)
    with open_bundle(path) as b:
        redo = "bootstrap" in b.pending or add
        chosen = merge_choices(_review_choices(b.review() if b.has("review") else None),
                               chosen_k)
        held = None if redo else {k: b.bootstrap(k) for k in b.bootstrap_ks()}
        unfitted = _unfitted_choices(b.fit(), merge_choices(chosen_k)).any()
        if not redo and not unfitted and (b.bootstrap() is None
                                          or not _missing_choices(b.fit(), held, chosen)):
            raise NothingToResume(f"{path}: no bootstrap is pending (the bundle is complete) and "
                             "every K chosen in the review has one; to add or redo the "
                             "bootstrap, give the project file with its [run.bootstrap] "
                             "(hxex resume hxex.toml)")
        if add and bootstrap is None:
            raise ValueError("adding a bootstrap needs its settings (bootstrap=)")
        if add and "bootstrap" not in b.pending:
            had = b.bootstrap() is not None
            log.info("%s: %s the bootstrap (%d resamples); the fits are kept", path,
                     "replacing" if had else "adding", bootstrap.n_boot)
        if b.manifest["configs"].get("run") is None:
            raise ValueError(f"{path}: the bundle does not record its run settings")
        config = RunConfig.from_dict(b.manifest["configs"]["run"])
        if bootstrap is not None:
            config = dataclasses.replace(config, bootstrap=bootstrap)
        if config.bootstrap is None and (redo or b.bootstrap() is not None):
            raise ValueError("the run had no bootstrap settings: give bootstrap=")
        measurements, fit, binned = b.measurements(), b.fit(), b.binned()
        if device is not None:  # the bootstrap takes its backend from the stored fit
            backend = dataclasses.replace(fit.config.backend, device=device)
            config = dataclasses.replace(
                config, fit=dataclasses.replace(config.fit, backend=backend))
            fit = dataclasses.replace(fit, config=dataclasses.replace(fit.config, backend=backend))
        if binned is None:
            raise ValueError(f"{path}: the bundle has no picked spectra")
        controls, quality, envelopes = b.controls(), b.quality(), b.envelopes()
        spectra = b.spectra() if b.has("spectra") else None
        review = b.review() if b.has("review") else None
        source, project, notes = b.spectra_source(), b.project, b.manifest.get("notes", "")
        provenance = b.manifest.get("provenance")
        excluded = b.manifest.get("excluded")
        peak_measure = b.peak_measure()
        project_file = b.manifest["configs"].get("project_file")
        stored_boot = None if redo else b.bootstrap()
        test = b.population_test()
    kept = np.isin(binned.spectrum_id, measurements.spectrum_id)
    reason = _drop_reasons(binned, kept)
    refitted = _fit_chosen(config, measurements, fit, merge_choices(chosen_k), cancel)
    if refitted is not fit and review is not None:
        review = review.mark_stale(refitted)  # a chosen K now fitted is current again
    fit = refitted
    if redo:
        plan = dataclasses.replace(_plan(measurements, config, (0.0, float(UNITS - _FINISH))),
                                   fit_share=0.0)
        boot, by_k = _bootstrap_part(config, plan, [fit.select(rows) for rows in plan.chunks],
                                     progress, cancel)
    else:
        boot, by_k = stored_boot, held or None
    by_k = _chosen_part(config, measurements, fit, by_k, chosen, progress, cancel)
    result = RunResult(
        config=config, binned=binned, kept=kept, dropped_reason=reason,
        measurements=measurements, fit=fit, bootstrap=boot, bootstrap_by_k=by_k,
        controls=controls, quality=quality, envelopes=envelopes, spectra_source=source,
        excluded=excluded, view_exclude=tuple(view_exclude), project_file=project_file,
        population_test=test, peak_measure=peak_measure,
    )
    return _finish(result, path, spectra=spectra, project=project, notes=notes, overwrite=True,
                   progress=progress, cancel=cancel, review=review, provenance=provenance)
