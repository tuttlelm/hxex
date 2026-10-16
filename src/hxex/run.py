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
    selected_from,
)
from .controls import ControlConfig, ControlTable, fit_controls, series_rates, series_rt
from .fit import FitConfig, FitResult, Measurements, fit_models
from .hooks import CancelHook, ProgressHook, check_cancel, report, start_stage
from .picking import (
    DEFAULT_PICKER_CONFIG,
    BinnedSet,
    EnvelopeConfig,
    EnvelopeTable,
    PickerConfig,
    pick_spectra,
    series_envelopes,
)
from .quality import QualityConfig, QualityFlags, assess_quality
from .spectra import SpectrumSet

if TYPE_CHECKING:
    from .report.figures import PlotConfig
    from .report.series import Reference

log = logging.getLogger(__name__)

UNITS = 1000  # the ``total`` of the progress hook
_FINISH = 10  # the last units: controls, quality flags, writing, figures
CPU_BUDGET = 4 * 2**30  # bytes per chunk on the CPU unless set
GPU_SHARE = 0.5  # share of the free GPU memory used by one chunk unless set
# memory of the fit (the bootstrap needs less) in units of one pmf array (shape points x bins x
# components per measurement). Bins that are whole numbers (every picked spectrum) use the
# integer convolution: peaks of 7.6 to 9.2 arrays measured on the CPU and the GPU for 945 and
# 3,727 measurements (docs/HISTORY.md item 46), so 12. Other bins evaluate the pmf at every
# shifted bin: 5.2 to 5.9 times 12 arrays on the GPU for 94 to 542 Bimodal measurements, so 75.
WORK_FACTOR = 12.0
WORK_FACTOR_DIRECT = 75.0


@dataclass(frozen=True)
class RunConfig:
    """Settings of :func:`run`.

    ``controls`` None means no back-exchange correction (K=1 must then not be required). The
    reference spectra of the noise level are those at the controls' undeuterated time if given,
    else at ``undeut_time``. ``n_max_factor`` times the exchangeable sites bounds ``n``.
    ``drop_no_signal`` drops spectra the picker found nothing in (``empty`` ones are always
    dropped). ``memory_budget_bytes`` and ``max_chunk`` (measurements) limit a chunk; None means
    automatic (see the module docstring) and no cap.
    """

    picker: PickerConfig = DEFAULT_PICKER_CONFIG
    fit: FitConfig = FitConfig()  # noqa: RUF009
    bootstrap: BootstrapConfig | None = None
    controls: ControlConfig | None = None
    quality: QualityConfig | None = QualityConfig()  # noqa: RUF009
    plots: PlotConfig | None = None
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS
    envelope: EnvelopeConfig = EnvelopeConfig()  # noqa: RUF009
    envelope_fallback_time: float | None = None
    undeut_time: float = 0.0
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
        if self.bootstrap is not None and not isinstance(self.bootstrap.ks, str):
            missing = [k for k in self.bootstrap.ks if k not in self.fit.k_values]
            if missing:
                raise ValueError(f"bootstrap.ks {missing} are outside the fitted K values")
        if self.controls is not None and self.fit.k_min > 1:
            raise ValueError("the controls come from the K=1 fits: set fit.k_min to 1")

    @property
    def reference_time(self) -> float:
        if self.controls is not None and self.controls.undeut_time is not None:
            return float(self.controls.undeut_time)
        return self.undeut_time

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = v.to_dict() if hasattr(v, "to_dict") else v
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunConfig:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown run setting(s): {', '.join(sorted(unknown))}")
        from .report.figures import PlotConfig

        parts = {
            "plots": PlotConfig, "picker": PickerConfig, "fit": FitConfig,
            "bootstrap": BootstrapConfig, "controls": ControlConfig, "quality": QualityConfig,
            "registry": chem.ModificationRegistry, "envelope": EnvelopeConfig,
        }
        kwargs = {
            k: (parts[k].from_dict(v) if k in parts and v is not None else v)
            for k, v in data.items()
        }
        return cls(**kwargs)


@dataclass(frozen=True)
class RunResult:
    """What a run produced. ``binned`` is every picked spectrum (flags included); the spectra
    where ``kept`` is False were dropped and are not in ``measurements`` or ``fit``;
    ``dropped_reason`` says why (``empty`` or ``no_signal``). ``bootstrap`` is at each
    measurement's selected K; ``bootstrap_by_k`` (when ``BootstrapConfig.ks`` asks for it) has
    the fixed-K bootstrap of every measurement at each K requested. ``spectra_source`` is how the
    raw spectra were read (their :attr:`~hxex.spectra.SpectrumSet.source`), written to the
    bundle even when the spectra are not."""

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
    ) -> list[Path]:
        """Write the series figures described by ``config.plots`` (the default settings if it
        is None); ``spectra`` adds the raw layer. Returns the files."""
        from .report import Results
        from .report.figures import PlotConfig, write_figures

        return write_figures(
            Results.from_run(self, spectra=spectra), directory, self.config.plots or PlotConfig(),
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
    ) -> Path:
        """Write the results bundle (see :mod:`hxex.io.bundle`); ``spectra`` adds the raw data,
        ``pending`` and ``review`` are as for :func:`~hxex.io.write_bundle`."""
        from .io.bundle import write_bundle

        return write_bundle(
            path, measurements=self.measurements, fit=self.fit, bootstrap=self.bootstrap,
            bootstrap_by_k=self.bootstrap_by_k, binned=self.binned, spectra=spectra,
            picker_config=self.config.picker,
            controls=self.controls, quality=self.quality, envelopes=self.envelopes,
            run_config=self.config.to_dict(),
            spectra_source=self.spectra_source if spectra is None or spectra.source is None
            else spectra.source,
            project=project, notes=notes, overwrite=overwrite, pending=pending, review=review,
        )


def chunk_bytes(n_bins: int, shape_len: int, k_max: int, n: int, itemsize: int,
                factor: float = WORK_FACTOR) -> float:
    """Estimated memory of a chunk of ``n`` measurements with ``n_bins`` bins and shapes of
    ``shape_len`` entries: the array of the binomial pmf of every shape position, bin and
    population, times the number of such arrays alive at once (``factor``: ``WORK_FACTOR`` for
    whole-number bins, ``WORK_FACTOR_DIRECT`` otherwise; measured peaks, which grow linearly
    with the chunk)."""
    return float(itemsize) * factor * n_bins * shape_len * k_max * n


def _work_factor(data: Measurements) -> float:
    """The memory factor of ``data``: whole-number bins use the integer convolution."""
    whole = all(len(b) == 0 or (np.all(b == np.round(b)) and b.min() >= 0) for b in data.bins)
    return WORK_FACTOR if whole else WORK_FACTOR_DIRECT


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
    factor = _work_factor(data)
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
    return list(fit.k_values) if config.bootstrap.ks == "all" else list(config.bootstrap.ks)


@dataclass
class _Plan:
    """The chunks of an analysis and its progress units: the fit's share of ``progress_range``
    and, per chunk, its edges (fraction of the measurements before it)."""

    chunks: list[np.ndarray]
    subs: list[Measurements]
    edges: np.ndarray
    lo: float
    hi: float
    fit_share: float


def _plan(data: Measurements, config: RunConfig, progress_range) -> _Plan:
    k_max = config.fit.k_max
    itemsize = config.fit.backend.torch_dtype().itemsize
    chunks = plan_chunks(data, k_max, itemsize, default_budget(config), config.max_chunk)
    log.info("%d measurements in %d chunk(s)", len(data), len(chunks))
    sizes = np.array([len(c) for c in chunks], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(sizes) / sizes.sum()])
    lo, hi = progress_range
    return _Plan(chunks, [data.select(rows) for rows in chunks], edges, lo, hi,
                 0.3 if config.bootstrap is not None else 1.0)


def _fit_part(
    data: Measurements, config: RunConfig, plan: _Plan, progress, cancel, mz_offset_ppm,
    rates=None, n_sites=None, rt=None, poor_envelope=None,
) -> tuple[list[FitResult], FitResult, ControlTable | None, QualityFlags | None]:
    """Fit every chunk, then the controls and the quality flags of the merged fit (which do not
    need the bootstrap). ``n_sites`` is the exchangeable sites of every measurement
    (:func:`site_counts`); without it they are read back from ``data.n_max`` (rounded; an ion
    without any site then counts one). ``poor_envelope``: the series whose envelope is poor."""
    lo, hi, share, edges = plan.lo, plan.hi, plan.fit_share, plan.edges
    fits: list[FitResult] = []
    start_stage(progress, "fit", lo, lo + (hi - lo) * share)
    for j, sub in enumerate(plan.subs):
        check_cancel(cancel)
        a = lo + (hi - lo) * share * edges[j]
        b = lo + (hi - lo) * share * edges[j + 1]
        fits.append(fit_models(sub, config.fit, progress=_Span(progress, a, b), cancel=cancel))
    fit = fits[0] if len(fits) == 1 else FitResult.merge(fits, plan.chunks)
    check_cancel(cancel)
    at = lo + (hi - lo) * share
    start_stage(progress, "controls and quality", at, at)
    controls = None
    if config.controls is not None:
        if n_sites is None:
            n_sites = np.rint(np.asarray(data.n_max, dtype=float) / config.n_max_factor)
        controls = fit_controls(fit, n_sites, config.controls, rates=rates, rt=rt)
    quality = None
    if config.quality is not None:
        quality = assess_quality(data, fit, config.quality, mz_offset_ppm=mz_offset_ppm,
                                 poor_envelope_series=poor_envelope)
    return fits, fit, controls, quality


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


def analyse(
    data: Measurements,
    config: RunConfig,
    *,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
    progress_range: tuple[float, float] = (0.0, float(UNITS)),
    mz_offset_ppm: np.ndarray | None = None,
) -> tuple[
    FitResult,
    BootstrapResult | None,
    dict[int, BootstrapResult] | None,
    ControlTable | None,
    QualityFlags | None,
]:
    """Fit (and bootstrap) ``data`` in chunks of whole series, then the controls and the quality
    flags of the merged result (``mz_offset_ppm``, the picker's offset of every measurement,
    for the m/z screen). The chunking does not change the result."""
    plan = _plan(data, config, progress_range)
    fits, fit, controls, quality = _fit_part(data, config, plan, progress, cancel, mz_offset_ppm)
    boot, by_k = _bootstrap_part(config, plan, fits, progress, cancel)
    return fit, boot, by_k, controls, quality


def site_counts(binned: BinnedSet, config: RunConfig) -> np.ndarray:
    """The exchangeable sites of every spectrum of ``binned``, ``(M,)`` integers: the backbone
    amides of its peptide and ion type plus its user ``Hex`` sites (owner: they stand for
    exchangeable amides on a modification), as :meth:`hxex.fit.Measurements.from_binned` counts
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
    keep_spectra: bool = True,
    figures_dir: str | Path | None = None,
    reference: Reference | None = None,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> RunResult:
    """Run the whole analysis on raw spectra (picked here) or on an already picked set.

    Spectra flagged ``empty`` (and ``no_signal`` if ``config.drop_no_signal``) are dropped and
    recorded. With ``out`` the results bundle is written there (with the raw spectra if the input
    was a :class:`SpectrumSet` and ``keep_spectra``). An existing ``out`` is refused before
    anything is fitted unless ``overwrite``; a bundle replaced with ``overwrite`` keeps its
    review layer (as :func:`resume_bootstrap` does). With ``config.plots`` the series figures
    are written to ``figures_dir`` (default: ``out`` with ``_figures`` added to its name);
    ``reference`` adds known populations to them. Raises ValueError if nothing is left.
    """
    review = _existing_review(out, overwrite) if out is not None else None
    if config.plots is not None and figures_dir is None:
        if out is None:
            raise ValueError("config.plots needs figures_dir or out to say where to write")
        figures_dir = Path(out).with_name(Path(out).name + "_figures")
    raw = data if isinstance(data, SpectrumSet) else None
    if config.envelope.mode != "theory" and raw is None:
        raise ValueError(f"envelope mode {config.envelope.mode!r} needs the raw spectra "
                         "(a SpectrumSet), not a picked set")
    pick_share = 0.05 if raw is not None else 0.0
    envelopes, clusters, shapes = None, None, None
    if raw is not None and config.envelope.mode != "theory":
        groups = None
        if config.envelope.share:
            from .io.mzml import envelope_groups

            groups = envelope_groups(raw.source)
            if not groups:
                log.warning("envelope share: the spectra were not read from scan files, so "
                            "every series keeps its own envelope")
        envelopes, by_series = series_envelopes(
            raw, config=config.envelope, reference_time=config.reference_time,
            fallback_time=config.envelope_fallback_time, picker=config.picker,
            registry=config.registry, groups=groups,
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
    if raw is not None:
        start_stage(progress, "pick", 0.0, UNITS * pick_share)
        binned = pick_spectra(
            raw, config.picker, registry=config.registry, clusters=clusters,
            progress=_Span(progress, 0.0, UNITS * pick_share), cancel=cancel,
        )
    else:
        binned = data
    drop_empty = np.asarray(binned.empty, dtype=bool)
    drop_nosig = np.asarray(binned.no_signal, dtype=bool) & config.drop_no_signal
    has_bins = np.diff(binned.offsets) > 0
    kept = ~drop_empty & ~drop_nosig & has_bins
    reason = np.where(drop_empty, "empty", np.where(~kept, "no_signal", "")).astype(str)
    if not kept.any():
        raise ValueError("every spectrum was dropped (empty or without signal)")
    if (~kept).any():
        log.warning("dropped %d of %d spectra: %d empty, %d without signal", int((~kept).sum()),
                    len(kept), int((reason == "empty").sum()), int((reason == "no_signal").sum()))
    picked = binned.select(kept)
    from .io.mzml import kinetics_spectra

    measurements = Measurements.from_binned(
        picked, n_max_factor=config.n_max_factor,
        undeut_time=config.reference_time, nterm_subtract=config.picker.nterm_subtract,
        amide_rules=config.picker.amide_rules, registry=config.registry, shapes=shapes,
        kinetics=None if raw is None else kinetics_spectra(raw.source, picked.spectrum_id),
    )
    plan = _plan(measurements, config, (UNITS * pick_share, float(UNITS - _FINISH)))
    rates = control_rates(picked, config)
    rt = series_rt(picked.data_id, picked.rt) if rates is not None else None
    poor = None if envelopes is None else envelopes.data_id[envelopes.poor]
    fits, fit, controls, quality = _fit_part(measurements, config, plan, progress, cancel,
                                             binned.mz_offset_ppm[kept], rates,
                                             site_counts(picked, config), rt=rt,
                                             poor_envelope=poor)
    result = RunResult(
        config=config, binned=binned, kept=kept, dropped_reason=reason,
        measurements=measurements, fit=fit, bootstrap=None, bootstrap_by_k=None,
        controls=controls, quality=quality, envelopes=envelopes,
        spectra_source=None if raw is None or raw.source is None else dict(raw.source),
    )
    spectra = raw if keep_spectra else None
    review = _review_for(review, fit)
    checkpoint = out is not None and config.bootstrap is not None
    if checkpoint:  # the fits are kept if the bootstrap is stopped or fails
        check_cancel(cancel)
        at = plan.lo + (plan.hi - plan.lo) * plan.fit_share
        start_stage(progress, "write fits", at, at)
        result.write(out, spectra=spectra, project=project, notes=notes, overwrite=overwrite,
                     pending=("bootstrap",), review=review)
        overwrite = True
    try:
        boot, by_k = _bootstrap_part(config, plan, fits, progress, cancel)
    except BaseException:
        if checkpoint:
            log.warning("the bootstrap did not finish: the fits are in %s; 'hxex resume' (or "
                        "hxex.run.resume_bootstrap) adds the bootstrap to them", out)
        raise
    result = dataclasses.replace(result, bootstrap=boot, bootstrap_by_k=by_k)
    return _finish(result, out, spectra=spectra, project=project, notes=notes,
                   overwrite=overwrite, figures_dir=figures_dir, raw=raw, reference=reference,
                   progress=progress, cancel=cancel, review=review)


def _existing_review(out, overwrite: bool):
    """Refuse an existing ``out`` unless ``overwrite`` (before anything is fitted, as the CLI
    does); with it, the review layer of the bundle there (None if it holds no review or is not
    a bundle: the write then refuses it as before)."""
    from .io.bundle import BundleError, open_bundle

    path = Path(out)
    if not path.exists():
        return None
    if not overwrite:
        raise BundleError(f"{path} exists (use overwrite=True to replace a bundle)")
    try:
        with open_bundle(path) as b:
            review = b.review() if b.has("review") else None
    except BundleError:
        return None
    if review is not None and len(review):
        log.info("%s: carrying its %d review entries over to the new bundle", path,
                 len(review))
    return review


def _review_for(review, fit: FitResult):
    """``review`` without the entries the new run cannot hold (spectra it does not have, a
    chosen K it did not fit), logged; None stays None."""
    if review is None or not len(review):
        return review
    known = set(map(str, fit.spectrum_id))
    keep = np.array([str(s) in known and (int(k) == 0 or int(k) in fit.k_values)
                     for s, k in zip(review.spectrum_id, review.k_override, strict=True)],
                    dtype=bool)
    if not keep.all():
        log.warning("%d of %d review entries do not apply to this run (spectra it dropped or a "
                    "K it did not fit) and are left out", int((~keep).sum()), len(keep))
    return dataclasses.replace(review, spectrum_id=review.spectrum_id[keep],
                               status=review.status[keep], k_override=review.k_override[keep],
                               note=review.note[keep], updated=review.updated[keep])


def _finish(result: RunResult, out, *, spectra, project, notes, overwrite, figures_dir, raw,
            reference, progress, cancel, review=None) -> RunResult:
    """Write the bundle (if ``out``) and the figures (if the run asks for them)."""
    if out is not None:
        check_cancel(cancel)
        start_stage(progress, "write", UNITS - _FINISH, UNITS)
        result.write(out, spectra=spectra, project=project, notes=notes, overwrite=overwrite,
                     review=review)
    if result.config.plots is not None:
        check_cancel(cancel)
        start_stage(progress, "figures", UNITS - _FINISH, UNITS)
        result.write_figures(figures_dir, spectra=raw, reference=reference, cancel=cancel)
    report(progress, UNITS, UNITS)
    return result


def resume_bootstrap(
    path: str | Path,
    *,
    bootstrap: BootstrapConfig | None = None,
    device: str | None = None,
    figures_dir: str | Path | None = None,
    reference: Reference | None = None,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> RunResult:
    """Add the bootstrap to the bundle a run wrote before its bootstrap (its ``pending`` is
    ``['bootstrap']``: the run was stopped or failed during the bootstrap), and rewrite it
    complete. The fits are not repeated, and the result is the one the run would have given
    (each measurement is seeded from its own id). The bootstrap settings are the run's unless
    ``bootstrap`` is given; ``device`` (``'cpu'``, ``'cuda'``, ...) replaces the run's, so a
    bundle fitted on a GPU can be resumed where there is none. The stored spectra, notes and
    review are kept; figures are written as by :func:`run` when the run's settings ask for
    them."""
    from .io.bundle import open_bundle

    path = Path(path)
    with open_bundle(path) as b:
        if "bootstrap" not in b.pending:
            raise ValueError(f"{path}: no bootstrap is pending (the bundle is complete)")
        if b.manifest["configs"].get("run") is None:
            raise ValueError(f"{path}: the bundle does not record its run settings")
        config = RunConfig.from_dict(b.manifest["configs"]["run"])
        if bootstrap is not None:
            config = dataclasses.replace(config, bootstrap=bootstrap)
        if config.bootstrap is None:
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
    if config.plots is not None and figures_dir is None:
        figures_dir = path.with_name(path.name + "_figures")
    kept = np.isin(binned.spectrum_id, measurements.spectrum_id)
    reason = np.where(np.asarray(binned.empty, dtype=bool), "empty",
                      np.where(~kept, "no_signal", "")).astype(str)
    plan = dataclasses.replace(_plan(measurements, config, (0.0, float(UNITS - _FINISH))),
                               fit_share=0.0)
    boot, by_k = _bootstrap_part(config, plan, [fit.select(rows) for rows in plan.chunks],
                                 progress, cancel)
    result = RunResult(
        config=config, binned=binned, kept=kept, dropped_reason=reason,
        measurements=measurements, fit=fit, bootstrap=boot, bootstrap_by_k=by_k,
        controls=controls, quality=quality, envelopes=envelopes, spectra_source=source,
    )
    return _finish(result, path, spectra=spectra, project=project, notes=notes, overwrite=True,
                   figures_dir=figures_dir, raw=spectra, reference=reference, progress=progress,
                   cancel=cancel, review=review)
