# SPDX-License-Identifier: GPL-3.0-or-later
"""Running a whole data set: pick, drop, fit, bootstrap, controls, quality flags, bundle.

:func:`run` takes raw spectra (or already picked ones), drops the spectra the picker flagged,
fits the rest in chunks of whole series, and returns a :class:`RunResult` (and writes a results
bundle if asked). A chunk holds whole series because the noise estimate and the control
references work within a series; each measurement is seeded from its own id, so the result does not
depend on how the data was split (tests/test_run.py). The chunk size is set by a memory budget
(by default a share of the free GPU memory, or a fixed amount on the CPU) and can be capped.

Progress is reported as ``progress(done, 1000)``: picking, the fit and the bootstrap each take a
share of the 1000 units (about 5, 30 and 65 with a bootstrap; 5 and 95 without), divided between
chunks by their number of measurements. It is an estimate of the time, not a count of work.
``cancel`` is polled throughout and raises :class:`~hxex.hooks.Cancelled`; nothing is written
then.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
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
from .controls import ControlConfig, ControlTable, fit_controls
from .fit import FitConfig, FitResult, Measurements, fit_models
from .hooks import CancelHook, ProgressHook, check_cancel, report
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
CPU_BUDGET = 4 * 2**30  # bytes per chunk on the CPU unless set
GPU_SHARE = 0.5  # share of the free GPU memory used by one chunk unless set
# memory of the fit (the bootstrap needs less) in units of one pmf array: measured on the GPU,
# the peak was 5.2 to 5.9 times 12 such arrays for 94 to 542 Bimodal measurements (float32)
WORK_FACTOR = 75.0


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
    ) -> Path:
        """Write the results bundle (see :mod:`hxex.io.bundle`); ``spectra`` adds the raw data."""
        from .io.bundle import write_bundle

        return write_bundle(
            path, measurements=self.measurements, fit=self.fit, bootstrap=self.bootstrap,
            bootstrap_by_k=self.bootstrap_by_k, binned=self.binned, spectra=spectra,
            picker_config=self.config.picker,
            controls=self.controls, quality=self.quality, envelopes=self.envelopes,
            run_config=self.config.to_dict(),
            spectra_source=self.spectra_source if spectra is None or spectra.source is None
            else spectra.source,
            project=project, notes=notes, overwrite=overwrite,
        )


def chunk_bytes(n_bins: int, shape_len: int, k_max: int, n: int, itemsize: int) -> float:
    """Estimated memory of a chunk of ``n`` measurements with ``n_bins`` bins and shapes of
    ``shape_len`` entries: the array of the binomial pmf of every shape position, bin and
    population, times the number of such arrays alive at once (``WORK_FACTOR``, from a
    measurement of the peak GPU memory of the fit, which grows linearly with the chunk)."""
    return float(itemsize) * WORK_FACTOR * n_bins * shape_len * k_max * n


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
    widths = np.array([len(b) for b in data.bins])
    series = [np.flatnonzero(data.data_id == s) for s in np.unique(data.data_id)]
    series.sort(key=lambda rows: -int(widths[rows].max()))
    chunks: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    count = 0
    for rows in series:
        if current:
            width = max(int(widths[r].max()) for r in [*current, rows])
            fits = chunk_bytes(width, shape_len, k_max, count + len(rows), itemsize) <= budget
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
        need = chunk_bytes(int(widths[rows].max()), shape_len, k_max, len(rows), itemsize)
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
    k_max = config.fit.k_max
    itemsize = config.fit.backend.torch_dtype().itemsize
    chunks = plan_chunks(data, k_max, itemsize, default_budget(config), config.max_chunk)
    log.info("%d measurements in %d chunk(s)", len(data), len(chunks))
    lo, hi = progress_range
    fit_share = 0.3 if config.bootstrap is not None else 1.0
    sizes = np.array([len(c) for c in chunks], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(sizes) / sizes.sum()])
    subs = [data.select(rows) for rows in chunks]
    fits: list[FitResult] = []
    for j, sub in enumerate(subs):  # every fit, then every bootstrap, so progress only increases
        check_cancel(cancel)
        a = lo + (hi - lo) * fit_share * edges[j]
        b = lo + (hi - lo) * fit_share * edges[j + 1]
        fits.append(fit_models(sub, config.fit, progress=_Span(progress, a, b), cancel=cancel))
    boots: list[BootstrapResult] = []
    fixed: list[dict[int, BootstrapResult]] = []
    ks = _bootstrap_ks(config, fits[0])
    if config.bootstrap is not None:
        for j, (sub, fit) in enumerate(zip(subs, fits, strict=True)):
            check_cancel(cancel)
            a = lo + (hi - lo) * (fit_share + (1 - fit_share) * edges[j])
            b = lo + (hi - lo) * (fit_share + (1 - fit_share) * edges[j + 1])
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
    fit = fits[0] if len(fits) == 1 else FitResult.merge(fits, chunks)
    boot = None
    if boots:
        boot = boots[0] if len(boots) == 1 else BootstrapResult.merge(boots, chunks)
    by_k = None
    if fixed:
        by_k = {k: fixed[0][k] if len(fixed) == 1
                else BootstrapResult.merge([f[k] for f in fixed], chunks) for k in ks}
    check_cancel(cancel)
    controls = None
    if config.controls is not None:
        controls = fit_controls(fit, data.n_max / config.n_max_factor, config.controls)
    quality = None
    if config.quality is not None:
        quality = assess_quality(data, fit, config.quality, mz_offset_ppm=mz_offset_ppm)
    return fit, boot, by_k, controls, quality


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
    was a :class:`SpectrumSet` and ``keep_spectra``). With ``config.plots`` the series figures
    are written to ``figures_dir`` (default: ``out`` with ``_figures`` added to its name);
    ``reference`` adds known populations to them. Raises ValueError if nothing is left.
    """
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
        envelopes, by_series = series_envelopes(
            raw, config=config.envelope, reference_time=config.reference_time,
            fallback_time=config.envelope_fallback_time, picker=config.picker,
            registry=config.registry,
        )
        clusters = {d: e.clusters for d, e in by_series.items()}
        shapes = {d: e.fit_shape for d, e in by_series.items()}
        n_back = int((envelopes.source != "reference").sum())
        if n_back:
            log.warning("%d of %d series took their envelope from the fallback or the theory",
                        n_back, len(envelopes))
    if raw is not None:
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
    measurements = Measurements.from_binned(
        binned.select(kept), n_max_factor=config.n_max_factor,
        undeut_time=config.reference_time, nterm_subtract=config.picker.nterm_subtract,
        amide_rules=config.picker.amide_rules, registry=config.registry, shapes=shapes,
    )
    fit, boot, by_k, controls, quality = analyse(
        measurements, config, progress=progress, cancel=cancel,
        progress_range=(UNITS * pick_share, float(UNITS)),
        mz_offset_ppm=binned.mz_offset_ppm[kept],
    )
    result = RunResult(
        config=config, binned=binned, kept=kept, dropped_reason=reason,
        measurements=measurements, fit=fit, bootstrap=boot, bootstrap_by_k=by_k,
        controls=controls, quality=quality, envelopes=envelopes,
        spectra_source=None if raw is None or raw.source is None else dict(raw.source),
    )
    if out is not None:
        result.write(out, spectra=raw if keep_spectra else None, project=project, notes=notes,
                     overwrite=overwrite)
    if config.plots is not None:
        check_cancel(cancel)
        result.write_figures(figures_dir, spectra=raw, reference=reference, cancel=cancel)
    report(progress, UNITS, UNITS)
    return result
