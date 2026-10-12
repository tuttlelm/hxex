# SPDX-License-Identifier: GPL-3.0-or-later
"""The results that plots and tables are drawn from.

:class:`Results` holds the objects a run produces (or a results bundle returns), so a fresh run
and a loaded bundle give the same plots. It adds nothing that is not in the bundle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import chem
from ..bootstrap import BootstrapResult
from ..controls import ControlTable
from ..fit import FitResult, Measurements
from ..picking import BinnedSet, PickerConfig
from ..quality import QualityFlags
from ..spectra import SpectrumSet

if TYPE_CHECKING:
    from ..io.bundle import Bundle, ReviewTable
    from ..run import RunResult


@dataclass(frozen=True)
class Settings:
    """How a run was made: the project, notes, creation time, package versions and the settings
    of every stage (``configs``, as in the bundle manifest). The typed properties read the
    stage settings back; they are None if the bundle did not record them."""

    project: str = ""
    created: str = ""
    notes: str = ""
    versions: dict[str, str] = field(default_factory=dict)
    configs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> Settings:
        return cls(
            project=str(manifest.get("project", "")), created=str(manifest.get("created", "")),
            notes=str(manifest.get("notes", "")), versions=dict(manifest.get("versions", {})),
            configs=dict(manifest.get("configs", {})),
        )

    @classmethod
    def from_run(cls, run: RunResult, project: str = "", notes: str = "") -> Settings:
        from ..io.bundle import _versions

        return cls(
            project=project, created=datetime.now(UTC).isoformat(timespec="seconds"),
            notes=notes, versions=_versions(),
            configs={"run": run.config.to_dict(), "spectra_source": run.spectra_source},
        )

    @property
    def run(self):
        """The :class:`hxex.run.RunConfig` of the run, or None."""
        from ..run import RunConfig

        data = self.configs.get("run")
        return None if data is None else RunConfig.from_dict(data)

    @property
    def spectra_source(self) -> dict | None:
        """How the raw spectra were read (:attr:`hxex.spectra.SpectrumSet.source`), or None."""
        return self.configs.get("spectra_source")

    @property
    def picker(self) -> PickerConfig | None:
        """The picker settings (from the stage entry, else from the run settings), or None."""
        data = self.configs.get("picker") or (self.configs.get("run") or {}).get("picker")
        return None if data is None else PickerConfig.from_dict(data)

    @property
    def registry(self) -> chem.ModificationRegistry:
        """The modification registry the sequences were read with (the default if not stored)."""
        data = (self.configs.get("run") or {}).get("registry")
        return chem.DEFAULT_MODIFICATIONS if data is None else chem.ModificationRegistry.from_dict(
            data)


@dataclass(frozen=True)
class Results:
    """Measurements, fit, intervals, controls, quality flags and the review of a run.

    ``binned`` (the picked table, with the predicted m/z of every bin) is optional. Raw spectra
    are optional too: ``spectra`` holds them in memory, or ``loader`` fetches some by id (for a
    bundle, which must still be open). ``bootstrap`` is at each measurement's selected K and
    ``bootstrap_by_k`` at fixed K values. ``review`` is the reviewer's layer; replace it with
    :meth:`with_review` (a ``Results`` is never edited).
    """

    measurements: Measurements
    fit: FitResult
    bootstrap: BootstrapResult | None = None
    bootstrap_by_k: dict[int, BootstrapResult] = field(default_factory=dict)
    controls: ControlTable | None = None
    quality: QualityFlags | None = None
    review: ReviewTable | None = None
    binned: BinnedSet | None = None
    spectra: SpectrumSet | None = None
    settings: Settings | None = None
    loader: Callable[[Sequence[str]], SpectrumSet | None] | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def from_bundle(cls, bundle: Bundle) -> Results:
        return cls(
            measurements=bundle.measurements(), fit=bundle.fit(), bootstrap=bundle.bootstrap(),
            bootstrap_by_k={k: bundle.bootstrap(k) for k in bundle.bootstrap_ks()},
            controls=bundle.controls(), quality=bundle.quality(), review=bundle.review(),
            binned=bundle.binned(),
            settings=Settings.from_manifest(bundle.manifest),
            loader=bundle.spectra if bundle.has("spectra") else None,
        )

    @classmethod
    def from_run(
        cls, run: RunResult, review: ReviewTable | None = None, spectra: SpectrumSet | None = None
    ) -> Results:
        return cls(
            measurements=run.measurements, fit=run.fit, bootstrap=run.bootstrap,
            bootstrap_by_k=dict(run.bootstrap_by_k or {}), controls=run.controls,
            quality=run.quality, review=review, binned=run.binned, spectra=spectra,
            settings=Settings.from_run(run),
        )

    def with_review(self, review: ReviewTable | None) -> Results:
        """The same results with another review layer (nothing else is copied or refitted)."""
        return replace(self, review=review)

    def __len__(self) -> int:
        return len(self.fit)

    def find(self, spectrum_id: str) -> int:
        """Position of a spectrum; KeyError if it is not in the results."""
        hits = np.flatnonzero(self.fit.spectrum_id == spectrum_id)
        if not len(hits):
            raise KeyError(spectrum_id)
        return int(hits[0])

    @property
    def effective_K(self) -> np.ndarray:
        """The K of every measurement with the reviewer's choice applied (else the selected K)."""
        if self.review is None:
            return self.fit.selected_K.copy()
        return self.review.effective_K(self.fit)

    def bootstrap_for(self, i: int, k: int) -> BootstrapResult | None:
        """The bootstrap that covers measurement ``i`` at K = ``k`` (the fixed-K one, else the
        selected-K one if ``k`` is that measurement's selected K), or None."""
        if k in self.bootstrap_by_k:
            return self.bootstrap_by_k[k]
        if self.bootstrap is not None and self.bootstrap.selected_K[i] == k:
            return self.bootstrap
        return None

    def mz(self, i: int) -> np.ndarray | None:
        """The predicted m/z of the bins of measurement ``i`` if the picked table is present."""
        if self.binned is None:
            return None
        try:
            j = self.binned.find(str(self.fit.spectrum_id[i]))
        except KeyError:
            return None
        return self.binned.spectrum(j)[0]

    def raw_spectrum(self, spectrum_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        """(m/z, intensity) of a raw spectrum, or None if the raw data is not available."""
        spectra = self.spectra
        if spectra is None and self.loader is not None:
            spectra = self.loader([spectrum_id])
        if spectra is None:
            return None
        try:
            return spectra.spectrum(spectra.find(spectrum_id))
        except KeyError:
            return None
