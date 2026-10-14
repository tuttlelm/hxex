# SPDX-License-Identifier: GPL-3.0-or-later
"""Thermo ``.raw`` input, read directly with Thermo's RawFileReader (no conversion to mzML).

:class:`ThermoFile` gives the same scan table and scans as :class:`~hxex.io.MzmlFile`, and
:func:`read_thermo` makes raw spectra exactly as :func:`~hxex.io.read_mzml` does, with the same
runs (:class:`~hxex.io.MzmlRun`), targets (:class:`~hxex.io.MzmlTarget`) and settings
(:class:`~hxex.io.MzmlConfig`); the read is recorded with ``reader`` ``'thermo'`` and repeated
by :func:`read_thermo_source` (or :func:`~hxex.io.read_source`).

Reading needs three things that are not part of hxex:

- pythonnet (``pip install "hxex[thermo]"``), which runs .NET code from Python;
- a .NET runtime: Mono on Linux and macOS (pythonnet's default there), the .NET Framework on
  Windows, or .NET 8 or newer with ``PYTHONNET_RUNTIME=coreclr``;
- Thermo's ``ThermoFisher.CommonCore.Data.dll`` and ``ThermoFisher.CommonCore.RawFileReader.dll``,
  under Thermo's own license (not GPL; they are not distributed with hxex; ``LICENSE-EXCEPTION``
  permits combining hxex with them). They come with
  ThermoRawFileParser releases and from Thermo's RawFileReader repository. Their directory is
  given as ``thermo_dir`` or in the environment variable ``HXEX_THERMO_DIR``.

The scans are as ThermoRawFileParser writes them to mzML (``-p``, profile data kept), which is
how this reader is checked: the same ids (``controllerType=0 controllerNumber=1 scan=N``),
times (s), filter strings, precursors, activations (EThcD is ETD plus supplemental beam-type
CID) and arrays (the scan's segmented data: profile points, or the peaks of a centroided scan).
The precursor m/z is the instrument's monoisotopic m/z when it found one, else the isolation
m/z; the charge is the instrument's (0 when it found none).

.NET and Thermo's assemblies are loaded into the Python process once, on the first file opened
(the one piece of process state here), from one directory for the life of the process.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import chem
from ..hooks import CancelHook, ProgressHook
from ..picking import DEFAULT_PICKER_CONFIG, PickerConfig
from ..spectra import SpectrumSet
from .errors import ReadError
from .mzml import (
    MzmlConfig,
    MzmlRun,
    MzmlTarget,
    Scan,
    ScanTable,
    Stage,
    _read_runs,
    _repeat_read,
    _run_paths,
    _ScanFormat,
)

log = logging.getLogger(__name__)

THERMO_DIR_ENV = "HXEX_THERMO_DIR"
_ASSEMBLIES = ("ThermoFisher.CommonCore.Data.dll", "ThermoFisher.CommonCore.RawFileReader.dll")

# Thermo's activation types (enum names) as the mzML activation names ThermoRawFileParser
# writes; an additional activation of the same stage (EThcD's HCD) is "supplemental ..."
_ACTIVATION_NAMES = {
    "CollisionInducedDissociation": "collision-induced dissociation",
    "HigherEnergyCollisionalDissociation": "beam-type collision-induced dissociation",
    "ElectronTransferDissociation": "electron transfer dissociation",
    "ElectronCaptureDissociation": "electron capture dissociation",
    "NegativeElectronTransferDissociation": "negative electron transfer dissociation",
    "UltraVioletPhotoDissociation": "ultraviolet photodissociation",
    "MultiPhotonDissociation": "photodissociation",
    "PQD": "pulsed q dissociation",
}
_SUPPLEMENTAL = {
    "CollisionInducedDissociation": "supplemental collision-induced dissociation",
    "HigherEnergyCollisionalDissociation": "supplemental beam-type collision-induced dissociation",
}


class ThermoError(ReadError):
    """A Thermo raw file that cannot be read, or a missing piece needed to read one."""


@dataclass(frozen=True)
class Reaction:
    """One precursor reaction of a scan event: the precursor ``mass`` (m/z), the isolation
    ``width`` and ``offset`` (m/z), the ``activation`` (Thermo's enum name, for example
    ``ElectronTransferDissociation``), ``multiple``, whether it is an additional activation
    of the stage before (EThcD's HCD), and the ``energy`` (collision energy, or the value Thermo
    gives other activations)."""

    mass: float
    width: float
    offset: float
    activation: str
    multiple: bool
    energy: float = np.nan


@dataclass(frozen=True)
class ScanInfo:
    """What the RawFileReader says about one scan, as plain values: scan ``number``, ``ms_order``
    (Thermo's; below 1 for parent, neutral-loss and neutral-gain scans), ``rt`` (minutes),
    ``filter``, ``centroid``, the event's ``reactions`` (in order, every stage),
    ``supplemental`` (the filter's ``sa``), and the trailer's ``mono_mz`` and ``charge``
    (0 when the instrument found none)."""

    number: int
    ms_order: int
    rt: float
    filter: str
    centroid: bool
    reactions: tuple[Reaction, ...] = ()
    supplemental: bool = False
    mono_mz: float = 0.0
    charge: int = 0


def scan_from_info(info: ScanInfo, first: int = 1) -> Scan:
    """The :class:`~hxex.io.mzml.Scan` (without arrays) of ``info``, as ThermoRawFileParser
    writes it to mzML, plus every stage of an MSn scan (``stages``; ThermoRawFileParser writes
    the last only). ``first`` is the file's first scan number (position 0)."""
    ms_level = info.ms_order if info.ms_order >= 1 else 2
    pmz, activation, isolation = np.nan, (), (np.nan, np.nan)
    stages: list[Stage] = []
    if ms_level > 1 and info.reactions:
        # a stage is a reaction that is not an additional activation, with those that follow it
        starts = [i for i, r in enumerate(info.reactions) if not r.multiple] or [0]
        for n, start in enumerate(starts):
            stage = info.reactions[start:starts[n + 1] if n + 1 < len(starts) else None]
            main = stage[0]
            names = [_ACTIVATION_NAMES.get(main.activation, main.activation.lower())]
            for r in stage[1:]:
                if info.supplemental and r.activation in _SUPPLEMENTAL:
                    names.append(_SUPPLEMENTAL[r.activation])
                else:
                    names.append(_ACTIVATION_NAMES.get(r.activation, r.activation.lower()))
            centre = main.mass + main.offset
            window = ((centre - main.width / 2, centre + main.width / 2) if main.width > 0
                      else (np.nan, np.nan))
            stages.append(Stage(main.mass, window, tuple(sorted(set(names))), main.energy))
        # the last stage made the scan
        last = stages[-1]
        activation, isolation = last.activation, last.isolation
        pmz = info.mono_mz if info.mono_mz > 0 else last.mz
        stages[-1] = replace(last, mz=float(pmz))
    return Scan(
        index=info.number - first,
        id=f"controllerType=0 controllerNumber=1 scan={info.number}",
        ms_level=ms_level,
        time=info.rt * 60.0,
        filter=info.filter,
        precursor_mz=float(pmz),
        precursor_charge=int(info.charge) if ms_level > 1 else 0,
        activation=activation,
        centroid=bool(info.centroid),
        mz=np.empty(0),
        intensity=np.empty(0),
        isolation=isolation,
        stages=tuple(stages),
    )


# --- .NET and Thermo's RawFileReader ------------------------------------------------------------

_LOADED: dict = {}  # the directory the assemblies came from and their API (one per process)


def _thermo_api(thermo_dir: str | Path | None) -> SimpleNamespace:
    """Thermo's RawFileReader API, loading .NET and the assemblies on first use."""
    where = thermo_dir if thermo_dir is not None else os.environ.get(THERMO_DIR_ENV)
    if not where:
        raise ThermoError(
            "reading Thermo raw files needs Thermo's RawFileReader assemblies "
            f"({', '.join(_ASSEMBLIES)}): give their directory as thermo_dir or in "
            f"{THERMO_DIR_ENV} (they come with ThermoRawFileParser, under Thermo's license)")
    where = Path(where).resolve()
    missing = [a for a in _ASSEMBLIES if not (where / a).is_file()]
    if missing:
        raise ThermoError(f"{where}: no {', '.join(missing)}")
    if _LOADED:
        if _LOADED["dir"] != where:
            raise ThermoError(f"Thermo's RawFileReader is already loaded from {_LOADED['dir']}; "
                              "a Python process can load it from one directory only")
        return _LOADED["api"]
    try:
        import pythonnet
    except ImportError:
        raise ThermoError('reading Thermo raw files needs pythonnet: pip install "hxex[thermo]"'
                          ) from None
    try:
        pythonnet.load()  # the runtime pythonnet chooses (PYTHONNET_RUNTIME), if not loaded
        import clr

        for a in _ASSEMBLIES:
            clr.AddReference(str(where / a))
        from System.Reflection import Assembly
        from ThermoFisher.CommonCore.Data.Business import Device
        from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter
    except Exception as exc:  # the runtime or an assembly that does not load
        raise ThermoError(f"could not load .NET and Thermo's RawFileReader from {where}: {exc}"
                          ) from exc
    version = Assembly.GetAssembly(RawFileReaderAdapter).GetName().Version.ToString()
    api = SimpleNamespace(adapter=RawFileReaderAdapter, device=Device, version=version)
    _LOADED.update(dir=where, api=api)
    return api


def _array(values) -> np.ndarray:
    """A .NET double array as a NumPy copy (a view would keep the .NET array pinned, which
    crashes Mono when Python exits)."""
    return np.array(values, dtype=np.float64)


class _RawFileReader:
    """One open raw file (its first MS device) through Thermo's RawFileReader."""

    def __init__(self, path: Path, thermo_dir: str | Path | None):
        api = _thermo_api(thermo_dir)
        self.version = f"RawFileReader {api.version}"
        # the target of a link: given the link, the RawFileReader misreads the file
        raw = api.adapter.FileFactory(str(Path(path).resolve()))
        if raw is None or raw.IsError:
            message = "" if raw is None else raw.FileError.ErrorMessage
            if raw is not None:
                raw.Dispose()
            raise ThermoError(f"{path}: not a readable Thermo raw file ({message})")
        if raw.InAcquisition:
            raw.Dispose()
            raise ThermoError(f"{path}: the file is still being acquired")
        if raw.GetInstrumentCountOfType(api.device.MS) < 1:
            raw.Dispose()
            raise ThermoError(f"{path}: no mass spectrometer data")
        raw.SelectInstrument(api.device.MS, 1)
        # keep the reference (lock mass) and exception peaks, as ThermoRawFileParser does by
        # default (else they are zero)
        raw.IncludeReferenceAndExceptionData = True
        self._raw = raw
        header = raw.RunHeaderEx
        self.first, self.last = int(header.FirstSpectrum), int(header.LastSpectrum)
        labels = raw.GetTrailerExtraHeaderInformation()
        self._trailer = {labels[i].Label.strip(): i for i in range(labels.Length)}

    def _trailer_value(self, number: int, label: str, default):
        i = self._trailer.get(label)
        if i is None:
            return default
        try:
            return type(default)(self._raw.GetTrailerExtraValue(number, i))
        except (TypeError, ValueError):
            return default

    def info(self, number: int) -> ScanInfo:
        raw = self._raw
        event = raw.GetScanEventForScanNumber(number)
        reactions = []
        for i in range(event.MassCount):
            r = event.GetReaction(i)
            reactions.append(Reaction(float(r.PrecursorMass), float(r.IsolationWidth),
                                      float(r.IsolationWidthOffset), str(r.ActivationType),
                                      bool(r.MultipleActivation), float(r.CollisionEnergy)))
        return ScanInfo(
            number=number,
            ms_order=int(event.MSOrder),
            rt=float(raw.RetentionTimeFromScanNumber(number)),
            filter=str(raw.GetScanEventStringForScanNumber(number)),
            centroid=bool(raw.GetScanStatsForScanNumber(number).IsCentroidScan),
            reactions=tuple(reactions),
            supplemental=str(event.SupplementalActivation) == "On",
            mono_mz=self._trailer_value(number, "Monoisotopic M/Z:", 0.0),
            charge=self._trailer_value(number, "Charge State:", 0),
        )

    def arrays(self, number: int) -> tuple[np.ndarray, np.ndarray]:
        # with the scan's statistics: without them the m/z values differ slightly from those
        # ThermoRawFileParser writes
        stats = self._raw.GetScanStatsForScanNumber(number)
        segmented = self._raw.GetSegmentedScanFromScanNumber(number, stats)
        return _array(segmented.Positions), _array(segmented.Intensities)

    def close(self) -> None:
        self._raw.Dispose()


def _open_reader(path: Path, thermo_dir: str | Path | None):
    """The RawFileReader of ``path`` (tests replace this with a reader of synthetic scans)."""
    return _RawFileReader(path, thermo_dir)


class ThermoFile:
    """A Thermo raw file: its scan table and single scans, as :class:`~hxex.io.MzmlFile`
    (positions start at 0 for the first scan). The file stays open until :meth:`close` (or the
    end of a ``with`` block)."""

    def __init__(self, path: str | Path, *, thermo_dir: str | Path | None = None):
        self.path = Path(path)
        if not self.path.is_file():
            raise ThermoError(f"{self.path}: no such file")
        self._reader = _open_reader(self.path, thermo_dir)
        self._rows: list[Scan] | None = None
        self._table: ScanTable | None = None

    @property
    def reader_version(self) -> str:
        """The RawFileReader that reads the file (recorded with the spectra)."""
        return self._reader.version

    def _scan_rows(self) -> list[Scan]:
        if self._rows is None:
            r = self._reader
            self._rows = [scan_from_info(r.info(n), r.first) for n in range(r.first, r.last + 1)]
        return self._rows

    def scans(self) -> ScanTable:
        if self._table is None:
            self._table = ScanTable.from_scans(self._scan_rows())
        return self._table

    def read_scans(
        self, indices, mz_windows: Sequence[tuple[float, float]] | None = None
    ) -> dict[int, Scan]:
        """The scans at the positions ``indices``, with their arrays (only the points inside
        ``mz_windows`` when given), by position."""
        rows = self._scan_rows()
        out = {}
        for i in sorted({int(i) for i in indices}):
            if not 0 <= i < len(rows):
                raise ThermoError(f"{self.path}: no scan {i}")
            mz, y = self._reader.arrays(self._reader.first + i)
            if len(mz) > 1 and np.any(np.diff(mz) < 0):
                order = np.argsort(mz, kind="stable")
                mz, y = mz[order], y[order]
            if mz_windows is not None:
                keep = np.zeros(len(mz), dtype=bool)
                for lo, hi in mz_windows:
                    keep |= (mz >= lo) & (mz <= hi)
                mz, y = mz[keep], y[keep]
            out[i] = replace(rows[i], mz=mz, intensity=y)
        return out

    def scan(self, index: int, mz_window: tuple[float, float] | None = None) -> Scan:
        """The scan at position ``index`` (as in the scan table), with its arrays."""
        return self.read_scans([index], None if mz_window is None else [mz_window])[index]

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _format(thermo_dir: str | Path | None) -> _ScanFormat:
    return _ScanFormat(
        "thermo", "a Thermo raw", lambda p: ThermoFile(p, thermo_dir=thermo_dir), ThermoError,
        extra=lambda f: {"thermo_reader": f.reader_version},
    )


def read_thermo(
    runs: Sequence[MzmlRun],
    targets: Sequence[MzmlTarget],
    *,
    project: str,
    config: MzmlConfig = MzmlConfig(),  # noqa: B008
    picker: PickerConfig = DEFAULT_PICKER_CONFIG,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
    data_root: str | Path | None = None,
    record_paths: str = "path",
    thermo_dir: str | Path | None = None,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> SpectrumSet:
    """Raw spectra of every target in every run of Thermo raw files: :func:`~hxex.io.read_mzml`
    for ``.raw`` files (the same arguments, layouts, spectrum ids and record), plus
    ``thermo_dir``, the directory of Thermo's assemblies (else ``HXEX_THERMO_DIR``; not
    recorded). The record's ``reader`` is ``'thermo'`` and ``thermo_reader`` names the
    RawFileReader version."""
    opened, recorded = _run_paths(runs, data_root, record_paths)
    return _read_runs(_format(thermo_dir), runs, opened, recorded, None, record_paths, targets,
                      project=project, config=config, picker=picker, registry=registry,
                      progress=progress, cancel=cancel)


def read_thermo_source(
    source,
    *,
    directory: str | Path | None = None,
    data_root: str | Path | None = None,
    on_change: str = "warn",
    thermo_dir: str | Path | None = None,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> SpectrumSet:
    """Repeat the read recorded in the ``source`` of spectra made by :func:`read_thermo`, as
    :func:`~hxex.io.read_mzml_source` (files found by name in ``directory`` or at their
    recorded paths, identities checked, ``on_change``), with ``thermo_dir`` as for
    :func:`read_thermo`."""
    return _repeat_read(_format(thermo_dir), source, directory=directory, data_root=data_root,
                        on_change=on_change, progress=progress, cancel=cancel)
