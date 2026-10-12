# SPDX-License-Identifier: GPL-3.0-or-later
"""mzML input: scans, summing, and spectra for peptides or fragments.

A small mzML 1.1 reader written with the standard library (no lxml): plain or indexed files,
uncompressed or zlib-compressed 32- or 64-bit arrays, profile or centroid scans. Numpress
compression is not supported and is reported. Thermo ``.raw`` files are converted first, for
example with ThermoRawFileParser (``mono ThermoRawFileParser.exe -i=FILE.raw -b=FILE.mzML -f=1
-p``; ``-f=1`` writes mzML and ``-p`` keeps profile data); converting is not part of hxex.

:class:`MzmlFile` gives the scan table (no arrays) and single scans; :func:`read_mzml` turns the
scans of several files into raw spectra (:class:`~hxex.spectra.SpectrumSet`) for a list of
targets (a peptide, the intact protein or a fragment ion, each a
:class:`~hxex.spectra.PeptideEntry`), in either of two layouts:

- a run **without** a fixed time (``MzmlRun.time`` None): every selected scan, or every group of
  scans (:class:`MzmlConfig`), is a time point whose time is the mean scan time plus the run's
  ``time_offset`` (labelling during the acquisition, as in continuous-labelling top-down work);
- a run **with** a fixed time (an LC-MS file per labelling time, or a control file): the selected
  scans of each target, within its retention-time window if it has one, are summed into one
  spectrum at that time.

Times are in seconds. Summing profile scans interpolates them onto one common m/z grid with the
scans' own spacing, without bridging the stretches a vendor left out (they are zero); summing
centroided scans merges peaks within ``sum_ppm``. On a real file, sums of 10 MS1 scans agree with
the same sums made by OpenMS (see docs/PLAN.md item 33).
"""

from __future__ import annotations

import base64
import logging
import re
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .. import chem
from ..hooks import CancelHook, ProgressHook, check_cancel, report
from ..picking import DEFAULT_PICKER_CONFIG, PickerConfig, plan_bins
from ..spectra import PeptideEntry, Spectrum, SpectrumSet, qualify_spectrum_id
from .errors import ReadError

log = logging.getLogger(__name__)

_NS = "{http://psi.hupo.org/ms/mzml}"
# controlled-vocabulary accessions used here
_MS_LEVEL = "MS:1000511"
_SCAN_TIME = "MS:1000016"
_FILTER = "MS:1000512"
_SELECTED_MZ = "MS:1000744"
_CHARGE = "MS:1000041"
_CENTROID = "MS:1000127"
_PROFILE = "MS:1000128"
_MZ_ARRAY = "MS:1000514"
_INT_ARRAY = "MS:1000515"
_FLOAT32 = "MS:1000521"
_FLOAT64 = "MS:1000523"
_ZLIB = "MS:1000574"
_NO_COMPRESSION = "MS:1000576"
_NUMPRESS = {"MS:1002312", "MS:1002313", "MS:1002314", "MS:1002746", "MS:1002747", "MS:1002748"}
_TIME_UNITS = {"UO:0000010": 1.0, "UO:0000031": 60.0, "UO:0000028": 1e-3, "UO:0000032": 3600.0}
# short names for the activation methods, matched against the activation cvParam names
ACTIVATIONS = {
    "ETD": "electron transfer dissociation",
    "ECD": "electron capture dissociation",
    "CID": "collision-induced dissociation",
    "HCD": "beam-type collision-induced dissociation",
    "ETHCD": "electron transfer/higher-energy collision dissociation",
    "UVPD": "ultraviolet photodissociation",
}


class MzmlError(ReadError):
    """An mzML file that cannot be read."""


@dataclass(frozen=True)
class Scan:
    """One scan: its position ``index`` and ``id`` in the file, ``ms_level``, ``time`` (s),
    ``filter`` (the vendor filter string, may be empty), ``precursor_mz`` and
    ``precursor_charge`` (NaN and 0 without a precursor), ``activation`` (names of the activation
    methods, lower case), ``centroid`` (True, False, or None if not stated), and the arrays (empty
    when they were not read)."""

    index: int
    id: str
    ms_level: int
    time: float
    filter: str
    precursor_mz: float
    precursor_charge: int
    activation: tuple[str, ...]
    centroid: bool | None
    mz: np.ndarray
    intensity: np.ndarray


def _cv(elem) -> dict[str, tuple[str, str]]:
    """accession -> (value, unit accession) of the cvParams directly under ``elem``."""
    return {p.get("accession"): (p.get("value", ""), p.get("unitAccession", ""))
            for p in elem.findall(f"{_NS}cvParam")}


def _user_param(elem, name: str) -> str:
    """The value of the userParam ``name`` directly under ``elem`` ('' if there is none)."""
    for p in elem.findall(f"{_NS}userParam"):
        if p.get("name") == name:
            return p.get("value", "")
    return ""


def _decode(bda, n_expected: int, where: str) -> tuple[str, np.ndarray]:
    """The array type (m/z or intensity accession) and the values of a binaryDataArray."""
    cv = _cv(bda)
    kind = _MZ_ARRAY if _MZ_ARRAY in cv else _INT_ARRAY if _INT_ARRAY in cv else ""
    if _NUMPRESS & set(cv):
        raise MzmlError(f"{where}: numpress compression is not supported; convert the file "
                        "without it (for example ThermoRawFileParser's default zlib)")
    if _FLOAT64 in cv:
        dtype = "<f8"
    elif _FLOAT32 in cv:
        dtype = "<f4"
    else:
        raise MzmlError(f"{where}: an array is neither 32- nor 64-bit float")
    text = bda.findtext(f"{_NS}binary") or ""
    raw = base64.b64decode(text)
    if _ZLIB in cv:
        raw = zlib.decompress(raw)
    values = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if n_expected and len(values) != n_expected:
        raise MzmlError(f"{where}: an array has {len(values)} values, expected {n_expected}")
    return kind, values


def _scan_from(elem, read_arrays: bool, mz_windows) -> Scan:
    """A :class:`Scan` from a ``spectrum`` element."""
    index = int(elem.get("index", -1))
    sid = elem.get("id", "")
    where = f"scan {index} ({sid})"
    cv = _cv(elem)
    ms_level = int(float(cv.get(_MS_LEVEL, ("1", ""))[0]))
    centroid = True if _CENTROID in cv else False if _PROFILE in cv else None
    time, filt = np.nan, ""
    scan = elem.find(f"{_NS}scanList/{_NS}scan")
    if scan is not None:
        scv = _cv(scan)
        if _SCAN_TIME in scv:
            value, unit = scv[_SCAN_TIME]
            time = float(value) * _TIME_UNITS.get(unit, 60.0)  # mzML's default unit: minutes
        filt = scv.get(_FILTER, ("", ""))[0] or _user_param(scan, "filter string")
    if not filt:
        filt = _user_param(elem, "filter string")  # OpenMS writes it on the spectrum
    pmz, pz, activation = np.nan, 0, ()
    prec = elem.find(f"{_NS}precursorList/{_NS}precursor")
    if prec is not None:
        ion = prec.find(f"{_NS}selectedIonList/{_NS}selectedIon")
        if ion is not None:
            icv = _cv(ion)
            if _SELECTED_MZ in icv:
                pmz = float(icv[_SELECTED_MZ][0])
            if _CHARGE in icv:
                pz = int(float(icv[_CHARGE][0]))
        act = prec.find(f"{_NS}activation")
        if act is not None:
            params = act.findall(f"{_NS}cvParam")
            activation = tuple(sorted(p.get("name", "").lower() for p in params
                                      if not p.get("value")))
    mz = intensity = np.empty(0)
    if read_arrays:
        n = int(elem.get("defaultArrayLength", 0))
        arrays = {}
        for bda in elem.findall(f"{_NS}binaryDataArrayList/{_NS}binaryDataArray"):
            kind, values = _decode(bda, n, where)
            if kind:
                arrays[kind] = values
        if _MZ_ARRAY not in arrays or _INT_ARRAY not in arrays:
            raise MzmlError(f"{where}: the m/z or the intensity array is missing")
        mz, intensity = arrays[_MZ_ARRAY], arrays[_INT_ARRAY]
        if mz_windows is not None:
            keep = np.zeros(len(mz), dtype=bool)
            for lo, hi in mz_windows:
                keep |= (mz >= lo) & (mz <= hi)
            mz, intensity = mz[keep], intensity[keep]
    return Scan(index, sid, ms_level, time, filt, pmz, pz, activation, centroid, mz, intensity)


def iter_scans(
    path: str | Path,
    *,
    read_arrays: bool = True,
    mz_windows: Sequence[tuple[float, float]] | None = None,
    keep=None,
) -> Iterator[Scan]:
    """Every scan of an mzML file in file order. ``mz_windows`` keeps only the points inside
    those m/z ranges (much less memory for a few targets); ``keep`` is an optional test on the
    scan's metadata (a :class:`Scan` without arrays) that decides whether its arrays are read and
    the scan is returned."""
    path = Path(path)
    try:
        for event, elem in ET.iterparse(path, events=("end",)):
            if event != "end":
                continue
            if elem.tag == f"{_NS}spectrum":
                meta = _scan_from(elem, False, None)
                if keep is None or keep(meta):
                    yield _scan_from(elem, read_arrays, mz_windows) if read_arrays else meta
                elem.clear()  # the spectrum's content is no longer needed
            elif elem.tag == f"{_NS}chromatogram":
                elem.clear()
    except ET.ParseError as exc:
        raise MzmlError(f"{path}: not a readable mzML file ({exc})") from exc


@dataclass(frozen=True)
class ScanTable:
    """The scans of a file without their arrays, as columns: ``index``, ``id``, ``ms_level``,
    ``time`` (s), ``filter``, ``precursor_mz``, ``precursor_charge``, ``activation`` (names
    joined with ``;``) and ``centroid`` (1, 0, or -1 if not stated)."""

    index: np.ndarray
    id: np.ndarray
    ms_level: np.ndarray
    time: np.ndarray
    filter: np.ndarray
    precursor_mz: np.ndarray
    precursor_charge: np.ndarray
    activation: np.ndarray
    centroid: np.ndarray

    def __len__(self) -> int:
        return len(self.index)

    def select(
        self,
        *,
        ms_level: int | None = None,
        activation: str | None = None,
        precursor_mz: float | None = None,
        precursor_tol: float = 1.0,
        precursor_charge: int | None = None,
        time_range: tuple[float, float] | None = None,
        filter_contains: str | None = None,
        filter_excludes: str | None = None,
    ) -> np.ndarray:
        """Positions (into this table) of the scans that match every condition given.
        ``activation`` is a short name (``ETD``, ``HCD``, ``CID``, ``ECD``, ``EThcD``, ``UVPD``)
        or any text found in the activation names; ``time_range`` is in seconds, inclusive;
        ``filter_contains`` and ``filter_excludes`` test the vendor filter string (for example
        ``'sid'`` separates scans with supplemental activation)."""
        keep = np.ones(len(self), dtype=bool)
        if ms_level is not None:
            keep &= self.ms_level == ms_level
        if activation is not None:
            want = ACTIVATIONS.get(activation.upper(), activation).lower()
            names = [str(a).split(";") for a in self.activation]
            keep &= np.array([want in n for n in names], dtype=bool)
        if precursor_mz is not None:
            keep &= np.abs(self.precursor_mz - precursor_mz) <= precursor_tol
        if precursor_charge is not None:
            keep &= self.precursor_charge == precursor_charge
        if time_range is not None:
            keep &= (self.time >= time_range[0]) & (self.time <= time_range[1])
        if filter_contains is not None:
            keep &= np.array([filter_contains in str(f) for f in self.filter], dtype=bool)
        if filter_excludes is not None:
            keep &= np.array([filter_excludes not in str(f) for f in self.filter], dtype=bool)
        return np.flatnonzero(keep)


class MzmlFile:
    """An mzML file: its scan table (read once, without arrays) and single scans by index (through
    the file's index when it has one, else by reading through the file)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise MzmlError(f"{self.path}: no such file")
        self._table: ScanTable | None = None
        self._offsets: dict[int, int] | None = None

    def scans(self) -> ScanTable:
        if self._table is None:
            rows = list(iter_scans(self.path, read_arrays=False))
            self._table = ScanTable(
                index=np.array([s.index for s in rows], dtype=np.int64),
                id=np.array([s.id for s in rows], dtype=str),
                ms_level=np.array([s.ms_level for s in rows], dtype=np.int64),
                time=np.array([s.time for s in rows], dtype=np.float64),
                filter=np.array([s.filter for s in rows], dtype=str),
                precursor_mz=np.array([s.precursor_mz for s in rows], dtype=np.float64),
                precursor_charge=np.array([s.precursor_charge for s in rows], dtype=np.int64),
                activation=np.array([";".join(s.activation) for s in rows], dtype=str),
                centroid=np.array([-1 if s.centroid is None else int(s.centroid) for s in rows],
                                  dtype=np.int64),
            )
        return self._table

    def _index_offsets(self) -> dict[int, int]:
        """Byte offsets of the spectra from an indexedmzML index (empty if there is none)."""
        if self._offsets is None:
            self._offsets = {}
            size = self.path.stat().st_size
            with self.path.open("rb") as fh:
                fh.seek(max(0, size - 4096))
                tail = fh.read().decode("utf-8", "replace")
                m = re.search(r"<indexListOffset>(\d+)</indexListOffset>", tail)
                if m:
                    fh.seek(int(m.group(1)))
                    text = fh.read().decode("utf-8", "replace")
                    block = re.search(r'<index name="spectrum">(.*?)</index>', text, re.S)
                    if block:
                        found = re.findall(r"<offset[^>]*>(\d+)</offset>", block.group(1))
                        self._offsets = {n: int(off) for n, off in enumerate(found)}
        return self._offsets

    def scan(self, index: int, mz_window: tuple[float, float] | None = None) -> Scan:
        """The scan at position ``index`` (as in the scan table), with its arrays."""
        offsets = self._index_offsets()
        windows = None if mz_window is None else [mz_window]
        if index in offsets:
            with self.path.open("rb") as fh:
                fh.seek(offsets[index])
                chunk = b""
                while b"</spectrum>" not in chunk:
                    more = fh.read(1 << 20)
                    if not more:
                        raise MzmlError(f"{self.path}: scan {index} is cut off")
                    chunk += more
            text = chunk[: chunk.index(b"</spectrum>") + len(b"</spectrum>")].decode("utf-8")
            text = text.replace("<spectrum ", '<spectrum xmlns="http://psi.hupo.org/ms/mzml" ', 1)
            return _scan_from(ET.fromstring(text), True, windows)
        for s in iter_scans(self.path, mz_windows=windows, keep=lambda m: m.index == index):
            return s
        raise MzmlError(f"{self.path}: no scan {index}")


# --- summing scans ----------------------------------------------------------------------------

def _resample(mz: np.ndarray, y: np.ndarray, grid: np.ndarray, gap_factor: float) -> np.ndarray:
    """``y`` interpolated onto ``grid``, zero outside the scan and inside its gaps: an interval
    wider than ``gap_factor`` times both of its neighbouring intervals is a stretch the vendor
    left out (zero intensity), not a line to draw between two points."""
    if len(mz) < 2:
        return np.zeros(len(grid))
    out = np.interp(grid, mz, y, left=0.0, right=0.0)
    step = np.diff(mz)
    is_gap = _gaps(mz, gap_factor)
    if is_gap.any():
        j = np.searchsorted(mz, grid, side="right") - 1  # interval of each grid point
        inside = (j >= 0) & (j < len(step))
        hit = np.zeros(len(grid), dtype=bool)
        hit[inside] = is_gap[j[inside]] & (grid[inside] > mz[j[inside]]) & (
            grid[inside] < mz[j[inside] + 1])
        out[hit] = 0.0
    return out


def _gaps(mz: np.ndarray, gap_factor: float) -> np.ndarray:
    """For each interval of ``mz``, whether it is a left-out stretch (see :func:`_resample`)."""
    step = np.diff(mz)
    before = np.concatenate([[np.inf], step[:-1]])
    after = np.concatenate([step[1:], [np.inf]])
    return (step > gap_factor * before) & (step > gap_factor * after)


def _common_grid(grids: Sequence[np.ndarray], gap_factor: float) -> np.ndarray:
    """One m/z grid for several profile scans: the points of the scan with the most points, plus
    the points of the others where it has no data (outside its range or inside its left-out
    stretches). Scans of one acquisition have the same spacing but grids offset by up to half a
    step, so the union of their points would be several times denser than any scan (which
    changes how many points a picking window holds); one grid keeps the original spacing."""
    order = sorted(range(len(grids)), key=lambda i: -len(grids[i]))
    grid = np.asarray(grids[order[0]], float)
    for i in order[1:]:
        x = np.asarray(grids[i], float)
        if len(grid) < 2:
            grid = np.unique(np.concatenate([grid, x]))
            continue
        j = np.searchsorted(grid, x)
        inside = (j > 0) & (j < len(grid))
        covered = np.zeros(len(x), dtype=bool)
        covered[inside] = ~_gaps(grid, gap_factor)[j[inside] - 1]
        covered |= np.isin(x, grid)
        if (~covered).any():
            grid = np.unique(np.concatenate([grid, x[~covered]]))
    return grid


def sum_profile(scans: Sequence[tuple[np.ndarray, np.ndarray]], gap_factor: float = 3.0):
    """The sum of profile scans on one common grid (:func:`_common_grid`), each scan
    interpolated there without bridging its left-out stretches (:func:`_resample`). Returns
    ``(mz, intensity)``."""
    scans = [(np.asarray(m, float), np.asarray(y, float)) for m, y in scans if len(m)]
    if not scans:
        return np.empty(0), np.empty(0)
    if len(scans) == 1:
        return scans[0]
    grid = _common_grid([m for m, _ in scans], gap_factor)
    total = np.zeros(len(grid))
    for m, y in scans:
        total += _resample(m, y, grid, gap_factor)
    return grid, total


def sum_centroid(scans: Sequence[tuple[np.ndarray, np.ndarray]], ppm: float = 10.0):
    """The sum of centroided scans: peaks closer than ``ppm`` (to their neighbour in m/z order)
    are merged, with summed intensity and intensity-weighted m/z. Returns ``(mz, intensity)``."""
    mz = np.concatenate([np.asarray(m, float) for m, _ in scans]) if scans else np.empty(0)
    y = np.concatenate([np.asarray(v, float) for _, v in scans]) if scans else np.empty(0)
    if len(mz) == 0:
        return mz, y
    order = np.argsort(mz, kind="stable")
    mz, y = mz[order], y[order]
    new = np.concatenate([[True], np.diff(mz) > mz[1:] * ppm / 1e6])
    group = np.cumsum(new) - 1
    total = np.bincount(group, weights=y)
    weight = np.where(total > 0, total, 1.0)
    centre = np.bincount(group, weights=mz * np.where(y > 0, y, 0.0))
    plain = np.bincount(group, weights=mz) / np.bincount(group)
    out_mz = np.where(total > 0, centre / weight, plain)
    return out_mz, total


# --- runs, targets and the spectra they give --------------------------------------------------

@dataclass(frozen=True)
class MzmlRun:
    """One mzML file of an experiment: its ``sample`` and replicate ``rep``. ``time`` is the
    labelling time of the whole file (an LC-MS file per time point, or a control file at the
    control's time), or None when every scan has its own time (the scan time plus
    ``time_offset``, in seconds)."""

    path: str | Path
    sample: str
    rep: int = 1
    time: float | None = None
    time_offset: float = 0.0


@dataclass(frozen=True)
class MzmlTarget:
    """What to extract: ``entry`` (peptide, charge, modification with the ion type, range) and
    which scans hold it: ``ms_level`` (1 for peptides and the intact protein, 2 for fragments),
    ``activation`` (for example ``ETD``), the precursor (``precursor_mz`` within the config's
    tolerance, ``precursor_charge``), the filter string (``filter_contains``,
    ``filter_excludes``) and ``rt_window`` (seconds; for a run with a fixed time the scans summed
    are those inside it)."""

    entry: PeptideEntry
    ms_level: int = 1
    activation: str | None = None
    precursor_mz: float | None = None
    precursor_charge: int | None = None
    rt_window: tuple[float, float] | None = None
    filter_contains: str | None = None
    filter_excludes: str | None = None


@dataclass(frozen=True)
class MzmlConfig:
    """How scans become spectra.

    For a run without a fixed time, ``group`` is ``'each'`` (one spectrum per scan), ``'every'``
    (``every`` consecutive selected scans summed) or ``'windows'`` (the scans inside each of the
    ``windows``, in seconds of scan time, summed). ``mz_margin`` is the m/z kept on either side
    of a target's predicted bins. ``sum_ppm`` merges centroided peaks; ``gap_factor`` finds the
    left-out stretches of profile scans; ``precursor_tol`` (m/z) matches a target's precursor.
    """

    group: str = "each"
    every: int = 1
    windows: tuple[tuple[float, float], ...] = ()
    min_time: float | None = None
    max_spectra: int | None = None
    mz_margin: float = 2.0
    sum_ppm: float = 10.0
    gap_factor: float = 3.0
    precursor_tol: float = 1.0

    def __post_init__(self):
        if self.group not in ("each", "every", "windows"):
            raise ValueError("group must be 'each', 'every' or 'windows'")
        if self.every < 1:
            raise ValueError("every must be at least 1")
        object.__setattr__(self, "windows", tuple((float(a), float(b)) for a, b in self.windows))
        if self.group == "windows" and not self.windows:
            raise ValueError("group='windows' needs windows")
        if any(b < a for a, b in self.windows):
            raise ValueError("a window must end after it starts")
        if self.max_spectra is not None and self.max_spectra < 2:
            raise ValueError("max_spectra must be at least 2")
        if (self.mz_margin < 0 or self.sum_ppm <= 0 or self.gap_factor <= 1
                or self.precursor_tol < 0):
            raise ValueError("need mz_margin >= 0, sum_ppm > 0, gap_factor > 1 and "
                             "precursor_tol >= 0")


def _groups(times: np.ndarray, config: MzmlConfig) -> list[np.ndarray]:
    """Positions (into the selected scans, in time order) of the scans of each spectrum."""
    n = len(times)
    if config.group == "each":
        return [np.array([i]) for i in range(n)]
    if config.group == "every":
        return [np.arange(i, min(i + config.every, n)) for i in range(0, n, config.every)]
    out = []
    for a, b in config.windows:
        inside = np.flatnonzero((times >= a) & (times <= b))
        if len(inside):
            out.append(inside)
    return out


def _sample_log(times: np.ndarray, n: int) -> np.ndarray:
    """Positions of at most ``n`` of the sorted ``times``, spaced evenly in log(time - first),
    always including the first and the last."""
    if len(times) <= n:
        return np.arange(len(times))
    rel = times - times[0]
    step = np.median(np.diff(times)) if len(times) > 1 else 1.0
    goal = np.concatenate([[0.0], np.geomspace(max(step, 1e-9), rel[-1], n - 1)])
    picks = sorted({int(np.argmin(np.abs(rel - g))) for g in goal} | {0, len(times) - 1})
    return np.array(picks[:n] if len(picks) > n else picks, dtype=np.int64)


def _target_window(target: MzmlTarget, picker: PickerConfig, registry, margin: float):
    e = target.entry
    plan = plan_bins(e.peptide, e.charge, modification=e.modification, config=picker,
                     registry=registry)
    return float(plan.pred_mz.min()) - margin, float(plan.pred_mz.max()) + margin


def _target_label(e: PeptideEntry) -> str:
    mod = f"-{e.modification.replace(' ', '_')}" if e.modification else ""
    return f"{e.peptide_range}-{e.peptide}-z{e.charge}{mod}"


def read_mzml(
    runs: Sequence[MzmlRun],
    targets: Sequence[MzmlTarget],
    *,
    project: str,
    config: MzmlConfig = MzmlConfig(),  # noqa: B008
    picker: PickerConfig = DEFAULT_PICKER_CONFIG,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> SpectrumSet:
    """Raw spectra of every target in every run (see the module docstring for the two layouts).

    Each run is read once, keeping only the points inside the targets' m/z windows (the
    predicted bins of ``picker`` plus ``config.mz_margin``). A series (``data_id``) is a sample and
    a target; replicate runs of a sample share it. A target with no matching scan in a run is
    skipped with a warning. ``spectrum_id`` is ``PROJECT:SAMPLE/FILE/TARGET/scan<i>`` (or
    ``scans<first>-<last>`` for a sum).
    """
    qualify_spectrum_id(project, "")
    if not runs or not targets:
        raise ValueError("read_mzml needs at least one run and one target")
    windows = [_target_window(t, picker, registry, config.mz_margin) for t in targets]
    series: dict[tuple[str, int], int] = {}
    spectra: list[Spectrum] = []
    for n_run, run in enumerate(runs):
        check_cancel(cancel)
        f = MzmlFile(run.path)
        table = f.scans()
        chosen = [
            table.select(ms_level=t.ms_level, activation=t.activation,
                         precursor_mz=t.precursor_mz, precursor_tol=config.precursor_tol,
                         precursor_charge=t.precursor_charge,
                         time_range=t.rt_window if run.time is not None else (
                             None if config.min_time is None else (config.min_time, np.inf)),
                         filter_contains=t.filter_contains, filter_excludes=t.filter_excludes)
            for t in targets
        ]
        # the scans of every spectrum, per target, before anything is read (so that sampling a
        # large file reads only the scans it keeps)
        plans: list[tuple[np.ndarray, list[np.ndarray]] | None] = []
        for rows in chosen:
            if not len(rows):
                plans.append(None)
                continue
            order = rows[np.argsort(table.time[rows], kind="stable")]
            groups = [np.arange(len(order))] if run.time is not None else _groups(
                table.time[order], config)
            if run.time is None and config.max_spectra is not None and len(groups) > 1:
                centre = np.array([table.time[order][g].mean() for g in groups])
                groups = [groups[i] for i in _sample_log(centre, config.max_spectra)]
            plans.append((order, groups))
        needed = {int(table.index[p[0][i]]) for p in plans if p is not None for g in p[1]
                  for i in g}
        if not needed:
            log.warning("%s: no scan matches any target", f.path.name)
            report(progress, n_run + 1, len(runs))
            continue
        by_index = {s.index: s for s in iter_scans(f.path, mz_windows=windows,
                                                   keep=lambda m, wanted=needed: m.index in wanted)}
        for t_i, (target, plan) in enumerate(zip(targets, plans, strict=True)):
            if plan is None:
                log.warning("%s: no scan for target %s", f.path.name, _target_label(target.entry))
                continue
            lo, hi = windows[t_i]
            key = (run.sample, t_i)
            data_id = series.setdefault(key, len(series))
            entry = replace(target.entry, sample=run.sample)
            order, groups = plan
            for g in groups:
                scans = [by_index[int(table.index[order[i]])] for i in g]
                parts = []
                for s in scans:
                    inside = (s.mz >= lo) & (s.mz <= hi)
                    parts.append((s.mz[inside], s.intensity[inside]))
                centroid = all(s.centroid for s in scans)
                if len(parts) == 1:
                    mz, y = parts[0]
                elif centroid:
                    mz, y = sum_centroid(parts, config.sum_ppm)
                else:
                    mz, y = sum_profile(parts, config.gap_factor)
                order_mz = np.argsort(mz, kind="stable")
                mz, y = np.asarray(mz, float)[order_mz], np.asarray(y, float)[order_mz]
                first, last = scans[0].index, scans[-1].index
                scan_text = f"scan{first}" if len(scans) == 1 else f"scans{first}-{last}"
                time = (float(run.time) if run.time is not None
                        else float(np.mean([s.time for s in scans])) + run.time_offset)
                local = f"{run.sample}/{f.path.name}/{_target_label(entry)}/{scan_text}"
                spectra.append(Spectrum(
                    mz, y, entry, time, run.rep, data_id=data_id,
                    label=f"{f.path.name} {scan_text}",
                    spectrum_id=qualify_spectrum_id(project, local),
                ))
        report(progress, n_run + 1, len(runs))
    if not spectra:
        raise MzmlError("no spectra: no scan matched any target in any run")
    return SpectrumSet.from_spectra(spectra)
