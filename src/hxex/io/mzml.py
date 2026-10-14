# SPDX-License-Identifier: GPL-3.0-or-later
"""mzML input: scans, summing, and spectra for peptides or fragments.

A small mzML 1.1 reader written with the standard library (no lxml): plain or indexed files,
uncompressed or zlib-compressed 32- or 64-bit arrays, profile or centroid scans. Numpress
compression is not supported and is reported. Thermo ``.raw`` files are read directly by
:mod:`hxex.io.thermo` (with Thermo's RawFileReader), or converted first, for example with
ThermoRawFileParser (``mono ThermoRawFileParser.exe -i=FILE.raw -b=FILE.mzML -f=1 -p``; ``-f=1``
writes mzML and ``-p`` keeps profile data), which gives the same scans.

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
the same sums made by OpenMS (see docs/HISTORY.md item 33).
"""

from __future__ import annotations

import base64
import json
import logging
import re
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .. import chem
from ..hooks import CancelHook, ProgressHook, check_cancel, report
from ..picking import DEFAULT_PICKER_CONFIG, PickerConfig, plan_bins
from ..spectra import PeptideEntry, Spectrum, SpectrumSet, qualify_spectrum_id
from .errors import ReadError
from .identity import CHANGE_ACTIONS, compare_identity, file_identity

log = logging.getLogger(__name__)

_NS = "{http://psi.hupo.org/ms/mzml}"
# controlled-vocabulary accessions used here
_MS_LEVEL = "MS:1000511"
_SCAN_TIME = "MS:1000016"
_FILTER = "MS:1000512"
_SELECTED_MZ = "MS:1000744"
_CHARGE = "MS:1000041"
_ISOLATION_TARGET = "MS:1000827"
_ISOLATION_LOWER = "MS:1000828"
_ISOLATION_UPPER = "MS:1000829"
_COLLISION_ENERGY = "MS:1000045"
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
# short names of the activation methods: the exact sets of activation cvParam names a scan of
# that kind has (Thermo writes EThcD as ETD plus supplemental beam-type CID; some converters use
# the single EThcD term)
ACTIVATIONS = {
    "ETD": ({"electron transfer dissociation"},),
    "ECD": ({"electron capture dissociation"},),
    "CID": ({"collision-induced dissociation"},),
    "HCD": ({"beam-type collision-induced dissociation"},),
    "ETHCD": ({"electron transfer dissociation",
               "supplemental beam-type collision-induced dissociation"},
              {"electron transfer/higher-energy collision dissociation"}),
    "ETCID": ({"electron transfer dissociation", "supplemental collision-induced dissociation"},),
    "UVPD": ({"ultraviolet photodissociation"},),
}


class MzmlError(ReadError):
    """An mzML file that cannot be read."""


@dataclass(frozen=True)
class Stage:
    """One isolation and activation stage of an MSn scan: the precursor ``mz``, its isolation
    window ``isolation`` (absolute m/z, NaN when not stated), the ``activation`` names (lower
    case, as :attr:`Scan.activation`) and the ``energy`` of the stage's main activation (as the
    instrument states it; NaN when not stated)."""

    mz: float
    isolation: tuple[float, float] = (np.nan, np.nan)
    activation: tuple[str, ...] = ()
    energy: float = np.nan


@dataclass(frozen=True)
class Scan:
    """One scan: its position ``index`` and ``id`` in the file, ``ms_level``, ``time`` (s),
    ``filter`` (the vendor filter string, may be empty), ``precursor_mz`` and
    ``precursor_charge`` (NaN and 0 without a precursor), ``activation`` (names of the activation
    methods, lower case), ``centroid`` (True, False, or None if not stated), the arrays (empty
    when they were not read) and ``isolation``, the precursor isolation window as absolute m/z
    (lower, upper; NaN when the file does not state it).

    The precursor, activation and isolation are those of the last stage, the one that made the
    scan. ``stages`` lists every stage, first to last (empty for MS1): the first is the ion
    isolated from the full scan (the parent: for a bottom-up or top-down experiment, the peptide
    or protein), and the activations of the stages before the last are the pre-activation (an
    MS3 scan that heats the precursor with HCD, isolates it again and fragments it with ETD).
    An MSn scan whose earlier stages are not stated has fewer than ``ms_level - 1`` stages."""

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
    isolation: tuple[float, float] = (np.nan, np.nan)
    stages: tuple[Stage, ...] = ()


# Thermo filter-string activation codes, as mzML activation names
_FILTER_ACTIVATIONS = {
    "cid": "collision-induced dissociation",
    "hcd": "beam-type collision-induced dissociation",
    "etd": "electron transfer dissociation",
    "ecd": "electron capture dissociation",
    "uvpd": "ultraviolet photodissociation",
    "pqd": "pulsed q dissociation",
}
_FILTER_STAGE = re.compile(r"(\d+(?:\.\d+)?)((?:@[a-z]+-?\d+(?:\.\d+)?)+)")
_FILTER_ACT = re.compile(r"@([a-z]+)(-?\d+(?:\.\d+)?)")


def filter_stages(filt: str) -> tuple[Stage, ...]:
    """The stages written in a Thermo filter string (``... ms3 313.0100@hcd15.00
    313.0000@etd25.00 [...]``): each precursor m/z with its activations and the energy of the
    first; a further activation of a stage is supplemental when the filter has ``sa`` (EThcD,
    ``etd50.00@hcd25.00``). The isolation windows are not in the filter (NaN). Empty when the
    filter states no stage."""
    m = re.search(r"\bms(\d+)\b(.*?)(\[|$)", filt)
    if not m or int(m.group(1)) < 2:
        return ()
    supplemental = bool(re.search(r"\bsa\b", filt[:m.start()]))
    stages = []
    for mz, acts in _FILTER_STAGE.findall(m.group(2)):
        codes = _FILTER_ACT.findall(acts)
        names = []
        for i, (code, _) in enumerate(codes):
            name = _FILTER_ACTIVATIONS.get(code, code)
            if i and supplemental and code in ("cid", "hcd"):
                name = "supplemental " + name
            names.append(name)
        stages.append(Stage(float(mz), activation=tuple(sorted(set(names))),
                            energy=float(codes[0][1])))
    return tuple(stages)


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
    pmz, pz, activation, isolation = np.nan, 0, (), (np.nan, np.nan)
    stages = []
    # several precursors (MSn written stage by stage) are taken as first to last
    for prec in elem.findall(f"{_NS}precursorList/{_NS}precursor"):
        pmz, activation, isolation, energy = np.nan, (), (np.nan, np.nan), np.nan
        window = prec.find(f"{_NS}isolationWindow")
        if window is not None:
            wcv = _cv(window)
            if _ISOLATION_TARGET in wcv:
                centre = float(wcv[_ISOLATION_TARGET][0])
                isolation = (centre - float(wcv.get(_ISOLATION_LOWER, ("nan", ""))[0]),
                             centre + float(wcv.get(_ISOLATION_UPPER, ("nan", ""))[0]))
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
            energy = float(_cv(act).get(_COLLISION_ENERGY, ("nan", ""))[0] or "nan")
        stages.append(Stage(pmz, isolation, activation, energy))
    if ms_level > 2 and 0 < len(stages) < ms_level - 1:
        # the earlier stages from the filter string (ThermoRawFileParser writes the last only)
        written = filter_stages(filt)
        if len(written) == ms_level - 1:
            stages = list(written[:ms_level - 1 - len(stages)]) + stages
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
    return Scan(index, sid, ms_level, time, filt, pmz, pz, activation, centroid, mz, intensity,
                isolation, tuple(stages))


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


def _parent(s: Scan) -> Stage:
    """The first stage of ``s`` (its precursor for MS2; NaN when not stated or for MS1)."""
    if s.ms_level == 2:
        return Stage(s.precursor_mz, s.isolation)
    if s.ms_level > 2 and len(s.stages) == s.ms_level - 1:
        return s.stages[0]
    return Stage(np.nan)


def _preactivation(s: Scan) -> tuple[str, float]:
    """The activation names (joined with ``;``) and the energy of the stages of ``s`` before the
    last (``''`` for MS1 and MS2, ``'?'`` when the file does not state them)."""
    if s.ms_level <= 2:
        return "", np.nan
    if len(s.stages) != s.ms_level - 1:
        return "?", np.nan
    names = sorted({n for st in s.stages[:-1] for n in st.activation})
    return ";".join(names), s.stages[0].energy


@dataclass(frozen=True)
class ScanTable:
    """The scans of a file without their arrays, as columns: ``index``, ``id``, ``ms_level``,
    ``time`` (s), ``filter``, ``precursor_mz``, ``precursor_charge``, ``activation`` (names
    joined with ``;``), ``centroid`` (1, 0, or -1 if not stated) and the precursor isolation
    window, ``isolation_lower`` and ``isolation_upper`` (absolute m/z, NaN if not stated), all of
    the last stage; ``energy``, the energy of the last stage's main activation (NaN if not
    stated); the parent (the first stage, see :class:`Scan`): ``parent_mz``, ``parent_lower`` and
    ``parent_upper`` (the same as the precursor for MS2; NaN for MS1, for an MSn scan whose
    earlier stages are not stated, and for an isolation window not stated); and the
    pre-activation (the activations of the stages before the last): ``preactivation`` (their
    names joined with ``;``, empty for none, ``?`` when the file does not state them) and
    ``preactivation_energy`` (of the first stage, NaN for none)."""

    index: np.ndarray
    id: np.ndarray
    ms_level: np.ndarray
    time: np.ndarray
    filter: np.ndarray
    precursor_mz: np.ndarray
    precursor_charge: np.ndarray
    activation: np.ndarray
    centroid: np.ndarray
    isolation_lower: np.ndarray
    isolation_upper: np.ndarray
    energy: np.ndarray
    parent_mz: np.ndarray
    parent_lower: np.ndarray
    parent_upper: np.ndarray
    preactivation: np.ndarray
    preactivation_energy: np.ndarray

    def __len__(self) -> int:
        return len(self.index)

    @classmethod
    def from_scans(cls, rows: Sequence[Scan]) -> ScanTable:
        """The table of ``rows`` (their arrays are not kept)."""
        parents = [_parent(s) for s in rows]
        pre = [_preactivation(s) for s in rows]
        return cls(
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
            isolation_lower=np.array([s.isolation[0] for s in rows], dtype=np.float64),
            isolation_upper=np.array([s.isolation[1] for s in rows], dtype=np.float64),
            energy=np.array([s.stages[-1].energy if s.stages else np.nan for s in rows],
                            dtype=np.float64),
            parent_mz=np.array([p.mz for p in parents], dtype=np.float64),
            parent_lower=np.array([p.isolation[0] for p in parents], dtype=np.float64),
            parent_upper=np.array([p.isolation[1] for p in parents], dtype=np.float64),
            preactivation=np.array([p[0] for p in pre], dtype=str),
            preactivation_energy=np.array([p[1] for p in pre], dtype=np.float64),
        )

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
        parent_mz: tuple[float, float] | None = None,
        preactivation: str | None = None,
    ) -> np.ndarray:
        """Positions (into this table) of the scans that match every condition given.
        ``activation`` is a short name, matched exactly (``ETD`` is ETD alone, not EThcD; ``HCD``,
        ``CID``, ``ECD``, ``EThcD``, ``ETciD``, ``UVPD``), or other text, found in any of the
        scan's activation names; ``time_range`` is in seconds, inclusive;
        ``filter_contains`` and ``filter_excludes`` test the vendor filter string (for example
        Thermo's ``sid=`` marks scans with in-source activation).

        ``parent_mz`` is the m/z range of the parent ion (lowest and highest m/z of its isotope
        envelope, undeuterated to fully deuterated): a scan matches when its parent's isolation
        window overlaps the range, or, when the window is not stated, when its parent m/z is
        within ``precursor_tol`` of it. ``preactivation`` is matched as ``activation`` against the
        activations before the last stage; ``'none'`` (or ``''``) asks for scans without one (MS2).
        A scan whose earlier stages are not stated matches neither condition."""
        keep = np.ones(len(self), dtype=bool)
        if ms_level is not None:
            keep &= self.ms_level == ms_level
        if activation is not None:
            keep &= _activation_match(self.activation, activation)
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
        if parent_mz is not None:
            lo, hi = parent_mz
            window = ~np.isnan(self.parent_lower) & ~np.isnan(self.parent_upper)
            with np.errstate(invalid="ignore"):
                overlap = (self.parent_lower <= hi) & (self.parent_upper >= lo)
                near = (self.parent_mz >= lo - precursor_tol) & (
                    self.parent_mz <= hi + precursor_tol)
            keep &= np.where(window, overlap, near)
        if preactivation is not None:
            known = self.preactivation != "?"
            if preactivation.strip().lower() in ("", "none"):
                keep &= self.preactivation == ""
            else:
                keep &= known & _activation_match(self.preactivation, preactivation)
        return np.flatnonzero(keep)


def _activation_match(column: np.ndarray, activation: str) -> np.ndarray:
    """Which rows of an activation column (names joined with ``;``) match ``activation`` (see
    :meth:`ScanTable.select`)."""
    names = [set(filter(None, str(a).split(";"))) for a in column]
    exact = ACTIVATIONS.get(activation.upper())
    if exact is not None:  # the scan's activation is exactly that method
        return np.array([any(n == e for e in exact) for n in names], dtype=bool)
    want = activation.lower()  # free text: found in any activation name
    return np.array([any(want in x for x in n) for n in names], dtype=bool)


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
            self._table = ScanTable.from_scans(list(iter_scans(self.path, read_arrays=False)))
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

    def read_scans(
        self, indices, mz_windows: Sequence[tuple[float, float]] | None = None
    ) -> dict[int, Scan]:
        """The scans at the positions ``indices``, with their arrays (only the points inside
        ``mz_windows`` when given), by position; read in one pass through the file."""
        wanted = {int(i) for i in indices}
        return {s.index: s for s in iter_scans(self.path, mz_windows=mz_windows,
                                               keep=lambda m: m.index in wanted)}

    def close(self) -> None:
        """Nothing to release (the file is opened for each read); as :class:`ThermoFile`."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

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

# a run's scan condition that is left to the targets
TARGET = "target"
SCAN_CONDITIONS = ("activation", "precursor_mz", "precursor_charge", "filter_contains",
                   "filter_excludes")


@dataclass(frozen=True)
class MzmlRun:
    """One mzML file of an experiment: its ``sample`` and replicate ``rep``. ``time`` is the
    labelling time of the whole file (an LC-MS file per time point, or a control file at the
    control's time), or None when every scan has its own time (the scan time plus
    ``time_offset``, in seconds).

    Two clocks: the **acquisition clock** is the scan time the file records (from the start of
    acquisition); the **exchange clock** is the time of the spectra, the scan time plus
    ``time_offset`` (when exchange started relative to acquisition, for example the dead time
    before the first scan). ``usable_from`` (seconds on the acquisition clock, set by the user
    for each file) leaves out the scans before it, for example spurious points while the
    infusion settles; it is not the same as ``time_offset``, which only moves the time axis.

    The scan conditions ``activation``, ``precursor_mz``, ``precursor_charge``,
    ``filter_contains`` and ``filter_excludes`` are the targets' (``'target'``, the default)
    unless set here, which replaces that condition of every target for this file only (None
    removes it). For example, a control file acquired with a constant ``sid=1.00`` takes
    ``filter_excludes=None`` while the kinetics files exclude ``sid``, and a file that records a
    wrong precursor charge takes ``precursor_charge=None``."""

    path: str | Path
    sample: str
    rep: int = 1
    time: float | None = None
    time_offset: float = 0.0
    usable_from: float | None = None
    activation: str | None = TARGET
    precursor_mz: float | str | None = TARGET
    precursor_charge: int | str | None = TARGET
    filter_contains: str | None = TARGET
    filter_excludes: str | None = TARGET

    def __post_init__(self):
        for name in ("precursor_mz", "precursor_charge"):
            if isinstance(getattr(self, name), str) and getattr(self, name) != TARGET:
                raise ValueError(f"{name} must be a number, None or {TARGET!r}")

    def conditions(self, target: MzmlTarget) -> dict:
        """The scan conditions of ``target`` in this file (this run's where it sets them)."""
        return {name: getattr(target, name) if getattr(self, name) == TARGET
                else getattr(self, name) for name in SCAN_CONDITIONS}

    def to_dict(self) -> dict:
        out = {"path": str(self.path), "sample": self.sample, "rep": int(self.rep),
               **({"usable_from": float(self.usable_from)} if self.usable_from is not None
                  else {}),
               "time": None if self.time is None else float(self.time),
               "time_offset": float(self.time_offset)}
        out.update({name: getattr(self, name) for name in SCAN_CONDITIONS
                    if getattr(self, name) != TARGET})  # only those set, as older records
        return out

    @classmethod
    def from_dict(cls, data: Mapping) -> MzmlRun:
        return cls(**_known(cls, data, "run"))


@dataclass(frozen=True)
class MzmlTarget:
    """What to extract: ``entry`` (peptide, charge, modification with the ion type, range) and
    which scans hold it: ``ms_level`` (1 for peptides and the intact protein, 2 for fragments),
    ``activation`` (for example ``ETD``), the precursor (``precursor_mz`` within the config's
    tolerance, ``precursor_charge``), the filter string (``filter_contains``,
    ``filter_excludes``) and ``rt_window`` (seconds; for a run with a fixed time the scans summed
    are those inside it). ``activation`` and the precursor are those of the last stage (the one
    that made the scan). For fragments, ``parent_mz`` is the m/z range of the parent ion (the
    peptide isolated from the full scan; see :meth:`ScanTable.select`), which ties the
    fragment's scans to its own parent when a file holds several precursors, and
    ``preactivation`` the activation before the last stage (``'HCD'`` for an MS3 scan that heats
    the precursor with HCD before ETD; ``'none'`` for none)."""

    entry: PeptideEntry
    ms_level: int = 1
    activation: str | None = None
    precursor_mz: float | None = None
    precursor_charge: int | None = None
    rt_window: tuple[float, float] | None = None
    filter_contains: str | None = None
    filter_excludes: str | None = None
    parent_mz: tuple[float, float] | None = None
    preactivation: str | None = None

    def __post_init__(self):
        if self.parent_mz is not None:
            lo, hi = (float(x) for x in self.parent_mz)
            if not lo <= hi:
                raise ValueError(f"parent_mz must be a range (low, high), got {self.parent_mz}")
            object.__setattr__(self, "parent_mz", (lo, hi))

    def to_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        e = self.entry
        out["entry"] = e.to_dict()
        out["rt_window"] = None if self.rt_window is None else list(self.rt_window)
        for name in ("parent_mz", "preactivation"):  # only when set, as older records
            if out[name] is None:
                del out[name]
        if self.parent_mz is not None:
            out["parent_mz"] = list(self.parent_mz)
        return out

    @classmethod
    def from_dict(cls, data: Mapping) -> MzmlTarget:
        values = _known(cls, data, "target")
        values["entry"] = PeptideEntry.from_dict(values["entry"])
        if values.get("rt_window") is not None:
            values["rt_window"] = tuple(float(x) for x in values["rt_window"])
        if values.get("parent_mz") is not None:
            values["parent_mz"] = tuple(float(x) for x in values["parent_mz"])
        return cls(**values)


@dataclass(frozen=True)
class MzmlConfig:
    """How scans become spectra.

    For a run without a fixed time, ``group`` is ``'each'`` (one spectrum per scan), ``'every'``
    (``every`` consecutive selected scans summed) or ``'windows'`` (the scans inside each of the
    ``windows``, in seconds of scan time, summed). ``min_time`` leaves out, in every such run, the
    scans before it; ``windows`` and ``min_time`` are on the acquisition clock (the scan time,
    before a run's ``time_offset``; see :class:`MzmlRun`, whose ``usable_from`` sets a start per
    file). ``mz_margin`` is the m/z kept on either side
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

    def to_dict(self) -> dict:
        out = asdict(self)
        out["windows"] = [list(w) for w in self.windows]
        return out

    @classmethod
    def from_dict(cls, data: Mapping) -> MzmlConfig:
        values = _known(cls, data, "mzML setting")
        values["windows"] = tuple(tuple(w) for w in values.get("windows", ()))
        return cls(**values)


def _known(cls, data: Mapping, what: str) -> dict:
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {what} field(s): {', '.join(sorted(unknown))}")
    return dict(data)


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


def _target_label(t: MzmlTarget) -> str:
    """The target in spectrum ids: the entry, then (for fragments) how the scans were chosen,
    for example ``1-5-RDKVQ-z1-ion_type:c@ETD-pz3``, or ``...-ms3-pre:HCD@ETD-pz3`` for MS3 scans
    with HCD before ETD (the parent's m/z range is not part of it: the entry names the parent)."""
    e = t.entry
    mod = f"-{e.modification.replace(' ', '_')}" if e.modification else ""
    scans = "".join([
        f"-ms{t.ms_level}" if t.ms_level > 2 else "",
        f"-pre:{t.preactivation}" if t.preactivation is not None else "",
        f"@{t.activation}" if t.activation else "",
        f"-p{t.precursor_mz:g}" if t.precursor_mz is not None else "",
        f"-pz{t.precursor_charge}" if t.precursor_charge is not None else "",
    ])
    condition = dict(e.extra).get("condition")
    return f"{e.peptide_range}-{e.peptide}-z{e.charge}{mod}{scans}" + (
        f"-{condition.replace(' ', '_')}" if condition else "")


PATH_MODES = ("path", "relative", "name")


def _recorded_path(path: Path, data_root: Path | None, mode: str) -> str:
    if mode == "name":
        return path.name
    if mode == "relative":
        try:
            return path.resolve().relative_to(data_root.resolve()).as_posix()
        except ValueError:
            raise ValueError(f"{path} is not inside the data root {data_root}") from None
    return str(path)


def read_mzml(
    runs: Sequence[MzmlRun],
    targets: Sequence[MzmlTarget],
    *,
    project: str,
    config: MzmlConfig = MzmlConfig(),  # noqa: B008
    picker: PickerConfig = DEFAULT_PICKER_CONFIG,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
    data_root: str | Path | None = None,
    record_paths: str = "path",
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> SpectrumSet:
    """Raw spectra of every target in every run (see the module docstring for the two layouts).

    Each run is read once, keeping only the points inside the targets' m/z windows (the
    predicted bins of ``picker`` plus ``config.mz_margin``). A series (``data_id``) is a sample and
    a target; replicate runs of a sample share it. A target with no matching scan in a run is
    skipped with a warning. ``spectrum_id`` is ``PROJECT:SAMPLE/FILE/TARGET/scan<i>`` (or
    ``scans<first>-<last>`` for a sum), where ``TARGET`` names the entry and, when given, the
    activation and the precursor (``@ETD-pz3``).

    The spectra's ``source`` records the arguments (runs, targets, ``config``, ``picker``,
    ``registry`` and ``project``), so that :func:`read_mzml_source` repeats the read, the target
    of every series (``series``: ``data_id`` to sample and target position; see
    :func:`mzml_series_targets`), which ties each result to its precursor and activation, and
    the identity of every file (``files``, see :func:`file_identity`), checked when the read is
    repeated.

    A relative run path is taken relative to ``data_root`` (when given). ``record_paths`` says
    how run paths are recorded: ``'path'`` as given, ``'relative'`` relative to ``data_root``
    (every file must be inside it), ``'name'`` the file name only (nothing about where the data
    lives, for results to be shared; the read is repeated with ``directory``).
    """
    opened, recorded = _run_paths(runs, data_root, record_paths)
    return _read_runs(_MZML, runs, opened, recorded, None, record_paths, targets,
                      project=project, config=config, picker=picker, registry=registry,
                      progress=progress, cancel=cancel)


def _run_paths(runs: Sequence[MzmlRun], data_root, record_paths: str):
    """The paths to open and the paths to record of ``runs`` (see :func:`read_mzml`)."""
    if record_paths not in PATH_MODES:
        raise ValueError(f"record_paths must be one of {', '.join(PATH_MODES)}")
    if record_paths == "relative" and data_root is None:
        raise ValueError("record_paths='relative' needs a data_root")
    root = None if data_root is None else Path(data_root)
    opened = [Path(r.path) if root is None else root / r.path for r in runs]
    return opened, [_recorded_path(p, root, record_paths) for p in opened]


@dataclass(frozen=True)
class _ScanFormat:
    """A format of scan files read with runs and targets: the ``reader`` name recorded in the
    spectra's source, ``what`` (for messages), ``open`` (path -> a context manager with
    ``path``, ``scans()`` and ``read_scans()``, as :class:`MzmlFile`), the ``error`` raised, and
    ``extra`` (open file -> more for the record, called for the first file)."""

    reader: str
    what: str
    open: object
    error: type
    extra: object = None


_MZML = _ScanFormat("mzml", "an mzML", MzmlFile, MzmlError)
# the readers whose spectra carry runs, targets and series (:func:`read_mzml` and
# :func:`hxex.io.read_thermo`)
SCAN_READERS = ("mzml", "thermo")


def _read_runs(
    fmt: _ScanFormat,
    runs: Sequence[MzmlRun],
    opened: Sequence[Path],
    recorded: Sequence[str],
    identities: Sequence[dict] | None,
    record_paths: str,
    targets: Sequence[MzmlTarget],
    *,
    project: str,
    config: MzmlConfig,
    picker: PickerConfig,
    registry: chem.ModificationRegistry,
    progress: ProgressHook | None,
    cancel: CancelHook | None,
) -> SpectrumSet:
    """:func:`read_mzml` (or another format's read) with the files to open (``opened``), the
    paths to record, and the file identities when they are already known (else they are
    computed)."""
    qualify_spectrum_id(project, "")
    if not runs or not targets:
        raise ValueError(f"read_{fmt.reader} needs at least one run and one target")
    windows = [_target_window(t, picker, registry, config.mz_margin) for t in targets]
    series: dict[tuple[str, int], int] = {}
    spectra: list[Spectrum] = []
    extra: dict = {}
    for n_run, run in enumerate(runs):
        check_cancel(cancel)
        with fmt.open(opened[n_run]) as f:
            if fmt.extra is not None and not extra:
                extra = fmt.extra(f)
            _read_run(f, run, targets, windows, series, spectra, project=project,
                      config=config)
        report(progress, n_run + 1, len(runs))
    if not spectra:
        raise fmt.error("no spectra: no scan matched any target in any run")
    source = {
        "reader": fmt.reader,
        "project": project,
        "runs": [replace(r, path=p).to_dict() for r, p in zip(runs, recorded, strict=True)],
        "record_paths": record_paths,
        "files": list(identities) if identities is not None else [
            file_identity(p) for p in opened],
        "targets": [t.to_dict() for t in targets],
        "config": config.to_dict(),
        "picker": picker.to_dict(),
        "registry": registry.to_dict(),
        "series": [{"data_id": d, "sample": sample, "target": t_i}
                   for (sample, t_i), d in sorted(series.items(), key=lambda kv: kv[1])],
        **extra,
    }
    # as stored (tuples become lists), so a record read back from a file compares equal
    return SpectrumSet.from_spectra(spectra, source=json.loads(json.dumps(source)))


def target_scans(table: ScanTable, run: MzmlRun, target: MzmlTarget,
                 config: MzmlConfig) -> np.ndarray:
    """Positions (in ``table``, one file's scans) of the scans of ``target`` in ``run``, as
    :func:`read_mzml` chooses them: the MS level and the scan conditions (the run's where it sets
    them), within the retention window for a run with a fixed time, or from ``config.min_time``
    otherwise, and from the run's ``usable_from`` (all on the acquisition clock: scan times)."""
    if run.time is not None:
        time_range = target.rt_window
    else:
        time_range = None if config.min_time is None else (config.min_time, np.inf)
    if run.usable_from is not None:
        lo, hi = time_range if time_range is not None else (-np.inf, np.inf)
        time_range = (max(lo, run.usable_from), hi)
    rows = table.select(ms_level=target.ms_level, precursor_tol=config.precursor_tol,
                        time_range=time_range, parent_mz=target.parent_mz,
                        preactivation=target.preactivation, **run.conditions(target))
    if target.ms_level > 2 and (target.parent_mz is not None or target.preactivation is not None):
        unknown = (table.ms_level == target.ms_level) & (table.preactivation == "?")
        if unknown.any():
            log.warning("%d MS%d scans do not state their earlier stages and are left out of %s",
                        int(unknown.sum()), target.ms_level, _target_label(target))
    return rows


def _read_run(f, run: MzmlRun, targets, windows, series, spectra, *, project, config) -> None:
    """Add the spectra of every target in the open file ``f`` (one run) to ``spectra``."""
    table = f.scans()
    chosen = [target_scans(table, run, t, config) for t in targets]
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
        return
    by_index = f.read_scans(sorted(needed), windows)
    for t_i, (target, plan) in enumerate(zip(targets, plans, strict=True)):
        if plan is None:
            log.warning("%s: no scan for target %s", f.path.name, _target_label(target))
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
            local = f"{run.sample}/{f.path.name}/{_target_label(target)}/{scan_text}"
            spectra.append(Spectrum(
                mz, y, entry, time, run.rep, data_id=data_id,
                label=f"{f.path.name} {scan_text}",
                spectrum_id=qualify_spectrum_id(project, local),
            ))


def mzml_series_targets(source: Mapping) -> dict[int, MzmlTarget]:
    """The target (entry, MS level, activation, precursor) of every series (``data_id``) of
    spectra read by :func:`read_mzml` or :func:`hxex.io.read_thermo`, from their ``source`` (or
    a bundle's ``spectra_source``)."""
    if source.get("reader") not in SCAN_READERS:
        raise ValueError(f"not an mzML or Thermo read: reader {source.get('reader')!r}")
    targets = [MzmlTarget.from_dict(t) for t in source["targets"]]
    return {int(s["data_id"]): targets[int(s["target"])] for s in source["series"]}


def read_mzml_source(
    source: Mapping,
    *,
    directory: str | Path | None = None,
    data_root: str | Path | None = None,
    on_change: str = "warn",
    progress: ProgressHook | None = None,
    cancel: CancelHook | None = None,
) -> SpectrumSet:
    """Repeat the read recorded in the ``source`` of spectra made by :func:`read_mzml` (from the
    spectra themselves, a spectra file or a results bundle).

    A file is looked up by name in ``directory`` when given (for data that has moved, or a
    record of names only), else at its recorded path, relative paths being taken relative to
    ``data_root`` (or the working directory). Each file's size and checksum
    (:func:`file_identity`) are compared with the record. A file that changed (data updated
    between runs) is read with a warning (``on_change='warn'``, the default) or refused
    (``'error'``, :class:`MzmlError`); either way the new spectra's ``source`` records the file
    as it is now, with the identity it had before under ``previous``, so the change stays
    visible in their spectra files and bundles. Unchanged, the record stays as it was."""
    return _repeat_read(_MZML, source, directory=directory, data_root=data_root,
                        on_change=on_change, progress=progress, cancel=cancel)


def _repeat_read(
    fmt: _ScanFormat,
    source: Mapping,
    *,
    directory: str | Path | None,
    data_root: str | Path | None,
    on_change: str,
    progress: ProgressHook | None,
    cancel: CancelHook | None,
) -> SpectrumSet:
    """:func:`read_mzml_source` for the format ``fmt``."""
    if source.get("reader") != fmt.reader:
        raise ValueError(f"not {fmt.what} read: reader {source.get('reader')!r}")
    if on_change not in CHANGE_ACTIONS:
        raise ValueError(f"on_change must be one of {', '.join(CHANGE_ACTIONS)}")
    runs = [MzmlRun.from_dict(r) for r in source["runs"]]  # "series" is derived, not an input
    files = source.get("files") or [None] * len(runs)
    root = None if data_root is None else Path(data_root)
    opened, identities = [], []
    for r, known in zip(runs, files, strict=True):
        check_cancel(cancel)
        p = Path(r.path)
        if directory is not None:
            p = Path(directory) / p.name
        elif not p.is_absolute() and root is not None:
            p = root / p
        if not p.is_file():
            raise fmt.error(f"{p} not found (give the directory that holds the data files)")
        now = compare_identity(p, known, on_change, fmt.error)
        opened.append(p)
        identities.append(now)
    recorded = [str(r.path) for r in runs]  # as the original read recorded them
    return _read_runs(
        fmt, runs, opened, recorded, identities, source.get("record_paths", "path"),
        [MzmlTarget.from_dict(t) for t in source["targets"]],
        project=source["project"], config=MzmlConfig.from_dict(source["config"]),
        picker=PickerConfig.from_dict(source["picker"]),
        registry=chem.ModificationRegistry.from_dict(source["registry"]),
        progress=progress, cancel=cancel,
    )
