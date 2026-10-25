# SPDX-License-Identifier: GPL-3.0-or-later
"""Arrow helpers shared by the results bundle and the standalone spectra file.

Light on purpose (numpy and pyarrow only), so that saving and loading spectra does not import
the fitting code or torch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..spectra import SpectrumSet


def _str(values) -> pa.Array:
    return pa.array([str(v) for v in values], type=pa.string())


# Arrow list columns index their values with int32 offsets
_MAX_LIST_VALUES = 2**31 - 1


def _list_offsets(offsets) -> pa.Array:
    """``offsets`` as the int32 offsets of a list column, refusing what would not fit."""
    offsets = np.asarray(offsets, dtype=np.int64)
    if len(offsets) and offsets[-1] > _MAX_LIST_VALUES:
        raise ValueError(f"a list column cannot hold {int(offsets[-1])} values (the limit is "
                         f"{_MAX_LIST_VALUES}); split the data into smaller sets")
    return pa.array(offsets.astype(np.int32), type=pa.int32())


def _ragged(arrays: Sequence[np.ndarray | None], value_type: pa.DataType) -> pa.Array:
    """A list column from one array per row (``None`` makes a null row)."""
    sizes = [0 if a is None else len(a) for a in arrays]
    offsets = _list_offsets(np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)]))
    parts = [np.asarray(a) for a in arrays if a is not None]
    flat = np.concatenate(parts) if parts else np.empty(0)
    mask = pa.array([a is None for a in arrays]) if any(a is None for a in arrays) else None
    return pa.ListArray.from_arrays(offsets, pa.array(flat, type=value_type), mask=mask)


def _flat_ragged(flat: np.ndarray, offsets: np.ndarray, value_type: pa.DataType) -> pa.Array:
    return pa.ListArray.from_arrays(_list_offsets(offsets), pa.array(flat, type=value_type))


def _rows(array2d: np.ndarray) -> list[np.ndarray]:
    """The rows of an ``(M, ...)`` array as flat arrays (for a list column)."""
    return [np.asarray(row).reshape(-1) for row in array2d]


def _col(table: pa.Table, name: str) -> np.ndarray:
    return table.column(name).to_numpy(zero_copy_only=False)


def _strs(table: pa.Table, name: str) -> np.ndarray:
    return np.array(table.column(name).to_pylist(), dtype=str)


def _ragged_read(table: pa.Table, name: str):
    """(flat values, offsets) of a list column; null rows count as empty."""
    col = table.column(name).combine_chunks()
    lengths = np.asarray(pc.fill_null(pc.list_value_length(col), 0).to_numpy(zero_copy_only=False))
    flat = col.flatten().to_numpy(zero_copy_only=False)
    return flat, np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


def _split(flat: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def _matrix(table: pa.Table, name: str, width: int, dtype=np.float64) -> np.ndarray:
    """A list column whose rows all have ``width`` values, as an ``(M, width)`` array."""
    flat, _ = _ragged_read(table, name)
    return np.asarray(flat, dtype=dtype).reshape(table.num_rows, width) if width else np.empty(
        (table.num_rows, 0), dtype=dtype
    )


_INDEX = ("spectrum_id", "sample", "peptide", "peptide_range", "file", "label", "modification")
_INDEX_INT = ("start_seq", "end_seq", "charge", "rep", "data_id")
# float columns a reader may leave unknown (NaN); absent in files written before them
_OPTIONAL = ("rt", "resolving_power", "resolving_power_mz", "rt_start", "rt_end")
# the role column ("undeut", "fulldeut" or ""): absent in files written before the roles, where
# the controls were known by their times (the reader derives it, see hxex.io.bundle)
ROLE = "role"


def _index_table(obj, optional: Sequence[str] = _OPTIONAL) -> dict[str, pa.Array]:
    """The index columns of a spectrum-like table ``obj`` (``rt`` and the other ``optional``
    float columns written when the object has them, NaN where unknown; ``role`` likewise)."""
    cols: dict[str, pa.Array] = {name: _str(getattr(obj, name)) for name in _INDEX}
    for name in _INDEX_INT:
        cols[name] = pa.array(np.asarray(getattr(obj, name), dtype=np.int64))
    cols["time"] = pa.array(np.asarray(obj.time, dtype=np.float64))
    for name in optional:
        if not hasattr(obj, name):
            continue
        values = getattr(obj, name)
        cols[name] = pa.array(np.asarray(values if values is not None
                                         else np.full(len(obj.time), np.nan), dtype=np.float64))
    role = getattr(obj, ROLE, None)
    cols[ROLE] = _str(role if role is not None else [""] * len(obj.time))
    return cols


def _index_arrays(t: pa.Table, optional: Sequence[str] = _OPTIONAL) -> dict[str, np.ndarray]:
    """The index columns of ``t`` as arrays (the ``optional`` float columns NaN when a table
    written before them has none; ``role`` all ``''`` then)."""
    out: dict[str, np.ndarray] = {name: _strs(t, name) for name in _INDEX}
    for name in _INDEX_INT:
        out[name] = np.asarray(_col(t, name), dtype=np.int64)
    out["time"] = np.asarray(_col(t, "time"), dtype=np.float64)
    for name in optional:
        out[name] = (np.asarray(_col(t, name), dtype=np.float64) if name in t.column_names
                     else np.full(t.num_rows, np.nan))
    out[ROLE] = _strs(t, ROLE) if ROLE in t.column_names else np.full(t.num_rows, "", dtype="<U8")
    return out


def _spectra_table(s: SpectrumSet) -> pa.Table:
    cols = _index_table(s)
    cols["mz"] = _flat_ragged(s.mz, s.offsets, pa.float64())
    cols["intensity"] = _flat_ragged(s.intensity, s.offsets, pa.float64())
    return pa.table(cols)


def _spectra_from_table(t: pa.Table) -> SpectrumSet:
    """The :class:`SpectrumSet` of a table with the layout of :func:`_spectra_table`."""
    mz, offsets = _ragged_read(t, "mz")
    intensity, _ = _ragged_read(t, "intensity")
    return SpectrumSet(
        mz=np.asarray(mz, dtype=np.float64), intensity=np.asarray(intensity, dtype=np.float64),
        offsets=offsets, **_index_arrays(t),
    )
