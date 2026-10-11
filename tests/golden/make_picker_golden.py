# SPDX-License-Identifier: GPL-3.0-or-later
"""Make picker_reference.npz from a pyHXExpress run that has both raw and picked data.

Usage: python tests/golden/make_picker_golden.py RUN_DIR TAG OUT.npz
e.g. .../batchrun_1pop_06Oct2026/TBA_run 06Oct2026 /path/outside/the/repo/picker_reference.npz

The output holds raw spectra of the owner's unpublished data and must NOT be committed: keep it
outside the repository and point HXEX_PICKER_GOLDEN at it (tests/test_picking.py).

RUN_DIR holds ``allrawdata_TAG.zip`` and ``alldeutdata_TAG.csv`` written by pyHXExpress's
``run_hdx_fits`` (SpecExport data). The picked table is the reference output of its
``peak_picker`` for the settings in ``hdxms_params_TAG.py`` (Zero_Filling 3, setNoise 50,
Nterm_subtract 2, Peak_Resolution 50, Env_threshold 0.1). A few spectra are chosen to cover the
cases that matter, using the picker under test only to classify them.
"""

import csv
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from hxex.picking import PickerConfig, pick_spectrum

CONFIG = PickerConfig(zero_filling=3, noise=50.0, nterm_subtract=2)


def load(run_dir, tag):
    run_dir = Path(run_dir)
    with zipfile.ZipFile(run_dir / f"allrawdata_{tag}.zip") as z:
        text = z.read(z.namelist()[0]).decode()
    raw, meta = defaultdict(list), {}
    for r in csv.DictReader(io.StringIO(text)):
        k = (r["file"], r["time"], r["rep"], r["charge"])
        raw[k].append((float(r["mz"]), float(r["Intensity"])))
        meta[k] = r
    ref = defaultdict(list)
    with (run_dir / f"alldeutdata_{tag}.csv").open() as fh:
        for r in csv.DictReader(fh):
            ref[(r["file"], r["time"], r["rep"], r["charge"])].append(r)
    return raw, meta, ref


def classify(k, raw, meta, ref):
    a = np.array(sorted(raw[k]))
    m = meta[k]
    p = pick_spectrum(a[:, 0], a[:, 1], m["peptide"], int(float(m["charge"])), config=CONFIG)
    if k not in ref:
        return "dropped_by_reference" if p.empty else "absent_from_reference"
    ry = np.array([float(r["Intensity"]) for r in ref[k]])
    n = min(len(ry), len(p.intensity))
    if len(ry) == len(p.intensity):
        return "same_length"
    if len(p.intensity) > len(ry):
        extra = p.intensity[len(ry):]
        frac = extra.max() / max(p.intensity.max(), 1e-300)
        if frac > 0.5:
            return "reference_missed_major_signal"
        if (extra >= p.threshold).any():
            return "reference_missed_signal"
        return "reference_stopped_early"
    return "reference_longer" if n else "other"


def main(run_dir, tag, per_class=2):
    raw, meta, ref = load(run_dir, tag)
    chosen = defaultdict(list)
    for k in raw:
        c = classify(k, raw, meta, ref)
        if len(chosen[c]) < per_class:
            chosen[c].append(k)
    keys = [k for ks in chosen.values() for k in ks]
    out = {"keys": np.array(["|".join(k) for k in keys]), "classes": np.array(
        [c for c, ks in chosen.items() for _ in ks])}
    out["peptide"] = np.array([meta[k]["peptide"] for k in keys])
    out["charge"] = np.array([int(float(meta[k]["charge"])) for k in keys])
    sizes = [len(raw[k]) for k in keys]
    out["raw_offsets"] = np.concatenate([[0], np.cumsum(sizes)])
    pts = np.vstack([np.array(sorted(raw[k])) for k in keys])
    out["raw_mz"], out["raw_intensity"] = pts[:, 0], pts[:, 1]
    in_ref = [k in ref for k in keys]
    out["in_reference"] = np.array(in_ref)
    rsizes = [len(ref[k]) if k in ref else 0 for k in keys]
    out["ref_offsets"] = np.concatenate([[0], np.cumsum(rsizes)])
    rows = [r for k in keys if k in ref for r in ref[k]]
    for col in ("mz", "Intensity", "n_deut"):
        out["ref_" + col] = np.array([float(r[col]) for r in rows])
    for col in ("env_width", "env_symm", "max_namides"):
        out["ref_" + col] = np.array([float(ref[k][0][col]) if k in ref else np.nan for k in keys])
    dest = Path(sys.argv[3])
    np.savez_compressed(dest, **out)
    print("wrote", dest, dest.stat().st_size, "bytes;", {c: len(v) for c, v in chosen.items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
