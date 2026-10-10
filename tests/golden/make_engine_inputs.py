# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate engine_inputs.npz: the engine's regression inputs from the Bimodal test set.

Usage: python tests/golden/make_engine_inputs.py BIMODAL_DIR OUT.npz

Builds the 542 measurements (23 series, Bradykinin_Lo excluded) the way the prototype notebook
(run_test.ipynb) does, from the binned sticks in
``output_fixed1pop_19Feb2024/alldeutdata_19Feb2024.csv``: bins are ``n_deut``, the shape is the
natural-abundance envelope from ``hxex.chem``, and the noise reference of a measurement is the
undeuterated control (time 0) of its series. The engine takes binned data, so it does not matter
that this table was made by an older picker.
"""

import csv
import sys
from pathlib import Path

import numpy as np

from hxex import chem

UNDEUT_TIME = 0.0
EXCLUDE = {"Bradykinin_Lo"}


def ragged(arrays):
    flat = np.concatenate(arrays)
    return flat, np.concatenate([[0], np.cumsum([len(a) for a in arrays])])


def main(bimodal_dir, out):
    path = Path(bimodal_dir) / "output_fixed1pop_19Feb2024" / "alldeutdata_19Feb2024.csv"
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    samples = [s for s in dict.fromkeys(r["sample"] for r in rows) if s not in EXCLUDE]
    bins, obs, shapes, group, times, namides, names = [], [], [], [], [], [], []
    for s in samples:
        sample_rows = [r for r in rows if r["sample"] == s]
        charges = sorted({int(float(r["charge"])) for r in sample_rows})
        for z in charges:
            g = s if len(charges) == 1 else f"{s}_z{z}"
            zrows = [r for r in sample_rows if int(float(r["charge"])) == z]
            peptide = zrows[0]["peptide"]
            for t in sorted({float(r["time"]) for r in zrows}):
                trows = [r for r in zrows if float(r["time"]) == t]
                reps = sorted({int(float(r["rep"])) for r in trows})
                for rep in reps:
                    spec = [r for r in trows if int(float(r["rep"])) == rep]
                    nd = [int(float(r["n_deut"])) for r in spec]
                    assert len(set(nd)) == len(nd)
                    bins.append(np.array(nd, float))
                    obs.append(np.array([float(r["Intensity"]) for r in spec]))
                    shapes.append(chem.natural_abundance(peptide))
                    group.append(g)
                    times.append(t)
                    namides.append(chem.count_amides(peptide))
                    suffix = f"_r{rep}" if len(reps) > 1 else ""
                    names.append(f"{g}_{spec[0]['time_idx']}{suffix}")
    m = len(obs)
    und = [
        np.array(
            [i for i in range(m) if group[i] == group[j] and abs(times[i] - UNDEUT_TIME) <= 1e-6],
            dtype=int,
        )
        for j in range(m)
    ]
    out_arrays = {}
    out_arrays["bins"], out_arrays["bins_off"] = ragged(bins)
    out_arrays["obs"], _ = ragged(obs)
    out_arrays["shape"], out_arrays["shape_off"] = ragged(shapes)
    out_arrays["und"], out_arrays["und_off"] = ragged(und)
    out_arrays["has_und"] = np.array([len(u) > 0 for u in und])
    out_arrays["group"] = np.array(group)
    out_arrays["name"] = np.array(names)
    out_arrays["time"] = np.array(times)
    out_arrays["namides"] = np.array(namides)
    np.savez_compressed(out, **out_arrays)
    print(f"wrote {out}: {m} measurements, {len(set(group))} series")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
