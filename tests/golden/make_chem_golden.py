# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate chem_reference.json from pyHXExpress's hxex_updating.py.

Usage: python tests/golden/make_chem_golden.py /path/to/hxex_updating.py

The reference functions are extracted with ``ast`` and run in a throwaway namespace, so the
reference module (which edits pyteomics' global tables) is never imported. Run this in its own
process; it is not used by the test suite, which reads the committed JSON.
"""

import ast
import json
import sys
import types
from pathlib import Path

import numpy as np
from brainpy import isotopic_variants
from pyteomics import mass

PEPTIDES = [
    "PEPTIDE", "PEPTIDEK", "YGGFL", "RDKVQKEYALFYKLD", "PPPGPAKL", "GGGGGGGG",
    "PEPacKTIDE", "acKPEPTIDE", "SpSTpYPK", "pTPEpYPEPK", "PEPTIDEacK", "MKWVTFISLLLLFSSAYS",
]
CHARGES = [1, 2, 3]
NTERM = [1, 2]


def load_reference(path):
    tree = ast.parse(Path(path).read_text())
    want = {"goodseq", "get_na_isotope", "count_amides"}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    ptm = [n for n in tree.body if isinstance(n, ast.Assign) and "std_aa_comp" in ast.unparse(n)]
    ns = {
        "np": np,
        "mass": mass,
        "isotopic_variants": isotopic_variants,
        "config": types.SimpleNamespace(Nterm_subtract=1),
    }
    exec(compile(ast.Module(ptm + funcs, []), str(path), "exec"), ns)
    return ns


def main(path):
    ref = load_reference(path)
    out = {"source": "hxex_updating.py", "amides": [], "mz": [], "shape": []}
    for pep in PEPTIDES:
        for n in NTERM:
            ref["config"].Nterm_subtract = n
            out["amides"].append({"seq": pep, "nterm": n, "n": ref["count_amides"](pep)})
        ref["config"].Nterm_subtract = 1
        for z in CHARGES:
            out["mz"].append({
                "seq": pep, "z": z,
                "mz": mass.calculate_mass(sequence=pep, show_unmodified_termini=True, charge=z),
            })
        shape = ref["get_na_isotope"](pep, 2)
        out["shape"].append({"seq": pep, "nterm": 1, "shape": [float(x) for x in shape]})
    dest = Path(__file__).with_name("chem_reference.json")
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", dest)


if __name__ == "__main__":
    main(sys.argv[1])
