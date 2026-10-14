# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate kint_reference.json from pacilab/hdx-rates-mixtures (the code hxex.kint ports).

Usage: python tests/golden/make_kint_golden.py /path/to/hdx-rates-mixtures OUT.json [--rounded]

The reference rounds intermediate values (rates to 1e-6 /s, factors to 0.01), which hxex.kint
does not; by default its ``.round(n)`` calls are stripped from a copy of its code, so the rates
should agree to floating-point precision (with ``--rounded``: to about 2 %). Needs numpy and
pandas (the reference's dependencies); run it in its own environment. It is not used by the test
suite, which reads the committed JSON.
"""

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

SEQUENCES = ["PEPTIDEKHHGLAAYS", "DEHKRCWFMNQVIGPST", "GGPAGLLAEDH", "ADE"]
PH = [2.5, 7.0, 8.5]
TEMPERATURES = [273.15, 298.0]  # K
FRACTIONS = [0.0, 0.1, 0.9, 1.0]
REFERENCES = ["PDLA", "3Ala"]


def main(repo: str, out: str, rounded: bool = False) -> None:
    code = Path(tempfile.mkdtemp()) / "python"
    shutil.copytree(Path(repo) / "python", code)
    if not rounded:
        for name in ("utils_kint.py", "utils_pL.py"):
            path = code / name
            path.write_text(re.sub(r"\.round\(\d\)", "", path.read_text()))
    sys.path.insert(0, str(code))
    from utils_kint import calculate_kint

    cases = []
    for seq in SEQUENCES:
        for ph in PH:
            for t in TEMPERATURES:
                for x in FRACTIONS:
                    for ref in REFERENCES:
                        k = calculate_kint(seq=seq, T=t, x=x, pH_read=ph, ref=ref)
                        cases.append({"sequence": seq, "ph_read": ph, "temperature_k": t,
                                      "d2o_fraction": x, "reference": ref,
                                      "kforw": [float(f"{v:.12g}") for v in k["kforw"]],
                                      "kback": [float(f"{v:.12g}") for v in k["kback"]]})
    Path(out).write_text(json.dumps({"source": "pacilab/hdx-rates-mixtures",
                                     "rounded": rounded, "cases": cases}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--rounded" in sys.argv[3:])
