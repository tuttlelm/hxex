# SPDX-License-Identifier: GPL-3.0-or-later
"""A fingerprint of the code that decides a fit (owner, 2026-11-11).

``hxex run`` keeps the earlier results' fits of the spectra whose input is unchanged
(:mod:`hxex.reuse`). It once required the very same hxex commit and package builds, so a commit
that changed only the window or the documentation, or the same project opened on a laptop
without a graphics card (a CPU build of PyTorch), fitted every spectrum again. What decides a
fit, given its input, is the engine: the fit, the model, the noise, the bootstrap, and the
population test. Their source files' contents give the fingerprint, so an uncommitted change to
them counts too, and a change anywhere else does not.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

# under src/hxex: the modules (or packages) that decide a fit, given its input
ENGINE = ("fit", "model", "noise.py", "bootstrap.py", "population_test.py")


@lru_cache(maxsize=1)
def engine_fingerprint() -> str:
    """The first 16 hex digits of the SHA-256 of the engine's source files (their paths
    within the package and their bytes, in path order)."""
    root = Path(__file__).resolve().parent
    files: list[Path] = []
    for name in ENGINE:
        path = root / name
        files += sorted(path.rglob("*.py")) if path.is_dir() else [path]
    h = hashlib.sha256()
    for path in files:
        h.update(path.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]
