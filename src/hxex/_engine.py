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
# under src/hxex: the modules (or packages) that decide what a read gives, given its settings
# and files: the readers, the shared spectrum types, the chemistry, and the picking the scan
# readers use (m/z windows, calibration, retention search). ``hxex run`` takes the stored
# spectra of the bundle it replaces in place of a read only when this is unchanged
READER = ("io/read.py", "io/specexport.py", "io/hxexpress.py", "io/pooled.py", "io/entries.py",
          "io/mzml.py", "io/thermo.py", "io/drift_gate.py", "io/elution_windows.py",
          "io/retention_search.py", "io/times.py", "io/_common.py", "spectra.py", "chem",
          "picking")
# what decides the picking, given the read spectra and the settings (the bins, the peak
# measure, the envelopes): a run keeps the picking stored in a bundle only when this is
# unchanged (the version number alone stays the same between commits)
PICKING = ("picking", "chem", "spectra.py")
# what decides the back-exchange correction and the quality flags, given the fits
CORRECTION = ("controls.py", "deamidation.py", "kint.py", "quality.py")


def _fingerprint(names) -> str:
    """The first 16 hex digits of the SHA-256 of the source files ``names`` (modules or
    packages under src/hxex): their paths within the package and their bytes, in path order."""
    root = Path(__file__).resolve().parent
    files: list[Path] = []
    for name in names:
        path = root / name
        files += sorted(path.rglob("*.py")) if path.is_dir() else [path]
    h = hashlib.sha256()
    for path in files:
        h.update(path.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def engine_fingerprint() -> str:
    """The fingerprint of the engine's source files (:data:`ENGINE`)."""
    return _fingerprint(ENGINE)


@lru_cache(maxsize=1)
def reader_fingerprint() -> str:
    """The fingerprint of the readers' source files (:data:`READER`)."""
    return _fingerprint(READER)


@lru_cache(maxsize=1)
def picking_fingerprint() -> str:
    """The fingerprint of the picking's source files (:data:`PICKING`)."""
    return _fingerprint(PICKING)


@lru_cache(maxsize=1)
def correction_fingerprint() -> str:
    """The fingerprint of the correction's source files (:data:`CORRECTION`)."""
    return _fingerprint(CORRECTION)
