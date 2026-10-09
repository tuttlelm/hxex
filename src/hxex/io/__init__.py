# SPDX-License-Identifier: GPL-3.0-or-later
"""Readers (HDExaminer SpecExport, HX-Express; later mzML, Thermo raw) and the results bundle.

Planned; not implemented yet (Phases 1 and 2).

Readers share one plug-in interface (path and metadata in, spectra out). The results-bundle
writer and loader (Parquet, one row per spectrum) live here.
"""
