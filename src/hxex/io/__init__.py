# SPDX-License-Identifier: GPL-3.0-or-later
"""Readers for the input formats (HX-Express spreadsheets, HDExaminer SpecExport, pooled folders).

Readers return raw spectra (:class:`hxex.spectra.SpectrumSet`); peak picking is separate. Later:
mzML, Thermo raw, and the results bundle (Parquet, one row per spectrum).
"""

from .entries import (
    filter_entries,
    read_fasta_records,
    read_fasta_sequence,
    read_peptide_table,
    sort_entries,
)
from .errors import ReadError
from .hxexpress import (
    parse_hxexpress_filename,
    read_hxexpress_file,
    scan_hxexpress,
)
from .pooled import CandidateMatch, match_candidates, scan_specexport_pooled
from .read import read_spectra, scan
from .specexport import (
    parse_specexport_filename,
    read_specexport_dir,
    scan_specexport,
)
from .times import DEFAULT_FULLDEUT_TIME, parse_hxexpress_time, parse_specexport_time

__all__ = [
    "Bundle",
    "BundleError",
    "ReviewTable",
    "SCHEMA_VERSION",
    "open_bundle",
    "write_bundle",
    "CandidateMatch",
    "DEFAULT_FULLDEUT_TIME",
    "ReadError",
    "filter_entries",
    "parse_hxexpress_filename",
    "parse_hxexpress_time",
    "parse_specexport_filename",
    "parse_specexport_time",
    "match_candidates",
    "read_fasta_records",
    "read_fasta_sequence",
    "read_hxexpress_file",
    "read_peptide_table",
    "read_specexport_dir",
    "read_spectra",
    "scan",
    "scan_hxexpress",
    "scan_specexport",
    "scan_specexport_pooled",
    "sort_entries",
]


_BUNDLE = {
    "Bundle", "BundleError", "ReviewTable", "SCHEMA_VERSION", "open_bundle", "write_bundle",
}


def __getattr__(name):
    """The bundle is loaded on first use: it needs the fitting code (and so torch), which the
    readers do not."""
    if name in _BUNDLE:
        from . import bundle

        return getattr(bundle, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
