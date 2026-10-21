# SPDX-License-Identifier: GPL-3.0-or-later
"""Readers for the input formats (HX-Express spreadsheets, HDExaminer SpecExport, pooled folders).

Readers return raw spectra (:class:`hxex.spectra.SpectrumSet`); peak picking is separate. mzML
files are read by :mod:`hxex.io.mzml` (scans, summing, peptides and fragments), and Thermo raw
files the same way by :mod:`hxex.io.thermo` (with Thermo's RawFileReader, not part of hxex).
"""

from .entries import (
    apply_retention_times,
    filter_entries,
    read_fasta_records,
    read_fasta_sequence,
    read_peptide_table,
    read_retention_times,
    sort_entries,
)
from .errors import ReadError
from .hxexpress import (
    parse_hxexpress_filename,
    read_hxexpress_file,
    scan_hxexpress,
)
from .identity import directory_identity
from .mzml import (
    MzmlConfig,
    MzmlError,
    MzmlFile,
    MzmlRun,
    MzmlTarget,
    ScanTable,
    envelope_groups,
    file_identity,
    iter_scans,
    kinetics_spectra,
    mzml_series_targets,
    read_mzml,
    read_mzml_source,
    sum_centroid,
    sum_profile,
)
from .outputs import OutputError, made_from, output_listing, replace_outputs
from .pooled import CandidateMatch, match_candidates, scan_specexport_pooled
from .read import read_source, read_spectra, read_spectra_source, scan, spectrum_index
from .specexport import (
    parse_specexport_filename,
    read_specexport_dir,
    scan_specexport,
)
from .thermo import ThermoError, ThermoFile, read_thermo, read_thermo_source
from .times import (
    TIME_SCALES,
    format_time,
    parse_hxexpress_label,
    parse_specexport_token,
    parse_time,
)

__all__ = [
    "MzmlConfig",
    "MzmlError",
    "MzmlFile",
    "MzmlRun",
    "MzmlTarget",
    "ScanTable",
    "ThermoError",
    "ThermoFile",
    "directory_identity",
    "envelope_groups",
    "file_identity",
    "iter_scans",
    "kinetics_spectra",
    "mzml_series_targets",
    "read_mzml",
    "read_mzml_source",
    "read_thermo",
    "read_thermo_source",
    "sum_centroid",
    "sum_profile",
    "load_spectra",
    "save_spectra",
    "spectra_info",
    "Bundle",
    "BundleError",
    "ReviewTable",
    "SCHEMA_VERSION",
    "open_bundle",
    "write_bundle",
    "bundle_manifest",
    "generations",
    "keep_previous",
    "provenance_of",
    "OutputError",
    "made_from",
    "output_listing",
    "replace_outputs",
    "CandidateMatch",
    "ReadError",
    "filter_entries",
    "parse_hxexpress_filename",
    "parse_hxexpress_label",
    "parse_specexport_filename",
    "parse_specexport_token",
    "parse_time",
    "format_time",
    "TIME_SCALES",
    "match_candidates",
    "read_fasta_records",
    "read_fasta_sequence",
    "read_hxexpress_file",
    "read_peptide_table",
    "read_retention_times",
    "apply_retention_times",
    "read_specexport_dir",
    "read_source",
    "read_spectra",
    "spectrum_index",
    "read_spectra_source",
    "scan",
    "scan_hxexpress",
    "scan_specexport",
    "scan_specexport_pooled",
    "sort_entries",
]


_BUNDLE = {
    "Bundle", "BundleError", "ReviewTable", "SCHEMA_VERSION", "open_bundle", "write_bundle",
    "bundle_manifest", "generations", "keep_previous", "provenance_of",
}
_SPECTRA_FILE = {"save_spectra", "load_spectra", "spectra_info"}


def __getattr__(name):
    """The bundle is loaded on first use: it needs the fitting code (and so torch), which the
    readers do not. The spectra file needs pyarrow only, which is also loaded on first use."""
    if name in _BUNDLE:
        from . import bundle

        return getattr(bundle, name)
    if name in _SPECTRA_FILE:
        from . import spectra_file

        return getattr(spectra_file, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
