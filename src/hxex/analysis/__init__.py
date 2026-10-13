# SPDX-License-Identifier: GPL-3.0-or-later
"""Post-fitting analysis: from fitted results to the protein (design: docs/ANALYSIS.md).

:func:`species_uptake` gives the fractional deuteration (``fD``) and population of every
species (peptide or fragment), time and rank at a fixed K, replicates and charge states
combined; :func:`residue_map` turns the peptides into residue-level values, and
:func:`coverage_table` gives each sample's redundancy and resolution per residue.
"""

from .residues import (
    CoverageTable,
    ResidueMap,
    SequenceMismatch,
    coverage_table,
    exchanging_residues,
    residue_map,
)
from .species import SpeciesUptake, species_uptake

__all__ = [
    "CoverageTable",
    "ResidueMap",
    "SequenceMismatch",
    "SpeciesUptake",
    "coverage_table",
    "exchanging_residues",
    "residue_map",
    "species_uptake",
]
