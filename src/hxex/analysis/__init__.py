# SPDX-License-Identifier: GPL-3.0-or-later
"""Post-fitting analysis: from fitted results to the protein (design: docs/ANALYSIS.md).

:func:`species_uptake` gives the fractional deuteration (``fD``) and population of every
species (peptide or fragment), time and rank at a fixed K, replicates and charge states
combined; :func:`residue_map` turns the peptides into residue-level values, and
:func:`coverage_table` gives each sample's redundancy and resolution per residue;
:func:`choose_populations` decides per residue whether two populations are supported;
:func:`compare_peptides`, :func:`compare_residues` and :func:`region_summary` compare samples.
"""

from .compare import (
    Annotations,
    PeptideComparison,
    Region,
    RegionTable,
    ResidueComparison,
    Site,
    compare_peptides,
    compare_populations,
    compare_residues,
    region_summary,
)
from .populations import (
    ChoiceConfig,
    PopulationChoice,
    choose_populations,
    lever_population,
)
from .residues import (
    CoverageTable,
    ResidueMap,
    SequenceMismatch,
    coverage_table,
    exchanging_residues,
    residue_map,
)
from .species import SpeciesFilters, SpeciesUptake, species_uptake

__all__ = [
    "Annotations",
    "PeptideComparison",
    "Region",
    "RegionTable",
    "ResidueComparison",
    "Site",
    "compare_peptides",
    "compare_populations",
    "compare_residues",
    "region_summary",
    "ChoiceConfig",
    "CoverageTable",
    "PopulationChoice",
    "ResidueMap",
    "SequenceMismatch",
    "SpeciesFilters",
    "SpeciesUptake",
    "choose_populations",
    "coverage_table",
    "exchanging_residues",
    "lever_population",
    "residue_map",
    "species_uptake",
]
