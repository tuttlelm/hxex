# SPDX-License-Identifier: GPL-3.0-or-later
"""Result tables and plots, drawn from loaded results only.

Plots are produced from :class:`Results` (a run's objects or a results bundle), so a fresh fit and
a loaded bundle plot identically. Plot functions return matplotlib figures and never show or
save them. The curves are computed in NumPy, but the result types come from ``hxex.fit``, so
importing this package also imports PyTorch.
"""

from .analysis import write_analysis
from .compare import (
    plot_butterfly,
    plot_peptide_differences,
    plot_region_bars,
    plot_residue_comparison,
    plot_residue_overlay,
    residue_comparison_figure,
    residue_overlay_figure,
    woods_figure,
)
from .coverage import (
    FD_CMAP,
    autowrap,
    coverage_figure,
    plot_coverage,
    plot_redundancy_resolution,
)
from .curves import component_curves
from .figures import (
    PlotConfig,
    figure_stems,
    select_series,
    series_figures,
    series_kinds,
    write_figures,
)
from .fragments import closure_figure, ladder_kinetics_figure, residue_rates_figure
from .overview import (
    controls_overview_data,
    k_comparison_data,
    k_selection_data,
    overview_figures,
    plot_controls_overview,
    plot_k_comparison,
    plot_k_selection,
    plot_quality_overview,
    quality_overview_data,
    write_overview,
)
from .protection import (
    protection_difference_figure,
    protection_figure,
    residue_protection_comparison_figure,
    residue_protection_figure,
)
from .residues import PROFILE_STYLES, plot_population_choice, plot_residue_profile
from .results import Results, Settings
from .series import (
    Reference,
    load_reference,
    plot_populations,
    plot_series_spectra,
    plot_uptake,
    population_data,
    series_info,
    uptake_data,
)
from .spectrum import plot_spectrum, spectrum_data
from .tables import (
    components_table,
    find_series,
    format_table,
    series_table,
    spectra_table,
    uptake_table,
    write_csv,
    write_tables,
)

__all__ = [
    "closure_figure",
    "protection_difference_figure",
    "protection_figure",
    "residue_protection_comparison_figure",
    "residue_protection_figure",
    "ladder_kinetics_figure",
    "residue_rates_figure",
    "write_analysis",
    "plot_butterfly",
    "plot_peptide_differences",
    "plot_region_bars",
    "plot_residue_comparison",
    "plot_residue_overlay",
    "residue_comparison_figure",
    "residue_overlay_figure",
    "woods_figure",
    "PROFILE_STYLES",
    "plot_population_choice",
    "plot_residue_profile",
    "FD_CMAP",
    "autowrap",
    "coverage_figure",
    "plot_coverage",
    "plot_redundancy_resolution",
    "PlotConfig",
    "Reference",
    "Results",
    "Settings",
    "component_curves",
    "components_table",
    "controls_overview_data",
    "find_series",
    "format_table",
    "k_comparison_data",
    "k_selection_data",
    "load_reference",
    "overview_figures",
    "plot_controls_overview",
    "plot_k_comparison",
    "plot_k_selection",
    "plot_populations",
    "plot_quality_overview",
    "plot_series_spectra",
    "plot_spectrum",
    "plot_uptake",
    "population_data",
    "quality_overview_data",
    "figure_stems",
    "select_series",
    "series_kinds",
    "series_figures",
    "series_info",
    "series_table",
    "spectra_table",
    "spectrum_data",
    "uptake_data",
    "uptake_table",
    "write_csv",
    "write_figures",
    "write_overview",
    "write_tables",
]
