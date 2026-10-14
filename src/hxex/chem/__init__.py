# SPDX-License-Identifier: GPL-3.0-or-later
"""Chemistry: m/z, composition, exchangeable amide count and natural-abundance shape.

Sequences use pyteomics modX labels (acK, pS, pT, pY built in, plus a user-extensible
:class:`ModificationRegistry` passed to pyteomics, never edited globally). Residues are
tokenised, not counted as characters. Every function takes an ion type (default ``'M'``).
"""

from .amides import (
    DEFAULT_AMIDE_RULES,
    AmideRules,
    amide_positions,
    count_amides,
    exchangeable_sites,
)
from .composition import (
    H_ATOM,
    composition,
    isotope_clusters,
    mixed_clusters,
    mz,
    natural_abundance,
    neutral_mass,
)
from .ions import check_ion_type, ion_terminus, ion_types
from .modifications import (
    DEFAULT_MODIFICATIONS,
    TERMINAL_GROUPS,
    ChemistryError,
    IonTypeError,
    ModificationRegistry,
    SequenceError,
    join_termini,
    parse_sequence,
    split_termini,
)
from .spec import ModificationSpec, parse_modification

__all__ = [
    "DEFAULT_AMIDE_RULES",
    "DEFAULT_MODIFICATIONS",
    "AmideRules",
    "ChemistryError",
    "IonTypeError",
    "ModificationRegistry",
    "ModificationSpec",
    "SequenceError",
    "TERMINAL_GROUPS",
    "check_ion_type",
    "composition",
    "amide_positions",
    "count_amides",
    "exchangeable_sites",
    "ion_terminus",
    "ion_types",
    "mz",
    "H_ATOM",
    "isotope_clusters",
    "mixed_clusters",
    "natural_abundance",
    "neutral_mass",
    "parse_modification",
    "join_termini",
    "parse_sequence",
    "split_termini",
]
