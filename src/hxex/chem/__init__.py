# SPDX-License-Identifier: GPL-3.0-or-later
"""Chemistry: m/z, composition, exchangeable amide count and natural-abundance shape.

Sequences use pyteomics modX labels (acK, pS, pT, pY built in, plus a user-extensible
:class:`ModificationRegistry` passed to pyteomics, never edited globally). Residues are
tokenised, not counted as characters. Every function takes an ion type (default ``'M'``).
"""

from .amides import (
    DEFAULT_AMIDE_RULES,
    AmideRules,
    count_amides,
    exchangeable_sites,
)
from .composition import composition, mz, natural_abundance, neutral_mass
from .ions import check_ion_type, ion_types
from .modifications import (
    DEFAULT_MODIFICATIONS,
    ChemistryError,
    IonTypeError,
    ModificationRegistry,
    SequenceError,
    parse_sequence,
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
    "check_ion_type",
    "composition",
    "count_amides",
    "exchangeable_sites",
    "ion_types",
    "mz",
    "natural_abundance",
    "neutral_mass",
    "parse_modification",
    "parse_sequence",
]
