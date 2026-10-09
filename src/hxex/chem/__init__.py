# SPDX-License-Identifier: GPL-3.0-or-later
"""Chemistry: m/z, composition, exchangeable amide count and natural-abundance shape.

Planned; not implemented yet (Phase 1).

Sequences use pyteomics modX labels (acK, pS, pT, pY built in, plus a user-extensible
registry passed to pyteomics, never edited globally). Residues are tokenised, not counted as
characters. Every function takes an ion type (default 'M').
"""
