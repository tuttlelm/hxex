# SPDX-License-Identifier: GPL-3.0-or-later
"""``python -m hxex``: the command line (as the ``hxex`` command), for a program that runs hxex
with a known Python (hxex's window starts its commands so)."""

import sys

from .cli import main

sys.exit(main())
