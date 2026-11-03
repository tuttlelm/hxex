# SPDX-License-Identifier: GPL-3.0-or-later
"""hxex's window: a desktop application beside the command line and Python, which it does not
replace (docs/PLAN.md, "GUI"). It needs the ``gui`` extra (``pip install "hxex[gui]"``: Qt for
Python, PySide6-Essentials).

Started with ``hxex gui [PATH]`` or ``hxex-gui [PATH]``, where PATH is a GUI session file
(``.hxexgui``), a project file, or a results bundle. The modules without Qt
(:mod:`hxex.gui.session`, :mod:`hxex.gui.figures`) import without the extra.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

log = logging.getLogger(__name__)

MISSING = ('the window needs the gui extra (Qt for Python): install it with '
           'pip install "hxex[gui]" (or uv tool install "hxex[gui]")')


def available() -> bool:
    """Whether Qt for Python is installed."""
    return importlib.util.find_spec("PySide6") is not None


def launch(path: str | Path | None = None, argv: Sequence[str] | None = None) -> int:
    """Open the window (on ``path`` when given) and return when it is closed."""
    if not available():
        raise RuntimeError(MISSING)
    from PySide6.QtWidgets import QApplication

    from .icon import follow_scheme
    from .window import MainWindow

    app = QApplication.instance() or QApplication(list(argv or sys.argv[:1]))
    app.setApplicationName("hxex")
    app.setOrganizationName("hxex")
    follow_scheme(app.setWindowIcon)  # the logo for a light or dark desktop
    window = MainWindow()
    window.show()
    if path is not None:
        path = Path(path)
        if not path.exists():
            log.error("%s does not exist", path)
        else:
            window.open_path(path)
    return app.exec()


def main(argv: Sequence[str] | None = None) -> int:
    """``hxex-gui [PATH]``: the window without the command line's other commands."""
    import argparse

    parser = argparse.ArgumentParser(prog="hxex-gui", description="hxex's window.")
    parser.add_argument("path", nargs="?", metavar="PATH",
                        help="a GUI session (.hxexgui), a project file, or a results bundle")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return launch(args.path)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
