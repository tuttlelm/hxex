# SPDX-License-Identifier: GPL-3.0-or-later
"""The window's appearance: the widget style (View > Appearance) and a light or dark palette
chosen at start (``hxex gui --dark`` / ``--light``).

Qt's widget styles differ by platform: Fusion and the old flat "Windows" style are compiled
into Qt and exist everywhere; the native-looking ones are plugins that exist only on their
own system (``windows11`` and ``windowsvista`` on Windows, ``macOS`` on a Mac). The menu
lists what the running Qt offers (:func:`styles`), with "System default" for the style Qt
chose at start; the choice is remembered in the window's settings (``style``, value
``system`` for the default) and applied at the next start before the window is made
(:func:`apply_saved`).

The colour scheme follows the desktop: Qt takes the desktop's palette, dark where the
desktop is dark and its platform theme reports it. A user who wants the other one starts
the window with ``--dark`` or ``--light``: hxex then sets its own palette (:func:`apply_scheme`)
once, before the window is made. Switching the scheme while the window runs through Qt's
``QStyleHints.setColorScheme`` was tried (2026-11-16) and dropped: on this GNOME desktop Qt's
portal theme ignores it, and where it works, going back does not restore the palette (widgets
polished under the dark palette and widgets with a style sheet keep parts of it). The dark
palette is fully drawn by Fusion; a native style honours it only partly. The figures are
matplotlib pages drawn on white in either scheme.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory

log = logging.getLogger(__name__)

SYSTEM = "system"  # the style Qt chose at start / the desktop's palette
SCHEMES = (SYSTEM, "light", "dark")
_default_style: str | None = None  # the style at start, before any change
# the style applied last (its key): Qt reports no name once an application style sheet is set
# (the window's tooltip sheet), so the name is kept here
_current: str | None = None

# the dark palette: Fusion draws every widget from it (the sample the owner approved, 2026-11-16)
DARK = {
    QPalette.ColorRole.Window: "#353535", QPalette.ColorRole.WindowText: "#e8e8e8",
    QPalette.ColorRole.Base: "#232323", QPalette.ColorRole.AlternateBase: "#2e2e2e",
    QPalette.ColorRole.ToolTipBase: "#353535", QPalette.ColorRole.ToolTipText: "#e8e8e8",
    QPalette.ColorRole.Text: "#e8e8e8", QPalette.ColorRole.Button: "#353535",
    QPalette.ColorRole.ButtonText: "#e8e8e8", QPalette.ColorRole.BrightText: "#ff6060",
    QPalette.ColorRole.Link: "#6ab0f3", QPalette.ColorRole.Highlight: "#2a82da",
    QPalette.ColorRole.HighlightedText: "#ffffff", QPalette.ColorRole.PlaceholderText: "#9a9a9a",
}
DISABLED_TEXT = "#808080"


def styles() -> list[str]:
    """The widget styles this Qt offers, as Qt names them (Fusion is always one)."""
    return list(QStyleFactory.keys())


def default_style() -> str:
    """The style Qt chose at start (recorded the first time it is asked, so ask before any
    change: :func:`apply_saved` does)."""
    global _default_style
    if _default_style is None:
        app = QApplication.instance()
        if app is None:  # nothing to record yet
            return ""
        name = app.style().objectName()
        if not name:  # a style sheet is set already: Qt no longer says; Fusion is Qt's own
            log.warning("The widget style at start is unknown (asked after a style sheet was "
                        "set); taking Fusion as the system default")
            name = "Fusion"
        _default_style = next((s for s in styles() if s.lower() == name.lower()), name)
    return _default_style


def current_style() -> str:
    """The style in use, as Qt names it (``system`` when it is the one chosen at start)."""
    name = _current or default_style()
    if name.lower() == default_style().lower():
        return SYSTEM
    return name


def apply_style(name: str) -> bool:
    """Use the widget style ``name`` (``system`` = the one Qt chose at start). An unknown name
    is logged and leaves the style as it is; the result says whether it was applied."""
    default_style()  # recorded before the change
    chosen = default_style() if name == SYSTEM else name
    known = next((s for s in styles() if s.lower() == chosen.lower()), None)
    if known is None:
        log.warning("Widget style %r is not available here (choices: %s); keeping %s",
                    name, ", ".join(styles()), default_style())
        return False
    global _current
    if (_current or default_style()).lower() != known.lower():
        QApplication.instance().setStyle(QStyleFactory.create(known))
    _current = known
    return True


def dark_palette() -> QPalette:
    """hxex's dark palette."""
    palette = QPalette()
    for role, colour in DARK.items():
        palette.setColor(role, QColor(colour))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(DISABLED_TEXT))
    return palette


def light_palette() -> QPalette:
    """The light palette: Qt's standard Fusion palette."""
    return QStyleFactory.create("Fusion").standardPalette()


def apply_scheme(name: str) -> None:
    """Set the application's palette for ``name``: ``dark`` (:func:`dark_palette`), ``light``
    (:func:`light_palette`) or ``system`` (the desktop's, as Qt gives it). Meant for the start,
    before the window is made (see the module's docstring)."""
    app = QApplication.instance()
    if name == "dark":
        app.setPalette(dark_palette())
    elif name == "light":
        app.setPalette(light_palette())
    elif name == SYSTEM:
        app.setPalette(app.style().standardPalette())
    else:
        raise ValueError(f"colour scheme {name!r} is not one of {', '.join(SCHEMES)}")


def saved_style(settings: QSettings) -> str:
    return str(settings.value("style", SYSTEM))


def apply_saved(settings: QSettings) -> None:
    """Apply the remembered widget style (before the window is made)."""
    default_style()
    apply_style(saved_style(settings))
