# SPDX-License-Identifier: GPL-3.0-or-later
"""One file watched for changes made outside the window (audit of 2026-11-15, Q-17).

Two things every watcher of the window must do: a file replaced (as *hxex* writes its files, and
as many editors save) is no longer watched by Qt, so it is watched again when its change is
reported; and the window's own write is not watched while it happens (on Windows a watched file
could not be replaced), so :meth:`WatchedFile.paused` stops watching for the length of a
``with`` block and watches the file again after it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QObject


class WatchedFile(QObject):
    """The file :meth:`watch` names, watched; ``changed(path)`` is called when another program
    changes it (after it is watched again)."""

    def __init__(self, parent: QObject | None, changed: Callable[[str], None]):
        super().__init__(parent)
        self.path: Path | None = None
        self._changed = changed
        self.watcher = QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._reported)

    def files(self) -> list[str]:
        """The files watched now (none or this one)."""
        return self.watcher.files()

    def watch(self, path: Path | None) -> bool:
        """Watch ``path`` (None: nothing) instead of the file watched until now; a file that
        does not exist yet is watched once it is written (:meth:`paused`). False when Qt could
        not watch an existing file (the system's limit on watchers reached)."""
        self.unwatch()
        self.path = Path(path) if path is not None else None
        return self._watch_again()

    def unwatch(self) -> None:
        if self.watcher.files():
            self.watcher.removePaths(self.watcher.files())

    @contextlib.contextmanager
    def paused(self):
        """Not watched inside the block (the window writing the file itself), watched again
        after it, whether the block ends normally or not."""
        self.unwatch()
        try:
            yield self
        finally:
            self._watch_again()

    def _watch_again(self) -> bool:
        if self.path is not None and self.path.exists() \
                and str(self.path) not in self.watcher.files():
            return self.watcher.addPath(str(self.path))
        return True

    def _reported(self, path: str) -> None:
        # a file replaced is no longer watched: watch it again
        if Path(path).exists() and path not in self.watcher.files():
            self.watcher.addPath(path)
        self._changed(path)


__all__ = ["WatchedFile"]
