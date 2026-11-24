# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading scan files in parallel: each file (with every run that reads it) is one task, done in
a pool of spawned processes when the files are large enough to be worth it, else in this one.

The results are the same either way: a task does exactly what the read does for that file in
this process, and its results are taken in the files' order. The log lines a task writes are
kept and written here in that order too, after the task's results, so a parallel read logs what
a read file by file logs, in the same order."""

from __future__ import annotations

import logging
import multiprocessing
import os
import pickle
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from contextlib import contextmanager
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from types import MappingProxyType

from ..hooks import CancelHook, check_cancel

MAX_READ_WORKERS = 8  # processes reading files at once, at most
PARALLEL_BYTES = 100_000_000  # files are read in parallel when they hold this much together
_WAIT = 0.5  # s between looks at the cancel hook while a task runs
# a process reading a file holds up to about this many times the file's size, plus its own
# start (measured: ion mobility mzML 5.5 times, Thermo raw 3 times)
MEMORY_PER_BYTE = 6
MEMORY_START = 300_000_000
MEMORY_SHARE = 0.6  # of the memory available, at most this much for the processes


def _view(items: dict) -> MappingProxyType:
    return MappingProxyType(items)


def _proxy(view: MappingProxyType):
    return _view, (dict(view),)


# what goes to and from a pool's processes holds read-only mappings (a peptide entry's extra
# columns, a modification's composition), which pickle does not copy: the pickler the pool uses
# (multiprocessing's own, not pickle's) sends each as a new view of a copy
ForkingPickler.register(MappingProxyType, _proxy)


def usable_cpus() -> int:
    """The CPUs this process may use."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS, Windows
        return os.cpu_count() or 1


def available_memory() -> int | None:
    """The memory (bytes) the system has available now; None when it cannot be told (half the
    physical memory where only that is known)."""
    try:
        with open("/proc/meminfo") as f:  # Linux
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    if sys.platform == "win32":
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                        ("page_total", ctypes.c_ulonglong), ("page_free", ctypes.c_ulonglong),
                        ("virtual_total", ctypes.c_ulonglong),
                        ("virtual_free", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]

        status = _Status()
        status.length = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available)
        return None
    try:  # macOS: the physical memory only
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2
    except (AttributeError, ValueError, OSError):
        return None


def read_workers(paths: Sequence[str | Path], workers: int | None, can: bool = True) -> int:
    """How many processes read the files ``paths`` (one task each): 1 (this process) when
    ``can`` is false (a format that cannot be opened in another process), for a single file,
    or, with ``workers`` None, when the files hold less than :data:`PARALLEL_BYTES` together;
    else ``workers`` (None: up to :data:`MAX_READ_WORKERS`, the CPUs this process may use, and
    as many as :data:`MEMORY_SHARE` of the available memory holds, each process taking
    :data:`MEMORY_PER_BYTE` times the largest file plus :data:`MEMORY_START`), never more than
    there are files."""
    if not can or len(paths) < 2:
        return 1
    if workers is None:
        sizes = []
        for p in paths:
            try:
                sizes.append(Path(p).stat().st_size)
            except OSError:
                pass
        if sum(sizes) < PARALLEL_BYTES:
            return 1
        workers = min(MAX_READ_WORKERS, usable_cpus())
        memory = available_memory()
        if memory is not None:
            each = MEMORY_PER_BYTE * max(sizes) + MEMORY_START
            workers = min(workers, int(MEMORY_SHARE * memory // each))
    return max(1, min(int(workers), len(paths)))


class _Kept(logging.Handler):
    """The log lines of a task, kept to be written by the process that asked for it."""

    def __init__(self):
        super().__init__()
        self.lines: list[tuple[str, int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append((record.name, record.levelno, record.getMessage()))


def _call(fn: Callable, level: int, task):
    """``fn(task)`` in a pool's process with hxex's log lines kept (at ``level``, the asking
    process's); returns the result and the lines. An error that cannot be sent back as it is
    is sent as a RuntimeError with its message."""
    logger = logging.getLogger("hxex")
    kept = _Kept()
    logger.handlers[:] = [kept]
    logger.setLevel(level)
    logger.propagate = False
    try:
        return fn(task), kept.lines
    except BaseException as exc:
        try:
            pickle.loads(pickle.dumps(exc))
        except Exception:  # noqa: BLE001  (an error whose arguments do not survive the trip)
            raise RuntimeError(f"{type(exc).__name__}: {exc}") from None
        raise


@contextmanager
def _main_left_out():
    """While a pool starts its processes, they are not told about the main script: a spawned
    process runs the script it is told about again before its first task, and a script that
    reads its project at the top (no ``if __name__ == "__main__":``) would read again in
    every process, which starts its own pool. The tasks are functions of hxex, which a
    process imports itself."""
    main = sys.modules.get("__main__")
    if main is None:
        yield
        return
    saved = {k: main.__dict__[k] for k in ("__file__", "__spec__") if k in main.__dict__}
    main.__dict__.pop("__file__", None)
    main.__spec__ = None
    try:
        yield
    finally:
        main.__dict__.pop("__spec__", None)
        main.__dict__.update(saved)


def _replay(lines: Sequence[tuple[str, int, str]]) -> None:
    for name, level, message in lines:
        logging.getLogger(name).log(level, "%s", message)


def map_tasks(fn: Callable, tasks: Sequence, workers: int,
              on_result: Callable[[int, object], None], cancel: CancelHook | None = None
              ) -> None:
    """``fn`` of every task in a pool of ``workers`` spawned processes, each with one thread of
    linear algebra; ``on_result(i, result)`` in the order of ``tasks``, after the log lines of
    task ``i`` are written. ``cancel`` is looked at while waiting; cancelling (or any error,
    such as a process that died) stops the processes at once."""
    from ..picking.profile import _single_threaded_children

    level = logging.getLogger("hxex").getEffectiveLevel()
    pool = ProcessPoolExecutor(max_workers=workers,
                               mp_context=multiprocessing.get_context("spawn"))
    try:
        with _single_threaded_children(), _main_left_out():  # the processes start here
            pending = [pool.submit(_call, fn, level, t) for t in tasks]
        for i, job in enumerate(pending):
            while True:
                check_cancel(cancel)
                try:
                    result, lines = job.result(timeout=_WAIT)
                    break
                except FuturesTimeout:
                    continue
            _replay(lines)
            on_result(i, result)
    except BaseException:
        processes = list((getattr(pool, "_processes", None) or {}).values())
        pool.shutdown(wait=False, cancel_futures=True)
        for process in processes:  # a task already running is stopped, not waited for
            process.terminate()
        raise
    pool.shutdown(wait=True)
