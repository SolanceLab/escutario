# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""One exclusive file lock for macOS, Linux and Windows.

fcntl.flock where fcntl exists; msvcrt.locking on Windows (one byte at offset 0). Importing
this module never fails on either system. The lock is held until `release` (or the file closes).
"""

from __future__ import annotations

import errno
import time

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # macOS, Linux
    msvcrt = None  # type: ignore[assignment]

_RETRY_S = 0.05
_BUSY = {errno.EACCES, errno.EDEADLK, getattr(errno, "EDEADLOCK", errno.EDEADLK)}


def _win_lock(f, blocking: bool) -> None:
    f.seek(0)
    mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
    while True:
        try:
            msvcrt.locking(f.fileno(), mode, 1)
            return
        except OSError as e:
            if e.errno not in _BUSY:
                raise
            if not blocking:
                raise BlockingIOError(e.errno, "lock is held elsewhere") from e
            time.sleep(_RETRY_S)  # 'resource deadlock' = still held; keep waiting


def acquire(f, blocking: bool = True) -> None:
    """Take an exclusive lock on open file `f`. Not blocking: raises BlockingIOError if held."""
    if fcntl is not None:
        fcntl.flock(f, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)
    elif msvcrt is not None:
        _win_lock(f, blocking)
    else:
        raise RuntimeError("no file locking available on this platform")


def release(f) -> None:
    if fcntl is not None:
        fcntl.flock(f, fcntl.LOCK_UN)
    elif msvcrt is not None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
