# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Windows compatibility, simulated: no fcntl / resource, a fake msvcrt. No real Windows needed."""

import builtins
import errno
import importlib
import sys
import types

import pytest

from escutario import _lock
from escutario import instrument as ins


class _FakeMsvcrt(types.SimpleNamespace):
    def __init__(self, busy_first=0):
        super().__init__(LK_LOCK=1, LK_NBLCK=2, LK_UNLCK=0, calls=[], busy=busy_first)

    def locking(self, fd, mode, nbytes):
        self.calls.append((mode, nbytes))
        if mode != self.LK_UNLCK and self.busy > 0:
            self.busy -= 1
            raise OSError(errno.EDEADLK, "resource deadlock")


def _windows(monkeypatch, fake):
    monkeypatch.setattr(_lock, "fcntl", None)
    monkeypatch.setattr(_lock, "msvcrt", fake)
    monkeypatch.setattr(_lock, "_RETRY_S", 0)


def test_msvcrt_lock_retries_deadlock_then_acquires_and_releases(monkeypatch, tmp_path):
    fake = _FakeMsvcrt(busy_first=1)
    _windows(monkeypatch, fake)
    with open(tmp_path / "x.lock", "a") as f:
        _lock.acquire(f)
        _lock.release(f)
    assert fake.calls == [(fake.LK_LOCK, 1), (fake.LK_LOCK, 1), (fake.LK_UNLCK, 1)]


def test_msvcrt_nonblocking_raises_blockingioerror_when_held(monkeypatch, tmp_path):
    fake = _FakeMsvcrt(busy_first=5)
    _windows(monkeypatch, fake)
    with open(tmp_path / "x.lock", "a") as f:
        with pytest.raises(BlockingIOError):
            _lock.acquire(f, blocking=False)
    assert fake.calls == [(fake.LK_NBLCK, 1)]


def test_posix_lock_is_exclusive_and_nonblocking_refuses(tmp_path):
    if _lock.fcntl is None:
        pytest.skip("no fcntl here")
    with open(tmp_path / "p.lock", "a") as a, open(tmp_path / "p.lock", "a") as b:
        _lock.acquire(a, blocking=False)
        with pytest.raises(BlockingIOError):
            _lock.acquire(b, blocking=False)
        _lock.release(a)
        _lock.acquire(b, blocking=False)


def test_peak_rss_is_none_without_resource(monkeypatch):
    monkeypatch.setattr(ins, "resource", None)
    assert ins.peak_rss_bytes() is None


def test_modules_import_with_fcntl_and_resource_hidden(monkeypatch):
    real_import = builtins.__import__

    def hiding(name, *a, **k):
        if name in ("fcntl", "resource"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    names = ["escutario._lock", "escutario.listen", "escutario.words", "escutario.worker", "escutario.instrument"]
    saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    monkeypatch.setattr(builtins, "__import__", hiding)
    try:
        for n in names:
            mod = importlib.import_module(n)
        assert sys.modules["escutario._lock"].fcntl is None
        assert sys.modules["escutario.instrument"].resource is None
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
        pkg = sys.modules["escutario"]
        for n, mod in saved.items():   # importing re-binds the package attribute; put the originals back
            setattr(pkg, n.split(".")[1], mod)
