# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Exit the test process without running native library destructors.

torch and onnxruntime are both loaded in one process by the pitch and notes tests. On macOS
their static destructors race at interpreter shutdown and abort with
"libc++abi: recursive_mutex lock failed", which pops a "Python quit unexpectedly" crash
dialog after every run even though all tests passed. Results are already reported by then,
so flush and leave with pytest's own exit status.
"""

import os
import sys


def pytest_unconfigure(config):
    if "torch" in sys.modules or "onnxruntime" in sys.modules:
        status = getattr(config, "_escutario_exitstatus", 0)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(int(status))


def pytest_sessionfinish(session, exitstatus):
    session.config._escutario_exitstatus = exitstatus


import pytest


@pytest.fixture(autouse=True)
def _worker_state_in_tmp(tmp_path, monkeypatch):
    """The worker's memory files (drop fingerprints, gallery skips, the words spend counter) must never be
    the real ones under out/ while tests run: a test fingerprint there would make the real worker skip a drop."""
    try:
        from escutario import worker as W
        monkeypatch.setattr(W, "DROP_HASHES", tmp_path / "state" / "drop-hashes.json")
        monkeypatch.setattr(W, "GALLERY_SKIPPED", tmp_path / "state" / "gallery-skipped.json")
        monkeypatch.setattr(W, "GALLERY_TO_ADD", tmp_path / "state" / "gallery-to-add.json")
        if hasattr(W, "DROP_PRINTS"):
            monkeypatch.setattr(W, "DROP_PRINTS", tmp_path / "state" / "drop-prints.json")
    except Exception:  # noqa: BLE001 — tests that never import the worker are unaffected
        pass
    try:
        from escutario import words as Wd
        monkeypatch.setattr(Wd, "USAGE_PATH", tmp_path / "state" / "words-usage.json")
    except Exception:  # noqa: BLE001
        pass
    yield
