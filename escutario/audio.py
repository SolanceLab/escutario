# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Audio decoding for Escutário — ffmpeg in, float32 numpy out."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np

MAX_SECONDS = 600  # hard cap on decoded audio; matches the fetch cap


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    return path


def _decode(path: str, sr: int, channels: int) -> np.ndarray:
    cmd = [
        _ffmpeg(), "-nostdin", "-v", "error", "-i", str(path),
        "-t", str(MAX_SECONDS), "-ac", str(channels), "-ar", str(sr),
        "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}: {proc.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def load_mono(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    return _decode(path, sr, 1), sr


def load_stereo(path: str, sr: int = 44100) -> tuple[np.ndarray, int]:
    x = _decode(path, sr, 2)
    return x.reshape(-1, 2).T.copy(), sr


def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")
