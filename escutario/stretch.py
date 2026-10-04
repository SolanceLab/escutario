# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Practice tools for Escutário — slow a passage down (pitch kept) and loop it.

`slow_down` shells out to the system Rubber Band CLI (argv list, no shell). `loop` is pure numpy.
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

RUBBERBAND = "/opt/homebrew/bin/rubberband"  # system Rubber Band CLI (v4.0.0 at time of writing)

MAX_LOOP_REPEATS = 100  # starting value: sanity cap on loop repeats
MAX_CROSSFADE_MS = 500.0  # starting value: longest crossfade accepted
PCM_SUBTYPES_KEPT = ("PCM_16", "PCM_24", "FLOAT")  # input subtypes copied to loop output; others become FLOAT
MIN_RATE = 0.25  # starting value: slowest practice speed allowed (25%)
MAX_RATE = 2.0  # starting value: fastest speed allowed (200%)
RUBBERBAND_TIMEOUT_S = 600  # starting value: generous cap for a full song on the R3 engine


@functools.lru_cache(maxsize=4)
def _supports_fine(binary: str) -> bool:
    """True if this rubberband build offers the R3 engine (`--fine`)."""
    try:
        proc = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "--fine" in (proc.stdout + proc.stderr)


def rubberband_argv(in_wav: Path, out_wav: Path, rate: float, binary: str = RUBBERBAND) -> list[str]:
    """argv for playing `in_wav` at `rate` × speed (0.75 = 75% speed), pitch unchanged."""
    argv = [binary, "--quiet"]
    if _supports_fine(binary):
        argv.append("--fine")
    # --time is the duration multiplier: 75% speed = 1/0.75 × as long. No pitch option -> pitch kept.
    argv += ["--time", repr(1.0 / rate), str(in_wav), str(out_wav)]
    return argv


def slow_down(in_wav: Path, out_wav: Path, rate: float) -> Path:
    """Write `out_wav` = `in_wav` at `rate` × speed with pitch preserved. Returns out_wav."""
    in_wav, out_wav = Path(in_wav), Path(out_wav)
    if not (MIN_RATE <= rate <= MAX_RATE):
        raise ValueError(f"rate {rate} outside [{MIN_RATE}, {MAX_RATE}]")
    if not in_wav.is_file():
        raise FileNotFoundError(in_wav)
    if in_wav.resolve() == out_wav.resolve():
        raise ValueError("out_wav must differ from in_wav")
    if not Path(RUBBERBAND).is_file():
        raise FileNotFoundError(f"Rubber Band CLI not found at {RUBBERBAND} (brew install rubberband)")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(rubberband_argv(in_wav, out_wav, rate), capture_output=True, text=True,
                          timeout=RUBBERBAND_TIMEOUT_S)
    if proc.returncode != 0 or not out_wav.is_file():
        raise RuntimeError(f"rubberband failed ({proc.returncode}): {(proc.stderr or proc.stdout)[-400:]}")
    return out_wav


def loop_length(segment_frames: int, repeats: int, crossfade_frames: int) -> int:
    """Frames in the looped output: each seam overlaps `crossfade_frames`."""
    return repeats * segment_frames - (repeats - 1) * crossfade_frames


def loop(in_wav: Path, start_s: float, end_s: float, out_wav: Path, repeats: int = 4,
         crossfade_ms: float = 20) -> Path:
    """Write `out_wav` = [start_s, end_s) of `in_wav` repeated `repeats` times.

    Each seam is an equal-power crossfade of `crossfade_ms` (the tail of one pass fades into the
    head of the next), so the output is `loop_length(segment, repeats, crossfade)` frames long.
    """
    in_wav, out_wav = Path(in_wav), Path(out_wav)
    if not (1 <= repeats <= MAX_LOOP_REPEATS):
        raise ValueError(f"repeats {repeats} outside [1, {MAX_LOOP_REPEATS}]")
    if not (0 <= crossfade_ms <= MAX_CROSSFADE_MS):
        raise ValueError(f"crossfade_ms {crossfade_ms} outside [0, {MAX_CROSSFADE_MS}]")
    if in_wav.resolve() == out_wav.resolve():
        raise ValueError("out_wav must differ from in_wav")
    info = sf.info(str(in_wav))
    sr = info.samplerate
    a, b = int(round(start_s * sr)), int(round(end_s * sr))
    if not (0 <= a < b <= info.frames):
        raise ValueError(f"loop range [{start_s}, {end_s}) s is outside the file (0–{info.frames / sr:.3f} s)")
    seg, _ = sf.read(str(in_wav), start=a, stop=b, dtype="float32", always_2d=True)  # (frames, ch)
    n = seg.shape[0]
    xf = min(int(round(crossfade_ms / 1000.0 * sr)), n // 2)

    out = np.zeros((loop_length(n, repeats, xf), seg.shape[1]), dtype=np.float32)
    if xf > 0:
        theta = np.linspace(0.0, np.pi / 2, xf, dtype=np.float32)[:, None]
        fade_in, fade_out = np.sin(theta), np.cos(theta)
    pos = 0
    for r in range(repeats):
        piece = seg.copy()
        if xf > 0 and r > 0:
            piece[:xf] *= fade_in
        if xf > 0 and r < repeats - 1:
            piece[-xf:] *= fade_out
        out[pos:pos + n] += piece
        pos += n - xf

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:  # equal-power sum of correlated audio can exceed full scale at a seam
        out /= peak
    subtype = info.subtype if info.subtype in PCM_SUBTYPES_KEPT else "FLOAT"
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), out, sr, subtype=subtype)
    return out_wav
