# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Escutário — track splitting.

Anne plays violin by ear and wants to isolate an instrument from a song to
learn it and play along. This module owns source separation: split a mix
into stems and build "everything but X" backing tracks.

Separation itself sits behind `SeparationBackend`, a small structural
interface. `DemucsBackend` (Demucs, via `demucs.api.Separator`) is the only
implementation today — Demucs has no violin stem, so a violin lands in its
catch-all "other" stem. A future named-instrument backend (a violin/strings
model, once a scout has picked one) plugs in behind the same interface;
it is NOT implemented here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np
import soundfile as sf

from escutario.audio import MAX_SECONDS, load_mono, load_stereo

# --- constants (starting values to calibrate) -------------------------------

DEFAULT_MODEL = "htdemucs_6s"  # 6-stem Demucs model: vocals, drums, bass, guitar, piano, other — CONTRACT default
PEAK_CEILING = 0.99  # minus() scales down only if the summed peak would exceed this — headroom under 0 dBFS before 16-bit quantization
OUTPUT_SR = 44100  # stems are always written at this rate — matches escutario.audio.load_stereo's default and Demucs's native rate


# --- backend interface -------------------------------------------------------


@runtime_checkable
class SeparationBackend(Protocol):
    """A pluggable source-separation backend.

    Anything that can turn a stereo 44.1kHz mix into named stems implements
    this shape — no inheritance required (it's a structural Protocol).

    Attributes
    ----------
    name: a short identifying string (e.g. "demucs:htdemucs_6s").
    stems: the stem names this backend produces, in a stable order.

    Slot for a future named-instrument backend
    -------------------------------------------
    Demucs's "other" stem swallows violin along with every instrument it
    doesn't name. A future `ViolinBackend` (or `StringsBackend`) — once a
    scout has picked a concrete model — implements this same Protocol with
    `stems = ["violin", "accompaniment"]` (or similar) and `split()` can
    take it as a drop-in replacement for `DemucsBackend`. Nothing here
    depends on Demucs internals beyond this interface, on purpose.
    """

    name: str
    stems: list[str]

    def separate(self, stereo_44k: np.ndarray) -> dict[str, np.ndarray]:
        """stereo_44k: shape (2, n) float32 at OUTPUT_SR. Returns {stem_name: (2, n) float32}."""
        ...


class DemucsBackend:
    """Demucs source separation via `demucs.api.Separator` (demucs 4.1.0 ships `demucs.api`)."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cpu", **separator_kwargs) -> None:
        from demucs import api as demucs_api  # local import: heavy (torch + model), only needed when actually separating

        self.name = f"demucs:{model}"
        self._separator = demucs_api.Separator(model=model, device=device, **separator_kwargs)
        self.stems = list(self._separator.model.sources)
        self.device = device

    def separate(self, stereo_44k: np.ndarray) -> dict[str, np.ndarray]:
        import torch

        wav = torch.from_numpy(np.ascontiguousarray(stereo_44k, dtype=np.float32))
        _, stems = self._separator.separate_tensor(wav, sr=OUTPUT_SR)
        return {name: tensor.detach().cpu().numpy().astype(np.float32) for name, tensor in stems.items()}


# --- result shape -------------------------------------------------------------


@dataclass
class SplitResult:
    stems: dict[str, Path]
    model: str
    device: str
    seconds_elapsed: float
    notes: list[str] = field(default_factory=list)


# --- device selection ---------------------------------------------------------


def _pick_device(requested: Optional[str]) -> str:
    if requested is not None:
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- wav io ---------------------------------------------------------------------


def _write_wav(path: Path, stereo: np.ndarray) -> None:
    """stereo: shape (2, n) float32. Writes 16-bit PCM WAV at OUTPUT_SR."""
    audio = np.clip(stereo, -1.0, 1.0).T  # soundfile wants (n_frames, n_channels)
    sf.write(str(path), audio, OUTPUT_SR, subtype="PCM_16")


def _read_wav_stereo(path: Path) -> np.ndarray:
    """Returns shape (n_frames, 2) float32."""
    data, _sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data


# --- public api -----------------------------------------------------------------


def split(
    path: Path,
    out_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    progress_cb: Optional[Callable[[dict], None]] = None,
) -> SplitResult:
    """Split `path` into stems, writing `<out_dir>/<stem>.wav` for each.

    Device: mps if available and not overridden, else cpu. On an MPS runtime
    error the separation is retried once on cpu, with a note recording the
    fallback.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    stereo, sr = load_stereo(str(path), OUTPUT_SR)
    if stereo.shape[1] / sr >= MAX_SECONDS:
        notes.append(f"input capped at {MAX_SECONDS}s (escutario.audio.MAX_SECONDS)")

    chosen_device = _pick_device(device)
    backend_kwargs = {}
    if progress_cb is not None:
        backend_kwargs["progress"] = True
        backend_kwargs["callback"] = progress_cb

    t0 = time.monotonic()
    backend = DemucsBackend(model=model, device=chosen_device, **backend_kwargs)
    try:
        stems_np = backend.separate(stereo)
    except RuntimeError as exc:
        if chosen_device != "mps":
            raise
        notes.append(f"MPS separation failed ({exc}); retried on cpu")
        chosen_device = "cpu"
        backend = DemucsBackend(model=model, device=chosen_device, **backend_kwargs)
        stems_np = backend.separate(stereo)
    seconds_elapsed = time.monotonic() - t0

    stem_paths: dict[str, Path] = {}
    for name, wav in stems_np.items():
        stem_path = out_dir / f"{name}.wav"
        _write_wav(stem_path, wav)
        stem_paths[name] = stem_path

    return SplitResult(
        stems=stem_paths,
        model=model,
        device=chosen_device,
        seconds_elapsed=seconds_elapsed,
        notes=notes,
    )


def load_stem_mono(result: SplitResult, stem: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load a split stem as mono at `sr`, for the analysers (breath.py, wail.py)."""
    if stem not in result.stems:
        raise KeyError(f"no such stem {stem!r}; available: {sorted(result.stems)}")
    return load_mono(str(result.stems[stem]), sr)


def minus(result: SplitResult, target: str, out_dir: Path) -> Path:
    """Write `<out_dir>/minus_<target>.wav` = sum of every stem except `target`.

    This is the play-along backing track: everything except the instrument
    Anne is isolating. Peak-safe — scales down only if the summed peak would
    exceed PEAK_CEILING, and records the gain applied in `result.notes`.
    """
    if target not in result.stems:
        raise KeyError(f"no such stem {target!r}; available: {sorted(result.stems)}")

    others = [name for name in result.stems if name != target]
    if not others:
        raise ValueError(f"'{target}' is the only stem; nothing to sum for minus()")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = [_read_wav_stereo(result.stems[name]) for name in others]
    min_len = min(a.shape[0] for a in arrays)  # stems can differ by a frame or two at model segment boundaries
    mix = np.zeros((min_len, 2), dtype=np.float64)
    for a in arrays:
        mix += a[:min_len].astype(np.float64)

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    gain = 1.0
    if peak > PEAK_CEILING:
        gain = PEAK_CEILING / peak
        result.notes.append(
            f"minus_{target}: scaled by {gain:.4f} to keep peak <= {PEAK_CEILING} (unscaled peak was {peak:.4f})"
        )

    out_path = out_dir / f"minus_{target}.wav"
    sf.write(str(out_path), (mix * gain).astype(np.float32), OUTPUT_SR, subtype="PCM_16")
    return out_path
