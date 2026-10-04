# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Tests for escutario/split.py.

Fast tests mock the separation backend (no demucs model load, no network).
One slow test actually runs Demucs end to end — gated behind the
ESCUTARIO_RUN_SLOW env var so `pytest -q` stays fast and network-free by
default, and skips cleanly if demucs/model weights aren't available.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import soundfile as sf

from escutario import split as split_mod
from escutario.split import PEAK_CEILING, SeparationBackend, SplitResult, minus, split

SR = 44100


# --- helpers -----------------------------------------------------------------


def _write_stereo_wav(path, mono_or_stereo: np.ndarray, sr: int = SR) -> None:
    arr = np.asarray(mono_or_stereo, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.stack([arr, arr])  # (2, n)
    sf.write(str(path), arr.T, sr, subtype="PCM_16")


def _synth_mix(seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    """A quiet sine 'melody' + light noise, stereo (2, n) float32 in [-1, 1]."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    melody = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    noise = 0.02 * np.random.default_rng(0).standard_normal(n)
    mono = (melody + noise).astype(np.float32)
    return np.stack([mono, mono])


class FakeBackend:
    """Implements SeparationBackend without touching demucs."""

    def __init__(self, model: str = "fake", device: str = "cpu", **kwargs):
        self.name = f"fake:{model}"
        self.stems = ["vocals", "drums", "bass"]
        self.device = device
        self.calls = kwargs

    def separate(self, stereo_44k: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "vocals": stereo_44k * 0.5,
            "drums": stereo_44k * 0.3,
            "bass": stereo_44k * 0.2,
        }


class FlakyOnMPSBackend:
    """Raises RuntimeError on 'mps', succeeds on 'cpu' — for the fallback test."""

    seen_devices: list[str] = []

    def __init__(self, model: str = "fake", device: str = "cpu", **kwargs):
        self.name = f"fake:{model}"
        self.stems = ["vocals", "drums"]
        self.device = device
        FlakyOnMPSBackend.seen_devices.append(device)

    def separate(self, stereo_44k: np.ndarray) -> dict[str, np.ndarray]:
        if self.device == "mps":
            raise RuntimeError("MPS backend out of memory (simulated)")
        return {"vocals": stereo_44k.copy(), "drums": stereo_44k * 0.5}


# --- backend interface ---------------------------------------------------------


def test_fake_backend_satisfies_separation_backend_protocol():
    backend = FakeBackend()
    assert isinstance(backend, SeparationBackend)
    out = backend.separate(_synth_mix(0.5))
    assert set(out.keys()) == {"vocals", "drums", "bass"}


# --- split() orchestration, backend mocked --------------------------------------


def test_split_writes_expected_stems_with_correct_length(tmp_path, monkeypatch):
    monkeypatch.setattr(split_mod, "DemucsBackend", FakeBackend)

    mix = _synth_mix(seconds=1.0)
    src = tmp_path / "in.wav"
    _write_stereo_wav(src, mix)

    result = split(src, tmp_path / "out", device="cpu")

    assert isinstance(result, SplitResult)
    assert result.device == "cpu"
    assert set(result.stems.keys()) == {"vocals", "drums", "bass"}

    expected_len = mix.shape[1]
    for name, path in result.stems.items():
        assert path.exists()
        data, sr = sf.read(str(path), always_2d=True)
        assert sr == SR
        assert abs(data.shape[0] - expected_len) <= 1  # allow ±1 frame
        assert data.shape[1] == 2


def test_split_defaults_to_mps_when_available(tmp_path, monkeypatch):
    seen = {}

    class RecordingBackend(FakeBackend):
        def __init__(self, model="fake", device="cpu", **kwargs):
            seen["device"] = device
            super().__init__(model=model, device=device, **kwargs)

    monkeypatch.setattr(split_mod, "DemucsBackend", RecordingBackend)
    monkeypatch.setattr(split_mod, "_pick_device", lambda requested: requested or "mps")

    mix = _synth_mix(seconds=0.5)
    src = tmp_path / "in.wav"
    _write_stereo_wav(src, mix)

    result = split(src, tmp_path / "out")
    assert seen["device"] == "mps"
    assert result.device == "mps"


def test_split_retries_on_cpu_after_mps_runtime_error(tmp_path, monkeypatch):
    FlakyOnMPSBackend.seen_devices = []
    monkeypatch.setattr(split_mod, "DemucsBackend", FlakyOnMPSBackend)

    mix = _synth_mix(seconds=0.5)
    src = tmp_path / "in.wav"
    _write_stereo_wav(src, mix)

    result = split(src, tmp_path / "out", device="mps")

    assert FlakyOnMPSBackend.seen_devices == ["mps", "cpu"]
    assert result.device == "cpu"
    assert any("cpu" in note and "MPS" in note for note in result.notes)
    assert set(result.stems.keys()) == {"vocals", "drums"}


def test_split_does_not_retry_when_cpu_backend_itself_fails(tmp_path, monkeypatch):
    class AlwaysFailsBackend:
        def __init__(self, model="fake", device="cpu", **kwargs):
            self.name = "fake"
            self.stems = ["vocals"]
            self.device = device

        def separate(self, stereo_44k):
            raise RuntimeError("boom, not an MPS issue")

    monkeypatch.setattr(split_mod, "DemucsBackend", AlwaysFailsBackend)

    mix = _synth_mix(seconds=0.5)
    src = tmp_path / "in.wav"
    _write_stereo_wav(src, mix)

    with pytest.raises(RuntimeError, match="boom"):
        split(src, tmp_path / "out", device="cpu")


# --- load_stem_mono ---------------------------------------------------------------


def test_load_stem_mono_reads_the_written_stem(tmp_path, monkeypatch):
    monkeypatch.setattr(split_mod, "DemucsBackend", FakeBackend)

    mix = _synth_mix(seconds=1.0)
    src = tmp_path / "in.wav"
    _write_stereo_wav(src, mix)

    result = split(src, tmp_path / "out", device="cpu")
    mono, sr = split_mod.load_stem_mono(result, "vocals", sr=22050)
    assert sr == 22050
    assert mono.ndim == 1
    assert mono.size > 0


def test_load_stem_mono_unknown_stem_raises_keyerror(tmp_path, monkeypatch):
    monkeypatch.setattr(split_mod, "DemucsBackend", FakeBackend)
    mix = _synth_mix(seconds=0.5)
    src = tmp_path / "in.wav"
    _write_stereo_wav(src, mix)
    result = split(src, tmp_path / "out", device="cpu")
    with pytest.raises(KeyError):
        split_mod.load_stem_mono(result, "violin")


# --- minus() ---------------------------------------------------------------------


def _make_result_from_arrays(tmp_path, stems: dict[str, np.ndarray]) -> SplitResult:
    stem_paths = {}
    for name, arr in stems.items():
        p = tmp_path / f"{name}.wav"
        sf.write(str(p), arr, SR, subtype="PCM_16")
        stem_paths[name] = p
    return SplitResult(stems=stem_paths, model="fake", device="cpu", seconds_elapsed=0.0, notes=[])


def test_minus_sums_all_stems_except_target(tmp_path):
    n = SR // 2
    vocals = np.full((n, 2), 0.1, dtype=np.float32)
    drums = np.full((n, 2), 0.2, dtype=np.float32)
    bass = np.full((n, 2), 0.1, dtype=np.float32)
    result = _make_result_from_arrays(tmp_path, {"vocals": vocals, "drums": drums, "bass": bass})

    out_path = minus(result, "vocals", tmp_path / "out")
    assert out_path.exists()
    assert out_path.name == "minus_vocals.wav"

    data, sr = sf.read(str(out_path), always_2d=True)
    assert sr == SR
    # drums(0.2) + bass(0.1) = 0.3, well under PEAK_CEILING -> no scaling
    assert np.allclose(data, 0.3, atol=1e-3)
    assert not result.notes  # no scaling note when under ceiling


def test_minus_scales_down_to_avoid_clipping_and_notes_gain(tmp_path):
    n = SR // 2
    drums = np.full((n, 2), 0.9, dtype=np.float32)
    bass = np.full((n, 2), 0.9, dtype=np.float32)  # sum peak = 1.8, over PEAK_CEILING
    vocals = np.zeros((n, 2), dtype=np.float32)
    result = _make_result_from_arrays(tmp_path, {"vocals": vocals, "drums": drums, "bass": bass})

    out_path = minus(result, "vocals", tmp_path / "out")
    data, sr = sf.read(str(out_path), always_2d=True)

    peak = float(np.max(np.abs(data)))
    assert peak <= PEAK_CEILING + 1e-2  # small tolerance for 16-bit quantization
    assert len(result.notes) == 1
    assert "scaled by" in result.notes[0]
    assert "minus_vocals" in result.notes[0]


def test_minus_unknown_target_raises_keyerror(tmp_path):
    n = 1000
    result = _make_result_from_arrays(
        tmp_path, {"vocals": np.zeros((n, 2), dtype=np.float32), "drums": np.zeros((n, 2), dtype=np.float32)}
    )
    with pytest.raises(KeyError):
        minus(result, "violin", tmp_path / "out")


def test_minus_only_stem_raises_valueerror(tmp_path):
    n = 1000
    result = _make_result_from_arrays(tmp_path, {"vocals": np.zeros((n, 2), dtype=np.float32)})
    with pytest.raises(ValueError):
        minus(result, "vocals", tmp_path / "out")


def test_minus_handles_stems_differing_by_a_frame(tmp_path):
    """Model segment boundaries can leave stems a frame or two off; minus() should trim, not crash."""
    a = np.full((1000, 2), 0.1, dtype=np.float32)
    b = np.full((998, 2), 0.1, dtype=np.float32)  # 2 frames shorter
    vocals = np.zeros((1000, 2), dtype=np.float32)
    result = _make_result_from_arrays(tmp_path, {"vocals": vocals, "drums": a, "bass": b})
    out_path = minus(result, "vocals", tmp_path / "out")
    data, _sr = sf.read(str(out_path), always_2d=True)
    assert data.shape[0] == 998


# --- real end-to-end run (slow, real Demucs model) --------------------------------

RUN_SLOW = os.environ.get("ESCUTARIO_RUN_SLOW") == "1"


def _synth_song_mix(seconds: float = 12.0, sr: int = SR) -> np.ndarray:
    """Harmonic 'melody' + noise-burst 'drums' + low sine 'bass', stereo (2, n) float32."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(42)

    # melody: a few harmonics of a moving note (roughly violin/vocal register)
    f0 = 330.0 + 20.0 * np.sin(2 * np.pi * 0.15 * t)
    melody = 0.25 * np.sin(2 * np.pi * f0 * t)
    melody += 0.08 * np.sin(2 * np.pi * 2 * f0 * t)
    melody += 0.04 * np.sin(2 * np.pi * 3 * f0 * t)

    # drums: noise bursts on an 0.5s grid
    drums = np.zeros(n, dtype=np.float64)
    burst_len = int(0.05 * sr)
    for start in range(0, n, int(0.5 * sr)):
        end = min(start + burst_len, n)
        drums[start:end] += 0.5 * rng.standard_normal(end - start)

    # bass: low sine
    bass = 0.2 * np.sin(2 * np.pi * 82.0 * t)

    mono = (melody + drums + bass).astype(np.float32)
    mono = np.clip(mono, -0.95, 0.95)
    return np.stack([mono, mono])


@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="set ESCUTARIO_RUN_SLOW=1 to run the real Demucs model (downloads weights)")
def test_real_split_end_to_end(tmp_path):
    try:
        import demucs  # noqa: F401
    except ImportError:
        pytest.skip("demucs not installed")

    mix = _synth_song_mix(seconds=12.0)
    src = tmp_path / "song.wav"
    _write_stereo_wav(src, mix)

    t0 = time.monotonic()
    try:
        result = split(src, tmp_path / "out")
    except Exception as exc:  # model weights unavailable / download failed offline, etc.
        pytest.skip(f"real Demucs model unavailable: {exc}")
    elapsed = time.monotonic() - t0

    expected_stems = {"vocals", "drums", "bass", "guitar", "piano", "other"}
    assert set(result.stems.keys()) == expected_stems

    expected_len = mix.shape[1]
    for name, path in result.stems.items():
        assert path.exists()
        data, sr = sf.read(str(path), always_2d=True)
        assert sr == SR
        assert abs(data.shape[0] - expected_len) <= 1

    out_path = minus(result, "vocals", tmp_path / "out")
    assert out_path.exists()

    print(
        f"\n[real split] device={result.device} model={result.model} "
        f"seconds_elapsed={result.seconds_elapsed:.1f} wall_elapsed={elapsed:.1f} notes={result.notes}"
    )
