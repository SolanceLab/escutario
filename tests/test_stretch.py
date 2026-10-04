# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from escutario import stretch

SR = 44100
needs_rubberband = pytest.mark.skipif(not Path(stretch.RUBBERBAND).is_file(), reason="rubberband CLI not installed")


def _tone(path: Path, freq=440.0, seconds=2.0, sr=SR, channels=1, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels == 2:
        x = np.stack([x, x], axis=1)
    sf.write(str(path), x, sr, subtype="PCM_16")
    return x


def _dominant_hz(x: np.ndarray, sr: int) -> float:
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x[len(x) // 4: 3 * len(x) // 4]  # avoid edges
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    k = int(np.argmax(spec[1:-1])) + 1
    a, b, c = np.log(spec[k - 1: k + 2] + 1e-12)
    delta = 0.5 * (a - c) / (a - 2 * b + c)  # parabolic interpolation
    return (k + delta) * sr / len(x)


@needs_rubberband
@pytest.mark.parametrize("channels", [1, 2])
def test_slow_down_half_speed_keeps_pitch(tmp_path, channels):
    src = tmp_path / "tone.wav"
    _tone(src, channels=channels)
    out = stretch.slow_down(src, tmp_path / "slow.wav", 0.5)
    y, sr = sf.read(str(out), dtype="float32")
    assert sr == SR
    assert abs(len(y) / sr - 4.0) <= 0.02 * 4.0
    assert abs(_dominant_hz(y, sr) - 440.0) <= 3.0


@needs_rubberband
def test_slow_down_three_quarters(tmp_path):
    src = tmp_path / "tone.wav"
    _tone(src, freq=659.25, seconds=3.0)
    y, sr = sf.read(str(stretch.slow_down(src, tmp_path / "s.wav", 0.75)), dtype="float32")
    assert abs(len(y) / sr - 4.0) <= 0.02 * 4.0
    assert abs(_dominant_hz(y, sr) - 659.25) <= 3.0


def test_argv_is_a_list_with_fine_engine_and_inverse_time(tmp_path):
    argv = stretch.rubberband_argv(tmp_path / "a b.wav", tmp_path / "o.wav", 0.75)
    assert isinstance(argv, list) and argv[0] == stretch.RUBBERBAND
    assert "--time" in argv and float(argv[argv.index("--time") + 1]) == pytest.approx(1 / 0.75)
    assert not any(a in argv for a in ("--pitch", "-p", "--frequency", "-f"))
    if Path(stretch.RUBBERBAND).is_file():
        assert "--fine" in argv
    assert str(tmp_path / "a b.wav") in argv  # path with a space stays one argument


def test_slow_down_rejects_bad_rate_and_same_path(tmp_path):
    src = tmp_path / "tone.wav"
    _tone(src, seconds=0.2)
    with pytest.raises(ValueError):
        stretch.slow_down(src, tmp_path / "o.wav", 0.0)
    with pytest.raises(ValueError):
        stretch.slow_down(src, tmp_path / "o.wav", 5.0)
    with pytest.raises(ValueError):
        stretch.slow_down(src, src, 0.5)
    with pytest.raises(FileNotFoundError):
        stretch.slow_down(tmp_path / "missing.wav", tmp_path / "o.wav", 0.5)


@pytest.mark.parametrize("repeats,xf_ms", [(4, 20), (1, 20), (3, 0), (2, 50)])
def test_loop_length(tmp_path, repeats, xf_ms):
    src = tmp_path / "tone.wav"
    _tone(src, seconds=3.0, channels=2)
    out = stretch.loop(src, 0.5, 1.5, tmp_path / "loop.wav", repeats=repeats, crossfade_ms=xf_ms)
    y, sr = sf.read(str(out), dtype="float32", always_2d=True)
    seg = SR  # 1.0 s
    xf = int(round(xf_ms / 1000 * SR))
    assert y.shape == (repeats * seg - (repeats - 1) * xf, 2)
    assert len(y) == stretch.loop_length(seg, repeats, xf)
    assert np.max(np.abs(y)) <= 1.0


def test_loop_seams_have_no_dropout(tmp_path):
    # steady tone: an equal-power crossfade must not dip toward silence at the seam
    src = tmp_path / "tone.wav"
    _tone(src, seconds=2.0, amp=0.5)
    y, _ = sf.read(str(stretch.loop(src, 0.0, 1.0, tmp_path / "l.wav", repeats=3, crossfade_ms=20)), dtype="float32")
    win = int(0.005 * SR)
    rms = np.sqrt(np.convolve(y ** 2, np.ones(win) / win, mode="valid"))
    assert rms[win:-win].min() > 0.1  # tone rms ≈ 0.35


def test_loop_rejects_bad_ranges(tmp_path):
    src = tmp_path / "tone.wav"
    _tone(src, seconds=1.0)
    for a, b in [(0.5, 0.5), (0.8, 0.2), (-0.1, 0.5), (0.5, 2.0)]:
        with pytest.raises(ValueError):
            stretch.loop(src, a, b, tmp_path / "o.wav")
    with pytest.raises(ValueError):
        stretch.loop(src, 0.0, 0.5, tmp_path / "o.wav", repeats=0)
    with pytest.raises(ValueError):
        stretch.loop(src, 0.0, 0.5, src)
