# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Tests for escutario.pitch — synthetic signals with known ground truth."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from escutario import pitch as P
from escutario.types import PitchTrack

SR = 22050

HAS_LIBROSA = importlib.util.find_spec("librosa") is not None
HAS_CREPE = P.crepe_available()

ENGINES = [
    pytest.param("crepe", marks=pytest.mark.skipif(not HAS_CREPE, reason="torch/torchcrepe not installed")),
    pytest.param("pyin", marks=pytest.mark.skipif(not HAS_LIBROSA, reason="librosa not installed")),
]


def harmonic(f0_track: np.ndarray, amps=(1.0, 0.5, 0.3, 0.2), sr: int = SR) -> np.ndarray:
    """A voice-like harmonic tone following a per-sample f0 track."""
    phase = 2 * np.pi * np.cumsum(f0_track) / sr
    y = sum(a * np.sin((k + 1) * phase) for k, a in enumerate(amps))
    return (0.3 * y / np.max(np.abs(y))).astype(np.float32)


def cents(a, b):
    return 1200 * np.log2(a / b)


def octave_jumps(track: PitchTrack) -> int:
    f = track.f0_hz
    both = (f[1:] > 0) & (f[:-1] > 0)
    d = np.abs(cents(np.where(both, f[1:], 1.0), np.where(both, f[:-1], 1.0)))
    return int((both & ((np.abs(d - 1200) <= 80) | (np.abs(d - 2400) <= 80))).sum())


def inner(track: PitchTrack, t0: float, t1: float) -> np.ndarray:
    return (track.times >= t0) & (track.times <= t1)


# ------------------------------------------------------------------ shape & API


@pytest.mark.parametrize("method", ENGINES)
def test_steady_hop_and_shapes(method):
    x = harmonic(np.full(SR, 220.0))
    tr = P.track_pitch(x, SR, method=method)
    assert isinstance(tr, PitchTrack)
    assert 0.0099 <= tr.hop_s <= 0.012
    assert np.allclose(np.diff(tr.times), tr.hop_s)
    assert len(tr.times) == len(tr.f0_hz) == len(tr.confidence)
    assert tr.times[-1] == pytest.approx(1.0, abs=0.03)
    assert np.all((tr.confidence >= 0) & (tr.confidence <= 1))


@pytest.mark.parametrize("method", ENGINES)
def test_confidence_boundary_matches_voicing(method):
    x = np.concatenate([np.zeros(SR // 2), harmonic(np.full(SR, 300.0)), np.zeros(SR // 2)]).astype(np.float32)
    tr = P.track_pitch(x, SR, method=method)
    voiced = tr.f0_hz > 0
    assert voiced.any() and (~voiced).any()
    assert np.all(tr.confidence[voiced] >= P.CONF_BOUNDARY)
    assert np.all(tr.confidence[~voiced] < P.CONF_BOUNDARY)


@pytest.mark.parametrize("method", ENGINES)
def test_steady_tone_is_accurate(method):
    tr = P.track_pitch(harmonic(np.full(SR, 261.63)), SR, method=method)
    m = inner(tr, 0.1, 0.9)
    assert (tr.f0_hz[m] > 0).mean() > 0.95
    err = np.abs(cents(tr.f0_hz[m & (tr.f0_hz > 0)], 261.63))
    assert np.median(err) < 10


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        P.track_pitch(np.zeros(100, np.float32), SR, method="yin")


def test_save_npz_round_trip(tmp_path):
    tr = PitchTrack(times=np.arange(5) * 0.01, f0_hz=np.array([0, 220, 221, 0, 0.0]),
                    confidence=np.array([0.1, 0.9, 0.8, 0.2, 0.0]), hop_s=0.01)
    path = tmp_path / "pitch.npz"
    P.save_npz(tr, path, SR)
    d = np.load(path)
    assert set(d.files) == {"times", "f0", "conf", "hop_s", "sr"}
    assert int(d["sr"]) == SR and float(d["hop_s"]) == 0.01
    back = P.load_npz(path)
    assert np.array_equal(back.f0_hz, tr.f0_hz) and np.array_equal(back.confidence, tr.confidence)


# ------------------------------------------------------------------ musical cases


# pyin's pitch grid + HMM smoothing under-reads extent (~40 of 60 cents), so its tolerance is wider
EXTENT_TOLERANCE_C = {"crepe": 15.0, "pyin": 25.0}


@pytest.mark.parametrize("method", ENGINES)
def test_vibrato_rate_and_extent_recovered(method):
    rate_hz, extent_c, base = 5.5, 60.0, 440.0  # extent = peak deviation in cents
    t = np.arange(2 * SR) / SR
    f0 = base * 2 ** (extent_c * np.sin(2 * np.pi * rate_hz * t) / 1200)
    tr = P.track_pitch(harmonic(f0), SR, method=method)
    m = inner(tr, 0.2, 1.8)
    assert (tr.f0_hz[m] > 0).mean() > 0.95
    c = cents(tr.f0_hz[m], base)
    c = c - c.mean()
    spec = np.abs(np.fft.rfft(c * np.hanning(len(c)), n=8192))
    freqs = np.fft.rfftfreq(8192, tr.hop_s)
    assert abs(freqs[spec.argmax()] - rate_hz) < 0.3
    measured_extent = np.sqrt(2) * c.std()
    assert abs(measured_extent - extent_c) < EXTENT_TOLERANCE_C[method]


@pytest.mark.parametrize("method", ENGINES)
def test_glide_is_followed(method):
    t = np.arange(SR) / SR
    f0 = 300.0 * 2 ** t  # one octave up over one second
    tr = P.track_pitch(harmonic(f0), SR, method=method)
    m = inner(tr, 0.1, 0.9)
    truth = 300.0 * 2 ** tr.times[m]
    ok = (tr.f0_hz[m] > 0) & (np.abs(cents(np.maximum(tr.f0_hz[m], 1.0), truth)) < 50)
    assert ok.mean() > 0.9
    assert octave_jumps(tr) == 0


@pytest.mark.parametrize("method", ENGINES[:1])
def test_two_tones_follow_the_stronger_without_octave_hops(method):
    n = 2 * SR
    strong = harmonic(np.full(n, 330.0))
    weak = harmonic(np.full(n, 523.25)) * 0.3
    tr = P.track_pitch(strong + weak, SR, method=method)
    m = inner(tr, 0.1, 1.9)
    on_strong = (tr.f0_hz[m] > 0) & (np.abs(cents(np.maximum(tr.f0_hz[m], 1.0), 330.0)) < 50)
    assert on_strong.mean() > 0.9
    assert octave_jumps(tr) == 0


@pytest.mark.skipif(not HAS_LIBROSA, reason="librosa not installed")
def test_pyin_two_tones_known_limit_steady_but_not_the_stronger():
    """Documented limit: pyin settles an octave below the stronger tone (their
    shared period) — steady, no hopping, but wrong. The crepe engine is the default."""
    n = 2 * SR
    tr = P.track_pitch(harmonic(np.full(n, 330.0)) + harmonic(np.full(n, 523.25)) * 0.3, SR, method="pyin")
    assert octave_jumps(tr) == 0


# ------------------------------------------------------------------ false positives & edges


@pytest.mark.parametrize("method", ENGINES)
def test_silence_is_unvoiced(method):
    tr = P.track_pitch(np.zeros(SR, np.float32), SR, method=method)
    assert len(tr.times) > 0
    assert not (tr.f0_hz > 0).any()


@pytest.mark.parametrize("method", ENGINES)
def test_white_noise_is_mostly_unvoiced(method):
    rng = np.random.default_rng(7)
    x = (0.1 * rng.standard_normal(2 * SR)).astype(np.float32)
    tr = P.track_pitch(x, SR, method=method)
    assert (tr.f0_hz > 0).mean() < 0.05


@pytest.mark.parametrize("method", ENGINES)
@pytest.mark.parametrize("n", [0, 1, 100, 1000])
def test_very_short_input_does_not_crash(method, n):
    x = harmonic(np.full(n, 220.0)) if n > 1 else np.zeros(n, np.float32)
    tr = P.track_pitch(x, SR, method=method)
    assert len(tr.times) == len(tr.f0_hz) == len(tr.confidence)
    assert np.all(np.isfinite(tr.f0_hz)) and np.all(np.isfinite(tr.confidence))


@pytest.mark.parametrize("method", ENGINES)
def test_other_sample_rates_give_the_same_pitch(method):
    sr = 44100
    t = np.arange(sr) / sr
    x = (0.3 * np.sin(2 * np.pi * 196.0 * t) + 0.15 * np.sin(2 * np.pi * 392.0 * t)).astype(np.float32)
    tr = P.track_pitch(x, sr, method=method)
    m = inner(tr, 0.1, 0.9) & (tr.f0_hz > 0)
    assert m.sum() > 50
    assert np.median(np.abs(cents(tr.f0_hz[m], 196.0))) < 15


# ------------------------------------------------------------------ decoder logic (fabricated activations, no network)


def _fake_activations(midi_per_frame: np.ndarray, strength: float = 0.9) -> np.ndarray:
    n = len(midi_per_frame)
    act = np.full((n, P.CREPE_BINS), 0.01, dtype=np.float32)
    bins = np.arange(P.CREPE_BINS)
    for i, m in enumerate(midi_per_frame):
        hz = 440.0 * 2 ** ((m - 69) / 12)
        centre = P._hz_to_bin(hz)
        act[i] = np.maximum(act[i], strength * np.exp(-0.5 * ((bins - centre) / 1.5) ** 2))
    return act


def _loud_y16(n_frames: int) -> np.ndarray:
    t = np.arange(n_frames * int(P.HOP_S * P.CREPE_SR)) / P.CREPE_SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_viterbi_ignores_a_two_frame_octave_flicker():
    midi = np.full(200, 68.0)
    midi[100:102] = 80.0  # two frames an octave up — a stray harmonic
    tr = P._crepe_decode(_fake_activations(midi), _loud_y16(200), decoder="viterbi")
    assert octave_jumps(tr) == 0
    inner_f0 = tr.f0_hz[5:-5]
    voiced = inner_f0 > 0
    assert (~voiced).sum() <= 4  # the flicker frames may drop to unvoiced, never jump
    assert np.all(np.abs(cents(inner_f0[voiced], 415.305)) < 30)


def test_viterbi_takes_a_sustained_leap():
    midi = np.concatenate([np.full(100, 68.0), np.full(100, 80.0)])  # a real octave leap that stays
    tr = P._crepe_decode(_fake_activations(midi), _loud_y16(200), decoder="viterbi")
    assert np.all(np.abs(cents(tr.f0_hz[120:195], 830.61)) < 30)
    assert np.all(np.abs(cents(tr.f0_hz[5:95], 415.305)) < 30)


def test_argmax_decoder_does_flicker():
    midi = np.full(200, 68.0)
    midi[100:104] = 80.0
    tr = P._crepe_decode(_fake_activations(midi), _loud_y16(200), decoder="argmax")
    assert octave_jumps(tr) >= 1  # the reason viterbi is the default


def test_degenerate_inference_is_retried_then_refused(monkeypatch):
    calls = []

    def constant(y16, capacity, device, *, batch=None):
        calls.append(batch)
        return np.full((1 + len(y16) // 160, P.CREPE_BINS), 0.039, dtype=np.float32)

    monkeypatch.setattr(P, "_crepe_activations", constant)
    monkeypatch.setattr(P, "_pick_device", lambda: "cpu")
    with pytest.raises(RuntimeError, match="constant activations"):
        P.track_pitch(harmonic(np.full(SR, 220.0)), SR, method="crepe")
    assert calls == [None, P.CREPE_RETRY_BATCH_FRAMES]


def test_constant_activations_on_silence_are_not_an_error(monkeypatch):
    monkeypatch.setattr(P, "_crepe_activations",
                        lambda y16, c, d, *, batch=None: np.full((1 + len(y16) // 160, P.CREPE_BINS), 0.039, np.float32))
    monkeypatch.setattr(P, "_pick_device", lambda: "cpu")
    tr = P.track_pitch(np.zeros(SR, np.float32), SR, method="crepe")
    assert not (tr.f0_hz > 0).any()


def test_auto_prefers_crepe_and_falls_back_to_pyin(monkeypatch):
    seen = []
    monkeypatch.setattr(P, "_track_crepe", lambda x, sr, model: seen.append(("crepe", model)) or P._empty_track())
    monkeypatch.setattr(P, "_track_pyin", lambda x, sr: seen.append(("pyin", None)) or P._empty_track())
    monkeypatch.setattr(P, "crepe_available", lambda: True)
    P.track_pitch(np.ones(10, np.float32), SR)
    monkeypatch.setattr(P, "crepe_available", lambda: False)
    P.track_pitch(np.ones(10, np.float32), SR)
    assert seen == [("crepe", P.CREPE_MODEL), ("pyin", None)]


def test_non_finite_audio_is_refused_not_silenced():
    import numpy as np, pytest
    from escutario.pitch import track_pitch
    x = np.zeros(16000, dtype=np.float32); x[100] = np.nan
    with pytest.raises(ValueError):
        track_pitch(x, 16000, method="pyin")
