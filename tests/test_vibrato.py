# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
import numpy as np

from escutario.types import PitchTrack
from escutario.vibrato import detect_vibrato

RNG = np.random.default_rng(3)
HOP = 0.01


def track(cents_curve, voiced=None):
    n = len(cents_curve)
    f0 = 440.0 * 2 ** (np.asarray(cents_curve) / 1200.0)
    conf = np.full(n, 0.9)
    if voiced is not None:
        f0 = np.where(voiced, f0, 0.0)
        conf = np.where(voiced, conf, 0.1)
    return PitchTrack(times=np.arange(n) * HOP, f0_hz=f0, confidence=conf, hop_s=HOP)


def held(dur, base_cents=0.0, rate=0.0, extent=0.0, jitter=0.0):
    t = np.arange(int(dur / HOP)) * HOP
    return base_cents + extent * np.sin(2 * np.pi * rate * t) + jitter * RNG.standard_normal(len(t))


def test_vibrato_rate_and_extent_recovered():
    r = detect_vibrato(track(held(2.0, rate=5.5, extent=60)))
    assert len(r["with_vibrato"]) == 1
    v = r["with_vibrato"][0]
    assert abs(v["rate_hz"] - 5.5) <= 0.6 and 45 <= v["extent_cents"] <= 70 and v["confidence"] == "high"


def test_straight_note_has_no_vibrato():
    r = detect_vibrato(track(held(2.0, jitter=3)))
    assert r["judged"] == 1 and r["with_vibrato"] == [] and r["straight"] == 1


def test_tiny_shimmer_below_audible_width_is_not_vibrato():
    r = detect_vibrato(track(held(2.0, rate=7.0, extent=7)))
    assert r["with_vibrato"] == []


def test_slow_drift_and_glide_are_not_vibrato():
    t = np.arange(int(2.0 / HOP)) * HOP
    r = detect_vibrato(track(80 * t))  # a steady 80 cents/s rise
    assert r["with_vibrato"] == []


def test_short_notes_are_not_judged():
    curve = np.concatenate([held(0.6, rate=6, extent=60), held(0.2, base_cents=900)])
    r = detect_vibrato(track(curve))
    assert r["judged"] == 0


def test_white_jitter_does_not_fake_vibrato():
    false = sum(1 for seed in range(20)
                if detect_vibrato(track(np.random.default_rng(seed).normal(0, 25, 200)))["with_vibrato"])
    assert false <= 2


def test_two_notes_judged_separately():
    curve = np.concatenate([held(1.5, 0, rate=6, extent=50), held(1.5, 700)])
    r = detect_vibrato(track(curve))
    assert r["judged"] == 2 and len(r["with_vibrato"]) == 1 and r["with_vibrato"][0]["start_s"] < 0.5


def test_empty_track():
    r = detect_vibrato(PitchTrack(times=np.array([]), f0_hz=np.array([]), confidence=np.array([]), hop_s=HOP))
    assert r["judged"] == 0


def test_wide_melodic_movement_is_not_vibrato():
    r = detect_vibrato(track(held(2.0, rate=5.0, extent=140)))
    assert r["with_vibrato"] == []


def test_irregular_wander_in_the_band_is_not_vibrato():
    rng = np.random.default_rng(11)
    t = np.arange(200) * HOP
    wander = 40 * np.sin(2 * np.pi * 5 * t + np.cumsum(rng.normal(0, 0.6, 200)))  # phase drifts: no steady wave
    r = detect_vibrato(track(wander))
    assert r["with_vibrato"] == []
