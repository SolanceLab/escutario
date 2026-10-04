# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Harmony on synthetic signals with known ground truth."""

import numpy as np
import pytest

pytest.importorskip("librosa")

from escutario.harmony import (
    _choose_key,
    _level_alternatives,
    analyze_harmony,
    estimate_energy,
    estimate_key,
    estimate_tempo,
    detect_sections,
)

SR = 22050
RNG = np.random.default_rng(7)


def tone(midis, dur, sr=SR, amp=0.2):
    t = np.arange(int(dur * sr)) / sr
    x = np.zeros_like(t)
    for m in midis:
        f = 440.0 * 2 ** ((m - 69) / 12)
        for h in range(1, 5):
            if h * f < sr / 2:
                x += (0.6 ** (h - 1)) * np.sin(2 * np.pi * h * f * t)
    env = np.minimum(1, np.minimum(t / 0.02, (dur - t) / 0.05))
    return (amp * x * env / max(1, len(midis))).astype(np.float32)


def clicks(bpm, dur, sr=SR, accents=None):
    x = np.zeros(int(dur * sr), dtype=np.float32)
    period = 60.0 / bpm
    burst = (RNG.standard_normal(int(0.01 * sr)) * np.exp(-np.arange(int(0.01 * sr)) / (0.002 * sr))).astype(np.float32)
    for i, t0 in enumerate(np.arange(0.1, dur - 0.05, period)):
        a = int(t0 * sr)
        gain = 0.8 if accents is None else accents[i % len(accents)]
        x[a:a + len(burst)] += gain * burst[: len(x) - a]
    return x


# --- key --------------------------------------------------------------------

def test_c_major_progression_is_c_major():
    C, F, G, Am = (48, 60, 64, 67), (53, 60, 65, 69), (55, 59, 62, 67), (57, 60, 64, 69)
    prog = [C, F, G, C, Am, F, G, C]
    x = np.concatenate([tone(ch, 2.0) for ch in prog] * 3)
    r = estimate_key(x, SR)
    assert r["key"] == "C major", r
    assert r["confidence"] in ("high", "medium")
    assert r["relative"]["key"] == "A minor"


def test_a_minor_progression_is_a_minor():
    Am, Dm, E, Am2 = (45, 57, 60, 64), (50, 57, 62, 65), (52, 56, 59, 64), (45, 57, 60, 64)
    x = np.concatenate([tone(ch, 2.0) for ch in (Am, Dm, E, Am2, Am, Dm, E, Am)] * 3)
    r = estimate_key(x, SR)
    assert r["key"] == "A minor", r


def test_scale_fit_rules_out_profile_fourth_confusion():
    # A C#-aeolian pitch histogram with a heavy F#: profiles alone lean to F# minor, but D# (not D) is used
    hist = np.zeros(12)
    for pc, w in {1: 1.0, 6: 0.95, 9: 0.82, 4: 0.71, 8: 0.69, 11: 0.66, 3: 0.55, 2: 0.29, 0: 0.3, 5: 0.29, 7: 0.31, 10: 0.28}.items():
        hist[pc] = w
    c = _choose_key(hist)
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    assert c["index"] >= 12 and names[c["index"] % 12] == "C#"


def test_silence_has_no_key():
    r = estimate_key(np.zeros(SR * 5, dtype=np.float32), SR)
    assert r["key"] is None and r["confidence"] == "low"


def test_white_noise_key_is_not_confident():
    r = estimate_key((0.1 * RNG.standard_normal(SR * 12)).astype(np.float32), SR)
    assert r["key"] is None or r["confidence"] != "high", r


# --- tempo ------------------------------------------------------------------

def test_click_track_120_is_120_and_not_ambiguous():
    r = estimate_tempo(clicks(120, 30), SR)
    assert abs(r["bpm"] - 120) <= 1.5, r
    assert r["ambiguous"] is False, r
    assert r["alternatives"] == []
    assert r["confidence"] == "high"
    ibi = np.diff(r["beat_times"])
    assert abs(np.median(ibi) - 0.5) < 0.02


def test_fast_click_track_offers_half_time():
    r = estimate_tempo(clicks(200, 30), SR)
    if abs(r["bpm"] - 200) <= 5:
        assert r["ambiguous"] and any(a["relation"] == "half" and abs(a["bpm"] - 100) <= 3 for a in r["alternatives"]), r
        assert r["confidence"] != "high"
    else:
        # the prior may pull to 100 directly; then 200 must be offered as the double (off-beats are as strong)
        assert abs(r["bpm"] - 100) <= 3, r
        assert r["ambiguous"] and any(a["relation"] == "double" for a in r["alternatives"]), r


def test_alternation_flags_half_time_for_accented_beats():
    alts = _level_alternatives(120.0, offbeat_ratio=0.2, alternation_ratio=0.4)
    assert [a["relation"] for a in alts] == ["half"] and alts[0]["bpm"] == 60.0
    assert _level_alternatives(120.0, offbeat_ratio=0.2, alternation_ratio=0.95) == []
    # Forbidden Fruit's old 199 reading: fast → half offered even with even beats
    assert any(a["relation"] == "half" and abs(a["bpm"] - 99.5) < 0.1 for a in _level_alternatives(199.0, 0.3, 0.9))
    # double offered only when off-beats sound and the double is tappable
    assert [a["relation"] for a in _level_alternatives(80.0, 0.9, 0.9)] == ["double"]
    assert _level_alternatives(117.5, 0.9, 0.9) == []


def test_silence_has_no_tempo():
    r = estimate_tempo(np.zeros(SR * 10, dtype=np.float32), SR)
    assert r["bpm"] is None and r["beat_times"] == []


# --- sections ---------------------------------------------------------------

def test_loud_section_change_is_a_boundary():
    quiet = np.concatenate([tone((60, 64, 67), 2.0, amp=0.05) for _ in range(10)])
    loud = np.concatenate([tone((53, 57, 60), 2.0, amp=0.5) + (0.15 * RNG.standard_normal(2 * SR)).astype(np.float32)
                           for _ in range(10)])
    x = np.concatenate([quiet, loud])
    secs = detect_sections(x, SR)
    strong = [s for s in secs if s["confidence"] != "low"]
    assert any(abs(s["t"] - 20.0) <= 1.5 for s in strong), secs
    assert all(abs(s["t"] - 20.0) <= 1.5 for s in strong), secs


def test_steady_signal_has_no_confident_sections():
    x = np.concatenate([tone((60, 64, 67), 2.0) for _ in range(20)])
    assert [s for s in detect_sections(x, SR) if s["confidence"] != "low"] == []


# --- energy + whole -----------------------------------------------------------

def test_loudest_moment_found():
    x = np.concatenate([tone((60,), 10, amp=0.05), tone((60,), 5, amp=0.5), tone((60,), 10, amp=0.05)])
    e = estimate_energy(x, SR)
    assert 10.0 <= e["loudest_t"] <= 15.0 and e["confidence"] == "high", e


def test_analyze_harmony_shape_and_duration():
    x = np.concatenate([tone((48, 60, 64, 67), 2.0) for _ in range(6)]) + clicks(120, 12)
    r = analyze_harmony(x, SR)
    assert set(r) >= {"key", "tempo", "sections", "energy", "duration_s"}
    assert r["duration_s"] == pytest.approx(12.0, abs=0.01)
    assert set(r["key"]) >= {"key", "confidence"}
    assert set(r["tempo"]) >= {"bpm", "confidence", "beat_times", "ambiguous", "alternatives"}
    assert set(r["energy"]) >= {"loudest_t", "loudest_db"}
    assert all(isinstance(t, float) for t in r["sections"])


def test_analyze_harmony_resamples_other_rates():
    x = tone((60, 64, 67), 6.0, sr=44100)
    r = analyze_harmony(x, 44100)
    assert r["duration_s"] == pytest.approx(6.0, abs=0.01)


def test_silence_whole():
    r = analyze_harmony(np.zeros(SR * 6, dtype=np.float32), SR)
    assert r["key"]["key"] is None and r["tempo"]["bpm"] is None and r["sections"] == []
