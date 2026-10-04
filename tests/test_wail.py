# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Tests for escutario.wail — synthetic voices with known ground truth."""

from __future__ import annotations

import re

import numpy as np
import pytest

from escutario import wail
from escutario.types import PitchTrack
from escutario.wail import analyze_voice, format_voice_section

SR = 22050
HOP = 0.01


# ---------------------------------------------------------------------------
# synthesis helpers
# ---------------------------------------------------------------------------

def cents_to_hz(c):
    c = np.asarray(c, dtype=float)
    return np.where(c > 0, 440.0 * 2 ** ((c - 6900.0) / 1200.0), 0.0)


def synth(f0, sr=SR, tilt=-12.0, amp=None, env=None, sub_db=None):
    """Harmonic voice-like tone following a per-sample f0 (0 = silence).

    tilt: harmonic roll-off in dB/octave. amp: per-sample (or scalar) peak amplitude.
    env: callable(freq_hz array) -> linear gain (formant shaping).
    sub_db: add a component at f0/2 this many dB re H1 (subharmonic).
    """
    f0 = np.asarray(f0, dtype=float)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    y = np.zeros_like(f0)
    for k in range(1, 120):
        fk = k * f0
        m = (fk < 0.45 * sr) & (f0 > 0)
        if not m.any():
            break
        a = 10 ** (tilt * np.log2(k) / 20)
        g = env(fk) if env is not None else 1.0
        y += a * g * m * np.sin(k * phase)
    if sub_db is not None:
        y += 10 ** (sub_db / 20) * (f0 > 0) * np.sin(0.5 * phase)
    y /= max(np.max(np.abs(y)), 1e-9)
    if amp is not None:
        y *= amp
    return y


def track(f0, sr=SR, conf=0.95):
    f0 = np.asarray(f0, dtype=float)
    t = np.arange(0, f0.size / sr, HOP)
    idx = np.minimum(np.round(t * sr).astype(int), f0.size - 1)
    f = f0[idx]
    return PitchTrack(times=t, f0_hz=f, confidence=np.where(f > 0, conf, 0.0), hop_s=HOP)


def seg(cents, seconds, sr=SR):
    return np.full(int(round(seconds * sr)), float(cents))


def glide(c0, c1, seconds, sr=SR):
    return np.linspace(c0, c1, int(round(seconds * sr)))


def vibrato(center, seconds, rate=6.0, extent=80.0, phase=0.0, sr=SR):
    t = np.arange(int(round(seconds * sr))) / sr
    return center + extent * np.sin(2 * np.pi * rate * t + phase)


def silence(seconds, sr=SR):
    return np.zeros(int(round(seconds * sr)))


def run(cents_contour, sr=SR, **kw):
    f0 = cents_to_hz(cents_contour)
    x = synth(f0, sr, **kw).astype(np.float32)
    return analyze_voice(x, sr, track(f0, sr))


def add_noise(x, snr_db, rng, ref_rms=None):
    n = rng.standard_normal(x.size)
    ref = ref_rms if ref_rms is not None else np.sqrt(np.mean(x[x != 0] ** 2))
    n *= ref / np.sqrt(np.mean(n ** 2)) / 10 ** (snr_db / 20)
    return n


A4, E5, A3 = 6900.0, 7600.0, 5700.0


# ---------------------------------------------------------------------------
# spectral tilt, ring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sr", [22050, 44100])
def test_alpha_ratio_orders_with_tilt_and_h1h2_matches(sr):
    contour = seg(5700 + 400, 2.0, sr)  # ~C#4 region, 277 Hz
    res = {t: run(contour, sr, tilt=t) for t in (-6.0, -12.0, -18.0)}
    alphas = [res[t]["tilt"]["alpha_ratio_db"] for t in (-6.0, -12.0, -18.0)]
    assert alphas[0] > alphas[1] > alphas[2]
    for t in (-6.0, -12.0, -18.0):
        tilt = res[t]["tilt"]
        assert tilt["confidence"] in ("high", "medium", "low")
        assert tilt["h1_h2_db"] == pytest.approx(-t, abs=1.5)
        assert tilt["h1_h2_confidence"] is not None


def test_ring_rises_with_3khz_resonance():
    contour = seg(A3, 2.0)
    boost = 10 ** (15 / 20) - 1
    env = lambda fk: 1 + boost * np.exp(-(((fk - 3000.0) / 400.0) ** 2))  # 2.5–3.5 kHz resonance
    plain = run(contour)["ring"]
    ringed = run(contour, env=env)["ring"]
    assert ringed["overall_db"] > plain["overall_db"] + 3.0
    assert ringed["per_phrase"] and ringed["per_phrase"][0]["ring_db"] == pytest.approx(ringed["overall_db"], abs=0.5)


def test_ring_label_tied_to_constants():
    contour = seg(A3, 2.0)
    r = run(contour, tilt=-3.0)["ring"]
    expected = ("strong" if r["overall_db"] >= wail.RING_STRONG_DB
                else "weak" if r["overall_db"] <= wail.RING_WEAK_DB else "moderate")
    assert r["label"] == expected


# ---------------------------------------------------------------------------
# HNR, grit vs breathy, subharmonics
# ---------------------------------------------------------------------------

def test_hnr_orders_with_snr():
    rng = np.random.default_rng(1)
    f0 = cents_to_hz(seg(A3, 2.0))
    clean = synth(f0)
    hnrs = []
    for snr in (25.0, 12.0, 3.0):
        x = (clean + add_noise(clean, snr, rng)).astype(np.float32)
        g = analyze_voice(x, SR, track(f0))["grit"]
        assert g is not None and g["confidence"] is not None
        hnrs.append(g["hnr_median_db"])
    assert hnrs[0] > hnrs[1] > hnrs[2]
    assert hnrs[1] == pytest.approx(12.0, abs=3.0)
    assert hnrs[2] == pytest.approx(3.0, abs=3.0)


def test_grit_when_loud_breathy_when_quiet():
    rng = np.random.default_rng(2)
    layout = [  # (seconds, amplitude, snr or None)
        (1.0, 0.5, None),
        (1.0, 1.0, 3.0),   # loud and noisy → grit
        (1.0, 0.5, None),
        (1.0, 0.1, 3.0),   # quiet and noisy → breathy
    ]
    x = []
    f0_all = []
    for i, (sec, amp, snr) in enumerate(layout):
        f0 = cents_to_hz(seg(A3 + 100 * i, sec))
        tone = synth(f0) * amp
        if snr is not None:
            tone = tone + add_noise(tone, snr, rng)
        x.append(tone)
        f0_all.append(f0)
        x.append(silence(0.3))
        f0_all.append(silence(0.3))
    x = np.concatenate(x).astype(np.float32)
    f0 = np.concatenate(f0_all)
    g = analyze_voice(x, SR, track(f0))["grit"]
    kinds = [(r["kind"], r["start_s"], r["end_s"]) for r in g["regions"]]
    grit = [k for k in kinds if k[0] == "grit"]
    breathy = [k for k in kinds if k[0] == "breathy"]
    assert len(grit) == 1 and 1.2 <= grit[0][1] and grit[0][2] <= 2.4
    assert len(breathy) == 1 and 3.8 <= breathy[0][1] and breathy[0][2] <= 5.0
    assert all(r["hnr_db"] < wail.HNR_LOW_DB for r in g["regions"])
    text = format_voice_section(analyze_voice(x, SR, track(f0)))
    assert "rough while loud" in text and "breathy" in text


def test_grit_regions_respect_minimum_duration_and_card_shows_tenths():
    rng = np.random.default_rng(4)
    f0 = cents_to_hz(seg(A3, 4.0))
    x = synth(f0)
    for start, dur in ((0.5, 0.06), (1.2, 0.12), (1.9, 0.16), (2.8, 0.4)):  # only the 0.4 s burst is long enough
        a, b = int(start * SR), int((start + dur) * SR)
        x[a:b] += add_noise(x[a:b], -3.0, rng)
    res = analyze_voice(x.astype(np.float32), SR, track(f0))
    regions = res["grit"]["regions"]
    assert len(regions) == 1
    r = regions[0]
    assert 2.7 <= r["start_s"] and r["end_s"] <= 3.3
    for reg in regions:
        assert reg["covered_s"] >= wail.GRIT_MIN_REGION_S - 1e-6
        assert reg["end_s"] - reg["start_s"] >= wail.GRIT_MIN_REGION_S - 0.011
    text = format_voice_section(res)
    m = re.search(r"at (\d+:\d\d\.\d)–(\d+:\d\d\.\d)", text)
    assert m and m.group(1) != m.group(2)
    assert wail._fmt_time_precise(100.17) == "1:40.2" and wail._fmt_time_precise(100.61) == "1:40.6"


def test_subharmonic_indicator():
    f0 = cents_to_hz(seg(A4, 2.0))
    clean = analyze_voice(synth(f0).astype(np.float32), SR, track(f0))["grit"]
    doubled = analyze_voice(synth(f0, sub_db=-8.0).astype(np.float32), SR, track(f0))["grit"]
    assert clean["subharmonics_present"] is False
    assert doubled["subharmonics_present"] is True
    assert doubled["subharmonic_db"] == pytest.approx(-8.0, abs=3.0)


# ---------------------------------------------------------------------------
# cry falls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("drop,dur", [(300.0, 0.4), (200.0, 0.25), (500.0, 0.5), (180.0, 0.15), (300.0, 0.1)])
def test_fall_detected_with_drop_within_50_cents(drop, dur):
    contour = np.concatenate([seg(E5, 1.0), glide(E5, E5 - drop, dur), silence(0.4)])
    falls = run(contour)["falls"]
    assert len(falls) == 1
    f = falls[0]
    assert f["drop_cents"] == pytest.approx(drop, abs=50)
    assert f["start_note"] == "E5"
    assert f["duration_s"] == pytest.approx(dur, abs=0.12)
    assert f["start_s"] == pytest.approx(1.0, abs=0.12)


def test_fall_after_vibrato_plateau_and_at_44k():
    contour = np.concatenate([vibrato(E5, 1.2, rate=5.5, extent=60, sr=44100),
                              glide(E5, E5 - 320, 0.4, 44100), silence(0.3, 44100)])
    falls = run(contour, sr=44100)["falls"]
    assert len(falls) == 1
    assert falls[0]["drop_cents"] == pytest.approx(320, abs=50)


def test_multiple_phrases_each_fall_counted():
    parts = []
    for d in (250.0, 400.0, 180.0):
        parts += [seg(A4, 0.8), glide(A4, A4 - d, 0.3), silence(0.5)]
    falls = run(np.concatenate(parts))["falls"]
    assert [f["drop_cents"] for f in falls] == pytest.approx([250, 400, 180], abs=50)


def test_slow_drift_is_not_a_cry_fall():
    contour = np.concatenate([seg(E5, 1.0), glide(E5, E5 - 300, 1.5), silence(0.3)])
    assert run(contour)["falls"] == []


def test_small_fall_below_threshold_ignored():
    contour = np.concatenate([seg(E5, 1.0), glide(E5, E5 - 80, 0.3), silence(0.3)])
    assert run(contour)["falls"] == []


# ---------------------------------------------------------------------------
# register breaks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c0,c1,direction", [(E5, E5 - 500, "down"), (A3, A3 + 700, "up"), (A4, A4 - 900, "down")])
def test_flip_detected_as_break(c0, c1, direction):
    contour = np.concatenate([seg(c0, 1.0), seg(c1, 1.0)])
    res = run(contour)
    assert len(res["breaks"]) == 1
    b = res["breaks"][0]
    assert b["direction"] == direction
    assert b["jump_cents"] == pytest.approx(c1 - c0, abs=30)
    assert b["time_s"] == pytest.approx(1.0, abs=0.03)
    assert res["falls"] == []
    assert res["voice_changes"]["count"] == 0


def test_flip_at_phrase_end_is_break_not_fall():
    contour = np.concatenate([seg(A4, 1.0), seg(A4 - 600, 0.3), silence(0.3)])
    res = run(contour)
    assert len(res["breaks"]) == 1
    assert res["falls"] == []


# ---------------------------------------------------------------------------
# octave slips, voice changes, fall caps (regressions from real Demucs stems)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("jump,reason", [(-1200, "octave_multiple"), (2400, "octave_multiple"),
                                         (-1150, "octave_multiple"), (1900, "over_max"), (3600, "octave_multiple")])
def test_big_jumps_are_voice_changes_not_flips(jump, reason):
    contour = np.concatenate([seg(A4, 1.0), seg(A4 + jump, 1.0)])
    res = run(contour)
    assert res["breaks"] == []
    vc = res["voice_changes"]
    assert vc["count"] == 1 and vc["events"][0]["reason"] == reason
    assert res["falls"] == []
    text = format_voice_section(res)
    assert "BREAKS : none detected" in text and "SLIPS  : 1 octave slips" in text


def test_octave_slip_on_sustained_note_not_a_flip():
    f0 = cents_to_hz(seg(E5, 2.0))
    x = synth(f0).astype(np.float32)
    tr = track(f0)
    tr.f0_hz[80:95] /= 2.0   # tracker drops an octave for 150 ms, then recovers
    tr.f0_hz[140:150] *= 4.0  # and jumps two octaves for 100 ms
    res = analyze_voice(x, SR, tr)
    assert res["breaks"] == []
    assert res["falls"] == []
    assert res["voice_changes"]["count"] == 4
    assert res["voice_changes"]["octave_multiple"] == 4


def test_octave_slip_at_phrase_end_rejected_as_fall_and_counted():
    # held octave drop at the end (caught by the jump detector)
    held = np.concatenate([seg(A4, 1.0), seg(A4 - 1200, 0.12), silence(0.3)])
    # fast octave glide straight into silence (no hold after: only the fall rule can catch it)
    # (lands for 30 ms: shorter than BREAK_HOLD_S, so the jump detector cannot confirm it)
    fast = np.concatenate([seg(A4, 1.0), glide(A4, A4 - 1200, 0.12), seg(A4 - 1200, 0.03), silence(0.3)])
    for contour in (held, fast):
        res = run(contour)
        assert res["falls"] == []
        assert res["breaks"] == []
        assert res["falls_rejected"]["octave_slip"] == 1
        assert any("octave slip" in n for n in res["notes"])


def test_fall_deeper_than_cap_rejected_and_counted():
    deep = run(np.concatenate([seg(E5, 1.0), glide(E5, E5 - 1000, 0.45), silence(0.3)]))
    assert deep["falls"] == []
    assert deep["falls_rejected"] == {"octave_slip": 0, "over_max": 1}
    assert any(f"deeper than {wail.FALL_MAX_CENTS:g}" in n for n in deep["notes"])
    ok = run(np.concatenate([seg(E5, 1.0), glide(E5, E5 - 800, 0.45), silence(0.3)]))
    assert len(ok["falls"]) == 1 and ok["falls"][0]["drop_cents"] == pytest.approx(800, abs=50)
    assert ok["falls_rejected"] == {"octave_slip": 0, "over_max": 0}


def test_slow_octave_glide_is_over_max_not_octave_slip():
    res = run(np.concatenate([seg(E5, 1.0), glide(E5, E5 - 1200, 0.5), silence(0.3)]))
    assert res["falls"] == []
    assert res["falls_rejected"]["over_max"] == 1 and res["falls_rejected"]["octave_slip"] == 0


def test_two_voice_stem_handoffs_are_voice_changes_not_flips():
    """Male ~G2–C4 alternating with female ~C4–E5 without gaps, plus an overlap burst."""
    # melodic steps inside each voice stay ≤ 300 ¢ so the only big jumps are the handoffs
    G2, As2, C3, E3, G3 = 4300.0, 4600.0, 4800.0, 5200.0, 5500.0
    A4_, C5, D5 = 6900.0, 7200.0, 7400.0
    male = lambda: np.concatenate([seg(G2, 0.4), seg(As2, 0.4), seg(C3, 0.4)])
    female = lambda: np.concatenate([seg(A4_, 0.4), seg(C5, 0.4), seg(D5, 0.4)])
    burst = np.concatenate([np.concatenate([seg(E3, 0.1), seg(D5, 0.1)]) for _ in range(4)]
                           + [seg(D5 - 600, 0.1), seg(D5, 0.1)])  # a ±600 ¢ wobble inside the burst
    contour = np.concatenate([
        male(), female(),            # C3 → A4 handoff (2100 ¢)
        male(),                      # D5 → G2 handoff (3100 ¢)
        silence(0.4), seg(G3, 0.8), seg(G3 + 500, 0.8), silence(0.4),  # one genuine 500 ¢ flip, far from any slip
        female(), burst, male(), silence(0.3),
    ])
    res = run(contour)
    assert len(res["breaks"]) == 1 and res["breaks"][0]["jump_cents"] == pytest.approx(500, abs=30)
    vc = res["voice_changes"]
    assert vc["count"] >= 10
    assert vc["near_slip"] >= 1
    assert all(abs(e["jump_cents"]) >= wail.VOICE_CHANGE_MIN_CENTS or e["reason"] != "over_max"
               for e in vc["events"])
    assert res["falls"] == []
    text = format_voice_section(res)
    assert "BREAKS : 1 register flip " in text
    assert "SLIPS  :" in text and "more than one voice" in text
    assert any("more than one voice" in n for n in res["notes"])
    assert res["multi_voice_suspected"] is True
    assert res["register"]["confidence"] == "low"
    assert res["ring"]["top_third_confidence"] in (None, "low")


def test_few_slips_do_not_flag_multi_voice():
    res = run(np.concatenate([seg(A4, 2.0), seg(A4 - 1200, 2.0), seg(A3 + 300, 2.0)]))
    assert res["voice_changes"]["count"] == 1
    assert res["multi_voice_suspected"] is False


def test_gap_between_notes_is_not_a_break():
    contour = np.concatenate([seg(A4, 1.0), silence(0.1), seg(A4 - 1200, 1.0)])
    assert run(contour)["breaks"] == []


def test_top_third_ring_and_alpha_isolate_high_notes():
    boost = 10 ** (15 / 20) - 1
    env = lambda fk: 1 + boost * np.exp(-(((fk - 3000.0) / 400.0) ** 2))
    low = synth(cents_to_hz(seg(A3, 3.0)))
    high = synth(cents_to_hz(seg(E5, 2.0)), env=env)
    x = np.concatenate([low, silence(0.3), high]).astype(np.float32)
    f0 = np.concatenate([cents_to_hz(seg(A3, 3.0)), silence(0.3), cents_to_hz(seg(E5, 2.0))])
    res = analyze_voice(x, SR, track(f0))
    ring, tilt = res["ring"], res["tilt"]
    assert ring["top_third_db"] is not None and ring["top_third_confidence"] is not None
    assert ring["top_third_db"] > ring["overall_db"] + 2.0
    assert tilt["alpha_top_third_db"] is not None and tilt["alpha_top_third_confidence"] is not None
    assert "TOP3RD : on top-third notes ring" in format_voice_section(res)
    steady = run(seg(A4, 2.0))
    assert steady["ring"]["top_third_db"] is None and steady["tilt"]["alpha_top_third_db"] is None


def test_single_frame_octave_error_is_not_a_break():
    f0 = cents_to_hz(seg(A4, 2.0))
    x = synth(f0).astype(np.float32)
    tr = track(f0)
    tr.f0_hz[100] *= 2.0
    tr.f0_hz[150] /= 2.0
    assert analyze_voice(x, SR, tr)["breaks"] == []


def test_fast_glide_under_400_cents_is_not_a_break():
    contour = np.concatenate([seg(A4, 1.0), glide(A4, A4 + 300, 0.05), seg(A4 + 300, 1.0)])
    assert run(contour)["breaks"] == []


# ---------------------------------------------------------------------------
# false positives: vibrato, steady tone, silence, noise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sr", [22050, 44100])
def test_vibrato_gives_no_falls_no_breaks(sr):
    parts = []
    # different phrase lengths so each phrase ends at a different vibrato phase
    for sec in (3.0, 1.37, 1.52, 1.61, 1.09):
        parts += [vibrato(E5, sec, rate=6.0, extent=80.0, sr=sr), silence(0.4, sr)]
    res = run(np.concatenate(parts), sr)
    assert res["falls"] == []
    assert res["breaks"] == []


@pytest.mark.parametrize("rate", [4.0, 5.0, 7.0, 8.0])
def test_vibrato_rates_give_no_falls(rate):
    parts = []
    for sec, ph in ((1.3, 0.0), (1.45, 1.5), (1.2, 3.0), (1.55, 4.5)):
        parts += [vibrato(A4, sec, rate=rate, extent=100.0, phase=ph), silence(0.4)]
    res = run(np.concatenate(parts))
    assert res["falls"] == []
    assert res["breaks"] == []


def test_jittered_vibrato_no_false_events_and_falls_still_found():
    """Tracker-like frame jitter (±12 ¢ sd) on random vibrato; seeded."""
    rng = np.random.default_rng(7)

    def jitter(tr):
        v = tr.f0_hz > 0
        tr.f0_hz[v] *= 2 ** (rng.normal(0, 12, v.sum()) / 1200)
        return tr

    parts = []
    for _ in range(12):
        parts += [vibrato(rng.uniform(5500, 7800), rng.uniform(0.8, 2.5), rate=rng.uniform(4, 8),
                          extent=rng.uniform(50, 100), phase=rng.uniform(0, 6.28)), silence(0.3)]
    f0 = cents_to_hz(np.concatenate(parts))
    res = analyze_voice(synth(f0).astype(np.float32), SR, jitter(track(f0)))
    assert res["falls"] == [] and res["breaks"] == []

    for _ in range(6):
        drop, dur = rng.uniform(200, 600), rng.uniform(0.12, 0.55)
        contour = np.concatenate([vibrato(7000, 1.2, rate=rng.uniform(4.5, 7), extent=70, phase=rng.uniform(0, 6.28)),
                                  glide(7000, 7000 - drop, dur), silence(0.3)])
        f0 = cents_to_hz(contour)
        falls = analyze_voice(synth(f0).astype(np.float32), SR, jitter(track(f0)))["falls"]
        assert len(falls) == 1 and falls[0]["drop_cents"] == pytest.approx(drop, abs=50)


def test_steady_tone_no_falls_no_breaks_no_level_correlation():
    res = run(seg(A4, 3.0))
    assert res["falls"] == []
    assert res["breaks"] == []
    assert res["level_with_pitch"] is None
    reg = res["register"]
    assert reg["low_note"] == reg["high_note"] == "A4"
    assert reg["top_third_share"] is None


def test_silence_and_noise_do_not_crash_and_report_nothing():
    rng = np.random.default_rng(3)
    n = int(2 * SR)
    empty = PitchTrack(times=np.arange(0, 2, HOP), f0_hz=np.zeros(200), confidence=np.zeros(200), hop_s=HOP)
    for x in (np.zeros(n, np.float32), (0.1 * rng.standard_normal(n)).astype(np.float32)):
        res = analyze_voice(x, SR, empty)
        assert res["register"] is None and res["ring"] is None and res["tilt"] is None
        assert res["falls"] == [] and res["breaks"] == [] and res["grit"] is None
        assert res["level_with_pitch"] is None
        text = format_voice_section(res)
        assert text.startswith("VOICE")
    res = analyze_voice(np.zeros(0, np.float32), SR,
                        PitchTrack(times=np.zeros(0), f0_hz=np.zeros(0), confidence=np.zeros(0), hop_s=HOP))
    assert res["falls"] == []


def test_low_confidence_frames_are_ignored():
    f0 = cents_to_hz(seg(A4, 2.0))
    x = synth(f0).astype(np.float32)
    res = analyze_voice(x, SR, track(f0, conf=0.2))
    assert res["register"] is None and res["ring"] is None


# ---------------------------------------------------------------------------
# level with pitch, register position
# ---------------------------------------------------------------------------

def _stepped(notes_cents, level_db, sec=0.5):
    contour, amp = [], []
    for c, db in zip(notes_cents, level_db):
        contour.append(seg(c, sec))
        amp.append(np.full(int(round(sec * SR)), 10 ** (db / 20)))
    return np.concatenate(contour), np.concatenate(amp)


def test_level_rising_with_pitch_positive_correlation():
    notes = [A3 + 200 * i for i in range(8)]
    contour, amp = _stepped(notes, [-24 + 3 * i for i in range(8)])
    f0 = cents_to_hz(contour)
    lwp = analyze_voice(synth(f0, amp=amp).astype(np.float32), SR, track(f0))["level_with_pitch"]
    assert lwp["r"] > 0.8
    assert lwp["slope_db_per_octave"] == pytest.approx(18.0, abs=4.0)
    assert lwp["confidence"] in ("high", "medium", "low")


def test_level_dropping_with_pitch_negative_correlation():
    notes = [A3 + 200 * i for i in range(8)]
    contour, amp = _stepped(notes, [-3 * i for i in range(8)])
    f0 = cents_to_hz(contour)
    lwp = analyze_voice(synth(f0, amp=amp).astype(np.float32), SR, track(f0))["level_with_pitch"]
    assert lwp["r"] < -0.8


def test_register_range_top_third_and_longest_top_note():
    # A3 … E5 with a 3 s held E5 in the middle
    contour = np.concatenate([seg(A3, 2.0), seg(A3 + 700, 2.0), seg(A4, 2.0), seg(E5, 3.0),
                              seg(A4, 1.0), silence(0.3), seg(E5, 1.0), seg(A3, 1.0)])
    reg = run(contour)["register"]
    assert reg["low_note"] == "A3" and reg["high_note"] == "E5"
    # top third of 1900 ¢ starts ~633 ¢ below E5 → only the E5 notes (4 s of 12 s)
    assert reg["top_third_share"] == pytest.approx(4.0 / 12.0, abs=0.04)
    ln = reg["longest_top_note"]
    assert ln["note"] == "E5"
    assert ln["duration_s"] == pytest.approx(3.0, abs=0.2)
    assert ln["start_s"] == pytest.approx(6.0, abs=0.2)


def test_vibrato_does_not_split_notes():
    contour = np.concatenate([seg(A3, 2.0), vibrato(E5, 3.0, rate=5.0, extent=90.0)])
    ln = run(contour)["register"]["longest_top_note"]
    assert ln["note"] == "E5" and ln["duration_s"] >= 2.5


# ---------------------------------------------------------------------------
# card
# ---------------------------------------------------------------------------

MOOD_WORDS = re.compile(r"\b(sad|angry|emotional|happy|joy|pain|anguish|longing|melanchol\w*|wail\w*)\b", re.I)


def test_format_voice_section_lines_and_no_mood_words():
    contour = np.concatenate([seg(A3, 1.5), silence(0.3), seg(A4, 1.5), glide(A4, E5, 0.2), seg(E5, 2.0),
                              glide(E5, E5 - 320, 0.4), silence(0.4), seg(A4, 1.0), seg(A4 - 600, 1.0)])
    res = run(contour)
    text = format_voice_section(res)
    for tag in ("VOICE  :", "RING   :", "TOP3RD :", "FALLS  :", "BREAKS :", "GRIT   :"):
        assert tag in text
    assert "range A3–E5" in text
    assert "1 cry-fall (e.g. E5 ↓" in text
    assert "1 register flip" in text
    assert not MOOD_WORDS.search(text)


def test_every_reported_measure_carries_confidence():
    contour = np.concatenate([seg(A3, 1.5), silence(0.3), seg(A4, 1.5), glide(A4, E5, 0.2), seg(E5, 2.0),
                              glide(E5, E5 - 320, 0.4), silence(0.4), seg(A4, 1.0), seg(A4 - 600, 1.0)])
    res = run(contour)
    for key in ("register", "ring", "tilt", "grit", "level_with_pitch"):
        assert res[key] is None or res[key]["confidence"] in ("high", "medium", "low")
    for item in res["falls"] + res["breaks"] + res["grit"]["regions"] + res["ring"]["per_phrase"]:
        assert item["confidence"] in ("high", "medium", "low")
