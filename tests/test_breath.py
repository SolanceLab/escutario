# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Tests for the breath organ — synthetic stems with known ground truth."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fixtures"))

from breath_synth import floor_noise, hihat_click, add_at, song, tone, truth_pitch_track  # noqa: E402
from escutario.breath import (  # noqa: E402
    NOTE_BREATHS_EDITED,
    detect_breaths,
    format_breath_section,
)
from escutario.types import PitchTrack  # noqa: E402

MOOD_WORDS = ("sad", "angry", "emotional", "happy", "joy", "anxious", "tender", "passionate", "longing")
TIME_TOL_S = 0.060
CONF = {"high", "medium", "low"}


def _matched(found: list[dict], truth: list[tuple[float, float]], tol: float = TIME_TOL_S) -> int:
    n = 0
    for s, e in truth:
        if any(abs(b["start_s"] - s) <= tol and abs(b["end_s"] - e) <= tol for b in found):
            n += 1
    return n


def _unvoiced_track(duration_s: float, hop_s: float = 0.01) -> PitchTrack:
    n = int(np.ceil(duration_s / hop_s))
    return PitchTrack(times=(np.arange(n) + 0.5) * hop_s, f0_hz=np.zeros(n), confidence=np.zeros(n), hop_s=hop_s)


def crude_pitch_track(y: np.ndarray, sr: int, hop_s: float = 0.01, frame_s: float = 0.04,
                      fmin: float = 70.0, fmax: float = 1000.0) -> PitchTrack:
    """A deliberately crude voicing/f0 estimate straight from the signal (normalised ACF peak)."""
    win, hop = int(frame_s * sr), int(hop_s * sr)
    n = 1 + (len(y) - win) // hop
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    frames = y[idx].astype(np.float64)
    frames = frames - frames.mean(axis=1, keepdims=True)
    energy = np.sum(frames ** 2, axis=1)
    spec = np.fft.rfft(frames * np.hanning(win), n=2 * win, axis=1)
    acf = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :win]
    acf = acf / (acf[:, :1] + 1e-12)
    lo, hi = int(sr / fmax), int(sr / fmin)
    lag = lo + np.argmax(acf[:, lo:hi], axis=1)
    peak = acf[np.arange(n), lag]
    rms_db = 10 * np.log10(energy / win + 1e-12)
    voiced = (peak >= 0.5) & (rms_db >= rms_db.max() - 30.0)
    f0 = np.where(voiced, sr / lag, 0.0)
    times = (np.arange(n) * hop + win / 2) / sr
    return PitchTrack(times=times, f0_hz=f0, confidence=np.where(voiced, peak, 0.0), hop_s=hop_s)


# ---------------------------------------------------------------------------
# True breaths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sr", [22050, 44100])
def test_band_passed_breaths_are_found_with_accurate_times(sr):
    y, truth = song(sr=sr)
    r = detect_breaths(y, sr, truth_pitch_track(truth))
    assert r["count"] == len(truth["breaths"]) == len(r["breaths"])
    assert _matched(r["breaths"], truth["breaths"]) == len(truth["breaths"])
    assert all(b["confidence"] == "high" for b in r["breaths"])
    for b in r["breaths"]:
        assert -26.0 <= b["level_db_rel_phrase"] <= -18.0  # synthesised at -22 dB
        assert b["duration_s"] == pytest.approx(b["end_s"] - b["start_s"], abs=0.002)
    assert len(r["phrases"]) == len(truth["phrases"])
    assert all(p["preceded_by_breath"] for p in r["phrases"])
    assert r["silent_gaps"] == []
    assert r["longest_phrase_s"] == pytest.approx(3.2, abs=0.05)
    assert r["leakage_period_s"] is None


def test_breath_with_low_mid_colouring_is_still_a_breath():
    # Real inhales often sit lower (300 Hz-3 kHz); a band-limited shape must not read as tonal.
    y, truth = song(breath_band=(300.0, 3000.0))
    r = detect_breaths(y, 22050, truth_pitch_track(truth))
    assert _matched(r["breaths"], truth["breaths"]) == len(truth["breaths"])


def test_breath_times_do_not_overlap_sung_tone():
    y, truth = song()
    r = detect_breaths(y, 22050, truth_pitch_track(truth))
    for b in r["breaths"]:
        for s, e, _ in truth["phrases"]:
            assert b["end_s"] <= s + 0.02 or b["start_s"] >= e - 0.02


# ---------------------------------------------------------------------------
# Silence, loud noise, residue, off-band noise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("digital", [False, True])
def test_pure_silence_gaps_are_silent_gaps_not_breaths(digital):
    y, truth = song(gap_kinds=("silence",) * 4, lead_breath_s=None, digital_silence_gaps=digital)
    r = detect_breaths(y, 22050, truth_pitch_track(truth))
    assert r["breaths"] == [] and r["count"] == 0
    assert len(r["silent_gaps"]) == len(truth["silent_gaps"]) == 4
    for g, (s, e) in zip(r["silent_gaps"], truth["silent_gaps"]):
        assert abs(g["start_s"] - s) <= TIME_TOL_S and abs(g["end_s"] - e) <= TIME_TOL_S
    assert not any(p["preceded_by_breath"] for p in r["phrases"])
    assert NOTE_BREATHS_EDITED in r["notes"]


def test_mixed_gaps_split_into_breaths_and_silent_gaps():
    y, truth = song(gap_kinds=("breath", "silence", "breath", "silence"))
    r = detect_breaths(y, 22050, truth_pitch_track(truth))
    assert _matched(r["breaths"], truth["breaths"]) == len(truth["breaths"]) == r["count"]
    assert len(r["silent_gaps"]) == 2
    assert [p["preceded_by_breath"] for p in r["phrases"]] == [True, True, False, True, False]


def test_long_instrumental_break_is_not_a_silent_gap_but_its_breath_is_found():
    sr = 22050
    y, truth = song(phrase_durs=(2.0, 2.0), gap_kinds=("silence",), gap_s=12.0, lead_breath_s=None)
    # the singer inhales just before coming back in after the break
    from breath_synth import breath_burst
    rng = np.random.default_rng(5)
    ref = float(np.sqrt(np.mean(tone(1.0, sr, 220.0) ** 2)))
    g0, g1 = truth["gaps"][0]
    b = breath_burst(0.5, sr, ref * 10 ** (-22 / 20), rng).astype(np.float32)
    add_at(y, b, g1 - 0.6, sr)
    r = detect_breaths(y, sr, truth_pitch_track(truth))
    assert r["silent_gaps"] == []
    assert _matched(r["breaths"], [(g1 - 0.6, g1 - 0.1)]) == 1
    assert r["phrases"][1]["preceded_by_breath"] is True


def test_noise_burst_as_loud_as_singing_is_not_a_breath():
    y, truth = song(gap_kinds=("loud",) * 4, lead_breath_s=None)
    r = detect_breaths(y, 22050, truth_pitch_track(truth))
    assert r["count"] == 0
    assert r["silent_gaps"] == []  # there was sound; it just wasn't a breath
    assert any("did not read as a breath" in n for n in r["notes"])


def test_faint_residue_far_below_singing_is_omitted():
    y, truth = song(breath_rel_db=-55.0)
    r = detect_breaths(y, 22050, truth_pitch_track(truth))
    assert r["count"] == 0


@pytest.mark.parametrize("band", [(6500.0, 10500.0), (100.0, 450.0)])
def test_off_band_hiss_or_rumble_is_not_a_breath(band):
    y, truth = song(breath_band=band)
    r = detect_breaths(y, 22050, truth_pitch_track(truth))
    assert r["count"] == 0


def test_click_shorter_than_a_breath_is_not_a_breath():
    y, truth = song(breath_s=(0.07,) * 4, lead_breath_s=None)
    r = detect_breaths(y, 22050, truth_pitch_track(truth))
    assert r["count"] == 0


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sr", [22050, 44100])
@pytest.mark.parametrize("period", [0.25, 0.125])
@pytest.mark.parametrize("level", [-6.0, -20.0])
def test_periodic_hihat_clicks_in_gaps_are_not_breaths(sr, period, level):
    y, truth = song(sr=sr, gap_kinds=("click",) * 4, lead_breath_s=None,
                    hihat_period_s=period, hihat_peak_db_rel_phrase=level)
    r = detect_breaths(y, sr, truth_pitch_track(truth))
    assert not any(b["confidence"] == "high" for b in r["breaths"])
    assert r["count"] == 0
    assert r["silent_gaps"] == []  # clicks are sound, not an edit
    assert r["leakage_period_s"] == pytest.approx(period, rel=0.1)
    assert any("hi-hat-like leakage" in n for n in r["notes"])


@pytest.mark.parametrize("period", [0.25, 0.125])
def test_breaths_under_loud_hihat_leakage_lose_high_confidence(period):
    y, truth = song(hihat_period_s=period)
    r = detect_breaths(y, 22050, truth_pitch_track(truth))
    assert r["leakage_period_s"] is not None
    assert all(b["confidence"] != "high" for b in r["breaths"])
    # every reported breath is still inside a true breath's neighbourhood
    for b in r["breaths"]:
        assert any(b["start_s"] < e + TIME_TOL_S and b["end_s"] > s - TIME_TOL_S for s, e in truth["breaths"])


def test_irregular_transients_do_not_trigger_the_leakage_note():
    y, truth = song(gap_kinds=("silence",) * 4, lead_breath_s=None)
    sr = truth["sr"]
    rng = np.random.default_rng(3)
    ref = float(np.sqrt(np.mean(tone(1.0, sr, 220.0) ** 2)))
    for g0, g1 in truth["gaps"]:
        for t in rng.uniform(g0 + 0.05, g1 - 0.2, size=2):
            add_at(y, hihat_click(sr, ref, rng).astype(np.float32), float(t), sr)
    r = detect_breaths(y, sr, truth_pitch_track(truth))
    assert r["leakage_period_s"] is None
    assert not any("leakage" in n for n in r["notes"])
    assert r["count"] == 0


# ---------------------------------------------------------------------------
# No singing / no gaps
# ---------------------------------------------------------------------------

def test_steady_white_noise_alone_has_no_phrases_or_breaths():
    sr = 22050
    rng = np.random.default_rng(1)
    y = (rng.standard_normal(sr * 5) * 0.1).astype(np.float32)
    for track in (_unvoiced_track(5.0), crude_pitch_track(y, sr)):
        r = detect_breaths(y, sr, track)
        assert r["phrases"] == [] and r["breaths"] == [] and r["count"] == 0
        assert r["silent_gaps"] == []
        assert r["longest_phrase_s"] == 0.0
    assert format_breath_section(r).startswith("BREATH : no sung phrases")


def test_continuous_tone_has_zero_breaths():
    sr = 22050
    rng = np.random.default_rng(2)
    total = 8.6
    y = floor_noise(int(total * sr), rng)
    add_at(y, tone(8.0, sr, 220.0), 0.3, sr)
    truth = {"phrases": [(0.3, 8.3, 220.0)], "duration_s": total}
    for track in (truth_pitch_track(truth), crude_pitch_track(y.astype(np.float32), sr)):
        r = detect_breaths(y.astype(np.float32), sr, track)
        assert r["count"] == 0
        assert len(r["phrases"]) == 1
        assert r["silent_gaps"] == []
        assert r["longest_phrase_s"] == pytest.approx(8.0, abs=0.1)


def test_empty_and_tiny_input_do_not_crash():
    r = detect_breaths(np.zeros(0, dtype=np.float32), 22050, _unvoiced_track(0.0))
    assert r["count"] == 0 and r["phrases"] == []
    r = detect_breaths(np.zeros(100, dtype=np.float32), 22050, _unvoiced_track(0.01))
    assert r["count"] == 0


# ---------------------------------------------------------------------------
# Robustness to an imperfect pitch track
# ---------------------------------------------------------------------------

def test_crude_signal_voicing_still_finds_the_breaths():
    y, truth = song()
    track = crude_pitch_track(y, truth["sr"])
    r = detect_breaths(y, truth["sr"], track)
    assert len(r["phrases"]) == len(truth["phrases"])
    assert _matched(r["breaths"], truth["breaths"]) >= len(truth["breaths"]) - 1
    assert r["count"] <= len(truth["breaths"])


def test_jittered_holey_track_still_finds_the_breaths():
    y, truth = song()
    track = truth_pitch_track(truth)
    rng = np.random.default_rng(11)
    f0, conf, times = track.f0_hz.copy(), track.confidence.copy(), track.times
    # phrase edges off by up to ±30 ms
    for s, e, hz in truth["phrases"]:
        for edge in (s, e):
            shift = rng.uniform(-0.03, 0.03)
            lo, hi = sorted((edge, edge + shift))
            m = (times >= lo) & (times < hi)
            inside = (times >= s) & (times < e)
            f0[m & inside], conf[m & inside] = 0.0, 0.05
            f0[m & ~inside], conf[m & ~inside] = hz, 0.6
    # consonant-sized dropouts (80 ms, never touching each other) inside phrases
    for s, e, _ in truth["phrases"]:
        idx = np.flatnonzero((times >= s + 0.1) & (times < e - 0.2))
        for i in idx[rng.integers(0, 5):: 30][:-1]:
            f0[i:i + 8], conf[i:i + 8] = 0.0, 0.05
    # a spurious 2-frame voiced blip in the middle of each breath
    for s, e in truth["breaths"]:
        m = np.flatnonzero(times >= (s + e) / 2)[:2]
        f0[m], conf[m] = 300.0, 0.4
    noisy = PitchTrack(times=times, f0_hz=f0, confidence=conf, hop_s=track.hop_s)
    r = detect_breaths(y, truth["sr"], noisy)
    assert _matched(r["breaths"], truth["breaths"]) >= len(truth["breaths"]) - 1
    assert len(r["phrases"]) == len(truth["phrases"])


# ---------------------------------------------------------------------------
# Shape of the result and the card
# ---------------------------------------------------------------------------

def test_result_schema_and_confidence_values():
    y, truth = song(gap_kinds=("breath", "silence", "breath", "breath"))
    r = detect_breaths(y, 22050, truth_pitch_track(truth))
    for key in ("breaths", "count", "phrases", "longest_phrase_s", "silent_gaps", "notes"):
        assert key in r
    for b in r["breaths"]:
        assert set(b) == {"start_s", "end_s", "duration_s", "level_db_rel_phrase", "confidence"}
        assert b["confidence"] in CONF
        assert all(isinstance(b[k], float) for k in ("start_s", "end_s", "duration_s", "level_db_rel_phrase"))
    for p in r["phrases"]:
        assert set(p) == {"start_s", "end_s", "duration_s", "preceded_by_breath"}
        assert isinstance(p["preceded_by_breath"], bool)
    for g in r["silent_gaps"]:
        assert set(g) == {"start_s", "end_s"}
    assert any("not evidence" in n for n in r["notes"])
    assert "candidates" not in r


def test_explain_lists_measured_candidates_and_drop_reasons():
    y, truth = song(gap_kinds=("breath", "loud", "breath", "breath"))
    r = detect_breaths(y, 22050, truth_pitch_track(truth), explain=True)
    kept = [c for c in r["candidates"] if "confidence" in c]
    dropped = [c for c in r["candidates"] if "dropped" in c]
    assert len(kept) == r["count"]
    assert any(c["dropped"] == "level vs phrase" for c in dropped)
    for c in kept:
        assert {"flatness", "band_share", "above_floor_db", "peak_to_mean", "points"} <= set(c)


def test_format_card_header_breath_lines_and_notes():
    y, truth = song(gap_kinds=("breath", "silence", "breath", "silence"), breath_s=(0.7, 0.3, 0.45, 0.3))
    r = detect_breaths(y, 22050, truth_pitch_track(truth))
    card = format_breath_section(r)
    lines = card.splitlines()
    assert re.match(r"^BREATH : \d+ heard \(conf high \d+ / medium \d+\) \| longest phrase \d+\.\ds at \d+:\d\d"
                    r" \| 2 silent gaps \(likely edits\) at \d+:\d\d, \d+:\d\d$", lines[0]), lines[0]
    breath_lines = [ln for ln in lines[1:] if not ln.strip().startswith("note:")]
    assert breath_lines, card
    assert all(re.match(r"^\s+\d+:\d\d\.\d\s+\d\.\d\ds\s+[+-]\d+ dB\s+(high|medium|low)\s+\S", ln) for ln in breath_lines), card
    assert any("long" in ln for ln in breath_lines)
    assert any("not evidence" in ln for ln in lines)
    assert len(lines) <= 1 + 5 + len(r["notes"])
    lowered = card.lower()
    assert not any(re.search(rf"\b{w}\b", lowered) for w in MOOD_WORDS)


def test_format_card_when_no_breaths_heard():
    y, truth = song(gap_kinds=("silence",) * 4, lead_breath_s=None)
    card = format_breath_section(detect_breaths(y, 22050, truth_pitch_track(truth)))
    assert card.splitlines()[0].startswith("BREATH : none heard | longest phrase")
    assert "4 silent gaps (likely edits)" in card
    assert format_breath_section({}) == ""


def test_breath_that_clears_gates_but_misses_margins_is_medium_not_high():
    # loud (-8 dB vs singing: misses the level margin) and long (1.0 s: outside the typical-duration margin)
    y, truth = song(breath_rel_db=-8.0, breath_s=(1.0,) * 4, gap_s=1.3, lead_breath_s=None)
    r = detect_breaths(y, 22050, truth_pitch_track(truth))
    assert _matched(r["breaths"], truth["breaths"], tol=0.08) == len(truth["breaths"])
    assert {b["confidence"] for b in r["breaths"]} == {"medium"}
    card = format_breath_section(r)
    assert "long, loud" in card
