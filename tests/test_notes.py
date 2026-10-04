# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Tests for escutario.notes — synthetic harmonic tones with known ground truth.

Runs the real basic-pitch ICASSP-2022 model (ONNX backend) on numpy-synthesized
signals; no network calls (the model ships inside the already-installed basic-pitch
package). Each call costs roughly one onnxruntime inference pass (~0.3-1s), so the
whole file runs in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from escutario.notes import _midi_to_note_name, transcribe_notes

SR = 22050


# ---------------------------------------------------------------------------
# synthesis helpers
# ---------------------------------------------------------------------------

def midi_to_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def harmonic_tone(freq: float, dur_s: float, sr: int = SR, n_harm: int = 3, tilt_db: float = -18.0, amp: float = 0.3) -> np.ndarray:
    """A clean harmonic tone: n_harm partials rolling off at tilt_db per octave,
    with a short raised-cosine-free linear fade to avoid onset/offset clicks that
    would register as spurious extra onsets.
    """
    n = int(round(dur_s * sr))
    t = np.arange(n) / sr
    y = np.zeros(n, dtype=np.float64)
    for k in range(1, n_harm + 1):
        a = 10 ** (tilt_db * np.log2(k) / 20)
        y += a * np.sin(2 * np.pi * freq * k * t)
    y = y / np.max(np.abs(y)) * amp
    fade = int(0.01 * sr)
    env = np.ones(n)
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    return (y * env).astype(np.float32)


def note_midis(notes: list[dict]) -> list[int]:
    return [n["midi"] for n in sorted(notes, key=lambda n: n["start_s"])]


# ---------------------------------------------------------------------------
# pure logic — no model
# ---------------------------------------------------------------------------

def test_midi_to_note_name():
    assert _midi_to_note_name(69) == "A4"
    assert _midi_to_note_name(60) == "C4"
    assert _midi_to_note_name(80) == "G#5"
    assert _midi_to_note_name(21) == "A0"


# ---------------------------------------------------------------------------
# model-backed
# ---------------------------------------------------------------------------

def test_single_sustained_note():
    """A single sustained A4 -> one note, right midi, duration within 60 ms."""
    x = harmonic_tone(midi_to_hz(69), 3.0)
    notes = transcribe_notes(x, SR)

    assert len(notes) == 1
    note = notes[0]
    assert note["midi"] == 69
    assert note["note_name"] == "A4"
    assert abs(note["dur_s"] - 3.0) < 0.06
    assert 0.0 <= note["salience"] <= 1.0


def test_two_note_chord():
    """A C4+E4 chord -> both notes present."""
    x = harmonic_tone(midi_to_hz(60), 2.0) + harmonic_tone(midi_to_hz(64), 2.0)
    x = (x / np.max(np.abs(x)) * 0.3).astype(np.float32)
    notes = transcribe_notes(x, SR)

    midis = {n["midi"] for n in notes}
    assert 60 in midis
    assert 64 in midis
    for n in notes:
        if n["midi"] in (60, 64):
            assert abs(n["dur_s"] - 2.0) < 0.1


def test_silence_yields_no_notes():
    x = np.zeros(int(3.0 * SR), dtype=np.float32)
    notes = transcribe_notes(x, SR)
    assert notes == []


def test_empty_array_yields_no_notes():
    x = np.zeros(0, dtype=np.float32)
    notes = transcribe_notes(x, SR)
    assert notes == []


def test_fast_stepwise_run_yields_separate_notes():
    """C4 D4 E4 F4, 0.3 s each -> four distinct, time-ordered notes."""
    step_midis = [60, 62, 64, 65]
    parts = [harmonic_tone(midi_to_hz(m), 0.3) for m in step_midis]
    x = np.concatenate(parts)
    notes = transcribe_notes(x, SR)

    assert note_midis(notes) == step_midis
    starts = [n["start_s"] for n in sorted(notes, key=lambda n: n["start_s"])]
    expected_starts = [0.0, 0.3, 0.6, 0.9]
    for actual, expected in zip(starts, expected_starts):
        assert abs(actual - expected) < 0.08


def test_resampling_from_44100():
    """Audio handed in at 44.1 kHz still transcribes correctly (internal resample)."""
    x = harmonic_tone(midi_to_hz(69), 3.0, sr=44100)
    notes = transcribe_notes(x, 44100)

    assert len(notes) == 1
    assert notes[0]["midi"] == 69
    assert abs(notes[0]["dur_s"] - 3.0) < 0.08


def test_notes_are_time_ordered():
    x = np.concatenate(
        [harmonic_tone(midi_to_hz(m), 0.3) for m in (65, 60, 64, 62)]
    )
    # deliberately out-of-pitch-order input; output must still be start_s-sorted
    notes = transcribe_notes(x, SR)
    starts = [n["start_s"] for n in notes]
    assert starts == sorted(starts)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_contiguous_pieces_of_one_pitch_merge_into_a_sustain():
    from escutario.notes import merge_sustains
    pieces = [(71.58, 1.59), (73.17, 0.34), (73.50, 0.21), (73.71, 0.29), (74.00, 0.14)]  # the mashup's G#5
    merged = merge_sustains([{"start_s": a, "dur_s": d, "midi": 80, "note_name": "G#5", "salience": 0.5} for a, d in pieces])
    assert len(merged) == 1 and abs(merged[0]["start_s"] - 71.58) < 1e-9 and abs(merged[0]["dur_s"] - 2.56) < 0.01


def test_repeated_notes_with_a_real_gap_stay_separate():
    from escutario.notes import merge_sustains
    notes = [{"start_s": 1.0, "dur_s": 0.2, "midi": 61, "note_name": "C#4", "salience": .6},
             {"start_s": 1.3, "dur_s": 0.2, "midi": 61, "note_name": "C#4", "salience": .6}]
    assert len(merge_sustains(notes)) == 2


def test_samples_by_channels_input_is_collapsed_on_the_channel_axis():
    import numpy as np
    from escutario.notes import transcribe_notes
    sr = 22050
    t = np.arange(sr) / sr
    tone = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    notes = transcribe_notes(np.stack([tone, tone], axis=1), sr)   # (n, 2)
    assert any(n["midi"] == 69 for n in notes)
