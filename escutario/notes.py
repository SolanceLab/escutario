# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""notes.py — polyphonic note transcription, straight from Spotify's basic-pitch model.

Uses the official `basic-pitch` PyPI package (Spotify, Apache-2.0) as a library, not a
reimplementation: this module is a thin numpy-array adapter around
`basic_pitch.inference` and `basic_pitch.note_creation`, which do the actual windowing,
model inference and polyphonic note decoding. Nothing here re-derives Spotify's
algorithm and nothing is copied from other wrappers.

Version pinned: basic-pitch==0.4.0. Model: the ICASSP-2022 checkpoint bundled inside the
package at `basic_pitch/saved_models/icassp_2022/nmp.onnx`,
sha256 2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec (Spotify's release).

TensorFlow-free install (verified: `python -c "import basic_pitch; assert not
basic_pitch.TF_PRESENT and not basic_pitch.CT_PRESENT and basic_pitch.ONNX_PRESENT"`)::

    .venv/bin/pip install --no-deps basic-pitch==0.4.0
    .venv/bin/pip install librosa mir-eval "pretty-midi>=0.2.9" "resampy==0.4.3" \
        scikit-learn typing-extensions setuptools

Do NOT install the `basic-pitch[tf]`, `[coreml]`, `[dev]` or plain `pip install
basic-pitch` form on macOS — basic-pitch's base requirements pull `coremltools`
unconditionally on Darwin and `tensorflow-macos` on Darwin + Python > 3.11, which is
this venv. `--no-deps` plus the five packages above is the whole (non-TF) runtime
`basic_pitch.note_creation` needs; with no tensorflow/coremltools/tflite-runtime
installed, `basic_pitch.inference.Model` self-selects its ONNX backend
(`onnxruntime.InferenceSession`, CPU execution provider) and loads `nmp.onnx` directly —
this is the "bundled ONNX model" path the build contract asked to try first.

`resampy` is pinned one patch above basic-pitch's declared upper bound (`<0.4.3`)
because 0.4.2 imports the now-removed `pkg_resources` at import time and fails outright
in this environment's setuptools (84.0.0, no bundled pkg_resources); 0.4.3 switched to
`importlib.resources` and has no such dependency. This is a metadata-only conflict —
`pip` prints a warning, nothing breaks — resampy is only used by basic-pitch's audio
loader for a resampling step this module doesn't otherwise touch.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import soundfile as sf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # basic_pitch sets this too; harmless with no TF installed

from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: E402 — after the TF env var, matching basic_pitch's own predict.py
from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP  # noqa: E402
from basic_pitch.inference import Model, run_inference  # noqa: E402
from basic_pitch.note_creation import model_output_to_notes  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — every one is a STARTING VALUE to calibrate. Values are Spotify's own
# ICASSP-2022 defaults (basic_pitch.inference.predict), used as the sane baseline
# rather than re-derived, since they're what the model was tuned against.
# ---------------------------------------------------------------------------

MIN_NOTE_LENGTH_MS = 127.70  # starting value to calibrate: Spotify's default minimum note length (~11 frames @ ~86 fps)
MIN_FREQUENCY_HZ = None  # starting value to calibrate: no low-frequency cutoff
MELODIA_TRICK = True  # starting value to calibrate: recover a held continuation note the onset head missed
MIDI_TEMPO = 120.0  # starting value to calibrate: only sets the (unused) PrettyMIDI tempo grid, not note timing
INFER_ONSETS = True  # starting value to calibrate: add onsets inferred from frame-energy jumps, not just the onset head
FRAME_THRESHOLD = 0.3  # starting value to calibrate: Spotify's default frame-activation threshold
MERGE_GAP_S = 0.03  # same-pitch pieces touching within this gap are one sustained note — set 16 Sep from the mashup G#5: the sound holds 71.57–74.35 s, basic-pitch returned five contiguous pieces
MAX_FREQUENCY_HZ = None  # starting value to calibrate: no high-frequency cutoff
ONSET_THRESHOLD = 0.5  # starting value to calibrate: Spotify's default onset-activation threshold

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_model: Model | None = None  # lazily-loaded singleton; reuses one onnxruntime session across calls


def _get_model() -> Model:
    global _model
    if _model is None:
        _model = Model(ICASSP_2022_MODEL_PATH)
    return _model


def _midi_to_note_name(midi: int) -> str:
    """Scientific pitch notation, sharps only (midi 60 -> "C4", 69 -> "A4")."""
    octave = midi // 12 - 1
    return f"{_NOTE_NAMES[midi % 12]}{octave}"


def transcribe_notes(x: np.ndarray, sr: int) -> list[dict]:
    """Polyphonic note transcription via Spotify's basic-pitch (ICASSP-2022, ONNX backend).

    x: mono float32 audio at any sample rate; sr: that rate. basic-pitch's own audio
    loader (librosa, inside `run_inference`) resamples internally to its native
    22050 Hz, so x need not already be at that rate — but see the runtime note below.

    Returns a list of {"start_s", "dur_s", "midi", "note_name", "salience"} dicts,
    time-ordered. salience is basic-pitch's own per-note `amplitude` — the mean
    note-posteriorgram activation over the note's frames (0..1) — which is exactly
    the contract's salience definition, read off rather than recomputed.

    Runtime note: basic-pitch's public entry points (`run_inference`,
    `get_audio_input`) accept a file path, not an array, and do their own
    `librosa.load` internally. Rather than reimplement that windowing/resampling
    ourselves (risking drift from Spotify's own framing), x is written to a temp WAV
    and handed to their loader, so every sample that reaches the model goes through
    Spotify's own code path end to end. This costs one small disk round-trip per call.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = np.squeeze(x)
    if x.ndim > 1:
        # collapse to mono along the CHANNEL axis, whichever way round it came: (channels, n) or (n, channels)
        x = x.mean(axis=0) if x.shape[0] < x.shape[1] else x.mean(axis=1)
    if x.size == 0:
        return []

    model = _get_model()

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(tmp_path, x, sr, subtype="FLOAT")
        model_output = run_inference(tmp_path, model)
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    min_note_len_frames = int(round(MIN_NOTE_LENGTH_MS / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))

    _midi_obj, note_events = model_output_to_notes(
        model_output,
        onset_thresh=ONSET_THRESHOLD,
        frame_thresh=FRAME_THRESHOLD,
        infer_onsets=INFER_ONSETS,
        min_note_len=min_note_len_frames,
        min_freq=MIN_FREQUENCY_HZ,
        max_freq=MAX_FREQUENCY_HZ,
        include_pitch_bends=False,  # not part of the contract's output shape; skips a pass we don't use
        melodia_trick=MELODIA_TRICK,
        midi_tempo=MIDI_TEMPO,
    )

    notes = []
    for start_s, end_s, midi_pitch, amplitude, _pitch_bend in note_events:
        midi_pitch = int(midi_pitch)
        notes.append(
            {
                "start_s": float(start_s),
                "dur_s": float(end_s - start_s),
                "midi": midi_pitch,
                "note_name": _midi_to_note_name(midi_pitch),
                "salience": float(amplitude),
            }
        )
    notes.sort(key=lambda n: n["start_s"])
    return merge_sustains(notes)


def merge_sustains(notes: list[dict]) -> list[dict]:
    """Join back-to-back pieces of the same pitch into one note. basic-pitch re-triggers onsets
    inside a long sung note; the pieces share a midi and touch end to start."""
    open_by_midi: dict[int, dict] = {}
    out: list[dict] = []
    for n in sorted(notes, key=lambda n: n["start_s"]):
        prev = open_by_midi.get(n["midi"])
        if prev is not None and n["start_s"] - (prev["start_s"] + prev["dur_s"]) <= MERGE_GAP_S:
            end = max(prev["start_s"] + prev["dur_s"], n["start_s"] + n["dur_s"])
            total = prev["dur_s"] + n["dur_s"]
            prev["salience"] = (prev["salience"] * prev["dur_s"] + n["salience"] * n["dur_s"]) / total if total > 0 else prev["salience"]
            prev["dur_s"] = end - prev["start_s"]
            continue
        n = dict(n)
        open_by_midi[n["midi"]] = n
        out.append(n)
    return out
