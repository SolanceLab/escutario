# Escutário — build contract (Phase 1)

Design: `specs/2026-09-16-escutario-design.md`.
Every module builder reads this file first. Interfaces here are fixed; internals are yours.

## Rules for every module

- **Python 3.12, venv at `.venv/`.** Run things with `.venv/bin/python`. Installed: numpy, onnxruntime; demucs + torch + soundfile being installed. You may `pip install` scipy / pytest if you need them.
- **Do NOT import, execute or copy code from `vendor/attune/`.** You may READ it for reference (Maii's `singing.py` vibrato + YIN are good prior art). Everything under `escutario/` is our own code.
- **Numbers, not moods.** No mood words in any output ("sad", "angry", "emotional"). Technique words that describe sound production are allowed: breath, glide, fall, break, ring, pressed, grit, sustained.
- **Every measurement carries a confidence** (`"high" | "medium" | "low"`) or is omitted. Never fake precision; omit rather than guess.
- **Every threshold is a named module-level constant** with a one-line comment saying it's a starting value to calibrate. No magic numbers in logic.
- **Tests:** `tests/test_<module>.py`, runnable with `.venv/bin/python -m pytest tests/test_<module>.py -q`. Synthesize signals with numpy (known ground truth) and also cover false positives (noise, silence, steady tone). No network in tests.
- **Only touch your own files** (`escutario/<module>.py`, `tests/test_<module>.py`, and fixture generators under `fixtures/`). Don't edit `audio.py`, `types.py`, or another builder's module; if you need a change there, say so in your report.
- **No git commands.** The session commits.
- Report back: files written, test output (pass counts), constants you chose and why, known limits.

## Shared (already written — use, don't edit)

- `escutario/audio.py` — `load_mono(path, sr=22050) -> (np.ndarray float32, sr)`, `load_stereo(path, sr=44100) -> (np.ndarray float32 shape (2, n), sr)`, `rms_db(x) -> float`. Decodes with ffmpeg.
- `escutario/types.py` — `PitchTrack` (times, f0_hz with 0 = unvoiced, confidence 0..1, hop_s).

## Modules

### `fetch.py` — sources
`fetch(source: str, out_dir: Path, *, track: str|None=None, artist: str|None=None, expected_duration_s: float|None=None) -> FetchResult`
- `source` is a URL or a local file path. URLs must be https and the host must match the allowlist exactly or as a subdomain: youtube.com, youtu.be, music.youtube.com, tiktok.com, instagram.com, soundcloud.com, bandcamp.com. Anything else → `FetchRefused` with a clear reason. Reject userinfo, non-default ports, non-https.
- `search(track, artist, expected_duration_s) -> FetchResult` for the Spotify pointer: `ytsearch5:` via yt-dlp metadata first (`--dump-json --flat-playlist` or equivalent), pick the candidate whose duration is closest, flag `duration_mismatch` if it differs from expected by more than 3 s, then download.
- Download with the system `yt-dlp` (`/opt/homebrew/bin/yt-dlp`) as a subprocess: audio only, `--no-playlist`, `--max-filesize 50M`, cap at 600 s (refuse longer using metadata before downloading).
- `FetchResult` dataclass: `path, source_url, title, artist (uploader/track if present), duration_s, platform, short_form: bool (tiktok, instagram, youtube shorts URL), notes: list[str]` (e.g. `"short-form source — may be sped up, pitched or cut; pitch and tempo describe this edit, not the original"`, duration mismatch).
- Tests mock the subprocess; no network.

### `breath.py` — the breath organ
`detect_breaths(vocal: np.ndarray, sr: int, pitch: PitchTrack) -> dict`
- Input is a SEPARATED vocal stem (mono). Find breath events in unvoiced spans between voiced phrases: roughly 150–900 ms long, energy above the stem's noise floor but clearly below the neighbouring phrase level, noise-like (high spectral flatness), energy concentrated around 1–6 kHz.
- Output: `{"breaths": [{"start_s", "end_s", "duration_s", "level_db_rel_phrase", "confidence"}], "count", "phrases": [{"start_s","end_s","duration_s","preceded_by_breath": bool}], "longest_phrase_s", "silent_gaps": [{"start_s","end_s"}], "notes": [...]}`.
- `silent_gaps` = gaps between phrases with NO breath energy (likely a production edit). Notes must say that absent breaths are not evidence the singer didn't breathe.
- Leakage guard: if short broadband transients recur periodically (hi-hat-like), lower confidence and add a note.
- `format_breath_section(result) -> str` — a compact human card section (`BREATH:` lines).
- Lyric placement (before line / mid-line) is done later by the card layer; just give accurate times.
- Tests: synthetic tone phrases separated by band-passed noise bursts (true breaths), by pure silence (silent gaps), with hi-hat-like periodic clicks (false positives), steady noise, and a continuous tone (zero breaths).

### `wail.py` — voice strength, texture and the wail
`analyze_voice(vocal: np.ndarray, sr: int, pitch: PitchTrack) -> dict` measuring the components of what Anne hears as "wailing" (tone + vocal strength). Measure each separately; do NOT combine them into a single "wail score" yet (that gets calibrated against her songs later):
- **Register position:** the singer's range in this song (5th–95th percentile of voiced f0), and the share of voiced time spent in the top third of that range; longest sustained note in the top third.
- **Ring:** the energy ratio of 2–4 kHz to 0–2 kHz while voiced (the singer's-formant "ring" of a projected voice), per phrase and overall.
- **Pressed or full voice / spectral tilt:** alpha ratio (energy 1–5 kHz vs 50 Hz–1 kHz, in dB) while voiced; also H1–H2 (first two harmonic amplitudes from f0) where confidence allows.
- **Loudness on high notes:** does level rise with pitch (correlation of dB with f0 in cents)?
- **Cry falls:** downward glides at note/phrase ends — start pitch, drop in cents, duration.
- **Register breaks / flips:** abrupt pitch jumps (> ~4 semitones within ≤ 60 ms) inside continuously voiced sound.
- **Grit:** roughness while loud — harmonics-to-noise ratio (HNR) on sustained notes, plus a subharmonic/jitter indicator; report whether low HNR co-occurs with high level (grit) or low level (breathiness).
- `format_voice_section(result) -> str` — `VOICE:` card lines.
- Tests: synthetic harmonic tones with controlled spectral tilt, an added 3 kHz formant boost, noise mixing for HNR, programmed falling glides and octave flips, plus false-positive cases (vibrato must not register as falls or breaks; a steady tone yields no falls).

### `split.py` — track splitting (Anne asked 17:17: "can you also split tracks?")
`split(path: Path, out_dir: Path, *, model: str = "htdemucs_6s", device: str|None=None, progress_cb=None) -> SplitResult`
- Demucs via its Python API (`demucs.api` if present in the installed version, otherwise `demucs.pretrained.get_model` + `demucs.apply.apply_model`). Default 6 stems: vocals, drums, bass, guitar, piano, other. `model="htdemucs"` gives 4.
- Device: `mps` if available, else `cpu`; fall back to cpu on an MPS error, with a note.
- Writes `<out_dir>/<stem>.wav` (44.1 kHz stereo 16-bit) and returns `SplitResult(stems: dict[str, Path], model, device, seconds_elapsed, notes)`. Also provide `load_stem_mono(result, "vocals", sr)` for the analysers.
- Cap input at 600 s.
- **Why it matters (Anne, 17:19):** she plays violin by ear and wants to isolate the violin to learn it and play along. Demucs has no violin stem (violin lands in `other`). So structure `split.py` around a small backend interface (`DemucsBackend` now; a named-instrument backend such as a violin/strings model gets added after a scout picks one). Also provide `minus(result, target: str, out_dir) -> Path`: writes `minus_<target>.wav` = the sum of every stem except `target` (the play-along backing track), peak-safe (no clipping).
- Tests: a small synthetic mix (a sine "voice" + noise "drums") → stems exist, correct names, correct length ± 1 frame; skip cleanly (pytest.skip) if demucs or the model weights are unavailable. Model weights download on first real use; the test may trigger that once.

---

# Phase 1b — make it entirely ours (Anne, 16 Sep 19:13)

Escutário's modules import nothing from `vendor/attune`, but the page data was produced with three of Maii's pieces. Replace them, then `vendor/attune` gets deleted. **You may READ vendor code for understanding; never copy it, never import it in `escutario/`.** Evidence files already on disk (read-only inputs for comparison):
- `out/<song>/stems/*.wav` (Demucs 6 stems), `out/<song>/source.wav`, songs: `apparition-x-unethical`, `forbidden-fruit`
- `out/<song>/pitch.npz` — Maii's YIN track (times, f0, conf, hop_s, sr) — the thing to beat
- `out/<song>/basic_pitch_vocals.json` — notes from Maii's basic-pitch wrapper — parity reference
- `out/<song>/attune.json` → `music` — Maii's key/tempo/sections — parity reference

**Anne's ear is the ground truth where she gave it** (mashup): a second voice layers in from 0:41; a G#5 sustain from ~1:12 (the YIN track loses it 1:10–1:17); a cengkok (one voice) 0:30.8–0:36.7; three vocal layers from 2:53.9; Faouzia's riff 3:25.8–3:31.6; the opening piano plays from 0:00.

### `pitch.py` — our pitch tracker
`track_pitch(vocal: np.ndarray, sr: int, *, method: str = "auto") -> PitchTrack` (types.PitchTrack; f0 0 = unvoiced; confidence 0..1). Candidates to evaluate and choose between on evidence: `librosa.pyin` and `torchcrepe` (both pip-installable; torch 2.14 + MPS present), optionally your own. Measure on the real mashup vocal stem: voiced coverage while the voice is audibly singing (level > −35 dBFS) in 1:09–1:18 and 0:30–0:37, octave-jump rate (frame-to-frame |Δ| within ±80¢ of 1200/2400), agreement with the basic-pitch notes (pitch class match share where both voiced), runtime on MPS/CPU for a 4-min stem. Synthetic tests: vibrato tone (rate/extent recovered), glide, silence, noise, two simultaneous tones (follows the stronger, no octave hopping). Also `save_npz(track, path, sr)` writing the same keys as `pitch.npz`. Pick the default by the numbers and report them in a table.

### `notes.py` — polyphonic notes, straight from Spotify's model
`transcribe_notes(x: np.ndarray, sr: int) -> list[dict]` with `start_s, dur_s, midi, note_name, salience`. Use Spotify basic-pitch directly: the official `basic-pitch` package with an ONNX/CoreML backend if it installs without TensorFlow, otherwise your own ONNX inference over the ICASSP-2022 `nmp.onnx` (Apache-2.0, credit Spotify; the file is already at `vendor/attune/models/basic-pitch-nmp.onnx`, sha256 2c3c1d14…; copy it to `models/` with a sha256 check). Parity: onset-matched share (±50 ms, same midi) vs `basic_pitch_vocals.json` on both songs; report it. Don't chase exact parity with a wrapper's choices; do report where and why you differ.

### `harmony.py` + `listen.py` — the song facts and the pipeline
`analyze_harmony(mix: np.ndarray, sr: int) -> dict` with `key {key, confidence}`, `tempo {bpm, confidence, beat_times}` (report double/half-time ambiguity honestly), `sections [t]`, `energy {loudest_t, loudest_db}`, `duration_s`. Own numpy or librosa. Compare against `attune.json → music` on both songs.
`escutario/listen.py` + `python -m escutario.listen <url|file> --slug <name> [--track --artist]`: fetch → split (Demucs) → track_pitch(vocals) → transcribe_notes(vocals) → breath, wail, entrances+arrangement_changes, count_voices+detect_riffs → analyze_harmony → writes `out/<slug>/*.json` and a `viz` entry compatible with `web/viz_data.json` (same keys the page uses today: read `web/build.py`, `web/viz_data.json` and `/tmp/viz_data.py` for the current shape). Imports `pitch.track_pitch` and `notes.transcribe_notes` — they're being written in parallel; code against the signatures above and test with fakes. No `vendor` import anywhere.
