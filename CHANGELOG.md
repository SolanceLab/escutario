# Changelog

All notable changes to Escutário are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-04

First public release, under the PolyForm Noncommercial License 1.0.0.

### Added
- `python -m escutario.listen`: fetch a song from a link (yt-dlp) or a local file, split it into six stems (Demucs), and measure pitch (torchcrepe with Escutário's own Viterbi decoding), notes (basic-pitch), held notes, vibrato, riffs, simultaneous voices, breath phrases, voice texture, arrangement arrivals, key, tempo, sections and loudness.
- A compact listening card: timestamped numbers with a confidence on each reading, never mood words.
- `escutario.instrument` and `escutario.breakdown`: the MVSep Mega 53-stem BS-Roformer checkpoint (sha256-pinned, downloaded on first use) for violin and string isolation, grouped into the instruments a listener would name, with echoes of the same sound folded away and the reason kept.
- `escutario.fingerprint`: recognises the same recording dropped twice under different names or containers.
- `escutario.words` (opt-in, sends audio to Google Gemini): timestamped lines per voice with delivery, checked against Escutário's own measurements, behind a daily call cap written to disk before each call.
- `escutario.worker` and a launchd template: a queue worker for the House of Solance's own API. It needs that API and a key; the rest of Escutário runs without it.
- `python -m escutario.view`: the listening score in your browser, served from your own machine. Choose a song, play it with the score following, fold lanes away, zoom, and mark moments and sections that reach you; marks are kept in `out/<name>/marks.json`. The window under the score shows the line being sung when the words step has run.

### Known limits
- Thresholds were tuned on a small number of songs and one listener's corrections. Grit is not calibrated, and quiet background vocal layers are undercounted.
