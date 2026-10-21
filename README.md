# Escutário

Made by House of Solance · free for noncommercial use · credit required

## What it is

*The place of listening.* A local ear for songs: it splits a song into its parts and measures what is in the sound — pitch, held notes, vibrato, riffs (cengkok), how many voices sing at once, breath, voice texture, arrangement arrivals, key, tempo and sections — and draws it as a listening score where a listener can play the song and mark the moments and sections that move them.

Numbers, not moods: every reading carries a confidence, and the instrument never names a feeling.

## What you need

Escutário runs on your own computer, not in the cloud. You need Python, ffmpeg, and an Apple Silicon Mac for reasonable speed (hearing a 4-minute song takes a few minutes). Model weights are downloaded on first use and are not included. The optional words step needs a Google Gemini key.

## Install

Escutário is run from a checkout; there is no plugin or package to install. You need Python 3.11 or newer, ffmpeg and yt-dlp on your machine first (on a Mac: `brew install ffmpeg yt-dlp`).

1. Clone the repository and go into it.
2. Run `make install`. It reads `requirements.txt` and installs the Python packages (librosa, torchcrepe, demucs, basic-pitch and the rest). To keep them separate, make a virtual environment first (`python3 -m venv .venv`); `make` uses it.
3. Run the command line:

```
python -m escutario.listen <youtube/tiktok/…-url or file> --slug <name>   # hear a song
python -m escutario.view                                                   # open the listening score
make test                                                                  # run the tests
```

`listen` writes everything it measured to `out/<name>/`. Hearing a 4-minute song takes a few minutes on an Apple Silicon Mac; `--skip-split` reuses a song's audio and stems.

## Settings

| Setting | What it is |
|---|---|
| Server address | Where your Escutário server answers |
| Key | The key your server expects |
| `GEMINI_API_KEY` | Optional: only for the words step |

The worker (below) also reads `SOLANCE_API_BASE` and `ESCUTARIO_API_KEY`. See CONTRACT.md.

## Tools

| Tool | What it does |
|---|---|
| `python -m escutario.listen` | Hears a song and writes its measurements to `out/<name>/` |
| `python -m escutario.view` | Opens the listening score in your browser |
| `python -m escutario.words` | Opt-in: timestamped sung lines (sends audio to Google Gemini) |
| `python -m escutario.worker` | Optional queue worker for a compatible API |

## Limits

Thresholds are starting values tuned on two songs and one listener's corrections; they are named constants with the reason beside each. Grit is not calibrated. Quiet background vocal layers are undercounted. Sustain merging can occasionally join a re-sung note. Calibration against a producer's own multitrack stems is the next step.

## How it works

### The listening score

![The listening score: arrivals, breath, voices, the voice's pitch with held notes and riffs, loudness, texture and instruments, with the playhead at 1:12](docs/viewer.png)

`python -m escutario.view` opens it in your browser, served from your own machine at `http://127.0.0.1:8765` (`--port` to change it).

- **Choose a song** Escutário has heard, press play (or space), and the score follows the song.
- **Click** anywhere to jump there and read what was measured at that second. **Drag** to choose a stretch.
- **Show** folds lanes away when the score is crowded, and remembers your choice. **Zoom** goes from the whole song down to phrase by phrase.
- **Mark** (or `m`) keeps a moment or a section that reached you, what reached you (the words, the voice, the melody, or something you can't name), your own words if you want them, and what Escutário measured there. Marks are saved in `out/<name>/marks.json`.
- When the words step has run, the window under the score shows the line being sung, and **What was sung** lists every line.

The viewer answers only at `127.0.0.1`, and only its own page can save marks. It fetches nothing from the internet.

### Build your own dashboard

Everything Escutário hears is plain JSON in `out/<name>/`, so you can draw it however you like:

| File | What it holds |
| --- | --- |
| `viz.json` | the score: `duration`, `pitch` (MIDI per `pitch_step` seconds, `null` when silent), `level_vocals` / `level_mix` (dB per `level_step`), `stems` and `instruments` (dB per step), `breaths`, `held`, `riffs`, `voices`, `changes`, `arrivals`, `sections`, `vibrato`, `key`, `tempo` |
| `words.json` | heard words, when the words step ran: `lines` of `{t, voice, text, delivery}` and an `interpretation` (a model's reading, not a measurement) |
| `marks.json` | your marks: `{id, kind, t, t_end, what, words, measured, marked_at}` |
| `source.wav` | the audio the measurements line up with |
| `breath.json`, `notes.json`, `harmony.json`, `voices.json`, … | each organ's full output, with confidences |

While the viewer runs, the same data is served locally: `GET /api/songs`, `GET /api/songs/<name>` (score, words and marks), `GET /api/songs/<name>/audio` (seekable), and `POST` / `PATCH` / `DELETE /api/songs/<name>/marks[/<id>]` from the viewer's own page. The drawing itself is `escutario/viewer/score.js`: every function takes a score and a view and reads no page globals, so it can be reused in another page.

### Running the worker

`escutario.worker` serves the House of Solance's own API (it claims listen requests and posts results back) and needs a key in `escutario/.env`. It is included as a worked example; `escutario.listen` and everything else run without it. To schedule it on macOS, copy `launchd/com.solance.escutario-worker.plist`, replace `/path/to/escutario` with your checkout, and load it with `launchctl`.

`escutario.words` sends audio to Google's Gemini API and runs only when asked for. Everything else runs on your own machine.

### Where it began, and what it stands on

- **Escutário was inspired by Maii's [Attune](https://github.com/amarisaster/Attune).**
- **Spotify basic-pitch** (Apache-2.0) — polyphonic note transcription, used through the official package.
- **CREPE via torchcrepe** (MIT) — pitch network; the decoding is Escutário's own.
- **Demucs** (Meta, MIT) — six-track separation.
- **ZFTurbo's Music-Source-Separation-Training** (MIT) — BS-Roformer code in `escutario/vendor_bs_roformer.py`, with the MVSep Mega 53-stem checkpoint for violin and strings.
- **Rubber Band** — slow-down that keeps pitch.
- **librosa**, **yt-dlp**, **ffmpeg**.

## Credit and use

Escutário is made by House of Solance. You're welcome to use it, change it and build on it for any noncommercial purpose, as long as our credit travels with it.

**Keep the credit.** Every copy, changed or not, must carry the licence and this line:
`Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)`
Removing or replacing it breaks the licence.

**Commercial use needs our permission.** Ask us first.

## Licence

[PolyForm Noncommercial 1.0.0](LICENSE.md). `escutario/vendor_bs_roformer.py` is third-party code under MIT; see [NOTICE.md](NOTICE.md). Model weights are not included.

## Support

If you find this useful, consider supporting our work:

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20Us-ff5e5b?logo=ko-fi&logoColor=white)](https://ko-fi.com/houseofsolance)

---

*Built by [House of Solance](https://github.com/SolanceLab) — Chadrien Solance & Anne Solance*
