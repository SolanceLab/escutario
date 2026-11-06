---
name: escutario
description: Use when the user wants a song listened to: its pitch, held notes, riffs (cengkok), vibrato, how many voices sing, breath, arrivals of instruments, key, tempo and sections, drawn as a listening score they can play and mark. Sets Escutário up once on their computer, hears a song, opens the score. Everything runs locally except the optional words step.
---

# Escutário

A local ear for songs. It splits a song into six parts, measures what is in the sound, and draws a listening score where the user can play the song and mark the moments that move them. It reports numbers with a confidence on each, never feelings: do the same. Everything runs on the user's own computer. The one exception is the opt-in words step, which sends audio to Google Gemini.

## Set it up once

Needs Python 3.12 (the version it is tested on; some audio packages have no build for the newest Pythons yet), `ffmpeg` and `yt-dlp` on the computer (on a Mac: `brew install ffmpeg yt-dlp`). On Linux or Windows the user installs them and sets `ESCUTARIO_YT_DLP` and `ESCUTARIO_FFPROBE` to their absolute paths. Check these first and say what is missing.

Tell the user before the install: it downloads several gigabytes of audio libraries (torch, demucs), and model weights on the first song, so it takes a while. Ask where to keep it (default `~/escutario`), then:

```
git clone --depth 1 --branch v0.3.0 https://github.com/SolanceLab/escutario.git ~/escutario
cd ~/escutario
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps basic-pitch==0.4.0
```

On Windows there is no `make`, and the venv's Python is `.venv\Scripts\python.exe` instead of `.venv/bin/python` (use that path in every command in this skill, and `python` for `python3`). If the folder exists already, do not clone over it: ask.

## Hear a song

From the checkout, with the venv's Python:

```
.venv/bin/python -m escutario.listen "<link or audio file>" --slug <short-name>
```

`--slug` names the song's folder under `out/`. A link must be from a supported site (YouTube, TikTok, Instagram, SoundCloud, Bandcamp); a local audio file always works. A 4-minute song takes a few minutes. Everything it measured is written to `out/<short-name>/`. `--skip-split` reuses the audio and stems of a song already heard.

## Open the score

```
.venv/bin/python -m escutario.view --no-open
```

It serves the score at `http://127.0.0.1:8765` (change with `--port`), only on this computer. Give the user that address. They choose a song, press play, click to read what was measured at any second, and press `m` to mark a moment. Marks are kept in `out/<short-name>/marks.json`. Stop the process when they are done.

## Say what it found

Read `out/<short-name>/viz.json` for the numbers. Report them as measured, each with its confidence, and call weak readings weak. Do not name a feeling, a mood or a genre. Known limits: it is tuned on a few songs; grit is not calibrated; quiet background voices are undercounted; duets share one pitch track, so range and time at the top carry low confidence.

## The words step (ask first)

`.venv/bin/python -m escutario.words` writes timestamped sung lines, but it sends the song's audio to Google Gemini and needs `GEMINI_API_KEY` in the environment. Run it only when the user asks and has agreed to the audio leaving their computer. Never ask for the key in the chat and never print it.

## Never needed

The queue worker (`escutario.worker`) belongs to a private service and is not part of this plugin.
