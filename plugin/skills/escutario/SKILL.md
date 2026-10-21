---
name: escutario
description: Use Escutário to hear a song as numbers (pitch, held notes, vibrato, riffs, voices, breath, arrivals, key, tempo, sections) and open its listening score.
---

# Escutário

Escutário is a local ear for songs. It splits a song into parts, measures what is in the sound, and draws a listening score where a listener can play the song and mark the moments that move them.

- To hear a song: `python -m escutario.listen <link or file> --slug <name>`. Results land in `out/<name>/` as plain JSON.
- To open the score: `python -m escutario.view` (served only at 127.0.0.1).

Limits: readings are numbers with a confidence, never feelings. Thresholds were tuned on two songs; grit is not calibrated; quiet background voices are undercounted. Hearing a 4-minute song takes a few minutes on an Apple Silicon Mac. The words step sends audio to Google Gemini and runs only when asked.
