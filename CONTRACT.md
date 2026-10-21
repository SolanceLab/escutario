# Escutário contract

## Doors it offers

| Door | What it answers |
|---|---|
| `python -m escutario.listen <link or file> --slug <name>` | Hears a song and writes everything measured to `out/<name>/` |
| `python -m escutario.view [--port N]` | The listening score, served only at `127.0.0.1:8765` |
| `GET /api/songs`, `GET /api/songs/<name>`, `GET /api/songs/<name>/audio` | The viewer's local data: songs, score, words and marks, seekable audio |
| `POST` / `PATCH` / `DELETE /api/songs/<name>/marks[/<id>]` | Marks, from the viewer's own page only |
| `python -m escutario.worker --once` | Optional queue worker for a compatible API: claims listen requests and posts results back |

## Data it owns

| Table / bucket / file | What is in it |
|---|---|
| `out/<name>/viz.json` | The score: pitch, levels, stems, breaths, held notes, riffs, voices, arrivals, sections, vibrato, key, tempo |
| `out/<name>/words.json` | Heard words, when the words step ran |
| `out/<name>/marks.json` | The listener's marks |
| `out/<name>/source.wav` and stems | The audio the measurements line up with |
| `out/<name>/*.json` | Each organ's full output, with confidences |
| `web/viz_data.json` | Page data merged from the songs heard |

## Settings and secrets it needs

Named here, never valued.

| Name | What it is |
|---|---|
| `SOLANCE_API_BASE` | Worker only: address of the API the worker serves |
| `ESCUTARIO_API_KEY` | Worker only: the worker's own key (secret) |
| `GEMINI_API_KEY` | Words step only: Google Gemini key (secret) |
| `ESCUTARIO_ENV_FILE` | Optional path to the `.env` file holding the three above |
