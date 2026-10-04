# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Words and delivery — what is sung, by which voice, and how, from a model that hears the recording.

Escutário's other organs measure the sound: timing, pitch, layers, breath. They cannot say what
the words are or whether a line is whispered, belted or strained. This organ sends the actual
audio to an audio-capable Gemini model and asks for timestamped lines per voice, the delivery of
each line in audible terms, and a short reading of the performance, marked as interpretation.

Two rules keep it honest:
  * Ear, not memory. The prompt carries no title or artist, so the model has to transcribe what it
    hears rather than recite lyrics it already knows.
  * Checked against the ear. `check_words` compares every line with Escutário's own measurements
    (was a voice actually sounding there? is a line called sustained where a held note was measured?)
    and the card reports where they disagree instead of trusting either side.

Privacy: this sends audio to Google, so it never runs as part of hearing a song. It runs only for an
explicit `words` request (escutario_words), made for a song already heard; a song from a private
file can be sent only by Penthouse or the Study (solance-api decides). Cost: every call is counted
in a daily cap that is written to disk BEFORE the request goes out, so a failed write refuses the call.

Run by hand on a song Escutário already heard:  python -m escutario.words out/<slug>
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
HTTP_TIMEOUT_S = 240
MAX_AUDIO_S = 600                   # matches listen's analysis cap; 64 kbit/s mono stays far under the 20 MB inline limit
MAX_AUDIO_BYTES = 12 * 1024 * 1024
REPO = Path(__file__).resolve().parents[1]
MODEL = "gemini-3.8-flash"
DAILY_CAP = 25                      # calls per day, across all songs
USAGE_PATH = REPO / "out" / ".words-usage.json"

MAX_DELIVERY = 160
MAX_INTERPRETATION = 1200
VOICES = ("female", "male", "both", "group", "unknown")
MAX_TEXT = 200
MAX_LINES = 200

VOICE_WINDOW_BEFORE_S = 1.0         # a line counts as heard by the ear if a voice sounds from t-1 s ...
HELD_MIN_S = 1.0
VOICE_WINDOW_AFTER_S = 2.5          # ... to t+2.5 s
SUSTAIN_WORDS = re.compile(r"\b(sustain\w*|held|hold\w*|elongat\w*|long note|stretched)\b", re.I)
AGREEMENT_FLOOR = 0.6               # below this share of lines with a measured voice, the words are marked unverified

_CONTROL = re.compile(r"[\x00-\x1f\x7f\u2028\u2029]")   # newlines too: model text must never start a card section of its own
MAX_OUTPUT_TOKENS = 8192            # bounds what one call can bill for output (thinking included)

PROMPT = """Listen to this recording. It may have one singer or several.
1. Transcribe the sung lyrics line by line. Give each line its start time in seconds from the start of the recording and the voice singing it: female, male, both, group, or unknown. Transcribe only what you can actually hear; write [unclear] for words you cannot make out. Do not fill in lyrics from memory of any song.
2. For each line, describe the vocal delivery in concrete audible terms (for example breathy, belted, strained, falsetto, whispered, clipped, sustained, vibrato, cracking), not feelings. Keep it short.
3. In 3 to 5 sentences, say what emotional character the performance conveys and which audible details that reading rests on. This is interpretation, and it should say so where it is unsure."""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "lines": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "t": {"type": "NUMBER"},
                    "voice": {"type": "STRING", "enum": list(VOICES)},
                    "text": {"type": "STRING"},
                    "delivery": {"type": "STRING"},
                },
                "required": ["t", "voice", "text", "delivery"],
            },
        },
        "interpretation": {"type": "STRING"},
    },
    "required": ["lines", "interpretation"],
}


class WordsError(RuntimeError):
    pass


# ---------------------------------------------------------------- daily cap

def take_from_daily_cap(path: Path = USAGE_PATH, cap: int = DAILY_CAP, today: str | None = None) -> int:
    """Count one call against today's cap, on disk, before the call. Returns calls used today.

    Fails closed: a reached cap, an unreadable or malformed counter, or a counter that cannot be
    written all refuse the call. A cap that cannot be read or recorded is not a cap. The whole
    read-increment-write holds an exclusive lock, so the worker and a hand run cannot share a count.
    """
    today = today or dt.date.today().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data: dict = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                raise WordsError("the daily words counter is unreadable; refusing to spend until it is fixed") from None
            if not isinstance(data, dict) or not all(isinstance(v, int) and v >= 0 for v in data.values()):
                raise WordsError("the daily words counter is malformed; refusing to spend until it is fixed")
        used = data.get(today, 0)
        if used >= cap:
            raise WordsError(f"daily cap of {cap} words calls reached")
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({today: used + 1}))
        os.replace(tmp, path)
        return used + 1


# ---------------------------------------------------------------- the call

def encode_audio(wav: Path, *, max_s: float = MAX_AUDIO_S) -> bytes:
    """Mono 64 kbit/s mp3 of the first max_s seconds; small enough to send inline."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "words.mp3"
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-t", str(max_s), "-i", str(wav),
               "-vn", "-ac", "1", "-b:a", "64k", str(out)]
        done = subprocess.run(cmd, capture_output=True, timeout=120)
        if done.returncode != 0 or not out.exists():
            raise WordsError("could not encode the audio for the words model")
        data = out.read_bytes()
    if len(data) > MAX_AUDIO_BYTES:
        raise WordsError("encoded audio is too large to send inline")
    return data


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None   # the API key must never follow a redirect to another origin


def _post(url: str, body: dict, api_key: str, timeout: float = HTTP_TIMEOUT_S) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode()).get("error", {}).get("status", "")
        except Exception:  # noqa: BLE001
            detail = ""
        raise WordsError(f"words model answered HTTP {e.code} {detail}".strip()) from None
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise WordsError(f"words model unreachable: {type(e).__name__}") from None


def _clean(s: object, limit: int) -> str:
    return _CONTROL.sub(" ", str(s or "")).strip()[:limit]


def parse_response(resp: dict, duration: float) -> dict:
    """Validate the model's answer into bounded, plain data. Everything in it is untrusted text."""
    try:
        parts = resp["candidates"][0]["content"]["parts"]
        text = next(p["text"] for p in reversed(parts) if isinstance(p, dict) and "text" in p and not p.get("thought"))
        raw = json.loads(text)
    except (KeyError, IndexError, TypeError, StopIteration, ValueError, AttributeError):
        try:
            reason = (resp.get("candidates") or [{}])[0].get("finishReason")
        except (AttributeError, IndexError, TypeError):
            reason = None
        raise WordsError(f"words model gave no usable answer{f' ({reason})' if reason else ''}") from None
    # a malformed shape is an error, never an empty song that gets cached as its words
    if not isinstance(raw, dict) or not isinstance(raw.get("lines"), list) or not isinstance(raw.get("interpretation", ""), str):
        raise WordsError("words model answered in the wrong shape")
    lines = []
    for item in raw["lines"][:MAX_LINES]:
        if not isinstance(item, dict):
            continue
        try:
            t = float(item.get("t"))
        except (TypeError, ValueError):
            continue
        if not (0 <= t <= duration + 2):
            continue
        text = _clean(item.get("text"), MAX_TEXT)
        if not text:
            continue
        voice = item.get("voice") if item.get("voice") in VOICES else "unknown"
        lines.append({"t": round(t, 1), "voice": voice, "text": text, "delivery": _clean(item.get("delivery"), MAX_DELIVERY)})
    lines.sort(key=lambda x: x["t"])
    return {"lines": lines, "interpretation": _clean(raw.get("interpretation"), MAX_INTERPRETATION)}


def hear_words(wav: Path, *, duration: float, api_key: str, model: str = MODEL, post=_post,
               encode=encode_audio, take_cap=take_from_daily_cap) -> dict:
    if not api_key:
        raise WordsError("no GEMINI_API_KEY")
    audio = encode(Path(wav))
    take_cap()
    body = {
        "contents": [{"parts": [{"inline_data": {"mime_type": "audio/mp3", "data": base64.b64encode(audio).decode()}},
                                {"text": PROMPT}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": MAX_OUTPUT_TOKENS,
                             "responseMimeType": "application/json", "responseSchema": RESPONSE_SCHEMA},
    }
    resp = post(f"{API_ROOT}/{model}:generateContent", body, api_key)
    out = parse_response(resp, duration)
    usage = resp.get("usageMetadata") or {}
    out.update({"model": model, "tokens": usage.get("totalTokenCount")})
    return out


# ---------------------------------------------------------------- checked against the ear

def check_words(words: dict, entry: dict) -> dict:
    """Mark each line against Escutário's own measurements; summarise the agreement."""
    lines = words.get("lines") or []
    counts = entry.get("voices") or []
    step = float(entry.get("voices_step") or 0.1)
    held = [h for h in (entry.get("held") or []) if h.get("d", 0) >= HELD_MIN_S]
    heard_n = sustain_claims = sustain_found = 0
    for i, ln in enumerate(lines):
        t = ln["t"]
        lo = max(0, int((t - VOICE_WINDOW_BEFORE_S) / step))
        hi = int((t + VOICE_WINDOW_AFTER_S) / step) + 1
        ln["ear_heard_voice"] = bool(counts) and any(c > 0 for c in counts[lo:hi])
        heard_n += ln["ear_heard_voice"]
        if SUSTAIN_WORDS.search(ln.get("delivery") or ""):
            sustain_claims += 1
            end = lines[i + 1]["t"] if i + 1 < len(lines) else t + 8
            ln["ear_held_note"] = any(t - 1 <= h["t"] <= max(end, t + 2) for h in held)
            sustain_found += ln["ear_held_note"]
    share = heard_n / len(lines) if lines else 0.0
    summary = {
        "lines": len(lines),
        "with_measured_voice": heard_n,
        "sustain_claims": sustain_claims,
        "sustain_with_held_note": sustain_found,
        "verified": bool(lines) and bool(counts) and share >= AGREEMENT_FLOOR,
    }
    return {**words, "lines": lines, "check": summary}


# ---------------------------------------------------------------- by hand

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m escutario.words", description="Hear the words of a song Escutário already heard.")
    ap.add_argument("song_dir", help="out/<slug> folder with source.wav and viz.json")
    a = ap.parse_args(argv)
    d = Path(a.song_dir)
    entry = json.loads((d / "viz.json").read_text())
    from escutario.worker import ENV_FILE, check_env_file, read_env
    check_env_file(ENV_FILE)
    key = os.environ.get("GEMINI_API_KEY") or read_env(ENV_FILE).get("GEMINI_API_KEY", "")
    words = check_words(hear_words(d / "source.wav", duration=float(entry.get("duration") or 0), api_key=key), entry)
    if (d / "words.json").is_symlink():
        (d / "words.json").unlink()
    (d / "words.json").write_text(json.dumps(words, ensure_ascii=False, indent=1))
    print(json.dumps(words["check"]), f"-> {d / 'words.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
