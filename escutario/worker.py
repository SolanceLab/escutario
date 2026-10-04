# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The listen worker — the Study Mac's side of the House's listen request line.

Run by launchd every minute (`python -m escutario.worker --once`). Each run:
  1. hears files Anne dropped into iCloud Drive/Escutário, then moves them to Heard/;
  2. claims link/path requests other rooms queued through solance-api and hears those.
Each hearing runs the full `listen` pipeline, builds the listening card and posts it to
solance-api (`POST /escutario/heard`), where every room of the House can read it.

Safety: one run at a time (file lock). Links are checked against fetch.py's host allowlist.
Paths are read only if they resolve inside iCloud Drive, Downloads, Music or Desktop and carry
an audio/video extension; anything else is refused and reported back, never read.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HTTP_TIMEOUT_S = 30
SETTLE_S = 20.0                     # a dropped file must be unchanged this long (iCloud may still be syncing it)
MAX_FILE_BYTES = 200 * 1024 * 1024
HOME = Path.home()
REPO = Path(__file__).resolve().parents[1]
ICLOUD = HOME / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
ENV_FILE = Path(os.environ.get("ESCUTARIO_ENV_FILE", str(REPO / ".env")))   # holds SOLANCE_API_BASE + the worker's own ESCUTARIO_API_KEY (platform escutario_worker, 0600, gitignored)
ALLOWED_ROOTS = [ICLOUD, HOME / "Downloads", HOME / "Music", HOME / "Desktop"]
FAILED_DIR_NAME = "Could not hear"
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus", ".mp4", ".mov", ".webm"}
HEARD_DIR_NAME = "Heard"
OUT_ROOT = REPO / "out"
DROP_DIR = ICLOUD / "Escutário"
GALLERY_TO_ADD = OUT_ROOT / ".gallery-to-add.json"   # link songs whose gallery add failed; retried every run
GALLERY_SKIPPED = OUT_ROOT / ".gallery-skipped.json"
UPLOAD_RETRY_S = 30 * 60                             # a song whose upload failed waits this long before the next try   # songs the gallery wants but whose recording is not on this Mac (logged once)
DROP_HASHES = OUT_ROOT / ".drop-hashes.json"      # sha256 of every dropped file already heard -> its song id
WORKER_NAME = "study-mac"
MAX_DROPS_PER_RUN = 3               # likewise for dropped files, so a pile of drops can't starve requests
LOCK_PATH = OUT_ROOT / ".worker.lock"
MAX_UPLOADS_PER_RUN = 3             # gallery uploads per run
MAX_REQUESTS_PER_RUN = 3            # keep one launchd run bounded; the next minute picks up the rest
DROP_PRINTS = OUT_ROOT / ".sound-prints.json"    # song id -> sound fingerprint of every song heard, to recognise the same song renamed
AUDIO_BITRATE = "160k"
USER_AGENT = "Escutario-Worker/1.0 (Study Mac)"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- config + api

def read_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None   # never follow: a redirect would carry the bearer key to another origin


_OPENER = urllib.request.build_opener(_NoRedirect)


@dataclass
class Api:
    base: str
    key: str

    def put_bytes(self, path: str, data: bytes, content_type: str) -> dict:
        req = urllib.request.Request(self.base.rstrip("/") + path, data=data, method="PUT", headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": content_type, "Content-Length": str(len(data)),
            "User-Agent": USER_AGENT})
        try:
            with _OPENER.open(req, timeout=HTTP_TIMEOUT_S * 4) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"solance-api PUT {path} -> {e.code}: {detail}") from None

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None   # bytes as budgeted, not \u-escaped
        req = urllib.request.Request(self.base.rstrip("/") + path, data=data, method=method, headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "User-Agent": USER_AGENT})
        try:
            with _OPENER.open(req, timeout=HTTP_TIMEOUT_S) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"solance-api {method} {path} -> {e.code}: {detail}") from None


def api_from_env(env: dict) -> Api:
    base = os.environ.get("SOLANCE_API_BASE") or env.get("SOLANCE_API_BASE")
    key = os.environ.get("ESCUTARIO_API_KEY") or env.get("ESCUTARIO_API_KEY")   # the worker's own key only, never a broader one
    if not base or not key:
        raise RuntimeError(f"no SOLANCE_API_BASE / ESCUTARIO_API_KEY found (env or {ENV_FILE})")
    if not base.startswith("https://"):
        raise RuntimeError("SOLANCE_API_BASE must be https")
    return Api(base, key)


def check_env_file(path: Path) -> None:
    """The key file must be a regular file owned by this user and unreadable by anyone else."""
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise RuntimeError(f"{path.name} must be a regular file, not a link")
    if st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise RuntimeError(f"{path.name} must be owned by you and private (chmod 600)")


# ---------------------------------------------------------------- helpers

_ABS_PATH = re.compile(r"(?:~|/)(?:[^\s'\"():,]+/)+([^\s'\"():,/]+)")


def redact(text: str, limit: int = 300) -> str:
    """What may leave the Mac: no home folder, no full paths (only the last name), bounded."""
    out = str(text).replace(str(HOME), "~")
    out = _ABS_PATH.sub(lambda m: m.group(1), out)
    return out[:limit]


def check_link(url: str) -> str:
    """A link request is re-checked here with Python's own URL parser, not trusted from the queue."""
    from escutario.fetch import _validate_url, FetchRefused
    if not url.startswith("https://") or any(c in url for c in "\\ \t\r\n"):
        raise ValueError("link must be a plain https address")
    try:
        _validate_url(url)
    except FetchRefused as e:
        raise ValueError(f"link refused: {redact(e)}") from None
    return url


def unique_dest(folder: Path, name: str) -> Path:
    """Never overwrite an archived file: song.wav, song (2).wav, song (3).wav ..."""
    folder.mkdir(exist_ok=True)
    target = folder / name
    stem, suffix, n = Path(name).stem, Path(name).suffix, 2
    while target.exists() or target.is_symlink():
        target = folder / f"{stem} ({n}){suffix}"
        n += 1
    return target


def make_slug(title: str, source: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40].strip("-") or "clip"
    digest = hashlib.sha1(f"{source}|{time.time_ns()}".encode()).hexdigest()[:6]
    return f"{base}-{digest}"


def check_path(raw: str) -> Path:
    """Resolve a requested Mac path and refuse anything outside the allowed folders."""
    if "\x00" in raw or "\n" in raw:
        raise ValueError("path contains a control character")
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        raise ValueError("path must be absolute or start with ~/")
    real = Path(os.path.realpath(p))
    roots = [Path(os.path.realpath(r)) for r in ALLOWED_ROOTS]
    if not any(real == r or r in real.parents for r in roots):
        raise ValueError("path is outside iCloud Drive, Downloads, Music and Desktop")
    if real.suffix.lower() not in AUDIO_EXTS:
        raise ValueError(f"not an audio or video file ({real.suffix or 'no extension'})")
    if not real.is_file():
        raise ValueError("no such file")
    if real.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("file is larger than 200 MB")
    return real


def compact_facts(entry: dict) -> dict:
    voices = entry.get("voices") or []
    sung = [c for c in voices if c > 0]
    vib = entry.get("vibrato") or {}
    return {
        "duration_s": entry.get("duration"),
        "key": (entry.get("key") or {}).get("key"),
        "tempo_bpm": (entry.get("tempo") or {}).get("bpm"),
        "sections": entry.get("sections"),
        "riffs": [{k: r.get(k) for k in ("start_s", "end_s", "notes", "low", "high", "confidence")} for r in (entry.get("riffs") or [])],
        "vibrato": {"judged": vib.get("judged"), "found": len(vib.get("with_vibrato") or [])},
        "two_voices_pct": round(100 * sum(1 for c in sung if c >= 2) / len(sung)) if sung else 0,
        "breaths": len(entry.get("breaths") or []),
        "longest_phrase_s": entry.get("longest_phrase_s"),
        "arrivals": [{"t": c.get("t"), "arrivals": c.get("arrivals")} for c in (entry.get("changes") or []) if not c.get("relabel")][:8],
    }


FACTS_WORDS_MAX_BYTES = 20 * 1024   # UTF-8 bytes; with a 7,900-character card the whole post stays under the API's 64 KB body limit


def facts_words(words: dict) -> dict:
    """Every line the model heard, as compact rows, trimmed to fit the facts budget (delivery goes first, then lines)."""
    out = {"model": words.get("model"), "check": words.get("check"), "interpretation": words.get("interpretation"),
           "lines": [[ln["t"], ln.get("voice"), ln["text"], ln.get("delivery", "")] for ln in words.get("lines") or []]}
    size = lambda: len(json.dumps(out, ensure_ascii=False).encode())   # noqa: E731
    if size() > FACTS_WORDS_MAX_BYTES:
        out["lines"] = [row[:3] for row in out["lines"]]
        out["delivery_dropped"] = True
    while out["lines"] and size() > FACTS_WORDS_MAX_BYTES:
        out["lines"].pop()
        out["lines_truncated"] = True
    return out


def gemini_key() -> str:
    return os.environ.get("GEMINI_API_KEY") or read_env(ENV_FILE).get("GEMINI_API_KEY", "")


def _regular(p: Path) -> bool:
    try:
        return stat.S_ISREG(os.lstat(p).st_mode)   # a planted symlink is not the recording that was heard
    except OSError:
        return False


def words_for_song(song: dict, *, again: bool = False, out_root: Path | None = None, words_fn=None) -> tuple[str, dict]:
    """The deliberate Google step for a song already heard. Returns (card with words, words for facts).

    Google is called at most once per song: words already on the song, or a result already saved on
    this Mac (say, when publishing failed last time), are reused unless the room asked `again`.
    Raises when the recording is gone or the model fails, so the request is reported failed.
    """
    from escutario.card import add_words_to_card
    existing = (song.get("facts") or {}).get("words")
    if existing and not again and isinstance(existing.get("lines"), list):
        return str(song.get("card") or ""), existing
    out_root = Path(out_root or OUT_ROOT).resolve()
    slug = str(song.get("slug") or "")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", slug):
        raise ValueError("the song has no usable slug")
    song_dir = out_root / slug
    if song_dir.is_symlink() or song_dir.resolve().parent != out_root:
        raise ValueError("the song's folder resolves outside out/")
    wav, viz, saved = song_dir / "source.wav", song_dir / "viz.json", song_dir / "words.json"
    if not (_regular(wav) and _regular(viz)):
        raise FileNotFoundError("the recording is no longer on the Study Mac; ask for it to be heard again first")
    entry = json.loads(viz.read_text())
    words = None
    if not again and _regular(saved):
        try:
            cached = json.loads(saved.read_text())
            if isinstance(cached, dict) and isinstance(cached.get("lines"), list) and cached.get("model"):
                words = cached
                log(f"reusing words already heard for {slug} (no second call)")
        except (OSError, ValueError):
            words = None
    if words is None:
        if words_fn is None:
            key = gemini_key()
            if not key:
                raise RuntimeError("no GEMINI_API_KEY on the Mac")
            from escutario.words import check_words, hear_words

            def words_fn(path, duration):
                return check_words(hear_words(path, duration=duration, api_key=key), entry)
        words = words_fn(wav, float(entry.get("duration") or song.get("duration_s") or 0))
        if saved.is_symlink():
            saved.unlink()
        tmp = saved.with_name(f".words.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(words, ensure_ascii=False, indent=1))
        os.replace(tmp, saved)
    return add_words_to_card(str(song.get("card") or ""), words), facts_words(words)


def hear(source: str, *, title: str, artist: str | None, display_source: str, ask: str | None = None, listen_fn=None, card_fn=None) -> dict:
    """Run the pipeline on a link or an already-checked file path; return the heard payload. Nothing leaves the Mac."""
    if listen_fn is None:
        from escutario.listen import listen as listen_fn
    if card_fn is None:
        from escutario.card import build_card as card_fn
    slug = make_slug(title or Path(source).stem, source)
    res = listen_fn(source, slug, track=title or None, artist=artist, out_root=OUT_ROOT, viz_path=None)
    entry = res.viz_entry
    notes = ([f"asked to listen for: {ask}"] if ask else []) + [redact(n, 200) for n in list(res.notes)[:3]]
    card = card_fn(entry, title=title, artist=artist or "", source=display_source, notes=notes)
    return {"slug": slug, "title": title or None, "artist": artist, "duration_s": entry.get("duration"),
            "card": card[:8000], "facts": compact_facts(entry)}


# ---------------------------------------------------------------- the two doors

def settled_drops(drop_dir: Path, now: float | None = None) -> list[Path]:
    if not drop_dir.is_dir():
        return []
    now = time.time() if now is None else now
    out = []
    if drop_dir.is_symlink():
        return []
    for p in sorted(drop_dir.iterdir()):
        try:
            st = os.lstat(p)
        except OSError:
            continue
        if not stat.S_ISREG(st.st_mode):   # no symlinks, folders, pipes or devices
            continue
        if not p.name.startswith(".") and p.suffix.lower() in AUDIO_EXTS and now - st.st_mtime >= SETTLE_S:
            out.append(p)
    return out[:MAX_DROPS_PER_RUN]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json_map(path: Path) -> dict:
    try:
        data = json.loads(path.read_text()) if path.is_file() else {}
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json_map(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def _sound_print(path: Path):
    """(bits, duration) of a recording."""
    from escutario import fingerprint as fp
    return fp.fingerprint_file(path), fp.duration_of(path)


def already_heard(path: Path, *, hashes_path: Path, prints_path: Path, print_fn=None):
    """(song id, how, sha256, print) when this recording was heard before — same bytes, or same sound under
    any name or container — else (None, None, sha256, print)."""
    from escutario import fingerprint as fp
    digest = file_sha256(path)
    by_bytes = _read_json_map(hashes_path).get(digest)
    if by_bytes:
        return by_bytes, "same file", digest, None
    try:
        bits, duration = (print_fn or _sound_print)(path)
    except Exception as e:  # noqa: BLE001 — an unreadable print must not stop a hearing; hearing reports a bad file itself
        log(f"could not take the sound print of {path.name}: {redact(e)}")
        return None, None, digest, None
    match = fp.find_same(bits, _read_json_map(prints_path), duration=duration)
    if match:
        return match[0], f"same sound and length ({match[1]:.2f} of bits differ)", digest, (bits, duration)
    return None, None, digest, (bits, duration)


def remember_heard(song_id: str, *, digest: str | None, bits, title: str, hashes_path: Path, prints_path: Path) -> None:
    from escutario import fingerprint as fp
    if digest:
        m = _read_json_map(hashes_path)
        m[digest] = song_id
        _write_json_map(hashes_path, m)
    if bits is not None:
        sound, duration = bits if isinstance(bits, tuple) else (bits, None)
        if sound is not None and len(sound):
            m = _read_json_map(prints_path)
            m[song_id] = {**fp.pack(sound, duration), "title": title[:200]}
            _write_json_map(prints_path, m)


def process_drops(api: Api, drop_dir: Path = DROP_DIR, hear_fn=hear, hashes_path: Path | None = None,
                  prints_path: Path | None = None, print_fn=None) -> int:
    """Hear what Anne dropped. The same song dropped again, under any name or in another container, is
    recognised (by its bytes, else by its sound) and set aside in Heard/ with a note, never heard twice."""
    hashes_path = hashes_path or DROP_HASHES
    prints_path = prints_path or DROP_PRINTS
    print_fn = print_fn or _sound_print
    done = 0
    for f in settled_drops(drop_dir):
        title = f.stem
        try:
            if f.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("file is larger than 200 MB")
            song_id, how, digest, bits = already_heard(f, hashes_path=hashes_path, prints_path=prints_path, print_fn=print_fn)
            if song_id:
                dest = unique_dest(drop_dir / HEARD_DIR_NAME, f.name)
                shutil.move(str(f), str(dest))
                known_title = (_read_json_map(prints_path).get(song_id) or {}).get("title") or "a song already on the shelf"
                with open(dest.with_name(dest.name + ".already-heard.txt"), "x") as note:   # exclusive: never write through a planted file
                    note.write(f"Already heard: {known_title} (song {song_id}). Found by {how}. It was not heard again.\n")
                log(f"drop {f.name} is already heard as song {song_id} ({how}); not heard twice")
                continue
            payload = hear_fn(str(f), title=title, artist=None, display_source=f"iCloud Drive/Escutário/{f.name}")
            payload["facts"] = {**(payload.get("facts") or {}), "sha256": digest}
            heard = api.call("POST", "/escutario/heard", {"origin": "drop", "source": f"iCloud Drive/Escutário/{f.name}", **payload})
            if heard.get("id"):
                remember_heard(heard["id"], digest=digest, bits=bits, title=title, hashes_path=hashes_path, prints_path=prints_path)
            shutil.move(str(f), str(unique_dest(drop_dir / HEARD_DIR_NAME, f.name)))
            log(f"heard drop {f.name} -> {heard.get('id')}")
            done += 1
        except Exception as e:  # noqa: BLE001 — one bad file must not stop the others
            try:
                dest = unique_dest(drop_dir / FAILED_DIR_NAME, f.name)
                shutil.move(str(f), str(dest))
                with open(dest.with_name(dest.name + ".why.txt"), "x") as why:   # exclusive: never write through a planted file
                    why.write(f"{type(e).__name__}: {redact(e, 1000)}\n")
            except Exception as e2:  # noqa: BLE001
                log(f"could not set aside {f.name}: {e2}")
            log(f"could not hear drop {f.name}: {redact(e)}")
    return done


class AlreadyHeard(Exception):
    pass


def process_requests(api: Api, hear_fn=hear, limit: int = MAX_REQUESTS_PER_RUN, words_fn=None) -> int:
    words_fn = words_fn or words_for_song
    done = 0
    for _ in range(limit):
        req = (api.call("POST", "/escutario/claim", {"worker": WORKER_NAME}) or {}).get("request")
        if not req:
            break
        rid, kind, source = req["id"], req["kind"], req["source"]
        token = req.get("claimed_at")   # the claim token: only this claim may close the request
        try:
            if kind == "words":
                if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", str(source)):
                    raise ValueError("a words request must name a heard song id")
                song = (api.call("GET", f"/escutario/heard?id={source}") or {}).get("heard") or {}
                card, words = words_fn(song, again=(req.get("note") == "again"))
                out = api.call("POST", "/escutario/words-heard", {"request_id": rid, "claimed_at": token, "card": card, "words": words})
                if out.get("request_done") is False:
                    log(f"WARNING words for song {source} were written but request {rid} was no longer this worker's claim")
                log(f"words heard for song {source} (request {rid})")
                done += 1
                continue
            if kind == "path":
                real = check_path(source)
                target, display = str(real), f"Mac file {real.name}"
                title = req.get("title") or real.stem
                song_id, how, digest, bits = already_heard(real, hashes_path=DROP_HASHES, prints_path=DROP_PRINTS)
                if song_id:
                    raise AlreadyHeard(f"this recording was already heard as song {song_id} ({how}); read it with escutario_heard")
            elif kind == "link":
                target = check_link(source)
                display, title = target, req.get("title") or ""
            else:
                raise ValueError(f"unknown request kind {kind!r}")
            payload = hear_fn(target, title=title, artist=req.get("artist"), display_source=display, ask=req.get("note"))
            heard = api.call("POST", "/escutario/heard", {"request_id": rid, "claimed_at": token, "origin": kind, "source": display, **payload})
            log(f"heard request {rid} ({kind})")
            if heard.get("id"):
                # every heard song leaves its sound print, so the same song dropped later is recognised
                try:
                    wav = OUT_ROOT / str(payload.get("slug") or "") / "source.wav"
                    if kind == "path":
                        remember_heard(heard["id"], digest=digest, bits=bits, title=title, hashes_path=DROP_HASHES, prints_path=DROP_PRINTS)
                    elif _regular(wav):
                        remember_heard(heard["id"], digest=None, bits=_sound_print(wav), title=title or "", hashes_path=DROP_HASHES, prints_path=DROP_PRINTS)
                except Exception as e:  # noqa: BLE001 — the card is safe; only duplicate detection for this song is lost
                    log(f"could not remember the sound of song {heard['id']}: {redact(e)}")
            if kind == "link" and heard.get("id"):
                # a song from a link joins the gallery by itself; a Mac file waits for Anne to add it
                try:
                    api.call("POST", f"/escutario/songs/{heard['id']}/gallery", {})
                except Exception as e:  # noqa: BLE001 — the card is already safe; the next run adds it
                    log(f"could not put song {heard['id']} on the gallery yet (will retry): {redact(e)}")
                    todo = _read_json_map(GALLERY_TO_ADD)
                    todo[heard["id"]] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    _write_json_map(GALLERY_TO_ADD, todo)
            done += 1
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {redact(e, 900)}"
            log(f"could not hear request {rid}: {msg}")
            try:
                api.call("POST", "/escutario/fail", {"request_id": rid, "claimed_at": token, "error": msg})
            except Exception as e2:  # noqa: BLE001
                log(f"could not report the failure for {rid}: {e2}")
    return done


def encode_gallery_audio(wav: Path) -> bytes:
    """AAC in an M4A box, the one format every browser (Safari included) plays from a blob."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "audio.m4a"
        done = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav), "-vn", "-ac", "2",
                               "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-movflags", "+faststart", str(out)],
                              capture_output=True, timeout=300)
        if done.returncode != 0 or not out.is_file():
            raise RuntimeError("could not encode the gallery audio")
        return out.read_bytes()


def process_gallery(api: Api, out_root: Path | None = None, encode=encode_gallery_audio, skipped_path: Path | None = None,
                    limit: int = MAX_UPLOADS_PER_RUN) -> int:
    """Put the audio and score of every song added to the gallery on the House's private shelf."""
    out_root = Path(out_root or OUT_ROOT).resolve()
    skipped_path = skipped_path or GALLERY_SKIPPED
    to_add = _read_json_map(GALLERY_TO_ADD)
    for sid in list(to_add):
        try:
            api.call("POST", f"/escutario/songs/{sid}/gallery", {})
            to_add.pop(sid)
        except Exception as e:  # noqa: BLE001
            log(f"still could not put song {sid} on the gallery: {redact(e)}")
    if to_add or GALLERY_TO_ADD.exists():
        _write_json_map(GALLERY_TO_ADD, to_add)
    pending = (api.call("GET", "/escutario/gallery/pending") or {}).get("songs") or []
    skipped = _read_json_map(skipped_path)
    now = time.time()
    # songs whose recording is gone, or whose upload failed recently, never block the ones behind them
    ready = [s for s in pending if str(s.get("id")) not in skipped
             or (isinstance(skipped.get(str(s.get("id"))), (int, float)) and now >= skipped[str(s.get("id"))])]
    done = 0
    for song in ready[:limit]:
        sid, slug, needs = str(song.get("id")), str(song.get("slug") or ""), list(song.get("needs") or [])
        song_dir = out_root / slug
        wav, viz = song_dir / "source.wav", song_dir / "viz.json"
        if (not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", slug) or song_dir.is_symlink()
                or song_dir.resolve().parent != out_root or not (_regular(wav) and _regular(viz))):
            skipped = _read_json_map(skipped_path)
            if sid not in skipped:
                log(f"gallery wants song {sid} but its recording is not on this Mac; ask for it to be heard again")
                skipped[sid] = time.strftime("%Y-%m-%dT%H:%M:%S")   # a date string: skipped for good, logged once
                _write_json_map(skipped_path, skipped)
            continue
        try:
            if "score" in needs:
                api.put_bytes(f"/escutario/songs/{sid}/media/score", viz.read_bytes(), "application/json")
            if "audio" in needs:
                api.put_bytes(f"/escutario/songs/{sid}/media/audio", encode(wav), "audio/mp4")
            log(f"song {sid} is on the gallery ({', '.join(needs)} uploaded)")
            done += 1
        except Exception as e:  # noqa: BLE001 — tried again after a cooldown, so it cannot block the queue
            log(f"could not upload song {sid} to the gallery (retrying in 30 min): {redact(e)}")
            skipped = _read_json_map(skipped_path)
            skipped[sid] = time.time() + UPLOAD_RETRY_S
            _write_json_map(skipped_path, skipped)
    return done


def run_once() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "a") as lock:   # 'a' never truncates
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0  # a hearing is already running; this minute's run steps aside
        DROP_DIR.mkdir(parents=True, exist_ok=True)
        check_env_file(ENV_FILE)
        api = api_from_env(read_env(ENV_FILE))
        n = process_drops(api)
        n += process_requests(api)
        n += process_gallery(api)
        return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Escutário listen worker (run by launchd)")
    ap.add_argument("--once", action="store_true", help="process what is waiting, then exit")
    ap.parse_args(argv)
    try:
        run_once()
        return 0
    except Exception:  # noqa: BLE001
        log("worker run failed:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
