# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""View — the listening score for any song Escutário has heard, on your own machine.

    python -m escutario.view                  # opens http://127.0.0.1:8765 in your browser
    python -m escutario.view --port 9000 --no-open

A small local server, standard library only. It lists the songs in out/, serves each one's score
(viz.json), audio (source.wav, seekable) and heard words (words.json, when the words step ran), and
keeps your marks in out/<song>/marks.json. Nothing leaves the machine.

It listens on 127.0.0.1 only and answers only requests addressed to that host and port, so a web page
elsewhere cannot reach it through a renamed address (DNS rebinding); writes must also come from the
viewer's own page (Origin check).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

DEFAULT_PORT = 8765
AUDIO_CHUNK = 256 * 1024
WHAT = {"words", "voice", "melody", "unnamed"}
MAX_MEASURED = 20                     # measurement lines kept beside a mark
STATIC = Path(__file__).resolve().parent / "viewer"
STATIC_FILES = {                      # the only files the page is made of
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "score.js": "text/javascript; charset=utf-8",
    "now.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "out"
MAX_BODY = 64 * 1024                  # one mark is a few hundred bytes
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")   # same rule listen uses for a song folder
MAX_MARKS = 500                       # per song
MAX_WORDS = 2000                      # characters of your own words on a mark

_marks_lock = threading.Lock()


# ---------------------------------------------------------------- songs on disk

def song_dir(slug: str, out_root: Path = OUT_ROOT) -> Path | None:
    """The folder for a slug, only if it is a heard song directly under out/."""
    if not SLUG_RE.fullmatch(slug or ""):
        return None
    d = (out_root / slug).resolve()
    if d.parent != out_root.resolve() or not (d / "viz.json").is_file():
        return None
    return d


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def song_meta(d: Path) -> dict:
    listen = _read_json(d / "listen.json") or {}
    fetch = _read_json(d / "fetch.json") or {}
    title = listen.get("track") or fetch.get("title") or d.name
    artist = listen.get("artist") or fetch.get("artist")
    viz = _read_json(d / "viz.json") or {}
    marks = read_marks(d)
    return {
        "slug": d.name,
        "title": str(title)[:200],
        "artist": str(artist)[:200] if artist else None,
        "duration_s": viz.get("duration") if isinstance(viz.get("duration"), (int, float)) else None,
        "has_audio": (d / "source.wav").is_file(),
        "has_words": (d / "words.json").is_file(),
        "marks_count": sum(1 for m in marks if m.get("kind") != "preview"),
        "heard_at": time.strftime("%Y-%m-%d", time.localtime((d / "viz.json").stat().st_mtime)),
    }


def list_songs(out_root: Path = OUT_ROOT) -> list[dict]:
    if not out_root.is_dir():
        return []
    songs = [song_meta(d) for d in sorted(out_root.iterdir()) if d.is_dir() and song_dir(d.name, out_root)]
    return sorted(songs, key=lambda s: s["heard_at"], reverse=True)


# ---------------------------------------------------------------- marks

def read_marks(d: Path) -> list[dict]:
    m = _read_json(d / "marks.json")
    return [x for x in m if isinstance(x, dict)] if isinstance(m, list) else []


def _write_marks(d: Path, marks: list[dict]) -> None:
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".marks", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(marks, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())   # a power cut must not leave an empty marks.json that the next save would trust
        os.chmod(tmp, 0o644)
        os.replace(tmp, d / "marks.json")
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _num(v, lo: float, hi: float) -> float | None:
    return round(float(v), 2) if isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi else None


def clean_mark(body, duration: float) -> tuple[dict | None, str | None]:
    """A new mark from the page, checked; (mark, None) or (None, reason)."""
    if not isinstance(body, dict):
        return None, "a mark is a JSON object"
    t = _num(body.get("t"), 0, duration)
    if t is None:
        return None, "t must be a second inside the song"
    t_end = None
    if body.get("t_end") is not None:
        t_end = _num(body.get("t_end"), 0, duration)
        if t_end is None or t_end <= t:
            return None, "t_end must come after t, inside the song"
    what = body.get("what")
    if not isinstance(what, list) or not what or not all(w in WHAT for w in what):
        return None, f"what must name at least one of {sorted(WHAT)}"
    words = body.get("words")
    if words is not None and not isinstance(words, str):
        return None, "words must be text"
    measured = body.get("measured") or []
    if not isinstance(measured, list) or not all(isinstance(x, str) for x in measured):
        return None, "measured must be a list of text"
    return {
        "id": uuid.uuid4().hex[:12],
        "kind": "section" if t_end is not None else "moment",
        "t": t, "t_end": t_end,
        "what": sorted(set(what)),
        "words": (words.strip()[:MAX_WORDS] or None) if words else None,
        "measured": [x[:300] for x in measured[:MAX_MEASURED]],
        "marked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, None


def add_mark(d: Path, body, duration: float) -> tuple[dict | None, str | None]:
    mark, err = clean_mark(body, duration)
    if err:
        return None, err
    with _marks_lock:
        marks = read_marks(d)
        if len(marks) >= MAX_MARKS:
            return None, f"this song already holds {MAX_MARKS} marks"
        marks.append(mark)
        marks.sort(key=lambda m: float(m.get("t", 0)))
        _write_marks(d, marks)
    return mark, None


def update_mark(d: Path, mark_id: str, body) -> tuple[dict | None, str | None]:
    if not isinstance(body, dict) or not (body.get("words") is None or isinstance(body.get("words"), str)):
        return None, "send {\"words\": text or null}"
    with _marks_lock:
        marks = read_marks(d)
        for m in marks:
            if m.get("id") == mark_id:
                w = body.get("words")
                m["words"] = (w.strip()[:MAX_WORDS] or None) if w else None
                _write_marks(d, marks)
                return m, None
    return None, "no such mark"


def delete_mark(d: Path, mark_id: str) -> bool:
    with _marks_lock:
        marks = read_marks(d)
        kept = [m for m in marks if m.get("id") != mark_id]
        if len(kept) == len(marks):
            return False
        _write_marks(d, kept)
    return True


# ---------------------------------------------------------------- the server

class Handler(BaseHTTPRequestHandler):
    server_version = "Escutario"
    sys_version = ""
    out_root: Path = OUT_ROOT
    timeout = 10   # a client that stalls mid-request releases its thread

    def log_message(self, fmt, *args):   # quiet: one line per request is noise for a local viewer
        pass

    # -- guards
    def _host_ok(self) -> bool:
        # another website may embed this address (an <audio> tag), so refuse anything a browser marks cross-site
        if (self.headers.get("Sec-Fetch-Site") or "").lower() == "cross-site":
            return False
        host = (self.headers.get("Host") or "").lower()
        port = self.server.server_address[1]
        return host in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        port = self.server.server_address[1]
        return origin in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    # -- replies
    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' data:; media-src 'self'; style-src 'self'; "
                         "font-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, obj):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _error(self, status: int, reason: str):
        self._json(status, {"error": reason})

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if n <= 0 or n > MAX_BODY:
            return None
        try:
            return json.loads(self.rfile.read(n))
        except ValueError:
            return None

    def _route(self):
        parts = [unquote(p) for p in urlsplit(self.path).path.split("/") if p]
        return parts

    # -- GET
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._host_ok():
            return self._error(HTTPStatus.FORBIDDEN, "this viewer only answers at 127.0.0.1")
        p = self._route()
        if not p:
            p = ["index.html"]
        if len(p) == 1 and p[0] in STATIC_FILES:
            return self._send(200, (STATIC / p[0]).read_bytes(), STATIC_FILES[p[0]])
        if p == ["api", "songs"]:
            return self._json(200, {"songs": list_songs(self.out_root)})
        if len(p) >= 3 and p[:2] == ["api", "songs"]:
            d = song_dir(p[2], self.out_root)
            if d is None:
                return self._error(404, "no heard song by that name")
            if len(p) == 3:
                return self._json(200, {
                    "song": song_meta(d),
                    "score": _read_json(d / "viz.json"),
                    "words": _read_json(d / "words.json"),
                    "marks": read_marks(d),
                })
            if len(p) == 4 and p[3] == "audio":
                return self._audio(d / "source.wav")
        return self._error(404, "not here")

    def _audio(self, path: Path):
        if not path.is_file():
            return self._error(404, "this song's audio is not on this machine")
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        status = 200
        if rng:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if not m or (not m.group(1) and not m.group(2)):
                return self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
            if len(m.group(1)) > 18 or len(m.group(2)) > 18:   # no file here is that big
                return self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:   # suffix range: the last N bytes
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                return self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = length
            try:
                while left > 0:
                    chunk = f.read(min(AUDIO_CHUNK, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass   # the browser stopped reading (a seek); nothing to clean up

    # -- writes
    def _write_guard(self):
        if not self._host_ok() or not self._origin_ok():
            self._error(HTTPStatus.FORBIDDEN, "marks can only be written from the viewer's own page")
            return None
        p = self._route()
        if len(p) < 4 or p[:2] != ["api", "songs"] or p[3] != "marks":
            self._error(404, "not here")
            return None
        d = song_dir(p[2], self.out_root)
        if d is None:
            self._error(404, "no heard song by that name")
            return None
        return d, p

    def do_POST(self):
        g = self._write_guard()
        if not g:
            return
        d, p = g
        if len(p) != 4:
            return self._error(404, "not here")
        body = self._body()
        if body is None:
            return self._error(400, "send one JSON mark")
        duration = (_read_json(d / "viz.json") or {}).get("duration") or 0
        mark, err = add_mark(d, body, float(duration))
        if err:
            return self._error(400, err)
        self._json(201, {"mark": mark})

    def do_PATCH(self):
        g = self._write_guard()
        if not g:
            return
        d, p = g
        if len(p) != 5:
            return self._error(404, "not here")
        mark, err = update_mark(d, p[4], self._body())
        if err:
            return self._error(404 if err == "no such mark" else 400, err)
        self._json(200, {"mark": mark})

    def do_DELETE(self):
        g = self._write_guard()
        if not g:
            return
        d, p = g
        if len(p) != 5:
            return self._error(404, "not here")
        if not delete_mark(d, p[4]):
            return self._error(404, "no such mark")
        self._json(200, {"deleted": p[4]})


def make_server(port: int = DEFAULT_PORT, out_root: Path = OUT_ROOT) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"out_root": Path(out_root)})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m escutario.view", description="Open the listening score for the songs Escutário has heard.")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--out-root", default=str(OUT_ROOT), help="parent of the per-song folders (default: out/)")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    a = ap.parse_args(argv)
    try:
        srv = make_server(a.port, Path(a.out_root))
    except OSError as e:
        print(f"could not listen on 127.0.0.1:{a.port} ({e.strerror}); try --port", file=sys.stderr)
        return 2
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"Escutário viewer at {url}  ({len(list_songs(Path(a.out_root)))} songs) — Ctrl+C to stop")
    if not a.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
