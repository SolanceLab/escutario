# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Escutário — fetch.py: bringing audio in from an allowed source.

Sources are either an allowlisted https URL (fetched via the system yt-dlp)
or a local file path. Nothing else is trusted: a lookalike host, a non-https
scheme, embedded userinfo, or a non-default port is refused before any
subprocess ever runs.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

# --- external binaries ------------------------------------------------------

YT_DLP_PATH = "/opt/homebrew/bin/yt-dlp"  # system install, per CONTRACT.md — not resolved via PATH
FFPROBE_PATH = "/opt/homebrew/bin/ffprobe"  # sibling of the ffmpeg audio.py already depends on

# --- policy constants (starting values, calibrate later) -------------------

SEARCH_CANDIDATES = 5  # ytsearchN for the spotify-pointer search, per CONTRACT.md
MAX_DURATION_S = 600  # hard cap on source duration; refuse anything longer, per CONTRACT.md
PROBE_TIMEOUT_S = 15  # ffprobe timeout for local files; starting value to calibrate
DURATION_MISMATCH_THRESHOLD_S = 3.0  # spotify-pointer duration flag threshold, per CONTRACT.md
DOWNLOAD_TIMEOUT_S = 300  # yt-dlp download timeout; starting value to calibrate
MAX_FILESIZE = "50M"  # yt-dlp --max-filesize cap, per CONTRACT.md
METADATA_TIMEOUT_S = 30  # yt-dlp --dump-json should be quick; starting value to calibrate

# host -> platform name. Subdomains of these are accepted (www., m., vt., ...);
# lookalikes (youtube.com.evil.net, notyoutube.com) are not, by construction below.
ALLOWED_HOSTS: dict[str, str] = {
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "music.youtube.com": "youtube",
    "tiktok.com": "tiktok",
    "instagram.com": "instagram",
    "soundcloud.com": "soundcloud",
    "bandcamp.com": "bandcamp",
}

SHORT_FORM_ALWAYS = {"tiktok", "instagram"}  # platforms that are short-form regardless of path

SHORT_FORM_NOTE = (
    "short-form source — may be sped up, pitched or cut; "
    "pitch and tempo describe this edit, not the original."
)


class FetchRefused(Exception):
    """The source failed policy: bad scheme/host/port/userinfo, or over the duration cap.

    Distinct from FetchError (below): a refusal means we chose not to try,
    not that an attempt failed.
    """


class FetchError(Exception):
    """yt-dlp or ffprobe itself failed: timeout, bad extractor result, no output, etc."""


@dataclass
class FetchResult:
    path: Path
    source_url: str
    title: str | None
    artist: str | None
    duration_s: float | None
    platform: str
    short_form: bool
    notes: list[str] = field(default_factory=list)


# --- host / URL policy -------------------------------------------------------


def _match_host(host: str) -> str | None:
    """Return the platform name if host is an allowed host or a subdomain of one."""
    host = host.lower()
    for allowed, platform in ALLOWED_HOSTS.items():
        if host == allowed or host.endswith("." + allowed):
            return platform
    return None


def _validate_url(url: str) -> tuple[str, str]:
    """Return (platform, matched_host), or raise FetchRefused with a clear reason."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise FetchRefused(f"refused: URL scheme must be https, got {parsed.scheme!r} ({url})")
    if parsed.username or parsed.password:
        raise FetchRefused(f"refused: URLs with embedded userinfo are not allowed ({url})")
    if parsed.port is not None and parsed.port != 443:
        raise FetchRefused(f"refused: non-default port {parsed.port} is not allowed ({url})")
    host = parsed.hostname
    if not host:
        raise FetchRefused(f"refused: could not parse a host from {url!r}")
    platform = _match_host(host)
    if platform is None:
        raise FetchRefused(f"refused: host {host!r} is not on the allowlist ({url})")
    return platform, host


def _is_short_form(platform: str, url: str) -> bool:
    if platform in SHORT_FORM_ALWAYS:
        return True
    if platform == "youtube":
        return "/shorts/" in urlparse(url).path
    return False


# --- subprocess plumbing (yt-dlp / ffprobe) ---------------------------------


def _run(cmd: list[str], *, timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise FetchError(f"timed out after {timeout}s: {' '.join(cmd)}") from exc
    except FileNotFoundError as exc:
        raise FetchError(f"binary not found: {cmd[0]}") from exc


def _yt_dlp_metadata(url: str) -> dict:
    """Metadata only — --dump-json simulates and never downloads."""
    proc = _run(
        [YT_DLP_PATH, "--dump-json", "--no-playlist", "--no-warnings", "--ignore-config", url],
        timeout=METADATA_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise FetchError(f"yt-dlp metadata lookup failed for {url}: {proc.stderr.strip()[:300]}")
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        raise FetchError(f"yt-dlp returned no metadata for {url}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise FetchError(f"yt-dlp metadata was not valid JSON for {url}") from exc


def _yt_dlp_search(query: str, n: int) -> list[dict]:
    proc = _run(
        [YT_DLP_PATH, "--dump-json", "--flat-playlist", "--no-warnings", "--ignore-config", f"ytsearch{n}:{query}"],
        timeout=METADATA_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise FetchError(f"yt-dlp search failed for {query!r}: {proc.stderr.strip()[:300]}")
    results: list[dict] = []
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not results:
        raise FetchError(f"yt-dlp search returned no usable results for {query!r}")
    return results


def _yt_dlp_download(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        YT_DLP_PATH,
        "-x", "--audio-format", "wav",
        "--no-playlist",
        "--max-filesize", MAX_FILESIZE,
        "--no-warnings", "--ignore-config",
        "--restrict-filenames", "--windows-filenames",
        "--print", "after_move:filepath",
        "-o", str(out_dir / "%(id)s.%(ext)s"),
        "--", url,
    ]
    proc = _run(cmd, timeout=DOWNLOAD_TIMEOUT_S)
    if proc.returncode != 0:
        raise FetchError(f"yt-dlp download failed for {url}: {proc.stderr.strip()[:300]}")
    lines = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        raise FetchError(f"yt-dlp download produced no output path for {url}")
    path = Path(lines[-1])
    if not path.exists():
        raise FetchError(f"yt-dlp reported {path} but it does not exist on disk")
    try:
        path.resolve().relative_to(out_dir.resolve())
    except ValueError:
        raise FetchError(f"yt-dlp wrote {path} outside {out_dir}; refusing it") from None
    return path


def _ffprobe_duration(path: Path) -> float | None:
    """Best-effort duration probe for a local file. None if ffprobe can't tell us."""
    cmd = [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)]
    try:
        proc = _run(cmd, timeout=PROBE_TIMEOUT_S)
    except FetchError:
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
        return float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


# --- public API ---------------------------------------------------------------


def fetch(
    source: str,
    out_dir: Path,
    *,
    track: str | None = None,
    artist: str | None = None,
    expected_duration_s: float | None = None,
) -> FetchResult:
    """Fetch audio from an allowlisted URL or a local file path.

    track/artist are used as metadata fallbacks when the source itself
    doesn't carry them (a local file has no uploader/title). expected_duration_s,
    when given, is checked against the fetched duration and a duration_mismatch
    note is added if they differ by more than DURATION_MISMATCH_THRESHOLD_S —
    the same check the Spotify pointer flow (search, below) uses to pick a
    candidate, offered here too in case a caller already knows what to expect.
    """
    out_dir = Path(out_dir)
    parsed = urlparse(source)

    if parsed.scheme in ("http", "https"):
        # http is routed here too, on purpose: _validate_url refuses it with a
        # clear "must be https" reason instead of it being silently treated
        # as a (nonexistent) local path named "http://...".
        return _fetch_url(source, out_dir, track=track, artist=artist, expected_duration_s=expected_duration_s)

    if parsed.scheme and parsed.scheme != "file":
        raise FetchRefused(f"refused: unsupported URL scheme {parsed.scheme!r} ({source})")

    local_path = Path(parsed.path) if parsed.scheme == "file" else Path(source)
    return _fetch_local(local_path, track=track, artist=artist, expected_duration_s=expected_duration_s)


def _duration_mismatch_note(actual_s: float, expected_s: float, label: str) -> str | None:
    mismatch = abs(actual_s - expected_s)
    if mismatch > DURATION_MISMATCH_THRESHOLD_S:
        return (
            f"duration_mismatch: {label} {actual_s:.1f}s vs expected {expected_s:.1f}s "
            f"(off by {mismatch:.1f}s)"
        )
    return None


def _fetch_url(
    url: str,
    out_dir: Path,
    *,
    track: str | None,
    artist: str | None,
    expected_duration_s: float | None,
) -> FetchResult:
    platform, _host = _validate_url(url)
    meta = _yt_dlp_metadata(url)

    duration = meta.get("duration")
    if duration is not None and (not math.isfinite(duration) or duration > MAX_DURATION_S):
        raise FetchRefused(f"refused: source is {duration:.0f}s, over the {MAX_DURATION_S}s cap ({url})")

    notes: list[str] = []
    short_form = _is_short_form(platform, meta.get("webpage_url") or url)
    if short_form:
        notes.append(SHORT_FORM_NOTE)
    if expected_duration_s is not None and duration is not None:
        note = _duration_mismatch_note(float(duration), expected_duration_s, "fetched")
        if note:
            notes.append(note)

    path = _yt_dlp_download(url, out_dir)

    return FetchResult(
        path=path,
        source_url=meta.get("webpage_url") or url,
        title=meta.get("title") or track,
        artist=meta.get("uploader") or meta.get("artist") or artist,
        duration_s=float(duration) if duration is not None else None,
        platform=platform,
        short_form=short_form,
        notes=notes,
    )


def _fetch_local(
    path: Path,
    *,
    track: str | None,
    artist: str | None,
    expected_duration_s: float | None,
) -> FetchResult:
    if not path.exists() or not path.is_file():
        raise FetchRefused(f"refused: local file not found: {path}")

    notes: list[str] = []
    duration = _ffprobe_duration(path)
    if duration is None:
        notes.append("duration not verified — ffprobe could not read this file")
    else:
        if not math.isfinite(duration) or duration > MAX_DURATION_S:
            raise FetchRefused(f"refused: local file is {duration:.0f}s, over the {MAX_DURATION_S}s cap ({path})")
        if expected_duration_s is not None:
            note = _duration_mismatch_note(duration, expected_duration_s, "file")
            if note:
                notes.append(note)

    return FetchResult(
        path=path,
        source_url=str(path),
        title=track or path.stem,
        artist=artist,
        duration_s=duration,
        platform="local",
        short_form=False,
        notes=notes,
    )


def search(track: str, artist: str | None, expected_duration_s: float | None, out_dir: Path) -> FetchResult:
    """Resolve a Spotify pointer (track/artist/duration — never audio) via YouTube search.

    NOTE on the interface: CONTRACT.md gives this signature as
    `search(track, artist, expected_duration_s) -> FetchResult`, with no
    out_dir. That can't work as written — the function ends in a download and
    has to write the file somewhere, and every other write path in this
    contract (fetch(), split()) takes an explicit out_dir rather than
    inventing one. I added out_dir as a fourth parameter rather than picking
    a hidden default directory; flagged in my report as a deliberate,
    minimal interface extension, not a silent deviation.
    """
    out_dir = Path(out_dir)
    query = f"{track} {artist}".strip() if artist else track
    candidates = _yt_dlp_search(query, SEARCH_CANDIDATES)

    def _cand_duration(c: dict) -> float:
        d = c.get("duration")
        return float(d) if d is not None else float("inf")

    if expected_duration_s is not None:
        best = min(candidates, key=lambda c: abs(_cand_duration(c) - expected_duration_s))
    else:
        timed = [c for c in candidates if c.get("duration") is not None]
        best = timed[0] if timed else candidates[0]

    duration = best.get("duration")
    if duration is not None and duration > MAX_DURATION_S:
        raise FetchRefused(f"refused: closest search match is {duration:.0f}s, over the {MAX_DURATION_S}s cap")

    video_id = best.get("id")
    raw_url = best.get("webpage_url") or best.get("url") or ""
    if raw_url and urlparse(raw_url).scheme:
        url = raw_url
    elif video_id:
        url = f"https://www.youtube.com/watch?v={video_id}"
    elif raw_url:
        # some yt-dlp versions put a bare id in "url" for flat-playlist search results
        url = f"https://www.youtube.com/watch?v={raw_url}"
    else:
        raise FetchError(f"search result had no usable id/url: {best!r}")

    # Search results should always be youtube — re-validate rather than trust blindly.
    _validate_url(url)

    notes: list[str] = []
    short_form = _is_short_form("youtube", url)
    if short_form:
        notes.append(SHORT_FORM_NOTE)
    if expected_duration_s is not None and duration is not None:
        note = _duration_mismatch_note(float(duration), expected_duration_s, "best match")
        if note:
            notes.append(note)

    path = _yt_dlp_download(url, out_dir)

    return FetchResult(
        path=path,
        source_url=url,
        title=best.get("title") or track,
        artist=best.get("uploader") or artist,
        duration_s=float(duration) if duration is not None else None,
        platform="youtube",
        short_form=short_form,
        notes=notes,
    )
