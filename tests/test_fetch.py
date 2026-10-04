# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Tests for escutario/fetch.py.

All subprocess.run calls (yt-dlp, ffprobe) are mocked — no network, ever.
The one exception is the smoke test at the bottom, which exercises only the
pure URL-validation helpers (no subprocess involved either).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from escutario import fetch


# --- fake subprocess.run plumbing ------------------------------------------


def _cmd_kind(cmd: list[str]) -> str:
    if cmd[0] == fetch.FFPROBE_PATH:
        return "ffprobe"
    if "--flat-playlist" in cmd:
        return "search"
    if "-x" in cmd:
        return "download"
    if "--dump-json" in cmd:
        return "metadata"
    return "unknown"


class FakeRun:
    """Drop-in replacement for subprocess.run, dispatched by command "kind"."""

    def __init__(self, responses: dict[str, object]):
        self.responses = responses
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        kind = _cmd_kind(cmd)
        if kind not in self.responses:
            raise AssertionError(f"unexpected subprocess call of kind {kind!r}: {cmd}")
        resp = self.responses[kind]
        if callable(resp) and not isinstance(resp, subprocess.CompletedProcess):
            resp = resp(cmd)
        return resp


def _ok(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)


def _fail(stderr: str = "boom") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)


def _metadata_json(**overrides) -> str:
    base = {
        "duration": 200,
        "title": "A Song",
        "uploader": "An Artist",
        "webpage_url": "https://www.youtube.com/watch?v=abc123",
    }
    base.update(overrides)
    return json.dumps(base) + "\n"


# --- allowlist: accepted hosts (subdomains ok) -----------------------------


@pytest.mark.parametrize(
    "url,expected_platform",
    [
        ("https://www.youtube.com/watch?v=abc123", "youtube"),
        ("https://m.youtube.com/watch?v=abc123", "youtube"),
        ("https://music.youtube.com/watch?v=abc123", "youtube"),
        ("https://youtu.be/abc123", "youtube"),
        ("https://vt.tiktok.com/ZSabc123/", "tiktok"),
        ("https://www.tiktok.com/@user/video/123", "tiktok"),
        ("https://www.instagram.com/reel/abc123/", "instagram"),
        ("https://soundcloud.com/artist/track", "soundcloud"),
        ("https://artist.bandcamp.com/track/song", "bandcamp"),
    ],
)
def test_allowed_hosts_and_subdomains(tmp_path, url, expected_platform):
    fake = FakeRun(
        {
            "metadata": _ok(_metadata_json(webpage_url=url)),
            "download": _ok(str(tmp_path / "abc123.wav") + "\n"),
        }
    )
    (tmp_path / "abc123.wav").write_bytes(b"\x00")
    with patch("escutario.fetch.subprocess.run", fake):
        result = fetch.fetch(url, tmp_path)
    assert result.platform == expected_platform
    assert isinstance(result, fetch.FetchResult)


# --- allowlist: refusals ----------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://youtube.com.evil.net/watch?v=abc123",
        "https://notyoutube.com/watch?v=abc123",
        "https://evil-tiktok.com/video/1",
        "https://faketiktok.com/video/1",
        "https://youtube.com.attacker.io/watch?v=x",
    ],
)
def test_lookalike_hosts_refused(tmp_path, url):
    with pytest.raises(fetch.FetchRefused):
        fetch.fetch(url, tmp_path)


def test_http_scheme_refused(tmp_path):
    with pytest.raises(fetch.FetchRefused, match="https"):
        fetch.fetch("http://www.youtube.com/watch?v=abc123", tmp_path)


def test_userinfo_refused(tmp_path):
    with pytest.raises(fetch.FetchRefused, match="userinfo"):
        fetch.fetch("https://user:pass@www.youtube.com/watch?v=abc123", tmp_path)


def test_nondefault_port_refused(tmp_path):
    with pytest.raises(fetch.FetchRefused, match="port"):
        fetch.fetch("https://www.youtube.com:8443/watch?v=abc123", tmp_path)


def test_default_port_443_is_allowed(tmp_path):
    url = "https://www.youtube.com:443/watch?v=abc123"
    fake = FakeRun(
        {
            "metadata": _ok(_metadata_json(webpage_url=url)),
            "download": _ok(str(tmp_path / "abc123.wav") + "\n"),
        }
    )
    (tmp_path / "abc123.wav").write_bytes(b"\x00")
    with patch("escutario.fetch.subprocess.run", fake):
        result = fetch.fetch(url, tmp_path)
    assert result.platform == "youtube"


def test_unsupported_scheme_refused(tmp_path):
    with pytest.raises(fetch.FetchRefused):
        fetch.fetch("ftp://youtube.com/watch?v=abc123", tmp_path)


# --- short-form detection ----------------------------------------------------


@pytest.mark.parametrize(
    "url,expected_short_form",
    [
        ("https://www.tiktok.com/@user/video/123", True),
        ("https://vt.tiktok.com/ZSabc/", True),
        ("https://www.instagram.com/reel/abc123/", True),
        ("https://www.instagram.com/p/abc123/", True),
        ("https://www.youtube.com/shorts/abc123", True),
        ("https://www.youtube.com/watch?v=abc123", False),
        ("https://soundcloud.com/artist/track", False),
        ("https://artist.bandcamp.com/track/song", False),
    ],
)
def test_short_form_flag(tmp_path, url, expected_short_form):
    fake = FakeRun(
        {
            "metadata": _ok(_metadata_json(webpage_url=url)),
            "download": _ok(str(tmp_path / "out.wav") + "\n"),
        }
    )
    (tmp_path / "out.wav").write_bytes(b"\x00")
    with patch("escutario.fetch.subprocess.run", fake):
        result = fetch.fetch(url, tmp_path)
    assert result.short_form is expected_short_form
    if expected_short_form:
        assert fetch.SHORT_FORM_NOTE in result.notes
    else:
        assert fetch.SHORT_FORM_NOTE not in result.notes


# --- duration cap: refused BEFORE any download happens ----------------------


def test_duration_over_cap_refused_before_download(tmp_path):
    url = "https://www.youtube.com/watch?v=abc123"
    fake = FakeRun({"metadata": _ok(_metadata_json(duration=900))})
    with patch("escutario.fetch.subprocess.run", fake):
        with pytest.raises(fetch.FetchRefused, match="600"):
            fetch.fetch(url, tmp_path)
    # only the metadata lookup ran — the download call never happened
    assert len(fake.calls) == 1
    assert _cmd_kind(fake.calls[0]) == "metadata"


def test_duration_exactly_at_cap_is_allowed(tmp_path):
    url = "https://www.youtube.com/watch?v=abc123"
    fake = FakeRun(
        {
            "metadata": _ok(_metadata_json(duration=600)),
            "download": _ok(str(tmp_path / "abc123.wav") + "\n"),
        }
    )
    (tmp_path / "abc123.wav").write_bytes(b"\x00")
    with patch("escutario.fetch.subprocess.run", fake):
        result = fetch.fetch(url, tmp_path)
    assert result.duration_s == 600


# --- happy path: fields are populated correctly -----------------------------


def test_fetch_happy_path_populates_fields(tmp_path):
    url = "https://www.youtube.com/watch?v=abc123"
    out_file = tmp_path / "abc123.wav"
    out_file.write_bytes(b"\x00")
    fake = FakeRun(
        {
            "metadata": _ok(_metadata_json(duration=210.5, title="A Song", uploader="An Artist")),
            "download": _ok(str(out_file) + "\n"),
        }
    )
    with patch("escutario.fetch.subprocess.run", fake):
        result = fetch.fetch(url, tmp_path, track="fallback title", artist="fallback artist")

    assert result.path == out_file
    assert result.source_url == url
    assert result.title == "A Song"
    assert result.artist == "An Artist"
    assert result.duration_s == 210.5
    assert result.platform == "youtube"
    assert result.short_form is False
    assert result.notes == []


def test_fetch_uses_track_artist_as_fallback_when_metadata_missing(tmp_path):
    url = "https://www.youtube.com/watch?v=abc123"
    out_file = tmp_path / "abc123.wav"
    out_file.write_bytes(b"\x00")
    fake = FakeRun(
        {
            "metadata": _ok(json.dumps({"duration": 100}) + "\n"),
            "download": _ok(str(out_file) + "\n"),
        }
    )
    with patch("escutario.fetch.subprocess.run", fake):
        result = fetch.fetch(url, tmp_path, track="fallback title", artist="fallback artist")
    assert result.title == "fallback title"
    assert result.artist == "fallback artist"


def test_fetch_flags_duration_mismatch_against_expected(tmp_path):
    url = "https://www.youtube.com/watch?v=abc123"
    out_file = tmp_path / "abc123.wav"
    out_file.write_bytes(b"\x00")
    fake = FakeRun(
        {
            "metadata": _ok(_metadata_json(duration=200)),
            "download": _ok(str(out_file) + "\n"),
        }
    )
    with patch("escutario.fetch.subprocess.run", fake):
        result = fetch.fetch(url, tmp_path, expected_duration_s=190)
    assert any("duration_mismatch" in n for n in result.notes)


def test_fetch_no_mismatch_note_within_threshold(tmp_path):
    url = "https://www.youtube.com/watch?v=abc123"
    out_file = tmp_path / "abc123.wav"
    out_file.write_bytes(b"\x00")
    fake = FakeRun(
        {
            "metadata": _ok(_metadata_json(duration=200)),
            "download": _ok(str(out_file) + "\n"),
        }
    )
    with patch("escutario.fetch.subprocess.run", fake):
        result = fetch.fetch(url, tmp_path, expected_duration_s=198)
    assert not any("duration_mismatch" in n for n in result.notes)


# --- yt-dlp failures ---------------------------------------------------------


def test_metadata_failure_raises_fetch_error(tmp_path):
    url = "https://www.youtube.com/watch?v=abc123"
    fake = FakeRun({"metadata": _fail("extractor error")})
    with patch("escutario.fetch.subprocess.run", fake):
        with pytest.raises(fetch.FetchError):
            fetch.fetch(url, tmp_path)


def test_download_failure_raises_fetch_error(tmp_path):
    url = "https://www.youtube.com/watch?v=abc123"
    fake = FakeRun(
        {
            "metadata": _ok(_metadata_json()),
            "download": _fail("max-filesize exceeded"),
        }
    )
    with patch("escutario.fetch.subprocess.run", fake):
        with pytest.raises(fetch.FetchError):
            fetch.fetch(url, tmp_path)


def test_download_missing_output_file_raises_fetch_error(tmp_path):
    url = "https://www.youtube.com/watch?v=abc123"
    fake = FakeRun(
        {
            "metadata": _ok(_metadata_json()),
            "download": _ok(str(tmp_path / "never_written.wav") + "\n"),
        }
    )
    with patch("escutario.fetch.subprocess.run", fake):
        with pytest.raises(fetch.FetchError):
            fetch.fetch(url, tmp_path)


def test_timeout_raises_fetch_error(tmp_path):
    url = "https://www.youtube.com/watch?v=abc123"

    def raise_timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

    with patch("escutario.fetch.subprocess.run", side_effect=raise_timeout):
        with pytest.raises(fetch.FetchError):
            fetch.fetch(url, tmp_path)


# --- local files --------------------------------------------------------------


def test_local_file_happy_path(tmp_path):
    local = tmp_path / "song.wav"
    local.write_bytes(b"\x00")
    fake = FakeRun({"ffprobe": _ok(json.dumps({"format": {"duration": "180.0"}}))})
    with patch("escutario.fetch.subprocess.run", fake):
        result = fetch.fetch(str(local), tmp_path, track="Local Song", artist="Local Artist")
    assert result.path == local
    assert result.platform == "local"
    assert result.short_form is False
    assert result.duration_s == 180.0
    assert result.title == "Local Song"
    assert result.artist == "Local Artist"
    assert result.notes == []


def test_local_file_over_cap_refused(tmp_path):
    local = tmp_path / "long.wav"
    local.write_bytes(b"\x00")
    fake = FakeRun({"ffprobe": _ok(json.dumps({"format": {"duration": "900.0"}}))})
    with patch("escutario.fetch.subprocess.run", fake):
        with pytest.raises(fetch.FetchRefused, match="600"):
            fetch.fetch(str(local), tmp_path)


def test_local_file_missing_refused_without_subprocess(tmp_path):
    missing = tmp_path / "nope.wav"
    fake = FakeRun({})
    with patch("escutario.fetch.subprocess.run", fake):
        with pytest.raises(fetch.FetchRefused, match="not found"):
            fetch.fetch(str(missing), tmp_path)
    assert fake.calls == []


def test_local_file_ffprobe_failure_notes_not_verified(tmp_path):
    local = tmp_path / "weird.wav"
    local.write_bytes(b"\x00")
    fake = FakeRun({"ffprobe": _fail("invalid data")})
    with patch("escutario.fetch.subprocess.run", fake):
        result = fetch.fetch(str(local), tmp_path)
    assert result.duration_s is None
    assert any("not verified" in n for n in result.notes)


def test_local_file_title_falls_back_to_stem(tmp_path):
    local = tmp_path / "My Cool Track.wav"
    local.write_bytes(b"\x00")
    fake = FakeRun({"ffprobe": _ok(json.dumps({"format": {"duration": "10.0"}}))})
    with patch("escutario.fetch.subprocess.run", fake):
        result = fetch.fetch(str(local), tmp_path)
    assert result.title == "My Cool Track"


# --- search() — the Spotify pointer flow --------------------------------------


def _search_lines(*entries) -> str:
    return "\n".join(json.dumps(e) for e in entries) + "\n"


def test_search_picks_closest_duration_candidate(tmp_path):
    out_file = tmp_path / "vid2.wav"
    out_file.write_bytes(b"\x00")
    entries = [
        {"id": "vid1", "title": "Live Version", "duration": 260, "uploader": "Someone"},
        {"id": "vid2", "title": "Studio Version", "duration": 202, "uploader": "The Artist"},
        {"id": "vid3", "title": "Sped Up", "duration": 150, "uploader": "Someone Else"},
    ]
    fake = FakeRun(
        {
            "search": _ok(_search_lines(*entries)),
            "download": _ok(str(out_file) + "\n"),
        }
    )
    with patch("escutario.fetch.subprocess.run", fake):
        result = fetch.search("Song Title", "The Artist", 200, tmp_path)

    assert result.title == "Studio Version"
    assert result.source_url == "https://www.youtube.com/watch?v=vid2"
    assert result.duration_s == 202
    assert not any("duration_mismatch" in n for n in result.notes)


def test_search_flags_duration_mismatch_on_best_available_match(tmp_path):
    out_file = tmp_path / "vid1.wav"
    out_file.write_bytes(b"\x00")
    entries = [
        {"id": "vid1", "title": "Only Candidate", "duration": 260, "uploader": "Someone"},
    ]
    fake = FakeRun(
        {
            "search": _ok(_search_lines(*entries)),
            "download": _ok(str(out_file) + "\n"),
        }
    )
    with patch("escutario.fetch.subprocess.run", fake):
        result = fetch.search("Song Title", "The Artist", 200, tmp_path)
    assert any("duration_mismatch" in n for n in result.notes)


def test_search_over_cap_candidate_refused(tmp_path):
    entries = [{"id": "vid1", "title": "Too Long", "duration": 900, "uploader": "Someone"}]
    fake = FakeRun({"search": _ok(_search_lines(*entries))})
    with patch("escutario.fetch.subprocess.run", fake):
        with pytest.raises(fetch.FetchRefused, match="600"):
            fetch.search("Song Title", "The Artist", 900, tmp_path)


def test_search_no_results_raises_fetch_error(tmp_path):
    fake = FakeRun({"search": _ok("")})
    with patch("escutario.fetch.subprocess.run", fake):
        with pytest.raises(fetch.FetchError):
            fetch.search("Nonexistent Song", "Nobody", 200, tmp_path)


# --- pure-function smoke test (no mocking, no subprocess, URL validation only) --


def test_smoke_url_validation_pure_functions():
    """Real, unmocked exercise of the pure URL-validation helpers.

    Deliberately NOT a download test — no subprocess runs here at all.
    """
    assert fetch._validate_url("https://www.youtube.com/watch?v=abc123") == ("youtube", "www.youtube.com")
    assert fetch._validate_url("https://vt.tiktok.com/ZSabc/") == ("tiktok", "vt.tiktok.com")

    with pytest.raises(fetch.FetchRefused):
        fetch._validate_url("https://notyoutube.com/watch?v=abc123")
    with pytest.raises(fetch.FetchRefused):
        fetch._validate_url("https://youtube.com.evil.net/watch?v=abc123")
    with pytest.raises(fetch.FetchRefused):
        fetch._validate_url("http://www.youtube.com/watch?v=abc123")
    with pytest.raises(fetch.FetchRefused):
        fetch._validate_url("https://user:pass@www.youtube.com/watch?v=abc123")
    with pytest.raises(fetch.FetchRefused):
        fetch._validate_url("https://www.youtube.com:8080/watch?v=abc123")

    assert fetch._is_short_form("tiktok", "https://vt.tiktok.com/ZSabc/") is True
    assert fetch._is_short_form("instagram", "https://www.instagram.com/p/abc/") is True
    assert fetch._is_short_form("youtube", "https://www.youtube.com/shorts/abc123") is True
    assert fetch._is_short_form("youtube", "https://www.youtube.com/watch?v=abc123") is False
