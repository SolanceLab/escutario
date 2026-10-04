# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
import os
import time
from pathlib import Path

import pytest

from escutario import worker as W


class FakeApi:
    def __init__(self, queue=None):
        self.queue = list(queue or [])
        self.calls = []

    def call(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path == "/escutario/claim":
            return {"request": self.queue.pop(0) if self.queue else None}
        if path == "/escutario/heard":
            return {"id": "heard-1"}
        return {"ok": True}


def fake_hear(target, *, title, artist, display_source, ask=None):
    return {"slug": "clip-abc123", "title": title, "artist": artist, "duration_s": 30.0, "card": "ESCUTÁRIO · card", "facts": {}}


def test_slug_matches_the_database_rule():
    import re
    for t in ["Forbidden Fruit", "", "!!!", "a" * 90, "Café Ñ"]:
        assert re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", W.make_slug(t, "src"))


def test_path_outside_allowed_folders_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "ALLOWED_ROOTS", [tmp_path / "allowed"])
    (tmp_path / "allowed").mkdir()
    (tmp_path / "secret").mkdir()
    good = tmp_path / "allowed" / "clip.m4a"; good.write_bytes(b"x")
    bad = tmp_path / "secret" / "clip.m4a"; bad.write_bytes(b"x")
    txt = tmp_path / "allowed" / "notes.txt"; txt.write_bytes(b"x")
    link = tmp_path / "allowed" / "sneaky.m4a"; os.symlink(bad, link)
    assert W.check_path(str(good)) == good.resolve()
    for p in (bad, txt, link, tmp_path / "allowed" / ".." / "secret" / "clip.m4a", tmp_path / "allowed" / "missing.m4a"):
        with pytest.raises(ValueError):
            W.check_path(str(p))
    with pytest.raises(ValueError):
        W.check_path("relative/clip.m4a")


def test_requests_are_heard_and_failures_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "ALLOWED_ROOTS", [tmp_path])
    f = tmp_path / "voice memo.m4a"; f.write_bytes(b"x")
    api = FakeApi([
        {"id": "r1", "kind": "link", "source": "https://youtu.be/x", "title": "Song", "artist": "A", "claimed_at": "2026-09-16T15:00:00.000Z"},
        {"id": "r2", "kind": "path", "source": str(f), "title": None, "artist": None},
        {"id": "r3", "kind": "path", "source": "/etc/passwd", "title": None, "artist": None},
    ])
    n = W.process_requests(api, hear_fn=fake_hear, limit=5)
    assert n == 2
    heard = [b for m, p, b in api.calls if p == "/escutario/heard"]
    assert heard[0]["request_id"] == "r1" and heard[0]["origin"] == "link" and heard[0]["claimed_at"] == "2026-09-16T15:00:00.000Z"
    assert heard[1]["request_id"] == "r2" and heard[1]["source"] == "Mac file voice memo.m4a"   # full path never leaves the Mac
    failed = [b for m, p, b in api.calls if p == "/escutario/fail"]
    assert failed and failed[0]["request_id"] == "r3" and "outside" in failed[0]["error"]


def test_drops_wait_until_settled_then_move(tmp_path):
    fresh = tmp_path / "just dropped.m4a"; fresh.write_bytes(b"x")
    old = tmp_path / "older.mp3"; old.write_bytes(b"x")
    past = time.time() - 60
    os.utime(old, (past, past))
    (tmp_path / "readme.txt").write_bytes(b"x")
    api = FakeApi()
    assert W.process_drops(api, drop_dir=tmp_path, hear_fn=fake_hear) == 1
    assert (tmp_path / "Heard" / "older.mp3").exists() and fresh.exists()
    body = [b for m, p, b in api.calls if p == "/escutario/heard"][0]
    assert body["origin"] == "drop" and body["source"] == "iCloud Drive/Escutário/older.mp3"


def test_a_drop_that_cannot_be_heard_is_moved_aside_with_a_reason(tmp_path):
    f = tmp_path / "broken.m4a"; f.write_bytes(b"x")
    past = time.time() - 60
    os.utime(f, (past, past))

    def boom(*a, **k):
        raise RuntimeError("ffmpeg could not decode")

    assert W.process_drops(FakeApi(), drop_dir=tmp_path, hear_fn=boom) == 0
    assert (tmp_path / "Could not hear" / "broken.m4a").exists()
    assert "ffmpeg" in (tmp_path / "Could not hear" / "broken.m4a.why.txt").read_text()


def test_env_file_parsing(tmp_path):
    e = tmp_path / ".env"
    e.write_text('# c\nSOLANCE_API_BASE="https://api.example"\nESCUTARIO_API_KEY=abc\nSOLANCE_API_KEY=broader\n')
    env = W.read_env(e)
    api = W.api_from_env(env)
    assert api.base == "https://api.example" and api.key == "abc"
    with pytest.raises(RuntimeError):
        W.api_from_env({"SOLANCE_API_BASE": "https://api.example", "SOLANCE_API_KEY": "broader"})   # never the broader key
    with pytest.raises(RuntimeError):
        W.api_from_env({"SOLANCE_API_BASE": "http://api.example", "ESCUTARIO_API_KEY": "abc"})
    e.chmod(0o644)
    with pytest.raises(RuntimeError):
        W.check_env_file(e)
    e.chmod(0o600)
    W.check_env_file(e)


def test_a_link_request_can_never_become_a_local_read(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "ALLOWED_ROOTS", [tmp_path])
    api = FakeApi([
        {"id": "a", "kind": "link", "source": "file:///etc/passwd", "claimed_at": "t"},
        {"id": "b", "kind": "link", "source": "/Users/someone/private.wav", "claimed_at": "t"},
        {"id": "c", "kind": "link", "source": "https://youtube.com\\@evil.test/x", "claimed_at": "t"},
        {"id": "d", "kind": "link", "source": "https://example.com/a.mp3", "claimed_at": "t"},
    ])
    heard_targets = []

    def spy(target, **k):
        heard_targets.append(target)
        return fake_hear(target, **k)

    assert W.process_requests(api, hear_fn=spy, limit=10) == 0
    assert heard_targets == []
    assert len([p for m, p, b in api.calls if p == "/escutario/fail"]) == 4


def test_errors_leaving_the_mac_carry_no_paths(tmp_path):
    msg = W.redact(f"FetchRefused: refused: local file is 900s ({W.HOME}/Music/Secret Demo/take3.wav) and /private/tmp/x/y.m4a")
    assert str(W.HOME) not in msg and "/Music/" not in msg and "/private/tmp" not in msg
    assert "y.m4a" in msg


def test_dropped_symlink_is_ignored_and_archive_never_overwrites(tmp_path):
    outside = tmp_path / "outside.wav"; outside.write_bytes(b"x")
    drops = tmp_path / "drops"; drops.mkdir()
    os.symlink(outside, drops / "sneaky.wav")
    past = time.time() - 60
    for name in ("song.wav",):
        (drops / name).write_bytes(b"x"); os.utime(drops / name, (past, past))
    (drops / "Heard").mkdir(); (drops / "Heard" / "song.wav").write_bytes(b"old")
    assert [p.name for p in W.settled_drops(drops)] == ["song.wav"]
    assert W.process_drops(FakeApi(), drop_dir=drops, hear_fn=fake_hear) == 1
    assert (drops / "Heard" / "song.wav").read_bytes() == b"old" and (drops / "Heard" / "song (2).wav").exists()
