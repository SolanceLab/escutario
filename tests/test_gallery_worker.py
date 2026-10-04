# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
import json
import os
from pathlib import Path

from escutario import worker as W

SONG = "1ed9b23c-fcd1-4a42-b787-aecf5ae95475"


class Api:
    def __init__(self, pending=None, queue=None):
        self.pending, self.q, self.calls, self.puts = list(pending or []), list(queue or []), [], []

    def call(self, m, p, b=None):
        self.calls.append((m, p, b))
        if p == "/escutario/gallery/pending":
            return {"songs": self.pending}
        if p == "/escutario/claim":
            return {"request": self.q.pop(0) if self.q else None}
        if p == "/escutario/heard":
            self.heard_n = getattr(self, "heard_n", 0) + 1
            return {"id": SONG if self.heard_n == 1 else f"{SONG[:-2]}{self.heard_n:02d}"}
        return {"ok": True}

    def put_bytes(self, p, data, ctype):
        self.puts.append((p, len(data), ctype))
        return {"ok": True}


def song_dir(root: Path, slug="mashup"):
    d = root / slug
    d.mkdir(parents=True)
    (d / "source.wav").write_bytes(b"RIFF")
    (d / "viz.json").write_text('{"duration": 1}')
    return d


def old(f: Path):
    t = f.stat().st_mtime - 120
    os.utime(f, (t, t))


def test_uploads_only_what_the_gallery_says_is_missing(tmp_path):
    song_dir(tmp_path)
    api = Api(pending=[{"id": SONG, "slug": "mashup", "needs": ["audio", "score"]}])
    n = W.process_gallery(api, out_root=tmp_path, encode=lambda wav: b"ftypAUDIO", skipped_path=tmp_path / "s.json")
    assert n == 1
    assert api.puts == [(f"/escutario/songs/{SONG}/media/score", 15, "application/json"),
                        (f"/escutario/songs/{SONG}/media/audio", 9, "audio/mp4")]
    api2 = Api(pending=[{"id": SONG, "slug": "mashup", "needs": ["score"]}])
    W.process_gallery(api2, out_root=tmp_path, encode=lambda wav: (_ for _ in ()).throw(AssertionError("no audio needed")), skipped_path=tmp_path / "s.json")
    assert [p for p, *_ in api2.puts] == [f"/escutario/songs/{SONG}/media/score"]


def test_a_song_whose_recording_is_gone_is_logged_once_not_every_minute(tmp_path, monkeypatch):
    lines = []
    monkeypatch.setattr(W, "log", lambda m: lines.append(m))
    pending = [{"id": SONG, "slug": "gone", "needs": ["audio"]}, {"id": "x", "slug": "../escape", "needs": ["audio"]}]
    for _ in range(3):
        W.process_gallery(Api(pending=pending), out_root=tmp_path, encode=lambda wav: b"", skipped_path=tmp_path / "s.json")
    assert len([m for m in lines if "not on this Mac" in m]) == 2


def test_a_symlinked_recording_is_never_uploaded(tmp_path):
    secret = tmp_path / "private.wav"; secret.write_bytes(b"RIFF")
    d = tmp_path / "out" / "mashup"; d.mkdir(parents=True)
    (d / "viz.json").write_text("{}")
    (d / "source.wav").symlink_to(secret)
    api = Api(pending=[{"id": SONG, "slug": "mashup", "needs": ["audio"]}])
    assert W.process_gallery(api, out_root=tmp_path / "out", encode=lambda wav: b"x", skipped_path=tmp_path / "s.json") == 0
    assert api.puts == []


def test_a_heard_link_joins_the_gallery_but_a_mac_file_does_not(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "ALLOWED_ROOTS", [tmp_path])
    f = tmp_path / "memo.m4a"; f.write_bytes(b"x")
    api = Api(queue=[{"id": "1", "kind": "link", "source": "https://youtu.be/abcdefghijk", "claimed_at": "t"},
                     {"id": "2", "kind": "path", "source": str(f), "claimed_at": "t"}])
    hear = lambda target, **k: {"slug": "s", "title": "t", "artist": None, "duration_s": 1.0, "card": "c", "facts": {}}  # noqa: E731
    W.process_requests(api, hear_fn=hear, limit=5)
    gallery_calls = [p for m, p, b in api.calls if p.endswith("/gallery")]
    assert gallery_calls == [f"/escutario/songs/{SONG}/gallery"]


import numpy as np
import pytest

from escutario import fingerprint as fp


def tone_song(seed: int, seconds: float = 20.0) -> np.ndarray:
    """A synthetic 'song': a seeded melody of tones with a little noise, at the fingerprint's rate."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fp.SR)) / fp.SR
    y = np.zeros_like(t)
    notes = rng.uniform(300, 1800, size=int(seconds * 4))
    for i, f in enumerate(notes):
        seg = (t >= i * 0.25) & (t < (i + 1) * 0.25)
        y[seg] = np.sin(2 * np.pi * f * t[seg]) + 0.5 * np.sin(2 * np.pi * f * 1.5 * t[seg])
    return (y + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def test_the_same_song_louder_and_with_extra_lead_in_matches_and_a_different_song_does_not():
    song = tone_song(1)
    again = np.concatenate([np.zeros(int(1.3 * fp.SR), dtype=np.float32), 0.6 * song])   # quieter, 1.3 s later
    other = tone_song(2)
    a, b, c = (fp.fingerprint_samples(x) for x in (song, again, other))
    assert fp.bit_error(a, b) < 0.1
    assert fp.bit_error(a, c) > 0.35
    known = {"song-a": fp.pack(a, 20.0), "song-c": fp.pack(c, 20.0)}
    assert fp.find_same(b, known, duration=21.3)[0] == "song-a"
    assert fp.find_same(fp.fingerprint_samples(tone_song(3)), known, duration=20.0) is None
    assert fp.find_same(a[:50], known, duration=20.0) is None          # too little sound to judge
    assert np.array_equal(fp.unpack(fp.pack(a)), a)
    assert fp.find_same(a, {"broken": {"frames": "x", "duration": 20}}, duration=20.0) is None


def test_the_same_opening_is_not_enough_a_different_length_or_silence_is_heard():
    song = tone_song(1)
    a = fp.fingerprint_samples(song)
    known = {"song-a": fp.pack(a, 20.0), "no-length": fp.pack(a)}
    assert fp.find_same(a, known, duration=240.0) is None          # an extended mix sharing the intro
    assert fp.find_same(a, known, duration=None) is None           # length unknown: only identical bytes are certain
    silent = fp.fingerprint_samples(np.zeros(int(20 * fp.SR), dtype=np.float32))
    assert not fp.informative(silent)
    assert fp.find_same(silent, {"quiet": fp.pack(silent, 20.0)}, duration=20.0) is None


FAKE_PRINTS = {b"same recording": 1, b"same recording, other container": 1, b"another recording": 2}


def fake_print(path):
    return fp.fingerprint_samples(tone_song(FAKE_PRINTS.get(Path(path).read_bytes(), 9))), 20.0


def test_the_same_recording_dropped_again_under_another_name_is_not_heard_twice(tmp_path):
    drops = tmp_path / "drops"; drops.mkdir()
    hashes = tmp_path / "hashes.json"
    heard = []

    def hear(target, **k):
        heard.append(Path(target).name)
        return {"slug": "s", "title": "t", "artist": None, "duration_s": 1.0, "card": "c", "facts": {"key": "A"}}

    a = drops / "prayer.m4a"; a.write_bytes(b"same recording"); old(a)
    api = Api()
    prints = tmp_path / "prints.json"
    assert W.process_drops(api, drop_dir=drops, hear_fn=hear, hashes_path=hashes, prints_path=prints, print_fn=fake_print) == 1
    posted = [b for m, p, b in api.calls if p == "/escutario/heard"][0]
    assert posted["facts"]["key"] == "A" and len(posted["facts"]["sha256"]) == 64
    assert json.loads(hashes.read_text()) == {posted["facts"]["sha256"]: SONG}

    b = drops / "prayer again.m4a"; b.write_bytes(b"same recording"); old(b)                       # same bytes
    d = drops / "prayer (export).m4a"; d.write_bytes(b"same recording, other container"); old(d)   # same sound, other bytes
    c = drops / "different.m4a"; c.write_bytes(b"another recording"); old(c)
    assert W.process_drops(api, drop_dir=drops, hear_fn=hear, hashes_path=hashes, prints_path=prints, print_fn=fake_print) == 1
    assert heard == ["prayer.m4a", "different.m4a"]
    names = sorted(p.name for p in (drops / "Heard").iterdir())
    assert names == ["different.m4a", "prayer (export).m4a", "prayer (export).m4a.already-heard.txt",
                     "prayer again.m4a", "prayer again.m4a.already-heard.txt", "prayer.m4a"]
    note = (drops / "Heard" / "prayer (export).m4a.already-heard.txt").read_text()
    assert "Already heard: prayer" in note and "same sound" in note


def test_a_mac_file_already_heard_is_reported_not_heard_again(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "ALLOWED_ROOTS", [tmp_path])
    f = tmp_path / "memo.m4a"; f.write_bytes(b"same recording")
    monkeypatch.setattr(W, "_sound_print", fake_print)
    W.remember_heard(SONG, digest=None, bits=fake_print(f), title="Prayer", hashes_path=W.DROP_HASHES, prints_path=W.DROP_PRINTS)
    api = Api(queue=[{"id": "r", "kind": "path", "source": str(f), "claimed_at": "t"}])
    heard = []
    assert W.process_requests(api, hear_fn=lambda t, **k: heard.append(t), limit=2) == 0
    assert heard == []
    fail = [b for m, p, b in api.calls if p == "/escutario/fail"][0]
    assert f"already heard as song {SONG}" in fail["error"]


def test_a_failed_upload_waits_its_turn_and_never_blocks_the_songs_behind_it(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "log", lambda m: None)
    for slug in ("a", "b", "c", "d"):
        song_dir(tmp_path, slug)
    pending = [{"id": x, "slug": x, "needs": ["audio"]} for x in ("a", "b", "c", "d")]

    class Flaky(Api):
        def put_bytes(self, p, data, ctype):
            if "/a/" in p:
                raise RuntimeError("upload failed")
            return super().put_bytes(p, data, ctype)

    skipped = tmp_path / "s.json"
    api = Flaky(pending=pending)
    W.process_gallery(api, out_root=tmp_path, encode=lambda wav: b"ftyp", skipped_path=skipped, limit=3)
    assert [p for p, *_ in api.puts] == ["/escutario/songs/b/media/audio", "/escutario/songs/c/media/audio"]
    api2 = Flaky(pending=pending)
    W.process_gallery(api2, out_root=tmp_path, encode=lambda wav: b"ftyp", skipped_path=skipped, limit=3)
    assert "/escutario/songs/d/media/audio" in [p for p, *_ in api2.puts]      # a is cooling down; d gets its turn
    assert not any("/a/" in p for p, *_ in api2.puts)


def test_a_gallery_add_that_failed_is_retried_on_the_next_run(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "log", lambda m: None)
    monkeypatch.setattr(W, "ALLOWED_ROOTS", [tmp_path])

    class DownOnce(Api):
        down = True

        def call(self, m, p, b=None):
            if p.endswith("/gallery") and DownOnce.down:
                DownOnce.down = False
                raise RuntimeError("503")
            return super().call(m, p, b)

    api = DownOnce(queue=[{"id": "1", "kind": "link", "source": "https://youtu.be/abcdefghijk", "claimed_at": "t"}])
    hear = lambda target, **k: {"slug": "s", "title": "t", "artist": None, "duration_s": 1.0, "card": "c", "facts": {}}  # noqa: E731
    W.process_requests(api, hear_fn=hear, limit=2)
    assert json.loads(W.GALLERY_TO_ADD.read_text()) != {}
    api2 = DownOnce()
    W.process_gallery(api2, out_root=tmp_path, encode=lambda wav: b"", skipped_path=tmp_path / "s.json")
    assert [p for m, p, b in api2.calls if p.endswith("/gallery")] == [f"/escutario/songs/{SONG}/gallery"]
    assert json.loads(W.GALLERY_TO_ADD.read_text()) == {}
