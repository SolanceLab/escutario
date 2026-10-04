# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The local viewer's server: songs, audio ranges, marks, and the guards that keep it local."""
import http.client
import json
import threading

import pytest

from escutario import view

SLUG = "a-song"


@pytest.fixture
def server(tmp_path):
    d = tmp_path / SLUG
    d.mkdir()
    (d / "viz.json").write_text(json.dumps({"duration": 120.0, "pitch": [], "key": {"key": "C minor"}}))
    (d / "listen.json").write_text(json.dumps({"track": "A Song", "artist": "Someone"}))
    (d / "source.wav").write_bytes(bytes(range(256)) * 40)          # 10,240 bytes
    (d / "words.json").write_text(json.dumps({"lines": [{"t": 1.0, "voice": "female", "text": "hello"}], "model": "m"}))
    (tmp_path / "not-a-song").mkdir()                                 # no viz.json: never listed
    srv = view.make_server(0, tmp_path)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, tmp_path
    srv.shutdown()
    srv.server_close()


def req(srv, method, path, body=None, headers=None, host=None):
    port = srv.server_address[1]
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    h = {"Host": host or f"127.0.0.1:{port}"}
    if body is not None:
        h["Content-Type"] = "application/json"
        h["Origin"] = f"http://127.0.0.1:{port}"
    h.update(headers or {})
    c.request(method, path, body=json.dumps(body) if body is not None else None, headers=h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, dict(r.getheaders()), data


def test_lists_only_heard_songs(server):
    srv, _ = server
    status, _, data = req(srv, "GET", "/api/songs")
    songs = json.loads(data)["songs"]
    assert status == 200 and [s["slug"] for s in songs] == [SLUG]
    assert songs[0]["title"] == "A Song" and songs[0]["has_audio"] and songs[0]["has_words"]


def test_song_carries_score_words_and_marks(server):
    srv, _ = server
    status, _, data = req(srv, "GET", f"/api/songs/{SLUG}")
    d = json.loads(data)
    assert status == 200 and d["score"]["duration"] == 120.0 and d["words"]["lines"][0]["text"] == "hello" and d["marks"] == []


def test_page_files_and_nothing_else(server):
    srv, _ = server
    assert req(srv, "GET", "/")[0] == 200
    assert req(srv, "GET", "/score.js")[0] == 200
    assert req(srv, "GET", "/view.py")[0] == 404
    assert req(srv, "GET", "/api/songs/..%2F..%2Fetc")[0] == 404
    assert req(srv, "GET", "/api/songs/not-a-song")[0] == 404


def test_audio_seeks_with_ranges(server):
    srv, _ = server
    status, h, data = req(srv, "GET", f"/api/songs/{SLUG}/audio", headers={"Range": "bytes=100-199"})
    assert status == 206 and len(data) == 100 and h["Content-Range"] == "bytes 100-199/10240" and data[0] == 100
    status, h, data = req(srv, "GET", f"/api/songs/{SLUG}/audio", headers={"Range": "bytes=-10"})
    assert status == 206 and len(data) == 10
    assert req(srv, "GET", f"/api/songs/{SLUG}/audio", headers={"Range": "bytes=99999-"})[0] == 416
    status, h, data = req(srv, "GET", f"/api/songs/{SLUG}/audio")
    assert status == 200 and len(data) == 10240 and h["Accept-Ranges"] == "bytes"


def test_marks_round_trip(server):
    srv, root = server
    status, _, data = req(srv, "POST", f"/api/songs/{SLUG}/marks", {"t": 10, "t_end": 14.5, "what": ["voice"], "words": " here ", "measured": ["Voice C4"]})
    mark = json.loads(data)["mark"]
    assert status == 201 and mark["kind"] == "section" and mark["words"] == "here"
    assert json.loads((root / SLUG / "marks.json").read_text())[0]["id"] == mark["id"]
    status, _, data = req(srv, "PATCH", f"/api/songs/{SLUG}/marks/{mark['id']}", {"words": "changed"})
    assert status == 200 and json.loads(data)["mark"]["words"] == "changed"
    status, _, _ = req(srv, "DELETE", f"/api/songs/{SLUG}/marks/{mark['id']}", headers={"Origin": f"http://127.0.0.1:{srv.server_address[1]}"})
    assert status == 200 and json.loads((root / SLUG / "marks.json").read_text()) == []
    assert req(srv, "DELETE", f"/api/songs/{SLUG}/marks/{mark['id']}", headers={"Origin": f"http://127.0.0.1:{srv.server_address[1]}"})[0] == 404


@pytest.mark.parametrize("body", [
    {"t": -1, "what": ["voice"]},                    # before the song
    {"t": 500, "what": ["voice"]},                   # after it
    {"t": 10, "t_end": 5, "what": ["voice"]},        # ends before it starts
    {"t": 10, "what": []},                           # nothing chosen
    {"t": 10, "what": ["feelings"]},                 # not one of the four
    {"t": 10, "what": ["voice"], "words": 7},        # words must be text
    {"t": True, "what": ["voice"]},                  # a bool is not a second
    [1, 2],                                          # not an object
])
def test_bad_marks_refused(server, body):
    srv, root = server
    assert req(srv, "POST", f"/api/songs/{SLUG}/marks", body)[0] == 400
    assert not (root / SLUG / "marks.json").exists()


def test_long_words_and_measurements_are_capped(server):
    srv, _ = server
    _, _, data = req(srv, "POST", f"/api/songs/{SLUG}/marks", {"t": 1, "what": ["words"], "words": "x" * 5000, "measured": ["m" * 900] * 40})
    m = json.loads(data)["mark"]
    assert len(m["words"]) == view.MAX_WORDS and len(m["measured"]) == view.MAX_MEASURED and len(m["measured"][0]) == 300


def test_only_answers_its_own_address(server):
    srv, _ = server
    assert req(srv, "GET", "/api/songs", host="evil.example")[0] == 403          # DNS rebinding
    port = srv.server_address[1]
    assert req(srv, "GET", "/api/songs", host=f"localhost:{port}")[0] == 200


def test_writes_need_the_viewers_own_origin(server):
    srv, root = server
    port = srv.server_address[1]
    body = {"t": 1, "what": ["voice"]}
    assert req(srv, "POST", f"/api/songs/{SLUG}/marks", body, headers={"Origin": "http://evil.example"})[0] == 403
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("POST", f"/api/songs/{SLUG}/marks", body=json.dumps(body), headers={"Host": f"127.0.0.1:{port}", "Content-Type": "application/json"})
    assert c.getresponse().status == 403                                              # no Origin at all
    c.close()
    assert not (root / SLUG / "marks.json").exists()


def test_oversized_body_refused(server):
    srv, _ = server
    assert req(srv, "POST", f"/api/songs/{SLUG}/marks", {"t": 1, "what": ["voice"], "words": "x" * (view.MAX_BODY + 10)})[0] == 400


def test_page_security_headers(server):
    srv, _ = server
    _, h, _ = req(srv, "GET", "/")
    assert "default-src 'self'" in h["Content-Security-Policy"] and "frame-ancestors 'none'" in h["Content-Security-Policy"]
    assert h["X-Content-Type-Options"] == "nosniff"


def test_hardening_from_review(server):
    srv, _ = server
    assert req(srv, "GET", "/api/songs", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403   # another site embedding us
    assert req(srv, "GET", "/api/songs", headers={"Sec-Fetch-Site": "same-origin"})[0] == 200
    assert req(srv, "GET", f"/api/songs/{SLUG}/audio", headers={"Range": "bytes=" + "9" * 5000 + "-"})[0] == 416
    assert req(srv, "GET", f"/api/songs/{SLUG}%0A")[0] == 404                                   # no trailing newline in a slug
    port = srv.server_address[1]
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.putrequest("POST", f"/api/songs/{SLUG}/marks", skip_host=True)
    for k, v in {"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}", "Content-Length": "abc"}.items():
        c.putheader(k, v)
    c.endheaders()
    assert c.getresponse().status == 400                                                        # a garbage length gets an answer
    c.close()
