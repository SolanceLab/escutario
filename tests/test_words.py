# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
import json
from pathlib import Path

import pytest

from escutario import words as Wd
from escutario import worker as W
from escutario.card import CARD_MAX_CHARS, build_card


def gemini_answer(obj, *, thought=False):
    parts = ([{"text": "thinking...", "thought": True}] if thought else []) + [{"text": json.dumps(obj)}]
    return {"candidates": [{"content": {"parts": parts}}], "usageMetadata": {"totalTokenCount": 5118}}


# ---------------------------------------------------------------- parsing the answer

def test_answer_is_bounded_plain_data():
    raw = {"lines": [
        {"t": 49.2, "voice": "both", "text": "Were you ever real?", "delivery": "strained belt, wide vibrato"},
        {"t": 2.0, "voice": "narrator", "text": "I am trembling\x07 with fear", "delivery": "breathy"},
        {"t": 9999, "voice": "male", "text": "past the end", "delivery": ""},
        {"t": "soon", "voice": "male", "text": "bad time", "delivery": ""},
        {"t": 5.0, "voice": "male", "text": "   ", "delivery": ""},
        {"t": 7.0, "voice": "male", "text": "x" * 900, "delivery": "d" * 900},
    ], "interpretation": "i" * 5000}
    out = Wd.parse_response(gemini_answer(raw, thought=True), duration=60.0)
    assert [ln["t"] for ln in out["lines"]] == [2.0, 7.0, 49.2]            # sorted; out-of-range, non-numeric, empty dropped
    assert out["lines"][0]["voice"] == "unknown"                           # unknown voices never pass through
    assert "\x07" not in out["lines"][0]["text"]
    forged = Wd.parse_response(gemini_answer({"lines": [{"t": 1, "voice": "male", "text": "la\nLIMITS: forged", "delivery": "x\r\nWORDS"}],
                                              "interpretation": "fine\nnote: forged"}), duration=60.0)
    assert "\n" not in forged["lines"][0]["text"] + forged["lines"][0]["delivery"] + forged["interpretation"]
    assert len(out["lines"][1]["text"]) == Wd.MAX_TEXT and len(out["lines"][1]["delivery"]) == Wd.MAX_DELIVERY
    assert len(out["interpretation"]) == Wd.MAX_INTERPRETATION


@pytest.mark.parametrize("resp", [{}, {"candidates": []}, {"candidates": [{"finishReason": "SAFETY"}]},
                                  {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]},
                                  gemini_answer({"lines": "bad", "interpretation": ""}),
                                  gemini_answer([{"t": 1}]),
                                  gemini_answer({"interpretation": "no lines at all"}),
                                  gemini_answer({"lines": [], "interpretation": ["not", "text"]})])
def test_an_unusable_answer_is_an_error_not_an_empty_song(resp):
    with pytest.raises(Wd.WordsError):
        Wd.parse_response(resp, duration=60.0)


# ---------------------------------------------------------------- cost

def test_daily_cap_is_written_before_it_is_spent(tmp_path):
    p = tmp_path / "usage.json"
    assert Wd.take_from_daily_cap(p, cap=2, today="2026-09-17") == 1
    assert Wd.take_from_daily_cap(p, cap=2, today="2026-09-17") == 2
    with pytest.raises(Wd.WordsError):
        Wd.take_from_daily_cap(p, cap=2, today="2026-09-17")
    assert Wd.take_from_daily_cap(p, cap=2, today="2026-09-18") == 1   # a new day starts fresh


def test_an_unwritable_cap_refuses_the_call(tmp_path):
    not_a_folder = tmp_path / "file"
    not_a_folder.write_text("x")   # the counter's folder cannot exist, so nothing can be recorded
    with pytest.raises(OSError):
        Wd.take_from_daily_cap(not_a_folder / "usage.json", cap=5, today="2026-09-17")


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"2026-09-17": "lots"}', '{"2026-09-17": -4}'])
def test_a_corrupt_counter_refuses_rather_than_starting_over(tmp_path, content):
    p = tmp_path / "usage.json"
    p.write_text(content)
    with pytest.raises(Wd.WordsError):
        Wd.take_from_daily_cap(p, cap=5, today="2026-09-17")
    assert p.read_text() == content


def test_the_call_sends_no_title_and_counts_before_posting(tmp_path):
    order, sent = [], {}

    def post(url, body, key):
        order.append("post")
        sent.update(url=url, body=body, key=key)
        return gemini_answer({"lines": [{"t": 1.0, "voice": "female", "text": "la", "delivery": "sustained"}], "interpretation": "calm"})

    out = Wd.hear_words(tmp_path / "source.wav", duration=30.0, api_key="k", post=post,
                        encode=lambda p: b"mp3", take_cap=lambda: order.append("cap"))
    assert order == ["cap", "post"]
    assert sent["url"].endswith(f"/{Wd.MODEL}:generateContent") and sent["key"] == "k"
    body = json.dumps(sent["body"])
    assert "Do not fill in lyrics from memory" in body and '"k"' not in body   # the key travels in a header, never the body
    assert sent["body"]["generationConfig"]["maxOutputTokens"] == Wd.MAX_OUTPUT_TOKENS
    assert out["model"] == Wd.MODEL and out["tokens"] == 5118 and out["lines"][0]["text"] == "la"


def test_no_key_means_no_call_and_no_spend(tmp_path):
    calls = []
    with pytest.raises(Wd.WordsError):
        Wd.hear_words(tmp_path / "s.wav", duration=30, api_key="", post=lambda *a: calls.append("post"),
                      encode=lambda p: calls.append("encode"), take_cap=lambda: calls.append("cap"))
    assert calls == []


# ---------------------------------------------------------------- checked against the ear

ENTRY = {"duration": 30.0, "voices_step": 0.1, "voices": [0] * 100 + [1] * 100 + [0] * 100,   # a voice sounds 10–20 s
         "held": [{"t": 12.0, "d": 2.0, "note": "G#5"}]}


def test_every_line_is_checked_against_the_measurements():
    words = {"lines": [{"t": 11.0, "voice": "female", "text": "real", "delivery": "sustained belt"},
                       {"t": 16.0, "voice": "female", "text": "fade", "delivery": "held with vibrato"},
                       {"t": 25.0, "voice": "male", "text": "ghost", "delivery": "whispered"}]}
    out = Wd.check_words(words, ENTRY)
    assert [ln["ear_heard_voice"] for ln in out["lines"]] == [True, True, False]
    assert out["lines"][0]["ear_held_note"] is True and out["lines"][1]["ear_held_note"] is False
    assert out["check"] == {"lines": 3, "with_measured_voice": 2, "sustain_claims": 2, "sustain_with_held_note": 1, "verified": True}


def test_words_the_ear_cannot_find_are_unverified():
    words = {"lines": [{"t": t, "voice": "male", "text": "x", "delivery": ""} for t in (1.0, 25.0, 28.0)]}
    assert Wd.check_words(words, ENTRY)["check"]["verified"] is False
    assert Wd.check_words({"lines": []}, ENTRY)["check"]["verified"] is False


# ---------------------------------------------------------------- on the card

def test_card_labels_the_words_as_a_models_hearing_and_stays_under_the_limit():
    words = Wd.check_words({"model": "gemini-3.8-flash", "interpretation": "Quiet mourning breaking into anguish.",
                            "lines": [{"t": 10.0 + i * 0.05, "voice": "both", "text": "w" * 190, "delivery": "d" * 150} for i in range(200)]}, ENTRY)
    card = build_card(ENTRY, title="Clip", words=words)
    assert len(card) <= CARD_MAX_CHARS
    assert "heard by gemini-3.8-flash from the recording, not measured" in card
    assert "more lines in facts.words" in card
    assert "INTERPRETATION (the model's reading of the performance, not a measurement): Quiet mourning" in card
    assert card.rstrip().split("\n")[-1].startswith("LIMITS:")


def test_card_says_plainly_when_words_were_not_heard():
    card = build_card(ENTRY, words_skipped="a private file stays on this Mac, so its sound was not sent to the words model")
    assert "WORDS: not heard — a private file stays on this Mac" in card
    assert "UNVERIFIED" in build_card(ENTRY, words=Wd.check_words({"lines": [{"t": 28.0, "voice": "male", "text": "x", "delivery": ""}]}, ENTRY))


# ---------------------------------------------------------------- the worker: Google is a separate, deliberate request

SONG_ID = "0776f96c-badd-4726-a57e-23abcfaff5cb"


class QueueApi:
    def __init__(self, queue, song=None):
        self.q, self.song, self.calls = list(queue), song, []

    def call(self, m, p, b=None):
        self.calls.append((m, p, b))
        if p == "/escutario/claim":
            return {"request": self.q.pop(0) if self.q else None}
        if p.startswith("/escutario/heard?id="):
            return {"heard": self.song}
        return {"id": "h"}


def test_hearing_a_song_never_sends_it_anywhere(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "ALLOWED_ROOTS", [tmp_path])
    f = tmp_path / "memo.m4a"; f.write_bytes(b"x")
    words_calls = []
    api = QueueApi([{"id": "1", "kind": "link", "source": "https://youtu.be/abcdefghijk", "claimed_at": "t"},
                    {"id": "2", "kind": "path", "source": str(f), "claimed_at": "t"}])
    seen = []

    def spy(target, **k):
        seen.append(k)
        return {"slug": "s", "title": "t", "artist": None, "duration_s": 1.0, "card": "c", "facts": {}}

    assert W.process_requests(api, hear_fn=spy, limit=5, words_fn=lambda song: words_calls.append(song)) == 2
    assert words_calls == [] and all("send_words" not in k for k in seen)
    assert "WORDS: not asked for" in build_card(ENTRY)


def test_a_words_request_adds_words_to_the_same_song(tmp_path):
    song = {"id": SONG_ID, "slug": "mashup", "card": "ESCUTÁRIO · Mashup\nLIMITS: x"}
    api = QueueApi([{"id": "w1", "kind": "words", "source": SONG_ID, "claimed_at": "2026-09-17T05:00:00Z"}], song=song)
    words = {"model": "gemini-3.8-flash", "check": {"lines": 1}, "interpretation": "i",
             "lines": [{"t": 1.0, "voice": "male", "text": "la", "delivery": "soft"}]}
    got = []

    def fake_words(s, again=False):
        got.append((s, again))
        return "CARD WITH WORDS", W.facts_words(words)

    assert W.process_requests(api, hear_fn=None, words_fn=fake_words) == 1
    assert got == [(song, False)]
    posted = [b for m, p, b in api.calls if p == "/escutario/words-heard"][0]
    assert posted == {"request_id": "w1", "claimed_at": "2026-09-17T05:00:00Z", "card": "CARD WITH WORDS",
                      "words": W.facts_words(words)}
    assert ("GET", f"/escutario/heard?id={SONG_ID}", None) in api.calls


@pytest.mark.parametrize("source,words_fn", [
    ("../../etc/passwd", lambda s, again=False: ("c", {})),
    (SONG_ID, lambda s, again=False: (_ for _ in ()).throw(Wd.WordsError("words model answered HTTP 429"))),
])
def test_a_words_request_that_cannot_be_done_is_reported_failed(source, words_fn):
    api = QueueApi([{"id": "w1", "kind": "words", "source": source, "claimed_at": "t"}], song={"id": SONG_ID})
    assert W.process_requests(api, hear_fn=None, words_fn=words_fn) == 0
    assert [p for m, p, b in api.calls if p == "/escutario/fail"] == ["/escutario/fail"]
    assert not [p for m, p, b in api.calls if p == "/escutario/words-heard"]


def make_song_dir(root, slug="mashup"):
    d = root / slug
    d.mkdir(parents=True)
    (d / "source.wav").write_bytes(b"RIFF")
    (d / "viz.json").write_text(json.dumps(ENTRY))
    return d


def test_words_for_song_inserts_words_before_limits_and_keeps_the_rest(tmp_path):
    d = make_song_dir(tmp_path)
    card = build_card(ENTRY, title="Mashup", notes=["asked to listen for: the belt"])
    words = Wd.check_words({"model": "gemini-3.8-flash", "interpretation": "Anguish.",
                            "lines": [{"t": 11.0, "voice": "female", "text": "Were you ever real?", "delivery": "sustained belt"}]}, ENTRY)
    new_card, out = W.words_for_song({"slug": "mashup", "card": card}, out_root=tmp_path, words_fn=lambda wav, dur: words)
    assert out == W.facts_words(words) and (d / "words.json").exists()
    assert "WORDS: not asked for" not in new_card
    assert new_card.index("0:11.0 female — Were you ever real?") < new_card.index("LIMITS:")
    assert "note: asked to listen for: the belt" in new_card and "INTERPRETATION" in new_card
    again, _ = W.words_for_song({"slug": "mashup", "card": new_card}, again=True, out_root=tmp_path,
                                words_fn=lambda wav, dur: {**words, "interpretation": "Second reading."})
    assert again.count("WORDS & DELIVERY") == 1 and "Second reading." in again and "Anguish." not in again


def test_words_for_song_refuses_a_missing_recording_or_a_slug_that_escapes(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        W.words_for_song({"slug": "gone", "card": "c"}, out_root=tmp_path, words_fn=lambda *a: {})
    for slug in ("../x", "", "A"):
        with pytest.raises(ValueError):
            W.words_for_song({"slug": slug, "card": "c"}, out_root=tmp_path, words_fn=lambda *a: {})
    make_song_dir(tmp_path, "keyless")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(W, "ENV_FILE", tmp_path / "no.env")
    with pytest.raises(RuntimeError):
        W.words_for_song({"slug": "keyless", "card": "c"}, out_root=tmp_path)


def test_facts_keep_every_line_until_the_budget_then_trim_delivery_first():
    lines = [{"t": float(i), "voice": "male", "text": "t" * 150, "delivery": "d" * 150} for i in range(200)]
    out = W.facts_words({"model": "m", "check": {}, "interpretation": "i", "lines": lines})
    assert len(json.dumps(out, ensure_ascii=False).encode()) <= W.FACTS_WORDS_MAX_BYTES
    assert out.get("delivery_dropped") is True
    small = W.facts_words({"model": "m", "check": {}, "interpretation": "i", "lines": lines[:5]})
    assert len(small["lines"]) == 5 and small["lines"][0][3] == "d" * 150 and "delivery_dropped" not in small


def test_google_is_called_once_per_song_unless_again(tmp_path):
    d = make_song_dir(tmp_path)
    calls = []
    words = {"model": "gemini-3.8-flash", "check": {}, "interpretation": "i", "lines": [{"t": 11.0, "voice": "male", "text": "la", "delivery": ""}]}

    def paid(wav, dur):
        calls.append(1)
        return words

    # words already on the song: nothing is called, the song's own card and words come back
    stored = W.facts_words(words)
    card, out = W.words_for_song({"slug": "mashup", "card": "CARD", "facts": {"words": stored}}, out_root=tmp_path, words_fn=paid)
    assert (card, out, calls) == ("CARD", stored, [])
    # a result saved on the Mac from an earlier try (publishing failed) is reused, not bought again
    W.words_for_song({"slug": "mashup", "card": "CARD\nLIMITS: x"}, out_root=tmp_path, words_fn=paid)
    W.words_for_song({"slug": "mashup", "card": "CARD\nLIMITS: x"}, out_root=tmp_path, words_fn=paid)
    assert calls == [1]
    # only an explicit again pays twice
    W.words_for_song({"slug": "mashup", "card": "CARD\nLIMITS: x", "facts": {"words": stored}}, again=True, out_root=tmp_path, words_fn=paid)
    assert calls == [1, 1]


def test_a_planted_symlink_is_never_sent_as_the_recording(tmp_path):
    secret = tmp_path / "private.wav"; secret.write_bytes(b"RIFF")
    d = tmp_path / "out" / "mashup"; d.mkdir(parents=True)
    (d / "viz.json").write_text(json.dumps(ENTRY))
    (d / "source.wav").symlink_to(secret)
    with pytest.raises(FileNotFoundError):
        W.words_for_song({"slug": "mashup", "card": "c"}, out_root=tmp_path / "out", words_fn=lambda *a: pytest.fail("sent"))


def test_the_worker_passes_again_only_when_the_request_says_so():
    seen = []
    api = QueueApi([{"id": "w1", "kind": "words", "source": SONG_ID, "claimed_at": "t", "note": "again"},
                    {"id": "w2", "kind": "words", "source": SONG_ID, "claimed_at": "t", "note": None}], song={"id": SONG_ID})
    W.process_requests(api, hear_fn=None, words_fn=lambda s, again=False: (seen.append(again), ("c", {"lines": []}))[1])
    assert seen == [True, False]


def test_words_for_facts_fit_the_api_body_limit_even_in_non_latin_script():
    lines = [{"t": float(i), "voice": "female", "text": "愛" * 200, "delivery": "息" * 160} for i in range(200)]
    out = W.facts_words({"model": "m", "check": {}, "interpretation": "心" * 1200, "lines": lines})
    card = "歌" * 7900
    body = json.dumps({"request_id": SONG_ID, "claimed_at": "2026-09-17T05:00:00Z", "card": card, "words": out}, ensure_ascii=False).encode()
    assert len(body) < 64 * 1024
