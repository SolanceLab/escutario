# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The listen pipeline with every organ faked: files written, page entry shape, merge rules."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from escutario import listen as L
from escutario.types import PitchTrack

DUR = 12.0
PAGE_KEYS = {
    "duration", "pitch_step", "pitch", "level_step", "level_vocals", "level_mix", "stem_step", "stems",
    "breaths", "silent_gaps", "longest_phrase_s", "falls", "flips", "slips", "grit", "ring", "tilt", "register",
    "hnr_median", "multi_voice", "held", "key", "tempo", "sections", "loudest_t", "dynamic_range",
    "changes", "arrivals", "voices_step", "voices", "riffs", "vibrato",
}


def sine_track(midi_by_time, dur=DUR, hop=0.01, conf=0.9):
    t = np.arange(0, dur, hop) + hop / 2
    f0 = np.zeros_like(t)
    for (a, b), m in midi_by_time.items():
        sel = (t >= a) & (t < b)
        f0[sel] = 440.0 * 2 ** ((m - 69) / 12)
    return PitchTrack(times=t, f0_hz=f0, confidence=np.where(f0 > 0, conf, 0.0), hop_s=hop)


@dataclass
class FakeFetch:
    path: Path
    source_url: str = "fake"
    title: str = "t"
    artist: str | None = None
    duration_s: float = DUR
    platform: str = "local"
    short_form: bool = False
    notes: list = field(default_factory=lambda: ["fake fetch note"])


@dataclass
class FakeSplit:
    stems: dict
    model: str = "fake"
    device: str = "cpu"
    seconds_elapsed: float = 0.0
    notes: list = field(default_factory=list)


CHANGES = [
    {"relabel": {"from": "other", "to": "piano", "pair_change_db": 0.1}, "t": 8.0, "strength": 0.94, "shifts": [], "arrivals": []},
] + [{"relabel": None, "t": float(i), "strength": round(0.1 * i, 2), "shifts": [], "arrivals": []} for i in range(1, 15)]
NOTES = [
    {"start_s": 1.0, "dur_s": 1.5, "midi": 60, "note_name": "C4", "salience": 0.7},    # duplicates the pitch-held C4 → dropped
    {"start_s": 1.1, "dur_s": 1.2, "midi": 64, "note_name": "E4", "salience": 0.6},    # layered → kept as poly
    {"start_s": 5.0, "dur_s": 1.0, "midi": 67, "note_name": "G4", "salience": 0.2},    # too faint
    {"start_s": 7.0, "dur_s": 0.5, "midi": 69, "note_name": "A4", "salience": 0.9},    # too short
]
EVENTS = [
    {"t": 0.25, "stem": "vocals", "kind": "enters", "change_db": 40.0, "confidence": "high"},
    {"t": 5.75, "stem": "piano", "kind": "enters", "change_db": 56.6, "confidence": "high"},   # relabel target near 8.0 → dropped
    {"t": 6.0, "stem": "drums", "kind": "rises", "change_db": 9.0, "confidence": "medium"},    # not high → dropped
    {"t": 9.0, "stem": "bass", "kind": "leaves", "change_db": -20.0, "confidence": "high"},    # not an arrival
]


def make_organs(calls):
    def rec(name, value):
        def f(*a, **k):
            calls.append(name)
            return value(*a, **k) if callable(value) else value
        return f

    def split(src, out_dir):
        calls.append("split")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stems = {}
        for s in L.STEM_ORDER:
            (out_dir / f"{s}.wav").write_bytes(b"")
            stems[s] = out_dir / f"{s}.wav"
        return FakeSplit(stems=stems)

    def load(path, sr):
        rng = np.random.default_rng(abs(hash((Path(path).name, sr))) % 2**32)
        return (0.1 * rng.standard_normal(int(DUR * sr))).astype(np.float32), sr

    def save_pitch(track, path, sr):
        calls.append("save_pitch")
        np.savez(path, times=track.times, f0=track.f0_hz, conf=track.confidence, hop_s=track.hop_s, sr=sr)

    track = sine_track({(1.0, 2.5): 60, (4.0, 4.5): 62, (6.0, 7.2): 67})
    return L.Organs(
        fetch=lambda src, out_dir, **kw: (calls.append("fetch"), FakeFetch(path=Path(src)))[1],
        split=split,
        track_pitch=rec("pitch", lambda v, sr, method="auto": track),
        save_pitch=save_pitch,
        transcribe_notes=rec("notes", NOTES),
        breath=rec("breath", {"breaths": [{"start_s": 3.0, "end_s": 3.3, "duration_s": 0.3, "level_db_rel_phrase": -12.0, "confidence": "high"}],
                              "count": 1, "phrases": [], "longest_phrase_s": 2.5, "silent_gaps": [{"start_s": 9.0, "end_s": 9.5}], "notes": []}),
        wail=rec("wail", {
            "falls": [{"start_s": 2.4, "end_s": 2.5, "duration_s": 0.1, "start_note": "C4", "drop_cents": 170.0, "confidence": "medium"}],
            "breaks": [{"time_s": 6.5, "from_note": "G4", "to_note": "C4", "jump_cents": -700.0}],
            "voice_changes": {"events": [{"time_s": 4.2}]},
            "grit": {"hnr_median_db": 15.0, "regions": [{"start_s": 6.1, "end_s": 6.4, "hnr_db": 9.0, "kind": "grit", "confidence": "medium"}]},
            "ring": {"overall_db": -12.0, "label": "moderate", "per_phrase": [{"start_s": 1.0, "end_s": 2.5, "ring_db": -11.0}, {"start_s": 6, "end_s": 7, "ring_db": None}]},
            "tilt": {"alpha_ratio_db": -8.0, "alpha_label": "full/pressed", "h1_h2_db": 1.0, "h1_h2_label": "modal"},
            "register": {"low_note": "C4", "high_note": "G4", "top_third_share": 0.3, "top_third_from_note": "E4", "longest_top_note": None},
            "multi_voice_suspected": False, "notes": [],
        }),
        entrances=rec("entrances", {"events": EVENTS, "openings": [], "hop_s": 0.25, "notes": []}),
        changes=rec("changes", CHANGES),
        count_voices=rec("count_voices", lambda notes, d: [1] * (int(d / 0.1) + 1)),
        vibrato=rec("vibrato", {"judged": 2, "with_vibrato": [{"start_s": 1.0, "dur_s": 1.2, "midi": 60, "rate_hz": 5.5, "extent_cents": 40, "confidence": "high"}], "straight": 1, "notes": []}),
        riffs=rec("riffs", [{"start_s": 6.0, "end_s": 7.0, "notes": 5, "notes_per_s": 5.0, "low": "E4", "high": "G4", "turns": 2, "contour": "x", "confidence": "high"}]),
        harmony=rec("harmony", {"key": {"key": "C major", "confidence": "high", "correlation": 0.9, "relative": {"key": "A minor"}},
                                "tempo": {"bpm": 120.0, "confidence": "high", "beat_times": [0.5, 1.0], "ambiguous": False, "alternatives": []},
                                "sections": [4.0, 8.0], "section_details": [], "energy": {"loudest_t": 6.2, "loudest_db": -8.0},
                                "duration_s": DUR}),
        load=load,
        write_source=lambda src, dst: (calls.append("write_source"), Path(dst).write_bytes(b""))[1],
    )


@pytest.fixture
def run(tmp_path):
    calls = []
    src = tmp_path / "song.m4a"
    src.write_bytes(b"")
    viz = tmp_path / "viz_data.json"
    viz.write_text(json.dumps({"other-song": {"duration": 1.0}, "demo": {"stale": True}}))
    res = L.listen(str(src), "demo", out_root=tmp_path / "out", viz_path=viz, organs=make_organs(calls))
    return res, calls, viz, tmp_path


def test_every_organ_runs_in_order(run):
    _, calls, _, _ = run
    order = ["fetch", "write_source", "split", "pitch", "save_pitch", "notes", "breath", "wail", "entrances", "changes",
             "harmony", "count_voices", "riffs", "vibrato"]
    assert [c for c in calls if c in order] == order


def test_files_written(run):
    res, _, _, tmp = run
    d = tmp / "out" / "demo"
    for name in ("fetch.json", "split.json", "pitch.npz", "notes.json", "breath.json", "voice.json", "entrances.json",
                 "harmony.json", "voices.json", "vibrato.json", "viz.json", "listen.json"):
        assert (d / name).exists(), name
    ent = json.loads((d / "entrances.json").read_text())
    assert set(ent) == {"entrances", "changes"}
    assert json.loads((d / "notes.json").read_text()) == NOTES
    assert json.loads((d / "listen.json").read_text())["notes"] == ["fake fetch note"]
    assert set(np.load(d / "pitch.npz").files) == {"times", "f0", "conf", "hop_s", "sr"}


def test_viz_entry_has_the_page_keys_and_shapes(run):
    res, _, _, _ = run
    e = res.viz_entry
    assert set(e) == PAGE_KEYS
    assert len(e["pitch"]) == int(DUR / 0.05)
    assert len(e["level_vocals"]) == len(e["level_mix"]) == int(DUR / 0.1)
    assert list(e["stems"]) == list(L.STEM_ORDER) and all(len(v) == int(DUR / 0.5) for v in e["stems"].values())
    assert e["pitch"][int(1.5 / 0.05)] == pytest.approx(60.0, abs=0.01) and e["pitch"][0] is None
    assert e["breaths"] == [{"t": 3.0, "d": 0.3, "c": "high", "rel": -12.0}]
    assert e["silent_gaps"] == [[9.0, 9.5]]
    assert e["flips"] == [{"t": 6.5, "frm": "G4", "to": "C4", "cents": -700.0}] and e["slips"] == [4.2]
    assert e["ring"]["per_phrase"] == [[1.0, 2.5, -11.0]]
    assert e["key"]["key"] == "C major" and e["key"]["confidence"] == "strong"
    assert e["tempo"]["bpm"] == 120.0 and e["tempo"]["c"] == "high"
    assert e["sections"] == [4.0, 8.0] and e["loudest_t"] == 6.2
    assert e["voices_step"] == 0.1 and len(e["voices"]) == int(DUR / 0.1) + 1
    assert isinstance(e["dynamic_range"], float)
    json.dumps(e)  # plain JSON only


def test_held_merges_single_voice_and_layered(run):
    held = run[0].viz_entry["held"]
    single = [h for h in held if h["src"] != "poly"]
    poly = [h for h in held if h["src"] == "poly"]
    assert [(h["note"], h["t"]) for h in single] == [("C4", 1.005), ("G4", 6.005)]
    assert all(set(h) == {"t", "d", "note", "src"} for h in single)
    assert poly == [{"t": 1.1, "d": 1.2, "note": "E4", "src": "poly", "sal": 0.6, "m": 64}]
    assert [h["t"] for h in held] == sorted(h["t"] for h in held)


def test_changes_top12_by_strength_sorted_by_time_and_arrivals_drop_relabels(run):
    e = run[0].viz_entry
    assert len(e["changes"]) == 12
    assert [c["t"] for c in e["changes"]] == sorted(c["t"] for c in e["changes"])
    assert min(c["strength"] for c in e["changes"]) >= 0.3
    assert any(c["relabel"] for c in e["changes"])
    assert e["arrivals"] == [{"t": 0.25, "stem": "vocals", "kind": "enters", "db": 40.0}]


def test_merge_keeps_other_songs_and_order(run):
    res, _, viz, _ = run
    data = json.loads(viz.read_text())
    assert list(data) == ["other-song", "demo"]
    assert data["other-song"] == {"duration": 1.0}
    assert data["demo"] == json.loads(json.dumps(res.viz_entry))


def test_merge_into_missing_file(tmp_path):
    p = tmp_path / "new.json"
    L.merge_into_viz(p, "a", {"x": np.float32(1.5), "y": [np.int64(2)]})
    assert json.loads(p.read_text()) == {"a": {"x": 1.5, "y": [2]}}


def test_skip_split_reuses_stems_and_source(tmp_path):
    calls = []
    d = tmp_path / "out" / "demo"
    (d / "stems").mkdir(parents=True)
    (d / "source.wav").write_bytes(b"")
    for s in ("vocals", "drums", "bass", "other"):
        (d / "stems" / f"{s}.wav").write_bytes(b"")
    res = L.listen(None, "demo", out_root=tmp_path / "out", skip_split=True, viz_path=None, organs=make_organs(calls))
    assert "fetch" not in calls and "split" not in calls
    assert list(res.viz_entry["stems"]) == ["vocals", "drums", "bass", "other"]
    assert any("guitar" in n and "piano" in n for n in res.notes)
    assert "viz_data" not in res.files


def test_skip_split_without_stems_fails_loudly(tmp_path):
    d = tmp_path / "out" / "demo"
    d.mkdir(parents=True)
    (d / "source.wav").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        L.listen(None, "demo", out_root=tmp_path / "out", skip_split=True, viz_path=None, organs=make_organs([]))


def test_held_notes_survive_vibrato_and_split_on_note_change():
    hop = 0.01
    t = np.arange(0, 4, hop) + hop / 2
    midi = np.where(t < 2.0, 60 + 0.5 * np.sin(2 * np.pi * 5.5 * t), 62.0)  # ±50¢ vibrato on C4, then D4
    f0 = 440 * 2 ** ((midi - 69) / 12)
    f0[(t > 1.0) & (t < 1.05)] = 0  # a 50 ms dropout inside the C4
    tr = PitchTrack(times=t, f0_hz=f0, confidence=np.full(len(t), 0.9), hop_s=hop)
    held = L.held_notes_from_pitch(tr)
    assert [h["note"] for h in held] == ["C4", "D4"], held
    assert held[0]["d"] == pytest.approx(2.0, abs=0.15)


def test_short_or_unconfident_notes_are_not_held():
    tr = sine_track({(1.0, 1.5): 60, (3.0, 5.0): 64}, conf=0.9)
    tr.confidence[(tr.times >= 3.0)] = 0.2
    assert L.held_notes_from_pitch(tr) == []
    assert L.held_notes_from_pitch(PitchTrack(np.zeros(0), np.zeros(0), np.zeros(0), 0.01)) == []


def test_cli_parses_and_calls_listen(monkeypatch):
    seen = {}
    monkeypatch.setattr(L, "listen", lambda *a, **k: (seen.update(args=a, kw=k), L.ListenResult(slug="s", out_dir=Path(".")))[1])
    assert L.main(["--slug", "s", "--skip-split", "--no-viz"]) == 0
    assert seen["args"] == (None, "s") and seen["kw"]["skip_split"] is True and seen["kw"]["viz_path"] is None


def test_listen_module_imports_nothing_from_vendor():
    src = Path(L.__file__).read_text()
    assert "vendor" not in src.replace("nothing here imports vendor code", "")


def test_review_hardening_slug_skip_split_and_nan(tmp_path):
    import json, math, pytest
    from escutario import listen as L
    for bad in ["../escape", "/abs", "a/b", "Upper", ""]:
        with pytest.raises(ValueError):
            L.listen(None, bad, out_root=tmp_path, viz_path=None)
    (tmp_path / "song").mkdir()
    with pytest.raises(ValueError):
        L.listen("https://youtu.be/x", "song", out_root=tmp_path, skip_split=True, viz_path=None)
    p = tmp_path / "n.json"
    L._write_json(p, {"a": float("nan"), "b": [1.0, float("inf")]})
    assert json.loads(p.read_text()) == {"a": None, "b": [1.0, None]}
