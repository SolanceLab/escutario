# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Listen — Escutário's whole pipeline, one command.

    python -m escutario.listen <url|file> --slug <name> [--track T --artist A]
    python -m escutario.listen --slug <name> --skip-split      # reuse out/<name>/source.wav + stems

fetch → split (Demucs) → track_pitch(vocals) → transcribe_notes(vocals) → breath, wail,
entrances + arrangement_changes, count_voices + detect_riffs → analyze_harmony →
`out/<slug>/*.json` and a score entry merged into `web/viz_data.json` (the data the
listening-score page reads).

Every organ is our own module; nothing here imports vendor code. Organs are looked up
lazily (pitch.py and notes.py load torch / onnx) and can be swapped for fakes in tests.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from escutario.types import PitchTrack

STEM_DURATION_TOLERANCE_S = 1.0
BREATH_MIN_CONFIDENCE = 0.8       # breath finds phrases only where singing is clearly pitched — set 16 Sep: at crepe's 0.5 voicing line breath tails and
MAX_ANALYSIS_S = 600.0            # audio.py decodes at most this much; a song this long would be analysed truncated
SILENT_DECODE_RMS = 1e-4               # about -80 dBFS: a decode that came back silent is a failure, not a quiet song
REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = REPO / "out"
MIN_ANALYSIS_S = 5.0              # shorter than this is not a song to analyse
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")  # one safe path component: the slug names a folder under out/
                                  # reverb read as voice and phrases merged (a 149 s 'phrase'); 0.8 gives 7-11 s longest phrases (plausibility, not ground truth)   # reused stems must match the source's length within this (same song, same split)
DEFAULT_VIZ_PATH = REPO / "web" / "viz_data.json"

ANALYSIS_SR = 22050          # rate the vocal/stem analysers read — matches the rate the organs were calibrated on
STEM_SR = 11025              # rate the arrangement organ reads the stems at — the rate its relabel timbre threshold was calibrated on (16 Sep; reproduces out/*/entrances.json exactly)
STEM_ORDER = ("vocals", "drums", "bass", "guitar", "piano", "other")  # page lane order (Demucs 6-stem names)
VIZ_LEVEL_SR = 4000          # level lanes are read at this rate (only RMS is needed) — same as the hand-run shaping script

# -- score lanes (shapes the page reads) — values match the shaping script they replace
PITCH_MIN_FRAMES = 2         # a lane cell needs at least this many confident frames — starting value
RELABEL_ARRIVAL_WINDOW_S = 3.0  # an arrival of the relabel's target stem this close to the relabel is the same sound renamed — set 16 Sep from the mashup (piano "enters" 5.75, relabel at 8.0)
SILENT_RMS = 1e-9            # RMS at/below this is written as SILENT_DB — starting value
PITCH_STEP_S = 0.05          # pitch lane resolution — page contract
LEVEL_STEP_S = 0.1           # vocal/mix level lane resolution — page contract
CHANGES_TOP_N = 12           # strongest arrangement changes shown — page contract
PITCH_CONF_MIN = 0.5         # frames at/above this confidence feed the lane; pitch.py's voicing boundary — starting value to calibrate
SILENT_DB = -90.0            # level written for digital silence — page contract
STEM_STEP_S = 0.5            # stem level lane resolution — page contract

# -- held notes
HELD_MIN_S = 0.9             # a held note lasts at least this long — page contract ("almost a second or more")
HELD_MAX_GAP_S = 0.1         # unvoiced holes up to this long don't end a note (consonant, tracker dropout) — starting value
POLY_HELD_MIN_SALIENCE = 0.35  # basic-pitch notes at/above this salience join the held lane as layered notes — set 16 Sep by hand on both songs
HELD_CONF_MIN = 0.5          # frames at/above this confidence count as sung — starting value to calibrate
HELD_SMOOTH_S = 0.2          # median smoothing of the pitch before segmenting (flattens vibrato) — starting value
HELD_TOL_SEMITONES = 0.6     # a note continues while smoothed pitch stays within this of the note's running mean — starting value
HELD_DEDUP_START_S = 0.6     # a layered note with the same midi starting this close to a single-voice held note is that note — derived 16 Sep from the page data (dropped at 0.30–0.58 s, kept from 0.70 s)

# -- dynamics
DYN_LOW_PCT = 5.0            # quiet end of the sung level range — starting value
DYN_HIGH_PCT = 95.0          # loud end of the sung level range — starting value
DYN_MIN_FRAMES = 20          # fewer sung level frames → dynamic range omitted — starting value

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _log(msg: str) -> None:
    print(f"[listen] {msg}", file=sys.stderr, flush=True)


# ===========================================================================
# organs — the real ones, looked up lazily
# ===========================================================================

def _real_fetch(source, out_dir, **kw):
    from escutario.fetch import fetch
    return fetch(source, out_dir, **kw)


def _real_split(path, out_dir):
    from escutario.split import split
    return split(Path(path), Path(out_dir))


def _real_track_pitch(vocal, sr, method="auto"):
    from escutario.pitch import track_pitch
    return track_pitch(vocal, sr, method=method)


def _real_save_pitch(track, path, sr):
    from escutario.pitch import save_npz
    save_npz(track, path, sr)


def _real_transcribe_notes(x, sr):
    from escutario.notes import transcribe_notes
    return transcribe_notes(x, sr)


def _real_breath(vocal, sr, pitch):
    from escutario.breath import detect_breaths
    return detect_breaths(vocal, sr, pitch)


def _real_wail(vocal, sr, pitch):
    from escutario.wail import analyze_voice
    return analyze_voice(vocal, sr, pitch)


def _real_entrances(stems, sr):
    from escutario.entrances import detect_entrances
    return detect_entrances(stems, sr)


def _real_changes(stems, sr, events):
    from escutario.entrances import arrangement_changes
    return arrangement_changes(stems, sr, events)


def _real_count_voices(notes, duration):
    from escutario.voicelines import count_voices
    return count_voices(notes, duration)


def _real_riffs(notes):
    from escutario.voicelines import detect_riffs
    return detect_riffs(notes)


def _real_vibrato(track):
    from escutario.vibrato import detect_vibrato
    return detect_vibrato(track)


def _real_harmony(mix, sr):
    from escutario.harmony import analyze_harmony
    return analyze_harmony(mix, sr)


def _real_load(path, sr):
    from escutario.audio import load_mono
    return load_mono(str(path), sr)


def _real_write_source(src_path, dst_path):
    """Decode any fetched file to out/<slug>/source.wav (44.1 kHz stereo 16-bit)."""
    import soundfile as sf
    from escutario.audio import load_stereo
    x, sr = load_stereo(str(src_path), 44100)
    sf.write(str(dst_path), np.clip(x, -1, 1).T, sr, subtype="PCM_16")


@dataclass
class Organs:
    fetch: Callable = _real_fetch
    split: Callable = _real_split
    track_pitch: Callable = _real_track_pitch
    save_pitch: Callable = _real_save_pitch
    transcribe_notes: Callable = _real_transcribe_notes
    breath: Callable = _real_breath
    wail: Callable = _real_wail
    entrances: Callable = _real_entrances
    changes: Callable = _real_changes
    count_voices: Callable = _real_count_voices
    riffs: Callable = _real_riffs
    vibrato: Callable = _real_vibrato
    harmony: Callable = _real_harmony
    load: Callable = _real_load
    write_source: Callable = _real_write_source


# ===========================================================================
# helpers
# ===========================================================================

def _jsonable(o: Any):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None   # NaN/Infinity would break the browser's JSON.parse
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "__dataclass_fields__"):
        return _jsonable({k: getattr(o, k) for k in o.__dataclass_fields__})
    return o


def _write_json(path: Path, obj: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _jsonable(obj)
    data = json.dumps(clean, separators=(",", ":"), allow_nan=False) if compact else json.dumps(clean, indent=1, allow_nan=False)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(data)
    os.replace(tmp, path)


def _db(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    return round(20 * np.log10(r), 1) if r > SILENT_RMS else SILENT_DB


def _level_lane(x: np.ndarray, sr: int, step_s: float) -> list[float]:
    h = int(round(step_s * sr))
    return [_db(x[i:i + h]) for i in range(0, len(x) - h + 1, h)]


def _note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def _midi(f0: np.ndarray) -> np.ndarray:
    out = np.full(len(f0), np.nan)
    v = f0 > 0
    out[v] = 69 + 12 * np.log2(f0[v] / 440.0)
    return out


def pitch_lane(track: PitchTrack, duration: float) -> list[float | None]:
    t, f0, cf = np.asarray(track.times), np.asarray(track.f0_hz), np.asarray(track.confidence)
    m = _midi(f0)
    out: list[float | None] = []
    for i in range(int(duration / PITCH_STEP_S)):
        sel = (t >= i * PITCH_STEP_S) & (t < (i + 1) * PITCH_STEP_S) & (f0 > 0) & (cf >= PITCH_CONF_MIN)
        out.append(round(float(np.median(m[sel])), 2) if sel.sum() >= PITCH_MIN_FRAMES else None)
    return out


def held_notes_from_pitch(track: PitchTrack) -> list[dict]:
    """Single-voice held notes: stretches where the (vibrato-smoothed) pitch stays on one note ≥ HELD_MIN_S."""
    t, f0, cf = np.asarray(track.times, float), np.asarray(track.f0_hz, float), np.asarray(track.confidence, float)
    n = len(t)
    if n == 0:
        return []
    hop = float(track.hop_s) if track.hop_s else (float(np.median(np.diff(t))) if n > 1 else 0.01)
    voiced = (f0 > 0) & (cf >= HELD_CONF_MIN)
    m = _midi(np.where(voiced, f0, 0.0))
    k = max(1, int(round(HELD_SMOOTH_S / hop)) // 2)
    smooth = np.full(n, np.nan)
    idx = np.flatnonzero(voiced)
    for i in idx:
        w = m[max(0, i - k):i + k + 1]
        smooth[i] = np.nanmedian(w)
    max_gap = int(round(HELD_MAX_GAP_S / hop))
    notes: list[dict] = []
    start = last = None
    total = 0.0
    count = 0
    vals: list[float] = []

    def close():
        if start is None:
            return
        dur = t[last] - t[start] + hop
        if dur >= HELD_MIN_S - 1e-9:
            midi = int(round(float(np.median(vals))))
            notes.append({"t": round(float(t[start]), 3), "d": round(float(dur), 3), "note": _note_name(midi), "m": midi})

    for i in idx:
        s = smooth[i]
        if start is not None and (i - last - 1) <= max_gap and abs(s - total / count) <= HELD_TOL_SEMITONES:
            last, total, count = i, total + s, count + 1
            vals.append(m[i])
            continue
        close()
        start, last, total, count, vals = i, i, s, 1, [m[i]]
    close()
    return notes


def merge_held(single: list[dict], poly_notes: list[dict]) -> list[dict]:
    """The page's held lane: single-voice held notes (src 'pitch') plus basic-pitch notes held ≥ HELD_MIN_S
    at salience ≥ POLY_HELD_MIN_SALIENCE (src 'poly', with 'sal' and 'm'), dropping a layered note that
    duplicates a single-voice one (same midi, start within HELD_DEDUP_START_S)."""
    out = [{"t": h["t"], "d": h["d"], "note": h["note"], "src": "pitch"} for h in single]
    for p in poly_notes:
        if p["dur_s"] < HELD_MIN_S or p.get("salience", 0.0) < POLY_HELD_MIN_SALIENCE:
            continue
        if any(h["m"] == p["midi"] and abs(h["t"] - p["start_s"]) <= HELD_DEDUP_START_S for h in single):
            continue
        out.append({"t": round(float(p["start_s"]), 2), "d": round(float(p["dur_s"]), 2),
                    "note": p.get("note_name") or _note_name(int(p["midi"])), "src": "poly",
                    "sal": round(float(p["salience"]), 2), "m": int(p["midi"])})
    return sorted(out, key=lambda h: h["t"])


def arrivals_for_page(events: list[dict], changes: list[dict]) -> list[dict]:
    relabels = [c for c in changes if c.get("relabel")]
    out = []
    for e in events:
        if e.get("confidence") != "high" or e.get("kind") not in ("enters", "rises"):
            continue
        if any(e["stem"] == c["relabel"]["to"] and abs(e["t"] - c["t"]) <= RELABEL_ARRIVAL_WINDOW_S for c in relabels):
            continue
        out.append({"t": e["t"], "stem": e["stem"], "kind": e["kind"], "db": e["change_db"]})
    return out


def changes_for_page(changes: list[dict]) -> list[dict]:
    top = sorted(changes, key=lambda c: -c["strength"])[:CHANGES_TOP_N]
    return sorted(top, key=lambda c: c["t"])


def sung_dynamic_range(vocal: np.ndarray, sr: int, track: PitchTrack) -> float | None:
    """Loud-minus-quiet level (DYN_HIGH_PCT − DYN_LOW_PCT percentile, dB) of the vocal stem while it sings."""
    h = int(round(LEVEL_STEP_S * sr))
    t, f0, cf = np.asarray(track.times), np.asarray(track.f0_hz), np.asarray(track.confidence)
    sung_t = t[(f0 > 0) & (cf >= PITCH_CONF_MIN)]
    if h == 0 or sung_t.size == 0:
        return None
    cells = np.unique((sung_t / LEVEL_STEP_S).astype(int))
    lv = [_db(vocal[c * h:(c + 1) * h]) for c in cells if (c + 1) * h <= len(vocal)]
    lv = [v for v in lv if v > SILENT_DB]
    if len(lv) < DYN_MIN_FRAMES:
        return None
    return round(float(np.percentile(lv, DYN_HIGH_PCT) - np.percentile(lv, DYN_LOW_PCT)), 1)


def build_viz_entry(*, duration: float, track: PitchTrack, level_vocals, level_mix, stems_levels, breath: dict,
                    voice: dict, held: list[dict], harmony: dict, dynamic_range, changes: list[dict],
                    arrivals: list[dict], voices: list[int], riffs: list[dict], vibrato: dict | None = None) -> dict:
    v = voice or {}
    ring = v.get("ring") or {}
    tilt = v.get("tilt") or {}
    reg = v.get("register") or {}
    grit = v.get("grit") or {}
    key = harmony.get("key") or {}
    tempo = harmony.get("tempo") or {}
    energy = harmony.get("energy") or {}
    return {
        "duration": duration,
        "pitch_step": PITCH_STEP_S, "pitch": pitch_lane(track, duration),
        "level_step": LEVEL_STEP_S, "level_vocals": level_vocals, "level_mix": level_mix,
        "stem_step": STEM_STEP_S, "stems": stems_levels,
        "breaths": [{"t": round(b["start_s"], 2), "d": b["duration_s"], "c": b["confidence"], "rel": b["level_db_rel_phrase"]}
                    for b in breath.get("breaths", [])],
        "silent_gaps": [[round(g["start_s"], 2), round(g["end_s"], 2)] for g in breath.get("silent_gaps", [])],
        "longest_phrase_s": breath.get("longest_phrase_s"),
        "falls": [{"t": round(f["start_s"], 2), "d": f["duration_s"], "note": f["start_note"], "cents": f["drop_cents"], "c": f["confidence"]}
                  for f in v.get("falls", [])],
        "flips": [{"t": round(x.get("time_s", x.get("start_s", 0)), 2), "frm": x.get("from_note"), "to": x.get("to_note"), "cents": x.get("jump_cents")}
                  for x in v.get("breaks", [])],
        "slips": [round(e["time_s"], 2) for e in (v.get("voice_changes") or {}).get("events", [])],
        "grit": [{"t0": round(g["start_s"], 2), "t1": round(g["end_s"], 2), "hnr": g["hnr_db"], "kind": g["kind"], "c": g["confidence"]}
                 for g in grit.get("regions", [])],
        "ring": {"overall": ring.get("overall_db"), "label": ring.get("label"),
                 "per_phrase": [[round(x["start_s"], 2), round(x["end_s"], 2), x["ring_db"]] for x in ring.get("per_phrase", []) if x.get("ring_db") is not None]},
        "tilt": {"alpha": tilt.get("alpha_ratio_db"), "alpha_label": tilt.get("alpha_label"), "h1h2": tilt.get("h1_h2_db"), "h1h2_label": tilt.get("h1_h2_label")},
        "register": {"low": reg.get("low_note"), "high": reg.get("high_note"), "top_share": reg.get("top_third_share"),
                     "top_from": reg.get("top_third_from_note"), "longest": reg.get("longest_top_note")},
        "hnr_median": grit.get("hnr_median_db"), "multi_voice": v.get("multi_voice_suspected"),
        "held": held,
        # the page flags confidence === 'weak' (the old vocabulary); 'level' carries ours
        "key": {"key": key.get("key"), "correlation": key.get("correlation"),
                "confidence": {"high": "strong", "medium": "medium"}.get(key.get("confidence"), "weak"), "level": key.get("confidence"),
                "alternative": (key.get("relative") or {}).get("key")},
        "tempo": {"bpm": tempo.get("bpm"), "c": tempo.get("confidence"), "ambiguous": tempo.get("ambiguous"),
                  "alternatives": [{"bpm": a["bpm"], "relation": a["relation"]} for a in tempo.get("alternatives", [])]},
        "sections": list(harmony.get("sections", [])),
        "loudest_t": energy.get("loudest_t"),
        "dynamic_range": dynamic_range,
        "changes": changes, "arrivals": arrivals,
        "voices_step": 0.1, "voices": voices, "riffs": riffs,
        "vibrato": vibrato or {"judged": 0, "with_vibrato": [], "straight": 0},
    }


def gate_track(track: PitchTrack, min_confidence: float) -> PitchTrack:
    """The same pitch track with frames below `min_confidence` marked unvoiced."""
    keep = np.asarray(track.confidence) >= min_confidence
    return PitchTrack(times=track.times, f0_hz=np.where(keep, track.f0_hz, 0.0),
                      confidence=np.where(keep, track.confidence, 0.0), hop_s=track.hop_s)


def merge_into_viz(viz_path: str | Path, slug: str, entry: dict) -> None:
    """Put `entry` under `slug` in the page data file, keeping every other song (and the key order)."""
    viz_path = Path(viz_path)
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    # hold a lock across read-modify-replace so two listens can't erase each other's song
    with open(viz_path.with_name(viz_path.name + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(viz_path.read_text()) if viz_path.exists() and viz_path.stat().st_size else {}
        if not isinstance(data, dict):
            raise ValueError(f"{viz_path} is not a JSON object of songs")
        data[slug] = entry
        _write_json(viz_path, data, compact=True)


# ===========================================================================
# the pipeline
# ===========================================================================

@dataclass
class ListenResult:
    slug: str
    out_dir: Path
    files: dict[str, Path] = field(default_factory=dict)
    viz_entry: dict = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def listen(source: str | None, slug: str, *, track: str | None = None, artist: str | None = None,
           out_root: str | Path = DEFAULT_OUT_ROOT, skip_split: bool = False, viz_path: str | Path | None = DEFAULT_VIZ_PATH,
           pitch_method: str = "auto", organs: Organs | None = None) -> ListenResult:
    o = organs or Organs()
    if not SLUG_RE.match(slug or ""):
        raise ValueError(f"slug {slug!r} must be lowercase letters, digits and hyphens (one folder name, no paths)")
    root = Path(out_root).resolve()
    out_dir = root / slug
    if out_dir.is_symlink() or out_dir.resolve().parent != root:
        raise ValueError(f"output folder for {slug!r} resolves outside {root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stems_dir = out_dir / "stems"
    source_wav = out_dir / "source.wav"
    res = ListenResult(slug=slug, out_dir=out_dir)
    clock = time.monotonic()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.monotonic()
        res.timings_s[name] = round(now - clock, 2)
        clock = now
        _log(f"{name} done in {res.timings_s[name]:.1f} s")

    def save(name: str, obj: Any, filename: str) -> None:
        path = out_dir / filename
        _write_json(path, obj)
        res.files[name] = path

    # 1. fetch
    if skip_split and source is not None:
        raise ValueError("--skip-split reuses this slug's existing audio and stems; drop the source, or use a new slug for a new song")
    if source is not None:
        fr = o.fetch(source, out_dir, track=track, artist=artist)
        if Path(fr.path).resolve() != source_wav.resolve():
            o.write_source(fr.path, source_wav)
        save("fetch", fr, "fetch.json")
        res.notes.extend(getattr(fr, "notes", []) or [])
        lap("fetch")
    elif not source_wav.exists():
        raise FileNotFoundError(f"no source given and {source_wav} does not exist")
    res.files["source"] = source_wav

    # 2. split
    if skip_split:
        missing = [s for s in STEM_ORDER if not (stems_dir / f"{s}.wav").exists()]
        if not (stems_dir / "vocals.wav").exists():
            raise FileNotFoundError(f"--skip-split needs existing stems in {stems_dir} (vocals.wav missing)")
        stem_paths = {s: stems_dir / f"{s}.wav" for s in STEM_ORDER if s not in missing}
        if missing:
            res.notes.append(f"stems missing and skipped: {', '.join(missing)}")
        _log(f"reusing stems in {stems_dir}")
    else:
        sp = o.split(source_wav, stems_dir)
        stem_paths = {k: Path(v) for k, v in sp.stems.items()}
        save("split", sp, "split.json")
        res.notes.extend(getattr(sp, "notes", []) or [])
        lap("split")

    # 3. load
    vocal, sr = o.load(stem_paths["vocals"], ANALYSIS_SR)
    mix, _ = o.load(source_wav, ANALYSIS_SR)
    mix_s, vocal_s = len(mix) / sr, len(vocal) / sr
    if mix_s < MIN_ANALYSIS_S:
        raise ValueError(f"the audio is only {mix_s:.1f} s; too short to hear as a song")
    if float(np.sqrt(np.mean(np.square(mix, dtype=np.float64)))) < SILENT_DECODE_RMS:
        raise ValueError("the audio is silent; refusing to report a song nobody sings in")
    if mix_s >= MAX_ANALYSIS_S - 0.5:
        raise ValueError(f"the audio is at least {MAX_ANALYSIS_S:.0f} s long; refusing to analyse a truncated song")
    if abs(mix_s - vocal_s) > STEM_DURATION_TOLERANCE_S:
        raise ValueError(f"stems ({vocal_s:.1f} s) don't match the source ({mix_s:.1f} s): they belong to a different song or split")
    stems = {name: o.load(p, STEM_SR)[0] for name, p in stem_paths.items()}
    lap("load")

    # 4. pitch + notes
    ptrack = o.track_pitch(vocal, sr, method=pitch_method)
    o.save_pitch(ptrack, out_dir / "pitch.npz", sr)
    res.files["pitch"] = out_dir / "pitch.npz"
    lap("pitch")
    notes = o.transcribe_notes(vocal, sr)
    save("notes", notes, "notes.json")
    lap("notes")

    # 5. the voice organs
    breath = o.breath(vocal, sr, gate_track(ptrack, BREATH_MIN_CONFIDENCE))
    save("breath", breath, "breath.json")
    lap("breath")
    voice = o.wail(vocal, sr, ptrack)
    save("voice", voice, "voice.json")
    lap("voice")

    # 6. the arrangement
    ent = o.entrances(stems, STEM_SR)
    changes = o.changes(stems, STEM_SR, ent.get("events", []))
    save("entrances", {"entrances": ent, "changes": changes}, "entrances.json")
    lap("entrances")

    # 7. song facts
    harmony = o.harmony(mix, sr)
    save("harmony", harmony, "harmony.json")
    lap("harmony")
    duration = float(harmony.get("duration_s") or round(len(mix) / sr, 2))

    # 8. voice lines
    voices = o.count_voices(notes, duration)
    riffs = o.riffs(notes)
    single_held = held_notes_from_pitch(ptrack)
    held = merge_held(single_held, notes)
    save("voices", {"step_s": 0.1, "counts": voices, "riffs": riffs, "held": held}, "voices.json")
    lap("voices")
    vib = o.vibrato(ptrack)
    save("vibrato", vib, "vibrato.json")
    lap("vibrato")

    # 9. the score entry
    vocal_4k, _ = o.load(stem_paths["vocals"], VIZ_LEVEL_SR)
    mix_4k, _ = o.load(source_wav, VIZ_LEVEL_SR)
    stems_levels = {}
    for name in STEM_ORDER:
        if name in stem_paths:
            x4, _ = o.load(stem_paths[name], VIZ_LEVEL_SR)
            stems_levels[name] = _level_lane(x4, VIZ_LEVEL_SR, STEM_STEP_S)
    entry = build_viz_entry(
        duration=duration, track=ptrack,
        level_vocals=_level_lane(vocal_4k, VIZ_LEVEL_SR, LEVEL_STEP_S), level_mix=_level_lane(mix_4k, VIZ_LEVEL_SR, LEVEL_STEP_S),
        stems_levels=stems_levels, breath=breath, voice=voice, held=held, harmony=harmony,
        dynamic_range=sung_dynamic_range(vocal, sr, ptrack),
        changes=changes_for_page(changes), arrivals=arrivals_for_page(ent.get("events", []), changes),
        voices=list(voices), riffs=riffs, vibrato=vib,
    )
    save("viz", entry, "viz.json")
    res.viz_entry = entry
    if viz_path is not None:
        merge_into_viz(viz_path, slug, entry)
        res.files["viz_data"] = Path(viz_path)
        _log(f"merged '{slug}' into {viz_path}")
    lap("viz")

    save("listen", {"slug": slug, "source": source, "track": track, "artist": artist, "skip_split": skip_split,
                    "pitch_method": pitch_method, "timings_s": res.timings_s, "notes": res.notes,
                    "files": {k: str(v) for k, v in res.files.items()}}, "listen.json")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m escutario.listen", description="Listen to a song end to end.")
    ap.add_argument("source", nargs="?", help="https URL on the allowlist or a local audio file (optional with --skip-split)")
    ap.add_argument("--slug", required=True, help="output folder name under out/ and the song id in the page data")
    ap.add_argument("--track", help="track title (metadata fallback)")
    ap.add_argument("--artist", help="artist (metadata fallback)")
    ap.add_argument("--skip-split", action="store_true", help="reuse out/<slug>/source.wav and out/<slug>/stems/*.wav")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="parent of the per-song folders (default: out/)")
    ap.add_argument("--viz", default=str(DEFAULT_VIZ_PATH), help="page data file to merge the entry into (default: web/viz_data.json)")
    ap.add_argument("--no-viz", action="store_true", help="write out/<slug>/viz.json only; leave the page data alone")
    ap.add_argument("--pitch-method", default="auto", help="pitch.track_pitch method (default: auto)")
    a = ap.parse_args(argv)
    res = listen(a.source, a.slug, track=a.track, artist=a.artist, out_root=a.out_root, skip_split=a.skip_split,
                 viz_path=None if a.no_viz else a.viz, pitch_method=a.pitch_method)
    for name, path in res.files.items():
        print(f"{name:10s} {path}")
    for n in res.notes:
        print(f"note: {n}")
    return 0


if __name__ == "__main__":
    # torch and onnxruntime race in their native destructors at interpreter exit on macOS
    # ("recursive_mutex lock failed" -> crash dialog). Everything is written by now; leave cleanly.
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code if isinstance(code, int) else 0)
