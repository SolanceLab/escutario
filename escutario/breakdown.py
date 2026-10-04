# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The instrument breakdown — which instruments are really in a song, and when each one plays.

The six-stem split (vocals, drums, bass, guitar, piano, other) hides most of an arrangement inside
"other": on Anne's mashup that meant the strings, the violin, the synth and the backing vocal. The
MVSep Mega 53-stem BS-Roformer names far more, but its stems overlap and bleed: the same sound
often lights up under two names (piano and keys moved together at r = 1.0, ukulele with strings at
0.95, mandolin with violin at 0.9 on the mashup). This module turns the 53 raw stems into the
instruments a listener would name:

  * stems for one instrument are merged (piano + keys + digital piano; drums + kick + snare + toms + hi-hat);
  * a group is shown only when it plays for a meaningful share of the song;
  * a group lower in the plausibility order whose loudness moves in lockstep with a group higher in
    it (correlation >= ECHO_CORRELATION) is folded away as an echo of that one, with the reason kept.

It writes `instruments` (shown rows) and `instruments_hidden` (with reasons) into the song's
viz.json, which is the score the House site draws. Separation is heavy (about 15 minutes for a
4-minute song on the Study Mac), so it runs on demand:

    python -m escutario.breakdown <slug>            # separate all 53 stems, then summarise
    python -m escutario.breakdown <slug> --reuse    # summarise an existing out/<slug>/mega/breakdown.json
    ... --upload <heard_id>                         # put the updated score on the House gallery
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "out"

# (shown name, member stems, family, unlikely-in-modern-pop). Order is plausibility: an earlier group wins an echo tie.
ACTIVE_DB = -42.0          # a half-second counts as playing above this level (same as the separation summary)
UNLIKELY_ECHO_CORRELATION = 0.8   # an instrument rare in modern recordings needs less evidence to count as an echo
MIN_ACTIVE_PCT = 5.0       # a group must play for at least this share of the song to get a row
ECHO_CORRELATION = 0.9     # loudness moving this closely with a more plausible group of another family = the same sound, named twice
GROUPS: list[tuple[str, list[str], str, bool]] = [
    ("Lead vocal", ["lead-vocal", "vocal"], "voice", False),
    ("Backing vocal", ["back-vocal"], "voice", False),
    ("Piano", ["piano", "keys", "digital-piano"], "keys", False),
    ("Synth", ["synth"], "keys", False),
    ("Drum kit", ["drums", "kick", "snare", "toms", "hh"], "drums", False),
    ("Bass", ["bass"], "bass", False),
    ("Electric guitar", ["electric-guitar", "guitar"], "guitar", False),
    ("Acoustic guitar", ["acoustic-guitar"], "guitar", False),
    ("Strings", ["strings", "bowed_strings"], "strings", False),
    ("Violin", ["violin"], "strings", False),
    ("Viola", ["viola"], "strings", False),
    ("Cello", ["cello"], "strings", False),
    ("Double bass", ["double-bass"], "strings", False),
    ("Percussion", ["percussion", "timpani", "congas", "tambourine", "triangle"], "percussion", False),
    ("Organ", ["organ"], "keys", False),
    ("Brass", ["brass", "trumpet", "trombone", "french-horn", "tuba"], "brass", False),
    ("Woodwinds", ["woodwind", "wind", "flute", "clarinet", "oboe", "bassoon", "saxophone"], "woodwinds", False),
    ("Bells & mallets", ["bells", "glockenspiel", "marimba", "wind-chimes"], "mallets", False),
    ("Harp", ["harp"], "plucked", False),
    ("Accordion", ["accordion"], "folk", True),
    ("Harmonica", ["harmonica"], "folk", True),
    ("Harpsichord", ["harpsichord"], "folk", True),
    ("Mandolin", ["mandolin"], "folk", True),
    ("Ukulele", ["ukulele"], "folk", True),
    ("Banjo", ["banjo"], "folk", True),
    ("Dobro", ["dobro"], "folk", True),
    ("Sitar", ["sitar"], "folk", True),
]
MIN_PEAK_DB = -40.0        # ... and reach at least this loud somewhere
# Within one family (a string section: violin, viola, cello) moving together is the section playing, not an echo.


def merge_lane(lanes: list[list[float]]) -> np.ndarray:
    """Loudest member at each step (dB)."""
    n = min(len(x) for x in lanes)
    return np.max(np.array([x[:n] for x in lanes], dtype=float), axis=0)


def summarise(stems: dict[str, dict], step: float) -> dict:
    """53 raw stem summaries -> {"instruments": [...shown...], "instruments_hidden": [...with reasons...]}."""
    shown: list[dict] = []
    hidden: list[dict] = []
    kept_lanes: list[tuple[str, np.ndarray, str]] = []
    for name, members, family, unlikely in GROUPS:
        present = [m for m in members if m in stems and isinstance(stems[m].get("lane_db"), list) and stems[m]["lane_db"]]
        if not present:
            continue
        lane = merge_lane([stems[m]["lane_db"] for m in present])
        active = lane > ACTIVE_DB
        pct = float(active.mean() * 100)
        peak = float(lane.max())
        if pct < MIN_ACTIVE_PCT or peak < MIN_PEAK_DB:
            continue
        echo = None
        threshold = UNLIKELY_ECHO_CORRELATION if unlikely else ECHO_CORRELATION
        for other, other_lane, other_family in kept_lanes:
            if other_family == family:
                continue
            n = min(len(lane), len(other_lane))
            if n > 8 and np.std(lane[:n]) > 0 and np.std(other_lane[:n]) > 0:
                r = float(np.corrcoef(lane[:n], other_lane[:n])[0, 1])
                if r >= threshold and (echo is None or r > echo[1]):
                    echo = (other, r)   # name the closest match, not merely the first
        entry = {
            "name": name, "stems": present, "step": step,
            "lane_db": [round(float(v), 1) for v in lane],
            "first_s": round(float(np.argmax(active) * step), 1) if active.any() else None,
            "active_pct": round(pct, 1), "peak_db": round(peak, 1),
        }
        if echo:
            hidden.append({"name": name, "reason": f"moves with {echo[0]} (r = {echo[1]:.2f}), likely the same sound named twice",
                           "active_pct": entry["active_pct"]})
            continue
        shown.append(entry)
        kept_lanes.append((name, lane, family))
    return {"instruments": shown, "instruments_hidden": hidden}


def separate_all(song_dir: Path, log=print) -> dict:
    """Run all 53 stems in groups the GPU can hold; keep stems that play; return the breakdown summary."""
    import soundfile as sf

    from escutario.instrument import MPS_MAX_HEADS, SAMPLE_RATE, RoformerBackend, ensure_model_files, load_config

    dest = song_dir / "mega"
    dest.mkdir(exist_ok=True)
    x, sr = sf.read(song_dir / "source.wav", dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        import resampy
        x = resampy.resample(x.T, sr, SAMPLE_RATE).T
    mix = np.ascontiguousarray(x.T[:2])
    _, yaml_path = ensure_model_files(download=False)
    names = load_config(yaml_path)["training"]["instruments"]
    backend = RoformerBackend(stems=[names[0]], download=False)
    step = 0.5
    hop = int(step * SAMPLE_RATE)
    summary: dict[str, dict] = {}
    t_all = time.perf_counter()
    groups = [names[i:i + MPS_MAX_HEADS] for i in range(0, len(names), MPS_MAX_HEADS)]
    for gi, group in enumerate(groups, 1):
        t0 = time.perf_counter()
        for name, y in backend.separate_ids(mix, group).items():
            mono = y.mean(axis=0)
            n = len(mono) // hop
            db = 20 * np.log10(np.sqrt(np.mean(mono[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12))
            active = db > ACTIVE_DB
            entry = {"active_pct": round(float(active.mean() * 100), 1), "peak_db": round(float(db.max()), 1),
                     "first_active_s": float(np.argmax(active) * step) if active.any() else None,
                     "lane_db": [round(float(v), 1) for v in db]}
            if entry["active_pct"] >= 2 or entry["peak_db"] > -30:
                sf.write(dest / f"{name}.wav", y.T, SAMPLE_RATE, subtype="PCM_16")
                entry["kept"] = True
            summary[name] = entry
        log(f"group {gi}/{len(groups)} in {time.perf_counter() - t0:.0f} s on {backend.device}")
    out = {"device": backend.device, "seconds": round(time.perf_counter() - t_all), "step_s": step, "stems": summary}
    (dest / "breakdown.json").write_text(json.dumps(out))
    return out


def apply_to_score(song_dir: Path, breakdown: dict) -> dict:
    """Write the grouped instruments into the song's viz.json (the score the site draws)."""
    viz_path = song_dir / "viz.json"
    viz = json.loads(viz_path.read_text())
    grouped = summarise(breakdown["stems"], float(breakdown.get("step_s") or 0.5))
    viz.update(grouped)
    tmp = viz_path.with_name(".viz.json.tmp")
    tmp.write_text(json.dumps(viz))
    tmp.replace(viz_path)
    return grouped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m escutario.breakdown", description="Name the instruments in a heard song.")
    ap.add_argument("slug")
    ap.add_argument("--reuse", action="store_true", help="use an existing out/<slug>/mega/breakdown.json instead of separating again")
    ap.add_argument("--upload", metavar="HEARD_ID", help="upload the updated score to the House gallery for this song")
    a = ap.parse_args(argv)
    song_dir = (OUT_ROOT / a.slug).resolve()
    if song_dir.parent != OUT_ROOT.resolve() or not (song_dir / "viz.json").is_file():
        print(f"no heard song at out/{a.slug}", file=sys.stderr)
        return 2
    if a.reuse:
        breakdown = json.loads((song_dir / "mega" / "breakdown.json").read_text())
    else:
        breakdown = separate_all(song_dir)
    grouped = apply_to_score(song_dir, breakdown)
    for i in grouped["instruments"]:
        print(f"shown  {i['name']:16s} from {i['first_s']} s, {i['active_pct']}% of the song")
    for h in grouped["instruments_hidden"]:
        print(f"hidden {h['name']:16s} {h['reason']}")
    if a.upload:
        from escutario.worker import ENV_FILE, api_from_env, check_env_file, read_env
        check_env_file(ENV_FILE)
        api = api_from_env(read_env(ENV_FILE))
        print(api.put_bytes(f"/escutario/songs/{a.upload}/media/score", (song_dir / "viz.json").read_bytes(), "application/json"))
    return 0


if __name__ == "__main__":
    # torch and onnxruntime race in their native destructors at exit on macOS; leave cleanly once everything is written
    code = main()
    sys.stdout.flush()
    os._exit(code)
