# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Voice lines — how many voices sing at once, and where one voice runs a riff.

Input is the note list from a polyphonic transcription of the separated vocal
track (basic-pitch: dicts with start_s, dur_s, midi, salience).

Two things that look alike in that list and are not (Anne, 16 Sep 2026, the
mashup at 0:32: "you mark 2 voices at once, but I don't hear it? It's a riff,
a cengkok"):
- A second voice: two notes sounding together for a while, at an interval a
  second singer would hold.
- One voice ornamenting: a riff (cengkok) is a fast run of neighbouring notes;
  each note's tail brushes the next, and the voice's own overtone can show up
  exactly an octave above. Neither is a second singer.

Pure Python + numpy. Numbers, not moods.
"""

from __future__ import annotations

import numpy as np

GHOST_OCTAVE_RATIO = 0.7      # a note exactly an octave (or two) from a stronger one, below this share of its salience, is that voice's overtone — starting value
COUNT_SMOOTH_S = 0.5          # median smoothing of the count — starting value
SIMUL_MIN_OVERLAP_S = 0.25    # two notes must sound together this long to count as two voices — starting value
SALIENCE_MIN = 0.3            # notes quieter than this are ignored — starting value
SIMUL_MIN_INTERVAL = 3        # ...and sit at least this many semitones apart (closer is a run brushing itself) — starting value
COUNT_STEP_S = 0.1            # resolution of the voice-count timeline — starting value
OCTAVE_TOL = 0                # semitones of slack when testing for an exact octave — starting value
MAX_VOICES = 4

RIFF_MAX_NOTE_S = 0.9         # notes inside an ornament; longer is a held note that ends it — starting value
RIFF_MAX_GAP_S = 0.4          # silence allowed between notes of one ornament — set 16 Sep: the cengkok holds a 0.34 s breath gap on on-time (official basic-pitch) notes
RIFF_MAX_LEAP = 5             # an ornament may leap back up this far (a fourth) and keep going — starting value (Faouzia's 'baby': E4 D#4 C#4 B3, back up to E4)
RIFF_MAX_STEP = 3             # neighbouring notes: at most this many semitones apart — starting value
RIFF_MIN_TURNS = 2            # the line changes direction at least this often — starting value
RIFF_MIN_RATE = 1.5           # notes per second across the ornament — starting value
RIFF_MAX_RANGE = 9            # an ornament circles a band up to a sixth (semitones) — set 16 Sep: the cengkok spans 5, Faouzia's 'baby' run 8
RIFF_STEPWISE_SHARE = 0.75    # share of moves that are neighbour steps — starting value
RIFF_MIN_NOTES = 5            # at least this many notes — starting value
RIFF_MAX_MEDIAN_NOTE_S = 0.5  # most of its notes are short — set 16 Sep: the cengkok's median note is 0.42 s once sustain pieces are merged; slow melodies are still excluded by RIFF_MIN_RATE
# Calibrated on Anne's cengkok at 0:30.8-0:36.5 in the mashup: 13 notes circling A3-E3 over ~6 s,
# mostly one-semitone steps with turns, some notes held 0.5-0.9 s in between. Speed alone would miss it.

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def _end(n: dict) -> float:
    return n["start_s"] + n["dur_s"]


def remove_octave_ghosts(notes: list[dict]) -> list[dict]:
    """Drop notes that are a stronger simultaneous note's overtone (exact octave, much weaker)."""
    notes = [n for n in notes if n.get("salience", 1.0) >= SALIENCE_MIN]
    keep = []
    for n in notes:
        ghost = False
        for m in notes:
            if m is n or m["salience"] <= n["salience"] or n["midi"] <= m["midi"]:
                continue   # an overtone sits ABOVE its fundamental: only the upper, weaker note can be a ghost
            interval = abs(n["midi"] - m["midi"])
            if interval in (12, 24) or (OCTAVE_TOL and min(abs(interval - 12), abs(interval - 24)) <= OCTAVE_TOL):
                overlap = min(_end(n), _end(m)) - max(n["start_s"], m["start_s"])
                if overlap > 0.5 * n["dur_s"] and n["salience"] < GHOST_OCTAVE_RATIO * m["salience"]:
                    ghost = True
                    break
        if not ghost:
            keep.append(n)
    return keep


def count_voices(notes: list[dict], duration_s: float) -> list[int]:
    """Voices singing at once, every COUNT_STEP_S seconds (0 = silence).

    At each moment the sounding notes are grouped: notes closer than SIMUL_MIN_INTERVAL
    semitones are one line brushing itself. A group only counts as its own voice if it
    sounds together with another group for at least SIMUL_MIN_OVERLAP_S."""
    clean = remove_octave_ghosts(notes)
    steps = int(duration_s / COUNT_STEP_S) + 1
    raw = np.zeros(steps, dtype=int)
    by_step: list[list[dict]] = [[] for _ in range(steps)]
    for n in clean:
        a = max(0, int(round(n["start_s"] / COUNT_STEP_S)))
        b = min(steps, int(round(_end(n) / COUNT_STEP_S)))
        for i in range(a, b):
            by_step[i].append(n)
    for i, active in enumerate(by_step):
        if not active:
            continue
        lasting = []
        for x in active:
            for y in active:
                if x is y or abs(x["midi"] - y["midi"]) < SIMUL_MIN_INTERVAL:
                    continue
                if min(_end(x), _end(y)) - max(x["start_s"], y["start_s"]) >= SIMUL_MIN_OVERLAP_S:
                    lasting.append(x)
                    break
        if not lasting:
            raw[i] = 1
            continue
        midis = sorted({x["midi"] for x in lasting})
        groups, group_start = 1, midis[0]
        for m in midis[1:]:
            if m - group_start >= SIMUL_MIN_INTERVAL:   # bounded span: 60/62/64 is two voices, not one chain
                groups += 1
                group_start = m
        raw[i] = min(MAX_VOICES, max(1, groups))
    k = max(1, int(round(COUNT_SMOOTH_S / COUNT_STEP_S)))
    return [int(np.median(raw[max(0, i - k // 2):i + k // 2 + 1])) for i in range(steps)]


def _melody_line(notes: list[dict]) -> list[dict]:
    """One line through the notes. Overlapping neighbours (a run brushing itself) both stay;
    overlapping notes far apart keep the more salient."""
    line: list[dict] = []
    for n in sorted(notes, key=lambda n: n["start_s"]):
        # a note that SOUNDS TOGETHER with the line and sits far away is another voice: resolve that first,
        # so octave folding can't turn a simultaneous harmony into a fake stepwise run
        if line and n["start_s"] < _end(line[-1]) - 0.05 and abs(n["midi"] - line[-1]["midi"]) > RIFF_MAX_LEAP:
            if n["salience"] > line[-1]["salience"]:
                line[-1] = n
            continue
        if line and abs(n["midi"] - line[-1]["midi"]) > RIFF_MAX_LEAP and n["start_s"] - _end(line[-1]) <= RIFF_MAX_GAP_S:
            # a tracker octave slip inside a line: fold it back if that makes it a neighbour step
            for shift in (-12, 12, -24, 24):
                if abs(n["midi"] + shift - line[-1]["midi"]) <= RIFF_MAX_LEAP:
                    n = dict(n, midi=n["midi"] + shift, octave_folded=True, heard_midi=n["midi"])
                    break
        line.append(n)
    return line


def detect_riffs(notes: list[dict]) -> list[dict]:
    """Ornaments sung by one voice (riff / cengkok): a run of neighbouring notes that circles a
    narrow band, turning direction, mostly short notes."""
    line = _melody_line(remove_octave_ghosts(notes))
    runs, run = [], []
    for n in line:
        if run and (n["start_s"] - _end(run[-1]) > RIFF_MAX_GAP_S or abs(n["midi"] - run[-1]["midi"]) > RIFF_MAX_LEAP):
            runs.append(run)
            run = []
        run.append(n)
        if n["dur_s"] > RIFF_MAX_NOTE_S:   # a held note lands the ornament
            runs.append(run)
            run = [n]
    runs.append(run)

    riffs = []
    for run in runs:
        if len(run) < RIFF_MIN_NOTES:
            continue
        midis = [x["midi"] for x in run]
        moves = [b - a for a, b in zip(midis, midis[1:])]
        moving = [m for m in moves if m != 0]   # a re-sung note is part of the ornament, not a leap
        stepwise = sum(1 for m in moving if abs(m) <= RIFF_MAX_STEP) / max(1, len(moving))
        signs = [1 if m > 0 else -1 for m in moves if m != 0]
        turns = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
        span = _end(run[-1]) - run[0]["start_s"]
        rate = len(run) / max(span, 1e-6)
        med = float(np.median([x["dur_s"] for x in run]))
        if (stepwise < RIFF_STEPWISE_SHARE or turns < RIFF_MIN_TURNS or max(midis) - min(midis) > RIFF_MAX_RANGE
                or med > RIFF_MAX_MEDIAN_NOTE_S or rate < RIFF_MIN_RATE):
            continue
        # folding keeps the contour but can land the whole run an octave off; report it in the
        # octave most of its notes were actually heard in
        heard = [x.get("heard_midi", x["midi"]) for x in run]
        offset = 12 * int(round(float(np.median(np.array(heard) - np.array(midis))) / 12))
        shown = [m + offset for m in midis]
        riffs.append({
            "start_s": round(run[0]["start_s"], 2), "end_s": round(_end(run[-1]), 2), "notes": len(run),
            "notes_per_s": round(rate, 1), "low": _name(min(shown)), "high": _name(max(shown)), "turns": turns,
            "contour": " ".join(_name(m) for m in shown),
            "octave_uncertain": any(x.get("octave_folded") for x in run),
            "confidence": "high" if len(run) >= 7 and stepwise >= 0.85 and turns >= 3 else "medium",
        })
    return riffs
