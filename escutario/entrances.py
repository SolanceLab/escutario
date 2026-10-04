# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Entrances — when a part of the arrangement arrives, swells, or drops out.

Anne's first marks (16 Sep 2026, the Apparition × UNETHICAL mashup) were about
arrivals: instruments joining, strings entering, a voice coming in underneath.
This organ listens to each separated track's level over time and names the
moments where one arrives or leaves, plus the moments where several arrive
together (the arrangement opening up).

Input is a dict of stem name -> mono float32 audio at one sample rate (the
Demucs split). Pure numpy. Numbers, not moods: every event carries its rise in
dB and a confidence.
"""

from __future__ import annotations

import numpy as np

SWAP_SHARE = 0.15       # a track losing / another gaining at least this share is a candidate swap — starting value
CHANGE_WINDOW_S = 4.0   # arrangement balance compared over this much before vs after — starting value
GROUP_S = 1.5           # arrivals of different tracks this close form one "opening" — starting value
AUDIBLE_PART_DB = 24.0  # a part counts as audible within this many dB of the loudest part — starting value
AFTER_S = 3.0           # how far ahead "after" looks; an arrival must be sustained this long — starting value
FALL_DB = 10.0          # minimum sustained drop to count as leaving — starting value to calibrate
AUDIBLE_BELOW_TOP_DB = 24.0  # ...and sit within this many dB of the loudest track at that time — starting value
BANDS = 24              # log-spaced bands for the tone-colour comparison — starting value
FRAME_S = 0.5           # RMS window per step — starting value to calibrate
RISE_DB = 8.0           # minimum sustained rise to count as an arrival — starting value to calibrate
RELABEL_MIN_TIMBRE = 0.8  # ...and the sound leaving one track matches the one arriving (band-shape cosine). Set from real stems 16 Sep: relabels 0.85–0.86, drums→vocals −0.36/−0.54, vocals→other 0.01
BEFORE_S = 3.0          # how far back "before" looks — starting value to calibrate
CHANGE_PEAK_S = 6.0     # one change per this window — starting value
HIGH_RISE_DB = 14.0     # rise at or above this is high confidence — starting value
AUDIBLE_MIN_DB = -42.0  # an arrival must reach at least this level — starting value to calibrate
CHANGE_MIN = 0.25       # minimum balance shift (0..1+) to report a change — starting value to calibrate
RELABEL_MAX_DB = 1.5    # the swapping pair's combined level changes no more than this: same sound, new label — starting value
SILENT_DB = -55.0       # below this a track is effectively silent — starting value to calibrate
HOP_S = 0.25            # level envelope step — starting value to calibrate
SUPPRESS_S = 8.0        # one event per track within this window (keep the strongest) — starting value
# Why relabel exists: Anne heard piano from the first second of the mashup; the splitter filed it
# under 'other' for 8 s, then 'piano', which read as "piano arrives at 0:08". (16 Sep 2026)
DB_EPS = 1e-10


def level_envelope(x: np.ndarray, sr: int) -> np.ndarray:
    """RMS level in dBFS every HOP_S seconds over FRAME_S windows."""
    hop = max(1, int(round(HOP_S * sr)))
    frame = max(hop, int(round(FRAME_S * sr)))
    n = max(0, 1 + (len(x) - frame) // hop) if len(x) >= frame else 0
    out = np.full(n, 10 * np.log10(DB_EPS), dtype=np.float64)
    sq = np.square(x.astype(np.float64))
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    for i in range(n):
        a = i * hop
        out[i] = 10 * np.log10((csum[a + frame] - csum[a]) / frame + DB_EPS)
    return out


def _power_mean_db(env: np.ndarray, a: int, b: int) -> float | None:
    a, b = max(0, a), min(len(env), b)
    if b <= a:
        return None
    return float(10 * np.log10(np.mean(np.power(10.0, env[a:b] / 10.0)) + DB_EPS))


def detect_entrances(stems: dict[str, np.ndarray], sr: int) -> dict:
    """Find arrivals, swells and departures per track, and group simultaneous arrivals."""
    envs = {name: level_envelope(x, sr) for name, x in stems.items()}
    if not envs:
        return {"events": [], "openings": [], "hop_s": HOP_S, "notes": ["no tracks given"]}
    n = min(len(e) for e in envs.values())
    if n == 0:
        return {"events": [], "openings": [], "hop_s": HOP_S, "notes": ["audio too short"]}
    envs = {k: v[:n] for k, v in envs.items()}
    top = np.max(np.stack(list(envs.values())), axis=0)
    bw, aw = int(round(BEFORE_S / HOP_S)), int(round(AFTER_S / HOP_S))
    t_of = lambda i: round((i * HOP_S) + FRAME_S / 2, 2)

    events = []
    for name, env in envs.items():
        cands = []
        for i in range(bw, n - aw + 1):
            before = _power_mean_db(env, i - bw, i)
            after = _power_mean_db(env, i, i + aw)
            if before is None or after is None:
                continue
            delta = after - before
            top_after = _power_mean_db(top, i, i + aw)
            if delta >= RISE_DB and after >= AUDIBLE_MIN_DB and top_after is not None and after >= top_after - AUDIBLE_BELOW_TOP_DB:
                kind = "enters" if before < SILENT_DB else "rises"
                cands.append((delta, i, kind, before, after))
            elif -delta >= FALL_DB and before >= AUDIBLE_MIN_DB:
                kind = "leaves" if after < SILENT_DB else "drops"
                cands.append((-delta, i, kind, before, after))
        # strongest first, suppress neighbours of the same track
        cands.sort(key=lambda c: -c[0])
        taken: list[int] = []
        for mag, i, kind, before, after in cands:
            if any(abs(i - j) * HOP_S < SUPPRESS_S for j in taken):
                continue
            taken.append(i)
            events.append({
                "t": t_of(i), "stem": name, "kind": kind,
                "change_db": round(after - before, 1), "before_db": round(before, 1), "after_db": round(after, 1),
                "confidence": "high" if mag >= HIGH_RISE_DB else "medium",
            })
    events.sort(key=lambda e: (e["t"], e["stem"]))

    openings = []
    arrivals = [e for e in events if e["kind"] in ("enters", "rises")]
    i = 0
    while i < len(arrivals):
        group = [arrivals[i]]
        j = i + 1
        while j < len(arrivals) and arrivals[j]["t"] - group[0]["t"] <= GROUP_S:
            if arrivals[j]["stem"] not in {g["stem"] for g in group}:
                group.append(arrivals[j])
            j += 1
        if len(group) >= 2:
            openings.append({"t": group[0]["t"], "stems": [g["stem"] for g in group],
                             "total_rise_db": round(sum(g["change_db"] for g in group), 1)})
        i = j if len(group) >= 2 else i + 1

    notes = ["levels come from separated tracks; an instrument the separator files under another name (strings under 'other' or 'piano') arrives under that name"]
    return {"events": events, "openings": openings, "hop_s": HOP_S, "notes": notes}


def _band_shape(x: np.ndarray, sr: int) -> np.ndarray | None:
    """Tone colour as a normalised log-band energy profile (60 Hz to min(8 kHz, Nyquist))."""
    if len(x) < 2048 or float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) < 1e-4:
        return None
    frame, hop = 2048, 1024
    win = np.hanning(frame)
    spec = np.zeros(frame // 2 + 1)
    for a in range(0, len(x) - frame + 1, hop):
        spec += np.abs(np.fft.rfft(x[a:a + frame] * win)) ** 2
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    edges = np.geomspace(60.0, min(8000.0, sr / 2 * 0.95), BANDS + 1)
    bands = np.array([spec[(freqs >= lo) & (freqs < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])
    v = np.log10(bands + 1e-12)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def arrangement_changes(stems: dict[str, np.ndarray], sr: int, events: list | None = None) -> list:
    """Moments where the balance of the arrangement shifts most: which parts carry the sound
    before versus after, regardless of overall loudness (mastered songs stay equally loud while
    parts come and go). Returns peaks, strongest first, each naming the biggest share shifts and
    any per-track arrivals within 2 s."""
    names = list(stems)
    if not names:
        return []
    envs = [level_envelope(stems[k], sr) for k in names]
    n = min(len(e) for e in envs)
    if n == 0:
        return []
    P = np.power(10.0, np.stack([e[:n] for e in envs]) / 10.0)
    k = int(round(CHANGE_WINDOW_S / HOP_S))
    if n <= 2 * k:
        return []
    nov = np.zeros(n)
    shifts = {}
    for i in range(k, n - k):
        b, a = P[:, i - k:i].mean(1), P[:, i:i + k].mean(1)
        sb, sa = b / (b.sum() + DB_EPS), a / (a.sum() + DB_EPS)
        aud = lambda v: int((10 * np.log10(v + DB_EPS) > 10 * np.log10(v.max() + DB_EPS) - AUDIBLE_PART_DB).sum())
        nov[i] = 0.5 * float(np.abs(sa - sb).sum()) + 0.1 * abs(aud(a) - aud(b))
        shifts[i] = sa - sb
    half = int(round(CHANGE_PEAK_S / HOP_S))
    out = []
    for i in range(k, n - k):
        if nov[i] < CHANGE_MIN or nov[i] < nov[max(0, i - half):i + half + 1].max():
            continue
        t = round(i * HOP_S + FRAME_S / 2, 2)
        d = shifts[i]
        order = np.argsort(-np.abs(d))[:3]
        # a swap whose pair keeps the same combined level is the splitter relabelling one sound
        relabel = None
        losers = [j for j in range(len(names)) if d[j] <= -SWAP_SHARE]
        gainers = [j for j in range(len(names)) if d[j] >= SWAP_SHARE]
        b_all, a_all = P[:, i - k:i].mean(1), P[:, i:i + k].mean(1)
        best = None
        for lj in losers:
            for gj in gainers:
                pair_change = abs(10 * np.log10((a_all[lj] + a_all[gj] + DB_EPS) / (b_all[lj] + b_all[gj] + DB_EPS)))
                if pair_change > RELABEL_MAX_DB or (best is not None and pair_change >= best[0]):
                    continue
                hop_n = int(round(HOP_S * sr))
                s0, s1 = max(0, (i - k) * hop_n), i * hop_n
                e1 = min(len(stems[names[gj]]), (i + k) * hop_n)
                before_shape = _band_shape(stems[names[lj]][s0:s1], sr)
                after_shape = _band_shape(stems[names[gj]][s1:e1], sr)
                if before_shape is None or after_shape is None or float(np.dot(before_shape, after_shape)) < RELABEL_MIN_TIMBRE:
                    continue
                best = (pair_change, lj, gj)
        if best is not None:
            relabel = {"from": names[best[1]], "to": names[best[2]], "pair_change_db": round(best[0], 2)}
        near = [e for e in (events or []) if abs(e["t"] - t) <= 2.0 and e["kind"] in ("enters", "rises")]
        if relabel:
            near = [e for e in near if e["stem"] != relabel["to"]]
        out.append({
            "relabel": relabel,
            "t": t, "strength": round(float(nov[i]), 2),
            "shifts": [{"stem": names[j], "share_change_pct": round(float(d[j]) * 100)} for j in order if abs(d[j]) >= 0.05],
            "arrivals": [f"{e['stem']} {'enters' if e['kind'] == 'enters' else 'rises +' + str(e['change_db']) + ' dB'}" for e in near],
        })
    out.sort(key=lambda c: -c["strength"])
    return out


def format_entrances_section(result: dict, limit: int = 8) -> str:
    def mmss(t: float) -> str:
        return f"{int(t // 60)}:{t % 60:04.1f}"
    ev = result.get("events", [])
    if not ev:
        return "ARRIVALS: none detected"
    arr = [e for e in ev if e["kind"] in ("enters", "rises")]
    dep = [e for e in ev if e["kind"] in ("leaves", "drops")]
    lines = [f"ARRIVALS: {len(arr)} arrivals, {len(dep)} departures, {len(result.get('openings', []))} openings"]
    for o in result.get("openings", [])[:limit]:
        lines.append(f"   {mmss(o['t'])}  opens up: {', '.join(o['stems'])} (+{o['total_rise_db']} dB together)")
    for e in sorted(arr, key=lambda e: -e["change_db"])[:limit]:
        verb = "enters" if e["kind"] == "enters" else f"rises +{e['change_db']} dB"
        lines.append(f"   {mmss(e['t'])}  {e['stem']} {verb} ({e['confidence']})")
    return "\n".join(lines)
