# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""wail.py — voice strength, texture and the wail.

Measures, SEPARATELY, the acoustic components of what Anne hears as "wailing"
(tone + vocal strength) on a separated vocal stem:

- register position   where in the singer's own range the song sits
- ring                2–4 kHz vs 0–2 kHz energy (singer's-formant region)
- spectral tilt       alpha ratio (1–5 kHz vs 50 Hz–1 kHz) and H1–H2
- level with pitch    does the voice get louder as it climbs?
- cry falls           downward glides released at phrase ends
- register breaks     abrupt pitch flips inside continuous voicing
- grit / breathiness  HNR on sustained notes, a subharmonic indicator, and
                      whether low HNR happens loud (grit) or quiet (breathy)

There is deliberately NO combined "wail score": the components get calibrated
against Anne's songs first. Every threshold below is a starting value.
"""

from __future__ import annotations

import math

import numpy as np

from .types import PitchTrack

# ---------------------------------------------------------------------------
# Constants — every one is a STARTING VALUE to calibrate against real songs.
# ---------------------------------------------------------------------------

# -- voicing / level gates
VOICED_CONF_MIN = 0.5  # starting value to calibrate: pitch confidence at/above which a frame counts as confidently voiced
LEVEL_FLOOR_DB = 40.0  # starting value to calibrate: voiced frames this far below the loudest are ignored (bleed, reverb tails)
LEVEL_WIN_S = 0.05  # starting value to calibrate: window for the per-pitch-frame level (dB) reading
HIGH_CONF_MIN = 0.8  # starting value to calibrate: confidence required before H1–H2 is read off the spectrum

# -- contour smoothing (vibrato removal) and segmentation
NOTE_TOL_CENTS = 50.0  # starting value to calibrate: a note is smoothed pitch staying within ± this of its running mean
NOTE_MIN_S = 0.15  # starting value to calibrate: shortest span counted as a note
PHRASE_MIN_S = 0.3  # starting value to calibrate: shortest phrase reported / examined
SMOOTH_WINDOWS_S = (0.125, 0.167)  # starting value to calibrate: two cascaded box filters (~150 ms total) with nulls on 8 Hz and 6 Hz vibrato
PHRASE_GAP_S = 0.15  # starting value to calibrate: unvoiced gaps up to this long stay inside one phrase

# -- register position
REGISTER_MEDIUM_VOICED_S = 5.0  # starting value to calibrate: voiced seconds for "medium" register confidence
REGISTER_MIN_VOICED_S = 1.0  # starting value to calibrate: less confident voicing than this → register omitted
REGISTER_LOW_PCT = 5.0  # starting value to calibrate: percentile of voiced pitch taken as the bottom of the range
REGISTER_HIGH_VOICED_S = 20.0  # starting value to calibrate: voiced seconds for "high" register confidence
REGISTER_HIGH_PCT = 95.0  # starting value to calibrate: percentile of voiced pitch taken as the top of the range
REGISTER_MIN_RANGE_CENTS = 200.0  # starting value to calibrate: below this span, "top third" is meaningless and omitted

# -- STFT
STFT_WIN_S = 2048 / 22050  # starting value to calibrate: ~93 ms window (2048 @ 22.05 kHz, 4096 @ 44.1 kHz)
STFT_HOP_FRACTION = 0.25  # starting value to calibrate: hop as a fraction of the window (512 @ 2048)

# -- ring
RING_STRONG_DB = -15.0  # starting value to calibrate: ring at/above this is labelled "strong"
RING_HIGH_BAND_HZ = (2000.0, 4000.0)  # starting value to calibrate: singer's-formant band for ring
RING_WEAK_DB = -25.0  # starting value to calibrate: ring at/below this is labelled "weak"; between is "moderate"
RING_LOW_BAND_HZ = (0.0, 2000.0)  # starting value to calibrate: reference band for ring

# -- alpha ratio / H1–H2
ALPHA_LOW_BAND_HZ = (50.0, 1000.0)  # starting value to calibrate: alpha ratio low band (Sundberg/Frokjaer-Jensen convention)
HARMONIC_SEARCH_MIN_BINS = 2  # starting value to calibrate: minimum peak search half-width in bins
H1H2_MIN_BINS = 3.0  # starting value to calibrate: f0 must exceed this many STFT bin spacings before H1–H2 is read
ALPHA_FLAT_DB = -10.0  # starting value to calibrate: alpha at/above this is labelled "full/pressed" (flat tilt)
ALPHA_HIGH_BAND_HZ = (1000.0, 5000.0)  # starting value to calibrate: alpha ratio high band
H1H2_BREATHY_DB = 8.0  # starting value to calibrate: H1–H2 at/above this is labelled "breathy-leaning"; between is "modal"
H1H2_MAX_F0_DRIFT_CENTS = 50.0  # starting value to calibrate: skip H1–H2 when f0 moves more than this across the window
HARMONIC_SEARCH_FRAC = 0.05  # starting value to calibrate: peak search half-width around h·f0, as a fraction of h·f0
H1H2_PRESSED_DB = 0.0  # starting value to calibrate: H1–H2 at/below this is labelled "pressed-leaning"
NYQUIST_MARGIN = 0.95  # starting value to calibrate: H2 must sit below this fraction of Nyquist to be read
ALPHA_STEEP_DB = -20.0  # starting value to calibrate: alpha at/below this is labelled "light" (steep tilt)

# -- sample-size → confidence for spectral measures
SPECTRAL_HIGH_FRAMES = 200  # starting value to calibrate: voiced STFT frames for "high" confidence (~4.6 s)
SPECTRAL_MEDIUM_FRAMES = 50  # starting value to calibrate: voiced STFT frames for "medium" confidence
PHRASE_RING_MIN_FRAMES = 5  # starting value to calibrate: fewer frames in a phrase → no per-phrase ring
SPECTRAL_MIN_FRAMES = 10  # starting value to calibrate: fewer voiced STFT frames → measure omitted

# -- level with pitch
LWP_MIN_CENTS_STD = 100.0  # starting value to calibrate: less pitch movement than this → correlation omitted
LWP_HIGH_CENTS_STD = 300.0  # starting value to calibrate: pitch spread (cents std) also needed for "high"
LWP_MIN_N = 30  # starting value to calibrate: fewer voiced frames → correlation omitted
LWP_HIGH_N = 300  # starting value to calibrate: frames for "high" confidence
LWP_TRACKS_R = 0.3  # starting value to calibrate: |r| at/above this is worded as level rising/dropping with pitch
LWP_MEDIUM_N = 100  # starting value to calibrate: frames for "medium" confidence

# -- cry falls
FALL_MAX_CENTS = 900.0  # starting value to calibrate: bigger drops are octave slips / voice handoffs, counted as rejected, not falls
FALL_MONO_FRAC = 0.85  # starting value to calibrate: share of smoothed steps that must be non-rising
FALL_OCTAVE_SLIP_MAX_S = 0.25  # starting value to calibrate: an octave-multiple drop faster than this is a tracker slip, not a glide
FALL_END_OUTLIER_CENTS = 100.0  # starting value to calibrate: end frames further than this below their median are tracker errors, ignored
FALL_MONO_STEP_TOL_CENTS = 2.0  # starting value to calibrate: a smoothed step up to +this still counts as "not rising"
FALL_MAX_S = 0.6  # starting value to calibrate: longest glide counted as a cry fall (slower = a drift, not a fall)
FALL_ONSET_TOL_CENTS = 20.0  # starting value to calibrate: onset = where smoothed pitch leaves the pre-fall top by this
FALL_PRE_WIN_S = 0.25  # starting value to calibrate: pre-fall pitch = median over this span (≥ one 4 Hz vibrato cycle)
FALL_PRE_MIN_S = 0.1  # starting value to calibrate: need at least this much pre-fall voicing to judge a fall
FALL_MIN_S = 0.05  # starting value to calibrate: shorter than this is a jump, not a glide
FALL_SMOOTH_DROP_FRAC = 0.5  # starting value to calibrate: smoothed contour must also fall ≥ this × FALL_MIN_CENTS
FALL_MIN_CENTS = 150.0  # starting value to calibrate: smallest drop counted as a cry fall
FALL_HIGH_DROP_CENTS = 300.0  # starting value to calibrate: drop for "high" fall confidence (with full pre-window)
FALL_END_WIN_S = 0.03  # starting value to calibrate: fall bottom = lowest raw pitch over this final span

# -- register breaks
BREAK_HOLD_S = 0.05  # starting value to calibrate: both sides must hold their pitch this long (kills 1-frame octave errors)
BREAK_MIN_CENTS = 400.0  # starting value to calibrate: jump bigger than this counts as a flip
BREAK_SLIP_NEIGHBOUR_S = 0.5  # starting value to calibrate: a flip this close to an octave slip belongs to the same slip burst
VOICE_CHANGE_MIN_CENTS = 1100.0  # starting value to calibrate: jumps at/above this are voice changes / octave slips, not register flips
BREAK_MAX_S = 0.06  # starting value to calibrate: the jump must happen within this span
MULTI_VOICE_SLIPS_PER_MIN = 3.0  # starting value to calibrate: slips per voiced minute at/above which pitch-based measures drop to "low"
OCTAVE_TOL_CENTS = 80.0  # starting value to calibrate: a jump within ± this of a multiple of 1200 ¢ counts as an octave multiple
MULTI_VOICE_MIN_SLIPS = 5  # starting value to calibrate: at least this many slips before suspecting a multi-voice stem

# -- grit / HNR
HNR_HIGH_FRAMES = 100  # starting value to calibrate: HNR frames for "high" confidence (2 s of sustained notes)
GRIT_REGION_HIGH_S = 0.4  # starting value to calibrate: region duration for "high" confidence
GRIT_MERGE_GAP_S = 0.1  # starting value to calibrate: low-HNR frames this close join one region
GRIT_LEVEL_REL_DB = 0.0  # starting value to calibrate: low HNR at/above (median voiced level + this) = grit
HNR_MIN_WIN_S = 0.02  # starting value to calibrate: shortest HNR window
HNR_PERIODS = 4.5  # starting value to calibrate: HNR window length in pitch periods
HNR_MAX_WIN_S = 0.08  # starting value to calibrate: longest HNR window (f0 below ~56 Hz is skipped)
HNR_LAG_SEARCH_FRAC = 0.1  # starting value to calibrate: autocorrelation peak search ± this fraction around 1/f0
HNR_LOW_DB = 12.0  # starting value to calibrate: HNR below this marks a rough (noisy) stretch
GRIT_MIN_REGION_S = 0.2  # starting value to calibrate: shortest low-HNR stretch reported
HNR_CAP_DB = 40.0  # starting value to calibrate: HNR reported no higher than this (estimator ceiling)
SUBHARMONIC_PRESENT_DB = -20.0  # starting value to calibrate: energy at f0/2 within this of H1 = subharmonics present
BREATHY_LEVEL_REL_DB = -6.0  # starting value to calibrate: low HNR at/below (median voiced level + this) = breathy
HNR_STEP_S = 0.02  # starting value to calibrate: HNR sampled every this many seconds on sustained notes
HNR_MEDIUM_FRAMES = 25  # starting value to calibrate: HNR frames for "medium" confidence
HNR_MIN_FRAMES = 10  # starting value to calibrate: fewer HNR frames → HNR omitted

# -- card
CARD_MAX_REGIONS = 3  # starting value to calibrate: grit regions listed on the card
OCTAVE_CENTS = 1200.0  # one octave in cents (a definition, not a threshold)

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_EPS = 1e-20


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _hz_to_cents(hz: np.ndarray) -> np.ndarray:
    """MIDI-cents: 6900 = A4 (440 Hz)."""
    hz = np.asarray(hz, dtype=np.float64)
    out = np.full(hz.shape, np.nan)
    ok = hz > 0
    out[ok] = 6900.0 + 1200.0 * np.log2(hz[ok] / 440.0)
    return out


def note_name(cents: float) -> str:
    m = int(round(cents / 100.0))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def _fmt_time(t: float) -> str:
    t = max(0.0, float(t))
    m = int(t // 60)
    return f"{m}:{int(t - 60 * m):02d}"


def _fmt_time_precise(t: float) -> str:
    t = max(0.0, float(t))
    m = int(t // 60)
    return f"{m}:{t - 60 * m:04.1f}"


def _near_octave_multiple(cents: float) -> bool:
    k = round(abs(cents) / OCTAVE_CENTS)
    return k >= 1 and abs(abs(cents) - k * OCTAVE_CENTS) <= OCTAVE_TOL_CENTS


def _db(p: float) -> float:
    return 10.0 * math.log10(max(p, _EPS))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, stop) with stop exclusive."""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def _box_edge(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average of odd length n that shrinks at the edges."""
    if n <= 1 or x.size == 0:
        return x.copy()
    h = n // 2
    c = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    idx = np.arange(x.size)
    lo = np.clip(idx - h, 0, x.size)
    hi = np.clip(idx + h + 1, 0, x.size)
    return (c[hi] - c[lo]) / (hi - lo)


def _smooth_run(cents: np.ndarray, hop_s: float) -> np.ndarray:
    out = cents
    for w in SMOOTH_WINDOWS_S:
        n = max(1, int(round(w / hop_s)))
        if n % 2 == 0:
            n += 1
        out = _box_edge(out, n)
    return out


def _count_conf(n: int, high: int, medium: int, minimum: int) -> str | None:
    if n >= high:
        return "high"
    if n >= medium:
        return "medium"
    if n >= minimum:
        return "low"
    return None


def _frame_levels_db(x: np.ndarray, sr: int, times: np.ndarray) -> np.ndarray:
    """RMS level in dB of a LEVEL_WIN_S window centred on each time."""
    if x.size == 0 or times.size == 0:
        return np.full(times.shape, -np.inf)
    c = np.concatenate(([0.0], np.cumsum(np.square(x, dtype=np.float64))))
    half = max(1, int(round(LEVEL_WIN_S * sr / 2)))
    centre = np.round(times * sr).astype(np.int64)
    lo = np.clip(centre - half, 0, x.size)
    hi = np.clip(centre + half, 0, x.size)
    n = np.maximum(hi - lo, 1)
    p = (c[hi] - c[lo]) / n
    return 10.0 * np.log10(np.maximum(p, _EPS))


def _parabolic(y_m1: float, y0: float, y_p1: float) -> tuple[float, float]:
    """Vertex offset and height of a parabola through three equally spaced points."""
    denom = y_m1 - 2.0 * y0 + y_p1
    if abs(denom) < 1e-12:
        return 0.0, y0
    off = 0.5 * (y_m1 - y_p1) / denom
    off = max(-1.0, min(1.0, off))
    return off, y0 - 0.25 * (y_m1 - y_p1) * off


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def analyze_voice(vocal: np.ndarray, sr: int, pitch: PitchTrack) -> dict:
    """Measure the components of the wail on a separated mono vocal stem."""
    x = np.asarray(vocal, dtype=np.float32).reshape(-1)
    times = np.asarray(pitch.times, dtype=np.float64)
    f0 = np.asarray(pitch.f0_hz, dtype=np.float64)
    conf = np.asarray(pitch.confidence, dtype=np.float64)
    hop_s = float(pitch.hop_s) if pitch.hop_s and pitch.hop_s > 0 else 0.01

    notes: list[str] = [
        "Measured on a separated vocal stem: separation artefacts and reverb lower HNR and "
        "can add 2–4 kHz energy; compare songs processed the same way.",
        "All thresholds are starting values awaiting calibration on reference songs; no combined wail score.",
    ]

    levels = _frame_levels_db(x, sr, times)
    mask = (f0 > 0) & (conf >= VOICED_CONF_MIN)
    if np.any(mask):
        top = float(np.max(levels[mask]))
        mask &= levels >= top - LEVEL_FLOOR_DB
    cents = _hz_to_cents(f0)

    # smoothed contour per contiguous confident run
    runs = [(a, b) for a, b in _runs(mask)]
    smooth = np.full(cents.shape, np.nan)
    for a, b in runs:
        smooth[a:b] = _smooth_run(cents[a:b], hop_s)

    phrases = _phrases(runs, times, hop_s)
    note_segs = _segment_notes(smooth, cents, runs, hop_s)
    voiced_s = float(np.count_nonzero(mask) * hop_s)
    ref_level = float(np.median(levels[mask])) if np.any(mask) else None

    register = _register(smooth, cents, mask, times, note_segs, hop_s, voiced_s, notes)
    result: dict = {
        "voiced_s": round(voiced_s, 2),
        "phrase_count": len(phrases),
        "register": register,
        "level_with_pitch": _level_with_pitch(levels, smooth, mask, notes),
    }

    top_thr = register.get("top_third_from_cents") if register else None
    spectral = _spectral(x, sr, times, f0, conf, mask, cents, smooth, top_thr, phrases, notes)
    result["ring"] = spectral["ring"]
    result["tilt"] = spectral["tilt"]

    jumps = _breaks(cents, runs, times, hop_s)
    flips, slips = _classify_jumps(jumps)
    result["breaks"] = flips
    result["voice_changes"] = {
        "count": len(slips),
        "octave_multiple": sum(1 for e in slips if e["reason"] == "octave_multiple"),
        "over_max": sum(1 for e in slips if e["reason"] == "over_max"),
        "near_slip": sum(1 for e in slips if e["reason"] == "near_slip"),
        "events": slips,
    }
    if slips:
        notes.append(f"{len(slips)} pitch jumps set aside as octave slips / voice changes (octave multiples, "
                     f"jumps ≥ {VOICE_CHANGE_MIN_CENTS:g} ¢, or flips inside those bursts): the stem likely holds "
                     "more than one voice (duet, harmony, doubles) or the pitch tracker slipped an octave. "
                     "Register range may be widened by the same slips.")
    voiced_min = voiced_s / 60.0
    multi = (len(slips) >= MULTI_VOICE_MIN_SLIPS and voiced_min > 0
             and len(slips) / voiced_min >= MULTI_VOICE_SLIPS_PER_MIN)
    result["multi_voice_suspected"] = bool(multi)
    if multi:
        for key in ("register", "level_with_pitch"):
            if result[key] is not None:
                result[key]["confidence"] = "low"
        if result["ring"] is not None and result["ring"]["top_third_confidence"] is not None:
            result["ring"]["top_third_confidence"] = "low"
        if result["tilt"] is not None and result["tilt"]["alpha_top_third_confidence"] is not None:
            result["tilt"]["alpha_top_third_confidence"] = "low"
        notes.append(f"Multi-voice stem suspected ({len(slips) / voiced_min:.0f} slips per voiced minute): range, "
                     "top-third share, level-with-pitch and top-third ring/alpha mix voices, confidence capped at low.")
    falls, rejected = _falls(cents, smooth, conf, phrases, times, hop_s, flips, slips)
    result["falls"] = falls
    result["falls_rejected"] = rejected
    if rejected["octave_slip"] or rejected["over_max"]:
        notes.append(f"Fall candidates rejected: {rejected['octave_slip']} octave slip / voice handoff "
                     f"(octave-multiple drop faster than {FALL_OCTAVE_SLIP_MAX_S:g} s, or crossing a voice change), "
                     f"{rejected['over_max']} deeper than {FALL_MAX_CENTS:g} ¢.")
    result["grit"] = _grit(x, sr, f0, conf, mask, times, levels, ref_level, note_segs,
                           spectral["subharmonic_frames"], notes)
    result["notes"] = notes
    return result


# ---------------------------------------------------------------------------
# Structure: phrases, notes
# ---------------------------------------------------------------------------

def _phrases(runs: list[tuple[int, int]], times: np.ndarray, hop_s: float) -> list[dict]:
    phrases: list[dict] = []
    gap_frames = PHRASE_GAP_S / hop_s
    for a, b in runs:
        if phrases and (a - phrases[-1]["runs"][-1][1]) <= gap_frames:
            phrases[-1]["runs"].append((a, b))
        else:
            phrases.append({"runs": [(a, b)]})
    out = []
    for p in phrases:
        a = p["runs"][0][0]
        b = p["runs"][-1][1]
        start = float(times[a] - hop_s / 2)
        end = float(times[b - 1] + hop_s / 2)
        if end - start >= PHRASE_MIN_S:
            out.append({"start_s": start, "end_s": end, "runs": p["runs"]})
    return out


def _segment_notes(smooth: np.ndarray, cents: np.ndarray, runs, hop_s: float) -> list[dict]:
    """Greedy segmentation: a note is smoothed pitch within ±NOTE_TOL_CENTS of its running mean."""
    min_frames = max(1, int(math.ceil(NOTE_MIN_S / hop_s - 1e-9)))
    out: list[dict] = []
    for a, b in runs:
        i = a
        while i < b:
            total = smooth[i]
            cnt = 1
            j = i + 1
            while j < b and abs(smooth[j] - total / cnt) <= NOTE_TOL_CENTS:
                total += smooth[j]
                cnt += 1
                j += 1
            if j - i >= min_frames:
                out.append({"i0": i, "i1": j, "cents": float(np.median(cents[i:j]))})
            i = j
    return out


# ---------------------------------------------------------------------------
# Register position
# ---------------------------------------------------------------------------

def _register(smooth, cents, mask, times, note_segs, hop_s, voiced_s, notes) -> dict | None:
    if voiced_s < REGISTER_MIN_VOICED_S:
        notes.append("Register omitted: too little confidently voiced singing.")
        return None
    vals = smooth[mask]
    lo = float(np.percentile(vals, REGISTER_LOW_PCT))
    hi = float(np.percentile(vals, REGISTER_HIGH_PCT))
    span = hi - lo
    conf = "high" if voiced_s >= REGISTER_HIGH_VOICED_S else (
        "medium" if voiced_s >= REGISTER_MEDIUM_VOICED_S else "low")
    reg = {
        "low_note": note_name(lo),
        "high_note": note_name(hi),
        "low_cents": round(lo, 1),
        "high_cents": round(hi, 1),
        "range_cents": round(span, 1),
        "top_third_from_note": None,
        "top_third_from_cents": None,
        "top_third_share": None,
        "longest_top_note": None,
        "confidence": conf,
    }
    if span < REGISTER_MIN_RANGE_CENTS:
        notes.append("Top-third share omitted: the sung range is too narrow to split into thirds.")
        return reg
    thr = hi - span / 3.0
    reg["top_third_from_note"] = note_name(thr)
    reg["top_third_from_cents"] = round(thr, 1)
    reg["top_third_share"] = round(float(np.mean(vals >= thr)), 3)
    top = [n for n in note_segs if n["cents"] >= thr]
    if top:
        best = max(top, key=lambda n: n["i1"] - n["i0"])
        reg["longest_top_note"] = {
            "note": note_name(best["cents"]),
            "start_s": round(float(times[best["i0"]] - hop_s / 2), 2),
            "duration_s": round((best["i1"] - best["i0"]) * hop_s, 2),
        }
    return reg


# ---------------------------------------------------------------------------
# Level with pitch
# ---------------------------------------------------------------------------

def _level_with_pitch(levels, smooth, mask, notes) -> dict | None:
    n = int(np.count_nonzero(mask))
    if n < LWP_MIN_N:
        return None
    c = smooth[mask]
    d = levels[mask]
    sd_c = float(np.std(c))
    if sd_c < LWP_MIN_CENTS_STD or float(np.std(d)) < 1e-6:
        notes.append("Level-with-pitch omitted: too little pitch movement to correlate.")
        return None
    r = float(np.corrcoef(c, d)[0, 1])
    slope = float(np.polyfit(c / 1200.0, d, 1)[0])
    if n >= LWP_HIGH_N and sd_c >= LWP_HIGH_CENTS_STD:
        confidence = "high"
    elif n >= LWP_MEDIUM_N:
        confidence = "medium"
    else:
        confidence = "low"
    return {"r": round(r, 3), "slope_db_per_octave": round(slope, 2), "n": n, "confidence": confidence}


# ---------------------------------------------------------------------------
# Spectral measures: ring, alpha ratio, H1–H2, subharmonic
# ---------------------------------------------------------------------------

def _stft_params(sr: int) -> tuple[int, int]:
    n_fft = int(2 ** round(math.log2(STFT_WIN_S * sr)))
    hop = max(1, int(round(n_fft * STFT_HOP_FRACTION)))
    return n_fft, hop


def _nearest(times: np.ndarray, t: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(times, t)
    idx = np.clip(idx, 1, max(1, times.size - 1))
    left = times[idx - 1]
    right = times[np.minimum(idx, times.size - 1)]
    return np.where(np.abs(t - left) <= np.abs(right - t), idx - 1, idx).clip(0, times.size - 1)


def _peak_db(mag_db: np.ndarray, freq: float, df: float, frac: float, min_bins: int) -> float | None:
    k = freq / df
    half = max(min_bins, int(math.ceil(k * frac)))
    lo = max(1, int(round(k)) - half)
    hi = min(mag_db.size - 2, int(round(k)) + half)
    if hi <= lo:
        return None
    i = lo + int(np.argmax(mag_db[lo:hi + 1]))
    _, h = _parabolic(mag_db[i - 1], mag_db[i], mag_db[i + 1])
    return float(h)


def _band_idx(n_fft: int, sr: int, band: tuple[float, float]) -> tuple[int, int]:
    df = sr / n_fft
    lo = int(math.ceil(band[0] / df))
    hi = min(n_fft // 2, int(math.floor(band[1] / df)))
    return lo, hi + 1


def _spectral(x, sr, times, f0, conf, mask, cents, smooth, top_thr, phrases, notes) -> dict:
    empty = {"ring": None, "tilt": None, "subharmonic_frames": []}
    if x.size == 0 or times.size == 0 or not np.any(mask):
        return empty
    n_fft, hop = _stft_params(sr)
    df = sr / n_fft
    n_frames = 1 + x.size // hop
    t_frames = np.arange(n_frames) * hop / sr
    q = n_fft / sr / 4.0
    i_c = _nearest(times, t_frames)
    i_l = _nearest(times, t_frames - q)
    i_r = _nearest(times, t_frames + q)
    tol = max(float(np.median(np.diff(times))) if times.size > 1 else 0.0, hop / sr)
    ok = (mask[i_c] & mask[i_l] & mask[i_r] & (np.abs(times[i_c] - t_frames) <= tol))
    sel = np.flatnonzero(ok)
    if sel.size == 0:
        return empty

    pad = n_fft // 2
    xp = np.concatenate((np.zeros(pad, np.float32), x, np.zeros(n_fft, np.float32)))
    win = np.hanning(n_fft).astype(np.float64)
    rl = _band_idx(n_fft, sr, RING_LOW_BAND_HZ)
    rh = _band_idx(n_fft, sr, RING_HIGH_BAND_HZ)
    al = _band_idx(n_fft, sr, ALPHA_LOW_BAND_HZ)
    ah = _band_idx(n_fft, sr, ALPHA_HIGH_BAND_HZ)

    ring_lo = np.zeros(sel.size)
    ring_hi = np.zeros(sel.size)
    alpha_lo = np.zeros(sel.size)
    alpha_hi = np.zeros(sel.size)
    h1h2: list[float] = []
    sub: list[tuple[float, float]] = []  # (time, dB of f0/2 relative to H1)

    chunk = 512
    for c0 in range(0, sel.size, chunk):
        ids = sel[c0:c0 + chunk]
        starts = ids * hop
        frames = np.stack([xp[s:s + n_fft] for s in starts]).astype(np.float64) * win
        power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        cs = np.cumsum(power, axis=1)
        cs = np.concatenate((np.zeros((cs.shape[0], 1)), cs), axis=1)
        ring_lo[c0:c0 + ids.size] = cs[:, rl[1]] - cs[:, rl[0]]
        ring_hi[c0:c0 + ids.size] = cs[:, rh[1]] - cs[:, rh[0]]
        alpha_lo[c0:c0 + ids.size] = cs[:, al[1]] - cs[:, al[0]]
        alpha_hi[c0:c0 + ids.size] = cs[:, ah[1]] - cs[:, ah[0]]
        mag_db = 10.0 * np.log10(np.maximum(power, _EPS))
        for row, fid in enumerate(ids):
            pc = i_c[fid]
            fz = f0[pc]
            h1 = None
            if fz > H1H2_MIN_BINS * df and 2 * fz < sr / 2 * NYQUIST_MARGIN:
                h1 = _peak_db(mag_db[row], fz, df, HARMONIC_SEARCH_FRAC, HARMONIC_SEARCH_MIN_BINS)
                drift = abs(cents[i_r[fid]] - cents[i_l[fid]])
                if (h1 is not None and conf[pc] >= HIGH_CONF_MIN and drift <= H1H2_MAX_F0_DRIFT_CENTS):
                    h2 = _peak_db(mag_db[row], 2 * fz, df, HARMONIC_SEARCH_FRAC, HARMONIC_SEARCH_MIN_BINS)
                    if h2 is not None:
                        h1h2.append(h1 - h2)
            if h1 is not None and fz / 2 > H1H2_MIN_BINS * df:
                s_db = _peak_db(mag_db[row], fz / 2, df, HARMONIC_SEARCH_FRAC, 1)
                if s_db is not None:
                    sub.append((float(t_frames[fid]), s_db - h1))

    n = int(sel.size)
    conf_label = _count_conf(n, SPECTRAL_HIGH_FRAMES, SPECTRAL_MEDIUM_FRAMES, SPECTRAL_MIN_FRAMES)
    if conf_label is None:
        notes.append("Ring and tilt omitted: too few voiced analysis frames.")
        return {"ring": None, "tilt": None, "subharmonic_frames": sub}

    ring_db = _db(ring_hi.sum()) - _db(ring_lo.sum())
    per_phrase = []
    ts = t_frames[sel]
    for p in phrases:
        m = (ts >= p["start_s"]) & (ts <= p["end_s"])
        k = int(np.count_nonzero(m))
        if k >= PHRASE_RING_MIN_FRAMES:
            per_phrase.append({
                "start_s": round(p["start_s"], 2),
                "end_s": round(p["end_s"], 2),
                "ring_db": round(_db(ring_hi[m].sum()) - _db(ring_lo[m].sum()), 1),
                "n_frames": k,
                "confidence": _count_conf(k, SPECTRAL_HIGH_FRAMES, SPECTRAL_MEDIUM_FRAMES,
                                          PHRASE_RING_MIN_FRAMES),
            })
    ring = {
        "overall_db": round(ring_db, 1),
        "label": ("strong" if ring_db >= RING_STRONG_DB else "weak" if ring_db <= RING_WEAK_DB else "moderate"),
        "n_frames": n,
        "confidence": conf_label,
        "per_phrase": per_phrase,
        "top_third_db": None,
        "top_third_n_frames": 0,
        "top_third_confidence": None,
    }
    top_alpha = None
    top_conf = None
    if top_thr is not None:
        top = smooth[i_c[sel]] >= top_thr
        k = int(np.count_nonzero(top))
        top_conf = _count_conf(k, SPECTRAL_HIGH_FRAMES, SPECTRAL_MEDIUM_FRAMES, SPECTRAL_MIN_FRAMES)
        ring["top_third_n_frames"] = k
        if top_conf is not None:
            ring["top_third_db"] = round(_db(ring_hi[top].sum()) - _db(ring_lo[top].sum()), 1)
            ring["top_third_confidence"] = top_conf
            top_alpha = round(_db(alpha_hi[top].sum()) - _db(alpha_lo[top].sum()), 1)

    alpha = _db(alpha_hi.sum()) - _db(alpha_lo.sum())
    tilt = {
        "alpha_ratio_db": round(alpha, 1),
        "alpha_label": ("full/pressed" if alpha >= ALPHA_FLAT_DB else "light" if alpha <= ALPHA_STEEP_DB
                        else "moderate tilt"),
        "n_frames": n,
        "confidence": conf_label,
        "h1_h2_db": None,
        "h1_h2_label": None,
        "h1_h2_n_frames": len(h1h2),
        "h1_h2_confidence": None,
        "alpha_top_third_db": top_alpha,
        "alpha_top_third_confidence": top_conf if top_alpha is not None else None,
    }
    h_conf = _count_conf(len(h1h2), SPECTRAL_HIGH_FRAMES, SPECTRAL_MEDIUM_FRAMES, SPECTRAL_MIN_FRAMES)
    if h_conf is not None:
        v = float(np.median(h1h2))
        tilt["h1_h2_db"] = round(v, 1)
        tilt["h1_h2_label"] = ("pressed-leaning" if v <= H1H2_PRESSED_DB else "breathy-leaning"
                               if v >= H1H2_BREATHY_DB else "modal")
        tilt["h1_h2_confidence"] = h_conf
    else:
        notes.append("H1–H2 omitted: too few high-confidence, stable-pitch frames.")
    notes.append("Alpha ratio rises with pitch on its own (high notes put H1 above 1 kHz); "
                 "H1–H2 is uncorrected for formants (H1*–H2*).")
    return {"ring": ring, "tilt": tilt, "subharmonic_frames": sub}


# ---------------------------------------------------------------------------
# Register breaks
# ---------------------------------------------------------------------------

def _breaks(cents, runs, times, hop_s) -> list[dict]:
    J = max(1, int(round(BREAK_MAX_S / hop_s)))
    H = max(1, int(round(BREAK_HOLD_S / hop_s)))
    events: list[dict] = []
    for a, b in runs:
        if b - a < 2 * H + 1:
            continue
        c = cents[a:b]
        n = c.size
        # jump over up to J frames, clipped at the run's end (voicing continuous by construction)
        jump = np.zeros(n)
        for i in range(n - 1):
            k = min(n - 1, i + J)
            jump[i] = c[k] - c[i]
        for sign in (1, -1):
            hits = (sign * jump) > BREAK_MIN_CENTS
            for g0, g1 in _runs(hits):
                last = g1 - 1
                pre_hi = g0 + 1
                pre_lo = pre_hi - H
                post_lo = min(n - 1, last + J)
                post_hi = post_lo + H
                if pre_lo < 0 or post_hi > n:
                    continue
                before = float(np.median(c[pre_lo:pre_hi]))
                after = float(np.median(c[post_lo:post_hi]))
                if sign * (after - before) <= BREAK_MIN_CENTS:
                    continue
                steps = np.diff(c[g0:min(n, last + J + 1)])
                at = g0 + int(np.argmax(sign * steps)) if steps.size else g0
                events.append({
                    "time_s": round(float(times[a + at] + hop_s / 2), 3),
                    "from_note": note_name(before),
                    "to_note": note_name(after),
                    "jump_cents": round(after - before, 0),
                    "direction": "up" if sign > 0 else "down",
                    "confidence": "high" if abs(after - before) >= 2 * BREAK_MIN_CENTS else "medium",
                })
    events.sort(key=lambda e: e["time_s"])
    return events


def _classify_jumps(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split jumps into register flips and octave slips / voice changes."""
    flips: list[dict] = []
    slips: list[dict] = []
    for e in events:
        size = abs(e["jump_cents"])
        if _near_octave_multiple(size):
            slips.append({**e, "reason": "octave_multiple"})
        elif size >= VOICE_CHANGE_MIN_CENTS:
            slips.append({**e, "reason": "over_max"})
        else:
            flips.append(e)
    kept: list[dict] = []
    for e in flips:
        if any(abs(e["time_s"] - s_["time_s"]) <= BREAK_SLIP_NEIGHBOUR_S
               for s_ in slips if s_["reason"] != "near_slip"):
            slips.append({**e, "reason": "near_slip"})
        else:
            kept.append(e)
    slips.sort(key=lambda e: e["time_s"])
    return kept, slips


# ---------------------------------------------------------------------------
# Cry falls
# ---------------------------------------------------------------------------

def _falls(cents, smooth, conf, phrases, times, hop_s, flips, slips) -> tuple[list[dict], dict]:
    out: list[dict] = []
    rejected = {"octave_slip": 0, "over_max": 0}
    W = max(2, int(round(FALL_MAX_S / hop_s)))
    P = max(1, int(round(FALL_PRE_WIN_S / hop_s)))
    P_min = max(1, int(round(FALL_PRE_MIN_S / hop_s)))
    E = max(1, int(round(FALL_END_WIN_S / hop_s)))
    for p in phrases:
        a, b = p["runs"][-1]
        e = b - 1
        if e - a < P_min + 2:
            continue
        s = smooth
        rs = max(a + P_min, e - W)
        if rs >= e:
            continue
        k_max = rs + int(np.argmax(s[rs:e + 1]))
        # a descent that was already under way before the window is a drift, not a cry fall
        if k_max == rs and rs > a:
            back = max(a, rs - W // 2)
            if float(np.max(s[back:rs])) > s[rs] + FALL_ONSET_TOL_CENTS:
                continue
        below = np.flatnonzero(s[k_max:e + 1] < s[k_max] - FALL_ONSET_TOL_CENTS)
        if below.size == 0:
            continue
        onset = k_max + max(0, int(below[0]) - 1)
        if float(times[e] - times[onset]) > FALL_MAX_S + hop_s:
            continue
        pre_lo = max(a, onset - P)
        if onset - pre_lo < P_min:
            continue
        pre_level = float(np.median(cents[pre_lo:onset + 1]))
        tail = cents[max(onset, e - E + 1):e + 1]
        tail_med = float(np.median(tail))
        end_level = float(np.min(tail[tail >= tail_med - FALL_END_OUTLIER_CENTS]))
        drop = pre_level - end_level
        if drop < FALL_MIN_CENTS:
            continue
        steps = np.diff(s[onset:e + 1])
        if steps.size == 0 or float(np.mean(steps <= FALL_MONO_STEP_TOL_CENTS)) < FALL_MONO_FRAC:
            continue
        if s[onset] - s[e] < FALL_MIN_CENTS * FALL_SMOOTH_DROP_FRAC:
            continue
        # timing from raw frames: smoothing stretches a fast drop, so the start is the last raw
        # frame still within FALL_ONSET_TOL_CENTS of the pre-fall pitch
        still_up = np.flatnonzero(cents[onset:e + 1] >= pre_level - FALL_ONSET_TOL_CENTS)
        raw_onset = onset + (int(still_up[-1]) if still_up.size else 0)
        t0 = float(times[raw_onset])
        t1 = float(times[e])
        duration = t1 - t0
        if any(t0 - hop_s <= br["time_s"] <= t1 + hop_s for br in flips):
            continue  # the drop is a register flip, already reported as a break
        if (any(t0 - hop_s <= sl["time_s"] <= t1 + hop_s for sl in slips)
                or (_near_octave_multiple(drop) and duration < FALL_OCTAVE_SLIP_MAX_S)):
            rejected["octave_slip"] += 1
            continue
        if drop > FALL_MAX_CENTS:
            rejected["over_max"] += 1
            continue
        if duration < FALL_MIN_S:
            continue  # a step, not a glide
        full_pre = onset - pre_lo >= P
        mean_conf = float(np.mean(conf[onset:e + 1]))
        if full_pre and drop >= FALL_HIGH_DROP_CENTS and mean_conf >= HIGH_CONF_MIN:
            c_label = "high"
        elif full_pre:
            c_label = "medium"
        else:
            c_label = "low"
        out.append({
            "start_s": round(t0, 2),
            "end_s": round(t1, 2),
            "duration_s": round(duration, 2),
            "start_note": note_name(pre_level),
            "drop_cents": round(drop, 0),
            "confidence": c_label,
        })
    return out, rejected


# ---------------------------------------------------------------------------
# Grit / breathiness
# ---------------------------------------------------------------------------

_WIN_AC_CACHE: dict[int, np.ndarray] = {}


def _window_autocorr(n: int) -> np.ndarray:
    ac = _WIN_AC_CACHE.get(n)
    if ac is None:
        w = np.hanning(n)
        nfft = 1 << (2 * n - 1).bit_length()
        ac = np.fft.ifft(np.abs(np.fft.fft(w, nfft)) ** 2).real[:n]
        ac = ac / ac[0]
        _WIN_AC_CACHE[n] = ac
    return ac


def hnr_frame(seg: np.ndarray, sr: int, f0: float) -> float | None:
    """Autocorrelation HNR (Boersma 1993 window correction) of one frame at a known f0."""
    n = seg.size
    if n < 8 or f0 <= 0:
        return None
    t0 = sr / f0
    lag_lo = max(2, int(math.floor(t0 * (1 - HNR_LAG_SEARCH_FRAC))))
    lag_hi = int(math.ceil(t0 * (1 + HNR_LAG_SEARCH_FRAC)))
    if lag_hi + 1 >= n // 2:
        return None
    y = (seg - np.mean(seg)) * np.hanning(n)
    nfft = 1 << (2 * n - 1).bit_length()
    ac = np.fft.ifft(np.abs(np.fft.fft(y, nfft)) ** 2).real[:n]
    if ac[0] <= _EPS:
        return None
    r = (ac / ac[0]) / np.maximum(_window_autocorr(n), 1e-9)
    i = lag_lo + int(np.argmax(r[lag_lo:lag_hi + 1]))
    _, peak = _parabolic(r[i - 1], r[i], r[i + 1])
    r_max = 1.0 / (1.0 + 10 ** (-HNR_CAP_DB / 10))
    peak = min(max(peak, 1e-6), r_max)
    return float(10.0 * math.log10(peak / (1.0 - peak)))


def _grit(x, sr, f0, conf, mask, times, levels, ref_level, note_segs, sub_frames, notes) -> dict | None:
    if x.size == 0 or not note_segs or ref_level is None:
        return None
    frame_s = float(times[1] - times[0]) if times.size > 1 else HNR_STEP_S
    step = max(1, int(round(HNR_STEP_S / frame_s)))
    step_s = step * frame_s
    rows: list[tuple[float, float, float]] = []  # (time, hnr, level)
    for nseg in note_segs:
        for i in range(nseg["i0"], nseg["i1"], step):
            if not mask[i]:
                continue
            fz = f0[i]
            n = int(round(min(max(HNR_PERIODS * sr / fz, HNR_MIN_WIN_S * sr), HNR_MAX_WIN_S * sr)))
            n += n % 2
            c = int(round(times[i] * sr))
            lo, hi = c - n // 2, c + n // 2
            if lo < 0 or hi > x.size:
                continue
            h = hnr_frame(x[lo:hi].astype(np.float64), sr, fz)
            if h is not None:
                rows.append((float(times[i]), h, float(levels[i])))
    conf_label = _count_conf(len(rows), HNR_HIGH_FRAMES, HNR_MEDIUM_FRAMES, HNR_MIN_FRAMES)
    if conf_label is None:
        notes.append("Grit omitted: too few sustained-note frames for HNR.")
        return None
    hnr = np.array([r[1] for r in rows])

    regions: list[dict] = []
    cur: list[tuple[float, float, float]] = []

    def close(group):
        if not group:
            return
        start = group[0][0] - step_s / 2
        end = group[-1][0] + step_s / 2
        covered = len(group) * step_s  # measured time only; merged gaps do not count toward the minimum
        if covered < GRIT_MIN_REGION_S - 1e-9 or end - start < GRIT_MIN_REGION_S - 1e-9:
            return
        rel = float(np.median([g[2] for g in group])) - ref_level
        kind = "grit" if rel >= GRIT_LEVEL_REL_DB else "breathy" if rel <= BREATHY_LEVEL_REL_DB else "rough"
        sh = [d for t, d in sub_frames if start <= t <= end]
        regions.append({
            "start_s": round(start, 2),
            "end_s": round(end, 2),
            "hnr_db": round(float(np.median([g[1] for g in group])), 1),
            "level_rel_db": round(rel, 1),
            "kind": kind,
            "subharmonic_db": round(float(np.median(sh)), 1) if sh else None,
            "covered_s": round(covered, 2),
            "confidence": "high" if covered >= GRIT_REGION_HIGH_S else "medium",
        })

    for row in rows:
        if row[1] < HNR_LOW_DB:
            if cur and row[0] - cur[-1][0] > GRIT_MERGE_GAP_S + 1e-9:
                close(cur)
                cur = []
            cur.append(row)
        else:
            close(cur)
            cur = []
    close(cur)

    note_times = [(float(times[n_["i0"]]), float(times[n_["i1"] - 1])) for n_ in note_segs]
    sh_vals = [d for t, d in sub_frames if any(a <= t <= b for a, b in note_times)]
    sub_conf = _count_conf(len(sh_vals), SPECTRAL_HIGH_FRAMES, SPECTRAL_MEDIUM_FRAMES, SPECTRAL_MIN_FRAMES)
    sub_db = round(float(np.median(sh_vals)), 1) if sub_conf else None
    return {
        "hnr_median_db": round(float(np.median(hnr)), 1),
        "hnr_p10_db": round(float(np.percentile(hnr, 10)), 1),
        "n_frames": len(rows),
        "confidence": conf_label,
        "subharmonic_db": sub_db,
        "subharmonics_present": (sub_db >= SUBHARMONIC_PRESENT_DB) if sub_db is not None else None,
        "subharmonic_confidence": sub_conf,
        "regions": regions,
    }


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------

def _signed(v: float) -> str:
    return f"{'−' if v < 0 else '+'}{abs(v):g}"


def format_voice_section(result: dict) -> str:
    lines: list[str] = []
    reg = result.get("register")
    if reg:
        s = f"VOICE  : range {reg['low_note']}–{reg['high_note']}"
        if reg.get("top_third_share") is not None:
            s += f" | {round(100 * reg['top_third_share'])}% of sung time in top third"
            ln = reg.get("longest_top_note")
            if ln:
                s += f", longest top note {ln['note']} held {ln['duration_s']:.1f}s at {_fmt_time(ln['start_s'])}"
        else:
            s += " | range too narrow for thirds"
        lines.append(s + f" ({reg['confidence']})")
    else:
        lines.append("VOICE  : not measured (too little confident voicing)")

    lwp = result.get("level_with_pitch")
    if lwp:
        r = lwp["r"]
        word = ("rises with pitch" if r >= LWP_TRACKS_R else "drops as pitch climbs" if r <= -LWP_TRACKS_R
                else "does not track pitch")
        lines.append(f"LEVEL  : level {word} (r={r:+.2f}, {_signed(lwp['slope_db_per_octave'])} dB/octave, "
                     f"n={lwp['n']}, {lwp['confidence']})")

    ring = result.get("ring")
    tilt = result.get("tilt")
    parts = []
    if ring:
        parts.append(f"2–4 kHz ring {_signed(ring['overall_db'])} dB ({ring['label']})")
    if tilt:
        parts.append(f"alpha ratio {_signed(tilt['alpha_ratio_db'])} dB ({tilt['alpha_label']})")
        if tilt.get("h1_h2_db") is not None:
            parts.append(f"H1–H2 {_signed(tilt['h1_h2_db'])} dB ({tilt['h1_h2_label']})")
    if parts:
        c = ring["confidence"] if ring else tilt["confidence"]
        lines.append("RING   : " + " | ".join(parts) + f" ({c})")
    if ring and ring.get("top_third_db") is not None:
        top = f"TOP3RD : on top-third notes ring {_signed(ring['top_third_db'])} dB"
        if tilt and tilt.get("alpha_top_third_db") is not None:
            top += f" | alpha ratio {_signed(tilt['alpha_top_third_db'])} dB"
        lines.append(top + f" (n={ring['top_third_n_frames']}, {ring['top_third_confidence']})")

    falls = result.get("falls") or []
    if falls:
        ex = max(falls, key=lambda f: f["drop_cents"])
        lines.append(f"FALLS  : {len(falls)} cry-fall{'s' if len(falls) != 1 else ''} (e.g. {ex['start_note']} ↓ "
                     f"{ex['drop_cents']:.0f}¢ over {ex['duration_s']:.1f}s at {_fmt_time(ex['start_s'])})")
    else:
        lines.append("FALLS  : none detected")

    breaks = result.get("breaks") or []
    if breaks:
        ex = max(breaks, key=lambda b: abs(b["jump_cents"]))
        lines.append(f"BREAKS : {len(breaks)} register flip{'s' if len(breaks) != 1 else ''} (e.g. "
                     f"{ex['from_note']}→{ex['to_note']} at {_fmt_time(ex['time_s'])})")
    else:
        lines.append("BREAKS : none detected")
    vc = result.get("voice_changes") or {}
    if vc.get("count"):
        lines.append(f"SLIPS  : {vc['count']} octave slips / voice changes set aside, not counted as flips or falls "
                     "(stem likely holds more than one voice, or the tracker slipped an octave)")

    grit = result.get("grit")
    if grit:
        wording = {"grit": "rough while loud", "breathy": "breathy (noisy while quiet)", "rough": "rough at mid level"}
        regions = sorted(grit["regions"], key=lambda r: r["covered_s"], reverse=True)[:CARD_MAX_REGIONS]
        regions.sort(key=lambda r: r["start_s"])
        if regions:
            body = "; ".join(f"{wording[r['kind']]} at {_fmt_time_precise(r['start_s'])}–{_fmt_time_precise(r['end_s'])} "
                             f"(HNR {r['hnr_db']:g} dB)" for r in regions)
        else:
            body = "no low-HNR stretches on sustained notes"
        body += f" | median HNR {grit['hnr_median_db']:g} dB"
        if grit.get("subharmonics_present"):
            body += f" | subharmonics present ({_signed(grit['subharmonic_db'])} dB re H1)"
        lines.append(f"GRIT   : {body}")
    return "\n".join(lines)
