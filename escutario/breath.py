# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The breath organ — find inhales in a separated vocal stem.

Pure numpy. Input is a mono vocal stem plus the PitchTrack for it.

How it listens:
  1. Phrases are runs of voiced frames from the pitch track, merged across
     short unvoiced gaps (consonants live there, not breaths).
  2. Breath candidates live in the last BREATH_SEARCH_S of each gap between
     phrases (and before the first phrase). Inside that window, frames that sit
     clearly above the stem's noise floor and are not tonal (a phrase's
     reverb tail is tonal) form candidate runs.
  3. Each run is measured: duration, level relative to the neighbouring
     phrases, level relative to the noise floor, noise-likeness (spectral
     flatness per 500 Hz chunk), share of energy in 1-6 kHz, and envelope
     smoothness (an inhale swells and fades; a click spikes).
  4. Hard gates drop what cannot be a breath. The survivors get a
     confidence tier from how many features clear their threshold with margin.
  5. A leakage guard looks for short, sharp, broadband transients recurring
     at a steady period (hi-hat bleed) and lowers confidence around them.

Omit rather than guess: a run that fails a gate is not reported. Line-sized
gaps with no breath energy at all are reported as silent gaps (likely
production edits); long instrumental breaks are not.
Every threshold below is a starting value to calibrate against real songs.
"""

from __future__ import annotations

import numpy as np

from .types import PitchTrack

# ---------------------------------------------------------------------------
# Constants — all starting values, to calibrate on real separated stems
# ---------------------------------------------------------------------------

# Phrases
VOICED_MIN_CONFIDENCE = 0.1      # starting value to calibrate: pitch frames below this confidence count as unvoiced
PHRASE_MERGE_GAP_S = 0.15        # starting value to calibrate: unvoiced gaps shorter than this are consonants, merged into the phrase
MIN_PHRASE_S = 0.15              # starting value to calibrate: voiced runs shorter than this are tracker blips, not phrases

# Short-frame level envelope
FLOOR_MIN_DB = -80.0             # starting value to calibrate: floor clamp so gated digital silence doesn't make every whisper "loud"
DB_EPS = 1e-10                   # starting value to calibrate: power epsilon before log (= -100 dB)
FLOOR_TOO_CLOSE_DB = 20.0        # starting value to calibrate: floor within this many dB of phrase level → note that quiet breaths may be missed
RMS_FRAME_S = 0.020              # starting value to calibrate: RMS window for the level envelope
FLOOR_PERCENTILE = 5.0           # starting value to calibrate: percentile of frame RMS (dB) taken as the stem's noise floor
RMS_HOP_S = 0.010                # starting value to calibrate: hop of the level envelope

# Candidate search inside a gap
GAP_EDGE_GUARD_S = 0.02          # starting value to calibrate: skip this much at each phrase edge (voicing-onset slop)
BREATH_SEARCH_S = 2.0            # starting value to calibrate: an inhale sits within this long before the phrase it feeds;
                                 # longer gaps (instrumental breaks) are only searched in their last BREATH_SEARCH_S
ACTIVE_ABOVE_FLOOR_DB = 6.0      # starting value to calibrate: a frame is "active" this far above the floor
RUN_HOLE_MERGE_S = 0.04          # starting value to calibrate: active runs split by holes shorter than this are one event

# Spectral analysis
TOTAL_BAND_HZ = (80.0, 11000.0)     # starting value to calibrate: reference band for the band-energy share (sr-independent)
SPEC_FRAME_S = 0.046             # starting value to calibrate: STFT window; resolves harmonics >= ~90 Hz apart
NOISE_CHUNK_HZ = 500.0           # starting value to calibrate: flatness is taken per chunk this wide, then energy-weighted — so a
NOISE_RANGE_HZ = (500.0, 8000.0)    # starting value to calibrate: where noise-likeness is judged
BREATH_BAND_HZ = (1000.0, 6000.0)   # starting value to calibrate: where inhale noise concentrates
                                 # band-limited breath still reads noise-like and only harmonic peaks read tonal
FRAME_TONAL_FLATNESS_MAX = 0.25  # starting value to calibrate: a frame whose noise-likeness is below this is tonal (reverb tail, bleed note)

# Hard gates — fail any and the run is not a breath
BAND_SHARE_GATE = 0.35           # starting value to calibrate: minimum share of energy in 1-6 kHz
LEVEL_REL_PHRASE_MIN_DB = -50.0  # starting value to calibrate: fainter than this relative to singing is residue, not an audible breath
FLATNESS_GATE = 0.30             # starting value to calibrate: run-averaged noise-likeness floor
LEVEL_REL_PHRASE_MAX_DB = -6.0   # starting value to calibrate: a breath is at least this much quieter than the singing
ABOVE_FLOOR_GATE_DB = 6.0        # starting value to calibrate: breath RMS at least this far above the noise floor
BREATH_MAX_S = 1.2               # starting value to calibrate: longer than this is sustained noise, not one inhale
BREATH_MIN_S = 0.12              # starting value to calibrate: shorter than this is a click or consonant spill

# Margins — each cleared adds one feature point toward the confidence tier
ENVELOPE_PEAK_TO_MEAN_MAX = 4.5  # starting value to calibrate: smooth swell (Hann ≈ 2.7) vs spiky clicks (≫ 5)
LEVEL_REL_PHRASE_MARGIN_DB = (-40.0, -10.0)  # starting value to calibrate: comfortably quieter than singing, comfortably audible
FLATNESS_MARGIN = 0.50           # starting value to calibrate: clearly noise-like
MEDIUM_MIN_POINTS = 3            # starting value to calibrate: this many → medium; fewer → low
HIGH_MIN_POINTS = 5              # starting value to calibrate: of 6 margin features, this many (incl. smooth envelope) → high
ENVELOPE_MEDIAN_S = 0.035        # starting value to calibrate: running median over the fine envelope; flattens click residue, keeps a swell
ABOVE_FLOOR_MARGIN_DB = 12.0     # starting value to calibrate: comfortably above the floor
ENVELOPE_FRAME_S = 0.005         # starting value to calibrate: fine energy envelope for shape and transients
BAND_SHARE_MARGIN = 0.55         # starting value to calibrate: clearly concentrated in 1-6 kHz
BREATH_CORE_S = (0.15, 0.90)     # starting value to calibrate: typical inhale duration

# Silent gaps
SILENT_GAP_MAX_S = 4.0           # starting value to calibrate: longer gaps are instrumental breaks, not a between-lines breath cut
SILENT_GAP_MIN_S = 0.30          # starting value to calibrate: shorter pauses can be staccato rests, not edits
SILENT_GAP_MAX_NOISE_S = 0.05    # starting value to calibrate: a gap with less non-tonal active sound than this has no breath energy

# Leakage guard (hi-hat-like periodic transients)
TRANSIENT_RISE_MAX_S = 0.015     # starting value to calibrate: ...within this long (an inhale takes ≥ 50 ms to swell)
TRANSIENT_DECAY_DB = 8.0         # starting value to calibrate: and falls at least this much...
TRANSIENT_OVERLAP_PAD_S = 0.02   # starting value to calibrate: a transient this close to a run counts as inside it
TRANSIENT_MAX_S = 0.080          # starting value to calibrate: ...within this long after the peak
HIHAT_JITTER_FRAC = 0.12         # starting value to calibrate: IOI within this fraction of the period of an integer multiple is "on grid"
HIHAT_MAX_SPAN_S = 8.0           # starting value to calibrate: intervals up to this (a sung phrase between two hats) are checked against the grid
TRANSIENT_MASK_MAX_S = 0.15      # starting value to calibrate: longest a single transient's decay is masked
HIHAT_MAX_PERIOD_S = 1.0         # starting value to calibrate: intervals up to this estimate the base period
TRANSIENT_MASK_DEPTH_DB = 20.0   # starting value to calibrate: ...or until it has decayed this far below its own peak, whichever is first
TRANSIENT_MIN_ZCR_HZ = 1500.0    # starting value to calibrate: zero crossings per second at the peak (broadband, not a tonal attack)
HIHAT_PERIODIC_FRACTION = 0.75   # starting value to calibrate: share of on-grid IOIs needed to call the transients periodic
HIHAT_BASE_CLUSTER = 1.5         # starting value to calibrate: base period = median of short intervals within this × their 25th percentile
HIHAT_MIN_EVENTS = 6             # starting value to calibrate: fewer transients than this can't establish a period
TRANSIENT_MASK_RELEASE_DB = 3.0  # starting value to calibrate: mask a transient until it decays to within this of what lay under it
HIHAT_MIN_SHORT_IOIS = 3         # starting value to calibrate: short intervals needed to estimate the base period
TRANSIENT_MASK_PRE_S = 0.010     # starting value to calibrate: masked lead before a transient peak (attack inside the frame)
TRANSIENT_MIN_ABOVE_FLOOR_DB = 15.0  # starting value to calibrate: a transient peak must stand this far above the floor
HIHAT_MIN_PERIOD_S = 0.08        # starting value to calibrate: faster than 32nds at 190 bpm is not a hat pattern
TRANSIENT_PEAK_HALFWIDTH_S = 0.020   # starting value to calibrate: transients closer than this (a flam) count once, the first
TRANSIENT_RISE_DB = 10.0         # starting value to calibrate: energy climbs at least this much (a click on a breath peak rises ~13 dB)...

NOTE_BREATHS_EDITED = (
    "producers often edit breaths out of a vocal, and separation can scrub quiet ones; "
    "an absent breath is not evidence the singer didn't breathe"
)


# ---------------------------------------------------------------------------
# Envelopes and spectra
# ---------------------------------------------------------------------------

def _frame_power(x: np.ndarray, sr: int, frame_s: float, hop_s: float):
    """Mean power per frame via cumulative sums. Returns (power, centre_times, hop, win)."""
    win = max(1, int(round(frame_s * sr)))
    hop = max(1, int(round(hop_s * sr)))
    n = len(x)
    if n < win:
        return np.zeros(0), np.zeros(0), hop, win
    cs = np.concatenate(([0.0], np.cumsum(np.square(x, dtype=np.float64))))
    starts = np.arange(0, n - win + 1, hop)
    power = (cs[starts + win] - cs[starts]) / win
    times = (starts + win / 2.0) / sr
    return np.maximum(power, 0.0), times, hop, win


def _to_db(p: np.ndarray | float):
    return 10.0 * np.log10(np.asarray(p, dtype=np.float64) + DB_EPS)


def _spectra(x: np.ndarray, sr: int, centres_s: np.ndarray):
    """Power spectra of Hann frames centred at the given times. Returns (P [frames x bins], freqs)."""
    win = max(16, int(round(SPEC_FRAME_S * sr)))
    n_fft = 1 << int(np.ceil(np.log2(win)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    if len(centres_s) == 0:
        return np.zeros((0, len(freqs))), freqs
    starts = np.round(centres_s * sr).astype(int) - win // 2
    lo = max(0, int(starts.min()))
    hi = min(len(x), int(starts.max()) + win)
    region = np.zeros(hi - lo + 2 * win)  # only the stretch these frames touch, zero-padded at the stem edges
    region[win:win + hi - lo] = x[lo:hi]
    idx = np.clip(starts - lo + win, 0, len(region) - win)[:, None] + np.arange(win)[None, :]
    frames = region[idx] * np.hanning(win)[None, :]
    P = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    return P, freqs


def _running_median(v: np.ndarray, width: int) -> np.ndarray:
    width = min(width | 1, v.size if v.size % 2 else v.size - 1)  # odd, and no wider than the data
    if width <= 1:
        return v
    half = width // 2
    padded = np.pad(v, (half, half), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, width), axis=1)


def _band_mask(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (freqs >= band[0]) & (freqs < band[1])


def _flatness(P: np.ndarray, axis: int = -1) -> np.ndarray:
    """Wiener entropy: geometric mean / arithmetic mean of power."""
    P = P + DB_EPS
    return np.exp(np.mean(np.log(P), axis=axis)) / np.mean(P, axis=axis)


def _noisiness(P: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Energy-weighted mean of per-chunk spectral flatness over NOISE_RANGE_HZ.

    Works on a 2-D [frames x bins] array (one value per frame) or a 1-D
    spectrum. Chunking makes it indifferent to the overall spectral shape.
    """
    P2 = np.atleast_2d(P)
    hi = min(NOISE_RANGE_HZ[1], freqs[-1])
    edges = np.arange(NOISE_RANGE_HZ[0], hi + 1e-9, NOISE_CHUNK_HZ)
    flats, weights = [], []
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo_e) & (freqs < hi_e)
        if np.count_nonzero(m) < 4:
            continue
        flats.append(_flatness(P2[:, m], axis=1))
        weights.append(P2[:, m].sum(axis=1))
    if not flats:
        out = _flatness(P2, axis=1)
    else:
        F, W = np.vstack(flats), np.vstack(weights) + DB_EPS
        out = (F * W).sum(axis=0) / W.sum(axis=0)
    return out if P.ndim == 2 else out[0]


# ---------------------------------------------------------------------------
# Phrases
# ---------------------------------------------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive-exclusive index runs where mask is True."""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _phrases_from_pitch(pitch: PitchTrack, duration_s: float) -> list[list[float]]:
    voiced = (pitch.f0_hz > 0) & (pitch.confidence >= VOICED_MIN_CONFIDENCE)
    half = pitch.hop_s / 2.0
    spans = [[float(pitch.times[i]) - half, float(pitch.times[j - 1]) + half] for i, j in _runs(voiced)]
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] < PHRASE_MERGE_GAP_S:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        s, e = max(0.0, s), min(duration_s, e)
        if e - s >= MIN_PHRASE_S:
            out.append([s, e])
    return out


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

def _find_transients(x: np.ndarray, sr: int, floor_db: float, phrases: list[list[float]]):
    """Short, sharp, broadband transients outside phrases.

    Returns (peak_times, mask_intervals): each mask interval covers the attack
    and the decay until the transient sinks back to what lay under it, so the
    rest of the gap can still be measured.
    """
    p, t, hop, win = _frame_power(x, sr, ENVELOPE_FRAME_S, ENVELOPE_FRAME_S)
    if p.size < 3:
        return np.zeros(0), []
    db = _to_db(p)
    in_phrase = np.zeros(len(t), dtype=bool)
    for s, e in phrases:
        in_phrase |= (t >= s) & (t < e)
    half = max(1, int(round(TRANSIENT_PEAK_HALFWIDTH_S / ENVELOPE_FRAME_S)))
    rise_n = max(1, int(round(TRANSIENT_RISE_MAX_S / ENVELOPE_FRAME_S)))
    decay_n = max(1, int(round(TRANSIENT_MAX_S / ENVELOPE_FRAME_S)))
    mask_n = max(1, int(round(TRANSIENT_MASK_MAX_S / ENVELOPE_FRAME_S)))
    cand = np.flatnonzero((db >= floor_db + TRANSIENT_MIN_ABOVE_FLOOR_DB) & ~in_phrase)
    times, masks = [], []
    last = -np.inf
    for i in cand:
        if i - last <= half or i - rise_n < 0 or i + 1 >= len(db):
            continue
        if db[i] < db[i - 1] or db[i] < db[i + 1]:
            continue  # not a local peak of the fine envelope
        under = float(db[i - rise_n:i].min())
        if under > db[i] - TRANSIENT_RISE_DB:
            continue
        after = db[i + 1:i + 1 + decay_n]
        if after.size == 0 or after.min() > db[i] - TRANSIENT_DECAY_DB:
            continue
        seg = x[i * hop:i * hop + win]
        zcr_hz = np.count_nonzero(np.diff(np.signbit(seg))) * sr / max(1, len(seg))
        if zcr_hz < TRANSIENT_MIN_ZCR_HZ:
            continue
        settle = max(max(under, floor_db + ACTIVE_ABOVE_FLOOR_DB) + TRANSIENT_MASK_RELEASE_DB,
                     float(db[i]) - TRANSIENT_MASK_DEPTH_DB)
        tail = db[i + 1:i + 1 + mask_n]
        below = np.flatnonzero(tail <= settle)
        j = i + 1 + (int(below[0]) if below.size else len(tail))
        times.append(float(t[i]))
        masks.append((float(t[i]) - ENVELOPE_FRAME_S / 2 - TRANSIENT_MASK_PRE_S,
                      float(t[min(j, len(t) - 1)]) + ENVELOPE_FRAME_S / 2))
        last = i
    return np.asarray(times), masks


def _periodic_leakage(transients: np.ndarray) -> float | None:
    """Return the period (s) if transients recur on a steady grid, else None.

    The base period comes from the short intervals (hats between breaths);
    every interval up to HIHAT_MAX_SPAN_S — including the long ones that
    straddle a sung phrase — must then land on a multiple of it.
    """
    if len(transients) < HIHAT_MIN_EVENTS:
        return None
    ioi = np.diff(np.sort(transients))
    short = ioi[(ioi >= HIHAT_MIN_PERIOD_S) & (ioi <= HIHAT_MAX_PERIOD_S)]
    if len(short) < HIHAT_MIN_SHORT_IOIS:
        return None
    p25 = float(np.percentile(short, 25))
    period = float(np.median(short[short <= HIHAT_BASE_CLUSTER * p25]))
    checked = ioi[ioi <= HIHAT_MAX_SPAN_S]
    if len(checked) < HIHAT_MIN_EVENTS - 1:
        return None
    k = np.round(checked / period)
    on_grid = (k >= 1) & (np.abs(checked - k * period) <= HIHAT_JITTER_FRAC * period)
    return period if float(np.mean(on_grid)) >= HIHAT_PERIODIC_FRACTION else None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _empty(notes: list[str]) -> dict:
    return {"breaths": [], "count": 0, "phrases": [], "longest_phrase_s": 0.0,
            "silent_gaps": [], "notes": notes}


def detect_breaths(vocal: np.ndarray, sr: int, pitch: PitchTrack, *, explain: bool = False) -> dict:
    """Find breath events between sung phrases of a separated vocal stem.

    Returns {"breaths", "count", "phrases", "longest_phrase_s", "silent_gaps", "notes"}
    plus "noise_floor_db" and "leakage_period_s" (None unless hi-hat-like leakage was found).
    explain=True adds "candidates": every measured run with its features and why it was kept or
    dropped — the view to calibrate thresholds against real songs.
    """
    x = np.asarray(vocal, dtype=np.float64).reshape(-1)
    duration_s = len(x) / float(sr) if sr > 0 else 0.0
    notes = [NOTE_BREATHS_EDITED]

    phrases = _phrases_from_pitch(pitch, duration_s)
    power, ftimes, _, _ = _frame_power(x, sr, RMS_FRAME_S, RMS_HOP_S)
    if not phrases or power.size == 0:
        notes.insert(0, "no sung phrases in the pitch track; no gaps to listen for breath in")
        out = _empty(notes)
        out.update(noise_floor_db=None, leakage_period_s=None)
        return out

    fdb = _to_db(power)
    floor_db = max(float(np.percentile(fdb, FLOOR_PERCENTILE)), FLOOR_MIN_DB)

    phrase_db = []
    for s, e in phrases:
        m = (ftimes >= s) & (ftimes < e)
        seg = x[int(s * sr):int(e * sr)]
        if m.any():
            phrase_db.append(float(np.median(fdb[m])))
        else:  # phrase shorter than one level frame
            phrase_db.append(float(_to_db(np.mean(seg ** 2))) if seg.size else floor_db)
    if float(np.median(phrase_db)) - floor_db < FLOOR_TOO_CLOSE_DB:
        notes.append(f"noise floor ({floor_db:.0f} dB) sits within {FLOOR_TOO_CLOSE_DB:.0f} dB of the singing; quiet breaths may be missed")

    transients, trans_masks = _find_transients(x, sr, floor_db, phrases)
    leak_period = _periodic_leakage(transients)
    if leak_period is not None:
        notes.append(f"periodic sharp transients in the gaps (hi-hat-like leakage, every ~{leak_period:.2f}s); "
                     "breaths near them capped at medium, breaths with clicks inside marked low, click-shaped runs dropped")

    env_p, env_t, _, _ = _frame_power(x, sr, ENVELOPE_FRAME_S, ENVELOPE_FRAME_S)

    # Search windows: (gap_start, gap_end, index of the phrase that follows, index of the phrase before or None).
    # Only the last BREATH_SEARCH_S of each gap is listened to.
    windows = [(0.0, phrases[0][0], 0, None)]
    for k in range(1, len(phrases)):
        windows.append((phrases[k - 1][1], phrases[k][0], k, k - 1))

    hole_n = int(round(RUN_HOLE_MERGE_S / RMS_HOP_S))
    breaths: list[dict] = []
    silent_gaps: list[dict] = []
    preceded = [False] * len(phrases)
    unclassified = 0
    candidates: list[dict] = []

    for w0, w1, nxt, prv in windows:
        s0, s1 = max(w0, w1 - BREATH_SEARCH_S) + GAP_EDGE_GUARD_S, w1 - GAP_EDGE_GUARD_S
        fm = np.flatnonzero((ftimes >= s0) & (ftimes < s1))
        if fm.size == 0:
            continue
        P, freqs = _spectra(x, sr, ftimes[fm])
        in_band = _band_mask(freqs, BREATH_BAND_HZ)
        total_band = _band_mask(freqs, (TOTAL_BAND_HZ[0], min(TOTAL_BAND_HZ[1], sr / 2.0)))
        tonal = _noisiness(P, freqs) < FRAME_TONAL_FLATNESS_MAX
        loud = fdb[fm] >= floor_db + ACTIVE_ABOVE_FLOOR_DB
        masked = np.zeros(fm.size, dtype=bool)
        fr0 = ftimes[fm] - RMS_FRAME_S / 2
        fr1 = ftimes[fm] + RMS_FRAME_S / 2
        for m0, m1 in trans_masks:
            if m1 >= s0 and m0 <= s1:
                masked |= (fr1 > m0) & (fr0 < m1)
        clean = loud & ~tonal & ~masked
        # A run spans clean frames and the masked transients between them; holes are closed.
        joined: list[list[int]] = []
        for a, b in _runs(clean | (masked & loud)):
            if joined and a - joined[-1][1] <= hole_n:
                joined[-1][1] = b
            else:
                joined.append([a, b])

        noise_s = float(np.count_nonzero(clean)) * RMS_HOP_S
        gap_hits = transients[(transients >= s0) & (transients <= s1)]
        found_here = False
        for a, b in joined:
            idx = a + np.flatnonzero(clean[a:b])
            if idx.size == 0:
                continue  # nothing but masked transients
            span0 = float(ftimes[fm[a]]) - RMS_HOP_S / 2.0  # untrimmed extent, masked clicks included
            span1 = float(ftimes[fm[b - 1]]) + RMS_HOP_S / 2.0
            a, b = int(idx[0]), int(idx[-1]) + 1
            start = float(ftimes[fm[a]]) - RMS_HOP_S / 2.0
            end = float(ftimes[fm[b - 1]]) + RMS_HOP_S / 2.0
            dur = end - start
            cand = {"start_s": round(start, 3), "end_s": round(end, 3), "duration_s": round(dur, 3)}
            candidates.append(cand)
            if dur < BREATH_MIN_S or dur > BREATH_MAX_S or idx.size * RMS_HOP_S < BREATH_MIN_S:
                cand["dropped"] = "duration"
                continue
            run_db = float(_to_db(np.mean(power[fm[idx]])))
            neighbours = [phrase_db[i] for i in (prv, nxt) if i is not None]
            rel_db = run_db - float(np.mean(neighbours))
            above_floor = run_db - floor_db
            meanP = P[idx].mean(axis=0)
            flat = float(_noisiness(meanP, freqs))
            share = float(meanP[in_band].sum() / (meanP[total_band].sum() + DB_EPS))
            cand.update(level_db_rel_phrase=round(rel_db, 1), above_floor_db=round(above_floor, 1),
                        flatness=round(flat, 3), band_share=round(share, 3))

            if not (LEVEL_REL_PHRASE_MIN_DB <= rel_db <= LEVEL_REL_PHRASE_MAX_DB):
                cand["dropped"] = "level vs phrase"
                continue
            if above_floor < ABOVE_FLOOR_GATE_DB:
                cand["dropped"] = "level vs floor"
                continue
            if flat < FLATNESS_GATE:
                cand["dropped"] = "flatness"
                continue
            if share < BAND_SHARE_GATE:
                cand["dropped"] = "band share"
                continue

            em = (env_t >= start) & (env_t < end)
            for m0, m1 in trans_masks:
                if m1 >= start and m0 <= end:
                    em &= ~((env_t >= m0) & (env_t < m1))
            ev = _running_median(env_p[em], max(1, int(round(ENVELOPE_MEDIAN_S / ENVELOPE_FRAME_S))))
            peak_to_mean = float(ev.max() / (ev.mean() + DB_EPS)) if ev.size else np.inf
            smooth = peak_to_mean <= ENVELOPE_PEAK_TO_MEAN_MAX
            hits = transients[(transients >= min(start, span0) - TRANSIENT_OVERLAP_PAD_S)
                              & (transients <= max(end, span1) + TRANSIENT_OVERLAP_PAD_S)]
            cand.update(peak_to_mean=round(peak_to_mean, 2), transients_inside=int(hits.size))
            if hits.size and not smooth:
                cand["dropped"] = "click-shaped"
                continue  # clicks with a little wash between them, not an inhale

            points = sum([
                BREATH_CORE_S[0] <= dur <= BREATH_CORE_S[1],
                LEVEL_REL_PHRASE_MARGIN_DB[0] <= rel_db <= LEVEL_REL_PHRASE_MARGIN_DB[1],
                above_floor >= ABOVE_FLOOR_MARGIN_DB,
                flat >= FLATNESS_MARGIN,
                share >= BAND_SHARE_MARGIN,
                smooth,
            ])
            if points >= HIGH_MIN_POINTS and smooth:
                tier = "high"
            elif points >= MEDIUM_MIN_POINTS:
                tier = "medium"
            else:
                tier = "low"
            if hits.size:
                if points < HIGH_MIN_POINTS:
                    cand.update(points=int(points), dropped="weak run around clicks")
                    continue  # a click's decay tail over a raised floor — a guess, so omitted
                # a real-looking inhale, but a click smears its edges and level
                tier = "low" if leak_period is not None else "medium"
            elif leak_period is not None and tier == "high":
                tier = "medium"
            cand.update(points=int(points), confidence=tier)

            breaths.append({
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "duration_s": round(dur, 3),
                "level_db_rel_phrase": round(rel_db, 1),
                "confidence": tier,
            })
            found_here = True

        if found_here:
            preceded[nxt] = True
        elif prv is not None:
            if SILENT_GAP_MIN_S <= (w1 - w0) <= SILENT_GAP_MAX_S and noise_s < SILENT_GAP_MAX_NOISE_S and gap_hits.size == 0:
                silent_gaps.append({"start_s": round(w0, 3), "end_s": round(w1, 3)})
            elif noise_s >= SILENT_GAP_MAX_NOISE_S or gap_hits.size:
                unclassified += 1

    if unclassified:
        notes.append(f"{unclassified} gap(s) held sound that did not read as a breath (too loud, too short, "
                     "click-shaped or off-band); not counted")

    phrase_out = [{"start_s": round(s, 3), "end_s": round(e, 3), "duration_s": round(e - s, 3),
                   "preceded_by_breath": preceded[i]} for i, (s, e) in enumerate(phrases)]
    result = {
        "breaths": breaths,
        "count": len(breaths),
        "phrases": phrase_out,
        "longest_phrase_s": round(max(e - s for s, e in phrases), 3),
        "silent_gaps": silent_gaps,
        "notes": notes,
        "noise_floor_db": round(floor_db, 1),
        "leakage_period_s": round(leak_period, 3) if leak_period is not None else None,
    }
    if explain:
        result["candidates"] = candidates
    return result


# ---------------------------------------------------------------------------
# Card section
# ---------------------------------------------------------------------------

FEEDS_PHRASE_TOL_S = 0.05        # starting value to calibrate: a breath may run this far past the phrase start it feeds (voicing lag)
LONG_BREATH_S = 0.60             # starting value to calibrate: an inhale this long is worth a line
LOUD_BREATH_REL_DB = -12.0       # starting value to calibrate: a breath this close to the singing level is worth a line
MAX_GAP_TIMES = 6                # starting value to calibrate: silent-gap times listed before "…"
MAX_BREATH_LINES = 5             # starting value to calibrate: notable-breath lines on the card


def _mmss(t: float, tenths: bool = False) -> str:
    m, s = divmod(max(0.0, t), 60.0)
    return f"{int(m)}:{s:04.1f}" if tenths else f"{int(m)}:{int(s):02d}"


def format_breath_section(result: dict) -> str:
    if not result:
        return ""
    phrases = result.get("phrases") or []
    breaths = result.get("breaths") or []
    notes = result.get("notes") or []
    if not phrases:
        lines = ["BREATH : no sung phrases found"]
        lines += [f"  note: {n}" for n in notes if n != NOTE_BREATHS_EDITED]
        return "\n".join(lines)

    tiers = {t: sum(1 for b in breaths if b["confidence"] == t) for t in ("high", "medium", "low")}
    if breaths:
        conf = " / ".join(f"{t} {tiers[t]}" for t in ("high", "medium", "low") if tiers[t] or t != "low")
        head = f"{len(breaths)} heard (conf {conf})"
    else:
        head = "none heard"
    longest = max(phrases, key=lambda p: p["duration_s"])
    bits = [head, f"longest phrase {longest['duration_s']:.1f}s at {_mmss(longest['start_s'])}"]
    gaps = result.get("silent_gaps") or []
    if gaps:
        times = ", ".join(_mmss(g["start_s"]) for g in gaps[:MAX_GAP_TIMES]) + (", …" if len(gaps) > MAX_GAP_TIMES else "")
        bits.append(f"{len(gaps)} silent gap{'s' if len(gaps) != 1 else ''} (likely edits) at {times}")
    lines = ["BREATH : " + " | ".join(bits)]

    # Notable: the breath before the longest phrase, then long or loud inhales.
    notable: list[tuple[dict, str]] = []
    before_longest = [b for b in breaths if b["end_s"] <= longest["start_s"] + FEEDS_PHRASE_TOL_S
                      and longest["start_s"] - b["start_s"] <= BREATH_SEARCH_S + FEEDS_PHRASE_TOL_S]
    if before_longest and longest["preceded_by_breath"]:
        notable.append((before_longest[-1], f"before the longest phrase ({longest['duration_s']:.1f}s)"))
    for b in sorted(breaths, key=lambda b: (-b["duration_s"], b["start_s"])):
        tags = []
        if b["duration_s"] >= LONG_BREATH_S:
            tags.append("long")
        if b["level_db_rel_phrase"] >= LOUD_BREATH_REL_DB:
            tags.append("loud")
        if tags and all(b is not n for n, _ in notable):
            notable.append((b, ", ".join(tags)))
    for b, why in sorted(notable[:MAX_BREATH_LINES], key=lambda nb: nb[0]["start_s"]):
        lines.append(f"  {_mmss(b['start_s'], True):>7}  {b['duration_s']:.2f}s  "
                     f"{b['level_db_rel_phrase']:+.0f} dB  {b['confidence']:<6} {why}")
    for n in notes:
        lines.append(f"  note: {n}")
    return "\n".join(lines)
