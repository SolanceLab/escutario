# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Vibrato — the periodic waver of a held note: how fast (Hz) and how wide (cents).

Anne's first ask (16 Sep 2026): "hear the breath, the pitch, feel the vibrato". Escutário's
own detector, working on the PitchTrack from `pitch.py`. On her first two songs the held
notes are sung dead straight, so the honest answer there is "no vibrato", and a detector
that invents one would be worse than none.

Method: find steady voiced stretches, remove the note's slow drift, then look for one
dominant modulation between 4 and 8 Hz that is wide enough to hear and clearly stronger
than the rest of the modulation spectrum. Pure numpy. Numbers, not moods.
"""

from __future__ import annotations

import numpy as np

from escutario.types import PitchTrack

VOICED_MIN_CONFIDENCE = 0.5   # frames below this are unvoiced (pitch.py's voicing line) — starting value
MIN_CYCLES = 3.5              # at least this many cycles inside the note — starting value
EXTENT_MAX_CENTS = 100.0      # wider than a semitone each way is melody or a riff moving, not a voice wavering — set 16 Sep (±125/±166¢ 'vibrato' on real songs)
RATE_MIN_HZ = 4.0             # vibrato rate band — starting value (typical sung vibrato 4.5–7 Hz)
PROMINENCE_MIN = 3.0          # the rate peak must be this many times the median of the 2–15 Hz spectrum — starting value
HIGH_PROMINENCE = 6.0
EXTENT_MIN_CENTS = 15.0       # half peak-to-peak width below this is inaudible waver — starting value
MIN_NOTE_S = 0.8              # vibrato needs several cycles to judge: at 5 Hz, 0.8 s is 4 — set 16 Sep after 0.6 s flagged melodic movement
RATE_MAX_HZ = 8.0
NOTE_BAND_CENTS = 150.0       # a held note's smoothed pitch stays within this band — starting value
FIT_MIN = 0.5                 # a sinusoid at the found rate must explain at least this share of the waver — set 16 Sep
HIGH_EXTENT_CENTS = 30.0      # extent at or above this with strong prominence is high confidence — starting value
SMOOTH_S = 0.25               # smoothing that removes vibrato to find the note's centre line — starting value


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, k - 1 - pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def detect_vibrato(pitch: PitchTrack) -> dict:
    """Vibrato on every held note long enough to judge. Returns notes judged and those with vibrato."""
    f0 = np.asarray(pitch.f0_hz, dtype=np.float64)
    conf = np.asarray(pitch.confidence, dtype=np.float64)
    times = np.asarray(pitch.times, dtype=np.float64)
    hop = float(pitch.hop_s)
    if f0.size == 0 or hop <= 0:
        return {"judged": 0, "with_vibrato": [], "straight": 0, "notes": ["no pitch track"]}
    fr = 1.0 / hop
    voiced = (f0 > 0) & (conf >= VOICED_MIN_CONFIDENCE)
    cents = np.zeros_like(f0)
    cents[voiced] = 1200.0 * np.log2(f0[voiced] / 440.0)
    k = max(1, int(round(SMOOTH_S * fr)))
    min_len = int(round(MIN_NOTE_S * fr))

    judged, found = 0, []
    for a, b in _runs(voiced):
        if b - a < min_len:
            continue
        seg = cents[a:b]
        centre = _smooth(seg, k)
        # split the voiced run into held notes where the centre line moves out of band
        start = 0
        bounds = []
        for i in range(1, len(seg) + 1):
            if i == len(seg) or abs(centre[i] - centre[start]) > NOTE_BAND_CENTS:
                bounds.append((start, i))
                start = i
        for s0, s1 in bounds:
            if s1 - s0 < min_len:
                continue
            judged += 1
            wave = seg[s0:s1] - centre[s0:s1]
            n = len(wave)
            spec = np.abs(np.fft.rfft(wave * np.hanning(n)))
            freqs = np.fft.rfftfreq(n, d=hop)
            band = (freqs >= RATE_MIN_HZ) & (freqs <= RATE_MAX_HZ)
            wide = (freqs >= 2.0) & (freqs <= 15.0)
            if not band.any() or wide.sum() < 4:
                continue
            idx = np.where(band)[0]
            peak = idx[np.argmax(spec[idx])]
            prominence = float(spec[peak] / (np.median(spec[wide]) + 1e-9))
            rate = float(freqs[peak])
            # second pass: a box exactly one vibrato cycle long cancels the wave, leaving the true centre line
            k2 = max(1, int(round(fr / max(rate, 1e-6))))
            wave = seg[s0:s1] - _smooth(seg[s0:s1], k2)
            extent = float((np.percentile(wave, 95) - np.percentile(wave, 5)) / 2.0)
            cycles = rate * n * hop
            tt = np.arange(len(wave)) * hop
            basis = np.column_stack([np.sin(2 * np.pi * rate * tt), np.cos(2 * np.pi * rate * tt), np.ones_like(tt)])
            coef, *_ = np.linalg.lstsq(basis, wave, rcond=None)
            resid = wave - basis @ coef
            fit = 1.0 - float(np.var(resid)) / (float(np.var(wave)) + 1e-9)
            if (prominence < PROMINENCE_MIN or extent < EXTENT_MIN_CENTS or extent > EXTENT_MAX_CENTS
                    or cycles < MIN_CYCLES or fit < FIT_MIN):
                continue
            t0 = float(times[a + s0])
            mean_midi = 69.0 + float(np.mean(centre[s0:s1])) / 100.0
            found.append({
                "start_s": round(t0, 2), "dur_s": round((s1 - s0) * hop, 2),
                "midi": int(round(mean_midi)), "rate_hz": round(rate, 1), "extent_cents": round(extent), "fit": round(fit, 2),
                "confidence": "high" if extent >= HIGH_EXTENT_CENTS and prominence >= HIGH_PROMINENCE else "medium",
            })
    return {"judged": judged, "with_vibrato": found, "straight": judged - len(found),
            "notes": ["vibrato is judged only on notes held at least %.1f s" % MIN_NOTE_S]}
