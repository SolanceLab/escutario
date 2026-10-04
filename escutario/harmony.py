# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Harmony — the song facts: key, tempo and beats, sections, loudest moment, duration.

Input is the full mix (mono float32). Built on librosa's feature extractors
(constant-Q chroma, onset strength, tempogram, MFCC); the decisions on top of
them are ours:

- Key: the pitch-class histogram of the harmonic part of the mix is first fitted
  to a seven-note scale (which notes the song uses), then the home note is chosen
  between that scale's major and minor readings by correlating against three
  published key profiles (Krumhansl-Kessler 1982, Temperley 2007 Kostka-Payne,
  Albrecht-Shanahan 2013). Profiles alone confuse a minor key with the minor key a
  fourth up (C# minor vs F# minor: they share six notes and the profiles reward a
  strong fifth as a tonic); the scale fit stops that when the notes rule one out.
  When the unrestricted profile winner disagrees, it is reported as an alternative
  and confidence drops.
- Tempo: the tempogram estimate with a log-normal prior around 120 BPM, a beat
  grid tracked at that tempo, and an explicit half/double-time reading: which other
  metrical level a listener could reasonably tap, and the evidence for it.
- Sections: Foote novelty (checkerboard kernel over self-similarity) on tone colour
  (MFCC), harmony (chroma) and level, at two time scales.
- Energy: K-weighted short-term level (3 s windows, the BS.1770 weighting curve,
  ungated) — the loudest stretch as heard, not the loudest single transient.

Numbers, not moods.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Analysis grid
# ---------------------------------------------------------------------------
ANALYSIS_SR = 22050          # everything is resampled to this before analysis — starting value (fixed grid)
CHROMA_HOP = 2048            # chroma hop for key (~93 ms) — starting value
MIN_DURATION_S = 4.0         # shorter audio gets no tempo/sections — starting value to calibrate
HOP = 512                    # onset / feature hop in samples at ANALYSIS_SR (~23 ms) — starting value

# ---------------------------------------------------------------------------
# Key
# ---------------------------------------------------------------------------
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)   # a definition, not a threshold
# Published key profiles, index 0 = tonic. (Definitions from the literature, not thresholds.)
KEY_WINDOW_S = 20.0              # key re-estimated on windows this long to report stability — starting value
PROFILES = {
    "krumhansl_kessler": (
        (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88),
        (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17),
    ),
    "temperley_kostka_payne": (
        (0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400),
        (0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330),
    ),
    "albrecht_shanahan": (
        (0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081),
        (0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.061, 0.052),
    ),
}
KEY_MIN_CHROMA_ENERGY = 1e-6     # below this the mix has no pitched content; key omitted — starting value
PAIR_MARGIN_HIGH = 0.10          # major/minor reading of the scale separated by this correlation for a clear home note — starting value
SCALE_MARGIN_HIGH = 0.04         # best scale must beat the next by this share for a clear scale — starting value to calibrate
SCALE_FIT_MIN = 0.65             # below this the song doesn't sit in one diatonic scale; confidence low — starting value to calibrate
PAIR_MARGIN_MIN = 0.03           # closer than this the home note is a coin toss — starting value to calibrate
KEY_WINDOW_AGREEMENT_HIGH = 0.70 # share of windows reaching the same key for a stable key — starting value to calibrate
SCALE_FIT_HIGH = 0.80            # share of chroma energy inside the best 7-note scale for a clear scale — starting value to calibrate

# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------
DOUBLE_MAX_BPM = 210.0           # a double-time reading faster than this isn't a tempo people tap (200 + slack for grid error) — starting value to calibrate
HALF_MIN_BPM = 50.0              # a half-time reading slower than this isn't offered — starting value to calibrate
BEAT_TIGHTNESS = 400.0           # how strictly beats keep to the estimated tempo — starting value to calibrate
ONSET_PEAK_FRAMES = 2            # onset strength read as the max within ± this many frames of a grid point — starting value
LOCAL_TEMPO_TOL = 0.04           # a window agrees with the global tempo within this fraction (any of ×½, ×1, ×2) — starting value
BEAT_CV_HIGH = 0.04              # inter-beat interval spread (std/mean) at or below this = steady grid — starting value to calibrate
LOCAL_TEMPO_WINDOW_S = 20.0      # tempo re-estimated on windows this long — starting value
FAST_BPM = 160.0                 # at or above this, half time is always a plausible felt tempo (Forbidden Fruit was once read as 199) — starting value
HALF_ALTERNATION_RATIO = 0.65    # weaker/stronger alternate-beat onset ratio at or below this → every other beat is accented — starting value
STEADY_SHARE_HIGH = 0.75         # share of agreeing windows needed for high confidence — starting value to calibrate
BEAT_OUTLIER_FRAC = 0.2         # beat intervals this far from the median are skipped when fitting the grid — starting value
TEMPO_PRIOR_BPM = 120.0          # centre of the log-normal tempo prior (librosa's default; where people tap) — starting value
BEAT_CV_MEDIUM = 0.10            # ...at or below this = mostly steady — starting value to calibrate
DOUBLE_OFFBEAT_RATIO = 0.70      # off-beat onsets at least this strong vs on-beat → double time is audible — starting value to calibrate

# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
SECTION_MIN_GAP_S = 8.0          # boundaries closer than this merge (keep the stronger) — starting value
SECTION_MIN_PROMINENCE = 0.15    # novelty peak prominence (0..1) to report a boundary — starting value to calibrate
SECTION_FRAME_S = 0.5            # features averaged into frames this long — starting value
N_MFCC = 13                      # tone-colour coefficients (first is dropped: it is level) — starting value
SECTION_EDGE_GUARD_S = 3.0       # no boundary this close to the start or end (kernel edge effect) — starting value
SECTION_KERNELS_S = (8.0, 16.0)  # checkerboard kernel half-widths (seconds) — starting values to calibrate
SECTION_MEDIUM_PROMINENCE = 0.25 # ...medium at or above — starting value to calibrate
LEVEL_SIM_DB = 6.0               # level frames this many dB apart are ~dissimilar (Gaussian width) — starting value
SECTION_HIGH_PROMINENCE = 0.40   # ...high confidence at or above — starting value to calibrate

# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------
ENERGY_WINDOW_S = 3.0            # short-term loudness window (BS.1770 short-term) — starting value
NEAR_LOUDEST_DB = 1.0            # seconds within this of the loudest are counted as a plateau — starting value
ENERGY_HOP_S = 0.1               # step of the loudness curve — starting value
LOUDEST_SEPARATE_S = 10.0        # "separate" loud moment = at least this far from the loudest — starting value
LOUDEST_MARGIN_MEDIUM_DB = 0.5   # ...by this → medium; closer → low (a mastered plateau) — starting value to calibrate
LOUDEST_MARGIN_HIGH_DB = 1.5     # loudest beats the next separate loud moment by this → high — starting value to calibrate
# BS.1770 K-weighting as two RBJ biquads (fit parameters from the standard's 48 kHz coefficients)
K_SHELF_FC = 1681.97445          # definition
K_SHELF_GAIN_DB = 3.99984385     # definition (BS.1770 pre-filter)
DB_FLOOR = -120.0                # level floor for digital silence — starting value
K_SHELF_Q = 0.7071752            # definition
K_HP_FC = 38.13547               # definition (RLB high-pass)
EPS = 1e-12
K_HP_Q = 0.5003270               # definition


# ===========================================================================
# helpers
# ===========================================================================

def _key_name(tonic: int, minor: bool) -> str:
    return f"{NOTE_NAMES[tonic % 12]} {'minor' if minor else 'major'}"


def _profile_scores(hist: np.ndarray) -> np.ndarray:
    """Mean correlation over the three profiles for 24 keys: 0-11 major, 12-23 minor."""
    out = np.zeros(24)
    for major, minor in PROFILES.values():
        maj, mnr = np.asarray(major), np.asarray(minor)
        for t in range(12):
            r = np.roll(hist, -t)
            if np.std(r) < EPS:
                continue
            out[t] += np.corrcoef(r, maj)[0, 1]
            out[12 + t] += np.corrcoef(r, mnr)[0, 1]
    return out / len(PROFILES)


def _scale_fit(hist: np.ndarray) -> np.ndarray:
    """Share of chroma energy inside each of the 12 major-scale note sets (index = major tonic)."""
    total = hist.sum() + EPS
    return np.array([hist[[(root + s) % 12 for s in MAJOR_SCALE]].sum() / total for root in range(12)])


def _choose_key(hist: np.ndarray) -> dict | None:
    if hist.sum() < KEY_MIN_CHROMA_ENERGY:
        return None
    hist = hist / hist.sum()
    scores = _profile_scores(hist)
    fits = _scale_fit(hist)
    order = np.argsort(-fits)
    best_root = int(order[0])
    scale_share, scale_margin = float(fits[order[0]]), float(fits[order[0]] - fits[order[1]])
    # the scale's two readings: major on its root, minor on its sixth degree
    maj_i, min_i = best_root, 12 + (best_root + 9) % 12
    pick, other = (maj_i, min_i) if scores[maj_i] >= scores[min_i] else (min_i, maj_i)
    pair_margin = float(scores[pick] - scores[other])
    unrestricted = int(np.argmax(scores))
    return {
        "index": pick, "other": other, "unrestricted": unrestricted, "scores": scores,
        "scale_share": scale_share, "scale_margin": scale_margin, "pair_margin": pair_margin,
    }


def _name_of(index: int) -> str:
    return _key_name(index % 12, index >= 12)


def estimate_key(y: np.ndarray, sr: int, *, harmonic: np.ndarray | None = None) -> dict:
    """Key with an honest confidence and the alternatives that nearly won."""
    import librosa

    yh = harmonic if harmonic is not None else librosa.effects.harmonic(y)
    if float(np.sqrt(np.mean(np.square(yh, dtype=np.float64)))) < 1e-5:
        return {"key": None, "confidence": "low", "notes": ["no pitched content"]}
    chroma = librosa.feature.chroma_cqt(y=yh, sr=sr, hop_length=CHROMA_HOP)
    choice = _choose_key(chroma.sum(axis=1))
    if choice is None:
        return {"key": None, "confidence": "low", "notes": ["no pitched content"]}
    notes: list[str] = []
    # stability: the same decision on windows
    frames = int(round(KEY_WINDOW_S * sr / CHROMA_HOP))
    votes: dict[str, int] = {}
    for a in range(0, max(1, chroma.shape[1] - frames // 2), max(1, frames)):
        c = _choose_key(chroma[:, a:a + frames].sum(axis=1))
        if c is not None:
            votes[_name_of(c["index"])] = votes.get(_name_of(c["index"]), 0) + 1
    n_win = sum(votes.values())
    window_agreement = votes.get(_name_of(choice["index"]), 0) / n_win if n_win else None
    agree = choice["unrestricted"] == choice["index"]
    points = sum([
        choice["scale_share"] >= SCALE_FIT_HIGH and choice["scale_margin"] >= SCALE_MARGIN_HIGH,
        choice["pair_margin"] >= PAIR_MARGIN_HIGH,
        agree,
        window_agreement is not None and window_agreement >= KEY_WINDOW_AGREEMENT_HIGH,
    ])
    if not agree:
        notes.append(
            f"key profiles alone prefer {_name_of(choice['unrestricted'])}; its notes fit the song worse "
            f"than {_name_of(choice['index'])}'s, so the home note is the uncertain part"
        )
    if choice["scale_share"] < SCALE_FIT_MIN or choice["pair_margin"] < PAIR_MARGIN_MIN:
        confidence = "low"
    elif points == 4 or (points == 3 and agree):
        confidence = "high"
    elif points >= 2:
        confidence = "medium"
    else:
        confidence = "low"
    s = choice["scores"]
    top = [int(i) for i in np.argsort(-s)[:3]]
    return {
        "key": _name_of(choice["index"]),
        "confidence": confidence,
        "correlation": round(float(s[choice["index"]]), 3),
        "relative": {"key": _name_of(choice["other"]), "correlation": round(float(s[choice["other"]]), 3)},
        "profile_top3": [{"key": _name_of(i), "correlation": round(float(s[i]), 3)} for i in top],
        "scale_share": round(choice["scale_share"], 3),
        "window_agreement": round(window_agreement, 2) if window_agreement is not None else None,
        "windows": n_win,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# tempo
# ---------------------------------------------------------------------------

def _onset_at(oenv: np.ndarray, frames: np.ndarray) -> np.ndarray:
    k = ONSET_PEAK_FRAMES
    frames = np.clip(np.asarray(frames, dtype=int), 0, len(oenv) - 1)
    return np.array([oenv[max(0, f - k):f + k + 1].max() for f in frames]) if len(frames) else np.zeros(0)


def _level_alternatives(bpm: float, offbeat_ratio: float | None, alternation_ratio: float | None) -> list[dict]:
    alts = []
    if 2 * bpm <= DOUBLE_MAX_BPM and offbeat_ratio is not None and offbeat_ratio >= DOUBLE_OFFBEAT_RATIO:
        alts.append({"bpm": round(2 * bpm, 1), "relation": "double",
                     "evidence": f"off-beats carry {offbeat_ratio:.2f} of the on-beat onset strength"})
    if bpm / 2 >= HALF_MIN_BPM:
        if bpm >= FAST_BPM:
            alts.append({"bpm": round(bpm / 2, 1), "relation": "half",
                         "evidence": f"{bpm:.0f} BPM is fast enough that most listeners feel it at half"})
        elif alternation_ratio is not None and alternation_ratio <= HALF_ALTERNATION_RATIO:
            alts.append({"bpm": round(bpm / 2, 1), "relation": "half",
                         "evidence": f"every other beat is accented (weaker/stronger onsets {alternation_ratio:.2f})"})
    return alts


def estimate_tempo(y: np.ndarray, sr: int) -> dict:
    import librosa

    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    if len(oenv) * HOP / sr < MIN_DURATION_S or float(oenv.max()) <= 0:
        return {"bpm": None, "confidence": "low", "beat_times": [], "ambiguous": False, "alternatives": [],
                "notes": ["too short or no onsets"]}
    bpm = float(librosa.feature.tempo(onset_envelope=oenv, sr=sr, hop_length=HOP, start_bpm=TEMPO_PRIOR_BPM)[0])
    _, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=HOP, start_bpm=bpm, tightness=BEAT_TIGHTNESS)
    beats = np.asarray(beats, dtype=int)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP)
    notes: list[str] = []
    if len(beats) < 4:
        return {"bpm": round(bpm, 1), "confidence": "low", "beat_times": [round(float(t), 3) for t in beat_times],
                "ambiguous": False, "alternatives": [], "notes": ["too few beats to check the grid"]}

    ibi = np.diff(beat_times)
    # beat frames are quantised to the hop; a straight line through all beats averages that out
    idx = np.arange(len(beat_times))
    inliers = np.abs(ibi - np.median(ibi)) <= BEAT_OUTLIER_FRAC * np.median(ibi)
    keep = np.concatenate([[True], inliers]) & np.concatenate([inliers, [True]])
    slope = float(np.polyfit(idx[keep], beat_times[keep], 1)[0]) if keep.sum() >= 4 else float(np.median(ibi))
    if abs(slope - np.median(ibi)) > BEAT_OUTLIER_FRAC * np.median(ibi):  # gaps in the grid skew the line
        slope = float(np.median(ibi))
    grid_bpm = 60.0 / slope
    cv = float(np.std(ibi) / (np.mean(ibi) + EPS))
    on = _onset_at(oenv, beats)
    off = _onset_at(oenv, ((beats[:-1] + beats[1:]) / 2).astype(int))
    offbeat_ratio = float(np.median(off) / (np.median(on) + EPS)) if len(off) else None
    even, odd = float(on[0::2].mean()), float(on[1::2].mean())
    alternation_ratio = min(even, odd) / (max(even, odd) + EPS)

    # local steadiness
    w = int(round(LOCAL_TEMPO_WINDOW_S * sr / HOP))
    local = []
    for a in range(0, max(1, len(oenv) - w // 2), max(1, w)):
        seg = oenv[a:a + w]
        if len(seg) * HOP / sr < LOCAL_TEMPO_WINDOW_S / 2 or seg.max() <= 0:
            continue
        local.append(float(librosa.feature.tempo(onset_envelope=seg, sr=sr, hop_length=HOP, start_bpm=grid_bpm)[0]))
    agree = [any(abs(t * m - grid_bpm) <= LOCAL_TEMPO_TOL * grid_bpm for m in (0.5, 1.0, 2.0)) for t in local]
    steady_share = float(np.mean(agree)) if local else None

    alts = _level_alternatives(grid_bpm, offbeat_ratio, alternation_ratio)
    ambiguous = bool(alts)
    if cv <= BEAT_CV_HIGH and (steady_share is None or steady_share >= STEADY_SHARE_HIGH):
        confidence = "high"
    elif cv <= BEAT_CV_MEDIUM:
        confidence = "medium"
    else:
        confidence = "low"
    if ambiguous and confidence == "high":
        confidence = "medium"
        notes.append("the beat grid is steady; which metrical level is 'the tempo' is the uncertain part")
    if steady_share is not None and steady_share < STEADY_SHARE_HIGH:
        notes.append(f"tempo differs across the song: only {steady_share:.0%} of {LOCAL_TEMPO_WINDOW_S:.0f} s windows agree")
    return {
        "bpm": round(grid_bpm, 1),
        "confidence": confidence,
        "beat_times": [round(float(t), 3) for t in beat_times],
        "ambiguous": ambiguous,
        "alternatives": alts,
        "beat_interval_cv": round(cv, 3),
        "offbeat_ratio": round(offbeat_ratio, 2) if offbeat_ratio is not None else None,
        "alternation_ratio": round(alternation_ratio, 2),
        "local_bpm": [round(t, 1) for t in local],
        "steady_share": round(steady_share, 2) if steady_share is not None else None,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def _pool(X: np.ndarray, n: int) -> np.ndarray:
    k = X.shape[1] // n
    return X[:, :k * n].reshape(X.shape[0], k, n).mean(axis=2)


def _cosine_ssm(X: np.ndarray) -> np.ndarray:
    Z = X - X.mean(axis=1, keepdims=True)
    Z = Z / (Z.std(axis=1, keepdims=True) + EPS)
    Z = Z / (np.linalg.norm(Z, axis=0, keepdims=True) + EPS)
    return Z.T @ Z


def _foote(S: np.ndarray, half: int) -> np.ndarray:
    """Checkerboard-kernel novelty, normalised by the kernel weight that lies inside the matrix."""
    n = S.shape[0]
    if n == 0 or half < 1:
        return np.zeros(n)
    g = np.exp(-0.5 * ((np.arange(2 * half) - half + 0.5) / (half / 2)) ** 2)
    ker = np.outer(g, g)
    ker[:half, half:] *= -1
    ker[half:, :half] *= -1
    out = np.zeros(n)
    for i in range(n):
        a, b = i - half, i + half
        a0, b0 = max(0, a), min(n, b)
        kk = ker[a0 - a:2 * half - (b - b0), a0 - a:2 * half - (b - b0)]
        out[i] = float((S[a0:b0, a0:b0] * kk).sum() / (np.abs(kk).sum() + EPS))
    return np.maximum(out, 0.0)


def detect_sections(y: np.ndarray, sr: int) -> list[dict]:
    import librosa
    from scipy.signal import find_peaks

    duration = len(y) / sr
    if duration < 2 * SECTION_KERNELS_S[0]:
        return []
    per = max(1, int(round(SECTION_FRAME_S * sr / HOP)))
    dt = per * HOP / sr
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=N_MFCC)[1:]
    rms = librosa.feature.rms(y=y, hop_length=HOP)
    n = min(chroma.shape[1], mfcc.shape[1], rms.shape[1])
    C, M = _pool(chroma[:, :n], per), _pool(mfcc[:, :n], per)
    L = 10 * np.log10(_pool(np.square(rms[:, :n]), per)[0] + EPS)
    L = np.maximum(L, L.max() - 60.0)
    level_ssm = 2 * np.exp(-0.5 * ((L[:, None] - L[None, :]) / LEVEL_SIM_DB) ** 2) - 1
    ssms = (_cosine_ssm(C), _cosine_ssm(M), level_ssm)
    total = np.zeros(C.shape[1])
    for half_s in SECTION_KERNELS_S:
        half = max(1, int(round(half_s / dt)))
        for S in ssms:
            nov = _foote(S, half)
            total += nov / (nov.max() + EPS)
    total /= len(SECTION_KERNELS_S) * len(ssms)
    total = total / (total.max() + EPS)
    peaks, props = find_peaks(total, distance=max(1, int(round(SECTION_MIN_GAP_S / dt))),
                              prominence=SECTION_MIN_PROMINENCE)
    out = []
    for p, prom in zip(peaks, props["prominences"]):
        t = (p + 0.5) * dt
        if t < SECTION_EDGE_GUARD_S or t > duration - SECTION_EDGE_GUARD_S:
            continue
        conf = "high" if prom >= SECTION_HIGH_PROMINENCE else "medium" if prom >= SECTION_MEDIUM_PROMINENCE else "low"
        out.append({"t": round(float(t), 1), "strength": round(float(prom), 2), "confidence": conf})
    return out


# ---------------------------------------------------------------------------
# energy
# ---------------------------------------------------------------------------

def _biquad_high_shelf(fs: float, fc: float, gain_db: float, q: float):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b = [A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha), -2 * A * ((A - 1) + (A + 1) * cw),
         A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
    a = [(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha, 2 * ((A - 1) - (A + 1) * cw),
         (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def _biquad_high_pass(fs: float, fc: float, q: float):
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
    a = [1 + alpha, -2 * cw, 1 - alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import lfilter

    b1, a1 = _biquad_high_shelf(sr, K_SHELF_FC, K_SHELF_GAIN_DB, K_SHELF_Q)
    b2, a2 = _biquad_high_pass(sr, K_HP_FC, K_HP_Q)
    return lfilter(b2, a2, lfilter(b1, a1, x.astype(np.float64)))


def loudness_curve(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """(centre times, K-weighted short-term level in dB) every ENERGY_HOP_S over ENERGY_WINDOW_S."""
    kw = k_weight(x, sr)
    win, hop = int(round(ENERGY_WINDOW_S * sr)), max(1, int(round(ENERGY_HOP_S * sr)))
    if len(kw) < win:
        win = len(kw)
    if win == 0:
        return np.zeros(0), np.zeros(0)
    csum = np.concatenate([[0.0], np.cumsum(np.square(kw))])
    starts = np.arange(0, len(kw) - win + 1, hop)
    ms = (csum[starts + win] - csum[starts]) / win
    db = np.maximum(-0.691 + 10 * np.log10(ms + EPS), DB_FLOOR)
    return (starts + win / 2) / sr, db


def estimate_energy(x: np.ndarray, sr: int) -> dict:
    times, db = loudness_curve(x, sr)
    if len(db) == 0 or db.max() <= DB_FLOOR:
        return {"loudest_t": None, "loudest_db": None, "confidence": "low", "notes": ["silent"]}
    i = int(np.argmax(db))
    far = np.abs(times - times[i]) >= LOUDEST_SEPARATE_S
    margin = float(db[i] - db[far].max()) if far.any() else None
    if margin is None or margin >= LOUDEST_MARGIN_HIGH_DB:
        conf = "high" if margin is not None else "medium"
    elif margin >= LOUDEST_MARGIN_MEDIUM_DB:
        conf = "medium"
    else:
        conf = "low"
    near = db >= db[i] - NEAR_LOUDEST_DB
    near_s = float(near.sum() * ENERGY_HOP_S)
    j0, j1 = i, i
    while j0 > 0 and near[j0 - 1]:
        j0 -= 1
    while j1 < len(db) - 1 and near[j1 + 1]:
        j1 += 1
    return {
        "loudest_t": round(float(times[i]), 1),
        "loudest_db": round(float(db[i]), 1),
        "unit": f"K-weighted short-term level ({ENERGY_WINDOW_S:.0f} s window, ungated, LUFS scale)",
        "confidence": conf,
        "margin_db": round(margin, 1) if margin is not None else None,
        "seconds_within_1db": round(near_s, 1),
        "loudest_span": [round(float(times[j0]), 1), round(float(times[j1]), 1)],
        "quietest_db": round(float(db.min()), 1),
    }


# ===========================================================================
# public
# ===========================================================================

def analyze_harmony(mix: np.ndarray, sr: int) -> dict:
    """Key, tempo + beats (with half/double-time reading), sections, loudest moment, duration."""
    import librosa

    x = np.asarray(mix, dtype=np.float32).reshape(-1)
    duration = round(len(x) / sr, 2) if sr else 0.0
    if sr != ANALYSIS_SR and len(x):
        x = librosa.resample(x, orig_sr=sr, target_sr=ANALYSIS_SR)
    sr_a = ANALYSIS_SR
    notes = ["measured on the full mix; section times are where tone colour, harmony or level change most, not labelled verse/chorus"]
    if len(x) / sr_a < MIN_DURATION_S or float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) < 1e-5:
        return {"key": {"key": None, "confidence": "low"},
                "tempo": {"bpm": None, "confidence": "low", "beat_times": [], "ambiguous": False, "alternatives": []},
                "sections": [], "section_details": [],
                "energy": estimate_energy(x, sr_a) if len(x) else {"loudest_t": None, "loudest_db": None, "confidence": "low"},
                "duration_s": duration, "notes": notes + ["too short or silent for song facts"]}
    harmonic = librosa.effects.harmonic(x)
    key = estimate_key(x, sr_a, harmonic=harmonic)
    tempo = estimate_tempo(x, sr_a)
    sections = detect_sections(x, sr_a)
    energy = estimate_energy(x, sr_a)
    return {
        "key": key,
        "tempo": tempo,
        "sections": [s["t"] for s in sections if s["confidence"] != "low"],
        "section_details": sections,
        "energy": energy,
        "duration_s": duration,
        "notes": notes,
    }


def format_harmony_section(result: dict) -> str:
    k, t, e = result.get("key") or {}, result.get("tempo") or {}, result.get("energy") or {}
    lines = []
    if k.get("key"):
        lines.append(f"KEY: {k['key']} ({k['confidence']})" + (f" · or {k['relative']['key']}" if k.get("relative") else ""))
    if t.get("bpm"):
        alt = "; ".join(f"{a['relation']} {a['bpm']:.0f}" for a in t.get("alternatives", []))
        lines.append(f"TEMPO: {t['bpm']:.1f} BPM ({t['confidence']})" + (f" · could be felt as {alt}" if alt else ""))
    if result.get("sections"):
        lines.append("SECTIONS: " + ", ".join(f"{int(s // 60)}:{s % 60:04.1f}" for s in result["sections"]))
    if e.get("loudest_t") is not None:
        lines.append(f"LOUDEST: {int(e['loudest_t'] // 60)}:{e['loudest_t'] % 60:04.1f} at {e['loudest_db']} ({e['confidence']})")
    return "\n".join(lines)
