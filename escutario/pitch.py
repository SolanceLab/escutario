# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Escutário's pitch tracker — one f0 track per separated vocal stem.

Two engines, one output shape (`types.PitchTrack`, steady 10 ms hop):

- ``"crepe"`` — the CREPE network (Kim et al. 2018) via the ``torchcrepe``
  package's weights, run on MPS when present. Decoding, voicing and
  confidence are our own: a local-transition Viterbi path over the 360
  pitch bins (octave hops cost many frames, so a stray harmonic cannot
  flip the line), a weighted cents read around the chosen bin (keeps
  vibrato extent), and a voicing decision from the network's periodicity.
- ``"pyin"`` — librosa's probabilistic YIN; CPU only, slower, kept as the
  no-torch fallback.

``method="auto"`` picks crepe when torch + torchcrepe import, else pyin.
The default was chosen on the real mashup stem — see the evidence table
in the build report (voiced coverage while singing, octave-jump rate,
agreement with basic-pitch notes, runtime).

Confidence semantics (both engines): 0..1, monotonic in the engine's own
voicing evidence, calibrated so that **0.5 is the voicing boundary** —
voiced frames carry confidence >= 0.5, unvoiced frames < 0.5 and f0 = 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .types import PitchTrack

# ---------------------------------------------------------------- range & hop
FMIN_HZ = 65.0     # starting value to calibrate: lowest sung f0 we accept (C2)
FMAX_HZ = 1100.0   # starting value to calibrate: highest sung f0 we accept (~C#6)
HOP_S = 0.010      # starting value to calibrate: analysis hop in seconds (steady grid)

# ---------------------------------------------------------------- crepe engine
CREPE_MODEL = "full"             # starting value to calibrate: network capacity; "tiny" is ~4-10x faster but read the mashup G#5 at 1:12 as E5 (median MIDI 75.8), "full" as G#5 (79.97)
CREPE_BATCH_FRAMES = {"full": 256, "tiny": 1024}  # frames per batch; measured: "full" at 1024 silently returns constant output on MPS (torch 2.14); 256 and 512 match CPU
VITERBI_MAX_STEP_BINS = 12       # starting value to calibrate: local moves span offsets -11..+11 bins (220 cents / 10 ms); the sweep and the ear checks ran with exactly this
MIN_VOICED_RUN_FRAMES = 3        # starting value to calibrate: shorter voiced islands are dropped as blips
WEIGHTED_READ_HALF_BINS = 4      # starting value to calibrate: bins each side averaged for the sub-bin cents read
CREPE_DECODER = "viterbi"        # starting value to calibrate: "viterbi" (local moves + penalised leaps), "viterbi-local" (local moves only) or "argmax" (per frame)
CREPE_SR = 16000                 # the network's fixed input rate (not tunable)
CREPE_WINDOW = 1024              # the network's fixed frame length in samples (not tunable)
VITERBI_JUMP_LOG_PENALTY = -15.0  # starting value to calibrate: log cost of a free leap in one hop; mashup sweep -6/-10/-15/-20: octave jumps 2.3/1.1/0.31/0.10 per 1k voiced pairs, coverage 1:09-1:18 0.90/0.87/0.84/0.84
PERIODICITY_SMOOTH_FRAMES = 3    # starting value to calibrate: median window on periodicity before voicing
CREPE_VOICING_THRESHOLD = 0.25   # starting value to calibrate: smoothed periodicity at/above which a frame is voiced (0.35/0.45 lost 7-16 pts of 1:09-1:18 coverage; quiet-frame voicing rose only 0.027->0.032)
VITERBI_EMISSION_FLOOR = 1e-4    # starting value to calibrate: added to normalised activations before the log
CREPE_CENTS_PER_BIN = 20.0       # the network's bin spacing in cents (not tunable)
SILENCE_DB = -60.0               # starting value to calibrate: frame RMS (dBFS) below which a frame is unvoiced outright
DEGENERATE_SPREAD = 1e-3         # starting value to calibrate: peak-activation range below which a non-silent track is a failed inference
CREPE_BINS = 360                 # the network's pitch bins (not tunable)
CREPE_RETRY_BATCH_FRAMES = 64    # batch size for the one retry when activations come back degenerate
CREPE_CENTS_OFFSET = 1997.3794084376191  # cents (re 10 Hz) of bin 0 in CREPE's layout (not tunable)

# ---------------------------------------------------------------- pyin engine
PYIN_CONF_PIVOT = 0.25           # starting value to calibrate: pyin voiced probability mapped to confidence 0.5 (voicing itself is pyin's own HMM flag)
PYIN_MIN_PROB = 0.02             # starting value to calibrate: pyin flags white noise at its 0.01 probability floor; below this is never voiced
PYIN_SR = 22050                  # rate pyin runs at; hop 220 samples ≈ HOP_S
PYIN_FRAME = 2048                # starting value to calibrate: pyin frame length in samples at PYIN_SR

DEFAULT_METHOD = "crepe"         # chosen on the mashup evidence; "auto" resolves to this when torch is present
METHODS = ("auto", "crepe", "crepe-tiny", "crepe-full", "pyin")
CONF_BOUNDARY = 0.5              # confidence value that marks the voicing decision (contract of this module)


# ============================================================ public API


def track_pitch(vocal: np.ndarray, sr: int, *, method: str = "auto") -> PitchTrack:
    """Track f0 of a mono vocal signal. f0 0 = unvoiced; confidence 0..1."""
    if method not in METHODS:
        raise ValueError(f"unknown pitch method {method!r}; choose one of {METHODS}")
    x = np.asarray(vocal, dtype=np.float32)
    if x.ndim == 2:  # tolerate (channels, n) input
        x = x.mean(axis=0)
    if x.size and not np.all(np.isfinite(x)):
        raise ValueError("audio contains NaN or infinite samples; refusing to report them as silence")
    if sr <= 0:
        raise ValueError("sample rate must be positive")

    if method == "auto":
        method = DEFAULT_METHOD if (DEFAULT_METHOD != "crepe" or crepe_available()) else "pyin"
    if x.size == 0:
        return _empty_track()
    if method.startswith("crepe"):
        model = CREPE_MODEL if method == "crepe" else method.split("-", 1)[1]
        return _track_crepe(x, sr, model=model)
    return _track_pyin(x, sr)


def save_npz(track: PitchTrack, path: str | Path, sr: int) -> None:
    """Write a track with the keys Maii's pitch.npz used: times, f0, conf, hop_s, sr."""
    np.savez(
        path,
        times=np.asarray(track.times, dtype=np.float64),
        f0=np.asarray(track.f0_hz, dtype=np.float64),
        conf=np.asarray(track.confidence, dtype=np.float64),
        hop_s=np.float64(track.hop_s),
        sr=np.int64(sr),
    )


def load_npz(path: str | Path) -> PitchTrack:
    """Read a track written by `save_npz` (or an older pitch.npz with the same keys)."""
    d = np.load(path)
    return PitchTrack(times=d["times"], f0_hz=d["f0"], confidence=d["conf"], hop_s=float(d["hop_s"]))


def crepe_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchcrepe  # noqa: F401
    except Exception:
        return False
    return True


# ============================================================ shared helpers


def _empty_track() -> PitchTrack:
    z = np.zeros(0, dtype=np.float64)
    return PitchTrack(times=z, f0_hz=z.copy(), confidence=z.copy(), hop_s=HOP_S)


def _resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x.astype(np.float32, copy=False)
    g = np.gcd(int(sr), int(target))
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def _frame_rms_db(x: np.ndarray, hop: int, win: int, n_frames: int) -> np.ndarray:
    """Centered frame RMS in dBFS on the hop grid."""
    pad = np.pad(x.astype(np.float64), (win // 2, win // 2 + hop))
    c = np.concatenate([[0.0], np.cumsum(pad * pad)])
    starts = np.arange(n_frames) * hop
    ends = np.minimum(starts + win, len(pad))
    energy = (c[ends] - c[starts]) / np.maximum(ends - starts, 1)
    return 10.0 * np.log10(np.maximum(energy, 1e-20))


def _drop_short_runs(voiced: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1 or not voiced.any():
        return voiced
    v = voiced.copy()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], v.astype(np.int8), [0]])))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s < min_len:
            v[s:e] = False
    return v


def _calibrate_conf(evidence: np.ndarray, threshold: float, voiced: np.ndarray) -> np.ndarray:
    """Map engine evidence to 0..1 with CONF_BOUNDARY at the voicing threshold.

    Frames the post-filters unvoiced (silence gate, blips) are pinned just
    below the boundary so the semantics "conf >= 0.5 ⇔ voiced" always hold.
    """
    e = np.clip(np.nan_to_num(evidence, nan=0.0), 0.0, 1.0)
    thr = min(max(threshold, 1e-6), 1 - 1e-6)
    above = CONF_BOUNDARY + (1 - CONF_BOUNDARY) * (e - thr) / (1 - thr)
    below = CONF_BOUNDARY * e / thr
    conf = np.where(e >= thr, above, below)
    conf = np.where(voiced, np.maximum(conf, CONF_BOUNDARY), np.minimum(conf, CONF_BOUNDARY - 1e-3))
    return np.clip(conf, 0.0, 1.0)


# ============================================================ crepe engine


_MODELS: dict[tuple[str, str], object] = {}


def _crepe_model(capacity: str, device: str):
    import torch
    import torchcrepe

    key = (capacity, device)
    if key not in _MODELS:
        net = torchcrepe.Crepe(capacity)
        weights = Path(torchcrepe.__file__).parent / "assets" / f"{capacity}.pth"
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        net.eval()
        _MODELS[key] = net.to(torch.device(device))
    return _MODELS[key]


def _pick_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _crepe_activations(y16: np.ndarray, capacity: str, device: str, *, batch: int | None = None) -> np.ndarray:
    """Per-frame sigmoid activations, shape (n_frames, 360), frames centered on i*hop."""
    import torch

    hop = int(round(HOP_S * CREPE_SR))
    n_frames = 1 + len(y16) // hop
    padded = np.pad(y16, (CREPE_WINDOW // 2, CREPE_WINDOW // 2 + CREPE_WINDOW))
    net = _crepe_model(capacity, device)
    idx = np.arange(CREPE_WINDOW)
    out = np.empty((n_frames, CREPE_BINS), dtype=np.float32)
    with torch.no_grad():
        batch = batch or CREPE_BATCH_FRAMES.get(capacity, CREPE_RETRY_BATCH_FRAMES)
        for b0 in range(0, n_frames, batch):
            b1 = min(n_frames, b0 + batch)
            frames = padded[(np.arange(b0, b1) * hop)[:, None] + idx[None, :]].astype(np.float32)
            frames -= frames.mean(axis=1, keepdims=True)
            frames /= np.maximum(frames.std(axis=1, keepdims=True), 1e-10)
            t = torch.from_numpy(frames).to(device)
            out[b0:b1] = net(t).float().cpu().numpy()
    return out


def _bins_to_hz(cents_re_10hz: np.ndarray) -> np.ndarray:
    return 10.0 * 2.0 ** (cents_re_10hz / 1200.0)


def _hz_to_bin(hz: float) -> float:
    return (1200.0 * np.log2(hz / 10.0) - CREPE_CENTS_OFFSET) / CREPE_CENTS_PER_BIN


def _viterbi_local(act: np.ndarray, lo: int, hi: int, *, jump_log_penalty: float | None) -> np.ndarray:
    """Most likely bin path within [lo, hi).

    Emission = activations renormalised per frame (floored, so crossing
    empty bins costs a bounded amount). Transition: a triangular local
    move of up to VITERBI_MAX_STEP_BINS per hop, plus — unless
    ``jump_log_penalty`` is None — a leap to any bin at a fixed log cost,
    so a new voice or a real interval leap is taken once its evidence
    outweighs the cost, while a one- or two-frame octave flicker is not.
    """
    a = act[:, lo:hi].astype(np.float64)
    n, s = a.shape
    emit = np.log(a / np.maximum(a.sum(axis=1, keepdims=True), 1e-12) + VITERBI_EMISSION_FLOOR)
    k = VITERBI_MAX_STEP_BINS
    offsets = np.arange(-k + 1, k)
    trans = np.log((k - np.abs(offsets)) / float(k * k))
    score = emit[0].copy()
    back = np.empty((n, s), dtype=np.int16)
    back[0] = np.arange(s)
    src = np.arange(s)[None, :] - offsets[:, None]          # (n_off, s): predecessor index
    valid = (src >= 0) & (src < s)
    src_c = np.clip(src, 0, s - 1)
    penalty = np.where(valid, trans[:, None], -np.inf)
    cols = np.arange(s)
    for t in range(1, n):
        cand = score[src_c] + penalty                        # (n_off, s)
        best = cand.argmax(axis=0)
        local = cand[best, cols]
        prev = src_c[best, cols]
        if jump_log_penalty is not None:
            j = int(score.argmax())
            leap = score[j] + jump_log_penalty
            take = leap > local
            local = np.where(take, leap, local)
            prev = np.where(take, j, prev)
        score = local + emit[t]
        back[t] = prev
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path + lo


def _weighted_cents(act: np.ndarray, bins: np.ndarray, lo: int, hi: int) -> np.ndarray:
    w = WEIGHTED_READ_HALF_BINS
    offs = np.arange(-w, w + 1)
    idx = np.clip(bins[:, None] + offs[None, :], lo, hi - 1)
    p = np.take_along_axis(act, idx, axis=1).astype(np.float64)
    cents = CREPE_CENTS_OFFSET + CREPE_CENTS_PER_BIN * idx
    return (p * cents).sum(axis=1) / np.maximum(p.sum(axis=1), 1e-12)


def _median_filter(v: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(v) < width:
        return v
    from scipy.ndimage import median_filter

    return median_filter(v, size=width, mode="nearest")


def _weighted_argmax_bins(act: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Per-frame strongest bin inside [lo, hi) — no temporal model."""
    return act[:, lo:hi].argmax(axis=1) + lo


def _track_crepe(x: np.ndarray, sr: int, *, model: str) -> PitchTrack:
    y = _resample(x, sr, CREPE_SR)
    device = _pick_device()
    try:
        act = _crepe_activations(y, model, device)
    except Exception:  # an MPS op failure falls back to CPU rather than failing the song
        if device == "cpu":
            raise
        device = "cpu"
        act = _crepe_activations(y, model, device)
    if _degenerate(act, y):
        act = _crepe_activations(y, model, device, batch=CREPE_RETRY_BATCH_FRAMES)
        if _degenerate(act, y):
            raise RuntimeError(
                f"crepe-{model} on {device} returned constant activations for non-silent audio; "
                "refusing to report it as unvoiced (try method='pyin')")
    return _crepe_decode(act, y)


def _degenerate(act: np.ndarray, y16: np.ndarray) -> bool:
    """Constant network output over audible input = a failed inference, not silence."""
    hop = int(round(HOP_S * CREPE_SR))
    audible = _frame_rms_db(y16, hop, CREPE_WINDOW, act.shape[0]) > SILENCE_DB
    if not np.all(np.isfinite(act)):
        return True   # NaN/Inf activations are a failed inference, never silence
    if audible.sum() < MIN_VOICED_RUN_FRAMES:
        return False
    sub = act[audible]
    peaks = sub.max(axis=1)
    # a real steady note can have steady peak strength; a failed backend also returns the SAME vector
    # every frame, so require both a flat peak and an identical argmax bin before calling it broken
    same_bin = np.unique(sub.argmax(axis=1)).size == 1
    return float(peaks.max() - peaks.min()) < DEGENERATE_SPREAD and same_bin


def _crepe_decode(act: np.ndarray, y16: np.ndarray, *, decoder: str | None = None,
                  threshold: float | None = None) -> PitchTrack:
    """Activations -> PitchTrack. decoder/threshold overrides exist for calibration runs."""
    decoder = CREPE_DECODER if decoder is None else decoder
    threshold = CREPE_VOICING_THRESHOLD if threshold is None else threshold
    hop = int(round(HOP_S * CREPE_SR))
    n = act.shape[0]
    lo = max(0, int(np.floor(_hz_to_bin(FMIN_HZ))))
    hi = min(CREPE_BINS, int(np.ceil(_hz_to_bin(FMAX_HZ))) + 1)
    if decoder == "viterbi":
        bins = _viterbi_local(act, lo, hi, jump_log_penalty=VITERBI_JUMP_LOG_PENALTY)
    elif decoder == "viterbi-local":
        bins = _viterbi_local(act, lo, hi, jump_log_penalty=None)
    elif decoder == "argmax":
        bins = _weighted_argmax_bins(act, lo, hi)
    else:
        raise ValueError(f"unknown crepe decoder {decoder!r}")
    f0 = _bins_to_hz(_weighted_cents(act, bins, lo, hi))
    periodicity = _median_filter(act[np.arange(n), bins].astype(np.float64), PERIODICITY_SMOOTH_FRAMES)
    level = _frame_rms_db(y16, hop, CREPE_WINDOW, n)
    voiced = (periodicity >= threshold) & (level > SILENCE_DB)
    voiced &= (f0 >= FMIN_HZ) & (f0 <= FMAX_HZ)
    voiced = _drop_short_runs(voiced, MIN_VOICED_RUN_FRAMES)
    conf = _calibrate_conf(periodicity, threshold, voiced)
    times = np.arange(n) * hop / CREPE_SR
    return PitchTrack(times=times, f0_hz=np.where(voiced, f0, 0.0), confidence=conf, hop_s=hop / CREPE_SR)


# ============================================================ pyin engine


def _track_pyin(x: np.ndarray, sr: int) -> PitchTrack:
    import librosa

    y = _resample(x, sr, PYIN_SR)
    hop = int(round(HOP_S * PYIN_SR))
    if len(y) < PYIN_FRAME:
        y = np.pad(y, (0, PYIN_FRAME - len(y)))
    f0, flag, prob = librosa.pyin(
        y, fmin=FMIN_HZ, fmax=FMAX_HZ, sr=PYIN_SR, frame_length=PYIN_FRAME, hop_length=hop, center=True,
    )
    n = len(f0)
    prob = np.nan_to_num(prob, nan=0.0)
    level = _frame_rms_db(y, hop, PYIN_FRAME, n)
    voiced = np.asarray(flag, dtype=bool) & np.isfinite(f0) & (prob >= PYIN_MIN_PROB) & (level > SILENCE_DB)
    voiced = _drop_short_runs(voiced, MIN_VOICED_RUN_FRAMES)
    conf = _calibrate_conf(prob, PYIN_CONF_PIVOT, voiced)
    times = np.arange(n) * hop / PYIN_SR
    return PitchTrack(times=times, f0_hz=np.where(voiced, np.nan_to_num(f0), 0.0), confidence=conf, hop_s=hop / PYIN_SR)
