# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Synthetic vocal-stem fixtures for the breath organ, with known ground truth.

Pure numpy. Tests import these builders directly; running this file writes
WAVs + truth JSON to fixtures/audio/ (gitignored) for listening by ear.

Every builder returns (signal float32, truth dict). truth carries:
  phrases:  [(start_s, end_s, f0_hz)]  where the sung tone is
  breaths:  [(start_s, end_s)]         where breath noise was placed
  gaps:     [(start_s, end_s)]         spans between phrases
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from escutario.types import PitchTrack  # noqa: E402

BREATH_REL_DB = -22.0   # breath RMS relative to phrase RMS — typical of an audible inhale
PHRASE_PEAK = 0.35      # linear peak of a sung tone before the harmonic sum is normalised
FLOOR_DBFS = -72.0      # stem residue under everything (a separated stem is never digital zero)


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def tone(duration_s: float, sr: int, f0: float, *, vibrato_hz: float = 5.5,
         vibrato_cents: float = 30.0, n_harm: int = 24, fade_s: float = 0.03) -> np.ndarray:
    """A sung-ish harmonic tone with light vibrato and soft attack/release."""
    n = int(round(duration_s * sr))
    t = np.arange(n) / sr
    cents = vibrato_cents * np.sin(2 * np.pi * vibrato_hz * t)
    inst_f = f0 * 2.0 ** (cents / 1200.0)
    phase = 2 * np.pi * np.cumsum(inst_f) / sr
    y = np.zeros(n)
    for k in range(1, n_harm + 1):
        if f0 * k * 1.02 >= sr / 2:
            break
        y += np.sin(k * phase) / k ** 1.3
    y *= PHRASE_PEAK / (np.max(np.abs(y)) + 1e-12)
    fade = max(1, int(fade_s * sr))
    env = np.ones(n)
    ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
    env[:fade] = ramp
    env[-fade:] = ramp[::-1]
    return (y * env).astype(np.float64)


def bandpassed_noise(n: int, sr: int, rng: np.random.Generator, lo_hz: float = 1000.0,
                     hi_hz: float = 6000.0, skirt_oct: float = 0.5) -> np.ndarray:
    """White noise shaped by a smooth band-pass (raised-cosine skirts in octaves)."""
    x = rng.standard_normal(n)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    lf = np.log2(np.maximum(f, 1.0))
    gain = np.ones_like(f)
    lo, hi = np.log2(lo_hz), np.log2(hi_hz)
    below = lf < lo
    gain[below] = np.cos(np.clip((lo - lf[below]) / skirt_oct, 0, 1) * np.pi / 2) ** 2
    above = lf > hi
    gain[above] = np.cos(np.clip((lf[above] - hi) / skirt_oct, 0, 1) * np.pi / 2) ** 2
    y = np.fft.irfft(spec * gain, n)
    return y / (_rms(y) + 1e-12)


def breath_burst(duration_s: float, sr: int, rms_lin: float, rng: np.random.Generator,
                 band=(1000.0, 6000.0)) -> np.ndarray:
    """An inhale: band-passed noise under a rise-then-fall (Hann) envelope."""
    n = int(round(duration_s * sr))
    env = np.hanning(n)
    y = bandpassed_noise(n, sr, rng, lo_hz=band[0], hi_hz=band[1]) * env
    return y * (rms_lin / (_rms(y) + 1e-12))


def hihat_click(sr: int, peak_lin: float, rng: np.random.Generator, tau_s: float = 0.02,
                length_s: float = 0.15) -> np.ndarray:
    """A hi-hat-like click: bright broadband noise, instant attack, exponential decay."""
    n = int(round(length_s * sr))
    x = bandpassed_noise(n, sr, rng, lo_hz=3000.0, hi_hz=min(12000.0, sr / 2 * 0.95))
    env = np.exp(-np.arange(n) / (tau_s * sr))
    y = x * env
    return y * (peak_lin / (np.max(np.abs(y)) + 1e-12))


def add_at(dst: np.ndarray, src: np.ndarray, start_s: float, sr: int) -> None:
    i = int(round(start_s * sr))
    j = min(len(dst), i + len(src))
    dst[i:j] += src[: j - i]


def floor_noise(n: int, rng: np.random.Generator, db: float = FLOOR_DBFS) -> np.ndarray:
    return rng.standard_normal(n) * _db_to_lin(db)


def song(sr: int = 22050, *, seed: int = 7, phrase_durs=(2.4, 1.8, 3.2, 2.0, 2.6),
         gap_kinds=("breath", "breath", "breath", "breath"), gap_s: float = 0.8,
         breath_s=(0.45, 0.30, 0.65, 0.40), breath_rel_db: float = BREATH_REL_DB,
         lead_in_s: float = 0.6, tail_s: float = 0.6, lead_breath_s: float | None = 0.35,
         hihat_period_s: float | None = None, hihat_peak_db_rel_phrase: float = -6.0,
         digital_silence_gaps: bool = False, breath_band=(1000.0, 6000.0)):
    """Tone phrases separated by gaps of a given kind.

    gap_kinds entries: "breath" (band-passed rise/fall noise), "silence"
    (floor only), "loud" (same noise burst but as loud as the singing),
    "click" (nothing but whatever hi-hat leakage is laid over the track).
    """
    rng = np.random.default_rng(seed)
    f0s = [220.0, 246.9, 196.0, 261.6, 233.1, 207.7, 293.7]
    total = lead_in_s + sum(phrase_durs) + gap_s * (len(phrase_durs) - 1) + tail_s
    n = int(round(total * sr))
    y = floor_noise(n, rng)
    truth = {"phrases": [], "breaths": [], "gaps": [], "silent_gaps": [], "sr": sr}
    t = lead_in_s
    phrase_rms = None
    for pi, dur in enumerate(phrase_durs):
        f0 = f0s[pi % len(f0s)]
        tn = tone(dur, sr, f0)
        phrase_rms = _rms(tn)
        add_at(y, tn, t, sr)
        truth["phrases"].append((t, t + dur, f0))
        t += dur
        if pi < len(phrase_durs) - 1:
            g0, g1 = t, t + gap_s
            truth["gaps"].append((g0, g1))
            kind = gap_kinds[pi]
            if kind in ("breath", "loud"):
                bd = breath_s[pi % len(breath_s)]
                rel = breath_rel_db if kind == "breath" else 0.0
                b = breath_burst(bd, sr, phrase_rms * _db_to_lin(rel), rng, band=breath_band)
                b_start = g1 - 0.08 - bd  # inhale sits just before the next line
                add_at(y, b, b_start, sr)
                if kind == "breath":
                    truth["breaths"].append((b_start, b_start + bd))
            elif kind == "silence":
                truth["silent_gaps"].append((g0, g1))
                if digital_silence_gaps:
                    i0, i1 = int(round(g0 * sr)) + int(0.03 * sr), int(round(g1 * sr)) - int(0.03 * sr)
                    y[i0:i1] = 0.0
            t = g1
    if lead_breath_s:
        ref = _rms(tone(1.0, sr, f0s[0]))
        b = breath_burst(lead_breath_s, sr, ref * _db_to_lin(breath_rel_db), rng, band=breath_band)
        b_start = lead_in_s - 0.08 - lead_breath_s
        add_at(y, b, b_start, sr)
        truth["breaths"].insert(0, (b_start, b_start + lead_breath_s))
    if hihat_period_s:
        ref = _rms(tone(1.0, sr, f0s[0]))
        peak = ref * _db_to_lin(hihat_peak_db_rel_phrase) * 3.0
        tt = 0.05
        while tt < total - 0.2:
            add_at(y, hihat_click(sr, peak, rng), tt, sr)
            tt += hihat_period_s
    truth["duration_s"] = total
    return y.astype(np.float32), truth


def truth_pitch_track(truth: dict, hop_s: float = 0.01, conf: float = 0.9) -> PitchTrack:
    """A PitchTrack built from where the tones really are."""
    n = int(np.ceil(truth["duration_s"] / hop_s))
    times = (np.arange(n) + 0.5) * hop_s
    f0 = np.zeros(n)
    c = np.full(n, 0.05)
    for s, e, hz in truth["phrases"]:
        m = (times >= s) & (times < e)
        f0[m] = hz
        c[m] = conf
    return PitchTrack(times=times, f0_hz=f0, confidence=c, hop_s=hop_s)


def write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "audio"
    cases = {
        "breaths": dict(),
        "silences": dict(gap_kinds=("silence",) * 4, lead_breath_s=None),
        "hihat_only": dict(gap_kinds=("click",) * 4, lead_breath_s=None, hihat_period_s=0.25),
        "breaths_with_hihat": dict(hihat_period_s=0.25),
    }
    for name, kw in cases.items():
        y, truth = song(**kw)
        write_wav(out / f"breath_{name}.wav", y, truth["sr"])
        (out / f"breath_{name}.json").write_text(json.dumps(truth, indent=2))
        print("wrote", out / f"breath_{name}.wav")
