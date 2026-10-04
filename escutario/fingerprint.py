# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Sound fingerprints — so the same song is heard once, whatever the file is called.

Anne dropped one song twice under two names (17 Sep). The files differed byte for byte (a different
container and tags), so a file hash could not see it; the sound was identical. This fingerprint
reads the sound itself: for each short frame it records, in 32 bits, whether energy rose or fell
between neighbouring frequency bands compared with the frame before (the approach of Haitsma &
Kalker's audio fingerprinting, written here from the idea, not from any library). Re-encoding,
renaming, a volume change or a little silence at the start barely move those bits; a different song
shares about half of them by chance.

Measured on the real files: the two prayer-song drops differ in 0.000 of their bits; any two
different songs differ in about 0.49. A match needs 0.25 or less.
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

import numpy as np

SR = 11025
N_FFT = 2048
BANDS = 33                   # 33 bands -> 32 difference bits per frame
HOP = 256
SECONDS = 90                 # the opening 90 s identifies a recording; the whole song is not needed
FMIN, FMAX = 300, 2000
MAX_SHIFT_S = 10.0           # tolerate up to 10 s of extra or missing lead-in
MIN_INFORMATIVE = 0.5        # at least half the frames must carry signal (silence has all-zero bits and matches anything silent)
DURATION_TOLERANCE_S = 2.0   # a same-sound match also needs the whole recordings to be the same length
MIN_OVERLAP_FRAMES = 200     # about 4.6 s of shared sound before a comparison counts
MATCH_BELOW = 0.25           # share of differing bits at or under which two recordings are the same


def decode(path: Path, seconds: float = SECONDS) -> np.ndarray:
    """The opening seconds of any audio or video file, as mono float samples (ffmpeg reads every container)."""
    done = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-t", str(seconds), "-i", str(path), "-vn",
         "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
        capture_output=True, timeout=120)
    if done.returncode != 0:
        raise RuntimeError("could not decode the audio for its fingerprint")
    return np.frombuffer(done.stdout, dtype="<i2").astype(np.float32) / 32768.0


def fingerprint_samples(y: np.ndarray) -> np.ndarray:
    """frames x 32 bits (uint8 0/1)."""
    import librosa

    if y.size < N_FFT * 2:
        return np.zeros((0, BANDS - 1), dtype=np.uint8)
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=BANDS, fmin=FMIN, fmax=FMAX, power=2.0)
    E = np.log(S + 1e-10)
    across = E[:-1, :] - E[1:, :]                       # neighbouring bands
    bits = (across[:, 1:] - across[:, :-1]) > 0         # compared with the frame before
    return bits.T.astype(np.uint8)


def fingerprint_file(path: Path) -> np.ndarray:
    return fingerprint_samples(decode(Path(path)))


def duration_of(path: Path) -> float | None:
    """The whole recording's length in seconds (ffprobe reads every container), or None when unknown."""
    done = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
                          capture_output=True, text=True, timeout=60)
    try:
        d = float(done.stdout.strip())
        return d if np.isfinite(d) and d > 0 else None
    except ValueError:
        return None


def informative(bits: np.ndarray) -> bool:
    """True when most frames carry signal; a silent or near-constant opening cannot identify a recording."""
    return len(bits) >= MIN_OVERLAP_FRAMES and float(np.mean(bits.any(axis=1))) >= MIN_INFORMATIVE


def pack(bits: np.ndarray, duration: float | None = None) -> dict:
    out = {"frames": int(bits.shape[0]), "bits": base64.b64encode(np.packbits(bits.reshape(-1)).tobytes()).decode()}
    if duration:
        out["duration"] = round(float(duration), 2)
    return out


def unpack(data: dict) -> np.ndarray:
    frames = int(data["frames"])
    flat = np.unpackbits(np.frombuffer(base64.b64decode(data["bits"]), dtype=np.uint8))[: frames * (BANDS - 1)]
    return flat.reshape(frames, BANDS - 1)


def bit_error(a: np.ndarray, b: np.ndarray, max_shift_s: float = MAX_SHIFT_S) -> float:
    """Smallest share of differing bits over every alignment within max_shift_s. 1.0 when they barely overlap."""
    max_shift = int(max_shift_s * SR / HOP)
    best = 1.0
    for s in range(-max_shift, max_shift + 1):
        x, y = (a[s:], b) if s >= 0 else (a, b[-s:])
        n = min(len(x), len(y))
        if n < MIN_OVERLAP_FRAMES:
            continue
        best = min(best, float(np.mean(x[:n] != y[:n])))
    return best


def find_same(bits: np.ndarray, known: dict[str, dict], threshold: float = MATCH_BELOW,
              duration: float | None = None) -> tuple[str, float] | None:
    """The id of an already-heard song with the same sound, and how close it was; None when it is new.

    A sound match alone never suppresses a hearing: the opening must carry signal, and the whole
    recordings must be the same length (so a live version or an extended mix sharing an intro is
    heard). Without a length on both sides there is no match; only identical bytes are certain.
    """
    if not informative(bits) or not duration:
        return None
    best: tuple[str, float] | None = None
    for song_id, packed in known.items():
        try:
            other_duration = packed.get("duration")
            if not other_duration or abs(float(other_duration) - float(duration)) > DURATION_TOLERANCE_S:
                continue
            other = unpack(packed)
            if not informative(other):
                continue
            err = bit_error(bits, other)
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        if err <= threshold and (best is None or err < best[1]):
            best = (song_id, err)
    return best
