# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Named-instrument separation for Escutário — the violin separator.

Anne plays violin by ear; this pulls the violin (and its string neighbours) out of a song so
she can learn the line, and builds the play-along backing track (mix minus violin).

Backend: ZFTurbo Music-Source-Separation-Training release v1.0.21, "MVSep Mega" BS-Roformer,
53 stems (violin, viola, cello, bowed_strings, strings, ...). Architecture is vendored in
`escutario/vendor_bs_roformer.py` (see its header for source commit + modifications).

Security:
- Model files come ONLY from the official ZFTurbo GitHub release, pinned by sha256 + size.
  A mismatch is refused; an existing mismatching file is never overwritten.
- The .ckpt is a pickle: it is loaded ONLY with torch.load(weights_only=True). There is no
  fallback to weights_only=False — a failure raises UnsafeCheckpoint.
- The yaml uses `!!python/tuple`; it is parsed with a SafeLoader that adds a tuple constructor
  and nothing else (no arbitrary python object construction).

Duck-types escutario.split.SeparationBackend: `name`, `stems`, `separate(stereo_44k)`.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import resource
import tempfile
import time
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

_RELEASE = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.21/"


@dataclass(frozen=True)
class ModelAsset:
    filename: str
    url: str
    sha256: str
    size_bytes: int | None = None


YAML_ASSET = ModelAsset(
    filename="mvsep_mega_model_bs_roformer_53_stems.yaml",
    url=_RELEASE + "mvsep_mega_model_bs_roformer_53_stems.yaml",
    sha256="7e198062a251587088adb91215a4f44ab59e67bd62fcc805cf54d6e7dfc51103",
    size_bytes=4184,
)
CKPT_ASSET = ModelAsset(
    filename="mvsep_mega_model_bs_roformer_53_stems_v1.ckpt",
    url=_RELEASE + "mvsep_mega_model_bs_roformer_53_stems_v1.ckpt",
    sha256="c62820893bbf86d4e734f966bd142d9157cfc8bb8e79e9d8f9ea553f3ff3519f",
    size_bytes=1368919887,
)

HASH_CHUNK_BYTES = 8 * 1024 * 1024  # read size for hashing/downloading; not a tuning value
DOWNLOAD_MAX_STALLS = 5  # consecutive resume attempts that deliver zero bytes before giving up; starting value
DOWNLOAD_TIMEOUT_S = 60  # per-connection socket timeout; starting value
ALLOWED_DOWNLOAD_PREFIX = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/"


class ChecksumMismatch(RuntimeError):
    """Raised when a model file's sha256 does not match the pinned value."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(HASH_CHUNK_BYTES)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def ensure_asset(asset: ModelAsset, models_dir: Path = MODELS_DIR, *, download: bool = True,
                 opener=urllib.request.urlopen) -> Path:
    """Return the verified local path of `asset`, downloading it if absent.

    - An existing file whose sha256 differs is REFUSED (ChecksumMismatch); it is never overwritten.
    - A download goes to a temp file in the same directory and is only renamed into place
      after its sha256 matches; a mismatching download is deleted and refused.
    - Only the official ZFTurbo GitHub release URL prefix is allowed.
    """
    if not asset.url.startswith(ALLOWED_DOWNLOAD_PREFIX):
        raise ValueError(f"refusing to download from non-official URL: {asset.url}")
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / asset.filename
    if dest.exists():
        got = sha256_file(dest)
        if got != asset.sha256:
            raise ChecksumMismatch(
                f"{dest} exists but sha256 is {got}, expected {asset.sha256}; "
                "refusing to use or overwrite it — inspect and remove it by hand")
        return dest
    if not download:
        raise FileNotFoundError(f"{dest} is missing and download=False")

    fd, tmp_name = tempfile.mkstemp(prefix=asset.filename + ".", suffix=".part", dir=models_dir)
    tmp = Path(tmp_name)
    try:
        h = hashlib.sha256()
        n_bytes = 0
        stalls = 0
        with os.fdopen(fd, "wb") as out:
            while True:
                req = urllib.request.Request(asset.url)
                if n_bytes:
                    req.add_header("Range", f"bytes={n_bytes}-")
                got_any = False
                try:
                    with opener(req, timeout=DOWNLOAD_TIMEOUT_S) as resp:
                        status = getattr(resp, "status", 200)
                        if n_bytes and status != 206:
                            raise ChecksumMismatch(
                                f"server ignored resume request (HTTP {status}); discarded — retry")
                        while True:
                            block = resp.read(HASH_CHUNK_BYTES)
                            if not block:
                                break
                            got_any = True
                            h.update(block)
                            out.write(block)
                            n_bytes += len(block)
                except (OSError, http.client.HTTPException):
                    pass  # connection dropped mid-stream; resume below
                if asset.size_bytes is None or n_bytes >= asset.size_bytes:
                    break
                stalls = 0 if got_any else stalls + 1
                if stalls >= DOWNLOAD_MAX_STALLS:
                    break
        if asset.size_bytes is not None and n_bytes != asset.size_bytes:
            raise ChecksumMismatch(
                f"download of {asset.url} was {n_bytes} bytes, expected {asset.size_bytes} "
                "(truncated or wrong file); discarded — retry")
        got = h.hexdigest()
        if got != asset.sha256:
            raise ChecksumMismatch(f"downloaded {asset.url} has sha256 {got}, expected {asset.sha256}; discarded")
        if dest.exists():  # appeared concurrently — never clobber
            raise ChecksumMismatch(f"{dest} appeared during download; refusing to overwrite")
        os.replace(tmp, dest)
        return dest
    finally:
        if tmp.exists():
            tmp.unlink()


def ensure_model_files(models_dir: Path = MODELS_DIR, *, download: bool = True) -> tuple[Path, Path]:
    """(ckpt_path, yaml_path), verified."""
    yaml_path = ensure_asset(YAML_ASSET, models_dir, download=download)
    ckpt_path = ensure_asset(CKPT_ASSET, models_dir, download=download)
    return ckpt_path, yaml_path


# ---------------------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------------------

DEFAULT_STEMS = ["violin", "viola", "cello", "bowed_strings", "strings"]
SAMPLE_RATE = 44100  # the model's native rate (yaml audio.sample_rate)
BACKEND_NAME = "mvsep_mega_53"

MPS_MAX_HEADS = 8  # starting value: most mask heads per pass verified on MPS (M4 Pro 24 GB, 10 s chunk); 53 heads on 5 s hit a silent Metal OOM
FADE_FRACTION = 0.1  # starting value: linear fade length as a fraction of chunk, same as upstream demix
CHUNK_SECONDS_DEFAULT = 10.0  # starting value: yaml training chunk (441000 samples); inference yaml says 20 s, halved for 24 GB unified memory
MIN_WINDOW_SUM = 1e-4  # floor on summed overlap-add weights before dividing; guards the very edges
PEAK_CEILING = 0.99  # starting value: max |sample| allowed in a backing track before it is scaled down
SILENT_INPUT_PEAK = 1e-4  # starting value: an input chunk quieter than this may legitimately yield all-zero stems
NUM_OVERLAP_DEFAULT = 2  # starting value: chunks per step (2 = 50% overlap), matches yaml inference.num_overlap
MIN_INPUT_SAMPLES = 4096  # below this the STFT (n_fft 2048) has too little to work with; input is zero-padded up to it


class UnsafeCheckpoint(RuntimeError):
    """The checkpoint could not be loaded with weights_only=True. We do not fall back."""


class _TupleSafeLoader(yaml.SafeLoader):
    """SafeLoader + `!!python/tuple` only."""


_TupleSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node, deep=True)),
)


def load_config(yaml_path: Path) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_TupleSafeLoader)  # noqa: S506 — SafeLoader subclass


# model kwargs we pass through to the vendored BSRoformer (yaml `model:` section)
_MODEL_KWARGS = (
    "dim", "depth", "stereo", "num_stems", "time_transformer_depth", "freq_transformer_depth",
    "linear_transformer_depth", "freqs_per_bands", "dim_head", "heads", "attn_dropout", "ff_dropout",
    "flash_attn", "dim_freqs_in", "stft_n_fft", "stft_hop_length", "stft_win_length", "stft_normalized",
    "mask_estimator_depth", "multi_stft_resolution_loss_weight", "multi_stft_resolutions_window_sizes",
    "multi_stft_hop_size", "multi_stft_normalized", "mlp_expansion_factor", "use_torch_checkpoint",
    "skip_connection",
)


def build_model(config: dict):
    from escutario.vendor_bs_roformer import BSRoformer
    m = config["model"]
    unknown = set(m) - set(_MODEL_KWARGS)
    if unknown:
        raise ValueError(f"unexpected model config keys (architecture drift?): {sorted(unknown)}")
    return BSRoformer(**{k: m[k] for k in _MODEL_KWARGS if k in m})


def load_state_dict_safely(ckpt_path: Path) -> dict:
    import torch
    try:
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True, mmap=True)
    except Exception as e:  # never fall back to weights_only=False
        raise UnsafeCheckpoint(
            f"torch.load(weights_only=True) refused {ckpt_path}: {type(e).__name__}: {e}. "
            "Not retrying with weights_only=False.") from e
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise UnsafeCheckpoint(f"checkpoint is a {type(obj).__name__}, expected a state dict")
    return obj


def load_model(models_dir: Path = MODELS_DIR, *, download: bool = True):
    """Verified download -> safe yaml -> vendored model -> weights_only load -> strict state dict.

    Returns (model in eval mode on cpu, float32; config dict; instrument list)."""
    import torch
    ckpt_path, yaml_path = ensure_model_files(models_dir, download=download)
    config = load_config(yaml_path)
    instruments = list(config["training"]["instruments"])
    if len(instruments) != config["model"]["num_stems"]:
        raise ValueError("yaml instrument list length != model.num_stems")
    model = build_model(config)
    state = load_state_dict_safely(ckpt_path)
    state = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    return model, config, instruments


# ---------------------------------------------------------------------------------------------
# chunked overlap-add inference (model-agnostic; tested with a fake model)
# ---------------------------------------------------------------------------------------------

def fade_window(chunk: int, fade: int, *, fade_in: bool = True, fade_out: bool = True) -> np.ndarray:
    w = np.ones(chunk, dtype=np.float32)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        if fade_in:
            w[:fade] = ramp
        if fade_out:
            w[-fade:] = ramp[::-1]
    return w


def chunked_apply(fn, mix: np.ndarray, chunk: int, num_overlap: int, n_out: int) -> np.ndarray:
    """Run fn over overlapping chunks of `mix` (shape (c, n)) and overlap-add the results.

    fn(chunk_array (c, chunk)) -> (n_out, c, chunk). Returns (n_out, c, n) float32.
    Edges are reflect-padded by (chunk - step) so every real sample sits under a full window;
    the first/last chunks keep a flat outer edge; results are divided by the summed weights.
    """
    c, n = mix.shape
    if n <= chunk:
        pad_to = max(n, MIN_INPUT_SAMPLES)
        x = np.pad(mix, ((0, 0), (0, pad_to - n)))
        return np.asarray(fn(x), dtype=np.float32)[:, :, :n]

    step = max(1, chunk // max(1, num_overlap))
    border = chunk - step
    mode = "reflect" if n > border else "constant"
    x = np.pad(mix, ((0, 0), (border, border)), mode=mode)
    total = x.shape[1]
    starts = list(range(0, max(total - chunk, 0) + 1, step))
    if starts[-1] + chunk < total:
        starts.append(starts[-1] + step)
    end = starts[-1] + chunk
    if end > total:
        x = np.pad(x, ((0, 0), (0, end - total)))
    fade = min(int(chunk * FADE_FRACTION), border)  # a fade longer than the overlap would leave seams under-weighted

    out = np.zeros((n_out, c, x.shape[1]), dtype=np.float32)
    weight = np.zeros(x.shape[1], dtype=np.float32)
    for i, s0 in enumerate(starts):
        w = fade_window(chunk, fade, fade_in=i > 0, fade_out=i < len(starts) - 1)
        y = np.asarray(fn(x[:, s0:s0 + chunk]), dtype=np.float32)
        out[:, :, s0:s0 + chunk] += y * w
        weight[s0:s0 + chunk] += w
    out /= np.maximum(weight, MIN_WINDOW_SUM)
    return out[:, :, border:border + n]


def looks_like_silent_gpu_failure(x: np.ndarray, out: np.ndarray) -> bool:
    """True if the input chunk has signal but some returned stem is exactly all zeros.

    A real forward pass multiplies a learned mask into the STFT, so even a stem the model deems
    absent comes back tiny but non-zero; exact zeros mean the GPU work never ran."""
    if float(np.max(np.abs(x))) <= SILENT_INPUT_PEAK:
        return False
    flat = out.reshape(out.shape[0], -1)
    return bool(np.any(~flat.any(axis=1)))


def subtract_stem(mix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """mix - target, scaled down as a whole (never clipped) if its peak exceeds PEAK_CEILING."""
    if mix.shape != target.shape:
        raise ValueError(f"shape mismatch: mix {mix.shape} vs target {target.shape}")
    out = (mix.astype(np.float32) - target.astype(np.float32))
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > PEAK_CEILING:
        out *= PEAK_CEILING / peak
    return out


def peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if os.uname().sysname == "Darwin" else int(r) * 1024  # macOS reports bytes, Linux KiB


# ---------------------------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------------------------

class RoformerBackend:
    """MVSep Mega 53-stem BS-Roformer. Duck-types escutario.split.SeparationBackend.

    Only the requested stems are returned. Internally the shared transformer trunk runs in full and
    only the requested stems' mask heads run (upstream `active_stem_ids`); each returned stem is
    identical to what an all-53 forward pass gives for it (checked in tests/test_instrument.py).
    """

    name = BACKEND_NAME

    def __init__(self, stems: list[str] | None = None, device: str | None = None, *,
                 models_dir: Path = MODELS_DIR, download: bool = True,
                 chunk_seconds: float = CHUNK_SECONDS_DEFAULT, num_overlap: int = NUM_OVERLAP_DEFAULT) -> None:
        self.stems = list(stems) if stems is not None else list(DEFAULT_STEMS)
        if not self.stems:
            raise ValueError("stems must not be empty")
        self.requested_device = device
        self.device: str | None = None
        self.models_dir = Path(models_dir)
        self.download = download
        self.chunk = int(round(chunk_seconds * SAMPLE_RATE))
        self.num_overlap = num_overlap
        self.notes: list[str] = []
        self.last_run: dict = {}
        self._model = None
        self._instruments: list[str] | None = None

    # -- loading ---------------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        model, config, instruments = load_model(self.models_dir, download=self.download)
        max_chunk = int(config["inference"]["chunk_size"])
        if self.chunk > max_chunk:
            raise ValueError(f"chunk {self.chunk} exceeds the model's inference chunk_size {max_chunk}")
        unknown = [s for s in self.stems if s not in instruments]
        if unknown:
            raise ValueError(f"unknown stems {unknown}; this model has: {instruments}")
        self._instruments = instruments
        dev = self.requested_device
        if dev is None:
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            model = model.to(dev)
        except Exception as e:  # noqa: BLE001
            self.notes.append(f"could not move model to {dev} ({type(e).__name__}: {e}); using cpu")
            dev = "cpu"
            model = model.to("cpu")
        self.device = dev
        self._model = model

    def _fall_back_to_cpu(self, err: Exception) -> None:
        import torch
        self.notes.append(f"switched from {self.device} to cpu: {type(err).__name__}: {str(err)[:300]}")
        self._model = self._model.to("cpu")
        if self.device == "mps":
            torch.mps.empty_cache()
        self.device = "cpu"

    # -- inference -------------------------------------------------------------------------
    def _run_chunk(self, x: np.ndarray, stem_ids: list[int]) -> np.ndarray:
        import torch
        with torch.inference_mode():
            if self.device != "cpu":
                try:
                    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None].to(self.device)
                    y = self._model(t, active_stem_ids=stem_ids)  # (1, n_stems, 2, chunk)
                    if self.device == "mps":
                        torch.mps.synchronize()
                    out = y[0].float().cpu().numpy()
                    if looks_like_silent_gpu_failure(x, out):
                        raise RuntimeError(
                            "returned exactly all-zero stem(s) for non-silent input — the signature of a "
                            "Metal command buffer that failed (e.g. out of memory) without raising")
                    return out
                except Exception as e:  # noqa: BLE001
                    self._fall_back_to_cpu(e)
            t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
            y = self._model(t, active_stem_ids=stem_ids)
            return y[0].float().numpy()

    def separate_ids(self, stereo_44k: np.ndarray, stems: list[str]) -> dict[str, np.ndarray]:
        import torch
        x = np.asarray(stereo_44k, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] != 2:
            raise ValueError(f"expected stereo shape (2, n), got {x.shape}")
        self._ensure_loaded()
        unknown = [s for s in stems if s not in self._instruments]
        if unknown:
            raise ValueError(f"unknown stems {unknown}; this model has: {self._instruments}")
        stem_ids = [self._instruments.index(s) for s in stems]
        if self.device == "mps" and len(stem_ids) > MPS_MAX_HEADS:
            self._fall_back_to_cpu(RuntimeError(
                f"{len(stem_ids)} stems requested; more than MPS_MAX_HEADS={MPS_MAX_HEADS} is unverified on MPS"))
        t0 = time.perf_counter()
        mps_peak = 0

        def fn(chunk):
            nonlocal mps_peak
            y = self._run_chunk(chunk, stem_ids)
            if self.device == "mps":
                mps_peak = max(mps_peak, int(torch.mps.driver_allocated_memory()))
            return y

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            out = chunked_apply(fn, x, self.chunk, self.num_overlap, len(stem_ids))
        self.last_run = {
            "device": self.device,
            "seconds": round(time.perf_counter() - t0, 2),
            "audio_seconds": round(x.shape[1] / SAMPLE_RATE, 2),
            "peak_rss_bytes": peak_rss_bytes(),
            "mps_driver_peak_bytes": mps_peak or None,
            "chunk_samples": self.chunk,
            "num_overlap": self.num_overlap,
        }
        return {s: out[i] for i, s in enumerate(stems)}

    def separate(self, stereo_44k: np.ndarray) -> dict[str, np.ndarray]:
        """stereo_44k: (2, n) float32 at 44.1 kHz -> {stem: (2, n) float32} for self.stems."""
        return self.separate_ids(stereo_44k, self.stems)

    def minus_target(self, mix_stereo: np.ndarray, target_stem: str = "violin",
                     target: np.ndarray | None = None) -> np.ndarray:
        """Play-along backing track: original mix minus the separated target, peak-safe.

        This model's stems don't sum to the mix, so we subtract from the mix rather than summing
        the other 52 stems. Pass `target` if it was already separated to skip a second run."""
        mix = np.asarray(mix_stereo, dtype=np.float32)
        if target is None:
            target = self.separate_ids(mix, [target_stem])[target_stem]
        return subtract_stem(mix, target)


def minus_target(mix_stereo: np.ndarray, target_stem: str = "violin", *,
                 backend: RoformerBackend | None = None, target: np.ndarray | None = None) -> np.ndarray:
    """Module-level convenience: mix − separated `target_stem` (peak-safe)."""
    if target is not None:
        return subtract_stem(np.asarray(mix_stereo, dtype=np.float32), target)
    backend = backend or RoformerBackend(stems=[target_stem])
    return backend.minus_target(mix_stereo, target_stem)
