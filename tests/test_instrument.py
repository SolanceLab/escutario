# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Tests for escutario.instrument — no network, no 1.4 GB model unless ESCUTARIO_REAL_MODEL=1."""

import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from escutario import instrument as ins


# ----------------------------------------------------------------------------- downloader

class _FakeResp(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status


def _asset(data: bytes, name="m.bin", size=True):
    return ins.ModelAsset(filename=name, url=ins.ALLOWED_DOWNLOAD_PREFIX + "v0/" + name,
                          sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data) if size else None)


def _opener_serving(data: bytes, drop_after: int | None = None, calls=None):
    def opener(req, timeout=None):
        rng = req.get_header("Range")
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        chunk = data[start:] if drop_after is None else data[start:start + drop_after]
        if calls is not None:
            calls.append(rng)
        return _FakeResp(chunk, status=206 if rng else 200)
    return opener


def test_download_verifies_and_places_file(tmp_path):
    data = os.urandom(50_000)
    p = ins.ensure_asset(_asset(data), tmp_path, opener=_opener_serving(data))
    assert p.read_bytes() == data
    assert list(tmp_path.glob("*.part")) == []


def test_download_resumes_after_truncation(tmp_path):
    data = os.urandom(30_000)
    calls = []
    p = ins.ensure_asset(_asset(data), tmp_path, opener=_opener_serving(data, drop_after=7_000, calls=calls))
    assert p.read_bytes() == data
    assert calls[0] is None and calls[1] == "bytes=7000-"


def test_download_hash_mismatch_is_refused_and_discarded(tmp_path):
    good, evil = os.urandom(1000), os.urandom(1000)
    with pytest.raises(ins.ChecksumMismatch):
        ins.ensure_asset(_asset(good), tmp_path, opener=_opener_serving(evil))
    assert list(tmp_path.iterdir()) == []


def test_existing_mismatching_file_is_never_overwritten(tmp_path):
    good = os.urandom(1000)
    a = _asset(good)
    (tmp_path / a.filename).write_bytes(b"tampered")
    called = []
    with pytest.raises(ins.ChecksumMismatch):
        ins.ensure_asset(a, tmp_path, opener=lambda *a, **k: called.append(1))
    assert (tmp_path / a.filename).read_bytes() == b"tampered"
    assert called == []


def test_existing_good_file_is_used_without_network(tmp_path):
    good = os.urandom(1000)
    a = _asset(good)
    (tmp_path / a.filename).write_bytes(good)
    assert ins.ensure_asset(a, tmp_path, opener=lambda *a, **k: pytest.fail("network used")) == tmp_path / a.filename


def test_non_official_url_refused(tmp_path):
    a = ins.ModelAsset("m.ckpt", "https://huggingface.co/someone/mirror/m.ckpt", "0" * 64)
    with pytest.raises(ValueError):
        ins.ensure_asset(a, tmp_path, opener=lambda *a, **k: pytest.fail("network used"))


def test_missing_without_download_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ins.ensure_asset(_asset(b"x"), tmp_path, download=False)


def test_pinned_hashes_are_the_decided_ones():
    assert ins.CKPT_ASSET.sha256 == "c62820893bbf86d4e734f966bd142d9157cfc8bb8e79e9d8f9ea553f3ff3519f"
    assert ins.YAML_ASSET.sha256 == "7e198062a251587088adb91215a4f44ab59e67bd62fcc805cf54d6e7dfc51103"
    for a in (ins.CKPT_ASSET, ins.YAML_ASSET):
        assert a.url.startswith("https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.21/")


# ----------------------------------------------------------------------------- yaml + checkpoint safety

def test_yaml_loader_allows_tuple_but_not_python_objects(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: !!python/tuple [1, 2]\n")
    assert ins.load_config(p) == {"a": (1, 2)}
    p.write_text("a: !!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(yaml.constructor.ConstructorError):
        ins.load_config(p)


def test_pickle_with_code_is_refused_without_fallback(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo should-never-run",))

    p = tmp_path / "evil.ckpt"
    torch.save({"x": Evil()}, str(p))
    seen = []
    real_load = torch.load

    def spy(*a, **k):
        seen.append(k.get("weights_only"))
        return real_load(*a, **k)

    monkeypatch.setattr(torch, "load", spy)
    with pytest.raises(ins.UnsafeCheckpoint):
        ins.load_state_dict_safely(p)
    assert seen == [True]  # exactly one attempt, weights_only=True


# ----------------------------------------------------------------------------- chunked inference

@pytest.mark.parametrize("n", [1000, 44100, 100_000, 123_457])
@pytest.mark.parametrize("overlap", [1, 2, 4])
def test_chunked_apply_identity_reconstructs(n, overlap):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, n)).astype(np.float32)
    out = ins.chunked_apply(lambda c: np.stack([c, 2 * c]), x, chunk=20_000, num_overlap=overlap, n_out=2)
    assert out.shape == (2, 2, n)
    np.testing.assert_allclose(out[0], x, atol=1e-4)
    np.testing.assert_allclose(out[1], 2 * x, atol=2e-4)


def test_chunked_apply_passes_fixed_size_chunks():
    sizes = []

    def fn(c):
        sizes.append(c.shape)
        return c[None]

    ins.chunked_apply(fn, np.zeros((2, 95_001), np.float32), chunk=10_000, num_overlap=2, n_out=1)
    assert set(sizes) == {(2, 10_000)}


def test_fade_window_edges():
    w = ins.fade_window(100, 10, fade_in=False, fade_out=True)
    assert w[0] == 1 and w[-1] == 0 and w[50] == 1


# ----------------------------------------------------------------------------- minus / peak safety

def test_subtract_stem_exact_when_quiet():
    rng = np.random.default_rng(1)
    mix = 0.3 * rng.standard_normal((2, 1000)).astype(np.float32).clip(-1, 1)
    tgt = 0.5 * mix
    np.testing.assert_allclose(ins.subtract_stem(mix, tgt), mix - tgt, atol=1e-7)


def test_subtract_stem_is_peak_safe_without_clipping():
    mix = np.array([[0.9, -0.9, 0.1], [0.0, 0.0, 0.0]], np.float32)
    tgt = np.array([[-0.9, 0.9, 0.0], [0.0, 0.0, 0.0]], np.float32)
    out = ins.subtract_stem(mix, tgt)
    assert np.max(np.abs(out)) == pytest.approx(ins.PEAK_CEILING)
    raw = mix - tgt
    np.testing.assert_allclose(out, raw * (ins.PEAK_CEILING / 1.8), atol=1e-6)  # scaled, shape preserved


def test_subtract_stem_shape_mismatch():
    with pytest.raises(ValueError):
        ins.subtract_stem(np.zeros((2, 10)), np.zeros((2, 9)))


def test_minus_target_uses_given_target_without_model():
    mix = np.full((2, 50), 0.4, np.float32)
    out = ins.minus_target(mix, "violin", target=np.full((2, 50), 0.1, np.float32))
    np.testing.assert_allclose(out, 0.3, atol=1e-6)


# ----------------------------------------------------------------------------- backend surface (no model)

def test_backend_surface_matches_split_protocol_without_loading(tmp_path):
    b = ins.RoformerBackend(models_dir=tmp_path, download=False)
    assert b.name == "mvsep_mega_53"
    assert b.stems == ["violin", "viola", "cello", "bowed_strings", "strings"]
    assert callable(b.separate)
    assert ins.RoformerBackend(stems=["violin"], models_dir=tmp_path, download=False).stems == ["violin"]
    with pytest.raises(ValueError):
        ins.RoformerBackend(stems=[], models_dir=tmp_path)
    with pytest.raises(ValueError):
        b.separate(np.zeros((1, 100), np.float32))  # mono rejected before any load
    with pytest.raises(FileNotFoundError):
        b.separate(np.zeros((2, 100), np.float32))  # no files, download=False -> no network


def test_backend_config_stems_include_violin_family():
    y = ins.MODELS_DIR / ins.YAML_ASSET.filename
    if not y.exists():
        pytest.skip("model yaml not downloaded")
    cfg = ins.load_config(y)
    inst = cfg["training"]["instruments"]
    assert len(inst) == cfg["model"]["num_stems"] == 53
    assert set(ins.DEFAULT_STEMS) <= set(inst)


# ----------------------------------------------------------------------------- real model (opt-in)

REAL = os.environ.get("ESCUTARIO_REAL_MODEL") == "1"
real_only = pytest.mark.skipif(not REAL, reason="set ESCUTARIO_REAL_MODEL=1 (loads the 1.4 GB checkpoint)")


@real_only
def test_real_checkpoint_loads_strict_and_active_heads_match_full_pass():
    torch = pytest.importorskip("torch")
    model, cfg, inst = ins.load_model(download=False)  # strict=True inside
    rng = np.random.default_rng(3)
    x = torch.from_numpy((0.1 * rng.standard_normal((1, 2, 44100))).astype(np.float32))
    ids = [inst.index(s) for s in ins.DEFAULT_STEMS]
    with torch.inference_mode():
        full = model(x)
        some = model(x, active_stem_ids=ids)
    assert full.shape == (1, 53, 2, 44100)
    torch.testing.assert_close(some, full[:, ids], rtol=1e-4, atol=1e-5)


# ----------------------------------------------------------------------------- silent GPU failure guard

def test_silent_gpu_failure_detector():
    x = np.full((2, 1000), 0.2, np.float32)
    quiet_but_real = np.full((3, 2, 1000), 1e-7, np.float32)
    assert not ins.looks_like_silent_gpu_failure(x, quiet_but_real)
    one_zero = quiet_but_real.copy(); one_zero[1] = 0
    assert ins.looks_like_silent_gpu_failure(x, one_zero)
    assert not ins.looks_like_silent_gpu_failure(np.zeros((2, 1000), np.float32), np.zeros((3, 2, 1000), np.float32))


def _fake_backend(device, zero_on_gpu=True):
    torch = pytest.importorskip("torch")

    class Fake(torch.nn.Module):
        def forward(self, t, active_stem_ids=None):
            y = t[:, None].repeat(1, len(active_stem_ids), 1, 1) * 0.5
            return torch.zeros_like(y) if (zero_on_gpu and t.device.type != "cpu") else y

    b = ins.RoformerBackend(stems=["violin", "viola"], download=False)
    b._model, b._instruments, b.device = Fake().to(device), ["violin", "viola", "cello"], device
    return b


@pytest.mark.skipif(not __import__("torch").backends.mps.is_available(), reason="needs MPS")
def test_mps_all_zero_output_falls_back_to_cpu_and_recomputes():
    b = _fake_backend("mps")
    x = np.random.default_rng(0).uniform(-0.5, 0.5, (2, 50_000)).astype(np.float32)
    out = b.separate(x)
    assert b.device == "cpu" and b.last_run["device"] == "cpu"
    assert any("all-zero" in n for n in b.notes)
    np.testing.assert_allclose(out["violin"], 0.5 * x, atol=1e-5)


@pytest.mark.skipif(not __import__("torch").backends.mps.is_available(), reason="needs MPS")
def test_too_many_heads_on_mps_uses_cpu():
    b = _fake_backend("mps", zero_on_gpu=False)
    b._instruments = [f"s{i}" for i in range(ins.MPS_MAX_HEADS + 1)]
    b.stems = list(b._instruments)
    b.separate(np.full((2, 5000), 0.1, np.float32))
    assert b.device == "cpu" and any("MPS_MAX_HEADS" in n for n in b.notes)


def test_unknown_stem_rejected_after_load():
    b = _fake_backend("cpu")
    with pytest.raises(ValueError):
        b.separate_ids(np.zeros((2, 5000), np.float32), ["theremin"])
