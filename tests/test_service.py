"""Tests for sampling, EMA math, checkpoint I/O, and the FastAPI service."""

import json
import os

import pytest
import torch
from fastapi.testclient import TestClient

from diffusion import app as app_module
from diffusion.sample import denormalize, generate_images, load_checkpoint, save_image_grid
from diffusion.train import EMA, build_model
from diffusion.unet import UNet


@pytest.fixture()
def tiny_conditional_ckpt(tmp_path):
    """A real (untrained) tiny conditional checkpoint: ema.pt + config.json."""
    torch.manual_seed(0)
    config = {
        "timesteps": 6,
        "schedule": "linear",
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "base_channels": 8,
        "channel_mults": [1, 2],
        "time_emb_dim": 16,
        "conditional": True,
        "num_classes": 10,
    }
    model = build_model({**config, "channel_mults": tuple(config["channel_mults"])})
    ema_path = tmp_path / "ema.pt"
    torch.save(EMA(model, decay=0.9).state_dict(), ema_path)
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)
    return str(ema_path)


def test_ema_update_moves_toward_new_weights():
    torch.manual_seed(0)
    model = UNet(base_channels=8, channel_mults=(1, 2), time_emb_dim=16)
    ema = EMA(model, decay=0.5)
    before = {k: v.clone() for k, v in ema.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    for k in before:
        expected = 0.5 * before[k] + 0.5 * model.state_dict()[k]
        assert torch.allclose(ema.state_dict()[k], expected, atol=1e-6)


def test_load_checkpoint_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_checkpoint(str(tmp_path / "nope.pt"))


def test_load_checkpoint_roundtrip(tiny_conditional_ckpt):
    model, diffusion, config = load_checkpoint(tiny_conditional_ckpt)
    assert config["conditional"] is True
    assert diffusion.timesteps == 6
    assert model.num_classes == 10
    model.eval()


def test_generate_images_shape_and_range(tiny_conditional_ckpt):
    model, diffusion, _ = load_checkpoint(tiny_conditional_ckpt)
    imgs = generate_images(model, diffusion, 4, digit=3, seed=0)
    assert imgs.shape == (4, 1, 28, 28)
    assert torch.isfinite(imgs).all()


def test_generate_images_rejects_bad_digit(tiny_conditional_ckpt):
    model, diffusion, _ = load_checkpoint(tiny_conditional_ckpt)
    with pytest.raises(ValueError, match="digit must be in"):
        generate_images(model, diffusion, 2, digit=12)


def test_generate_images_reproducible_with_seed(tiny_conditional_ckpt):
    model, diffusion, _ = load_checkpoint(tiny_conditional_ckpt)
    a = generate_images(model, diffusion, 2, digit=1, seed=42)
    b = generate_images(model, diffusion, 2, digit=1, seed=42)
    assert torch.equal(a, b)


def test_denormalize_range():
    x = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(denormalize(x), torch.tensor([0.0, 0.5, 1.0]))


def test_save_image_grid_writes_png(tmp_path):
    imgs = torch.randn(4, 1, 28, 28).clamp(-1, 1)
    path = str(tmp_path / "grid.png")
    save_image_grid(imgs, path)
    assert os.path.exists(path)
    with open(path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


def test_api_health():
    client = TestClient(app_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_api_generate_without_checkpoint_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "CHECKPOINT_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.setattr(app_module, "_model_cache", None)
    client = TestClient(app_module.app)
    resp = client.post("/generate", json={"n_images": 2})
    assert resp.status_code == 503
    assert "Train the model first" in resp.json()["detail"]


def test_api_generate_validation_errors():
    client = TestClient(app_module.app)
    assert client.post("/generate", json={"n_images": 0}).status_code == 422
    assert client.post("/generate", json={"n_images": 99}).status_code == 422
    assert client.post("/generate", json={"digit": 10}).status_code == 422
    assert client.post("/generate", json={"digit": -1}).status_code == 422


def test_api_generate_returns_png(monkeypatch, tiny_conditional_ckpt):
    monkeypatch.setattr(app_module, "CHECKPOINT_PATH", tiny_conditional_ckpt)
    monkeypatch.setattr(app_module, "_model_cache", None)
    client = TestClient(app_module.app)
    resp = client.post("/generate", json={"n_images": 4, "digit": 5, "seed": 1})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"
