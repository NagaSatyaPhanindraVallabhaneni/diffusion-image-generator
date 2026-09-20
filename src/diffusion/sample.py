"""Sampling utilities: load a checkpoint and generate images from pure noise."""

import io
import json
import os

import torch
from PIL import Image
from torchvision.utils import make_grid

from .ddpm import GaussianDiffusion
from .train import build_model


def load_checkpoint(checkpoint_path, device="cpu"):
    """Load EMA weights + config saved by train.py.

    Expects ``ema.pt`` (state dict) and ``config.json`` side by side.
    Returns (model, diffusion, config) with the model in eval mode.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.json not found next to checkpoint: {config_path}"
        )
    with open(config_path) as f:
        config = json.load(f)
    config["channel_mults"] = tuple(config["channel_mults"])
    model = build_model(config, device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    diffusion = GaussianDiffusion(
        timesteps=config["timesteps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        schedule=config["schedule"],
    )
    return model, diffusion, config


@torch.no_grad()
def generate_images(model, diffusion, n_images, digit=None, seed=0, device="cpu"):
    """Generate n_images from pure noise. Returns a tensor in [-1, 1]."""
    if digit is not None and model.num_classes is None:
        raise ValueError("digit requested but the model was trained unconditionally")
    if digit is not None and not (0 <= digit < model.num_classes):
        raise ValueError(f"digit must be in [0, {model.num_classes - 1}]")
    torch.manual_seed(seed)
    y = (
        torch.full((n_images,), digit, dtype=torch.long, device=device)
        if digit is not None
        else None
    )
    images, _ = diffusion.p_sample_loop(
        model, (n_images, 1, 28, 28), y=y, device=device
    )
    return images


def denormalize(images):
    """[-1, 1] -> [0, 1] for display."""
    return images.clamp(-1, 1).add(1).div(2)


def save_image_grid(images, path, nrow=None):
    """Save a grid of [-1, 1] images as a PNG."""
    nrow = nrow or int(images.shape[0] ** 0.5)
    grid = make_grid(denormalize(images).cpu(), nrow=nrow, padding=2, pad_value=1.0)
    array = (grid.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    Image.fromarray(array[:, :, 0] if array.shape[2] == 1 else array).save(path)
    return path


def grid_png_bytes(images, nrow=None):
    """Render a grid of [-1, 1] images to PNG bytes (for the API)."""
    nrow = nrow or int(images.shape[0] ** 0.5)
    grid = make_grid(denormalize(images).cpu(), nrow=nrow, padding=2, pad_value=1.0)
    array = (grid.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    img = Image.fromarray(array[:, :, 0] if array.shape[2] == 1 else array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
