"""Training loop for the DDPM: noise-prediction MSE + EMA of weights + checkpointing."""

import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .ddpm import GaussianDiffusion
from .unet import UNet, count_parameters

# torchvision's bundled MNIST mirrors are dead as of 2026; prepend a working one
# so `datasets.MNIST(..., download=True)` keeps working out of the box.
try:
    from torchvision.datasets import MNIST as _MNIST

    _WORKING_MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    if _WORKING_MIRROR not in _MNIST.mirrors:
        _MNIST.mirrors = [_WORKING_MIRROR, *_MNIST.mirrors]
except Exception:
    pass

DEFAULTS = {
    "timesteps": 200,
    "schedule": "linear",
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "base_channels": 16,
    "channel_mults": (1, 2, 4),
    "time_emb_dim": 64,
    "conditional": False,
    "num_classes": 10,
    "batch_size": 128,
    "epochs": 10,
    "lr": 2e-4,
    # NOTE: 0.999 is the literature standard for 100k+ step runs, but on a short
    # demo run (a few hundred steps) it leaves the EMA dominated by the random
    # initialization (0.999^470 ~= 0.62). 0.99 is the sane default here.
    "ema_decay": 0.99,
    "subset": 6000,
    "num_workers": 0,
    "seed": 0,
    "out_dir": "models",
    "data_dir": "data",
}


class EMA:
    """Exponential moving average of model weights (standard diffusion practice)."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.clone().detach() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for name, param in model.state_dict().items():
            if param.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(
                    param.detach(), alpha=1.0 - self.decay
                )

    def state_dict(self):
        return self.shadow

    def copy_to(self, model):
        model.load_state_dict(self.shadow)

    def load_state_dict(self, state):
        self.shadow = {k: v.clone().detach() for k, v in state.items()}


def build_model(config, device="cpu"):
    model = UNet(
        in_channels=1,
        base_channels=config["base_channels"],
        channel_mults=tuple(config["channel_mults"]),
        time_emb_dim=config["time_emb_dim"],
        num_classes=config["num_classes"] if config["conditional"] else None,
    ).to(device)
    return model


def train_step(model, diffusion, optimizer, x, y, device):
    """One gradient step of the DDPM objective. Returns the scalar loss."""
    x = x.to(device)
    y = y.to(device) if y is not None else None
    b = x.shape[0]
    t = torch.randint(0, diffusion.timesteps, (b,), device=device).long()
    noise = torch.randn_like(x)
    x_noisy = diffusion.q_sample(x, t, noise=noise)
    pred_noise = model(x_noisy, t, y)
    loss = F.mse_loss(pred_noise, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train(config=None, progress=True, resume_from=None, reset_ema=False):
    """Train the diffusion model. Returns a history dict with losses and wall time.

    Args:
        config: overrides for DEFAULTS.
        progress: print per-epoch losses.
        resume_from: path to a ``last.pt`` checkpoint to continue training from.
        reset_ema: on resume, re-seed the EMA from the loaded weights instead of
            continuing the saved EMA shadow (useful when changing ema_decay).
    """
    cfg = {**DEFAULTS, **(config or {})}
    torch.manual_seed(cfg["seed"])
    device = "cpu"
    started = time.time()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # -> [-1, 1]
    )
    dataset = datasets.MNIST(cfg["data_dir"], train=True, download=True, transform=transform)
    if cfg["subset"] and cfg["subset"] < len(dataset):
        dataset = Subset(dataset, range(cfg["subset"]))
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )

    model = build_model(cfg, device)
    diffusion = GaussianDiffusion(
        timesteps=cfg["timesteps"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        schedule=cfg["schedule"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    ema = EMA(model, decay=cfg["ema_decay"])

    os.makedirs(cfg["out_dir"], exist_ok=True)
    print(f"UNet parameters: {count_parameters(model):,} | steps/epoch: {len(loader)}")

    start_epoch = 0
    epoch_losses = []
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if reset_ema:
            ema = EMA(model, decay=cfg["ema_decay"])
            print("EMA re-seeded from resumed weights")
        else:
            ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from {resume_from} at epoch {start_epoch + 1}")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        total, count = 0.0, 0
        for x, y in loader:
            y = y if cfg["conditional"] else None
            total += train_step(model, diffusion, optimizer, x, y, device)
            count += 1
            ema.update(model)
        avg = total / max(count, 1)
        epoch_losses.append(avg)
        if progress:
            print(f"epoch {epoch + 1}/{cfg['epochs']}  loss={avg:.5f}")
        # Keep a resume-able checkpoint plus the EMA weights the API/sampler use.
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            os.path.join(cfg["out_dir"], "last.pt"),
        )
        torch.save(ema.state_dict(), os.path.join(cfg["out_dir"], "ema.pt"))

    serializable = {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in cfg.items()
        if k not in ("out_dir", "data_dir")
    }
    with open(os.path.join(cfg["out_dir"], "config.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    wall_time = time.time() - started
    history = {
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "wall_time_s": wall_time,
        "params": count_parameters(model),
        "config": serializable,
    }
    print(f"done in {wall_time / 60:.1f} min | final loss {history['final_loss']:.5f}")
    return history


def dry_run():
    """Smoke test: 2 training steps on random tensors, no MNIST download."""
    print("dry run: 2 steps on synthetic tensors (no dataset download)")
    cfg = {**DEFAULTS, "timesteps": 8}
    device = "cpu"
    model = build_model(cfg, device)
    diffusion = GaussianDiffusion(timesteps=cfg["timesteps"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    for step in range(2):
        x = torch.randn(4, 1, 28, 28)
        loss = train_step(model, diffusion, optimizer, x, None, device)
        print(f"  step {step + 1}: loss={loss:.5f}")
    print("dry run OK")
