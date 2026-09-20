"""The money shot: render the denoising trajectory from pure noise to a clean image."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .sample import denormalize


@torch.no_grad()
def render_denoising_trajectory(
    model, diffusion, digit=None, seed=0, num_frames=8, device="cpu", save_path=None
):
    """Sample once while snapshotting x_t at ``num_frames`` timesteps.

    Returns the path of the saved PNG (a row of frames from t=T noise to t=0).
    """
    torch.manual_seed(seed)
    y = (
        torch.tensor([digit], dtype=torch.long, device=device)
        if digit is not None
        else None
    )
    timesteps = diffusion.timesteps
    record_at = sorted(
        {round(timesteps - 1 - i * (timesteps - 1) / (num_frames - 1)) for i in range(num_frames)}
    )
    _, snapshots = diffusion.p_sample_loop(
        model, (1, 1, 28, 28), y=y, device=device, record_at=record_at
    )
    frames = [snapshots[t] for t in sorted(record_at, reverse=True)]

    fig, axes = plt.subplots(1, num_frames, figsize=(2.2 * num_frames, 2.6))
    for ax, frame, t in zip(axes, frames, sorted(record_at, reverse=True)):
        img = denormalize(frame)[0, 0].cpu().numpy()
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={t}", fontsize=11, color="#9fb3c8")
        ax.axis("off")
    title = "Denoising trajectory" + (f" — digit {digit}" if digit is not None else "")
    fig.suptitle(title, fontsize=14, color="#e6edf3", y=0.98)
    fig.patch.set_facecolor("#0d1117")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return save_path


@torch.no_grad()
def render_sample_grid(
    model, diffusion, n_images=16, digit=None, seed=0, device="cpu", save_path=None
):
    """Generate n_images and render them as a labeled grid PNG."""
    from .sample import generate_images

    images = generate_images(model, diffusion, n_images, digit=digit, seed=seed, device=device)
    nrow = int(n_images**0.5)
    fig, axes = plt.subplots(nrow, nrow, figsize=(nrow * 1.8, nrow * 1.8))
    for ax, img in zip(axes.flat, denormalize(images)):
        ax.imshow(img[0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    title = "Generated samples" + (f" — digit {digit}" if digit is not None else "")
    fig.suptitle(title, fontsize=14, color="#e6edf3")
    fig.patch.set_facecolor("#0d1117")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return save_path
