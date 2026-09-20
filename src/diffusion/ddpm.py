"""Denoising Diffusion Probabilistic Models (DDPM), implemented from scratch.

Reference: Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models"
(NeurIPS 2020). No high-level diffusion library is used anywhere here —
every schedule, closed-form posterior, and sampling step is derived below.

Forward process:  q(x_t | x_0) = N(sqrt(a_bar_t) * x_0, (1 - a_bar_t) * I)
Reverse process:  p(x_{t-1} | x_t) = N(mu_theta(x_t, t), sigma_t^2 * I),
where the network predicts the noise eps added in the forward process.
"""

import math

import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """The original DDPM linear variance schedule."""
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule from Nichol & Dhariwal (2021): gentler early noising."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0, 0.999)


def sinusoidal_timestep_embedding(timesteps, dim):
    """Sinusoidal embedding of integer timesteps (Transformer-style).

    Args:
        timesteps: LongTensor of shape [N].
        dim: embedding dimension (must be even).

    Returns:
        FloatTensor of shape [N, dim].
    """
    if dim % 2 == 1:
        raise ValueError("timestep embedding dim must be even")
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / (half - 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


def extract(schedule, t, x_shape):
    """Index per-timestep scalars from a [T] schedule and reshape for broadcasting.

    Returns a tensor shaped [N, 1, 1, ...] matching the rank of x_shape.
    """
    out = schedule.to(t.device).gather(-1, t)
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


class GaussianDiffusion:
    """Closed-form forward diffusion and learned reverse (denoising) sampling."""

    def __init__(self, timesteps=200, beta_start=1e-4, beta_end=0.02, schedule="linear"):
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule: {schedule!r}")

        self.timesteps = timesteps
        self.schedule = schedule
        self.betas = betas

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_recip_alphas_cumprod = 1.0 / self.sqrt_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)
        # q(x_{t-1} | x_t, x_0) variance, Ho et al. eq. (7)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # Ho et al. eq. (11): mu = 1/sqrt(a_t) * (x_t - b_t / sqrt(1 - a_bar_t) * eps)
        self.mu_coef_xt = self.sqrt_recip_alphas
        self.mu_coef_eps = betas / self.sqrt_one_minus_alphas_cumprod

    def q_sample(self, x_start, t, noise=None):
        """Diffuse x_0 to x_t in closed form (no loop over t needed)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, pred_noise):
        """Recover the predicted x_0 from x_t and predicted noise."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * pred_noise
        )

    def p_mean_variance(self, model, x_t, t, y=None):
        """Reverse-process mean and variance: p(x_{t-1} | x_t)."""
        pred_noise = model(x_t, t, y)
        model_mean = (
            extract(self.mu_coef_xt, t, x_t.shape) * x_t
            - extract(self.mu_coef_eps, t, x_t.shape) * pred_noise
        )
        model_variance = extract(self.posterior_variance, t, x_t.shape)
        return model_mean, model_variance

    @torch.no_grad()
    def p_sample(self, model, x_t, t, y=None):
        """One reverse step: x_t -> x_{t-1}."""
        mean, variance = self.p_mean_variance(model, x_t, t, y)
        noise = torch.randn_like(x_t)
        # No noise is added at the final step (t = 0).
        nonzero_mask = (t != 0).float().view(t.shape[0], *([1] * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * torch.sqrt(variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, y=None, device="cpu", record_at=None):
        """Sample from pure noise, running the full reverse chain.

        Args:
            model: noise-predicting network with signature model(x, t, y).
            shape: (N, C, H, W) tensor shape to generate.
            y: optional class labels LongTensor [N] for conditional models.
            record_at: optional iterable of timesteps at which to snapshot x_t.

        Returns:
            (final_images, snapshots) where snapshots maps t -> tensor at that step.
        """
        record_at = set(record_at) if record_at else set()
        img = torch.randn(shape, device=device)
        snapshots = {}
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, y)
            if i in record_at:
                snapshots[i] = img.detach().cpu().clone()
        return img, snapshots
