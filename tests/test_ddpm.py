"""Tests for the DDPM math: schedules, closed-form noising, reverse steps."""

import pytest
import torch

from diffusion.ddpm import (
    GaussianDiffusion,
    cosine_beta_schedule,
    extract,
    linear_beta_schedule,
    sinusoidal_timestep_embedding,
)


def test_linear_schedule_endpoints():
    betas = linear_beta_schedule(100, beta_start=1e-4, beta_end=0.02)
    assert betas.shape == (100,)
    assert betas[0].item() == pytest.approx(1e-4)
    assert betas[-1].item() == pytest.approx(0.02)
    assert torch.all(betas > 0)


def test_cosine_schedule_shape_and_bounds():
    betas = cosine_beta_schedule(100)
    assert betas.shape == (100,)
    assert torch.all(betas >= 0) and torch.all(betas < 1)


def test_invalid_schedule_raises():
    with pytest.raises(ValueError, match="unknown beta schedule"):
        GaussianDiffusion(timesteps=10, schedule="quadratic")


def test_timestep_embedding_shape():
    t = torch.tensor([0, 5, 99])
    emb = sinusoidal_timestep_embedding(t, 64)
    assert emb.shape == (3, 64)


def test_timestep_embedding_deterministic_and_distinct():
    t = torch.tensor([7, 7, 42])
    emb = sinusoidal_timestep_embedding(t, 32)
    assert torch.equal(emb[0], emb[1])
    assert not torch.equal(emb[0], emb[2])


def test_timestep_embedding_odd_dim_raises():
    with pytest.raises(ValueError, match="must be even"):
        sinusoidal_timestep_embedding(torch.tensor([1]), 33)


def test_extract_broadcast_shape():
    schedule = torch.arange(10).float()
    t = torch.tensor([2, 5])
    out = extract(schedule, t, (2, 3, 4, 4))
    assert out.shape == (2, 1, 1, 1)
    assert out[0].item() == pytest.approx(2.0)
    assert out[1].item() == pytest.approx(5.0)


def test_q_sample_matches_closed_form():
    torch.manual_seed(0)
    diffusion = GaussianDiffusion(timesteps=50)
    x0 = torch.randn(4, 1, 8, 8)
    t = torch.tensor([0, 10, 25, 49])
    noise = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, t, noise=noise)
    # Manual computation of sqrt(a_bar_t) * x0 + sqrt(1 - a_bar_t) * noise.
    ab = diffusion.alphas_cumprod[t].view(4, 1, 1, 1)
    expected = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
    assert torch.allclose(x_t, expected, atol=1e-6)


def test_q_sample_at_t0_with_zero_noise_scales_by_sqrt_ab0():
    diffusion = GaussianDiffusion(timesteps=50)
    x0 = torch.randn(2, 1, 8, 8)
    t = torch.zeros(2, dtype=torch.long)
    x_t = diffusion.q_sample(x0, t, noise=torch.zeros_like(x0))
    # With zero noise, q_sample is exactly sqrt(a_bar_0) * x_0.
    expected = diffusion.sqrt_alphas_cumprod[0] * x0
    assert torch.allclose(x_t, expected, atol=1e-6)


def test_q_sample_pure_noise_at_high_t():
    torch.manual_seed(1)
    diffusion = GaussianDiffusion(timesteps=1000, beta_start=1e-4, beta_end=0.02)
    x0 = torch.randn(64, 1, 8, 8)
    t = torch.full((64,), 999)
    x_t = diffusion.q_sample(x0, t)
    # At t=T the signal is almost entirely destroyed: correlation with x0 ~ 0.
    corr = (x_t * x0).mean().item()
    assert abs(corr) < 0.05


def test_predict_start_from_noise_roundtrip():
    torch.manual_seed(2)
    diffusion = GaussianDiffusion(timesteps=50)
    x0 = torch.randn(3, 1, 8, 8)
    t = torch.tensor([5, 20, 40])
    noise = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, t, noise=noise)
    x0_hat = diffusion.predict_start_from_noise(x_t, t, noise)
    assert torch.allclose(x0_hat, x0, atol=1e-4)


def test_p_sample_loop_shapes_and_snapshots():
    from diffusion.unet import UNet

    torch.manual_seed(3)
    diffusion = GaussianDiffusion(timesteps=10)
    model = UNet(base_channels=8, channel_mults=(1, 2), time_emb_dim=16)
    model.eval()
    final, snaps = diffusion.p_sample_loop(
        model, (2, 1, 8, 8), record_at=[9, 5, 0]
    )
    assert final.shape == (2, 1, 8, 8)
    assert set(snaps.keys()) == {9, 5, 0}
    for img in snaps.values():
        assert img.shape == (2, 1, 8, 8)
    assert torch.isfinite(final).all()


def test_posterior_variance_positive():
    diffusion = GaussianDiffusion(timesteps=100)
    assert torch.all(diffusion.posterior_variance[1:] > 0)
    assert diffusion.alphas_cumprod[-1].item() < 1.0
    assert diffusion.alphas_cumprod[0].item() > 0.9


def test_alphas_cumprod_monotonic_decreasing():
    diffusion = GaussianDiffusion(timesteps=100)
    diffs = diffusion.alphas_cumprod[1:] - diffusion.alphas_cumprod[:-1]
    assert torch.all(diffs < 0)
