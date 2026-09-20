"""Tests for the UNet: shapes, conditioning behavior, and error handling."""

import pytest
import torch

from diffusion.unet import UNet, count_parameters


def tiny_unet(**kwargs):
    defaults = dict(base_channels=8, channel_mults=(1, 2), time_emb_dim=16)
    defaults.update(kwargs)
    return UNet(**defaults)


def test_unet_output_shape_matches_input():
    model = tiny_unet()
    x = torch.randn(2, 1, 28, 28)
    t = torch.tensor([0, 5])
    out = model(x, t)
    assert out.shape == x.shape


def test_unet_output_shape_smaller_resolution():
    model = tiny_unet()
    x = torch.randn(1, 1, 16, 16)
    t = torch.tensor([3])
    assert model(x, t).shape == (1, 1, 16, 16)


def test_unet_timestep_changes_output():
    torch.manual_seed(0)
    model = tiny_unet()
    model.eval()
    x = torch.randn(2, 1, 28, 28)
    out_early = model(x, torch.tensor([1, 1]))
    out_late = model(x, torch.tensor([50, 50]))
    assert not torch.allclose(out_early, out_late)


def test_conditional_embedding_changes_output():
    torch.manual_seed(0)
    model = tiny_unet(num_classes=10)
    model.eval()
    x = torch.randn(2, 1, 28, 28)
    t = torch.tensor([10, 10])
    out_a = model(x, t, torch.tensor([3, 3]))
    out_b = model(x, t, torch.tensor([7, 7]))
    assert out_a.shape == x.shape
    assert not torch.allclose(out_a, out_b)


def test_conditional_model_requires_labels():
    model = tiny_unet(num_classes=10)
    with pytest.raises(ValueError, match="class labels y are required"):
        model(torch.randn(1, 1, 28, 28), torch.tensor([1]))


def test_unconditional_model_rejects_labels():
    model = tiny_unet()
    with pytest.raises(ValueError, match="y must be None"):
        model(torch.randn(1, 1, 28, 28), torch.tensor([1]), torch.tensor([1]))


def test_count_parameters_positive():
    model = tiny_unet()
    n = count_parameters(model)
    assert n > 1000
    # Full demo config should be a few million parameters, not hundreds of millions.
    big = UNet(base_channels=32, channel_mults=(1, 2, 4), time_emb_dim=128)
    assert 500_000 < count_parameters(big) < 50_000_000


def test_unet_gradients_flow():
    model = tiny_unet()
    x = torch.randn(2, 1, 28, 28)
    loss = model(x, torch.tensor([1, 2])).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
