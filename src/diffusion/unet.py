"""A small UNet for noise prediction, built from scratch.

Time information enters through sinusoidal timestep embeddings (plus an
optional learned class embedding for conditional generation), injected into
every residual block. The network predicts the noise eps added at timestep t,
which the diffusion process uses to denoise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddpm import sinusoidal_timestep_embedding


class ResBlock(nn.Module):
    """Two convs with GroupNorm + SiLU and additive time-embedding conditioning."""

    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(temb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.residual(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Noise-prediction UNet: forward(x_t, t, y=None) -> predicted noise, same shape as x_t."""

    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
        num_classes=None,
        dropout=0.0,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels = []
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResBlock(ch, out_ch, time_emb_dim, dropout),
                        ResBlock(out_ch, out_ch, time_emb_dim, dropout),
                    ]
                )
            )
            skip_channels.append(out_ch)
            ch = out_ch
            if i < len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))

        self.mid = nn.ModuleList(
            [
                ResBlock(ch, ch, time_emb_dim, dropout),
                ResBlock(ch, ch, time_emb_dim, dropout),
            ]
        )

        # Upsampling path mirrors the down path; one upsample per downsample.
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for j, i in enumerate(reversed(range(len(channel_mults)))):
            skip_ch = skip_channels[i]
            if j > 0:
                self.upsamples.append(Upsample(ch))
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResBlock(ch + skip_ch, skip_ch, time_emb_dim, dropout),
                        ResBlock(skip_ch, skip_ch, time_emb_dim, dropout),
                    ]
                )
            )
            ch = skip_ch

        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
        )

    def forward(self, x, t, y=None):
        if self.num_classes is not None and y is None:
            raise ValueError("this is a conditional model: class labels y are required")
        if self.num_classes is None and y is not None:
            raise ValueError("this is an unconditional model: y must be None")
        temb = self.time_mlp(sinusoidal_timestep_embedding(t, self.time_emb_dim))
        if self.num_classes is not None:
            temb = temb + self.class_emb(y)

        h = self.input_conv(x)
        residuals = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, temb)
            residuals.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)
        for block in self.mid:
            h = block(h, temb)
        for j, i in enumerate(reversed(range(len(self.down_blocks)))):
            if j > 0:
                h = self.upsamples[j - 1](h)
            h = torch.cat([h, residuals[i]], dim=1)
            for block in self.up_blocks[j]:
                h = block(h, temb)
        return self.out(h)


def count_parameters(model):
    """Total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
