"""Dual-encoder 3D DeCNN from Jiang et al., Fig. 1."""
import torch
from torch import nn
from torch.nn import functional as F


WIDTHS = (32, 64, 128, 256, 512)


def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
    )


def pool2(x):
    if not (x.is_cuda and torch.are_deterministic_algorithms_enabled() and x.requires_grad):
        return F.max_pool3d(x, 2)
    # Stride-two windows never overlap, so a reduction over the eight values
    # gives the same pooling rule without the CUDA atomic backward kernel.
    b, c, z, h, w = x.shape
    windows = x.reshape(b, c, z // 2, 2, h // 2, 2, w // 2, 2)
    return windows.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(b, c, z // 2, h // 2, w // 2, 8).max(-1).values


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(conv(cin, cout), conv(cout, cout), conv(cout, cout))
            for cin, cout in zip((1,) + WIDTHS[:-1], WIDTHS)
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        skips = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index == 4:
                break
            skips.append(x)
            x = pool2(self.dropout(x) if index == 3 else x)
        return self.dropout(x), skips


class DecoderLevel(nn.Module):
    def __init__(self, cin, cout, upsampling):
        super().__init__()
        options = {"align_corners": False} if upsampling == "trilinear" else {}
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsampling, **options), conv(cin, cout))
        self.decode = nn.Sequential(conv(3 * cout, cout), conv(cout, cout))

    def forward(self, x, gated, prior):
        return self.decode(torch.cat((self.up(x), gated, prior), dim=1))


class DeCNN3D(nn.Module):
    def __init__(self, upsampling="nearest"):
        super().__init__()
        if upsampling not in {"nearest", "trilinear"}:
            raise ValueError("DeCNN upsampling must be nearest or trilinear")
        self.gated = Encoder()
        self.prior = Encoder()
        self.decoder = nn.ModuleList(
            DecoderLevel(cin, cout, upsampling)
            for cin, cout in zip((2 * WIDTHS[-1], *reversed(WIDTHS[1:-1])), reversed(WIDTHS[:-1]))
        )
        self.head = nn.Sequential(nn.Conv3d(WIDTHS[0], 1, 1), nn.ReLU())
        nn.init.constant_(self.head[0].bias, 0.1)
        self.upsampling = upsampling

    def forward(self, gated, prior):
        if gated.ndim != 5 or gated.shape != prior.shape or gated.shape[1] != 1 or min(gated.shape) < 1:
            raise ValueError("DeCNN expects two matching [N,1,Z,H,W] tensors")
        shape = gated.shape[-3:]
        pad = tuple(value for n in reversed(shape) for value in (0, (-n) % 16))
        if self.training:
            if min(shape) < 16 or any(pad):
                raise ValueError("DeCNN training blocks must have real dimensions divisible by 16")
            if gated.shape[0] * (shape[0] // 16) * (shape[1] // 16) * (shape[2] // 16) <= 1:
                raise ValueError("DeCNN training requires more than one bottleneck value per channel")
        if any(pad):
            gated = F.pad(gated, pad, mode="replicate")
            prior = F.pad(prior, pad, mode="replicate")
        g, g_skips = self.gated(gated)
        p, p_skips = self.prior(prior)
        decoded = torch.cat((g, p), dim=1)
        for level, gs, ps in zip(self.decoder, reversed(g_skips), reversed(p_skips)):
            decoded = level(decoded, gs, ps)
        z, h, w = shape
        return self.head(decoded)[..., :z, :h, :w]
