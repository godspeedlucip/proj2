from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _cbr(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU())


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        growth = channels // 2
        self.projection = nn.Conv2d(in_channels, channels, 1)
        self.first = _cbr(channels, growth)
        self.second = _cbr(channels + growth, growth)
        self.merge = nn.Sequential(nn.Conv2d(channels + 2 * growth, channels, 3,
                                             padding=1, bias=False), nn.BatchNorm2d(channels))

    def forward(self, inputs):
        projection = self.projection(inputs)
        first = self.first(projection)
        second = self.second(torch.cat((projection, first), dim=1))
        residual = self.merge(torch.cat((projection, first, second), dim=1))
        return F.relu(projection + residual)


class RestoreNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        widths = (32, 64, 128, 256, 256)
        self.encoder = nn.ModuleList(ResidualDenseBlock(a, b) for a, b in zip((1, *widths), widths))
        self.decoder = nn.ModuleList(ResidualDenseBlock(a + b, b)
                                     for a, b in zip(widths[:0:-1], widths[-2::-1]))
        self.head = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, inputs):
        height, width = inputs.shape[-2:]
        pad_h, pad_w = (-height) % 16, (-width) % 16
        padded = inputs
        # Reflect one axis at a time. Replicate only where reflection is
        # undefined because the image is smaller than its required padding.
        if pad_w:
            padded = F.pad(padded, (0, pad_w, 0, 0), mode="reflect" if pad_w < width else "replicate")
        if pad_h:
            padded = F.pad(padded, (0, 0, 0, pad_h), mode="reflect" if pad_h < height else "replicate")
        features = [self.encoder[0](padded)]
        for encoder in self.encoder[1:]:
            features.append(encoder(F.max_pool2d(features[-1], 2)))
        value = features[-1]
        for decoder, skip in zip(self.decoder, reversed(features[:-1])):
            up = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = decoder(torch.cat((up, skip), dim=1))
        return inputs + self.head(value)[..., :height, :width]


class Critic2D(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (32, 32, 64, 64, 128, 128, 256, 256)
        self.layers = nn.ModuleList(nn.Conv2d(a, b, 3, stride=1 + i % 2, padding=1)
                                    for i, (a, b) in enumerate(zip((2, *channels), channels)))
        self.head = nn.Linear(256, 1)
        row = torch.tensor([1., 2., 1.])
        self.register_buffer("lowpass", torch.outer(row, row)[None, None] / 16)

    def adapt(self, image):
        mode = "reflect" if min(image.shape[-2:]) > 1 else "replicate"
        low = F.conv2d(F.pad(image, (1, 1, 1, 1), mode=mode), self.lowpass)
        return torch.cat((image, image - low), dim=1)

    def forward(self, value, return_features=False):
        features = []
        for i, layer in enumerate(self.layers):
            value = F.leaky_relu(layer(value), 0.2)
            if return_features and i % 2 == 1:
                features.append(value)
        score = self.head(value.mean(dim=(2, 3))).squeeze(1)
        return (score, features) if return_features else score


class _VMConv(nn.Conv3d):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(in_channels, out_channels, 3, stride=stride,
                         padding=1 if stride == 1 else 0)
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, inputs):
        if self.stride == (2, 2, 2):
            inputs = F.pad(inputs, tuple(pad for length in reversed(inputs.shape[-3:])
                                        for pad in (length % 2, 1)))
        return F.leaky_relu(super().forward(inputs), 0.2)


class RegisNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList(_VMConv(a, b, 2) for a, b in zip((2, 16, 32, 32), (16, 32, 32, 32)))
        self.d4 = _VMConv(32, 32)
        self.d3 = _VMConv(64, 32)
        self.d2 = _VMConv(64, 32)
        self.d1 = nn.Sequential(_VMConv(48, 32), _VMConv(32, 32))
        self.d0 = nn.Sequential(_VMConv(34, 16), _VMConv(16, 16))
        self.head = nn.Conv3d(16, 3, 3, padding=1)
        nn.init.normal_(self.head.weight, std=1e-5)
        nn.init.zeros_(self.head.bias)

    def _fine(self, half_resolution, original, start, keep_start, keep_end):
        indices = torch.arange(start, start + original.shape[2], device=original.device) // 2
        up = F.interpolate(half_resolution.index_select(2, indices), scale_factor=(1, 2, 2), mode="nearest")
        return self.head(self.d0(torch.cat((up, original), dim=1)))[:, :, keep_start:keep_end]

    def forward(self, moving, fixed):
        shape = moving.shape[-3:]
        inputs = torch.cat((moving, fixed), dim=1)
        padding = tuple(pad for n in reversed(shape) for pad in (0, (-n) % 16))
        if any(padding):
            inputs = F.pad(inputs, padding, mode="replicate")
        features = [inputs]
        for layer in self.encoder:
            features.append(layer(features[-1]))
        value = self.d4(features[-1])
        for layer, skip in zip((self.d3, self.d2, self.d1), reversed(features[1:-1])):
            up = F.interpolate(value, scale_factor=2, mode="nearest")
            value = layer(torch.cat((up, skip), dim=1))
        # The full-resolution head has three 3x3x3 convolutions. Computing
        # slabs with a three-voxel halo bounds its activation memory.
        depth, height, width = inputs.shape[-3:]
        slab = max(1, 1_048_576 // (height * width * len(inputs)))
        if depth <= slab:
            field = self._fine(value, inputs, 0, 0, depth)
        else:
            slabs = []
            run = checkpoint if torch.is_grad_enabled() else None
            for start in range(0, depth, slab):
                end = min(start + slab, depth)
                lower, upper = max(0, start - 3), min(depth, end + 3)
                args = (value, inputs[:, :, lower:upper], lower, start - lower, end - lower)
                slabs.append(run(self._fine, *args, use_reentrant=False) if run else self._fine(*args))
            field = torch.cat(slabs, dim=2)
        z, y, x = shape
        return field[..., :z, :y, :x]
