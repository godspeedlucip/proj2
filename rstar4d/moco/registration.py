"""RegisNet3D ported from the local DPI-MoCo reproduction at b709c28."""

from __future__ import annotations

import torch
from itertools import product
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

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


def local_ncc(moving, fixed, window=9, centers=None):
    if window < 1 or window % 2 == 0:
        raise ValueError("NCC window must be a positive odd integer")
    depth, height, width = moving.shape[-3:]
    slab = max(1, 1_048_576 // (height * width * len(moving)))
    if depth <= slab:
        return _ncc_region(moving, fixed, window, 0, depth, centers)
    radius = window // 2
    score = moving.new_zeros(())
    for start in range(0, depth, slab):
        end = min(start + slab, depth)
        lower, upper = max(0, start - radius), min(depth, end + radius)
        args = (moving[:, :, lower:upper], fixed[:, :, lower:upper], window,
                start - lower, end - lower, None if centers is None else centers[:, :, start:end])
        count = end - start if centers is None else int(args[-1].sum())
        if not count:
            continue
        value = (checkpoint(_ncc_region, *args, use_reentrant=False)
                 if torch.is_grad_enabled() else _ncc_region(*args))
        score = score + value * (count / (depth if centers is None else int(centers.sum())))
    return score


def _ncc_region(moving, fixed, window, start, end, centers):
    products = torch.cat((moving, fixed, moving.square(), fixed.square(), moving * fixed), dim=1)
    radius = window // 2
    for axis in range(3):
        kernel = tuple(window if i == axis else 1 for i in range(3))
        padding = tuple(radius if i == axis else 0 for i in range(3))
        if products.shape[axis + 2] < window:
            explicit = tuple(p for value in reversed(padding) for p in (value, value))
            products = F.pad(products, explicit)
            padding = (0, 0, 0)
        products = F.avg_pool3d(products, kernel, stride=1, padding=padding,
                                count_include_pad=True, divisor_override=1)
    m, f, m2, f2, mf = products[:, :, start:end].chunk(5, dim=1)
    count = window ** 3
    covariance = mf - m * f / count
    var_m = (m2 - m.square() / count).clamp_min(0)
    var_f = (f2 - f.square() / count).clamp_min(0)
    scores = -(covariance.square() / (var_m * var_f + 1e-5))
    return scores.mean() if centers is None else scores.masked_select(centers).mean()


def field_smoothness(field):
    differences = [field.diff(dim=axis).square().mean() for axis in (2, 3, 4)
                   if field.shape[axis] > 1]
    return sum(differences, field.new_zeros(())) / 3


def estimate_field(model, moving, fixed, limit, tile=None, halo=64):
    """Use aligned blocks with sufficient halo for the four encoder strides."""
    shape = moving.shape[-3:]
    if tile is None or all(width >= size for width, size in zip(tile, shape)):
        return model(moving, fixed).clamp(-limit, limit)
    output = moving.new_empty((len(moving), 3, *shape))
    for origin in product(*(range(0, size, width) for size, width in zip(shape, tile))):
        end = [min(start + width, size) for start, width, size in zip(origin, tile, shape)]
        lower = [max(0, (start - halo) // 16 * 16) for start in origin]
        upper = [min(size, (stop + halo + 15) // 16 * 16) for size, stop in zip(shape, end)]
        crop = tuple(slice(a, b) for a, b in zip(lower, upper))
        inner = tuple(slice(a - offset, b - offset) for a, b, offset in zip(origin, end, lower))
        target = tuple(slice(a, b) for a, b in zip(origin, end))
        field = model(moving[(Ellipsis,) + crop], fixed[(Ellipsis,) + crop])
        output[(Ellipsis,) + target] = field[(Ellipsis,) + inner].clamp(-limit, limit)
    return output
