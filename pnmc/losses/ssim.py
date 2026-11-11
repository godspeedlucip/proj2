"""11x11 valid-convolution Gaussian SSIM on the declared [-1, 1] domain."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_kernel(dtype, device):
    axis = torch.arange(-5, 6, dtype=dtype, device=device)
    vector = torch.exp(-axis.square() / (2 * 1.5 ** 2))
    vector = vector / vector.sum()
    return torch.outer(vector, vector)[None, None]


def ssim_loss(pred: torch.Tensor, target: torch.Tensor,
              valid_mask: torch.Tensor = None) -> torch.Tensor:
    if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1] != 1:
        raise ValueError("SSIM expects matching [B,1,H,W] images")
    if min(pred.shape[-2:]) < 11:
        raise ValueError("valid SSIM window requires at least 11x11 pixels")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("SSIM images must be finite")
    x = (pred + 1) / 2
    y = (target + 1) / 2
    kernel = _gaussian_kernel(x.dtype, x.device)
    mean_x = F.conv2d(x, kernel)
    mean_y = F.conv2d(y, kernel)
    variance_x = (F.conv2d(x.square(), kernel) - mean_x.square()).clamp_min(0)
    variance_y = (F.conv2d(y.square(), kernel) - mean_y.square()).clamp_min(0)
    covariance = F.conv2d(x * y, kernel) - mean_x * mean_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    similarity = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2) / (
        (mean_x.square() + mean_y.square() + c1) *
        (variance_x + variance_y + c2)))
    if valid_mask is None:
        valid = torch.ones_like(similarity, dtype=torch.bool)
    else:
        if valid_mask.shape != pred.shape:
            raise ValueError("valid_mask must match SSIM image shape")
        mask = valid_mask.to(dtype=x.dtype)
        if not torch.isfinite(mask).all() or not torch.logical_or(mask == 0, mask == 1).all():
            raise ValueError("valid_mask must be finite and binary")
        counts = F.conv2d(mask, torch.ones_like(kernel))
        valid = counts == 121
    count = valid.flatten(1).sum(dim=1)
    if (count == 0).any():
        raise ValueError("no valid SSIM window for at least one image")
    mean_per_image = (similarity * valid).flatten(1).sum(dim=1) / count
    return 1 - mean_per_image.mean()
