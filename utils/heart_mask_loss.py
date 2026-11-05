"""Quality-aware reconstruction losses for optional heart masks."""

import math

import torch
import torch.nn.functional as F


def _per_sample_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    numerator = (values * weights).flatten(1).sum(1)
    denominator = weights.flatten(1).sum(1)
    if torch.any(denominator <= 0):
        raise ValueError("mask region or boundary band is empty")
    return numerator / denominator


def _boundary_band(mask: torch.Tensor, radius: int) -> torch.Tensor:
    kernel = 2 * radius + 1
    padded = F.pad(mask, (radius, radius, radius, radius), value=0)
    dilated = F.max_pool2d(padded, kernel, stride=1)
    eroded = -F.max_pool2d(-padded, kernel, stride=1)
    return (dilated - eroded).clamp(0, 1)


def _soft_region(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = math.ceil(3 * sigma)
    positions = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
    kernel_1d = torch.exp(-(positions ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 2 * radius + 1, 2 * radius + 1)
    return F.conv2d(mask, kernel_2d, padding=radius)


def _sobel_difference(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    horizontal = pred.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8
    vertical = horizontal.transpose(-1, -2)
    difference = pred - target
    return (
        F.conv2d(difference, horizontal, padding=1).abs()
        + F.conv2d(difference, vertical, padding=1).abs()
    ) / 2


def heart_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    quality: torch.Tensor,
    *,
    mode: str,
    radius: int,
    lambda_exact: float,
    lambda_rough: float,
    lambda_boundary: float,
    lambda_gradient: float,
    rough_sigma: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Return full-image MSE and optional region/boundary loss components."""
    if mode not in {"full", "roi", "boundary", "gradient"}:
        raise ValueError(f"unsupported loss mode: {mode}")
    if (pred.ndim != 4 or pred.shape[1] != 1 or pred.shape != target.shape
            or pred.shape != mask.shape or quality.shape != (pred.shape[0],)):
        raise ValueError("pred, target, and mask must have shape [N,1,H,W]; quality must have shape [N]")
    if radius < 1 or rough_sigma <= 0:
        raise ValueError("radius and rough_sigma must be positive")
    if any(value < 0 for value in (lambda_exact, lambda_rough, lambda_boundary, lambda_gradient)):
        raise ValueError("loss coefficients must be nonnegative")
    if not torch.isin(quality, quality.new_tensor([-1, 0, 1])).all():
        raise ValueError("quality must be -1, 0, or 1")

    squared_error = (pred - target).square()
    full = squared_error.mean()
    zero = full.new_zeros(())
    parts = {
        "full": full,
        "roi_exact": zero,
        "roi_rough": zero,
        "boundary": zero,
        "gradient": zero,
    }
    if mode != "full":
        exact = quality == 1
        rough = ~exact
        batch_size = pred.shape[0]
        if exact.any():
            exact_mask = mask[exact]
            parts["roi_exact"] = _per_sample_mean(squared_error[exact], exact_mask).sum() / batch_size
            if mode in {"boundary", "gradient"}:
                band = _boundary_band(exact_mask, radius)
                parts["boundary"] = _per_sample_mean(squared_error[exact], band).sum() / batch_size
                if mode == "gradient":
                    edge_error = _sobel_difference(pred[exact], target[exact])
                    parts["gradient"] = _per_sample_mean(edge_error, band).sum() / batch_size
        if rough.any():
            soft_mask = _soft_region(mask[rough], rough_sigma)
            parts["roi_rough"] = _per_sample_mean(squared_error[rough], soft_mask).sum() / batch_size

    parts["total"] = (
        parts["full"]
        + lambda_exact * parts["roi_exact"]
        + lambda_rough * parts["roi_rough"]
        + lambda_boundary * parts["boundary"]
        + lambda_gradient * parts["gradient"]
    )
    return parts
