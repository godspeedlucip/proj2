from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.nn.functional import grid_sample


def ordered_pairs(labels, expected, mode):
    order = {label: i for i, label in enumerate(expected)}
    return {
        target: [source for source in labels if source != target
                 and (mode == "all" or abs(order[source] - order[target]) == 1)]
        for target in labels
    }


def pull_grid(field):
    """Convert target-to-source XYZ voxel offsets to a normalized sampling grid."""
    z, y, x = field.shape[-3:]
    axes = [torch.arange(n, device=field.device, dtype=field.dtype) for n in (z, y, x)]
    zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
    coords = (xx + field[:, 0], yy + field[:, 1], zz + field[:, 2])
    grid = torch.stack([2 * (p + .5) / n - 1 for p, n in zip(coords, (x, y, z))], dim=-1)
    inside = (coords[0] >= 0) & (coords[0] <= x - 1)
    inside &= (coords[1] >= 0) & (coords[1] <= y - 1)
    inside &= (coords[2] >= 0) & (coords[2] <= z - 1)
    return grid, inside


def contribution(image, support, field):
    weighted = torch.where(support, image, 0.)
    if field is None:
        return weighted, support.to(image.dtype)
    grid, inside = pull_grid(field)
    values = torch.stack([weighted, support.to(image.dtype)])[None]
    sampled = grid_sample(values, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    numerator, weight = torch.where(inside[:, None], sampled, 0.)[0].unbind()
    return numerator, weight


class Accumulator:
    def __init__(self, image, support):
        self.support = support
        self.numerator, self.weight = contribution(image, support, None)
        self.external = torch.zeros_like(image)
        self.count = support.to(torch.int32)
        self.near = torch.zeros_like(image)
        self.far = torch.zeros_like(image)

    def add(self, image, support, field, *, adjacent):
        numerator, weight = contribution(image, support, field)
        weight.masked_fill_(~self.support, 0.)
        self.numerator += torch.where(self.support, numerator, 0.)
        self.weight += weight
        self.external += weight
        self.count += weight > 0
        if adjacent:
            self.near += weight
        else:
            self.far += weight

    def result(self):
        valid = self.weight > 0
        safe_weight = torch.where(valid, self.weight, 1.)
        return {
            "d": self.numerator / safe_weight, "valid_mask": valid,
            "weight": self.weight, "external_weight": self.external,
            "contribution_count": self.count, "raw_sum": self.numerator,
            "near_weight": self.near, "far_weight": self.far,
        }


def highpass(residual, support, sigma_mm, spacing):
    sigma = [sigma_mm / step for step in spacing]
    numerator = gaussian_filter(np.where(support, residual, 0.).astype(np.float64),
                                sigma, mode="constant", truncate=3.)
    denominator = gaussian_filter(support.astype(np.float64), sigma, mode="constant", truncate=3.)
    local_mean = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return np.where(support & (denominator > 0), residual - local_mean, 0.)


def reinject(baseline, source, support, alpha, sigma_mm, spacing):
    if alpha == 0:
        return baseline.copy()
    return (baseline + alpha * highpass(source - baseline, support, sigma_mm, spacing)).astype(baseline.dtype)


def field_quality(field, support, spacing):
    """Summarize field on the target geometric support using one-slice halos."""
    field = np.asarray(field, dtype=np.float32)
    shape = support.shape
    physical_scale = np.asarray(spacing[::-1], dtype=np.float32)[:, None, None, None]
    valid = folded = outside = 0
    total_mm = maximum_mm = 0.
    for start in range(0, shape[0], 16):
        stop = min(start + 16, shape[0])
        low, high = max(start - 1, 0), min(stop + 1, shape[0])
        inner = slice(start - low, stop - low)
        selected = support[start:stop]
        count = int(selected.sum())
        if not count:
            continue
        region = field[:, low:high]
        magnitude = np.linalg.norm(region[:, inner] * physical_scale, axis=0)[selected]
        total_mm += float(magnitude.sum(dtype=np.float64))
        maximum_mm = max(maximum_mm, float(magnitude.max()))
        valid += count
        axes = [np.arange(start, stop)[:, None, None],
                np.arange(shape[1])[None, :, None], np.arange(shape[2])[None, None, :]]
        in_bounds = selected.copy()
        for i, axis in enumerate(axes[::-1]):
            coordinate = axis + region[i, inner]
            in_bounds &= (coordinate >= 0) & (coordinate <= shape[2 - i] - 1)
        outside += count - int(in_bounds.sum())
        # Voxel-coordinate and physical-coordinate Jacobians are similar matrices.
        jacobian = np.empty((*selected.shape, 3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                axis = 2 - j
                derivative = np.gradient(region[i], axis=axis)[inner] if region.shape[axis + 1] > 1 else 0.
                jacobian[..., i, j] = derivative + (i == j)
        folded += int((np.linalg.det(jacobian[selected]) <= 0).sum())
    return {
        "target_voxels": valid, "mean_displacement_mm": total_mm / valid if valid else None,
        "max_displacement_mm": maximum_mm if valid else None,
        "out_of_bounds_fraction": outside / valid if valid else None,
        "folded_fraction": folded / valid if valid else None,
    }
