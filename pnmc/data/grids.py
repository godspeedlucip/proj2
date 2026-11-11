"""Scanner-frame grids, complete projection support, and external resampling."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .schema import GridSpec


def build_full_grid(source_grid: GridSpec, *, collar_mm: float = 10.0,
                    max_axis_voxels: int = 256):
    """Return a centered isotropic Q and its Q0 voxel-center indicator."""
    if not math.isfinite(collar_mm) or collar_mm < 0 or max_axis_voxels <= 0:
        raise ValueError("invalid collar or target size")
    source_shape = np.asarray(source_grid.shape_zyx[::-1], dtype=np.int64)
    source_spacing = np.asarray(source_grid.spacing_xyz_mm, dtype=np.float64)
    edge_lengths = source_shape * source_spacing
    spacing = (float(max(edge_lengths)) + 2 * collar_mm) / max_axis_voxels
    count = np.ceil((edge_lengths + 2 * collar_mm) / spacing - 1e-12).astype(int)
    count[:2] = ((count[:2] + 15) // 16) * 16
    count = np.maximum(count, 1)
    direction = np.asarray(source_grid.direction_xyz, dtype=np.float64)
    source_center = np.asarray(source_grid.origin_xyz_mm) + direction @ (
        (source_shape - 1) * source_spacing / 2)
    origin = source_center - direction @ ((count - 1) * spacing / 2)
    grid = GridSpec(tuple(int(x) for x in count[::-1]), (spacing,) * 3,
                    tuple(float(x) for x in origin), source_grid.direction_xyz,
                    source_grid.frame_id)
    axes = []
    for axis in range(3):
        # Q and Q0 have the same direction and center. Test physical voxel
        # centers against the source voxel-box boundaries, not against mu.
        source_index = ((torch.arange(int(count[axis]), dtype=torch.float64)
                         - (int(count[axis]) - 1) / 2) * spacing / source_spacing[axis]
                        + (int(source_shape[axis]) - 1) / 2)
        axes.append((source_index >= -.5 - 1e-9) &
                    (source_index <= source_shape[axis] - .5 + 1e-9))
    mask = axes[2][:, None, None] & axes[1][None, :, None] & axes[0][None, None, :]
    return grid, mask


def canonicalize_to_scanner(volume: torch.Tensor, grid: GridSpec,
                            scanner_grid: GridSpec) -> torch.Tensor:
    """Sample an external [z,y,x] volume at scanner-grid voxel centers."""
    if tuple(volume.shape) != grid.shape_zyx or volume.ndim != 3:
        raise ValueError("volume shape and source grid disagree")
    if not torch.is_floating_point(volume):
        raise ValueError("volume must be floating point")
    device, dtype = volume.device, volume.dtype
    matrix = torch.as_tensor(grid.direction_xyz, device=device, dtype=torch.float64)
    target_matrix = torch.as_tensor(scanner_grid.direction_xyz, device=device,
                                    dtype=torch.float64)
    source_origin = torch.as_tensor(grid.origin_xyz_mm, device=device, dtype=torch.float64)
    target_origin = torch.as_tensor(scanner_grid.origin_xyz_mm, device=device,
                                    dtype=torch.float64)
    z, y, x = [torch.arange(n, device=device, dtype=torch.float64)
               for n in scanner_grid.shape_zyx]
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    target_local = torch.stack((xx * scanner_grid.spacing_xyz_mm[0],
                                yy * scanner_grid.spacing_xyz_mm[1],
                                zz * scanner_grid.spacing_xyz_mm[2]), dim=-1)
    world = target_local @ target_matrix.T + target_origin
    source_local = (world - source_origin) @ torch.linalg.inv(matrix).T
    source_index = source_local / torch.as_tensor(grid.spacing_xyz_mm, device=device,
                                                  dtype=torch.float64)
    normalized = torch.stack([2 * (source_index[..., i] + .5) /
                              grid.shape_zyx[::-1][i] - 1 for i in range(3)], dim=-1)
    sampled = F.grid_sample(volume[None, None], normalized[None].to(dtype),
                            mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled[0, 0]
