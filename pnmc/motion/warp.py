"""Millimetre target-to-source pull with its discrete trilinear transpose."""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

from pnmc.data.schema import GridSpec


def _source_indices(field: torch.Tensor, source_grid: GridSpec,
                    target_grid: GridSpec) -> torch.Tensor:
    if tuple(field.shape) != (3, *target_grid.shape_zyx):
        raise ValueError("field must be [3,z,y,x] on the target grid")
    if not torch.is_floating_point(field) or not torch.isfinite(field).all():
        raise ValueError("field must be finite floating mm displacement")
    device = field.device
    dtype = torch.float64 if field.dtype == torch.float64 else torch.float32
    z, y, x = (torch.arange(n, device=device, dtype=dtype)
               for n in target_grid.shape_zyx)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    target_local = torch.stack((xx * target_grid.spacing_xyz_mm[0],
                                yy * target_grid.spacing_xyz_mm[1],
                                zz * target_grid.spacing_xyz_mm[2]), dim=-1)
    target_direction = torch.as_tensor(target_grid.direction_xyz, dtype=dtype, device=device)
    source_direction = torch.as_tensor(source_grid.direction_xyz, dtype=dtype, device=device)
    target_origin = torch.as_tensor(target_grid.origin_xyz_mm, dtype=dtype, device=device)
    source_origin = torch.as_tensor(source_grid.origin_xyz_mm, dtype=dtype, device=device)
    world = target_local @ target_direction.T + target_origin + field.permute(1, 2, 3, 0).to(dtype)
    source_local = (world - source_origin) @ torch.linalg.inv(source_direction).T
    spacing = torch.as_tensor(source_grid.spacing_xyz_mm, dtype=dtype, device=device)
    return source_local / spacing


def warp_pull(source: torch.Tensor, field_t_to_k: torch.Tensor,
              source_grid: GridSpec, target_grid: GridSpec) -> torch.Tensor:
    """At target r, sample source at r + target-to-source displacement (mm)."""
    if tuple(source.shape) != source_grid.shape_zyx or source.ndim != 3:
        raise ValueError("source volume and source grid disagree")
    if not torch.is_floating_point(source) or not torch.isfinite(source).all():
        raise ValueError("source volume must be finite floating values")
    if source.device != field_t_to_k.device:
        raise ValueError("source and field must be on the same device")
    index = _source_indices(field_t_to_k, source_grid, target_grid)
    counts = source_grid.shape_zyx[::-1]
    normalized = torch.stack([2 * (index[..., axis] + .5) / counts[axis] - 1
                              for axis in range(3)], dim=-1)
    sampled = F.grid_sample(source[None, None], normalized[None].to(source.dtype),
                            mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled[0, 0]


def warp_transpose(target_values: torch.Tensor, field_t_to_k: torch.Tensor,
                   source_grid: GridSpec, target_grid: GridSpec) -> torch.Tensor:
    """Scatter with the exact weights used by `warp_pull` for image values."""
    if tuple(target_values.shape) != target_grid.shape_zyx or target_values.ndim != 3:
        raise ValueError("target values and target grid disagree")
    if not torch.is_floating_point(target_values) or not torch.isfinite(target_values).all():
        raise ValueError("target values must be finite floating values")
    if target_values.device != field_t_to_k.device:
        raise ValueError("target values and field must be on the same device")
    index = _source_indices(field_t_to_k, source_grid, target_grid).to(target_values.dtype)
    lower = torch.floor(index).to(torch.int64)
    fraction = index - lower.to(index.dtype)
    nx, ny, nz = source_grid.shape_zyx[::-1]
    result = torch.zeros(source_grid.shape_zyx, dtype=target_values.dtype,
                         device=target_values.device)
    flat_result = result.reshape(-1)
    values = target_values.reshape(-1)
    for dx, dy, dz in itertools.product((0, 1), repeat=3):
        x = lower[..., 0] + dx
        y = lower[..., 1] + dy
        z = lower[..., 2] + dz
        valid = (x >= 0) & (x < nx) & (y >= 0) & (y < ny) & (z >= 0) & (z < nz)
        weight = ((fraction[..., 0] if dx else 1-fraction[..., 0]) *
                  (fraction[..., 1] if dy else 1-fraction[..., 1]) *
                  (fraction[..., 2] if dz else 1-fraction[..., 2])).reshape(-1)
        linear = (z * ny * nx + y * nx + x).reshape(-1)
        flat_result.scatter_add_(0, linear[valid.reshape(-1)],
                                 values[valid.reshape(-1)] * weight[valid.reshape(-1)])
    return result


def resample_field(field: torch.Tensor, from_grid: GridSpec,
                   to_grid: GridSpec) -> torch.Tensor:
    """Resample xyz physical components without scaling millimetres by voxels."""
    if tuple(field.shape) != (3, *from_grid.shape_zyx):
        raise ValueError("field shape and from_grid disagree")
    zero = torch.zeros((3, *to_grid.shape_zyx), dtype=field.dtype, device=field.device)
    return torch.stack([warp_pull(field[i], zero, from_grid, to_grid)
                        for i in range(3)], dim=0)
