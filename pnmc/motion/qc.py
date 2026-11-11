"""Geometry-derived registration region and final displacement-field checks."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from pnmc.data.schema import GridSpec


def motion_support(phase_sensitivities: list, support_mask: torch.Tensor,
                   grid: GridSpec, *, threshold: float = .01,
                   taper_mm: float = 10.):
    if (tuple(support_mask.shape) != grid.shape_zyx or support_mask.dtype != torch.bool or
            not phase_sensitivities):
        raise ValueError("motion support shape/type or sensitivities invalid")
    if not (0 < threshold < 1) or taper_mm <= 0:
        raise ValueError("invalid motion support threshold or taper")
    m0 = support_mask.clone()
    for sensitivity in phase_sensitivities:
        if tuple(sensitivity.shape) != grid.shape_zyx or not torch.isfinite(sensitivity).all():
            raise ValueError("sensitivity shape or finite values invalid")
        maximum = float(sensitivity.max())
        if maximum <= 0 or torch.any(sensitivity < 0):
            raise ValueError("sensitivity must be nonnegative with positive coverage")
        m0 &= sensitivity >= threshold * maximum
    if not bool(m0.any()):
        raise ValueError("empty motion coverage intersection")
    padded = np.pad(m0.cpu().numpy(), 1, mode="constant", constant_values=False)
    distance = distance_transform_edt(padded,
                                      sampling=grid.spacing_xyz_mm[::-1])[1:-1, 1:-1, 1:-1]
    taper = torch.as_tensor(np.clip(distance / taper_mm, 0., 1.),
                            dtype=torch.float32, device=support_mask.device)
    core = m0 & (taper >= 1. - 1e-7)
    if not bool(core.any()):
        raise ValueError("empty motion core after physical erosion")
    return m0, core, taper


def quality_check(field: torch.Tensor, *, grid: GridSpec,
                  m0: torch.Tensor, core: torch.Tensor) -> dict:
    if (tuple(field.shape) != (3, *grid.shape_zyx) or
            tuple(m0.shape) != grid.shape_zyx or tuple(core.shape) != grid.shape_zyx):
        raise ValueError("field/QC masks and grid shape disagree")
    if not torch.isfinite(field).all():
        raise ValueError("motion field must be finite")
    if not bool(core.any()) or bool((core & ~m0).any()):
        raise ValueError("motion core must be nonempty subset of M0")
    if bool((field[:, ~m0].abs() > 1e-5).any()):
        raise ValueError("final motion field must vanish outside M0")
    values = field.detach().cpu().numpy().astype(np.float64, copy=False)
    m0_np = m0.cpu().numpy()
    core_np = core.cpu().numpy()
    matrix = np.asarray(grid.direction_xyz, dtype=np.float64) @ np.diag(grid.spacing_xyz_mm)
    inverse = np.linalg.inv(matrix)
    zz, yy, xx = np.indices(grid.shape_zyx, dtype=np.float64)
    index = np.stack((xx, yy, zz), axis=-1)
    displaced = index + np.moveaxis(values, 0, -1) @ inverse.T
    limits = np.asarray(grid.shape_zyx[::-1], dtype=np.float64) - .5
    if np.any((displaced[core_np] < -.5 - 1e-6) |
              (displaced[core_np] > limits + 1e-6)):
        raise ValueError("motion maps core outside Q boundary")
    gradients = np.zeros((*grid.shape_zyx, 3, 3), dtype=np.float64)
    for component in range(3):
        by_array_axis = []
        for axis, length in enumerate(grid.shape_zyx):
            if length == 1:
                by_array_axis.append(np.zeros(grid.shape_zyx, dtype=np.float64))
            else:
                by_array_axis.append(np.gradient(values[component], axis=axis))
        gradients[..., component, :] = np.stack(by_array_axis[::-1], axis=-1) @ inverse
    jacobian = gradients + np.eye(3)
    determinant = np.linalg.det(jacobian)
    minimum = float(determinant[m0_np].min())
    if minimum <= 0 or not np.isfinite(minimum):
        raise ValueError("folded motion field: nonpositive Jacobian in M0")
    return {"status": "passed", "min_jacobian_m0": minimum,
            "max_displacement_mm": float(np.linalg.norm(values, axis=0)[m0_np].max()),
            "m0_voxels": int(m0_np.sum()), "core_voxels": int(core_np.sum())}
