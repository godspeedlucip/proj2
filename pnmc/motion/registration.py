"""Multi-scale fixed-to-moving SimpleITK diffeomorphic Demons adapter."""

from __future__ import annotations

import importlib

import numpy as np
import torch

from pnmc.config import RegistrationConfig
from pnmc.data.schema import GridSpec


def _import_sitk():
    try:
        return importlib.import_module("SimpleITK")
    except (ImportError, OSError) as exc:
        raise RuntimeError("SimpleITK unavailable for physical registration") from exc


def _field_sigma_pixels(sigma_mm: float, spacing_xyz_mm):
    if sigma_mm < 0 or len(spacing_xyz_mm) != 3 or any(x <= 0 for x in spacing_xyz_mm):
        raise ValueError("invalid physical field sigma or spacing")
    return tuple(float(sigma_mm / float(value)) for value in spacing_xyz_mm)


def _sitk_field_to_tensor(array: np.ndarray) -> torch.Tensor:
    values = np.asarray(array)
    if values.ndim != 4 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError("SimpleITK field must be finite [z,y,x,3]")
    return torch.from_numpy(np.asarray(values, dtype=np.float32).copy()).permute(3, 0, 1, 2).contiguous()


def _image(values: torch.Tensor, grid: GridSpec, sitk):
    image = sitk.GetImageFromArray(values.detach().cpu().numpy().astype(np.float32, copy=True))
    image.SetSpacing(grid.spacing_xyz_mm)
    image.SetOrigin(grid.origin_xyz_mm)
    image.SetDirection(tuple(np.asarray(grid.direction_xyz).reshape(-1)))
    return image


def register(moving_k: torch.Tensor, fixed_t: torch.Tensor, grid: GridSpec,
             config: RegistrationConfig):
    """Return d[t→k] on the fixed target grid, with physical xyz components."""
    if (tuple(moving_k.shape) != grid.shape_zyx or
            tuple(fixed_t.shape) != grid.shape_zyx or moving_k.ndim != 3):
        raise ValueError("moving/fixed shape and registration grid disagree")
    if not torch.isfinite(moving_k).all() or not torch.isfinite(fixed_t).all():
        raise ValueError("registration images must be finite")
    if not isinstance(config, RegistrationConfig):
        raise ValueError("registration requires RegistrationConfig")
    sitk = _import_sitk()
    fixed_image = _image(fixed_t, grid, sitk)
    moving_image = _image(moving_k, grid, sitk)
    field = None
    levels = []
    for factor, iterations, sigma_mm in zip(config.shrink_factors,
                                             config.iterations,
                                             config.image_sigma_mm):
        if sigma_mm > 0:
            fixed_smooth = sitk.SmoothingRecursiveGaussian(fixed_image, float(sigma_mm))
            moving_smooth = sitk.SmoothingRecursiveGaussian(moving_image, float(sigma_mm))
        else:
            fixed_smooth, moving_smooth = fixed_image, moving_image
        if factor > 1:
            fixed_level = sitk.Shrink(fixed_smooth, [int(factor)] * 3)
            moving_level = sitk.Shrink(moving_smooth, [int(factor)] * 3)
        else:
            fixed_level, moving_level = fixed_smooth, moving_smooth
        if field is None:
            field = sitk.Image(fixed_level.GetSize(), sitk.sitkVectorFloat64, 3)
            field.CopyInformation(fixed_level)
        else:
            field = sitk.Resample(field, fixed_level, sitk.Transform(),
                                  sitk.sitkLinear, 0.0, sitk.sitkVectorFloat64)
        registration = sitk.DiffeomorphicDemonsRegistrationFilter()
        registration.SetNumberOfIterations(int(iterations))
        registration.SetUseImageSpacing(True)
        registration.SetSmoothDisplacementField(True)
        registration.SetSmoothUpdateField(False)
        pixel_sigmas = _field_sigma_pixels(config.field_sigma_mm,
                                            fixed_level.GetSpacing())
        registration.SetStandardDeviations(pixel_sigmas)
        try:
            field = registration.Execute(fixed_level, moving_level, field)
        except RuntimeError as exc:
            raise ValueError("SimpleITK Demons registration failed") from exc
        elapsed = int(registration.GetElapsedIterations())
        levels.append({"shrink_factor": int(factor),
                       "image_sigma_mm": float(sigma_mm),
                       "field_sigma_pixels_xyz": list(pixel_sigmas),
                       "spacing_xyz_mm": list(fixed_level.GetSpacing()),
                       "iterations_requested": int(iterations),
                       "iterations_elapsed": elapsed,
                       "stop_reason": "iteration_limit" if elapsed >= iterations else "converged",
                       "metric": float(registration.GetMetric()),
                       "rms_change": float(registration.GetRMSChange())})
    field = sitk.Resample(field, fixed_image, sitk.Transform(),
                          sitk.sitkLinear, 0.0, sitk.sitkVectorFloat64)
    tensor = _sitk_field_to_tensor(sitk.GetArrayFromImage(field))
    if tuple(tensor.shape[1:]) != grid.shape_zyx:
        raise ValueError("registration field grid differs from target")
    report = {"direction": "target_fixed_to_source_moving", "unit": "mm",
              "components": "xyz", "grid_sha256": grid.fingerprint(),
              "levels": levels}
    return tensor, report
