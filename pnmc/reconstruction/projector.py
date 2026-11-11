"""Scanner-grid TIGRE Siddon forward and explicit approximate matched backprojection."""

from __future__ import annotations

import copy
import importlib
import math

import numpy as np
import torch

from pnmc.data.schema import GridSpec


def _import_tigre():
    try:
        return importlib.import_module("tigre")
    except (ImportError, OSError, KeyError) as exc:
        raise RuntimeError("TIGRE backend unavailable for physical projection") from exc


def _scalar_or_views(value, indices: np.ndarray, total: int, name: str):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        if not np.isfinite(array):
            raise ValueError(name + " must be finite")
        return float(array)
    if array.shape != (total,) or not np.isfinite(array).all():
        raise ValueError(name + " must be scalar or one value per view")
    return np.array(array[indices], dtype=np.float32, copy=True)


def _vector_or_views(value, indices: np.ndarray, total: int, width: int, name: str):
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (width,):
        selected = array.copy()
    elif array.shape == (total, width):
        selected = np.array(array[indices], dtype=np.float32, copy=True)
    else:
        raise ValueError(name + " must have a constant or per-view vector")
    if not np.isfinite(selected).all():
        raise ValueError(name + " must be finite")
    return selected


class TigreProjector:
    """TIGRE adapter; `matched` Atb is measured approximate, not exact A^T."""

    def __init__(self, geometry: dict, grid: GridSpec, angles_rad: np.ndarray,
                 *, device, fdk_padding_vu=(0, 0)):
        self.geometry = copy.deepcopy(geometry)
        self.grid = grid
        self.angles_rad = np.asarray(angles_rad, dtype=np.float32).copy()
        if (self.angles_rad.ndim != 1 or not len(self.angles_rad) or
                not np.isfinite(self.angles_rad).all()):
            raise ValueError("angles_rad must be a non-empty finite vector")
        self.device = torch.device(device)
        self.fdk_padding_vu = tuple(int(x) for x in fdk_padding_vu)
        if len(self.fdk_padding_vu) != 2 or any(x < 0 for x in self.fdk_padding_vu):
            raise ValueError("FDK detector padding must be two non-negative integers")
        if not np.allclose(grid.direction_xyz, np.eye(3), rtol=0, atol=1e-6):
            raise ValueError("TIGRE grid must first be canonicalized to scanner axes")

    def _views(self, view_indices) -> np.ndarray:
        indices = np.asarray(view_indices)
        if (indices.ndim != 1 or len(indices) == 0 or indices.dtype.kind not in "iu" or
                np.any(indices < 0) or np.any(indices >= len(self.angles_rad)) or
                len(set(indices.tolist())) != len(indices)):
            raise ValueError("view indices must be unique, non-empty and in range")
        return np.asarray(indices, dtype=np.int64)

    def _geometry_for_views(self, view_indices, *, padding_vu=(0, 0)):
        indices = self._views(view_indices)
        tigre = _import_tigre()
        geo = tigre.geometry()
        geo.mode = "cone"
        geo.nVoxel = np.asarray(self.grid.shape_zyx, dtype=np.int32)
        geo.dVoxel = np.asarray(self.grid.spacing_xyz_mm[::-1], dtype=np.float32)
        geo.sVoxel = geo.nVoxel * geo.dVoxel
        detector = np.asarray(self.geometry["nDetector_vu"], dtype=np.int32)
        if detector.shape != (2,) or np.any(detector <= 0):
            raise ValueError("nDetector_vu must be two positive values")
        pad = np.asarray(padding_vu, dtype=np.int32)
        if pad.shape != (2,) or np.any(pad < 0):
            raise ValueError("invalid detector padding")
        geo.nDetector = detector + 2 * pad
        geo.dDetector = np.asarray(self.geometry["dDetector_vu_mm"], dtype=np.float32)
        if geo.dDetector.shape != (2,) or np.any(geo.dDetector <= 0):
            raise ValueError("dDetector_vu_mm must be two positive values")
        geo.sDetector = geo.nDetector * geo.dDetector
        total = len(self.angles_rad)
        geo.DSO = _scalar_or_views(self.geometry["DSO_mm"], indices, total, "DSO_mm")
        geo.DSD = _scalar_or_views(self.geometry["DSD_mm"], indices, total, "DSD_mm")
        geo.offDetector = _vector_or_views(self.geometry["offDetector_vu_mm"],
                                           indices, total, 2, "offDetector_vu_mm")
        geo.rotDetector = _vector_or_views(self.geometry["rotDetector_xyz_rad"],
                                           indices, total, 3, "rotDetector_xyz_rad")
        geo.COR = _scalar_or_views(self.geometry["COR_mm"], indices, total, "COR_mm")
        geo.offOrigin = _vector_or_views(self.geometry["offOrigin_zyx_mm"],
                                         indices, total, 3, "offOrigin_zyx_mm")
        geo.accuracy = .5
        return tigre, geo, self.angles_rad[indices].copy()

    def _tensor_array(self, value: torch.Tensor, expected_shape: tuple) -> np.ndarray:
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
            raise ValueError("tensor shape disagrees with physical geometry")
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError("physical tensor must contain finite floating values")
        return np.ascontiguousarray(value.detach().to(device="cpu", dtype=torch.float32).numpy())

    def project(self, volume: torch.Tensor, view_indices: np.ndarray) -> torch.Tensor:
        indices = self._views(view_indices)
        array = self._tensor_array(volume, self.grid.shape_zyx)
        tigre, geo, angles = self._geometry_for_views(indices)
        result = tigre.Ax(array, geo, angles, projection_type="Siddon")
        return torch.from_numpy(np.asarray(result, dtype=np.float32).copy()).to(self.device)

    def backproject(self, residual: torch.Tensor, view_indices: np.ndarray) -> torch.Tensor:
        indices = self._views(view_indices)
        shape = (len(indices), *self.geometry["nDetector_vu"])
        array = self._tensor_array(residual, shape)
        tigre, geo, angles = self._geometry_for_views(indices)
        result = tigre.Atb(array, geo, angles, backprojection_type="matched")
        return torch.from_numpy(np.asarray(result, dtype=np.float32).copy()).to(self.device)

    def fdk(self, projections: torch.Tensor, view_indices: np.ndarray) -> torch.Tensor:
        from .fdk import run_fdk
        return run_fdk(self, projections, view_indices)
