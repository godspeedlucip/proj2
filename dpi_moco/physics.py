from __future__ import annotations

import numpy as np


def pigae(sparse, prior, reprojection, mode):
    correction = prior - reprojection
    if mode == "printed":
        return sparse - correction
    if mode == "static_consistent":
        return sparse + correction
    if mode == "off":
        return sparse.copy()
    raise ValueError(f"Unknown PIGAE mode: {mode}")


class TigreProjector:
    """Select scanner geometry by original projection index for both operators."""

    def __init__(self, grid, geometry, angles, fdk):
        try:
            import tigre
        except (ImportError, OSError) as error:
            raise RuntimeError("Prepare/infer requires a CUDA environment with TIGRE") from error
        if not np.allclose(grid["direction_xyz"], np.eye(3), rtol=0, atol=1e-6):
            raise ValueError("TIGRE reconstruction grid must be aligned to scanner axes")
        self.tigre = tigre
        self.shape = tuple(grid["shape_zyx"])
        self.spacing = np.asarray(grid["spacing_xyz_mm"][::-1], dtype=np.float32)
        self.center = (np.asarray(grid["origin_xyz_mm"][::-1])
                       + (np.asarray(self.shape) - 1) * self.spacing / 2)
        self.angles = np.asarray(angles, dtype=np.float32)
        self.detector = tuple(geometry["nDetector_vu"])
        self.detector_spacing = np.asarray(geometry["dDetector_vu_mm"], dtype=np.float32)
        self.padding = tuple(fdk["padding_vu"])
        self.fdk_settings = fdk
        if (len(self.detector) != 2 or any(int(x) != x or x <= 0 for x in self.detector)
                or self.detector_spacing.shape != (2,) or np.any(self.detector_spacing <= 0)
                or not np.isfinite(self.detector_spacing).all()):
            raise ValueError("Invalid detector geometry")
        if len(self.padding) != 2 or any(int(p) != p or not 0 <= p <= n for p, n in zip(self.padding, self.detector)):
            raise ValueError("Invalid detector padding")
        self.geometry = {}
        for name, value, width in (
            ("DSO", geometry["DSO_mm"], None), ("DSD", geometry["DSD_mm"], None),
            ("offDetector", geometry.get("offDetector_vu_mm", [0, 0]), 2),
            ("rotDetector", geometry.get("rotDetector_xyz_rad", [0, 0, 0]), 3),
            ("COR", geometry.get("COR_mm", 0), None),
            ("offOrigin", self.center + np.asarray(geometry.get("origin_shift_zyx_mm", [0, 0, 0])), 3),
        ):
            value = np.asarray(value, dtype=np.float32)
            fixed_shape = (width,) if width else ()
            if value.shape not in {fixed_shape, (len(self.angles), *fixed_shape)} or not np.isfinite(value).all():
                raise ValueError(f"Invalid constant or per-view geometry: {name}")
            self.geometry[name] = (value, value.shape == fixed_shape)
        if (np.any(self.geometry["DSO"][0] <= 0) or
                np.any(self.geometry["DSD"][0] <= self.geometry["DSO"][0])):
            raise ValueError("Require distances DSD > DSO > 0")
        if fdk["dowang"] and not self.geometry["offDetector"][1]:
            raise ValueError("TIGRE Wang weighting requires a fixed detector offset")

    def _geometry_for_views(self, views, padding):
        if (len(views) == 0 or np.any(views < 0) or np.any(views >= len(self.angles))
                or len(np.unique(views)) != len(views)):
            raise ValueError("Expected unique, in-range projection indices")
        geo = self.tigre.geometry()
        geo.mode = "cone"
        geo.nVoxel = np.asarray(self.shape, dtype=np.int32)
        geo.dVoxel = self.spacing.copy()
        geo.sVoxel = geo.nVoxel * geo.dVoxel
        geo.nDetector = np.asarray(self.detector, dtype=np.int32) + 2 * np.asarray(padding, dtype=np.int32)
        geo.dDetector = self.detector_spacing.copy()
        geo.sDetector = geo.nDetector * geo.dDetector
        geo.accuracy = 0.5
        for key, (values, fixed) in self.geometry.items():
            value = values if fixed else values[views]
            setattr(geo, key, float(value) if value.ndim == 0 else value.copy())
        return geo, self.angles[views].copy()

    def project(self, volume, views):
        volume = np.ascontiguousarray(volume, dtype=np.float32)
        if volume.shape != self.shape or not np.isfinite(volume).all():
            raise ValueError("Invalid volume for projection")
        geometry, angles = self._geometry_for_views(views, (0, 0))
        return self.tigre.Ax(volume, geometry, angles, projection_type="Siddon")

    def fdk(self, projections, views):
        if projections.shape != (len(views), *self.detector) or not np.isfinite(projections).all():
            raise ValueError("Invalid projection data for FDK")
        projections = np.ascontiguousarray(projections, dtype=np.float32)
        # Match the source generator and PNMC: reflect the low edge and append
        # the high-edge strip in acquisition order on each detector axis.
        for axis, padding in enumerate(self.padding, start=1):
            if padding:
                values = np.moveaxis(projections, axis, -1)
                values = np.concatenate((values[..., :padding][..., ::-1], values,
                                         values[..., -padding:]), axis=-1)
                projections = np.moveaxis(values, -1, axis)
        geometry, angles = self._geometry_for_views(views, self.padding)
        result = self.tigre.algorithms.fdk(np.ascontiguousarray(projections), geometry, angles,
                                           filter=self.fdk_settings["filter"],
                                           dowang=self.fdk_settings["dowang"])
        if result.shape != self.shape or not np.isfinite(result).all():
            raise ValueError("Invalid TIGRE FDK output")
        return result
