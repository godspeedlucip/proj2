"""Vendored user CT/TIGRE reference helpers. See ../provenance.json.
Executable function bodies are retained to preserve geometry, padding and mask semantics.
"""
from __future__ import annotations
import csv, hashlib, json, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from scipy.ndimage import affine_transform, zoom
N_PROJECTIONS=600
ANGLE_SPAN_DEG=300.0
VOXEL_SPACING_MULTIPLIER=1.5
N_DETECTOR=(474,648)
D_DETECTOR_MM=(0.616,0.616)
DSO_MM=750.0
DSD_MM=1250.0
MIN_PHASE=1
MAX_PHASE=11
SOURCE_PADDING_HU=-2048.0
PADDING_ROW_PIXELS=100
PADDING_COL_PIXELS=100
FOV_OUTPUT_TRANSVERSE_PIXELS=256
FOV_MASK_DIAMETER_SHRINK_PIXELS=10


class TigreTool:
    """TIGRE helpers plus independent per-instance geometry and angle state."""

    DEFAULT_START_ANGLE = 0.0
    DEFAULT_END_ANGLE = 360.0
    DEFAULT_N_ANGLES = 360
    geometry: Any | None = None
    geo: Any | None = None
    angles: np.ndarray | None = None
    start_angle: float | None = None
    end_angle: float | None = None
    n_angles: int | None = None

    def __init__(self, geometry: Any | None = None, angles: np.ndarray | None = None):
        """Create an independent TIGRE configuration holder.

        Each instance owns its geometry and angle array, so multiple
        configurations can coexist in one notebook without overwriting one
        another.
        """
        self.geometry = geometry
        self.geo = geometry
        self.angles = None if angles is None else np.asarray(angles, dtype=np.float32)
        self.start_angle = None
        self.end_angle = None
        self.n_angles = None if angles is None else int(len(self.angles))

    def set_geometry(
        self,
        volume_shape: Iterable[int],
        voxel_spacing: Iterable[float],
        n_detector: Iterable[int] = (512, 512),
        d_detector: Iterable[float] = (0.3, 0.3),
        dso: float = 780.0,
        dsd: float = 1200.0,
        mode: str = "cone",
    ):
        """Build and store this instance's TIGRE geometry."""
        self.geometry = self.build_tigre_geometry(
            volume_shape,
            voxel_spacing,
            n_detector=n_detector,
            d_detector=d_detector,
            dso=dso,
            dsd=dsd,
            mode=mode,
        )
        self.geo = self.geometry
        return self.geometry

    def set_angles(
        self,
        start_angle: float | None = None,
        end_angle: float | None = None,
        angles: int | None = None,
    ) -> np.ndarray:
        """Create and store angles from ``(start_angle, end_angle, angles)``.

        Angle bounds are specified in degrees, matching ``make_angles``.
        ``angles`` is the number of views, and the end angle is excluded.
        """
        start_angle = self.DEFAULT_START_ANGLE if start_angle is None else start_angle
        end_angle = self.DEFAULT_END_ANGLE if end_angle is None else end_angle
        angles = self.DEFAULT_N_ANGLES if angles is None else angles
        self.angles = self.make_angles(angles, start_angle, end_angle)
        self.start_angle = float(start_angle)
        self.end_angle = float(end_angle)
        self.n_angles = int(angles)
        return self.angles

    @staticmethod
    def hu_to_mu(hu: np.ndarray, mu_water: float = 0.02) -> np.ndarray:
        """Convert Hounsfield units to non-negative linear attenuation values."""
        hu = np.asarray(hu, dtype=np.float32)
        return np.maximum(mu_water * (1.0 + hu / 1000.0), 0.0).astype(np.float32)

    @staticmethod
    def make_angles(
        n_angles: int,
        start_deg: float = 0.0,
        end_deg: float = 360.0,
    ) -> np.ndarray:
        """Return evenly spaced TIGRE angles in radians without the endpoint."""
        if n_angles < 1:
            raise ValueError("n_angles must be positive")
        return np.deg2rad(
            np.linspace(start_deg, end_deg, n_angles, endpoint=False)
        ).astype(np.float32)

    @staticmethod
    def build_tigre_geometry(
        volume_shape: Iterable[int],
        voxel_spacing: Iterable[float],
        n_detector: Iterable[int] = (512, 512),
        d_detector: Iterable[float] = (0.3, 0.3),
        dso: float = 780.0,
        dsd: float = 1200.0,
        mode: str = "cone",
    ):
        """Create a TIGRE geometry using the ``(z, y, x)`` convention."""
        import tigre

        geo = tigre.geometry()
        geo.mode = mode
        geo.DSO = float(dso)
        geo.DSD = float(dsd)
        geo.nDetector = np.asarray(tuple(n_detector), dtype=np.int32)
        geo.dDetector = np.asarray(tuple(d_detector), dtype=np.float32)
        geo.sDetector = geo.nDetector * geo.dDetector
        geo.nVoxel = np.asarray(tuple(volume_shape), dtype=np.int32)
        geo.dVoxel = np.asarray(tuple(voxel_spacing), dtype=np.float32)
        geo.sVoxel = geo.nVoxel * geo.dVoxel
        geo.offOrigin = np.zeros(3, dtype=np.float32)
        geo.offDetector = np.zeros(2, dtype=np.float32)
        geo.rotDetector = np.zeros(3, dtype=np.float32)
        geo.COR = 0.0
        geo.accuracy = 0.5
        return geo

    set_geo = build_tigre_geometry

    @staticmethod
    def project_tigre(
        volume: np.ndarray,
        geometry: Any,
        angles: np.ndarray,
    ) -> np.ndarray:
        """Simulate cone-beam projections with TIGRE's ``Ax`` operator."""
        import tigre

        projections = tigre.Ax(
            np.asarray(volume, dtype=np.float32),
            geometry,
            np.asarray(angles, dtype=np.float32),
        )
        return np.asarray(projections, dtype=np.float32)

    def project(
        self,
        volume: np.ndarray,
        geometry: Any | None = None,
        angles: np.ndarray | None = None,
    ) -> np.ndarray:
        """Project using stored state unless geometry or angles are provided."""
        geometry = self.geometry if geometry is None else geometry
        angles = self.angles if angles is None else angles
        if geometry is None or angles is None:
            raise ValueError("configure geometry and angles before calling project")
        return self.project_tigre(volume, geometry, angles)

    @staticmethod
    def reconstruct_tigre(
        projections: np.ndarray,
        geometry: Any,
        angles: np.ndarray,
        use_filter: bool = True,
    ) -> np.ndarray:
        """Reconstruct projections with TIGRE FDK or its unfiltered variant."""
        import tigre.algorithms as algorithms

        projections = np.asarray(projections, dtype=np.float32)
        angles = np.asarray(angles, dtype=np.float32)
        if use_filter:
            reconstruction = algorithms.fdk(projections, geometry, angles)
        else:
            reconstruction = algorithms.fdk_no_filter(projections, geometry, angles)
        return np.asarray(reconstruction, dtype=np.float32)

    def reconstruct(
        self,
        projections: np.ndarray,
        geometry: Any | None = None,
        angles: np.ndarray | None = None,
        use_filter: bool = True,
    ) -> np.ndarray:
        """Reconstruct using stored state unless geometry or angles are provided."""
        geometry = self.geometry if geometry is None else geometry
        angles = self.angles if angles is None else angles
        if geometry is None or angles is None:
            raise ValueError("configure geometry and angles before calling reconstruct")
        return self.reconstruct_tigre(projections, geometry, angles, use_filter=use_filter)

    @staticmethod
    def reconstruct_volume(
        volume: np.ndarray,
        spacing: Iterable[float],
        angles: np.ndarray,
        *,
        input_domain: str = "hu",
        n_detector: Iterable[int] = (512, 512),
        d_detector: Iterable[float] = (0.3, 0.3),
        dso: float = 780.0,
        dsd: float = 1200.0,
        use_filter: bool = True,
        return_projection: bool = False,
    ):
        """Project and reconstruct one volume using the shared TIGRE helpers."""
        if input_domain == "hu":
            volume_for_projection = TigreTool.hu_to_mu(volume)
        elif input_domain == "mu":
            volume_for_projection = np.asarray(volume, dtype=np.float32)
        else:
            raise ValueError("input_domain must be 'hu' or 'mu'")

        geometry = TigreTool.build_tigre_geometry(
            volume_for_projection.shape,
            spacing,
            n_detector=n_detector,
            d_detector=d_detector,
            dso=dso,
            dsd=dsd,
        )
        projections = TigreTool.project_tigre(volume_for_projection, geometry, angles)
        reconstruction = TigreTool.reconstruct_tigre(
            projections,
            geometry,
            angles,
            use_filter=use_filter,
        )
        if return_projection:
            return reconstruction, projections, geometry
        return reconstruction, geometry

class ProjectionTool:
    @staticmethod
    def pad_linatech(projection: np.ndarray, padding: int) -> np.ndarray:
        projection = np.asarray(projection)
        if projection.ndim != 3:
            raise ValueError("projection must be a 3-D array")
        if padding < 0:
            raise ValueError("padding must be non-negative")
        if padding == 0:
            return projection.copy()
        if padding > projection.shape[1]:
            raise ValueError("padding cannot exceed the detector-column count")
        left = projection[:, padding - 1 :: -1, :]
        right = projection[:, -padding:, :]
        return np.concatenate((left, projection, right), axis=1)

@dataclass(frozen=True)
class SeriesSettings:
    """Per-series acquisition and FOV settings from the CSV file."""

    shift_up_voxels: int = 0
    shift_down_voxels: int = 0
    shift_left_voxels: int = 0
    shift_right_voxels: int = 0
    voxel_spacing_multiplier: float = VOXEL_SPACING_MULTIPLIER
    n_detector_rows: int = N_DETECTOR[0]
    n_detector_cols: int = N_DETECTOR[1]
    d_detector_row_mm: float = D_DETECTOR_MM[0]
    d_detector_col_mm: float = D_DETECTOR_MM[1]
    dso_mm: float = DSO_MM
    dsd_mm: float = DSD_MM
    n_projections: int = N_PROJECTIONS
    angle_span_deg: float = ANGLE_SPAN_DEG
    min_phase: int = MIN_PHASE
    max_phase: int = MAX_PHASE
    padding_row_pixels: int = 100
    padding_col_pixels: int = 100
    fov_output_pixels: int = 256
    # Transverse FOV width in source voxels. Zero retains the legacy crop.
    fov_source_pixels: int = 0
    fov_mask_shrink_pixels: int = 10

    @property
    def n_detector(self) -> tuple[int, int]:
        return self.n_detector_rows, self.n_detector_cols

    @property
    def d_detector_mm(self) -> tuple[float, float]:
        return self.d_detector_row_mm, self.d_detector_col_mm

    def validate(self) -> None:
        for name in ("shift_up_voxels", "shift_down_voxels", "shift_left_voxels",
                     "shift_right_voxels", "padding_row_pixels", "padding_col_pixels",
                     "fov_mask_shrink_pixels", "fov_source_pixels"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("n_detector_rows", "n_detector_cols", "n_projections", "fov_output_pixels"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("voxel_spacing_multiplier", "d_detector_row_mm", "d_detector_col_mm",
                     "dso_mm", "dsd_mm", "angle_span_deg"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.padding_row_pixels > self.n_detector_rows or self.padding_col_pixels > self.n_detector_cols:
            raise ValueError("Projection padding cannot exceed detector size")

def load_series_settings(csv_path: Path) -> dict[str, SeriesSettings]:
    """Read one row per series, using zero translation for empty shift cells."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Series settings CSV not found: {csv_path}")
    defaults = asdict(SeriesSettings())
    int_fields = {name for name, value in defaults.items() if isinstance(value, int)}
    result: dict[str, SeriesSettings] = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        # Older settings CSVs omit the optional wide-FOV column.
        required = {"series", *(set(defaults) - {"fov_source_pixels"})}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Settings CSV must contain columns: {sorted(required)}")
        for row in reader:
            name = (row["series"] or "").strip().replace("\\", "/").strip("/")
            if not name:
                raise ValueError("Settings CSV contains an empty series name")
            if name in result:
                raise ValueError(f"Duplicate settings for series: {name}")
            values = {
                key: (int(row[key]) if key in int_fields else float(row[key]))
                if row.get(key) is not None and row[key].strip() else default
                for key, default in defaults.items()
            }
            settings = SeriesSettings(**values)
            settings.validate()
            result[name] = settings
    return result

def calibrated_volume(
    hu: np.ndarray, polynomial: np.poly1d, air_threshold: float, air_value: float,
    *, apply_air_threshold: bool = True,
) -> np.ndarray:
    """Calibrate measured CT voxels only; leave source padding at zero."""
    result = np.empty(hu.shape, dtype=np.float32)
    for start in range(0, hu.shape[0], 16):
        end = min(start + 16, hu.shape[0])
        source = np.asarray(hu[start:end], dtype=np.float32)
        valid = source != SOURCE_PADDING_HU
        destination = result[start:end]
        destination.fill(0.0)
        if not np.any(valid):
            continue
        volume_mu = TigreTool.hu_to_mu(source[valid])
        if apply_air_threshold:
            destination[valid] = calibrate_attenuation(
                volume_mu, polynomial, air_threshold, air_value
            )
        else:
            destination[valid] = polynomial(np.asarray(volume_mu, dtype=np.float64))
    return result

def load_attenuation_calibration(csv_path: Path) -> tuple[np.poly1d, float, float, dict[str, Any]]:
    """Load the quadratic attenuation calibration used by the notebook."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Calibration CSV not found: {csv_path}. Pass --calibration-csv to set it."
        )

    positions = ("contrast", "tissue", "air", "bone")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Calibration CSV has no header: {csv_path}")
        reader.fieldnames = [field.strip() if field is not None else "" for field in reader.fieldnames]
        rows = {
            row[""].strip(): row
            for row in reader
            if row.get("") and row[""].strip()
        }

    required = [f"{prefix}_{position}" for prefix in ("truth", "moni") for position in positions]
    missing = [key for key in required if key not in rows]
    if missing:
        raise ValueError(f"Missing calibration rows in {csv_path}: {missing}")

    public_means = np.asarray(
        [float(rows[f"moni_{position}"]["Mean"]) for position in positions],
        dtype=np.float64,
    )
    measured_means = np.asarray(
        [float(rows[f"truth_{position}"]["Mean"]) for position in positions],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(public_means)) or not np.all(np.isfinite(measured_means)):
        raise ValueError("Attenuation calibration means must be finite numbers")

    coefficients = np.polyfit(public_means, measured_means, deg=2)
    polynomial = np.poly1d(coefficients)
    air_value = float(rows["truth_air"]["Mean"])
    tissue_value = float(rows["truth_tissue"]["Mean"])
    air_threshold = (air_value + tissue_value) / 2.0
    details = {
        "source_csv": str(csv_path),
        "input_means_public_dataset": public_means.tolist(),
        "target_means_measured": measured_means.tolist(),
        "polynomial_degree": 2,
        "polynomial_coefficients": coefficients.tolist(),
        "air_threshold": air_threshold,
        "air_value": air_value,
    }
    return polynomial, air_threshold, air_value, details

def calibrate_attenuation(
    volume_mu: np.ndarray,
    polynomial: np.poly1d,
    air_threshold: float,
    air_value: float,
) -> np.ndarray:
    calibrated = polynomial(np.asarray(volume_mu, dtype=np.float64))
    calibrated = np.where(calibrated < air_threshold, air_value, calibrated)
    return np.asarray(calibrated, dtype=np.float32)

def reconstruct_prior(
    projections: np.ndarray,
    geometry: Any,
    angles: np.ndarray,
    tigre_tool: TigreTool,
    *,
    padding_row_pixels: int = PADDING_ROW_PIXELS,
    padding_col_pixels: int = PADDING_COL_PIXELS,
    fov_output_transverse_pixels: int = FOV_OUTPUT_TRANSVERSE_PIXELS,
    fov_source_pixels: int = 0,
    fov_mask_diameter_shrink_pixels: int = FOV_MASK_DIAMETER_SHRINK_PIXELS,
    use_view_geometry: bool = False,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Run FDK, optionally directly on a wider fixed-size transverse grid."""
    if tuple(projections.shape[1:]) != tuple(int(v) for v in geometry.nDetector):
        raise ValueError(
            f"Projection detector shape {projections.shape[1:]} does not match "
            f"geometry {tuple(int(v) for v in geometry.nDetector)}"
        )

    if padding_row_pixels < 0 or padding_col_pixels < 0:
        raise ValueError("Projection padding must be non-negative")
    if fov_output_transverse_pixels < 1 or fov_source_pixels < 0:
        raise ValueError("FOV pixel counts must be valid")
    padded = ProjectionTool.pad_linatech(projections, padding=padding_row_pixels)
    padded = np.transpose(padded, (0, 2, 1))
    padded = ProjectionTool.pad_linatech(padded, padding=padding_col_pixels)
    padded = np.transpose(padded, (0, 2, 1))
    padded_detector = np.asarray(geometry.nDetector, dtype=np.int32).copy()
    padded_detector += np.asarray(
        [2 * padding_row_pixels, 2 * padding_col_pixels], dtype=np.int32
    )
    if tuple(padded.shape[1:]) != tuple(int(v) for v in padded_detector):
        raise RuntimeError("Padded projection size does not match padded detector geometry")

    object_fov_mm = (
        np.asarray(geometry.sDetector, dtype=np.float64)
        * float(geometry.DSO)
        / float(geometry.DSD)
    )
    fov_size_mm_zyx = np.asarray(
        [object_fov_mm[0], object_fov_mm[1], object_fov_mm[1]], dtype=np.float64
    )
    fov_shape = np.floor(fov_size_mm_zyx / np.asarray(geometry.dVoxel, dtype=np.float64)).astype(np.int32)
    fov_shape = np.maximum(fov_shape, 1)
    fov_shape = np.minimum(fov_shape, np.asarray(geometry.nVoxel, dtype=np.int32))
    output_spacing = np.asarray(geometry.dVoxel, dtype=np.float64).copy()
    if fov_source_pixels:
        requested_mm = fov_source_pixels * output_spacing[1:]
        if np.any(requested_mm > object_fov_mm[1] + 1e-6):
            raise ValueError("Requested transverse FOV exceeds the unpadded detector FOV")
        fov_shape[1:] = fov_output_transverse_pixels
        output_spacing[1:] *= fov_source_pixels / fov_output_transverse_pixels

    fov_geometry = tigre_tool.set_geometry(
        volume_shape=tuple(int(v) for v in fov_shape),
        voxel_spacing=tuple(float(v) for v in output_spacing),
        n_detector=tuple(int(v) for v in padded_detector),
        d_detector=tuple(float(v) for v in geometry.dDetector),
        dso=float(geometry.DSO),
        dsd=float(geometry.DSD),
        mode=getattr(geometry, "mode", "cone"),
    )
    if use_view_geometry:
        from tigre.algorithms import fdk
        for name in ("offDetector", "rotDetector", "COR"):
            if hasattr(geometry, name):
                setattr(fov_geometry, name, getattr(geometry, name))
        reconstruction = fdk(padded, fov_geometry, angles, filter="ram_lak", dowang=False)
    else:
        reconstruction = TigreTool.reconstruct_tigre(
            projections=padded,
            geometry=fov_geometry,
            angles=angles,
        )

    nz, ny, nx = reconstruction.shape
    z_mm = (np.arange(nz, dtype=np.float64) - (nz - 1) / 2) * float(fov_geometry.dVoxel[0])
    y_mm = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2) * float(fov_geometry.dVoxel[1])
    x_mm = (np.arange(nx, dtype=np.float64) - (nx - 1) / 2) * float(fov_geometry.dVoxel[2])
    radius_y = object_fov_mm[1] / 2 - (fov_mask_diameter_shrink_pixels / 2) * float(fov_geometry.dVoxel[1])
    radius_x = object_fov_mm[1] / 2 - (fov_mask_diameter_shrink_pixels / 2) * float(fov_geometry.dVoxel[2])
    if radius_y <= 0 or radius_x <= 0:
        raise ValueError("FOV mask radius is not positive; check detector geometry")
    fov_mask = (
        (np.abs(z_mm)[:, None, None] <= object_fov_mm[0] / 2)
        & (
            (y_mm[None, :, None] / radius_y) ** 2
            + (x_mm[None, None, :] / radius_x) ** 2
            <= 1
        )
    )
    reconstruction = np.where(fov_mask, reconstruction, 0).astype(np.float32, copy=False)

    if fov_output_transverse_pixels < 1:
        raise ValueError("FOV output pixels must be positive")
    if min(ny, nx) < fov_output_transverse_pixels:
        # Some series have a smaller physical detector FOV at their native
        # voxel spacing. Resample the transverse plane to the common output
        # size while preserving its physical field of view.
        reconstruction = zoom(
            reconstruction,
            zoom=(1.0, fov_output_transverse_pixels / ny, fov_output_transverse_pixels / nx),
            order=1,
            mode="nearest",
            prefilter=False,
        )
        resized_y, resized_x = reconstruction.shape[1:]
        if resized_y != fov_output_transverse_pixels or resized_x != fov_output_transverse_pixels:
            reconstruction = reconstruction[
                :, :fov_output_transverse_pixels, :fov_output_transverse_pixels
            ]
            pad_y = max(0, fov_output_transverse_pixels - reconstruction.shape[1])
            pad_x = max(0, fov_output_transverse_pixels - reconstruction.shape[2])
            if pad_y or pad_x:
                reconstruction = np.pad(
                    reconstruction,
                    ((0, 0), (pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)),
                    mode="constant",
                )
        spacing_values = np.asarray(fov_geometry.dVoxel, dtype=np.float64).copy()
        spacing_values[1] *= ny / fov_output_transverse_pixels
        spacing_values[2] *= nx / fov_output_transverse_pixels
    else:
        y_start = (ny - fov_output_transverse_pixels) // 2
        x_start = (nx - fov_output_transverse_pixels) // 2
        reconstruction = reconstruction[
            :, y_start : y_start + fov_output_transverse_pixels,
            x_start : x_start + fov_output_transverse_pixels,
        ]
        spacing_values = np.asarray(fov_geometry.dVoxel, dtype=np.float64)
    spacing = tuple(float(v) for v in spacing_values)
    return reconstruction, spacing

def phase_grid(
    source_shape: tuple[int, int, int],
    source_spacing: tuple[float, float, float],
    settings: SeriesSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the FDK FOV size and source-array sampling offset."""
    settings.validate()
    if len(source_shape) != 3 or len(source_spacing) != 3:
        raise ValueError("Expected a 3-D source shape and spacing")
    if any(v <= 0 or not np.isfinite(v) for v in source_spacing):
        raise ValueError("Source voxel spacing must be finite and positive")
    used_spacing = np.asarray(source_spacing, np.float64) * settings.voxel_spacing_multiplier
    object_fov_yx = (
        np.asarray(settings.n_detector, np.float64)
        * np.asarray(settings.d_detector_mm, np.float64)
        * settings.dso_mm / settings.dsd_mm
    )
    fov_mm = np.asarray([object_fov_yx[0], object_fov_yx[1], object_fov_yx[1]])
    fov_shape = np.floor(fov_mm / used_spacing).astype(np.int32)
    fov_shape = np.maximum(fov_shape, 1)
    fov_shape = np.minimum(fov_shape, np.asarray(source_shape, np.int32))
    # The projection geometry has offOrigin=[0,-down*dY,-right*dX].
    # Therefore the centered reconstruction grid samples the source at
    # +down/+right indices. Half-voxel offsets cover odd/even size changes.
    shifts = np.asarray([
        0,
        settings.shift_down_voxels - settings.shift_up_voxels,
        settings.shift_right_voxels - settings.shift_left_voxels,
    ], np.float64)
    offset = (np.asarray(source_shape, np.float64) - fov_shape) / 2 + shifts
    return fov_shape, used_spacing, offset

def make_phase_gt(
    hu: Any,
    source_spacing: tuple[float, float, float],
    settings: SeriesSettings,
    polynomial: np.poly1d,
    air_threshold: float,
    air_value: float,
    *,
    return_valid_mask: bool = False,
) -> (tuple[np.ndarray, tuple[float, float, float]]
      | tuple[np.ndarray, tuple[float, float, float], np.ndarray]):
    """Sample calibrated CT on the same grid and postprocess as sparse FDK."""
    source_shape = tuple(int(v) for v in hu.shape)
    fov_shape, used_spacing, offset = phase_grid(source_shape, source_spacing, settings)
    source_step = np.ones(3, dtype=np.float64)
    output_shape = fov_shape.copy()
    if settings.fov_source_pixels:
        detector_fov_mm = (
            np.asarray(settings.n_detector, np.float64)
            * np.asarray(settings.d_detector_mm, np.float64)
            * settings.dso_mm / settings.dsd_mm
        )
        if np.any(settings.fov_source_pixels * used_spacing[1:] > detector_fov_mm[1] + 1e-6):
            raise ValueError("Requested transverse FOV exceeds the unpadded detector FOV")
        output_shape[1:] = settings.fov_output_pixels
        source_step[1:] = settings.fov_source_pixels / settings.fov_output_pixels
        shifts = np.asarray([
            0,
            settings.shift_down_voxels - settings.shift_up_voxels,
            settings.shift_right_voxels - settings.shift_left_voxels,
        ], np.float64)
        offset = (np.asarray(source_shape, np.float64) - 1) / 2 + shifts - (output_shape - 1) / 2 * source_step
    output_spacing = used_spacing * source_step
    # Read only the input slab needed by this FOV. This also works with a
    # nibabel proxy, avoiding a full 3-D float32 copy of every CT phase.
    first = np.floor(offset).astype(int)
    last = np.ceil(offset + source_step * (output_shape - 1)).astype(int) + 1
    start = np.maximum(first, 0)
    stop = np.minimum(last, source_shape)
    if np.any(stop <= start):
        sampled = np.zeros(tuple(output_shape), np.float32)
        valid_weights = np.zeros(tuple(output_shape), np.float32)
    else:
        source = np.asarray(hu[tuple(slice(int(a), int(b)) for a, b in zip(start, stop))], np.float32)
        calibrated = calibrated_volume(
            source, polynomial, air_threshold, air_value, apply_air_threshold=False
        )
        transform = np.diag(source_step)
        sample_offset = offset - start
        sampled = affine_transform(
            calibrated,
            matrix=transform,
            offset=sample_offset,
            output_shape=tuple(int(v) for v in output_shape),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
            output=np.float32,
        )
        valid_weights = affine_transform(
            (source != SOURCE_PADDING_HU).astype(np.float32),
            matrix=transform,
            offset=sample_offset,
            output_shape=tuple(int(v) for v in output_shape),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
            output=np.float32,
        )

    object_fov_yx = (
        np.asarray(settings.n_detector, np.float64)
        * np.asarray(settings.d_detector_mm, np.float64)
        * settings.dso_mm / settings.dsd_mm
    )
    nz, ny, nx = sampled.shape
    z_mm = (np.arange(nz) - (nz - 1) / 2) * output_spacing[0]
    y_mm = (np.arange(ny) - (ny - 1) / 2) * output_spacing[1]
    x_mm = (np.arange(nx) - (nx - 1) / 2) * output_spacing[2]
    radius_y = object_fov_yx[1] / 2 - settings.fov_mask_shrink_pixels / 2 * output_spacing[1]
    radius_x = object_fov_yx[1] / 2 - settings.fov_mask_shrink_pixels / 2 * output_spacing[2]
    if radius_y <= 0 or radius_x <= 0:
        raise ValueError("FOV mask radius is not positive; check detector geometry")
    mask = (
        (np.abs(z_mm)[:, None, None] <= object_fov_yx[0] / 2)
        & ((y_mm[None, :, None] / radius_y) ** 2
           + (x_mm[None, None, :] / radius_x) ** 2 <= 1)
    )
    valid_weights *= mask
    sampled = np.where(mask, sampled, 0).astype(np.float32, copy=False)

    width = settings.fov_output_pixels
    if settings.fov_source_pixels:
        valid = valid_weights >= 1.0 - 1e-6
        sampled[~valid] = 0.0
        result = (sampled, tuple(float(v) for v in output_spacing))
        return (*result, valid.astype(np.uint8)) if return_valid_mask else result
    if min(ny, nx) < width:
        scale = (1.0, width / ny, width / nx)
        sampled = zoom(
            sampled, scale,
            order=1, mode="nearest", prefilter=False,
        )
        valid_weights = zoom(
            valid_weights, scale, order=1, mode="nearest", prefilter=False,
        )
        if sampled.shape[1:] != (width, width):
            sampled = sampled[:, :width, :width]
            valid_weights = valid_weights[:, :width, :width]
            pad_y, pad_x = width - sampled.shape[1], width - sampled.shape[2]
            sampled = np.pad(
                sampled,
                ((0, 0), (pad_y // 2, pad_y - pad_y // 2),
                 (pad_x // 2, pad_x - pad_x // 2)),
            )
            valid_weights = np.pad(
                valid_weights,
                ((0, 0), (pad_y // 2, pad_y - pad_y // 2),
                 (pad_x // 2, pad_x - pad_x // 2)),
            )
        output_spacing[1] *= ny / width
        output_spacing[2] *= nx / width
    else:
        y_start, x_start = (ny - width) // 2, (nx - width) // 2
        sampled = sampled[:, y_start:y_start + width, x_start:x_start + width]
        valid_weights = valid_weights[:, y_start:y_start + width, x_start:x_start + width]
    valid = valid_weights >= 1.0 - 1e-6
    sampled[~valid] = 0.0
    result = (sampled, tuple(float(v) for v in output_spacing))
    return (*result, valid.astype(np.uint8)) if return_valid_mask else result

REQUIRED_COLUMNS = (
    "series", "poly_a2", "poly_a1", "poly_a0", "truth_air",
    "prior_air", "truth_tissue", "prior_tissue", "truth_contrast",
    "prior_contrast", "fit_design_degree", "contrast_residual_mu", "warnings",
)

def validate_calibration_round(csv_path: Path | None, round_root: Path) -> None:
    """Reject an explicitly named round CSV for a different source round."""
    if csv_path is None:
        return
    source_match = re.search(r"round[-_]?([0-9]+)", round_root.name, re.IGNORECASE)
    csv_match = re.search(r"round[-_]?([0-9]+)", csv_path.name, re.IGNORECASE)
    if source_match and csv_match and source_match.group(1) != csv_match.group(1):
        raise ValueError(
            f"Calibration {csv_path.name} is for round {csv_match.group(1)}, "
            f"but source {round_root.name} is round {source_match.group(1)}"
        )

def load_series_calibration(
    csv_path: Path, series: str,
) -> tuple[np.poly1d, float, float, dict[str, Any]]:
    """Select exactly one series row; retain old global compare.csv support."""
    csv_path = csv_path.resolve(strict=True)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        if "series" not in columns:
            polynomial, air_threshold, air_value, details = load_attenuation_calibration(csv_path)
            details["calibration_scope"] = "global_legacy"
            details["calibration_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            return polynomial, air_threshold, air_value, details
        missing_columns = set(REQUIRED_COLUMNS) - set(columns)
        if missing_columns:
            raise ValueError(f"Missing calibration columns in {csv_path}: {sorted(missing_columns)}")
        matches = [row for row in reader if row["series"].strip() == series]
    if len(matches) != 1:
        raise ValueError(f"Expected one calibration row for {series!r} in {csv_path}; found {len(matches)}")
    row = matches[0]
    numeric = {}
    for name in REQUIRED_COLUMNS[1:-1]:
        numeric[name] = float(row[name])
        if not np.isfinite(numeric[name]):
            raise ValueError(f"Nonfinite {name} for {series!r} in {csv_path}")
    degree = int(numeric["fit_design_degree"])
    if degree not in (1, 2):
        raise ValueError(f"Invalid fit_design_degree={degree} for {series!r}")
    coefficients = [numeric[name] for name in ("poly_a2", "poly_a1", "poly_a0")]
    polynomial = np.poly1d(coefficients)
    air_value = numeric["truth_air"]
    air_threshold = (air_value + numeric["truth_tissue"]) / 2.0
    # A row-specific digest avoids invalidating unrelated series when another
    # row in the same round CSV changes.
    row_digest = hashlib.sha256(json.dumps(
        {name: row[name] for name in REQUIRED_COLUMNS},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    details = {
        "source_csv": str(csv_path),
        "series": series,
        "calibration_scope": "per_series",
        "calibration_sha256": row_digest,
        "input_means_public_dataset": [numeric[f"prior_{name}"] for name in ("contrast", "tissue", "air")],
        "target_means_measured": [numeric[f"truth_{name}"] for name in ("contrast", "tissue", "air")],
        "polynomial_degree": 2,
        "fit_design_degree": degree,
        "polynomial_coefficients": coefficients,
        "air_threshold": air_threshold,
        "air_value": air_value,
        "contrast_residual_mu": numeric["contrast_residual_mu"],
        "warnings": row["warnings"],
    }
    return polynomial, air_threshold, air_value, details

def apply_reconstruction_calibration(image: np.ndarray, polynomial: np.poly1d) -> np.ndarray:
    """Apply the reviewed fit after FDK, preserving exact zero background."""
    result = np.asarray(image, dtype=np.float32)
    if result.ndim != 3:
        raise ValueError(f"Expected 3-D reconstructed image; got {result.shape}")
    if not result.flags.writeable:
        result = result.copy()
    for z0 in range(0, result.shape[0], 16):
        z1 = min(z0 + 16, result.shape[0])
        raw = result[z0:z1].copy()
        mapped = np.polyval(polynomial, raw).astype(np.float32)
        mapped[raw == 0] = 0.0
        result[z0:z1] = mapped
    return result
