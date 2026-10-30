"""Generate FDK prior images from the first contrast-enhanced phase per pig series.

The processing settings follow ``4dlung_projection_demo.ipynb``. By default,
the script scans ``data/pig/AHFP-Round-*`` for contrast-enhanced series whose
first phase is ``IM00001_ct.nii.gz`` and writes a TIFF and NIfTI prior image
for each series under ``outputs/pig_first_phase_prior``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import zoom


CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.tool import MedicalImageTool, ProjectionTool, TigreTool  # noqa: E402


PHASE_FILENAME = "IM00001_ct.nii.gz"
OUTPUT_RELATIVE_ROOT = Path("outputs") / "pig_first_phase_prior"

# Values match the first-phase projection section of the notebook.
VOXEL_SPACING_MULTIPLIER = 1.0
N_PROJECTIONS = 100
ANGLE_SPAN_DEG = 300.0
N_DETECTOR = (476, 647)
D_DETECTOR_MM = (0.616, 0.616)
DSO_MM = 750.0
DSD_MM = 1250.0
PADDING_ROW_PIXELS = 100
PADDING_COL_PIXELS = 100
FOV_OUTPUT_TRANSVERSE_PIXELS = 256
FOV_MASK_DIAMETER_SHRINK_PIXELS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project and FDK-reconstruct the first contrast-enhanced CT phase "
            "for each pig series in every AHFP round."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "data" / "pig",
        help="Root containing AHFP-Round-* directories (default: data/pig).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / OUTPUT_RELATIVE_ROOT,
        help="Directory for prior images and the processing manifest.",
    )
    parser.add_argument(
        "--calibration-csv",
        type=Path,
        default=REPO_ROOT.parent / "compare.csv",
        help="Notebook attenuation calibration CSV (default: ../compare.csv).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs that already have both TIFF and NIfTI files.",
    )
    return parser.parse_args()


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


def find_contrast_series(input_root: Path) -> list[Path]:
    """Find first-phase files under contrast-enhanced series in AHFP rounds."""
    if not input_root.is_dir():
        raise FileNotFoundError(f"Pig data root not found: {input_root}")
    matches = []
    for path in input_root.glob("AHFP-Round-*/**/" + PHASE_FILENAME):
        if any("with_contrast" in part.lower() for part in path.parts):
            matches.append(path)
    return sorted(matches)


def pad_projections(projections: np.ndarray) -> np.ndarray:
    """Apply the same two detector-axis paddings used before notebook FDK."""
    padded = ProjectionTool.pad_linatech(projections, padding=PADDING_ROW_PIXELS)
    padded = np.transpose(padded, (0, 2, 1))
    padded = ProjectionTool.pad_linatech(padded, padding=PADDING_COL_PIXELS)
    return np.transpose(padded, (0, 2, 1))


def reconstruct_prior(
    projections: np.ndarray,
    geometry: Any,
    angles: np.ndarray,
    tigre_tool: TigreTool,
    *,
    padding_row_pixels: int = PADDING_ROW_PIXELS,
    padding_col_pixels: int = PADDING_COL_PIXELS,
    fov_output_transverse_pixels: int = FOV_OUTPUT_TRANSVERSE_PIXELS,
    fov_mask_diameter_shrink_pixels: int = FOV_MASK_DIAMETER_SHRINK_PIXELS,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Run FDK and apply the notebook's circular FOV mask and center crop."""
    if tuple(projections.shape[1:]) != tuple(int(v) for v in geometry.nDetector):
        raise ValueError(
            f"Projection detector shape {projections.shape[1:]} does not match "
            f"geometry {tuple(int(v) for v in geometry.nDetector)}"
        )

    if padding_row_pixels < 0 or padding_col_pixels < 0:
        raise ValueError("Projection padding must be non-negative")
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

    fov_geometry = tigre_tool.set_geometry(
        volume_shape=tuple(int(v) for v in fov_shape),
        voxel_spacing=tuple(float(v) for v in geometry.dVoxel),
        n_detector=tuple(int(v) for v in padded_detector),
        d_detector=tuple(float(v) for v in geometry.dDetector),
        dso=float(geometry.DSO),
        dsd=float(geometry.DSD),
        mode=getattr(geometry, "mode", "cone"),
    )
    reconstruction = TigreTool.reconstruct_tigre(
        projections=padded,
        geometry=fov_geometry,
        angles=angles,
    )

    nz, ny, nx = reconstruction.shape
    z_mm = (np.arange(nz, dtype=np.float64) - (nz - 1) / 2) * float(geometry.dVoxel[0])
    y_mm = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2) * float(geometry.dVoxel[1])
    x_mm = (np.arange(nx, dtype=np.float64) - (nx - 1) / 2) * float(geometry.dVoxel[2])
    radius_y = object_fov_mm[1] / 2 - (fov_mask_diameter_shrink_pixels / 2) * float(geometry.dVoxel[1])
    radius_x = object_fov_mm[1] / 2 - (fov_mask_diameter_shrink_pixels / 2) * float(geometry.dVoxel[2])
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


def process_series(
    source_path: Path,
    output_dir: Path,
    *,
    polynomial: np.poly1d,
    air_threshold: float,
    air_value: float,
) -> dict[str, Any]:
    volume_image = MedicalImageTool.read(source_path)
    volume_hu = np.asarray(volume_image.data, dtype=np.float32)
    if volume_hu.ndim != 3:
        raise ValueError(f"Expected a 3-D CT volume, got {volume_hu.shape}: {source_path}")
    spacing_original = tuple(
        float(value) for value in volume_image.metadata["header"].get_zooms()[:3]
    )
    spacing_used = tuple(value * VOXEL_SPACING_MULTIPLIER for value in spacing_original)

    tigre_tool = TigreTool()
    geometry = tigre_tool.set_geometry(
        volume_shape=volume_hu.shape,
        voxel_spacing=spacing_used,
        n_detector=N_DETECTOR,
        d_detector=D_DETECTOR_MM,
        dso=DSO_MM,
        dsd=DSD_MM,
    )
    angles = tigre_tool.set_angles(0.0, ANGLE_SPAN_DEG, N_PROJECTIONS)
    volume_mu = calibrate_attenuation(
        TigreTool.hu_to_mu(volume_hu), polynomial, air_threshold, air_value
    )
    projections = TigreTool.project_tigre(volume_mu, geometry, angles)
    prior_image, output_spacing = reconstruct_prior(projections, geometry, angles, tigre_tool)

    output_dir.mkdir(parents=True, exist_ok=True)
    tiff_path = output_dir / "prior_image_phase01.tif"
    nifti_path = output_dir / "prior_image_phase01.nii.gz"
    MedicalImageTool.save(prior_image, tiff_path)
    MedicalImageTool.save(prior_image, nifti_path, voxel_spacing=output_spacing)
    return {
        "status": "complete",
        "source": str(source_path),
        "output_tiff": str(tiff_path),
        "output_nifti": str(nifti_path),
        "volume_shape_zyx": list(volume_hu.shape),
        "voxel_spacing_original_mm_zyx": list(spacing_original),
        "voxel_spacing_used_mm_zyx": list(spacing_used),
        "projection_shape": list(projections.shape),
        "prior_image_shape": list(prior_image.shape),
        "prior_image_voxel_spacing_mm_zyx": list(output_spacing),
        "prior_image_min": float(np.min(prior_image)),
        "prior_image_max": float(np.max(prior_image)),
    }


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    calibration_path = args.calibration_csv.resolve()
    polynomial, air_threshold, air_value, calibration_details = load_attenuation_calibration(
        calibration_path
    )
    source_paths = find_contrast_series(input_root)
    if not source_paths:
        raise FileNotFoundError(
            f"No {PHASE_FILENAME} files under contrast-enhanced series in {input_root}"
        )

    manifest_path = output_root / "prior_image_manifest.json"
    manifest: dict[str, Any] = {
        "source_root": str(input_root),
        "output_root": str(output_root),
        "phase_filename": PHASE_FILENAME,
        "phase_label": 1,
        "attenuation_calibration": calibration_details,
        "settings": {
            "voxel_spacing_multiplier": VOXEL_SPACING_MULTIPLIER,
            "n_projections": N_PROJECTIONS,
            "angle_start_deg": 0.0,
            "angle_span_deg": ANGLE_SPAN_DEG,
            "angle_endpoint_included": False,
            "n_detector_yx": list(N_DETECTOR),
            "d_detector_mm_yx": list(D_DETECTOR_MM),
            "dso_mm": DSO_MM,
            "dsd_mm": DSD_MM,
            "padding_row_pixels": PADDING_ROW_PIXELS,
            "padding_col_pixels": PADDING_COL_PIXELS,
            "fov_output_transverse_pixels": FOV_OUTPUT_TRANSVERSE_PIXELS,
        },
        "series": [],
    }
    output_root.mkdir(parents=True, exist_ok=True)

    for source_path in source_paths:
        relative_path = source_path.relative_to(input_root)
        series_output_dir = output_root / relative_path.parent
        tiff_path = series_output_dir / "prior_image_phase01.tif"
        nifti_path = series_output_dir / "prior_image_phase01.nii.gz"
        if not args.overwrite and tiff_path.is_file() and nifti_path.is_file():
            record = {
                "status": "skipped_existing",
                "source": str(source_path),
                "output_tiff": str(tiff_path),
                "output_nifti": str(nifti_path),
            }
            manifest["series"].append(record)
            print(f"Skip existing: {relative_path}")
            continue

        try:
            record = process_series(
                source_path,
                series_output_dir,
                polynomial=polynomial,
                air_threshold=air_threshold,
                air_value=air_value,
            )
            print(f"Complete: {relative_path} -> {record['prior_image_shape']}")
        except Exception as exc:
            record = {
                "status": "failed",
                "source": str(source_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"Failed: {relative_path}: {type(exc).__name__}: {exc}")
        finally:
            gc.collect()
        manifest["series"].append(record)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    completed = sum(item["status"] == "complete" for item in manifest["series"])
    skipped = sum(item["status"] == "skipped_existing" for item in manifest["series"])
    failed = [item for item in manifest["series"] if item["status"] == "failed"]
    print(
        f"Finished: {completed} generated, {skipped} skipped, {len(failed)} failed "
        f"out of {len(source_paths)} series. Manifest: {manifest_path}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
