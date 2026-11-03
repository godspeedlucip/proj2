"""Generate a calibrated ground-truth volume for each uniform-cycle phase.

The output grid follows ``generate_round1_contrast_recon.py``: the same
per-series geometry, source-voxel FOV shift, circular FOV mask, 256-pixel
transverse crop or resize, voxel spacing, and TIFF/NIfTI formats. No
projection or FDK is performed. Output names are ``phase_XX_gt`` under
``outputs/round1_contrast_gt/<series>``.
When ``fov_source_pixels`` is positive in the settings CSV, GT is sampled
directly on the same wider fixed-size physical grid as reconstruction.

Examples::

    python generate_round1_contrast_gt.py --series Stent_2_7596 --list
    python generate_round1_contrast_gt.py --series Stent_2_7596
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, zoom

if __package__:
    from .uniform_prior import load_attenuation_calibration
    from .uniform_recon import (
        DEFAULT_ROUND_ROOT, DEFAULT_SETTINGS_CSV, REPO_ROOT, SeriesSettings,
        calibrated_volume, discover_series, load_series_settings, save_image,
        select_series, settings_signature,
    )
else:
    from uniform_prior import load_attenuation_calibration
    from uniform_recon import (
        DEFAULT_ROUND_ROOT, DEFAULT_SETTINGS_CSV, REPO_ROOT, SeriesSettings,
        calibrated_volume, discover_series, load_series_settings, save_image,
        select_series, settings_signature,
    )


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "gt"
DEFAULT_RECON_ROOT = REPO_ROOT / "outputs" / "recon"


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
) -> tuple[np.ndarray, tuple[float, float, float]]:
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
        sampled = np.full(tuple(output_shape), air_value, np.float32)
    else:
        source = np.asarray(hu[tuple(slice(int(a), int(b)) for a, b in zip(start, stop))], np.float32)
        calibrated = calibrated_volume(source, polynomial, air_threshold, air_value)
        sampled = affine_transform(
            calibrated,
            matrix=np.diag(source_step),
            offset=offset - start,
            output_shape=tuple(int(v) for v in output_shape),
            order=1,
            mode="constant",
            cval=float(air_value),
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
    sampled = np.where(mask, sampled, 0).astype(np.float32, copy=False)

    width = settings.fov_output_pixels
    if settings.fov_source_pixels:
        return sampled, tuple(float(v) for v in output_spacing)
    if min(ny, nx) < width:
        sampled = zoom(
            sampled, (1.0, width / ny, width / nx),
            order=1, mode="nearest", prefilter=False,
        )
        if sampled.shape[1:] != (width, width):
            sampled = sampled[:, :width, :width]
            pad_y, pad_x = width - sampled.shape[1], width - sampled.shape[2]
            sampled = np.pad(
                sampled,
                ((0, 0), (pad_y // 2, pad_y - pad_y // 2),
                 (pad_x // 2, pad_x - pad_x // 2)),
            )
        output_spacing[1] *= ny / width
        output_spacing[2] *= nx / width
    else:
        y_start, x_start = (ny - width) // 2, (nx - width) // 2
        sampled = sampled[:, y_start:y_start + width, x_start:x_start + width]
    return sampled, tuple(float(v) for v in output_spacing)


def process_series(
    series_dir: Path,
    phase_paths: dict[int, Path],
    output_dir: Path,
    polynomial: np.poly1d,
    air_threshold: float,
    air_value: float,
    settings: SeriesSettings,
    *,
    reconstruction_dir: Path | None = None,
) -> dict[str, Any]:
    """Write one GT pair for every source phase and check any sparse partner."""
    phase_shapes: dict[str, list[int]] = {}
    source_shape = None
    source_spacing = None
    output_spacing = None
    for phase, path in phase_paths.items():
        image = nib.load(str(path))
        shape = tuple(int(v) for v in image.shape)
        spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
        if source_shape is None:
            source_shape, source_spacing = shape, spacing
        elif shape != source_shape or not np.allclose(spacing, source_spacing, rtol=0, atol=1e-5):
            raise ValueError(f"Phase shape or spacing differs from first phase: {path}")
        gt, output_spacing = make_phase_gt(
            image.dataobj, spacing, settings, polynomial, air_threshold, air_value
        )
        sparse_path = (
            reconstruction_dir / f"phase_{phase:02d}_reconstruction.nii.gz"
            if reconstruction_dir is not None else None
        )
        if sparse_path is not None and sparse_path.is_file():
            sparse = nib.load(str(sparse_path))
            sparse_spacing = sparse.header.get_zooms()[:3]
            if sparse.shape != gt.shape or not np.allclose(
                sparse_spacing, output_spacing, rtol=0, atol=1e-5
            ):
                raise ValueError(f"GT shape or spacing differs from sparse reconstruction: {sparse_path}")
        save_image(gt, output_dir, f"phase_{phase:02d}_gt", output_spacing)
        phase_shapes[str(phase)] = list(gt.shape)
        print(f"  phase {phase:02d}: GT {gt.shape}", flush=True)
        del image, gt
        gc.collect()
    return {
        "status": "complete",
        "source_directory": str(series_dir),
        "phase_files": {str(k): str(v) for k, v in phase_paths.items()},
        "volume_shape_zyx": list(source_shape),
        "original_voxel_spacing_mm_zyx": list(source_spacing),
        "gt_voxel_spacing_mm_zyx": list(output_spacing),
        "phase_shapes_zyx": phase_shapes,
        "fov_shift_voxels_down_right": [
            settings.shift_down_voxels - settings.shift_up_voxels,
            settings.shift_right_voxels - settings.shift_left_voxels,
        ],
        "settings": asdict(settings),
    }


def output_paths(output_dir: Path, phases: tuple[int, ...]) -> list[Path]:
    return [output_dir / f"phase_{phase:02d}_gt{suffix}"
            for phase in phases for suffix in (".tif", ".nii.gz")]


def signature(
    settings: SeriesSettings, calibration_csv: Path, phase_paths: dict[int, Path]
) -> dict[str, Any]:
    return {
        "settings": settings_signature(settings),
        "calibration_csv": {
            "path": str(calibration_csv.resolve()),
            "sha256": hashlib.sha256(calibration_csv.read_bytes()).hexdigest(),
        },
        "phase_files": {
            str(phase): {"path": str(path.resolve()), "size": path.stat().st_size,
                         "mtime_ns": path.stat().st_mtime_ns}
            for phase, path in phase_paths.items()
        },
    }


def outputs_current(output_dir: Path, phases: tuple[int, ...], config: dict[str, Any]) -> bool:
    if not all(path.is_file() and path.stat().st_size > 0 for path in output_paths(output_dir, phases)):
        return False
    config_path = output_dir / "gt_config.json"
    try:
        return json.loads(config_path.read_text(encoding="utf-8")) == config
    except (OSError, ValueError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-root", type=Path, default=DEFAULT_ROUND_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--recon-root", type=Path, default=DEFAULT_RECON_ROOT)
    parser.add_argument("--calibration-csv", type=Path, default=REPO_ROOT / "settings" / "compare.csv")
    parser.add_argument("--settings-csv", type=Path, default=DEFAULT_SETTINGS_CSV)
    parser.add_argument("--series", nargs="+", metavar="NAME",
                        help="Pig IDs or relative series paths (space or comma separated).")
    parser.add_argument("--list", action="store_true", help="List source series without generating GT.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing GT pairs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    round_root = args.round_root.resolve()
    series = select_series(discover_series(round_root), round_root, args.series)
    if not series:
        raise FileNotFoundError(f"No contrast-enhanced CT series found under {round_root}")
    settings_by_series = load_series_settings(args.settings_csv.resolve())
    for directory, paths in series:
        relative = directory.relative_to(round_root).as_posix()
        if relative not in settings_by_series:
            raise ValueError(f"Settings CSV is missing series: {relative}")
        print(f"{relative}: {len(paths)} phases {tuple(paths)}")
    if args.list:
        return 0

    calibration_path = args.calibration_csv.resolve()
    polynomial, air_threshold, air_value, calibration_details = load_attenuation_calibration(calibration_path)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {
        "round_root": str(round_root), "settings_csv": str(args.settings_csv.resolve()),
        "attenuation_calibration": calibration_details, "series": [],
    }
    failures = 0
    for directory, phase_paths in series:
        relative = directory.relative_to(round_root)
        destination = output_root / relative
        reconstruction_dir = args.recon_root.resolve() / relative
        settings = settings_by_series[relative.as_posix()]
        try:
            config = signature(settings, calibration_path, phase_paths)
            recon_config_path = reconstruction_dir / "reconstruction_config.json"
            if recon_config_path.is_file():
                recon_config = json.loads(recon_config_path.read_text(encoding="utf-8"))
                if (recon_config.get("settings") != settings_signature(settings)
                        or recon_config.get("calibration_csv") != config["calibration_csv"]):
                    raise ValueError(
                        f"Reconstruction settings or calibration differ from current GT inputs: {recon_config_path}"
                    )
            if not args.overwrite and outputs_current(destination, tuple(phase_paths), config):
                record: dict[str, Any] = {"status": "skipped_existing"}
            else:
                print(f"Processing {relative}", flush=True)
                record = process_series(
                    directory, phase_paths, destination,
                    polynomial, air_threshold, air_value, settings,
                    reconstruction_dir=reconstruction_dir,
                )
                (destination / "gt_config.json").write_text(
                    json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        except Exception as exc:
            failures += 1
            record = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            print(f"Failed {relative}: {exc}", flush=True)
        record["source_directory"] = str(directory)
        record["output_directory"] = str(destination)
        record["settings"] = asdict(settings)
        manifest["series"].append(record)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        gc.collect()
    print(f"Finished {len(series)} series; {failures} failed. Manifest: {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
