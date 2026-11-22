"""Generate per-phase GT by resampling the source CT on the CPU."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, zoom

from generate_round1_contrast_recon import (
    INTENSITY_PROCESSING,
    OUTPUT_SUFFIXES,
    POST_CALIBRATION_PROCESSING,
    SOURCE_PADDING_HU,
    SeriesSettings,
    calibrated_volume,
    discover_series,
    load_series_settings,
    save_selected_image,
    select_series,
    settings_signature,
)
from series_calibration import (
    apply_reconstruction_calibration,
    load_series_calibration,
    validate_calibration_round,
)


GT_GENERATION = "source_ct_resampling_v1"
POST_RESAMPLING_PROCESSING = "polynomial_post_resampling_zero_preserved_v1"
IDENTITY_POLYNOMIAL = np.poly1d([1.0, 0.0])


def resize_transverse(volume: np.ndarray, width: int) -> np.ndarray:
    """Resize the transverse plane to the requested matrix size."""
    ny, nx = volume.shape[1:]
    resized = zoom(
        volume, (1.0, width / ny, width / nx), order=1, mode="nearest", prefilter=False,
    )[:, :width, :width]
    pad_y, pad_x = width - resized.shape[1], width - resized.shape[2]
    if pad_y or pad_x:
        resized = np.pad(
            resized,
            ((0, 0), (pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)),
        )
    return resized


def sample_phase_gt(
    data: Any, spacing: tuple[float, ...], settings: SeriesSettings,
) -> tuple[np.ndarray, tuple[float, ...], np.ndarray]:
    """Return attenuation, voxel spacing and the valid mask on the output grid."""
    source_shape = np.asarray(data.shape)
    source_spacing = np.asarray(spacing, dtype=np.float64)
    if source_shape.size != 3 or source_spacing.size != 3:
        raise ValueError("Expected a 3-D CT volume and three voxel spacings")
    if not np.all(np.isfinite(source_spacing) & (source_spacing > 0)):
        raise ValueError("Source voxel spacing must be finite and positive")

    detector_fov = (
        np.asarray(settings.n_detector, dtype=np.float64)
        * np.asarray(settings.d_detector_mm, dtype=np.float64)
        * settings.dso_mm / settings.dsd_mm
    )
    voxel_spacing = source_spacing * settings.voxel_spacing_multiplier
    fov_axial, fov_transverse = detector_fov
    fov_extent = np.array((fov_axial, fov_transverse, fov_transverse))
    shape = np.clip(np.floor(fov_extent / voxel_spacing).astype(int), 1, source_shape)
    shift = np.array((
        0,
        settings.shift_down_voxels - settings.shift_up_voxels,
        settings.shift_right_voxels - settings.shift_left_voxels,
    ))
    width = settings.fov_output_pixels
    step = np.ones(3)
    if settings.fov_source_pixels:
        if np.any(settings.fov_source_pixels * voxel_spacing[1:] > fov_transverse + 1e-6):
            raise ValueError("Requested transverse FOV exceeds the unpadded detector FOV")
        shape[1:] = width
        step[1:] = settings.fov_source_pixels / width
        offset = (source_shape - 1) / 2 + shift - (shape - 1) / 2 * step
    else:
        offset = (source_shape - shape) / 2 + shift
    output_spacing = voxel_spacing * step

    # A NIfTI proxy can load only the source slab needed for interpolation.
    start = np.maximum(0, np.floor(offset).astype(int))
    stop = np.minimum(source_shape, np.ceil(offset + step * (shape - 1)).astype(int) + 1)
    if np.any(stop <= start):
        gt = np.zeros(tuple(shape), dtype=np.float32)
        weights = np.zeros_like(gt)
    else:
        slab = np.asarray(data[tuple(slice(a, b) for a, b in zip(start, stop))], dtype=np.float32)
        mu = calibrated_volume(slab, IDENTITY_POLYNOMIAL, 0.0, 0.0, apply_air_threshold=False)
        sampling = {
            "matrix": np.diag(step), "offset": offset - start, "output_shape": tuple(shape),
            "order": 1, "mode": "constant", "cval": 0.0, "prefilter": False,
            "output": np.float32,
        }
        gt = affine_transform(mu, **sampling)
        weights = affine_transform((slab != SOURCE_PADDING_HU).astype(np.float32), **sampling)

    ny, nx = gt.shape[1:]
    z, y, x = [
        (np.arange(n) - (n - 1) / 2) * spacing_mm
        for n, spacing_mm in zip(gt.shape, output_spacing)
    ]
    radius_y, radius_x = fov_transverse / 2 - settings.fov_mask_shrink_pixels / 2 * output_spacing[1:]
    if min(radius_y, radius_x) <= 0:
        raise ValueError("FOV mask radius is not positive; check detector geometry")
    inside_fov = (
        (np.abs(z)[:, None, None] <= fov_axial / 2)
        & ((y[None, :, None] / radius_y) ** 2 + (x[None, None, :] / radius_x) ** 2 <= 1)
    )
    gt[~inside_fov] = 0.0
    weights *= inside_fov

    if not settings.fov_source_pixels:
        if min(ny, nx) < width:
            gt = resize_transverse(gt, width)
            weights = resize_transverse(weights, width)
            output_spacing[1:] *= (ny / width, nx / width)
        else:
            y0, x0 = (ny - width) // 2, (nx - width) // 2
            gt = gt[:, y0:y0 + width, x0:x0 + width]
            weights = weights[:, y0:y0 + width, x0:x0 + width]

    valid = weights >= 1.0 - 1e-6
    gt[~valid] = 0.0
    return gt, tuple(map(float, output_spacing)), valid.astype(np.uint8)


def gt_stems(phase: int) -> tuple[str, str]:
    return f"phase_{phase:02d}_gt", f"phase_{phase:02d}_gt_valid_mask"


def check_reconstruction(
    directory: Path, phase: int, shape: tuple[int, ...], spacing: tuple[float, ...],
) -> None:
    stem = f"phase_{phase:02d}_reconstruction"
    nifti = directory / f"{stem}.nii.gz"
    if nifti.is_file():
        image = nib.load(str(nifti))
        if image.shape != shape or not np.allclose(image.header.get_zooms()[:3], spacing, rtol=0, atol=1e-5):
            raise ValueError(f"GT shape or spacing differs from sparse reconstruction: {nifti}")
        return

    tiff = directory / f"{stem}.tif"
    if tiff.is_file():
        from tifffile import TiffFile

        # TIFF outputs from this pipeline carry no voxel spacing.
        with TiffFile(tiff) as image:
            if image.series[0].shape != shape:
                raise ValueError(f"GT shape differs from sparse reconstruction: {tiff}")


def generate_series(
    phases: dict[int, Path], destination: Path, settings: SeriesSettings,
    output_format: str, polynomial: np.poly1d | None, reconstruction: Path | None,
) -> dict[str, Any]:
    source_shape = source_spacing = None
    phase_shapes = {}
    for phase, path in phases.items():
        ct = nib.load(str(path))
        spacing = ct.header.get_zooms()[:3]
        if source_shape is None:
            source_shape, source_spacing = ct.shape, spacing
        elif ct.shape != source_shape or not np.allclose(spacing, source_spacing, rtol=0, atol=1e-5):
            raise ValueError(f"Phase shape or spacing differs from first phase: {path}")
        gt, output_spacing, valid = sample_phase_gt(ct.dataobj, spacing, settings)
        if polynomial is not None:
            gt = apply_reconstruction_calibration(gt, polynomial)
        if reconstruction is not None:
            check_reconstruction(reconstruction, phase, gt.shape, output_spacing)
        for stem, image in zip(gt_stems(phase), (gt, valid)):
            save_selected_image(image, destination, stem, output_spacing, output_format)
        phase_shapes[str(phase)] = list(gt.shape)
        print(f"  phase {phase:02d}: GT {gt.shape}", flush=True)
        del ct, gt, valid, image

    return {
        "status": "complete",
        "gt_generation": GT_GENERATION,
        "gt_views_per_phase": 0,
        "output_format": output_format,
        "calibration_application": "post_resampling" if polynomial is not None else "none",
        "phase_files": {str(phase): str(path) for phase, path in phases.items()},
        "phase_shapes_zyx": phase_shapes,
        "volume_shape_zyx": list(source_shape),
        "original_voxel_spacing_mm_zyx": list(map(float, source_spacing)),
        "gt_voxel_spacing_mm_zyx": list(output_spacing),
        "fov_shift_voxels_down_right": [
            settings.shift_down_voxels - settings.shift_up_voxels,
            settings.shift_right_voxels - settings.shift_left_voxels,
        ],
    }


def configuration(
    settings: SeriesSettings, phases: dict[int, Path], output_format: str,
    calibration: dict[str, str] | None,
) -> dict[str, Any]:
    phase_files = {}
    for phase, path in phases.items():
        metadata = path.stat()
        phase_files[str(phase)] = {
            "path": str(path.resolve()), "size": metadata.st_size, "mtime_ns": metadata.st_mtime_ns,
        }
    return {
        "settings": settings_signature(settings),
        "gt_generation": GT_GENERATION,
        "output_format": output_format,
        "intensity_processing": POST_RESAMPLING_PROCESSING if calibration else INTENSITY_PROCESSING,
        "source_out_of_bounds_value": 0.0,
        "invalid_source_hu": SOURCE_PADDING_HU,
        "invalid_attenuation_value": 0.0,
        "valid_mask_rule": "all_interpolation_sources_valid_and_inside_output_fov",
        "calibration_csv": calibration,
        "phase_files": phase_files,
    }


def outputs_current(config_path: Path, config: dict[str, Any]) -> bool:
    try:
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            return False
        paths = (
            config_path.parent / f"{stem}{suffix}"
            for phase in config["phase_files"]
            for stem in gt_stems(int(phase))
            for suffix in OUTPUT_SUFFIXES[config["output_format"]]
        )
        return all(path.is_file() and path.stat().st_size > 0 for path in paths)
    except (OSError, ValueError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-root", type=Path, required=True)
    parser.add_argument("--settings-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--recon-root", type=Path, help="Optional sparse reconstruction grid to verify.")
    parser.add_argument("--output-format", choices=tuple(OUTPUT_SUFFIXES), default="tif")
    parser.add_argument("--series", nargs="+", metavar="NAME", help="Pig IDs or relative series paths.")
    parser.add_argument("--list", action="store_true", help="List source series without generating GT.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing GT pairs.")
    calibration = parser.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--calibration-csv", type=Path, help="Apply calibration after resampling.")
    calibration.add_argument("--no-calibration", action="store_true", help="Use uncalibrated HU-to-mu conversion.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    round_root = args.round_root.resolve()
    settings_path = args.settings_csv.resolve()
    settings_by_series = load_series_settings(settings_path)
    series = select_series(discover_series(round_root), round_root, args.series)
    if not series:
        raise FileNotFoundError(f"No contrast-enhanced CT series found under {round_root}")
    for directory, phases in series:
        relative = directory.relative_to(round_root).as_posix()
        if relative not in settings_by_series:
            raise ValueError(f"Settings CSV is missing series: {relative}")
        print(f"{relative}: {len(phases)} phases {tuple(phases)}")
    if args.list:
        return 0

    calibration_path = args.calibration_csv.resolve(strict=True) if args.calibration_csv else None
    validate_calibration_round(calibration_path, round_root)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    recon_root = args.recon_root.resolve() if args.recon_root else None
    manifest_path = output_root / "manifest.json"
    manifest = {
        "round_root": str(round_root),
        "settings_csv": str(settings_path),
        "gt_generation": GT_GENERATION,
        "attenuation_calibration": {
            "source_csv": str(calibration_path) if calibration_path else None,
            "mode": "requested" if calibration_path else "none",
        },
        "series": [],
    }
    failures = 0
    for directory, phases in series:
        relative = directory.relative_to(round_root)
        settings = settings_by_series[relative.as_posix()]
        destination = output_root / relative
        reconstruction = recon_root / relative if recon_root is not None else None
        details = None
        try:
            polynomial = calibration = None
            if calibration_path is None:
                details = {
                    "calibration_scope": "none", "source_csv": None,
                    "polynomial_coefficients": [1.0, 0.0],
                }
            else:
                polynomial, _, _, details = load_series_calibration(calibration_path, relative.as_posix())
                calibration = {"path": str(calibration_path), "sha256": details["calibration_sha256"]}
            details["air_threshold_applied"] = False
            details["application"] = "post_resampling" if polynomial is not None else "none"
            config = configuration(settings, phases, args.output_format, calibration)
            if reconstruction is not None:
                recon_config_path = reconstruction / "reconstruction_config.json"
                if recon_config_path.is_file():
                    recon_config = json.loads(recon_config_path.read_text(encoding="utf-8"))
                    expected = {
                        "settings": config["settings"],
                        "intensity_processing": POST_CALIBRATION_PROCESSING if calibration else INTENSITY_PROCESSING,
                        "calibration_csv": calibration,
                    }
                    if any(recon_config.get(key) != value for key, value in expected.items()):
                        raise ValueError(
                            f"Reconstruction settings or calibration differ from current GT inputs: {recon_config_path}"
                        )
            config_path = destination / "gt_config.json"
            if not args.overwrite and outputs_current(config_path, config):
                record = {"status": "skipped_existing"}
            else:
                print(f"Processing {relative}", flush=True)
                # A failed or interrupted regeneration must not keep an old completion marker.
                config_path.unlink(missing_ok=True)
                record = generate_series(phases, destination, settings, args.output_format, polynomial, reconstruction)
                config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            failures += 1
            record = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            print(f"Failed {relative}: {exc}", flush=True)
        record.update(
            source_directory=str(directory), output_directory=str(destination),
            attenuation_calibration=details, settings=asdict(settings),
        )
        manifest["series"].append(record)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Finished {len(series)} series; {failures} failed. Manifest: {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
