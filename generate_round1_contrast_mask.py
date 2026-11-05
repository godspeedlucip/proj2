"""Generate class-preserving masks on the Round-1 GT/reconstruction grid.

The source is ``data/pig/AHFP-Round-1/<series>/IMxxxxx_mask.nii.gz``.
Only phases with a source mask are written. Class IDs are preserved; mask
resampling supports nearest, linear and cubic B-spline interpolation (the
``polynomial`` option is an alias for cubic). The
linear/cubic methods interpolate each foreground class separately, threshold
the scores, and select the strongest class. The default is linear interpolation
with a 0.5 threshold. Grid/FOV/crop rules mirror
``generate_round1_contrast_gt.py``. No masks are copied across phases.

Examples::

    python generate_round1_contrast_mask.py --series Control_1_7569 --list
    python generate_round1_contrast_mask.py --series Control_1_7569
    python generate_round1_contrast_mask.py --series Control_1_7569 --phase 1 \
        --mask-resampling polynomial --mask-threshold 0.5
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, zoom


REPO_ROOT = Path(__file__).resolve().parent
if not (REPO_ROOT / "generate_round1_contrast_gt.py").is_file():
    REPO_ROOT = REPO_ROOT.parent  # permits staging/testing from data/
sys.path.insert(0, str(REPO_ROOT))

from generate_round1_contrast_gt import phase_grid  # noqa: E402
from generate_round1_contrast_recon import (  # noqa: E402
    DEFAULT_ROUND_ROOT,
    DEFAULT_SETTINGS_CSV,
    SeriesSettings,
    discover_series,
    load_series_settings,
    save_image,
    select_series,
    settings_signature,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "round1_contrast_mask"
DEFAULT_GT_ROOT = REPO_ROOT / "outputs" / "round1_contrast_gt"
DEFAULT_RECON_ROOT = REPO_ROOT / "outputs" / "round1_contrast_recon"
MASK_NAME = re.compile(r"IM(\d+)_mask\.nii\.gz", re.IGNORECASE)


def source_masks(series_dir: Path, ct_paths: dict[int, Path], phase: int | None = None) -> dict[int, Path]:
    """Return only annotated phases; never synthesize an unlabelled phase."""
    result: dict[int, Path] = {}
    for path in sorted(series_dir.glob("IM*_mask.nii.gz")):
        match = MASK_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected mask filename: {path}")
        source_phase = int(match.group(1))
        if source_phase not in ct_paths:
            raise ValueError(f"Mask has no same-phase CT: {path}")
        if source_phase in result:
            raise ValueError(f"Duplicate mask for phase {source_phase}: {path}")
        result[source_phase] = path
    if phase is not None:
        if phase not in result:
            raise ValueError(f"No source mask for phase {phase} in {series_dir}")
        return {phase: result[phase]}
    return result


def _score_to_labels(scores, labels, threshold, shape):
    """Resolve per-class probabilities without creating fractional class IDs."""
    result = np.zeros(shape, np.int16)
    best = np.zeros(shape, np.float32)
    for label, score in zip(labels, scores):
        np.clip(score, 0, 1, out=score)
        choose = (score >= threshold) & (score > best)
        result[choose] = label
        best[choose] = score[choose]
    return result


def _sampling_grid(source_shape, source_spacing, settings):
    """Use the exact geometry formulas in make_phase_gt, without CT values."""
    fov_shape, used_spacing, offset = phase_grid(source_shape, source_spacing, settings)
    step = np.ones(3, np.float64)
    shape = fov_shape.copy()
    if settings.fov_source_pixels:
        detector_fov_mm = (
            np.asarray(settings.n_detector, np.float64)
            * np.asarray(settings.d_detector_mm, np.float64)
            * settings.dso_mm / settings.dsd_mm
        )
        if np.any(settings.fov_source_pixels * used_spacing[1:] > detector_fov_mm[1] + 1e-6):
            raise ValueError("Requested transverse FOV exceeds the unpadded detector FOV")
        shape[1:] = settings.fov_output_pixels
        step[1:] = settings.fov_source_pixels / settings.fov_output_pixels
        shifts = np.asarray([
            0,
            settings.shift_down_voxels - settings.shift_up_voxels,
            settings.shift_right_voxels - settings.shift_left_voxels,
        ], np.float64)
        offset = (np.asarray(source_shape, np.float64) - 1) / 2 + shifts - (shape - 1) / 2 * step
    return shape, used_spacing * step, step, offset


def make_phase_mask(
    source: Any,
    source_spacing: tuple[float, float, float],
    settings: SeriesSettings,
    *,
    resampling: str = "linear",
    threshold: float = 0.5,
    chunk_depth: int = 32,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Map one labelled volume to the paired Round-1 image grid.

    The input may be a nibabel proxy. Reading slabs limits memory use for
    the original float64 NIfTI segmentation volumes.
    """
    if chunk_depth < 1:
        raise ValueError("chunk_depth must be positive")
    if resampling not in ("nearest", "linear", "cubic", "polynomial"):
        raise ValueError("resampling must be nearest, linear, cubic or polynomial")
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    order = {"nearest": 0, "linear": 1, "cubic": 3, "polynomial": 3}[resampling]
    source_shape = tuple(int(v) for v in source.shape)
    if len(source_shape) != 3:
        raise ValueError("Source mask must be 3-D")
    shape, output_spacing, step, offset = _sampling_grid(source_shape, source_spacing, settings)
    sampled = np.zeros(tuple(int(v) for v in shape), dtype=np.int16)
    seen_labels: set[int] = set()
    for z0 in range(0, shape[0], chunk_depth):
        chunk_shape = np.asarray((min(chunk_depth, shape[0] - z0), shape[1], shape[2]), np.int32)
        chunk_offset = offset + np.asarray((z0 * step[0], 0, 0), np.float64)
        first = np.floor(chunk_offset).astype(int)
        last = np.ceil(chunk_offset + step * (chunk_shape - 1)).astype(int) + 1
        halo = 8 if order == 3 else 0
        start = np.maximum(first - halo, 0)
        stop = np.minimum(last + halo, source_shape)
        if np.any(stop <= start):
            continue
        slab = np.asarray(source[tuple(slice(int(a), int(b)) for a, b in zip(start, stop))])
        if not np.issubdtype(slab.dtype, np.number) or not np.isfinite(slab).all():
            raise ValueError("Source mask contains nonfinite or nonnumeric labels")
        if np.any(slab != np.rint(slab)):
            raise ValueError("Source mask labels must be integer valued")
        if np.any((slab < np.iinfo(np.int16).min) | (slab > np.iinfo(np.int16).max)):
            raise ValueError("Source mask labels exceed int16 range")
        seen_labels.update(int(v) for v in np.unique(slab))
        transform_kwargs = dict(
            matrix=np.diag(step), offset=chunk_offset - start,
            output_shape=tuple(int(v) for v in chunk_shape),
            order=order, mode="constant", cval=0, prefilter=(order == 3),
        )
        if order == 0:
            chunk = affine_transform(slab.astype(np.int16), output=np.int16, **transform_kwargs)
        else:
            labels = sorted(int(v) for v in np.unique(slab) if v != 0)
            scores = (
                affine_transform((slab == label).astype(np.float32), output=np.float32,
                                 **transform_kwargs)
                for label in labels
            )
            chunk = _score_to_labels(scores, labels, threshold, tuple(int(v) for v in chunk_shape))
        sampled[z0:z0 + chunk_shape[0]] = chunk
    if not any(value != 0 for value in seen_labels):
        raise ValueError("Source mask has no foreground in the sampled region")

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
    fov = (
        (np.abs(z_mm)[:, None, None] <= object_fov_yx[0] / 2)
        & ((y_mm[None, :, None] / radius_y) ** 2
           + (x_mm[None, None, :] / radius_x) ** 2 <= 1)
    )
    sampled *= fov  # multiplication preserves class IDs; bitwise AND does not

    width = settings.fov_output_pixels
    if settings.fov_source_pixels:
        result = sampled
    elif min(ny, nx) < width:
        def resize_plane(plane):
            resized = zoom(plane, (width / ny, width / nx),
                           order=order, mode="nearest", prefilter=(order == 3))
            if resized.shape != (width, width):
                resized = resized[:width, :width]
                pad_y, pad_x = width - resized.shape[0], width - resized.shape[1]
                resized = np.pad(resized, ((pad_y // 2, pad_y - pad_y // 2),
                                           (pad_x // 2, pad_x - pad_x // 2)))
            return resized

        result = np.zeros((nz, width, width), np.int16)
        for z in range(nz):
            if order == 0:
                result[z] = resize_plane(sampled[z])
            else:
                labels = [int(v) for v in np.unique(sampled[z]) if v != 0]
                scores = (resize_plane((sampled[z] == label).astype(np.float32))
                          for label in labels)
                result[z] = _score_to_labels(scores, labels, threshold, (width, width))
        output_spacing[1] *= ny / width
        output_spacing[2] *= nx / width
    else:
        y0, x0 = (ny - width) // 2, (nx - width) // 2
        result = sampled[:, y0:y0 + width, x0:x0 + width]
    result = np.ascontiguousarray(result)
    if result.shape[1:] != (width, width):
        raise RuntimeError("Mask output size differs from the GT grid")
    if not np.any(result):
        raise ValueError("Mask foreground is outside the processed image FOV")
    return result, tuple(float(v) for v in output_spacing)


def _check_partner(path: Path, mask: np.ndarray, spacing, label: str) -> None:
    if not path.is_file():
        return
    image = nib.load(str(path))
    if image.shape != mask.shape or not np.allclose(image.header.get_zooms()[:3], spacing, atol=1e-5):
        raise ValueError(f"{label} geometry differs from generated mask: {path}")
    expected_affine = np.diag([*spacing, 1.0])
    if not np.allclose(image.affine, expected_affine, atol=1e-5):
        raise ValueError(f"{label} affine differs from generated mask: {path}")


def process_series(
    series_dir: Path,
    ct_paths: dict[int, Path],
    output_dir: Path,
    settings: SeriesSettings,
    *,
    gt_dir: Path | None = None,
    recon_dir: Path | None = None,
    overwrite: bool = False,
    phase: int | None = None,
    resampling: str = "linear",
    threshold: float = 0.5,
) -> dict[str, Any]:
    masks = source_masks(series_dir, ct_paths, phase)
    if not masks:
        return {"status": "no_source_masks", "source_directory": str(series_dir), "phases": {}}
    planned = [output_dir / f"phase_{phase:02d}_mask{suffix}"
               for phase in masks for suffix in (".tif", ".nii.gz")]
    if not overwrite and any(path.exists() for path in planned):
        raise FileExistsError(f"Mask output exists in {output_dir}; choose a new directory or use --overwrite")
    records = {}
    for phase, mask_path in masks.items():
        ct = nib.load(str(ct_paths[phase]))
        original = nib.load(str(mask_path))
        if ct.shape != original.shape or not np.allclose(ct.affine, original.affine, atol=1e-5):
            raise ValueError(f"Phase {phase}: CT and mask source grids differ")
        source_spacing = tuple(float(v) for v in ct.header.get_zooms()[:3])
        mask, spacing = make_phase_mask(original.dataobj, source_spacing, settings,
                                        resampling=resampling, threshold=threshold)
        if gt_dir is not None:
            _check_partner(gt_dir / f"phase_{phase:02d}_gt.nii.gz", mask, spacing, "GT")
        if recon_dir is not None:
            _check_partner(recon_dir / f"phase_{phase:02d}_reconstruction.nii.gz", mask, spacing, "Reconstruction")
        labels, counts = np.unique(mask, return_counts=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_image(mask, output_dir, f"phase_{phase:02d}_mask", spacing)
        records[str(phase)] = {
            "source_ct": str(ct_paths[phase]), "source_mask": str(mask_path),
            "source_shape_zyx": list(ct.shape),
            "source_spacing_mm_zyx": list(source_spacing),
            "output_shape_zyx": list(mask.shape),
            "output_spacing_mm_zyx": list(spacing),
            "label_counts": {str(int(label)): int(count) for label, count in zip(labels, counts)},
            "resampling": resampling,
            "threshold": threshold if resampling != "nearest" else None,
        }
        print(f"  phase {phase:02d}: {mask.shape}, labels={records[str(phase)]['label_counts']}", flush=True)
        del ct, original, mask
        gc.collect()
    return {"status": "complete", "source_directory": str(series_dir), "phases": records}


def output_paths(output_dir: Path, phases: tuple[int, ...]) -> list[Path]:
    return [output_dir / f"phase_{phase:02d}_mask{suffix}"
            for phase in phases for suffix in (".tif", ".nii.gz")]


def signature(settings: SeriesSettings, masks: dict[int, Path], ct_paths: dict[int, Path],
              resampling: str = "linear", threshold: float = 0.5) -> dict[str, Any]:
    def identity(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {
        "settings": settings_signature(settings),
        "label_policy": "preserve_integer_source_classes",
        "mask_interpolation": resampling,
        "mask_threshold": threshold if resampling != "nearest" else None,
        "algorithm_version": 2,
        "mask_files": {str(phase): identity(path) for phase, path in masks.items()},
        "ct_files": {str(phase): identity(ct_paths[phase]) for phase in masks},
    }


def outputs_current(output_dir: Path, phases: tuple[int, ...], config: dict[str, Any]) -> bool:
    if not all(path.is_file() and path.stat().st_size > 0 for path in output_paths(output_dir, phases)):
        return False
    try:
        return json.loads((output_dir / "mask_config.json").read_text(encoding="utf-8")) == config
    except (OSError, ValueError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-root", type=Path, default=DEFAULT_ROUND_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--recon-root", type=Path, default=DEFAULT_RECON_ROOT)
    parser.add_argument("--settings-csv", type=Path, default=DEFAULT_SETTINGS_CSV)
    parser.add_argument("--series", nargs="+", metavar="NAME",
                        help="Pig ID or relative series path; may list several")
    parser.add_argument("--phase", type=int, help="Only this annotated source phase")
    parser.add_argument("--mask-resampling", choices=("nearest", "linear", "polynomial", "cubic"),
                        default="linear", help="Default: linear; polynomial/cubic is third-order B-spline interpolation")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="Foreground threshold for linear/polynomial one-hot scores (default: 0.5)")
    parser.add_argument("--list", action="store_true", help="List annotated phases without writing")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing mask outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.mask_threshold) or not 0 < args.mask_threshold <= 1:
        raise ValueError("--mask-threshold must be in (0, 1]")
    round_root = args.round_root.resolve()
    series = select_series(discover_series(round_root), round_root, args.series)
    if not series:
        raise FileNotFoundError(f"No contrast-enhanced CT series found under {round_root}")
    settings_by_series = load_series_settings(args.settings_csv.resolve())
    output_root = args.output_root.resolve()
    for directory, ct_paths in series:
        relative = directory.relative_to(round_root).as_posix()
        if relative not in settings_by_series:
            raise ValueError(f"Settings CSV is missing series: {relative}")
        print(f"{relative}: annotated phases {tuple(source_masks(directory, ct_paths, args.phase))}")
    if args.list:
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"round_root": str(round_root), "settings_csv": str(args.settings_csv.resolve()), "series": []}
    failures = 0
    for directory, ct_paths in series:
        relative = directory.relative_to(round_root)
        masks = source_masks(directory, ct_paths, args.phase)
        settings = settings_by_series[relative.as_posix()]
        destination = output_root / relative
        try:
            if not masks:
                record = {"status": "no_source_masks", "phases": {}}
            else:
                config = signature(settings, masks, ct_paths,
                                   args.mask_resampling, args.mask_threshold)
                if outputs_current(destination, tuple(masks), config) and not args.overwrite:
                    record = {"status": "skipped_existing", "phases": tuple(masks)}
                else:
                    record = process_series(
                        directory, ct_paths, destination, settings,
                        gt_dir=args.gt_root.resolve() / relative,
                        recon_dir=args.recon_root.resolve() / relative,
                        overwrite=args.overwrite,
                        phase=args.phase,
                        resampling=args.mask_resampling,
                        threshold=args.mask_threshold,
                    )
                    (destination / "mask_config.json").write_text(
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
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        gc.collect()
    print(f"Finished {len(series)} series; {failures} failed. Manifest: {output_root / 'manifest.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
