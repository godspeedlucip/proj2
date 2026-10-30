"""Create dynamic prior and per-phase sparse FDK images for AHFP Round 1.

This is the batch version of the projection/reconstruction sections in
``4dlung_projection_demo.ipynb``. Each contrast-enhanced CT series supplies
11 phase volumes. The same 600-view cycle mapping is used for every series.
Edit ``data/round1_contrast_recon_settings.csv`` to tune each series. Positive
``shift_down_voxels`` moves the sampled FOV toward larger row indices in the
original CT slices; positive ``shift_right_voxels`` moves it toward larger
column indices. To move anatomy downward in the reconstructed image, move the
sampled FOV upward in the source CT. The shifts use source CT voxel counts
before reconstruction.

Run with ``--list`` to inspect inputs without TIGRE or GPU work. For example,
``--series Stent_2_7596 --list`` previews one pig, and
``--series Stent_2_7596`` reconstructs only that pig. Changing its CSV row
automatically invalidates prior outputs for that series.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from generate_pig_first_phase_prior import (
    calibrate_attenuation,
    load_attenuation_calibration,
    reconstruct_prior,
)
from moni.phase_mapping import build_projection_phase_mapping
from utils.tool import MedicalImageTool, TigreTool


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROUND_ROOT = REPO_ROOT / "data" / "pig" / "AHFP-Round-1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "round1_contrast_recon"
DEFAULT_SETTINGS_CSV = REPO_ROOT / "data" / "round1_contrast_recon_settings.csv"
PHASE_NAME = re.compile(r"IM(\d+)_ct\.nii\.gz", re.IGNORECASE)

# Geometry and acquisition settings from the notebook's dynamic prior section.
N_PROJECTIONS = 600
ANGLE_SPAN_DEG = 300.0
VOXEL_SPACING_MULTIPLIER = 1.5
N_DETECTOR = (474, 648)
D_DETECTOR_MM = (0.616, 0.616)
DSO_MM = 750.0
DSD_MM = 1250.0
MIN_PHASE = 1
MAX_PHASE = 11


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
                     "fov_mask_shrink_pixels"):
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
        required = {"series", *defaults}
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
                if row[key] is not None and row[key].strip() else default
                for key, default in defaults.items()
            }
            settings = SeriesSettings(**values)
            settings.validate()
            result[name] = settings
    return result


def select_series(
    series: list[tuple[Path, dict[int, Path]]], round_root: Path, selectors: list[str] | None
) -> list[tuple[Path, dict[int, Path]]]:
    """Filter by pig ID or relative series path, rejecting unknown names."""
    if not selectors:
        return series
    names = [part.strip().replace("\\", "/").strip("/")
             for value in selectors for part in value.split(",") if part.strip()]
    selected: set[Path] = set()
    for name in names:
        matches = [directory for directory, _ in series
                   if name in (directory.parent.name, directory.relative_to(round_root).as_posix())]
        if not matches:
            raise ValueError(f"Unknown series: {name}")
        selected.update(matches)
    return [(directory, paths) for directory, paths in series if directory in selected]


def discover_series(round_root: Path) -> list[tuple[Path, dict[int, Path]]]:
    """Find every contrast-enhanced CT series, excluding masks and controls."""
    if not round_root.is_dir():
        raise FileNotFoundError(f"Round 1 directory does not exist: {round_root}")
    grouped: dict[Path, dict[int, Path]] = {}
    for path in round_root.rglob("*_ct.nii.gz"):
        if not any("with_contrast" in part.lower() for part in path.relative_to(round_root).parts[:-1]):
            continue
        match = PHASE_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected CT phase filename: {path}")
        phase = int(match.group(1))
        phases = grouped.setdefault(path.parent, {})
        if phase in phases:
            raise ValueError(f"Duplicate phase {phase} in {path.parent}")
        phases[phase] = path
    return [(directory, dict(sorted(phases.items()))) for directory, phases in sorted(grouped.items())]


def phase_indices(
    cycle_csv: Path, labels: tuple[int, ...], *, n_projections: int = N_PROJECTIONS,
    angle_span_deg: float = ANGLE_SPAN_DEG, min_phase: int = MIN_PHASE,
    max_phase: int = MAX_PHASE,
) -> dict[int, np.ndarray]:
    """Map all view indices to the available phase labels using cycle.csv."""
    for required in (min_phase, max_phase):
        if required not in labels:
            raise ValueError(f"Missing phase {required}; available phases: {labels}")
    mapping = build_projection_phase_mapping(
        cycle_csv,
        n_phases=len(labels),
        min_phase=min_phase,
        max_phase=max_phase,
        phase_labels=labels,
        start_index=0,
        n_projections=n_projections,
        angle_span_deg=angle_span_deg,
    )
    if [row.projection_index for row in mapping] != list(range(n_projections)):
        raise ValueError("Cycle mapping must cover every projection exactly once")
    result = {
        phase: np.asarray(
            [row.projection_index for row in mapping if row.phase_label == phase],
            dtype=np.int64,
        )
        for phase in labels
    }
    empty = [phase for phase, indices in result.items() if len(indices) == 0]
    if empty:
        raise ValueError(f"No projections mapped to phases: {empty}")
    return result


def calibrated_volume(
    hu: np.ndarray, polynomial: np.poly1d, air_threshold: float, air_value: float
) -> np.ndarray:
    """Apply notebook attenuation calibration in slabs to limit peak RAM."""
    result = np.empty(hu.shape, dtype=np.float32)
    for start in range(0, hu.shape[0], 16):
        end = min(start + 16, hu.shape[0])
        result[start:end] = calibrate_attenuation(
            TigreTool.hu_to_mu(hu[start:end]), polynomial, air_threshold, air_value
        )
    return result


def output_paths(output_dir: Path, labels: tuple[int, ...]) -> list[Path]:
    stems = ["prior_image"] + [f"phase_{phase:02d}_reconstruction" for phase in labels]
    return [output_dir / f"{stem}{suffix}" for stem in stems for suffix in (".tif", ".nii.gz")]


def configuration_signature(
    settings: SeriesSettings, cycle_csv: Path, calibration_csv: Path
) -> dict[str, Any]:
    """Record the settings and source CSV contents that produced the images."""
    def source(path: Path) -> dict[str, str | None]:
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        }

    return {
        "settings": asdict(settings),
        "cycle_csv": source(cycle_csv),
        "calibration_csv": source(calibration_csv),
    }


def outputs_current(
    output_dir: Path, labels: tuple[int, ...], signature: dict[str, Any]
) -> bool:
    """Skip a series only when all images and their settings match."""
    paths = output_paths(output_dir, labels)
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return False
    sidecar = output_dir / "reconstruction_config.json"
    if not sidecar.is_file():
        return False
    try:
        return json.loads(sidecar.read_text(encoding="utf-8")) == signature
    except (OSError, ValueError):
        return False


def save_image(image: np.ndarray, output_dir: Path, stem: str, spacing: tuple[float, ...]) -> None:
    MedicalImageTool.save(image, output_dir / f"{stem}.tif")
    MedicalImageTool.save(image, output_dir / f"{stem}.nii.gz", voxel_spacing=spacing)


def process_series(
    series_dir: Path,
    phase_paths: dict[int, Path],
    output_dir: Path,
    indices_by_phase: dict[int, np.ndarray],
    polynomial: np.poly1d,
    air_threshold: float,
    air_value: float,
    *,
    settings: SeriesSettings | None = None,
) -> dict[str, Any]:
    """Project phase CTs once, then FDK the full prior and each sparse phase."""
    settings = SeriesSettings() if settings is None else settings
    settings.validate()
    labels = tuple(phase_paths)
    first_image = MedicalImageTool.read(phase_paths[labels[0]])
    volume_shape = tuple(first_image.data.shape)
    if len(volume_shape) != 3:
        raise ValueError(f"Expected 3-D CT volume: {phase_paths[labels[0]]}")
    spacing = tuple(float(v) for v in first_image.metadata["header"].get_zooms()[:3])
    del first_image

    tool = TigreTool()
    used_spacing = tuple(v * settings.voxel_spacing_multiplier for v in spacing)
    geometry = tool.set_geometry(
        volume_shape=volume_shape,
        voxel_spacing=used_spacing,
        n_detector=settings.n_detector,
        d_detector=settings.d_detector_mm,
        dso=settings.dso_mm,
        dsd=settings.dsd_mm,
    )
    # TIGRE offOrigin translates the input CT relative to the scanner. A
    # negative offset moves the sampled FOV down/right in source array axes.
    shift_down = settings.shift_down_voxels - settings.shift_up_voxels
    shift_right = settings.shift_right_voxels - settings.shift_left_voxels
    geometry.offOrigin = np.asarray(
        [0.0, -shift_down * used_spacing[1], -shift_right * used_spacing[2]],
        dtype=np.float32,
    )
    angles = tool.set_angles(0.0, settings.angle_span_deg, settings.n_projections)
    projections = np.empty((settings.n_projections, *settings.n_detector), dtype=np.float32)
    phase_view_counts: dict[str, int] = {}

    for phase, path in phase_paths.items():
        image = MedicalImageTool.read(path)
        hu = np.asarray(image.data, dtype=np.float32)
        image_spacing = tuple(float(v) for v in image.metadata["header"].get_zooms()[:3])
        if hu.shape != volume_shape or not np.allclose(image_spacing, spacing, rtol=0, atol=1e-5):
            raise ValueError(f"Phase shape or spacing differs from first phase: {path}")
        volume_mu = calibrated_volume(hu, polynomial, air_threshold, air_value)
        selected = indices_by_phase[phase]
        phase_projections = TigreTool.project_tigre(volume_mu, geometry, angles[selected])
        expected_shape = (len(selected), *settings.n_detector)
        if phase_projections.shape != expected_shape:
            raise RuntimeError(f"Projection shape {phase_projections.shape} != {expected_shape}: {path}")
        projections[selected] = phase_projections
        phase_view_counts[str(phase)] = len(selected)
        print(f"  phase {phase:02d}: projected {len(selected)} views", flush=True)
        del image, hu, volume_mu, phase_projections
        gc.collect()

    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_options = {
        "padding_row_pixels": settings.padding_row_pixels,
        "padding_col_pixels": settings.padding_col_pixels,
        "fov_output_transverse_pixels": settings.fov_output_pixels,
        "fov_mask_diameter_shrink_pixels": settings.fov_mask_shrink_pixels,
    }
    prior, output_spacing = reconstruct_prior(
        projections, geometry, angles, tool, **reconstruction_options
    )
    save_image(prior, output_dir, "prior_image", output_spacing)
    prior_shape = list(prior.shape)
    del prior
    gc.collect()

    phase_shapes: dict[str, list[int]] = {}
    for phase in labels:
        selected = indices_by_phase[phase]
        sparse, sparse_spacing = reconstruct_prior(
            projections[selected], geometry, angles[selected], tool,
            **reconstruction_options,
        )
        save_image(sparse, output_dir, f"phase_{phase:02d}_reconstruction", sparse_spacing)
        phase_shapes[str(phase)] = list(sparse.shape)
        print(f"  phase {phase:02d}: reconstructed {sparse.shape}", flush=True)
        del sparse
        gc.collect()

    return {
        "status": "complete",
        "source_directory": str(series_dir),
        "phase_files": {str(phase): str(path) for phase, path in phase_paths.items()},
        "phase_view_counts": phase_view_counts,
        "volume_shape_zyx": list(volume_shape),
        "original_voxel_spacing_mm_zyx": list(spacing),
        "used_voxel_spacing_mm_zyx": list(used_spacing),
        "fov_shift_voxels_down_right": [shift_down, shift_right],
        "projection_off_origin_mm_zyx": geometry.offOrigin.tolist(),
        "settings": asdict(settings),
        "reconstruction_voxel_spacing_mm_zyx": list(output_spacing),
        "prior_shape_zyx": prior_shape,
        "phase_shapes_zyx": phase_shapes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-root", type=Path, default=DEFAULT_ROUND_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cycle-csv", type=Path, default=REPO_ROOT / "data" / "cycle.csv")
    parser.add_argument("--calibration-csv", type=Path, default=REPO_ROOT.parent / "compare.csv")
    parser.add_argument("--settings-csv", type=Path, default=DEFAULT_SETTINGS_CSV)
    parser.add_argument(
        "--series", nargs="+", metavar="NAME",
        help="Only reconstruct these pig IDs or relative series paths (space or comma separated).",
    )
    parser.add_argument("--list", action="store_true", help="List input series and phase counts only.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute complete output series.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    round_root = args.round_root.resolve()
    series = select_series(discover_series(round_root), round_root, args.series)
    if not series:
        raise FileNotFoundError(f"No contrast-enhanced CT series found under {round_root}")
    settings_by_series = load_series_settings(args.settings_csv.resolve())
    missing_settings = [directory.relative_to(round_root).as_posix()
                        for directory, _ in series
                        if directory.relative_to(round_root).as_posix() not in settings_by_series]
    if missing_settings:
        raise ValueError(f"Settings CSV is missing series: {missing_settings}")
    for directory, paths in series:
        relative = directory.relative_to(round_root).as_posix()
        settings = settings_by_series[relative]
        print(f"{relative}: {len(paths)} phases {tuple(paths)}, "
              f"FOV down/right = {settings.shift_down_voxels - settings.shift_up_voxels}/"
              f"{settings.shift_right_voxels - settings.shift_left_voxels} voxels")
    if args.list:
        return 0

    calibration = load_attenuation_calibration(args.calibration_csv.resolve())
    polynomial, air_threshold, air_value, calibration_details = calibration
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {
        "round_root": str(round_root),
        "cycle_csv": str(args.cycle_csv.resolve()),
        "settings_csv": str(args.settings_csv.resolve()),
        "attenuation_calibration": calibration_details,
        "series": [],
    }
    failures = 0
    for directory, phase_paths in series:
        labels = tuple(phase_paths)
        destination = output_root / directory.relative_to(round_root)
        settings = settings_by_series[directory.relative_to(round_root).as_posix()]
        try:
            signature = configuration_signature(
                settings, args.cycle_csv.resolve(), args.calibration_csv.resolve()
            )
            indices = phase_indices(
                args.cycle_csv.resolve(), labels,
                n_projections=settings.n_projections,
                angle_span_deg=settings.angle_span_deg,
                min_phase=settings.min_phase,
                max_phase=settings.max_phase,
            )
            if not args.overwrite and outputs_current(destination, labels, signature):
                record: dict[str, Any] = {"status": "skipped_existing"}
            else:
                print(f"Processing {directory.relative_to(round_root)}", flush=True)
                record = process_series(
                    directory, phase_paths, destination, indices,
                    polynomial, air_threshold, air_value, settings=settings,
                )
                (destination / "reconstruction_config.json").write_text(
                    json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        except Exception as exc:
            failures += 1
            record = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            print(f"Failed {directory.relative_to(round_root)}: {exc}", flush=True)
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
