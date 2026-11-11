"""Import simulated measurements using the original generator's scanner frame."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np

from .grids import build_full_grid
from .schema import ArrayRef, CaseManifest, GridSpec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _source_record(metadata: dict, recon_dir: Path) -> dict:
    matches = [item for item in metadata.get("series", [])
               if Path(item.get("output_directory", "")).resolve() == recon_dir]
    if len(matches) != 1 or matches[0].get("status") != "complete":
        raise ValueError("series has no unique complete source record")
    record = matches[0]
    source = Path(record["source_directory"]).resolve()
    round_root = Path(metadata["round_root"]).resolve()
    if not _inside(source, round_root) or source.parent.parent != round_root:
        raise ValueError("source series conflicts with aggregate round_root")
    if tuple(source.parts[-2:]) != tuple(recon_dir.parts[-2:]):
        raise ValueError("source series and reconstruction path disagree")
    return record


def _source_grid(record: dict) -> GridSpec:
    try:
        shape = record["volume_shape_zyx"]
        spacing = record["used_voxel_spacing_mm_zyx"]
        offset = record["projection_off_origin_mm_zyx"]
    except KeyError as exc:
        raise ValueError("source projection volume grid metadata missing: " + str(exc)) from exc
    if len(shape) != 3 or len(spacing) != 3 or len(offset) != 3:
        raise ValueError("source grid shape/spacing/offOrigin must have three values")
    shape = tuple(int(x) for x in shape)
    spacing_xyz = tuple(float(x) for x in spacing[::-1])
    offset_xyz = np.asarray(offset[::-1], dtype=np.float64)
    origin = offset_xyz - (np.asarray(shape[::-1]) - 1) * np.asarray(spacing_xyz) / 2
    return GridSpec(shape, spacing_xyz, tuple(float(x) for x in origin),
                    ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                    "tigre_generator_array_zyx")


def _audit_headers(record: dict, grid: GridSpec) -> None:
    source = Path(record["source_directory"]).resolve()
    original_spacing = record.get("original_voxel_spacing_mm_zyx")
    for label, filename in record.get("phase_files", {}).items():
        path = Path(filename).resolve()
        if not _inside(path, source) or not path.is_file():
            raise ValueError("phase source file/series mismatch: " + str(label))
        image = nib.load(str(path))  # Header only: never load CT voxels here.
        if tuple(image.shape) != grid.shape_zyx:
            raise ValueError("source CT header shape disagrees with generator record")
        if original_spacing is not None and not np.allclose(
                image.header.get_zooms()[:3], original_spacing, rtol=0, atol=1e-5):
            raise ValueError("source CT header spacing disagrees with generator record")


def _mapping(mapping_csv: Path, count: int):
    with mapping_csv.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(not {"projection_index", "phase_label", "angle_deg"}.issubset(row)
                       for row in rows):
        raise ValueError("mapping CSV needs projection_index, phase_label and angle_deg")
    try:
        rows.sort(key=lambda row: int(row["projection_index"]))
        indices = [int(row["projection_index"]) for row in rows]
        angles = [math.radians(float(row["angle_deg"])) for row in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid projection index or angle") from exc
    if indices != list(range(count)) or any(not math.isfinite(angle) for angle in angles):
        raise ValueError("mapping must contain every projection index exactly once")
    phases = [str(row["phase_label"]) for row in rows]
    if any(not phase for phase in phases):
        raise ValueError("mapping has empty phase label")
    return phases, angles, indices


def import_uniform_case(recon_dir: Path, mapping_csv: Path, *, source_metadata: Path,
                        subject_id: str) -> CaseManifest:
    recon_dir = Path(recon_dir).resolve()
    mapping_csv = Path(mapping_csv).resolve()
    source_metadata = Path(source_metadata).resolve()
    if not subject_id.strip():
        raise ValueError("subject_id is required")
    if source_metadata.parent != recon_dir.parent.parent:
        raise ValueError("source metadata and reconstruction round conflict")
    metadata = _read_json(source_metadata)
    record = _source_record(metadata, recon_dir)
    config_path = recon_dir / "reconstruction_config.json"
    config = _read_json(config_path)
    if record.get("settings") != config.get("settings"):
        raise ValueError("series generator settings conflict with case config")
    settings = config["settings"]
    for name in ("cycle_csv", "calibration_csv"):
        signature = config.get(name)
        if not isinstance(signature, dict) or not Path(signature.get("path", "")).is_file():
            raise ValueError(name + " source signature missing")
        if _sha256(Path(signature["path"])) != signature.get("sha256"):
            raise ValueError(name + " source signature changed")
    grid = _source_grid(record)
    _audit_headers(record, grid)
    projection_path = recon_dir / "projections.npy"
    projections = np.load(projection_path, mmap_mode="r", allow_pickle=False)
    if projections.ndim != 3 or projections.dtype.kind != "f":
        raise ValueError("projections must be floating [view,v,u]")
    view_count, n_v, n_u = projections.shape
    if view_count != int(settings["n_projections"]):
        raise ValueError("generator settings and projection view count disagree")
    if (n_v, n_u) != (int(settings["n_detector_rows"]), int(settings["n_detector_cols"])):
        raise ValueError("generator detector and projection shape disagree")
    phases, angles, indices = _mapping(mapping_csv, view_count)
    counts = Counter(phases)
    if counts != {str(key): int(value) for key, value in record["phase_view_counts"].items()}:
        raise ValueError("mapping phase view counts disagree with source series")
    if set(counts) != set(record.get("phase_files", {})):
        raise ValueError("mapping phases disagree with source CT phases")
    # The generator's offOrigin is an array-axis (z,y,x) shift in mm.
    offset = [float(x) for x in record["projection_off_origin_mm_zyx"]]
    reconstruction_grid, _ = build_full_grid(grid)
    geometry = {
        "DSO_mm": float(settings["dso_mm"]), "DSD_mm": float(settings["dsd_mm"]),
        "nDetector_vu": [n_v, n_u],
        "dDetector_vu_mm": [float(settings["d_detector_row_mm"]),
                             float(settings["d_detector_col_mm"])],
        "offDetector_vu_mm": [0., 0.], "rotDetector_xyz_rad": [0., 0., 0.],
        "COR_mm": 0., "offOrigin_zyx_mm": offset,
    }
    provenance = {
        "source_kind": "simulated", "frame_convention": "tigre_generator_array_zyx",
        "source_metadata_sha256": _sha256(source_metadata),
        "case_config_sha256": _sha256(config_path),
        "mapping_csv_sha256": _sha256(mapping_csv),
        "cycle_csv_sha256": config["cycle_csv"]["sha256"],
        "calibration_csv_sha256": config["calibration_csv"]["sha256"],
        "source_directory": str(Path(record["source_directory"]).resolve()),
        "source_phase_files": record["phase_files"],
        "attenuation_calibration": metadata.get("attenuation_calibration"),
        "normalization_status": "pending_train_fit",
    }
    return CaseManifest("1", subject_id, recon_dir.parent.name + "/" + recon_dir.name,
                        recon_dir.name, sorted(counts, key=lambda item: int(item)
                                              if item.isdigit() else item),
                        ArrayRef(projection_path, None, "view,v,u", "line_integral",
                                 _sha256(projection_path)),
                        angles, phases, indices, geometry, grid, reconstruction_grid,
                        {}, None, provenance)


def case_to_dict(case: CaseManifest) -> dict:
    return {
        "schema_version": case.schema_version, "subject_id": case.subject_id,
        "case_id": case.case_id, "acquisition_id": case.acquisition_id,
        "phase_ids": case.phase_ids, "projections": case.projections.to_dict(),
        "angles_rad": case.angles_rad, "phase_id_per_view": case.phase_id_per_view,
        "view_id": case.view_id, "geometry": case.geometry,
        "projection_volume_grid": case.projection_volume_grid.to_dict(),
        "reconstruction_grid": case.reconstruction_grid.to_dict(),
        "arrays": case.arrays,
        "normalization": asdict(case.normalization) if case.normalization else None,
        "provenance": case.provenance,
    }
