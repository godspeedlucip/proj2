"""Manifest loading and measurement validation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .schema import (ArrayRef, CaseManifest, GridSpec, NetworkManifest,
                     NormalizationSpec, SliceRecord, _required_fields)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("non-finite JSON value: " + token)))
    if not isinstance(value, dict):
        raise ValueError("manifest root must be an object")
    return value


def _vector(value, base_dir: Path):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return ArrayRef.from_dict(value, base_dir)
    raise ValueError("view vector must be a list or ArrayRef")


def _open_array(ref: ArrayRef):
    if ref.path.suffix.lower() == ".npy":
        if ref.key is not None:
            raise ValueError("NPY array key must be null")
        return np.load(ref.path, mmap_mode="r", allow_pickle=False)
    if ref.path.suffix.lower() == ".npz":
        if ref.key is None:
            raise ValueError("NPZ array key is required")
        with np.load(ref.path, allow_pickle=False) as archive:
            if ref.key not in archive:
                raise ValueError("NPZ key missing: " + ref.key)
            return archive[ref.key]
    raise ValueError("unsupported array extension: " + ref.path.suffix)


def vector_values(value):
    return _open_array(value).tolist() if isinstance(value, ArrayRef) else value


def _parse_nested_refs(value, base_dir: Path):
    if isinstance(value, dict) and {"path", "axes", "unit", "sha256"}.issubset(value):
        return ArrayRef.from_dict(value, base_dir)
    if isinstance(value, dict):
        return {key: _parse_nested_refs(item, base_dir) for key, item in value.items()}
    return value


def load_network_manifest(path: Path) -> NetworkManifest:
    path = Path(path).resolve()
    raw = _load_json(path)
    _required_fields(raw, {"schema_version", "records", "split_subjects", "grids",
                           "provenance"}, "network manifest")
    grids = {name: GridSpec.from_dict(spec) for name, spec in raw["grids"].items()}
    splits = raw["split_subjects"]
    if set(splits) != {"train", "valid", "test"}:
        raise ValueError("split_subjects must have train, valid, test")
    membership = {}
    for split, subjects in splits.items():
        if len(subjects) != len(set(subjects)):
            raise ValueError("duplicate subject within split " + split)
        for subject in subjects:
            if subject in membership:
                raise ValueError("subject {} appears in multiple splits".format(subject))
            membership[subject] = split
    records = []
    identities = set()
    for item in raw["records"]:
        _required_fields(item, {"subject_id", "case_id", "acquisition_id", "phase_id",
                                "slice_index", "grid_id", "arrays"}, "slice record")
        if item["subject_id"] not in membership:
            raise ValueError("record subject absent from split_subjects")
        if item["grid_id"] not in grids:
            raise ValueError("record grid_id absent from grids")
        identity = tuple(item[name] for name in ("subject_id", "case_id", "acquisition_id",
                                                   "phase_id", "slice_index"))
        if identity in identities:
            raise ValueError("duplicate slice identity: {}".format(identity))
        identities.add(identity)
        arrays = {name: ArrayRef.from_dict(ref, path.parent)
                  for name, ref in item["arrays"].items()}
        if "input" not in arrays or "prior" not in arrays:
            raise ValueError("slice record needs input and prior")
        records.append(SliceRecord(item["subject_id"], item["case_id"],
                                   item["acquisition_id"], str(item["phase_id"]),
                                   int(item["slice_index"]), item["grid_id"], arrays))
    return NetworkManifest(str(raw["schema_version"]), records, splits, grids,
                           raw["provenance"], path, hashlib.sha256(path.read_bytes()).hexdigest())


def load_case_manifest(path: Path, *, require_physics: bool) -> CaseManifest:
    path = Path(path).resolve()
    raw = _load_json(path)
    _required_fields(raw, {"schema_version", "subject_id", "case_id", "acquisition_id",
                           "phase_ids", "provenance"}, "case manifest")
    if require_physics:
        _required_fields(raw, {"projections", "angles_rad", "phase_id_per_view", "view_id",
                               "geometry", "projection_volume_grid", "reconstruction_grid",
                               "normalization"}, "physical case manifest")
    base = path.parent
    case = CaseManifest(
        schema_version=str(raw["schema_version"]), subject_id=raw["subject_id"],
        case_id=raw["case_id"], acquisition_id=raw["acquisition_id"],
        phase_ids=[str(x) for x in raw["phase_ids"]],
        projections=ArrayRef.from_dict(raw["projections"], base) if raw.get("projections") else None,
        angles_rad=_vector(raw.get("angles_rad", []), base),
        phase_id_per_view=_vector(raw.get("phase_id_per_view", []), base),
        view_id=_vector(raw.get("view_id", []), base),
        geometry=raw.get("geometry", {}),
        projection_volume_grid=GridSpec.from_dict(raw["projection_volume_grid"])
        if raw.get("projection_volume_grid") else None,
        reconstruction_grid=GridSpec.from_dict(raw["reconstruction_grid"])
        if raw.get("reconstruction_grid") else None,
        arrays=_parse_nested_refs(raw.get("arrays", {}), base),
        normalization=NormalizationSpec.from_dict(raw["normalization"])
        if raw.get("normalization") else None,
        provenance=raw["provenance"], manifest_path=path,
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    if require_physics:
        validate_case(case, scan_arrays=False)
    return case


def _numeric_vector(value, length: int, name: str, positive: bool = False) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError("{} must have {} values".format(name, length))
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError("{} must be finite numeric values".format(name))
        if positive and item <= 0:
            raise ValueError("{} must be positive".format(name))


def _validate_geometry(geometry: dict, detector_shape) -> None:
    _required_fields(geometry, {"DSO_mm", "DSD_mm", "nDetector_vu", "dDetector_vu_mm",
                                "offDetector_vu_mm", "rotDetector_xyz_rad", "COR_mm",
                                "offOrigin_zyx_mm"}, "geometry")
    _numeric_vector(geometry["nDetector_vu"], 2, "nDetector_vu", True)
    if any(type(x) is not int for x in geometry["nDetector_vu"]):
        raise ValueError("nDetector_vu must contain integers")
    for key, count in (("dDetector_vu_mm", 2), ("offDetector_vu_mm", 2),
                       ("rotDetector_xyz_rad", 3), ("offOrigin_zyx_mm", 3)):
        _numeric_vector(geometry[key], count, key, key == "dDetector_vu_mm")
    for key in ("DSO_mm", "DSD_mm", "COR_mm"):
        value = geometry[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("{} must be finite".format(key))
    if not 0 < geometry["DSO_mm"] < geometry["DSD_mm"]:
        raise ValueError("require 0 < DSO_mm < DSD_mm")
    if detector_shape is not None and tuple(geometry["nDetector_vu"]) != tuple(detector_shape):
        raise ValueError("detector geometry and projection shape disagree")


def _check_ref(ref: ArrayRef, *, scan: bool):
    if not ref.path.is_file():
        raise ValueError("array path does not exist: " + str(ref.path))
    if scan:
        digest = hashlib.sha256()
        with ref.path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != ref.sha256:
            raise ValueError("array sha256 mismatch: " + str(ref.path))
    try:
        array = _open_array(ref)
    except ValueError as exc:
        raise ValueError("object or pickle array rejected: {}".format(ref.path)) from exc
    if array.dtype.kind not in "fiub":
        raise ValueError("object or unsupported array dtype: " + str(array.dtype))
    if scan:
        flat = array.reshape(-1)
        for first in range(0, flat.size, 1_000_000):
            if not np.isfinite(flat[first:first + 1_000_000]).all():
                raise ValueError("non-finite values in " + str(ref.path))
    return array.shape


def validate_case(case: CaseManifest, *, scan_arrays: bool) -> dict:
    if not case.phase_ids or len(case.phase_ids) != len(set(case.phase_ids)):
        raise ValueError("phase_ids must be non-empty and unique")
    if case.projections is None or case.projection_volume_grid is None or case.reconstruction_grid is None:
        raise ValueError("physical case needs projections and both grids")
    # Raw simulated measurements precede training-set normalization fitting.
    # The pending state must be explicit; network consumers require a fitted spec.
    if (case.normalization is None and
            case.provenance.get("normalization_status") != "pending_train_fit"):
        raise ValueError("physical case needs fitted or explicitly pending normalization")
    if case.projections.axes != "view,v,u" or case.projections.unit != "line_integral":
        raise ValueError("projection axes/unit must be view,v,u and line_integral")
    angles = vector_values(case.angles_rad)
    phases = vector_values(case.phase_id_per_view)
    views = vector_values(case.view_id)
    if len(angles) != len(phases) or len(angles) != len(views) or not angles:
        raise ValueError("angles, phases and view_id lengths must agree and be non-empty")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in angles):
        raise ValueError("angles must be finite")
    if len(set(views)) != len(views):
        raise ValueError("duplicate view_id")
    unknown = set(phases).difference(case.phase_ids)
    if unknown:
        raise ValueError("unknown phase labels: {}".format(sorted(unknown)))
    empty = set(case.phase_ids).difference(phases)
    if empty:
        raise ValueError("empty phase: {}".format(sorted(empty)))
    for ref in (case.angles_rad, case.phase_id_per_view, case.view_id):
        if isinstance(ref, ArrayRef):
            _check_ref(ref, scan=scan_arrays)
    detector_shape = None
    if case.projections.path.suffix.lower() == ".npy" or scan_arrays:
        shape = _check_ref(case.projections, scan=scan_arrays)
        if len(shape) != 3 or shape[0] != len(angles):
            raise ValueError("projection shape must be [view,v,u] with all views")
        detector_shape = shape[1:]
    elif not case.projections.path.is_file():
        raise ValueError("projection path does not exist")
    _validate_geometry(case.geometry, detector_shape)
    return {"status": "passed", "case_id": case.case_id, "view_count": len(views),
            "phase_view_counts": {phase: len(indices) for phase, indices
                                  in case.view_indices_by_phase.items()},
            "arrays_scanned": scan_arrays}
