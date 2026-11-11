"""Shared data and coordinate contracts for PNMC."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


PhaseId = str


def _digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _required_fields(value: dict, names: set, label: str) -> None:
    missing = names.difference(value)
    if missing:
        raise ValueError("{} missing fields: {}".format(label, sorted(missing)))


@dataclass(frozen=True)
class GridSpec:
    shape_zyx: Tuple[int, int, int]
    spacing_xyz_mm: Tuple[float, float, float]
    origin_xyz_mm: Tuple[float, float, float]
    direction_xyz: Tuple[Tuple[float, float, float], ...]
    frame_id: str

    def __post_init__(self) -> None:
        shape = tuple(self.shape_zyx)
        spacing = tuple(float(x) for x in self.spacing_xyz_mm)
        origin = tuple(float(x) for x in self.origin_xyz_mm)
        direction = tuple(tuple(float(x) for x in row) for row in self.direction_xyz)
        if len(shape) != 3 or any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError("shape_zyx must contain three positive integers")
        if len(spacing) != 3 or any(not math.isfinite(x) or x <= 0 for x in spacing):
            raise ValueError("spacing_xyz_mm must contain three positive finite values")
        if len(origin) != 3 or any(not math.isfinite(x) for x in origin):
            raise ValueError("origin_xyz_mm must contain three finite values")
        if len(direction) != 3 or any(len(row) != 3 for row in direction):
            raise ValueError("direction_xyz must be 3x3")
        matrix = np.asarray(direction, dtype=np.float64)
        if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-8:
            raise ValueError("direction_xyz must be finite and invertible")
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id is required")
        object.__setattr__(self, "shape_zyx", shape)
        object.__setattr__(self, "spacing_xyz_mm", spacing)
        object.__setattr__(self, "origin_xyz_mm", origin)
        object.__setattr__(self, "direction_xyz", direction)

    @classmethod
    def from_dict(cls, value: dict) -> "GridSpec":
        _required_fields(value, {"shape_zyx", "spacing_xyz_mm", "origin_xyz_mm",
                                 "direction_xyz", "frame_id"}, "grid")
        return cls(**{key: value[key] for key in ("shape_zyx", "spacing_xyz_mm",
                   "origin_xyz_mm", "direction_xyz", "frame_id")})

    def to_dict(self) -> dict:
        return {"shape_zyx": list(self.shape_zyx),
                "spacing_xyz_mm": list(self.spacing_xyz_mm),
                "origin_xyz_mm": list(self.origin_xyz_mm),
                "direction_xyz": [list(row) for row in self.direction_xyz],
                "frame_id": self.frame_id}

    def fingerprint(self) -> str:
        return _digest_json(self.to_dict())


@dataclass(frozen=True)
class ArrayRef:
    path: Path
    key: Optional[str]
    axes: str
    unit: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.axes or not self.unit:
            raise ValueError("array axes and unit are required")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256.lower()):
            raise ValueError("array sha256 must be a 64-character hex digest")

    @classmethod
    def from_dict(cls, value: dict, base_dir: Path) -> "ArrayRef":
        _required_fields(value, {"path", "key", "axes", "unit", "sha256"}, "array")
        path = Path(value["path"])
        if not path.is_absolute():
            path = base_dir / path
        return cls(path.resolve(), value["key"], value["axes"], value["unit"],
                   value["sha256"])

    def to_dict(self) -> dict:
        return {"path": str(self.path), "key": self.key, "axes": self.axes,
                "unit": self.unit, "sha256": self.sha256}


@dataclass(frozen=True)
class NormalizationSpec:
    mode: str
    a: float
    b: float
    fit_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.mode not in {"affine_clipped", "legacy50"}:
            raise ValueError("unknown normalization mode: {}".format(self.mode))
        if not math.isfinite(self.a) or not math.isfinite(self.b):
            raise ValueError("normalization endpoints must be finite")
        if self.mode == "affine_clipped" and self.b <= self.a:
            raise ValueError("affine normalization requires b > a")

    @classmethod
    def from_dict(cls, value: dict) -> "NormalizationSpec":
        _required_fields(value, {"mode", "a", "b", "fit_manifest_sha256"},
                         "normalization")
        return cls(value["mode"], float(value["a"]), float(value["b"]),
                   value["fit_manifest_sha256"])


@dataclass(frozen=True)
class SliceRecord:
    subject_id: str
    case_id: str
    acquisition_id: str
    phase_id: PhaseId
    slice_index: int
    grid_id: str
    arrays: Dict[str, ArrayRef]


@dataclass(frozen=True)
class NetworkManifest:
    schema_version: str
    records: List[SliceRecord]
    split_subjects: Dict[str, List[str]]
    grids: Dict[str, GridSpec]
    provenance: dict
    manifest_path: Optional[Path] = None
    manifest_sha256: str = ""


VectorRef = Union[List[Any], ArrayRef]


@dataclass(frozen=True)
class CaseManifest:
    schema_version: str
    subject_id: str
    case_id: str
    acquisition_id: str
    phase_ids: List[PhaseId]
    projections: Optional[ArrayRef]
    angles_rad: VectorRef
    phase_id_per_view: VectorRef
    view_id: VectorRef
    geometry: dict
    projection_volume_grid: Optional[GridSpec]
    reconstruction_grid: Optional[GridSpec]
    arrays: dict
    normalization: Optional[NormalizationSpec]
    provenance: dict
    manifest_path: Optional[Path] = None
    manifest_sha256: str = ""

    @property
    def view_indices_by_phase(self) -> Dict[PhaseId, List[int]]:
        from .manifest import vector_values
        groups = {phase: [] for phase in self.phase_ids}
        for index, phase in enumerate(vector_values(self.phase_id_per_view)):
            groups[phase].append(index)
        return groups
