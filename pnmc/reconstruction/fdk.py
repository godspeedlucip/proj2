"""Full-support FDK inputs using the installed TIGRE standard approximation."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from pnmc.data.manifest import load_case_manifest, validate_case
from pnmc.data.schema import ArrayRef, CaseManifest
from pnmc.data.uniform_import import case_to_dict
from pnmc.data.volume_io import save_volume


def _pad_axis(values: np.ndarray, padding: int, axis: int) -> np.ndarray:
    """Match the original generator's left reflection and right continuation."""
    if padding == 0:
        return values
    if padding > values.shape[axis]:
        raise ValueError("FDK padding exceeds detector size")
    left = np.flip(np.take(values, np.arange(padding), axis=axis), axis=axis)
    right = np.take(values, np.arange(values.shape[axis]-padding,
                                      values.shape[axis]), axis=axis)
    return np.concatenate((left, values, right), axis=axis)


def run_fdk(projector, projections: torch.Tensor, view_indices: np.ndarray) -> torch.Tensor:
    indices = projector._views(view_indices)
    detector = tuple(projector.geometry["nDetector_vu"])
    array = projector._tensor_array(projections, (len(indices), *detector))
    pad_v, pad_u = projector.fdk_padding_vu
    array = _pad_axis(_pad_axis(array, pad_v, 1), pad_u, 2)
    tigre, geo, angles = projector._geometry_for_views(indices,
                                                        padding_vu=projector.fdk_padding_vu)
    # Installed TIGRE's FDK calls filtering(..., parker=False) and uses
    # 2*pi/n_views in filtering.py. Keep the complete subset in one call.
    result = tigre.algorithms.fdk(np.array(array, dtype=np.float32, order="C", copy=True),
                                  geo, angles,
                                  filter="ram_lak", dowang=False)
    result = np.asarray(result, dtype=np.float32)
    if tuple(result.shape) != projector.grid.shape_zyx or not np.isfinite(result).all():
        raise ValueError("FDK output shape or finite values invalid")
    return torch.from_numpy(result.copy()).to(projector.device)


def _ref_dict(value):
    if isinstance(value, ArrayRef):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _ref_dict(item) for key, item in value.items()}
    return value


def build_fdk_inputs(case: CaseManifest, projector, output: Path) -> CaseManifest:
    validate_case(case, scan_arrays=False)
    if case.reconstruction_grid.fingerprint() != projector.grid.fingerprint():
        raise ValueError("FDK projector grid differs from case reconstruction grid")
    if not np.allclose(projector.angles_rad, case.angles_rad, rtol=0, atol=1e-6):
        raise ValueError("FDK angles differ from case")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("FDK output directory already contains files")
    output.mkdir(parents=True, exist_ok=True)
    measurements = np.load(case.projections.path, mmap_mode="r", allow_pickle=False)
    all_views = np.arange(len(case.angles_rad), dtype=np.int64)
    prior = save_volume(output / "prior.npy",
                        projector.fdk(torch.from_numpy(np.asarray(measurements).copy()), all_views),
                        projector.grid)
    sparse = {}
    for phase in case.phase_ids:
        indices = np.asarray(case.view_indices_by_phase[phase], dtype=np.int64)
        values = torch.from_numpy(np.asarray(measurements[indices], dtype=np.float32).copy())
        sparse[phase] = save_volume(output / ("sparse_" + str(phase) + ".npy"),
                                    projector.fdk(values, indices), projector.grid)
    config = {"mode": "tigre_standard_approx", "filter": "ram_lak",
              "parker": False, "dowang": False,
              "angle_scale": "2*pi/n_views_per_full_FDK_call",
              "padding_vu": list(projector.fdk_padding_vu),
              "projection_sha256": case.projections.sha256,
              "grid_sha256": projector.grid.fingerprint(),
              "tigre_version": importlib.metadata.version("pytigre")}
    config["sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    derived = replace(case, arrays={**case.arrays, "prior": prior,
                                    "sparse_by_phase": sparse},
                      provenance={**case.provenance, "fdk": config})
    payload = case_to_dict(derived)
    payload["arrays"] = _ref_dict(derived.arrays)
    path = output / "case_fdk.json"
    temporary = output / "case_fdk.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return load_case_manifest(path, require_physics=True)
